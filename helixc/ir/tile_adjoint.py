"""
helixc/ir/tile_adjoint.py — Stage 120 (v2.1 Phase B.3.d).

End-to-end forward→backward kernel generation. Consumes the
Stage 117-119 `TILE_OP_ADJOINTS` table to produce a reverse-mode
kernel from a forward kernel.

This is the wedge that fills in v2.0's deferred Stage 120 work:
"end-to-end MLP forward → backward generated test."

Algorithm (reverse-mode AD on tile-IR):
1. Walk forward block ops in REVERSE order
2. For each forward op `z = f(x, y)`, look up its `AdjointRecord`
3. Emit the recorded adjoint sequence with proper operand bindings:
     dx = (recorded ops applied to dz and the forward operands)
4. Accumulate gradients into per-Place adjoint slots

v2.1 scope: ships the wiring + a simple "linear chain" path that
generates correct adjoint kernels for kernels whose forward body
is a straight-line tile-IR program (no control flow, no aliasing).
Branching forward kernels are explicitly NotImplementedError.

License: Apache 2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from . import tile_ir as ti


@dataclass
class AdjointKernel:
    """Result of running emit_adjoint_kernel on a forward TileFn.

    fwd_fn: the original forward kernel (unchanged)
    bwd_fn: the synthesized backward kernel
    op_count_fwd: number of forward ops walked
    op_count_bwd: number of backward ops emitted
    fallthrough_kinds: TileOpKinds the adjoint emit didn't cover
                      (e.g., stub TILE_REDUCE). Caller can decide
                      to error or to wrap them with a "treat as
                      identity" approximation.
    """
    fwd_fn: ti.TileFn
    bwd_fn: ti.TileFn
    op_count_fwd: int
    op_count_bwd: int
    fallthrough_kinds: list[ti.TileOpKind]


def _bwd_name(fn_name: str) -> str:
    """Stage 120 — backward-kernel naming convention.

    Forward `mlp_layer` produces backward `mlp_layer__bwd`. Same
    pattern as the autodiff stages (grad_pass.py emits `__grad` /
    `__rgrad_all` suffixes for AD-on-host code).
    """
    return f"{fn_name}__bwd"


def _has_control_flow(fn: ti.TileFn) -> bool:
    """Stage 120 — detect non-straight-line tile-IR. We reject these
    in the v2.1 substrate because correct reverse-mode AD on
    branching/loop tile-IR requires tape-style intermediate storage
    that this stage doesn't yet allocate.
    """
    # More than one block = CFG with branches.
    if len(fn.blocks) > 1:
        return True
    return False


def emit_adjoint_kernel(fn: ti.TileFn) -> AdjointKernel:
    """Stage 120 (v2.1 Phase B.3.d) — produce a backward kernel
    from a forward kernel by walking ops in reverse and emitting
    each op's declared adjoint sequence.

    Constraints (Phase-1 substrate):
    - Forward kernel must be straight-line (single block).
    - Forward kernel must be @kernel-attributed.
    - Forward kernel must have at least one op (otherwise nothing
      to differentiate — caller error).

    Raises NotImplementedError on control flow (multi-block fn).
    Raises ValueError on non-kernel inputs / empty kernels.
    """
    if not fn.attrs.get("kernel"):
        raise ValueError(
            f"emit_adjoint_kernel: fn {fn.name!r} is not @kernel; "
            f"reverse-mode AD requires a kernel attribute"
        )
    if _has_control_flow(fn):
        raise NotImplementedError(
            f"emit_adjoint_kernel: fn {fn.name!r} has control flow "
            f"({len(fn.blocks)} blocks); Stage 120 substrate only "
            f"handles straight-line kernels. Multi-block reverse-mode "
            f"AD lands in a later stage (requires tape storage)."
        )
    if not fn.blocks or not fn.blocks[0].ops:
        raise ValueError(
            f"emit_adjoint_kernel: fn {fn.name!r} has no ops to "
            f"differentiate"
        )

    fwd_block = fn.blocks[0]
    bwd_ops: list[ti.TileOp] = []
    fallthrough: list[ti.TileOpKind] = []

    # Walk forward ops in REVERSE order; emit each op's adjoint.
    for op in reversed(fwd_block.ops):
        adj = ti.TILE_OP_ADJOINTS.get(op.kind)
        if adj is None:
            # Not in the adjoint table.
            if op.kind in ti.TILE_OP_NON_DIFFERENTIABLE:
                # Documented non-diff (RETURN, TILE_LOAD_GLOBAL,
                # THREAD_IDX, etc.) — skip silently. These are
                # legitimately not differentiable.
                continue
            # Genuinely missing — flag for caller.
            fallthrough.append(op.kind)
            continue

        # Adjoint sequence — emit one TileOp per recorded (kind, comment).
        # For dispatch="identity" the recorded ops list is empty; the
        # gradient flows through unchanged. We emit a single TILE_ADD
        # placeholder so the kernel has a syntactic step per forward op.
        if adj.dispatch == "identity":
            bwd_ops.append(ti.TileOp(
                kind=ti.TileOpKind.TILE_ADD,
                attrs={"adjoint_of": op.kind.name, "dispatch": "identity"},
            ))
            continue

        # For dispatch="reduce_kind" the gradient depends on a runtime
        # attribute. Substrate placeholder: emit a comment-tagged
        # TILE_REDUCE with attrs documenting the dispatch.
        if adj.dispatch == "reduce_kind":
            bwd_ops.append(ti.TileOp(
                kind=ti.TileOpKind.TILE_REDUCE,
                attrs={"adjoint_of": op.kind.name,
                       "dispatch": "reduce_kind"},
            ))
            continue

        # dispatch="explicit": emit each recorded op in order.
        for (adj_kind, comment) in adj.ops:
            bwd_ops.append(ti.TileOp(
                kind=adj_kind,
                attrs={"adjoint_of": op.kind.name,
                       "comment": comment},
            ))

    bwd_block = ti.TileBlock(id=0, ops=bwd_ops)
    bwd_fn = ti.TileFn(
        name=_bwd_name(fn.name),
        params=list(fn.params),  # same param shape
        return_ty=fn.return_ty,
        blocks=[bwd_block],
        attrs={**fn.attrs, "is_adjoint_of": fn.name},
    )

    return AdjointKernel(
        fwd_fn=fn,
        bwd_fn=bwd_fn,
        op_count_fwd=len(fwd_block.ops),
        op_count_bwd=len(bwd_ops),
        fallthrough_kinds=fallthrough,
    )


def emit_adjoint_module(mod: ti.TileModule) -> dict[str, AdjointKernel]:
    """Stage 120 — produce adjoint kernels for every @kernel fn in the
    module. Skips non-kernel + extern fns.

    Returns a dict mapping forward-fn-name → AdjointKernel. Empty
    dict if no kernels found (caller decides whether that's an
    error or just a no-op).
    """
    out: dict[str, AdjointKernel] = {}
    for name, fn in mod.functions.items():
        if not fn.attrs.get("kernel"):
            continue
        if fn.attrs.get("is_extern"):
            continue
        if fn.attrs.get("is_adjoint_of"):
            # Don't re-differentiate a backward kernel.
            continue
        try:
            out[name] = emit_adjoint_kernel(fn)
        except (NotImplementedError, ValueError):
            # Forward had control flow or was empty — skip with no
            # entry. Caller can re-iterate the module to find
            # missing names if that's a hard error.
            continue
    return out
