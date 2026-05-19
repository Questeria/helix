"""
helixc/ir/tile_adjoint.py — Stage 120 (v2.1 Phase B.3.d) + R1 audit-fix.

End-to-end forward→backward kernel-shell generation. Consumes the
Stage 117-119 `TILE_OP_ADJOINTS` table to produce a reverse-mode
kernel-shell from a forward kernel.

This is the wedge that fills in v2.0's deferred Stage 120 work:
"end-to-end MLP forward → backward generated test."

Substrate scope (R1 audit-fix honest disclosure):
- Walks forward block ops in REVERSE order.
- For each forward op `z = f(x, y)`, looks up its `AdjointRecord` and
  emits the recorded adjoint OP-KIND SHELLS with provenance attrs
  (`adjoint_of`, `comment`, `dispatch`, runtime-attr passthroughs).
- The emitted ops carry NO `operands` and NO `results` — this stage
  ships the dispatcher and op-kind sequencing only. Full SSA value
  wiring (binding gradient values to forward operands and accumulating
  into adjoint slots) is deferred to a later stage that will also
  introduce tape storage for control-flow kernels.
- Branching/looping forward kernels (multi-block) are explicitly
  NotImplementedError.

License: Apache 2.0
"""

from __future__ import annotations

from dataclasses import dataclass

from . import tile_ir as ti


@dataclass
class AdjointKernel:
    """Result of running emit_adjoint_kernel on a forward TileFn.

    fwd_fn: the original forward kernel (unchanged)
    bwd_fn: the synthesized backward kernel-shell
    op_count_fwd: number of forward ops walked
    op_count_bwd: number of backward op-shells emitted
    fallthrough_kinds: TileOpKinds neither in TILE_OP_ADJOINTS nor in
                      TILE_OP_NON_DIFFERENTIABLE — these are genuine
                      gaps in the canonical table. A non-empty list
                      means the backward kernel is INCOMPLETE; callers
                      should consult `complete` and decide whether to
                      reject or proceed with a partial gradient.
    """
    fwd_fn: ti.TileFn
    bwd_fn: ti.TileFn
    op_count_fwd: int
    op_count_bwd: int
    fallthrough_kinds: list[ti.TileOpKind]

    @property
    def complete(self) -> bool:
        """Stage 120 R1 audit-fix: structural completeness flag.

        True iff every forward op had a known adjoint declaration
        (either differentiable via TILE_OP_ADJOINTS or explicitly
        non-differentiable). False if any forward op-kind landed in
        `fallthrough_kinds` — the backward kernel is missing gradient
        contributions for those ops and must not be treated as a
        faithful adjoint.
        """
        return not self.fallthrough_kinds


@dataclass
class AdjointModule:
    """Result of running emit_adjoint_module on a TileModule — R1 audit-fix.

    Splits "successfully differentiated" from "skipped, with reason"
    so callers can distinguish:
      - non-kernel host fn (intentional)
      - extern fn (intentional)
      - already-an-adjoint fn (intentional, prevents double-diff)
      - empty kernel (caller bug, surfaced explicitly)
      - multi-block control flow (substrate limitation)

    Without this split the four+ skip reasons would be indistinguishable
    from "differentiated successfully but absent from output" — a real
    silent-failure trap.
    """
    kernels: dict[str, AdjointKernel]
    skipped: dict[str, str]  # fn_name -> reason

    @property
    def total_seen(self) -> int:
        return len(self.kernels) + len(self.skipped)


def _bwd_name(fn_name: str) -> str:
    """Stage 120 — backward-kernel naming convention.

    Forward `mlp_layer` produces backward `mlp_layer__bwd`. Same
    pattern as the autodiff stages (grad_pass.py emits `__grad` /
    `__rgrad_all` suffixes for AD-on-host code).
    """
    return f"{fn_name}__bwd"


def _has_control_flow(fn: ti.TileFn) -> bool:
    """Stage 120 — multi-block kernels indicate CFG branches.

    Correct reverse-mode AD on branching tile-IR requires tape-style
    intermediate storage that this stage doesn't yet allocate.
    """
    return len(fn.blocks) > 1


def emit_adjoint_kernel(fn: ti.TileFn) -> AdjointKernel:
    """Stage 120 (v2.1 Phase B.3.d) — produce a backward kernel-shell
    from a forward kernel by walking ops in reverse and emitting
    each op's declared adjoint sequence.

    Constraints (Phase-1 substrate):
    - Forward kernel must be straight-line (single block).
    - Forward kernel must be @kernel-attributed.
    - Forward kernel must have at least one op (otherwise nothing
      to differentiate — caller error).

    The emitted backward ops carry no operands/results — this is
    a kernel-shell synthesis stage, not a full SSA-wiring pass.

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

        # R1 audit-fix C2: dispatch="identity" means the gradient
        # flows through unchanged with no intermediate ops. The
        # canonical table encodes this as ops=(); the dispatcher
        # honors that and emits zero backward ops. A previous
        # version invented a fake TILE_ADD placeholder which was
        # an un-wired no-op pretending to be a backward step.
        if adj.dispatch == "identity":
            continue

        # R1 audit-fix C3: dispatch="reduce_kind" must propagate the
        # forward op's runtime-keyed attr (e.g. "sum" vs "max") into
        # the backward shell so the downstream lowering can pick
        # broadcast vs scatter. Previously the attr was dropped on
        # the floor, leaving the backward indistinguishable across
        # reduce kinds.
        if adj.dispatch == "reduce_kind":
            bwd_ops.append(ti.TileOp(
                kind=ti.TileOpKind.TILE_REDUCE,
                attrs={
                    "adjoint_of": op.kind.name,
                    "dispatch": "reduce_kind",
                    "reduce_kind": op.attrs.get("reduce_kind"),
                },
            ))
            continue

        # dispatch="explicit": emit each recorded op-shell in order.
        # AdjointRecord.__post_init__ guarantees ops is non-empty
        # when dispatch=="explicit" (validated at table construction).
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


def emit_adjoint_module(mod: ti.TileModule) -> AdjointModule:
    """Stage 120 — produce adjoint kernels for every @kernel fn in the
    module. R1 audit-fix: returns AdjointModule with `kernels` +
    `skipped` so non-kernel / extern / already-adjoint / control-flow /
    empty-body skips are visible to callers (no silent absence).
    """
    kernels: dict[str, AdjointKernel] = {}
    skipped: dict[str, str] = {}
    for name, fn in mod.functions.items():
        if not fn.attrs.get("kernel"):
            skipped[name] = "non-kernel"
            continue
        if fn.attrs.get("is_extern"):
            skipped[name] = "extern"
            continue
        if fn.attrs.get("is_adjoint_of"):
            skipped[name] = "already-adjoint"
            continue
        try:
            kernels[name] = emit_adjoint_kernel(fn)
        except NotImplementedError as e:
            skipped[name] = f"NotImplementedError: {e}"
        except ValueError as e:
            skipped[name] = f"ValueError: {e}"
    return AdjointModule(kernels=kernels, skipped=skipped)
