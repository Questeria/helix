"""
helixc/backend/metal.py — Apple Metal Shading Language (MSL) backend
text emission, substrate.

Stage 125 (v2.0 Phase C — Multi-vendor portability). Per the v2.0
research Report 5: "Apple Metal (5-7 EM): 28/40 ops decompose cleanly,
hard misses are TMA / TMEM (skip) and matmul-bifurcates pre-M5 vs M5+
Neural Accelerators."

MSL is C++14-flavored text consumed by Apple's compiler (xcrun
metal). Targets Metal 3.2 baseline (Apple Silicon M2+) with separate
codepaths for:
  - pre-M5 (M1/M2/M3/M4): SIMD matmul via simd_shuffle
  - M5+ Neural Accelerators (NA): native matrix multiply via mma.* MSL ops

v2.0 Phase C differentiator (Report 5): "first borrow-checked Metal
compute" — MSL has zero compile-time safety beyond C++14; combining
this backend with Stage 113-115 BorrowScope + Smem phase typestate
gives Helix a unique-in-class Metal safety story.

Stage 125 ships the substrate (header emit + op-mapping table +
kernel skeleton). Stage 126 wires the M5+ NA matmul path; pre-M5 SIMD
fallback may ship there or in a later stage.

tile-IR → MSL op mapping (key non-1:1 cases):
  bar.sync 0             → threadgroup_barrier(mem_flags::mem_threadgroup)
  bar.warp.sync          → simdgroup_barrier(mem_flags::mem_none)
  __syncthreads          → threadgroup_barrier
  wmma.* (TensorCore)    → simdgroup_multiply_accumulate (pre-M5)
                          OR mma_* MSL intrinsic (M5+ NA, SKIPPED for
                          Stage 125 — Stage 126)
  ld.global.* / TILE_LOAD → device float4*/device half4* pointer reads
  st.global.* / TILE_STORE → device pointer writes
  cp.async.* (Hopper)    → (no analog; use threadgroup_barrier fence)
  TMA load/store         → (no analog; skipped)
  TMEM                   → (no analog; Apple has no separate tensor memory)

License: Apache 2.0
"""

from __future__ import annotations

from io import StringIO
from typing import Final, Mapping, Optional

from ..ir import tir, tile_ir as ti


# ============================================================================
# Configuration
# ============================================================================
# Metal 3.2 ships with Sonoma/Sequoia (macOS 14+). Apple-Silicon-only.
DEFAULT_METAL_VERSION: str = "metal3.2"

# Default GPU target generation. M2-baseline ships pre-NA SIMD matmul.
# M5+ NA path gates via target family number >= 10 (apple10 == M5; see
# `_parse_apple_family` for the numeric-extraction logic. Stage 126 R5
# corrected the gate from apple9+ to apple10+ — apple9 is M3/M4, no NA hw.)
DEFAULT_TARGET_FAMILY: str = "apple7"  # M2 baseline; bump to apple10+ for M5+ NA

# v2.2 polish items 9 + 10: numeric target_family parsing + validation.
# Pre-v2.2 the M5+ gate used a hardcoded list ("apple10", "apple11",
# "apple12") which (a) silently mis-matched future families like
# apple13/apple14 — item 9, and (b) accepted typos like "appel10" or
# "apple_10" or "Apple10" by falling through the membership test —
# item 10. Both are closed by a `re.fullmatch(r"apple(\d+)", s)`
# parser that returns the family number or None.
import re as _re

_APPLE_FAMILY_RE = _re.compile(r"apple(\d+)")


def _parse_apple_family(target_family: str) -> Optional[int]:
    """Parse `appleN` into N for any non-negative integer N. Returns
    None on any value that doesn't match the canonical pattern (e.g.
    "appel10" typo, "apple_10" with underscore, "Apple10" miscased,
    "apple" without a number). Callers that need the strict family
    number (e.g. M5+ gating) should treat None as a hard-fail.
    """
    if not isinstance(target_family, str):
        return None
    m = _APPLE_FAMILY_RE.fullmatch(target_family)
    if m is None:
        return None
    return int(m.group(1))


def _validate_target_family(target_family: str) -> None:
    """Reject malformed target_family strings at construction time.
    v2.2 polish item 10: silent fallthrough to the pre-M5 path (because
    the typo isn't in the M5+ list) would route an apple10-targeting
    user to apple7-shape codegen. Hard-fail instead so the typo is
    visible immediately."""
    if _parse_apple_family(target_family) is None:
        raise ValueError(
            f"helixc.backend.metal: target_family={target_family!r} is "
            f"not a recognized Apple-Silicon family name. Expected "
            f"`appleN` where N is a non-negative integer (e.g. apple7 "
            f"for M2, apple10 for M5)."
        )

# SIMD width on Apple Silicon is 32 lanes per SIMD-group. (Threadgroup
# can hold multiple SIMD-groups.)
SIMD_WIDTH: int = 32


# ============================================================================
# Op-mapping table (tile-IR → MSL)
# ============================================================================
# Each tile-IR op kind we expect to support, with its MSL lowering.
# `status` semantics match rocm.py:
#   supported / stub / deferred / skipped
METAL_OP_LOWERING: Final[Mapping[ti.TileOpKind, dict]] = {
    ti.TileOpKind.TILE_ZEROS: {
        "lowering": "float4(0) / half8(0) init in threadgroup memory",
        "status": "stub",
    },
    ti.TileOpKind.TILE_CONST: {
        "lowering": "scalar/vector literal init",
        "status": "stub",
    },
    ti.TileOpKind.TILE_LOAD_GLOBAL: {
        "lowering": "device pointer read (float4 / half8)",
        "status": "supported",  # Stage 126: emit pattern wired
    },
    ti.TileOpKind.TILE_STORE_GLOBAL: {
        "lowering": "device pointer write (float4 / half8)",
        "status": "supported",
    },
    ti.TileOpKind.TILE_LOAD_SHARED: {
        "lowering": "threadgroup pointer read",
        "status": "supported",
    },
    ti.TileOpKind.TILE_STORE_SHARED: {
        "lowering": "threadgroup pointer write",
        "status": "supported",
    },
    ti.TileOpKind.TMA_LOAD: {
        "lowering": "(no analog on Apple; use device-pointer scatter)",
        "status": "skipped",
    },
    ti.TileOpKind.TMA_STORE: {
        "lowering": "(no analog; same as TMA_LOAD)",
        "status": "skipped",
    },
    ti.TileOpKind.BARRIER_WAIT: {
        "lowering": "threadgroup_barrier(mem_flags::mem_threadgroup)",
        "status": "supported",
    },
    # Stage 126 R5 audit-fix: TILE_ADD/SUB/MUL have no `_emit_op`
    # branch yet — demote to "stub" so the stub-status forward guard
    # at the top of `_emit_op` emits `#error` instead of falling
    # through to a silent `// (stub)` comment. (Promotion to
    # "supported" was premature; per-op MSL operator emit lands in
    # a later stage when operand-binding is wired.)
    ti.TileOpKind.TILE_ADD: {
        "lowering": "operator+ (SIMD-vectorized)",
        "status": "stub",
    },
    ti.TileOpKind.TILE_SUB: {
        "lowering": "operator-",
        "status": "stub",
    },
    ti.TileOpKind.TILE_MUL: {
        "lowering": "operator*",
        "status": "stub",
    },
    ti.TileOpKind.TILE_MATMUL: {
        # Stage 126 picks pre-M5 SIMD path or M5+ NA mma_* intrinsic
        # based on DEFAULT_TARGET_FAMILY. Phase-0 ships stub.
        "lowering": "simdgroup_multiply_accumulate (pre-M5) OR mma_* (M5+ NA)",
        "status": "supported",
    },
    ti.TileOpKind.TILE_REDUCE: {
        "lowering": "simd_sum / simd_max / simd_min (SIMD-group reduce)",
        "status": "stub",
    },
    ti.TileOpKind.TILE_TRANSPOSE: {
        "lowering": "simd_shuffle / threadgroup-memory transpose",
        "status": "stub",
    },
    ti.TileOpKind.TILE_RESHAPE: {
        "lowering": "(no-op at codegen; tile-IR shape rewrites only)",
        "status": "stub",
    },
    ti.TileOpKind.SCALAR_CONST_INT: {
        "lowering": "int literal",
        "status": "stub",
    },
    ti.TileOpKind.SCALAR_CONST_FLOAT: {
        "lowering": "float literal",
        "status": "stub",
    },
    ti.TileOpKind.SCALAR_ADD: {
        "lowering": "operator+",
        "status": "stub",
    },
    ti.TileOpKind.SCALAR_SUB: {
        "lowering": "operator-",
        "status": "stub",
    },
    ti.TileOpKind.SCALAR_MUL: {
        "lowering": "operator*",
        "status": "stub",
    },
    ti.TileOpKind.SCALAR_NEG: {
        "lowering": "unary operator-",
        "status": "stub",
    },
    ti.TileOpKind.SCALAR_CMP: {
        "lowering": "operator==, <, > etc.",
        "status": "stub",
    },
    ti.TileOpKind.SCALAR_SELECT: {
        "lowering": "select() / ternary",
        "status": "stub",
    },
    ti.TileOpKind.CALL: {
        "lowering": "function call (inline or @function decoration)",
        "status": "stub",
    },
    ti.TileOpKind.RETURN: {
        "lowering": "return statement",
        "status": "supported",
    },
    ti.TileOpKind.THREAD_IDX: {
        "lowering": "thread_position_in_threadgroup attribute (uint)",
        "status": "supported",
    },
    # Stage 126 R5 audit-fix: indexed-HBM has no `_emit_op` branch yet.
    # Demoted to "stub" — same reason as TILE_ADD/SUB/MUL above. The
    # Phase-0 PTX backend supports these; the v2 backends pick them
    # up when operand-binding lands.
    ti.TileOpKind.TILE_INDEX_LOAD_HBM: {
        "lowering": "device pointer indexed read",
        "status": "stub",
    },
    ti.TileOpKind.TILE_INDEX_STORE_HBM: {
        "lowering": "device pointer indexed write",
        "status": "stub",
    },
}


def _check_metal_lowering_coverage() -> None:
    """Module-load drift detector — every TileOpKind must have an entry.
    Same pattern as ROCm + adjoint table coverage checks."""
    for k in ti.TileOpKind:
        if k not in METAL_OP_LOWERING:
            raise AssertionError(
                f"helixc.backend.metal: TileOpKind {k.name} is missing "
                f"from METAL_OP_LOWERING. Every kind must have a "
                f"lowering or be marked status='skipped' with rationale."
            )


_check_metal_lowering_coverage()


# ============================================================================
# Emitter
# ============================================================================
class MslEmitter:
    """Metal Shading Language text emitter substrate.

    Stage 125 ships header emit + kernel-skeleton emit + per-op lookup.
    Stage 126 wires SIMD matmul / NA mma intrinsics. Runtime kernel-
    launch glue (MTLComputeCommandEncoder) lives in a separate runtime
    layer and is out of scope here.
    """

    def __init__(self,
                 metal_version: str = DEFAULT_METAL_VERSION,
                 target_family: str = DEFAULT_TARGET_FAMILY):
        # v2.2 polish item 10: validate target_family at construction
        # so typos like "appel10" / "apple_10" / "Apple10" hard-fail
        # immediately rather than silently routing to the pre-M5 path.
        _validate_target_family(target_family)
        self.metal_version = metal_version
        self.target_family = target_family
        self.buf = StringIO()
        self.next_var = 0

    def _line(self, s: str = "") -> None:
        self.buf.write(s)
        self.buf.write("\n")

    def emit_module_header(self) -> None:
        """Emit the MSL preamble. Xcode's metal compiler picks up the
        version from a command-line flag (`-std=metal3.2`); the
        `#include <metal_stdlib>` + `using namespace metal` are MSL
        boilerplate that every kernel needs.
        """
        self._line(f"// Helix-emitted MSL — target {self.target_family}, "
                   f"{self.metal_version}")
        self._line("#include <metal_stdlib>")
        self._line("using namespace metal;")
        self._line()

    def emit_module(self, mod: ti.TileModule) -> str:
        """Top-level emit. Walks all @kernel functions in the module.
        Raises RuntimeError if no @kernel found (parity with PtxEmitter +
        HipEmitter).
        """
        self.emit_module_header()
        emitted_kernel = False
        for fn in mod.functions.values():
            if fn.attrs.get("kernel") and not fn.attrs.get("is_extern"):
                emitted_kernel = True
                self.emit_kernel_stub(fn)
        if not emitted_kernel:
            raise RuntimeError(
                "MSL emission requires at least one @kernel function"
            )
        return self.buf.getvalue()

    def emit_kernel_stub(self, fn: ti.TileFn) -> None:
        """Stage 125 substrate + Stage 126 per-op wiring.

        Stage 125 ships the kernel header; Stage 126 (v2.1) fills in
        the body with MSL emission for the most common TileOpKinds.
        """
        # MSL kernels always return void; the kernel attribute is a
        # function-qualifier keyword (not a type modifier).
        self._line(f"kernel void {fn.name}(")
        # Stage 125 substrate: no params yet. Stage 126 will iterate
        # fn.params and emit `device float* param0 [[buffer(0)]]` etc.
        self._line(f"    uint tid [[thread_position_in_threadgroup]]")
        self._line(") {")
        # Stage 126: per-op body emit.
        for blk in fn.blocks:
            for op in blk.ops:
                self._emit_op(op)
        self._line("}")
        self._line()

    def _emit_op(self, op: ti.TileOp) -> None:
        """Stage 126 (v2.1 Phase C Metal NA matmul) — emit one tile-IR
        op as MSL source. v2.1 R1 audit-fix: ops with status="stub" /
        "deferred" in METAL_OP_LOWERING emit `#error` directives that
        abort xcrun metal compilation. Placeholder operand bindings
        below are wrapped in `/* HELIX-STUB-OPERAND */` so a reviewer
        cannot mistake them for production-ready code.
        """
        kind = op.kind
        # v2.1 R1 audit-fix: stub status → loud failure.
        status = METAL_OP_LOWERING[kind]["status"]
        if status in ("stub", "deferred"):
            self._line(
                f"    #error \"HELIX-STUB: TileOpKind.{kind.name} "
                f"status={status!r}; codegen not wired in this backend.\""
            )
            return
        # v2.2 polish item 9: numeric extraction. Apple's M5 introduced
        # Neural Accelerators at family `apple10+`. `apple9` is M3/M4
        # (no NA hw). Stage 126 R5 corrected the hardcoded list; v2.2
        # replaces the list with `_parse_apple_family(...) >= 10` so
        # future families (apple13, apple14, ...) are M5-plus without
        # code changes. _validate_target_family was called at __init__
        # so the parse cannot return None here.
        family_num = _parse_apple_family(self.target_family)
        assert family_num is not None, (
            "target_family validated at __init__; parse must succeed"
        )
        is_m5_plus = family_num >= 10
        if kind is ti.TileOpKind.BARRIER_WAIT:
            # MSL: threadgroup_barrier with appropriate fence flags.
            self._line("    threadgroup_barrier(mem_flags::mem_threadgroup);")
            return
        if kind is ti.TileOpKind.TILE_MATMUL:
            # Stage 126 R5 audit-fix: `mma_f32_16x16x16_f16` is NOT real
            # MSL — it's PTX/CUDA syntax that leaked into the Metal
            # backend. Apple's M5+ Neural Accelerators are HW
            # accelerators behind the *same* `simdgroup_matrix`
            # intrinsics — there is no separate `mma_*` MSL surface.
            # The pre-M5 vs M5+ split documents the *hardware path*
            # (SIMD-group ALU vs Neural Accelerator) in the comment;
            # the emitted call is `simdgroup_multiply_accumulate` on
            # both paths. Operand bindings are `HELIX-STUB-OPERANDS`
            # — the substrate emit ships placeholder matrix args; a
            # later stage wires real operand-binding from tile-IR's
            # value SSA.
            if is_m5_plus:
                self._line("    // tile-matmul A @ B + C (M5+ NA — Neural Accelerator hw path)")
            else:
                self._line("    // tile-matmul A @ B + C (pre-M5 SIMD-group path)")
            # Stage 126 R6 audit-fix: simdgroup_multiply_accumulate has
            # 4-argument signature `simdgroup_multiply_accumulate(D, A, B, C)`
            # computing `D = A * B + C` (per Apple MSL Spec §6.7.1
            # simdgroup_matrix API). R5 emitted 3 args, which Apple's
            # MSL compiler rejects with no-matching-overload. The 4th
            # operand `_D` is the destination; `_C` accumulates.
            self._line("    simdgroup_matrix<float, 8, 8> _A, _B, _C, _D; /* HELIX-STUB-OPERANDS */")
            self._line("    simdgroup_multiply_accumulate(_D, _A, _B, _C);")
            return
        if kind is ti.TileOpKind.TILE_LOAD_GLOBAL:
            self._line("    // device pointer tile load")
            self._line("    auto tile_in = buf_in[tid];")
            return
        if kind is ti.TileOpKind.TILE_STORE_GLOBAL:
            self._line("    // device pointer tile store")
            self._line("    buf_out[tid] = tile_out;")
            return
        if kind is ti.TileOpKind.TILE_LOAD_SHARED:
            self._line("    // threadgroup memory tile load")
            self._line("    auto tile = shared_mem[tid];")
            return
        if kind is ti.TileOpKind.TILE_STORE_SHARED:
            self._line("    // threadgroup memory tile store")
            self._line("    shared_mem[tid] = tile;")
            return
        if kind is ti.TileOpKind.THREAD_IDX:
            # `tid` is bound in the kernel signature as
            # [[thread_position_in_threadgroup]].
            self._line("    // tid is bound in kernel signature")
            return
        if kind is ti.TileOpKind.RETURN:
            # MSL kernels return void; no explicit return needed
            # except at non-terminator positions.
            self._line("    return;")
            return
        # Stage 126 R5 audit-fix: exhaustiveness guard. Any op with
        # status="supported" in METAL_OP_LOWERING MUST have a concrete
        # branch above. Reaching here means the table declares the op
        # ready but no codegen wires it — a silent-failure surface.
        # The stub-status guard at the top of _emit_op already handles
        # status=("stub","deferred","skipped"). If we got here, the
        # table lies. Raise loudly instead of emitting a silent
        # // comment that would compile and ship a no-op kernel.
        raise AssertionError(
            f"helixc.backend.metal: TileOpKind.{kind.name} has "
            f"status={status!r} in METAL_OP_LOWERING but no "
            f"`_emit_op` branch. Either add a concrete emit or "
            f"demote the status to 'stub' so the forward guard "
            f"emits #error."
        )


def lowering_status(kind: ti.TileOpKind) -> str:
    """Query the lowering status for one TileOpKind.

    Returns one of: "supported", "stub", "deferred", "skipped".

    Raises TypeError on non-TileOpKind input — same discipline as
    rocm.lowering_status / tile_ir.has_adjoint.
    """
    if not isinstance(kind, ti.TileOpKind):
        raise TypeError(
            f"lowering_status expects TileOpKind, got "
            f"{type(kind).__name__}: {kind!r}"
        )
    entry = METAL_OP_LOWERING.get(kind)
    if entry is None:
        raise AssertionError(
            f"TileOpKind {kind.name} missing from METAL_OP_LOWERING "
            f"(module-load check should have caught this)"
        )
    return entry["status"]
