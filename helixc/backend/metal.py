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
# M5+ NA path gates via DEFAULT_TARGET_FAMILY == "apple9" or higher.
DEFAULT_TARGET_FAMILY: str = "apple7"  # M2 baseline; bump to apple9 for M5+ NA

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
        "status": "stub",
    },
    ti.TileOpKind.TILE_STORE_GLOBAL: {
        "lowering": "device pointer write (float4 / half8)",
        "status": "stub",
    },
    ti.TileOpKind.TILE_LOAD_SHARED: {
        "lowering": "threadgroup pointer read",
        "status": "stub",
    },
    ti.TileOpKind.TILE_STORE_SHARED: {
        "lowering": "threadgroup pointer write",
        "status": "stub",
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
        "status": "stub",
    },
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
        "status": "stub",
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
        "status": "stub",
    },
    ti.TileOpKind.THREAD_IDX: {
        "lowering": "thread_position_in_threadgroup attribute (uint)",
        "status": "stub",
    },
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
        """Stage 125 substrate: emit the kernel signature + empty body.
        MSL kernels use the `kernel void` attribute. Per-op codegen
        lands in Stage 126+.
        """
        # MSL kernels always return void; the kernel attribute is a
        # function-qualifier keyword (not a type modifier).
        self._line(f"kernel void {fn.name}(")
        # Stage 125 substrate: no params yet. Stage 126 will iterate
        # fn.params and emit `device float* param0 [[buffer(0)]]` etc.
        self._line(f"    uint tid [[thread_position_in_threadgroup]]")
        self._line(") {")
        # Stub body — Stage 126 replaces with per-op MSL.
        self._line("    // Stage 125 substrate: per-op codegen lands Stage 126+")
        self._line("}")
        self._line()


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
