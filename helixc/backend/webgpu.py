"""
helixc/backend/webgpu.py — WebGPU / WGSL backend (text emission, substrate).

Stage 127 (v2.0 Phase C — Multi-vendor portability). Per the v2.0
research Report 5: "WebGPU/WGSL (3-5 EM): 20/40 ops decompose cleanly,
hard misses are TMA / TMEM (skip) and TILE_MATMUL (no Tensor Cores —
hand-rolled tile loop, ~1 TFLOPS ceiling)."

WGSL (WebGPU Shading Language) is the browser-portable shader IR.
Lowering is text-only — no LLVM IR / MLIR detour. Targets WGSL spec
2024+ (compute shader profile).

v2.0 Phase C differentiator (Report 5): "first type-safe browser ML"
— WGSL has the weakest type system of all targets; every Helix safety
feature is a differentiator here.

Stage 127 ships the substrate (header emit + op-mapping table +
kernel skeleton). Stage 128 wires the hand-rolled tile-loop matmul
(no Tensor Cores; ~1 TFLOPS ceiling on M2 vs ~80 TFLOPS for native MSL).

tile-IR → WGSL op mapping (key non-1:1 cases):
  bar.sync 0          → workgroupBarrier()
  bar.warp.sync       → subgroupBarrier() (WGSL 2024+ subgroup ext)
  wmma.* (TensorCore) → (no analog; hand-rolled tile loop)
  ld.global / TILE_LOAD → storage buffer pointer reads
  st.global / TILE_STORE → storage buffer pointer writes
  cp.async.* (Hopper) → (no analog)
  TMA load/store      → (no analog; skipped)
  TMEM                → (no analog; skipped)

License: Apache 2.0
"""

from __future__ import annotations

from io import StringIO
from typing import Final, Mapping, Optional

from ..ir import tir, tile_ir as ti


# ============================================================================
# Configuration
# ============================================================================
DEFAULT_WGSL_VERSION: str = "wgsl-2024"

# WGSL compute-shader workgroup_size — must be set on each @compute
# kernel. Default 64 = SIMD-friendly across NVIDIA / AMD / Apple.
DEFAULT_WORKGROUP_SIZE: int = 64


# ============================================================================
# Op-mapping table (tile-IR → WGSL)
# ============================================================================
WEBGPU_OP_LOWERING: Final[Mapping[ti.TileOpKind, dict]] = {
    ti.TileOpKind.TILE_ZEROS: {
        "lowering": "var<workgroup>/private array<f32, N> = array(0.0, ...)",
        "status": "stub",
    },
    ti.TileOpKind.TILE_CONST: {
        "lowering": "literal expression",
        "status": "stub",
    },
    ti.TileOpKind.TILE_LOAD_GLOBAL: {
        "lowering": "var<storage, read> buf : array<f32>; let v = buf[i]",
        "status": "stub",
    },
    ti.TileOpKind.TILE_STORE_GLOBAL: {
        "lowering": "var<storage, read_write> buf; buf[i] = v",
        "status": "stub",
    },
    ti.TileOpKind.TILE_LOAD_SHARED: {
        "lowering": "var<workgroup> shared : array<f32, N>; let v = shared[i]",
        "status": "stub",
    },
    ti.TileOpKind.TILE_STORE_SHARED: {
        "lowering": "var<workgroup>; shared[i] = v",
        "status": "stub",
    },
    ti.TileOpKind.TMA_LOAD: {
        "lowering": "(no analog; WebGPU has no async DMA primitive)",
        "status": "skipped",
    },
    ti.TileOpKind.TMA_STORE: {
        "lowering": "(no analog; same as TMA_LOAD)",
        "status": "skipped",
    },
    ti.TileOpKind.BARRIER_WAIT: {
        "lowering": "workgroupBarrier()",
        "status": "stub",
    },
    ti.TileOpKind.TILE_ADD: {
        "lowering": "operator+",
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
        # No Tensor Cores in WGSL. Stage 128 will emit a hand-rolled
        # tile-loop matmul; ~1 TFLOPS ceiling vs ~80 TFLOPS native.
        "lowering": "hand-rolled tile-loop matmul (Stage 128; ~1 TFLOPS)",
        "status": "stub",
    },
    ti.TileOpKind.TILE_REDUCE: {
        "lowering": "workgroup-tree reduction (manual; no subgroupAdd in baseline WGSL)",
        "status": "stub",
    },
    ti.TileOpKind.TILE_TRANSPOSE: {
        "lowering": "workgroup-memory transpose (manual scatter/gather)",
        "status": "stub",
    },
    ti.TileOpKind.TILE_RESHAPE: {
        "lowering": "(no-op at codegen; tile-IR shape rewrites only)",
        "status": "stub",
    },
    ti.TileOpKind.SCALAR_CONST_INT: {
        "lowering": "i32 literal",
        "status": "stub",
    },
    ti.TileOpKind.SCALAR_CONST_FLOAT: {
        "lowering": "f32 literal",
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
        "lowering": "select() builtin",
        "status": "stub",
    },
    ti.TileOpKind.CALL: {
        "lowering": "fn call (inline; WGSL has no recursion)",
        "status": "stub",
    },
    ti.TileOpKind.RETURN: {
        "lowering": "return statement",
        "status": "stub",
    },
    ti.TileOpKind.THREAD_IDX: {
        "lowering": "@builtin(local_invocation_id) local_id : vec3<u32>",
        "status": "stub",
    },
    ti.TileOpKind.TILE_INDEX_LOAD_HBM: {
        "lowering": "storage buffer indexed read",
        "status": "stub",
    },
    ti.TileOpKind.TILE_INDEX_STORE_HBM: {
        "lowering": "storage buffer indexed write",
        "status": "stub",
    },
}


def _check_webgpu_lowering_coverage() -> None:
    """Module-load drift detector — every TileOpKind must have an entry.
    Same pattern as rocm + metal + adjoint coverage checks."""
    for k in ti.TileOpKind:
        if k not in WEBGPU_OP_LOWERING:
            raise AssertionError(
                f"helixc.backend.webgpu: TileOpKind {k.name} is missing "
                f"from WEBGPU_OP_LOWERING. Every kind must have a "
                f"lowering or be marked status='skipped' with rationale."
            )


_check_webgpu_lowering_coverage()


# ============================================================================
# Emitter
# ============================================================================
class WgslEmitter:
    """WebGPU / WGSL text emitter substrate.

    Stage 127 ships the header emit + kernel-skeleton + per-op lookup.
    Stage 128 wires hand-rolled tile-loop matmul. Runtime kernel-launch
    via WebGPU JS API (createComputePipeline + dispatch) lives outside
    this backend.
    """

    def __init__(self,
                 wgsl_version: str = DEFAULT_WGSL_VERSION,
                 workgroup_size: int = DEFAULT_WORKGROUP_SIZE):
        self.wgsl_version = wgsl_version
        self.workgroup_size = workgroup_size
        self.buf = StringIO()

    def _line(self, s: str = "") -> None:
        self.buf.write(s)
        self.buf.write("\n")

    def emit_module_header(self) -> None:
        """Emit a WGSL-2024 module preamble. WGSL does not have a
        formal #version directive; comments document the spec level
        Helix expects (storage buffers + workgroup memory baseline)."""
        self._line(f"// Helix-emitted WGSL — spec {self.wgsl_version}")
        self._line(f"// Workgroup size default: {self.workgroup_size}")
        self._line()

    def emit_module(self, mod: ti.TileModule) -> str:
        """Top-level emit. Walks all @kernel functions in the module.
        Raises RuntimeError if no @kernel found (parity with the other
        backends).
        """
        self.emit_module_header()
        emitted_kernel = False
        for fn in mod.functions.values():
            if fn.attrs.get("kernel") and not fn.attrs.get("is_extern"):
                emitted_kernel = True
                self.emit_kernel_stub(fn)
        if not emitted_kernel:
            raise RuntimeError(
                "WGSL emission requires at least one @kernel function"
            )
        return self.buf.getvalue()

    def emit_kernel_stub(self, fn: ti.TileFn) -> None:
        """Stage 127 substrate: emit the kernel signature + empty body.
        WGSL compute kernels use @compute @workgroup_size(N) attribute
        + fn KERNEL_NAME() block. Per-op codegen lands Stage 128+.
        """
        self._line(f"@compute @workgroup_size({self.workgroup_size})")
        self._line(f"fn {fn.name}(")
        self._line("    @builtin(local_invocation_id) local_id: vec3<u32>")
        self._line(") {")
        # Stub body — Stage 128 replaces with per-op WGSL.
        self._line("    // Stage 127 substrate: per-op codegen lands Stage 128+")
        self._line("}")
        self._line()


def lowering_status(kind: ti.TileOpKind) -> str:
    """Query the lowering status for one TileOpKind.

    Returns one of: "supported", "stub", "deferred", "skipped".

    Raises TypeError on non-TileOpKind input — same discipline as
    rocm.lowering_status / metal.lowering_status.
    """
    if not isinstance(kind, ti.TileOpKind):
        raise TypeError(
            f"lowering_status expects TileOpKind, got "
            f"{type(kind).__name__}: {kind!r}"
        )
    entry = WEBGPU_OP_LOWERING.get(kind)
    if entry is None:
        raise AssertionError(
            f"TileOpKind {kind.name} missing from WEBGPU_OP_LOWERING "
            f"(module-load check should have caught this)"
        )
    return entry["status"]
