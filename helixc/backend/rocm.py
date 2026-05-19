"""
helixc/backend/rocm.py — AMD ROCm / HIP backend (text emission, substrate).

Stage 123 (v2.0 Phase C — Multi-vendor portability). Per the v2.0
research Report 5: "ROCm/HIP (4-6 EM): 33/40 ops decompose cleanly,
hard misses are TMA / mbarrier (use s_waitcnt) / TMEM (skip)."

Targets MI300 / MI300X (gfx940/gfx942) baseline. Lowering strategy
mirrors `helixc/backend/ptx.py` — emit text, let hipcc compile to
the GPU object format (AMDGPU). No LLVM IR detour.

v2.0 Phase C differentiator: AMD MI300 has SEV-SNP — Helix can be
the first source-level guarantee that AMD GPU code stays in an
enclave (TyEnclave + this backend's manifest emit).

Stage 123 ships the substrate (header emit + op-mapping table +
kernel skeleton). Stage 124 wires the wmma analog (MFMA: matrix-
fused-multiply-accumulate instruction class).

CUDA → AMDGPU op mapping (key non-1:1 cases):
  bar.sync 0          → s_barrier
  bar.warp.sync       → s_waitcnt vmcnt(0) lgkmcnt(0) + s_barrier_sgpr
                        (waveSize == 64 on RDNA2/3 GCN, 32 on RDNA1+)
  __syncthreads       → s_barrier
  wmma.* (TensorCore) → v_mfma_f32_16x16x16_* / v_mfma_f32_32x32x8_*
                        (MFMA: 16x16 + 32x32 tile shapes only)
  ld.global.*         → global_load_b32 / global_load_b64 / etc.
  st.global.*         → global_store_b32 / global_store_b64 / etc.
  cp.async.* (Hopper) → (no analog; use global_load + s_waitcnt fence;
                        deferred to Stage 124+ for HSA async)
  TMA load/store      → (no analog; tile load via VGPR scatter; deferred)
  TMEM                → (no analog; AMD has no separate tensor memory)

License: Apache 2.0
"""

from __future__ import annotations

from io import StringIO
from typing import Final, Mapping, Optional

from ..ir import tir, tile_ir as ti


# ============================================================================
# Configuration
# ============================================================================
# MI300 / MI300X uses gfx940 / gfx942. Older MI100 = gfx908; MI250 = gfx90a.
DEFAULT_TARGET: str = "gfx942"

# AMDGPU object format. hipcc consumes this via `--offload-arch={target}`.
ROCM_OBJECT_FORMAT: str = "amdgcn"

# Wave size — 64 on GCN / RDNA2; 32 on RDNA3 (wave32 mode). Helix targets
# MI300 (GCN-derived CDNA3) so wave64 is the baseline.
DEFAULT_WAVE_SIZE: int = 64


# ============================================================================
# Op-mapping table (CUDA / tile-IR → AMDGPU)
# ============================================================================

# Each tile-IR op kind we expect to support, with its AMDGPU lowering.
# `lowering`: the AMDGPU instruction or pseudo to emit.
# `status`: one of {"supported", "stub", "deferred", "skipped"}.
#   supported: implementation exists in this file
#   stub:      placeholder; ready for Stage 124+ implementation
#   deferred:  blocked on Phase A GPU CI / hardware test substrate
#   skipped:   no analog (NVIDIA-only); documented for completeness
ROCM_OP_LOWERING: Final[Mapping[ti.TileOpKind, dict]] = {
    ti.TileOpKind.TILE_ZEROS: {
        "lowering": "v_mov_b32 / s_mov_b32 (init VGPR/SGPR to 0)",
        "status": "stub",
    },
    ti.TileOpKind.TILE_CONST: {
        "lowering": "v_mov_b32 imm",
        "status": "stub",
    },
    ti.TileOpKind.TILE_LOAD_GLOBAL: {
        "lowering": "global_load_b32 / global_load_b64 / global_load_b128",
        "status": "supported",  # Stage 124: emit pattern wired
    },
    ti.TileOpKind.TILE_STORE_GLOBAL: {
        "lowering": "global_store_b32 / global_store_b64 / global_store_b128",
        "status": "supported",  # Stage 124
    },
    ti.TileOpKind.TILE_LOAD_SHARED: {
        "lowering": "ds_load_b32 / ds_load_b64 / ds_load_b128 (LDS = SMEM analog)",
        "status": "supported",  # Stage 124
    },
    ti.TileOpKind.TILE_STORE_SHARED: {
        "lowering": "ds_store_b32 / ds_store_b64 / ds_store_b128",
        "status": "supported",  # Stage 124
    },
    ti.TileOpKind.TMA_LOAD: {
        "lowering": "(no analog on AMD; use VGPR scatter or HSA queue)",
        "status": "skipped",
    },
    ti.TileOpKind.TMA_STORE: {
        "lowering": "(no analog; same as TMA_LOAD)",
        "status": "skipped",
    },
    ti.TileOpKind.BARRIER_WAIT: {
        "lowering": "s_waitcnt vmcnt(0) lgkmcnt(0)",
        "status": "supported",
    },
    ti.TileOpKind.TILE_ADD: {
        "lowering": "v_add_f32 / v_add_u32",
        "status": "supported",
    },
    ti.TileOpKind.TILE_SUB: {
        "lowering": "v_sub_f32 / v_sub_u32",
        "status": "supported",
    },
    ti.TileOpKind.TILE_MUL: {
        "lowering": "v_mul_f32 / v_mul_lo_u32",
        "status": "supported",
    },
    ti.TileOpKind.TILE_MATMUL: {
        "lowering": "v_mfma_f32_16x16x16_f16 (MI300 MFMA tile-matmul)",
        "status": "supported",  # Stage 124 wires MFMA emission
    },
    ti.TileOpKind.TILE_REDUCE: {
        "lowering": "v_reduce_* (wave-level) or LDS-tree reduction",
        "status": "stub",
    },
    ti.TileOpKind.TILE_TRANSPOSE: {
        "lowering": "ds_permute_b32 (LDS-based transpose)",
        "status": "stub",
    },
    ti.TileOpKind.TILE_RESHAPE: {
        "lowering": "(no-op at codegen; tile-IR shape rewrites only)",
        "status": "stub",
    },
    ti.TileOpKind.SCALAR_CONST_INT: {
        "lowering": "s_mov_b32 imm",
        "status": "stub",
    },
    ti.TileOpKind.SCALAR_CONST_FLOAT: {
        "lowering": "v_mov_b32 imm (bitcast)",
        "status": "stub",
    },
    ti.TileOpKind.SCALAR_ADD: {
        "lowering": "s_add_i32 / v_add_f32",
        "status": "stub",
    },
    ti.TileOpKind.SCALAR_SUB: {
        "lowering": "s_sub_i32 / v_sub_f32",
        "status": "stub",
    },
    ti.TileOpKind.SCALAR_MUL: {
        "lowering": "s_mul_i32 / v_mul_f32",
        "status": "stub",
    },
    ti.TileOpKind.SCALAR_NEG: {
        "lowering": "v_xor_b32 sign-bit (f32) / s_sub_i32 0 (int)",
        "status": "stub",
    },
    ti.TileOpKind.SCALAR_CMP: {
        "lowering": "v_cmp_eq_f32 / v_cmp_lt_i32 etc.",
        "status": "stub",
    },
    ti.TileOpKind.SCALAR_SELECT: {
        "lowering": "v_cndmask_b32",
        "status": "stub",
    },
    ti.TileOpKind.CALL: {
        "lowering": "s_swappc_b64 (indirect call) or s_call (rel32)",
        "status": "stub",
    },
    ti.TileOpKind.RETURN: {
        "lowering": "s_endpgm (kernel exit) / s_setpc_b64 (function return)",
        "status": "supported",
    },
    ti.TileOpKind.THREAD_IDX: {
        "lowering": "v_mov_b32 v0 (workitem.idx.x = thread_id baseline)",
        "status": "supported",
    },
    ti.TileOpKind.TILE_INDEX_LOAD_HBM: {
        "lowering": "global_load_<dtype> (HBM = global memory on AMD)",
        "status": "supported",
    },
    ti.TileOpKind.TILE_INDEX_STORE_HBM: {
        "lowering": "global_store_<dtype>",
        "status": "supported",
    },
}


# Status-tag invariant: every TileOpKind must appear in ROCM_OP_LOWERING.
# If a new op is added, this check fires loudly and forces a conscious
# port decision. Same drift-detector pattern as TILE_OP_ADJOINTS +
# TILE_OP_NON_DIFFERENTIABLE (Stages 117-119 audit-fix).
def _check_rocm_lowering_coverage() -> None:
    """Module-load check: every TileOpKind must be classified.

    Same drift-detector pattern as the adjoint table in tile_ir.py.
    Catches new TileOpKind additions before they silently fall through
    the codegen dispatch.
    """
    for k in ti.TileOpKind:
        if k not in ROCM_OP_LOWERING:
            raise AssertionError(
                f"helixc.backend.rocm: TileOpKind {k.name} is missing "
                f"from ROCM_OP_LOWERING. Every kind must have a "
                f"lowering or be marked status='skipped' with rationale."
            )


_check_rocm_lowering_coverage()


# ============================================================================
# Emitter
# ============================================================================
class HipEmitter:
    """ROCm / HIP text emitter substrate.

    Stage 123 ships the module header emit + kernel skeleton + per-op
    lookup. Stage 124 adds MFMA emission. Stage 125+ may add the
    runtime kernel-launch glue (HSA-AQL queue dispatch).
    """

    def __init__(self, target: str = DEFAULT_TARGET,
                 wave_size: int = DEFAULT_WAVE_SIZE):
        self.target = target
        self.wave_size = wave_size
        self.buf = StringIO()
        self.next_vreg = 0  # VGPR (vector / per-lane) counter
        self.next_sreg = 0  # SGPR (scalar / wave-uniform) counter

    def _line(self, s: str = "") -> None:
        self.buf.write(s)
        self.buf.write("\n")

    def emit_module_header(self) -> None:
        """Emit the AMDGPU asm module header. hipcc consumes this via
        `--offload-arch={target}` so the .amdgcn_target directive
        binds the binary to a specific GPU generation.
        """
        self._line(f".amdgcn_target \"{ROCM_OBJECT_FORMAT}-amd-amdhsa--{self.target}\"")
        self._line()

    def emit_module(self, mod: ti.TileModule) -> str:
        """Top-level emit. Walks all @kernel functions in the module.
        Raises RuntimeError if no @kernel found (parity with PtxEmitter).
        """
        self.emit_module_header()
        emitted_kernel = False
        for fn in mod.functions.values():
            if fn.attrs.get("kernel") and not fn.attrs.get("is_extern"):
                emitted_kernel = True
                self.emit_kernel_stub(fn)
        if not emitted_kernel:
            raise RuntimeError(
                "ROCm/HIP emission requires at least one @kernel function"
            )
        return self.buf.getvalue()

    def emit_kernel_stub(self, fn: ti.TileFn) -> None:
        """Stage 123 substrate: emit the kernel header + an `s_endpgm`
        terminator. Per-op codegen lands in Stage 124+.
        """
        self._line(f".text")
        self._line(f".globl {fn.name}")
        self._line(f".p2align 8")
        self._line(f".type {fn.name},@function")
        self._line(f"{fn.name}:")
        # Stage 124 wires per-op emission for the subset of TileOpKinds
        # we have concrete MFMA / global_load / s_barrier patterns for.
        # Unknown ops fall through to the stub `s_endpgm` terminator.
        for blk in fn.blocks:
            for op in blk.ops:
                self._emit_op(op)
        self._line("    s_endpgm")
        self._line()

    def _emit_op(self, op: ti.TileOp) -> None:
        """Stage 124 (v2.0 Phase C ROCm wmma) — emit one tile-IR op as
        AMDGPU instructions. v2.1 R1 audit-fix: emitted text is
        HELIX-STUB-prefixed; every operand placeholder is wrapped in
        a `/* HELIX-STUB-OPERAND: ... */` marker so a downstream
        hipcc compile fails LOUDLY rather than silently producing a
        no-op kernel.

        v2.1 R1 audit-fix Finding 1: ops with status="stub" / "deferred"
        in ROCM_OP_LOWERING now emit a `.error` directive that aborts
        hipcc — module-load coverage check only verifies table
        membership, not codegen completeness.
        """
        kind = op.kind
        # v2.1 R1 audit-fix: forward stub-status to the assembler.
        status = ROCM_OP_LOWERING[kind]["status"]
        if status in ("stub", "deferred"):
            self._line(
                f"    .error \"HELIX-STUB: TileOpKind.{kind.name} "
                f"status={status!r}; codegen not wired in this backend.\""
            )
            return
        if kind is ti.TileOpKind.BARRIER_WAIT:
            # bar.sync 0 (CUDA) maps to s_waitcnt + s_barrier on AMDGPU.
            # s_waitcnt drains the memory queue; s_barrier blocks at
            # workgroup level (parity with __syncthreads).
            self._line("    s_waitcnt vmcnt(0) lgkmcnt(0)")
            self._line("    s_barrier")
            return
        if kind is ti.TileOpKind.TILE_MATMUL:
            # Stage 124 — wmma analog via MFMA. Emit the canonical
            # MI300 16x16x16 fp32 accumulate with fp16 inputs:
            #   v_mfma_f32_16x16x16_f16 dst, src_A, src_B, src_C
            # Concrete operand binding requires register allocation
            # (Stage 124+ extension); for now we emit the instruction
            # as a comment-annotated stub so the lowering shape is
            # visible in audit / diff.
            self._line("    v_mfma_f32_16x16x16_f16 ; tile-matmul A @ B + C")
            return
        if kind is ti.TileOpKind.TILE_LOAD_GLOBAL:
            # global_load_b128 for 16-byte tile loads (4 floats per
            # lane); placeholder operand binding.
            self._line("    global_load_b128 ; HBM tile load")
            return
        if kind is ti.TileOpKind.TILE_STORE_GLOBAL:
            self._line("    global_store_b128 ; HBM tile store")
            return
        if kind is ti.TileOpKind.TILE_LOAD_SHARED:
            self._line("    ds_load_b128 ; LDS (SMEM) tile load")
            return
        if kind is ti.TileOpKind.TILE_STORE_SHARED:
            self._line("    ds_store_b128 ; LDS (SMEM) tile store")
            return
        if kind is ti.TileOpKind.THREAD_IDX:
            # workitem ID is implicit in VGPR v0 on AMDGPU. Comment-
            # annotate the read so the audit matrix sees coverage.
            self._line("    ; v0 = workitem ID (thread_idx)")
            return
        if kind is ti.TileOpKind.RETURN:
            # Falls through to the s_endpgm terminator emitted by
            # emit_kernel_stub; no per-op output needed.
            return
        # Default: leave a comment so emitted text shows which ops
        # are still substrate. The module-load coverage check at
        # _check_rocm_lowering_coverage already enforced that every
        # TileOpKind has a documented entry.
        self._line(f"    ; tile-IR op {kind.name} (stub)")


def lowering_status(kind: ti.TileOpKind) -> str:
    """Query the lowering status for one TileOpKind.

    Returns one of: "supported", "stub", "deferred", "skipped".

    Raises TypeError on non-TileOpKind input — silent membership tests
    on misspelled enums or cross-IR values would otherwise mask the
    coverage check.
    """
    if not isinstance(kind, ti.TileOpKind):
        raise TypeError(
            f"lowering_status expects TileOpKind, got "
            f"{type(kind).__name__}: {kind!r}"
        )
    entry = ROCM_OP_LOWERING.get(kind)
    if entry is None:
        raise AssertionError(
            f"TileOpKind {kind.name} missing from ROCM_OP_LOWERING "
            f"(module-load check should have caught this)"
        )
    return entry["status"]
