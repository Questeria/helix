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
from ._lowering_schema import (  # v2.3 item 2 shared schema
    OpLowering, VALID_STATUSES, is_loud_stub_status,
)


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
ROCM_OP_LOWERING: Final[Mapping[ti.TileOpKind, OpLowering]] = {
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
        # v2.1 R1 audit-fix Finding H1 (code-reviewer): the AMDGPU LDS
        # mnemonics are ds_read_b32 / ds_read_b64 / ds_read_b128 — NOT
        # ds_load_*. llvm-mc would reject ds_load_b128 outright. The
        # pre-R1 text passed tests only because the asserts matched
        # the (wrong) emitted token. Fixed both the table doc + emit
        # text + test asserts in this audit-fix.
        "lowering": "ds_read_b32 / ds_read_b64 / ds_read_b128 (LDS = SMEM analog)",
        "status": "supported",  # Stage 124
    },
    ti.TileOpKind.TILE_STORE_SHARED: {
        "lowering": "ds_write_b32 / ds_write_b64 / ds_write_b128",
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
    # v2.1 R1 audit-fix Finding H3 (silent-failure-hunter): the
    # following 3 ops were marked status="supported" but had no
    # emit branch in `_emit_op`, so a kernel using +/-/* would
    # compile to an empty stub kernel with no error. The honest
    # state is "stub": the emit pattern is documented in the
    # lowering string but not yet wired. Demoted accordingly;
    # the exhaustiveness guard at the end of `_emit_op` now
    # forces future re-promotion to include a concrete branch.
    ti.TileOpKind.TILE_ADD: {
        "lowering": "v_add_f32 / v_add_u32",
        "status": "stub",
    },
    ti.TileOpKind.TILE_SUB: {
        "lowering": "v_sub_f32 / v_sub_u32",
        "status": "stub",
    },
    ti.TileOpKind.TILE_MUL: {
        "lowering": "v_mul_f32 / v_mul_lo_u32",
        "status": "stub",
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
    # v2.1 R1 audit-fix Finding H3 (silent-failure-hunter): same
    # phantom-supported bug as TILE_ADD/SUB/MUL — these were marked
    # "supported" but had no _emit_op branch. Demoted to "stub" so
    # the .error directive at line 294 fires loudly.
    ti.TileOpKind.TILE_INDEX_LOAD_HBM: {
        "lowering": "global_load_<dtype> (HBM = global memory on AMD)",
        "status": "stub",
    },
    ti.TileOpKind.TILE_INDEX_STORE_HBM: {
        "lowering": "global_store_<dtype>",
        "status": "stub",
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
        # v2.3 5-clean-gate BE MEDIUM-1 audit-fix: validate the status
        # value against the shared VALID_STATUSES set at module load.
        # Pre-fix a typo like "stubb" passed the coverage check, was
        # treated as not-loud-stub by the `in ("stub","deferred")`
        # guard, and the op fell to the exhaustiveness AssertionError.
        # Catching it here names the offending kind + bad value.
        status = ROCM_OP_LOWERING[k]["status"]
        if status not in VALID_STATUSES:
            raise AssertionError(
                f"helixc.backend.rocm: ROCM_OP_LOWERING[{k.name}] has "
                f"status={status!r}, not in {sorted(VALID_STATUSES)}."
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
        AMDGPU instructions.

        Status-to-emit policy (v2.1 R1 audit-fix):
          * status in {"stub", "deferred"} → emit `.error "HELIX-STUB: …"`
            so hipcc aborts loudly. Pre-R1 silently fell through.
          * status == "skipped" → emit
            `.error "HELIX-SKIPPED: …"`. Pre-R1 fell through to the
            catch-all comment (silent miscompile, e.g. TMA_LOAD on AMD).
          * status == "supported" → MUST have a concrete `if kind is …`
            branch below. The exhaustiveness guard at the end of this
            function fires AssertionError if a "supported" op reaches
            it; that catches the Stage-120 R2/R3 lesson at the codegen
            layer (table-claims-supported but codegen has no branch).
        """
        kind = op.kind
        status = ROCM_OP_LOWERING[kind]["status"]
        # v2.3 5-clean-gate BE MEDIUM-1 audit-fix: gate on the shared
        # `is_loud_stub_status` helper rather than an inline
        # `status in ("stub","deferred")` literal — so the schema in
        # _lowering_schema.py is the single source of truth. If a 5th
        # loud status is ever added there, this guard picks it up.
        if is_loud_stub_status(status):
            if status == "skipped":
                # v2.1 R1 audit-fix Finding H2 (silent-failure-hunter):
                # pre-R1 the "skipped" status fell through to a silent
                # comment because only "stub"/"deferred" hit the .error
                # branch. TMA_LOAD/TMA_STORE on the ROCm path would
                # produce a benign-looking AMDGPU kernel with no error,
                # the exact failure class Stage 120 R3 closed for grad.
                # "skipped" means "no analog on this target" — must be
                # LOUDER than a stub, not quieter.
                self._line(
                    f"    .error \"HELIX-SKIPPED: TileOpKind.{kind.name} "
                    f"has no AMD analog (NVIDIA-only); routing to ROCm "
                    f"backend is a bug.\""
                )
            else:  # "stub" or "deferred"
                self._line(
                    f"    .error \"HELIX-STUB: TileOpKind.{kind.name} "
                    f"status={status!r}; codegen not wired in this "
                    f"backend.\""
                )
            return
        if kind is ti.TileOpKind.BARRIER_WAIT:
            # bar.sync 0 (CUDA) maps to s_waitcnt + s_barrier on AMDGPU.
            self._line("    s_waitcnt vmcnt(0) lgkmcnt(0)")
            self._line("    s_barrier")
            return
        if kind is ti.TileOpKind.TILE_MATMUL:
            # Stage 124 — MFMA wmma analog. Canonical MI300 16x16x16
            # fp32-accumulate with fp16 inputs. v2.x re-audit R1 (BE
            # 5-clean-gate HIGH): this and the five memory / index ops
            # below emit substrate-level text with operands NOT bound.
            # The `HELIX-STUB-OPERANDS` marker (parity with metal.py /
            # webgpu.py) makes gpu_ci.validate_emit flag the kernel as
            # non-functional. Pre-fix rocm omitted the marker its
            # sibling backends carry, so an operand-less ROCm kernel
            # passed gpu_ci mock validation as if it were functional.
            self._line("    v_mfma_f32_16x16x16_f16 ; tile-matmul "
                       "A @ B + C HELIX-STUB-OPERANDS")
            return
        if kind is ti.TileOpKind.TILE_LOAD_GLOBAL:
            self._line("    global_load_b128 ; HBM tile load "
                       "HELIX-STUB-OPERANDS")
            return
        if kind is ti.TileOpKind.TILE_STORE_GLOBAL:
            self._line("    global_store_b128 ; HBM tile store "
                       "HELIX-STUB-OPERANDS")
            return
        if kind is ti.TileOpKind.TILE_LOAD_SHARED:
            # v2.1 R1 audit-fix Finding H1: ds_read_b*, not ds_load_b*.
            # The latter is not a valid AMDGPU mnemonic.
            self._line("    ds_read_b128 ; LDS (SMEM) tile load "
                       "HELIX-STUB-OPERANDS")
            return
        if kind is ti.TileOpKind.TILE_STORE_SHARED:
            self._line("    ds_write_b128 ; LDS (SMEM) tile store "
                       "HELIX-STUB-OPERANDS")
            return
        if kind is ti.TileOpKind.THREAD_IDX:
            self._line("    ; v0 = workitem ID (thread_idx) "
                       "HELIX-STUB-OPERANDS")
            return
        if kind is ti.TileOpKind.RETURN:
            # Falls through to the s_endpgm terminator emitted by
            # emit_kernel_stub; no per-op output needed.
            return
        # v2.1 R1 audit-fix Finding H3/M1 (silent-failure-hunter):
        # exhaustiveness guard. status == "supported" reaching here
        # means the table claims supported but no branch exists.
        # Pre-R1 such ops silently emitted `; tile-IR op KIND (stub)`,
        # and the audit confirmed 5 such phantom-supported entries
        # (TILE_ADD/SUB/MUL/INDEX_LOAD_HBM/INDEX_STORE_HBM); those
        # are now demoted to status="stub" in ROCM_OP_LOWERING. The
        # guard remains so future drift fires loudly.
        raise AssertionError(
            f"ROCm._emit_op: TileOpKind.{kind.name} has "
            f"status='supported' in ROCM_OP_LOWERING but no codegen "
            f"branch in `_emit_op`. Either add a branch or demote "
            f"the table entry to status='stub'."
        )


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
