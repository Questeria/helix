"""
helixc/backend/ptx.py — NVIDIA PTX backend (text emission).

PTX is a *text-format virtual ISA*. We don't need an assembler — we just
write strings that the NVIDIA driver JIT-compiles to SASS at module-load
time, or that ptxas pre-compiles offline if you prefer AOT.

v0.1 scope: emit PTX for tile-level kernels marked `@kernel`. The emitted
PTX targets sm_75+ (Turing baseline) by default; raise to sm_90+ when
TMA / WGMMA are needed.

This is a STUB — real codegen (matmul, tile_load, etc.) lands incrementally.
What we have here:
  - PTX module header (.version, .target, .address_size)
  - Empty kernel emission
  - Scalar arithmetic on i32/f32 inside kernels
  - Test that the emitted text is syntactically plausible (round-trips
    through ptxas if available, else just shape-checked)

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
PTX_VERSION = "8.3"           # corresponds to CUDA 12.3 baseline
DEFAULT_TARGET = "sm_75"      # Turing+ (RTX 3070 is sm_86; RTX 5090 is sm_120)
ADDRESS_SIZE = 64


# ============================================================================
# Op-mapping table (tile-IR → PTX). v2.2 polish item 1: hand-maintained
# PTX_BASELINE_STATUS in tile_ir_audit.py was a drift hazard — the PTX
# baseline now lives here next to the emit code, mirroring the
# rocm/metal/webgpu pattern. `tile_ir_audit._lookup_ptx` reads from
# this table so the audit matrix and the per-backend tables share one
# source of truth.
#
# Each entry:
#   `lowering`: a short human-readable description of the PTX text we
#               emit (or expect to emit) for this kind.
#   `status`: one of {"supported", "stub", "deferred", "skipped"}.
#     supported: PtxEmitter.emit_op has a concrete branch.
#     stub:      placeholder emit (no real codegen).
#     deferred:  blocked on a later stage.
#     skipped:   no PTX analog; documented for completeness.
#
# Status values mirror the v2.1 audit matrix at the v2.1.0 5-clean
# gate (per docs/V2_PLAN.md, every TILE_* op was "stub" — substrate
# emit exists but operand binding is incomplete).
PTX_OP_LOWERING: Final[Mapping[ti.TileOpKind, OpLowering]] = {
    # v2.3 5-clean-gate BE HIGH-1 audit-fix: TILE_ZEROS / TILE_ADD /
    # TILE_SUB / TILE_MUL / TILE_MATMUL have REAL Stage 64 emit
    # branches in PtxEmitter.emit_op (the "102 PTX pins green" from
    # v1.0 — register-fill, elementwise add/sub/mul, wmma.mma.sync
    # Tensor-Core matmul). The v2.2-added PTX_OP_LOWERING table
    # blanket-marked every TILE_* op "stub" by mirroring the v2.1
    # audit matrix, which was wrong for PTX specifically — PTX
    # predates the v2 backends and shipped this codegen in Stage 64.
    # Status corrected to "supported" to match the dispatcher; the
    # forward guard at the top of emit_op now keys on these values.
    ti.TileOpKind.TILE_ZEROS:           {"lowering": "mov.u32 / mov.f32 (init VGPR/SGPR to 0)",  "status": "supported"},
    ti.TileOpKind.TILE_CONST:           {"lowering": "mov.u32 / mov.f32 (constant load)",         "status": "stub"},
    ti.TileOpKind.TILE_LOAD_GLOBAL:     {"lowering": "ld.global.{b32,b64,b128}",                  "status": "stub"},
    ti.TileOpKind.TILE_STORE_GLOBAL:    {"lowering": "st.global.{b32,b64,b128}",                  "status": "stub"},
    ti.TileOpKind.TILE_LOAD_SHARED:     {"lowering": "ld.shared.{b32,b64,b128}",                  "status": "stub"},
    ti.TileOpKind.TILE_STORE_SHARED:    {"lowering": "st.shared.{b32,b64,b128}",                  "status": "stub"},
    ti.TileOpKind.TMA_LOAD:             {"lowering": "cp.async.bulk.tensor (Hopper TMA)",         "status": "stub"},
    ti.TileOpKind.TMA_STORE:            {"lowering": "cp.async.bulk.tensor.store",                "status": "stub"},
    ti.TileOpKind.BARRIER_WAIT:         {"lowering": "bar.sync / bar.cta.sync",                   "status": "stub"},
    ti.TileOpKind.TILE_ADD:             {"lowering": "add.{f32,s32} (tile-level)",                "status": "supported"},
    ti.TileOpKind.TILE_SUB:             {"lowering": "sub.{f32,s32}",                             "status": "supported"},
    ti.TileOpKind.TILE_MUL:             {"lowering": "mul.{f32,s32} / mul.lo.s32",                "status": "supported"},
    ti.TileOpKind.TILE_MATMUL:          {"lowering": "wmma.mma.sync (m16n16k16 Tensor Cores)",   "status": "supported"},
    ti.TileOpKind.TILE_REDUCE:          {"lowering": "shfl.sync.bfly + ld.shared (warp tree)",   "status": "stub"},
    ti.TileOpKind.TILE_TRANSPOSE:       {"lowering": "ld.shared / st.shared with index swap",    "status": "stub"},
    ti.TileOpKind.TILE_RESHAPE:         {"lowering": "noop (reshape is a view-level operation)", "status": "stub"},
    ti.TileOpKind.SCALAR_CONST_INT:     {"lowering": "mov.s32 / mov.b32",                         "status": "supported"},
    ti.TileOpKind.SCALAR_CONST_FLOAT:   {"lowering": "mov.f32",                                   "status": "supported"},
    ti.TileOpKind.SCALAR_ADD:           {"lowering": "add.{s32,f32}",                             "status": "supported"},
    ti.TileOpKind.SCALAR_SUB:           {"lowering": "sub.{s32,f32}",                             "status": "supported"},
    ti.TileOpKind.SCALAR_MUL:           {"lowering": "mul.{lo.s32,f32}",                          "status": "supported"},
    ti.TileOpKind.SCALAR_NEG:           {"lowering": "neg.{s32,f32}",                             "status": "supported"},
    ti.TileOpKind.SCALAR_CMP:           {"lowering": "setp.{eq,ne,lt,le,gt,ge}.{s32,f32}",       "status": "supported"},
    # v2.2 polish item 1 R1 audit-fix CRIT-1: SCALAR_SELECT and CALL
    # were phantom-supported — they had no emit branch in PtxEmitter
    # but the hand-maintained PTX_BASELINE_STATUS dict claimed
    # "supported". Demoted to "stub" to match reality. The parity
    # guard at the bottom of emit_op now raises AssertionError if
    # a kind tagged "supported" reaches the dispatcher unbranched.
    ti.TileOpKind.SCALAR_SELECT:        {"lowering": "selp.{b32,f32}",                            "status": "stub"},
    ti.TileOpKind.CALL:                 {"lowering": "call.uni / call",                           "status": "stub"},
    ti.TileOpKind.RETURN:               {"lowering": "ret (.uni for kernels)",                    "status": "supported"},
    ti.TileOpKind.THREAD_IDX:           {"lowering": "%tid.x / %tid.y / %tid.z",                  "status": "supported"},
    ti.TileOpKind.TILE_INDEX_LOAD_HBM:  {"lowering": "ld.global.<dtype> via base+index",          "status": "supported"},
    ti.TileOpKind.TILE_INDEX_STORE_HBM: {"lowering": "st.global.<dtype> via base+index",          "status": "supported"},
}


# Status-tag invariant: every TileOpKind must appear in PTX_OP_LOWERING.
# Same drift-detector pattern as rocm/metal/webgpu — adding a new kind
# fires loudly at module-load until a conscious port decision is made.
def _check_ptx_lowering_coverage() -> None:
    """Module-load: every TileOpKind must be classified for PTX."""
    for k in ti.TileOpKind:
        if k not in PTX_OP_LOWERING:
            raise AssertionError(
                f"helixc.backend.ptx: TileOpKind {k.name} is missing "
                f"from PTX_OP_LOWERING. Every kind must have a "
                f"lowering or be marked status='skipped' with rationale."
            )
        # v2.3 5-clean-gate BE MEDIUM-1 audit-fix: validate status
        # against the shared VALID_STATUSES set at module load.
        status = PTX_OP_LOWERING[k]["status"]
        if status not in VALID_STATUSES:
            raise AssertionError(
                f"helixc.backend.ptx: PTX_OP_LOWERING[{k.name}] has "
                f"status={status!r}, not in {sorted(VALID_STATUSES)}."
            )


_check_ptx_lowering_coverage()


def lowering_status(kind: ti.TileOpKind) -> str:
    """Query the PTX lowering status for one TileOpKind.

    Returns one of: "supported", "stub", "deferred", "skipped".

    Raises TypeError on non-TileOpKind input — silent membership tests
    on misspelled enums or cross-IR values would otherwise mask the
    coverage check. Parity with rocm.lowering_status,
    metal.lowering_status, webgpu.lowering_status.
    """
    if not isinstance(kind, ti.TileOpKind):
        raise TypeError(
            f"lowering_status expects TileOpKind, got "
            f"{type(kind).__name__}: {kind!r}"
        )
    entry = PTX_OP_LOWERING.get(kind)
    if entry is None:
        raise AssertionError(
            f"TileOpKind {kind.name} missing from PTX_OP_LOWERING "
            f"(module-load check should have caught this)"
        )
    return entry["status"]


# ============================================================================
# Emitter
# ============================================================================
class PtxEmitter:
    def __init__(self, target: str = DEFAULT_TARGET):
        self.target = target
        self.buf = StringIO()
        self.next_reg = 0
        # Map TileValue.id -> PTX register name like "%r3"
        self.reg_map: dict[int, str] = {}
        # Stage 16 — per-kernel state. Maps HBM tile param NAME (from
        # TILE_INDEX_LOAD_HBM/STORE_HBM attrs) to (param_index, dtype).
        # Built in emit_kernel by scanning kernel params; reset per kernel.
        self.hbm_param_map: dict[str, tuple[int, str]] = {}
        # Stage 16 — independent per-prefix counters so %r3 / %f3 / %rd3
        # are independent register pools. (Legacy `next_reg` was shared
        # and would collide.)
        self.next_reg_by_prefix: dict[str, int] = {}

    # Audit A3-MEDIUM-1: per-prefix register pool caps. The .reg
    # declarations in emit_kernel reserved %p<8>, %r<32>, %rd<8>, %f<32>
    # — beyond those, the emitted PTX referenced registers (e.g. %r32,
    # %r33, ...) that weren't declared. The CUDA driver's PTX assembler
    # rejects undeclared registers but the bootstrap pipeline didn't
    # surface that until ptxas. We bumped the declared pool sizes to
    # 256 and added an explicit overflow check so an over-cap kernel
    # raises a Python-level error pinned to the codegen site.
    _REG_POOL_CAP = 256

    def _new_reg(self, prefix: str = "r") -> str:
        # Use per-prefix counter so int / fp / 64-bit pools stay separate
        # and the PTX reads cleanly. Maintains legacy `next_reg` for the
        # tiny chance an old caller relies on the global counter.
        c = self.next_reg_by_prefix.get(prefix, 0)
        if c >= self._REG_POOL_CAP:
            raise RuntimeError(
                f"PTX register pool overflow: prefix '%{prefix}' exceeded "
                f"{self._REG_POOL_CAP} registers. Bump _REG_POOL_CAP and the "
                f"matching .reg declaration in emit_kernel, or split the kernel."
            )
        n = f"%{prefix}{c}"
        self.next_reg_by_prefix[prefix] = c + 1
        self.next_reg += 1
        return n

    def _line(self, s: str = "") -> None:
        self.buf.write(s)
        self.buf.write("\n")

    # ---- module ----
    def emit_module_header(self) -> None:
        self._line(f".version {PTX_VERSION}")
        self._line(f".target {self.target}")
        self._line(f".address_size {ADDRESS_SIZE}")
        self._line()

    def emit_module(self, mod: ti.TileModule) -> str:
        self.emit_module_header()
        emitted_kernel = False
        for fn in mod.functions.values():
            if fn.attrs.get("kernel") and not fn.attrs.get("is_extern"):
                emitted_kernel = True
                self.emit_kernel(fn)
        if not emitted_kernel:
            raise RuntimeError(
                "PTX emission requires at least one @kernel function"
            )
        return self.buf.getvalue()

    # ---- kernel ----
    def emit_kernel(self, fn: ti.TileFn) -> None:
        self._validate_kernel_params(fn)
        params_str = ", ".join(self._format_param(p, i) for i, p in enumerate(fn.params))
        self._line(f".visible .entry {fn.name}({params_str})")
        self._line("{")
        self.next_reg = 0
        self.next_reg_by_prefix = {}
        self.reg_map = {}
        # Stage 16 — build the HBM tile param map. The TileValue.ty for an
        # HBM tile param is a TIRTileTy; its name_hint matches the source
        # parameter name (so `TILE_INDEX_LOAD attrs={'name':'a'}` can find
        # the right `.param .u64 param_0` slot).
        self.hbm_param_map = {}
        for i, p in enumerate(fn.params):
            if isinstance(p.ty, tir.TIRTileTy) \
                    and p.ty.memspace.lower() == "hbm":
                if p.name_hint:
                    self.hbm_param_map[p.name_hint] = (i, p.ty.dtype.name)
        # Reserve a register pool. Audit A3-MEDIUM-1: bumped from
        # %p<8>/%r<32>/%rd<8>/%f<32> to %p<256>/%r<256>/%rd<256>/%f<256>
        # so kernels with many SSA values in flight don't silently emit
        # references to undeclared registers. Per-prefix overflow is
        # checked at codegen time in _new_reg (raises RuntimeError).
        self._line(f"    .reg .pred  %p<{self._REG_POOL_CAP}>;")
        self._line(f"    .reg .b32   %r<{self._REG_POOL_CAP}>;")
        self._line(f"    .reg .b64   %rd<{self._REG_POOL_CAP}>;")
        self._line(f"    .reg .f32   %f<{self._REG_POOL_CAP}>;")
        # Stage 64 Inc 1 — Tier 2 #6: declare a 16-bit register pool
        # for bf16 + f16 tile values. PTX uses .b16 as the underlying
        # register class; values are interpreted as bf16/f16 by the
        # ld/st suffixes.
        self._line(f"    .reg .b16   %h<{self._REG_POOL_CAP}>;")
        self._line()
        for blk in fn.blocks:
            self._line(f"BB{blk.id}:")
            for op in blk.ops:
                self.emit_op(op)
        self._line("    ret;")
        self._line("}")
        self._line()

    def emit_device_func(self, fn: ti.TileFn) -> None:
        raise RuntimeError(
            f"PTX device function emission is not supported yet for {fn.name!r}"
        )

    def _format_param(self, p: ti.TileValue, idx: int) -> str:
        # Kernel params are in `.param` space; addresses come in as .b64
        # For v0.1 we treat all params as .b64 (pointer-like)
        return f".param .b64 param_{idx}"

    def _ptx_type_str(self, ty: tir.TIRType) -> str:
        if isinstance(ty, tir.TIRScalar):
            # Audit 28.8 cycle 21 C20-1 (HIGH): pointer-width aliases
            # isize/usize must be 64-bit (matching typecheck.py canon
            # + the cycle-19 backend classifier fix + cycle-20 const_fold
            # fix). Pre-fix the `.get(..., ".b32")` silently fell back
            # to 32-bit for isize/usize. Same defect class as C13-1/
            # C16-1/C18-1/C19-1 — silent narrowing where one backend's
            # width contract disagreed with the canon.
            mapping = {
                "i8": ".b8", "i16": ".b16", "i32": ".b32", "i64": ".b64",
                "u8": ".b8", "u16": ".b16", "u32": ".b32", "u64": ".b64",
                "isize": ".b64", "usize": ".b64",
                "bool": ".pred",
                "f16": ".f16", "bf16": ".bf16", "f32": ".f32", "f64": ".f64",
                "char": ".b8",
            }
            # Cycle 1 Batch BE silent-failure HIGH-3 fix: pre-fix,
            # mapping.get(ty.name, ".b32") silently emitted .b32 for
            # unknown TIRScalar dtypes — ptxas may accept the wrong
            # register declaration without parse error, producing
            # wrong machine code at runtime. Post-fix: KeyError → raise.
            try:
                return mapping[ty.name]
            except KeyError:
                raise RuntimeError(
                    f"_ptx_type_str: unsupported TIRScalar dtype "
                    f"{ty.name!r}; add to mapping (Cycle 1 Batch BE "
                    f"silent-failure HIGH-3 fix)"
                )
        if isinstance(ty, tir.TIRUnit):
            return ""
        # Cycle 1 Batch BE silent-failure HIGH-3 fix: pre-fix, the bare
        # `return ".b64"` silently emitted .b64 for any non-TIRScalar,
        # non-TIRUnit type (TIRTensorTy, TIRTileTy, future struct types,
        # etc.) — wrong PTX with no error. Post-fix: raise on unknown
        # TIRType to surface the missing dispatch arm.
        raise RuntimeError(
            f"_ptx_type_str: unsupported TIRType {type(ty).__name__} "
            f"({ty!r}); add an explicit dispatch arm (Cycle 1 Batch BE "
            f"silent-failure HIGH-3 fix)"
        )

    def _validate_kernel_params(self, fn: ti.TileFn) -> None:
        if not isinstance(fn.return_ty, tir.TIRUnit):
            raise RuntimeError(
                "PTX kernels with non-unit returns are not supported yet"
            )
        for p in fn.params:
            if (isinstance(p.ty, tir.TIRTileTy)
                    and p.ty.memspace.lower() == "hbm"):
                self._require_supported_hbm_dtype(p.ty.dtype.name)
                if len(p.ty.shape) != 1:
                    raise RuntimeError(
                        "PTX HBM tile parameters must be 1D; "
                        f"got {len(p.ty.shape)}D"
                    )
                continue
            raise RuntimeError(
                "PTX kernel parameter is not supported yet; "
                "only HBM tile parameters are currently lowered"
            )

    def _require_reg(self, op: ti.TileOp, operand_index: int, role: str) -> str:
        if operand_index >= len(op.operands):
            raise RuntimeError(f"missing PTX operand for {role}")
        reg = self.reg_map.get(op.operands[operand_index].id)
        if reg is None:
            raise RuntimeError(
                f"missing PTX register for {role}; "
                "only lowered values are supported"
            )
        return reg

    def _scalar_type_name(self, value: ti.TileValue) -> str | None:
        if isinstance(value.ty, tir.TIRScalar):
            return value.ty.name
        return None

    def _require_scalar_type(
        self, value: ti.TileValue, role: str, allowed: set[str]
    ) -> str:
        name = self._scalar_type_name(value)
        if name not in allowed:
            allowed_s = ", ".join(sorted(allowed))
            raise RuntimeError(
                f"unsupported PTX {role} type {name}; "
                f"only {allowed_s} is currently lowered"
            )
        return name

    def _require_scalar_result_type(
        self, op: ti.TileOp, expected: str, role: str
    ) -> None:
        self._require_scalar_type(op.results[0], role, {expected})

    def _require_reg_class(self, reg: str, prefix: str, role: str) -> None:
        if reg.startswith("%rd"):
            actual = "%rd"
        elif reg.startswith("%r"):
            actual = "%r"
        elif reg.startswith("%f"):
            actual = "%f"
        elif reg.startswith("%p"):
            actual = "%p"
        else:
            actual = reg
        if actual != prefix:
            raise RuntimeError(
                f"unsupported PTX {role} register {reg}; "
                f"expected {prefix} register class"
            )

    def _require_result_count(self, op: ti.TileOp, count: int, role: str) -> None:
        if len(op.results) != count:
            raise RuntimeError(f"{role} expects exactly {count} result(s)")

    def _require_operand_count(self, op: ti.TileOp, count: int, role: str) -> None:
        if len(op.operands) != count:
            raise RuntimeError(f"{role} expects exactly {count} operand(s)")

    def _require_supported_hbm_dtype(self, dtype: str) -> None:
        # Stage 64 Inc 1 — Tier 2 #6: lift bf16 + f16 from HBM
        # rejection. 16-bit floats use the `%h` register pool (.b16
        # register class in PTX) and `.b16` / `.bf16` ld/st suffixes.
        # Pre-Stage-64: only f32 + i32 were allowed; bf16 was tagged
        # in _DTYPE_SIZE / _DTYPE_PTX_LOAD but rejected here, blocking
        # any tile<bf16, ...> HBM round-trip.
        if dtype not in {"f32", "i32", "bf16", "f16"}:
            raise RuntimeError(
                f"unsupported PTX HBM tile dtype {dtype}; "
                "only f32, i32, bf16, f16 HBM tile elements "
                "are currently lowered"
            )

    def _require_hbm_dtype_attr(self, op: ti.TileOp) -> str:
        if "dtype" not in op.attrs:
            raise RuntimeError("missing PTX HBM tile dtype attr")
        dtype = str(op.attrs["dtype"])
        self._require_supported_hbm_dtype(dtype)
        return dtype

    def _require_hbm_index_reg(self, op: ti.TileOp, operand_index: int) -> str:
        value = op.operands[operand_index]
        self._require_scalar_type(value, "HBM tile index", {"i32"})
        reg = self._require_reg(op, operand_index, "HBM tile index")
        self._require_reg_class(reg, "%r", "HBM tile index")
        return reg

    def _require_hbm_value_reg(
        self, op: ti.TileOp, operand_index: int, dtype: str, role: str
    ) -> str:
        # Stage 64 Inc 1 — Tier 2 #6: extend value-reg dispatch for
        # bf16 + f16. Both 16-bit floats use the `%h` register class
        # (PTX .b16 register pool) and scalar type `bf16` / `f16` at
        # the TIR layer.
        value = op.operands[operand_index]
        expected = {"f32": ("f32", "%f"),
                    "i32": ("i32", "%r"),
                    "bf16": ("bf16", "%h"),
                    "f16": ("f16", "%h")}[dtype]
        self._require_scalar_type(value, role, {expected[0]})
        reg = self._require_reg(op, operand_index, role)
        self._require_reg_class(reg, expected[1], role)
        return reg

    def _require_hbm_load_result_type(self, op: ti.TileOp, dtype: str) -> None:
        if not op.results:
            return
        # Stage 64 Inc 1: result type mapping for bf16 + f16.
        expected = {"f32": "f32", "i32": "i32",
                    "bf16": "bf16", "f16": "f16"}[dtype]
        self._require_scalar_type(op.results[0], "HBM tile load result", {expected})

    # ---- ops ----
    def emit_op(self, op: ti.TileOp) -> None:
        # v2.3 5-clean-gate BE HIGH-1 audit-fix: loud-stub forward
        # guard. PTX was the one backend missing this — rocm/metal/
        # webgpu all open `_emit_op` with a status check that emits a
        # loud target-language directive for stub/deferred/skipped
        # ops. Without it, PTX could (a) emit phantom-functional
        # assembly for a table-declared stub, or (b) abort with a
        # raw Python RuntimeError instead of a `HELIX-STUB`-tokened
        # `.error` directive that downstream tooling (which
        # string-matches HELIX_STUB_TOKEN) can detect. `.error` is a
        # real PTX directive — ptxas aborts on it, so the failure is
        # loud at assemble time, parity with rocm's `.error`.
        # v2.3 5-clean-gate BE MEDIUM-1 audit-fix: gate on the shared
        # `is_loud_stub_status` helper so _lowering_schema.py is the
        # single source of truth for which statuses are loud-stub.
        status = PTX_OP_LOWERING[op.kind]["status"]
        if is_loud_stub_status(status):
            if status == "skipped":
                self._line(
                    f'    .error "HELIX-SKIPPED: TileOpKind.'
                    f'{op.kind.name} has no PTX analog; routing it '
                    f'here is a bug."'
                )
            else:  # "stub" or "deferred"
                self._line(
                    f'    .error "HELIX-STUB: TileOpKind.{op.kind.name} '
                    f'status={status!r}; codegen not wired in the PTX '
                    f'backend."'
                )
            return
        # v0.1: only handle a tiny scalar subset for sanity testing
        if op.kind == ti.TileOpKind.SCALAR_CONST_INT:
            self._require_operand_count(op, 0, "SCALAR_CONST_INT")
            self._require_result_count(op, 1, "SCALAR_CONST_INT")
            self._require_scalar_type(
                op.results[0], "integer constant", {"bool", "i32"}
            )
            if "value" not in op.attrs:
                raise RuntimeError("SCALAR_CONST_INT requires value attr")
            r = self._new_reg("r")
            v = op.attrs["value"]
            result_ty = self._scalar_type_name(op.results[0])
            if result_ty == "bool":
                if type(v) is bool:
                    v = 1 if v else 0
                elif type(v) is int and v in (0, 1):
                    pass
                else:
                    raise RuntimeError(
                        "SCALAR_CONST_INT bool value must be true/false or 0/1"
                    )
            elif type(v) is not int:
                raise RuntimeError("SCALAR_CONST_INT i32 value must be an int")
            elif v < -(2 ** 31) or v > (2 ** 31 - 1):
                raise RuntimeError("SCALAR_CONST_INT i32 value out of range")
            self._line(f"    mov.b32 {r}, {v};")
            if op.results:
                self.reg_map[op.results[0].id] = r
            return
        if op.kind == ti.TileOpKind.SCALAR_ADD:
            self._require_operand_count(op, 2, "SCALAR_ADD")
            self._require_result_count(op, 1, "SCALAR_ADD")
            # Stage 16 — float and int operand dispatch. If either operand
            # is in a %f register (TILE_INDEX_LOAD_HBM result), emit
            # `add.f32`; otherwise the integer fallback.
            a_reg = self._require_reg(op, 0, "scalar add lhs")
            b_reg = self._require_reg(op, 1, "scalar add rhs")
            a_ty = self._require_scalar_type(op.operands[0], "scalar add lhs",
                                             {"f32", "i32"})
            b_ty = self._require_scalar_type(op.operands[1], "scalar add rhs",
                                             {"f32", "i32"})
            if a_ty != b_ty:
                raise RuntimeError("unsupported PTX mixed scalar add types")
            self._require_scalar_result_type(op, a_ty, "scalar add result")
            if a_ty == "f32":
                self._require_reg_class(a_reg, "%f", "scalar add lhs")
                self._require_reg_class(b_reg, "%f", "scalar add rhs")
                r = self._new_reg("f")
                self._line(f"    add.f32 {r}, {a_reg}, {b_reg};")
            else:
                self._require_reg_class(a_reg, "%r", "scalar add lhs")
                self._require_reg_class(b_reg, "%r", "scalar add rhs")
                r = self._new_reg("r")
                self._line(f"    add.s32 {r}, {a_reg}, {b_reg};")
            if op.results:
                self.reg_map[op.results[0].id] = r
            return
        if op.kind == ti.TileOpKind.SCALAR_MUL:
            self._require_operand_count(op, 2, "SCALAR_MUL")
            self._require_result_count(op, 1, "SCALAR_MUL")
            a_reg = self._require_reg(op, 0, "scalar multiply lhs")
            b_reg = self._require_reg(op, 1, "scalar multiply rhs")
            a_ty = self._require_scalar_type(op.operands[0], "scalar multiply lhs",
                                             {"f32", "i32"})
            b_ty = self._require_scalar_type(op.operands[1], "scalar multiply rhs",
                                             {"f32", "i32"})
            if a_ty != b_ty:
                raise RuntimeError("unsupported PTX mixed scalar multiply types")
            self._require_scalar_result_type(op, a_ty, "scalar multiply result")
            if a_ty == "f32":
                self._require_reg_class(a_reg, "%f", "scalar multiply lhs")
                self._require_reg_class(b_reg, "%f", "scalar multiply rhs")
                r = self._new_reg("f")
                self._line(f"    mul.f32 {r}, {a_reg}, {b_reg};")
            else:
                self._require_reg_class(a_reg, "%r", "scalar multiply lhs")
                self._require_reg_class(b_reg, "%r", "scalar multiply rhs")
                r = self._new_reg("r")
                self._line(f"    mul.lo.s32 {r}, {a_reg}, {b_reg};")
            if op.results:
                self.reg_map[op.results[0].id] = r
            return
        if op.kind == ti.TileOpKind.SCALAR_SUB:
            self._require_operand_count(op, 2, "SCALAR_SUB")
            self._require_result_count(op, 1, "SCALAR_SUB")
            a_reg = self._require_reg(op, 0, "scalar subtract lhs")
            b_reg = self._require_reg(op, 1, "scalar subtract rhs")
            a_ty = self._require_scalar_type(op.operands[0], "scalar subtract lhs",
                                             {"f32", "i32"})
            b_ty = self._require_scalar_type(op.operands[1], "scalar subtract rhs",
                                             {"f32", "i32"})
            if a_ty != b_ty:
                raise RuntimeError("unsupported PTX mixed scalar subtract types")
            self._require_scalar_result_type(op, a_ty, "scalar subtract result")
            if a_ty == "f32":
                self._require_reg_class(a_reg, "%f", "scalar subtract lhs")
                self._require_reg_class(b_reg, "%f", "scalar subtract rhs")
                r = self._new_reg("f")
                self._line(f"    sub.f32 {r}, {a_reg}, {b_reg};")
            else:
                self._require_reg_class(a_reg, "%r", "scalar subtract lhs")
                self._require_reg_class(b_reg, "%r", "scalar subtract rhs")
                r = self._new_reg("r")
                self._line(f"    sub.s32 {r}, {a_reg}, {b_reg};")
            if op.results:
                self.reg_map[op.results[0].id] = r
            return
        if op.kind == ti.TileOpKind.SCALAR_NEG:
            self._require_operand_count(op, 1, "SCALAR_NEG")
            self._require_result_count(op, 1, "SCALAR_NEG")
            a_reg = self._require_reg(op, 0, "scalar neg operand")
            a_ty = self._require_scalar_type(op.operands[0], "scalar neg operand",
                                             {"f32", "i32"})
            self._require_scalar_result_type(op, a_ty, "scalar neg result")
            if a_ty == "f32":
                self._require_reg_class(a_reg, "%f", "scalar neg operand")
                r = self._new_reg("f")
                self._line(f"    neg.f32 {r}, {a_reg};")
            else:
                self._require_reg_class(a_reg, "%r", "scalar neg operand")
                r = self._new_reg("r")
                self._line(f"    neg.s32 {r}, {a_reg};")
            if op.results:
                self.reg_map[op.results[0].id] = r
            return
        if op.kind == ti.TileOpKind.SCALAR_CONST_FLOAT:
            self._require_operand_count(op, 0, "SCALAR_CONST_FLOAT")
            self._require_result_count(op, 1, "SCALAR_CONST_FLOAT")
            self._require_scalar_type(
                op.results[0], "float constant", {"f32"}
            )
            if "value" not in op.attrs:
                raise RuntimeError("SCALAR_CONST_FLOAT requires value attr")
            # f32 constant. Emit via mov.f32 with the hex bit pattern so
            # ptxas accepts the exact bits unambiguously.
            import struct
            v = op.attrs["value"]
            if type(v) is not float:
                raise RuntimeError("SCALAR_CONST_FLOAT value must be a float")
            bits = struct.unpack("<I", struct.pack("<f", v))[0]
            r = self._new_reg("f")
            self._line(f"    mov.f32 {r}, 0f{bits:08X};")
            if op.results:
                self.reg_map[op.results[0].id] = r
            return
        if op.kind == ti.TileOpKind.SCALAR_CMP:
            self._require_operand_count(op, 2, "SCALAR_CMP")
            self._require_result_count(op, 1, "SCALAR_CMP")
            self._require_scalar_type(op.results[0], "scalar compare result", {"bool"})
            if "cmp" not in op.attrs:
                raise RuntimeError("SCALAR_CMP requires cmp attr")
            cmp_op = op.attrs["cmp"]
            cmp_map = {"cmp.eq": "eq", "cmp.ne": "ne", "cmp.lt": "lt",
                        "cmp.le": "le", "cmp.gt": "gt", "cmp.ge": "ge"}
            if cmp_op not in cmp_map:
                raise RuntimeError(f"unsupported PTX scalar compare op {cmp_op!r}")
            cmp_suffix = cmp_map[cmp_op]
            a_reg = self._require_reg(op, 0, "scalar compare lhs")
            b_reg = self._require_reg(op, 1, "scalar compare rhs")
            a_ty = self._require_scalar_type(op.operands[0], "scalar compare lhs",
                                             {"f32", "i32"})
            b_ty = self._require_scalar_type(op.operands[1], "scalar compare rhs",
                                             {"f32", "i32"})
            if a_ty != b_ty:
                raise RuntimeError("unsupported PTX mixed scalar compare types")
            # Result lives in a %p predicate register. Phase-0 emits the
            # signed-int or f32 form based on the lowered register class.
            r = self._new_reg("p")
            if a_ty == "f32":
                self._require_reg_class(a_reg, "%f", "scalar compare lhs")
                self._require_reg_class(b_reg, "%f", "scalar compare rhs")
                self._line(f"    setp.{cmp_suffix}.f32 {r}, {a_reg}, {b_reg};")
            else:
                self._require_reg_class(a_reg, "%r", "scalar compare lhs")
                self._require_reg_class(b_reg, "%r", "scalar compare rhs")
                self._line(f"    setp.{cmp_suffix}.s32 {r}, {a_reg}, {b_reg};")
            if op.results:
                self.reg_map[op.results[0].id] = r
            return
        if op.kind == ti.TileOpKind.RETURN:
            if op.operands or op.results:
                raise RuntimeError("PTX kernels cannot return values")
            return
        # Stage 16 — GPU primitives.
        if op.kind == ti.TileOpKind.THREAD_IDX:
            # `thread_idx()` returns the i32 special-reg value. attrs={dim,sreg}
            # where dim is "x"/"y"/"z" and sreg is "tid" (default), "ctaid"
            # (block index), or "ntid" (block dim). Maps to PTX `mov.u32
            # %r, %<sreg>.<dim>`.
            self._require_operand_count(op, 0, "THREAD_IDX")
            self._require_result_count(op, 1, "THREAD_IDX")
            self._require_scalar_type(op.results[0], "THREAD_IDX result", {"i32"})
            if "dim" not in op.attrs or "sreg" not in op.attrs:
                raise RuntimeError("THREAD_IDX requires explicit dim and sreg attrs")
            dim = op.attrs["dim"]
            sreg = op.attrs["sreg"]
            if dim not in {"x", "y", "z"}:
                raise RuntimeError(f"unsupported PTX THREAD_IDX dim {dim!r}")
            if sreg not in {"tid", "ctaid", "ntid"}:
                raise RuntimeError(f"unsupported PTX THREAD_IDX sreg {sreg!r}")
            r = self._new_reg("r")
            self._line(f"    mov.u32 {r}, %{sreg}.{dim};")
            if op.results:
                self.reg_map[op.results[0].id] = r
            return
        if op.kind == ti.TileOpKind.TILE_INDEX_LOAD_HBM:
            self._require_operand_count(op, 1, "TILE_INDEX_LOAD_HBM")
            self._require_result_count(op, 1, "TILE_INDEX_LOAD_HBM")
            # `name[i]` for an HBM tile param. Sequence:
            #   ld.param.u64  %rdN, [<kernel>_param_<idx>]   (kernel-arg ptr)
            #   cvta.to.global.u64 %rdM, %rdN                (generic -> global)
            #   mul.wide.s32  %rdK, <idx>, sizeof(dtype)     (byte offset)
            #   add.s64       %rdL, %rdM, %rdK
            #   ld.global.<dtype> %fN, [%rdL]
            # Phase-0 collapses some of these: we just emit the textbook
            # PTX sequence for clarity; ptxas will optimize.
            name = op.attrs.get("name")
            dtype = self._require_hbm_dtype_attr(op)
            slot = self.hbm_param_map.get(name)
            if slot is None:
                raise RuntimeError(
                    f"HBM tile {name!r} not in PTX param map; "
                    "only lowered HBM tile parameters are supported"
                )
            param_idx, dtype_in_map = slot
            if str(dtype) != dtype_in_map:
                raise RuntimeError(
                    f"HBM tile {name!r} dtype mismatch: op requested {dtype}, "
                    f"but kernel param is {dtype_in_map}"
                )
            self._require_hbm_load_result_type(op, str(dtype))
            idx_reg = self._require_hbm_index_reg(op, 0)
            base = self._new_reg("rd")    # raw param-space pointer
            gen = self._new_reg("rd")     # generic-space pointer (after cvta)
            off = self._new_reg("rd")     # byte offset (idx * sizeof)
            addr = self._new_reg("rd")    # final ld address
            self._line(f"    ld.param.u64 {base}, [{self._kernel_param_label(param_idx)}];")
            self._line(f"    cvta.to.global.u64 {gen}, {base};")
            self._line(f"    mul.wide.s32 {off}, {idx_reg}, {self._dtype_size(dtype)};")
            self._line(f"    add.s64 {addr}, {gen}, {off};")
            dst_prefix = self._ld_reg_prefix(dtype)
            dst = self._new_reg(dst_prefix)
            self._line(f"    ld.global.{self._ptx_load_suffix(dtype)} {dst}, [{addr}];")
            if op.results:
                self.reg_map[op.results[0].id] = dst
            return
        if op.kind == ti.TileOpKind.TILE_INDEX_STORE_HBM:
            self._require_operand_count(op, 2, "TILE_INDEX_STORE_HBM")
            self._require_result_count(op, 0, "TILE_INDEX_STORE_HBM")
            # `name[i] = v` for an HBM tile param. operands: [idx, value].
            name = op.attrs.get("name")
            dtype = self._require_hbm_dtype_attr(op)
            slot = self.hbm_param_map.get(name)
            if slot is None:
                raise RuntimeError(
                    f"HBM tile {name!r} not in PTX param map; "
                    "only lowered HBM tile parameters are supported"
                )
            param_idx, dtype_in_map = slot
            if str(dtype) != dtype_in_map:
                raise RuntimeError(
                    f"HBM tile {name!r} dtype mismatch: op requested {dtype}, "
                    f"but kernel param is {dtype_in_map}"
                )
            idx_reg = self._require_hbm_index_reg(op, 0)
            val_reg = self._require_hbm_value_reg(
                op, 1, str(dtype), "HBM tile store value"
            )
            base = self._new_reg("rd")
            gen = self._new_reg("rd")
            off = self._new_reg("rd")
            addr = self._new_reg("rd")
            self._line(f"    ld.param.u64 {base}, [{self._kernel_param_label(param_idx)}];")
            self._line(f"    cvta.to.global.u64 {gen}, {base};")
            self._line(f"    mul.wide.s32 {off}, {idx_reg}, {self._dtype_size(dtype)};")
            self._line(f"    add.s64 {addr}, {gen}, {off};")
            self._line(f"    st.global.{self._ptx_load_suffix(dtype)} [{addr}], {val_reg};")
            return
        # Stage 64 Inc 2 — TILE_ZEROS: emit N consecutive `mov.f32
        # %fX, 0f00000000;` register-fills to zero a tile of N f32
        # elements. The result is a register-tile (no SMEM
        # allocation); subsequent ops on this tile would need to
        # know the base register + length. Phase-0 minimum-viable
        # closure of Stage 64 Inc 2 — proves the dispatch path
        # is wired without committing to a full SMEM allocation
        # strategy (Inc 3+ work).
        #
        # Attrs: dtype (f32/i32 supported in Phase-0); length
        # (number of elements; must be int). Result: 1 TileValue
        # representing the base register; downstream consumers
        # are not yet wired (Inc 3+ will add TILE_ADD / TILE_MUL).
        if op.kind == ti.TileOpKind.TILE_ZEROS:
            self._require_operand_count(op, 0, "TILE_ZEROS")
            self._require_result_count(op, 1, "TILE_ZEROS")
            dtype = op.attrs.get("dtype", "f32")
            length = op.attrs.get("length")
            if not isinstance(length, int) or length <= 0:
                raise RuntimeError(
                    f"TILE_ZEROS requires positive int 'length' attr; "
                    f"got {length!r}")
            if dtype not in ("f32", "i32"):
                raise RuntimeError(
                    f"TILE_ZEROS Phase-0: only f32 / i32 supported; "
                    f"got {dtype!r} (Inc 3+ will extend)")
            prefix = "f" if dtype == "f32" else "r"
            zero_lit = "0f00000000" if dtype == "f32" else "0"
            mov_op = "mov.b32" if dtype == "i32" else "mov.f32"
            base_reg = None
            for i in range(length):
                reg = self._new_reg(prefix)
                if base_reg is None:
                    base_reg = reg
                self._line(f"    {mov_op} {reg}, {zero_lit};")
            # Map the result TileValue to the base register so
            # downstream ops (Inc 3+) can find the tile.
            if op.results:
                self.reg_map[op.results[0].id] = base_reg
            return
        # Stage 64 Inc 3 — TILE_ADD / TILE_SUB / TILE_MUL: elementwise
        # binary ops on register-tiles. Both operands must be lowered
        # register-tiles (e.g. results of TILE_ZEROS) of the same
        # dtype + length. For length N, emit N elementwise PTX
        # instructions (`add.f32`, `sub.f32`, `mul.f32` for f32;
        # `add.s32`, `sub.s32`, `mul.lo.s32` for i32) iterating over
        # consecutive registers from each operand's base. The result
        # is allocated as N fresh consecutive registers; the result
        # TileValue maps to the new base register.
        #
        # Phase-0 / speculative-parallel-Inc-3 scope: f32 + i32 only
        # (mirrors Inc 2's TILE_ZEROS scope). Other dtypes fail
        # closed with a clear "Inc 4+ will extend" message. No SMEM
        # round-trip — purely register-resident. No user-code path
        # triggers these ops yet (that's Inc 4+ frontend work);
        # tests instantiate ops directly.
        if op.kind in (ti.TileOpKind.TILE_ADD,
                       ti.TileOpKind.TILE_SUB,
                       ti.TileOpKind.TILE_MUL):
            kind_name = op.kind.name  # "TILE_ADD" / "TILE_SUB" / "TILE_MUL"
            self._require_operand_count(op, 2, kind_name)
            self._require_result_count(op, 1, kind_name)
            lhs_val = op.operands[0]
            rhs_val = op.operands[1]
            # Both operands must be register-tiles. Read dtype +
            # length from the TileValue's TIRTileTy.
            if not isinstance(lhs_val.ty, tir.TIRTileTy):
                raise RuntimeError(
                    f"{kind_name} lhs is not a tile type")
            if not isinstance(rhs_val.ty, tir.TIRTileTy):
                raise RuntimeError(
                    f"{kind_name} rhs is not a tile type")
            lhs_dtype = lhs_val.ty.dtype.name
            rhs_dtype = rhs_val.ty.dtype.name
            if lhs_dtype != rhs_dtype:
                raise RuntimeError(
                    f"{kind_name} requires matching dtypes; "
                    f"got lhs={lhs_dtype} rhs={rhs_dtype}")
            dtype = lhs_dtype
            if dtype not in ("f32", "i32"):
                raise RuntimeError(
                    f"{kind_name} Phase-0: only f32 / i32 supported; "
                    f"got {dtype!r} (Inc 4+ will extend)")
            # Length must match between operands. Each operand is a
            # 1-D tile; shape is (DimConst(N),). Allow only
            # statically-known DimConst lengths in Phase-0.
            def _tile_length(val: ti.TileValue, role: str) -> int:
                shape = val.ty.shape
                if len(shape) != 1:
                    raise RuntimeError(
                        f"{kind_name} {role} must be 1-D tile; "
                        f"got {len(shape)}-D")
                dim = shape[0]
                if not isinstance(dim, tir.DimConst):
                    raise RuntimeError(
                        f"{kind_name} {role} requires static "
                        f"DimConst length; got {dim!r}")
                return dim.value
            lhs_len = _tile_length(lhs_val, "lhs")
            rhs_len = _tile_length(rhs_val, "rhs")
            if lhs_len != rhs_len:
                raise RuntimeError(
                    f"{kind_name} requires matching lengths; "
                    f"got lhs={lhs_len} rhs={rhs_len}")
            length = lhs_len
            if length <= 0:
                raise RuntimeError(
                    f"{kind_name} requires positive length; "
                    f"got {length}")
            # Both operands must be in reg_map (lowered already).
            lhs_base = self.reg_map.get(lhs_val.id)
            if lhs_base is None:
                raise RuntimeError(
                    f"{kind_name} lhs has no PTX register; "
                    "operand must be a lowered register-tile")
            rhs_base = self.reg_map.get(rhs_val.id)
            if rhs_base is None:
                raise RuntimeError(
                    f"{kind_name} rhs has no PTX register; "
                    "operand must be a lowered register-tile")
            # Validate register class matches dtype expectation.
            expected_prefix = "%f" if dtype == "f32" else "%r"
            self._require_reg_class(lhs_base, expected_prefix,
                                    f"{kind_name} lhs")
            self._require_reg_class(rhs_base, expected_prefix,
                                    f"{kind_name} rhs")
            # Parse the base index from each register name; operand
            # registers were allocated contiguously by the producing
            # op (e.g. Inc 2's TILE_ZEROS calls _new_reg() N times in
            # sequence), so element i lives at base + i.
            def _base_idx(reg: str, role: str) -> int:
                # reg looks like "%f5" or "%r12" — strip the prefix.
                stripped = reg.lstrip("%")
                # Trim alpha prefix ("f"/"r"/"rd"/"h"/"p").
                i = 0
                while i < len(stripped) and not stripped[i].isdigit():
                    i += 1
                try:
                    return int(stripped[i:])
                except ValueError:
                    raise RuntimeError(
                        f"{kind_name} {role} register {reg!r} has "
                        "no parseable numeric index")
            lhs_idx0 = _base_idx(lhs_base, "lhs")
            rhs_idx0 = _base_idx(rhs_base, "rhs")
            # Choose the PTX mnemonic per (kind, dtype).
            if dtype == "f32":
                mnemonic = {
                    ti.TileOpKind.TILE_ADD: "add.f32",
                    ti.TileOpKind.TILE_SUB: "sub.f32",
                    ti.TileOpKind.TILE_MUL: "mul.f32",
                }[op.kind]
                result_prefix = "f"
            else:  # i32
                mnemonic = {
                    ti.TileOpKind.TILE_ADD: "add.s32",
                    ti.TileOpKind.TILE_SUB: "sub.s32",
                    # PTX i32 mul keeps the low 32 bits.
                    ti.TileOpKind.TILE_MUL: "mul.lo.s32",
                }[op.kind]
                result_prefix = "r"
            # Allocate N contiguous result registers; emit one
            # elementwise op per index.
            result_base = None
            prefix_char = expected_prefix.lstrip("%")
            for i in range(length):
                rdst = self._new_reg(result_prefix)
                if result_base is None:
                    result_base = rdst
                lhs_reg = f"%{prefix_char}{lhs_idx0 + i}"
                rhs_reg = f"%{prefix_char}{rhs_idx0 + i}"
                self._line(
                    f"    {mnemonic} {rdst}, {lhs_reg}, {rhs_reg};")
            if op.results:
                self.reg_map[op.results[0].id] = result_base
            return
        # Stage 64 Inc 4 (Stage 106) — TILE_MATMUL via NVIDIA wmma
        # Tensor Core fragments. Canonical m16n16k16 shape:
        #   A: 16x16 of f16/bf16 -> 4 .b32 packed-pair regs per thread
        #   B: 16x16 of f16/bf16 -> 4 .b32 packed-pair regs per thread
        #   C: 16x16 of f32      -> 8 .f32 regs per thread (accumulator)
        #   D: 16x16 of f32      -> 8 .f32 regs per thread (result)
        # Emits a single `wmma.mma.sync.aligned.m16n16k16.row.col.f32.
        # {f16|bf16}.{f16|bf16}.f32 {d0..d7},{a0..a3},{b0..b3},{c0..c7};`
        # line. The operand fragment-loading lifecycle (wmma.load.a/b/c.
        # sync from SMEM) is Inc 5 / SMEM staging work; Inc 4 ships the
        # CORE matmul instruction so user-code path can validate the
        # fragment-tile shape contract.
        #
        # Phase-0 scope: m16n16k16 only (other shapes require new
        # opcode variants); A=B dtype f16 or bf16; C=D dtype f32.
        # Other dtype combinations (f32×f32 via f32 Tensor Cores,
        # tf32, int8, fp8) fail closed with a clear "Inc 5+ will
        # extend" message.
        if op.kind == ti.TileOpKind.TILE_MATMUL:
            self._require_operand_count(op, 3, "TILE_MATMUL")
            self._require_result_count(op, 1, "TILE_MATMUL")
            a_val, b_val, c_val = op.operands
            d_val = op.results[0]
            # Operand shape gate: all 4 must be 1-D register-tiles
            # with statically-known DimConst lengths matching the
            # m16n16k16 packed-fragment layout.
            def _frag_len(val: ti.TileValue, role: str,
                          expected: int) -> int:
                if not isinstance(val.ty, tir.TIRTileTy):
                    raise RuntimeError(
                        f"TILE_MATMUL {role} is not a tile type")
                shape = val.ty.shape
                if len(shape) != 1:
                    raise RuntimeError(
                        f"TILE_MATMUL {role} must be 1-D fragment "
                        f"tile; got {len(shape)}-D")
                dim = shape[0]
                if not isinstance(dim, tir.DimConst):
                    raise RuntimeError(
                        f"TILE_MATMUL {role} requires static "
                        f"DimConst length; got {dim!r}")
                if dim.value != expected:
                    raise RuntimeError(
                        f"TILE_MATMUL {role} must have length "
                        f"{expected} for canonical m16n16k16 "
                        f"fragment; got {dim.value}")
                return dim.value
            _frag_len(a_val, "A", 4)
            _frag_len(b_val, "B", 4)
            _frag_len(c_val, "C", 8)
            _frag_len(d_val, "D", 8)
            # Dtype gate.
            ab_dtype = a_val.ty.dtype.name
            if ab_dtype != b_val.ty.dtype.name:
                raise RuntimeError(
                    f"TILE_MATMUL A/B dtypes must match; got "
                    f"A={ab_dtype} B={b_val.ty.dtype.name}")
            if ab_dtype not in ("f16", "bf16"):
                raise RuntimeError(
                    f"TILE_MATMUL Phase-0: A/B dtype must be f16 or "
                    f"bf16; got {ab_dtype!r} (Inc 5+ will add f32/"
                    f"tf32/int8/fp8 Tensor Core variants)")
            if c_val.ty.dtype.name != "f32":
                raise RuntimeError(
                    f"TILE_MATMUL Phase-0: C accumulator dtype must "
                    f"be f32; got {c_val.ty.dtype.name!r}")
            if d_val.ty.dtype.name != "f32":
                raise RuntimeError(
                    f"TILE_MATMUL Phase-0: D result dtype must be "
                    f"f32; got {d_val.ty.dtype.name!r}")
            # Resolve base registers for A/B/C. A and B fragments
            # are .b32 packed (each holds 2 × f16 / bf16); use the
            # `%r` pool for them since .b32 == 32-bit untyped.
            # C is .f32, so use the `%f` pool.
            a_base = self.reg_map.get(a_val.id)
            b_base = self.reg_map.get(b_val.id)
            c_base = self.reg_map.get(c_val.id)
            if a_base is None:
                raise RuntimeError(
                    "TILE_MATMUL A has no PTX register; operand "
                    "must be a lowered register-tile (consider "
                    "wmma.load.a.sync to populate)")
            if b_base is None:
                raise RuntimeError(
                    "TILE_MATMUL B has no PTX register; operand "
                    "must be a lowered register-tile (consider "
                    "wmma.load.b.sync to populate)")
            if c_base is None:
                raise RuntimeError(
                    "TILE_MATMUL C has no PTX register; operand "
                    "must be a lowered register-tile (consider "
                    "wmma.load.c.sync to populate)")
            # Parse fragment base indices.
            def _base_idx_for_matmul(reg: str, role: str) -> int:
                stripped = reg.lstrip("%")
                i = 0
                while i < len(stripped) and not stripped[i].isdigit():
                    i += 1
                try:
                    return int(stripped[i:])
                except ValueError:
                    raise RuntimeError(
                        f"TILE_MATMUL {role} register {reg!r} has "
                        f"no parseable numeric index")
            a_idx0 = _base_idx_for_matmul(a_base, "A")
            b_idx0 = _base_idx_for_matmul(b_base, "B")
            c_idx0 = _base_idx_for_matmul(c_base, "C")
            # Validate register-class assignments. A/B are .b32
            # packed (treat as %r); C is .f32 (%f). Diagnose pool
            # mismatches up-front rather than letting ptxas reject.
            self._require_reg_class(a_base, "%r", "TILE_MATMUL A")
            self._require_reg_class(b_base, "%r", "TILE_MATMUL B")
            self._require_reg_class(c_base, "%f", "TILE_MATMUL C")
            # Allocate D as 8 fresh contiguous %f registers.
            d_base = None
            for i in range(8):
                rd = self._new_reg("f")
                if d_base is None:
                    d_base = rd
            d_idx0 = _base_idx_for_matmul(d_base, "D")
            # Build the operand register lists.
            a_regs = ",".join(f"%r{a_idx0 + i}" for i in range(4))
            b_regs = ",".join(f"%r{b_idx0 + i}" for i in range(4))
            c_regs = ",".join(f"%f{c_idx0 + i}" for i in range(8))
            d_regs = ",".join(f"%f{d_idx0 + i}" for i in range(8))
            self._line(
                f"    wmma.mma.sync.aligned.m16n16k16.row.col."
                f"f32.{ab_dtype}.{ab_dtype}.f32 "
                f"{{{d_regs}}}, {{{a_regs}}}, {{{b_regs}}}, "
                f"{{{c_regs}}};")
            self.reg_map[d_val.id] = d_base
            return
        # v2.2 polish item 1 R1 audit-fix CRIT-1: parity exhaustiveness
        # guard. If we fell through and the table claims this kind is
        # "supported", that's drift between the canonical table and
        # the dispatcher — caller will see a misleading RuntimeError
        # ("unsupported") for something the table swore was supported.
        # Raise AssertionError instead so the drift surfaces as a
        # framework-internal bug, not a user-facing capability claim.
        # Matches rocm/metal/webgpu's phantom-supported defense.
        declared = PTX_OP_LOWERING.get(op.kind, {}).get("status")
        if declared == "supported":
            raise AssertionError(
                f"PTX_OP_LOWERING declares {op.kind.name} status="
                f"'supported' but PtxEmitter.emit_op has no matching "
                f"branch — table/dispatcher drift. Either add an emit "
                f"branch or demote the table entry to status='stub'."
            )
        raise RuntimeError(
            f"unsupported PTX op {op.kind.value}; "
            "add lowering before emitting PTX"
        )

    # Stage 16 — helpers for HBM addressing.
    # Audit 28.8 cycle 21 C20-1 (HIGH): include isize/usize as 8-byte
    # entries to match the canonical 64-bit treatment established by
    # typecheck.py / x86_64 backend / const_fold. Pre-fix the `.get`
    # default at line 343 silently fell back to 4 bytes, producing
    # wrong stride for tensor<isize, ...> HBM tile elements.
    # Stage 28.9 cycle 35 audit-R (conf 82): include "bool" as a
    # 1-byte / u8 entry. `_ptx_type_str` already maps bool→.pred
    # (line 170) but these addressing tables omitted bool, so a
    # hypothetical tile<bool, ...> HBM op would silently fall back
    # to 4 bytes / u32 — same defect class as the isize/usize gap.
    # Phase-0 doesn't exercise tile<bool> today; this is a latent-
    # silent-narrowing hardening matching the cycle 21 C20-1 fix.
    _DTYPE_SIZE = {"i8": 1, "u8": 1, "i16": 2, "u16": 2, "f16": 2, "bf16": 2,
                    "i32": 4, "u32": 4, "f32": 4, "i64": 8, "u64": 8, "f64": 8,
                    "isize": 8, "usize": 8, "bool": 1}
    _DTYPE_PTX_LOAD = {"i8": "s8", "u8": "u8", "i16": "s16", "u16": "u16",
                        "f16": "f16", "bf16": "bf16",
                        "i32": "s32", "u32": "u32", "f32": "f32",
                        "i64": "s64", "u64": "u64", "f64": "f64",
                        "isize": "s64", "usize": "u64", "bool": "u8"}

    def _dtype_size(self, dtype: str) -> int:
        # Cycle 1 Batch BE silent-failure HIGH-2 + type-design HIGH-3 fix:
        # pre-fix, `dict.get(dtype, 4)` silently defaulted unknown dtypes
        # to 4 bytes — producing wrong PTX stride for any future dtype
        # (fp8/int4/etc.) added without updating the table. Cycle 21
        # C20-1 partially fixed this by adding missing entries
        # (isize/usize/bool) but preserved the defective default shape.
        # Post-fix: KeyError → loud RuntimeError naming the missing dtype.
        try:
            return self._DTYPE_SIZE[dtype]
        except KeyError:
            raise RuntimeError(
                f"PTX backend: unsupported dtype {dtype!r} for stride "
                f"computation; add to _DTYPE_SIZE + _DTYPE_PTX_LOAD + "
                f"_ld_reg_prefix together (Cycle 1 Batch BE silent-"
                f"failure HIGH-2 fix)"
            )

    def _ptx_load_suffix(self, dtype: str) -> str:
        # Cycle 1 Batch BE silent-failure HIGH-2 fix: same pattern as
        # _dtype_size — pre-fix `.get(dtype, "u32")` silently defaulted
        # to u32 load suffix for unknown dtypes, picking the wrong PTX
        # register class. Post-fix: loud raise.
        try:
            return self._DTYPE_PTX_LOAD[dtype]
        except KeyError:
            raise RuntimeError(
                f"PTX backend: unsupported dtype {dtype!r} for PTX "
                f"load/store suffix; add to _DTYPE_PTX_LOAD + "
                f"_DTYPE_SIZE + _ld_reg_prefix together (Cycle 1 "
                f"Batch BE silent-failure HIGH-2 fix)"
            )

    def _ld_reg_prefix(self, dtype: str) -> str:
        # Pick a sensible register pool by dtype family.
        # Audit 28.8 cycle 21 C20-1: include isize/usize in the
        # 64-bit register pool ('rd') alongside i64/u64.
        # Stage 64 Inc 1 — Tier 2 #6: split 16-bit floats (f16/bf16)
        # to the `%h` pool (.b16 register class in PTX); 32/64-bit
        # floats keep `%f`. Pre-Stage-64 conflated all float widths.
        # Cycle 1 Batch BE silent-failure MEDIUM-5 fix: replace the
        # bare `return "r"` fallback with a raise so unknown dtypes
        # surface loudly rather than silently routing 64-bit values
        # into the 32-bit register pool.
        if dtype in ("f16", "bf16"):
            return "h"
        if dtype in ("f32", "f64"):
            return "f"
        if dtype in ("i64", "u64", "isize", "usize"):
            return "rd"
        if dtype in ("i8", "u8", "i16", "u16", "i32", "u32", "bool", "char"):
            return "r"
        raise RuntimeError(
            f"PTX backend: unsupported dtype {dtype!r} for register-pool "
            f"selection; add to _ld_reg_prefix (Cycle 1 Batch BE silent-"
            f"failure MEDIUM-5 fix)"
        )

    def _kernel_param_label(self, param_idx: int) -> str:
        # Phase-0 uses the same labels as `_format_param`. Centralized so
        # the convention is easy to change later.
        return f"param_{param_idx}"


def emit_ptx(tile_module: ti.TileModule, target: str = DEFAULT_TARGET) -> str:
    return PtxEmitter(target).emit_module(tile_module)


def kernel_only_module(module: tir.Module) -> tir.Module:
    """Return a TIR module containing only PTX-emitted kernel functions."""
    return type(module)(
        functions={
            name: fn for name, fn in module.functions.items()
            if fn.attrs.get("kernel")
        },
        next_value_id=module.next_value_id,
        next_block_id=module.next_block_id,
    )


def validate_kernel_tile_lowering(module: tir.Module) -> None:
    """Fail before host DCE can hide unsupported operations inside kernels."""
    kernel_mod = kernel_only_module(module)
    if not kernel_mod.functions:
        return
    tile_mod = ti.lower_to_tile(kernel_mod)
    ptx = emit_ptx(tile_mod)
    # v2.3 5-clean-gate BE HIGH-1 changed emit_op's stub/deferred/skipped
    # handling from a raised RuntimeError to a `.error "HELIX-..."`
    # directive written into the PTX text. This validator's contract is
    # to RAISE on an unsupported kernel op — and the x86_64 host path
    # embeds kernel PTX without ever running ptxas, so the directive
    # alone would be silent. Detect it here (parity with the PTX CLI's
    # own `.error "HELIX-` check) so the rejection stays loud.
    if '.error "HELIX-' in ptx:
        raise RuntimeError(
            "emitted kernel contains a HELIX-STUB / HELIX-SKIPPED "
            "directive — an unsupported tile-IR op (e.g. a non-inlined "
            "helper CALL inside a @kernel) reached PTX codegen; the "
            "kernel is non-functional and ptxas would reject it"
        )
    if getattr(module, "_helix_kernel_tile_validation_blocked_by_dce", False):
        raise RuntimeError(
            "kernel tile validation must run before DCE/FDCE"
        )
    setattr(module, "_helix_kernel_tile_validated", True)


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import atexit
    import dataclasses
    import sys
    from ..frontend import ast_nodes as A
    from ..frontend.autodiff import take_diff_warnings
    from ..frontend.lexer import LexError
    from ..frontend.parser import parse, ParseError, STDLIB_STRICT_ENV
    from ..frontend.typecheck import typecheck
    from ..frontend.flatten_modules import flatten_modules, FlattenError
    from ..frontend.flatten_impls import flatten_impls, DuplicateMethodError
    from ..frontend.struct_mono import monomorphize_structs
    from ..frontend.monomorphize import monomorphize_safe
    from ..frontend.trace_pass import validate_trace_attrs
    from ..frontend.panic_pass import validate_panic_args, validate_unwind
    from ..frontend.unsafe_pass import check_unsafe_ops
    from ..frontend.autotune import validate_autotune_prog
    from ..frontend.grad_pass import grad_pass
    from ..frontend.totality import check_totality
    from ..frontend.deprecated_pass import emit_warnings as emit_deprecated_warnings
    from ..ir.lower_ast import lower
    from ..ir.passes.const_fold import fold_module
    from ..ir.passes.cse import cse_module
    from ..ir.passes.fdce import diagnostic_function_names
    from ..ir.passes.effect_check import (
        check_module as effect_check_module,
        report_diagnostics as report_effect_diagnostics,
    )
    from ..ir.tile_ir import lower_to_tile

    ad_policy = "warn"
    take_diff_warnings()

    def _drain_cli_ad_warnings() -> int:
        ad_warnings = take_diff_warnings()
        if not ad_warnings:
            return 0
        label = "ERROR" if ad_policy == "error" else "warning"
        print(f"   ad:        {len(ad_warnings)} {label}(s)", file=sys.stderr)
        for warning in ad_warnings:
            print(f"     helixc: {warning}", file=sys.stderr)
        if ad_policy == "error":
            return 1
        return 0

    atexit.register(_drain_cli_ad_warnings)

    def _called_fn_names(value: object) -> set[str]:
        names: set[str] = set()
        seen: set[int] = set()

        def visit(node: object) -> None:
            if node is None or isinstance(node, (str, int, float, bool)):
                return
            if isinstance(node, (list, tuple)):
                for item in node:
                    visit(item)
                return
            oid = id(node)
            if oid in seen:
                return
            seen.add(oid)
            if isinstance(node, A.Call) and isinstance(node.callee, A.Name):
                names.add(node.callee.name)
            if dataclasses.is_dataclass(node):
                for field in dataclasses.fields(node):
                    visit(getattr(node, field.name))

        visit(value)
        return names

    def _kernel_reachable_program(prog: A.Program) -> A.Program:
        fn_by_name = {
            it.name: it for it in prog.items if isinstance(it, A.FnDecl)
        }
        keep: set[str] = {
            it.name for it in prog.items
            if isinstance(it, A.FnDecl) and "kernel" in it.attrs
        }
        queue = list(keep)
        while queue:
            fn = fn_by_name.get(queue.pop())
            if fn is None:
                continue
            for callee in _called_fn_names(fn.body):
                if callee in fn_by_name and callee not in keep:
                    keep.add(callee)
                    queue.append(callee)
        return A.Program(
            module=prog.module,
            items=[
                it for it in prog.items
                if not isinstance(it, A.FnDecl) or it.name in keep
            ],
        )

    def _ty_mentions_diff(ty: object) -> bool:
        if ty is None:
            return False
        if isinstance(ty, A.TyGeneric) and ty.base == "D":
            return True
        if dataclasses.is_dataclass(ty):
            for field in dataclasses.fields(ty):
                if _ty_mentions_diff(getattr(ty, field.name)):
                    return True
        if isinstance(ty, (list, tuple)):
            return any(_ty_mentions_diff(item) for item in ty)
        return False

    def _fn_mentions_diff_signature(fn: A.FnDecl) -> bool:
        if _ty_mentions_diff(fn.return_ty):
            return True
        return any(_ty_mentions_diff(p.ty) for p in fn.params)

    def _reachable_function_names(prog: A.Program) -> set[str]:
        fn_by_name = {
            it.name: it for it in prog.items if isinstance(it, A.FnDecl)
        }
        keep: set[str] = {
            it.name for it in prog.items
            if isinstance(it, A.FnDecl)
            and (it.name == "main" or "kernel" in it.attrs)
        }
        queue = list(keep)
        while queue:
            fn = fn_by_name.get(queue.pop())
            if fn is None:
                continue
            for callee in _called_fn_names(fn.body):
                if callee in fn_by_name and callee not in keep:
                    keep.add(callee)
                    queue.append(callee)
        return keep

    def _drop_unreachable_diff_signature_fns(prog: A.Program) -> A.Program:
        reachable = _reachable_function_names(prog)
        return A.Program(
            module=prog.module,
            items=[
                it for it in prog.items
                if (not isinstance(it, A.FnDecl)
                    or not _fn_mentions_diff_signature(it)
                    or it.name in reachable)
            ],
        )

    cli_args = sys.argv[1:]
    # Restart 49 B2 + B3: -h/--help prints a real banner to stdout and exits
    # 0. The banner enumerates all currently-accepted flags including the
    # restart-46 additions (-O0/-O1/-O2/-O3, --no-opt) and the restart-47
    # additions (-l, --no-color/--color, --hash/--hash-cons). Bare-invocation
    # also prints the banner (was missing pre-restart-49).
    _ptx_usage_banner = (
        "usage: python -m helixc.backend.ptx <input.hx> "
        "[--strict] [--no-opt] [-O0|-O1|-O2|-O3] "
        "[--stdlib] [--no-stdlib] "
        "[-Wad=warn|error] [-Wdeprecated=warn|error] "
        "[-l <libname>] [-l<libname>] "
        "[--no-color] [--color] "
        "[--hash] [--hash-cons]"
    )
    if "-h" in cli_args or "--help" in cli_args:
        print(_ptx_usage_banner)
        sys.exit(0)
    if not cli_args:
        print(_ptx_usage_banner, file=sys.stderr)
        sys.exit(2)
    # Restart 46 B2: accept --no-opt and -O0/-O1/-O2/-O3 for flag parity
    # with helixc.check and helixc.backend.x86_64. The PTX text emitter
    # currently always runs fold + cse with no per-level staging, so these
    # flags are accepted as a no-op; the goal is to close the parity gap
    # so users can pass the same flags they pass to helixc.check.
    # Restart 47 B4: extended to --no-color, --color, --hash, --hash-cons
    # plus -l/-l<name> for symmetric parity.
    allowed_flags = {
        "--strict", "--stdlib", "--no-stdlib",
        "--no-opt", "-O0", "-O1", "-O2", "-O3",
        "--no-color", "--color", "--hash", "--hash-cons",
    }
    warning_policies: dict[str, str] = {}
    known_warning_names = {"ad", "deprecated"}
    flags: set[str] = set()
    paths: list[str] = []
    unknown_flags: list[str] = []
    _iter = iter(cli_args)
    for flag in _iter:
        if flag in allowed_flags:
            flags.add(flag)
            continue
        if flag.startswith("-W"):
            body = flag[2:]
            if "=" in body:
                name, val = body.split("=", 1)
                if name not in known_warning_names:
                    unknown_flags.append(flag)
                    continue
                if val not in ("warn", "error"):
                    unknown_flags.append(flag)
                    continue
                warning_policies[name] = val
            else:
                if body not in known_warning_names:
                    unknown_flags.append(flag)
                    continue
                warning_policies[body] = "warn"
            continue
        if flag == "-l":
            # Restart 47 B4: consume the following library-name argument
            # (no-op for PTX which does not link host libraries).
            try:
                next(_iter)
            except StopIteration:
                print("error: ptx: -l requires a library name", file=sys.stderr)
                sys.exit(2)
            continue
        if flag.startswith("-l") and len(flag) > 2:
            # Joined form: -lm (no-op).
            continue
        if flag.startswith("-"):
            unknown_flags.append(flag)
        else:
            paths.append(flag)
    if unknown_flags:
        for flag in unknown_flags:
            print(f"error: ptx: unknown flag {flag}", file=sys.stderr)
        sys.exit(2)
    if "--stdlib" in flags and "--no-stdlib" in flags:
        print(
            "error: ptx: conflicting stdlib flags: choose --stdlib or --no-stdlib",
            file=sys.stderr,
        )
        sys.exit(2)
    strict = "--strict" in flags
    ad_policy = warning_policies.get("ad", "warn")
    include_stdlib = "--no-stdlib" not in flags
    if not paths:
        print("error: ptx: missing input path", file=sys.stderr)
        sys.exit(2)
    if len(paths) > 1:
        print("error: ptx: expected at most one input path", file=sys.stderr)
        sys.exit(2)
    filename = paths[0]
    try:
        with open(paths[0], encoding="utf-8") as f:
            src = f.read()
    except OSError as e:
        print(f"error: ptx: cannot read {paths[0]}: {e}", file=sys.stderr)
        sys.exit(2)
    except UnicodeDecodeError as e:
        print(f"error: ptx: encoding error reading source: {e}", file=sys.stderr)
        sys.exit(2)
    try:
        prog = parse(src, filename=filename, include_stdlib=include_stdlib)
    except LexError as e:
        print("LEX ERROR:", file=sys.stderr)
        print(f"  {filename}:{e}", file=sys.stderr)
        sys.exit(1)
    except ParseError as e:
        print("PARSE ERROR:", file=sys.stderr)
        rendered = e.render(source=src, filename=filename, color=False)
        for line in rendered.splitlines():
            print(f"  {line}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        msg = str(e)
        if not msg:
            msg = f"stdlib file missing with {STDLIB_STRICT_ENV}=1"
        print(f"error: ptx: {msg}", file=sys.stderr)
        sys.exit(2)
    try:
        flatten_modules(prog)
        flatten_impls(prog)
        prog, struct_diags = monomorphize_structs(prog)
        if struct_diags:
            for e in struct_diags:
                print(f"error: struct-mono: {e}", file=sys.stderr)
            sys.exit(1)
        _mono_count, mono_diags = monomorphize_safe(prog)
        if mono_diags:
            for e in mono_diags:
                print(f"error: fn-mono: {e}", file=sys.stderr)
            sys.exit(1)
    except FlattenError as e:
        print(f"error: mod-flatten: {e}", file=sys.stderr)
        sys.exit(1)
    except DuplicateMethodError as e:
        print(f"error: impl-flatten: {e}", file=sys.stderr)
        sys.exit(1)
    errs = typecheck(prog)
    if errs:
        for e in errs:
            print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    totality_fails = check_totality(prog)
    if totality_fails:
        print(
            f"warning: totality: {len(totality_fails)} fn(s) NOT proven total",
            file=sys.stderr,
        )
        for name, reason in totality_fails:
            print(
                f"warning: [trap 21001] totality: {name}: {reason}",
                file=sys.stderr,
            )
        if strict:
            print(
                f"\n{len(totality_fails)} totality warning(s); --strict aborts.",
                file=sys.stderr,
            )
            sys.exit(1)
    deprecated_policy = warning_policies.get("deprecated", "warn")
    deprecated_warnings = emit_deprecated_warnings(prog)
    if deprecated_warnings:
        label = "ERROR" if deprecated_policy == "error" else "warning"
        print(
            f"   deprecated: {len(deprecated_warnings)} {label}(s)",
            file=sys.stderr,
        )
        for warning in deprecated_warnings:
            print(f"     {warning}", file=sys.stderr)
        if deprecated_policy == "error":
            sys.exit(1)
    validation_groups = [
        ("trace", validate_trace_attrs(prog)),
        ("panic", validate_panic_args(prog)),
        ("unwind", validate_unwind(prog)),
        ("unsafe", check_unsafe_ops(prog)),
        ("autotune", validate_autotune_prog(prog)),
    ]
    had_validation_error = False
    for phase, diags in validation_groups:
        for e in diags:
            print(f"error: {phase}: {e}", file=sys.stderr)
            had_validation_error = True
    if had_validation_error:
        sys.exit(1)
    try:
        full_eff = []
        try:
            full_prog = _drop_unreachable_diff_signature_fns(prog)
            grad_pass(full_prog)
            full_mod = lower(full_prog)
            full_scope = None
            if include_stdlib:
                full_scope = diagnostic_function_names(full_mod)
            full_eff = effect_check_module(
                full_mod, only_functions=full_scope)
        # Restart 48 B2: do NOT swallow loud-fail signals. Re-raise the
        # loud-fail discipline set (NotImplementedError, AssertionError)
        # plus user-interrupt classes; only convert genuine domain errors
        # into one-line diagnostics. Mirrors restart 47 B1's narrowing of
        # lower_ast._resolve_monomorphized_struct_type.
        except (NotImplementedError, AssertionError, KeyboardInterrupt,
                SystemExit, MemoryError):
            raise
        except Exception as e:
            print(f"error: ptx: validation: {e}", file=sys.stderr)
            sys.exit(1)
        kernel_prog = _kernel_reachable_program(prog)
        # Stage 56 / Tier 2 #8 — expand @autotune @kernel fns into
        # N specialized variants (one per cross-product config) BEFORE
        # grad_pass + lower. Each variant is a deep-copied FnDecl with
        # its body's Name(KEY) refs replaced by IntLit(VAL). The
        # original `@autotune` fn is replaced by its expansion.
        # Pre-Stage-56, only one un-specialized variant was emitted.
        from ..frontend.autotune_expand import expand_autotune_kernels
        kernel_prog = expand_autotune_kernels(kernel_prog)
        grad_pass(kernel_prog)
        tir_mod = lower(kernel_prog)
        pre_effect_scope = None
        if include_stdlib:
            pre_effect_scope = diagnostic_function_names(tir_mod)
        pre_eff = effect_check_module(
            tir_mod, only_functions=pre_effect_scope)
        fold_module(tir_mod)
        cse_module(tir_mod)
        post_effect_scope = None
        if include_stdlib:
            post_effect_scope = diagnostic_function_names(tir_mod)
        post_eff = effect_check_module(
            tir_mod, only_functions=post_effect_scope)
        eff_errs = list(full_eff)
        for err in pre_eff:
            if err not in eff_errs:
                eff_errs.append(err)
        seen_eff_errs = set(eff_errs)
        for err in post_eff:
            if err not in seen_eff_errs:
                eff_errs.append(err)
                seen_eff_errs.add(err)
        hard_count = report_effect_diagnostics(eff_errs, stderr=sys.stderr)
        if hard_count > 0 and strict:
            print(
                f"\n{hard_count} effect-check warning(s); --strict aborts.",
                file=sys.stderr,
            )
            sys.exit(1)
        kernel_mod = kernel_only_module(tir_mod)
        tile_mod = lower_to_tile(kernel_mod)
        ptx = emit_ptx(tile_mod)
        ad_rc = _drain_cli_ad_warnings()
        if ad_rc != 0:
            sys.exit(ad_rc)
        print(ptx)
        # v2.3 5-clean-gate BE HIGH-1 audit-fix: emit_op's loud-stub
        # forward guard writes a `.error "HELIX-STUB..."` / `.error
        # "HELIX-SKIPPED..."` directive into the PTX text when a
        # stub/deferred/skipped op reaches codegen (e.g. a CALL in a
        # kernel that upstream inlining did not eliminate). `.error`
        # makes ptxas abort — but a caller checking *helixc's own*
        # exit code (not piping to ptxas) would otherwise see 0.
        # Surface the non-functional-kernel signal at the CLI exit
        # code too. The PTX text is already printed above so the
        # directive + the offending op are visible to the user.
        if '.error "HELIX-' in ptx:
            print(
                "error: ptx: emitted kernel contains a HELIX-STUB / "
                "HELIX-SKIPPED directive — an unsupported tile-IR op "
                "reached PTX codegen; the kernel is non-functional "
                "and ptxas would reject it.",
                file=sys.stderr,
            )
            sys.exit(1)
    # Restart 48 B2: preserve loud-fail discipline at the outermost handler
    # too; only convert domain errors into one-line diagnostics.
    except (NotImplementedError, AssertionError, KeyboardInterrupt,
            SystemExit, MemoryError):
        raise
    except Exception as e:
        print(f"error: ptx: {e}", file=sys.stderr)
        sys.exit(1)
