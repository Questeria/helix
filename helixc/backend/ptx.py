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
from typing import Optional

from ..ir import tir, tile_ir as ti


# ============================================================================
# Configuration
# ============================================================================
PTX_VERSION = "8.3"           # corresponds to CUDA 12.3 baseline
DEFAULT_TARGET = "sm_75"      # Turing+ (RTX 3070 is sm_86; RTX 5090 is sm_120)
ADDRESS_SIZE = 64


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
            }
            return mapping.get(ty.name, ".b32")
        if isinstance(ty, tir.TIRUnit):
            return ""
        return ".b64"

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
        if dtype not in {"f32", "i32"}:
            raise RuntimeError(
                f"unsupported PTX HBM tile dtype {dtype}; "
                "only f32 and i32 HBM tile elements are currently lowered"
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
        value = op.operands[operand_index]
        expected = {"f32": ("f32", "%f"),
                    "i32": ("i32", "%r")}[dtype]
        self._require_scalar_type(value, role, {expected[0]})
        reg = self._require_reg(op, operand_index, role)
        self._require_reg_class(reg, expected[1], role)
        return reg

    def _require_hbm_load_result_type(self, op: ti.TileOp, dtype: str) -> None:
        if not op.results:
            return
        expected = {"f32": "f32", "i32": "i32"}[dtype]
        self._require_scalar_type(op.results[0], "HBM tile load result", {expected})

    # ---- ops ----
    def emit_op(self, op: ti.TileOp) -> None:
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
        return self._DTYPE_SIZE.get(dtype, 4)

    def _ptx_load_suffix(self, dtype: str) -> str:
        return self._DTYPE_PTX_LOAD.get(dtype, "u32")

    def _ld_reg_prefix(self, dtype: str) -> str:
        # Pick a sensible register pool by dtype family.
        # Audit 28.8 cycle 21 C20-1: include isize/usize in the
        # 64-bit register pool ('rd') alongside i64/u64.
        if dtype in ("f16", "bf16", "f32", "f64"):
            return "f"
        if dtype in ("i64", "u64", "isize", "usize"):
            return "rd"
        return "r"

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
    emit_ptx(tile_mod)
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
    if not cli_args:
        print("error: ptx: missing input path", file=sys.stderr)
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
        except Exception as e:
            print(f"error: ptx: validation: {e}", file=sys.stderr)
            sys.exit(1)
        kernel_prog = _kernel_reachable_program(prog)
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
    except Exception as e:
        print(f"error: ptx: {e}", file=sys.stderr)
        sys.exit(1)
