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
        for fn in mod.functions.values():
            if fn.attrs.get("kernel"):
                self.emit_kernel(fn)
            else:
                # Non-kernel functions: emit as .func (device-only) for v0.1
                self.emit_device_func(fn)
        return self.buf.getvalue()

    # ---- kernel ----
    def emit_kernel(self, fn: ti.TileFn) -> None:
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
        # Minimal stub: empty .func
        ret_str = self._ptx_type_str(fn.return_ty)
        ret_decl = f".func ({ret_str} %retval)" if ret_str else ".func"
        params_str = ", ".join(self._format_param(p, i) for i, p in enumerate(fn.params))
        self._line(f"{ret_decl} {fn.name}({params_str})")
        self._line("{")
        self._line("    ret;")
        self._line("}")
        self._line()

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

    # ---- ops ----
    def emit_op(self, op: ti.TileOp) -> None:
        # v0.1: only handle a tiny scalar subset for sanity testing
        if op.kind == ti.TileOpKind.SCALAR_CONST_INT:
            r = self._new_reg("r")
            v = op.attrs.get("value", 0)
            self._line(f"    mov.b32 {r}, {int(v)};")
            if op.results:
                self.reg_map[op.results[0].id] = r
            return
        if op.kind == ti.TileOpKind.SCALAR_ADD:
            # Stage 16 — float and int operand dispatch. If either operand
            # is in a %f register (TILE_INDEX_LOAD_HBM result), emit
            # `add.f32`; otherwise the integer fallback.
            a_reg = self.reg_map.get(op.operands[0].id, "%r0")
            b_reg = self.reg_map.get(op.operands[1].id, "%r1")
            if a_reg.startswith("%f") or b_reg.startswith("%f"):
                r = self._new_reg("f")
                self._line(f"    add.f32 {r}, {a_reg}, {b_reg};")
            else:
                r = self._new_reg("r")
                self._line(f"    add.s32 {r}, {a_reg}, {b_reg};")
            if op.results:
                self.reg_map[op.results[0].id] = r
            return
        if op.kind == ti.TileOpKind.SCALAR_MUL:
            a_reg = self.reg_map.get(op.operands[0].id, "%r0")
            b_reg = self.reg_map.get(op.operands[1].id, "%r1")
            if a_reg.startswith("%f") or b_reg.startswith("%f"):
                r = self._new_reg("f")
                self._line(f"    mul.f32 {r}, {a_reg}, {b_reg};")
            else:
                r = self._new_reg("r")
                self._line(f"    mul.lo.s32 {r}, {a_reg}, {b_reg};")
            if op.results:
                self.reg_map[op.results[0].id] = r
            return
        if op.kind == ti.TileOpKind.SCALAR_SUB:
            a_reg = self.reg_map.get(op.operands[0].id, "%r0")
            b_reg = self.reg_map.get(op.operands[1].id, "%r1")
            if a_reg.startswith("%f") or b_reg.startswith("%f"):
                r = self._new_reg("f")
                self._line(f"    sub.f32 {r}, {a_reg}, {b_reg};")
            else:
                r = self._new_reg("r")
                self._line(f"    sub.s32 {r}, {a_reg}, {b_reg};")
            if op.results:
                self.reg_map[op.results[0].id] = r
            return
        if op.kind == ti.TileOpKind.SCALAR_NEG:
            a_reg = self.reg_map.get(op.operands[0].id, "%r0")
            if a_reg.startswith("%f"):
                r = self._new_reg("f")
                self._line(f"    neg.f32 {r}, {a_reg};")
            else:
                r = self._new_reg("r")
                self._line(f"    neg.s32 {r}, {a_reg};")
            if op.results:
                self.reg_map[op.results[0].id] = r
            return
        if op.kind == ti.TileOpKind.SCALAR_CONST_FLOAT:
            # f32 constant. Emit via mov.f32 with the hex bit pattern so
            # ptxas accepts the exact bits unambiguously.
            import struct
            v = float(op.attrs.get("value", 0.0))
            bits = struct.unpack("<I", struct.pack("<f", v))[0]
            r = self._new_reg("f")
            self._line(f"    mov.f32 {r}, 0f{bits:08X};")
            if op.results:
                self.reg_map[op.results[0].id] = r
            return
        if op.kind == ti.TileOpKind.SCALAR_CMP:
            cmp_op = op.attrs.get("cmp", "cmp.eq")
            cmp_map = {"cmp.eq": "eq", "cmp.ne": "ne", "cmp.lt": "lt",
                        "cmp.le": "le", "cmp.gt": "gt", "cmp.ge": "ge"}
            cmp_suffix = cmp_map.get(cmp_op, "eq")
            a_reg = self.reg_map.get(op.operands[0].id, "%r0")
            b_reg = self.reg_map.get(op.operands[1].id, "%r1")
            # Result lives in a %p predicate register. Phase-0 emits the
            # signed-int form; float compares would need setp.<cmp>.f32.
            r = self._new_reg("p")
            self._line(f"    setp.{cmp_suffix}.s32 {r}, {a_reg}, {b_reg};")
            if op.results:
                self.reg_map[op.results[0].id] = r
            return
        if op.kind == ti.TileOpKind.RETURN:
            return
        # Stage 16 — GPU primitives.
        if op.kind == ti.TileOpKind.THREAD_IDX:
            # `thread_idx()` returns the i32 special-reg value. attrs={dim,sreg}
            # where dim is "x"/"y"/"z" and sreg is "tid" (default), "ctaid"
            # (block index), or "ntid" (block dim). Maps to PTX `mov.u32
            # %r, %<sreg>.<dim>`.
            dim = op.attrs.get("dim", "x")
            sreg = op.attrs.get("sreg", "tid")
            r = self._new_reg("r")
            self._line(f"    mov.u32 {r}, %{sreg}.{dim};")
            if op.results:
                self.reg_map[op.results[0].id] = r
            return
        if op.kind == ti.TileOpKind.TILE_INDEX_LOAD_HBM:
            # `name[i]` for an HBM tile param. Sequence:
            #   ld.param.u64  %rdN, [<kernel>_param_<idx>]   (kernel-arg ptr)
            #   cvta.to.global.u64 %rdM, %rdN                (generic -> global)
            #   mul.wide.s32  %rdK, <idx>, sizeof(dtype)     (byte offset)
            #   add.s64       %rdL, %rdM, %rdK
            #   ld.global.<dtype> %fN, [%rdL]
            # Phase-0 collapses some of these: we just emit the textbook
            # PTX sequence for clarity; ptxas will optimize.
            name = op.attrs.get("name")
            dtype = op.attrs.get("dtype", "f32")
            slot = self.hbm_param_map.get(name)
            if slot is None:
                # Trap-id 97001: HBM tile name not found (lowering bug).
                self._line(f"    // ERROR trap 97001: HBM tile {name!r} not in param map")
                return
            param_idx, _dtype_in_map = slot
            idx_reg = self.reg_map.get(op.operands[0].id, "%r0")
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
            # `name[i] = v` for an HBM tile param. operands: [idx, value].
            name = op.attrs.get("name")
            dtype = op.attrs.get("dtype", "f32")
            slot = self.hbm_param_map.get(name)
            if slot is None:
                self._line(f"    // ERROR trap 97001: HBM tile {name!r} not in param map")
                return
            param_idx, _dtype_in_map = slot
            idx_reg = self.reg_map.get(op.operands[0].id, "%r0")
            val_reg = self.reg_map.get(op.operands[1].id, "%f0")
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
        # Unhandled — emit a comment for visibility, don't crash
        self._line(f"    // TODO: {op.kind.value}")

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


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import sys
    from ..frontend.parser import parse
    from ..ir.lower_ast import lower
    from ..ir.tile_ir import lower_to_tile

    if len(sys.argv) < 2:
        src = sys.stdin.read()
    else:
        with open(sys.argv[1]) as f:
            src = f.read()
    prog = parse(src)
    tir_mod = lower(prog)
    tile_mod = lower_to_tile(tir_mod)
    print(emit_ptx(tile_mod))
