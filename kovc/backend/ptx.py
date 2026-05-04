"""
kovc/backend/ptx.py — NVIDIA PTX backend (text emission).

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

    def _new_reg(self, prefix: str = "r") -> str:
        n = f"%{prefix}{self.next_reg}"
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
        self.reg_map = {}
        # Reserve a small register pool
        self._line("    .reg .pred  %p<8>;")
        self._line("    .reg .b32   %r<32>;")
        self._line("    .reg .b64   %rd<8>;")
        self._line("    .reg .f32   %f<32>;")
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
            mapping = {
                "i8": ".b8", "i16": ".b16", "i32": ".b32", "i64": ".b64",
                "u8": ".b8", "u16": ".b16", "u32": ".b32", "u64": ".b64",
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
            a = self.reg_map.get(op.operands[0].id, "%r0")
            b = self.reg_map.get(op.operands[1].id, "%r1")
            r = self._new_reg("r")
            self._line(f"    add.s32 {r}, {a}, {b};")
            if op.results:
                self.reg_map[op.results[0].id] = r
            return
        if op.kind == ti.TileOpKind.SCALAR_MUL:
            a = self.reg_map.get(op.operands[0].id, "%r0")
            b = self.reg_map.get(op.operands[1].id, "%r1")
            r = self._new_reg("r")
            self._line(f"    mul.lo.s32 {r}, {a}, {b};")
            if op.results:
                self.reg_map[op.results[0].id] = r
            return
        if op.kind == ti.TileOpKind.RETURN:
            return
        # Unhandled — emit a comment for visibility, don't crash
        self._line(f"    // TODO: {op.kind.value}")


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
