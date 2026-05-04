"""
kovc/backend/x86_64.py — minimal x86-64 backend (Linux ELF emission).

v0.1 scope (the smallest viable subset):
- Functions with i32 parameters and i32 return type
- Constants
- Add, sub, mul (i32)
- Calls (between user functions)
- Return

Calling convention (System V AMD64 ABI for Linux):
  - First 6 integer args: rdi, rsi, rdx, rcx, r8, r9
  - Return value: rax
  - Callee-saved: rbx, rbp, r12-r15
  - Caller-saved: rax, rcx, rdx, rsi, rdi, r8-r11

For the entry point (`main`), we emit a special wrapper that:
  - Calls main()
  - Uses return value as exit status
  - sys_exit syscall

For simplicity v0.1 uses naive register allocation: every IR value gets a
unique stack slot. We reload to/from rax, rcx, rdx for arithmetic. This is
slow but correct and easy to verify.

License: Apache 2.0
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional

from ..ir import tir


# ============================================================================
# Constants
# ============================================================================
ELF_BASE = 0x400000
CODE_OFFSET = 0x1000   # code segment starts at this file offset (page-aligned)
ENTRY_OFFSET = 0x1000  # entry virtual address: ELF_BASE + ENTRY_OFFSET


# ============================================================================
# Code emitter — appends bytes, tracks fixups for forward references
# ============================================================================
@dataclass
class Fixup:
    """Pending byte location that needs to be patched once the target address is known."""
    offset: int       # offset into code buffer
    target: str       # symbol name to resolve
    size: int         # bytes (4 = rel32 displacement)
    rel_base: int     # offset to subtract from target_addr (= offset + size)


@dataclass
class CodeBuf:
    bytes_: bytearray = field(default_factory=bytearray)
    symbols: dict[str, int] = field(default_factory=dict)   # function name -> offset
    fixups: list[Fixup] = field(default_factory=list)

    def emit(self, *bs: int) -> None:
        self.bytes_.extend(bs)

    def emit_bytes(self, data: bytes) -> None:
        self.bytes_.extend(data)

    def offset(self) -> int:
        return len(self.bytes_)

    def define_symbol(self, name: str) -> None:
        self.symbols[name] = self.offset()

    def patch(self) -> None:
        """Resolve all fixups."""
        for f in self.fixups:
            if f.target not in self.symbols:
                raise ValueError(f"unresolved symbol: {f.target}")
            target_addr = self.symbols[f.target]
            disp = target_addr - f.rel_base
            if f.size == 4:
                struct.pack_into("<i", self.bytes_, f.offset, disp)
            else:
                raise ValueError(f"unsupported fixup size: {f.size}")
        self.fixups.clear()


# ============================================================================
# Instruction encoders (just what we need for v0.1)
# ============================================================================
class Asm:
    """Tiny assembler emitting raw bytes for the x86-64 instructions we need."""

    def __init__(self, buf: CodeBuf):
        self.b = buf

    # ---- prologue/epilogue (System V AMD64) ----
    def push_rbp(self) -> None:
        self.b.emit(0x55)                      # push rbp

    def pop_rbp(self) -> None:
        self.b.emit(0x5D)                      # pop rbp

    def mov_rbp_rsp(self) -> None:
        self.b.emit(0x48, 0x89, 0xE5)          # mov rbp, rsp

    def mov_rsp_rbp(self) -> None:
        self.b.emit(0x48, 0x89, 0xEC)          # mov rsp, rbp

    def sub_rsp_imm32(self, imm: int) -> None:
        # 48 81 EC <imm32>
        self.b.emit(0x48, 0x81, 0xEC)
        self.b.emit_bytes(struct.pack("<i", imm))

    def add_rsp_imm32(self, imm: int) -> None:
        # 48 81 C4 <imm32>
        self.b.emit(0x48, 0x81, 0xC4)
        self.b.emit_bytes(struct.pack("<i", imm))

    def ret(self) -> None:
        self.b.emit(0xC3)

    # ---- arithmetic on eax (32-bit forms; high half auto-zeroed) ----
    def mov_eax_imm32(self, imm: int) -> None:
        # B8 <imm32>
        self.b.emit(0xB8)
        self.b.emit_bytes(struct.pack("<i", imm))

    def mov_eax_mem_rbp(self, disp8: int) -> None:
        """mov eax, [rbp + disp8]"""
        if -128 <= disp8 <= 127:
            self.b.emit(0x8B, 0x45, disp8 & 0xFF)
        else:
            # disp32
            self.b.emit(0x8B, 0x85)
            self.b.emit_bytes(struct.pack("<i", disp8))

    def mov_mem_rbp_eax(self, disp8: int) -> None:
        """mov [rbp + disp8], eax"""
        if -128 <= disp8 <= 127:
            self.b.emit(0x89, 0x45, disp8 & 0xFF)
        else:
            self.b.emit(0x89, 0x85)
            self.b.emit_bytes(struct.pack("<i", disp8))

    def mov_ecx_mem_rbp(self, disp8: int) -> None:
        if -128 <= disp8 <= 127:
            self.b.emit(0x8B, 0x4D, disp8 & 0xFF)
        else:
            self.b.emit(0x8B, 0x8D)
            self.b.emit_bytes(struct.pack("<i", disp8))

    def add_eax_ecx(self) -> None:
        self.b.emit(0x01, 0xC8)                # add eax, ecx

    def sub_eax_ecx(self) -> None:
        self.b.emit(0x29, 0xC8)                # sub eax, ecx

    def imul_eax_ecx(self) -> None:
        self.b.emit(0x0F, 0xAF, 0xC1)          # imul eax, ecx

    def neg_eax(self) -> None:
        self.b.emit(0xF7, 0xD8)                # neg eax

    # ---- arg-register moves (to load function args from rdi/rsi/etc into stack slots) ----
    def mov_mem_rbp_edi(self, disp8: int) -> None:
        if -128 <= disp8 <= 127:
            self.b.emit(0x89, 0x7D, disp8 & 0xFF)
        else:
            self.b.emit(0x89, 0xBD)
            self.b.emit_bytes(struct.pack("<i", disp8))

    def mov_mem_rbp_esi(self, disp8: int) -> None:
        if -128 <= disp8 <= 127:
            self.b.emit(0x89, 0x75, disp8 & 0xFF)
        else:
            self.b.emit(0x89, 0xB5)
            self.b.emit_bytes(struct.pack("<i", disp8))

    def mov_mem_rbp_edx(self, disp8: int) -> None:
        if -128 <= disp8 <= 127:
            self.b.emit(0x89, 0x55, disp8 & 0xFF)
        else:
            self.b.emit(0x89, 0x95)
            self.b.emit_bytes(struct.pack("<i", disp8))

    # ECX, R8D, R9D — extended arg-register stores
    def mov_mem_rbp_ecx(self, disp8: int) -> None:
        if -128 <= disp8 <= 127:
            self.b.emit(0x89, 0x4D, disp8 & 0xFF)
        else:
            self.b.emit(0x89, 0x8D)
            self.b.emit_bytes(struct.pack("<i", disp8))

    def mov_mem_rbp_r8d(self, disp8: int) -> None:
        # 44 89 45 <disp8>   mov [rbp+disp], r8d (REX.R)
        if -128 <= disp8 <= 127:
            self.b.emit(0x44, 0x89, 0x45, disp8 & 0xFF)
        else:
            self.b.emit(0x44, 0x89, 0x85)
            self.b.emit_bytes(struct.pack("<i", disp8))

    def mov_mem_rbp_r9d(self, disp8: int) -> None:
        # 44 89 4D <disp8>   mov [rbp+disp], r9d
        if -128 <= disp8 <= 127:
            self.b.emit(0x44, 0x89, 0x4D, disp8 & 0xFF)
        else:
            self.b.emit(0x44, 0x89, 0x8D)
            self.b.emit_bytes(struct.pack("<i", disp8))

    def mov_edi_eax(self) -> None:
        self.b.emit(0x89, 0xC7)               # mov edi, eax

    # ---- arg-register loads (caller side: load stack slot -> arg register) ----
    def mov_edi_mem_rbp(self, disp8: int) -> None:
        if -128 <= disp8 <= 127:
            self.b.emit(0x8B, 0x7D, disp8 & 0xFF)
        else:
            self.b.emit(0x8B, 0xBD)
            self.b.emit_bytes(struct.pack("<i", disp8))

    def mov_esi_mem_rbp(self, disp8: int) -> None:
        if -128 <= disp8 <= 127:
            self.b.emit(0x8B, 0x75, disp8 & 0xFF)
        else:
            self.b.emit(0x8B, 0xB5)
            self.b.emit_bytes(struct.pack("<i", disp8))

    def mov_edx_mem_rbp(self, disp8: int) -> None:
        if -128 <= disp8 <= 127:
            self.b.emit(0x8B, 0x55, disp8 & 0xFF)
        else:
            self.b.emit(0x8B, 0x95)
            self.b.emit_bytes(struct.pack("<i", disp8))

    def mov_ecx_mem_rbp(self, disp8: int) -> None:  # noqa
        # already defined above for the arithmetic path; this is the
        # arg-load form (same encoding as the existing one)
        if -128 <= disp8 <= 127:
            self.b.emit(0x8B, 0x4D, disp8 & 0xFF)
        else:
            self.b.emit(0x8B, 0x8D)
            self.b.emit_bytes(struct.pack("<i", disp8))

    def mov_r8d_mem_rbp(self, disp8: int) -> None:
        # 44 8B 45 <disp8>   mov r8d, [rbp+disp]
        if -128 <= disp8 <= 127:
            self.b.emit(0x44, 0x8B, 0x45, disp8 & 0xFF)
        else:
            self.b.emit(0x44, 0x8B, 0x85)
            self.b.emit_bytes(struct.pack("<i", disp8))

    def mov_r9d_mem_rbp(self, disp8: int) -> None:
        # 44 8B 4D <disp8>
        if -128 <= disp8 <= 127:
            self.b.emit(0x44, 0x8B, 0x4D, disp8 & 0xFF)
        else:
            self.b.emit(0x44, 0x8B, 0x8D)
            self.b.emit_bytes(struct.pack("<i", disp8))

    # ---- control flow ----
    def call_rel32(self, target: str) -> None:
        # E8 <rel32>
        self.b.emit(0xE8)
        offset = self.b.offset()
        self.b.emit_bytes(b"\x00\x00\x00\x00")
        self.b.fixups.append(Fixup(
            offset=offset, target=target, size=4,
            rel_base=offset + 4,
        ))

    def syscall(self) -> None:
        self.b.emit(0x0F, 0x05)

    # ---- comparisons ----
    def cmp_eax_ecx(self) -> None:
        # 39 C8  cmp eax, ecx
        self.b.emit(0x39, 0xC8)

    # setcc al — produces 1 if condition else 0 in al; we then movzx to eax
    def sete_al(self) -> None:    self.b.emit(0x0F, 0x94, 0xC0)
    def setne_al(self) -> None:   self.b.emit(0x0F, 0x95, 0xC0)
    def setl_al(self) -> None:    self.b.emit(0x0F, 0x9C, 0xC0)
    def setle_al(self) -> None:   self.b.emit(0x0F, 0x9E, 0xC0)
    def setg_al(self) -> None:    self.b.emit(0x0F, 0x9F, 0xC0)
    def setge_al(self) -> None:   self.b.emit(0x0F, 0x9D, 0xC0)

    def movzx_eax_al(self) -> None:
        # 0F B6 C0   movzx eax, al
        self.b.emit(0x0F, 0xB6, 0xC0)


# ============================================================================
# Function compiler (one IR function -> machine code)
# ============================================================================
class FnCompiler:
    """Compiles a single Tensor IR function to x86-64 machine code."""

    def __init__(self, fn: tir.FnIR, asm: Asm):
        self.fn = fn
        self.asm = asm
        # Map SSA value id -> stack frame offset (relative to rbp). Negative = below rbp.
        self.slots: dict[int, int] = {}
        self.next_slot: int = 0   # will decrement as we allocate

    def _alloc_slot(self, v: tir.Value) -> int:
        # Allocate 8 bytes per value (we treat everything as int64-aligned for simplicity)
        self.next_slot -= 8
        self.slots[v.id] = self.next_slot
        return self.next_slot

    def _slot_of(self, v: tir.Value) -> int:
        return self.slots[v.id]

    def compile(self) -> None:
        # Pre-allocate slots for all SSA values used in the body
        for blk in self.fn.blocks:
            for op in blk.ops:
                for r in op.results:
                    self._alloc_slot(r)
        # Pre-allocate slots for params
        for p in self.fn.params:
            self._alloc_slot(p)

        frame_size = (-self.next_slot + 15) & ~15  # 16-byte align

        # Emit function symbol
        self.asm.b.define_symbol(self.fn.name)

        # Prologue
        self.asm.push_rbp()
        self.asm.mov_rbp_rsp()
        if frame_size > 0:
            self.asm.sub_rsp_imm32(frame_size)

        # Spill args from arg registers into stack slots (System V ABI: 6 regs)
        ARG_SPILLS = [
            self.asm.mov_mem_rbp_edi,
            self.asm.mov_mem_rbp_esi,
            self.asm.mov_mem_rbp_edx,
            self.asm.mov_mem_rbp_ecx,
            self.asm.mov_mem_rbp_r8d,
            self.asm.mov_mem_rbp_r9d,
        ]
        for i, p in enumerate(self.fn.params):
            if i >= len(ARG_SPILLS):
                raise NotImplementedError(f"v0.1 supports up to {len(ARG_SPILLS)} parameters")
            slot = self._slot_of(p)
            ARG_SPILLS[i](slot)

        # Emit ops in order
        for op in self.fn.entry.ops:
            self._emit_op(op, frame_size)

    def _emit_op(self, op: tir.Op, frame_size: int) -> None:
        if op.kind == tir.OpKind.CONST_INT:
            slot = self._slot_of(op.results[0])
            self.asm.mov_eax_imm32(int(op.attrs["value"]))
            self.asm.mov_mem_rbp_eax(slot)
            return
        if op.kind == tir.OpKind.ADD:
            l_slot = self._slot_of(op.operands[0])
            r_slot = self._slot_of(op.operands[1])
            res_slot = self._slot_of(op.results[0])
            self.asm.mov_eax_mem_rbp(l_slot)
            self.asm.mov_ecx_mem_rbp(r_slot)
            self.asm.add_eax_ecx()
            self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.SUB:
            l_slot = self._slot_of(op.operands[0])
            r_slot = self._slot_of(op.operands[1])
            res_slot = self._slot_of(op.results[0])
            self.asm.mov_eax_mem_rbp(l_slot)
            self.asm.mov_ecx_mem_rbp(r_slot)
            self.asm.sub_eax_ecx()
            self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.MUL:
            l_slot = self._slot_of(op.operands[0])
            r_slot = self._slot_of(op.operands[1])
            res_slot = self._slot_of(op.results[0])
            self.asm.mov_eax_mem_rbp(l_slot)
            self.asm.mov_ecx_mem_rbp(r_slot)
            self.asm.imul_eax_ecx()
            self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.NEG:
            slot = self._slot_of(op.operands[0])
            res_slot = self._slot_of(op.results[0])
            self.asm.mov_eax_mem_rbp(slot)
            self.asm.neg_eax()
            self.asm.mov_mem_rbp_eax(res_slot)
            return
        # Comparisons
        cmp_setters = {
            tir.OpKind.CMP_EQ: self.asm.sete_al,
            tir.OpKind.CMP_NE: self.asm.setne_al,
            tir.OpKind.CMP_LT: self.asm.setl_al,
            tir.OpKind.CMP_LE: self.asm.setle_al,
            tir.OpKind.CMP_GT: self.asm.setg_al,
            tir.OpKind.CMP_GE: self.asm.setge_al,
        }
        if op.kind in cmp_setters:
            l_slot = self._slot_of(op.operands[0])
            r_slot = self._slot_of(op.operands[1])
            res_slot = self._slot_of(op.results[0])
            self.asm.mov_eax_mem_rbp(l_slot)
            self.asm.mov_ecx_mem_rbp(r_slot)
            self.asm.cmp_eax_ecx()
            cmp_setters[op.kind]()
            self.asm.movzx_eax_al()
            self.asm.mov_mem_rbp_eax(res_slot)
            return
        # SELECT (cond, a, b) — a if cond else b. v0.1: branchless via cmov-equivalent
        # using arithmetic: result = (cond * a) + ((1-cond) * b). Implementing as
        # eax = b; if cond { eax = a; } using a small branch.
        if op.kind == tir.OpKind.SELECT:
            cond_slot = self._slot_of(op.operands[0])
            a_slot = self._slot_of(op.operands[1])
            b_slot = self._slot_of(op.operands[2])
            res_slot = self._slot_of(op.results[0])
            # Load cond, test
            self.asm.mov_eax_mem_rbp(cond_slot)
            # test eax, eax
            self.asm.b.emit(0x85, 0xC0)
            # je over-a (placeholder, 8-bit)
            # We need a forward jump. Use a fixup-friendly relative computation.
            # For simplicity, compute the jump distance manually.
            # Sequence: je SKIP_A; mov eax, [rbp+a]; jmp END; SKIP_A: mov eax, [rbp+b]; END: mov [rbp+res], eax
            # Sizes: each mov_eax_mem_rbp is 3 bytes (when disp8); jmp rel8 is 2 bytes.
            # je rel8 = 2 bytes; "load a" = 3 bytes; jmp rel8 = 2 bytes; "load b" = 3 bytes.
            # je target: skip past (load_a + jmp) = 3 + 2 = 5
            self.asm.b.emit(0x74, 0x05)             # je +5
            self.asm.mov_eax_mem_rbp(a_slot)        # 3 bytes (assumes disp8)
            self.asm.b.emit(0xEB, 0x03)             # jmp +3 (skip load_b)
            self.asm.mov_eax_mem_rbp(b_slot)        # 3 bytes
            # END:
            self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.CALL:
            target = op.attrs.get("target", "?")
            ARG_REGS_LOAD = [
                self.asm.mov_edi_mem_rbp,
                self.asm.mov_esi_mem_rbp,
                self.asm.mov_edx_mem_rbp,
                self.asm.mov_ecx_mem_rbp,
                self.asm.mov_r8d_mem_rbp,
                self.asm.mov_r9d_mem_rbp,
            ]
            for i, arg in enumerate(op.operands):
                if i >= len(ARG_REGS_LOAD):
                    raise NotImplementedError("v0.1 supports up to 6 call args")
                arg_slot = self._slot_of(arg)
                ARG_REGS_LOAD[i](arg_slot)
            self.asm.call_rel32(str(target))
            if op.results:
                res_slot = self._slot_of(op.results[0])
                self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.RETURN:
            if op.operands:
                slot = self._slot_of(op.operands[0])
                self.asm.mov_eax_mem_rbp(slot)
            else:
                self.asm.mov_eax_imm32(0)
            # Epilogue
            self.asm.mov_rsp_rbp()
            self.asm.pop_rbp()
            self.asm.ret()
            return
        # Unsupported op — emit nothing (placeholder); v0.2 will lower
        # tensor ops to runtime calls.


# ============================================================================
# ELF emission
# ============================================================================
def emit_elf(code: bytes, entry_offset: int = ENTRY_OFFSET) -> bytes:
    """Wrap the given code bytes in a minimal x86-64 Linux ELF executable."""
    # Layout:
    #   0x00: ELF header (64 B)
    #   0x40: Program header (56 B)
    #   0x1000: Code (page-aligned)
    code_vaddr = ELF_BASE + entry_offset
    file_size_to_code = CODE_OFFSET
    total_filesz = file_size_to_code + len(code)
    total_memsz = total_filesz

    # ELF64 header
    ehdr = b"\x7fELF"            # EI_MAG
    ehdr += b"\x02"              # EI_CLASS = ELFCLASS64
    ehdr += b"\x01"              # EI_DATA = LSB
    ehdr += b"\x01"              # EI_VERSION
    ehdr += b"\x00"              # EI_OSABI = SysV
    ehdr += b"\x00"              # EI_ABIVERSION
    ehdr += b"\x00" * 7          # EI_PAD
    ehdr += struct.pack("<H", 2)              # e_type = ET_EXEC
    ehdr += struct.pack("<H", 0x3E)           # e_machine = EM_X86_64
    ehdr += struct.pack("<I", 1)              # e_version
    ehdr += struct.pack("<Q", code_vaddr)     # e_entry
    ehdr += struct.pack("<Q", 64)             # e_phoff
    ehdr += struct.pack("<Q", 0)              # e_shoff
    ehdr += struct.pack("<I", 0)              # e_flags
    ehdr += struct.pack("<H", 64)             # e_ehsize
    ehdr += struct.pack("<H", 56)             # e_phentsize
    ehdr += struct.pack("<H", 1)              # e_phnum
    ehdr += struct.pack("<H", 0)              # e_shentsize
    ehdr += struct.pack("<H", 0)              # e_shnum
    ehdr += struct.pack("<H", 0)              # e_shstrndx

    # Program header (PT_LOAD, R+X)
    phdr = struct.pack("<I", 1)               # p_type = PT_LOAD
    phdr += struct.pack("<I", 5)              # p_flags = R | X
    phdr += struct.pack("<Q", 0)              # p_offset = 0
    phdr += struct.pack("<Q", ELF_BASE)       # p_vaddr
    phdr += struct.pack("<Q", ELF_BASE)       # p_paddr
    phdr += struct.pack("<Q", total_filesz)   # p_filesz
    phdr += struct.pack("<Q", total_memsz)    # p_memsz
    phdr += struct.pack("<Q", 0x1000)         # p_align

    # Padding from end of phdr (offset 120 = 0x78) to CODE_OFFSET (0x1000)
    pad_size = CODE_OFFSET - len(ehdr) - len(phdr)
    pad = b"\x00" * pad_size

    return ehdr + phdr + pad + code


# ============================================================================
# Top-level compile
# ============================================================================
def compile_module_to_elf(module: tir.Module, entry_fn: str = "main") -> bytes:
    """Compile a TIR Module to an x86-64 Linux ELF executable.

    The function `entry_fn` (default "main") becomes the ELF entry point.
    Its return value (i32, in eax) becomes the process exit status via the
    sys_exit syscall emitted as an ELF entry stub.
    """
    if entry_fn not in module.functions:
        raise ValueError(f"module has no function {entry_fn!r}; "
                         f"available: {list(module.functions)}")

    buf = CodeBuf()
    asm = Asm(buf)

    # Entry stub: call entry_fn, then exit with eax as status
    buf.define_symbol("_start")
    asm.call_rel32(entry_fn)
    # rdi = rax (status from main)
    asm.mov_edi_eax()
    # rax = 60 (sys_exit)
    asm.mov_eax_imm32(60)
    asm.syscall()

    # Compile each function
    for fn in module.functions.values():
        FnCompiler(fn, asm).compile()

    buf.patch()
    return emit_elf(bytes(buf.bytes_))


if __name__ == "__main__":
    import sys
    from ..frontend.parser import parse
    from ..ir.lower_ast import lower

    if len(sys.argv) < 3:
        print("usage: python -m kovc.backend.x86_64 <input.kov> <output.bin>",
              file=sys.stderr)
        sys.exit(1)
    with open(sys.argv[1]) as f:
        src = f.read()
    prog = parse(src)
    mod = lower(prog)
    elf = compile_module_to_elf(mod)
    with open(sys.argv[2], "wb") as f:
        f.write(elf)
    import os
    os.chmod(sys.argv[2], 0o755)
    print(f"Wrote {sys.argv[2]} ({len(elf)} bytes)")
