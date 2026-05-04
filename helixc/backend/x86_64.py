"""
helixc/backend/x86_64.py — minimal x86-64 backend (Linux ELF emission).

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

# Reflection cells: 64 i64 mutable cells appended at the end of the code
# segment. Each cell is 8 bytes, addressed by index. Used for verifier-gated
# self-modification (quote/splice/modify primitives).
HELIX_NUM_CELLS = 64
HELIX_CELL_SIZE = 8


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
        # B8 <imm32>. Accept any 32-bit bit pattern (signed or unsigned) by
        # masking to 32 bits and re-packing as unsigned. Required for callers
        # that pass packed float bits — e.g. the bit pattern for -1.0
        # (0xBF800000) doesn't fit in signed i32.
        self.b.emit(0xB8)
        self.b.emit_bytes(struct.pack("<I", imm & 0xFFFFFFFF))

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

    def cdq(self) -> None:
        self.b.emit(0x99)                      # sign-extend eax into edx

    def idiv_ecx(self) -> None:
        # F7 F9   idiv ecx (signed); edx:eax / ecx -> eax=quotient, edx=remainder
        self.b.emit(0xF7, 0xF9)

    def mov_eax_edx(self) -> None:
        # 89 D0   mov eax, edx
        self.b.emit(0x89, 0xD0)

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

    def jmp_rel32(self, target: str) -> None:
        # E9 <rel32>
        self.b.emit(0xE9)
        offset = self.b.offset()
        self.b.emit_bytes(b"\x00\x00\x00\x00")
        self.b.fixups.append(Fixup(
            offset=offset, target=target, size=4,
            rel_base=offset + 4,
        ))

    def je_rel32(self, target: str) -> None:
        # 0F 84 <rel32>
        self.b.emit(0x0F, 0x84)
        offset = self.b.offset()
        self.b.emit_bytes(b"\x00\x00\x00\x00")
        self.b.fixups.append(Fixup(
            offset=offset, target=target, size=4,
            rel_base=offset + 4,
        ))

    def jne_rel32(self, target: str) -> None:
        # 0F 85 <rel32>
        self.b.emit(0x0F, 0x85)
        offset = self.b.offset()
        self.b.emit_bytes(b"\x00\x00\x00\x00")
        self.b.fixups.append(Fixup(
            offset=offset, target=target, size=4,
            rel_base=offset + 4,
        ))

    def test_eax_eax(self) -> None:
        # 85 C0
        self.b.emit(0x85, 0xC0)

    # ---- RIP-relative load/store for reflection cells ----
    def mov_rax_rip_rel(self, target: str) -> None:
        """mov rax, [rip + disp32]   (load 64-bit)
        48 8B 05 <disp32>"""
        self.b.emit(0x48, 0x8B, 0x05)
        offset = self.b.offset()
        self.b.emit_bytes(b"\x00\x00\x00\x00")
        self.b.fixups.append(Fixup(offset=offset, target=target,
                                   size=4, rel_base=offset + 4))

    def mov_rip_rel_rax(self, target: str) -> None:
        """mov [rip + disp32], rax   (store 64-bit)
        48 89 05 <disp32>"""
        self.b.emit(0x48, 0x89, 0x05)
        offset = self.b.offset()
        self.b.emit_bytes(b"\x00\x00\x00\x00")
        self.b.fixups.append(Fixup(offset=offset, target=target,
                                   size=4, rel_base=offset + 4))

    def lea_rax_rip_rel(self, target: str) -> None:
        """lea rax, [rip + disp32]   (load address)
        48 8D 05 <disp32>"""
        self.b.emit(0x48, 0x8D, 0x05)
        offset = self.b.offset()
        self.b.emit_bytes(b"\x00\x00\x00\x00")
        self.b.fixups.append(Fixup(offset=offset, target=target,
                                   size=4, rel_base=offset + 4))

    def mov_rax_mem_rax_rcx8(self) -> None:
        """mov rax, [rax + rcx*8]
        48 8B 04 C8  (REX.W + 8B + ModRM(00, 000, 100) + SIB(scale=11, idx=001, base=000))"""
        self.b.emit(0x48, 0x8B, 0x04, 0xC8)

    def mov_mem_rax_rcx8_rdx(self) -> None:
        """mov [rax + rcx*8], rdx
        48 89 14 C8"""
        self.b.emit(0x48, 0x89, 0x14, 0xC8)

    def mov_rcx_mem_rbp(self, disp: int) -> None:
        """mov rcx, [rbp + disp]   (load 64-bit into rcx)
        48 8B 4D <disp8>  or  48 8B 8D <disp32>"""
        if -128 <= disp <= 127:
            self.b.emit(0x48, 0x8B, 0x4D, disp & 0xFF)
        else:
            self.b.emit(0x48, 0x8B, 0x8D)
            self.b.emit_bytes(struct.pack("<i", disp))

    def mov_rdx_mem_rbp(self, disp: int) -> None:
        """mov rdx, [rbp + disp]   (load 64-bit into rdx)
        48 8B 55 <disp8>  or  48 8B 95 <disp32>"""
        if -128 <= disp <= 127:
            self.b.emit(0x48, 0x8B, 0x55, disp & 0xFF)
        else:
            self.b.emit(0x48, 0x8B, 0x95)
            self.b.emit_bytes(struct.pack("<i", disp))

    # ============================================================
    # Float (SSE) instructions — operate on xmm0/xmm1
    # ============================================================
    def movss_xmm0_mem_rbp(self, disp: int) -> None:
        # F3 0F 10 45 disp8   (mod=01, reg=000=xmm0, r/m=101=rbp+disp)
        if -128 <= disp <= 127:
            self.b.emit(0xF3, 0x0F, 0x10, 0x45, disp & 0xFF)
        else:
            self.b.emit(0xF3, 0x0F, 0x10, 0x85)
            self.b.emit_bytes(struct.pack("<i", disp))

    def movss_xmm1_mem_rbp(self, disp: int) -> None:
        # F3 0F 10 4D disp8   (reg=001=xmm1)
        if -128 <= disp <= 127:
            self.b.emit(0xF3, 0x0F, 0x10, 0x4D, disp & 0xFF)
        else:
            self.b.emit(0xF3, 0x0F, 0x10, 0x8D)
            self.b.emit_bytes(struct.pack("<i", disp))

    def movss_mem_rbp_xmm0(self, disp: int) -> None:
        # F3 0F 11 45 disp8
        if -128 <= disp <= 127:
            self.b.emit(0xF3, 0x0F, 0x11, 0x45, disp & 0xFF)
        else:
            self.b.emit(0xF3, 0x0F, 0x11, 0x85)
            self.b.emit_bytes(struct.pack("<i", disp))

    def addss_xmm0_xmm1(self) -> None:  # xmm0 = xmm0 + xmm1
        self.b.emit(0xF3, 0x0F, 0x58, 0xC1)

    def subss_xmm0_xmm1(self) -> None:
        self.b.emit(0xF3, 0x0F, 0x5C, 0xC1)

    def mulss_xmm0_xmm1(self) -> None:
        self.b.emit(0xF3, 0x0F, 0x59, 0xC1)

    def divss_xmm0_xmm1(self) -> None:
        self.b.emit(0xF3, 0x0F, 0x5E, 0xC1)

    def cvttss2si_eax_xmm0(self) -> None:
        # F3 0F 2C C0   (truncating float -> signed int32)
        self.b.emit(0xF3, 0x0F, 0x2C, 0xC0)

    def cvtsi2ss_xmm0_eax(self) -> None:
        # F3 0F 2A C0
        self.b.emit(0xF3, 0x0F, 0x2A, 0xC0)

    def comiss_xmm0_xmm1(self) -> None:
        # NP 0F 2F C1
        self.b.emit(0x0F, 0x2F, 0xC1)

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
    # Unsigned variants (used for float compares: ucomiss sets CF/ZF in the
    # unsigned-compare sense — CF=1 when xmm0 < xmm1).
    def seta_al(self) -> None:    self.b.emit(0x0F, 0x97, 0xC0)
    def setae_al(self) -> None:   self.b.emit(0x0F, 0x93, 0xC0)
    def setb_al(self) -> None:    self.b.emit(0x0F, 0x92, 0xC0)
    def setbe_al(self) -> None:   self.b.emit(0x0F, 0x96, 0xC0)

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
        # Mutable variable name -> stack slot
        self.var_slots: dict[str, int] = {}
        # Arrays: name -> (base_slot_offset, length, element_size_in_bytes)
        # Elements occupy contiguous 8-byte slots starting at base_slot_offset
        # (base_slot_offset is the offset of element 0; elem i is at base + i*8)
        self.array_info: dict[str, tuple[int, int, int]] = {}

    def _alloc_var(self, name: str) -> int:
        if name in self.var_slots:
            return self.var_slots[name]
        self.next_slot -= 8
        self.var_slots[name] = self.next_slot
        return self.next_slot

    def _alloc_array(self, name: str, length: int, elem_size: int = 8) -> int:
        """Allocate a contiguous block of length * elem_size bytes on the stack.
        We use 8 bytes per element (i32 zero-padded) for simplicity in v0.1."""
        if name in self.array_info:
            return self.array_info[name][0]
        # Reserve length * 8 bytes
        self.next_slot -= length * 8
        # base_slot_offset points to element 0
        base = self.next_slot
        self.array_info[name] = (base, length, 8)
        return base

    def _alloc_slot(self, v: tir.Value) -> int:
        # Allocate 8 bytes per value (we treat everything as int64-aligned for simplicity)
        self.next_slot -= 8
        self.slots[v.id] = self.next_slot
        return self.next_slot

    def _slot_of(self, v: tir.Value) -> int:
        return self.slots[v.id]

    def compile(self) -> None:
        # Pre-allocate slots for arrays (ALLOC_ARRAY ops) before vars/SSA values
        for blk in self.fn.blocks:
            for op in blk.ops:
                if op.kind == tir.OpKind.ALLOC_ARRAY:
                    name = op.attrs.get("name")
                    length = int(op.attrs.get("length", 0))
                    if name and name not in self.array_info:
                        self._alloc_array(name, length)

        # Pre-allocate slots for mutable variables (ALLOC_VAR ops)
        for blk in self.fn.blocks:
            for op in blk.ops:
                if op.kind == tir.OpKind.ALLOC_VAR:
                    name = op.attrs.get("name")
                    if name and name not in self.var_slots:
                        self._alloc_var(name)

        # Pre-allocate slots for all SSA values across ALL blocks (not just entry)
        for blk in self.fn.blocks:
            for p in blk.params:
                self._alloc_slot(p)
            for op in blk.ops:
                for r in op.results:
                    self._alloc_slot(r)
        # Pre-allocate slots for fn params (they share entry block params slot conceptually)
        for p in self.fn.params:
            if p.id not in self.slots:
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

        # Emit each block in order, with a label per block
        for blk in self.fn.blocks:
            block_label = f"{self.fn.name}_bb{blk.id}"
            self.asm.b.define_symbol(block_label)
            for op in blk.ops:
                self._emit_op(op, frame_size)

    def _is_float_type(self, ty: tir.TIRType) -> bool:
        return isinstance(ty, tir.TIRScalar) and ty.name in ("f16", "bf16", "f32", "f64")

    def _emit_idiv_guarded(self, l_slot: int, r_slot: int, res_slot: int,
                           *, want_quotient: bool) -> None:
        """Emit signed 32-bit integer division with the INT_MIN/-1 trap-avoidance
        guard. On x86, `idiv ecx` raises #DE when eax = INT_MIN and ecx = -1
        (the quotient INT_MIN/-1 = INT_MIN+1 doesn't fit signed 32, but at the
        hardware level the well-known trap is INT_MIN/-1 producing #DE).
        We define INT_MIN/-1 = INT_MIN (matching wraparound) and INT_MIN%-1 = 0.

        Sequence:
          mov  eax, [l]
          mov  ecx, [r]
          cmp  ecx, -1
          jne  do_div
          cmp  eax, 0x80000000
          jne  do_div
          ; INT_MIN/-1 path: skip idiv, set quotient=INT_MIN or remainder=0
          jmp  done
        do_div:
          cdq
          idiv ecx
          (if !want_quotient: mov eax, edx)
        done:
          mov  [res], eax
        """
        buf = self.asm.b
        self.asm.mov_eax_mem_rbp(l_slot)
        self.asm.mov_ecx_mem_rbp(r_slot)

        # cmp ecx, -1   (83 F9 FF; rel8 sign-extended imm8)
        buf.emit(0x83, 0xF9, 0xFF)
        # jne do_div  (placeholder rel8)
        buf.emit(0x75, 0x00)
        jne1_disp_off = buf.offset() - 1
        jne1_after = buf.offset()

        # cmp eax, 0x80000000   (3D <imm32> = cmp eax, imm32)
        buf.emit(0x3D)
        buf.emit_bytes(struct.pack("<I", 0x80000000))
        # jne do_div
        buf.emit(0x75, 0x00)
        jne2_disp_off = buf.offset() - 1
        jne2_after = buf.offset()

        # INT_MIN/-1 path: produce eax = INT_MIN (for div) or 0 (for mod)
        if want_quotient:
            # mov eax, 0x80000000
            self.asm.mov_eax_imm32(0x80000000)
        else:
            # xor eax, eax  (31 C0)
            buf.emit(0x31, 0xC0)
        # jmp done (rel8 placeholder)
        buf.emit(0xEB, 0x00)
        jmp_done_disp_off = buf.offset() - 1
        jmp_done_after = buf.offset()

        # do_div:
        do_div_addr = buf.offset()
        self.asm.cdq()
        self.asm.idiv_ecx()
        if not want_quotient:
            self.asm.mov_eax_edx()

        done_addr = buf.offset()

        # Patch the three rel8 jumps
        d1 = do_div_addr - jne1_after
        d2 = do_div_addr - jne2_after
        d3 = done_addr - jmp_done_after
        for d, name in ((d1, "jne1"), (d2, "jne2"), (d3, "jmp_done")):
            if not (-128 <= d <= 127):
                raise ValueError(f"idiv-guard {name} disp out of rel8: {d}")
        buf.bytes_[jne1_disp_off] = d1 & 0xFF
        buf.bytes_[jne2_disp_off] = d2 & 0xFF
        buf.bytes_[jmp_done_disp_off] = d3 & 0xFF

        self.asm.mov_mem_rbp_eax(res_slot)

    def _emit_op(self, op: tir.Op, frame_size: int) -> None:
        if op.kind == tir.OpKind.CONST_INT:
            slot = self._slot_of(op.results[0])
            self.asm.mov_eax_imm32(int(op.attrs["value"]))
            self.asm.mov_mem_rbp_eax(slot)
            return
        if op.kind == tir.OpKind.CONST_FLOAT:
            # Pack the f32 value into 4 bytes, store at the result's slot via eax
            slot = self._slot_of(op.results[0])
            value = float(op.attrs["value"])
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
            # mov eax, <bits>; mov [rbp+slot], eax
            self.asm.mov_eax_imm32(bits)
            self.asm.mov_mem_rbp_eax(slot)
            return
        if op.kind == tir.OpKind.CAST:
            src_slot = self._slot_of(op.operands[0])
            res_slot = self._slot_of(op.results[0])
            from_ty = op.operands[0].ty
            to_ty = op.results[0].ty
            # i32 -> f32: load int, cvtsi2ss, store float
            if not self._is_float_type(from_ty) and self._is_float_type(to_ty):
                self.asm.mov_eax_mem_rbp(src_slot)
                self.asm.cvtsi2ss_xmm0_eax()
                self.asm.movss_mem_rbp_xmm0(res_slot)
                return
            # f32 -> i32: load float, cvttss2si, store int
            if self._is_float_type(from_ty) and not self._is_float_type(to_ty):
                self.asm.movss_xmm0_mem_rbp(src_slot)
                self.asm.cvttss2si_eax_xmm0()
                self.asm.mov_mem_rbp_eax(res_slot)
                return
            # Same kind: just memory copy
            if self._is_float_type(from_ty) == self._is_float_type(to_ty):
                self.asm.mov_eax_mem_rbp(src_slot)
                self.asm.mov_mem_rbp_eax(res_slot)
                return
            return
        if op.kind == tir.OpKind.ADD:
            l_slot = self._slot_of(op.operands[0])
            r_slot = self._slot_of(op.operands[1])
            res_slot = self._slot_of(op.results[0])
            if self._is_float_type(op.results[0].ty):
                self.asm.movss_xmm0_mem_rbp(l_slot)
                self.asm.movss_xmm1_mem_rbp(r_slot)
                self.asm.addss_xmm0_xmm1()
                self.asm.movss_mem_rbp_xmm0(res_slot)
            else:
                self.asm.mov_eax_mem_rbp(l_slot)
                self.asm.mov_ecx_mem_rbp(r_slot)
                self.asm.add_eax_ecx()
                self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.SUB:
            l_slot = self._slot_of(op.operands[0])
            r_slot = self._slot_of(op.operands[1])
            res_slot = self._slot_of(op.results[0])
            if self._is_float_type(op.results[0].ty):
                self.asm.movss_xmm0_mem_rbp(l_slot)
                self.asm.movss_xmm1_mem_rbp(r_slot)
                self.asm.subss_xmm0_xmm1()
                self.asm.movss_mem_rbp_xmm0(res_slot)
            else:
                self.asm.mov_eax_mem_rbp(l_slot)
                self.asm.mov_ecx_mem_rbp(r_slot)
                self.asm.sub_eax_ecx()
                self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.MUL:
            l_slot = self._slot_of(op.operands[0])
            r_slot = self._slot_of(op.operands[1])
            res_slot = self._slot_of(op.results[0])
            if self._is_float_type(op.results[0].ty):
                self.asm.movss_xmm0_mem_rbp(l_slot)
                self.asm.movss_xmm1_mem_rbp(r_slot)
                self.asm.mulss_xmm0_xmm1()
                self.asm.movss_mem_rbp_xmm0(res_slot)
            else:
                self.asm.mov_eax_mem_rbp(l_slot)
                self.asm.mov_ecx_mem_rbp(r_slot)
                self.asm.imul_eax_ecx()
                self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.DIV:
            l_slot = self._slot_of(op.operands[0])
            r_slot = self._slot_of(op.operands[1])
            res_slot = self._slot_of(op.results[0])
            if self._is_float_type(op.results[0].ty):
                self.asm.movss_xmm0_mem_rbp(l_slot)
                self.asm.movss_xmm1_mem_rbp(r_slot)
                self.asm.divss_xmm0_xmm1()
                self.asm.movss_mem_rbp_xmm0(res_slot)
            else:
                self._emit_idiv_guarded(l_slot, r_slot, res_slot, want_quotient=True)
            return
        if op.kind == tir.OpKind.MOD:
            l_slot = self._slot_of(op.operands[0])
            r_slot = self._slot_of(op.operands[1])
            res_slot = self._slot_of(op.results[0])
            self._emit_idiv_guarded(l_slot, r_slot, res_slot, want_quotient=False)
            return
        if op.kind == tir.OpKind.NEG:
            slot = self._slot_of(op.operands[0])
            res_slot = self._slot_of(op.results[0])
            self.asm.mov_eax_mem_rbp(slot)
            self.asm.neg_eax()
            self.asm.mov_mem_rbp_eax(res_slot)
            return
        # Comparisons. Integer and float paths are different:
        #   integer: cmp eax, ecx + signed setcc (setl/setg/...)
        #   float:   ucomiss xmm0, xmm1 + unsigned setcc (setb/seta/...).
        # The float path uses unsigned condition codes because ucomiss writes
        # CF=1 when xmm0 < xmm1 (and CF=0 when xmm0 >= xmm1) — i.e. the
        # below/above semantics of unsigned cmp. The integer path's signed
        # setcc on raw float bit patterns silently miscompiled negative-
        # value compares (e.g. -2.0 < -0.001 returned false because the
        # integer interpretation of those bit patterns reverses the order).
        int_cmp_setters = {
            tir.OpKind.CMP_EQ: self.asm.sete_al,
            tir.OpKind.CMP_NE: self.asm.setne_al,
            tir.OpKind.CMP_LT: self.asm.setl_al,
            tir.OpKind.CMP_LE: self.asm.setle_al,
            tir.OpKind.CMP_GT: self.asm.setg_al,
            tir.OpKind.CMP_GE: self.asm.setge_al,
        }
        float_cmp_setters = {
            tir.OpKind.CMP_EQ: self.asm.sete_al,
            tir.OpKind.CMP_NE: self.asm.setne_al,
            tir.OpKind.CMP_LT: self.asm.setb_al,
            tir.OpKind.CMP_LE: self.asm.setbe_al,
            tir.OpKind.CMP_GT: self.asm.seta_al,
            tir.OpKind.CMP_GE: self.asm.setae_al,
        }
        if op.kind in int_cmp_setters:
            l_slot = self._slot_of(op.operands[0])
            r_slot = self._slot_of(op.operands[1])
            res_slot = self._slot_of(op.results[0])
            # Choose path by operand type, not result type (result is bool).
            if (self._is_float_type(op.operands[0].ty) or
                    self._is_float_type(op.operands[1].ty)):
                self.asm.movss_xmm0_mem_rbp(l_slot)
                self.asm.movss_xmm1_mem_rbp(r_slot)
                self.asm.comiss_xmm0_xmm1()
                float_cmp_setters[op.kind]()
            else:
                self.asm.mov_eax_mem_rbp(l_slot)
                self.asm.mov_ecx_mem_rbp(r_slot)
                self.asm.cmp_eax_ecx()
                int_cmp_setters[op.kind]()
            self.asm.movzx_eax_al()
            self.asm.mov_mem_rbp_eax(res_slot)
            return
        # SELECT (cond, a, b) — a if cond else b.
        # Branch sequence:
        #   mov eax, [cond]
        #   test eax, eax
        #   je SKIP_A          ; if cond == 0 jump to b
        #   mov eax, [a]       ; size depends on a's displacement (disp8 vs disp32)
        #   jmp END
        # SKIP_A:
        #   mov eax, [b]
        # END:
        #   mov [res], eax
        # We can't pre-compute the je/jmp displacements because the mov sizes
        # depend on whether the slots fit in disp8. Emit placeholder rel8s
        # then patch with the actual byte-counted distance.
        if op.kind == tir.OpKind.SELECT:
            cond_slot = self._slot_of(op.operands[0])
            a_slot = self._slot_of(op.operands[1])
            b_slot = self._slot_of(op.operands[2])
            res_slot = self._slot_of(op.results[0])
            buf = self.asm.b
            # mov eax, [cond]
            self.asm.mov_eax_mem_rbp(cond_slot)
            # test eax, eax
            buf.emit(0x85, 0xC0)
            # je rel8 — write 0x74 + placeholder, remember placeholder offset
            buf.emit(0x74, 0x00)
            je_disp_off = buf.offset() - 1
            je_after = buf.offset()
            # load a
            self.asm.mov_eax_mem_rbp(a_slot)
            # jmp rel8 — placeholder
            buf.emit(0xEB, 0x00)
            jmp_disp_off = buf.offset() - 1
            jmp_after = buf.offset()
            # SKIP_A: load b
            skip_a_addr = buf.offset()
            self.asm.mov_eax_mem_rbp(b_slot)
            end_addr = buf.offset()
            # Patch je: skip past (load_a + jmp), targeting skip_a_addr
            je_disp = skip_a_addr - je_after
            jmp_disp = end_addr - jmp_after
            if not (-128 <= je_disp <= 127) or not (-128 <= jmp_disp <= 127):
                # Should not happen for a single mov + jmp, but guard anyway —
                # current load is at most 7 bytes (disp32 + REX), so disp <= 9.
                raise ValueError(
                    f"SELECT branch displacement out of rel8 range: "
                    f"je={je_disp}, jmp={jmp_disp}"
                )
            buf.bytes_[je_disp_off] = je_disp & 0xFF
            buf.bytes_[jmp_disp_off] = jmp_disp & 0xFF
            # store result
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
        # Branches
        if op.kind == tir.OpKind.BR:
            # br target_block(value) — copy value into target block's param slot,
            # then jmp to label.
            target_id = op.attrs["target_block"]
            target_label = f"{self.fn.name}_bb{target_id}"
            # Find target block to get its param slot (for now: single param assumed)
            target_blk = next((b for b in self.fn.blocks if b.id == target_id), None)
            if target_blk is None:
                raise ValueError(f"BR to unknown block {target_id}")
            if op.operands and target_blk.params:
                src_slot = self._slot_of(op.operands[0])
                dst_slot = self._slot_of(target_blk.params[0])
                self.asm.mov_eax_mem_rbp(src_slot)
                self.asm.mov_mem_rbp_eax(dst_slot)
            self.asm.jmp_rel32(target_label)
            return
        if op.kind == tir.OpKind.COND_BR:
            # cond_br cond, true_block, false_block
            cond_slot = self._slot_of(op.operands[0])
            true_id = op.attrs["true_block"]
            false_id = op.attrs["false_block"]
            true_label = f"{self.fn.name}_bb{true_id}"
            false_label = f"{self.fn.name}_bb{false_id}"
            self.asm.mov_eax_mem_rbp(cond_slot)
            self.asm.test_eax_eax()
            # If cond != 0, jump to true_label; else jump to false_label
            self.asm.jne_rel32(true_label)
            self.asm.jmp_rel32(false_label)
            return

        # Mutable variables
        if op.kind == tir.OpKind.ALLOC_VAR:
            # Slot already pre-allocated; nothing to emit
            return
        if op.kind == tir.OpKind.LOAD_VAR:
            name = op.attrs["name"]
            var_slot = self.var_slots[name]
            res_slot = self._slot_of(op.results[0])
            self.asm.mov_eax_mem_rbp(var_slot)
            self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.STORE_VAR:
            name = op.attrs["name"]
            var_slot = self.var_slots[name]
            src_slot = self._slot_of(op.operands[0])
            self.asm.mov_eax_mem_rbp(src_slot)
            self.asm.mov_mem_rbp_eax(var_slot)
            return

        # AGI primitives
        if op.kind == tir.OpKind.QUOTE:
            # QUOTE returns a stable cell handle in [0, HELIX_NUM_CELLS). The
            # handle is derived at compile time from the AST hash; runtime
            # state cells live in the binary's writable region.
            res_slot = self._slot_of(op.results[0])
            handle = int(op.attrs.get("ast_handle", 0)) % HELIX_NUM_CELLS
            self.asm.mov_eax_imm32(handle & 0xFFFFFFFF)
            self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.SPLICE:
            # splice(handle): load the i64 value at __helix_cell_<handle>.
            # Handle is dynamic (runtime operand value); we sign-extend it
            # to 64 bits then index the cell array. CRITICAL: a malicious
            # or buggy handle outside [0, HELIX_NUM_CELLS) would read past
            # the cells region into other code memory. We bounds-check
            # before indexing and return 0 for OOB.
            in_slot = self._slot_of(op.operands[0])
            res_slot = self._slot_of(op.results[0])
            buf = self.asm.b
            # ecx = handle
            self.asm.mov_ecx_mem_rbp(in_slot)
            # cmp ecx, 0   (83 F9 00) ; jl bad   (7C rel8)
            buf.emit(0x83, 0xF9, 0x00)
            buf.emit(0x7C, 0x00)
            jl_off = buf.offset() - 1
            jl_after = buf.offset()
            # cmp ecx, HELIX_NUM_CELLS  (81 F9 imm32) ; jge bad  (7D rel8)
            buf.emit(0x81, 0xF9)
            buf.emit_bytes(struct.pack("<i", HELIX_NUM_CELLS))
            buf.emit(0x7D, 0x00)
            jge_off = buf.offset() - 1
            jge_after = buf.offset()
            # In-range path:
            # movsxd rcx, ecx
            buf.emit(0x48, 0x63, 0xC9)
            self.asm.lea_rax_rip_rel("__helix_state_base")
            self.asm.mov_rax_mem_rax_rcx8()
            # jmp done
            buf.emit(0xEB, 0x00)
            jmp_done_off = buf.offset() - 1
            jmp_done_after = buf.offset()
            # bad: eax = 0 (and rax = 0 — write through eax suffices since
            # mov eax, imm32 zero-extends into rax)
            bad_addr = buf.offset()
            self.asm.mov_eax_imm32(0)
            done_addr = buf.offset()
            # Patch jl→bad, jge→bad, jmp→done
            for d, off, name in (
                (bad_addr - jl_after, jl_off, "jl"),
                (bad_addr - jge_after, jge_off, "jge"),
                (done_addr - jmp_done_after, jmp_done_off, "jmp_done"),
            ):
                if not (-128 <= d <= 127):
                    raise ValueError(f"SPLICE {name} disp out of rel8: {d}")
                buf.bytes_[off] = d & 0xFF
            self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.MODIFY:
            # modify(handle, new_value, verifier_fn_name):
            #   call verifier(handle, new_value) → eax (truthy=accept)
            #   if accepted: state[handle] = new_value, return 1
            #   else: return 0
            handle_slot = self._slot_of(op.operands[0])
            new_val_slot = self._slot_of(op.operands[1])
            res_slot = self._slot_of(op.results[0])
            verifier_name = op.attrs.get("verifier_fn")
            if not isinstance(verifier_name, str):
                # Fallback: legacy behavior (just check verifier slot truthy).
                # Operand[2] in the legacy form was a runtime value.
                if len(op.operands) >= 3:
                    legacy_slot = self._slot_of(op.operands[2])
                    self.asm.mov_eax_mem_rbp(legacy_slot)
                    self.asm.test_eax_eax()
                    self.asm.setne_al()
                    self.asm.movzx_eax_al()
                    self.asm.mov_mem_rbp_eax(res_slot)
                else:
                    self.asm.mov_eax_imm32(0)
                    self.asm.mov_mem_rbp_eax(res_slot)
                return

            buf = self.asm.b
            # Bounds-check the handle BEFORE calling the verifier so that
            # OOB never reaches the cell array. ecx = handle.
            self.asm.mov_ecx_mem_rbp(handle_slot)
            buf.emit(0x83, 0xF9, 0x00)        # cmp ecx, 0
            buf.emit(0x7C, 0x00)              # jl bad
            jl_off = buf.offset() - 1
            jl_after = buf.offset()
            buf.emit(0x81, 0xF9)              # cmp ecx, HELIX_NUM_CELLS
            buf.emit_bytes(struct.pack("<i", HELIX_NUM_CELLS))
            buf.emit(0x7D, 0x00)              # jge bad
            jge_off = buf.offset() - 1
            jge_after = buf.offset()

            # In-range: pass args to verifier (edi=handle, esi=new_value)
            self.asm.mov_edi_mem_rbp(handle_slot)
            self.asm.mov_esi_mem_rbp(new_val_slot)
            self.asm.call_rel32(verifier_name)
            # eax now holds verifier's return value
            self.asm.test_eax_eax()
            # je skip_apply (rel8 placeholder)
            buf.emit(0x74, 0x00)
            je_off = buf.offset() - 1
            je_after = buf.offset()

            # Apply: state[handle] = new_value (sign-extend 32-bit value to 64)
            # rcx = handle (sign-extended)
            self.asm.mov_ecx_mem_rbp(handle_slot)
            buf.emit(0x48, 0x63, 0xC9)   # movsxd rcx, ecx
            # rdx = new_value (sign-extended)
            self.asm.mov_eax_mem_rbp(new_val_slot)
            # movsxd rdx, eax: 48 63 D0
            buf.emit(0x48, 0x63, 0xD0)
            # rax = state base address
            self.asm.lea_rax_rip_rel("__helix_state_base")
            # [rax + rcx*8] = rdx
            self.asm.mov_mem_rax_rcx8_rdx()
            # eax = 1 (success)
            self.asm.mov_eax_imm32(1)
            # jmp done (rel8)
            buf.emit(0xEB, 0x00)
            jmp_done_off = buf.offset() - 1
            jmp_done_after = buf.offset()

            # skip_apply: eax = 0  (verifier rejected)
            skip_addr = buf.offset()
            self.asm.mov_eax_imm32(0)
            # jmp done — share the OOB path's done since both produce eax=0
            buf.emit(0xEB, 0x00)
            skip_jmp_off = buf.offset() - 1
            skip_jmp_after = buf.offset()

            # bad (OOB handle): eax = 0
            bad_addr = buf.offset()
            self.asm.mov_eax_imm32(0)
            done_addr = buf.offset()

            # Patch jumps:
            #   je_off       → skip_addr  (verifier returned 0)
            #   jmp_done_off → done_addr  (apply branch jumps over both fallbacks)
            #   jl_off       → bad_addr   (handle < 0)
            #   jge_off      → bad_addr   (handle >= NUM_CELLS)
            #   skip_jmp_off → done_addr  (after eax=0, fall through to done)
            jumps = (
                (skip_addr - je_after, je_off, "je_skip"),
                (done_addr - jmp_done_after, jmp_done_off, "jmp_done"),
                (bad_addr - jl_after, jl_off, "jl_bad"),
                (bad_addr - jge_after, jge_off, "jge_bad"),
                (done_addr - skip_jmp_after, skip_jmp_off, "skip_to_done"),
            )
            for d, off, name in jumps:
                if not (-128 <= d <= 127):
                    raise ValueError(f"MODIFY {name} disp out of rel8: {d}")
                buf.bytes_[off] = d & 0xFF

            self.asm.mov_mem_rbp_eax(res_slot)
            return

        # Arrays
        if op.kind == tir.OpKind.ALLOC_ARRAY:
            return  # already pre-allocated
        if op.kind == tir.OpKind.LOAD_ELEM:
            name = op.attrs["name"]
            base, length, esize = self.array_info[name]
            # Index is the operand
            idx_slot = self._slot_of(op.operands[0])
            res_slot = self._slot_of(op.results[0])
            # Compute address: rcx = idx * 8; rdx = rbp + base; eax = [rdx + rcx]
            # Simpler: use rcx as index in 64-bit, scale via [rbp + rcx*8 + base]
            # mov ecx, [rbp + idx_slot]    (8B 4D <disp>)
            self.asm.mov_ecx_mem_rbp(idx_slot)
            # movsxd rcx, ecx (sign-extend ecx to rcx)  48 63 C9
            self.asm.b.emit(0x48, 0x63, 0xC9)
            # mov eax, [rbp + rcx*8 + base]
            # 8B 84 CD <disp32>   mov eax, [rbp + rcx*8 + disp32]
            self.asm.b.emit(0x8B, 0x84, 0xCD)
            self.asm.b.emit_bytes(struct.pack("<i", base))
            self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.STORE_ELEM:
            name = op.attrs["name"]
            base, length, esize = self.array_info[name]
            idx_slot = self._slot_of(op.operands[0])
            val_slot = self._slot_of(op.operands[1])
            # rcx = idx (sign-extended)
            self.asm.mov_ecx_mem_rbp(idx_slot)
            self.asm.b.emit(0x48, 0x63, 0xC9)
            # eax = value
            self.asm.mov_eax_mem_rbp(val_slot)
            # mov [rbp + rcx*8 + base], eax
            # 89 84 CD <disp32>
            self.asm.b.emit(0x89, 0x84, 0xCD)
            self.asm.b.emit_bytes(struct.pack("<i", base))
            return
        # Unsupported op — emit nothing (placeholder); v0.2 will lower
        # tensor ops to runtime calls.


# ============================================================================
# ELF emission
# ============================================================================
def emit_elf(code: bytes, entry_offset: int = ENTRY_OFFSET) -> bytes:
    """Wrap the given code bytes in a minimal x86-64 Linux ELF executable.

    Note: the segment is mapped R+W+X (flag = 7) so the reflection-cells
    region embedded at the end of `code` can be modified at runtime by
    MODIFY ops. This is a deliberate choice for an experimental compiler;
    a production version would split into separate R-X .text and R-W .data
    segments to enforce W^X.
    """
    # Layout:
    #   0x00: ELF header (64 B)
    #   0x40: Program header (56 B)
    #   0x1000: Code + reflection cells (page-aligned)
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

    # Program header (PT_LOAD, R+W+X — see docstring above)
    phdr = struct.pack("<I", 1)               # p_type = PT_LOAD
    phdr += struct.pack("<I", 7)              # p_flags = R | W | X
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

    # Reflection cells: append HELIX_NUM_CELLS * 8 bytes of zero-init storage
    # immediately after the code. The base symbol is __helix_state_base.
    # MODIFY/SPLICE codegen uses RIP-relative LEA to address it.
    buf.define_symbol("__helix_state_base")
    buf.emit_bytes(b"\x00" * (HELIX_NUM_CELLS * HELIX_CELL_SIZE))

    buf.patch()
    return emit_elf(bytes(buf.bytes_))


if __name__ == "__main__":
    import sys
    from ..frontend.parser import parse
    from ..frontend.typecheck import typecheck
    from ..frontend.grad_pass import grad_pass
    from ..ir.lower_ast import lower
    from ..ir.passes.const_fold import fold_module
    from ..ir.passes.dce import dce_module
    from ..ir.passes.cse import cse_module

    if len(sys.argv) < 3:
        print("usage: python -m helixc.backend.x86_64 <input.hx> <output.bin> [--strict] [--no-opt]",
              file=sys.stderr)
        sys.exit(1)
    strict = "--strict" in sys.argv
    no_opt = "--no-opt" in sys.argv
    with open(sys.argv[1]) as f:
        src = f.read()
    prog = parse(src)
    # Pre-pass: rewrite `grad(f)` calls into references to generated f__grad
    # functions. Adds new FnDecls to the program.
    grad_count = grad_pass(prog)
    if grad_count > 0:
        print(f"grad: {grad_count} grad(f) call(s) rewritten", file=sys.stderr)
    # Type-check; print as warnings, abort if --strict
    type_errors = typecheck(prog)
    if type_errors:
        for e in type_errors:
            print(f"warning: {e}", file=sys.stderr)
        if strict:
            print(f"\n{len(type_errors)} type error(s); --strict aborts.",
                  file=sys.stderr)
            sys.exit(1)
        else:
            print(f"\n({len(type_errors)} type warning(s); compiling anyway. "
                  f"Use --strict to fail on warnings.)", file=sys.stderr)
    mod = lower(prog)
    # Optimization passes (run twice — fold can expose new CSE opportunities, etc.)
    if not no_opt:
        folded = fold_module(mod)
        if folded > 0:
            print(f"const-fold: {folded} ops folded", file=sys.stderr)
        cse_count = cse_module(mod)
        if cse_count > 0:
            print(f"cse: {cse_count} duplicate ops merged", file=sys.stderr)
        removed = dce_module(mod)
        if removed > 0:
            print(f"dce: {removed} ops removed", file=sys.stderr)
    elf = compile_module_to_elf(mod)
    with open(sys.argv[2], "wb") as f:
        f.write(elf)
    import os
    os.chmod(sys.argv[2], 0o755)
    print(f"Wrote {sys.argv[2]} ({len(elf)} bytes)")
