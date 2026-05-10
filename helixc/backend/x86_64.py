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
from . import elf_dyn


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
# Arena capacity: 32K i32 slots = 128KB. Slot 0 reserved for the cursor;
# slots 1..HELIX_ARENA_CAP available for user data. Sized to fit a self-
# hosted compiler's working set (AST nodes + IR ops + symbol table) for
# small-to-medium programs without reallocation.
# 2 097 152 slots × 4 bytes = 8 MB BSS arena. Sized for self-host:
# the bootstrap source (lexer + parser + kovc, ~111 KB) lands as
# 111 K slots; each Helix source byte gets pushed as a one-byte
# value into a full i32 slot. Tokens add ~30 K * 4 = 120 K slots;
# AST adds ~5 K nodes * 5 slots = 25 K. ELF output is ~30 K. Total
# ~290 K slots — well under 2 M with room for compile-time state
# (fn_table, patch_table, str_state). The arena lives in BSS so
# the cap bump doesn't inflate produced binary file sizes.
HELIX_ARENA_CAP = 2097152
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
class FFIFixup:
    """Stage 16.5: pending FFI call site (`FF 15 <rel32>`).

    The 4 zero-bytes at `offset..offset+4` need to be patched once the
    .got.plt vaddr is known. The displacement is:
        got_vaddr_for(symbol) - (code_vaddr + offset + 4)
    `code_vaddr` is the absolute load vaddr of the code segment's start
    (NOT the code_offset within the file). The codegen records the
    file-relative offset; the patcher converts using the layout planner.
    """
    offset: int       # byte offset within code buffer
    symbol: str       # FFI symbol name (e.g. "puts")


@dataclass
class CodeBuf:
    bytes_: bytearray = field(default_factory=bytearray)
    symbols: dict[str, int] = field(default_factory=dict)   # function name -> offset
    fixups: list[Fixup] = field(default_factory=list)
    # Stage 16.5: pending FFI call sites.
    ffi_fixups: list[FFIFixup] = field(default_factory=list)
    # Stage 16.5: collected DynLinkInfo (lives on the buf so codegen
    # ops in any function can record imports and the driver can pick
    # them up at finalize time).
    dyn: "elf_dyn.DynLinkInfo" = field(default_factory=lambda: elf_dyn.DynLinkInfo())

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

    # ============================================================
    # 64-bit (i64) integer operations — REX.W + same opcodes.
    # Phase 1.4: full native-width arithmetic.
    # ============================================================
    def mov_rax_imm64(self, imm: int) -> None:
        # 48 B8 imm64
        self.b.emit(0x48, 0xB8)
        self.b.emit_bytes(struct.pack("<q", imm))

    def mov_rax_mem_rbp(self, disp: int) -> None:
        # 48 8B 45 disp8 / 48 8B 85 disp32
        if -128 <= disp <= 127:
            self.b.emit(0x48, 0x8B, 0x45, disp & 0xFF)
        else:
            self.b.emit(0x48, 0x8B, 0x85)
            self.b.emit_bytes(struct.pack("<i", disp))

    def mov_mem_rbp_rax(self, disp: int) -> None:
        # 48 89 45 disp8 / 48 89 85 disp32
        if -128 <= disp <= 127:
            self.b.emit(0x48, 0x89, 0x45, disp & 0xFF)
        else:
            self.b.emit(0x48, 0x89, 0x85)
            self.b.emit_bytes(struct.pack("<i", disp))

    def mov_rcx_mem_rbp(self, disp: int) -> None:
        if -128 <= disp <= 127:
            self.b.emit(0x48, 0x8B, 0x4D, disp & 0xFF)
        else:
            self.b.emit(0x48, 0x8B, 0x8D)
            self.b.emit_bytes(struct.pack("<i", disp))

    def add_rax_rcx(self) -> None:
        self.b.emit(0x48, 0x01, 0xC8)          # add rax, rcx

    def sub_rax_rcx(self) -> None:
        self.b.emit(0x48, 0x29, 0xC8)          # sub rax, rcx

    def imul_rax_rcx(self) -> None:
        self.b.emit(0x48, 0x0F, 0xAF, 0xC1)    # imul rax, rcx

    def cqo(self) -> None:
        # 48 99   sign-extend rax into rdx:rax (the 64-bit sibling of cdq)
        self.b.emit(0x48, 0x99)

    def idiv_rcx(self) -> None:
        # 48 F7 F9
        self.b.emit(0x48, 0xF7, 0xF9)

    def cmp_rax_rcx(self) -> None:
        self.b.emit(0x48, 0x39, 0xC8)          # cmp rax, rcx

    def neg_rax(self) -> None:
        self.b.emit(0x48, 0xF7, 0xD8)          # neg rax

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

    # ---- 64-bit (i64) arg-register loads (REX.W + same opcodes) ----
    def mov_rdi_mem_rbp(self, disp: int) -> None:
        # 48 8B 7D disp8 / 48 8B BD disp32
        if -128 <= disp <= 127:
            self.b.emit(0x48, 0x8B, 0x7D, disp & 0xFF)
        else:
            self.b.emit(0x48, 0x8B, 0xBD)
            self.b.emit_bytes(struct.pack("<i", disp))

    def mov_rsi_mem_rbp(self, disp: int) -> None:
        if -128 <= disp <= 127:
            self.b.emit(0x48, 0x8B, 0x75, disp & 0xFF)
        else:
            self.b.emit(0x48, 0x8B, 0xB5)
            self.b.emit_bytes(struct.pack("<i", disp))

    def mov_rdx_mem_rbp(self, disp: int) -> None:
        if -128 <= disp <= 127:
            self.b.emit(0x48, 0x8B, 0x55, disp & 0xFF)
        else:
            self.b.emit(0x48, 0x8B, 0x95)
            self.b.emit_bytes(struct.pack("<i", disp))

    def mov_rcx_arg_mem_rbp(self, disp: int) -> None:
        if -128 <= disp <= 127:
            self.b.emit(0x48, 0x8B, 0x4D, disp & 0xFF)
        else:
            self.b.emit(0x48, 0x8B, 0x8D)
            self.b.emit_bytes(struct.pack("<i", disp))

    def mov_r8_mem_rbp(self, disp: int) -> None:
        # 4C 8B 45 disp8 / 4C 8B 85 disp32
        if -128 <= disp <= 127:
            self.b.emit(0x4C, 0x8B, 0x45, disp & 0xFF)
        else:
            self.b.emit(0x4C, 0x8B, 0x85)
            self.b.emit_bytes(struct.pack("<i", disp))

    def mov_r9_mem_rbp(self, disp: int) -> None:
        if -128 <= disp <= 127:
            self.b.emit(0x4C, 0x8B, 0x4D, disp & 0xFF)
        else:
            self.b.emit(0x4C, 0x8B, 0x8D)
            self.b.emit_bytes(struct.pack("<i", disp))

    # ---- 64-bit arg-register STORE-to-stack (callee param spill) ----
    def mov_mem_rbp_rdi(self, disp: int) -> None:
        # 48 89 7D disp8 / 48 89 BD disp32
        if -128 <= disp <= 127:
            self.b.emit(0x48, 0x89, 0x7D, disp & 0xFF)
        else:
            self.b.emit(0x48, 0x89, 0xBD)
            self.b.emit_bytes(struct.pack("<i", disp))

    def mov_mem_rbp_rsi(self, disp: int) -> None:
        if -128 <= disp <= 127:
            self.b.emit(0x48, 0x89, 0x75, disp & 0xFF)
        else:
            self.b.emit(0x48, 0x89, 0xB5)
            self.b.emit_bytes(struct.pack("<i", disp))

    def mov_mem_rbp_rdx(self, disp: int) -> None:
        if -128 <= disp <= 127:
            self.b.emit(0x48, 0x89, 0x55, disp & 0xFF)
        else:
            self.b.emit(0x48, 0x89, 0x95)
            self.b.emit_bytes(struct.pack("<i", disp))

    def mov_mem_rbp_rcx(self, disp: int) -> None:
        if -128 <= disp <= 127:
            self.b.emit(0x48, 0x89, 0x4D, disp & 0xFF)
        else:
            self.b.emit(0x48, 0x89, 0x8D)
            self.b.emit_bytes(struct.pack("<i", disp))

    def mov_mem_rbp_r8(self, disp: int) -> None:
        if -128 <= disp <= 127:
            self.b.emit(0x4C, 0x89, 0x45, disp & 0xFF)
        else:
            self.b.emit(0x4C, 0x89, 0x85)
            self.b.emit_bytes(struct.pack("<i", disp))

    def mov_mem_rbp_r9(self, disp: int) -> None:
        if -128 <= disp <= 127:
            self.b.emit(0x4C, 0x89, 0x4D, disp & 0xFF)
        else:
            self.b.emit(0x4C, 0x89, 0x8D)
            self.b.emit_bytes(struct.pack("<i", disp))

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

    def call_qword_ptr_rip_rel_ffi(self, symbol: str) -> None:
        """Stage 16.5: indirect call through GOT entry (BIND_NOW).

        Emits `FF 15 <rel32>` — call qword ptr [rip + disp32]. The disp32
        is patched after layout planning to point at the symbol's slot
        in .got.plt. The dynamic linker pre-fills that slot with the
        resolved function address (BIND_NOW), so this becomes an
        indirect call to the actual library routine.
        """
        self.b.emit(0xFF, 0x15)
        offset = self.b.offset()
        self.b.emit_bytes(b"\x00\x00\x00\x00")
        self.b.ffi_fixups.append(FFIFixup(offset=offset, symbol=symbol))
        # Register the import so the ELF emitter knows to allocate a GOT slot.
        self.b.dyn.add_import(symbol)

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

    # --- Generic movss for arbitrary xmm0..xmm7 (for SysV float arg passing).
    # Encoding: F3 0F 10 (load) or 0F 11 (store), then ModRM with reg=xmmN.
    # ModRM encoding when r/m is [rbp + disp]: mod=01 disp8 / mod=10 disp32,
    # rm=101 (rbp). reg field is xmm number (0-7).
    def _movss_load_xmmN(self, n: int, disp: int) -> None:
        modrm_disp8 = (0b01 << 6) | (n << 3) | 0b101
        modrm_disp32 = (0b10 << 6) | (n << 3) | 0b101
        if -128 <= disp <= 127:
            self.b.emit(0xF3, 0x0F, 0x10, modrm_disp8, disp & 0xFF)
        else:
            self.b.emit(0xF3, 0x0F, 0x10, modrm_disp32)
            self.b.emit_bytes(struct.pack("<i", disp))

    def _movss_store_xmmN(self, n: int, disp: int) -> None:
        modrm_disp8 = (0b01 << 6) | (n << 3) | 0b101
        modrm_disp32 = (0b10 << 6) | (n << 3) | 0b101
        if -128 <= disp <= 127:
            self.b.emit(0xF3, 0x0F, 0x11, modrm_disp8, disp & 0xFF)
        else:
            self.b.emit(0xF3, 0x0F, 0x11, modrm_disp32)
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

    def ucomiss_xmm0_xmm1(self) -> None:
        # 0F 2E C1   ucomiss xmm0, xmm1 (unordered: SNaN doesn't raise #IA)
        # CF=1 if xmm0 < xmm1; ZF=1 if equal or unordered (NaN).
        self.b.emit(0x0F, 0x2E, 0xC1)

    # ============================================================
    # Double-precision (f64) SSE2 instructions — same shape as
    # the f32 (movss / addss / ...) ops above, but with the F2
    # prefix instead of F3 and 8-byte slot loads/stores. Phase-1.1.
    # ============================================================
    def movsd_xmm0_mem_rbp(self, disp: int) -> None:
        # F2 0F 10 45 disp8 (mod=01, reg=000=xmm0, r/m=101=rbp+disp)
        if -128 <= disp <= 127:
            self.b.emit(0xF2, 0x0F, 0x10, 0x45, disp & 0xFF)
        else:
            self.b.emit(0xF2, 0x0F, 0x10, 0x85)
            self.b.emit_bytes(struct.pack("<i", disp))

    def movsd_xmm1_mem_rbp(self, disp: int) -> None:
        if -128 <= disp <= 127:
            self.b.emit(0xF2, 0x0F, 0x10, 0x4D, disp & 0xFF)
        else:
            self.b.emit(0xF2, 0x0F, 0x10, 0x8D)
            self.b.emit_bytes(struct.pack("<i", disp))

    def movsd_mem_rbp_xmm0(self, disp: int) -> None:
        if -128 <= disp <= 127:
            self.b.emit(0xF2, 0x0F, 0x11, 0x45, disp & 0xFF)
        else:
            self.b.emit(0xF2, 0x0F, 0x11, 0x85)
            self.b.emit_bytes(struct.pack("<i", disp))

    def _movsd_load_xmmN(self, n: int, disp: int) -> None:
        modrm_disp8 = (0b01 << 6) | (n << 3) | 0b101
        modrm_disp32 = (0b10 << 6) | (n << 3) | 0b101
        if -128 <= disp <= 127:
            self.b.emit(0xF2, 0x0F, 0x10, modrm_disp8, disp & 0xFF)
        else:
            self.b.emit(0xF2, 0x0F, 0x10, modrm_disp32)
            self.b.emit_bytes(struct.pack("<i", disp))

    def _movsd_store_xmmN(self, n: int, disp: int) -> None:
        modrm_disp8 = (0b01 << 6) | (n << 3) | 0b101
        modrm_disp32 = (0b10 << 6) | (n << 3) | 0b101
        if -128 <= disp <= 127:
            self.b.emit(0xF2, 0x0F, 0x11, modrm_disp8, disp & 0xFF)
        else:
            self.b.emit(0xF2, 0x0F, 0x11, modrm_disp32)
            self.b.emit_bytes(struct.pack("<i", disp))

    def addsd_xmm0_xmm1(self) -> None:
        self.b.emit(0xF2, 0x0F, 0x58, 0xC1)

    def subsd_xmm0_xmm1(self) -> None:
        self.b.emit(0xF2, 0x0F, 0x5C, 0xC1)

    def mulsd_xmm0_xmm1(self) -> None:
        self.b.emit(0xF2, 0x0F, 0x59, 0xC1)

    def divsd_xmm0_xmm1(self) -> None:
        self.b.emit(0xF2, 0x0F, 0x5E, 0xC1)

    def cvttsd2si_eax_xmm0(self) -> None:
        # F2 0F 2C C0   (truncating f64 -> signed int32)
        self.b.emit(0xF2, 0x0F, 0x2C, 0xC0)

    def cvtsi2sd_xmm0_eax(self) -> None:
        # F2 0F 2A C0
        self.b.emit(0xF2, 0x0F, 0x2A, 0xC0)

    def ucomisd_xmm0_xmm1(self) -> None:
        # 66 0F 2E C1
        self.b.emit(0x66, 0x0F, 0x2E, 0xC1)

    def comiss_xmm0_xmm1(self) -> None:
        # NP 0F 2F C1
        self.b.emit(0x0F, 0x2F, 0xC1)

    def syscall(self) -> None:
        self.b.emit(0x0F, 0x05)

    # ---- imm-to-reg moves used by syscall plumbing ----
    def mov_edi_imm32(self, imm: int) -> None:
        # BF <imm32>
        self.b.emit(0xBF)
        self.b.emit_bytes(struct.pack("<I", imm & 0xFFFFFFFF))

    def mov_esi_imm32(self, imm: int) -> None:
        # BE <imm32>
        self.b.emit(0xBE)
        self.b.emit_bytes(struct.pack("<I", imm & 0xFFFFFFFF))

    def mov_edx_imm32(self, imm: int) -> None:
        # BA <imm32>
        self.b.emit(0xBA)
        self.b.emit_bytes(struct.pack("<I", imm & 0xFFFFFFFF))

    def lea_rsi_rip_rel(self, target: str) -> None:
        """lea rsi, [rip + disp32]   48 8D 35 <disp32>"""
        self.b.emit(0x48, 0x8D, 0x35)
        offset = self.b.offset()
        self.b.emit_bytes(b"\x00\x00\x00\x00")
        self.b.fixups.append(Fixup(offset=offset, target=target,
                                   size=4, rel_base=offset + 4))

    def lea_rdi_rip_rel(self, target: str) -> None:
        """lea rdi, [rip + disp32]   48 8D 3D <disp32> (Stage 16.5: FFI)"""
        self.b.emit(0x48, 0x8D, 0x3D)
        offset = self.b.offset()
        self.b.emit_bytes(b"\x00\x00\x00\x00")
        self.b.fixups.append(Fixup(offset=offset, target=target,
                                   size=4, rel_base=offset + 4))

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
    # Parity-flag setters for IEEE-NaN-aware float comparisons.
    # PF=1 after ucomiss iff either operand is NaN ("unordered").
    # setp/setnp into cl so we can AND/OR with the al result of the
    # primary cmp setter without clobbering al.
    def setp_cl(self) -> None:    self.b.emit(0x0F, 0x9A, 0xC1)
    def setnp_cl(self) -> None:   self.b.emit(0x0F, 0x9B, 0xC1)
    # 8-bit logical AND/OR between al and cl (used to combine the
    # primary cmp with the parity-flag check).
    def and_al_cl(self) -> None:  self.b.emit(0x20, 0xC8)
    def or_al_cl(self) -> None:   self.b.emit(0x08, 0xC8)

    # 32-bit bitwise ops: surface syntax `& | ^` lowered through TIR
    # BIT_AND/BIT_OR/BIT_XOR. Each takes (eax, ecx) -> eax.
    def and_eax_ecx(self) -> None:  self.b.emit(0x21, 0xC8)
    def or_eax_ecx(self) -> None:   self.b.emit(0x09, 0xC8)
    def xor_eax_ecx(self) -> None:  self.b.emit(0x31, 0xC8)
    # 64-bit variants for i64 operands.
    def and_rax_rcx(self) -> None:  self.b.emit(0x48, 0x21, 0xC8)
    def or_rax_rcx(self) -> None:   self.b.emit(0x48, 0x09, 0xC8)
    def xor_rax_rcx(self) -> None:  self.b.emit(0x48, 0x31, 0xC8)
    # Shifts. x86 shift instructions take the shift count in CL (low byte
    # of ECX). Caller must mov_ecx_mem_rbp(r_slot) before invoking.
    # SAR (arithmetic right shift) preserves the sign bit; SHR (logical)
    # zero-fills. Helix's `>>` is signed -> SAR.
    def shl_eax_cl(self) -> None:  self.b.emit(0xD3, 0xE0)
    def sar_eax_cl(self) -> None:  self.b.emit(0xD3, 0xF8)
    def shl_rax_cl(self) -> None:  self.b.emit(0x48, 0xD3, 0xE0)
    def sar_rax_cl(self) -> None:  self.b.emit(0x48, 0xD3, 0xF8)
    # Bitwise unary NOT (~): one's complement.
    def not_eax(self) -> None:  self.b.emit(0xF7, 0xD0)
    def not_rax(self) -> None:  self.b.emit(0x48, 0xF7, 0xD0)

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
        # Pending strings (sym, bytes) emitted by PRINT ops in this function;
        # collected by the module driver and appended to the binary.
        self._pending_strings: list[tuple[str, bytes]] = []

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
                self._check_float_supported(p.ty)
                self._alloc_slot(p)
            for op in blk.ops:
                for r in op.results:
                    self._check_float_supported(r.ty)
                    self._alloc_slot(r)
        # Pre-allocate slots for fn params (they share entry block params slot conceptually)
        for p in self.fn.params:
            self._check_float_supported(p.ty)
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

        # Spill args from arg registers into stack slots. SysV ABI splits by
        # type: int args land in (edi, esi, edx, ecx, r8d, r9d); float args
        # land in (xmm0..xmm7). Each class has its own counter.
        INT_SPILLS = [
            self.asm.mov_mem_rbp_edi,
            self.asm.mov_mem_rbp_esi,
            self.asm.mov_mem_rbp_edx,
            self.asm.mov_mem_rbp_ecx,
            self.asm.mov_mem_rbp_r8d,
            self.asm.mov_mem_rbp_r9d,
        ]
        INT_SPILLS_64 = [
            self.asm.mov_mem_rbp_rdi,
            self.asm.mov_mem_rbp_rsi,
            self.asm.mov_mem_rbp_rdx,
            self.asm.mov_mem_rbp_rcx,
            self.asm.mov_mem_rbp_r8,
            self.asm.mov_mem_rbp_r9,
        ]
        int_idx = 0
        xmm_idx = 0
        for p in self.fn.params:
            slot = self._slot_of(p)
            if self._is_float_type(p.ty):
                if xmm_idx >= 8:
                    raise NotImplementedError("v0.1 supports up to 8 float params")
                if self._is_f64_type(p.ty):
                    self.asm._movsd_store_xmmN(xmm_idx, slot)
                else:
                    self.asm._movss_store_xmmN(xmm_idx, slot)
                xmm_idx += 1
            else:
                if int_idx >= len(INT_SPILLS):
                    raise NotImplementedError(
                        f"v0.1 supports up to {len(INT_SPILLS)} int params"
                    )
                if self._is_i64_type(p.ty):
                    INT_SPILLS_64[int_idx](slot)
                else:
                    INT_SPILLS[int_idx](slot)
                int_idx += 1

        # Emit each block in order, with a label per block
        for blk in self.fn.blocks:
            block_label = f"{self.fn.name}_bb{blk.id}"
            self.asm.b.define_symbol(block_label)
            for op in blk.ops:
                self._emit_op(op, frame_size)

    def _is_float_type(self, ty: tir.TIRType) -> bool:
        return isinstance(ty, tir.TIRScalar) and ty.name in ("f16", "bf16", "f32", "f64")

    def _is_f64_type(self, ty: tir.TIRType) -> bool:
        return isinstance(ty, tir.TIRScalar) and ty.name == "f64"

    def _is_i64_type(self, ty: tir.TIRType) -> bool:
        return isinstance(ty, tir.TIRScalar) and ty.name == "i64"

    def _is_u64_type(self, ty: tir.TIRType) -> bool:
        # Stage 16.5: u64 is the IR type for raw pointers and FFI-arg widening.
        return isinstance(ty, tir.TIRScalar) and ty.name == "u64"

    def _check_float_supported(self, ty: tir.TIRType) -> None:
        """Phase 1 supports f32 and f64. f16/bf16 still need the F16C
        / AVX-512 paths — error on those. Treating them as f32 silently
        corrupts results, so we error explicitly."""
        if isinstance(ty, tir.TIRScalar) and ty.name in ("f16", "bf16"):
            raise NotImplementedError(
                f"x86_64 backend supports only f32 and f64 currently; "
                f"got '{ty.name}'. Either change to f32/f64 or implement "
                f"the F16C / AVX-512 codegen path."
            )

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

        # Bug C fix: guard div-by-zero. cmp ecx, 0; je zero_path.
        # zero_path produces eax=0 for both quotient and remainder
        # (matching common safe-divide convention; no SIGFPE).
        buf.emit(0x83, 0xF9, 0x00)              # cmp ecx, 0
        buf.emit(0x75, 0x05)                    # jne +5 (skip zero_path)
        # zero_path: xor eax, eax (2 bytes); jmp done (3 bytes — rel8)
        buf.emit(0x31, 0xC0)                    # xor eax, eax
        # We'll patch the jmp_done offset below; emit placeholder.
        buf.emit(0xEB, 0x00)                    # jmp done placeholder
        zero_jmp_disp_off = buf.offset() - 1
        zero_jmp_after = buf.offset()

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

        # Patch the four rel8 jumps (incl. the new zero_jmp).
        d0 = done_addr - zero_jmp_after
        d1 = do_div_addr - jne1_after
        d2 = do_div_addr - jne2_after
        d3 = done_addr - jmp_done_after
        for d, name in ((d0, "zero_jmp"), (d1, "jne1"),
                        (d2, "jne2"), (d3, "jmp_done")):
            if not (-128 <= d <= 127):
                raise ValueError(f"idiv-guard {name} disp out of rel8: {d}")
        buf.bytes_[zero_jmp_disp_off] = d0 & 0xFF
        buf.bytes_[jne1_disp_off] = d1 & 0xFF
        buf.bytes_[jne2_disp_off] = d2 & 0xFF
        buf.bytes_[jmp_done_disp_off] = d3 & 0xFF

        self.asm.mov_mem_rbp_eax(res_slot)

    def _emit_op(self, op: tir.Op, frame_size: int) -> None:
        if op.kind == tir.OpKind.CONST_INT:
            slot = self._slot_of(op.results[0])
            value = int(op.attrs["value"])
            if self._is_i64_type(op.results[0].ty):
                self.asm.mov_rax_imm64(value)
                self.asm.mov_mem_rbp_rax(slot)
            else:
                self.asm.mov_eax_imm32(value & 0xFFFFFFFF)
                self.asm.mov_mem_rbp_eax(slot)
            return
        if op.kind == tir.OpKind.CONST_BOOL:
            # bool is stored as i32: 0 for false, 1 for true.
            slot = self._slot_of(op.results[0])
            self.asm.mov_eax_imm32(1 if bool(op.attrs["value"]) else 0)
            self.asm.mov_mem_rbp_eax(slot)
            return
        if op.kind == tir.OpKind.CONST_FLOAT:
            slot = self._slot_of(op.results[0])
            value = float(op.attrs["value"])
            if self._is_f64_type(op.results[0].ty):
                # Pack as 8 bytes; store via two 32-bit moves (lo then hi)
                bits64 = struct.unpack("<Q", struct.pack("<d", value))[0]
                lo = bits64 & 0xFFFFFFFF
                hi = (bits64 >> 32) & 0xFFFFFFFF
                self.asm.mov_eax_imm32(lo)
                self.asm.mov_mem_rbp_eax(slot)
                self.asm.mov_eax_imm32(hi)
                self.asm.mov_mem_rbp_eax(slot + 4)
            else:
                bits = struct.unpack("<I", struct.pack("<f", value))[0]
                self.asm.mov_eax_imm32(bits)
                self.asm.mov_mem_rbp_eax(slot)
            return
        if op.kind == tir.OpKind.BITCAST:
            # Bit-level reinterpret: same bytes, different type label.
            # f32 <-> i32: 4-byte mov; f64 <-> i64: 8-byte mov.
            src_slot = self._slot_of(op.operands[0])
            res_slot = self._slot_of(op.results[0])
            res_ty = op.results[0].ty
            wide = self._is_f64_type(res_ty) or self._is_i64_type(res_ty) \
                   or self._is_f64_type(op.operands[0].ty) \
                   or self._is_i64_type(op.operands[0].ty)
            if wide:
                self.asm.mov_rax_mem_rbp(src_slot)
                self.asm.mov_mem_rbp_rax(res_slot)
            else:
                self.asm.mov_eax_mem_rbp(src_slot)
                self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.CAST:
            src_slot = self._slot_of(op.operands[0])
            res_slot = self._slot_of(op.results[0])
            from_ty = op.operands[0].ty
            to_ty = op.results[0].ty
            from_is_f64 = self._is_f64_type(from_ty)
            to_is_f64 = self._is_f64_type(to_ty)
            from_is_float = self._is_float_type(from_ty)
            to_is_float = self._is_float_type(to_ty)
            from_is_i64 = self._is_i64_type(from_ty)
            to_is_i64 = self._is_i64_type(to_ty)
            # i64 -> i32: just take low 32 bits via 32-bit mov.
            if from_is_i64 and not to_is_float and not to_is_i64:
                self.asm.mov_eax_mem_rbp(src_slot)
                self.asm.mov_mem_rbp_eax(res_slot)
                return
            # i32 -> i64: load 32-bit, sign-extend via movsxd rax, eax.
            if not from_is_float and not from_is_i64 and to_is_i64:
                self.asm.mov_eax_mem_rbp(src_slot)
                # movsxd rax, eax = 48 63 C0 (sign-extend 32->64)
                self.asm.b.emit(0x48, 0x63, 0xC0)
                self.asm.mov_mem_rbp_rax(res_slot)
                return
            # i64 -> f64: cvtsi2sd with REX.W.
            if from_is_i64 and to_is_f64:
                self.asm.mov_rax_mem_rbp(src_slot)
                # F2 48 0F 2A C0 = cvtsi2sd xmm0, rax
                self.asm.b.emit(0xF2, 0x48, 0x0F, 0x2A, 0xC0)
                self.asm.movsd_mem_rbp_xmm0(res_slot)
                return
            # i64 -> i64: copy 8 bytes
            if from_is_i64 and to_is_i64:
                self.asm.mov_rax_mem_rbp(src_slot)
                self.asm.mov_mem_rbp_rax(res_slot)
                return
            # i32 -> f64
            if not from_is_float and to_is_f64:
                self.asm.mov_eax_mem_rbp(src_slot)
                self.asm.cvtsi2sd_xmm0_eax()
                self.asm.movsd_mem_rbp_xmm0(res_slot)
                return
            # i32 -> f32
            if not from_is_float and to_is_float:
                self.asm.mov_eax_mem_rbp(src_slot)
                self.asm.cvtsi2ss_xmm0_eax()
                self.asm.movss_mem_rbp_xmm0(res_slot)
                return
            # f64 -> i32
            if from_is_f64 and not to_is_float:
                self.asm.movsd_xmm0_mem_rbp(src_slot)
                self.asm.cvttsd2si_eax_xmm0()
                self.asm.mov_mem_rbp_eax(res_slot)
                return
            # f32 -> i32
            if from_is_float and not to_is_float:
                self.asm.movss_xmm0_mem_rbp(src_slot)
                self.asm.cvttss2si_eax_xmm0()
                self.asm.mov_mem_rbp_eax(res_slot)
                return
            # Same float-or-not: memory copy. For f64-to-f64, copy 8 bytes.
            if from_is_f64 and to_is_f64:
                self.asm.mov_eax_mem_rbp(src_slot)
                self.asm.mov_mem_rbp_eax(res_slot)
                self.asm.mov_eax_mem_rbp(src_slot + 4)
                self.asm.mov_mem_rbp_eax(res_slot + 4)
                return
            if from_is_float == to_is_float:
                self.asm.mov_eax_mem_rbp(src_slot)
                self.asm.mov_mem_rbp_eax(res_slot)
                return
            return
        if op.kind == tir.OpKind.ADD:
            l_slot = self._slot_of(op.operands[0])
            r_slot = self._slot_of(op.operands[1])
            res_slot = self._slot_of(op.results[0])
            if self._is_f64_type(op.results[0].ty):
                self.asm.movsd_xmm0_mem_rbp(l_slot)
                self.asm.movsd_xmm1_mem_rbp(r_slot)
                self.asm.addsd_xmm0_xmm1()
                self.asm.movsd_mem_rbp_xmm0(res_slot)
            elif self._is_float_type(op.results[0].ty):
                self.asm.movss_xmm0_mem_rbp(l_slot)
                self.asm.movss_xmm1_mem_rbp(r_slot)
                self.asm.addss_xmm0_xmm1()
                self.asm.movss_mem_rbp_xmm0(res_slot)
            elif self._is_i64_type(op.results[0].ty):
                self.asm.mov_rax_mem_rbp(l_slot)
                self.asm.mov_rcx_mem_rbp(r_slot)
                self.asm.add_rax_rcx()
                self.asm.mov_mem_rbp_rax(res_slot)
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
            if self._is_f64_type(op.results[0].ty):
                self.asm.movsd_xmm0_mem_rbp(l_slot)
                self.asm.movsd_xmm1_mem_rbp(r_slot)
                self.asm.subsd_xmm0_xmm1()
                self.asm.movsd_mem_rbp_xmm0(res_slot)
            elif self._is_float_type(op.results[0].ty):
                self.asm.movss_xmm0_mem_rbp(l_slot)
                self.asm.movss_xmm1_mem_rbp(r_slot)
                self.asm.subss_xmm0_xmm1()
                self.asm.movss_mem_rbp_xmm0(res_slot)
            elif self._is_i64_type(op.results[0].ty):
                self.asm.mov_rax_mem_rbp(l_slot)
                self.asm.mov_rcx_mem_rbp(r_slot)
                self.asm.sub_rax_rcx()
                self.asm.mov_mem_rbp_rax(res_slot)
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
            if self._is_f64_type(op.results[0].ty):
                self.asm.movsd_xmm0_mem_rbp(l_slot)
                self.asm.movsd_xmm1_mem_rbp(r_slot)
                self.asm.mulsd_xmm0_xmm1()
                self.asm.movsd_mem_rbp_xmm0(res_slot)
            elif self._is_float_type(op.results[0].ty):
                self.asm.movss_xmm0_mem_rbp(l_slot)
                self.asm.movss_xmm1_mem_rbp(r_slot)
                self.asm.mulss_xmm0_xmm1()
                self.asm.movss_mem_rbp_xmm0(res_slot)
            elif self._is_i64_type(op.results[0].ty):
                self.asm.mov_rax_mem_rbp(l_slot)
                self.asm.mov_rcx_mem_rbp(r_slot)
                self.asm.imul_rax_rcx()
                self.asm.mov_mem_rbp_rax(res_slot)
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
            if self._is_f64_type(op.results[0].ty):
                self.asm.movsd_xmm0_mem_rbp(l_slot)
                self.asm.movsd_xmm1_mem_rbp(r_slot)
                self.asm.divsd_xmm0_xmm1()
                self.asm.movsd_mem_rbp_xmm0(res_slot)
            elif self._is_float_type(op.results[0].ty):
                self.asm.movss_xmm0_mem_rbp(l_slot)
                self.asm.movss_xmm1_mem_rbp(r_slot)
                self.asm.divss_xmm0_xmm1()
                self.asm.movss_mem_rbp_xmm0(res_slot)
            elif self._is_i64_type(op.results[0].ty):
                # i64 division: cqo + idiv rcx. Skip the INT_MIN/-1 guard
                # for now (rare edge case); div-by-zero still SIGFPEs.
                self.asm.mov_rax_mem_rbp(l_slot)
                self.asm.mov_rcx_mem_rbp(r_slot)
                self.asm.cqo()
                self.asm.idiv_rcx()
                self.asm.mov_mem_rbp_rax(res_slot)
            else:
                self._emit_idiv_guarded(l_slot, r_slot, res_slot, want_quotient=True)
            return
        if op.kind == tir.OpKind.MOD:
            l_slot = self._slot_of(op.operands[0])
            r_slot = self._slot_of(op.operands[1])
            res_slot = self._slot_of(op.results[0])
            if self._is_i64_type(op.results[0].ty):
                # i64 modulo: cqo + idiv rcx; remainder lands in rdx.
                # Match the i64 DIV path: skip the INT_MIN/-1 guard
                # (rare edge case); div-by-zero still SIGFPEs.
                self.asm.mov_rax_mem_rbp(l_slot)
                self.asm.mov_rcx_mem_rbp(r_slot)
                self.asm.cqo()
                self.asm.idiv_rcx()
                self.asm.mov_mem_rbp_rdx(res_slot)
                return
            self._emit_idiv_guarded(l_slot, r_slot, res_slot, want_quotient=False)
            return
        # Bitwise integer ops: 32-bit and-eax-ecx / or-eax-ecx / xor-eax-ecx,
        # 64-bit AND/OR/XOR via REX.W variants. Float operands are nonsense
        # for bitwise (caller's typecheck rejects them); we still default to
        # 32-bit emission for safety.
        if op.kind == tir.OpKind.BIT_AND:
            l_slot = self._slot_of(op.operands[0])
            r_slot = self._slot_of(op.operands[1])
            res_slot = self._slot_of(op.results[0])
            if self._is_i64_type(op.results[0].ty):
                self.asm.mov_rax_mem_rbp(l_slot)
                self.asm.mov_rcx_mem_rbp(r_slot)
                self.asm.and_rax_rcx()
                self.asm.mov_mem_rbp_rax(res_slot)
            else:
                self.asm.mov_eax_mem_rbp(l_slot)
                self.asm.mov_ecx_mem_rbp(r_slot)
                self.asm.and_eax_ecx()
                self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.BIT_OR:
            l_slot = self._slot_of(op.operands[0])
            r_slot = self._slot_of(op.operands[1])
            res_slot = self._slot_of(op.results[0])
            if self._is_i64_type(op.results[0].ty):
                self.asm.mov_rax_mem_rbp(l_slot)
                self.asm.mov_rcx_mem_rbp(r_slot)
                self.asm.or_rax_rcx()
                self.asm.mov_mem_rbp_rax(res_slot)
            else:
                self.asm.mov_eax_mem_rbp(l_slot)
                self.asm.mov_ecx_mem_rbp(r_slot)
                self.asm.or_eax_ecx()
                self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.BIT_XOR:
            l_slot = self._slot_of(op.operands[0])
            r_slot = self._slot_of(op.operands[1])
            res_slot = self._slot_of(op.results[0])
            if self._is_i64_type(op.results[0].ty):
                self.asm.mov_rax_mem_rbp(l_slot)
                self.asm.mov_rcx_mem_rbp(r_slot)
                self.asm.xor_rax_rcx()
                self.asm.mov_mem_rbp_rax(res_slot)
            else:
                self.asm.mov_eax_mem_rbp(l_slot)
                self.asm.mov_ecx_mem_rbp(r_slot)
                self.asm.xor_eax_ecx()
                self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.SHL:
            l_slot = self._slot_of(op.operands[0])
            r_slot = self._slot_of(op.operands[1])
            res_slot = self._slot_of(op.results[0])
            if self._is_i64_type(op.results[0].ty):
                self.asm.mov_rax_mem_rbp(l_slot)
                self.asm.mov_rcx_mem_rbp(r_slot)
                self.asm.shl_rax_cl()
                self.asm.mov_mem_rbp_rax(res_slot)
            else:
                self.asm.mov_eax_mem_rbp(l_slot)
                self.asm.mov_ecx_mem_rbp(r_slot)
                self.asm.shl_eax_cl()
                self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.SHR:
            l_slot = self._slot_of(op.operands[0])
            r_slot = self._slot_of(op.operands[1])
            res_slot = self._slot_of(op.results[0])
            if self._is_i64_type(op.results[0].ty):
                self.asm.mov_rax_mem_rbp(l_slot)
                self.asm.mov_rcx_mem_rbp(r_slot)
                self.asm.sar_rax_cl()
                self.asm.mov_mem_rbp_rax(res_slot)
            else:
                self.asm.mov_eax_mem_rbp(l_slot)
                self.asm.mov_ecx_mem_rbp(r_slot)
                self.asm.sar_eax_cl()
                self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.BIT_NOT:
            slot = self._slot_of(op.operands[0])
            res_slot = self._slot_of(op.results[0])
            if self._is_i64_type(op.results[0].ty):
                self.asm.mov_rax_mem_rbp(slot)
                self.asm.not_rax()
                self.asm.mov_mem_rbp_rax(res_slot)
            else:
                self.asm.mov_eax_mem_rbp(slot)
                self.asm.not_eax()
                self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.NEG:
            slot = self._slot_of(op.operands[0])
            res_slot = self._slot_of(op.results[0])
            ty = op.operands[0].ty
            if self._is_i64_type(ty):
                self.asm.mov_rax_mem_rbp(slot)
                self.asm.neg_rax()
                self.asm.mov_mem_rbp_rax(res_slot)
                return
            if self._is_f64_type(ty):
                # f64 negation: copy 8 bytes, then flip sign bit (bit 63
                # = high bit of byte 7). Avoid integer neg semantics.
                self.asm.mov_eax_mem_rbp(slot)
                self.asm.mov_mem_rbp_eax(res_slot)
                self.asm.mov_eax_mem_rbp(slot + 4)
                self.asm.mov_mem_rbp_eax(res_slot + 4)
                # xor BYTE PTR [rbp + res_slot + 7], 0x80
                disp = res_slot + 7
                if -128 <= disp <= 127:
                    self.asm.b.emit(0x80, 0x75, disp & 0xFF, 0x80)
                else:
                    self.asm.b.emit(0x80, 0xB5)
                    self.asm.b.emit_bytes(struct.pack("<i", disp))
                    self.asm.b.emit(0x80)
                return
            if self._is_float_type(ty):
                # f32 negation: copy 4 bytes, flip sign bit at byte +3.
                self.asm.mov_eax_mem_rbp(slot)
                self.asm.mov_mem_rbp_eax(res_slot)
                disp = res_slot + 3
                if -128 <= disp <= 127:
                    self.asm.b.emit(0x80, 0x75, disp & 0xFF, 0x80)
                else:
                    self.asm.b.emit(0x80, 0xB5)
                    self.asm.b.emit_bytes(struct.pack("<i", disp))
                    self.asm.b.emit(0x80)
                return
            # Integer NEG: two's-complement, neg eax.
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
                if (self._is_f64_type(op.operands[0].ty) or
                        self._is_f64_type(op.operands[1].ty)):
                    self.asm.movsd_xmm0_mem_rbp(l_slot)
                    self.asm.movsd_xmm1_mem_rbp(r_slot)
                    self.asm.ucomisd_xmm0_xmm1()
                else:
                    self.asm.movss_xmm0_mem_rbp(l_slot)
                    self.asm.movss_xmm1_mem_rbp(r_slot)
                    # ucomiss (0F 2E) — unordered compare; SNaN inputs don't
                    # raise #IA. comiss (0F 2F) is also available but only
                    # differs on SNaN exception behavior, which we don't need.
                    self.asm.ucomiss_xmm0_xmm1()
                float_cmp_setters[op.kind]()
                # IEEE 754 NaN handling: ucomiss with NaN sets ZF=1, PF=1,
                # CF=1 (the "unordered" combination). The base setters above
                # would erroneously fire for several relations:
                #   sete  (CMP_EQ): would say NaN==NaN (wrong; should be 0)
                #   setne (CMP_NE): would say NaN!=NaN false (wrong; should be 1)
                #   setb  (CMP_LT): would say NaN<x true (wrong; should be 0)
                #   setbe (CMP_LE): would say NaN<=x true (wrong; should be 0)
                # The seta/setae setters already produce 0 in the NaN case
                # (CF=1 makes them fail), so they need no fixup.
                #
                # Fix: AND/OR the al result with a parity-based guard.
                if op.kind == tir.OpKind.CMP_EQ:
                    # ordered AND equal: al &= !PF
                    self.asm.setnp_cl()
                    self.asm.and_al_cl()
                elif op.kind == tir.OpKind.CMP_NE:
                    # not-equal OR unordered: al |= PF
                    self.asm.setp_cl()
                    self.asm.or_al_cl()
                elif op.kind in (tir.OpKind.CMP_LT, tir.OpKind.CMP_LE):
                    # ordered AND (less / less-or-equal): al &= !PF
                    self.asm.setnp_cl()
                    self.asm.and_al_cl()
            elif self._is_i64_type(op.operands[0].ty) or self._is_i64_type(op.operands[1].ty):
                self.asm.mov_rax_mem_rbp(l_slot)
                self.asm.mov_rcx_mem_rbp(r_slot)
                self.asm.cmp_rax_rcx()
                int_cmp_setters[op.kind]()
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
            res_ty = op.results[0].ty
            is_f64 = self._is_f64_type(res_ty)
            is_i64 = self._is_i64_type(res_ty)
            buf = self.asm.b
            # mov eax, [cond]
            self.asm.mov_eax_mem_rbp(cond_slot)
            # test eax, eax
            buf.emit(0x85, 0xC0)
            # je rel8 — write 0x74 + placeholder, remember placeholder offset
            buf.emit(0x74, 0x00)
            je_disp_off = buf.offset() - 1
            je_after = buf.offset()
            # load a (use 64-bit mov / movsd for f64/i64 so all 8 bytes flow)
            if is_f64:
                self.asm._movsd_load_xmmN(0, a_slot)
            elif is_i64:
                self.asm.mov_rax_mem_rbp(a_slot)
            else:
                self.asm.mov_eax_mem_rbp(a_slot)
            # jmp rel8 — placeholder
            buf.emit(0xEB, 0x00)
            jmp_disp_off = buf.offset() - 1
            jmp_after = buf.offset()
            # SKIP_A: load b
            skip_a_addr = buf.offset()
            if is_f64:
                self.asm._movsd_load_xmmN(0, b_slot)
            elif is_i64:
                self.asm.mov_rax_mem_rbp(b_slot)
            else:
                self.asm.mov_eax_mem_rbp(b_slot)
            end_addr = buf.offset()
            # Patch je: skip past (load_a + jmp), targeting skip_a_addr
            je_disp = skip_a_addr - je_after
            jmp_disp = end_addr - jmp_after
            if not (-128 <= je_disp <= 127) or not (-128 <= jmp_disp <= 127):
                raise ValueError(
                    f"SELECT branch displacement out of rel8 range: "
                    f"je={je_disp}, jmp={jmp_disp}"
                )
            buf.bytes_[je_disp_off] = je_disp & 0xFF
            buf.bytes_[jmp_disp_off] = jmp_disp & 0xFF
            # store result (64-bit for wide types)
            if is_f64:
                self.asm.movsd_mem_rbp_xmm0(res_slot)
            elif is_i64:
                self.asm.mov_mem_rbp_rax(res_slot)
            else:
                self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.CALL:
            target = op.attrs.get("target", "?")
            INT_REGS = [
                self.asm.mov_edi_mem_rbp,
                self.asm.mov_esi_mem_rbp,
                self.asm.mov_edx_mem_rbp,
                self.asm.mov_ecx_mem_rbp,
                self.asm.mov_r8d_mem_rbp,
                self.asm.mov_r9d_mem_rbp,
            ]
            INT_REGS_64 = [
                self.asm.mov_rdi_mem_rbp,
                self.asm.mov_rsi_mem_rbp,
                self.asm.mov_rdx_mem_rbp,
                self.asm.mov_rcx_arg_mem_rbp,
                self.asm.mov_r8_mem_rbp,
                self.asm.mov_r9_mem_rbp,
            ]
            # SysV ABI splits args by class: int → INT_REGS, float → xmm0..xmm7.
            # Each class has its own counter.
            int_idx = 0
            xmm_idx = 0
            for arg in op.operands:
                arg_slot = self._slot_of(arg)
                if self._is_float_type(arg.ty):
                    if xmm_idx >= 8:
                        raise NotImplementedError(
                            "v0.1 supports up to 8 float args via xmm0..xmm7"
                        )
                    if self._is_f64_type(arg.ty):
                        self.asm._movsd_load_xmmN(xmm_idx, arg_slot)
                    else:
                        self.asm._movss_load_xmmN(xmm_idx, arg_slot)
                    xmm_idx += 1
                else:
                    if int_idx >= len(INT_REGS):
                        raise NotImplementedError(
                            "v0.1 supports up to 6 int args"
                        )
                    if self._is_i64_type(arg.ty):
                        INT_REGS_64[int_idx](arg_slot)
                    else:
                        INT_REGS[int_idx](arg_slot)
                    int_idx += 1
            self.asm.call_rel32(str(target))
            if op.results:
                res_slot = self._slot_of(op.results[0])
                # SysV: float return in xmm0, int return in eax.
                if self._is_f64_type(op.results[0].ty):
                    self.asm.movsd_mem_rbp_xmm0(res_slot)
                elif self._is_float_type(op.results[0].ty):
                    self.asm.movss_mem_rbp_xmm0(res_slot)
                elif self._is_i64_type(op.results[0].ty):
                    self.asm.mov_mem_rbp_rax(res_slot)
                else:
                    self.asm.mov_mem_rbp_eax(res_slot)
            return
        # ====================================================================
        # Stage 16.5 — FFI ops
        # ====================================================================
        if op.kind == tir.OpKind.FFI_CALL:
            # Indirect call through GOT entry resolved by the dynamic linker.
            # Arg shuffle is identical to CALL: int args -> rdi/rsi/rdx/rcx/r8/r9
            # (or 64-bit forms when the IR type is i64/u64/pointer-shaped).
            target = str(op.attrs.get("target", "?"))
            INT_REGS = [
                self.asm.mov_edi_mem_rbp,
                self.asm.mov_esi_mem_rbp,
                self.asm.mov_edx_mem_rbp,
                self.asm.mov_ecx_mem_rbp,
                self.asm.mov_r8d_mem_rbp,
                self.asm.mov_r9d_mem_rbp,
            ]
            INT_REGS_64 = [
                self.asm.mov_rdi_mem_rbp,
                self.asm.mov_rsi_mem_rbp,
                self.asm.mov_rdx_mem_rbp,
                self.asm.mov_rcx_arg_mem_rbp,
                self.asm.mov_r8_mem_rbp,
                self.asm.mov_r9_mem_rbp,
            ]
            int_idx = 0
            for arg in op.operands:
                arg_slot = self._slot_of(arg)
                if int_idx >= len(INT_REGS):
                    raise NotImplementedError(
                        "FFI_CALL supports up to 6 int/pointer args (Phase-0)")
                # Pointer-shaped IR types are u64 — use the 64-bit move.
                if self._is_i64_type(arg.ty) or self._is_u64_type(arg.ty):
                    INT_REGS_64[int_idx](arg_slot)
                else:
                    INT_REGS[int_idx](arg_slot)
                int_idx += 1
            # Indirect call through GOT entry.
            self.asm.call_qword_ptr_rip_rel_ffi(target)
            if op.results:
                res_slot = self._slot_of(op.results[0])
                # libc fns return int (eax) or pointer (rax). Stage 16.5
                # only wires int returns; other shapes deferred.
                if self._is_i64_type(op.results[0].ty) or self._is_u64_type(op.results[0].ty):
                    self.asm.mov_mem_rbp_rax(res_slot)
                else:
                    self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.STR_PTR:
            # Address of a string literal — emit `lea rax, [rip + sym]`
            # then store as 64-bit pointer to result slot.
            text = op.attrs.get("text", "")
            assert isinstance(text, str)
            data = text.encode("utf-8")
            sym = f"__helix_strptr_{id(op):x}"
            self._pending_strings.append((sym, data))
            self.asm.lea_rax_rip_rel(sym)
            res_slot = self._slot_of(op.results[0])
            self.asm.mov_mem_rbp_rax(res_slot)
            return
        if op.kind == tir.OpKind.RETURN:
            if op.operands:
                slot = self._slot_of(op.operands[0])
                # SysV: float return in xmm0, int return in eax/rax.
                if self._is_f64_type(op.operands[0].ty):
                    self.asm.movsd_xmm0_mem_rbp(slot)
                elif self._is_float_type(op.operands[0].ty):
                    self.asm.movss_xmm0_mem_rbp(slot)
                elif self._is_i64_type(op.operands[0].ty):
                    self.asm.mov_rax_mem_rbp(slot)
                else:
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
                operand_ty = op.operands[0].ty
                if self._is_f64_type(operand_ty):
                    self.asm._movsd_load_xmmN(0, src_slot)
                    self.asm.movsd_mem_rbp_xmm0(dst_slot)
                elif self._is_i64_type(operand_ty):
                    self.asm.mov_rax_mem_rbp(src_slot)
                    self.asm.mov_mem_rbp_rax(dst_slot)
                else:
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
            # 32-bit movs for i32 / f32 (bit pattern round-trips). 64-bit
            # movs for i64 / f64 so the upper 4 bytes survive the copy.
            res_ty = op.results[0].ty
            if self._is_f64_type(res_ty):
                self.asm._movsd_load_xmmN(0, var_slot)
                self.asm.movsd_mem_rbp_xmm0(res_slot)
            elif self._is_float_type(res_ty):
                self.asm.movss_xmm0_mem_rbp(var_slot)
                self.asm.movss_mem_rbp_xmm0(res_slot)
            elif self._is_i64_type(res_ty):
                self.asm.mov_rax_mem_rbp(var_slot)
                self.asm.mov_mem_rbp_rax(res_slot)
            else:
                self.asm.mov_eax_mem_rbp(var_slot)
                self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.STORE_VAR:
            name = op.attrs["name"]
            var_slot = self.var_slots[name]
            src_slot = self._slot_of(op.operands[0])
            src_ty = op.operands[0].ty
            if self._is_f64_type(src_ty):
                self.asm._movsd_load_xmmN(0, src_slot)
                self.asm.movsd_mem_rbp_xmm0(var_slot)
            elif self._is_float_type(src_ty):
                self.asm.movss_xmm0_mem_rbp(src_slot)
                self.asm.movss_mem_rbp_xmm0(var_slot)
            elif self._is_i64_type(src_ty):
                self.asm.mov_rax_mem_rbp(src_slot)
                self.asm.mov_mem_rbp_rax(var_slot)
            else:
                self.asm.mov_eax_mem_rbp(src_slot)
                self.asm.mov_mem_rbp_eax(var_slot)
            return

        # AGI primitives
        if op.kind == tir.OpKind.ARENA_PUSH:
            # cursor (slot 0) holds the count of used slots after slot 0.
            # Push: load cursor into eax, store value at base + (cursor+1)*4,
            # increment cursor, return old cursor (the new slot's index).
            buf = self.asm.b
            val_slot = self._slot_of(op.operands[0])
            res_slot = self._slot_of(op.results[0])
            # Load value into edx so we can use eax for cursor arithmetic.
            self.asm.mov_eax_mem_rbp(val_slot)   # eax = value
            buf.emit(0x89, 0xC2)                 # mov edx, eax
            # Load cursor (32-bit at offset 0 of arena base).
            self.asm.lea_rax_rip_rel("__helix_arena_base")
            buf.emit(0x8B, 0x08)                 # mov ecx, [rax]
            # Bounds check: cursor must be < HELIX_ARENA_CAP. Overflow
            # writes past the data section. On overflow: store -1 to
            # res_slot and skip the actual push (audit-10 critical fix).
            buf.emit(0x81, 0xF9)                 # cmp ecx, HELIX_ARENA_CAP
            buf.emit_bytes(struct.pack("<I", HELIX_ARENA_CAP))
            # jb in_bounds (+7 — skip mov-eax-imm32 [5 bytes] + jmp [2 bytes])
            buf.emit(0x72, 0x07)
            # Overflow path: mov eax, -1 (5 bytes); jmp store_result (+12 over in_bounds)
            self.asm.mov_eax_imm32(0xFFFFFFFF)   # eax = -1
            buf.emit(0xEB, 0x0C)                 # jmp store_result (+12 over in_bounds 12 bytes)
            # in_bounds: store value at base + (cursor+1)*4.
            # rax = base, rcx = cursor. Use SIB: [rax + rcx*4 + 4].
            buf.emit(0x89, 0x54, 0x88, 0x04)     # mov [rax + rcx*4 + 4], edx
            # Increment cursor and store back.
            buf.emit(0xFF, 0xC1)                 # inc ecx
            buf.emit(0x89, 0x08)                 # mov [rax], ecx
            # Result: the OLD cursor (i.e. the slot index just written).
            buf.emit(0xFF, 0xC9)                 # dec ecx (recover old)
            buf.emit(0x89, 0xC8)                 # mov eax, ecx
            # store_result:
            self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.ARENA_GET:
            # Return arena[idx + 1] (slot 0 is cursor). Out-of-bounds
            # (negative when interpreted unsigned, or >= CAP) returns 0
            # — defined behavior, no trap. Use jb (unsigned-below) so
            # negative ecx fails the test.
            buf = self.asm.b
            idx_slot = self._slot_of(op.operands[0])
            res_slot = self._slot_of(op.results[0])
            self.asm.mov_ecx_mem_rbp(idx_slot)     # ecx = index
            buf.emit(0x81, 0xF9)                   # cmp ecx, HELIX_ARENA_CAP
            buf.emit_bytes(struct.pack("<I", HELIX_ARENA_CAP))
            # Layout: jb in_bounds (+7) ; mov eax, 0 (5) ; jmp store (+11) ;
            #         in_bounds: lea (7) ; mov eax, [rax+rcx*4+4] (4) ;
            #         store: mov_mem_rbp_eax res_slot
            buf.emit(0x72, 0x07)                   # jb in_bounds (skip 5+2 below)
            self.asm.mov_eax_imm32(0)              # 5 bytes (out-of-bounds value)
            buf.emit(0xEB, 0x0B)                   # jmp store (skip 7+4)
            # in_bounds:
            self.asm.lea_rax_rip_rel("__helix_arena_base")  # 7 bytes
            buf.emit(0x8B, 0x44, 0x88, 0x04)       # mov eax, [rax+rcx*4+4]
            # store:
            self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.ARENA_SET:
            # Out-of-bounds set silently no-ops (cursor untouched, no
            # store). Layout uses pre-loaded value in edx so the
            # bounds-check displacement only spans known-size ops.
            buf = self.asm.b
            idx_slot = self._slot_of(op.operands[0])
            val_slot = self._slot_of(op.operands[1])
            # Load value first (varies in size, but doesn't affect the
            # post-cmp branch). Then ecx, then check.
            self.asm.mov_eax_mem_rbp(val_slot)     # eax = value
            buf.emit(0x89, 0xC2)                   # mov edx, eax (2 bytes)
            self.asm.mov_ecx_mem_rbp(idx_slot)     # ecx = index
            buf.emit(0x81, 0xF9)                   # cmp ecx, HELIX_ARENA_CAP (6 bytes)
            buf.emit_bytes(struct.pack("<I", HELIX_ARENA_CAP))
            # Layout: jae skip (+11) ; lea (7) ; mov [rax+rcx*4+4], edx (4) ; skip:
            buf.emit(0x73, 0x0B)                   # jae skip (skip 7+4)
            self.asm.lea_rax_rip_rel("__helix_arena_base")  # 7 bytes
            buf.emit(0x89, 0x54, 0x88, 0x04)       # mov [rax+rcx*4+4], edx (4 bytes)
            # skip:
            if op.results:
                res_slot = self._slot_of(op.results[0])
                self.asm.mov_eax_imm32(0)
                self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.STR_BYTE:
            # Load one byte from a literal string at runtime index.
            text = op.attrs.get("text", "")
            assert isinstance(text, str)
            data = text.encode("utf-8")
            sym = f"__helix_strbyte_{id(op):x}"
            self._pending_strings.append((sym, data))
            buf = self.asm.b
            idx_slot = self._slot_of(op.operands[0])
            res_slot = self._slot_of(op.results[0])
            self.asm.mov_ecx_mem_rbp(idx_slot)         # ecx = idx
            self.asm.lea_rax_rip_rel(sym)              # rax = &literal
            # Bounds check: if idx >= len, return 0.
            buf.emit(0x81, 0xF9)                       # cmp ecx, len
            buf.emit_bytes(struct.pack("<i", len(data)))
            # jb (unsigned below) — catches negative indices too. Signed
            # `jl` would let idx=-1 fall through to the movzx and read
            # one byte BEFORE the literal.
            buf.emit(0x72, 0x07)                       # jb in_range (+7)
            self.asm.mov_eax_imm32(0)                  # out-of-range → 0 (5 bytes)
            buf.emit(0xEB, 0x04)                       # jmp store_result (+4 over movzx)
            # in_range: movzx eax, byte [rax + rcx]
            buf.emit(0x0F, 0xB6, 0x04, 0x08)           # movzx eax, [rax+rcx]
            self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.ARENA_LEN:
            # Return cursor (the i32 at slot 0 of the arena).
            buf = self.asm.b
            res_slot = self._slot_of(op.results[0])
            self.asm.lea_rax_rip_rel("__helix_arena_base")
            buf.emit(0x8B, 0x00)                 # mov eax, [rax]
            self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.PRINT:
            kind = op.attrs.get("_kind", "print_str")
            if kind == "write_file":
                # open(path, O_WRONLY|O_CREAT|O_TRUNC, 0o644)
                # write(fd, content, len)
                # close(fd)
                path_str = op.attrs["path"]
                content = op.attrs["content"]
                assert isinstance(path_str, str) and isinstance(content, str)
                # Path needs a NUL terminator since open() expects C-string
                path_data = path_str.encode("utf-8") + b"\x00"
                content_data = content.encode("utf-8")
                path_sym = f"__helix_path_{id(op):x}"
                content_sym = f"__helix_content_{id(op):x}"
                self._pending_strings.append((path_sym, path_data))
                self._pending_strings.append((content_sym, content_data))

                # ---- sys_open(path, flags=O_WRONLY|O_CREAT|O_TRUNC, mode=0o644) ----
                # We use the stack directly via push/pop to persist the fd
                # and write's return across syscalls. This avoids relying
                # on callee-saved registers (which would require prologue
                # save/restore not currently emitted) and avoids fragile
                # post-prologue frame extension.
                buf = self.asm.b
                buf.emit(0x48, 0x8D, 0x3D)  # lea rdi, [rip + disp32]
                off = buf.offset()
                buf.emit_bytes(b"\x00\x00\x00\x00")
                buf.fixups.append(Fixup(offset=off, target=path_sym,
                                         size=4, rel_base=off + 4))
                self.asm.mov_esi_imm32(0x241)  # O_WRONLY|O_CREAT|O_TRUNC
                self.asm.mov_edx_imm32(0x1A4)  # mode 0644
                self.asm.mov_eax_imm32(2)      # sys_open
                self.asm.syscall()
                # rax = fd. Push rax to stack.   50 = push rax
                buf.emit(0x50)

                # ---- write(fd, content, len) ----
                # mov rdi, [rsp]  — load fd from top of stack
                # 48 8B 3C 24
                buf.emit(0x48, 0x8B, 0x3C, 0x24)
                self.asm.lea_rsi_rip_rel(content_sym)
                self.asm.mov_edx_imm32(len(content_data))
                self.asm.mov_eax_imm32(1)  # sys_write
                self.asm.syscall()
                # rax = write_ret. Push rax → [write_ret, fd] on stack.
                buf.emit(0x50)

                # ---- close(fd) ----
                # mov rdi, [rsp + 8]  — load fd (now under write_ret)
                # 48 8B 7C 24 08
                buf.emit(0x48, 0x8B, 0x7C, 0x24, 0x08)
                self.asm.mov_eax_imm32(3)  # sys_close
                self.asm.syscall()
                # close's return is in rax — discard. Pop write_ret and fd.
                # pop rax  (= write_ret)
                buf.emit(0x58)
                # pop rcx  (= fd, discarded)  59
                buf.emit(0x59)
                # eax now holds write_ret. If negative, that's the diagnostic
                # return; otherwise return 0.
                #   cmp eax, 0
                buf.emit(0x83, 0xF8, 0x00)
                # jl keep_eax  (rel8)
                buf.emit(0x7C, 0x00)
                jl_off = buf.offset() - 1
                jl_after = buf.offset()
                # write_ret >= 0: eax = 0 (success)
                self.asm.mov_eax_imm32(0)
                keep_addr = buf.offset()
                d = keep_addr - jl_after
                if not (-128 <= d <= 127):
                    raise ValueError(f"write_file jl disp out of rel8: {d}")
                buf.bytes_[jl_off] = d & 0xFF

                if op.results:
                    res_slot = self._slot_of(op.results[0])
                    self.asm.mov_mem_rbp_eax(res_slot)
                return

            if kind == "read_file_to_arena":
                # FULL implementation: opens path (O_RDONLY), reads up to
                # BUF_SIZE bytes into a stack buffer, pushes each byte into
                # the arena (one i32 slot per byte; the low 8 bits hold the
                # byte value). Returns the number of bytes successfully
                # pushed (= bytes read, capped at the remaining arena
                # capacity). The byte-push loop (audit-fixed) is now correct
                # and exercised by the bootstrap pipeline test.
                path_str = op.attrs["path"]
                assert isinstance(path_str, str)
                path_data = path_str.encode("utf-8") + b"\x00"
                path_sym = f"__helix_rftoa_path_{id(op):x}"
                self._pending_strings.append((path_sym, path_data))
                buf = self.asm.b
                # IMPORTANT: BUF_SIZE must fit in signed 8-bit disp (max 127)
                # if we use the disp8 sub-rsp form, OR use the imm32 form
                # for larger sizes. Bug: previously BUF_SIZE=128 used disp8
                # which sign-extended to -128 — adding 128 to rsp instead
                # of subtracting (clobbering parent stack frame).
                # Approach-A bump: 256K → 1M. The bootstrap source
                # (lexer.hx + parser.hx + kovc.hx + driver_main) had
                # crept up to ~261 KB, leaving < 1 KB of margin against
                # 256 KB. ANY new fn or @pure helper added to kovc.hx
                # tipped k1_input over the buffer; K1's read truncated;
                # K1 produced a K2 missing tail-end fns; K2 SIGILLed.
                # Mis-attributed for weeks as a "cascade-depth bug"
                # (see docs/BOOTSTRAP_CASCADE_BUG.md, probe 10). Bump
                # to 1 MB gives ~4× headroom and uses disp32 form.
                # Keep this value in lock-step with the four BUF_SIZE
                # constants in helixc/bootstrap/kovc.hx's
                # emit_read_file_to_arena_body so K1 (Python-emitted)
                # and K2 (kovc.hx-emitted) agree on the read buffer.
                BUF_SIZE = 0x100000

                # ---- sys_open(path, O_RDONLY=0) ----
                buf.emit(0x48, 0x8D, 0x3D)            # lea rdi, [rip+disp]
                off = buf.offset()
                buf.emit_bytes(b"\x00\x00\x00\x00")
                buf.fixups.append(Fixup(offset=off, target=path_sym,
                                         size=4, rel_base=off + 4))
                self.asm.mov_esi_imm32(0)              # O_RDONLY
                self.asm.mov_edx_imm32(0)
                self.asm.mov_eax_imm32(2)              # sys_open
                self.asm.syscall()
                # Push fd to stack, allocate read buffer (disp32 form).
                buf.emit(0x50)                          # push rax (fd)
                # sub rsp, imm32 form: 48 81 EC imm32
                buf.emit(0x48, 0x81, 0xEC)
                buf.emit_bytes(struct.pack("<I", BUF_SIZE))

                # ---- read(fd, buf=rsp, BUF_SIZE) ----
                # mov rdi, [rsp+BUF_SIZE] using disp32 form:
                #   48 8B BC 24 disp32
                buf.emit(0x48, 0x8B, 0xBC, 0x24)
                buf.emit_bytes(struct.pack("<I", BUF_SIZE))
                buf.emit(0x48, 0x89, 0xE6)              # mov rsi, rsp
                self.asm.mov_edx_imm32(BUF_SIZE)
                self.asm.mov_eax_imm32(0)               # sys_read
                self.asm.syscall()
                # Save bytes-read in r10.
                buf.emit(0x49, 0x89, 0xC2)

                # Audit fix: truncation sentinel. If sys_read returned
                # exactly BUF_SIZE bytes, the file was at-or-beyond the
                # buffer and we silently lost data. This is the EXACT
                # failure mode that caused the original cascade-bug
                # (file size crept up to within 1 KB of the 256 KB buffer
                # → silent truncation → K2 missing tail-end fns → SIGILL
                # at runtime far downstream). Trap loudly here instead
                # of producing a corrupt output binary.
                #   cmp r10, BUF_SIZE      (49 81 FA imm32)
                #   jne +2                 (75 02)
                #   ud2                    (0F 0B)
                buf.emit(0x49, 0x81, 0xFA)
                buf.emit_bytes(struct.pack("<I", BUF_SIZE))
                buf.emit(0x75, 0x02)
                buf.emit(0x0F, 0x0B)

                # ---- close(fd) ----
                buf.emit(0x48, 0x8B, 0xBC, 0x24)        # mov rdi, [rsp+BUF_SIZE] disp32
                buf.emit_bytes(struct.pack("<I", BUF_SIZE))
                self.asm.mov_eax_imm32(3)               # sys_close
                self.asm.syscall()

                # If r10 < 0, set r10 = 0.
                buf.emit(0x4D, 0x85, 0xD2)              # test r10, r10
                buf.emit(0x7D, 0x03)                    # jns +3 (skip xor)
                buf.emit(0x4D, 0x31, 0xD2)              # xor r10, r10

                # Now push each byte of the read buffer to the arena.
                # rcx = byte counter, r10 = bytes_read (limit).
                buf.emit(0x31, 0xC9)                    # xor ecx, ecx
                # loop_start:
                loop_start = buf.offset()
                buf.emit(0x4C, 0x39, 0xD1)              # cmp rcx, r10
                buf.emit(0x7D, 0x00)                    # jge end (placeholder)
                jge_off = buf.offset() - 1
                jge_after = buf.offset()
                # movzx eax, byte [rsp + rcx]  (buffer[rcx])
                buf.emit(0x0F, 0xB6, 0x04, 0x0C)
                buf.emit(0x89, 0xC2)                    # mov edx, eax
                # Inline arena_push.
                self.asm.lea_rax_rip_rel("__helix_arena_base")
                buf.emit(0x44, 0x8B, 0x18)              # mov r11d, [rax]
                # cmp r11d, HELIX_ARENA_CAP
                buf.emit(0x41, 0x81, 0xFB)
                buf.emit_bytes(struct.pack("<I", HELIX_ARENA_CAP))
                buf.emit(0x72, 0x02)                    # jb in_bounds (+2)
                buf.emit(0xEB, 0x0B)                    # jmp loop_advance (+11)
                # in_bounds:
                buf.emit(0x42, 0x89, 0x54, 0x98, 0x04)  # mov [rax+r11*4+4], edx
                buf.emit(0x41, 0xFF, 0xC3)              # inc r11d
                buf.emit(0x44, 0x89, 0x18)              # mov [rax], r11d
                # loop_advance:
                buf.emit(0x48, 0xFF, 0xC1)              # inc rcx
                # jmp loop_start (rel8 backward)
                buf.emit(0xEB, 0x00)                    # placeholder
                jmp_back_off = buf.offset() - 1
                jmp_back_after = buf.offset()
                back_disp = loop_start - jmp_back_after
                if not (-128 <= back_disp <= 127):
                    raise ValueError("read_file_to_arena loop disp out of rel8")
                buf.bytes_[jmp_back_off] = back_disp & 0xFF
                # end:
                end_addr = buf.offset()
                fwd_disp = end_addr - jge_after
                if not (-128 <= fwd_disp <= 127):
                    raise ValueError("read_file_to_arena jge disp out of rel8")
                buf.bytes_[jge_off] = fwd_disp & 0xFF

                # Restore stack (disp32 form for the buffer).
                buf.emit(0x48, 0x81, 0xC4)              # add rsp, imm32
                buf.emit_bytes(struct.pack("<I", BUF_SIZE))
                buf.emit(0x48, 0x83, 0xC4, 0x08)        # add rsp, 8 (fd)

                # Return r10 (bytes pushed).
                buf.emit(0x4C, 0x89, 0xD0)              # mov rax, r10
                if op.results:
                    res_slot = self._slot_of(op.results[0])
                    self.asm.mov_mem_rbp_eax(res_slot)
                return

            if kind == "write_file_to_arena":
                # Symmetric to read_file_to_arena: open the file
                # O_WRONLY|O_CREAT|O_TRUNC mode 0644, then write
                # n_bytes from the arena (one byte per slot, low byte)
                # via per-byte sys_write. Return the count actually
                # written. Operands: arena_start (slot 0), n_bytes (slot 1).
                path_str = op.attrs["path"]
                assert isinstance(path_str, str)
                path_data = path_str.encode("utf-8") + b"\x00"
                path_sym = f"__helix_wftoa_path_{id(op):x}"
                self._pending_strings.append((path_sym, path_data))
                buf = self.asm.b

                arena_start_slot = self._slot_of(op.operands[0])
                n_bytes_slot = self._slot_of(op.operands[1])

                # Save callee-saved registers we will use as state
                # carriers across syscalls (rbx, r12-r15).
                buf.emit(0x53)                              # push rbx
                buf.emit(0x41, 0x54)                        # push r12
                buf.emit(0x41, 0x55)                        # push r13
                buf.emit(0x41, 0x56)                        # push r14

                # Reserve 16-byte stack frame: 1 byte buffer + 8 bytes fd.
                buf.emit(0x48, 0x83, 0xEC, 0x10)            # sub rsp, 16

                # ---- sys_open(path, O_WRONLY|O_CREAT|O_TRUNC, 0644) ----
                buf.emit(0x48, 0x8D, 0x3D)                  # lea rdi, [rip+disp]
                off = buf.offset()
                buf.emit_bytes(b"\x00\x00\x00\x00")
                buf.fixups.append(Fixup(offset=off, target=path_sym,
                                         size=4, rel_base=off + 4))
                self.asm.mov_esi_imm32(0x241)               # O_WRONLY|O_CREAT|O_TRUNC
                self.asm.mov_edx_imm32(0x1A4)               # mode 0644
                self.asm.mov_eax_imm32(2)                   # sys_open
                self.asm.syscall()
                buf.emit(0x48, 0x89, 0x44, 0x24, 0x08)      # mov [rsp+8], rax (fd)

                # If fd < 0, skip the loop and return 0.
                buf.emit(0x48, 0x85, 0xC0)                  # test rax, rax
                # Place a forward jl placeholder; we'll patch after we
                # know the body length.
                buf.emit(0x7C, 0x00)                        # jl error_close (placeholder)
                err_jmp_off = buf.offset() - 1

                # Initialize state regs.
                # r12d = arena_start, r13d = n_bytes, r14d = counter (0).
                self.asm.mov_eax_mem_rbp(arena_start_slot)  # eax = arena_start
                buf.emit(0x41, 0x89, 0xC4)                  # mov r12d, eax
                self.asm.mov_eax_mem_rbp(n_bytes_slot)      # eax = n_bytes
                buf.emit(0x41, 0x89, 0xC5)                  # mov r13d, eax
                buf.emit(0x45, 0x31, 0xF6)                  # xor r14d, r14d

                # Loop:
                #   if r14 >= r13 -> done
                #   eax = arena[r12 + r14]
                #   [rsp] = al
                #   sys_write(fd, &[rsp], 1)
                #   inc r14
                #   jmp loop
                loop_start = buf.offset()
                buf.emit(0x45, 0x39, 0xEE)                  # cmp r14d, r13d
                buf.emit(0x7D, 0x00)                        # jge done (placeholder)
                jge_off = buf.offset() - 1
                jge_after = buf.offset()

                # idx = r12 + r14 (in eax)
                buf.emit(0x44, 0x89, 0xE0)                  # mov eax, r12d
                buf.emit(0x44, 0x01, 0xF0)                  # add eax, r14d

                # Load arena base into rdx for the lea.
                self.asm.lea_rax_rip_rel("__helix_arena_base")  # rax = base
                # We need to use an index register for [rax + idx*4 + 4].
                # Move idx to rcx.
                buf.emit(0x44, 0x89, 0xE1)                  # mov ecx, r12d
                buf.emit(0x44, 0x01, 0xF1)                  # add ecx, r14d
                # Bounds check: ecx < HELIX_ARENA_CAP
                buf.emit(0x81, 0xF9)                        # cmp ecx, HELIX_ARENA_CAP
                buf.emit_bytes(struct.pack("<I", HELIX_ARENA_CAP))
                buf.emit(0x73, 0x00)                        # jae done (placeholder)
                jae_off = buf.offset() - 1
                jae_after = buf.offset()
                buf.emit(0x8B, 0x44, 0x88, 0x04)            # mov eax, [rax + rcx*4 + 4]
                buf.emit(0x88, 0x04, 0x24)                  # mov [rsp], al

                # sys_write(fd, &byte, 1)
                buf.emit(0x48, 0x8B, 0x7C, 0x24, 0x08)      # mov rdi, [rsp+8]
                buf.emit(0x48, 0x89, 0xE6)                  # mov rsi, rsp
                self.asm.mov_edx_imm32(1)                   # rdx = 1
                self.asm.mov_eax_imm32(1)                   # sys_write
                self.asm.syscall()

                # inc r14
                buf.emit(0x41, 0xFF, 0xC6)                  # inc r14d
                # Loop back.
                buf.emit(0xEB, 0x00)                        # jmp loop_start (placeholder)
                back_jmp_off = buf.offset() - 1
                back_jmp_after = buf.offset()
                back_disp = loop_start - back_jmp_after
                if not (-128 <= back_disp <= 127):
                    raise ValueError("write_file_to_arena loop disp out of rel8")
                buf.bytes_[back_jmp_off] = back_disp & 0xFF

                # done:
                done_addr = buf.offset()
                # patch jge and jae forward jumps
                fwd1 = done_addr - jge_after
                fwd2 = done_addr - jae_after
                if not (-128 <= fwd1 <= 127):
                    raise ValueError("write_file_to_arena jge disp out of rel8")
                if not (-128 <= fwd2 <= 127):
                    raise ValueError("write_file_to_arena jae disp out of rel8")
                buf.bytes_[jge_off] = fwd1 & 0xFF
                buf.bytes_[jae_off] = fwd2 & 0xFF

                # close(fd)
                buf.emit(0x48, 0x8B, 0x7C, 0x24, 0x08)      # mov rdi, [rsp+8]
                self.asm.mov_eax_imm32(3)                   # sys_close
                self.asm.syscall()

                # Return r14 (count).
                buf.emit(0x44, 0x89, 0xF0)                  # mov eax, r14d
                # Skip error path.
                buf.emit(0xEB, 0x00)                        # jmp epilogue (placeholder)
                skip_err_off = buf.offset() - 1
                skip_err_after = buf.offset()

                # error_close: open failed; rax already < 0; return 0.
                err_addr = buf.offset()
                err_disp = err_addr - (err_jmp_off + 1)
                if not (-128 <= err_disp <= 127):
                    raise ValueError("write_file_to_arena err jmp disp out of rel8")
                buf.bytes_[err_jmp_off] = err_disp & 0xFF
                self.asm.mov_eax_imm32(0)                   # return 0

                # epilogue:
                ep_addr = buf.offset()
                skip_disp = ep_addr - skip_err_after
                if not (-128 <= skip_disp <= 127):
                    raise ValueError("write_file_to_arena skip-err disp out of rel8")
                buf.bytes_[skip_err_off] = skip_disp & 0xFF

                # Tear down stack and restore callee-saved regs.
                buf.emit(0x48, 0x83, 0xC4, 0x10)            # add rsp, 16
                buf.emit(0x41, 0x5E)                        # pop r14
                buf.emit(0x41, 0x5D)                        # pop r13
                buf.emit(0x41, 0x5C)                        # pop r12
                buf.emit(0x5B)                              # pop rbx

                if op.results:
                    res_slot = self._slot_of(op.results[0])
                    self.asm.mov_mem_rbp_eax(res_slot)
                return

            if kind == "read_file_int":
                # Opens path read-only, reads 4 bytes into a stack buffer,
                # closes the fd, returns those 4 bytes interpreted as i32.
                # On any error or short read, returns 0.
                path_str = op.attrs["path"]
                assert isinstance(path_str, str)
                path_data = path_str.encode("utf-8") + b"\x00"
                path_sym = f"__helix_path_{id(op):x}"
                self._pending_strings.append((path_sym, path_data))
                buf = self.asm.b

                # ---- sys_open(path, O_RDONLY=0, mode=0) ----
                buf.emit(0x48, 0x8D, 0x3D)  # lea rdi, [rip + path_sym]
                off = buf.offset()
                buf.emit_bytes(b"\x00\x00\x00\x00")
                buf.fixups.append(Fixup(offset=off, target=path_sym,
                                         size=4, rel_base=off + 4))
                self.asm.mov_esi_imm32(0)      # O_RDONLY
                self.asm.mov_edx_imm32(0)      # mode (ignored)
                self.asm.mov_eax_imm32(2)      # sys_open
                self.asm.syscall()
                # rax = fd. Push rax (fd) onto stack.
                buf.emit(0x50)
                # Allocate 8 bytes on the stack for the read buffer
                # (sub rsp, 8 — really 4 needed, but stay 8-aligned).
                # 48 83 EC 08
                buf.emit(0x48, 0x83, 0xEC, 0x08)
                # Initialise buffer to 0 so a short read leaves the high
                # bytes clean: mov qword [rsp], 0 → 48 C7 04 24 00 00 00 00
                buf.emit(0x48, 0xC7, 0x04, 0x24, 0x00, 0x00, 0x00, 0x00)

                # ---- read(fd, buf=rsp, 4) ----
                # mov rdi, [rsp+8]  (fd is below the buffer)
                buf.emit(0x48, 0x8B, 0x7C, 0x24, 0x08)
                # mov rsi, rsp  (48 89 E6)
                buf.emit(0x48, 0x89, 0xE6)
                self.asm.mov_edx_imm32(4)
                self.asm.mov_eax_imm32(0)  # sys_read
                self.asm.syscall()
                # rax = bytes read. Push to stack — Linux syscalls clobber
                # rcx and r11, so we cannot use a register to hold this
                # across the upcoming close syscall.
                buf.emit(0x50)  # push rax  → stack: [bytes_read, buf, fd]

                # ---- close(fd) ----
                # mov rdi, [rsp+16]  (fd is now two slots down)
                buf.emit(0x48, 0x8B, 0x7C, 0x24, 0x10)
                self.asm.mov_eax_imm32(3)  # sys_close
                self.asm.syscall()

                # Pop bytes_read into rcx for the comparison.
                # pop rcx (= 8B C1)... actually pop rcx is just 0x59
                buf.emit(0x59)
                # If read returned exactly 4, eax = [rsp]; else eax = 0.
                # mov eax, [rsp]   (8B 04 24)
                buf.emit(0x8B, 0x04, 0x24)
                # cmp ecx, 4   (83 F9 04)
                buf.emit(0x83, 0xF9, 0x04)
                # je keep   (74 rel8)
                buf.emit(0x74, 0x00)
                je_off = buf.offset() - 1
                je_after = buf.offset()
                # short / error: eax = 0
                self.asm.mov_eax_imm32(0)
                keep_addr = buf.offset()
                d = keep_addr - je_after
                if not (-128 <= d <= 127):
                    raise ValueError(f"read_file_int je disp out of rel8: {d}")
                buf.bytes_[je_off] = d & 0xFF

                # Tear down stack: pop the 8-byte buffer + the fd push.
                # add rsp, 8  (buffer)  →  48 83 C4 08
                buf.emit(0x48, 0x83, 0xC4, 0x08)
                # add rsp, 8  (fd push) →  48 83 C4 08
                buf.emit(0x48, 0x83, 0xC4, 0x08)

                if op.results:
                    res_slot = self._slot_of(op.results[0])
                    self.asm.mov_mem_rbp_eax(res_slot)
                return

            if kind == "print_int":
                # Convert i32 -> ASCII decimal and write(1, buf, len).
                # Strategy: load value into eax, build digits backwards
                # into a 12-byte stack buffer, then issue sys_write.
                # For simplicity, only positive values are supported in
                # v0.1; negative values produce a leading '-' but no
                # two's-complement handling (i.e. INT_MIN gets garbled
                # which is acceptable for diagnostic output).
                val_slot = self._slot_of(op.operands[0])
                self.asm.mov_eax_mem_rbp(val_slot)
                # Allocate 16 bytes on stack for the digit buffer.
                buf = self.asm.b
                # rbx is callee-saved under SysV — print_int uses bl/ebx
                # as a sign flag, so we must preserve the caller's rbx.
                # `push rbx` (1 byte: 53) + matching pop after the
                # syscall keeps the ABI invariant intact for any future
                # caller that depends on rbx surviving the call.
                buf.emit(0x53)                    # push rbx
                buf.emit(0x48, 0x83, 0xEC, 0x10)  # sub rsp, 16
                # Digit pointer starts past the end (rsp+15) and walks down.
                # Use rdi as the digit pointer, ecx as divisor=10.
                buf.emit(0x48, 0x8D, 0x7C, 0x24, 0x10)  # lea rdi, [rsp+16]
                # mov ecx, 10  (B9 0A 00 00 00)
                buf.emit(0xB9, 0x0A, 0x00, 0x00, 0x00)
                # If value is negative, neg it and remember sign.
                # bl = sign byte (0=pos, 1=neg)
                buf.emit(0x31, 0xDB)            # xor ebx, ebx
                buf.emit(0x85, 0xC0)            # test eax, eax
                buf.emit(0x79, 0x04)            # jns +4 (skip neg branch)
                buf.emit(0xF7, 0xD8)            # neg eax
                buf.emit(0xB3, 0x01)            # mov bl, 1
                # Digit loop: divide eax by 10, remainder→digit char.
                # do { dec rdi; mov[rdi], (eax%10)+'0'; eax /= 10; } while eax>0
                # First iteration always runs (handles eax=0 case).
                # Loop start label position:
                loop_start = buf.offset()
                buf.emit(0x31, 0xD2)            # xor edx, edx
                buf.emit(0xF7, 0xF1)            # div ecx (eax /= 10, edx = digit)
                buf.emit(0x80, 0xC2, 0x30)      # add dl, '0'
                buf.emit(0x48, 0xFF, 0xCF)      # dec rdi
                buf.emit(0x88, 0x17)            # mov [rdi], dl
                buf.emit(0x85, 0xC0)            # test eax, eax
                # jnz back to loop_start: rel8 = loop_start - (offset_after_jnz)
                jnz_at = buf.offset()
                rel = loop_start - (jnz_at + 2)
                buf.emit(0x75, rel & 0xFF)      # jnz loop_start
                # If negative, prepend '-'.
                buf.emit(0x80, 0xFB, 0x01)      # cmp bl, 1
                buf.emit(0x75, 0x06)            # jne +6 (skip the prepend)
                buf.emit(0x48, 0xFF, 0xCF)      # dec rdi
                buf.emit(0xC6, 0x07, 0x2D)      # mov byte [rdi], '-'
                # write(fd=1, buf=rdi, len=(rsp+16)-rdi).
                buf.emit(0x48, 0x8D, 0x74, 0x24, 0x10)  # lea rsi, [rsp+16]
                buf.emit(0x48, 0x29, 0xFE)      # sub rsi, rdi  (rsi = end - start)
                # Compute len in edx, ptr in rsi (rdi → start of digits).
                buf.emit(0x48, 0x89, 0xF2)      # mov rdx, rsi  (len)
                buf.emit(0x48, 0x89, 0xFE)      # mov rsi, rdi  (ptr)
                self.asm.mov_edi_imm32(1)       # fd=1 (stdout)
                self.asm.mov_eax_imm32(1)       # sys_write
                self.asm.syscall()
                # Restore stack, store return value in result slot.
                buf.emit(0x48, 0x83, 0xC4, 0x10)  # add rsp, 16
                buf.emit(0x5B)                    # pop rbx (restore caller's rbx)
                if op.results:
                    res_slot = self._slot_of(op.results[0])
                    self.asm.mov_mem_rbp_eax(res_slot)
                return

            # Default: print_str — write the string bytes to stdout.
            text = op.attrs.get("text", "")
            assert isinstance(text, str)
            data = text.encode("utf-8")
            # Emit a unique symbol per call site
            sym = f"__helix_str_{id(op):x}"
            # Stash the bytes for later emission with their symbol
            self._pending_strings.append((sym, data))
            self.asm.mov_edi_imm32(1)
            self.asm.lea_rsi_rip_rel(sym)
            self.asm.mov_edx_imm32(len(data))
            self.asm.mov_eax_imm32(1)  # sys_write
            self.asm.syscall()
            # PRINT's result slot gets the syscall return (#bytes written)
            if op.results:
                res_slot = self._slot_of(op.results[0])
                self.asm.mov_mem_rbp_eax(res_slot)
            return
        if op.kind == tir.OpKind.TRAP:
            # Stage 28.5 — panic("msg"). Emit the message to stderr (fd=2)
            # via sys_write, then sys_exit with the trap-id as the status
            # (truncated to a byte by the kernel). Execution does NOT
            # return from a TRAP op; we still fill the result slot for
            # SSA bookkeeping but no subsequent op observes it.
            text = op.attrs.get("text", "")
            trap_id = int(op.attrs.get("trap_id", 28501))
            assert isinstance(text, str)
            # Render a small header so the user sees BOTH the trap id and
            # the message at runtime, e.g. "panic[28501]: oh no\n".
            full = f"panic[{trap_id}]: {text}\n"
            data = full.encode("utf-8")
            sym = f"__helix_panic_{id(op):x}"
            self._pending_strings.append((sym, data))
            # sys_write(2, ptr, len)
            self.asm.mov_edi_imm32(2)  # fd = stderr
            self.asm.lea_rsi_rip_rel(sym)
            self.asm.mov_edx_imm32(len(data))
            self.asm.mov_eax_imm32(1)  # sys_write
            self.asm.syscall()
            # sys_exit(trap_id & 0xFF) — Linux truncates the status to a
            # byte; we pass the low 8 bits of the trap_id so users see a
            # distinctive non-zero exit code (e.g. 28501 & 0xFF = 0x55).
            self.asm.mov_edi_imm32(trap_id & 0xFF)
            self.asm.mov_eax_imm32(60)  # sys_exit
            self.asm.syscall()
            # ud2 belt-and-braces: if the exit syscall somehow returns
            # (shouldn't), trap loudly rather than fall through.
            self.asm.b.emit(0x0F, 0x0B)
            # Fill the result slot with the trap-id so SSA bookkeeping
            # (and any IR-level dead-code analysis) sees a defined value.
            # This code is unreachable at runtime — sys_exit never returns.
            if op.results:
                res_slot = self._slot_of(op.results[0])
                self.asm.mov_eax_imm32(trap_id)
                self.asm.mov_mem_rbp_eax(res_slot)
            return
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

            # In-range: pass args to verifier. ABI:
            #   default i32 verifier — edi=handle, esi=new_value
            #   f32 verifier (modify_f) — edi=handle, xmm0=new_value
            value_kind = op.attrs.get("value_kind", "i32")
            self.asm.mov_edi_mem_rbp(handle_slot)
            if value_kind == "f32":
                self.asm.movss_xmm0_mem_rbp(new_val_slot)
            else:
                self.asm.mov_esi_mem_rbp(new_val_slot)
            self.asm.call_rel32(verifier_name)
            # eax now holds verifier's return value
            self.asm.test_eax_eax()
            # je skip_apply (rel8 placeholder)
            buf.emit(0x74, 0x00)
            je_off = buf.offset() - 1
            je_after = buf.offset()

            # Apply: state[handle] = new_value (sign-extend 32-bit value to 64)
            # rcx = handle (sign-extended — handle is always a non-negative i32)
            self.asm.mov_ecx_mem_rbp(handle_slot)
            buf.emit(0x48, 0x63, 0xC9)   # movsxd rcx, ecx
            # rdx = new_value. For i32 we sign-extend; for f32 we zero-extend
            # so the cell's upper 32 bits stay clean. Negative floats have bit
            # 31 set; sign-extending would corrupt the high half of the cell.
            self.asm.mov_eax_mem_rbp(new_val_slot)
            if value_kind == "f32":
                # mov edx, eax (89 C0 → no, that's mov eax, eax; 89 C2 is
                # mov edx, eax). Implicitly zero-extends rdx upper 32 bits.
                buf.emit(0x89, 0xC2)
            else:
                # movsxd rdx, eax  (48 63 D0)
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
def emit_elf(code: bytes, entry_offset: int = ENTRY_OFFSET,
             extra_memsz: int = 0) -> bytes:
    """Wrap the given code bytes in a minimal x86-64 Linux ELF executable.

    `extra_memsz` extends p_memsz beyond p_filesz so the kernel zero-fills
    additional pages (BSS-style). Used to give compiled binaries an arena
    region without paying file-size cost for the zeros. The additional
    virtual address range starts at p_vaddr + p_filesz.

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
    total_memsz = total_filesz + extra_memsz

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

    # Pre-scan: do any user fns contain FFI_CALL ops? If yes, the binary
    # needs libc, so the entry stub uses libc's `exit` (which flushes
    # stdout/stderr) instead of a raw sys_exit syscall — otherwise output
    # from puts/printf via stdout is lost when stdout is a pipe and the
    # program exits before glibc's atexit handlers run.
    uses_ffi = any(
        op.kind == tir.OpKind.FFI_CALL
        for fn in module.functions.values()
        if not fn.attrs.get("is_extern")
        for blk in fn.blocks
        for op in blk.ops
    )

    # Entry stub: call entry_fn, then exit with eax as status.
    buf.define_symbol("_start")
    asm.call_rel32(entry_fn)
    # rdi = rax (status from main)
    asm.mov_edi_eax()
    if uses_ffi:
        # Indirect `call qword ptr [rip + got@exit]`. After the call,
        # if it ever returns (it shouldn't), `ud2` traps loudly so we
        # don't silently fall through into the next function.
        asm.call_qword_ptr_rip_rel_ffi("exit")
        buf.emit(0x0F, 0x0B)  # ud2
    else:
        # Libc-free: raw sys_exit syscall.
        asm.mov_eax_imm32(60)
        asm.syscall()

    # Compile each function and harvest any pending PRINT-string bytes.
    # Stage 16.5: extern "C" fns have no body to compile — they're just
    # signatures. Calls to them are routed to FFI_CALL ops in lower_ast,
    # which the FnCompiler emits as `call qword ptr [rip + got_slot]`.
    # Stage 16: @kernel fns also have no host body — they're PTX text
    # embedded after the code (below). Skip them in the x86 compile loop.
    pending_strings: list[tuple[str, bytes]] = []
    kernel_fns: list[tir.FnIR] = []
    for fn in module.functions.values():
        if fn.attrs.get("is_extern"):
            continue
        if fn.attrs.get("kernel"):
            kernel_fns.append(fn)
            continue
        fc = FnCompiler(fn, asm)
        fc.compile()
        pending_strings.extend(fc._pending_strings)

    # Append PRINT string bodies. Each gets its symbol so the RIP-relative
    # LEA from the syscall sequence resolves to the literal bytes.
    for sym, data in pending_strings:
        buf.define_symbol(sym)
        buf.emit_bytes(data)

    # Stage 16 — embed PTX text for each @kernel fn into the binary's
    # read-only data region (currently appended to the code segment, which
    # is mapped R-X — close enough for inspection-via-readelf in Phase-0;
    # at runtime an actual cuModuleLoadData would copy it out). The host
    # never executes these bytes; they're addressed only by reading from
    # the file or doing a RIP-relative LEA at kernel-launch time (Phase-1).
    if kernel_fns:
        from ..ir.tile_ir import lower_to_tile
        from .ptx import PtxEmitter, DEFAULT_TARGET
        # We emit a single PTX module containing every @kernel fn (since
        # PTX modules are text-only this is cheap). The lowering step is
        # idempotent — re-running lower_to_tile on the same Module just
        # produces a fresh TileModule.
        tile_mod = lower_to_tile(module)
        ptx_emitter = PtxEmitter(DEFAULT_TARGET)
        ptx_emitter.emit_module_header()
        for kf in kernel_fns:
            tile_fn = tile_mod.functions.get(kf.name)
            if tile_fn is not None:
                ptx_emitter.emit_kernel(tile_fn)
        # Also emit non-kernel device fns referenced from kernels — Phase-0
        # we just skip them (the vec_add capstone test doesn't use any).
        ptx_text = ptx_emitter.buf.getvalue()
        # Null-terminate so a runtime cuModuleLoadData call (Phase-1) sees
        # a proper C string. Phase-0 only needs the bytes to be addressable.
        ptx_bytes = ptx_text.encode("utf-8") + b"\x00"
        # One symbol per kernel (RIP-relative LEA targets), plus a single
        # module-wide symbol for the full text — all defined BEFORE the
        # bytes so they all point at the start of the PTX blob. (Phase-0
        # uses one shared PTX module per binary; isolation can land later.)
        buf.define_symbol("__helix_ptx_module")
        for kf in kernel_fns:
            buf.define_symbol(f"__helix_ptx_{kf.name}")
        buf.emit_bytes(ptx_bytes)
        # End-of-PTX marker so a runtime stub can compute the byte length
        # from `__helix_ptx_module_end - __helix_ptx_module`.
        buf.define_symbol("__helix_ptx_module_end")

    # Reflection cells: append HELIX_NUM_CELLS * 8 bytes of zero-init storage
    # immediately after the code. The base symbol is __helix_state_base.
    # MODIFY/SPLICE codegen uses RIP-relative LEA to address it.
    buf.define_symbol("__helix_state_base")
    buf.emit_bytes(b"\x00" * (HELIX_NUM_CELLS * HELIX_CELL_SIZE))

    # Arena region: a single shared bump-allocated i32 buffer. Slot 0
    # holds the cursor (current length); slots 1..HELIX_ARENA_CAP hold
    # data. Used by self-host machinery for AST/IR/symbol-table storage.
    # Lives in BSS — symbol points at the position right after the code
    # but the bytes are NOT in the file. p_memsz > p_filesz makes the
    # kernel zero-fill the arena range. This keeps produced binaries
    # small (~2-30K instead of 130K+) and lets us bump HELIX_ARENA_CAP
    # to large values without disk-cost penalty.
    buf.define_symbol("__helix_arena_base")
    arena_extra = (HELIX_ARENA_CAP + 1) * 4

    buf.patch()

    # Stage 16.5: if any FFI imports were recorded during codegen, emit a
    # dynamic-link ELF with the appropriate phdrs + .dynamic / .dynsym /
    # .dynstr / .rela.plt / .got.plt sections. Otherwise fall back to the
    # libc-free single-PT_LOAD path.
    if buf.dyn.has_imports():
        layout = elf_dyn.plan_layout(bytes(buf.bytes_), buf.dyn, arena_extra)
        # Patch each FFI fixup to reference the correct GOT slot's vaddr,
        # rip-relative to the call instruction.
        for fx in buf.ffi_fixups:
            slot_idx = buf.dyn._imports_set[fx.symbol]
            target_addr = layout.got_addr(slot_idx)
            # call_qword_ptr [rip+disp32] is FF 15 <disp32>; the rip used
            # by the CPU at decode is the address of the instruction
            # immediately after the disp32 (= call_site_vaddr + 6).
            # `fx.offset` is the file/buf offset of the disp32 bytes,
            # which equals call_site_offset + 2. The instruction-after-
            # disp32 in vaddr space is code_vaddr + fx.offset + 4.
            rip_after = layout.code_vaddr + fx.offset + 4
            disp = target_addr - rip_after
            if not (-(1 << 31) <= disp < (1 << 31)):
                raise OverflowError(
                    f"FFI call disp32 out of range for {fx.symbol}: {disp}")
            struct.pack_into("<i", buf.bytes_, fx.offset, disp)
        # Entry symbol "_start" is at the very beginning of the code buf.
        entry_off = buf.symbols.get("_start", 0)
        return elf_dyn.emit_elf_dyn(bytes(buf.bytes_), buf.dyn,
                                    entry_offset=entry_off,
                                    arena_extra=arena_extra)
    return emit_elf(bytes(buf.bytes_), extra_memsz=arena_extra)


if __name__ == "__main__":
    import sys
    from ..frontend.parser import parse
    from ..frontend.typecheck import typecheck
    from ..frontend.grad_pass import grad_pass
    from ..frontend.monomorphize import monomorphize
    from ..frontend.struct_mono import monomorphize_structs
    from ..frontend.flatten_modules import flatten_modules
    from ..frontend.flatten_impls import flatten_impls
    from ..ir.lower_ast import lower
    from ..ir.passes.const_fold import fold_module
    from ..ir.passes.dce import dce_module
    from ..ir.passes.cse import cse_module
    from ..ir.passes.fdce import fdce_module
    from ..ir.passes.effect_check import check_module as effect_check_module
    from ..frontend.totality import check_totality
    from ..frontend.hash_cons import hash_cons

    if len(sys.argv) < 3:
        print("usage: python -m helixc.backend.x86_64 <input.hx> <output.bin> "
              "[--strict] [--no-opt] [--no-stdlib]",
              file=sys.stderr)
        sys.exit(1)
    strict = "--strict" in sys.argv
    no_opt = "--no-opt" in sys.argv
    no_stdlib = "--no-stdlib" in sys.argv
    with open(sys.argv[1]) as f:
        src = f.read()
    # Auto-include stdlib by default. The fdce / dce passes drop unused
    # stdlib fns so the binary cost is zero. Pass --no-stdlib to compile
    # without it (only useful for stdlib internals or custom-runtime tests).
    prog = parse(src, include_stdlib=not no_stdlib)
    mod_count = flatten_modules(prog)
    if mod_count > 0:
        print(f"mod: {mod_count} item(s) lifted from block modules", file=sys.stderr)
    impl_count = flatten_impls(prog)
    if impl_count > 0:
        print(f"impl: {impl_count} method(s) lifted from impl blocks", file=sys.stderr)
    # Audit 28.8 A3/B1 — parametric-struct mono must run BEFORE fn mono
    # so the fn-mono pass picks up the mangled struct names.
    prog, sm_diags = monomorphize_structs(prog)
    if sm_diags:
        for d in sm_diags:
            print(f"warning: struct-mono: {d}", file=sys.stderr)
    mono_count = monomorphize(prog)
    if mono_count > 0:
        print(f"mono: {mono_count} generic instantiation(s)", file=sys.stderr)
    grad_count = grad_pass(prog)
    if grad_count > 0:
        print(f"grad: {grad_count} grad(f) call(s) rewritten", file=sys.stderr)
    # Type-check; print as warnings, abort if --strict.
    # NOTE: grad_pass internally invokes lower_matches(), which desugars
    # match -> if/let chains. So typecheck sees the lowered form, which
    # surfaces fake "enum variant has payload" / "if/else branches differ"
    # warnings against patterns like Option::Some(x). These are typecheck
    # imprecision against the lowered form — the original Match was valid.
    # Suppressing those would require either teaching typecheck about the
    # lowered form, or splitting grad_pass so lower_matches runs after
    # typecheck. Tracked as a separate audit item.
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
    # Stage 20 — AST hash-cons. Identical sub-expressions across the
    # program now share a single Python object. Lowering treats shared
    # nodes idempotently (same value-id reuse), so the IR module is
    # smaller — fewer SSA values, fewer ops for the downstream passes
    # to walk. Trap 20001 raises if the SHA-256 hasher reports two
    # structurally distinct subtrees colliding (it shouldn't, but the
    # guard exists so a future hash-fn swap surfaces silently-wrong
    # sharing).
    n_shared = hash_cons(prog)
    if n_shared > 0:
        print(f"hash-cons: {n_shared} AST node(s) deduped", file=sys.stderr)

    # Stage 21 — totality check on the AST (structural-recursion).
    # Runs before lowering so the diagnostic points at the original source.
    # Non-@partial recursive functions without a strictly-decreasing
    # parameter are flagged. --strict turns the warning into an abort.
    tot_fails = check_totality(prog)
    if tot_fails:
        for name, reason in tot_fails:
            print(f"warning: [trap 21001] totality: {name}: {reason}",
                  file=sys.stderr)
        if strict:
            print(f"\n{len(tot_fails)} totality failure(s); --strict aborts.",
                  file=sys.stderr)
            sys.exit(1)

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
        f_removed = fdce_module(mod)
        if f_removed > 0:
            print(f"fdce: {f_removed} unused fn(s) removed", file=sys.stderr)

    # Stage 19 — IR-level effect check. Runs AFTER all optimization passes
    # because fdce/dce can prune call edges (removing transitive effects)
    # and we want the post-opt closure to be authoritative. Reports each
    # @pure fn whose closure is non-empty (trap 19001) and each fn whose
    # declared effect set differs from its actual closure.
    eff_errs = effect_check_module(mod)
    if eff_errs:
        for e in eff_errs:
            print(f"warning: [trap 19001] effect-check: {e}", file=sys.stderr)
        if strict:
            print(f"\n{len(eff_errs)} effect-check failure(s); --strict aborts.",
                  file=sys.stderr)
            sys.exit(1)

    elf = compile_module_to_elf(mod)
    with open(sys.argv[2], "wb") as f:
        f.write(elf)
    import os
    os.chmod(sys.argv[2], 0o755)
    print(f"Wrote {sys.argv[2]} ({len(elf)} bytes)")
