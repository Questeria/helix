"""
helixc/backend/elf_dyn.py — ELF emission with dynamic linking (Stage 16.5).

Produces a Linux x86-64 ELF executable that imports symbols from libc.so.6
via the standard dynamic linker (/lib64/ld-linux-x86-64.so.2). Used by
Helix programs that contain `extern "C" fn ...;` declarations.

Approach (Phase-0):
- BIND_NOW: every imported symbol resolved at exec start; no PLT
  trampolines, no lazy binding, no PLT0.
- Single .got.plt (one 8-byte slot per imported symbol).
- One R_X86_64_JUMP_SLOT relocation per imported symbol in .rela.plt.
- Indirect calls: `call qword ptr [rip + got_offset]` (FF 15 rel32).
- Single R+W+X PT_LOAD covering ehdr + phdrs + interp + code +
  reflection cells + dyn-link tables. Same W^X-relaxed posture as the
  libc-free binary, kept consistent for Phase-0 simplicity.

Layout (single PT_LOAD, plus PT_PHDR/PT_INTERP/PT_DYNAMIC views):

    [ehdr]                     0x000
    [phdrs]
    [interp string]            "/lib64/ld-linux-x86-64.so.2\\0"
    [pad to 0x1000]
    [code]                     starts at file offset 0x1000
    [reflection cells]
    [string literal bodies]
    [dyn-link tables]
        .dynstr
        .dynsym
        .hash (SYSV)
        .rela.plt
        .dynamic
        .got.plt   ← writable; loader fills in resolved addresses
    [BSS arena tail]           p_memsz > p_filesz (zero-filled by kernel)

License: Apache 2.0
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field


# ============================================================================
# ELF / x86-64 constants we need
# ============================================================================
ELFMAG = b"\x7fELF"
ELFCLASS64 = 2
ELFDATA2LSB = 1
EV_CURRENT = 1
ELFOSABI_SYSV = 0
ET_EXEC = 2
EM_X86_64 = 0x3E

# Program header types
PT_NULL = 0
PT_LOAD = 1
PT_DYNAMIC = 2
PT_INTERP = 3
PT_PHDR = 6

# Program header flags
PF_X = 0x1
PF_W = 0x2
PF_R = 0x4

# Dynamic table tags (subset)
DT_NULL = 0
DT_NEEDED = 1
DT_PLTRELSZ = 2
DT_PLTGOT = 3
DT_HASH = 4
DT_STRTAB = 5
DT_SYMTAB = 6
DT_STRSZ = 10
DT_SYMENT = 11
DT_RELA = 7
DT_RELASZ = 8
DT_RELAENT = 9
DT_PLTREL = 20
DT_JMPREL = 23
DT_BIND_NOW = 24
DT_FLAGS = 30
DT_FLAGS_1 = 0x6ffffffb
# Flag bits
DF_BIND_NOW = 0x8
DF_1_NOW = 0x1

# Symbol info (st_info)
STB_GLOBAL = 1
STT_FUNC = 2
STT_NOTYPE = 0


def st_info(bind: int, type_: int) -> int:
    return (bind << 4) | (type_ & 0xF)


# Relocation types (x86-64)
R_X86_64_JUMP_SLOT = 7

# ELF64 sizes
SIZE_EHDR = 64
SIZE_PHDR = 56
SIZE_SYM = 24       # Elf64_Sym
SIZE_RELA = 24      # Elf64_Rela
SIZE_DYN = 16       # Elf64_Dyn

# Layout — single R+W+X PT_LOAD starting at ELF_BASE (0x400000).
ELF_BASE = 0x400000
CODE_OFFSET = 0x1000        # file offset where code begins
CODE_VADDR = ELF_BASE + CODE_OFFSET


# ============================================================================
# DynamicLinkInfo: collected from the codegen for each unique extern symbol
# ============================================================================
@dataclass
class DynLinkInfo:
    """One per binary. Ordered list of imports + bookkeeping the codegen
    fills as it sees FFI_CALL ops. The ELF emitter consumes this to lay
    out .dynsym/.dynstr/.rela.plt/.got.plt and produce the dyn-segments."""
    # Imports in the order they were first seen. Index in this list is the
    # GOT-entry slot index. Symbol name (e.g. "puts").
    imports: list[str] = field(default_factory=list)
    _imports_set: dict[str, int] = field(default_factory=dict)
    # Library names that DT_NEEDED entries should reference. Default just
    # libc.so.6; later stages can add libcuda.so.1 etc.
    needed_libs: list[str] = field(default_factory=lambda: ["libc.so.6"])

    def add_import(self, name: str) -> int:
        """Idempotent — return the GOT slot index (zero-based) for `name`."""
        if name in self._imports_set:
            return self._imports_set[name]
        idx = len(self.imports)
        self.imports.append(name)
        self._imports_set[name] = idx
        return idx

    def has_imports(self) -> bool:
        return len(self.imports) > 0

    def got_entry_count(self) -> int:
        return len(self.imports)


# ============================================================================
# Layout planner
# ============================================================================
@dataclass
class DynLayout:
    """All vaddrs are absolute (already include ELF_BASE). All offsets are
    file offsets within the produced ELF."""
    interp_str: bytes
    interp_vaddr: int
    interp_offset: int
    interp_size: int

    code_vaddr: int
    code_offset: int
    code_size: int           # code bytes from the codegen (including string lits, cells)

    dynstr_bytes: bytes
    dynstr_vaddr: int
    dynstr_offset: int
    dynstr_size: int

    dynsym_bytes: bytes
    dynsym_vaddr: int
    dynsym_offset: int
    dynsym_size: int

    hash_bytes: bytes
    hash_vaddr: int
    hash_offset: int
    hash_size: int

    rela_plt_bytes: bytes
    rela_plt_vaddr: int
    rela_plt_offset: int
    rela_plt_size: int

    dynamic_bytes: bytes
    dynamic_vaddr: int
    dynamic_offset: int
    dynamic_size: int

    got_plt_vaddr: int
    got_plt_offset: int
    got_plt_size: int

    # Combined PT_LOAD bounds.
    load_filesz: int
    load_memsz: int

    n_phdrs: int

    def got_addr(self, idx: int) -> int:
        return self.got_plt_vaddr + idx * 8


def plan_layout(code: bytes, dyn: DynLinkInfo, arena_extra: int) -> DynLayout:
    """Plan the absolute file-offset and vaddr of every region.

    The PT_LOAD covers the entire range from ehdr (offset 0) up to the
    end of .got.plt. arena_extra is added to memsz for the BSS arena.
    """
    interp_str = b"/lib64/ld-linux-x86-64.so.2\x00"
    n_phdrs = 4  # PHDR, INTERP, LOAD, DYNAMIC
    phdr_size = n_phdrs * SIZE_PHDR
    interp_offset = SIZE_EHDR + phdr_size
    interp_size = len(interp_str)
    interp_vaddr = ELF_BASE + interp_offset

    if interp_offset + interp_size > CODE_OFFSET:
        raise RuntimeError(
            f"phdrs+interp ({interp_offset+interp_size}) exceed code offset "
            f"{CODE_OFFSET}")

    code_offset = CODE_OFFSET
    code_vaddr = CODE_VADDR
    code_size = len(code)

    # ---- .dynstr ----
    dynstr_data = bytearray()
    dynstr_data.append(0)
    str_offsets: dict[str, int] = {"": 0}

    def add_str(s: str) -> int:
        if s in str_offsets:
            return str_offsets[s]
        off = len(dynstr_data)
        str_offsets[s] = off
        dynstr_data.extend(s.encode("utf-8"))
        dynstr_data.append(0)
        return off

    lib_offsets = [add_str(lib) for lib in dyn.needed_libs]
    sym_name_offsets = [add_str(name) for name in dyn.imports]
    while len(dynstr_data) % 8:
        dynstr_data.append(0)
    dynstr_bytes = bytes(dynstr_data)

    # ---- .dynsym ----
    dynsym_data = bytearray(SIZE_SYM)  # symbol 0 = STN_UNDEF
    for sym_off in sym_name_offsets:
        dynsym_data.extend(struct.pack("<I", sym_off))
        dynsym_data.append(st_info(STB_GLOBAL, STT_FUNC))
        dynsym_data.append(0)
        dynsym_data.extend(struct.pack("<H", 0))   # st_shndx = SHN_UNDEF
        dynsym_data.extend(struct.pack("<Q", 0))   # st_value
        dynsym_data.extend(struct.pack("<Q", 0))   # st_size
    dynsym_bytes = bytes(dynsym_data)

    # ---- .hash (SYSV) ----
    n_syms = 1 + len(dyn.imports)
    nbucket = 1
    nchain = n_syms
    bucket_vals = [1 if n_syms > 1 else 0]
    chain_vals = [0]  # UND has no successor
    for i in range(1, n_syms - 1):
        chain_vals.append(i + 1)
    if n_syms > 1:
        chain_vals.append(0)  # last symbol ends the chain
    hash_data = bytearray()
    hash_data.extend(struct.pack("<I", nbucket))
    hash_data.extend(struct.pack("<I", nchain))
    for b in bucket_vals:
        hash_data.extend(struct.pack("<I", b))
    for c in chain_vals:
        hash_data.extend(struct.pack("<I", c))
    while len(hash_data) % 8:
        hash_data.append(0)
    hash_bytes = bytes(hash_data)

    # ---- Place dyn-link tables right after the code buf in the same load segment.
    # Tables order: .dynstr, .dynsym, .hash, .rela.plt, .dynamic, .got.plt
    cur_off = code_offset + code_size
    # 8-byte align — dyn tables expect alignment.
    cur_off = (cur_off + 7) & ~7

    dynstr_offset = cur_off
    dynstr_vaddr = ELF_BASE + dynstr_offset
    dynstr_size = len(dynstr_bytes)
    cur_off += dynstr_size

    dynsym_offset = cur_off
    dynsym_vaddr = ELF_BASE + dynsym_offset
    dynsym_size = len(dynsym_bytes)
    cur_off += dynsym_size

    hash_offset = cur_off
    hash_vaddr = ELF_BASE + hash_offset
    hash_size = len(hash_bytes)
    cur_off += hash_size

    # ---- .rela.plt placeholder (we know its size) ----
    rela_plt_size = SIZE_RELA * len(dyn.imports)
    rela_plt_offset = cur_off
    rela_plt_vaddr = ELF_BASE + rela_plt_offset
    cur_off += rela_plt_size

    # ---- .dynamic placeholder ----
    n_dyn_entries = len(dyn.needed_libs) + 12
    dynamic_size = n_dyn_entries * SIZE_DYN
    dynamic_offset = cur_off
    dynamic_vaddr = ELF_BASE + dynamic_offset
    cur_off += dynamic_size

    # ---- .got.plt (8 bytes per import) ----
    got_plt_offset = cur_off
    got_plt_vaddr = ELF_BASE + got_plt_offset
    got_plt_size = 8 * len(dyn.imports)
    cur_off += got_plt_size

    # ---- Build .rela.plt now we know GOT addresses ----
    rela_data = bytearray()
    for i in range(len(dyn.imports)):
        r_offset = got_plt_vaddr + i * 8
        sym_idx = i + 1
        r_info = (sym_idx << 32) | R_X86_64_JUMP_SLOT
        r_addend = 0
        rela_data.extend(struct.pack("<QQq", r_offset, r_info, r_addend))
    rela_plt_bytes = bytes(rela_data)
    assert len(rela_plt_bytes) == rela_plt_size

    # ---- Build .dynamic ----
    dyn_entries: list[tuple[int, int]] = []
    for lib_off in lib_offsets:
        dyn_entries.append((DT_NEEDED, lib_off))
    dyn_entries.append((DT_HASH, hash_vaddr))
    dyn_entries.append((DT_STRTAB, dynstr_vaddr))
    dyn_entries.append((DT_SYMTAB, dynsym_vaddr))
    dyn_entries.append((DT_STRSZ, dynstr_size))
    dyn_entries.append((DT_SYMENT, SIZE_SYM))
    dyn_entries.append((DT_PLTGOT, got_plt_vaddr))
    dyn_entries.append((DT_PLTRELSZ, rela_plt_size))
    dyn_entries.append((DT_PLTREL, DT_RELA))
    dyn_entries.append((DT_JMPREL, rela_plt_vaddr))
    dyn_entries.append((DT_FLAGS, DF_BIND_NOW))
    dyn_entries.append((DT_FLAGS_1, DF_1_NOW))
    dyn_entries.append((DT_NULL, 0))
    assert len(dyn_entries) == n_dyn_entries
    dynamic_data = bytearray()
    for tag, val in dyn_entries:
        dynamic_data.extend(struct.pack("<qQ", tag, val))
    dynamic_bytes = bytes(dynamic_data)

    load_filesz = cur_off  # everything from offset 0 to end of .got.plt
    load_memsz = load_filesz + arena_extra

    return DynLayout(
        interp_str=interp_str,
        interp_vaddr=interp_vaddr,
        interp_offset=interp_offset,
        interp_size=interp_size,
        code_vaddr=code_vaddr,
        code_offset=code_offset,
        code_size=code_size,
        dynstr_bytes=dynstr_bytes,
        dynstr_vaddr=dynstr_vaddr,
        dynstr_offset=dynstr_offset,
        dynstr_size=dynstr_size,
        dynsym_bytes=dynsym_bytes,
        dynsym_vaddr=dynsym_vaddr,
        dynsym_offset=dynsym_offset,
        dynsym_size=dynsym_size,
        hash_bytes=hash_bytes,
        hash_vaddr=hash_vaddr,
        hash_offset=hash_offset,
        hash_size=hash_size,
        rela_plt_bytes=rela_plt_bytes,
        rela_plt_vaddr=rela_plt_vaddr,
        rela_plt_offset=rela_plt_offset,
        rela_plt_size=rela_plt_size,
        dynamic_bytes=dynamic_bytes,
        dynamic_vaddr=dynamic_vaddr,
        dynamic_offset=dynamic_offset,
        dynamic_size=dynamic_size,
        got_plt_vaddr=got_plt_vaddr,
        got_plt_offset=got_plt_offset,
        got_plt_size=got_plt_size,
        load_filesz=load_filesz,
        load_memsz=load_memsz,
        n_phdrs=n_phdrs,
    )


def emit_elf_dyn(code: bytes, dyn: DynLinkInfo,
                 entry_offset: int = 0,
                 arena_extra: int = 0) -> bytes:
    """Emit a Linux x86-64 ELF executable with dynamic linking.

    `code` is the raw code-segment bytes (entry stub + user fns + strings
    + reflection cells + any other in-segment data). `entry_offset` is
    the byte offset of `_start` within the code buffer (0 unless callers
    insert content before the entry stub). `arena_extra` extends the
    PT_LOAD memsz so the kernel zero-fills the arena range — same
    semantics as emit_elf().
    """
    layout = plan_layout(code, dyn, arena_extra)

    # ---- ELF header ----
    ehdr = bytearray()
    ehdr += ELFMAG
    ehdr += bytes([ELFCLASS64, ELFDATA2LSB, EV_CURRENT, ELFOSABI_SYSV])
    ehdr += b"\x00" * 8                # EI_PAD
    ehdr += struct.pack("<H", ET_EXEC)
    ehdr += struct.pack("<H", EM_X86_64)
    ehdr += struct.pack("<I", EV_CURRENT)
    ehdr += struct.pack("<Q", layout.code_vaddr + entry_offset)
    ehdr += struct.pack("<Q", SIZE_EHDR)               # e_phoff
    ehdr += struct.pack("<Q", 0)                       # e_shoff
    ehdr += struct.pack("<I", 0)                       # e_flags
    ehdr += struct.pack("<H", SIZE_EHDR)               # e_ehsize
    ehdr += struct.pack("<H", SIZE_PHDR)               # e_phentsize
    ehdr += struct.pack("<H", layout.n_phdrs)          # e_phnum
    ehdr += struct.pack("<H", 0)                       # e_shentsize
    ehdr += struct.pack("<H", 0)                       # e_shnum
    ehdr += struct.pack("<H", 0)                       # e_shstrndx
    assert len(ehdr) == SIZE_EHDR

    # ---- Program headers ----
    ph_off = SIZE_EHDR
    ph_size = layout.n_phdrs * SIZE_PHDR
    phdr_phdr = struct.pack(
        "<IIQQQQQQ",
        PT_PHDR, PF_R,
        ph_off, ELF_BASE + ph_off, ELF_BASE + ph_off,
        ph_size, ph_size, 8,
    )
    phdr_interp = struct.pack(
        "<IIQQQQQQ",
        PT_INTERP, PF_R,
        layout.interp_offset, layout.interp_vaddr, layout.interp_vaddr,
        layout.interp_size, layout.interp_size, 1,
    )
    # Single PT_LOAD: R+W+X covering everything.
    phdr_load = struct.pack(
        "<IIQQQQQQ",
        PT_LOAD, PF_R | PF_W | PF_X,
        0, ELF_BASE, ELF_BASE,
        layout.load_filesz, layout.load_memsz, 0x1000,
    )
    phdr_dynamic = struct.pack(
        "<IIQQQQQQ",
        PT_DYNAMIC, PF_R | PF_W,
        layout.dynamic_offset, layout.dynamic_vaddr, layout.dynamic_vaddr,
        layout.dynamic_size, layout.dynamic_size, 8,
    )
    phdrs = phdr_phdr + phdr_interp + phdr_load + phdr_dynamic
    assert len(phdrs) == ph_size

    # ---- Assemble the ELF file ----
    out = bytearray()
    out.extend(ehdr)
    out.extend(phdrs)
    assert len(out) == layout.interp_offset
    out.extend(layout.interp_str)
    while len(out) < layout.code_offset:
        out.append(0)
    assert len(out) == layout.code_offset
    out.extend(code)
    # Pad to dynstr offset (covers any 8-byte alignment).
    while len(out) < layout.dynstr_offset:
        out.append(0)
    assert len(out) == layout.dynstr_offset
    out.extend(layout.dynstr_bytes)
    assert len(out) == layout.dynsym_offset
    out.extend(layout.dynsym_bytes)
    assert len(out) == layout.hash_offset
    out.extend(layout.hash_bytes)
    assert len(out) == layout.rela_plt_offset
    out.extend(layout.rela_plt_bytes)
    assert len(out) == layout.dynamic_offset
    out.extend(layout.dynamic_bytes)
    assert len(out) == layout.got_plt_offset
    out.extend(b"\x00" * layout.got_plt_size)
    assert len(out) == layout.load_filesz
    return bytes(out)
