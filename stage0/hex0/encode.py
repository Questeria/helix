#!/usr/bin/env python3
"""
encode.py — Tiny Python "assembler" that produces hex0.bin from hand-encoded
instruction bytes with symbolic labels.

NOT a shipped artifact. The shipped artifact is hex0.bin (and its annotated
form hex0.hex). This script's role is purely to resolve label addresses into
relative jump displacements — work I would otherwise do by hand on paper.

Every byte not involving a label displacement is hand-typed below; every
displacement is computed from labels in a two-pass scheme.

Verification path:
  1. Run this script -> hex0.bin
  2. Run `xxd hex0.bin` -> compare to hex0.hex (which annotates every byte)
  3. Run `objdump -D -b binary -m i386:x86-64 -M intel \\
            --adjust-vma=0x600000 hex0.bin`
     -> compare disassembly against the mnemonics in hex0.s
  4. Run hex0.bin against test fixtures -> must match Python reference

License: Apache 2.0
"""

import struct
import sys


# Labels resolved in two passes: first pass computes addresses, second emits.
ELF_BASE = 0x600000

# We construct the output incrementally. Each item is either bytes or a
# Label or RelJump or Rel32Jump or PlaceholderForFilesz.

# Approach: build a list of "chunks". Each chunk is one of:
#   ('bytes', bytes)
#   ('label', name)            -> records current offset as symbol
#   ('rel8', target_label)     -> 1-byte displacement, offset = target - (here+1)
#   ('rel32', target_label)    -> 4-byte displacement, offset = target - (here+4)
#   ('imm32', label_or_value)  -> 4-byte little-endian (used for ELF fields)
#   ('imm64', label_or_value)  -> 8-byte little-endian
# 'filesz' is a special label whose value is the total file size at end.

chunks = []


def emit(b):
    """Emit raw bytes."""
    chunks.append(("bytes", b))


def label(name):
    chunks.append(("label", name))


def rel8(target):
    chunks.append(("rel8", target))


def rel32(target):
    chunks.append(("rel32", target))


def imm32_label(name):
    chunks.append(("imm32_label", name))


def imm64_label(name):
    chunks.append(("imm64_label", name))


def imm64(value):
    chunks.append(("bytes", struct.pack("<Q", value)))


def imm32(value):
    chunks.append(("bytes", struct.pack("<I", value)))


def imm16(value):
    chunks.append(("bytes", struct.pack("<H", value)))


# ============================================================================
# ELF64 header (64 bytes, file offsets 0x00-0x3F)
# ============================================================================
label("ehdr")
emit(b"\x7fELF")            # 0x00 e_ident[EI_MAG]
emit(b"\x02")               # 0x04 EI_CLASS = ELFCLASS64
emit(b"\x01")               # 0x05 EI_DATA  = ELFDATA2LSB
emit(b"\x01")               # 0x06 EI_VERSION = EV_CURRENT
emit(b"\x00")               # 0x07 EI_OSABI = System V
emit(b"\x00")               # 0x08 EI_ABIVERSION
emit(b"\x00" * 7)           # 0x09 EI_PAD
imm16(2)                    # 0x10 e_type = ET_EXEC
imm16(0x3E)                 # 0x12 e_machine = EM_X86_64
imm32(1)                    # 0x14 e_version
imm64_label("entry_vaddr")  # 0x18 e_entry
imm64(64)                   # 0x20 e_phoff = 64
imm64(0)                    # 0x28 e_shoff = 0
imm32(0)                    # 0x30 e_flags = 0
imm16(64)                   # 0x34 e_ehsize
imm16(56)                   # 0x36 e_phentsize
imm16(1)                    # 0x38 e_phnum
imm16(0)                    # 0x3A e_shentsize
imm16(0)                    # 0x3C e_shnum
imm16(0)                    # 0x3E e_shstrndx

# ============================================================================
# Program header (56 bytes, file offsets 0x40-0x77)
# ============================================================================
label("phdr")
imm32(1)                    # 0x40 p_type = PT_LOAD
imm32(5)                    # 0x44 p_flags = PF_R | PF_X
imm64(0)                    # 0x48 p_offset = 0
imm64_label("base_vaddr")   # 0x50 p_vaddr
imm64_label("base_vaddr")   # 0x58 p_paddr (mirror)
imm64_label("filesz")       # 0x60 p_filesz
imm64_label("filesz")       # 0x68 p_memsz (= filesz, no .bss)
imm64(0x1000)               # 0x70 p_align

# ============================================================================
# Code (entry = ELF_BASE + 0x78)
# ============================================================================
label("_start")

# xor ebp, ebp                    ; clear high-nibble flag
emit(b"\x31\xED")

label("read_loop")
# push rax                        ; reserve 1-byte stack buffer at [rsp]
emit(b"\x50")
# xor eax, eax                    ; sys_read = 0
emit(b"\x31\xC0")
# xor edi, edi                    ; fd = stdin
emit(b"\x31\xFF")
# mov rsi, rsp                    ; buf = stack
emit(b"\x48\x89\xE6")
# mov edx, 1                      ; count = 1
emit(b"\xBA\x01\x00\x00\x00")
# syscall
emit(b"\x0F\x05")
# test rax, rax                   ; rax = bytes read (0 = EOF, <0 = error)
emit(b"\x48\x85\xC0")
# jle do_exit (rel32, forward, distance > 127)
emit(b"\x0F\x8E"); rel32("do_exit")
# movzx eax, byte [rsp]           ; load read byte into al, zero rest
emit(b"\x0F\xB6\x04\x24")
# pop rcx                         ; deallocate stack slot (rcx clobbered ok)
emit(b"\x59")

# ---- dispatch on character in al ----

# cmp al, '#' / je skip_comment
emit(b"\x3C\x23"); emit(b"\x74"); rel8("skip_comment")
# cmp al, ';' / je skip_comment
emit(b"\x3C\x3B"); emit(b"\x74"); rel8("skip_comment")
# cmp al, ' ' / je read_loop
emit(b"\x3C\x20"); emit(b"\x74"); rel8("read_loop")
# cmp al, '\t' / je read_loop
emit(b"\x3C\x09"); emit(b"\x74"); rel8("read_loop")
# cmp al, '\n' / je read_loop
emit(b"\x3C\x0A"); emit(b"\x74"); rel8("read_loop")
# cmp al, '\r' / je read_loop
emit(b"\x3C\x0D"); emit(b"\x74"); rel8("read_loop")

# cmp al, '0' / jl read_loop
emit(b"\x3C\x30"); emit(b"\x7C"); rel8("read_loop")
# cmp al, '9' / jle digit_0_9
emit(b"\x3C\x39"); emit(b"\x7E"); rel8("digit_0_9")

# cmp al, 'A' / jl read_loop
emit(b"\x3C\x41"); emit(b"\x7C"); rel8("read_loop")
# cmp al, 'F' / jle digit_A_F
emit(b"\x3C\x46"); emit(b"\x7E"); rel8("digit_A_F")

# cmp al, 'a' / jl read_loop
emit(b"\x3C\x61"); emit(b"\x7C"); rel8("read_loop")
# cmp al, 'f' / jle digit_a_f
emit(b"\x3C\x66"); emit(b"\x7E"); rel8("digit_a_f")
# jmp read_loop                   ; above 'f': invalid, skip
emit(b"\xEB"); rel8("read_loop")

# ---- digit conversion ----

label("digit_0_9")
# sub al, 0x30                    ; '0'..'9' -> 0..9
emit(b"\x2C\x30")
# jmp got_nibble
emit(b"\xEB"); rel8("got_nibble")

label("digit_A_F")
# sub al, 0x37                    ; 'A'..'F' -> 10..15
emit(b"\x2C\x37")
# jmp got_nibble
emit(b"\xEB"); rel8("got_nibble")

label("digit_a_f")
# sub al, 0x57                    ; 'a'..'f' -> 10..15
emit(b"\x2C\x57")
# fall through to got_nibble

label("got_nibble")
# test bpl, 1                     ; need REX prefix to access bpl
# encoding: REX (0x40) + opcode F6 /0 + ModR/M for bpl + imm8
# 40 F6 C5 01
emit(b"\x40\xF6\xC5\x01")
# jnz combine
emit(b"\x75"); rel8("combine")

# First nibble: shift, store in bl, set flag
# shl al, 4
emit(b"\xC0\xE0\x04")
# mov bl, al
emit(b"\x88\xC3")
# mov ebp, 1
emit(b"\xBD\x01\x00\x00\x00")
# jmp read_loop
emit(b"\xEB"); rel8("read_loop")

label("combine")
# or al, bl                       ; al = (high << 4) | low
emit(b"\x08\xD8")
# push rax                        ; reserve stack slot
emit(b"\x50")
# mov [rsp], al                   ; place byte at top of stack
emit(b"\x88\x04\x24")
# xor eax, eax
emit(b"\x31\xC0")
# inc eax                         ; eax = 1 (sys_write)
emit(b"\xFF\xC0")
# mov edi, eax                    ; fd = stdout = 1
emit(b"\x89\xC7")
# mov rsi, rsp                    ; buf
emit(b"\x48\x89\xE6")
# mov edx, eax                    ; count = 1
emit(b"\x89\xC2")
# syscall
emit(b"\x0F\x05")
# pop rax                         ; deallocate
emit(b"\x58")
# xor ebp, ebp                    ; clear high-nibble flag
emit(b"\x31\xED")
# jmp read_loop  (rel32, distance > 127 backward)
emit(b"\xE9"); rel32("read_loop")

label("skip_comment")
# push rax
emit(b"\x50")
# xor eax, eax / xor edi, edi / mov rsi, rsp / mov edx, 1 / syscall
emit(b"\x31\xC0\x31\xFF\x48\x89\xE6\xBA\x01\x00\x00\x00\x0F\x05")
# test rax, rax / jle do_exit
emit(b"\x48\x85\xC0")
emit(b"\x7E"); rel8("do_exit")
# movzx eax, byte [rsp] / pop rcx
emit(b"\x0F\xB6\x04\x24\x59")
# cmp al, '\n' / jne skip_comment
emit(b"\x3C\x0A")
emit(b"\x75"); rel8("skip_comment")
# jmp read_loop  (rel32, > 127 backward)
emit(b"\xE9"); rel32("read_loop")

label("do_exit")
# xor edi, edi                    ; status = 0
emit(b"\x31\xFF")
# mov eax, 60                     ; sys_exit
emit(b"\xB8\x3C\x00\x00\x00")
# syscall
emit(b"\x0F\x05")

label("end")


# ============================================================================
# Two-pass resolver
# ============================================================================
def resolve():
    # Pass 1: assign addresses
    addr = 0
    addresses = {}
    for kind, *rest in chunks:
        if kind == "bytes":
            addr += len(rest[0])
        elif kind == "label":
            addresses[rest[0]] = addr
        elif kind == "rel8":
            addr += 1
        elif kind == "rel32":
            addr += 4
        elif kind in ("imm32_label", "imm64_label"):
            addr += 4 if kind == "imm32_label" else 8
        else:
            raise ValueError(f"unknown chunk kind: {kind}")
    addresses["filesz"] = addr
    addresses["base_vaddr"] = ELF_BASE
    addresses["entry_vaddr"] = ELF_BASE + addresses["_start"]

    # Pass 2: emit bytes
    out = bytearray()
    addr = 0
    for kind, *rest in chunks:
        if kind == "bytes":
            out.extend(rest[0])
            addr += len(rest[0])
        elif kind == "label":
            pass
        elif kind == "rel8":
            target = addresses[rest[0]]
            disp = target - (addr + 1)
            if not -128 <= disp <= 127:
                raise ValueError(f"rel8 to {rest[0]}: disp {disp} out of range "
                                 f"(target=0x{target:x}, here=0x{addr:x})")
            out.append(disp & 0xFF)
            addr += 1
        elif kind == "rel32":
            target = addresses[rest[0]]
            disp = target - (addr + 4)
            out.extend(struct.pack("<i", disp))
            addr += 4
        elif kind == "imm32_label":
            value = addresses[rest[0]]
            out.extend(struct.pack("<I", value))
            addr += 4
        elif kind == "imm64_label":
            value = addresses[rest[0]]
            out.extend(struct.pack("<Q", value))
            addr += 8

    return bytes(out), addresses


def main():
    binary, addresses = resolve()
    with open("hex0.bin", "wb") as f:
        f.write(binary)
    print(f"Wrote hex0.bin: {len(binary)} bytes")
    print(f"Entry: 0x{addresses['entry_vaddr']:x}")
    print(f"_start at file offset 0x{addresses['_start']:x}")
    print(f"do_exit at file offset 0x{addresses['do_exit']:x}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
