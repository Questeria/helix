# ELF64 + Linux syscalls + x86-64 instruction encoding reference

Captured 2026-05-03 from research agent. Use for hex0 design.

## Minimum ELF64 header (64 bytes)

| Offset | Size | Bytes | Field | Meaning |
|--------|------|-------|-------|---------|
| 0x00 | 4 | `7F 45 4C 46` | e_ident[EI_MAG] | ELF magic |
| 0x04 | 1 | `02` | EI_CLASS | ELFCLASS64 |
| 0x05 | 1 | `01` | EI_DATA | little-endian |
| 0x06 | 1 | `01` | EI_VERSION | EV_CURRENT |
| 0x07 | 1 | `00` | EI_OSABI | System V (or 03 = FreeBSD) |
| 0x08 | 1 | `00` | EI_ABIVERSION | 0 |
| 0x09 | 7 | `00`*7 | EI_PAD | padding |
| 0x10 | 2 | `02 00` | e_type | ET_EXEC |
| 0x12 | 2 | `3E 00` | e_machine | EM_X86_64 |
| 0x14 | 4 | `01 00 00 00` | e_version | 1 |
| 0x18 | 8 | (entry vaddr LE) | e_entry | 0x600078 |
| 0x20 | 8 | `40 00 00 00 00 00 00 00` | e_phoff | 64 |
| 0x28 | 8 | `00`*8 | e_shoff | 0 |
| 0x30 | 4 | `00 00 00 00` | e_flags | 0 |
| 0x34 | 2 | `40 00` | e_ehsize | 64 |
| 0x36 | 2 | `38 00` | e_phentsize | 56 |
| 0x38 | 2 | `01 00` | e_phnum | 1 |
| 0x3A | 2 | `00 00` | e_shentsize | 0 |
| 0x3C | 2 | `00 00` | e_shnum | 0 |
| 0x3E | 2 | `00 00` | e_shstrndx | 0 |

## Program header (56 bytes)

| Offset | Size | Bytes | Field | Meaning |
|--------|------|-------|-------|---------|
| 0x40 | 4 | `01 00 00 00` | p_type | PT_LOAD |
| 0x44 | 4 | `05 00 00 00` | p_flags | PF_R \| PF_X |
| 0x48 | 8 | `00`*8 | p_offset | 0 |
| 0x50 | 8 | `00 00 60 00 00 00 00 00` | p_vaddr | 0x600000 |
| 0x58 | 8 | `00 00 60 00 00 00 00 00` | p_paddr | mirror |
| 0x60 | 8 | (filesz LE) | p_filesz | total bytes |
| 0x68 | 8 | (memsz LE) | p_memsz | ≥ filesz |
| 0x70 | 8 | `00 10 00 00 00 00 00 00` | p_align | 0x1000 |

Required: `p_vaddr ≡ p_offset (mod p_align)` — with offset=0 and vaddr=0x600000, both 0 mod 0x1000 ✓.

## Linux x86_64 syscalls

Calling convention:
- rax = syscall number
- rdi, rsi, rdx, r10, r8, r9 = args 1–6
- `syscall` instr (bytes `0F 05`)
- rax = return (or -errno)
- rcx, r11 clobbered; others preserved

| Name | rax |
|---|---|
| read | 0 |
| write | 1 |
| open | 2 |
| close | 3 |
| exit | 60 |
| exit_group | 231 |

stdin = 0, stdout = 1, stderr = 2.

## x86-64 instruction encodings (hex0 needs)

```
0F 05                    syscall
B8 ib*4                  mov eax, imm32
BF ib*4                  mov edi, imm32
BE ib*4                  mov esi, imm32
BA ib*4                  mov edx, imm32
B0 ib                    mov al, imm8
6A ib ; 58               push imm8 ; pop rax  (3-byte set-rax-small)
48 89 E6                 mov rsi, rsp
48 8B 34 24              mov rsi, [rsp]
50–57                    push rax,rcx,rdx,rbx,rsp,rbp,rsi,rdi
58–5F                    pop  same
3C ib                    cmp al, imm8
2C ib                    sub al, imm8
24 ib                    and al, imm8
04 ib                    add al, imm8
C0 E0 04                 shl al, 4
08 D8                    or  al, bl
93                       xchg eax, ebx
8A 06                    mov al, [rsi]
89 2E                    mov [rsi], ebp
74/75/7C/7D/7E/7F ib     je/jne/jl/jge/jle/jg rel8
EB ib                    jmp rel8
E9 id*4                  jmp rel32
E8 id*4                  call rel32
C3                       ret
31 FF                    xor edi, edi (zero rdi)
31 F6                    xor esi, esi
31 C0                    xor eax, eax
```

REX prefix `0x40 | (W<<3) | (R<<2) | (X<<1) | B` — `0x48` = REX.W (64-bit operand).

## Canonical hex0 reference

- `oriansj/stage0-posix-amd64/hex0_AMD64.hex0` — 229 bytes, fully annotated
- URL: https://github.com/oriansj/stage0-posix-amd64/blob/master/hex0_AMD64.hex0
- Stdin/stdout-only variant ~180–200 bytes (drops `open`/`close` calls + filename arg parsing)

## Sources

- Brian Raiter — Tiny ELF Executables — http://www.muppetlabs.com/~breadbox/software/tiny/teensy.html
- oriansj/stage0-posix-amd64 — https://github.com/oriansj/stage0-posix-amd64
- Filippo Valsorda's syscall table — https://filippo.io/linux-syscall-table/
- Linux kernel `arch/x86/entry/syscalls/syscall_64.tbl`
- OSDev x86-64 instruction encoding wiki — https://wiki.osdev.org/X86-64_Instruction_Encoding
- elf(5) man page — https://man7.org/linux/man-pages/man5/elf.5.html
