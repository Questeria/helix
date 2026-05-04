# hex0.bytes.md — annotated bytes for hex0.bin

**Status: NOT YET HAND-ENCODED.** This file is the canonical byte-by-byte annotation
that produces `hex0.bin`. It is populated in the next session, after one of:

1. **Path A (preferred):** `nasm` is installed (`sudo apt install nasm` in WSL),
   we run `nasm -f bin -o hex0.nasm.bin hex0.s`, then we hand-type bytes from
   the resulting `hex0.nasm.bin` into this annotated form, byte-by-byte. We
   then build `hex0.bin` from this file using `xxd -r -p` (audit-only tool) and
   `cmp` it against `hex0.nasm.bin` for byte-identical equivalence.

2. **Path B (purer, slower):** Compute every byte by hand from the Intel SDM
   and the encoding reference in `docs/research/2026-05-03-elf-syscalls-encoding.md`.
   Each instruction's bytes derived independently, jump offsets back-filled in
   a second pass. Cross-check against the `oriansj/stage0-posix-amd64`
   `hex0_AMD64.hex0` annotated source for analogous instructions.

## Format

```
<file_offset_hex>  <bytes>  ; mnemonic — comment
```

Example (placeholder):
```
0x0000  7F 45 4C 46           ; ELF magic
0x0004  02                    ; EI_CLASS = ELFCLASS64
...
0x0078  31 ED                 ; xor ebp, ebp                   ; _start
0x007A  50                    ; push rax                       ; read_loop
...
```

## Layout (planned)

| Section | File offset range | Size |
|---|---|---|
| ELF header | 0x00–0x3F | 64 bytes |
| Program header | 0x40–0x77 | 56 bytes |
| `_start` / `read_loop` | 0x78–... | ~50 bytes |
| dispatch (cmp/je chain) | ... | ~40 bytes |
| `digit_0_9` / `digit_A_F` / `digit_a_f` | ... | ~12 bytes |
| `got_nibble` / `combine` | ... | ~50 bytes |
| `skip_comment` | ... | ~25 bytes |
| `do_exit` | ... | ~12 bytes |

Approximate total: 260–320 bytes.

## Verification

Once both `hex0.bin` and `hex0.nasm.bin` exist:
- `cmp hex0.bin hex0.nasm.bin` → must be byte-identical
- `./hex0.bin` must pass all tests in `test/`
- `python hex0_reference.py` must produce the same output as `./hex0.bin` on every fixture
- `objdump -d` disassembly of `hex0.bin` must match the assembly comments in `hex0.s`

After verification, `hex0.bin` is the canonical shipped artifact. `hex0.nasm.bin`
is deleted; nasm is never invoked again for shipping. The annotated
`hex0.bytes.md` is the canonical *source* of `hex0.bin` going forward.
