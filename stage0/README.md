# Kovostov-Native — stage0

The bottom of the bootstrap chain. Hand-encoded bytes that the user can audit one byte at a time. Everything in this directory is the *root of trust*.

## Chain overview

```
[hand-encoded bytes]
       │
       ▼
   hex0  ── reads hex characters from stdin, writes bytes to stdout
       │
       ▼
   hex1  ── adds: labels, comments
       │
       ▼
   M0    ── minimal macro assembler (mnemonics, register names)
       │
       ▼
   M1    ── richer macros, basic structures
       │
       ▼
   M2-Planet  ── tiny C subset compiler (vendored at this point)
       │
       ▼
   helix-libc + helixc-bootstrap  ── written in M2 C-subset
       │
       ▼
   helixc  ── self-hosted in Helix
```

## Authorship policy

- **stage0/hex0/** — **fully hand-authored from raw bytes**. This is the literal "raw binary as starting point" hard constraint. Each byte is reasoned-about and annotated. Audit cross-checked against `oriansj/stage0-posix-amd64/hex0_AMD64.hex0` for byte-level encoding correctness, but the bytes shipped here are ours.
- **stage0/hex1/** through **stage0/M2-Planet/** — re-evaluation gate at month-2. Default: vendor `oriansj/stage0-posix` + `oriansj/mescc-tools` + `oriansj/M2-Planet` at pinned tags (saves ~6 person-months). Each binary is byte-auditable against its annotated source.
- **stage0/helix-libc/** — written by us in M2-Planet C-subset. Avoids GPL-3.0 contagion. ~200–500 LOC.

## Verification

Every shipped binary has:
- Source (annotated text form: hex digits with `;` comments OR M0/M1 macro source OR C source)
- Binary (`.bin`)
- SHA-256 hash file (`.sha256`)
- Build script that produces the binary from source using only the previous stage's tools
- Disassembly-based audit notes (`disasm.md`) where applicable

## Targets

- Bootstrap chain: **linux-x86_64 ELF** (via WSL2 on Windows)
- Eventual `helixrt` runtime: Windows-native (CUDA Driver API → RTX 5090). Bootstrap and runtime can target different OSes.

## License

GPL-3.0 sources (vendored stage0-posix / mescc-tools / M2-Planet) are kept separately from helix-libc/helixc-bootstrap which are Apache 2.0. The two trees are statically separable.
