# Stage0 / Live Bootstrap / GNU Mes ecosystem

Captured 2026-05-03 from research agent. Use to decide how much of the chain we author from scratch vs adopt.

## Inventory

| Project | URL | Role |
|---|---|---|
| bootstrappable.org | https://bootstrappable.org/ | umbrella initiative |
| oriansj/stage0 | https://github.com/oriansj/stage0 | original chain (Knight VM + POSIX) |
| oriansj/stage0-posix | https://github.com/oriansj/stage0-posix | the canonical POSIX chain |
| oriansj/bootstrap-seeds | https://github.com/oriansj/bootstrap-seeds | 256-byte seed binaries |
| oriansj/M2-Planet | https://github.com/oriansj/M2-Planet | C-subset compiler |
| oriansj/M2libc | https://github.com/oriansj/M2libc | minimal libc in M2 C-subset |
| oriansj/mescc-tools | https://github.com/oriansj/mescc-tools | M1 macro asm + hex2 linker + kaem shell |
| GNU Mes | https://www.gnu.org/software/mes/ | Scheme interpreter + MesCC |
| Live Bootstrap | https://github.com/fosslinux/live-bootstrap | end-to-end automated chain to GCC + Linux |

All licensed GPL-3.0+. Active as of 2025–2026.

## Recommended approach for Kovostov-Native

**Hybrid (decided 2026-05-03):**

1. **Hex0 — write our own** from raw bytes. This is the literal "raw binary as starting point" hard constraint and is the durable claim. ~150–200 bytes hand-encoded. Cross-check against `oriansj/stage0-posix-amd64/hex0_AMD64.hex0` for byte-level correctness verification (audit-only, not adoption).
2. **Hex1 → M2-Planet — re-evaluate at month-2 gate**. Default: vendor `oriansj/stage0-posix` and `oriansj/mescc-tools` and `oriansj/M2-Planet` at pinned tags, treat as audited upstream. Saves ~6 person-months. Each binary is auditable byte-by-byte against its own annotated source if needed.
3. **helix-libc — write our own** in M2-Planet C subset (~200–500 LOC). Avoids GPL-3.0 contagion from M2libc into helixc-bootstrap.
4. **helixc-bootstrap.c — write our own** in M2-Planet C subset (~5–10 kLOC).
5. **helixc — self-host in Helix.**

The user's hard constraint "raw binary as starting point" is satisfied by step 1 (we author every byte of the seed). Steps 2–5 are about chain pragmatism, not the constraint.

## M2-Planet C subset

Supported:
- `#define`, `#include`
- structs, arrays, pointers, function pointers
- `if/else`, `for`, `while`, `do/while`, `switch`, `goto`
- inline asm via `asm("...")` strings
- `char`, `int`, `long`, signed/unsigned

Not supported / limited:
- no floats / doubles
- no standard library
- no bitfields, no `union` (or very limited), no variadic functions outside M2libc
- minimal preprocessor
- not full C89

Size: ~3,000 lines in `cc.c` + readers/emitters. Test suite: 40+ tests.

## Pinned upstream versions (commit checkpoints)

To be set when we begin Phase 0b adoption:
- stage0-posix: tag `1.9.1` (2025-08-17)
- M2-Planet: tag `1.13.1` (2025-08-17)
- mescc-tools: latest tagged release matching the above
- bootstrap-seeds: pin commit hash

## Effort estimates (single dev + AI assistance)

| Option | Optimistic | Likely | Pessimistic |
|---|---|---|---|
| (a) Pure from-scratch hex0 → M2-equivalent | 6 PM | 9 PM | 18 PM |
| (b) Adopt hex0–M0, write own M1+ | 3 PM | 5 PM | 9 PM |
| **(c) Adopt through M2-Planet, write helixc-bootstrap.c** | 1 PM | 2 PM | 4 PM |
| (c) + write own helix-libc to escape GPL-3 | +0.5 PM | +1 PM | +2 PM |

## Gotchas

1. Linux ELF/syscall churn — modern kernel pickier about PT_GNU_STACK, e_phoff, mapped permissions
2. Architecture lock-in — each new ISA = fresh hex0 seed
3. `-O2` "helpful" gcc breaks the audit premise — use `-nostdinc -nostdlib -static -fno-stack-protector` everywhere
4. M2libc syscall numbers differ across Linux/FreeBSD/NetBSD — pick one
5. Determinism: no `__DATE__`/`__TIME__`, sorted readdir, pinned tool versions
6. **License contagion**: M2libc is GPL-3.0; if we link it into helixc-bootstrap, the binary inherits GPL-3.0. Solution: write our own helix-libc shim (~200–500 LOC).
7. bootstrap-seeds repo says "NEVER TRUST ANYTHING IN HERE" — audit ASCII vs binary once, commit both, pin SHA256.
8. M2-Planet's test corpus is shallow — write our own conformance tests for helixc-bootstrap features.
