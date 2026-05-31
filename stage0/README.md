# Kovostov-Native — stage0

The bottom of the bootstrap chain. Hand-encoded bytes that the user can audit one byte at a time. Everything in this directory is the *root of trust*.

## Chain overview

```
[299 hand-encoded bytes]
       │
       ▼
   hex0       ── hex chars (stdin) → bytes (stdout); skips ws + ; / # comments
       │
       ▼
   hex1       ── adds single-char labels
       │
       ▼
   hex2       ── adds long labels + absolute addresses (acts as a linker)
       │
       ▼
   catm       ── file concatenation (catm OUT in1 in2 …); replaces cat/shell redirect
       │
       ▼
   M0         ── macro assembler: M1 assembly (mnemonics, named regs, macros) → hex2
       │
       ▼
   cc_amd64   ── minimal C compiler: C subset → M1
       │
       ▼
   M2-Planet  ── full self-hosting C compiler (vendored; last vendored rung)
       │
       ▼
   helixc-bootstrap seed  ── WE wrote it, in the M2 C-subset (Apache-2.0)   ✅ DONE
       │
       ▼
   helixc     ── = the existing, frozen kovc (self-hosted in Helix, K2==K3)
```

## Ladder status (verified, byte-exact + reproducible + WSL-tested)

| # | Rung | Role | Bytes | `.bin` SHA-256 (prefix) | Source | Status |
|---|------|------|-------|--------------------------|--------|--------|
| 1 | `hex0` | hex → bytes | 299 | `cc1d1741…` | **hand-authored** (frozen root) | ✅ done |
| 2 | `hex1` | + labels | 622 | `c264a212…` | vendored, built by hex0 | ✅ done |
| 3 | `hex2` | + long labels / linker | 1519 | `6c69c7e6…` | vendored, built by hex1 | ✅ done |
| 4 | `catm` | concatenation | 299 | `911d19bf…` | vendored, built by hex2 | ✅ done |
| 5 | `M0` | macro assembler | 1684 | `db97dff1…` | vendored, built by catm+hex2 | ✅ done |
| 6 | `cc_amd64` | minimal C compiler | 17976 | `ea0054d1…` | vendored, built by M0 | ✅ done |
| 7 | `M2-Planet` | full C compiler | 200561 | `724b9e2d…` | vendored, built by cc_amd64 | ✅ done¹ |
| 8 | `helixc-bootstrap` seed | bridge to Helix | 62467 | `9837db12…` | **original, Apache-2.0** (M2 C-subset) | ✅ done² |
| 9 | `helixc` (kovc) | the Helix compiler | 587092 | seed-minted K1′ | frozen Helix source, minted by the seed | ✅ done³ |

¹ Core capability tested (compiles C → runs → correct exit). The self-host
fixpoint (M2 rebuilds M2 byte-stably) is investigated but not yet holding — see
`M2-Planet/PROVENANCE.md`; left open and honest, not faked.
² The vendored ladder ends at rung 7. Rung 8 is the first original work: the
`helixc-bootstrap` seed (`seed.c`, Apache-2.0, M2 C-subset) that replaces Python
as the K1 minter. User chose Option A (2026-05-30, `../docs/K_HELIX_TOP_SCOPING.md`);
the seed is written, built by the ladder (62467 B, sha `9837db12`), and passes
17/17 tests. There is no separate `helix-libc`: kovc emits self-contained ELFs,
so the seed is the only original artifact.
³ The seed mints a helixc (K1′, 587092 B) from the frozen Helix sources, proven
byte-identical to the Python-minted compiler by a diverse double-compile — see
`../docs/K_DDC_RESULT.md`. Bootstrap Python-deletion-ready; full deletion (K4)
is user-gated.

Vendor pins: `stage0-posix-amd64` @ `15535f88`, `M2-Planet` @ `761c2af5`,
`M2libc` @ `b8bb2a01`. Each rung carries its own `PROVENANCE.md` with full source
SHAs; byte-exactness is enforced by `.gitattributes` (`* -text`) and verified
with `git cat-file blob :<path> | sha256sum` against upstream.

## Authorship policy

- **stage0/hex0/** — **fully hand-authored from raw bytes**. This is the literal "raw binary as starting point" hard constraint. Each byte is reasoned-about and annotated. Audit cross-checked against `oriansj/stage0-posix-amd64/hex0_AMD64.hex0` for byte-level encoding correctness, but the bytes shipped here are ours. **Frozen** (any change is a user-flag event).
- **stage0/hex1/** through **stage0/M2-Planet/** — vendored, per user decision 2026-05-30 (Option 1: build everything directly from binary for full trust; vendor audited sources, build each from our prior rung). Sources pulled from `oriansj/stage0-posix-amd64` + `oriansj/M2-Planet` + `oriansj/M2libc` at pinned commits; **no pre-built binary is ever trusted** — every rung is rebuilt by the rung below it, reproducibly, and byte-audited against its annotated source. **All seven rungs built and verified** (see status table).
- **stage0/helixc-bootstrap/** — the first **original** work: `seed.c`, written by us in the M2-Planet C-subset (Apache-2.0, avoids GPL-3.0 contagion). Built by the ladder, it mints a helixc byte-identical to the Python-minted one (diverse double-compile — see `../docs/K_DDC_RESULT.md`). No separate `helix-libc` is needed: kovc emits self-contained ELFs, so the seed is the only original artifact. **DONE + DDC-verified.**

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
