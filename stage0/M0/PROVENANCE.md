# stage0/M0 — provenance

`M0` is the fifth rung of the Kovostov-Native bootstrap ladder, and the first
"real" tool: a **macro assembler** that turns M1-style assembly (mnemonics,
named registers, macros, labels) into hex2. It is the architecture-specific
seed of M1 / the C-compiler chain.

Per project policy (`stage0/README.md`; user decision 2026-05-30), the middle
rungs are **vendored from canonical, community-audited sources at pinned
commits** and **built by the prior rungs** — no pre-built binary is ever
trusted.

## Vendored sources (oriansj/stage0-posix-amd64 @ `15535f88e25825f01a0de275b6d45f77e618bd6b`, GPL-3.0-or-later)

| File | Role | SHA-256 |
|---|---|---|
| `M0_AMD64.hex2` | the M0 program (hex2 format) | `a9692351b88c00eb492da448ecbc237057300cf49734f766161a94bb2340b034` |
| `ELF-amd64.hex2` | ELF header, prepended to the program by catm | `bfad808d3b41b7eac274fbe44e0e02fffe67c764b55b0c9b2a35f2a49f2ae418` |
| `test/cc_amd64.M1` | the cc_amd64 C-compiler seed, an M1-assembly test input | `599c0c6af8da48aaad834c06138c384b9485ab80f7c178c87c0c77d29d65cedc` |

Kept statically separable from our Apache-2.0 `helix-libc` / `helixc-bootstrap`.

## How `M0.bin` is built (the trust chain)

The canonical mescc-tools phase-3 recipe — catm prepends the ELF header to the
M0 program, then hex2 assembles it:

```
../catm/catm.bin M0.hex2 ELF-amd64.hex2 M0_AMD64.hex2
../hex2/hex2.bin M0.hex2 M0.bin
```

No assembler, no pre-built binary. Run `bash build.sh` **under WSL**. It
rebuilds, ELF-checks, disassembles, verifies `M0.sha256` reproducibility, and
runs the tests.

## What M0 does (verified)

M0 assembles M1 assembly into hex2. Verified end-to-end: `M0.bin` assembles the
real `cc_amd64.M1` C-compiler seed into ~61 KB of hex2, which (with the ELF
header, assembled by hex2) yields a valid ~18 KB `cc_amd64` ELF. So M0 emits
correct, runnable machine code.

## Next rung

`cc_amd64` — M0 assembles `cc_amd64.M1` -> `cc_amd64.hex2`, catm + hex2 build the
binary. Then M2-Planet (the full C compiler), which additionally needs the
`oriansj/M2libc` sources (separate repo, to be vendored + pinned at that rung).
