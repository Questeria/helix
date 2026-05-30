# stage0/hex1 — provenance

`hex1` is the second rung of the Kovostov-Native bootstrap ladder. Per the
project policy (`stage0/README.md`; user decision 2026-05-30), the middle rungs
(hex1 … M2-Planet) are **vendored from canonical, community-audited sources at
pinned commits** and **built by our own `hex0`** — no pre-built binary is ever
trusted. (The trust root, `stage0/hex0/`, is fully hand-authored.)

## Vendored source

| | |
|---|---|
| File | `hex1_AMD64.hex0` |
| Upstream | https://github.com/oriansj/stage0-posix-amd64 |
| Pinned commit | `15535f88e25825f01a0de275b6d45f77e618bd6b` |
| Upstream authors | Jeremiah Orians, Andrius Štikonas |
| License | **GPL-3.0-or-later** (kept statically separable from our Apache-2.0 `helix-libc` / `helixc-bootstrap`) |
| Source SHA-256 | `1f53a60ca14a408f1f3a1715f3103c2b819508f6362d23fca4a772d39aa323c2` |

`test/02-hex2-source.hex1` is the upstream `hex2_AMD64.hex1` (same commit), used
only as a label-exercising test fixture (not shipped as a rung here).

## How `hex1.bin` is built (the trust chain)

`hex1_AMD64.hex0` is hex1's program written in **hex0 format** (hex pairs +
`#`/`;` comments). Our hand-authored, verified `../hex0/hex0.bin` decodes it:

```
../hex0/hex0.bin < hex1_AMD64.hex0 > hex1.bin
```

No assembler, no pre-built binary. Run `bash build.sh` **under WSL** (these are
Linux ELFs — Git Bash cannot execute them). It rebuilds, ELF-checks,
disassembles, verifies `hex1.sha256` reproducibility, and runs the tests.

## What hex1 adds over hex0

Single-character labels (`:label` definitions + references) and richer comments
— enough to make `hex2`'s source writable. Verified end-to-end: `hex1.bin`
(built by our `hex0`) assembles the real upstream `hex2` source into a valid
x86-64 ELF, and remains a strict superset of `hex0` for plain hex.

## I/O note

Our `hex0` is **stdin → stdout**; from `hex1` upward the vendored stage0-posix
tools use **argv files** (`hex1 INPUT OUTPUT`). This is fine: `hex0` is only the
*builder* of `hex1.bin` (it decodes the source); the resulting `hex1` program
adopts the stage0-posix argv convention used by every rung above it.
