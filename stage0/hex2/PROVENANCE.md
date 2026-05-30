# stage0/hex2 — provenance

`hex2` is the third rung of the Kovostov-Native bootstrap ladder. Per project
policy (`stage0/README.md`; user decision 2026-05-30), the middle rungs are
**vendored from canonical, community-audited sources at pinned commits** and
**built by the prior rung** — no pre-built binary is ever trusted.

## Vendored sources (all from oriansj/stage0-posix-amd64 @ `15535f88e25825f01a0de275b6d45f77e618bd6b`, GPL-3.0-or-later)

| File | Role | SHA-256 |
|---|---|---|
| `hex2_AMD64.hex1` | hex2 program, written in hex1 format (the rung source) | `d4c011ac6f56bf82fed4a42a10547c5cbf5801a024b66e987f145925484d86e8` |
| `test/ELF-amd64.hex2` | ELF header, for the M0-assembly test fixture | `bfad808d3b41b7eac274fbe44e0e02fffe67c764b55b0c9b2a35f2a49f2ae418` |
| `test/M0_AMD64.hex2` | M0 assembler program body, for the M0-assembly test fixture | `a9692351b88c00eb492da448ecbc237057300cf49734f766161a94bb2340b034` |

License kept statically separable from our Apache-2.0 `helix-libc` /
`helixc-bootstrap`.

## How `hex2.bin` is built (the trust chain)

`hex2_AMD64.hex1` is hex2's program written in **hex1 format**. Our `hex1.bin`
(built by our hand-authored `hex0`) assembles it:

```
../hex1/hex1.bin hex2_AMD64.hex1 hex2.bin
```

No assembler, no pre-built binary. Run `bash build.sh` **under WSL**. It
rebuilds, ELF-checks, disassembles, verifies `hex2.sha256` reproducibility, and
runs the tests.

## What hex2 adds over hex1

Long (multi-character) labels and absolute + relative addressing — enough to
write the M0 macro assembler. Verified end-to-end (the canonical mescc-tools
phase-3 recipe): `hex2.bin` assembles `cat(ELF-amd64.hex2, M0_AMD64.hex2)` into
a valid 1684-byte M0 ELF, and remains a strict superset of hex0/hex1 for plain
hex.

## Build recipe reference (mescc-tools-mini-kaem.kaem, upstream)

```
hex0 hex1_AMD64.hex0 hex1            # rung 2 (we use stdin hex0)
hex1 hex2_AMD64.hex1 hex2           # rung 3 (THIS rung)
hex2 catm_AMD64.hex2 catm           # rung 4 (next: catm)
catm M0.hex2 ELF-amd64.hex2 M0_AMD64.hex2 ; hex2 M0.hex2 M0   # rung 5 (M0)
```
