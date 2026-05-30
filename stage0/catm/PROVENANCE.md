# stage0/catm — provenance

`catm` is the fourth rung of the Kovostov-Native bootstrap ladder. It is a tiny
file-concatenation tool (`catm OUTPUT in1 in2 … inN` writes the bytes of every
input, in order, to OUTPUT) — it removes the need for `cat` or shell redirection
in the rungs above (used to prepend the ELF header to each program from M0 up).

Per project policy (`stage0/README.md`; user decision 2026-05-30), the middle
rungs are **vendored from canonical, community-audited sources at pinned
commits** and **built by the prior rung** — no pre-built binary is ever trusted.

## Vendored source (oriansj/stage0-posix-amd64 @ `15535f88e25825f01a0de275b6d45f77e618bd6b`, GPL-3.0-or-later)

| File | SHA-256 |
|---|---|
| `catm_AMD64.hex2` | `55f659754208a897074a188f93dc4202a9e58ebdacdcbc05fd8bc391467cdc80` |

Kept statically separable from our Apache-2.0 `helix-libc` / `helixc-bootstrap`.

## How `catm.bin` is built (the trust chain)

`catm_AMD64.hex2` is catm's program written in **hex2 format**. Our `hex2.bin`
(built by hex1, built by our hand-authored hex0) assembles it:

```
../hex2/hex2.bin catm_AMD64.hex2 catm.bin
```

No assembler, no pre-built binary. Run `bash build.sh` **under WSL**. It
rebuilds, ELF-checks, disassembles, verifies `catm.sha256` reproducibility, and
runs the tests (concatenation order + binary-safety incl. NUL bytes).

## Role in the ladder

From M0 onward the recipe is `catm prog.hex2 ELF-amd64.hex2 body.hex2` then
`hex2 prog.hex2 prog`. So catm is the glue for every subsequent rung. (The hex2
rung's own M0 test used plain `cat`; from here we use this byte-exact catm.)
