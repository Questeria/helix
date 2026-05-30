# stage0/cc_amd64 — provenance

`cc_amd64` is the sixth rung of the Kovostov-Native bootstrap ladder, and the
first **C compiler**: it compiles a small C subset to M1 assembly. It is the
tool that bootstraps M2-Planet (the full self-hosting C compiler), which in turn
will build our `helix-libc` + `helixc-bootstrap`.

Per project policy (`stage0/README.md`; user decision 2026-05-30), the middle
rungs are **vendored from canonical, community-audited sources at pinned
commits** and **built by the prior rungs** — no pre-built binary is ever
trusted.

## Vendored sources (oriansj/stage0-posix-amd64 @ `15535f88e25825f01a0de275b6d45f77e618bd6b`, GPL-3.0-or-later)

| File | Role | SHA-256 |
|---|---|---|
| `cc_amd64.M1` | the cc_amd64 compiler program (M1 assembly) | `599c0c6af8da48aaad834c06138c384b9485ab80f7c178c87c0c77d29d65cedc` |
| `amd64_defs.M1` | amd64 macro/register definitions (prepended to compiled output) | `d25c1151039ef3f31d67e8fde591ddd4dd945b91ef818e24136d30d2321f0f15` |
| `libc-core.M1` | minimal libc + `_start` (prepended to compiled output) | `d0f9b01d7eb88575be5930c31dd1e9257af6486c35a23a961e5c3bd91edcc155` |
| `ELF-amd64.hex2` | ELF header, prepended to the program by catm | `bfad808d3b41b7eac274fbe44e0e02fffe67c764b55b0c9b2a35f2a49f2ae418` |

`cc_amd64.M1` is byte-identical to the seed vendored at `stage0/M0/test/cc_amd64.M1`
(same upstream, same commit). Kept statically separable from our Apache-2.0
`helix-libc` / `helixc-bootstrap`.

## How `cc_amd64.bin` is built (the trust chain)

The canonical mescc-tools phase-4 recipe — M0 assembles the compiler program to
hex2, catm prepends the ELF header, hex2 assembles the binary:

```
../M0/M0.bin     cc_amd64.M1      cc.hex2
../catm/catm.bin cc_full.hex2 ELF-amd64.hex2 cc.hex2
../hex2/hex2.bin cc_full.hex2 cc_amd64.bin
```

No assembler, no pre-built binary. Run `bash build.sh` **under WSL**. It
rebuilds, ELF-checks, disassembles, verifies `cc_amd64.sha256` reproducibility,
and runs the tests.

## What cc_amd64 does (verified)

cc_amd64 compiles C (a subset) to M1 assembly. Verified end-to-end: it compiles
`int main() { return 42; }` and an arithmetic program to M1, which — concatenated
with `amd64_defs.M1` + `libc-core.M1`, assembled by M0, given the ELF header, and
assembled by hex2 — produces runnable binaries that exit with the correct codes
(42, and 6*7=42). So cc_amd64 emits correct machine code, end to end from our
hand-authored hex0 root.

## Next rung

`M2-Planet` — the full, self-hosting C compiler. It needs two additional
upstream repos vendored + pinned at that rung: `oriansj/M2-Planet` (the compiler
sources) and `oriansj/M2libc` (its standard library). cc_amd64 + M0 build the
first M2-Planet; M2-Planet then rebuilds itself. After M2-Planet we WRITE our own
Apache-2.0 `helix-libc` + `helixc-bootstrap` in the M2 C subset (original work —
the user is checkpointed at that transition).
