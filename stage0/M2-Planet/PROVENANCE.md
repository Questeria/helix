# stage0/M2-Planet — provenance

`M2` (M2-Planet) is the seventh rung of the Kovostov-Native bootstrap ladder, and
the **full, self-hosting C compiler**. cc_amd64 (rung 6) builds the first M2;
from M2 upward the toolchain compiles real C with structs, function pointers,
and the M2libc standard library. It is the compiler in whose C subset we will
write our own Apache-2.0 `helix-libc` + `helixc-bootstrap`.

Per project policy (`stage0/README.md`; user decision 2026-05-30), the middle
rungs are **vendored from canonical, community-audited sources at pinned
commits** and **built only by the prior rungs** — no pre-built binary is ever
trusted. The GPL-3.0 vendored trees are kept statically separable from our
Apache-2.0 helix top.

## Vendored sources

### oriansj/M2-Planet @ `761c2af5eee5bc2c27945b0ec896be26b8f5939b` (GPL-3.0-or-later)

| File | SHA-256 |
|---|---|
| `M2-Planet/cc.h` | `4ea75a3ae7e9559bc8f80c07892d14f31525dba99e075587548a38685319a7c4` |
| `M2-Planet/cc_globals.c` | `9ca4b663b780802979bb59acaacc35e9f0238526737dde2a3c8ca94ef753ebf5` |
| `M2-Planet/cc_reader.c` | `c54db85d4bbf5f66afe2f1c6f8ba2b969a9d4584356c9d2970227ba1f6e33d27` |
| `M2-Planet/cc_strings.c` | `723809f52ab919ac540e574cdd196c02aba2ce662753417e4b47958416656a4c` |
| `M2-Planet/cc_types.c` | `710a5b53ad66c3f25cccdd00ac645aa44136b11b72dcef7aabfc51be0dc8520c` |
| `M2-Planet/cc_emit.c` | `fee9eef38574837b7e57a0aeef635a3b0dbb49cef4c693292cf27d90bcd1176a` |
| `M2-Planet/cc_core.c` | `8158d8b58e277d7cca75d68fae2b1348a36e438c6f17e3afb447df36100816a5` |
| `M2-Planet/cc_macro.c` | `8f1b2360f82cca6cb2fe85f2af62b9e40316c6bd79eec5d0d97483cc921e819c` |
| `M2-Planet/cc.c` | `86abb1982886e1eb00df12f31c14f11fa8c747d3fceba435e71e7008470cc6ef` |

### oriansj/M2libc @ `b8bb2a0159a7376716a396ec6f6bc29dd27857b5` (GPL-3.0-or-later / public-domain-equivalent per repo)

Build-time (compiled into M2 itself):

| File | SHA-256 |
|---|---|
| `M2libc/amd64/linux/bootstrap.c` | `ac3405bba7eccae673ea8427836d6a04a87ac414821b1c0416953e68420e7aca` |
| `M2libc/bootstrap.c` | `c2c69c02b525e0a6f0eb471f9c8a812fc22bd348782012642894feb8aca80c17` |
| `M2libc/bootstrappable.c` | `efe16699e165d1ebad2d8f942383aea11c23dcc783f291c3dbf7b6d199fdc831` |

Test/assemble-time (M2's *output* pairs with these — its own calling convention):

| File | SHA-256 |
|---|---|
| `M2libc/amd64/amd64_defs.M1` | `6357f709e6ef5e08cdde62121c90b8e66ea2dbdd87be259e41dc8d04c549b330` |
| `M2libc/amd64/libc-core.M1` | `4490a493143408b98bcca6dac399db915775b695551e480f41d7a40b5c9b7b64` |
| `M2libc/amd64/ELF-amd64.hex2` | `6adcbe26d1ebc345008f55d542371ddf71b961cfc0990950a7bb133771b5f6a6` |

## How `M2.bin` is built (the trust chain)

The canonical mescc-tools-mini-kaem Phase-5 recipe — catm concatenates the
bootstrap libc + all M2-Planet sources (exact order in `build.sh`), cc_amd64
compiles to M1, catm prepends cc_amd64's paired defs, M0 assembles to hex2, catm
prepends the ELF header, hex2 links:

```
catm     M2-0.c   <12 sources, recipe order>
cc_amd64 M2-0.c   M2-0.M1
catm     M2-0-0.M1 ../cc_amd64/amd64_defs.M1 ../cc_amd64/libc-core.M1 M2-0.M1
M0       M2-0-0.M1 M2-0.hex2
catm     M2-0-0.hex2 ../cc_amd64/ELF-amd64.hex2 M2-0.hex2
hex2     M2-0-0.hex2 M2.bin
```

No assembler, no pre-built binary. Run `bash build.sh` **under WSL**. It
rebuilds, ELF-checks, disassembles, verifies `M2.sha256` reproducibility
(`724b9e2d…`, 200561 bytes), and runs the tests.

## What M2 does (verified)

M2 compiles C to M1 assembly. Verified end-to-end: M2 compiles `int main(){
return 42; }` and an arithmetic program (`6*7`) in bootstrap-mode; the emitted M1
— assembled with **M2libc's** amd64 defs (M2's output uses a different calling
convention than cc_amd64, so cc_amd64's defs would segfault) — produces runnable
binaries that exit with the correct codes. So M2 emits correct machine code, end
to end from our hand-authored hex0 root.

## Known-future hardening (NOT yet done — not faked)

A full **self-host fixpoint** (M2 recompiles its own sources via `-f` flags like
the recipe's later phases, producing a second-gen M2 proven byte-stable) is the
strongest possible test for a self-hosting compiler and is planned as a follow-up
hardening step. A first quick attempt (feeding M2 the single concatenated
`M2-0.c`) did not succeed — M2 wants its sources as separate `-f` units — so it
is deliberately left out of `run_tests.sh` rather than asserted falsely. The rung
ships green on its real, demonstrated capability (compiles C → runs → correct
exit).

## Next rung

After M2-Planet, the ladder leaves vendored territory: **we write `helix-libc` +
`helixc-bootstrap`** in the M2 C subset (original Apache-2.0 work, statically
separable from these GPL trees). The user is checkpointed at that transition
before it starts. Then `helixc` (= the existing, frozen kovc).
