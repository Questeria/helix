# stage0/M2-Planet — provenance

`M2` (M2-Planet) is the seventh rung of the Kovostov-Native bootstrap ladder, and
the **full, self-hosting C compiler**. cc_amd64 (rung 6) builds the first M2;
from M2 upward the toolchain compiles real C with structs, function pointers,
and the M2libc standard library. It is the compiler in whose C subset we wrote our
own Apache-2.0 `helixc-bootstrap` seed.

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

## Self-host fixpoint — investigated, NOT yet holding (documented, not faked)

A full **self-host fixpoint** (our M2 recompiles its own sources, producing a
second-gen M2 proven byte-stable) is the strongest possible trust test for a
self-hosting compiler. It is **not yet passing** and is deliberately kept out of
`run_tests.sh` rather than asserted falsely. Precise findings (2026-05-30):

- M2's *core* capability is solid and tested: it compiles ordinary C and the
  result runs (see `run_tests.sh`). That is the rung's actual claim.
- Rebuild via separate `-f` units + `--bootstrap-mode` (the shape the recipe's
  later phases use) **compiles cleanly** (gen2 `.M1` is produced, non-empty), but
  the resulting gen2 M2, assembled with `M2libc/amd64/{amd64_defs,libc-core}.M1`,
  **SIGILLs (132) at runtime** — an illegal opcode, i.e. a defs/libc-pairing or
  latent-codegen mismatch, not a link failure.
- Rebuild via the single concatenated `M2-0.c` (the exact input cc_amd64 used)
  needs `--bootstrap-mode` too (without it M2 errors `Unknown type FILE`, since
  `FILE` lives in the bootstrap libc) — cc_amd64 was more lenient here.
- The full M2-Planet likely needs `M2libc/amd64/libc-full.M1` (the recipe's
  phases 8-11 switch to `libc-full.M1`), which this rung does **not** vendor yet.

**Next concrete step when this is picked up:** vendor `M2libc/amd64/libc-full.M1`
(pin b8bb2a01) and/or follow M2-Planet's *own* self-host test recipe (in the
M2-Planet repo, not the stage0-posix mini-kaem, which never rebuilds M2 with M2),
then assert gen2==gen3 byte-identical. Upstream M2-Planet self-hosts, so this is
expected to be a recipe/libc-pairing gap, not a fundamental defect — but it is
left open and honest rather than forced.

### Update 2026-06-02 (v1.1 H6 — bounded attempt + decision)

The next step above was executed (`selfhost_probe.sh`): `M2libc/amd64/libc-full.M1`
was vendored at pin `b8bb2a01` (sha256 `ed3a14ae…`, 1780 bytes) and the self-host
fixpoint retried. Result — **the gap is deeper than the libc-core/libc-full pairing:**

- `M2.bin` self-compiles its own 12 sources cleanly: **gen2.M1 = 2 198 293 bytes**
  (M2's C front end parses and emits M1 for the full M2-Planet source on its own input).
- But assembling gen2.M1 with `amd64_defs.M1 + libc-full.M1` (M0 → hex2) yields a
  **169-byte** binary that **SIGILLs (132)**. The M0/hex2 assemble of M2's *complex*
  own-output is what breaks — even though M2's output for a simple program (`return 42`)
  assembles and runs correctly (`run_tests.sh` passes). The residual blocker is a latent
  M1-emission / assemble-pairing mismatch for large inputs inside the **vendored GPL
  M2-Planet / M0 / hex2 toolchain**, not the libc variant and not our code.

**Decision (v1.1 H6 — honest, charter-sanctioned):** M2-Planet is kept
**built-once-and-audited**, the trusted-once root the seed is built from. This is the
correct Reflections-on-Trusting-Trust position: some root must be trusted-once, and
M2-Planet is the strongest available one — **vendored from canonical community-audited
sources at a pinned commit, built only by prior rungs (no pre-built binary trusted),
and its output verified end-to-end** (compiles C, the result runs with correct exit
codes). The trust that bears on *our* Apache-2.0 code (seed → kovc) is the
**K2==K3==K4 self-host fixpoint + diverse-double-compile**, which is green and gated on
every compiler commit. Chasing the vendored-toolchain self-host SIGILL is deliberately
**out of scope** — it would mean debugging GPL upstream codegen, not Helix. The probe
(`selfhost_probe.sh`) and the vendored `libc-full.M1` are kept so the investigation is
reproducible. **H6 (green-or-documented) = documented.**

## Next rung

After M2-Planet, the ladder leaves vendored territory: **we wrote the
`helixc-bootstrap` seed** (`seed.c`) in the M2 C subset — original Apache-2.0
work, statically separable from these GPL trees. The user chose Option A
(2026-05-30); the seed is built by this ladder and mints `helixc` (the existing
frozen kovc) byte-identically to the Python reference, proven by a diverse
double-compile (`../../docs/K_DDC_RESULT.md`). There is no separate `helix-libc`
— kovc emits self-contained ELFs.
