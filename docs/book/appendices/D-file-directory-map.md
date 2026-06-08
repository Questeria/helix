# Appendix D — File & directory map

*What this appendix covers: a navigational map of the Kovostov-Native repository for humans and
AI agents — the top-level layout, what lives in each area (`helixc/{bootstrap, stdlib, examples,
runtime}`, the from-raw ladder under `stage0/`, the `scripts/` gate and reproduction drivers, the
`docs/` tree including this book, the fenced numpy oracle under `verification/`, and the CI under
`.github/`), and the load-bearing files you should know by name. It is built from the actual tree
at tag `v1.3-release`; every path below was listed from the working copy before this page was
written.*

This is a **reference map**, not a build walkthrough — for the build see
[Build from raw](../part2-setup-build/02-build-from-raw.md), and for push-button verification see
[Reproduce & verify the trust chain](../part2-setup-build/04-reproduce-verify-trust.md). For a
categorized index of the example *programs* specifically, see
[Appendix E — Example index](E-example-index.md); this appendix maps the whole repository, not
just the examples.

> **For AI agents:** this page is a map, not the territory. The authoritative file list is always
> the tree itself — `git ls-files` and a directory listing. Before you cite or open a path, deref
> it; if this map and the repository disagree, the repository wins and this appendix is the bug —
> flag it, do not silently follow stale prose. The repo also contains many untracked scratch
> artifacts at the top level (build outputs like `_*.bin`, `*.csv`); treat **committed** files
> (`git ls-files`) as the source of truth, not whatever happens to sit in a working copy.

---

## D.1 The repository at a glance

The trust-bearing tree is small and regular. The committed parts that matter for building,
running, and verifying Helix are:

```text
Kovostov-Native/
├── README.md                     project overview + honest status (v1.3)
├── QUICKSTART.md                 build-and-run quickstart
├── LICENSE                       Apache-2.0 (source license)
├── helixc/                       the Helix compiler sources, stdlib, examples, runtime
│   ├── bootstrap/                kovc itself, written in Helix (lexer/parser/kovc/evaluator)
│   ├── stdlib/                   the 21 committed .hx stdlib modules
│   ├── examples/                 the 98 committed example .hx programs (+ .ref.ptx)
│   └── runtime/                  the C host launcher + transformer trainer (capstone)
├── stage0/                       the from-raw ladder: hex0 … M2-Planet → seed
│   ├── hex0/ hex1/ hex2/         the raw-binary rungs (hex0 is hand-authored)
│   ├── catm/ M0/ cc_amd64/       concatenator, macro assembler, minimal C compiler
│   ├── M2-Planet/                the full C compiler (last vendored rung) + M2libc
│   └── helixc-bootstrap/         the seed (seed.c) + the self-host fixpoint + the gcc-DDC
├── scripts/                      the gate, the trust reproduction, the GPU corpora
│   ├── gate_kovc.sh              the universal gate (prints GATE_PASS)
│   ├── reproduce_trust.sh        one-command from-raw reproduction (prints REPRODUCE_TRUST: PASS)
│   └── capstone_audit.sh         the GPU capstone audit (CUDA host only)
├── verification/                 the fenced numpy oracle (the only committed .py) + witnesses
│   └── oracle/oracle_train.py    the independent numpy oracle for the capstone
├── docs/                         all documentation, incl. the trust records and this book
│   ├── TRUST_CHAIN_CLOSED.md     the verified state + every residual (the honesty ceiling)
│   ├── CLEAN_REPRODUCTION.md     rebuild the core chain from a clean checkout
│   ├── lang/spec.md              the language reference
│   └── book/                     *Helix: The Complete Guide* (this book)
└── .github/workflows/            CI that reproduces the trust core on a clean runner
    └── trust-reproduce.yml       runs scripts/reproduce_trust.sh on ubuntu-latest
```

The single committed Python file is the load-bearing fact behind the "Python-free toolchain"
claim: `git ls-files "*.py"` returns **exactly one** path,
[`verification/oracle/oracle_train.py`](../../../verification/oracle/oracle_train.py), which is a
verification *oracle* and never part of the compile/run path. The static fence in
[`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) step `[1]` asserts that count
(see [Reproduce & verify the trust chain §2](../part2-setup-build/04-reproduce-verify-trust.md)).

> **For AI agents:** the fence is a hard invariant, not a guideline. `git ls-files "*.py" | wc -l`
> must be `1` and `git ls-files "*.c" "*.h" | wc -l` must be `24`. If you add a `.py` or a C/H file
> anywhere in the tree, you break the fence and the reproduction script fails closed. Treat the
> single `.py` (`verification/oracle/oracle_train.py`) as off-limits to the toolchain.

---

## D.2 `helixc/` — the compiler, library, examples, runtime

`helixc/` holds everything written *in Helix* (plus the small C host harness for the GPU
capstone). It has four subdirectories.

### `helixc/bootstrap/` — kovc itself, written in Helix

This is **the compiler** — `kovc`, the from-scratch Helix compiler written in Helix that emits
x86-64 ELF directly. It is the most load-bearing source in the repository: the self-host fixpoint
is *this* code compiling *this* code. Four files:

| File | Role |
|---|---|
| [`helixc/bootstrap/lexer.hx`](../../../helixc/bootstrap/lexer.hx) | the lexer |
| [`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx) | the parser |
| [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) | typecheck → IR → lowering passes → x86-64 ELF back end (+ the PTX emitter) |
| [`helixc/bootstrap/evaluator.hx`](../../../helixc/bootstrap/evaluator.hx) | a Helix evaluator/interpreter (the zero-lineage behavioral cross-check substrate) |

These three — `lexer.hx`, `parser.hx`, `kovc.hx` — are the canonical source the gate regenerates
its fixpoint inputs from (see D.4). The trust chain's whole capability claim rests on them: the
`seed` compiles them into `K1`, and `K1 → K2 → K3 → K4` lands on a byte-identical fixed point.

### `helixc/stdlib/` — the standard library (21 modules)

The committed standard library is **21 `.hx` modules** in
[`helixc/stdlib/`](../../../helixc/stdlib/). They are compiled and exercised by the gate's feature
corpus. Grouped by theme:

- **Numerics / ML:** [`transcendentals.hx`](../../../helixc/stdlib/transcendentals.hx),
  [`ieee754.hx`](../../../helixc/stdlib/ieee754.hx),
  [`nn.hx`](../../../helixc/stdlib/nn.hx),
  [`tensor.hx`](../../../helixc/stdlib/tensor.hx),
  [`autodiff.hx`](../../../helixc/stdlib/autodiff.hx),
  [`autodiff_reverse.hx`](../../../helixc/stdlib/autodiff_reverse.hx),
  [`mnist.hx`](../../../helixc/stdlib/mnist.hx),
  [`checkpoint.hx`](../../../helixc/stdlib/checkpoint.hx).
- **Collections / data:** [`vec.hx`](../../../helixc/stdlib/vec.hx),
  [`hashmap.hx`](../../../helixc/stdlib/hashmap.hx),
  [`string.hx`](../../../helixc/stdlib/string.hx),
  [`iterators.hx`](../../../helixc/stdlib/iterators.hx),
  [`option.hx`](../../../helixc/stdlib/option.hx),
  [`result.hx`](../../../helixc/stdlib/result.hx),
  [`csv.hx`](../../../helixc/stdlib/csv.hx).
- **AGI-oriented primitives:** [`agi_search.hx`](../../../helixc/stdlib/agi_search.hx),
  [`agi_match.hx`](../../../helixc/stdlib/agi_match.hx),
  [`agi_memory.hx`](../../../helixc/stdlib/agi_memory.hx),
  [`agi_world.hx`](../../../helixc/stdlib/agi_world.hx).
- **Provenance / safety:** [`provenance.hx`](../../../helixc/stdlib/provenance.hx),
  [`safety.hx`](../../../helixc/stdlib/safety.hx).

> **Note:** "21 modules" is the **committed file count** in `helixc/stdlib/`. The README quotes a
> larger historical figure ("16 modules, ~455 functions") and points at
> `helix_website/HELIX_REFERENCE.md` for live per-module function counts; for the authoritative
> module list, list the directory. (There is a *separate*, two-file top-level `stdlib/` —
> `collections.hx`, `string.hx` — which is **not** the compiler's standard library; the modules the
> gate and examples use are the ones under `helixc/stdlib/`.)

### `helixc/examples/` — the example corpus (98 programs)

[`helixc/examples/`](../../../helixc/examples/) holds **98 committed `.hx` programs** — the
dogfood ML programs, the flagship self-improving agent, small feature probes, and the kernels used
by the GPU PTX regression. A subset of these are compiled **and run** by the gate with a checked
exit code; the rest are readable demonstration sources. The canonical first program lives here:

**Verified example** — [`helixc/examples/exit42.hx`](../../../helixc/examples/exit42.hx)
(compiled and run by the gate's feature corpus; the produced ELF exits with status `42`):

```helix
// First end-to-end Helix program: compiles, links, runs, exits with status 42.
//
// Demonstrates: function decl, return, integer literal.
// Verification: $? == 42 after running the produced ELF.

fn main() -> i32 {
    42
}
```

The gate asserts this with the literal corpus line `chk "$EX/exit42.hx" 42` in
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) (where `$EX` is the `helixc/examples`
directory): it compiles the program with the freshly self-hosted `K2`, runs the ELF, and checks
the exit code is `42`. This directory also holds the three committed reference PTX files the gate's
GPU text regression compares against —
[`helixc/examples/vector_add_kernel.ref.ptx`](../../../helixc/examples/vector_add_kernel.ref.ptx),
[`helixc/examples/tiled_matmul_kernel.ref.ptx`](../../../helixc/examples/tiled_matmul_kernel.ref.ptx),
and [`helixc/examples/tf32_matmul_kernel.ref.ptx`](../../../helixc/examples/tf32_matmul_kernel.ref.ptx)
— plus the matching kernel `.hx` sources. For the categorized per-program index, see
[Appendix E](E-example-index.md).

### `helixc/runtime/` — the C host launcher and capstone trainer

The **only** C in `helixc/` lives here, and only the GPU capstone uses it — it is the host-side
launcher and training loop that drives `kovc`-emitted PTX kernels on a CUDA device:

- [`helixc/runtime/cuda_launch.c`](../../../helixc/runtime/cuda_launch.c) — the C host launcher.
- [`helixc/runtime/train_transformer.c`](../../../helixc/runtime/train_transformer.c) — the
  end-to-end transformer training harness (the capstone).
- `init_weights.bin`, `loss_curve.csv`, `oracle_curve.csv` — capstone inputs/outputs (shared
  initial weights and recorded loss curves).

> **Residual:** this C host launcher is part of the GPU **trusted-once** boundary. The CPU path is
> all-the-way-down from raw binary; the GPU path is hand-auditable from `hex0` to PTX, but below PTX
> it trusts NVIDIA's closed `ptxas`, the CUDA driver, the GPU hardware, **and this C launcher**. See
> [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md).

---

## D.3 `stage0/` — the from-raw ladder

[`stage0/`](../../../stage0/) is the root of trust: the from-raw ladder
`hex0 → hex1 → hex2 → catm → M0 → cc_amd64 → M2-Planet → seed`, each rung built **only** by the
rung below it, with no trusted pre-built compiler. The build, the per-rung commands, and the pinned
rung hashes are the subject of [Build from raw](../part2-setup-build/02-build-from-raw.md); here is
the directory layout.

Two files at the top of `stage0/` are worth knowing:
[`stage0/README.md`](../../../stage0/README.md) (the per-rung status table, authorship policy, and
vendor pins) and [`stage0/MESCC_TOOLS_PROVENANCE.md`](../../../stage0/MESCC_TOOLS_PROVENANCE.md)
(provenance for the vendored mescc-tools lineage).

Each rung is its own directory. Per `stage0/README.md`, **every** rung ships the same four
artifacts: the **source** in auditable text form, the **binary** (`.bin`), a **`.sha256`** hash
file, and a **`build.sh`** that builds it using only the previous stage's tools (plus
`run_tests.sh` and an audit `disasm.txt`). For example,
[`stage0/hex0/`](../../../stage0/hex0/) contains `hex0.hex` (the hand-authored hex source),
`hex0.bin`, [`hex0.sha256`](../../../stage0/hex0/hex0.sha256),
[`build.sh`](../../../stage0/hex0/build.sh), `run_tests.sh`, `disasm.txt`, the byte-annotation
`hex0.bytes.md`, and a `test/` directory.

The rungs:

| Dir | Rung | Built by | Pinned `.bin` SHA-256 (prefix) |
|---|---|---|---|
| [`stage0/hex0/`](../../../stage0/hex0/) | `hex0` (299 hand-authored bytes; **frozen**) | `xxd -r -p` of `hex0.hex` | `cc1d1741…` |
| [`stage0/hex1/`](../../../stage0/hex1/) | `hex1` (+ single-char labels) | hex0 | `c264a212…` |
| [`stage0/hex2/`](../../../stage0/hex2/) | `hex2` (+ long labels / linker) | hex1 | `6c69c7e6…` |
| [`stage0/catm/`](../../../stage0/catm/) | `catm` (concatenator) | hex2 | `911d19bf…` |
| [`stage0/M0/`](../../../stage0/M0/) | `M0` (macro assembler) | catm + hex2 | `db97dff1…` |
| [`stage0/cc_amd64/`](../../../stage0/cc_amd64/) | `cc_amd64` (minimal C compiler) | M0 + catm + hex2 | `ea0054d1…` |
| [`stage0/M2-Planet/`](../../../stage0/M2-Planet/) | `M2-Planet` (full C compiler; last vendored rung) | cc_amd64 + catm + M0 + hex2 | `724b9e2d…` |
| [`stage0/helixc-bootstrap/`](../../../stage0/helixc-bootstrap/) | `seed` (the first original artifact) | M2-Planet + catm + M0 + hex2 | `9837db12…` |

A few directory notes that surprise readers:

- `stage0/M2-Planet/` carries both the vendored compiler (`M2-Planet/`) and its libc
  (`M2libc/`), plus its own [`PROVENANCE.md`](../../../stage0/M2-Planet/PROVENANCE.md). Its
  `.bin` is the produced `M2.bin`; its honest residual (M2's *own* self-host fixpoint is
  investigated, not yet holding) is recorded there and does not affect the ladder — see
  [Build from raw, Rung 7](../part2-setup-build/02-build-from-raw.md).
- `stage0/` also contains the vendored helper directories `blood-elf/`, `hex2-linker/`, and `M1/`
  from the mescc-tools lineage; the rungs the ladder actually walks are the eight in the table.

### `stage0/helixc-bootstrap/` — the seed, the fixpoint, and the gcc-DDC

This rung is where the vendored ladder ends and Helix's own code begins, and it is the busiest
directory in `stage0/`. It holds the `seed`, the self-host-fixpoint machinery, and the
trusting-trust defense. The load-bearing files:

| Path | Role |
|---|---|
| [`stage0/helixc-bootstrap/seed.c`](../../../stage0/helixc-bootstrap/seed.c) | **the seed source** — the Apache-2.0 C-subset compiler (≈1,368 lines) that mints `kovc`. The first original artifact in the chain. |
| [`stage0/helixc-bootstrap/build.sh`](../../../stage0/helixc-bootstrap/build.sh) | builds `seed.bin` from `seed.c` using M2-Planet + catm + M0 + hex2. |
| [`stage0/helixc-bootstrap/seed.sha256`](../../../stage0/helixc-bootstrap/seed.sha256) | the pinned seed hash (`9837db12…`). |
| [`stage0/helixc-bootstrap/assemble_k1.hx`](../../../stage0/helixc-bootstrap/assemble_k1.hx) | concatenates the frozen compiler sources into the fixpoint inputs. **Hardcodes the absolute `/mnt/c/Projects/Kovostov-Native/...` build path** — the one disclosed path-rewrite caveat (handled by `reproduce_trust.sh` step `[0]`). |
| [`stage0/helixc-bootstrap/assemble_k1.sh`](../../../stage0/helixc-bootstrap/assemble_k1.sh) | the shell wrapper that drives the source regeneration. |
| [`stage0/helixc-bootstrap/ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh) | **the gcc diverse-double-compile** — proves `gcc` (zero M2-Planet ancestry) and the from-raw seed produce a byte-identical `K1`. Prints `DDC_ANCHOR_OK`. |
| [`stage0/helixc-bootstrap/drivers/`](../../../stage0/helixc-bootstrap/drivers/) | the entry-point drivers (`driver_k1src.hx`, `driver_k1input.hx`, `driver_k1ptxdrv.hx`). |
| `k1src.hx`, `k1input.hx`, `k1ptxdrv.hx` | the **regenerated** fixpoint sources (assembled from `kovc.hx`/`lexer.hx`/`parser.hx`); the gate re-derives them, so treat them as build artifacts, not hand-edited source. |
| [`stage0/helixc-bootstrap/corpus_gen/`](../../../stage0/helixc-bootstrap/corpus_gen/) | committed feature-corpus programs the gate reads directly as `$GENC` (generics/traits/closures/wide-field/`i64` probes, …). |
| [`stage0/helixc-bootstrap/corpus/`](../../../stage0/helixc-bootstrap/corpus/) | a committed set of small feature `.hx` fixtures (arithmetic, comparisons, bitops, `match`, …). |
| [`stage0/helixc-bootstrap/test/`](../../../stage0/helixc-bootstrap/test/), `run_tests.sh`, `test_runner.hx` | the seed's own 17/17 behavioral tests. |
| [`stage0/helixc-bootstrap/README.md`](../../../stage0/helixc-bootstrap/README.md) | the rung's documentation. |

> **Note:** the produced `seed.bin` is **gitignored and not tracked** — the committed tree carries
> `seed.c` (and `seed.sha256`) but not the binary, so the seed *must* be re-derived from raw. A
> [`.gitignore`](../../../stage0/helixc-bootstrap/.gitignore) in this directory enforces that, and
> [`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md) confirms `git ls-files
> stage0/helixc-bootstrap/seed.bin` is empty.

> **For AI agents:** if you build the fixpoint *outside* `/mnt/c/Projects/Kovostov-Native`, either
> run the whole thing through [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh)
> (its step `[0]` rewrites the hardcoded path) or apply the same `sed` swap first. Running
> `assemble_k1.sh` from a non-canonical directory **silently reads from and writes to the canonical
> dir** (rc=0, but no files in your checkout) — a false-positive "success." Do not treat a
> non-canonical fixpoint run as verified.

---

## D.4 `scripts/` — the gate and the reproduction drivers

[`scripts/`](../../../scripts/) holds the executable harnesses. Most files are scratch probes
(prefixed `_`, e.g. `_g3_*`, `_m4_*`) — useful history, not load-bearing. The ones that matter:

| Script | Role | Success token |
|---|---|---|
| [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) | **the gate.** Self-host fixpoint + 109-program feature corpus + GPU PTX **text** regression + negative diagnostics. Every compiler change must pass it. | `GATE_PASS` (exit `0`) |
| [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) | **the one-command reproduction.** Static fence → from-raw ladder (deletes rung binaries first) → the gate → the gcc-DDC. CPU-only. | `REPRODUCE_TRUST: PASS` |
| [`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh) | **the GPU capstone audit** — runs the transformer-on-PTX capability proof. Needs a CUDA host; not run in CI. | (CUDA-host verdict) |
| [`scripts/feature_corpus.sh`](../../../scripts/feature_corpus.sh) | a standalone feature compile+run corpus harness (the gate inlines its own corpus; see below). | — |

The pinned anchors the reproduction enforces are declared verbatim near the top of
[`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh): `SEED_SHA=9837db12…`,
`K1_SHA=84363adb…`, `FIX_SHA=0992dddd…`. The full stage-by-stage account of what each check proves
is [Reproduce & verify the trust chain](../part2-setup-build/04-reproduce-verify-trust.md).

There is also a family of GPU corpus drivers — `gpu_corpus.sh`, `gpu_elementwise_corpus.sh`,
`gpu_reduction_corpus.sh`, `gpu_redux_bwd_corpus.sh`, `gpu_transpose_corpus.sh`,
`gpu_attention_corpus.sh`, `gpu_tf32_corpus.sh`, `gpu_perf_corpus.sh`, and
`gpu_attention_corpus.sh` — used to exercise the PTX back end on a CUDA host.

> **For AI agents:** key off exit status and literal tokens, never English. The gate prints
> `GATE_PASS` and exits `0` on success (its four sub-anchors are the verbatim lines
> `FIXPOINT OK (K2==K3==K4 byte-identical AND == pinned known-good)`,
> `CORPUS: 109 passed, 0 failed`, `CHECK_ERR: 4 passed, 0 failed`, and a `GPU PTX REGRESSION OK`
> line). `reproduce_trust.sh` prints `REPRODUCE_TRUST: PASS`. Both **modify the working tree**
> (delete rung binaries, rewrite the hardcoded path) — run them on a throwaway clone, never on a
> tree you want pristine.

**Where the 109-program corpus actually lives.** The gate's feature corpus is assembled from three
places, which is worth knowing if you go looking for "the corpus directory":

1. `$EX` — committed programs in [`helixc/examples/`](../../../helixc/examples/) (e.g.
   `exit42.hx`, `gradient_descent.hx`, `dogfood_18_pat_struct_showcase.hx`).
2. `$CD` (`/tmp/corpus`) — small probe programs **generated inline by the gate at run time** via
   `gen <name> <<'EOF' … EOF` heredocs inside
   [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh). These are committed *in the gate script
   body*, not as separate files; do not expect a `/tmp/corpus` directory in the tree.
3. `$GENC` — committed programs read directly from
   [`stage0/helixc-bootstrap/corpus_gen/`](../../../stage0/helixc-bootstrap/corpus_gen/).

So there is no single `corpus/` folder that contains all 109 entries; the count is the sum of these
sources, checked one program at a time by the `chk` helper.

---

## D.5 `verification/` — the fenced numpy oracle

[`verification/`](../../../verification/) is deliberately *outside* the toolchain. It holds the
independent oracle and the behavioral cross-check witnesses — the evidence that backs the capability
and DDC claims, none of which ever runs as part of building `kovc`.

- [`verification/oracle/oracle_train.py`](../../../verification/oracle/oracle_train.py) — **the
  one committed `.py` in the repository.** It is the independent numpy oracle for the GPU capstone:
  it trains the same transformer from the shared initial weights and never reads Helix's training
  trajectory, so a converged loss within ~0% of it is genuine corroboration, not circularity. Its
  [`README.md`](../../../verification/oracle/README.md) states the independence contract.
- [`verification/py_witness/`](../../../verification/py_witness/) — the behavioral cross-check
  witness for the broader v1.1 language surface (a zero-lineage interpreter battery and its result
  logs, e.g. `T1_FINDINGS.md`, `crosscheck.sh`, `run_witness.sh`). This is the *behavioral*
  cross-check, not a byte-identical second compiler.

> **Residual:** the byte-identical, hash-pinned DDC covers the **`seed → K1`** surface. The broader
> v1.1 language surface is cross-checked **behaviorally** here, and that witness is out-of-tree (not
> clean-checkout reproducible). External third-party reproduction on independent hardware remains the
> one open increment — see [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md).

> **For AI agents:** despite its directory name, `py_witness/` is **not** part of the compile/run
> path and does not violate the Python-free toolchain fence — its shell scripts and Markdown are
> committed, but the fence counts `*.py` files, and the only committed `.py` is the oracle. Do not
> treat anything under `verification/` as a build dependency.

---

## D.6 `docs/` — documentation, trust records, and this book

[`docs/`](../../../docs/) is large: alongside the canonical records it carries a deep history of
plans and per-stage audit logs (`audit-stage*.md`, `stage*-progress*.md`, the `K_*` / `HELIX_*`
planning docs). For navigation, the files that are *load-bearing today*:

**The canonical trust records (cite these for any trust/perf/GPU claim):**

- [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) — the verified state **and
  every residual**, stated plainly. This is the honesty ceiling for the book: no chapter may claim
  past its §R residuals.
- [`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md) — how to rebuild the core
  chain from a clean checkout, including the "Where it walls" path-rewrite caveat.
- [`docs/CURRENT_HEAD_AUDIT_PACKET.md`](../../../docs/CURRENT_HEAD_AUDIT_PACKET.md) — the committed
  proof extract for the `v1.3-release` tag (the three pinned anchors with their verbatim verdict
  lines).

**The language and feature references:**

- [`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) and
  [`docs/lang/spec.md`](../../../docs/lang/spec.md) — the language reference.
- [`docs/lang/tutorial.md`](../../../docs/lang/tutorial.md) — a beginner guide.
- [`docs/lang/agi-features.md`](../../../docs/lang/agi-features.md) — the AGI-oriented features
  deep dive; [`docs/lang/trap-ids.md`](../../../docs/lang/trap-ids.md) and
  [`docs/lang/hbs.md`](../../../docs/lang/hbs.md) — trap IDs and the HBS sample notes.
- [`docs/HELIX_PURPOSE.md`](../../../docs/HELIX_PURPOSE.md),
  [`docs/ROADMAP.md`](../../../docs/ROADMAP.md), and
  [`docs/research/WAVE1_FINDINGS.md`](../../../docs/research/WAVE1_FINDINGS.md) — purpose,
  roadmap, and synthesized research direction.

**This book — [`docs/book/`](../../../docs/book/):**

```text
docs/book/
├── README.md           the book's front matter
├── STYLE_GUIDE.md      binding author rules (audience, terminology, the honesty rule)
├── SUMMARY.md          the table of contents (mdBook-style)
├── part1-orientation/  Part I (shipped)
├── part2-setup-build/  Part II (shipped)
├── part3-language/     Part III (shipped)
├── part4-stdlib/       Part IV — the standard library (shipped)
├── part5-compiler/     Part V — the compiler kovc (shipped)
├── part9-for-ai-agents/ Part IX — the AI-operator manual (shipped)
└── appendices/         this appendix (D) lives here, alongside Appendix E
```

The full chapter list — every Part (I–IX) and every appendix (A–H), all shipped — is in
[`docs/book/SUMMARY.md`](../../../docs/book/SUMMARY.md). Authors must read
[`docs/book/STYLE_GUIDE.md`](../../../docs/book/STYLE_GUIDE.md) before writing.

> **For AI agents:** `docs/` mixes live records with a large archive of dated stage logs. When you
> need the *current* truth, read in order: `git log --oneline -8`, then
> `docs/TRUST_CHAIN_CLOSED.md`, then `docs/CLEAN_REPRODUCTION.md`, then `scripts/gate_kovc.sh`.
> Do not cite a dated `stage*`/`audit-stage*` file as the current state — it is historical.

---

## D.7 `.github/` — continuous reproduction

[`.github/`](../../../.github/) holds exactly one workflow:
[`.github/workflows/trust-reproduce.yml`](../../../.github/workflows/trust-reproduce.yml). Its job
is short because all the work lives in the committed script — it checks out a clean clone, installs
the audit tools (`xxd`, `binutils`, `gcc`, `file`, `coreutils`), and runs
`bash scripts/reproduce_trust.sh` on a clean `ubuntu-latest` runner, on every push/PR to `main`,
on manual `workflow_dispatch`, and on a weekly schedule. It is CPU-only by design (the hosted
runner has no GPU), so the transformer capstone is verified separately on a CUDA host via
[`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh). The job is green **only** if the
full ladder rebuild, the self-host fixpoint, and the gcc-DDC all reproduce the pinned hashes
byte-for-byte. This is the "independent operator" reproduction path described in
[Reproduce & verify the trust chain §5](../part2-setup-build/04-reproduce-verify-trust.md).

---

## D.8 Quick lookup — the load-bearing files

If you remember nothing else from this map, remember these paths:

| Want to… | Open |
|---|---|
| read the compiler source | [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx), `lexer.hx`, `parser.hx` |
| read the from-raw root of trust | [`stage0/hex0/hex0.hex`](../../../stage0/hex0/) (299 hand-authored bytes) |
| read the seed source | [`stage0/helixc-bootstrap/seed.c`](../../../stage0/helixc-bootstrap/seed.c) |
| run the gate | [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) → `GATE_PASS` |
| reproduce the trust chain | [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) → `REPRODUCE_TRUST: PASS` |
| run the trusting-trust defense | [`stage0/helixc-bootstrap/ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh) → `DDC_ANCHOR_OK` |
| find the pinned anchors | top of [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) (`SEED_SHA`/`K1_SHA`/`FIX_SHA`) |
| find the PTX reference fixtures | [`helixc/examples/*.ref.ptx`](../../../helixc/examples/) |
| read the honest residuals | [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) |
| see what CI runs | [`.github/workflows/trust-reproduce.yml`](../../../.github/workflows/trust-reproduce.yml) |

---

**Next:** [Appendix E — Example index](E-example-index.md) — a categorized, per-program index of
the 98 example `.hx` files under `helixc/examples/`, with the gate-asserted exit codes called out.
