# Helix — Build and Run Quickstart

This is the fastest path from a fresh checkout to the **shipped** Helix toolchain: the
from‑raw‑binary, Python‑free `kovc` compiler (`hex0 → seed → kovc`).

> **Source of truth.** For the verified status + every honest residual, read **[`README.md` §Status](README.md#status-v13-2026-06-05)**
> and **[`docs/CLEAN_REPRODUCTION.md`](docs/CLEAN_REPRODUCTION.md)** (rebuild the core chain from a
> clean checkout). This page is an entry‑point; those docs are the authority for the precise claims,
> and nothing here is meant to contradict them.

> **Investor demo (GPT‑2 on Helix):** to see the real (unchanged) GPT‑2 — a 2019 base model, trust not speed — run on this from‑raw stack, see [`docs/HELIX_GPT2_DEMO_RUNBOOK.md`](docs/HELIX_GPT2_DEMO_RUNBOOK.md). Live chat: `bash scripts/serve_chat_demo.sh` then open <http://127.0.0.1:8848/?source=sse>. One‑command attestation: `bash scripts/gpt2_demo_attest.sh`.

## What ships

The Helix toolchain is built **entirely from a raw‑binary root — there is no trusted pre‑built
compiler**. `hex0` (299 hand‑authored hex bytes) → … → `seed` (an Apache‑2.0 C‑subset compiler) →
`kovc` (the Helix compiler, `helixc/bootstrap/{lexer,parser,kovc}.hx`, self‑hosted in Helix), each
rung built **only by the prior rung**. The toolchain is **Python‑free**: the repo holds **exactly
one** committed `.py` (`verification/oracle/oracle_train.py`), a fenced numpy verification *oracle*
that is **never** part of the compile/run path. `gcc` is used only as an independent *auditor* (the
diverse‑double‑compile of the seed), never to produce a shipped artifact.

The trust chain is **complete to PTX**: a real GPT‑2 forward runs through `kovc`‑emitted GPU (PTX)
kernels, gated green token‑for‑token vs an independent reference. Below PTX it relies on NVIDIA's
closed `ptxas` + driver — the one trusted‑once boundary, stated openly in `README.md` §Status. The
CPU path is all‑the‑way‑down from raw binary.

## Prerequisites

- **WSL2 + Linux** on Windows, or any Linux (for the from‑raw build + running the produced ELFs)
- **gcc** (the diverse‑double‑compile auditor) + a CUDA toolchain & RTX‑class GPU (for the GPU
  capstone / GPT‑2 demo). The CPU trust core needs **no GPU**.

No Python is required for the shipped toolchain. (Python 3.10+ is needed only for the fenced numpy
audit oracle and the historical frontend in the appendix below.)

## Reproduce the trust core (one command, no GPU)

The byte‑identical from‑raw trust core is reproducible by **one committed command on a clean
checkout** — CPU‑only, ~1 minute:

```bash
bash scripts/reproduce_trust.sh
```

It deletes every pre‑built rung binary, rebuilds the whole `hex0 → seed` ladder (each rung
self‑verifying its `.sha256`), runs the self‑host fixpoint (`seed → K1 → K2 → K3 → K4` with
**K2 == K3 == K4 byte‑for‑byte**) and the gcc diverse‑double‑compile, and asserts the pinned anchors,
exiting nonzero on any mismatch. `.github/workflows/trust-reproduce.yml` runs it on a clean
`ubuntu-latest` runner on every push/PR, so the core is reproducible push‑button by any third party.

## Compile a Helix program with the shipped `kovc`

The universal gate is the source of truth for what `kovc` accepts and proves:

```bash
bash scripts/gate_kovc.sh
```

It runs the self‑host fixpoint + a 109‑program feature corpus (integer widths; floats incl.
bf16/f16 arithmetic; control flow; generics; traits + default methods; closures incl.
capturing‑by‑value; pattern matching; wide struct fields; structured `path:line:col` diagnostics) +
a `ptxas`‑free PTX byte‑diff + diagnostics, and prints `GATE_PASS` only when all legs are green.

To build `kovc` from the raw seed and compile your own `.hx`, follow `docs/CLEAN_REPRODUCTION.md`
(Step 2 builds the ladder; the seed‑minted `kovc` then compiles Helix programs — including its own
source — directly to a Linux x86‑64 ELF, no assembler/linker/libc).

## What works today

- Hand‑authored 299‑byte ELF (`stage0/hex0/hex0.bin`) — the raw‑binary foundation.
- **Self‑hosting Helix‑native compiler** (`helixc/bootstrap/{lexer,parser,kovc}.hx`, `kovc`): a
  complete lexer + parser + x86‑64‑ELF code generator written *in Helix*, built from the raw‑binary
  `seed` (no Python) into a native binary that compiles Helix programs — including its own source.
  Proven **byte‑identical self‑host fixpoint** (K2 == K3 == K4), gated by the 109‑program corpus.
- **GPU path complete to PTX**: `kovc`‑emitted GPU kernels run a real GPT‑2 forward, gated
  token‑for‑token vs an independent numpy reference (`docs/HELIX_GPT2_DEMO_RUNBOOK.md`).
- **Source‑level forward + reverse‑mode autodiff** as language built‑ins (`grad`, `grad_rev`,
  `grad_rev_all`), with chain rules across user‑defined calls (via inlining) and stdlib
  transcendentals (analytic rules).
- **Verifier‑gated reflection runtime** (mutable cells; `quote`/`splice_f`/`modify_f` call your
  verifier before committing) and **IR‑level effect verification** (`@pure` transitively prohibited
  from effectful code).
- 8 unique compile‑time AGI type‑system features (Presburger shapes, `D<T>`, memory tiers, agents,
  etc.) and a stdlib in `helixc/stdlib/*.hx` (see `helix_website/HELIX_REFERENCE.md` for live
  per‑module counts).
- 6 dogfood programs running real ML in Helix‑emitted binaries (gradient descent, linear regression,
  affine fit, ReLU/XOR net, logistic regression w/ sigmoid+BCE+multi‑output AD, and a
  self‑improving‑agent flagship).

> The older "Stage NN" / "K‑bootstrap chunk counter" / Python‑parity‑matrix framing is **superseded**
> (see `README.md` §Status). For live state, read `git log --oneline -8`,
> `docs/TRUST_CHAIN_CLOSED.md`, `docs/CLEAN_REPRODUCTION.md`, and `scripts/gate_kovc.sh`.

## Project layout

```
helix/
├── stage0/                # The from-raw ladder: hex0 (299-byte ELF) → … → seed (the C-subset compiler)
├── helixc/
│   ├── bootstrap/         # kovc: the self-hosted Helix compiler (lexer.hx, parser.hx, kovc.hx)
│   ├── stdlib/            # Helix stdlib (*.hx)
│   ├── runtime/           # Category-B host harnesses (GPU/CPU launchers, tokenizer, importer, serve)
│   └── (historical frontend; see appendix)
├── docs/
│   ├── CLEAN_REPRODUCTION.md      # rebuild the core chain from a clean checkout
│   ├── TRUST_CHAIN_CLOSED.md      # verified state + every residual
│   ├── HELIX_GPT2_DEMO_RUNBOOK.md # the GPT-2-on-Helix investor demo
│   └── lang/{spec,tutorial,agi-features}.md
└── scripts/
    ├── reproduce_trust.sh         # one-command from-raw trust core (CPU-only)
    └── gate_kovc.sh               # universal gate (self-host fixpoint + corpus + PTX)
```

## What makes Helix different

Helix is being built to combine:
1. **Compile‑time tensor shape checking** via Presburger arithmetic — catches matmul dimension bugs before code runs.
2. **Effect/capability typing** — `@pure` cannot accidentally call `@io`.
3. **Differentiable types `D<T>`** — gradient flow tracked at the type level.
4. **Memory‑tier types** — `WorkingMem` / `EpisodicMem` / `SemanticMem` / `ProceduralMem` distinguished, transitions explicit.
5. **Reflection primitives** — `quote { ... }`, `splice`, `modify` (verifier‑gated).
6. **Agent declarations** — society‑of‑mind cognitive architecture in the type system.
7. **Symbolic autodiff** — derivatives computed at compile time, not at runtime.

Helix is now being optimized **for AI to USE and EXTEND, not for human developers**. Where
ergonomics conflicts with structural regularity, structural regularity wins. See
`docs/lang/agi-features.md` for the deep dive.

## License

**Source-available, verify-only** — the Helix Source-Available License; see [`LICENSE`](LICENSE). You may audit and reproduce Helix to verify it; other use requires a separate license. Vendored `stage0/` tools keep their own (GPLv3, etc.) licenses. (Earlier docs mention an intended Apache-2.0 / open-source release; those are superseded by `LICENSE`.)

---

## Appendix — the historical Python frontend (NOT the shipped path)

> **This section is historical and is NOT the shipped compile/run path.** The early Python‑hosted
> `helixc` frontend (`python -m helixc.backend.x86_64`, `python -m helixc.frontend.*`) was the
> bootstrap‑era prototype. It is retained for reference only and is **not** part of the from‑raw,
> Python‑free `kovc` toolchain that ships (the repo's sole committed `.py` is the fenced numpy audit
> oracle). Do not use these commands to evaluate the project — use the from‑raw `kovc` path above.

The historical frontend exposed a Python CLI for compiling, type‑checking, and emitting symbolic
derivatives from `.hx` source (`python -m helixc.backend.x86_64 in.hx out.bin`,
`python -m helixc.frontend.typecheck`, `python -m helixc.frontend.autodiff_cli`). Those drivers, the
historical pytest suite, and the `helixc/frontend|ir|backend` Python packages predate the self‑hosted
`kovc` and are no longer the path the project's claims rest on. The shipped equivalents are: build
`kovc` from raw (`docs/CLEAN_REPRODUCTION.md`) and gate it with `scripts/gate_kovc.sh`; autodiff is a
language built‑in in `kovc` (`grad` / `grad_rev` / `grad_rev_all`).
