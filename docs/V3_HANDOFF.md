# Helix v3.0 — Handoff

**Last updated:** 2026-05-21 · **Repo:** `C:/Projects/Kovostov-Native` ·
**Branch:** `main` (clean; all work pushed to `origin/main` @ `f352fd4`)

This is the orientation document for whoever continues the Helix v3.0
compiler rewrite. Read it first, then `docs/V3_PLAN.md` (the full plan
plus a per-chunk changelog — the source of truth) and
`docs/V3_STAGE210_MLIR_DECISION.md` (the ratified MLIR-migration
decision).

---

## 1. What Helix is

Helix is a new programming language and its compiler (`helixc/`,
Python-hosted). The pipeline: frontend (lexer / parser / typechecker)
→ Tensor IR (`helixc/ir/tir.py`) → Tile IR (`helixc/ir/tile_ir.py`) →
backends (x86-64, PTX, …). v3.0 is "the big rewrite" — adding an
industrial MLIR + LLVM backend path alongside the home-grown one.

## 2. Where the project stands (2026-05-21)

- **v2.0–v2.5: released** (the home-grown GPU compiler + autodiff +
  register allocator).
- **v3.0: in progress** — 19 numbered stages across three phases:
  - **Phase D (Stages 200–208): COMPLETE.**
  - **Phase E (Stages 210–216): in progress — Stages 210, 211, 212
    CLOSED.**
  - **Phase F (Stages 220–222): not started.**
- **`V3_STAGES_DONE = 12` of 19** (`scripts/helix_status.py`) —
  ~63 % of v3.0 stages, ~95 % overall toward v3.0.
- The working tree is clean; everything is committed and pushed.

## 3. Phase E — the MLIR migration (the current frontier)

Ratified strategy (`docs/V3_STAGE210_MLIR_DECISION.md`):

- **Hybrid dialects** — upstream MLIR dialects (`func` / `arith` /
  `math` / `linalg` / `vector` / `memref` / `gpu`) for the ~80–85 %
  numerical / structural op core; a small custom `helix` dialect for
  the ~15–20 % Helix-specific ops.
- **Mock path** — the dev machine has no MLIR toolchain. Code must
  NEVER `import mlir` at module top level; all MLIR imports are lazy /
  probed. A toolchain-free structural validator (`validate.py`)
  returns a tri-state `PASSED` / `FAILED` / `DEFERRED`.
- **Migration discipline** — additive (the home-grown tile-IR →
  backends path stays as the reversible fallback), parity-gated
  (Stage 215), fail-closed (any construct the translator cannot
  faithfully emit raises `MLIRTranslationError` — never wrong output).

**Phase E stages:**

- **Stage 210 — CLOSED** — the MLIR dependency + dialect-strategy
  decision.
- **Stage 211 — CLOSED** — the MLIR substrate:
  `helixc/ir/mlir/{toolchain,mapping,helix_dialect,validate}.py`.
- **Stage 212 — CLOSED** — the tile-IR → MLIR translator,
  `helixc/ir/mlir/emit.py`.

### The translator (`emit.py`) — current capability

`emit_mlir_module(tile_ir.TileModule) -> str` walks a Tile-IR module
and emits MLIR textual IR. It faithfully renders every IR type and
**17 of the 29 `tile_ir.TileOpKind`s**: the scalar `arith` core
(`const_int` / `const_float`, `add` / `sub` / `mul`, `neg`),
compare / select, the elementwise + layout-transform `vector` tile ops
(`add` / `sub` / `mul` / `zeros` / `reshape` / `transpose`),
`func.call`, and the GPU thread-index read. Per-op emitters return
`list[str]` (one MLIR line per element).

The other **12 op kinds fail closed by deliberate, documented design**
(see the `emit.py` module docstring for the full rationale):

- async `TMA_LOAD` / `TMA_STORE` / `BARRIER_WAIT` — RESIDUAL; the
  nvgpu-vs-`helix` async-abstraction decision is a Stage-213 concern;
- the `memref` movement ops (`TILE_LOAD/STORE_GLOBAL`,
  `TILE_LOAD/STORE_SHARED`) and `TILE_CONST` — stub-status, no defined
  operand / result signature;
- `TILE_MATMUL` (→ `vector.contract`) and `TILE_REDUCE`
  (→ `vector.multi_reduction`) — attribute-heavy (affine indexing
  maps, iterator types, reduction kinds); need MLIR-encoding design;
- `TILE_INDEX_LOAD/STORE_HBM` — need a `memref` type bridge plus a
  kernel-parameter-name → SSA-value resolution.

## 4. What's next (in order)

1. **(Optional) finish Stage 212's deferred ops** — the 12 above.
   Stage 212 is "closed enough" per the plan, but a future run can add
   these emitters. The attribute-heavy ones (matmul / reduce / memref
   / index-hbm) need design work first.
2. **Stage 213 — MLIR → backends.**
3. **Stage 214 — the progressive-lowering pass pipeline.**
4. **Stage 215 — the MLIR-vs-tile-IR parity gate** (verify the new
   path matches the home-grown path).
5. **Stage 216 — the end-of-Phase-E 5-clean-gate.**
6. **Phase F (Stages 220–222)** — backend unification, the Stage-221
   cutover, the v3.0.0 5-clean-gate + git tag.
7. **206-R residual ops** — additive LLVM-lowering chunks (print_int,
   write_file / read_file_to_arena, TRACE, arena, QUOTE-family) needed
   before the Stage-221 cutover.

When `v3.0.0` is tagged, v3.0 is done.

## 5. The working discipline (follow this)

- **One coherent chunk per work unit.** A stage is built in small
  numbered chunks.
- **Per-chunk audit on three axes** — before committing a code chunk,
  review it (and fix every HIGH-severity or must-fix issue, then
  re-review until clean) for:
  1. *Silent failures* — swallowed errors, silent fallbacks, code that
     emits plausible-but-wrong output instead of failing loudly.
  2. *Type design* — can illegal states be represented? weak typing?
     unenforced invariants?
  3. *General correctness* — bugs, logic errors, wrong output,
     convention drift.
- **Fail-closed always** — the translator never emits guessed / wrong
  MLIR; an unsupported construct raises `MLIRTranslationError`.
- **Mock-path-first** — build with pure text; never `import mlir` at
  module top level; shape-check with `validate.mock_validate_mlir`.
- **Each phase closes with a 5-clean-gate** — an audit across the
  frontend / IR / backend / runtime / tests.
- **Commit each chunk; push to `origin/main` after each commit.**
- **When a v3.0 stage closes, bump `V3_STAGES_DONE` in
  `scripts/helix_status.py`** so the progress numbers stay accurate.

## 6. How to build & test

- **Fast MLIR-path verification** (use this for `helixc/ir/mlir/`
  work): `python -m pytest helixc/tests/ -k mlir -q` — ~156 tests,
  ~30 s, bounded.
- **The full suite is SLOW (not broken).**
  `python -m pytest helixc/tests/` is a large integration suite
  (~4,500 tests; `test_codegen.py` alone is ~1,000 real
  compile-and-run tests that assemble + link + execute binaries) — it
  runs for a long time. `pytest-xdist` is installed:
  `python -m pytest helixc/tests/ -n auto` parallelizes it. NEVER pipe
  pytest through `| tail` — it buffers all output until exit, so a
  slow run looks exactly like a hang.
- The stdlib parse is cached process-wide, so repeated
  `parse(include_stdlib=True)` calls are fast.

## 7. Environment & MLIR facts

- Dev machine: Windows, Python 3.13, bare pip + venv. **No MLIR
  toolchain, no conda.** WSL is available. (This is why the MLIR work
  uses the mock path; real `mlir-opt` validation is a binding-gated
  future concern.)
- MLIR facts the translator relies on: `arith` ops are
  elementwise-polymorphic over scalars and vectors (same mnemonics);
  MLIR integers are signless (signedness is per-op); MLIR float
  literals require a decimal point; float `!=` is `une`
  (unordered-not-equal); `gpu.thread_id` yields `index`, not `i32`.

## 8. Hard constraints (always)

- **Never read `C:/Projects/Neptune/api.env`.**
- **Never force-push to `main`; never skip git hooks** (`--no-verify`
  etc.). If a hook fails, fix the underlying issue.
- The compiler and its build must not depend on external AI APIs.

## 9. Key files

| Path | What |
|------|------|
| `docs/V3_PLAN.md` | The full v3.0 plan + per-chunk changelog — **source of truth.** |
| `docs/V3_STAGE210_MLIR_DECISION.md` | The ratified MLIR dialect-strategy decision. |
| `docs/V3_HANDOFF.md` | This document. |
| `scripts/helix_status.py` | Progress reporter; `V3_STAGES_DONE` lives here. |
| `helixc/ir/tir.py` | Tensor IR (`OpKind`). |
| `helixc/ir/tile_ir.py` | Tile IR (`TileOpKind`, 29 members). |
| `helixc/ir/mlir/` | Phase-E MLIR substrate: `toolchain.py`, `mapping.py`, `helix_dialect.py`, `validate.py`, `emit.py`. |
| `helixc/tests/test_mlir_*.py` | The MLIR-path tests. |
