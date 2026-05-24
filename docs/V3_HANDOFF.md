# Helix v3.0 — Handoff

**Last updated:** 2026-05-21 · **Repo:** `C:/Projects/Kovostov-Native` ·
**Branch:** `main` (verify live state with `git status --short --branch`
and `git log -1 --oneline`)

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
  - **Phase E (Stages 210–216): in progress — Stages 210, 211, 212, 213
    CLOSED; Stage 214 not yet started.**
  - **Phase F (Stages 220–222): not started.**
- **`V3_STAGES_DONE = 13` of 19** (`scripts/helix_status.py`) —
  ~68 % of v3.0 stages, ~95 % overall toward v3.0.

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
- **Stage 213 — CLOSED** — chunks A-C shipped the mock-path-first
  backend-target scaffold, real `mlir-opt` validation dispatch, and
  the fail-closed backend pass-pipeline runner contract. The 2026-05-24
  audit batches closed all HIGH and MEDIUM findings from the audit
  packet (control predicates, memref access, arith.constant value/type,
  scf.for bounds, generic-function-body terminator, vector
  transfer_read / shape_cast / multi_reduction, llvm.func symbol
  binding, LLVM aggregate/vector typed-value, C-like impossible
  declarations). The holistic close audit closed two more HIGH
  findings (empty `input_symbols` bypass; generic-form `"func.func"`
  strict-static skip). 31/31 strict canaries; 411 MLIR slice.
  Documented structural-tightening items (PASSED brand bypass via
  `object.__new__`, `dict[str, str|None]` SSA-type conflation,
  `MLIRBackendResult` three-state shape) are deferred — they are
  design-tightening opportunities, not silent-failure shapes, and
  the existing in-file fail-closed discipline + post_init invariants
  handle the accidental-misuse axis. See
  `docs/HELIX_MLIR_AUDIT_PACKET.md` "2026-05-24 Checkpoint D" for
  the close-audit details.

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

### The Stage 213 backend scaffold (`backends.py`)

`helixc/ir/mlir/backends.py` is the first Stage 213 seam. It defines
the five targets the MLIR path must eventually feed (`llvm_ir`, `ptx`,
`rocm_hip`, `metal_msl`, `webgpu_wgsl`), maps the existing GPU backend
enum to the four GPU targets, records each target's required MLIR
dialects, and returns a frozen tri-state `MLIRBackendResult` from
`lower_mlir_to_backend(...)`.

Important: it is a scaffold, not a real lowering yet. Every target's
pass pipeline and output validator are explicitly unwired. Malformed
MLIR fails before any support probe; mock-valid MLIR returns
`DEFERRED` with explicit findings unless a real verifier, a declared
pipeline, `mlir-opt`, and a target output validator are all present.
If a future branch declares a pipeline before wiring the target output
validator, `lower_mlir_to_backend` still returns `DEFERRED` rather than
claiming a backend pass from transformed MLIR alone. The private runner
requires passed real validation, argv-list `mlir-opt` dispatch, a
non-empty readable artifact, and a clean target output validator before
`PASSED` is representable. The result type rejects silent illegal
states (mutable findings, non-bool pass flags, whitespace tool names,
blank/non-string output text, and promoting deferred validation into a
pass).

Current Stage 213 verification: 38 `test_mlir_backends.py` tests; the
fast MLIR slice is 205 passing tests on this machine.

`helixc/ir/mlir/validate.py` now also has the Stage-213 real validator
seam, `validate_mlir_with_toolchain(...)`. It runs
`mock_validate_mlir` first, fails immediately on malformed MLIR before
tool probing, invokes `mlir-opt` when available, and returns DEFERRED
with support details when `mlir-opt` is absent. The real dispatch
requires a zero exit and a non-empty output artifact before returning
PASSED. Current Stage 213 validation/backend verification: 69 focused
tests; the fast MLIR slice is 205 passing tests on this machine.

## 4. What's next (in order)

1. **(Optional) finish Stage 212's deferred ops** — the 12 above.
   Stage 212 is "closed enough" per the plan, but a future run can add
   these emitters. The attribute-heavy ones (matmul / reduce / memref
   / index-hbm) need design work first.
2. **Stage 214 — the progressive-lowering pass pipeline** (next).
   Stage 213 is CLOSED. The five `_MLIR_BACKEND_OUTPUT_VALIDATORS_AUTHORITY`
   table entries are still `None`, deliberately; Stage 214 wires the
   real pass pipelines and the target output validators together.
   Before starting, run `python scripts\mlir_audit_canaries.py` in
   `--strict` mode — it should report 31/31 passed.

   **Design question that Stage 214 must resolve first**: `mlir-opt`
   alone only lowers between MLIR dialects — its output is still MLIR
   text (in `llvm.func` / `nvvm.kernel` / `rocdl.kernel` / `spirv.func`
   form), NOT raw LLVM IR / PTX / HIP / MSL / WGSL. The Stage-213
   runner at `_run_mlir_opt_pipeline` (`backends.py:3766-3777`)
   explicitly REJECTS MLIR-shaped output with the finding "produced
   MLIR, not a target artifact; the artifact translation step is not
   wired". The downstream translate tool is target-specific:

   - **LLVM_IR**: `mlir-translate --mlir-to-llvmir` reads LLVM-dialect
     MLIR text and emits raw LLVM IR.
   - **PTX**: typically `mlir-translate --mlir-to-llvmir` then `llc
     -mtriple=nvptx64 -mcpu=sm_80` — two stages.
   - **ROCM_HIP**: similar but for AMDGPU triple.
   - **METAL_MSL** / **WEBGPU_WGSL**: routed through SPIR-V (via
     `--convert-gpu-to-spirv` + `--spirv-translate-module-to-binary`)
     and then `spirv-cross` (Metal) or `tint` (WGSL) — two-stage.

   Stage 214 must choose ONE of:

   - **Approach A — extend pipeline-tuple semantics**: the pipeline
     stays a `tuple[str, ...]` but a new convention encodes a final
     translate-tool reference (e.g. `("--convert-func-to-llvm", "--",
     "mlir-translate", "--mlir-to-llvmir")`). The runner splits on
     `--` and dispatches the suffix to the named tool. **Risk**:
     overloads pipeline-tuple semantics; the `--` delimiter rule is
     easy to drift on.
   - **Approach B — add a parallel `_MLIR_BACKEND_TRANSLATORS_AUTHORITY`
     table**: `dict[MLIRBackendTarget, tuple[str, str, tuple[str, ...]]
     | None]` where the tuple is (tool-name, flag, follow-up-args).
     The runner chains mlir-opt → mlir-translate → optional follow-up
     (llc / spirv-cross / tint) and at each stage verifies the
     artifact shape. **Risk**: 3 tools deep means 3 invocations to
     time-out / detect / read; lots of moving parts.
   - **Approach C — keep mlir-opt-only and update the output
     validators to accept LLVM-dialect MLIR**: the `_llvm_ir_artifact_
     is_plausible` predicate is broadened to accept either raw LLVM IR
     or MLIR text in LLVM dialect. The translate step is deferred to
     Stage 215 / 221. **Risk**: the "raw artifact" contract is the
     entire reason Stage 213 fail-closed at the runner — relaxing it
     here defeats the purpose; downstream consumers still need raw
     LLVM IR.

   **Recommendation for the next iteration**: Approach B. Define
   `_MLIR_BACKEND_TRANSLATORS_AUTHORITY` with `None` for every target
   in chunk A (mirroring how the validator table started), then wire
   targets one at a time in subsequent chunks. The runner gains a
   single new chain after the mlir-opt step that consults this table
   and invokes `mlir-translate` (+ any further tool) with the same
   argv-list / timeout / brand-the-result rigor the rest of the
   runner uses. The dev machine has neither tool, so production
   results stay DEFERRED with informative findings until a future
   toolchain becomes available.

   **Chunk-A scope** (next iteration): only the type and the table.
   Pipeline tuples and validators stay `()` / `None` for now. The
   `_check_mlir_backend_tables` drift-guard gets one new clause
   enforcing the translator table is total over `MLIRBackendTarget`.

   **Chunks A, B, C, D, E, F shipped 2026-05-24** (chunks D and E
   include their own audit-fix batches — 3 HIGH + 3 must-fix MEDIUM
   on D, 2 must-fix MEDIUM on E; chunk F audit verdict was SHIP with
   no HIGH/must-fix MEDIUM). State of Stage 214:

   - **Chunk A** — translator-step table scaffold
     (`_MLIR_BACKEND_TRANSLATORS_AUTHORITY`, `backend_translator()`,
     drift-guard clause) — `backends.py`. None-everywhere baseline so
     subsequent chunks can wire one target at a time without breaking
     the gate.
   - **Chunk B** — `mlir-translate` toolchain plumbing
     (`MLIRSupport.mlir_translate`, `can_use_mlir_translate()`,
     probe extended) — `toolchain.py`. Independent third surface
     alongside bindings + `mlir-opt`; `is_available()` deliberately
     stays "bindings or mlir-opt".
   - **Chunk C** — private `_run_mlir_translate_step(...)` helper
     (`backends.py`). Same subprocess hygiene as the rest of the
     runner family (argv-list, timeout, captured Timeout/OSError/
     nonzero diagnostics, blank-output rejection). Returns
     `(output_text, findings)`.

   **Chunk D done**: `_run_mlir_opt_pipeline` now chains
   `_run_mlir_translate_step` when the target's translator entry is
   populated. `lower_mlir_to_backend` gates on translator-vs-
   `support.mlir_translate`. `MLIRBackendResult.output_provenance`
   gains `mlir-translate=<path>` and `mlir-translate-flag=<flag>`
   entries when the chain ran. `lowering_tool` stays the primary
   tool path (documented). The drift-guard re-checks the translator
   tuple at the runner boundary (catches monkeypatched malformed
   entries). Non-empty `follow_up_args` returns FAILED with a clear
   "chunk-E chained-tool runner required" finding (the next stage of
   the chain dispatcher is not yet wired). `__all__` pins the public
   surface so private runners are not in `from backends import *`.

   **Chunk E done**: LLVM_IR target wired end-to-end. Pipeline is
   the 8-arg canonical mlir-opt lowering (audit caught a missing
   `--convert-index-to-llvm` pass — added with explanatory comment).
   Translator is `("mlir-translate", "--mlir-to-llvmir", ())` — empty
   `follow_up_args` so LLVM_IR does not need the chained-tool runner.
   Output validator is `_llvm_ir_output_validator` which wraps
   `_llvm_ir_artifact_is_plausible` and returns
   evidence=(validator=..., predicate=..., target=llvm_ir) on pass.
   End-to-end e2e test (mocked subprocess) verifies the full chain
   produces PASSED with mlir-translate path + flag in
   `output_provenance`.

   **Chunk F done**: chained third-stage tool runner shipped. New
   private `_run_chained_tool_step(input_text, *, tool_path, args,
   timeout_s)` helper. `_run_mlir_opt_pipeline` gains `chained_tool`
   parameter; invokes the chained tool when `follow_up_args` is
   non-empty (replacing the chunk-D fail-closed). `MLIRSupport` gains
   `llc: Optional[str]` field and `chained_tool_path(name)` lookup
   method. `lower_mlir_to_backend` resolves the chained tool path and
   adds a soft-DEFERRED gate when the tool isn't on PATH.
   `output_provenance` records `chained-tool=<path>`,
   `chained-tool-name=<name>`, `chained-tool-args=<args>`.

   **Chunks G+ scope** (next iterations): wire each GPU target's
   pipeline + translator entry + output validator together. Per the
   chunk-E pattern (which wired LLVM_IR end-to-end in one chunk).
   Suggested order PTX → ROCM_HIP → METAL_MSL → WEBGPU_WGSL.

   **Chunk G scope** (next iteration): wire PTX end-to-end.
   - Define `_MLIR_BACKEND_LOWERING_PIPELINES_AUTHORITY[PTX]`. Modern
     MLIR PTX lowering: `--gpu-kernel-outlining`,
     `--convert-gpu-to-nvvm`, then the LLVM-dialect lowering passes
     from chunk E (arith / func / cf / vector / index / memref) so
     the dialect-MLIR output is ready for `mlir-translate
     --mlir-to-llvmir` and then `llc`.
   - Define `_MLIR_BACKEND_TRANSLATORS_AUTHORITY[PTX] =
     ("mlir-translate", "--mlir-to-llvmir",
       ("llc", "-mtriple=nvptx64", "-mcpu=sm_80"))`.
   - Define `_MLIR_BACKEND_OUTPUT_VALIDATORS_AUTHORITY[PTX]` as a
     `_ptx_output_validator` callable that uses
     `_looks_like_backend_output(PTX, ...)` -> `_ptx_artifact_is_plausible`
     and returns `MLIRBackendOutputValidation` clean candidate with
     evidence on pass.
   - Tests pin the full three-stage chain with mocked subprocess
     invocations. Verify PASSED end-to-end. Check provenance
     includes the chained-tool entries.
   - 3-clean audit, commit + push + Telegram.

   **Stage 214 close** (after chunks G/H/I/J): when all five targets
   are wired and audited, run the Stage-214 holistic close audit
   (silent-failure / type-design / code-review across the whole
   backends.py + toolchain.py changes); fix any HIGH or must-fix
   MEDIUM; close the stage; bump `V3_STAGES_DONE` to 14.
3. **Stage 215 — the MLIR-vs-tile-IR parity gate** (verify the new
   path matches the home-grown path).
4. **Stage 216 — the end-of-Phase-E 5-clean-gate.**
5. **Phase F (Stages 220–222)** — backend unification, the Stage-221
   cutover, the v3.0.0 5-clean-gate + git tag.
6. **206-R residual ops** — additive LLVM-lowering chunks (print_int,
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
- **Current MLIR audit canaries**:
  `python scripts\mlir_audit_canaries.py` reports the known-open
  verifier/backend proof families; `--strict` is the pre-commit gate
  once those families are fixed.
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
| `helixc/ir/mlir/` | Phase-E MLIR substrate: `toolchain.py`, `mapping.py`, `helix_dialect.py`, `validate.py`, `emit.py`, `backends.py`. |
| `helixc/tests/test_mlir_*.py` | The MLIR-path tests. |

## 10. Current MLIR Audit Restart Note (2026-05-21 23:23Z)

The latest accelerated heartbeat stopped at an uncommitted, tested checkpoint.
Do not commit until strict canaries are clean.

- Closed: quoted-symbol/interface correspondence and a large sibling set in
  `validate.py` / `backends.py` / MLIR tests.
- Verified: focused validator/backend tests `240 passed`; MLIR slice
  `376 passed, 4347 deselected`; compileall clean; `git diff --check` clean
  except LF-to-CRLF warnings.
- Still open: `scripts\mlir_audit_canaries.py --strict` fails
  fake-validator bad-type, fake-validator addf-i32, and GPU backend symbol
  binding.
- Next restart source: `docs\HELIX_MLIR_AUDIT_PACKET.md` section
  `2026-05-21 23:23Z Heartbeat Checkpoint`.

## 11. Current MLIR Audit Restart Note (2026-05-22 01:16Z)

The latest accelerated heartbeat stopped at an uncommitted, tested checkpoint.
Do not commit yet.

- Closed/advanced: GPU backend symbol-binding canary now passes; PTX wrong
  entry reports a missing PTX entry for `expected`. The chunk added target-aware
  symbol extraction for PTX, ROCm/HIP, Metal MSL, and WGSL, plus many sibling
  regression tests in `helixc/tests/test_mlir_backends.py`.
- Verified: backend tests `94 passed`; focused validator/backend tests
  `254 passed`; MLIR slice `390 passed, 4347 deselected`; compileall clean;
  `git diff --check` clean except LF-to-CRLF warnings.
- Still open: strict canaries fail fake-validator bad-type and fake-validator
  `arith.addf` over `i32`.
- Still open from the final GPU-family audit: WGSL malformed parameter names
  and attributed params without identifiers still bind; PTX `.func` forms with
  `.reg` return params or `.noreturn` are currently false-rejected.
- Next restart source: `docs\HELIX_MLIR_AUDIT_PACKET.md` section
  `2026-05-22 01:16Z Heartbeat Checkpoint`.

## 12. Current MLIR Audit Stop Note (2026-05-22)

The latest stop point is uncommitted but tested. Treat the older restart notes
above as historical; this note and `docs\HELIX_MLIR_AUDIT_PACKET.md` are the
current source.

- Closed since the 01:16Z checkpoint: all strict MLIR audit canaries now pass;
  WGSL malformed parameter names and missing parameter identifiers reject; PTX
  `.reg` function params and `.noreturn` function directives accept; malformed
  PTX predicate guards reject.
- Closed from the post-fix audit findings: unsupported obvious function types,
  duplicate `func.func` symbols, empty returns from non-void functions,
  same-line function declaration boundary drift, malformed backend output
  symbol extraction, and loose MLIR/backend tool identity.
- Verified: `python scripts\mlir_audit_canaries.py --strict` -> `7 passed /
  0 failed`; focused validator/backend tests -> `266 passed`; MLIR slice ->
  `402 passed, 4347 deselected`; compileall clean; `git diff --check` clean
  except LF-to-CRLF warnings.
- Still required before any commit: re-run all three audit axes from scratch.
  The prior audit round was BLOCKED before these fixes landed, so this is not
  yet committable.
- Next restart source: `docs\HELIX_MLIR_AUDIT_PACKET.md` section
  `2026-05-22 Stop Checkpoint After Audit Fix Batch`.

## 13. Current MLIR Audit Stop Note (2026-05-22 Third Audit Round)

The latest stop point is uncommitted, tested, and audit-blocked. Do not commit
or push this packet until the open audit findings below are fixed and all three
audit axes rerun clean.

- Closed since the previous stop: strict MLIR canaries now pass `12/12`;
  focused validator/backend tests pass `274`; the MLIR pytest slice passes
  `410`; compileall is clean; `git diff --check` reports only line-ending
  warnings.
- Third audit status: silent-failure and type-design axes returned BLOCKED; the
  general-review axis was stopped before completion at the user's request.
- Open HIGH: control predicates are still underchecked (`scf.if`,
  `cf.cond_br`, `cf.assert` can accept non-`i1` predicates); memref access
  semantics are still underchecked; several constants/vector/loop semantics
  still accept invalid forms; generic function bodies can bypass canonical
  terminator/static checks.
- Open MEDIUM: generic `llvm.func` input symbol binding is still skipped in one
  path; LLVM typed-value validation can accept scalar constants for
  aggregate/vector returns; HIP/MSL C-like preflight still accepts impossible
  declarations/statements in some cases.
- Next restart source: `docs\HELIX_MLIR_AUDIT_PACKET.md` section
  `2026-05-22 Stop Checkpoint After Third Audit Round`.
