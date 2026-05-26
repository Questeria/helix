# Helix v3.0 — Handoff

**Last updated:** 2026-05-26 · **Repo:** `C:/Projects/Kovostov-Native` ·
**Branch:** `main` (verify live state with `git status --short --branch`
and `git log -1 --oneline`)

> **HARD CONSTRAINT (user directive 2026-05-26):** At v1.0 release,
> the project must contain **zero non-Helix runtime code**. Python
> helixc must be deleted (K4 mandatory, not optional). GPU/MLIR/Tile
> ops must be ported to the bootstrap — they cannot stay in Python
> forever. Full text + verification criteria in
> [`docs/K_BOOTSTRAP_HARD_CONSTRAINT.md`](K_BOOTSTRAP_HARD_CONSTRAINT.md).

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

## 2. Where the project stands (2026-05-24)

- **v2.0–v3.0: released**.
- **v3.0.0 RELEASED 2026-05-24** — all 19 numbered stages across
  three phases complete:
  - **Phase D (Stages 200–208): COMPLETE.**
  - **Phase E (Stages 210–216): COMPLETE.**
  - **Phase F (Stages 220–222): COMPLETE.** Stage 220 (shared
    Backend Protocol) closed in commit 71bbe8a. Stage 221
    (cutover — LLVM becomes the canonical backend; `x86_64.py`
    marked LEGACY through v3.0.x) closed in commit b9fe8be.
    Stage 222 (end-of-v3.0 5-clean-gate + `v3.0.0` tag) closed
    in this commit.
- **`V3_STAGES_DONE = 19` of 19** (`scripts/helix_status.py`) —
  100 % of v3.0 stages.

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

   **Chunks A, B, C, D, E, F, G, H shipped 2026-05-24** (chunks D
   and E include their own audit-fix batches — 3 HIGH + 3 must-fix
   MEDIUM on D, 2 must-fix MEDIUM on E; chunk F audit verdict was
   SHIP with no HIGH/must-fix MEDIUM; chunks G and H needed
   test-helper migration but no fail-closed defects). State of
   Stage 214:

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

   **Chunk G done**: PTX wired end-to-end. Pipeline starts with
   `--gpu-kernel-outlining`, then the LLVM-dialect lowering passes
   from chunk E, then `--convert-gpu-to-nvvm`. Translator chains
   `llc -mtriple=nvptx64 -mcpu=sm_80 -O2`. Output validator wraps
   `_ptx_artifact_is_plausible`. Test-helper `_register_output_validator`
   was updated to also reset the target's translator to None by
   default so pre-chunk-G tests that assumed un-wired translator keep
   their original intent; chunk-D/F tests that wire a translator now
   call `_register_*_validator` BEFORE `_wire_translator`. 6 new
   chunk-G tests pin validator behaviour, pipeline shape, translator
   shape, and the toolchain-absent DEFERRED gate.

   **Chunks H/I/J scope** (next iterations): wire the remaining GPU
   targets ROCM_HIP, METAL_MSL, WEBGPU_WGSL. Each needs a per-target
   pipeline + translator entry + output validator AND a new chained
   tool path in MLIRSupport.

   **Chunk H done**: ROCM_HIP wired end-to-end. Pipeline mirrors
   PTX except `--convert-gpu-to-rocdl` swaps in for NVVM. Translator
   chains `llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -O2`. Output
   validator wraps `_rocm_hip_artifact_is_plausible`. No new
   MLIRSupport field needed — `llc` already wired in chunk F.

   **Chunks I/J scope**: METAL_MSL and WEBGPU_WGSL both route via
   SPIR-V. They need additional `MLIRSupport` fields and a binary
   variant of `_run_mlir_translate_step`:
   - `mlir-translate --serialize-spirv` produces a BINARY SPIR-V
     module — chunk C's text-only helper cannot read it.
   - METAL_MSL chains `spirv-cross --msl` to convert SPIR-V →
     Metal Shading Language (text).
   - WEBGPU_WGSL chains `tint --format wgsl` to convert SPIR-V →
     WGSL (text).

   **Chunk I scope** (next iteration): add the binary-output helper
   AND wire METAL_MSL end-to-end together (the helper is a small
   addition and is exercised by the METAL_MSL chain immediately).
   - Add `_run_mlir_translate_step_binary(input_text, *,
     mlir_translate, flag, timeout_s)` returning `(bytes | None,
     findings)`. Same subprocess hygiene as the text variant; opens
     the output file with `"rb"`.
   - Add `_run_chained_tool_step_binary_input(input_bytes, *,
     tool_path, args, timeout_s)` that writes the bytes to the
     input file and reads stdout / out file as TEXT (Metal MSL is
     text output even though its input is SPIR-V binary).
   - Extend `MLIRSupport` with `spirv_cross: Optional[str]` and
     extend `chained_tool_path` to map `"spirv-cross"` → it.
   - Define `_METAL_MSL_LOWERING_PIPELINE` (mirror PTX/ROCM but
     `--convert-gpu-to-spirv` in place of the rocdl/nvvm pass);
     `_METAL_MSL_TRANSLATOR` = `("mlir-translate",
     "--serialize-spirv", ("spirv-cross", "--msl"))`;
     `_metal_msl_output_validator` wraps
     `_metal_msl_artifact_is_plausible`.
   - Wire the runner: when the registered translator's flag is
     `"--serialize-spirv"`, route through the binary helper +
     binary-input chained-tool helper.

   **Chunk J scope** (subsequent iteration): wire WEBGPU_WGSL using
   the binary helper from chunk I. New `MLIRSupport.tint` field;
   `_WEBGPU_WGSL_TRANSLATOR` = `("mlir-translate", "--serialize-spirv",
   ("tint", "--format", "wgsl"))`.

   **Stage 214 close** (after chunks H/I/J): when all five targets
   are wired and audited, run the Stage-214 holistic close audit
   (silent-failure / type-design / code-review across the whole
   backends.py + toolchain.py changes); fix any HIGH or must-fix
   MEDIUM; close the stage; bump `V3_STAGES_DONE` to 14.
3. **Stage 215 — the MLIR-vs-tile-IR parity gate** (in progress).

   **Chunk A done** (2026-05-24, commit 94d85fb): the harness
   skeleton in `helixc/ir/mlir/parity.py`. Defines `ParityStatus`
   (PARITY_HOLDS / PARITY_FAILED / PARITY_DEFERRED), `ParityResult`
   frozen dataclass (final, post_init-guarded — same discipline as
   the Stage 213 result types), and
   `mlir_vs_tile_ir_parity_check(module, target, *, support)`. The
   entry point catches `MLIRTranslationError`, then maps the MLIR
   backend chain's status to a parity verdict (PASSED -> HOLDS,
   FAILED -> FAILED, DEFERRED -> DEFERRED).

   **Chunk B done** (2026-05-24, commit 0814fcd): the harness now
   runs BOTH paths from one entry. `_run_tile_ir_path(module, target)`
   dispatches to the per-target home-grown emitter (PTX / HIP / MSL /
   WGSL); LLVM_IR returns a structural deferral pointing at the
   Phase-D parity gate (Stage 207) because the home-grown LLVM path
   consumes `tir.Module`, not `TileModule`. `ParityResult` gains a
   `tile_ir_output: Optional[str]` field that records the home-grown
   artifact when present. Home-grown failure short-circuits to
   PARITY_FAILED before the MLIR side runs.

   **Chunk C done** (2026-05-24, commit 33de085): per-target
   normalization (strip comments + cosmetic-prefix lines + fold
   whitespace) and SHA-256 comparison of normalized forms.
   PARITY_HOLDS only when both paths' artifacts agree after
   normalization.

   **Stage 215 CLOSED** (2026-05-24): the silent-failure axis of
   the close audit flagged 2 HIGH + 3 MEDIUM findings — all
   addressed in the chunk-D audit-fix batch:
   - HIGH-1: refuse to mint PARITY_HOLDS on empty normalized forms
     (two comments-only artifacts would otherwise SHA-256-collide
     on the empty string).
   - HIGH-2: added `is_positive_assertion()` helper that returns
     True only for PARITY_HOLDS; LLVM_IR findings now carry a
     `mlir_side_status=` machine-readable prefix so callers cannot
     misread PARITY_DEFERRED as a ship-approved assertion.
   - MEDIUM-1: `_run_tile_ir_path` preserves the full traceback in
     findings; re-raises MemoryError instead of swallowing it.
   - MEDIUM-3: normalization strips C-style `/* ... */` block
     comments (PTX / HIP / MSL / WGSL).
   The type-design + code-review axes of the close audit were
   deferred to Phase E's 5-clean-gate (Stage 216), where they will
   run across the entire E-phase surface.

   `V3_STAGES_DONE` bumped from 14 to 15.
4. **Stage 216 — the end-of-Phase-E 5-clean-gate** — CLOSED 2026-05-24.
   Type-design audit (2 HIGH + 3 MEDIUM) and code-review audit
   (1 MEDIUM) ran against the entire Phase-E surface. All HIGH and
   must-fix MEDIUM findings closed in the chunk-D fix batch:
   - HIGH-1: `_HOMEGROWN_EMITTERS` value type tightened to
     `Callable[[], BackendEmitter]`; runtime `isinstance` check
     added in `_run_tile_ir_path` for belt-and-suspenders.
   - HIGH-2: New `_check_parity_target_tables()` drift guard runs
     at module load — refuses any partial coverage of the
     MLIRBackendTarget set in `_TARGET_LINE_COMMENT_MARKER` /
     `_TARGET_COSMETIC_LINE_PREFIXES` / `_HOMEGROWN_EMITTERS`.
   - MEDIUM-1: `is_positive_assertion()` promoted to the whole
     Phase-E result-type family (`MLIRValidation`,
     `MLIRBackendResult`, `ParityResult`) — release-gate callers
     have a uniform DEFERRED-safe predicate.
   - MEDIUM-3: `ParityResult.__post_init__` delegates target
     validation to the same `_require_backend_target` helper the
     other result types use.
   - Code-review M1: Cosmetic-prefix lists for LLVM_IR / ROCM_HIP /
     METAL_MSL / WEBGPU_WGSL are now pinned by tests so accidental
     drift is impossible (the PTX entry already had a test).
   Phase E (Stages 210–216) is COMPLETE.
5. **Phase F (Stages 220–222)** — backend unification, the Stage-221
   cutover, the v3.0.0 5-clean-gate + git tag.

   **Stage 220 — CLOSED 2026-05-24** (commit 71bbe8a). The shared
   `Backend` dataclass + `_BACKEND_REGISTRY` + `get_backend(target)`
   accessor unify the four GPU emitters behind one interface; LLVM_IR
   is intentionally excluded (Stage 221's cutover retires the
   home-grown LLVM path).

   **Stage 221 — DESTRUCTIVE, NEEDS EXPLICIT USER CONFIRMATION
   BEFORE STARTING.** The plan (`docs/V3_PLAN.md` line 130-133)
   states verbatim: "CUTOVER (destructive; user checkpoint). With
   parity gates 207 + 215 green, retire `x86_64.py` and the
   home-grown tile-IR behind a flag, then remove. Recommend
   explicit user confirmation here even under blanket authority —
   it is the one irreversible step."

   The cron-tick worker MUST NOT auto-start Stage 221. When the
   next tick fires and sees `V3_STAGES_DONE = 17` with Stage 221
   pending, the worker should:
   - Either do the prerequisite 206-R residual ops (additive
     LLVM-lowering chunks for print_int / write_file /
     read_file_to_arena / TRACE / arena / QUOTE-family — these
     need to land before the cutover so the LLVM path is feature-
     complete enough to replace x86_64.py);
   - Or send a Telegram noting the cutover gate and stop.

   **Stage 222 — end-of-v3.0 5-clean-gate + tag `v3.0.0`**. Depends
   on Stage 221 done.

6. **206-R residual ops** — additive LLVM-lowering chunks needed
   before the Stage-221 cutover. Safe for autonomous worker to start.

   **print_int — SHIPPED 2026-05-24** (commit c7b7cec).
   `helixc/backend/llvm_ir.py` gained an internal-helper-function
   registry (`_HelperFunctionSpec`, `_FFIDeclareSpec`,
   `_HELPER_FUNCTIONS` via `MappingProxyType`,
   `_check_helper_function_table` drift guard) and the
   `@__helix_print_int(i32) -> i32` helper (i32->ASCII digit-loop +
   `write(1, buf, len)`, five basic blocks; bit-for-bit parity with
   `x86_64.py::print_int` including INT_MIN). `PRINT._kind="print_int"`
   now emits `call i32 @__helix_print_int(i32 %v)`; the helper text
   is emitted once per module via the registry. Audit-fix batch
   landed in the same commit: HIGH-1 (helper-vs-FFI-declare
   collision), 4 MEDIUMs (registry tightening + house-style
   migration to `@dataclass(frozen=True, slots=True)` with
   `__init_subclass__` and `__post_init__` raising `ValueError`).

   **write_file — SHIPPED 2026-05-24** (commit ac366c6). Inline
   lowering in `helixc/backend/llvm_ir.py`: registers `@open`,
   `@write`, `@close` libc declares + the path (NUL-terminated) and
   content string globals; emits the six-instruction sequence
   `open -> write -> close -> trunc -> icmp slt -> select`. Constants
   match x86_64.py (577 = O_WRONLY|O_CREAT|O_TRUNC, 420 = 0o644).
   Audit-fix batch: HIGH-1 (embedded NUL in path silently truncated
   via open() C-string semantics — now rejected); 3 cross-backend
   contract gaps (short-write success masking, close errors discarded,
   open errors propagated as -EBADF from downstream write) are
   documented inline as Stage 207 parity-gate decisions matching
   the existing `# NOTE (Stage 207 parity)` discipline; LOW-3 test
   added that locks the embedded-NUL-in-content-preserved contract.

   **arena infrastructure + ARENA_PUSH — SHIPPED 2026-05-24** (this
   chunk). Foundational shared infrastructure for the ARENA op family
   and read_file_to_arena. New `_ModuleGlobalSpec` frozen dataclass
   + `_MODULE_GLOBALS` `MappingProxyType` registry (parallel to
   `_HELPER_FUNCTIONS`), extended `_HelperFunctionSpec` with a
   `module_globals: tuple[str, ...] = ()` field, `_check_module_global_table`
   drift guard (three invariants: key-vs-name; forward dependency
   resolution; helper-vs-global-name collision rejection). The arena
   global `@__helix_arena_base = internal global [2097153 x i32]
   zeroinitializer` (CAP=2097152 matches x86_64.py) is emitted only
   when at least one helper actually pulls it in. The
   `@__helix_arena_push(i32) -> i32` helper is a 3-block multi-block
   LLVM function (entry / in_bounds / exit with phi) — overflow is
   folded into entry's else branch. ARENA_PUSH lowers to a single
   `call i32 @__helix_arena_push(...)`. Audit-fix batch: MEDIUM
   (helper-vs-module-global name collision drift-guard); silent-
   failure M-2/M-3 (docstring + block-count drift); test gaps
   (module-global-vs-FFI-declare collision; helper-vs-global registry
   collision; empty-name/def; `module_globals` duplicate detection;
   parity-sensitive helper text pinning).

   **ARENA_GET / ARENA_SET / ARENA_LEN — SHIPPED 2026-05-24** (commit
   7a0d332). Three more arena ops, each its own small helper sharing
   the `@__helix_arena_base` global from the prior chunk. GET + SET
   are 3-block bounds-checked routines (overflow returns 0 from GET,
   silently no-ops from SET — matches x86_64.py); LEN is a single
   load. ARENA_SET's op handler tolerates an optional result slot
   (TIR says no result; x86_64.py tolerates one) — the helper always
   returns i32 0, the op handler binds or discards uniformly. Audit-
   fix batch: NO HIGH/MEDIUM across all three axes; 1 LOW (lint
   `E741` ambiguous `l` variable name) + two coverage gaps (ARENA_SET
   non-i32 result, ARENA_SET >1 results) addressed.

   **ARENA_PUSH_PAIR / ARENA_PUSH_TRIPLE — SHIPPED 2026-05-24** (commit
   b0bf3d4). Atomic multi-slot pushes. PAIR (2 i32 operands, threshold
   `cursor >= CAP - 1`) writes at cursor+1/cursor+2, advances cursor
   by 2, returns OLD cursor or -1. TRIPLE (3 operands, threshold
   `cursor >= CAP - 2`) writes at cursor+1/2/3, advances by 3. Both
   atomic-or-none: on overflow neither/none of the writes happen AND
   the cursor does not advance — pinned by structural tests that
   assert no `store i32` lives outside the `in_bounds:` block.
   Shared `_HELIX_ARENA_GLOBALS` constant introduced (all six arena
   helpers now reference the same tuple by name). Audit-fix batch:
   1 MUST-FIX MEDIUM (missing TRIPLE non-i32-left test — exercises
   loop's first iteration), 1 LOW (stale "ARENA family" mention in
   PRINT catchall), 1 LOW (exit-block atomic-test pinning), plus
   helper-text/structural/dedup pinning for both helpers.

   **TRACE_ENTRY / TRACE_EXIT — SHIPPED 2026-05-24** (commit 7ce6280).
   First VOID-returning helper. Two new module globals
   (`@__helix_trace_count: i32`, `@__helix_trace_buf: [2*CAP x i32]`,
   CAP=1024 matches x86); `@__helix_trace_event(i32 fn_id, i32 kind)
   -> void` 3-block helper (full-buffer fail-closed, atomic-or-none
   on overflow). New `_intern_trace_fn_ids` pre-pass in `emit_module`
   walks every TRACE op in deterministic order to assign a stable
   per-module fn-id table, then passes the table to every per-fn
   emitter so a fn_name appearing across functions resolves to one
   id. TRACE_EXIT emits a `bitcast` keepalive on its optional operand
   to force an LLVM-IR use (mirrors x86's `mov eax, [slot]` load
   for liveness — without this, LLVM DCE could drop sole-use defs).
   Type-design polish landed: `_HelperFunctionSpec` gained an
   explicit `ret_ty` field cross-checked against the helper text by
   `_check_helper_function_table` (call-site `call <ret_ty>
   @<name>(...)` can no longer drift from `define internal <ret_ty>`
   silently; `mock_validate_ll` does not catch this otherwise);
   `_FnEmitter.trace_fn_ids` typed as `Mapping[str, int]` to prevent
   per-op-handler mutation. Audit-fix batch: NO HIGH; 3 MEDIUMs
   (stale PRINT catchall, TRACE_EXIT operand drop, "concurrent
   mutation" diagnostic misattribution) + 1 LOW (kernel-skip in
   pre-pass) all closed.

   **read_file_to_arena — SHIPPED 2026-05-24** (commit b3c9546).
   Adds `_HelperFunctionSpec.helper_deps` for TRANSITIVE helper-call
   dependencies. `_register_helper_function` becomes recursive with
   idempotency-via-set; `_check_helper_function_table` adds body
   cross-check + DFS-based CYCLE DETECTION (audit HIGH-1). Body
   cross-check now strips `;` comments via `_strip_llvm_comment`
   (audit HIGH-2). The helper is 6 blocks: opens O_RDONLY, reads
   up to 1 MiB, traps via `@llvm.trap()` on truncation (matches
   x86's `ud2`), sign-clamps nread, per-byte push to shared arena.
   Four Stage 207 parity notes inline. Polish landed:
   `_SUPPORTED_PRINT_KINDS` frozenset constant, `_validate_path_attr`
   shared helper.

   **QUOTE / SPLICE / MODIFY — SHIPPED 2026-05-24** (this chunk —
   the FINAL 206-R chunk). AGI metaprogramming primitives + new
   `@__helix_state_base = [64 x i64]` reflection-cells global
   (matches x86's HELIX_NUM_CELLS=64, HELIX_CELL_SIZE=8). QUOTE:
   pure inline `add i32 0, <ast_handle % NUM_CELLS>` (compile-
   time constant; bool ast_handle rejected via `type(...) is int`
   matching CONST_INT discipline; negative handles wrap into
   [0,NUM_CELLS) via Python `%` matching x86 exactly). SPLICE:
   3-block helper, bounds-checked i64 load + trunc to i32,
   returns 0 on OOB. MODIFY: 4-block helper takes
   `(i32 handle, i32 new_value, ptr verifier)` — bounds-check,
   call verifier through function pointer, conditional store,
   3-way exit phi (0/0/1). MODIFY also has a LEGACY FALLBACK
   (audit HIGH-1) for the 3-operand-no-verifier_fn form that
   x86_64.py supports — emits inline `icmp ne i32 %op2, 0; zext`
   instead of the helper call. Without this branch, programs
   using the dynamic-verifier form would compile on x86 but fail
   on LLVM (real parity divergence — existing test_codegen.py:4779
   would break post-cutover). Only i32 value_kind lowered; f32/f64
   variants raise loudly (deferred to a polymorphic-helper follow-
   up). REFLECT_HASH unimplemented in BOTH backends — lands in
   the LLVM catchall fail-closed, matching x86's
   NotImplementedError.

   **ALL 206-R ADDITIVE PREP IS DONE.**

   **Stage 221 — CUTOVER COMPLETE 2026-05-24** (this commit). User
   green-lit on the same day. Pragmatic two-step "behind a flag,
   then remove" interpretation given the ~1000+ test_codegen.py
   tests depend on x86_64's compile_module_to_elf for runnable
   binaries (the LLVM real-execution toolchain integration is
   itself a v3.1 deliverable). Sub-step 221a (this commit):
   - Added `--emit-llvm-ir` to `helixc/check.py` -- the CANONICAL
     v3.0+ backend output (parse -> typecheck -> lower -> emit
     LLVM IR text). Threaded through `_KNOWN_LONG_FLAGS`,
     `stdout_modes` mutex set, and the lower-gate so the flag
     actually drives the IR pipeline.
   - Marked `--emit-asm` (x86_64 ELF hex dump) as LEGACY in the
     CLI help text, with a pointer to `--emit-llvm-ir`.
   - Updated `helixc/backend/x86_64.py` module docstring to mark
     the module as LEGACY (retained through v3.0.x for the
     Stage 207 parity gate and the test_codegen.py compile+run
     suite); v3.1 cleanup completes the deletion once LLVM
     toolchain integration matures.
   3-clean audit: ship (small, additive, well-tested chunk).

   Sub-step 221b (DEFERRED to v3.1): actual deletion of
   `x86_64.py`. Requires (a) LLVM-toolchain-driven test execution
   for the test_codegen.py corpus, (b) migration of every
   `from .backend.x86_64 import compile_module_to_elf` caller
   (check.py: 2 sites; examples/run.py: 1; lower_ast.py: 1
   constant import; ~10 test files). The deferral is honest:
   v3.0.0 ships with LLVM as the canonical / advertised default
   backend; x86_64 lingers as a legacy implementation detail
   selectable via the LEGACY `--emit-asm` and `-o` paths.

   **Stage 222 — END-OF-v3.0 5-CLEAN-GATE + tag v3.0.0** (next
   chunk). When this commit ships, V3_STAGES_DONE = 18. Stage 222
   closes v3.0:
   - 5-axis audit across frontend / IR / backend / runtime / tests.
   - Bump V3_STAGES_DONE = 19.
   - Mark v3.0 status = "released" in scripts/helix_status.py
     VERSIONS table.
   - Create git tag `v3.0.0`, push.
   - Final Telegram.
   - Stop the loop (per standing prompt: "When v3.0.0 is
     released, the job is done — stop the loop").

   The remaining "x86 lowers, LLVM doesn't" cases are deferred-
   FEATURE shapes (float / struct support; the f32/f64 value_kind
   variants of SPLICE/MODIFY; REFLECT_HASH which is also
   unimplemented in x86) — not 206-R residual ops. The parity
   fixture points at f64 SPLICE to keep the UNCOVERED test
   meaningful.

When `v3.0.0` is tagged, v3.0 is done.

## 4a. Post-v3.0 — v3.1 cleanup track (in progress)

After v3.0.0 ships, the v3.1 cleanup track addresses the residual
items the cutover left behind. The autonomous worker can advance
steps 1-5 freely; **step 6 (x86_64.py deletion) needs explicit user
acknowledgement** per the same gate that protected Stage 221a/221b.

- **v3.1 step 1 — LLVM compile-to-ELF wrapper.** SHIPPED (commit
  `ef72950`). Adds `compile_module_to_elf_via_llvm` /
  `compile_module_to_elf_via_llvm_full` in `llvm_toolchain.py`.
  Tri-state (PASSED / FAILED / DEFERRED); host-OS guard returns
  DEFERRED on non-Linux without `HELIX_LLVM_CROSS=1`.
- **v3.1 step 3 — `_codegen_backend.py` test-side seam.** SHIPPED
  (commit `f5bfd6d`). `selected_backend()` reads
  `HELIX_TEST_BACKEND`; default is `"x86"`; `"llvm"` swaps every
  call site. `compile_or_skip` translates `LLVMToolchainAbsent`
  to `pytest.skip(...)`. 6 import sites in `test_codegen.py`
  migrated to use the helper.
- **v3.1 step 4 — f32 / f64 polymorphic SPLICE / MODIFY.** SHIPPED
  (commit `3359ba7`). Four new helpers
  (`__helix_splice_f32/f64`, `__helix_modify_f32/f64`) and two
  `MappingProxyType` dispatch tables (`_SPLICE_DISPATCH`,
  `_MODIFY_DISPATCH`) as the single source of truth for the
  polymorphic value_kind set. Validation set is derived from the
  dispatch keys — cannot drift. 8 new tests pin emission + SSOT.
  Note: positive-emission tests bypass `emit_module` (because the
  module-level float-return rejection still stands) via
  `_FnEmitter._emit_op` direct drive; this is documented as
  load-bearing-temporary until the broader float-arithmetic
  support lifts the float-return restriction.
- **v3.1 step 5 — REFLECT_HASH LLVM lowering.** SHIPPED (commit
  `3809baf`). `__helix_reflect_hash(i32 handle) -> i32` helper:
  bounds-check + i64 load from cell + splitmix64 finalizer
  (Stafford mix13) + truncate to i32. Multiplier constants are
  derived from named hex source-of-truth (`_SPLITMIX64_C1/C2_HEX`)
  via `hex - (1<<64)`; a module-load assertion
  `_check_reflect_hash_constants` pins the round-trip so a future
  typo crashes at import. **Audit caught a CRITICAL bug here**:
  the original hand-derived C2 decimal corresponded to a different
  hex than documented; both auditors (silent-failure-hunter and
  code-reviewer) independently flagged it. Fixed before shipping.
  7 new tests; x86_64.py still has no REFLECT_HASH arm — the
  asymmetry resolves at step 6.
- **v3.1 step 3b — bulk test-file migration to the seam.** SHIPPED
  (commit `630be9f`). 19 test files (58 import sites; biggest is
  `test_ir.py` @ 37) swapped `from helixc.backend.x86_64 import
  compile_module_to_elf` to the `_codegen_backend` helper. The
  helper's API is drop-in; no semantic change at default
  (`HELIX_TEST_BACKEND` unset → routes to x86). 1196 tests across
  the migrated surface verified green.
- **v3.1 step 6a — shared-constants module.** SHIPPED (commit
  `56859d1`). Runtime layout constants (HELIX_NUM_CELLS,
  HELIX_CELL_SIZE, HELIX_ARENA_CAP, HELIX_TRACE_CAP, SYSV_STACK_*)
  extracted to `helixc/backend/_shared_constants.py`. Both
  backends + `lower_ast.py` + `test_stage44` now read from one
  source of truth. Audit caught two converged MEDIUMs (small-int
  `is`-cache silent-pass + missing mirror contract pin); both
  fixed with source-grep checks paralleling the lower_ast pattern.
  Closes the `lower_ast → x86_64` and `test_stage44 → x86_64`
  constant dependencies that previously blocked the delete.
- **v3.1 step 6 — Delete x86_64.py.** **RE-SCOPED 2026-05-25.**
  The user's directive shifted the goal from "delete one Python
  file" to "the end product is completely in Helix compiled in
  Helix from raw binary, zero Python anywhere." Step 6 is now
  absorbed into the **K-Bootstrap track** — see
  `docs/HELIX_K_BOOTSTRAP_MASTER_PLAN.md`. The Python `helixc/`
  package (including `x86_64.py`, but also `llvm_ir.py`, the
  frontend, the driver, every pass) deletes together when the
  Helix-in-Helix path (`kovc.hx`) reaches feature parity AND
  the parity harness is green AND a trusted bootstrap seed
  exists. That is a multi-month effort.
- **v3.1 step 7 — Tag `v3.1.0`.** **READY** under option D from
  the 2026-05-25 decision table. The cleanup-track work that
  shipped (steps 3a/3b/4/5/6a) is coherent and audit-clean. The
  deletion piece is now K-track. Tagging v3.1.0 NOW gives a
  clean release boundary between the v3 cleanup era and the
  K-bootstrap era; v3.2 can then be redefined as the first
  K-track milestone or as the real-execution parity gate (the
  two now overlap — Stage 207 parity is exactly what Track P
  of the K-bootstrap needs).

When `v3.1.0` is tagged, the cleanup track closes. The
K-bootstrap track begins per `docs/HELIX_K_BOOTSTRAP_MASTER_PLAN.md`.

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
