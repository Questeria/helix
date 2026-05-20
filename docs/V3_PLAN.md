# Helix v3.0 Implementation Plan

v3.0 — the industrialization release. Opened 2026-05-20, immediately
after v2.5.0 (the PTX register-allocation emitter wiring) shipped.

## Premise & authority

v3.0 replaces two home-grown subsystems with industry-standard
infrastructure:

- **MLIR migration** — the home-grown tile-IR (`helixc/ir/tile_ir.py`,
  ~520 lines, plus `tir.py` ~600) → MLIR dialects.
- **LLVM IR backend** — the hand-rolled x86_64 ELF emitter
  (`helixc/backend/x86_64.py`, ~5500 lines, + `elf_dyn.py`) → textual
  LLVM IR consumed by `opt` + `llc`.

A note of honesty carried forward from the v2.0 research and the v2.x
`V2_PLAN.md` "v3.0 horizon": that research recommended **deferring the
v3.0 rewrite until an anchor customer or a hard performance ceiling
forces it** — the home-grown stack was sufficient for Phase 0 → v2.5.
v3.0 nonetheless proceeds, under the user's explicit standing
authority ("go as far as v3.0 without my approval"). To keep that
proceeding responsible, v3.0 is structured so it is **reversible at
every stage until a single, clearly-marked cutover** (Stage 221) — see
"Migration strategy" below.

## Why now / why these two

- The home-grown tile-IR has no general pass infrastructure, no
  progressive-lowering framework, and a fixed op set. MLIR supplies
  all three, plus reusable upstream dialects (`linalg`, `vector`,
  `gpu`, `llvm`).
- The x86_64 emitter hand-rolls register allocation, instruction
  selection and ELF encoding for one target. LLVM supplies industrial
  regalloc + isel for every target it supports — and the v2.4/v2.5
  linear-scan allocator, while correct, is a fraction of what LLVM's
  allocators do.
- v2.x already retired the two smaller v3.0-deferred items — real-HW
  dispatch (v2.4 item 13) and emitted-kernel register allocation
  (v2.4 item 15 / v2.5 Edit B). What remains is genuinely the "big
  rewrite".

## Migration strategy — parallel-path, parity-gated, additive-first

The single most important constraint: v3.0 must never leave the
compiler broken between stages.

1. **Additive.** Each new subsystem (the LLVM backend, the MLIR path)
   is built as a NEW module ALONGSIDE the existing one. The tile-IR
   and `x86_64.py` keep working untouched until their replacement is
   proven.
2. **Parity-gated.** A replacement is "proven" only when a parity
   harness shows it produces output observably identical to the
   incumbent across the entire `helixc/tests/` program corpus.
3. **One cutover.** Only after a parity gate passes does the incumbent
   get retired — and that retirement (Stage 221) is the ONE
   destructive step, explicitly flagged as a user checkpoint.
4. **External tools are optional at rest.** MLIR and LLVM command-line
   tools (`mlir-opt`, `opt`, `llc`) may be absent on a given machine.
   Every stage ships a mock-validation path that needs no toolchain
   and gates real dispatch behind tool-detection — the exact pattern
   `helixc/backend/gpu_ci.py` already uses for real-HW GPU dispatch.

## Phases

v3.0 sequences the LOWER-risk half first.

### Phase D — LLVM IR backend (the x86_64 replacement)
A new host backend that consumes the existing host IR and emits
textual LLVM IR. Purely additive: `x86_64.py` is untouched. This phase
also stands up the LLVM toolchain integration that Phase E reuses —
MLIR's standard lowering target is the `llvm` dialect → LLVM IR.

### Phase E — MLIR migration (the tile-IR replacement)
A parallel MLIR path: tile-IR → MLIR, MLIR → the existing backends.
Higher risk — the tile-IR is consumed by all five backends, the
adjoint, and every IR pass — so it follows Phase D, reusing Phase D's
LLVM path as MLIR's lowering target and Phase D's parity harness.

### Phase F — Backend unification & cutover
The deferred v2.2 Item 2 (a shared backend Protocol/ABC across all
backends), then the single cutover that retires the incumbents.

## Stages

Numbers are a starting layout; like `V2_PLAN.md` they will grow and
split as work reveals detail. The 200+ range keeps clear of the v2.x
110–131 range.

### Phase D — LLVM IR backend
- **Stage 200** — LLVM IR emitter substrate. Confirm `x86_64.py`'s
  input IR + entry points; create `helixc/backend/llvm_ir.py`; emit
  textual LLVM IR for the scalar core (module header / target triple,
  `define`, integer constants, `add`/`sub`/`mul`, `ret`). Mock-
  validate the `.ll` shape.
- **Stage 201** — LLVM toolchain detection + dispatch. Detect
  `llvm-as` / `opt` / `llc` / `clang`; assemble emitted IR to an
  object behind tool-detection, mirroring `gpu_ci` real-HW dispatch.
- **Stage 202** — control flow: host-IR basic blocks → LLVM IR
  labels, `br`, `phi`.
- **Stage 203** — full scalar op set: comparisons, `select`, `neg`,
  unsigned / narrow-width parity with `x86_64.py`.
- **Stage 204** — memory & aggregates: loads/stores, structs, arrays,
  GEPs (`x86_64.py`'s largest surface — expect sub-stages).
- **Stage 205** — calls & ABI: function calls, calling convention,
  FFI parity.
- **Stage 206** — runtime & intrinsics: panic, bounds checks,
  overflow traps — whatever `x86_64.py` lowers specially.
- **Stage 207** — PARITY GATE. A harness runs every `helixc/tests/`
  program through both the x86_64 and the LLVM path and asserts
  identical observable behaviour.
- **Stage 208** — end-of-Phase-D 5-clean-gate.

### Phase E — MLIR migration
- **Stage 210** — MLIR dependency + dialect-strategy decision.
  Evaluate MLIR Python-binding availability; decide upstream dialects
  vs. a custom Helix dialect vs. hybrid; write the decision record.
- **Stage 211** — Helix MLIR dialect / mapping substrate.
- **Stage 212** — tile-IR → MLIR translation (parallel path).
- **Stage 213** — MLIR → backends: lower MLIR so the 4 GPU backends +
  the Phase-D LLVM backend consume it.
- **Stage 214** — progressive-lowering pass pipeline.
- **Stage 215** — PARITY GATE: MLIR path vs. the home-grown tile-IR
  path, across all backends.
- **Stage 216** — end-of-Phase-E 5-clean-gate.

### Phase F — Unification & cutover
- **Stage 220** — shared backend Protocol/ABC (deferred v2.2
  Item 2): one `Backend` interface across PTX/ROCm/Metal/WebGPU/LLVM.
- **Stage 221** — CUTOVER (destructive; user checkpoint). With parity
  gates 207 + 215 green, retire `x86_64.py` and the home-grown tile-IR
  behind a flag, then remove. Recommend explicit user confirmation
  here even under blanket authority — it is the one irreversible step.
- **Stage 222** — end-of-v3.0 5-clean-gate + tag `v3.0.0`.

## Per-stage audit protocol

Unchanged from v2.x. Each stage closes with a 3-clean audit
(silent-failure-hunter / type-design-analyzer / code-reviewer); any
HIGH or must-fix MEDIUM → fix → re-audit until clean. Each phase and
the v3.0 release close with a 5-clean-gate.

## Test-suite invocation note

The canonical test command is **`pytest helixc/tests/`** — 4031
tests, collects clean. A bare `pytest` from the repo root hits ~51
"import file mismatch" collection errors (it collects modules outside
`helixc/tests/`). v3.0 adds a `pytest.ini` pinning `testpaths` so the
bare command is correct too; the parity harnesses (Stages 207, 215)
depend on a clean, unambiguous full-suite run.

## Stage tracking

| Stage | Title | Ship | Audit | Notes |
|-------|-------|------|-------|-------|
| 200 | LLVM IR emitter substrate | ✓ | 3-clean ✓ | Phase D — CLOSED; see status note |
| 201 | LLVM toolchain detection + dispatch | ✓ | 3-clean ✓ | Phase D — CLOSED |
| 202 | Control flow (blocks, br, phi) | ✓ | 3-clean ✓ | Phase D — CLOSED |
| 203 | Scalar op set (cmp, select, neg, div/mod, bitwise) | ✓ | 3-clean ✓ · cont. audit pending | Phase D — cont. chunk shipped |
| 204–208 | Phase D — LLVM IR backend | — | — | planned |
| 210–216 | Phase E — MLIR migration | — | — | planned |
| 220–222 | Phase F — unification & cutover | — | — | planned |

## Status notes

- 2026-05-20 — v3.0 opened. v2.5.0 released (tag `v2.5.0`). This plan
  drafted as the v3.0 scoping pass; `pytest.ini` testpaths fix shipped
  alongside. Next: Stage 200 — the LLVM IR emitter substrate.
- 2026-05-20 — pre-v3.0 v2.x re-audit gate **CLOSED** (see
  docs/V2_PLAN.md): R1–R8, 6 gate re-runs, 4068 tests green. Phase D
  unpaused.
- 2026-05-20 — **Stage 200 shipped — LLVM IR emitter substrate.**
  `helixc/backend/llvm_ir.py` — an additive textual-LLVM-IR backend
  consuming the same `tir.Module` that `x86_64.py::compile_module_to_
  elf` consumes; scalar core (module triple, `define`, integer
  const/add/sub/mul, `ret`) + `mock_validate_ll`; 19 tests
  (`test_llvm_ir.py`). x86_64.py untouched — purely additive. The
  per-stage 3-clean audit found 1 HIGH (binop emitted the result type
  without checking operand types) + 2 must-fix MEDIUM (function name
  not escaped into the `@` global; `mock_validate_ll` matched `define`
  only at column 0) — all fixed in the same batch (operand-type check;
  `_llvm_global_name` quotes out-of-grammar names; strip-based
  validation). 3-clean re-run dispatched. Deferred to backlog: an
  `Operand` tagged-union refactor (Stage 202), a CONST_INT range check
  (Stage 201's `llc` catches it), `char` dtype width, `nsw`/`nuw`
  overflow-flag parity (a Stage 207 decision). Next: Stage 201 — LLVM
  toolchain detection + dispatch.
- 2026-05-20 — **Stage 200 — 3-clean audit CLOSED.** The per-stage
  audit ran three rounds (silent-failure-hunter / type-design-analyzer
  / code-reviewer). Round 1: 1 HIGH + 2 must-fix MEDIUM → fixed in the
  ship commit (`88a45b0`). Round 2: type-design + code-review CLEAN;
  silent-failure-hunter found 1 must-fix MEDIUM + 1 MEDIUM → fixed
  (`d9adcee`). Round 3 (silent-failure re-confirm): 0 HIGH, 0 must-fix
  (1 MEDIUM + 1 LOW, both non-blocking — `mock_validate_ll` robustness
  on hand-written / future `.ll` that the Stage-200 emitter never
  produces; Stage 201's real `llvm-as` supersedes the mock path). All
  three audit surfaces reached 0 HIGH / 0 must-fix-MEDIUM — the
  3-clean criterion is satisfied; **Stage 200 is CLOSED**. Non-blocking
  backlog carried forward: the round-3 `mock_validate_ll` MEDIUM+LOW,
  plus the round-1 deferrals (Operand tagged-union refactor, CONST_INT
  range check, `char` dtype width, `nsw`/`nuw` overflow parity). Next:
  Stage 201 — LLVM toolchain detection + dispatch.
- 2026-05-20 — **Stage 201 shipped — LLVM toolchain detection +
  dispatch.** New `helixc/backend/llvm_toolchain.py` (a separate
  module, mirroring how `gpu_ci.py` separates dispatch from the
  emitters): `detect_llvm_tools()` finds `llvm-as`/`opt`/`llc`/`clang`
  via `shutil.which`; `dispatch_validate_ll()` always runs the
  toolchain-free `mock_validate_ll`, and when `llvm-as` is present
  assembles the IR for real (`llvm-as` → bitcode, then `llc` → native
  object). gpu_ci dispatch discipline throughout — subprocess timeout
  + OSError captured as findings, a 0-exit-with-no-artifact treated as
  a failure, a frozen tri-state `LLVMDispatchResult`
  (PASSED/FAILED/DEFERRED) whose `__post_init__` makes "fail without a
  diagnostic" unrepresentable. A tool-less machine yields DEFERRED,
  never FAILED, so CI stays green. 13 tests (`test_llvm_toolchain.py`)
  — dispatch orchestration verified deterministically via a
  monkeypatched `subprocess.run`, plus 2 skipif-guarded real-`llvm-as`
  tests. Per-stage 3-clean audit dispatched. Next: Stage 202 — control
  flow.
- 2026-05-20 — **Stage 201 — 3-clean audit CLOSED.** Round 1:
  type-design + code-review both 0 HIGH / 0 must-fix; the
  silent-failure-hunter found 1 must-fix MEDIUM (`real_tool`
  misattributed an `llc`-stage failure to `llvm-as`) → fixed
  (`e66e15e`: `last_tool` tracking so `real_tool` reports the deepest
  tool reached, a `_check_llvm_toolchain_drift` module-load guard, and
  2 llc-leg tests). Round 2 (silent-failure re-confirm): 0 HIGH,
  0 must-fix, 1 LOW (a benign cleanup-only `rmtree(ignore_errors=True)`
  that deliberately matches gpu_ci's four dispatchers). All three
  audit surfaces at 0 HIGH / 0 must-fix-MEDIUM — **Stage 201 CLOSED**.
  Phase-E prep (Stage 210 MLIR dialect-strategy decision record) is
  being drafted in parallel by a background agent. Next: Stages
  202 + 203, batched — control flow + the full scalar op set, one
  3-clean audit for the pair.
- 2026-05-20 — **Stages 202 + 203 shipped (LLVM control flow + scalar
  op set).** Stage 202 (`d7e5aad`): `_FnEmitter` rewritten for
  multi-block — every tir block a labelled LLVM basic block, BR /
  COND_BR → LLVM `br`, tir block-params → `phi` (a pre-pass registers
  every value up front so a loop-header phi can forward-reference a
  back-edge value). Stage 203 (this commit): the six integer
  comparisons → `icmp` (signed or unsigned predicate chosen per
  operand dtype), SELECT → `select i1`, NEG → `sub 0, x`; the unsigned
  integer dtypes (u8/u16/u32/u64/usize) + isize added to the LLVM
  type map. Fail-closed throughout — entry-block, terminator,
  i1-condition, and operand/result type-match guards. 34
  `test_llvm_ir` tests pass; `x86_64.py` untouched. The per-stage
  3-clean audit is dispatched once, batched across both stages. Still
  open in the "full scalar op set": integer division/remainder and
  the bitwise ops (a Stage 203-continuation chunk). Next after the
  audit: Stage 204 — memory & aggregates.
- 2026-05-20 — **Stages 202 + 203 — batched 3-clean audit CLOSED.**
  The batched audit found 1 HIGH + 2 must-fix MEDIUM, fixed in
  `5bf41b6`: `_emit_phis` now type-checks each phi incoming against
  the block parameter and guards against duplicate predecessors;
  `mock_validate_ll` checks "the body's last instruction is a
  terminator" rather than requiring a `ret`, so a valid `ret`-less
  infinite loop passes. The round-2 re-run returned 0 HIGH / 0
  must-fix on all three surfaces — 2 LOW only, both unreachable today
  (a `mock_validate_ll` label-only-empty-block gap the emitter cannot
  produce; a `_compute_predecessors` duplicate-block-id collapse the
  monotonic-id IRBuilder cannot produce) — backlogged. **Stages 202 +
  203 CLOSED.** Still open in the "full scalar op set": integer
  division/remainder + bitwise ops. Next: the Stage 203 continuation
  (div/mod + bitwise), then Stage 204 — memory & aggregates.
- 2026-05-20 — **Stage 203 continuation shipped — LLVM integer
  division/remainder + bitwise ops.** Completes the "full scalar op
  set". DIV / MOD lower to `sdiv`/`srem` (signed) or `udiv`/`urem`
  (unsigned), the form chosen per operand dtype; BIT_AND / BIT_OR /
  BIT_XOR and the left shift SHL lower to the sign-agnostic LLVM
  `and`/`or`/`xor`/`shl`; the right shift SHR lowers to arithmetic
  `ashr` (signed) or logical `lshr` (unsigned); the unary BIT_NOT
  lowers to `xor x, -1`. The DIV/MOD/SHR sign-dependent set lives in a
  new `_LLVM_SIGNED_BINOPS` table; the sign-agnostic set extends
  `_LLVM_SCALAR_BINOPS` and reuses the existing binop branch (one
  arity / type-match guard for all). NEG and BIT_NOT now share one
  unary branch. A Stage-207-parity NOTE records the three deferred UB
  questions (no `nsw`/`nuw`; div-by-zero and `sdiv INT_MIN,-1`;
  over-width shift → poison). 13 new tests; 65 passed + 2 skipped
  across the two LLVM test files; `x86_64.py` untouched. Per-stage
  3-clean audit dispatched. Next after the audit: Stage 204 — memory &
  aggregates.
