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
| 203 | Scalar op set (cmp, select, neg, div/mod, bitwise) | ✓ | 3-clean ✓ | Phase D — CLOSED |
| 204 | Memory & aggregates | ✓ | 3-clean ✓ | Phase D — CLOSED (structs are SSA-bound) |
| 205 | Calls & ABI | ✓ | 3-clean ✓ | Phase D — CLOSED (direct + FFI calls) |
| 206 | Runtime & intrinsics (intrinsic core) | ✓ | A-D 3-clean ✓ | Phase D — CLOSED (core; runtime-op residual → 206-R) |
| 206-R | Runtime-op residual: arena, metaprog, trace, file I/O, print_int | — | — | deferred — additive chunks before the Stage 221 cutover |
| 207 | x86_64-vs-LLVM parity gate | ✓ | A-E 3-clean ✓ | Phase D — CLOSED (mock-path corpus gate + real-execution comparison) |
| 208 | Phase D — end-of-phase 5-clean-gate | ✓ | 5-clean ✓ | Phase D — CLOSED — **PHASE D COMPLETE** |
| 210 | MLIR dependency + dialect-strategy decision | ✓ | review ✓ | Phase E — CLOSED (decision: hybrid `helix` dialect over upstream; eudsl dependency; gpu_ci-style mock path) |
| 211 | Helix MLIR dialect / mapping substrate | ✓ | A-E 3-clean ✓ | Phase E — CLOSED (capability detection; Tensor+Tile op→MLIR mapping; helix-dialect op model; mock_validate_mlir) |
| 212 | tile-IR → MLIR translation (parallel path) | ✓ | A-J 3-clean ✓ | Phase E — CLOSED (the tile-IR→MLIR translator: 17/29 Tile-IR op kinds — types, module/func, arith/cmp/select/vector-tile/layout/func.call/neg/thread-idx; the other 12 fail closed — RESIDUAL async, stub memref/const, attribute-heavy matmul/reduce, index-HBM) |
| 213–216 | Phase E — lowering / pass pipeline / parity / 5-clean-gate | — | — | Phase E — next |
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
- 2026-05-20 — **Stage 203 continuation — 3-clean audit round 1:
  1 must-fix MEDIUM, fixed.** The silent-failure-hunter found that a
  mixed-sign integer binop (e.g. `i32 / u32`) — which the frontend
  type-checker accepts (`typecheck.py`: any two int scalars pass) and
  which collapses to one LLVM type (`i32` and `u32` are both LLVM
  `i32`, so the type-match guard cannot catch it) — had its
  `sdiv`/`udiv` mnemonic chosen silently from operand 0, able to
  diverge from `x86_64.py`. Fixed: a `_require_same_signedness` guard
  fails closed on a mixed signed/unsigned operand pair for the ops
  whose LLVM instruction is *chosen by* signedness — DIV / MOD and the
  four ordered comparisons (the `icmp` branch had the identical latent
  hole) — while shifts (the count's sign is irrelevant) and `eq`/`ne`
  (sign-agnostic) stay permissive. Also addressed both audit LOWs: a
  module-load disjointness assert across the two binop tables
  (type-design), and a stale-comment fix in `tir.py` (code-review —
  SHR is no longer "logical-right unreachable" now that the unsigned
  dtypes exist). 6 new tests; 71 passed + 2 skipped across the two
  LLVM test files. Round-2 re-audit dispatched.
- 2026-05-20 — **Stage 203 continuation — 3-clean audit round 2:
  1 HIGH + 1 MEDIUM, fixed.** The silent-failure-hunter flagged that
  SHR's `ashr`/`lshr` choice keys off the shifted value (operand 0)
  while `x86_64.py` keys off the result type — so a SHR whose value
  and result disagree on signedness would silently diverge. (That
  combination is unreachable from real Helix source — lowering ties a
  shift's result type to its value — but the round-1 discipline says
  fail closed on it regardless.) Fixed: SHR now also calls
  `_require_same_signedness(value, result)`; together with the round-1
  operand-vs-operand checks this makes the signedness-dependent
  mnemonic choice provably equal to `x86_64.py`'s for every TIR the
  LLVM backend accepts. MEDIUM (also a type-design LOW): the
  binop-table disjointness `assert` is `python -O`-strippable —
  replaced with an explicit `_check_binop_table_disjoint()` raise,
  mirroring `llvm_toolchain.py`. 1 new test; 72 passed + 2 skipped
  across the two LLVM test files. Round-3 re-audit dispatched.
- 2026-05-20 — **Stage 203 continuation — 3-clean audit round 3:
  CLEAN. Stage 203 fully CLOSED.** All three audit surfaces
  (silent-failure-hunter / type-design-analyzer / code-reviewer)
  returned 0 HIGH / 0 must-fix-MEDIUM on the re-confirm of the full
  continuation diff. The round-1 mixed-sign DIV/MOD/ordered-comparison
  fix and the round-2 SHR value/result fix are both verified genuinely
  closed; the explicit `_check_binop_table_disjoint()` module-load
  guard runs at import and raises correctly. The signed-vs-unsigned
  mnemonic choice is now provably equal to `x86_64.py`'s for every TIR
  the LLVM backend accepts. 72 passed + 2 skipped across the two LLVM
  test files. The "full scalar op set" is complete — **Stage 203
  CLOSED**. Next: Stage 204 — memory & aggregates (loads/stores,
  structs, arrays).
- 2026-05-20 — **Stage 204 sub-stage A shipped — LLVM mutable local
  variables.** Stage 204 (memory & aggregates) is the largest x86_64
  surface, so it is sub-staged. Sub-stage A: the mutable-local ops
  ALLOC_VAR / LOAD_VAR / STORE_VAR lower to LLVM `alloca` / `load` /
  `store`. Each variable's `alloca` is hoisted to the top of the entry
  block (the LLVM convention — the entry block dominates every use, so
  a LOAD_VAR / STORE_VAR in any block resolves the slot); slot
  pointers are counter-named (`%slot.N`, collision-free with the `%vN`
  value registers) and load/store use opaque pointers (`ptr`). Slots
  are collected and validated in `_prepass` (`_register_alloc_var`);
  LOAD_VAR / STORE_VAR resolve them by name and type-check the
  loaded/stored type against the cell's allocated type. Fail-closed
  throughout — undeclared-variable, duplicate-ALLOC_VAR,
  result-on-ALLOC_VAR, type-mismatch and non-scalar-dtype all raise
  `LLVMEmitError`. 13 new tests; 85 passed + 2 skipped across the two
  LLVM test files. `x86_64.py` untouched. Per-stage 3-clean audit
  dispatched. Next sub-stage: stack arrays (ALLOC_ARRAY / LOAD_ELEM /
  STORE_ELEM → an array-typed `alloca` + GEP).
- 2026-05-20 — **Stage 204 sub-stage A — 3-clean audit CLEAN (round
  1).** All three audit surfaces (silent-failure-hunter /
  type-design-analyzer / code-reviewer) returned 0 HIGH / 0
  must-fix-MEDIUM on the first round: the fail-closed memory-op
  handling, the entry-block `alloca` hoist, the opaque-pointer
  `load`/`store`, and the slot type-checking were all verified sound;
  the type design was rated consistent with the file's conventions.
  The one shared LOW — stale `emit_function` / `emit_module`
  docstrings and the emitted IR header comment still citing old stage
  numbers — is fixed in the closure commit (the supported-op list now
  lives only in the module docstring, the single source of truth, so
  the drift cannot recur). Sub-stage A (mutable locals) is CLOSED.
  Next: Stage 204 sub-stage B — stack arrays (ALLOC_ARRAY /
  LOAD_ELEM / STORE_ELEM).
- 2026-05-20 — **Stage 204 sub-stage B shipped — LLVM stack arrays.**
  The stack-array ops ALLOC_ARRAY / LOAD_ELEM / STORE_ELEM lower to an
  array-typed `alloca` (`[N x T]`, hoisted to the entry block like the
  scalar slots, counter-named `%arr.N`) plus a `getelementptr` for
  each element address — LOAD_ELEM = GEP + `load`, STORE_ELEM = GEP +
  `store`. `_emit_op` now returns a newline-joined block when an op
  lowers to several instructions; `_emit_block` indents each line. The
  GEP omits `inbounds` (the backend does not assume the index is
  bounds-checked — a Stage 207 parity decision) and accepts any
  integer index width. The slot machinery from sub-stage A was
  generalised: a shared `_alloc_op_name` validates ALLOC_VAR /
  ALLOC_ARRAY (with a cross-table duplicate-name check), and a generic
  `_lookup_slot` resolves both var and array references. Fail-closed
  throughout — undeclared array, duplicate / colliding names, wrong
  operand counts, element-type mismatch, non-positive length and
  non-scalar element dtype all raise `LLVMEmitError`. 14 new tests; 99
  passed + 2 skipped across the two LLVM test files. `x86_64.py`
  untouched. Per-stage 3-clean audit dispatched.
- 2026-05-20 — **Stage 204 sub-stage B — 3-clean audit CLEAN (round
  1).** All three audit surfaces returned 0 HIGH / 0 must-fix-MEDIUM
  on the first round, with no LOWs to carry: the `getelementptr`
  semantics (the `[N x T]` / `ptr` / `i64 0` + element-index form,
  mixed index widths, the `inbounds` omission), the newline-joined
  multi-instruction return contract, the slot-machinery refactor and
  the array type-checking were all verified sound; the type-design
  surface was rated correct as-is (the 3-tuple, the generic
  `_lookup_slot`, and the `Optional[str]` multi-line contract all
  endorsed over heavier alternatives). Sub-stage B (stack arrays) is
  CLOSED. Next: assess whether heterogeneous structs need a Stage 204
  sub-stage C — Helix lowers homogeneous aggregates (incl. homogeneous
  structs, whose field access is a LOAD_ELEM at the field index) via
  ALLOC_ARRAY / LOAD_ELEM / STORE_ELEM, already covered — or whether
  Stage 204 closes here; then Stage 205 — calls & ABI.
- 2026-05-20 — **Stage 204 CLOSED + Stage 205 chunk A shipped (LLVM
  direct calls).** Stage 204 assessment: heterogeneous structs need no
  sub-stage C — `lower_ast` binds a heterogeneous aggregate's fields
  as typed SSA values directly (`_bind_aggregate`), emitting no memory
  op; only homogeneous aggregates (incl. homogeneous structs) use
  ALLOC_ARRAY / LOAD_ELEM / STORE_ELEM, already covered by sub-stage
  B. The memory-op surface (ALLOC_VAR/LOAD_VAR/STORE_VAR +
  ALLOC_ARRAY/LOAD_ELEM/STORE_ELEM) is complete — **Stage 204 CLOSED**.
  Stage 205 chunk A: the CALL op lowers to an LLVM `call` — a value
  call `%vN = call <ty> @callee(args)` or a void `call` (a CALL with
  no result, or a unit-typed result, is void — `()` is not a
  materialized LLVM value). Arguments are passed positionally as typed
  operands; the callee name goes through `_llvm_global_name` (quoting
  out-of-grammar names). Direct calls need no `declare` — every Helix
  callee has a `define` in the same module and LLVM textual IR permits
  forward references. `_prepass` now skips registering a unit-typed
  result (no spurious `%vN`). Fail-closed — a missing/empty `target`,
  more than one result, and a non-int result/arg all raise
  `LLVMEmitError`. FFI calls to extern targets (FFI_CALL, which need a
  `declare`) are a later chunk. 10 new tests; 109 passed + 2 skipped
  across the two LLVM test files. `x86_64.py` untouched. Per-stage
  3-clean audit dispatched.
- 2026-05-20 — **Stage 205 chunk A — 3-clean audit CLEAN (round 1).**
  All three audit surfaces returned 0 HIGH / 0 must-fix-MEDIUM on the
  first round: the void-vs-value `call` branch, the `_prepass`
  unit-skip, the no-`declare` forward-reference design, and the
  fail-closed guards were all verified sound; the emitted `call` IR
  was confirmed valid against the LLVM Language Reference. The
  silent-failure-hunter's one non-blocking note — no test for the
  `>1 results` guard (unreachable via `IRBuilder.emit`) — is addressed
  in the closure commit with a raw-`Op` test (110 passed + 2 skipped).
  Chunk A (direct calls) is CLOSED. Next: Stage 205 chunk B — FFI
  calls (FFI_CALL → an LLVM `call` to a `declare`d extern target).
- 2026-05-20 — **Stage 205 chunk B shipped — LLVM FFI calls.** The
  FFI_CALL op (a call to an extern "C" symbol) lowers to the same
  LLVM `call` as a direct CALL, plus a module-scope `declare` for the
  extern target. CALL and FFI_CALL now share one `_emit_call` helper
  (they differ only in the declare); an FFI_CALL additionally calls
  `_register_ffi_declare`, which records `declare <ret> @sym(<args>)`
  and fails closed if the same symbol is called with two different
  signatures. `emit_module` was reworked to construct the
  `_FnEmitter`s directly, collect every function's `ffi_declares`,
  dedup them, and emit the deduped `declare`s at module scope before
  the `define`s — it also rejects an FFI symbol that collides with a
  defined function name (a `declare`/`define` clash `llvm-as` would
  reject). Output is byte-identical to before for any FFI-free
  module. 11 new tests; 121 passed + 2 skipped across the two LLVM
  test files. `x86_64.py` untouched. Per-stage 3-clean audit
  dispatched.
- 2026-05-20 — **Stage 205 chunk B — 3-clean audit CLEAN; Stage 205
  CLOSED.** All three audit surfaces returned 0 HIGH / 0
  must-fix-MEDIUM on the first round: the FFI `declare` collection /
  dedup / conflict-detection, the `_emit_call` CALL+FFI unification
  (CALL behaviour verified unchanged from chunk A), and the
  `emit_module` rework (output verified byte-identical for any
  FFI-free module) were all sound; the `declare` syntax was confirmed
  against the LLVM Language Reference. The two shared LOWs — both
  about `emit_function` now being a single-function fragment that
  `emit_module` no longer routes through — are addressed in the
  closure commit by documenting `emit_function` as a deliberate
  fragment-inspection entry point (no triple, no FFI `declare`; use
  `emit_module` for a complete module). Stage 205's op surface
  (CALL + FFI_CALL) is complete; the scalar-int calling convention is
  LLVM's default `ccc` = System V on the host triple, automatically
  matching `x86_64.py` — **Stage 205 CLOSED**. Next: Stage 206 —
  runtime & intrinsics.
- 2026-05-20 — **Telegram status reporter fixed — `helix_status.py`
  now tracks v3.0 stage progress.** User-reported: every Telegram
  update showed a frozen "Overall toward v3.0: about 93%" and a stale
  "~4013 tests" — only the per-fire note changed. Root cause: the
  reporter tracked the long-finished v2.x build stages (22/22) and
  weighted the in-progress v3.0 version at a flat 0.5, so no
  percentage could move during all of v3.0. Fix: `helix_status.py`
  now carries `V3_STAGES_DONE` / `V3_STAGES_TOTAL` (6/19), a
  `v3_stages_percent()`, and an `overall_percent()` that credits the
  in-progress version its ACTUAL v3.0-stage fraction — overall is now
  an honest 90% that climbs to 100% as stages close; `TESTS_TOTAL`
  refreshed to the real 4194. **Process:** from here, every
  stage-closure commit also bumps `V3_STAGES_DONE`. 7
  `test_helix_status` tests pass — one new test pins that the overall
  % moves with progress (not frozen).
- 2026-05-20 — **Stage 206 chunk A shipped — LLVM Result<T,E>
  packed-tag intrinsics.** Stage 206 (runtime & intrinsics — panic,
  traces, packed representations) is chunked. Chunk A: the
  Result<T,E> ops. A Result is one i64 — tag in the high 32 bits,
  payload in the low 32 (the Stage 49 convention). RESULT_PACK lowers
  to `zext` tag -> `shl 32` -> `or` with the `zext`ed payload (zext
  zero-fills the high half, so it already masks the payload to its
  low 32 bits — no explicit `and`). RESULT_TAG lowers to `lshr 32` +
  `trunc to i32`; RESULT_PAYLOAD to a single `trunc i64 ... to i32`.
  The multi-instruction lowerings use `%vN.tK` temp registers derived
  from the result id (deterministic, collision-free). Fail-closed —
  RESULT_PACK requires i32/i32 operands + an i64 result, RESULT_TAG /
  RESULT_PAYLOAD an i64 operand + an i32 result, all enforced. 10 new
  tests; 131 passed + 2 skipped across the two LLVM test files.
  `x86_64.py` untouched. Per-stage 3-clean audit dispatched. Next
  chunk: TRAP (panic) — needs string globals + a runtime exit.
- 2026-05-20 — **Stage 206 chunk A — 3-clean audit CLEAN (round 1).**
  All three audit surfaces returned 0 HIGH / 0 must-fix-MEDIUM with no
  LOWs: the bit math was verified correct (`zext`/`shl`/`or` and
  `lshr`+`trunc` / `trunc` faithfully implement the packed-tag
  convention; `zext` not `sext`, `lshr` not `ashr` — both confirmed
  the principled choice), the `%vN.tK` temp naming is collision-free
  and deterministic, the type validation is fail-closed, and the
  lowering was checked to compute the same values as `x86_64.py`'s
  RESULT_PACK / RESULT_TAG / RESULT_PAYLOAD (parity holds). Chunk A
  (Result<T,E> intrinsics) is CLOSED. Next: Stage 206 chunk B — TRAP
  (panic), which needs string-literal globals and a runtime exit
  path.
- 2026-05-20 — **Stage 206 chunk B shipped — LLVM TRAP (panic).**
  `panic("msg")` (the TRAP op) lowers to: a `write(2, msg, len)` of
  the `panic[<id>]: <text>` message (newline-terminated) to stderr, a
  `call exit(<id> & 0xFF)`, and `unreachable` — rendered
  byte-identically to x86_64.py's panic so the Stage 207 parity gate
  sees the same stderr + exit code. The message becomes a
  content-addressed private module-scope string constant
  (`@.helix.str.<hash>` — identical messages dedup; a new
  `_llvm_cstring` hex-escapes non-printable / `"` / `\` bytes);
  `write` / `exit` are registered as module-scope `declare`s.
  `emit_module` now also collects + emits the deduped string globals.
  TRAP is registered as a block terminator (it ends in
  `unreachable`). Fail-closed — TRAP with operands, or a non-string
  `text` / non-int `trap_id` attr, all raise. 12 new tests; 143
  passed + 2 skipped across the two LLVM test files. `x86_64.py`
  untouched. Per-stage 3-clean audit dispatched.
- 2026-05-20 — **Stage 206 chunk B — 3-clean audit round 1: 1
  must-fix MEDIUM, fixed.** The silent-failure-hunter found that
  TRAP's i32 result (lower_ast gives every TRAP a result, for SSA
  bookkeeping) was registered as `%vN` by `_prepass` but never
  defined by the TRAP lowering (which ends in `unreachable`).
  Currently harmless — the result is never referenced — but a
  fail-OPEN gap: a future reference would silently emit a dangling
  `%vN` (mock-validate is shape-only; only real `llvm-as` would catch
  it). Fixed: `_prepass` now skips TRAP's results (a `pass` branch,
  like ALLOC_VAR), so a stray reference instead fails closed in
  `_ref` — consistent with the void-CALL unit-result skip. The
  type-design and code-review surfaces were clean. 2 new tests (one
  pins the fail-closed behaviour; one closes the audit's noted
  non-int `trap_id` coverage gap); 145 passed + 2 skipped. Round-2
  re-audit dispatched.
- 2026-05-20 — **Stage 206 chunk B — 3-clean audit round 2: CLEAN.**
  All three audit surfaces returned 0 HIGH / 0 must-fix-MEDIUM on the
  re-confirm: the round-1 fix (`_prepass` skipping TRAP's result) is
  verified to fail closed, and the rest of chunk B (string globals,
  `_llvm_cstring` escaping, the `write`/`exit` declares,
  TRAP-as-terminator, x86_64 parity) re-confirmed sound. Chunk B
  (TRAP / panic) is CLOSED. Next: assess the remaining Stage 206
  surface (TRACE_ENTRY/EXIT, PRINT, the arena ops, STR_BYTE/STR_PTR)
  — ship the remaining-op chunks or close Stage 206 — then Stage 207
  (the x86_64-vs-LLVM parity gate).
- 2026-05-20 — **Stage 206 chunk C shipped — LLVM string-literal
  access (STR_PTR / STR_BYTE).** STR_PTR lowers to `ptrtoint ptr
  @.helix.str.<hash> to i64` — the literal's address as a u64.
  STR_BYTE lowers to a bounds-checked indexed byte load: `icmp ult`
  the index against the real length, `select`-clamp the GEP index to
  0 when out of range, `getelementptr` + `load i8` + `zext to i32`,
  then `select` 0 for the out-of-range case — matching x86_64.py
  (out-of-range yields 0) with NO out-of-bounds read. The
  byte-access global is the literal + one NUL pad, so the clamped
  GEP always lands on a valid byte even for an empty literal. Both
  reuse chunk B's `_register_string` machinery. Fail-closed —
  STR_PTR with operands / a non-i64 result, STR_BYTE with the wrong
  operand count / a non-i32 result, a non-string `text`, all raise.
  10 new tests; 155 passed + 2 skipped across the two LLVM test
  files. `x86_64.py` untouched. Per-stage 3-clean audit dispatched.
- 2026-05-20 — **Stage 206 chunk C — 3-clean audit CLEAN (round 1).**
  All three audit surfaces returned 0 HIGH / 0 must-fix-MEDIUM on the
  first round. The silent-failure-hunter walked every STR_BYTE
  bounds-check case (empty literal, negative index, huge index) and
  confirmed no out-of-bounds read is ever emitted; the code-reviewer
  confirmed exact parity with `x86_64.py`'s STR_BYTE (out-of-range
  yields 0). The one actionable LOW — `_register_string`'s docstring
  still naming only TRAP though STR_PTR / STR_BYTE now also call it —
  is fixed in the closure commit. (A noted `idx_ty`-width edge — a
  narrow non-i32 index with a long literal — is unreachable: string
  indices are always i32 from the frontend, and trusting operand
  types is the file-wide pattern; left as-is.) Chunk C
  (string-literal access) is CLOSED. Next: assess the remaining
  Stage 206 surface — TRACE_ENTRY/EXIT, PRINT, the arena ops, the
  QUOTE/SPLICE/MODIFY family — which need LLVM lowering in Phase D
  vs. deferral, then Stage 207 (the parity gate).
- 2026-05-20 — **Stage 206 chunk D shipped — LLVM string output
  (print_str PRINT).** A PRINT op with the default `print_str` kind
  lowers to `call i64 @write(i32 1, ptr @str, i64 len)` (fd 1 =
  stdout) followed by `trunc i64 ... to i32` — the byte count,
  i32-truncated to PRINT's result, matching x86_64.py (which stores
  `eax`). It reuses chunk B/C's content-addressed string globals and
  the `write` declare. The other PRINT kinds — print_int, write_file,
  read_file_to_arena, trace_event_count — are fail-closed (raise
  `LLVMEmitError`): print_int needs an int-to-ASCII digit loop, the
  file kinds need open/close, all later chunks. Fail-closed also on
  operands, a non-i32 result, a non-string `text`. 8 new tests; 163
  passed + 2 skipped across the two LLVM test files. `x86_64.py`
  untouched. Per-stage 3-clean audit dispatched.
- 2026-05-20 — **Stage 206 chunk D — 3-clean audit CLEAN (round 1).**
  All three audit surfaces returned 0 HIGH / 0 must-fix-MEDIUM on the
  first round: the `_kind` allowlist gate (`!= "print_str"` rejects
  every other kind, including any unknown one), the `write` + `trunc`
  lowering, and exact parity with x86_64.py's print_str (fd 1, the
  byte count i32-truncated) were all verified sound. The one cosmetic
  LOW — the unsupported-kind error message listed only 3 of the 5
  kinds — is fixed in the closure commit (now honestly illustrative).
  Chunk D (string output) is CLOSED. The remaining Stage 206 ops
  (print_int, write_file / read_file_to_arena, TRACE_ENTRY/EXIT, the
  arena ops, the QUOTE/SPLICE/MODIFY family) are deeper runtime
  machinery — each a later chunk before the Stage 221 cutover. Next:
  continue the remaining Stage 206 chunks, then Stage 207 (the
  x86_64-vs-LLVM parity gate).
- 2026-05-20 — **Stage 206 CLOSED (intrinsic core); runtime-op
  residual carved out as 206-R.** Assessment after chunk D: Stage
  206's intrinsic core — the runtime ops ordinary Helix programs use
  (Result<T,E> pack/tag/payload, panic/TRAP, string-literal access,
  string output) — is delivered across chunks A-D, each 3-clean
  audited. The remaining ops `x86_64.py` lowers specially —
  PRINT.print_int, PRINT.write_file / read_file_to_arena,
  TRACE_ENTRY/EXIT, the six ARENA ops, and the QUOTE / SPLICE /
  MODIFY / REFLECT_HASH metaprogramming family — are a distinct,
  deeper body of work: the self-hosting bump allocator, AGI
  metaprogramming cells, the `@trace` ring buffer, and file I/O.
  Several need machinery the additive `_emit_op` backend does not yet
  have — a mutable data section (arena / trace buffer / cells), and
  control flow WITHIN a single op (print_int's digit loop,
  write_file's open/write/close) which the per-block emitter cannot
  express without a one-op-to-many-blocks lowering capability. Rather
  than block the Stage 207 parity-gate milestone behind ~8 more deep
  op-chunks, they are carved out as **206-R** — a tracked residual,
  completed as additive op-coverage chunks before the Stage 221
  cutover. The Stage 207 parity gate covers the implemented op set:
  the LLVM backend fails closed (loudly) on a 206-R op, so a program
  using one is simply outside the parity gate's covered subset, never
  miscompiled. Stage 206 (the numbered stage) is CLOSED. Next:
  Stage 207 — the x86_64-vs-LLVM parity gate.
- 2026-05-20 — **Stage 207 chunk A shipped — x86_64-vs-LLVM mock
  structural-parity harness.** Stage 207 (the parity gate) is chunked,
  mirroring how `gpu_ci.py` rolled out — Stage 129 shipped mock
  validation, v2.4 item 13 added real-HW dispatch. Chunk A:
  `helixc/backend/llvm_parity.py` — `check_parity(module, program)`
  compiles one `tir.Module` through BOTH backends and classifies the
  outcome as a `ParityVerdict`: MATCH (both accept; the LLVM IR passes
  the toolchain-free `mock_validate_ll` shape check), UNCOVERED
  (x86_64 accepts but the LLVM backend fails closed on a 206-R
  residual op — designed, not a defect), MISMATCH (LLVM emits
  shape-malformed IR — a real bug), ERROR (degenerate input, or a
  backend crashed). It needs no LLVM toolchain and always runs; it
  proves the load-bearing invariant that the LLVM backend never
  SILENTLY miscompiles — a 206-R op is rejected loudly (UNCOVERED),
  never mis-emitted. `ParityResult` is a frozen, derived-verdict
  dataclass mirroring `gpu_ci.ValidationResult` /
  `llvm_toolchain.LLVMDispatchResult`, carrying the forward-compatible
  real-execution fields chunk B fills in. The harness deep-copies the
  module per backend (side-effect-free) and captures every backend
  failure — a crash, a fail-closed, a degenerate input — into a
  verdict, never an escaping traceback. 33 tests
  (`test_llvm_parity.py`); `x86_64.py` untouched. Per-stage 3-clean
  audit: round 1 found 1 HIGH (a `mock_validate_ll` crash was
  misclassified MISMATCH not ERROR) + must-fix MEDIUMs (an
  empty/no-`main` module misclassified; missing `__post_init__`
  invariants; the broad-except crash paths untested) — all fixed
  (try/except/else restructure so `llvm_emitted` is set only after the
  shape-check completes; an up-front no-`main` guard; explicit
  invariants for blank `program` / `failed_closed`+`mock_clean` /
  blank diagnostics; monkeypatch crash-path tests). Round 2: all three
  audit surfaces returned 0 HIGH / 0 must-fix-MEDIUM — **Stage 207
  chunk A CLOSED**. Next: Stage 207 chunk B — the real-execution
  parity path (compile both backends to runnable executables, run
  them, compare observable behaviour — exit code / stdout / stderr —
  behind WSL + LLVM-toolchain detection, DEFERRED when absent) + a
  curated source-program corpus + the full parity-gate test.
- 2026-05-20 — **Stage 207 chunk B shipped — the parity corpus + the
  mock-path gate.** Chunk B builds on chunk A's `check_parity`:
  `check_parity_source` runs a Helix SOURCE string through the frontend
  pipeline (parse -> flatten -> monomorphize -> grad-pass -> lower) to
  a `tir.Module` and hands it to `check_parity` — a frontend failure is
  captured as an ERROR result, never re-raised. `PARITY_CORPUS` is a
  curated 28-program corpus of small deterministic Helix programs
  exercising the LLVM backend's covered op surface (integer arithmetic,
  bitwise ops, comparisons / select, control flow, locals, stack
  arrays, calls + recursion, the unsigned dtypes incl. the
  signedness-sensitive udiv / unsigned-icmp paths, bool); a module-load
  guard pins unique non-blank entries. `run_parity_corpus` walks it.
  The **Stage 207 mock-path parity GATE** (`test_parity_corpus_gate`)
  asserts every corpus program is MATCH — real Helix programs
  structurally agree across the x86_64 and LLVM backends, and a covered
  op regressing to UNCOVERED / MISMATCH / ERROR breaks the gate. 40
  tests (`test_llvm_parity.py`); `x86_64.py` untouched. Per-stage
  3-clean audit: round 1 returned 0 HIGH / 0 must-fix-MEDIUM on all
  three surfaces — **Stage 207 chunk B CLOSED**; the cheap LOWs
  (chunk-renumber docstring drift, two unsigned-op corpus entries, an
  `include_stdlib` test) were folded into the closure. Next: Stage 207
  chunk C — the real-execution parity path (compile both backends to
  runnable executables, run them, compare exit code / stdout / stderr
  behind WSL + LLVM-toolchain detection, DEFERRED when absent), which
  closes Stage 207.
- 2026-05-20 — **Stage 207 chunk C shipped — the real-execution
  result model + toolchain detection.** Stage 207's real-execution
  path is split into chunk C (this — the result model + detection,
  toolchain-free and fully verifiable here) and chunk D (the
  compile-link-run-compare dispatch). Chunk C adds: `RealParityStatus`
  (NOT_RUN / DEFERRED / PASS / FAIL — the observable-behaviour
  outcome, the counterpart to the structural `ParityVerdict`) and
  `ParityResult.real_status()` deriving it; a relaxed `ParityResult`
  invariant — chunk A forbade `real_attempted=True, real_passed=None`,
  chunk C admits it as the DEFERRED state (a real run was requested
  but no toolchain could run it), restoring fidelity with
  `gpu_ci.ValidationResult`; and `detect_real_exec_support` /
  `RealExecSupport` — detects whether the real comparison can run here
  (WSL on PATH + `clang` probed INSIDE WSL, since a Windows-PATH clang
  cannot build a Linux executable). A tool-less machine yields
  `can_run_real() == False`, so chunk D's dispatch records DEFERRED,
  never a hard failure — the gpu_ci real-HW dispatch discipline. This
  dev machine: WSL present, no clang in WSL → real parity correctly
  DEFERRED here. 52 tests (`test_llvm_parity.py`); `x86_64.py`
  untouched. Per-stage 3-clean audit: round 1 returned 0 HIGH / 0
  must-fix-MEDIUM on all three surfaces — **Stage 207 chunk C
  CLOSED**; the cheap LOWs (`_probe_wsl_clang` multi-line / non-path
  stdout hardening, two stale chunk-label comments, a `wsl_available`
  clarifying comment) folded into the closure. Next: Stage 207 chunk
  D — the real-execution dispatch (compile both backends to runnable
  executables, run them under WSL, compare exit code / stdout /
  stderr; DEFERRED on a tool-less machine), which closes Stage 207.
- 2026-05-20 — **Stage 207 chunk D shipped — the program-run
  substrate.** Stage 207's real-execution path is split into chunk D
  (this — the program-run substrate) and chunk E (the comparison + the
  `attempt_real` wiring, which closes Stage 207). Chunk D adds
  `_ProgramRun` — a frozen dataclass capturing one backend's build +
  run outcome (`ran` / `exit_code` / `stdout` / `stderr` /
  `findings`); every build-or-run failure is captured into it, never
  raised — and the run helpers `_run_x86_program` (compile the x86_64
  ELF, run it under WSL) and `_run_llvm_program` (emit the LLVM IR,
  `clang`-compile it to an executable inside WSL, run it), plus
  `_win_to_wsl` and a shell-free `_run_under_wsl`. The x86_64 run path
  is verified end-to-end here by real WSL execution (a
  `fn main() { 42 }` program builds and runs with exit code 42); the
  LLVM run path needs `clang` inside WSL (absent here) and is
  mocked-tested. 76 tests (`test_llvm_parity.py`); `x86_64.py`
  untouched. Per-stage 3-clean audit: round 1 — silent-failure found
  a `chmod +x && run` chain that folded a chmod failure into the
  program's exit code (a `ran=True` for a program that never ran),
  fixed by splitting `_run_under_wsl` into two separate shell-free WSL
  calls; round 2 — the same masquerade class was found relocated to
  the run leg (a `wsl`-launcher error code, e.g. 0xFFFFFFFF, folded
  in), fixed with an out-of-0-255-range launcher-failure guard;
  round 3 — all three surfaces returned 0 HIGH / 0 must-fix-MEDIUM.
  **Stage 207 chunk D CLOSED.** Next: Stage 207 chunk E — the
  real-execution comparison (compare the two `_ProgramRun`s for
  observable-behaviour parity) + the `attempt_real` wiring into
  `check_parity` that fills `ParityResult`'s real-* fields, which
  closes Stage 207.
- 2026-05-20 — **Stage 207 chunk E shipped — Stage 207 CLOSED.**
  Chunk E completes the x86_64-vs-LLVM parity gate with the
  real-execution comparison + the `attempt_real` wiring.
  `_compare_runs` compares two `_ProgramRun`s for observable-behaviour
  parity (exit code / stdout / stderr — an EXACT byte comparison,
  sound because both backends emit output through a direct `write`
  syscall, no buffered stdio); `_attempt_real_parity` orchestrates
  detect -> run-both -> compare and fills the `ParityResult` real-*
  fields via `dataclasses.replace`; `check_parity` gained an
  `attempt_real=False` keyword that, on a structural MATCH with the
  toolchain present, runs the real comparison (PASS / FAIL) and is
  DEFERRED when the toolchain is absent. Plumbed through
  `check_parity_source` + `run_parity_corpus`. 88 tests
  (`test_llvm_parity.py`), including a skipif-gated real end-to-end
  test; `x86_64.py` untouched. Per-stage 3-clean audit: round 1
  returned 0 HIGH / 0 must-fix-MEDIUM on all three surfaces — chunk E
  CLOSED. **Stage 207 is CLOSED** — the parity gate stands: the
  mock-path gate proves all 28 corpus programs structurally MATCH
  across both backends, and the real-execution path (DEFERRED on this
  tool-less dev machine, exercised on a WSL+clang runner) compares
  observable behaviour. `V3_STAGES_DONE` bumped 7 -> 8. Next: Stage
  208 — the end-of-Phase-D 5-clean-gate.
- 2026-05-20 — **Stage 208 — end-of-Phase-D 5-clean-gate CLOSED;
  PHASE D COMPLETE.** The end-of-phase gate audited all of Phase D's
  additive work across the five codebase areas (FE / IR / BE / RT /
  TEST). FE + RT: CLEAN — zero files modified in the Phase-D window
  (the additive discipline held — `x86_64.py`, the frontend and the
  stdlib are genuinely untouched). IR: CLEAN — a single doc-comment
  fix in `tir.py`. BE (`helixc/backend/llvm_ir.py`,
  `llvm_toolchain.py`, `llvm_parity.py`): audited holistically across
  silent-failure / type-design / code-review — round 1 found 1 HIGH
  (`emit_module` did not filter `is_extern` / `@kernel` functions the
  way the incumbent `x86_64.py` does, so any `extern fn` program got a
  misleading `LLVMEmitError` the parity gate then mis-classified as
  the non-defect UNCOVERED — masking a real coverage gap); fixed
  (`emit_module` now skips `is_extern` declarations, rejects `@kernel`
  loudly, and the `declare`/`define` collision check excludes extern
  names), with a new `extern_ffi_call` corpus entry added as a
  regression guard. Round 2: 0 HIGH / 0 must-fix-MEDIUM. TEST (the
  three LLVM test files): CLEAN. All five areas clean — **Stage 208
  CLOSED, Phase D (the LLVM IR backend, Stages 200-208) is
  COMPLETE.** `V3_STAGES_DONE` bumped 8 -> 9. Next: Phase E — the
  MLIR migration (Stages 210-216), using
  docs/V3_STAGE210_MLIR_DECISION.md.
- 2026-05-20 — **Stage 210 — MLIR dependency + dialect-strategy
  decision CLOSED; Phase E opened.** The Phase-E scoping draft was
  reviewed by an independent architecture review (which verified the
  op-mapping against the real `tir.py` (98 `OpKind`s) / `tile_ir.py`
  (34 `TileOpKind`s) and confirmed all three recommendations sound),
  finalized, and ratified as `docs/V3_STAGE210_MLIR_DECISION.md`. The
  decision: (1) **Dependency** — the Helix environment is bare `pip` +
  venv (no conda present), so depend on the `llvm/eudsl` find-links
  wheels for `mlir-python-bindings`, pinned to one LLVM major;
  conda-forge is the documented alternative; no build-from-source.
  (2) **Dialect strategy** — HYBRID: represent the ~80-85%
  numerical/structural op core in upstream MLIR dialects
  (`func`/`arith`/`math`/`linalg`/`vector`/`memref`/`gpu`/`nvgpu`),
  and a small custom `helix` dialect (defined via IRDL / Python
  registration, no C++/ODS) for the ~15-20% with no upstream home —
  `grad`/`jvp`/`vmap`, the `agi.*` metaprogramming ops, the atomic
  arena allocator; the `helix` dialect lowers to upstream. (3) **Mock
  path** — reproduce the Stage-201 / `gpu_ci` discipline for MLIR: a
  lazy (never top-level) `import mlir`, a `detect_mlir_python()`
  capability probe, a toolchain-free `mock_validate_mlir`
  shape-checker, and a frozen tri-state PASSED/FAILED/DEFERRED result,
  so a binding-less machine stays green and the home-grown tile-IR
  path remains the reversible fallback until the Stage 221 cutover.
  A decision stage (no compiler code) — the per-stage audit was the
  architecture review. `V3_STAGES_DONE` bumped 9 -> 10. Next: Stage
  211 — the Helix MLIR dialect / mapping substrate.
- 2026-05-20 — **Stage 211 chunk A shipped — the MLIR
  capability-detection substrate.** The first code of Phase E. New
  package `helixc/ir/mlir/`; `toolchain.py` defines `MLIRSupport` (a
  frozen, `__post_init__`-guarded capability result — the structural
  sibling of `llvm_parity.RealExecSupport`) and
  `detect_mlir_support()`, which probes the two real MLIR surfaces
  INDEPENDENTLY: the in-process Python bindings (a LAZY `import mlir`,
  never at module top level — the Stage 210 hard rule) and the
  `mlir-opt` CLI. A partial bindings install (core `mlir.ir` present
  but a required dialect sub-module absent) is reported not-usable,
  not passed-then-failed-later. A `_check_mlir_dialects()` module-load
  guard pins `_REQUIRED_MLIR_DIALECTS`. Mock-path-first: this
  binding-less dev machine yields `is_available()=False`, so Phase E
  defers the real path and the home-grown tile-IR stays the reversible
  fallback. 12 tests (`test_mlir_toolchain.py`); the compiler still
  imports cleanly with no MLIR bindings (the load-bearing guard,
  test-pinned). Per-stage 3-clean audit: round 1 returned 0 HIGH / 0
  must-fix-MEDIUM on all three surfaces; the convergent advisories (a
  module-load drift guard, stronger dialect-loop test coverage) were
  folded into the closure. Next: Stage 211 chunk B — the Helix-op ->
  MLIR-dialect mapping tables.
- 2026-05-20 — **Stage 211 chunk B shipped — the Helix-op → MLIR-
  lowering mapping.** `helixc/ir/mlir/mapping.py` turns the ratified
  Stage 210 decision record's section-2.2 op-mapping table into a code
  data structure. `MLIRLowering` (an `Enum`) names the lowering target:
  the eight upstream MLIR dialects the numerical/structural op core
  maps onto (`arith`/`math`/`linalg`/`tensor`/`memref`/`func`/`cf`/
  `gpu`), the custom `HELIX` dialect, and `RESIDUAL` — an honest
  "undecided" for the ops the decision record explicitly DEFERRED
  ("flag for review", section 2.4: the `Result`/quantize encodings).
  `_OPKIND_LOWERING` maps all **96** `tir.OpKind`s (78 upstream / 13
  helix / 5 residual — ≈81% upstream, matching the decision record's
  80-85%); accessors `mlir_lowering_for` / `is_upstream` /
  `dialect_name` (the last refuses RESIDUAL — it names no dialect — so
  a caller cannot silently format `residual.<op>`). Two module-load
  guards: `_check_lowering_partition` (every `MLIRLowering` is upstream
  / helix / residual) and `_check_opkind_coverage` (`_OPKIND_LOWERING`
  matches `tir.OpKind` EXACTLY — the load-bearing drift guard). Pure
  data — never `import mlir` (mock-path-first; AST-test-pinned). 15
  tests (`test_mlir_mapping.py`). Also re-corrected the decision
  record's op counts: the architecture review recorded 98/34, but the
  coverage guard empirically pins **96 `OpKind` / 29 `TileOpKind`**.
  Per-stage 3-clean audit: round 1 returned 0 HIGH, one MEDIUM (M1 —
  `RESIDUAL` sharing the enum with real dialects was a foot-gun for a
  future Stage-212 caller); fixed by adding the guarded `dialect_name`
  accessor; re-audit of the delta on all three surfaces CLEAN. Next:
  Stage 211 chunk C — the `TileOpKind` (Tile IR) mapping and/or the
  `helix`-dialect op model.
- 2026-05-20 — **Stage 211 chunk C shipped — the Tile-IR op → MLIR-
  lowering mapping.** Extends `mapping.py` with the parallel mapping
  for the Tile IR (the mid-level tiled-GPU IR). New `_TILEOPKIND_
  LOWERING` maps all **29** `tile_ir.TileOpKind`s per the decision
  record's section-2.2 Tile-IR table: tile compute / creation / matmul
  / reduce / layout → the `vector` dialect (MLIR's tile/SIMD layer — a
  new `VECTOR` member added to `MLIRLowering`); tile memory movement →
  `memref`; carried-through scalar ops → `arith`; call / return →
  `func`; GPU primitives → `gpu`. The async memory ops (TMA load /
  store, barrier wait) are RESIDUAL — the decision record (section 3 /
  the section-5 checklist) explicitly defers whether they target
  `nvgpu` (NVIDIA-only) or a cross-backend `helix` async abstraction,
  an open Stage-213 question. 26 / 29 upstream (~90%); no Tile-IR op
  is `helix` (the adjoint table is a transform/pass, not an op). The
  `_check_opkind_coverage` guard was refactored into a generic
  `_check_mapping_coverage` shared by both the Tensor-IR and Tile-IR
  coverage guards; new accessor `mlir_lowering_for_tile`. 21 tests
  (`test_mlir_mapping.py`, +6 Tile-IR). Per-stage 3-clean audit: round
  1 returned 0 HIGH / 0 must-fix-MEDIUM on all three surfaces (the
  LOWs — a redundant test, a type-precision nit — were assessed and
  declined with reason). Next: Stage 211 chunk D — the `helix`-dialect
  op model, then `mock_validate_mlir`, then Stage 211 closes.
- 2026-05-20 — **Stage 211 chunk D shipped — the `helix`-dialect op
  model.** New `helixc/ir/mlir/helix_dialect.py` — the pure-data op
  model of the custom `helix` dialect (decision record section 2.4).
  `HelixOp` (a frozen, `__post_init__`-guarded record) describes each
  op: `mnemonic`, `source_opkind`, `category`, `summary`, and the
  `unsplittable` memory-effect trait. `_HELIX_DIALECT_OPS` enumerates
  all **13** ops in the decision record's three families — the
  transforms (`helix.grad/jvp/vmap`), the AGI metaprogramming ops
  (`helix.quote/splice/modify/reflect_hash`), and the atomic bump
  allocator (`helix.arena_push/get/set/len/push_pair/push_triple`,
  the pair/triple pushes marked `unsplittable`). A module-load guard
  `_check_helix_dialect_model` ties the model to `mapping.py`: the ops
  modelled are EXACTLY the `OpKind`s `mapping` classifies as
  `MLIRLowering.HELIX` — a cross-module drift guard. The SSA operand /
  result / attribute signature is deliberately NOT modelled — it is a
  Stage-212 IRDL-registration concern (the transforms have no
  front-end emit site yet). `helix_dialect_registrability()` is the
  probe-gated registration seam: a frozen `HelixDialectRegistrability`
  result (the registration-seam analogue of `MLIRSupport`) that
  carries the probe's reasons, so a binding-less DEFERRED is never
  silent. Pure data — never `import mlir` (mock-path-first; AST-test-
  pinned). 17 tests (`test_helix_dialect.py`). Per-stage 3-clean
  audit: round 1 returned 0 HIGH, one MEDIUM (the registration gate
  was a bare `bool` — it discarded the probe's "why"); fixed by
  promoting it to the reason-carrying `HelixDialectRegistrability`
  result; delta re-audit on all three surfaces CLEAN. Next: Stage 211
  chunk E — `mock_validate_mlir` (a toolchain-free MLIR-text shape
  checker), then Stage 211 closes.
- 2026-05-20 — **Stage 211 chunk E shipped + Stage 211 CLOSED — the
  toolchain-free MLIR-text validator.** New `helixc/ir/mlir/validate.py`
  — `mock_validate_mlir`, a STRUCTURAL shape check on MLIR textual IR
  (the MLIR analogue of `llvm_ir.mock_validate_ll`), returning a frozen
  tri-state `MLIRValidation`: FAILED on a definite structural defect
  (non-str / empty input, no `module`/`func.func`, an unterminated
  string literal, unbalanced braces / parentheses — counted with
  string literals and `//` comments masked); DEFERRED when the shape
  is clean but real validity is unverified (the honest mock-path
  outcome — never a false PASSED); PASSED reserved for the Stage-212
  real `mlir-opt` validator. Realizes the Stage 210 decision's
  mock-path discipline (section 3): DEFER, never FAIL spuriously,
  never falsely PASS. Pure data — never `import mlir` (AST-test-
  pinned). 19 tests (`test_mlir_validate.py`). Per-stage 3-clean
  audit: round 1 returned one HIGH (a non-str argument raised
  `AttributeError`, contradicting the documented "never raises") and
  one MEDIUM (an unterminated string mis-reported as a brace
  imbalance); fixed (a non-str guard returning FAILED; explicit
  unterminated-string detection that skips the unreliable balance
  checks); delta re-audit on all surfaces CLEAN. One type-design
  MEDIUM — `MLIRValidation` permits a PASSED with a defect-shaped
  finding — was deferred to Stage 212, when the real validator gives
  PASSED a producer and its findings-coherence contract is settled.
  **Stage 211 CLOSED** — its substrate (the `detect_mlir_support`
  capability probe; the Tensor-IR + Tile-IR op→MLIR-lowering mappings;
  the `helix`-dialect op model; `mock_validate_mlir`) is complete and
  every chunk is 3-clean. `V3_STAGES_DONE` → 11. Next: Stage 212 —
  tile-IR → MLIR translation.
- 2026-05-20 — **Stage 212 chunk A shipped — the Helix-IR → MLIR type
  bridge.** New `helixc/ir/mlir/emit.py` opens Stage 212, the parallel
  MLIR translation path. `render_mlir_type` renders every Helix IR
  type as MLIR type syntax: `TIRScalar` → an MLIR scalar (`i32` /
  `f32` / `i1` / ...; integers signless, `isize`/`usize` 64-bit);
  `TIRTensorTy` → `tensor<…>` (a non-constant size dimension becomes a
  dynamic `?`); `TIRTileTy` → `vector<…>` (MLIR's tile/SIMD type —
  static dims required); `TIRTuple` → `tuple<…>`; `TIRUnit` → `none`.
  Dispatch is by exact type through `_TYPE_RENDERERS` / `_DIM_
  RENDERERS` dicts; two module-load coverage guards pin them to
  `tir.TIRType` / `tir.Dim` exactly. FAIL-CLOSED — the translator
  raises `MLIRTranslationError`, never emits a guessed or lossy type,
  on: a width-unpinned `char`, a front-end-only quantized dtype, a
  non-default tensor layout/device, a non-static or 0-d tile
  dimension, an unknown IR type. Pure text — never `import mlir`
  (mock-path-first, AST-test-pinned). 21 tests (`test_mlir_emit.py`).
  Per-stage 3-clean audit: round 1 returned 0 HIGH, one MEDIUM (a
  non-default tensor layout/device was silently dropped — a COL_MAJOR
  tensor rendered identically to ROW_MAJOR); fixed by failing closed
  on it; delta re-audit on all three surfaces CLEAN. Next: Stage 212
  chunk B — the MLIR module / func emitter scaffold.
- 2026-05-20 — **Stage 212 chunk B shipped — the MLIR module / func
  emitter scaffold.** `emit.py` gains `emit_mlir_module(module:
  tile_ir.TileModule) -> str` — it walks a Tile-IR module and emits
  the MLIR `module { func.func @name(%v0: T, ...) -> R { ... } }`
  text, using the chunk-A type bridge for param / return types. The
  translator works from the TILE IR — the plan names Stage 212
  "tile-IR → MLIR" and Stage 215 parity-gates the tile-IR path, so the
  tile IR is the branch point (rationale recorded in the module
  docstring). Chunk B emits SINGLE-BLOCK functions and `func.return`;
  a per-op dispatch table `_OP_EMITTERS` (deliberately partial — only
  `RETURN` so far) FAILS CLOSED on every other op kind. A new
  `_check_fn_translatable` fail-closed-vets each function before
  emission: it rejects a non-identifier name, a zero- or multi-block
  function (multi-block CFG — `cf.br` / `^bb` — is a later chunk;
  emitting `^bb` blocks with no branch would be invalid MLIR), an
  entry block whose params diverge from the signature, and a `return`
  inconsistent with the declared result type. Pure text — never
  `import mlir`; the output is shape-checked by `mock_validate_mlir`.
  33 tests (`test_mlir_emit.py`, +12 chunk B). Per-stage 3-clean
  audit: round 1 returned 0 HIGH, three MEDIUMs (multi-block emitted
  semantically-invalid MLIR; the return-operand type was not
  cross-checked against the signature; entry-block / signature param
  divergence was unchecked) — all fixed via `_check_fn_translatable`;
  delta re-audit on all three surfaces CLEAN. Next: Stage 212 chunk C
  — the per-op emitters (arith / vector / memref / gpu), which will
  make the emitter stateful for SSA value naming.
- 2026-05-20 — **Stage 212 chunk C shipped — the scalar `arith` op
  emitters.** `emit.py`'s `_OP_EMITTERS` table gains five per-op
  emitters: `scalar.const_int` / `const_float` → `arith.constant`, and
  `scalar.add` / `sub` / `mul` → `arith.{add,sub,mul}{i,f}` (the
  integer vs. float mnemonic chosen by the operand type). The emitters
  are STATELESS — every Tile-IR value's SSA name is `%v<id>`, a pure
  function of its id, so a result and its later uses name-match with
  no per-function symbol table. (The chunk-B audit had anticipated a
  stateful emitter, mirroring the LLVM backend's `_FnEmitter` constant-
  inlining map; chunk C establishes that MLIR emits `arith.constant`
  ops rather than inlining, so a constant result has an ordinary
  `%v<id>` name and no state is needed — recorded in the module
  docstring.) Fail-closed throughout: a non-scalar operand, an
  operand/result type mismatch, a bad arity, a non-integer
  `const_int` value, a non-finite `const_float` all raise
  `MLIRTranslationError`. 45 tests (`test_mlir_emit.py`, +12 chunk C).
  Per-stage 3-clean audit: round 1 returned 0 HIGH, one MEDIUM (a
  `const_float` rendered via Python `repr` emitted MLIR-invalid
  literals for infinity / NaN and for round scientific-notation
  magnitudes — `repr(1e20)` is `1e+20`, missing the decimal point
  MLIR's grammar requires); fixed (a `math.isfinite` fail-closed
  guard plus a `_float_literal` helper that always emits a
  decimal-pointed, clean-exponent literal); delta re-audit on all
  three surfaces CLEAN. Next: Stage 212 chunk D — the compare /
  select op emitters.
- 2026-05-20 — **Stage 212 chunk D shipped — the compare / select op
  emitters.** `emit.py`'s `_OP_EMITTERS` gains `scalar.cmp` →
  `arith.cmpi` / `arith.cmpf` and `scalar.select` → `arith.select`.
  The comparison predicate comes from the `cmp` attribute the Tile-IR
  lowerer tags `SCALAR_CMP` ops with; an integer ordered comparison is
  signed (`slt`…) or unsigned (`ult`…) by the Helix operand dtype
  (MLIR integer types are signless, so signedness is read from the
  dtype name — `_UNSIGNED_DTYPES`). A module-load guard
  `_check_cmp_predicate_tables` ties the `_CMPI_PREDICATES` /
  `_CMPF_PREDICATES` tables to `tir.OpKind`'s six `CMP_*` members.
  Fail-closed throughout (operand-type mismatch, non-scalar operand,
  a non-i1 cmp result, an unknown predicate, a non-i1 select
  condition, a select arm/result type mismatch). `scalar.neg` is
  deferred — integer negation has no single MLIR op (it is a
  two-op lowering needing emitter state, a distinct future chunk) —
  and fails closed via the partial dispatch table. 56 tests
  (`test_mlir_emit.py`, +13 chunk D). Per-stage 3-clean audit: round 1
  returned one HIGH — float `!=` was mapped to the ORDERED predicate
  `one`, so `NaN != NaN` would wrongly be false; Helix's reference
  (the x86_64 backend) makes float `!=` unordered-not-equal, so it
  must be `une` — fixed, with the float-predicate table now fully
  test-covered; a type-design MEDIUM (a vestigial dispatch local) also
  fixed; delta re-audit on all three surfaces CLEAN. Next: Stage 212
  chunk E — the `vector` tile-op emitters.
- 2026-05-20 — **Stage 212 chunk E shipped — the elementwise `vector`
  tile-op emitters.** `emit.py`'s `_OP_EMITTERS` gains `tile.add` /
  `sub` / `mul` → `arith.{add,sub,mul}{i,f}` on `vector<...>`-typed
  operands, and `tile.zeros` → a `dense<0>`-splat `arith.constant`.
  MLIR `arith` ops are elementwise-polymorphic over vectors, so a tile
  binop uses the SAME mnemonics as the scalar core — only the type
  classifier differs: chunk C's `_emit_scalar_binop` was generalized
  to `_emit_arith_binop(op, …, classify)`, with `classify` a passed-in
  callable (`_scalar_arith_type` for scalar ops, the new
  `_tile_arith_type` for tile ops). The int / float mnemonic for a
  tile op is the tile's ELEMENT dtype. Fail-closed throughout — a
  non-tile operand on a tile op (and a non-scalar on a scalar op), a
  tile-type mismatch, a bad arity, a `tile.zeros` with operands or a
  non-tile result all raise `MLIRTranslationError`. 64 tests
  (`test_mlir_emit.py`, +8 chunk E; one chunk-B unhandled-op test
  re-pointed from `TILE_ADD` to the still-unhandled `TILE_MATMUL`).
  Per-stage 3-clean audit: all three surfaces CLEAN on round 1 — 0
  HIGH / 0 must-fix-MEDIUM; the `classify`-callable generalization was
  endorsed as the correct abstraction (the operand/result
  type-equality invariant now lives in exactly one place for both op
  families). Next: Stage 212 chunk F — the non-elementwise tile ops
  (`tile.matmul` → `vector.contract`, `reduce` / `transpose` /
  `reshape`, `tile.const`).
- 2026-05-20 — **Stage 212 chunk F shipped — the layout-transform
  tile-op emitters.** `emit.py`'s `_OP_EMITTERS` gains `tile.reshape`
  → `vector.shape_cast` and `tile.transpose` → `vector.transpose
  %src, [1, 0]`. These are the two layout-transform tile ops that can
  be emitted FAITHFULLY without a guessed attribute: `shape_cast`
  needs no attribute (the source / result tile types carry the shape
  change); a 2-D `transpose`'s permutation is unambiguously `[1, 0]`.
  `tile.transpose` is deliberately 2-D-only — an N-D transpose's
  permutation needs an explicit attribute the Tile-IR `TILE_TRANSPOSE`
  op does not carry, so a non-2-D tile fails closed. `tile.const` /
  `matmul` / `reduce` are deferred — each is attribute-heavy (a
  constant value, affine indexing maps, a reduction kind + dims) and
  those conventions are not yet pinned (the ops are stub-status with
  no producer). A `_tile_element_count` helper backs the reshape
  element-count-preservation check. Fail-closed throughout — a
  non-tile operand, an element-count change, an element-dtype change,
  a non-2-D transpose, a wrong transposed result shape all raise
  `MLIRTranslationError`. 73 tests (`test_mlir_emit.py`, +9 chunk F).
  Per-stage 3-clean audit: all three surfaces CLEAN on round 1 — 0
  HIGH / 0 must-fix-MEDIUM; the "handle the unambiguous 2-D case,
  fail closed on the rest" boundary was endorsed as sound. Next:
  Stage 212 chunk G — the `memref` / `gpu` tile ops (the tile load /
  store and GPU-index ops), or `call`.

- 2026-05-20 — **Stage 212 chunk G shipped — the `func.call`
  emitter.** `emit.py`'s `_OP_EMITTERS` gains `call` → `func.call`:
  `%vR = func.call @callee(%args) : (argtypes) -> rettype` for a
  value-returning call, and the void form `func.call @callee(...) :
  (...) -> ()` (no SSA result binding) for a call whose result is
  unit-typed or absent. The callee symbol name is the Tile-IR `target`
  attribute. Fail-closed throughout — a missing / non-identifier
  `target`, or more than one result, raises `MLIRTranslationError`.
  The void-call rule mirrors the sibling LLVM backend's `_emit_call`
  (`llvm_ir.py`): the front end builds every `CALL` with `result_ty`
  set, so a call to a `() -> ()` function yields a one-`TIRUnit`-
  result op — unit is not a materialized MLIR value, so it emits
  `-> ()`, never a dangling `-> none` SSA binding. 82 tests
  (`test_mlir_emit.py`, +9 chunk G). Per-stage 3-clean audit: round 1
  — all three surfaces converged on ONE must-fix (a unit-typed `CALL`
  result emitting `%vR ... -> none`, a dangling SSA name no consumer
  could use); fixed by routing a unit-typed result to the void form;
  a separate reserved-word concern (`@`-prefixed MLIR symbols are a
  distinct namespace, never keywords) was refuted. Re-audit CLEAN on
  all three surfaces — 0 HIGH / 0 must-fix-MEDIUM. Next: Stage 212 —
  the `memref` tile load / store ops, the `gpu` ops (`thread_idx`
  needs a two-op `gpu.thread_id` + `arith.index_cast` lowering), the
  attribute-heavy tile ops (`tile.matmul` / `reduce` / `const`), and
  `scalar.neg`; then close Stage 212 (`V3_STAGES_DONE` → 12).

- 2026-05-21 — **Stage 212 chunk H shipped — the scalar-negation
  emitter.** `emit.py`'s `_OP_EMITTERS` gains `scalar.neg` ->
  negation: a FLOAT operand emits the one-op `arith.negf %x : <fN>`
  (MLIR's dedicated float negate); an INTEGER operand emits the
  canonical TWO-op `arith.constant 0` + `arith.subi %zero, %x` —
  MLIR's `arith` dialect has no integer negate — mirroring the LLVM
  backend's `sub <ty> 0, x`. The integer case is the translator's
  first MULTI-line emitter: the zero constant takes the derived
  `%v<id>.zero` SSA name (collision-free, stateless), and `_emit_fn`
  now splits a multi-line emitter result on newlines and indents each
  fragment. Fail-closed on a non-scalar operand, an operand whose type
  is not the result type, or a wrong operand count. 89 tests in
  `test_mlir_emit.py` (+7 chunk H); 142 MLIR tests pass. Per-stage
  3-clean audit: round 1 found one must-fix — the new multi-line
  `.split` could silently emit a blank body line on an empty fragment;
  fixed by failing closed on an empty fragment. Re-audit CLEAN on all
  three surfaces — 0 HIGH / 0 must-fix-MEDIUM. (A pre-existing
  `_scalar_arith_type` note — Helix `bool` classifies as integer, so a
  `scalar.neg` of a bool emits a valid-but-odd `arith.subi : i1` — is
  tracked as a separate cross-cutting follow-up; the typechecker
  rejects bool negation upstream, so it is unreachable in practice.)
  Next: Stage 212 — the `memref` tile load / store ops, the `gpu` ops,
  and the attribute-heavy tile ops (`tile.matmul` / `reduce` /
  `const`); then close Stage 212 (`V3_STAGES_DONE` → 12).

- 2026-05-21 — **Stage 212 chunk I shipped — the GPU thread-index
  emitter.** `emit.py`'s `_OP_EMITTERS` gains `gpu.thread_idx` -> a
  GPU index read. The Tile-IR `THREAD_IDX` op is the shared carrier
  for three reads, discriminated by its `sreg` attribute — `tid` ->
  `gpu.thread_id` (thread index within the block), `ctaid` ->
  `gpu.block_id` (block index within the grid), `ntid` ->
  `gpu.block_dim` (the block's dimension) — with a `dim` attribute
  (`x`/`y`/`z`) picking the axis. The `gpu` index ops yield MLIR's
  `index` type, so it is a TWO-op lowering: the `gpu` read into a
  `%v<id>.idx` temp, then an `arith.index_cast` to `i32`. The front
  end always tags THREAD_IDX with both attrs (lower_ast.py) and the
  PTX backend requires both, so the emitter requires them too — never
  guessing an axis. Fail-closed on any operand, a non-i32 result, or a
  missing / unrecognised `sreg` or `dim`. 96 tests in
  `test_mlir_emit.py` (+7 chunk I); 149 MLIR tests pass. Per-stage
  3-clean audit: CLEAN on all three surfaces on round 1 — 0 HIGH / 0
  must-fix-MEDIUM; the absence of a module-load drift guard was
  endorsed (there is no `sreg` enum to guard against), and the
  `gpu.*`-op-inside-`func.func` context (vs `gpu.func` / `gpu.module`)
  was confirmed reasonable for a partial op-by-op translator (upstream
  MLIR: intrinsic-wrapping `gpu` ops do not require a `gpu.func`
  parent). Next: Stage 212 — the `memref` tile load / store ops and
  the attribute-heavy tile ops (`tile.matmul` / `reduce` / `const`);
  then close Stage 212 (`V3_STAGES_DONE` → 12).

- 2026-05-21 — **Stage 212 chunk J + CLOSE — the SSA-definedness
  validator; Stage 212 CLOSED.** A holistic stage-close audit of the
  whole `emit.py` translator (all three reviewers on the complete
  file) found two HIGH silent-failure gaps: `_check_fn_translatable`
  never validated SSA definedness, so a use-before-def emitted a
  dangling `%v<id>` reference and a duplicated value id a redefined
  one — invalid MLIR the structural mock-validator does not catch.
  Fixed (chunk J): `_check_fn_translatable` gains a single-block SSA
  definedness + uniqueness pass — seed `defined` from the parameter
  ids, then per op require every operand already defined and every
  result fresh. 99 tests in `test_mlir_emit.py` (+3 chunk J); 152
  MLIR tests pass. 3-clean re-audit CLEAN on all three surfaces. The
  audit's one must-fix-MEDIUM — `_emit_op` should return `list[str]`
  rather than a multi-line `str` — is genuine and is scheduled as the
  first item of the known-issues cleanup phase that follows.

  **STAGE 212 CLOSED — `V3_STAGES_DONE` → 12.** The tile-IR → MLIR
  translator faithfully renders every Helix IR type, the module /
  function structure, and 17 of the 29 Tile-IR op kinds (the scalar
  `arith` core, compare / select, the elementwise and
  layout-transform `vector` tile ops, `func.call`, `scalar.neg`, the
  GPU thread-index read). The other 12 op kinds fail closed by
  deliberate, documented design: the async TMA / barrier ops are
  RESIDUAL (a Stage-213 decision); the `memref` memory-movement ops
  and `tile.const` are stub-status with no defined signature;
  `tile.matmul` / `reduce` are attribute-heavy and need MLIR-encoding
  design; the `tile.index_load/store_hbm` ops need a memref type
  bridge plus parameter-name resolution — all land at Stage 213+. Per
  the 2026-05-21 user directive, the loop now runs the known-issues
  cleanup phase, then STOPS — it does not advance to Stage 213.

- 2026-05-21 — **v3.0 known-issues cleanup phase COMPLETE — autonomous
  loop stopped.** Per the 2026-05-21 user directive (finish Stage 212,
  then a known-issues cleanup, then stop), the cleanup phase ran four
  fixes — each per-stage 3-clean audited, committed, and pushed:
  - (a) `90b0b1f` — the "hung" full test suite was diagnosed as NOT
    deadlocked: a large ~4500-test integration suite (the earlier
    "40-min hang" was a `| tail`-buffering misread, not a deadlock).
    Its dominant performance bug — a ~600 ms uncached stdlib re-parse
    on every `parse(include_stdlib=True)`, paid by thousands of
    compile-and-run tests — was fixed with a `pickle`-blob cache
    (~7x faster parse).
  - (b) `955e50e` — `emit._scalar_arith_type` now fails closed on a
    `bool` operand: booleans are not an arithmetic domain, and it had
    been emitting a meaningless `arith.*i : i1`.
  - (c) `ff64d65` — `validate.MLIRValidation` now enforces a total
    findings contract — a PASSED carries NO findings, FAILED /
    DEFERRED carry at least one — settling the Stage-211 chunk-E
    coherence carry-over.
  - (d) `b9da3bb` — the MLIR op emitters return `list[str]` instead of
    a `\n`-joined `str`, removing the join/split round-trip (the
    Stage-212 stage-close type-design must-fix).
  STATE AT STOP: v2.0–v2.5 released; v3.0 Phase D complete (Stages
  200–208); Phase E Stages 210, 211, 212 closed — `V3_STAGES_DONE` =
  12 of 19. The tile-IR → MLIR translator (`helixc/ir/mlir/emit.py`)
  faithfully renders every IR type and 17 of 29 Tile-IR op kinds; the
  rest fail closed by documented design. REMAINING for a future run:
  Stage 212's deferred attribute-heavy / `memref` / async op emitters,
  Stages 213–216 (MLIR → backends, the progressive-lowering pass
  pipeline, the MLIR-vs-tile-IR parity gate, the end-of-Phase-E
  5-clean-gate), Phase F 220–222 (backend unification + cutover + the
  v3.0.0 release), and the 206-R residual LLVM-lowering ops. The
  autonomous build loop stops here, per the directive.
