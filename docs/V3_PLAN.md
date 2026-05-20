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
| 206 | Runtime & intrinsics (chunked) | chunk A,B,C ✓ | A,B,C 3-clean ✓ | Phase D — Result/panic/string CLOSED |
| 207–208 | Phase D — LLVM IR backend | — | — | planned |
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
