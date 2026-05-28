# 5-axis end-of-phase audit protocol (K2.AG, 2026-05-28)

The K-bootstrap stop criterion is "Python-ready-to-delete state + 5
consecutive clean END-OF-PHASE audits." Each end-of-phase audit is
a 5-axis sweep (FE / IR / BE / RT / TEST). Per the original loop-prompt
phrasing: "All 8 axes HIGH-confidence clean. Repeat 5x across separate
ticks. ANY HIGH or must-fix MEDIUM resets the counter to 0."

(8 axes = 5 phase + 3 holistic — silent-failure-hunter,
type-design-analyzer, code-reviewer applied across phases.)

## Speed-up rationale (K2.AG)

Naive serialization: 5 sweeps × 5 phase-axes × ~5-10 min/subagent =
**125-250 min** = ~25 ticks at 5-10 min each. Each subagent runs in
its own Agent call sequentially.

Parallelized via Agent-in-single-message: 5 phase-axes dispatched
concurrently in ONE message per sweep = 5 sweeps × **5-10 min** = 25-50
min = **5 ticks total**. Each tick is one full 5-axis sweep.

**Net saving: ~20 ticks (~3-4 hours wall-clock) at the stop-criterion
gate.** Worth building the dispatch template now so it's ready when
the gate triggers.

## Per-axis subagent dispatch templates

Each axis gets a `pr-review-toolkit:*` subagent with a phase-specific
focus. Templates below assume the subagent reads only the files in
its phase to avoid context bloat.

### Axis 1 — FE (Frontend: lexer + parser)

**Subagent**: `pr-review-toolkit:silent-failure-hunter`
**Files**: `helixc/bootstrap/lexer.hx`, `helixc/bootstrap/parser.hx`
**Focus prompts**:
- Token-tag stability across versions
- parse_primary cascade ordering (post-K1.F31-F52 macro family)
- Brace-cascade depth tracking (currently 38 closers per K1.F51-F52 audit)
- K3.S reserved-kw reject coverage on all IDENT operands
- `is_assert_*_form` detectors mutual disjointness (see partition table
  comment block at the assert family)
- Lifetime-annotation skip (K1.CQ) and ref-of-T (K1.DR/DU/DV/F5h) coverage
- Closure / generic-fn registration in sb scratch slots

Cap report at ~400 words. HIGH/MEDIUM/LOW + file:line + 1-sentence cause.

### Axis 2 — IR (tile_ir, mlir-emit, lowering)

**Subagent**: `pr-review-toolkit:type-design-analyzer`
**Files**: `helixc/ir/tile_ir.py`, `helixc/ir/mlir/*.py`, `helixc/lower/*.py`
**Focus prompts**:
- TileIR op-set completeness vs the bootstrap's __tile_zeros/add/sub/mul/matmul (K1.F23c-F27)
- mlir/emit.py text-emission correctness (Stage 212 chunks A-F shipped)
- mlir/validate.py tri-state (PASSED/FAILED/DEFERRED) consistency
- mlir/mapping.py TileIR-to-MLIR-dialect mapping completeness
- Lowering passes order (parse → grad_pass → lower → validate → optimize → codegen)

Cap report at ~400 words.

### Axis 3 — BE (codegen: x86_64 + GPU + MLIR backends)

**Subagent**: `pr-review-toolkit:code-reviewer`
**Files**: `helixc/bootstrap/kovc.hx`, `helixc/backend/*.py`,
`helixc/ir/mlir/toolchain.py`
**Focus prompts**:
- AST_CALL codegen rax-vs-eax discipline (relevant to the K1.F5g2
  struct-return blocker; trap-id capture via K2.AF helper)
- bn_state slot allocations (currently slots 174-178 for tile ops)
- patch-table arena_base displacement patching (K3.A audit-fix)
- Mixed-type binop widening guards (K3.B audit-fix; K1.F8/F8b/F8c/F8d closures)
- Backend dispatch (Stage 220 Backend Protocol)

Cap report at ~400 words.

### Axis 4 — RT (Runtime: arena + trace + panic + exit)

**Subagent**: `pr-review-toolkit:silent-failure-hunter`
**Files**: arena/trace/panic emit helpers in `helixc/bootstrap/kovc.hx`
**Focus prompts**:
- Arena bounds-check guards on tile-op read/write (K3.U/V/W audit closure)
- Trace event ring-buffer depth-1 (K1.F20b)
- Panic message format match with Python helixc (K1.AE/AH/AI)
- Exit-code u8 truncation discipline (K1.F26 finding)
- POSIX sys_exit handling

Cap report at ~400 words.

### Axis 5 — TEST (Test coverage + parity harness)

**Subagent**: `pr-review-toolkit:code-reviewer`
**Files**: `helixc/tests/test_codegen.py`, `helixc/tests/test_k2_parity.py`,
`docs/K_BOOTSTRAP_FEATURE_MATRIX.md`
**Focus prompts**:
- K2_CORPUS size and category coverage
- bootstrap-only-superset items correctly excluded from K2_CORPUS
- Macro-family / tile-op / impl-block-self / generic-fn structural findings
  reflected in matrix (FUNCTIONAL PARITY vs KOVC-MISSING)
- test_codegen.py per-chunk K1.F* coverage
- Counter discipline: K_BOOTSTRAP_CHUNKS_DONE bumps every commit
- Telegram policy: milestone / 5-chunk / 60-min thresholds

Cap report at ~400 words.

## Dispatch convention

When the loop reaches the stop-criterion gate, dispatch all 5 axes in
ONE Agent block (single message, 5 tool calls). Wait for all 5 to
return. Synthesize verdict: "all 8 axes HIGH-confidence clean" or
"reset counter to 0 on first HIGH/must-fix-MEDIUM finding."

If clean: record audit-clean signal, increment the consecutive-counter,
ScheduleWakeup for the next sweep. If counter reaches 5, send the
FINAL Telegram with the 5-clean summary attached and OMIT the
ScheduleWakeup — loop terminates per the stop-criterion.

If not-clean: address findings, re-sweep next tick. Counter goes back
to 0.

## Estimated stop-gate timeline

Once Python-ready-to-delete state is reached (currently ~150-235 chunks
remaining):
- Best case (all 5 sweeps clean first try): **5 ticks × 5 min = 25 min**.
- Realistic (1-2 sweeps find HIGH findings, fix-tick between):
  **8-12 ticks = 40-90 min**.
- Worst case (multiple findings reset counter): up to ~30 ticks.

This protocol caps the worst case meaningfully and makes the best case
fast. Without it: even the best case is 25 ticks = ~2 hours.
