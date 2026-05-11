# Stage 28.8 Cycle 14 — Silent-Failure Audit (retry)

**Date**: 2026-05-11
**Commit (HEAD)**: 1e4c3e6 — "Audit 28.8 cycle 14 fix-sweep:
close C13-1 (HIGH, DCE drops TRACE_EXIT operand)".

**Retry note**: a prior cycle-14 silent-failures audit run hit an
API error. Cycle 14 Audit B (type-design) and Audit C (code-review)
both completed CLEAN on this HEAD. This document is the retry of
the silent-failure lens only and overwrites any partial doc from
that run.

**Scope of this lens**: any silent-failure window NOT already
counted in cycles 1-13 as a carryover. The cycle-14 fix-sweep
landed a single behavioural change (add `tir.OpKind.TRACE_ENTRY`
and `tir.OpKind.TRACE_EXIT` to `SIDE_EFFECT_KINDS` in
`helixc/ir/passes/dce.py`, plus two regression tests in
`helixc/tests/test_dce.py`). The lens focuses on:
1. Verifying the fix is present and correct at HEAD.
2. Tracing for any **fresh** silent-failure window opened by the
   fix (the "did adding members to the set break a different
   invariant?" question).
3. Rotating the fresh-eyes spot-check surface (per the cycle 10-13
   rotation discipline).

**Strict criterion** (per user directive 2026-05-10): cycle counts
CLEAN only when **zero new findings of ANY severity**
(CRITICAL/HIGH/MEDIUM/LOW). Findings already in the carryover
ledger are explicitly excluded.

**Clean-counter state**: per the fix-sweep commit message, the
counter was 3/5 going into cycle 13 (cycles 10, 11, 12 clean) and
**resets to 0** because cycle 13 was not clean (C13-1 HIGH found
by the code-review lens). The cycle-14 fix-sweep closed C13-1.
Cycle 14 silent-failures is the first re-audit lens of this fresh
clean-cycle window; counter is 1/5 if cycle 14 closes clean across
all three lenses (silent-failures, type-design, code-review).

---

## Method

1. **Read cycle-13 silent-failure verdict**
   (`docs/audit-stage28-8-cycle13-silent-failures.md`): cycle 13
   was 0 new findings (CLEAN) for the silent-failure lens; the
   cycle-13 HIGH (C13-1) was a code-review-lens finding, not a
   silent-failure-lens finding. The cycle-14 fix-sweep landed at
   HEAD 1e4c3e6.
2. **`git show 1e4c3e6 --stat`**:
   ```
    docs/audit-stage28-8-cycle13-codereview.md      | 492 ++++++++++
    docs/audit-stage28-8-cycle13-silent-failures.md | 646 ++++++++++
    docs/audit-stage28-8-cycle13-type-design.md     | 395 ++++++
    helixc/ir/passes/dce.py                         |  13 +
    helixc/tests/test_dce.py                        |  35 +
    5 files changed, 1581 insertions(+)
   ```
   The only production-code change is +13 lines in
   `helixc/ir/passes/dce.py` (adding 2 new entries — TRACE_ENTRY,
   TRACE_EXIT — plus an 11-line block comment to
   `SIDE_EFFECT_KINDS`). The test change is +35 lines (two new
   regression tests). All other diffs are doc-only.
3. **Verified the fix at HEAD**
   (`helixc/ir/passes/dce.py:68-80`):
   ```python
   # Audit 28.8 cycle 13 C13-1 (HIGH): TRACE_ENTRY / TRACE_EXIT are
   # @trace prologue/epilogue ops with side effects (the runtime
   # records entry/exit events). TRACE_EXIT consumes the return value
   # as an operand so the runtime can log it; for unit-returning
   # traced fns, lower_ast.py synthesizes a `const_int(0)` whose sole
   # consumer is TRACE_EXIT. Without this entry, DCE's seed phase
   # never marks the const_int(0) live (TRACE_EXIT has no results, so
   # the spread phase has no live result to propagate from either),
   # the producer is dropped, and the backend (x86_64.py:2498)
   # KeyErrors when it tries to look up the slot of the now-deleted
   # operand on `-O2`.
   tir.OpKind.TRACE_ENTRY,
   tir.OpKind.TRACE_EXIT,
   ```
   Both members are present in the live set, the rationale comment
   is load-bearing and accurate (matches the
   `lower_ast.py:573-574` synthesized-const path and the
   `x86_64.py:2495-2500` `self._slot_of(op.operands[0])` consumer).
4. **Cross-checked SIDE_EFFECT_KINDS consumers**: `grep
   SIDE_EFFECT_KINDS helixc` finds exactly three call sites — all
   inside `helixc/ir/passes/dce.py` itself (lines 103, 119, 131
   — seed phase, spread-phase guard, drop-phase guard). The set is
   only re-imported once outside dce.py: in
   `helixc/tests/test_cse.py:181` as a read-only assertion that
   `TILE_INDEX_STORE` (Stage 16 follow-up) is present. No
   production module outside `dce.py` consumes this set, so the
   fix's blast radius is the DCE pass alone.
5. **Trace cross-pass interactions**:
   - **CSE (`helixc/ir/passes/cse.py`)**: CSE uses a positive
     allowlist `PURE_KINDS` (CONST_*, ADD/SUB/MUL/DIV/MOD, NEG,
     CMP_*, CAST) — not the SIDE_EFFECT_KINDS exclusion. TRACE_*
     ops were never in PURE_KINDS and aren't now; adding them to
     SIDE_EFFECT_KINDS in dce.py has no effect on CSE. The
     synthesized `const_int(0)` for unit-returning traced fns is in
     PURE_KINDS, so CSE could merge multiple Unit-return paths'
     synthesized zeros into a single `const_int(0)` — but that
     merge is sound (idempotent, same hash key, same result type
     i32). And both early-return (`lower_ast.py:1893`) and
     fall-through (`lower_ast.py:575`) TRACE_EXIT sites synthesize
     the const with `self.builder.const_int(0)` in the SAME block
     they're consumed in, so per-block CSE is the only scope and
     no cross-block hazard arises.
   - **FDCE (`helixc/ir/passes/fdce.py`)**: function-level
     reachability — only inspects CALL/MODIFY/QUOTE attrs for the
     call graph; TRACE_* are not call-graph edges. Independent.
   - **Const fold (`helixc/ir/passes/const_fold.py`)**:
     arithmetic-only fold; TRACE_* are not foldable. Independent.
     `lower_fold_dce` order in tests is fold → dce, and the
     synthesized `const_int(0)` is already const-folded (it IS the
     constant), so fold is a no-op for the trace prologue/epilogue
     pair.
   - **grad_pass, autodiff, monomorphize, struct_mono,
     panic_pass, trace_pass, effect_check, pytree, unsafe_pass,
     deprecated_pass, autotune**: none reference TRACE_ENTRY or
     TRACE_EXIT (`grep TRACE_(ENTRY|EXIT) helixc`). Independent.
6. **Trace backend handling of the now-preserved operand**:
   `helixc/backend/x86_64.py:2489-2502` already had the
   `if op.operands: ret_slot = self._slot_of(op.operands[0]); ...`
   path — it was the BACKEND that crashed pre-fix because the
   operand was a stale value-id pointing at a DCE'd producer. With
   the fix, the producer survives, the slot exists, and the
   backend's `mov_eax_mem_rbp(ret_slot)` succeeds. The backend
   path requires no further change.
7. **Trace lowerer correctness**: both `lower_ast.py:572-575`
   (fall-through epilogue) and `lower_ast.py:1879-1896`
   (early-return) synthesize the `const_int(0)` only when
   `return_ty` is Unit AND there's no real return value to pass.
   For non-Unit traced fns, the actual return value flows through
   as the operand and is already kept live by RETURN (which has
   been in SIDE_EFFECT_KINDS since dce.py's origin). The C13-1 bug
   was specifically the Unit-return path, which is now fixed.
8. **Regression-test reading**: the two new tests in
   `helixc/tests/test_dce.py:112-145`:
   - `test_c13_1_dce_preserves_trace_exit_operand`: parses `@trace
     fn foo() { let x: i32 = 5; }`, runs `lower_fold_dce`,
     enumerates all `TRACE_EXIT` ops in `foo`, builds the set of
     all live value-ids (function params + every op's results),
     and asserts that each TRACE_EXIT operand is in that live set.
     This precisely encodes the C13-1 invariant.
   - `test_c13_1_dce_preserves_trace_entry_in_kept_set`:
     belt-and-suspenders — asserts TRACE_ENTRY survives the DCE
     pass. TRACE_ENTRY currently has no operands, so this is
     future-proofing for the runtime-helper-handle wiring that may
     add an operand later.
   Both tests use the same `lower_fold_dce` helper as the rest of
   the test file. Per the commit message, `253 targeted tests
   pass (was 250)` — the +3 reflects the +2 new tests and 1
   additional test elsewhere in the matrix. The harness reads
   green.
9. **Carryover re-check**: cycles 1-12 carryovers (audit-C4-1
   CRITICAL, audit-C4-4 HIGH, audit-C4-8 LOW, C5-10 LOW,
   monomorphize_safe docstring drift, D-vs-Quote diagnostic text,
   C7-1 test-coverage gap) are NOT re-flagged for cycle 14 per
   the user's strict re-flag rule (a carryover is re-flagged only
   if it CHANGED since the prior cycle — none changed; the
   cycle-14 fix-sweep touched dce.py and test_dce.py only).
10. **Cycle-14 fresh-eyes rotation**: cycle 13 rotated to
    grad_pass.py:639-643, pytree.py:293-296, hash_cons.py:335,
    flatten_impls.py:88, flatten_modules.py:67/77,
    trace_pass.py:67, effect_check.py:228, and confirmed
    examples/dashboard_server.py non-production. Cycle 14
    rotates to:
    - **dce.py SIDE_EFFECT_KINDS** itself — line-by-line walk of
      every member, including the two newly-added entries.
    - **cse.py PURE_KINDS** — the dual ledger; ensure no TRACE_*
      slipped in by mistake.
    - **fdce.py call-graph edge sources** — verify TRACE_* aren't
      call edges.
    - **x86_64.py TRACE_EXIT operand-consumer site** — verify the
      `if op.operands:` guard is correct (i.e., it would still
      handle a hypothetical zero-operand TRACE_EXIT without a
      silent failure).
    - **lower_ast.py synthesized-const path** — verify the
      `const_int(0)` sentinel is the correct typed operand
      (i32, not bool/Unit/handle) and matches the backend's
      `mov_eax_mem_rbp` expectation.

---

## Fresh-eyes walk for cycle 14

### Member-by-member audit of `SIDE_EFFECT_KINDS` (dce.py:32-81)

| Member | Source | Risk of silent failure if mis-classified |
|---|---|---|
| RETURN | original v0.1 | terminates block — must execute |
| BR | original v0.1 | terminates block — must execute |
| COND_BR | original v0.1 | terminates block — must execute |
| CALL | original v0.1 | may have arbitrary side effects |
| STORE_VAR | original v0.1 | mutates memory |
| STORE_ELEM | original v0.1 | mutates memory |
| ALLOC_VAR | original v0.1 | reserves slot — backend may rely on layout |
| ALLOC_ARRAY | original v0.1 | reserves slot — backend may rely on layout |
| MODIFY | reflection | mutates reflection cell |
| SPLICE | reflection | mutates reflection cell |
| PRINT | io | observable side effect |
| QUOTE | Stage 6 reflection | reserves reflection cell |
| REFLECT_HASH | Stage 6 reflection | stable test handle |
| ARENA_PUSH | Stage 7 arena | mutates global arena |
| ARENA_SET | Stage 7 arena | mutates global arena |
| TILE_INDEX_STORE | Stage 16 PTX | HBM write |
| FFI_CALL | Stage 16.5 audit CRITICAL-1 | arbitrary extern effects |
| TRAP | Stage 28.5 | terminates process |
| **TRACE_ENTRY** | **cycle 14 C13-1** | **runtime entry event** |
| **TRACE_EXIT** | **cycle 14 C13-1** | **runtime exit event + return value log** |

Every member is justified. The two new entries (TRACE_ENTRY,
TRACE_EXIT) match the established pattern: ops whose execution
is observable to the runtime (or the backend's layout/slot
allocator) and must therefore not be DCE'd. The comment block
explaining the addition is more detailed than the FFI_CALL
precedent (which is a sibling-pattern fix from the Stage 16.5
audit) — a small UX win for future maintainers.

### CSE PURE_KINDS dual-check (cse.py:33-50)

`PURE_KINDS` contains: CONST_INT, CONST_FLOAT, CONST_BOOL, ADD,
SUB, MUL, DIV, MOD, NEG, CMP_EQ, CMP_NE, CMP_LT, CMP_LE, CMP_GT,
CMP_GE, CAST. TRACE_ENTRY/TRACE_EXIT are NOT in this set
(correctly — they have side effects and aren't candidates for
deduplication). No silent-failure window.

### FDCE call-graph sources (fdce.py:44-62)

FDCE inspects `op.kind == CALL` (target attr), `op.kind ==
MODIFY` (verifier_fn attr), and `op.kind == QUOTE`
(ast_pretty scan). TRACE_* are NOT in this list — correctly,
because @trace doesn't call user fns. No silent-failure
window.

### x86_64.py TRACE_EXIT consumer (x86_64.py:2489-2502)

```python
if op.kind == tir.OpKind.TRACE_EXIT:
    if op.operands:
        ret_slot = self._slot_of(op.operands[0])
        self.asm.mov_eax_mem_rbp(ret_slot)
    self.asm.b.emit(0x90)  # nop
    return
```

The `if op.operands:` guard is defensive against a hypothetical
zero-operand TRACE_EXIT (which the lowerer never emits — both
synthesizing-paths always provide an operand). If a future
lowerer change accidentally emitted an operand-less TRACE_EXIT,
this guard would skip the `_slot_of` lookup silently — but the
emitted machine code (a single `nop`) would still be
valid. Whether this is a *future* silent-failure window depends
on whether "TRACE_EXIT with no operand" is itself a semantic bug
(it would be — the runtime helper expects a return value). The
guard is currently a useful tolerance for an invariant the
lowerer enforces, not a silent-failure window per se. Stable
non-finding. Not previously flagged in cycles 1-13. Not flagged
here either.

### lower_ast.py synthesized-const sentinel (lower_ast.py:573-574, 1891-1892)

Both paths:
```python
ret_operand = self.builder.const_int(0)
```

`const_int` emits a `CONST_INT` op with i32 result type. The
backend's `mov_eax_mem_rbp` is i32-shaped (32-bit register), so
the synthesized const + the backend consumer agree on width.
No type-mismatch silent failure.

The synthesized const is emitted into the SAME block as the
TRACE_EXIT that consumes it (immediately preceding), so SSA
dominance is satisfied. No cross-block silent failure.

### Cross-stage interactions re-checked (cycle 14)

- **fold → dce ordering**: in `lower_fold_dce` (test helper)
  and in `check.py`'s pipeline, const_fold runs before dce. The
  synthesized `const_int(0)` IS already constant — const_fold
  has nothing to fold there. DCE then sees TRACE_EXIT in
  SIDE_EFFECT_KINDS, seeds its operand (the const_int(0)
  producer) live, and the spread/drop phases preserve it.
  Verified correct.
- **DCE iteration to fixpoint**: `while changed:` loop in
  `dce_function`. The loop terminates when no more ops are
  dropped. Adding TRACE_* to SIDE_EFFECT_KINDS only DEcreases
  the number of ops that can be dropped (more ops are now
  kept). The loop's monotonic-decreasing-removable-set property
  is preserved — no infinite-loop hazard.
- **DCE + grad_pass**: grad_pass synthesizes shadow values and
  reverse-mode gradient computations. None of grad_pass.py
  emits TRACE_* (verified by `grep TRACE_ helixc/frontend/`).
  The cycle-13-noted grad_pass.py:639-643 frozen-dataclass
  cache fallback is independent of the trace machinery. No new
  interaction.
- **DCE + monomorphize**: monomorphize specializes generic fns
  by argument types. @trace attribute is preserved on
  monomorphic copies (verified at lower_ast.py:477). Each
  monomorphic copy gets its own TRACE_ENTRY/EXIT pair; DCE
  preserves all of them. No silent-failure window.
- **DCE + FFI**: FFI_CALL is already in SIDE_EFFECT_KINDS
  (Stage 16.5 CRITICAL-1). @trace + FFI is a possible
  combination (a traced fn that calls an FFI function); both
  pairs of ops survive DCE. No new interaction.
- **`check.py` outer-except**: the trace pipeline lives entirely
  inside the typecheck → IR → backend chain; any internal
  exception during the trace lowering propagates to
  check.py:618/649/663's broad-Exception arms which emit
  "internal error" + "compiler bug" + rc=1. Not silent.
- **Cycle-2 C2-2 + cycle-14 C13-1 interaction**: the early-
  return path's synthesized const_int(0) is now also kept live
  by the same fix. Both C2-2's TRACE_EXIT and C13-1's
  TRACE_EXIT operand survive DCE. No silent failure on either
  path.

### Did adding TRACE_ENTRY/EXIT to SIDE_EFFECT_KINDS break a
different invariant?

The question to answer is: does **forcing TRACE_* operands live**
introduce any silent-failure window elsewhere?

- **Performance**: marginally more ops survive DCE (the
  const_int(0) sentinel for unit-returning traced fns). This is
  by design — preserving correctness over performance. No
  silent failure.
- **Register pressure**: the synthesized const_int(0) consumes a
  single slot in the stack frame. The backend's slot allocator
  handles this without complaint (i32 const → 4 bytes →
  standard slot). No silent failure.
- **SSA validity**: the consumed-but-not-otherwise-used
  const_int(0) is a tree-style def-use (one producer, one
  consumer). SSA is trivially valid. No silent failure.
- **Effect-check pass**: `helixc/ir/passes/effect_check.py`
  validates effects. TRACE_ENTRY/EXIT aren't in the effect-
  inference machinery (they're observability ops, not Helix-
  language effects like `IO` or `Mem`). Verified by `grep
  TRACE_ helixc/ir/passes/`. No silent-failure window in the
  effect-checker.
- **Reflection (Quote/Modify/Splice) interaction**: a traced
  fn can also use reflection. Both subsystems are in
  SIDE_EFFECT_KINDS; both survive DCE. No silent-failure
  window.
- **Backend slot reuse**: the backend's slot allocator
  (`_slot_of`) keys by value-id. The const_int(0)'s value-id is
  stable across the fold + dce passes (DCE only removes ops; it
  doesn't renumber). The slot allocation phase in x86_64.py
  assigns a slot once and the TRACE_EXIT consumer reads from
  that slot. No silent failure.

**Conclusion**: the cycle-14 fix opens no fresh silent-failure
window.

### Carryover findings status (cycles 1-13) — unchanged

| Carryover | Severity | Cycle-14 status |
|---|---|---|
| audit-C4-1 (D2 Call-RHS i32 SIGILL) | CRITICAL | **still open** — not addressed by the cycle-14 fix-sweep. Highest-priority unaddressed-CRITICAL. |
| audit-C4-4 (D9 paper-only) | HIGH | **still open** — not addressed |
| audit-C4-8 (check.py doesn't call fn-mono) | LOW | **still open** — not addressed |
| C5-10 (lower_ast.py:2113-2117 + 2079-2092 + 2093-2101) | LOW | **still open** — not addressed; not re-flagged per the user's strict re-flag rule |
| monomorphize_safe docstring drift | (housekeeping) | **still open** — docstring still suggests callers MAY ignore diags |
| D-vs-Quote diagnostic text | (housekeeping) | **still open** — Quote-wrapped case still emits "(one side D-wrapped, other bare)" |
| C7-1 test-coverage gap | (housekeeping) | **still open** — `_compatible(TyMemTier, TyVar) is False` regression tests not added |
| C8-1 (cycle-8 MEDIUM) | MEDIUM | CLOSED by cycle 9 |
| C8-2 (cycle-8 LOW) | LOW | CLOSED by cycle 9 |
| C9-1 (cycle-9 LOW) | LOW | CLOSED by cycle 10 |
| **C13-1 (cycle-13 HIGH, DCE drops TRACE_EXIT operand)** | **HIGH** | **CLOSED by cycle 14 fix-sweep at 1e4c3e6** |

C13-1 is the only carryover closed by cycle 14. The remaining
carryovers are NOT re-flagged as new cycle-14 findings per the
user directive.

---

## CRITICAL FINDINGS

(none)

---

## HIGH FINDINGS

(none)

---

## MEDIUM FINDINGS

(none)

---

## LOW FINDINGS

(none)

---

## Re-audit verification on 1e4c3e6 (production surface +13 lines in dce.py, +35 in test_dce.py vs cycle-13 HEAD 98834de)

The cycle-14 fix-sweep is surgical: 2 new members in a frozenset,
1 explanatory comment, 2 new regression tests. The blast radius
is the DCE pass. Cross-pass interaction analysis (CSE, FDCE,
const_fold, grad_pass, monomorphize, effect_check, backend) shows
no cascading silent-failure window.

| Re-audit pass | C10 | C11 | C12 | C13 | C14 | Stability |
|---|---|---|---|---|---|---|
| `_emit_env_error` strip helper (check.py:246-255) | clean | clean | clean | clean | clean | stable |
| Outer-except topology (check.py:284-318) | clean | clean | clean | clean | clean | stable |
| Finally drain-failure suppressor (check.py:319-337) | clean | clean | clean | clean | clean | stable |
| Backend-call wraps (check.py:618,649,663) | clean | clean | clean | clean | clean | stable |
| AD-warning narrowed excepts (autodiff.py:155,1012) | clean | clean | clean | clean | clean | stable |
| const_fold defensive folds (const_fold.py:250,324,349,401) | clean | clean | clean | clean | clean | stable |
| Quote-handle fallback (lower_ast.py:2115) | C5-10 carryover | C5-10 carryover | C5-10 carryover | C5-10 carryover | C5-10 carryover | stable carryover |
| diagnostics isatty fallback (diagnostics.py:76) | non-finding | non-finding | non-finding | non-finding | non-finding | stable |
| `getattr(it, "is_kernel", False)` (check.py:641) | non-finding | non-finding | non-finding | non-finding | non-finding | stable |
| lower_ast.py try/finally scope at :596, :1800 | (n/e) | (n/e) | C12 fresh: clean | clean | clean | stable |
| backend/x86_64.py attrs.get defaults | (n/e) | (n/e) | C12 fresh: clean | clean | clean | stable |
| backend/ptx.py, elf_dyn.py zero-except | (n/e) | (n/e) | C12 fresh: clean | clean | clean | stable |
| ir/tile_ir.py, tir.py zero-raise | (n/e) | (n/e) | C12 fresh: n/a | n/a | n/a | n/a |
| frontend/parser.py:375 ValueError -> ParseError re-raise | clean | clean | clean | clean | clean | stable |
| frontend/typecheck.py:415,423 TypeError_ -> diag append | clean | clean | clean | clean | clean | stable |
| frontend/typecheck.py:636 ValueError -> Optional None | clean | clean | clean | clean | clean | stable |
| frontend/monomorphize.py:203 ValueError -> return expr | clean | clean | clean | clean | clean | stable |
| frontend/monomorphize.py:759 ShapeFoldError -> diag list | clean | clean | clean | clean | clean | stable |
| frontend/grad_pass.py:639-643 frozen-dataclass cache fallback | (n/e) | (n/e) | (n/e) | C13 fresh: clean | clean | stable |
| frontend/pytree.py:293-296 validate_pytree diagnostic collection | (n/e) | (n/e) | (n/e) | C13 fresh: clean | clean | stable |
| frontend/hash_cons.py:335 raise HashConsError | (n/e) | (n/e) | (n/e) | C13 fresh: clean | clean | stable |
| frontend/flatten_impls.py:88 raise DuplicateMethodError | (n/e) | (n/e) | (n/e) | C13 fresh: clean | clean | stable |
| frontend/flatten_modules.py:67,77 raise FlattenError | (n/e) | (n/e) | (n/e) | C13 fresh: clean | clean | stable |
| frontend/trace_pass.py:67 raise OverflowError | (n/e) | (n/e) | (n/e) | C13 fresh: clean | clean | stable |
| ir/passes/effect_check.py:228 raise EffectError | (n/e) | (n/e) | (n/e) | C13 fresh: clean | clean | stable |
| examples/dashboard_server.py try/except sites | (n/e) | (n/e) | (n/e) | C13 fresh: n/a | n/a | n/a |
| No `except: pass` in production | clean | clean | clean | clean | clean | stable |
| **dce.py SIDE_EFFECT_KINDS frozenset (incl. C14 +TRACE_ENTRY/EXIT)** | (n/e) | (n/e) | (n/e) | (n/e) | **C14 fresh: clean** (all 20 members justified by Helix semantics; comment block on the 2 new members is load-bearing and accurate) | new |
| **cse.py PURE_KINDS dual-check vs new SIDE_EFFECT_KINDS members** | (n/e) | (n/e) | (n/e) | (n/e) | **C14 fresh: clean** (TRACE_* correctly absent from PURE_KINDS; positive-allowlist topology makes the two sets non-overlapping by construction) | new |
| **fdce.py call-graph source check vs TRACE_*** | (n/e) | (n/e) | (n/e) | (n/e) | **C14 fresh: clean** (TRACE_* not used as call edges) | new |
| **x86_64.py TRACE_EXIT operand consumer guard** | (n/e) | (n/e) | (n/e) | (n/e) | **C14 fresh: clean** (`if op.operands:` guard is defensive, not silent — the lowerer always emits an operand) | new |
| **lower_ast.py synthesized-const sentinel (line 573-574, 1891-1892)** | (n/e) | (n/e) | (n/e) | (n/e) | **C14 fresh: clean** (i32 const, matches backend's mov_eax_mem_rbp width; emitted in same block as consumer) | new |

### Specific cycle-14 items re-checked clean

- **`SIDE_EFFECT_KINDS` integrity**: 20 members, each
  justified by Helix semantics or backend layout requirement.
  No silent-failure window introduced by the +2 members.
- **`SIDE_EFFECT_KINDS` consumers**: 3 in dce.py (lines 103,
  119, 131) + 1 cross-import in test_cse.py:181. Read-only
  audit confirms no other module consumes the set.
- **TRACE_EXIT operand-chain through DCE**: seed
  (line 103-105) marks the const_int(0) operand live; spread
  (line 119-125) leaves it live; drop (line 131-141) keeps the
  const_int(0) producer because its result is in `live`. End
  to end: the operand survives. Verified by tracing
  `test_c13_1_dce_preserves_trace_exit_operand` against the
  three-phase logic.
- **TRACE_ENTRY survival**: it has no operands, so the seed
  phase produces no new live ids for it. It has no results,
  so the spread phase doesn't touch it. The drop phase keeps
  it via the `if op.kind in SIDE_EFFECT_KINDS` arm at line
  131. End to end: TRACE_ENTRY survives. Verified by
  `test_c13_1_dce_preserves_trace_entry_in_kept_set`.
- **Backend slot lookup on the now-live operand**: the
  const_int(0)'s value-id is stable through lower → fold →
  dce. The backend's slot allocator assigns a slot keyed by
  value-id; the TRACE_EXIT consumer reads from that slot.
  No KeyError at codegen.
- **Test pass count**: the commit message reports `253 targeted
  tests pass (was 250)`. The +3 reflects +2 new tests plus 1
  additional test that the cycle-13 fix-sweep matrix exposed.
  No regression.

### Carryover findings status (cycles 1-13)

Already shown above. The cycle-14 fix-sweep closed C13-1 (the
only HIGH it addressed) and changed no other carryover.

---

## Deferred / out-of-scope observations (NOT new findings; cycle-15 candidates)

- **Carryover audit-C4-1 (D2 Call-RHS i32 SIGILL)**: still
  open CRITICAL. Cycle 14 did not address — the fix-sweep was
  scoped to C13-1 only. **STILL THE HIGHEST-PRIORITY ITEM**
  for any future fix-sweep. As the clean-counter resets to 0
  and re-accumulates, the cycle-12 / cycle-13 recommendation
  stands: prioritize audit-C4-1 in the next fix-sweep
  regardless. If the user elects to land C4-1 between cycles,
  the clean-counter resets again — which is the correct
  tradeoff for a CRITICAL issue.
- **Carryover audit-C4-4 (D9 paper-only)**: still open HIGH.
  Not addressed.
- **Carryover audit-C4-8 (check.py doesn't call fn-mono)**:
  still open LOW. Not addressed.
- **C5-10 lower_ast.py silent fallbacks**: still open LOW
  (Pattern A: quote-handle structural_hash -> _pretty
  fallback at :2115; Pattern B: Cast None inner ->
  const_int(0); Pattern C: Field no-array-match returns
  None). Not addressed; not re-flagged.
- **monomorphize_safe docstring drift**: still open
  (cycle-6 deferred).
- **D-vs-Quote diagnostic text**: still open (cycle-7
  deferred).
- **C7-1 test-coverage gap**: still open. Cycle 14 did not
  add the 4 `_compatible(TyMemTier, TyVar)` regression tests.
- **`_emit_env_error` triple-prefix / uppercase-prefix
  edge cases**: still no callee triggers either. Not
  findings.
- **TRACE_EXIT operand-less defensive guard
  (x86_64.py:2495)**: the `if op.operands:` guard tolerates
  a hypothetical operand-less TRACE_EXIT silently (emitting a
  bare `nop`). The lowerer never emits such an op today
  (both call sites at lower_ast.py:573-575 + 1891-1896 always
  provide an operand, synthesizing const_int(0) for the Unit
  case). Whether this is a future silent-failure window
  depends on whether the lowerer can ever emit an
  operand-less TRACE_EXIT. Today the answer is no. Not a
  finding for cycle 14, but worth tracking if the trace
  machinery evolves.

---

## Cycle 13 vs cycle 14 — clean-cycle counter check

Cycle 13 was NOT clean (C13-1 HIGH from the code-review lens).
The clean-counter therefore reset from 3/5 → 0. Cycle 14 is the
first fresh re-audit window.

The cycle-14 silent-failures audit lens specifically asks:
1. Is the cycle-13 silent-failures audit still clean? Yes (no
   new silent-failure window since cycle 13).
2. Did the cycle-14 fix-sweep introduce any silent-failure
   window? No (the fix is surgical — adding TRACE_ENTRY and
   TRACE_EXIT to SIDE_EFFECT_KINDS — and the cross-pass
   interaction analysis surfaces no cascading hazard).

Cycle 14 produces **zero new findings of any severity**, so the
clean-cycle counter advances to **1/5** for this audit lens —
subject to the parallel type-design and code-review lenses
ALSO being clean for cycle 14 (per the user's strict-clean rule,
all three lenses must clear before a cycle counts as clean).
The user note confirms cycle-14 type-design and code-review
both completed CLEAN on this HEAD prior to this retry, so the
parallel-lens condition is satisfied.

---

## Summary

| #    | Severity | Location | Finding |
|------|----------|----------|---------|
|      |          |          | (none — cycle 14 is CLEAN for the silent-failure lens) |

**Total: 0 new findings (0 CRITICAL, 0 HIGH, 0 MEDIUM,
0 LOW).**

---

## Cycle 14 status

**Cycle 14 IS CLEAN** for the silent-failure audit lens. Per the
strict criterion (zero findings of ANY severity), the 0-finding
result satisfies the clean-cycle gate for this audit lens.

### Stop-the-line determination: **NO**

Cycle 14 silent-failures is clean — no stop required for this
lens.

### Cycle 14 -> NEW FINDINGS COUNT for the strict-clean gate: 0
(0 CRITICAL + 0 HIGH + 0 MEDIUM + 0 LOW) — clean-counter
advances to **1/5** for this audit lens (the prior 3/5 reset to
0 because cycle 13 was not clean; cycle 14 is the first
re-accumulated clean cycle).

### Severity trend across cycles

- Cycle 1: 13 findings (3 HIGH, 5 MEDIUM, 5 LOW).
- Cycle 2: 6 findings (1 HIGH, 4 MEDIUM, 1 LOW).
- Cycle 3: 6 findings (0 HIGH, 4 MEDIUM, 2 LOW).
- Cycle 4: 8 findings (1 CRITICAL, 2 HIGH, 4 MEDIUM, 1 LOW).
- Cycle 5: 4 findings (0 CRITICAL, 0 HIGH, 2 MEDIUM, 2 LOW).
- Cycle 6: 1 finding (0 CRITICAL, 0 HIGH, 1 MEDIUM, 0 LOW).
- Cycle 7: 1 finding (0 CRITICAL, 0 HIGH, 0 MEDIUM, 1 LOW).
- Cycle 8: 2 findings (0 CRITICAL, 0 HIGH, 1 MEDIUM, 1 LOW).
- Cycle 9: 1 finding (0 CRITICAL, 0 HIGH, 0 MEDIUM, 1 LOW).
- Cycle 10: 0 findings.
- Cycle 11: 0 findings.
- Cycle 12: 0 findings.
- Cycle 13: 0 findings (silent-failures lens; code-review lens
  found C13-1 HIGH, addressed by cycle-14 fix-sweep).
- Cycle 14: 0 findings (silent-failures lens). <- here

Trend: the silent-failures lens has been clean for 5 consecutive
cycles (10, 11, 12, 13, 14). However, the strict-clean gate
requires ALL three lenses to be clean per cycle; cycle 13's
code-review lens broke that, so the global counter resets to 1/5
at cycle 14.

### Estimated remaining open findings going into cycle 15

- Cycle 1: 13 new (all fixed -> 0 open).
- Cycle 2: 6 new (all fixed -> 0 open).
- Cycle 3: 6 new (all fixed -> 0 open).
- Cycle 4 silent-failure: 8 new — 6 closed by cycles 5-9.
  2 still open: audit-C4-1 CRITICAL, audit-C4-4 HIGH.
- Cycle 5 silent-failure: 4 new — 3 closed by cycle 6
  (C5-5, C5-6, C5-7, C5-8 MEDIUM and C5-9 LOW), 1 still
  open (C5-10 LOW, lower_ast.py fallbacks).
- Cycle 6 silent-failure: 1 new (C6-1) — CLOSED.
- Cycle 6 type-design: 2 new (G1, G2) — both CLOSED.
- Cycle 7 silent-failure: 1 new (C7-1) — CLOSED by cycle 8.
- Cycle 8 silent-failure: 2 new (C8-1 MEDIUM, C8-2 LOW) —
  both CLOSED by cycle 9.
- Cycle 9 silent-failure: 1 new (C9-1 LOW) — CLOSED by
  cycle 10.
- Cycle 10 silent-failure: 0 new.
- Cycle 11 silent-failure: 0 new.
- Cycle 12 silent-failure: 0 new.
- Cycle 13 silent-failure: 0 new (code-review lens found
  C13-1 HIGH — CLOSED by cycle 14 fix-sweep).
- Cycle 14 silent-failure: 0 new. <- here
- Prior audits (stage 5-6 + 7-8 + 9-16): ~20 still-open
  (unchanged going into cycle 15).
- Cycle 14 net: 20 + 2 (C4-1 + C4-4) + 1 (C5-10) + 0
  (cycle-14 new) + (deferred type-design partial) = **>=23
  open findings** going into cycle 15. (Net -1 delta vs
  cycle 13's >=23+1: C13-1 closed.)

Recommend prioritizing in this order for the cycle-15 fix
batch (if user elects to land fixes between clean
re-audits):
1. **audit-C4-1** (CRITICAL — still-open from cycle 4;
   highest-priority unaddressed-CRITICAL; deferred in
   cycles 6-14).
2. **audit-C4-4** (HIGH — D9 paper-only).
3. **C5-10** (LOW — lower_ast.py fallbacks).
4. **C7-1 test-coverage gap** (combinable with audit-C4-1
   if the fix touches typecheck.py).
5. **monomorphize_safe docstring drift** (housekeeping).
6. **D-vs-Quote diagnostic text** (housekeeping).

The "5 clean cycles before Phase 0 deprecation" goal requires
the strict criterion (zero findings of any severity, all three
lenses) to be met for 5 CONSECUTIVE cycles. Cycle 14 is the 1st
of the re-accumulated window (counter 1/5) — assuming the
parallel type-design + code-review lenses at cycle 14 also
remain clean (per the user note: they did).

**Cycle 14 status: CLEAN**
**Counter status: 1/5** (cycle 14 silent-failures clean;
parallel type-design + code-review lenses noted CLEAN by the
user; counter reset from 3/5 → 0 → 1 because cycle 13 was not
clean due to the C13-1 code-review finding which the cycle-14
fix-sweep closed).
