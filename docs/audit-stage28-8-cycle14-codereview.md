# Stage 28.8 Pre-29 Audit Gate — Cycle 14, Audit C: Code Review

**Date**: 2026-05-11
**Commit**: 1e4c3e6 (read-only)
**Scope**: Audit C (general code-review) on the cycle-14 fix-sweep
landed at 1e4c3e6 to close cycle-13 finding C13-1 (HIGH, conf 95).
Per the cycle-14 prompt, this is the new cycle 1 of 5 — the
clean-streak counter was reset to 0 by cycle 13's non-clean
verdict, and a fresh 5-consecutive-clean run is required to clear
the Stage 29 gate.

**Cycle-counter status going in**: 0/5 (reset by cycle-13's HIGH
finding).

**Method**:

(a) Read `docs/audit-stage28-8-cycle13-codereview.md` to recover
    the exact contract C13-1 documented and the conf-95 evidence
    base.

(b) Ran `git show 1e4c3e6 --stat` and `git show 1e4c3e6 --
    helixc/ir/passes/dce.py helixc/tests/test_dce.py` to confirm
    the fix is isolated to the two intended files and consists
    of (i) two enum entries appended to `SIDE_EFFECT_KINDS` with
    a 12-line explanatory comment, and (ii) two new tests in
    `test_dce.py`.

(c) Verified the enum entries `tir.OpKind.TRACE_ENTRY` and
    `tir.OpKind.TRACE_EXIT` actually exist at `tir.py:301-302`
    (the comment block at `tir.py:286-300` documents the operand
    contract: TRACE_EXIT's `operand[0]` is the return value).

(d) Re-read the post-fix `dce.py` end-to-end (143 LOC) to confirm
    no other code paths needed to change. The fix is purely
    additive in the seed set; the keep-arm at `dce.py:134-136`
    (`if not op.results: new_ops.append(op)`) — which was the
    proximate enabler of the dangling-operand bug — is left
    intact, which is correct because the seed-set membership
    short-circuits at `dce.py:131-133` first.

(e) Re-read `lower_ast.py:567-583` to re-confirm the producer
    site (the synthesized `const_int(0)` at line 574 consumed
    only by TRACE_EXIT at line 575) is unchanged.

(f) Re-read `x86_64.py:2489-2502` to confirm the backend
    operand-consumer (the `_slot_of(op.operands[0])` at line
    2498) is unchanged. The comment at lines 2492-2494 ("The
    return value operand is still consumed so liveness analysis
    keeps the value alive past the trace call") is now a true
    statement of contract rather than a buggy assumption — the
    cycle-13 evidence that flagged the comment as encoding the
    bug is no longer applicable.

(g) Ran `python helixc/tests/test_dce.py`:
    `8 passed, 0 failed` — including both new C13-1 tests.

(h) Ran the cycle-13 end-to-end reproducer (`@trace fn foo() {
    let x: i32 = 5; }` + `-O2`) through
    `parse -> typecheck -> lower -> fold -> dce -> compile_module_to_elf`:
    `OK: end-to-end -O2 compile succeeded, ELF bytes: 4691`.
    Pre-fix this path crashed with `KeyError: 1` at
    `x86_64.py:2498`. The crash is closed.

(i) Ran `helixc/tests/test_trace.py` (the existing trace test
    surface): `8 passed` (21 passed in the wider pytest slice
    that exercises trace-related items). No regression.

(j) Verified the new `test_c13_1_dce_preserves_trace_exit_operand`
    test genuinely catches the contract it claims to pin: with
    `SIDE_EFFECT_KINDS` artificially restored to its pre-fix
    state (i.e. with both TRACE_ENTRY and TRACE_EXIT discarded),
    the test fails with the documented assertion text "TRACE_EXIT
    operand id=1 was DCE'd (producer dropped); backend will
    KeyError at -O2". The regression test is correctly wired.

(k) Re-checked SIDE_EFFECT_KINDS for any other no-result side-
    effect-bearing ops that the same audit logic would flag.
    The set is now complete with respect to every emit site in
    `lower_ast.py` whose op carries an operand and produces no
    result.

(l) Spot-walked the diff for unrelated drift: no other files in
    `helixc/` were touched (`git show 1e4c3e6 --stat` shows the
    only non-doc changes are dce.py +13 and test_dce.py +35;
    everything else is the cycle-13 doc set).

**Reporting threshold**: confidence >= 80 (strict criterion per
user directive 2026-05-10).

**Result**: **0 findings at or above the confidence-80 reporting
threshold.**

---

## Summary table

| ID | Severity | Confidence | Component | Issue |
|----|----------|------------|-----------|-------|
| _(none)_ | — | — | — | No high-confidence issues. |

---

## Verification of the three audit-prompt acceptance points

The cycle-14 prompt specifies three structural checks for this
re-verification. Each is confirmed:

### 1. "TRACE_ENTRY/TRACE_EXIT enum members exist in tir.OpKind?"

**Yes.** `helixc/ir/tir.py:301-302` defines:

```python
TRACE_ENTRY = "trace.entry"
TRACE_EXIT  = "trace.exit"
```

inside the `OpKind` enum, with a documentation block at lines
286-300 spelling out the `attrs["fn_name"]` + `operand[0] =
return value` contract. The dce.py additions at lines 79-80
reference these by their fully-qualified `tir.OpKind.TRACE_ENTRY`
/ `tir.OpKind.TRACE_EXIT` symbols, so the fix is structurally
sound and would `AttributeError` at import time if the enum
members were absent — additional belt-and-suspenders.

### 2. "Tests use proper fixtures and exercise the post-DCE invariant?"

**Mostly yes, with one specific caveat (sub-80 quality note,
not a finding).**

The primary regression test
`test_c13_1_dce_preserves_trace_exit_operand` (test_dce.py:112-131)
uses the file-level `lower_fold_dce` helper (test_dce.py:14-18,
which composes parse -> lower -> fold_module -> dce_module — the
same composition every other dce test uses, so the fixture is
canonical), constructs the exact unit-returning `@trace` shape
that cycle-13 identified as the bug-trigger, and asserts the
post-DCE invariant directly: for every surviving TRACE_EXIT,
each operand-id must be present in
`producers + fn.params` of the post-DCE module. This is precisely
the dangling-operand check the cycle-13 audit harness used. The
sub-80 concern I considered and dismissed: I ran the test against
a SIDE_EFFECT_KINDS artificially reverted to its pre-fix state
and confirmed it fails with the documented assertion text. The
regression test is genuine, not a tautology.

The sibling test `test_c13_1_dce_preserves_trace_entry_in_kept_set`
(test_dce.py:134-144) is weaker as a regression guard: TRACE_ENTRY
has no operands in current `lower_ast.py`, so even with the fix
artificially removed the test still passes (the only assertion
is on the surviving op count, and clause-(b) `not op.results`
already preserved TRACE_ENTRY pre-fix). The test docstring is
candid about this ("this is mostly future-proofing if a runtime
helper-handle is added to its operand list later"), so it is a
present-state tautology dressed as a forward-compat guard. I
rated this concern at conf 35: not a bug in the fix (the fix
itself is correct), not a missing-coverage gap that breaks the
strict criterion (the load-bearing test for the actual cycle-13
contract is the TRACE_EXIT one, which is solid), just a slightly
mis-targeted sibling test. Sub-80, not promoted.

### 3. "Comment block in dce.py is accurate?"

**Yes.** The 12-line comment at dce.py:68-78 states four
specific claims; all four are independently verified:

(a) "TRACE_ENTRY / TRACE_EXIT are @trace prologue/epilogue ops
    with side effects (the runtime records entry/exit events)"
    — confirmed by the lowering site at `lower_ast.py:475-484`
    (TRACE_ENTRY emission) and `lower_ast.py:567-583`
    (TRACE_EXIT emission), plus the contract documented at
    `tir.py:286-300` ("the backend will pass to the runtime
    `__helix_trace_entry(name_ptr)` / `__helix_trace_exit(
    name_ptr, ret_val)` for recording").

(b) "TRACE_EXIT consumes the return value as an operand so the
    runtime can log it" — confirmed at
    `lower_ast.py:575-576` (`self.builder.emit(tir.OpKind.TRACE_EXIT,
    ret_operand, attrs={"fn_name": fn.name})`).

(c) "for unit-returning traced fns, lower_ast.py synthesizes a
    `const_int(0)` whose sole consumer is TRACE_EXIT" —
    confirmed at `lower_ast.py:572-574`.

(d) "the backend (x86_64.py:2498) KeyErrors when it tries to
    look up the slot of the now-deleted operand on `-O2`" —
    confirmed at `x86_64.py:2498` (`ret_slot =
    self._slot_of(op.operands[0])`) and `x86_64.py:864`
    (`return self.slots[v.id]`). The `-O2` gate is at
    `check.py:579-588` (DCE only runs at `opt_level >= 2`).

All four claims are accurate. No comment-text drift.

---

## Why no findings at >= 80

The cycle-14 fix is the minimal correct closure of C13-1:

1. **Right diagnosis**: the fix targets the seed-set, which is
   the exact phase of `dce_function` where the bug originates
   (operands of seed-set ops are walked; operands of clause-(b)
   no-result-but-kept ops are not).

2. **Right scope**: only TRACE_ENTRY and TRACE_EXIT are added.
   I cross-checked the full `tir.OpKind` enum (90+ members)
   against the post-fix SIDE_EFFECT_KINDS to look for any other
   no-result, operand-bearing, side-effect-carrying ops that
   would have the same bug shape. None found in the current
   IR. The cycle-13 below-threshold item B13-2 (the dangling
   `TENSOR_STORE` enum member) is still a speculative future-
   proofing concern at conf 35, not a present hazard.

3. **Right regression test**: the primary test in test_dce.py
   uses the same lower+fold+dce composition as the rest of the
   file's tests (no fixture drift), constructs the exact
   trigger shape, and asserts the post-DCE invariant — verified
   to fail when the fix is artificially removed.

4. **End-to-end repro closed**: the cycle-13 prompt's exact
   reproducer (`helixc check ... -o ... -O2` on the unit-
   returning `@trace` source) now succeeds, producing a 4691-byte
   ELF instead of crashing with `KeyError: 1`.

5. **No regression in adjacent surface**: `test_trace.py`
   (the 8 existing @trace tests) all still pass.

6. **No collateral edits**: the diff is exactly 48 net lines
   across the two intended files; nothing else in `helixc/`
   moved.

---

## Below-threshold observations from this cycle

The following items were rated below conf 80 and are NOT counted
as findings. Surfaced here for cumulative carryover only.

### B14-1 — `test_c13_1_dce_preserves_trace_entry_in_kept_set` is a present-state tautology (conf 35)

The sibling test passes regardless of whether the fix is applied,
because TRACE_ENTRY has no operands in current lower_ast.py and
the clause-(b) `not op.results` keep-arm preserves it
unconditionally. The test docstring acknowledges this as
"future-proofing", but as written the test does not constrain
the contract under audit (C13-1 was specifically about
TRACE_EXIT's operand being dropped). Promoting to a hard finding
would require a clearer mandate that every regression test
must demonstrably fail without its corresponding fix — the
audit prompt does not state this. Conf 35.

### B14-2 — DCE comment block at the top of dce.py (lines 14-22) does not list TRACE_ENTRY/TRACE_EXIT in the "Side-effecting op kinds" docstring summary (conf 30)

The high-level docstring summary at `dce.py:14-22` enumerates
RETURN, BR, COND_BR, CALL, STORE_VAR, STORE_ELEM, ALLOC_VAR,
ALLOC_ARRAY, MODIFY, SPLICE, "io.print, etc." as the side-
effecting kinds. The actual set at lines 32-80 is larger
(QUOTE, REFLECT_HASH, ARENA_PUSH, ARENA_SET, TILE_INDEX_STORE,
FFI_CALL, TRAP, TRACE_ENTRY, TRACE_EXIT). The docstring is
stale in general — this is a pre-existing drift that predates
cycle 14 — and the cycle-14 commit did not refresh it. Not a
correctness issue. Conf 30.

### B14-3 — No invariant-checker test asserting "every op's operand-id is in (producers + params + block_params)" after DCE on a randomized IR (conf 50)

The cycle-13 B13-3 observation (a global post-DCE invariant
checker would have caught C13-1 immediately) still applies.
The cycle-14 fix closes the specific instance but does not add
the general defensive pass. A future audit could promote this
to a fix-sweep candidate. Conf 50, unchanged from cycle 13.

---

## Below-threshold re-evaluation (cycles 6-13 carryover)

Walked the cumulative below-threshold concern list once more.

- **Cycle 6** (TRAP-const test additions; monomorphize_safe
  docstring drift): no change. Conf <= 40.
- **Cycle 7** (D-vs-Quote diagnostic text): no change. Conf 30.
- **Cycle 8** (C7-1 `_compatible(TyMemTier, TyVar)` coverage gap):
  no change. Conf 55.
- **Cycle 9** (OSError edge cases; pathological strip-shape):
  no change. Conf 25-55.
- **Cycle 10 / 11 / 12** (test ordering; signature; commit-
  message hygiene; strict-stdlib end-to-end coverage):
  no change. Conf 10-55.
- **Cycle 13** (B13-1 dce_function outer-loop re-seed; B13-2
  TENSOR_STORE speculative; B13-3 missing invariant checker):
  no change. Conf 35-55.

No promotions to >= 80 from prior cycles. No new findings this
cycle.

---

## Open prior findings (not addressed this cycle)

Per cumulative carryover from cycle 13:

- **audit-C4-1** (CRITICAL — D2 Call-RHS i32 SIGILL): still open.
- **audit-C4-4** (HIGH — D9 paper-only): still open.
- **audit-C4-8 deferred** (LOW — check.py doesn't call fn-mono):
  still open.
- **monomorphize_safe docstring drift** (cycle-6 deferred):
  still open.
- **D-vs-Quote diagnostic text** (cycle-7 deferred): still open.
- **C7-1 close test-coverage gap** (cycle-8 housekeeping):
  still open.
- **C13-1** (HIGH — DCE drops TRACE_EXIT operand on unit-
  returning @trace fns): **CLOSED** by 1e4c3e6.

---

## Verdict

**Cycle 14 Audit C: CLEAN — 0 findings at or above the
confidence-80 reporting threshold.**

Per user directive 2026-05-10 (strict criterion): zero findings
of any severity at the threshold qualifies as clean.

**Cycle-counter advances from 0/5 -> 1/5** (pending the parallel
silent-failures and type-design audits for this cycle; the
overall clean-streak counter advances only if all three of A/B/C
are clean).

Stage 29 gating remains: **5 consecutive fully-clean cycles
required**; this is the first.

---

## Cycle 14 status

- Audit C result: **CLEAN — 0 findings.**
- Counter going in: 0/5 (reset by cycle 13's HIGH).
- Counter going out: **1/5 (advance, contingent on parallel
  audits A and B for cycle 14).**
- Production HEAD audited: 1e4c3e6 (cycle-14 fix-sweep).
- C13-1 verification: **closed.** Fix is well-isolated
  (additive set membership only, 12-line comment), regression
  test in test_dce.py genuinely catches the contract (verified
  by reverting the fix locally), TRACE_ENTRY/TRACE_EXIT enum
  members are confirmed to exist at tir.py:301-302, comment
  block in dce.py:68-78 is fully accurate against the
  lower_ast / x86_64 source-of-truth.
- Stage 29 gating: **BLOCKED on 4 more consecutive clean cycles
  (15, 16, 17, 18).**
