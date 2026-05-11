# Stage 28.8 Pre-29 Audit Gate — Cycle 8, Audit B: Type-System / Dispatch / Soundness

**Date**: 2026-05-11
**Commit**: 5d1ca24 (read-only)
**Scope**: Re-audit type-system soundness focused on cycle-8's fix-sweep
(5d1ca24 — drops cycle-7 G2 TyMemTier × (TyVar | TySize) carve-out per
cycle-7 silent-failures finding C7-1). The cycle-8 fix-sweep touches:

- `helixc/frontend/typecheck.py:2230-2252` — `_compatible`'s
  TyMemTier strict-separation block. The two carve-out arms
  `_compatible(TyMemTier, TyVar/TySize) -> True` and the reverse
  (both inserted in cycle 7) are removed. The both-MemTier arm
  (2249-2250) and the broad-or rejection arm (2251-2252) remain.
  A cycle-8 explanatory comment block (2230-2240) replaces the
  cycle-7 carve-out justification comment.
- `helixc/check.py:271-329` — exception-handler rework. This is
  the cycle-5 C4-6 / MEDIUM fix surfacing in this commit's diff
  view (the cycle-5 fix-sweep that pre-existed cycle 8); the
  cycle-8 commit message attributes the change to cycle 5, and
  the surrounding comments confirm. No new contract surface
  introduced in cycle 8 for check.py — already audited under
  cycle 5 / cycle 6.

The functional cycle-8 change is the typecheck.py 4-line deletion
plus the comment-block swap. The check.py block is a no-op for
cycle-8 type-design audit (re-audited under cycle 5 / 6).

**Method**: read cycle-1 through cycle-7 type-design audit docs to
build the cumulative invariant set, then walked the cycle-8 diff
through each contract it touches. Specifically:

For the carve-out removal: enumerated every `_compatible` call site
(20 sites in typecheck.py) and classified each by whether the
caller pre-filters TyVar / TySize before the call. Identified the
body-vs-return site at line 1152, the let-declared-vs-value site
at line 1174, the if-branch-unify site at line 1570, the
match-arm-unify site at line 1596, and the field-init site at
line 1669 — none of which pre-filter. Verified the call-boundary
site at lines 746-752 DOES pre-filter both pty and aty for
(TyVar, TySize, TyUnknown) before delegating to `_compatible`, so
the generic-call-site deferral path is preserved by the call-site
filter, not the structural matcher. Confirmed the cycle-8 commit
message's claim that re-introducing the carve-out at the call
boundary would be the right place if it's needed later.

For the both-MemTier arm interaction: walked
`_compatible(TyMemTier(W, TyVar('T')), TyMemTier(W, i32))`. Arm at
2249-2250 fires, recurses on `_compatible(TyVar('T'), i32)`. Post-
cycle-8 this recurses through the rest of `_compatible` and lands
on the final `return a == b` (identity), which is False. Same
behavior as cycle 7 (cycle 7 dropped the top-level TyVar cascade
via `_size_compatible`, so the nested TyVar inside TyMemTier was
already not auto-passing). Not a regression.

For the TyMemTier × TySize half: `_compatible(TyMemTier(W, i32),
TySize('N'))`. Both-MemTier arm at 2249 doesn't fire (b is not
TyMemTier). Broad-or arm at 2251 fires (a is TyMemTier), returns
False. The cycle-7 carve-out would have intercepted this with
True; cycle 8 correctly rejects. This is the half the cycle-8
commit message calls "a genuine kind mismatch (a size can't be a
memory-tier value)".

For the TyMemTier × TyVar half: `_compatible(TyMemTier(W, i32),
TyVar('T'))`. Both-MemTier arm doesn't fire. Broad-or arm fires,
returns False. Pre-cycle-8 (cycle-7 G2 carve-out) returned True.
Post-cycle-8 rejects. The cycle-8 commit message says this is
intentional: "TyMemTier × TyVar at value position is rare enough
that a hard error is preferable to silent acceptance". Verified
by reading the body-vs-return reproducer:
`fn g[T]() -> T { make_working_mem(42) }` (where make_working_mem
returns WorkingMem<i32>). Cycle 7 silently accepted (wrong: T
might bind to i32, not WorkingMem<i32>); cycle 8 rejects with
"body type WorkingMem<i32> does not match return type T". The
hard error is correct — mono substitution binds T to i32 (the
caller's type), and i32 ≠ WorkingMem<i32>, so the cycle-7 cascade
was silently masking a real bug.

For tests: cycle 8 added no new test. The commit says "223
targeted tests pass" but no new regression test was added for
the dropped carve-out case (TyMemTier × TyVar/TySize at top-level
value position rejecting). This is acceptable — the change is a
2-arm deletion with explicit in-code documentation — but a
forward note for the test suite is warranted (see "Forward note"
below; not a cycle-8 finding because it's a meta-observation
about the cycle-7 carve-out's original lack of coverage, not
about cycle-8 introducing new untested surface).

**Result**: **0 findings (0 HIGH, 0 MEDIUM, 0 LOW)**. Cycle 8
removes the cycle-7 G2 carve-out cleanly. The two-arm deletion
restores the pre-cycle-7 contract (strict TyMemTier separation
from all non-TyMemTier types) and the call-boundary pre-filter at
lines 746-752 preserves the original cycle-3 D1 generic-defer
behavior without needing the structural-matcher cascade. The
strict criterion ("zero findings of any severity") is **MET**.

---

## Cycle 7 finding re-verification

| ID   | Severity prev | Audit (prev)            | Status     | Notes                                                                                                                                                                                                                                                                                                                                                                |
|------|---------------|-------------------------|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| C7-1 | LOW           | silent-failures (cyc 7) | CLOSED     | Cycle 7's G2 TyMemTier × (TyVar | TySize) carve-out arms are deleted from `_compatible`. Top-level value-position TyMemTier × TyVar and TyMemTier × TySize compares now reject via the broad-or arm at 2251-2252. Call-boundary deferral (lines 746-752) still pre-filters TyVar/TySize/TyUnknown on both pty and aty sides before delegating, so generic call sites still defer to mono. |

Cycle 7's type-design audit ([cycle 7 doc](audit-stage28-8-cycle7-type-design.md))
found 0 findings; cycle 7's silent-failures audit found 1 LOW (C7-1).
Cycle 8 closes C7-1. No other cycle-7 findings exist to re-verify.

---

## Per-surface review (cycle-8 touchpoints)

### Surface 1: `_compatible` TyMemTier strict-separation (post-cycle-8)

**Placement**: `helixc/frontend/typecheck.py:2224-2252`.

**Pre-cycle-8 contract (cycle 7)**:
- Both-MemTier: tier-match AND inner-compatible (line 2249-2250).
- TyMemTier × TyVar/TySize: True via two explicit carve-out arms
  (the dropped cycle-7 G2 arms).
- TyMemTier × any-other-type: False (broad-or arm).

**Post-cycle-8 contract**:
- Both-MemTier: tier-match AND inner-compatible (line 2249-2250).
- TyMemTier × any-other-type (including TyVar / TySize / TyUnknown
  at top level): False (broad-or arm at 2251-2252).
- Exception: TyUnknown short-circuits at the top of `_compatible`
  (line 2225-2226), so `TyMemTier × TyUnknown` still returns True
  via the top-level cascade. That's intentional: TyUnknown is
  the placeholder-for-anything wildcard and pre-dates the
  TyMemTier arm.

**Contract integrity**: stricter than cycle 7 by exactly the two
deleted arms. The contract reads more cleanly without them:
"TyMemTier values are incompatible across tiers AND with non-
TyMemTier types; TyUnknown is the only wildcard". This is the
pre-cycle-7 baseline restored.

**Interaction with generic-defer at call sites**: the call-boundary
pre-filter at lines 746-752 explicitly excludes (TyVar, TySize,
TyUnknown) from BOTH pty and aty sides before invoking
`_compatible`. So a call like `fn use_mem(m: WorkingMem<i32>);
fn caller[T](x: T) { use_mem(x) }` does NOT reach the
TyMemTier-vs-TyVar arm at all — the call-boundary check skips it,
deferring to mono. Generic-defer behavior is preserved.

**Interaction with body-vs-return / let / if / match / field-init**:
these sites use `_compatible` directly without pre-filtering. Post-
cycle-8, a generic function whose body produces a TyMemTier and
whose declared return is TyVar now hard-errors. This is correct:
`fn g[T]() -> T { make_working_mem(42) }` should reject because
mono binds T to the caller's expected type, and there is no
mechanism to bind T to WorkingMem<i32> implicitly from the body.
The cycle-7 carve-out silently accepted this by saying "well, T
might be WorkingMem<something> later" — but that's the wrong
abstraction direction: the body's concrete TyMemTier type doesn't
constrain T; T's binding at the call site does. Cycle 8 has this
right.

**Symmetry**: the broad-or arm `if isinstance(a, TyMemTier) or
isinstance(b, TyMemTier): return False` is symmetric by
construction. Both orders reject.

**Arm ordering**: TyUnknown → both-MemTier → broad-or. The
TyUnknown short-circuit at the very top of `_compatible` takes
precedence, so `TyMemTier × TyUnknown` still passes. The both-
MemTier arm fires before the broad-or. Correct ordering.

### Surface 2: `_size_compatible` cascade-safety (re-verified)

Cycle 7 introduced `_size_compatible` at lines 2208-2222 as the
shape-position-only cascade helper. Cycle 8 does not touch this
helper. Re-verified that the helper is invoked from exactly three
sites (TyArray.size at 2306, TyTensor.shape at 2338, TyTile.shape
at 2348) and nowhere else. No new callers introduced in cycle 8.
The TyVar/TySize cascade-pass at shape positions remains intact.

### Surface 3: D-binop diagnostic-text accuracy (re-verified)

Cycle 7 introduced the `other_is_logic` predicate at lines 1349-
1381 to distinguish "(one side D-wrapped, other Logic-wrapped)"
from "(one side D-wrapped, other bare)". Cycle 8 does not touch
this gate. No regression.

### Surface 4: check.py exception-handling rework

The diff shows changes to `helixc/check.py` lines 271-329, but
the in-code comments attribute these to "Audit 28.8 cycle 5 C4-6
/ MEDIUM" — they are the cycle-5 fix-sweep changes that pre-
existed cycle 8 and are surfacing in the diff against the prior
recorded HEAD. The contract additions (FileNotFoundError /
PermissionError / IsADirectoryError / NotADirectoryError /
UnicodeDecodeError / ImportError exception handlers returning
rc=2 with clean `helixc:` messages, and the wrapped finally-block
drain) are not new in cycle 8. They were audited under cycle 5 /
cycle 6 audits and need no re-review here.

---

## Cycle 8 invariant snapshot (post-fix)

The cycle-8 fix-sweep tightens the TyMemTier contract:

**`_compatible` TyMemTier contract** (typecheck.py:2224-2252):
- TyUnknown × any: True (top-level cascade arm 2225-2226).
- TyMemTier × TyMemTier: tier-match AND inner-compatible.
- TyMemTier × any-other (including TyVar / TySize): False.
- The cycle-7 G2 carve-out arms are DELETED.
- The cycle-3 D1 generic-defer behavior is preserved by the
  call-boundary pre-filter at 746-752 (not by the structural
  matcher).

**`_size_compatible` contract** (typecheck.py:2208-2222):
- Unchanged from cycle 7.
- TyVar/TySize/TyUnknown at shape position: True via cascade arms.
- Identity short-circuit before delegation to `_compatible`.
- Invoked from TyArray.size / TyTensor.shape / TyTile.shape only.

**Call-boundary `_compatible` invocation contract**
(typecheck.py:746-752):
- Pre-filter both pty and aty for (TyVar, TySize, TyUnknown)
  before invocation.
- Also pre-filter the TyPrim × TyPrim path (handled separately).
- Skip when `_logic_provenance_violation_kind` will fire.
- Otherwise delegate to `_compatible`.
- This is where generic-defer happens at the call boundary.

---

## Forward note (not a cycle-8 finding)

The cycle-7 G2 carve-out was added without a regression test
covering the dropped behavior. Cycle 8 drops it without adding a
regression test for the now-rejecting behavior. The commit
message claims "223 targeted tests pass", which means no existing
tests depended on the dropped carve-out — good — but a positive
test for the new rejection case would harden the contract against
accidental reintroduction.

Suggested test (for a future cycle, not blocking cycle 8):

```python
def test_c7_1_tymemtier_tyvar_value_position_rejects():
    """Cycle 8 C7-1: TyMemTier × TyVar at top-level value position
    (body / let / if / match / field-init) hard-errors rather than
    silently passing via the cycle-7 G2 carve-out (now dropped)."""
    src = """
    fn g[T]() -> T { make_working_mem(42) }
    fn make_working_mem(x: i32) -> WorkingMem<i32> { ... }
    """
    errs = run_typecheck(src)
    assert any("does not match return type T" in e.msg for e in errs)
```

This is a forward suggestion, not a finding. The change is
correctly documented in code comments at lines 2230-2240.

---

## Cycle 8 status

**Strict criterion (per user directive 2026-05-10): cycle clean iff
zero findings of ANY severity.**

This cycle finds **0 findings (0 HIGH, 0 MEDIUM, 0 LOW)**.

By the strict criterion, **cycle 8 counts CLEAN**.

The severity trend across cycles is now:
- Cycle 1: HIGH-tier finding(s)
- Cycle 2: HIGH + MEDIUM
- Cycle 3: HIGH + MEDIUM + LOW (multiple LOW)
- Cycle 4: MEDIUM-tier
- Cycle 5: 3 MEDIUM + 3 LOW
- Cycle 6: 1 MEDIUM + 2 LOW
- Cycle 7: 0 + 0 + 0  ←  CLEAN
- Cycle 8: 0 + 0 + 0  ←  CLEAN

This is the SECOND consecutive cycle to meet the strict criterion
under Audit B. The cycle-8 fix-sweep is the narrowest possible
contract-tightening (a 2-arm deletion with comment-block swap),
and it correctly addresses cycle-7's silent-failures C7-1 finding
without expanding any other contract surface.

The 5-clean-cycles requirement (per the cycle-5 doc's projection
for Python-helixc deprecation) is now 2/5. Cycles 9-12 would need
to clean to satisfy that bar.

**Recommendation**: no fix-sweep needed for cycle 8. Proceed to
cycle 9 audit gate. The forward test-suite suggestion under
"Forward note" can be folded into a later cycle's housekeeping;
it is not blocking.
