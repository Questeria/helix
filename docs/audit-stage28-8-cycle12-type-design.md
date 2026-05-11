# Stage 28.8 Pre-29 Audit Gate — Cycle 12, Audit B: Type-System / Dispatch / Soundness

**Date**: 2026-05-11
**Commit**: df825ac (read-only)
**Cycle-10 baseline**: c2e36d4 (last commit that touched any code or
test under `helixc/`).
**Scope**: Re-audit the type-system / dispatch / soundness surface
across `helixc/frontend/typecheck.py`,
`helixc/frontend/monomorphize.py`, `helixc/frontend/struct_mono.py`,
`helixc/frontend/autodiff.py`, and `helixc/check.py` after the
cycle-11 doc-only delta and the cycle-12 verify-stability tick (no
code change).

Cycle 12 is a **docs-only delta** on top of cycle 11. The full delta
from c2e36d4..HEAD is:

```
docs/audit-stage28-8-cycle10-codereview.md      | 589 ++++++++++++++++++++++++
docs/audit-stage28-8-cycle10-silent-failures.md | 516 +++++++++++++++++++++
docs/audit-stage28-8-cycle10-type-design.md     | 314 +++++++++++++
docs/audit-stage28-8-cycle11-codereview.md      | 372 +++++++++++++++
docs/audit-stage28-8-cycle11-silent-failures.md | 469 +++++++++++++++++++
docs/audit-stage28-8-cycle11-type-design.md     | 343 ++++++++++++++
6 files changed, 2603 insertions(+)
```

No production-code file is touched. No test file is touched. Verified
by:

```
git diff c2e36d4..HEAD -- helixc/
(empty — entire helixc tree is byte-identical to c2e36d4)
```

```
git log --oneline c2e36d4..HEAD -- helixc/
(empty — no commits touch helixc since c2e36d4)
```

```
git diff c2e36d4..HEAD -- helixc/frontend/typecheck.py \
                          helixc/frontend/monomorphize.py \
                          helixc/frontend/struct_mono.py \
                          helixc/frontend/autodiff.py \
                          helixc/check.py
(empty — the five scoped files are byte-identical to c2e36d4)
```

The three commits since c2e36d4 are:
- c2e36d4 → 6968755 is the cycle-9 fix-sweep (pre-baseline).
- c2e36d4 itself (cycle-10 tests-only).
- 9685c3a: persist cycle-10 audit docs.
- df825ac: persist cycle-11 audit docs.

Per the user directive, the cycle-12 task **is** this audit; there is
no separate "cycle-12 production change". The counter advances
purely on audit stability.

**Method**:

1. Read `docs/audit-stage28-8-cycle10-type-design.md` and
   `docs/audit-stage28-8-cycle11-type-design.md` to load the most
   recent baseline. Cycle 10 introduced regression tests for C8-1
   and C8-2 (closing C9-1 LOW); cycle 11 was a docs-only persistence
   of the cycle-10 audit reports.
2. Ran `git diff c2e36d4..HEAD --stat`, `git diff c2e36d4..HEAD --
   helixc/`, and `git log --oneline c2e36d4..HEAD -- helixc/` to
   confirm cycle 12 has zero production-code or test delta since the
   cycle-10 baseline.
3. Re-read the four named contract surfaces from the cycle-11 audit
   directly against the bytes at HEAD:
   - `_compatible` TyMemTier strict-separation contract
     (typecheck.py:2248-2276): unchanged. Top-level `_compatible`
     still rejects `TyMemTier × non-TyMemTier` and requires both
     sides MemTier to have matching `.tier` strings + structurally
     compatible `.inner`. The cycle-8 C7-1 carve-out drop is still
     in place (no `TyMemTier × TyVar` cascade leak).
   - `_size_compatible` shape-position cascade
     (typecheck.py:2232-2246): unchanged. Still requires explicit
     `_size_compatible` callee for TyVar / TySize cascade; top-
     level `_compatible` no longer cascades on TyVar (preserving
     the cycle-6 F1-cascade-removal that restored body-vs-return-
     type errors).
   - `_check_call_basic` symmetric `TyVar / TySize / TyUnknown`
     filter (typecheck.py:687-757; filter at 746-752): unchanged.
     Both `pty` and `aty` are excluded from the boundary
     `_compatible` call via `not isinstance(..., (TyVar, TySize,
     TyUnknown))`. The cycle-5 C4-3 fix that introduced symmetric
     `aty` exclusion is preserved.
   - `Monomorphizer.run` iteration order
     (monomorphize.py:433-492): unchanged. Generic fns are NOT
     walked at top level (cycle-5 C4-4 key fix #1); only
     non-generic items + promoted clones are walked; clones are
     promoted into the walk set each pass so nested turbofish
     substitutions get followed (cycle-5 C4-4 key fix #2).
     Fixed-point loop on `changed` flag preserved.
4. Re-read the cumulative cycle-1 through cycle-11 type-design
   findings to confirm no prior-cycle invariant has a latent
   regression pathway. (Cycle 12 introduces no code, so by
   construction it cannot regress any invariant; this is a
   defensive confirmation that the cycle-10 baseline matches the
   bytes at HEAD.)

**Findings summary**: No type-system contract is touched. Re-verified
by inspection of the HEAD-against-baseline diff (empty across the
five scoped files and the helixc tree overall) and direct re-read of
the four named contract surfaces.

**Result**: **0 findings (0 HIGH, 0 MEDIUM, 0 LOW)**. Cycle 12 is a
pure verify-stability tick. No new audit doc adds any production-code
or test surface; no type-system surface is added, modified, or
removed. The strict criterion ("zero findings of any severity at
confidence ≥ 80") is **MET**.

---

## Cycle 11 finding re-verification

| ID | Severity prev | Audit (prev) | Status | Notes |
|---|---|---|---|---|
| — | n/a | type-design (cycle 11) | n/a (was CLEAN) | Cycle 11 was CLEAN; no findings to re-verify under this audit category. The cycle-10 forward notes (empty-string, nested-prefix, leading-whitespace edge cases for `_emit_env_error`, plus the contributor-style raise-prefix convention codification) carry over unchanged through cycle 11 and remain non-blocking in cycle 12. |

No prior-cycle type-design findings need re-verification. Cycle 11
type-design was CLEAN and cycle 12 does not touch any production-code
or test surface.

---

## Per-surface review (cycle-12 touchpoints)

Cycle 12 has no production-code or test touchpoints. The directive
itself describes cycle 12 as a "verify-stability" tick: re-run the
audit against an unchanged codebase to advance the strict-criterion
counter. By construction, there are no per-surface changes to review.

For completeness, the only file-system delta possible in cycle 12 is
the writing of three audit docs (codereview, silent-failures, this
one) into `docs/`. Audit docs are not a code surface and have no
type-design implication.

---

## Spot-check: re-read the four named contract surfaces

### Surface 1: `_compatible` TyMemTier strict-separation
**Location**: `helixc/frontend/typecheck.py:2248-2276`.

The contract:
- `TyUnknown × *` → True (defer).
- `TyMemTier × TyMemTier` → require `.tier` string equality AND
  recursive structural `_compatible` on `.inner`.
- `TyMemTier × non-TyMemTier` → False (hard reject; no cascade for
  TyVar / TySize at this position).

**Status against cycle-10 baseline**: byte-identical (empty `git
diff` over the file). The cycle-8 C7-1 carve-out drop is preserved
(no carve-out for `TyMemTier × (TyVar | TySize)` at top-level). The
cycle-5 F4 / MEDIUM deferred-enhancement comment about tier
subsumption (raw string equality, no HBM ⊆ DDR matrix) is preserved
as-is — still a deferred enhancement, not a finding.

### Surface 2: `_size_compatible` shape-position cascade
**Location**: `helixc/frontend/typecheck.py:2232-2246`.

The contract:
- `TyVar × *` or `* × TyVar` at size position → True (cascade).
- `TySize × *` or `* × TySize` → True.
- `TyUnknown × *` → True.
- `a == b` → True.
- Else fall through to `_compatible` (which does NOT cascade on
  TyVar at top level — that's the cycle-6 F1 / C5-1 fix).

**Status against cycle-10 baseline**: byte-identical. The
shape-position-only cascade boundary introduced by cycle-7 C6-1 is
preserved. The docstring explanation that body-position cascades
correctly emit "body type i32 does not match return type T" is
unchanged.

### Surface 3: `_check_call_basic` symmetric filter
**Location**: `helixc/frontend/typecheck.py:687-757` (filter at
lines 746-752).

The contract: at the call boundary, the general `_compatible` check
fires only when **both** sides are non-TyVar, non-TySize,
non-TyUnknown AND not both TyPrim (TyPrim-vs-TyPrim is handled by
the earlier name-equality arm) AND no Logic-provenance specialized
diagnostic will fire.

**Status against cycle-10 baseline**: byte-identical. The cycle-5
C4-3 symmetric `aty` filter is in place (both `pty` AND `aty`
checked for `(TyVar, TySize, TyUnknown)` exclusion). The pre-fix
asymmetric-pty-only filter, which would have caused a false-positive
`expects i32, got T` on the canonical
`fn use_x[T](v: T) -> i32 { check_x(v) }` generic-adapter pattern,
is not present.

### Surface 4: `Monomorphizer.run` iteration order
**Location**: `helixc/frontend/monomorphize.py:433-492`.

The contract:
- Generic fns (with `.generics`) are NOT walked at top level
  (cycle-5 C4-4 key fix #1).
- Non-generic items + promoted clones are walked; promoted clones
  are tracked in a separate list to avoid re-promoting.
- New clones get promoted into the walk set each pass so nested
  turbofish substitutions get followed across iterations (cycle-5
  C4-4 key fix #2).
- Fixed-point loop terminates when `changed` stays False across a
  full pass.
- After the loop, instantiated clones are appended to
  `prog.items`; original generic fns are kept intact so legacy
  un-turbofished call sites still resolve via the lower path.

**Status against cycle-10 baseline**: byte-identical. Both cycle-5
C4-4 key fixes are present. The "generic fns kept intact post-mono
for legacy lower path" backward-compatibility note in the docstring
is unchanged.

---

## Other surfaces (re-verified, not touched in cycle 12)

### typecheck.py (cycles 1-11)
Cycle 12 does not modify `helixc/frontend/typecheck.py`. All
prior-cycle invariants preserved by the empty diff:

- `_compatible` TyMemTier strict-separation contract.
- `_compatible` TyQuote / TyDiff / TyLogic kind+inner check.
- `_size_compatible` shape-position cascade.
- `_check_call_basic` symmetric `(TyVar, TySize, TyUnknown)` filter.
- `_check_call_basic` Logic-provenance B:C10 batching.
- D-binop diagnostic-text accuracy.

### monomorphize.py (cycles 1-11)
Cycle 12 does not modify `helixc/frontend/monomorphize.py`. All
prior-cycle invariants preserved by the empty diff:

- `Monomorphizer.run` iteration order (non-generic + promoted only).
- Clone-promotion-each-pass for nested turbofish.
- Original-generic-fn retention for legacy lower path.

### struct_mono.py (cycles 1-11)
Cycle 12 does not modify `helixc/frontend/struct_mono.py`. All
prior-cycle invariants preserved by the empty diff.

### autodiff.py (cycles 1-11)
Cycle 12 does not modify `helixc/frontend/autodiff.py`. All
prior-cycle invariants preserved by the empty diff.

### check.py (cycles 1-11)
Cycle 12 does not modify `helixc/check.py`. The `_emit_env_error`
helper contract and the `main()` outer-dispatch classifier contract
are unchanged from cycle 9 (CLEAN) / cycle 10 (CLEAN) / cycle 11
(CLEAN).

---

## Cumulative invariant snapshot (post-cycle-12)

No new invariants introduced. The cycle-10 invariant snapshot
(itself unchanged from cycle 9 and re-verified through cycles 11 and
12) remains authoritative. For completeness, the authoritative
contracts as of HEAD:

**`_compatible`** (typecheck.py:2248-2276):
- `TyUnknown × *` → True.
- `TyMemTier × TyMemTier` → tier eq + structural inner.
- `TyMemTier × non-TyMemTier` → False (no carve-out).
- `TyQuote × TyQuote` → structural inner; `TyQuote × non-TyQuote`
  → False.
- `TyDiff × TyDiff` → structural inner; `TyDiff × non-TyDiff` →
  False.
- `TyLogic × TyLogic` → structural inner; `TyLogic × non-TyLogic`
  → False (specialized provenance diagnostic fires elsewhere).
- Remaining cases delegate to per-kind matchers.

**`_size_compatible`** (typecheck.py:2232-2246):
- TyVar / TySize / TyUnknown at either side → True (size-position
  cascade).
- Else fall through to `_compatible`.

**`_check_call_basic`** (typecheck.py:687-757):
- Arity check first.
- TyPrim × TyPrim → name equality (with `size_N` loose handling).
- Else, when both sides are non-(TyVar, TySize, TyUnknown), not
  both TyPrim, and no Logic-provenance violation, defer to
  `_compatible`.
- Logic-provenance violations collected and batched (B:C10) into a
  single grouped diagnostic when 2+ params violate.

**`Monomorphizer.run`** (monomorphize.py:433-492):
- Walk non-generic items + promoted clones only.
- Promote new instantiations each pass; iterate until fixed point.
- Append clones to `prog.items` post-loop; keep generic fns intact.

---

## Cycle 12 status

**Strict criterion (per user directive 2026-05-10): cycle clean iff
zero findings of ANY severity at confidence ≥ 80.**

This cycle finds **0 findings (0 HIGH, 0 MEDIUM, 0 LOW)**.

By the strict criterion, **cycle 12 counts CLEAN**.

**Counter status (5-clean-consecutive gate under the strict
criterion, per user directive 2026-05-10)**: was **2/5** after cycle
11. With cycle 12 also CLEAN, the counter advances to **3/5**. Two
more clean cycles (13, 14) are required before Stage 29 can proceed
(drop Python helixc).

The severity trend across cycles, against the strict-criterion bar:
- Cycle 1: HIGH-tier finding(s) — not clean
- Cycle 2: HIGH + MEDIUM — not clean
- Cycle 3: HIGH + MEDIUM + LOW (multiple LOW) — not clean
- Cycle 4: MEDIUM-tier — not clean
- Cycle 5: 3 MEDIUM + 3 LOW — not clean
- Cycle 6: 1 MEDIUM + 2 LOW — not clean
- Cycle 7: 0 + 0 + 0 — pre-directive era; CLEAN under loose
  criterion only
- Cycle 8: 0 + 0 + 0 — same
- Cycle 9: 0 + 0 + 0 — same
- Cycle 10: 0 + 0 + 0 — **first clean cycle counted under strict
  criterion** per user directive 2026-05-10 → counter 1/5
- Cycle 11: 0 + 0 + 0 — CLEAN → counter 2/5
- Cycle 12: 0 + 0 + 0 — CLEAN → counter **3/5**

**Recommendation**: no fix-sweep needed for cycle 12. Proceed to
cycle 13 audit gate.

---

## Forward notes (not cycle-12 findings)

Carried forward unchanged from cycle 11 (themselves carried from
cycle 10). None are blocking.

1. **Empty-string edge case for `_emit_env_error`**: no test asserts
   `_emit_env_error("")` produces `helixc: ` (and remains stable
   across refactors). No production callee passes empty. Not
   blocking.

2. **Nested-prefix edge case for `_emit_env_error`**: no test
   asserts `_emit_env_error("helixc: helixc: foo")` strips exactly
   one layer. No production callee produces nested prefixes. Not
   blocking.

3. **Whitespace-handling edge case for `_emit_env_error`**: no test
   asserts `_emit_env_error("   helixc: foo")` produces a single-
   prefix output. Not blocking.

4. **Convention note for raise-message prefix**: a contributor-style
   doc could codify the implicit cycle-9 contract (callees MAY
   include a single `helixc:` prefix; MUST NOT nest). Not blocking.

5. **Cycle-13 baseline confirmation**: cycles 10, 11, and 12 have
   all been doc/tests-only commits, and cycle 12 has been entirely
   docs-only (no test addition). If cycle 13 is also doc-only, the
   counter will reach 4/5 without ever exercising the production
   typecheck/mono path under a fresh code delta. This is fine under
   the strict criterion (zero findings is zero findings regardless
   of code delta), but if a non-trivial production change lands
   between cycles 12 and 13, the next audit needs to give the diff
   a full read rather than relying on the empty-diff shortcut used
   in cycles 10, 11, and 12. Not a finding — process note for
   future audit runs. (This is the same forward note carried from
   cycle 11; the cycle number in the note shifts each cycle.)

6. **Audit-cadence observation**: three consecutive docs-only or
   tests-only cycles (10, 11, 12) raises the question of whether
   the audit cadence is now exercising stability rather than fault-
   finding. The strict criterion measures stability by design, so
   this is the expected steady-state once the type system reaches a
   fixed point. The remaining counter advances (13, 14) will either
   confirm a true fixed point or surface a regression if any
   production code lands. Not a finding — process observation.
