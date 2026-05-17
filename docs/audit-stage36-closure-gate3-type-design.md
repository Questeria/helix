# Stage 36 closure gate-3 type-design audit

**HEAD**: 2a6aedd (commit message names this as the gate-3 / closure
commit; user prompt said `97dbfbc` which is gate-2 prep — auditing
the cumulative diff at `2a6aedd` as instructed by the scope spec
`git diff e7c3552..HEAD -- helixc/`)
**Scope**: git diff e7c3552..HEAD -- helixc/
**Date**: 2026-05-16

## Findings

### H1: parent_right_at(0) silently reads arena[0] — Inc 15 family-uniformity is incomplete on the runtime side (HIGH, confidence 95/100)

**Location**: `helixc/ir/lower_ast.py:2145-2155`
**Pattern**: Asymmetric runtime guard across a family of related builtins; the type-design "uniformity" claim is honored at typecheck but not at the IR lowering layer that matters for the actual silent-read class.

**Evidence** (empirically reproduced; probe deleted, reproduction is two
files):

```hx
fn main() -> i32 {
    let _h = register_derivation(11, 22);
    parent_right_at(0)   // <-- handle=0 (null sentinel)
}
```

Compiling + running this on HEAD yields exit code **11**, not -1
(low byte 255). The lowering computes `base_idx = 0 - 1 = -1`, then
`_safe_arena_get(base_idx, 1)` computes `eff_idx = -1 + 1 = 0`, which
is in-bounds, so it reads `arena[0]` = 11 (the left value of the
previously registered derivation).

Control: `parent_at(0, 1)` on the same program yields exit code **255**
(Inc 15's new `handle <= 0` runtime guard at `lower_ast.py:2192-2195`
short-circuits to -1).

Control: `parent_left_at(0)` is fine because `_safe_arena_get(-1, 0)`
sees `eff_idx = -1`, the `ge_zero` check fails, and the SELECT returns
-1. parent_left_at's safety is incidental (the SUB-1 with offset 0
lands on a negative `eff_idx` that the existing bounds check catches);
parent_right_at's symmetric SUB-1 + offset 1 cancels the safety.

**Impact**:
- The Inc 15 closure narrative says "Family is now uniformly strict-i32"
  (`typecheck.py:2962`) and "defeating the audit's hidden-error #3
  (parent_at(0, 1) silently reading arena[0])" (`lower_ast.py:2173`).
  But the **identical silent-read pattern** survives in `parent_right_at`
  — a 1-arg sibling in the same family. The audit narrative implies
  the silent-read class is closed; it is not.
- Downstream `evidence_right(0)` (stdlib alias, no runtime guard of its
  own) inherits the same silent leak — reproduced empirically with
  the same exit code 11.
- The fix is mechanically obvious and Inc 15 already paid the design
  cost for `parent_at`: a single `handle <= 0 → -1` SELECT guard on
  the result of `parent_right_at`'s lowering. Five extra TIR ops.
- Closure-gate semantics: this is a fresh discovery on gate-3 that
  Inc 15 created the conditions for (by claiming family-uniformity)
  but did not actually close. The pre-Inc-15 code had the same bug;
  Inc 15 perpetuated it while declaring the family fixed.

**Fix**: In `lower_ast.py:2145-2155`, after computing `base_idx`,
guard with `idx > 0` and SELECT to -1 if false. Same pattern Inc 15
already wrote for `parent_at` (`lines 2189-2228`); copy the
`handle_valid` + SELECT-on-failure shape. Cost: ~5 extra TIR ops per
`parent_right_at` callsite. No public-API change. Add the matching
canary test `test_stage36_closure_parent_right_at_null_handle_returns_sentinel`
mirroring the existing Inc 15 `parent_at_null_handle_returns_neg_one_runtime`
test.

(For symmetry, `parent_left_at` should get the same explicit guard
even though its current behavior is accidentally-correct — the guard
makes the family invariant grep-able rather than dependent on
`_safe_arena_get`'s clamp interacting just-right with SUB-1.)

---

### M1: has_evidence false-negative when caller registers a legitimate -1 source ID — doc is silent on the failure mode it actually has (MEDIUM, confidence 80/100)

**Location**: `helixc/stdlib/provenance.hx:30-45`
**Pattern**: Sentinel value collision between user-provided data and
the OOB marker; the type-contract doc explicitly enumerates one
failure mode (false positives) while remaining silent on the
symmetric false-negative mode that the implementation also has.

**Evidence** (empirically reproduced):

```hx
fn main() -> i32 {
    let h = register_derivation(0 - 1, 22);   // user passes -1
    if has_evidence(h) == 0 { 42 } else { 99 }
}
```

Exit code: **42**. The handle `h` is a perfectly valid registered
derivation (1-based handle = 1, arena slots [0,1] = [-1, 22]), but
`has_evidence(h)` returns 0 because `parent_left_at(h)` returns the
user-provided -1, which the predicate (`parent_left_at(h) == -1 then 0
else 1`) cannot distinguish from the OOB sentinel.

The Inc 15 doc rewrite at `provenance.hx:30-39` explicitly calls out
the false-positive direction:

> "NECESSARY-BUT-NOT-SUFFICIENT predicate for the handle to refer to
> a real `register_derivation*` call — the Phase-0 arena has no
> per-handle tag, so a slot whose value happens to be non-(-1) for
> any reason will pass this check."

It says nothing about the false-negative direction: a slot whose value
happens to be exactly -1 (legitimate user data) will FAIL the check.
This contradicts "necessary" — `has_evidence(h)` returning 1 is meant
to be a necessary condition for "h is valid", which means
`has_evidence(h) = 0` should imply "h is not valid", but here a fully
valid h gets 0.

**Impact**:
- A user reading the doc believes `has_evidence(h) == 0` is reliable
  evidence that h is invalid. It is not. They will write defensive
  code that silently drops valid derivations whose first source ID
  happens to be -1.
- This is the same Phase-0 sentinel-collision class the Inc 9 A1
  bounds-check sentinel introduced; Inc 15 had the opportunity to
  document the symmetric hazard in the same doc-tightening pass that
  added the false-positive warning, and missed it.
- The right fix is doc-only at Phase-0 (the underlying ambiguity
  cannot be removed without a per-record arity word, which is the
  deferred Inc 16 work). The wrong fix is to silently change the
  predicate semantics; this would break the Inc 13 canaries.

**Fix**: Extend the `provenance.hx:30-39` doc block with one paragraph:

```
// SECOND failure mode (false-negative): if the caller legitimately
// passes -1 as a source ID (e.g., to mark "no upstream"), the slot
// value collides with the Inc 9 A1 OOB sentinel and has_evidence
// returns 0 even for a fully valid handle. Until the Inc 16 per-
// record arity word lands, callers should avoid -1 as a source ID
// or use direct parent_at/parent_*_at with their own validity tag.
```

No code change. Add one canary test pinning the documented behavior
(value-of -1 source → has_evidence returns 0), so a future predicate
change is noticed.

---

### M2: parent_at typecheck error format diverges from the rest of the strict-i32 family — remediation hint omitted (MEDIUM, confidence 70/100)

**Location**: `helixc/frontend/typecheck.py:3013-3018`
**Pattern**: Asymmetric error message format across a family that
Inc 15 explicitly claims to have unified.

**Evidence**: Compare the four sibling error messages now in the file:

```python
# to_logic_bool   (2922-2927):
"...must be exactly i32, got {t} (pre-Inc-9 also accepted i64/u32/u64
 but those silently truncated in downstream BIT_AND ops)"

# register_derivation   (2945-2952):
"...must be exactly i32 source id, got {t} (pre-Inc-11 also accepted
 i64/u32/u64 but those silently truncated in downstream arena push ops)"

# parent_left_at / parent_right_at   (2965-2972, Inc 15 addition):
"...must be exactly i32 derivation handle, got {t} (pre-Inc-15 also
 accepted i64/u32/u64 but those silently truncated in downstream arena read)"

# register_derivation3   (2982-2990, Inc 14):
"...must be exactly i32 source id, got {t} (Inc 11 C1 family: i64/u32/
 u64 would silently truncate in downstream arena push ops)"

# parent_at   (3013-3018, Inc 14, NOT updated by Inc 15):
"...must be exactly i32, got {t}"        ← no remediation note
```

The `parent_at` arg-type error is the only one in the strict-i32
family without the "pre-Inc-XX also accepted X but silently truncated"
diagnostic. Inc 15 explicitly added that note to `parent_left_at`/
`parent_right_at` for "family uniformity" (`typecheck.py:2956`); the
same family argument applies to `parent_at` and was missed in the
same edit window.

A secondary smell at the same site: the strict-i32 type-error loop
runs to completion before the literal-slot-bounds check, so a caller
who passes both wrong-typed args AND an out-of-range literal slot
gets 3 errors (one per arg + the slot-bounds error) when 2 would
suffice. Reproduced empirically with `parent_at(let h: i64, 5)` —
output: arg-1 strict-i32 error + literal-slot-5-out-of-range error,
even though the slot value is moot when the call won't typecheck.

**Impact**:
- A user passing `i64` to `parent_at` sees a terse error and may not
  know the same input would have compiled pre-Inc-14 with silent
  truncation. The other family members all give them this context.
- Closure-gate hygiene: Inc 15's "family is now uniformly strict-i32"
  narrative is true at the predicate level but observably violated at
  the error-message level. Future readers diffing audits will notice.
- Error noise (the 3-error case) is minor but compounds in test-loop
  iteration speed when fuzzing.

**Fix**: Extend `parent_at` arg-type error (typecheck.py:3014-3016) to
match the sibling format:

```python
self.errors.append(TypeError_(
    f"parent_at(handle, slot): arg {'12'[i]} must be exactly i32, "
    f"got {self._fmt(t)} (Inc 14 has been strict-i32 since landing; "
    f"pre-strict variants in the family silently truncated)",
    expr.span,
))
```

And gate the literal-slot-bounds check on "did the arg types
typecheck" (e.g., `if all type-ok and slot literal out of range`).
Trivial structural change; ~3 lines.

---

### L1: parent_left_at/parent_right_at strict-i32 error remediation note misfires on non-int types (LOW, confidence 75/100)

**Location**: `helixc/frontend/typecheck.py:2965-2972`
**Pattern**: Error-message remediation hint assumes a narrow input
distribution and lies for inputs outside it.

**Evidence**: The new error reads:

> "arg must be exactly i32 derivation handle, got Logic<i32> (pre-Inc-15
> also accepted i64/u32/u64 but those silently truncated in downstream
> arena read)"

Empirically reproduced by passing a `Logic<i32>` to `parent_left_at`.
The "pre-Inc-15 also accepted i64/u32/u64" hint is only true for those
three specific TyPrim types — the pre-Inc-15 `_is_int_scalar` check
did NOT accept `Logic<i32>` (Logic types weren't in
`_NUMERIC_INT_PRIMS`). A user reading this error believes "ah, I used
to be allowed to pass Logic and the code silently truncated" — which is
false. Logic was always rejected.

The same misfire applies to `register_derivation`'s "pre-Inc-11"
note (`typecheck.py:2945-2952`) and `to_logic_bool`'s "pre-Inc-9"
note (`typecheck.py:2923-2927`) — Inc 15 perpetuated the pattern
rather than fixing it.

**Impact**:
- Misleading diagnostic when the user's actual mistake is a wholly
  different type category (Logic, struct, function, etc.).
- Documentation that drifts further from reality with each increment.
- A user filing a bug or asking for help will paste the misleading
  message verbatim.

**Fix**: Gate the remediation hint on `_is_int_scalar(arg_tys[0])` —
i.e., only mention the truncation history when the user actually
passed a wider int that pre-fix would have been silently accepted.
For Logic / struct / etc., show just the bare "must be exactly i32"
message. Three-line change at each of the three sites.

Alternative (smaller, less satisfying): drop the parenthetical
remediation entirely and rely on the audit doc reference instead.

---

### L2: trace_evidence still uses parent_right_at internally — feeds the H1 silent-leak into the documented "honest" trace output (LOW, confidence 65/100)

**Location**: `helixc/stdlib/provenance.hx:96-105`
**Pattern**: Higher-level "fixed" API silently leaks the lower-level
bug it claims to abstract away from.

**Evidence**:

```hx
fn trace_evidence(handle: i32) -> i32 {
    print_str("h=");
    print_int(handle);
    print_str(" slot0=");
    print_int(parent_left_at(handle));   // safe (idx-1 lands negative)
    print_str(" slot1=");
    print_int(parent_right_at(handle));  // <-- H1 silent-leak on h=0
    print_str("\n");
    has_evidence(handle)
}
```

The Inc 15 doc on this function says the relabel from "L= R=" to
"slot0= slot1=" was done "so 3-parent handles aren't silently
mislabelled by the same diagnostic helper" (`provenance.hx:91-93`).
But on `trace_evidence(0)` — the null-handle diagnostic case the
existing test `test_stage36_inc13_trace_evidence_returns_zero_for_null_handle`
exercises — the printed slot1 will be `arena[0]` (= the previous
derivation's left value) rather than -1, because of finding H1.

The existing canary test asserts `"h=0 slot0=-1 slot1=-1\n"` and
passes only because the test runs in isolation with an empty arena.
Once any prior `register_derivation` has happened in the same program,
`trace_evidence(0)` will print `"h=0 slot0=-1 slot1=<leaked-arena-0>\n"`
— a silently-wrong diagnostic.

**Impact**:
- The user-facing "honest trace" output is dishonest in the multi-
  derivation case for null handles. Diagnostic helpers that lie are
  worse than diagnostic helpers that crash.
- The fix for H1 (above) also closes this L2 — they are the same
  underlying bug. Listing L2 separately because it shows the audit
  blast radius beyond the obviously-affected primitive.

**Fix**: Subsumed by H1 fix. If H1 is fixed by adding the
`handle <= 0 → -1` guard inside `parent_right_at`'s lowering, this
function automatically prints `-1` for slot1 when handle=0.

If H1 fix is deferred, add an explicit `if handle <= 0` early-return
inside `trace_evidence` printing the all-sentinels line. Strictly
worse than fixing H1 directly (the leak still affects other
callsites) but at least the documented-honest output stays honest.

---

## Summary

5 findings total: **1 HIGH**, **2 MEDIUM**, **2 LOW**.

**Total findings**: 1 HIGH (parent_right_at runtime silent-leak), 2
MEDIUM (has_evidence false-negative doc gap; parent_at error-format
divergence + over-reporting), 2 LOW (strict-i32 remediation note
misfires on non-int categories; trace_evidence inherits the H1 leak).

**One-paragraph summary**: Inc 15 made a clean, well-documented push
toward family-uniformity in the provenance primitives, but the
unification is incomplete in one materially load-bearing place:
`parent_right_at` and its stdlib alias `evidence_right` still
silently read arena[0] when handed the null sentinel handle 0, while
the freshly-guarded `parent_at(0, 1)` correctly returns -1. This is
the exact silent-failure pattern Inc 15 advertised as closed (cited
verbatim in the lowering comment as "defeating the audit's
hidden-error #3"). Empirically reproduced: handle=0 + a prior
register_derivation produces `parent_right_at(0) == arena[0]` instead
of -1. The Inc 15 design intent (uniform handle<=0 guard across the
family) needs one ~5-line copy from the new parent_at lowering into
the parent_right_at lowering; the cost is trivial relative to the
audit-narrative debt of leaving the asymmetry in place at closure.
Secondary findings cluster around doc/error-message precision: the
new `has_evidence` "necessary-but-not-sufficient" doc rewrite is
asymmetric (warns of false positives but is silent on the
symmetric false-negative when the user legitimately stores -1), the
parent_at typecheck error is the only one in the strict-i32 family
without the family-standard remediation hint, and the new
remediation hints themselves misfire on inputs outside the int
category they were written for.

**Path to doc**:
`C:\Projects\Kovostov-Native\docs\audit-stage36-closure-gate3-type-design.md`
