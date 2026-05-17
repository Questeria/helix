# Stage 40 Inc 4 closure gate-3 type-design audit

**HEAD**: `d914557` (Stage 40 Inc 4 closure gate-2 fix sweep)
**Base**: `0aea911` (Stage 39 CLOSED) — Stage 40 surface = `cb36bbc..d914557`
**Gate-2 fix-sweep surface**: `e8fb593..d914557`
**Scope**: `helixc/frontend/typecheck.py` (132-line gate-2 delta; only
typecheck.py changed — `autodiff.py` + `lower_ast.py` untouched),
`helixc/tests/test_stage40_modal.py` (+227-line gate-2 backfill),
`helixc/examples/dogfood_13_modal_lifecycle.hx`.
**Date**: 2026-05-17
**Auditor brief**: type-design lens on the gate-2 delta + the
re-examined Stage 40 surface — looking for invariant inconsistency,
encoded-as-convention smells, asymmetric guards, and any NEW design
issues the gate-2 fix sweep itself introduced (meta-question). Gate-3
strictness: anything below HIGH conf 80 → OBS. Cycle counter: 0/3.
CLEAN target. Reference template: `docs/audit-stage40-inc4-gate2-type-design.md`.
**Prior-gate context** (do NOT re-flag): F1 generalize to cross-modal,
F2 user-fn shadow + `_shadowed_builtin_names` dispatch suppression,
M1 TyUnknown skip on F1 guard, eager-init of shadow set, F1
named-binding bypass (pinned + deferred to Phase-1 taint pass).

## VERDICT: 1 HIGH, 0 MEDIUM, 0 LOW, 4 OBS

---

## Summary table

| ID | Sev | Conf | Title |
|----|-----|------|-------|
| H1 | HIGH | 86 | The H2 dispatch-suppression invariant is asymmetric: gate-2 fixed the OUTER bare-name dispatch but the F1 cross-modal launder guard still dispatches off the INNER call's syntactic name without checking `_shadowed_builtin_names` — a user fn `fn from_uncertain(x: i32) -> i32` triggers a false-positive `launders a Uncertain<T> into Known<T>` diagnostic even though no `Uncertain<T>` ever existed in the program. Same invariant gap symmetrically present for all 4 `from_*` eliminators |
| OBS-A | OBS | 78 | `_modal_upgrade_hint` is a third dispatch table parallel to `_modal_transitions` + `_modal_elim_kind` with no machine-checkable link between them — adding a future Phase-1 `("uncertain", "believed"): "accept_uncertain"` requires touching 3 sites (`_modal_transitions`, `_BUILTIN_NAMES`, `_modal_upgrade_hint`); the F1 fallback hint would silently become a lie ("Phase-0 has no X -> Y transition") for the new arrow until a maintainer remembers all three sites |
| OBS-B | OBS | 75 | F2 shadow-diagnostic body is stale post-H2: the diagnostic still says "the user definition is unreachable from any call site that uses the bare name" — but the gate-2 H2 fix flipped exactly that invariant (the user-fn IS now reachable via `<<shadowed_builtin_skip>>` fall-through). Diagnostic text drift makes the error message a lie about its own framework's behavior |
| OBS-C | OBS | 72 | F1 diagnostic body hardcodes `<T>` placeholder; for cross-stage composition the actual inner is informative (`Believed<Past<i32>> -> Known<Past<i32>>` reports as "Believed<T> into Known<T>"). The lossy diagnostic costs the user the one piece of context that distinguishes "I forgot a transition" from "I have a cross-stage modeling bug" |
| OBS-D | OBS | 65 | All 4 wrapper types (TyMemTier/TyFrame/TyTemporal/TyModal) use a `str` discriminator instead of an `Enum`; the Stage 40 surface inherits this and uses string-literal tuples `("believed", "known")` as dict keys. Typo-tolerant smell. NOT a Stage 40 regression — Stage 37/38/39 share the pattern. Documented for cross-stage Phase-1 cleanup pass |

---

## Findings

### H1: F1 cross-modal launder guard ignores `_shadowed_builtin_names` for the INNER call → false-positive cascade (HIGH, confidence 86/100)

**Location**: `helixc/frontend/typecheck.py:3550-3585` (F1 cross-modal
guard predicate inside `_modal_intro` dispatch arm).

**Pattern**: The gate-2 H2 fix established the invariant "when a user
fn shadows a reserved builtin, the call dispatch falls through to
user-fn lookup; the user sees ONE shadow diagnostic, not 1 + N
noise." H2 implements this at the OUTER bare-name dispatch site via
the sentinel rebind:

```py
# typecheck.py:2854-2863
if isinstance(expr.callee, A.Name):
    bn = expr.callee.name
    # Stage 40 closure gate-2 H2 fix: ...
    if bn in self._shadowed_builtin_names:
        bn = "<<shadowed_builtin_skip>>"
```

But the F1 cross-modal launder guard (added by gate-2's F1
generalization) inspects `expr.args[0].callee.name` — the INNER
call's syntactic name — to decide whether to fire the laundering
diagnostic, and does NOT consult `_shadowed_builtin_names` for the
inner name:

```py
# typecheck.py:3550-3557
if (len(expr.args) >= 1
        and isinstance(expr.args[0], A.Call)
        and isinstance(expr.args[0].callee, A.Name)
        and expr.args[0].callee.name
            in _modal_elim_kind
        and not isinstance(arg_tys[0], TyUnknown)):
    source_kind = _modal_elim_kind[
        expr.args[0].callee.name]
```

The gate-2 M1 fix added `not isinstance(arg_tys[0], TyUnknown)` to
suppress F1 when the inner produced its own diagnostic. But a
user-fn that successfully type-checks (e.g., returns `i32` not
`TyUnknown`) sneaks past M1: the inner call evaluates as a normal
user-fn invocation returning a concrete type, but the F1 guard reads
the inner's *syntactic name* and concludes "the user wrote
`from_uncertain` so this must be Uncertain laundering" — false.

**Reproduction** (live probe against `d914557`):

```py
src = """fn from_uncertain(x: i32) -> i32 { x }
fn main() -> i32 {
    let k: Known<i32> = into_known(from_uncertain(42));
    from_known(k)
}"""
errs = typecheck(parse(src, include_stdlib=True))
# 2 errors:
#   1:1: function 'from_uncertain' shadows a reserved builtin name
#   3:25: into_known(from_uncertain(...)) launders a Uncertain<T>
#         into Known<T> with no epistemic-upgrade audit.
```

The second diagnostic is a false positive: no `Uncertain<T>` value
ever existed in this program. The user's `from_uncertain` is a
benign `i32 -> i32` passthrough. The F1 guard misreads it as the
builtin Uncertain eliminator because it consults the SYNTAX (the
identifier text) rather than the SEMANTICS (whether that identifier
resolves to the builtin or a shadowed user-fn).

**Why HIGH not MEDIUM**: this is an invariant-consistency gap, not a
cosmetic bug. The gate-2 H2 fix declared a precise discipline ("one
shadow diagnostic, zero cascading noise"). The H2 audit explicitly
counted "1 shadow + 0 noise" as the success condition. The F1 guard
silently violates the H2 invariant at one of its callsites. A
type-design audit at gate-3 strictness MUST flag this — the
invariant was named, the test suite asserts it for one path, the
other path quietly admits the cascade.

The bug is symmetric across all 4 `from_*` eliminators: any user fn
that happens to share a name with one of `from_known`,
`from_believed`, `from_goal`, `from_uncertain` will trigger the same
false-positive cascade whenever a call site happens to write
`into_X(from_Y(...))`. The probability of a real program hitting
this is non-zero — `from_uncertain` and `from_known` are extremely
generic names that user code might plausibly want for unrelated
purposes (e.g., a state-machine `from_uncertain(state)` that
transitions out of a generic "uncertain" application state).

**Why not raise to BLOCKING**: the cascade is bounded (it fires at
most once per `into_X(from_Y(...))` call site), the shadow diagnostic
itself still fires correctly so the user IS told to rename, and the
F1 false-positive's hint ("Phase-0 has no Uncertain -> Known
transition") at least points at the right family of concepts. The
user is not silently misled in a security-critical way — they get
two diagnostics where the right answer is one.

**Why this is a gate-2-introduced issue not a gate-1 one**: gate-1
F1 only fired on `from_uncertain` as inner (single-source pattern).
Gate-2 generalized to all 4 inners (`_modal_elim_kind` dict) AND
gate-2 introduced the H2 dispatch-suppression set. The two gate-2
changes together create the asymmetry: H2 added the suppression
mechanism, F1-generalized added the new consumer, but the new
consumer doesn't apply the suppression. The gate-2 type-design
audit did not catch this because the audit considered F1 and H2
independently rather than as interacting subsystems. This is
precisely the "meta-question" the gate-3 brief Q5 asks about — yes,
the gate-2 fix sweep introduced a new type-design issue its own
audit missed.

**Concrete consequences if left as-is**:

- Any user who happens to rename one of the 4 modal eliminators
  (legitimate when migrating from non-Stage-40 code, or when the
  user's domain happens to use the word "uncertain" for something
  unrelated) gets a false-positive launder diagnostic on every
  cross-modal call site that uses the bare name as inner. The user
  reads the diagnostic, sees "Uncertain<T> into Known<T>", and is
  led to investigate a non-existent epistemic-upgrade issue rather
  than the actual shadow problem.
- The H2 invariant promise ("1 shadow + 0 noise") is materially
  weakened. The gate-2 H2 test
  (`test_stage40_gate2_h2_shadowing_emits_one_diagnostic_not_three`)
  passes because it tests the OUTER-dispatch path; it does not test
  the F1-guard-as-secondary-consumer path.
- A future stage that adds a new from-style eliminator (say, an
  arena-style `from_arena`) would need to remember to apply the
  `_shadowed_builtin_names` check at every guard that consumes the
  inner syntactic name. The pattern is not centralized — each
  consumer must remember the discipline independently.

**Fix options** (in increasing scope):

1. **One-line predicate addition** in the F1 guard (recommended):

   ```py
   if (len(expr.args) >= 1
           and isinstance(expr.args[0], A.Call)
           and isinstance(expr.args[0].callee, A.Name)
           and expr.args[0].callee.name in _modal_elim_kind
           and expr.args[0].callee.name           # <-- new
               not in self._shadowed_builtin_names  # <-- new
           and not isinstance(arg_tys[0], TyUnknown)):
   ```

   Mechanically isomorphic to the OUTER `bn` skip; just applied to
   the inner callee name. ~2 lines. No behavior change for the
   non-shadowed paths.

2. **Centralize the suppression check** in a helper that all
   consumers of `expr.callee.name` use uniformly. Reduces the
   discipline-distribution surface for future stages. Larger
   refactor; defer to Phase-1.

3. **Add a regression test** pinning the post-fix behavior:

   ```py
   def test_stage40_gate3_f1_inner_name_shadow_suppresses_guard():
       src = '''fn from_uncertain(x: i32) -> i32 { x }
       fn main() -> i32 {
           let k: Known<i32> = into_known(from_uncertain(42));
           from_known(k)
       }'''
       errs = typecheck(parse(src, include_stdlib=True))
       launder_errs = [e for e in errs if "launders" in str(e)]
       assert launder_errs == [], (
           "H2 invariant must apply to F1 inner-name dispatch too")
       shadow_errs = [e for e in errs
                      if "shadows a reserved builtin" in str(e)]
       assert len(shadow_errs) == 1
   ```

The recommended discharge is (1) + (3): one-line fix + one
regression test. Gate-3 closure on this finding is straightforward.

---

## OBS-A: `_modal_upgrade_hint` is a third dispatch table parallel to `_modal_transitions` with no machine-checkable link (OBS, confidence 78/100)

**Location**: `helixc/frontend/typecheck.py:3534-3541` vs `:3620-3623`.

The gate-2 F1 generalization added a NEW lookup table specifically
for the hint text:

```py
_modal_upgrade_hint = {
    ("believed", "known"):
        "use `confirm(b)` — the audited "
        "Believed -> Known epistemic upgrade",
    ("goal", "known"):
        "use `act_on(g)` — the audited "
        "Goal -> Known epistemic upgrade",
}
```

This table is logically dependent on `_modal_transitions`:

```py
_modal_transitions = {
    "confirm": ("believed", "known"),
    "act_on":  ("goal",     "known"),
}
```

The two encode the same information from different directions
(`_modal_transitions` is verb→(src,dst); `_modal_upgrade_hint` is
(src,dst)→hint), with no machine-checkable link. If a future Phase-1
stage adds a third transition `accept_uncertain: Uncertain -> Believed`,
the maintainer must touch THREE sites:

1. `_modal_transitions["accept_uncertain"] = ("uncertain", "believed")`
2. `_BUILTIN_NAMES` += `{"accept_uncertain"}` (frozen set at line 1966)
3. `_modal_upgrade_hint[("uncertain", "believed")] = "use ..."`

Forgetting (3) is the silent-failure mode: the F1 fallback hint
would fire for the new direction with text "Phase-0 has no Uncertain
-> Believed transition" — a lie, because Phase-1 DOES now have
that transition. The user reads the hint, concludes the direction is
unsupported, and writes a workaround instead of using the new
verb.

**Why OBS not LOW**: this is a maintenance-trap, not an active bug
today. The pattern is straightforwardly extensible AS LONG AS the
maintainer remembers all three sites. The L1 finding in the gate-2
type-design audit flagged the analogous issue for the F1
launder-target set; this is the same anti-pattern in a different
direction. Gate-3 strictness (HIGH conf 80) puts this below the bar
because the failure mode requires a future-stage extension to
manifest, and the existing tests would catch the wrong-hint
regression IF a transition-specific test asserted the hint text.

**Recommended improvement** (defer to Phase-1, low priority):
derive `_modal_upgrade_hint` keys from `_modal_transitions.values()`:

```py
# Compute hint table by inverting _modal_transitions so adding a
# new entry to _modal_transitions automatically populates the hint
# table (verb name extracted from the transitions dict key).
_modal_upgrade_hint = {
    direction: f"use `{verb}(v)` — the audited "
               f"{direction[0].capitalize()} -> "
               f"{direction[1].capitalize()} epistemic upgrade"
    for verb, direction in _modal_transitions.items()
}
```

3 lines, single source of truth, no risk of the two tables drifting.
The downside is hint text becomes mechanically uniform (no
per-direction wording tuning) — minor, since the current two hints
are already mechanically uniform up to verb substitution.

---

## OBS-B: F2 shadow-diagnostic body is stale post-H2 — claims user-fn is "unreachable" but H2 made it reachable (OBS, confidence 75/100)

**Location**: `helixc/frontend/typecheck.py:920-933` (F2 shadow
diagnostic body) vs `:2856-2863` (H2 fall-through behavior).

The F2 diagnostic body (added gate-1, unchanged at gate-2) reads:

```py
self.errors.append(TypeError_(
    f"function {fn.name!r} shadows a reserved builtin "
    f"name; rename the function to avoid silent dispatch "
    f"dead-coding (the typechecker resolves the builtin "
    f"first, so the user definition is unreachable from "
    f"any call site that uses the bare name)",
    fn.span,
    hint=...,
))
```

The parenthetical claim "the user definition is unreachable from any
call site that uses the bare name" was TRUE pre-H2 (gate-1
behavior). The gate-2 H2 fix explicitly inverted this: post-H2, the
typechecker rewrites `bn` to `<<shadowed_builtin_skip>>` and falls
THROUGH to user-fn lookup at line 3704 (`if isinstance(expr.callee,
A.Name) and expr.callee.name in self.functions`). The user
definition IS now reachable from any bare-name call site — verified
live: a user `fn confirm(s: bool) -> i32` post-shadow successfully
dispatches to the user-fn when called as `confirm(true)`.

The F2 diagnostic now lies about its own framework's behavior. A
careful user reading the diagnostic would conclude "my code is
dead-coded" and either (a) rename the fn (which IS the right action,
so the harm is bounded) or (b) be confused when the dead-coded fn
actually runs and produces output.

**Why OBS not LOW**: the directional outcome (rename the fn) is
correct regardless of diagnostic accuracy — the user-fn IS reachable
but the typechecker still rejects the program at the shadow check, so
the program never reaches IR generation. No silent runtime
divergence. The lie is only about the explanation, not the behavior.

**Why flagged at gate-3**: type-design discipline says diagnostic
text is part of the user-facing type system contract. A diagnostic
that contradicts framework behavior trains the user to distrust
diagnostic explanations — a corrosive cost over time. Cheap to fix.

**Recommended re-phrasing** (mechanical, ~3 lines):

```py
f"function {fn.name!r} shadows a reserved builtin name. "
f"Stage 40 gate-2 suppresses builtin dispatch at shadowed "
f"call sites (the user-fn IS reached) but the shadow itself "
f"is rejected here to prevent silent dispatch drift across "
f"refactors; rename the function to use the builtin freely",
```

Or simpler: drop the parenthetical entirely and keep only the
imperative rename hint.

---

## OBS-C: F1 launder diagnostic hardcodes `<T>` placeholder; loses inner-type information for cross-stage composition (OBS, confidence 72/100)

**Location**: `helixc/frontend/typecheck.py:3574-3585` (F1 error
construction).

The diagnostic body uses literal `<T>` placeholders:

```py
self.errors.append(TypeError_(
    f"{bn}(from_{source_kind}(...)) "
    f"launders a "
    f"{source_kind.capitalize()}<T> "
    f"into "
    f"{target_kind.capitalize()}<T> "
    f"with no epistemic-upgrade audit.",
    ...))
```

For raw-inner programs this is fine: `Believed<i32> -> Known<i32>`
reports as `Believed<T> into Known<T>` — the `<T>` is generic enough
to read.

For cross-stage composition (Stage 40's stated advertised use case
per `TyModal` docstring: "Composes with TyTemporal: `Known<Past<i32>>`
= 'I directly observed this past fact'"), the diagnostic loses the
composition information:

```
src = """fn main() -> i32 {
    let p: Past<i32> = into_past(42);
    let bp: Believed<Past<i32>> = into_believed(p);
    let kp: Known<Past<i32>> = into_known(from_believed(bp));
    from_past(from_known(kp))
}"""
# -> 4:32: into_known(from_believed(...)) launders a Believed<T> into Known<T>
```

The user is dealing with `Believed<Past<i32>>` (an inferred past
fact being miscategorized as an observed past fact — a meaningful
modeling distinction) but the diagnostic reports `Believed<T>`. The
user must mentally substitute `T = Past<i32>` to understand whether
the launder is in the modal layer (which it is) or the temporal
layer (which it isn't).

**Why OBS not LOW**: the diagnostic is correct in the modal-layer
sense — F1 IS a modal-layer guard, and `T` IS opaque to the guard's
reasoning. The `<T>` is honest about what F1 inspects (only the
modal kind, regardless of inner). The user CAN reconstruct the full
type from the spans + the surrounding `let` declarations.

**Why flagged at gate-3**: the Stage 40 OPENS commit body markets
cross-stage composition as a headline feature. Diagnostic quality is
part of the user contract on a headline feature. A 1-line fix using
`self._fmt(arg_tys[0])` would report the actual inner type
information for both source and target.

**Recommended improvement** (1-line, optional):

```py
inner_ty = self._fmt(arg_tys[0].inner) if (
    isinstance(arg_tys[0], TyModal)) else "T"
# ... then use {inner_ty} in the diagnostic body
```

The fallback to `"T"` preserves current behavior when the M1 guard
already accepted the inner as a well-typed modal-wrapped value.

---

## OBS-D: All 4 wrapper types use `str` discriminators; tuple-string keys in `_modal_upgrade_hint`; not a Stage-40 regression (OBS, confidence 65/100)

**Location**: `helixc/frontend/typecheck.py:232-280` (4 wrapper
dataclasses).

All four Stage-37-onwards wrapper types use `str` discriminator
fields:

- `TyMemTier.tier: str` (line 237) — `"working" / "episodic" / "semantic" / "procedural"`
- `TyFrame.frame: str` (line 250) — `"world" / "robot" / "camera"`
- `TyTemporal.kind: str` (line 263) — `"past" / "present" / "future" / "eternal"`
- `TyModal.kind: str` (line 279) — `"known" / "believed" / "goal" / "uncertain"`

The valid-value docstring lives ONLY in the comment; nothing at the
type level prevents `TyModal(kind="madeup", inner=...)`. The
gate-2-added `_modal_upgrade_hint` further uses tuple-string keys
(`("believed", "known")`) — typos at the dict-construction site
silently produce dead entries that never match at runtime.

A typed alternative would be `Literal["known", "believed", "goal",
"uncertain"]` or an `Enum`-backed discriminator, which would
mechanically prevent the typo class of bugs.

**Why OBS not LOW**: this is a PRE-STAGE-40 pattern shared by every
wrapper family. Stage 40 inherits it unchanged. The gate-3 brief Q3
explicitly asks "should it be tightened? (note as OBS if you flag
it, since it's not a Stage 40 regression)" — flagging accordingly.

**Why low confidence (65)**: in practice, the dicts that USE these
discriminators (`_modal_intro`, `_modal_elim_kind`,
`_modal_transitions`) are co-located with the discriminator's
canonical-string set, and tests cover the round-trips for each. A
typo would land in a failing test rather than at runtime. The
encoded-as-convention pattern has held for 4 stages without a
discriminator-typo bug landing in production. Cleanup is welcome
but not urgent.

**Recommended improvement** (Phase-1, cross-stage cleanup pass):

```py
from typing import Literal

ModalKind = Literal["known", "believed", "goal", "uncertain"]

@dataclass(frozen=True)
class TyModal(Type):
    kind: ModalKind
    inner: Type
```

Mypy/pyright would catch construction-site typos at lint time. Test
suite unchanged. Defer to a coordinated 4-wrapper-family cleanup
rather than a Stage-40-only patch.

---

## Cross-cutting observation: gate-3 type-design surface is one inner-shadow guard away from CLEAN

The gate-3 type-design audit finds the gate-2 fix sweep introduced
exactly ONE new type-design issue (H1) — the H2 invariant is
asymmetric across its consumers. The gate-2 audit considered F1 and
H2 in isolation and missed the interaction. The fix is one-line +
one regression test.

The remaining items are observational:

- **OBS-A**: The 3-table dispatch (`_modal_transitions` +
  `_modal_elim_kind` + `_modal_upgrade_hint`) has a Phase-1
  extension-trap. Mechanical fix (derive hint table from
  transitions table). Defer.
- **OBS-B**: F2 diagnostic body is stale post-H2 (claims
  unreachable when post-fix the user-fn IS reachable). Cosmetic but
  trains user distrust. Cheap rephrase.
- **OBS-C**: F1 diagnostic uses `<T>` placeholder; loses inner-type
  information for advertised cross-stage composition use case. 1-line
  fix using `self._fmt`.
- **OBS-D**: String-discriminator anti-pattern shared across all 4
  wrapper families. Defer to coordinated cleanup.

If H1 is addressed (1-line predicate addition + 1 regression test,
~10 lines total), the gate-3 type-design audit closes CLEAN.

## Recommended priority for Stage 40 closure gate-3

1. **Must-fix before Stage 40 closes (gate-3 invariant-discipline)**:
   H1's inner-name shadow check + regression test. ~10 lines.
2. **Should-fix before closes (cheap cosmetic)**: OBS-B F2
   diagnostic re-phrasing. ~3 lines.
3. **Nice-to-have**: OBS-C inner-type in F1 diagnostic. ~2 lines.
4. **Defer to Phase-1**: OBS-A hint-table derivation, OBS-D
   cross-stage discriminator cleanup.

**Verdict**: 1 HIGH (interaction-bug, gate-2-introduced, gate-2
audit missed) + 4 OBS — gate-3 type-design is one one-line fix
away from CLEAN, identical in structure to gate-2's own one-test
discharge profile.
