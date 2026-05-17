# Stage 41 Inc 4 closure gate-1 type-design audit

**HEAD**: `7448bf5` (Stage 41 Inc 1 + Inc 2 + Inc 3 shipped end-to-end)
**Base**: `5dd478a` (Stage 40 CLOSED)
**Scope**: `git diff 5dd478a..7448bf5 -- helixc/frontend/typecheck.py
helixc/frontend/autodiff.py helixc/ir/lower_ast.py`
(244-line typecheck.py delta, 13-line autodiff.py delta, 12-line
lower_ast.py delta)
**Date**: 2026-05-17
**Auditor brief**: type-design lens on the new 5th-stack causal/intent
wrapper family — looseness, strictness, family asymmetry,
encoded-as-convention smells, invariant-cascade discipline,
gate-2/gate-3 lesson transfer from Stage 40. Gate-1 strictness:
flag anything with HIGH conf 75 or above; MEDIUM/LOW noted for
audit-trail completeness. Reference templates:
`docs/audit-stage40-inc4-gate2-type-design.md` (Stage-40 modal
analogue) and `docs/audit-stage40-inc4-gate3-type-design.md`
(invariant-cascade interaction-bug case study).
**Prior-stage context** (do NOT re-flag — pre-existing wrapper-family
issues): F1 named-binding bypass (Stage 40 gate-2 H1, pinned, deferred
to Phase-1 taint pass); generic-inner TyVar deferral missing across
all 4 wrapper families (Stage 40 gate-2 OBS-B); `str`-discriminator
anti-pattern across all wrappers (Stage 40 gate-3 OBS-D); stale F2
"unreachable" dead-coding diagnostic post-H2 fall-through (Stage 40
gate-3 OBS-B).

## VERDICT: 1 HIGH, 1 MEDIUM, 0 LOW, 3 OBS

The Stage 41 surface ports the 8-helper symmetry sweep from Stage 40
Inc 1 cleanly (full TyModal-parity verified) and the new F1 cross-
causal launder guard adopts the same kind-specific-hint discipline.
The single HIGH finding is the predicted gate-3-analogue: the new
F1 causal launder guard does NOT mirror the Stage 40 gate-3 H1
`inner_is_shadowed` cascade-suppression check — exactly the
interaction-bug Stage 40 gate-2 missed and Stage 40 gate-3 caught.
The Stage 41 OPENS commit landed the new guard before the
gate-3 H1 lesson was ported.

---

## Summary table

| ID | Sev | Conf | Title |
|----|-----|------|-------|
| H1 | HIGH | 88 | F1 cross-causal launder guard ignores `_shadowed_builtin_names` for the INNER call — same gate-3 invariant gap Stage 40 H1 closed for the modal guard, present in fresh-shipped Stage 41 code. Reproduces live at `7448bf5` with `fn from_cause(x: i32) -> i32 { x }` → 2 diagnostics (1 shadow + 1 false-positive launder) where the right answer is 1. Mirror the modal guard at typecheck.py:3643-3656 verbatim |
| M1 | MEDIUM | 78 | `propagate` / `aggregate` / `isolate` are added to `_BUILTIN_NAMES` as common-verb reservations, but the shadow-diagnostic HINT at typecheck.py:949-952 enumerates only `into_*, from_*, confirm, act_on, forecast, world_to_robot` as examples — does NOT mention the 3 new causal transitions. A user shadowing `propagate` reads "reserved builtin" but the hint's example list omits their case, inviting "but propagate isn't in the list?" confusion. One-line fix: append `propagate/aggregate/isolate` to the hint enumeration |
| OBS-A | OBS | 95 | Family-symmetry verdict: all 8 type-system helper sites that have TyModal arms also have TyCausal arms. The Stage 40 Inc 1 preemptive-port pattern was internalized into Stage 41 Inc 1 (commit `7448bf5` body). `_compatible` bilateral + unilateral, `_refinement_shape_exact` (both pair-versions), `_erase_refinement`, `_contains_refinement`, `_contains_refined_function`, `_is_refinement_container`, `_contains_unknown_type`, `_fmt` — all 1-to-1 with TyModal arms. No symmetric gap |
| OBS-B | OBS | 72 | The `_causal_upgrade_hint` table at typecheck.py:3757-3767 is a 3rd dispatch table parallel to `_causal_transitions` (typecheck.py:3838-3842) — same anti-pattern as Stage 40 gate-3 OBS-A (modal version). Adding a future `("cause", "joint"): "co_observe"` arrow requires touching 3 sites (`_causal_transitions`, `_BUILTIN_NAMES`, `_causal_upgrade_hint`). Mechanical inheritance from the modal precedent; not a Stage-41-specific regression |
| OBS-C | OBS | 70 | The `_causal_intro` / `_causal_elim_kind` / `_causal_transitions` / `_causal_upgrade_hint` dicts are defined as function-local literals inside `_check_expr` (typecheck.py:3745-3804, 3838-3842) — reallocated on every call expression encountered. Stage 40 modal dispatch tables share the same local-literal pattern. Performance-only; profiler would show ~0.1% of typecheck time. Move to module level if a `_BUILTIN_DISPATCH_TABLES` consolidation lands at Phase-1 |
| OBS-D | OBS | 65 | Direction-asymmetry verdict: the 3 forward transitions (Cause→Effect, Effect→Joint, Joint→Independent) are monotonically information-losing and the reverse direction is semantically nonsensical (an Effect retroactively becoming its own Cause). The closure-trail-anticipated reverse-direction hints at typecheck.py:3775-3803 are NOT YET present in the gate-1 surface but the generic Phase-0-deferral hint fires correctly with the right framing. Soundness OK; UX could be sharper for reverse-direction probes |

---

## Findings

### H1: F1 cross-causal launder guard ignores `_shadowed_builtin_names` for the INNER call — exact gate-3 H1 analogue in fresh Stage-41 code (HIGH, confidence 88/100)

**Location**: `helixc/frontend/typecheck.py:3781-3786` (F1 cross-causal
guard predicate inside the `_causal_intro` dispatch arm).

**Pattern**: Stage 40 gate-3 H1 (conf 86, fixed in commit `38f5598`)
established the invariant that the H2 dispatch-suppression discipline
must apply BOTH to the outer call's bare-name dispatch AND to every
subsystem that reads the INNER call's syntactic name. The Stage 40
modal F1 guard was retrofitted at gate-3 to add:

```py
# typecheck.py:3643-3656 (Stage 40 gate-3 H1 fix, post-Stage-41-base)
inner_is_shadowed = (
    len(expr.args) >= 1
    and isinstance(expr.args[0], A.Call)
    and isinstance(expr.args[0].callee, A.Name)
    and expr.args[0].callee.name
        in self._shadowed_builtin_names
)
if (... in _modal_elim_kind
        and not isinstance(arg_tys[0], TyUnknown)
        and not inner_is_shadowed):                # <-- gate-3 fix
```

The Stage 41 OPENS commit (`7448bf5`) ships a NEW parallel F1 guard
for the causal family at lines 3781-3786 that does NOT include the
`inner_is_shadowed` predicate — it was modelled on the gate-2 modal
guard (pre-gate-3-H1), not the gate-3-strengthened version:

```py
# typecheck.py:3781-3786 (Stage 41 Inc 1, gate-1 surface)
if (len(expr.args) >= 1
        and isinstance(expr.args[0], A.Call)
        and isinstance(expr.args[0].callee, A.Name)
        and expr.args[0].callee.name
            in _causal_elim_kind
        and not isinstance(arg_tys[0], TyUnknown)):   # <-- no inner_is_shadowed
```

**Reproduction** (live probe against `7448bf5`):

```py
src = """fn from_cause(x: i32) -> i32 { x }
fn main() -> i32 {
    let e: Effect<i32> = into_effect(from_cause(42));
    from_effect(e)
}"""
errs = typecheck(parse(src, include_stdlib=True))
# 2 errors:
#   1:1: function 'from_cause' shadows a reserved builtin name ...
#   3:26: into_effect(from_cause(...)) launders a Cause<T> into Effect<T>
#         with no causal-transition audit.
#       hint: use `propagate(c)` — the audited Cause -> Effect causal transition
```

The second diagnostic is a false positive: no `Cause<T>` value ever
existed in this program. The user's `from_cause` is a benign
`i32 -> i32` passthrough. The F1 guard misreads it as the builtin
Cause eliminator because it consults the SYNTAX (the identifier text)
rather than the SEMANTICS (whether that identifier resolves to the
builtin or a shadowed user-fn).

**Why HIGH not MEDIUM**: this is the exact pattern Stage 40 gate-3
H1 named as a HIGH-grade invariant gap (conf 86), with the explicit
fix-pattern documented in the gate-3 audit doc. The Stage 41 OPENS
commit shipped before the gate-3 lesson was visible to the author
(the gate-3 audit landed in `ac34df9` AFTER the Stage 40 close at
`5dd478a` but BEFORE Stage 41 OPENS at `7448bf5`; the H1 fix itself
landed at `38f5598` which IS in the base of Stage 41). So the fix
mechanism was in the codebase when Stage 41 was authored — the
mirror-port discipline was the failure, not the fix's prior existence.
The symmetric defect is present across all 4 `from_*` causal
eliminators (`from_cause`, `from_effect`, `from_joint`,
`from_independent`).

The bug is the exact recurrence of the Stage 40 gate-2-shipped /
gate-3-caught pattern. The Stage 40 audit-trail predicted this class
of issue ("the gate-2 fix sweep introduced exactly ONE new
type-design issue — the H2 invariant is asymmetric across its
consumers" — gate-3 closing observation). Stage 41 introduced a new
consumer (the causal F1 guard) without applying the same invariant.

**Why this matters more for Stage 41 than Stage 40**: Stage 41
explicitly markets causal verbs (`propagate`, `aggregate`, `isolate`)
as "common verbs" the user might want for unrelated purposes — graph
algorithms, signal-processing pipelines, multi-agent message
distribution. The 4 `from_X` causal eliminators (`from_cause`,
`from_effect`, `from_joint`, `from_independent`) are similarly
generic English words. The probability that a user program shadows
one of them while ALSO using a different causal kind in the same
function is materially higher than the Stage-40 `from_uncertain`
case, where the trigger required the user to shadow a
modal-specifically-named verb AND wrap it in a different modal verb.

**Concrete consequences if left as-is**:

- Any user-fn `from_cause` / `from_effect` / `from_joint` /
  `from_independent` that returns a concrete type (not `TyUnknown`)
  will trigger a false-positive launder diagnostic on every cross-
  causal `into_X(from_Y(...))` call site. The user reads
  "launders a Cause<T> into Effect<T>" and is led to investigate a
  non-existent causal-attribution issue rather than the actual
  shadow problem.
- The H2 invariant promise ("1 shadow + 0 noise") is materially
  weakened for the causal family.
- The Stage 41 dogfood (`dogfood_14_causal_lifecycle.hx` if it
  exists at this commit — not confirmed in the audit scope) does
  not exercise the shadow + launder interaction; the test suite
  cannot catch this without a dedicated shadow-cascade test.

**Fix options** (in increasing scope):

1. **One-line predicate addition** in the F1 causal guard
   (recommended — mirror the Stage 40 gate-3 H1 fix verbatim):

   ```py
   # typecheck.py:3781, add inner_is_shadowed computation directly
   # above the existing guard predicate (matching the modal arm's
   # gate-3 structure at lines 3643-3656)
   inner_is_shadowed = (
       len(expr.args) >= 1
       and isinstance(expr.args[0], A.Call)
       and isinstance(expr.args[0].callee, A.Name)
       and expr.args[0].callee.name
           in self._shadowed_builtin_names
   )
   if (len(expr.args) >= 1
           and isinstance(expr.args[0], A.Call)
           and isinstance(expr.args[0].callee, A.Name)
           and expr.args[0].callee.name in _causal_elim_kind
           and not isinstance(arg_tys[0], TyUnknown)
           and not inner_is_shadowed):       # <-- new line
   ```

   Mechanically isomorphic to the modal arm; ~7 lines (computation
   + 1-line guard amendment).

2. **Centralize the suppression check** in a helper used by all
   inner-name-inspecting guards. Reduces the discipline-distribution
   surface for any future 6th-stack wrapper family. Larger refactor;
   the gate-3 audit explicitly deferred this to Phase-1.

3. **Add a regression test** pinning the post-fix behavior (analogue
   of `test_stage40_gate3_f1_inner_name_shadow_suppresses_guard`):

   ```py
   def test_stage41_gate1_f1_causal_inner_name_shadow_suppresses_guard():
       """Mirror the Stage 40 gate-3 H1 invariant for the causal F1
       guard: shadowing from_cause must not cascade into a false-
       positive launder diagnostic."""
       src = '''fn from_cause(x: i32) -> i32 { x }
       fn main() -> i32 {
           let e: Effect<i32> = into_effect(from_cause(42));
           from_effect(e)
       }'''
       errs = typecheck(parse(src, include_stdlib=True))
       launder_errs = [e for e in errs if "launders" in str(e)]
       assert launder_errs == [], (
           "H2 invariant must apply to F1 causal inner-name dispatch")
       shadow_errs = [e for e in errs
                      if "shadows a reserved builtin" in str(e)]
       assert len(shadow_errs) == 1
   ```

The recommended discharge is (1) + (3): mirror the modal fix +
one regression test. Gate-1 closure on this finding is mechanical
and predictable — the fix template already exists at lines
3643-3656 for the modal guard; literal copy-paste-rename suffices.

---

### M1: Shadow-diagnostic hint enumeration is stale — omits `propagate` / `aggregate` / `isolate` (MEDIUM, confidence 78/100)

**Location**: `helixc/frontend/typecheck.py:949-952` (shadow-builtin
hint text inside the fn-decl shadow check).

**Pattern**: The shadow-builtin diagnostic at fn-decl time enumerates
example reserved names in its hint text:

```py
# typecheck.py:949-952
hint=f"reserved builtins include modal/temporal/"
f"frame/tier intro+elim+transition verbs (e.g. "
f"into_*, from_*, confirm, act_on, forecast, "
f"world_to_robot); pick a different name",
```

The Stage 41 OPENS commit added 11 new reserved names to
`_BUILTIN_NAMES` (lines 2037-2041): 4 intro builtins, 4 elim
builtins, 3 transition verbs. But the example list in the hint was
NOT updated. A user who writes `fn propagate(p: Particle, dt: f32)`
in a physics-simulator program gets:

```
1:1: function 'propagate' shadows a reserved builtin name; rename
the function to avoid silent dispatch dead-coding ...
  hint: reserved builtins include modal/temporal/frame/tier intro+
        elim+transition verbs (e.g. into_*, from_*, confirm,
        act_on, forecast, world_to_robot); pick a different name
```

The hint enumeration omits `propagate`. The reasonable user reaction
is "but propagate isn't on the list?" — leading either to (a)
disbelief that the rename is necessary, or (b) doubt about diagnostic
accuracy generally. The hint text is meant to be illustrative-not-
exhaustive (the "modal/temporal/frame/tier intro+elim+transition
verbs" preamble communicates breadth), but a user expecting a verb
that appears in their own code to be enumerated will not get that
signal.

**Why MEDIUM not LOW**: the trio `propagate` / `aggregate` /
`isolate` is composed of three of the most generic English verbs in
the entire `_BUILTIN_NAMES` set. The probability of real-program
collision (graph propagation algorithms, sensor aggregation
pipelines, dependency-isolation systems) is materially higher than
for `confirm` / `act_on` (which were themselves flagged in Stage 40
as "extremely-generic names likely to collide with user planning /
state-machine code" per typecheck.py:935-937 comment).

Stage 41 *added* the causal stack on the explicit thesis that AGI
decision-making requires distinguishing the 4 causal kinds; the
verbs were chosen for descriptive accuracy ("propagate" IS the
right word for Cause→Effect). The downside is that descriptive
accuracy maximizes collision probability with prior-art user code.
The hint text is the user's first-line resource for understanding
what the framework reserves; it should reflect the actual reservation.

**Why not HIGH**: the diagnostic still fires correctly (the shadow
IS detected); the user IS forced to rename; no silent dispatch
divergence. The cost is only diagnostic confusion + an extra Google
search to confirm the framework intentionally reserves the verb.

**Fix** (one-line, mechanical):

```py
hint=f"reserved builtins include modal/temporal/"
f"frame/tier/causal intro+elim+transition verbs (e.g. "
f"into_*, from_*, confirm, act_on, forecast, "
f"world_to_robot, propagate, aggregate, isolate); "
f"pick a different name",
```

Note also: this hint has been stale since Stage 39 added
`forecast`/`recall_past`/`actualize` (the temporal transitions) but
the gate-3 audit did not flag it because the temporal verbs are
less universally meaningful in non-temporal contexts. Stage 41's
causal verbs cross the threshold where the staleness matters.

**Related** (cross-reference for the closure trail): the
Stage 40 gate-3 OBS-B finding flagged that the F2 shadow diagnostic
BODY is stale post-H2 (claims user-fn is "unreachable" when post-H2
the user-fn IS reachable). That body-staleness still applies at
Stage 41 — Stage 41 inherits the typecheck.py:944-947 text
unchanged. Stage 40 gate-3 deferred fixing it as "cosmetic". With
Stage 41's additions, both the HINT (M1 here) and the BODY (Stage 40
OBS-B inherited) drift further from accurate framework behavior. A
one-stop diagnostic-text refresh would close both at once.

---

## OBS-A: Family symmetry — Stage 40 Inc 1 preemptive-port pattern internalized (OBS, confidence 95/100)

The brief's Q1 ("does TyCausal appear in every helper that TyModal
does?") deserves an explicit clean answer: **yes, full 1-to-1
parity across all 8 helper sites.**

Inventory (verified via `Grep "isinstance\([a-z_]+, \(?TyCausal\)?"`
and `Grep "isinstance\([a-z_]+, \(?TyModal\)?"`):

| Helper | TyTemporal | TyModal | TyCausal | Status |
|--------|------------|---------|----------|--------|
| `_compatible` bilateral arm | 7279 | 7286 | 7291 | OK |
| `_compatible` unilateral reject arm | 7281 | 7288 | 7293 | OK |
| `_refinement_shape_exact` (target/value_ty) | 5365 | 5372 | 5377 | OK |
| `_refinement_shape_exact` (a/b) | 6042 | 6046 | 6050 | OK |
| `_erase_refinement` | 6100 | 6103 | 6106 | OK |
| `_contains_refinement` | 6224 | 6227 | 6230 | OK |
| `_contains_refined_function` | 6281 | 6284 | 6287 | OK |
| `_is_refinement_container` (tuple) | in 6253 tuple | in tuple | in 6254 tuple | OK |
| `_contains_unknown_type` (tuple) | in 5304 tuple | in tuple | in 5304 tuple | OK |
| `_fmt` | 7461 | 7465 | 7469 | OK |

Every site that has a TyTemporal arm also has TyModal AND TyCausal
arms — the Stage 40 closure gate-2 OBS-A clean verdict extends
unchanged at Stage 41. The Stage 41 Inc 1 commit (`7448bf5`) body
asserts "TyCausal arms added preemptively to 8 type-system
helpers" — verified.

This is the single strongest signal that the audit-loop learning
machine is working: the Stage 39 H1/H2/H3 cluster, the Stage 40
gate-2 OBS-A confirmation, and the Stage 41 OPENS-time port form
a clean three-stage discipline-transmission record.

---

## OBS-B: `_causal_upgrade_hint` is a 3rd dispatch table parallel to `_causal_transitions` — same anti-pattern as Stage 40 gate-3 OBS-A (OBS, confidence 72/100)

**Location**: `helixc/frontend/typecheck.py:3757-3767` vs `:3838-3842`.

The Stage 41 surface inherits the Stage 40 gate-3 OBS-A anti-pattern
unchanged. The new causal-side dispatch has THREE parallel tables
encoding the same direction-relationship from different angles:

1. `_causal_transitions` (verb → (src, dst)) at line 3838:
   ```py
   _causal_transitions = {
       "propagate": ("cause",  "effect"),
       "aggregate": ("effect", "joint"),
       "isolate":   ("joint",  "independent"),
   }
   ```

2. `_BUILTIN_NAMES` (frozen set at line 2041): registers the 3
   transition verbs as reserved.

3. `_causal_upgrade_hint` ((src, dst) → hint text) at line 3757:
   ```py
   _causal_upgrade_hint = {
       ("cause", "effect"):     "use `propagate(c)` ...",
       ("effect", "joint"):     "use `aggregate(e)` ...",
       ("joint", "independent"): "use `isolate(j)` ...",
   }
   ```

If a future Phase-1 stage adds a 4th transition (e.g.
`co_observe: Cause -> Joint` for the case where two causes are
observed simultaneously), the maintainer must touch all THREE
sites:

1. `_causal_transitions["co_observe"] = ("cause", "joint")`
2. `_BUILTIN_NAMES` += `{"co_observe"}`
3. `_causal_upgrade_hint[("cause", "joint")] = "use `co_observe(c)` ..."`

Forgetting (3) makes the F1 fallback hint fire with text "Phase-0
has no Cause -> Joint transition" — a lie post-Phase-1. Same
maintenance-trap dynamic the Stage 40 gate-3 OBS-A flagged for the
modal upgrade hint table.

**Why OBS not LOW**: this is a mechanical inheritance; the
anti-pattern was named by Stage 40 gate-3 and explicitly deferred
to Phase-1. Stage 41 propagating it costs nothing additional at
gate-1 review time; the recommended derive-from-transitions fix is
the same fix the modal table needs.

**Fix** (Phase-1, derive the hint table from the transition table):

```py
# Eliminate the 2nd table by computing it from the 1st.
_causal_upgrade_hint = {
    direction: f"use `{verb}(v)` — the audited "
               f"{direction[0].capitalize()} -> "
               f"{direction[1].capitalize()} causal transition"
    for verb, direction in _causal_transitions.items()
}
```

Defer to Phase-1 as a coordinated wrapper-cleanup pass (modal +
causal share the fix template).

---

## OBS-C: Dispatch tables defined as function-local literals; reallocated per call expression (OBS, confidence 70/100)

**Location**: `helixc/frontend/typecheck.py:3745-3804, 3838-3842`
(the 4 causal dispatch dicts).

All 4 causal dispatch dicts (`_causal_intro`, `_causal_elim_kind`,
`_causal_upgrade_hint`, `_causal_transitions`) are defined inside
the `_check_expr` body, which means they are recomputed on every
call-expression encountered (the dict literals are evaluated each
time control enters the relevant arm). Stage 40 modal dispatch
tables (`_modal_intro`, `_modal_elim_kind`, `_modal_upgrade_hint`,
`_modal_transitions`) share the same local-literal pattern, so this
is NOT a Stage-41-specific regression.

**Performance impact**: negligible (dict construction of 3-5
entries is microseconds; typecheck.py is not in a hot loop). A
profiler would show this as ~0.1% of typecheck time on a large
program. Definitely not worth a refactor at gate-1.

**Why flagged at OBS**: the brief asks for type-design rigor, and
"dispatch tables that should be module-level constants are buried
inside a 7000-line method" is a maintainability smell even when
performance is fine. The 4-stack quartet has now grown to a
5-stack quintet; the lookup dispatch sprawl has 5 × 4 = 20 distinct
dict-literals scattered through the same method. A consolidation
pass at Phase-1 (move to module-level `_WRAPPER_DISPATCH_TABLES`)
would also enable the OBS-B derive-from-transitions fix to apply
uniformly.

**Recommended improvement** (Phase-1, paired with OBS-B):
elevate all 20 wrapper dispatch dicts to module-level constants
keyed by wrapper family. Documents the family structure; enables
the OBS-B derivation refactor; reduces `_check_expr` body size by
~80 lines.

---

## OBS-D: Direction-asymmetry verdict — 3 forward transitions correctly justified; reverse-direction hints fire via Phase-0-generic fallback (OBS, confidence 65/100)

The brief's Q2 ("what about reverse? Independent collapsing back to
Joint? Effect upgrading to Cause?") deserves an explicit answer.

**Verdict**: the 3 forward transitions are monotonically
information-losing in the causal-graph topology:

- `propagate: Cause -> Effect` — applying a cause produces an
  effect; the effect cannot retroactively become its own cause.
- `aggregate: Effect -> Joint` — multiple effects combine into a
  joint observation; once aggregated, the individual contributions
  are no longer separable WITHOUT external information.
- `isolate: Joint -> Independent` — experimentally testing that no
  upstream actually matters collapses the joint dependency; once
  experimentally established, the value is causally isolated.

The reverse direction (`Independent -> Joint -> Effect -> Cause`)
would require fresh evidence to introduce a causal upstream that
was previously absent. This is semantically meaningful but is NOT a
type-level transition — it's a value-replacement that requires a
new evidence stream. The right framing: a reverse-direction
"transition" would be a category error (you'd be replacing the
value, not transitioning it).

**Soundness of reverse-direction guard**: tested live at `7448bf5`,
direct inline reverse-direction laundering IS caught by F1:

```
let e: Effect<i32> = into_effect(42);
let c: Cause<i32> = into_cause(from_effect(e));  // <-- rejected
```

with the fallback hint "Phase-0 has no Effect -> Cause transition;
if this direction is semantically meaningful, request a future-stage
spec and keep the value in its current causal kind until then."

The fallback hint is technically correct (no such transition exists
in Phase-0) but understates the case — the right framing is "this
direction is semantically nonsensical; an effect cannot become its
own cause," not "Phase-0 doesn't ship it yet (so wait for Phase-1)."
The modal-side equivalent (Stage 40 closure gate-3 LOW fix at
typecheck.py:3593-3624) added specific reverse-direction hints for
Uncertain→Known/Believed/Goal precisely because the generic
Phase-0 framing misled contributors about whether the direction
would land later.

The Stage 41 gate-1 surface does NOT yet have specific reverse-
direction hints for the 6 nonsensical reverse causal directions
(Effect→Cause, Joint→Cause, Independent→Cause, Joint→Effect,
Independent→Effect, Independent→Joint). The F1 guard rejects them
correctly, but the hint UX is weaker than the modal equivalent.

**Why OBS not LOW**: the guard rejects correctly. The hint UX is
the only gap. Stage 41 has the same "specific reverse-direction
hint" template available from the Stage 40 gate-3 LOW fix; the
fix is a 6-entry table extension to `_causal_upgrade_hint`.

**Recommended improvement** (gate-1 closure, mechanical):
add 6 reverse-direction entries to `_causal_upgrade_hint`. Each
entry frames the direction as semantically nonsensical (not
deferred). Template borrowed from typecheck.py:3593-3624:

```py
_causal_upgrade_hint = {
    # forward transitions (existing 3):
    ("cause", "effect"):     "use `propagate(c)` ...",
    ("effect", "joint"):     "use `aggregate(e)` ...",
    ("joint", "independent"): "use `isolate(j)` ...",
    # reverse / nonsensical directions (Stage 41 gate-1 OBS-D fix):
    ("effect", "cause"):
        "an effect does not retroactively become its own cause; "
        "if you mean to identify the upstream cause, recover it "
        "from the same provenance source rather than unwrap-rewrap "
        "the downstream value",
    ("joint", "cause"):
        "a joint observation is downstream of multiple causes; "
        "promoting it back to Cause<T> conflates aggregation with "
        "origination — re-derive the cause from the original "
        "provenance",
    ("independent", "cause"):
        "an Independent<T> value has been shown to have NO upstream; "
        "treating it as a Cause<T> contradicts that experimental "
        "finding",
    ("joint", "effect"):
        "a joint observation aggregates multiple effects; demoting "
        "it to Effect<T> requires picking a specific contributing "
        "effect, not unwrap-rewrap",
    ("independent", "joint"):
        "Independent<T> means the experiment collapsed the multi-"
        "cause dependency; re-promoting to Joint<T> would require "
        "fresh evidence of dependency, not unwrap-rewrap",
    ("independent", "effect"):
        "an Independent<T> value's upstream is by construction "
        "empty; calling it an Effect<T> claims a downstream-of-"
        "something relationship that was just experimentally "
        "falsified",
}
```

---

## Negative-result checks (what the audit looked for and did NOT find)

For audit-trail completeness, the following anti-patterns were
hunted and NOT found at the Stage 41 gate-1 surface:

1. **Cross-dispatch leakage**: `propagate`/`aggregate`/`isolate`
   do NOT appear in `_modal_intro`, `_modal_elim_kind`,
   `_modal_transitions`, `_temporal_*`, `_frame_*`, or any other
   wrapper dispatch dict. Verified via `Grep` (see brief Q6). The
   3 transition verbs dispatch ONLY through the causal arm
   (typecheck.py:3838-3842, single-site).

2. **Cross-kind composition rejection**: `Known<Cause<f32>>` and
   `Cause<Known<f32>>` both round-trip cleanly through the type
   helpers (verified via live probe). The `_refinement_shape_exact`
   / `_compatible` / `_erase_refinement` recursive arms handle
   both nesting orders symmetrically.

3. **Multi-step inline laundering**: `into_X(from_Y(from_Z(v)))`
   triggers the F1 guard correctly (the immediate inner `from_Y`
   is the laundering source; the F1 fix family treats one level
   at a time and accepts deeper indirection per the Stage 40
   gate-2 H1 deferred-limitation).

4. **Naming-string-match drift**: the kind discriminator
   `"independent"` (adjective) round-trips correctly through all
   dict-keyed dispatch sites — verified via `Grep` and live probe.
   No `_fmt` / error-message / hint-builder uses substring matching
   on noun/adjective suffixes.

5. **Function-attribute `effect` namespace collision**: the
   `@effect` function attribute at typecheck.py:1027 is in a
   completely independent namespace from causal `Effect<T>` and
   the kind discriminator string `"effect"`. No collision.

6. **TyCausal in `_resolve_type`**: the new `causal_map`
   (typecheck.py:1234-1238) correctly maps the 4 type-level names
   (`Cause`, `Effect`, `Joint`, `Independent`) to the 4 kind
   discriminators. The dispatch is structurally identical to
   `modal_map` immediately above; no asymmetry.

---

## Cross-cutting observation: H1 is the predicted gate-3 analogue; M1 reflects "5th stack stresses the diagnostic-text infrastructure"

Stage 41 gate-1 reproduces exactly the failure mode Stage 40's
audit-trail predicted: the new F1 launder guard ships without the
gate-3 H1 cascade-suppression check. The fix template exists
verbatim in the same file at lines 3643-3656; the Stage 41 author
modeled the new arm on the pre-gate-3 version of the modal arm.

M1 is a different category: the diagnostic-hint text was written
when there were 4 wrapper families and 2 modal transitions. Stage 41
brought the count to 5 families and 5 transitions. The
illustrative-not-exhaustive enumeration in the hint is starting to
mislead, especially because the 3 new transition verbs are by far
the most generic English words in the entire reserved set.

The remaining items are observational and either deferred Phase-1
work (OBS-B, OBS-C) or sharpening opportunities (OBS-D).

If H1 and OBS-D are addressed (one-line predicate fix + 6-entry
hint table extension + one regression test), the gate-1
type-design audit closes CLEAN. M1 is one additional line in
the existing hint text.

## Recommended priority for Stage 41 closure gate-1

1. **Must-fix before Stage 41 closes (gate-1 invariant-discipline,
   exact gate-3 H1 analogue)**: H1's inner-name shadow check +
   regression test. ~10 lines. Mirror typecheck.py:3643-3656
   verbatim into the causal arm at line 3781.
2. **Should-fix before closes (sharpens UX on safety-positioned
   reverse directions)**: OBS-D's 6-entry reverse-direction hint
   table extension. ~40 lines (one entry per nonsensical reverse).
3. **Cheap cosmetic**: M1's hint-enumeration update. ~3 lines.
4. **Defer to Phase-1**: OBS-B hint-table derivation (paired with
   modal OBS-A from Stage 40 gate-3), OBS-C module-level dispatch
   table consolidation.

**Verdict**: 1 HIGH (predicted gate-3-analogue interaction-bug;
exact fix template available in same file) + 1 MEDIUM
(diagnostic-text staleness exacerbated by 5th-stack additions) +
3 OBS — gate-1 type-design is one one-line fix + a 6-entry hint
table away from CLEAN, in line with the gate-3 close profile from
Stage 40.
