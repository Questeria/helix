# Stage 39 post-Inc-3 type-design audit

**HEAD**: 01b3b86 (Stage 39 OPENS: Inc 0 + Inc 1 + Inc 2 + Inc 3 — temporal types)
**Scope**: `git diff 9fcc621..01b3b86 -- helixc/frontend/typecheck.py
helixc/frontend/autodiff.py helixc/ir/lower_ast.py
helixc/examples/dogfood_12_temporal_lifecycle.hx
helixc/tests/test_stage39_temporal.py`
**Date**: 2026-05-17
**Auditor brief**: type-design lens — looseness, strictness, family
asymmetry, encoded-as-convention smells, runtime-vs-compile-time gaps.
Stage 39 is a verbatim port of the Stage 37 (tier) + Stage 38 (frame)
playbook — invariants should now hunt for the *same* failure modes
the Stage 38 closure gate-1 audit found (H1: `_compatible` arms;
H2: refinement-visiting helpers). Stage 38 fixed those before
closure. Stage 39 ships pre-fix.

## Summary table

| ID | Sev | Conf | Title |
|----|-----|------|-------|
| H1 | HIGH | 92 | `TyTemporal` missing from `_compatible` — both unilateral rejection arm and bilateral kind+inner-recursion arm absent; cross-kind / wrapped-vs-raw silent acceptance at function-call boundary for refined / generic / shape-symbolic inners |
| H2 | HIGH | 90 | `TyTemporal` missing from 6 refinement-visiting helpers — refinements under a temporal wrapper are invisible to every refinement pass; `_join_branch_types` silently drops, `_refinement_shape_exact` rejects all concrete callers, `_is_refinement_container` returns False |
| M1 | MEDIUM | 80 | `TyTemporal.kind: str` encodes a closed 4-value enum as a string — identical anti-pattern to TyMemTier.tier + TyFrame.frame; typo `TyTemporal(kind="passt", ...)` constructs cleanly and surfaces as garbled diagnostics |
| M2 | MEDIUM | 78 | All 12 new Stage 39 builtins ship without remediation hints — replays Stage 38 M2 verbatim; cross-temporal mistakes give bare "requires Past<T>, got Future<i32>" with no transition-suggestion |
| M3 | MEDIUM | 70 | `from_X` / transition diagnostics print `Past<T>` / `Present<T>` (matching source syntax), but `_fmt` capitalization map sits at line 6859 — adding a 5th kind requires touching 4 string-literal sites scattered through `_resolve_type`, `_temporal_intro`, `_temporal_elim`, `_temporal_transitions`, and `_fmt` with no compile-time check for parity |
| L1 | LOW | 72 | `into_X` accepts `TyTemporal` as inner — `into_past(into_future(x))` builds `Past<Future<T>>` with no diagnostic; family-symmetric with tier/frame but semantically nonsense for temporal (a fact can't be "past inside future") |
| L2 | LOW | 60 | `Eternal` is silently special-cased via absence — neither `_temporal_transitions` nor any rejection comment encodes "Eternal is timeless"; the property is enforced only by the dictionary being empty in two slots and a single test (`test_stage39_inc2_eternal_never_transitions`); reviewer must reason from absence |
| OBS | OBS | 90 | Family-symmetry verdict: Stage 39 is a verbatim Stage 38 port with the **same** pre-closure holes Stage 38 had. The Stage 38 H1/H2 fixes from commit `1c6e047` did NOT propagate to TyTemporal — copying the fix pattern is mechanical |

---

## Findings

### H1: TyTemporal missing from `_compatible` — both unilateral rejection arm and bilateral kind+inner-recursion arm absent (HIGH, confidence 92/100)

**Location**: `helixc/frontend/typecheck.py:6679-6691` (`_compatible`)
**Pattern**: Verbatim replay of Stage 38 H1. Every other wrapper
type (`TyDiff`, `TyLogic`, `TyMemTier`, `TyFrame`, `TyQuote`,
`TyTuple`, `TyArray`, `TyRef`, `TyPtr`, `TyFn`, `TyTensor`, `TyTile`)
has BOTH a bilateral arm (`isinstance(a, T) and isinstance(b, T)`)
AND a unilateral rejection arm (`isinstance(a, T) or isinstance(b, T):
return False`). **TyTemporal has neither.**

**Citation** (grep-able):
```
grep -nE "isinstance\(a, Ty(MemTier|Frame|Temporal)\)" helixc/frontend/typecheck.py
  6679:        if isinstance(a, TyMemTier) and isinstance(b, TyMemTier):
  6681:        if isinstance(a, TyMemTier) or isinstance(b, TyMemTier):
  6688:        if isinstance(a, TyFrame) and isinstance(b, TyFrame):
  6690:        if isinstance(a, TyFrame) or isinstance(b, TyFrame):
  (no TyTemporal matches)
```

Total `isinstance(_, TyTemporal)` sites in the entire file: 4
(arg dispatch at 3327, 3356; `_fmt` at 6858; one constructor return
in resolver at 1130). The compile-time-compatibility predicate is
unaware of `TyTemporal` entirely; the type falls through to the
final `a == b` dataclass-equality fallback at line ~6816.

**What survives unchecked** (each case slips through because the
`a == b` fallthrough is frozen-dataclass equality, not the
family-required structural recursion the other wrappers use):

1. **Refined inner mismatch**: `Past<{x:i32|x>0}>` as a function
   parameter type. The call site passes `Past<i32>` (or vice versa);
   frozen-dataclass `a == b` compares `TyRefined(...) != TyPrim("i32")`
   and rejects, when the intended semantics is "the refined inner is
   a more-specific subtype; propagate the compile-time guarantee
   through the wrapper". Every other wrapper goes through
   `_compatible(a.inner, b.inner)` which delegates refinement reasoning
   correctly.
2. **TyVar inner deferral**: `fn record_history<T>(p: Past<T>)` called
   with a concrete `Past<i32>`. The generic-parameter deferral arm
   in `_compatible` fires only when the wrapper arm calls
   `_compatible(a.inner, b.inner)` — which TyTemporal currently
   doesn't. `Past<TyVar("T")>` vs `Past<TyPrim("i32")>` compares
   unequal via dataclass equality; the call typechecks-fails.
3. **TySize / shape-symbolic inner**: `Past<[i32; 4]>` vs
   `Past<[i32; N]>` at a generic function-parameter site falls to
   `a == b` which compares sizes by identity rather than via
   `_size_compatible`.
4. **Cross-kind mistake involving wrappers**: `fn takes_past(p:
   Past<i32>)` called with `Future<i32>` — dataclass equality
   rejects via `kind="past" != "future"`, but with no
   "cross-kind requires explicit transition" hint; users see a
   generic "argument type mismatch" rather than a temporal-aware
   diagnostic.
5. **Bare-vs-wrapped silent acceptance** at function call boundary:
   `fn takes_past(p: Past<i32>)` called with bare `i32`. `a == b`
   compares `TyTemporal(...) != TyPrim("i32")` → False → caught.
   But the explicit unilateral arm convention is what makes this
   *intentional* rather than accidental — without it the next
   refactor of `_compatible` could re-open the hole.

**Impact**: The whole point of temporal typing (per `TyTemporal`
docstring at typecheck.py:255-264: "Cross-temporal transitions move
values between kinds") is undermined for any non-trivial inner type.
Real AGI temporal-reasoning code will use generic functions over
`Past<T>` (e.g., `fn replay<T>(p: Past<T>) -> T`); every such
generic site silently fails to typecheck because the wrapper arm
doesn't defer to `_compatible(a.inner, b.inner)`.

**Fix**: Add the two-arm pair to `_compatible`, mirroring the
TyMemTier / TyFrame arms exactly. Insert directly after the TyFrame
pair at line 6690-6691:

```py
# Stage 39 type-design H1 fix: TyTemporal wrapper arm. Same family
# convention as TyMemTier / TyFrame — refined / generic / shape-
# symbolic inners need the explicit recursive `_compatible`
# delegation rather than the dataclass-equality fallback.
if isinstance(a, TyTemporal) and isinstance(b, TyTemporal):
    return a.kind == b.kind and self._compatible(a.inner, b.inner)
if isinstance(a, TyTemporal) or isinstance(b, TyTemporal):
    return False
```

Five lines. No public-API change. Add canary tests mirroring the
Stage 38 frame-vs-bare + frame-with-refined-inner tests at
`test_stage38_frames.py:284-330` for temporal kinds. The Stage 38
H1 test pattern names:
- `test_stage38_world_frame_param_rejects_bare_i32_call`
- `test_stage38_world_frame_param_with_refined_inner_compatible`
- `test_stage38_world_frame_param_with_generic_inner_compatible`

Parallel temporal versions:
- `test_stage39_past_param_rejects_bare_i32_call`
- `test_stage39_past_param_with_refined_inner_compatible`
- `test_stage39_past_param_with_generic_inner_compatible`

---

### H2: TyTemporal missing from 6 refinement-visiting helpers (HIGH, confidence 90/100)

**Location**: `helixc/frontend/typecheck.py:4841, 5504, 5550, 5662,
5680-5684, 5703` (6 helper sites).
**Pattern**: Verbatim replay of Stage 38 H2. Every TyMemTier-handling
site has a sibling that walks through the wrapper to its inner type.
Stage 38 added the corresponding TyFrame arms in commit `1c6e047`
(closure fix sweep). Stage 39 ships without the parallel TyTemporal
arms in any of the same six sites.

**Citation** (grep-able, paired site by site):

| Helper | TyMemTier line | TyFrame line | TyTemporal status |
|--------|----------------|--------------|-------------------|
| `_refinement_shape_exact` (target/value_ty pair, 4789-4870) | 4841 | 4848 | **missing** |
| `_refinement_shape_exact` (a/b pair, 5475-5524) | 5504 | 5507 | **missing** |
| `_erase_refinement` | 5550 | 5552 | **missing** |
| `_contains_refinement` | 5662 | 5664 | **missing** |
| `_is_refinement_container` (tuple at 5680-5684) | 5683 (in tuple) | 5683 (in tuple) | **missing from tuple** |
| `_contains_refined_function` | 5703 | 5705 | **missing** |

```
grep -nE "isinstance\(\w+, TyTemporal\)" helixc/frontend/typecheck.py
  3327:                    if (isinstance(arg_tys[0], TyTemporal)
  3356:                    if (isinstance(arg_tys[0], TyTemporal)
  6858:        if isinstance(t, TyTemporal):
```

3 use-sites total in the entire file — vs the 11+ TyFrame has post-
fix and the 13+ TyMemTier has. The 6 visitor gaps are all of the
same kind: a structural-recursion arm that should walk through the
wrapper to its inner type, but doesn't because the wrapper kind
wasn't added to the dispatch list.

**Impact** (concrete, reproducible):

- `Past<{x:i32|x>0}>` as a function parameter type: the call-site
  refinement-shape check at `typecheck.py:4841` falls through
  (TyTemporal arm missing). The bare `a == b` fallback at the end
  of `_refinement_shape_exact` then rejects any concrete call-site
  value because `TyTemporal("past", TyRefined(...)) !=
  TyTemporal("past", TyPrim("i32"))`. Net effect: refined-inner
  temporal parameters become uncallable.
- `_contains_refinement(Past<{x:f32|x.is_finite()}>)` returns False
  — so any pass that says "if this type contains a refinement, emit
  refinement-checking ops" will skip the temporal-wrapped refinement,
  producing a silent runtime hole. The most concrete consequence is
  in `_join_branch_types` at line 5588-5605: when joining branches
  that return `Past<{...}>`, the refinement-container check fails,
  so the join silently drops to bare types rather than emitting a
  shape-mismatch diagnostic.
- `_is_refinement_container` (line 5680-5684) is the predicate that
  the rest of the refinement system uses to decide which types could
  plausibly carry refinement metadata. The tuple `(TyArray, TyTuple,
  TyRef, TyPtr, TyFn, TyDiff, TyLogic, TyQuote, TyMemTier, TyFrame,
  TyTensor, TyTile)` does NOT include `TyTemporal` — so every
  container-aware refinement transform skips temporal wrappers
  entirely. Symmetric Stage 38 finding: the gate-1 audit fix
  (commit 1c6e047) added `TyFrame` to this tuple.
- `_erase_refinement(Past<{x:i32|x>0}>)` returns the input unchanged
  (the function fall-through is `return ty`), so any pass that calls
  `_erase_refinement` to widen a refined type to its base type
  silently keeps the refinement under a temporal wrapper.
- `_contains_refined_function(Past<fn(i32)->{r:i32|r>0}>)` returns
  False, so the branch-join refined-function check at line 5575-5577
  fails to fire for temporal-wrapped function values.

**Fix**: One line per helper, mirroring the TyMemTier / TyFrame arms
verbatim with `TyTemporal(ty.kind, ...)` substituted. Total ~9 lines:

```py
# _refinement_shape_exact (target/value_ty version, after line 4851):
if isinstance(target, TyTemporal) and isinstance(value_ty, TyTemporal):
    return (target.kind == value_ty.kind
            and self._refinement_shape_exact(value_ty.inner, target.inner))

# _refinement_shape_exact (a/b version, after line 5509):
if isinstance(a, TyTemporal) and isinstance(b, TyTemporal):
    return (a.kind == b.kind
            and self._refinement_shape_exact(a.inner, b.inner))

# _erase_refinement (after line 5553):
if isinstance(ty, TyTemporal):
    return TyTemporal(ty.kind, self._erase_refinement(ty.inner))

# _contains_refinement (after line 5665):
if isinstance(ty, TyTemporal):
    return self._contains_refinement(ty.inner, _seen_structs)

# _is_refinement_container (line 5683): add TyTemporal to the tuple:
return isinstance(ty, (
    TyArray, TyTuple, TyRef, TyPtr, TyFn, TyDiff, TyLogic, TyQuote,
    TyMemTier, TyFrame, TyTemporal, TyTensor, TyTile,
))

# _contains_refined_function (after line 5706):
if isinstance(ty, TyTemporal):
    return self._contains_refined_function(ty.inner)
```

Each is a verbatim TyMemTier / TyFrame copy. Canary tests should put
a refined inner under a temporal wrapper at a function parameter
position; the test passes today only because no such test exists.
Add a parallel of `test_stage38_frames.py:336-345`:

```py
def test_stage39_temporal_is_refinement_container():
    """Type-design H2: TyTemporal must be in `_is_refinement_container`
    so the join-branches refinement-shape check fires for temporal-
    wrapped values across branches."""
    from helixc.frontend.typecheck import TyTemporal, TyPrim
    tc = TypeChecker(parse("fn main() -> i32 { 0 }"))
    assert tc._is_refinement_container(TyTemporal("past", TyPrim("i32")))
```

---

### M1: `TyTemporal.kind: str` encodes a closed 4-value enum as a string (MEDIUM, confidence 80/100)

**Location**: `helixc/frontend/typecheck.py:263` (`kind: str`)
**Pattern**: Identical anti-pattern to TyMemTier.tier + TyFrame.frame.
Closed finite domain encoded as a stringly-typed field on a frozen
dataclass. No validation; no exhaustiveness checking; typos compile.

**Citation**:
```
grep -nE "(kind|tier|frame): str" helixc/frontend/typecheck.py
  237:    tier: str        # "working", "episodic", "semantic", "procedural"
  250:    frame: str       # "world", "robot", "camera"
  263:    kind: str        # "past", "present", "future", "eternal"
```

Concrete consequences:

- `TyTemporal(kind="passt", inner=...)` constructs without error.
  Downstream `_temporal_elim[bn] == "past"` checks then never match,
  producing "from_past() requires Past<T>, got Past<i32>" diagnostics
  where the printed inner type looks identical because `_fmt` at
  line 6858-6862 maps unknown kinds to the raw string via
  `cap.get(t.kind, t.kind)`.
- Adding a fifth kind (e.g. `Hypothetical`) requires touching at
  least 5 string-literal sites: `_resolve_type` temporal_map,
  `_temporal_intro`, `_temporal_elim`, `_temporal_transitions`, `_fmt`
  cap — with no compile-time check that they're all updated.
- The user prompt explicitly asks "Naming convention parity with
  Stage 38 (TyFrame uses Kind enum; TyTemporal should similarly
  enumerate Past/Present/Future/Eternal)" — the answer is that
  TyFrame does NOT use a Kind enum either (still `frame: str`).
  Stage 39 has parity-with-tiers/frames; both families share the
  smell. The Stage 38 gate-1 M1 finding documented this same
  trade-off for frames.

**Why MEDIUM not HIGH**: All `TyTemporal` construction in Stage 39
routes through hard-coded dictionaries (`temporal_map` at line 1124,
`_temporal_intro` at 3298, `_temporal_elim` at 3313,
`_temporal_transitions` at 3342). No typo can sneak in through user
code. The bug-class is closed by construction in Phase-0, but the
type doesn't enforce its own invariants and the "closed by
construction" property is one accidental `TyTemporal(kind=user_str,
...)` away from being violated.

**Fix options** (in increasing strength):

1. **Validate in `__post_init__`** — frozen dataclass override:
   ```py
   def __post_init__(self):
       if self.kind not in {"past", "present", "future", "eternal"}:
           raise ValueError(f"TyTemporal.kind must be one of "
                            f"past/present/future/eternal, got {self.kind!r}")
   ```
   Catches the bug at construction; ~5 lines.

2. **Use a typed enum** (FrameKind/TierKind/TemporalKind):
   ```py
   class TemporalKind(Enum):
       PAST = "past"
       PRESENT = "present"
       FUTURE = "future"
       ETERNAL = "eternal"
   ```
   Stronger invariant; forces every site (including `_fmt`, the
   intro/elim dicts, the transition dict) to use the enum members.
   The right time to do this is *together* with TyMemTier + TyFrame
   so all three families stay uniform.

3. **Defer + document**: add a unit test that pickle-roundtrips
   the dictionaries against the docstring kind list. Cheapest.

Recommend (1) for this stage if appetite exists, and (2) when next
refactoring the tier/frame families — the closed-domain smell is
real but the construction discipline in Stage 39 makes it latent
rather than active.

---

### M2: All 12 new Stage 39 builtins ship without remediation hints (MEDIUM, confidence 78/100)

**Location**: `helixc/frontend/typecheck.py:3330-3335` (from_X),
`3360-3365` (transitions).
**Pattern**: Verbatim replay of Stage 38 M2. The Stage 37 post-
closure fix (commit `a8ab17b`) explicitly addressed the family-
standard remediation-hint asymmetry for `parent_at`. Stage 38
flagged but deferred. Stage 39 inherits the deferral.

**Citation**:
```
grep -n '"{bn}() requires' helixc/frontend/typecheck.py
  3171:                        f"{bn}() requires "       # tier unwrap_X
  3202:                        f"{bn}() requires "       # tier transition (consolidate/recall)
  (Stage 38 frame uses different format: "{bn}() requires WorldFrame<T>, ...")
  3331:                        f"{bn}() requires "       # NEW Stage 39 from_X
  3361:                        f"{bn}() requires "       # NEW Stage 39 transitions
```

The bare diagnostic is:
```
from_past() requires Past<T>, got Future<i32>
to_past() requires Present<T>, got Past<i32>
```

A useful hint for from_X would point at the right elim *or* the
right transition:
```
from_past() requires Past<T>, got Future<i32>
   (did you mean from_future, or apply to_past after first
    actualize-ing through Present?)
```

And for transitions:
```
to_past() requires Present<T>, got Past<i32>
   (input is already in past; no transition needed.
    Use from_past to unwrap, or recall_past then to_past
    to round-trip through Present.)
```

The hint adds (a) the historical context for why the constraint
exists (no Past→Past identity, no Past→Future direct), (b) the
immediate corrective action.

**Why MEDIUM not LOW**: The Stage 37 post-closure precedent is
explicit and Stage 38 closure-gate M2 documented the same
asymmetry. The fix is cheap (one helper used across 8 sites; the
4 transition sites need a slightly different message because they
also reject Eternal as input). The cost of NOT fixing compounds
with each cross-temporal mistake a user makes — and AGI temporal
reasoning will produce many such mistakes during dogfood.

**Why not HIGH**: the family is internally consistent today (all 12
sites use the same hintless format). The smell is anticipatory.

**Fix**: Define `_temporal_kind_mismatch_hint(want_kind: str, got:
Type) -> str` and `_temporal_transition_mismatch_hint(builtin:
str, src_kind: str, got: Type) -> str`. Apply to 8 + 4 = 12 sites.
Shape mirrors `_strict_i32_truncation_hint` (typecheck.py:6183-6199).
~25 lines total.

---

### M3: Dispatch-path parity requires coordinated 5-site edits with no compile-time check (MEDIUM, confidence 70/100)

**Location**: `helixc/frontend/typecheck.py:1124-1130` (resolver
temporal_map), `1917-1920` (builtins list), `3298-3303`
(_temporal_intro), `3313-3318` (_temporal_elim), `3342-3347`
(_temporal_transitions), `6859-6861` (_fmt cap).
**Pattern**: Adding a 5th temporal kind (or renaming an existing
one) requires touching 5 separate dictionary literals + a builtins-
list tuple, with no compile-time check that they cover the same
set. The Stage 38 M3 finding documented this same smell for the
6-site frame dispatch; Stage 39 has 5 sites for kinds + 4
transition arrows.

**Citation**:
```
grep -nE '"past":|"present":|"future":|"eternal":' helixc/frontend/typecheck.py
  1126:                "Past": "past",
  1127:                "Present": "present",
  1128:                "Future": "future",
  1129:                "Eternal": "eternal",
  3299:                    "into_past":    "past",
  3300:                    "into_present": "present",
  3301:                    "into_future":  "future",
  3302:                    "into_eternal": "eternal",
  3314:                    "from_past":    "past",
  3315:                    "from_present": "present",
  3316:                    "from_future":  "future",
  3317:                    "from_eternal": "eternal",
  3343:                    "to_past":     ("present", "past"),
  3344:                    "forecast":    ("present", "future"),
  3345:                    "recall_past": ("past",    "present"),
  3346:                    "actualize":   ("future",  "present"),
  6860:                   "future": "Future", "eternal": "Eternal"}
```

5 dictionary literals + 1 capitalization map + 1 builtins-list
tuple + 1 IR identity-lowering tuple (lower_ast.py:2005-2012) + 2
AD-pure registrations (autodiff.py:105, 184) + 1
`_FRAME_IDENTITY_AD_NAMES` set. Total parallel sites to keep in
sync: 8. None enforced by type system or test.

**Impact**: Phase-1 Stage 39+ work (adding `Hypothetical<T>`,
`Counterfactual<T>`, or sub-temporal-kind metadata) will touch all
8 sites; a missed site silently degrades to "unknown temporal kind"
at runtime via the `cap.get(t.kind, t.kind)` fallback in `_fmt`.

**Why MEDIUM**: same as M1 — construction discipline closes the
hole today, but the implicit cross-site invariant is fragile.

**Fix**: A single source-of-truth dictionary:
```py
TEMPORAL_KINDS = {
    "past":     {"src_name": "Past",     "intro": "into_past",
                 "elim": "from_past"},
    "present":  ...,
    ...
}
TEMPORAL_TRANSITIONS = {
    "to_past":     ("present", "past"),
    "forecast":    ("present", "future"),
    "recall_past": ("past",    "present"),
    "actualize":   ("future",  "present"),
}
```
And derive `_temporal_intro`, `_temporal_elim`, `temporal_map`,
the cap-map, and the AD-pure tuples from these. Same approach
applies to TyMemTier and TyFrame — the win is uniform.

---

### L1: `into_X` accepts `TyTemporal` as inner — `into_past(into_future(x))` builds `Past<Future<T>>` (LOW, confidence 72/100)

**Location**: `helixc/frontend/typecheck.py:3304-3312`.
**Pattern**: An introducer that accepts already-introduced values
without rejection. Stage 38 L1 flagged this for frames; Stage 39
inherits the smell.

The `TyTemporal(kind=_temporal_intro[bn], inner=arg_tys[0])` line
at 3311-3312 does not inspect `arg_tys[0]`; if it's already a
`TyTemporal`, the new wrapper is silently stacked.

**Citation**:
```
grep -n "_temporal_intro\[bn\]" helixc/frontend/typecheck.py
  3299: (... dict definition)
  3304:                if bn in _temporal_intro:
  3311:                    return TyTemporal(kind=_temporal_intro[bn],
```

**Concrete nonsense that typechecks today**:
- `into_past(into_future(42))` → `Past<Future<i32>>` — a fact that
  is "past inside future" has no temporal semantics. The dogfood
  doesn't exercise this path, no test catches it.
- `into_eternal(into_past(42))` → `Eternal<Past<i32>>` —
  particularly contradictory because Eternal is documented as
  timeless yet wraps a Past tag.

**Why LOW**: family-symmetric with tiers (where layering may be
meaningful — "episodic memory of a working-memory snapshot") and
with frames (where the Stage 38 L1 finding was deferred). No
active bug yet.

**Why flagged**: temporal kinds have a stronger exclusivity property
than tiers and arguably stronger than frames — a fact is in *exactly
one* temporal kind at a time. Stage 39 currently treats them
identically to the layering-permissive families.

**Fix** (optional, defer to gate-2):
```py
if bn in _temporal_intro:
    if len(arg_tys) != 1:
        ... # existing arity check
    if isinstance(arg_tys[0], TyTemporal):
        self.errors.append(TypeError_(
            f"{bn}() input is already in {arg_tys[0].kind} kind; "
            f"use a transition (to_past / forecast / recall_past / "
            f"actualize) to change kinds, or from_{arg_tys[0].kind}() "
            f"to unwrap first",
            expr.span,
        ))
        return TyUnknown(hint=bn)
    return TyTemporal(kind=_temporal_intro[bn], inner=arg_tys[0])
```

---

### L2: `Eternal` timelessness encoded by absence — no rejection arm names it (LOW, confidence 60/100)

**Location**: `helixc/frontend/typecheck.py:3342-3347` (transitions
dict) + `test_stage39_temporal.py:333-347` (test).
**Pattern**: A first-class invariant ("Eternal does not transition,
because it's timeless") is encoded purely by the *absence* of
entries in the `_temporal_transitions` dictionary. A reviewer must
reason from absence; the dispatch path produces a generic "requires
PresentName<T>, got Eternal<i32>"-shape diagnostic that hints at
the destination's source-kind requirement rather than the actual
Eternal-is-timeless property.

**Citation** (the absence proves the case):
```
grep -nE '\("eternal"' helixc/frontend/typecheck.py
  (no matches — Eternal never appears as src OR dst in _temporal_transitions)
```

vs the explicit-rejection convention in the test:
```py
# test_stage39_temporal.py:333-347
def test_stage39_inc2_eternal_never_transitions():
    """Eternal<T> isn't a source of any transition (timeless)..."""
    for fn in ("to_past", "forecast", "recall_past", "actualize"):
        src = f"...from_past({fn}(e))"
        ...assert errs
```

**Impact**: a Phase-1 Stage 39+ extension that genuinely needs
"Eternal can transition to Present for snapshot read-out" would
silently add a `("eternal", "present")` arrow to the dict without
the broader system recognizing this is a semantic regression. The
invariant lives only in `test_stage39_inc2_eternal_never_transitions`
and the `TyTemporal` docstring at typecheck.py:255-264.

**Why LOW**: the test does catch the regression, and the Phase-0
discipline (all transitions enumerated in one 5-line dict) makes
audit-by-inspection feasible.

**Why flagged**: the user explicitly asked about "rejection matrix
exhaustiveness" and "naming convention parity". The exhaustiveness
property is real (Past↔Present↔Future round-tripping; Eternal
isolated) but its encoding is implicit. An explicit named-constant
+ comment would make the invariant machine-checkable:

```py
# Stage 39: Eternal is timeless — it has no source-kind transition
# (no entry where src="eternal") and no destination-kind transition
# (no entry where dst="eternal"). To violate this, both halves of
# the predicate below must change.
_TEMPORAL_ETERNAL_IS_ISOLATED = (
    not any(src == "eternal" for src, _ in _temporal_transitions.values())
    and not any(dst == "eternal" for _, dst in _temporal_transitions.values())
)
assert _TEMPORAL_ETERNAL_IS_ISOLATED, \
    "Stage 39 invariant: Eternal does not transition"
```

Plus an additional `test_stage39_no_transition_produces_eternal`
test (the existing test only covers Eternal-as-source; nothing
covers Eternal-as-destination).

---

## Cross-cutting observation: pattern-port did not propagate the closure-gate fixes

Stage 39 is a **verbatim** port of the Stage 38 (frame) playbook,
which was itself a verbatim port of the Stage 37 (tier) playbook.
This is good — the family is uniform — except the Stage 38 closure
gate-1 fixes from commit `1c6e047` (H1: `_compatible` arms; H2:
refinement-visiting helpers) were NEVER ported back into the
template. Stage 39 ships **pre-fix** with the same 6+2 holes Stage
38 had pre-closure.

The fix is mechanical:
1. **H1 (5 lines)**: copy the TyFrame `_compatible` arms verbatim
   with `TyTemporal` / `kind`.
2. **H2 (9 lines)**: copy the 6 TyFrame refinement-helper arms
   verbatim with `TyTemporal` / `kind`.

Stage 38 H1+H2 fix landed in commit `1c6e047` with 14 + 14 = 28
lines of canary tests at `test_stage38_frames.py:282-360`. Add a
parallel `_temporal_` test cluster of equivalent size.

The M1/M2/L1/L2 findings are anticipatory and the Stage 38 gate-1
report deferred each of them to gate-2 or later — same call for
Stage 39 unless appetite has shifted.

## Recommended priority for Stage 39 closure gate-1

1. **Must-fix before Stage 39 closes**: H1, H2 — silent acceptance
   holes that undermine the announced purpose of temporal typing.
   Fix is verbatim TyFrame-copy. Confidence: high.
2. **Should-fix before Stage 39 closes**: M2 (remediation hints) —
   cheap; Stage 37+38 precedent for deferring this creates rework
   debt; cross-temporal mistakes are unusually frequent in the
   target AGI temporal-reasoning workload.
3. **Defer to gate-2 or Stage 40**: M1, M3, L1, L2 — design-level
   questions that need a temporal-vs-frame-vs-tier specification
   decision, not just a code patch. L2 is borderline; the test
   coverage gap (no Eternal-as-destination test) is worth a 2-line
   backfill regardless.

**Verdict**: 2 HIGH + 3 MEDIUM + 2 LOW + 1 OBS — gate-1 NOT CLEAN; H1+H2 are blocking, mechanical TyFrame-port fixes.
