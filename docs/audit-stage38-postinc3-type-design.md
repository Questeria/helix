# Stage 38 post-Inc-3 type-design audit

**HEAD**: b427f4f (Stage 38 Inc 3: spatial-frame lifecycle dogfood)
**Scope**: `git diff 86c2ce4~1..HEAD -- helixc/frontend/typecheck.py
helixc/frontend/autodiff.py helixc/ir/lower_ast.py
helixc/examples/dogfood_11_spatial_frames.hx
helixc/tests/test_stage38_frames.py`
**Date**: 2026-05-16
**Auditor brief**: type-design lens — looseness, strictness, family
asymmetry, encoded-as-convention smells, runtime-vs-compile-time gaps.

## Summary table

| ID | Sev | Conf | Title |
|----|-----|------|-------|
| H1 | HIGH | 90 | `TyFrame` missing from `_compatible` wrapper arms — cross-frame and bare-vs-wrapped silent acceptance at the function-call boundary |
| H2 | HIGH | 88 | `TyFrame` missing from refinement-visiting helpers — refinements under a frame wrapper are invisible to 5 refinement passes |
| M1 | MEDIUM | 80 | `TyFrame.frame: str` (and `TyMemTier.tier: str`) encode a closed 3-value enum as a string — typo-tolerant at construction, no enum-exhaustiveness anywhere |
| M2 | MEDIUM | 75 | All 12 new Stage 38 builtins ship without remediation hints — same family-standard-hint asymmetry Stage 37 post-closure fixed for `parent_at` |
| M3 | MEDIUM | 70 | All 12 new Stage 38 builtins silently return `TyUnknown` on wrong arity — no arity diagnostic, same hole the tier family already had |
| L1 | LOW | 70 | `into_X` accepts `TyFrame` as inner — `into_world(into_robot(x))` builds `WorldFrame<RobotFrame<T>>` with no diagnostic; family-symmetric with tiers but still a real smell for frames |
| L2 | LOW | 60 | `into_X(T) -> FrameName<T>` accepts any `T` — `WorldFrame<String>` / `WorldFrame<TyFn(...)>` typechecks; spatial frames over non-numeric inners are nonsensical but unrejected |

---

## Findings

### H1: TyFrame missing from `_compatible` wrapper arms — cross-frame and bare-vs-wrapped silent acceptance at the function-call boundary (HIGH, confidence 90/100)

**Location**: `helixc/frontend/typecheck.py:6532-6635` (`_compatible`)
**Pattern**: Family-asymmetry hole in the type-compatibility predicate.
Every other wrapper type (`TyDiff`, `TyLogic`, `TyMemTier`, `TyQuote`,
`TyTuple`, `TyArray`, `TyRef`, `TyPtr`, `TyFn`, `TyTensor`, `TyTile`)
has BOTH a bilateral arm (`isinstance(a, T) and isinstance(b, T)`)
AND a unilateral rejection arm (`isinstance(a, T) or isinstance(b, T):
return False`) that closes the "raw value passed where wrapped expected
/ wrapped value passed where raw expected" silent-acceptance hole
explicitly named in the cycle-3 D1 closure comments at
`typecheck.py:6543-6547`. **`TyFrame` has neither arm.**

**Citation** (grep-able):
```
grep -n "isinstance(a, TyFrame)" helixc/frontend/typecheck.py
  (no matches)
grep -n "isinstance(a, TyMemTier)" helixc/frontend/typecheck.py
  5366:        if isinstance(a, TyMemTier) and isinstance(b, TyMemTier):
  6532:        if isinstance(a, TyMemTier) and isinstance(b, TyMemTier):
  6534:        if isinstance(a, TyMemTier) or isinstance(b, TyMemTier):
```

**What survives unchecked** (each of these passes `_compatible` today
because the fallthrough is `a == b` at line 6636, which catches the
cross-frame case via frozen-dataclass equality but skips the
intended-wrapper-vs-raw asymmetry the family established):

- `fn takes_world(w: WorldFrame<i32>)` called with bare `i32`:
  `_compatible(TyFrame, TyPrim)` skips every typed arm and reaches
  `a == b` which returns False — so this case is actually caught.
- `fn takes_raw(i: i32)` called with `WorldFrame<i32>`:
  symmetric — `a == b` catches it.
- `fn takes_world(w: WorldFrame<i32>)` called with `RobotFrame<i32>`:
  `a == b` on two TyFrame instances differs by `frame=` field
  — `"world" != "robot"` → False → caught.

So at FIRST glance the dataclass equality fallthrough plugs the hole.
But four real cases still leak:

1. **Refinement-shape inner mismatch** (interaction with H2): if
   either side has a refined inner, the bare `a == b` compares
   TyRefined identities, not refinement-shape — so
   `WorldFrame<{x: i32 | x > 0}>` and `WorldFrame<i32>` fall to
   `a == b` which gives False (refinement-typed != bare-typed),
   when the intended semantics is "refined inner is more specific
   than bare inner and the wrapper should propagate that knowledge".
   Every other wrapper goes through `_compatible(a.inner, b.inner)`
   which DOES delegate refinement reasoning correctly.
2. **TyVar / TySize inner deferral**: every other wrapper recursively
   calls `_compatible(a.inner, b.inner)`, which has dedicated arms
   for generic-parameter deferral (TyVar) and size deferral (TySize).
   The frame's `a == b` fallthrough does NOT defer — so a
   `WorldFrame<TyVar("T")>` parameter never matches against a
   `WorldFrame<i32>` call site, where the tier equivalent does.
3. **TyArray inner**: `WorldFrame<[i32; 4]>` vs `WorldFrame<[i32; N]>`
   — the tier path uses `_compatible(elem, elem)` which delegates
   to `_size_compatible`; the frame path uses `a == b` which
   compares sizes by identity.
4. **Future cross-family wrappers**: when Phase-1 adds another
   wrapper kind (Skill subdomain, sensor-modality, etc.), the
   family-established invariant is "each wrapper has a unilateral
   rejection arm" — adding new arms in the same order is mechanical.
   Skipping the TyFrame arm makes the implicit convention easier
   to miss next time.

**Impact**: cross-frame mistakes at the function-call boundary that
involve refined inners, generic parameters, or shape-symbolic inners
slip through silently. The whole point of frame typing (per
`TyFrame` docstring at typecheck.py:243-249: "Cross-frame operations
require explicit transforms") is undermined for any non-trivial
inner type.

**Fix**: Add two arms to `_compatible`, mirroring the TyMemTier
arms exactly (`typecheck.py:6532-6535`):

```py
if isinstance(a, TyFrame) and isinstance(b, TyFrame):
    return a.frame == b.frame and self._compatible(a.inner, b.inner)
if isinstance(a, TyFrame) or isinstance(b, TyFrame):
    return False
```

Insert directly after the TyMemTier pair at line 6535. Five lines.
No public-API change. Add canary tests mirroring the existing
tier-vs-bare and tier-vs-other-tier function-boundary tests for
frames (e.g., `test_stage38_world_frame_param_rejects_bare_i32_call`,
`test_stage38_world_frame_param_with_refined_inner_compatible`).

---

### H2: TyFrame missing from refinement-visiting helpers — refinements under a frame wrapper are invisible to 5 refinement passes (HIGH, confidence 88/100)

**Location**: `helixc/frontend/typecheck.py:5392-5568` (5 visitor
helpers).
**Pattern**: New wrapper added; structural-recursion visitors not
updated. Every TyMemTier-handling site has a sibling that walks
through the wrapper to its inner type. TyFrame is missing from all
of them.

**Citation** (grep-able, paired with the TyMemTier reference site):

| Helper | Line (TyMemTier present) | TyFrame status |
|--------|--------------------------|----------------|
| `_erase_refinement` | 5409 | **missing** |
| `_refinement_shape_exact` | 4710 | **missing** |
| `_contains_refinement` | 5519 | **missing** |
| `_is_refinement_container` | 5538 | **missing** (not in the tuple) |
| `_contains_refined_function` | 5558 | **missing** |

```
grep -n "TyMemTier" helixc/frontend/typecheck.py | grep -E "5409|4710|5519|5538|5558"
  4710:        if isinstance(target, TyMemTier) and isinstance(value_ty, TyMemTier):
  5409:        if isinstance(ty, TyMemTier):
  5519:        if isinstance(ty, TyMemTier):
  5538:            TyMemTier, TyTensor, TyTile,
  5558:        if isinstance(ty, TyMemTier):
grep -n "TyFrame" helixc/frontend/typecheck.py
  242:class TyFrame(Type):
  3198:                    if (isinstance(arg_tys[0], TyFrame)
  3225:                    if (isinstance(arg_tys[0], TyFrame)
  6698:        if isinstance(t, TyFrame):
```

3 use-sites total, vs the 11+ TyMemTier has. The 5 visitor gaps are
all of the same kind: a structural-recursion arm that should walk
through the wrapper to its inner type, but doesn't because the
wrapper kind wasn't added to the dispatch list.

**Impact** (concrete, reproducible):

- `WorldFrame<{x: i32 | x > 0}>` as a function parameter type: the
  call-site refinement-shape check at `typecheck.py:4710` falls
  through (TyFrame arm missing). The bare `a == b` fallback at
  the end of `_refinement_shape_exact` then rejects any concrete
  call-site value because `TyFrame("world", TyRefined(...)) !=
  TyFrame("world", TyPrim("i32"))`. Net effect: refined-inner
  frame parameters become uncallable.
- `_contains_refinement(WorldFrame<{x:i32|...}>)` returns False
  — so any pass that says "if this type contains a refinement,
  emit refinement-checking ops" will skip the frame-wrapped
  refinement, producing a silent runtime hole.
- `_is_refinement_container` is the predicate that the rest of
  the refinement system uses to decide which types could plausibly
  carry refinement metadata. Returning False for TyFrame means
  every container-aware refinement transform skips frame wrappers
  entirely.

**Fix**: One line per helper, mirroring the TyMemTier arm verbatim
with `TyFrame(ty.frame, ...)` substituted. Total ~7 lines:

```py
# _erase_refinement (5410):
if isinstance(ty, TyFrame):
    return TyFrame(ty.frame, self._erase_refinement(ty.inner))
# _refinement_shape_exact (4713):
if isinstance(target, TyFrame) and isinstance(value_ty, TyFrame):
    return (target.frame == value_ty.frame
            and self._refinement_shape_exact(value_ty.inner, target.inner))
# _contains_refinement (5520):
if isinstance(ty, TyFrame):
    return self._contains_refinement(ty.inner, _seen_structs)
# _is_refinement_container (5538): add TyFrame to the tuple.
# _contains_refined_function (5559):
if isinstance(ty, TyFrame):
    return self._contains_refined_function(ty.inner)
```

Each is a verbatim TyMemTier copy. Canary tests should put a refined
inner under a frame wrapper at a function parameter position; the
test passes today only because no such test exists.

---

### M1: `TyFrame.frame: str` (and `TyMemTier.tier: str`) encode a closed 3-value enum as a string — typo-tolerant at construction, no enum-exhaustiveness anywhere (MEDIUM, confidence 80/100)

**Location**: `helixc/frontend/typecheck.py:250` (`frame: str`)
**Pattern**: Closed finite domain encoded as a stringly-typed field
on a frozen dataclass. No validation; no exhaustiveness checking;
typos compile.

**Citation**:
```
grep -n 'frame: str' helixc/frontend/typecheck.py
  250:    frame: str       # "world", "robot", "camera"
grep -n 'tier: str' helixc/frontend/typecheck.py
  237:    tier: str        # "working", "episodic", "semantic", "procedural"
```

The user prompt explicitly flagged this: "Is TyFrame(frame_name: str,
inner: Type) the right encoding, or should `frame` be a typed enum?
(Compare Stage 37 TyMemTier.)" — the answer is that **both** TyMemTier
and TyFrame have this smell. Stage 38 inherited it from Stage 37
verbatim.

Concrete consequences:

- `TyFrame(frame="wrold", inner=...)` constructs without error.
  Downstream `_frame_elim[bn] == "world"` checks then never match,
  producing surprising "from_world() requires WorldFrame<T>, got
  WorldFrame<T>" diagnostics (the printed inner type looks identical
  because `_fmt` at line 6699 maps unknown frame keys to the raw
  string).
- `_fmt`'s `cap.get(t.frame, t.frame)` at line 6699 falls back to
  the raw string for unknown frames — so a typo'd frame string
  surfaces as e.g. "wrold<i32>" in user diagnostics rather than
  raising or asserting.
- Adding a fourth frame (e.g. `SensorFrame`) requires touching at
  least 5 string-literal sites scattered across `_resolve_type`,
  `_frame_intro`, `_frame_elim`, `_frame_transforms`, and `_fmt` —
  with no compile-time check that they're all updated. The Stage 37
  closure gate-1 LOW finding (S37-CLEAN1-001) is exactly this class
  for the tier family.

**Why MEDIUM not HIGH**: in practice all `TyFrame` construction in
Stage 38 routes through dictionaries (`frame_map` at 1100,
`_frame_intro` at 3183, `_frame_elim` at 3191, `_frame_transforms`
at 3215) whose values are hard-coded literals — no typo can sneak
in through user code. The bug-class is closed by construction in
Phase-0, but the type doesn't enforce its own invariants and the
"closed by construction" property is one accidental
`TyFrame(frame=user_string, ...)` away from being violated.

**Fix options** (in increasing strength):

1. **Validate in `__post_init__`** — frozen dataclass override:
   ```py
   def __post_init__(self):
       if self.frame not in {"world", "robot", "camera"}:
           raise ValueError(f"TyFrame.frame must be one of "
                            f"world/robot/camera, got {self.frame!r}")
   ```
   Catches the bug at construction; minimal code; ~5 lines.

2. **Use a typed enum**:
   ```py
   class FrameKind(Enum):
       WORLD = "world"
       ROBOT = "robot"
       CAMERA = "camera"
   ```
   Stronger invariant; forces every site (including `_fmt`, the
   transforms dict) to use the enum members. Touches more sites
   but the enum-exhaustiveness check at mypy time becomes free.
   Same lift required for TyMemTier (8 sites) — co-fixing both is
   the right call so the family stays uniform.

3. **Defer**: document explicitly that all `TyFrame` construction
   must route through the dictionaries above; add a unit test that
   pickle-roundtrips the dictionaries to confirm they cover the
   full enum set. Cheapest, weakest.

Recommend (1) for this stage and (2) when next refactoring TyMemTier
— the closed-domain smell is real but the construction discipline
in Stage 38 makes it latent rather than active.

---

### M2: All 12 new Stage 38 builtins ship without remediation hints — same family-standard-hint asymmetry Stage 37 post-closure fixed for `parent_at` (MEDIUM, confidence 75/100)

**Location**: `helixc/frontend/typecheck.py:3196-3206` (from_X),
`3223-3234` (transforms).
**Pattern**: Type-error diagnostic format inconsistency. The Stage 37
post-closure M2 fix (commit `a8ab17b`) explicitly addressed this for
the strict-i32 family by adding `_strict_i32_truncation_hint` — the
family-standard "pre-Inc-N also accepted i64/u32/u64" parenthetical
remediation hint that `parent_at` was previously missing. The Stage
38 frame family establishes the OPPOSITE convention (no hints), so
the asymmetry hasn't yet bitten — but adding a single hint later
will force a parallel sweep of all 12 sites.

**Citation**:
```
grep -n '"{bn}() requires' helixc/frontend/typecheck.py
  3171:                        f"{bn}() requires "       # tier unwrap_X
  3202:                        f"{bn}() requires "       # frame from_X
  3230:                        f"{bn}() requires "       # frame transform
```

The bare diagnostic is:
```
from_world() requires WorldFrame<T>, got RobotFrame<i32>
```

Compare the strict-i32 family hint:
```
parent_at(handle, slot): arg 1 must be exactly i32, got Logic<i32>
   (pre-Inc-14 also accepted i64/u32/u64 but those silently
    truncated in downstream arena read)
```

The hint adds two things: (a) the historical context that explains
why the constraint exists, (b) the immediate corrective action
("use into_world(your_value) first, or cast through from_robot()
if your value is already in robot frame"). The frame diagnostics
have neither.

**Why MEDIUM not LOW**: The user prompt flagged this exact pattern
from the Stage 37 post-closure audit. The fix is cheap (one hint
function used in 6 sites + 6 sites = 12), the cost of NOT fixing
compounds with each cross-frame mistake a user makes in real code.
A reasonable hint for from_X:

```
from_world() requires WorldFrame<T>, got RobotFrame<i32>
   (did you mean from_robot, or apply robot_to_world first?)
```

And for transforms:

```
world_to_robot() requires WorldFrame<T>, got RobotFrame<i32>
   (input is already in robot frame; apply robot_to_world if you
    want world-frame output, or use the identity if no change needed)
```

**Why not HIGH**: the family is internally consistent today (all 12
sites use the same hintless format). The smell is anticipatory —
when the FIRST hint gets added to one site, the asymmetry will be
real, and history (Stage 37 parent_at) shows that's likely.

**Fix**: Define a `_frame_mismatch_hint(want: str, got: Type) -> str`
helper. Apply to all 12 sites. Same shape as
`_strict_i32_truncation_hint` (typecheck.py:6183-6199). ~15 lines
total.

---

### M3: All 12 new Stage 38 builtins silently return `TyUnknown` on wrong arity — no arity diagnostic, same hole the tier family already had (MEDIUM, confidence 70/100)

**Location**: `helixc/frontend/typecheck.py:3188, 3196, 3223` (the
`bn in {...} and len(arg_tys) == 1` guards).
**Pattern**: The arity check is part of the predicate that selects
the per-builtin arm; if arity mismatches, the arm doesn't fire and
execution falls through to the `if expr.callee.name in self.functions`
arm at line 3283. Since `into_world` etc. are builtins (NOT in
`self.functions`), the dispatch eventually returns
`TyUnknown(hint="call")` at line 3302 without ever emitting an
arity-mismatch diagnostic.

**Citation**:
```
grep -n "len(arg_tys) == 1" helixc/frontend/typecheck.py | head -10
  3156:                if bn in _tier_intro_elim and len(arg_tys) == 1:  # tier
  3165:                if bn in _tier_unwrap and len(arg_tys) == 1:      # tier
  3188:                if bn in _frame_intro and len(arg_tys) == 1:      # NEW
  3196:                if bn in _frame_elim and len(arg_tys) == 1:       # NEW
  3223:                if bn in _frame_transforms and len(arg_tys) == 1: # NEW
```

Compare the strict-i32 family's pattern which DOES emit a per-arity
arm (line 2969 `if bn == "register_derivation" and len(arg_tys) == 2`)
plus an explicit "got N args" diagnostic downstream — those builtins
catch arity mismatches. The frame/tier families do not.

**Reproducible silent hole**:
```hx
fn main() -> i32 {
    let f = into_world();          // arity 0, no error
    let g = into_world(1, 2);      // arity 2, no error
    0
}
```
Result: `f` and `g` both get type `TyUnknown(hint="call")`; downstream
uses surface as unrelated "unknown type" cascades rather than the
true root cause "into_world expects 1 arg".

**Why MEDIUM**: this is inherited from Stage 37 (the tier intro_elim/
unwrap arms at 3156/3165 have the same hole), so Stage 38 didn't
introduce it. But Stage 38 doubled the surface (12 new builtins) and
the cascade-diagnostic confusion that comes from `TyUnknown` returns
is a real user-experience regression. Flagged as MEDIUM rather than
LOW because the user explicitly asked about "places where types are
too loose" and arity-silence is exactly that.

**Why not HIGH**: no actual unsoundness — the program still type-fails
downstream, just with a confusing cascade. Real-world Phase-0 users
hit this at most once per builtin per learning curve.

**Fix**: Add explicit arity-mismatch diagnostics outside the arity
guard, e.g.:

```py
if bn in _frame_intro:
    if len(arg_tys) != 1:
        self.errors.append(TypeError_(
            f"{bn}() takes exactly 1 argument, got {len(arg_tys)}",
            expr.span,
        ))
        return TyUnknown(hint=bn)
    return TyFrame(frame=_frame_intro[bn], inner=arg_tys[0])
```

Repeat for `_frame_elim` and `_frame_transforms`. Also consider
backfilling the tier family for symmetry. ~30 lines total across both
families.

---

### L1: `into_X` accepts `TyFrame` as inner — `into_world(into_robot(x))` builds `WorldFrame<RobotFrame<T>>` with no diagnostic; family-symmetric with tiers but still a real smell for frames (LOW, confidence 70/100)

**Location**: `helixc/frontend/typecheck.py:3188-3190`.
**Pattern**: An introducer that accepts already-introduced values
without rejection. Tiers tolerate this (e.g.,
`EpisodicMem<WorkingMem<T>>` could plausibly model "a working-memory
slot that was promoted to episodic"). Frames do not — a coordinate
cannot meaningfully be "in the world frame inside the robot frame";
the second wrap is semantically nonsense.

**Citation**:
```
grep -n "_frame_intro and len(arg_tys)" helixc/frontend/typecheck.py
  3188:                if bn in _frame_intro and len(arg_tys) == 1:
```

The `TyFrame(frame=_frame_intro[bn], inner=arg_tys[0])` line at 3189-90
does not inspect `arg_tys[0]`; if it's already a `TyFrame`, the new
wrapper is silently stacked.

**Why LOW**: family-symmetric with tiers (the Stage 37 pattern allows
the same), and the dogfood at `examples/dogfood_11_spatial_frames.hx`
doesn't exercise nested-frame paths. No active bug yet.

**Why flagged**: the user explicitly asked whether frame and tier
families should differ in invariants. Frames have a stronger
exclusivity property (a coord is in exactly one frame at a time);
tiers can layer (an episodic memory of a working-memory snapshot).
Stage 38 currently treats them identically, losing the stronger
frame invariant.

**Fix** (optional, defer to gate-2 audit cycle):
```py
if bn in _frame_intro and len(arg_tys) == 1:
    if isinstance(arg_tys[0], TyFrame):
        self.errors.append(TypeError_(
            f"{bn}() input is already in {arg_tys[0].frame} frame; "
            f"use a {arg_tys[0].frame}_to_{_frame_intro[bn]}() "
            f"transform to switch frames",
            expr.span,
        ))
        return TyUnknown(hint=bn)
    return TyFrame(frame=_frame_intro[bn], inner=arg_tys[0])
```

---

### L2: `into_X(T) -> FrameName<T>` accepts any `T` — `WorldFrame<String>` / `WorldFrame<TyFn(...)>` typechecks; spatial frames over non-numeric inners are nonsensical but unrejected (LOW, confidence 60/100)

**Location**: `helixc/frontend/typecheck.py:3188-3190`.
**Pattern**: Generic introducer with no inner-type predicate. Real
spatial frames apply only to numeric-vector-like inners (`Vec3<f32>`,
`(f32, f32, f32)`, `[f32; 3]`). `WorldFrame<String>` typechecks today.

**Why LOW**: Phase-0 dogfood uses `WorldFrame<i32>` deliberately (the
dogfood comment at `dogfood_11_spatial_frames.hx:34` says "exit code
42 iff THREE independent observations cycle through all three frames
correctly" — `i32` is the simplest payload that exercises the
type-system plumbing without dragging in vector types). Constraining
inner to numeric would force the dogfood and tests to lift the
payload type, which is premature for Phase-0.

**Why flagged**: real-world usage will hit `WorldFrame<String>` and
silently produce nonsense. Phase-1's first "real" spatial example will
need a TyFrame inner-predicate; the Stage 38 design hasn't reserved
the constraint slot.

**Fix** (Phase-1+, do not block Stage 38):
- Add a `valid_inners: ClassVar[Predicate]` to `TyFrame` (frame-kind
  may dictate different valid inners — e.g., camera frame might allow
  `(f32, f32)` 2D pixel coords, world might require 3D).
- For Phase-0, document the deferral in the `TyFrame` docstring at
  typecheck.py:243-249.

---

## Cross-cutting observation: family-uniformity vs family-specialization

Stage 37 (tiers) and Stage 38 (frames) currently share an identical
type-design pattern: a frozen dataclass with a `str` discriminator
plus a generic inner. The Stage 38 implementation is a verbatim
TyMemTier port — same dict-driven dispatch, same identity lowering,
same `_fmt` capitalization map.

**Pro**: Stage 38 inherits Stage 37's mature design with zero new
risk surface. Family symmetry is good for cognitive load.

**Con**: The two domains have different invariants (tier layering
is meaningful, frame layering is not — see L1; tier exclusivity is
weak — HBM ⊆ DDR per the comment at line 6525-6531, frame exclusivity
is total). Treating them identically loses the stronger frame
invariants.

The H1 + H2 findings are the most urgent — they're "Stage 38 is missing
the family-required arms that Stage 37 has", not "Stage 38 should
differ from Stage 37". Fixing those brings TyFrame to parity with
TyMemTier. The M1/L1/L2 findings are about whether parity-with-tiers
is the right design at all, given frames' stronger invariants —
those are gate-2+ work, not Stage 38 closure-gate work.

## Recommended priority for Stage 38 closure gate

1. **Must-fix before Stage 38 closes**: H1, H2 — these are silent
   acceptance holes that undermine the announced purpose of frame
   typing. Fix is mechanical (verbatim TyMemTier-copy arms).
2. **Should-fix before Stage 38 closes**: M2 (remediation hints) —
   cheap; the Stage 37 post-closure precedent makes deferring this
   create rework debt.
3. **Defer to gate-2 or Stage 39**: M1, M3, L1, L2 — design-level
   questions that need a frame-vs-tier specification decision, not
   just a code patch.

