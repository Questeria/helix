# Stage 40 Progress - 2026-05-17

## Stage Goal

Stage 40 is **Modal / Epistemic types** per the AGI-shaped
ROADMAP Phase 2 continuation. Completes the semantic-type
quartet started at Stage 37:

- **Stage 37 — memory tier**: WHERE a fact lives
  (working / episodic / semantic / procedural).
- **Stage 38 — spatial frame**: WHICH coordinate system
  a value is expressed in (world / robot / camera).
- **Stage 39 — temporal kind**: WHEN a fact was true
  (past / present / future / eternal).
- **Stage 40 — modal kind**: WHAT EPISTEMIC STATUS a
  proposition has (known / believed / goal / uncertain).

Mirrors the Stage 37/38/39 playbook exactly: nominal tag +
inner type, intro/elim builtins, identity IR lowering,
preemptive AD_KNOWN_PURE_CALLS registration, dogfood,
3-clean-gate closure.

## Beginner meaning

Real-world AGI reasoning needs to track WHY it accepts a
proposition. "The robot is at position X" matters
differently depending on whether the AGI KNOWS this
(direct observation), BELIEVES it (inference from other
facts), wants it to be true (GOAL), or is UNCERTAIN about
it. Stage 40 makes modal status part of the type system so
treating a goal as a known fact (a category mistake at the
heart of many AI safety failures) is caught at compile
time.

## The 4 modal kinds

- **Known<T>** — directly observed / proven fact (highest
  epistemic confidence; "I see this").
- **Believed<T>** — inferred from other facts (could be
  wrong if its sources are wrong; "this should be true").
- **Goal<T>** — desired state, not currently true (the
  AGI is trying to make this true; "I want this").
- **Uncertain<T>** — partial information / probabilistic
  ("might be true; need more data").

## Why these 4 (not Lean / KT4 / etc.)

The 4 picked here are AGI-decision-loop primitives, not
formal-modal-logic operators. A Lean-style necessity/
possibility split is the wrong level — at the level where
an agent decides what to act on, the salient distinctions
are observation vs inference vs goal vs uncertainty. These
4 map directly to the planning/execution loop:

- Known facts ground the current world model.
- Believed facts extend it via inference.
- Goals are the targets the planner moves toward.
- Uncertain facts gate information-gathering actions.

A future increment may add modal transitions (e.g.,
`confirm: Believed<T> -> Known<T>` when a belief is
observed) but Stage 40 Inc 1+2 only ships intro/elim +
one obvious transition (`act_on: Goal<T> -> Known<T>`
when an agent achieves a goal).

## Increment 0 - Open Stage 40 (Convention Declaration)

Same conventions as Stage 37/38/39: combined audit-and-fix
per increment, 3 consecutive clean audit gates close the
stage, self-host gate green before every commit, Phase-0
Python-side scaffolding (in-Helix migration is a Phase 2
finalization task).

## Increment 1 - Modal Constructors + Eliminators

8 new typecheck-recognized builtins (4 kinds × 2):

- `into_known(v: T) -> Known<T>` / `from_known(m: Known<T>) -> T`
- `into_believed(v: T) -> Believed<T>` / `from_believed(m: Believed<T>) -> T`
- `into_goal(v: T) -> Goal<T>` / `from_goal(m: Goal<T>) -> T`
- `into_uncertain(v: T) -> Uncertain<T>` / `from_uncertain(m: Uncertain<T>) -> T`

All 8 lower as identity at IR. Wrong-kind
`from_X(into_Y(...))` fires a typecheck diagnostic. All 8
added preemptively to AD_KNOWN_PURE_CALLS and to
`_FRAME_IDENTITY_AD_NAMES` (chain rule is identity, same
as Stage 38/39 wrappers).

Same 8 type-system helpers gain TyModal arms in the same
commit (close the H1/H2/H3/F2/F6 lessons from Stage 39
preemptively): `_compatible`, `_refinement_shape_exact`,
`_refinement_proof_carried`, `_erase_refinement`,
`_contains_refinement`, `_is_refinement_container`,
`_contains_refined_function`, `_contains_unknown_type`.

## Increment 2 - Modal Transitions (epistemic upgrades)

2 cross-modal transition builtins for the meaningful
directions (the others are deliberately omitted — see
"deferred transitions" below):

- `confirm(b: Believed<T>) -> Known<T>` (an inferred belief
  becomes a known fact when directly observed; the planner
  promotes a working hypothesis to ground truth).
- `act_on(g: Goal<T>) -> Known<T>` (the agent achieves a
  goal; what was desired is now observed-true). This is
  also the natural way to model "execution success":
  Goal<position(0,0)> becomes Known<position(0,0)>.

### Deferred transitions

- `Known -> Believed` (epistemic downgrade): rare and
  dangerous — usually a sign the AGI should re-check.
- `Goal -> Believed`: ambiguous semantics (a "hoped-for"
  belief vs "I have evidence"). Skip until needed.
- `Uncertain -> anything`: requires a confidence threshold
  parameter — bigger spec; defer.
- `confirm`/`act_on` operating on `Eternal<T>` (the Stage 39
  temporal kind): out of scope; Stage 40 modal kinds compose
  with temporal kinds at the type level naturally (e.g.,
  `Known<Past<i32>>` = "I directly observed this past fact"
  vs `Believed<Past<i32>>` = "I inferred this past fact").

## Increment 3 - Modal Lifecycle Dogfood

`helixc/examples/dogfood_13_modal_lifecycle.hx` — 3
propositions flow Goal → Known (act_on) and Believed →
Known (confirm), then unwrap. Plus a recall-style side-
check with the Uncertain kind (no transition; just
intro/elim sanity). Witness pattern matches Stage 37/38/39
dogfoods (product of binary witnesses × sum gates exit 42).

## Increment 4 - Stage 40 Closure (3/3 clean gates)

Same protocol as Stage 35/36/37/38/39.

### Gate-1 fix sweep (commit `e8fb593`)

- **F1 (HIGH conf 90)** — block direct `into_X(from_uncertain(u))`
  Uncertain-laundering. Without this guard, the entire epistemic
  discipline can be bypassed by an unwrap-rewrap and the
  AI-safety motivation for Stage 40 evaporates.
- **F2 (MEDIUM conf 90)** — reject user functions whose name
  shadows a reserved builtin. Stage 36-40 inherited the silent
  dead-coding; Stage 40 makes it acute (`confirm` / `act_on` are
  generic verbs likely to collide with user planning code).

10 regression tests pin both fixes.

### Gate-2 fix sweep (post-audit)

Three specialist auditors (silent-failure-hunter,
type-design-analyzer, code-reviewer) returned 2 HIGH + 2 MEDIUM
+ 5 LOW + 7 OBS against the gate-1 surface. Fix sweep:

- **HIGH silent-failure (cross-modal launder asymmetry)** —
  generalize the gate-1 Uncertain-only guard to ALL cross-modal
  `into_X(from_Y(v))` where X != Y. Phase-0 has only `confirm`
  (Believed -> Known) and `act_on` (Goal -> Known) as audited
  upgrade paths; everything else is rejected with a kind-specific
  hint pointing at the legitimate transition (or noting deferral
  when none exists). 6 new regression tests including a 12-combo
  cross-modal matrix.
- **HIGH type-design (named-binding bypass)** — the F1 guard is
  syntactic; let-binding decomposes the inline pattern and slips
  through. Documented as a Phase-0 known limitation (taint-
  tracking is a future-stage spec); test pins the limitation so
  a future stage can flip the assertion to confirm closure.
- **MEDIUM code-review (F2 cascade)** — pre-fix, a shadowed
  builtin name produced 1 shadow error + N call-site builtin
  errors per use. Track shadowed names in `_shadowed_builtin_names`
  and skip builtin dispatch for those names at call sites; the
  user sees one diagnostic, not a noise cascade.
- **MEDIUM code-review (F1 false-positive on TyUnknown)** — the
  F1 launder guard now requires `arg_tys[0]` to be a real type,
  not TyUnknown. Prevents the structurally-false "launders" message
  when the inner `from_X` itself failed.

### Known limitations (documented for Phase-1 follow-up)

- **F1 syntactic-only**: let-binding (`let r = from_X(v); into_Y(r)`)
  and helper-fn indirection bypass the F1 guard. Phase-1 task: add
  a taint-tracking pass that propagates Uncertain-origin (and
  cross-modal origin) through bindings and call boundaries.
- **TyVar-defer gap** (pre-existing, all 4 wrapper families):
  `fn id[T](p: Known<T>) -> Known<T> { p }` called with concrete
  `Known<i32>` doesn't currently typecheck. Symmetric across
  Stage 37/38/39/40 — not a Stage 40 regression. Phase-1 task.
- **`_FRAME_IDENTITY_AD_NAMES` docstring drift**: the name now
  covers 34 entries (12 frames + 12 temporals + 10 modals).
  Cosmetic rename to `_IDENTITY_WRAPPER_AD_NAMES` deferred so
  Stage 40 doesn't churn a Stage-38-era invariant name.
