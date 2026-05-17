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
