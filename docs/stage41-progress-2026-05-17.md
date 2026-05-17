# Stage 41 Progress - 2026-05-17

## Stage Goal

Stage 41 is **Causality / intent types** — the 5th semantic-type
family extending the AGI quartet completed at Stage 40. Mirrors
the Stage 37/38/39/40 playbook exactly: nominal tag + inner type,
intro/elim builtins, identity IR lowering, preemptive
AD_KNOWN_PURE_CALLS + `_FRAME_IDENTITY_AD_NAMES` registration,
all 8 type-system helper arms added at Inc 1 (preemptive H1/H2/
H3/F2/F6 closure), 3-clean-gate closure.

Beginner meaning: real-world AGI reasoning needs to track WHY
something is true beyond "it was observed" (Stage 40 Modal). The
robot reaching position X is a Cause if it triggers a downstream
plan revision, an Effect if it followed from some upstream
decision, a Joint observation if multiple causes contributed. AGI
that mis-attributes causation makes systematically wrong decisions
about which knob to turn next.

## The 4 causal kinds

- **Cause<T>** — this value is an upstream input to downstream
  reasoning ("this is what caused the next thing").
- **Effect<T>** — this value is a downstream output of upstream
  reasoning ("this happened because of something").
- **Joint<T>** — this value aggregates multiple causes ("multiple
  things led to this; the system has multiple knobs to turn").
- **Independent<T>** — this value is causally isolated ("no
  upstream / no downstream; safe to reason about in isolation").

## Why these 4 (not Pearl-style do-calculus)

These 4 are AGI-decision-loop primitives, not formal causal-
inference operators. Pearl's do-calculus operates on the
inference layer; Stage 41 operates on the typing layer. A formal
do-operator could later compose with the Stage 41 wrappers
(`do<Cause<T>>` would say "intervene on this upstream value")
without changing the Phase-0 wrapper-shift semantics.

## Increment 0 - Open Stage 41 (Convention Declaration)

Same conventions as Stage 37/38/39/40: combined audit-and-fix per
increment, 3 consecutive clean audit gates close the stage, self-
host gate green before every commit, Phase-0 Python-side
scaffolding (in-Helix migration is a Phase 2 finalization task).
Stage 40 lesson applied: add TyCausal arms to all 8 type-system
helpers at Inc 1 rather than at audit time.

## Increment 1 - Causal Constructors + Eliminators

8 new typecheck-recognized builtins (4 kinds × 2):

- `into_cause(v: T) -> Cause<T>` / `from_cause(m: Cause<T>) -> T`
- `into_effect(v: T) -> Effect<T>` / `from_effect(m: Effect<T>) -> T`
- `into_joint(v: T) -> Joint<T>` / `from_joint(m: Joint<T>) -> T`
- `into_independent(v: T) -> Independent<T>` /
  `from_independent(m: Independent<T>) -> T`

All 8 lower as identity at IR. Wrong-kind `from_X(into_Y(...))`
fires typecheck diagnostic. All 8 added preemptively to
AD_KNOWN_PURE_CALLS + `_FRAME_IDENTITY_AD_NAMES`.

8 type-system helpers gain TyCausal arms in the same commit:
`_compatible`, `_refinement_shape_exact`, `_refinement_proof_carried`,
`_erase_refinement`, `_contains_refinement`,
`_is_refinement_container`, `_contains_refined_function`,
`_contains_unknown_type`. Also `_resolve_type` modal-style arm +
`_fmt` arm.

## Increment 2 - Causal Transitions

3 cross-causal transition builtins for the meaningful directions:

- `propagate(c: Cause<T>) -> Effect<T>` (this cause has been
  applied; what was an upstream input is now a downstream
  observation — the planner enacts a cause and notes the effect).
- `aggregate(e: Effect<T>) -> Joint<T>` (this single-source
  effect becomes a multi-source joint observation when other
  causes also contribute — used at the type level to track
  "this output now depends on multiple knobs").
- `isolate(j: Joint<T>) -> Independent<T>` (the AGI has confirmed
  by experiment that no upstream actually matters for this value
  — the multi-cause observation has been collapsed to causal
  independence; an information-gain action result).

Same gate-3 lesson applied preemptively: the F1 cross-modal
laundering guard at Stage 40 (`into_X(from_Y(v))` rejection) is
also applied at Stage 41 — `into_cause(from_effect(e))` and other
cross-kind direct strip-rewraps are rejected with kind-specific
hints pointing at the audited transition.

## Increment 3 - Causal Lifecycle Dogfood

`helixc/examples/dogfood_14_causal_lifecycle.hx` — 3 propositions
flow through the full Cause → Effect (propagate) → Joint
(aggregate) → Independent (isolate) lifecycle, then unwrap.
Witness pattern matches Stage 37/38/39/40 dogfoods (product of
binary witnesses × sum gates exit 42). Plus a cross-stage
composition probe: `Known<Cause<f32>>` = "I directly observed
that this was a cause" — the 5-stack quintet (memory / spatial /
temporal / modal / causal) composes orthogonally at the type
level.

## Increment 4 - Stage 41 Closure (3/3 clean gates)

Same protocol as Stage 35/36/37/38/39/40.
