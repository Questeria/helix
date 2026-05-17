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

### Gate 1 (post-Inc-3) — fix sweep landed

3 audit lanes spawned. De-duped findings:
- **F1 inner_is_shadowed parity (HIGH)** — flagged by all 3
  lanes (silent-failure F1, type-design F1, code-review M1).
  The Stage 40 closure gate-3 H1 amendment (skip F1 launder
  guard when inner from_X is user-shadowed) was missed when
  copying the guard forward to Stage 41. Without it, shadowing
  `from_cause` + writing `into_effect(from_cause(c))` produced
  1 shadow + 1 arg-mismatch + 1 spurious launder = 3 errors
  (modal version produced 2).
- **Safety-anchored hints (LOW)** — 6 reverse causal
  directions (effect->cause, joint->cause, independent->cause,
  independent->joint, independent->effect, joint->effect) need
  kind-specific framing instead of the generic "Phase-0 has no
  X -> Y transition" fallback that misleadingly suggests a
  future feature when the direction is semantically incoherent.
- **Dogfood witness strength (LOW-1)** — cross-stack probe
  used input 1 (degenerate); changed to 7 so identity-laundering
  bugs that mapped any input to 1 wouldn't silently pass.

Gate-1 fix sweep (commit 246c33f):
- F1 inner_is_shadowed predicate added at causal site, mirrors
  modal site verbatim.
- 6 safety-anchored hints added to `_causal_upgrade_hint`.
- Dogfood cs probe uses 7 (non-degenerate).
- 1 regression test pinning F1 parity.

Deferred (multi-stage refactors, all LOW, explicitly out of
scope mid-stage):
- LOW-2: rename "aggregate return type" diagnostic strings to
  disambiguate from the new `aggregate` builtin (~14 sites).
- LOW-3: rename `_FRAME_IDENTITY_AD_NAMES` to
  `_IDENTITY_AD_CHAIN_RULE_NAMES` (set now covers 5 wrapper
  families, only 12 of 45 entries are frames). Multi-stage
  rename with import lockstep.

### Gate 2 (post-gate-1) — ALL 3 LANES CLEAN

- **silent-failure**: GATE CLEAN. Verified F1 parity fix at
  line 3828-3841 mirrors modal site line 3643-3656 verbatim.
  Safety-anchored hints correctly applied. Dogfood cs probe
  consistent (7 at both production + assertion).
- **type-design**: GATE CLEAN. TyCausal coverage at 9 TyModal
  sites verified parallel. 12-combo cross-causal matrix
  covers all upgrade/downgrade directions.
- **code-review**: GATE CLEAN. 23 tests pass. Working tree
  clean at 246c33f. Deferrals documented.

### Gate 3 (post-gate-2) — ALL 3 LANES CLEAN

- **silent-failure**: GATE 3 CLEAN. All TyCausal coverage
  present; F1 verbatim parity with modal including
  `inner_is_shadowed`. 4 of 12 cross-causal directions
  fall through to generic "Phase-0 has no transition"
  hint — acceptable per gate-1 LOW rubric (only obviously
  incoherent directions need safety-anchored framing).
- **type-design**: GATE 3 CLEAN. Full parity scan across 8
  helpers + AD + IR identity arm shows no asymmetry.
- **code-review**: GATE 3 CLEAN. 23/23 tests pass. Demo
  exits 42. No findings at confidence >= 80.

### STAGE 41 CLOSED 2026-05-17 at Inc 4 (3/3 clean audit gates)

5-stack AGI semantic-type quintet complete:

- Stage 37 — memory tier:   WHERE a fact lives.
- Stage 38 — spatial frame: WHICH coordinate system.
- Stage 39 — temporal kind: WHEN it was true.
- Stage 40 — modal kind:    WHY we accept it.
- Stage 41 — causal kind:   WHY it is true / what it causes.

All five families compose orthogonally at the type level
(e.g. `Known<Past<Cause<f32>>>` = "I directly observed that
this past fact was a cause"). 14 dogfood programs total (was
13). Self-host cascade still byte-identical G2..G4 fixpoint.

Stage 42 opens next per ROADMAP Phase 2.
