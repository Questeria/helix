# Stage 39 Progress - 2026-05-17

## Stage Goal

Stage 39 is **Temporal types** per ROADMAP Phase 2. Mirrors the
Stage 37 (tiered memory) and Stage 38 (spatial frames) playbook
exactly: nominal tag + inner type, intro/elim builtins, identity
IR lowering, preemptive AD_KNOWN_PURE_CALLS registration.

Beginner meaning: real-world AGI reasoning needs to track WHEN a
fact was true. "The robot saw a cat at coordinate X" only matters
if we know whether that was 5 seconds ago or 5 days ago. Stage 39
makes temporal status part of the type system so past/present/future
distinctions are caught at compile time.

## The 4 temporal kinds

- **Past<T>** — happened in the past (immutable history)
- **Present<T>** — happening now (current observation)
- **Future<T>** — will happen (prediction / commitment)
- **Eternal<T>** — timeless / always true (mathematical fact,
  physical constant)

## Increment 0 - Open Stage 39 (Convention Declaration)

Same conventions as Stage 37/38: combined audit-and-fix per
increment, 3 consecutive clean audit gates close the stage,
self-host gate green before every commit, Phase-0 Python-side
scaffolding (in-Helix migration is a Phase 2 finalization task).

## Increment 1 - Temporal Constructors + Eliminators

8 new typecheck-recognized builtins (4 kinds × 2):
- `into_past(v: T) -> Past<T>` / `from_past(m: Past<T>) -> T`
- `into_present(v: T) -> Present<T>` / `from_present(m: Present<T>) -> T`
- `into_future(v: T) -> Future<T>` / `from_future(m: Future<T>) -> T`
- `into_eternal(v: T) -> Eternal<T>` / `from_eternal(m: Eternal<T>) -> T`

All 8 lower as identity at IR. Wrong-kind `from_X(into_Y(...))`
fires a typecheck diagnostic. All 8 added preemptively to
AD_KNOWN_PURE_CALLS.

## Increment 2 - Temporal Transitions

4 cross-temporal transition builtins (the meaningful directions):
- `to_past(p: Present<T>) -> Past<T>` (present becomes past as
  time passes)
- `forecast(p: Present<T>) -> Future<T>` (project current state
  forward)
- `recall_past(p: Past<T>) -> Present<T>` (bring a past observation
  into current focus for reasoning)
- `actualize(f: Future<T>) -> Present<T>` (a predicted future
  becomes the present when it arrives)

Eternal<T> doesn't transition — it's timeless. Transitions involving
Eternal can be added in a later increment if needed.

## Increment 3 - Temporal Lifecycle Dogfood

`helixc/examples/dogfood_12_temporal_lifecycle.hx` — 3 observations
flow Present → Future (forecast) → Present (actualize) → Past
(to_past), then unwrap. Witness pattern matches Stage 37/38 dogfoods.

## Increment 4 - Stage 39 Closure (3/3 clean gates)

Same protocol as Stage 35/36/37/38.
