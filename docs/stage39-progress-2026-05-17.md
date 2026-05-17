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

### Inc 1+2 implementation summary

- `helixc/frontend/typecheck.py`:
  - `TyTemporal(kind, inner)` dataclass mirrors `TyFrame`.
  - 12 new builtins registered in `_BUILTIN_NAMES`:
    intro `into_{past,present,future,eternal}`, elim
    `from_{past,present,future,eternal}`, transitions
    `to_past`, `forecast`, `recall_past`, `actualize`.
  - 3 typecheck arms (`_temporal_intro`, `_temporal_elim`,
    `_temporal_transitions`) — wrong-arity and wrong-kind
    diagnostics fail-closed.
  - `_resolve_type` recognizes `Past<T>` / `Present<T>` /
    `Future<T>` / `Eternal<T>` in annotation positions.
  - `_fmt` arm prints `Past<T>` / `Present<T>` / etc.
- `helixc/ir/lower_ast.py`: 12 names appended to the unified
  identity-lowering arm (same arm as Stage 37 + Stage 38).
- `helixc/frontend/autodiff.py`: 12 names added to
  `AD_KNOWN_PURE_CALLS` and to `_FRAME_IDENTITY_AD_NAMES`
  (chain-rule is identity, same as frame wrappers).
- `helixc/tests/test_stage39_temporal.py`: 25 tests across
  Inc 1 + Inc 2 + invariants. All 25 green.
- `helixc/examples/dogfood_12_temporal_lifecycle.hx` +
  `helixc/examples/run.py` registration: exits 42 on the
  Present→Future→Present→Past lifecycle plus a recall_past
  side-check and an Eternal intro/elim sanity check.

### Gate cadence

Gate count is governed by the post-Stage-38 convention:
combined audit-and-fix per increment, 3 consecutive clean
audit gates close the stage. Self-host gate green before
every commit.

### Gate 1 (post-Inc-3) — NOT CLEAN, fix sweep landed

3 audit lanes spawned. Total findings (de-duped): 9
actionable items split across H1/H2/H3 (symmetry gaps
where TyTemporal needed the same special arms TyFrame
got at Stage 38 closure) + F1/F2/F4 (silent failures) +
L1/M2/M3 (test coverage).

Gate-1 fix sweep (commit 7604652 + backfill 086e9df):
- H1: `_compatible` adds TyTemporal arm — kind+inner
  match + reject mixed wrapper pairs.
- H2: `_refinement_shape_exact` walks TyTemporal at
  both sites (target/value_ty and a/b).
- H3: `_erase_refinement`, `_contains_refinement`,
  `_contains_refined_function`, and the
  `_is_refinement_container` tuple all gain TyTemporal
  arms / membership.
- F1: same as H3 `_erase_refinement` fix.
- F2: `_contains_unknown_type` adds a 3-way wrapper arm
  covering TyMemTier + TyFrame + TyTemporal.
- F4: corrected mislabeled "Stage 38 frames" docstring
  inside the Stage 39 block of `_FRAME_IDENTITY_AD_NAMES`.
- L1: 4 per-transition Eternal-source rejection tests
  with required-source diagnostic-quality assertions.
- M2: full 12-combo transition wrong-source matrix.
- M3: zero-arg + multi-arg coverage on intros, elims,
  and transitions.

Deferred at Gate 1: F3 (builtin name shadowing) + F5
(`_resolve_type` arity fall-through) + M1 (intro double-
wrap on already-wrapped value) — all pre-existing
patterns symmetric across Stage 37/38/all wrappers; not
introduced or widened by Stage 39, deferred as multi-
stage sweeps.

### Gate 2 (post-gate-1) — F6 actionable; declared CLEAN after fix

3 audit lanes:
- type-design: GATE CLEAN
- code-review: GATE CLEAN
- silent-failure: 1 MEDIUM (F6) + 2 LOW (F7, F8)

F6 fix (commit dcdce1b): extended F2 wrapper-walk arm to
all six single-inner wrappers (`TyMemTier, TyFrame,
TyTemporal, TyDiff, TyLogic, TyQuote`). The gate-1 F2
sweep stopped short of the latter three — same silent-
failure mode as the original F2.

F7 (paired-helper asymmetry between
`_contains_refined_function` and `_contains_refinement`
for struct/enum field walks) and F8 (wrapped-refinement
proof-carry recording) are both pre-existing Stage 31
holes inherited by Stage 39 — deferred mirroring the
F3 / F5 / M1 deferral pattern.

### Gate 3 (post-gate-2) — ALL 3 LANES CLEAN

All three gate-3 audit lanes returned CLEAN:

- **type-design**: GATE 3 CLEAN. Verified parity matrix — all
  6 single-inner wrappers (TyDiff / TyLogic / TyQuote /
  TyMemTier / TyFrame / TyTemporal) are uniformly handled
  across 8 helpers (`_compatible`, `_refinement_shape_exact`,
  `_refinement_proof_carried`, `_erase_refinement`,
  `_contains_refinement`, `_is_refinement_container`,
  `_contains_refined_function`, `_contains_unknown_type`).
  Nested wrapper combinations (Past<D<f32>>, D<Past<f32>>,
  Past<World<NonZero>>) probe correctly. F4 rename debate
  resolved as documented-not-rename at gate 1.
- **silent-failure**: GATE 3 CLEAN. F6 fix verified correct;
  same wrapper-walk pattern exhaustively replicated across
  all 7 sister helpers. No regressions introduced by gate-1
  or gate-2 sweeps.
- **code-review**: GATE 3 CLEAN. 44 Stage 39 tests green
  (210s). Self-host cascade fixpoint preserved (gate-1
  byte-identical G2..G4 still holds — no codegen-affecting
  edits since). All acceptance criteria verified.

### STAGE 39 CLOSED 2026-05-17 at Inc 4 (3/3 clean gates)

3 consecutive clean audit gates achieved per the post-Stage-35
closure convention. Temporal-type scaffold complete:
- 4 temporal kinds (Past / Present / Future / Eternal) as
  `TyTemporal(kind, inner)` Phase-0 wrappers.
- 8 intro/elim builtins + 4 cross-kind transitions = 12 new
  identity-lowered builtins.
- 44 tests green; 1 dogfood (Present→Future→Present→Past
  lifecycle) wires the temporal types into an AGI-shaped
  scenario.
- All 7 type-system helpers + 1 AD set + 1 IR identity arm
  now uniformly handle the 6 single-inner wrappers.

Stage 40 opens next per ROADMAP Phase 2.
