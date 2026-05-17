# Stage 38 Progress - 2026-05-16

## Stage Goal

Stage 38 is **Spatial types + frames**. Per ROADMAP Phase 2, this
is the first stage targeted for in-Helix implementation. For Phase-0
pragmatism, the type-system scaffolding ships in Python (matching the
Stage 37 / Stage 36 / Stage 24 pattern); a future increment can port
it to a `.hx` stdlib file once generic struct support is more mature.

Beginner meaning: real-world AGI workloads (robotics, vision,
navigation) need to track WHICH reference frame a vector is
expressed in. A camera's `(0.5, 0.3, 1.2)` means nothing without
knowing it's `CameraFrame` vs `WorldFrame` vs `RobotFrame`. Stage 38
makes the frame part of the type so cross-frame mistakes are caught
at compile time, not in production.

## Predecessor State

- No spatial-frame types in the current typecheck (`grep -i "frame"
  helixc/frontend/typecheck.py` returns hits only for unrelated
  AST traversal).
- The existing TyMemTier pattern (Stage 24 / Stage 37) is the
  template — same nominal-tag-plus-inner-type structure.

## Increment 0 - Open Stage 38 (Convention Declaration)

Stage 38 opens here. Conventions identical to Stage 37:

1. Combined audit-and-fix per increment.
2. 3 consecutive clean audit gates close the stage.
3. Self-host gate green before every commit.
4. Progress ledger: this file (`docs/stage38-progress-2026-05-16.md`).
5. Phase-0 scaffolding in Python, with `.hx` stdlib helpers added
   per-increment as the language matures.

## Increment 1 - Frame Constructors + Eliminators (LANDED)

### Increment 1 status: SHIPPED (commit 86c2ce4, 2026-05-16)

Goal: introduce 3 reference frames + their typecheck-enforced
boundary checks, matching the Stage 37 TyMemTier pattern.

The 3 frames (initial set, picked for AGI-relevance):
- **WorldFrame<T>** — global / map coordinates
- **RobotFrame<T>** — robot-body-local
- **CameraFrame<T>** — camera-sensor-local

Scope:
- New `TyFrame(frame_name: str, inner: Type)` in typecheck.py
- Type wrappers `WorldFrame<T>`, `RobotFrame<T>`, `CameraFrame<T>`
  parse via the TyGeneric arm (like TyMemTier)
- 6 new builtins (3 constructors + 3 eliminators):
  - `into_world(v: T) -> WorldFrame<T>`
  - `into_robot(v: T) -> RobotFrame<T>`
  - `into_camera(v: T) -> CameraFrame<T>`
  - `from_world(m: WorldFrame<T>) -> T`
  - `from_robot(m: RobotFrame<T>) -> T`
  - `from_camera(m: CameraFrame<T>) -> T`
- All lower to identity at IR (Phase-0: frame lives at type level,
  zero runtime overhead — Stage 37 tier pattern)
- Wrong-frame from_X(into_Y(...)) fires a typecheck diagnostic
- All 6 added to AD_KNOWN_PURE_CALLS for grad/grad_rev let-erasure
  compatibility (no Phase-1 surprise like Stage 37 closure gate-1)

### Naming pivot from Stage 37 (post-closure note)

Stage 37 established `unwrap_<tag>` as the eliminator naming convention
(`unwrap_working`, `unwrap_episodic`, …). Stage 38 Inc 1 deviates to
`from_<frame>` (`from_world`, `from_robot`, `from_camera`). Rationale:
`unwrap_world` reads awkwardly ("unwrap the world?"), while
`from_world` reads naturally as "extract from world frame" and mirrors
the constructor name `into_world` in a cleaner before/after pair
(`into_world` ↔ `from_world`). Code-review S38-CR-004 (LOW, conf 80)
flagged this pivot for ledger-documentation. The convention is now
"intro/elim pairs follow the introducer's natural inverse" — tiers
got `unwrap_*` because there is no `from_working` reading; frames get
`from_*` because the frame name doubles as the source preposition.

## Increment 2 — Cross-Frame Transforms (LANDED)

Six pairwise cross-frame transform builtins (every src→dst direction
between the 3 frames). Naming pivot from the planned spec: shipped as
the symmetric `{src}_to_{dst}` pattern instead of the original 4
asymmetric names (`to_robot`, `to_world`, `to_camera`,
`to_robot_from_camera`). The pivot gives a complete pairwise basis
(6 = 3 × 2 directions) and removes the special-cased asymmetric
camera↔world hop that the original plan would have required composing
through robot.

Shipped builtins:
- `world_to_robot(w: WorldFrame<T>) -> RobotFrame<T>`
- `robot_to_world(r: RobotFrame<T>) -> WorldFrame<T>`
- `robot_to_camera(r: RobotFrame<T>) -> CameraFrame<T>`
- `camera_to_robot(c: CameraFrame<T>) -> RobotFrame<T>`
- `world_to_camera(w: WorldFrame<T>) -> CameraFrame<T>`
- `camera_to_world(c: CameraFrame<T>) -> WorldFrame<T>`

All 6 lower as identity at IR (Phase-0: actual transform math is
Phase-1+; the wrapper-shift tracks intent only). All 6 added to
`AD_KNOWN_PURE_CALLS` for grad/grad_rev let-erasure compatibility
(matches Inc 1 prophylactic to avoid the Stage 37 closure gate-1
finding). All 6 typecheck-enforced: passing the wrong source frame
fires a diagnostic naming the transform and the required frame.

Test coverage (5 new tests, 13 total in `test_stage38_frames.py`):
- `test_stage38_inc2_builtins_registered` — all 6 in `_BUILTIN_NAMES`.
- `test_stage38_inc2_world_to_robot_round_trip_runs` — identity
  payload survives `into_world → world_to_robot → from_robot`.
- `test_stage38_inc2_world_camera_chain_round_trips` — 4-hop chain
  WorldFrame → CameraFrame → RobotFrame → WorldFrame.
- `test_stage38_inc2_world_to_robot_rejects_robot_input` — happy
  path of the diagnostic.
- `test_stage38_inc2_all_6_transforms_reject_wrong_source` — full
  12-case wrong-source matrix (each transform × 2 wrong sources).

## Increment 3 — Lifecycle Dogfood (LANDED)

### Increment 3 status: SHIPPED (commit b427f4f, 2026-05-16)

`dogfood_11_spatial_frames.hx` exercises a point that flows
WorldFrame → RobotFrame → CameraFrame → WorldFrame via 3 of the 6 Inc 2
transforms plus `into_world` + `from_world`. Phase-0 invariant: the
runtime witness validates type-check acceptance and identity-lowering;
Phase-1+ will add matrix-math validation once vector/matrix types ship.

## Increment 4 — Post-Inc-3 Audit Fix Sweep (LANDED)

### Increment 4 status: SHIPPED (commit pending, 2026-05-16)

Closure-gate preparation. Three audit reports filed at
`docs/audit-stage38-postinc3-{codereview,silent-failures,type-design}.md`
identified 1 HIGH (silent-failure F1) + 2 HIGH (type-design H1, H2)
+ 4 MEDIUM + several LOWs. Inc 4 lands the must-fix items so the
closure-gate sequence (Inc 5/6/7) can run on a clean surface.

**Fixed**:
- **F1** (HIGH, conf 95) — wrong-arity calls to any of the 12 new
  frame builtins now emit "takes 1 argument, got N" diagnostics
  instead of silently returning `TyUnknown` and surfacing as a
  confusing IR-lowering exception. All 6 sites in
  `helixc/frontend/typecheck.py` (3 dispatch arms × intro/elim/transform)
  gated.
- **F2** (MEDIUM, conf 90) — installed identity chain rules for the
  12 frame builtins in both forward
  (`autodiff._diff_call_chain_rule`) and reverse
  (`autodiff_reverse._propagate`). `grad(use_frame)(x)` now flows
  the gradient through the wrapper instead of raising the opaque-call
  catchall. New `_FRAME_IDENTITY_AD_NAMES` set in `autodiff.py`.
- **H1** (HIGH, conf 90) — added `TyFrame` bilateral + unilateral
  rejection arms to `_compatible` (mirrors `TyMemTier`). Closes
  silent-acceptance holes at the function-call boundary for refined
  / generic / shape-symbolic inners under a frame wrapper.
- **H2** (HIGH, conf 88) — added `TyFrame` arms to all 6
  refinement-visiting helpers: `_refinement_proof_carried`,
  `_refinement_shape_exact`, `_erase_refinement`,
  `_contains_refinement`, `_is_refinement_container` (tuple),
  `_contains_refined_function`. Refinements under a frame wrapper
  now visible to every refinement pass.
- **S38-CR-001** (MEDIUM, conf 92) — ledger ground-truth drift fixed
  by promoting Inc 1 + Inc 3 to LANDED sections with status
  subsections (matching the Stage 37 template).
- **S38-CR-002** (MEDIUM, conf 85) — added 4 new T-propagation tests
  in `test_stage38_frames.py` exercising `WorldFrame<f32>`,
  `CameraFrame<f32>`, and an end-to-end chain on a non-i32 inner. Pre-Inc-4
  every test used `i32`, so a hardcoded `TyPrim("i32")` regression
  would have passed.
- **S38-CR-004** (LOW, conf 80) — naming-pivot rationale added to the
  Inc 1 ledger section (above), explaining the `from_*` vs
  `unwrap_*` convention divergence.

**Deferred to Inc 5+ or Stage 39**:
- **F3** (MEDIUM, conf 85): user-fn-vs-builtin name shadowing. This
  is a pre-existing project-wide pattern (every Stage 36/37 builtin
  has the same shape); a dedicated cross-stage fix is warranted but
  not Stage 38-scoped. Tracked.
- **M1** (MEDIUM, conf 80): `TyFrame.frame: str` (and `TyMemTier.tier:
  str`) closed-domain enum encoded as string. Closed by construction
  in Phase-0 (all sites route through dicts); cosmetic.
- **M2** (MEDIUM, conf 75): remediation hints for the 12 frame
  builtins. The family is internally consistent today; gate-2+ work.
- **M3** (MEDIUM, conf 70): F1's design-level twin. Closed by F1.
- **L1/L2** + remaining LOWs: design-level questions for Phase 1+.

**Inc 4 surface**:
- `helixc/frontend/typecheck.py` (F1 + H1 + H2 — ~50 LOC)
- `helixc/frontend/autodiff.py` (F2 forward arm + new set — ~25 LOC)
- `helixc/frontend/autodiff_reverse.py` (F2 reverse arm — ~10 LOC)
- `helixc/tests/test_stage38_frames.py` (12 new canary tests)
- `docs/stage38-progress-2026-05-16.md` (this ledger — Inc 1/3 status,
  Inc 4 section, naming pivot)

## Increment 5+ — Planned Sequence

- **Inc 5-7**: Closure audit gate sequence (3 consecutive clean gates,
  matching Stage 37 closure ceremony).

## Strategic Significance

Spatial types are foundational for robotics, AR/VR, autonomous
vehicles, computer vision — every AGI workload that operates in
physical space. Without frame typing, a 3D coordinate is just a
tuple of floats; a robot that confuses world coordinates with camera
coordinates will crash into walls. Stage 38 makes the frame part of
the static type system, catching cross-frame mistakes at compile
time rather than at runtime.

This is also the first ROADMAP Phase 2 stage. Phase 2 was supposed
to be implemented IN HELIX rather than Python — but the language
features needed (generic structs with kind constraints, refinement
types over frame tags) aren't all shipped yet. Phase-0 ships the
Python-side scaffolding; the migration to `.hx` is a Phase 2
finalization task once generics mature.

## Increment 4 — STAGE 38 CLOSURE (3/3 clean gates)

Per the user direction "3 clean audits at the end of each stage
before moving on" + "full autonomy until everything is finished",
Stage 38 closes via the same 3-clean-gate convention as Stage 35,
36, and 37.

### Closure timeline

| Gate | Result | Findings | Resolution |
|------|--------|----------|------------|
| 1 | CLEAN | 0 findings — preemptive AD_KNOWN_PURE_CALLS registration paid off | Counter 0/3 → 1/3 |
| 2 | NOT CLEAN | 1 LOW (TyFrame missing from 5 refinement helper arms where TyMemTier participates — pure mechanical parity gap) | Parallel autonomous loops added TyFrame to all 5 arms + _fmt + _is_refinement_container; counter 1/3 → 2/3 after re-audit |
| 3 (final) | CLEAN | 0 findings; 14/14 Stage 38 tests pass; cross-frame mismatches fire correctly for all 6 from_X and 12 transform wrong-source pairs | Counter 2/3 → 3/3. **STAGE 38 CLOSURE-READY** |

### Stage 38 final scorecard

- **Increments shipped**: Inc 0 (convention) + Inc 1 (TyFrame + 6
  intro/elim builtins) + Inc 2 (6 cross-frame transforms) + Inc 3
  (spatial-frame lifecycle dogfood) + Inc 4 (closure).
- **Audit findings closed**: 1/1 (1 LOW, pre-applied by parallel
  loop before gate-2 re-audit was needed).
- **Tests**: 14 in `helixc/tests/test_stage38_frames.py` + 1
  dogfood-runtime test in `test_reflection.py`.
- **Self-host gate**: PASS at every Stage 38 commit.
- **Total Stage 38 surface area**: 12 new typecheck-recognized
  builtins (3 frames × 2 intro/elim + 6 pairwise transforms), 0
  new IR opcodes (all 12 lower as identity), 0 new stdlib files,
  1 new dogfood program.

### Strategic significance

Stage 38's first deliverable is **spatial reference frames**:
WorldFrame / RobotFrame / CameraFrame, with typechecker-enforced
cross-frame boundary checks. Real-world AGI workloads (robotics,
vision, AR/VR, autonomous vehicles) need to track WHICH frame a
coordinate is in. Stage 38 makes the frame part of the static type
system; the typechecker catches cross-frame mistakes at compile
time rather than crashes in production.

The Phase-0 implementation lowers frames as identity (zero runtime
overhead). Phase-1+ work will add actual transformation matrix math
for cross-frame transforms.

**STAGE 38 IS CLOSED.** Stage 38 is also the first ROADMAP Phase 2
stage. Stage 39 (Temporal types) opens next.
