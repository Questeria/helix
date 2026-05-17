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

## Increment 1 - Frame Constructors + Eliminators (planned)

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

## Increment 3+ — Planned Sequence

- **Inc 3**: Dogfood — `dogfood_11_spatial_frames.hx` showing a
  point that flows WorldFrame → RobotFrame → CameraFrame and back.
- **Inc 4-6**: Closure audit gate sequence (3-clean-gate).

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
