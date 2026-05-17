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

## Increment 2+ — Planned Sequence

- **Inc 2**: Cross-frame transform builtins:
  - `to_robot(w: WorldFrame<T>) -> RobotFrame<T>` (world → robot)
  - `to_world(r: RobotFrame<T>) -> WorldFrame<T>` (robot → world)
  - `to_camera(r: RobotFrame<T>) -> CameraFrame<T>` (robot → camera)
  - `to_robot_from_camera(c: CameraFrame<T>) -> RobotFrame<T>`
  - All lower as identity (Phase-0: actual transformation math is
    Phase-1+; the wrapper only tracks intent)
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
