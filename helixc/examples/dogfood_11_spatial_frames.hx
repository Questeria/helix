// dogfood_11_spatial_frames.hx — Stage 38 Increment 3 dogfood.
//
// Spatial-frame lifecycle reasoner: a 3D point's identity payload
// flows through every reference frame via cross-frame transforms,
// returning to the original frame intact. First dogfood that
// exercises the Stage 38 Inc 1 + Inc 2 spatial-typing primitives in
// an AGI-shaped scenario.
//
// What this dogfood demonstrates:
//   1. A coordinate enters via into_world (the origin frame —
//      where map/GPS coordinates live).
//   2. world_to_robot — transform from world frame to robot-local
//      (e.g., the robot's odometry just told us where it is on
//      the map).
//   3. robot_to_camera — transform from robot-local to camera-local
//      (the camera is mounted at a known offset on the robot body).
//   4. camera_to_world — direct camera-to-world transform (skips
//      the robot intermediate — the dogfood exercises this direction
//      to confirm symmetric pairwise basis).
//   5. from_world unwraps the result.
//
// Real-world parallels:
//   - SLAM pipelines compose world→robot→camera transforms at every
//     frame to project map landmarks into the current camera view.
//   - AR overlays go camera→world to anchor virtual objects to
//     real-world positions.
//
// Exit code 42 iff THREE independent observations cycle through all
// three frames correctly. Witness is collapse-resistant: each
// observation must round-trip exactly, AND the chain must be
// type-correct end-to-end (any wrong-frame transform call would have
// failed at typecheck before this binary was even produced).

@pure
fn cycle_through_frames(raw: i32) -> i32 {
    // Step 1: coordinate enters world frame (GPS/map).
    let w: WorldFrame<i32> = into_world(raw);
    // Step 2: transform to robot-local (odometry-aware).
    let r: RobotFrame<i32> = world_to_robot(w);
    // Step 3: transform to camera-local (sensor-aware).
    let c: CameraFrame<i32> = robot_to_camera(r);
    // Step 4: direct camera→world transform (skips robot — the
    // symmetric pairwise basis means this is a single hop, not a
    // composition through robot).
    let w_back: WorldFrame<i32> = camera_to_world(c);
    // Step 5: unwrap back to raw i32 for the witness.
    from_world(w_back)
}

fn main() -> i32 {
    let obs1: i32 = cycle_through_frames(10);
    let obs2: i32 = cycle_through_frames(14);
    let obs3: i32 = cycle_through_frames(18);

    // Per-observation binary witnesses:
    let obs1_ok: i32 = if obs1 == 10 { 1 } else { 0 };
    let obs2_ok: i32 = if obs2 == 14 { 1 } else { 0 };
    let obs3_ok: i32 = if obs3 == 18 { 1 } else { 0 };

    // Product of 3 binary witnesses; any single regression collapses
    // to 0 → final exit code 0 not 42.
    let all_ok: i32 = obs1_ok * obs2_ok * obs3_ok;

    // Sum of recalled observations: 10 + 14 + 18 = 42.
    all_ok * (obs1 + obs2 + obs3)
}
