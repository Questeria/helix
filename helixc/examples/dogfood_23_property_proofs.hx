// dogfood_23_property_proofs.hx
//
// Stage 85 end-to-end demo: exercises the 5 @property fns shipped
// in helixc/stdlib/safety.hx (Stages 78 + 82) with concrete
// input values. This is the first Helix program to run the
// Stage 77 @property scaffolding fns as actual runtime assertions
// (vs. just being registered as property fns at typecheck time
// for a future external runner).
//
// For each property fn, we feed a fixed test set of f32 inputs
// (5 values: -100, -1, 0, 1, 100) and count failures. If every
// property holds for every input, the failure counter stays at 0
// and main returns 42.
//
// This validates end-to-end that:
//   1. safety.hx loads cleanly with the stdlib
//   2. Each @property fn typechecks + lowers + runs
//   3. The round-trip invariants (wrap+unwrap = identity) hold
//      at runtime for representative inputs
//   4. The Stage 77 @property registry + Stage 78 stdlib helpers
//      + Stage 82 new-wrapper helpers all integrate

@pure
fn check_one(b: bool, failed: i32) -> i32 {
    if b { failed } else { failed + 1 }
}

fn main() -> i32 {
    let mut failed: i32 = 0;

    // Conf round-trip across 5 representative inputs.
    failed = check_one(safety_conf_roundtrip_is_identity(-100.0_f32), failed);
    failed = check_one(safety_conf_roundtrip_is_identity(-1.0_f32), failed);
    failed = check_one(safety_conf_roundtrip_is_identity(0.0_f32), failed);
    failed = check_one(safety_conf_roundtrip_is_identity(1.0_f32), failed);
    failed = check_one(safety_conf_roundtrip_is_identity(100.0_f32), failed);

    // Taint round-trip.
    failed = check_one(safety_taint_roundtrip_is_identity(-100.0_f32), failed);
    failed = check_one(safety_taint_roundtrip_is_identity(-1.0_f32), failed);
    failed = check_one(safety_taint_roundtrip_is_identity(0.0_f32), failed);
    failed = check_one(safety_taint_roundtrip_is_identity(1.0_f32), failed);
    failed = check_one(safety_taint_roundtrip_is_identity(100.0_f32), failed);

    // Enclave round-trip.
    failed = check_one(safety_enclave_roundtrip_is_identity(-100.0_f32), failed);
    failed = check_one(safety_enclave_roundtrip_is_identity(-1.0_f32), failed);
    failed = check_one(safety_enclave_roundtrip_is_identity(0.0_f32), failed);
    failed = check_one(safety_enclave_roundtrip_is_identity(1.0_f32), failed);
    failed = check_one(safety_enclave_roundtrip_is_identity(100.0_f32), failed);

    // Counterfactual round-trip.
    failed = check_one(safety_cfact_roundtrip_is_identity(-100.0_f32), failed);
    failed = check_one(safety_cfact_roundtrip_is_identity(-1.0_f32), failed);
    failed = check_one(safety_cfact_roundtrip_is_identity(0.0_f32), failed);
    failed = check_one(safety_cfact_roundtrip_is_identity(1.0_f32), failed);
    failed = check_one(safety_cfact_roundtrip_is_identity(100.0_f32), failed);

    // Deadline round-trip.
    failed = check_one(safety_deadline_roundtrip_is_identity(-100.0_f32), failed);
    failed = check_one(safety_deadline_roundtrip_is_identity(-1.0_f32), failed);
    failed = check_one(safety_deadline_roundtrip_is_identity(0.0_f32), failed);
    failed = check_one(safety_deadline_roundtrip_is_identity(1.0_f32), failed);
    failed = check_one(safety_deadline_roundtrip_is_identity(100.0_f32), failed);

    // Total: 5 properties × 5 inputs = 25 assertions. All must pass.
    if failed == 0 { 42 } else { 99 }
}
