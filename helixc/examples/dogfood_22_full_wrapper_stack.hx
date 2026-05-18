// dogfood_22_full_wrapper_stack.hx
//
// Stage 84 end-to-end demo: exercises ALL 11 Tier-S/A wrappers
// shipped in Stages 68-83 in a single compileable Helix program.
// Validates that the full wrapper stack:
//   1. Each constructor (`__wrap_*`) lifts a plain f32 cleanly
//   2. Each opt-out builtin strips its wrapper cleanly
//   3. All wrappers identity-erase at IR / codegen
//   4. Layered wrappers compose in the canonical order
//      (Attribution > Enclave > Counterfactual > Taint > DP > Conf >
//       Domain > Robust > Energy > Deadline > Quant > T)
//
// Test pattern: thread a value (starting at 1.0) through each of
// the 11 wrappers, immediately strip it, and accumulate the
// stripped value into an i32 counter. If every wrap+strip is
// identity-correct, the counter reaches 42.

fn main() -> i32 {
    let v: f32 = 1.0_f32;

    // 1. Conf
    let r1 = __lift_conf(__wrap_conf(v));
    // 2. Taint
    let r2 = __declassify(__wrap_taint(v));
    // 3. DP
    let r3 = __exhaust_dp(__wrap_dp(v));
    // 4. Quant
    let r4 = __upcast_quant(__wrap_quant(v));
    // 5. Domain
    let r5 = __assert_in_dist(__wrap_domain(v));
    // 6. Robust
    let r6 = __widen_robustness(__wrap_robust(v));
    // 7. Energy
    let r7 = __exhaust_energy(__wrap_energy(v));
    // 8. Enclave
    let r8 = __exit_enclave(__wrap_enclave(v));
    // 9. Counterfactual
    let r9 = __as_actual(__wrap_cfact(v));
    // 10. Deadline
    let r10 = __miss_deadline(__wrap_deadline(v));
    // 11. Attribution
    let r11 = __attribute_verified(__wrap_attr(v));

    // Sum all 11 — each is 1.0, so total = 11.0.
    let total = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10 + r11;
    let n = total as i32;  // 11

    // Now do one layered round-trip: Confidential<Conf<f32>> →
    // Conf<f32> → f32. Tests that opt-out preserves inner wrappers.
    let layered: Confidential<Conf<f32>> = __wrap_taint(__wrap_conf(10.0_f32));
    let inner_only: Conf<f32> = __declassify(layered);
    let stripped: f32 = __lift_conf(inner_only);
    let m = stripped as i32;  // 10

    // 11 + 10 + 21 sentinel = 42
    n + m + 21
}
