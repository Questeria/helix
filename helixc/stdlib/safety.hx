// safety.hx — Tier-S/A wrapper helpers (Stages 68-77)
//
// Stage 78: pure-Helix stdlib that demonstrates and exercises the
// new compile-time wrapper types shipped in Stages 68-76. These
// helpers serve two purposes:
//   1. Validate that the wrappers work in non-test, production-
//      shaped code (real stdlib parsed alongside user programs).
//   2. Provide ergonomic wrapper-construction shorthands so users
//      don't have to reach for `__wrap_*` builtins directly.
//
// The wrappers are identity-erased at IR / codegen, so these helpers
// have zero runtime overhead — they exist purely as compile-time
// metadata channels.

// ============================================================
// Confidence (Conf<T>) — high/med/low/precise tiers.
// ============================================================

@pure
fn as_conf(x: f32) -> Conf<f32> {
    __wrap_conf(x)
}

@pure
fn strip_conf_f32(x: Conf<f32>) -> f32 {
    __lift_conf(x)
}

// ============================================================
// Information flow (Taint) — public/internal/confidential/secret.
// ============================================================

@pure
fn classify_f32(x: f32) -> Confidential<f32> {
    __wrap_taint(x)
}

@pure
fn declassify_f32(x: Confidential<f32>) -> f32 {
    __declassify(x)
}

// ============================================================
// Differential privacy (DP) — epsilon-budgeted values.
// ============================================================

@pure
fn as_private_f32(x: f32) -> Private<f32> {
    __wrap_dp(x)
}

@pure
fn exhaust_private_f32(x: Private<f32>) -> f32 {
    __exhaust_dp(x)
}

// ============================================================
// Quantization (Q4/Q8/Q16) — bit-width-tagged values.
// ============================================================

@pure
fn quantize_f32(x: f32) -> Q8<f32> {
    __wrap_quant(x)
}

@pure
fn dequantize_f32(x: Q8<f32>) -> f32 {
    __upcast_quant(x)
}

// ============================================================
// Out-of-distribution (Domain) — in/out/unknown distribution.
// ============================================================

@pure
fn tag_in_dist_f32(x: f32) -> InDist<f32> {
    __wrap_domain(x)
}

@pure
fn assert_in_dist_f32(x: InDist<f32>) -> f32 {
    __assert_in_dist(x)
}

// ============================================================
// Adversarial robustness (Robust) — perturbation-budget-tagged.
// ============================================================

@pure
fn assert_robust_f32(x: f32) -> Robust<f32> {
    __wrap_robust(x)
}

@pure
fn widen_robust_f32(x: Robust<f32>) -> f32 {
    __widen_robustness(x)
}

// ============================================================
// Energy budget (Energy) — joules-spent-tagged.
// ============================================================

@pure
fn measure_energy_f32(x: f32) -> Energy<f32> {
    __wrap_energy(x)
}

@pure
fn exhaust_energy_f32(x: Energy<f32>) -> f32 {
    __exhaust_energy(x)
}

// ============================================================
// Property-based test (Stage 77 @property scaffolding).
// ============================================================

@property
@pure
fn safety_conf_roundtrip_is_identity(x: f32) -> bool {
    // Round-trip through Conf must give the original value, since
    // the wrapper is identity-erased at runtime.
    let wrapped = as_conf(x);
    let unwrapped = strip_conf_f32(wrapped);
    unwrapped == x
}

@property
@pure
fn safety_taint_roundtrip_is_identity(x: f32) -> bool {
    let wrapped = classify_f32(x);
    let unwrapped = declassify_f32(wrapped);
    unwrapped == x
}
