// v1.3 V4 (charter §1 V4): bf16 CONVERT ROUND-TRIP f32 -> bf16 -> f32 within
// bf16 precision. The f32 literal 1.1 is NOT exactly representable in bf16; the
// f32->bf16 conversion (round-to-nearest-even of the top 16 bits) rounds it to
// 1.1015625 (the nearest bf16; truncation would give 1.09375). Storing it as a
// bf16 then reading it back as f32 (bf16->f32 is the identity -- bf16 is stored
// as f32-valid top-16 bits) must yield EXACTLY 1.1015625. We compare the
// round-tripped value to that known bf16-rounded f32 reference: equal -> 42,
// else 0 (fail-closed; a truncating conversion stores 1.09375 and fails).
fn main() -> i32 {
    let x: bf16 = 1.1_bf16;            // f32 1.1 -> RNE bf16 -> 1.1015625
    let back: f32 = x as f32;          // bf16 -> f32 (identity) = 1.1015625
    let ref32: f32 = 1.1015625_f32;    // the known bf16-rounded reference
    if back == ref32 { 42 } else { 0 } // 42 iff the round-trip is bit-exact RNE
}
