// helixc/stdlib/ieee754.hx — IEEE 754 conversion in Phase-0 Helix.
//
// Phase 1.10 step 3c. Converts a decimal value (integer_part, frac_value,
// frac_digits) into the IEEE 754 f32 bit pattern. Integer-only arithmetic
// — works in any Helix subset that supports i32 + while loops + if-else.
//
// IEEE 754 f32 layout (32 bits, little-endian):
//   bit  31    : sign (0 = positive)
//   bits 30-23 : exponent (8 bits, biased by 127)
//   bits 22- 0 : mantissa (23 bits, implicit leading 1)
//
// For value v > 0: find k such that 2^k <= v < 2^(k+1).
//   exp_field = k + 127
//   mantissa  = floor((v / 2^k - 1) * 2^23)  -- in [0, 2^23 - 1]
//
// All math here is integer:
//   v_scaled = integer_part * 10^d + frac_value
//   v = v_scaled / 10^d
//   pow10 = 10^d
//   threshold(k) = 2^k * pow10
//
// LIMITATIONS:
//   - Only positive values. Caller XORs sign bit externally.
//   - residual * 2 inside the mantissa-extraction loop must fit in i32.
//     For practical literals like 1.5, 3.14, 100.25, we're safe (residual
//     < threshold < pow10 * 2^k < 2^31).
//
// License: Apache 2.0

@pure
fn f32_bits_pow10(d: i32) -> i32 {
    let mut p: i32 = 1;
    let mut i: i32 = 0;
    while i < d { p = p * 10; i = i + 1; }
    p
}

// Compute 2^bit (bit in [0..30]).
@pure
fn f32_bits_pow2(bit: i32) -> i32 {
    let mut v: i32 = 1;
    let mut i: i32 = 0;
    while i < bit { v = v * 2; i = i + 1; }
    v
}

// Main conversion function.
// Returns the unsigned 32-bit IEEE 754 f32 bit pattern as an i32
// (top bit = 0; caller adds sign separately).
@pure
fn f32_bits_pos(integer_part: i32, frac_value: i32, frac_digits: i32) -> i32 {
    let pow10 = f32_bits_pow10(frac_digits);
    let v_scaled = integer_part * pow10 + frac_value;
    if v_scaled == 0 {
        0
    } else {
        // Find binary exponent k: largest k such that 2^k * pow10 <= v_scaled.
        let mut k: i32 = 0;
        let mut threshold: i32 = pow10;
        let mut keep: i32 = 1;
        while keep == 1 {
            // Avoid threshold*2 overflow: check if threshold > v_scaled / 2.
            if threshold > v_scaled / 2 { keep = 0; }
            else {
                threshold = threshold * 2;
                k = k + 1;
            }
        }
        // 2^k * pow10 = threshold; threshold <= v_scaled < 2 * threshold.
        //
        // Extract mantissa bits one by one:
        //   residual starts at v_scaled - threshold (in [0, threshold))
        //   Each iteration doubles residual; if residual >= threshold,
        //   set the corresponding mantissa bit and subtract threshold.
        let mut residual = v_scaled - threshold;
        let mut mantissa: i32 = 0;
        let mut bit: i32 = 22;
        while bit >= 0 {
            residual = residual * 2;
            if residual >= threshold {
                mantissa = mantissa + f32_bits_pow2(bit);
                residual = residual - threshold;
            }
            bit = bit - 1;
        }
        // Pack: exp_field << 23 + mantissa.
        let exp_field = k + 127;
        exp_field * f32_bits_pow2(23) + mantissa
    }
}
