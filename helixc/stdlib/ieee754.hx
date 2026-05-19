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

// Cycle 3 R1 fix batch 20 (RT HIGH-1):
// Pre-fix: f32_bits_pow10(d) for d>=10 silently wrapped i32 (10^10 ~= 1e10
// > INT32_MAX ~= 2.15e9). f32_bits_pow2(bit) for bit>=31 silently produced
// INT32_MIN (the sign bit). Callers received corrupted bit patterns with no
// out-of-band signal.
//
// Post-fix: both functions return INT32_MIN (0 - 2147483648 - 1 via two's
// complement = -2147483648) for out-of-range inputs. Existing callers that
// use the safe range (d in [0..9], bit in [0..30]) get identical behavior;
// new code can check `if result == INT32_MIN { /* range overflow */ }`.
// Original comment said "bit in [0..30]" but no enforcement — now enforced.
@pure
fn f32_bits_pow10(d: i32) -> i32 {
    // 10^10 = 10000000000 > INT32_MAX (2147483647). Clamp at d=9.
    if d < 0 { 0 - 2147483647 - 1 }
    else if d > 9 { 0 - 2147483647 - 1 }
    else {
        let mut p: i32 = 1;
        let mut i: i32 = 0;
        while i < d { p = p * 10; i = i + 1; }
        p
    }
}

// Compute 2^bit (bit in [0..30]).
@pure
fn f32_bits_pow2(bit: i32) -> i32 {
    // 2^31 = INT32_MIN (sign bit); 2^30 = 1073741824 fits. Clamp at 30.
    if bit < 0 { 0 - 2147483647 - 1 }
    else if bit > 30 { 0 - 2147483647 - 1 }
    else {
        let mut v: i32 = 1;
        let mut i: i32 = 0;
        while i < bit { v = v * 2; i = i + 1; }
        v
    }
}

// Main conversion function.
// Returns the unsigned 32-bit IEEE 754 f32 bit pattern as an i32
// (top bit = 0; caller adds sign separately).
//
// Cycle 3 R1 fix batch 20 (RT HIGH-2):
// integer_part * pow10 + frac_value silently wraps i32 for large literals.
// Post-fix: detect upstream f32_bits_pow10 sentinel (INT32_MIN) and
// short-circuit. f32_bits_pos returns INT32_MIN on out-of-range frac_digits.
// Multiplicative overflow on integer_part * pow10 also detected.
@pure
fn f32_bits_pos(integer_part: i32, frac_value: i32, frac_digits: i32) -> i32 {
    let pow10 = f32_bits_pow10(frac_digits);
    // Detect upstream sentinel: pow10 returned INT32_MIN.
    if pow10 == 0 - 2147483647 - 1 { 0 - 2147483647 - 1 }
    else if pow10 > 0 && integer_part != 0 && (integer_part > 2147483647 / pow10 || integer_part < (0 - 2147483647 - 1) / pow10) {
        // integer_part * pow10 would overflow i32. Return sentinel.
        0 - 2147483647 - 1
    } else {
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
        // Note: f32_bits_pow2(23) = 8388608. exp_field * 8388608 fits in i32
        // for exp_field in [0..255] (max 255*8388608 = 2138083328 < INT32_MAX).
        let exp_field = k + 127;
        exp_field * f32_bits_pow2(23) + mantissa
    }
    }
}

// f32 +0.0 bit pattern. Pure constant for callers that need a typed
// zero seed without paying for the f32_bits_pos(0,0,0) loop guard.
@pure
fn f32_bits_zero() -> i32 {
    0
}

// f32 1.0 bit pattern (0x3F800000 = 1065353216). Matches the IEEE 754
// canonical representation: sign=0, exp=127 (biased), mantissa=0.
// Used as the multiplicative identity seed for autodiff / scaling work.
@pure
fn f32_bits_one() -> i32 {
    1065353216
}

// Negative-valued IEEE 754 f32 bit pattern. Companion to f32_bits_pos:
// computes the positive bit pattern, then XORs the sign bit (1 << 31).
// For the (0, 0, 0) input this yields the IEEE 754 -0.0 bit pattern
// (0x80000000), which is bit-distinct from +0.0 but compares equal
// numerically — semantically the right answer for IEEE 754.
@pure
fn f32_bits_neg(integer_part: i32, frac_value: i32, frac_digits: i32) -> i32 {
    let pos = f32_bits_pos(integer_part, frac_value, frac_digits);
    pos ^ (1 << 31)
}
