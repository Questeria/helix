// v1.3 V2 (charter §1 V2): a u64 LITERAL near 2^64-1 -- the FULL unsigned range,
// proving NO sign bug and NO truncation in the wide-literal decode.
//
// 18446744073709551615_u64 = 2^64-1 (all 64 bits set, 20 digits). As an UNSIGNED
// value it is the maximum u64. As a SIGNED i64 the identical bit pattern is -1.
// 9223372036854775807_u64 = 2^63-1 = i64::MAX.
//
// The comparison `(2^64-1) > (2^63-1)` is the discriminator:
//   * UNSIGNED (correct):  2^64-1 > 2^63-1  -> TRUE   (seta/setb path via u64)
//   * SIGNED   (the bug):  -1     > 2^63-1  -> FALSE  (setg path)
// So a sign/truncation defect in the literal decode (or in the compare dispatch)
// flips the result. We compute exit = 42 ONLY when the unsigned compare holds,
// else 0. This also confirms the full 64-bit literal materialized (a high-half
// truncation would leave `big` = 0xFFFFFFFF, far below 2^63-1, also giving 0).
fn main() -> i32 {
    let big: u64 = 18446744073709551615_u64;   // 2^64-1 (full unsigned range)
    let imax: u64 = 9223372036854775807_u64;    // 2^63-1 = i64::MAX
    if big > imax { 42 } else { 0 }             // unsigned -> 42 ; signed-bug -> 0
}
