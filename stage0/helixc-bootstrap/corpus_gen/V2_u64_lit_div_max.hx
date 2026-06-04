// v1.3 V2 (charter §1 V2): full-range u64 LITERAL through an unsigned DIVIDE --
// a second, divide-path discriminator (independent of the compare path in
// V2_u64_lit_near_max) that also exposes any sign/truncation defect.
//
// 18446744073709551615_u64 = 2^64-1 (all bits set; -1 if misread as signed).
//  9223372036854775807_u64 = 2^63-1 = i64::MAX.
// (2^64-1) / (2^63-1):
//   * UNSIGNED (correct):  18446744073709551615 / 9223372036854775807 = 2
//     (xor rdx,rdx; div rcx -- the u64 path)
//   * SIGNED   (the bug):  -1 / 9223372036854775807 = 0  (cqo; idiv)
// Exit = 2 only when both literals decoded full-width AND the divide is
// unsigned; a high-half truncation of the dividend (-> 0xFFFFFFFF) would give
// 0xFFFFFFFF / 9223372036854775807 = 0. So 2 is a tight unsigned-correct witness.
fn main() -> i32 {
    let big: u64 = 18446744073709551615_u64;   // 2^64-1
    let imax: u64 = 9223372036854775807_u64;    // 2^63-1
    (big / imax) as i32                          // unsigned -> 2 ; signed/trunc-bug -> 0
}
