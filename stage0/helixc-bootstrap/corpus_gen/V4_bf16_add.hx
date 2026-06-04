// v1.3 V4 (charter §1 V4): bf16 ADDITION computes correctly within bf16
// precision vs an f32 reference. This SHIPS the v1.2 bf16/f16 "storage-only"
// bound (arith TRAPPED SIGILL 2001); bf16 add now goes convert-op-convert:
// the two bf16 operands (stored as f32-valid top-16 patterns) add in f32
// (addss), and the f32 sum is ROUNDED back to bf16 with round-to-nearest-even.
//
// 256.0_bf16 + 3.0_bf16: 256, 3 are EXACT in bf16. The f32 sum is 259.0, which
// is NOT representable in bf16 (at exponent 8 the bf16 step is 2). RNE rounds
// 259 to the EVEN neighbour 260 (260/2=130 even; 258/2=129 odd). A truncating
// (non-rounding) path would give 258. Comparing `c as i32` to 260 (a full i32
// compare -- no exit-byte wrap) verifies the RNE round-back BIT-EXACTLY, not
// merely "no crash": exit 42 iff the rounded sum is exactly 260, else 0 (a
// truncating path lands on 258 and fails closed to 0). The sentinel return
// avoids the 8-bit exit-code wrap that returning 260 directly would suffer
// (260 & 0xFF = 4).
fn main() -> i32 {
    let a: bf16 = 256.0_bf16;
    let b: bf16 = 3.0_bf16;
    let c: bf16 = a + b;                     // f32 sum 259.0 -> RNE bf16 -> 260.0
    if (c as i32) == 260 { 42 } else { 0 }   // 42 iff RNE 260 (trunc path -> 258 -> 0)
}
