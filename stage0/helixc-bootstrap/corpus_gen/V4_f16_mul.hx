// v1.3 f16 GAP FIX (charter §1 V4): f16 SAME-TYPE MULTIPLICATION computes
// correctly via F16C AND with ROUND-TO-NEAREST-EVEN -- a SHARP discriminator
// that distinguishes the F16C path from BOTH (a) the old silent-wrong bf16/
// integer path (~0) AND (b) a hypothetical truncating narrow.
//
// 7.0_f16 * 293.0_f16: 7 and 293 are EXACT in f16 (integers <= 2048). The exact
// f32 product is 2051.0. 2051 is NOT representable in f16: at this magnitude
// ([2048, 4096)) the f16 step is 4, so the two neighbours are 2048 and 2052.
// 2051 sits 3/4-ULP above 2048, so it is NEARER to 2052 -> RNE (vcvtps2ph
// imm8=0) rounds UP to 2052. A TRUNCATING narrow would floor to 2048; the old
// silent-wrong path lands on ~0. So the result is 2052 ONLY IF the F16C path is
// used AND it rounds to-nearest-even -- not coincidentally right. Comparing
// `c as i32` to 2052 (a full i32 compare) proves both facts bit-exactly:
// exit 42 iff exactly 2052, else 0 (trunc -> 2048 -> 0; silent-wrong -> ~0 -> 0).
fn main() -> i32 {
    let a: f16 = 7.0_f16;
    let b: f16 = 293.0_f16;
    let c: f16 = a * b;                       // f32 product 2051 -> RNE f16 -> 2052 (trunc -> 2048; silent -> ~0)
    if (c as i32) == 2052 { 42 } else { 0 }   // 42 iff F16C + RNE gives 2052
}
