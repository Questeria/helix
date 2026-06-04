// v1.3 V4 (charter §1 V4): bf16 MULTIPLICATION correct within bf16 precision
// vs an f32 reference. bf16 mul goes convert-op-convert: the two bf16 operands
// (f32-valid top-16 patterns) multiply in f32 (mulss), and the f32 product is
// ROUNDED back to bf16 (round-to-nearest-even).
//
// 17.0_bf16 * 19.0_bf16: 17, 19 are EXACT in bf16. The f32 product is 323.0,
// NOT representable in bf16 (exponent 8 -> step 2). RNE rounds 323 to the EVEN
// neighbour 324 (324/2=162 even; 322/2=161 odd). A truncating path would give
// 322 -- so exit 324 verifies the RNE round-back of the PRODUCT bit-exactly.
fn main() -> i32 {
    let a: bf16 = 17.0_bf16;
    let b: bf16 = 19.0_bf16;
    let c: bf16 = a * b;        // f32 product 323.0 -> RNE bf16 -> 324.0
    c as i32                    // 324 (a truncating path would give 322)
}
