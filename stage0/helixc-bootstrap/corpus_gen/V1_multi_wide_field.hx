// v1.3 V1 (P0): MULTI-FIELD struct mixing i64 / f64 / i32, each read at its
// correct offset AND width. This catches an offset/width bug that a single-field
// struct cannot: the i64 field (slot 0, 8-byte), the f64 field (slot 1, 8-byte),
// and an i32 field (slot 2, 4-byte) must each be read with the right load width
// at the right slot. Pre-fix the i64 field truncated (silent) and the f64 field
// SIGILLed; the i32 field was always fine. The fix makes all three correct.
//   m.big  : 6_000_000_000 (> 2^32) / 1e8 = 60   (i64, slot 0, 8-byte read)
//   m.d    : 11.0 * 2.0 = 22.0 -> 22              (f64, slot 1, 8-byte read + SSE)
//   m.small: 40 (i32, slot 2, 4-byte read)
//   result : 60 + 22 + 40 - 80 = 42
// All three values runtime-derived so nothing constant-folds; distinct so a
// dropped/mis-offset/mis-width field changes the exit code.
struct Mix { big: i64, d: f64, small: i32 }
fn main() -> i32 {
    let mut bk: i64 = 0_i64;
    let mut i = 0;
    while i < 6 { bk = bk + 1000000000_i64; i = i + 1; }   // bk = 6_000_000_000 (> 2^32)
    let mut dk: f64 = 0.0_f64;
    let mut j = 0;
    while j < 11 { dk = dk + 1.0_f64; j = j + 1; }         // dk = 11.0
    let mut sk = 0;
    let mut q = 0;
    while q < 8 { sk = sk + 5; q = q + 1; }                // sk = 40
    let m = Mix { big: bk, d: dk, small: sk };
    let g: i64 = 100000000_i64;                            // 1e8
    let big_part: i32 = (m.big / g) as i32;                // 6e9 / 1e8 = 60 (truncated -> 8)
    let d_part: i32 = (m.d * 2.0_f64) as i32;              // 11.0 * 2.0 = 22.0 -> 22
    let small_part: i32 = m.small;                          // 40
    big_part + d_part + small_part - 80                     // 60 + 22 + 40 - 80 = 42
}
