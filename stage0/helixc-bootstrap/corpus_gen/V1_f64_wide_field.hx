// v1.3 V1 (P0): f64 WIDE STRUCT FIELD read + arithmetic correct. A struct field
// holding an f64 must read back full-width AND be f64-typed so f64 arithmetic
// works. Pre-fix the field READ was 4-byte + i32-typed, so `a.v + 0.0_f64` hit
// the mixed-type guard -> ud2/SIGILL (exit 132, fail-closed). The fix reads the
// full 8 bytes (the IEEE-754 bit pattern lands in rax) and types the result f64
// (tag 2), so the add routes through the SSE f64 path. Reference: the same value
// held in a plain f64 LOCAL; the field-read result must equal it. Field value is
// runtime-derived (loop) so it is not folded.
struct FBig { v: f64 }
fn main() -> i32 {
    let mut x: f64 = 0.0_f64;
    let mut i = 0;
    while i < 3 { x = x + 7.0_f64; i = i + 1; }   // x = 21.0 at runtime
    let b = FBig { v: x };
    let ref_local: f64 = x;                       // independent f64 reference (a plain local)
    let from_field: f64 = b.v * 2.0_f64;          // 8-byte f64 field read + f64 arith -> 42.0
    let r: f64 = ref_local * 2.0_f64;             // 42.0 from the reference
    // Both must equal 42.0; emit (from_field) only if it matches the reference,
    // else 0 (so a wrong/truncated f64 field read fails the gate, not silently passes).
    let ok: i32 = if (from_field - r) < 0.5_f64 { if (r - from_field) < 0.5_f64 { 1 } else { 0 } } else { 0 };
    if ok == 1 { from_field as i32 } else { 0 }   // 42 iff the f64 field read matches the reference
}
