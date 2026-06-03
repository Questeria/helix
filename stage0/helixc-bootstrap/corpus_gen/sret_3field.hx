// T3 §1.6 struct-return-by-value (2026-06-03): a 3-field (24-byte, > 16-byte)
// struct returned BY VALUE -- the size class the real SysV ABI returns via a
// hidden sret pointer. kovc returns it as a pointer-to-slot-run; the fix
// copies all 3 slots into the caller frame and reads each field. Distinct
// field values so a dropped/misaligned slot would change the sum. Runtime-
// derived base so nothing constant-folds.
struct Trip { a: i32, b: i32, c: i32 }
fn mk(base: i32) -> Trip { Trip { a: base, b: base + 10, c: base + 22 } }
fn main() -> i32 {
    let mut base = 0;
    let mut i = 0;
    while i < 5 { base = base + 2; i = i + 1; }   // base = 10
    let t = mk(base);
    // a=10, b=20, c=32 -> but sum=62; subtract 20 to land on 42 so the
    // exit-code byte holds it AND all three fields must be correct.
    t.a + t.b + t.c - 20                            // 10+20+32-20 = 42
}
