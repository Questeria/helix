// T3 §1.6 struct-return-by-value (2026-06-03): a 5-field (40-byte) struct
// returned BY VALUE, all five fields read. Stresses the multi-slot copy
// well past the 16-byte boundary; each field distinct (2,4,8,16,12) so any
// slot drop/overlap shows in the sum. Runtime-derived first field so the
// constructor cannot be folded to a constant pointer at compile time.
struct Five { a: i32, b: i32, c: i32, d: i32, e: i32 }
fn mk(a: i32) -> Five { Five { a: a, b: 4, c: 8, d: 16, e: 12 } }
fn main() -> i32 {
    let mut a = 0;
    let mut i = 0;
    while i < 2 { a = a + 1; i = i + 1; }          // a = 2
    let f = mk(a);
    f.a + f.b + f.c + f.d + f.e                     // 2+4+8+16+12 = 42
}
