// T3 §1.6 struct-return-by-value (2026-06-03): a 2-field (16-byte) struct
// returned BY VALUE, then BOTH fields read + summed in the caller. This is
// the <=16-byte size class the SysV ABI returns in rax:rdx; kovc represents
// it as a pointer-to-slot-run, so the fix copies the run into the caller
// frame. Runtime-derived: a=10, b=32 are passed through a fn so they are
// not folded; exercises slot 0 ([rax+0]) AND slot 1 ([rax+8]) reads.
struct Pair { a: i32, b: i32 }
fn mk(a: i32, b: i32) -> Pair { Pair { a: a, b: b } }
fn main() -> i32 {
    let mut a = 0;
    let mut i = 0;
    while i < 10 { a = a + 1; i = i + 1; }       // a = 10
    let p = mk(a, 32);
    p.a + p.b                                     // 10 + 32 = 42
}
