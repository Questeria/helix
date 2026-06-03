// T3 §1.6 struct-return-by-value (2026-06-03): a 1-field struct returned
// BY VALUE from a fn, then its field read in the caller. Pre-fix the
// callee returned a pointer into its own reclaimed frame AND the caller
// stored that pointer with a 32-bit truncating mov -> SIGSEGV (139). The
// aggregate-return fix copies the run into the caller frame (64-bit store
// + post-call copy). Runtime-derived so nothing folds: build h from a
// loop sum so the value is not a compile-time constant.
struct S { h: i32 }
fn mk(n: i32) -> S { S { h: n } }
fn main() -> i32 {
    let mut acc = 0;
    let mut i = 0;
    while i < 6 { acc = acc + 7; i = i + 1; }   // acc = 42 at runtime
    let x = mk(acc);
    x.h
}
