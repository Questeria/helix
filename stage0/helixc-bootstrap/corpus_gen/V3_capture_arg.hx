// v1.3 V3 (2026-06-04): a CAPTURING closure passed BY VALUE as an argument to
// a higher-order fn that INVOKES it -- the v1.2 M-6 residual (capturing closure
// passed by value trapped SIGSEGV). The fix: a capturing closure compiles to a
// real closure OBJECT (arena env {code_ptr, captured-values}); its value is a
// tagged env-index that survives a by-value i32 param; the callee's indirect
// dispatch untags it, loads the code ptr from the object, and passes the env so
// the body reads its captures from the object.
//
// Capture semantics: CAPTURE-BY-VALUE AT CLOSURE-CREATION.
//
// x = 40 (captured); c = |y| x + y; apply(c, 2) -> c(2) -> 40 + 2 = 42.
// All inputs are loop-derived so nothing constant-folds.
fn apply(f: i32, v: i32) -> i32 { f(v) }
fn main() -> i32 {
    let mut x = 0;
    let mut i = 0;
    while i < 8 { x = x + 5; i = i + 1; }   // x = 40 (captured by value)
    let c = |y| x + y;                        // capturing closure object
    apply(c, 2)                               // 40 + 2 = 42
}
