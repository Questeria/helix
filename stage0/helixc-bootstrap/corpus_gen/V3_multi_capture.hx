// v1.3 V3 (2026-06-04): a closure capturing MULTIPLE locals, passed BY VALUE as
// an argument + invoked, reads ALL of its captures correctly. Exercises the
// closure object's multi-cell env [code_ptr, cap_a, cap_b, cap_c] and the
// env-based capture reads __arena_get(__cenv + 1 + k) for k = 0,1,2.
//
// Capture semantics: CAPTURE-BY-VALUE AT CLOSURE-CREATION (3 captures).
//
// a=10, b=20, c=5 (all loop-derived); f = |y| a + b + c + y;
// apply(f, 7) -> f(7) -> 10 + 20 + 5 + 7 = 42.
fn apply(f: i32, v: i32) -> i32 { f(v) }
fn main() -> i32 {
    let mut a = 0;
    let mut b = 0;
    let mut c = 0;
    let mut i = 0;
    while i < 5 { a = a + 2; b = b + 4; c = c + 1; i = i + 1; }  // a=10 b=20 c=5
    let f = |y| a + b + c + y;        // captures 3 locals by value
    apply(f, 7)                        // 10 + 20 + 5 + 7 = 42
}
