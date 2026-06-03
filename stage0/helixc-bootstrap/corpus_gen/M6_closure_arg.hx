// T3 §1.6 M-6 (2026-06-03): pass a (non-capturing) CLOSURE AS AN ARGUMENT
// to a higher-order fn that INVOKES it. Pre-fix the closure literal in arg
// position lowered to AST_INT(0) (the fn pointer was lost), so the callee
// did `call *0` -> SIGSEGV(139). The fix returns AST_VAR(__closure_<id>)
// for a no-capture closure -> codegen A2a emits `lea rax,[rip+__closure]`
// (a real fn pointer) -> the callee indirect-calls it (A2b `call r11`).
//
// Exercises:
//   apply1(f, x)      -> f(x)            single closure arg, one call
//   twice(f, x)       -> f(f(x))         the SAME closure invoked twice
//   apply2(g, a, b)   -> g(a) + g(b)     closure arg with two invocations
//   choose(sel,f,h,x) -> picks f or h    TWO distinct closures as args
// All inputs are loop-derived so nothing constant-folds.
fn apply1(f: i32, x: i32) -> i32 { f(x) }
fn twice(f: i32, x: i32) -> i32 { f(f(x)) }
fn apply2(g: i32, a: i32, b: i32) -> i32 { g(a) + g(b) }
fn choose(sel: i32, f: i32, h: i32, x: i32) -> i32 {
    if sel == 0 { f(x) } else { h(x) }
}
fn main() -> i32 {
    let mut base = 0;
    let mut i = 0;
    while i < 10 { base = base + 1; i = i + 1; }   // base = 10
    // apply1: (base+1) + 1 = 12  via |y| y+1
    let r1 = apply1(|y| y + 1, base + 1);          // 12
    // twice: ((base-2)*2)*2 ... use +3 increments: |y| y+3 applied twice to 0 = 6
    let r2 = twice(|y| y + 3, 0);                  // 6
    // apply2: double(4) + double(3) = 8 + 6 = 14 via |y| y*2
    let r3 = apply2(|y| y * 2, 4, 3);              // 14
    // choose: sel=0 picks f=|y| y+10 -> 0+10 = 10
    let r4 = choose(0, |y| y + 10, |y| y - 100, 0);// 10
    // choose: sel=1 picks h=|y| y-? -> use |y| y to pass through 0
    let r5 = choose(1, |y| y + 999, |y| y, 0);     //  0
    // sum = 12 + 6 + 14 + 10 + 0 = 42
    r1 + r2 + r3 + r4 + r5
}
