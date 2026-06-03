// T3 §1.6 M-6 (2026-06-03): regression guard for the M-6 fix. The fix
// changes a NON-capturing closure literal's value node from AST_INT(0) to
// AST_VAR(__closure_<id>) (a fn pointer). This MUST NOT disturb the
// CAPTURING-closure by-name call path (a capturing closure still returns
// AST_INT(0); its captured env is injected positionally at the call site).
// Runtime-derived so nothing constant-folds.
fn main() -> i32 {
    let mut n = 0;
    let mut i = 0;
    while i < 6 { n = n + 5; i = i + 1; }   // n = 30 (captured)
    let c = |x| x + n;                       // capturing closure, n = 30
    c(2) + c(10) - 30                         // 32 + 40 - 30 = 42
}
