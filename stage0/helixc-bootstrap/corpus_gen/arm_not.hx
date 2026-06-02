// DDC-broad probe: AST_NOT (tag 27) logical-NOT, dynamically executed.
// !cond on a runtime-computed boolean. Both branches reachable; here cond is
// runtime-false so !cond is true and the `then` arm runs.
fn main() -> i32 {
    let mut x = 0;
    let mut i = 0;
    while i < 3 { x = x + 1; i = i + 1; }     // x = 3 at runtime
    let cond = x > 100;                        // runtime false
    if !cond { 42 } else { 7 }                 // AST_NOT -> 42
}
