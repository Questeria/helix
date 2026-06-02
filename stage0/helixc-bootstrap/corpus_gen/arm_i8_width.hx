// DDC-broad probe: AST_INTLIT_I8 (tag 39) width arm, dynamically executed.
// i8 arithmetic accumulated at runtime, then widened to i32 for the exit code.
fn main() -> i32 {
    let mut x: i8 = 0_i8;
    let mut i: i32 = 0;
    while i < 6 { x = x + 7_i8; i = i + 1; }  // x = 42 at runtime (i8 add)
    x as i32                                    // 42
}
