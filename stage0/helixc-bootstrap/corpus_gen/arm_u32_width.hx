// DDC-broad probe: AST_INTLIT_U32 (tag 36) width arm, dynamically executed.
// A u32 value wraps past 2^31 at runtime; the wrapped bits are read back as i32.
fn main() -> i32 {
    let mut x: u32 = 0_u32;
    let mut i: i32 = 0;
    while i < 6 { x = x + 7_u32; i = i + 1; } // x = 42 at runtime (u32 add)
    x as i32                                    // 42
}
