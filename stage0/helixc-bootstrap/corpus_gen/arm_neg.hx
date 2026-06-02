// DDC-broad probe: AST_NEG (tag 9) dynamically executed.
// Negation is applied to a runtime variable (not a literal), so the `neg eax`
// path runs at runtime and cannot be constant-folded to a literal.
fn main() -> i32 {
    let mut a = 0;
    let mut i = 0;
    while i < 7 { a = a + 10; i = i + 1; }   // a = 70 at runtime
    let n = -a;                               // AST_NEG on a runtime value -> -70
    n + 112                                   // 42
}
