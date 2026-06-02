// DDC-broad probe: AST_MOD (tag 24) integer modulo, dynamically executed.
// Operands are runtime values so the idiv/cdq remainder path actually runs.
fn main() -> i32 {
    let mut a = 0;
    let mut i = 0;
    while i < 10 { a = a + 25; i = i + 1; }   // a = 250 at runtime
    let mut b = 0;
    let mut j = 0;
    while j < 4 { b = b + 13; j = j + 1; }    // b = 52 at runtime
    a % b                                      // 250 % 52 = 42
}
