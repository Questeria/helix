// DDC-broad probe: AST_BNOT (tag 26) bitwise-NOT, dynamically executed on a
// runtime value. ~x == -x-1 (two's complement). x=85 (runtime) -> ~85 = -86.
fn main() -> i32 {
    let mut x = 0;
    let mut i = 0;
    while i < 5 { x = x + 17; i = i + 1; }   // x = 85 at runtime
    let b = ~x;                               // AST_BNOT -> -86
    b + 128                                   // 42
}
