struct Pair(i32, i32);
fn main() -> i32 {
    // tuple struct: a struct with positional (unnamed) fields, accessed .0 / .1
    let p: Pair = Pair(40, 2);
    p.0 + p.1   // 42
}
