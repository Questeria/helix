fn apply(f: F, x: i32) -> i32 { f(x) }
fn main() -> i32 {
    apply(|y| y + 1, 41)
}
