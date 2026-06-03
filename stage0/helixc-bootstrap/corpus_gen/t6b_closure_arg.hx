fn apply(f: i32, x: i32) -> i32 { x + 1 }
fn main() -> i32 {
    let g = |y: i32| y + 1;
    let r = g(20) + g(20);
    r + 0
}
