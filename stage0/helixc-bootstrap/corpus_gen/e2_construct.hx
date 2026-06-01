enum Opt[T] { Some(T), None }
fn main() -> i32 {
    let o = Opt::<i32>::Some(42);
    0
}
