enum Opt[T] { Some(T), None }
fn main() -> i32 {
    let o = Opt::Some(42);
    match o {
        Opt::Some(x) => x,
        Opt::None => 0
    }
}
