enum Opt { Some(i32), None }
fn main() -> i32 {
    let o = Opt::Some(42);
    match o {
        Opt::Some(x) => x,
        Opt::None => 0
    }
}
