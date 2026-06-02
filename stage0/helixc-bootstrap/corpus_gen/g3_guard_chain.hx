fn main() -> i32 {
    let x = 7;
    match x {
        n if n > 100 => 1,
        n if n > 5 => 2,
        _ => 0
    }
}
