fn main() -> i32 {
    let x = 3;
    match x {
        n if n > 5 => 1,
        _ => 0
    }
}
