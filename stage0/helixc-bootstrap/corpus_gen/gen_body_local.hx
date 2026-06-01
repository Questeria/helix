fn dbl[T](x: T) -> T {
    let y: T = x + x;
    y
}
fn main() -> i32 {
    let r: f32 = dbl::<f32>(2.5_f32);
    r as i32
}
