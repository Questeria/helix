fn quad[T](x: T) -> T {
    let y: T = x + x;
    y + y
}
fn main() -> i32 {
    let r: f32 = quad::<f32>(1.0_f32);
    r as i32
}
