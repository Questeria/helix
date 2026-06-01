fn add2[T](a: T, b: T) -> T { a + b }
fn main() -> i32 {
    let r: f32 = add2::<f32>(2.0_f32, 3.0_f32);
    r as i32
}
