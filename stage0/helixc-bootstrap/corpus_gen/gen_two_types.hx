fn id[T](x: T) -> T { x }
fn main() -> i32 {
    let a = id::<i32>(40);
    let b: f32 = id::<f32>(2.0_f32);
    a + (b as i32)
}
