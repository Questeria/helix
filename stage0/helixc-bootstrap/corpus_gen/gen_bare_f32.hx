fn id[T](x: T) -> T { x }
fn main() -> i32 {
    let r: f32 = id(3.0_f32);
    r as i32
}
