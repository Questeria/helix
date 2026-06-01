struct Box[T] { v: T }
fn main() -> i32 {
    let a = Box::<f32>{ v: 2.0_f32 };
    let b = Box::<f32>{ v: 3.0_f32 };
    let s: f32 = a.v + b.v;
    s as i32
}
