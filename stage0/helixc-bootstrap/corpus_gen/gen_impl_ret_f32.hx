struct Box[T] { v: T }
impl<T> Box<T> { fn get(self) -> f32 { self.v } }
fn main() -> i32 {
    let a = Box::<f32>{ v: 5.0 };
    let s: f32 = a.get();
    s as i32
}
