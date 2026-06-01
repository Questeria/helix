struct Box[T] { v: T }
impl<T> Box<T> { fn get(self) -> T { self.v } }
fn main() -> i32 {
    let a = Box::<f32>{ v: 5.0 };
    let s: f32 = a.get();
    s as i32
}
