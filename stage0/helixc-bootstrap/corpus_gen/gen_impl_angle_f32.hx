struct Box[T] { v: T }
impl<T> Box<T> { fn get(self) -> T { self.v } }
fn main() -> i32 {
    let a = Box::<f32>{ v: 2.0 };
    let b = Box::<f32>{ v: 3.0 };
    let s: f32 = a.get() + b.get();
    s as i32
}
