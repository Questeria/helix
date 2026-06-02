struct Box[T] { v: T }
impl<T> Box<T> { fn area(self) -> i32 { 7 } }
fn main() -> i32 {
    let a = Box::<f32>{ v: 5.0 };
    a.area()
}
