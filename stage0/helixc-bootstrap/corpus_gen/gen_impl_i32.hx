struct Box[T] { v: T }
impl[T] Box[T] {
    fn get(self) -> T { self.v }
}
fn main() -> i32 {
    let a = Box::<i32>{ v: 2 };
    let b = Box::<i32>{ v: 3 };
    a.get() + b.get()
}
