struct Box[T] { v: T }
fn main() -> i32 {
    let a = Box::<i32>{ v: 2 };
    let b = Box::<i32>{ v: 3 };
    a.v + b.v
}
