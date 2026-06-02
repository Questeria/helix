trait Shape { fn area(self) -> i32 }
struct A { x: i32 }
struct B { x: i32 }
impl Shape for A { fn area(self) -> i32 { self.x } }
impl Shape for B { fn area(self) -> i32 { self.x * 2 } }
fn main() -> i32 { let a = A { x: 14 }; let b = B { x: 14 }; a.area() + b.area() }
