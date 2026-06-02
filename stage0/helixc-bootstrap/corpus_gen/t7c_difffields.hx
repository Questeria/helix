trait Shape { fn area(self) -> i32 }
struct A { p: i32 }
struct B { q: i32 }
impl Shape for A { fn area(self) -> i32 { self.p } }
impl Shape for B { fn area(self) -> i32 { self.q } }
fn main() -> i32 { let a = A { p: 20 }; let b = B { q: 22 }; a.area() + b.area() }
