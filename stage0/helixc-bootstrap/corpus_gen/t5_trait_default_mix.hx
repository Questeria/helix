trait Greet { fn hello(self) -> i32 { 10 } }
struct A { x: i32 }
struct B { x: i32 }
impl Greet for A {}
impl Greet for B { fn hello(self) -> i32 { 32 } }
fn main() -> i32 {
    let a = A { x: 0 };
    let b = B { x: 0 };
    a.hello() + b.hello()
}
