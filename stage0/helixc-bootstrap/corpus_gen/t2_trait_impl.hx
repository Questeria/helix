trait Greet { fn hello(self) -> i32 }
struct P { x: i32 }
impl Greet for P { fn hello(self) -> i32 { self.x + 5 } }
fn main() -> i32 {
    let p = P { x: 37 };
    p.hello()
}
