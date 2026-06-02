trait Greet { fn hello(self) -> i32 { 42 } }
struct P { x: i32 }
impl Greet for P {}
fn main() -> i32 {
    let p = P { x: 1 };
    p.hello()
}
