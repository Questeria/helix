struct P { x: i32 }
impl P { fn get(self) -> i32 { self.x } }
fn main() -> i32 { let p = P { x: 42 }; p.get() }
