trait Shape { fn area(self) -> i32 }
struct Rec { w: i32, h: i32 }
impl Shape for Rec { fn area(self) -> i32 { self.w * self.h } }
fn main() -> i32 { let r = Rec { w: 6, h: 7 }; r.area() }
