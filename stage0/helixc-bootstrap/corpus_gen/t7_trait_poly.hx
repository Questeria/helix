trait Shape { fn area(self) -> i32 }
struct Sq { s: i32 }
struct Rec { w: i32, h: i32 }
impl Shape for Sq { fn area(self) -> i32 { self.s * self.s } }
impl Shape for Rec { fn area(self) -> i32 { self.w * self.h } }
fn main() -> i32 {
    let sq = Sq { s: 5 };
    let r = Rec { w: 17, h: 1 };
    sq.area() + r.area()
}
