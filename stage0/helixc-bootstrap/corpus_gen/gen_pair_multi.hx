struct Pair[T] { a: T, b: T }
impl<T> Pair<T> {
    fn first(self) -> T { self.a }
    fn second(self) -> T { self.b }
}
fn main() -> i32 {
    let pi = Pair::<i32>{ a: 3, b: 4 };
    let pf = Pair::<f32>{ a: 2.0, b: 3.0 };
    let si: i32 = pi.first() + pi.second();
    let sf: f32 = pf.first() + pf.second();
    si + (sf as i32)
}
