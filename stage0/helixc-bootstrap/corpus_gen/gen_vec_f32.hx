struct Vec[T] { base: i32, len: i32 }
impl<T> Vec<T> {
    fn at(self, i: i32) -> T { __arena_get(self.base + i) }
}
fn main() -> i32 {
    let b = __arena_len();
    let x: f32 = 2.0;
    let y: f32 = 3.0;
    __arena_push(x);
    __arena_push(y);
    let v = Vec::<f32>{ base: b, len: 2 };
    let s: f32 = v.at(0) + v.at(1);
    s as i32
}
