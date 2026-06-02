struct Vec[T] { base: i32, len: i32 }
impl<T> Vec<T> {
    fn at(self, i: i32) -> T { __arena_get(self.base + i) }
}
fn main() -> i32 {
    let b = __arena_len();
    __arena_push(10);
    __arena_push(20);
    __arena_push(12);
    let v = Vec::<i32>{ base: b, len: 3 };
    v.at(0) + v.at(1) + v.at(2)
}
