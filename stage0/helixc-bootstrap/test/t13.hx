fn main() -> i32 {
    let a = __arena_push(10);
    let b = __arena_push(20);
    let c = __arena_push(12);
    __arena_set(b, 30);
    __arena_get(a) + __arena_get(b) + __arena_get(c) + __arena_len()
}
