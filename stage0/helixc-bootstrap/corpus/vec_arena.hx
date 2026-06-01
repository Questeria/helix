fn vec_new() -> i32 { __arena_push(0) }
fn vec_push(v: i32, x: i32) -> i32 { let n = __arena_get(v); __arena_set(v, n + 1); __arena_push(x) }
fn vec_len(v: i32) -> i32 { __arena_get(v) }
fn vec_get(v: i32, i: i32) -> i32 { __arena_get(v + 1 + i) }
fn main() -> i32 {
    let v = vec_new();
    vec_push(v, 10);
    vec_push(v, 20);
    vec_push(v, 12);
    vec_get(v, 0) + vec_get(v, 1) + vec_get(v, 2) + vec_len(v)
}
