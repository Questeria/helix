fn main() -> i32 {
    let a = __arena_push(72);
    let b = __arena_push(105);
    let c = __arena_push(33);
    write_file_to_arena("/tmp/seedout.bin", 0, 3)
}
