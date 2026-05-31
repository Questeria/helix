fn main() -> i32 {
    let n = read_file_to_arena("/tmp/seed_read_test.txt");
    let first = __arena_get(0);
    n + first
}
