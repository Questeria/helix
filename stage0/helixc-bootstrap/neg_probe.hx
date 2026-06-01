// neg_probe.hx -- negative control for the Helix test runner: run_one must DETECT a wrong
// expected exit (not always-pass). Runs exit42.hx (exits 42) but asserts 99; run_one should
// return 0 (mismatch), so main returns 1. The gate expects this binary to exit 1.
fn run_one(start: i32, len: i32, expected: i32) -> i32 {
    write_file_to_arena("/tmp/k2_in.hx", start, len);
    run_process("/tmp/K2.bin");
    set_exec("/tmp/k2_out.bin");
    let got = run_process("/tmp/k2_out.bin");
    if got == expected { 1 } else { 0 }
}
fn main() -> i32 {
    let s = __arena_len();
    let l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/helixc/examples/exit42.hx");
    if run_one(s, l, 99) == 0 { 1 } else { 0 }
}
