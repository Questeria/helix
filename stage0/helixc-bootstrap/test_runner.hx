// test_runner.hx -- Helix v1.1 H1 sub-item 2: Helix-native test runner (retire the shell
// corpus check() loop). For each corpus program: stage it to /tmp/k2_in.hx, run K2 to
// compile it (K2's driver reads /tmp/k2_in.hx -> writes /tmp/k2_out.bin), set_exec + run
// the result, compare the exit code to the expected value. Uses run_process (STRLIT path
// -> child exit code) + set_exec -- kovc-ONLY builtins, so this is compiled by K2 (NOT the
// seed). Returns the number of FAILURES (0 = all pass), so the runner's own exit code is
// the verdict. MINIMAL version (3 example programs) to prove the run_process orchestration;
// expand to the full corpus once proven.
//
// Path-must-be-literal constraint (read_file_to_arena / run_process take string literals):
// each program's read_file_to_arena is inlined in main; run_one does the fixed-path rest.

fn run_one(start: i32, len: i32, expected: i32) -> i32 {
    write_file_to_arena("/tmp/k2_in.hx", start, len);
    run_process("/tmp/K2.bin");
    set_exec("/tmp/k2_out.bin");
    let got = run_process("/tmp/k2_out.bin");
    if got == expected { 1 } else { 0 }
}

fn main() -> i32 {
    let mut fail = 0;

    let s1 = __arena_len();
    let l1 = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/helixc/examples/exit42.hx");
    if run_one(s1, l1, 42) == 0 { fail = fail + 1; }

    let s2 = __arena_len();
    let l2 = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/helixc/examples/matmul_2x2.hx");
    if run_one(s2, l2, 69) == 0 { fail = fail + 1; }

    let s3 = __arena_len();
    let l3 = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/helixc/examples/hbs_sample_option.hx");
    if run_one(s3, l3, 42) == 0 { fail = fail + 1; }

    fail
}
