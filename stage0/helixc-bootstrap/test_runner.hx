// test_runner.hx -- Helix v1.1 H1 sub-item 2: Helix-native test runner (retires the shell
// corpus check() loop). For each corpus program: stage it to /tmp/k2_in.hx, run_process
// K2 to compile (K2's driver reads /tmp/k2_in.hx -> writes /tmp/k2_out.bin), set_exec +
// run_process the result, compare the exit code to the expected value. Uses run_process
// (STRLIT path -> child exit) + set_exec -- kovc-only builtins, so this is compiled by K2.
// Returns the number of FAILURES (0 = all 35 pass). Expected exits mirror feature_corpus.sh.
// 7 example programs (helixc/examples/) + 28 generated programs (stage0/helixc-bootstrap/
// corpus/, produced by .stage33-logs/extract_corpus.sh from scripts/feature_corpus.sh).

fn run_one(start: i32, len: i32, expected: i32) -> i32 {
    write_file_to_arena("/tmp/k2_in.hx", start, len);
    run_process("/tmp/K2.bin");
    set_exec("/tmp/k2_out.bin");
    let got = run_process("/tmp/k2_out.bin");
    if got == expected { 1 } else { 0 }
}

fn main() -> i32 {
    let mut fail = 0;
    let mut s = 0;
    let mut l = 0;

    // --- 7 example files ---
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/helixc/examples/exit42.hx"); if run_one(s, l, 42) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/helixc/examples/matmul_2x2.hx"); if run_one(s, l, 69) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/helixc/examples/hbs_sample_enum_struct.hx"); if run_one(s, l, 129) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/helixc/examples/hbs_sample_option.hx"); if run_one(s, l, 42) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/helixc/examples/hbs_sample_recursion.hx"); if run_one(s, l, 120) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/helixc/examples/dogfood_18_pat_struct_showcase.hx"); if run_one(s, l, 42) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/helixc/examples/gradient_descent.hx"); if run_one(s, l, 42) == 0 { fail = fail + 1; }

    // --- 28 generated corpus files ---
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/result_inline.hx"); if run_one(s, l, 42) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/i64_basic.hx"); if run_one(s, l, 42) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/i64_mul_beyond.hx"); if run_one(s, l, 6) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/i64_div_beyond.hx"); if run_one(s, l, 50) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/i64_cmp.hx"); if run_one(s, l, 1) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/i64_neg.hx"); if run_one(s, l, 5) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/u64_shr.hx"); if run_one(s, l, 1) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/u8_wrap.hx"); if run_one(s, l, 42) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/u16_wrap.hx"); if run_one(s, l, 42) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/i16_ovf.hx"); if run_one(s, l, 42) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/assoc_sub.hx"); if run_one(s, l, 5) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/assoc_div.hx"); if run_one(s, l, 10) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/cmp_ne.hx"); if run_one(s, l, 1) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/cmp_ge.hx"); if run_one(s, l, 1) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/cmp_le.hx"); if run_one(s, l, 1) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/bit_andor.hx"); if run_one(s, l, 9) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/bit_xor.hx"); if run_one(s, l, 240) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/bit_shl.hx"); if run_one(s, l, 16) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/arr_idx.hx"); if run_one(s, l, 20) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/while_sum.hx"); if run_one(s, l, 10) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/while_break.hx"); if run_one(s, l, 7) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/f64_add.hx"); if run_one(s, l, 4) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/f64_mul.hx"); if run_one(s, l, 12) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/tuple2.hx"); if run_one(s, l, 7) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/impl_method.hx"); if run_one(s, l, 42) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/match_or.hx"); if run_one(s, l, 10) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/match_range.hx"); if run_one(s, l, 1) == 0 { fail = fail + 1; }
    s = __arena_len(); l = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/corpus/vec_arena.hx"); if run_one(s, l, 45) == 0 { fail = fail + 1; }

    fail
}
