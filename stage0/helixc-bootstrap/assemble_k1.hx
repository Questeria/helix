// assemble_k1.hx -- Helix-native port of assemble_k1.sh (Helix v1.1 H1: retire shell).
// Pure file concatenation: builds k1src.hx / k1input.hx / k1ptxdrv.hx =
//   strip_demo(lexer.hx) + parser.hx(whole) + strip_demo(kovc.hx) + driver_fragment,
// all CR-stripped (frozen sources are CRLF; output must be LF, byte-identical to the
// shell version). The 3 driver-main fragments live in drivers/driver_{k1src,k1input,
// k1ptxdrv}.hx (captured from gen_driver). read_file_to_arena's path arg MUST be a
// string literal, so all paths are hard-coded (absolute WSL). Outputs go to /tmp/hx_*
// for the byte-identical gate vs assemble_k1.sh.
//
// strip_demo keep-length = byte offset of the START of the dashes line ('// ----')
// that immediately precedes the '// Demo:' line (mirrors the awk in assemble_k1.sh).

// find the keep-length of a source: offset of the dashes line before "// Demo:".
fn find_keep_len(base: i32, len: i32) -> i32 {
    let mut demo_at = 0 - 1;
    let mut i = 1;
    while i + 8 <= len {
        if __arena_get(base + i - 1) == 10 {
            let mut ok = 1;
            if __arena_get(base + i) != 47 { ok = 0; }
            if __arena_get(base + i + 1) != 47 { ok = 0; }
            if __arena_get(base + i + 2) != 32 { ok = 0; }
            if __arena_get(base + i + 3) != 68 { ok = 0; }
            if __arena_get(base + i + 4) != 101 { ok = 0; }
            if __arena_get(base + i + 5) != 109 { ok = 0; }
            if __arena_get(base + i + 6) != 111 { ok = 0; }
            if __arena_get(base + i + 7) != 58 { ok = 0; }
            if ok == 1 { if demo_at < 0 { demo_at = i; } }
        }
        i = i + 1;
    }
    let mut j = demo_at - 2;
    let mut prev_nl = 0 - 1;
    while j >= 0 {
        if __arena_get(base + j) == 10 { prev_nl = j; j = 0 - 1; } else { j = j - 1; }
    }
    prev_nl + 1
}

// append bytes [base, base+len) to the arena tail, skipping CR (0x0D = 13).
fn append_stripped(base: i32, len: i32) -> i32 {
    let mut i = 0;
    while i < len {
        let b = __arena_get(base + i);
        if b != 13 { __arena_push(b); }
        i = i + 1;
    }
    0
}

fn main() -> i32 {
    // read the 3 frozen sources once
    let lex_base = __arena_len();
    let lex_len = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/helixc/bootstrap/lexer.hx");
    let par_base = __arena_len();
    let par_len = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/helixc/bootstrap/parser.hx");
    let kov_base = __arena_len();
    let kov_len = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/helixc/bootstrap/kovc.hx");
    let lex_keep = find_keep_len(lex_base, lex_len);
    let kov_keep = find_keep_len(kov_base, kov_len);

    // variant 1: k1src
    let d1_base = __arena_len();
    let d1_len = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/drivers/driver_k1src.hx");
    let o1_base = __arena_len();
    append_stripped(lex_base, lex_keep);
    append_stripped(par_base, par_len);
    append_stripped(kov_base, kov_keep);
    append_stripped(d1_base, d1_len);
    let o1_len = __arena_len() - o1_base;
    write_file_to_arena("/tmp/hx_k1src.hx", o1_base, o1_len);

    // variant 2: k1input
    let d2_base = __arena_len();
    let d2_len = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/drivers/driver_k1input.hx");
    let o2_base = __arena_len();
    append_stripped(lex_base, lex_keep);
    append_stripped(par_base, par_len);
    append_stripped(kov_base, kov_keep);
    append_stripped(d2_base, d2_len);
    let o2_len = __arena_len() - o2_base;
    write_file_to_arena("/tmp/hx_k1input.hx", o2_base, o2_len);

    // variant 3: k1ptxdrv
    let d3_base = __arena_len();
    let d3_len = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/drivers/driver_k1ptxdrv.hx");
    let o3_base = __arena_len();
    append_stripped(lex_base, lex_keep);
    append_stripped(par_base, par_len);
    append_stripped(kov_base, kov_keep);
    append_stripped(d3_base, d3_len);
    let o3_len = __arena_len() - o3_base;
    write_file_to_arena("/tmp/hx_k1ptxdrv.hx", o3_base, o3_len);
    0
}
