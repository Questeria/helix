
fn main() -> i32 {
    let src_start = __arena_len();
    let src_len = read_file_to_arena("/tmp/k1_in.hx");
    let tok_base = __arena_len();
    lex(src_start, src_len);
    let ast_root = parse_top(tok_base);
    // H-3: compile-time file:line:col diagnostic on a parse error.
    // Clean input (e.g. the self-host source) hits no AST_ERR, so
    // err_off < 0 and the normal emit path runs byte-identically.
    let err_off = find_first_err_offset(ast_root);
    if err_off >= 0 {
        print_str("/tmp/k1_in.hx");
        report_parse_diag(src_start, err_off);
        1
    } else {
        let total = emit_elf_for_ast_to_path(ast_root);
        let elf_start = __arena_len() - total;
        write_file_to_arena("/tmp/k1_out.bin", elf_start, total)
    }
}
