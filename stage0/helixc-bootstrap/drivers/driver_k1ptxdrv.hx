
fn main() -> i32 {
    let src_start = __arena_len();
    let src_len = read_file_to_arena("/tmp/kernel_in.hx");
    let tok_base = __arena_len();
    lex(src_start, src_len);
    let ast_root = parse_top(tok_base);
    let total = emit_auto_for_ast_to_path(ast_root);
    let ptx_start = __arena_len() - total;
    write_file_to_arena("/tmp/out.ptx", ptx_start, total)
}
