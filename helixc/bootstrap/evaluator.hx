// Stage-3 tree-walking evaluator for the bootstrap chain.
//
// Walks the AST produced by stage-2 parser and evaluates it to an
// i32. Runs the entire lex -> parse -> eval pipeline in pure Helix
// against a source file on disk. This is the simplest possible
// stage-3 — full machine-code emission requires a sys_write builtin
// we haven't added yet, but a working evaluator already proves the
// front-end is solid end-to-end on real source files.
//
// Environment layout: a stack of 3-slot (name_byte_start, name_len,
// value) records in the arena. Lookups walk backwards (most-recent
// binding wins, providing lexical shadowing).
//
// AST tag table mirrors helixc/bootstrap/parser.hx:
//   0 INT, 1 VAR, 2 ADD, 3 SUB, 4 MUL, 5 DIV, 6 LT, 7 IF, 8 LET,
//   9 NEG, 99 ERR.
//
// License: Apache 2.0.

// --------------------------------------------------------------
// Compare two byte-spans in the arena. Borrowed verbatim from
// parser.hx so the evaluator is self-contained.
// --------------------------------------------------------------
@pure
fn ev_byte_eq(src_a: i32, len_a: i32, src_b: i32, len_b: i32) -> i32 {
    if len_a != len_b { 0 }
    else {
        let mut i: i32 = 0;
        let mut ok: i32 = 1;
        while i < len_a {
            if ok == 1 {
                let ba = __arena_get(src_a + i);
                let bb = __arena_get(src_b + i);
                if ba != bb { ok = 0; };
            };
            i = i + 1;
        }
        ok
    }
}

// --------------------------------------------------------------
// Push (name_start, name_len, value) onto the env stack and
// return 0. The caller tracks env_top by reading __arena_len()
// before/after this call.
// --------------------------------------------------------------
fn env_push(name_start: i32, name_len: i32, value: i32) -> i32 {
    __arena_push(name_start);
    __arena_push(name_len);
    __arena_push(value);
    0
}

// --------------------------------------------------------------
// Look up a name (given as byte-span) in the env. env_base is the
// index of the first env slot; env_top is the index just past the
// last slot. Walks 3 slots at a time from end to start. Returns
// 0 if the name is unbound (not great, but defensible — the parser
// shouldn't produce free variables in well-formed input).
// --------------------------------------------------------------
fn env_lookup(env_base: i32, env_top: i32,
              name_start: i32, name_len: i32) -> i32 {
    let mut i: i32 = env_top - 3;
    let mut found: i32 = 0;
    let mut value: i32 = 0;
    while i >= env_base {
        if found == 0 {
            let ns = __arena_get(i);
            let nl = __arena_get(i + 1);
            if ev_byte_eq(ns, nl, name_start, name_len) == 1 {
                value = __arena_get(i + 2);
                found = 1;
            };
            i = i - 3;
        } else {
            i = env_base - 1;
        };
    }
    value
}

// --------------------------------------------------------------
// AST accessors.
// --------------------------------------------------------------
@pure fn ast_tag(idx: i32) -> i32 { __arena_get(idx) }
@pure fn ast_p1(idx: i32) -> i32  { __arena_get(idx + 1) }
@pure fn ast_p2(idx: i32) -> i32  { __arena_get(idx + 2) }
@pure fn ast_p3(idx: i32) -> i32  { __arena_get(idx + 3) }

// --------------------------------------------------------------
// Walk the AST and produce a value. SysV 6-int-param limit again:
// we pass (idx, env_base, env_top) — three params. Subexpressions
// extending the env push and pass a new env_top.
// --------------------------------------------------------------
fn eval_ast(idx: i32, env_base: i32, env_top: i32) -> i32 {
    let t = ast_tag(idx);
    let p1 = ast_p1(idx);
    let p2 = ast_p2(idx);
    let p3 = ast_p3(idx);
    if t == 0 { p1 }
    else { if t == 1 {
        // VAR: name span at (p1, p2)
        env_lookup(env_base, env_top, p1, p2)
    } else { if t == 2 {
        eval_ast(p1, env_base, env_top) + eval_ast(p2, env_base, env_top)
    } else { if t == 3 {
        eval_ast(p1, env_base, env_top) - eval_ast(p2, env_base, env_top)
    } else { if t == 4 {
        eval_ast(p1, env_base, env_top) * eval_ast(p2, env_base, env_top)
    } else { if t == 5 {
        let r = eval_ast(p2, env_base, env_top);
        if r == 0 { 0 } else { eval_ast(p1, env_base, env_top) / r }
    } else { if t == 6 {
        let l = eval_ast(p1, env_base, env_top);
        let r = eval_ast(p2, env_base, env_top);
        if l < r { 1 } else { 0 }
    } else { if t == 7 {
        let c = eval_ast(p1, env_base, env_top);
        if c != 0 {
            eval_ast(p2, env_base, env_top)
        } else {
            eval_ast(p3, env_base, env_top)
        }
    } else { if t == 8 {
        // LET: p1=name_start, p2=name_len, p3 = body_idx, p4 = value_idx.
        // Audit-14: split out of legacy packed encoding to support
        // arena indices > 65535 (full self-host).
        let body_idx = p3;
        let value_idx = __arena_get(idx + 4);
        let v = eval_ast(value_idx, env_base, env_top);
        env_push(p1, p2, v);
        eval_ast(body_idx, env_base, env_top + 3)
    } else { if t == 9 {
        0 - eval_ast(p1, env_base, env_top)
    } else {
        0    // ERR or unknown
    }}}}}}}}}}
}

// --------------------------------------------------------------
// Top-level driver: lex + parse + eval.
// --------------------------------------------------------------
fn run_source(src_start: i32, src_len: i32) -> i32 {
    let tok_base = __arena_len();
    lex(src_start, src_len);
    let ast_root = parse_top(tok_base);
    let env_base = __arena_len();
    eval_ast(ast_root, env_base, env_base)
}
