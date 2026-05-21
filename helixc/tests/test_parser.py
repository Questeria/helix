"""Tests for helixc.frontend.parser."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse, ParseError
from helixc.frontend import ast_nodes as ast


def first_item(src: str) -> ast.Item:
    p = parse(src)
    assert len(p.items) >= 1, "expected at least one item"
    return p.items[0]


# ============================================================================
# Module + use
# ============================================================================
def test_module_decl():
    p = parse("module foo::bar")
    assert p.module is not None
    assert p.module.path == ["foo", "bar"]


def test_module_with_semi():
    p = parse("module foo::bar;")
    assert p.module.path == ["foo", "bar"]


def test_use_decl():
    p = parse("use core::tensor;")
    assert isinstance(p.items[0], ast.UseDecl)
    assert p.items[0].path == ["core", "tensor"]


# ============================================================================
# Function declarations
# ============================================================================
def test_simplest_fn():
    p = parse("fn foo() {}")
    fn = p.items[0]
    assert isinstance(fn, ast.FnDecl)
    assert fn.name == "foo"
    assert fn.params == []
    assert fn.return_ty is None
    assert fn.body.stmts == []
    assert fn.body.final_expr is None


def test_fn_with_params_and_return():
    p = parse("fn add(a: i32, b: i32) -> i32 { a + b }")
    fn = p.items[0]
    assert isinstance(fn, ast.FnDecl)
    assert fn.name == "add"
    assert len(fn.params) == 2
    assert fn.params[0].name == "a"
    assert isinstance(fn.params[0].ty, ast.TyName)
    assert fn.params[0].ty.name == "i32"
    assert isinstance(fn.return_ty, ast.TyName)
    assert fn.return_ty.name == "i32"
    assert fn.body.final_expr is not None
    assert isinstance(fn.body.final_expr, ast.Binary)


def test_fn_with_generics():
    p = parse("fn id[T](x: T) -> T { x }")
    fn = p.items[0]
    assert len(fn.generics) == 1
    assert fn.generics[0].name == "T"
    assert fn.generics[0].kind == "type"


def test_fn_with_size_generic():
    p = parse("fn take[N: size](a: i32) -> i32 { a }")
    fn = p.items[0]
    assert fn.generics[0].kind == "size"


def test_fn_with_attr():
    p = parse("@kernel fn foo() {}")
    fn = p.items[0]
    assert "kernel" in fn.attrs


def test_struct_with_attr_stage66_inc4():
    """Stage 66 Inc 4 — `@copy` (and other attrs) on a struct decl
    parse cleanly and populate `StructDecl.attrs`. Empty case stays
    `[]` so existing tests are unaffected."""
    p = parse("@copy struct Pt { x: f32, y: f32 }")
    s = p.items[0]
    assert s.__class__.__name__ == "StructDecl"
    assert "copy" in s.attrs

    # plain struct with no attrs
    p2 = parse("struct Heavy { x: f32, y: f32 }")
    s2 = p2.items[0]
    assert s2.attrs == []


def test_fn_with_pub():
    p = parse("pub fn foo() {}")
    fn = p.items[0]
    assert fn.is_pub


def test_fn_with_where_clauses():
    p = parse("fn f[N: size, M: size]() where N % 16 == 0, M >= N, {}")
    fn = p.items[0]
    assert len(fn.where_clauses) == 2


def test_stage31_type_alias_with_refinement_where_clause():
    p = parse("type Probability = f64 where 0.0 <= self <= 1.0;")
    alias = p.items[0]
    assert isinstance(alias, ast.TypeAlias)
    assert alias.name == "Probability"
    assert isinstance(alias.target, ast.TyName)
    assert alias.target.name == "f64"
    assert len(alias.where_clauses) == 1


# ============================================================================
# Tensor and tile types
# ============================================================================
def test_tensor_type():
    p = parse("fn f(x: tensor<f32, [4, 8]>) {}")
    fn = p.items[0]
    ty = fn.params[0].ty
    assert isinstance(ty, ast.TyTensor)
    assert isinstance(ty.dtype, ast.TyName)
    assert ty.dtype.name == "f32"
    assert len(ty.shape) == 2


def test_tensor_with_device():
    p = parse("fn f(x: tensor<bf16, [N, M], gpu(0)>) {}")
    fn = p.items[0]
    ty = fn.params[0].ty
    assert isinstance(ty, ast.TyTensor)
    assert ty.device is not None


def test_tile_type():
    p = parse("fn f(x: tile<bf16, [16, 16], smem>) {}")
    fn = p.items[0]
    ty = fn.params[0].ty
    assert isinstance(ty, ast.TyTile)
    assert isinstance(ty.memspace, ast.Name)
    assert ty.memspace.name == "smem"


# ============================================================================
# Structs and enums
# ============================================================================
def test_struct_decl():
    p = parse("struct Foo { x: i32, y: f32 }")
    s = p.items[0]
    assert isinstance(s, ast.StructDecl)
    assert s.name == "Foo"
    assert len(s.fields) == 2


def test_enum_decl():
    p = parse("enum Color { Red, Green, Blue(i32, i32) }")
    e = p.items[0]
    assert isinstance(e, ast.EnumDecl)
    assert len(e.variants) == 3
    assert e.variants[2].name == "Blue"
    assert len(e.variants[2].payload_tys) == 2


# ============================================================================
# Let / const / assignment
# ============================================================================
def test_let_simple():
    p = parse("fn f() { let x = 42; }")
    fn = p.items[0]
    s = fn.body.stmts[0]
    assert isinstance(s, ast.Let)
    assert s.name == "x"
    assert isinstance(s.value, ast.IntLit)
    assert s.value.value == 42


def test_let_typed():
    p = parse("fn f() { let x: i32 = 42; }")
    fn = p.items[0]
    s = fn.body.stmts[0]
    assert isinstance(s, ast.Let)
    assert isinstance(s.ty, ast.TyName)
    assert s.ty.name == "i32"


def test_let_mut():
    p = parse("fn f() { let mut x = 0; }")
    s = p.items[0].body.stmts[0]
    assert isinstance(s, ast.Let)
    assert s.is_mut


def test_assignment():
    p = parse("fn f() { let mut x = 0; x = 1; }")
    s2 = p.items[0].body.stmts[1]
    assert isinstance(s2, ast.ExprStmt)
    assert isinstance(s2.expr, ast.Assign)
    assert s2.expr.op == "="


def test_compound_assign():
    p = parse("fn f() { let mut x = 0; x += 1; }")
    s2 = p.items[0].body.stmts[1]
    assert isinstance(s2.expr, ast.Assign)
    assert s2.expr.op == "+="


# ============================================================================
# Expressions
# ============================================================================
def test_arith_precedence():
    p = parse("fn f() -> i32 { 1 + 2 * 3 }")
    e = p.items[0].body.final_expr
    assert isinstance(e, ast.Binary) and e.op == "+"
    assert isinstance(e.right, ast.Binary) and e.right.op == "*"


def test_paren_overrides_precedence():
    p = parse("fn f() -> i32 { (1 + 2) * 3 }")
    e = p.items[0].body.final_expr
    assert isinstance(e, ast.Binary) and e.op == "*"
    assert isinstance(e.left, ast.Binary) and e.left.op == "+"


def test_unary_minus():
    p = parse("fn f() -> i32 { -42 }")
    e = p.items[0].body.final_expr
    assert isinstance(e, ast.Unary) and e.op == "-"


def test_call_and_index():
    p = parse("fn f() -> i32 { foo(1, 2)[3] }")
    e = p.items[0].body.final_expr
    assert isinstance(e, ast.Index)
    assert isinstance(e.callee, ast.Call)


def test_field_access():
    p = parse("fn f() { foo.bar.baz; }")
    e = p.items[0].body.stmts[0].expr
    assert isinstance(e, ast.Field)


def test_path():
    p = parse("fn f() { tensor::zeros(); }")
    e = p.items[0].body.stmts[0].expr
    assert isinstance(e, ast.Call)
    assert isinstance(e.callee, ast.Path)
    assert e.callee.segments == ["tensor", "zeros"]


def test_turbofish():
    p = parse("fn f() { foo::<i32>(42); }")
    e = p.items[0].body.stmts[0].expr
    assert isinstance(e, ast.Call)
    assert isinstance(e.callee, ast.Name)
    assert len(e.callee.generics) == 1


def test_if_expr():
    p = parse("fn f() -> i32 { if true { 1 } else { 2 } }")
    e = p.items[0].body.final_expr
    assert isinstance(e, ast.If)


def test_if_else_if():
    p = parse("fn f() -> i32 { if true { 1 } else if false { 2 } else { 3 } }")
    e = p.items[0].body.final_expr
    assert isinstance(e, ast.If)
    assert isinstance(e.else_, ast.If)


def test_for_loop():
    p = parse("fn f() { for i in 0 .. 10 { i; } }")
    e = p.items[0].body.stmts[0].expr
    assert isinstance(e, ast.For)
    assert e.var_name == "i"
    assert isinstance(e.iter_expr, ast.Range)


def test_match():
    p = parse("fn f(x: i32) -> i32 { match x { 0 => 1, _ => 2 } }")
    e = p.items[0].body.final_expr
    assert isinstance(e, ast.Match)
    assert len(e.arms) == 2


def test_array_literal():
    p = parse("fn f() { let xs = [1, 2, 3]; }")
    s = p.items[0].body.stmts[0]
    assert isinstance(s.value, ast.ArrayLit)
    assert len(s.value.elems) == 3


def test_tuple_literal():
    p = parse("fn f() { let t = (1, 2.0, 3); }")
    s = p.items[0].body.stmts[0]
    assert isinstance(s.value, ast.TupleLit)
    assert len(s.value.elems) == 3


def test_unit_value():
    p = parse("fn f() { let u = (); }")
    s = p.items[0].body.stmts[0]
    assert isinstance(s.value, ast.TupleLit)
    assert s.value.elems == []


# ============================================================================
# Realistic programs
# ============================================================================
def test_matmul_signature_only():
    src = """
    fn matmul[N: size, M: size, P: size](
        a: tensor<f32, [N, M]>,
        b: tensor<f32, [M, P]>,
    ) -> tensor<f32, [N, P]>
    where N % 16 == 0, M % 16 == 0, P % 16 == 0,
    {
        let c = tensor::zeros();
        c
    }
    """
    fn = first_item(src)
    assert isinstance(fn, ast.FnDecl)
    assert fn.name == "matmul"
    assert len(fn.generics) == 3
    assert all(g.kind == "size" for g in fn.generics)
    assert len(fn.params) == 2
    assert isinstance(fn.return_ty, ast.TyTensor)
    assert len(fn.where_clauses) == 3


def test_kernel_attribute():
    src = """
    @kernel
    fn add[N: size](a: tile<f32, [N], reg>) -> tile<f32, [N], reg> {
        a
    }
    """
    fn = first_item(src)
    assert "kernel" in fn.attrs


def test_multiple_items():
    src = """
    module my::module

    use core::tensor;

    pub fn add(a: i32, b: i32) -> i32 { a + b }

    struct Point { x: f32, y: f32 }

    enum Maybe[T] { None, Some(T) }
    """
    p = parse(src)
    assert p.module is not None
    assert len(p.items) == 4
    assert isinstance(p.items[0], ast.UseDecl)
    assert isinstance(p.items[1], ast.FnDecl)
    assert isinstance(p.items[2], ast.StructDecl)
    assert isinstance(p.items[3], ast.EnumDecl)


def test_const_decl():
    p = parse("const MAX: i32 = 100;")
    c = p.items[0]
    assert isinstance(c, ast.ConstDecl)
    assert c.name == "MAX"


def test_grad_call():
    src = "fn f() { let g = grad(loss); }"
    s = first_item(src).body.stmts[0]
    assert isinstance(s.value, ast.Call)
    assert isinstance(s.value.callee, ast.Name)
    assert s.value.callee.name == "grad"


# ============================================================================
# Error cases
# ============================================================================
def test_missing_fn_name():
    try:
        parse("fn () {}")
        assert False
    except ParseError:
        pass


def test_missing_semi():
    try:
        parse("fn f() { let x = 42 }")
        assert False
    except ParseError:
        pass


def test_unclosed_brace():
    try:
        parse("fn f() {")
        assert False
    except ParseError:
        pass


def test_unclosed_block_error_names_open_brace():
    """Cycle-3 audit: a truncated block should produce an error that
    points at the unclosed `{`, not 'expected expression got EOF'."""
    try:
        parse("fn f() { if true { 1 } else { 0 }")
        assert False, "should have raised"
    except ParseError as e:
        msg = str(e)
        assert "unclosed" in msg.lower() and "1:8" in msg, f"got {msg}"


def test_range_with_arithmetic_rhs_groups_correctly():
    """`0..n*2` must parse as Range(0, Binary(n,*,2)), not
    Binary(Range(0,n),*,2)."""
    import helixc.frontend.ast_nodes as A
    p = parse("fn f() -> i32 { let mut s = 0; for i in 0 .. 3 * 2 { s = s + i; }; s }")
    fn = next(item for item in p.items if isinstance(item, A.FnDecl))
    for_stmt = fn.body.stmts[1]
    if isinstance(for_stmt, A.ExprStmt):
        for_stmt = for_stmt.expr
    assert isinstance(for_stmt, A.For), f"expected For, got {type(for_stmt).__name__}"
    assert isinstance(for_stmt.iter_expr, A.Range), \
        f"iter_expr should be Range, got {type(for_stmt.iter_expr).__name__}"
    end = for_stmt.iter_expr.end
    assert isinstance(end, A.Binary) and end.op == "*", \
        f"end should be Binary(*), got {type(end).__name__}"


# ============================================================================
# Agent declarations (Phase 3-viii)
# ============================================================================
def test_simple_agent():
    src = """
    agent Planner {
        fn propose(state: i32) -> i32;
    }
    """
    p = parse(src)
    assert isinstance(p.items[0], ast.AgentDecl)
    assert p.items[0].name == "Planner"
    assert len(p.items[0].methods) == 1
    assert p.items[0].methods[0].name == "propose"


def test_multi_method_agent():
    src = """
    agent Critic {
        fn evaluate(state: i32, action: i32) -> bool;
        fn score(action: i32) -> f32;
    }
    """
    p = parse(src)
    a = p.items[0]
    assert isinstance(a, ast.AgentDecl)
    assert len(a.methods) == 2
    assert a.methods[0].name == "evaluate"
    assert len(a.methods[0].params) == 2
    assert a.methods[1].name == "score"


def test_pub_agent():
    src = "pub agent Foo { fn bar() -> i32; }"
    a = parse(src).items[0]
    assert isinstance(a, ast.AgentDecl)
    assert a.is_pub


def test_agent_alongside_fn():
    src = """
    agent A { fn act() -> i32; }
    fn main() -> i32 { 0 }
    """
    p = parse(src)
    assert len(p.items) == 2
    assert isinstance(p.items[0], ast.AgentDecl)
    assert isinstance(p.items[1], ast.FnDecl)


def test_struct_lit_parses():
    """Struct literal `Point { x: 10, y: 20 }` parses to StructLit."""
    src = """
    struct Point { x: i32, y: i32 }
    fn main() -> i32 {
        let p = Point { x: 10, y: 20 };
        p.x
    }
    """
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, ast.FnDecl))
    let_stmt = fn.body.stmts[0]
    assert isinstance(let_stmt, ast.Let)
    assert isinstance(let_stmt.value, ast.StructLit)
    assert let_stmt.value.name == "Point"
    assert len(let_stmt.value.fields) == 2
    assert let_stmt.value.fields[0][0] == "x"
    assert let_stmt.value.fields[1][0] == "y"


def test_struct_lit_disambiguated_from_block():
    """`if cond { ... }` should not be parsed as a struct literal even
    though the syntax overlaps. Disambig: struct lit requires `{ IDENT :`."""
    src = """
    fn main() -> i32 {
        if true { 42 } else { 0 }
    }
    """
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, ast.FnDecl))
    if_expr = fn.body.final_expr
    assert isinstance(if_expr, ast.If), \
        f"expected If, got {type(if_expr).__name__}"


def test_partial_attribute_parses():
    """`@partial` attribute should appear in FnDecl.attrs."""
    src = """
    @partial
    fn loop_forever() -> i32 {
        loop { 0 }
    }
    """
    prog = parse(src)
    fn = prog.items[0]
    assert isinstance(fn, ast.FnDecl)
    assert "partial" in fn.attrs, f"expected 'partial' in attrs, got {fn.attrs}"


def test_total_attribute_parses():
    """`@total` attribute should appear in FnDecl.attrs."""
    src = """
    @total
    fn safe_add(a: i32, b: i32) -> i32 { a + b }
    """
    prog = parse(src)
    fn = prog.items[0]
    assert isinstance(fn, ast.FnDecl)
    assert "total" in fn.attrs, f"expected 'total' in attrs, got {fn.attrs}"


def test_match_arm_span_covers_pattern():
    """Each arm.span should point at the start of its pattern (including
    or-pattern alternatives and range pattern endpoints)."""
    src = """
fn f(x: i32) -> i32 {
    match x {
        0..10 => 1,
        20 | 30 | 40 => 2,
        y => y,
    }
}
"""
    prog = parse(src)
    fn = prog.items[0]
    match_expr = fn.body.final_expr
    assert isinstance(match_expr, ast.Match)
    arm0 = match_expr.arms[0]
    assert arm0.span == arm0.pattern.span, \
        f"arm.span {arm0.span} != pattern.span {arm0.pattern.span}"
    arm1 = match_expr.arms[1]
    assert arm1.span.line == arm1.pattern.span.line
    assert arm1.span.col == arm1.pattern.span.col
    arm2 = match_expr.arms[2]
    assert arm2.span == arm2.pattern.span


def test_range_pattern_parses():
    """Pattern `0..10 => ...` parses as PatRange (exclusive)."""
    src = """
    fn f(x: i32) -> i32 {
        match x {
            0..10 => 1,
            10..=20 => 2,
            _ => 0,
        }
    }
    """
    prog = parse(src)
    fn = prog.items[0]
    assert isinstance(fn, ast.FnDecl)
    match_expr = fn.body.final_expr
    assert isinstance(match_expr, ast.Match)
    arm0 = match_expr.arms[0]
    assert isinstance(arm0.pattern, ast.PatRange)
    assert arm0.pattern.inclusive is False
    arm1 = match_expr.arms[1]
    assert isinstance(arm1.pattern, ast.PatRange)
    assert arm1.pattern.inclusive is True


def test_or_pattern_parses():
    """Pattern `1 | 2 | 3 => ...` parses as PatOr with three alternatives."""
    src = """
    fn f(x: i32) -> i32 {
        match x {
            1 | 2 | 3 => 42,
            _ => 0,
        }
    }
    """
    prog = parse(src)
    fn = prog.items[0]
    assert isinstance(fn, ast.FnDecl)
    match_expr = fn.body.final_expr
    assert isinstance(match_expr, ast.Match)
    arm0 = match_expr.arms[0]
    assert isinstance(arm0.pattern, ast.PatOr), \
        f"expected PatOr, got {type(arm0.pattern).__name__}"
    assert len(arm0.pattern.alts) == 3
    # All three alts are literal patterns
    for alt in arm0.pattern.alts:
        assert isinstance(alt, ast.PatLit)


# ============================================================================
# Stage 16.5 — FFI / extern "C"
# ============================================================================
def test_extern_c_fn_decl_parses():
    """`extern "C" fn puts(s: *const u8) -> i32;` parses to FnDecl with
    is_extern=True, no body, *const u8 param type."""
    src = 'extern "C" fn puts(s: *const u8) -> i32;'
    prog = parse(src)
    assert len(prog.items) == 1
    fn = prog.items[0]
    assert isinstance(fn, ast.FnDecl)
    assert fn.name == "puts"
    assert fn.is_extern is True
    assert fn.extern_abi == "C"
    # body is an empty placeholder Block
    assert isinstance(fn.body, ast.Block)
    assert fn.body.stmts == []
    assert fn.body.final_expr is None
    # param type is *const u8 (TyPtr)
    assert len(fn.params) == 1
    assert isinstance(fn.params[0].ty, ast.TyPtr)
    assert fn.params[0].ty.is_mut is False
    inner = fn.params[0].ty.inner
    assert isinstance(inner, ast.TyName) and inner.name == "u8"


def test_extern_c_only_c_abi_supported():
    """`extern "rust"` is rejected — only "C" is supported in Phase-0."""
    src = 'extern "rust" fn foo() -> i32;'
    try:
        parse(src)
        assert False, "expected ParseError"
    except Exception as e:
        assert "extern \"C\"" in str(e) or "supported" in str(e)


def test_ptr_mut_type_parses():
    """`fn write(buf: *mut u8) -> i32 { 0 }` parses with TyPtr(is_mut=True)."""
    src = 'fn write(buf: *mut u8) -> i32 { 0 }'
    prog = parse(src)
    fn = prog.items[0]
    assert isinstance(fn, ast.FnDecl)
    pty = fn.params[0].ty
    assert isinstance(pty, ast.TyPtr)
    assert pty.is_mut is True


# ============================================================================
# Stdlib merge (Audit 28.8 A8)
# ============================================================================
def test_stdlib_merges_fn_decls():
    """Baseline: include_stdlib=True still merges FnDecls (e.g. vec_new)."""
    prog = parse("fn main() -> i32 { 42 }\n", include_stdlib=True)
    fn_names = {it.name for it in prog.items
                if isinstance(it, ast.FnDecl)}
    # vec_new is a known transcendental in stdlib/vec.hx.
    assert "vec_new" in fn_names, \
        "stdlib FnDecls should still be merged when include_stdlib=True"


def test_stdlib_parse_is_cached():
    """The parsed stdlib is cached process-wide: after a
    `parse(include_stdlib=True)`, `_STDLIB_PARSE_CACHE` holds a pickle
    blob per stdlib file, so later calls skip the redundant re-parse."""
    import helixc.frontend.parser as _p
    _p._STDLIB_PARSE_CACHE.clear()
    parse("fn main() -> i32 { 0 }\n", include_stdlib=True)
    assert _p._STDLIB_PARSE_CACHE, "stdlib parse should be cached"
    assert all(isinstance(v, bytes)
               for v in _p._STDLIB_PARSE_CACHE.values()), \
        "cache values are pickle blobs"


def test_stdlib_cache_yields_independent_asts():
    """The cache must hand each compilation its OWN stdlib AST — two
    `parse(include_stdlib=True)` calls share no AST objects, so a
    downstream pass mutating one program's stdlib nodes cannot corrupt
    another's."""
    p1 = parse("fn main() -> i32 { 0 }\n", include_stdlib=True)
    p2 = parse("fn main() -> i32 { 0 }\n", include_stdlib=True)

    def stdlib_fns(prog):
        return {it.name: it for it in prog.items
                if isinstance(it, ast.FnDecl)
                and "__stdlib" in getattr(it, "attrs", [])}

    f1, f2 = stdlib_fns(p1), stdlib_fns(p2)
    shared = sorted(set(f1) & set(f2))
    assert shared, "expected stdlib fns merged into both programs"
    for name in shared:
        assert f1[name] is not f2[name], \
            f"cache leaked a shared AST object for {name!r}"
    # Mutating one program's stdlib node must not touch the other's.
    f1[shared[0]].attrs.append("__isolation_probe__")
    assert "__isolation_probe__" not in f2[shared[0]].attrs, \
        "cache leaked mutable state across compilations"


def test_stdlib_cached_parse_matches_fresh():
    """A cache hit yields the same merged stdlib as a cold re-parse —
    same item kinds and names — so caching changes performance only,
    never behaviour."""
    import helixc.frontend.parser as _p

    def item_signature(prog):
        return sorted((type(it).__name__, getattr(it, "name", None))
                      for it in prog.items)

    _p._STDLIB_PARSE_CACHE.clear()
    fresh = parse("fn main() -> i32 { 0 }\n", include_stdlib=True)
    cached = parse("fn main() -> i32 { 0 }\n", include_stdlib=True)
    assert item_signature(fresh) == item_signature(cached), \
        "a cached stdlib parse must match a cold re-parse"


def test_stdlib_merges_struct_decls(tmp_path, monkeypatch):
    """Audit 28.8 A8: StructDecl items in stdlib were silently dropped.
    Simulate a stdlib drop-in that exports a struct + an impl block,
    and verify both survive the merge."""
    import os as _os
    import shutil
    import helixc.frontend.parser as _p

    # Build a fake stdlib in tmp_path.
    fake_stdlib_dir = tmp_path / "stdlib"
    fake_stdlib_dir.mkdir()
    fake_file = fake_stdlib_dir / "fake_pair.hx"
    fake_file.write_text(
        "struct Pair { a: i32, b: i32 }\n"
        "fn pair_new(a: i32, b: i32) -> i32 { 0 }\n"
        "const PAIR_MAX: i32 = 100;\n"
    )

    # Monkey-patch parser to use a tiny stdlib list pointing at our fake.
    monkeypatch.setattr(_p, "STDLIB_FILES", ["fake_pair.hx"])
    # Also patch the path resolution by temporarily relocating the
    # stdlib directory via _merge_stdlib's relative computation —
    # since _merge_stdlib derives stdlib_dir from __file__, we can't
    # easily redirect. Instead, write the fake file into the real
    # stdlib dir and clean up after.
    real_stdlib = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(_p.__file__))),
        "stdlib",
    )
    real_fake = _os.path.join(real_stdlib, "fake_pair.hx")
    shutil.move(str(fake_file), real_fake)
    try:
        prog = _p.parse("fn main() -> i32 { 0 }\n", include_stdlib=True)
    finally:
        _os.remove(real_fake)

    struct_names = {it.name for it in prog.items
                    if isinstance(it, ast.StructDecl)}
    fn_names = {it.name for it in prog.items
                if isinstance(it, ast.FnDecl)}
    const_names = {it.name for it in prog.items
                   if isinstance(it, ast.ConstDecl)}
    assert "Pair" in struct_names, \
        "Audit 28.8 A8: StructDecl items must be merged from stdlib"
    assert "pair_new" in fn_names
    assert "PAIR_MAX" in const_names, \
        "Audit 28.8 A8: ConstDecl items must be merged from stdlib"


def test_stdlib_user_wins_on_conflict(tmp_path, monkeypatch):
    """When the user defines a fn with the same name as a stdlib fn,
    the user version takes precedence (and the stdlib copy is skipped
    silently — same as pre-A8 behaviour for FnDecls, now applied
    uniformly across kinds)."""
    import os as _os
    import shutil
    import helixc.frontend.parser as _p

    fake_file = tmp_path / "fake_dup.hx"
    fake_file.write_text("fn my_helper() -> i32 { 1 }\n")
    real_stdlib = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(_p.__file__))),
        "stdlib",
    )
    real_fake = _os.path.join(real_stdlib, "fake_dup.hx")
    shutil.move(str(fake_file), real_fake)
    monkeypatch.setattr(_p, "STDLIB_FILES", ["fake_dup.hx"])
    try:
        prog = _p.parse(
            "fn my_helper() -> i32 { 999 }\nfn main() -> i32 { my_helper() }\n",
            include_stdlib=True,
        )
    finally:
        _os.remove(real_fake)

    helpers = [it for it in prog.items
               if isinstance(it, ast.FnDecl) and it.name == "my_helper"]
    assert len(helpers) == 1, \
        f"expected exactly one my_helper after merge, got {len(helpers)}"
    # User-defined version wins (its body returns 999, not 1).


def test_stdlib_user_type_name_wins_over_alias_conflict():
    """User nominal types must block same-named stdlib type aliases.

    Without this, `struct Probability` could coexist with the default
    `type Probability = ...`, and type resolution would silently choose
    the alias first.
    """
    prog = parse(
        "struct Probability { v: i32 }\n"
        "enum Confidence { Exact }\n"
        "fn main() -> i32 { 0 }\n",
        include_stdlib=True,
    )
    probability_items = [
        it for it in prog.items
        if getattr(it, "name", None) == "Probability"
        and isinstance(it, (ast.TypeAlias, ast.StructDecl, ast.EnumDecl))
    ]
    confidence_items = [
        it for it in prog.items
        if getattr(it, "name", None) == "Confidence"
        and isinstance(it, (ast.TypeAlias, ast.StructDecl, ast.EnumDecl))
    ]
    assert len(probability_items) == 1
    assert isinstance(probability_items[0], ast.StructDecl)
    assert len(confidence_items) == 1
    assert isinstance(confidence_items[0], ast.EnumDecl)


def test_stdlib_missing_file_strict_env_raises(monkeypatch):
    """Audit 28.8 A8: with HELIXC_STDLIB_STRICT=1, a missing stdlib
    file raises FileNotFoundError instead of being silently skipped."""
    import helixc.frontend.parser as _p

    monkeypatch.setattr(_p, "STDLIB_FILES", ["does_not_exist_anywhere.hx"])
    monkeypatch.setenv(_p.STDLIB_STRICT_ENV, "1")
    try:
        _p.parse("fn main() -> i32 { 0 }\n", include_stdlib=True)
    except FileNotFoundError as e:
        assert "does_not_exist_anywhere.hx" in str(e)
        return
    raise AssertionError(
        "expected FileNotFoundError with HELIXC_STDLIB_STRICT=1 and "
        "a missing stdlib file"
    )


def test_stdlib_missing_file_default_lenient(monkeypatch, capsys):
    """Default (env unset): missing stdlib files emit a warning but
    don't raise — backward-compat."""
    import helixc.frontend.parser as _p

    monkeypatch.setattr(_p, "STDLIB_FILES", ["does_not_exist_anywhere.hx"])
    monkeypatch.delenv(_p.STDLIB_STRICT_ENV, raising=False)
    prog = _p.parse("fn main() -> i32 { 0 }\n", include_stdlib=True)
    cap = capsys.readouterr()
    assert "stdlib file missing" in cap.err
    # main still parsed.
    assert any(isinstance(it, ast.FnDecl) and it.name == "main"
               for it in prog.items)


# ============================================================================
# Test runner
# ============================================================================
def main():
    # Tests requiring pytest fixtures (tmp_path / monkeypatch / capsys /
    # etc.) are skipped here — the manual runner can't synthesize
    # fixtures. They're still discovered by `pytest helixc/tests/test_parser.py`
    # which DOES wire fixtures, so coverage is preserved.
    import inspect
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    passed = 0
    failed = 0
    skipped = 0
    for name, fn in tests:
        try:
            sig = inspect.signature(fn)
            required = [p for p in sig.parameters.values()
                        if p.default is inspect.Parameter.empty
                        and p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                       inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        except (TypeError, ValueError):
            required = []
        if required:
            print(f"SKIP {name}: needs pytest fixtures {[p.name for p in required]}")
            skipped += 1
            continue
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    summary = f"{passed} passed, {failed} failed"
    if skipped:
        summary += f", {skipped} skipped"
    print(f"\n{summary}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
