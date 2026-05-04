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


def test_fn_with_pub():
    p = parse("pub fn foo() {}")
    fn = p.items[0]
    assert fn.is_pub


def test_fn_with_where_clauses():
    p = parse("fn f[N: size, M: size]() where N % 16 == 0, M >= N, {}")
    fn = p.items[0]
    assert len(fn.where_clauses) == 2


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


# ============================================================================
# Test runner
# ============================================================================
def main():
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    passed = 0
    failed = 0
    for name, fn in tests:
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
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
