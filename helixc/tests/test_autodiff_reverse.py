"""Unit tests for the reverse-mode AD engine.

These tests check the symbolic shape of the gradient — not numerics. Numerics
are exercised end-to-end via test_codegen tests for grad_rev(...)."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.frontend import ast_nodes as A
from helixc.frontend.autodiff import fmt
from helixc.frontend.autodiff_reverse import differentiate_reverse


def _body_of(src: str) -> A.Expr:
    """Parse a single-fn program and return the function body."""
    prog = parse(src)
    fn = next(item for item in prog.items if isinstance(item, A.FnDecl))
    return fn.body


def test_const_zero_gradient():
    body = _body_of("fn f(x: f32) -> f32 { 5.0 }")
    grads = differentiate_reverse(body, ["x"])
    assert fmt(grads["x"]) == "0", f"got {fmt(grads['x'])}"


def test_x_squared_grad_is_2x():
    body = _body_of("fn f(x: f32) -> f32 { x * x }")
    grads = differentiate_reverse(body, ["x"])
    # Reverse mode: adjoint left = 1*x = x; adjoint right = 1*x = x. Sum = x+x = 2x.
    # The simplifier may fold (1*x)+(1*x) -> x+x or (x+x).
    out = fmt(grads["x"])
    assert "x" in out and "+" in out, f"expected sum-of-x form, got {out}"


def test_linear_gradient_x():
    body = _body_of("fn f(x: f32, y: f32) -> f32 { 3.0 * x + 5.0 * y }")
    grads = differentiate_reverse(body, ["x", "y"])
    # ∂f/∂x = 3, ∂f/∂y = 5. Reverse mode produces 1*3 = 3 (after simplify).
    assert fmt(grads["x"]) == "3", f"got {fmt(grads['x'])}"
    assert fmt(grads["y"]) == "5", f"got {fmt(grads['y'])}"


def test_quadratic_two_vars():
    body = _body_of("fn f(x: f32, y: f32) -> f32 { x * x + y * y }")
    grads = differentiate_reverse(body, ["x", "y"])
    # ∂f/∂x = 2x; reverse mode emits x+x.
    out_x = fmt(grads["x"])
    out_y = fmt(grads["y"])
    assert "x" in out_x and "+" in out_x
    assert "y" in out_y and "+" in out_y


def test_subtraction():
    body = _body_of("fn f(x: f32, y: f32) -> f32 { x - y }")
    grads = differentiate_reverse(body, ["x", "y"])
    # ∂f/∂x = 1, ∂f/∂y = -1.
    assert fmt(grads["x"]) == "1", f"got {fmt(grads['x'])}"
    # -1 may render as "-1" or "(-1)"
    out_y = fmt(grads["y"])
    assert "1" in out_y and "-" in out_y, f"got {out_y}"


def test_division_quotient_rule():
    body = _body_of("fn f(x: f32, y: f32) -> f32 { x / y }")
    grads = differentiate_reverse(body, ["x", "y"])
    # ∂f/∂x = 1/y. After simplify: (1/y).
    out_x = fmt(grads["x"])
    assert "/" in out_x and "y" in out_x
    # ∂f/∂y = -x/(y*y).
    out_y = fmt(grads["y"])
    assert "-" in out_y, f"expected negative, got {out_y}"


def test_chain_via_letbinding():
    # f = (x+1)*(x+2); ∂f/∂x = (x+2) + (x+1) = 2x+3
    body = _body_of("""
    fn f(x: f32) -> f32 {
        let a = x + 1.0;
        let b = x + 2.0;
        a * b
    }
    """)
    grads = differentiate_reverse(body, ["x"])
    # After inlining: (x+1)*(x+2). Reverse: adj_l = 1*(x+2), adj_r = 1*(x+1).
    # x appears twice, so summed contributions = (x+2) + (x+1) symbolically.
    out = fmt(grads["x"])
    # The expression should reference x (positive count).
    assert out.count("x") >= 2, f"expected x referenced multiple times, got {out}"


def test_unary_negation():
    body = _body_of("fn f(x: f32) -> f32 { -x }")
    grads = differentiate_reverse(body, ["x"])
    # ∂(-x)/∂x = -1.
    out = fmt(grads["x"])
    assert "-1" in out or out == "(-1)", f"got {out}"


def test_param_not_in_expr():
    # If a parameter is not used in the expression, its gradient is 0.
    body = _body_of("fn f(x: f32, y: f32) -> f32 { x * x }")
    grads = differentiate_reverse(body, ["x", "y"])
    assert fmt(grads["y"]) == "0", f"got {fmt(grads['y'])}"


def test_multi_use_of_param_sums():
    # f = x + x + x; ∂f/∂x = 3. Reverse mode: each + propagates 1 to its
    # operands, so x's bucket is [1, 1, 1]; sum = 3.
    body = _body_of("fn f(x: f32) -> f32 { x + x + x }")
    grads = differentiate_reverse(body, ["x"])
    out = fmt(grads["x"])
    # After simplification 1+1+1 should fold to 3.
    assert out == "3", f"got {out}"


def test_match_bool_propagates_per_arm():
    # ∂(match true { true => 2x, false => x+1 })/∂x = match { true => 2, false => 1 }
    body = _body_of("""
    fn f(x: f32) -> f32 {
        match true {
            true => x * 2.0,
            false => x + 1.0
        }
    }
    """)
    grads = differentiate_reverse(body, ["x"])
    out = fmt(grads["x"])
    assert "match" in out, f"expected match in gradient, got {out}"
    assert "2" in out and "1" in out, f"expected per-arm derivatives, got {out}"


def test_match_int_with_wildcard():
    body = _body_of("""
    fn f(x: f32, k: i32) -> f32 {
        match k {
            0 => x * 3.0,
            1 => x * 5.0,
            _ => 0.0
        }
    }
    """)
    grads = differentiate_reverse(body, ["x"])
    out = fmt(grads["x"])
    assert "match" in out, f"expected match wrapper, got {out}"
    assert "3" in out and "5" in out, f"expected per-arm derivatives 3 and 5, got {out}"


def test_match_with_pattern_shadow_is_zero():
    # PatBind 'x' shadows the parameter inside the arm; the body's 'x'
    # refers to the bound name, not the param, so gradient is 0.
    body = _body_of("""
    fn f(x: f32) -> f32 {
        match 1.0 {
            x => x,
            _ => 0.0
        }
    }
    """)
    grads = differentiate_reverse(body, ["x"])
    out = fmt(grads["x"])
    assert out == "0", f"shadowed param should give 0, got {out}"


def test_match_no_param_use():
    # Match where no arm references the param → gradient is 0.
    body = _body_of("""
    fn f(x: f32) -> f32 {
        match true {
            true => 5.0,
            false => 7.0
        }
    }
    """)
    grads = differentiate_reverse(body, ["x"])
    out = fmt(grads["x"])
    assert out == "0", f"no param use should give 0, got {out}"


def test_match_chain_rule():
    # f = (match true { true => 2x, false => 3x })^2
    # Per arm: d(let_body^2)/dx = 2*let_body * d(let_body)/dx
    # arm 1: 2*(2x)*2 = 8x; arm 2: 2*(3x)*3 = 18x. We just check the
    # gradient AST contains a match wrapper and references x.
    body = _body_of("""
    fn f(x: f32) -> f32 {
        let y = match true {
            true => x * 2.0,
            false => x * 3.0
        };
        y * y
    }
    """)
    grads = differentiate_reverse(body, ["x"])
    out = fmt(grads["x"])
    assert "match" in out and "x" in out, f"got {out}"


def test_two_param_if_zero_literals_distinct_objects():
    """Cycle-4 audit: the FloatLit(0.0) used as a placeholder when one
    arm has no contribution must be a distinct Python object per
    parameter — sharing would let in-place mutation passes corrupt
    cross-parameter gradient ASTs."""
    body = _body_of("""
    fn f(x: f32, y: f32, c: bool) -> f32 {
        if c { x } else { y }
    }
    """)
    grads = differentiate_reverse(body, ["x", "y"])

    def collect_zero_ids(node, results):
        if node is None:
            return
        if isinstance(node, A.FloatLit) and node.value == 0.0:
            results.append(id(node))
        for attr in ("cond", "then", "else_", "final_expr",
                     "left", "right", "operand", "value"):
            if hasattr(node, attr):
                collect_zero_ids(getattr(node, attr), results)
        for attr in ("stmts", "args", "elems"):
            if hasattr(node, attr):
                for c in getattr(node, attr) or []:
                    collect_zero_ids(c, results)

    zx, zy = [], []
    collect_zero_ids(grads["x"], zx)
    collect_zero_ids(grads["y"], zy)
    shared = set(zx) & set(zy)
    assert not shared, f"x and y grads share zero literal objects: {shared}"


# ----------------------------------------------------------------------
# Stage 35 — reverse-mode model-field leaves.
# ----------------------------------------------------------------------
def _field_path(node: A.Expr) -> str | None:
    if isinstance(node, A.Name):
        return node.name
    if isinstance(node, A.Field):
        base = _field_path(node.obj)
        if base is None:
            return None
        return f"{base}.{node.name}"
    return None


def _collect_field_paths(node: A.Expr | None) -> set[str]:
    out: set[str] = set()

    def walk(expr: A.Expr | None) -> None:
        if expr is None:
            return
        path = _field_path(expr)
        if isinstance(expr, A.Field) and path is not None:
            out.add(path)
        for attr in ("cond", "then", "else_", "final_expr", "left", "right",
                     "operand", "value", "callee", "scrutinee"):
            if hasattr(expr, attr):
                walk(getattr(expr, attr))
        for attr in ("stmts", "args", "elems", "indices"):
            if hasattr(expr, attr):
                for child in getattr(expr, attr) or []:
                    walk(child)
        if hasattr(expr, "arms"):
            for arm in getattr(expr, "arms") or []:
                walk(arm.body)

    walk(node)
    return out


def test_stage35_reverse_ad_accumulates_model_field_leaves():
    body = _body_of("""
    struct Model { w1: f32, w2: f32 }
    fn loss(m: Model, x: f32) -> f32 {
        m.w1 * x + m.w2 * x
    }
    """)
    grads = differentiate_reverse(body, ["m.w1", "m.w2", "x"])

    assert fmt(grads["m.w1"]) == "x", f"got {fmt(grads['m.w1'])}"
    assert fmt(grads["m.w2"]) == "x", f"got {fmt(grads['m.w2'])}"
    x_fields = _collect_field_paths(grads["x"])
    assert {"m.w1", "m.w2"} <= x_fields, (
        f"x gradient should keep both model-field coefficients, got {x_fields}"
    )


def test_stage35_reverse_ad_accumulates_nested_model_field_leaf():
    body = _body_of("""
    struct Layer { w: f32 }
    struct Model { layer: Layer }
    fn loss(m: Model, x: f32) -> f32 {
        m.layer.w * x
    }
    """)
    grads = differentiate_reverse(body, ["m.layer.w", "x"])

    assert fmt(grads["m.layer.w"]) == "x", (
        f"got {fmt(grads['m.layer.w'])}"
    )
    assert "m.layer.w" in _collect_field_paths(grads["x"])


def test_stage35_reverse_ad_treats_non_target_field_as_coefficient():
    from helixc.frontend import autodiff
    autodiff.take_diff_warnings()
    body = _body_of("""
    struct Model { w: f32 }
    fn loss(m: Model, x: f32) -> f32 {
        m.w * x
    }
    """)
    grads = differentiate_reverse(body, ["x"])
    warnings = autodiff.take_diff_warnings()

    assert fmt(grads["x"]) != "0", "x gradient should keep m.w coefficient"
    assert "m.w" in _collect_field_paths(grads["x"])
    assert not any("unhandled expression kind" in w for w in warnings), (
        f"field coefficient should not produce an unhandled-node warning: "
        f"{warnings}"
    )


# ----------------------------------------------------------------------
# Audit 28.8 cycle 2 C2-3 — reverse-mode emits warnings for unhandled
# Unary / Binary ops (pre-fix it silently zeroed gradient contribution).
# ----------------------------------------------------------------------
def test_c2_3_reverse_binary_modulo_warns():
    """C2-3: `x % 2` in a reverse-mode AD'd expression must emit an
    85001 warning. Pre-fix the Binary arm fell through to a silent
    return for any op outside `{+, -, *, /}`."""
    from helixc.frontend import autodiff
    autodiff.take_diff_warnings()  # drain
    body = _body_of("fn f(x: i32) -> i32 { x % 2 }")
    differentiate_reverse(body, ["x"])
    warnings = autodiff.take_diff_warnings()
    assert any("85001" in w and "'%'" in w for w in warnings), (
        f"expected AD warning for binary `%`, got: {warnings}"
    )


def test_c2_3_reverse_binary_bitwise_warns():
    """C2-3: `x & 1` (bitwise) inside reverse-mode AD must warn —
    bitwise ops have no defined local derivative."""
    from helixc.frontend import autodiff
    autodiff.take_diff_warnings()
    body = _body_of("fn f(x: i32) -> i32 { x & 1 }")
    differentiate_reverse(body, ["x"])
    warnings = autodiff.take_diff_warnings()
    assert any("85001" in w for w in warnings), (
        f"expected AD warning for bitwise `&`, got: {warnings}"
    )


def test_c2_3_reverse_unary_not_warns():
    """C2-3: `!flag` (boolean NOT) inside reverse-mode AD must warn."""
    from helixc.frontend import autodiff
    autodiff.take_diff_warnings()
    body = _body_of("fn f(b: bool) -> bool { !b }")
    differentiate_reverse(body, ["b"])
    warnings = autodiff.take_diff_warnings()
    assert any("85001" in w and "'!'" in w for w in warnings), (
        f"expected AD warning for unary `!`, got: {warnings}"
    )


def test_c2_3_reverse_arithmetic_no_warn():
    """C2-3 inverse: `+ - * /` and unary `-` must NOT spuriously warn —
    they have correct derivatives."""
    from helixc.frontend import autodiff
    autodiff.take_diff_warnings()
    body = _body_of("fn f(x: f32) -> f32 { -((x + 1.0) * x) / 2.0 }")
    differentiate_reverse(body, ["x"])
    warnings = autodiff.take_diff_warnings()
    assert not any("85001" in w for w in warnings), (
        f"unexpected AD warning on arithmetic ops: {warnings}"
    )


def test_c3_5_inline_lets_recurses_through_cast():
    """Audit 28.8 cycle 3 C3-5: `_inline_lets` must recurse through
    A.Cast (and Call/Field/Index/Match/...) so let-bound names under
    those positions get substituted. Pre-fix the cycle-2 C2-3 `%` warn
    fired only when `%` was at top level; the dominant idiom

        let r = (x as i64) % 2_i64;
        r as f64

    silently skipped the warn because `_inline_lets` didn't recurse
    into the outer Cast around Name('r'). Now the Name resolves to
    the Binary, the Binary is visited, and the `%` warn fires."""
    from helixc.frontend import autodiff
    autodiff.take_diff_warnings()
    body = _body_of(
        "fn f(x: f32) -> f32 { "
        "    let r = (x as i64) % 2_i64; "
        "    r as f32 "
        "}"
    )
    differentiate_reverse(body, ["x"])
    warnings = autodiff.take_diff_warnings()
    assert any("85001" in w and "'%'" in w for w in warnings), (
        f"expected `%` warn to fire after Cast-around-Name "
        f"substitution; got warnings: {warnings}"
    )


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
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
