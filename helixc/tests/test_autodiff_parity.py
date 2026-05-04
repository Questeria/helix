"""Property test: forward-mode and reverse-mode AD must agree.

Generates a corpus of small symbolic functions, computes the gradient
both ways, and asserts the simplified forms agree on a battery of
concrete inputs. Catches drift between the two engines as either grows
new ops or chain rules.
"""

from __future__ import annotations
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.frontend import ast_nodes as A
from helixc.frontend.autodiff import differentiate
from helixc.frontend.autodiff_reverse import differentiate_reverse


# Small evaluator over the AST subset both engines emit. Returns float.
def _eval(expr: A.Expr, env: dict[str, float]) -> float:
    if isinstance(expr, A.IntLit):
        return float(expr.value)
    if isinstance(expr, A.FloatLit):
        return float(expr.value)
    if isinstance(expr, A.Name):
        return env[expr.name]
    if isinstance(expr, A.Call) and isinstance(expr.callee, A.Name):
        # Built-in transcendentals — evaluate using Python's math.
        name = expr.callee.name
        args = [_eval(a, env) for a in expr.args]
        if len(args) == 1:
            x = args[0]
            if name == "__exp": return math.exp(x)
            if name == "__log": return math.log(x)
            if name == "__sin": return math.sin(x)
            if name == "__cos": return math.cos(x)
            if name == "__sqrt": return math.sqrt(x)
            if name == "__tanh": return math.tanh(x)
            if name == "__softplus": return math.log(1.0 + math.exp(x))
            if name == "__sigmoid": return 1.0 / (1.0 + math.exp(-x))
            if name == "__silu": return x * (1.0 / (1.0 + math.exp(-x)))
            if name == "__relu": return max(0.0, x)
            if name == "__abs": return abs(x)
        if len(args) == 2 and name == "__powi":
            x, n = args
            n_i = int(n)
            if n_i <= 0:
                return 1.0
            cap = min(n_i, 16)
            return x ** cap
        # Unknown call — return 0 (the AD engine treats it as opaque).
        return 0.0
    if isinstance(expr, A.Unary) and expr.op == "-":
        return -_eval(expr.operand, env)
    if isinstance(expr, A.Binary):
        l = _eval(expr.left, env)
        r = _eval(expr.right, env)
        if expr.op == "+": return l + r
        if expr.op == "-": return l - r
        if expr.op == "*": return l * r
        if expr.op == "/":
            if r == 0:
                return float("nan")
            return l / r
        if expr.op in (">", "<", ">=", "<=", "==", "!="):
            cmp_map = {
                ">": l > r, "<": l < r, ">=": l >= r, "<=": l <= r,
                "==": l == r, "!=": l != r,
            }
            return 1.0 if cmp_map[expr.op] else 0.0
    if isinstance(expr, A.Block):
        if expr.final_expr is None:
            return 0.0
        return _eval(expr.final_expr, env)
    if isinstance(expr, A.If):
        cond_v = _eval(expr.cond, env)
        if cond_v != 0:
            target = expr.then
        else:
            target = expr.else_ if expr.else_ is not None else None
        if target is None:
            return 0.0
        if isinstance(target, A.Block):
            return _eval(target, env)
        return _eval(target, env)
    raise NotImplementedError(f"eval: {type(expr).__name__}")


def _fn_body_from(src: str) -> A.Block:
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl))
    return fn.body


def _agree(src: str, var: str, points: list[float]):
    body = _fn_body_from(src)
    fwd = differentiate(body, var)
    rev = differentiate_reverse(body, [var])[var]
    for x in points:
        env = {var: x}
        a = _eval(fwd, env)
        b = _eval(rev, env)
        if math.isnan(a) and math.isnan(b):
            continue
        assert abs(a - b) < 1e-3, (
            f"src={src!r} at {var}={x}: forward={a}, reverse={b}"
        )


def test_parity_polynomials():
    cases = [
        "fn f(x: f32) -> f32 { x * x }",
        "fn f(x: f32) -> f32 { x * x + 3.0 * x + 1.0 }",
        "fn f(x: f32) -> f32 { (x - 1.0) * (x + 1.0) }",
        "fn f(x: f32) -> f32 { x - x }",
        "fn f(x: f32) -> f32 { x * x * x }",
    ]
    for src in cases:
        _agree(src, "x", [-2.0, -0.5, 0.0, 0.5, 1.0, 3.0])


def test_parity_division():
    cases = [
        "fn f(x: f32) -> f32 { 1.0 / x }",
        "fn f(x: f32) -> f32 { x / (x + 1.0) }",
    ]
    for src in cases:
        _agree(src, "x", [0.5, 1.0, 2.0, 5.0])


def test_parity_neg_and_subtraction():
    cases = [
        "fn f(x: f32) -> f32 { 0.0 - x }",
        "fn f(x: f32) -> f32 { 5.0 - x * x }",
        "fn f(x: f32) -> f32 { -(x * x) }",
    ]
    for src in cases:
        _agree(src, "x", [-2.0, -1.0, 0.0, 1.0, 2.0])


def test_parity_transcendentals():
    cases = [
        "fn f(x: f32) -> f32 { __exp(x) }",
        "fn f(x: f32) -> f32 { __sin(x) }",
        "fn f(x: f32) -> f32 { __cos(x) }",
        "fn f(x: f32) -> f32 { __sigmoid(x) }",
        "fn f(x: f32) -> f32 { __tanh(x) }",
        "fn f(x: f32) -> f32 { __softplus(x) }",
        "fn f(x: f32) -> f32 { __silu(x) }",
        "fn f(x: f32) -> f32 { __relu(x) }",
        "fn f(x: f32) -> f32 { __abs(x) }",
        "fn f(x: f32) -> f32 { x * __sigmoid(x) + __tanh(x) }",
    ]
    for src in cases:
        # Avoid x=0 for __abs / __relu (gradient is undefined there)
        _agree(src, "x", [-2.0, -0.5, 0.5, 1.5])


def test_pow_int_parity():
    """Forward + reverse mode AD on __powi(x, n) must agree at several inputs."""
    cases = [
        "fn f(x: f32) -> f32 { __powi(x, 2) }",
        "fn f(x: f32) -> f32 { __powi(x, 3) }",
        "fn f(x: f32) -> f32 { __powi(x, 4) }",
        "fn f(x: f32) -> f32 { __powi(x, 2) + __powi(x, 3) }",
    ]
    for src in cases:
        _agree(src, "x", [-2.0, -0.5, 0.5, 1.5, 3.0])


def test_pow_int_cap_consistency():
    """__powi runtime caps n at 16. The AD chain rule must match the cap
    so the gradient is consistent with what runtime returns. For n>16,
    forward and reverse should both produce the saturated derivative
    (16 * x^15) — and they must agree with each other."""
    cases = [
        "fn f(x: f32) -> f32 { __powi(x, 17) }",
        "fn f(x: f32) -> f32 { __powi(x, 20) }",
        "fn f(x: f32) -> f32 { __powi(x, 100) }",
    ]
    for src in cases:
        _agree(src, "x", [1.0, 1.5, 2.0])


def test_parity_multi_variable():
    body = _fn_body_from("fn f(x: f32, y: f32) -> f32 { x*x + 2.0*x*y + y*y }")
    # Check both partials.
    for var in ("x", "y"):
        fwd = differentiate(body, var)
        rev = differentiate_reverse(body, [var])[var]
        for x in (1.0, 2.0):
            for y in (-1.0, 0.5, 3.0):
                env = {"x": x, "y": y}
                a = _eval(fwd, env)
                b = _eval(rev, env)
                assert abs(a - b) < 1e-3, \
                    f"d/d{var} mismatch at ({x}, {y}): fwd={a}, rev={b}"


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
