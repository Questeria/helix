"""Stage 24: tests for provenance-typed neuro-symbolic types.

D<Logic<T>> — differentiable relational/logical wrapper. Phase-0
scope: parse + type-level representation. Real fuzzy/AD semantics are
deferred to v0.2. Trap 24100 reserved (Audit 28.8 A4 reassigned from
24001, which kovc.hx already uses for AST_MOD bf16).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.frontend.typecheck import (
    typecheck,
    TyLogic,
    TyDiff,
    TyPrim,
    TyStruct,
    TypeChecker,
)
from helixc.frontend import ast_nodes as A


def test_parse_logic_type_in_signature():
    """Logic<T> appears as a parameter type."""
    src = """
struct Person { age: i32, ht: f64 }
fn lift(p: Person) -> Logic<Person> { p as Logic<Person> }
"""
    # Even if `as Logic<Person>` isn't a real coercion, parsing must
    # succeed — we test that the parser accepts `Logic<...>` in type
    # position.
    prog = parse(src.replace(" as Logic<Person>", ""))
    # Inject a stub fn that just returns a Person (parser only test).
    fns = [it for it in prog.items if isinstance(it, A.FnDecl)]
    assert any(f.name == "lift" for f in fns)


def test_parse_d_logic_t_in_param():
    """D<Logic<T>> nests correctly."""
    src = """
struct Person { age: i32 }
fn likely_parent(a: D<Logic<Person>>, b: D<Logic<Person>>) -> D<Logic<bool>> {
    a
}
"""
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl)
              and it.name == "likely_parent")
    assert len(fn.params) == 2


def test_typecheck_resolves_logic_t():
    """Logic<i32> param resolves to TyLogic(TyPrim('i32'))."""
    src = """
fn idl(x: Logic<i32>) -> Logic<i32> { x }
"""
    prog = parse(src)
    errs = typecheck(prog)
    assert errs == [], f"unexpected errors: {[str(e) for e in errs]}"
    tc = TypeChecker(prog)
    tc.check()
    sig = tc.functions["idl"]
    assert isinstance(sig.params[0][1], TyLogic)
    inner = sig.params[0][1].inner
    assert isinstance(inner, TyPrim) and inner.name == "i32"


def test_typecheck_resolves_d_logic_t():
    """D<Logic<i32>> resolves to TyDiff(TyLogic(TyPrim('i32')))."""
    src = """
fn id(x: D<Logic<i32>>) -> D<Logic<i32>> { x }
"""
    prog = parse(src)
    errs = typecheck(prog)
    assert errs == [], f"errors: {[str(e) for e in errs]}"
    tc = TypeChecker(prog)
    tc.check()
    sig = tc.functions["id"]
    pty = sig.params[0][1]
    assert isinstance(pty, TyDiff)
    assert isinstance(pty.inner, TyLogic)
    assert isinstance(pty.inner.inner, TyPrim)
    assert pty.inner.inner.name == "i32"


def test_typecheck_logic_struct_inner():
    """Logic<Person> with Person a user-declared struct."""
    src = """
struct Person { age: i32 }
fn id(p: Logic<Person>) -> Logic<Person> { p }
"""
    prog = parse(src)
    errs = typecheck(prog)
    assert errs == []
    tc = TypeChecker(prog)
    tc.check()
    sig = tc.functions["id"]
    pty = sig.params[0][1]
    assert isinstance(pty, TyLogic)
    assert isinstance(pty.inner, TyStruct)
    assert pty.inner.name == "Person"


def test_fmt_logic_type():
    """The pretty-printer renders TyLogic as `Logic<...>`."""
    tc = TypeChecker(parse("fn main() -> i32 { 1 }"))
    tc.check()
    t = TyLogic(inner=TyPrim("i32"))
    assert tc._fmt(t) == "Logic<i32>"
    nested = TyDiff(inner=TyLogic(inner=TyPrim("f64")))
    assert tc._fmt(nested) == "D<Logic<f64>>"


def test_fmt_logic_with_provenance():
    """Provenance tag shows as `Logic<T>@tag`."""
    tc = TypeChecker(parse("fn main() -> i32 { 1 }"))
    tc.check()
    t = TyLogic(inner=TyPrim("i32"), provenance="parent_of")
    assert tc._fmt(t) == "Logic<i32>@parent_of"


def test_logic_passthrough_arity_phase0():
    """Audit 28.8 B2: trap 24100 NOW fires when a non-Logic value is
    passed where a Logic-typed parameter is required. Previously this
    test documented "Phase-0: no error expected" — but that left
    silent provenance violations. The Tier-3 moat (D<Logic<T>>) is
    only meaningful if the boundary is actually checked."""
    src = """
fn lift(x: Logic<i32>) -> Logic<i32> { x }
fn user_main() -> i32 {
    let v: i32 = 5;
    let r = lift(v);
    0
}
"""
    prog = parse(src)
    errs = typecheck(prog)
    # Audit 28.8 B2: should emit a trap-24100 diagnostic.
    assert any("24100" in str(e) for e in errs), \
        f"expected trap 24100 diagnostic for non-Logic arg into Logic param, " \
        f"got: {[str(e) for e in errs]}"


def test_logic_passthrough_strip_provenance():
    """Audit 28.8 B2: also fires the other direction — passing
    Logic<T> where plain T is expected silently strips the provenance
    wrapper."""
    src = """
fn use_raw(x: i32) -> i32 { x }
fn user_main() -> i32 {
    let v: Logic<i32> = logic_atom(5);
    use_raw(v)
}
fn logic_atom(x: i32) -> Logic<i32> { x }
"""
    prog = parse(src)
    errs = typecheck(prog)
    assert any("24100" in str(e) for e in errs), \
        f"expected trap 24100 for Logic value into plain param, " \
        f"got: {[str(e) for e in errs]}"


def test_logic_binop_propagation():
    """Audit 28.8 B2: TyLogic propagates through binops. Previously
    `T + Logic<T>` returned T (left operand wins), silently dropping
    the Logic wrapper. Now the result is Logic<T> regardless of which
    side carries the wrapper."""
    from helixc.frontend.typecheck import TypeChecker, TyLogic, TyPrim
    from helixc.frontend import ast_nodes as A

    # Construct a synthetic Binary manually since the source-level path
    # would hit our new 24100 boundary check before we could observe
    # the binop alone.
    span = A.Span(line=1, col=1)
    left = TyLogic(inner=TyPrim("i32"))
    right = TyPrim("i32")

    # Build a tiny prog and run binop result manually via the same
    # propagation rule we just added.
    tc = TypeChecker(A.Program(module=None, items=[]))
    # Directly exercise the rule by simulating arg/result types through
    # _check_expr on a manually-built Binary AST.
    l_node = A.FloatLit(span=span, value=0.0)
    r_node = A.FloatLit(span=span, value=0.0)
    bin_node = A.Binary(span=span, op="+", left=l_node, right=r_node)
    # Monkey-patch _check_expr so child types are our chosen ones.
    orig = tc._check_expr

    def stubbed(e, sc):
        if e is l_node:
            return left
        if e is r_node:
            return right
        return orig(e, sc)
    tc._check_expr = stubbed
    from helixc.frontend.typecheck import Scope
    result = orig(bin_node, Scope())
    # Result should be TyLogic — wrapper preserved through the binop.
    assert isinstance(result, TyLogic), \
        f"expected TyLogic propagation, got {result!r}"
    assert isinstance(result.inner, TyPrim) and result.inner.name == "i32"


def test_logic_diff_binop_composes():
    """Audit 28.8 B2: `D<Logic<T>> + Logic<T>` composes to
    `D<Logic<T>>` (Diff wraps Logic, both wrappers preserved)."""
    from helixc.frontend.typecheck import (
        TypeChecker, TyLogic, TyDiff, TyPrim, Scope,
    )
    from helixc.frontend import ast_nodes as A

    span = A.Span(line=1, col=1)
    left = TyDiff(inner=TyLogic(inner=TyPrim("f64")))
    right = TyLogic(inner=TyPrim("f64"))

    tc = TypeChecker(A.Program(module=None, items=[]))
    l_node = A.FloatLit(span=span, value=0.0)
    r_node = A.FloatLit(span=span, value=0.0)
    bin_node = A.Binary(span=span, op="+", left=l_node, right=r_node)
    orig = tc._check_expr

    def stubbed(e, sc):
        if e is l_node:
            return left
        if e is r_node:
            return right
        return orig(e, sc)
    tc._check_expr = stubbed
    result = orig(bin_node, Scope())
    assert isinstance(result, TyDiff), f"expected TyDiff outer, got {result!r}"
    assert isinstance(result.inner, TyLogic), \
        f"expected TyLogic inner, got {result.inner!r}"


def test_trap_24100_reserved():
    """Document trap 24100 (provenance violation) is reserved.

    Audit 28.8 A4: the original reservation was 24001, but the bootstrap
    compiler kovc.hx:4220-4221 already emits 24001 for `bf16 % bf16`
    (per the bootstrap's `AST_tag * 1000 + sub_id` convention, where
    AST_MOD has tag 24 → trap id 24001). Two distinct conditions
    claiming the same id is a debugging black hole, so Stage 24's
    provenance reservation is moved to 24100. The bootstrap's 24001 for
    bf16 MOD remains as-is (it ships routinely).

    Phase-0 doesn't emit 24100 from the Python typechecker — semantics
    are deferred. The reservation is documented here so future stages
    (and any future audit cycle) can find the namespace claim.
    """
    RESERVED = 24100
    assert RESERVED == 24100


def test_trap_24001_belongs_to_ast_mod():
    """Audit 28.8 A4 follow-on: 24001 is the bootstrap-side bf16 MOD
    trap (kovc.hx:4220-4221). Documented here so a future cycle that
    re-checks the Stage-24 reservation doesn't accidentally double-
    claim 24001 again."""
    AST_MOD_TAG = 24
    AST_MOD_SUB_ID = 1
    assert AST_MOD_TAG * 1000 + AST_MOD_SUB_ID == 24001


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
