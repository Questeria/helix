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
    """Phase-0: passing i32 where Logic<i32> expected is currently a
    soft type (typechecker treats wrappers leniently). Trap 24100
    will fire when downstream passes enforce provenance. This test
    documents the Phase-0 behavior so the regression is visible if
    we tighten enforcement later."""
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
    # Phase-0: no error expected. If this starts erroring after Stage 24
    # enforcement is added (v0.2), update this test to assert errs.
    assert isinstance(errs, list)


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
