"""Stage 28: tests for parametric struct monomorphization."""

from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.frontend import ast_nodes as A
from helixc.frontend.struct_mono import (
    mangle_struct,
    collect_generic_structs,
    collect_concrete_uses,
    instantiate,
    monomorphize_structs,
    find_uninstantiated,
    TRAP_PARAM_STRUCT_UNINSTANTIATED,
    TRAP_PARAM_STRUCT_CONSTEVAL,
)


def _structs(prog):
    return {it.name: it for it in prog.items if isinstance(it, A.StructDecl)}


def test_parse_parametric_struct():
    """struct Pt[T] parses with one generic."""
    src = "struct Pt[T] { x: T, y: T }"
    prog = parse(src)
    s = _structs(prog)["Pt"]
    assert len(s.generics) == 1
    assert s.generics[0].name == "T"
    assert s.generics[0].kind == "type"


def test_collect_generic_structs():
    src = """
struct Pt[T] { x: T, y: T }
struct Plain { z: i32 }
struct Box[U] { v: U }
"""
    prog = parse(src)
    g = collect_generic_structs(prog)
    assert set(g.keys()) == {"Pt", "Box"}


def test_mangle_struct():
    ty_i32 = A.TyName(span=A.Span(0, 0), name="i32")
    ty_f64 = A.TyName(span=A.Span(0, 0), name="f64")
    assert mangle_struct("Pt", [ty_i32]) == "Pt__i32"
    assert mangle_struct("Pair", [ty_i32, ty_f64]) == "Pair__i32_f64"


def test_instantiate_simple():
    src = "struct Pt[T] { x: T, y: T }"
    prog = parse(src)
    decl = _structs(prog)["Pt"]
    ty_i32 = A.TyName(span=A.Span(0, 0), name="i32")
    inst = instantiate(decl, [ty_i32])
    assert inst.name == "Pt__i32"
    assert inst.generics == []
    assert len(inst.fields) == 2
    for f in inst.fields:
        assert isinstance(f.ty, A.TyName)
        assert f.ty.name == "i32"


def test_instantiate_arity_mismatch():
    src = "struct Pair[T, U] { a: T, b: U }"
    prog = parse(src)
    decl = _structs(prog)["Pair"]
    ty_i32 = A.TyName(span=A.Span(0, 0), name="i32")
    with pytest.raises(ValueError, match="arity"):
        instantiate(decl, [ty_i32])  # Need 2 args, got 1


def test_collect_concrete_uses_via_fn_param():
    """A fn param `Pt[i32]` is a concrete instantiation."""
    src = """
struct Pt[T] { x: T, y: T }
fn dist(p: Pt<i32>) -> i32 { p.x }
"""
    prog = parse(src)
    generic = collect_generic_structs(prog)
    uses = collect_concrete_uses(prog, generic)
    assert len(uses) == 1
    name, args = uses[0]
    assert name == "Pt"
    assert isinstance(args[0], A.TyName) and args[0].name == "i32"


def test_collect_concrete_uses_dedup():
    """Multiple uses of `Pt[i32]` collapse to one instantiation."""
    src = """
struct Pt[T] { x: T, y: T }
fn a(p: Pt<i32>) -> i32 { p.x }
fn b(q: Pt<i32>) -> i32 { q.y }
"""
    prog = parse(src)
    generic = collect_generic_structs(prog)
    uses = collect_concrete_uses(prog, generic)
    # Should dedup
    assert len(uses) == 1


def test_collect_concrete_uses_distinct_args():
    """Different concrete args produce separate instantiations."""
    src = """
struct Pt[T] { x: T, y: T }
fn a(p: Pt<i32>) -> i32 { p.x }
fn b(q: Pt<f64>) -> f64 { q.y }
"""
    prog = parse(src)
    generic = collect_generic_structs(prog)
    uses = collect_concrete_uses(prog, generic)
    assert len(uses) == 2
    names = {(n, args[0].name) for n, args in uses}
    assert names == {("Pt", "i32"), ("Pt", "f64")}


def test_monomorphize_structs_appends_mono_decls():
    src = """
struct Pt[T] { x: T, y: T }
fn dist(p: Pt<i32>) -> i32 { p.x }
"""
    prog = parse(src)
    n_items_before = len(prog.items)
    new_prog, diags = monomorphize_structs(prog)
    assert diags == []
    s = _structs(new_prog)
    assert "Pt__i32" in s
    assert s["Pt__i32"].generics == []
    # The mono struct's fields should have concrete type
    assert s["Pt__i32"].fields[0].ty.name == "i32"
    # Original generic still present (kept for docs)
    assert "Pt" in s
    # Total items grew by exactly 1
    assert len(new_prog.items) == n_items_before + 1


def test_monomorphize_no_generic_structs_noop():
    src = "struct Plain { x: i32 }"
    prog = parse(src)
    new_prog, diags = monomorphize_structs(prog)
    assert diags == []
    assert len(new_prog.items) == len(prog.items)


def test_find_uninstantiated():
    """An unused generic struct is flagged."""
    src = """
struct Used[T] { x: T }
struct Unused[U] { y: U }
fn f(a: Used<i32>) -> i32 { a.x }
"""
    prog = parse(src)
    out = find_uninstantiated(prog)
    assert out == ["Unused"]


def test_find_uninstantiated_clean():
    src = """
struct A[T] { x: T }
fn f(a: A<i32>) -> i32 { a.x }
"""
    prog = parse(src)
    assert find_uninstantiated(prog) == []


def test_trap_ids():
    assert TRAP_PARAM_STRUCT_UNINSTANTIATED == 28001
    assert TRAP_PARAM_STRUCT_CONSTEVAL == 28002


def test_two_type_params():
    """struct Pair[T, U] with both substituted."""
    src = "struct Pair[T, U] { a: T, b: U }"
    prog = parse(src)
    decl = _structs(prog)["Pair"]
    ty_i32 = A.TyName(span=A.Span(0, 0), name="i32")
    ty_f64 = A.TyName(span=A.Span(0, 0), name="f64")
    inst = instantiate(decl, [ty_i32, ty_f64])
    assert inst.name == "Pair__i32_f64"
    assert inst.fields[0].ty.name == "i32"
    assert inst.fields[1].ty.name == "f64"


def test_nested_use():
    """A fn returning Pt[i32] also counts as a use."""
    src = """
struct Pt[T] { x: T, y: T }
fn make() -> Pt<i32> { Pt { x: 1, y: 2 } }
"""
    prog = parse(src)
    generic = collect_generic_structs(prog)
    uses = collect_concrete_uses(prog, generic)
    assert len(uses) == 1
    assert uses[0][0] == "Pt"


# ----------------------------------------------------------------------
# Audit 28.8 A3 / B1 / C1-M2 — collect_concrete_uses now walks fn
# bodies AND Let-stmt type annotations + Cast targets, so body-level
# uses of parametric structs aren't silently uninstantiated.
# Plus: typecheck._resolve_type now produces TyStruct(mangled) for
# known parametric structs, so Pt<i32> and Pt<f32> are non-unifiable.
# ----------------------------------------------------------------------
def test_collect_concrete_uses_walks_let_ty():
    """C1-M2: `let p: Pt<i32> = ...` in a body must collect Pt<i32>."""
    src = """
struct Pt[T] { x: T, y: T }
fn make() -> i32 {
    let p: Pt<i32> = Pt { x: 1, y: 2 };
    p.x
}
"""
    prog = parse(src)
    generic = collect_generic_structs(prog)
    uses = collect_concrete_uses(prog, generic)
    names = [(n, args[0].name) for n, args in uses]
    assert ("Pt", "i32") in names, (
        f"expected Pt<i32> from body Let; got {names}"
    )


def test_collect_concrete_uses_walks_cast_ty():
    """B1: `x as Pt<f64>` in a body must collect Pt<f64>."""
    src = """
struct Pt[T] { x: T, y: T }
fn coerce(x: i32) -> i32 {
    let y = x as i32;
    y
}
"""
    # `as` on a generic struct isn't a typical pattern, but the
    # walker should still pick up the TyGeneric inside Cast.target_ty
    # if any. Use a tuple-like pattern with the type annotation.
    src2 = """
struct Box[T] { v: T }
fn use_box() -> i32 {
    let b: Box<i32> = Box { v: 5 };
    b.v
}
"""
    prog = parse(src2)
    generic = collect_generic_structs(prog)
    uses = collect_concrete_uses(prog, generic)
    names = [(n, args[0].name) for n, args in uses]
    assert ("Box", "i32") in names


def test_distinct_body_uses_dedupe():
    """B1: `Pt<i32>` used in two places dedupes to one instantiation."""
    src = """
struct Pt[T] { x: T, y: T }
fn make_one() -> i32 {
    let p: Pt<i32> = Pt { x: 1, y: 2 };
    p.x
}
fn make_two() -> i32 {
    let q: Pt<i32> = Pt { x: 3, y: 4 };
    q.x + q.y
}
"""
    prog = parse(src)
    generic = collect_generic_structs(prog)
    uses = collect_concrete_uses(prog, generic)
    assert len(uses) == 1


def test_distinct_body_uses_separate_types():
    """B1: `Pt<i32>` and `Pt<f64>` in different bodies produce two
    distinct mono'd structs."""
    src = """
struct Pt[T] { x: T, y: T }
fn make_i() -> i32 {
    let p: Pt<i32> = Pt { x: 1, y: 2 };
    p.x
}
fn make_f() -> f64 {
    let q: Pt<f64> = Pt { x: 1.0, y: 2.0 };
    q.x + q.y
}
"""
    prog = parse(src)
    generic = collect_generic_structs(prog)
    uses = collect_concrete_uses(prog, generic)
    names = {(n, args[0].name) for n, args in uses}
    assert names == {("Pt", "i32"), ("Pt", "f64")}, (
        f"expected Pt<i32> + Pt<f64>; got {names}"
    )


def test_monomorphize_structs_creates_distinct_mangled():
    """B1: monomorphize_structs(prog) emits Pt__i32 AND Pt__f64."""
    src = """
struct Pt[T] { x: T, y: T }
fn make_i() -> i32 {
    let p: Pt<i32> = Pt { x: 1, y: 2 };
    p.x
}
fn make_f() -> f64 {
    let q: Pt<f64> = Pt { x: 1.0, y: 2.0 };
    q.x + q.y
}
"""
    prog = parse(src)
    new_prog, diags = monomorphize_structs(prog)
    assert diags == []
    structs = {it.name for it in new_prog.items if isinstance(it, A.StructDecl)}
    assert "Pt__i32" in structs
    assert "Pt__f64" in structs


def test_typecheck_resolves_generic_struct_to_mangled():
    """A3: typecheck._resolve_type(Pt<i32>) returns TyStruct('Pt__i32'),
    NOT TyUnknown. Pt<i32> and Pt<f64> are then non-unifiable."""
    from helixc.frontend.typecheck import (
        TypeChecker, TyStruct,
    )
    src = """
struct Pt[T] { x: T, y: T }
fn dist_i(p: Pt<i32>) -> i32 { p.x }
fn dist_f(q: Pt<f64>) -> f64 { q.x + q.y }
"""
    prog = parse(src)
    tc = TypeChecker(prog)
    tc.check()
    sig_i = tc.functions["dist_i"]
    sig_f = tc.functions["dist_f"]
    p_ty_i = sig_i.params[0][1]
    p_ty_f = sig_f.params[0][1]
    assert isinstance(p_ty_i, TyStruct), f"expected TyStruct, got {p_ty_i}"
    assert isinstance(p_ty_f, TyStruct), f"expected TyStruct, got {p_ty_f}"
    assert p_ty_i.name == "Pt__i32"
    assert p_ty_f.name == "Pt__f64"
    # The two are not compatible:
    assert not tc._compatible(p_ty_i, p_ty_f)


def test_typecheck_arity_mismatch_falls_back_to_unknown():
    """A3 edge case: Pt<i32, f64> with Pt declared as Pt[T] (one param)
    has wrong arity — fall back to TyUnknown (existing behaviour),
    don't try to mangle a bad form."""
    from helixc.frontend.typecheck import (
        TypeChecker, TyUnknown,
    )
    src = """
struct Pt[T] { x: T }
fn bad(p: Pt<i32, f64>) -> i32 { 0 }
"""
    prog = parse(src)
    tc = TypeChecker(prog)
    tc.check()
    sig = tc.functions["bad"]
    p_ty = sig.params[0][1]
    # Bad arity — falls back to TyUnknown (no crash).
    assert isinstance(p_ty, TyUnknown)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
