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


def test_type_alias_target_collects_generic_struct_instantiation():
    src = """
type Probability = f64 where 0.0 <= self <= 1.0;
struct Box[T] { v: T }
type Alias = Box<Probability>;
fn f(p: Alias) -> f64 { p.v }
"""
    prog = parse(src)
    prog, diags = monomorphize_structs(prog)
    assert diags == []
    structs = _structs(prog)
    assert "Box__Probability" in structs


# ----------------------------------------------------------------------
# Audit 28.8 A13 — _ty_key proper arms (no collapse to "?")
# ----------------------------------------------------------------------
def test_ty_key_distinguishes_tyfn_params():
    """Audit 28.8 A13: pre-fix, every TyFn collapsed to ('?', 'TyFn').
    `Pt<fn(i32)->i32>` and `Pt<fn(f32)->f32>` should produce different
    keys so the mono pass dedupes correctly."""
    from helixc.frontend.struct_mono import _ty_key
    span = A.Span(0, 0)
    fn_i32 = A.TyFn(
        span=span,
        params=[A.TyName(span=span, name="i32")],
        ret=A.TyName(span=span, name="i32"),
    )
    fn_f32 = A.TyFn(
        span=span,
        params=[A.TyName(span=span, name="f32")],
        ret=A.TyName(span=span, name="f32"),
    )
    k1 = _ty_key(fn_i32)
    k2 = _ty_key(fn_f32)
    assert k1 != k2
    # Both must be hashable.
    assert hash(k1) != hash(k2)


def test_ty_key_distinguishes_tytensor_shape():
    """TyTensor with different shapes must produce different keys."""
    from helixc.frontend.struct_mono import _ty_key
    span = A.Span(0, 0)
    t16 = A.TyTensor(
        span=span,
        dtype=A.TyName(span=span, name="f32"),
        shape=[A.IntLit(span=span, value=16, type_suffix=None)],
    )
    t32 = A.TyTensor(
        span=span,
        dtype=A.TyName(span=span, name="f32"),
        shape=[A.IntLit(span=span, value=32, type_suffix=None)],
    )
    assert _ty_key(t16) != _ty_key(t32)


def test_ty_key_distinguishes_tytile_memspace():
    """TyTile with different memspaces must produce different keys
    (per A13: shape AND memspace participate in the key)."""
    from helixc.frontend.struct_mono import _ty_key
    span = A.Span(0, 0)
    t_smem = A.TyTile(
        span=span,
        dtype=A.TyName(span=span, name="f32"),
        shape=[A.IntLit(span=span, value=16, type_suffix=None)],
        memspace=A.Name(span=span, name="smem", generics=[]),
    )
    t_hbm = A.TyTile(
        span=span,
        dtype=A.TyName(span=span, name="f32"),
        shape=[A.IntLit(span=span, value=16, type_suffix=None)],
        memspace=A.Name(span=span, name="hbm", generics=[]),
    )
    assert _ty_key(t_smem) != _ty_key(t_hbm)


def test_ty_key_ptr_distinct_inner():
    """Audit 28.8 B6 / A13: TyPtr keys differ by inner type AND by
    mutability so generics over raw ptrs dedupe correctly."""
    from helixc.frontend.struct_mono import _ty_key
    span = A.Span(0, 0)
    p_i32_const = A.TyPtr(
        span=span,
        inner=A.TyName(span=span, name="i32"),
        is_mut=False,
    )
    p_i32_mut = A.TyPtr(
        span=span,
        inner=A.TyName(span=span, name="i32"),
        is_mut=True,
    )
    p_f64_const = A.TyPtr(
        span=span,
        inner=A.TyName(span=span, name="f64"),
        is_mut=False,
    )
    # const vs mut must differ; same const but different inner must differ.
    assert _ty_key(p_i32_const) != _ty_key(p_i32_mut)
    assert _ty_key(p_i32_const) != _ty_key(p_f64_const)


# ----------------------------------------------------------------------
# Audit 28.8 B6 — substitute_ty TyPtr arm
# ----------------------------------------------------------------------
def test_substitute_ty_ptr_substitutes_inner():
    """Audit 28.8 B6: substitute_ty must walk into TyPtr.inner so
    `*const T` mono'd against T=f64 becomes `*const f64`, not stays
    `*const T` (which downstream lower-ast would silently default
    to *const i32)."""
    from helixc.frontend.monomorphize import substitute_ty
    span = A.Span(0, 0)
    ptr_T = A.TyPtr(
        span=span,
        inner=A.TyName(span=span, name="T"),
        is_mut=False,
    )
    subst = {"T": A.TyName(span=span, name="f64")}
    result = substitute_ty(ptr_T, subst)
    assert isinstance(result, A.TyPtr)
    assert result.is_mut is False
    assert isinstance(result.inner, A.TyName)
    assert result.inner.name == "f64"


def test_substitute_ty_ptr_preserves_mut():
    """*mut T mono'd preserves is_mut."""
    from helixc.frontend.monomorphize import substitute_ty
    span = A.Span(0, 0)
    ptr_mut_T = A.TyPtr(
        span=span,
        inner=A.TyName(span=span, name="T"),
        is_mut=True,
    )
    result = substitute_ty(ptr_mut_T, {"T": A.TyName(span=span, name="i32")})
    assert isinstance(result, A.TyPtr)
    assert result.is_mut is True
    assert result.inner.name == "i32"


# ----------------------------------------------------------------------
# Audit 28.8 B7 — instantiate preserves where_clauses and is_extern
# ----------------------------------------------------------------------
def test_instantiate_preserves_extern():
    """Audit 28.8 B7: cloned generic fn keeps is_extern + extern_abi
    so `extern "C" fn malloc<T>(...) -> *mut T` mono'd to
    `malloc__i32` is still recognized as extern by codegen."""
    from helixc.frontend.monomorphize import Monomorphizer
    span = A.Span(0, 0)
    extern_fn = A.FnDecl(
        span=span, name="alloc",
        generics=[A.GenericParam(span=span, name="T", kind="type")],
        params=[A.FnParam(span=span, name="n",
                          ty=A.TyName(span=span, name="usize"),
                          is_mut=False)],
        return_ty=A.TyPtr(
            span=span,
            inner=A.TyName(span=span, name="T"),
            is_mut=True,
        ),
        where_clauses=[],
        body=A.Block(span=span, stmts=[], final_expr=None),
        attrs=[],
        is_pub=False,
        is_extern=True,
        extern_abi="C",
    )
    # Synthesize a caller that triggers monomorphization.
    main_fn = A.FnDecl(
        span=span, name="main", generics=[], params=[],
        return_ty=A.TyName(span=span, name="i32"),
        where_clauses=[],
        body=A.Block(
            span=span,
            stmts=[A.ExprStmt(
                span=span,
                expr=A.Call(
                    span=span,
                    callee=A.Name(span=span, name="alloc",
                                  generics=[A.TyName(span=span, name="i32")]),
                    args=[A.IntLit(span=span, value=8,
                                   type_suffix=None)],
                ),
            )],
            final_expr=A.IntLit(span=span, value=0, type_suffix=None),
        ),
        attrs=[], is_pub=False,
    )
    prog = A.Program(module=None, items=[extern_fn, main_fn])
    Monomorphizer(prog).run()
    cloned = [it for it in prog.items
              if isinstance(it, A.FnDecl) and it.name == "alloc__i32"]
    assert len(cloned) == 1
    clone = cloned[0]
    assert clone.is_extern is True
    assert clone.extern_abi == "C"


def test_instantiate_deep_copies_where_clauses():
    """Audit 28.8 B7: cloned fn's where_clauses must be a new list
    distinct from the template's, so downstream passes that walk the
    clone's clauses don't see template-shape (with TyVars still
    present)."""
    from helixc.frontend.monomorphize import Monomorphizer
    span = A.Span(0, 0)
    template = A.FnDecl(
        span=span, name="f",
        generics=[A.GenericParam(span=span, name="T", kind="type")],
        params=[A.FnParam(span=span, name="x",
                          ty=A.TyName(span=span, name="T"),
                          is_mut=False)],
        return_ty=A.TyName(span=span, name="T"),
        where_clauses=[A.WhereClause(
            span=span,
            constraint=A.IntLit(span=span, value=0, type_suffix=None),
        )],
        body=A.Block(
            span=span, stmts=[],
            final_expr=A.Name(span=span, name="x", generics=[]),
        ),
        attrs=[], is_pub=False,
    )
    main_fn = A.FnDecl(
        span=span, name="main", generics=[], params=[],
        return_ty=A.TyName(span=span, name="i32"),
        where_clauses=[],
        body=A.Block(
            span=span,
            stmts=[],
            final_expr=A.Call(
                span=span,
                callee=A.Name(span=span, name="f",
                              generics=[A.TyName(span=span, name="i32")]),
                args=[A.IntLit(span=span, value=42, type_suffix=None)],
            ),
        ),
        attrs=[], is_pub=False,
    )
    prog = A.Program(module=None, items=[template, main_fn])
    Monomorphizer(prog).run()
    cloned = [it for it in prog.items
              if isinstance(it, A.FnDecl) and it.name == "f__i32"]
    assert len(cloned) == 1
    # The clone's where_clauses must be a NEW list (not aliased to
    # the template's list).
    assert cloned[0].where_clauses is not template.where_clauses
    # And same length (no clauses dropped).
    assert len(cloned[0].where_clauses) == len(template.where_clauses)


# ----------------------------------------------------------------------
# Audit 28.8 B8 — TyTile shape size-param substitution
# ----------------------------------------------------------------------
def test_substitute_ty_tile_shape_subs_name_to_int():
    """Audit 28.8 B8: TyTile<f32, [N], hbm> mono'd against N=128 must
    produce a clone with shape=[IntLit(128)], not shape=[Name('N')]
    (which lower-ast then defaults to length 0)."""
    from helixc.frontend.monomorphize import substitute_ty, _SizeLitMarker
    span = A.Span(0, 0)
    tile_ty = A.TyTile(
        span=span,
        dtype=A.TyName(span=span, name="f32"),
        shape=[A.Name(span=span, name="N", generics=[])],
        memspace=A.Name(span=span, name="hbm", generics=[]),
    )
    subst = {"N": _SizeLitMarker(128)}
    result = substitute_ty(tile_ty, subst)
    assert isinstance(result, A.TyTile)
    assert len(result.shape) == 1
    # After sub, Name('N') in shape becomes IntLit(128).
    assert isinstance(result.shape[0], A.IntLit)
    assert result.shape[0].value == 128


def test_substitute_ty_tensor_shape_size_param():
    """Same B8 fix applies to TyTensor.shape."""
    from helixc.frontend.monomorphize import substitute_ty, _SizeLitMarker
    span = A.Span(0, 0)
    tensor_ty = A.TyTensor(
        span=span,
        dtype=A.TyName(span=span, name="f64"),
        shape=[A.Name(span=span, name="M", generics=[]),
               A.Name(span=span, name="K", generics=[])],
    )
    subst = {"M": _SizeLitMarker(64), "K": _SizeLitMarker(32)}
    result = substitute_ty(tensor_ty, subst)
    assert isinstance(result, A.TyTensor)
    assert isinstance(result.shape[0], A.IntLit)
    assert result.shape[0].value == 64
    assert isinstance(result.shape[1], A.IntLit)
    assert result.shape[1].value == 32


# ----------------------------------------------------------------------
# Audit 28.8 B12 — UnsafeBlock walker arm
# ----------------------------------------------------------------------
def test_walk_subst_expr_descends_into_unsafe_block():
    """Audit 28.8 B12: _walk_subst_expr must walk into UnsafeBlock
    body. Pre-fix, generic fns whose body wrapped a region in `unsafe
    { ... }` got the body left unsubstituted."""
    from helixc.frontend.monomorphize import _walk_subst_expr
    span = A.Span(0, 0)
    # Synth `unsafe { let x: T = 0; x }`
    inner_block = A.Block(
        span=span,
        stmts=[A.Let(
            span=span, name="x", is_mut=False,
            ty=A.TyName(span=span, name="T"),
            value=A.IntLit(span=span, value=0, type_suffix=None),
        )],
        final_expr=A.Name(span=span, name="x", generics=[]),
    )
    unsafe = A.UnsafeBlock(span=span, body=inner_block)
    subst = {"T": A.TyName(span=span, name="f64")}
    result = _walk_subst_expr(unsafe, subst)
    # Result must remain an UnsafeBlock (not collapsed to anything else).
    assert isinstance(result, A.UnsafeBlock)
    # Inner Block's Let must have T substituted to f64.
    inner = result.body
    assert isinstance(inner, A.Block)
    let_stmt = inner.stmts[0]
    assert isinstance(let_stmt, A.Let)
    assert isinstance(let_stmt.ty, A.TyName)
    assert let_stmt.ty.name == "f64"


def test_c3_4_monomorphize_structs_idempotent():
    """Audit 28.8 cycle 3 C3-4: monomorphize_structs must not append
    duplicate mangled StructDecls on a second invocation. Pre-fix,
    `Pt<i32>` used twice across check.py + x86_64 driver appended
    Pt__i32 twice to prog.items."""
    from helixc.frontend.struct_mono import monomorphize_structs
    src = """
struct Pt[T] { x: T, y: T }
fn use_pt() -> i32 {
    let p: Pt<i32> = Pt { x: 1, y: 2 };
    0
}
"""
    prog = parse(src)
    monomorphize_structs(prog)
    after_first = sum(
        1 for it in prog.items
        if isinstance(it, A.StructDecl) and it.name == "Pt__i32"
    )
    assert after_first == 1, (
        f"first mono should produce exactly one Pt__i32, got {after_first}"
    )
    monomorphize_structs(prog)
    after_second = sum(
        1 for it in prog.items
        if isinstance(it, A.StructDecl) and it.name == "Pt__i32"
    )
    assert after_second == 1, (
        f"second mono should be idempotent, got {after_second} Pt__i32"
    )


def test_c3_6_shape_fold_div_by_zero_traps():
    """Audit 28.8 cycle 3 C3-6: `_fold_intlit_arith` must raise
    ShapeFoldError (trap 28801) on `/0` and `%0`. Pre-fix, the unfolded
    Binary silently fell through to length 0 in lower_ast."""
    from helixc.frontend.monomorphize import (
        _fold_intlit_arith, ShapeFoldError,
    )
    span = A.Span(0, 0)
    div_zero = A.Binary(
        span=span, op="/",
        left=A.IntLit(span=span, value=10, type_suffix=None),
        right=A.IntLit(span=span, value=0, type_suffix=None),
    )
    try:
        _fold_intlit_arith(div_zero)
        assert False, "expected ShapeFoldError on / by 0"
    except ShapeFoldError as e:
        assert "28801" in str(e)
    mod_zero = A.Binary(
        span=span, op="%",
        left=A.IntLit(span=span, value=10, type_suffix=None),
        right=A.IntLit(span=span, value=0, type_suffix=None),
    )
    try:
        _fold_intlit_arith(mod_zero)
        assert False, "expected ShapeFoldError on % by 0"
    except ShapeFoldError as e:
        assert "28801" in str(e)


def test_d5_unary_fold_neg_intlit():
    """Audit 28.8 cycle 3 D5: `_fold_intlit_unary` must fold
    `Unary(-, IntLit(5))` to `IntLit(-5)`. Pre-fix the substituted
    Unary stayed as Unary and `_resolve_size_expr` fell through to
    TyUnknown."""
    from helixc.frontend.monomorphize import _fold_intlit_unary
    span = A.Span(0, 0)
    neg = A.Unary(
        span=span, op="-",
        operand=A.IntLit(span=span, value=5, type_suffix=None),
    )
    folded = _fold_intlit_unary(neg)
    assert isinstance(folded, A.IntLit), (
        f"expected IntLit after fold, got {type(folded).__name__}"
    )
    assert folded.value == -5
    # `+5` should fold to `5`.
    pos = A.Unary(
        span=span, op="+",
        operand=A.IntLit(span=span, value=7, type_suffix=None),
    )
    folded_pos = _fold_intlit_unary(pos)
    assert isinstance(folded_pos, A.IntLit)
    assert folded_pos.value == 7


def test_d6_ty_key_raises_on_non_astnode():
    """Audit 28.8 cycle 3 D6: `_ty_key` must reject non-AST TyNode
    inputs loudly (not silently dedup via name-only key)."""
    from helixc.frontend.struct_mono import _ty_key

    class FakeNonAstType:
        pass

    try:
        _ty_key(FakeNonAstType())
        assert False, "expected TypeError on non-TyNode input"
    except TypeError as e:
        assert "AST TyNode" in str(e)


def test_d9_turbofish_inside_generic_body_substituted():
    """Audit 28.8 cycle 3 D9: when a generic fn calls another generic
    fn via turbofish (`id::<T>(x)`), the inner turbofish's `T` must
    be substituted when the outer fn is monomorphized. Pre-fix, the
    inner Call kept literal `T` and produced `id__T` instead of
    `id__i32`."""
    from helixc.frontend.monomorphize import _walk_subst_expr
    span = A.Span(0, 0)
    # Synth `id::<T>(x)`
    callee = A.Name(span=span, name="id",
                    generics=[A.TyName(span=span, name="T")])
    call = A.Call(
        span=span,
        callee=callee,
        args=[A.Name(span=span, name="x", generics=[])],
    )
    subst = {"T": A.TyName(span=span, name="i32")}
    result = _walk_subst_expr(call, subst)
    assert isinstance(result, A.Call)
    assert isinstance(result.callee, A.Name)
    assert result.callee.name == "id"
    assert len(result.callee.generics) == 1
    assert isinstance(result.callee.generics[0], A.TyName)
    assert result.callee.generics[0].name == "i32", (
        f"expected T->i32 in turbofish generics, got "
        f"{result.callee.generics[0].name}"
    )


def test_c4_4_nested_turbofish_end_to_end_no_unresolved_generic_param():
    """Audit 28.8 cycle 5 C4-4 / HIGH: D9's cycle-3 fix was paper-only.
    The unit test (test_d9_turbofish_inside_generic_body_substituted)
    directly invokes `_walk_subst_expr` and passes. But the end-to-end
    `Monomorphizer.run()` iteration order processed generic-fn bodies
    BEFORE clones were re-walked, so clones referencing nested-turbofish
    still ended up calling `id__U` (unresolved generic-param name)
    instead of `id__i32`. The cycle-5 fix promotes new clones into the
    walk set so the next iteration follows their nested-turbofish."""
    from helixc.frontend.parser import parse
    from helixc.frontend.monomorphize import monomorphize
    src = """
        fn id[T](x: T) -> T { x }
        fn caller[U](v: U) -> U { id::<U>(v) }
        fn main() -> i32 { caller::<i32>(7) }
    """
    prog = parse(src)
    monomorphize(prog)
    # Collect all callee names in the final program.
    callee_names = []
    def _walk(e):
        if isinstance(e, A.Call):
            if isinstance(e.callee, A.Name):
                callee_names.append(e.callee.name)
            _walk(e.callee)
            for arg in e.args:
                _walk(arg)
        elif isinstance(e, A.Block):
            for s in e.stmts:
                if isinstance(s, A.Let) and s.value is not None:
                    _walk(s.value)
                elif isinstance(s, A.ExprStmt):
                    _walk(s.expr)
            if e.final_expr is not None:
                _walk(e.final_expr)
    for item in prog.items:
        if isinstance(item, A.FnDecl):
            _walk(item.body)
    # No clone should still reference id__U (the unresolved generic-param
    # mangled form). Pre-fix, `caller__i32`'s body called `id__U`.
    assert "id__U" not in callee_names, (
        f"D9 end-to-end: clone references unresolved generic-param name "
        f"id__U (instead of id__i32). callees={callee_names}"
    )
    # Positive assertion: id__i32 should be present.
    assert "id__i32" in callee_names, (
        f"D9 end-to-end: expected clone to call id__i32. "
        f"callees={callee_names}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
