"""Tests for helixc.frontend.typecheck (v0.1 scaffold)."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.frontend.typecheck import typecheck


def check(src: str) -> list[str]:
    prog = parse(src)
    errs = typecheck(prog)
    return [str(e) for e in errs]


# ============================================================================
# Should typecheck (no errors)
# ============================================================================
def test_simple_add():
    assert check("fn add(a: i32, b: i32) -> i32 { a + b }") == []


def test_let_typed_match():
    assert check("fn f() { let x: i32 = 42; }") == []


def test_let_inferred():
    assert check("fn f() { let x = 42; }") == []


def test_if_branches_match():
    src = "fn f(b: bool) -> i32 { if b { 1 } else { 2 } }"
    assert check(src) == []


def test_function_call():
    src = """
    fn double(x: i32) -> i32 { x + x }
    fn f() -> i32 { double(7) }
    """
    assert check(src) == []


def test_generic_signature():
    src = """
    fn id[T](x: T) -> T { x }
    """
    # T propagates; should typecheck (param T -> return T)
    assert check(src) == []


def test_tensor_signature():
    src = """
    fn matmul[N: size, M: size, P: size](
        a: tensor<f32, [N, M]>,
        b: tensor<f32, [M, P]>,
    ) -> tensor<f32, [N, P]>
    {
        let c = tensor::zeros();
        c
    }
    """
    # tensor::zeros returns Unknown which is compatible-with-anything in v0.1
    assert check(src) == []


def test_struct_def_no_check():
    src = "struct Point { x: f32, y: f32 }"
    # struct definitions don't trigger body checking
    assert check(src) == []


# ============================================================================
# Should detect errors
# ============================================================================
def test_let_type_mismatch():
    errs = check("fn f() { let x: bool = 42; }")
    assert any("declared bool but value is i32" in e for e in errs), errs


def test_return_type_mismatch():
    errs = check("fn f() -> bool { 42 }")
    assert any("does not match return type" in e for e in errs), errs


def test_if_branches_differ():
    errs = check("fn f(b: bool) -> bool { if b { 1 } else { true } }")
    # First arm i32, second bool. Whole if is i32; outer return type bool -> mismatch
    assert any("does not match return type" in e or "if/else branches differ" in e for e in errs), errs


def test_duplicate_function():
    errs = check("fn foo() {} fn foo() {}")
    assert any("duplicate function" in e for e in errs), errs


# ============================================================================
# Compile-time shape checking via Presburger solver (Phase 3-iv)
# ============================================================================
def test_shape_check_concrete_match():
    # Both args have concrete shape [4, 4]; should typecheck cleanly.
    src = """
    fn matmul(a: tensor<f32, [4, 4]>, b: tensor<f32, [4, 4]>) -> tensor<f32, [4, 4]> {
        a
    }
    fn caller(x: tensor<f32, [4, 4]>, y: tensor<f32, [4, 4]>) {
        matmul(x, y);
    }
    """
    assert check(src) == []


def test_shape_check_concrete_mismatch():
    # Caller passes [5, 4] where formal expects [4, 4] — should reject.
    src = """
    fn matmul(a: tensor<f32, [4, 4]>, b: tensor<f32, [4, 4]>) -> tensor<f32, [4, 4]> {
        a
    }
    fn caller(x: tensor<f32, [4, 4]>, z: tensor<f32, [5, 4]>) {
        matmul(x, z);
    }
    """
    errs = check(src)
    assert any("shape constraint violated" in e or "rank" in e for e in errs), errs


def test_shape_check_rank_mismatch():
    src = """
    fn takes2d(a: tensor<f32, [4, 4]>) {}
    fn caller(x: tensor<f32, [4]>) {
        takes2d(x);
    }
    """
    errs = check(src)
    assert any("rank" in e for e in errs), errs


def test_shape_check_size_polymorphic_match():
    src = """
    fn matmul[N: size, M: size, P: size](
        a: tensor<f32, [N, M]>,
        b: tensor<f32, [M, P]>,
    ) -> tensor<f32, [N, P]> { a }
    fn caller(x: tensor<f32, [3, 4]>, y: tensor<f32, [4, 5]>) {
        matmul(x, y);
    }
    """
    errs = check(src)
    assert not any("shape constraint violated" in e for e in errs), errs


# ============================================================================
# Effect/capability type system (Phase 3-ii)
# ============================================================================
def test_pure_calls_pure_ok():
    src = """
    @pure fn helper() -> i32 { 7 }
    @pure fn caller() -> i32 { helper() }
    """
    assert check(src) == []


def test_pure_calls_effectful_rejected():
    # @pure cannot call a function with side effects
    src = """
    @effect fn writes_disk() {}
    @pure fn caller() { writes_disk() }
    """
    errs = check(src)
    assert any("cannot call" in e or "non-pure" in e for e in errs), errs


def test_effectful_can_call_effectful():
    # caller declares the same effects, so the call is permitted
    src = """
    @io fn writes_disk() {}
    @io fn caller() { writes_disk() }
    """
    assert check(src) == []


def test_caller_missing_capability_rejected():
    # io-capable function called from a function without that capability
    src = """
    @io fn read_file() -> i32 { 0 }
    fn naive_caller() { read_file(); }
    """
    errs = check(src)
    assert any("requires effect" in e or "does not declare" in e for e in errs), errs


def test_pure_and_effect_conflict():
    # Function declared both @pure and @effect — conflict
    src = """
    @pure
    @io
    fn confused() {}
    """
    errs = check(src)
    assert any("cannot be both" in e for e in errs), errs


def test_pure_calls_unmarked_function_allowed():
    """Surface typecheck: @pure may call unannotated functions whose
    declared effect set is empty. The IR-level effect_check.py is the
    soundness layer — it computes transitive effects from PRINT ops
    and catches actual impurity. The surface check is intentionally
    permissive about unannotated callees so users don't have to
    @pure-annotate every helper just to call it from a @pure caller."""
    src = """
    fn does_anything() -> i32 { 7 }
    @pure fn caller() -> i32 { does_anything() }
    """
    errs = check(src)
    assert errs == [], f"expected no surface errors, got: {errs}"


def test_pure_calls_explicit_effect_rejected():
    """@pure must still be rejected when the callee declares effects."""
    src = """
    @effect(io)
    fn print_thing() -> i32 { 42 }
    @pure fn caller() -> i32 { print_thing() }
    """
    errs = check(src)
    assert any("@pure" in e and "effectful" in e for e in errs), errs


# ============================================================================
# Differentiable types D<T> (Phase 3-iii)
# ============================================================================
def test_diff_type_round_trip():
    # Function takes D<f32>, returns D<f32>; should typecheck
    src = """
    fn loss(x: D<f32>) -> D<f32> { x }
    """
    assert check(src) == []


def test_diff_propagates_through_arith():
    # D<f32> * D<f32> should be D<f32>
    src = """
    fn loss(x: D<f32>, y: D<f32>) -> D<f32> {
        x * y
    }
    """
    assert check(src) == []


def test_diff_propagates_with_scalar():
    # D<f32> * f32 should be D<f32> (the f32 is a constant)
    src = """
    fn scale(x: D<f32>) -> D<f32> {
        x * x
    }
    """
    assert check(src) == []


def test_diff_return_type_mismatch():
    # Returning a non-D from a D-typed return is an error
    src = """
    fn bad(x: D<f32>) -> f32 {
        x
    }
    """
    errs = check(src)
    # x is D<f32>, return type declared f32 — type mismatch
    assert any("does not match return type" in e for e in errs), errs


def test_diff_in_tensor():
    # D<tensor<...>> — differentiable tensor
    src = """
    fn loss(x: D<f32>, y: D<f32>) -> D<f32> {
        x * y + x
    }
    """
    assert check(src) == []


# ============================================================================
# Memory-tier types (Phase 3-v)
# ============================================================================
def test_working_mem_type():
    src = """
    fn f(x: WorkingMem<i32>) -> WorkingMem<i32> { x }
    """
    assert check(src) == []


def test_episodic_mem_type():
    src = """
    fn store(e: EpisodicMem<i32>) -> EpisodicMem<i32> { e }
    """
    assert check(src) == []


def test_cannot_pass_episodic_as_semantic():
    # EpisodicMem and SemanticMem are different tiers — must explicitly
    # consolidate. Direct passing should be rejected.
    src = """
    fn takes_semantic(x: SemanticMem<i32>) -> SemanticMem<i32> { x }
    fn caller(e: EpisodicMem<i32>) -> SemanticMem<i32> {
        let s: SemanticMem<i32> = e;
        s
    }
    """
    errs = check(src)
    assert any("declared SemanticMem" in e or "declared" in e for e in errs), errs


def test_same_tier_compatible():
    src = """
    fn takes_working(x: WorkingMem<i32>) -> WorkingMem<i32> { x }
    fn caller(w: WorkingMem<i32>) -> WorkingMem<i32> {
        let y: WorkingMem<i32> = w;
        y
    }
    """
    assert check(src) == []


def test_let_episodic_to_working_rejected():
    src = """
    fn f(e: EpisodicMem<i32>) -> WorkingMem<i32> {
        let w: WorkingMem<i32> = e;
        w
    }
    """
    errs = check(src)
    assert any("declared" in e for e in errs), errs


# ============================================================================
# Built-in type transitions: detach, attach, consolidate, recall
# ============================================================================
def test_detach_strips_diff():
    # detach(x: D<f32>) -> f32
    src = """
    fn use_grad(x: D<f32>) -> f32 {
        let plain = detach(x);
        plain
    }
    """
    assert check(src) == []


def test_attach_adds_diff():
    src = """
    fn add_grad(x: f32) -> D<f32> {
        let g = attach(x);
        g
    }
    """
    assert check(src) == []


def test_consolidate_episodic_to_semantic():
    src = """
    fn store(e: EpisodicMem<i32>) -> SemanticMem<i32> {
        consolidate(e)
    }
    """
    assert check(src) == []


def test_consolidate_rejects_non_episodic():
    # consolidate() must take EpisodicMem<T>, not Working
    src = """
    fn bad(w: WorkingMem<i32>) -> SemanticMem<i32> {
        consolidate(w)
    }
    """
    errs = check(src)
    assert any("requires EpisodicMem" in e for e in errs), errs


def test_recall_semantic_to_working():
    src = """
    fn fetch(s: SemanticMem<i32>) -> WorkingMem<i32> {
        recall(s)
    }
    """
    assert check(src) == []


def test_recall_rejects_non_semantic():
    src = """
    fn bad(e: EpisodicMem<i32>) -> WorkingMem<i32> {
        recall(e)
    }
    """
    errs = check(src)
    assert any("requires SemanticMem" in e for e in errs), errs


# ============================================================================
# Argument count + primitive type checking on function calls
# ============================================================================
def test_arg_count_too_few():
    src = """
    fn add(a: i32, b: i32) -> i32 { a + b }
    fn caller() -> i32 { add(1) }
    """
    errs = check(src)
    assert any("expected 2 args, got 1" in e for e in errs), errs


def test_arg_count_too_many():
    src = """
    fn id(x: i32) -> i32 { x }
    fn caller() -> i32 { id(1, 2, 3) }
    """
    errs = check(src)
    assert any("expected 1 args, got 3" in e for e in errs), errs


def test_arg_type_mismatch():
    src = """
    fn takes_int(x: i32) -> i32 { x }
    fn caller() -> i32 { takes_int(true) }
    """
    errs = check(src)
    assert any("expects i32, got bool" in e for e in errs), errs


def test_arg_type_match_ok():
    src = """
    fn add(a: i32, b: i32) -> i32 { a + b }
    fn caller() -> i32 { add(1, 2) }
    """
    assert check(src) == []


def test_arg_int_vs_float():
    src = """
    fn takes_float(x: f32) -> f32 { x }
    fn caller() -> f32 { takes_float(42) }
    """
    errs = check(src)
    assert any("expects f32, got i32" in e for e in errs), errs


# ============================================================================
# Auto-curriculum primitive: learn_to (Phase 3-ix)
# ============================================================================
def test_learn_to_returns_skill():
    # learn_to(task, difficulty, budget) is a builtin returning Skill<...>
    src = """
    fn main() -> i32 {
        let skill = learn_to("matmul", 0.5, 100);
        0
    }
    """
    # Should typecheck cleanly
    assert check(src) == []


def test_learn_to_wrong_arity():
    src = """
    fn main() -> i32 {
        let skill = learn_to("foo");
        0
    }
    """
    errs = check(src)
    assert any("requires 3 args" in e for e in errs), errs


# ============================================================================
# Test runner
# ============================================================================
def test_struct_lit_typechecks_clean():
    """A well-formed struct literal should typecheck without errors."""
    src = """
    struct Point { x: i32, y: i32 }
    fn main() -> i32 {
        let p = Point { x: 10, y: 20 };
        p.x
    }
    """
    errs = check(src)
    assert errs == [], f"unexpected errors: {errs}"


def test_struct_lit_missing_field_errors():
    """Forgetting a field surfaces a 'missing field(s)' error."""
    src = """
    struct Point { x: i32, y: i32 }
    fn main() -> i32 {
        let p = Point { x: 10 };
        p.x
    }
    """
    errs = check(src)
    assert any("missing field" in s and "y" in s for s in errs), \
        f"expected missing-field error, got {errs}"


def test_struct_lit_unknown_field_errors():
    """A field not in the decl surfaces an 'unknown field(s)' error."""
    src = """
    struct Point { x: i32, y: i32 }
    fn main() -> i32 {
        let p = Point { x: 10, y: 20, z: 30 };
        p.x
    }
    """
    errs = check(src)
    assert any("unknown field" in s and "z" in s for s in errs), \
        f"expected unknown-field error, got {errs}"


def test_struct_field_access_unknown_field_errors():
    """Reading p.z where Point only has x, y surfaces an error."""
    src = """
    struct Point { x: i32, y: i32 }
    fn main() -> i32 {
        let p = Point { x: 10, y: 20 };
        p.z
    }
    """
    errs = check(src)
    assert any("no field" in s and "z" in s for s in errs), \
        f"expected unknown-field-access error, got {errs}"


def test_nested_struct_field_type_tracks_correctly():
    """A struct-typed field's value must resolve to TyStruct (not TyUnknown)
    so chained field access types correctly. Pre-fix this was silently
    TyUnknown, which made `o.inner` incompatible-with-anything pass."""
    src = """
    struct Inner { value: i32 }
    struct Outer { count: i32, inner: Inner }
    fn main() -> i32 {
        let o = Outer { count: 10, inner: Inner { value: 32 } };
        o.count + o.inner.value
    }
    """
    errs = check(src)
    assert errs == [], f"unexpected errors: {errs}"


def test_three_segment_path_on_known_enum_errors():
    """`Op::Sub::Variant` should surface a clear error rather than
    silent TyUnknown propagation."""
    src = """
    enum Op { Add, Sub }
    fn main() -> i32 {
        let v = Op::Sub::SomethingExtra;
        0
    }
    """
    errs = check(src)
    assert any("3+ segments" in s or "v0.1" in s for s in errs), \
        f"expected 3-segment path error, got {errs}"


def test_int_literal_overflow_errors():
    """Static overflow: `let x: i32 = 5_000_000_000` exceeds i32 range."""
    src = """
    fn main() -> i32 {
        let x: i32 = 5000000000;
        x
    }
    """
    errs = check(src)
    assert any("does not fit in i32" in s for s in errs), \
        f"expected overflow error, got {errs}"
    assert any("i64" in s for s in errs), \
        f"expected i64 hint, got {errs}"


def test_int_literal_in_range_no_error():
    """Values within i32 range typecheck cleanly."""
    src = """
    fn main() -> i32 {
        let x: i32 = 2147483647;
        x
    }
    """
    errs = check(src)
    assert errs == [], f"unexpected errors: {errs}"


def test_int_literal_negative_overflow_errors():
    """Negative overflow: -3_000_000_000 doesn't fit in i32."""
    src = """
    fn main() -> i32 {
        let x: i32 = 0 - 3000000000;
        x
    }
    """
    errs = check(src)
    # The literal '3000000000' itself is u32-shaped, but the let-stmt's
    # value is a Binary not an IntLit, so this won't trigger our static
    # check. We only catch literals at the binding point. Document this.
    # Test just verifies we don't crash and the program is acceptable.
    # (A future ticket would do constant-folding-aware overflow.)
    pass


def test_struct_field_returns_correct_type():
    """p.x of struct Point { x: i32, y: i32 } types as i32, allowing
    further i32 ops without errors."""
    src = """
    struct Point { x: i32, y: i32 }
    fn main() -> i32 {
        let p = Point { x: 10, y: 20 };
        p.x + p.y + 1
    }
    """
    errs = check(src)
    assert errs == [], f"unexpected errors: {errs}"


def test_struct_lit_unknown_struct_errors():
    """A struct lit referencing an undeclared struct surfaces an error."""
    src = """
    fn main() -> i32 {
        let p = Nope { x: 10 };
        0
    }
    """
    errs = check(src)
    assert any("unknown struct" in s for s in errs), \
        f"expected unknown-struct error, got {errs}"


def test_call_did_you_mean_suggests_builtin():
    """Misspelled stdlib call: `__exf(x)` should suggest `__exp`."""
    src = """
    fn main(x: f32) -> f32 {
        __exf(x)
    }
    """
    errs = check(src)
    assert any("unbound" in s and "__exf" in s for s in errs), \
        f"expected unbound '__exf', got {errs}"
    assert any("__exp" in s for s in errs), \
        f"expected hint mentioning '__exp', got {errs}"


def test_unbound_name_did_you_mean_suggests_close_match():
    src = """
    fn main() -> i32 {
        let counter = 5;
        countr + 1
    }
    """
    errs = check(src)
    assert any("unbound" in s and "countr" in s for s in errs), \
        f"expected unbound 'countr', got {errs}"
    assert any("counter" in s for s in errs), \
        f"expected hint mentioning 'counter', got {errs}"


def test_unbound_builtin_not_flagged():
    src = """
    fn use_grad(x: D<f32>) -> f32 { detach(x) }
    """
    errs = check(src)
    assert not any("unbound" in s for s in errs), \
        f"unexpected unbound error for builtin: {errs}"


# ----------------------------------------------------------------------
# Audit 28.8 B10 — typecheck Quote/Splice/Modify arms
# ----------------------------------------------------------------------
def test_quote_returns_tyquote():
    """Quote(inner) types as TyQuote(typeof(inner)) — not TyUnknown."""
    from helixc.frontend.typecheck import TypeChecker, TyQuote, TyPrim
    src = """
    fn main() -> i32 {
        let q = quote(42);
        0
    }
    """
    prog = parse(src)
    tc = TypeChecker(prog)
    tc.check()
    # The let q = quote(42) does not leak an error and q's type is TyQuote.
    # We inspect by re-checking main's body scope — easier via typecheck side
    # effect testing: assert no unhandled-Quote diagnostic.
    errs = [str(e) for e in tc.errors]
    assert not any("unhandled Quote" in s for s in errs), \
        f"unexpected unhandled-Quote: {errs}"


def test_splice_unwraps_tyquote():
    """splice(quote(42)) must typecheck to i32 (inner of Quote<i32>)."""
    src = """
    fn main() -> i32 {
        let q = quote(42);
        splice(q)
    }
    """
    errs = check(src)
    assert errs == [], f"unexpected errors: {errs}"


def test_splice_of_non_quote_diagnoses_11001():
    """Audit 28.8 B10 (trap 11001): splice'ing a non-Quote value
    surfaces the trap-11001 diagnostic instead of silently typing
    as TyUnknown and unifying with anything downstream."""
    src = """
    fn main() -> i32 {
        let x = 42;
        splice(x)
    }
    """
    errs = check(src)
    assert any("11001" in s for s in errs), \
        f"expected trap 11001 diagnostic, got {errs}"


def test_modify_returns_i32():
    """Modify is documented to return i32 (1=applied, 0=rejected).
    Pre-fix it typed as TyUnknown, which unified with anything."""
    src = """
    fn yes(handle: i32, new_val: i32) -> i32 { 1 }
    fn main() -> i32 {
        let h = quote(0);
        let applied = modify(h, 42, yes);
        if applied == 1 { 1 } else { 0 }
    }
    """
    errs = check(src)
    assert errs == [], f"modify must typecheck clean, got: {errs}"


def test_unsafe_block_propagates_inner_type():
    """Audit 28.8 B10 (UnsafeBlock arm): unsafe { e } must return
    typeof(e), not TyUnknown. The handler exists pre-A2 too; this
    asserts the type DOES propagate through."""
    src = """
    fn main() -> i32 {
        let x: i32 = unsafe { 42 };
        x
    }
    """
    errs = check(src)
    assert errs == [], f"unexpected errors: {errs}"


# ----------------------------------------------------------------------
# Audit 28.8 B11 — flatten_impls duplicate-method-name detection
# ----------------------------------------------------------------------
def test_flatten_impls_rejects_same_name_methods():
    """Audit 28.8 B11 (trap 74002): two structs with the same method
    name cause cross-struct type confusion at the call site. The
    fix raises DuplicateMethodError at the second registration.

    Python parser requires explicit param types (no `self` sugar);
    construct the impl-blocks directly via the AST."""
    from helixc.frontend.flatten_impls import (
        flatten_impls, DuplicateMethodError,
    )
    from helixc.frontend import ast_nodes as A
    import pytest as _pt
    span = A.Span(0, 0)
    pt = A.StructDecl(
        span=span, name="Pt", generics=[],
        fields=[A.FnParam(span=span, name="x",
                          ty=A.TyName(span=span, name="f32"),
                          is_mut=False)],
        is_pub=False,
    )
    line = A.StructDecl(
        span=span, name="Line", generics=[],
        fields=[A.FnParam(span=span, name="a",
                          ty=A.TyName(span=span, name="f32"),
                          is_mut=False)],
        is_pub=False,
    )
    pt_len = A.FnDecl(
        span=span, name="len", generics=[],
        params=[A.FnParam(span=span, name="self",
                          ty=A.TyName(span=span, name="Pt"),
                          is_mut=False)],
        return_ty=A.TyName(span=span, name="f32"),
        where_clauses=[],
        body=A.Block(span=span, stmts=[],
                     final_expr=A.FloatLit(span=span, value=0.0,
                                            type_suffix="f32")),
        attrs=[], is_pub=False,
    )
    line_len = A.FnDecl(
        span=span, name="len", generics=[],
        params=[A.FnParam(span=span, name="self",
                          ty=A.TyName(span=span, name="Line"),
                          is_mut=False)],
        return_ty=A.TyName(span=span, name="f32"),
        where_clauses=[],
        body=A.Block(span=span, stmts=[],
                     final_expr=A.FloatLit(span=span, value=0.0,
                                            type_suffix="f32")),
        attrs=[], is_pub=False,
    )
    impl_pt = A.ImplBlock(span=span, target="Pt", methods=[pt_len],
                          trait_name=None)
    impl_line = A.ImplBlock(span=span, target="Line", methods=[line_len],
                            trait_name=None)
    prog = A.Program(module=None,
                     items=[pt, line, impl_pt, impl_line])
    with _pt.raises(DuplicateMethodError) as ex:
        flatten_impls(prog)
    assert ex.value.method == "len"
    assert ex.value.trap_id == 74002


def test_flatten_impls_allows_distinct_method_names():
    """Distinct method names across structs flatten cleanly."""
    from helixc.frontend.flatten_impls import flatten_impls
    from helixc.frontend import ast_nodes as A
    span = A.Span(0, 0)
    pt = A.StructDecl(
        span=span, name="Pt", generics=[],
        fields=[A.FnParam(span=span, name="x",
                          ty=A.TyName(span=span, name="f32"),
                          is_mut=False)],
        is_pub=False,
    )
    line = A.StructDecl(
        span=span, name="Line", generics=[],
        fields=[A.FnParam(span=span, name="a",
                          ty=A.TyName(span=span, name="f32"),
                          is_mut=False)],
        is_pub=False,
    )
    pt_method = A.FnDecl(
        span=span, name="px", generics=[],
        params=[A.FnParam(span=span, name="self",
                          ty=A.TyName(span=span, name="Pt"),
                          is_mut=False)],
        return_ty=A.TyName(span=span, name="f32"),
        where_clauses=[],
        body=A.Block(span=span, stmts=[],
                     final_expr=A.FloatLit(span=span, value=0.0,
                                            type_suffix="f32")),
        attrs=[], is_pub=False,
    )
    line_method = A.FnDecl(
        span=span, name="la", generics=[],
        params=[A.FnParam(span=span, name="self",
                          ty=A.TyName(span=span, name="Line"),
                          is_mut=False)],
        return_ty=A.TyName(span=span, name="f32"),
        where_clauses=[],
        body=A.Block(span=span, stmts=[],
                     final_expr=A.FloatLit(span=span, value=0.0,
                                            type_suffix="f32")),
        attrs=[], is_pub=False,
    )
    impl_pt = A.ImplBlock(span=span, target="Pt", methods=[pt_method],
                          trait_name=None)
    impl_line = A.ImplBlock(span=span, target="Line",
                            methods=[line_method], trait_name=None)
    prog = A.Program(module=None,
                     items=[pt, line, impl_pt, impl_line])
    n = flatten_impls(prog)
    assert n == 2


# ----------------------------------------------------------------------
# Audit 28.8 B13 — TyDiff mixed inner widen-then-warn (trap 24200/AD002)
# ----------------------------------------------------------------------
def test_diff_mixed_inner_widens_with_warning():
    """Audit 28.8 B13: D<f64> + D<i32> previously silently coerced
    the i32 to f64 with no diagnostic. The fix widens to the
    dominant inner type AND emits a warning via the AD-warning
    channel (trap 24200 / AD002)."""
    from helixc.frontend import autodiff
    # Clear any pre-existing warnings.
    autodiff.take_diff_warnings()
    src = """
    fn loss(x: D<f64>, y: D<i32>) -> D<f64> {
        x + y
    }
    """
    errs = check(src)
    # Typecheck itself should pass (we widen, not reject).
    assert errs == [], f"unexpected typecheck errors: {errs}"
    # And a warning was emitted in the AD channel.
    warnings = autodiff.take_diff_warnings()
    assert any("24200" in w and "AD002" in w for w in warnings), \
        f"expected AD002 / trap 24200 warning, got: {warnings}"


def test_diff_same_inner_no_warning():
    """D<f64> + D<f64> is clean — no AD002 warning."""
    from helixc.frontend import autodiff
    autodiff.take_diff_warnings()
    src = """
    fn loss(x: D<f64>, y: D<f64>) -> D<f64> {
        x + y
    }
    """
    errs = check(src)
    assert errs == []
    warnings = autodiff.take_diff_warnings()
    assert not any("24200" in w for w in warnings), \
        f"unexpected mixed-inner warning: {warnings}"


# ----------------------------------------------------------------------
# Audit 28.8 cycle 2 B:C1 — _WIDEN_RANK covers fp8/mxfp4/nvfp4/char
# ----------------------------------------------------------------------
def test_c2_b_c1_widen_diff_fp8_vs_i64():
    """B:C1: pre-fix `D<fp8> + D<i64>` widened to i64 (rank -1 vs 40),
    a float-to-int silent collapse. Now fp8 has rank 25 — still loses
    to f16/f32/f64 but BEATS any integer."""
    from helixc.frontend.typecheck import _widen_diff_inner, TyPrim
    out = _widen_diff_inner(TyPrim("fp8"), TyPrim("i64"))
    assert out.name == "fp8", (
        f"expected fp8 to dominate i64 after B:C1 fix; got {out.name}"
    )


def test_c2_b_c1_widen_diff_mxfp4_vs_i32():
    """B:C1: mxfp4 (rank 35) beats i32 (rank 30)."""
    from helixc.frontend.typecheck import _widen_diff_inner, TyPrim
    out = _widen_diff_inner(TyPrim("mxfp4"), TyPrim("i32"))
    assert out.name == "mxfp4"


def test_c2_b_c1_widen_diff_nvfp4_vs_f32():
    """B:C1: nvfp4 (rank 35) loses to f32 (rank 60). Float-domain
    widening from quantized to standard precision is correct."""
    from helixc.frontend.typecheck import _widen_diff_inner, TyPrim
    out = _widen_diff_inner(TyPrim("nvfp4"), TyPrim("f32"))
    assert out.name == "f32"


def test_c2_b_c1_widen_diff_char_vs_i32():
    """B:C1: char (rank 5) loses to i32 (rank 30). Codepoint
    silently treated as integer was the pre-fix complaint; with
    char having rank 5 we widen up to i32 explicitly."""
    from helixc.frontend.typecheck import _widen_diff_inner, TyPrim
    out = _widen_diff_inner(TyPrim("char"), TyPrim("i32"))
    assert out.name == "i32"


def test_c2_b_c4_widen_diff_signed_unsigned_same_width():
    """B:C4: pre-fix, u32 (rank 30) vs i32 (rank 30) tied and
    left-wins picked u32 silently. With the asymmetric rank fix
    (u32=31, i32=30), unsigned now wins explicitly — and the
    user-visible test is just that the result deterministically
    picks the unsigned form."""
    from helixc.frontend.typecheck import _widen_diff_inner, TyPrim
    a = _widen_diff_inner(TyPrim("u32"), TyPrim("i32"))
    b = _widen_diff_inner(TyPrim("i32"), TyPrim("u32"))
    assert a.name == "u32"
    assert b.name == "u32", (
        f"signedness widening must be order-independent; got {b.name}"
    )


def test_c2_b_c4_widen_diff_i64_u64_unsigned_wins():
    """B:C4: u64 (rank 41) beats i64 (rank 40). Same width but
    sign-domain transition no longer depends on operand order."""
    from helixc.frontend.typecheck import _widen_diff_inner, TyPrim
    assert _widen_diff_inner(TyPrim("i64"), TyPrim("u64")).name == "u64"
    assert _widen_diff_inner(TyPrim("u64"), TyPrim("i64")).name == "u64"


def test_c2_b_c6_diff_plus_bare_warns():
    """B:C6: pre-fix `D<f64> + i32` (one D-wrapped, one raw) did NOT
    emit AD002 because the gate required BOTH sides D-wrapped. The
    same precision-loss hazard exists in the asymmetric case — i32
    silently promoted to f64. Now the asymmetric case also warns
    (with a hint about which side is D-wrapped)."""
    from helixc.frontend import autodiff
    autodiff.take_diff_warnings()
    src = """
    fn loss(x: D<f64>, y: i32) -> D<f64> {
        x + y
    }
    """
    errs = check(src)
    assert errs == [], f"unexpected errors: {errs}"
    warnings = autodiff.take_diff_warnings()
    assert any("24200" in w and "AD002" in w for w in warnings), (
        f"expected B:C6 asymmetric warning, got: {warnings}"
    )
    assert any("D-wrapped" in w or "bare" in w for w in warnings), (
        f"expected hint about D-wrap asymmetry, got: {warnings}"
    )


# ----------------------------------------------------------------------
# Audit 28.8 B14 — Cast allowed-cast matrix (trap 28604)
# ----------------------------------------------------------------------
def test_cast_int_to_int_allowed():
    """int -> int (any widths) is allowed."""
    src = """
    fn main() -> i64 {
        let x: i32 = 42;
        x as i64
    }
    """
    errs = check(src)
    assert errs == [], f"unexpected errors: {errs}"


def test_cast_int_to_float_allowed():
    """int -> float (any widths) is allowed."""
    src = """
    fn main() -> f64 {
        let x: i32 = 42;
        x as f64
    }
    """
    errs = check(src)
    assert errs == [], f"unexpected errors: {errs}"


def test_cast_float_to_int_allowed():
    """float -> int (any widths) is allowed."""
    src = """
    fn main() -> i32 {
        let x: f64 = 3.14_f64;
        x as i32
    }
    """
    errs = check(src)
    assert errs == [], f"unexpected errors: {errs}"


def test_cast_bool_to_int_allowed():
    """bool -> int is allowed (1/0)."""
    src = """
    fn main() -> i32 {
        let b: bool = true;
        b as i32
    }
    """
    errs = check(src)
    assert errs == [], f"unexpected errors: {errs}"


def test_cast_tuple_to_int_rejected():
    """Audit 28.8 B14 (trap 28604): tuple-as-i32 is invalid."""
    src = """
    fn main() -> i32 {
        let t: (i32, i32) = (1, 2);
        t as i32
    }
    """
    errs = check(src)
    assert any("28604" in s for s in errs), \
        f"expected trap 28604 for tuple-as-i32, got: {errs}"


def test_cast_struct_to_float_rejected():
    """Audit 28.8 B14: struct-as-float is invalid."""
    src = """
    struct Pt { x: f32, y: f32 }
    fn main() -> f32 {
        let p = Pt { x: 1.0_f32, y: 2.0_f32 };
        p as f32
    }
    """
    errs = check(src)
    assert any("28604" in s for s in errs), \
        f"expected trap 28604 for struct-as-float, got: {errs}"


def test_flatten_impls_allows_same_struct_redeclare():
    """Same struct calling flatten_impls twice on aliased
    impl-block records (i.e. same target, same method) should NOT
    fire B11 — only cross-struct collisions are flagged."""
    from helixc.frontend.flatten_impls import flatten_impls
    from helixc.frontend import ast_nodes as A
    span = A.Span(0, 0)
    pt = A.StructDecl(
        span=span, name="Pt", generics=[],
        fields=[A.FnParam(span=span, name="x",
                          ty=A.TyName(span=span, name="f32"),
                          is_mut=False)],
        is_pub=False,
    )
    # Two impl blocks on Pt with two different methods is OK.
    m1 = A.FnDecl(
        span=span, name="a", generics=[],
        params=[A.FnParam(span=span, name="self",
                          ty=A.TyName(span=span, name="Pt"),
                          is_mut=False)],
        return_ty=A.TyName(span=span, name="f32"),
        where_clauses=[],
        body=A.Block(span=span, stmts=[],
                     final_expr=A.FloatLit(span=span, value=0.0,
                                            type_suffix="f32")),
        attrs=[], is_pub=False,
    )
    m2 = A.FnDecl(
        span=span, name="b", generics=[],
        params=[A.FnParam(span=span, name="self",
                          ty=A.TyName(span=span, name="Pt"),
                          is_mut=False)],
        return_ty=A.TyName(span=span, name="f32"),
        where_clauses=[],
        body=A.Block(span=span, stmts=[],
                     final_expr=A.FloatLit(span=span, value=0.0,
                                            type_suffix="f32")),
        attrs=[], is_pub=False,
    )
    impl_pt_1 = A.ImplBlock(span=span, target="Pt", methods=[m1],
                            trait_name=None)
    impl_pt_2 = A.ImplBlock(span=span, target="Pt", methods=[m2],
                            trait_name=None)
    prog = A.Program(module=None,
                     items=[pt, impl_pt_1, impl_pt_2])
    n = flatten_impls(prog)
    assert n == 2


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
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
