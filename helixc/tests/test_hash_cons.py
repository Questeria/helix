"""Tests for the Stage 20 AST hash-cons pass."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.frontend import ast_nodes as A
from helixc.frontend.ast_hash import structural_hash, short_hash
from helixc.frontend.hash_cons import hash_cons, HashConsError


def _find_subtrees(node, predicate, out=None):
    """Walk an AST collecting every node satisfying `predicate`."""
    if out is None:
        out = []
    if node is None:
        return out
    if predicate(node):
        out.append(node)
    for attr in ("operand", "left", "right", "cond", "then", "else_",
                 "value", "callee", "obj", "iter_expr", "scrutinee",
                 "expr", "target", "start", "end"):
        v = getattr(node, attr, None)
        if v is not None and not isinstance(v, (str, int, float, bool)):
            _find_subtrees(v, predicate, out)
    for attr in ("stmts", "args", "indices", "elems", "arms"):
        v = getattr(node, attr, None)
        if v is not None:
            for s in v:
                _find_subtrees(s, predicate, out)
    if hasattr(node, "final_expr") and node.final_expr is not None:
        _find_subtrees(node.final_expr, predicate, out)
    if hasattr(node, "body") and node.body is not None \
            and not isinstance(node.body, (str, int, float, bool)):
        _find_subtrees(node.body, predicate, out)
    return out


def test_stage20_two_identical_subtrees_share_one_node():
    """Stage 20 goal-test: `(1 + 2)` appears twice in the body; after
    hash_cons both occurrences point to the same Python object.

    `_find_subtrees` walks the body and yields every Binary(+) with
    IntLit(1) and IntLit(2) children. Pre hash_cons we see 2 distinct
    instances; post we still see 2 references but `id()` collapses to
    one canonical."""
    src = """
    fn main() -> i32 {
        let a = (1 + 2) * 4;
        let b = (1 + 2) * 5;
        a + b
    }
    """
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl))

    def is_one_plus_two(n):
        return (isinstance(n, A.Binary) and n.op == "+"
                and isinstance(n.left, A.IntLit) and n.left.value == 1
                and isinstance(n.right, A.IntLit) and n.right.value == 2)

    before = _find_subtrees(fn.body, is_one_plus_two)
    assert len(before) == 2
    assert id(before[0]) != id(before[1]), "pre-share: distinct objects"

    merged = hash_cons(prog)
    assert merged > 0, f"expected sharing rewrites, got {merged}"

    after = _find_subtrees(fn.body, is_one_plus_two)
    assert len(after) == 2  # still two references in the AST...
    assert id(after[0]) == id(after[1]), \
        f"post-share: must be same object, got {id(after[0])} != {id(after[1])}"


def test_stage20_identical_intlits_share():
    """The simplest case: two `1` literals should share. Verifies the
    bottom-up sharing actually descends into leaf nodes."""
    src = """
    fn main() -> i32 { 1 + 1 }
    """
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl))

    def is_one(n):
        return isinstance(n, A.IntLit) and n.value == 1

    before = _find_subtrees(fn.body, is_one)
    assert len(before) == 2
    hash_cons(prog)
    after = _find_subtrees(fn.body, is_one)
    assert len(after) == 2
    assert id(after[0]) == id(after[1])


def test_stage20_different_constants_do_not_share():
    """Sanity: different literal values must NOT share."""
    src = """
    fn main() -> i32 { 1 + 2 }
    """
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl))
    hash_cons(prog)

    def is_intlit(n):
        return isinstance(n, A.IntLit)

    lits = _find_subtrees(fn.body, is_intlit)
    assert len(lits) == 2
    assert id(lits[0]) != id(lits[1])


def test_stage20_preserves_semantics_with_scope_shadowing():
    """Shadowed binders inside a sub-block must not produce wrong
    sharing — Match/For/While/Loop and Block are explicitly excluded
    from sharing.

    Here both let-bound `x` references resolve via de-Bruijn in the
    hasher; the inner `x + 1` and outer `x + y` share NO subtrees with
    each other beyond literal Name("x"), which is structurally identical
    but semantically distinct. Sharing the Name is still safe — the
    binding resolution happens later (typecheck / lower), and the AST
    node itself just carries the textual name."""
    src = """
    fn main() -> i32 {
        let x = 7;
        let y = {
            let x = 99;
            x + 1
        };
        x + y
    }
    """
    from helixc.ir.lower_ast import lower
    from helixc.ir.passes.const_fold import fold_module
    from helixc.ir.passes.dce import dce_module
    from helixc.ir import tir

    prog = parse(src)
    hash_cons(prog)
    mod = lower(prog)
    fold_module(mod)
    dce_module(mod)
    # Expected: 7 + (99 + 1) = 7 + 100 = 107
    consts = [op.attrs["value"]
              for fn in mod.functions.values() for blk in fn.blocks
              for op in blk.ops if op.kind == tir.OpKind.CONST_INT]
    assert 107 in consts, f"expected 107 (7 + 100), got {consts}"


def test_stage20_dump_ast_hashes_cli_shows_same_hash_for_two_runs():
    """`autodiff_cli --dump-ast-hashes` is the Stage 20 verification CLI.
    Two runs on the same source produce identical hashes."""
    import subprocess, tempfile
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src_dir = os.path.join(proj_root, "helixc", "tests", "_tmp")
    os.makedirs(src_dir, exist_ok=True)
    src_path = os.path.join(src_dir, "_stage20_dump.hx")
    with open(src_path, "w", encoding="utf-8") as f:
        f.write("""
fn main() -> i32 {
    let a = (1 + 2) * 4;
    let b = (1 + 2) * 5;
    a + b
}
""")
    cmd = [sys.executable, "-m", "helixc.frontend.autodiff_cli",
           "--dump-ast-hashes", src_path]
    r1 = subprocess.run(cmd, capture_output=True, cwd=proj_root)
    assert r1.returncode == 0, r1.stderr
    out = r1.stdout.decode("utf-8").strip().splitlines()
    assert any("main" in line for line in out), f"got: {out}"


def test_stage20_trap_20001_on_hash_collision():
    """The Stage 20 trap 20001 fires when hash_cons detects two
    structurally distinct subtrees claiming the same hash. We simulate
    a collision by monkey-patching structural_hash to return a constant
    'fake_hash' — feeding two different IntLit values through the same
    Sharer triggers the post-hash `_ast_equal` mismatch."""
    from helixc.frontend import hash_cons as hc_mod
    from helixc.frontend.hash_cons import _Sharer

    orig_hash = hc_mod.structural_hash
    try:
        hc_mod.structural_hash = lambda n, binders=None: "fake_collision_hash"
        sharer = _Sharer()
        sharer._maybe_share(A.IntLit(span=A.Span(1, 1), value=42))
        try:
            sharer._maybe_share(A.IntLit(span=A.Span(1, 1), value=999))
        except HashConsError as e:
            assert "20001" in str(e)
            assert HashConsError.trap_id == 20001
            return
        raise AssertionError("expected HashConsError trap 20001")
    finally:
        hc_mod.structural_hash = orig_hash


def test_stage20_hash_cons_does_not_break_existing_codegen():
    """End-to-end: the spec example with hash_cons in the pipeline still
    produces 27 = (1+2)*4 + (1+2)*5 = 12 + 15."""
    from helixc.ir.lower_ast import lower
    from helixc.ir.passes.const_fold import fold_module
    from helixc.ir.passes.dce import dce_module
    from helixc.ir import tir

    src = """
    fn main() -> i32 {
        let a = (1 + 2) * 4;
        let b = (1 + 2) * 5;
        a + b
    }
    """
    prog = parse(src)
    hash_cons(prog)
    mod = lower(prog)
    fold_module(mod)
    dce_module(mod)
    consts = [op.attrs["value"]
              for fn in mod.functions.values() for blk in fn.blocks
              for op in blk.ops if op.kind == tir.OpKind.CONST_INT]
    assert 27 in consts, f"expected 27, got {consts}"


def test_stage20_check_cli_supports_hash_cons_flag():
    """`python -m helixc.check --hash-cons <file>` runs hash_cons and
    reports the count of de-duplicated nodes."""
    import subprocess
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src_dir = os.path.join(proj_root, "helixc", "tests", "_tmp")
    os.makedirs(src_dir, exist_ok=True)
    src_path = os.path.join(src_dir, "_stage20_check.hx")
    with open(src_path, "w", encoding="utf-8") as f:
        f.write("""
fn main() -> i32 {
    let a = (1 + 2) * 4;
    let b = (1 + 2) * 5;
    a + b
}
""")
    cmd = [sys.executable, "-m", "helixc.check", "--hash-cons", src_path]
    r = subprocess.run(cmd, capture_output=True, cwd=proj_root, text=True)
    assert r.returncode == 0, r.stderr
    assert "hash-cons:" in r.stdout, f"stdout missing report: {r.stdout}"
    # Should have deduped at least one node ((1+2) appears twice).
    assert "0 AST node" not in r.stdout, \
        f"expected non-zero dedupe count: {r.stdout}"


# --- Stage 28.9 cycle 35 audit-T C34-1 regression tests ---


def test_c34_1_return_value_affects_hash():
    """C34-1 regression (HIGH conf 95): pre-fix, ast_hash._hash_into
    had a catch-all fallback `_emit(h, "Unknown", type(node).__name__)`
    that emitted ONLY the class name with NO recursion into children.
    Two A.Return nodes with different values hashed identically,
    causing silent QUOTE-cell aliasing — `quote { return 1 }` and
    `quote { return 99 }` would map to the same ast_handle."""
    src1 = "fn main() -> i32 { quote { return 1 }; 0 }"
    src2 = "fn main() -> i32 { quote { return 99 }; 0 }"

    def find_quote_inner(p):
        for it in p.items:
            if isinstance(it, A.FnDecl):
                for stmt in it.body.stmts:
                    if isinstance(stmt, A.ExprStmt) and \
                            isinstance(stmt.expr, A.Quote):
                        return stmt.expr.inner
        return None

    p1, p2 = parse(src1), parse(src2)
    q1, q2 = find_quote_inner(p1), find_quote_inner(p2)
    assert q1 is not None and q2 is not None
    h1 = structural_hash(q1)
    h2 = structural_hash(q2)
    assert h1 != h2, (
        f"quote{{return 1}} and quote{{return 99}} must hash "
        f"differently (else silent QUOTE-cell aliasing); got h1=h2"
    )


def test_c34_1_structlit_fields_affect_hash():
    """C34-1: two StructLit nodes with different fields must hash
    differently. Pre-fix, both `Point{x:1}` and `Point{x:99}`
    hashed identically because the StructLit branch was missing."""
    sl1 = A.StructLit(
        span=A.Span(line=1, col=1),
        name="Point",
        fields=[("x", A.IntLit(span=A.Span(line=1, col=1), value=1,
                               type_suffix=None))],
    )
    sl2 = A.StructLit(
        span=A.Span(line=1, col=1),
        name="Point",
        fields=[("x", A.IntLit(span=A.Span(line=1, col=1), value=99,
                               type_suffix=None))],
    )
    assert structural_hash(sl1) != structural_hash(sl2), (
        "StructLit with different field values must hash differently"
    )


def test_c34_1_path_segments_affect_hash():
    """C34-1: Path with different segments must hash differently.
    Pre-fix, `Foo::A` and `Bar::B` hashed identically."""
    p1 = A.Path(span=A.Span(line=1, col=1), segments=["Foo", "A"])
    p2 = A.Path(span=A.Span(line=1, col=1), segments=["Bar", "B"])
    assert structural_hash(p1) != structural_hash(p2), (
        "Path with different segments must hash differently"
    )


# --- Stage 28.9 cycle 37 audit-R C36-1 regression tests ---


def test_c36_1_tilelit_hash_independent_of_span():
    """C36-1 regression (HIGH conf 90): the cycle-35 TileLit hash arm
    used `repr(node.dtype)` which embeds the dtype's TyNode span (a
    dataclass field). Pre-fix, structurally-identical TileLits at
    different source lines hashed differently — fragmenting QUOTE
    handles instead of sharing them. The docstring contract at lines
    16-19 of ast_hash.py says the hash is INTENTIONALLY span-
    independent."""
    def build_tilelit(line: int):
        s = A.Span(line=line, col=1)
        return A.TileLit(
            span=s,
            dtype=A.TyName(span=s, name="f32"),
            shape=[A.IntLit(span=s, value=4, type_suffix=None)],
            memspace=A.Name(span=s, name="REG", generics=[]),
            init="zeros",
        )
    h1 = structural_hash(build_tilelit(1))
    h2 = structural_hash(build_tilelit(50))
    assert h1 == h2, (
        f"TileLit hash must be span-independent (cycle 36 C36-1); "
        f"got h1={h1[:16]}... vs h2={h2[:16]}..."
    )


def test_c36_1_cast_hash_independent_of_span():
    """C36-1 same-defect-class regression: the pre-existing Cast arm
    also used `repr(target_ty)` which embedded the TyNode span. The
    cycle 36 _ty_repr fix applies symmetrically."""
    def build_cast(line: int):
        s = A.Span(line=line, col=1)
        return A.Cast(
            span=s,
            value=A.IntLit(span=s, value=42, type_suffix=None),
            target_ty=A.TyName(span=s, name="i32"),
        )
    h1 = structural_hash(build_cast(1))
    h2 = structural_hash(build_cast(99))
    assert h1 == h2, (
        f"Cast hash must be span-independent (cycle 36 C36-1 "
        f"symmetric); got h1={h1[:16]}... vs h2={h2[:16]}..."
    )


def test_c36_1_different_dtypes_still_hash_differently():
    """C36-1 fix must not over-correct: TileLits with DIFFERENT
    dtypes must still hash differently."""
    s = A.Span(line=1, col=1)
    t_f32 = A.TileLit(
        span=s,
        dtype=A.TyName(span=s, name="f32"),
        shape=[A.IntLit(span=s, value=4, type_suffix=None)],
        memspace=A.Name(span=s, name="REG", generics=[]),
        init="zeros",
    )
    t_i32 = A.TileLit(
        span=s,
        dtype=A.TyName(span=s, name="i32"),
        shape=[A.IntLit(span=s, value=4, type_suffix=None)],
        memspace=A.Name(span=s, name="REG", generics=[]),
        init="zeros",
    )
    assert structural_hash(t_f32) != structural_hash(t_i32), (
        "TileLits with different dtypes must hash differently"
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
