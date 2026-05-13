"""Stage 28.8.2 — tests for the shared AST walker library.

Background: cycles 1-11 caught ~20 attribute-list drift bugs across
panic_pass, unsafe_pass, deprecated_pass, grad_pass._expr_has_grad,
and struct_mono.visit_expr. The shared ``ast_walker.ASTVisitor`` base
class replaces those hand-rolled walkers with a single dataclass-
introspecting traversal.

These tests pin the contract:
  * generic_visit reaches every Expr child via dataclass-field
    introspection (no per-pass attribute list to drift).
  * Specifically the cycle-1 drift cases — For.iter_expr, Match arm
    guards, Index.indices, Range.start/end, etc. — are reached.
  * Returning False from a visit_X override stops recursion into that
    subtree (skip-marker contract).
  * Visit dispatch picks the type-specific override when present.

License: Apache 2.0
"""

from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from helixc.frontend import ast_nodes as A
from helixc.frontend.ast_walker import ASTVisitor
from helixc.frontend.parser import parse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _RecordingVisitor(ASTVisitor):
    """Visitor that records the type name of every node visited.
    Used to assert which nodes the walker reaches for a given source.
    """

    def __init__(self) -> None:
        self.visited: list[str] = []

    def generic_visit(self, node):
        self.visited.append(type(node).__name__)
        super().generic_visit(node)


def _visit_all(src: str) -> list[str]:
    """Parse, walk, return the list of visited node type names."""
    prog = parse(src, include_stdlib=False)
    v = _RecordingVisitor()
    for item in prog.items:
        v.visit(item)
    return v.visited


# ---------------------------------------------------------------------------
# Cycle-1 drift cases — must be reached by generic_visit
# ---------------------------------------------------------------------------

def test_walker_visits_for_iter_expr():
    """Stage 28.8.2: For.iter_expr is a cycle-1 drift miss site.
    Each pass's hand-rolled walker forgot to include `iter_expr` in
    its attr list until cycle 1 fix-sweeps; the shared library
    reaches it via dataclass-field introspection."""
    src = "fn f() -> i32 { for x in 0..10 { x; }; 0 }"
    visited = _visit_all(src)
    assert "For" in visited
    assert "Range" in visited, f"Range (iter_expr) not visited: {visited}"


def test_walker_visits_match_arm_guard():
    """Stage 28.8.2: MatchArm.guard is a cycle-1 drift miss site.
    Each pass's walker silently dropped guard exprs until cycles 1-2
    fix-sweeps; the shared library reaches them."""
    src = "fn f() -> i32 { match 0 { x if x > 0 => 1, _ => 0 } }"
    visited = _visit_all(src)
    assert "Match" in visited
    assert "MatchArm" in visited
    # The guard `x > 0` is a Binary — reachable via MatchArm.guard.
    assert "Binary" in visited, f"Match arm guard not walked: {visited}"


def test_walker_visits_index_indices():
    """Stage 28.8.2: Index.indices is a cycle-1 drift miss site.
    `arr[expr]` placed inside any other walker-host meant the inner
    expr was silently invisible. The shared library walks the list."""
    src = "fn f(arr: [i32; 4]) -> i32 { arr[2 + 1] }"
    visited = _visit_all(src)
    assert "Index" in visited
    # `2 + 1` is a Binary inside Index.indices — must be reached.
    assert "Binary" in visited, (
        f"Index.indices inner Binary not walked: {visited}"
    )


def test_walker_visits_range_start_end():
    """Stage 28.8.2: Range.start / Range.end are cycle-1 drift sites.
    Used in `for i in 0..10` and standalone slice exprs."""
    src = "fn f() -> i32 { let _r = 0..(2 + 3); 0 }"
    visited = _visit_all(src)
    assert "Range" in visited
    # `2 + 3` is a Binary inside Range.end — must be reached.
    assert "Binary" in visited, (
        f"Range.end inner Binary not walked: {visited}"
    )


def test_walker_visits_modify_target_transformation_verifier():
    """Stage 28.8.2: Modify has three Expr children (target,
    transformation, verifier). Cycle-2 found these were silently
    skipped because the hand-rolled walkers had only `target` in
    the attr list. Shared library reaches all three via field
    introspection."""
    # Modify isn't easily expressed in source syntax — synthesise
    # an AST manually.
    span = A.Span(line=1, col=1)
    inner_a = A.IntLit(span=span, value=1, type_suffix=None)
    inner_b = A.IntLit(span=span, value=2, type_suffix=None)
    inner_c = A.IntLit(span=span, value=3, type_suffix=None)
    modify = A.Modify(span=span, target=inner_a,
                      transformation=inner_b, verifier=inner_c)
    v = _RecordingVisitor()
    v.visit(modify)
    # All three IntLit children must be visited.
    assert v.visited.count("IntLit") == 3, (
        f"Modify children not all visited: {v.visited}"
    )


def test_walker_visits_struct_lit_field_exprs():
    """Stage 28.8.2: StructLit.fields is list[tuple[str, Expr]]. The
    walker must descend into the tuple's Expr element. Cycle-2 found
    several walkers silently dropped these."""
    src = (
        "struct Pt { x: i32, y: i32 }\n"
        "fn f() -> i32 { let _p = Pt { x: 1 + 2, y: 3 }; 0 }"
    )
    visited = _visit_all(src)
    assert "StructLit" in visited
    # `1 + 2` inside the x field — must be reached.
    assert "Binary" in visited, (
        f"StructLit field expr not walked: {visited}"
    )


# ---------------------------------------------------------------------------
# Skip-marker contract
# ---------------------------------------------------------------------------

def test_walker_skip_marker_stops_descent():
    """Stage 28.8.2: returning False from visit_X must stop recursion
    into that subtree."""

    class _SkipVisitor(ASTVisitor):
        def __init__(self) -> None:
            self.visited: list[str] = []

        def visit_Binary(self, node):
            self.visited.append("Binary(skipped)")
            return False  # don't recurse into left/right

        def generic_visit(self, node):
            self.visited.append(type(node).__name__)
            super().generic_visit(node)

    span = A.Span(line=1, col=1)
    inner_l = A.IntLit(span=span, value=1, type_suffix=None)
    inner_r = A.IntLit(span=span, value=2, type_suffix=None)
    binary = A.Binary(span=span, op="+", left=inner_l, right=inner_r)
    v = _SkipVisitor()
    v.visit(binary)
    # Binary visited (override fired); inner IntLits skipped (return False).
    assert "Binary(skipped)" in v.visited
    assert "IntLit" not in v.visited, (
        f"skip-marker did not stop descent: {v.visited}"
    )


def test_walker_dispatch_picks_specific_override():
    """Stage 28.8.2: visit() must dispatch to visit_<TypeName> when
    present, falling back to generic_visit otherwise."""

    class _DispatchVisitor(ASTVisitor):
        def __init__(self) -> None:
            self.binary_seen = 0
            self.intlit_seen = 0

        def visit_Binary(self, node):
            self.binary_seen += 1

        def visit_IntLit(self, node):
            self.intlit_seen += 1

    span = A.Span(line=1, col=1)
    inner_l = A.IntLit(span=span, value=1, type_suffix=None)
    inner_r = A.IntLit(span=span, value=2, type_suffix=None)
    binary = A.Binary(span=span, op="+", left=inner_l, right=inner_r)
    v = _DispatchVisitor()
    v.visit(binary)
    assert v.binary_seen == 1
    assert v.intlit_seen == 2  # one per IntLit child


# ---------------------------------------------------------------------------
# Type-field skipping (default behaviour)
# ---------------------------------------------------------------------------

def test_walker_default_skips_type_fields():
    """Stage 28.8.2: by default the walker does NOT descend into type
    fields (Cast.target_ty, Let.ty). The pre-fix walkers all skipped
    these — visitors that need to walk type fields should override
    the relevant visit_X method explicitly."""
    src = "fn f() -> i32 { let x: i32 = 0; x }"
    visited = _visit_all(src)
    # No TyName / TyNode in the visited list (the i32 annotation is a
    # TyName that lives in Let.ty — should be skipped by default).
    for name in visited:
        assert not name.startswith("Ty"), (
            f"unexpected type-field walk: visited {name}: {visited}"
        )


# ---------------------------------------------------------------------------
# Full traversal sanity
# ---------------------------------------------------------------------------

def test_walker_covers_typical_program():
    """Stage 28.8.2: a typical program with let, if, match, for,
    binary, calls — assert every Expr subtype used is reached."""
    src = (
        "fn helper(x: i32) -> i32 { x + 1 }\n"
        "fn main() -> i32 {\n"
        "    let mut sum = 0;\n"
        "    for i in 0..10 {\n"
        "        if i > 5 {\n"
        "            sum = sum + helper(i);\n"
        "        }\n"
        "    }\n"
        "    match sum { 0 => 0, _ => sum }\n"
        "}\n"
    )
    visited = _visit_all(src)
    for expected in ("FnDecl", "Block", "Let", "For", "Range", "If",
                     "Binary", "Assign", "Call", "Match", "MatchArm",
                     "Name", "IntLit"):
        assert expected in visited, (
            f"expected to visit {expected}; visited={set(visited)}"
        )
