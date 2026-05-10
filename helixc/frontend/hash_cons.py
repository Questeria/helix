"""
helixc/frontend/hash_cons.py — Phase-0 AST hash-cons (structural sharing).

After parsing, the AST contains many subtrees that are structurally
identical but live as distinct Python objects (e.g., the literal `1`
appears as a different `IntLit(1)` instance every time it's written in
the source). Hash-consing replaces structurally-equivalent subtrees
with a SINGLE shared Python object, so `id(left.left)` and
`id(right.left)` are the same when `left` and `right` both contain
`(1 + 2)`.

The Stage 20 spec example:

    fn main() -> i32 {
        let a = (1 + 2) * 4;
        let b = (1 + 2) * 5;
        a + b
    }

After hash_cons, both `(1 + 2)` sub-ASTs point to the same Binary
node, observable via `id()` and via `--dump-ast-hashes` showing both
copies hashing to the same value (the pre-share hash is unchanged
because hashing is content-addressed).

Phase-0 sharing rules:
  - Only hash-cons expressions, not types or fn decls.
  - Skip nodes that are scope-sensitive in a way our hash doesn't
    capture (Match arms with pattern binders; While/Loop bodies that
    side-effect through outer state). Practically: we always hash
    structural-only, so sharing is sound for *pure* expression
    subtrees. Statements remain unique.
  - Sharing does mutate the surviving subtree's span to the FIRST
    occurrence's span; diagnostics for a downstream pass that points at
    the second occurrence will show the first one's line/col. This is
    acceptable in Phase-0 because diagnostics already happen at the
    typecheck / totality stage which run BEFORE hash-cons.

The pass returns the count of nodes that were de-duplicated (i.e.
replaced with a reference to an earlier-seen object).

Trap-id 20001: hash collision detected. The structural hasher uses
SHA-256, which is collision-resistant — but we still cheap-check
post-share equivalence via `_ast_equal`, and raise on any false
positive so a future hash regression surfaces loudly.

License: Apache 2.0
"""

from __future__ import annotations

from typing import Any

from . import ast_nodes as A
from .ast_hash import structural_hash


class HashConsError(Exception):
    """Raised when hash-cons detects a hash collision (trap 20001)."""
    trap_id = 20001


# Node classes safe to hash-cons. Expressions that are pure-functions of
# their children (no implicit binders, no side-channel state) qualify.
# We keep Block/Match/For/While/Loop out — their semantics depend on
# scope and execution order, and sharing them would couple unrelated
# control-flow regions. (Spec intent is to share constant subtrees, so
# this conservatism is right.)
_SHAREABLE = (
    A.IntLit, A.FloatLit, A.BoolLit, A.StrLit, A.CharLit,
    A.Name,
    A.Unary, A.Binary,
    A.Call,
    A.If,           # If is functional in Helix (returns a value)
    A.Cast,
    A.TupleLit, A.ArrayLit,
    A.Field,
    A.Index,
    A.Range,
)


def hash_cons(prog: A.Program) -> int:
    """Walk every fn body in `prog` and de-duplicate structurally-equal
    sub-ASTs. Returns the count of sharing-rewrites made."""
    sharer = _Sharer()
    for item in prog.items:
        if isinstance(item, A.FnDecl):
            sharer.share_in(item.body)
    return sharer.merged


def _ast_equal(a: Any, b: Any) -> bool:
    """Structural equality on shareable AST nodes. Used as the
    second-stage check after hash-equality to defend against the (very
    unlikely) SHA-256 collision (trap 20001). Conservative: returns
    True only when classes match AND every child equals."""
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, A.IntLit):
        return a.value == b.value
    if isinstance(a, A.FloatLit):
        return a.value == b.value
    if isinstance(a, A.BoolLit):
        return a.value == b.value
    if isinstance(a, A.StrLit):
        return a.value == b.value
    if isinstance(a, A.CharLit):
        return a.value == b.value
    if isinstance(a, A.Name):
        return a.name == b.name
    if isinstance(a, A.Unary):
        return a.op == b.op and _ast_equal(a.operand, b.operand)
    if isinstance(a, A.Binary):
        return (a.op == b.op
                and _ast_equal(a.left, b.left)
                and _ast_equal(a.right, b.right))
    if isinstance(a, A.Call):
        if len(a.args) != len(b.args):
            return False
        if not _ast_equal(a.callee, b.callee):
            return False
        return all(_ast_equal(x, y) for x, y in zip(a.args, b.args))
    if isinstance(a, A.If):
        return (_ast_equal(a.cond, b.cond)
                and _ast_equal(a.then, b.then)
                and _ast_equal(a.else_, b.else_))
    if isinstance(a, A.Cast):
        return (_ast_equal(a.value, b.value)
                and repr(a.target_ty) == repr(b.target_ty))
    if isinstance(a, A.TupleLit):
        if len(a.elems) != len(b.elems):
            return False
        return all(_ast_equal(x, y) for x, y in zip(a.elems, b.elems))
    if isinstance(a, A.ArrayLit):
        if len(a.elems) != len(b.elems):
            return False
        return all(_ast_equal(x, y) for x, y in zip(a.elems, b.elems))
    if isinstance(a, A.Field):
        return a.name == b.name and _ast_equal(a.obj, b.obj)
    if isinstance(a, A.Index):
        if len(a.indices) != len(b.indices):
            return False
        if not _ast_equal(a.callee, b.callee):
            return False
        return all(_ast_equal(x, y) for x, y in zip(a.indices, b.indices))
    if isinstance(a, A.Range):
        return _ast_equal(a.start, b.start) and _ast_equal(a.end, b.end)
    # Conservative default: treat as unequal (don't share). Hash already
    # matched, so this only matters when we hit a class _ast_equal
    # forgot to enumerate.
    return False


class _Sharer:
    def __init__(self):
        self.merged: int = 0
        # Hash -> canonical Python object. The canonical is the first
        # occurrence of a subtree with that hash.
        self._canon: dict[str, Any] = {}

    def share_in(self, node: Any) -> Any:
        """In-place: descend into `node`, replacing each child reference
        with a shared canonical when one exists. Returns `node` (or its
        canonical if `node` itself is shareable and a canonical exists)."""
        if node is None:
            return None

        # Block — walk its statements and final_expr; don't share the
        # Block itself (scope-sensitive).
        if isinstance(node, A.Block):
            for i, st in enumerate(node.stmts):
                node.stmts[i] = self._share_stmt(st)
            if node.final_expr is not None:
                node.final_expr = self._maybe_share(node.final_expr)
            return node

        # Match / For / While / Loop — bodies are NOT shared (scope/CF).
        # We still walk into children for sub-expression sharing.
        if isinstance(node, A.Match):
            node.scrutinee = self._maybe_share(node.scrutinee)
            for arm in node.arms:
                if arm.guard is not None:
                    arm.guard = self._maybe_share(arm.guard)
                arm.body = self._maybe_share(arm.body)
            return node
        if isinstance(node, A.For):
            node.iter_expr = self._maybe_share(node.iter_expr)
            if isinstance(node.body, A.Block):
                self.share_in(node.body)
            else:
                node.body = self._maybe_share(node.body)
            return node
        if isinstance(node, A.While):
            node.cond = self._maybe_share(node.cond)
            if isinstance(node.body, A.Block):
                self.share_in(node.body)
            else:
                node.body = self._maybe_share(node.body)
            return node
        if isinstance(node, A.Loop):
            if isinstance(node.body, A.Block):
                self.share_in(node.body)
            else:
                node.body = self._maybe_share(node.body)
            return node

        # Everything else: try to share it directly.
        return self._maybe_share(node)

    # ------------------------------------------------------------------
    # Statement walker (Let / ConstStmt / ExprStmt / etc.)
    # ------------------------------------------------------------------
    def _share_stmt(self, stmt: Any) -> Any:
        # Let / ConstStmt: only the RHS expression is shareable.
        if isinstance(stmt, A.Let):
            if stmt.value is not None:
                stmt.value = self._maybe_share(stmt.value)
            return stmt
        if isinstance(stmt, A.ConstStmt):
            stmt.value = self._maybe_share(stmt.value)
            return stmt
        if isinstance(stmt, A.ExprStmt):
            stmt.expr = self._maybe_share(stmt.expr)
            return stmt
        # Other statement kinds (return, break, continue, assign): walk
        # children where they exist.
        for attr in ("expr", "value", "target"):
            v = getattr(stmt, attr, None)
            if v is not None and not isinstance(v, (str, int, float, bool)):
                setattr(stmt, attr, self._maybe_share(v))
        return stmt

    # ------------------------------------------------------------------
    # Expression sharer
    # ------------------------------------------------------------------
    def _maybe_share(self, node: Any) -> Any:
        """If `node` is shareable, descend into children, then look up
        its canonical by hash. Returns the canonical (or `node` if none
        existed, in which case `node` becomes the canonical)."""
        if node is None:
            return None
        if not isinstance(node, _SHAREABLE):
            # Walk in (Block / For / etc.) but don't share at this level.
            return self.share_in(node)

        # First, share all children — bottom-up so the hash reflects
        # already-shared sub-trees.
        if isinstance(node, A.Unary):
            node.operand = self._maybe_share(node.operand)
        elif isinstance(node, A.Binary):
            node.left = self._maybe_share(node.left)
            node.right = self._maybe_share(node.right)
        elif isinstance(node, A.Call):
            node.callee = self._maybe_share(node.callee)
            for i, a in enumerate(node.args):
                node.args[i] = self._maybe_share(a)
        elif isinstance(node, A.If):
            node.cond = self._maybe_share(node.cond)
            node.then = self._maybe_share(node.then)
            node.else_ = self._maybe_share(node.else_)
        elif isinstance(node, A.Cast):
            node.value = self._maybe_share(node.value)
        elif isinstance(node, (A.TupleLit, A.ArrayLit)):
            for i, e in enumerate(node.elems):
                node.elems[i] = self._maybe_share(e)
        elif isinstance(node, A.Field):
            node.obj = self._maybe_share(node.obj)
        elif isinstance(node, A.Index):
            node.callee = self._maybe_share(node.callee)
            for i, e in enumerate(node.indices):
                node.indices[i] = self._maybe_share(e)
        elif isinstance(node, A.Range):
            node.start = self._maybe_share(node.start)
            node.end = self._maybe_share(node.end)

        # Now hash & look up.
        h = structural_hash(node)
        canon = self._canon.get(h)
        if canon is None:
            self._canon[h] = node
            return node
        # Collision check: structural equality.
        if not _ast_equal(canon, node):
            raise HashConsError(
                f"[trap 20001] hash collision: two structurally distinct "
                f"AST nodes share hash {h[:12]}: "
                f"{type(canon).__name__} vs {type(node).__name__}"
            )
        # Hash + structure match — share.
        self.merged += 1
        return canon
