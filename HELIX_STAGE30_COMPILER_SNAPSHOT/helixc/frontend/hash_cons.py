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


def _ty_equal(a: Any, b: Any) -> bool:
    """Stage 28.9 cycle 41 audit-T C39-A1 fix (conf 55): true
    structural equality on TyNode subclasses without folding
    Expr-typed sub-fields through SHA-256.

    Mirrors `ast_hash._ty_repr` shape (1-1 mapping of subclasses to
    field checks) but recurses Expr-typed fields via `_ast_equal`
    instead of digest comparison. This preserves the cycle-37
    invariant that `_ast_equal` is exact structural equality used
    to disambiguate SHA-256 hash buckets — the disambiguator must
    NOT itself rely on the same hash.

    Span fields are intentionally NOT compared (per the docstring
    contract of `ast_hash`: hashing and structural identity are
    span-independent).
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if type(a) is not type(b):
        return False
    if isinstance(a, A.TyName):
        return a.name == b.name
    if isinstance(a, A.TyTuple):
        if len(a.elems) != len(b.elems):
            return False
        return all(_ty_equal(x, y) for x, y in zip(a.elems, b.elems))
    if isinstance(a, A.TyArray):
        return _ty_equal(a.elem, b.elem) and _ast_equal(a.size, b.size)
    if isinstance(a, A.TyRef):
        return _ty_equal(a.inner, b.inner) and a.is_mut == b.is_mut
    if isinstance(a, A.TyPtr):
        return _ty_equal(a.inner, b.inner) and a.is_mut == b.is_mut
    if isinstance(a, A.TyFn):
        if len(a.params) != len(b.params):
            return False
        if not all(_ty_equal(p1, p2) for p1, p2 in zip(a.params, b.params)):
            return False
        return _ty_equal(a.ret, b.ret)
    if isinstance(a, A.TyTensor):
        if not _ty_equal(a.dtype, b.dtype):
            return False
        if len(a.shape) != len(b.shape):
            return False
        if not all(_ast_equal(s1, s2) for s1, s2 in zip(a.shape, b.shape)):
            return False
        return (_ast_equal(a.device, b.device)
                and _ast_equal(a.layout, b.layout))
    if isinstance(a, A.TyTile):
        if not _ty_equal(a.dtype, b.dtype):
            return False
        if len(a.shape) != len(b.shape):
            return False
        if not all(_ast_equal(s1, s2) for s1, s2 in zip(a.shape, b.shape)):
            return False
        return _ast_equal(a.memspace, b.memspace)
    if isinstance(a, A.TyGeneric):
        if a.base != b.base:
            return False
        if len(a.args) != len(b.args):
            return False
        return all(_ty_equal(x, y) for x, y in zip(a.args, b.args))
    # Unknown TyNode subclass — fail loudly (matches the cycle 14/15
    # walker-fallback discipline).
    raise NotImplementedError(
        f"_ty_equal: unhandled TyNode subclass {type(a).__name__}; "
        f"add an explicit arm in hash_cons.py"
    )


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
        # Stage 28.9 cycle 45 audit-T C44-1 fix (conf 92): mirror
        # the structural_hash arm — `1_i32` and `1_i64` are
        # semantically distinct.
        return a.value == b.value and a.type_suffix == b.type_suffix
    if isinstance(a, A.FloatLit):
        # Same rationale as IntLit.
        return a.value == b.value and a.type_suffix == b.type_suffix
    if isinstance(a, A.BoolLit):
        return a.value == b.value
    if isinstance(a, A.StrLit):
        return a.value == b.value
    if isinstance(a, A.CharLit):
        return a.value == b.value
    if isinstance(a, A.Name):
        # Stage 28.9 cycle 45 audit-T C44-1 fix (conf 92): also
        # compare `generics`. In the current pipeline
        # monomorphize_safe strips generics before hash_cons so
        # this is latent, but the equality function must be
        # semantics-preserving regardless of caller order.
        # Generics are TyNodes — use `_ty_equal`.
        if a.name != b.name:
            return False
        if len(a.generics) != len(b.generics):
            return False
        return all(_ty_equal(g1, g2)
                   for g1, g2 in zip(a.generics, b.generics))
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
        # Stage 28.9 cycle 37 audit-R C36-1 fix follow-on: use the
        # span-stripped TyNode comparison so structural_hash and
        # _ast_equal stay in lockstep.
        # Stage 28.9 cycle 41 audit-T C39-A1 fix (conf 55): the
        # cycle-37/39 implementation compared `_ty_repr(a.target_ty)
        # == _ty_repr(b.target_ty)`, which folded Expr-typed TyNode
        # fields (TyArray.size, TyTensor.shape, etc.) through
        # SHA-256 digests via _expr_canon. That weakened the
        # collision-resistance guarantee `_ast_equal` exists to
        # provide post-hash-bucket. Now use `_ty_equal` which
        # recurses Expr-typed fields via `_ast_equal` itself — true
        # exact structural equality, no SHA-256 dependency at the
        # collision-disambiguation layer.
        return (_ast_equal(a.value, b.value)
                and _ty_equal(a.target_ty, b.target_ty))
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
    # Block — used as the body of If/For/While/Loop. Compared structurally
    # by statements + final_expr. (Blocks aren't themselves shared, but
    # they appear as children of shared If nodes, so the comparison must
    # descend through them to confirm the parent's structural identity.)
    if isinstance(a, A.Block):
        if len(a.stmts) != len(b.stmts):
            return False
        for sa, sb in zip(a.stmts, b.stmts):
            if not _stmt_equal(sa, sb):
                return False
        return _ast_equal(a.final_expr, b.final_expr)
    # Stage 28.9 cycle 71 type-design CN-3 deferred (sub-75 conf 70):
    # the audit flagged this fallback as architecturally compromising
    # the trap-20001 collision-defense contract (uses SHA-256 to
    # defend against SHA-256 collisions). A loud-fail conversion was
    # attempted but exposed multiple missing explicit arms (Assign,
    # Return, Break, Continue, Quote, Splice, Modify, UnsafeBlock,
    # TileLit, StructLit, For, While, Loop, Match, Path) that
    # legitimate compile paths reach. Deferred to a dedicated cycle
    # that adds all the missing arms incrementally with tests for
    # each. For now the conservative-fallback comment is preserved
    # (cycle-70 CN-3 at conf 70 was below the 75% gate).
    # Conservative default: fall back to hash equality. Reaching this
    # branch means _ast_equal hit a node type the explicit enumeration
    # doesn't cover; SHA-256 is collision-resistant so trusting the
    # hash is a safe (and very rare) fallback.
    return structural_hash(a) == structural_hash(b)


def _stmt_equal(a: Any, b: Any) -> bool:
    """Structural equality for statements (used inside Block comparison)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, A.Let):
        # Stage 28.9 cycle 49 audit-R C47-A1 fix (HIGH): drop the
        # `a.name == b.name` check — _hash_into elides Let.name per
        # de-Bruijn alpha-equivalence, so requiring name match here
        # falsely tripped trap 20001 ("structurally distinct AST
        # nodes share hash") for alpha-equivalent let-blocks.
        # Cycle 49 audit-R C47-A2 fix (HIGH): include is_mut in
        # comparison — `let mut x = e` and `let x = e` are
        # semantically distinct (drives ALLOC_VAR vs plain bind).
        # Cycle 49 audit-R C47-A3 fix (MED): include declared ty.
        if a.is_mut != b.is_mut:
            return False
        if not _ty_equal(a.ty, b.ty):
            return False
        return _ast_equal(a.value, b.value)
    if isinstance(a, A.ConstStmt):
        # Stage 28.9 cycle 49: align with Let — name elided per
        # de-Bruijn, ty must match.
        if not _ty_equal(a.ty, b.ty):
            return False
        return _ast_equal(a.value, b.value)
    if isinstance(a, A.ExprStmt):
        return _ast_equal(a.expr, b.expr)
    return structural_hash(a) == structural_hash(b)


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

        # Everything else: shareable expressions go through _maybe_share;
        # truly-unknown node types (Quote / Splice / Modify / patterns /
        # statements that hit this path) just walk their primary child
        # attributes and return as-is. This is what closes the
        # share_in <-> _maybe_share loop for nodes that are neither in
        # _SHAREABLE nor in the special block/match/for/while/loop set.
        if isinstance(node, _SHAREABLE):
            return self._maybe_share(node)

        # Conservative walker for anything else (Quote / Splice /
        # Modify / Assign / ...). Walk known child attributes; leave the
        # node itself untouched.
        for attr in ("inner", "target", "transformation", "verifier",
                     "value", "operand", "expr"):
            v = getattr(node, attr, None)
            if v is None or isinstance(v, (str, int, float, bool)):
                continue
            # Avoid AST type nodes; only recurse into AST expr / block
            # subtrees, which are dataclass instances.
            if hasattr(v, "span"):
                setattr(node, attr, self.share_in(v))
        return node

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
