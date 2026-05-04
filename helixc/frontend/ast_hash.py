"""
helixc/frontend/ast_hash.py — structural content-addressed hashing of AST nodes.

Every AST subtree gets a stable SHA-256 hash derived from its constructor +
child hashes + literal/attr fields, with bound names canonicalized to
de-Bruijn indices. Two AST subtrees that are alpha-equivalent and
structurally identical hash to the same value.

This is the foundation for:
- content-addressed quote() handles (replace the current Python-hash mod 64)
- cross-call caching of differentiate() results (memoize by body hash)
- e-graph rewriting (each e-class is a hash bucket)
- detecting duplicate gradient subexpressions before lowering
- mechanically-checkable program-edit provenance for AGI self-modification

The hash is INTENTIONALLY independent of:
- source spans (line/col)
- formatting / whitespace
- bound-variable names (alpha-equivalence)

It IS dependent on:
- structural shape (a + b ≠ b + a unless commutativity proven separately)
- literal values (constants matter)
- attribute keys + values that change semantics

License: Apache 2.0
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from . import ast_nodes as A


def structural_hash(node: A.Expr | A.Block | A.Stmt | A.FnDecl | None,
                    binders: Optional[dict[str, int]] = None) -> str:
    """Return a hex-encoded SHA-256 of the structural shape of `node`.

    `binders` maps a name -> de-Bruijn-style index (counted from the
    innermost binder). Bound-variable references are hashed by their
    index, not name, so alpha-equivalent terms hash equally.
    """
    h = hashlib.sha256()
    _hash_into(h, node, dict(binders or {}))
    return h.hexdigest()


def _emit(h: "hashlib._Hash", tag: str, *parts: object) -> None:
    """Write a labelled, length-prefixed record into the hash. Each part
    is converted to bytes via repr for primitives, or a recursive call
    for nested structures."""
    h.update(tag.encode("utf-8"))
    h.update(b":")
    for p in parts:
        b = repr(p).encode("utf-8") if not isinstance(p, bytes) else p
        h.update(len(b).to_bytes(4, "big"))
        h.update(b)
    h.update(b";")


def _hash_into(h: "hashlib._Hash", node: Any,
               binders: dict[str, int]) -> None:
    if node is None:
        _emit(h, "None")
        return
    if isinstance(node, A.IntLit):
        _emit(h, "IntLit", node.value)
        return
    if isinstance(node, A.FloatLit):
        _emit(h, "FloatLit", node.value)
        return
    if isinstance(node, A.BoolLit):
        _emit(h, "BoolLit", node.value)
        return
    if isinstance(node, A.StrLit):
        _emit(h, "StrLit", node.value)
        return
    if isinstance(node, A.CharLit):
        _emit(h, "CharLit", node.value)
        return
    if isinstance(node, A.Name):
        # Bound name → use the de-Bruijn-style depth index.
        # Free name → use the literal name.
        idx = binders.get(node.name)
        if idx is not None:
            _emit(h, "BoundVar", idx)
        else:
            _emit(h, "FreeName", node.name)
        return
    if isinstance(node, A.Unary):
        _emit(h, "Unary", node.op)
        _hash_into(h, node.operand, binders)
        return
    if isinstance(node, A.Binary):
        _emit(h, "Binary", node.op)
        _hash_into(h, node.left, binders)
        _hash_into(h, node.right, binders)
        return
    if isinstance(node, A.Call):
        _emit(h, "Call", len(node.args))
        _hash_into(h, node.callee, binders)
        for a in node.args:
            _hash_into(h, a, binders)
        return
    if isinstance(node, A.If):
        _emit(h, "If")
        _hash_into(h, node.cond, binders)
        _hash_into(h, node.then, binders)
        _hash_into(h, node.else_, binders)
        return
    if isinstance(node, A.Block):
        _emit(h, "Block", len(node.stmts))
        # Each let-binding extends the binder set; we use de-Bruijn-style
        # indexing by counting depth from the innermost binder out.
        local = dict(binders)
        depth = max(local.values(), default=-1) + 1
        for stmt in node.stmts:
            if isinstance(stmt, A.Let) and stmt.value is not None:
                _emit(h, "Let")
                _hash_into(h, stmt.value, local)
                # Bind after evaluating the rhs (not letrec).
                local[stmt.name] = depth
                depth += 1
            elif isinstance(stmt, A.ConstStmt):
                _emit(h, "Const")
                _hash_into(h, stmt.value, local)
                local[stmt.name] = depth
                depth += 1
            elif isinstance(stmt, A.ExprStmt):
                _emit(h, "ExprStmt")
                _hash_into(h, stmt.expr, local)
            else:
                _emit(h, "Stmt", type(stmt).__name__)
        if node.final_expr is not None:
            _hash_into(h, node.final_expr, local)
        else:
            _emit(h, "NoFinal")
        return
    if isinstance(node, A.For):
        # NB: don't include var_name in the tag emission — the de-Bruijn
        # binder map below already encodes the binding, so two For-loops
        # that differ only in loop variable name should hash equally.
        _emit(h, "For")
        _hash_into(h, node.iter_expr, binders)
        local = dict(binders)
        depth = max(local.values(), default=-1) + 1
        local[node.var_name] = depth
        _hash_into(h, node.body, local)
        return
    if isinstance(node, A.While):
        _emit(h, "While")
        _hash_into(h, node.cond, binders)
        _hash_into(h, node.body, binders)
        return
    if isinstance(node, A.Loop):
        _emit(h, "Loop")
        _hash_into(h, node.body, binders)
        return
    if isinstance(node, A.Cast):
        _emit(h, "Cast")
        _hash_into(h, node.value, binders)
        # Type name — the parser constructs TyName; we hash by string repr.
        _emit(h, "Ty", repr(node.target_ty))
        return
    if isinstance(node, A.Assign):
        _emit(h, "Assign", node.target.name if isinstance(node.target, A.Name) else "?")
        _hash_into(h, node.value, binders)
        return
    if isinstance(node, A.Index):
        _emit(h, "Index", len(node.indices))
        _hash_into(h, node.callee, binders)
        for i in node.indices:
            _hash_into(h, i, binders)
        return
    if isinstance(node, A.Quote):
        _emit(h, "Quote")
        _hash_into(h, node.inner, binders)
        return
    if isinstance(node, A.Splice):
        _emit(h, "Splice")
        _hash_into(h, node.inner, binders)
        return
    if isinstance(node, A.Modify):
        _emit(h, "Modify")
        _hash_into(h, node.target, binders)
        _hash_into(h, node.transformation, binders)
        _hash_into(h, node.verifier, binders)
        return
    if isinstance(node, A.Range):
        _emit(h, "Range")
        _hash_into(h, node.start, binders)
        _hash_into(h, node.end, binders)
        return
    if isinstance(node, A.TupleLit):
        _emit(h, "TupleLit", len(node.elems))
        for e in node.elems:
            _hash_into(h, e, binders)
        return
    if isinstance(node, A.ArrayLit):
        _emit(h, "ArrayLit", len(node.elems))
        for e in node.elems:
            _hash_into(h, e, binders)
        return
    if isinstance(node, A.Field):
        _emit(h, "Field", node.name)
        _hash_into(h, node.obj, binders)
        return
    if isinstance(node, A.Match):
        _emit(h, "Match", len(node.arms))
        _hash_into(h, node.scrutinee, binders)
        for arm in node.arms:
            _emit(h, "Arm")
            _hash_pattern(h, arm.pattern, binders)
            # Extend binder map with any names introduced by the pattern
            # so that references in the guard / body resolve via index,
            # making `y => y+1` and `z => z+1` hash equally.
            local = dict(binders)
            depth = max(local.values(), default=-1) + 1
            for name in _pattern_binders(arm.pattern):
                local[name] = depth
                depth += 1
            _hash_into(h, arm.guard, local)
            _hash_into(h, arm.body, local)
        return
    if isinstance(node, A.FnDecl):
        # Hash a top-level fn by: name, attrs, param COUNT + TYPES (not
        # names — alpha-equivalence), body. Param names get de-Bruijn
        # indices in the body's binder env.
        _emit(h, "FnDecl", node.name, tuple(node.attrs))
        local: dict[str, int] = {}
        for i, p in enumerate(node.params):
            local[p.name] = i
            _emit(h, "Param", repr(p.ty))   # type only, not name
        _hash_into(h, node.body, local)
        return
    # Fallback for anything we haven't enumerated: hash the type name.
    # This is conservative: it prevents collisions but two different
    # instances of the same class share a hash.
    _emit(h, "Unknown", type(node).__name__)


def _pattern_binders(pat: A.Pattern) -> list[str]:
    """Names introduced by the pattern, in left-to-right order. Used to
    extend the de-Bruijn binder map for arm-body hashing."""
    if isinstance(pat, A.PatBind):
        return [pat.name]
    if isinstance(pat, A.PatTuple):
        out: list[str] = []
        for sub in pat.elems:
            out.extend(_pattern_binders(sub))
        return out
    if isinstance(pat, A.PatOr):
        # All alternatives must bind the same names — take from first.
        return _pattern_binders(pat.alts[0]) if pat.alts else []
    return []


def _hash_pattern(h: "hashlib._Hash", pat: A.Pattern,
                  binders: dict[str, int]) -> None:
    """Hash a match pattern. Binder names use de-Bruijn-style depth (so
    `x => x+1` and `y => y+1` hash equally as patterns); literal/range
    endpoints are folded in by their normal expression hash."""
    if isinstance(pat, A.PatWildcard):
        _emit(h, "PatWildcard")
        return
    if isinstance(pat, A.PatBind):
        _emit(h, "PatBind")
        return
    if isinstance(pat, A.PatLit):
        _emit(h, "PatLit")
        _hash_into(h, pat.value, binders)
        return
    if isinstance(pat, A.PatTuple):
        _emit(h, "PatTuple", len(pat.elems))
        for sub in pat.elems:
            _hash_pattern(h, sub, binders)
        return
    if isinstance(pat, A.PatOr):
        _emit(h, "PatOr", len(pat.alts))
        for alt in pat.alts:
            _hash_pattern(h, alt, binders)
        return
    if isinstance(pat, A.PatRange):
        _emit(h, "PatRange", pat.inclusive)
        _hash_into(h, pat.lo, binders)
        _hash_into(h, pat.hi, binders)
        return
    _emit(h, "PatUnknown", type(pat).__name__)


def short_hash(h: str) -> str:
    """Return a 12-hex-char abbreviation suitable for human inspection."""
    return h[:12]
