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


def _expr_canon(e: Optional[A.Expr]) -> str:
    """Stage 28.9 cycle 39 audit-R C38-1 fix (conf 92): canonical
    span-stripped digest of an Expr (or None). Used by `_ty_repr`
    for the Expr-typed fields of TyArray.size, TyTensor.shape /
    .device / .layout, TyTile.shape / .memspace.

    Returns the same digest the parent hasher would emit for `e`
    — recursing via `structural_hash` (which is itself span-
    independent) rather than `repr(e)` (which embeds the dataclass
    span field).

    `None` → sentinel string `"<none>"`. (Note: this is distinct
    from `_ty_repr`'s tuple sentinel `("None",)` — the two helpers
    use different None markers but neither contract crosses the
    other's usage site, so the asymmetry is benign. Cycle 40
    code-review C40-R1 conf 82 fix: docstring previously claimed
    they matched, which was inaccurate.)
    """
    if e is None:
        return "<none>"
    return structural_hash(e)


def _ty_repr(ty: Optional[A.TyNode]) -> tuple:
    """Stage 28.9 cycle 36 audit-R C36-1 fix (conf 90) + cycle 39
    audit-R C38-1 fix (conf 92): canonical span-stripped
    representation of a TyNode for hashing.

    Pre-cycle-36 `repr(node.dtype)` was used at the TileLit / Cast
    arms — but dataclass-generated repr embeds every field
    including `span`, so structurally identical types at different
    source lines produced different hashes. This violated the
    docstring contract (lines 16-19) that the hash is INTENTIONALLY
    independent of source spans.

    Cycle 36 introduced `_ty_repr` covering the TyNode subclasses,
    but used `repr(...)` on Expr-typed fields (sizes, shapes,
    device/layout/memspace) — a one-level fix that left a deeper
    span leak: a Cast to `[i32; 3]` at different source lines still
    hashed differently because the IntLit(3) inside TyArray.size
    embedded its own span via repr.

    Cycle 39 introduces `_expr_canon` which recurses via
    `structural_hash` so the Expr's span is properly stripped at
    every layer. Output is a nested tuple of (type_class_name,
    *field_values) — fully span-independent for every supported
    TyNode shape.
    """
    if ty is None:
        return ("None",)
    cls = type(ty).__name__
    if isinstance(ty, A.TyName):
        return (cls, ty.name)
    if isinstance(ty, A.TyTuple):
        return (cls, tuple(_ty_repr(e) for e in ty.elems))
    if isinstance(ty, A.TyArray):
        return (cls, _ty_repr(ty.elem), _expr_canon(ty.size))
    if isinstance(ty, A.TyRef):
        return (cls, _ty_repr(ty.inner), ty.is_mut)
    if isinstance(ty, A.TyPtr):
        return (cls, _ty_repr(ty.inner), ty.is_mut)
    if isinstance(ty, A.TyFn):
        return (cls, tuple(_ty_repr(p) for p in ty.params), _ty_repr(ty.ret))
    if isinstance(ty, A.TyTensor):
        return (cls, _ty_repr(ty.dtype),
                tuple(_expr_canon(e) for e in ty.shape),
                _expr_canon(ty.device), _expr_canon(ty.layout))
    if isinstance(ty, A.TyTile):
        return (cls, _ty_repr(ty.dtype),
                tuple(_expr_canon(e) for e in ty.shape),
                _expr_canon(ty.memspace))
    if isinstance(ty, A.TyGeneric):
        return (cls, ty.base, tuple(_ty_repr(a) for a in ty.args))
    # Unknown TyNode subclass — fail loudly per cycle 14/15 catchall
    # discipline.
    raise NotImplementedError(
        f"_ty_repr: unhandled TyNode subclass {cls}; "
        f"add an explicit arm in ast_hash.py"
    )


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
        # Stage 28.9 cycle 45 audit-T C44-1 fix (conf 92):
        # type_suffix is semantically load-bearing (`1_i32` vs `1_i64`
        # become different TIRScalar result types in lower_ast).
        # Pre-fix, the hash emitted only `node.value`, so
        # `1_i32` and `1_i64` shared a hash key and hash_cons
        # collapsed them — producing an i64 ADD with TIRScalar(i32)
        # result type. Same defect class as C34-1 (silent class-
        # only fallback): a semantic field invisible to the
        # structural-identity layer.
        _emit(h, "IntLit", node.value, node.type_suffix or "<no_suffix>")
        return
    if isinstance(node, A.FloatLit):
        # Stage 28.9 cycle 45 audit-T C44-1 fix (conf 92): same
        # rationale as IntLit — `1.0_f32` vs `1.0_f64` are
        # semantically distinct.
        _emit(h, "FloatLit", node.value, node.type_suffix or "<no_suffix>")
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
        # Stage 28.9 cycle 45 audit-T C44-1 fix (conf 92): also
        # hash `generics` (turbofish args, e.g. `foo::<i32>`). In
        # the current pipeline monomorphize_safe strips generics
        # before hash_cons so this is currently latent, but the
        # hash function should be semantics-preserving regardless
        # of caller order. Generics are TyNodes — emit via
        # span-stripped `_ty_repr`.
        idx = binders.get(node.name)
        if idx is not None:
            _emit(h, "BoundVar", idx)
        else:
            _emit(h, "FreeName", node.name)
        if node.generics:
            _emit(h, "NameGenericsCount", len(node.generics))
            for g in node.generics:
                _emit(h, "NameGeneric", _ty_repr(g))
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
            if isinstance(stmt, A.Let):
                # Stage 28.9 cycle 49 audit-R C47-A2 + C47-A3 fix:
                # emit `is_mut` (drives ALLOC_VAR + STORE_VAR vs plain
                # bind in lower_ast.py:1014-1024) and `ty` (the
                # declared type annotation, used by typecheck). Name
                # remains elided per de-Bruijn alpha-equivalence —
                # the cycle-47 C47-A1 finding noted that
                # `_stmt_equal` Let requires name match while hash
                # elides; aligning the equality side too in
                # hash_cons.py.
                # Stage 28.9 cycle 51 audit-R C49-A2 fix: handle
                # Let.value=None case (uninitialized `let x: i32;`).
                # Pre-fix that fell through to the generic
                # `_emit(h, "Stmt", type(stmt).__name__)` arm,
                # losing is_mut, ty, and the value-presence
                # distinction. Now the header is emitted always
                # and the value is folded in only when present
                # (with a sentinel emit otherwise).
                _emit(h, "Let", stmt.is_mut)
                _emit(h, "LetTy", _ty_repr(stmt.ty)
                      if stmt.ty is not None else ("None",))
                if stmt.value is not None:
                    _emit(h, "LetHasValue")
                    _hash_into(h, stmt.value, local)
                else:
                    _emit(h, "LetNoValue")
                # Bind after evaluating the rhs (not letrec).
                local[stmt.name] = depth
                depth += 1
            elif isinstance(stmt, A.ConstStmt):
                # Cycle 49 audit-R C47-A3: emit `ty` for ConstStmt
                # (similar to Let — declared type is semantic).
                _emit(h, "Const")
                _emit(h, "ConstTy", _ty_repr(stmt.ty)
                      if stmt.ty is not None else ("None",))
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
        # Stage 28.9 cycle 36 audit-R C36-1 fix (conf 90): use the
        # span-stripped `_ty_repr` canonicalizer instead of the bare
        # `repr(target_ty)`. Pre-fix, dataclass-generated repr on
        # TyNode embedded the `span` field, causing identical
        # `42 as i32` casts at different source lines to hash
        # differently — violating the docstring contract.
        _emit(h, "Ty", _ty_repr(node.target_ty))
        return
    if isinstance(node, A.Assign):
        # Stage 28.9 cycle 47 audit-T C46-1 fix (conf 90): emit
        # `op` (the assignment operator: `=`, `+=`, `-=`, `*=`,
        # `/=`, `%=`) which is semantically load-bearing — parser
        # and codegen distinguish them.
        # Also recurse into the FULL `target` Expr instead of
        # emitting only the name-as-string. Pre-fix, two assigns
        # to different fields/indices (`a.x = 1` vs `a.y = 1`,
        # `arr[0] = v` vs `arr[1] = v`) both hashed as `target="?"`.
        # Same defect class as C44-1 / C34-1.
        _emit(h, "Assign", node.op)
        _hash_into(h, node.target, binders)
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
        # Stage 28.9 cycle 49 audit-T C48-1 fix (conf 88): the
        # cycle-35 fix to ast_hash addressed Expr/Pattern subclasses
        # but the FnDecl arm (Item-level) still used the deprecated
        # `repr(p.ty)` pattern AND elided many semantic fields.
        # Multiple gaps fixed in this single pass:
        #   - Param.ty: `repr(p.ty)` → `_ty_repr(p.ty)` (span strip,
        #     same as C36-1 / C38-1 for Cast/TileLit).
        #   - return_ty: was completely absent → now emitted via
        #     _ty_repr (fn f()->i32 vs fn f()->i64 with same body
        #     would have collided pre-fix).
        #   - is_pub, is_extern, extern_abi: ABI/visibility distinctions
        #     are semantically load-bearing (lower_ast emits FFI_CALL
        #     vs internal call paths differently).
        #   - attrs: was hashed via tuple — preserved but explicitly
        #     sorted now since the parser MAY emit attrs in any order
        #     (e.g. `@pure @kernel` vs `@kernel @pure` should hash
        #     identically).
        #   - generics + where_clauses: hashed via repr-stripping for
        #     now (TODO follow-on cycle to recurse into GenericParam
        #     and WhereClause shapes; deferred because their hash
        #     surface is comparatively small).
        # Name is hashed (FnDecl.name IS semantically load-bearing —
        # it's the call-target identifier, NOT a bound variable).
        # Param names get de-Bruijn indices in the body's binder env,
        # so alpha-equivalence of body still holds (cycle-20 invariant).
        _emit(h, "FnDecl", node.name,
              tuple(sorted(node.attrs)),
              node.is_pub, node.is_extern, node.extern_abi or "")
        # Param types via span-stripped canonicalizer.
        # Stage 28.9 cycle 51 audit-R C49-A3 fix (LOW): emit
        # FnParam.is_mut alongside the type. Currently not
        # load-bearing in lower_ast but is in monomorphize for
        # the `fn f(mut x: i32)` form — symmetric with Let.is_mut
        # fix and prevents a future regression when codegen
        # starts distinguishing mut-bound params.
        local: dict[str, int] = {}
        for i, p in enumerate(node.params):
            local[p.name] = i
            _emit(h, "Param", _ty_repr(p.ty),
                  getattr(p, "is_mut", False))
        # return_ty (Optional[TyNode]) via _ty_repr (handles None).
        _emit(h, "ReturnTy", _ty_repr(node.return_ty))
        # Stage 28.9 cycle 51 audit-R C49-A1 + audit-T C49-1 fix
        # (conf 92): generics/where_clauses now use span-stripped
        # canonicalization instead of bare `repr(...)` which embedded
        # GenericParam.span and WhereClause.span (+ recursive Expr
        # spans inside constraint). Same defect class the cycle-36
        # _ty_repr fix closed for TyNode but on a deferred field.
        _emit(h, "FnGenericsCount", len(node.generics))
        for g in node.generics:
            # GenericParam has fields (name, kind); span elided.
            _emit(h, "FnGeneric", g.name, g.kind)
        _emit(h, "FnWhereClausesCount", len(node.where_clauses))
        for w in node.where_clauses:
            # WhereClause has a constraint Expr; recurse via
            # _expr_canon (the cycle-39 span-strip helper).
            _emit(h, "FnWhereClause", _expr_canon(w.constraint))
        _hash_into(h, node.body, local)
        return
    # Stage 28.9 cycle 35 audit-T C34-1 fix (conf 95): add explicit
    # arms for AST subclasses previously falling into the catch-all
    # `_emit(h, "Unknown", type(node).__name__)`. Pre-fix, that
    # fallback emitted ONLY the class name with NO recursion into
    # children, so any two instances of the same uncovered class
    # hashed identically. lower_ast.py uses structural_hash() as the
    # QUOTE handle key — `quote { return 1 }` and `quote { return 99 }`
    # therefore collapsed to the same ast_handle (verified by direct
    # repro), causing silent QUOTE-cell aliasing at runtime.
    # Same defect class as cycle 14 C14-3 / cycle 15 C15-1 silent-
    # accept fallback in match_lower walkers — applied symmetrically
    # here. The catch-all is now a loud NotImplementedError so any
    # future AST subclass forces an explicit hash-arm decision.
    if isinstance(node, A.Path):
        _emit(h, "Path", tuple(node.segments))
        return
    if isinstance(node, A.StructLit):
        _emit(h, "StructLit", node.name, len(node.fields))
        for fname, fval in node.fields:
            _emit(h, "StructField", fname)
            _hash_into(h, fval, binders)
        return
    if isinstance(node, A.Return):
        _emit(h, "Return")
        if node.value is not None:
            _hash_into(h, node.value, binders)
        else:
            _emit(h, "ReturnNoValue")
        return
    if isinstance(node, A.Break):
        _emit(h, "Break")
        if getattr(node, "value", None) is not None:
            _hash_into(h, node.value, binders)
        else:
            _emit(h, "BreakNoValue")
        return
    if isinstance(node, A.Continue):
        _emit(h, "Continue")
        return
    if isinstance(node, A.UnsafeBlock):
        _emit(h, "UnsafeBlock")
        _hash_into(h, node.body, binders)
        return
    if isinstance(node, A.TileLit):
        # Stage 28.9 cycle 36 audit-R C36-1 fix (conf 90): use
        # `_ty_repr` to span-strip the dtype. Pre-fix the bare
        # `repr(node.dtype)` embedded the dtype's TyNode span,
        # fragmenting QUOTE handles for structurally identical
        # TileLits at different source lines.
        _emit(h, "TileLit", _ty_repr(node.dtype), node.init)
        for s in node.shape:
            _hash_into(h, s, binders)
        _hash_into(h, node.memspace, binders)
        return
    # Stage 59 follow-on: ModBlock arm — needed by module_hash so it
    # can recurse into nested modules. Hashes the name + items
    # recursively (a ModBlock's identity is determined by its name
    # plus its declarative content).
    if isinstance(node, A.ModBlock):
        _emit(h, "ModBlock", node.name)
        for it in node.items:
            _hash_into(h, it, binders)
        return
    # Stage 59 follow-on: ModuleDecl arm — header-syntax `module
    # path::to::name`. Identity is the path segments only (no body).
    if isinstance(node, A.ModuleDecl):
        _emit(h, "ModuleDecl")
        for seg in node.path:
            _emit(h, "PathSeg", seg)
        return
    # Stage 59 follow-on: StructDecl arm — needed by program_hash /
    # program_signature_hash to include struct definitions in the
    # whole-program hash. A struct's identity is the name + ordered
    # fields (name + type per field). Field order matters because it
    # affects memory layout / ABI.
    if isinstance(node, A.StructDecl):
        _emit(h, "StructDecl", node.name,
              tuple(sorted(node.attrs)) if hasattr(node, "attrs") else ())
        # Generic params (if any) via the same span-strip canonicalizer
        # used by FnDecl.
        if hasattr(node, "generics"):
            _emit(h, "StructGenericsCount", len(node.generics))
            for g in node.generics:
                _emit(h, "StructGeneric", g.name, g.kind)
        # Fields in declaration order; each field is name + type.
        _emit(h, "FieldCount", len(node.fields))
        for f in node.fields:
            _emit(h, "Field", f.name, _ty_repr(f.ty))
        return
    # Loud-fail catchall — matches the cycle-14/15 NotImplementedError
    # discipline in match_lower._collect_binds and _pattern_test_expr.
    # Any future AST subclass that lands without an explicit hash arm
    # surfaces here loudly, preventing the silent-collision regression
    # that cycle-34 C34-1 caught.
    span_str = (
        f"{node.span.line}:{node.span.col}"
        if getattr(node, "span", None) is not None else "?"
    )
    raise NotImplementedError(
        f"ast_hash._hash_into at {span_str}: unhandled AST subclass "
        f"{type(node).__name__}; add an explicit arm in ast_hash.py "
        f"to declare its structural hash. (helixc internal bug — "
        f"please file an issue.)"
    )


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
    if isinstance(pat, A.PatVariant):
        out: list[str] = []
        for sub in pat.sub_patterns:
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
        # Stage 28.9 cycle 47 audit-T C46-2 fix (conf 88): emit
        # `is_mut` which distinguishes `Some(x) =>` from
        # `Some(mut x) =>` for mutability-checking. The `name`
        # field is intentionally elided (de-Bruijn alpha-
        # equivalence), but `is_mut` is semantically load-bearing.
        _emit(h, "PatBind", pat.is_mut)
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
    if isinstance(pat, A.PatVariant):
        # Hash by path segments + recursive sub-pattern hashes. Crucial
        # for AD memoization correctness: two `match m { Some(x) => x }`
        # vs `match m { None => 0 }` must hash differently.
        segs = tuple(pat.path.segments)
        _emit(h, "PatVariant", segs, len(pat.sub_patterns))
        for sub in pat.sub_patterns:
            _hash_pattern(h, sub, binders)
        return
    if isinstance(pat, A.PatStruct):
        # Stage 59 / Tier 4 #15: struct destructuring pattern.
        # Hash by struct name + ordered (field_name, sub_hash) pairs +
        # ignore_rest flag. Field order matters (the user wrote them
        # in that order); to canonicalize would require sorting, but
        # that's a design choice for a future polish increment.
        _emit(h, "PatStruct", pat.name, len(pat.fields),
              pat.ignore_rest)
        for (fname, sub) in pat.fields:
            _emit(h, "PatStructField", fname)
            _hash_pattern(h, sub, binders)
        return
    _emit(h, "PatUnknown", type(pat).__name__)


def short_hash(h: str) -> str:
    """Return a 12-hex-char abbreviation suitable for human inspection."""
    return h[:12]


# ============================================================================
# Stage 58 / Tier 4 #13 — content-addressed modules
# ============================================================================
def fn_signature_hash(fn: "A.FnDecl") -> str:
    """Stage 59 follow-on / Tier 4 #13 polish — hash a function's
    SIGNATURE only (name + param types + return type), excluding the
    body. Two fns with the same signature but different bodies hash
    identically here (whereas `structural_hash(fn)` would differ).

    Use cases:
    - ABI compat check: did the public signature change between two
      versions of a fn? If signature_hash differs, callers need
      recompilation; if only body changed (full hash differs but
      signature_hash matches), it's an internal refactor.
    - Overload-set deduplication: two fns with the same signature
      cannot coexist (Helix has no overloading).
    - Trait-impl matching: same trait method requires same signature.

    Decisions baked in:
    - Param names are alpha-equivalent (de Bruijn-style), so renaming
      `fn f(x: i32)` to `fn f(y: i32)` doesn't change signature_hash
    - Param mutability IS included (mut differs from immutable)
    - Effect attributes (@pure, @effect(io)) ARE included since they
      are observable to callers
    """
    h = hashlib.sha256()
    _emit(h, "FnSig", fn.name)
    # Param count + per-param type hash.
    _emit(h, "Params", len(fn.params))
    for p in fn.params:
        _emit(h, "ParamMut", p.is_mut)
        _emit(h, "ParamTy", _ty_repr(p.ty))
    _emit(h, "RetTy", _ty_repr(fn.return_ty))
    # Effect attrs that affect callers' purity contracts.
    relevant_attrs = sorted(
        a for a in fn.attrs
        if a.startswith("effect:") or a in ("pure", "is_pure")
    )
    _emit(h, "Effects", tuple(relevant_attrs))
    return h.hexdigest()


def module_hash(decl) -> str:
    """Stage 58 / Tier 4 #13 — structural hash of a `mod X { ... }`
    block (`A.ModBlock` in the AST). The hash covers:
      - Module name (so two modules with same items hash differently)
      - All items in declaration order (FnDecl/StructDecl/UseDecl/
        ConstDecl/ImplDecl/TraitDecl/nested ModBlock)
    Span-independent + insensitive to internal naming via the existing
    `structural_hash` machinery.

    Accepts either A.ModBlock (block-syntax module with items) or
    A.ModuleDecl (header-syntax `module path::to::name`); the latter
    hashes its path segments only (no items to traverse).

    Use case: content-addressed package references — a `use X::Y` can
    cite the imported module by hash, allowing the compiler to detect
    drift between import declaration and resolved module body.
    """
    h = hashlib.sha256()
    if isinstance(decl, A.ModBlock):
        _emit(h, "ModBlock", decl.name)
        for it in decl.items:
            item_h = (structural_hash(it)
                       if _is_hashable_item(it) else "<opaque>")
            _emit(h, "Item", type(it).__name__, item_h)
    elif isinstance(decl, A.ModuleDecl):
        _emit(h, "ModuleDecl")
        for seg in decl.path:
            _emit(h, "PathSeg", seg)
    else:
        raise TypeError(
            f"module_hash: expected ModBlock or ModuleDecl, got "
            f"{type(decl).__name__}"
        )
    return h.hexdigest()


def program_hash(prog: "A.Program") -> str:
    """Stage 58 / Tier 4 #13 — structural hash of a full Program.
    Covers all top-level items in declaration order. Stable across
    span / formatting / bound-name changes.

    Use case: caching the entire compilation result by source-program
    content hash; detecting that a re-parsed file is semantically
    identical to a cached version.
    """
    h = hashlib.sha256()
    _emit(h, "Program")
    for it in prog.items:
        item_h = structural_hash(it) if _is_hashable_item(it) else "<opaque>"
        _emit(h, "Item", type(it).__name__, item_h)
    return h.hexdigest()


def program_signature_hash(prog: "A.Program") -> str:
    """Stage 59 follow-on / Tier 4 #13 polish — ABI-level hash of a
    Program: covers fn SIGNATURES + struct DEFINITIONS but NOT fn
    BODIES.

    Two programs with the same public surface (callers of either
    program would behave identically AS LONG AS the body changes are
    semantically equivalent) hash identically here.

    Distinguishing from program_hash:
    - program_hash: every byte of structural content matters
    - program_signature_hash: only the public/exported surface

    Specifically:
    - FnDecl: fn_signature_hash (name + param types + return type
      + effect attrs); body ignored
    - StructDecl: full structural_hash (struct definitions ARE part
      of the ABI — adding/removing/reordering fields breaks callers)
    - Other items: structural_hash (conservatively included)

    Use cases:
    - ABI compat check between two versions of a library: same
      program_signature_hash ⇒ caller-observable surface unchanged
    - Public-surface stability gate: assert program_signature_hash
      hasn't drifted across releases
    - Allow internal refactors to land without bumping a 'public
      contract' hash version
    """
    h = hashlib.sha256()
    _emit(h, "ProgramSig")
    for it in prog.items:
        if isinstance(it, A.FnDecl):
            _emit(h, "FnSig", fn_signature_hash(it))
        elif _is_hashable_item(it):
            _emit(h, "Item", type(it).__name__, structural_hash(it))
        else:
            _emit(h, "Item", type(it).__name__, "<opaque>")
    return h.hexdigest()


def _is_hashable_item(item: Any) -> bool:
    """An item is hashable here if `_hash_into` has an explicit arm
    for it. Today: FnDecl + StructDecl + ConstDecl + ModuleDecl (via
    recursive call) + most other top-level items handled by the
    `_hash_into` catchall. We let `structural_hash` raise for genuinely
    unhandled subclasses (per the cycle-14/15 loud-fail discipline);
    this helper exists as a forward-compat hook in case some items
    need an opaque sentinel rather than a hash."""
    return True
