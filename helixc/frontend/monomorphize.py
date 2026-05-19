"""
helixc/frontend/monomorphize.py

Monomorphization pass: instantiates generic functions with concrete type args.

Input: A.Program with FnDecls that have generic params (e.g., fn id[T](x: T) -> T)
       and call sites with turbofish (e.g., id::<i32>(x))
Output: A.Program with concrete instantiations replacing generic calls.

Algorithm (cycle 5 C4-4 / HIGH revision):
  1. Collect all FnDecls with non-empty generics into a generics-table.
  2. Walk ONLY the non-generic items (do not mutate generic-fn bodies
     in place). Find Calls whose callee is Name with non-empty generics.
  3. For each unique (fn_name, type_args) pair:
     - Generate mangled name (id__i32, pair__i32_f64).
     - Clone the FnDecl, substituting type variables in params/return_ty/body.
       The clone's body has turbofish args fully substituted via
       `_walk_subst_expr`'s Call arm (cycle 3 D9: substitutes
       `callee.generics`).
     - Add the clone to a promoted set so it gets walked in the next
       iteration (NOT promoted directly to prog.items mid-iteration).
     - Replace the callee in the OUTER (non-generic) caller with
       Name(mangled_name, generics=[]).
  4. Repeat the iteration over (non-generic items + promoted clones)
     until no more changes. Each clone may call other generics; those
     get walked next iteration.
  5. Append all clones to prog.items at the end. Original generic
     FnDecls remain in prog.items so legacy un-turbofished call sites
     still resolve (the cycle-3 backward-compat behavior).

Cycle 5 C4-4 / HIGH note: cycle-3 D9 (`_walk_subst_expr` Call arm)
WAS necessary — turbofish substitution inside cloned bodies. But
the iteration order was wrong: in the original cycle-3 algorithm,
generic-fn bodies were rewritten in-place during iteration 1
(replacing `id::<U>(v)` with `id__U(v)` mangled-no-turbofish),
BEFORE iteration 2 cloned `caller__i32` with subst `{U: i32}` —
the clone's `_walk_subst_expr` then had nothing to substitute.
Cycle 5 fixes this by not walking generic-fn bodies at top level;
clones inherit the original generic body (turbofish intact) and
get walked via `_walk_subst_expr` per clone.

License: Apache 2.0
"""

from __future__ import annotations
from copy import deepcopy
from typing import Optional

from . import ast_nodes as A
from .ast_hash import structural_hash


def mangle(name: str, ty_args: list[A.TyNode]) -> str:
    """Generate a unique mangled name for a (fn_name, type_args) pair.

    Conventions:
      id::<i32>           -> id__i32
      pair::<i32, f64>    -> pair__i32_f64
      box::<(i32, i64)>   -> box__tup_i32_i64
    """
    parts = [_mangle_ty(t) for t in ty_args]
    return f"{name}__" + "_".join(parts)


def _mangle_shape_expr(e) -> str:
    """Stage 28.9 cycle 51 audit-T C49-3 fix (conf 85) +
    cycle 53 audit-T C52-MS1/C52-MS2/C52-CR1/C52-MS3 fix
    (HIGH conf 90): a span-free string digest of a shape-typed Expr
    (TyArray.size, TyTensor.shape element / device / layout,
    TyTile.shape element / memspace).

    Strategy change in cycle 53: instead of enumerating every Expr
    subclass — which created the same drift hazard that produced
    cycle-34/36/38/44 (semantic field invisible to identity layer)
    — delegate to `structural_hash` (the canonical span-stripped
    AST canonicalizer from ast_hash.py). This:

      1. Closes C52-MS1 (no Call arm → `cuda(0)` vs `cuda(1)`
         collapsed). structural_hash handles ALL Expr subclasses.
      2. Closes C52-MS2 (silent catchall `x_<class>` collapsed
         distinct exprs). structural_hash raises NotImplementedError
         on unknown subclasses (cycle-35 discipline).
      3. Closes C52-CR1 (theoretical `_` separator non-injectivity
         in the prior enum-form). SHA-256 hex digests are
         unambiguous.
      4. Closes C52-MS3 drift with struct_mono._shape_key — though
         struct_mono's own canonicalizer remains distinct, neither
         can silently drift further than structural_hash itself.

    The output is `"h<12-hex-prefix>"` — short enough for mangled
    fn-name embedding, with SHA-256 collision resistance making the
    truncation safe in practice.
    """
    if e is None:
        return "none"
    # Restart 61 B2: remove dead `try/except (...): raise` block. The
    # previous handler caught (TypeError, AttributeError,
    # NotImplementedError) only to immediately re-raise without
    # decoration — a no-op. The handler claimed it added "mangle-site
    # as breadcrumb" but never modified the exception. Let the
    # structural_hash exceptions propagate directly; the call site is
    # already visible in the traceback. Eliminates dead code that
    # implies safety it does not provide.
    h = structural_hash(e)
    return "h" + h[:12]


def _mangle_ty(t: A.TyNode) -> str:
    if isinstance(t, A.TyName):
        return t.name
    if isinstance(t, A.TyTuple):
        return "tup_" + "_".join(_mangle_ty(e) for e in t.elems)
    if isinstance(t, A.TyArray):
        # Stage 28.9 cycle 51 audit-T C49-3 fix (conf 85): include
        # the size in the mangled name. Pre-fix, two generic fn
        # instantiations at different array sizes (`f::<[i32; 3]>`
        # and `f::<[i32; 5]>`) produced identical mangled names
        # `arr_i32`, so the second call site silently used the
        # first instantiation's body — wrong array length at codegen.
        # Mirrors struct_mono._ty_key C2-5 fix.
        return ("arr_" + _mangle_ty(t.elem)
                + "__" + _mangle_shape_expr(t.size))
    if isinstance(t, A.TyRef):
        return ("refmut_" if t.is_mut else "ref_") + _mangle_ty(t.inner)
    if isinstance(t, A.TyPtr):
        # Cycle 51 C49-3 follow-on: TyPtr was missing from the
        # original switch; the catch-all "X" was returned instead.
        return ("ptrmut_" if t.is_mut else "ptr_") + _mangle_ty(t.inner)
    if isinstance(t, A.TyFn):
        return "fn_" + "_".join(_mangle_ty(p) for p in t.params) + "_to_" + _mangle_ty(t.ret)
    if isinstance(t, A.TyTensor):
        # Cycle 51 C49-3: include shape + device + layout per
        # struct_mono._ty_key. Pre-fix, two TyTensor at different
        # shapes collapsed to `tensor_<dtype>`.
        shape_part = "_".join(_mangle_shape_expr(s) for s in t.shape)
        return ("tensor_" + _mangle_ty(t.dtype)
                + "__shape_" + shape_part
                + "__dev_" + _mangle_shape_expr(t.device)
                + "__layout_" + _mangle_shape_expr(t.layout))
    if isinstance(t, A.TyTile):
        # Cycle 51 C49-3: include shape + memspace.
        shape_part = "_".join(_mangle_shape_expr(s) for s in t.shape)
        return ("tile_" + _mangle_ty(t.dtype)
                + "__shape_" + shape_part
                + "__mem_" + _mangle_shape_expr(t.memspace))
    if isinstance(t, A.TyGeneric):
        return t.base + "_" + "_".join(_mangle_ty(a) for a in t.args)
    # Stage 28.9 cycle 71 type-design CN-2 fix (HIGH conf 78):
    # pre-fix `return "X"` silently collapsed any future unrecognized
    # TyNode subclass to the same mangled key. Sister modules already
    # use the loud-fail discipline (struct_mono._ty_key raises
    # TypeError; hash_cons._ty_equal raises NotImplementedError;
    # flatten_impls/_modules._rewrite_expr raise NotImplementedError).
    # Layer-0/1 expansion (refinement types, confidence types, tiered
    # memory, etc.) will add new TyNode subclasses — silent collapse
    # would cause silent codegen mis-link. Promote to loud-fail so
    # future additions force explicit dispatch here.
    raise NotImplementedError(
        f"_mangle_ty: unhandled TyNode subclass {type(t).__name__} "
        f"at {getattr(t, 'span', '?')!r}. Add an explicit arm "
        f"to monomorphize._mangle_ty (and sibling switches in "
        f"struct_mono._ty_key, hash_cons._ty_equal, ast_hash._ty_repr "
        f"for cross-cutting consistency)."
    )


# Audit 28.8 cycle 4 C4-1: trap-id constant promoted from a literal
# embedded in the diagnostic message to a module-level identifier so
# the registry in docs/lang/trap-ids.md cross-references a real symbol.
TRAP_SHAPE_FOLD_ZERO_DIV = 28801


class ShapeFoldError(ValueError):
    """Audit 28.8 cycle 3 C3-6: raised when shape-time constant fold
    hits a hard error (currently division/modulo by zero). Trap 28801
    (= TRAP_SHAPE_FOLD_ZERO_DIV).
    Caught by the typechecker / driver and surfaced as a user-facing
    diagnostic — silent-fallthrough → length 0 is no longer allowed."""

    trap_id: int = TRAP_SHAPE_FOLD_ZERO_DIV

    def __init__(self, msg: str, span: A.Span):
        super().__init__(msg)
        self.span = span
        self.msg = msg


def _fold_intlit_arith(expr: A.Expr) -> A.Expr:
    """Audit 28.8 cycle 2 B:C11: one-level constant-fold for shape
    expressions of the form Binary(IntLit, op, IntLit). Pre-fix,
    `_subst_shape_expr` substituted `Name("N")` → `IntLit(64)` but
    left the surrounding Binary unfolded, so `[N*2; 16]` mono'd
    against N=64 stayed as `Binary(*, IntLit(64), IntLit(2))` instead
    of `IntLit(128)`. Downstream lower-ast.py defaulted non-IntLit
    shapes to 0 — silent miscount.

    Applies only to commutative arithmetic with two IntLit operands.
    Non-foldable shapes (e.g. one side still has a Name leaf) flow
    through unchanged.

    Audit 28.8 cycle 3 C3-6: `/ 0` and `% 0` now raise ShapeFoldError
    (trap 28801) instead of silently leaving the Binary unfolded and
    letting lower_ast default the length to 0."""
    if not isinstance(expr, A.Binary):
        return expr
    if not (isinstance(expr.left, A.IntLit) and isinstance(expr.right, A.IntLit)):
        return expr
    lv = expr.left.value
    rv = expr.right.value
    op = expr.op
    if op == "+":
        return A.IntLit(span=expr.span, value=lv + rv, type_suffix=None)
    if op == "-":
        return A.IntLit(span=expr.span, value=lv - rv, type_suffix=None)
    if op == "*":
        return A.IntLit(span=expr.span, value=lv * rv, type_suffix=None)
    if op == "/":
        if rv == 0:
            raise ShapeFoldError(
                f"{expr.span.line}:{expr.span.col}: division by zero "
                f"in shape expression (trap {TRAP_SHAPE_FOLD_ZERO_DIV})",
                expr.span,
            )
        return A.IntLit(span=expr.span, value=lv // rv, type_suffix=None)
    if op == "%":
        if rv == 0:
            raise ShapeFoldError(
                f"{expr.span.line}:{expr.span.col}: modulo by zero "
                f"in shape expression (trap {TRAP_SHAPE_FOLD_ZERO_DIV})",
                expr.span,
            )
        return A.IntLit(span=expr.span, value=lv % rv, type_suffix=None)
    return expr


def _fold_intlit_unary(expr: A.Expr) -> A.Expr:
    """Audit 28.8 cycle 3 D5: symmetric one-level fold for
    `Unary(-, IntLit)` and `Unary(+, IntLit)`. Pre-fix, `[T; -N]` after
    substitution stayed `Unary(-, IntLit(N))` and `_resolve_size_expr`
    fell through to TyUnknown (`size expr Unary`) — silent
    miscount."""
    if not isinstance(expr, A.Unary):
        return expr
    if not isinstance(expr.operand, A.IntLit):
        return expr
    if expr.op == "-":
        return A.IntLit(span=expr.span,
                        value=-expr.operand.value,
                        type_suffix=None)
    if expr.op == "+":
        return expr.operand
    return expr


def _subst_shape_expr(expr: A.Expr, subst: dict[str, A.TyNode]) -> A.Expr:
    """Audit 28.8 B8: substitute size-kind generic params inside a tile
    or tensor shape expression.

    `subst` maps generic-param names to either TyNode (for type-kind)
    or — by convention used at the call site — a sentinel pseudo-type
    that wraps an IntLit (for size-kind, treated below). The shape AST
    uses Expr (typically Name or IntLit), so we walk and rewrite
    `Name(generic)` → `IntLit(value)` when the substituted value is a
    size literal.

    Pre-fix, substitute_ty for TyTile only handled `dtype` and shared
    `shape` + `memspace` directly — so `Tile<f32, [N], HBM>` mono'd
    against `N=128` produced a clone with `shape=[Name("N")]` (NOT
    `[IntLit(128)]`), and the lower-ast.py path defaulted shape[0] to
    0 when the leading element wasn't an IntLit. Trap 16003 is now
    reserved for that path.

    Audit 28.8 cycle 2 B:C11: after substitution, fold Binary(IntLit,
    op, IntLit) → IntLit so `[N*2; 16]` with N=64 becomes `[128; 16]`."""
    if expr is None:
        return expr
    if isinstance(expr, A.Name):
        # If the generic-param substitution provides a size literal,
        # replace with IntLit. Otherwise leave as-is.
        repl = subst.get(expr.name)
        if isinstance(repl, A.TyName) and repl.name.startswith("size_"):
            try:
                return A.IntLit(span=expr.span,
                                value=int(repl.name[len("size_"):]),
                                type_suffix=None)
            except ValueError:
                return expr
        # Audit 28.8 B8: callers may pass a raw IntLit through subst
        # by wrapping it as a TyArray-of-sorts, but the canonical
        # path is via _SizeLit (private sentinel below).
        if isinstance(repl, _SizeLitMarker):
            return A.IntLit(span=expr.span, value=repl.value,
                            type_suffix=None)
        return expr
    if isinstance(expr, A.Binary):
        folded = A.Binary(span=expr.span, op=expr.op,
                          left=_subst_shape_expr(expr.left, subst),
                          right=_subst_shape_expr(expr.right, subst))
        # Audit 28.8 cycle 2 B:C11: fold if both children are IntLits.
        return _fold_intlit_arith(folded)
    if isinstance(expr, A.Unary):
        folded = A.Unary(span=expr.span, op=expr.op,
                         operand=_subst_shape_expr(expr.operand, subst))
        # Audit 28.8 cycle 3 D5: post-fold the Unary so `-N` with N=5
        # becomes `IntLit(-5)` rather than `Unary(-, IntLit(5))`.
        return _fold_intlit_unary(folded)
    return expr


class _SizeLitMarker:
    """Internal sentinel for size-kind generic substitutions. Callers
    of substitute_ty pass `subst[N] = _SizeLitMarker(128)` to indicate
    a size-kind binding rather than a type binding. Code that walks
    `subst` and only handles TyNode treats this as opaque (it skips
    over it). Audit 28.8 B8."""
    __slots__ = ("value",)
    def __init__(self, value: int):
        self.value = value


def substitute_ty(t: A.TyNode, subst: dict[str, A.TyNode]) -> A.TyNode:
    """Substitute generic type variables with concrete types throughout t.

    Audit 28.8 B6: added TyPtr arm. Pre-fix, `fn deref<T>(p: *const T)
    -> T { unsafe { *p } }` mono'd with T=f64 silently left the param
    as `*const T` (TyName('T') inside), and the lowering path defaulted
    the inner element width to i32 — silent type-pun.

    Audit 28.8 B8: TyTile.shape and TyTensor.shape now have their
    Name(N) entries substituted when N is a size-kind generic. The
    `subst` dict carries `_SizeLitMarker(int)` sentinels for size
    bindings."""
    if isinstance(t, A.TyName):
        if t.name in subst:
            repl = subst[t.name]
            if isinstance(repl, _SizeLitMarker):
                # Size-kind generic used as TyName — convert to a
                # placeholder TyName encoding the value. Callers that
                # then walk types interpret `size_N` as a literal.
                return A.TyName(span=t.span, name=f"size_{repl.value}")
            return deepcopy(repl)
        return t
    if isinstance(t, A.TyTuple):
        return A.TyTuple(span=t.span, elems=[substitute_ty(e, subst) for e in t.elems])
    if isinstance(t, A.TyArray):
        # Audit 28.8 cycle 2 B:C8: substitute the size expression too,
        # mirroring TyTile/TyTensor shape sub. Pre-fix `[T; N]` mono'd
        # against {T=f64, N=8} produced `[f64; N]` (Name unchanged),
        # and downstream codegen took 0 as the silent default length.
        return A.TyArray(span=t.span,
                         elem=substitute_ty(t.elem, subst),
                         size=_subst_shape_expr(t.size, subst))
    if isinstance(t, A.TyRef):
        return A.TyRef(span=t.span, inner=substitute_ty(t.inner, subst), is_mut=t.is_mut)
    if isinstance(t, A.TyPtr):
        # Audit 28.8 B6 (trap reservation: handled at lower-ast level).
        return A.TyPtr(span=t.span,
                       inner=substitute_ty(t.inner, subst),
                       is_mut=t.is_mut)
    if isinstance(t, A.TyFn):
        return A.TyFn(span=t.span,
                      params=[substitute_ty(p, subst) for p in t.params],
                      ret=substitute_ty(t.ret, subst))
    if isinstance(t, A.TyTensor):
        # Audit 28.8 B8: substitute size-kind generics in shape exprs.
        new_shape = [_subst_shape_expr(s, subst) for s in t.shape]
        return A.TyTensor(span=t.span, dtype=substitute_ty(t.dtype, subst),
                          shape=new_shape, device=t.device, layout=t.layout)
    if isinstance(t, A.TyTile):
        # Audit 28.8 B8: same for TyTile.
        new_shape = [_subst_shape_expr(s, subst) for s in t.shape]
        return A.TyTile(span=t.span, dtype=substitute_ty(t.dtype, subst),
                        shape=new_shape, memspace=t.memspace)
    if isinstance(t, A.TyGeneric):
        return A.TyGeneric(span=t.span, base=t.base,
                           args=[substitute_ty(a, subst) for a in t.args])
    return t


def _walk_subst_expr(e: A.Expr, subst: dict[str, A.TyNode]) -> A.Expr:
    """Walk an expression tree, substituting types where they appear (Cast.target_ty,
    nested Block/If/Match/etc).
    """
    if isinstance(e, A.Cast):
        return A.Cast(span=e.span,
                      value=_walk_subst_expr(e.value, subst),
                      target_ty=substitute_ty(e.target_ty, subst))
    if isinstance(e, A.Block):
        return A.Block(span=e.span,
                       stmts=[_walk_subst_stmt(s, subst) for s in e.stmts],
                       final_expr=_walk_subst_expr(e.final_expr, subst) if e.final_expr is not None else None)
    if isinstance(e, A.If):
        else_ = e.else_
        if isinstance(else_, A.Block):
            else_ = _walk_subst_expr(else_, subst)
        elif isinstance(else_, A.If):
            else_ = _walk_subst_expr(else_, subst)
        return A.If(span=e.span,
                    cond=_walk_subst_expr(e.cond, subst),
                    then=_walk_subst_expr(e.then, subst),
                    else_=else_)
    if isinstance(e, A.Match):
        return A.Match(span=e.span,
                       scrutinee=_walk_subst_expr(e.scrutinee, subst),
                       arms=[A.MatchArm(span=a.span, pattern=a.pattern,
                                        guard=_walk_subst_expr(a.guard, subst) if a.guard else None,
                                        body=_walk_subst_expr(a.body, subst)) for a in e.arms])
    if isinstance(e, A.For):
        return A.For(span=e.span, var_name=e.var_name,
                     iter_expr=_walk_subst_expr(e.iter_expr, subst),
                     body=_walk_subst_expr(e.body, subst))
    if isinstance(e, A.While):
        return A.While(span=e.span,
                       cond=_walk_subst_expr(e.cond, subst),
                       body=_walk_subst_expr(e.body, subst))
    if isinstance(e, A.Loop):
        return A.Loop(span=e.span, body=_walk_subst_expr(e.body, subst))
    if isinstance(e, A.Binary):
        return A.Binary(span=e.span, op=e.op,
                        left=_walk_subst_expr(e.left, subst),
                        right=_walk_subst_expr(e.right, subst))
    if isinstance(e, A.Unary):
        return A.Unary(span=e.span, op=e.op, operand=_walk_subst_expr(e.operand, subst))
    if isinstance(e, A.Call):
        # Audit 28.8 cycle 3 D9: also substitute the callee's
        # turbofish generics list. Pre-fix, `caller[T]` calling
        # `id::<T>(x)` cloned to `caller__i32` with the body still
        # holding `Call(Name('id', generics=[T]))` — leaving an
        # unresolved type-var that downstream codegen interpreted as
        # the literal type name 'T'. Now T is substituted to i32 so
        # the mono iteration discovers `id::<i32>` and produces
        # `id__i32`.
        new_callee = e.callee
        if isinstance(new_callee, A.Name) and new_callee.generics:
            new_callee = A.Name(
                span=new_callee.span,
                name=new_callee.name,
                generics=[substitute_ty(g, subst)
                          for g in new_callee.generics],
            )
        return A.Call(span=e.span,
                      callee=_walk_subst_expr(new_callee, subst),
                      args=[_walk_subst_expr(a, subst) for a in e.args])
    if isinstance(e, A.Index):
        return A.Index(span=e.span,
                       callee=_walk_subst_expr(e.callee, subst),
                       indices=[_walk_subst_expr(a, subst) for a in e.indices])
    if isinstance(e, A.Field):
        return A.Field(span=e.span, obj=_walk_subst_expr(e.obj, subst), name=e.name)
    if isinstance(e, A.TupleLit):
        return A.TupleLit(span=e.span, elems=[_walk_subst_expr(x, subst) for x in e.elems])
    if isinstance(e, A.ArrayLit):
        return A.ArrayLit(span=e.span, elems=[_walk_subst_expr(x, subst) for x in e.elems])
    if isinstance(e, A.StructLit):
        return A.StructLit(span=e.span, name=e.name,
                           fields=[(n, _walk_subst_expr(v, subst)) for (n, v) in e.fields])
    if isinstance(e, A.Assign):
        return A.Assign(span=e.span, target=_walk_subst_expr(e.target, subst),
                        op=e.op, value=_walk_subst_expr(e.value, subst))
    if isinstance(e, A.Return):
        return A.Return(span=e.span,
                        value=_walk_subst_expr(e.value, subst) if e.value is not None else None)
    if isinstance(e, A.Break):
        return A.Break(span=e.span,
                       value=_walk_subst_expr(e.value, subst) if e.value is not None else None)
    if isinstance(e, A.Range):
        return A.Range(span=e.span,
                       start=_walk_subst_expr(e.start, subst) if e.start is not None else None,
                       end=_walk_subst_expr(e.end, subst) if e.end is not None else None)
    if isinstance(e, A.Quote):
        return A.Quote(span=e.span, inner=_walk_subst_expr(e.inner, subst))
    if isinstance(e, A.Splice):
        return A.Splice(span=e.span, inner=_walk_subst_expr(e.inner, subst))
    if isinstance(e, A.Modify):
        return A.Modify(span=e.span,
                        target=_walk_subst_expr(e.target, subst),
                        transformation=_walk_subst_expr(e.transformation, subst),
                        verifier=_walk_subst_expr(e.verifier, subst))
    # Audit 28.8 B12: UnsafeBlock — generics through unsafe regions
    # need substitution. Pre-fix, `fn read<T>(p: *const T) -> T {
    # unsafe { let x: T = *p; x } }` mono'd against T=f64 left the
    # inner `let x: T` unsubstituted; downstream lower-ast resolved
    # T as TyUnknown and silently defaulted the read to i32.
    if isinstance(e, A.UnsafeBlock):
        return A.UnsafeBlock(span=e.span,
                             body=_walk_subst_expr(e.body, subst))
    # Cycle 1 audit fix (Auditor 5 HIGH-1): TileLit was missing from
    # the expression-body walker. A generic fn like `fn alloc<T>() {
    # tile<T, [128], REG>::zeros() }` mono'd against T=f32 left both
    # the dtype (Name("T")) and the memspace expression unsubstituted
    # in the cloned body. Downstream `lower_ast` then resolved T to
    # TyUnknown and silently defaulted the element width — the same
    # class of defect Audit 28.8 B12 already fixed for UnsafeBlock.
    if isinstance(e, A.TileLit):
        return A.TileLit(
            span=e.span,
            dtype=substitute_ty(e.dtype, subst),
            shape=[_walk_subst_expr(s, subst) for s in e.shape],
            memspace=_walk_subst_expr(e.memspace, subst),
            init=e.init,
        )
    return e


def _walk_subst_stmt(s: A.Stmt, subst: dict[str, A.TyNode]) -> A.Stmt:
    if isinstance(s, A.Let):
        return A.Let(span=s.span, name=s.name, is_mut=s.is_mut,
                     ty=substitute_ty(s.ty, subst) if s.ty is not None else None,
                     value=_walk_subst_expr(s.value, subst) if s.value is not None else None)
    if isinstance(s, A.ExprStmt):
        return A.ExprStmt(span=s.span, expr=_walk_subst_expr(s.expr, subst))
    if isinstance(s, A.ConstStmt):
        return A.ConstStmt(span=s.span, name=s.name,
                           ty=substitute_ty(s.ty, subst),
                           value=_walk_subst_expr(s.value, subst))
    return s


# ============================================================================
# Call-site rewriting + instantiation
# ============================================================================
class Monomorphizer:
    def __init__(self, prog: A.Program):
        self.prog = prog
        self.generic_fns: dict[str, A.FnDecl] = {}
        for item in prog.items:
            if isinstance(item, A.FnDecl) and item.generics:
                self.generic_fns[item.name] = item
        self.instantiated: dict[tuple[str, str], A.FnDecl] = {}

    def run(self) -> int:
        """Run monomorphization. Returns count of new fns added.

        Generic fns with at least one turbofish call site get cloned per
        unique type-arg-tuple. Generic fns without any turbofish call sites
        (only un-annotated calls like `identity(42)`) are left in place so
        the legacy "T silently i32" lower path still works for backward
        compatibility.
        """
        if not self.generic_fns:
            return 0
        # Iteratively rewrite calls. Each pass may add new fns whose bodies
        # contain further generic calls. Fixed point when no new instantiations.
        #
        # Audit 28.8 cycle 5 C4-4 / HIGH: D9's cycle-3 fix was paper-only.
        # The unit test against `_walk_subst_expr` passes, but the end-to-
        # end pipeline still produced broken clones. Two key fixes:
        #
        # 1. Generic fns are NOT walked at top level. Their bodies are
        #    walked ONLY through `_instantiate`'s `_walk_subst_expr` (per
        #    clone, with the binding subst). If we rewrite generic fn
        #    bodies in-place during iteration, the rewrite replaces
        #    `id::<U>(v)` with `id__U(v)` (mangled, no turbofish) BEFORE
        #    `caller__i32`'s clone is made — the clone's `_walk_subst_expr`
        #    then has nothing to substitute (literal name `id__U`).
        #
        # 2. Clones must be re-walked in subsequent iterations so their
        #    own nested turbofish (post-subst) get followed. We promote
        #    new clones into the walk set each pass.
        promoted: list[A.FnDecl] = []
        changed = True
        while changed:
            changed = False
            # Walk only non-generic items + promoted clones (which are also
            # non-generic; they're concrete instantiations).
            for item in list(self.prog.items):
                if isinstance(item, A.FnDecl) and not item.generics:
                    new_body = self._rewrite_calls_in_block(item.body, item)
                    if new_body is not item.body:
                        item.body = new_body
                        changed = True
            for fn in list(promoted):
                # Promoted clones are non-generic by construction.
                new_body = self._rewrite_calls_in_block(fn.body, fn)
                if new_body is not fn.body:
                    fn.body = new_body
                    changed = True
            # Promote any newly-instantiated clones into the walk set so
            # the next iteration re-processes their bodies for further
            # nested-turbofish substitution.
            if self.instantiated:
                for key, fn in list(self.instantiated.items()):
                    if fn not in promoted:
                        promoted.append(fn)
                        changed = True
        # Append instantiated clones; keep original generic fns intact so
        # legacy un-turbofished call sites keep resolving.
        added = len(self.instantiated)
        self.prog.items = list(self.prog.items) + list(self.instantiated.values())
        return added

    def _rewrite_calls_in_block(self, blk: A.Block, owner: A.FnDecl) -> A.Block:
        new_stmts = []
        any_change = False
        for s in blk.stmts:
            ns = self._rewrite_calls_in_stmt(s, owner)
            if ns is not s:
                any_change = True
            new_stmts.append(ns)
        new_final = blk.final_expr
        if blk.final_expr is not None:
            new_final = self._rewrite_calls_in_expr(blk.final_expr, owner)
            if new_final is not blk.final_expr:
                any_change = True
        if any_change:
            return A.Block(span=blk.span, stmts=new_stmts, final_expr=new_final)
        return blk

    def _rewrite_calls_in_stmt(self, s: A.Stmt, owner: A.FnDecl) -> A.Stmt:
        if isinstance(s, A.Let):
            new_value = s.value
            if s.value is not None:
                new_value = self._rewrite_calls_in_expr(s.value, owner)
            if new_value is s.value:
                return s
            return A.Let(span=s.span, name=s.name, is_mut=s.is_mut, ty=s.ty, value=new_value)
        if isinstance(s, A.ExprStmt):
            new_e = self._rewrite_calls_in_expr(s.expr, owner)
            if new_e is s.expr:
                return s
            return A.ExprStmt(span=s.span, expr=new_e)
        if isinstance(s, A.ConstStmt):
            new_e = self._rewrite_calls_in_expr(s.value, owner)
            if new_e is s.value:
                return s
            return A.ConstStmt(span=s.span, name=s.name, ty=s.ty, value=new_e)
        return s

    def _rewrite_calls_in_expr(self, e: A.Expr, owner: A.FnDecl) -> A.Expr:
        if isinstance(e, A.Call):
            new_callee = self._rewrite_calls_in_expr(e.callee, owner)
            new_args = [self._rewrite_calls_in_expr(a, owner) for a in e.args]
            # Detect generic call site
            if isinstance(new_callee, A.Name) and new_callee.generics and new_callee.name in self.generic_fns:
                fn = self.generic_fns[new_callee.name]
                if len(new_callee.generics) != len(fn.generics):
                    # Mismatch — leave alone, typecheck will flag
                    return A.Call(span=e.span, callee=new_callee, args=new_args)
                # Build substitution map
                subst: dict[str, A.TyNode] = {}
                for gp, ty_arg in zip(fn.generics, new_callee.generics):
                    subst[gp.name] = ty_arg
                mangled = mangle(fn.name, new_callee.generics)
                key = (fn.name, mangled)
                if key not in self.instantiated:
                    self.instantiated[key] = self._instantiate(fn, mangled, subst)
                # Replace callee with non-generic Name
                new_callee = A.Name(span=new_callee.span, name=mangled, generics=[])
                return A.Call(span=e.span, callee=new_callee, args=new_args)
            if new_callee is e.callee and all(a is b for a, b in zip(new_args, e.args)):
                return e
            return A.Call(span=e.span, callee=new_callee, args=new_args)
        if isinstance(e, A.Block):
            return self._rewrite_calls_in_block(e, owner)
        if isinstance(e, A.If):
            new_cond = self._rewrite_calls_in_expr(e.cond, owner)
            new_then = self._rewrite_calls_in_expr(e.then, owner)
            new_else = e.else_
            if new_else is not None:
                new_else = self._rewrite_calls_in_expr(new_else, owner)
            if new_cond is e.cond and new_then is e.then and new_else is e.else_:
                return e
            return A.If(span=e.span, cond=new_cond, then=new_then, else_=new_else)
        if isinstance(e, A.Match):
            new_scrut = self._rewrite_calls_in_expr(e.scrutinee, owner)
            new_arms = []
            any_change = new_scrut is not e.scrutinee
            for arm in e.arms:
                new_body = self._rewrite_calls_in_expr(arm.body, owner)
                new_guard = arm.guard
                if arm.guard is not None:
                    new_guard = self._rewrite_calls_in_expr(arm.guard, owner)
                if new_body is arm.body and new_guard is arm.guard:
                    new_arms.append(arm)
                else:
                    any_change = True
                    new_arms.append(A.MatchArm(span=arm.span, pattern=arm.pattern,
                                               guard=new_guard, body=new_body))
            if any_change:
                return A.Match(span=e.span, scrutinee=new_scrut, arms=new_arms)
            return e
        if isinstance(e, A.For):
            new_it = self._rewrite_calls_in_expr(e.iter_expr, owner)
            new_body = self._rewrite_calls_in_expr(e.body, owner)
            if new_it is e.iter_expr and new_body is e.body:
                return e
            return A.For(span=e.span, var_name=e.var_name, iter_expr=new_it, body=new_body)
        if isinstance(e, A.While):
            new_cond = self._rewrite_calls_in_expr(e.cond, owner)
            new_body = self._rewrite_calls_in_expr(e.body, owner)
            if new_cond is e.cond and new_body is e.body:
                return e
            return A.While(span=e.span, cond=new_cond, body=new_body)
        if isinstance(e, A.Loop):
            new_body = self._rewrite_calls_in_expr(e.body, owner)
            if new_body is e.body:
                return e
            return A.Loop(span=e.span, body=new_body)
        if isinstance(e, A.Binary):
            new_l = self._rewrite_calls_in_expr(e.left, owner)
            new_r = self._rewrite_calls_in_expr(e.right, owner)
            if new_l is e.left and new_r is e.right:
                return e
            return A.Binary(span=e.span, op=e.op, left=new_l, right=new_r)
        if isinstance(e, A.Unary):
            new_op = self._rewrite_calls_in_expr(e.operand, owner)
            if new_op is e.operand:
                return e
            return A.Unary(span=e.span, op=e.op, operand=new_op)
        if isinstance(e, A.Cast):
            new_v = self._rewrite_calls_in_expr(e.value, owner)
            if new_v is e.value:
                return e
            return A.Cast(span=e.span, value=new_v, target_ty=e.target_ty)
        if isinstance(e, A.Index):
            new_callee = self._rewrite_calls_in_expr(e.callee, owner)
            new_idx = [self._rewrite_calls_in_expr(i, owner) for i in e.indices]
            if new_callee is e.callee and all(a is b for a, b in zip(new_idx, e.indices)):
                return e
            return A.Index(span=e.span, callee=new_callee, indices=new_idx)
        if isinstance(e, A.Field):
            new_obj = self._rewrite_calls_in_expr(e.obj, owner)
            if new_obj is e.obj:
                return e
            return A.Field(span=e.span, obj=new_obj, name=e.name)
        if isinstance(e, A.TupleLit):
            new_elems = [self._rewrite_calls_in_expr(x, owner) for x in e.elems]
            if all(a is b for a, b in zip(new_elems, e.elems)):
                return e
            return A.TupleLit(span=e.span, elems=new_elems)
        if isinstance(e, A.ArrayLit):
            new_elems = [self._rewrite_calls_in_expr(x, owner) for x in e.elems]
            if all(a is b for a, b in zip(new_elems, e.elems)):
                return e
            return A.ArrayLit(span=e.span, elems=new_elems)
        if isinstance(e, A.StructLit):
            new_fields = [(n, self._rewrite_calls_in_expr(v, owner)) for (n, v) in e.fields]
            if all(nv is ov for (_, nv), (_, ov) in zip(new_fields, e.fields)):
                return e
            return A.StructLit(span=e.span, name=e.name, fields=new_fields)
        if isinstance(e, A.Assign):
            new_t = self._rewrite_calls_in_expr(e.target, owner)
            new_v = self._rewrite_calls_in_expr(e.value, owner)
            if new_t is e.target and new_v is e.value:
                return e
            return A.Assign(span=e.span, target=new_t, op=e.op, value=new_v)
        if isinstance(e, A.Return):
            new_v = e.value
            if e.value is not None:
                new_v = self._rewrite_calls_in_expr(e.value, owner)
            if new_v is e.value:
                return e
            return A.Return(span=e.span, value=new_v)
        if isinstance(e, A.Break):
            new_v = e.value
            if e.value is not None:
                new_v = self._rewrite_calls_in_expr(e.value, owner)
            if new_v is e.value:
                return e
            return A.Break(span=e.span, value=new_v)
        if isinstance(e, A.Range):
            new_s = e.start
            new_end = e.end
            if e.start is not None:
                new_s = self._rewrite_calls_in_expr(e.start, owner)
            if e.end is not None:
                new_end = self._rewrite_calls_in_expr(e.end, owner)
            if new_s is e.start and new_end is e.end:
                return e
            return A.Range(span=e.span, start=new_s, end=new_end)
        if isinstance(e, A.Quote):
            new_inner = self._rewrite_calls_in_expr(e.inner, owner)
            if new_inner is e.inner:
                return e
            return A.Quote(span=e.span, inner=new_inner)
        if isinstance(e, A.Splice):
            new_inner = self._rewrite_calls_in_expr(e.inner, owner)
            if new_inner is e.inner:
                return e
            return A.Splice(span=e.span, inner=new_inner)
        if isinstance(e, A.Modify):
            new_t = self._rewrite_calls_in_expr(e.target, owner)
            new_tr = self._rewrite_calls_in_expr(e.transformation, owner)
            new_v = self._rewrite_calls_in_expr(e.verifier, owner)
            if new_t is e.target and new_tr is e.transformation and new_v is e.verifier:
                return e
            return A.Modify(span=e.span, target=new_t, transformation=new_tr, verifier=new_v)
        return e

    def _instantiate(self, fn: A.FnDecl, mangled: str, subst: dict[str, A.TyNode]) -> A.FnDecl:
        """Clone fn, substitute T-vars with concrete types, drop generics.

        Audit 28.8 B7: deep-copies where_clauses with substitution
        applied per clause's constraint expression. Pre-fix the clone
        shared the template's `where_clauses` list directly — so
        downstream passes that consumed the clone saw the template's
        unsubstituted clauses (e.g. `where T: Eq` with TyName('T')
        still present).

        Also propagates `is_extern` and `extern_abi`. Pre-fix the
        clone was implicitly non-extern, which meant
        `extern "C" fn malloc<T>(n: usize) -> *mut T` mono'd to
        `malloc__i32` would have been emitted as a normal user-fn
        with empty body → ud2 trap at runtime. Phase-0 still treats
        generic-over-extern as a parse-time refusal upstream, but
        the propagation is correct here for completeness."""
        new_params = [
            A.FnParam(span=p.span, name=p.name,
                      ty=substitute_ty(p.ty, subst), is_mut=p.is_mut)
            for p in fn.params
        ]
        new_ret = substitute_ty(fn.return_ty, subst) if fn.return_ty is not None else None
        new_body_block = _walk_subst_expr(fn.body, subst)
        if not isinstance(new_body_block, A.Block):
            new_body_block = fn.body
        new_where = [
            A.WhereClause(
                span=w.span,
                constraint=_walk_subst_expr(w.constraint, subst),
            )
            for w in fn.where_clauses
        ]
        return A.FnDecl(
            span=fn.span,
            name=mangled,
            generics=[],
            params=new_params,
            return_ty=new_ret,
            where_clauses=new_where,
            body=new_body_block,
            attrs=list(fn.attrs),
            is_pub=fn.is_pub,
            is_extern=fn.is_extern,
            extern_abi=fn.extern_abi,
        )


def monomorphize(prog: A.Program) -> int:
    """Run monomorphization on a program. Returns count of fns added."""
    return Monomorphizer(prog).run()


def monomorphize_safe(prog: A.Program) -> tuple[int, list[str]]:
    """Audit 28.8 cycle 4 C4-5 / E3: ShapeFoldError-safe entry point
    around `monomorphize`. Pre-fix the fn-mono path's uncaught raise
    was misattributed by the C3-3 outer wrapper as `internal error /
    compiler bug` — the user-facing diagnostic lost the trap-28801
    structured form that `monomorphize_structs` already produces.

    Returns (count, diags). On a clean run, diags is empty. On a
    ShapeFoldError, the caller should treat the diag as a typecheck
    error and abort the pipeline (callers that don't care can ignore
    diags; the count is 0 in that case).
    """
    try:
        return monomorphize(prog), []
    except ShapeFoldError as e:
        return 0, [str(e)]
