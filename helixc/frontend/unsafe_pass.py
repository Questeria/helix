"""
helixc/frontend/unsafe_pass.py — Stage 28.6: unsafe block.

`unsafe { ... }` is an explicit capability boundary: raw-pointer ops
(deref, arithmetic), FFI calls without effect-check, and untyped
memcpy may only appear inside an `unsafe` block. Outside, they trap
28601.

Trap-id reservations:
  * 28601 — raw-ptr deref / arithmetic outside `unsafe`
  * 28602 — extern "C" call outside `unsafe` (Phase-0 lenient; trap
            reserved for future strict mode)

Phase-0:
  * Parser supports `unsafe { ... }` (existing _parse_primary).
  * `find_unsafe_blocks(prog)` lists every UnsafeBlock with its span.
  * `find_raw_ptr_ops(prog)` lists every Unary('*'), Binary on pointer
    types, etc. (Phase-0: just classifies syntactic forms; no type
    info needed for the basic check).
  * `check_unsafe_ops(prog)` returns diagnostics for raw-ptr ops
    outside any enclosing unsafe block.

License: Apache 2.0
"""

from __future__ import annotations

from typing import Optional

from . import ast_nodes as A


TRAP_UNSAFE_OP_OUTSIDE = 28601
TRAP_EXTERN_CALL_OUTSIDE_UNSAFE = 28602


def _walk(node, callback, context: Optional[dict] = None) -> None:
    """Recursive walk, threading a context dict so children can know
    whether they are inside an UnsafeBlock. The context's 'in_unsafe'
    key is True iff we're currently descending under an UnsafeBlock."""
    if node is None:
        return
    if context is None:
        context = {"in_unsafe": False}

    # Push unsafe-frame if this is an UnsafeBlock
    if isinstance(node, A.UnsafeBlock):
        prev = context["in_unsafe"]
        context["in_unsafe"] = True
        callback(node, context)
        _walk(node.body, callback, context)
        context["in_unsafe"] = prev
        return

    callback(node, context)

    # Generic descent
    if isinstance(node, A.Block):
        for s in node.stmts:
            _walk(s, callback, context)
        if node.final_expr is not None:
            _walk(node.final_expr, callback, context)
        return

    # Audit 28.8: attr list aligned with panic_pass / deprecated_pass
    # so the three walkers no longer drift. `then_branch`/`else_branch`
    # are NOT AST attrs — they were a legacy mismatch (same family of
    # bugs as panic_pass's pre-fix list). Removed. Added `iter_expr`,
    # `obj`, `target`, `start`, `end`, `guard`, `inner`, `transformation`,
    # `verifier` so Cast/For/Range/MatchArm.guard subexprs are visited.
    for attr in ("expr", "left", "right", "operand", "cond", "then",
                 "else_", "value", "scrutinee", "callee", "init",
                 "rhs", "body", "iter_expr", "obj", "target",
                 "start", "end", "guard", "inner", "transformation",
                 "verifier"):
        sub = getattr(node, attr, None)
        if sub is not None and hasattr(sub, "span"):
            _walk(sub, callback, context)

    # `indices` added so `arr[expr]` is visited.
    for attr in ("args", "stmts", "fields", "elems", "arms", "indices"):
        seq = getattr(node, attr, None)
        if seq is None:
            continue
        try:
            for it in seq:
                if isinstance(it, tuple):
                    for sub in it:
                        if hasattr(sub, "span"):
                            _walk(sub, callback, context)
                elif hasattr(it, "span"):
                    _walk(it, callback, context)
        except TypeError:
            # Re-raise rather than swallow: silently skipping iteration
            # errors masks AST-shape regressions in later stages.
            raise


def find_unsafe_blocks(prog: A.Program) -> list[A.UnsafeBlock]:
    """Return all UnsafeBlock nodes in the program (in fn bodies)."""
    out: list[A.UnsafeBlock] = []
    for it in prog.items:
        if isinstance(it, A.FnDecl) and not it.is_extern:
            def cb(n, ctx, out=out):
                if isinstance(n, A.UnsafeBlock):
                    out.append(n)
            _walk(it.body, cb)
    return out


def _is_raw_ptr_op(node) -> bool:
    """A syntactic predicate for raw-ptr ops. Phase-0:
      * Unary('*', operand) — deref
      * Cast with target_ty = TyPtr — raw-pointer cast (`x as *mut T`)
        (Audit 28.8 B3: previously only Unary was matched, so casts
        flowed past every unsafe gate; the typecheck-level gate now
        emits trap 28603 directly, but we ALSO surface here so
        existing pipelines that depend on check_unsafe_ops continue
        to see ptr-casts as ops requiring `unsafe`.)
      * Binary on a TyPtr-typed operand — arithmetic (we don't have
        type info here; this would need integration with typecheck).
    Phase-0 returns True for syntactic `*ptr` (Unary deref) OR
    `x as *T` (Cast with TyPtr target).
    """
    if isinstance(node, A.Unary) and node.op == "*":
        return True
    if isinstance(node, A.Cast) and isinstance(node.target_ty, A.TyPtr):
        return True
    return False


def find_raw_ptr_ops(prog: A.Program) -> list[tuple[str, A.Span, bool]]:
    """List every syntactic raw-ptr op (Phase-0: Unary deref) in fn
    bodies, with (fn_name, span, in_unsafe)."""
    out: list[tuple[str, A.Span, bool]] = []
    for it in prog.items:
        if isinstance(it, A.FnDecl) and not it.is_extern:
            def cb(n, ctx, out=out, fname=it.name):
                if _is_raw_ptr_op(n):
                    out.append((fname, n.span, ctx["in_unsafe"]))
            _walk(it.body, cb)
    return out


def check_unsafe_ops(prog: A.Program) -> list[str]:
    """Return diagnostic strings for raw-ptr ops *outside* any
    enclosing unsafe block."""
    diags: list[str] = []
    for fname, span, in_unsafe in find_raw_ptr_ops(prog):
        if not in_unsafe:
            diags.append(
                f"{span.line}:{span.col}: raw-pointer op in fn {fname!r} "
                f"outside unsafe block (trap 28601)"
            )
    return diags
