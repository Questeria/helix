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
from .ast_walker import ASTVisitor


TRAP_UNSAFE_OP_OUTSIDE = 28601
TRAP_EXTERN_CALL_OUTSIDE_UNSAFE = 28602


# Stage 28.8.2: unsafe_pass walker migrated to ASTVisitor base class.
# Pre-fix this module hand-rolled `_walk(node, callback, context)` with
# the same attribute-list drift problem as panic_pass (cycles 1-2 C1-H1
# / A6 / A5 fix-sweeps had to manually synchronize the three lists).
# The shared library traverses dataclass fields by introspection; the
# in_unsafe context is now a Python-instance attribute (push/pop in
# visit_UnsafeBlock via the skip-marker / explicit-descent pattern).


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


class _UnsafeBlockCollector(ASTVisitor):
    """Stage 28.8.2: collect every UnsafeBlock node into a list."""

    def __init__(self, out: list[A.UnsafeBlock]):
        self.out = out

    def visit_UnsafeBlock(self, node: A.UnsafeBlock) -> None:
        self.out.append(node)
        # generic_visit recurses into node.body after this returns.


def find_unsafe_blocks(prog: A.Program) -> list[A.UnsafeBlock]:
    """Return all UnsafeBlock nodes in the program (in fn bodies).

    Stage 28.8.2: uses ``_UnsafeBlockCollector(ASTVisitor)``.
    Stage 28.9 cycle 60 audit-R C59-1: iter_fn_decls recursion through
    ImplBlock/ModBlock so mod-nested fns are scanned.
    """
    out: list[A.UnsafeBlock] = []
    from .ast_walker import iter_fn_decls
    for fn in iter_fn_decls(prog):
        if not fn.is_extern:
            _UnsafeBlockCollector(out).visit(fn.body)
    return out


class _RawPtrOpVisitor(ASTVisitor):
    """Stage 28.8.2: walk a fn body recording every raw-ptr op
    (Unary deref or pointer Cast) along with whether the call site
    is inside an enclosing UnsafeBlock.

    The ``in_unsafe`` flag is tracked via a per-instance counter
    pushed/popped around ``visit_UnsafeBlock`` recursion. Because
    UnsafeBlocks can nest (cycle-2 audit verified this is parser-
    legal), we use a counter rather than a boolean — every entry
    increments, every exit decrements, the predicate is `counter > 0`.

    Uses the skip-marker pattern (``return False`` + explicit
    ``self.generic_visit(node)``) so the override owns descent and
    the unsafe-frame stays balanced.
    """

    def __init__(self, fn_name: str,
                 out: list[tuple[str, A.Span, bool]]):
        self.fn_name = fn_name
        self.out = out
        # Unsafe-frame depth. > 0 means we're inside one or more
        # nested UnsafeBlocks; raw-ptr ops at this depth are legal.
        self._unsafe_depth = 0

    @property
    def _in_unsafe(self) -> bool:
        return self._unsafe_depth > 0

    def visit_UnsafeBlock(self, node: A.UnsafeBlock):
        self._unsafe_depth += 1
        try:
            # Manually descend into the body so the frame stays in
            # scope. Return False to suppress the post-visit
            # generic_visit (which would otherwise double-descend).
            self.generic_visit(node)
        finally:
            self._unsafe_depth -= 1
        return False

    def visit_Unary(self, node: A.Unary) -> None:
        if _is_raw_ptr_op(node):
            self.out.append((self.fn_name, node.span, self._in_unsafe))

    def visit_Cast(self, node: A.Cast) -> None:
        if _is_raw_ptr_op(node):
            self.out.append((self.fn_name, node.span, self._in_unsafe))


def find_raw_ptr_ops(prog: A.Program) -> list[tuple[str, A.Span, bool]]:
    """List every syntactic raw-ptr op (Phase-0: Unary deref or
    pointer Cast) in fn bodies, with (fn_name, span, in_unsafe).

    Stage 28.8.2: uses ``_RawPtrOpVisitor(ASTVisitor)`` — the
    ``in_unsafe`` flag is tracked via a push/pop counter around
    UnsafeBlock recursion.

    Stage 28.9 cycle 60 audit-R C59-1: iter_fn_decls recursion.
    """
    out: list[tuple[str, A.Span, bool]] = []
    from .ast_walker import iter_fn_decls
    for fn in iter_fn_decls(prog):
        if not fn.is_extern:
            _RawPtrOpVisitor(fn.name, out).visit(fn.body)
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
