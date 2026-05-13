"""
helixc/frontend/ast_walker.py — shared AST traversal infrastructure.

Stage 28.8.2: extract the attribute-list dispatch pattern out of
``panic_pass``, ``unsafe_pass``, ``deprecated_pass``,
``grad_pass._expr_has_grad``, and ``struct_mono.visit_expr``. Before
this module, each of those passes hand-rolled its own ``_walk*``
helper with a literal list of attribute names. Audit cycles 1-11
caught the lists drifting ~4 different ways (see
docs/helix-pre-phase-A-finalization-research.md A1 / A5).

This library introduces a single source of truth: ``ASTVisitor``
uses ``dataclasses.fields(node)`` to enumerate every child slot of
the node by introspection, so adding a new field to an AST class
automatically extends the walker. There is no per-pass attribute
list to drift.

Design choices:

  * **No per-pass attribute lists.** ``generic_visit`` introspects the
    dataclass schema. The only way to lose coverage is to remove a
    field from the dataclass itself, which the type checker / tests
    would catch.

  * **Skip-marker via subclass override.** A ``visit_X`` method that
    returns ``False`` (the exact literal, not a falsy value) stops
    recursion into that subtree. This supports "I handled this node,
    don't descend further" without forcing every visitor to track
    descent state manually. (``None`` does NOT stop recursion — see
    ``generic_visit``.)

  * **Read-only walker, not rewriter.** ``ASTVisitor`` calls back on
    every node it encounters; it does NOT replace nodes in the tree.
    Rewriter passes (like ``grad_pass._rewrite_in_expr``) keep their
    bespoke recursion because the rewrite semantics differ per-node.

  * **Type fields are NOT walked by default.** ``Cast.target_ty``,
    ``Let.ty``, etc. carry ``TyNode`` data which most passes don't
    care about. Subclasses can override ``visit_Cast`` etc. to walk
    type fields explicitly if needed. (The pre-fix walkers all
    skipped type fields, so this matches their behavior.)

  * **Spans are not walked.** ``Span`` is a frozen dataclass; the
    pre-fix walkers all skipped it via the ``hasattr(sub, "span")``
    gate. ``generic_visit`` skips ``Span`` explicitly.

  * **MatchArm is a special case.** It's not an Expr/Stmt but holds
    Expr children (guard, body). Pass-callers that walk Match should
    rely on the Match visit recursing into arms via generic_visit's
    field iteration (which iterates ``arms`` and sees each MatchArm
    is a dataclass with guard/body slots).

License: Apache 2.0
"""

from __future__ import annotations

import dataclasses
from typing import Any

from . import ast_nodes as A


# Type fields are not walked by the default generic_visit — the pre-fix
# walkers all skipped them. Subclasses that need type-walking should
# override the relevant visit_X method explicitly.
_TYPE_FIELD_NAMES = frozenset({
    "ty",          # Let.ty, ConstStmt.ty, ConstDecl.ty
    "target_ty",   # Cast.target_ty
    "return_ty",   # FnDecl.return_ty
    "dtype",       # TileLit.dtype (TyNode — has internal exprs but skipped by default)
})

# Fields that always hold structural metadata, not child nodes worth
# visiting. NOTE: ``value`` is NOT in this list — Let/ConstStmt/Return/
# Break/Assign/ConstDecl all carry an Expr in ``value``, while
# IntLit/BoolLit/CharLit/StrLit/FloatLit carry a primitive. The
# ``_is_ast_node`` check at iteration time filters out primitives.
_NON_NODE_FIELD_NAMES = frozenset({
    "span",
    "module",        # str on Program
    "op",            # str on Binary/Unary/Assign (operator name)
    "type_suffix",   # str on IntLit/FloatLit
    "is_mut",        # bool on Let/TyRef/TyPtr
    "is_pub",        # bool on decls
    "is_extern",     # bool on FnDecl
    "extern_abi",    # str on FnDecl
    "attrs",         # list[str] on FnDecl
    "var_name",      # str on For
    "generics",      # list of TyNode on Name/FnDecl/StructDecl/EnumDecl
    "trait_name",    # Optional[str] on ImplBlock
})


def _is_ast_node(obj: Any) -> bool:
    """Return True iff ``obj`` is an AST node that may carry child
    nodes worth visiting. Excludes:

      * ``None``
      * Primitives (``str``, ``int``, ``bool``, ``float``)
      * ``Span`` (frozen, no children)
      * ``TyNode`` subclasses (type fields skipped by default)
    """
    if obj is None:
        return False
    if isinstance(obj, (str, int, bool, float, tuple)):
        # tuples on a field would be inspected via _is_iterable_of_nodes below;
        # bare scalar tuples like `(0, 0)` should not be walked.
        return False
    if isinstance(obj, A.Span):
        return False
    if isinstance(obj, A.TyNode):
        return False
    # Anything else that is a dataclass instance: treat as walkable.
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def _iter_child_nodes(node: Any) -> "Iterable[Any]":
    """Yield every AST child of ``node`` reachable via its dataclass fields.

    For list-valued fields, yields each element. For tuple-valued
    fields holding AST nodes (e.g. ``StructLit.fields`` which is
    ``list[tuple[str, Expr]]``), yields each tuple element that is
    itself a node.

    Skips ``Span``, ``TyNode``, and non-node primitive fields per the
    module-level filters above.
    """
    if not dataclasses.is_dataclass(node) or isinstance(node, type):
        return
    for f in dataclasses.fields(node):
        if f.name in _NON_NODE_FIELD_NAMES:
            continue
        if f.name in _TYPE_FIELD_NAMES:
            continue
        val = getattr(node, f.name, None)
        if val is None:
            continue
        if _is_ast_node(val):
            yield val
        elif isinstance(val, list):
            for item in val:
                if _is_ast_node(item):
                    yield item
                elif isinstance(item, tuple):
                    for sub in item:
                        if _is_ast_node(sub):
                            yield sub
        elif isinstance(val, tuple):
            for sub in val:
                if _is_ast_node(sub):
                    yield sub


class ASTVisitor:
    """Base class for AST traversal passes.

    Subclasses override ``visit_<NodeType>`` for nodes of interest.
    Any node without an override is dispatched to ``generic_visit``,
    which recurses into every child of the node via dataclass-field
    introspection.

    Return ``False`` from a ``visit_X`` method to stop recursion into
    that subtree. Returning anything else (including ``None``) causes
    the visitor to also call ``generic_visit(node)`` after the
    override returns — so an override sees the node first, then its
    children. (This matches the post-order convenience of the pre-fix
    walkers: callback-first, then descend.)

    To make the override solely responsible for descent (e.g. when it
    needs to push/pop a context frame around the recursion), return
    ``False`` and call ``self.generic_visit(node)`` explicitly inside
    the override.

    See ``unsafe_pass.UnsafeVisitor`` for the context-frame pattern.
    See ``panic_pass.PanicVisitor`` for the simple-callback pattern.
    """

    def visit(self, node: Any) -> Any:
        """Dispatch ``node`` to its ``visit_<TypeName>`` handler if one
        exists, otherwise to ``generic_visit``. Returns whatever the
        handler returned (callers usually ignore the result; ``False``
        skips post-visit descent).
        """
        if node is None:
            return None
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method is None:
            return self.generic_visit(node)
        result = method(node)
        # If the override returned False, treat it as "I'll handle
        # descent myself". Otherwise, recurse into the subtree.
        if result is False:
            return result
        self.generic_visit(node)
        return result

    def generic_visit(self, node: Any) -> None:
        """Recurse into every child of ``node`` discovered via
        ``dataclasses.fields`` introspection. Skips ``Span``,
        ``TyNode``, and non-node primitive fields by construction.

        Subclasses generally do NOT override this — the whole point of
        the shared walker is to remove per-pass attribute lists. If
        you find yourself needing to customise descent for a specific
        node type, override ``visit_<NodeType>`` and call
        ``self.generic_visit(node)`` (or skip it) yourself.
        """
        for child in _iter_child_nodes(node):
            self.visit(child)


def iter_fn_decls(prog) -> "Iterable":
    """Stage 28.9 cycle 60 audit-R C59-1 helper: yield every `A.FnDecl`
    reachable from `prog`, recursing through `A.ImplBlock.methods` and
    `A.ModBlock.items`. Centralises the walker-drift discipline so
    Item-level passes (panic, unsafe, trace, and partially deprecated /
    totality) share the same item-walk surface — a future Item subclass
    that holds FnDecls forces an explicit dispatch decision in ONE
    place instead of N.

    **Pre-/post-flatten contract** (cycle-61 CN-2 follow-up — adds the
    docstring text that the cycle-60 commit message claimed but never
    actually inserted, then cycle-63 CN-2 corrected the pipeline
    claim after silent-failure audit empirically verified the gap):

    Safe to call EITHER pre-flatten OR post-flatten — both work.
    - Pre-flatten: recurses through ImplBlock.methods + ModBlock.items
      to expose nested fns to scanners. Load-bearing — do NOT remove
      the recursion arms thinking they're dead code.
    - Post-flatten (after `flatten_impls` + `flatten_modules`): the
      impl/mod branches are no-ops because the prior passes have
      already lifted those items to top-level FnDecls. The recursion
      adds zero work in this case; only top-level items remain.

    Driver call ordering (cycle-63 CN-2 correction):
    - `helixc/check.py` runs `flatten_impls` + `flatten_modules`
      (cycle-63 added the second) BEFORE invoking panic/unsafe/trace/
      deprecated/totality passes. So by cycle-63 it consumes the
      post-flatten case.
    - `helixc/backend/x86_64.py` also runs both flatten passes before
      monomorphize/codegen. Same post-flatten case.
    Pre-cycle-63 `helixc check` did NOT run flatten_modules, so the
    ModBlock recursion was load-bearing there; the cycle-63 fix
    aligned both drivers on the same post-flatten invariant. The
    pre-flatten case remains supported for direct-API callers
    (tests, REPL, ad-hoc tools) that bypass the canonical driver
    sequence.

    Pre-cycle-58 every pass iterated `prog.items` filtered for
    `isinstance(it, A.FnDecl)` and missed mod-/impl-nested fns in the
    `helixc check` surface tool. cycle-58 fixed deprecated_pass +
    totality in-place; cycle-60 routes panic/unsafe/trace through
    this helper. deprecated_pass.find_deprecated_decls remains
    separate because it collects ALL decl kinds (not just FnDecl)
    and was further restricted to post-flatten-only in cycle-61
    after the C59-3 recursion was found to introduce a name-
    collision bug (CN-1).
    """
    from . import ast_nodes as A

    def _walk(items):
        for it in items:
            if isinstance(it, A.FnDecl):
                yield it
            elif isinstance(it, A.ImplBlock):
                for m in it.methods:
                    if isinstance(m, A.FnDecl):
                        yield m
            elif isinstance(it, A.ModBlock):
                yield from _walk(it.items)

    yield from _walk(prog.items)


__all__ = ["ASTVisitor", "iter_fn_decls"]
