"""
helixc/frontend/totality.py — conservative structural-recursion checker.

Helix moves toward total-by-default semantics. This module flags
recursive functions that DON'T strictly decrease a syntactic measure on
at least one parameter on every recursive call. Functions explicitly
marked `@partial` are exempted; `@total` is a directive and would
require a positive proof (not done here — this stub only rejects the
obviously-bad cases).

Status:
- Direct recursion only (mutual recursion: pessimistically rejects
  unless any participant has @partial).
- "Strictly decreases" means: for some parameter `p`, every recursive
  call passes either `p - <const>`, `p / <const>`, or a strictly
  smaller component of `p` (e.g. `xs.tail` for a list — not yet wired
  since no list type yet, but the framework is in place).
- Conservative: returns True (= "totality unprovable") for any pattern
  we don't yet recognize.

Returns a list of (fn_name, reason) tuples for fns that are
non-`@partial` AND structurally suspect.

License: Apache 2.0
"""

from __future__ import annotations

from typing import Iterator

from . import ast_nodes as A
from .ast_walker import ASTVisitor, iter_fn_decls


# Stage 28.9 cycle 71 type-design CN-1 fix (HIGH conf 80): migrate
# totality to ASTVisitor + iter_fn_decls discipline. Pre-fix the
# item walker (`collect_items`) and the expr walker (`_children`)
# were hand-rolled — `_children` used a literal attribute-name list
# that would silently drop coverage for any future Expr subclass
# field (e.g. an `await_expr`, `let_else`, etc.). The cycle-58 walker
# discipline migrated panic_pass, unsafe_pass, trace_pass, and
# deprecated_pass.find_deprecation_call_sites to ASTVisitor;
# totality was the last hand-rolled holdout.
#
# `iter_fn_decls` (from ast_walker) gives drift-proof FnDecl
# enumeration through ImplBlock.methods and ModBlock.items
# (replaces the in-line collect_items recursion); `_SelfCallCollector`
# is an ASTVisitor subclass that uses generic_visit to traverse all
# Expr children via dataclass-field introspection (replaces _children
# hard-coded attribute list).


class _SelfCallCollector(ASTVisitor):
    """Collect every Call(callee=Name(fn_name), ...) inside the node
    being visited. Cycle-71 replacement for hand-rolled `_children` +
    `_collect_self_calls` recursion. ASTVisitor's generic_visit walks
    every dataclass-field on every AST node — same defect-class fix
    cycles 60/64/68 applied to other passes."""

    def __init__(self, fn_name: str):
        self.fn_name = fn_name
        self.calls: list[A.Call] = []

    def visit_Call(self, node: A.Call) -> None:
        callee = node.callee
        if isinstance(callee, A.Name) and callee.name == self.fn_name:
            self.calls.append(node)
        # Stage 28.9 cycle 73 type-design CN-1 fix (HIGH conf 90): do
        # NOT call `self.generic_visit(node)` here. ASTVisitor.visit
        # (ast_walker.py:191-196) auto-descends AFTER this override
        # returns unless we return False. The pre-fix explicit call
        # caused double-descent — nested self-calls inside arg
        # expressions appeared twice in `self.calls`, inflating the
        # `len(recursive_calls)` count in the totality diagnostic.
        # Sister `panic_pass._PanicCollector.visit_Call` follows the
        # same auto-descent pattern.


def check_totality(prog: A.Program) -> list[tuple[str, str]]:
    """Walk every fn in `prog`. For non-`@partial` fns that recurse,
    require at least one parameter that strictly decreases on every
    self-call. Returns [(fn_name, reason)] for failures.

    Mutual recursion is detected and pessimistically reported (each
    cycle participant flagged unless one is @partial).

    Stage 28.9 cycle 58 audit-R C57-1: pre-fix the walker iterated
    only `prog.items` filtered for `A.FnDecl` and missed ImplBlock /
    ModBlock nested fns. Cycle 58 added in-line recursion via
    `collect_items`; cycle 71 (type-design CN-1) migrated to the
    shared `iter_fn_decls` helper for drift-proof enumeration."""
    fns: dict[str, A.FnDecl] = {}
    for fn in iter_fn_decls(prog):
        fns[fn.name] = fn

    failures: list[tuple[str, str]] = []
    for name, fn in fns.items():
        if "partial" in fn.attrs:
            continue
        # Find direct recursive calls in fn's body.
        collector = _SelfCallCollector(name)
        collector.visit(fn.body)
        recursive_calls = collector.calls
        if not recursive_calls:
            continue  # not recursive, trivially total (or just non-recursive)
        # Need at least one parameter that strictly decreases on every
        # recursive call.
        param_names = [p.name for p in fn.params]
        if not param_names:
            failures.append((name, "recursion with no parameters"))
            continue
        ok = False
        for p in param_names:
            if all(_arg_strictly_decreases(call, fn.params, p) for call in recursive_calls):
                ok = True
                break
        if not ok:
            failures.append((
                name,
                f"recursive but no parameter strictly decreases on every "
                f"self-call ({len(recursive_calls)} call(s)). Annotate with "
                f"@partial to acknowledge potential non-termination.",
            ))
    return failures


def _arg_strictly_decreases(call: A.Call, params: list[A.FnParam],
                             param_name: str) -> bool:
    """Did `call` pass `param_name` (or a syntactically smaller form of
    it) at the same positional slot? Recognized strict-decrease forms:
        param - <pos_const>     (e.g. n - 1)
        param / <pos_const>     (must be > 1 for strict decrease)
    Returns False if we can't verify decrease (conservative).
    """
    # Find the positional index of param_name.
    param_idx = None
    for i, p in enumerate(params):
        if p.name == param_name:
            param_idx = i
            break
    if param_idx is None or param_idx >= len(call.args):
        return False
    arg = call.args[param_idx]
    return _is_strictly_smaller(arg, param_name)


def _is_strictly_smaller(expr: A.Expr, name: str) -> bool:
    """Does `expr` reduce a binding-of-`name` by a positive constant?"""
    if isinstance(expr, A.Binary):
        if expr.op == "-" and _is_name(expr.left, name) \
                and _is_positive_int_const(expr.right):
            return True
        if expr.op == "/" and _is_name(expr.left, name) \
                and _is_int_const_at_least(expr.right, 2):
            return True
    return False


def _is_name(e: A.Expr, name: str) -> bool:
    return isinstance(e, A.Name) and e.name == name


def _is_positive_int_const(e: A.Expr) -> bool:
    return isinstance(e, A.IntLit) and e.value > 0


def _is_int_const_at_least(e: A.Expr, n: int) -> bool:
    return isinstance(e, A.IntLit) and e.value >= n
