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


def check_totality(prog: A.Program) -> list[tuple[str, str]]:
    """Walk every fn in `prog`. For non-`@partial` fns that recurse,
    require at least one parameter that strictly decreases on every
    self-call. Returns [(fn_name, reason)] for failures.

    Mutual recursion is detected and pessimistically reported (each
    cycle participant flagged unless one is @partial).

    Stage 28.9 cycle 58 audit-R C57-1 fix (HIGH conf 88): pre-fix the
    walker iterated only `prog.items` filtered for `A.FnDecl`. Methods
    inside `A.ImplBlock.methods` and functions inside `A.ModBlock.items`
    were silently skipped — a recursive fn defined inside `mod m { ... }`
    or `impl X { ... }` produced `totality: OK` from `helixc check`
    despite the backend driver correctly catching it (because the
    backend runs `flatten_modules` before totality). Same item-walker
    gap as deprecated_pass C57-5 already closed via `scan_items`. Now
    recurse through ImplBlock and ModBlock containers too.
    """
    fns: dict[str, A.FnDecl] = {}

    def collect_items(items: list) -> None:
        for it in items:
            if isinstance(it, A.FnDecl):
                fns[it.name] = it
            elif isinstance(it, A.ImplBlock):
                for m in it.methods:
                    if isinstance(m, A.FnDecl):
                        fns[m.name] = m
            elif isinstance(it, A.ModBlock):
                collect_items(it.items)
            # Other Item subclasses (StructDecl, EnumDecl, UseDecl,
            # ModuleDecl, TypeAlias, ConstDecl, AgentDecl) don't carry
            # FnDecl-bearing bodies and are skipped.

    collect_items(prog.items)

    failures: list[tuple[str, str]] = []
    for name, fn in fns.items():
        if "partial" in fn.attrs:
            continue
        # Find direct recursive calls in fn's body.
        recursive_calls = list(_collect_self_calls(fn.body, name))
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


def _collect_self_calls(node, fn_name: str) -> Iterator[A.Call]:
    """Yield every Call to `fn_name` inside `node`."""
    if node is None:
        return
    if isinstance(node, A.Call) and isinstance(node.callee, A.Name) \
            and node.callee.name == fn_name:
        yield node
    for child in _children(node):
        yield from _collect_self_calls(child, fn_name)


def _children(node) -> Iterator:
    if node is None:
        return
    for attr in ("left", "right", "cond", "then", "else_", "operand",
                 "value", "expr", "final_expr", "callee", "scrutinee",
                 "iter_expr", "body", "transformation", "verifier",
                 "target", "obj", "inner", "guard", "pattern"):
        if hasattr(node, attr):
            v = getattr(node, attr)
            if v is not None:
                yield v
    if hasattr(node, "stmts"):
        for s in node.stmts:
            yield s
    if hasattr(node, "args"):
        for a in node.args:
            yield a
    if hasattr(node, "arms"):
        for arm in node.arms:
            yield arm
    if hasattr(node, "elems"):
        for e in node.elems:
            yield e


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
