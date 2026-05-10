"""
helixc/frontend/panic_pass.py — Stage 28.5: panic / abort policy.

`panic("msg")` is a builtin call that emits a trap with id 28501 and
the message string in `.rodata` (for the runtime to print before
aborting). Phase-0 default: abort (no unwinding).

The `@unwind` fn attribute is reserved for future setjmp/longjmp-based
unwinding; Phase-0 parses but emits a diagnostic (trap 28502).

Trap-id reservations:
  * 28501 — panic invoked (emitted by `panic("...")` calls)
  * 28502 — @unwind attribute not yet supported

Phase-0:
  * Builtin name `panic` registered in typecheck _BUILTIN_NAMES.
  * `collect_panics(prog)` returns list of (fn_name, span, msg) for
    each `panic("...")` call site found.
  * `validate_panic_args(prog)` enforces:
      - panic takes exactly one arg
      - the arg must be a string literal (StrLit)
  * `find_unwind_attrs(prog)` lists fns carrying @unwind (which
    Phase-0 rejects with a diagnostic).

License: Apache 2.0
"""

from __future__ import annotations

from typing import Optional

from . import ast_nodes as A


TRAP_PANIC_INVOKED = 28501
TRAP_UNWIND_NOT_SUPPORTED = 28502


def _walk_exprs(node, callback) -> None:
    """Recursive walk over a Block / Expr / Stmt, calling
    `callback(expr)` for each expression node encountered. Phase-0
    conservative — handles the common forms used in panic detection.

    Audit 28.8 fixes:
      * **C1-H1 / A6** (HIGH): scalar-attr list now matches the canonical
        AST schema in `ast_nodes.py`. `A.If` has `then` / `else_` (not
        `then_branch` / `else_branch` — those names don't exist on any
        node), and `A.For` has `iter_expr`. Without these, `panic()` calls
        inside `if` / `else` / `for` were silently invisible to
        `collect_panics` / `validate_panic_args`.
      * **C1-L1** (LOW): callback only fires on actual expression nodes
        (`isinstance(node, A.Expr)`). Previously `hasattr(node, "span")`
        also caught `A.Block` and `A.ExprStmt` — no false-positives today
        (because `_is_panic_call` rejects non-Calls), but fragile against
        future callback changes.
    """
    if node is None:
        return
    # Invoke callback ONLY on real expression nodes — not statement
    # wrappers or block containers. (C1-L1.)
    if isinstance(node, A.Expr):
        callback(node)
    # Block has stmts + final_expr
    if isinstance(node, A.Block):
        for s in node.stmts:
            _walk_exprs(s, callback)
        if node.final_expr is not None:
            _walk_exprs(node.final_expr, callback)
        return
    # ExprStmt wraps an expr
    expr_attr = getattr(node, "expr", None)
    if expr_attr is not None and hasattr(expr_attr, "span"):
        _walk_exprs(expr_attr, callback)
    # Recurse into common sub-expression containers. Attr names match
    # `ast_nodes.py`: If has `then`/`else_`; For has `iter_expr`.
    for attr in ("left", "right", "operand", "cond", "then", "else_",
                 "value", "scrutinee", "callee", "init",
                 "rhs", "body", "iter_expr", "obj", "target",
                 "start", "end", "guard", "inner", "transformation",
                 "verifier"):
        sub = getattr(node, attr, None)
        if sub is not None and hasattr(sub, "span"):
            _walk_exprs(sub, callback)
    for attr in ("args", "stmts", "fields", "elems", "arms",
                 "indices"):
        seq = getattr(node, attr, None)
        if seq is None:
            continue
        try:
            for it in seq:
                if isinstance(it, tuple):
                    for sub in it:
                        if hasattr(sub, "span"):
                            _walk_exprs(sub, callback)
                elif hasattr(it, "span"):
                    _walk_exprs(it, callback)
        except TypeError:
            # Re-raise rather than silently swallow: a TypeError on
            # iteration means the AST shape changed in a way we didn't
            # anticipate; hiding it would mask real walker bugs.
            raise
    # Final expression of a block (when not handled above)
    final = getattr(node, "final_expr", None)
    if final is not None and hasattr(final, "span") and not isinstance(node, A.Block):
        _walk_exprs(final, callback)


def _is_panic_call(expr) -> bool:
    if not isinstance(expr, A.Call):
        return False
    callee = expr.callee
    return isinstance(callee, A.Name) and callee.name == "panic"


def collect_panics(prog: A.Program) -> list[tuple[str, A.Span, Optional[str]]]:
    """For each `panic("...")` call in any fn body, return
    (fn_name, span, msg_or_None)."""
    out: list[tuple[str, A.Span, Optional[str]]] = []
    for it in prog.items:
        if not isinstance(it, A.FnDecl) or it.is_extern:
            continue

        def cb(e, fn_name=it.name):
            if _is_panic_call(e):
                msg: Optional[str] = None
                if e.args and isinstance(e.args[0], A.StrLit):
                    msg = e.args[0].value
                out.append((fn_name, e.span, msg))

        _walk_exprs(it.body, cb)
    return out


def validate_panic_args(prog: A.Program) -> list[str]:
    """For every `panic(...)` call, enforce single-string-literal arg.
    Returns diagnostic strings."""
    diags: list[str] = []
    for it in prog.items:
        if not isinstance(it, A.FnDecl) or it.is_extern:
            continue

        def cb(e, fn_name=it.name, diags=diags):
            if not _is_panic_call(e):
                return
            if len(e.args) != 1:
                diags.append(
                    f"{e.span.line}:{e.span.col}: panic in {fn_name!r}: "
                    f"expected 1 arg, got {len(e.args)}"
                )
                return
            arg = e.args[0]
            if not isinstance(arg, A.StrLit):
                diags.append(
                    f"{e.span.line}:{e.span.col}: panic in {fn_name!r}: "
                    f"arg must be a string literal"
                )
        _walk_exprs(it.body, cb)
    return diags


def find_unwind_attrs(prog: A.Program) -> list[A.FnDecl]:
    """Return all fns carrying @unwind. Phase-0 emits diagnostics for
    these (the attribute is reserved but not yet implemented)."""
    out: list[A.FnDecl] = []
    for it in prog.items:
        if isinstance(it, A.FnDecl) and "unwind" in it.attrs:
            out.append(it)
    return out


def validate_unwind(prog: A.Program) -> list[str]:
    """Diagnostic for each @unwind use (trap 28502)."""
    diags: list[str] = []
    for fn in find_unwind_attrs(prog):
        diags.append(
            f"{fn.span.line}:{fn.span.col}: @unwind on fn {fn.name!r}: "
            f"not yet supported (trap 28502 reserved)"
        )
    return diags
