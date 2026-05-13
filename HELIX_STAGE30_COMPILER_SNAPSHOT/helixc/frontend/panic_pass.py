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
from .ast_walker import ASTVisitor


TRAP_PANIC_INVOKED = 28501
TRAP_UNWIND_NOT_SUPPORTED = 28502


# Stage 28.8.2: panic_pass walker migrated to ASTVisitor base class.
# Pre-fix this module hand-rolled `_walk_exprs` with a literal list of
# attribute names that audit cycles 1-3 caught drifting (C1-H1 / A6 / A5).
# The shared library introspects dataclass fields, so adding new Expr
# subtypes or fields no longer silently drops walker coverage.


def _is_panic_call(expr) -> bool:
    if not isinstance(expr, A.Call):
        return False
    callee = expr.callee
    return isinstance(callee, A.Name) and callee.name == "panic"


class _PanicCollector(ASTVisitor):
    """Stage 28.8.2: collect every `panic("...")` call site as
    (fn_name, span, msg_or_None) tuples. Drop-in replacement for the
    pre-fix `_walk_exprs` + bespoke callback closure.
    """

    def __init__(self, fn_name: str,
                 out: list[tuple[str, A.Span, Optional[str]]]):
        self.fn_name = fn_name
        self.out = out

    def visit_Call(self, node: A.Call) -> None:
        # Record panic("..."); generic_visit handles recursing into the
        # callee + args after this returns.
        if _is_panic_call(node):
            msg: Optional[str] = None
            if node.args and isinstance(node.args[0], A.StrLit):
                msg = node.args[0].value
            self.out.append((self.fn_name, node.span, msg))


def collect_panics(prog: A.Program) -> list[tuple[str, A.Span, Optional[str]]]:
    """For each `panic("...")` call in any fn body, return
    (fn_name, span, msg_or_None).

    Stage 28.8.2: uses ``_PanicCollector(ASTVisitor)`` rather than the
    pre-fix `_walk_exprs` helper — the shared library is drift-proof
    against future AST field additions.
    """
    out: list[tuple[str, A.Span, Optional[str]]] = []
    # Stage 28.9 cycle 60 audit-R C59-1 fix (HIGH conf 88): iter_fn_decls
    # recurses through ImplBlock.methods and ModBlock.items so panic call
    # sites inside `mod m { fn x() { panic("...") } }` are caught even
    # in the `helixc check` surface tool (which does not run
    # flatten_modules before this pass). Same walker-drift defect class
    # as C57-1..C57-5.
    from .ast_walker import iter_fn_decls
    for fn in iter_fn_decls(prog):
        if fn.is_extern:
            continue
        _PanicCollector(fn.name, out).visit(fn.body)
    return out


class _PanicArgsValidator(ASTVisitor):
    """Stage 28.8.2: emit a diagnostic for every panic(...) call whose
    args don't match the contract (exactly one StrLit). Replaces the
    pre-fix `_walk_exprs` + closure callback pattern.
    """

    def __init__(self, fn_name: str, diags: list[str]):
        self.fn_name = fn_name
        self.diags = diags

    def visit_Call(self, node: A.Call) -> None:
        if not _is_panic_call(node):
            return
        if len(node.args) != 1:
            self.diags.append(
                f"{node.span.line}:{node.span.col}: panic in {self.fn_name!r}: "
                f"expected 1 arg, got {len(node.args)}"
            )
            return
        arg = node.args[0]
        if not isinstance(arg, A.StrLit):
            self.diags.append(
                f"{node.span.line}:{node.span.col}: panic in {self.fn_name!r}: "
                f"arg must be a string literal"
            )


def validate_panic_args(prog: A.Program) -> list[str]:
    """For every `panic(...)` call, enforce single-string-literal arg.
    Returns diagnostic strings.

    Stage 28.8.2: uses ``_PanicArgsValidator(ASTVisitor)``.

    Stage 28.9 cycle 60 audit-R C59-1: iter_fn_decls recursion.
    """
    diags: list[str] = []
    from .ast_walker import iter_fn_decls
    for fn in iter_fn_decls(prog):
        if fn.is_extern:
            continue
        _PanicArgsValidator(fn.name, diags).visit(fn.body)
    return diags


def find_unwind_attrs(prog: A.Program) -> list[A.FnDecl]:
    """Return all fns carrying @unwind. Phase-0 emits diagnostics for
    these (the attribute is reserved but not yet implemented).

    Stage 28.9 cycle 60 audit-R C59-1: iter_fn_decls recursion."""
    out: list[A.FnDecl] = []
    from .ast_walker import iter_fn_decls
    for fn in iter_fn_decls(prog):
        if "unwind" in fn.attrs:
            out.append(fn)
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
