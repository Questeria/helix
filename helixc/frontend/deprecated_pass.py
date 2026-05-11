"""
helixc/frontend/deprecated_pass.py — Stage 28.7: @deprecated + @since.

`@deprecated("msg")` on a fn / struct / enum decl emits a compile-time
warning at every call site of that symbol. `@since("v0.3")` is a
documentation marker (no compile-time effect; surface for --doc).

CLI integration:
  * `-Wdeprecated`        => warn (default)
  * `-Wdeprecated=error`  => promote warnings to errors (handled in
                              helixc/check.py — the caller stores the
                              returned list locally)

Trap-id reservations: none — warnings only.

Phase-0:
  * Parser stores attrs as ["deprecated", "deprecated:msg"]
    (and similarly for @since).
  * deprecation_msg(fn) -> Optional[str] reads the message back.
  * find_deprecated_decls(prog) -> {name: msg}.
  * find_deprecation_call_sites(prog) -> [(callee_name, span, msg)].
  * emit_warnings(prog) returns a list of warning strings. The CLI
    driver (`helixc/check.py`) stores the result locally; we no longer
    monkey-patch an undeclared `_deprecation_warnings` attribute onto
    A.Program (Audit 28.8 C1-M1 — coupling AST data model to pass
    output meant a second `emit_warnings` call silently overwrote
    the first).

License: Apache 2.0
"""

from __future__ import annotations

from typing import Optional

from . import ast_nodes as A
from .ast_walker import ASTVisitor


# Stage 28.8.2: deprecated_pass walker migrated to ASTVisitor base class.
# Pre-fix this module hand-rolled `_walk_call_sites` with the same
# attribute-list drift problem as panic_pass / unsafe_pass (audit A5
# HIGH fix-sweep had to manually synchronize the list — `obj`, `target`,
# `iter_expr`, `start`, `end`, `guard`, `inner` etc.). The shared
# library traverses dataclass fields by introspection.


def _attr_msg(attrs: list[str], name: str) -> Optional[str]:
    """Extract the message arg from attrs for a string-arg attribute.

    Looks for '<name>:<msg>' entries; returns None if the attribute
    is present but has no message, and None if absent."""
    prefix = f"{name}:"
    for a in attrs:
        if a.startswith(prefix):
            return a[len(prefix):]
    if name in attrs:
        return ""  # present without msg
    return None


def deprecation_msg(decl) -> Optional[str]:
    """Return the message string for @deprecated, or None if not
    deprecated."""
    attrs = getattr(decl, "attrs", None)
    if attrs is None:
        return None
    return _attr_msg(attrs, "deprecated")


def since_marker(decl) -> Optional[str]:
    """Return the @since version string, or None if absent."""
    attrs = getattr(decl, "attrs", None)
    if attrs is None:
        return None
    return _attr_msg(attrs, "since")


def find_deprecated_decls(prog: A.Program) -> dict[str, str]:
    """Return {name: msg} for every deprecated top-level fn/struct/enum.
    Empty string msg = bare @deprecated."""
    out: dict[str, str] = {}
    for it in prog.items:
        name = getattr(it, "name", None)
        if name is None:
            continue
        msg = deprecation_msg(it)
        if msg is not None:
            out[name] = msg or ""
    return out


class _DeprecationCallSiteCollector(ASTVisitor):
    """Stage 28.8.2: record every Call to a deprecated symbol as a
    (callee_name, span, msg) tuple. Drop-in replacement for the
    pre-fix `_walk_call_sites` + closure callback pattern.

    The `deps` dict maps name -> msg; visit_Call checks
    `isinstance(node.callee, A.Name)` and looks up the name in deps.
    """

    def __init__(self, deps: dict[str, str],
                 out: list[tuple[str, A.Span, str]]):
        self.deps = deps
        self.out = out

    def visit_Call(self, node: A.Call) -> None:
        callee = node.callee
        if isinstance(callee, A.Name) and callee.name in self.deps:
            self.out.append((callee.name, node.span, self.deps[callee.name]))


def find_deprecation_call_sites(prog: A.Program) -> list[tuple[str, A.Span, str]]:
    """For every call to a deprecated fn, return (callee_name, span,
    deprecation_msg).

    Stage 28.8.2: uses ``_DeprecationCallSiteCollector(ASTVisitor)``.
    """
    deps = find_deprecated_decls(prog)
    if not deps:
        return []
    out: list[tuple[str, A.Span, str]] = []
    for it in prog.items:
        if not isinstance(it, A.FnDecl) or it.is_extern:
            continue
        _DeprecationCallSiteCollector(deps, out).visit(it.body)
    return out


def emit_warnings(prog: A.Program) -> list[str]:
    """Run the @deprecated pass; return a list of warning strings.

    The CLI / driver decides whether to log warnings, escalate to
    errors via `-Wdeprecated=error`, etc.

    Audit 28.8 C1-M1: prior versions monkey-patched
    `prog._deprecation_warnings = out` — an undeclared attribute on the
    Program dataclass that caused a second `emit_warnings(prog)` call
    to silently overwrite the first. The pass now ONLY returns the
    list; callers must store it locally. This decouples the AST data
    model from analysis-pass output, and makes multiple invocations
    on the same program well-defined.
    """
    sites = find_deprecation_call_sites(prog)
    out: list[str] = []
    for name, span, msg in sites:
        if msg:
            out.append(
                f"{span.line}:{span.col}: call to deprecated {name!r}: {msg}"
            )
        else:
            out.append(
                f"{span.line}:{span.col}: call to deprecated {name!r}"
            )
    return out
