"""
helixc/frontend/deprecated_pass.py — Stage 28.7: @deprecated + @since.

`@deprecated("msg")` on a fn / struct / enum decl emits a compile-time
warning at every call site of that symbol. `@since("v0.3")` is a
documentation marker (no compile-time effect; surface for --doc).

CLI integration:
  * `-Wdeprecated`        => warn (default)
  * `-Wdeprecated=error`  => promote warnings to errors (handled in
                              helixc/check.py via prog._deprecation_warnings)

Trap-id reservations: none — warnings only.

Phase-0:
  * Parser stores attrs as ["deprecated", "deprecated:msg"]
    (and similarly for @since).
  * deprecation_msg(fn) -> Optional[str] reads the message back.
  * find_deprecated_decls(prog) -> {name: msg}.
  * find_deprecation_call_sites(prog) -> [(callee_name, span, msg)].
  * emit_warnings(prog) attaches a list to prog._deprecation_warnings.

License: Apache 2.0
"""

from __future__ import annotations

from typing import Optional

from . import ast_nodes as A


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


def _walk_call_sites(node, callback) -> None:
    """Recursive walk yielding every Call node found in node."""
    if node is None:
        return
    if isinstance(node, A.Call):
        callback(node)
    # Iterate over containers + sub-attrs (same as panic_pass walker)
    if isinstance(node, A.Block):
        for s in node.stmts:
            _walk_call_sites(s, callback)
        if node.final_expr is not None:
            _walk_call_sites(node.final_expr, callback)
        return
    for attr in ("expr", "left", "right", "operand", "cond", "then",
                 "else_", "value", "scrutinee", "callee", "init",
                 "rhs", "body", "then_branch", "else_branch"):
        sub = getattr(node, attr, None)
        if sub is not None and hasattr(sub, "span"):
            _walk_call_sites(sub, callback)
    for attr in ("args", "stmts", "fields", "elems", "arms"):
        seq = getattr(node, attr, None)
        if seq is None:
            continue
        try:
            for it in seq:
                if isinstance(it, tuple):
                    for sub in it:
                        if hasattr(sub, "span"):
                            _walk_call_sites(sub, callback)
                elif hasattr(it, "span"):
                    _walk_call_sites(it, callback)
        except TypeError:
            pass


def find_deprecation_call_sites(prog: A.Program) -> list[tuple[str, A.Span, str]]:
    """For every call to a deprecated fn, return (callee_name, span,
    deprecation_msg)."""
    deps = find_deprecated_decls(prog)
    if not deps:
        return []
    out: list[tuple[str, A.Span, str]] = []
    for it in prog.items:
        if not isinstance(it, A.FnDecl) or it.is_extern:
            continue

        def cb(call, deps=deps, out=out):
            callee = call.callee
            if isinstance(callee, A.Name) and callee.name in deps:
                out.append((callee.name, call.span, deps[callee.name]))
        _walk_call_sites(it.body, cb)
    return out


def emit_warnings(prog: A.Program) -> list[str]:
    """Run the @deprecated pass; attach a list of warning strings to
    `prog._deprecation_warnings` (also returns it). The CLI / driver
    reads this attribute to decide whether to log warnings, escalate
    to errors via `-Wdeprecated=error`, etc."""
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
    # Stash on prog so helixc/check.py can pick up.
    prog._deprecation_warnings = out
    return out
