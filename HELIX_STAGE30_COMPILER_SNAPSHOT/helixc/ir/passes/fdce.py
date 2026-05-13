"""
helixc/ir/passes/fdce.py — module-level (function-level) dead-code elimination.

Where dce.py removes unused ops within a function, this pass removes entire
unused functions from the module. Useful after grad_pass or other rewrites
that may strand the original `loss` function once its only callers were
rewritten to `loss__grad`.

Algorithm:
  1. Mark `entry_fn` (default "main") and any function whose name starts
     with `pub_` (heuristic for exported pubs) as roots.
  2. Compute the call graph: function f calls g if f contains a CALL op
     whose target attr is g.
  3. Mark transitively-reachable functions live; drop the rest.

Skips removal if `entry_fn` is missing — we don't want to silently empty
the module.

License: Apache 2.0
"""

from __future__ import annotations

from .. import tir


def fdce_module(module: tir.Module, entry_fn: str = "main") -> int:
    """Remove unreachable functions from `module`. Returns the count of
    functions dropped."""
    if entry_fn not in module.functions:
        return 0

    # Build the call graph. Functions are "called" via:
    #   - direct CALL op (target attr)
    #   - MODIFY op's verifier_fn attr (verifier-gated reflection)
    #   - QUOTE op's ast_pretty: a pretty-printed AST may name any fn
    #     that appears free in the quoted expression — splicing the
    #     quote at runtime indirectly invokes those fns. Conservative:
    #     scan ast_pretty for identifiers that match module fn names.
    import re
    _ID_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
    all_fn_names = set(module.functions.keys())
    callees: dict[str, set[str]] = {}
    for name, fn in module.functions.items():
        called = set()
        for blk in fn.blocks:
            for op in blk.ops:
                if op.kind == tir.OpKind.CALL:
                    target = op.attrs.get("target")
                    if isinstance(target, str):
                        called.add(target)
                elif op.kind == tir.OpKind.MODIFY:
                    vfn = op.attrs.get("verifier_fn")
                    if isinstance(vfn, str):
                        called.add(vfn)
                elif op.kind == tir.OpKind.QUOTE:
                    pretty = op.attrs.get("ast_pretty", "")
                    if isinstance(pretty, str) and pretty:
                        for ident in _ID_RE.findall(pretty):
                            if ident in all_fn_names:
                                called.add(ident)
        callees[name] = called

    # Roots: entry_fn + any pub-prefixed function (cheap interop hook)
    # + Stage 16 @kernel fns (called from GPU launch, not from host code;
    #   keeping them visible to the backend so PTX text gets emitted).
    live: set[str] = set()
    worklist: list[str] = [entry_fn]
    for name, fn in module.functions.items():
        if fn.attrs.get("is_pub"):
            worklist.append(name)
        elif fn.attrs.get("kernel"):
            worklist.append(name)

    while worklist:
        n = worklist.pop()
        if n in live:
            continue
        if n not in module.functions:
            continue
        live.add(n)
        for c in callees.get(n, ()):
            if c not in live:
                worklist.append(c)

    # Drop any function not in `live`
    dead = [n for n in module.functions if n not in live]
    for n in dead:
        del module.functions[n]
    return len(dead)
