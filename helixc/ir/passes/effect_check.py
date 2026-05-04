"""
helixc/ir/passes/effect_check.py — IR-level effect / capability verification.

The frontend typecheck (typecheck.py:_check_call_effects) checks effects at AST
level for direct named callees, but it can be bypassed by:
  - indirect calls (function-pointer values),
  - calls to unresolved names,
  - special ops like MODIFY/SPLICE/PRINT that aren't surface-level "calls".

This IR pass runs AFTER lowering and is the authoritative effect checker. It
walks the call graph in tir.Module form, computes each function's effect
closure (own ops' effects ∪ transitive callees' effects), and rejects any
function whose closure exceeds its declared effects.

Effect labels (string set):
  - "io"           — observable I/O (PRINT, io.* calls)
  - "modify_self"  — code-modifying / reflective writes (MODIFY, SPLICE)
  - "alloc"        — heap-ish allocation (currently we have only stack arrays;
                     reserved for future)
  - "unknown"      — indirect call to an unresolved function: must be assumed
                     to have any effect

Function attrs (set in lower_ast from frontend):
  - is_pure: True/absent. A @pure function's closure must be empty.
  - is_pub: marks exported entry; not relevant for effect checking.
  - declared effects: stored in attrs as keys whose value is True; a function's
    declared effects are everything in attrs except meta-keys (is_pure, is_pub,
    "pure", "kernel", and the AD-pass markers).

License: Apache 2.0
"""

from __future__ import annotations

from .. import tir


# Ops that produce side-effects in the runtime model. The set names match the
# string labels we attach to functions via @effect(...).
OP_EFFECTS: dict[tir.OpKind, frozenset[str]] = {
    tir.OpKind.PRINT: frozenset({"io"}),
    tir.OpKind.MODIFY: frozenset({"modify_self"}),
    tir.OpKind.SPLICE: frozenset({"modify_self"}),
    # QUOTE is read-only (returns an AST handle); no effect.
}


# Attribute keys that are NOT effect labels. Anything else in fn.attrs is
# treated as a declared effect.
META_ATTRS = frozenset({"is_pub", "is_pure", "pure", "kernel"})


class EffectError(Exception):
    """Raised when a function's IR effects exceed its declared effects."""
    pass


def declared_effects(fn: tir.FnIR) -> frozenset[str]:
    """Effects this function explicitly declares it may have. @pure means {}."""
    if fn.attrs.get("is_pure") or fn.attrs.get("pure"):
        return frozenset()
    return frozenset(k for k in fn.attrs.keys() if k not in META_ATTRS
                     and fn.attrs[k] is True)


def is_pure_decl(fn: tir.FnIR) -> bool:
    return bool(fn.attrs.get("is_pure") or fn.attrs.get("pure"))


def own_op_effects(fn: tir.FnIR) -> frozenset[str]:
    """Union of effects implied by the ops in this function's body."""
    eff: set[str] = set()
    for blk in fn.blocks:
        for op in blk.ops:
            if op.kind in OP_EFFECTS:
                eff |= OP_EFFECTS[op.kind]
    return frozenset(eff)


def callees(fn: tir.FnIR) -> set[str]:
    out: set[str] = set()
    for blk in fn.blocks:
        for op in blk.ops:
            if op.kind == tir.OpKind.CALL:
                target = op.attrs.get("target")
                if isinstance(target, str):
                    out.add(target)
                else:
                    out.add("<indirect>")
            elif op.kind == tir.OpKind.MODIFY:
                # Verifier functions called via MODIFY's verifier_fn attr
                # contribute to this function's effect closure.
                vfn = op.attrs.get("verifier_fn")
                if isinstance(vfn, str):
                    out.add(vfn)
    return out


def compute_closure(module: tir.Module) -> dict[str, frozenset[str]]:
    """Effect closure per function = own_op_effects ∪ ⋃ callees' closures.

    For unresolved callees ("<indirect>" or names not in the module), we
    contribute "unknown" — so the caller must declare it (or the check fails).
    Computed by fixpoint iteration on the call graph.
    """
    closure: dict[str, set[str]] = {n: set(own_op_effects(fn))
                                     for n, fn in module.functions.items()}
    callee_map: dict[str, set[str]] = {n: callees(fn)
                                        for n, fn in module.functions.items()}
    changed = True
    while changed:
        changed = False
        for n in list(module.functions.keys()):
            before = len(closure[n])
            for c in callee_map[n]:
                if c == "<indirect>":
                    closure[n].add("unknown")
                elif c in module.functions:
                    closure[n] |= closure[c]
                else:
                    # Unknown external — assume worst case
                    closure[n].add("unknown")
            if len(closure[n]) > before:
                changed = True
    return {n: frozenset(s) for n, s in closure.items()}


def check_module(module: tir.Module) -> list[str]:
    """Return list of human-readable error messages. Empty list = pass."""
    closure = compute_closure(module)
    errors: list[str] = []
    for name, fn in module.functions.items():
        clos = closure[name]
        if is_pure_decl(fn):
            if clos:
                errors.append(
                    f"@pure function {name!r} has actual effects "
                    f"{{{', '.join(sorted(clos))}}} (must be empty)"
                )
        else:
            decl = declared_effects(fn)
            extra = clos - decl
            if extra:
                errors.append(
                    f"function {name!r} declares effects "
                    f"{{{', '.join(sorted(decl)) or '(none)'}}} but actually "
                    f"has {{{', '.join(sorted(clos))}}}; missing: "
                    f"{{{', '.join(sorted(extra))}}}"
                )
    return errors


def verify_module(module: tir.Module) -> None:
    """Raise EffectError on the first violation."""
    errs = check_module(module)
    if errs:
        raise EffectError("\n".join(errs))
