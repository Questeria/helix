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
    # Stage 28.5 — TRAP terminates the process (sys_exit). Treated as an
    # io effect because the panic message is written to stderr before
    # exit. @pure fns cannot panic.
    tir.OpKind.TRAP: frozenset({"io"}),
    # Audit 28.8 cycle 22 C22-1 (HIGH): FFI_CALL (extern "C" calls)
    # is a side effect — calling puts/free/sock_*/etc. is observably
    # impure. The DCE pass added FFI_CALL to its SIDE_EFFECT_KINDS in
    # Stage 16.5 but the parallel effect_check pass never propagated
    # the same hardening, so `@pure fn p() { extern_puts(...) }`
    # silently passed both AST and IR effect checks.
    tir.OpKind.FFI_CALL: frozenset({"ffi"}),
    # Audit 28.8 cycle 22 C22-3 (HIGH): ARENA_PUSH / ARENA_SET mutate
    # a global region — same pattern as MODIFY/SPLICE. Reachable from
    # @pure via the __arena_push / __arena_set builtins.
    tir.OpKind.ARENA_PUSH: frozenset({"arena"}),
    tir.OpKind.ARENA_SET: frozenset({"arena"}),
    # Audit 28.8 cycle 22 C22-2/C22-4/C22-5 (LOW, defense in depth):
    # gated-unreachable from @pure today but a stale gate could open
    # them. Add the explicit effect labels so any future regression
    # surfaces immediately.
    #
    # QUOTE / REFLECT_HASH reserve a runtime reflection-cell handle.
    tir.OpKind.QUOTE: frozenset({"reflect"}),
    tir.OpKind.REFLECT_HASH: frozenset({"reflect"}),
    # TILE_INDEX_STORE writes HBM (the kernel-launch observable).
    tir.OpKind.TILE_INDEX_STORE: frozenset({"tile_io"}),
    # TRACE_ENTRY / TRACE_EXIT log runtime trace events.
    tir.OpKind.TRACE_ENTRY: frozenset({"trace"}),
    tir.OpKind.TRACE_EXIT: frozenset({"trace"}),
}


# Attribute keys that are NOT effect labels. Anything else in fn.attrs is
# treated as a declared effect.
META_ATTRS = frozenset({
    "is_pub", "is_pure", "pure", "kernel",
    # Stage 16.5 FFI markers — set by lower_ast for extern fns, not
    # effect labels.
    "is_extern", "extern_abi",
    # Stage 16 GPU launch marker; @kernel maps to attr "kernel" already
    # covered above; aliasing left here for forward-compat.
    "device",
    # @partial / @total are AST-level totality directives, not effects.
    "partial", "total",
    # Audit / debug markers introduced by post-AD passes (kept META so
    # they don't masquerade as declared effects).
    "checkpoint", "verifier",
})


class EffectError(Exception):
    """Raised when a function's IR effects exceed its declared effects."""
    # Stage 19 trap-ids surfaced by the IR-level effect check:
    #   19001 — @pure (or under-declared) function actually has effects
    #   19002 — declared effect was never exercised by the body's IR
    trap_id_pure_violation = 19001
    trap_id_unused_effect = 19002


def declared_effects(fn: tir.FnIR) -> frozenset[str]:
    """Effects this function explicitly declares it may have. @pure means {}.

    The parser stores @effect(io) as the attribute key "effect:io" (kept
    namespaced so the bare token "io" cannot accidentally collide with a
    future "io" function attribute). Here we strip that prefix so the
    declared set uses the same labels as OP_EFFECTS / the per-op effect
    labels (e.g. PRINT contributes "io", not "effect:io"). Without the
    strip, a function declaring @effect(io) and actually calling PRINT
    would falsely report missing: {io} AND unused: {effect:io}, which is
    exactly the Stage 19 regression test_stage19_trap_19002_does_not_fire
    used to catch.
    """
    if fn.attrs.get("is_pure") or fn.attrs.get("pure"):
        return frozenset()
    declared: set[str] = set()
    for k, v in fn.attrs.items():
        if v is not True:
            continue
        if k in META_ATTRS:
            continue
        if k.startswith("effect:"):
            declared.add(k[len("effect:"):])
        else:
            # A bare attribute name (e.g. "io" directly) — kept for
            # backward compat with hand-built tir modules in tests.
            declared.add(k)
    return frozenset(declared)


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
            elif op.kind == tir.OpKind.FFI_CALL:
                # Audit 28.8 cycle 22 C22-1 (HIGH): FFI_CALL must also
                # populate the callee set so the call-graph closure
                # propagates "ffi" effect to transitive callers.
                target = op.attrs.get("target")
                if isinstance(target, str):
                    out.add(target)
                else:
                    out.add("<indirect-ffi>")
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
                elif c == "<indirect-ffi>":
                    # Stage 28.9 cycle 16 audit-C C1 fix (conf 85): the
                    # `<indirect-ffi>` sentinel emitted by `callees()` for
                    # FFI_CALL ops with non-string target was previously
                    # falling through to the generic `unknown` branch,
                    # losing the more specific "ffi" effect label. A fn
                    # declaring @effect(ffi) with an indirect FFI call
                    # would see "unknown" (not "ffi") in its closure, so
                    # `declared_effects` match check would falsely fire
                    # 19002 ("declared unused effect"). Add "ffi" too so
                    # the caller's declaration is honored; also add
                    # "unknown" so any other effects of the indirect
                    # target are conservatively assumed.
                    closure[n].add("ffi")
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
    """Return list of human-readable error messages. Empty list = pass.

    Reports the two Stage-19 trap classes:
      19001 — @pure / under-declared function actually has effects.
      19002 — declared effect was never exercised by the body's closure.

    19001 is a hard violation: the program lies about its own purity.
    19002 is a soft warning: dead annotations. We emit it through the
    same list so the backend can decide whether to fail-strict on it
    (currently it does not — declared-unused effects are a code smell,
    not a correctness violation).

    "unknown" callees are never required to be re-declared (a function
    may reasonably declare nothing and still call an externally-defined
    intrinsic), so 19002 only fires on named effect labels.
    """
    closure = compute_closure(module)
    errors: list[str] = []
    for name, fn in module.functions.items():
        clos = closure[name]
        if is_pure_decl(fn):
            if clos:
                errors.append(
                    f"@pure function {name!r} has actual effects "
                    f"{{{', '.join(sorted(clos))}}} (must be empty) "
                    f"[trap {EffectError.trap_id_pure_violation}]"
                )
        else:
            decl = declared_effects(fn)
            extra = clos - decl
            if extra:
                errors.append(
                    f"function {name!r} declares effects "
                    f"{{{', '.join(sorted(decl)) or '(none)'}}} but actually "
                    f"has {{{', '.join(sorted(clos))}}}; missing: "
                    f"{{{', '.join(sorted(extra))}}} "
                    f"[trap {EffectError.trap_id_pure_violation}]"
                )
            # Stage 19 trap 19002: declared effect that the closure does
            # not actually need. Skip "unknown" (it always names the
            # opaque-callee bucket, not a real label). The 19002 check is
            # informational; check_module returns it in the same list.
            unused = decl - clos - frozenset({"unknown"})
            if unused:
                errors.append(
                    f"function {name!r} declares unused effect(s) "
                    f"{{{', '.join(sorted(unused))}}} "
                    f"[trap {EffectError.trap_id_unused_effect}]"
                )
    return errors


def verify_module(module: tir.Module) -> None:
    """Raise EffectError on the first violation."""
    errs = check_module(module)
    if errs:
        raise EffectError("\n".join(errs))
