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

import sys
from typing import IO, Literal

from .. import tir


# Stage 28.9 cycle 32 audit-T C30-5 fix (conf 65): explicit public
# API surface so `from helixc.ir.passes.effect_check import *` and
# external auditors have a stable reference. Private helpers
# (_is_meta_attr, _HARD_EFFECT_TRAP_IDS, _INFO_EFFECT_TRAP_IDS) are
# omitted as the underscore prefix signals.
__all__ = [
    "OP_EFFECTS",
    "PURITY_OBSERVER_EFFECTS",
    "META_ATTRS",
    "META_ATTR_PREFIXES",
    "EffectError",
    "EffectSeverity",
    "declared_effects",
    "is_pure_decl",
    "own_op_effects",
    "callees",
    "compute_closure",
    "check_module",
    "verify_module",
    "classify_effect_error",
    "report_diagnostics",
]


# Ops that produce side-effects in the runtime model. The set names match the
# string labels we attach to functions via @effect(...).
OP_EFFECTS: dict[tir.OpKind, frozenset[str]] = {
    tir.OpKind.PRINT: frozenset({"io"}),
    tir.OpKind.MODIFY: frozenset({"modify_self"}),
    tir.OpKind.SPLICE: frozenset({"modify_self"}),
    # Stage 28.9 cycle 21 audit-T C20-T5 fix (conf 92): the stale
    # "QUOTE is read-only; no effect" line that lived here was
    # contradicted by the cycle-22 C22-2 entry below mapping QUOTE to
    # frozenset({"reflect"}). The actual policy (see
    # PURITY_OBSERVER_EFFECTS) is: QUOTE carries the "reflect" label
    # but @pure is permitted to use it because reflection is an
    # observer of program structure, not a semantic effect.
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
    # explicit effect labels for ops that are gated-unreachable from
    # @pure today, so any future regression surfaces immediately.
    #
    # QUOTE / REFLECT_HASH reserve a runtime reflection-cell handle —
    # they're observers of program structure, not effects on program
    # semantics. (See PURITY_OBSERVER_EFFECTS below.)
    tir.OpKind.QUOTE: frozenset({"reflect"}),
    tir.OpKind.REFLECT_HASH: frozenset({"reflect"}),
    # TILE_INDEX_STORE writes HBM (the kernel-launch observable).
    tir.OpKind.TILE_INDEX_STORE: frozenset({"tile_io"}),
    # TRACE_ENTRY / TRACE_EXIT log runtime trace events to a separate
    # buffer; per trace_pass.py:110-112 documented policy, @trace on
    # @pure is allowed (trace is observability, not effect).
    tir.OpKind.TRACE_ENTRY: frozenset({"trace"}),
    tir.OpKind.TRACE_EXIT: frozenset({"trace"}),
}


# Stage 28.9 cycle 21 audit-T C20-T1 fix (conf 95): the language policy
# (trace_pass.py:110-112) and reflective-quoting semantics (quote returns
# an AST handle; the value is a reflection-cell pointer) both define
# "trace" and "reflect" as RUNTIME OBSERVERS, not semantic effects.
# A @pure function is permitted to contain TRACE_ENTRY/TRACE_EXIT and
# QUOTE/REFLECT_HASH ops — these observe program structure / execution
# without participating in the value semantics that purity reasons
# about. The labels remain in OP_EFFECTS so non-pure callees declaring
# @effect(trace) / @effect(reflect) still get the unused-effect check
# (19002); only the @pure violation check (19001) exempts them.
PURITY_OBSERVER_EFFECTS: frozenset[str] = frozenset({"trace", "reflect"})


# Attribute keys that are NOT effect labels. Anything else in fn.attrs is
# treated as a declared effect.
META_ATTRS = frozenset({
    "is_pub", "is_pure", "pure", "kernel",
    # Stage 16.5 FFI markers — set by lower_ast for the extern fn's
    # OWN FnIR, not effect labels.
    # Stage 28.9 cycle 24 audit-R C23-4 (conf 82): the "ffi" effect
    # added by cycle-22 C22-1 to OP_EFFECTS is attributed to CALLERS
    # (via FFI_CALL ops) — not to the extern declaration itself.
    # Keeping is_extern / extern_abi in META_ATTRS is correct and
    # intentional: an extern fn declaration does NOT contribute "ffi"
    # to its own declared_effects set. The 19002 unused-effect check
    # treats extern fn declarations cleanly (no false-positive).
    "is_extern", "extern_abi",
    # Stage 16 GPU launch marker; @kernel maps to attr "kernel" already
    # covered above; aliasing left here for forward-compat.
    "device",
    # @partial / @total are AST-level totality directives, not effects.
    "partial", "total",
    # Audit / debug markers introduced by post-AD passes (kept META so
    # they don't masquerade as declared effects).
    "checkpoint", "verifier",
    # Stage 28.9 cycle 21 audit-T C20-T2 fix (conf 95): the parser emits
    # these attribute keys (parser.py:275-292) for first-class language
    # features that are NOT effects. Without these, a fn declared with
    # @trace / @deprecated("x") / @autotune(K: [v]) / @since("vN") was
    # silently mis-flagged with trap 19002 (declared unused effect) by
    # IR effect_check, because the bare name fell through declared_effects'
    # "anything not in META_ATTRS is a declared effect" rule.
    # `trace` is special: it IS an effect label in OP_EFFECTS, but the
    # @trace attribute marks instrumentation, not effect declaration.
    # @effect(trace) (using effect:trace prefix) is the explicit declaration.
    "trace",
    "autotune",
    "deprecated",
    "since",
    # @grad markers from autodiff pipeline (forward/reverse mode tags).
    "grad",
    # Stage 28.9 cycle 35 audit-T C34-2 fix (conf 92): @inline is a
    # compiler hint (documented at ast_nodes.py:452), not an effect.
    # Pre-fix, `@inline fn f() -> i32 { 42 }` (without @pure) tripped
    # trap 19002 "declares unused effect(s) {inline}". Same defect
    # class as cycle 21 C20-T2 which added trace/autotune/deprecated/
    # since to META_ATTRS.
    "inline",
})


# Stage 28.9 cycle 21 audit-T C20-T2 fix (conf 95): attribute keys with
# value payloads (e.g. "deprecated:msg-text", "autotune:TILE=16,32",
# "since:v0.3") are emitted by the parser alongside the bare key. They
# must also be filtered out of declared_effects' bare-name fallback.
# An attribute key is META if its colon-prefix matches any of these.
META_ATTR_PREFIXES: tuple[str, ...] = (
    "autotune:",
    "deprecated:",
    "since:",
)


def _is_meta_attr(key: str) -> bool:
    """Stage 28.9 cycle 21 C20-T2: True if `key` is a non-effect
    attribute (either a bare META_ATTRS member or a value-carrying
    META_ATTR_PREFIXES form)."""
    if key in META_ATTRS:
        return True
    for prefix in META_ATTR_PREFIXES:
        if key.startswith(prefix):
            return True
    return False


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
        # Stage 28.9 cycle 21 C20-T2: META check uses _is_meta_attr to
        # cover both bare keys (e.g. "trace") and value-carrying forms
        # (e.g. "autotune:TILE=16,32", "deprecated:old-fn-msg").
        if _is_meta_attr(k):
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
            # Stage 28.9 cycle 21 audit-T C20-T1 fix (conf 95): exempt
            # PURITY_OBSERVER_EFFECTS (trace, reflect) from the @pure
            # violation check. Per documented language policy:
            #   - @trace @pure fn is allowed (trace_pass.py:110-112)
            #   - quote { ... } in @pure fn is allowed (reflection is
            #     observability, not effect)
            # Without this exemption, the cycle-22 hardening that added
            # TRACE_ENTRY/TRACE_EXIT and QUOTE/REFLECT_HASH to OP_EFFECTS
            # silently became a stricter-than-documented policy.
            effective_clos = clos - PURITY_OBSERVER_EFFECTS
            if effective_clos:
                errors.append(
                    f"@pure function {name!r} has actual effects "
                    f"{{{', '.join(sorted(effective_clos))}}} (must be empty) "
                    f"[trap {EffectError.trap_id_pure_violation}]"
                )
        else:
            decl = declared_effects(fn)
            # Stage 28.9 cycle 21 audit-T C20-T1 (conf 95): subtract
            # PURITY_OBSERVER_EFFECTS from the actual closure before
            # computing missing-declarations. Trace/reflect are
            # universally allowed (observers, not effects) — no
            # function ever needs to declare them, regardless of
            # @pure status. Without this, a non-pure fn body that
            # uses TRACE_ENTRY/TRACE_EXIT (via @trace) but doesn't
            # declare @effect(trace) trips 19001.
            extra = (clos - PURITY_OBSERVER_EFFECTS) - decl
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


# Stage 28.9 cycle 28 audit-R C27-2/C27-3 fix (conf 82+78): shared
# trap-id classifier so check.py and x86_64.py don't duplicate
# substring matching against the check_module() output format. The
# message format is owned by check_module() in this file — these sets
# stay in sync with the f-strings above.
#
# Stage 28.9 cycle 30 audit-T C28-TD3 (conf 70): underscore-prefixed
# to mark them as module-private. External callers should go through
# `classify_effect_error()`, never index these sets directly.
_HARD_EFFECT_TRAP_IDS: frozenset[int] = frozenset({EffectError.trap_id_pure_violation})
_INFO_EFFECT_TRAP_IDS: frozenset[int] = frozenset({EffectError.trap_id_unused_effect})

# Stage 28.9 cycle 30 audit-R C29-4 (conf 72) + cycle 32 audit-R
# C31-4 fix (conf 92): runtime disjoint check at module load. A
# future commit that accidentally adds an INFO id (e.g. 19002) to
# BOTH sets — a plausible copy-paste pattern — would silently
# classify it as "hard" because of HARD-first ordering in
# `classify_effect_error`. Pre-cycle-32 this used `assert` which
# Python strips under `-O`; now an explicit `raise RuntimeError`
# survives optimized mode.
_disjoint_overlap = _HARD_EFFECT_TRAP_IDS & _INFO_EFFECT_TRAP_IDS
if _disjoint_overlap:
    raise RuntimeError(
        f"effect-check trap-id sets must be disjoint; overlap="
        f"{_disjoint_overlap}"
    )


# Stage 28.9 cycle 30 audit-T C28-TD1 (conf 72): explicit Literal
# type for the severity return. A typo at the call site (e.g. `sev
# == "hrd"`) is now caught by static type checkers; bare-string
# return invited silent escalation through the `else` fail-closed
# branch.
# Stage 28.9 cycle 32 audit-T C30-5 (conf 65): renamed Severity →
# EffectSeverity (namespaced; the bare "Severity" name collides
# conceptually with typecheck/deprecation severity vocabularies).
EffectSeverity = Literal["hard", "info", "unknown"]


def classify_effect_error(message: str) -> EffectSeverity:
    """Classify a single message from check_module() into one of:
        "hard"    — trap 19001 @pure or under-declared violation; the
                    caller should treat it as a strict-mode hard error.
        "info"    — trap 19002 declared-unused effect; informational
                    only, never causes failure.
        "unknown" — message doesn't match a recognized trap-id. Caller
                    should fail-closed (treat as hard + emit a meta
                    warning about the unknown id).

    The substring discriminator uses the full bracketed `[trap NNNNN]`
    token, anchored by the brackets, so future trap-ids of the form
    `19001N` cannot accidentally match `19001` as a prefix substring.
    """
    for tid in _HARD_EFFECT_TRAP_IDS:
        if f"[trap {tid}]" in message:
            return "hard"
    for tid in _INFO_EFFECT_TRAP_IDS:
        if f"[trap {tid}]" in message:
            return "info"
    return "unknown"


def report_diagnostics(
    eff_errs: list[str] | None,
    *,
    stderr: IO[str] | None = None,
) -> int:
    """Stage 28.9 cycle 30 audit-R C29-5 (conf 68): extract the
    duplicated dispatch shell from check.py and x86_64.py into a
    shared helper. Returns the hard-error count so the caller can
    decide --strict abort semantics (rc=1 vs sys.exit).

    Stage 28.9 cycle 32 audit-R C31-R2 / C31-1 / C30-1 fix (conf
    88+90+92): `stderr` has a real default (None → sys.stderr) so
    the None-sentinel body branch is actually reachable. Pre-fix
    the keyword-only param had no default — every call site had to
    pass `stderr=` explicitly and the `if stderr is None` guard was
    dead code.

    Stage 28.9 cycle 32 audit-R C31-2 fix (conf 90): `eff_errs` is
    `list[str] | None` with a defensive None→0 early return,
    mirroring the cycle-29 C29-1 None-rejection discipline in
    `const_fold.FoldError`. A caller that forgets to capture
    `check_module`'s return value (or refactors it to Optional)
    now gets a clean 0 instead of an opaque TypeError-on-iteration.

    Stage 28.9 cycle 30 audit-R C29-6 (conf 62) + cycle 32 audit-T
    C30-7 fix (conf 55): info diagnostics use `info:` prefix
    instead of gcc/clang's `note:`. Standalone `note:` is unusual
    outside an attached-context pattern; `info:` is unambiguous and
    grep-uniform with `warning:`.

    Stage 28.9 cycle 32 audit-R C31-5 fix (conf 72): unknown-trap-id
    line uses `warning: effect-check: <msg>` prefix matching the
    hard-severity format. The cycle-30 `helixc: warning: ...`
    prefix was the only inconsistent shape across the three branches.
    """
    if stderr is None:
        stderr = sys.stderr
    if eff_errs is None:
        # Treat None as "no diagnostics produced".
        return 0
    hard_count = 0
    for e in eff_errs:
        sev = classify_effect_error(e)
        if sev == "hard":
            print(f"warning: effect-check: {e}", file=stderr)
            hard_count += 1
        elif sev == "info":
            print(f"info: effect-check: {e}", file=stderr)
        else:
            # Unknown trap-id — fail-closed. Format matches the
            # hard branch (grep-uniform) but includes a descriptive
            # "unknown trap-id" sub-message so the maintainer
            # notices the new id.
            print(
                f"warning: effect-check: unknown trap-id; "
                f"classifying as hard: {e}", file=stderr,
            )
            hard_count += 1
    return hard_count
