"""
helixc/frontend/autodiff.py — source-level forward-mode automatic differentiation.

When the user writes `grad(loss)`, the compiler walks loss's AST body and
generates a derivative AST. This module provides `differentiate(expr, var)`
that returns the symbolic derivative of `expr` with respect to `var`.

Supported expressions:
- IntLit / FloatLit  (derivative is 0)
- Name == var        (derivative is 1)
- Name != var        (derivative is 0)
- Binary +, -        (linearity)
- Binary *           (product rule)
- Binary /           (quotient rule)
- Unary -            (negation)
- Calls              (NOT YET — would need chain rule + known derivatives
                      of builtin functions)
- Block / If         (NOT YET — needs control-flow handling)

This is forward-mode AD. For ML loss functions you'd typically want
reverse-mode; that's a future enhancement.

License: Apache 2.0
"""

from __future__ import annotations

import copy as _copy
from typing import Optional

from . import ast_nodes as A
from .ast_hash import structural_hash


# Module-level memoization for `differentiate()`. Keyed on:
#   (structural_hash(expr), var, fn_table_signature)
# A returned value is a deepcopy so callers can mutate freely without
# corrupting the cache. Cleared by `clear_diff_cache()` if needed.
_DIFF_CACHE: dict[tuple[str, str, str], A.Expr] = {}
_DIFF_CACHE_HITS = [0]
_DIFF_CACHE_MISSES = [0]


# Builtins with known pure behavior in the AD surface. Some have analytic
# chain rules; opaque min/max/clamp/sign variants remain non-inlined so missing
# differentiability cannot be hidden behind conditional bodies.
AD_KNOWN_PURE_CALLS = {
    "__exp", "__log", "__sin", "__cos", "__sqrt",
    "__relu", "__sigmoid", "__tanh", "__softplus",
    "__silu", "__abs", "__gelu", "__powi", "__bce",
    "__log_stable",
    "__exp_f64", "__log_f64", "__sin_f64", "__cos_f64",
    "__sqrt_f64", "__relu_f64", "__sigmoid_f64", "__abs_f64",
    "__min", "__max", "__clamp",
    "__min_i32", "__max_i32", "__clamp_i32",
    "__min_f64", "__max_f64", "__clamp_f64",
    "__sign", "__sign_f64",
    # Stage 36 Increment 6: provenance + fuzzy logic primitives are
    # AD-pure. prove/unwrap_logic/attach/detach lower as identity at
    # IR. fuzzy_and/fuzzy_or/fuzzy_not lower to MUL/ADD/SUB chains
    # which AD already knows how to differentiate. Registering them
    # here lets `grad(loss)` and `grad_rev(loss)` flow gradients
    # through Logic-typed sub-expressions.
    "prove", "unwrap_logic",
    "attach", "detach",
    "fuzzy_and", "fuzzy_or", "fuzzy_not",
    # Stage 36 Increment 8: fuzzy XOR + implication.
    "fuzzy_xor", "fuzzy_implies",
    # Stage 37 Inc 4 — tiered memory identity-lowerings registered as
    # AD-pure (closes Stage 37 closure gate-1 LOW finding). At Phase-0
    # the TyMemTier wrappers and cross-tier transitions are pure
    # identity at IR (see lower_ast.py:1977-1990) — they don't mutate
    # state, don't call effectful runtime hooks, just pass the inner
    # value through. The let-inlining AD-erasability check needs them
    # here to avoid the misleading "cannot erase side-effecting let"
    # diagnostic when an unused `let _ = into_working(x);` appears
    # inside a grad/grad_rev body. When/if Phase-1 introduces real
    # tier-id arena mutation (planned Stage 37 Inc 5), removal will
    # mirror the Inc 11 H1 fix that removed register_derivation from
    # this set once it gained ARENA_PUSH_PAIR side effects.
    "into_working", "into_episodic", "into_semantic", "into_procedural",
    "unwrap_working", "unwrap_episodic", "unwrap_semantic", "unwrap_procedural",
    "consolidate", "recall",
    # Stage 38 Inc 1 + Inc 2 — spatial-frame identity-lowerings.
    # AD_KNOWN_PURE_CALLS governs let-erasability (see
    # _is_ad_erasable_expr). Stage 38 post-Inc-3 silent-failure F2 fix
    # additionally installs identity chain rules for the same 12 names
    # in both forward (_diff_call_chain_rule) and reverse (_propagate)
    # via _IDENTITY_AD_CHAIN_RULE_NAMES — so `grad(use_frame)` now flows
    # gradients through the wrapper rather than raising the opaque-call
    # catchall.
    "into_world", "into_robot", "into_camera",
    "from_world", "from_robot", "from_camera",
    "world_to_robot", "robot_to_world",
    "robot_to_camera", "camera_to_robot",
    "world_to_camera", "camera_to_world",
    # Stage 39 Inc 1 + Inc 2 — temporal identity-lowerings (4 kinds
    # × intro/elim = 8, plus 4 cross-kind transitions = 12). All
    # Phase-0 pure-identity at IR (see lower_ast.py). Same let-
    # erasability rationale as Stage 37/38: prevent the misleading
    # "cannot erase side-effecting let" trap on an unused
    # `let _ = into_past(x);` inside grad/grad_rev. When/if Phase-1
    # introduces real per-tick temporal-id arena mutation, the
    # removal pattern mirrors the Inc 11 H1 register_derivation fix.
    "into_past", "into_present", "into_future", "into_eternal",
    "from_past", "from_present", "from_future", "from_eternal",
    "to_past", "forecast", "recall_past", "actualize",
    # Stage 40 Inc 1 + Inc 2 — modal/epistemic identity-lowerings
    # (4 kinds × intro/elim = 8 + 2 transitions = 10). Phase-0
    # pure-identity at IR. Modal kinds compose with temporal kinds
    # at the type level: `Known<Past<f32>>` differentiates same as
    # raw `f32` through chained identity wrappers.
    "into_known", "into_believed", "into_goal", "into_uncertain",
    "from_known", "from_believed", "from_goal", "from_uncertain",
    "confirm", "act_on",
    # Stage 41 Inc 1 + Inc 2 — causal/intent identity-lowerings
    # (4 kinds × intro/elim = 8 + 3 transitions = 11). Phase-0
    # pure-identity at IR. Composes with the 4-stack quartet —
    # `Known<Cause<f32>>` differentiates same as raw f32.
    "into_cause", "into_effect", "into_joint", "into_independent",
    "from_cause", "from_effect", "from_joint", "from_independent",
    "propagate", "aggregate", "isolate",
    # Stage 36 Increment 9 post-Inc-8 audit C2 LOW fix: register the
    # boolean-algebra builtins as AD-pure. They're all integer-valued
    # (so the AD derivative is 0 for differentiable use cases), but
    # the let-inlining AD-erasability check (_is_ad_erasable_expr)
    # needs them in this set to avoid the "cannot erase side-effecting
    # let" trap when a function transitively calls them in a
    # grad/grad_rev path.
    "and_logic", "or_logic", "not_logic",
    "xor_logic", "implies_logic", "eq_logic", "if_logic",
    "to_logic_bool",
    # parent_*_at are pure arena READS — no mutation, safe to erase
    # if the result is unused inside a differentiated function.
    "parent_left_at", "parent_right_at",
    # Stage 36 Inc 14: parent_at is the generic indexed accessor
    # (parent_at(handle, slot)). Same pure-read semantics as
    # parent_left_at/parent_right_at.
    "parent_at",
    # NOTE: `derive` and `register_derivation` were briefly listed
    # here by the Inc 9 C2 LOW fix, but the Inc 9 B2 fix (commit
    # 707deff) made both functions perform an ARENA_PUSH_PAIR side
    # effect. Stage 36 Inc 11 post-Inc-10 silent-failure H1 + type-
    # design B3 fix: removed from the pure set so that an unused
    # `let _h = register_derivation(p, q);` inside grad/grad_rev no
    # longer gets silently erased by `_inline_lets`. Calling them in
    # a differentiated function now correctly raises
    # NotImplementedError("AD cannot erase side-effecting ...") —
    # the user must hoist the call outside the differentiator.
}


# Stage 36 Inc 12 — close Inc 11 type-design B2 MEDIUM deferral.
#
# These builtins are integer-valued boolean Logic ops. They are
# kept in AD_KNOWN_PURE_CALLS so let-inlining doesn't trap on an
# unused `let _ = and_logic(...)` inside a grad/grad_rev body,
# but a *differentiated* call site has no meaningful chain rule
# (the derivative of integer truth-table arithmetic is 0 almost
# everywhere and undefined at the step). Pre-Inc-12 such a call
# silently produced a zero derivative; user code intending
# `fuzzy_and` (which IS differentiable) got a vacuous gradient
# with no diagnostic.
#
# Post-Inc-12 both forward `_diff_call_chain_rule` and reverse
# `_propagate` consult this set BEFORE the chain-rule arms and
# raise NotImplementedError with a message pointing at the fuzzy
# alternative — same fail-closed discipline as the Inc 9 B3 fix
# for `prove(x, x)` and the Stage 35 opaque-call NotImplementedError.
AD_INTEGER_VALUED_LOGIC = frozenset({
    "and_logic", "or_logic", "not_logic",
    "xor_logic", "implies_logic", "eq_logic", "if_logic",
    "to_logic_bool",
})


# Stage 38 post-Inc-3 silent-failure F2 fix (MEDIUM, conf 90):
# the 12 frame builtins are identity-lowered at IR (the wrapper is
# a compile-time tag; the runtime value passes through unchanged).
# The chain rule for an identity wrapper IS identity:
# `d(into_world(u))/dx = du/dx`. Pre-fix the opaque-call catchall at
# autodiff.py:1058 / autodiff_reverse.py:683 raised
# NotImplementedError on any `grad(use_frame)` site, contradicting
# the AD-pure registration's implied "frame ops are differentiable"
# contract. Tier ops (Stage 37) still raise; a future increment may
# backfill the same arm there once the symmetric question is settled.
# Stage 43 Inc 1 LOW-3 fix: renamed from `_FRAME_IDENTITY_AD_NAMES`
# to reflect that the set now covers all 5 wrapper families
# (frame + temporal + modal + causal — only 12 of 45 entries are
# actual frames). Stage 47 dropped the backwards-compat alias.
_IDENTITY_AD_CHAIN_RULE_NAMES = frozenset({
    "into_world", "into_robot", "into_camera",
    "from_world", "from_robot", "from_camera",
    "world_to_robot", "robot_to_world",
    "robot_to_camera", "camera_to_robot",
    "world_to_camera", "camera_to_world",
    # Stage 39 Inc 1 + Inc 2 — temporal kinds share the same identity
    # chain rule as frames: `d(into_past(u))/dx = du/dx`. Phase-0
    # wrapper-shift semantics; the only structural difference from
    # frames is the wrapper tag, which the AD pass never inspects.
    # Reusing the frame-identity arm avoids a parallel set + duplicate
    # test surface. (Stage 39 closure gate-1 silent-failure F4 fix:
    # do NOT call this block "Stage 38 frames" — that's the wrong
    # stage label and breaks grep-by-stage audits.)
    "into_past", "into_present", "into_future", "into_eternal",
    "from_past", "from_present", "from_future", "from_eternal",
    "to_past", "forecast", "recall_past", "actualize",
    # Stage 40 Inc 1 + Inc 2 — modal/epistemic kinds share the same
    # identity chain rule as frames/temporals: the wrapper tag is
    # opaque to AD; gradient flows through the single inner arg. Added
    # preemptively (Stage 39 lesson F4: do not relabel the block on a
    # later stage's behalf; keep the stage tag accurate so grep-by-
    # stage audits work). Covers 4 intro + 4 elim + 2 transitions.
    "into_known", "into_believed", "into_goal", "into_uncertain",
    "from_known", "from_believed", "from_goal", "from_uncertain",
    "confirm", "act_on",
    # Stage 41 Inc 1 + Inc 2 — causal/intent kinds: 4 intro + 4
    # elim + 3 transitions = 11 names. Same identity chain rule
    # as the prior 4 wrapper families.
    "into_cause", "into_effect", "into_joint", "into_independent",
    "from_cause", "from_effect", "from_joint", "from_independent",
    "propagate", "aggregate", "isolate",
})

# Stage 47: dropped the `_FRAME_IDENTITY_AD_NAMES` backwards-
# compat alias (Stage 43 Inc 1 LOW-3 deferral). Three stages
# of grace period elapsed (44, 45, 46); all internal use sites
# now reference `_IDENTITY_AD_CHAIN_RULE_NAMES` directly. Any
# external importer must do the rename.


# Mapping from integer-valued Logic op to its closest fuzzy
# (differentiable) replacement, surfaced in the NotImplementedError
# message. Ops without a 1:1 fuzzy twin (eq_logic / if_logic /
# to_logic_bool) point at the general guidance.
_LOGIC_FUZZY_HINTS = {
    "and_logic": "fuzzy_and",
    "or_logic": "fuzzy_or",
    "not_logic": "fuzzy_not",
    "xor_logic": "fuzzy_xor",
    "implies_logic": "fuzzy_implies",
}


def _raise_integer_logic_in_ad(callee_name: str, mode: str) -> None:
    """Stage 36 Inc 12: refuse to silently zero through an integer
    boolean Logic op in differentiated code. `mode` is 'forward' or
    'reverse' for diagnostic provenance."""
    hint = _LOGIC_FUZZY_HINTS.get(callee_name)
    if hint:
        suggestion = (
            f"use {hint!r} (continuous fuzzy-logic relaxation) if a "
            "gradient is needed; otherwise hoist this call outside the "
            "differentiated function"
        )
    else:
        suggestion = (
            "this op has no differentiable fuzzy twin; replace it with "
            "fuzzy_and / fuzzy_or / fuzzy_not composition or hoist the "
            "call outside the differentiated function"
        )
    raise NotImplementedError(
        f"{mode}-mode AD: {callee_name!r} is integer-valued boolean "
        f"logic and has no chain rule; {suggestion}"
    )


# Audit 28.8 B5: trap 85001 — AD assumed 0 derivative for an unhandled
# expression kind. Both forward (_diff) and reverse (_propagate) used
# to fall through to "return 0" / "no contribution" for any unmatched
# node — Quote, Splice, Modify, UnsafeBlock, Cast on a non-arithmetic
# target. The user got `grad(f)(x) = 0` with no diagnostic.
#
# Fix: each unhandled-node site now appends a diagnostic to this
# module-level list. The CLI (helixc/check.py) flushes the list at
# the end of compilation and prints warnings to stderr; `-Wad=error`
# promotes them to errors. Tests can drain via `take_diff_warnings`.
_DIFF_WARNINGS: list[str] = []

# Trap-id reservation for "AD assumed 0 derivative" diagnostic.
TRAP_AD_ASSUMED_ZERO = 85001


# Audit 28.8 cycle 2 B:C9: shared numeric-type set for AD Cast arms.
# Pre-fix the Cast arms in both autodiff._diff and
# autodiff_reverse._propagate hardcoded a 14-element list that
# omitted `bool`, `char`, `fp8`, `mxfp4`, `nvfp4` — all five of
# which are accepted by typecheck's `_is_numeric_scalar`. So `x as
# bool` (valid per the matrix) inside a grad-rewritten fn emitted
# a spurious 85001 warning. Unify via a frozenset that mirrors
# typecheck's numeric domain.
#
# Note: bool/char casts are technically discontinuous, so a future
# enhancement could flag them with a separate diagnostic ("AD
# through discontinuous cast"). Phase-0: accept as numeric, do not
# spuriously warn.
NUMERIC_FOR_AD: frozenset[str] = frozenset({
    "i8", "i16", "i32", "i64", "isize",
    "u8", "u16", "u32", "u64", "usize",
    "f16", "bf16", "f32", "f64",
    "fp8", "mxfp4", "nvfp4",
    "bool", "char",
})


def take_diff_warnings() -> list[str]:
    """Atomically read-and-clear the module-level AD warning list.

    Callers (helixc/check.py and tests) should drain this list at a
    well-defined point so warnings from a previous compilation unit
    don't leak into the next. Multiple drains across one compile
    aggregate (the list is reset only on take())."""
    global _DIFF_WARNINGS
    out = _DIFF_WARNINGS
    _DIFF_WARNINGS = []
    return out


def _ad_warn(node, reason: str) -> None:
    """Record an AD diagnostic for an unhandled expression. Includes
    source span when available + trap-id 85001 for grep-ability."""
    span = getattr(node, "span", None)
    kind = type(node).__name__
    line_col = (f"{span.line}:{span.col}: " if span is not None
                else "")
    _DIFF_WARNINGS.append(
        f"{line_col}AD: assumed 0 derivative for {kind} ({reason}) "
        f"(trap {TRAP_AD_ASSUMED_ZERO})"
    )


def clear_diff_cache() -> None:
    """Reset the differentiate-memo cache (for tests)."""
    _DIFF_CACHE.clear()
    _DIFF_CACHE_HITS[0] = 0
    _DIFF_CACHE_MISSES[0] = 0


def diff_cache_stats() -> tuple[int, int]:
    """Return (hits, misses) since last `clear_diff_cache()`."""
    return (_DIFF_CACHE_HITS[0], _DIFF_CACHE_MISSES[0])


def _fn_table_sig(fn_table: dict[str, "A.FnDecl"] | None) -> str:
    # Stage 28.9 cycle 53 audit-R C52-AD1 fix (HIGH): include body
    # hash so any body change invalidates the cache.
    # Stage 28.9 cycle 55 audit-T C54-AD1 + C54-AD2 + C54-AD3 follow-on
    # fixes (HIGH/MED/MED):
    #   - C54-AD1: include `tuple(sorted(fn.attrs))`. `_inline_user_calls`
    #     at line ~365 reads `"pure" in fn.attrs` to decide whether
    #     to inline; two fn_tables with same body but different @pure
    #     marker produce different inlining → different derivatives.
    #   - C54-AD2: include `len(fn.params)` (arity). With de-Bruijn
    #     body hashing, `fn g(x,y) = x` and `fn g(x) = x` produce
    #     the SAME body hash but differ in inlining gating
    #     (`len(fn.params) == len(args)` check at call sites).
    #   - C54-AD3: catch NotImplementedError too. structural_hash
    #     raises NIE on unknown AST subclasses (cycle-35 loud-fail
    #     discipline); the autodiff cache's INTENT is to bypass-cache
    #     on hash failure (sentinel path below), but NIE wasn't
    #     caught so it would propagate up and crash the caller
    #     instead of degrading gracefully.
    if not fn_table:
        return ""
    parts: list[str] = []
    for name in sorted(fn_table.keys()):
        fn = fn_table[name]
        try:
            body_hash = structural_hash(fn.body)
        except (TypeError, ValueError, AttributeError, NotImplementedError):
            # Hash failure → use a sentinel that differs from any
            # legitimate hash. Bypass-cache effect; preserves
            # correctness.
            body_hash = f"<unhashable:{id(fn.body)}>"
        # Include arity + attrs alongside body hash so all three
        # dimensions `_inline_user_calls` actually uses are captured.
        attrs_part = ",".join(sorted(fn.attrs))
        parts.append(f"{name}/{len(fn.params)}/{attrs_part}/{body_hash}")
    return "|".join(parts)


def differentiate(expr: A.Expr, var: str,
                  fn_table: dict[str, "A.FnDecl"] | None = None) -> A.Expr:
    """Return the AST of d(expr)/d(var), simplified.

    Memoized by structural hash of `expr` + var + fn_table signature.
    Returns a deepcopy of the cached deriv so callers can mutate.

    Optionally accepts a `fn_table` mapping function names to FnDecls. When
    provided, calls to user-defined @pure functions in the expression are
    inlined (their bodies substituted for the call) before differentiation.
    This makes grad work across function boundaries — `grad(f)` where f's
    body calls a helper `g(x)` propagates the gradient through g.

    If `expr` is a Block, the block's let-bindings are inlined first so
    that subsequent uses of the bound names refer to their definitions.
    """
    # Audit 28.8 cycle 2 (deferred observation #20): pre-fix this
    # `except Exception: key = None` silently disabled the cache on
    # any hash failure with NO diagnostic. Future AST extensions
    # could quietly skip caching forever (perf regression, no signal).
    # Narrowed to the actually-expected hashing exceptions; on a
    # genuine hash failure we still bypass the cache but emit a
    # warning via the AD channel so the user can spot the recurring
    # miss.
    try:
        key = (structural_hash(expr), var, _fn_table_sig(fn_table))
    except (TypeError, ValueError, AttributeError, NotImplementedError) as e:
        # Stage 28.9 cycle 55 audit-R C54-AD3 fix (MED): catch
        # NotImplementedError too — structural_hash raises NIE for
        # unknown AST subclasses (cycle-35 loud-fail discipline).
        # Without this catch, a novel AST node in `expr` would
        # propagate NIE through the cache layer and crash the caller
        # instead of gracefully bypassing the cache.
        key = None  # hash failure → bypass cache
        _ad_warn(
            expr,
            f"differentiate cache bypassed: hashing failed "
            f"({type(e).__name__}: {e}) — perf regression but "
            f"correctness preserved",
        )
    if key is not None and key in _DIFF_CACHE:
        _DIFF_CACHE_HITS[0] += 1
        return _copy.deepcopy(_DIFF_CACHE[key])

    if fn_table:
        expr = _inline_user_calls(expr, fn_table)
    inlined = _inline_lets(expr, {})
    deriv = _diff(inlined, var)
    out = _simplify(deriv)

    if key is not None:
        _DIFF_CACHE_MISSES[0] += 1
        _DIFF_CACHE[key] = _copy.deepcopy(out)
    return out


def _is_inferably_pure(fn: "A.FnDecl",
                        fn_table: dict[str, "A.FnDecl"],
                        visiting: frozenset[str] | None = None) -> bool:
    """Stage 13: infer whether a user fn is safe to inline for AD without an
    explicit `@pure` attribute. A fn is inferably pure iff its body uses only
    expressions whose gradient is well-defined and whose evaluation has no
    observable side-effects:

      - literals (int, float, bool, char, str)
      - parameter names / let-bound names
      - arithmetic Binary/Unary
      - If with pure cond/then/else
      - Block with pure let-stmts and pure final_expr (no Assign)
      - Calls to inferably-pure user fns or known transcendental builtins

    Anything else (Assign, For, While, Loop, Match, Index, calls to
    non-pure fns or unknown builtins) -> not inferred pure. The caller
    falls back to "leave as opaque call, derivative is 0" — same behaviour
    as before Stage 13.

    Used by `_inline_user_calls` so the plan test in Stage 13
    (`fn g(x) = x*x; fn f(x) = g(x)+x; grad(f)(3)=7`) works without forcing
    the user to mark every arithmetic helper `@pure`.
    """
    visiting = visiting or frozenset()
    # Cycles: if we're already inferring fn, treat as pure (the caller's
    # visiting-set in _inline_user_calls will block re-inlining anyway).
    if fn.name in visiting:
        return True
    new_visiting = visiting | {fn.name}

    def is_pure_expr(e) -> bool:
        if e is None:
            return True
        if isinstance(e, (A.IntLit, A.FloatLit, A.BoolLit, A.CharLit, A.StrLit)):
            return True
        if isinstance(e, A.Name):
            return True
        if isinstance(e, A.Binary):
            return is_pure_expr(e.left) and is_pure_expr(e.right)
        if isinstance(e, A.Unary):
            return is_pure_expr(e.operand)
        if isinstance(e, A.Cast):
            return is_pure_expr(e.value)
        if isinstance(e, A.Block):
            for s in e.stmts:
                if isinstance(s, A.Let) and s.value is not None:
                    if not is_pure_expr(s.value):
                        return False
                elif isinstance(s, A.ConstStmt):
                    if not is_pure_expr(s.value):
                        return False
                elif isinstance(s, A.ExprStmt):
                    if not is_pure_expr(s.expr):
                        return False
                else:
                    # Assign/For/While/Loop/Return/etc. -> impure.
                    return False
            return is_pure_expr(e.final_expr)
        if isinstance(e, A.If):
            return (is_pure_expr(e.cond)
                    and is_pure_expr(e.then)
                    and is_pure_expr(e.else_))
        if isinstance(e, A.Call):
            if not isinstance(e.callee, A.Name):
                return False
            cname = e.callee.name
            # Recurse into args first.
            for a in e.args:
                if not is_pure_expr(a):
                    return False
            if cname in AD_KNOWN_PURE_CALLS:
                return True
            # User fn — recursively check.
            if cname in fn_table:
                return _is_inferably_pure(fn_table[cname], fn_table,
                                           new_visiting)
            # Unknown callee — conservative reject.
            return False
        # Anything else (Match, For, While, Loop, Assign, Return, Index,
        # Tuple-related, struct-related): not inferable as pure for AD.
        return False

    return is_pure_expr(fn.body)


def _inline_user_calls(expr: A.Expr, fn_table: dict[str, "A.FnDecl"],
                        depth: int = 0, max_depth: int = 4,
                        visiting: frozenset[str] | None = None,
                        unroll_counts: dict[str, int] | None = None,
                        max_unroll: int = 3) -> A.Expr:
    """Walk `expr` and replace each Call(Name(f), args) where f is a known
    inlinable function in `fn_table` with a deepcopy of f's body, with each
    parameter substituted by the corresponding argument expression.

    A function is inlinable if either it has the `@pure` attribute OR its
    body is inferably pure (Stage 13: only arithmetic / pure-call chain).
    Stage 13 added the inferred-purity path so plain helper fns work in
    `grad(f)` without forcing the user to mark every arithmetic helper.

    Skips:
      - Transcendental builtins (`__exp`, `__log`, etc.) — they have
        analytic AD chain rules already wired into _diff.
      - Functions currently in `visiting` (mutual / direct recursion
        guard — prevents exponential AST expansion when inlining cycles
        like a→b→a). Stage 13: traps via the trap-id 87001 documented in
        the plan; runtime impact is now "leave call as opaque so AD fails
        closed unless a chain rule exists".
      - depth >= max_depth (safety net).
      - Functions not in fn_table (treated as opaque external).
      - Extern declarations and bodyless functions (left opaque; AD engines
        must handle or reject the call explicitly).

    Stage 54 Inc 3b — bounded recursive unrolling:
      For functions in `visiting` (would otherwise be left opaque),
      check `unroll_counts.get(fn_name, 0) < max_unroll`. If so,
      AND the recursive call args are literal-reducible
      (`_args_are_unroll_safe`), allow ONE more inline pass
      with the counter incremented. Enables `power(x, 3)`-style
      recursive helpers with literal counter to unroll into
      `x * x * x` at compile time, exposing the gradient.
      max_unroll=3 caps exponential growth at 3 levels per fn.
    """
    import copy as _copy

    # Functions with analytic AD chain rules in _diff_call_chain_rule /
    # autodiff_reverse._propagate. Inlining these would force the AD
    # engine to differentiate through their (potentially conditional)
    # bodies instead of using the closed-form derivative — producing
    # silently-wrong gradients when the body uses if/while.
    TRANSCENDENTALS = AD_KNOWN_PURE_CALLS
    visiting = visiting or frozenset()
    unroll_counts = unroll_counts if unroll_counts is not None else {}

    def _is_literal_only(e: A.Expr) -> bool:
        """Recursively check if `e` has NO Name leaves (i.e., constant-
        folds to a literal). IntLit/FloatLit/BoolLit/StrLit/CharLit
        are literals; Binary/Unary/Cast of literal-only children are
        literal-only. Anything else (Name, Call, Block, etc.) breaks
        the chain — return False."""
        if e is None:
            return False
        if isinstance(e, (A.IntLit, A.FloatLit, A.BoolLit,
                          A.StrLit, A.CharLit)):
            return True
        if isinstance(e, A.Binary):
            return (_is_literal_only(e.left)
                    and _is_literal_only(e.right))
        if isinstance(e, A.Unary):
            return _is_literal_only(e.operand)
        if isinstance(e, A.Cast):
            return _is_literal_only(e.value)
        return False

    def _args_are_unroll_safe(args: list[A.Expr]) -> bool:
        """Stage 54 Inc 3b + Cycle 1 Auditor 4 HIGH-1 tightening:
        a recursive call is unroll-safe if AT LEAST ONE arg is a
        literal INTEGER (constant-folds with no Name leaves AND is
        an integer-typed literal) — the assumption being that
        integer literals are the canonical recursion-bounding
        pattern (`power(x, 3)`, `factorial(n: 5)`, etc.).

        Pre-Cycle-1: ANY literal counted, including float padding
        and the `0` in `loop_ish(x+1, 0)` where the literal arg
        doesn't actually drive recursion — silently unrolling a
        non-terminating recursion 3 levels deep produced a
        partially-unrolled gradient that compiled and returned a
        wrong-but-plausible value. Auditor 4 flagged this as HIGH.

        Post-Cycle-1: require at least one IntLit (or Cast(IntLit)).
        Float / bool / str literals don't count as recursion-bounds.
        This eliminates the `_unused=0` false positive while still
        catching `power(x, 3)` (where 3 is IntLit) and the existing
        Stage 54 Inc 3b regression test cases."""
        for a in args:
            if _is_literal_only(a) and _has_int_literal_leaf(a):
                return True
        return False

    def _has_int_literal_leaf(e: A.Expr) -> bool:
        """Helper for Cycle 1 HIGH-1 tightening: True iff `e` is or
        recursively contains an A.IntLit leaf. Used to gate unroll
        on integer-typed literals only."""
        if isinstance(e, A.IntLit):
            return True
        if isinstance(e, A.Binary):
            return (_has_int_literal_leaf(e.left)
                    or _has_int_literal_leaf(e.right))
        if isinstance(e, A.Unary):
            return _has_int_literal_leaf(e.operand)
        if isinstance(e, A.Cast):
            return _has_int_literal_leaf(e.value)
        return False

    def go(e: A.Expr) -> A.Expr:
        if isinstance(e, A.Call):
            new_callee = go(e.callee)
            new_args = [go(a) for a in e.args]
            # Stage 54 Inc 3b bounded recursive unrolling: when callee
            # IS in visiting (would otherwise leave opaque), check
            # unroll_counts and args; allow one more inline pass.
            is_recursive_unroll = False
            if (isinstance(new_callee, A.Name)
                    and new_callee.name in fn_table
                    and new_callee.name not in TRANSCENDENTALS
                    and new_callee.name in visiting
                    and unroll_counts.get(new_callee.name, 0) < max_unroll
                    and _args_are_unroll_safe(new_args)
                    and depth < max_depth):
                is_recursive_unroll = True
            if (isinstance(new_callee, A.Name)
                    and new_callee.name in fn_table
                    and new_callee.name not in TRANSCENDENTALS
                    and (new_callee.name not in visiting
                         or is_recursive_unroll)
                    and depth < max_depth):
                fn = fn_table[new_callee.name]
                if (getattr(fn, "is_extern", False)
                        or fn.body is None
                        or getattr(fn.body, "final_expr", None) is None):
                    return A.Call(span=e.span, callee=new_callee, args=new_args)
                # Stage 13: inline if @pure OR inferably pure (arithmetic/
                # pure-call chain). Other fns may have effects whose
                # differentiation is unsound — leave as opaque call.
                if not _is_inferably_pure(fn, fn_table):
                    return A.Call(span=e.span, callee=new_callee, args=new_args)
                if len(fn.params) != len(new_args):
                    return A.Call(span=e.span, callee=new_callee, args=new_args)
                # Build substitution map: param name -> arg expression
                substitutions = {p.name: a for p, a in zip(fn.params, new_args)}
                # Deepcopy the body so we don't share references with the
                # original function (downstream passes mutate in-place).
                body_copy = _copy.deepcopy(fn.body)
                substituted = _substitute_names(body_copy, substitutions)
                # Recursively inline within the substituted body. Add this
                # function to the visiting set so any recursive (direct or
                # mutual) call back to it is treated as opaque (or
                # unrolled if Inc 3b's _args_are_unroll_safe + max_unroll
                # guards allow it).
                next_unroll = dict(unroll_counts)
                if is_recursive_unroll:
                    next_unroll[new_callee.name] = (
                        next_unroll.get(new_callee.name, 0) + 1)
                return _inline_user_calls(substituted, fn_table, depth + 1,
                                           max_depth,
                                           visiting | {new_callee.name},
                                           next_unroll, max_unroll)
            return A.Call(span=e.span, callee=new_callee, args=new_args)
        if isinstance(e, A.Binary):
            return A.Binary(span=e.span, op=e.op, left=go(e.left), right=go(e.right))
        if isinstance(e, A.Unary):
            return A.Unary(span=e.span, op=e.op, operand=go(e.operand))
        if isinstance(e, A.Block):
            new_stmts = []
            for s in e.stmts:
                if isinstance(s, A.Let) and s.value is not None:
                    new_stmts.append(A.Let(span=s.span, name=s.name,
                                            ty=s.ty, value=go(s.value),
                                            is_mut=s.is_mut))
                elif isinstance(s, A.ConstStmt):
                    new_stmts.append(A.ConstStmt(span=s.span, name=s.name,
                                                  ty=s.ty, value=go(s.value)))
                elif isinstance(s, A.ExprStmt):
                    # Stage 54 gate-1 silent-failure fix: pre-fix,
                    # ExprStmt fell to the else branch and was
                    # appended unchanged. That made Inc 3a's
                    # For/While/Loop arms in go() unreachable for
                    # the typical case where a loop is wrapped in
                    # ExprStmt (the parser emits `while ... { };`
                    # as ExprStmt(While(...))). Now we recurse into
                    # the wrapped expr so the walker can reach
                    # loop bodies AND any helper-calls used at
                    # statement position (e.g., `pure_helper(x);`).
                    new_stmts.append(A.ExprStmt(span=s.span,
                                                  expr=go(s.expr)))
                else:
                    new_stmts.append(s)
            new_final = go(e.final_expr) if e.final_expr is not None else None
            return A.Block(span=e.span, stmts=new_stmts, final_expr=new_final)
        if isinstance(e, A.If):
            # Recurse into then/else regardless of whether they're Blocks
            # — defensively handle hand-built ASTs with bare-expr branches.
            new_then = go(e.then) if e.then is not None else None
            new_else = go(e.else_) if e.else_ is not None else None
            return A.If(span=e.span, cond=go(e.cond),
                        then=new_then, else_=new_else)
        # Stage 54 Inc 3a: loop-body descent. Pre-fix, A.For/A.While/
        # A.Loop were returned as-is, so any pure-helper calls inside
        # the loop body were never inlined — subsequent AD passes saw
        # them as opaque calls and failed closed (or returned zero
        # silently). Descending the walker into loop bodies matches
        # the existing If-branch / Block-stmt descent pattern.
        # Note: this only handles helper-call inlining inside loop
        # bodies — the LOOP ITSELF still requires the AD pass to know
        # how to differentiate iteration, which is NOT implemented.
        # Stage 54 post-close HIGH-2 docstring honesty fix (prior
        # version incorrectly claimed "loops are mostly unrolled at
        # compile time by other passes" — that's only true for the
        # tile matmul small-loop unroller in ir/lower_ast.py, NOT
        # for arbitrary helper-fn loops). Reality: a loop reaching
        # _diff or _propagate is caught by an explicit fail-loud
        # NotImplementedError arm (post-close CRITICAL-2 fix in
        # _diff; CRITICAL-1 mirror landed at gate-5 / 9424133 for
        # _propagate). If you add an iterative-AD arm in the
        # future, remove the raise arms so the descent here
        # actually feeds them.
        if isinstance(e, A.For):
            new_iter = go(e.iter_expr)
            new_body = go(e.body)
            assert isinstance(new_body, A.Block), \
                "loop body must remain a Block after walker"
            return A.For(span=e.span, var_name=e.var_name,
                         iter_expr=new_iter, body=new_body)
        if isinstance(e, A.While):
            new_cond = go(e.cond)
            new_body = go(e.body)
            assert isinstance(new_body, A.Block), \
                "while body must remain a Block after walker"
            return A.While(span=e.span, cond=new_cond, body=new_body)
        if isinstance(e, A.Loop):
            new_body = go(e.body)
            assert isinstance(new_body, A.Block), \
                "loop body must remain a Block after walker"
            return A.Loop(span=e.span, body=new_body)
        # Stage 54 gate-2 sweep: descent into remaining AST kinds.
        # Pre-fix these all fell to `return e`, so any pure-helper
        # call wrapped inside one of these forms (e.g. `arr[h(x)]`,
        # `(h(x), 0)`, `Point{x: h(x)}`, `unsafe { h(x) }`,
        # `return h(x)`, `h(x) as i32`) would never be inlined,
        # leaving the AD pass to fail-closed or silently zero.
        # Same defect class as the For/While/Loop arms from Inc 3a.
        if isinstance(e, A.Cast):
            return A.Cast(span=e.span, value=go(e.value),
                          target_ty=e.target_ty)
        if isinstance(e, A.Index):
            return A.Index(span=e.span, callee=go(e.callee),
                           indices=[go(i) for i in e.indices])
        if isinstance(e, A.Field):
            return A.Field(span=e.span, obj=go(e.obj), name=e.name)
        if isinstance(e, A.TupleLit):
            return A.TupleLit(span=e.span,
                              elems=[go(x) for x in e.elems])
        if isinstance(e, A.ArrayLit):
            return A.ArrayLit(span=e.span,
                              elems=[go(x) for x in e.elems])
        if isinstance(e, A.StructLit):
            return A.StructLit(
                span=e.span, name=e.name,
                fields=[(n, go(v)) for (n, v) in e.fields],
            )
        if isinstance(e, A.UnsafeBlock):
            new_body = go(e.body)
            assert isinstance(new_body, A.Block), \
                "unsafe body must remain a Block after walker"
            return A.UnsafeBlock(span=e.span, body=new_body)
        if isinstance(e, A.Match):
            new_arms = []
            for arm in e.arms:
                new_guard = (go(arm.guard) if arm.guard is not None
                             else None)
                new_body = go(arm.body)
                new_arms.append(A.MatchArm(
                    span=arm.span, pattern=arm.pattern,
                    guard=new_guard, body=new_body,
                ))
            return A.Match(span=e.span, scrutinee=go(e.scrutinee),
                           arms=new_arms)
        if isinstance(e, A.Assign):
            return A.Assign(span=e.span, op=e.op,
                            target=go(e.target), value=go(e.value))
        if isinstance(e, A.Return):
            new_val = (go(e.value) if e.value is not None else None)
            return A.Return(span=e.span, value=new_val)
        if isinstance(e, A.Break):
            new_val = (go(e.value) if e.value is not None else None)
            return A.Break(span=e.span, value=new_val)
        if isinstance(e, A.Range):
            new_start = (go(e.start) if e.start is not None else None)
            new_end = (go(e.end) if e.end is not None else None)
            return A.Range(span=e.span, start=new_start, end=new_end)
        # Cycle 1 Auditor 4 HIGH-2 fix: pre-fix, A.Modify / A.Quote /
        # A.Splice / A.TileLit / A.Path all fell to `return e`, so any
        # pure-helper call wrapped inside one of these forms was left
        # opaque (e.g., `modify(target=h(x), ...)` or `tile<f32, [h(x)]>`
        # or `quote { h(x) }`). The AD pass then saw the inner call
        # as opaque and either failed closed (loud, acceptable) OR
        # silently returned zero gradient (the gradient defect Auditor
        # 4 flagged). Mirrors the gate-4 fix for `_name_appears_in`
        # that closed the same AST-kind gap one analysis layer over.
        if isinstance(e, A.Modify):
            return A.Modify(
                span=e.span,
                target=go(e.target),
                transformation=go(e.transformation),
                verifier=go(e.verifier),
            )
        if isinstance(e, A.Quote):
            return A.Quote(span=e.span, inner=go(e.inner))
        if isinstance(e, A.Splice):
            return A.Splice(span=e.span, inner=go(e.inner))
        if isinstance(e, A.TileLit):
            return A.TileLit(
                span=e.span,
                dtype=e.dtype,
                shape=[go(s) for s in e.shape],
                memspace=go(e.memspace),
                init=e.init,
            )
        if isinstance(e, A.Path):
            # Path is a dotted identifier reference (segments: list[str])
            # with no expression-shaped children, so there's nothing to
            # walk. Return unchanged. Listed here explicitly to document
            # that we considered + dismissed the case rather than
            # falling through silently.
            return e
        return e

    return go(expr)


def _substitute_names(expr: A.Expr, subs: dict[str, A.Expr]) -> A.Expr:
    """Replace each occurrence of A.Name(n) where n in subs with subs[n].
    Block-scoped: a `let` shadowing a substituted name removes it from the
    scope of the rest of the block."""
    import copy as _copy

    def go(e: A.Expr, env: dict[str, A.Expr]) -> A.Expr:
        if isinstance(e, A.Name):
            if e.name in env:
                # Each substitution site gets its own copy so downstream
                # in-place mutation doesn't cross-contaminate.
                return _copy.deepcopy(env[e.name])
            return e
        if isinstance(e, A.Binary):
            return A.Binary(span=e.span, op=e.op,
                            left=go(e.left, env), right=go(e.right, env))
        if isinstance(e, A.Unary):
            return A.Unary(span=e.span, op=e.op, operand=go(e.operand, env))
        if isinstance(e, A.Call):
            return A.Call(span=e.span, callee=go(e.callee, env),
                          args=[go(a, env) for a in e.args])
        if isinstance(e, A.Assign):
            # Stage 54 gate-1 silent-failure HIGH-1 extended fix:
            # Assign is an Expr (e.g. `x = y` parsed as
            # Assign(target=Name(x), op="=", value=Name(y))).
            # Pre-fix, Assign fell to `return e` so its
            # target/value children weren't walked, leaving
            # any param-name on either side un-substituted
            # after helper inlining. Same defect class as the
            # For/While/Loop/Match arm gap.
            return A.Assign(span=e.span, op=e.op,
                            target=go(e.target, env),
                            value=go(e.value, env))
        if isinstance(e, A.If):
            new_then = (_go_block(e.then, env) if isinstance(e.then, A.Block)
                        else go(e.then, env))
            new_else = (_go_block(e.else_, env)
                        if e.else_ is not None and isinstance(e.else_, A.Block)
                        else (go(e.else_, env) if e.else_ is not None else None))
            return A.If(span=e.span, cond=go(e.cond, env),
                        then=new_then, else_=new_else)
        if isinstance(e, A.Block):
            return _go_block(e, env)
        # Stage 54 closure gate-1 silent-failure HIGH-1 fix
        # (latent bug exposed by Inc 3a): Loop and Match arms.
        # Pre-Inc-3a, `_inline_user_calls` never descended into
        # loop bodies, so helpers called inside loops were left
        # opaque — never inlined, never substituted, never
        # triggered this bug. Inc 3a opened the path; now a
        # helper whose body contains a For/While/Loop/Match
        # would fail to substitute params inside those forms,
        # silently producing wrong gradients (or unbound-name
        # type errors, depending on shadowing).
        if isinstance(e, A.For):
            # The loop variable shadows any param substitution
            # for the same name within the body.
            inner_env = dict(env)
            inner_env.pop(e.var_name, None)
            return A.For(
                span=e.span, var_name=e.var_name,
                iter_expr=go(e.iter_expr, env),
                body=_go_block(e.body, inner_env),
            )
        if isinstance(e, A.While):
            return A.While(
                span=e.span, cond=go(e.cond, env),
                body=_go_block(e.body, env),
            )
        if isinstance(e, A.Loop):
            return A.Loop(span=e.span, body=_go_block(e.body, env))
        if isinstance(e, A.Match):
            new_arms = []
            for arm in e.arms:
                # Pattern bindings shadow incoming substitutions.
                # Stage 54 post-close HIGH-3 fix: deep destructure
                # patterns (PatTuple / PatVariant / PatOr containing
                # PatBind) silently leaked caller-scope substitutions
                # into bindings that the inner pattern was meant to
                # shadow — a real wrong-gradient hazard for any
                # inlined helper whose body matches on Option /
                # Result / tuple destructure. Hard-fail loudly until
                # proper recursive binder analysis lands (gate-2
                # audit option-(b), aligned with the silent-failure-
                # discipline default). PatBind / PatWildcard / PatLit
                # cover the AD-inlining safe subset.
                allowed = (A.PatBind, A.PatWildcard, A.PatLit)
                if not isinstance(arm.pattern, allowed):
                    raise NotImplementedError(
                        f"_substitute_names: pattern kind "
                        f"{type(arm.pattern).__name__} not supported "
                        f"inside AD-inlined Match arm; "
                        f"only PatBind/PatWildcard/PatLit are "
                        f"covered today. Refactor the helper to "
                        f"avoid destructuring patterns, or extend "
                        f"the substituter with recursive binder "
                        f"collection."
                    )
                arm_env = dict(env)
                if isinstance(arm.pattern, A.PatBind):
                    arm_env.pop(arm.pattern.name, None)
                new_guard = (
                    go(arm.guard, arm_env)
                    if arm.guard is not None else None
                )
                new_body = (
                    _go_block(arm.body, arm_env)
                    if isinstance(arm.body, A.Block)
                    else go(arm.body, arm_env)
                )
                new_arms.append(A.MatchArm(
                    span=arm.span, pattern=arm.pattern,
                    guard=new_guard, body=new_body,
                ))
            return A.Match(
                span=e.span, scrutinee=go(e.scrutinee, env),
                arms=new_arms,
            )
        # Stage 54 gate-2 sweep parity: substitute through the
        # remaining AST kinds. Same defect class as gate-1's
        # For/While/Loop/Match gap. Pre-fix, a helper containing
        # `arr[p]`, `obj.p`, `(p, 0)`, `[p]`, `Point{x: p}`,
        # `unsafe { p }`, `return p`, `p as f64`, or `0..p` would
        # not have `p` substituted with the actual arg when
        # inlined — leaving an unbound name in the caller scope.
        if isinstance(e, A.Cast):
            return A.Cast(span=e.span, value=go(e.value, env),
                          target_ty=e.target_ty)
        if isinstance(e, A.Index):
            return A.Index(
                span=e.span, callee=go(e.callee, env),
                indices=[go(i, env) for i in e.indices],
            )
        if isinstance(e, A.Field):
            return A.Field(span=e.span, obj=go(e.obj, env),
                           name=e.name)
        if isinstance(e, A.TupleLit):
            return A.TupleLit(
                span=e.span,
                elems=[go(x, env) for x in e.elems],
            )
        if isinstance(e, A.ArrayLit):
            return A.ArrayLit(
                span=e.span,
                elems=[go(x, env) for x in e.elems],
            )
        if isinstance(e, A.StructLit):
            return A.StructLit(
                span=e.span, name=e.name,
                fields=[(n, go(v, env)) for (n, v) in e.fields],
            )
        if isinstance(e, A.UnsafeBlock):
            return A.UnsafeBlock(
                span=e.span,
                body=_go_block(e.body, env),
            )
        if isinstance(e, A.Return):
            new_val = (go(e.value, env)
                       if e.value is not None else None)
            return A.Return(span=e.span, value=new_val)
        if isinstance(e, A.Break):
            new_val = (go(e.value, env)
                       if e.value is not None else None)
            return A.Break(span=e.span, value=new_val)
        if isinstance(e, A.Range):
            new_start = (go(e.start, env)
                         if e.start is not None else None)
            new_end = (go(e.end, env)
                       if e.end is not None else None)
            return A.Range(span=e.span, start=new_start, end=new_end)
        return e

    def _go_block(blk: A.Block, env: dict[str, A.Expr]) -> A.Block:
        local_env = dict(env)
        new_stmts = []
        for s in blk.stmts:
            if isinstance(s, A.Let) and s.value is not None:
                new_val = go(s.value, local_env)
                # The let shadows any incoming substitution for the same name
                local_env.pop(s.name, None)
                new_stmts.append(A.Let(span=s.span, name=s.name, ty=s.ty,
                                        value=new_val, is_mut=s.is_mut))
            elif isinstance(s, A.ConstStmt):
                new_val = go(s.value, local_env)
                local_env.pop(s.name, None)
                new_stmts.append(A.ConstStmt(span=s.span, name=s.name,
                                              ty=s.ty, value=new_val))
            elif isinstance(s, A.ExprStmt):
                # Stage 54 gate-1 silent-failure HIGH-1 extended
                # fix (parallel to the inliner's Block-stmts
                # ExprStmt-descent fix): pre-fix, ExprStmt fell
                # to the else branch and the wrapped expression's
                # children were never walked. After Inc 3a opened
                # loop-body inlining, this meant a helper with
                # `while ... { x = x + p; }` would have its
                # param-references inside the Assign stay
                # un-substituted (silently wrong gradient).
                new_stmts.append(A.ExprStmt(span=s.span,
                                              expr=go(s.expr, local_env)))
            else:
                new_stmts.append(s)
        new_final = go(blk.final_expr, local_env) if blk.final_expr is not None else None
        return A.Block(span=blk.span, stmts=new_stmts, final_expr=new_final)

    return go(expr, subs)


def _is_reassigned_after(stmts: list, name: str, start_idx: int) -> bool:
    """Return True if any statement after `start_idx` reassigns `name` via
    A.Assign anywhere in its expression tree. Used by `_inline_lets` to
    decide whether a `let mut` is effectively single-assignment (safe to
    inline) or genuinely mutable (must be left alone)."""
    def _has_assign(node) -> bool:
        if node is None:
            return False
        if isinstance(node, A.Assign):
            if isinstance(node.target, A.Name) and node.target.name == name:
                return True
            return _has_assign(node.value)
        # Recurse into common containers
        for attr in ("operand", "left", "right", "value", "expr",
                     "scrutinee", "cond", "iter_expr", "callee", "obj"):
            if hasattr(node, attr) and _has_assign(getattr(node, attr)):
                return True
        for attr in ("args", "elems", "stmts", "indices", "alts",
                     "sub_patterns"):
            if hasattr(node, attr):
                for child in getattr(node, attr) or []:
                    if _has_assign(child):
                        return True
        if isinstance(node, A.Block):
            for s in node.stmts:
                if _has_assign(s):
                    return True
            if node.final_expr is not None and _has_assign(node.final_expr):
                return True
        # Cycle 1 Batch FE re-audit Auditor 3 (fresh-eyes) MEDIUM-3
        # fix: explicitly recurse into For/While/Loop bodies. Pre-fix,
        # the attribute-name list at line ~1078 doesn't include `body`,
        # so a For/While/Loop with an Assign nested inside its body
        # would NOT be detected as reassigning. _inline_lets could
        # then incorrectly inline a `let mut acc = 0` that gets
        # reassigned inside a nested loop, producing wrong gradients.
        # Block-recursion below catches the body if it's reached via
        # an explicit body-field walk; this loop-arm makes that walk
        # happen.
        if isinstance(node, (A.For, A.While, A.Loop)):
            if _has_assign(node.body):
                return True
        if isinstance(node, A.If):
            if _has_assign(node.then) or _has_assign(node.else_):
                return True
        if isinstance(node, A.Match):
            for arm in node.arms:
                if _has_assign(arm.body):
                    return True
        return False
    for i in range(start_idx + 1, len(stmts)):
        if _has_assign(stmts[i]):
            return True
    return False


def _is_ad_erasable_expr(expr: A.Expr | None) -> bool:
    """True when dropping or inlining an expression cannot hide effects.

    `_inline_lets` erases let statements whose names are not part of the final
    differentiated expression. That is only sound for expressions made from
    literals, names, arithmetic, conditionals, blocks, and AD-known pure
    builtins. Unknown calls and allocator-style helpers must survive as errors,
    not disappear before the differentiator sees them.
    """
    if expr is None:
        return True
    if isinstance(expr, (A.IntLit, A.FloatLit, A.BoolLit, A.StrLit, A.CharLit)):
        return True
    if isinstance(expr, (A.Name, A.Path, A.Continue, A.TileLit)):
        return True
    if isinstance(expr, A.Unary):
        return _is_ad_erasable_expr(expr.operand)
    if isinstance(expr, A.Binary):
        return (_is_ad_erasable_expr(expr.left)
                and _is_ad_erasable_expr(expr.right))
    if isinstance(expr, A.Cast):
        return _is_ad_erasable_expr(expr.value)
    if isinstance(expr, A.Call):
        if not isinstance(expr.callee, A.Name):
            return False
        if expr.callee.name not in AD_KNOWN_PURE_CALLS:
            return False
        return all(_is_ad_erasable_expr(a) for a in expr.args)
    if isinstance(expr, A.If):
        return (_is_ad_erasable_expr(expr.cond)
                and _is_ad_erasable_expr(expr.then)
                and _is_ad_erasable_expr(expr.else_))
    if isinstance(expr, A.Match):
        if not _is_ad_erasable_expr(expr.scrutinee):
            return False
        for arm in expr.arms:
            if not _is_ad_erasable_expr(arm.guard):
                return False
            if not _is_ad_erasable_expr(arm.body):
                return False
        return True
    if isinstance(expr, A.ArrayLit):
        return all(_is_ad_erasable_expr(e) for e in expr.elems)
    if isinstance(expr, A.TupleLit):
        return all(_is_ad_erasable_expr(e) for e in expr.elems)
    if isinstance(expr, A.StructLit):
        return all(_is_ad_erasable_expr(v) for _, v in expr.fields)
    if isinstance(expr, A.Range):
        return (_is_ad_erasable_expr(expr.start)
                and _is_ad_erasable_expr(expr.end))
    if isinstance(expr, A.Field):
        return _is_ad_erasable_expr(expr.obj)
    if isinstance(expr, A.Index):
        return (_is_ad_erasable_expr(expr.callee)
                and all(_is_ad_erasable_expr(i) for i in expr.indices))
    if isinstance(expr, A.Block):
        for stmt in expr.stmts:
            if isinstance(stmt, A.Let):
                if not _is_ad_erasable_expr(stmt.value):
                    return False
            elif isinstance(stmt, A.ConstStmt):
                if not _is_ad_erasable_expr(stmt.value):
                    return False
            elif isinstance(stmt, A.ExprStmt):
                if not _is_ad_erasable_expr(stmt.expr):
                    return False
            else:
                return False
        return _is_ad_erasable_expr(expr.final_expr)
    return False


def _find_opaque_call_name(expr: A.Expr | None) -> str | None:
    """Find the first un-AD-erasable Call(Name) in `expr`. Returns the
    callee name when found (so error messages can identify the
    specific opaque call), else None. Used to give better error
    messages than the generic "side-effecting block" text when the
    real issue is an opaque user/extern call."""
    if expr is None:
        return None
    if isinstance(expr, A.Call) and isinstance(expr.callee, A.Name):
        if expr.callee.name not in AD_KNOWN_PURE_CALLS:
            return expr.callee.name
    # Recurse into common containers
    for attr in ("value", "expr", "left", "right", "operand",
                 "cond", "then", "else_", "final_expr"):
        sub = getattr(expr, attr, None)
        if isinstance(sub, A.Expr):
            name = _find_opaque_call_name(sub)
            if name is not None:
                return name
    stmts = getattr(expr, "stmts", None)
    if isinstance(stmts, list):
        for s in stmts:
            v = getattr(s, "value", None)
            if v is None:
                v = getattr(s, "expr", None)
            if isinstance(v, A.Expr):
                name = _find_opaque_call_name(v)
                if name is not None:
                    return name
    return None


def _raise_if_ad_erases_effect(expr: A.Expr | None, context: str,
                                 mode: str = "forward") -> None:
    if not _is_ad_erasable_expr(expr):
        # If the offending shape is just an unrecognized Call, give a
        # better error mentioning the call name + the AD mode. The
        # tests `test_grad_rejects_opaque_call_in_loss` (forward) +
        # `test_grad_rev_rejects_opaque_call_in_loss` (reverse) pin
        # this format: `{mode}-mode AD ... {call_name}`.
        opaque_name = _find_opaque_call_name(expr)
        if opaque_name is not None:
            raise NotImplementedError(
                f"{mode}-mode AD does not support opaque call "
                f"{opaque_name!r}; add a chain rule or inline a "
                f"differentiable helper"
            )
        raise NotImplementedError(
            f"{mode}-mode AD cannot erase side-effecting {context}; "
            "move allocation/effects outside the differentiated function"
        )


def _inline_lets(expr: A.Expr | None, env: dict[str, A.Expr],
                  mode: str = "forward") -> A.Expr | None:
    """Walk expr, replacing references to let-bound names with the bound
    expression. Used to flatten blocks before differentiation.

    `mode` ("forward" | "reverse") drives the error-message label
    in `_raise_if_ad_erases_effect` so the
    test_grad_rev_rejects_opaque_call_in_loss + forward sibling get
    the mode-specific error they expect."""
    if expr is None:
        return None
    if isinstance(expr, (A.IntLit, A.FloatLit, A.BoolLit, A.StrLit, A.CharLit)):
        return expr
    if isinstance(expr, A.Name):
        if expr.name in env:
            return env[expr.name]
        return expr
    if isinstance(expr, A.Unary):
        return A.Unary(span=expr.span, op=expr.op,
                       operand=_inline_lets(expr.operand, env))
    if isinstance(expr, A.Binary):
        return A.Binary(span=expr.span, op=expr.op,
                        left=_inline_lets(expr.left, env),
                        right=_inline_lets(expr.right, env))
    if isinstance(expr, A.Block):
        local_env = dict(env)
        for stmt in expr.stmts:
            # Inline `let` and `let mut` bindings whose name is never
            # reassigned within the rest of the block. The conservative
            # mut-skip from audit-10 was over-restrictive: it produced
            # gradient 0 for `let mut acc = x; acc` (no reassignment),
            # because `acc` had no inlining and the differentiator
            # treated the bare Name as a non-var. Cycle-4 fix: only
            # skip mutable lets that ARE actually reassigned later in
            # the same block. Single-assignment mut bindings are
            # functionally pure for AD purposes.
            if (isinstance(stmt, A.Let) and stmt.value is not None
                    and (not stmt.is_mut
                         or not _is_reassigned_after(expr.stmts, stmt.name, expr.stmts.index(stmt)))):
                _raise_if_ad_erases_effect(stmt.value, f"let {stmt.name!r}", mode)
                local_env[stmt.name] = _inline_lets(stmt.value, local_env, mode)
            # ExprStmt: ignore (no derivative meaning)
            # ConstStmt: similar to Let (immutable by construction)
            elif isinstance(stmt, A.ConstStmt):
                _raise_if_ad_erases_effect(stmt.value, f"const {stmt.name!r}", mode)
                local_env[stmt.name] = _inline_lets(stmt.value, local_env, mode)
            elif isinstance(stmt, A.ExprStmt):
                _raise_if_ad_erases_effect(stmt.expr, "expression statement", mode)
        if expr.final_expr is not None:
            _raise_if_ad_erases_effect(
                expr.final_expr, "block final expression", mode)
            return _inline_lets(expr.final_expr, local_env, mode)
        # Audit 28.8 cycle 2 (deferred observation #18): pre-fix this
        # returned FloatLit(0.0) silently when a Block had stmts but no
        # final expression. The let-stmts were inlined into env (so any
        # later use of bound names would resolve) but the Block's value
        # defaulted to 0 with no diagnostic. Now we WARN so the user
        # can spot the missing tail expression in an AD context.
        _ad_warn(
            expr,
            "empty block in AD context: no final expression — "
            "assumed 0",
        )
        return A.FloatLit(span=expr.span, value=0.0)
    if isinstance(expr, A.If):
        # Inline both branches and re-wrap in an If — the inliner only flattens
        # let-bindings, branch selection stays a runtime decision. Differentiate
        # both branches; the derivative is then the same conditional.
        # Audit 28.8 cycle 4 C4-3: also inline `expr.cond`. Pre-fix the
        # cond was passed through unmodified, so `let g = grad(loss);
        # if g(x) > 0.0 { ... }` left `g` unsubstituted in the cond.
        # Symmetric with While/For/Match coverage in cycle 3.
        new_cond = _inline_lets(expr.cond, env)
        new_then = _inline_lets(expr.then, env) if isinstance(expr.then, A.Block) else expr.then
        new_else = None
        if expr.else_ is not None:
            if isinstance(expr.else_, A.Block):
                new_else = _inline_lets(expr.else_, env)
            else:
                new_else = _inline_lets(expr.else_, env)
        # Wrap any non-block result in a Block(final_expr=) so If's children
        # are valid. The inliner returns expressions, not blocks.
        def _wrap(e: A.Expr | None) -> A.Block | None:
            if e is None:
                return None
            if isinstance(e, A.Block):
                return e
            return A.Block(span=e.span, stmts=[], final_expr=e)
        wrapped_then = _wrap(new_then)
        wrapped_else = _wrap(new_else)
        return A.If(span=expr.span, cond=new_cond,
                    then=wrapped_then, else_=wrapped_else)
    # Audit 28.8 cycle 3 C3-5: extend _inline_lets to recurse through every
    # Expr subtype that can contain Name leaves. Pre-fix the function fell
    # through to `return expr` for Cast / Call / Field / Index / Match /
    # ArrayLit / TupleLit / StructLit / Range / Return / Break / Assign /
    # UnsafeBlock / Loop / For / While / Quote / Splice / Modify — so
    # any let-bound name appearing under those positions was never
    # substituted, defeating the reverse-mode `_ad_warn` reach (C2-3).
    if isinstance(expr, A.Cast):
        return A.Cast(span=expr.span,
                      value=_inline_lets(expr.value, env),
                      target_ty=expr.target_ty)
    if isinstance(expr, A.Call):
        new_args = [_inline_lets(a, env) for a in expr.args]
        # Callee is generally a Name or Path — Name lookups in env would
        # turn it into a different expression entirely, which breaks
        # ordinary calls. Only substitute when the resolved value is
        # itself a Name/Path (alias-of-callee).
        new_callee = expr.callee
        if isinstance(expr.callee, A.Name) and expr.callee.name in env:
            cand = env[expr.callee.name]
            # Audit 28.8 cycle 4 E6: preserve the original callee's
            # generics list (turbofish) when aliasing. Pre-fix, `let g
            # = mk_grad; g::<f64>(x)` aliased to `mk_grad(x)` and
            # dropped the `::<f64>` annotation, defeating monomorphization.
            if isinstance(cand, A.Name):
                new_callee = A.Name(
                    span=cand.span, name=cand.name,
                    generics=(list(expr.callee.generics)
                              if expr.callee.generics
                              else list(cand.generics)),
                )
            elif isinstance(cand, A.Path):
                new_callee = cand
        # Audit 28.8 cycle 4 E8: walk Field-typed callees so
        # `obj.method()` with `obj` let-bound substitutes properly.
        elif isinstance(expr.callee, A.Field):
            new_callee = _inline_lets(expr.callee, env)
        return A.Call(span=expr.span, callee=new_callee, args=new_args)
    if isinstance(expr, A.Field):
        return A.Field(span=expr.span,
                       obj=_inline_lets(expr.obj, env),
                       name=expr.name)
    if isinstance(expr, A.Index):
        return A.Index(
            span=expr.span,
            callee=_inline_lets(expr.callee, env),
            indices=[_inline_lets(i, env) for i in expr.indices],
        )
    if isinstance(expr, A.ArrayLit):
        return A.ArrayLit(
            span=expr.span,
            elems=[_inline_lets(e, env) for e in expr.elems],
        )
    if isinstance(expr, A.TupleLit):
        return A.TupleLit(
            span=expr.span,
            elems=[_inline_lets(e, env) for e in expr.elems],
        )
    if isinstance(expr, A.StructLit):
        return A.StructLit(
            span=expr.span,
            name=expr.name,
            fields=[(n, _inline_lets(v, env)) for (n, v) in expr.fields],
        )
    if isinstance(expr, A.Range):
        return A.Range(
            span=expr.span,
            start=_inline_lets(expr.start, env),
            end=_inline_lets(expr.end, env),
        )
    if isinstance(expr, A.Return):
        return A.Return(
            span=expr.span,
            value=_inline_lets(expr.value, env),
        )
    if isinstance(expr, A.Break):
        return A.Break(
            span=expr.span,
            value=_inline_lets(expr.value, env),
        )
    if isinstance(expr, A.Assign):
        return A.Assign(
            span=expr.span,
            target=_inline_lets(expr.target, env),
            op=expr.op,
            value=_inline_lets(expr.value, env),
        )
    if isinstance(expr, A.UnsafeBlock):
        body = expr.body
        new_body = _inline_lets(body, env) if isinstance(body, A.Block) else body
        if not isinstance(new_body, A.Block):
            new_body = A.Block(span=body.span, stmts=[], final_expr=new_body)
        return A.UnsafeBlock(span=expr.span, body=new_body)
    if isinstance(expr, A.Match):
        new_arms = []
        for arm in expr.arms:
            new_arms.append(A.MatchArm(
                span=arm.span,
                pattern=arm.pattern,
                guard=(_inline_lets(arm.guard, env)
                       if arm.guard is not None else None),
                body=_inline_lets(arm.body, env),
            ))
        return A.Match(
            span=expr.span,
            scrutinee=_inline_lets(expr.scrutinee, env),
            arms=new_arms,
        )
    if isinstance(expr, A.Loop):
        body = expr.body
        new_body = _inline_lets(body, env) if isinstance(body, A.Block) else body
        if not isinstance(new_body, A.Block):
            new_body = A.Block(span=body.span, stmts=[], final_expr=new_body)
        return A.Loop(span=expr.span, body=new_body)
    if isinstance(expr, A.For):
        body = expr.body
        new_body = _inline_lets(body, env) if isinstance(body, A.Block) else body
        if not isinstance(new_body, A.Block):
            new_body = A.Block(span=body.span, stmts=[], final_expr=new_body)
        return A.For(
            span=expr.span,
            var_name=expr.var_name,
            iter_expr=_inline_lets(expr.iter_expr, env),
            body=new_body,
        )
    if isinstance(expr, A.While):
        body = expr.body
        new_body = _inline_lets(body, env) if isinstance(body, A.Block) else body
        if not isinstance(new_body, A.Block):
            new_body = A.Block(span=body.span, stmts=[], final_expr=new_body)
        return A.While(
            span=expr.span,
            cond=_inline_lets(expr.cond, env),
            body=new_body,
        )
    if isinstance(expr, A.Quote):
        return A.Quote(span=expr.span, inner=_inline_lets(expr.inner, env))
    if isinstance(expr, A.Splice):
        return A.Splice(span=expr.span, inner=_inline_lets(expr.inner, env))
    if isinstance(expr, A.Modify):
        return A.Modify(
            span=expr.span,
            target=_inline_lets(expr.target, env),
            transformation=_inline_lets(expr.transformation, env),
            verifier=_inline_lets(expr.verifier, env),
        )
    # Audit 28.8 cycle 4 C4-1: leaf-like exprs that hold no let-bindable
    # children — Path (qualified name like `Maybe::None`), Continue
    # (statement-expr with no children), TileLit (compile-time shape +
    # init marker). Pre-fix the catch-all fired spurious 85001 warnings
    # for any enum-variant reference in a differentiated fn body —
    # `-Wad=error` then failed to compile legitimate AD code.
    if isinstance(expr, A.Path):
        return expr
    if isinstance(expr, A.Continue):
        return expr
    if isinstance(expr, A.TileLit):
        # Audit 28.8 cycle 6 C5-3 / F4: TileLit has Expr children
        # (shape: list[Expr], memspace: Expr). The cycle-4 identity arm
        # dropped let-bound names appearing in those positions. Walk
        # children so `let N = 4; tile<f32, [N], REG>::zeros()` (the
        # legitimate user idiom) substitutes correctly.
        return A.TileLit(
            span=expr.span,
            dtype=expr.dtype,
            shape=[_inline_lets(s, env) for s in expr.shape],
            memspace=_inline_lets(expr.memspace, env),
            init=expr.init,
        )
    # Catch-all fallthrough: warn loud so future AST extensions surface
    # immediately rather than silently dropping let-bindings. Only fires
    # when an Expr subtype is genuinely unhandled (not just a no-op
    # leaf — those are explicit arms above).
    #
    # Cycle 4 C4-3: do NOT pre-embed the trap id in the reason — _ad_warn
    # appends `(trap {TRAP_AD_ASSUMED_ZERO})` to every message. Pre-fix,
    # the rendered warning contained `(trap 85001)` twice.
    _ad_warn(
        expr,
        f"_inline_lets fell through on Expr subtype "
        f"'{type(expr).__name__}' — let-bindings beyond this point may "
        f"not be substituted",
    )
    return expr


# ============================================================================
# Differentiation rules
# ============================================================================
def _diff(expr: A.Expr, var: str) -> A.Expr:
    """Recursively compute the derivative AST."""
    span = expr.span
    if isinstance(expr, A.IntLit):
        return A.IntLit(span=span, value=0)
    if isinstance(expr, A.FloatLit):
        return A.FloatLit(span=span, value=0.0)
    if isinstance(expr, A.BoolLit):
        return A.IntLit(span=span, value=0)
    if isinstance(expr, A.Name):
        if expr.name == var:
            return A.FloatLit(span=span, value=1.0)
        return A.FloatLit(span=span, value=0.0)
    if isinstance(expr, A.Unary) and expr.op == "-":
        # d(-a)/dx = -da/dx
        return A.Unary(span=span, op="-", operand=_diff(expr.operand, var))
    if isinstance(expr, A.Binary):
        l = expr.left
        r = expr.right
        dl = _diff(l, var)
        dr = _diff(r, var)
        if expr.op == "+":
            # d(a+b)/dx = da/dx + db/dx
            return A.Binary(span=span, op="+", left=dl, right=dr)
        if expr.op == "-":
            return A.Binary(span=span, op="-", left=dl, right=dr)
        if expr.op == "*":
            # Product rule: d(a*b)/dx = (da/dx)*b + a*(db/dx)
            term1 = A.Binary(span=span, op="*", left=dl, right=r)
            term2 = A.Binary(span=span, op="*", left=l, right=dr)
            return A.Binary(span=span, op="+", left=term1, right=term2)
        if expr.op == "/":
            # Quotient rule: d(a/b)/dx = (da*b - a*db) / (b*b)
            num1 = A.Binary(span=span, op="*", left=dl, right=r)
            num2 = A.Binary(span=span, op="*", left=l, right=dr)
            num = A.Binary(span=span, op="-", left=num1, right=num2)
            denom = A.Binary(span=span, op="*", left=r, right=r)
            return A.Binary(span=span, op="/", left=num, right=denom)
    if isinstance(expr, A.If):
        # d/dx (if c then a else b) = if c then da/dx else db/dx.
        # Cond contributes nothing — it's a discrete choice, not differentiable.
        d_then = _diff_block_or_expr(expr.then, var, span)
        d_else = (_diff_block_or_expr(expr.else_, var, span)
                  if expr.else_ is not None
                  else A.Block(span=span, stmts=[], final_expr=A.FloatLit(span=span, value=0.0)))
        return A.If(span=span, cond=expr.cond, then=d_then, else_=d_else)
    if isinstance(expr, A.Block):
        return _diff_block_or_expr(expr, var, span)
    if isinstance(expr, A.Call):
        # Chain rule for known transcendentals: d(f(u))/dx = f'(u) * du/dx.
        # The call is rewritten so the derivative goes through the same
        # named function whose derivative is hardcoded here.
        deriv = _diff_call_chain_rule(expr, var, span)
        if deriv is not None:
            return deriv
        # Audit 28.8 B5 / Stage 35: unknown call sites fail closed.
        # Stage 35: this branch now raises instead of returning a zero
        # derivative, so unsupported gradients cannot hide behind warnings.
        callee = getattr(expr.callee, "name", "<?>")
        raise NotImplementedError(
            f"forward-mode AD does not support opaque call {callee!r}; "
            "add a chain rule or inline a differentiable helper"
        )
    # Audit 28.8 B5: Cast arm. Numeric `x as f64` propagates the
    # derivative through (chain-rule factor is 1 for numeric widening).
    # Non-numeric Cast (e.g., `x as *T`) returns 0 with a warning.
    if isinstance(expr, A.Cast):
        tgt = expr.target_ty
        # Audit 28.8 cycle 2 B:C9: shared NUMERIC_FOR_AD set covers
        # bool/char/fp8/mxfp4/nvfp4 too.
        if isinstance(tgt, A.TyName) and tgt.name in NUMERIC_FOR_AD:
            # Inner derivative carries through.
            return _diff(expr.value, var)
        _ad_warn(expr, f"cast to non-numeric target "
                       f"{type(tgt).__name__}")
        return A.FloatLit(span=span, value=0.0)
    # Audit 28.8 B5: Quote/Splice/Modify/UnsafeBlock fall here and were
    # previously silently zeroed. Now we WARN. UnsafeBlock specifically
    # should propagate AD through its body — handle that case.
    if isinstance(expr, A.UnsafeBlock):
        body = expr.body
        if isinstance(body, A.Block):
            return _diff_block_or_expr(body, var, span)
        return _diff(body, var)
    if isinstance(expr, (A.Quote, A.Splice, A.Modify)):
        _ad_warn(expr, f"{type(expr).__name__} is not differentiable")
        return A.FloatLit(span=span, value=0.0)
    # Stage 54 post-close CRITICAL-2: explicit fail-loud arms for
    # AST kinds that Inc 3a's loop-body descent now feeds into
    # _diff. Pre-Inc-3a these were unreachable (helper-call inside
    # a loop body stayed opaque and hit the loud
    # NotImplementedError above for opaque calls). Post-Inc-3a the
    # helper is inlined into the loop body, and the For/While/Loop
    # node flows straight into _diff. Without explicit arms it
    # falls through to the warn-and-zero catchall below — that
    # warning is a soft trap suppressed unless `-Wad=error`, so
    # the user gets a silent-zero derivative on any loop-containing
    # differentiable expression. Raise loudly, mirroring the
    # Stage 35 opaque-call discipline and the reverse-mode
    # _propagate arms landed at gate-5 / 9424133.
    if isinstance(expr, (A.For, A.While, A.Loop)):
        kind = type(expr).__name__
        raise NotImplementedError(
            f"forward-mode AD does not differentiate through {kind} "
            f"bodies; unroll the loop or move the gradient-bearing "
            f"computation outside"
        )
    if isinstance(expr, A.Match):
        raise NotImplementedError(
            "forward-mode AD does not differentiate through Match "
            "expressions; rewrite as If/else over the discriminant "
            "or inline a differentiable helper"
        )
    if isinstance(expr, (A.Assign, A.Return, A.Break, A.Continue)):
        kind = type(expr).__name__
        raise NotImplementedError(
            f"forward-mode AD does not differentiate through {kind} "
            f"statements; these are control flow, not differentiable "
            f"expressions"
        )
    if isinstance(expr, A.Range):
        raise NotImplementedError(
            "forward-mode AD does not differentiate through Range "
            "expressions; ranges are iterators, not numeric values"
        )
    # v2.x re-audit R3 (FE-N1): aggregate construction / element access.
    # The empty TupleLit `()` is Helix's unit value — match_lower emits
    # it as the unreachable exhaustiveness-fallthrough arm of a lowered
    # `match`, and unit carries no numeric value, so its derivative is
    # structurally 0 (handled here like a literal). Non-empty aggregates
    # and element/field access (Index / Field / StructLit / non-empty
    # TupleLit / ArrayLit) had no arm and fell through to the
    # warn-and-zero catchall below — `_ad_warn` is a soft trap (85001)
    # suppressed unless `-Wad=error`, so a function depending on its
    # variable through one silently got a zero derivative with no
    # diagnostic. The autodiff_cli `differentiate` path has no
    # `_reject_unsupported_grad_signature` gate, so this is
    # user-reachable. Fail loudly for those, mirroring the
    # For/While/Loop arms — real aggregate derivatives are a v3.0+
    # language feature, not an audit-fix.
    if isinstance(expr, A.TupleLit) and not expr.elems:
        return A.FloatLit(span=span, value=0.0)
    if isinstance(expr, (A.Index, A.Field, A.StructLit, A.TupleLit,
                         A.ArrayLit)):
        kind = type(expr).__name__
        raise NotImplementedError(
            f"forward-mode AD does not differentiate through {kind} "
            f"(aggregate construction / element access); Phase-0 AD is "
            f"scalar-valued — extract the scalar component before "
            f"differentiating"
        )
    # Genuinely-unknown — warn loudly.
    _ad_warn(expr, "unhandled expression kind")
    return A.FloatLit(span=expr.span, value=0.0)


def _name_appears_in(expr: A.Expr, name: str) -> bool:
    """Cheap syntactic check: does `Name(name)` appear anywhere
    in `expr`? Strict over-approximation is fine (false
    positives only cause a noisy warn, not bad codegen).
    Used by Stage 54 gate-2 MEDIUM-5 to detect when __clamp's
    lo/hi positions reference the differentiation variable."""
    if expr is None:
        return False
    if isinstance(expr, A.Name):
        return expr.name == name
    # Recurse into common containers
    for attr in ("operand", "left", "right", "value", "expr",
                 "scrutinee", "cond", "iter_expr", "callee", "obj",
                 "target", "then", "else_", "body", "final_expr",
                 "start", "end"):
        sub = getattr(expr, attr, None)
        if sub is not None and isinstance(sub, A.Expr):
            if _name_appears_in(sub, name):
                return True
    for attr in ("args", "elems", "indices"):
        sublist = getattr(expr, attr, None)
        if isinstance(sublist, list):
            for s in sublist:
                if isinstance(s, A.Expr) and _name_appears_in(s, name):
                    return True
    # Stage 54 gate-3 HIGH-1 fix: StructLit.fields is a special
    # shape (list[tuple[str, Expr]]) — pre-fix, the generic list
    # walker iterated tuples and isinstance(tuple, A.Expr) was
    # False so all fields were silently skipped. The
    # `__clamp(x, Point{x: w}.x, 1.0)` case evaded the MEDIUM-5
    # warn from `_stage54_clamp_chain_rule`.
    if isinstance(expr, A.StructLit):
        for (_field_name, v) in expr.fields:
            if isinstance(v, A.Expr) and _name_appears_in(v, name):
                return True
    # Stage 54 gate-3 HIGH-2 fix: Match.arms is list[MatchArm]
    # (not Expr), so the generic list walker skipped it. Same
    # silent-evade-warn defect class as StructLit.fields above.
    if isinstance(expr, A.Match):
        for arm in expr.arms:
            if (arm.guard is not None
                    and _name_appears_in(arm.guard, name)):
                return True
            if (isinstance(arm.body, A.Expr)
                    and _name_appears_in(arm.body, name)):
                return True
    # Stage 54 gate-4 HIGH-1 fix: Modify/Quote/Splice/TileLit
    # children. Modify has transformation+verifier (not in
    # generic attr list); Quote/Splice have `inner` (not in
    # list); TileLit has `shape` (list[Expr], not in
    # args/elems/indices) and `memspace` (Expr). Same silent-
    # evade-warn defect class as the StructLit/Match arms above.
    if isinstance(expr, A.Modify):
        if (_name_appears_in(expr.transformation, name)
                or _name_appears_in(expr.verifier, name)):
            return True
    if isinstance(expr, (A.Quote, A.Splice)):
        if _name_appears_in(expr.inner, name):
            return True
    if isinstance(expr, A.TileLit):
        for s in expr.shape:
            if isinstance(s, A.Expr) and _name_appears_in(s, name):
                return True
        if _name_appears_in(expr.memspace, name):
            return True
    # Block stmts: walk Let.value, ConstStmt.value, ExprStmt.expr.
    # Stage 54 gate-4 MEDIUM-2 hardening: use independent gets
    # rather than `or`-short-circuit to be robust against any
    # future Expr subclass that overrides __bool__ (today all
    # dataclasses are truthy, so the bug is latent but the
    # pattern was fragile).
    stmts = getattr(expr, "stmts", None)
    if isinstance(stmts, list):
        for s in stmts:
            for stmt_attr in ("value", "expr"):
                v = getattr(s, stmt_attr, None)
                if (v is not None and isinstance(v, A.Expr)
                        and _name_appears_in(v, name)):
                    return True
    return False


def _stage54_min_max_chain_rule(
    call: "A.Call", var: str, span, name: str,
) -> "A.Expr":
    """Stage 54 Inc 1: 2-arg chain rule for __min/__max + _f64
    variants. Subgradient at equality is asymmetric and the
    asymmetry POINTS IN OPPOSITE DIRECTIONS for min vs max
    (the asymmetric `<=`/`>=` vs strict `<`/`>` makes the
    choice deterministic and forward-reverse symmetric within
    each operator):
      - __min attributes the equality-case gradient to the
        FIRST (left) arg via `a <= b`.
      - __max attributes the equality-case gradient to the
        SECOND (right) arg via `b >= a`.
    Both choices are valid 1-sided subgradient picks; the
    operator-pair asymmetry exists to keep each indicator's
    LHS bound to that arg, which forward/reverse mirroring
    relies on.

    __min(a, b): df/da = 1 if a <= b else 0; df/db = 1 if b < a else 0
    __max(a, b): df/da = 1 if a >  b else 0; df/db = 1 if b >= a else 0

    At a==b: __min gives da=1, db=0; __max gives da=0, db=1.
    (Stage 54 gate-1 Finding 4 corrected "picks 0" → "picks
    lexically-first"; post-close HIGH-1 corrected the still-
    misleading "lexically-first" claim — that's only true for
    __min, not __max.)

    Returns adj_a * da/dvar + adj_b * db/dvar with the two
    indicators inlined as A.If expressions.
    """
    if len(call.args) != 2:
        # Wrong arity — fall through to caller's None path so the
        # downstream type checker emits the right error rather than
        # AD producing nonsense.
        return None
    a = call.args[0]
    b = call.args[1]
    # Stage 54 gate-4 MEDIUM-3 consistency-with-clamp fix: warn
    # when BOTH args depend on `var` — the user is differentiating
    # through a kink at the equality point, where the indicator-
    # based subgradient is mathematically valid but the user
    # gets no diagnostic that gradients can be discontinuous
    # near a == b. Mirrors clamp's MEDIUM-5 lo/hi warn.
    if _name_appears_in(a, var) and _name_appears_in(b, var):
        _ad_warn(
            call,
            f"{name} with both args depending on '{var}' — "
            f"subgradient is defined via lexically-first "
            f"convention but gradients are discontinuous at "
            f"a == b. Confirm this is the intended behavior.",
        )
    da = _diff(a, var)
    db = _diff(b, var)
    suffix = "f64" if name.endswith("_f64") else None

    def flit(v: float) -> "A.FloatLit":
        return A.FloatLit(span=span, value=v, type_suffix=suffix)

    def indicator(left_op: str, left_a: "A.Expr",
                  right_b: "A.Expr") -> "A.If":
        """if left_a OP right_b { 1.0 } else { 0.0 }"""
        return A.If(
            span=span,
            cond=A.Binary(span=span, op=left_op,
                          left=_copy.deepcopy(left_a),
                          right=_copy.deepcopy(right_b)),
            then=A.Block(span=span, stmts=[], final_expr=flit(1.0)),
            else_=A.Block(span=span, stmts=[], final_expr=flit(0.0)),
        )

    if name in ("__min", "__min_f64"):
        # df/da = 1 if a <= b else 0
        # df/db = 1 if b <  a else 0
        ind_a = indicator("<=", a, b)
        ind_b = indicator("<", b, a)
    else:  # __max, __max_f64
        # df/da = 1 if a >  b else 0
        # df/db = 1 if b >= a else 0
        ind_a = indicator(">", a, b)
        ind_b = indicator(">=", b, a)

    term_a = A.Binary(span=span, op="*", left=ind_a, right=da)
    term_b = A.Binary(span=span, op="*", left=ind_b, right=db)
    return A.Binary(span=span, op="+", left=term_a, right=term_b)


def _stage54_clamp_chain_rule(
    call: "A.Call", var: str, span, name: str,
) -> "A.Expr":
    """Stage 54 Inc 1: 3-arg chain rule for __clamp + _f64 variant.

    __clamp(x, lo, hi): df/dx = 1 if lo <= x <= hi else 0;
    lo and hi treated as non-differentiable constants (df/dlo =
    df/dhi = 0). Returns adj_x * indicator.

    Stage 54 gate-2 MEDIUM-5 fix: when `var` actually appears in
    `lo` or `hi` (e.g., `__clamp(x, weight*0.1, weight*0.9)`
    with var=weight), the dlo/dhi contributions are silently
    DROPPED — which per CLAUDE.md silent-failure ban requires
    an `_ad_warn` so the user knows their gradient is incomplete.
    """
    if len(call.args) != 3:
        return None
    x = call.args[0]
    lo = call.args[1]
    hi = call.args[2]
    dx = _diff(x, var)
    # Gate-2 MEDIUM-5: warn loudly if dlo/dhi would be nonzero
    # (var depends on lo or hi). _name_appears_in is a cheap
    # syntactic check — strict over-approximation is fine
    # (false positives only cause a noisy warn, not bad codegen).
    if _name_appears_in(lo, var) or _name_appears_in(hi, var):
        _ad_warn(
            call,
            f"__clamp dlo/dhi w.r.t. '{var}' silently dropped — "
            f"gradient is incomplete. Treat lo/hi as constants "
            f"or rewrite the expression to detach them from "
            f"the differentiation graph.",
        )
    suffix = "f64" if name.endswith("_f64") else None

    def flit(v: float) -> "A.FloatLit":
        return A.FloatLit(span=span, value=v, type_suffix=suffix)

    # lo <= x AND x <= hi
    lo_ok = A.Binary(span=span, op="<=",
                     left=_copy.deepcopy(lo),
                     right=_copy.deepcopy(x))
    hi_ok = A.Binary(span=span, op="<=",
                     left=_copy.deepcopy(x),
                     right=_copy.deepcopy(hi))
    both = A.Binary(span=span, op="&&", left=lo_ok, right=hi_ok)
    indicator = A.If(
        span=span,
        cond=both,
        then=A.Block(span=span, stmts=[], final_expr=flit(1.0)),
        else_=A.Block(span=span, stmts=[], final_expr=flit(0.0)),
    )
    return A.Binary(span=span, op="*", left=indicator, right=dx)


def _diff_call_chain_rule(call: A.Call, var: str,
                          span: A.Span) -> Optional[A.Expr]:
    """Apply the analytic derivative for known transcendental builtins.
    Returns None if the callee isn't a recognised transcendental."""
    if not isinstance(call.callee, A.Name):
        return None
    # Stage 36 Inc 12 — close Inc 11 type-design B2 MEDIUM deferral.
    # Integer-valued boolean Logic ops have no meaningful chain rule;
    # pre-fix they silently produced a zero derivative. Fail loud
    # before any chain-rule arm runs.
    if call.callee.name in AD_INTEGER_VALUED_LOGIC:
        _raise_integer_logic_in_ad(call.callee.name, "forward")
    # Handle __powi(x, n) separately: 2-arg with n literal int.
    # d(x^n)/dx = n * x^(n-1) * dx/dvar.
    if call.callee.name == "__powi" and len(call.args) == 2:
        x = call.args[0]
        n_arg = call.args[1]
        if isinstance(n_arg, A.IntLit):
            n_val = n_arg.value
            dx = _diff(x, var)
            if n_val <= 0 or n_val > 16:
                # __powi(x, n) returns constant 1.0 for n <= 0 or n > 16
                # (stdlib transcendentals.hx) — derivative is 0. Previously
                # we capped n_val to 16 here, producing a wrong gradient
                # `16 * x^15` for n > 16 even though the function itself
                # is constant at those inputs.
                return A.FloatLit(span=span, value=0.0)
            # n * __powi(x, n-1) * dx
            n_lit = A.FloatLit(span=span, value=float(n_val))
            n_minus_one = A.IntLit(span=span, value=n_val - 1)
            x_pow = A.Call(span=span,
                           callee=A.Name(span=span, name="__powi"),
                           args=[x, n_minus_one])
            return A.Binary(span=span, op="*",
                            left=A.Binary(span=span, op="*",
                                          left=n_lit, right=x_pow),
                            right=dx)
        # Non-literal n: fall through to zero derivative.
    if call.callee.name == "__bce" and len(call.args) == 2:
        p = call.args[0]
        y = call.args[1]
        dp = _diff(p, var)
        dy = _diff(y, var)

        def f(v: float) -> A.FloatLit:
            return A.FloatLit(span=span, value=v)

        def binary(op: str, a: A.Expr, b: A.Expr) -> A.Binary:
            return A.Binary(span=span, op=op, left=a, right=b)

        def calln(fn: str, args: list[A.Expr]) -> A.Call:
            return A.Call(span=span, callee=A.Name(span=span, name=fn), args=args)

        eps = f(0.000001)
        hi = f(0.999999)
        p_safe = calln("__clamp", [_copy.deepcopy(p), f(0.000001), f(0.999999)])
        one_minus_p = binary("-", f(1.0), _copy.deepcopy(p_safe))
        denom = binary("*", _copy.deepcopy(p_safe), one_minus_p)
        raw_dp = binary("/", binary("-", _copy.deepcopy(p_safe), _copy.deepcopy(y)), denom)
        cond_lo = binary("<", _copy.deepcopy(p), eps)
        cond_hi = binary(">", _copy.deepcopy(p), hi)
        zero = f(0.0)
        gated_dp_hi = A.If(
            span=span,
            cond=cond_hi,
            then=A.Block(span=span, stmts=[], final_expr=f(0.0)),
            else_=A.Block(span=span, stmts=[], final_expr=raw_dp),
        )
        deriv_p = A.If(
            span=span,
            cond=cond_lo,
            then=A.Block(span=span, stmts=[], final_expr=zero),
            else_=A.Block(span=span, stmts=[], final_expr=gated_dp_hi),
        )
        log_one_minus = calln("__log_stable", [binary("-", f(1.0), _copy.deepcopy(p_safe))])
        log_p = calln("__log_stable", [_copy.deepcopy(p_safe)])
        deriv_y = binary("-", log_one_minus, log_p)
        return binary("+", binary("*", deriv_p, dp), binary("*", deriv_y, dy))
    # Stage 36 Increment 6: 2-arg fuzzy logic operators. These must be
    # handled before the `len(call.args) != 1` early return below
    # (mirrors the __powi and __bce placement above).
    if call.callee.name == "fuzzy_and" and len(call.args) == 2:
        # d(a*b)/dx = a'*b + a*b'
        a, b = call.args
        da = _diff(a, var)
        db = _diff(b, var)
        ab_term = A.Binary(span=span, op="*", left=da,
                           right=_copy.deepcopy(b))
        ba_term = A.Binary(span=span, op="*",
                           left=_copy.deepcopy(a), right=db)
        return A.Binary(span=span, op="+", left=ab_term, right=ba_term)
    if call.callee.name == "fuzzy_or" and len(call.args) == 2:
        # d(a + b - a*b)/dx = a'*(1-b) + b'*(1-a)
        a, b = call.args
        da = _diff(a, var)
        db = _diff(b, var)
        one_minus_b = A.Binary(span=span, op="-",
                               left=A.FloatLit(span=span, value=1.0),
                               right=_copy.deepcopy(b))
        one_minus_a = A.Binary(span=span, op="-",
                               left=A.FloatLit(span=span, value=1.0),
                               right=_copy.deepcopy(a))
        return A.Binary(
            span=span, op="+",
            left=A.Binary(span=span, op="*", left=da, right=one_minus_b),
            right=A.Binary(span=span, op="*", left=db, right=one_minus_a))
    # Stage 36 Increment 8: fuzzy_xor and fuzzy_implies chain rules.
    # fuzzy_xor(a, b) = a + b - 2*a*b
    # d/da = 1 - 2*b, d/db = 1 - 2*a
    if call.callee.name == "fuzzy_xor" and len(call.args) == 2:
        a, b = call.args
        da = _diff(a, var)
        db = _diff(b, var)
        two_b = A.Binary(span=span, op="*",
                         left=A.FloatLit(span=span, value=2.0),
                         right=_copy.deepcopy(b))
        two_a = A.Binary(span=span, op="*",
                         left=A.FloatLit(span=span, value=2.0),
                         right=_copy.deepcopy(a))
        coeff_a = A.Binary(span=span, op="-",
                           left=A.FloatLit(span=span, value=1.0),
                           right=two_b)
        coeff_b = A.Binary(span=span, op="-",
                           left=A.FloatLit(span=span, value=1.0),
                           right=two_a)
        return A.Binary(
            span=span, op="+",
            left=A.Binary(span=span, op="*", left=da, right=coeff_a),
            right=A.Binary(span=span, op="*", left=db, right=coeff_b))
    # fuzzy_implies(a, b) = 1 - a + a*b
    # d/da = -1 + b, d/db = a
    if call.callee.name == "fuzzy_implies" and len(call.args) == 2:
        a, b = call.args
        da = _diff(a, var)
        db = _diff(b, var)
        coeff_a = A.Binary(span=span, op="-",
                           left=_copy.deepcopy(b),
                           right=A.FloatLit(span=span, value=1.0))
        return A.Binary(
            span=span, op="+",
            left=A.Binary(span=span, op="*", left=da, right=coeff_a),
            right=A.Binary(span=span, op="*", left=db,
                           right=_copy.deepcopy(a)))
    # prove(value, source) is a 2-arg identity wrapper. The source tag
    # is non-differentiable so the chain rule is identity on the first
    # arg.
    #
    # Stage 36 Inc 9 catch-up — type-design B3 fix: guard against a
    # differentiable source-tag expression. Pre-fix, `prove(x, x)`
    # silently returned `_diff(x, var)` — the second `x` (the source
    # tag) was dropped from the chain rule with no diagnostic. Now we
    # require the source-tag to be a literal integer; runtime-loaded
    # source IDs need to flow through `register_derivation` so the
    # autodiff path stays provably non-aliased with differentiable vars.
    if call.callee.name == "prove" and len(call.args) == 2:
        if not isinstance(call.args[1], A.IntLit):
            raise NotImplementedError(
                "autodiff: prove(value, source): source must be an "
                "integer literal in differentiated code (got "
                f"{type(call.args[1]).__name__}); use "
                "register_derivation for dynamic source tags so AD "
                "can statically see the tag is non-differentiable"
            )
        return _diff(call.args[0], var)
    # Stage 38 post-Inc-3 silent-failure F2 fix (MEDIUM): frame
    # identity wrappers — chain rule is identity on the single arg.
    # Stage 43 LOW-3 rename: now covers all 5 wrapper families
    # (frame + temporal + modal + causal); set name updated.
    if (call.callee.name in _IDENTITY_AD_CHAIN_RULE_NAMES
            and len(call.args) == 1):
        return _diff(call.args[0], var)
    # Stage 54 Inc 1: chain-rule arms for __min/__max/__clamp/__sign
    # (11 names: 4 base + _i32 + _f64 variants). Previously these
    # were in AD_KNOWN_PURE_CALLS (let-erasable) but hit the
    # opaque-call catchall in the chain-rule dispatch — returning
    # zero+warn in forward mode, raising NotImplementedError in
    # reverse. Now wired with proper subgradient semantics:
    # - __min(a, b): adj_a * 1[a<=b] + adj_b * 1[b<a]
    # - __max(a, b): adj_a * 1[a>b] + adj_b * 1[b>=a]
    # - __clamp(x, lo, hi): adj_x * 1[lo<=x AND x<=hi]
    #   (lo/hi treated as non-differentiable constants)
    # - __sign(x): derivative is 0 (distributional sense)
    # _i32 variants: derivative is 0 (integer-valued, non-diff)
    name = call.callee.name
    if name in ("__min", "__min_f64", "__max", "__max_f64"):
        return _stage54_min_max_chain_rule(call, var, span, name)
    if name in ("__clamp", "__clamp_f64"):
        return _stage54_clamp_chain_rule(call, var, span, name)
    # _i32 variants of min/max/clamp + all __sign variants:
    # derivative is 0. Stage 54 gate-2 MEDIUM-4 fix: emit IntLit
    # for _i32 variants (the gradient zero must match the
    # function's return type for downstream typecheck consistency
    # when the gradient feeds into i32-arithmetic). Pre-fix used
    # FloatLit(0.0, suffix=None) which printed as "0.0" and
    # confused i32 contexts. __sign(x) returns x's type — AD
    # convention is f64 unless the source explicitly carries i32.
    if name in ("__min_i32", "__max_i32", "__clamp_i32"):
        return A.IntLit(span=span, value=0)
    if name in ("__sign", "__sign_f64"):
        # Stage 54 post-close type-design HIGH-1 fix: bare __sign
        # is declared `f32 -> f32` in stdlib/transcendentals.hx
        # :365, so its derivative-zero literal must carry an f32
        # type (suffix=None, the f32 default in helixc). Only the
        # _f64 variant gets type_suffix="f64". Pre-fix both
        # variants stamped f64, which would coerce-or-mismatch
        # downstream f32 arithmetic contexts.
        suffix = "f64" if name.endswith("_f64") else None
        return A.FloatLit(span=span, value=0.0, type_suffix=suffix)
    if len(call.args) != 1:
        return None
    u = call.args[0]
    du = _diff(u, var)

    def mul(a: A.Expr, b: A.Expr) -> A.Expr:
        return A.Binary(span=span, op="*", left=a, right=b)

    def call1(fn: str, arg: A.Expr) -> A.Expr:
        return A.Call(span=span, callee=A.Name(span=span, name=fn), args=[arg])

    def flit(v: float, suffix: str | None = None) -> A.FloatLit:
        return A.FloatLit(span=span, value=v, type_suffix=suffix)

    if name == "__log_stable":
        # __log_stable returns a fixed sentinel for x <= 0, so its local
        # derivative is 0 on that branch and 1/x on the positive branch.
        cond = A.Binary(span=span, op="<=", left=_copy.deepcopy(u),
                        right=flit(0.0))
        recip = A.Binary(span=span, op="/",
                         left=flit(1.0), right=_copy.deepcopy(u))
        gated = A.If(
            span=span,
            cond=cond,
            then=A.Block(span=span, stmts=[], final_expr=flit(0.0)),
            else_=A.Block(span=span, stmts=[], final_expr=recip),
        )
        return mul(gated, du)
    if name == "__exp_f64":
        return mul(call1("__exp_f64", u), du)
    if name == "__log_f64":
        recip = A.Binary(span=span, op="/",
                         left=flit(1.0, "f64"), right=u)
        return mul(recip, du)
    if name == "__sin_f64":
        return mul(call1("__cos_f64", u), du)
    if name == "__cos_f64":
        neg_sin = A.Unary(span=span, op="-", operand=call1("__sin_f64", u))
        return mul(neg_sin, du)
    if name == "__sqrt_f64":
        sqrt_u = call1("__sqrt_f64", u)
        denom = A.Binary(span=span, op="*",
                         left=flit(2.0, "f64"), right=sqrt_u)
        recip = A.Binary(span=span, op="/",
                         left=flit(1.0, "f64"), right=denom)
        return mul(recip, du)
    if name == "__relu_f64":
        cond = A.Binary(span=span, op=">", left=u, right=flit(0.0, "f64"))
        gated = A.If(span=span, cond=cond,
                     then=A.Block(span=span, stmts=[],
                                  final_expr=flit(1.0, "f64")),
                     else_=A.Block(span=span, stmts=[],
                                   final_expr=flit(0.0, "f64")))
        return mul(gated, du)
    if name == "__sigmoid_f64":
        s1 = call1("__sigmoid_f64", _copy.deepcopy(u))
        s2 = call1("__sigmoid_f64", _copy.deepcopy(u))
        one_minus = A.Binary(span=span, op="-",
                             left=flit(1.0, "f64"), right=s1)
        return mul(mul(s2, one_minus), du)
    if name == "__abs_f64":
        u_copy = _copy.deepcopy(u)
        zero = flit(0.0, "f64")
        cond_pos = A.Binary(span=span, op=">", left=u_copy,
                            right=flit(0.0, "f64"))
        cond_neg = A.Binary(span=span, op="<", left=_copy.deepcopy(u),
                            right=flit(0.0, "f64"))
        inner_else = A.If(span=span, cond=cond_neg,
                          then=A.Block(span=span, stmts=[],
                                       final_expr=flit(-1.0, "f64")),
                          else_=A.Block(span=span, stmts=[],
                                        final_expr=zero))
        gated = A.If(span=span, cond=cond_pos,
                     then=A.Block(span=span, stmts=[],
                                  final_expr=flit(1.0, "f64")),
                     else_=A.Block(span=span, stmts=[],
                                   final_expr=inner_else))
        return mul(gated, du)

    if name == "__exp":
        # d(exp(u))/dx = exp(u) * du/dx
        return mul(call1("__exp", u), du)
    if name == "__log":
        # d(log(u))/dx = (1/u) * du/dx
        recip = A.Binary(span=span, op="/",
                         left=A.FloatLit(span=span, value=1.0), right=u)
        return mul(recip, du)
    if name == "__sin":
        # d(sin(u))/dx = cos(u) * du/dx
        return mul(call1("__cos", u), du)
    if name == "__cos":
        # d(cos(u))/dx = -sin(u) * du/dx
        neg_sin = A.Unary(span=span, op="-", operand=call1("__sin", u))
        return mul(neg_sin, du)
    if name == "__sqrt":
        # d(sqrt(u))/dx = (1 / (2*sqrt(u))) * du/dx
        sqrt_u = call1("__sqrt", u)
        denom = A.Binary(span=span, op="*",
                         left=A.FloatLit(span=span, value=2.0), right=sqrt_u)
        recip = A.Binary(span=span, op="/",
                         left=A.FloatLit(span=span, value=1.0), right=denom)
        return mul(recip, du)
    if name == "__relu":
        # d(relu(u))/dx = (1 if u > 0 else 0) * du/dx
        # IMPORTANT: cond and else_ each get their OWN FloatLit(0.0) — they
        # must not share a node, otherwise downstream in-place AST mutation
        # passes (grad_pass alias resolution) corrupt both branches at once.
        cond = A.Binary(span=span, op=">", left=u,
                        right=A.FloatLit(span=span, value=0.0))
        gated = A.If(span=span, cond=cond,
                     then=A.Block(span=span, stmts=[],
                                  final_expr=A.FloatLit(span=span, value=1.0)),
                     else_=A.Block(span=span, stmts=[],
                                   final_expr=A.FloatLit(span=span, value=0.0)))
        return mul(gated, du)
    if name == "__sigmoid":
        # d(sigmoid(u))/dx = sigmoid(u) * (1 - sigmoid(u)) * du/dx
        # The two __sigmoid(u) call nodes get DEEPCOPIES of u so the second
        # call doesn't share its argument tree with the first — protects
        # against in-place mutation by later passes.
        s1 = call1("__sigmoid", _copy.deepcopy(u))
        s2 = call1("__sigmoid", _copy.deepcopy(u))
        one_minus = A.Binary(span=span, op="-",
                             left=A.FloatLit(span=span, value=1.0), right=s1)
        return mul(mul(s2, one_minus), du)
    if name == "__tanh":
        # d(tanh(u))/dx = (1 - tanh(u)^2) * du/dx. Two distinct __tanh(u)
        # call nodes (each with deep-copied u) so neither side of the
        # square shares structure with the other — same protection used
        # by __sigmoid below to survive in-place AST mutation by
        # downstream passes.
        t1 = call1("__tanh", _copy.deepcopy(u))
        t2 = call1("__tanh", _copy.deepcopy(u))
        t_sq = A.Binary(span=span, op="*", left=t1, right=t2)
        one_minus = A.Binary(span=span, op="-",
                             left=A.FloatLit(span=span, value=1.0), right=t_sq)
        return mul(one_minus, du)
    if name == "__softplus":
        # d(softplus(u))/dx = sigmoid(u) * du/dx
        return mul(call1("__sigmoid", u), du)
    if name == "__silu":
        # d(silu(u))/dx = sigmoid(u) + u * sigmoid(u) * (1 - sigmoid(u)) * du/dx
        # = sigmoid(u) * (1 + u * (1 - sigmoid(u))) * du/dx
        s1 = call1("__sigmoid", _copy.deepcopy(u))
        s2 = call1("__sigmoid", _copy.deepcopy(u))
        one_minus_s = A.Binary(span=span, op="-",
                               left=A.FloatLit(span=span, value=1.0), right=s2)
        u_times_oms = A.Binary(span=span, op="*", left=_copy.deepcopy(u),
                               right=one_minus_s)
        inner = A.Binary(span=span, op="+",
                         left=A.FloatLit(span=span, value=1.0),
                         right=u_times_oms)
        return mul(mul(s1, inner), du)
    if name == "__gelu":
        # Tanh-approx GELU derivative:
        # 0.5*(1+tanh(inner)) + 0.5*u*(1-tanh(inner)^2)*inner'
        c = A.FloatLit(span=span, value=0.7978846)
        x2 = A.Binary(span=span, op="*",
                      left=_copy.deepcopy(u), right=_copy.deepcopy(u))
        x3 = A.Binary(span=span, op="*", left=_copy.deepcopy(x2),
                      right=_copy.deepcopy(u))
        inner_arg = A.Binary(
            span=span,
            op="+",
            left=_copy.deepcopy(u),
            right=A.Binary(span=span, op="*",
                           left=A.FloatLit(span=span, value=0.044715),
                           right=x3),
        )
        inner = A.Binary(span=span, op="*", left=c, right=inner_arg)
        t1 = call1("__tanh", _copy.deepcopy(inner))
        t2 = call1("__tanh", _copy.deepcopy(inner))
        first = A.Binary(
            span=span,
            op="*",
            left=A.FloatLit(span=span, value=0.5),
            right=A.Binary(span=span, op="+",
                           left=A.FloatLit(span=span, value=1.0),
                           right=t1),
        )
        one_minus_t2 = A.Binary(
            span=span,
            op="-",
            left=A.FloatLit(span=span, value=1.0),
            right=A.Binary(span=span, op="*", left=t2,
                           right=call1("__tanh", _copy.deepcopy(inner))),
        )
        inner_prime = A.Binary(
            span=span,
            op="*",
            left=A.FloatLit(span=span, value=0.7978846),
            right=A.Binary(
                span=span,
                op="+",
                left=A.FloatLit(span=span, value=1.0),
                right=A.Binary(span=span, op="*",
                               left=A.FloatLit(span=span, value=0.134145),
                               right=x2),
            ),
        )
        second = A.Binary(
            span=span,
            op="*",
            left=A.Binary(span=span, op="*",
                          left=A.FloatLit(span=span, value=0.5),
                          right=_copy.deepcopy(u)),
            right=A.Binary(span=span, op="*", left=one_minus_t2,
                           right=inner_prime),
        )
        return mul(A.Binary(span=span, op="+", left=first, right=second), du)
    if name == "__abs":
        # d(abs(u))/dx = sign(u) * du/dx; at u=0 use 0.
        # Implement as if u>0 then 1 else (if u<0 then -1 else 0) * du.
        u_copy = _copy.deepcopy(u)
        zero = A.FloatLit(span=span, value=0.0)
        cond_pos = A.Binary(span=span, op=">", left=u_copy,
                            right=A.FloatLit(span=span, value=0.0))
        cond_neg = A.Binary(span=span, op="<", left=_copy.deepcopy(u),
                            right=A.FloatLit(span=span, value=0.0))
        inner_else = A.If(span=span, cond=cond_neg,
                          then=A.Block(span=span, stmts=[],
                                       final_expr=A.FloatLit(span=span, value=-1.0)),
                          else_=A.Block(span=span, stmts=[], final_expr=zero))
        gated = A.If(span=span, cond=cond_pos,
                     then=A.Block(span=span, stmts=[],
                                  final_expr=A.FloatLit(span=span, value=1.0)),
                     else_=A.Block(span=span, stmts=[], final_expr=inner_else))
        return mul(gated, du)
    # Stage 36 Increment 6: forward-mode chain rules for 1-arg wrapper
    # builtins (unwrap_logic, attach, detach are identity; fuzzy_not
    # is 1 - a, derivative -a'). prove and 2-arg fuzzy_* live above
    # the `len(call.args) != 1` gate.
    if name in ("unwrap_logic", "attach", "detach"):
        return du
    if name == "fuzzy_not":
        # d(1 - a)/dx = -a'
        return A.Unary(span=span, op="-", operand=du)
    return None


def _diff_block_or_expr(node: A.Expr | A.Block, var: str, span: A.Span) -> A.Block:
    """Differentiate a Block by differentiating its final_expr; or wrap a bare
    Expr in a single-final-expr block. The result is always a Block, suitable
    for use as a then/else child of an If."""
    if isinstance(node, A.Block):
        if node.final_expr is None:
            return A.Block(span=span, stmts=[], final_expr=A.FloatLit(span=span, value=0.0))
        d = _diff(node.final_expr, var)
        return A.Block(span=node.span, stmts=[], final_expr=d)
    d = _diff(node, var)
    return A.Block(span=span, stmts=[], final_expr=d)


# ============================================================================
# Simplification — fold trivial terms (0+x, x+0, 0*x, 1*x, etc.)
# ============================================================================
def _simplify(expr: A.Expr) -> A.Expr:
    if isinstance(expr, A.Binary):
        l = _simplify(expr.left)
        r = _simplify(expr.right)
        # Fold constant arithmetic
        l_val = _const_value(l)
        r_val = _const_value(r)
        if l_val is not None and r_val is not None:
            # Audit 28.8 cycle 2 (deferred observation #19): pre-fix
            # this `except Exception: pass` swallowed every error in
            # constant folding, falling through to the unsimplified
            # expression with no diagnostic. Narrowed to the actually
            # expected arithmetic exceptions (overflow, zero-divide,
            # value, type). Anything else surfaces as a real bug.
            try:
                if expr.op == "+":
                    return _make_const(l_val + r_val, expr.span)
                if expr.op == "-":
                    return _make_const(l_val - r_val, expr.span)
                if expr.op == "*":
                    return _make_const(l_val * r_val, expr.span)
                if expr.op == "/" and r_val != 0:
                    return _make_const(l_val / r_val, expr.span)
            except (OverflowError, ZeroDivisionError, ValueError, TypeError):
                # Genuine arithmetic limits — fall through unsimplified.
                pass
        # 0 + x = x
        if expr.op == "+":
            if _is_zero(l):
                return r
            if _is_zero(r):
                return l
        # x - 0 = x;  0 - x = -x
        if expr.op == "-":
            if _is_zero(r):
                return l
            if _is_zero(l):
                return A.Unary(span=expr.span, op="-", operand=r)
        # 0 * x = 0;  x * 0 = 0;  1 * x = x;  x * 1 = x
        if expr.op == "*":
            if _is_zero(l) or _is_zero(r):
                return A.FloatLit(span=expr.span, value=0.0)
            if _is_one(l):
                return r
            if _is_one(r):
                return l
        return A.Binary(span=expr.span, op=expr.op, left=l, right=r)
    if isinstance(expr, A.Unary):
        sub = _simplify(expr.operand)
        # -(-x) = x
        if expr.op == "-" and isinstance(sub, A.Unary) and sub.op == "-":
            return sub.operand
        # -0 = 0
        if expr.op == "-" and _is_zero(sub):
            return A.FloatLit(span=expr.span, value=0.0)
        return A.Unary(span=expr.span, op=expr.op, operand=sub)
    if isinstance(expr, A.If):
        # Recursively simplify branches.
        new_then = _simplify_block(expr.then) if expr.then is not None else None
        new_else = _simplify_block(expr.else_) if expr.else_ is not None else None
        return A.If(span=expr.span, cond=expr.cond, then=new_then, else_=new_else)
    if isinstance(expr, A.Block):
        return _simplify_block(expr)
    return expr


def _simplify_block(blk: A.Block) -> A.Block:
    if blk.final_expr is None:
        return blk
    return A.Block(span=blk.span, stmts=blk.stmts,
                   final_expr=_simplify(blk.final_expr))


def _is_zero(e: A.Expr) -> bool:
    return ((isinstance(e, A.IntLit) and e.value == 0)
            or (isinstance(e, A.FloatLit) and e.value == 0.0))


def _is_one(e: A.Expr) -> bool:
    return ((isinstance(e, A.IntLit) and e.value == 1)
            or (isinstance(e, A.FloatLit) and e.value == 1.0))


def _const_value(e: A.Expr):
    if isinstance(e, A.IntLit):
        return e.value
    if isinstance(e, A.FloatLit):
        return e.value
    if isinstance(e, A.Unary) and e.op == "-":
        v = _const_value(e.operand)
        if v is not None:
            return -v
    return None


def _make_const(value, span: A.Span) -> A.Expr:
    if isinstance(value, int):
        return A.IntLit(span=span, value=value)
    return A.FloatLit(span=span, value=float(value))


# ============================================================================
# Pretty print (for testing / showing derivatives)
# ============================================================================
def fmt(expr: A.Expr) -> str:
    if isinstance(expr, A.IntLit):
        return str(expr.value)
    if isinstance(expr, A.FloatLit):
        return f"{expr.value:g}"
    if isinstance(expr, A.BoolLit):
        return "true" if expr.value else "false"
    if isinstance(expr, A.Name):
        return expr.name
    if isinstance(expr, A.Binary):
        return f"({fmt(expr.left)} {expr.op} {fmt(expr.right)})"
    if isinstance(expr, A.Unary):
        return f"({expr.op}{fmt(expr.operand)})"
    if isinstance(expr, A.Call):
        callee = fmt(expr.callee) if not isinstance(expr.callee, A.Name) else expr.callee.name
        return f"{callee}({', '.join(fmt(a) for a in expr.args)})"
    if isinstance(expr, A.Block):
        if expr.final_expr is not None and not expr.stmts:
            return fmt(expr.final_expr)
        return f"<Block>"
    if isinstance(expr, A.If):
        then_s = fmt(expr.then) if expr.then is not None else "()"
        else_s = fmt(expr.else_) if expr.else_ is not None else "()"
        return f"if {fmt(expr.cond)} {{ {then_s} }} else {{ {else_s} }}"
    if isinstance(expr, A.Match):
        arms = []
        for arm in expr.arms:
            arms.append(f"{_fmt_pattern(arm.pattern)} => {fmt(arm.body)}")
        return f"match {fmt(expr.scrutinee)} {{ {', '.join(arms)} }}"
    return f"<{type(expr).__name__}>"


def _fmt_pattern(pat: A.Pattern) -> str:
    if isinstance(pat, A.PatWildcard):
        return "_"
    if isinstance(pat, A.PatLit):
        return fmt(pat.value)
    if isinstance(pat, A.PatBind):
        prefix = "mut " if pat.is_mut else ""
        return f"{prefix}{pat.name}"
    if isinstance(pat, A.PatRange):
        return f"{fmt(pat.lo)}..{fmt(pat.hi)}"
    if isinstance(pat, A.PatOr):
        return " | ".join(_fmt_pattern(a) for a in pat.alts)
    if isinstance(pat, A.PatTuple):
        return f"({', '.join(_fmt_pattern(e) for e in pat.elems)})"
    if isinstance(pat, A.PatVariant):
        segs = pat.path.segments if hasattr(pat.path, "segments") else (
            pat.path if isinstance(pat.path, list) else [str(pat.path)]
        )
        path = "::".join(segs)
        if pat.sub_patterns:
            return f"{path}({', '.join(_fmt_pattern(s) for s in pat.sub_patterns)})"
        return path
    return f"<{type(pat).__name__}>"
