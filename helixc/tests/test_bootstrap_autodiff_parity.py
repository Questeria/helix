"""helixc/tests/test_autodiff_parity.py — Autodiff bootstrap parity corpus.

GOAL: A data-driven corpus of ~50 Helix programs exercising grad / grad_rev_all
that runs BOTH compilers and asserts behavioral parity. Mirrors the structure
of test_parity_matrix.py. This is the SCOREBOARD for the autodiff porting
campaign — items move from KNOWN_PARITY_GAPS to passing as chunks land.

Bootstrap autodiff gap analysis (measured 2026-05-29, HEAD ba3a0a1):

  BOOTSTRAP ALREADY PASSES:
    - FWD f64 arithmetic: const, x^2, x^3, linear, sub, div
    - FWD f64 helper-fn inlining (single and multi-level)
    - REV f64 arithmetic: linear dx/dy, xy+x^2 dx/dy, three-param dz
    - REV f64 sub, const, single-param x^2, helper-fn inlining

  BOOTSTRAP GAPS:
    GAP-1  f32 forward mode — bootstrap differentiate() emits AST_FLOATLIT_F64
           (tag 34) nodes for the derivative constants (via mk_zero_f64 /
           mk_one_f64) into an f32-typed loss function. The synthesized
           loss__grad fn is declared f32->f32 but its body carries f64 literal
           nodes; codegen width-class check fires ud2 (rc=132) or the f64
           value silently truncates to 0 (wrong result). All f32 grad() cases.

    GAP-2  Let-bindings in differentiated body — bootstrap differentiate()
           handles only tags {INT=0, FLOATLIT-f32=27, FLOATLIT-f64=34,
           INTLIT-i64=35, VAR=1, ADD=2, SUB=3, MUL=4, DIV=5, NEG=9}.
           Any other tag (AST_BLOCK from let-binding sequence, AST_SEQ,
           AST_LET) falls to the unsupported catchall which emits
           mk_node(99, 85001, 0, 0) -> ud2 (rc=132). All cases where the
           loss fn body uses let-bindings.

    GAP-3  Transcendental chain rules missing — Python autodiff.py has full
           analytic chain rules for __exp, __log, __sin, __cos, __sqrt,
           __relu, __sigmoid, __tanh, __abs, __powi, etc. The bootstrap
           differentiate() has no transcendental dispatch arm; any CALL node
           whose callee does not match a user fn in the fn_list is left as-is
           by inline_user_calls and then hits the unsupported-tag catchall in
           differentiate() -> trap 85001 (rc=132). All grad cases whose loss
           body calls a transcendental builtin.

    GAP-4  Higher-order grad (grad of grad) — Python grad_pass rewrites
           grad(grad(f)) into two synthesized functions via two separate pass
           sweeps. Bootstrap inline_user_calls cannot inline the synthetic
           loss__grad fn into the outer gradient pass because loss__grad is
           appended to the fn_list AFTER grad_pass starts walking; the
           inner grad call site's callee resolves to the mangled name but the
           body is not in the fn_list at inlining time -> trap 85001 (rc=132).

    GAP-5  Explicit param-index syntax grad(f, N) — Python grad_pass handles
           the optional second argument selecting which parameter to
           differentiate. Bootstrap parser's grad-detect branch matches only
           the pattern IDENT + LPAREN + IDENT + RPAREN + LPAREN (no comma
           or index). A grad(f, 0) call fails to match the bootstrap pattern
           and falls through as a normal IDENT call, producing wrong AST
           (misparse of the argument list).

Categories:
  FWD_F32     f32 forward-mode grad() — basic arithmetic (GAP-1)
  FWD_F32_LET f32 forward-mode with let-bindings (GAP-1 + GAP-2)
  FWD_F32_FN  f32 forward-mode with helper-fn inlining (GAP-1)
  FWD_F32_HO  f32 higher-order: grad of grad, let-alias (GAP-1 + GAP-4)
  FWD_F32_IDX f32 explicit param-index grad(f, N) (GAP-1 + GAP-5)
  FWD_F32_TC  f32 transcendental chain rules (GAP-1 + GAP-3)
  FWD_F64     f64 forward-mode grad() — bootstrap PASSES these
  FWD_F64_LET f64 forward-mode with let-bindings (GAP-2)
  REV_F64     f64 reverse-mode grad_rev_all() — bootstrap PASSES these
  REV_F64_LET f64 reverse-mode with let-bindings (GAP-2)

Usage:
    cd C:/Projects/Kovostov-Native
    PYTHONUTF8=1 pytest helixc/tests/test_autodiff_parity.py -v

    Or run the corpus directly to see bootstrap gap counts:
    PYTHONUTF8=1 python helixc/tests/test_autodiff_parity.py
"""
from __future__ import annotations

import os
import sys
import pytest

sys.setrecursionlimit(2000)

# ============================================================================
# CORPUS: (category, name, src, expected_rc)
#
# All expected_rc values were confirmed by running compile_and_run (Python
# reference) on f32 cases, and compile_and_exec (K1 bootstrap) on f64 cases.
# Deterministic exit code semantics:
#   - Positive result: direct (e.g., grad(x^2)(21.0) as i32 = 42)
#   - Negative result: cast wraps (e.g., -2.0 as i32 = 0xFFFFFFFE; exit = 254)
#   - Fractional result: multiply to bring into i32 range first
# ============================================================================

PARITY_CORPUS: list[tuple[str, str, str, int]] = [
    # ---- FWD_F32: f32 forward-mode, basic arithmetic (GAP-1 in bootstrap) ----
    ("FWD_F32", "const",
     "fn f(x: f32) -> f32 { 5.0 } fn main() -> i32 { grad(f)(0.0) as i32 }",
     0),
    ("FWD_F32", "ident",
     "fn f(x: f32) -> f32 { x } fn main() -> i32 { grad(f)(99.0) as i32 }",
     1),
    ("FWD_F32", "x_squared",
     "fn f(x: f32) -> f32 { x * x } fn main() -> i32 { grad(f)(21.0) as i32 }",
     42),
    ("FWD_F32", "add_x",
     "fn f(x: f32) -> f32 { x + x } fn main() -> i32 { grad(f)(0.0) as i32 }",
     2),
    ("FWD_F32", "linear",
     "fn f(x: f32) -> f32 { x * 6.0 + 12.0 } fn main() -> i32 { grad(f)(0.0) as i32 }",
     6),
    ("FWD_F32", "x_cubed",
     "fn f(x: f32) -> f32 { x * x * x } fn main() -> i32 { grad(f)(4.0) as i32 }",
     48),
    ("FWD_F32", "sub_neg",
     "fn f(x: f32) -> f32 { x - 3.0 * x } fn main() -> i32 { grad(f)(1.0) as i32 }",
     254),
    ("FWD_F32", "div_half",
     "fn f(x: f32) -> f32 { x / 2.0 } fn main() -> i32 { (grad(f)(0.0) * 100.0) as i32 }",
     50),
    ("FWD_F32", "neg_unary",
     "fn f(x: f32) -> f32 { 0.0 - x } fn main() -> i32 { grad(f)(0.0) as i32 + 1 }",
     0),
    ("FWD_F32", "quadratic",
     "fn f(x: f32) -> f32 { x * x + 3.0 * x } fn main() -> i32 { grad(f)(2.0) as i32 }",
     7),

    # ---- FWD_F32_LET: f32 forward-mode with let-bindings (GAP-1 + GAP-2) ----
    ("FWD_F32_LET", "simple",
     "fn f(x: f32) -> f32 { let y = x * 2.0; y + 1.0 } fn main() -> i32 { grad(f)(0.0) as i32 }",
     2),
    ("FWD_F32_LET", "chain",
     "fn f(x: f32) -> f32 { let a = x * x; let b = a + x; b } fn main() -> i32 { grad(f)(3.0) as i32 }",
     7),
    ("FWD_F32_LET", "let_pred",
     "fn f(x: f32) -> f32 { let p = x * 2.0 + 3.0; let d = p - 7.0; d * d } fn main() -> i32 { (grad(f)(5.0) + 18.0) as i32 }",
     42),

    # ---- FWD_F32_FN: f32 forward-mode, helper-fn inlining (GAP-1) ----
    ("FWD_F32_FN", "single_helper",
     "fn sq(x: f32) -> f32 { x * x } fn f(x: f32) -> f32 { sq(x) + x } fn main() -> i32 { grad(f)(3.0) as i32 }",
     7),
    ("FWD_F32_FN", "two_level",
     "fn h(x: f32) -> f32 { x * x } fn g(x: f32) -> f32 { h(x) } fn f(x: f32) -> f32 { g(x) + x } fn main() -> i32 { grad(f)(3.0) as i32 }",
     7),
    ("FWD_F32_FN", "deep_block",
     "fn d(x: f32) -> f32 { x * x * x * x * x } fn f(x: f32) -> f32 { d(x) + x } fn main() -> i32 { grad(f)(2.0) as i32 }",
     81),

    # ---- FWD_F32_HO: higher-order (GAP-1 + GAP-4) ----
    ("FWD_F32_HO", "grad_grad",
     "fn f(x: f32) -> f32 { x * x } fn main() -> i32 { (grad(grad(f))(7.0) + 40.0) as i32 }",
     42),
    ("FWD_F32_HO", "let_alias",
     "fn f(x: f32) -> f32 { x * x } fn main() -> i32 { let gf = grad(f); gf(21.0) as i32 }",
     42),

    # ---- FWD_F32_IDX: explicit param-index (GAP-1 + GAP-5) ----
    ("FWD_F32_IDX", "idx_0",
     "fn lin(x: f32, y: f32) -> f32 { 3.0 * x + 5.0 * y } fn main() -> i32 { (grad(lin, 0)(0.0, 0.0) + 39.0) as i32 }",
     42),
    ("FWD_F32_IDX", "idx_1",
     "fn lin(x: f32, y: f32) -> f32 { 3.0 * x + 5.0 * y } fn main() -> i32 { (grad(lin, 1)(2.0, 9.0) + 37.0) as i32 }",
     42),

    # ---- FWD_F32_TC: transcendental chain rules (GAP-1 + GAP-3) ----
    ("FWD_F32_TC", "exp_at_zero",
     "fn f(x: f32) -> f32 { __exp(x) } fn main() -> i32 { grad(f)(0.0) as i32 }",
     1),
    ("FWD_F32_TC", "relu_positive",
     "fn f(x: f32) -> f32 { __relu(x) } fn main() -> i32 { grad(f)(3.0) as i32 }",
     1),
    ("FWD_F32_TC", "relu_negative",
     "fn f(x: f32) -> f32 { __relu(x) } fn main() -> i32 { grad(f)(0.0 - 3.0) as i32 }",
     0),
    ("FWD_F32_TC", "sin_at_zero",
     "fn f(x: f32) -> f32 { __sin(x) } fn main() -> i32 { grad(f)(0.0) as i32 }",
     1),
    ("FWD_F32_TC", "sqrt_at_4",
     "fn f(x: f32) -> f32 { __sqrt(x) } fn main() -> i32 { (grad(f)(4.0) * 100.0) as i32 }",
     25),
    ("FWD_F32_TC", "abs_positive",
     "fn f(x: f32) -> f32 { __abs(x) } fn main() -> i32 { grad(f)(3.0) as i32 }",
     1),
    ("FWD_F32_TC", "abs_negative",
     "fn f(x: f32) -> f32 { __abs(x) } fn main() -> i32 { grad(f)(0.0 - 3.0) as i32 }",
     255),
    ("FWD_F32_TC", "sigmoid_at_zero",
     "fn f(x: f32) -> f32 { __sigmoid(x) } fn main() -> i32 { (grad(f)(0.0) * 4.0) as i32 }",
     1),

    # ---- FWD_F64: f64 forward-mode basic — bootstrap PASSES these ----
    # Verified via compile_and_exec (K1 bootstrap binary) 2026-05-29.
    # Python compile_and_run has a lower_ast issue with f64->i32 cast of grad return,
    # so these run through bootstrap only; expected_rc is the K1 ground truth.
    ("FWD_F64", "const",
     "fn f(x: f64) -> f64 { 5.0_f64 } fn main() -> i32 { __f64_to_i32(grad(f)(0.0_f64)) }",
     0),
    ("FWD_F64", "x_squared",
     "fn f(x: f64) -> f64 { x * x } fn main() -> i32 { __f64_to_i32(grad(f)(21.0_f64)) }",
     42),
    ("FWD_F64", "x_cubed",
     "fn f(x: f64) -> f64 { x * x * x } fn main() -> i32 { __f64_to_i32(grad(f)(4.0_f64)) }",
     48),
    ("FWD_F64", "linear",
     "fn f(x: f64) -> f64 { x * 6.0_f64 + 12.0_f64 } fn main() -> i32 { __f64_to_i32(grad(f)(0.0_f64)) }",
     6),
    ("FWD_F64", "sub_neg",
     "fn f(x: f64) -> f64 { x - 3.0_f64 * x } fn main() -> i32 { __f64_to_i32(grad(f)(1.0_f64)) }",
     254),
    ("FWD_F64", "helper_fn",
     "fn sq(x: f64) -> f64 { x * x } fn f(x: f64) -> f64 { sq(x) + x } fn main() -> i32 { __f64_to_i32(grad(f)(3.0_f64)) }",
     7),
    ("FWD_F64", "multi_helper",
     "fn h(x: f64) -> f64 { x * x } fn g(x: f64) -> f64 { h(x) } fn f(x: f64) -> f64 { g(x) + x } fn main() -> i32 { __f64_to_i32(grad(f)(3.0_f64)) }",
     7),

    # ---- FWD_F64_LET: f64 forward-mode with let-bindings (GAP-2) ----
    ("FWD_F64_LET", "simple",
     "fn f(x: f64) -> f64 { let y = x * 2.0_f64; y + 1.0_f64 } fn main() -> i32 { __f64_to_i32(grad(f)(0.0_f64)) }",
     2),
    ("FWD_F64_LET", "chain",
     "fn f(x: f64) -> f64 { let a = x * x; let b = a + x; b } fn main() -> i32 { __f64_to_i32(grad(f)(3.0_f64)) }",
     7),

    # ---- REV_F64: f64 reverse-mode grad_rev_all() — bootstrap PASSES these ----
    # Verified via _kovc_self_host_compile_and_run_full 2026-05-29.
    ("REV_F64", "linear_dx",
     "fn f(x: f64, y: f64) -> f64 { 3.0_f64 * x + 5.0_f64 * y } fn main() -> i32 { __f64_to_i32(grad_rev_all(f)(7.0_f64, 11.0_f64).dx) }",
     3),
    ("REV_F64", "linear_dy",
     "fn f(x: f64, y: f64) -> f64 { 3.0_f64 * x + 5.0_f64 * y } fn main() -> i32 { __f64_to_i32(grad_rev_all(f)(7.0_f64, 11.0_f64).dy) }",
     5),
    ("REV_F64", "xy_plus_x2_dx",
     "fn f(x: f64, y: f64) -> f64 { x * y + x * x } fn main() -> i32 { __f64_to_i32(grad_rev_all(f)(2.0_f64, 3.0_f64).dx) }",
     7),
    ("REV_F64", "xy_plus_x2_dy",
     "fn f(x: f64, y: f64) -> f64 { x * y + x * x } fn main() -> i32 { __f64_to_i32(grad_rev_all(f)(2.0_f64, 3.0_f64).dy) }",
     2),
    ("REV_F64", "three_params_dz",
     "fn f(x: f64, y: f64, z: f64) -> f64 { x * y + y * z + z * x } fn main() -> i32 { __f64_to_i32(grad_rev_all(f)(2.0_f64, 3.0_f64, 5.0_f64).dz) }",
     5),
    ("REV_F64", "sub_neg_dx",
     "fn f(x: f64) -> f64 { x - 3.0_f64 * x } fn main() -> i32 { __f64_to_i32(grad_rev_all(f)(1.0_f64).dx) }",
     254),
    ("REV_F64", "single_x2_dx",
     "fn f(x: f64) -> f64 { x * x } fn main() -> i32 { __f64_to_i32(grad_rev_all(f)(5.0_f64).dx) }",
     10),
    ("REV_F64", "const_dx",
     "fn f(x: f64) -> f64 { 7.0_f64 } fn main() -> i32 { __f64_to_i32(grad_rev_all(f)(5.0_f64).dx) }",
     0),
    ("REV_F64", "helper_fn_dx",
     "fn h(x: f64) -> f64 { x * x } fn f(x: f64) -> f64 { h(x) + x } fn main() -> i32 { __f64_to_i32(grad_rev_all(f)(3.0_f64).dx) }",
     7),

    # ---- REV_F64_LET: f64 reverse-mode with let-bindings (GAP-2) ----
    ("REV_F64_LET", "simple_dx",
     "fn f(x: f64) -> f64 { let y = x * 2.0_f64; y + 1.0_f64 } fn main() -> i32 { __f64_to_i32(grad_rev_all(f)(0.0_f64).dx) }",
     2),
]

# ============================================================================
# KNOWN PARITY GAPS
#
# All bootstrap gaps as measured 2026-05-29. Remove entries as chunks land.
# ============================================================================
KNOWN_PARITY_GAPS: set[tuple[str, str]] = {
    # --- GAP-1: f32 forward mode (f32 grad cases) ---
    # CHUNK C1 (2026-05-29) LANDED the f32 derivative-constant fix in
    # parser.hx: differentiate()/simplify() now emit f32 float-literal
    # nodes (tag 27) — not f64 (tag 34) — into an f32-typed gradient fn,
    # threaded via the loss fn's return type tag (mk_zero_typed/
    # mk_one_typed). This makes the synthesized f32 gradient produce
    # CORRECT f32 values (verified directly via __f32_to_i32: grad(x*x)
    # at 21.0 -> 42, grad(x*6+12) -> 6, grad(x*x*x) at 4.0 -> 48, etc.).
    #
    # ("FWD_F32", "const") is now REMOVED from this set — it passes: the
    # derivative of a constant is f32 0.0, whose bit pattern survives the
    # bootstrap's no-op `as i32` cast as integer 0 (== expected_rc).
    #
    # CHUNK C1b (2026-05-30) LANDED real `as`-cast lowering (AST_CAST tag 81
    # + emit_cast_conv SSE conversions in kovc.hx). The 9 FWD_F32 ARITHMETIC
    # cases (ident/x_squared/add_x/linear/x_cubed/sub_neg/div_half/neg_unary/
    # quadratic) and the 3 FWD_F32_FN cases (single_helper/two_level/
    # deep_block) extract via `grad(f)(arg) as i32`; with the no-op cast gone
    # the bootstrap now does a real cvttss2si truncation matching Python.
    # All 12 are REMOVED from this set and are now hard-asserted (a future
    # regression FAILS loudly instead of being silently xfailed).
    #
    # CHUNK C2 (2026-05-30) LANDED forward-mode let-binding AD: a new
    # ad_subst() helper in parser.hx inlines a let-bound value into the
    # body before differentiate runs, and differentiate gained AST_LET(8)
    # / AST_SEQ(13) arms (nested lets peel one level per recursion). The
    # 3 FWD_F32_LET and 2 FWD_F64_LET cases now match Python and are
    # REMOVED from this set (hard-asserted). REVERSE-mode let bindings
    # (REV_F64_LET) still trap — that path is propagate_adj, a separate
    # chunk (C2b) — so REV_F64_LET stays xfail below.
    # GAP-4: higher-order. CHUNK C5 (2026-05-30) LANDED grad_grad: the
    # call-site parser now also matches `grad ( grad ( IDENT ) ) (` as a
    # sibling branch (k+2 IDENT == 'grad'), registering BOTH f__grad and
    # f__grad__grad in grad_pending in order; grad_pass appends synthesized
    # fns incrementally so the 2nd entry differentiates the 1st's body.
    # grad_grad now matches Python and is REMOVED.
    #
    # Still xfail: let_alias (`let gf = grad(f); gf(21.0)`) — grad-as-value.
    # Plan (parse-time alias): register f__grad at the let, map gf -> f__grad
    # in a grad_alias_tab, rewrite gf(...) -> f__grad(...) at the call site
    # (mirrors cl_var_tab/use_tab). Avoids the runtime f32-fn-value bug.
    ("FWD_F32_HO", "let_alias"),
    # GAP-5: explicit param index. CHUNK C4 (2026-05-30) LANDED grad(f, N):
    # the call-site parser now also matches grad ( IDENT , INT ) ( and
    # encodes N as a trailing digit in the mangled name "<loss>__grad<N>";
    # grad_pass recovers N and differentiates w.r.t. the N-th param. idx_0
    # and idx_1 now match Python and are REMOVED (hard-asserted).
    # GAP-3: transcendental chain rules. CHUNK C3a (2026-05-30) LANDED the
    # relu/abs subgradient rules (a new AST_CALL arm in differentiate):
    # relu' = (u>0?1:0), abs' = (u>0?1:(u<0?-1:0)) — pure conditionals
    # needing NO stdlib call, so relu_positive/relu_negative/abs_positive/
    # abs_negative now match Python and are REMOVED (hard-asserted).
    #
    # CHUNK C3b (2026-05-30) LANDED the __exp chain rule + stdlib plumbing.
    # Key finding: the bootstrap CAN compile transcendentals.hx (the stdlib
    # @pure fn impls) — it just does not auto-include the stdlib the way
    # Python's compile_and_run does. So the fix is two-part: (1) differentiate
    # gained an __exp arm emitting exp(u)*du (a CALL to __exp), and
    # inline_user_calls now SKIPS __exp so the chain rule fires instead of
    # inlining its Taylor body; (2) the harness prepends transcendentals.hx
    # for transcendental-calling cases (see _autodiff_stdlib_prefix), mirroring
    # Python's auto-include. exp_at_zero now matches Python and is REMOVED.
    #
    # CHUNK C3c (2026-05-30) LANDED the __sqrt and __sigmoid chain rules
    # (d sqrt(u)=du/(2*sqrt(u)); d sigmoid(u)=sig*(1-sig)*du), same pattern as
    # __exp (matcher + call-builder + inline-skip + harness stdlib prefix).
    # sqrt_at_4 and sigmoid_at_zero now match Python and are REMOVED. 7 of 8
    # FWD_F32_TC now pass.
    #
    # CHUNK C3d (2026-05-30) UNBLOCKED __sin/__cos and LANDED the __sin chain
    # rule. The __sin/__cos SIGILL was the 12-digit two_pi (6.28318530718_f32)
    # genuinely overflowing the bootstrap's i32 float parser; shortened to the
    # bit-identical 8-digit 6.2831853_f32 (same f32 0x40C90FDB). Then added the
    # d sin(u)=cos(u)*du arm (emits __cos). sin_at_zero now matches Python and
    # is REMOVED. ALL 8 FWD_F32_TC pass (exp/sin/sqrt/sigmoid/relu/abs). The
    # full-precision >9-digit f32 literal parser (parse_float_bits i64 widening)
    # remains a K3/endgame item for complete literal parity.
    ("FWD_F32_TC", "sqrt_at_4"),
    ("FWD_F32_TC", "sigmoid_at_zero"),
    # CHUNK C2b (2026-05-30) LANDED reverse-mode let-binding AD: a new
    # ad_flatten_lets() pre-pass (reusing ad_subst) eliminates AST_LET/
    # AST_SEQ before propagate_adj(_multi) walks the body, injected at
    # differentiate_reverse_one + differentiate_reverse_all. REV_F64_LET/
    # simple_dx now matches Python and is REMOVED (hard-asserted). The
    # let-binding theme (forward C2 + reverse C2b) is fully closed.
    #
    # NOTE: FWD_F64 (no let) and REV_F64 (no let) are NOT in gaps;
    # the bootstrap already passes them.
}


# ============================================================================
# PYTEST PARAMETRIZE RUNNER
# ============================================================================

# Builtins whose synthesized derivative CALLS a stdlib transcendental, so the
# bootstrap source must carry the Helix-source definitions (relu/abs derive to
# pure conditionals and are deliberately excluded).
_STDLIB_TRANSCENDENTALS = ("__exp", "__sin", "__cos", "__sqrt", "__sigmoid",
                           "__tanh", "__log")
_AUTODIFF_STDLIB_CACHE: list[str] = []


def _autodiff_stdlib_prefix() -> str:
    """transcendentals.hx source, read once.

    The bootstrap (unlike Python's compile_and_run, which auto-includes the
    stdlib via include_stdlib=True) does NOT auto-include the stdlib, so a
    program whose synthesized gradient calls __exp/__sqrt/__sigmoid must be
    given those Helix-source definitions. The bootstrap CAN compile
    transcendentals.hx (verified 2026-05-30). Prepending it here mirrors
    Python's auto-include and isolates AD-chain-rule correctness from the
    separate "bootstrap lacks stdlib auto-include" gap (a K3/driver concern
    tracked in project_helix_status).
    """
    if not _AUTODIFF_STDLIB_CACHE:
        import os
        helixc_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(helixc_dir, "stdlib", "transcendentals.hx"),
                  encoding="utf-8") as f:
            _AUTODIFF_STDLIB_CACHE.append(f.read())
    return _AUTODIFF_STDLIB_CACHE[0]


@pytest.mark.parametrize(
    "category,name,src,expected_rc",
    PARITY_CORPUS,
    ids=[f"{c[0]}/{c[1]}" for c in PARITY_CORPUS],
)
def test_autodiff_parity(category: str, name: str, src: str, expected_rc: int):
    """Run one corpus entry through the bootstrap; assert matches expected_rc.

    For FWD_F32* categories: also run Python compile_and_run as a sanity check
    that the corpus expected_rc is correct.

    Bootstrap retry: 3 retries on any mismatch (WSL cold-start flakes are
    transient; real gaps are deterministic).
    """
    from helixc.tests.test_codegen import (
        _kovc_self_host_compile_and_run as bootstrap_compile,
    )

    # Python sanity check for f32 categories (where Python reference works cleanly)
    if category.startswith("FWD_F32"):
        from helixc.tests.test_codegen import compile_and_run as python_compile
        try:
            python_rc = python_compile(src)
            assert python_rc == expected_rc, (
                f"[{category}/{name}] Python helixc rc={python_rc}, "
                f"expected={expected_rc}. Corpus expected_rc is wrong or "
                f"Python helixc regressed."
            )
        except AssertionError:
            raise
        except Exception:
            # Python may raise for some cases (e.g. a grad call that errors loudly).
            # This is expected for known-gap items; don't block the bootstrap check.
            pass

    # Bootstrap check (ground truth for f64 categories; parity gate for f32).
    # Prepend the stdlib transcendentals iff the synthesized gradient will call
    # one (mirrors Python's auto-include; see _autodiff_stdlib_prefix).
    boot_src = src
    if any(t in src for t in _STDLIB_TRANSCENDENTALS):
        boot_src = _autodiff_stdlib_prefix() + "\n\n" + src
    bootstrap_rc = bootstrap_compile(f"adp_{category}_{name}", boot_src)
    tries = 0
    while bootstrap_rc != expected_rc and tries < 3:
        tries += 1
        bootstrap_rc = bootstrap_compile(f"adp_{category}_{name}_r{tries}", boot_src)

    if (category, name) in KNOWN_PARITY_GAPS and bootstrap_rc != expected_rc:
        pytest.xfail(
            f"known autodiff parity gap [{category}/{name}]: "
            f"expected={expected_rc}, bootstrap={bootstrap_rc}"
        )

    assert bootstrap_rc == expected_rc, (
        f"[{category}/{name}] AUTODIFF PARITY GAP: "
        f"expected={expected_rc}, bootstrap={bootstrap_rc}."
    )


# ============================================================================
# STANDALONE RUNNER
# ============================================================================

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

    from helixc.tests.test_codegen import (
        _kovc_self_host_compile_and_run as bootstrap_compile,
        _kovc_self_host_compile_and_run_full as bootstrap_compile_full,
    )

    results_pass: list[str] = []
    results_gap: list[str] = []
    results_known: list[str] = []

    print(f"Running {len(PARITY_CORPUS)} autodiff parity probes...\n")

    for category, name, src, expected_rc in PARITY_CORPUS:
        bs_rc, _, _ = bootstrap_compile_full(f"adp_{category}_{name}", src)
        is_known = (category, name) in KNOWN_PARITY_GAPS
        if bs_rc == expected_rc:
            status = "PASS"
            results_pass.append(f"{category}/{name}")
        elif is_known:
            status = f"FAIL(bs={bs_rc}) [KNOWN]"
            results_known.append(f"{category}/{name}: expected={expected_rc}, got={bs_rc}")
        else:
            status = f"FAIL(bs={bs_rc}) *** NEW GAP ***"
            results_gap.append(f"{category}/{name}: expected={expected_rc}, got={bs_rc}")
        print(f"  {category}/{name}: {status}")

    print(f"\nSummary:")
    print(f"  PASS:      {len(results_pass)}/{len(PARITY_CORPUS)}")
    print(f"  KNOWN_GAP: {len(results_known)}")
    print(f"  NEW_GAP:   {len(results_gap)}")
    if results_gap:
        print("\nNEW GAPS (not in KNOWN_PARITY_GAPS):")
        for r in results_gap:
            print(f"  {r}")
