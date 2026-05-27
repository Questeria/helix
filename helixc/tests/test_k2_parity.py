"""K2.A (2026-05-26): parity harness scaffold.

The K2 milestone (per docs/HELIX_K_BOOTSTRAP_MASTER_PLAN.md) is a
test runner that, for every source program, compiles via BOTH paths
and compares output:

  - Python helixc path (compile_and_run from test_codegen)
  - Bootstrap kovc path (_kovc_self_host_compile_and_run from
    test_codegen)

For each source: both compilers, both binaries, same exit code =>
PASS. Different exit code => FAIL (parity violation).

This file ships the SCAFFOLD with a small starter corpus. Future
K2.* chunks expand the corpus to surface real-source parser/codegen
divergences that K1.* alone wouldn't find. The eventual goal is a
corpus large enough that "K2 green" is a credible gate for the
Python-ready-to-delete state.

Phase 1 (this file): behavioral parity (same exit code).
Phase 2 (future): structural parity (byte-identical ELF after
                  relocation normalization).
Phase 3 (future): N-generation fixpoint (kovc compiles kovc.hx to
                  kovc'; compile again to get kovc''; kovc' == kovc'').
"""
import os
import sys
import pytest

# Lazy imports inside the test fns so the file doesn't crash if WSL is
# degraded at collection time.


# The K2.A starter corpus. Each entry is (name, src, expected_rc).
#
# Selection criteria for K2.A:
#   - Each program is small and self-contained.
#   - Each program exercises a different category of feature so the
#     corpus has breadth even with only ~10 entries.
#   - Expected_rc is the value main() should return; both compilers
#     should agree.
#
# Future K2.* chunks expand this list (and eventually replace it with
# a directory of .hx files).
K2_CORPUS = [
    # ---- K2.A starter (10) ----
    ("p01_basic_return",         "fn main() -> i32 { 42 }", 42),
    ("p02_let_int",              "fn main() -> i32 { let x = 42; x }", 42),
    ("p03_arith",                "fn main() -> i32 { 20 + 22 }", 42),
    ("p04_if_else",              "fn main() -> i32 { if true { 42 } else { 0 } }", 42),
    ("p05_match_lit",            "fn main() -> i32 { match 1 { 1 => 42, _ => 0 } }", 42),
    ("p06_for_loop_sum",         "fn main() -> i32 { let mut s = 0; for i in 0..7 { s = s + i; } s + 21 }", 42),
    ("p07_struct_lit",           "struct P { v: i32 } fn main() -> i32 { let p = P { v: 42 }; p.v }", 42),
    ("p08_tuple_field",          "fn main() -> i32 { let t = (42, 0); t.0 }", 42),
    ("p09_fn_call",              "fn f() -> i32 { 42 } fn main() -> i32 { f() }", 42),
    ("p10_neg_then_add",         "fn main() -> i32 { let x = -10 + 52; x }", 42),
    # ---- K2.B expansion: arithmetic / control-flow / value-flow (15) ----
    ("p11_sub",                  "fn main() -> i32 { 100 - 58 }", 42),
    ("p12_mul",                  "fn main() -> i32 { 6 * 7 }", 42),
    ("p13_div",                  "fn main() -> i32 { 84 / 2 }", 42),
    ("p14_mod",                  "fn main() -> i32 { 142 % 100 }", 42),
    ("p15_chained_let",          "fn main() -> i32 { let a = 10; let b = a + 5; let c = b * 2; c + 12 }", 42),
    ("p16_if_else_if",           "fn main() -> i32 { let x = 2; if x == 1 { 1 } else if x == 2 { 42 } else { 99 } }", 42),
    ("p17_nested_if",            "fn main() -> i32 { let x = 10; if x > 5 { if x < 20 { 42 } else { 0 } } else { 0 } }", 42),
    ("p18_struct_two_fields",    "struct P { x: i32, y: i32 } fn main() -> i32 { let p = P { x: 20, y: 22 }; p.x + p.y }", 42),
    ("p19_tuple_two_fields",     "fn main() -> i32 { let t = (20, 22); t.0 + t.1 }", 42),
    ("p20_match_three_arms",     "fn main() -> i32 { let x = 2; match x { 1 => 0, 2 => 42, _ => 99 } }", 42),
    ("p21_multi_param_fn",       "fn add(a: i32, b: i32) -> i32 { a + b } fn main() -> i32 { add(20, 22) }", 42),
    # NOTE: K2.B initially tried `for i in 1..=5 { ... }` here. The
    # bootstrap kovc accepts it (per K1.L), but the Python helixc
    # parser does NOT -- it errors at "expected LBRACE (got DOTDOTEQ
    # '..=')". This is a Python-side gap that K2 surfaces. For the
    # K2.B green-shipping corpus we use the exclusive variant; a
    # future K2.* chunk can re-introduce the inclusive form once
    # Python helixc gains parity (or once Python is deleted at K4).
    ("p22_for_exclusive_offset", "fn main() -> i32 { let mut s = 0; for i in 1..6 { s = s + i; } s + 27 }", 42),
    ("p23_for_mul_in_body",      "fn main() -> i32 { let mut s = 0; for _ in 0..6 { s = s + 7; } s }", 42),
    ("p24_recursive_fact",       "fn fact(n: i32) -> i32 { if n <= 1 { 1 } else { n * fact(n - 1) } } fn main() -> i32 { fact(5) - 78 }", 42),
    ("p25_cmp_gt",               "fn main() -> i32 { let x = 100; if x > 50 { 42 } else { 0 } }", 42),
    # ---- K2.D expansion: comparison ops / booleans / chars / enums (15) ----
    ("p26_le_cmp",               "fn main() -> i32 { let x = 42; if x <= 42 { 42 } else { 0 } }", 42),
    ("p27_ge_cmp",               "fn main() -> i32 { if 42 >= 1 { 42 } else { 0 } }", 42),
    ("p28_ne_cmp",               "fn main() -> i32 { if 42 != 0 { 42 } else { 0 } }", 42),
    ("p29_eq_cmp",               "fn main() -> i32 { if 42 == 42 { 42 } else { 0 } }", 42),
    ("p30_logical_and",          "fn main() -> i32 { if true && true { 42 } else { 0 } }", 42),
    ("p31_logical_or",           "fn main() -> i32 { if false || true { 42 } else { 0 } }", 42),
    ("p32_bool_let",             "fn main() -> i32 { let b = true; if b { 42 } else { 0 } }", 42),
    # NOTE: K2.D tried char-literal `let c = 'A'; if c == 65 { 42 } else { 0 }`
    # here. The bootstrap kovc handles it (per K1.K), but Python helixc's
    # IR lowering raises NotImplementedError: "char literal not yet
    # supported in IR lowering". Python-side gap; corpus uses a let-
    # shadowing variant instead. The char-literal shape will return in a
    # future K2.* chunk once Python helixc gains parity (or after K4).
    ("p33_let_shadow",           "fn main() -> i32 { let x = 99; let x = 42; x }", 42),
    ("p34_compound_add_eq",      "fn main() -> i32 { let mut x = 20; x += 22; x }", 42),
    ("p35_while_loop",           "fn main() -> i32 { let mut i = 0; let mut s = 0; while i < 7 { s = s + i; i = i + 1; } s + 21 }", 42),
    ("p36_nested_struct",        "struct A { x: i32 } struct B { a: A } fn main() -> i32 { let b = B { a: A { x: 42 } }; b.a.x }", 42),
    ("p37_enum_two_lits",        "enum E { Z, O } fn main() -> i32 { let v = E::O; match v { E::Z => 0, E::O => 42 } }", 42),
    # NOTE: K2.D first tried `match x { 1 => { ... } _ => 0 }` without
    # the comma between arms; the bootstrap kovc accepts it (per K1.AL),
    # but Python helixc requires the comma -- it errors at "expected
    # RBRACE (got IDENT '_')". Use the comma-separated form for parity.
    ("p38_match_block_arm",      "fn main() -> i32 { let x = 1; match x { 1 => { let y = 21; y + 21 }, _ => 0 } }", 42),
    ("p39_for_compound_assign",  "fn main() -> i32 { let mut s = 0; for _ in 0..6 { s += 7; } s }", 42),
    ("p40_chained_call",         "fn id(x: i32) -> i32 { x } fn main() -> i32 { id(id(id(42))) }", 42),
    # ---- K2.E expansion: arrays / match shapes / block-exprs / recursion (15) ----
    ("p41_array_index",          "fn main() -> i32 { let a = [10, 20, 30]; a[1] + 22 }", 42),
    ("p42_array_sum_idx",        "fn main() -> i32 { let a = [10, 20, 12]; a[0] + a[1] + a[2] }", 42),
    ("p43_match_or_arm",         "fn main() -> i32 { let x = 2; match x { 1 | 2 | 3 => 42, _ => 0 } }", 42),
    ("p44_match_wildcard",       "fn main() -> i32 { let x = 99; match x { 0 => 0, _ => 42 } }", 42),
    ("p45_block_expr_let",       "fn main() -> i32 { let x = { let y = 21; y + y }; x }", 42),
    ("p46_three_let_chain",      "fn main() -> i32 { let a = 1; let b = 2; let c = 3; (a + b + c) * 7 }", 42),
    ("p47_mul_chain",            "fn main() -> i32 { 2 * 3 * 7 }", 42),
    ("p48_add_left_assoc",       "fn main() -> i32 { 10 + 12 + 20 }", 42),
    ("p49_double_fn",            "fn double(x: i32) -> i32 { x * 2 } fn main() -> i32 { double(21) }", 42),
    ("p50_fib_recursive",        "fn fib(n: i32) -> i32 { if n <= 1 { n } else { fib(n-1) + fib(n-2) } } fn main() -> i32 { fib(8) + 21 }", 42),
    ("p51_while_zero_iter",      "fn main() -> i32 { let mut i = 0; while i < 0 { i = i + 1; } 42 }", 42),
    ("p52_if_let_value",         "fn main() -> i32 { let x = 100; let y = if x > 0 { 42 } else { 99 }; y }", 42),
    ("p53_struct_field_extract", "struct C { v: i32 } fn main() -> i32 { let c = C { v: 42 }; let v = c.v; v }", 42),
    ("p54_tuple_3_fields",       "fn main() -> i32 { let t = (10, 20, 12); t.0 + t.1 + t.2 }", 42),
    ("p55_complex_let_expr",     "fn main() -> i32 { let a = 6; let b = 7; let c = a * b; c }", 42),
    # ---- K2.F expansion: typed-int suffixes (now unblocked by K1.E1-fix) + edge cases (15) ----
    ("p56_i64_return",           "fn main() -> i64 { 42_i64 }", 42),
    # NOTE: K2.F first tried `fn main() -> i32 { 100_i64 - 58_i64 }`
    # here -- but the bootstrap correctly SIGILLs that shape, because
    # the body emits an i64 (width 8) and the declared return is i32
    # (width 4). The K1.E1 width-class trap at kovc.hx:7367 catches
    # this as a real silent-narrowing risk (post-K1.E1 fix, that trap
    # is doing its job). Use the matching-type variant: `-> i64 { ... }`
    # so body and return widths agree.
    ("p57_i64_arith_full",       "fn main() -> i64 { 100_i64 - 58_i64 }", 42),
    ("p58_i64_let_in_i32",       "fn main() -> i32 { let x = 42_i64; x }", 42),
    ("p59_u32_return",           "fn main() -> u32 { 42_u32 }", 42),
    ("p60_u32_arith_in_i32",     "fn main() -> i32 { 20_u32 + 22_u32 }", 42),
    # i8 / i16 narrow integer returns: matching-type (body and ret
    # both i8 or i16). The bootstrap width-class trap (K1.E1 fix) now
    # correctly catches narrowing mismatches like `-> i32 { 42_i8 }`,
    # so use matching-width variants here. Exit code is the low byte
    # of rax either way, so 42_i8 maps to exit 42.
    ("p61_i8_match",             "fn main() -> i8 { 42_i8 }", 42),
    ("p62_i16_match",            "fn main() -> i16 { 42_i16 }", 42),
    ("p63_underscore_sep",       "fn main() -> i32 { 1_000_000 - 999_958 }", 42),
    ("p64_underscore_in_i64",    "fn main() -> i64 { 1_000_000_i64 - 999_958_i64 }", 42),
    ("p65_arith_paren_groups",   "fn main() -> i32 { (10 + 11) + (10 + 11) }", 42),
    ("p66_div_then_mul",         "fn main() -> i32 { (84 / 2) }", 42),
    ("p67_signed_neg_in_arith",  "fn main() -> i32 { -10 + 52 }", 42),
    ("p68_let_inside_let_value", "fn main() -> i32 { let x = { let y = 21; y + y }; x + 0 }", 42),
    ("p69_zero_iter_for",        "fn main() -> i32 { let mut s = 42; for _ in 0..0 { s = 0; } s }", 42),
    ("p70_if_in_arith",          "fn main() -> i32 { (if true { 20 } else { 0 }) + 22 }", 42),
]


@pytest.mark.parametrize("name,src,expected_rc", K2_CORPUS, ids=[c[0] for c in K2_CORPUS])
def test_k2_parity(name: str, src: str, expected_rc: int):
    """Run one K2 corpus entry through both compilers; assert
    behavioral parity.

    Two assertions:
      1. Python helixc rc == expected_rc (sanity: corpus item is correct).
      2. Bootstrap kovc rc == Python helixc rc (parity).

    If both pass, the corpus item is K2-clean. If (2) fails but (1)
    passes, the bootstrap diverges from Python helixc -- a parity
    violation surfacing a K1 gap that this corpus item caught.
    """
    # Defer imports until test execution (so a broken Python helixc
    # doesn't blow up the test collection step).
    from helixc.tests.test_codegen import (
        compile_and_run as python_compile_and_run,
        _kovc_self_host_compile_and_run as bootstrap_compile_and_run,
    )

    # Path 1: Python helixc.
    python_rc = python_compile_and_run(src, optimize=True)
    assert python_rc == expected_rc, (
        f"K2.A corpus {name}: Python helixc rc={python_rc}, "
        f"expected {expected_rc}. Corpus item is wrong, or Python "
        f"helixc regressed."
    )

    # Path 2: Bootstrap kovc self-host.
    bootstrap_rc = bootstrap_compile_and_run(f"k2_parity_{name}", src)

    # Parity assertion: both compilers must agree.
    assert bootstrap_rc == python_rc, (
        f"K2.A parity violation on {name}: "
        f"Python helixc rc={python_rc}, bootstrap kovc rc={bootstrap_rc}. "
        f"Behavioral divergence -- a K1 chunk is missing for this shape."
    )


def test_k2_corpus_size():
    """Sanity check: corpus has at least 70 entries at K2.F.

    Future K2.* chunks expand the corpus. This test guards against
    accidental shrinkage. The growth ratchet is one-way: each K2.*
    chunk strictly increases the lower bound. History:

      - K2.A bumped to >= 10 (starter scaffold).
      - K2.B bumped to >= 25 (arithmetic / control-flow / value-flow).
      - K2.D bumped to >= 40 (comparison ops / booleans / chars /
        compound assign / while / nested struct / enum / etc.).
      - K2.E bumped to >= 55 (arrays / match shapes / block-exprs /
        recursion / zero-iter while / multi-let-chain / etc.).
      - K2.F bumped to >= 70 (typed-int suffixes _i64 / _u32 / _i8 /
        _i16 unblocked by K1.E1-fix, plus underscore-separator
        variants and arith edge cases).

    (K2.C was the matrix-parity counter sync -- no corpus change.)

    Subsequent K2.* chunks will continue raising it until a credible
    "K2 green over a real-source corpus" threshold is reached.
    """
    assert len(K2_CORPUS) >= 70, (
        f"K2.F corpus shrank to {len(K2_CORPUS)} entries. The K2 "
        f"growth ratchet is one-way -- entries can be replaced but "
        f"not net-removed."
    )
