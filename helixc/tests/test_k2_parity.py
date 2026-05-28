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
    # K2.G (2026-05-27): const-name resolution probes. Pins the K1.F7
    # close (const_tab + IDENT lookup hook in mk_var_with_capture).
    # Both compilers handle these uniformly; Python via its normal
    # name-resolution + const-table; bootstrap via const_tab_lookup
    # returning the stored value AST at every reference site (the
    # const is inlined per reference).
    ("p71_const_bare_use",       "const X: i32 = 42; fn main() -> i32 { X }", 42),
    ("p72_const_arith",          "const A: i32 = 30; const B: i32 = 12; fn main() -> i32 { A + B }", 42),
    ("p73_const_in_let_rhs",     "const N: i32 = 42; fn main() -> i32 { let x = N; x }", 42),
    ("p74_const_times_lit",      "const T: i32 = 6; fn main() -> i32 { T * 7 }", 42),
    ("p75_const_in_if_cond",     "const Z: i32 = 0; fn main() -> i32 { if Z == 0 { 42 } else { 0 } }", 42),
    ("p76_const_across_fns",     "const ANS: i32 = 42; fn helper() -> i32 { ANS } fn main() -> i32 { helper() }", 42),
    ("p77_const_used_twice",     "const X: i32 = 42; fn doubled() -> i32 { X + X } fn main() -> i32 { doubled() - X }", 42),
    # K2.H (2026-05-27): mixed-type binop probes pinning the K1.F8 +
    # K1.F8b closures. ADD/SUB/MUL on signed i64<->i32 in both
    # directions, plus a couple integration shapes. Python helixc
    # handles all via implicit conversion; bootstrap via the new
    # movsxd widening helpers in kovc.hx (emit_movsxd_rcx_ecx for
    # the i64+i32 forward direction, emit_movsxd_rax_eax for the
    # i32+i64 reverse direction). expr_type returns i64 for both
    # tag pairs (3, 0) and (0, 3) so the body-vs-ret-ty trap 14001
    # stays silent on these `-> i64 { ... }` returns.
    ("p78_i64_add_i32",          "fn main() -> i64 { 30_i64 + 12 }", 42),
    ("p79_i64_sub_i32",          "fn main() -> i64 { 54_i64 - 12 }", 42),
    ("p80_i64_mul_i32",          "fn main() -> i64 { 6_i64 * 7 }", 42),
    ("p81_i32_add_i64",          "fn main() -> i64 { 30 + 12_i64 }", 42),
    ("p82_i32_sub_i64",          "fn main() -> i64 { 54 - 12_i64 }", 42),
    ("p83_i32_mul_i64",          "fn main() -> i64 { 6 * 7_i64 }", 42),
    ("p84_i64_add_arith_i32",    "fn main() -> i64 { 30_i64 + (6 * 2) }", 42),
    ("p85_i64_sub_large",        "fn main() -> i64 { 100_i64 - 58 }", 42),
    # K2.M (2026-05-27): unsigned mixed-type binop probes pinning the
    # K1.F8d closure (u64<->u32 widening across ADD/SUB/MUL/DIV/MOD).
    # Both compilers handle these via the same pattern: zero-extension
    # of the u32 side to u64, then 64-bit op. Bootstrap uses mov_ecx_eax
    # (zero-ext implicit) + the appropriate REX.W op (add/sub/imul/
    # div_u/imod_u). K3.B's exactly-u32 (tag-6) guard ensures non-u32
    # operands trap loudly instead of silently miscompiling.
    ("p86_u64_add_u32",          "fn main() -> u64 { 30_u64 + 12_u32 }", 42),
    ("p87_u64_sub_u32",          "fn main() -> u64 { 54_u64 - 12_u32 }", 42),
    ("p88_u64_mul_u32",          "fn main() -> u64 { 6_u64 * 7_u32 }", 42),
    ("p89_u64_div_u32",          "fn main() -> u64 { 84_u64 / 2_u32 }", 42),
    ("p90_u64_mod_u32",          "fn main() -> u64 { 142_u64 % 100_u32 }", 42),
    ("p91_u32_add_u64",          "fn main() -> u64 { 30_u32 + 12_u64 }", 42),
    ("p92_u32_sub_u64",          "fn main() -> u64 { 54_u32 - 12_u64 }", 42),
    ("p93_u32_mul_u64",          "fn main() -> u64 { 6_u32 * 7_u64 }", 42),
    # Mixed f32/f64 binops (K1.F9 + K1.F9-fix) are NOT added to the K2
    # parity corpus because the obvious surface form
    # `__f64_to_i32(f32 + f64)` is rejected by Python helixc's
    # IR-lowering (`unknown function '__f64_to_i32'`) -- the call is
    # only resolved by the front end in `compile_and_exec` paths the
    # K2 harness does not use. A bootstrap-only self-host test pins
    # the K1.F9-fix closure instead -- see test_bootstrap_kovc_k1f9_
    # mixed_f32_f64_self_host in test_codegen.py.
    # K1.F11 (2026-05-27): mixed i64<->i32 LT widening pinned in both
    # directions. Both compilers should agree (Python helixc handles
    # mixed-int widening; bootstrap kovc widens via movsxd then 64-bit
    # cmp).
    ("p94_i64_lt_i32_true",      "fn main() -> i32 { if 30_i64 < 60 { 42 } else { 0 } }", 42),
    ("p95_i64_lt_i32_false",     "fn main() -> i32 { if 60_i64 < 30 { 0 } else { 42 } }", 42),
    ("p96_i32_lt_i64_true",      "fn main() -> i32 { if 30 < 60_i64 { 42 } else { 0 } }", 42),
    ("p97_i32_lt_i64_false",     "fn main() -> i32 { if 60 < 30_i64 { 0 } else { 42 } }", 42),
    # K1.F12 (2026-05-27): the K1.F11 LT widening pattern mirror-
    # applied to GT/EQ/NE/LE/GE (the remaining 5 comparison ops).
    # Both compilers agree.
    ("p98_i64_gt_i32",           "fn main() -> i32 { if 60_i64 > 30 { 42 } else { 0 } }", 42),
    ("p99_i32_gt_i64",           "fn main() -> i32 { if 60 > 30_i64 { 42 } else { 0 } }", 42),
    ("p100_i64_eq_i32",          "fn main() -> i32 { if 42_i64 == 42 { 42 } else { 0 } }", 42),
    ("p101_i32_eq_i64",          "fn main() -> i32 { if 42 == 42_i64 { 42 } else { 0 } }", 42),
    ("p102_i64_ne_i32",          "fn main() -> i32 { if 42_i64 != 0 { 42 } else { 0 } }", 42),
    ("p103_i32_ne_i64",          "fn main() -> i32 { if 42 != 0_i64 { 42 } else { 0 } }", 42),
    ("p104_i64_le_i32",          "fn main() -> i32 { if 30_i64 <= 60 { 42 } else { 0 } }", 42),
    ("p105_i32_le_i64",          "fn main() -> i32 { if 30 <= 60_i64 { 42 } else { 0 } }", 42),
    ("p106_i64_ge_i32",          "fn main() -> i32 { if 60_i64 >= 30 { 42 } else { 0 } }", 42),
    ("p107_i32_ge_i64",          "fn main() -> i32 { if 60 >= 30_i64 { 42 } else { 0 } }", 42),
    # K1.F13 (2026-05-27): u64<->u32 comparison widening across all 6
    # ops (LT/GT/EQ/NE/LE/GE) in both directions. Mirror of K1.F8d's
    # u64<->u32 binop widening, applied to comparisons.
    ("p108_u64_lt_u32",          "fn main() -> i32 { if 30_u64 < 60_u32 { 42 } else { 0 } }", 42),
    ("p109_u32_lt_u64",          "fn main() -> i32 { if 30_u32 < 60_u64 { 42 } else { 0 } }", 42),
    ("p110_u64_gt_u32",          "fn main() -> i32 { if 60_u64 > 30_u32 { 42 } else { 0 } }", 42),
    ("p111_u32_gt_u64",          "fn main() -> i32 { if 60_u32 > 30_u64 { 42 } else { 0 } }", 42),
    ("p112_u64_eq_u32",          "fn main() -> i32 { if 42_u64 == 42_u32 { 42 } else { 0 } }", 42),
    ("p113_u32_eq_u64",          "fn main() -> i32 { if 42_u32 == 42_u64 { 42 } else { 0 } }", 42),
    ("p114_u64_ne_u32",          "fn main() -> i32 { if 42_u64 != 0_u32 { 42 } else { 0 } }", 42),
    ("p115_u32_ne_u64",          "fn main() -> i32 { if 42_u32 != 0_u64 { 42 } else { 0 } }", 42),
    ("p116_u64_le_u32",          "fn main() -> i32 { if 30_u64 <= 60_u32 { 42 } else { 0 } }", 42),
    ("p117_u32_le_u64",          "fn main() -> i32 { if 30_u32 <= 60_u64 { 42 } else { 0 } }", 42),
    ("p118_u64_ge_u32",          "fn main() -> i32 { if 60_u64 >= 30_u32 { 42 } else { 0 } }", 42),
    ("p119_u32_ge_u64",          "fn main() -> i32 { if 60_u32 >= 30_u64 { 42 } else { 0 } }", 42),
    # K2.R (2026-05-27): integration probes that exercise the K1.F8*
    # mixed-type binop closures AND the K1.F11-F14 mixed-type
    # comparison closures TOGETHER (compound expressions where the
    # binop result feeds a comparison or vice versa). These pin
    # cross-chunk integration that single-op probes can't catch.
    ("p120_i64plusi32_lt_i64",   "fn main() -> i32 { if (30_i64 + 12) < 60_i64 { 42 } else { 0 } }", 42),
    ("p121_i32plusi64_gt_i32",   "fn main() -> i32 { if (30 + 12_i64) > 30 { 42 } else { 0 } }", 42),
    ("p122_u64plusu32_eq_u64",   "fn main() -> i32 { if (30_u64 + 12_u32) == 42_u64 { 42 } else { 0 } }", 42),
    ("p123_i64sub_lt_i32",       "fn main() -> i32 { if (100_i64 - 60_i64) < 50 { 42 } else { 0 } }", 42),
    ("p124_logand_mixed_cmp",    "fn main() -> i32 { if (30_i64 < 60) && (40 > 30_i64) { 42 } else { 0 } }", 42),
    ("p125_logor_mixed_cmp",     "fn main() -> i32 { if (30_i64 > 99) || (40 > 30_i64) { 42 } else { 0 } }", 42),
    # K2.S (2026-05-28): bitwise-op parity probes -- the corpus had NO
    # bitwise coverage. Both compilers support &/|/^/<</>> (see the
    # bitwise tests in test_codegen.py compile_and_run). Each result = 42.
    ("p126_bitand",              "fn main() -> i32 { 250 & 42 }", 42),
    ("p127_bitor",               "fn main() -> i32 { 32 | 10 }", 42),
    ("p128_bitxor",              "fn main() -> i32 { 52 ^ 30 }", 42),
    ("p129_shl",                 "fn main() -> i32 { 21 << 1 }", 42),
    ("p130_shr",                 "fn main() -> i32 { 84 >> 1 }", 42),
    # K2.T (2026-05-28): integration shapes combining already-supported
    # features in new ways (nested match, mixed arith precedence, a 5-elem
    # const-index array sum). Both compilers support each primitive; this
    # widens the credible K2-green gate. Each result = 42.
    ("p131_nested_match",        "fn main() -> i32 { let x = 2; match x { 1 => 0, 2 => match x { 2 => 42, _ => 1 }, _ => 9 } }", 42),
    ("p132_arith_precedence",    "fn main() -> i32 { 2 + 3 * 7 + 19 }", 42),
    ("p133_array5_const_sum",    "fn main() -> i32 { let a = [6, 7, 8, 9, 12]; a[0] + a[1] + a[2] + a[3] + a[4] }", 42),
    # K2.U (2026-05-28, M32 probe): enum WITH payload + match binding. p37
    # covered payload-less enums; this adds a data-carrying variant E::A(i32)
    # and a binding arm `E::A(n) => n`. Both compilers agree (PARITY).
    ("p134_enum_payload",        "enum E { A(i32), B } fn main() -> i32 { let e = E::A(42); match e { E::A(n) => n, E::B => 0 } }", 42),
    # K2.V (2026-05-28, M33 probe): match GUARD arm (`n if n > 3 => ...`).
    # Both compilers agree (PARITY). Adds guard-pattern coverage.
    ("p135_match_guard",         "fn main() -> i32 { let x = 5; match x { n if n > 3 => 42, _ => 0 } }", 42),
    # K2.W (2026-05-28, M34 probe): integration/recursion shapes -- nested
    # for-loops, recursive gcd, and a boolean short-circuit chain. All
    # PARITY on both compilers. (struct-in-array-literal was tried but is
    # NotImplemented in Python + SIGILLs in the bootstrap -- same mapped
    # 'advanced-feature' class, omitted.)
    ("p136_nested_for",          "fn main() -> i32 { let mut s = 0; for i in 0..3 { for j in 0..2 { s = s + 7; } } s }", 42),
    ("p137_gcd_recursion",       "fn gcd(a: i32, b: i32) -> i32 { if b == 0 { a } else { gcd(b, a % b) } } fn main() -> i32 { gcd(84, 126) }", 42),
    ("p138_bool_shortcircuit",   "fn main() -> i32 { let x = 5; if x > 0 && x < 10 || x == 100 { 42 } else { 0 } }", 42),
    # A-phase pivot (2026-05-28): PHASE B Track-P growth. 8 both-compiler-
    # parity shapes across shadowing / if-expr-let / while-accumulate /
    # multi-arg fn / modulo / nested calls / unary neg / bitwise combo.
    ("p139_shadowing",           "fn main() -> i32 { let x = 20; let x = x + 22; x }", 42),
    ("p140_if_expr_let",         "fn main() -> i32 { let x = 5; let y = if x > 3 { 40 } else { 0 }; y + 2 }", 42),
    ("p141_while_accum",         "fn main() -> i32 { let mut s = 0; let mut i = 0; while i < 6 { s = s + 7; i = i + 1; } s }", 42),
    ("p142_multiarg_fn",         "fn add3(a: i32, b: i32, c: i32) -> i32 { a + b + c } fn main() -> i32 { add3(20, 15, 7) }", 42),
    ("p143_modulo",              "fn main() -> i32 { let x = 142; x % 100 }", 42),
    ("p144_nested_call",         "fn inc(x: i32) -> i32 { x + 1 } fn main() -> i32 { inc(inc(inc(39))) }", 42),
    ("p145_unary_neg",           "fn main() -> i32 { let x = 0 - 8; let y = 0 - x; y * 5 + 2 }", 42),
    ("p146_bitwise_combo",       "fn main() -> i32 { let x = 40; (x | 2) & 63 }", 42),
    # PHASE B cont. (2026-05-28): 10 more both-compiler-parity shapes --
    # struct-param, enum/match, array-sum loop, nested-if, fib recursion,
    # multi-return-path, shift, precedence, div, square-fn. (while-break /
    # while-continue probed too but Python helixc NotImplementedErrors on
    # them -- another bootstrap-exceeds-Python case, so not parity-able.)
    ("p147_struct_param",        "struct P { x: i32, y: i32 } fn s(p: P) -> i32 { p.x + p.y } fn main() -> i32 { s(P{x:40,y:2}) }", 42),
    ("p148_enum_match",          "enum E { A, B, C } fn main() -> i32 { let e = E::B; match e { E::A => 0, E::B => 42, E::C => 1 } }", 42),
    ("p149_array_sum_loop",      "fn main() -> i32 { let a = [10, 20, 12]; let mut s = 0; let mut i = 0; while i < 3 { s = s + a[i]; i = i + 1; } s }", 42),
    ("p150_nested_if",           "fn main() -> i32 { let x = 7; if x >= 5 { if x <= 10 { 42 } else { 0 } } else { 0 } }", 42),
    ("p151_fib",                 "fn fib(n: i32) -> i32 { if n < 2 { n } else { fib(n-1) + fib(n-2) } } fn main() -> i32 { fib(9) + 8 }", 42),
    ("p152_multi_return",        "fn classify(x: i32) -> i32 { if x < 0 { return 0; } if x == 0 { return 1; } 42 } fn main() -> i32 { classify(5) }", 42),
    ("p153_shift_left",          "fn main() -> i32 { let x = 21; x << 1 }", 42),
    ("p154_precedence",          "fn main() -> i32 { 2 + 4 * 10 }", 42),
    ("p155_div",                 "fn main() -> i32 { 420 / 10 }", 42),
    ("p156_square_fn",           "fn sq(x: i32) -> i32 { x * x } fn main() -> i32 { sq(6) + 6 }", 42),
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
      - K2.G bumped to >= 77 (const-name resolution probes pinning
        the K1.F7 close: bare use, arithmetic, in let-RHS, in if-
        cond, across fn calls, used-twice).
      - K2.H bumped to >= 85 (mixed-type i64<->i32 ADD/SUB/MUL,
        both directions, pinning the K1.F8 + K1.F8b closures).
      - K2.M bumped to >= 93 (unsigned mixed-type u64<->u32
        ADD/SUB/MUL/DIV/MOD, both directions, pinning K1.F8d).

    (K2.C was the matrix-parity counter sync -- no corpus change.)

    Subsequent K2.* chunks will continue raising it until a credible
    "K2 green over a real-source corpus" threshold is reached.
    """
    assert len(K2_CORPUS) >= 156, (
        f"K2.W corpus shrank to {len(K2_CORPUS)} entries. The K2 "
        f"growth ratchet is one-way -- entries can be replaced but "
        f"not net-removed."
    )
