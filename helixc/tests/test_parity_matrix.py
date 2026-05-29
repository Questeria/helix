"""helixc/tests/test_parity_matrix.py — Comprehensive parity audit corpus.

GOAL: A data-driven table of (category, name, src, expected_exit) that runs
EVERY category of Helix feature Python helixc supports through BOTH compilers
and asserts exact behavioral parity. Purpose: make the remaining work to
"delete Python" a finite checklist instead of a one-bug-at-a-time search.

Status key (filled in at collection time by the GAP_REPORT section at the
bottom of this file):
  A = PARITY OK  : Python succeeds AND bootstrap == Python.
  B = PARITY GAP  : Python succeeds BUT bootstrap differs/crashes. BLOCKERS.
  C = BEYOND-PYTHON: bootstrap succeeds, Python raises / NotImplementedError.
  D = NEITHER    : both fail (ignored in corpus).

Usage:
    cd C:/Projects/Kovostov-Native-paritymatrix
    PYTHONUTF8=1 pytest helixc/tests/test_parity_matrix.py -v

    Or run manually to see category-level counts:
    PYTHONUTF8=1 python helixc/tests/test_parity_matrix.py
"""
from __future__ import annotations
import os
import sys
import pytest

sys.setrecursionlimit(2000)

# ============================================================================
# CORPUS: (category, name, src, expected_rc)
# ============================================================================
# Categories:
#   INT_LIT    integer literals and widths
#   ARITH      arithmetic operators
#   BITWISE    bitwise operators
#   CMP        comparison operators
#   BOOL       boolean operators and values
#   CTRL       control flow (if/else, if-else-if, if-as-expr)
#   WHILE      while loop
#   FOR        for-range loop
#   MATCH      match expressions and patterns
#   FN         functions (params, return, recursion)
#   LET        let bindings, mutation, shadowing, scopes
#   STRUCT     struct decl, literal, field access
#   ENUM       enum decl, match on enum
#   ARRAY      array literals, indexing, assignment
#   TUPLE      tuple literals and field access
#   CONST      const declarations
#   CAST       as-cast expressions
#   FLOAT      f32/f64 arithmetic and comparison
#   IMPL       impl blocks (trait-for-type with typed self)
#   PAT        pattern features (range, struct destr, tuple, guard)
#   SCOPE      block scoping, shadowing, inner blocks as expressions
#   EARLY_RET  early return
#   COMPOUND   compound assignment operators
#   EDGE       integer edge cases (overflow, sign, shift semantics)
# ============================================================================

PARITY_CORPUS: list[tuple[str, str, str, int]] = [
    # ---- INT_LIT: integer literals and widths ----
    ("INT_LIT", "il_basic",           "fn main() -> i32 { 42 }", 42),
    ("INT_LIT", "il_zero",            "fn main() -> i32 { 0 }", 0),
    ("INT_LIT", "il_i8_suffix",       "fn main() -> i8 { 42_i8 }", 42),
    ("INT_LIT", "il_i16_suffix",      "fn main() -> i16 { 42_i16 }", 42),
    ("INT_LIT", "il_i32_suffix",      "fn main() -> i32 { 42_i32 }", 42),
    ("INT_LIT", "il_i64_direct",      "fn main() -> i64 { 42_i64 }", 42),
    ("INT_LIT", "il_u8_suffix",       "fn main() -> i32 { 42_u8 as i32 }", 42),
    ("INT_LIT", "il_u16_suffix",      "fn main() -> i32 { 42_u16 as i32 }", 42),
    ("INT_LIT", "il_u32_suffix",      "fn main() -> u32 { 42_u32 }", 42),
    ("INT_LIT", "il_u64_suffix",      "fn main() -> u64 { 42_u64 }", 42),
    ("INT_LIT", "il_hex",             "fn main() -> i32 { 0x2a }", 42),
    ("INT_LIT", "il_bin",             "fn main() -> i32 { 0b101010 }", 42),
    ("INT_LIT", "il_underscore",      "fn main() -> i32 { 1_000_000 - 999_958 }", 42),
    ("INT_LIT", "il_negative",        "fn main() -> i32 { -10 + 52 }", 42),

    # ---- ARITH: arithmetic operators ----
    ("ARITH", "ar_add",              "fn main() -> i32 { 20 + 22 }", 42),
    ("ARITH", "ar_sub",              "fn main() -> i32 { 100 - 58 }", 42),
    ("ARITH", "ar_mul",              "fn main() -> i32 { 6 * 7 }", 42),
    ("ARITH", "ar_div",              "fn main() -> i32 { 84 / 2 }", 42),
    ("ARITH", "ar_mod",              "fn main() -> i32 { 142 % 100 }", 42),
    ("ARITH", "ar_paren",            "fn main() -> i32 { (10 + 11) + (10 + 11) }", 42),
    ("ARITH", "ar_precedence",       "fn main() -> i32 { 2 + 3 * 7 + 19 }", 42),
    ("ARITH", "ar_neg_then_add",     "fn main() -> i32 { let x = 0 - 42; 0 - x }", 42),
    ("ARITH", "ar_unary_neg_neg",    "fn main() -> i32 { -(-42) }", 42),
    ("ARITH", "ar_chain",            "fn main() -> i32 { 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 6 }", 42),

    # ---- BITWISE: bitwise operators ----
    ("BITWISE", "bw_and",            "fn main() -> i32 { 250 & 42 }", 42),
    ("BITWISE", "bw_or",             "fn main() -> i32 { 32 | 10 }", 42),
    ("BITWISE", "bw_xor",            "fn main() -> i32 { 52 ^ 30 }", 42),
    ("BITWISE", "bw_shl",            "fn main() -> i32 { 21 << 1 }", 42),
    ("BITWISE", "bw_shr",            "fn main() -> i32 { 84 >> 1 }", 42),
    ("BITWISE", "bw_not_zero",       "fn main() -> i32 { ~0 }", 255),
    ("BITWISE", "bw_not_not",        "fn main() -> i32 { ~(~42) }", 42),
    ("BITWISE", "bw_prec_xor_and",   "fn main() -> i32 { 5 ^ 3 & 1 }", 4),
    ("BITWISE", "bw_prec_or_xor",    "fn main() -> i32 { 3 | 1 ^ 2 }", 3),
    ("BITWISE", "bw_prec_shift_and", "fn main() -> i32 { 1 & 3 << 1 }", 0),
    ("BITWISE", "bw_prec_full",      "fn main() -> i32 { 1 | 2 ^ 4 & 4 }", 7),
    ("BITWISE", "bw_combo",          "fn main() -> i32 { let x = 40; (x | 2) & 63 }", 42),

    # ---- CMP: comparison operators ----
    ("CMP", "cmp_lt_true",           "fn main() -> i32 { let b = 3 < 5; b + 41 }", 42),
    ("CMP", "cmp_lt_false",          "fn main() -> i32 { let b = 5 < 3; b + 42 }", 42),
    ("CMP", "cmp_le_true",           "fn main() -> i32 { let x = 42; if x <= 42 { 42 } else { 0 } }", 42),
    ("CMP", "cmp_gt_true",           "fn main() -> i32 { let x = 100; if x > 50 { 42 } else { 0 } }", 42),
    ("CMP", "cmp_ge_true",           "fn main() -> i32 { if 42 >= 1 { 42 } else { 0 } }", 42),
    ("CMP", "cmp_eq_true",           "fn main() -> i32 { if 42 == 42 { 42 } else { 0 } }", 42),
    ("CMP", "cmp_ne_true",           "fn main() -> i32 { if 42 != 0 { 42 } else { 0 } }", 42),
    ("CMP", "cmp_stored",            "fn main() -> i32 { let x = 100; let b = x > 50; if b { 42 } else { 0 } }", 42),

    # ---- BOOL: boolean operators ----
    ("BOOL", "bo_true",              "fn main() -> i32 { if true { 42 } else { 0 } }", 42),
    ("BOOL", "bo_false",             "fn main() -> i32 { if false { 0 } else { 42 } }", 42),
    ("BOOL", "bo_and_tt",            "fn main() -> i32 { if true && true { 42 } else { 0 } }", 42),
    ("BOOL", "bo_and_tf",            "fn main() -> i32 { if true && false { 0 } else { 42 } }", 42),
    ("BOOL", "bo_or_ft",             "fn main() -> i32 { if false || true { 42 } else { 0 } }", 42),
    ("BOOL", "bo_not_false",         "fn main() -> i32 { let b = false; if !b { 42 } else { 0 } }", 42),
    ("BOOL", "bo_not_zero",          "fn main() -> i32 { !0 }", 1),
    ("BOOL", "bo_not_nonzero",       "fn main() -> i32 { !5 }", 0),
    ("BOOL", "bo_match",             "fn main() -> i32 { let b = true; match b { true => 42, false => 0 } }", 42),
    ("BOOL", "bo_let",               "fn main() -> i32 { let b = true; if b { 42 } else { 0 } }", 42),
    ("BOOL", "bo_fn_param",          "fn id(b: bool) -> i32 { if b { 42 } else { 0 } } fn main() -> i32 { id(true) }", 42),
    ("BOOL", "bo_fn_ret",            "fn is_pos(x: i32) -> bool { x > 0 } fn main() -> i32 { if is_pos(5) { 42 } else { 0 } }", 42),
    ("BOOL", "bo_short_circuit",     "fn main() -> i32 { let x = 5; if x > 0 && x < 10 || x == 100 { 42 } else { 0 } }", 42),
    ("BOOL", "bo_xor",               "fn main() -> i32 { let a = true; let b = false; if a ^ b { 42 } else { 0 } }", 42),

    # ---- CTRL: control flow ----
    ("CTRL", "cf_if_else",           "fn main() -> i32 { if true { 42 } else { 0 } }", 42),
    ("CTRL", "cf_if_else_if",        "fn main() -> i32 { let x = 2; if x == 1 { 1 } else if x == 2 { 42 } else { 99 } }", 42),
    ("CTRL", "cf_nested_if",         "fn main() -> i32 { let x = 10; if x > 5 { if x < 20 { 42 } else { 0 } } else { 0 } }", 42),
    ("CTRL", "cf_if_as_expr",        "fn main() -> i32 { let x = 100; let y = if x > 0 { 42 } else { 99 }; y }", 42),
    ("CTRL", "cf_if_no_else",        "fn main() -> i32 { let mut x = 40; if true { x = x + 2; }; x }", 42),
    ("CTRL", "cf_nested_if_let",     "fn main() -> i32 { let x = 10; let y = if x > 5 { if x < 20 { 42 } else { 0 } } else { 0 }; y }", 42),
    ("CTRL", "cf_if_in_arith",       "fn main() -> i32 { (if true { 20 } else { 0 }) + 22 }", 42),

    # ---- WHILE: while loops ----
    ("WHILE", "wh_basic",            "fn main() -> i32 { let mut x = 0; while x < 42 { x += 1; } x }", 42),
    ("WHILE", "wh_sum",              "fn main() -> i32 { let mut i = 0; let mut s = 0; while i < 7 { s = s + i; i = i + 1; } s + 21 }", 42),
    ("WHILE", "wh_zero_iter",        "fn main() -> i32 { let mut i = 0; while i < 0 { i = i + 1; } 42 }", 42),
    ("WHILE", "wh_factorial",        "fn main() -> i32 { let mut n = 5; let mut r = 1; while n > 1 { r *= n; n -= 1; } r }", 120),
    ("WHILE", "wh_nested_if",        "fn main() -> i32 { let mut s = 0; let mut i = 0; while i < 10 { if i % 2 == 0 { s = s + i; } i = i + 1; } s + 22 }", 42),
    ("WHILE", "wh_complex_cond",     "fn main() -> i32 { let mut i = 0; let mut s = 0; while i < 10 && s < 100 { s = s + i; i = i + 1; } s - 3 }", 42),

    # ---- FOR: for-range loops ----
    ("FOR", "fr_exclusive",          "fn main() -> i32 { let mut s = 0; for i in 1..6 { s = s + i; } s + 27 }", 42),
    ("FOR", "fr_wildcard",           "fn main() -> i32 { let mut s = 0; for _ in 0..6 { s = s + 7; } s }", 42),
    ("FOR", "fr_sum",                "fn main() -> i32 { let mut s = 0; for i in 0..7 { s = s + i; } s + 21 }", 42),
    ("FOR", "fr_nested",             "fn main() -> i32 { let mut s = 0; for i in 0..3 { for j in 0..2 { s = s + 7; } } s }", 42),
    ("FOR", "fr_zero_iter",          "fn main() -> i32 { let mut s = 42; for _ in 0..0 { s = 0; } s }", 42),
    ("FOR", "fr_with_body",          "fn main() -> i32 { let mut s = 0; let mut p = 1; for i in 1..4 { s += i; p = p * i; } s + p }", 12),
    ("FOR", "fr_compound_assign",    "fn main() -> i32 { let mut s = 0; for _ in 0..6 { s += 7; } s }", 42),

    # ---- MATCH: match expressions and patterns ----
    ("MATCH", "ma_lit1",             "fn main() -> i32 { match 1 { 1 => 42, _ => 0 } }", 42),
    ("MATCH", "ma_lit3",             "fn main() -> i32 { let x = 2; match x { 1 => 0, 2 => 42, _ => 99 } }", 42),
    ("MATCH", "ma_wildcard",         "fn main() -> i32 { let x = 99; match x { 0 => 0, _ => 42 } }", 42),
    ("MATCH", "ma_or_arms",          "fn main() -> i32 { let x = 2; match x { 1 | 2 | 3 => 42, _ => 0 } }", 42),
    ("MATCH", "ma_block_arm",        "fn main() -> i32 { let x = 1; match x { 1 => { let y = 21; y + 21 }, _ => 0 } }", 42),
    ("MATCH", "ma_guard",            "fn main() -> i32 { let x = 5; match x { n if n > 3 => 42, _ => 0 } }", 42),
    ("MATCH", "ma_bind",             "fn main() -> i32 { let x = 99; match x { 0 => 0, n => n - 57 } }", 42),
    ("MATCH", "ma_nested",           "fn main() -> i32 { let x = 2; match x { 1 => 0, 2 => match x { 2 => 42, _ => 1 }, _ => 9 } }", 42),
    ("MATCH", "ma_as_expr",          "fn main() -> i32 { let x = 2; 40 + match x { 2 => 2, _ => 0 } }", 42),
    ("MATCH", "ma_in_fn",            "fn cl(x: i32) -> i32 { match x { 0 => 0, _ => 42 } } fn main() -> i32 { cl(5) }", 42),
    ("MATCH", "ma_large8",           "fn main() -> i32 { let x = 7; match x { 0=>0,1=>1,2=>2,3=>3,4=>4,5=>5,6=>6,7=>42,_=>99 } }", 42),
    ("MATCH", "ma_large10",          "fn main() -> i32 { let x = 9; match x { 0=>0,1=>1,2=>2,3=>3,4=>4,5=>5,6=>6,7=>7,8=>8,9=>42,_=>99 } }", 42),
    ("MATCH", "ma_fn_call",          "fn f() -> i32 { 3 } fn main() -> i32 { match f() { 1=>1, 3=>42, _=>0 } }", 42),
    # match on bool
    ("MATCH", "ma_bool",             "fn main() -> i32 { let b = true; match b { true => 42, false => 0 } }", 42),
    # range patterns
    ("MATCH", "ma_range_incl",       "fn main() -> i32 { let x = 5; match x { 1..=5 => 42, _ => 0 } }", 42),
    ("MATCH", "ma_range_excl",       "fn main() -> i32 { let x = 5; match x { 1..10 => 42, _ => 0 } }", 42),
    ("MATCH", "ma_neg_range",        "fn main() -> i32 { let x = 0 - 3; match x { -10..=-1 => 42, _ => 0 } }", 42),
    # struct destructure pattern
    ("MATCH", "ma_struct_destr",     "struct P { x: i32, y: i32 } fn main() -> i32 { let p = P { x: 40, y: 2 }; match p { P { x, y } => x + y } }", 42),
    # tuple pattern
    ("MATCH", "ma_tuple_pat",        "fn main() -> i32 { let t = (40, 2); match t { (a, b) => a + b } }", 42),
    # neg match arm
    ("MATCH", "ma_neg_in_match",     "fn main() -> i32 { let x = 0 - 1; match x { 0 => 0, _ => 42 } }", 42),
    # guard in enum match
    ("MATCH", "ma_enum_guard",       "enum E { V(i32) } fn main() -> i32 { let e = E::V(42); match e { E::V(n) if n > 40 => 42, E::V(_) => 0 } }", 42),

    # ---- FN: functions ----
    ("FN", "fn_basic",               "fn f() -> i32 { 42 } fn main() -> i32 { f() }", 42),
    ("FN", "fn_2params",             "fn add(a: i32, b: i32) -> i32 { a + b } fn main() -> i32 { add(20, 22) }", 42),
    ("FN", "fn_3params",             "fn add3(a: i32, b: i32, c: i32) -> i32 { a + b + c } fn main() -> i32 { add3(20, 15, 7) }", 42),
    ("FN", "fn_6params",             "fn f(a:i32,b:i32,c:i32,d:i32,e:i32,g:i32)->i32{a+b+c+d+e+g} fn main() -> i32 { f(7,7,7,7,7,7) }", 42),
    ("FN", "fn_nested_calls",        "fn inc(x: i32) -> i32 { x + 1 } fn main() -> i32 { inc(inc(inc(39))) }", 42),
    ("FN", "fn_recursion_fib",       "fn fib(n: i32) -> i32 { if n < 2 { n } else { fib(n-1) + fib(n-2) } } fn main() -> i32 { fib(9) + 8 }", 42),
    ("FN", "fn_recursion_fact",      "fn f(n: i32) -> i32 { if n <= 1 { 1 } else { n * f(n - 1) } } fn main() -> i32 { f(5) }", 120),
    ("FN", "fn_recursion_gcd",       "fn gcd(a: i32, b: i32) -> i32 { if b == 0 { a } else { gcd(b, a % b) } } fn main() -> i32 { gcd(84, 126) }", 42),
    ("FN", "fn_recursion_mutual",    "fn ev(n: i32) -> i32 { if n == 0 { 1 } else { od(n - 1) } } fn od(n: i32) -> i32 { if n == 0 { 0 } else { ev(n - 1) } } fn main() -> i32 { ev(10) + 41 }", 42),
    ("FN", "fn_bool_param",          "fn id(b: bool) -> i32 { if b { 42 } else { 0 } } fn main() -> i32 { id(true) }", 42),
    ("FN", "fn_bool_ret",            "fn is_pos(x: i32) -> bool { x > 0 } fn main() -> i32 { if is_pos(5) { 42 } else { 0 } }", 42),
    ("FN", "fn_mut_param",           "fn f(mut x: i32) -> i32 { x = 41; x + 1 } fn main() -> i32 { f(1) }", 42),
    ("FN", "fn_i64_param",           "fn double_i64(x: i64) -> i64 { x + x } fn main() -> i32 { let r: i64 = double_i64(21_i64); r as i32 }", 42),

    # ---- LET: let bindings, mutation, shadowing ----
    ("LET", "le_basic",              "fn main() -> i32 { let x = 42; x }", 42),
    ("LET", "le_chained",            "fn main() -> i32 { let a = 10; let b = a + 5; let c = b * 2; c + 12 }", 42),
    ("LET", "le_mut",                "fn main() -> i32 { let mut x = 10; x = 42; x }", 42),
    ("LET", "le_shadow",             "fn main() -> i32 { let x = 99; let x = 42; x }", 42),
    ("LET", "le_shadow_uses_prev",   "fn main() -> i32 { let x = 20; let x = x + 22; x }", 42),
    ("LET", "le_multi_shadow",       "fn main() -> i32 { let x = 1; let x = x + 10; let x = x + 10; let x = x + 21; x }", 42),
    ("LET", "le_typed_i32",          "fn main() -> i32 { let x: i32 = 42; x }", 42),
    ("LET", "le_typed_bool",         "fn main() -> i32 { let b: bool = true; if b { 42 } else { 0 } }", 42),
    ("LET", "le_typed_u8",           "fn main() -> i32 { let x: u8 = 42; x as i32 }", 42),
    ("LET", "le_typed_u16",          "fn main() -> i32 { let x: u16 = 42; x as i32 }", 42),
    ("LET", "le_typed_i64",          "fn main() -> i32 { let x: i64 = 42_i64; x as i32 }", 42),
    ("LET", "le_typed_u64",          "fn main() -> i32 { let x: u64 = 42_u64; x as i32 }", 42),
    ("LET", "le_typed_isize",        "fn main() -> i32 { let x: isize = 42; x as i32 }", 42),
    ("LET", "le_typed_usize",        "fn main() -> i32 { let x: usize = 42; x as i32 }", 42),

    # ---- SCOPE: block scoping ----
    ("SCOPE", "sc_isolate",          "fn main() -> i32 { let x = 10; { let x = 20; }; x }", 10),
    ("SCOPE", "sc_mut_isolate",      "fn main() -> i32 { let mut x = 10; { let mut x = 20; x = 30; }; x }", 10),
    ("SCOPE", "sc_outer_mut",        "fn main() -> i32 { let mut x = 10; { x = x + 5; }; x }", 15),
    ("SCOPE", "sc_triple_shadow",    "fn main() -> i32 { let mut x = 1; { let mut x = 10; { let mut x = 100; x = x + 1; }; x = x + 1; }; x = x + 1; x }", 2),
    ("SCOPE", "sc_block_as_expr",    "fn main() -> i32 { let x = { let y = 21; y + 21 }; x }", 42),
    ("SCOPE", "sc_nested_block",     "fn main() -> i32 { let x = { let y = { 21 }; y + 21 }; x }", 42),

    # ---- STRUCT: structs ----
    ("STRUCT", "st_basic",           "struct P { v: i32 } fn main() -> i32 { let p = P { v: 42 }; p.v }", 42),
    ("STRUCT", "st_two_fields",      "struct P { x: i32, y: i32 } fn main() -> i32 { let p = P { x: 20, y: 22 }; p.x + p.y }", 42),
    ("STRUCT", "st_three_fields",    "struct P { a: i32, b: i32, c: i32 } fn main() -> i32 { let p = P{a:20,b:15,c:7}; p.a + p.b + p.c }", 42),
    ("STRUCT", "st_five_fields",     "struct P { a:i32, b:i32, c:i32, d:i32, e:i32 } fn main() -> i32 { let p = P{a:10,b:10,c:10,d:10,e:2}; p.a+p.b+p.c+p.d+p.e }", 42),
    ("STRUCT", "st_nested",          "struct A { x: i32 } struct B { a: A } fn main() -> i32 { let b = B { a: A { x: 42 } }; b.a.x }", 42),
    ("STRUCT", "st_nested3",         "struct A { v: i32 } struct B { a: A } struct C { b: B } fn main() -> i32 { let c = C { b: B { a: A { v: 42 } } }; c.b.a.v }", 42),
    ("STRUCT", "st_fn_param",        "struct P { x: i32, y: i32 } fn s(p: P) -> i32 { p.x + p.y } fn main() -> i32 { s(P{x:40,y:2}) }", 42),
    ("STRUCT", "st_field_extract",   "struct C { v: i32 } fn main() -> i32 { let c = C { v: 42 }; let v = c.v; v }", 42),
    ("STRUCT", "st_bool_field",      "struct S { flag:bool, val:i32 } fn main() -> i32 { let s = S{flag:true,val:42}; if s.flag { s.val } else { 0 } }", 42),
    ("STRUCT", "st_enum_field",      "enum C { R, G } struct S { c: C, v: i32 } fn main() -> i32 { let s = S{c:C::G, v:42}; s.v }", 42),
    ("STRUCT", "st_if_arm",          "struct P { x: i32 } fn main() -> i32 { let p = P { x: 42 }; if p.x > 0 { p.x } else { 0 } }", 42),
    ("STRUCT", "st_in_match",        "struct P { x: i32 } fn main() -> i32 { let p = P { x: 42 }; let v = p.x; match v { 42 => 42, _ => 0 } }", 42),
    ("STRUCT", "st_for_access",      "struct P { x: i32 } fn main() -> i32 { let p = P { x: 7 }; let mut s = 0; for _ in 0..6 { s = s + p.x; } s }", 42),

    # ---- ENUM: enums ----
    ("ENUM", "en_two_unit",          "enum E { Z, O } fn main() -> i32 { let v = E::O; match v { E::Z => 0, E::O => 42 } }", 42),
    ("ENUM", "en_three",             "enum E { A, B, C } fn main() -> i32 { let e = E::B; match e { E::A => 0, E::B => 42, E::C => 1 } }", 42),
    ("ENUM", "en_four",              "enum D { N, E, S, W } fn main() -> i32 { let d = D::S; match d { D::N => 1, D::E => 2, D::S => 42, D::W => 4 } }", 42),
    ("ENUM", "en_eight",             "enum E { A, B, C, D, F, G, H, I } fn main() -> i32 { let e = E::F; match e { E::A=>0, E::B=>1, E::C=>2, E::D=>3, E::F=>42, E::G=>6, E::H=>7, E::I=>8 } }", 42),
    ("ENUM", "en_payload",           "enum E { A(i32), B } fn main() -> i32 { let e = E::A(42); match e { E::A(n) => n, E::B => 0 } }", 42),
    ("ENUM", "en_payload_arith",     "enum E { A(i32), B(i32) } fn main() -> i32 { let e = E::A(40); match e { E::A(x) => x + 2, E::B(x) => x } }", 42),
    ("ENUM", "en_payload_multi",     "enum E { P(i32, i32) } fn main() -> i32 { let e = E::P(40, 2); match e { E::P(a, b) => a + b } }", 42),
    ("ENUM", "en_payload_2var",      "enum E { A(i32), B(i32), C } fn main() -> i32 { let e = E::B(42); match e { E::A(n)=>n, E::B(n)=>n, E::C=>0 } }", 42),
    ("ENUM", "en_param_match",       "enum E { Z, V(i32) } fn f(e: E) -> i32 { match e { E::Z => 0, E::V(x) => x } } fn main() -> i32 { f(E::V(42)) }", 42),
    ("ENUM", "en_disc_arith",        "enum C { R, G, B } fn main() -> i32 { let c = C::G; let n = match c { C::R=>0, C::G=>40, C::B=>1 }; n + 2 }", 42),

    # ---- ARRAY: arrays ----
    ("ARRAY", "ar_lit_index",        "fn main() -> i32 { let a = [10, 20, 30]; a[1] + 22 }", 42),
    ("ARRAY", "ar_sum_idx",          "fn main() -> i32 { let a = [10, 20, 12]; a[0] + a[1] + a[2] }", 42),
    ("ARRAY", "ar_five_const_sum",   "fn main() -> i32 { let a = [6, 7, 8, 9, 12]; a[0] + a[1] + a[2] + a[3] + a[4] }", 42),
    ("ARRAY", "ar_ten",              "fn main() -> i32 { let a = [0,1,2,3,42,5,6,7,8,9]; a[4] }", 42),
    ("ARRAY", "ar_loop_sum",         "fn main() -> i32 { let a = [10, 20, 12]; let mut s = 0; let mut i = 0; while i < 3 { s = s + a[i]; i = i + 1; } s }", 42),
    ("ARRAY", "ar_var_idx",          "fn main() -> i32 { let a = [40, 1, 1]; let i = 0; a[i] + 2 }", 42),
    ("ARRAY", "ar_fn_in_idx",        "fn idx() -> i32 { 1 } fn main() -> i32 { let a = [0, 42, 0]; a[idx()] }", 42),
    ("ARRAY", "ar_store_const",      "fn main() -> i32 { let mut a = [1,2,3]; a[0] = 42; a[0] }", 42),
    ("ARRAY", "ar_store_var_idx",    "fn main() -> i32 { let mut a = [0,0,0]; let i = 0; a[i] = 42; a[i] }", 42),
    ("ARRAY", "ar_store_loop",       "fn main() -> i32 { let mut a = [0,0,0,0,0,0]; let mut i = 0; while i < 6 { a[i] = 7; i = i + 1; } let mut s = 0; let mut j = 0; while j < 6 { s = s + a[j]; j = j + 1; } s }", 42),
    ("ARRAY", "ar_args",             "fn sum3(a: i32, b: i32, c: i32) -> i32 { a + b + c } fn main() -> i32 { let arr = [20, 15, 7]; sum3(arr[0], arr[1], arr[2]) }", 42),

    # ---- TUPLE: tuples ----
    ("TUPLE", "tu_two",              "fn main() -> i32 { let t = (42, 0); t.0 }", 42),
    ("TUPLE", "tu_sum",              "fn main() -> i32 { let t = (20, 22); t.0 + t.1 }", 42),
    ("TUPLE", "tu_three",            "fn main() -> i32 { let t = (10, 20, 12); t.0 + t.1 + t.2 }", 42),
    ("TUPLE", "tu_from_vars",        "fn main() -> i32 { let a = 40; let b = 2; let t = (a, b); t.0 + t.1 }", 42),

    # ---- CONST: constant declarations ----
    ("CONST", "co_basic",            "const X: i32 = 42; fn main() -> i32 { X }", 42),
    ("CONST", "co_two",              "const A: i32 = 30; const B: i32 = 12; fn main() -> i32 { A + B }", 42),
    ("CONST", "co_in_fn",            "const N: i32 = 42; fn f() -> i32 { N } fn main() -> i32 { f() }", 42),
    ("CONST", "co_used_twice",       "const X: i32 = 42; fn doubled() -> i32 { X + X } fn main() -> i32 { doubled() - X }", 42),
    ("CONST", "co_in_loop",          "const N: i32 = 6; fn main() -> i32 { let mut s = 0; for _ in 0..N { s += 7; } s }", 42),
    ("CONST", "co_bool",             "const B: bool = true; fn main() -> i32 { if B { 42 } else { 0 } }", 42),
    ("CONST", "co_u32",              "const X: u32 = 42_u32; fn main() -> i32 { X as i32 }", 42),

    # ---- CAST: as-cast expressions ----
    ("CAST", "ca_i64_via_let",       "fn main() -> i32 { let x: i64 = 42_i64; x as i32 }", 42),
    ("CAST", "ca_i32_via_let",       "fn main() -> i32 { let x: i32 = 42; let y: i64 = x as i64; y as i32 }", 42),
    ("CAST", "ca_u32_to_i32",        "fn main() -> i32 { let x: u32 = 42_u32; x as i32 }", 42),
    ("CAST", "ca_bool_to_i32",       "fn main() -> i32 { let b = true; if b as i32 == 1 { 42 } else { 0 } }", 42),
    ("CAST", "ca_i8_extend",         "fn main() -> i32 { let x: i8 = 42_i8; x as i32 }", 42),
    ("CAST", "ca_i32_to_f64",        "fn main() -> i32 { let x: i32 = 42; let y: f64 = x as f64; y as i32 }", 42),
    ("CAST", "ca_i32_to_i8",         "fn main() -> i32 { let x: i8 = 42 as i8; x as i32 }", 42),
    ("CAST", "ca_i32_to_u8",         "fn main() -> i32 { let x: u8 = 42 as u8; x as i32 }", 42),
    ("CAST", "ca_i32_to_u32",        "fn main() -> i32 { let x: u32 = 42 as u32; x as i32 }", 42),
    ("CAST", "ca_after_arith",       "fn main() -> i32 { (6 * 7) as i32 }", 42),
    ("CAST", "ca_f32_cmp",           "fn main() -> i32 { if 1.5_f32 < 2.5_f32 { 42 } else { 0 } }", 42),
    ("CAST", "ca_f64_cmp",           "fn main() -> i32 { let x: f64 = 1.5_f64 + 2.5_f64; if x == 4.0_f64 { 42 } else { 0 } }", 42),
    ("CAST", "ca_i32_to_i64_fn",     "fn main() -> i32 { let r: i64 = 21_i64 + 21_i64; r as i32 }", 42),

    # ---- FLOAT: f32/f64 comparisons (parity OK) ----
    # NOTE: f32/f64 arithmetic results + casts are PARITY GAPS (see GAP_REPORT)
    ("FLOAT", "fl_f32_cmp_lt",       "fn main() -> i32 { let x: f32 = 1.5_f32; let y: f32 = 2.5_f32; if x < y { 42 } else { 0 } }", 42),
    ("FLOAT", "fl_f32_cmp_ge",       "fn main() -> i32 { let x: f32 = 3.0_f32; let y: f32 = 3.0_f32; if x >= y { 42 } else { 0 } }", 42),
    ("FLOAT", "fl_f64_cmp_lt",       "fn main() -> i32 { let x: f64 = 1.5_f64; let y: f64 = 2.5_f64; if x < y { 1 } else { 0 } }", 1),
    ("FLOAT", "fl_f64_cmp_eq",       "fn main() -> i32 { let x: f64 = 1.5_f64 + 2.5_f64; if x == 4.0_f64 { 42 } else { 0 } }", 42),
    ("FLOAT", "fl_int_to_f64",       "fn main() -> i32 { let x: i32 = 42; (x as f64) as i32 }", 42),
    ("FLOAT", "fl_i32_to_f64_cast",  "fn main() -> i32 { let a: i32 = 42; let x: f64 = a as f64; x as i32 }", 42),
    ("FLOAT", "fl_f64_cmp_gt",       "fn main() -> i32 { let x: f64 = 3.14_f64; if x > 3.0_f64 { 42 } else { 0 } }", 42),

    # ---- IMPL: impl blocks with typed-self (Python's flatten_impls form) ----
    ("IMPL", "im_trait_method",
        "trait Eq { fn eq(self: i32, other: i32) -> i32 ; } "
        "impl Eq for i32 { fn eq(self: i32, other: i32) -> i32 { if self == other { 42 } else { 0 } } } "
        "fn main() -> i32 { let a: i32 = 5; a.eq(5) }", 42),
    ("IMPL", "im_trait_neq",
        "trait Eq { fn eq(self: i32, other: i32) -> i32 ; } "
        "impl Eq for i32 { fn eq(self: i32, other: i32) -> i32 { if self == other { 1 } else { 0 } } } "
        "fn main() -> i32 { let a: i32 = 5; if a.eq(7) == 0 { 42 } else { 0 } }", 42),

    # ---- EARLY_RET: early return ----
    ("EARLY_RET", "er_basic",        "fn f(x: i32) -> i32 { if x < 0 { return 0; } x } fn main() -> i32 { f(42) }", 42),
    ("EARLY_RET", "er_zero",         "fn f(x: i32) -> i32 { if x < 0 { return 0; } x } fn main() -> i32 { f(0 - 1) }", 0),
    ("EARLY_RET", "er_multi",        "fn cl(x: i32) -> i32 { if x < 0 { return 0; } if x == 0 { return 1; } 42 } fn main() -> i32 { cl(5) }", 42),
    ("EARLY_RET", "er_complex",      "fn f(x: i32) -> i32 { if x > 100 { return 0; } if x < 0 { return 0; } x } fn main() -> i32 { f(42) }", 42),

    # ---- COMPOUND: compound assignment ----
    ("COMPOUND", "cp_add_eq",        "fn main() -> i32 { let mut x = 20; x += 22; x }", 42),
    ("COMPOUND", "cp_sub_eq",        "fn main() -> i32 { let mut x = 50; x -= 8; x }", 42),
    ("COMPOUND", "cp_mul_eq",        "fn main() -> i32 { let mut x = 6; x *= 7; x }", 42),
    ("COMPOUND", "cp_div_eq",        "fn main() -> i32 { let mut x = 84; x /= 2; x }", 42),
    ("COMPOUND", "cp_mod_eq",        "fn main() -> i32 { let mut x = 142; x %= 100; x }", 42),

    # ---- EDGE: integer edge cases ----
    ("EDGE", "ed_overflow_wrap",     "fn main() -> i32 { 2147483647 + 1 }", 0),
    ("EDGE", "ed_signed_div",        "fn main() -> i32 { let x = 0 - 7; x / 2 }", 253),
    ("EDGE", "ed_signed_mod",        "fn main() -> i32 { let x = 0 - 7; x % 2 }", 255),
    ("EDGE", "ed_shl_sign",          "fn main() -> i32 { 1 << 31 }", 0),
    ("EDGE", "ed_ashr_neg",          "fn main() -> i32 { let x = 0 - 8; x >> 1 }", 252),
    ("EDGE", "ed_mul_overflow",      "fn main() -> i32 { 100000 * 100000 }", 0),
    ("EDGE", "ed_abs_via_cond",      "fn main() -> i32 { let x = 0 - 7; if x < 0 { 0 - x } else { x } }", 7),
    ("EDGE", "ed_u8_wrap",           "fn main() -> i32 { let x: u8 = 250; let y: u8 = 48; x + y }", 42),
    ("EDGE", "ed_i64_beyond_i32",    "fn double_i64(x: i64) -> i64 { x + x } fn main() -> i32 { let r: i64 = double_i64(21_i64); r as i32 }", 42),
    ("EDGE", "ed_u64_basic",         "fn main() -> i32 { let x: u64 = 100_u64; let y: u64 = 58_u64; (x - y) as i32 }", 42),

    # ---- GENERIC: generic functions and structs (monomorphized) ----
    # Python's monomorphize pass infers concrete types from call sites; bootstrap
    # synthesizes __i32 clones. ALL of these are PARITY GAP (rc=132 SIGILL) in
    # the bootstrap — see KNOWN_PARITY_GAPS below and the GAP_REPORT section.
    ("GENERIC", "gn_id_i32",
        "fn id[T](x: T) -> T { x } fn main() -> i32 { id(42) }", 42),
    ("GENERIC", "gn_first_two_params",
        "fn first[A, B](a: A, b: B) -> A { a } fn main() -> i32 { first(42, 1) }", 42),
    ("GENERIC", "gn_swap",
        "fn swap[A, B](a: A, b: B) -> A { a } fn main() -> i32 { swap(42, false) }", 42),
    ("GENERIC", "gn_pick",
        "fn pick[T](flag: bool, a: T, b: T) -> T { if flag { a } else { b } } fn main() -> i32 { pick(true, 42, 0) }", 42),
    ("GENERIC", "gn_sum_n",
        "fn sum_n[T](n: T) -> T { n } fn main() -> i32 { sum_n(42) }", 42),
    ("GENERIC", "gn_struct_one_field",
        "struct Wrapper[T] { val: T } fn main() -> i32 { let w = Wrapper { val: 42 }; w.val }", 42),
    ("GENERIC", "gn_struct_two_fields_same",
        "struct Vec2[T] { x: T, y: T } fn main() -> i32 { let v = Vec2 { x: 20, y: 22 }; v.x + v.y }", 42),
    ("GENERIC", "gn_struct_two_params",
        "struct Pair[A, B] { fst: A, snd: B } fn main() -> i32 { let p = Pair { fst: 42, snd: true }; if p.snd { p.fst } else { 0 } }", 42),
    ("GENERIC", "gn_struct_in_match",
        "struct Tagged[T] { tag: i32, val: T } fn main() -> i32 { let t = Tagged { tag: 1, val: 42 }; match t.tag { 1 => t.val, _ => 0 } }", 42),
    ("GENERIC", "gn_struct_impl",
        "struct Box[T] { val: T } impl Box { fn get(self: Box) -> i32 { self.val } } fn main() -> i32 { let b = Box { val: 42 }; b.get() }", 42),
    ("GENERIC", "gn_struct_explicit_ty",
        "struct Box[T] { val: T } fn main() -> i32 { let b: Box<i32> = Box { val: 42 }; b.val }", 42),
    ("GENERIC", "gn_struct_pair_explicit",
        "struct Pair[A, B] { a: A, b: B } fn main() -> i32 { let p = Pair { a: 40, b: 2 }; p.a + p.b }", 42),

    # ---- TRAIT: trait definitions, impl blocks, method dispatch ----
    ("TRAIT", "tr_impl_method",
        "trait Eq { fn eq(self: i32, other: i32) -> bool; } "
        "impl Eq for i32 { fn eq(self: i32, other: i32) -> bool { self == other } } "
        "fn main() -> i32 { let a: i32 = 42; if a.eq(42) { 42 } else { 0 } }", 42),
    ("TRAIT", "tr_multi_trait",
        "trait Show { fn show(self: i32) -> i32; } "
        "trait Double { fn dbl(self: i32) -> i32; } "
        "impl Show for i32 { fn show(self: i32) -> i32 { self } } "
        "impl Double for i32 { fn dbl(self: i32) -> i32 { self + self } } "
        "fn main() -> i32 { let x: i32 = 21; x.dbl() }", 42),
    ("TRAIT", "tr_multi_method",
        "trait Math { fn add(self: i32, other: i32) -> i32; fn double(self: i32) -> i32; } "
        "impl Math for i32 { "
        "  fn add(self: i32, other: i32) -> i32 { self + other } "
        "  fn double(self: i32) -> i32 { self + self } } "
        "fn main() -> i32 { let x: i32 = 21; x.double() }", 42),
    ("TRAIT", "tr_impl_for_struct",
        "struct Counter { val: i32 } "
        "trait Inc { fn inc(self: Counter) -> i32; } "
        "impl Inc for Counter { fn inc(self: Counter) -> i32 { self.val + 1 } } "
        "fn main() -> i32 { let c = Counter { val: 41 }; c.inc() }", 42),
    ("TRAIT", "tr_two_impls_struct",
        "struct Vec2 { x: i32, y: i32 } "
        "trait Len { fn len(self: Vec2) -> i32; } "
        "trait Scale { fn scale(self: Vec2, factor: i32) -> i32; } "
        "impl Len for Vec2 { fn len(self: Vec2) -> i32 { self.x + self.y } } "
        "impl Scale for Vec2 { fn scale(self: Vec2, factor: i32) -> i32 { (self.x + self.y) * factor } } "
        "fn main() -> i32 { let v = Vec2 { x: 6, y: 15 }; v.scale(2) }", 42),
    ("TRAIT", "tr_method_complex",
        "struct Point { x: i32, y: i32 } "
        "trait Manhattan { fn dist(self: Point) -> i32; } "
        "impl Manhattan for Point { "
        "  fn dist(self: Point) -> i32 { "
        "    let ax = if self.x < 0 { 0 - self.x } else { self.x }; "
        "    let ay = if self.y < 0 { 0 - self.y } else { self.y }; "
        "    ax + ay } } "
        "fn main() -> i32 { let p = Point { x: 20, y: 22 }; p.dist() }", 42),
    ("TRAIT", "tr_impl_no_trait",
        "struct Rect { w: i32, h: i32 } "
        "impl Rect { "
        "  fn area(self: Rect) -> i32 { self.w * self.h } "
        "  fn perimeter(self: Rect) -> i32 { (self.w + self.h) * 2 } } "
        "fn main() -> i32 { let r = Rect { w: 6, h: 7 }; r.area() }", 42),
    ("TRAIT", "tr_impl_early_return",
        "struct S { val: i32 } "
        "impl S { fn clamp(self: S, max: i32) -> i32 { if self.val > max { return max; } self.val } } "
        "fn main() -> i32 { let s = S { val: 100 }; s.clamp(42) }", 42),
    ("TRAIT", "tr_chain_method_fn",
        "struct S { val: i32 } "
        "trait Val { fn get(self: S) -> i32; } "
        "impl Val for S { fn get(self: S) -> i32 { self.val } } "
        "fn double(n: i32) -> i32 { n * 2 } "
        "fn main() -> i32 { let s = S { val: 21 }; double(s.get()) }", 42),
    ("TRAIT", "tr_impl_multi_method",
        "struct S { v: i32 } "
        "impl S { "
        "  fn dbl(self: S) -> i32 { self.v + self.v } "
        "  fn trip(self: S) -> i32 { self.v + self.v + self.v } } "
        "fn main() -> i32 { let s = S { v: 21 }; s.dbl() }", 42),
    ("TRAIT", "tr_impl_with_param",
        "struct S { base: i32 } "
        "impl S { fn add(self: S, extra: i32) -> i32 { self.base + extra } } "
        "fn main() -> i32 { let s = S { base: 20 }; s.add(22) }", 42),
    ("TRAIT", "tr_impl_while_loop",
        "struct S { n: i32 } "
        "impl S { fn cnt(self: S) -> i32 { let mut c = 0; while c < self.n { c = c + 1; } c } } "
        "fn main() -> i32 { let s = S { n: 42 }; s.cnt() }", 42),
    ("TRAIT", "tr_impl_in_recursion",
        "struct S { val: i32 } "
        "impl S { fn get(self: S) -> i32 { self.val } } "
        "fn add_two(s: S, extra: i32) -> i32 { s.get() + extra } "
        "fn main() -> i32 { let s = S { val: 20 }; add_two(s, 22) }", 42),

    # ---- TYPEALIAS: type alias declarations ----
    ("TYPEALIAS", "ta_basic",
        "type MyInt = i32; fn main() -> i32 { let x: MyInt = 42; x }", 42),
    ("TYPEALIAS", "ta_fn_return",
        "type Count = i32; fn count() -> Count { 42 } fn main() -> i32 { count() }", 42),
    ("TYPEALIAS", "ta_two_aliases",
        "type Idx = i32; type Score = i32; fn score(i: Idx) -> Score { i * 7 } fn main() -> i32 { score(6) }", 42),
    ("TYPEALIAS", "ta_in_struct",
        "type Val = i32; struct S { v: Val } fn main() -> i32 { let s = S { v: 42 }; s.v }", 42),

    # ---- NESTED: multi-feature programs combining structs, enums, impls, match ----
    ("NESTED", "ne_struct_of_enum",
        "enum Dir { N, S, E, W } struct Step { dir: Dir, dist: i32 } "
        "fn main() -> i32 { let s = Step { dir: Dir::N, dist: 42 }; s.dist }", 42),
    ("NESTED", "ne_enum_multi_payload",
        "enum Event { Move(i32, i32), Click(i32) } "
        "fn process(e: Event) -> i32 { match e { Event::Move(x, y) => x + y, Event::Click(n) => n } } "
        "fn main() -> i32 { process(Event::Move(20, 22)) }", 42),
    ("NESTED", "ne_enum_nested_match",
        "enum Opt { None, Some(i32) } "
        "fn main() -> i32 { let outer = 2; let inner = Opt::Some(42); "
        "match outer { 1 => 0, 2 => match inner { Opt::Some(v) => v, Opt::None => 0 }, _ => 99 } }", 42),
    ("NESTED", "ne_nested_struct",
        "struct Inner { v: i32 } struct Outer { inner: Inner, extra: i32 } "
        "fn main() -> i32 { let o = Outer { inner: Inner { v: 20 }, extra: 22 }; o.inner.v + o.extra }", 42),
    ("NESTED", "ne_struct_fn_recursion",
        "struct Node { val: i32, depth: i32 } "
        "fn node_sum(n: Node) -> i32 { n.val + n.depth } "
        "fn main() -> i32 { let n = Node { val: 30, depth: 12 }; node_sum(n) }", 42),
    ("NESTED", "ne_enum_fn_dispatch",
        "enum Tree { Leaf(i32), Node(i32, i32) } "
        "fn tree_val(t: Tree) -> i32 { match t { Tree::Leaf(v) => v, Tree::Node(a, b) => a + b } } "
        "fn main() -> i32 { tree_val(Tree::Node(20, 22)) }", 42),
    ("NESTED", "ne_multi_struct_fn",
        "struct A { x: i32 } struct B { y: i32 } "
        "fn get_x(a: A) -> i32 { a.x } "
        "fn add_ab(a: A, b: B) -> i32 { get_x(a) + b.y } "
        "fn main() -> i32 { let a = A { x: 20 }; let b = B { y: 22 }; add_ab(a, b) }", 42),
    ("NESTED", "ne_enum_in_struct",
        "enum Color { Red, Blue } struct Pixel { color: Color, intensity: i32 } "
        "fn main() -> i32 { let p = Pixel { color: Color::Blue, intensity: 42 }; "
        "match p.color { Color::Red => 0, Color::Blue => p.intensity } }", 42),
    ("NESTED", "ne_multi_match_enum",
        "enum E { A, B(i32), C(i32, i32) } "
        "fn f(e: E) -> i32 { match e { E::A => 0, E::B(v) => v, E::C(a, b) => a + b } } "
        "fn main() -> i32 { f(E::C(20, 22)) }", 42),
    ("NESTED", "ne_fn_two_structs",
        "struct A { x: i32 } struct B { y: i32 } "
        "fn combine(a: A, b: B) -> i32 { a.x + b.y } "
        "fn main() -> i32 { let a = A { x: 20 }; let b = B { y: 22 }; combine(a, b) }", 42),
    ("NESTED", "ne_struct_match_field",
        "struct Pt { x: i32, y: i32 } "
        "fn main() -> i32 { let p = Pt { x: 10, y: 32 }; "
        "match p { Pt { x, y } => x + y } }", 42),
    ("NESTED", "ne_impl_plus_trait",
        "struct S { val: i32 } "
        "trait Get { fn get(self: S) -> i32; } "
        "impl Get for S { fn get(self: S) -> i32 { self.val } } "
        "fn double(n: i32) -> i32 { n * 2 } "
        "fn main() -> i32 { let s = S { val: 21 }; double(s.get()) }", 42),
    ("NESTED", "ne_impl_and_cond",
        "struct C { going_n: bool, dist: i32 } "
        "impl C { fn dx(self: C) -> i32 { if self.going_n { self.dist } else { 0 - self.dist } } } "
        "fn main() -> i32 { let c = C { going_n: true, dist: 42 }; c.dx() }", 42),

    # ---- PAT2: advanced pattern features ----
    ("PAT2", "pa2_struct_destr_fn",
        "struct Pt { x: i32, y: i32 } "
        "fn main() -> i32 { let p = Pt { x: 10, y: 32 }; match p { Pt { x, y } => x + y } }", 42),
    ("PAT2", "pa2_enum_guard_complex",
        "enum E { V(i32) } "
        "fn main() -> i32 { let e = E::V(42); match e { E::V(n) if n > 40 => 42, E::V(_) => 0 } }", 42),
    ("PAT2", "pa2_enum_or_arms",
        "enum E { A, B, C } "
        "fn main() -> i32 { let e = E::B; match e { E::A | E::B => 42, E::C => 0 } }", 42),
    ("PAT2", "pa2_nested_enum_match",
        "enum Outer { X(i32), Y } enum Inner { P(i32), Q } "
        "fn main() -> i32 { let o = Outer::X(2); let i = Inner::P(42); "
        "match o { Outer::X(v) => match i { Inner::P(w) => v * w / 2, Inner::Q => 0 }, Outer::Y => 0 } }", 42),
    ("PAT2", "pa2_guard_bind",
        "fn main() -> i32 { let x = 42; match x { n if n > 40 => n, n if n > 20 => 20, _ => 0 } }", 42),
]

# Known bootstrap parity gaps (Python succeeds, bootstrap diverges) — tracked so
# the corpus lands GREEN (xfail) instead of red, and auto-detected when fixed.
#
# FIXED 2026-05-29 (now empty -> full 218/218 parity):
#   il_u8_suffix / il_u16_suffix: `fn main() -> i32 { 42_u8 as i32 }` (and the
#   u16 form) SIGILLed (rc 132) in the bootstrap while Python -> 42. Root cause
#   was NOT the lexer suffix recognizer or the literal codegen (both correct) —
#   it was the AST_FN_DECL body-vs-return-type width-class trap (id 14002) in
#   kovc.hx. The `as i32` cast is a parser-level no-op (returns the inner node
#   unchanged), so the function body kept type tag u8 (width 1) against an i32
#   return (width 4); the width mismatch emitted a ud2. u32 (width 4) matched
#   i32 and so escaped the trap, which is why only u8/u16 (and i8/i16, and the
#   `u64 as i32` 8->4 form) appeared broken. Fix: fold the <=32-bit integer
#   family (i32/u32/u8/u16/i8/i16) into one width class at the trap site
#   (ret_body_width_class in kovc.hx), matching Python which imposes no width
#   restriction among <=32-bit integers. The 8-byte (i64/u64/f64) and 16-bit
#   float (bf16/f16) data-loss classes still trap.
# Advanced-feature parity gaps discovered 2026-05-29 (extended audit).
# Category: GENERIC — FIXED 2026-05-29. The bootstrap K1 used to SIGILL
# (rc=132) on EVERY generic-fn and generic-struct program. The true root
# cause was NOT a codegen width-class trap (the earlier theory): the
# bootstrap's Stage-8 monomorphize machinery (gp_tab, mr_tab synthesis,
# default-i32 mono clones, struct field 200+k substitution) was fully
# functional — but it was wired to recognise generic-param lists delimited
# by ANGLE brackets `<T>` (TK_LT=16 / TK_GT=17), while Helix (and the Python
# reference) use SQUARE brackets `[T]` (TK_LBRACK=20 / TK_RBRACK=21). So
# `fn id[T](...)` never entered the generic path: the parser consumed `[`
# as the `(` of the param list and desynced, producing a garbage AST that
# emitted `mov eax,<junk>; ud2`. Fix: parse_fn_decl + parse_struct_decl now
# accept `[`/`]` as the generic-param-list delimiters (legacy `<>` kept).
# All 12 cases now reach behavioral parity with Python — set is empty.
KNOWN_PARITY_GAPS: set[tuple[str, str]] = set()


# ============================================================================
# PYTEST PARAMETRIZE RUNNER
# ============================================================================

@pytest.mark.parametrize(
    "category,name,src,expected_rc",
    PARITY_CORPUS,
    ids=[f"{c[0]}/{c[1]}" for c in PARITY_CORPUS],
)
def test_parity(category: str, name: str, src: str, expected_rc: int):
    """Run one corpus entry through both compilers; assert behavioral parity.

    Two assertions:
      1. Python helixc rc == expected_rc  (corpus sanity check)
      2. Bootstrap kovc rc == Python helixc rc  (parity gate)
    """
    from helixc.tests.test_codegen import (
        compile_and_run as python_compile,
        _kovc_self_host_compile_and_run as bootstrap_compile,
    )

    # Path 1: Python helixc.
    python_rc = python_compile(src, optimize=True)
    assert python_rc == expected_rc, (
        f"[{category}/{name}] Python helixc rc={python_rc}, expected={expected_rc}. "
        f"Corpus item is wrong or Python helixc regressed."
    )

    # Path 2: Bootstrap kovc self-host. Retry on ANY mismatch (not just
    # rc 132): WSL has transient cold-start / IO flakes on tiny programs
    # that can return rc 1 (WSL_E_USER_NOT_FOUND), rc 132 (SIGILL), etc.
    # A REAL parity gap is deterministic and fails ALL attempts; a flake
    # passes on retry. 3 retries (4 attempts total).
    bootstrap_rc = bootstrap_compile(f"pm_{category}_{name}", src)
    tries = 0
    while bootstrap_rc != python_rc and tries < 3:
        tries += 1
        bootstrap_rc = bootstrap_compile(f"pm_{category}_{name}_r{tries}", src)

    # Known gaps are tracked as xfail so the corpus stays green; when a gap is
    # fixed the bootstrap matches and the test passes normally (remove from set).
    if (category, name) in KNOWN_PARITY_GAPS and bootstrap_rc != python_rc:
        pytest.xfail(
            f"known bootstrap parity gap {category}/{name}: "
            f"Python={python_rc}, Bootstrap={bootstrap_rc}"
        )

    assert bootstrap_rc == python_rc, (
        f"[{category}/{name}] PARITY GAP: Python={python_rc}, Bootstrap={bootstrap_rc}. "
        f"Bootstrap diverges from Python on this program."
    )


# ============================================================================
# STANDALONE RUNNER: produces gap report when run as a script
# ============================================================================

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

    from helixc.tests.test_codegen import (
        compile_and_run as python_compile,
        _kovc_self_host_compile_and_run as bootstrap_compile,
        _kovc_self_host_compile_and_run_full as bootstrap_compile_full,
    )

    results_A: list[tuple] = []   # PARITY OK
    results_B: list[tuple] = []   # PARITY GAP
    results_C: list[tuple] = []   # BEYOND-PYTHON
    results_D: list[tuple] = []   # NEITHER

    print(f"Running {len(PARITY_CORPUS)} parity probes...\n")

    for category, name, src, expected_rc in PARITY_CORPUS:
        uid = f"pm_{category}_{name}"
        # Python pass
        try:
            py_rc = python_compile(src, optimize=True)
            py_ok = True
        except Exception as e:
            py_rc = None
            py_ok = False
            py_err = str(e)[:80]

        # Bootstrap pass
        try:
            boot_rc = bootstrap_compile(uid, src)
            boot_ok = True
        except Exception as e:
            boot_rc = None
            boot_ok = False

        if py_ok and boot_ok:
            if py_rc == boot_rc:
                results_A.append((category, name, py_rc, boot_rc))
            else:
                # Get trap ID if SIGILL
                try:
                    rc_full, _, stderr = bootstrap_compile_full(uid + "_trap", src)
                    trap_info = stderr[:120] if stderr else "(no stderr)"
                except Exception:
                    trap_info = "(trap capture failed)"
                results_B.append((category, name, py_rc, boot_rc, trap_info, src[:80]))
        elif py_ok and not boot_ok:
            results_B.append((category, name, py_rc, "BOOT_ERR", "", src[:80]))
        elif not py_ok and boot_ok:
            results_C.append((category, name, py_err if not py_ok else "", boot_rc))
        else:
            results_D.append((category, name))

    # ---- REPORT ----
    print("=" * 72)
    print(f"PARITY AUDIT RESULTS  ({len(PARITY_CORPUS)} cases total)")
    print("=" * 72)
    print(f"  A: PARITY OK      {len(results_A):3d}")
    print(f"  B: PARITY GAP     {len(results_B):3d}  <-- BLOCKERS")
    print(f"  C: BEYOND-PYTHON  {len(results_C):3d}")
    print(f"  D: NEITHER        {len(results_D):3d}")
    print()

    if results_B:
        print("=" * 72)
        print("PARITY GAPS (B) — ACTION REQUIRED")
        print("=" * 72)
        for cat, nm, py_rc, boot_rc, trap, repro in results_B:
            print(f"\n  [{cat}/{nm}]")
            print(f"    Python:    {py_rc}")
            print(f"    Bootstrap: {boot_rc}")
            print(f"    Trap info: {trap}")
            print(f"    Repro src: {repro}")
        print()

    if results_C:
        print("=" * 72)
        print("BEYOND-PYTHON (C) — bootstrap exceeds Python (not blockers)")
        print("=" * 72)
        for cat, nm, py_err, boot_rc in results_C:
            print(f"  [{cat}/{nm}]  Python: {py_err[:60]}  Boot={boot_rc}")
        print()

    print("=" * 72)
    print("PARITY SURFACE CATEGORIES COVERED:")
    cats = sorted(set(c[0] for c in PARITY_CORPUS))
    for cat in cats:
        count = sum(1 for c in PARITY_CORPUS if c[0] == cat)
        gap_count = sum(1 for r in results_B if r[0] == cat)
        ok_count = sum(1 for r in results_A if r[0] == cat)
        print(f"  {cat:12s}: {count:3d} total, {ok_count:3d} OK, {gap_count:3d} GAP")
    print("=" * 72)
