"""Bootstrap stdlib-parity corpus.

Mirrors test_bootstrap_autodiff_parity.py: a data-driven corpus of small
stdlib-using programs, each run through BOTH the Python reference compiler
(compile_and_run, which auto-includes the stdlib via include_stdlib=True)
AND the Helix-native bootstrap compiler, asserting behavioral parity.

The bootstrap (unlike Python) does NOT auto-include the stdlib, so the
needed Helix-source stdlib module(s) are PREPENDED to the bootstrap source
(MODULE_DEPS + _stdlib_prefix). The bootstrap CAN compile the stdlib
modules; this corpus measures that it compiles stdlib-USING programs to the
same result as Python. (The eventual "bootstrap auto-includes stdlib on its
own" — so production programs need no manual prefix — is a separate K3/
driver concern tracked in project_helix_status.)

Items move from KNOWN_PARITY_GAPS to passing as bootstrap stdlib-compilation
bugs are fixed. Programs return a plain i32 (no float-return / float-cast
fragility); many are designed to return 42.
"""

import os

import pytest

# Stdlib modules to PREPEND for each corpus module-label (the bootstrap has
# no auto-include + no fn-DCE, so a label's deps must all be present). The
# label is the corpus's first tuple field; the values are helixc/stdlib/*.hx
# basenames (sans extension), prepended in order.
MODULE_DEPS: dict[str, list[str]] = {
    "option": ["option"],
    "result": ["result"],
    "vec": ["vec"],
    "iter": ["vec", "iterators"],       # iterators.hx + vec.hx base (some iter cases construct inputs via vec_new/vec_push)
    "hashmap": ["tensor", "hashmap"],   # hashmap ops call into tensor.hx
    "string": ["string"],
    "nn": ["tensor", "nn"],             # nn fns reference tensor.hx helpers
    "math": ["transcendentals"],        # math i32 helpers live in transcendentals.hx
    "tensor": ["tensor"],               # ti1d_*/ti2d_* dense tensor ops (self-contained)
}


# ============================================================================
# PARITY CORPUS  (module, name, src, expected_rc) — all independently verified
# 2026-05-30: BOOT == PY == expected, except the KNOWN_PARITY_GAPS below.
# ============================================================================
PARITY_CORPUS: list[tuple[str, str, str, int]] = [
    # ---- option (option.hx; enum Option { Some(i32), None }) ----
    ("option", "unwrap",
     "fn main() -> i32 { let o = Option::Some(42); option_unwrap_or(o, 0) }", 42),
    ("option", "max",
     "fn main() -> i32 { let a = Option::Some(3); let b = Option::Some(9); option_max(a, b) }", 9),
    ("option", "sum",
     "fn main() -> i32 { let a = Option::Some(7); let b = Option::None; option_sum(a, b) }", 7),

    # ---- result (result.hx; enum Result { Ok(i32), Err(i32) }) ----
    ("result", "unwrap",
     "fn main() -> i32 { let r = Result::Ok(42); result_unwrap_or(r, 0) }", 42),
    ("result", "err_code",
     "fn main() -> i32 { let r = Result::Err(7); result_err_code_or(r, 0) }", 7),
    ("result", "is_ok_false",
     "fn main() -> i32 { let r = Result::Err(5); result_is_ok(r) }", 0),

    # ---- vec (vec.hx; carry-pair Vec<i32>: arena start + threaded count) ----
    ("vec", "push_sum",
     "fn main() -> i32 { let s = vec_new(); let c0 = vec_push(s, 0, 5); "
     "let c1 = vec_push(s, c0, 7); let c2 = vec_push(s, c1, 30); vec_sum(s, c2) }", 42),
    ("vec", "max",
     "fn main() -> i32 { let s = vec_new(); let c0 = vec_push(s, 0, 12); "
     "let c1 = vec_push(s, c0, 30); let c2 = vec_push(s, c1, 7); vec_max(s, c2) }", 30),
    ("vec", "index_of",
     "fn main() -> i32 { let s = vec_new(); let c0 = vec_push(s, 0, 11); "
     "let c1 = vec_push(s, c0, 22); let c2 = vec_push(s, c1, 33); vec_index_of(s, c2, 33) }", 2),

    # ---- iterators (iterators.hx; range_to_vec + folds over arena vecs) ----
    ("iter", "fold_sum",
     "fn main() -> i32 { let s = range_to_vec(0, 5); vec_fold_op(s, 5, 0, 0) }", 10),
    ("iter", "dot",
     "fn main() -> i32 { let s = range_to_vec(1, 4); vec_dot(s, s, 3) }", 14),
    ("iter", "count_gt",
     "fn main() -> i32 { let s = range_to_vec(0, 10); vec_count_gt(s, 10, 6) }", 3),

    # ---- hashmap (hashmap.hx + tensor.hx) ----
    ("hashmap", "put_get",
     "fn main() -> i32 { let h = hashmap_new(8); hashmap_put(h, 8, 3, 42); hashmap_get(h, 8, 3, 0) }", 42),
    ("hashmap", "size",
     "fn main() -> i32 { let h = hashmap_new(8); hashmap_put(h, 8, 1, 10); "
     "hashmap_put(h, 8, 2, 20); hashmap_size(h, 8) }", 2),

    # ---- string (string.hx; carry-pair byte string, 1 byte/arena slot) ----
    ("string", "to_int",
     "fn main() -> i32 { let s = string_new(); let n1 = string_push(s, 0, 52); "
     "let n2 = string_push(s, n1, 50); string_to_int(s, n2) }", 42),
    ("string", "count_byte",
     "fn main() -> i32 { let s = string_new(); let n1 = string_push(s, 0, 65); "
     "let n2 = string_push(s, n1, 66); let n3 = string_push(s, n2, 65); "
     "string_count_byte(s, n3, 65) }", 2),

    # ---- nn (nn.hx + tensor.hx; pure scalar helpers) ----
    ("nn", "sgd_step_scalar",
     "fn main() -> i32 { sgd_step_scalar(50, 4, 2) }", 42),
    ("nn", "lin_reg_grad_b",
     "fn main() -> i32 { lin_reg_grad_b(3, 1, 2, 6) }", 2),

    # ---- math (transcendentals.hx; i32 helpers) ----
    ("math", "max_i32",
     "fn main() -> i32 { __max_i32(42, 7) }", 42),
    ("math", "clamp_i32",
     "fn main() -> i32 { __clamp_i32(100, 0, 42) }", 42),
    ("math", "abs_i32",
     "fn main() -> i32 { __abs_i32(0 - 42) }", 42),

    # ==== Expansion 2026-05-30 (harvested from test_codegen.py validated
    #      tests; each independently BOOT==PY==expected verified). ====
    ("option", "or_zero",
     "fn main() -> i32 { let a = Option::Some(42); let b = Option::None; option_or_zero(a) + option_or_zero(b) }", 42),
    ("option", "or_neg",
     "fn main() -> i32 { let a = Option::Some(43); let b = Option::None; option_or_neg(a) + option_or_neg(b) }", 42),
    ("option", "eq_some",
     "fn main() -> i32 { let a = Option::Some(42); let b = Option::Some(99); let c = Option::None; let n_match = option_eq_some(a, 42); let n_no = option_eq_some(b, 42); let n_none = option_eq_some(c, 42); n_match * 42 + n_no + n_none }", 42),
    ("option", "min",
     "fn main() -> i32 { let a = Option::Some(22); let b = Option::Some(20); let none = Option::None; let m1 = option_min(a, b); let m2 = option_min(a, none); m1 + m2 }", 42),
    ("result", "or_zero",
     "fn main() -> i32 { let a = Result::Ok(42); let b = Result::Err(99); result_or_zero(a) + result_or_zero(b) }", 42),
    ("result", "eq_ok",
     "fn main() -> i32 { let a = Result::Ok(42); let b = Result::Ok(99); let c = Result::Err(7); let n_match = result_eq_ok(a, 42); let n_no = result_eq_ok(b, 42); let n_err = result_eq_ok(c, 42); n_match * 42 + n_no + n_err }", 42),
    ("vec", "contains",
     "fn main() -> i32 { let s = vec_new(); let c0 = vec_push(s, 0, 11); let c1 = vec_push(s, c0, 22); let c2 = vec_push(s, c1, 33); let hit = vec_contains(s, c2, 22); let miss = vec_contains(s, c2, 99); (hit - miss) * 42 }", 42),
    # vec_abs_sum + vec_sum_squares live in iterators.hx (i64-accumulator L1/L2
    # helpers), so they are keyed "iter" (deps vec + iterators), NOT "vec" (the
    # earlier "vec" keying prepended only vec.hx, which lacks the fns -> the
    # bootstrap compiled a call to an undefined fn and trapped: a harness mis-
    # attribution, not the real bug). The REAL bug was K1.CAST-SX (2026-05-30):
    # a negative i32 `as i64` was ZERO-extended, so vec_abs_sum's `if v < 0` /
    # `if acc > hi` saturation mis-saturated to 2147483647. Fixed in kovc.hx
    # emit_cast_conv; now BOOT == PY == 42.
    ("iter", "abs_sum",
     "fn main() -> i32 { let v = vec_new(); let n0 = vec_push(v, 0, 5); let n1 = vec_push(v, n0, -3); let n2 = vec_push(v, n1, -8); let n3 = vec_push(v, n2, 1); vec_abs_sum(v, n3) * 2 + 8 }", 42),
    ("iter", "sum_squares",
     "fn main() -> i32 { let v = vec_new(); let n0 = vec_push(v, 0, 1); let n1 = vec_push(v, n0, 2); let n2 = vec_push(v, n1, 3); let n3 = vec_push(v, n2, 4); vec_sum_squares(v, n3) + 12 }", 42),
    ("string", "from_to_int_roundtrip",
     "fn main() -> i32 { let start = __arena_len(); let n = string_from_int(123); let v = string_to_int(start, n); if v == 123 { 42 } else { 1 } }", 42),
    ("string", "last_index_of",
     "fn main() -> i32 { let s = string_new(); let n0 = string_push(s, 0, 97); let n1 = string_push(s, n0, 98); let n2 = string_push(s, n1, 99); let n3 = string_push(s, n2, 98); let n4 = string_push(s, n3, 97); let last_b = string_last_index_of(s, n4, 98); let last_z = string_last_index_of(s, n4, 122); if last_b == 3 { if last_z == 0 - 1 { 42 } else { 1 } } else { 2 } }", 42),
    ("math", "min_max_clamp",
     "fn main() -> i32 { let a = __min_i32(5, 3); let b = __max_i32(5, 3); let c = __clamp_i32(100, 0, 10); a + b + c }", 18),
    ("math", "sign_i32",
     "fn main() -> i32 { let a = __sign_i32(7); let b = __sign_i32(0 - 5); let c = __sign_i32(0); a * 40 - b * 2 + c }", 42),

    # ==== Expansion 3 (2026-05-30, post K1.CAST-SX): harvested from validated
    #      test_codegen.py programs; each INDEPENDENTLY BOOT==PY==expected
    #      verified via _probe_corpus_expand (Python oracle = source of truth).
    #      Heavy on i64-accumulator + INT32-saturation paths that the
    #      sign-extension cast fix (fb0c75c) made trustworthy under comparison.
    #      NOTE: vec_l1_distance/sum_pure/map_square/cumsum/zip_mul/
    #      l2_squared_distance live in iterators.hx -> keyed "iter" (not "vec").
    # ---- iter (iterators.hx i64-accumulator vec helpers + saturation) ----
    ("iter", "l1_distance",
     "fn main() -> i32 { let a = vec_new(); let a0 = vec_push(a, 0, 3); let a1 = vec_push(a, a0, 7); let a2 = vec_push(a, a1, 2); let b = vec_new(); let b0 = vec_push(b, 0, 1); let b1 = vec_push(b, b0, 4); let b2 = vec_push(b, b1, 8); vec_l1_distance(a, b, a2) * 4 - 2 }", 42),
    ("iter", "sum_pure",
     "fn main() -> i32 { let v = __arena_len(); __arena_push(10); __arena_push(15); __arena_push(17); vec_sum_pure(v, 3) }", 42),
    ("iter", "map_square",
     "fn main() -> i32 { let v = vec_new(); let n0 = vec_push(v, 0, 1); let n1 = vec_push(v, n0, 2); let n2 = vec_push(v, n1, 3); let n3 = vec_push(v, n2, -4); let dst = vec_map_square(v, n3); let original_third = vec_get(v, 3); if original_third == -4 { vec_sum(dst, n3) + 12 } else { 0 } }", 42),
    ("iter", "cumsum",
     "fn main() -> i32 { let v = vec_new(); let n0 = vec_push(v, 0, 1); let n1 = vec_push(v, n0, 2); let n2 = vec_push(v, n1, 3); let n3 = vec_push(v, n2, 4); let n4 = vec_push(v, n3, 5); let dst = vec_cumsum(v, n4); let last = vec_get(dst, 4); if last == 15 { last * 2 + 12 } else { 0 } }", 42),
    ("iter", "dot_saturates",
     "fn main() -> i32 { let a = __arena_len(); __arena_push(46341); __arena_push(46341); let b = __arena_len(); __arena_push(46341); __arena_push(46341); let r = vec_dot(a, b, 2); if r == 2147483647 { 42 } else { 7 } }", 42),
    ("iter", "l2_sqdist_saturates",
     "fn main() -> i32 { let a = __arena_len(); __arena_push(46341); __arena_push(46341); let b = __arena_len(); __arena_push(0); __arena_push(0); let r = vec_l2_squared_distance(a, b, 2); if r == 2147483647 { 42 } else { 7 } }", 42),
    ("iter", "zip_mul_saturates",
     "fn main() -> i32 { let a = __arena_len(); __arena_push(46341); let b = __arena_len(); __arena_push(46341); let r = vec_zip_mul(a, b, 1); let v = __arena_get(r); if v == 2147483647 { 42 } else { 7 } }", 42),
    # ---- vec (vec.hx i64-accumulator saturation) ----
    ("vec", "sum_saturates",
     "fn main() -> i32 { let s = __arena_len(); __arena_push(2147483647); __arena_push(2147483647); let r = vec_sum(s, 2); if r == 2147483647 { 42 } else { 7 } }", 42),
    # ---- tensor (tensor.hx dense i32 tensor ops + saturation) ----
    ("tensor", "ti1d_l1_norm",
     "fn main() -> i32 { let x = t1d_new(5); ti1d_set(x, 0, 3); ti1d_set(x, 1, 0 - 10); ti1d_set(x, 2, 5); ti1d_set(x, 3, 0 - 20); ti1d_set(x, 4, 4); ti1d_l1_norm(x, 5) }", 42),
    ("tensor", "ti1d_l2_norm_sq",
     "fn main() -> i32 { let x = t1d_new(4); ti1d_set(x, 0, 3); ti1d_set(x, 1, 4); ti1d_set(x, 2, 1); ti1d_set(x, 3, 2); ti1d_l2_norm_sq(x, 4) + 12 }", 42),
    ("tensor", "ti1d_dot",
     "fn main() -> i32 { let x = ti2d_new(1, 3); ti1d_set(x, 0, 1); ti1d_set(x, 1, 2); ti1d_set(x, 2, 3); let y = ti2d_new(1, 3); ti1d_set(y, 0, 10); ti1d_set(y, 1, 20); ti1d_set(y, 2, 30); ti1d_dot(x, y, 3) }", 140),
    ("tensor", "ti2d_matmul",
     "fn main() -> i32 { let a = ti2d_new(2, 2); ti2d_set(a, 2, 0, 0, 1); ti2d_set(a, 2, 0, 1, 2); ti2d_set(a, 2, 1, 0, 3); ti2d_set(a, 2, 1, 1, 4); let b = ti2d_new(2, 2); ti2d_set(b, 2, 0, 0, 5); ti2d_set(b, 2, 0, 1, 6); ti2d_set(b, 2, 1, 0, 7); ti2d_set(b, 2, 1, 1, 8); let c = ti2d_new(2, 2); ti2d_matmul(a, 2, 2, b, 2, c); ti2d_get(c, 2, 0, 0) + ti2d_get(c, 2, 1, 1) }", 69),
    ("tensor", "ti2d_matmul_saturates",
     "fn main() -> i32 { let a = ti2d_new(2, 2); ti2d_set(a, 2, 0, 0, 46341); ti2d_set(a, 2, 0, 1, 46341); ti2d_set(a, 2, 1, 0, 0); ti2d_set(a, 2, 1, 1, 0); let b = ti2d_new(2, 2); ti2d_set(b, 2, 0, 0, 46341); ti2d_set(b, 2, 0, 1, 0); ti2d_set(b, 2, 1, 0, 46341); ti2d_set(b, 2, 1, 1, 0); let c = ti2d_new(2, 2); ti2d_matmul(a, 2, 2, b, 2, c); let c00 = ti2d_get(c, 2, 0, 0); if c00 == 2147483647 { 42 } else { 7 } }", 42),
    ("tensor", "ti1d_prod",
     "fn main() -> i32 { let x = t1d_new(4); ti1d_set(x, 0, 1); ti1d_set(x, 1, 2); ti1d_set(x, 2, 3); ti1d_set(x, 3, 7); ti1d_prod(x, 4) }", 42),
    # ---- hashmap (hashmap.hx + tensor.hx) ----
    ("hashmap", "collision_probing",
     "fn main() -> i32 { let m = hashmap_new(4); hashmap_put(m, 4, 0, 10); hashmap_put(m, 4, 4, 20); hashmap_put(m, 4, 8, 12); hashmap_get(m, 4, 0, 0) + hashmap_get(m, 4, 4, 0) + hashmap_get(m, 4, 8, 0) }", 42),
    ("hashmap", "sum_values",
     "fn main() -> i32 { let m = hashmap_new(8); hashmap_put(m, 8, 1, 10); hashmap_put(m, 8, 2, 15); hashmap_put(m, 8, 3, 17); hashmap_sum_values(m, 8) }", 42),
    ("hashmap", "argmax_key",
     "fn main() -> i32 { let m = hashmap_new(8); hashmap_put(m, 8, 1, 10); hashmap_put(m, 8, 2, 5); hashmap_put(m, 8, 42, 100); hashmap_argmax_key(m, 8) }", 42),
    ("hashmap", "has_size",
     "fn main() -> i32 { let m = hashmap_new(8); hashmap_put(m, 8, 100, 1); hashmap_put(m, 8, 200, 1); hashmap_put(m, 8, 300, 1); let h1 = hashmap_has(m, 8, 200); let h2 = hashmap_has(m, 8, 999); let s = hashmap_size(m, 8); h1 * 30 + s * 4 + h2 }", 42),
    # ---- nn (nn.hx + tensor.hx; scalar losses, argmax/min, dense layer) ----
    ("nn", "argmax",
     "fn main() -> i32 { let x = t1d_new(3); ti1d_set(x, 0, 3); ti1d_set(x, 1, 7); ti1d_set(x, 2, 2); argmax(x, 3) }", 1),
    ("nn", "argmin",
     "fn main() -> i32 { let x = t1d_new(4); ti1d_set(x, 0, 3); ti1d_set(x, 1, 7); ti1d_set(x, 2, 2); ti1d_set(x, 3, 5); argmin(x, 4) }", 2),
    ("nn", "mse_loss",
     "fn main() -> i32 { let y = t1d_new(2); ti1d_set(y, 0, 3); ti1d_set(y, 1, 5); let t = t1d_new(2); ti1d_set(t, 0, 4); ti1d_set(t, 1, 5); mse_loss(y, t, 2) }", 1),
    ("nn", "mae_loss",
     "fn main() -> i32 { let y = t1d_new(3); ti1d_set(y, 0, 3); ti1d_set(y, 1, 5); ti1d_set(y, 2, 9); let t = t1d_new(3); ti1d_set(t, 0, 4); ti1d_set(t, 1, 7); ti1d_set(t, 2, 5); mae_loss(y, t, 3) }", 7),
    ("nn", "count_correct",
     "fn main() -> i32 { let p = t1d_new(5); ti1d_set(p, 0, 1); ti1d_set(p, 1, 2); ti1d_set(p, 2, 3); ti1d_set(p, 3, 4); ti1d_set(p, 4, 5); let t = t1d_new(5); ti1d_set(t, 0, 1); ti1d_set(t, 1, 9); ti1d_set(t, 2, 3); ti1d_set(t, 3, 4); ti1d_set(t, 4, 9); count_correct(p, t, 5) }", 3),
    ("nn", "dense_layer",
     "fn main() -> i32 { let x = t1d_new(2); ti1d_set(x, 0, 3); ti1d_set(x, 1, 1); let w = ti2d_new(2, 2); ti2d_set(w, 2, 0, 0, 2); ti2d_set(w, 2, 0, 1, 1); ti2d_set(w, 2, 1, 0, 1); ti2d_set(w, 2, 1, 1, 2); let b = t1d_new(2); ti1d_set(b, 0, 0); ti1d_set(b, 1, 0); let z = t1d_new(2); dense_layer_forward(w, 2, 2, x, b, z); ti1d_sum(z, 2) }", 12),
    ("nn", "training_step_converges",
     "fn main() -> i32 { let mut w: i32 = 0; let mut i: i32 = 0; while i < 10 { let g = lin_reg_grad_w(w, 0, 1, 10); let step = if g > 0 { 1 } else { if g < 0 { 0 - 1 } else { 0 } }; w = w - step; i = i + 1; } w }", 10),
    ("nn", "lin_reg_grad_w_saturates",
     "fn main() -> i32 { let g = lin_reg_grad_w(46341, 0, 46341, 0); if g == 2147483647 { 42 } else { 7 } }", 42),
    # ---- string (string.hx scanning helpers) ----
    ("string", "count_lines",
     "fn main() -> i32 { let s = string_new(); let s1 = string_push(s, 0, 97); let s2 = string_push(s, s1, 10); let s3 = string_push(s, s2, 98); let s4 = string_push(s, s3, 10); let s5 = string_push(s, s4, 99); let s6 = string_push(s, s5, 10); string_count_lines(s, s6) * 14 }", 42),
    ("string", "eq_ignore_case",
     "fn main() -> i32 { let a = string_new(); let a1 = string_push(a, 0, 65); let a2 = string_push(a, a1, 98); let a3 = string_push(a, a2, 67); let b = __arena_len(); let b1 = string_push(b, 0, 97); let b2 = string_push(b, b1, 66); let b3 = string_push(b, b2, 99); string_eq_ignore_case_ascii(a, a3, b, b3) * 42 }", 42),

    # ==== Expansion 4 (2026-05-30): more tensor reductions/elementwise +
    #      string; each INDEPENDENTLY BOOT==PY==expected verified via probe
    #      (Python oracle = truth). Broadens dense-tensor codegen coverage.
    ("tensor", "axpy",
     "fn main() -> i32 { let x = t1d_new(3); ti1d_set(x, 0, 1); ti1d_set(x, 1, 1); ti1d_set(x, 2, 1); let y = t1d_new(3); ti1d_set(y, 0, 1); ti1d_set(y, 1, 2); ti1d_set(y, 2, 3); ti1d_axpy(y, 2, x, 3); ti1d_sum(y, 3) }", 12),
    ("tensor", "matvec",
     "fn main() -> i32 { let w = ti2d_new(2, 2); ti2d_set(w, 2, 0, 0, 1); ti2d_set(w, 2, 0, 1, 2); ti2d_set(w, 2, 1, 0, 3); ti2d_set(w, 2, 1, 1, 4); let x = t1d_new(2); ti1d_set(x, 0, 10); ti1d_set(x, 1, 20); let y = t1d_new(2); ti2d_matvec(w, 2, 2, x, y); ti1d_sum(y, 2) }", 160),
    ("tensor", "relu_then_add",
     "fn main() -> i32 { let x = t1d_new(3); ti1d_set(x, 0, 0 - 3); ti1d_set(x, 1, 0); ti1d_set(x, 2, 4); let r = t1d_new(3); ti1d_relu(x, r, 3); let b = t1d_new(3); ti1d_set(b, 0, 1); ti1d_set(b, 1, 2); ti1d_set(b, 2, 3); let z = t1d_new(3); ti1d_add(r, b, z, 3); ti1d_sum(z, 3) }", 10),
    ("tensor", "sub",
     "fn main() -> i32 { let x = t1d_new(4); ti1d_set(x, 0, 10); ti1d_set(x, 1, 20); ti1d_set(x, 2, 30); ti1d_set(x, 3, 40); let y = t1d_new(4); ti1d_set(y, 0, 1); ti1d_set(y, 1, 2); ti1d_set(y, 2, 3); ti1d_set(y, 3, 4); let z = t1d_new(4); ti1d_sub(x, y, z, 4); ti1d_sum(z, 4) - 48 }", 42),
    ("tensor", "mul",
     "fn main() -> i32 { let x = t1d_new(3); ti1d_set(x, 0, 2); ti1d_set(x, 1, 3); ti1d_set(x, 2, 5); let y = t1d_new(3); ti1d_set(y, 0, 3); ti1d_set(y, 1, 4); ti1d_set(y, 2, 2); let z = t1d_new(3); ti1d_mul(x, y, z, 3); ti1d_sum(z, 3) + 14 }", 42),
    ("tensor", "max",
     "fn main() -> i32 { let x = t1d_new(5); ti1d_set(x, 0, 3); ti1d_set(x, 1, 7); ti1d_set(x, 2, 1); ti1d_set(x, 3, 42); ti1d_set(x, 4, 5); ti1d_max(x, 5) }", 42),
    ("tensor", "argmin",
     "fn main() -> i32 { let x = t1d_new(5); ti1d_set(x, 0, 10); ti1d_set(x, 1, 20); ti1d_set(x, 2, 5); ti1d_set(x, 3, 30); ti1d_set(x, 4, 15); ti1d_argmin(x, 5) * 20 + 2 }", 42),
    ("tensor", "clamp",
     "fn main() -> i32 { let x = t1d_new(3); ti1d_set(x, 0, 0 - 5); ti1d_set(x, 1, 3); ti1d_set(x, 2, 100); let dst = t1d_new(3); ti1d_clamp(x, 0, 50, dst, 3); ti1d_sum(dst, 3) - 11 }", 42),
    ("tensor", "transpose",
     "fn main() -> i32 { let src = ti2d_new(2, 3); ti2d_set(src, 3, 0, 0, 1); ti2d_set(src, 3, 0, 1, 2); ti2d_set(src, 3, 0, 2, 3); ti2d_set(src, 3, 1, 0, 4); ti2d_set(src, 3, 1, 1, 5); ti2d_set(src, 3, 1, 2, 6); let dst = ti2d_new(3, 2); ti2d_transpose(src, 2, 3, dst); let s = ti2d_get(dst, 2, 0, 0) + ti2d_get(dst, 2, 0, 1) + ti2d_get(dst, 2, 1, 0) + ti2d_get(dst, 2, 1, 1) + ti2d_get(dst, 2, 2, 0) + ti2d_get(dst, 2, 2, 1); s * 2 }", 42),
    ("string", "to_lower",
     "fn main() -> i32 { let s = string_new(); let s1 = string_push(s, 0, 72); let s2 = string_push(s, s1, 69); let s3 = string_push(s, s2, 76); let s4 = string_push(s, s3, 76); let s5 = string_push(s, s4, 79); let lower = string_to_lower(s, s5); let first = string_get(lower, 0); first - 62 }", 42),
]


# ============================================================================
# KNOWN PARITY GAPS — bootstrap stdlib-compilation divergences (xfail).
# Remove entries as the underlying bootstrap bug is fixed.
# ============================================================================
KNOWN_PARITY_GAPS: set[tuple[str, str]] = {
    # (empty) — all corpus entries pass under both Python and the bootstrap.
    #
    # FIXED 2026-05-30: vec_abs_sum / vec_sum_squares (keyed "iter") were a
    # TWO-part finding. (a) Harness mis-attribution: keyed "vec" they got only
    # vec.hx prepended, but the fns live in iterators.hx -> the bootstrap
    # compiled a call to an undefined fn and trapped (rc132). Re-keyed "iter"
    # (deps vec + iterators) so the def is present. (b) The REAL codegen bug,
    # exposed once defined: a negative i32 `as i64` was ZERO-extended (kovc.hx
    # emit_cast_conv int->int no-op), so the `if v < 0` / `if acc > hi`
    # saturation mis-saturated to 2147483647. Fixed with a movsxd sign-extend
    # for EXACTLY-i32 -> i64/u64 widening (K1.CAST-SX); guarded directly in
    # test_parity_matrix.py CAST (ca_neg_*).
    #
    # FIXED 2026-05-30: option/sum was the symptom of a deep nested-match
    # codegen bug. A nested match in ANY arm of a match re-init'd the SHARED
    # match_state region (kovc.hx single fail_state+end_table at bn_state+
    # 84..117), orphaning the enclosing match's already-recorded merge-jump
    # -> the parent arm's jmp kept its placeholder (jmp +0) and fell through
    # into the next arm, re-executing it (option_sum(Some,Some) -> the 2nd
    # payload; (Some,None) -> 0). Fixed in emit_one_match_arm by saving/
    # restoring the 34-slot match_state region across each arm body (it now
    # nests via the call stack); single/sequential matches unaffected. Was a
    # real CPU-language gap too (the 277-case corpus didn't cover it).
}


_STDLIB_CACHE: dict[str, str] = {}


def _stdlib_prefix(modules: list[str]) -> str:
    """Concatenate the given helixc/stdlib/<module>.hx sources (cached).

    The bootstrap does not auto-include the stdlib (Python does, via
    parse(include_stdlib=True)), so a stdlib-using program must carry the
    Helix-source definitions. Generalizes the autodiff harness's
    _autodiff_stdlib_prefix to an arbitrary module set.
    """
    helixc_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parts = []
    for m in modules:
        if m not in _STDLIB_CACHE:
            with open(os.path.join(helixc_dir, "stdlib", m + ".hx"),
                      encoding="utf-8") as f:
                _STDLIB_CACHE[m] = f.read()
        parts.append(_STDLIB_CACHE[m])
    return "\n\n".join(parts)


@pytest.mark.parametrize(
    "module,name,src,expected_rc",
    PARITY_CORPUS,
    ids=[f"{c[0]}/{c[1]}" for c in PARITY_CORPUS],
)
def test_stdlib_parity(module: str, name: str, src: str, expected_rc: int):
    """Run one corpus entry through Python (sanity) + the bootstrap; assert
    behavioral parity. The bootstrap source gets the module's stdlib deps
    prepended (Python auto-includes the stdlib itself).

    Bootstrap retry: 3 retries on any mismatch (WSL cold-start flakes are
    transient; real gaps are deterministic).
    """
    from helixc.tests.test_codegen import (
        _kovc_self_host_compile_and_run as bootstrap_compile,
        compile_and_run as python_compile,
    )

    # Python sanity check: the corpus expected_rc must match the reference.
    python_rc = python_compile(src)
    assert python_rc == expected_rc, (
        f"[{module}/{name}] Python helixc rc={python_rc}, expected="
        f"{expected_rc}. Corpus expected_rc is wrong or Python regressed."
    )

    # Bootstrap check with the needed stdlib module(s) prepended.
    boot_src = _stdlib_prefix(MODULE_DEPS[module]) + "\n\n" + src
    bootstrap_rc = bootstrap_compile(f"stdlib_{module}_{name}", boot_src)
    tries = 0
    while bootstrap_rc != expected_rc and tries < 3:
        tries += 1
        bootstrap_rc = bootstrap_compile(f"stdlib_{module}_{name}_r{tries}", boot_src)

    if (module, name) in KNOWN_PARITY_GAPS and bootstrap_rc != expected_rc:
        pytest.xfail(
            f"known stdlib parity gap [{module}/{name}]: "
            f"expected={expected_rc}, bootstrap={bootstrap_rc}"
        )

    assert bootstrap_rc == expected_rc, (
        f"[{module}/{name}] STDLIB PARITY GAP: "
        f"expected={expected_rc}, bootstrap={bootstrap_rc}."
    )
