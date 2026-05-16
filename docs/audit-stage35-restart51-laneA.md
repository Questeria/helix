# Lane A Audit Report — Stage 35 Restart 51

**HEAD**: `7b945fa Record Stage 35 restart 50 lane audit reports`
**Scope**: Runtime / stdlib safety. Read-only audit; fixes applied separately.

## Summary

Reviewed all 16 stdlib files under `helixc/stdlib/`. Confirmed prior restart 45-50 fixes remain in place. Found **5 new issues**: 1 HIGH, 2 MEDIUM, 2 LOW.

---

## A1 — `__log_f64` lacks domain guard (x <= 0 produces wrong results) — HIGH

**File**: `helixc/stdlib/transcendentals.hx:249-263`
**Function**: `__log_f64`
**Bug family**: New `__log`/`__exp`/`__sqrt` site lacking domain check

**Issue**: `__log_f64(x)` is a raw 7-term Taylor series `y - y²/2 + ... where y = x - 1`. There is no guard for `x <= 0`. The f32 sibling `__log` (line 84) is identically unguarded — but the guarded range-reduced stable version `__log_stable` exists only in f32. When `x <= 0`, `y = x - 1 <= -1` and the series diverges; for `x = 0`, `y = -1` giving a polynomial result nowhere near `-inf`. Downstream `d_log_v` in `autodiff.hx:85` calls `__log_f64(a_v)` without checking the domain (only `d_log_dx` at line 87 guards `a_v <= 0`), so the **value** returned for any `a_v <= 0` is a nonsense finite number — a caller using the loss to decide convergence will continue training with corrupted loss.

**Sibling sweep**:
| Function | Domain guard? |
|---|---|
| `__log` (f32, line 84) | No guard — but safe f32 call sites go through `__log_stable` |
| `__log_stable` (f32, line 94) | Yes — `x <= 0` returns sentinel |
| `__log_f64` (f64, line 249) | No guard — NEW unguarded site |
| `d_log_v` (autodiff.hx:85) | No guard — calls `__log_f64` directly |
| `d_log_dx` (autodiff.hx:87) | Guarded `a_v <= 0` |

**Suggested fix**: Add `__log_stable_f64` mirroring `__log_stable`, or inline guard at top of `__log_f64`: `if x <= 0.0_f64 { 0.0_f64 - 1000000.0_f64 } else { /* body */ }`. Update `d_log_v` to call the stable version.

**Suggested canary**: `test_log_f64_domain_guard`: assert `__log_f64(0.0_f64) < -999999.0_f64` and `__log_f64(-1.0_f64) < -999999.0_f64`.

---

## A2 — `clip_grad_norm_f32` division by `norm` when `norm` is NaN — MEDIUM

**File**: `helixc/stdlib/nn.hx:165-181`
**Function**: `clip_grad_norm_f32`
**Bug family**: NaN propagation through new numerical helpers

**Issue**: Guards `norm_sq <= 0.0_f32` but not `norm_sq != norm_sq` (NaN). If any gradient slot is NaN, `tf1d_l2_norm_sq` returns NaN. IEEE 754: `NaN <= 0.0_f32` evaluates false. Falls into else branch: `__sqrt(NaN) = 0.0`, then `scale = target / 0.0`, then `tf1d_scale_inplace` poisons every gradient slot with NaN/inf. A single corrupt slot becomes total gradient destruction.

**Sibling sweep**:
| Site | NaN guard on accumulator? |
|---|---|
| `adam_f32_step` (nn.hx:236) | Yes — `raw_denom != raw_denom` |
| `layer_norm_f32` (nn.hx:586) | Yes — `denom != denom` |
| `softmax_layer` (nn.hx:678) | Yes — `sum_e != sum_e` |
| `clip_grad_norm_f32` (nn.hx:170) | No — missing NaN test |

**Suggested fix**: `if (norm_sq <= 0.0_f32) || (norm_sq != norm_sq) { 0 } else { ... }`.

**Suggested canary**: `test_clip_grad_norm_nan_input`: populate a 4-element f32 tensor with one NaN element, call `clip_grad_norm_f32`, assert no output slot is NaN.

---

## A3 — `string_to_int` signed overflow on large positive inputs — MEDIUM

**File**: `helixc/stdlib/string.hx:185-207`
**Function**: `string_to_int`
**Bug family**: Integer overflow in width-promotion math

**Issue**: `acc = acc * 10 + (b - 48)` in plain i32. Parsing "2147483648" (INT32_MAX+1): after "214748364", acc=214748364; then `*10` = 2147483640 (fits); `+8` = 2147483648 silently wraps to INT32_MIN. Returns wrong large-negative value with no error signal.

**Sibling sweep**:
| Site | Overflow protection? |
|---|---|
| `string_from_int` (string.hx:91) | Safe — outputs digits |
| `string_to_int` (string.hx:185) | No — i32 multiply wraps |
| `hashmap_sum_values` (hashmap.hx:365) | Yes — i64 accumulator |
| `hashmap_increment` (hashmap.hx:257) | Yes — i64 intermediate |

**Suggested fix**: Promote `acc` to i64, saturate at INT32_MAX/INT32_MIN.

**Suggested canary**: `test_string_to_int_overflow`: parse "2147483648", assert result is saturated to INT32_MAX not wrapped to INT32_MIN.

---

## A4 — `vec_zip_mod` and `vec_zip_div` trap on division-by-zero without guard — LOW

**File**: `helixc/stdlib/iterators.hx:611-619` and `672-680`
**Function**: `vec_zip_mod`, `vec_zip_div`
**Bug family**: Division-by-zero in new arithmetic helpers

**Issue**: Functions deliberately delegate trap to runtime per comment ("std behavior, not stdlib's concern"). But this is inconsistent with the fail-closed discipline of `hashmap_hash`, `ti1d_mean`, `__rand_step` etc., and in arena-runtime without OS exception recovery a trap is a process crash.

**Sibling sweep**:
| Site | Zero-divisor guard? |
|---|---|
| `hashmap_hash` (hashmap.hx:69) | Yes — `if cap <= 0 { 0 }` |
| `ti1d_mean` (tensor.hx:595) | Yes — `if n <= 0 { 0 }` |
| `vec_zip_div` (iterators.hx:672) | No — traps |
| `vec_zip_mod` (iterators.hx:611) | No — traps |

**Suggested fix**: Per-element guard `let bv = __arena_get(b + i); if bv == 0 { __arena_push(0); } else { __arena_push(__arena_get(a + i) / bv); }`.

**Suggested canary**: `test_vec_zip_div_zero_divisor`: call with denominator vec containing 0, assert result slot is 0 and no trap.

---

## A5 — `vec_negate_inplace` + `vec_map_neg` signed overflow on INT32_MIN — LOW

**File**: `helixc/stdlib/iterators.hx:432-440` (negate_inplace), `:176` (map_neg)
**Bug family**: Signed INT32_MIN wrap (sibling of `__abs_i32` caveat)

**Issue**: `0 - v` wraps when `v == INT32_MIN`. Result is INT32_MIN (no-op). `__abs_i32` documents this; these don't.

**Sibling sweep**:
| Site | INT32_MIN handled? |
|---|---|
| `__abs_i32` (transcendentals.hx:532) | Documented, no guard |
| `vec_negate_inplace` (iterators.hx:432) | No guard, no doc |
| `vec_map_neg` (iterators.hx:176) | No guard, no doc |
| `string_from_int` (string.hx:98) | Correctly special-cased |

**Suggested fix**: `let nv = if v == (0 - 2147483647 - 1) { 2147483647 } else { 0 - v };` (saturate to INT32_MAX).

**Suggested canary**: `test_vec_negate_inplace_int32_min`: 1-element vec with INT32_MIN, assert post-call value is INT32_MAX.

---

## Cross-cutting clean areas confirmed

- All 5 magic constants distinct (t1d=1001001, t2d=2002001, rev_tape=3003001, hashmap=7007001, tree_node=7107001).
- All 14 arena-span overflow guards present.
- `__sqrt`/`__sqrt_f64` both guard `x <= 0`.
- `__exp`/`__exp_f64` both cap k at ±48/±1023.
- `__adam_step`, `adam_f32_step`, `layer_norm_f32`, `softmax_layer`, `dense_classifier_sgd_step_f32` all NaN-fail-closed.
- `tanh_layer` delegates to `__tanh`; sigmoid/softplus/silu/gelu layers delegate correctly.
- `rev_tape_valid` and `rev_adj_cap` both reject `arena_span_in_tensor_payload`.
- `hashmap_ok` and `tree_node_ok` use `arena_span_in_tensor_payload` properly.
- `ti1d_prod` i64+saturate; `hashmap_load_factor_x100` i64 numerator.
- `string_from_int` INT32_MIN special-cased.
- `__powi` returns 1.0 for n > 16.
