# Stage 35 Restart 54 — Lane A (Runtime / Safety) Audit

**HEAD**: c4cb7a3
**Date**: 2026-05-16
**Mode**: Read-only audit (fixes happen in a separate sweep)

## Summary

7 findings: 4 HIGH + 2 MEDIUM + 1 LOW. All seven are missed siblings of
the i64-saturation family that has dominated restarts 50–53. The biggest
new exposure is the reverse-mode autodiff tape: every forward record AND
the entire backward gradient accumulator are still in raw i32 (silent
wrap), which makes downstream `rev_grad(...)` results meaningless once
any intermediate exceeds INT32_MAX. Clean families verified: forged
handles (13 validators), magic-constant uniqueness (13 distinct),
arena-span overflow guards (all sites), float-domain guards
(`__sqrt`/`__log` family), stale-state resurrection in
`wm_clear`/`hashmap_clear`/`bindings_rewind`/`ep_record`/tape rebuild.

## Findings

### A1 (HIGH): reverse-mode autodiff tape stores forward values + propagates adjoints in raw i32

- **File:function:line**:
  - `helixc/stdlib/autodiff_reverse.hx:198-234` (`rev_add`, `rev_sub`, `rev_mul`, `rev_neg` — forward-value store)
  - `helixc/stdlib/autodiff_reverse.hx:438-465` (`rev_backward` — kind=1/2/3/4 adjoint propagation)
- **Bug family**: i64-saturation siblings missed in restart 51/52/53 sweep — sibling of `ti1d_dot` (restart 51 A3), `ti2d_matvec`/`ti2d_matmul` (restart 52 A1), `vec_dot`/`attention_dot` (restart 53 A1/A4) and `sgd_step_array` (restart 53 A7). Reverse-mode AD performs the same `acc += a * b` accumulator pattern these fixes were intended to harden, but the autodiff tape was never lifted to i64.
- **Sibling sweep** (forward record):
  | Op | File:line | Forward value computation | i64 + saturation? |
  |---|---|---|---|
  | `rev_add` | 198-206 | `av + bv` (line 204) | NO — silent wrap |
  | `rev_sub` | 208-216 | `av - bv` (line 214) | NO — silent wrap |
  | `rev_mul` | 218-226 | `av * bv` (line 224) | NO — silent wrap (`av=bv=46341` overflows) |
  | `rev_neg` | 228-234 | `0 - av` (line 232) | NO — INT32_MIN wraps to INT32_MIN |
- **Sibling sweep** (backward adjoint accumulation):
  | Kind | rev_backward block | Update | i64 + saturation? |
  |---|---|---|---|
  | 1 add | 442-444 | `adj[in1] + adj_i`, `adj[in2] + adj_i` | NO |
  | 2 sub | 449-451 | `adj[in1] + adj_i`, `adj[in2] - adj_i` | NO |
  | 3 mul | 456-460 | `adj[in1] + adj_i * v_b`, `adj[in2] + adj_i * v_a` | NO — DOUBLE silent wrap (the i32 multiply AND the add) |
  | 4 neg | 463-465 | `adj[in1] - adj_i` | NO |
- **Nearby-but-safe sites**:
  - `rev_seed` (350-359): scalar write only — no arithmetic, safe.
  - `rev_grad` (361-368): read-only, safe.
  - `rev_value_at` (236-240): read-only, safe.
  - `rev_alloc_adjoints` and `rev_tape_valid`: arena/guard arithmetic
    only, no user value math.
- **Suggested regression**:
  `test_stage35_restart54_rev_mul_forward_saturates_on_i32_overflow`
  and
  `test_stage35_restart54_rev_backward_mul_adjoint_saturates_on_i32_overflow`
  (use `rev_leaf(50000)` × 2, `rev_mul`, then `rev_backward` and assert
  the gradient is INT32_MAX rather than the wrapped negative).
- **Severity rationale**: HIGH — this silently invalidates every
  user-facing autodiff result the moment intermediate values exceed
  i32 magnitude. Worse than the previous siblings because the
  adjoint-multiply step compounds the wrap (adj_i * v_b can wrap, then
  adj[in1] + (that wrapped value) wraps again). A user "training" with
  reverse-mode AD gets silently corrupted gradients and never sees an
  error. Forward record + backward accumulation must BOTH be lifted to
  i64 with INT32 saturation, mirroring the `attention_dot` precedent
  that already does forward (dot) + accumulator + normalize in i64.

### A2 (HIGH): `ti1d_mul` element-wise integer-tensor multiply silently wraps

- **File:function:line**: `helixc/stdlib/tensor.hx:476-490` (`ti1d_mul`)
- **Bug family**: i64-saturation siblings — sibling of `ti1d_axpy`,
  `ti1d_mul_scalar`, `ti1d_add_scalar` (all restart 53 A5) and of
  `vec_dot_pure` (restart 53 A1). Restart 53 protected the dot-product
  paths and the scalar-broadcast paths but the **Hadamard** (element-
  wise) integer multiply was missed.
- **Sibling sweep**:
  | Function | File:line | Per-element i64 + saturation? |
  |---|---|---|
  | `ti1d_add` | 440-454 | NO (subtle: sum of two i32 fits in i33, only one bit of headroom — still silently wraps at +INT32_MAX + 1 to INT32_MIN) |
  | `ti1d_sub` | 458-472 | NO (same: a - b wraps for boundary i32 values) |
  | `ti1d_mul` | 476-490 | NO — **WORST**: a single 46341 × 46341 multiply wraps |
  | `ti1d_add_scalar` | 793-811 | YES (restart 53 A5) |
  | `ti1d_mul_scalar` | 816-834 | YES (restart 53 A5) |
  | `ti1d_axpy` | 349-368 | YES (restart 53 A5) |
- **Nearby-but-safe sites**:
  - `tf1d_add`/`tf1d_sub`/`tf1d_mul` (888-938): f32 math, no integer
    overflow concern.
  - `ti1d_relu` (423-437): pure max, no multiply.
- **Suggested regression**:
  `test_stage35_restart54_ti1d_mul_hadamard_saturates_on_i32_overflow`
  (build two 1-element ti1d with value 46341, call `ti1d_mul`, assert
  result is INT32_MAX rather than the wrapped value).
- **Severity rationale**: HIGH — element-wise Hadamard is a primary
  primitive for any tensor caller that doesn't go through `ti2d_matmul`
  / dot. The restart-53 narrative says the campaign was sweeping all
  i32 product paths; this one was missed.

### A3 (HIGH): `vec_zip_mul` element-wise vec multiply silently wraps

- **File:function:line**: `helixc/stdlib/iterators.hx:222-230` (`vec_zip_mul`)
- **Bug family**: Same as A2 — `vec_zip_mul` is the iterators.hx mirror
  of `ti1d_mul`. Sibling of `vec_dot`/`vec_dot_pure` (restart 53 A1).
- **Sibling sweep**:
  | Function | File:line | i64 + saturation? |
  |---|---|---|
  | `vec_dot` | 362-374 | YES (restart 53 A1) |
  | `vec_dot_pure` | 1545-1557 | YES (restart 53 A1) |
  | `vec_zip_add` | 212-220 | NO — silent wrap on overflow |
  | `vec_zip_sub` | 316-324 | NO — silent wrap on boundary |
  | `vec_zip_mul` | 222-230 | NO — **WORST**: per-element multiply wraps |
- **Nearby-but-safe sites**: `vec_zip_min`/`vec_zip_max` (376-398),
  `vec_zip_eq`/`vec_zip_lt`/`vec_zip_gt`/etc. (720-851) — all
  comparison-only, no arithmetic.
- **Suggested regression**:
  `test_stage35_restart54_vec_zip_mul_saturates_on_i32_overflow`.
- **Severity rationale**: HIGH — `vec_zip_mul` is documented as the
  basic element-wise product. Following the restart-53 standard of
  saturating any `a[i] * b[i]` site.

### A4 (HIGH): `vec_window_sum` rolling accumulator stays in i32

- **File:function:line**: `helixc/stdlib/iterators.hx:985-1008` (`vec_window_sum`)
- **Bug family**: i64-saturation sibling — restart 53 A2 covered
  `vec_cumsum`, `vec_mean`, `vec_sum_pure`, `vec_abs_sum` but missed
  the rolling-window sum, which uses **the same `acc + arena_get` /
  `acc - arena_get` pattern**.
- **Sibling sweep**:
  | Function | File:line | i64 + saturation? |
  |---|---|---|
  | `vec_cumsum` | 560-574 | YES (restart 53 A2) |
  | `vec_sum_pure` | 1526-1538 | YES (restart 53 A2) |
  | `vec_abs_sum` | 403-414 | YES (restart 53 A2) |
  | `vec_mean` | 739-754 | YES (restart 53 A2) |
  | `vec_window_sum` | 985-1008 | NO — rolling sum stays in i32 |
  | `vec_sum_in_range` | 1196-1204 | NO — same `total + arena_get` in i32 |
- **Nearby-but-safe sites**: `vec_window_max`/`vec_window_min`
  (1081-1124) — max/min only, no accumulator.
- **Suggested regression**:
  `test_stage35_restart54_vec_window_sum_saturates_on_i32_overflow`
  (3 elements each at INT32_MAX/2, window=2 → first window sum
  saturates to INT32_MAX rather than wraps; second window also
  saturates).
- **Severity rationale**: HIGH — the rolling-sum pattern compounds the
  problem: an i32 wrap mid-window then gets *subtracted* on the next
  slide, producing arbitrary garbage in *every subsequent* output slot,
  not just the wrapped one.

### A5 (MEDIUM): `vec_l1_distance` and `vec_l2_squared_distance` i32 accumulator

- **File:function:line**:
  - `helixc/stdlib/iterators.hx:503-512` (`vec_l1_distance`)
  - `helixc/stdlib/iterators.hx:515-524` (`vec_l2_squared_distance`)
- **Bug family**: i64-saturation sibling — restart 53 A2 fixed
  `vec_sum_squares` to i64 + saturate, but the **distance** companions
  (which do the same `d*d` accumulation, just on `a[i] - b[i]`) were
  missed.
- **Sibling sweep**:
  | Function | File:line | i64 + saturation? |
  |---|---|---|
  | `vec_sum_squares` | 418-429 | YES (already saturated) |
  | `ti1d_l2_norm_sq` | 1090-1107 | YES (restart 51 A5) |
  | `ti1d_l1_norm` | 1069-1085 | YES (restart 51 A4) |
  | `mae_loss` | 1024-1041 | YES (restart 51 A6) |
  | `mse_loss` | 83-99 | YES (restart 51 A1) |
  | `vec_l1_distance` | 503-512 | NO — i32 sum-of-abs-diff |
  | `vec_l2_squared_distance` | 515-524 | NO — i32 sum-of-squared-diff (single `d=46341` wraps) |
- **Nearby-but-safe sites**: `vec_max_abs` (527-537) — picks single max,
  no accumulation.
- **Suggested regression**:
  `test_stage35_restart54_vec_l2_squared_distance_saturates` (two
  1-element vecs `a=46341`, `b=0`, assert result is INT32_MAX, not
  wrapped negative).
- **Severity rationale**: MEDIUM — these are standard ML distance
  primitives (k-means, nearest-neighbour). Same wrap risk as
  `vec_sum_squares` which the campaign already fixed. Not HIGH because
  callers comparing distances often only need monotonicity, but a
  wrapped negative compares strictly less than every positive distance,
  so the bug *inverts* nearest-neighbour ranking once the comparison
  involves a wrapped accumulator.

### A6 (MEDIUM): `lin_reg_grad_w` / `lin_reg_grad_b` / `sgd_step_scalar` raw i32

- **File:function:line**:
  - `helixc/stdlib/nn.hx:156-161` (`lin_reg_grad_w`)
  - `helixc/stdlib/nn.hx:165-170` (`lin_reg_grad_b`)
  - `helixc/stdlib/nn.hx:118-120` (`sgd_step_scalar`)
- **Bug family**: i64-saturation siblings — the *array* versions
  (`sgd_step_array`) got the per-element i64 + saturation in restart 53
  A7, but the *scalar* helpers exposed at the same API level didn't.
- **Sibling sweep**:
  | Function | File:line | i64 + saturation? |
  |---|---|---|
  | `sgd_step_array` | 129-149 | YES (restart 53 A7) |
  | `sgd_step_scalar` | 118-120 | NO — `w - lr * g` wraps for `lr * g >= INT32_MAX` |
  | `lin_reg_grad_w` | 156-161 | NO — `w * x + b` and `2 * err * x` all i32 |
  | `lin_reg_grad_b` | 165-170 | NO — `w * x + b` and `2 * err` both i32 |
- **Nearby-but-safe sites**: `count_correct` (1066-1081), `argmax` /
  `argmin` (58-75, 998-1015) — no arithmetic accumulators.
- **Suggested regression**:
  `test_stage35_restart54_lin_reg_grad_w_saturates_on_i32_overflow`
  (call with `w = x = 50000, b = 0, target = 0`; assert returned
  gradient is saturated rather than wrapped).
- **Severity rationale**: MEDIUM — these scalar helpers are documented
  as demo / educational. Real training uses `sgd_step_array`. But the
  whole point of restart 53 was to fix the family, and leaving the
  scalar mirror unfixed means a follow-up audit will keep finding it.

### A7 (LOW): `vec_zip_add`/`vec_zip_sub`/`vec_map_*_scalar` + `vec_offset_inplace`/`vec_scale_inplace` per-element i32 wrap

- **File:function:line** (cluster):
  - `helixc/stdlib/iterators.hx:212-220` (`vec_zip_add`)
  - `helixc/stdlib/iterators.hx:316-324` (`vec_zip_sub`)
  - `helixc/stdlib/iterators.hx:156-164` (`vec_map_add_scalar`)
  - `helixc/stdlib/iterators.hx:166-174` (`vec_map_mul_scalar`)
  - `helixc/stdlib/iterators.hx:465-473` (`vec_scale_inplace`)
  - `helixc/stdlib/iterators.hx:475-483` (`vec_offset_inplace`)
  - `helixc/stdlib/iterators.hx:1145-1156` (`vec_pairwise_diff`)
  - `helixc/stdlib/iterators.hx:1360-1371` (`vec_pairwise_sum`)
  - `helixc/stdlib/iterators.hx:1375-1383` (`vec_offset_alloc`)
  - `helixc/stdlib/iterators.hx:142-154` (`vec_fold_op` op=0 add / op=1 mul)
  - `helixc/stdlib/iterators.hx:1196-1204` (`vec_sum_in_range`)
- **Bug family**: Same as A2/A3/A6 — every per-element arithmetic op
  emits raw i32 results. Strictly speaking restart 51 A5 and restart 53
  established the discipline that **any** per-element write that can
  exceed i32 should saturate.
- **Sibling sweep**: all 11 sites above use the i32 pattern
  `__arena_push(get(a) + k)` or `__arena_set(...)` without saturation.
  Already-fixed companions: `ti1d_add_scalar`, `ti1d_mul_scalar`,
  `ti1d_axpy` (all in tensor.hx). The iterators.hx mirrors and the
  `vec_fold_op` reducer were missed.
- **Nearby-but-safe sites**: `vec_clamp_inplace` (431-440) — clamp
  cannot overflow. `vec_fill_inplace`, `vec_swap_inplace` — pure copy.
  `vec_map_neg` and `vec_negate_inplace` — restart 51 A5 already
  saturated.
- **Suggested regression**: one canary covering the family:
  `test_stage35_restart54_iterators_arithmetic_helpers_saturate_on_i32_overflow`
  exercising `vec_zip_add`/`vec_zip_sub`/`vec_zip_mul`/`vec_window_sum`/
  `vec_l2_squared_distance` at the boundary.
- **Severity rationale**: LOW — these are per-element ops where the
  *caller* observes the wrap on the immediate next read, rather than
  the silent-multiply-then-sum pattern that hides the wrap inside an
  accumulator. So the wrap is "noisier" and more recoverable. Still,
  fixing the family as a single sweep is cheaper than patching each
  site individually next restart.

## Clean families

- **Forged-handle validators (13)**: all magic-bearing handle ok
  predicates (`t1d_capacity_ok`, `t2d_shape_ok`, `hashmap_ok`,
  `bfs_ok`, `visited_ok`, `pq_ok`, `wm_ok`, `ep_ok`, `wmt_ok`,
  `wml_ok`, `tree_node_ok`, `bindings_storage_ok`, `rev_tape_valid` +
  `rev_adj_cap`) still gate on magic + footer + `arena_span_in_tensor_payload`.
  Re-verified vs restart 47 baseline; no new handle introduced since.
- **Magic-constant uniqueness**: 13 distinct constants
  (1001001 t1d, 2002001 t2d, 3003001 rev_tape, 4004001 wm, 5005001 ep,
  6006001 wmt, 6007001 wml, 6106101 bfs, 6206201 visited, 6306301 pq,
  7007001 hashmap, 7008001 bindings, 7107001 tree_node). Confirmed by
  reading every `_magic()` site.
- **Arena-span overflow guards**: all `*_ok` validators use the same
  `start > INT32_MAX - slot_count` guard pattern; re-verified.
- **Float-domain guards**: `__sqrt` (133), `__sqrt_f64` (188-189),
  `__log_stable` (94-95), `__log_stable_f64` (265-266) all guard
  `x <= 0`. `__adam_step` (534-541) and `adam_f32_step`
  (262-268) NaN-fail-closed. `clip_grad_norm_f32` (196) NaN-fail-closed
  (restart 51 A2). `softmax_layer` (704), `layer_norm_f32` (612),
  `dense_classifier_sgd_step_f32` (844), `attention_softmax_f32`
  (588-589) all NaN-fail-closed. Re-verified vs restart 51 baseline.
- **Stale-state resurrection in rewind/clear/reset surfaces**:
  `wm_clear` (89-96) zeros size+tick so stale entries are ignored
  (count-bounded iteration). `hashmap_clear` (183-193) zeros occupancy
  flag — keys/values become dead. `bindings_rewind` (293-313) writes
  `tree_invalid_value()` over each cleared slot. `ep_record` (274-296)
  writes over the slot before incrementing count. `rev_alloc_adjoints`
  (252-284) re-derives the snapshot footer guard so any post-allocation
  tape edit invalidates the adjoint buffer. Clean.

---

LANE_A_TOTAL: 7 findings (H=4 M=2 L=1) | 5 clean families
