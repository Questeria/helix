# Lane A Audit Report — Stage 35 Restart 56

**HEAD**: `218ffd0 Fix Stage 35 fifty-sixth restart findings`
**Scope**: Runtime / stdlib safety. Read-only audit; fixes applied separately.
**Status**: Reconstructed retroactively from commit `278d46a` source diff, filed by restart 57's catch-up sweep.

## Summary

Reviewed `helixc/stdlib/*.hx`, with a focus on integer max-abs helpers (sibling of restart 51 A5's INT32_MIN sweep) and float-tensor reductions (NaN-poison risk). Found **3 findings**.

## Findings

### A1 HIGH — `tf1d_sum` NaN poisons the entire sum

- **File / function**: `helixc/stdlib/tensor.hx` `tf1d_sum`.
- **Bug**: A single NaN slot in the input array causes the running `total` to become NaN, and NaN + anything = NaN, so the entire reduction returns NaN. Any downstream consumer that reads `tf1d_sum` then sees NaN for a single bad input.
- **Realistic trigger**: float arrays produced by an upstream op that already fail-closed on a NaN slot (e.g. softmax with one bad logit). The fail-closed precedents in `softmax_layer`, `layer_norm_f32`, `clip_grad_norm_f32`, and `adam_f32_step` set the convention that NaN should be treated as "garbage in one slot," not "garbage in every output."
- **Sibling sweep candidates**: `tf1d_dot`, `tf1d_l1_norm`, `tf1d_max_abs`, `tf1d_sum_in_range`. **Not applied in restart 56's fix** — the original commit comment claimed it but the code change was only to `tf1d_sum`. Restart 57's catch-up trimmed the comment; restart 58 Lane A should pick up the actual sibling sweep as a deliberate work item.
- **Fix**: NaN-skip via `let v = ...; if v == v { total = total + v; }`. NaN is the only value not equal to itself; the conditional skips it.
- **Severity**: HIGH (silent corruption of every downstream consumer for a single bad input).
- **Regression canary**: `test_stage35_restart56_tf1d_sum_nan_skip_fails_closed`.

### A2 HIGH — `ti1d_max_abs` returns 0 when input contains INT32_MIN

- **File / function**: `helixc/stdlib/tensor.hx` `ti1d_max_abs`.
- **Bug**: `0 - INT32_MIN` wraps back to INT32_MIN (i32 two's-complement). The function then runs `if av > best { best = av; }` with `av = INT32_MIN` and `best = 0`; `INT32_MIN > 0` is false, so the negative slot is silently dropped. If the only large-magnitude value in the array is INT32_MIN, the function returns 0 instead of INT32_MAX.
- **Realistic trigger**: any int32 vector that flows through `vec_negate_inplace` (restart 51 A5) or sees user-provided INT32_MIN constants.
- **Sibling sweep**: `vec_max_abs` in `iterators.hx` has the same bug (A3 below). Both fixed in the same sweep.
- **Fix**: `let av = if v == ((0 - 2147483647) - 1) { 2147483647 } else { if v < 0 { 0 - v } else { v } };` — explicit INT32_MIN test before the negate.
- **Severity**: HIGH (silent corruption; the answer is wrong, not fail-closed-wrong).
- **Regression canary**: shared with A3, `test_stage35_restart56_max_abs_saturates_on_int32_min`.

### A3 MEDIUM — `vec_max_abs` companion of A2

- **File / function**: `helixc/stdlib/iterators.hx` `vec_max_abs`.
- **Bug**: identical to A2 but in the iterators.hx companion function.
- **Fix**: same INT32_MIN special-case.
- **Severity**: MEDIUM (companion of a HIGH; rated lower because callers usually go through the tensor.hx variant first).
- **Regression canary**: shared with A2.

## Clean families swept

- Forged handles, magic uniqueness, arena span validation: nothing new since restart 47's broad sweep.
- Fail-closed numerical helpers in transcendentals: restart 55 closed the sin/cos range-reduction gap.
- i64-saturation discipline: still clean across the new code surfaces touched in restart 55.

---

LANE_A_TOTAL: 3 findings (H=2 M=1 L=0) | 3 clean families
