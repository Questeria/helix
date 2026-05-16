# Lane A Audit Report — Stage 35 Restart 52

**HEAD**: `a4ad9a0 Fix Stage 35 fifty-first restart findings`
**Scope**: Runtime / stdlib safety. Read-only audit; fixes applied separately.

## Summary

Reviewed all 16 stdlib files under `helixc/stdlib/`. Confirmed prior restart 45-51 fixes remain in place. Found **1 new HIGH finding**: a missed sibling in the i64 saturation family that restart 51 A3 (`ti1d_dot`) opened — the 2D `ti2d_matvec` and `ti2d_matmul` were not swept.

---

## A1 — `ti2d_matvec` + `ti2d_matmul` integer accumulator silently wraps — HIGH

**Files**: `helixc/stdlib/tensor.hx:381-411` (`ti2d_matvec`), `:735-757` (`ti2d_matmul`)
**Bug family**: Integer accumulator overflow (sibling of restart 51 A3 `ti1d_dot`)

**Issue**: Both functions use `let mut acc: i32 = 0` and then sum `w[r,c] * x[c]` (or `a[r,k] * b[k,c]`) across the inner dimension. A single term `w*x` with `|w| ≈ |x| ≈ 46341` already overflows i32; the running accumulator wraps faster. Restart 51 A3 fixed `ti1d_dot` with a per-iteration i64 accumulator + INT32 saturation; the same pattern needed to extend to the 2D paths.

**Sibling sweep**:
| Function | Saturated? |
|---|---|
| `ti1d_sum` | Yes (restart 51 A2) |
| `ti1d_dot` | Yes (restart 51 A3) |
| `ti1d_l1_norm` | Yes (restart 51 A4) |
| `ti1d_l2_norm_sq` | Yes (restart 51 A5) |
| `ti1d_prod` | Yes (restart 50 A3) |
| `ti2d_matvec` | NO — fix needed |
| `ti2d_matmul` | NO — fix needed |

**Suggested fix**: Lift `acc` to i64, clamp to `[INT32_MIN, INT32_MAX]` per iteration, cast back to i32 on write. Identical pattern to the `ti1d_dot` fix.

**Suggested canary**: `test_stage35_restart52_ti2d_matvec_saturates_on_i32_overflow` + `test_stage35_restart52_ti2d_matmul_saturates_on_i32_overflow`.

## Clean families swept

- Forged-handle validators: all `*_ok` / `*_valid` (13 magic-bearing handles) correctly call `arena_span_in_tensor_payload`. Clean.
- Magic-constant uniqueness: 13 distinct constants. Clean (pinned by `test_stage35_stdlib_magic_constants_unique`).
- Arena-span overflow guards: all validators have the `start > INT32_MAX - len` guard. Clean.
- Stale-state resurrection in rewind/clear/reset: all sites zero stale slots or guard via occupancy/count. Clean.
- Float-domain guards on `__sqrt` / `__log` / `__log_f64`: all callers fail-closed at singularities. Clean.

---

LANE_A_TOTAL: 1 finding (H=1 M=0 L=0) | 5 clean families
