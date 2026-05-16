# Lane A Audit Report — Stage 35 Restart 55

**HEAD**: `e34b4d6 Fix Stage 35 fifty-fifth restart findings`
**Scope**: Runtime / stdlib safety. Read-only audit; fixes applied separately.
**Status**: Reconstructed retroactively from commit `218ffd0` source diff (the restart 55 fix sweep landed without paired lane docs). The reconstructed audit reflects what restart 55 actually found and fixed; restart 57's catch-up sweep (Increment 76) added this stub plus the missing regression canary.

## Summary

Reviewed `helixc/stdlib/*.hx`, with a focus on transcendentals after restart 54's reverse-AD saturation sweep brought the f32/f64 numerics families back into audit scope. Found **1 finding**.

## Findings

### A1 HIGH — `__sin`, `__cos`, `__sin_f64`, `__cos_f64` lacked range reduction

- **Files / functions**: `helixc/stdlib/transcendentals.hx` lines ~71-91 (`__sin`, `__cos`), lines ~242-264 (`__sin_f64`, `__cos_f64`).
- **Bug**: All four functions evaluated the 4-term Taylor series at the raw input `x`. The Taylor series is only accurate for `|x| < π/2 ≈ 1.57`; outside that band the truncated series error grows polynomially and outside `|x| > 2π` the result is meaningless.
- **Realistic trigger**: any signal-processing or rotation accumulator that lets phase grow without bound (typical idiom in DSP, robotics, autodiff over trig). The user-visible symptom is silently wrong f32 outputs that propagate into downstream numerics.
- **Sibling sweep**: `__exp` and `__exp_f64` already have explicit range reduction (lines ~50-65 and ~210-230). Restart 54 closed the reverse-AD saturation gap (Increment 73 A1) which brought the f32/f64 numerics surface back into review.
- **Fix**: Range-reduce `x` into `[-π, π]` before the Taylor series using `k = round(x / 2π); xr = x - k * 2π`. The `round` step uses the `(+0.5 / -0.5) -> i32` cast trick to stay arena-pure (no extern math calls). The f64 mirror uses the higher-precision `6.283185307179586_f64` constant.
- **Severity**: HIGH (silent numerical corruption, not a fail-closed wrap).
- **Regression canary**: `test_stage35_restart55_sin_range_reduces_at_large_angle`, `test_stage35_restart55_cos_range_reduces_at_large_angle`, `test_stage35_restart55_sin_f64_range_reduces_at_large_angle`. Each evaluates the function at `5π` (which should be close to `0` for `sin` / `−1` for `cos`).

## Clean families swept

- Forged handles: restart 54's `autodiff_reverse.hx` sweep verified the magic-bearing validators on the reverse-AD tape; nothing new since.
- Arena span validation: no new validators added.
- Magic-constant uniqueness: 13 distinct (verified via restart 46 invariant test).
- Stale state resurrection: no new rewind / restore / reset surfaces.
- Fail-closed numerical helpers: Adam clamp, layer_norm var+eps==0, autodiff div-by-zero rules all in place.
- i64-saturation discipline (the dominant restart 47/51/52/53/54 family): with reverse-AD closed in restart 54, no new missed siblings discovered in restart 55's pass.

---

LANE_A_TOTAL: 1 finding (H=1 M=0 L=0) | 6 clean families
