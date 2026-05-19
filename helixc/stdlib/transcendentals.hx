// helixc/stdlib/transcendentals.hx — auto-included transcendental functions.
//
// These provide Taylor-series approximations of common math functions plus
// a Newton-iteration sqrt. The autodiff engines (autodiff.py /
// autodiff_reverse.py) special-case calls to these names with the analytic
// chain rules — calling __exp / __log / __sin / __cos / __sqrt inside a
// loss function produces correct gradients.
//
// Range: each Taylor approximation is accurate for small |x| (roughly
// |x| < 1.5 for sin/cos, x near 1 for log, |x| < 4 for exp). Production
// code would do range reduction; v0.1 keeps the surface simple.

// Internal: 8-term Taylor series for exp(r), accurate for |r| < ~1.
@pure fn __exp_taylor(r: f32) -> f32 {
    let x2 = r * r;
    let x3 = x2 * r;
    let x4 = x2 * x2;
    let x5 = x4 * r;
    let x6 = x3 * x3;
    let x7 = x6 * r;
    1.0 + r
        + x2 * 0.5
        + x3 * 0.16666667
        + x4 * 0.04166667
        + x5 * 0.00833333
        + x6 * 0.00138889
        + x7 * 0.00019841
}

// Range-reduced exp(x) = 2^k * exp(r) where x = k*ln2 + r, |r| < ln2/2.
// Accurate for any x in roughly [-50, 50] (covers all NN logits).
//   ln(2) ≈ 0.69314718
//   1/ln(2) ≈ 1.44269504
@pure fn __exp(x: f32) -> f32 {
    // k = round(x / ln2). The cast `as i32` truncates toward zero, so for
    // positive `x*1/ln2` we get the right answer adding 0.5 then casting;
    // for negative `x*1/ln2` truncation toward zero gives the WRONG
    // direction. We compute floor(x/ln2 + 0.5) by handling the two signs
    // separately.
    //   z = x * 1/ln2 + 0.5
    //   if z >= 0: k = (z as i32)
    //   if z <  0: k = (z as i32) - 1   if (z as i32) as f32 > z else (z as i32)
    let z = x * 1.44269504 + 0.5;
    let k_trunc = z as i32;
    let k = if z >= 0.0 { k_trunc }
            else { if (k_trunc as f32) > z { k_trunc - 1 } else { k_trunc } };
    // r = x - k*ln2
    let r = x - (k as f32) * 0.69314718;
    let k_corrected = k;
    // exp(r) via Taylor; accurate for |r| < ln2/2 ≈ 0.347
    let exp_r = __exp_taylor(r);
    // 2^k via the integer power loop (cap at ±48 to stay in f32 range).
    let kc = if k_corrected > 48 { 48 }
             else { if k_corrected < (0 - 48) { 0 - 48 } else { k_corrected } };
    let mut scale: f32 = 1.0;
    if kc >= 0 {
        let mut i: i32 = 0;
        while i < kc { scale = scale * 2.0; i = i + 1; }
    } else {
        let neg_kc = 0 - kc;
        let mut i: i32 = 0;
        while i < neg_kc { scale = scale * 0.5; i = i + 1; }
    }
    scale * exp_r
}

// Restart 56 A1: range-reduce x into [-π, π] before the Taylor series.
// The 4-term series is only accurate for |x| < π/2 ≈ 1.57; without
// reduction, |x| > 2π produces nonsense. Mirrors __exp range reduction
// discipline. Two-pi constant inlined as 6.28318530718; round-via-i32-
// cast handles |x| up to ~2 billion before integer overflow becomes a
// concern (well past any realistic angle).
@pure fn __sin(x: f32) -> f32 {
    let two_pi = 6.28318530718_f32;
    let k = ((x / two_pi) + (if x >= 0.0_f32 { 0.5_f32 } else { 0.0_f32 - 0.5_f32 })) as i32;
    let xr = x - (k as f32) * two_pi;
    // sin(x) = x - x³/3! + x⁵/5! - x⁷/7!
    let x2 = xr * xr;
    let x3 = x2 * xr;
    let x5 = x3 * x2;
    let x7 = x5 * x2;
    xr - x3 * 0.16666667_f32 + x5 * 0.00833333_f32 - x7 * 0.00019841_f32
}

@pure fn __cos(x: f32) -> f32 {
    let two_pi = 6.28318530718_f32;
    let k = ((x / two_pi) + (if x >= 0.0_f32 { 0.5_f32 } else { 0.0_f32 - 0.5_f32 })) as i32;
    let xr = x - (k as f32) * two_pi;
    // cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6!
    let x2 = xr * xr;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    1.0_f32 - x2 * 0.5_f32 + x4 * 0.04166667_f32 - x6 * 0.00138889_f32
}

// Cycle 3 R1 fix batch 20 (RT MEDIUM-10): pre-fix returned garbage Taylor
// expansion for x <= 0 (y = x-1 < -1, polynomial diverges). Caller passing
// x=0 got a wildly negative value indistinguishable from a "valid" log.
// Post-fix: return NaN (0.0/0.0) for x <= 0 as the canonical out-of-band
// signal. For x near 1 (use case the function was designed for), behavior
// unchanged.
@pure fn __log(x: f32) -> f32 {
    if x <= 0.0_f32 { 0.0_f32 / 0.0_f32 }
    else {
        // log(x) for x near 1: log(1+y) = y - y²/2 + y³/3 - y⁴/4 + y⁵/5
        let y = x - 1.0;
        let y2 = y * y;
        let y3 = y2 * y;
        let y4 = y3 * y;
        let y5 = y4 * y;
        y - y2 * 0.5 + y3 * 0.33333333 - y4 * 0.25 + y5 * 0.2
    }
}

@pure fn __log_stable(x: f32) -> f32 {
    if x <= 0.0 {
        0.0 - 1000000.0
    } else {
        let mut m: f32 = x;
        let mut k: i32 = 0;
        while m > 1.41421356 {
            m = m * 0.5;
            k = k + 1;
        }
        while m < 0.70710678 {
            m = m * 2.0;
            k = k - 1;
        }
        let z = (m - 1.0) / (m + 1.0);
        let z2 = z * z;
        let z3 = z * z2;
        let z5 = z3 * z2;
        let z7 = z5 * z2;
        let z9 = z7 * z2;
        let z11 = z9 * z2;
        let z13 = z11 * z2;
        let core = 2.0 * (z
            + z3 * 0.33333333
            + z5 * 0.2
            + z7 * 0.14285714
            + z9 * 0.11111111
            + z11 * 0.09090909
            + z13 * 0.07692308);
        core + (k as f32) * 0.69314718
    }
}

@pure fn __sqrt(x: f32) -> f32 {
    // Defined as 0 for x <= 0 to avoid divide-by-zero in Newton's method
    // (y0 = x*0.5 + 0.5; for x = -1, y0 = 0 -> x/y0 = inf, propagates NaN
    // through the iteration). For x = 0, y0 = 0.5, but x/y0 = 0 keeps the
    // sequence at 0.25, 0.125, ... never reaching 0. We define sqrt(0) = 0
    // explicitly for both edges.
    if x <= 0.0 {
        0.0
    } else {
        let y0 = x * 0.5 + 0.5;
        let y1 = (y0 + x / y0) * 0.5;
        let y2 = (y1 + x / y1) * 0.5;
        let y3 = (y2 + x / y2) * 0.5;
        let y4 = (y3 + x / y3) * 0.5;
        y4
    }
}

// =========================================================================
// f64 transcendentals (Phase 1.5). Mirror the f32 versions above with f64
// arithmetic so callers that pick f64 don't lose precision. Backed by the
// wide-load codegen fix in SELECT/BR/LOAD_VAR/STORE_VAR.
// =========================================================================

@pure fn __exp_taylor_f64(r: f64) -> f64 {
    let x2 = r * r;
    let x3 = x2 * r;
    let x4 = x2 * x2;
    let x5 = x4 * r;
    let x6 = x3 * x3;
    let x7 = x6 * r;
    1.0_f64 + r
        + x2 * 0.5_f64
        + x3 * 0.16666666666666666_f64
        + x4 * 0.041666666666666664_f64
        + x5 * 0.008333333333333333_f64
        + x6 * 0.001388888888888889_f64
        + x7 * 0.0001984126984126984_f64
}

@pure fn __exp_f64(x: f64) -> f64 {
    let z = x * 1.4426950408889634_f64 + 0.5_f64;
    let k_trunc = z as i32;
    let k = if z >= 0.0_f64 { k_trunc }
            else { if (k_trunc as f64) > z { k_trunc - 1 } else { k_trunc } };
    let r = x - (k as f64) * 0.6931471805599453_f64;
    let exp_r = __exp_taylor_f64(r);
    let kc = if k > 1023 { 1023 }
             else { if k < (0 - 1023) { 0 - 1023 } else { k } };
    let mut scale: f64 = 1.0_f64;
    if kc >= 0 {
        let mut i: i32 = 0;
        while i < kc { scale = scale * 2.0_f64; i = i + 1; }
    } else {
        let neg_kc = 0 - kc;
        let mut i: i32 = 0;
        while i < neg_kc { scale = scale * 0.5_f64; i = i + 1; }
    }
    scale * exp_r
}

@pure fn __sqrt_f64(x: f64) -> f64 {
    if x <= 0.0_f64 {
        0.0_f64
    } else {
        let y0 = x * 0.5_f64 + 0.5_f64;
        let y1 = (y0 + x / y0) * 0.5_f64;
        let y2 = (y1 + x / y1) * 0.5_f64;
        let y3 = (y2 + x / y2) * 0.5_f64;
        let y4 = (y3 + x / y3) * 0.5_f64;
        let y5 = (y4 + x / y4) * 0.5_f64;
        let y6 = (y5 + x / y5) * 0.5_f64;
        y6
    }
}

@pure fn __sigmoid_f64(x: f64) -> f64 {
    if x > 30.0_f64 { 1.0_f64 }
    else { if x < 0.0_f64 - 30.0_f64 { 0.0_f64 }
           else { 1.0_f64 / (1.0_f64 + __exp_f64(0.0_f64 - x)) }
    }
}

@pure fn __relu_f64(x: f64) -> f64 {
    if x > 0.0_f64 { x } else { 0.0_f64 }
}

@pure fn __abs_f64(x: f64) -> f64 {
    if x < 0.0_f64 { 0.0_f64 - x } else { x }
}

@pure fn __min_f64(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

@pure fn __max_f64(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

@pure fn __clamp_f64(x: f64, lo: f64, hi: f64) -> f64 {
    if x < lo { lo } else { if x > hi { hi } else { x } }
}

// Restart 56 A1 (f64 mirror): range-reduce x into [-π, π] before Taylor.
@pure fn __sin_f64(x: f64) -> f64 {
    let two_pi = 6.283185307179586_f64;
    let k = ((x / two_pi) + (if x >= 0.0_f64 { 0.5_f64 } else { 0.0_f64 - 0.5_f64 })) as i32;
    let xr = x - (k as f64) * two_pi;
    let x2 = xr * xr;
    let x3 = x2 * xr;
    let x5 = x3 * x2;
    let x7 = x5 * x2;
    xr - x3 * 0.16666666666666666_f64
       + x5 * 0.008333333333333333_f64
       - x7 * 0.0001984126984126984_f64
}

@pure fn __cos_f64(x: f64) -> f64 {
    let two_pi = 6.283185307179586_f64;
    let k = ((x / two_pi) + (if x >= 0.0_f64 { 0.5_f64 } else { 0.0_f64 - 0.5_f64 })) as i32;
    let xr = x - (k as f64) * two_pi;
    let x2 = xr * xr;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    1.0_f64 - x2 * 0.5_f64
            + x4 * 0.041666666666666664_f64
            - x6 * 0.001388888888888889_f64
}

// Cycle 3 R1 fix batch 20 (RT MEDIUM-10): f64 sibling of __log; same
// domain guard added. Returns NaN for x <= 0.
@pure fn __log_f64(x: f64) -> f64 {
    if x <= 0.0_f64 { 0.0_f64 / 0.0_f64 }
    else {
        let y = x - 1.0_f64;
        let y2 = y * y;
        let y3 = y2 * y;
        let y4 = y3 * y;
        let y5 = y4 * y;
        let y6 = y5 * y;
        let y7 = y6 * y;
        y - y2 * 0.5_f64
          + y3 * 0.3333333333333333_f64
          - y4 * 0.25_f64
          + y5 * 0.2_f64
          - y6 * 0.16666666666666666_f64
          + y7 * 0.14285714285714285_f64
    }
}

@pure fn __log_stable_f64(x: f64) -> f64 {
    if x <= 0.0_f64 {
        0.0_f64 - 1000000.0_f64
    } else {
        let mut m: f64 = x;
        let mut k: i32 = 0;
        while m > 1.4142135623730951_f64 {
            m = m * 0.5_f64;
            k = k + 1;
        }
        while m < 0.7071067811865476_f64 {
            m = m * 2.0_f64;
            k = k - 1;
        }
        let z = (m - 1.0_f64) / (m + 1.0_f64);
        let z2 = z * z;
        let z3 = z * z2;
        let z5 = z3 * z2;
        let z7 = z5 * z2;
        let z9 = z7 * z2;
        let z11 = z9 * z2;
        let z13 = z11 * z2;
        let core = 2.0_f64 * (z
            + z3 * 0.3333333333333333_f64
            + z5 * 0.2_f64
            + z7 * 0.14285714285714285_f64
            + z9 * 0.1111111111111111_f64
            + z11 * 0.09090909090909091_f64
            + z13 * 0.07692307692307693_f64);
        core + (k as f64) * 0.6931471805599453_f64
    }
}

// Sigmoid: bounded activation in (0, 1). Now backed by range-reduced exp.
// Asymptotic short-circuit only for very large |x|.
@pure fn __sigmoid(x: f32) -> f32 {
    if x > 30.0 { 1.0 }
    else { if x < 0.0 - 30.0 { 0.0 }
           else { 1.0 / (1.0 + __exp(0.0 - x)) }
    }
}

// ReLU: x if x > 0, else 0. Common neural-net activation.
@pure fn __relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

// Trivial always-accept verifier. Used by grad_rev_all to write
// computed gradient values into reflection cells without per-update
// verification (gradients are derived deterministically; the safety
// gate is on the LATER weight update step, not on saving the gradient).
@verifier
fn __always_accept(h: i32, v: f32) -> i32 {
    1
}

@verifier
fn __always_accept_f64(h: i32, v: f64) -> i32 {
    1
}

// =========================================================================
// Math helpers — operate on f32 scalars.
// =========================================================================

@pure fn __abs(x: f32) -> f32 {
    if x < 0.0 { 0.0 - x } else { x }
}

@pure fn __min(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

@pure fn __max(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

@pure fn __clamp(x: f32, lo: f32, hi: f32) -> f32 {
    if x < lo { lo } else { if x > hi { hi } else { x } }
}

@pure fn __sign(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { if x < 0.0 { 0.0 - 1.0 } else { 0.0 } }
}

// Floor: nearest integer ≤ x. Cast-and-correct via i32. For |x| outside
// i32 range (~±2.1e9) the cast invokes UB; we guard by passing through
// untouched (large floats are already integer-valued for any practical
// purpose since float spacing exceeds 1.0 above 2^23).
@pure fn __floor(x: f32) -> f32 {
    if x > 2000000000.0 { x }
    else { if x < 0.0 - 2000000000.0 { x }
           else {
               let i = x as i32;
               let f = i as f32;
               if f > x { f - 1.0 } else { f }
           }
    }
}

@pure fn __ceil(x: f32) -> f32 {
    if x > 2000000000.0 { x }
    else { if x < 0.0 - 2000000000.0 { x }
           else {
               let i = x as i32;
               let f = i as f32;
               if f < x { f + 1.0 } else { f }
           }
    }
}

// Integer power: x^n via iterative multiplication. Loop bounded at 16
// iterations as a safety cap. Returns 1.0 for n <= 0 OR n > 16 (out of
// range). Previously saturated silently to x^16 for n > 16, which
// disagreed with the docstring and silently produced wrong results.
@pure fn __powi(x: f32, n: i32) -> f32 {
    if n <= 0 { 1.0 }
    else { if n > 16 { 1.0 }
    else {
        let mut result: f32 = 1.0;
        let mut i: i32 = 0;
        while i < n {
            result = result * x;
            i = i + 1;
        }
        result
    }}
}

// Cycle 2 Batch RT fix batch 17 (silent-failure MEDIUM-5):
// Pre-fix: __powi returned 1.0 for BOTH n<=0 (mathematically correct,
// x^0 = 1) AND n>16 (out-of-range error). Polynomial-feature generator
// computing x^k for k in [0..32] silently got all-ones for k > 16,
// breaking downstream regression with NO error path.
// Post-fix: __powi_checked takes a caller-supplied sentinel for the
// out-of-range case. Caller distinguishes by passing e.g. -1.0e30_f32
// or any domain-impossible value. Original __powi preserved for
// backward compat with the existing "cap at 16" contract.
@pure fn __powi_checked(x: f32, n: i32, sentinel: f32) -> f32 {
    if n > 16 { sentinel }
    else { __powi(x, n) }
}

// =========================================================================
// Modern activation functions.
// =========================================================================

// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1). Range (-1, 1). Now that __exp
// is range-reduced, this is accurate over the full f32 range. Asymptotic
// short-circuit kept only for very large |x| to avoid producing inf/inf.
@pure fn __tanh(x: f32) -> f32 {
    if x > 20.0 { 1.0 }
    else { if x < 0.0 - 20.0 { 0.0 - 1.0 }
           else {
               let e2 = __exp(2.0 * x);
               (e2 - 1.0) / (e2 + 1.0)
           }
    }
}

// Softplus: smooth approximation of relu. Range: (0, ∞).
//   softplus(x) = log(1 + exp(x))
// __log is a 5-term Taylor around 1, accurate only for x ≈ 1 ⇒ this
// formulation works only for very small x. For larger x the asymptotic
// limit log(exp(x)) = x is exact to machine precision; for very negative
// x, exp(x) ≈ 0 so softplus(x) ≈ exp(x). Saturate via the standard
// numerical stable form: softplus(x) = max(x, 0) + log(1 + exp(-|x|)).
// Since __log is only good near 1, we approximate log(1+y) ≈ y for very
// small y (large |x|): the asymptote.
@pure fn __softplus(x: f32) -> f32 {
    if x > 20.0 { x }
    else { if x < 0.0 - 20.0 { __exp(x) }
           else {
               // |x| within the accurate range of __log(1+y) when
               // y = exp(x) - 1 stays small. For x in [-1, 1], y is
               // in [-0.63, 1.72] — passable. Outside [-1, 1] but
               // within [-15, 15] we use the symmetric formulation:
               //   softplus(x) = max(x, 0) + log(1 + exp(-|x|))
               // and approximate log(1+y) ≈ y - y²/2 for the small y.
               if x > 0.0 {
                   x + __log_stable(1.0 + __exp(0.0 - x))
               } else {
                   __log_stable(1.0 + __exp(x))
               }
           }
    }
}

// SiLU / Swish: x * sigmoid(x). Used in modern transformers.
// Smooth, non-monotonic, self-gated.
@pure fn __silu(x: f32) -> f32 {
    x * __sigmoid(x)
}

// GELU (Gaussian Error Linear Unit): tanh-based approximation per
// Hendrycks & Gimpel. Used in BERT, GPT-2, Llama, etc.
//   gelu(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
// √(2/π) ≈ 0.7978846
@pure fn __gelu(x: f32) -> f32 {
    let x3 = x * x * x;
    let inner = 0.7978846 * (x + 0.044715 * x3);
    0.5 * x * (1.0 + __tanh(inner))
}

// =========================================================================
// Loss functions (per-example; user sums across batches).
// =========================================================================

// Mean squared error: (pred - target)^2. Symmetric quadratic.
@pure fn __mse(pred: f32, target: f32) -> f32 {
    let d = pred - target;
    d * d
}

// Mean absolute error: |pred - target|. More robust to outliers.
@pure fn __mae(pred: f32, target: f32) -> f32 {
    __abs(pred - target)
}

// Binary cross-entropy: -[y log(p) + (1-y) log(1-p)]. p must be in (0, 1).
// We clamp away from 0/1 to avoid log(0) → -inf.
@pure fn __bce(p: f32, y: f32) -> f32 {
    let p_safe = __clamp(p, 0.000001, 0.999999);
    0.0 - (y * __log_stable(p_safe) + (1.0 - y) * __log_stable(1.0 - p_safe))
}

// Huber loss: quadratic for |d| < delta, linear beyond. Robust to outliers.
@pure fn __huber(pred: f32, target: f32, delta: f32) -> f32 {
    let d = __abs(pred - target);
    if d < delta { 0.5 * d * d } else { delta * (d - 0.5 * delta) }
}

// =========================================================================
// Deterministic PRNG (xorshift32). Seedable; reproducible across runs.
// Not cryptographic. Use for weight init, dropout masks, etc.
// =========================================================================

// Small LCG sized to fit safely in i32 multiplication.
// State range: [0, 32768). Period: at MOST 32768 — possibly less because
// the 0-state is remapped to 1, which can collapse two trajectories into
// one. Empirical period is verified by the corresponding test. Adequate
// for seeded reproducibility in test programs; NOT strong enough for
// statistical sampling. Previous 48271×x version overflowed i32 for any
// seed > ~44k; the 25173×x multiply peaks at 8.25e8, well under i32 max.
@pure fn __rand_step(s: i32) -> i32 {
    // Numerical Recipes parameters: s_{n+1} = (s_n * 25173 + 13849) mod 32768
    let raw = s * 25173 + 13849;
    let m = raw % 32768;
    if m < 0 { m + 32768 } else { if m == 0 { 1 } else { m } }
}

// Map a state to a float in [0, 1). State should be in [0, 32768).
@pure fn __rand_float(s: i32) -> f32 {
    (s as f32) * 0.000030517578125   // 1 / 32768
}

// Map a state to a float in [-bound, bound] — useful for weight init.
@pure fn __rand_uniform(s: i32, bound: f32) -> f32 {
    (__rand_float(s) - 0.5) * 2.0 * bound
}

// =========================================================================
// Optimizer-step helpers.
// =========================================================================

// SGD step: w_new = w - lr * g. Returns the new weight value.
@pure fn __sgd_step(w: f32, g: f32, lr: f32) -> f32 {
    w - lr * g
}

// SGD with momentum: returns (new_w, new_v) — but Helix doesn't have
// tuple returns yet. Instead, callers store v in a separate cell and
// pass it in. This helper just computes the new velocity.
@pure fn __momentum_step_v(v: f32, g: f32, beta: f32) -> f32 {
    beta * v + g
}

// Adam-like step (single-step, no bias correction for simplicity).
// Returns the parameter update direction; callers do w := w - lr * step.
// Restart 47 A2: clamp v to >= 0 before __sqrt to match the in-arena
// adam_f32_step (nn.hx) and the layer_norm_f32 negative-eps clamp
// precedent. A negative v from upstream would otherwise produce
// __sqrt(v) = 0 -> raw_denom = eps -> m / tiny eps -> spurious large step.
@pure fn __adam_step(m: f32, v: f32, eps: f32) -> f32 {
    let safe_v = if v < 0.0_f32 { 0.0_f32 } else { v };
    let raw_denom = __sqrt(safe_v) + eps;
    // Restart 50 A2: also fail-closed on NaN (raw_denom != raw_denom),
    // matching the in-arena adam_f32_step + softmax_layer idiom.
    if (raw_denom <= 0.0_f32) || (raw_denom != raw_denom) { 0.0_f32 }
    else { m / raw_denom }
}

// =========================================================================
// Integer min/max/clamp. The float versions __min/__max/__clamp already
// exist higher up; these are i32-typed for callers that stay in integer
// arithmetic (loop bounds, opcode dispatch, totality measure tracking).
// =========================================================================

@pure fn __min_i32(a: i32, b: i32) -> i32 {
    if a < b { a } else { b }
}

@pure fn __max_i32(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}

@pure fn __clamp_i32(x: i32, lo: i32, hi: i32) -> i32 {
    if x < lo { lo } else { if x > hi { hi } else { x } }
}

// Integer absolute value. Mirror of __abs (f32) on the i32 side.
// Restart 61 A3: saturate INT32_MIN to INT32_MAX. Pre-fix, x = INT32_MIN
// silently wrapped back to INT32_MIN because -INT32_MIN is not
// representable as i32; that broke the |x| >= 0 postcondition for the
// canonical abs helper. Same family as vec_negate_inplace /
// vec_map_neg (restart 51 A5), ti1d_max_abs / vec_max_abs (restart 56
// A2/A3), and vec_map_abs (restart 58 A2). Now defined for all i32
// inputs.
@pure fn __abs_i32(x: i32) -> i32 {
    if x == ((0 - 2147483647) - 1) { 2147483647 }
    else { if x < 0 { 0 - x } else { x } }
}

// Integer sign (-1 / 0 / +1). Mirror of __sign (f32) on the i32 side.
@pure fn __sign_i32(x: i32) -> i32 {
    if x > 0 { 1 } else { if x < 0 { 0 - 1 } else { 0 } }
}

// f64 sign (-1.0 / 0.0 / +1.0). Mirror of __sign (f32) at f64 precision.
@pure fn __sign_f64(x: f64) -> f64 {
    if x > 0.0_f64 { 1.0_f64 }
    else { if x < 0.0_f64 { 0.0_f64 - 1.0_f64 } else { 0.0_f64 } }
}

// f64 mean-squared error per-example. Mirror of __mse (f32) at f64
// precision; useful when callers stay in f64 to avoid the f32-precision
// loss of (pred - target)^2 for large or near-equal values.
@pure fn __mse_f64(pred: f64, target: f64) -> f64 {
    let d = pred - target;
    d * d
}
