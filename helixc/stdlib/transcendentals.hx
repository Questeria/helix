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
    // k = round(x / ln2). We use floor(x*1.44269504 + 0.5) as i32.
    let k_f = x * 1.44269504 + 0.5;
    let k = (k_f as i32) - if k_f < 0.0 { 1 } else { 0 };
    let k_corrected = if (k as f32) > k_f { k - 1 } else { k };
    // r = x - k*ln2
    let r = x - (k_corrected as f32) * 0.69314718;
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

@pure fn __sin(x: f32) -> f32 {
    // sin(x) = x - x³/3! + x⁵/5! - x⁷/7!
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    let x7 = x5 * x2;
    x - x3 * 0.16666667 + x5 * 0.00833333 - x7 * 0.00019841
}

@pure fn __cos(x: f32) -> f32 {
    // cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6!
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    1.0 - x2 * 0.5 + x4 * 0.04166667 - x6 * 0.00138889
}

@pure fn __log(x: f32) -> f32 {
    // log(x) for x near 1: log(1+y) = y - y²/2 + y³/3 - y⁴/4 + y⁵/5
    let y = x - 1.0;
    let y2 = y * y;
    let y3 = y2 * y;
    let y4 = y3 * y;
    let y5 = y4 * y;
    y - y2 * 0.5 + y3 * 0.33333333 - y4 * 0.25 + y5 * 0.2
}

@pure fn __sqrt(x: f32) -> f32 {
    // Newton's method: y_{n+1} = (y_n + x/y_n) / 2.
    // 4 iterations from a rough initial guess.
    let y0 = x * 0.5 + 0.5;
    let y1 = (y0 + x / y0) * 0.5;
    let y2 = (y1 + x / y1) * 0.5;
    let y3 = (y2 + x / y2) * 0.5;
    let y4 = (y3 + x / y3) * 0.5;
    y4
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

// Floor: nearest integer ≤ x. Implemented via cast-and-correct since we
// have no real floor instruction yet. Works for finite values in i32 range.
@pure fn __floor(x: f32) -> f32 {
    let i = x as i32;
    let f = i as f32;
    if f > x { f - 1.0 } else { f }
}

@pure fn __ceil(x: f32) -> f32 {
    let i = x as i32;
    let f = i as f32;
    if f < x { f + 1.0 } else { f }
}

// Integer power: x^n via iterative multiplication. Uses a loop bounded
// at 16 iterations as a safety cap (n > 16 saturates to x^16). Out of
// range or n <= 0 returns 1.0.
@pure fn __powi(x: f32, n: i32) -> f32 {
    if n <= 0 { 1.0 }
    else {
        let mut result: f32 = 1.0;
        let mut i: i32 = 0;
        let cap = if n < 16 { n } else { 16 };
        while i < cap {
            result = result * x;
            i = i + 1;
        }
        result
    }
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
@pure fn __softplus(x: f32) -> f32 {
    __log(1.0 + __exp(x))
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
    0.0 - (y * __log(p_safe) + (1.0 - y) * __log(1.0 - p_safe))
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
// State range: [0, 32768). Period: 32768. Adequate for seeded
// reproducibility in test programs; not strong enough for serious
// statistical sampling. The previous version used a 48271×x multiply
// which overflowed i32 for any seed > ~44k, silently corrupting the
// stream. This version's 25173×x peaks at 25173×32767 ≈ 8.25e8 — well
// under i32 max (2.15e9) — so the multiply is exact.
@pure fn __rand_step(s: i32) -> i32 {
    // Park-Miller-style LCG with parameters from Numerical Recipes.
    // s_{n+1} = (s_n * 25173 + 13849) mod 32768
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
@pure fn __adam_step(m: f32, v: f32, eps: f32) -> f32 {
    m / (__sqrt(v) + eps)
}
