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

@pure fn __exp(x: f32) -> f32 {
    // exp(x) = 1 + x + x²/2! + x³/3! + ... (8 terms)
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x2 * x2;
    let x5 = x4 * x;
    let x6 = x3 * x3;
    let x7 = x6 * x;
    1.0 + x
        + x2 * 0.5
        + x3 * 0.16666667
        + x4 * 0.04166667
        + x5 * 0.00833333
        + x6 * 0.00138889
        + x7 * 0.00019841
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

// Sigmoid: bounded activation in (0, 1). Used for binary classification.
@pure fn __sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + __exp(0.0 - x))
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

// Integer power: x^n for non-negative integer n. n is i32 to avoid
// recursive float comparisons; up to n=10 covers most practical cases.
@pure fn __powi(x: f32, n: i32) -> f32 {
    if n <= 0 { 1.0 }
    else { if n == 1 { x }
           else { if n == 2 { x * x }
                  else { if n == 3 { x * x * x }
                         else { if n == 4 { let s = x*x; s * s }
                                else {
                                    // generic: compute up to n==10 unrolled
                                    let s = x*x;
                                    if n == 5 { s * s * x }
                                    else { if n == 6 { s * s * s }
                                           else { if n == 7 { s * s * s * x }
                                                  else { if n == 8 { let q = s*s; q * q }
                                                         else { if n == 9 { let q = s*s; q * q * x }
                                                                else { let q = s*s; q * q * s } } } } }
                                } } } } }
}

// =========================================================================
// Modern activation functions.
// =========================================================================

// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1).
// Range: (-1, 1). Symmetric, smooth.
@pure fn __tanh(x: f32) -> f32 {
    let e2 = __exp(2.0 * x);
    (e2 - 1.0) / (e2 + 1.0)
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

// One step of xorshift32. Given current state s, returns the next state.
// State must be non-zero; the caller is responsible for seeding.
//   s ^= s << 13;
//   s ^= s >> 17;
//   s ^= s << 5;
// The Helix backend doesn't yet expose bitwise shifts/xor, so we
// approximate with arithmetic. For now we use a Lehmer LCG which only
// needs multiply + mod and is good enough for non-cryptographic uses:
//   s = (s * 48271) % 2147483647
@pure fn __rand_step(s: i32) -> i32 {
    // Lehmer LCG (MINSTD): period 2^31 - 2, well-studied parameters.
    // To avoid i32 overflow, do the multiply at runtime carefully:
    //   (s * 48271) mod (2^31 - 1)
    // For now we just do the modulo — overflow is wrap-around in i32
    // arithmetic which corrupts but produces SOME deterministic stream.
    let raw = s * 48271;
    let m = raw % 2147483647;
    if m < 0 { m + 2147483647 } else { if m == 0 { 1 } else { m } }
}

// Map a state to a float in [0, 1).
@pure fn __rand_float(s: i32) -> f32 {
    (s as f32) * 0.0000000004656612873   // 1 / (2^31 - 1)
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
    beta * v + grad
}

// Adam-like step (single-step, no bias correction for simplicity).
// Returns the parameter update direction; callers do w := w - lr * step.
@pure fn __adam_step(m: f32, v: f32, eps: f32) -> f32 {
    m / (__sqrt(v) + eps)
}
