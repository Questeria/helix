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
