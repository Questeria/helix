// helixc/stdlib/autodiff.hx — forward-mode autodiff via dual-number style.
//
// Phase 2.1: a small Helix-side autodiff library. Each "dual number" is
// represented by two separate f64s (value + derivative) passed by the
// caller. This avoids the current limitation that struct fields lower
// to i32 slots (so an `f64` field would silently truncate).
//
// Convention: every dual op takes pairs (a_v, a_dx, b_v, b_dx, ...) and
// returns a tuple (val, dx) — but since Helix doesn't have multi-return
// yet, we expose two functions per op: `<op>_v` returns the value,
// `<op>_dx` returns the derivative.
//
// To compute df/dx for f(x), seed dx=1.0_f64 for x and dx=0.0_f64 for
// constants:
//   x_v = 3.0_f64, x_dx = 1.0_f64
//   f(x) = x*x + 2*x + 1
//   v = mul_v(x_v, x_dx, x_v, x_dx);  dx = mul_dx(x_v, x_dx, x_v, x_dx)
//   ... etc.
//
// License: Apache 2.0

@pure fn d_add_v(a_v: f64, a_dx: f64, b_v: f64, b_dx: f64) -> f64 { a_v + b_v }
@pure fn d_add_dx(a_v: f64, a_dx: f64, b_v: f64, b_dx: f64) -> f64 { a_dx + b_dx }

@pure fn d_sub_v(a_v: f64, a_dx: f64, b_v: f64, b_dx: f64) -> f64 { a_v - b_v }
@pure fn d_sub_dx(a_v: f64, a_dx: f64, b_v: f64, b_dx: f64) -> f64 { a_dx - b_dx }

// d/dx (a*b) = a'b + a b'  (product rule)
@pure fn d_mul_v(a_v: f64, a_dx: f64, b_v: f64, b_dx: f64) -> f64 { a_v * b_v }
@pure fn d_mul_dx(a_v: f64, a_dx: f64, b_v: f64, b_dx: f64) -> f64 {
    a_dx * b_v + a_v * b_dx
}

// d/dx (a/b) = (a' b - a b') / b^2
@pure fn d_div_v(a_v: f64, a_dx: f64, b_v: f64, b_dx: f64) -> f64 { a_v / b_v }
@pure fn d_div_dx(a_v: f64, a_dx: f64, b_v: f64, b_dx: f64) -> f64 {
    (a_dx * b_v - a_v * b_dx) / (b_v * b_v)
}

// d/dx (-a) = -a'
@pure fn d_neg_v(a_v: f64, a_dx: f64) -> f64 { 0.0_f64 - a_v }
@pure fn d_neg_dx(a_v: f64, a_dx: f64) -> f64 { 0.0_f64 - a_dx }

// d/dx exp(a) = exp(a) * a'
@pure fn d_exp_v(a_v: f64, a_dx: f64) -> f64 { __exp_f64(a_v) }
@pure fn d_exp_dx(a_v: f64, a_dx: f64) -> f64 { __exp_f64(a_v) * a_dx }

// d/dx sigmoid(a) = sigmoid(a) * (1 - sigmoid(a)) * a'
@pure fn d_sigmoid_v(a_v: f64, a_dx: f64) -> f64 { __sigmoid_f64(a_v) }
@pure fn d_sigmoid_dx(a_v: f64, a_dx: f64) -> f64 {
    let s = __sigmoid_f64(a_v);
    s * (1.0_f64 - s) * a_dx
}

// d/dx sqrt(a) = a' / (2 * sqrt(a))
@pure fn d_sqrt_v(a_v: f64, a_dx: f64) -> f64 { __sqrt_f64(a_v) }
@pure fn d_sqrt_dx(a_v: f64, a_dx: f64) -> f64 {
    a_dx / (2.0_f64 * __sqrt_f64(a_v))
}

// d/dx (a*a) — convenient shorthand. d_sq_dx(x, 1.0) = 2x.
@pure fn d_sq_v(a_v: f64, a_dx: f64) -> f64 { a_v * a_v }
@pure fn d_sq_dx(a_v: f64, a_dx: f64) -> f64 { 2.0_f64 * a_v * a_dx }

// d/dx (a + scalar) = a'
@pure fn d_add_const_v(a_v: f64, a_dx: f64, c: f64) -> f64 { a_v + c }
@pure fn d_add_const_dx(a_v: f64, a_dx: f64, c: f64) -> f64 { a_dx }

// d/dx (a * scalar) = scalar * a'
@pure fn d_scale_v(a_v: f64, a_dx: f64, c: f64) -> f64 { a_v * c }
@pure fn d_scale_dx(a_v: f64, a_dx: f64, c: f64) -> f64 { a_dx * c }

// d/dx ln(a) = a' / a  (defined for a > 0)
@pure fn d_log_v(a_v: f64, a_dx: f64) -> f64 { __log_f64(a_v) }
@pure fn d_log_dx(a_v: f64, a_dx: f64) -> f64 { a_dx / a_v }

// d/dx (1/a) = -a' / a^2
@pure fn d_recip_v(a_v: f64, a_dx: f64) -> f64 { 1.0_f64 / a_v }
@pure fn d_recip_dx(a_v: f64, a_dx: f64) -> f64 {
    (0.0_f64 - a_dx) / (a_v * a_v)
}

// d/dx sin(a) = cos(a) * a'
@pure fn d_sin_v(a_v: f64, a_dx: f64) -> f64 { __sin_f64(a_v) }
@pure fn d_sin_dx(a_v: f64, a_dx: f64) -> f64 { __cos_f64(a_v) * a_dx }

// d/dx cos(a) = -sin(a) * a'
@pure fn d_cos_v(a_v: f64, a_dx: f64) -> f64 { __cos_f64(a_v) }
@pure fn d_cos_dx(a_v: f64, a_dx: f64) -> f64 {
    (0.0_f64 - __sin_f64(a_v)) * a_dx
}

// d/dx relu(a) = 1 if a > 0, else 0  (subgradient at 0 = 0)
@pure fn d_relu_v(a_v: f64, a_dx: f64) -> f64 { __relu_f64(a_v) }
@pure fn d_relu_dx(a_v: f64, a_dx: f64) -> f64 {
    if a_v > 0.0_f64 { a_dx } else { 0.0_f64 }
}
