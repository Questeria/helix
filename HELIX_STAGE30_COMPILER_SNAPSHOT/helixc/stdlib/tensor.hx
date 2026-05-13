// helixc/stdlib/tensor.hx — arena-backed tensor primitives.
//
// Phase 2.2: 1D and 2D tensor primitives over the global arena. Each
// "tensor" is represented by a start index + shape (rows, cols). Like
// Vec but with row-major 2D access. f32-element by default since most
// AGI/NN math is f32 in production.
//
// Convention: build one tensor at a time; the global arena means
// interleaved tensors mix.
//
// API (1D, f32):
//   t1d_new(n)                  -> i32   reserve n slots, return start
//   t1d_set(start, i, x)        -> i32   write element i (writes f32 bits)
//   t1d_get(start, i)           -> f32   read element i
//   t1d_sum(start, n)           -> f32   sum of n elements
//   t1d_dot(a_start, b_start, n)-> f32   inner product
//   t1d_axpy(y_start, a, x_start, n) -> i32  y[i] += a*x[i]; returns 0
//
// API (2D, f32, row-major):
//   t2d_new(rows, cols)         -> i32   reserve rows*cols slots
//   t2d_set(start, cols, i, j, x) -> i32  M[i,j] = x
//   t2d_get(start, cols, i, j)  -> f32   M[i,j]
//   t2d_matvec(W_start, W_rows, W_cols, x_start, y_start) -> i32
//                                        y = W @ x; returns 0
//
// LIMITATION: f32 stored as i32-bit-pattern in the arena. Read/write
// uses cvtsi2ss-style casts via __arena_set/get. Direct float store
// to arena slot needs bit-reinterpret which Helix lacks; we work around
// by storing as i32 and reading as f32 via type-coerced loads (works
// because helixc treats both as 4-byte words on the stack/arena).
//
// License: Apache 2.0

@pure fn t1d_new(n: i32) -> i32 {
    let start = __arena_len();
    let mut i: i32 = 0;
    while i < n {
        __arena_push(0);
        i = i + 1;
    }
    start
}

fn t1d_set_i32_bits(start: i32, i: i32, bits: i32) -> i32 {
    __arena_set(start + i, bits);
    0
}

fn t1d_get_i32_bits(start: i32, i: i32) -> i32 {
    __arena_get(start + i)
}

// Integer-tensor variants (no float-bit-cast needed). These are the
// safe variants until float<->arena bit-reinterpret lands as a
// codegen primitive.
@pure fn ti1d_get(start: i32, i: i32) -> i32 {
    __arena_get(start + i)
}

fn ti1d_set(start: i32, i: i32, x: i32) -> i32 {
    __arena_set(start + i, x);
    x
}

@pure fn ti1d_sum(start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < n {
        total = total + __arena_get(start + i);
        i = i + 1;
    }
    total
}

@pure fn ti1d_dot(a_start: i32, b_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < n {
        total = total + __arena_get(a_start + i) * __arena_get(b_start + i);
        i = i + 1;
    }
    total
}

fn ti1d_axpy(y_start: i32, a: i32, x_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let cur = __arena_get(y_start + i);
        let xi = __arena_get(x_start + i);
        __arena_set(y_start + i, cur + a * xi);
        i = i + 1;
    }
    0
}

// 2D row-major access: M[i,j] lives at slot start + i*cols + j.
@pure fn ti2d_new(rows: i32, cols: i32) -> i32 {
    let n = rows * cols;
    t1d_new(n)
}

fn ti2d_set(start: i32, cols: i32, i: i32, j: i32, x: i32) -> i32 {
    __arena_set(start + i * cols + j, x);
    x
}

@pure fn ti2d_get(start: i32, cols: i32, i: i32, j: i32) -> i32 {
    __arena_get(start + i * cols + j)
}

// y = W @ x. W is rows*cols, x is cols, y is rows.
fn ti2d_matvec(w_start: i32, w_rows: i32, w_cols: i32,
               x_start: i32, y_start: i32) -> i32 {
    let mut r: i32 = 0;
    while r < w_rows {
        let mut c: i32 = 0;
        let mut acc: i32 = 0;
        while c < w_cols {
            acc = acc + __arena_get(w_start + r * w_cols + c) * __arena_get(x_start + c);
            c = c + 1;
        }
        __arena_set(y_start + r, acc);
        r = r + 1;
    }
    0
}

// Element-wise: y[i] = relu(x[i]) for i in [0, n). Integer relu.
fn ti1d_relu(x_start: i32, y_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let xi = __arena_get(x_start + i);
        let v = if xi > 0 { xi } else { 0 };
        __arena_set(y_start + i, v);
        i = i + 1;
    }
    0
}

// Element-wise add: z[i] = x[i] + y[i]. Returns 0.
fn ti1d_add(x_start: i32, y_start: i32, z_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        __arena_set(z_start + i,
                    __arena_get(x_start + i) + __arena_get(y_start + i));
        i = i + 1;
    }
    0
}

// Integer element-wise subtraction: z[i] = x[i] - y[i].
// Companion to ti1d_add. z must be pre-allocated.
fn ti1d_sub(x_start: i32, y_start: i32, z_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        __arena_set(z_start + i,
                    __arena_get(x_start + i) - __arena_get(y_start + i));
        i = i + 1;
    }
    0
}

// Integer element-wise multiplication (Hadamard): z[i] = x[i] * y[i].
// For inner product use ti1d_dot.
fn ti1d_mul(x_start: i32, y_start: i32, z_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        __arena_set(z_start + i,
                    __arena_get(x_start + i) * __arena_get(y_start + i));
        i = i + 1;
    }
    0
}

// =========================================================================
// Phase 2.2 step 2: f32 tensor primitives via __bits_of_f32 reinterpret.
// =========================================================================
// f32 values stored as their IEEE 754 bit pattern in arena slots (4 bytes
// each, same width as i32). The codegen primitive __bits_of_f32 /
// __f32_from_bits relabels the same 4 bytes — no instruction emitted, just
// a type-system shim.

fn tf1d_set(start: i32, i: i32, x: f32) -> i32 {
    __arena_set(start + i, __bits_of_f32(x));
    0
}

@pure fn tf1d_get(start: i32, i: i32) -> f32 {
    __f32_from_bits(__arena_get(start + i))
}

@pure
fn tf1d_sum(start: i32, n: i32) -> f32 {
    let mut i: i32 = 0;
    let mut total: f32 = 0.0_f32;
    while i < n {
        total = total + __f32_from_bits(__arena_get(start + i));
        i = i + 1;
    }
    total
}

@pure
fn tf1d_dot(a_start: i32, b_start: i32, n: i32) -> f32 {
    let mut i: i32 = 0;
    let mut total: f32 = 0.0_f32;
    while i < n {
        let av = __f32_from_bits(__arena_get(a_start + i));
        let bv = __f32_from_bits(__arena_get(b_start + i));
        total = total + av * bv;
        i = i + 1;
    }
    total
}

fn tf1d_axpy(y_start: i32, a: f32, x_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let cur = __f32_from_bits(__arena_get(y_start + i));
        let xi = __f32_from_bits(__arena_get(x_start + i));
        __arena_set(y_start + i, __bits_of_f32(cur + a * xi));
        i = i + 1;
    }
    0
}

fn tf1d_relu(x_start: i32, y_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let xi = __f32_from_bits(__arena_get(x_start + i));
        let v = if xi > 0.0_f32 { xi } else { 0.0_f32 };
        __arena_set(y_start + i, __bits_of_f32(v));
        i = i + 1;
    }
    0
}

@pure
fn tf2d_get(start: i32, cols: i32, i: i32, j: i32) -> f32 {
    __f32_from_bits(__arena_get(start + i * cols + j))
}

fn tf2d_set(start: i32, cols: i32, i: i32, j: i32, x: f32) -> i32 {
    __arena_set(start + i * cols + j, __bits_of_f32(x));
    0
}

fn tf2d_matvec(w_start: i32, w_rows: i32, w_cols: i32,
               x_start: i32, y_start: i32) -> i32 {
    let mut r: i32 = 0;
    while r < w_rows {
        let mut c: i32 = 0;
        let mut acc: f32 = 0.0_f32;
        while c < w_cols {
            let wv = __f32_from_bits(__arena_get(w_start + r * w_cols + c));
            let xv = __f32_from_bits(__arena_get(x_start + c));
            acc = acc + wv * xv;
            c = c + 1;
        }
        __arena_set(y_start + r, __bits_of_f32(acc));
        r = r + 1;
    }
    0
}

// =========================================================================
// Phase 2 perfection: reductions, more ops.
// =========================================================================

// Integer-tensor mean (returns floor(sum/n)).
@pure fn ti1d_mean(start: i32, n: i32) -> i32 {
    if n == 0 { 0 }
    else { ti1d_sum(start, n) / n }
}

// Integer-tensor product.
@pure fn ti1d_prod(start: i32, n: i32) -> i32 {
    if n == 0 { 1 }
    else {
        let mut i: i32 = 0;
        let mut p: i32 = 1;
        while i < n { p = p * __arena_get(start + i); i = i + 1; }
        p
    }
}

// Integer-tensor min.
@pure fn ti1d_min(start: i32, n: i32) -> i32 {
    if n == 0 { 0 }
    else {
        let mut best = __arena_get(start);
        let mut i: i32 = 1;
        while i < n {
            let v = __arena_get(start + i);
            if v < best { best = v; }
            i = i + 1;
        }
        best
    }
}

// Integer-tensor max (companion to ti1d_min).
@pure fn ti1d_max(start: i32, n: i32) -> i32 {
    if n == 0 { 0 }
    else {
        let mut best = __arena_get(start);
        let mut i: i32 = 1;
        while i < n {
            let v = __arena_get(start + i);
            if v > best { best = v; }
            i = i + 1;
        }
        best
    }
}

// Integer-tensor argmin (returns index of smallest element).
// Companion to ti1d_argmax. n == 0 returns -1.
@pure fn ti1d_argmin(start: i32, n: i32) -> i32 {
    if n == 0 { 0 - 1 }
    else {
        let mut best = __arena_get(start);
        let mut best_idx: i32 = 0;
        let mut i: i32 = 1;
        while i < n {
            let v = __arena_get(start + i);
            if v < best { best = v; best_idx = i; };
            i = i + 1;
        }
        best_idx
    }
}

// Integer-tensor argmax (returns index of largest element).
@pure fn ti1d_argmax(start: i32, n: i32) -> i32 {
    if n == 0 { 0 - 1 }
    else {
        let mut best_idx: i32 = 0;
        let mut best_val = __arena_get(start);
        let mut i: i32 = 1;
        while i < n {
            let v = __arena_get(start + i);
            if v > best_val { best_val = v; best_idx = i; }
            i = i + 1;
        }
        best_idx
    }
}

// Tensor zeros: allocate n slots already initialized to 0.
@pure fn ti1d_zeros(n: i32) -> i32 { t1d_new(n) }

// Tensor ones: allocate n slots and fill with 1.
fn ti1d_ones(n: i32) -> i32 {
    let s = t1d_new(n);
    let mut i: i32 = 0;
    while i < n { __arena_set(s + i, 1); i = i + 1; }
    s
}

// 2D row-major matmul: C = A @ B.
//   A is (a_rows x a_cols), B is (a_cols x b_cols), C is (a_rows x b_cols).
fn ti2d_matmul(a_start: i32, a_rows: i32, a_cols: i32,
               b_start: i32, b_cols: i32, c_start: i32) -> i32 {
    let mut r: i32 = 0;
    while r < a_rows {
        let mut c: i32 = 0;
        while c < b_cols {
            let mut k: i32 = 0;
            let mut acc: i32 = 0;
            while k < a_cols {
                let av = __arena_get(a_start + r * a_cols + k);
                let bv = __arena_get(b_start + k * b_cols + c);
                acc = acc + av * bv;
                k = k + 1;
            }
            __arena_set(c_start + r * b_cols + c, acc);
            c = c + 1;
        }
        r = r + 1;
    }
    0
}

// Reshape: copy n elements from src to dst. (For row-major tensors a
// reshape is a no-op as long as total element count matches; for safety
// we expose copy as the explicit form.)
fn ti1d_copy(src: i32, dst: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        __arena_set(dst + i, __arena_get(src + i));
        i = i + 1;
    }
    0
}

// Broadcasting: y[i] = x[i] + scalar (element-wise add of scalar to vec).
fn ti1d_add_scalar(x_start: i32, scalar: i32, y_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        __arena_set(y_start + i, __arena_get(x_start + i) + scalar);
        i = i + 1;
    }
    0
}

fn ti1d_mul_scalar(x_start: i32, scalar: i32, y_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        __arena_set(y_start + i, __arena_get(x_start + i) * scalar);
        i = i + 1;
    }
    0
}

// f32 reductions
@pure fn tf1d_mean(start: i32, n: i32) -> f32 {
    if n == 0 { 0.0_f32 }
    else { tf1d_sum(start, n) / (n as f32) }
}

@pure fn tf1d_max(start: i32, n: i32) -> f32 {
    if n == 0 { 0.0_f32 }
    else {
        let mut best = __f32_from_bits(__arena_get(start));
        let mut i: i32 = 1;
        while i < n {
            let v = __f32_from_bits(__arena_get(start + i));
            if v > best { best = v; }
            i = i + 1;
        }
        best
    }
}

@pure fn tf1d_min(start: i32, n: i32) -> f32 {
    if n == 0 { 0.0_f32 }
    else {
        let mut best = __f32_from_bits(__arena_get(start));
        let mut i: i32 = 1;
        while i < n {
            let v = __f32_from_bits(__arena_get(start + i));
            if v < best { best = v; }
            i = i + 1;
        }
        best
    }
}

@pure fn tf1d_argmax(start: i32, n: i32) -> i32 {
    if n == 0 { 0 - 1 }
    else {
        let mut best_idx: i32 = 0;
        let mut best_val = __f32_from_bits(__arena_get(start));
        let mut i: i32 = 1;
        while i < n {
            let v = __f32_from_bits(__arena_get(start + i));
            if v > best_val { best_val = v; best_idx = i; }
            i = i + 1;
        }
        best_idx
    }
}

// f32 element-wise: z[i] = x[i] + y[i].
fn tf1d_add(x_start: i32, y_start: i32, z_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let xv = __f32_from_bits(__arena_get(x_start + i));
        let yv = __f32_from_bits(__arena_get(y_start + i));
        __arena_set(z_start + i, __bits_of_f32(xv + yv));
        i = i + 1;
    }
    0
}

fn tf1d_sub(x_start: i32, y_start: i32, z_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let xv = __f32_from_bits(__arena_get(x_start + i));
        let yv = __f32_from_bits(__arena_get(y_start + i));
        __arena_set(z_start + i, __bits_of_f32(xv - yv));
        i = i + 1;
    }
    0
}

fn tf1d_mul(x_start: i32, y_start: i32, z_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let xv = __f32_from_bits(__arena_get(x_start + i));
        let yv = __f32_from_bits(__arena_get(y_start + i));
        __arena_set(z_start + i, __bits_of_f32(xv * yv));
        i = i + 1;
    }
    0
}

// f32 broadcasting: y[i] = x[i] + scalar.
fn tf1d_add_scalar(x_start: i32, scalar: f32, y_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let xv = __f32_from_bits(__arena_get(x_start + i));
        __arena_set(y_start + i, __bits_of_f32(xv + scalar));
        i = i + 1;
    }
    0
}

fn tf1d_mul_scalar(x_start: i32, scalar: f32, y_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let xv = __f32_from_bits(__arena_get(x_start + i));
        __arena_set(y_start + i, __bits_of_f32(xv * scalar));
        i = i + 1;
    }
    0
}

// 2D row-major f32 matmul: C = A @ B.
//   A is (a_rows x a_cols), B is (a_cols x b_cols), C is (a_rows x b_cols).
fn tf2d_matmul(a_start: i32, a_rows: i32, a_cols: i32,
               b_start: i32, b_cols: i32, c_start: i32) -> i32 {
    let mut r: i32 = 0;
    while r < a_rows {
        let mut c: i32 = 0;
        while c < b_cols {
            let mut k: i32 = 0;
            let mut acc: f32 = 0.0_f32;
            while k < a_cols {
                let av = __f32_from_bits(__arena_get(a_start + r * a_cols + k));
                let bv = __f32_from_bits(__arena_get(b_start + k * b_cols + c));
                acc = acc + av * bv;
                k = k + 1;
            }
            __arena_set(c_start + r * b_cols + c, __bits_of_f32(acc));
            c = c + 1;
        }
        r = r + 1;
    }
    0
}


// f32-tensor zeros: allocate n slots; arena push 0 leaves the bit
// pattern as IEEE +0.0 which is exactly what we want.
@pure fn tf1d_zeros(n: i32) -> i32 { t1d_new(n) }

// f32-tensor ones: allocate n slots and fill with bits-of(1.0_f32).
fn tf1d_ones(n: i32) -> i32 {
    let s = t1d_new(n);
    let one_bits = __bits_of_f32(1.0_f32);
    let mut i: i32 = 0;
    while i < n {
        __arena_set(s + i, one_bits);
        i = i + 1;
    }
    s
}

// ti2d_transpose(src, rows, cols, dst): transpose an integer-tensor.
// `src` is rows*cols (row-major); `dst` becomes cols*rows (also row-
// major, where the new rows are the old columns). Caller pre-allocates
// dst with t1d_new(rows*cols). Out-of-place; src untouched.
fn ti2d_transpose(src: i32, rows: i32, cols: i32, dst: i32) -> i32 {
    let mut r: i32 = 0;
    while r < rows {
        let mut c: i32 = 0;
        while c < cols {
            __arena_set(dst + c * rows + r, __arena_get(src + r * cols + c));
            c = c + 1;
        }
        r = r + 1;
    }
    0
}

// ti1d_clamp(x, lo, hi, dst, n): elementwise clamp each x[i] into
// [lo, hi]. Out-of-place; result written to dst (caller pre-allocates
// dst with t1d_new(n) or shares with x for in-place).
fn ti1d_clamp(x_start: i32, lo: i32, hi: i32, dst: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let v = __arena_get(x_start + i);
        let cv = if v < lo { lo } else { if v > hi { hi } else { v } };
        __arena_set(dst + i, cv);
        i = i + 1;
    }
    0
}

// ti1d_l1_norm(x, n): L1 norm = sum of |x[i]|. @pure (read-only).
@pure fn ti1d_l1_norm(start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < n {
        let v = __arena_get(start + i);
        let av = if v < 0 { 0 - v } else { v };
        total = total + av;
        i = i + 1;
    }
    total
}

// ti1d_l2_norm_sq(x, n): squared L2 norm = sum(x[i]^2). Mirrors
// ti1d_dot(x, x, n). Distinct fn for clarity at call sites and
// future overflow-safe variants. @pure.
@pure fn ti1d_l2_norm_sq(start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < n {
        let v = __arena_get(start + i);
        total = total + v * v;
        i = i + 1;
    }
    total
}

// ti1d_eq_count(a, b, n): count of indices i where a[i] == b[i].
// Returns an integer in [0, n]. Useful for accuracy/agreement metrics.
@pure fn ti1d_eq_count(a_start: i32, b_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < n {
        if __arena_get(a_start + i) == __arena_get(b_start + i) {
            total = total + 1;
        };
        i = i + 1;
    }
    total
}

// tf1d_l2_norm_sq(x, n): squared L2 norm = sum(x[i]^2). f32 mirror of
// ti1d_l2_norm_sq. Builds on tf1d_dot(x, x, n) — distinct fn for clarity.
fn tf1d_l2_norm_sq(start: i32, n: i32) -> f32 {
    tf1d_dot(start, start, n)
}

// tf1d_l1_norm(x, n): L1 norm = sum of |x[i]|. f32 mirror of ti1d_l1_norm.
// Reads each f32, computes absolute value via SSE fabs (sign-bit clear),
// accumulates. Loop is the bf32 form so float promotion is implicit.
fn tf1d_l1_norm(start: i32, n: i32) -> f32 {
    let mut i: i32 = 0;
    let mut total: f32 = 0.0_f32;
    while i < n {
        let v = __f32_from_bits(__arena_get(start + i));
        let av = if v < 0.0_f32 { 0.0_f32 - v } else { v };
        total = total + av;
        i = i + 1;
    }
    total
}

// tf2d_transpose(src, rows, cols, dst): out-of-place transpose for f32
// tensors. f32 mirror of ti2d_transpose. Caller pre-allocates dst with
// t1d_new(rows*cols).
fn tf2d_transpose(src: i32, rows: i32, cols: i32, dst: i32) -> i32 {
    let mut r: i32 = 0;
    while r < rows {
        let mut c: i32 = 0;
        while c < cols {
            __arena_set(dst + c * rows + r, __arena_get(src + r * cols + c));
            c = c + 1;
        }
        r = r + 1;
    }
    0
}

// tf1d_clamp(x, lo, hi, dst, n): elementwise clamp each x[i] into [lo, hi].
// f32 mirror of ti1d_clamp.
fn tf1d_clamp(x_start: i32, lo: f32, hi: f32, dst: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let v = __f32_from_bits(__arena_get(x_start + i));
        let cv = if v < lo { lo } else { if v > hi { hi } else { v } };
        __arena_set(dst + i, __bits_of_f32(cv));
        i = i + 1;
    }
    0
}

// tf1d_argmin(start, n): @pure. Index of the smallest f32 element.
// Returns -1 if n == 0.
@pure fn tf1d_argmin(start: i32, n: i32) -> i32 {
    if n == 0 { 0 - 1 }
    else {
        let mut i: i32 = 1;
        let mut best_idx: i32 = 0;
        let mut best: f32 = __f32_from_bits(__arena_get(start));
        while i < n {
            let v = __f32_from_bits(__arena_get(start + i));
            if v < best { best = v; best_idx = i; }
            i = i + 1;
        }
        best_idx
    }
}

// tf1d_running_sum(start, n): allocate a new vec where r[i] = sum
// over x[0..=i]. Like vec_cumsum but for f32. r[0] = x[0]. Useful
// for prefix-sum queries.
fn tf1d_running_sum(start: i32, n: i32) -> i32 {
    let s: i32 = __arena_len();
    if n == 0 { s }
    else {
        let mut acc: f32 = 0.0_f32;
        let mut i: i32 = 0;
        while i < n {
            acc = acc + __f32_from_bits(__arena_get(start + i));
            __arena_push(__bits_of_f32(acc));
            i = i + 1;
        }
        s
    }
}

// tf1d_negate(start, dst, n): write -x[i] to dst[i] for all i. Out-of-
// place; caller pre-allocates dst with t1d_new(n) (or shares with x for
// in-place).
fn tf1d_negate(x_start: i32, dst: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let v = __f32_from_bits(__arena_get(x_start + i));
        __arena_set(dst + i, __bits_of_f32(0.0_f32 - v));
        i = i + 1;
    }
    0
}

// tf1d_scale_inplace(start, n, scalar): multiply every element by
// scalar in place. Mirror of vec_offset_inplace for f32.
fn tf1d_scale_inplace(start: i32, n: i32, scalar: f32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let v = __f32_from_bits(__arena_get(start + i));
        __arena_set(start + i, __bits_of_f32(v * scalar));
        i = i + 1;
    }
    start
}

// tf2d_add(a, b, c, rows, cols): elementwise 2D add c = a + b. All
// three matrices share row-major layout with `cols` columns.
fn tf2d_add(a: i32, b: i32, c: i32, rows: i32, cols: i32) -> i32 {
    let n = rows * cols;
    let mut i: i32 = 0;
    while i < n {
        let av = __f32_from_bits(__arena_get(a + i));
        let bv = __f32_from_bits(__arena_get(b + i));
        __arena_set(c + i, __bits_of_f32(av + bv));
        i = i + 1;
    }
    0
}

// tf2d_scale_inplace(start, rows, cols, scalar): multiply every element
// of the 2D matrix in place by scalar.
fn tf2d_scale_inplace(start: i32, rows: i32, cols: i32, scalar: f32) -> i32 {
    let n = rows * cols;
    let mut i: i32 = 0;
    while i < n {
        let v = __f32_from_bits(__arena_get(start + i));
        __arena_set(start + i, __bits_of_f32(v * scalar));
        i = i + 1;
    }
    0
}

// tf1d_max_abs(start, n): @pure. Max of |x[i]| (Linf norm). Returns
// 0.0 for empty input.
@pure
fn tf1d_max_abs(start: i32, n: i32) -> f32 {
    let mut i: i32 = 0;
    let mut best: f32 = 0.0_f32;
    while i < n {
        let v = __f32_from_bits(__arena_get(start + i));
        let av = if v < 0.0_f32 { 0.0_f32 - v } else { v };
        if av > best { best = av; };
        i = i + 1;
    }
    best
}

// tf1d_axpby(x, y, a, b, n): in-place compute y[i] = a*x[i] + b*y[i].
// BLAS-style level-1 op. Caller mutates y in place.
fn tf1d_axpby(x_start: i32, y_start: i32, a: f32, b: f32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let xv = __f32_from_bits(__arena_get(x_start + i));
        let yv = __f32_from_bits(__arena_get(y_start + i));
        __arena_set(y_start + i, __bits_of_f32(a * xv + b * yv));
        i = i + 1;
    }
    0
}

// tf2d_sub(a, b, c, rows, cols): elementwise 2D subtract c = a - b.
fn tf2d_sub(a: i32, b: i32, c: i32, rows: i32, cols: i32) -> i32 {
    let n = rows * cols;
    let mut i: i32 = 0;
    while i < n {
        let av = __f32_from_bits(__arena_get(a + i));
        let bv = __f32_from_bits(__arena_get(b + i));
        __arena_set(c + i, __bits_of_f32(av - bv));
        i = i + 1;
    }
    0
}

// tf2d_mul(a, b, c, rows, cols): elementwise 2D Hadamard (NOT matmul).
fn tf2d_mul(a: i32, b: i32, c: i32, rows: i32, cols: i32) -> i32 {
    let n = rows * cols;
    let mut i: i32 = 0;
    while i < n {
        let av = __f32_from_bits(__arena_get(a + i));
        let bv = __f32_from_bits(__arena_get(b + i));
        __arena_set(c + i, __bits_of_f32(av * bv));
        i = i + 1;
    }
    0
}

// tf1d_argmax_in_range(start, lo, hi): @pure. Index of largest f32
// in x[lo..hi). Returns -1 if hi <= lo.
@pure
fn tf1d_argmax_in_range(start: i32, lo: i32, hi: i32) -> i32 {
    if hi <= lo { 0 - 1 }
    else {
        let mut i: i32 = lo + 1;
        let mut best_idx: i32 = lo;
        let mut best: f32 = __f32_from_bits(__arena_get(start + lo));
        while i < hi {
            let v = __f32_from_bits(__arena_get(start + i));
            if v > best { best = v; best_idx = i; }
            i = i + 1;
        }
        best_idx
    }
}

// tf1d_sum_in_range(start, lo, hi): @pure. Sum of x[lo..hi). 0.0 if
// hi <= lo. Useful for partial accumulators.
@pure
fn tf1d_sum_in_range(start: i32, lo: i32, hi: i32) -> f32 {
    let mut i: i32 = lo;
    let mut total: f32 = 0.0_f32;
    while i < hi {
        total = total + __f32_from_bits(__arena_get(start + i));
        i = i + 1;
    }
    total
}

// tf2d_row_sum(start, rows, cols, dst): for each row r, write
// sum(M[r, *]) to dst[r]. dst pre-allocated by caller (size = rows).
fn tf2d_row_sum(start: i32, rows: i32, cols: i32, dst: i32) -> i32 {
    let mut r: i32 = 0;
    while r < rows {
        let mut c: i32 = 0;
        let mut acc: f32 = 0.0_f32;
        while c < cols {
            acc = acc + __f32_from_bits(__arena_get(start + r * cols + c));
            c = c + 1;
        }
        __arena_set(dst + r, __bits_of_f32(acc));
        r = r + 1;
    }
    0
}

// tf2d_col_sum(start, rows, cols, dst): for each col c, write
// sum(M[*, c]) to dst[c]. dst pre-allocated by caller (size = cols).
fn tf2d_col_sum(start: i32, rows: i32, cols: i32, dst: i32) -> i32 {
    let mut c: i32 = 0;
    while c < cols {
        let mut r: i32 = 0;
        let mut acc: f32 = 0.0_f32;
        while r < rows {
            acc = acc + __f32_from_bits(__arena_get(start + r * cols + c));
            r = r + 1;
        }
        __arena_set(dst + c, __bits_of_f32(acc));
        c = c + 1;
    }
    0
}

// tf1d_arange(start_val, n): allocate a new f32 vec of length n with
// values [start_val, start_val + 1.0, ..., start_val + (n-1)]. Returns
// the new vec start. Useful for index-vec pairs and test setup.
fn tf1d_arange(start_val: f32, n: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    let mut v: f32 = start_val;
    while i < n {
        __arena_push(__bits_of_f32(v));
        v = v + 1.0_f32;
        i = i + 1;
    }
    s
}

// tf1d_dot_with_offset(a, a_off, b, b_off, n): dot product over slices
// a[a_off..a_off+n] and b[b_off..b_off+n]. @pure.
@pure
fn tf1d_dot_with_offset(a: i32, a_off: i32, b: i32, b_off: i32, n: i32) -> f32 {
    let mut i: i32 = 0;
    let mut total: f32 = 0.0_f32;
    while i < n {
        let av = __f32_from_bits(__arena_get(a + a_off + i));
        let bv = __f32_from_bits(__arena_get(b + b_off + i));
        total = total + av * bv;
        i = i + 1;
    }
    total
}

// tf2d_diag(m, rows_eq_cols, dst): for a square matrix M of side N,
// extract the diagonal into dst (size N). Requires square (rows == cols).
fn tf2d_diag(m: i32, n: i32, dst: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        __arena_set(dst + i, __arena_get(m + i * n + i));
        i = i + 1;
    }
    0
}

// tf2d_eye(n): allocate a new n*n identity matrix (1.0 on diagonal,
// 0.0 elsewhere). Returns the new start index.
fn tf2d_eye(n: i32) -> i32 {
    let s: i32 = __arena_len();
    let one_bits = __bits_of_f32(1.0_f32);
    let mut r: i32 = 0;
    while r < n {
        let mut c: i32 = 0;
        while c < n {
            if r == c { __arena_push(one_bits); }
            else { __arena_push(0); };
            c = c + 1;
        }
        r = r + 1;
    }
    s
}

// tf2d_trace(m, n): @pure. Sum of diagonal elements of an n*n matrix.
@pure
fn tf2d_trace(m: i32, n: i32) -> f32 {
    let mut i: i32 = 0;
    let mut total: f32 = 0.0_f32;
    while i < n {
        total = total + __f32_from_bits(__arena_get(m + i * n + i));
        i = i + 1;
    }
    total
}

// tf1d_lerp(a, b, t, dst, n): linear interpolation. dst[i] = a[i] +
// t * (b[i] - a[i]). Useful for numerical interpolation between two
// vectors. dst pre-allocated (n slots).
fn tf1d_lerp(a: i32, b: i32, t: f32, dst: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let av = __f32_from_bits(__arena_get(a + i));
        let bv = __f32_from_bits(__arena_get(b + i));
        let v = av + t * (bv - av);
        __arena_set(dst + i, __bits_of_f32(v));
        i = i + 1;
    }
    0
}

// tf2d_norm_frobenius_sq(start, rows, cols): @pure. Squared Frobenius
// norm — sum of squares of all elements. Mirror of tf1d_l2_norm_sq for
// 2D matrices.
@pure
fn tf2d_norm_frobenius_sq(start: i32, rows: i32, cols: i32) -> f32 {
    let n = rows * cols;
    tf1d_l2_norm_sq(start, n)
}

// tf2d_zeros(rows, cols): allocate a new rows*cols matrix filled with 0.0_f32.
@pure
fn tf2d_zeros(rows: i32, cols: i32) -> i32 {
    t1d_new(rows * cols)
}

// tf2d_ones(rows, cols): allocate a new rows*cols matrix filled with 1.0_f32.
fn tf2d_ones(rows: i32, cols: i32) -> i32 {
    let n = rows * cols;
    let s = t1d_new(n);
    let one_bits = __bits_of_f32(1.0_f32);
    let mut i: i32 = 0;
    while i < n {
        __arena_set(s + i, one_bits);
        i = i + 1;
    }
    s
}

// tf2d_max_abs(start, rows, cols): @pure. Largest absolute value across
// all elements of a 2D matrix. 0.0 if rows*cols == 0.
@pure
fn tf2d_max_abs(start: i32, rows: i32, cols: i32) -> f32 {
    let n = rows * cols;
    tf1d_max_abs(start, n)
}

// tf1d_count_above(start, n, threshold): @pure. Count elements > threshold.
@pure
fn tf1d_count_above(start: i32, n: i32, threshold: f32) -> i32 {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < n {
        let v = __f32_from_bits(__arena_get(start + i));
        if v > threshold { total = total + 1; }
        i = i + 1;
    }
    total
}

// tf1d_count_below(start, n, threshold): @pure. Companion.
@pure
fn tf1d_count_below(start: i32, n: i32, threshold: f32) -> i32 {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < n {
        let v = __f32_from_bits(__arena_get(start + i));
        if v < threshold { total = total + 1; }
        i = i + 1;
    }
    total
}

// tf1d_count_eq_zero(start, n): @pure. Count elements equal to 0.0_f32.
@pure
fn tf1d_count_eq_zero(start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    let zero_bits = 0;
    while i < n {
        if __arena_get(start + i) == zero_bits { total = total + 1; }
        i = i + 1;
    }
    total
}

// tf1d_is_empty(start, n): @pure. 1 if n == 0.
@pure
fn tf1d_is_empty(start: i32, n: i32) -> i32 {
    if n == 0 { 1 } else { 0 }
}

// ti1d_count_above(start, n, threshold): @pure. Count int elements > threshold.
@pure
fn ti1d_count_above(start: i32, n: i32, threshold: i32) -> i32 {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < n {
        if __arena_get(start + i) > threshold { total = total + 1; }
        i = i + 1;
    }
    total
}

// ti1d_count_below(start, n, threshold): @pure. Companion.
@pure
fn ti1d_count_below(start: i32, n: i32, threshold: i32) -> i32 {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < n {
        if __arena_get(start + i) < threshold { total = total + 1; }
        i = i + 1;
    }
    total
}

// ti1d_max_abs(start, n): @pure. Max of |x[i]| for ints. 0 if empty.
@pure
fn ti1d_max_abs(start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    let mut best: i32 = 0;
    while i < n {
        let v = __arena_get(start + i);
        let av = if v < 0 { 0 - v } else { v };
        if av > best { best = av; }
        i = i + 1;
    }
    best
}

// ti1d_is_empty(start, n): @pure. 1 if n == 0.
@pure
fn ti1d_is_empty(start: i32, n: i32) -> i32 {
    if n == 0 { 1 } else { 0 }
}

// tf1d_first(start, n): @pure. f32 v[0]. 0.0_f32 if empty.
@pure
fn tf1d_first(start: i32, n: i32) -> f32 {
    if n == 0 { 0.0_f32 } else { __f32_from_bits(__arena_get(start)) }
}

// tf1d_last(start, n): @pure. f32 v[n-1]. 0.0_f32 if empty.
@pure
fn tf1d_last(start: i32, n: i32) -> f32 {
    if n == 0 { 0.0_f32 } else { __f32_from_bits(__arena_get(start + n - 1)) }
}

// ti1d_first(start, n): @pure. v[0] or 0 if empty.
@pure
fn ti1d_first(start: i32, n: i32) -> i32 {
    if n == 0 { 0 } else { __arena_get(start) }
}

// ti1d_last(start, n): @pure. v[n-1] or 0 if empty.
@pure
fn ti1d_last(start: i32, n: i32) -> i32 {
    if n == 0 { 0 } else { __arena_get(start + n - 1) }
}

// ti1d_count_eq(start, n, target): @pure. Count of v[i] == target.
@pure
fn ti1d_count_eq(start: i32, n: i32, target: i32) -> i32 {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < n {
        if __arena_get(start + i) == target { total = total + 1; }
        i = i + 1;
    }
    total
}

// ti1d_count_pos(start, n): @pure. Count of strictly positive elements.
@pure
fn ti1d_count_pos(start: i32, n: i32) -> i32 {
    ti1d_count_above(start, n, 0)
}

// ti1d_count_neg(start, n): @pure. Count of strictly negative elements.
@pure
fn ti1d_count_neg(start: i32, n: i32) -> i32 {
    ti1d_count_below(start, n, 0)
}

// ti1d_clone_alloc(start, n): allocate new vec copy. Mirror of vec_clone_alloc.
fn ti1d_clone_alloc(start: i32, n: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < n {
        __arena_push(__arena_get(start + i));
        i = i + 1;
    }
    s
}
