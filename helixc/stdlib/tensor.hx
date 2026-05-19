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
//   t2d_new(rows, cols)         -> i32   reserve rows*cols data slots
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

@pure fn t1d_magic() -> i32 { 1001001 }

@pure fn t1d_footer(n: i32) -> i32 {
    0 - t1d_magic() - n - 1
}

fn t1d_new(n: i32) -> i32 {
    let safe_n = if n < 0 { 0 } else { n };
    __arena_push(t1d_magic());
    __arena_push(safe_n);
    __arena_push(t1d_footer(safe_n));
    let start = __arena_len();
    let mut i: i32 = 0;
    while i < safe_n {
        __arena_push(0);
        i = i + 1;
    }
    __arena_push(t1d_footer(safe_n));
    start
}

@pure fn t1d_capacity_ok(start: i32, n: i32) -> i32 {
    if start < 0 { 0 }
    else { if n < 0 { 0 }
    else { if start < 3 { 0 }
    else { if start >= __arena_len() { 0 }
    else {
        let magic = __arena_get(start - 3);
        if magic == t1d_magic() {
            let len = __arena_get(start - 2);
            let guard = __arena_get(start - 1);
            if len < 0 { 0 }
            else { if n > len { 0 }
            else { if len > 2147483647 - start { 0 }
            else { if start + len >= __arena_len() { 0 }
            else { if guard != t1d_footer(len) { 0 }
            else { if __arena_get(start + len) != t1d_footer(len) { 0 } else { 1 } } } } } }
        } else { if magic == t2d_magic() {
            let rows = __arena_get(start - 2);
            let cols = __arena_get(start - 1);
            let len2 = t2d_len(rows, cols);
            if len2 <= 0 { 0 }
            else { if n > len2 { 0 }
            else { if t2d_shape_ok(start, rows, cols) == 0 { 0 } else { 1 } } }
        } else { 0 } }
    }}}}
}

@pure fn t1d_range_ok(start: i32, off: i32, n: i32) -> i32 {
    if off < 0 { 0 }
    else { if n < 0 { 0 }
    else { if n == 0 { t1d_capacity_ok(start, off) }
    else { if off > 2147483647 - n { 0 }
    else { t1d_capacity_ok(start, off + n) } } } }
}

@pure fn t1d_slice_ok(start: i32, n: i32) -> i32 {
    if t1d_capacity_ok(start, n) != 0 { 1 }
    else { if start < 0 { 0 }
    else { if n < 0 { 0 }
    else { if start >= __arena_len() { 0 }
    else { if n == 0 { 0 }
    else { if start > 2147483647 - n { 0 }
    else {
        let end = start + n;
        let mut base = start - 1;
        let mut ok = 0;
        while base >= 3 {
            if ok == 0 {
                let magic = __arena_get(base - 3);
                if magic == t1d_magic() {
                    let len = __arena_get(base - 2);
                    let guard = __arena_get(base - 1);
                    if len >= 0 {
                    if guard == t1d_footer(len) {
                    if base + len < __arena_len() {
                    if __arena_get(base + len) == t1d_footer(len) {
                    if start >= base {
                    if end <= base + len { ok = 1; };
                    };
                    };
                    };
                    };
                    };
                } else { if magic == t2d_magic() {
                    let rows = __arena_get(base - 2);
                    let cols = __arena_get(base - 1);
                    let len2 = t2d_len(rows, cols);
                    if len2 > 0 {
                    if base + len2 < __arena_len() {
                    if __arena_get(base + len2) == t2d_footer(rows, cols) {
                    if start >= base {
                    if end <= base + len2 { ok = 1; };
                    };
                    };
                    };
                    };
                } else { ok = ok; } }
            };
            base = base - 1;
        }
        ok
    }}}}}}
}

@pure fn t2d_len(rows: i32, cols: i32) -> i32 {
    if rows <= 0 { 0 }
    else { if cols <= 0 { 0 }
    else { if rows > 2147483647 / cols { 0 } else { rows * cols } } }
}

@pure fn t2d_alloc_len(rows: i32, cols: i32) -> i32 {
    if rows <= 0 { 0 }
    else { if cols <= 0 { 0 }
    else { if rows > 2147483647 / cols { 1 } else { rows * cols } } }
}

@pure fn t2d_magic() -> i32 { 2002001 }

@pure fn t2d_footer(rows: i32, cols: i32) -> i32 {
    0 - t2d_magic() - rows - cols
}

@pure fn t2d_error() -> i32 { 35001 }

fn t2d_new(rows: i32, cols: i32) -> i32 {
    let n = t2d_len(rows, cols);
    if n <= 0 {
        let fallback = t2d_alloc_len(rows, cols);
        if fallback <= 0 { __arena_len() } else { t1d_new(fallback) }
    } else {
        let start = __arena_len();
        __arena_push(t2d_magic());
        __arena_push(rows);
        __arena_push(cols);
        let mut i: i32 = 0;
        while i < n {
            __arena_push(0);
            i = i + 1;
        }
        __arena_push(t2d_footer(rows, cols));
        start + 3
    }
}

@pure fn arena_span_in_tensor_payload(span_start: i32, span_len: i32) -> i32 {
    if span_start < 0 { 0 }
    else { if span_len <= 0 { 0 }
    else { if span_start > 2147483647 - span_len { 0 }
    else {
        let span_end = span_start + span_len;
        let mut base = span_start;
        let mut found = 0;
        while base >= 3 {
            if found == 0 {
                let magic = __arena_get(base - 3);
                if magic == t1d_magic() {
                    let len = __arena_get(base - 2);
                    let guard = __arena_get(base - 1);
                    if len >= 0 {
                    if guard == t1d_footer(len) {
                    if base + len < __arena_len() {
                    if __arena_get(base + len) == t1d_footer(len) {
                    if span_start >= base {
                    if span_end <= base + len { found = 1; };
                    };
                    };
                    };
                    };
                    };
                } else { if magic == t2d_magic() {
                    let rows = __arena_get(base - 2);
                    let cols = __arena_get(base - 1);
                    let len2 = t2d_len(rows, cols);
                    if len2 > 0 {
                    if base + len2 < __arena_len() {
                    if __arena_get(base + len2) == t2d_footer(rows, cols) {
                    if span_start >= base {
                    if span_end <= base + len2 { found = 1; };
                    };
                    };
                    };
                    };
                } else { found = found; } }
            };
            base = base - 1;
        }
        found
    } } }
}

@pure fn t2d_offset(start: i32, cols: i32, i: i32, j: i32) -> i32 {
    if start < 0 { 0 - 1 }
    else { if start < 3 { 0 - 1 } else {
    let rows = __arena_get(start - 2);
    if __arena_get(start - 3) != t2d_magic() { 0 - 1 }
    else { if __arena_get(start - 1) != cols { 0 - 1 }
    else { if t2d_shape_ok(start, rows, cols) == 0 { 0 - 1 }
    else { if cols <= 0 { 0 - 1 }
    else { if rows <= 0 { 0 - 1 }
    else { if i < 0 { 0 - 1 }
    else { if i >= rows { 0 - 1 }
    else { if j < 0 { 0 - 1 }
    else { if j >= cols { 0 - 1 }
    else { if i > (2147483647 - j) / cols { 0 - 1 } else {
        let linear = i * cols + j;
        if linear > 2147483647 - start { 0 - 1 } else { start + linear }
    }}}}}}}}}}}}
}

@pure fn t2d_shape_ok(start: i32, rows: i32, cols: i32) -> i32 {
    if t2d_len(rows, cols) == 0 { 0 }
    else { if start < 3 { 0 }
    else { if __arena_get(start - 3) != t2d_magic() { 0 }
    else { if __arena_get(start - 2) != rows { 0 }
    else { if __arena_get(start - 1) != cols { 0 }
    else {
        let n = t2d_len(rows, cols);
        if n > 2147483647 - start { 0 }
        else { if start + n >= __arena_len() { 0 }
        else { if __arena_get(start + n) != t2d_footer(rows, cols) { 0 }
        else { if n > 2147483647 - 4 { 0 }
        else { if arena_span_in_tensor_payload(start - 3, n + 4) != 0 { 0 }
        else { 1 } } } } }
    }}}}}
}

@pure fn t2d_shape_status(start: i32, rows: i32, cols: i32) -> i32 {
    if rows <= 0 { 0 }
    else { if cols <= 0 { 0 }
    else { if t2d_len(rows, cols) == 0 { t2d_error() }
    else { if t2d_shape_ok(start, rows, cols) == 0 { t2d_error() }
    else { 0 } }}}
}

fn t1d_set_i32_bits(start: i32, i: i32, bits: i32) -> i32 {
    if i < 0 { t2d_error() }
    else { if t1d_slice_ok(start, i + 1) == 0 { t2d_error() }
    else {
        __arena_set(start + i, bits);
        0
    }}
}

fn t1d_get_i32_bits(start: i32, i: i32) -> i32 {
    if i < 0 { 0 }
    else { if t1d_slice_ok(start, i + 1) == 0 { 0 }
    else { __arena_get(start + i) } }
}

// Integer-tensor variants (no float-bit-cast needed). These are the
// safe variants until float<->arena bit-reinterpret lands as a
// codegen primitive.
@pure fn ti1d_get(start: i32, i: i32) -> i32 {
    if i < 0 { 0 }
    else { if t1d_slice_ok(start, i + 1) == 0 { 0 }
    else { __arena_get(start + i) } }
}

fn ti1d_set(start: i32, i: i32, x: i32) -> i32 {
    if i < 0 { t2d_error() }
    else { if t1d_slice_ok(start, i + 1) == 0 { t2d_error() }
    else {
        __arena_set(start + i, x);
        x
    }}
}

// Restart 51 A2: i64 accumulator + INT32 saturation. Matches the
// ti1d_prod / hashmap_sum_values / mse_loss precedent. Prior i32 total
// silently wrapped for any non-trivial integer-tensor sum.
@pure fn ti1d_sum(start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(start, n) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut total: i64 = 0_i64;
    let hi: i64 = 2147483647_i64;
    let lo: i64 = (0_i64 - 2147483647_i64) - 1_i64;
    while i < n {
        total = total + (__arena_get(start + i) as i64);
        if total > hi { total = hi; }
        else { if total < lo { total = lo; } };
        i = i + 1;
    }
    total as i32
    }}
}

// Restart 51 A3: i64 accumulator + INT32 saturation. Single |a[i]*b[i]|
// term can overflow i32 at values around 46341. Same family as
// mse_loss / ti1d_sum.
@pure fn ti1d_dot(a_start: i32, b_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(a_start, n) == 0 { 0 }
    else { if t1d_slice_ok(b_start, n) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut total: i64 = 0_i64;
    let hi: i64 = 2147483647_i64;
    let lo: i64 = (0_i64 - 2147483647_i64) - 1_i64;
    while i < n {
        total = total + (__arena_get(a_start + i) as i64) * (__arena_get(b_start + i) as i64);
        if total > hi { total = hi; }
        else { if total < lo { total = lo; } };
        i = i + 1;
    }
    total as i32
    }}}
}

// Restart 53 A5: i64 intermediate + INT32 saturation per element.
// Sibling of ti1d_dot / ti2d_matmul saturation; the per-element
// write still needed protection because a single a*xi term can
// overflow i32 at modest magnitudes (~46341).
fn ti1d_axpy(y_start: i32, a: i32, x_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(y_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    let hi: i64 = 2147483647_i64;
    let lo: i64 = (0_i64 - 2147483647_i64) - 1_i64;
    while i < n {
        let cur: i64 = __arena_get(y_start + i) as i64;
        let xi: i64 = __arena_get(x_start + i) as i64;
        let mut v: i64 = cur + (a as i64) * xi;
        if v > hi { v = hi; }
        else { if v < lo { v = lo; } };
        __arena_set(y_start + i, v as i32);
        i = i + 1;
    }
    0
    }}}
}

// 2D row-major access: M[i,j] lives at slot start + i*cols + j.
fn ti2d_new(rows: i32, cols: i32) -> i32 {
    t2d_new(rows, cols)
}

fn ti2d_set(start: i32, cols: i32, i: i32, j: i32, x: i32) -> i32 {
    let off = t2d_offset(start, cols, i, j);
    if off < 0 { t2d_error() }
    else {
        __arena_set(off, x);
        x
    }
}

// DEPRECATED for safety-critical new code (batch 15 deprecation
// sweep): returns 0 on OOB which collides with legitimate sparse
// zero. Use ti2d_get_strict (returns INT32_MIN on OOB) or
// ti2d_in_bounds for explicit bounds check.
@pure fn ti2d_get(start: i32, cols: i32, i: i32, j: i32) -> i32 {
    let off = t2d_offset(start, cols, i, j);
    if off < 0 { 0 } else { __arena_get(off) }
}

// Cycle 1 Batch RT fix batch 12 (silent-failure HIGH-5):
// Pre-fix: ti2d_get OOB silently returned 0 — indistinguishable
// from a legitimate sparse zero. Heavily used by ti2d_matmul,
// nn.argmax_rows, etc. Caller had no way to detect OOB.
// Post-fix: ti2d_in_bounds + ti2d_get_strict (INT32_MIN on OOB).
// Original ti2d_get unchanged for backward compat.
@pure fn ti2d_in_bounds(start: i32, cols: i32, i: i32, j: i32) -> i32 {
    let off = t2d_offset(start, cols, i, j);
    if off < 0 { 0 } else { 1 }
}

@pure fn ti2d_get_strict(start: i32, cols: i32, i: i32, j: i32) -> i32 {
    let off = t2d_offset(start, cols, i, j);
    if off < 0 { (0_i32 - 2147483647_i32) - 1_i32 }
    else { __arena_get(off) }
}

// y = W @ x. W is rows*cols, x is cols, y is rows.
fn ti2d_matvec(w_start: i32, w_rows: i32, w_cols: i32,
               x_start: i32, y_start: i32) -> i32 {
    if w_rows <= 0 { 0 }
    else { if w_cols <= 0 { 0 }
    else { if t2d_len(w_rows, w_cols) == 0 { t2d_error() }
    else { if t2d_shape_ok(w_start, w_rows, w_cols) == 0 { t2d_error() }
    else { if t1d_slice_ok(x_start, w_cols) == 0 { t2d_error() }
    else { if t1d_slice_ok(y_start, w_rows) == 0 { t2d_error() }
    else {
    // Restart 52 A1: i64 accumulator + INT32 saturation per output cell.
    // Sibling of restart 51 A3 (ti1d_dot) — the 1D case got the fix but
    // the 2D matvec was missed in that sweep. Single |w|*|x| term
    // overflows i32 at values around 46341; running sum wraps faster.
    let mut r: i32 = 0;
    let hi: i64 = 2147483647_i64;
    let lo: i64 = (0_i64 - 2147483647_i64) - 1_i64;
    while r < w_rows {
        let mut c: i32 = 0;
        let mut acc: i64 = 0_i64;
        while c < w_cols {
            acc = acc + (__arena_get(w_start + r * w_cols + c) as i64) * (__arena_get(x_start + c) as i64);
            if acc > hi { acc = hi; }
            else { if acc < lo { acc = lo; } };
            c = c + 1;
        }
        __arena_set(y_start + r, acc as i32);
        r = r + 1;
    }
    0
    }}}}}}
}

// Element-wise: y[i] = relu(x[i]) for i in [0, n). Integer relu.
fn ti1d_relu(x_start: i32, y_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(y_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    while i < n {
        let xi = __arena_get(x_start + i);
        let v = if xi > 0 { xi } else { 0 };
        __arena_set(y_start + i, v);
        i = i + 1;
    }
    0
    }}}
}

// Element-wise add: z[i] = x[i] + y[i]. Returns 0.
// Restart 54 A2: per-element i64 intermediate + INT32 saturation.
// Sibling sweep of restart 53 A5 (ti1d_axpy / *_scalar) extended to
// the binary integer Hadamard ops the family missed.
fn ti1d_add(x_start: i32, y_start: i32, z_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(y_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(z_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    let hi: i64 = 2147483647_i64;
    let lo: i64 = (0_i64 - 2147483647_i64) - 1_i64;
    while i < n {
        let mut v: i64 = (__arena_get(x_start + i) as i64) + (__arena_get(y_start + i) as i64);
        if v > hi { v = hi; }
        else { if v < lo { v = lo; } };
        __arena_set(z_start + i, v as i32);
        i = i + 1;
    }
    0
    }}}}
}

// Integer element-wise subtraction: z[i] = x[i] - y[i].
// Companion to ti1d_add. z must be pre-allocated.
// Restart 54 A2: per-element i64 intermediate + INT32 saturation.
fn ti1d_sub(x_start: i32, y_start: i32, z_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(y_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(z_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    let hi: i64 = 2147483647_i64;
    let lo: i64 = (0_i64 - 2147483647_i64) - 1_i64;
    while i < n {
        let mut v: i64 = (__arena_get(x_start + i) as i64) - (__arena_get(y_start + i) as i64);
        if v > hi { v = hi; }
        else { if v < lo { v = lo; } };
        __arena_set(z_start + i, v as i32);
        i = i + 1;
    }
    0
    }}}}
}

// Integer element-wise multiplication (Hadamard): z[i] = x[i] * y[i].
// For inner product use ti1d_dot.
// Restart 54 A2: per-element i64 intermediate + INT32 saturation —
// a single 46341 × 46341 multiply otherwise silently wraps.
fn ti1d_mul(x_start: i32, y_start: i32, z_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(y_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(z_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    let hi: i64 = 2147483647_i64;
    let lo: i64 = (0_i64 - 2147483647_i64) - 1_i64;
    while i < n {
        let mut v: i64 = (__arena_get(x_start + i) as i64) * (__arena_get(y_start + i) as i64);
        if v > hi { v = hi; }
        else { if v < lo { v = lo; } };
        __arena_set(z_start + i, v as i32);
        i = i + 1;
    }
    0
    }}}}
}

// =========================================================================
// Phase 2.2 step 2: f32 tensor primitives via __bits_of_f32 reinterpret.
// =========================================================================
// f32 values stored as their IEEE 754 bit pattern in arena slots (4 bytes
// each, same width as i32). The codegen primitive __bits_of_f32 /
// __f32_from_bits relabels the same 4 bytes — no instruction emitted, just
// a type-system shim.

fn tf1d_set(start: i32, i: i32, x: f32) -> i32 {
    if i < 0 { t2d_error() }
    else { if t1d_slice_ok(start, i + 1) == 0 { t2d_error() }
    else {
        __arena_set(start + i, __bits_of_f32(x));
        0
    }}
}

@pure fn tf1d_get(start: i32, i: i32) -> f32 {
    if i < 0 { 0.0_f32 }
    else { if t1d_slice_ok(start, i + 1) == 0 { 0.0_f32 }
    else { __f32_from_bits(__arena_get(start + i)) } }
}

@pure
// Restart 56 A1 (filed in ledger Increment 75 by the restart 57
// catch-up sweep; the original commit number 57 in the prior comment
// was off-by-one with the actual restart number): NaN-skip discipline.
// A single NaN slot would otherwise poison the entire sum (NaN + anything
// = NaN). Matches the softmax_layer / layer_norm_f32 / clip_grad_norm_f32
// / adam_f32_step NaN-fail-closed precedent — distinguish "garbage in
// one slot" from "garbage in every output". The sibling sweep across
// tf1d_dot / tf1d_l1_norm / tf1d_max_abs / tf1d_sum_in_range is a
// deliberate carry-forward into restart 58 Lane A.
fn tf1d_sum(start: i32, n: i32) -> f32 {
    if n <= 0 { 0.0_f32 }
    else { if t1d_slice_ok(start, n) == 0 { 0.0_f32 }
    else {
    let mut i: i32 = 0;
    let mut total: f32 = 0.0_f32;
    while i < n {
        let v = __f32_from_bits(__arena_get(start + i));
        if v == v { total = total + v; };
        i = i + 1;
    }
    total
    }}
}

// Restart 58 A1: NaN-skip on per-element product. Single NaN slot in
// either input no longer poisons the entire dot product. Same family
// as restart 57 tf1d_sum.
@pure
fn tf1d_dot(a_start: i32, b_start: i32, n: i32) -> f32 {
    if n <= 0 { 0.0_f32 }
    else { if t1d_slice_ok(a_start, n) == 0 { 0.0_f32 }
    else { if t1d_slice_ok(b_start, n) == 0 { 0.0_f32 }
    else {
    let mut i: i32 = 0;
    let mut total: f32 = 0.0_f32;
    while i < n {
        let av = __f32_from_bits(__arena_get(a_start + i));
        let bv = __f32_from_bits(__arena_get(b_start + i));
        let prod = av * bv;
        if prod == prod { total = total + prod; };
        i = i + 1;
    }
    total
    }}}
}

fn tf1d_axpy(y_start: i32, a: f32, x_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(y_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    while i < n {
        let cur = __f32_from_bits(__arena_get(y_start + i));
        let xi = __f32_from_bits(__arena_get(x_start + i));
        __arena_set(y_start + i, __bits_of_f32(cur + a * xi));
        i = i + 1;
    }
    0
    }}}
}

fn tf1d_relu(x_start: i32, y_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(y_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    while i < n {
        let xi = __f32_from_bits(__arena_get(x_start + i));
        let v = if xi > 0.0_f32 { xi } else { 0.0_f32 };
        __arena_set(y_start + i, __bits_of_f32(v));
        i = i + 1;
    }
    0
    }}}
}

// DEPRECATED for safety-critical new code (batch 15 deprecation
// sweep): returns 0.0_f32 on OOB which collides with legitimate
// sparse zero. Use tf2d_get_or(...sentinel) with a caller-supplied
// f32 sentinel, or tf2d_in_bounds for explicit bounds check.
@pure
fn tf2d_get(start: i32, cols: i32, i: i32, j: i32) -> f32 {
    let off = t2d_offset(start, cols, i, j);
    if off < 0 { 0.0_f32 } else { __f32_from_bits(__arena_get(off)) }
}

// Cycle 1 Batch RT fix batch 12 (silent-failure HIGH-5):
// f32 analog of ti2d_get_strict — caller passes an explicit sentinel
// (typically a value impossible-by-domain, e.g. -1.0e30_f32 for
// non-negative tensors). Original tf2d_get unchanged.
@pure
fn tf2d_get_or(start: i32, cols: i32, i: i32, j: i32, sentinel: f32) -> f32 {
    let off = t2d_offset(start, cols, i, j);
    if off < 0 { sentinel }
    else { __f32_from_bits(__arena_get(off)) }
}

@pure fn tf2d_in_bounds(start: i32, cols: i32, i: i32, j: i32) -> i32 {
    let off = t2d_offset(start, cols, i, j);
    if off < 0 { 0 } else { 1 }
}

fn tf2d_set(start: i32, cols: i32, i: i32, j: i32, x: f32) -> i32 {
    let off = t2d_offset(start, cols, i, j);
    if off < 0 { t2d_error() }
    else {
        __arena_set(off, __bits_of_f32(x));
        0
    }
}

// Restart 58 A4 (Increment 77 catch-up sweep): NaN-skip on per-cell
// product. Without it, one NaN slot in w (or x) poisons the entire
// matvec output row (or every output cell). Matches the tf1d_dot
// NaN-skip pattern.
fn tf2d_matvec(w_start: i32, w_rows: i32, w_cols: i32,
               x_start: i32, y_start: i32) -> i32 {
    if w_rows <= 0 { 0 }
    else { if w_cols <= 0 { 0 }
    else { if t2d_len(w_rows, w_cols) == 0 { t2d_error() }
    else { if t2d_shape_ok(w_start, w_rows, w_cols) == 0 { t2d_error() }
    else { if t1d_slice_ok(x_start, w_cols) == 0 { t2d_error() }
    else { if t1d_slice_ok(y_start, w_rows) == 0 { t2d_error() }
    else {
    let mut r: i32 = 0;
    while r < w_rows {
        let mut c: i32 = 0;
        let mut acc: f32 = 0.0_f32;
        while c < w_cols {
            let wv = __f32_from_bits(__arena_get(w_start + r * w_cols + c));
            let xv = __f32_from_bits(__arena_get(x_start + c));
            let prod = wv * xv;
            if prod == prod { acc = acc + prod; };
            c = c + 1;
        }
        __arena_set(y_start + r, __bits_of_f32(acc));
        r = r + 1;
    }
    0
    }}}}}}
}

// =========================================================================
// Phase 2 perfection: reductions, more ops.
// =========================================================================

// Integer-tensor mean (returns floor(sum/n)).
@pure fn ti1d_mean(start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { ti1d_sum(start, n) / n }
}

// Integer-tensor product.
// Restart 50 A3: i64 accumulator + INT32 saturation so a product that
// would silently wrap an i32 (e.g. 32 elements of value 2 → 2^32 → 0
// previously) now returns INT32_MAX / INT32_MIN. Mirrors the
// `hashmap_sum_values` saturating-i64 precedent. Callers needing the
// exact i64 product should accumulate themselves.
@pure fn ti1d_prod(start: i32, n: i32) -> i32 {
    if n <= 0 { 1 }
    else { if t1d_slice_ok(start, n) == 0 { 1 }
    else {
        let mut i: i32 = 0;
        let mut p: i64 = 1_i64;
        let lo: i64 = 0_i64 - 2147483647_i64 - 1_i64;
        let hi: i64 = 2147483647_i64;
        while i < n {
            p = p * (__arena_get(start + i) as i64);
            if p > hi { p = hi; }
            else { if p < lo { p = lo; } };
            i = i + 1;
        }
        p as i32
    }}
}

// Integer-tensor min.
@pure fn ti1d_min(start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(start, n) == 0 { 0 }
    else {
        let mut best = __arena_get(start);
        let mut i: i32 = 1;
        while i < n {
            let v = __arena_get(start + i);
            if v < best { best = v; }
            i = i + 1;
        }
        best
    }}
}

// Integer-tensor max (companion to ti1d_min).
@pure fn ti1d_max(start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(start, n) == 0 { 0 }
    else {
        let mut best = __arena_get(start);
        let mut i: i32 = 1;
        while i < n {
            let v = __arena_get(start + i);
            if v > best { best = v; }
            i = i + 1;
        }
        best
    }}
}

// Integer-tensor argmin (returns index of smallest element).
// Companion to ti1d_argmax. n == 0 returns -1.
@pure fn ti1d_argmin(start: i32, n: i32) -> i32 {
    if n <= 0 { 0 - 1 }
    else { if t1d_slice_ok(start, n) == 0 { 0 - 1 }
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
    }}
}

// Integer-tensor argmax (returns index of largest element).
@pure fn ti1d_argmax(start: i32, n: i32) -> i32 {
    if n <= 0 { 0 - 1 }
    else { if t1d_slice_ok(start, n) == 0 { 0 - 1 }
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
    }}
}

// Tensor zeros: allocate n slots already initialized to 0.
fn ti1d_zeros(n: i32) -> i32 { t1d_new(n) }

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
    if a_rows <= 0 { 0 }
    else { if a_cols <= 0 { 0 }
    else { if b_cols <= 0 { 0 }
    else { if t2d_len(a_rows, a_cols) == 0 { t2d_error() }
    else { if t2d_len(a_cols, b_cols) == 0 { t2d_error() }
    else { if t2d_len(a_rows, b_cols) == 0 { t2d_error() }
    else { if t2d_shape_ok(a_start, a_rows, a_cols) == 0 { t2d_error() }
    else { if t2d_shape_ok(b_start, a_cols, b_cols) == 0 { t2d_error() }
    else { if t2d_shape_ok(c_start, a_rows, b_cols) == 0 { t2d_error() }
    else {
    // Restart 52 A1: i64 accumulator + INT32 saturation per output cell.
    // Sibling of restart 51 A3 (ti1d_dot) extended to 2D matmul.
    let mut r: i32 = 0;
    let hi: i64 = 2147483647_i64;
    let lo: i64 = (0_i64 - 2147483647_i64) - 1_i64;
    while r < a_rows {
        let mut c: i32 = 0;
        while c < b_cols {
            let mut k: i32 = 0;
            let mut acc: i64 = 0_i64;
            while k < a_cols {
                let av: i64 = __arena_get(a_start + r * a_cols + k) as i64;
                let bv: i64 = __arena_get(b_start + k * b_cols + c) as i64;
                acc = acc + av * bv;
                if acc > hi { acc = hi; }
                else { if acc < lo { acc = lo; } };
                k = k + 1;
            }
            __arena_set(c_start + r * b_cols + c, acc as i32);
            c = c + 1;
        }
        r = r + 1;
    }
    0
    }}}}}}}}}
}

// Reshape: copy n elements from src to dst. (For row-major tensors a
// reshape is a no-op as long as total element count matches; for safety
// we expose copy as the explicit form.)
fn ti1d_copy(src: i32, dst: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(src, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(dst, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    while i < n {
        __arena_set(dst + i, __arena_get(src + i));
        i = i + 1;
    }
    0
    }}}
}

// Broadcasting: y[i] = x[i] + scalar (element-wise add of scalar to vec).
// Restart 53 A5: per-element i64 intermediate + INT32 saturation.
fn ti1d_add_scalar(x_start: i32, scalar: i32, y_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(y_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    let hi: i64 = 2147483647_i64;
    let lo: i64 = (0_i64 - 2147483647_i64) - 1_i64;
    let s: i64 = scalar as i64;
    while i < n {
        let mut v: i64 = (__arena_get(x_start + i) as i64) + s;
        if v > hi { v = hi; }
        else { if v < lo { v = lo; } };
        __arena_set(y_start + i, v as i32);
        i = i + 1;
    }
    0
    }}}
}

// Restart 53 A5: per-element i64 intermediate + INT32 saturation.
// Sibling of ti1d_axpy / ti1d_add_scalar; protects the per-element
// product from silent i32 wrap-around.
fn ti1d_mul_scalar(x_start: i32, scalar: i32, y_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(y_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    let hi: i64 = 2147483647_i64;
    let lo: i64 = (0_i64 - 2147483647_i64) - 1_i64;
    let s: i64 = scalar as i64;
    while i < n {
        let mut v: i64 = (__arena_get(x_start + i) as i64) * s;
        if v > hi { v = hi; }
        else { if v < lo { v = lo; } };
        __arena_set(y_start + i, v as i32);
        i = i + 1;
    }
    0
    }}}
}

// f32 reductions
@pure fn tf1d_mean(start: i32, n: i32) -> f32 {
    if n <= 0 { 0.0_f32 }
    else { tf1d_sum(start, n) / (n as f32) }
}

// Restart 58 A7 (Increment 77 catch-up sweep): NaN-at-index-0 robustness.
// Bare-init `best = arena_get(start)` would freeze the result at NaN if
// the first slot is NaN (IEEE-754 `v > NaN` is false). Initialize with
// `seen = 0` sentinel and adopt the first non-NaN slot as the running
// best. All-NaN input returns 0.0 (existing convention for empty input).
@pure fn tf1d_max(start: i32, n: i32) -> f32 {
    if n <= 0 { 0.0_f32 }
    else { if t1d_slice_ok(start, n) == 0 { 0.0_f32 }
    else {
        let mut best: f32 = 0.0_f32;
        let mut seen: i32 = 0;
        let mut i: i32 = 0;
        while i < n {
            let v = __f32_from_bits(__arena_get(start + i));
            if v == v {
                if seen == 0 { best = v; seen = 1; }
                else { if v > best { best = v; }; };
            };
            i = i + 1;
        }
        best
    }}
}

@pure fn tf1d_min(start: i32, n: i32) -> f32 {
    if n <= 0 { 0.0_f32 }
    else { if t1d_slice_ok(start, n) == 0 { 0.0_f32 }
    else {
        let mut best: f32 = 0.0_f32;
        let mut seen: i32 = 0;
        let mut i: i32 = 0;
        while i < n {
            let v = __f32_from_bits(__arena_get(start + i));
            if v == v {
                if seen == 0 { best = v; seen = 1; }
                else { if v < best { best = v; }; };
            };
            i = i + 1;
        }
        best
    }}
}

@pure fn tf1d_argmax(start: i32, n: i32) -> i32 {
    if n <= 0 { 0 - 1 }
    else { if t1d_slice_ok(start, n) == 0 { 0 - 1 }
    else {
        let mut best_idx: i32 = 0;
        let mut best_val: f32 = 0.0_f32;
        let mut seen: i32 = 0;
        let mut i: i32 = 0;
        while i < n {
            let v = __f32_from_bits(__arena_get(start + i));
            if v == v {
                if seen == 0 { best_val = v; best_idx = i; seen = 1; }
                else { if v > best_val { best_val = v; best_idx = i; }; };
            };
            i = i + 1;
        }
        best_idx
    }}
}

// f32 element-wise: z[i] = x[i] + y[i].
fn tf1d_add(x_start: i32, y_start: i32, z_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(y_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(z_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    while i < n {
        let xv = __f32_from_bits(__arena_get(x_start + i));
        let yv = __f32_from_bits(__arena_get(y_start + i));
        __arena_set(z_start + i, __bits_of_f32(xv + yv));
        i = i + 1;
    }
    0
    }}}}
}

fn tf1d_sub(x_start: i32, y_start: i32, z_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(y_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(z_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    while i < n {
        let xv = __f32_from_bits(__arena_get(x_start + i));
        let yv = __f32_from_bits(__arena_get(y_start + i));
        __arena_set(z_start + i, __bits_of_f32(xv - yv));
        i = i + 1;
    }
    0
    }}}}
}

fn tf1d_mul(x_start: i32, y_start: i32, z_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(y_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(z_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    while i < n {
        let xv = __f32_from_bits(__arena_get(x_start + i));
        let yv = __f32_from_bits(__arena_get(y_start + i));
        __arena_set(z_start + i, __bits_of_f32(xv * yv));
        i = i + 1;
    }
    0
    }}}}
}

// f32 broadcasting: y[i] = x[i] + scalar.
fn tf1d_add_scalar(x_start: i32, scalar: f32, y_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(y_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    while i < n {
        let xv = __f32_from_bits(__arena_get(x_start + i));
        __arena_set(y_start + i, __bits_of_f32(xv + scalar));
        i = i + 1;
    }
    0
    }}}
}

fn tf1d_mul_scalar(x_start: i32, scalar: f32, y_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(y_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    while i < n {
        let xv = __f32_from_bits(__arena_get(x_start + i));
        __arena_set(y_start + i, __bits_of_f32(xv * scalar));
        i = i + 1;
    }
    0
    }}}
}

// 2D row-major f32 matmul: C = A @ B.
//   A is (a_rows x a_cols), B is (a_cols x b_cols), C is (a_rows x b_cols).
// Restart 58 A4 (Increment 77 catch-up sweep): NaN-skip per-cell, same
// pattern as tf2d_matvec.
fn tf2d_matmul(a_start: i32, a_rows: i32, a_cols: i32,
               b_start: i32, b_cols: i32, c_start: i32) -> i32 {
    if a_rows <= 0 { 0 }
    else { if a_cols <= 0 { 0 }
    else { if b_cols <= 0 { 0 }
    else { if t2d_len(a_rows, a_cols) == 0 { t2d_error() }
    else { if t2d_len(a_cols, b_cols) == 0 { t2d_error() }
    else { if t2d_len(a_rows, b_cols) == 0 { t2d_error() }
    else { if t2d_shape_ok(a_start, a_rows, a_cols) == 0 { t2d_error() }
    else { if t2d_shape_ok(b_start, a_cols, b_cols) == 0 { t2d_error() }
    else { if t2d_shape_ok(c_start, a_rows, b_cols) == 0 { t2d_error() }
    else {
    let mut r: i32 = 0;
    while r < a_rows {
        let mut c: i32 = 0;
        while c < b_cols {
            let mut k: i32 = 0;
            let mut acc: f32 = 0.0_f32;
            while k < a_cols {
                let av = __f32_from_bits(__arena_get(a_start + r * a_cols + k));
                let bv = __f32_from_bits(__arena_get(b_start + k * b_cols + c));
                let prod = av * bv;
                if prod == prod { acc = acc + prod; };
                k = k + 1;
            }
            __arena_set(c_start + r * b_cols + c, __bits_of_f32(acc));
            c = c + 1;
        }
        r = r + 1;
    }
    0
    }}}}}}}}}
}


// f32-tensor zeros: allocate n slots; arena push 0 leaves the bit
// pattern as IEEE +0.0 which is exactly what we want.
fn tf1d_zeros(n: i32) -> i32 { t1d_new(n) }

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
    if rows <= 0 { 0 }
    else { if cols <= 0 { 0 }
    else { if t2d_len(rows, cols) == 0 { t2d_error() }
    else { if t2d_shape_ok(src, rows, cols) == 0 { t2d_error() }
    else { if t2d_shape_ok(dst, cols, rows) == 0 { t2d_error() }
    else {
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
    }}}}}
}

// ti1d_clamp(x, lo, hi, dst, n): elementwise clamp each x[i] into
// [lo, hi]. Out-of-place; result written to dst (caller pre-allocates
// dst with t1d_new(n) or shares with x for in-place).
fn ti1d_clamp(x_start: i32, lo: i32, hi: i32, dst: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(dst, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    while i < n {
        let v = __arena_get(x_start + i);
        let cv = if v < lo { lo } else { if v > hi { hi } else { v } };
        __arena_set(dst + i, cv);
        i = i + 1;
    }
    0
    }}}
}

// ti1d_l1_norm(x, n): L1 norm = sum of |x[i]|. @pure (read-only).
// Restart 51 A4: i64 accumulator + INT32 saturation, plus INT32_MIN
// special case for abs (0 - INT32_MIN wraps back to INT32_MIN in i32).
@pure fn ti1d_l1_norm(start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(start, n) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut total: i64 = 0_i64;
    let hi: i64 = 2147483647_i64;
    while i < n {
        let v: i64 = __arena_get(start + i) as i64;
        let av: i64 = if v < 0_i64 { 0_i64 - v } else { v };
        total = total + av;
        if total > hi { total = hi; };
        i = i + 1;
    }
    total as i32
    }}
}

// ti1d_l2_norm_sq(x, n): squared L2 norm = sum(x[i]^2). Mirrors
// ti1d_dot(x, x, n). Distinct fn for clarity at call sites and
// future overflow-safe variants. @pure.
@pure fn ti1d_l2_norm_sq(start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(start, n) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    // Restart 51 A5: i64 accumulator + INT32 saturation. Single |v|>=46341
    // makes v*v exceed INT32_MAX. Matches mse_loss / ti1d_dot precedent.
    let mut total: i64 = 0_i64;
    let hi: i64 = 2147483647_i64;
    while i < n {
        let v: i64 = __arena_get(start + i) as i64;
        total = total + v * v;
        if total > hi { total = hi; };
        i = i + 1;
    }
    total as i32
    }}
}

// ti1d_eq_count(a, b, n): count of indices i where a[i] == b[i].
// Returns an integer in [0, n]. Useful for accuracy/agreement metrics.
@pure fn ti1d_eq_count(a_start: i32, b_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(a_start, n) == 0 { 0 }
    else { if t1d_slice_ok(b_start, n) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < n {
        if __arena_get(a_start + i) == __arena_get(b_start + i) {
            total = total + 1;
        };
        i = i + 1;
    }
    total
    }}}
}

// tf1d_l2_norm_sq(x, n): squared L2 norm = sum(x[i]^2). f32 mirror of
// ti1d_l2_norm_sq. Builds on tf1d_dot(x, x, n) — distinct fn for clarity.
fn tf1d_l2_norm_sq(start: i32, n: i32) -> f32 {
    tf1d_dot(start, start, n)
}

// tf1d_l1_norm(x, n): L1 norm = sum of |x[i]|. f32 mirror of ti1d_l1_norm.
// Reads each f32, computes absolute value via SSE fabs (sign-bit clear),
// accumulates. Loop is the bf32 form so float promotion is implicit.
// Restart 58 A2: NaN-skip. Same family as tf1d_sum / tf1d_dot.
fn tf1d_l1_norm(start: i32, n: i32) -> f32 {
    if n <= 0 { 0.0_f32 }
    else { if t1d_slice_ok(start, n) == 0 { 0.0_f32 }
    else {
    let mut i: i32 = 0;
    let mut total: f32 = 0.0_f32;
    while i < n {
        let v = __f32_from_bits(__arena_get(start + i));
        if v == v {
        let av = if v < 0.0_f32 { 0.0_f32 - v } else { v };
        total = total + av;
        };
        i = i + 1;
    }
    total
    }}
}

// tf2d_transpose(src, rows, cols, dst): out-of-place transpose for f32
// tensors. f32 mirror of ti2d_transpose. Caller pre-allocates dst with
// t1d_new(rows*cols).
fn tf2d_transpose(src: i32, rows: i32, cols: i32, dst: i32) -> i32 {
    if rows <= 0 { 0 }
    else { if cols <= 0 { 0 }
    else { if t2d_len(rows, cols) == 0 { t2d_error() }
    else { if t2d_shape_ok(src, rows, cols) == 0 { t2d_error() }
    else { if t2d_shape_ok(dst, cols, rows) == 0 { t2d_error() }
    else {
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
    }}}}}
}

// tf1d_clamp(x, lo, hi, dst, n): elementwise clamp each x[i] into [lo, hi].
// f32 mirror of ti1d_clamp.
fn tf1d_clamp(x_start: i32, lo: f32, hi: f32, dst: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(dst, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    while i < n {
        let v = __f32_from_bits(__arena_get(x_start + i));
        let cv = if v < lo { lo } else { if v > hi { hi } else { v } };
        __arena_set(dst + i, __bits_of_f32(cv));
        i = i + 1;
    }
    0
    }}}
}

// tf1d_argmin(start, n): @pure. Index of the smallest f32 element.
// Returns -1 if n == 0.
// Restart 58 A7 (Increment 77 catch-up sweep): NaN-at-index-0 robustness.
// Same idiom as tf1d_argmax.
@pure fn tf1d_argmin(start: i32, n: i32) -> i32 {
    if n <= 0 { 0 - 1 }
    else { if t1d_slice_ok(start, n) == 0 { 0 - 1 }
    else {
        let mut best_idx: i32 = 0;
        let mut best: f32 = 0.0_f32;
        let mut seen: i32 = 0;
        let mut i: i32 = 0;
        while i < n {
            let v = __f32_from_bits(__arena_get(start + i));
            if v == v {
                if seen == 0 { best = v; best_idx = i; seen = 1; }
                else { if v < best { best = v; best_idx = i; }; };
            };
            i = i + 1;
        }
        best_idx
    }}
}

// tf1d_running_sum(start, n): allocate a new vec where r[i] = sum
// over x[0..=i]. Like vec_cumsum but for f32. r[0] = x[0]. Useful
// for prefix-sum queries.
// Restart 61 A1: NaN-skip on per-element accumulation. Same family as
// tf1d_sum / tf1d_dot / tf1d_l1_norm / tf1d_max_abs / tf1d_sum_in_range
// (restart 57 + 58). Without this, a single NaN slot poisons every
// subsequent prefix sum, breaking partial-accumulator semantics.
fn tf1d_running_sum(start: i32, n: i32) -> i32 {
    let s = t1d_new(n);
    if n <= 0 { s }
    else { if t1d_slice_ok(start, n) == 0 { s }
    else {
        let mut acc: f32 = 0.0_f32;
        let mut i: i32 = 0;
        while i < n {
            let v = __f32_from_bits(__arena_get(start + i));
            if v == v { acc = acc + v; };
            __arena_set(s + i, __bits_of_f32(acc));
            i = i + 1;
        }
        s
    }}
}

// tf1d_negate(start, dst, n): write -x[i] to dst[i] for all i. Out-of-
// place; caller pre-allocates dst with t1d_new(n) (or shares with x for
// in-place).
fn tf1d_negate(x_start: i32, dst: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(dst, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    while i < n {
        let v = __f32_from_bits(__arena_get(x_start + i));
        __arena_set(dst + i, __bits_of_f32(0.0_f32 - v));
        i = i + 1;
    }
    0
    }}}
}

// tf1d_scale_inplace(start, n, scalar): multiply every element by
// scalar in place. Mirror of vec_offset_inplace for f32.
fn tf1d_scale_inplace(start: i32, n: i32, scalar: f32) -> i32 {
    if n <= 0 { start }
    else { if t1d_slice_ok(start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    while i < n {
        let v = __f32_from_bits(__arena_get(start + i));
        __arena_set(start + i, __bits_of_f32(v * scalar));
        i = i + 1;
    }
    start
    }}
}

// tf2d_add(a, b, c, rows, cols): elementwise 2D add c = a + b. All
// three matrices share row-major layout with `cols` columns.
fn tf2d_add(a: i32, b: i32, c: i32, rows: i32, cols: i32) -> i32 {
    let n = t2d_len(rows, cols);
    let st_a = t2d_shape_status(a, rows, cols);
    let st_b = t2d_shape_status(b, rows, cols);
    let st_c = t2d_shape_status(c, rows, cols);
    if st_a != 0 { st_a }
    else { if st_b != 0 { st_b }
    else { if st_c != 0 { st_c } else {
    let mut i: i32 = 0;
    while i < n {
        let av = __f32_from_bits(__arena_get(a + i));
        let bv = __f32_from_bits(__arena_get(b + i));
        __arena_set(c + i, __bits_of_f32(av + bv));
        i = i + 1;
    }
    0
    }}}
}

// tf2d_scale_inplace(start, rows, cols, scalar): multiply every element
// of the 2D matrix in place by scalar.
fn tf2d_scale_inplace(start: i32, rows: i32, cols: i32, scalar: f32) -> i32 {
    let n = t2d_len(rows, cols);
    let st = t2d_shape_status(start, rows, cols);
    if st != 0 { st } else {
    let mut i: i32 = 0;
    while i < n {
        let v = __f32_from_bits(__arena_get(start + i));
        __arena_set(start + i, __bits_of_f32(v * scalar));
        i = i + 1;
    }
    0
    }
}

// tf1d_max_abs(start, n): @pure. Max of |x[i]| (Linf norm). Returns
// 0.0 for empty input.
// Restart 58 A3: NaN-skip. NaN compares false in both directions so
// would silently skip via `av > best` anyway, but the abs step itself
// could produce a spurious NaN; explicit guard documents intent and
// matches the tf1d_sum / tf1d_dot / tf1d_l1_norm pattern.
@pure
fn tf1d_max_abs(start: i32, n: i32) -> f32 {
    if n <= 0 { 0.0_f32 }
    else { if t1d_slice_ok(start, n) == 0 { 0.0_f32 }
    else {
    let mut i: i32 = 0;
    let mut best: f32 = 0.0_f32;
    while i < n {
        let v = __f32_from_bits(__arena_get(start + i));
        if v == v {
        let av = if v < 0.0_f32 { 0.0_f32 - v } else { v };
        if av > best { best = av; };
        };
        i = i + 1;
    }
    best
    }}
}

// tf1d_axpby(x, y, a, b, n): in-place compute y[i] = a*x[i] + b*y[i].
// BLAS-style level-1 op. Caller mutates y in place.
fn tf1d_axpby(x_start: i32, y_start: i32, a: f32, b: f32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(y_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    while i < n {
        let xv = __f32_from_bits(__arena_get(x_start + i));
        let yv = __f32_from_bits(__arena_get(y_start + i));
        __arena_set(y_start + i, __bits_of_f32(a * xv + b * yv));
        i = i + 1;
    }
    0
    }}}
}

// tf2d_sub(a, b, c, rows, cols): elementwise 2D subtract c = a - b.
fn tf2d_sub(a: i32, b: i32, c: i32, rows: i32, cols: i32) -> i32 {
    let n = t2d_len(rows, cols);
    let st_a = t2d_shape_status(a, rows, cols);
    let st_b = t2d_shape_status(b, rows, cols);
    let st_c = t2d_shape_status(c, rows, cols);
    if st_a != 0 { st_a }
    else { if st_b != 0 { st_b }
    else { if st_c != 0 { st_c } else {
    let mut i: i32 = 0;
    while i < n {
        let av = __f32_from_bits(__arena_get(a + i));
        let bv = __f32_from_bits(__arena_get(b + i));
        __arena_set(c + i, __bits_of_f32(av - bv));
        i = i + 1;
    }
    0
    }}}
}

// tf2d_mul(a, b, c, rows, cols): elementwise 2D Hadamard (NOT matmul).
fn tf2d_mul(a: i32, b: i32, c: i32, rows: i32, cols: i32) -> i32 {
    let n = t2d_len(rows, cols);
    let st_a = t2d_shape_status(a, rows, cols);
    let st_b = t2d_shape_status(b, rows, cols);
    let st_c = t2d_shape_status(c, rows, cols);
    if st_a != 0 { st_a }
    else { if st_b != 0 { st_b }
    else { if st_c != 0 { st_c } else {
    let mut i: i32 = 0;
    while i < n {
        let av = __f32_from_bits(__arena_get(a + i));
        let bv = __f32_from_bits(__arena_get(b + i));
        __arena_set(c + i, __bits_of_f32(av * bv));
        i = i + 1;
    }
    0
    }}}
}

// tf1d_argmax_in_range(start, n, lo, hi): @pure. Index of largest f32
// in x[lo..hi). Returns -1 if bounds are invalid.
// Restart 58 A7 (Increment 77 catch-up sweep): NaN-at-lo robustness.
// Same idiom as tf1d_argmax — adopt the first non-NaN slot instead of
// bare-init at index lo.
@pure
fn tf1d_argmax_in_range(start: i32, n: i32, lo: i32, hi: i32) -> i32 {
    if n < 0 { 0 - 1 }
    else { if lo < 0 { 0 - 1 }
    else { if hi > n { 0 - 1 }
    else { if hi <= lo { 0 - 1 }
    else { if t1d_slice_ok(start, hi) == 0 { 0 - 1 }
    else {
        let mut best_idx: i32 = lo;
        let mut best: f32 = 0.0_f32;
        let mut seen: i32 = 0;
        let mut i: i32 = lo;
        while i < hi {
            let v = __f32_from_bits(__arena_get(start + i));
            if v == v {
                if seen == 0 { best = v; best_idx = i; seen = 1; }
                else { if v > best { best = v; best_idx = i; }; };
            };
            i = i + 1;
        }
        best_idx
    }}}}}
}

// tf1d_sum_in_range(start, n, lo, hi): @pure. Sum of x[lo..hi). 0.0 if
// bounds are invalid. Useful for partial accumulators.
// Restart 58 A1 (missed carry-forward sibling — landed in Increment 77
// catch-up sweep): NaN-skip via `if v == v`. Same family as tf1d_sum /
// tf1d_dot / tf1d_l1_norm / tf1d_max_abs.
@pure
fn tf1d_sum_in_range(start: i32, n: i32, lo: i32, hi: i32) -> f32 {
    if n < 0 { 0.0_f32 }
    else { if lo < 0 { 0.0_f32 }
    else { if hi > n { 0.0_f32 }
    else { if hi <= lo { 0.0_f32 }
    else { if t1d_slice_ok(start, hi) == 0 { 0.0_f32 }
    else {
    let mut i: i32 = lo;
    let mut total: f32 = 0.0_f32;
    while i < hi {
        let v = __f32_from_bits(__arena_get(start + i));
        if v == v { total = total + v; };
        i = i + 1;
    }
    total
    }}}}}
}

// tf2d_row_sum(start, rows, cols, dst): for each row r, write
// sum(M[r, *]) to dst[r]. dst pre-allocated by caller (size = rows).
// Restart 58 A5 (Increment 77 catch-up sweep): NaN-skip per-cell so a
// single NaN slot poisons only that row's partial sum, not every output.
// Matches the tf1d_sum NaN-skip pattern.
fn tf2d_row_sum(start: i32, rows: i32, cols: i32, dst: i32) -> i32 {
    if rows <= 0 { 0 }
    else { if cols <= 0 { 0 }
    else { if t2d_len(rows, cols) == 0 { t2d_error() }
    else { if t2d_shape_ok(start, rows, cols) == 0 { t2d_error() }
    else { if t1d_slice_ok(dst, rows) == 0 { t2d_error() }
    else {
    let mut r: i32 = 0;
    while r < rows {
        let mut c: i32 = 0;
        let mut acc: f32 = 0.0_f32;
        while c < cols {
            let v = __f32_from_bits(__arena_get(start + r * cols + c));
            if v == v { acc = acc + v; };
            c = c + 1;
        }
        __arena_set(dst + r, __bits_of_f32(acc));
        r = r + 1;
    }
    0
    }}}}}
}

// tf2d_col_sum(start, rows, cols, dst): for each col c, write
// sum(M[*, c]) to dst[c]. dst pre-allocated by caller (size = cols).
fn tf2d_col_sum(start: i32, rows: i32, cols: i32, dst: i32) -> i32 {
    if rows <= 0 { 0 }
    else { if cols <= 0 { 0 }
    else { if t2d_len(rows, cols) == 0 { t2d_error() }
    else { if t2d_shape_ok(start, rows, cols) == 0 { t2d_error() }
    else { if t1d_slice_ok(dst, cols) == 0 { t2d_error() }
    else {
    let mut c: i32 = 0;
    while c < cols {
        let mut r: i32 = 0;
        let mut acc: f32 = 0.0_f32;
        while r < rows {
            let v = __f32_from_bits(__arena_get(start + r * cols + c));
            if v == v { acc = acc + v; };
            r = r + 1;
        }
        __arena_set(dst + c, __bits_of_f32(acc));
        c = c + 1;
    }
    0
    }}}}}
}

// tf1d_arange(start_val, n): allocate a new f32 vec of length n with
// values [start_val, start_val + 1.0, ..., start_val + (n-1)]. Returns
// the new vec start. Useful for index-vec pairs and test setup.
fn tf1d_arange(start_val: f32, n: i32) -> i32 {
    let s = t1d_new(n);
    let mut i: i32 = 0;
    let mut v: f32 = start_val;
    while i < n {
        __arena_set(s + i, __bits_of_f32(v));
        v = v + 1.0_f32;
        i = i + 1;
    }
    s
}

// tf1d_dot_with_offset(a, a_off, b, b_off, n): dot product over slices
// a[a_off..a_off+n] and b[b_off..b_off+n]. @pure.
// Restart 58 A3 (Increment 77 catch-up sweep): NaN-skip on per-element
// product. Mirrors the tf1d_dot fix.
@pure
fn tf1d_dot_with_offset(a: i32, a_off: i32, b: i32, b_off: i32, n: i32) -> f32 {
    if a_off < 0 { 0.0_f32 }
    else { if b_off < 0 { 0.0_f32 }
    else { if n <= 0 { 0.0_f32 }
    else { if t1d_range_ok(a, a_off, n) == 0 { 0.0_f32 }
    else { if t1d_range_ok(b, b_off, n) == 0 { 0.0_f32 }
    else {
    let mut i: i32 = 0;
    let mut total: f32 = 0.0_f32;
    while i < n {
        let av = __f32_from_bits(__arena_get(a + a_off + i));
        let bv = __f32_from_bits(__arena_get(b + b_off + i));
        let prod = av * bv;
        if prod == prod { total = total + prod; };
        i = i + 1;
    }
    total
    }}}}}
}

// tf2d_diag(m, rows, cols, dst): for a square matrix M, extract the diagonal
// into dst. Rectangular or empty shapes are no-ops.
fn tf2d_diag(m: i32, rows: i32, cols: i32, dst: i32) -> i32 {
    if rows <= 0 { 0 }
    else { if cols <= 0 { 0 }
    else { if rows != cols { 0 }
    else { if t2d_len(rows, cols) == 0 { t2d_error() }
    else { if t2d_shape_ok(m, rows, cols) == 0 { t2d_error() }
    else { if t1d_slice_ok(dst, rows) == 0 { t2d_error() }
    else {
    let n = rows;
    let mut i: i32 = 0;
    while i < n {
        __arena_set(dst + i, __arena_get(m + i * n + i));
        i = i + 1;
    }
    0
    }}}}}}
}

// tf2d_eye(n): allocate a new n*n identity matrix (1.0 on diagonal,
// 0.0 elsewhere). Returns the new start index.
fn tf2d_eye(n: i32) -> i32 {
    let s = t2d_new(n, n);
    if n <= 0 { s }
    else { if t2d_len(n, n) == 0 { s }
    else {
        let one_bits = __bits_of_f32(1.0_f32);
        let mut r: i32 = 0;
        while r < n {
            let mut c: i32 = 0;
            while c < n {
                if r == c { __arena_set(s + r * n + c, one_bits); };
                c = c + 1;
            }
            r = r + 1;
        }
        s
    }}
}

// tf2d_trace(m, rows, cols): @pure. Sum of diagonal elements of a square
// matrix. Rectangular or empty shapes return 0.0.
// Restart 58 A5 (Increment 77 catch-up sweep): NaN-skip on diagonal
// element accumulation. Mirrors tf1d_sum / tf2d_row_sum.
@pure
fn tf2d_trace(m: i32, rows: i32, cols: i32) -> f32 {
    if rows <= 0 { 0.0_f32 }
    else { if cols <= 0 { 0.0_f32 }
    else { if rows != cols { 0.0_f32 }
    else { if t2d_len(rows, cols) == 0 { 0.0_f32 }
    else { if t2d_shape_ok(m, rows, cols) == 0 { 0.0_f32 }
    else {
    let n = rows;
    let mut i: i32 = 0;
    let mut total: f32 = 0.0_f32;
    while i < n {
        let v = __f32_from_bits(__arena_get(m + i * n + i));
        if v == v { total = total + v; };
        i = i + 1;
    }
    total
    }}}}}
}

// tf1d_lerp(a, b, t, dst, n): linear interpolation. dst[i] = a[i] +
// t * (b[i] - a[i]). Useful for numerical interpolation between two
// vectors. dst pre-allocated (n slots).
fn tf1d_lerp(a: i32, b: i32, t: f32, dst: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(a, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(b, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(dst, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    while i < n {
        let av = __f32_from_bits(__arena_get(a + i));
        let bv = __f32_from_bits(__arena_get(b + i));
        let v = av + t * (bv - av);
        __arena_set(dst + i, __bits_of_f32(v));
        i = i + 1;
    }
    0
    }}}}
}

// tf2d_norm_frobenius_sq(start, rows, cols): @pure. Squared Frobenius
// norm — sum of squares of all elements. Mirror of tf1d_l2_norm_sq for
// 2D matrices.
@pure
fn tf2d_norm_frobenius_sq(start: i32, rows: i32, cols: i32) -> f32 {
    let n = t2d_len(rows, cols);
    if t2d_shape_ok(start, rows, cols) == 0 { 0.0_f32 }
    else { tf1d_l2_norm_sq(start, n) }
}

// tf2d_zeros(rows, cols): allocate a new rows*cols matrix filled with 0.0_f32.
fn tf2d_zeros(rows: i32, cols: i32) -> i32 {
    t2d_new(rows, cols)
}

// tf2d_ones(rows, cols): allocate a new rows*cols matrix filled with 1.0_f32.
fn tf2d_ones(rows: i32, cols: i32) -> i32 {
    let n = t2d_len(rows, cols);
    let s = t2d_new(rows, cols);
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
    let n = t2d_len(rows, cols);
    if t2d_shape_ok(start, rows, cols) == 0 { 0.0_f32 }
    else { tf1d_max_abs(start, n) }
}

// tf1d_count_above(start, n, threshold): @pure. Count elements > threshold.
@pure
fn tf1d_count_above(start: i32, n: i32, threshold: f32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(start, n) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < n {
        let v = __f32_from_bits(__arena_get(start + i));
        if v > threshold { total = total + 1; }
        i = i + 1;
    }
    total
    }}
}

// tf1d_count_below(start, n, threshold): @pure. Companion.
@pure
fn tf1d_count_below(start: i32, n: i32, threshold: f32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(start, n) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < n {
        let v = __f32_from_bits(__arena_get(start + i));
        if v < threshold { total = total + 1; }
        i = i + 1;
    }
    total
    }}
}

// tf1d_count_eq_zero(start, n): @pure. Count elements equal to 0.0_f32.
@pure
fn tf1d_count_eq_zero(start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(start, n) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    let zero_bits = 0;
    while i < n {
        if __arena_get(start + i) == zero_bits { total = total + 1; }
        i = i + 1;
    }
    total
    }}
}

// tf1d_is_empty(start, n): @pure. 1 if n <= 0.
@pure
fn tf1d_is_empty(start: i32, n: i32) -> i32 {
    if n <= 0 { 1 } else { 0 }
}

// ti1d_count_above(start, n, threshold): @pure. Count int elements > threshold.
@pure
fn ti1d_count_above(start: i32, n: i32, threshold: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(start, n) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < n {
        if __arena_get(start + i) > threshold { total = total + 1; }
        i = i + 1;
    }
    total
    }}
}

// ti1d_count_below(start, n, threshold): @pure. Companion.
@pure
fn ti1d_count_below(start: i32, n: i32, threshold: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(start, n) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < n {
        if __arena_get(start + i) < threshold { total = total + 1; }
        i = i + 1;
    }
    total
    }}
}

// ti1d_max_abs(start, n): @pure. Max of |x[i]| for ints. 0 if empty.
// Restart 57 A3: INT32_MIN special-case. 0 - INT32_MIN wraps back to
// INT32_MIN; the `av > best` test (best starts at 0) is false, so the
// function silently returns 0 instead of the correct INT32_MAX
// saturation. Same family as restart 51 A5 (vec_negate_inplace).
@pure
fn ti1d_max_abs(start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(start, n) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut best: i32 = 0;
    while i < n {
        let v = __arena_get(start + i);
        let av = if v == ((0 - 2147483647) - 1) { 2147483647 }
                 else { if v < 0 { 0 - v } else { v } };
        if av > best { best = av; }
        i = i + 1;
    }
    best
    }}
}

// ti1d_is_empty(start, n): @pure. 1 if n <= 0.
@pure
fn ti1d_is_empty(start: i32, n: i32) -> i32 {
    if n <= 0 { 1 } else { 0 }
}

// tf1d_first(start, n): @pure. f32 v[0]. 0.0_f32 if empty.
@pure
fn tf1d_first(start: i32, n: i32) -> f32 {
    if n <= 0 { 0.0_f32 }
    else { if t1d_slice_ok(start, 1) == 0 { 0.0_f32 }
    else { __f32_from_bits(__arena_get(start)) } }
}

// tf1d_last(start, n): @pure. f32 v[n-1]. 0.0_f32 if empty.
@pure
fn tf1d_last(start: i32, n: i32) -> f32 {
    if n <= 0 { 0.0_f32 }
    else { if t1d_slice_ok(start, n) == 0 { 0.0_f32 }
    else { __f32_from_bits(__arena_get(start + n - 1)) } }
}

// ti1d_first(start, n): @pure. v[0] or 0 if empty.
@pure
fn ti1d_first(start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(start, 1) == 0 { 0 }
    else { __arena_get(start) } }
}

// ti1d_last(start, n): @pure. v[n-1] or 0 if empty.
@pure
fn ti1d_last(start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(start, n) == 0 { 0 }
    else { __arena_get(start + n - 1) } }
}

// ti1d_count_eq(start, n, target): @pure. Count of v[i] == target.
@pure
fn ti1d_count_eq(start: i32, n: i32, target: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(start, n) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < n {
        if __arena_get(start + i) == target { total = total + 1; }
        i = i + 1;
    }
    total
    }}
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
    let s = t1d_new(n);
    if n <= 0 { s }
    else { if t1d_slice_ok(start, n) == 0 { s }
    else {
    let mut i: i32 = 0;
    while i < n {
        __arena_set(s + i, __arena_get(start + i));
        i = i + 1;
    }
    s
    }}
}
