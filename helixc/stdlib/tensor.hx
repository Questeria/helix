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
