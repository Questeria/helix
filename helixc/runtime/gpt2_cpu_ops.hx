// gpt2_cpu_ops.hx -- GPT-2 124M CPU FORWARD ops, PURE HELIX, compiled by the
// self-hosted kovc (rebuildable from the raw seed). NO ptxas / NO GPU boundary.
//
// This is the op-dispatch ELF the C harness (cpu_host.c) drives tile-by-tile.
// ALL ARITHMETIC lives here (the trust claim); the C harness only moves bytes
// (mmap weights, stage tiles, gather embeddings, head pack/scatter, GEMM N-tiling).
//
// Staging protocol (the driver_k1input.hx pattern):
//   - The harness writes ONE input file /tmp/gpc/in.bin = a 6-i32 LE header
//     [op, d0, d1, d2, d3, d4] immediately followed by the input f32 tile(s) as
//     little-endian raw bytes. read_file_to_arena packs 1 byte/i32-slot, so we
//     reassemble each LE f32 from its 4 bytes (b0 + b1*256 + b2*65536 + b3*2^24,
//     i32 wraparound reproduces the exact 2's-complement bit pattern) then
//     __f32_from_bits. The op computes, then writes the output f32 tile to
//     /tmp/gpc/out.bin as LE bytes (write_file_to_arena emits the low byte/slot).
//   - Tiles are kept < the 1 MB read buffer (256K floats) and the working set
//     under helix_arena_cap()=6,291,456 slots; the harness tiles big GEMMs.
//
// Opcodes (op = header[0]):
//   1 = LAYERNORM_AFFINE : in = x[rows*cols] ++ gamma[cols] ++ beta[cols]
//       header [1, rows, cols, 0,0,0] ; out = y[rows*cols] (biased/population var, eps=1e-5)
//   2 = MATMUL           : in = A[M*K] ++ B[K*N]
//       header [2, M, K, N, 0,0] ; out = C[M*N] = A @ B (row-major)
//   3 = MATMUL_BIAS      : in = A[M*K] ++ B[K*N] ++ bias[N]
//       header [3, M, K, N, 0,0] ; out = C[M*N] = A @ B + bias (row broadcast)
//   4 = GELU             : in = x[n]                ; header [4, n, 0,0,0,0] ; out = gelu_new(x)[n]
//   5 = ADD              : in = a[n] ++ b[n]        ; header [5, n, 0,0,0,0] ; out = a+b [n]
//   6 = SOFTMAX_CAUSAL   : in = scores[rows*cols]   ; header [6, rows, cols, 0,0,0]
//       out = causal_softmax_row(scores * 0.125)[rows*cols]: for query row i reduce over
//       keys [0..i], write 0 for keys > i; scores pre-scaled by 1/sqrt(64)=0.125.
//
// License: Apache 2.0.

// ---- f32 bit helpers (the byte<->float bridge over read/write_file_to_arena) ----

// Reassemble the idx-th little-endian f32 from a byte-packed arena region.
@pure fn rd_f32(base: i32, idx: i32) -> f32 {
    let p = base + idx * 4;
    let b0 = __arena_get(p);
    let b1 = __arena_get(p + 1);
    let b2 = __arena_get(p + 2);
    let b3 = __arena_get(p + 3);
    // i32 multiply/add wrap mod 2^32 -> the exact 2's-complement bit pattern.
    let bits = b0 + b1 * 256 + b2 * 65536 + b3 * 16777216;
    __f32_from_bits(bits)
}

// Reassemble the idx-th little-endian i32 (header field) from a byte-packed region.
@pure fn rd_i32(base: i32, idx: i32) -> i32 {
    let p = base + idx * 4;
    let b0 = __arena_get(p);
    let b1 = __arena_get(p + 1);
    let b2 = __arena_get(p + 2);
    let b3 = __arena_get(p + 3);
    b0 + b1 * 256 + b2 * 65536 + b3 * 16777216
}

// Append a f32 as 4 little-endian bytes to the arena tail (one byte per slot).
fn push_f32_le(v: f32) -> i32 {
    let bits = __bits_of_f32(v);
    // (x % 256 + 256) % 256 gives the unsigned low byte for negative i32 too;
    // subtract each byte before /256 so truncated division yields the right next byte.
    let c0 = (bits % 256 + 256) % 256;
    let r1 = (bits - c0) / 256;
    let c1 = (r1 % 256 + 256) % 256;
    let r2 = (r1 - c1) / 256;
    let c2 = (r2 % 256 + 256) % 256;
    let r3 = (r2 - c2) / 256;
    let c3 = (r3 % 256 + 256) % 256;
    __arena_push(c0);
    __arena_push(c1);
    __arena_push(c2);
    __arena_push(c3);
    0
}

// ---- transcendentals (faithful copies of stdlib transcendentals.hx) ----

@pure fn cpu_exp_taylor(r: f32) -> f32 {
    let x2 = r * r;
    let x3 = x2 * r;
    let x4 = x2 * x2;
    let x5 = x4 * r;
    let x6 = x3 * x3;
    let x7 = x6 * r;
    1.0_f32 + r
        + x2 * 0.5_f32
        + x3 * 0.16666667_f32
        + x4 * 0.04166667_f32
        + x5 * 0.00833333_f32
        + x6 * 0.00138889_f32
        + x7 * 0.00019841_f32
}

@pure fn cpu_exp(x: f32) -> f32 {
    let z = x * 1.44269504_f32 + 0.5_f32;
    let k_trunc = z as i32;
    let k = if z >= 0.0_f32 { k_trunc }
            else { if (k_trunc as f32) > z { k_trunc - 1 } else { k_trunc } };
    let r = x - (k as f32) * 0.69314718_f32;
    let exp_r = cpu_exp_taylor(r);
    let kc = if k > 48 { 48 }
             else { if k < (0 - 48) { 0 - 48 } else { k } };
    let mut scale: f32 = 1.0_f32;
    if kc >= 0 {
        let mut i: i32 = 0;
        while i < kc { scale = scale * 2.0_f32; i = i + 1; }
    } else {
        let neg_kc = 0 - kc;
        let mut i: i32 = 0;
        while i < neg_kc { scale = scale * 0.5_f32; i = i + 1; }
    }
    scale * exp_r
}

@pure fn cpu_tanh(x: f32) -> f32 {
    if x > 20.0_f32 { 1.0_f32 }
    else { if x < 0.0_f32 - 20.0_f32 { 0.0_f32 - 1.0_f32 }
           else {
               let e2 = cpu_exp(2.0_f32 * x);
               (e2 - 1.0_f32) / (e2 + 1.0_f32)
           }
    }
}

// gelu_new (HF tanh GELU) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715 x^3))).
// tanh saturates at +/-20 so GPT-2-scale c_fc activations (~+/-12 -> inner ~70)
// stay finite (the same overflow-safety the GPU path's gpu_gelu_stable needs).
@pure fn cpu_gelu(x: f32) -> f32 {
    let x3 = x * x * x;
    let inner = 0.7978846_f32 * (x + 0.044715_f32 * x3);
    0.5_f32 * x * (1.0_f32 + cpu_tanh(inner))
}

// ---- the ops (operate on byte-packed input region in_base; emit f32 LE) ----

// LayerNorm affine over each row: y = (x-mean)/sqrt(var+eps)*gamma + beta.
// Variance is biased/population (divide by cols), eps=1e-5 -- matches the oracle
// (numpy x.var() default) and stdlib layer_norm_f32. Fail-closed on denom<=0/NaN.
fn op_layernorm(in_base: i32, rows: i32, cols: i32) -> i32 {
    let x_base = in_base + 24;                  // 6 i32 header = 24 bytes
    let g_off = rows * cols;                     // gamma starts after x
    let b_off = rows * cols + cols;              // beta starts after gamma
    let eps = 0.00001_f32;
    let mut r: i32 = 0;
    while r < rows {
        let row = r * cols;
        // mean
        let mut s: f32 = 0.0_f32;
        let mut i: i32 = 0;
        while i < cols { s = s + rd_f32(x_base, row + i); i = i + 1; }
        let mean = s / (cols as f32);
        // population variance
        let mut v: f32 = 0.0_f32;
        let mut j: i32 = 0;
        while j < cols {
            let d = rd_f32(x_base, row + j) - mean;
            v = v + d * d;
            j = j + 1;
        }
        v = v / (cols as f32);
        let denom = __fsqrt(v + eps);
        if (denom <= 0.0_f32) || (denom != denom) {
            let mut k: i32 = 0;
            while k < cols { push_f32_le(0.0_f32); k = k + 1; }
        } else {
            let inv = 1.0_f32 / denom;
            let mut k2: i32 = 0;
            while k2 < cols {
                let xn = (rd_f32(x_base, row + k2) - mean) * inv;
                let g = rd_f32(x_base, g_off + k2);
                let b = rd_f32(x_base, b_off + k2);
                push_f32_le(xn * g + b);
                k2 = k2 + 1;
            }
        };
        r = r + 1;
    }
    0
}

// C[M,N] = A[M,K] @ B[K,N], optional + bias[N] broadcast over rows.
// NaN-skip per product (matches stdlib tf2d_matmul). Reads A, B (and bias) from
// the byte-packed input; emits C as f32 LE in row-major order.
fn op_matmul(in_base: i32, m: i32, k: i32, n: i32, has_bias: i32) -> i32 {
    let a_base = in_base + 24;
    let b_off = m * k;                           // B starts after A
    let bias_off = m * k + k * n;                // bias starts after B
    let mut r: i32 = 0;
    while r < m {
        let arow = r * k;
        let mut c: i32 = 0;
        while c < n {
            let mut acc: f32 = 0.0_f32;
            let mut t: i32 = 0;
            while t < k {
                let av = rd_f32(a_base, arow + t);
                let bv = rd_f32(a_base, b_off + t * n + c);
                let prod = av * bv;
                if prod == prod { acc = acc + prod; };
                t = t + 1;
            }
            if has_bias == 1 {
                let bz = rd_f32(a_base, bias_off + c);
                acc = acc + bz;
            };
            push_f32_le(acc);
            c = c + 1;
        }
        r = r + 1;
    }
    0
}

// GELU over n elements.
fn op_gelu(in_base: i32, n: i32) -> i32 {
    let x_base = in_base + 24;
    let mut i: i32 = 0;
    while i < n {
        push_f32_le(cpu_gelu(rd_f32(x_base, i)));
        i = i + 1;
    }
    0
}

// Elementwise add a[n] + b[n] (residual).
fn op_add(in_base: i32, n: i32) -> i32 {
    let a_base = in_base + 24;
    let b_off = n;
    let mut i: i32 = 0;
    while i < n {
        let av = rd_f32(a_base, i);
        let bv = rd_f32(a_base, b_off + i);
        push_f32_le(av + bv);
        i = i + 1;
    }
    0
}

// Causal row softmax with the 1/sqrt(64)=0.125 score scale folded in.
// For query row i, the softmax reduces over keys [0..i] only; keys > i get 0
// (numerically identical to a -inf add, no -inf literal). Max-subtract for
// stability; fail-closed (uniform over the valid prefix) on sum<=0 or NaN.
fn op_softmax_causal(in_base: i32, rows: i32, cols: i32) -> i32 {
    let s_base = in_base + 24;
    let scale = 0.125_f32;
    let mut i: i32 = 0;
    while i < rows {
        let row = i * cols;
        let valid = i + 1;                        // keys 0..i inclusive
        // max over the valid prefix (scaled scores)
        let mut mx: f32 = rd_f32(s_base, row) * scale;
        let mut j: i32 = 1;
        while j < valid {
            let sv = rd_f32(s_base, row + j) * scale;
            if sv > mx { mx = sv; };
            j = j + 1;
        }
        // exp-sum over the valid prefix
        let mut sum_e: f32 = 0.0_f32;
        let mut j2: i32 = 0;
        while j2 < valid {
            let e = cpu_exp(rd_f32(s_base, row + j2) * scale - mx);
            sum_e = sum_e + e;
            j2 = j2 + 1;
        }
        // emit normalized probs for keys 0..i, then 0 for keys > i
        if (sum_e <= 0.0_f32) || (sum_e != sum_e) {
            let inv_v = if valid > 0 { 1.0_f32 / (valid as f32) } else { 0.0_f32 };
            let mut c: i32 = 0;
            while c < cols {
                if c < valid { push_f32_le(inv_v); } else { push_f32_le(0.0_f32); };
                c = c + 1;
            }
        } else {
            let inv = 1.0_f32 / sum_e;
            let mut c2: i32 = 0;
            while c2 < cols {
                if c2 < valid {
                    let e = cpu_exp(rd_f32(s_base, row + c2) * scale - mx);
                    push_f32_le(e * inv);
                } else {
                    push_f32_le(0.0_f32);
                };
                c2 = c2 + 1;
            }
        };
        i = i + 1;
    }
    0
}

fn main() -> i32 {
    let in_base = __arena_len();
    let nbytes = read_file_to_arena("/tmp/gpc/in.bin");
    if nbytes < 24 { return 1; }
    let op = rd_i32(in_base, 0);
    let d0 = rd_i32(in_base, 1);
    let d1 = rd_i32(in_base, 2);
    let d2 = rd_i32(in_base, 3);
    let out_base = __arena_len();
    if op == 1 {
        op_layernorm(in_base, d0, d1);
    } else { if op == 2 {
        op_matmul(in_base, d0, d1, d2, 0);
    } else { if op == 3 {
        op_matmul(in_base, d0, d1, d2, 1);
    } else { if op == 4 {
        op_gelu(in_base, d0);
    } else { if op == 5 {
        op_add(in_base, d0);
    } else { if op == 6 {
        op_softmax_causal(in_base, d0, d1);
    } else {
        return 2;
    }}}}}};
    let out_len = __arena_len() - out_base;
    write_file_to_arena("/tmp/gpc/out.bin", out_base, out_len)
}
