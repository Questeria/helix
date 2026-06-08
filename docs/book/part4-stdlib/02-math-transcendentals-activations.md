# Math, transcendentals & activations

*What this chapter covers:* the **numeric surface** of Helix's standard library — the f32/f64
floating-point math the compiler `kovc` lowers to SSE instructions directly (the real math
*builtins*), and the Taylor-series / Newton-iteration transcendentals, activation functions, loss
functions, PRNG, and optimizer-step helpers written *in Helix* on top of them in
[`helixc/stdlib/transcendentals.hx`](../../../helixc/stdlib/transcendentals.hx). For each, this
chapter quotes the *real* source verbatim as Fragments, cites it, and is scrupulous about the one
boundary that matters: which names are compiler builtins, which are library `.hx` functions, and
exactly what the gate does and does not prove about them.

This chapter sits between the [Standard library overview](01-overview.md) (which drew the
builtins-vs-modules line) and [Tensors, collections & I/O](03-tensors-collections-io.md) (the data
half). It is the *library-level* numerics companion to Part III's
[Autodiff & the AGI-oriented features](../part3-language/05-autodiff-agi-features.md), which owns
the *language-level* `grad` story and the gate-proven `gradient_descent.hx`; this chapter does
**not** re-derive `grad` — it documents the math the differentiable surface is built on. The
authoritative builtin reference is [`docs/HELIX_V1_STDLIB.md`](../../../docs/HELIX_V1_STDLIB.md) and
the spec's §5 ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md)).

> **For AI agents:** the load-bearing distinction in this chapter is **builtin vs library
> function**, and you must keep them apart. The f32/f64 *arithmetic and the small SSE primitive set*
> (`__fsqrt`, `__fabs`, `__i32_to_f32`, `__f32_to_i32`, `__bits_of_f32`/`__f32_from_bits`) are
> compiler builtins `kovc` emits inline. The *transcendentals and activations* (`__exp`, `__sin`,
> `__sigmoid`, `__relu`, `__gelu`, …) are **ordinary `@pure fn`s defined in
> `helixc/stdlib/transcendentals.hx`** — not builtins. The shipping `kovc` only *recognizes their
> names* for the autodiff chain rule (in `helixc/bootstrap/parser.hx`); the function bodies must be
> co-parsed. Verify any name with `grep -n 'fn __exp' helixc/stdlib/transcendentals.hx` and
> `grep -rn '__fsqrt' helixc/bootstrap/kovc.hx`.

---

## 1. The two layers of "math" in Helix

Exactly as the [overview](01-overview.md) split "the standard library" into compiler builtins and
`.hx` modules, the math surface has two layers, and conflating them is the most common error:

1. **The float math builtins.** `kovc` lowers f32/f64 `+ - * /` to SSE instructions, and recognizes
   a small set of single-purpose primitive names that it emits as inline SSE2 — square root,
   absolute value, the int↔float conversions, and the bit-reinterprets. These are not written in
   Helix; they are codegen paths in [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx).
   The spec lists them under *Builtins & intrinsics*
   ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §5).

2. **The transcendental / activation / loss library.**
   [`helixc/stdlib/transcendentals.hx`](../../../helixc/stdlib/transcendentals.hx) is one ordinary
   Helix source file that implements `exp`, `log`, `sin`, `cos`, `sqrt`, the activations (`sigmoid`,
   `relu`, `tanh`, `gelu`, `silu`, `softplus`), loss functions (`mse`, `mae`, `bce`, `huber`), a
   seeded PRNG, and optimizer-step helpers — all in terms of layer 1's float arithmetic. It is
   library code you can read, copy, and call; it is not compiler magic.

The honest one-sentence summary: **layer 1 is the irreducible SSE math the compiler emits; layer 2
is a single `.hx` file of approximations written over it.** The rest of this chapter walks both,
and is explicit at each step about which layer a name belongs to and what the gate proves.

---

## 2. The float math builtins (layer 1)

### 2.1 f32/f64 arithmetic lowers to SSE

Floating-point `+ - * /` on `f32` and `f64` operands lower directly to scalar SSE instructions —
`addss`/`subss`/`mulss`/`divss` for `f32`, and the `…sd` doubles for `f64`. The back end's emitters
are one-liners over a shared SSE-binop encoder:

**Fragment** (the f32 SSE arithmetic emitters; excerpt of
[`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx)):

```helix
fn emit_addss() -> i32 { emit_sse_binop(0x58) }
fn emit_subss() -> i32 { emit_sse_binop(0x5C) }
fn emit_mulss() -> i32 { emit_sse_binop(0x59) }
fn emit_divss() -> i32 { emit_sse_binop(0x5E) }
```

Which instruction a given `a + b` lowers to is decided by the operands' **type tags** — the same
type-tag dispatch the compiler chapters describe (see
[Part V — Front end](../part5-compiler/01-front-end.md) for `ty_ident_to_tag`, and
[Part V — IR & passes](../part5-compiler/02-ir-and-passes.md) for the `AST_ADD` cascade that routes
`f32`→`addss`, `f64`→`addsd`, `i32`→integer add, and traps on a mixed pair). An `f32` value is
stored as its IEEE-754 **bit pattern** in an `i32`-wide slot; this is the same representation the
tensor chapter documents, reinterpreted by the zero-instruction `__bits_of_f32` / `__f32_from_bits`
shims ([Tensors, collections & I/O §2.2](03-tensors-collections-io.md)).

### 2.2 The single-purpose SSE primitives

Beyond arithmetic, `kovc` recognizes a small set of **single-argument** float primitives by name and
emits the corresponding SSE2 instruction inline. The spec enumerates the set
([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §5):

> - **f32/f64 math** (SSE): `__fadd/__fsub/__fmul/__fdiv/__fneg/__fsqrt/__fabs/__fmin/__fmax`,
>   `__i32_to_f32`/`__f32_to_i32`, bit reinterprets; f64 equivalents + `__f64_pack`/`__bits_{lo,hi}_f64`
>   [impl; the f32 set is capstone-exercised].

The two most useful — and the ones to reach for instead of the library `__sqrt` / `__abs` when you
want exact hardware semantics — are `__fsqrt` (a true `sqrtss`) and `__fabs`. The back end documents
both at their installation site, including the crucial distinction from the library Newton-iteration
`__sqrt`:

**Fragment** ([`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx); excerpt — the
`__fsqrt` / `__fabs` builtin notes):

```helix
    // Phase 1.10 step 5g: "__fsqrt" (95 95 102 115 113 114 116) — 7
    // chars. Single-arg f32 square root via SSE2 sqrtss xmm0, xmm0.
    // Hardware-direct primitive (vs the Newton-iteration __sqrt in
    // helixc/stdlib/transcendentals.hx). Result is the f32 bit pattern
    // in eax. NaN inputs propagate (sqrtss preserves NaN), negatives
    // produce a quiet NaN.
```

So there are *two* square roots in Helix, and they are different things: `__fsqrt` is a single
`sqrtss` (exact, hardware, NaN-propagating), while `__sqrt` (covered in §3) is a five-iteration
Newton method written in Helix. The `__i32_to_f32` / `__f32_to_i32` pair are likewise real builtins
— `cvtsi2ss` and the truncating `cvttss2si`.

> **For AI agents:** prefer the `__f*` builtins (`__fsqrt`, `__fabs`) over the library `__sqrt` /
> `__abs` when you need exact IEEE-754 hardware behavior — they are one SSE instruction each and
> propagate NaN per the ISA. The library versions trade exactness for self-hostability (`__sqrt` is
> a bounded Newton iteration; see §3.3). Both type their result as `f32` because `kovc`'s
> `is_f32_expr` matches the `__f*` name prefix
> ([`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx)).

> **Residual:** the spec marks the f32/f64 math builtin set `[impl; the f32 set is
> capstone-exercised]` — i.e. the f32 primitives are exercised end-to-end by the transformer
> capstone, but the *full* `__f*` set does not each carry an independent gate exit-code assertion the
> way the corpus programs do. Where you need a pinned float result, anchor on a gate-asserted program
> ([Appendix E §E.2](../appendices/E-example-index.md)) and verify the rest by compiling and running
> it yourself.

---

## 3. The transcendentals library — `transcendentals.hx` (layer 2)

[`helixc/stdlib/transcendentals.hx`](../../../helixc/stdlib/transcendentals.hx) is a single
committed `.hx` file of `@pure` functions. It uses **only** float arithmetic, casts, and the
`0.0 / 0.0` NaN idiom — nothing outside the shipping intrinsic surface — so it is *within the
intrinsic surface* in the sense of [§1.2 of the tensors chapter](03-tensors-collections-io.md): its
code is realistic for the shipping `kovc` when co-parsed with a program, but it carries **no
standing module-level compile-proof in the gate** (the same tier as `iterators.hx`). It is not
design-stage — it does not depend on any absent intrinsic — but no gate row compiles it as a
standalone unit, and **no gate-asserted program calls a transcendental** (the gate's float witness
is `grad` over a polynomial loss; see §6).

> **For AI agents:** `transcendentals.hx` is **within the intrinsic surface** (every `__*` it calls
> is either a real `kovc` builtin from §2 or another function defined in the same file), so it
> compiles when co-parsed — but it is **not** gate-asserted, and **no committed gate row exercises a
> transcendental's exit code.** Do not assert `__exp`/`__sin`/`__sigmoid` "are gate-proven." They are
> readable, runnable library source; pin a number by compiling a caller yourself.

### 3.1 The range note, read precisely

The file header states a range limitation, and it is worth quoting exactly because it is easy to
over-read:

**Fragment** (header of
[`helixc/stdlib/transcendentals.hx`](../../../helixc/stdlib/transcendentals.hx); excerpt):

```helix
// Range: each Taylor approximation is accurate for small |x| (roughly
// |x| < 1.5 for sin/cos, x near 1 for log, |x| < 4 for exp). Production
// code would do range reduction; v0.1 keeps the surface simple.
```

The precise reading: the bare **Taylor kernels** are small-`|x|` only. But the *public* `__exp`,
`__sin`, and `__cos` in this file now **range-reduce** before calling their kernels, so they are
accurate over a far wider span than the header's kernel-level note suggests. `__exp`, for instance,
reduces `x = k·ln2 + r` and scales by `2^k`, and its own comment claims accuracy "for any x in
roughly [-50, 50] (covers all NN logits)." Do not present `__exp`/`__sin`/`__cos` as small-`|x|`
only; do present `__log` (a 5-term series around 1) and the internal `__exp_taylor` that way.

### 3.2 Range-reduced `exp`

`__exp` is the canonical range-reduced function and the one most other code depends on (sigmoid,
tanh, softplus all call it). The reduction and the integer `2^k` scaling, verbatim:

**Fragment** (the public `__exp` from
[`helixc/stdlib/transcendentals.hx`](../../../helixc/stdlib/transcendentals.hx); excerpt — the
range-reduction core):

```helix
@pure fn __exp(x: f32) -> f32 {
    // ... sign/rounding comment elided ...
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
```

The `__sin` / `__cos` pair apply the same idea against `2π` (reducing the angle into one period
before a 4-term series), so they too are accurate well past the header's kernel-level `|x| < 1.5`.
The file also carries `f64` siblings of every function (`__exp_f64`, `__sin_f64`, `__sqrt_f64`, …)
with more series terms for the wider mantissa.

### 3.3 `log` and `sqrt`: the fail-closed numerics

Two functions show the file's calibrated-honesty discipline — the same posture as the trust chain
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md)), applied to library numerics:
prefer a detectable NaN sentinel over a plausible-but-wrong value.

`__log` returns a **NaN** (`0.0 / 0.0`) outside its domain rather than a diverging polynomial,
and the comment records the exact bug that motivated it:

**Fragment** (`__log` from
[`helixc/stdlib/transcendentals.hx`](../../../helixc/stdlib/transcendentals.hx); excerpt):

```helix
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
```

`__log` is genuinely accurate only for `x` near 1 (it is a 5-term series around 1); for general
positive `x` use `__log_stable`, which normalizes the mantissa into `[1/√2, √2]` with an integer
exponent before an `atanh`-style series. `__sqrt` is the Newton method paired with `__fsqrt` from
§2; its domain handling is likewise explicit:

**Fragment** (`__sqrt` from
[`helixc/stdlib/transcendentals.hx`](../../../helixc/stdlib/transcendentals.hx); excerpt):

```helix
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
```

Because the bare `__sqrt` returns `0.0` for *both* `sqrt(0)` and a domain error (`x < 0`), the file
adds `__sqrt_strict`, which returns NaN for `x < 0` while preserving `__sqrt(0) = 0` — the same
ambiguity-sentinel pattern the data modules use (`_strict` companions in
[Tensors, collections & I/O §1.3](03-tensors-collections-io.md)).

---

## 4. Activations

The activation functions are the part of `transcendentals.hx` most relevant to the ML capstone, and
they are built directly on §3's `__exp` and friends. The header for the file labels this block
"Modern activation functions"; the canonical set is `__sigmoid`, `__relu`, `__tanh`, `__silu`,
`__gelu`, and `__softplus` (plus f64 siblings for sigmoid/relu).

`__sigmoid` is the bounded `(0,1)` activation, asymptotically short-circuited for large `|x|` so it
never forms `inf/inf`:

**Fragment** (`__sigmoid` from
[`helixc/stdlib/transcendentals.hx`](../../../helixc/stdlib/transcendentals.hx); excerpt):

```helix
@pure fn __sigmoid(x: f32) -> f32 {
    if x > 30.0 { 1.0 }
    else { if x < 0.0 - 30.0 { 0.0 }
           else { 1.0 / (1.0 + __exp(0.0 - x)) }
    }
}
```

`__relu` is the trivial rectifier, and `__gelu` is the tanh-based Hendrycks–Gimpel approximation used
in modern transformers — note it is built from `__tanh`, which is in turn built from `__exp`:

**Fragment** (`__relu` and `__gelu` from
[`helixc/stdlib/transcendentals.hx`](../../../helixc/stdlib/transcendentals.hx); excerpt):

```helix
@pure fn __relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}
```

```helix
// GELU (Gaussian Error Linear Unit): tanh-based approximation per
// Hendrycks & Gimpel. Used in BERT, GPT-2, Llama, etc.
//   gelu(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
// √(2/π) ≈ 0.7978846
@pure fn __gelu(x: f32) -> f32 {
    let x3 = x * x * x;
    let inner = 0.7978846 * (x + 0.044715 * x3);
    0.5 * x * (1.0 + __tanh(inner))
}
```

`__silu` is `x * sigmoid(x)`; `__softplus` is the numerically-stable `log(1 + exp(·))` with the
large-`|x|` asymptote handled explicitly. Every one composes the §3 primitives — which is exactly why
the autodiff pass only needs analytic chain rules for the leaf transcendentals (`__exp`, `__sin`,
`__sqrt`, `__sigmoid`): the activations differentiate through composition.

> **Note — these are the *CPU* activations.** The capstone's GPU path has its own activation
> intrinsics emitted to PTX (`__gpu_exp`, `__gpu_rsqrt`, …;
> [`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §5), and a
> `gpu_gelu_kernel.hx` example, covered in **[Part VII — GPU Codegen](../part7-gpu/01-ptx-backend.md)**. Do not conflate
> `transcendentals.hx`'s `__gelu` with the GPU GELU kernel; they are different code on different
> targets.

---

## 5. Loss functions, the PRNG, and optimizer helpers

The tail of `transcendentals.hx` rounds out a training-oriented numeric toolkit. All three groups
are ordinary `@pure` library functions (layer 2).

**Loss functions** (per-example; the caller sums across a batch): `__mse` (squared error), `__mae`
(absolute error), `__bce` (binary cross-entropy, clamping `p` away from 0/1 to avoid `log(0)`), and
`__huber` (quadratic-then-linear). The simplest is the canonical squared error:

**Fragment** (`__mse` from
[`helixc/stdlib/transcendentals.hx`](../../../helixc/stdlib/transcendentals.hx); excerpt):

```helix
// Mean squared error: (pred - target)^2. Symmetric quadratic.
@pure fn __mse(pred: f32, target: f32) -> f32 {
    let d = pred - target;
    d * d
}
```

**A seeded PRNG.** `__rand_step` is a small LCG (`s_{n+1} = (s_n * 25173 + 13849) mod 32768`) with
`__rand_float` / `__rand_uniform` mappers for weight init and dropout masks. The header is candid
that it is *not* cryptographic and has a short period — fine for reproducible test programs, not for
statistical sampling.

**Optimizer-step helpers.** `__sgd_step` (`w - lr·g`), `__momentum_step_v`, and `__adam_step` are
single-step update primitives. `__adam_step` is the one that shows the fail-closed discipline again —
it clamps a negative second-moment estimate before the `__sqrt`, and fails closed (returns `0.0`) on
a non-positive or NaN denominator:

**Fragment** (`__adam_step` from
[`helixc/stdlib/transcendentals.hx`](../../../helixc/stdlib/transcendentals.hx); excerpt):

```helix
@pure fn __adam_step(m: f32, v: f32, eps: f32) -> f32 {
    let safe_v = if v < 0.0_f32 { 0.0_f32 } else { v };
    let raw_denom = __sqrt(safe_v) + eps;
    // Restart 50 A2: also fail-closed on NaN (raw_denom != raw_denom),
    // matching the in-arena adam_f32_step + softmax_layer idiom.
    if (raw_denom <= 0.0_f32) || (raw_denom != raw_denom) { 0.0_f32 }
    else { m / raw_denom }
}
```

The file also carries integer-typed `__min_i32` / `__max_i32` / `__clamp_i32` / `__abs_i32` /
`__sign_i32` for callers that stay in integer arithmetic (loop bounds, opcode dispatch), where
`__abs_i32` saturates `INT32_MIN` to `INT32_MAX` rather than wrapping.

---

## 6. What the gate proves about this surface

The standing compile-proof has a precise edge here, and the book holds to it.

- **The f32/f64 arithmetic and the SSE primitive set** (`__fsqrt`, `__fabs`, the conversions, the
  bit-reinterprets) are real compiler builtins, exercised end-to-end by the transformer **capstone**
  (`[impl; the f32 set is capstone-exercised]`,
  [`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §5). The capstone's
  honest status — and every GPU residual — is in
  [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md).
- **`transcendentals.hx` is within the intrinsic surface but not gate-asserted as a module**, and
  **no gate corpus row calls a transcendental or activation.** It is real, runnable library source;
  it simply is not one of the programs the gate pins an exit code on.
- **The one gate-proven float-math program is `grad` over a polynomial loss.** The forward-mode
  `grad` rewrite, applied to an `f32` `(x − 3)^2` loss, is compiled and run by the gate
  (`chk "$EX/gradient_descent.hx" 42`, [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh)). That
  program exercises real `f32` SSE arithmetic end-to-end through the self-hosted K2, but it calls **no
  transcendental** — it is the proof that float math *compiles and runs*, not a proof of the
  transcendental approximations. Its full source and a deep walk-through live in
  [Part III — Autodiff & the AGI-oriented features §2](../part3-language/05-autodiff-agi-features.md);
  this chapter cross-references it rather than duplicating it.

Where a transcendental or activation genuinely *is* used in a committed program, it is in
non-gate-asserted demonstration sources — e.g.
[`hbs_sample_loss_fn.hx`](../../../helixc/examples/hbs_sample_loss_fn.hx) calls `__sigmoid` and
`__powi` in a hand-written SGD loop, and [`dogfood_04_xor_relu.hx`](../../../helixc/examples/dogfood_04_xor_relu.hx)
composes reverse-mode `grad_rev` through the stdlib `__relu`. These are real, committed, readable
([Appendix E](../appendices/E-example-index.md)), but they carry no gate-pinned exit code, so this
chapter quotes them only as Fragments. The `hbs_sample_loss_fn.hx` loss is the clearest taste of a
transcendental in use:

**Fragment** (excerpt of [`helixc/examples/hbs_sample_loss_fn.hx`](../../../helixc/examples/hbs_sample_loss_fn.hx);
the loss and its hand-written gradient — not a gate-asserted program):

```helix
@pure @total
fn loss(w: f32) -> f32 {
    let diff = w - 3.0;
    let sq = __powi(diff, 2);
    let sig = __sigmoid(w);
    sq + sig * 0.1
}

@pure @total
fn loss_grad_manual(w: f32) -> f32 {
    // Closed-form gradient written by hand.
    let diff = w - 3.0;
    let lin = 2.0 * diff;
    let sig = __sigmoid(w);
    let one_minus_sig = 1.0 - sig;
    lin + sig * one_minus_sig * 0.1
}
```

> **For AI agents:** if a task needs a *gate-accepted, runnable* float program today, copy the
> `grad`-over-loss shape of [`gradient_descent.hx`](../../../helixc/examples/gradient_descent.hx)
> (exit `42`, [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh)). To *use* a transcendental,
> co-parse `transcendentals.hx` with your program (there is no import — see
> [overview §"How a program uses stdlib code"](01-overview.md)) and verify the result by compiling
> and running it; do not assume a pinned exit code exists for `__exp`/`__sigmoid`/etc.

---

## 7. How this feeds autodiff

The reason the transcendentals carry their odd `__`-prefixed names is that `kovc`'s autodiff pass
**recognizes those exact names** to wire in analytic chain rules. The forward-mode `grad` rewrite,
when it differentiates a loss that calls `__exp` / `__sqrt` / `__sigmoid` / `__sin`, emits the
derivative call (`d exp(u) = exp(u)·du`, `d sin(u) = cos(u)·du`, …) — and that synthesized gradient
**resolves only when `transcendentals.hx` is co-parsed**, because the derivative is itself one of
these library functions. The parser says so directly at the matcher:

**Fragment** ([`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx); excerpt — the
analytic-transcendental note for the autodiff pass):

```helix
// C3b (2026-05-30): matcher + call-builder for the analytic transcendental
// __exp. Unlike relu/abs (whose derivative is a pure conditional), exp's
// derivative IS exp, so the synthesized gradient must CALL __exp — which
// resolves only when the stdlib transcendentals are present in the compiled
// program (the autodiff harness prepends transcendentals.hx for these cases;
```

This is the seam between the two layers: the *compiler* (layer 1, in
[`helixc/bootstrap/`](../../../helixc/bootstrap/)) knows the *names* and the chain rules; the
*function bodies* (layer 2, in `transcendentals.hx`) supply the actual math. The full forward-mode
`grad` story — what it proves, and the honest boundary around the design-stage `autodiff.hx` /
`autodiff_reverse.hx` libraries — is [Part III §2 and §4](../part3-language/05-autodiff-agi-features.md);
this chapter does not repeat it.

---

**Next:** [Tensors, collections & I/O](03-tensors-collections-io.md) — the data half of the standard
library: the arena-backed tensor stack, the growable vector / hash map / byte string (with their
gate-proven inlined corpus shapes), and the CSV / MNIST readers, each grounded against the source and
the gate.
