# Tensors, collections & I/O

*What this chapter covers:* the **data and I/O surface** of Helix's standard library — the
arena-backed tensor stack (`tensor.hx`), the growable integer vector (`vec.hx`), the
integer-keyed hash map (`hashmap.hx`), the byte string (`string.hx`), the CSV line/field
iterator (`csv.hx`), the iterator-style vector combinators (`iterators.hx`), and the MNIST
IDX reader (`mnist.hx`). For each module this chapter quotes the *real* core API verbatim as
Fragments, cites the source, and grounds usage in a real, gate-proven program where one
exists. It is a developer reference, so it is organised around two things you must hold
together: **what the API is**, and **what the honest status of that API is** under the
shipping, self-hosting compiler.

> **For AI agents:** read this chapter's §1 before you act on any module name below. The
> load-bearing fact is a boundary. The files in [`helixc/stdlib/`](../../../helixc/stdlib/)
> are real, committed Helix source, but **the gate does not compile them as modules** —
> Helix has no external-module loader yet, so [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh)
> proves the *patterns* by **inlining** them into standalone corpus programs
> (`H1_vec.hx`, `H1_hashmap.hx`, `H2_string.hx`, the generated `vec_arena.hx`). Some modules
> (`csv.hx`, `mnist.hx`, and `string.hx`'s float parser) additionally call intrinsics that are
> **absent from `helixc/bootstrap/`**, so they do not compile under the shipping `kovc` at all.
> Treat those as design-stage library source. Verify any intrinsic with
> `grep -rE '__name' helixc/bootstrap/` before claiming it exists.

This chapter is the proven counterpart's "collections & I/O" half of what
[Part III — "Autodiff & the AGI-oriented features"](../part3-language/05-autodiff-agi-features.md)
set up for the *designed* surface; it uses the same three-tier honesty discipline established
there, and it does not repeat the language-level material from
[Part III — "Types: widths, structs & enums"](../part3-language/02-types.md). The companion
builtin reference is [`docs/HELIX_V1_STDLIB.md`](../../../docs/HELIX_V1_STDLIB.md).

---

## 1. The arena memory model, and the honest status of these modules

### 1.1 One global arena of i32 slots

There is no allocator, no garbage collector, and no `free` in Helix. The entire heap is a
single contiguous **arena** of `i32` slots, BSS-zeroed at load and grown only by appending.
Everything in this chapter — every tensor, vector, hash map, and string — is a *view* over
that one arena, described by an integer **start index** (and usually a length or shape). The
arena builtins are the foundation, and they are the part the gate proves directly:

> | `__arena_len` | `() -> i32` | current cursor (slot count) | [corpus-proven] |
> | `__arena_get` | `(i: i32) -> i32` | read slot `i` | [corpus-proven] |
> | `__arena_set` | `(i: i32, v: i32) -> i32` | write slot `i` | [corpus-proven] |
> | `__arena_push` | `(v: i32) -> i32` | append `v`, return its index | [corpus-proven] |
>
> — [`docs/HELIX_V1_STDLIB.md`](../../../docs/HELIX_V1_STDLIB.md) (a)

Three consequences run through every module below, and the module headers state them
explicitly:

- **Containers carry their handle, not a pointer.** The pervasive idiom is the *carry-pair*:
  a container is `(start, count)` or `(start, cap)` integers that the caller threads through
  every call. [`helixc/stdlib/vec.hx`](../../../helixc/stdlib/vec.hx) names it:

  **Fragment** (header of [`helixc/stdlib/vec.hx`](../../../helixc/stdlib/vec.hx); not a
  complete program):

  ```helix
  // Phase 1.9: a "carry-pair" Vec<i32>. The vec is represented by two
  // integers — start (arena index of slot 0) and count (current length).
  // The caller threads start+count through pushes; the values are stored
  // in the global arena.
  ```

- **Interleaved builds mix.** Because the arena is global and append-only, you must build one
  growable container at a time. Pushing to two growable vectors in alternation interleaves
  their slots. From the same header:

  **Fragment** ([`helixc/stdlib/vec.hx`](../../../helixc/stdlib/vec.hx); excerpt):

  ```helix
  // Convention: build one Vec at a time; interleaved pushes mix slots
  // because the arena is global. For AGI work this is fine since most
  // list-building is sequential.
  ```

- **There is no reclamation.** A "clear" zeroes occupancy flags but does not return slots
  ([`hashmap_clear`](../../../helixc/stdlib/hashmap.hx) is explicit: "Key/value slots are not
  cleared (they're dead until the bucket is reused)"). Once pushed, a slot lives for the
  process. Fixed-capacity containers (the `_checked` Vec, the hash map) bound their footprint;
  growable ones (`vec_push`, the iterator combinators) grow monotonically.

### 1.2 What the gate actually proves — and what it does not

The 98 programs in [`helixc/examples/`](../../../helixc/examples/) and the in-gate feature
corpus are compiled and run by [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step
`[4]` via the freshly self-hosted **K2** compiler, each asserting an exit code. That is the
standing compile-proof, and it is real. But it does **not** cover the `stdlib/*.hx` files in
this chapter directly, for two distinct reasons:

1. **No external-module loader.** The gate compiles each corpus program *standalone*. To prove
   the collection and string *patterns*, it **inlines** a self-contained version into a single
   `.hx` file with an `fn main`. The corpus comments say so in as many words:

   **Fragment** (header of the gate-proven
   [`stage0/helixc-bootstrap/corpus_gen/H1_vec.hx`](../../../stage0/helixc-bootstrap/corpus_gen/H1_vec.hx);
   excerpt):

   ```helix
   // Inlined from stdlib/collections.hx (the gate
   // compiles each corpus program standalone; no external-module loader yet).
   ```

2. **Some modules use intrinsics the shipping compiler does not have.** A `grep` over
   [`helixc/bootstrap/`](../../../helixc/bootstrap/) finds `__arena_get`/`__arena_set`/
   `__arena_push`/`__arena_len` and the float bit-casts `__bits_of_f32`/`__f32_from_bits`
   (so `tensor.hx`, `vec.hx`, `hashmap.hx`, `iterators.hx`, and the *byte/int* core of
   `string.hx` are written entirely within the shipping intrinsic surface). It does **not**
   find `__str_find_byte`, `__parse_i32`, `__str_byte_at`, or `__strlit_to_arena` — so
   `csv.hx`, `mnist.hx`, and `string.hx`'s `string_to_f64` parser depend on intrinsics that
   only ever existed in the deleted Python `helixc` frontend (the same class of gap documented
   for the autodiff libraries in
   [Part III §4](../part3-language/05-autodiff-agi-features.md)).

The honest tiering for this chapter, then:

- **Pattern-proven** — the arena container *pattern* is gate-proven by an inlined corpus
  program with an asserted exit code (`vec_arena.hx` → `45`; `H1_vec.hx`, `H1_hashmap.hx`,
  `H2_string.hx` → `42`). The `stdlib/*.hx` module that the pattern is drawn from is *not*
  itself the compiled artifact.
- **Within-intrinsic-surface** — the module calls only intrinsics present in the shipping
  `kovc` (`tensor.hx`, `vec.hx`, `hashmap.hx`, `iterators.hx`, byte/int `string.hx`). Its code
  is realistic for the shipping compiler, but it has **no standing module-level compile-proof**
  in the gate.
- **Design-stage** — the module calls intrinsics absent from `helixc/bootstrap/` and so does
  **not** compile under the current `kovc` (`csv.hx`, `mnist.hx`, `string_to_f64`). It is a
  faithful record of intended API, not a shipped capability.

> **For AI agents:** when a task needs a *runnable, gate-accepted* growable container today,
> copy the inlined pattern from `H1_vec.hx` / `H1_hashmap.hx` / `H2_string.hx` (or the minimal
> `vec_arena.hx`), **not** an `import` of the stdlib module — there is no import. Never assert
> that `csv.hx`/`mnist.hx`/`string_to_f64` "work"; their intrinsics are not in
> `helixc/bootstrap/`.

### 1.3 A fail-closed-then-saturate discipline runs through all of it

Before the per-module reference, one cross-cutting design fact worth internalising: these
modules were hardened by repeated audit cycles, and two patterns recur everywhere.

- **Saturating i64 accumulators.** Any reduction that could overflow `i32` accumulates in
  `i64` and saturates to `INT32_MAX`/`INT32_MIN` rather than silently wrapping. You will see
  the same idiom in `vec_sum`, `ti1d_dot`, `ti2d_matmul`, `hashmap_sum_values`, and dozens of
  siblings.
- **Strict variants with out-of-band sentinels.** Many readers historically returned `0` on
  both "corruption" and "the answer is genuinely 0," which is ambiguous. The fix was additive:
  a `_strict` companion returns `INT32_MIN` (or `NaN` for `f32`) on corruption/empty so the
  caller can disambiguate. The original is preserved for backward compatibility and usually
  marked **deprecated for new code**.

These are not cosmetic. They are the same calibrated-honesty posture as the trust chain itself
(see [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md)) applied to library
numerics: prefer a detectable sentinel over a plausible-but-wrong value.

---

## 2. The tensor stack — `tensor.hx`

[`helixc/stdlib/tensor.hx`](../../../helixc/stdlib/tensor.hx) is the largest data module
(2236 lines) and the one most relevant to the planned capstone chapters. It provides 1-D and
2-D tensor primitives over the arena, in **two parallel families**: integer (`ti*` / `t1d_*`)
and `f32` (`tf*`). It is **CPU-only**: it contains no `@kernel`, `tile<>`, or `__tile_*`
references — the GPU/PTX path is entirely separate (it lives in `kovc`'s emitters and the
`gpu_*` example kernels, covered in the planned **Part VII — GPU Codegen**). Do not conflate
`tensor.hx`'s `tf2d_matmul` with the GPU GEMM kernels; they are different code on different
targets.

### 2.1 Layout: header, payload, footer

A 1-D tensor is allocated with a magic header, a length, and matching footer guards; the
returned `start` points at element 0. A 2-D tensor stores rows/cols in its header and is
row-major (`M[i,j]` at `start + i*cols + j`).

**Fragment** (the 1-D constructor from
[`helixc/stdlib/tensor.hx`](../../../helixc/stdlib/tensor.hx); excerpt):

```helix
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
```

The header/footer guards exist so the bounds-checkers (`t1d_slice_ok`, `t2d_shape_ok`) can
detect a corrupt or forged handle before any read or write. Every consequential op calls one
of them first.

### 2.2 The f32 representation: bit-pattern in an i32 slot

There is no separate float arena. An `f32` is stored as its IEEE-754 **bit pattern** in an
i32 slot, reinterpreted on read/write via the `__bits_of_f32` / `__f32_from_bits` intrinsics —
which the module notes emit no instruction, they are a type-system relabel:

**Fragment** ([`helixc/stdlib/tensor.hx`](../../../helixc/stdlib/tensor.hx); excerpt):

```helix
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
```

> **Note:** the integer (`ti*`) family was the historically-safe path because it needs no
> bit-reinterpret; the module's in-body comment introducing them calls the `ti*` functions "the
> safe variants until float<->arena bit-reinterpret lands as a codegen primitive." With `__bits_of_f32` /
> `__f32_from_bits` present in the shipping `kovc`, the `tf*` family is on equal footing for
> the float bitcast — both families are written entirely within the shipping intrinsic surface
> (the module as a whole still has no module-level gate proof, per §1.2).

### 2.3 The core API (real signatures)

The module's own header is the canonical signature list. The 1-D and 2-D `f32` cores:

**Fragment** (API summary from the header of
[`helixc/stdlib/tensor.hx`](../../../helixc/stdlib/tensor.hx); excerpt):

```helix
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
```

In the implementation the concrete `f32` element accessors are spelled `tf1d_set` / `tf1d_get`
/ `tf2d_set` / `tf2d_get` and the matrix ops `tf2d_matvec` / `tf2d_matmul` (the `t1d_*` / `t2d_*`
names in the header are the conceptual API; the integer variants are `ti1d_*` / `ti2d_*`). The
general `f32` GEMM is real and row-major:

**Fragment** (the f32 matmul from
[`helixc/stdlib/tensor.hx`](../../../helixc/stdlib/tensor.hx); excerpt — the inner
accumulation, NaN-skipped):

```helix
// 2D row-major f32 matmul: C = A @ B.
//   A is (a_rows x a_cols), B is (a_cols x b_cols), C is (a_rows x b_cols).
fn tf2d_matmul(a_start: i32, a_rows: i32, a_cols: i32,
               b_start: i32, b_cols: i32, c_start: i32) -> i32 {
    // ... shape checks elided ...
    let mut acc: f32 = 0.0_f32;
    while k < a_cols {
        let av = __f32_from_bits(__arena_get(a_start + r * a_cols + k));
        let bv = __f32_from_bits(__arena_get(b_start + k * b_cols + c));
        let prod = av * bv;
        if prod == prod { acc = acc + prod; };
        k = k + 1;
    }
    __arena_set(c_start + r * b_cols + c, __bits_of_f32(acc));
    // ...
}
```

The `if prod == prod` test is the NaN-skip discipline: a single NaN slot poisons only that
product, not the whole output cell (`NaN != NaN` per IEEE-754, so the guard drops it). The
integer GEMM `ti2d_matmul` is the same shape but uses an i64 accumulator with INT32 saturation
instead of NaN-skip.

### 2.4 Breadth, reductions, and the strict family

Beyond construct/set/get/matmul, the module is wide. A representative (non-exhaustive) tour of
real function names you can rely on being present:

- **Element-wise (f32):** `tf1d_add`, `tf1d_sub`, `tf1d_mul`, `tf1d_axpy`, `tf1d_axpby`,
  `tf1d_relu`, `tf1d_add_scalar`, `tf1d_mul_scalar`, `tf1d_scale_inplace`, `tf1d_negate`,
  `tf1d_clamp`, `tf1d_lerp`.
- **Reductions (f32):** `tf1d_sum`, `tf1d_mean`, `tf1d_max`, `tf1d_min`, `tf1d_argmax`,
  `tf1d_argmin`, `tf1d_l1_norm`, `tf1d_l2_norm_sq`, `tf1d_max_abs`, `tf1d_sum_in_range`,
  `tf1d_dot`, `tf1d_dot_with_offset`.
- **2-D (f32):** `tf2d_matvec`, `tf2d_matmul`, `tf2d_transpose`, `tf2d_add`, `tf2d_sub`,
  `tf2d_mul`, `tf2d_scale_inplace`, `tf2d_row_sum`, `tf2d_col_sum`, `tf2d_eye`, `tf2d_diag`,
  `tf2d_trace`, `tf2d_norm_frobenius_sq`, `tf2d_zeros`, `tf2d_ones`, `tf2d_max_abs`.
- **Integer mirrors:** the same surface under `ti1d_*` / `ti2d_*` (e.g. `ti1d_sum`, `ti1d_dot`,
  `ti2d_matmul`, `ti2d_matvec`, `ti2d_transpose`, `ti1d_relu`, `ti1d_clamp`).
- **Strict companions:** for nearly every reduction there is a `_strict` form returning a
  sentinel on corruption — `ti1d_sum_strict`, `tf1d_sum_strict`, `tf1d_max_strict`, etc.

Two deprecation notes you should heed when reading the module:

- `ti2d_get` / `tf2d_get` return `0` / `0.0` on out-of-bounds, which collides with a legitimate
  sparse zero. New code should prefer `ti2d_get_strict` (returns `INT32_MIN` on OOB),
  `tf2d_get_or(..., sentinel)` (caller-supplied sentinel), or the explicit `ti2d_in_bounds` /
  `tf2d_in_bounds` check. The module marks both `get`s **deprecated for safety-critical new
  code**.

> **For AI agents:** the planned **Part VII** capstone walks the *GPU* tensor path (PTX
> kernels: `gpu_matmul_atb`, `gpu_softmax`, `gpu_layernorm_*`, `gpu_adam`, …), not this CPU
> module. If you need a tensor op proven *end-to-end*, that is the capstone (PTX, the load-
> bearing ML stdlib, `[capstone-proven]` in
> [`docs/HELIX_V1_STDLIB.md`](../../../docs/HELIX_V1_STDLIB.md) (d)); `tensor.hx` is the CPU
> reference surface, not the capstone's compute path, and it carries no module-level gate proof.

---

## 3. The growable integer vector — `vec.hx`

[`helixc/stdlib/vec.hx`](../../../helixc/stdlib/vec.hx) is the carry-pair `Vec<i32>` introduced
in §1.1. It has two API tiers.

### 3.1 The unchecked (legacy) carry-pair API

The original API trusts the caller; `vec_new` returns the current arena length, `vec_push`
appends and returns the new count, and reads have **no bounds check**:

**Fragment** ([`helixc/stdlib/vec.hx`](../../../helixc/stdlib/vec.hx); excerpt):

```helix
@pure
fn vec_new() -> i32 {
    __arena_len()
}

fn vec_push(start: i32, count: i32, x: i32) -> i32 {
    __arena_push(x);
    count + 1
}

@pure
fn vec_get(start: i32, i: i32) -> i32 {
    __arena_get(start + i)
}
```

Reductions over an unchecked vec are saturating: `vec_sum` / `vec_product` accumulate in `i64`
and clamp to the i32 range; `vec_max`, `vec_first`, `vec_last`, `vec_index_of`, `vec_contains`,
`vec_eq`, and `vec_reverse_inplace` round out the legacy surface.

### 3.2 The checked API (recommended) and its footgun

A later audit added a magic/cap/footer-guarded tier — `vec_new_checked(cap)`, `vec_ok`,
`vec_set_checked`, `vec_get_checked(..., sentinel)` — which validates the handle before access.
The header is blunt that these are recommended for safety-critical code, and equally blunt
about the trap of mixing the two tiers:

**Fragment** ([`helixc/stdlib/vec.hx`](../../../helixc/stdlib/vec.hx); excerpt of the FOOTGUN
warning):

```helix
// FOOTGUN WARNING:
//   `vec_push(s, count, x)` ignores `s` and pushes to the arena
//   tip. If `s` came from `vec_new_checked(cap)`, the push will
//   OVERWRITE THE FOOTER once count reaches cap (and corrupt
//   adjacent state before then). DO NOT mix vec_push with
//   vec_new_checked — use vec_set_checked exclusively for
//   checked vecs.
```

The module is also honest that the checked tier defends no shipping caller yet ("the
infrastructure exists but defends NO running code today … 'infrastructure-only' closure, not
'threat surface eliminated'"). It is real, additive safety scaffolding awaiting opt-in.

### 3.3 The gate-proven growth pattern

The *growth-on-push* version of this `Vec` is what the gate actually compiles and runs — as an
inlined `struct Vec[T]` corpus program. Here it is in full; it is the real runnable proof that
the arena carry-pair container works end to end through the self-hosting compiler.

**Verified example** —
[`stage0/helixc-bootstrap/corpus_gen/H1_vec.hx`](../../../stage0/helixc-bootstrap/corpus_gen/H1_vec.hx)
(compiled + run by the gate via the freshly self-hosted K2;
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]` asserts exit `42`):

```helix
struct Vec[T] { h: i32 }

fn vec_with_cap(cap0: i32) -> i32 {
    let h = __arena_push(0);
    __arena_push(0);
    __arena_push(0);
    let data = __arena_len();
    let mut i = 0;
    while i < cap0 { __arena_push(0); i = i + 1; };
    __arena_set(h + 1, cap0);
    __arena_set(h + 2, data);
    h
}
fn vec_grow(h: i32) -> i32 {
    let len = __arena_get(h);
    let cap = __arena_get(h + 1);
    let old = __arena_get(h + 2);
    let ncap = cap * 2;
    let ndata = __arena_len();
    let mut i = 0;
    while i < ncap { __arena_push(0); i = i + 1; };
    let mut j = 0;
    while j < len { __arena_set(ndata + j, __arena_get(old + j)); j = j + 1; };
    __arena_set(h + 1, ncap);
    __arena_set(h + 2, ndata);
    0
}
fn vec_push_raw(h: i32, x: i32) -> i32 {
    let len = __arena_get(h);
    let cap = __arena_get(h + 1);
    if len >= cap { vec_grow(h); };
    let data = __arena_get(h + 2);
    __arena_set(data + len, x);
    __arena_set(h, len + 1);
    0
}
impl<T> Vec<T> {
    fn len(self) -> i32 { __arena_get(self.h) }
    fn cap(self) -> i32 { __arena_get(self.h + 1) }
    fn get(self, i: i32) -> T { __arena_get(__arena_get(self.h + 2) + i) }
    fn set(self, i: i32, x: i32) -> i32 { __arena_set(__arena_get(self.h + 2) + i, x); 0 }
    fn push(self, x: i32) -> i32 { vec_push_raw(self.h, x) }
    fn pop(self) -> T {
        let n = __arena_get(self.h) - 1;
        __arena_set(self.h, n);
        __arena_get(__arena_get(self.h + 2) + n)
    }
}

fn main() -> i32 {
    let v = Vec::<i32>{ h: vec_with_cap(2) };
    let mut i = 1;
    while i <= 8 {
        v.push(i);            // 8 pushes from cap 2 -> grows 2->4->8
        i = i + 1;
    };
    // growth happened: 8 elements held in a region that began at cap 2.
    if v.len() != 8 { return 90; };
    if v.cap() != 8 { return 91; };       // proves two relocations occurred
    // read every element back: 1+2+...+8 = 36
    let mut sum = 0;
    let mut k = 0;
    while k < v.len() { sum = sum + v.get(k); k = k + 1; };
    if sum != 36 { return 92; };
    // mutate in place: set element 0 (1 -> 7), sum is now 36-1+7 = 42
    v.set(0, 7);
    // pop the last element (8); len -> 7
    let last = v.pop();
    if last != 8 { return 93; };
    if v.len() != 7 { return 94; };
    // recompute the live sum: 7 + 2 + 3 + 4 + 5 + 6 + 7 = 34 (index 0 now 7,
    // index 7's old value 8 popped off)
    let mut sum2 = 0;
    let mut j = 0;
    while j < v.len() { sum2 = sum2 + v.get(j); j = j + 1; };
    // 34 (live sum) + 8 (popped value) = 42
    sum2 + last
}
```

The gate runs this with the line `chk "$GENC/H1_vec.hx" 42` in
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh); `chk` compiles the file with K2 and
asserts the binary's exit. Exit `42` means: two relocations occurred (cap grew `2→4→8`), all
eight elements read back (`sum == 36`), and `set`/`pop`/`len` all behaved (`34 + 8 == 42`).

A minimal arena-vector (no growth, no struct) is also gate-proven, generated inline by the gate
as `vec_arena.hx` and asserted at exit `45` (`chk "$CD/vec_arena.hx" 45`) — four free functions
`vec_new`/`vec_push`/`vec_len`/`vec_get` over the arena. That is the smallest runnable arena
container in the corpus; the `stdlib/vec.hx` module is the elaborated reference version of the
same idea.

> **For AI agents:** the difference between `stdlib/vec.hx` and `H1_vec.hx` matters. `H1_vec.hx`
> is the compiled, exit-42 proof and stores `len`/`cap`/`data` in a 3-cell header with **growth
> on push**. `stdlib/vec.hx`'s legacy `vec_push(start, count, x)` does **not** grow and ignores
> `start` — it appends at the arena tip. If you need growth in a real program, copy the
> `H1_vec.hx` shape.

---

## 4. The integer hash map — `hashmap.hx`

[`helixc/stdlib/hashmap.hx`](../../../helixc/stdlib/hashmap.hx) is a fixed-capacity,
linear-probing map keyed by `i32`, valued by `i32`. Each bucket consumes three arena slots
(occupancy flag, key, value); the carry-pair is `(start, cap)`.

**Fragment** (API summary from the header of
[`helixc/stdlib/hashmap.hx`](../../../helixc/stdlib/hashmap.hx); excerpt):

```helix
// API:
//   hashmap_new(cap)                      -> i32   start = arena index of bucket 0
//   hashmap_put(start, cap, k, v)         -> i32   1 if new, 0 if updated, -1 if full
//   hashmap_get(start, cap, k, default)   -> i32   value or default if missing
//   hashmap_has(start, cap, k)            -> i32   1 if present, 0 otherwise
//   hashmap_size(start, cap)              -> i32   count of occupied buckets
```

The probe is the textbook open-addressing loop: hash to a bucket, scan forward on collision,
wrap at `cap`, bail after `cap` probes (the map is full). `hashmap_put` returns `1`/`0`/`-1`
for new/updated/full; `hashmap_get` takes a caller-supplied default for the missing case. The
module also ships a large analytics surface — `hashmap_keys` / `hashmap_values` (push occupied
entries into a fresh arena slice, in bucket order, index-aligned), `hashmap_increment` (the
accumulator-map / word-count pattern), `hashmap_swap`, `hashmap_clear`, `hashmap_sum_values`,
`hashmap_max_value` / `hashmap_min_value`, `hashmap_argmax_key`, `hashmap_load_factor_x100`,
and many `*_strict` companions.

The same ambiguity-sentinel discipline applies: `hashmap_get` returns `default` for *both*
"absent" and "corrupt," so the module adds `hashmap_status` (0 = ok, 1 = corrupt) and
`hashmap_get_strict` (returns `INT32_MIN` on corruption). It marks the plain `hashmap_get`
**deprecated for safety-critical new code**.

The gate-proven version is, again, an inlined corpus program that exercises a **forced hash
collision** resolved by probing.

**Verified example** —
[`stage0/helixc-bootstrap/corpus_gen/H1_hashmap.hx`](../../../stage0/helixc-bootstrap/corpus_gen/H1_hashmap.hx)
(compiled + run by the gate via K2; [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh)
step `[4]` asserts exit `42`). Its `main` is the load-bearing part:

```helix
fn main() -> i32 {
    let m = hm_new(8);
    // COLLISION: 3, 11, 19 all -> bucket 3. probing places them at 3,4,5.
    hm_insert(m, 3, 10);
    hm_insert(m, 11, 20);
    hm_insert(m, 19, 5);
    // a non-colliding key (6 -> bucket 6) for good measure.
    hm_insert(m, 6, 100);
    // overwrite an existing key (must NOT add a new slot, must replace val).
    hm_insert(m, 6, 7);
    // four distinct keys -> count must be 4 (the overwrite did not grow it).
    if hm_len(m) != 4 { return 80; };
    // read every collided key back by key (probe walks the chain):
    if hm_get(m, 3)  != 10 { return 81; };
    if hm_get(m, 11) != 20 { return 82; };
    if hm_get(m, 19) != 5  { return 83; };
    if hm_get(m, 6)  != 7  { return 84; };   // sees the overwritten value
    // a key that was never inserted must miss.
    if hm_get(m, 99) != hm_missing() { return 85; };
    // contains() on present (incl. a probed/collided key) and absent keys.
    if hm_contains(m, 19) != 1 { return 86; };   // present, found via probe
    if hm_contains(m, 99) != 0 { return 87; };   // absent
    if hm_contains(m, 27) != 0 { return 88; };   // 27&7=3 (collides) but absent
    // value sum of the four live keys: 10 + 20 + 5 + 7 = 42.
    hm_get(m, 3) + hm_get(m, 11) + hm_get(m, 19) + hm_get(m, 6)
}
```

Exit `42` proves: three keys colliding into bucket 3 were probed into 3/4/5 and read back
correctly, an overwrite replaced rather than grew (`count == 4`), a never-inserted key missed,
and `contains` discriminated present / absent / *collides-but-absent* (key 27 hashes to bucket
3 but was never inserted). The inlined `hm_*` functions use the same mask-and-probe logic as
`stdlib/hashmap.hx`'s `hashmap_hash` + `hashmap_put`/`hashmap_get`.

---

## 5. The byte string — `string.hx`

[`helixc/stdlib/string.hx`](../../../helixc/stdlib/string.hx) is the carry-pair `String`: one
ASCII byte per `i32` arena slot (lower 8 bits), threaded as `(start, len)`. The header is
candid about the space tradeoff — one byte per slot is wasteful but indexing-simple, and AGI
work is tensor-dominated, not string-dominated. The same two-tier `string_new`/`string_push`/
`string_get` (unchecked) plus `string_new_checked`/`string_ok`/`string_get_checked` (guarded)
structure as `vec.hx` applies.

The module is wide and mostly arena-only (so within the shipping intrinsic surface). The core,
verbatim from the header:

**Fragment** (API summary from the header of
[`helixc/stdlib/string.hx`](../../../helixc/stdlib/string.hx); excerpt):

```helix
// API:
//   string_new()                            -> i32   start = current arena length
//   string_push(start, len, b)              -> i32   pushes byte; returns new len
//   string_get(start, i)                    -> i32   byte at index i (lower 8 bits)
//   string_eq(a, an, b, bn)                 -> i32   1 if equal, 0 otherwise
//   string_index_of(start, len, byte)       -> i32   first index of byte, -1 if missing
//   string_starts_with(s, sn, p, pn)        -> i32   1 if s starts with p
//   string_from_int(n)                      -> i32   appends ASCII digits of n to arena ...
//   string_to_int(start, len)               -> i32   parses decimal int from slice ...
```

Beyond the core it provides: case folding (`string_to_upper` / `string_to_lower`,
ASCII-only — no UTF-8), search/compare (`string_contains`, `string_compare`,
`string_last_index_of`, `string_index_of_n`, `string_first_index_at_or_after`), slicing /
building (`string_concat`, `string_substring`, `string_repeat`, `string_pad_left` /
`string_pad_right` / `string_pad_center`, `string_strip_byte`, `string_replace_byte`),
classification (`string_is_ascii`, `string_is_digit_only`, `string_count_alpha`,
`string_count_digit`, `string_count_lines`), trimming helpers
(`string_trim_left_byte` / `string_trim_right_byte`, returning a skip count / trimmed length),
and strict / validated parsers (`string_is_int` + `string_to_int_strict`). `string_to_int`
itself saturates via an i64 accumulator and **silently skips non-digit bytes** — use
`string_is_int` as a gate first if malformed input must be rejected.

One honest limit on `string.hx`: its **float parser is design-stage**. `string_to_f64` (and
`string_to_f64_strict`) compose `__parse_i32` and `__str_find_byte`, both of which are absent
from `helixc/bootstrap/` (§1.2), so that pair does **not** compile under the shipping `kovc`.
The byte/int string surface above does not depend on them.

The gate-proven version of the rich string is the inlined `H2_string.hx` corpus program — the
same carry-pair handle pattern as `H1_vec.hx`, with growth-on-push exercised by a concat that
crosses the initial capacity.

**Verified example** —
[`stage0/helixc-bootstrap/corpus_gen/H2_string.hx`](../../../stage0/helixc-bootstrap/corpus_gen/H2_string.hx)
(compiled + run by the gate via K2; [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh)
step `[4]` asserts exit `42`). The build-and-concat core (a Fragment of that program's body):

```helix
fn str_new() -> i32 { str_with_cap(4) }
fn str_push_byte(s: String, b: i32) -> i32 { str_push_raw(s.h, b) }
fn str_len(s: String) -> i32 { __arena_get(s.h) }
fn str_byte_at(s: String, i: i32) -> i32 { __arena_get(__arena_get(s.h + 2) + i) }
fn str_concat(a: String, b: String) -> i32 {
    let rh = str_new();
    let la = str_len(a);
    let mut i = 0;
    while i < la { str_push_raw(rh, str_byte_at(a, i)); i = i + 1; };
    let lb = str_len(b);
    let mut j = 0;
    while j < lb { str_push_raw(rh, str_byte_at(b, j)); j = j + 1; };
    rh
}
```

Its `main` builds `"Hel" ++ "lix" = "Hellix"` byte-by-byte (the concat result starts at cap 4,
receives 6 bytes, and so forces a grow `4→8`), indexes bytes back out, and asserts `str_eq`
against an independently-built `"Hellix"` plus three negative cases (unequal same-length,
unequal different-length, one-byte diff). Exit is `str_len("Hellix") * 7 == 42`.

> **Note:** in `H2_string.hx`, `str_new`/`str_concat` return the arena **handle** (an `i32`),
> wrapped at the call site as `String { h: ... }`, because returning a `struct` *by value* from
> a function mis-lowers in this from-raw compiler (a known v-next codegen gap; passing a struct
> by value as a **parameter** is fully supported). This is a real constraint on building
> string-returning helpers today, documented in the program's own header and in
> [`docs/HELIX_COMPLETION.md`](../../../docs/HELIX_COMPLETION.md) (H-2).

---

## 6. Iterator-style vector combinators — `iterators.hx`

[`helixc/stdlib/iterators.hx`](../../../helixc/stdlib/iterators.hx) is the Phase-1 "iterator"
layer over arena `Vec<i32>`. Helix has no closures usable here, so the combinators are
**specialised by an op-tag or a scalar argument** rather than taking a function. Output-
producing ops (`map`/`zip`/`filter`/`range`) append to the arena and return the new start
index. It uses only arena intrinsics, so its code is within the shipping intrinsic surface
(no module-level gate proof, per §1.2).

**Fragment** (API summary from the header of
[`helixc/stdlib/iterators.hx`](../../../helixc/stdlib/iterators.hx); excerpt):

```helix
// API:
//   range_to_vec(lo, hi)                 -> i32   appends lo..hi (exclusive); returns start
//   vec_min(start, count)                -> i32   smallest element (0 if empty)
//   vec_count_eq(start, count, target)   -> i32   how many elements equal target
//   vec_fold_op(start, count, init, op)  -> i32   reduce with op 0:add 1:mul 2:max 3:min
//   vec_map_add_scalar(start, count, k)  -> i32   appends [x+k for x in v]; returns new start
//   vec_map_mul_scalar(start, count, k)  -> i32   appends [x*k for x in v]; returns new start
//   vec_zip_add(a, b, count)             -> i32   appends [a[i]+b[i]]; returns new start
//   vec_filter_lt(start, count, t)       -> i32   appends elems < t; returns kept count.
//   vec_argmin(start, count)             -> i32   index of smallest element (-1 if empty).
//   vec_dot(a, b, count)                 -> i32   dot product sum(a[i]*b[i]).
```

The `vec_fold_op` reducer is the clearest illustration of the closure-free design: instead of
passing a binary function, you pass an integer `op` tag selecting add/mul/max/min, and the add
and mul branches use a saturating i64 accumulator:

**Fragment** ([`helixc/stdlib/iterators.hx`](../../../helixc/stdlib/iterators.hx); excerpt of
`vec_fold_op`):

```helix
@pure
fn vec_fold_op(start: i32, count: i32, init: i32, op: i32) -> i32 {
    let mut i: i32 = 0;
    let mut acc_i: i64 = init as i64;
    let mut acc_b: i32 = init;
    let hi: i64 = 2147483647_i64;
    let lo: i64 = (0_i64 - 2147483647_i64) - 1_i64;
    while i < count {
        let v = __arena_get(start + i);
        if op == 0 {
            acc_i = acc_i + (v as i64);
            if acc_i > hi { acc_i = hi; }
            else { if acc_i < lo { acc_i = lo; } };
        }
        // ... op == 1 (mul, saturating), op == 2 (max), op == 3 (min) ...
        i = i + 1;
    }
    if op == 0 { acc_i as i32 }
    else { if op == 1 { acc_i as i32 }
    else { acc_b } }
}
```

The module is large and ML-flavoured. A representative inventory of real names:
`vec_map_add_scalar` / `vec_map_mul_scalar` / `vec_map_neg` / `vec_map_abs` / `vec_map_relu` /
`vec_map_square` / `vec_map_clamp`; `vec_zip_add` / `vec_zip_sub` / `vec_zip_mul` /
`vec_zip_min` / `vec_zip_max` / `vec_zip_div` / `vec_zip_mod` and the boolean
`vec_zip_eq` / `vec_zip_lt` / … family; `vec_filter_lt` / `gt` / `eq` / `le` / `ge` / `ne`
(each returning the *kept count* — save `__arena_len()` before the call to recover the new
start); reductions `vec_dot`, `vec_abs_sum`, `vec_sum_squares`, `vec_l1_distance`,
`vec_l2_squared_distance`, `vec_max_abs`, `vec_mean`; cumulative / windowed
`vec_cumsum`, `vec_diff`, `vec_pairwise_diff`, `vec_window_sum` / `vec_window_max` /
`vec_window_min`, `vec_running_max` / `vec_running_min`; and structural
`vec_argsort` (selection sort returning an index permutation), `vec_unique_alloc`,
`vec_dedup_consecutive`, `vec_intersect`, `vec_difference`, `vec_take` / `vec_drop` /
`vec_concat` / `vec_reverse_alloc` / `vec_rotate_left_alloc`.

Two honesty notes the module makes about itself:

- **Divide/modulo fail closed.** `vec_zip_div` / `vec_zip_mod` return an `INT32_MIN` sentinel on
  divide-by-zero (and guard the `INT32_MIN / -1` / `INT32_MIN % -1` SIGFPE corner) rather than
  pushing a `0` that would collide with a legitimate `0/x` result.
- **The append-and-return-start contract.** `range_to_vec`, the `vec_map_*`/`vec_zip_*` ops, and
  the allocating structural ops all append to the arena tip and return the new start index. The
  `vec_filter_*` ops instead return the *kept count*, so the caller must capture `__arena_len()`
  immediately before the call to learn where the filtered slice begins. Mixing two such builds
  interleaves them (§1.1).

---

## 7. CSV iteration — `csv.hx` *(design-stage)*

[`helixc/stdlib/csv.hx`](../../../helixc/stdlib/csv.hx) is a stdlib-only line/field iterator
over an arena-resident CSV blob. It is **design-stage** for the shipping toolchain: it calls
`__str_find_byte`, `__str_byte_at`, `__parse_i32`, and `__strlit_to_arena`, none of which are
in `helixc/bootstrap/` (§1.2). Document it as the intended API; do not present it as compiling
under the current `kovc`.

The iteration model is the closure-free cursor pattern — the caller threads an `offset` through
the API, exactly as the module header shows:

**Fragment** (header of [`helixc/stdlib/csv.hx`](../../../helixc/stdlib/csv.hx); the documented
usage pattern, not a runnable program under shipping `kovc`):

```helix
//   let mut off: i32 = 0;
//   while csv_has_next_line(blob, blob_len, off) == 1 {
//       let line_off = off;
//       let line_len = csv_line_len(blob, blob_len, off);
//       // ... process line ...
//       off = csv_next_line_offset(blob, blob_len, off);
//   }
```

The real API is `csv_has_next_line`, `csv_line_len`, `csv_next_line_offset`, `csv_field_len`,
`csv_next_field_offset`, `csv_count_lines`, `csv_count_fields`, and the numeric shim
`csv_parse_field_i32`. Fields split on `,` (byte 44); lines on `\n` (byte 10).

The module is honest about a **scan-cap limit** that any user must respect:

**Fragment** ([`helixc/stdlib/csv.hx`](../../../helixc/stdlib/csv.hx); excerpt of the chunk-cap
note):

```helix
// IMPORTANT — 256-byte chunk cap:
//   `__str_find_byte` is compile-time-unrolled to MAX_SCAN=256, so a
//   single call sees at most 256 bytes starting at the given offset.
//   For lines longer than 256 bytes, csv_line_len returns the full
//   length by chaining multiple find_byte calls (up to 4 chunks =
//   1024-byte max line). Lines longer than 1024 bytes are truncated
//   for the line_len accounting; raise MAX_CHUNKS below if needed.
```

Because silent truncation is itself a hazard, the module adds detectors —
`csv_line_was_truncated`, `csv_count_lines_was_capped` / `csv_count_lines_strict` (returns
`INT32_MIN` when the 65536-line iteration cap is hit), and the field-count equivalents — so a
caller can tell "exactly N" from "capped at N." That is the same sentinel discipline as the
other modules, applied to truncation.

---

## 8. The MNIST IDX reader — `mnist.hx` *(design-stage)*

[`helixc/stdlib/mnist.hx`](../../../helixc/stdlib/mnist.hx) parses LeCun **IDX**-format headers
and bounds-checks the body. It does **no file I/O** — the caller is expected to have loaded the
IDX bytes into the arena already (real file I/O is an unshipped increment). And it is
**design-stage** for the shipping toolchain: every accessor calls `__str_byte_at`, which is
absent from `helixc/bootstrap/` (§1.2). Treat it as the intended header-parse API.

The header documents both the IDX format and the canonical MNIST files:

**Fragment** (header of [`helixc/stdlib/mnist.hx`](../../../helixc/stdlib/mnist.hx); excerpt):

```helix
// IDX format (LeCun's MNIST format):
//   bytes 0..1   : 0x00 0x00 (zero padding)
//   byte  2      : dtype code (0x08=u8, 0x09=i8, 0x0B=i16, 0x0C=i32,
//                              0x0D=f32, 0x0E=f64)
//   byte  3      : ndims  (1, 2, 3, ...)
//   bytes 4..4+(ndims*4) : ndims big-endian u32 dimension sizes
//   body         : product(dims) * dtype_size bytes
//
// MNIST canonical files:
//   train-images-idx3-ubyte: magic 0x00000803, ndims=3, dims=[60000, 28, 28]
//   train-labels-idx1-ubyte: magic 0x00000801, ndims=1, dims=[60000]
```

The API: `mnist_idx_magic_ok`, `mnist_idx_dtype`, `mnist_idx_ndims`, `mnist_idx_header_size`,
`mnist_idx_dim(i)` (big-endian u32 decode), `mnist_idx_body_offset` / `mnist_idx_body_len_bytes`,
`mnist_idx_dtype_size`, `mnist_idx_expected_body_len`, `mnist_idx_validate`, `mnist_idx_u8_at`,
and `mnist_idx_image_pixel(img_idx, row, col)` for the canonical 3-D u8 case. The validator is
the right place to start a load — it cross-checks the header against the body length and
fails closed on any mismatch or overflow:

**Fragment** ([`helixc/stdlib/mnist.hx`](../../../helixc/stdlib/mnist.hx); excerpt of
`mnist_idx_validate`):

```helix
@pure
fn mnist_idx_validate(blob: i32, blob_len: i32) -> i32 {
    if mnist_idx_magic_ok(blob, blob_len) == 0 { 0 }
    else {
        let dt = mnist_idx_dtype(blob, blob_len);
        if mnist_idx_dtype_size(dt) == 0 { 0 }
        else {
            let header = mnist_idx_header_size(blob, blob_len);
            if header > blob_len { 0 }
            else {
                let expect = mnist_idx_expected_body_len(blob, blob_len);
                if expect == 0 - 2147483647 - 1 { 0 }
                else {
                    let actual = blob_len - header;
                    if expect == actual { 1 } else { 0 }
                }
            }
        }
    }
}
```

Two honest notes the module makes:

- **Overflow fails the validate.** `mnist_idx_expected_body_len` guards every dimension multiply
  and returns an `INT32_MIN` sentinel on overflow; `mnist_idx_validate` checks for that sentinel
  and rejects, so a corrupt large file (e.g. `[1000,1000,1000,1000]` u8 = 10¹²) cannot silently
  pass.
- **The hot accessors are unchecked by default.** `mnist_idx_u8_at` / `mnist_idx_image_pixel`
  have no bounds check (they assume you called `validate` first); the `_checked` variants
  (`mnist_idx_u8_at_checked`, `mnist_idx_image_pixel_checked`) return `-1` on out-of-bounds.
  The header advises hot training loops to cache `(h, w)` and call `mnist_idx_u8_at` with a
  pre-computed offset rather than paying the per-pixel header re-read of `mnist_idx_image_pixel`.

---

## 9. Quick reference: module status at a glance

| Module | Carry-pair / handle | Intrinsics beyond `__arena_*` | Shipping-compiler status |
|--------|---------------------|-------------------------------|--------------------------|
| [`tensor.hx`](../../../helixc/stdlib/tensor.hx) | `start` (+ shape) | `__bits_of_f32`, `__f32_from_bits` (present) | within intrinsic surface; CPU-only; no module-level gate proof |
| [`vec.hx`](../../../helixc/stdlib/vec.hx) | `(start, count)` / `(start, cap)` | none | within intrinsic surface; pattern gate-proven via `H1_vec.hx`/`vec_arena.hx` |
| [`hashmap.hx`](../../../helixc/stdlib/hashmap.hx) | `(start, cap)` | none | within intrinsic surface; pattern gate-proven via `H1_hashmap.hx` |
| [`string.hx`](../../../helixc/stdlib/string.hx) | `(start, len)` | byte/int core: none; `string_to_f64`: `__parse_i32`/`__str_find_byte` (absent) | byte/int core within intrinsic surface (pattern gate-proven via `H2_string.hx`); float parser design-stage |
| [`iterators.hx`](../../../helixc/stdlib/iterators.hx) | `(start, count)` | none | within intrinsic surface; no module-level gate proof |
| [`csv.hx`](../../../helixc/stdlib/csv.hx) | `(blob, blob_len, off)` | `__str_find_byte`, `__str_byte_at`, `__parse_i32`, `__strlit_to_arena` (absent) | **design-stage** (does not compile under shipping `kovc`) |
| [`mnist.hx`](../../../helixc/stdlib/mnist.hx) | `(blob, blob_len)` | `__str_byte_at` (absent) | **design-stage** (does not compile under shipping `kovc`) |

The one-line summary: Helix's data + I/O library is a set of arena views threaded by integer
handles, written with a consistent saturate-then-fail-closed numerics discipline. The container
*patterns* (growable `Vec`, probing `HashMap`, byte `String`) are gate-proven by inlined corpus
programs that exit `42`/`45`; the elaborated `stdlib/*.hx` modules are real reference source
that the gate does not compile as modules, and the `csv`/`mnist`/float-parse surfaces depend on
intrinsics absent from the shipping compiler. When in doubt, prefer the inlined corpus shape
([`H1_vec.hx`](../../../stage0/helixc-bootstrap/corpus_gen/H1_vec.hx),
[`H1_hashmap.hx`](../../../stage0/helixc-bootstrap/corpus_gen/H1_hashmap.hx),
[`H2_string.hx`](../../../stage0/helixc-bootstrap/corpus_gen/H2_string.hx)) and verify any
intrinsic against [`helixc/bootstrap/`](../../../helixc/bootstrap/).

---

**Next:** [Part V — The Compiler (kovc): Front end](../part5-compiler/01-front-end.md) turns from
the library that runs *on* `kovc` to `kovc` itself — the front end (lexer, parser, typecheck), the
[IR and lowering passes](../part5-compiler/02-ir-and-passes.md), and the
[x86-64 ELF back end](../part5-compiler/03-x86-backend.md) that lowers every arena `__arena_*` call
and `__bits_of_f32` relabel seen in this chapter. The GPU tensor path that the capstone actually
trains on is **Part VII — GPU Codegen** *(planned)*; until it ships,
[`docs/HELIX_V1_STDLIB.md`](../../../docs/HELIX_V1_STDLIB.md) is the authoritative builtin
reference and [Part IX — "Driving Helix"](../part9-for-ai-agents/01-driving-helix.md) is the
operator manual for compiling and running Helix programs.
