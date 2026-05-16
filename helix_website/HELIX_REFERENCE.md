# Helix — Complete Reference

> A comprehensive reference for Claude Design to build the public Helix website. This is a marketing/design draft, not the source of truth for current shipped capability. Before publishing, check `README.md`, `docs/stage35-progress-2026-05-15.md`, and live tests. Clearly distinguish implemented features from roadmap targets.

---

## Table of Contents

1. [What is Helix](#what-is-helix)
2. [Core Philosophy](#core-philosophy)
3. [Quick Start](#quick-start)
4. [Language Reference](#language-reference)
5. [Standard Library](#standard-library)
6. [Autodiff](#autodiff)
7. [Tile & Tensor Types](#tile--tensor-types)
8. [Foreign Function Interface (FFI)](#foreign-function-interface-ffi)
9. [Reflection Runtime](#reflection-runtime)
10. [Compilation Pipeline](#compilation-pipeline)
11. [Bootstrap Chain](#bootstrap-chain)
12. [Compiler Architecture](#compiler-architecture)
13. [Tooling & CLI](#tooling--cli)
14. [Open-Source Commitments](#open-source-commitments)
15. [Comparisons](#comparisons)
16. [Code Samples Gallery](#code-samples-gallery)
17. [Glossary](#glossary)
18. [Visual Identity Suggestions](#visual-identity-suggestions)

---

## What is Helix

**Helix is a from-scratch ML-native systems language being built from a raw-binary bootstrap toward a self-hosting compiler, with first-class autodiff and tile/tensor primitives as core language goals.**

### One-line pitches (pick one for the hero)

- "A language stack growing from a 299-byte audited bootstrap root."
- "Systems performance, math notation, autodiff in the language."
- "A language stack growing toward self-hosted machine learning, built from raw binary."
- "Helix: from one audited machine-code root toward a self-hosting ML compiler."

### Three-pillar pitch (use as feature triplet)

🔓 **Open-source weights, data, and code.** Apache 2.0 source · CC-BY 4.0 docs · CC0 model weights. Public training data only.

⚙️ **Bootstrapped from raw binary.** The verified hex0 bootstrap root is 299 bytes. The long-term chain is designed to climb from that root toward a self-hosted Helix compiler; today the production compiler is still Python-hosted `helixc`.

🧠 **ML-first, not ML-bolted-on.** Autodiff (`grad`, `grad_rev_all`), checkpoint rematerialization, tile/tensor types, and an x86 + PTX backend are part of the language — not a library.

### The 30-second elevator pitch

> Helix is a programming language designed for machine learning and auditable systems work. The project starts from a 299-byte raw-binary bootstrap root and is working toward a self-hosted compiler, while today's production compiler remains Python-hosted `helixc`. It aims to combine type safety, symbolic autodiff, tensor/tile primitives, and verifier-gated self-improvement in one open stack. The toolchain is open source under permissive licenses (Apache 2.0 code, CC-BY 4.0 docs, CC0 model weights when produced). Helix is the foundation of the Kovostov AGI project: an attempt to build AGI on a fully open, fully auditable stack.

---

## Core Philosophy

### 1. No silent corruption

Every place where the compiler could silently produce wrong code traps with a unique trap-id (convention: `AST_TAG * 1000 + sub_id`). When a compile-time invariant fails, the produced binary contains a `ud2` instruction (SIGILL) with the trap-id encoded — clear signal vs. silent garbage. 23+ silent-corruption bugs were found and fixed during development; all have public reproducer + status entries on `/audits`.

### 2. Growing from raw binary

The live Helix bootstrap floor is the 299-byte `hex0` artifact. The current
production compiler remains Python-hosted `helixc`; the zero-external-toolchain
story applies to the target bootstrap chain once later links are implemented
and verified. The target chain is:

Target chain: `hex0` (299 bytes, hand-built) → `hex1` → `M0` → `M1` → `M2-Planet` → `kovc-bootstrap` → self-hosted Helix compiler.

The design goal is that each link compiles the next and can be byte-audited
from `hex0`. Today, only the `hex0` root is live and measured; later links are
roadmap targets. `libc` is optional for user programs and is only needed when
code uses FFI.

### 3. ML-first language design

Helix is designed for the kind of code ML researchers actually write: numerical kernels, autodiff-friendly functions, tile-based linear algebra. Forward and reverse-mode autodiff are language features, not library calls. Tiles and tensors are types in the type system. The PTX backend emits GPU kernels alongside x86.

### 4. Total by default

Functions are checked for structural recursion at compile time. Non-terminating functions must be explicitly marked `@partial`. This rules out a class of "stuck infinite loop" bugs that plague Python ML code.

### 5. Open by commitment

- **Code**: Apache 2.0 (you can use it commercially)
- **Documentation**: CC-BY 4.0
- **Model weights**: CC0 (public domain)
- **Training data**: only data with explicit public-use rights

---

## Quick Start

### Hello, World!

```rust
fn main() -> i32 { 42 }
```

That's it. Compile with `kovc hello.hx` and you get a 4KB ELF binary that returns 42 from `main`. No `import std`. No `printf`. Just the value.

### A real program

```rust
struct Vec2 { x: f64, y: f64 }

fn dot(a: Vec2, b: Vec2) -> f64 {
    a.x * b.x + a.y * b.y
}

fn main() -> i32 {
    let u = Vec2 { 3.0_f64, 4.0_f64 };
    let v = Vec2 { 1.0_f64, 2.0_f64 };
    let result = dot(u, v);
    result as i32   // 11
}
```

### Autodiff in the language

```rust
fn loss(x: f64) -> f64 { x * x + 3.0_f64 * x }

fn main() -> f64 {
    grad(loss)(2.0_f64)   // 7.0 — derivative is 2x + 3, evaluated at x=2
}
```

### Tile-based matrix multiplication

```rust
fn main() -> f32 {
    let a = tile<f32, [4, 4], REG>::ones();
    let b = tile<f32, [4, 4], REG>::ones();
    let c = tile_matmul(a, b);
    c.get(0, 0)   // 4.0 — sum of products of two row-of-ones, column-of-ones
}
```

---

## Language Reference

### Lexical structure

#### Keywords (reserved identifiers)

```
let mut if else while fn struct enum match
mod use trait impl for extern
Quote Splice modify verifier
@pure @effect @partial @total @checkpoint @kernel @autotune
@deprecated @since
```

#### Literals

| Form | Type | Example |
|------|------|---------|
| `42` | `i32` (default int) | `42` |
| `42_i64` | `i64` | `1000000000000_i64` |
| `42_u32` | `u32` | `42_u32` |
| `42_u64` | `u64` | `42_u64` |
| `42_u8` | `u8` | `255_u8` |
| `42_u16` | `u16` | `42_u16` |
| `-1_i8` | `i8` | `-1_i8` |
| `42_i16` | `i16` | `42_i16` |
| `3.14_f32` | `f32` | `3.14_f32` |
| `3.14_f64` | `f64` | `3.14_f64` |
| `0.5_bf16` | `bf16` | `0.5_bf16` |
| `"hello\n"` | string slice | `"hello\n"` |
| `true`, `false` | `bool` (i32 0/1) | `true` |

#### Operators

| Category | Operators |
|----------|-----------|
| Arithmetic | `+ - * / %` (signed and unsigned dispatch) |
| Bitwise | `& \| ^ ~ << >>` (arithmetic shift; logical via `as u32`) |
| Comparison | `== != < <= > >=` |
| Logical | `&& \|\| !` |
| Assignment | `=` (only for `let mut` bindings) |
| Type | `as` (explicit cast) |
| Path | `::` (modules, enum variants, turbofish) |
| Field | `.IDENT` and `.NUM` (positional tuple/struct) |
| Index | `[expr]` |
| Pattern | `=>` (match arm), `..` (range exclusive), `..=` (range inclusive) |
| Closure | `\|x, y\| body` |

#### Comments

```rust
// Line comment
/* Block comment */
/// Doc comment (auto-extracted by `kovc --doc`)
```

### Type system

#### Scalar types (12 built-in)

| Tag | Type | Bits | Range | Storage |
|-----|------|------|-------|---------|
| 0   | `i32` | 32 | -2^31 to 2^31-1 | 4-byte stack slot |
| 1   | `f32` | 32 | IEEE 754 binary32 | 4-byte stack slot |
| 2   | `f64` | 64 | IEEE 754 binary64 | 8-byte stack slot |
| 3   | `i64` | 64 | -2^63 to 2^63-1 | 8-byte stack slot |
| 4   | `bf16` | 16 | brain-float-16 | 4-byte slot, low 16 bits zero |
| 6   | `u32` | 32 | 0 to 2^32-1 | 4-byte stack slot |
| 7   | `u8` | 8 | 0 to 255 | 4-byte slot, high 24 bits zero |
| 8   | `u16` | 16 | 0 to 65535 | 4-byte slot, high 16 bits zero |
| 9   | `u64` | 64 | 0 to 2^64-1 | 8-byte stack slot |
| 10  | `i8` | 8 | -128 to 127 | 4-byte slot, sign-extended |
| 11  | `i16` | 16 | -32768 to 32767 | 4-byte slot, sign-extended |

bf16 is recognized for ML-oriented source compatibility and roadmap work.
Full bf16 arithmetic lowering, Blackwell tensor-core integration, and numeric
parity across backends are target capabilities, not shipped Stage 35 behavior.

#### Aggregate types

- **Tuples**: `(a, b, c)` — heterogeneous fixed-arity. Field access via `.0`, `.1`, `.2`. Up to 16 elements.
- **Arrays**: `[a, b, c]` — homogeneous fixed-size. Index access via `arr[i]`. Up to 16 elements.
- **Structs**: `struct Pt { x: i32, y: i32 }` — named fields, by-value pass to functions, nested structs supported.
- **Enums**: `enum Maybe { None, Some(i32) }` — discriminated unions with payload variants.

#### Generic types

- **Generic functions**: `fn id<T>(x: T) -> T { x }` — monomorphized at parse time. Mangled name in fn_table: `id__i32`, `id__f64`.
- **Bounded generics**: `fn cmp<T: Eq>(a: T, b: T) -> i32 { T::eq(a, b) }` — trait bounds checked at instantiation.
- **Generic structs**: `struct Pair<A, B> { a: A, b: B }`.

#### Pointer types

- `*const T` — read-only pointer (FFI use case).
- `*mut T` — read-write pointer (FFI use case).
- `&T`, `&mut T` — Helix-managed references (Phase-1, planned).

### Variables and bindings

```rust
let x = 5;                    // immutable, type inferred (i32)
let y: f64 = 2.5_f64;         // type-annotated
let mut counter = 0;          // mutable
counter = counter + 1;        // assignment requires `let mut`

let (a, b, c) = (1, 2, 3);    // tuple destructure (Phase-1)
```

Type inference is local: the right-hand side determines the binding type. Function parameter and return types are required.

### Control flow

#### `if` / `else if` / `else`

```rust
fn classify(n: i32) -> i32 {
    if n < 0 { 1 }
    else if n == 0 { 2 }
    else if n < 10 { 3 }
    else { 4 }
}
```

`if` is an expression and yields a value. The else branch is required when the if-expression is used as a value.

#### `while`

```rust
let mut i = 0;
let mut sum = 0;
while i < 10 {
    sum = sum + i;
    i = i + 1;
}
```

`while` always yields `0`. For value-producing loops, use recursion or fold patterns.

#### `match`

```rust
fn describe(n: i32) -> i32 {
    match n {
        0       => 100,
        1..10   => 200,         // exclusive range
        10..=20 => 300,         // inclusive range
        v       => v + 1,       // bind
        _       => 999,         // wildcard
    }
}
```

Pattern kinds:
- **PatLit**: literal match (`0`, `1.0_f64`)
- **PatBind**: binds the scrutinee value to a name (`v`)
- **PatWildcard**: matches anything, no bind (`_`)
- **PatRange**: `0..10` (exclusive), `0..=10` (inclusive)
- **PatVariant**: enum variant with payload (`Maybe::Some(v)`)
- **PatTuple**: tuple destructure (`(0, _)`, `(x, y)`)
- **PatOr**: alternation (`a | b | c`) — variants with same shape
- **Guards**: `if cond` after pattern

Exhaustiveness is checked at compile time. Non-exhaustive matches trap at runtime with id 62001.

### Functions

#### Basic function declaration

```rust
fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

- Parameter types and return type are required
- Body is a single expression (the last expression's value is returned)
- No explicit `return` keyword (Phase-0); blocks yield their last expression

#### Generic functions (Stage 8)

```rust
fn id<T>(x: T) -> T { x }
fn pair<A, B>(a: A, b: B) -> A { a }

fn main() -> i32 {
    let x = id::<i32>(42);          // turbofish
    let p = pair::<i32, f64>(7, 1.0_f64);
    x + (p as i32)
}
```

Generics are monomorphized at parse time. Each instantiation produces a distinct mangled function (`id__i32`, `pair__i32_f64`). Phase-0 cap: 32 instantiations per program. Recursive generic functions trigger trap 87001.

#### Bounded generics (Stage 8.5)

```rust
trait Eq {
    fn eq(self, other: Self) -> i32;
}

impl Eq for i32 {
    fn eq(self, other: i32) -> i32 {
        if self == other { 1 } else { 0 }
    }
}

fn cmp<T: Eq>(a: T, b: T) -> i32 {
    T::eq(a, b)
}

fn main() -> i32 {
    cmp::<i32>(5, 5)   // 1
}
```

Trait bounds are resolved at monomorphization. Method-call sugar `a.eq(b)` lowers to `i32__eq(a, b)` via the impl table.

#### Closures (Stage 9)

```rust
fn main() -> i32 {
    let a = 10;
    let c = |x| x + a;     // captures `a` by value
    c(5)                   // 15
}
```

Closures lower at parse time to a synthetic struct (env) plus a synthetic function. Captured variables become struct fields. Phase-0 cap: 4 captures per closure, nesting depth 1.

#### Function attributes

| Attribute | Effect |
|-----------|--------|
| `@pure` | Function has no side effects. Required for AD bodies and `verifier` blocks. |
| `@effect(io.read_file)` | Declares an effect. Effect-check pass enforces effect propagation. |
| `@partial` | Function may not terminate. Required for Collatz-style functions. |
| `@total` | Function is provably total (default; explicit form for documentation). |
| `@checkpoint` | In reverse-mode AD, recompute this segment instead of saving activations. |
| `@kernel` | Emit as PTX kernel (GPU). |
| `@autotune(grid=[...])` | Generate parameter sweeps; pick fastest. |
| `@deprecated("reason")` | Compile warning when called. |
| `@since("v0.3")` | Documentation marker. |

### Structs (Stage 5)

#### Declaration and construction

```rust
struct Pt { x: i32, y: i32 }
struct Line { from: Pt, to: Pt }   // nested

fn main() -> i32 {
    let p = Pt { 10, 32 };          // positional construction
    let q = Pt { x: 10, y: 32 };    // named construction (Phase-1)
    let l = Line { Pt { 1, 2 }, Pt { 3, 4 } };
    p.x + p.y + l.from.x + l.to.y
}
```

#### Field access

- Positional: `p.0`, `p.1`
- Named: `p.x`, `p.y`
- Chained: `l.from.x`

#### Pass-by-value to functions

```rust
fn area(p: Pt) -> i32 {
    p.x * p.y
}

fn main() -> i32 {
    area(Pt { 6, 7 })   // 42
}
```

The SysV ABI is followed: structs ≤ 16 bytes pass via pointer in `rdi`; larger via memory. Helix's bootstrap implementation uses pointer-pass for all struct args (simpler ABI; same result).

### Enums (Stage 6)

#### Declaration

```rust
enum Color { Red, Green, Blue }
enum Maybe { None, Some(i32) }
enum Tree { Leaf(i32), Node(i32, i32) }    // multi-payload variants
```

Variants are 0-indexed by declaration order. `Color::Red == 0`, `Color::Green == 1`, `Color::Blue == 2`.

#### Construction

```rust
let c = Color::Green;                  // discriminant value: 1
let m = Maybe::Some(42);               // (discriminant: 1, payload: 42)
let n = Tree::Node(3, 5);              // (discriminant: 1, payload: (3, 5))
```

Unit variants with no payload are represented as plain `i32` (the discriminant). Payload variants are 16-byte structs (discriminant + payload).

#### Discriminant and payload access

```rust
let m = Maybe::Some(42);
let value = __enum_payload(m, 0);     // 42
```

But typically you use `match`:

```rust
match m {
    Maybe::None    => 0,
    Maybe::Some(v) => v,
}
```

### Modules (Stage 10)

```rust
mod geometry {
    fn area(w: i32, h: i32) -> i32 {
        w * h
    }

    mod circle {
        fn area(r: i32) -> i32 {
            r * r * 314 / 100
        }
    }
}

use geometry::area;
use geometry::circle::area as circle_area;

fn main() -> i32 {
    area(3, 4) + circle_area(5)
}
```

Modules flatten at parse time: `geometry::area` becomes the mangled function `geometry__area`. `use` aliases to the mangled name.

### Reflection (Stage 11)

Helix has language-level reflection via `Quote`, `Splice`, and `modify`. Cells are runtime-mutable AST handles.

```rust
fn always_true(_: i32) -> i32 { 1 }

fn main() -> i32 {
    let h = Quote(1 + 2);                       // cell handle to `1 + 2`
    let initial = Splice(h);                    // 3
    modify(h, 42, always_true(0));              // verifier-gated mutation
    let updated = Splice(h);                    // 42
    initial + updated                            // 45
}
```

- `Quote(expr)` — register a cell, return its handle (i32)
- `Splice(handle)` — fetch the cell's current value
- `modify(handle, new_value, verifier_expr)` — if verifier returns truthy, update the cell

Phase-0 stores cell *values* (not full ASTs). Phase-1 will support full AST cells with `__quote_hash` (FNV-1a Phase-0; SHA-256 Stage 17) for content addressing.

---

## Standard Library

The Helix standard library lives in `helixc/stdlib/` and is written in Helix itself (no `unsafe`, no FFI for core operations).

### Core (`std::core`)

- `__abs_i32(x) -> i32` — absolute value
- `__sign_i32(x) -> i32` — -1 / 0 / 1
- `__sign_f64(x) -> f64`
- `__min_i32(a, b)`, `__max_i32(a, b)`, `__clamp_i32(v, lo, hi)`
- `__min_f32`, `__max_f32`, `__clamp_f32` (and f64 variants)

### IEEE 754 introspection (`std::ieee754`)

- `__bits_of_f32(x: f32) -> i32` — bit pattern
- `__f32_of_bits(b: i32) -> f32`
- `__bits_hi_f64(x: f64) -> i32`, `__bits_lo_f64(x: f64) -> i32`
- `__f64_pack(hi: i32, lo: i32) -> f64`
- `__f64_to_f32(x: f64) -> f32`, `__f32_to_f64(x: f32) -> f64`
- `__f64_to_i32(x: f64) -> i32` (truncation)
- `f32_bits_zero()`, `f32_bits_one()`, `f32_bits_neg(b: i32)`

### Math (`std::math`)

- `__exp(x: f64) -> f64`, `__log(x: f64) -> f64`
- `__sin(x: f64) -> f64`, `__cos(x: f64) -> f64`, `__tan(x: f64) -> f64`
- `__sqrt(x: f64) -> f64`
- `__powi(x: f64, n: i32) -> f64` — integer power, n cap 16
- `__sigmoid(x: f64) -> f64`, `__relu(x: f64) -> f64`, `__tanh(x: f64) -> f64`

All transcendentals participate in the autodiff chain rule.

### Autodiff (`std::autodiff`)

- `grad(f)(x)` — forward-mode derivative
- `grad_rev_all(f)(x, y, z) -> Grad { dx, dy, dz }` — reverse-mode all gradients
- `d_abs(x)`, `d_max_const(x, c)`, `d_min_const(x, c)`, `d_sub_const(x, c)` — convenience derivatives

### Neural network primitives (`std::nn`)

- `argmin(arr, n) -> i32`, `argmax(arr, n) -> i32`
- `mae_loss_f32(pred, target) -> f32`, `mae_loss_f64`
- `mse_f64(pred, target) -> f64`
- `count_correct(pred, target, n) -> i32`

### AGI search primitives (`std::agi_search`)

- `bfs_is_empty`, `bfs_push`, `bfs_pop` — BFS frontier
- `pq_is_empty`, `pq_peek_min`, `pq_push`, `pq_pop` — priority queue
- `visited_count`, `visited_mark`, `visited_check`

### Option / Result (`std::option`)

- `enum Option<T> { None, Some(T) }`
- `option_min`, `option_sum`, `option_eq`, `option_or_one`

### Collections (`std::vec`, `std::hashmap`, planned)

Phase-1 introduces `Vec<T>`, `HashMap<K, V>`, `String`, iterators with method-chain syntax.

---

## Autodiff

Autodiff is built into the language. Both forward and reverse modes are supported, plus `@checkpoint` for memory-efficient reverse-mode.

### Forward mode (Stage 12)

Compute the derivative `df/dx` at a point:

```rust
fn loss(x: f64) -> f64 { x * x + 3.0_f64 * x }

fn main() -> f64 {
    grad(loss)(2.0_f64)   // 7.0  (derivative is 2x + 3, eval at x=2)
}
```

Differentiation rules implemented:

| Rule | Form |
|------|------|
| Constant | `d(c) = 0` |
| Variable | `d(x) = 1` if x is the diff variable, else 0 |
| Linearity | `d(a + b) = d(a) + d(b)`, `d(a - b) = d(a) - d(b)` |
| Product | `d(a * b) = d(a) * b + a * d(b)` |
| Quotient | `d(a / b) = (d(a) * b - a * d(b)) / (b * b)` |
| Negation | `d(-a) = -d(a)` |
| Chain (transcendentals) | `d(f(g(x))) = f'(g(x)) * d(g(x))` |
| User-defined | `d(g(x))` inlines `g` body, then differentiates |

Algebraic simplifier folds `0+x → x`, `x*1 → x`, `0*x → 0`, `-(-x) → x`, etc.

### AD across user-defined functions (Stage 13)

Helper function calls inline at differentiation time:

```rust
fn helper(x: f64) -> f64 { x * x }
fn loss(x: f64) -> f64 { helper(x) + x }

fn main() -> f64 {
    grad(loss)(3.0_f64)   // 7.0  (d(x^2 + x) = 2x + 1, at x=3)
}
```

Recursion is detected and trapped (87001) to avoid infinite inlining. Mutual recursion through purity inference.

### Reverse mode (Stage 14)

Compute all gradients in one backward sweep:

```rust
fn loss(x: f64, y: f64) -> f64 { x * y + x * x }

struct Grad { dx: f64, dy: f64 }

fn main() -> f64 {
    let g = grad_rev_all(loss)(2.0_f64, 3.0_f64);
    g.dx   // 7.0  (∂/∂x (xy + x²) = y + 2x = 3 + 4 = 7)
}
```

Reverse mode is implemented as adjoint propagation through the AST. Each parameter has a bucket; the loss body is walked top-down with the current adjoint expression, and at each leaf binding `Name(param)`, the adjoint is appended to that param's bucket. Buckets are summed at the end.

### `@checkpoint` rematerialization (Stage 14.5)

In reverse mode, `@checkpoint` segments are NOT cached during forward — instead, they are recomputed during backward. Trades compute for memory.

```rust
@checkpoint
fn deep_block(x: f64) -> f64 { x * x * x * x * x }

fn loss(x: f64) -> f64 { deep_block(x) + x }

fn main() -> f64 {
    grad_rev_all(loss)(2.0_f64).dx   // 81  (5x⁴ + 1, at x=2)
}
```

Phase-0 enforces that `@checkpoint` bodies are pure; trap 90001 fires on impure checkpoints.

---

## Tile & Tensor Types

Helix has tiles and tensors as first-class types — no NumPy-style runtime dispatch. Each tile knows its dtype, shape, and memspace at compile time.

### Tile syntax (Stage 15)

```rust
let a = tile<f32, [4, 4], REG>::ones();
let b = tile<f32, [4, 4], REG>::zeros();
let c = tile_matmul(a, b);
let v: f32 = c.get(0, 0);   // 0.0
```

#### Type parameters

- **Dtype**: `f32`, `f64`, `i32` for the covered live paths; `bf16` is a
  recognized target surface under active backend work.
- **Shape**: `[rows, cols]` — compile-time integer tuple
- **Memspace**: `REG` (registers/stack) is the covered live path. `SMEM`
  (shared memory), `HBM` (global memory), and `TMEM` (Blackwell tensor memory)
  are target design surfaces for later PTX/GPU stages.

Phase-0 supports REG only. The current PTX backend emits text for covered
kernels, but GPU execution and non-REG memory-space behavior are not shipped
capabilities yet.

### Operations

| Operation | Description |
|-----------|-------------|
| `tile<...>::zeros()` | Allocate tile, init to 0 |
| `tile<...>::ones()` | Allocate tile, init to 1 |
| `tile_load(tensor, [row, col])` | Load tile from larger tensor at offset |
| `tile_store(tile, tensor, [row, col])` | Store tile back to tensor |
| `tile_matmul(a, b) -> c` | Matrix multiplication |
| `t.get(row, col)` | Single-element load |

### Tensor type (Phase-1)

`tensor<f32, [N, M]>` is the dynamically-shaped abstract type. Shape and memspace are inferred per use. Tensors are typically lowered to tile chunks at codegen.

### Compile-time shape checks

```rust
let a = tile<f32, [4, 4], REG>::ones();
let b = tile<f32, [3, 3], REG>::ones();
let c = tile_matmul(a, b);   // ERROR at compile time: shape mismatch
```

Trap 95001 fires for runtime-detected matmul shape mismatches.

---

## Foreign Function Interface (FFI)

Helix interoperates with C via `extern "C"` declarations (Stage 16.5).

### Declaring an external function

```rust
extern "C" fn puts(s: *const u8) -> i32;
extern "C" fn malloc(size: u64) -> *mut u8;
extern "C" fn free(ptr: *mut u8);
```

### Calling external functions

```rust
extern "C" fn puts(s: *const u8) -> i32;

fn main() -> i32 {
    let msg: *const u8 = "hello\n\0".as_ptr();
    puts(msg)   // prints "hello", returns 6
}
```

### `repr(C)` structs

```rust
#[repr(C)]
struct CRect { x: i32, y: i32, w: i32, h: i32 }

extern "C" fn draw_rect(r: CRect) -> i32;
```

`repr(C)` enforces C-ABI layout: natural alignment per field, no field reordering. Without it, Helix uses a uniform 8-byte slot model.

### How it works

The Helix compiler emits an ELF binary with a `.dynsym` section and `R_X86_64_PLT32` relocations for FFI calls. At link time, the dynamic linker resolves symbols against `libc.so.6` (or any shared library specified via `-l`).

The pure Helix bootstrap binary (no FFI) has zero shared-library dependencies. Only user programs that explicitly use `extern "C"` get a libc dependency.

---

## Reflection Runtime

Stage 11 introduces reflection cells — runtime-mutable AST handles. Useful for symbolic AI, reactive programming, and verifier-gated code rewrites.

### Quote, Splice, modify

```rust
fn always_true(_: i32) -> i32 { 1 }

fn main() -> i32 {
    let h = Quote(1 + 2);                  // cell handle (i32)
    Splice(h)                              // 3 — load cell value
    
    modify(h, 42, always_true(0));         // verifier-gated update
    Splice(h)                              // 42 — verifier passed
}
```

### Verifier semantics

- The verifier expression is evaluated.
- If the result is non-zero (truthy), the cell is updated.
- If the result is zero, the modify is silently a no-op.
- Trap 84001 fires if the verifier expression has side effects (Phase-1).

### Hash-based content addressing

`__quote_hash(handle) -> i32` returns a structural hash of the cell's current AST. Used for caching and de-duplication. Phase-0 uses FNV-1a; Stage 17 hash-cons upgrades to SHA-256.

### Use cases

- **Reactive computation**: a cell's value depends on others; modifying triggers recomputation.
- **Verified code rewrite**: a cell holds a pure function body; a verifier ensures rewrites preserve semantics.
- **Memoization**: identical computations share cells (content-addressed).

---

## Compilation Pipeline

```
┌──────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌────────┐
│  Source  │ -> │  Lexer  │ -> │ Parser  │ -> │   IR    │ -> │  ELF   │
│  (.hx)   │    │ (tokens)│    │  (AST)  │    │ (TIR)   │    │ binary │
└──────────┘    └─────────┘    └─────────┘    └─────────┘    └────────┘
                                    │              │
                                    │              ▼
                              monomorph        const-fold
                              flatten          CSE / DCE
                              modules          effect check
                              closures         hash-cons
                              grad_pass        totality
                              grad_rev_pass
                              reflect_pass
```

### Lexer (Stage 0)

- Tokenizes source bytes into a stream of `(tag, payload, src_start, src_len)` 4-tuples
- Token tags: `INT`, `IDENT`, `FLOATLIT`, `LPAREN`, `RPAREN`, `LBRACE`, `RBRACE`, `LBRACK`, `RBRACK`, `COMMA`, `COLON`, `SEMI`, `DOT`, `EQ`, `LT`, `GT`, `PLUS`, `MINUS`, `STAR`, `SLASH`, `PERCENT`, `AMP`, `PIPE`, `CARET`, `TILDE`, `BANG`, `AT`, `FATARROW`, `DOTDOT`, `DOTDOTEQ`, `COLONCOLON`, etc.
- Handles all 12 numeric type suffixes (`_i32`, `_i64`, `_u8`, …, `_bf16`)
- Multi-line string literals and full escape sequences (`\n`, `\t`, `\\`, `\"`, `\xHH`)

### Parser (Stage 1)

- Recursive-descent with classic precedence climbing
- Builds an arena-allocated AST (each node is a 4-7 slot record)
- 100+ AST tags covering literals, binops, unary ops, control flow, fn decls, struct/enum decls, patterns, generics, traits, impls, modules, closures, reflection, autodiff calls, tile ops
- Pattern-match parser handles all 9 pattern kinds with guards
- Generic parser handles turbofish disambiguation (`f::<T>(x)` is a generic call; `f<T>(x)` is `(f < T) > (x)`)

### Monomorphization pass

- Pre-codegen pass that walks the AST_FN_LIST
- For each generic call (turbofish), clones the template fn with type substitution
- Builds the mono_table mapping `(orig_name, [type_args])` to mangled name
- Handles trait-bound resolution via the impl_table

### Module flatten pass

- Lifts nested module items to top-level with mangled names: `mod foo { fn bar }` → `fn foo__bar`
- Resolves `use` aliases

### Closure lower pass

- Identifies free variables in closure bodies
- Synthesizes anonymous struct (env) + fn (body)
- Replaces closure literal with env construction

### Autodiff passes

- `grad_pass` (forward) walks the AST, finds `grad(loss)(args)` calls, generates `loss__grad` synthetic fn
- `grad_reverse_pass` (reverse) does the same for `grad_rev_all(loss)(args)`, returning a struct of all gradients

### Reflection pass

- Allocates compile-time cell handles for `Quote(expr)` calls
- Embeds cell-table region in the produced binary's BSS

### Optimization passes

| Pass | Stage | Purpose |
|------|-------|---------|
| const-fold | 17 | `2 + 3 → 5`, algebraic identities |
| CSE | 18 | Common subexpression elimination |
| DCE | 18 | Dead code elimination |
| FDCE | 18 | Float dead code elimination (handles NaN edge cases) |
| effect-check | 19 | Verify `@effect(io.read_file)` propagation |
| hash-cons | 20 | Structural sharing of identical AST subtrees |
| totality-check | 21 | Structural recursion check; flag non-`@partial` non-terminating fns |

### Codegen

- **x86-64**: emits System V ABI ELF binaries directly (no link step). All instruction encodings hand-rolled (`48 89 45 F8 = mov [rbp-8], rax`).
- **PTX**: emits NVIDIA PTX text for `@kernel` fns; embedded in `.rodata` of host binary.
- **WebAssembly** (Phase-2): planned for browser playground.

---

## Bootstrap Chain

The trust root of Helix is **299 bytes of hand-encoded x86-64 machine code**: `hex0`.

```
                    ┌─────────────────┐
                    │  hex0 (299 B)   │  hand-encoded; "0123456789ABCDEF" + space
                    └────────┬────────┘
                             │ compiles
                             ▼
                    ┌─────────────────┐
                    │  hex1 (~700 B)  │  recognizes labels and references
                    └────────┬────────┘
                             │ compiles
                             ▼
                    ┌─────────────────┐
                    │  M0 (~3 KB)     │  minimal macro assembler
                    └────────┬────────┘
                             │ compiles
                             ▼
                    ┌─────────────────┐
                    │  M1 (~8 KB)     │  proper macro assembler with directives
                    └────────┬────────┘
                             │ compiles
                             ▼
                    ┌─────────────────┐
                    │  M2-Planet      │  C compiler (ANSI C subset)
                    │     (~30 KB)    │
                    └────────┬────────┘
                             │ compiles
                             ▼
                    ┌─────────────────┐
                    │ kovc-bootstrap  │  initial Helix compiler in C
                    │     (~80 KB)    │
                    └────────┬────────┘
                             │ compiles
                             ▼
                    ┌─────────────────┐
                    │  helixc         │  current Python-hosted compiler
                    └─────────────────┘  (Helix self-host remains target)
```

Current bootstrap root to audit: **299 bytes**. Later self-hosting steps remain roadmap targets until the Helix compiler can compile itself reproducibly.

This is in the spirit of [Bootstrappable Builds](https://bootstrappable.org) and [GNU Mes](https://www.gnu.org/software/mes/), but built from scratch for an ML-focused language.

---

## Compiler Architecture

### Repo layout

```
Kovostov-Native/
├── stage0/             # hex0, hex1, M0, M1, M2-Planet
│   ├── hex0/           # 299-byte audited bootstrap root
│   ├── hex1/
│   └── ...
├── helixc/             # the Helix compiler
│   ├── bootstrap/      # kovc.hx (self-host target, in Helix)
│   │   ├── lexer.hx
│   │   ├── parser.hx
│   │   └── kovc.hx     # codegen + IR + driver
│   ├── frontend/       # Python reference implementation (used during dev)
│   │   ├── lexer.py
│   │   ├── parser.py
│   │   ├── typecheck.py
│   │   ├── monomorphize.py
│   │   ├── flatten_modules.py
│   │   ├── flatten_impls.py
│   │   ├── autodiff.py
│   │   ├── autodiff_reverse.py
│   │   └── totality.py
│   ├── ir/             # IR layer
│   │   ├── tir.py      # Tensor IR ops + use-def
│   │   ├── lower_ast.py
│   │   ├── tile_ir.py
│   │   └── passes/
│   │       ├── const_fold.py
│   │       ├── cse.py
│   │       ├── dce.py
│   │       └── fdce.py
│   ├── backend/        # codegen targets
│   │   ├── x86_64.py   # ELF + SysV ABI
│   │   ├── ptx.py      # NVIDIA PTX
│   │   └── elf_dyn.py  # FFI dynamic linking
│   ├── stdlib/         # standard library, in Helix
│   │   ├── core.hx
│   │   ├── ieee754.hx
│   │   ├── math.hx
│   │   ├── nn.hx
│   │   ├── option.hx
│   │   └── autodiff.hx
│   ├── tests/          # 2,304 tests collected in restart 27 fix verification
│   │   ├── test_codegen.py
│   │   ├── test_parser.py
│   │   ├── test_match.py
│   │   ├── test_reflection.py
│   │   ├── test_ffi.py
│   │   └── ...
│   └── examples/
│       ├── hbs_sample_calculator.hx
│       ├── hbs_sample_loss_fn.hx
│       └── hbs_sample_enum_struct.hx
├── docs/
│   ├── lang/hbs.md     # Helix Bootstrap Subset spec
│   └── audit-stage4-followup.md
└── helix_website/      # this directory; the public website
```

### Why a Python reference + a Helix self-host?

The Python implementation (`helixc/frontend/...`) is the *executable specification*. It runs the test suite, produces reference outputs, and is used to bootstrap the initial `kovc-bootstrap` binary.

The Python-hosted `helixc` is currently the production compiler. The Helix self-host (`helixc/bootstrap/kovc.hx`) is the bootstrap target: once it can compile itself and user programs reproducibly, Python can become a reference implementation instead of the main compiler.

### Audit cycles

Each stage of Helix went through multi-agent audit cycles. Three specialist agents (code-reviewer, silent-failure-hunter, type-design-analyzer) review every commit. Findings are tracked in `docs/audit-stage4-followup.md` style — each finding has a unique ID, severity, reproducer, status, and resolution commit.

Stage 30 historically used **5 consecutive clean audits** with zero new
findings. Stage 35 uses a faster **3-clean-gate** policy after each fix sweep;
the current Stage 35 ledger remains `0/3` until a fresh restart produces no new
findings.

---

## Tooling & CLI

### `kovc` — the compiler driver

```
kovc <source.hx> [options]

Options:
  -o <file>             Output binary path (default: a.out)
  --emit-ir             Emit textual IR to stdout, don't generate binary
  --emit-asm            Emit x86-64 assembly listing
  --emit-ptx            Emit PTX text for @kernel fns
  --dump-ast-hashes     Print structural hashes of AST nodes (for hash-cons)
  --check-only          Parse + typecheck + totality, no codegen
  --no-bootstrap-cache  Disable bootstrap binary caching (testing)
  --target=x86_64       (default) emit x86-64 ELF
  --target=wasm32       emit WebAssembly module (Phase-2)
  -O0 / -O1 / -O2 / -O3 Optimization level
  -l <libname>          Link external library (FFI)
  --version
  --doc                 Extract /// doc comments to markdown
```

### `kovc check` — fast typechecker

For editor integrations: parses and typechecks without emitting code.

### `kovc fmt` — code formatter (Phase-1)

### `kovc test` — test runner (Phase-1)

### Helix Language Server (Phase-1)

LSP server providing diagnostics, hover, go-to-definition, completion. Built on top of `kovc check`.

### Diagnostics

Errors include source-with-caret display:

```
error[64001]: pattern type mismatch
   ┌─ src/main.hx:5:15
   │
 5 │     match x { 0.5_f64 => 1, _ => 0 }
   │               ^^^^^^^ expected i32, found f64
   │
   │ help: cast the literal: `0.5_f64 as i32`
```

Did-you-mean suggestions for stdlib calls:

```
error: undefined function `__exf`

   help: did you mean `__exp`?
```

---

## Open-Source Commitments

### Licenses

- **Source code**: Apache 2.0
- **Documentation**: CC-BY 4.0
- **Logos and brand**: CC-BY 4.0
- **Model weights** (when shipped): CC0 (public domain)

### Training data policy

The Kovostov AGI project (which Helix is the foundation for) commits to training only on publicly licensed or public-domain data. No GPT, Claude, or Gemini outputs in the training set. No copyrighted code without explicit license. Full provenance for every byte.

### Reproducibility

- Deterministic rebuilds are a verification target for the compiler and bootstrap chain.
- The live `hex0` bootstrap root is byte-auditable today. Rebuilding the full
  compiler from `hex0` is a reproducibility target until the later bootstrap
  links are implemented and verified.
- All training data manifests are public and content-addressed.

### Governance

- **BDFL**: project initiator (currently the Kovostov-Native author).
- **RFC process**: language changes require an RFC issue + 14-day comment period.
- **Code review**: every commit requires at least one reviewer.
- **Audit cycles**: every stage transition requires multi-agent audit + clean cycle.

---

## Comparisons

### Helix vs Rust

| Dimension | Helix | Rust |
|-----------|-------|------|
| Bootstrap | Live 299-byte `hex0` root; full chain target | Requires LLVM, GCC |
| Autodiff | Built-in | External crate (`burn`, `dfdx`) |
| GPU | PTX backend in language | External (`cudarc`, etc.) |
| Memory model | Region/arena (Phase-0) | Borrow checker |
| Macros | Reflection (Quote/Splice) | `macro_rules!` + proc macros |
| Compile time | <1s for 10K LOC | 30s+ |

### Helix vs Mojo

| Dimension | Helix | Mojo |
|-----------|-------|------|
| License | Apache 2.0, fully open | Proprietary (partial) |
| Bootstrap | Self-host target growing from live `hex0` | Closed-source binary |
| Tile types | First-class | First-class |
| Autodiff | Built-in | External (uses MAX engine) |
| Backend | x86 + PTX (planned) | x86 + GPU via MLIR |

### Helix vs Triton

| Dimension | Helix | Triton |
|-----------|-------|--------|
| Scope | General-purpose ML language | GPU kernel DSL only |
| Host code | Same language | Python only |
| Autodiff | Yes | No (use `torch.autograd`) |
| Type system | Full | Limited |

### Helix vs Python + JAX

| Dimension | Helix | Python + JAX |
|-----------|-------|--------------|
| Performance | x86 native | XLA via Python overhead |
| Static typing | Yes | Optional (mypy) |
| Compile-time errors | Yes | No |
| Distribution | Single binary | `pip install` cascade |
| Bootstrap | 299-byte live root; self-hosting target | Depends on Python ecosystem |

---

## Code Samples Gallery

Use these directly on the website. Each is small, self-contained, and demonstrates a feature.

### #1 — Hello, 42 (the canonical first program)

```rust
fn main() -> i32 { 42 }
```

### #2 — Arithmetic and let bindings

```rust
fn main() -> i32 {
    let x = 5;
    let y = x * x;
    y - 8           // 17
}
```

### #3 — Control flow

```rust
fn fib(n: i32) -> i32 {
    if n < 2 { n }
    else { fib(n - 1) + fib(n - 2) }
}

fn main() -> i32 { fib(10) }   // 55
```

### #4 — Mutable state and while loops

```rust
fn main() -> i32 {
    let mut sum = 0;
    let mut i = 1;
    while i <= 10 {
        sum = sum + i;
        i = i + 1;
    }
    sum   // 55
}
```

### #5 — Tuples and field access

```rust
fn main() -> i32 {
    let t = (10, 20, 30);
    t.0 + t.1 + t.2   // 60
}
```

### #6 — Arrays

```rust
fn main() -> i32 {
    let arr = [1, 2, 3, 4, 5];
    arr[0] + arr[2] + arr[4]   // 9
}
```

### #7 — Structs

```rust
struct Pt { x: i32, y: i32 }

fn area(p: Pt) -> i32 {
    p.x * p.y
}

fn main() -> i32 {
    area(Pt { 6, 7 })   // 42
}
```

### #8 — Nested structs

```rust
struct Pt { x: i32, y: i32 }
struct Line { from: Pt, to: Pt }

fn main() -> i32 {
    let l = Line { Pt { 10, 0 }, Pt { 0, 32 } };
    l.from.x + l.to.y   // 42
}
```

### #9 — Enums with payloads

```rust
enum Maybe { None, Some(i32) }

fn main() -> i32 {
    let m = Maybe::Some(42);
    match m {
        Maybe::None    => 0,
        Maybe::Some(v) => v,
    }
}
```

### #10 — Pattern matching with ranges

```rust
fn classify(n: i32) -> i32 {
    match n {
        0       => 100,
        1..10   => 200,
        10..=20 => 300,
        _       => 999,
    }
}

fn main() -> i32 { classify(15) }   // 300
```

### #11 — Tuple patterns

```rust
fn main() -> i32 {
    let p = (1, 2);
    match p {
        (0, _) => 100,
        (1, y) => y,
        _      => 0,
    }
}
```

### #12 — Generic functions

```rust
fn id<T>(x: T) -> T { x }

fn main() -> i32 {
    id::<i32>(42)
}
```

### #13 — Traits

```rust
trait Eq {
    fn eq(self, other: Self) -> i32;
}

impl Eq for i32 {
    fn eq(self, other: i32) -> i32 {
        if self == other { 1 } else { 0 }
    }
}

fn main() -> i32 {
    let a = 5;
    let b = 5;
    a.eq(b)   // 1
}
```

### #14 — Closures

```rust
fn main() -> i32 {
    let a = 10;
    let c = |x| x + a;
    c(5)   // 15
}
```

### #15 — Modules

```rust
mod geometry {
    fn rect_area(w: i32, h: i32) -> i32 { w * h }
}

fn main() -> i32 {
    geometry::rect_area(6, 7)   // 42
}
```

### #16 — Forward-mode autodiff

```rust
fn loss(x: f64) -> f64 { x * x + 3.0_f64 * x }

fn main() -> f64 {
    grad(loss)(2.0_f64)   // 7.0
}
```

### #17 — Reverse-mode autodiff

```rust
fn loss(x: f64, y: f64) -> f64 { x * y + x * x }

struct Grad { dx: f64, dy: f64 }

fn main() -> f64 {
    grad_rev_all(loss)(2.0_f64, 3.0_f64).dx   // 7.0
}
```

### #18 — Tile matmul

```rust
fn main() -> f32 {
    let a = tile<f32, [4, 4], REG>::ones();
    let b = tile<f32, [4, 4], REG>::ones();
    let c = tile_matmul(a, b);
    c.get(2, 3)   // 4.0
}
```

### #19 — Reflection (Quote/Splice/modify)

```rust
fn always_true(_: i32) -> i32 { 1 }

fn main() -> i32 {
    let h = Quote(0);
    modify(h, 42, always_true(0));
    Splice(h)   // 42
}
```

### #20 — FFI (calling libc)

```rust
extern "C" fn puts(s: *const u8) -> i32;

fn main() -> i32 {
    let msg: *const u8 = "hello\n\0".as_ptr();
    puts(msg)
}
```

---

## Glossary

| Term | Meaning |
|------|---------|
| **kovc** | The Helix compiler binary (named after Kovostov). |
| **kovc.hx** | The Helix source file that, when compiled by kovc, produces kovc. (Self-hosting.) |
| **hex0** | The 299-byte hand-encoded x86-64 program at the root of the bootstrap chain. |
| **HBS** | Helix Bootstrap Subset — the minimal language subset that kovc.hx itself uses. Documented in `docs/lang/hbs.md`. |
| **AST tag** | An i32 numeric ID for an AST node kind. 100+ tags total. |
| **Trap-id** | A unique numeric ID embedded after a `ud2` instruction; `AST_TAG * 1000 + sub_id`. |
| **Mono pass** | Monomorphization — generates concrete versions of generic functions. |
| **Adjoint** | In reverse-mode AD, the gradient flowing backward through a node. |
| **Tile** | A small fixed-shape, fixed-dtype, fixed-memspace array. |
| **PTX** | NVIDIA's virtual ISA for GPU kernels. Helix emits PTX text directly. |
| **REG / SMEM / HBM / TMEM** | Memspace tags: registers, shared memory, global memory, tensor memory. |
| **AD** | Automatic differentiation (forward or reverse mode). |
| **`@checkpoint`** | Function attribute: in reverse-mode AD, recompute this segment instead of caching activations. |
| **`@pure`** | Function attribute: no side effects. Required for AD bodies and verifier blocks. |

---

## Visual Identity Suggestions

For the website, here's a visual direction proposal — Claude Design can adapt freely.

### Aesthetic axis 1: "from raw metal"

- **Colors**: deep terminal black (#0A0E14) base, hex-digit green (#7FE83A) accent, off-white (#E6E6E6) text
- **Typography**: monospace (JetBrains Mono / IBM Plex Mono) for both headlines and body
- **Background motif**: ambient hex digits (00 1F 80 5C 4D ...) scrolling slowly behind sections
- **Imagery**: byte-level diagrams, CPU register tables, ELF section layouts

### Aesthetic axis 2: "scientific notebook"

- **Colors**: warm off-white (#FAF8F1) base, ink blue (#1A2B4A) accent, terra-cotta (#C26543) callout
- **Typography**: serif body (Lora / Source Serif), monospace code (Fira Code), KaTeX for math
- **Background motif**: subtle paper texture, hand-drawn-feel diagrams
- **Imagery**: gradient flow diagrams (∂L/∂x), mathematical equations, arrows

### Aesthetic axis 3: "futuristic minimalism" (recommended for broad appeal)

- **Colors**: pure white (#FFFFFF) or near-black (#0F1117) base, single accent color (electric purple #8B5CF6 or cyan #06B6D4)
- **Typography**: sans-serif headlines (Inter / Geist), monospace code (JetBrains Mono)
- **Background motif**: very subtle grid, generous whitespace
- **Imagery**: polished diagrams with lots of negative space, gradient highlights on key terms

### Distinctive visual elements (must-haves regardless of axis)

1. **The byte counter** — animated counter at the top: "Built from 299 bytes". Counts up on scroll.
2. **The bootstrap chain explorer** — interactive: click each link in the chain (hex0 → ... → kovc), see its size, its source, what it produced.
3. **Compilation animation** — recurring motif: tokens flow into trees, trees flow into IR, IR flows into hex bytes. Use it on Hero, Features, How-it-works.
4. **Math notation** — KaTeX for autodiff page. Render `∂/∂x (x² + 3x) = 2x + 3` properly.
5. **Hex byte viewer** — when showing produced binaries, render them as a beautiful hex dump (like `xxd` but styled).
6. **Trap-id callouts** — small ⚠️ cards for each silent-corruption bug we found and fixed. Builds trust.

### Logo concept

Two elements interleaved:
- A double-helix (DNA-like) representing the language name
- A hex-byte trail showing the bootstrap (e.g., `48 89 E5 ...`)

Or: a single character `λ` in monospace inside a hex bracket `[λ]`. Clean, short, suggests "function + low-level".

---

## Page-by-Page Wireframe Suggestions

### Landing (`/`)

```
┌─────────────────────────────────────────────────────────────┐
│  Helix.dev                          Docs · Playground · GH  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   A compiler that builds itself                             │
│   from 299 bytes of hex.                                    │
│                                                             │
│   The open-source ML language with autodiff,                │
│   tile types, and GPU codegen — all built                   │
│   on a fully reproducible bootstrap.                        │
│                                                             │
│   [ Try it in your browser ]   [ Read the docs ]            │
│                                                             │
│   ┌───────────────────┐   ┌─────────────────────────┐       │
│   │ fn loss(x: f64)   │   │  Tokens → AST → IR →    │       │
│   │   -> f64 {        │ → │  x86 bytes → exec       │       │
│   │     x*x + 3*x     │   │                         │       │
│   │ }                 │   │  Result: 7.0_f64        │       │
│   │                   │   │                         │       │
│   │ grad(loss)(2.0)   │   │  (animated)             │       │
│   └───────────────────┘   └─────────────────────────┘       │
│                                                             │
│   🔓 Open · ⚙️ Bootstrapped · 🧠 ML-first                   │
│                                                             │
│   Built from 299 bytes toward a self-hosted compiler        │
│   ┌───────────────────────────────────────────────────┐     │
│   │ ▓ 299 ▓▓▓ 700 ▓▓▓▓▓ 3K ▓▓▓▓▓▓▓ 8K ▓▓▓▓▓▓▓▓▓ 30K  │     │
│   └───────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Playground (`/playground`)

```
┌─────────────────────────────────────────────────────────────┐
│  Helix.dev > Playground                                     │
├──────────────────────────┬──────────────────────────────────┤
│                          │  Tokens │ AST │ IR │ Bytes │ Out │
│  fn main() -> i32 {      │ ┌────────────────────────────┐   │
│    let x = 5;            │ │ INT(5) IDENT(x) ...        │   │
│    x * x                 │ │                            │   │
│  }                       │ │ AST_LET                    │   │
│                          │ │  ├── name: x               │   │
│  [ Run ▶ ]               │ │  ├── value: AST_INT(5)     │   │
│  [ Examples ▼ ]          │ │  └── body:                 │   │
│  [ Share 🔗 ]            │ │      AST_MUL               │   │
│                          │ │      ├── AST_VAR(x)        │   │
│                          │ │      └── AST_VAR(x)        │   │
│                          │ └────────────────────────────┘   │
│                          │  Output: 25                      │
└──────────────────────────┴──────────────────────────────────┘
```

### Bootstrap chain (`/bootstrap-chain`)

```
┌─────────────────────────────────────────────────────────────┐
│  How Helix grows from 299 bytes                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ●────●────●────●────●────●────●                           │
│  hex0 hex1  M0   M1  M2  kovc-bs kovc                       │
│  299B 700B  3KB  8KB 30KB  80KB  target                     │
│                                                             │
│   ┌───────────────────────────────────────┐                 │
│   │  hex0: 299 bytes                      │                 │
│   │                                       │                 │
│   │  31 C0 B8 ...                         │                 │
│   │  (full hex dump shown)                │                 │
│   │                                       │                 │
│   │  This program reads hex digits        │                 │
│   │  from stdin and writes their byte     │                 │
│   │  values to stdout. That's it.         │                 │
│   │                                       │                 │
│   │  [ See source assembly ]              │                 │
│   └───────────────────────────────────────┘                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Stats and Numbers (Use Throughout)

- **299 bytes** — current hex0 binary size
- **Python-hosted helixc** — current production compiler implementation
- **2,304 live tests collected** — restart 27 fix verification; rerun scoped pytest collection before publishing
- **30+ stages** — Approach A roadmap
- **23 silent-corruption bugs** — found and disclosed during development
- **9 audit passes** — multi-agent code review cycles
- **0 toolchain dependencies** — for the bootstrap chain
- **self-hosting target** — not shipped yet
- **100+ AST tags** — language richness
- **12 numeric types** — i32/i64/u8-u64/i8-i16/f32/f64/bf16
- **39 stages + amendments** — full Approach A scope

(Use these for "by-the-numbers" sections, infographics, and copy.)

---

## Closing Note

Helix is built to last. Every design decision favors transparency, reproducibility, and the long-term goal of an open-source AGI stack. The website should communicate this: not just "here's another language", but "here's a serious, principled effort to build the foundation for the next generation of ML systems — and you can verify every byte yourself."

The website is the public face of that effort. Make it as honest, technical, and beautiful as the language itself.

— *Draft website reference, reviewed for current-status honesty on 2026-05-15*
