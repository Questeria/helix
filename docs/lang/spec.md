# Helix Language Specification (v0.1 draft)

**Status**: living draft — updated as the implementation teaches us.
**Date**: 2026-05-03

Helix is the from-scratch programming language for the Kovostov AGI project. Designed to be (a) AI-author-friendly, (b) tile/tensor-native for GPU kernels, (c) compile cleanly to x86-64 and NVIDIA PTX without LLVM, (d) shape-typed with optional refinements.

## Design principles (in priority order)

1. **One canonical formatting.** No tabs vs spaces debate. No alternative syntaxes. AI-friendly = reproducible-friendly.
2. **No hidden state.** No globals, no implicit allocators, no monkey patching. All effects are typed and explicit.
3. **Tile-first.** Tiles are first-class values with memory-space tags. The compiler tracks where every tensor lives.
4. **Shape-typed.** Sizes are type parameters; mismatches caught at compile time. Optional refinements via Presburger arithmetic.
5. **Composable transformations.** `grad`, `vmap`, `jit`, `device` are functions that take and return functions. JAX-style.
6. **Functional core, imperative boundary.** Pure expressions inside, effects at the seam.
7. **No surprises.** No operator overloading on user types beyond a fixed list. No implicit conversions.

## File extension and source structure

- `.hx` — Helix source files
- `module path/to/module` declaration at the top
- One module per file is conventional but not required

## Lexical grammar

### Whitespace and comments
- Whitespace: space, tab, newline, CR (significant only as separator; no semantic indentation in v0.1)
- Single-line comments: `// to end of line`
- Block comments: `/* nest-able */`

### Identifiers
- `[a-zA-Z_][a-zA-Z0-9_]*`
- Conventions: `snake_case` for values, `PascalCase` for types, `SCREAMING_SNAKE` for compile-time constants

### Keywords (reserved)
```
fn let mut const type struct enum trait impl
if else match for while loop break continue return
true false
i8 i16 i32 i64 isize  u8 u16 u32 u64 usize
bf16 f16 f32 f64 fp8 mxfp4 nvfp4 ternary
bool char tile tensor
where as in of
pub priv mod use
async await
device cpu gpu hbm smem reg tmem
kernel grad jvp vjp vmap
size                    // size-type keyword for shape parameters
```

### Literals
- Integer: `42`, `0xFF`, `0b1010`, `0o755`, optionally typed: `42_i32`, `0xFF_u8`
- Float: `3.14`, `1e-5`, optionally typed: `3.14_f32`, `1e-5_bf16`
- Ternary: `-1`, `0`, `+1` with type annotation `: ternary`
- String: `"hello\n"`, supports `\n \t \r \\ \" \0 \xNN \u{NNNN}`
- Char: `'a'`, `'\n'`
- Bool: `true`, `false`

### Operators
```
+ - * / %         // arithmetic
== != < > <= >=   // comparison
&& ||             // logical
& | ^ << >> ~     // bitwise
= += -= *= /= %=  // assignment / compound
-> => :: . ..     // structural (-> return type, => match arm, .. range)
( ) [ ] { } , ;   // delimiters
@                 // attribute (e.g., @kernel, @inline)
?                 // refinement-clause separator
```

## Type system

### Primitive types
| Type | Bits | Notes |
|---|---|---|
| `i8` `i16` `i32` `i64` | 8/16/32/64 | signed integers |
| `u8` `u16` `u32` `u64` | 8/16/32/64 | unsigned integers |
| `usize` `isize` | platform-word | for indexing |
| `bool` | 1 (stored 1 byte) | `true`/`false` |
| `char` | 32 | Unicode scalar value |
| `bf16` `f16` `f32` `f64` | 16/16/32/64 | IEEE-754 / bfloat16 |
| `fp8` | 8 | E4M3 by default; `fp8_e5m2` for E5M2 |
| `mxfp4` `nvfp4` | 4-bit + scale | block-scaled per OCP MX or NVIDIA spec |
| `ternary` | ~1.58 | `{-1, 0, +1}` packed |

### Aggregate types
- Tuples: `(T1, T2, T3)`
- Structs: `struct Foo { x: i32, y: f32 }`
- Enums: `enum Color { Red, Green, Blue(i32) }` (sum types)
- Arrays (statically sized): `[T; N]` where N is `size`
- Slices (dynamically sized views): `&[T]`, `&mut [T]`
- References: `&T` (immutable), `&mut T` (unique mutable)
- Function pointers: `fn(i32, i32) -> i32`

### Tensor and Tile types

```
tensor<dtype, [d1, d2, ..., dN], device, layout>
tile<dtype, [d1, d2, ..., dN], memspace>
```

- `dtype`: any element type (scalar OR `(format, block, scale_format)` triple for block-scaled)
- `[d1, ..., dN]`: shape — dimensions can be:
  - integer literals: `[64, 128]`
  - `size` parameters: `[N, M]` (compile-time named)
  - `Dyn`: `[Dyn, M]` (dynamic, runtime-checked at boundary)
- `device`: `cpu`, `gpu(0)`, `gpu(D)` (where D is a `device` parameter), or omitted (= polymorphic)
- `layout`: `row_major` (default), `col_major`, `blocked(B1, B2)`, or omitted (compiler chooses)
- `memspace`: for `tile` only — `hbm`, `smem`, `reg`, `tmem`

Examples:
```
tensor<f32, [64, 128]>                          // 64x128 f32, default device, default layout
tensor<bf16, [N, M], gpu(0)>                    // size-polymorphic, on GPU 0
tensor<(mxfp4, 32, e8m0), [N, K], gpu(D)>       // block-scaled
tile<bf16, [16, 16], smem>                      // 16x16 bf16 tile in shared memory
tile<bf16, [16, 16], reg>                       // same, in registers
```

### Size types

`size` is a special type kind: a non-negative integer known at compile time, used as a type parameter.

```
fn matmul[N: size, M: size, P: size](
    a: tensor<f32, [N, M]>,
    b: tensor<f32, [M, P]>,
) -> tensor<f32, [N, P]> { ... }
```

Sizes can be combined arithmetically in types:

```
fn concat[N: size, M: size, D: size](
    a: tensor<f32, [N, D]>,
    b: tensor<f32, [M, D]>,
) -> tensor<f32, [N + M, D]> { ... }
```

The constraint solver uses linear arithmetic over `size` (Presburger arithmetic — decidable, fast).

### Refinement clauses

Optional `where` clauses on size parameters, discharged by Presburger arithmetic:

```
fn block_matmul[N: size, M: size, P: size, B: size](
    a: tensor<bf16, [N, M]>,
    b: tensor<bf16, [M, P]>,
) -> tensor<bf16, [N, P]>
where
    N % B == 0,
    M % B == 0,
    P % B == 0,
{ ... }
```

### Linearity / ownership

Buffers (the underlying memory) are affine: each owns one mutable reference at a time. Tensors are *views* into buffers; multiple immutable views may coexist.

```
fn add_inplace[N: size](dst: &mut tensor<f32, [N]>, src: &tensor<f32, [N]>) { ... }
```

## Functions

```
fn name[T1: kind, T2: kind, N: size](arg1: T1, arg2: T2) -> ReturnT
where <constraints>
{
    // body
}
```

- Generic parameters in `[ ... ]`
- Value parameters in `( ... )`
- Return type after `->` (omit for unit `()`)
- Constraints in `where` clause

### Pure vs effectful

```
@pure
fn add(a: i32, b: i32) -> i32 { a + b }

@effect(io)
fn print(s: &str) { ... }

@effect(rng)
fn rand_uniform[N: size](shape: [N]) -> tensor<f32, [N]> { ... }
```

`@pure` is required for kernel functions and for functions consumed by `grad`/`jvp`/`vjp`.

### Kernels

Functions marked `@kernel` are compiled to GPU device code:

```
@kernel
fn matmul_tile[N: size, M: size, P: size](
    a: tile<bf16, [N, M], smem>,
    b: tile<bf16, [M, P], smem>,
    c: tile<f32, [N, P], reg>,
) {
    // tile-level code
}
```

## Expressions

Standard fare, with these notes:
- `if` is an expression: `let x = if c { 1 } else { 2 };`
- Block expressions: last-statement-no-semicolon = block value
- `match` expressions on enums, integers, tuples
- Tensor literals: `tensor::<f32>([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])`
- Tensor indexing: `a[i, j]` desugars to `tensor::index(a, i, j)`
- Tile primitive operations: `tile::load(...)`, `tile::store(...)`, `tile::matmul(...)`

## Autodiff API

```
fn loss(x: tensor<f32, [N]>) -> f32 { ... }

let g = grad(loss);                  // g : tensor<f32, [N]> -> tensor<f32, [N]>
let (val, grad_val) = value_and_grad(loss)(x);
let v = vjp(loss, x);                // v : (vector) -> tensor<f32, [N]>
let j = jacrev(loss);                // jacobian-reverse
```

`grad`, `jvp`, `vjp` etc are *compiler primitives* exposed as library functions. The compiler implements `jvp` rules per primitive, derives `linearize` + `transpose`, and composes for the rest.

## Composable transformations

```
let f_jit = jit(f);
let f_vmapped = vmap(f, axis=0);
let f_gpu = device(gpu(0))(f);
let composed = jit(grad(vmap(loss, axis=0)));
```

## Module system (v0.1)

```
module path::to::module

use core::tensor;
use core::tile::{load, store, matmul};

pub fn exposed() { ... }
priv fn internal() { ... }
```

## Reserved for v0.2+
- Traits / impls (Rust-style)
- Async/await
- Macros (compile-time)
- Custom autodiff rules (`@custom_jvp`, `@custom_vjp`)
- Concurrency primitives
- FFI (`extern "C"`)

## Out of scope (likely never)
- Inheritance
- Implicit conversions between numeric types
- Operator overloading on user types
- Reflection / runtime type info (in user-facing surface; compiler internals are different)
- Garbage collection

## Example program

```helix
module examples::matmul

@pure
fn matmul[N: size, M: size, P: size](
    a: tensor<bf16, [N, M], gpu(0)>,
    b: tensor<bf16, [M, P], gpu(0)>,
) -> tensor<f32, [N, P], gpu(0)>
where N % 16 == 0, M % 16 == 0, P % 16 == 0,
{
    let mut c = tensor::zeros::<f32, [N, P]>(gpu(0));
    for ti in 0 .. N / 16 {
        for tj in 0 .. P / 16 {
            let mut acc = tile::zeros::<f32, [16, 16], reg>();
            for tk in 0 .. M / 16 {
                let a_tile = tile::load_smem(a, [ti * 16, tk * 16]);
                let b_tile = tile::load_smem(b, [tk * 16, tj * 16]);
                acc = tile::matmul(a_tile, b_tile, acc);
            }
            tile::store(c, [ti * 16, tj * 16], acc);
        }
    }
    c
}
```

## Implementation status (2026-05-03)

- Spec: this document, v0.1 draft
- Lexer: in progress (Python prototype in `helixc/frontend/lexer.py`)
- Parser: not started
- Type checker: not started
- IR: not started
- Codegen: not started
