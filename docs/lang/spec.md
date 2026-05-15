# Helix Language Specification (v0.1 draft)

**Status**: living draft — updated as the implementation teaches us.
**Date**: 2026-05-15

Helix is a from-scratch programming language for AGI development and high-certainty computing. Kovostov is its first flagship system, but Helix is meant for any project that needs auditable, reproducible, uncertainty-aware software: AI/AGI, scientific research, medicine, genomics, physics, mathematics, robotics, infrastructure, and other domains where software should reason from evidence rather than hide assumptions. Designed to be (a) AI-author-friendly, (b) tile/tensor-native for GPU kernels, (c) compile cleanly to x86-64 and an early Phase-0 NVIDIA PTX subset without LLVM, (d) shape-typed with optional refinements.

## Design principles (in priority order)

1. **One canonical formatting.** No tabs vs spaces debate. No alternative syntaxes. AI-friendly = reproducible-friendly.
2. **No hidden state.** No globals, no implicit allocators, no monkey patching. All effects are typed and explicit.
3. **Tile-first.** Tiles are first-class values with memory-space tags. The compiler tracks where every tensor lives.
4. **Shape-typed.** Sizes are type parameters; mismatches caught at compile time. Optional refinements via Presburger arithmetic.
5. **Composable transformations.** `grad`, `grad_rev`, and `grad_rev_all` are current compiler-rewritten surfaces for scalar floating-point functions. `vmap`, `jit`, and `device` remain design targets.
6. **Functional core, imperative boundary.** Pure expressions inside, effects at the seam.
7. **No surprises.** No operator overloading on user types beyond a fixed list. No implicit conversions.
8. **Reduce uncertainty.** When uncertainty can be represented, bounded, proven, or surfaced to the caller, the language should make that explicit instead of letting it disappear into runtime convention.

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

#### Integer arithmetic semantics
Phase 0 backend implements **two's-complement wraparound** for all integer
operations: `i32::MAX + 1 == i32::MIN`, `i32::MIN - 1 == i32::MAX`,
multiplication and shift overflow truncate modulo `2^32`. This matches Rust
release-mode and C unsigned semantics. Helix does NOT trap on overflow
(no UBSan-equivalent runtime checks). Static checks (typecheck) flag
literal-out-of-range errors at compile time but do not analyze dynamic
overflow. `idiv` by zero is guarded in codegen to return 0 (matching the
spec's "defined behavior on edge cases"); `INT_MIN / -1` returns `INT_MIN`
(no `#DE` hardware trap). Float arithmetic follows IEEE-754: NaN compares
unordered, inf propagates, no exception flags are observable.

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

`@pure` is required for functions consumed by current AD rewrites (`grad`,
`grad_rev`, `grad_rev_all`) and will also be required by future `jvp`/`vjp`
surfaces.
Kernel purity and effect rules are being tightened separately; current
Phase-0 PTX tests accept bare `@kernel` functions when their body lowers to
the supported device subset.

### Kernels

Functions marked `@kernel` are intended to compile to GPU device code. Current
Phase-0 PTX support is intentionally narrow: 1D HBM tile parameters with `f32`
or `i32` element types and a small scalar-op subset. SMEM/REG tiles, `bf16`
kernel parameters, and full tiled matmul remain design targets rather than
shipped behavior.

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
fn loss(x: f32) -> f32 { x * x }

let g = grad(loss);                  // g : f32 -> f32
let rg = grad_rev(loss);             // reverse-mode scalar gradient
let all = grad_rev_all(loss);        // writes named scalar gradients
```

Current Stage 35 contract: these surfaces support `f32`/`f64` scalar
parameters. Aggregate parameters, integer parameters, and opaque calls without
known chain rules fail closed. Pytree leaf expansion, `value_and_grad`, `jvp`,
`vjp`, `jacrev`, and tensor-valued gradients are design targets, not current
public behavior.

## Composable transformations

```
// Design target, not the current Stage 35 public surface:
// let f_jit = jit(f);
// let f_vmapped = vmap(f, axis=0);
// let f_gpu = device(gpu(0))(f);
// let composed = jit(grad(vmap(loss, axis=0)));
```

Current compiler behavior is intentionally narrower: scalar `grad`,
`grad_rev`, and `grad_rev_all` are available first, with broader composition
added only after each transform has verifier-backed tests.

## FFI Status

`extern "C"` FFI exists as a Stage 16.5 feature for the current native backend.
It is effect-checked as `ffi`, participates in dynamic-link emission, and has
focused tests for integer, pointer, and `f32` ABI routing. Cross-platform ABI
coverage, Python/CUDA/ROCm interop, ownership-preserving wrappers, and richer
capability contracts remain future work.

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

## Implementation status (2026-05-04)

- Spec: this document, v0.1 draft (foundation-complete; HBS frozen pending)
- Lexer: **shipped** — `helixc/frontend/lexer.py`, ~120 token kinds, 1-char lookahead, multichar `..= -> => :: << >>` etc.
- Parser: **shipped** — `helixc/frontend/parser.py`, hand-written recursive descent with caret-rendering ParseError. Covers fn/struct/enum (incl. payload-bearing variants)/let-mut/match (with guards, or-patterns, range patterns, payload extraction)/if/while/loop/for/break/continue/assign/cast/tuple lit + `.0` access/struct lit/enum lit + Path::Variant/array lit/index/field access/call/binary+unary ops with precedence/return.
- Type checker: **shipped** — `helixc/frontend/typecheck.py`, ~1400 LOC. Includes: name resolution with did-you-mean (Levenshtein), struct field-type tracking via TyStruct, enum variant + exhaustiveness check, pattern binders into arm scope, totality stub (`@partial`/`@total`), static int-literal overflow check, source-with-caret error rendering. Optional Presburger solver for refinement constraints.
- IR: **shipped** — `helixc/ir/tir.py` defines ~50 OpKinds; `helixc/ir/lower_ast.py` (~1700 LOC) converts AST → IR with multi-slot ABI for structs/enums and arena-indirection for recursive enums. Passes: const_fold (with SSA value forwarding for x*1 / x+0 / x/1 / x%1), CSE, DCE, FDCE.
- Codegen: **shipped** — `helixc/backend/x86_64.py` (~1900 LOC) emits ELF + raw x86-64 machine code. SysV calling convention (6 int regs + 8 xmm regs). Includes IEEE-754 NaN-correct float comparisons, sys_open/write/close for `write_file`, RIP-relative reflection cells (HELIX_NUM_CELLS=64 i64) and arena (HELIX_ARENA_CAP=32K i32).

## HBS subset status (target: HBS-frozen)

Self-host bootstrap subset is being progressively closed. Foundation pieces shipped 2026-05-04:

- ✅ Pattern matching (literal, range, or, payload, guard, wildcard)
- ✅ Structs (incl. nested, with TyStruct propagation)
- ✅ Enums (tag-only + payload + recursive — recursive uses arena-indirection per chibicc/ocaml-bootstrap precedent)
- ✅ Tuples (`(a, b, c).N`)
- ✅ Pass-by-value of structs/enums to fns (multi-slot ABI; recursive enums pass as i32 arena index)
- ✅ Inline enum constructor as fn arg (`f(Some(42))`) and as fn return body
- ✅ Totality stub: structural-recursion check on `p - k` / `p / k` patterns
- ✅ Stdlib helpers: __min_i32 / __max_i32 / __clamp_i32 + transcendentals + __powi
- ✅ String literals: __strlen / __strbyte / __streq
- ✅ Arena: __arena_push / __arena_get / __arena_set / __arena_len (32K i32 region)
- ✅ Symbol-table assoc-list pattern (Helix-side, no new IR)
- ✅ helixc-check developer CLI: parse + typecheck + totality + --hash + --emit-ir, with caret error display

### Known HBS limitations as of 2026-05-04

These intentionally-deferred items shape the bootstrap path:

1. **No struct-by-value RETURN.** Callees return one i32 (typically the first slot). `@pure fn build_pair() -> Pair { Pair { a: 1, b: 2 } }` only carries `a` back. Workaround: use output params (write into an arena slot, return the index) or return tuples-of-i32. Tracked as Tier F follow-up.
2. **No mutable struct fields.** All struct fields are immutable after construction. Workaround: store field values in arena cells and reach them via `__arena_set` against a known index.
3. **No `for x in slice`.** Only `for i in 0..n` (i.e. integer ranges). Iteration over arena segments uses while loops with manual index counters.
4. **`read_file_to_arena(path)` IS implemented** as of 2026-05-04 (bug fix: fixes disp8 sign-extension on the read buffer). Bootstrap lexer can now load source bytes.
5. **No Vec<T>.** Arenas are monotonic — once pushed, never freed. The (start, count) carry-pair pattern fills the gap for compile-once workloads. Real malloc/free is a v0.2 concern.
6. **No HashMap.** Linear-scan assoc-list works at hundreds-of-symbols scale (HBS bootstrap). Hash bucket added as a polish item.
7. **3+-segment paths.** `crate::EnumName::Variant` is supported as a Phase-0 alias for `EnumName::Variant`. Other 3+-segment paths (e.g. `module::sub::Variant`) raise `NotImplementedError` at lowering rather than silently lowering to const_int(0). v0.1 has no module system.
8. **`@total` is conservative.** Recognizes `f(p - k)` and `f(p / k>=2)` but not Collatz-style or accumulator-based decrease. Use `@partial` for those.
9. **Inline enum-with-payload return value DOES work** (audit-7 cycle), but enum returns from fns of recursive-enum type fall back to the index encoding — preserved across calls correctly via the multi-slot ABI.
10. **Float coverage.** Scalar `f32` and `f64` are supported on the x86_64 path used by Stage 35 AD and FFI work. Scalar `f16`, `bf16`, and `fp8` remain future targets. Phase-0 PTX currently accepts only 1D HBM `tile<f32, ...>` and `tile<i32, ...>` kernel parameters; `tile<bf16, ...>`, SMEM, and REG kernel examples are design targets, not current public backend support.
11. **Generic type params lower to i32-sized ABI.** `fn id[T](x: T) -> T { x }` works for i32 type args; calls with i64/f32/struct args silently truncate. Callers should specialize manually until v0.2 monomorphization ships.

### Bugs fixed in 2026-05-04 deep-research cycle (8 fixes; 510 → 576 tests)

- **read_file_to_arena disp8 sign-extension**: `sub rsp, 128` with disp8 sign-extends 0x80 to -128, so it added 128 to rsp instead of subtracting. Fixed via `BUF_SIZE = 0x40` (fits signed disp8 cleanly).
- **Reverse-mode AD through match**: `autodiff_reverse` returned 0 for any function containing a Match. Now propagates per-arm contributions and wraps them as a Match with the same scrutinee.
- **f64/f16/bf16 scalar use silently truncated**: this cycle first hardened the compiler to fail instead of truncating. Later Stage 35 work enabled x86_64 scalar `f64`; `f16` and `bf16` still fail closed until their backends are implemented.
- **`let mut x` shadowing**: inner `let mut x` shared the outer's stack slot; codegen's name→slot table aliased them. Lowerer now mangles shadowed IR names (`x`, `x__1`, `x__2`).
- **Const-fold integer wraparound**: `INT_MAX + 1` folded to 2147483648 in Python (no wrap), breaking comparisons. Fix wraps every fold result to the target type's signed range so optimized + unoptimized builds agree.
- **`@effect(io)` parser**: parser dropped the `(io)` argument; typecheck couldn't tell which effects a function declared. Now records `effect:io`, `effect:rng` etc.
- **Arena GET/SET bounds checks**: `__arena_get(-1)` read memory before the arena base; `__arena_set` past the cap silently wrote into adjacent state. Both now cmp/jb-guarded with unsigned compare against `HELIX_ARENA_CAP`.
- **Negative-bound range patterns** (`-10..=-1`): parser rejected `MINUS` in a pattern atom. Now eats negative literals as `Unary("-", IntLit(...))` for both lo and hi range bounds.
- **3+-segment paths**: silently lowered to `const_int(0)` and made all match arms collide. `crate::E::V` now works as 2-segment alias; other 3+-segment paths error explicitly.

### Test count + audit history

As of 2026-05-04 deep-research cycle 2: **576 tests** in `helixc/tests/` all green. Determinism verified after commit `cd6667c` fixed a WSL test-bin race that was producing flaky failures.

10 audit-pass cycles + 2 deep-research cycles have caught and fixed 33 real bugs. Audit candidates explicitly considered:
- ARENA bounds (off-by-one), buffer +1 slot, bumped to fix
- STR_BYTE signed-vs-unsigned bounds (jl→jb)
- IEEE-754 NaN PF flag in float compare
- ast_hash For-loop encoding broke alpha-equivalence
- match_lower binds shared between guard/body
- || lowered as ADD producing >1 polluting ==
- &amp;&amp; lowered as ADD producing 1 for true&&false
- CONST_BOOL backend handler missing (true == true read garbage)
- DCE could drop QUOTE / ARENA_PUSH / ARENA_SET ops
- FDCE didn't trace fns reachable only via QUOTE
- Parser progress guard, _tok→_peek typo
- __powi cap mismatch with stdlib (n>16)
- PatOr binder leak (asymmetric binders silently visible in body)
- _resolve_type missing TyStruct case
- Cyclic struct path corruption in flat-path
- 1-field structs missed array-binding path
- Nested StructLit with Name fields filled with zeros
- Sub-struct Field as fn arg not expanded
- PatLit-of-Path test against array binding always-true
- Inline rec-enum ctor as fn arg miscompiled
- Inline rec-enum ctor as fn return value miscompiled
- WSL test_bin race producing flaky 528→527→521 result counts
