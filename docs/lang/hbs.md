# Helix Bootstrap Subset (HBS) — v0.1 spec

**Status**: living draft — frozen pieces marked FROZEN, in-flux pieces marked DRAFT.
**Date**: 2026-05-04
**Goal**: identify the minimal Helix fragment needed to host a self-hosted compiler. Once HBS is closed the Python `helixc` becomes a parser + codegen shell while the rest of the compiler (typecheck, AD, IR passes) is itself written in Helix using only HBS features.

This doc draws the line: features INSIDE HBS must be implemented in `helixc` (Python) and verified to bootstrap. Features OUTSIDE HBS may be added later but a self-hosted compiler is required to compile only HBS-level Helix code.

## Why a subset?

A full self-host means rewriting `helixc.frontend`, `helixc.ir`, `helixc.backend` in Helix. If we let the compiler use every feature of the language, every feature must be supported by both `helixc` AND the self-hosted compiler simultaneously — a chicken-and-egg loop. A frozen HBS pins a tractable target: write `helixc` once for HBS, then anything written in HBS can also self-compile.

## The line — what's IN, what's OUT

### IN HBS (must self-compile)

Types — FROZEN:
- `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`
- `bool`
- `unit` (the `()` type)
- `&T`, `&mut T` (immutable / mutable references — non-aliasing, lifetime-erased)
- Fixed-size arrays `[T; N]` where N is a literal int
- Structs with named fields (no generics)
- Enums (sum types) with named variants — payload-bearing variants out (deferred)
- `fn(T1, T2) -> R` function pointers

Statements — FROZEN:
- `let x = e;`, `let mut x = e;`, `let x: T = e;`
- `if cond { ... } else if ... { ... } else { ... }`
- `while cond { ... }`
- `loop { ... }` with `break` / `continue`
- `for i in 0..n { ... }` (only literal-bounded ranges, plus `for i in 0..=n`)
- `return e;` and the implicit "tail expression as block result"
- `match` with `PatLit`, `PatBind`, `PatWildcard`, `PatTuple`, `PatOr`, `PatRange` and arm guards

Expressions — FROZEN:
- Integer / float / bool / string / char literals
- `+ - * / %`, `< <= > >= == !=`, `&& ||`, `! - ~`, `<< >>`, `& | ^`
- Function calls
- Field access `e.f`
- Array index `e[i]` and array literal `[e; n]`
- Casts `e as T`
- Block expressions `{ ... }` with implicit tail value

Top-level — FROZEN:
- `fn name(p: T) -> R { ... }`
- `pub fn`
- `struct`, `enum`, `type` aliases
- `const NAME: T = literal;`
- `use path::to::thing;`
- Attributes: `@pure`, `@partial`, `@total`, `@kernel` (parsed; semantics may not all be enforced)

### OUT OF HBS (can use in user code, NOT in self-hosted compiler)

Types — OUT:
- Generics (`fn map<T, U>(...)`)
- Associated types / traits / impls
- `D<T>` (differentiable wrapper) — AD is separate
- Tile / tensor types — those are GPU-side
- Memory-tier types `EpisodicMem<T>`, `SemanticMem<T>`, `WorkingMem<T>`
- `Skill<...>` (learn_to result)
- `AstNode` (quote/splice — reflection)
- Async / Future / `await`

Effects — OUT:
- Algebraic effect handlers (Tier 3)
- Capability/effect rows beyond the four pre-defined attrs

Reflection — OUT:
- `quote { ... }`, `splice(...)`, `splice_f(...)`, `modify(...)`, `modify_f(...)`
- `grad(f)`, `grad_rev(f)`, `grad_rev_all(f, ...)`

The self-hosted HBS compiler emits IR for these builtins as opaque calls; only the IR layer below knows their semantics, mirroring how today's `helixc` works.

## Bootstrap order (proposed)

Phase 0 (NOW — DRAFT): write a `helixc/bootstrap_compiler/` directory in Helix that compiles the HBS-only Helix-source-of-helixc back to itself. Order:
1. Lexer → token stream
2. Parser → AST
3. AST hasher (`structural_hash` for HBS subset)
4. Typechecker (HBS types only)
5. IR lowering (HBS expressions → tir.Op)
6. Const-fold + DCE + CSE passes
7. x86-64 emitter (HBS-only — no SIMD intrinsics yet)

Each phase is a Helix module compiled by today's Python helixc. After phase 7, we can run "Python helixc + Helix backend" as a hybrid, then drop more pieces of Python over time.

Phase 1: AST as first-class Helix value (WAVE1 #7) — promote `quote` from "stable hash" to a real ADT. Once we have that, compiler passes become ordinary Helix functions.

Phase 2: migrate one pass at a time to Helix. Order suggested: DCE → const-fold → CSE → typecheck → IR lowering. Each migration is a separate PR with parity tests against Python.

## What's verified today

As of 2026-05-04 (commit `bbf7a05`):
- AST hashing (`helixc.frontend.ast_hash.structural_hash`) covers all HBS expression types except `Quote`/`Splice`/`Modify` (those are reflection — OUT of HBS but we hash them anyway for content-addressing).
- Pattern matching with all 7 sub-features (binders, exhaustiveness, guards, or-patterns, ranges, codegen, AD) — all FROZEN HBS.
- Totality stub (`helixc.frontend.totality`) checks structural recursion for HBS-recursion shapes (`p - k`, `p / k`).
- Test count: 440 (all green); 13 commits this session including the full pattern-matching epic.

## Open questions

1. Do we want generics in HBS? They simplify writing the typechecker (`type Scope = Map<str, Type>` rather than a hand-rolled assoc list). Cost: bigger HBS compiler. Lean: yes, eventually, but defer to phase 1.
2. Do we want trait-based dispatch in HBS? Pattern matching gives us the same expressiveness for sum types. Lean: no.
3. Allocator strategy for HBS compiler: arena-only or allow heap? Lean: arena-only inside the compiler — easier to verify, no GC, every AST/IR node lives in a typed arena keyed by pass.
4. String handling: do we need a `Vec<char>` or can the HBS compiler operate entirely on `&str`-slice + index pairs? Lean: index pairs, pass alongside the source buffer.
5. Floating point in HBS: the typechecker doesn't need it (all sizes are integers). The IR lowerer does (it has to handle f32 ops). Decision: include `f32`/`f64` in HBS types but the HBS compiler itself uses only integers.

## Acceptance criterion for "HBS frozen"

The grammar and stdlib of HBS are considered frozen when:

- Every feature listed in "IN HBS" has a passing parser test, typecheck test, IR-lowering test, and codegen test.
- A reference HBS-only program of ≥500 LOC (e.g. a small calculator with `match` dispatch) compiles and runs end-to-end.
- The Helix-rewrite of any one Python pass (DCE is the cheapest target) passes parity against the Python implementation on a 50-program corpus.

When all three boxes are checked, we ship `hbs-1.0` and freeze the syntax.
