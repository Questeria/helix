# Helix Pre-Self-Host Feature Research

> Historical pre-Stage-29 research snapshot; not live gate evidence for Stage
> 35. Live status is tracked in `docs/ROADMAP.md` and
> `docs/stage35-progress-2026-05-15.md`.

**Date**: 2026-05-10
**Context**: Before Stage 29 (drop Python reference), what should Helix add?
**Scope**: Read-only research. No source modifications.
**Author**: Research agent commissioned by Kovostov-Native lead.

---

## Executive Summary

At this snapshot, Helix as of Stage 28.8 was a remarkably complete ML-first systems language. The
core language carries twelve scalar types (i8..u64 + bf16/f16/f32/f64), full
sum types, pattern matching with exhaustiveness checking, generics with trait
bounds, closures lowered through env-passing, modules + impl blocks, raw-binary
ELF codegen with optional FFI, two autodiff modes plus `@checkpoint`
rematerialization, a tile/tensor type-family with a working PTX backend, and
six pipeline passes (const-fold, CSE, DCE, FDCE, effect-check, hash-cons,
totality). The Python frontend at `helixc/frontend/` is the executable
specification (~10k lines of Python passes); the bootstrap at
`helixc/bootstrap/` is ~12k lines of Helix code that compiles itself.

The gap that matters most before Stage 29 is **NOT** more language surface.
The bootstrap kovc.hx is *thinner* than the Python reference along three
specific axes that, if not closed, make post-Stage-29 evolution painful:

  1. Several frontend passes exist only in `helixc/frontend/*.py` (match
     lowering, struct monomorphization, pytree flatten/unflatten, autotune
     variant generation, pre-codegen validation passes for unsafe/panic/
     deprecated/trace). After Stage 29 the bootstrap must own these passes or
     the language is "self-hosted in name only" — the user still depends on
     Python to compile any program that uses those features.
  2. Several language features are **specced in `HELIX_REFERENCE.md` and used
     by the Python frontend but parsed-then-no-op'd by the bootstrap**:
     string interpolation is absent, `?` operator missing, `let-else` missing,
     named-arg construction `Foo { x: ..., y: ... }` exists in Python AST but
     the bootstrap parses positional only, struct field-name disambiguation
     hasn't been pushed all the way through. These are cheap to add now and
     expensive to retrofit because every change forces the byte-identical
     gate at Stage 29 to be re-baselined.
  3. The Tier-3 strategic moat — provenance-typed neuro-symbolic
     (`D<Logic<T>>`, `TyMemTier`) — exists at type-level only. Without a
     concrete fuzzy/AD semantics or a memory-tier cost model wired into
     codegen, the moat is rhetorical, not load-bearing.

**Top 3 recommendations (HIGH priority, do before Stage 29):**

  1. **Port match-lowering, struct-mono, pytree, panic/unsafe/deprecated
     validation passes into kovc.hx** — anything that the Python reference
     does that has no bootstrap counterpart is a Stage-29-blocker. (~3 stages
     of work; addresses 8 Python-only files.)
  2. **Add the `?` operator + `let-else` + `let mut` shadowing rules + string
     interpolation `f"..."` + named struct-lit fields.** These are the
     language-ergonomics fixes the bootstrap will *itself* benefit from
     during future stdlib evolution. ~1.5 stages, all low-risk.
  3. **Wire D<Logic<T>> fuzzy-AND/OR/NOT codegen + TyMemTier cost annotations
     into the IR**, so that the Tier-3 moat actually produces measurably
     different lowering. ~2 stages, including a probabilistic-type smoke
     test against an SVHN-shaped neuro-symbolic toy.

Everything in Categories 1 and 4 below is more speculative; only the items
explicitly tagged HIGH are recommended for pre-29 inclusion.

---

## Snapshot of current state (used to scope recommendations)

| Layer | Implementation | LoC |
|------|--------|------|
| Python frontend | `helixc/frontend/*.py` — 24 modules | ~10,000 |
| Python IR + passes | `helixc/ir/*.py` + `helixc/ir/passes/*.py` | ~4,000 |
| Python backends | x86_64, ptx, elf_dyn | ~4,000 |
| Bootstrap (Helix) | lexer.hx + parser.hx + kovc.hx + evaluator.hx | ~12,400 |
| Stdlib (Helix) | 16 `.hx` modules | ~7,400 |
| Tests | 38 pytest files | — |

Python-only frontend passes (no bootstrap counterpart yet): `match_lower.py`,
`struct_mono.py`, `pytree.py`, `autotune.py`, `panic_pass.py`,
`unsafe_pass.py`, `deprecated_pass.py`, `trace_pass.py`, `presburger.py`,
`grad_pass.py`, `autodiff_reverse.py`, `monomorphize.py`, `flatten_modules.py`,
`flatten_impls.py`, `hash_cons.py`, `totality.py`, `ast_hash.py`,
`diagnostics.py`. Several of these are partially mirrored in the bootstrap
(monomorphization, flatten, hash-cons, totality have working bootstrap
implementations); others are entirely Python-side specs.

---

## Recommended additions BEFORE Stage 29 (HIGH priority)

These items are categorized as `BLOCKER` (Stage 29 cannot land cleanly
without them) or `STRONGLY-PREFERRED` (Stage 29 can land but the resulting
self-hosted compiler will silently lose a feature relative to Python).

### H1. Port Python-only frontend passes into kovc.hx — BLOCKER

**Definition**: Several passes run ONLY against the Python frontend. After
Stage 29, source files that exercise these passes will compile differently
(or fail) under the self-hosted compiler.

**Why this matters**: The whole point of "byte-identical" verification at
Stage 29 is that every test compiles the same way. If `match` desugaring or
struct monomorphization is Python-only, then the moment we drop Python, those
features compile via *whatever the bootstrap does today*, which is provably a
proper subset.

**Concrete files to address**:

| Python file | What it does | Bootstrap status |
|-------------|--------------|------------------|
| `match_lower.py` | Desugars `Match → If/Let` chains | Partial in kovc.hx (literal + variant patterns work; PatTuple, PatOr, PatRange tested but with explicit traps; guards Phase-1) |
| `struct_mono.py` | Monomorphizes `struct Foo<T>` | NOT in bootstrap (Stage 28 wires struct mono table in Python only). |
| `pytree.py` | Flattens nested struct → leaf paths for AD | NOT in bootstrap |
| `autotune.py` | Cross-product variant generation | NOT in bootstrap |
| `panic_pass.py` | Validates `panic("msg")` arity + str-lit arg | Partial — kovc.hx Audit 28.8 A1 wired panic → TRAP, but Python-side validates args. |
| `unsafe_pass.py` | Validates raw-ptr ops are inside `unsafe {}` | Partial — kovc.hx Audit 28.8 A2 wired the gate. |
| `deprecated_pass.py` | Walks call sites for @deprecated symbols | NOT in bootstrap |
| `trace_pass.py` | Static spec for @trace fn prologue/epilogue | NOT in bootstrap |
| `presburger.py` | Shape constraint solver for tile/tensor sigs | NOT in bootstrap |

**Sketch (e.g., for struct_mono in kovc.hx)**:
```rust
// In kovc.hx: after parse_program, walk fn-list for calls where the callee's
// declaring fn has TyGeneric param types. For each (fn_name, ty_args), if
// not already in mono_table:
//   1. Clone the fn body via mk_node recursively
//   2. Substitute TyName(g) where g is in the generic_params list
//   3. Mangle name: f__T1_T2 (mirror Stage 8's _mangle_ty)
//   4. Register in mono_table; add cloned fn to fn_list
```

**Retrofit difficulty if deferred**: HIGH. Once Python is dropped, any bug in
the bootstrap version cannot be cross-checked against a known-good reference.
Worse, the test suite that today validates "Python and bootstrap agree" will
silently change semantics (only the bootstrap voice remains) and the audit
gate will not catch silently-divergent behavior because there's no oracle.

**Scope estimate**: 3-4 stage-equivalents. Each pass should land as its own
stage with audit cycle: 28.9 (match_lower port + property tests), 28.10
(struct_mono port), 28.11 (panic/unsafe/deprecated/trace passes; small,
batchable), 28.12 (pytree + autotune ports; partial — full PTX-side variant
dispatch can defer).

---

### H2. The `?` operator for Option/Result — STRONGLY-PREFERRED

**Definition**: `expr?` desugars to `match expr { Ok(x) => x, Err(e) =>
return Err(e) }` (or for `Option<T>`, the `None` arm short-circuits).

**Why for AGI**: The whole stdlib `Result<T, E>` pattern is unusable in
practice without `?`. Every AGI primitive that can fail (parsing
observations, recall, planning) becomes a deeply-nested `match` ladder.
Adding `?` after self-host means changing every stdlib usage to the new form
under a hard reference-byte-identical constraint.

**Sketch**:
```rust
fn parse(s: String) -> Result {
    let len = str_len(s)?;          // desugars to:
    // match str_len(s) { Result::Ok(v) => v, Result::Err(e) => return Result::Err(e) }
    let head = str_byte_at(s, 0)?;
    Result::Ok(len + head)
}
```

**Current absence**: Stdlib uses positional `match` ladders everywhere — see
`option.hx` (`option_unwrap_or`, `option_eq_some`) and `result.hx`. Both
files would benefit from a `?` desugaring.

**Retrofit difficulty if deferred**: MEDIUM. Adding `?` later means rewriting
all stdlib functions that currently use the manual ladder, and the byte-
identical Stage 29 gate would have to be re-baselined. Adding pre-29 means
the rewrites happen against the Python reference and Stage 29 picks up the
new form cleanly.

**Scope estimate**: 0.5 stage-equivalents (one parser change + one match-
lower rule + ~20 stdlib rewrites + audit cycle).

---

### H3. String interpolation `f"..."` (or printf-like) — STRONGLY-PREFERRED

**Definition**: `f"hello {name}"` desugars to `str_concat("hello ",
to_string(name))`. Phase-0 form is a simple template-with-named-holes.

**Why for AGI**: Every diagnostic, every Telegram update, every panic
message currently has to be ad-hoc concatenated. The compiler's own
diagnostics module (Stage 22 caret renderer) is Python-only because the
bootstrap has no string-formatting story. If we drop Python, the new
self-hosted diagnostics will be string-concat soup.

**Current absence**: Bootstrap has `str_concat`, `str_len`, `str_eq`,
`str_byte_at`, but no `to_string<T>` and no template syntax. Format
strings are concatenated by hand in kovc.hx's error paths.

**Sketch (Python-frontend version, port to bootstrap)**:
```rust
// f"x={x}, y={y}" desugars to:
str_concat(str_concat("x=", i32_to_string(x)), str_concat(", y=", i32_to_string(y)))
// or a faster fold:
fmt_buf(&["x=", &x, ", y=", &y])  // varargs trait, Phase-1
```

**Retrofit difficulty if deferred**: MEDIUM. The diagnostics module rewrite
when added later means every error string in kovc.hx changes shape.

**Scope estimate**: 1 stage. Add `f"..."` lexer mode, parse interpolations
into `StrInterp { parts: list[Either<str, Expr>] }`, lower to chained
`str_concat` (Phase-0) or to a `Fmt` trait (Phase-1 — defer).

---

### H4. `let-else` for early-return on pattern match — STRONGLY-PREFERRED

**Definition**: `let Pat = expr else { ... };` runs the else block (which
must diverge) if `expr` doesn't match `Pat`, otherwise binds. Equivalent to
`let X = match expr { Pat => v, _ => { else_block } };` but with the
"diverge" requirement enforced.

**Why for AGI**: The bootstrap is full of `match self_check { Result::Ok(v)
=> v, Result::Err(_) => { emit_trap_with_id(...); 0 } }` patterns where the
trap-path returns a dummy value. `let-else` makes the divergence explicit
and lets the type-checker prove the post-let path is reachable only on
success.

**Current absence**: Not in parser, not in AST. The fallback `match` form
works but produces less-readable bootstrap code.

**Sketch**:
```rust
fn parse_int(s: String) -> i32 {
    let Result::Ok(v) = parse_int_inner(s) else {
        emit_trap_with_id(99001);
    };
    v   // here `v` is bound and we know the success path
}
```

**Retrofit difficulty if deferred**: LOW (it's just sugar) — but adding now
makes the kovc.hx port of frontend passes substantially shorter.

**Scope estimate**: 0.3 stage-equivalents.

---

### H5. Named struct-lit fields throughout kovc.hx — STRONGLY-PREFERRED

**Definition**: `Foo { x: 10, y: 32 }` is parsed by the Python frontend
(see `ast_nodes.py:StructLit`), but the bootstrap parses positional-only:
`Foo { 10, 32 }`. This means kovc.hx code-bases lose the documentation
value of named fields, and once a struct grows a field, every literal must
be rewritten in position order.

**Why now**: The bootstrap parser is the canonical user of structs in
post-29 code. Every audit fix that adds a new field today requires updating
*every* StructLit call site rather than just adding `field: value` in the
new struct.

**Current absence**: `parse_struct_lit` in `parser.hx` recognizes positional
form; named-field form is parsed by `parser.py` but unused by the bootstrap
because no bootstrap source uses it.

**Sketch**: Extend the bootstrap parser's `parse_struct_lit` to accept
`IDENT ':' expr` arms (already lexed) and reorder to declared field order
via the struct table.

**Retrofit difficulty if deferred**: HIGH for stdlib evolution. Every
StructDecl change after Stage 29 forces re-ordering at every call site, and
the byte-identical gate at Stage 30 would have to re-baseline.

**Scope estimate**: 0.5 stage-equivalents.

---

### H6. Source-position propagation through codegen — STRONGLY-PREFERRED

**Definition**: Every emitted byte gets associated with the source span that
produced it, so runtime traps (and a future debugger / coverage tool) can
report `file.hx:line:col` rather than `trap-id 14001`.

**Why now**: Today, `emit_trap_with_id(14001)` produces a SIGILL with the
trap-id in eax. The user has to look up the trap-id in code comments to
find what fired. Adding source-position propagation now (alongside the
existing trap-id scheme) means the post-29 debug story can build on it.
Adding it after Stage 29 means re-architecting kovc.hx's emit-table.

**Sketch**: Each `emit_byte` call records `(byte_offset, src_span)` in a
side-table arena. The driver writes the side-table to a `.dbg` companion
file alongside the ELF (later: DWARF, Stage 28.13).

**Current absence**: Trap-ids exist but no source mapping.

**Retrofit difficulty if deferred**: HIGH. Bootstrap codegen is structured
around "emit byte, advance counter"; threading a parallel source-span
arena through every emit is invasive.

**Scope estimate**: 1.5 stages — 28.13 emit `.dbg` side-table, 28.14 wire
into trap-emit so SIGILL reports a meaningful position.

---

### H7. ABI stabilization marker on `extern "C"` decls — BLOCKER for FFI growth

**Definition**: The bootstrap's `extern "C"` handling (`backend/elf_dyn.py`,
ported partially) treats every extern call as SysV-x86_64. After Stage 29,
adding aarch64 / Win64 ABIs would require changes to every FFI site.

**Why now**: Stage 16.5 landed `extern "C"` but assumes SysV throughout.
The Python `backend/elf_dyn.py` has the relocation tables hard-coded; the
bootstrap mirrors them. Without a `cdecl` / `sysv64` / `aapcs64` discrim
in the AST, future cross-platform work is constrained to "remove the
sysv64 assumption everywhere" — expensive.

**Sketch**:
```rust
extern "C" fn malloc(size: u64) -> *mut u8;        // implicit sysv64
extern "sysv64" fn malloc(size: u64) -> *mut u8;   // explicit (default)
extern "win64" fn HeapAlloc(...) -> ...;           // Windows ABI
```

**Current absence**: `extern_abi: Optional[str]` is on FnDecl
(`ast_nodes.py:459`); parser only recognizes "C". Bootstrap is "C"-only.

**Retrofit difficulty if deferred**: MEDIUM-HIGH. Cross-target ABI work is
fundamentally a thread-through-codegen task; doing it for n+1 ABIs is much
easier than for n+5.

**Scope estimate**: 0.5 stage now (just the AST + parser accept), 2-3
stages later for actual codegen across ABIs. Pre-29 work is the syntax + AST
shape.

---

## Recommended additions for v0.2 (MEDIUM, deferrable)

These should be specced before Stage 29 (one paragraph in the reference doc)
but the implementation can land after.

### M1. Higher-kinded types (HKT) for monad-like abstractions

**Definition**: A type can be generic over a *type constructor*, e.g.
`fn map<F: Functor, A, B>(fa: F<A>, f: fn(A) -> B) -> F<B>`. Helix today
has generics-over-types but not generics-over-type-constructors.

**Why for AGI**: Option, Result, Vec, HashMap, and (soon) Future all share
the Functor / Monad shape. Without HKT, the stdlib duplicates `map_option`,
`map_result`, `map_vec`, etc.

**Current absence**: The Python `Type` enum has TyVar but no TyConstructor.

**Retrofit difficulty if deferred**: MEDIUM-HIGH. HKT touches the inference
algorithm; adding it later is a typechecker rewrite. But the language
*works* without HKT — Rust still doesn't have it.

**Verdict**: Spec for v0.2, defer implementation.

---

### M2. Type families / associated types in traits

**Definition**: `trait Iter { type Item; fn next(self) -> Option<Self::Item>;
}` — a trait with a type member.

**Why for AGI**: The iterator protocol is currently un-implementable in
Helix because traits don't have associated types. The `iterators.hx` stdlib
is a pile of i32-specialized free fns.

**Current absence**: Trait decls only carry method signatures
(`ast_nodes.py:Trait/`impl: methods only).

**Retrofit difficulty if deferred**: MEDIUM. Trait machinery is mostly in
flatten_impls.py; adding `type` members is incremental.

**Verdict**: Spec for v0.2, defer.

---

### M3. Const generics with arithmetic

**Definition**: `fn fft<const N: usize>(...)` where `N` participates in
arithmetic at the type level (e.g., `where N % 2 == 0`).

**Why for AGI**: Tile shapes are already const generics — `tile<f32, [4, 4],
REG>`. But there's no arithmetic at type level. So `tile_matmul<f32,
[N, K], [K, M]> -> tile<f32, [N, M]>` cannot be written; the user instantiates
explicitly.

**Current absence**: Presburger solver exists for size constraints
(`presburger.py`), but it's not wired into the bootstrap and only handles
linear cases.

**Retrofit difficulty if deferred**: MEDIUM. Presburger lives in Python;
porting to bootstrap is incremental.

**Verdict**: Wire Presburger into bootstrap as M3 in v0.2.

---

### M4. Variance annotations on generic params

**Definition**: `struct Stack<+T>` says `T` is covariant; `fn foo<-T>` says
contravariant. Without these, Helix has to be conservative (invariant) at
every use of a generic.

**Why now**: Pytree/AD already trips on invariance — a `D<f64>` is not
substitutable for an `f64` even though semantically the former is "more
informative". Adding variance later requires re-doing every type-equality
check.

**Verdict**: Spec for v0.2 with leaning towards `+`/`-`/invariant notation.

---

### M5. Generic Option<T> / Result<T, E> (drop i32 specialization)

**Definition**: The stdlib's `Option` and `Result` are i32-specialized
(`enum Option { Some(i32), None }`). Real generic versions need the Phase-1
type-tagged enum payloads.

**Why now**: Every other primitive (f32, f64, struct types) cannot use
Option/Result today. Adding generic enums means the entire stdlib rewrite,
which post-Stage-29 means byte-identical re-baselining at Stage 30.

**Retrofit difficulty if deferred**: HIGH for stdlib, but the *language*
already supports `Pair<A, B>`. The blocker is enum-payload-tagged codegen.

**Verdict**: Land before Stage 29 if Stage 6 enum codegen can be extended.
Otherwise spec for v0.2.

---

### M6. Where-clauses on user code (not just internal type checker)

**Definition**: Allow `fn matmul<N, M, K>(...) where N >= 1, M >= 1, K >= 1`
to express runtime-checked invariants.

**Why for AGI**: The bootstrap is full of "this index must be in range" checks
that fire as traps. Where-clauses + a small SMT-lite check at the typechecker
would catch many of them at compile time.

**Current absence**: `WhereClause` exists in AST (`ast_nodes.py:439`) but
the typechecker captures into `self.constraints` and never solves them.

**Retrofit difficulty if deferred**: LOW (parser is done; typechecker is the
work).

**Verdict**: Spec for v0.2; cheap implementation.

---

### M7. Iterator protocol + for-in over user types

**Definition**: `for x in collection` where `collection: impl Iterator`.

**Why for AGI**: `agi_search.hx` and `iterators.hx` would shrink dramatically.

**Verdict**: Blocked on M2 (associated types). Spec for v0.2.

---

### M8. Operator overloading via traits (`Add`, `Mul`, etc.)

**Definition**: `impl Add for Pt { fn add(self, other: Pt) -> Pt }` lets
`p + q` dispatch to `Pt::add(p, q)`.

**Why for AGI**: Tensor algebra, complex numbers, dual numbers (forward AD)
all need this. Today they're emulated with named fns like `t_add`.

**Verdict**: Spec for v0.2; trait infrastructure is in place.

---

## v0.3+ wishlist (LOW priority, exploratory)

### L1. Lifetime regions / borrow checking

Helix's `unsafe` + `*const T` / `*mut T` is enough for FFI. A full borrow
checker is a 6-12 month project and would slow language-evolution velocity.
Recommended **OUT** of v0.x (matches APPROACH_A_PLAN.md "out of scope" line).

### L2. Algebraic effects / row-typed effects

The current `@effect(io.read_file)` attribute is a closed set with explicit
propagation. Full row-typed effects (effect polymorphism) would let
`fn higher_order<E>(f: fn() -> i32 @E) -> i32 @E` thread effects through.
Cool but speculative. v0.3+.

### L3. Refinement types / dependent-ish types

`fn safe_div(x: i32, y: i32 where y != 0) -> i32` — proves at the call
site that the divisor is nonzero. The Presburger solver could grow into
this. v0.3+.

### L4. Async/await OR coroutines / generators

The keywords are reserved (`async`, `await` in `lexer.py:98`) but no
implementation. Useful for ARC-AGI-3-style interactive games. v0.3+.

### L5. GADTs (Generalized Algebraic Data Types)

Variant payload types can refine the type parameter. Useful for tagless-
final encodings. v0.3+ if at all.

### L6. Existential types / `impl Trait` return

`fn foo() -> impl Iterator` — useful if M2 (associated types) ever lands.
v0.3+.

### L7. Inline assembly safe wrapping

`asm! { mov rax, 0x80; syscall }` block. The bootstrap already emits raw
machine code; user-level asm would be a wrapper around the same emit-byte
infrastructure. v0.3+.

### L8. Soft-typed values (probabilistic types, distributions)

`fn flip() -> Bernoulli<bool>` — useful for AGI but speculative. Could be
implemented as a library on top of TyLogic. v0.3+.

### L9. WebAssembly backend

For the playground. Spec'd in HELIX_REFERENCE.md ("--target=wasm32"). Useful
but post-AGI. v0.3+.

### L10. Bytecode VM target

For interpretation / fast-iteration. Useful for REPL and for the metacircular
evaluator at `evaluator.hx`. v0.3+.

### L11. LLVM IR backend

A safety net for cross-platform codegen. Goes against the
"no toolchain dependencies" goal but useful if a serious port target
emerges (riscv, aarch64). v0.3+.

### L12. Tree-sitter grammar / LSP server / formatter / linter

Tooling polish. Listed in HELIX_REFERENCE.md "Tooling appendix" as
post-shipping. v0.3+.

### L13. Cargo-style package manager (`kovpkg`)

Path-based modules are sufficient through v1.0 per APPROACH_A_PLAN.md.

---

## Out-of-scope (will NOT add)

Repeating and refining the original "out of scope" list from
APPROACH_A_PLAN.md:

- **Garbage collector**: value semantics + arena ownership are the model.
  GC is a 12-month project that would constrain ML-kernel codegen.
- **OOP class hierarchies**: traits cover the only useful case (dispatch).
  No inheritance, no `virtual`, no diamond problem.
- **Multiple inheritance**: ditto.
- **Implicit conversions / coercions**: `as` is explicit. Implicit `i32 →
  f64` makes type errors silent.
- **Full borrow checker**: per L1 above. Uniqueness types (already implicit
  via value semantics) are sufficient for AGI/ML.
- **Lean-4 proof-carrying terms**: defer until external research adoption
  demonstrates need.
- **JIT / REPL**: AOT + autotune covers the production case; an interpreter
  could live in `evaluator.hx` as a separate artifact.
- **Cargo-style package manager**: path-based modules through v1.0.
- **Row-polymorphic effects**: closed-set `@effect` is enough.

---

## Cross-cutting concerns

### CC1. Test oracle preservation

Once Python is dropped at Stage 29, the test suite at `helixc/tests/` no
longer has a Python implementation to cross-check against. Recommendation:
**add a `tests/golden/` directory of expected-bytes files for every test
case before Stage 29**, so the self-hosted compiler at Stage 30 can
byte-compare its output against frozen Python-compiled output without
needing the Python interpreter live.

### CC2. Stage 28.8 audit findings as feature signals

Audit cycle 1 (the cycle that produced today's three audit docs) found
issues like A6/C1-H1 (panic_pass walker missing if/else arms), C1-M1
(monkey-patched program attribute), and A2 (unsafe gate not wired). Many
of these are symptoms of pass-walker fragility — the AST has 40+ node types
and every new pass has to manually walk them all. **A general-purpose AST
visitor / walker library would close this class of bugs at the root.** Not
a language feature per se, but a stdlib-level abstraction.

### CC3. Diagnostics reproducibility

`diagnostics.py` is Python-only. The bootstrap reports trap-ids only.
Recommendation: port a minimal `render_caret(file, line, col, msg)` into
the bootstrap *before* Stage 29 so the post-29 error story isn't a
regression.

### CC4. Reproducible build manifest

Once Python is dropped, the build process needs a self-contained manifest:
"these N hex0 bytes + these M `.hx` files reproduce this exact `kovc`
binary." Today it's implicit. Recommendation: ship a
`reproducible-build.toml` (or similar; spec-only, no package manager) that
lists input SHAs + output SHA. Pre-29 is the right time because the input
list is stable.

### CC5. Trap-id namespace governance

The bootstrap uses `AST_TAG * 1000 + sub_id` for trap-ids. Today this
namespace is managed ad-hoc; Audit 28.8 A4 found `24001` was double-claimed
(AST_MOD's bf16 modulo AND Stage 24 provenance). Recommendation: a single
`docs/trap-ids.md` registry that enumerates every trap-id with status,
maintained as a checked-in CSV / JSON so Stage 28.13 audit cycles can
verify uniqueness automatically. Doesn't need to be a language feature; it's
a workflow rule.

### CC6. Telemetry / `@trace` runtime story

`@trace` is parsed but the runtime trace buffer isn't wired in the
bootstrap. For AGI episode logging (which is the use case), a working
`@trace` is more important than another autodiff mode. Land the bootstrap-
side TraceBuffer before Stage 29.

### CC7. The "second-system" trap

The natural temptation is to add every feature in Category 1 (type system)
before Stage 29 — "while we have the Python oracle." Don't. The Stage 29
gate exists to *force discipline*. Recommended scope: only the H1-H7 items
above. Everything else is post-29.

---

## Suggested staging

Concrete proposal for what should slot in *before* Stage 29 actually fires.
Ordered by criticality:

| Stage | Feature | Scope | Why this order |
|-------|---------|-------|----------------|
| **28.9** | Port `match_lower` into kovc.hx | 1 stage | Foundational — many other passes need full match semantics. Done first so subsequent passes use match cleanly. |
| **28.10** | Port `struct_mono` into kovc.hx | 1 stage | Stage 28 already shipped struct mono in Python; kovc.hx needs parity. Critical for stdlib growth. |
| **28.11** | Port panic / unsafe / deprecated / trace validation passes | 1 stage | Small individually; batch as one stage. Each is <200 LoC. |
| **28.12** | Add `?` operator + `let-else` + named struct-lit fields | 1 stage | Ergonomic cluster; one parser pass + 3 desugaring rules. Lifts every subsequent stage's code-readability. |
| **28.13** | Add `f"..."` string interpolation + bootstrap-side render_caret | 1 stage | Diagnostics independence from Python. |
| **28.14** | Source-position side-table + `.dbg` companion file | 1 stage | Foundation for debugger; ABI-stable across Stage 29. |
| **28.15** | `extern_abi` syntax acceptance + `tests/golden/` byte-baseline | 0.5 stage | Cross-platform readiness without committing to non-SysV codegen. |
| **28.16** | Generic Option/Result (drop i32 spec) + `?` over both | 1 stage | Wraps up the ergonomics push. Only if Stage 6 enum codegen extends cleanly; otherwise defer to v0.2. |
| **28.17** | Trap-id registry doc + Stage 28.8-style audit cycle | 0.5 stage | Governance step; clean baseline before Stage 29. |
| **28.18** | Bootstrap-side `@trace` runtime (TraceBuffer + entry/exit emit) | 1 stage | AGI feature most-used post-self-host. |
| **28.19** | D<Logic<T>> fuzzy AND/OR/NOT codegen + smoke test | 1.5 stages | Tier-3 moat made real. Optional if the moat is judged rhetorical. |
| **28.20** | TyMemTier cost-model wiring into IR (annotate; no scheduler yet) | 1 stage | Spec-only; defer scheduler to v0.2. |
| **28.21** | 5 consecutive clean audits over the new surface | 0-5 stages | Standard Stage 28.8 protocol applied to the new code. |
| **29** | Drop Python reference | as designed | |
| **30** | 5 consecutive clean audits, self-host only | as designed | |

**Total pre-29 scope**: ~11-12 stages, dominated by ports and ergonomics. No
type-system speculative work (H1-H7 only). Estimated wall time: 2-3 months at
current cadence based on Stages 22-28.7 throughput.

**Alternative aggressive variant**: skip 28.16 (generic Option/Result) and
28.19-28.20 (Tier-3 moat work) — defer to v0.2. Pre-29 scope drops to ~8
stages. Recommended if user wants Stage 29 sooner.

**Alternative minimal variant**: only do 28.9-28.13 + 28.17 + 28.21. Pre-29
scope drops to ~5 stages. The Tier-3 moat ships as "spec only" and the
generic Option/Result waits for v0.2. This is the lowest-risk path that
still preserves Python parity at Stage 29.

---

## Recommended NOT to do before Stage 29

Per CC7 ("second-system trap"), explicitly defer:

- HKT (Category M1)
- Type families (M2)
- Const-generic arithmetic (M3)
- Variance annotations (M4)
- Where-clause solver (M6)
- Iterator protocol (M7)
- Operator overloading (M8)
- ALL of Category L

These are real features that the AGI work will eventually want, but each
adds 1-3 months to the pre-29 timeline. The reference can spec them as
"v0.2 plans" — the spec is cheap, the implementation is not.

---

## Specific recommendations tied to existing files

Reading the codebase, the following surfaced as concrete, actionable items:

1. **`helixc/frontend/match_lower.py:31-36`** — the pattern test list
   includes `PatRange`, `PatOr`, `PatTuple` but the bootstrap parser at
   `helixc/bootstrap/parser.hx:5800-5945` parses these into AST tags
   (64-70) without bootstrap-side desugaring. The desugaring is needed
   inside `kovc.hx`'s match codegen (around line 5500). Currently the
   bootstrap codegen for PatTuple inside `match` uses a hand-rolled
   conjunction; PatOr is partially handled. Port match_lower's lowering
   rules into kovc.hx as a pre-codegen pass.

2. **`helixc/frontend/struct_mono.py:38-39`** — TRAP IDs 28001/28002 are
   reserved for Stage 28 parametric structs. The bootstrap recognizes
   `struct Pt<T>` but doesn't perform the instantiation walk. kovc.hx needs
   a `struct_mono_walk` analogous to its existing `mono_table` for fns.

3. **`helixc/stdlib/option.hx:14-15`** — explicit Phase 1.9 limitation:
   "Option is i32-specialised because generics over enum variants require
   type-tagged-payload codegen (Phase 2 item)." This is M5. Without it the
   stdlib is awkward for f64/f32/struct payloads.

4. **`helixc/frontend/trace_pass.py:20-23`** — comment says "Runtime trace
   buffer wiring (entry/exit emission into the binary prologue/epilogue)
   is bootstrap-side; this module exists so the Python typechecker + a
   Python-side simulator can validate the design before kovc.hx implements
   it." That bootstrap-side wiring should land pre-Stage-29 (item 28.18).

5. **`helixc/frontend/pytree.py:20-22`** — "Phase-0 cap" of 4 levels of
   nesting. Bootstrap doesn't pytree at all. Item 28.12 (or 28.18 if AGI-
   facing) should port flatten_pytree / unflatten_pytree into kovc.hx.

6. **`helixc/frontend/typecheck.py:115-181`** — TyDiff, TyLogic, TyMemTier,
   TySkill are all defined in the type system but the AD pipeline doesn't
   *use* TyLogic for fuzzy semantics. Item 28.19 wires this in.

7. **`helixc/frontend/autotune.py:38-40`** — TRAP 27001 reserved, but the
   variant Cartesian-product walker is Python-only. Item 28.11 (or 28.12
   if @autotune is judged a v0.1 feature).

8. **`helixc/frontend/parser.py:74-79`** — `KW_ASYNC` and `KW_AWAIT` are
   reserved but unused. Don't implement; just confirm they're not
   accidentally usable by user code (mitigation: ensure the parser emits
   "unsupported" error if used pre-v0.3).

9. **`helixc/frontend/deprecated_pass.py:38-46`** — Audit 28.8 C1-M1 found
   monkey-patching of `_deprecation_warnings` on Program. The Python fix
   landed. The bootstrap port needs to use a side-arena, not modify the
   AST in-place. Item 28.11.

10. **`helixc/bootstrap/kovc.hx:5440-5485`** — body-vs-return-type 8-byte
    mismatch trap. Audit cycle 2's fix is solid but the underlying
    `expr_type` cascade has 10+ tag arms. The Stage 1.6 refactor (already
    landed) reduced this from O(N) to O(1) for the common case, but each
    new type adds an arm. Item 28.18 should consider extracting this into
    a generated dispatch table.

---

## Note: features that don't help

Things the user might expect to come up but actually aren't priorities:

- **More numeric types** (fp8, mxfp4, nvfp4, ternary keywords are reserved):
  the actual codegen for these is GPU-specific and waits on Stage 16 PTX
  maturity. Reserving the keywords is enough; full implementation is post-
  AGI training.
- **Concurrency primitives** (channels, locks, atomics): not needed for the
  bootstrap compiler. Add when the AGI runtime needs them.
- **GUI / rendering**: out of scope for the language.
- **Web stack** (HTTP, JSON parsing): build with FFI when needed.
- **Database connectivity**: out of scope.
- **Cross-compilation to embedded targets**: out of scope until AGI ships.

---

## Final recommendation

The pre-Stage-29 work breaks into two clean phases:

**Phase A (BLOCKER work)**: Stages 28.9-28.13 + 28.17 — port Python-only
passes, add ergonomic primitives (`?`, `let-else`, named struct-lit, `f"..."`,
render_caret), establish trap-id governance. Without this, Stage 29 leaves
the language regressed.

**Phase B (Strategic moat)**: Stages 28.14-28.20 — source-position
side-table, extern_abi syntax, generic Option/Result, bootstrap-side
@trace, Tier-3 moat wiring. Without this, Stage 29 ships a self-hosted
compiler that doesn't realize its strategic ambitions.

User decision point: ship Phase A only (minimum viable) at Stage 29, or
include Phase B for a stronger v0.1?

Both paths preserve "AGI-foundational language" — Phase A is the discipline
choice; Phase B is the ambition choice. Recommend Phase A as the default
because the ambition items (especially Tier-3 moat wiring) require external
benchmarks the project doesn't have yet. Tier-3 moat is best landed in v0.2
when its semantics are validated against a real neuro-symbolic task.

---

*End of pre-self-host research. ~6,500 words.*
