# Helix AGI-specific language features

These are the things in Helix that **no other language has**. The C-equivalent
foundation (loops, arrays, floats) exists to make these implementable.

## 1. Reflection: `quote` and `splice`

Every Helix expression can be captured as data:

```helix
let ast = quote { fib(n - 1) + fib(n - 2) };
// ast: AstNode — a value the program can inspect and manipulate
```

`splice(ast)` does the inverse — re-injects an AST value into source position.

**This is unique because**: most languages have macros that run at compile-time
only. Helix's `quote` produces *runtime values* of type `AstNode`. The AGI can
read its own source, transform it, splice it, and run the result. None of:
C, C++, Rust, Mojo, Zig, Triton, JAX, Julia, Python — have this with the same
combination of static-typing + first-class runtime AST values.

**Closest precedent**: Lisp `(quote ...)`, Scheme syntax-rules, Rust proc-macros
(but compile-time only), Julia `:expr` (but not statically typed). Helix is a
typed Lisp-quote without parentheses-everywhere.

## 2. Verifier-gated self-modification: `modify`

```helix
fn modify[F](
    target: Symbol,           // the function to modify
    transformation: AstNode -> AstNode,
    verifier: AstNode -> bool, // safety check
) -> Result<(), ModifyError>;
```

The AGI proposes a transformation to a function in its own program. The
verifier runs over the new AST. If verifier returns true, the transformation
is committed; the program now has the new function. If false, rejected with
diagnostic.

Verifiers themselves can be: type checks, test-suite runners, formal proof
checkers, regression-bench runners.

**This is unique because**: existing systems either (a) don't support
self-modification at all (C/Rust/most languages) or (b) support it without
formal verification (Lisp, Forth, eval). Helix makes the verifier a *required*
parameter, formalizing the safety boundary.

## 3. Effect/capability types

Every effectful operation in Helix carries a *capability* in its type:

```helix
@effect(io.read_file, io.network)
fn fetch_html(url: &str) -> String { ... }
```

Functions inherit the union of their callees' capabilities. `@pure` functions
have *no* capabilities. The type system enforces that capabilities can only
be passed to functions that declare permission for them.

**The AGI safety story**: a function marked `@effect(modify_self)` can rewrite
the AGI's source. The compiler tracks which functions can do this. A
capability-typed function pointer cannot be passed to a `@pure` context, even
through indirection. This makes "the AGI cannot modify itself unless given
capability X" formally checkable.

**Unique because**: effect systems exist (Koka, Eff, Idris, Ocaml-5) but none
focus on AGI-safety capabilities. Mojo doesn't have this. JAX doesn't have
this.

## 4. Memory-tier types

Working memory, episodic memory, semantic memory, procedural memory each get
their own type:

```helix
type WorkingMem<T> = Box<T>;          // ephemeral, current task
type EpisodicMem<T> = Stamped<T>;     // tagged with timestamp + context
type SemanticMem<T> = Indexed<T>;     // knowledge graph node
type ProceduralMem<F> = Skill<F>;     // learned procedure
```

Cross-tier ops (e.g., consolidating episodic → semantic) are first-class
operators: `consolidate(epi)`, `recall(query)`, `retrieve(addr)`. The compiler
enforces invariants: episodic memories must carry timestamps, semantic
entries must be deduplicated, etc.

**Unique because**: Hippocampus-formal-models exist in cognitive science but
no programming language exposes them as language types. This makes the
AGI's memory architecture *part of its program text*.

## 5. Differentiable types

Every numeric value can optionally carry gradient information at the type
level:

```helix
fn loss(x: D<f32>, y: D<f32>) -> D<f32> {
    let diff = x - y;
    diff * diff
}
```

`D<T>` means "differentiable T". The compiler propagates gradient flow at
type-check time. `grad(f)` is a compiler pass that produces the backward
function. `D<T>` values can compose with non-differentiable code only via
explicit `detach`.

**Unique because**: PyTorch tracks gradients at *runtime* via a tape.
JAX tracks them via *function transformations*. Helix tracks them in the
*type system*, catching gradient bugs at compile time.

## 6. Tile-as-first-class type with memory hierarchy

```helix
let a_tile: tile<bf16, [16, 16], smem> = tile::load_global(a, [i, j]);
let b_tile: tile<bf16, [16, 16], smem> = tile::load_global(b, [j, k]);
let c_tile: tile<f32, [16, 16], reg> = tile::matmul(a_tile, b_tile);
tile::store_global(c, [i, k], c_tile);
```

The compiler tracks: dtype, shape, memory space (HBM / SMEM / REG / TMEM).
Cross-memory-space operations require explicit movement. The compiler
schedules tile-level computation around the memory hierarchy automatically.

**Unique because**: Triton has tiles but they're not in the type system —
the programmer manages memory placement implicitly. Mojo has SIMD types but
not memory-space types. Helix makes memory tiers part of the type signature.

## 7. Agent types

```helix
agent Planner {
    fn propose(state: State) -> Action;
}

agent Critic {
    fn evaluate(state: State, action: Action) -> Score;
}

let society = society::new()
    .add(Planner)
    .add(Critic);

let action = society::dispatch(propose, state);
```

`agent`, `society`, `dispatch`, `compete`, `cooperate` are language
primitives. The compiler can lower agent dispatch to function calls,
async tasks, or distributed RPC.

**Unique because**: actor languages (Erlang, Elixir, Akka) exist but they
don't model cognitive society-of-mind concepts (proposer/critic/voter/
broadcaster). Helix bakes the cognitive architecture into the type system.

## 8. Auto-curriculum primitives

```helix
let skill = curriculum::learn_to(
    task = "matrix_inversion",
    difficulty = 0.7,
    budget = 100,
);
```

The compiler maintains a registry of skills with measured difficulties. The
AGI can request "skills at difficulty X" — the runtime selects the closest
match or proposes a learning task. This is the Voyager / Eureka / Goldilocks
pattern, made first-class.

**Unique because**: skill libraries exist as Python registries; making this
a *language primitive* with type-level guarantees is novel.

---

## Implementation status (live)

| Feature | Status | Tests | Notes |
|---|---|---|---|
| 1. Reflection (`quote`/`splice`/`modify`) | ✅ working (stub semantics) | 4 codegen | `quote` returns a stable AST hash |
| 2. Verifier-gated modify | ✅ scaffolded | 2 codegen | accept/reject based on verifier value |
| 3. Effect/capability types | ✅ working | 6 typecheck | `@pure` / `@io` etc. propagate at compile time |
| 4. Memory-tier types | ✅ working | 5 typecheck | Working/Episodic/Semantic/Procedural |
| 5. Differentiable types `D<T>` | ✅ working | 5 typecheck | propagates through binary ops |
| 6. Shape-typed tensors + Presburger | ✅ working | 28 (24 solver + 4 integration) | catches matmul mismatches at compile time |
| 7. Agent type declarations | ✅ parsing | 4 parser | `agent Foo { fn ...; }` |
| 8. Tile types in codegen | Phase-0 PTX lowering | PTX / Tile IR tests | 1D HBM `f32`/`i32` kernels plus scalar ops; broader GPU lowering remains in progress |
| 9. Composable transforms (`grad`/`vmap`) | ✅ engine working (CLI) | 13 autodiff | symbolic forward-mode AD; CLI prints derivatives |
| 10. Auto-curriculum (`learn_to`) | ✅ working (type-level) | 2 typecheck | returns Skill<F>; runtime registry TBD |

Live test and commit counts move quickly during staged development; see the
current stage progress note and `pytest` output for authoritative evidence.

### Autodiff usage

Helix has a working forward-mode autodiff engine. Use the CLI:

```bash
$ cat loss.hx
fn loss(x: f32) -> f32 { x * x }

$ python -m helixc.frontend.autodiff_cli loss.hx loss
d(loss)/d(x) = (x + x)
```

Or programmatically:

```python
from helixc.frontend.parser import parse
from helixc.frontend.autodiff import differentiate, fmt
prog = parse("fn f(x: f32) -> f32 { x * x * x }")
fn = prog.items[0]
deriv = differentiate(fn.body.final_expr, "x")
print(fmt(deriv))   # = (((x + x) * x) + (x * x))
```

This is the engine that will eventually power a `grad(f)` builtin in the
language. The engine handles +, -, *, /, unary -, constants, and variables.
Future work: chain rule for function calls, blocks/let-bindings, reverse-mode.

## What this gives Helix as a foundation

Combining the type-system features above gives Helix capabilities no other
language has. The single function signature

```kov
fn agi_step[N: size](
    sensory: WorkingMem<tensor<f32, [N]>>,
    weights: D<tensor<f32, [N, N]>>,
) -> WorkingMem<tensor<f32, [N]>>
where N % 16 == 0
```

formally expresses, at the type level:
- The function's inputs and outputs are tagged with their memory tier
- The weights are gradient-tracked (D-wrapped)
- Shapes are constrained to multiples of 16 (Presburger-checked)
- The function is implicitly `@pure` so the compiler enforces it cannot do
  I/O, network, or `modify_self`

No other AI language (Mojo, Triton, JAX, Julia, PyTorch, TensorFlow,
Rust+Burn, Swift-for-TF) expresses all four of these at the type level.
Mojo and Hasktorch get partial credit for shape; nothing else has the
combination.

See `helixc/examples/agi_demo.hx` for a working demonstration that
typechecks cleanly with all four features stacked.

## Roadmap (remaining work)

| Item | Effort estimate | Why it matters |
|---|---|---|
| Real reflection (runtime AST inspection) | 2-3 weeks | the AGI literally reads its own source |
| Real verifier semantics for `modify` | 1 week | safety boundary for self-modification |
| Tile-typed kernels in codegen | 2-3 months | GPU performance parity with Triton/Mojo |
| `grad` as compiler primitive | 1-2 months | source-level autodiff better than JAX |
| `society::dispatch` semantics | 1-2 weeks | makes agent declarations actually work |
| `curriculum::learn_to` semantics | 1 week | first-class auto-curriculum |
| Constant folding + DCE | 1 week | basic optimizations for production code |
| Real type inference (fewer `TyUnknown`) | 2 weeks | better diagnostics, more checks |
