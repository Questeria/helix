# Appendix E — Example index

*What this chapter covers: a categorized, navigational index of the real Helix example
programs that ship under [`helixc/examples/`](../../../helixc/examples/) (100 `.hx` files),
grouped by theme, plus the example programs the gate compiles and runs with a fixed exit
code. For each entry you get the path, a one-line description of what it demonstrates, and —
where the gate asserts it — the exit code. Use this as a lookup table: find a feature, jump
to a real file you can compile.*

This appendix is a **reference index**, not a tutorial. It does not re-teach the language
(Part III, *planned*) or the build (see [Part II — Setup & Build](../part2-setup-build/02-build-from-raw.md)).
Every file named here exists in the repository at tag `v1.3-release`; the listing was taken
directly from `helixc/examples/`. Nothing here is invented — if a feature has no real file, it
is simply not listed.

> **For AI agents:** the authoritative file list is the directory itself. Before you cite or
> open an example, dereference the path under [`helixc/examples/`](../../../helixc/examples/)
> rather than trusting this prose. If this index and the directory disagree, the directory
> wins and this appendix is the bug — flag it.

---

## E.1 How to read this index, and what "gate-asserted" means

The examples split into two trust tiers:

1. **Gate-asserted programs.** A subset of `helixc/examples/*.hx` is compiled **and run** by
   the gate — [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`, the feature
   corpus — with its exit code checked against a literal expected value (the `chk` helper at
   `scripts/gate_kovc.sh:306`). These are the strongest tier: they are part of the standing
   compile-and-run proof that ships green (the gate prints `GATE_PASS`; corpus 109/0). For
   these, this appendix states the exact exit code and cites the `chk` line.
2. **Compile-corpus / demonstration programs.** The remaining `.hx` files in the directory are
   real, committed example programs that exercise specific features. Most are not individually
   exit-code-asserted by the gate; treat them as readable, illustrative sources. Where a file's
   own header comment states an expected result, this index repeats that *as the file claims
   it*, not as an independent gate assertion.

> **For AI agents:** only the files in §E.2 carry a gate-asserted exit code. The `chk` helper
> runs the freshly self-hosted `K2.bin` over each program (`scripts/gate_kovc.sh:308–311`) and
> compares the process exit status to the expected integer. For every *other* example, do **not**
> claim a specific exit code unless you have just compiled and run it yourself with the real
> toolchain — the directory listing proves the file exists, not that it returns any particular value.

A note on terminology used below: the compiler is `kovc`; the self-host fixpoint is
`seed → K1 → K2 → K3 → K4` (see [Trust at a glance](../part1-orientation/04-trust-at-a-glance.md));
the GPU back end is **complete to PTX, not SASS** (§E.7).

---

## E.2 Gate-asserted example programs (exit codes proven by the gate)

These seven programs live in `helixc/examples/` **and** are run by the gate's feature corpus
with a checked exit code. (The corpus also runs many small inline-generated and `corpus_gen/`
fixtures that are not files under `helixc/examples/`; those are out of scope for an *examples*
index — the full corpus story is **Part VIII — The gate and the feature corpus** *(planned)*; for
the overview today see [Trust at a glance](../part1-orientation/04-trust-at-a-glance.md).)

| File (under `helixc/examples/`) | Demonstrates | Gate exit code | `chk` site |
|---|---|---|---|
| [`exit42.hx`](../../../helixc/examples/exit42.hx) | First end-to-end program: `fn main`, integer return | **42** | `scripts/gate_kovc.sh:313` |
| [`matmul_2x2.hx`](../../../helixc/examples/matmul_2x2.hx) | 2×2 matmul by scalar ops; returns `trace(A*B)` | **69** | `scripts/gate_kovc.sh:313` |
| [`hbs_sample_enum_struct.hx`](../../../helixc/examples/hbs_sample_enum_struct.hx) | enum + struct: a flat 2-D-shape calculator dispatching on an enum kind | **129** | `scripts/gate_kovc.sh:313` |
| [`hbs_sample_option.hx`](../../../helixc/examples/hbs_sample_option.hx) | payload-bearing enum (`Maybe::Some(42)`) + match payload extraction | **42** | `scripts/gate_kovc.sh:314` |
| [`hbs_sample_recursion.hx`](../../../helixc/examples/hbs_sample_recursion.hx) | recursion over enum payloads; computes `5! = 120` via a Maybe-style state machine | **120** | `scripts/gate_kovc.sh:314` |
| [`dogfood_18_pat_struct_showcase.hx`](../../../helixc/examples/dogfood_18_pat_struct_showcase.hx) | struct-destructuring pattern match (flat / literal-field / nested) | **42** | `scripts/gate_kovc.sh:315` |
| [`gradient_descent.hx`](../../../helixc/examples/gradient_descent.hx) | one gradient-descent step minimizing `(x-3)^2` | **42** | `scripts/gate_kovc.sh:315` |

The simplest of these is the canonical first program.

**Verified example** — [`helixc/examples/exit42.hx`](../../../helixc/examples/exit42.hx)
(compiled and run by the gate; `chk "$EX/exit42.hx" 42`, `scripts/gate_kovc.sh:313` ⇒ `$? == 42`):

```helix
// First end-to-end Helix program: compiles, links, runs, exits with status 42.
//
// Demonstrates: function decl, return, integer literal.
// Verification: $? == 42 after running the produced ELF.

fn main() -> i32 {
    42
}
```

The next-simplest gate-asserted program computes a real value rather than a constant: a 2×2
matrix product reduced to its trace.

**Verified example** — [`helixc/examples/matmul_2x2.hx`](../../../helixc/examples/matmul_2x2.hx)
(gate-asserted `chk "$EX/matmul_2x2.hx" 69`, `scripts/gate_kovc.sh:313` ⇒ `$? == 69`), `fn main` body:

```helix
fn main() -> i32 {
    // We're capped at 3 args per fn in the v0.1 codegen, so call with
    // partial args via a wrapper. Actually let's compute directly inline:
    //   A = [[1, 2], [3, 4]]
    //   B = [[5, 6], [7, 8]]
    let a00 = 1; let a01 = 2; let a10 = 3; let a11 = 4;
    let b00 = 5; let b01 = 6; let b10 = 7; let b11 = 8;
    let c00 = a00 * b00 + a01 * b10;
    let c11 = a10 * b01 + a11 * b11;
    c00 + c11   // 19 + 50 = 69
}
```

> **Note:** the `// capped at 3 args` comment is a historical artifact of the v0.1 codegen this
> file was first written under. The shipping `kovc` supports many parameters (the gate's
> `f8`/`f9`/`f11` corpus fixtures exercise 8-, 9-, and 11-argument calls via the SysV stack-pass
> path, `scripts/gate_kovc.sh:335`). The example still compiles and returns 69; the comment is
> just stale narration inside the file.

> **For AI agents:** to reproduce any row in §E.2 manually, run the gate
> (`bash scripts/gate_kovc.sh` under WSL) and read the `PASS <file> (<rc>)` lines it prints for
> step `[4]`. Do not paraphrase the expected codes — they are the integers in the `chk` calls.

---

## E.3 Language basics and first programs

Small programs that introduce the surface syntax, output, and end-to-end compile path.

| File | Demonstrates |
|---|---|
| [`hello.hx`](../../../helixc/examples/hello.hx) | first program — module decl, `@pure fn add`, lexer+parser end-to-end |
| [`hello_world.hx`](../../../helixc/examples/hello_world.hx) | visible output: `print_str` (write to stdout) + `write_file` builtins |
| [`exit42.hx`](../../../helixc/examples/exit42.hx) | minimal `fn main` returning an integer exit code (gate-asserted, §E.2) |
| [`mandelbrot.hx`](../../../helixc/examples/mandelbrot.hx) | Mandelbrot set rendered to stdout as ASCII over a 60×24 grid |
| [`full_demo.hx`](../../../helixc/examples/full_demo.hx) | a single end-to-end demonstration program |

> **For AI agents:** [`hello.hx`](../../../helixc/examples/hello.hx) opens with `module examples::hello`
> and is library-shaped (no `fn main`); it is a parser/lexer exercise, not an executable with a
> defined exit code. [`hello_world.hx`](../../../helixc/examples/hello_world.hx) is the one that
> prints and writes a file. Pick the right one for your purpose.

---

## E.4 Types, control flow, and pattern matching

Programs centered on the type system, `match`, and the dogfood scaffolds that walk specific
feature increments. The numbered `dogfood_*` series tracks the language as it was built; the
`hbs_sample_*` / `hbs_integration_*` series are "Helix-builds-the-shapes-a-compiler-needs"
dogfood programs (ASTs, symbol tables, evaluators).

### Pattern matching, enums, structs

| File | Demonstrates |
|---|---|
| [`hbs_sample_enum_struct.hx`](../../../helixc/examples/hbs_sample_enum_struct.hx) | enum kind + struct shapes, dispatch by match (gate ⇒ 129, §E.2) |
| [`hbs_sample_option.hx`](../../../helixc/examples/hbs_sample_option.hx) | payload enum + match extraction (gate ⇒ 42, §E.2) |
| [`hbs_sample_recursion.hx`](../../../helixc/examples/hbs_sample_recursion.hx) | recursion through enum payloads (gate ⇒ 120, §E.2) |
| [`hbs_sample_tree_eval.hx`](../../../helixc/examples/hbs_sample_tree_eval.hx) | tiny AST evaluator over enum constants |
| [`hbs_sample_ast_eval.hx`](../../../helixc/examples/hbs_sample_ast_eval.hx) | recursive-enum AST evaluator |
| [`hbs_sample_visitor.hx`](../../../helixc/examples/hbs_sample_visitor.hx) | an "AST visitor" pattern composed from recent features |
| [`hbs_sample_constant_fold.hx`](../../../helixc/examples/hbs_sample_constant_fold.hx) | a real compiler pass in Helix: constant folding |
| [`hbs_sample_helix_ast.hx`](../../../helixc/examples/hbs_sample_helix_ast.hx) | a Helix-side AST for a tiny calculator language |
| [`hbs_sample_calculator.hx`](../../../helixc/examples/hbs_sample_calculator.hx) | expression evaluator exercising the Tier-A surface |
| [`hbs_sample_symbol_table.hx`](../../../helixc/examples/hbs_sample_symbol_table.hx) | assoc-list symbol table on the arena builtins |
| [`hbs_sample_lexer_skeleton.hx`](../../../helixc/examples/hbs_sample_lexer_skeleton.hx) | tokenizer skeleton on the string + arena builtins |
| [`metacircular_eval.hx`](../../../helixc/examples/metacircular_eval.hx) | metacircular evaluator: Helix interpreting a Helix `Expr` enum |
| [`symbolic_algebra.hx`](../../../helixc/examples/symbolic_algebra.hx) | symbolic differentiate + simplify + evaluate over a recursive enum |
| [`dogfood_18_pat_struct_showcase.hx`](../../../helixc/examples/dogfood_18_pat_struct_showcase.hx) | struct-destructuring match surface (gate ⇒ 42, §E.2) |
| [`dogfood_19_pat_struct_guards.hx`](../../../helixc/examples/dogfood_19_pat_struct_guards.hx) | struct destructuring composed with match guards |

### Result, the `?` operator, and integration tests

| File | Demonstrates |
|---|---|
| [`dogfood_16_result_basic.hx`](../../../helixc/examples/dogfood_16_result_basic.hx) | `Result<T, E>` typecheck scaffolding (first 2-parameter wrapper) |
| [`dogfood_17_try_operator.hx`](../../../helixc/examples/dogfood_17_try_operator.hx) | `?` postfix propagation operator over `Result<T, E>` |
| [`hbs_integration_calculator.hx`](../../../helixc/examples/hbs_integration_calculator.hx) | integration test: build + evaluate a small expression |
| [`hbs_integration_bst.hx`](../../../helixc/examples/hbs_integration_bst.hx) | integration test: binary search tree with insert/lookup |
| [`hbs_integration_arena_stress.hx`](../../../helixc/examples/hbs_integration_arena_stress.hx) | stress test: many values pushed into the arena via a `while` loop |
| [`hbs_pattern_struct_return.hx`](../../../helixc/examples/hbs_pattern_struct_return.hx) | simulating struct-return-by-value via an arena output cell |
| [`hbs_lib_vec.hx`](../../../helixc/examples/hbs_lib_vec.hx) | a `Vec<i32>` built on the arena allocator |
| [`hbs_reference_500loc.hx`](../../../helixc/examples/hbs_reference_500loc.hx) | reference program exercising a broad slice of shipped features |

> **For AI agents:** the in-gate generics, traits, closures, and pattern-guard fixtures
> (`gen_*`, `t*_*`, `g*_*`, `V3_*` etc.) are **not** in `helixc/examples/`; they live under
> `stage0/helixc-bootstrap/corpus_gen/` and are referenced as `$GENC/...` in the gate
> (`scripts/gate_kovc.sh:322` onward). For generics-and-traits *examples* in this directory, the
> dogfood ML/AGI programs in §E.5 are the closest in-tree demonstrations; for the proven
> per-feature coverage, read the gate's `corpus_gen` rows directly.

---

## E.5 Stdlib demos, ML, and the AGI feature programs

This is the largest narrative group: programs that exercise the standard library
([21 modules under `helixc/stdlib/`](../../../helixc/stdlib/)) — autodiff, tensors, NN layers,
provenance, safety wrappers, checkpointing — and the compile-time "AGI" type-system features.

### The `dogfood_*` series (built incrementally, lowest to highest stage)

| File | Demonstrates |
|---|---|
| [`dogfood_01_one_param.hx`](../../../helixc/examples/dogfood_01_one_param.hx) | learn one parameter minimizing `(w-7)^2` by real gradient descent |
| [`dogfood_02_linreg.hx`](../../../helixc/examples/dogfood_02_linreg.hx) | linear regression over 4 training pairs (`y = w*x`) |
| [`dogfood_03_affine.hx`](../../../helixc/examples/dogfood_03_affine.hx) | fit `y = w*x + b` using f32 reflection cells |
| [`dogfood_04_xor_relu.hx`](../../../helixc/examples/dogfood_04_xor_relu.hx) | a 2-layer ReLU net touching XOR |
| [`dogfood_05_binary_classifier.hx`](../../../helixc/examples/dogfood_05_binary_classifier.hx) | sigmoid binary classifier trained with BCE + reverse-mode AD |
| [`dogfood_06_provenance_datalog.hx`](../../../helixc/examples/dogfood_06_provenance_datalog.hx) | Datalog-shaped reasoning over provenance-typed truth values |
| [`dogfood_07_provenance_sgd.hx`](../../../helixc/examples/dogfood_07_provenance_sgd.hx) | learn a fuzzy-logic rule via SGD with gradients through logic ops |
| [`dogfood_08_two_param_fuzzy_rule.hx`](../../../helixc/examples/dogfood_08_two_param_fuzzy_rule.hx) | two-parameter fuzzy rule learned via SGD |
| [`dogfood_09_knowledge_graph.hx`](../../../helixc/examples/dogfood_09_knowledge_graph.hx) | knowledge-graph reasoner: facts + rules + provenance recovery |
| [`dogfood_10_memory_tiers.hx`](../../../helixc/examples/dogfood_10_memory_tiers.hx) | memory-tier lifecycle: working / episodic / semantic recall |
| [`dogfood_11_spatial_frames.hx`](../../../helixc/examples/dogfood_11_spatial_frames.hx) | spatial-frame lifecycle: a 3-D point across reference frames |
| [`dogfood_12_temporal_lifecycle.hx`](../../../helixc/examples/dogfood_12_temporal_lifecycle.hx) | temporal-kind lifecycle transitions over an observation |
| [`dogfood_13_modal_lifecycle.hx`](../../../helixc/examples/dogfood_13_modal_lifecycle.hx) | modal/epistemic lifecycle: a proposition through modal upgrades |
| [`dogfood_14_causal_lifecycle.hx`](../../../helixc/examples/dogfood_14_causal_lifecycle.hx) | causal/intent lifecycle transitions |
| [`dogfood_15_agi_planning_loop.hx`](../../../helixc/examples/dogfood_15_agi_planning_loop.hx) | planning loop exercising all 5 semantic-type families together |
| [`dogfood_20_e2e_train_checkpoint.hx`](../../../helixc/examples/dogfood_20_e2e_train_checkpoint.hx) | end-to-end: train a 2-param model, then checkpoint it |
| [`dogfood_21_typed_security_stack.hx`](../../../helixc/examples/dogfood_21_typed_security_stack.hx) | the Tier-S/A typed-wrapper security stack |
| [`dogfood_22_full_wrapper_stack.hx`](../../../helixc/examples/dogfood_22_full_wrapper_stack.hx) | all 11 Tier-S/A wrappers in one compilable program |
| [`dogfood_23_property_proofs.hx`](../../../helixc/examples/dogfood_23_property_proofs.hx) | the 5 `@property` fns from `helixc/stdlib/safety.hx` with concrete inputs |

(`dogfood_16`–`19` are listed in §E.4 — they center on `Result`/`?` and pattern matching.)

### Standalone ML / numerics demos

| File | Demonstrates |
|---|---|
| [`nn_forward.hx`](../../../helixc/examples/nn_forward.hx) | forward pass `y = relu(W*x + b)` of a 1-hidden-layer net |
| [`helix_grad_descent.hx`](../../../helixc/examples/helix_grad_descent.hx) | scalar regression by gradient descent through Helix's own reverse-mode AD |
| [`gradient_descent.hx`](../../../helixc/examples/gradient_descent.hx) | one gradient step on `(x-3)^2` (gate ⇒ 42, §E.2) |
| [`hbs_sample_loss_fn.hx`](../../../helixc/examples/hbs_sample_loss_fn.hx) | small float-AD pipeline over the stdlib |
| [`sat_solver.hx`](../../../helixc/examples/sat_solver.hx) | a DPLL SAT solver (CNF satisfiability) |

### Agent / world-model demos (Q-learning and the AGI substrate)

These compose the stdlib AGI primitives into small grid-world agents. Several have companion
HTML/output artifacts in the directory (e.g. `dashboard.html`, `DEMO_OUTPUTS.txt`).

| File | Demonstrates |
|---|---|
| [`agi_demo.hx`](../../../helixc/examples/agi_demo.hx) | showcase of compile-time AGI type features (its header notes it is illustrative, not all codegen-backed) |
| [`agi_substrate_demo.hx`](../../../helixc/examples/agi_substrate_demo.hx) | all Phase 2/3/4 primitives composed in one program |
| [`self_improving_agent.hx`](../../../helixc/examples/self_improving_agent.hx) | flagship AGI-primitives demo combining multiple feature families |
| [`tutorial_agent.hx`](../../../helixc/examples/tutorial_agent.hx) | tutorial grid-world solver over the Phase-4 primitives |
| [`visual_agent.hx`](../../../helixc/examples/visual_agent.hx) | live agent on a 6×6 grid, printing the grid as it moves |
| [`dashboard_agent.hx`](../../../helixc/examples/dashboard_agent.hx) | 10×10 grid pathfinding agent with blocked cells |
| [`dashboard_qlearn.hx`](../../../helixc/examples/dashboard_qlearn.hx) | Q-learning agent that improves across episodes |
| [`dashboard_nn_agent.hx`](../../../helixc/examples/dashboard_nn_agent.hx) | neural-network agent on the same grid |
| [`multi_goal.hx`](../../../helixc/examples/multi_goal.hx) | Q-learning on a 10×10 grid with three goal cells |
| [`pickup_deliver.hx`](../../../helixc/examples/pickup_deliver.hx) | pickup-and-deliver agent over an extended state space |
| [`fog_of_war.hx`](../../../helixc/examples/fog_of_war.hx) | Q-learning without distance-based reward shaping |

> **Note:** [`agi_demo.hx`](../../../helixc/examples/agi_demo.hx) states in its own header that it
> "won't run" because it uses types without codegen — it is a *spec-illustrative* program for the
> compile-time type features, not an executable. Do not treat it as runnable.

> **For AI agents:** two files are prefixed with an underscore —
> [`_nn_compiled.hx`](../../../helixc/examples/_nn_compiled.hx) and
> [`_qlearn_compiled.hx`](../../../helixc/examples/_qlearn_compiled.hx). Their header comments name
> the dashboard agents they were generated from (`dashboard_nn_agent.hx` / `dashboard_qlearn.hx`),
> i.e. they are derived/compiled artifacts of those demos, not new programs. Prefer the named
> source files over the underscore-prefixed derivatives.

---

## E.6 The stdlib itself (21 modules)

The examples in §E.5 lean on the standard library. The modules are not in `helixc/examples/`;
they ship under [`helixc/stdlib/`](../../../helixc/stdlib/) and are part of the gate-proven
surface (the gate's library-backed corpus fixtures inline these shapes). The 21 modules:

| Module | Area |
|---|---|
| [`vec.hx`](../../../helixc/stdlib/vec.hx) | dynamic vector |
| [`hashmap.hx`](../../../helixc/stdlib/hashmap.hx) | hash map |
| [`string.hx`](../../../helixc/stdlib/string.hx) | arena-backed string |
| [`option.hx`](../../../helixc/stdlib/option.hx) | `Option` type |
| [`result.hx`](../../../helixc/stdlib/result.hx) | `Result` type |
| [`iterators.hx`](../../../helixc/stdlib/iterators.hx) | iterator helpers |
| [`csv.hx`](../../../helixc/stdlib/csv.hx) | CSV I/O |
| [`tensor.hx`](../../../helixc/stdlib/tensor.hx) | tensors |
| [`autodiff.hx`](../../../helixc/stdlib/autodiff.hx) | forward-mode autodiff |
| [`autodiff_reverse.hx`](../../../helixc/stdlib/autodiff_reverse.hx) | reverse-mode autodiff |
| [`nn.hx`](../../../helixc/stdlib/nn.hx) | neural-network layers |
| [`mnist.hx`](../../../helixc/stdlib/mnist.hx) | MNIST helpers |
| [`transcendentals.hx`](../../../helixc/stdlib/transcendentals.hx) | exp/tanh/gelu and friends |
| [`ieee754.hx`](../../../helixc/stdlib/ieee754.hx) | float bit-level helpers |
| [`checkpoint.hx`](../../../helixc/stdlib/checkpoint.hx) | model checkpointing |
| [`provenance.hx`](../../../helixc/stdlib/provenance.hx) | provenance-typed values |
| [`safety.hx`](../../../helixc/stdlib/safety.hx) | `@property` safety wrappers |
| [`agi_match.hx`](../../../helixc/stdlib/agi_match.hx) | AGI: matching |
| [`agi_memory.hx`](../../../helixc/stdlib/agi_memory.hx) | AGI: memory tiers |
| [`agi_search.hx`](../../../helixc/stdlib/agi_search.hx) | AGI: search |
| [`agi_world.hx`](../../../helixc/stdlib/agi_world.hx) | AGI: world model |

For a stdlib walkthrough, see Part IV (*planned*); this index only points at the files an
example program imports.

---

## E.7 GPU kernels (the `*_kernel.hx` family — 34 files)

`helixc/examples/` contains **34** GPU kernel programs (every file matching `*_kernel.hx`).
These are concrete, non-generic `@kernel` functions that `kovc` lowers to **PTX**. The kernel
family is the input corpus for the GPU back end and the capstone.

> **Residual:** the GPU path is **complete to PTX, not SASS**. `kovc` emits hand-auditable PTX;
> below PTX, NVIDIA's closed `ptxas`, the CUDA driver, the GPU hardware, and the C host launcher
> are trusted. The reference target is a single GPU (`sm_86`, RTX 3070 Laptop); kernel
> performance is a **fraction of cuBLAS** (~50–67.5% on that box), and the end-to-end capstone
> speedup is **7.0–8.7×** (Amdahl-bound), not ≥10×. Loss parity (the hard gate) holds at ~0%. See
> [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R for every residual, and
> Part VII (*planned*) for the full GPU story. Never read these kernels as "GPU machine code" —
> they stop at PTX.

How the kernels are gate-checked: the gate does **not** run PTX on a GPU. Instead it performs a
**PTX text regression** — it re-mints the PTX driver from the edited `kovc.hx` and byte-compares
the emitted PTX against a committed reference, for three anchored kernels
(`scripts/gate_kovc.sh` steps `[1]`, `[3]`). The three committed references are:

- [`vector_add_kernel.ref.ptx`](../../../helixc/examples/vector_add_kernel.ref.ptx) — anchors
  [`vector_add_kernel.hx`](../../../helixc/examples/vector_add_kernel.hx) (`scripts/gate_kovc.sh:44`).
- [`tiled_matmul_kernel.ref.ptx`](../../../helixc/examples/tiled_matmul_kernel.ref.ptx) — anchors
  [`tiled_matmul_kernel.hx`](../../../helixc/examples/tiled_matmul_kernel.hx) (`scripts/gate_kovc.sh:162`).
- [`tf32_matmul_kernel.ref.ptx`](../../../helixc/examples/tf32_matmul_kernel.ref.ptx) — the TF32
  Tensor-Core reference (committed alongside its kernel).

> **For AI agents:** the gate's GPU leg is a **pure-text** PTX regression (re-emit and
> `cmp -s` against the committed `.ref.ptx`); it needs no GPU and no `ptxas`. A *missing* committed
> reference is treated as a **real gate failure**, not a benign GPU-absent skip
> (`scripts/gate_kovc.sh:49`). Genuine GPU-hardware execution lives elsewhere
> (`scripts/capstone_audit.sh`), not in this gate.

The 34 kernels, grouped by what they emit:

**Elementwise / first-light (4).**
[`vector_add_kernel.hx`](../../../helixc/examples/vector_add_kernel.hx) (`c[i]=a[i]+b[i]`, the
DoD "GPU executes" kernel),
[`vector_mul_kernel.hx`](../../../helixc/examples/vector_mul_kernel.hx) (`c[i]=a[i]*b[i]`),
[`vector_reverse_kernel.hx`](../../../helixc/examples/vector_reverse_kernel.hx) (`c[i]=a[n-1-i]`,
first kernel reading a scalar param),
[`gpu_test_exp_kernel.hx`](../../../helixc/examples/gpu_test_exp_kernel.hx).

**Activations & their backward passes (5).**
[`gpu_relu_kernel.hx`](../../../helixc/examples/gpu_relu_kernel.hx) (`max(a[i],0)`),
[`gpu_gelu_kernel.hx`](../../../helixc/examples/gpu_gelu_kernel.hx) (tanh-approx GELU),
[`gpu_gelu_backward_kernel.hx`](../../../helixc/examples/gpu_gelu_backward_kernel.hx),
[`gpu_softmax_kernel.hx`](../../../helixc/examples/gpu_softmax_kernel.hx) (row-wise, first reduction),
[`gpu_softmax_backward_kernel.hx`](../../../helixc/examples/gpu_softmax_backward_kernel.hx).

**LayerNorm forward / backward (7).**
[`gpu_layernorm_kernel.hx`](../../../helixc/examples/gpu_layernorm_kernel.hx),
[`gpu_layernorm_fwd_save_kernel.hx`](../../../helixc/examples/gpu_layernorm_fwd_save_kernel.hx),
[`gpu_layernorm_backward_dx_kernel.hx`](../../../helixc/examples/gpu_layernorm_backward_dx_kernel.hx),
[`gpu_layernorm_backward_dgb_kernel.hx`](../../../helixc/examples/gpu_layernorm_backward_dgb_kernel.hx),
[`gpu_layernorm_backward_dgb_pm_kernel.hx`](../../../helixc/examples/gpu_layernorm_backward_dgb_pm_kernel.hx),
[`gpu_row_mean_kernel.hx`](../../../helixc/examples/gpu_row_mean_kernel.hx),
[`gpu_ce_softmax_grad_kernel.hx`](../../../helixc/examples/gpu_ce_softmax_grad_kernel.hx)
(cross-entropy + softmax gradient).

**Matmul: naive, scaled, transposed (4).**
[`naive_matmul_kernel.hx`](../../../helixc/examples/naive_matmul_kernel.hx) (one thread per output cell),
[`gpu_matmul_abt_kernel.hx`](../../../helixc/examples/gpu_matmul_abt_kernel.hx) (`A @ B^T`, unscaled),
[`gpu_matmul_atb_kernel.hx`](../../../helixc/examples/gpu_matmul_atb_kernel.hx) (`A^T @ B`),
[`gpu_qkt_kernel.hx`](../../../helixc/examples/gpu_qkt_kernel.hx) (scaled `Q @ K^T` for attention).

**Tiled (shared-memory) and Tensor-Core matmul (4).**
[`tiled_matmul_kernel.hx`](../../../helixc/examples/tiled_matmul_kernel.hx) (SMEM-tiled GEMM, the
GPU critical-path kernel; PTX-regression-anchored),
[`tiled_matmul_abt_kernel.hx`](../../../helixc/examples/tiled_matmul_abt_kernel.hx) (tiled `A @ B^T`),
[`tiled_matmul_atb_kernel.hx`](../../../helixc/examples/tiled_matmul_atb_kernel.hx) (tiled `A^T @ B`),
[`tf32_matmul_kernel.hx`](../../../helixc/examples/tf32_matmul_kernel.hx) (TF32 `mma.sync`
Tensor-Core matmul; PTX-regression-anchored).

**Block-reduction variants (5).**
[`layernorm_blockred_kernel.hx`](../../../helixc/examples/layernorm_blockred_kernel.hx),
[`layernorm_fwd_save_blockred_kernel.hx`](../../../helixc/examples/layernorm_fwd_save_blockred_kernel.hx),
[`layernorm_backward_dx_blockred_kernel.hx`](../../../helixc/examples/layernorm_backward_dx_blockred_kernel.hx),
[`softmax_blockred_kernel.hx`](../../../helixc/examples/softmax_blockred_kernel.hx),
[`softmax_backward_blockred_kernel.hx`](../../../helixc/examples/softmax_backward_blockred_kernel.hx)
(these `*_blockred_*` forms use a cooperative in-block reduction).

**Fused attention and optimizer / scaling (the remainder).**
[`flash_attention_kernel.hx`](../../../helixc/examples/flash_attention_kernel.hx) (fused
flash-style attention via a single `__flash_attention` intrinsic; online softmax),
[`gpu_adam_kernel.hx`](../../../helixc/examples/gpu_adam_kernel.hx) (in-place Adam step),
[`gpu_affine_kernel.hx`](../../../helixc/examples/gpu_affine_kernel.hx) (affine `W*x+b` form),
[`gpu_scale_inplace_kernel.hx`](../../../helixc/examples/gpu_scale_inplace_kernel.hx) (in-place scale),
[`gpu_scale_rt_kernel.hx`](../../../helixc/examples/gpu_scale_rt_kernel.hx) (runtime-scalar scale).

That is all 34 `*_kernel.hx` files. To read one as a Fragment, here is the documented shape of
the first-light kernel — an elementwise add over f32 global arrays.

**Fragment** (excerpt of the header of
[`helixc/examples/vector_add_kernel.hx`](../../../helixc/examples/vector_add_kernel.hx); not a
complete program — the body is omitted):

```helix
// GPU first-light kernel (Helix v1.0 DoD criterion #3 -- "GPU executes").
//
// A concrete, NON-generic @kernel: c[i] = a[i] + b[i] over f32 global arrays,
```

> **For AI agents:** to *change* any kernel's emitted PTX intentionally, you must re-mint and
> re-commit its `.ref.ptx` with a stated reason (the charter "step 2" discipline noted at
> `scripts/gate_kovc.sh:42`), or the gate fails the PTX regression. An x86-only `kovc` change must
> leave all three references byte-identical.

---

## E.8 Quick lookup — "I want an example of…"

| I want… | Start at |
|---|---|
| the absolute minimum program | [`exit42.hx`](../../../helixc/examples/exit42.hx) (gate ⇒ 42) |
| printing to stdout / writing a file | [`hello_world.hx`](../../../helixc/examples/hello_world.hx) |
| enums + structs + match | [`hbs_sample_enum_struct.hx`](../../../helixc/examples/hbs_sample_enum_struct.hx) (gate ⇒ 129) |
| payload enums + extraction | [`hbs_sample_option.hx`](../../../helixc/examples/hbs_sample_option.hx) (gate ⇒ 42) |
| recursion | [`hbs_sample_recursion.hx`](../../../helixc/examples/hbs_sample_recursion.hx) (gate ⇒ 120) |
| struct-destructuring patterns | [`dogfood_18_pat_struct_showcase.hx`](../../../helixc/examples/dogfood_18_pat_struct_showcase.hx) (gate ⇒ 42) |
| `Result` and `?` | [`dogfood_16_result_basic.hx`](../../../helixc/examples/dogfood_16_result_basic.hx), [`dogfood_17_try_operator.hx`](../../../helixc/examples/dogfood_17_try_operator.hx) |
| a real compiler pass in Helix | [`hbs_sample_constant_fold.hx`](../../../helixc/examples/hbs_sample_constant_fold.hx), [`metacircular_eval.hx`](../../../helixc/examples/metacircular_eval.hx) |
| autodiff / training | [`gradient_descent.hx`](../../../helixc/examples/gradient_descent.hx) (gate ⇒ 42), [`dogfood_01_one_param.hx`](../../../helixc/examples/dogfood_01_one_param.hx) |
| a neural net forward pass | [`nn_forward.hx`](../../../helixc/examples/nn_forward.hx) |
| a GPU kernel (elementwise) | [`vector_add_kernel.hx`](../../../helixc/examples/vector_add_kernel.hx) |
| a GPU GEMM | [`tiled_matmul_kernel.hx`](../../../helixc/examples/tiled_matmul_kernel.hx), [`tf32_matmul_kernel.hx`](../../../helixc/examples/tf32_matmul_kernel.hx) |
| a Q-learning agent | [`dashboard_qlearn.hx`](../../../helixc/examples/dashboard_qlearn.hx) |

---

**Next:** **Appendix F — The trusted computing base** *(planned)* — what the trust chain still
rests on (the irreducible TCB), and the residuals that bound every claim in this book. Until it
ships, the authoritative record of the TCB and every residual is
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R.
