# Helix Roadmap

This document captures the prioritized feature list synthesized from two
deep-research passes (2026-05-04). It's a forward-looking plan; not
everything here will land, and priorities will shift as dogfooding
reveals which features actually matter.

## Current state (350 tests passing)

- Working from-scratch x86-64 ELF compiler
- Forward + reverse-mode symbolic AD with chain rules for __exp, __log,
  __sin, __cos, __sqrt, __relu, __sigmoid
- IR-level effect/capability enforcement
- Verifier-gated reflection runtime (64 mutable cells, real verifier
  function calls with SysV ABI)
- f32 reflection cells (splice_f / modify_f)
- 4 dogfood programs running real gradient descent
- Stdlib for transcendentals auto-included

## Tier 1 — must-have next (do first)

These are blockers for any real ML training, in priority order.

1. **Transcendentals** ✅ DONE — Taylor series approximations for
   exp/log/sin/cos/sqrt + their AD chain rules. Stdlib auto-included.

2. **AD across user-defined function calls.** `grad_rev(loss)` where
   `loss` calls helper functions currently treats those calls as
   opaque (gradient = 0). Need to inline at AD time, or chain-rule
   through call ops by analytically differentiating the callee. **3-4
   weeks.** Without this every loss must be one big inlined function.

3. **Multi-output reverse-mode AD.** Currently `grad_rev(f, n)` runs a
   separate AD pass per parameter index. For real models with thousands
   of parameters this is N× too expensive. The reverse-mode engine
   already collects per-parameter buckets — extend the API to return
   the full dict and emit the gradient as a function returning multiple
   values (via output array).

4. **Strings + file I/O** with capability-typed `@effect(io.read_file)`.
   Without these, programs only output exit codes — no model can train
   end-to-end. **2-3 weeks.**

5. **Stack-passed overflow args.** SysV ABI's xmm0..xmm7 covers the
   first 8 float params; the 9th must be passed on the stack. Hit
   during XOR perceptron dogfooding. **1 week.**

## Tier 2 — high value (do after Tier 1)

6. **Tensor codegen** with explicit memory-space movement
   (HBM/SMEM/REG/TMEM). The type system has tile types; the codegen
   doesn't lower them. Without this, no SIMD elementwise, no matmul,
   no real performance. **2-3 months.**

7. **JAX-style pytrees.** `grad(loss)(model)` where `model` is a nested
   struct. Composes `grad/vmap/jit` over arbitrary tree-structured
   parameter sets. Critical for real model architectures. **2 weeks
   on top of tier-1 #3.**

8. **Triton-style autotune.** `@autotune(BLOCK_M=[16,32,64], ...)` for
   `@kernel` functions. Compiler emits N variants, runtime picks per
   shape. Builds on tile codegen. **2 weeks.**

9. **Mojo-style parametric structs.** `Linear[In: size, Out: size, T:
   type]` monomorphized at compile time per shape and dtype. Required
   for dtype-flexible code (f32/bf16/fp8 specialized). **3 weeks.**

## Tier 3 — strategic differentiator

10. **Provenance-typed neuro-symbolic primitives** (`D<Logic<T>>`).
    Differentiable relational data with provenance semirings (Scallop /
    Lobster pattern). Statically-typed, gradient-traced symbolic
    reasoning. **NO OTHER AI LANGUAGE HAS THIS.** This is the
    strategic moat against Mojo/JAX/Triton — they're tensor-only.
    Unlocks trainable knowledge graphs, gradient-traced planners,
    end-to-end-differentiable retrieval. **4-6 weeks for MVP.**

11. **Trace-based introspection** for `quote`/`modify`. Capture
    execution traces (variable values, control-flow) so verifiers can
    check trace-equivalence on a held-out input set. Aligns with Meta
    CWM's empirical finding that traces are the right substrate for
    AI to reason about its own code. **3-4 weeks.**

12. **Lean-4-style proof-carrying terms.** Verifiers receive a Proof
    object, not a bool. The compiler validates the proof. The
    difference between sandboxing and provable safety. **Large**
    (months) but bounded — the kernel is a few thousand lines.

## Tier 4 — table-stakes infrastructure

13. **Module system** with content-addressed packages (modules
    referenced by AST hash). **2 weeks.**

14. **Result<T,E> + ? operator.** Error handling beyond panic. **1
    week.**

15. **Pattern matching with guards + or-patterns + nested
    destructuring.** Critical for AST-walking inside quote/splice.
    **2 weeks.**

16. **Borrow checker.** Compile-time aliasing safety. Eliminates an
    entire class of bugs in tile/buffer code. **Months.**

17. **Multiple dispatch** for tile/tensor ops. Julia-style. Natural
    fit for kernel selection by `(tile<bf16,smem>, tile<f32,reg>)`
    pairs. **3 weeks.**

## Deliberately NOT doing

- Mojo's full Python-compatibility race (lost on calendar time)
- Full JSON/HTTP/async runtime (scope explosion)
- JIT compilation (AOT-with-cached-specialization is enough)
- MCP/tool-calling primitives (wrong layer; agent harness lives above
  the language)
- Multi-vendor GPU support before NVIDIA path is solid

## Notes on the AGI angle

The quietly-load-bearing observation from research agent #1: Helix's
combination of (a) compile-time-enforced effect system, (b)
verifier-gated reflection runtime, (c) source-level reverse-mode AD,
(d) memory-tier types, and (e) future provenance-typed
neuro-symbolic primitives is a genuinely unoccupied niche. No other
AI language hits this combination. The "better than existing on the
AGI axis" target is achievable; the language doesn't need to win on
GEMM throughput to win on AGI.

## Sequencing note

Tier 1 + Tier 2 #6 together are roughly 4-5 months of work and
unblock real model training. Tier 3 #10 (neuro-symbolic primitives)
should start scoping in parallel since it's the strategic moat and
the work is mostly orthogonal to GPU codegen.
