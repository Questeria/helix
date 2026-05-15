# Wave 1 Research Findings (2026-05-04 overnight)

Four parallel research agents synthesized concrete features for Helix's
evolution. Below is the consolidated prioritized list of additions —
the union and ranking of all four agent outputs.

## Cross-cutting themes (multiple agents converged on these)

These appeared in 3+ of the 4 reports and represent the highest-confidence
recommendations:

1. **Hash-consed / content-addressed AST.** Every top-level def is keyed
   by SHA of its normalized AST. Two structurally-equal expressions
   share identity. Enables: e-graph rewriting, dedupe of gradient
   sub-expressions, mechanical verification of "did edit X break Y."
   Effort: 3-4 weeks.

2. **Pattern matching with guards + nested destructuring.** Every agent
   rated this Tier 1 even though our roadmap had it Tier 4. Required
   for self-hosted compiler passes (AST-walking is the bulk of compiler
   code). Effort: 2 weeks.

3. **Algebraic effect handlers (Koka-style).** Replace flat @effect
   tags with row-polymorphic effects + handlers. Lets the AGI run the
   same code under sandbox/replay/mock handlers. Also lets AD become a
   library-level effect rather than a compiler pass. Effort: 4-8 weeks.

4. **Refinement-reflected verifiers.** Verifiers accept proof terms
   (not just bool). Discharge via SMT (Z3/CVC5) at modify-time.
   Effort: 6-8 weeks.

5. **Total-by-default with @partial annotation.** Functions structurally
   recursive by default; non-total ones must opt in. Makes verifiers,
   partial evaluators, and NbE evaluators decidable on the total subset.
   Effort: 2-3 weeks.

## Helix-specific architecture additions

6. **Helix Bootstrap Subset (HBS) spec.** Carve out the minimal fragment
   needed to host a self-hosted compiler (i32, structs, enums, arrays,
   if/while, fn pointers, match w/o guards, @effect; no generics, GC,
   D<T>, async). Freeze the grammar. 1 week.

7. **AST as a first-class Helix value (not just a Python type).**
   Promote `quote` from "stable hash" to a real algebraic data type
   defined in Helix. Lets passes be ordinary Helix functions. 3 weeks.

8. **Migrate compiler passes from Python into Helix one at a time.**
   Start with DCE, CSE, const-fold. After ~10 passes migrated, the
   Python helixc is mostly a parser+codegen shell. 1 week per pass.

9. **E-graph + equality saturation as the optimizer.** Replace the
   pass pipeline with an e-graph; rewrites are first-class data. The
   AGI extends the compiler by adding rewrites, not pipeline stages.
   4-6 weeks for MVP.

## AGI-specific (the strategic moat)

10. **Provenance-typed `D<S, T>` with semiring parameter.** Generalize
    differentiable types to carry provenance semirings (max-min,
    top-k-proofs, gradient). Foundation for differentiable
    Datalog / neuro-symbolic reasoning. Scallop/Lobster pattern.
    3-5 weeks.

11. **Trace-equivalence verifier (TEV) using Code World Models.**
    Use an LLM trace oracle (Meta CWM) to predict execution traces;
    accept modify iff the new code's predicted trace matches old's
    on a witness set. Closes the "verify a refactor with no spec"
    hole. 3-5 weeks given effect handlers.

12. **Tactic-style proof terms for verifiers (Lean 4 model).** Let
    verifiers be tactic blocks that build proof terms. Failure
    surfaces residual goals as next-step search targets — exactly
    the LLM-driven verification interface. 4-5 weeks.

## ML architecture choices for Kovostov AGI

The ML-from-scratch agent recommended four genuinely-novel directions
fitting both Helix's primitives and consumer hardware (RTX 5090 +
Ryzen 9950X):

- **RWKV-7 / Mamba-3** as the substrate (linear-time RNN, infinite
  context, fits Helix's verifier-gated state mutation pattern).
- **Test-Time Training (TTT)** with fast weights: every inference also
  trains on its own self-supervised loss. Maps directly to verifier-
  gated weight mutations.
- **Recurrent depth / latent reasoning** (Geiping et al. 2025): unroll
  a recurrent block at test time without emitting CoT tokens. Helix's
  effect tracking can guarantee convergence-or-bail.
- **RLVR (RL with Verifiable Rewards):** Skip human labels; reward IS
  the verifier function. Helix's effect+verifier system natively
  encodes this without extra runtime infrastructure.
- **Energy-Based Models for reasoning:** dE/dt ≤ 0 monotonicity is a
  natural Helix invariant.

The ML-from-scratch agent concluded the single largest novelty win is
a Helix-native pipeline where weight updates are conditional on
Lean-verified properties of the resulting model, with rewards from
formal verifiers rather than human labels. The intended differentiator is
combining verifier-gated weight updates, formal reward signals, and Helix's
effect/provenance substrate in one language/runtime.

## Recommended implementation order (next ~4 months)

**Tier 1 (weeks 1-4): foundation**
- HBS spec frozen
- Pattern matching with guards
- Hash-consed AST
- Total-by-default annotation

**Tier 2 (weeks 5-10): extensibility**
- AST as Helix value
- Migrate first 3 passes (DCE, CSE, const-fold) to Helix
- E-graph rewriting layer
- Refinement-reflected verifiers (SMT-discharged)

**Tier 3 (weeks 11-16): AGI primitives**
- Algebraic effect handlers
- Provenance-typed D<S, T>
- Trace-equivalence verifier
- Tactic-style proof terms

**Tier 4 (weeks 17+): Kovostov AGI bring-up**
- RWKV-7 / Mamba-3 substrate in Helix
- TTT + RLVR loops with Helix verifier as reward
- Latent reasoning architecture
- Lean 4 bridge for formally-verified weight updates

## Sources by agent

- [agent A (AI-for-AI lang)](research_outputs/wave1-ai-for-ai.md) — Unison, hash-consing, e-graphs, NbE, refinement types, effect handlers, MetaOCaml staging
- [agent B (ML from scratch)](research_outputs/wave1-ml-scratch.md) — RWKV-7, Mamba-3, TTT, recurrent depth, EBM, RLVR, JEPA, hypernets, distillation, neuro-symbolic
- [agent C (compiler self-improve)](research_outputs/wave1-compiler.md) — HBS, AST-as-value, e-graph rewrites, AlphaEvolve precedent, CakeML self-bootstrap
- [agent D (verifier semantics)](research_outputs/wave1-verifiers.md) — D<Logic<T>>, refinement-reflected verifiers, handler-typed effects, trace-equivalence, tactic verifiers

(The full agent outputs are in the conversation log; this file is the synthesized actionable summary.)
