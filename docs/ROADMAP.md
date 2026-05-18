# Helix Roadmap

This document captures the prioritized feature list synthesized from two
deep-research passes (2026-05-04). It's a forward-looking plan; not
everything here will land, and priorities will shift as dogfooding
reveals which features actually matter.

## Current state (Stages 35-48 CLOSED 2026-05-16 to 2026-05-17)

Burst summary (14 stages closed in <48h, all via the
3-clean-gate protocol):

- **Stage 35** — AI/ML capability push (restart 65 closure).
- **Stage 36** — Strategic AGI features: Tier 3 #10
  provenance-typed neuro-symbolic primitives shipped
  end-to-end (Logic<T>, fuzzy_and/or/not/xor/implies,
  ARENA_PUSH_PAIR/TRIPLE IR opcodes, provenance.hx stdlib,
  4 dogfood programs).
- **Stages 37-41** — AGI semantic-type quintet:
  - Stage 37: memory tier (WorkingMem/EpisodicMem/SemanticMem/
    ProceduralMem).
  - Stage 38: spatial frames (WorldFrame/RobotFrame/CameraFrame
    + 6 cross-frame transforms).
  - Stage 39: temporal kinds (Past/Present/Future/Eternal +
    4 transitions).
  - Stage 40: modal/epistemic (Known/Believed/Goal/Uncertain
    + 2 audited upgrades confirm/act_on; F1 cross-modal
    laundering guard catches AI-safety category mistakes).
  - Stage 41: causal/intent (Cause/Effect/Joint/Independent
    + 3 transitions propagate/aggregate/isolate).
  All 5 families compose orthogonally at the type level
  (e.g. `Known<Past<Cause<f32>>>` = "directly observed past
  cause"), zero runtime cost (Phase-0 identity-lowered).
- **Stage 42** — quintet cohesion proven via dogfood_15 (4-deep
  wrapper stack: `Known<Present<WorldFrame<Cause<i32>>>` →
  `Believed<Future<WorldFrame<Effect<i32>>>`).
- **Stage 43** — deferred-items cleanup (AD-set rename, F5
  arity arms, M1 double-wrap rejection with direction-aware
  hints across all 5 families).
- **Stage 44** — Tier 1 #5 stack-passed overflow float args
  (9+ float args now pass via SysV stack for both CALL and
  FFI_CALL; first Tier-1 ML blocker closed).
- **Stage 45** — ROADMAP status drift refresh + Stage 46+
  sequencing.
- **Stage 46** — Tier 4 #14 Inc 1 Result<T, E> typecheck-
  side scaffolding. First two-parameter wrapper family in
  the Helix type system. 8 builtins (Ok/Err/unwrap_ok/
  unwrap_err/is_ok/is_err/map_ok/map_err). 4 of 8 are
  Phase-0-typecheck-rejected pending the Stage 48+ runtime
  tag; the other 4 ship with a 3-layer wrong-arm safety
  net (TyUnknown-hint provenance + `_result_constructor_
  provenance` map + Assign-arm invalidation + non-
  constructor-RHS pop) that catches the 4 distinct silent-
  miscompile patterns (inference, typed-let, mutable
  reassignment, cross-function name leak) audit lanes
  surfaced across the 3-gate closure.
- **Stage 47** — slim consolidation: drop the Stage 43
  deferred `_FRAME_IDENTITY_AD_NAMES` backwards-compat
  alias (3-stage grace period elapsed), refresh ROADMAP
  "Current state" to include Stage 46, re-sequence Stage
  48-50 picks.
- **Stage 48** — Tier 4 #14 Inc 2 `?` propagation operator
  (parser desugar `expr?` → `__try(expr)` + typecheck
  enclosing-fn return-type check + IR identity-lowering
  matching Phase-0 Ok-shape stance). 18 stage tests + 27
  Stage 46 tests = 45 Result-family tests green. Cumulative
  4-gate audit cycle (gate-1 F2 typed-let Err provenance
  HIGH, gate-2 F1 inner-block shadow HIGH + M5 cross-fn
  carry FIX, gate-3 G3-F1a/b/c outer-name-mutation 3-vehicle
  cascade HIGH, gate-3 verification 3/3 CLEAN) — Stage 46's
  cascading-defect rhythm repeated; same convergence on a
  sound design. Stage 49 runtime tag will eliminate the
  F1-dynamic / F5-aggregate / F6-assign / MED-1-map_ok
  Phase-0 known-defect equivalence class with one fix.
  Result<T,E> in fn-signature positions now lowers to T
  (the Ok inner) — prerequisite for `?` to compile.

16 dogfood programs total (dogfood_17_try_operator added).
Self-host cascade still byte-identical G2..G4 fixpoint
throughout the entire burst.

## Next-stage sequencing (post-Stage-45)

Re-sequenced after Stage 46-47 closed:

- **Stage 46** ✅ DONE — Tier 4 #14 Inc 1 Result<T,E>
  typecheck-side scaffolding (3-gate closure caught 4
  silent-miscompile patterns; 3-layer wrong-arm safety net
  shipped).
- **Stage 47** ✅ DONE — slim consolidation: dropped Stage
  43 deferred `_FRAME_IDENTITY_AD_NAMES` alias + ROADMAP
  refresh.
- **Stage 48** ✅ DONE — Tier 4 #14 Inc 2 `?` propagation
  operator (parser desugar + typecheck + Phase-0 identity
  IR lowering). 4-gate audit cascade with 9 fixed + 13
  deferred to Stage 49. Cascading-defect rhythm matched
  Stage 46; final 3-lane verification CLEAN.
- **Stage 49** ✅ DONE 2026-05-17 — Tier 4 #14 Inc 3
  runtime Ok/Err tag. UNLOCKED the 4 previously-rejected
  builtins (is_ok, is_err, map_err, unwrap-wrong-arm) AND
  eliminated the whole Phase-0 Result-defect equivalence class
  (F1-dynamic / F5-aggregate-field / F6-conditional-assign /
  MED-1-map_ok) in one fix — `?` is now a real conditional-
  branch IR arm with packed-i64 RESULT_PACK/RESULT_TAG/
  RESULT_PAYLOAD opcodes; unwrap_ok/unwrap_err carry a runtime
  tag-check that panics on wrong-arm; Stage 48's typecheck
  guards retained as defense-in-depth. 4-gate audit cascade
  (gates 1+2+3+4) caught 5 HIGHs total, all fixed inline; gate-4
  verification CLEAN. 48 Stage 49 tests + 78 Stage 46+48
  unchanged. Self-host cascade preserved. dogfood_16 + dogfood_17
  still exit 42.
- **Stage 50** (ABORTED 2026-05-17): bootstrap `grad_rev_all`
  infrastructure swap. Inc 1 (commit f4e94fc) + Inc 2 (commit
  76b7735) added multi-bucket helpers and swapped the production
  caller, but gate-1 silent-failure audit caught a HIGH cascade-
  break: `scripts/selfhost_cascade.py` fails at G2 with SIGILL
  (exit 132). Bisection showed the regression isn't algorithmic
  in the new helpers — adding even 2 trivial `__probe_a/b` fns
  to Stage 49 HEAD reproduces the SIGILL. Hidden coupling between
  bootstrap source-size and the Python seed compiler. The smaller
  `test_selfhost_cascade.py` unit tests passed because they stub
  the report formatter rather than running the real cascade —
  a coverage gap the Stage 49 baseline didn't surface. Stage 50
  reverted in commit [next] (parser.hx restored to a410b67).
  Closure docs + audit findings retained as historical record
  (docs/stage50-plan-2026-05-17.md + audit-stage50-*.md).
- **Stage 50 follow-up** (replaces both Stage 50 and Stage 51):
  root-cause the seed-compiler source-size fragility FIRST,
  then port the multi-bucket infrastructure + single-walk
  algorithm together once the cascade can tolerate source
  changes. Estimated 2-3 stages depending on how the
  fragility's root cause splits.
- **Stage 52** (in flight 2026-05-17, gates 1-13 + Inc 1-12
  shipped, 21+ launder paths caught via 9 wrapper-AST kinds,
  awaiting 3-clean-gate closure confirmation): modal-origin
  taint-tracking pass closing the Stage 40 closure gate-1 H1
  known limitation ("let-binding bypass of F1 syntactic guard").
  Inc 8 (e9d3d6d) UnsafeBlock arm; Inc 9 (006df58) F2 Literal
  propagation + ModalKind runtime guard at _register_fn; Inc 10
  (40a791d) Cast arm; Inc 11 (9ab8123) Unary + Binary arms
  (proactive cascade-break — scanned all 32 A.Expr nodes for
  wrapper-class gaps); Inc 12 (this commit) cache-at-block-exit
  for inner-let-bound Name lookup after scope pop. Wrapper-AST
  coverage table now: Name, Call, Block, UnsafeBlock, Cast,
  Unary, Binary, If, Match.
  Inc 1 (c274059), Inc 2 (2925121), Inc 3 (c9d8915) shipped the
  initial three launder paths (let-binding, while/for Assign,
  match-arm). Inc 5 (1fbebe2) shipped loop body union. Inc 6
  (0d133c9) shipped recursive yield-from-modal detection. Inc 7
  (this gate-10 fix) unified the builtin into_X consult through
  `_modal_origin_of_expr`, closing 17+ distinct launder paths via:
  - PatBind taint propagation (gate-4 HIGH-1, ccca046)
  - PatBind hoisted above guard check (gate-5 HIGH-1, fb9ad42)
  - Call-form match scrutinee (gate-6 CRITICAL-1, fb9ad42)
  - Name-alias let/Assign (gate-6 CRITICAL-2, fb9ad42)
  - PatOr-of-same-PatBind (gate-6 CRITICAL-3, fb9ad42)
  - Cleared-vs-installed refinement in A.If/A.Match union
    (gate-6 latent-bug fix + gate-7 kept_somewhere extension)
  - `_last_modal_assigns_popped` defensive clear (gate-7 type-
    design HIGH-1)
  - Unified `_modal_origin_of_expr` helper at all 3 install sites
  - Gate-7 silent-failure HIGH-3: if-no-else / match-arm clear
    with identity-arm preserved → FIRE (semantic flip from
    drop-on-conflict to safety-first conservative-fire)
  See `docs/stage52-progress-2026-05-17.md` gate-N closure
  subsections for cascading-defect rhythm details. Loop-body
  union (gate-7 HIGH-1+2) deferred to Stage 52 Inc 5 or rolled
  into Stage 53. Helper-fn indirection deferred to Stage 53
  (different defect class — inter-procedural taint).
- **Stage 53** (Inc 1+2 shipped 2026-05-17, commits 179678d +
  2550492): helper-fn indirection taint propagation. **Inc 1
  CLOSED Stage 40 H1 in full** — the LAST modal-launder bypass
  is now caught. Implementation: `_fn_modal_return_kind` dict
  populated in Pass 1 (`_register_fn`) + `_modal_origin_of_expr`
  extension + call-site launder check (mirror of F1 into_X
  pattern) + `_MODAL_UPGRADE_HINT` hoist. All 3 install sites
  (Let-RHS, Assign-RHS, match-scrutinee) get Stage 53 coverage
  automatically via the unified helper. Inc 2 added regression
  pins for Assign + match-scrutinee paths. Diagnostic message:
  "launders Uncertain<T> into Known<T> via helper-fn indirection".
  Stage 53 Inc 3+ deferred: inter-procedural taint flow when
  helper body itself contains laundering patterns (currently
  caught by intra-fn Stage 52 closures within each fn body
  individually).
(Gate-2 code-review M2: stale `(proposed)` entries for Stages 50,
51, 52 removed — they were superseded by the SHIPPED / ABORTED
entries above. Stages 50, 51, 52 now live under their actual
status in the "Next-stage sequencing" block earlier in this file.)

Re-evaluate at the next stage close.

Re-evaluated post-Stage-48: Stage 49 promoted to highest-
payoff next pick (eliminates the entire Phase-0 Result
known-defect equivalence class in one runtime-tag fix).

- Working from-scratch x86-64 ELF compiler
- Forward + reverse-mode symbolic AD with chain rules for __exp, __log,
  __sin, __cos, __sqrt, __relu, __sigmoid
- IR-level effect/capability enforcement
- Verifier-gated reflective-cell scaffold (64 mutable cells, real verifier
  function calls with SysV ABI; real runtime AST reflection remains future work)
- f32/f64 reflection cells (splice_f / splice_f64 / modify_f / modify_f64)
- 15 dogfood programs/tests running real gradient descent + provenance-typed Datalog + SGD-learns-a-fuzzy-rule + two-param fuzzy-rule learning + knowledge-graph reasoner with provenance recovery + memory-tier lifecycle reasoner + spatial-frame lifecycle reasoner + temporal-kind lifecycle reasoner + modal/epistemic lifecycle reasoner + causal/intent lifecycle reasoner + AGI quintet cohesion planning-loop + a self-improving-agent flagship that composes them (16 programs total: 15 dogfood + 1 flagship; see `helixc/examples/dogfood_*.hx` and `helixc/examples/self_improving_agent.hx`. The newest, `dogfood_15_agi_planning_loop.hx`, demonstrates Stage 42 Increment 1 — a robot perception-plan cycle carrying values through 4-deep wrapper stacks `Known<Present<WorldFrame<Cause<i32>>>>` → `Believed<Future<WorldFrame<Effect<i32>>>>` end-to-end)
- Stdlib for transcendentals auto-included
- Stage 35 status: CLOSED 2026-05-16 at restart 65 (3/3 clean audit gates; full ledger in docs/stage35-progress-2026-05-15.md)

## Tier 1 — must-have next (do first)

These are blockers for any real ML training, in priority order.

1. **Transcendentals** ✅ DONE — Taylor series approximations for
   exp/log/sin/cos/sqrt + their AD chain rules. Stdlib auto-included.

2. **AD across user-defined function calls.** `grad(loss)` and
   `grad_rev(loss)` inline supported pure helper calls before
   differentiation. Opaque/bodyless calls now fail closed instead of
   producing a zero-gradient surrogate. Remaining work is broader
   chain-rule registration and richer helper coverage. **In progress.**

3. **Multi-output reverse-mode AD.** ✅ DONE (Python-side) —
   `differentiate_reverse(expr, param_names)` in
   `helixc/frontend/autodiff_reverse.py` produces a
   `dict[param_name -> gradient_expr]` in ONE walk via per-param
   bucket accumulation. `grad_rev_all(f)(p1, p2, base)` is wired
   end-to-end and used by `dogfood_05_binary_classifier.hx` etc.
   **Bootstrap-side TODO**: `helixc/bootstrap/parser.hx:5402`
   still does N-walks (one per param). A future stage will
   port the single-walk algorithm to the in-Helix compiler.

4. **Richer strings + file I/O** with capability-typed
   `@effect(io.read_file)`. Basic literal/string diagnostic IO (`print_str`,
   `print_int`) and narrow file builtins exist; Stage 35 still needs the
   capability-typed dataset/checkpoint workflows and broader string/file APIs
   required for end-to-end model training. **2-3 weeks.**

5. **Stack-passed overflow args.** ✅ DONE 2026-05-17 (Stage 44).
   SysV ABI's xmm0..xmm7 covers the first 8 float params; the 9th
   now correctly passes on the stack for both internal CALL and
   FFI_CALL. f32 (4-byte) and f64 (8-byte) payloads both wired.
   Mixed int+float overflow rejects cleanly before rsp mutation.
   Int overflow (>6 ints) deferred — same infrastructure shape if
   a future stage needs it.

## Tier 2 — high value (do after Tier 1)

6. **Tensor codegen** with explicit memory-space movement
   (HBM/SMEM/REG/TMEM). Phase-0 PTX lowering currently supports 1D
   HBM `tile<f32, ...>` / `tile<i32, ...>` kernels plus a small scalar-op
   subset. Broader tensor/tile codegen, SMEM/REG tiles, `bf16`, matmul,
   and performance-oriented GPU lowering remain future work. **2-3 months.**

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
    reasoning. This is a strategic target against tensor-only AI stacks
    like Mojo/JAX/Triton.
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
verifier-gated reflective-cell scaffold with future AST reflection target,
(c) source-level reverse-mode AD,
(d) memory-tier types, and (e) future provenance-typed
neuro-symbolic primitives appears to be an under-served niche. A strong
AGI-axis target is achievable without requiring Helix to lead on GEMM
throughput first.

## Sequencing note

Tier 1 + Tier 2 #6 together are roughly 4-5 months of work and
unblock real model training. Tier 3 #10 (neuro-symbolic primitives)
should start scoping in parallel since it is a strategic differentiator and
the work is mostly orthogonal to GPU codegen.

## Post-Stage-31 Sequence

Stage 31 shifted the practical bottleneck from "can we test enough?" to "can we
test fast enough while preserving trust?" The next stages should reflect that.

### Stage 32 - Verification Speed Infrastructure

Purpose: make every later stage faster without weakening any gate.

Beginner meaning: the test system should tell us exactly what is slow, retry
only the part that flaked, and distribute work evenly across the machine.

Important features:
- Slow-test telemetry by test node, not only by shard.
- Machine-readable timing summaries. Stage 32 starts with
  `.stage31-logs/pytest-shard-timings.json`.
- Duration-weighted shard assignment. Stage 32 added optional per-test timing
  weights with stable-hash fallback.
- Failed-shard retry evidence.
- Changed-file to focused-test mapping. Stage 32 adds
  `scripts/stage32_select_tests.py` for conservative first-pass pytest
  selection before the full gate.

Importance: very high. This does not make Helix more powerful directly, but it
removes waiting from every future compiler and AI feature.

Relative work: medium to large.

### Stage 33 - Self-Host Parity And Python Removal Path

Purpose: return to the central bootstrap goal: Helix should compile Helix.

Beginner meaning: keep moving remaining Python-only compiler abilities into the
Helix compiler until the Python compiler can become only a historical
reference.

Important features:
- Port remaining Python-only validation/frontend passes.
- Keep bootstrap compiler behavior byte-stable and deterministic.
- Strengthen self-host cascade tests.
- Expand binary-level comparison gates.

Importance: highest. This is the independence milestone.

Relative work: large to very large.

### Stage 34 - Proof And Refinement Expansion

Purpose: make Helix better at proving uncertainty-reducing claims.

Beginner meaning: Helix should understand more safety rules at compile time and
produce trustworthy proof artifacts about them.

Important features:
- More refinement predicate shapes.
- Better proof artifact workflows.
- SMT-backed implication checks when ready.
- Clean integration with proof gates.

Importance: very high. This is central to Helix being an uncertainty-reducing
language, not only a fast language.

Relative work: large.

### Stage 35 - AI/ML Capability Push

Status: CLOSED 2026-05-16 at restart 65 (3/3 clean gates; see Stage 35
progress ledger Increment 82 in docs/stage35-progress-2026-05-15.md for
the closure narrative).

Purpose: unlock more real model-training code in Helix.

Beginner meaning: make Helix better at gradients, model structures, tensors,
and eventually GPU execution.

Important features:
- Multi-output reverse-mode AD.
- Pytrees for nested model parameters.
- Tile/tensor lowering.
- PTX/GPU path and FFI support.
- Autotune for generated kernels.

Importance: highest for practical AI usefulness.

Relative work: very large.

### Stage 36 - Strategic AGI Features

Status: CLOSED 2026-05-16 at Inc 16 (3/3 clean gates; see Stage 36
progress ledger Increment 16 in docs/stage36-progress-2026-05-16.md
for the closure narrative). First deliverable was the Tier 3 #10
provenance-typed neuro-symbolic primitives — shipped end-to-end with
4 dogfood programs, 5 audit cycles (22/23 actionable findings closed),
2 new IR opcodes (ARENA_PUSH_PAIR, ARENA_PUSH_TRIPLE), 1 new stdlib
file (provenance.hx), and self-host gate green throughout. The
remaining Stage 36 feature families (trace-based introspection,
verifier-gated self-modification, memory/knowledge types) carry
forward as Inc 17+ work.

Purpose: build the features that make Helix meaningfully different from normal
ML languages.

Beginner meaning: give Helix native tools for evidence, provenance,
self-inspection, and safe self-improvement.

Important features:
- Provenance-typed neuro-symbolic primitives.
- Trace-based introspection.
- Verifier-gated self-modification with stronger proof objects.
- Memory/knowledge types as the language grows toward the broader vision.

Importance: existential to the long-term Helix mission.

Relative work: very large to hardest.
