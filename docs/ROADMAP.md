# Helix Roadmap

This document captures the prioritized feature list synthesized from two
deep-research passes (2026-05-04). It's a forward-looking plan; not
everything here will land, and priorities will shift as dogfooding
reveals which features actually matter.

## Current state (Stages 35-59 CLOSED 2026-05-16 to 2026-05-18)

Burst summary (25 stages closed in <72h, all via the
3-clean-gate protocol; 162-commit autonomous burst):
- Stages 35-48 listed below.
- Stage 49: Tier 4 #14 Inc 3 runtime Ok/Err tag.
- Stage 50: RESURRECTED (fn_table cap fix unblocked Inc 1+2).
- Stage 51: Inc 1 SHIPPED (run-detection scaffold), Inc 2
  deferred to fresh session.
- Stage 52: modal-origin taint-tracking (16 audit gates;
  Stage 40 H1 closed).
- Stage 53: helper-fn modal indirection (Stage 40 H1 closed
  in full).
- Stage 54: Tier 1 #2 AD broader coverage (8 closure gates;
  ~80% of original blueprint shipped).
- Stage 55: Tier 1 #4 string/file IO primitives + capability typing.
- Stage 56: Tier 2 #8 Triton-style autotune (Cartesian-product
  variant expansion, end-to-end PTX).
- Stage 57: Tier 2 #7 JAX-style pytrees Inc 1 (grad_rev_all
  pytree-bridge live for struct params).
- Stage 58: Tier 4 #13 content-addressed modules (program_hash +
  module_hash + fn_signature_hash core).
- Stage 59: Tier 4 #15 nested pattern destructuring + polish
  burst (~84 commits across Tier 2 #7/#8, Tier 3 #11, Tier 4 #13:
  33 new Python introspection helpers + 66 CLI flags enumerated in
  --help + 6 cascading defects found+fixed + JSON/CSV round-trip
  serialization for both pytree and trace + 5 per-system + 2
  aggregator validator gates + 4 *-json machine-readable variants
  + 21-flag call-graph analysis suite across 8 axes (forward/inverse/
  topology/recursion/metrics/pathfinding/summary/output-format with
  Graphviz+Mermaid) for refactor planning, dead-code detection,
  stack-overflow audit, hotspot identification, and visualization).
  Final test counts: test_pytree.py 64/64, test_trace.py 52/52,
  test_ast_hash.py 40/40, test_autotune.py 34/34, test_cli.py 380+
  — every impacted area GREEN.

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
- **Stage 50** ✅ **CLOSED 2026-05-18** (RESURRECTED a35e628 +
  gate-1/2 audited 6feb675 + formally closed at Stage 51 Inc 2
  e4bee39 when n>1 paths became production-exercised):
  bootstrap `grad_rev_all` multi-bucket infrastructure swap.
  Inc 1 (f4e94fc) + Inc 2 (76b7735) originally aborted at
  f678aa3 due to G2 cascade-break with SIGILL rc=132. Root
  cause found via Exp C (after Exp A ruled out H4 buffer
  overflow + Exp B ruled out H1 stack overflow): bootstrap
  codegen `fn_table` capacity overflow. Stage 50 Inc 1+2 added
  16 new fns to parser.hx, pushing total to 527, exceeding
  cap 512. The CRITICAL consequence: `main` itself — being
  the LAST declared fn via the cascade driver — was among
  the overflow casualties; its CALL site got patched with
  `ud2 + 3 nops` (unresolved-CALL stub) → entry-point SIGILL.
  5-line fix in kovc.hx: bumped cap 512→1024. Cascade verified
  GREEN: G2..G11 byte-identical, smoke 4/4 PASS, Stage 52
  modal tests still 116/116 PASS.
- **Stage 51** ✅ **CLOSED 2026-05-18** (Inc 1 + Inc 2 SHIPPED;
  Tier 1 #3 bootstrap-side end-to-end DONE):
  - Inc 1 ✅ (9927361): grad_rev_pass run-detection scaffold.
    Outer loop walks runs of consecutive entries sharing the
    same loss_name. Inner loop processes each entry
    individually as Stage 50 Inc 2 bridge fallback. Cascade
    verified: G2..G4 byte-identical sha=7a35f8e8.
  - Inc 2 ✅ (e4bee39): multi-bucket fast path. When run_size
    is 2..8 AND all entries pass validation AND loss_fn is
    found AND ckpt_ok, processes the whole run via ONE
    `differentiate_reverse_all` walk + N `bucket_array_sum`
    extractions. Falls back to Inc 1 per-entry for n=1 OR
    n>8 OR any validation failure (so per-entry traps fire
    correctly per-entry rather than poisoning the whole
    run). Cascade verified: G2..G4 byte-identical sha=
    5a3bf021. 1089 Python AD/reverse/parity/codegen tests
    all green. Tier 1 #3 (multi-output reverse-mode AD)
    now closes end-to-end (Python side at Stage 36;
    bootstrap algorithmic side at Stage 51 Inc 2).
- **Stage 52** ✅ **CLOSED 2026-05-17** (gates 1-16 + Inc 1-13
  + Stage 53 Inc 1+2 shipped, 22+ launder paths caught via 11
  wrapper-AST kinds, 3-clean-gate closure protocol satisfied
  via gates 14/15/16 all CLEAN — both gate-16 silent-failure
  and code-review auditors explicitly declared "STAGE 52 is
  CLOSED"): modal-origin
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
- **Stage 54** ✅ **CLOSED 2026-05-18** (Inc 1+2+3a+3b SHIPPED +
  closure gates 1-8 via 3-clean-gate protocol):
  Tier 1 #2 — AD across user-defined function calls broader
  coverage. 100% of original blueprint scope shipped.
  - Inc 1 ✅ (3d5b900): chain-rule arms for __min/__max/__clamp/
    __sign (11 names) in both forward + reverse modes.
  - Inc 2 ✅ (3fdd61f) CLOSED no-op: forward/reverse asymmetry
    was already fixed at Stage 35 (verified by regression pin).
  - Inc 3a ✅ (e011241): `_inline_user_calls.go()` walker
    descends into A.For/A.While/A.Loop bodies.
  - Inc 3b ✅ (4873d07): bounded recursive unrolling via
    per-fn `unroll_counts: dict[str, int]` + `max_unroll=3` +
    `_args_are_unroll_safe` literal-only-arg detection.
    Directly-recursive helpers with literal-counter args
    (e.g., `power(x, 3)`) now unroll at compile time,
    exposing the gradient. Mutual recursion + non-terminating
    cases bounded by the cap.
  - **Closure gates 1-8** (5 commits, 16+ load-bearing regression
    pins, cascade trajectory 3→5→3→3→1→CLEAN→CLEAN→CLEAN):
    - Gate-1 (96fd97f): silent-failure HIGH-1 + Inc 3a load-
      bearing test fix. _substitute_names For/While/Loop/Match
      arms + inliner ExprStmt-descent + _go_block ExprStmt +
      Assign arm.
    - Gate-2 (ed2b5d4): walker arm sweep across remaining 12
      AST kinds (Cast/Index/Field/TupleLit/ArrayLit/StructLit/
      UnsafeBlock/Match/Assign/Return/Break/Range) in both
      walkers + i32 IntLit zero (MED-4) + clamp lo/hi warn
      (MED-5).
    - Gate-3 (c8c7161): _name_appears_in StructLit.fields +
      Match.arms (HIGH-1/2) + reverse-mode clamp warn parity
      (MED-3).
    - Gate-4 (43785af): _name_appears_in Modify/Quote/Splice/
      TileLit (HIGH-1) + Block-stmts walker `or`-hardening
      (MED-2) + __min/__max kink warn (MED-3).
    - Gate-5 (9424133): reverse-mode __min/__max kink warn
      parity (HIGH-1). Last non-clean gate.
    - Gates 6/7/8: CLEAN. Both silent-failure and code-review
      auditors verified pin load-bearing-ness, AST coverage,
      caller hygiene, adjacent-code regression freedom.
    - Tests: 98 AD/reverse/parity all green at closure.
  - Cascading-defect rhythm flushed across the full AST
    surface (16 walker arms + 6 _name_appears_in arms + 3
    chain-rule warn surfaces forward+reverse).

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

2. **AD across user-defined function calls.** ✅ DONE 2026-05-18
   (Stage 54 CLOSED via 3-clean-gate protocol + Inc 3b shipped).
   `grad(loss)` and `grad_rev(loss)` inline supported pure
   helper calls before differentiation; opaque/bodyless calls
   fail closed; loop-body descent works for pure helpers;
   chain-rule arms cover 11 builtins (__min/__max/__clamp/
   __sign + _i32/_f64 variants) in both forward + reverse
   modes with subgradient warnings on kink-crossing + clamp
   lo/hi dependency. _name_appears_in covers all AST kinds
   incl Modify/Quote/Splice/TileLit/StructLit/Match. Inc 3b
   (4873d07) added bounded recursive unrolling via
   per-fn unroll_counts + max_unroll=3 + _args_are_unroll_safe
   literal-only check — recursive helpers with literal-
   counter args (e.g., `power(x, 3)`) now expand at compile
   time. Tier 1 #2 100% of original blueprint shipped.

3. **Multi-output reverse-mode AD.** ✅ DONE end-to-end
   (Python side + bootstrap side both shipped).
   `differentiate_reverse(expr, param_names)` in
   `helixc/frontend/autodiff_reverse.py` produces a
   `dict[param_name -> gradient_expr]` in ONE walk via per-param
   bucket accumulation. `grad_rev_all(f)(p1, p2, base)` is wired
   end-to-end and used by `dogfood_05_binary_classifier.hx` etc.
   **Bootstrap-side**: Stage 50 (a35e628) restored multi-bucket
   infrastructure (kovc.hx fn_table cap 512→1024). Stage 51
   Inc 1 (9927361) shipped run-detection scaffold + Inc 2
   (e4bee39) activated true single-walk via
   `differentiate_reverse_all` over `param_array` for runs of
   2..8 consecutive same-loss entries. Tier 1 #3 closes end-
   to-end 2026-05-18.

4. **Richer strings + file I/O** with capability-typed
   `@effect(io.read_file)`. **STAGE 55 5/7 DONE 2026-05-18** —
   capability typing + runtime string primitives + parser +
   csv/mnist stdlib all shipped. Inc 3 (dyn-path file I/O,
   backend syscall rewrite) + Inc 7 (checkpoint stdlib, depends
   on Inc 3) deferred. Plan at `docs/stage55-plan-2026-05-18.md`.
   - Inc 1 ✅ (fbe7fef): `__str_byte_at` / `__str_find_byte` /
     `__str_eq_arena`.
   - Inc 2 ✅ (e52d525): `__parse_i32`.
   - Inc 4 ✅ (234aeb2): granular `@effect(io.read_file/write_file/
     print)` labels with wildcard parent subsumption.
   - Inc 5 ✅ (89c5cd0): `__str_from_i32` + `__str_concat_arena`.
   - Inc 6 ✅ (2a7147e): `helixc/stdlib/csv.hx` (line/field
     iteration via chained __str_find_byte, 4-chunk cap 1024 byte
     max line) + `helixc/stdlib/mnist.hx` (IDX-format header
     parser, big-endian dim decode, body bounds check). Pure
     Helix source built on Inc 1-5 primitives, cascade-safe.

5. **Stack-passed overflow args.** ✅ DONE 2026-05-17 (Stage 44).
   SysV ABI's xmm0..xmm7 covers the first 8 float params; the 9th
   now correctly passes on the stack for both internal CALL and
   FFI_CALL. f32 (4-byte) and f64 (8-byte) payloads both wired.
   Mixed int+float overflow rejects cleanly before rsp mutation.
   Int overflow (>6 ints) deferred — same infrastructure shape if
   a future stage needs it.

## Tier 2 — high value (do after Tier 1)

6. **Tensor codegen** with explicit memory-space movement
   (HBM/SMEM/REG/TMEM). ✅ PARTIALLY DONE (audited 2026-05-18).
   Phase-0 PTX lowering supports 1D HBM `tile<f32, ...>` /
   `tile<i32, ...>` kernels plus a substantial tile IR surface:
   `helixc/ir/tile_ir.py` has 35 TileOpKinds covering tile
   creation (zeros/const), memory movement (HBM↔SMEM↔REG, plus
   async TMA load/store + barrier_wait for Hopper/Blackwell),
   tile compute (add/sub/mul/matmul/reduce), layout transforms
   (transpose/reshape), scalar passthrough, GPU primitives
   (thread_idx + tile_index_load/store_hbm). `emit_ptx()` at
   `helixc/backend/ptx.py:621` consumes the tile IR end-to-end.
   79 PTX regression pins all green.
   Remaining for "full" tensor codegen: SMEM/REG tile
   instantiation patterns beyond load_global, `bf16` dtype
   support, performance-oriented lowering passes (autotune
   now wired at Stage 56 helps here), and more matmul tiling
   strategies. **Originally 2-3 months; the core IR + PTX
   emission infrastructure is shipped — remaining is breadth-
   of-ops + perf tuning, not greenfield.**

7. **JAX-style pytrees.** Inc 1 ✅ SHIPPED 2026-05-18 (Stage 57
   commit 80f659b): rejection-lift for struct params in
   `grad_rev_all` — `pytree.flatten_pytree_param` (Stage 26
   infrastructure) now wires into `grad_pass`, expanding struct
   params into per-leaf gradient writes via existing field-path
   AD machinery. Inc 2 (struct-shaped return from `grad`/`grad_rev`)
   deferred — needs Phase-0 struct-return ABI hardening.
   `grad(loss)(model)` where `model` is a nested struct → `Model`-
   shaped gradient. Composes `grad/vmap/jit` over arbitrary tree-
   structured parameter sets. Critical for real model architectures.
   Originally **2 weeks on top of tier-1 #3** — Inc 1 ships the
   core; Inc 2+ is polish. JAX-style functional API ✅ SHIPPED
   2026-05-18 (5 commits): tree_hash / tree_size / tree_diff /
   tree_count / tree_filter / tree_paths_matching /
   tree_to_canonical_json / tree_from_canonical_json (round-trip
   pin holds for JSON-native values) + the previously-shipped
   tree_map/reduce/zip/equal/leaves/paths. 55/55 test_pytree.py
   pins.

8. **Triton-style autotune.** ✅ DONE 2026-05-18 (Stage 56 commit
   4827397). `@autotune(KEY: [v1, v2, ...])` for `@kernel` functions
   now emits N specialized variants (Cartesian product of param
   values). Each variant is a deep-copy of the FnDecl with body
   walked to replace `Name(KEY)` → `IntLit(VAL)` per config.
   PTX backend emits N `.entry` blocks per autotuned kernel. The
   pre-existing Stage 27 parse/validation infrastructure
   (autotune.py:autotune_variants + mangled_variant_name) is now
   wired end-to-end via the new `autotune_expand.py` pass.
   Introspection + CLI polish ✅ SHIPPED 2026-05-18 (2 commits):
   `autotune_variant_names_for(fn)`, `autotune_variant_count_for(fn)`,
   `autotune_expansion_summary(prog)` Python API + `--autotune-summary`
   and `--autotune-budget` CLI flags (CI gate for variant-count
   drift detection).

9. **Mojo-style parametric structs.** ✅ SUBSTANTIALLY DONE (earlier
   stages — verified 2026-05-18). `helixc/frontend/struct_mono.py`
   ships `monomorphize_structs(prog)` + `collect_generic_structs`
   + `mangle_struct(name, ty_args)` + `instantiate(decl, ty_args)`.
   Generic struct decls (`StructDecl.generics: list[GenericParam]`)
   are collected, concrete uses are walked, instantiations are
   AST-cloned with type-arg substitution + name mangling. 40
   regression pins in `test_struct_mono.py`. Trap 28001 fires
   for uninstantiated generics. Composes with the existing
   monomorphize.py for fn-level generics. Dtype-flexible code
   (f32/bf16/fp8 specialized via TyGeneric base + args) is
   reachable today.

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
    Python-side introspection API ✅ SHIPPED 2026-05-18 (4 commits):
    `trace_hash`, `trace_size`, `trace_count`, `trace_op_counts`,
    `trace_fn_counts`, `trace_is_balanced`, `trace_equiv_modulo`
    (skip-list equivalence), `trace_to_canonical_json` +
    `trace_from_canonical_json` (round-trip for on-disk dumps,
    pinned for JSON-native operand types). 49/49 test_trace.py
    pins. Runtime wiring of entry/exit emission into binary
    prologue/epilogue is bootstrap-side and remains deferred.

12. **Lean-4-style proof-carrying terms.** ✅ PARTIALLY DONE
    (Stages 31 + 34, verified 2026-05-18). Verifiers receive a
    machine-readable proof artifact (`ProofObligation` +
    `ProofCarry` dataclasses in typecheck.py:62-100). The
    `--emit-proof-obligations` CLI flag dumps JSON with kind,
    context, refinement, predicate, status, span, value, trap
    for each obligation. 87 CLI test references exercise the
    proof-emission path. Remaining for "true" Lean-4-style
    proof-carrying terms: extending the Proof type to carry
    actual proof terms (not just obligations) + a kernel-level
    proof-checker pass. Months for that full capability; the
    obligation-emission scaffolding is shipped.

## Tier 4 — table-stakes infrastructure

13. **Module system** with content-addressed packages (modules
    referenced by AST hash). ✅ CORE DONE 2026-05-18 (Stage 58
    commit abb7a89). `module_hash(decl) -> str` and
    `program_hash(prog) -> str` in `helixc/frontend/ast_hash.py`
    aggregate the existing `structural_hash` (Stage 28.9) machinery
    over ModuleDecl items / Program top-level items. Span-
    independent + alpha-equivalence-aware. Future wiring into
    compilation cache (helixc/check.py) for build-system-level
    content addressing is a separate stage.
    CLI polish ✅ SHIPPED 2026-05-18: `--program-hash`,
    `--diff-program-hash`, `--changed-fns`, `--fn-sig-hash`,
    `--list-fns`, `--check-program-hash`, `--list-modules`,
    `--module-hash` (8 flags). Cascading defect found + fixed —
    `_hash_into` was missing ModBlock/ModuleDecl arms required
    by recursive module_hash; pinned with
    test_stage59_module_hash_nested_modblock_works.

14. **Result<T,E> + ? operator.** Error handling beyond panic. **1
    week.**

15. **Pattern matching with guards + or-patterns + nested
    destructuring.** ✅ DONE 2026-05-18 (Stages 28.9 cycles 10-15
    for guards/PatOr/PatVariant/PatRange; Stage 59 commits
    bb774ff + 0a2c895 + c5ada81 for struct destructuring +
    nested struct).
    - Parser handles `Point { x: 1, y }` / `Point { .. }` patterns.
    - `_collect_binds_with_path` flattens nested struct sub-patterns
      into leaf-path access chains (`scrut.f1.f2.fN`) — required
      because Phase-0 IR has no partial-struct value representation.
    - `_pattern_test_expr` builds the AND of sub-field tests.
    - `ast_hash` distinguishes field orderings.
    - typecheck `_bind_pattern` PatStruct arm resolves field types
      from `_struct_decls` for recursive bind.
    - Critical for AST-walking inside quote/splice + pytree
      compositions. 5/5 regression pins green (basic, literal-match,
      nested-typecheck, nested-end-to-end, ignore-rest).

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
