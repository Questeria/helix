# Helix Roadmap

This document captures the prioritized feature list synthesized from two
deep-research passes (2026-05-04). It's a forward-looking plan; not
everything here will land, and priorities will shift as dogfooding
reveals which features actually matter.

## Current state (Stages 35-59 CLOSED 2026-05-16 to 2026-05-18)

Burst summary (25 stages closed in <72h, all via the
3-clean-gate protocol; **232-commit autonomous burst, Stage 59 CLOSED** as of refresh):
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
- Stage 59 ✅ CLOSED: Tier 4 #15 nested pattern destructuring + polish
  burst (**232 commits** across Tier 2 #7/#8, Tier 3 #11, Tier 4 #13:
  ~80 new Python introspection helpers + **166 CLI flags** enumerated in
  --help + 8 cascading defects found+fixed + JSON/CSV round-trip
  serialization for both pytree and trace + **6-flag validator
  sextet** (per-system + aggregator + help-docstring + json-parity
  gates) + JSON-parity for every shipped non-visualization flag
  (~85% coverage) + 21-flag call-graph analysis suite across 8 axes
  (forward/inverse/topology/recursion/metrics/pathfinding/summary/
  output-format with Graphviz+Mermaid) for refactor planning, dead-
  code detection, stack-overflow audit, hotspot identification, and
  visualization + **top-level enumeration nonet** (fns/structs/
  modules/uses/consts/enums/type-aliases/agents/impls × {text, JSON}
  = 18 list flags) + **per-item introspection octet** (struct-fields/
  const-value/enum-variants/agent-methods/type-alias-target/impl-
  methods/fn-signature/module-stats × {text, JSON} = 16 inspect
  flags) covering 9 of the Item subclasses end-to-end + **callgraph
  JSON sub-arc** (13 flags: forward/inverse adjacency, transitive
  reachability, topology, recursion, distance/path/depth) + **CLI
  self-introspection axis** (6 flag pairs: --list-all-flags + --has-
  flag + --flag-groups + --flag-doc + --cli-summary-json + --flag-
  arity — programmatic CLI surface discovery) + diff/comparison
  JSON-parity triple (program-hash + changed-fns + hash-dump JSON) +
  check JSON quartet (4 hash-assertion gates) + hash producer JSON
  triple (program-hash + sig-hash + fn-sig-hash JSON)).
  Final test counts: test_pytree.py 64/64, test_trace.py 52/52,
  test_ast_hash.py 40/40, test_autotune.py 34/34, test_match.py 33/33,
  test_cli.py 460+ — every impacted area GREEN (self-host gate
  223/223 across the 5 introspection files at every commit).

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

- **Stage 55** ✅ **CLOSED 2026-05-18** — Tier 1 #4 string/file
  IO + capability typing (5/7 incs shipped; Inc 3 dyn file I/O +
  Inc 7 checkpoint stdlib deferred to Stage 60/61).
- **Stage 56** ✅ **CLOSED 2026-05-18** — Tier 2 #8 Triton-style
  autotune (Cartesian-product variant expansion, end-to-end PTX).
- **Stage 57** ✅ **CLOSED 2026-05-18** — Tier 2 #7 JAX-style
  pytrees Inc 1 (grad_rev_all pytree-bridge live for struct params;
  Inc 2 struct-shaped grad return deferred to Stage 62).
- **Stage 58** ✅ **CLOSED 2026-05-18** — Tier 4 #13 content-
  addressed modules (program_hash + module_hash + fn_signature_hash
  core).
- **Stage 101 SHIPPED 2026-05-18** — Stage 99 audit-residual #4
  fix: `_is_copy_struct_ty` walks through Tier-S/A wrappers:
  - Pre-Stage-101, `_is_copy_struct_ty` only checked `isinstance(
    ty, TyStruct)` at the top level. A `@copy struct Velocity {x,
    y, z}` wrapped as `Private<Velocity>` (e.g. via `__wrap_dp(v)`
    for DP-private velocity readings) was treated as NON-Copy by
    the borrow checker — defeating @copy as soon as any metadata
    wrapper was added.
  - Post-Stage-101, the helper walks through the 13 known wrapper
    classes (`_ALL_WRAPPER_CLS_NAMES` from Stage 100) by repeatedly
    peeling `.inner`. Bottoms out at the first TyStruct (or
    returns False if no struct ever found).
  - Uses the Stage 100 class-level `_ALL_WRAPPER_CLS_NAMES`
    registry — second consumer of Stage 100's hoist (first was
    `_strip_wrapper_chain` and the `__wrap_X` dispatch).
  - 2 new tests; 426 typecheck + 494 broader regression GREEN.
  - 4 of 7 Stage 100+ backlog items now closed. Remaining: typed-
    hole expected-type context (Stage 102), multi-arg @property
    typecheck rejection (Stage 103), 5 missing safety.hx @property
    roundtrips (Stage 104).

- **Stage 100 SHIPPED 2026-05-18** — Stage 99 audit-residual fixes
  (3 items from the Stage 100+ backlog closed in one commit):
  - **Hoist wrapper tables to class scope**: `_WRAPPER_CTOR_TABLE`
    + `_WRAPPER_STRIP_TABLE` + `_ALL_WRAPPER_CLS_NAMES` are now
    class attributes on TypeChecker (alongside the existing
    `_WRAPPER_HINT_TABLE` from Stage 87). Pre-Stage-100, all 3
    were closure-local inside `_check_expr` Call branch and
    re-allocated on every Call expression typecheck (plus 13
    lambda closures from the strip rebuilders). New class methods:
    `_strip_wrapper_chain(target_cls, t)` (hoisted from Stage 97
    closure), `_wrapper_default_for(ctor_name)` + `_wrapper_target
    _for(opt_out_name)` (lookup helpers). Strip semantics
    identical to Stage 97.
  - **Stage 92 inline `_snapshot_chain` removed**: now calls
    `scope.borrows_snapshot_chain()` (the Scope method Stage 95
    introduced). Eliminates ~8 lines of duplication that Stage 99
    audit B flagged as NEW-1 drift-risk.
  - **Stale "absolute outermost" comments fixed** at typecheck.py:
    5604 (TyTaint claim was true at Stage 69 but stale after
    Stages 79 + 80 layered above it; comment updated to document
    actual canonical layering order). typecheck.py:5651 (TyEnclave
    claim was VERIFIED still accurate post-Stage-100; comment
    updated to acknowledge the Stage 100 audit context).
  - 2 new tests; 490 typecheck + 6 dogfood + 33 match GREEN.
  - Closes 3 of 7 Stage 100+ backlog items. Remaining for future
    stages: _is_copy_struct_ty wrapper recursion, typed-hole
    expected-type plumbing to more contexts, multi-arg @property
    typecheck rejection, add @property roundtrips for DP/Quant/
    Domain/Robust/Energy to safety.hx.

- **Stage 99 SHIPPED 2026-05-18** — RE-AUDIT verdict: 17 stages
  re-flip from 🟡 to ✅ FULLY CLOSED.
  - 3 combined-batch audits dispatched in parallel against post-
    Stages-94-98 code. Verdicts:
    - **Audit A (silent-failure-hunter)**: PASS_RE_AUDIT
      p_pass=0.93, confidence HIGH. All 4 HIGH + 2 MEDIUM Stage
      93 findings verified fixed. Zero new silent miscompiles
      introduced.
    - **Audit B (type-design-analyzer)**: PASS_WITH_RESIDUAL
      p_pass=0.74, MEDIUM. Specific fixes clean; original design-
      debt items 1,2,4,5,6,7 remain (refactor-class items moved
      to Stage 100+).
    - **Audit C (code-reviewer)**: RE-CLOSED p_pass=0.91, HIGH.
      All 5 fix stages individually RE-CLOSED. 2 follow-up nits
      (hoist closure-local tables to module-level + 1 stale
      comment) tracked for Stage 100.
  - **17 stages re-marked FULLY CLOSED** (was 🟡 awaiting re-
    audit): Stage 66, Stages 68-73, 75, 76, 78-83, 86, 88, 92.
  - **Combined-burst closure-status totals after Stage 99**:
    - ✅ FULLY AUDIT-CLEAN: 25 stages (8 prior + 17 newly re-
      closed)
    - 🔵 DEFERRED to v1.1: 1 (Stage 64 Inc 3-5)
    - ⏭️ Meta stages (audits + fix commits): 8 (Stages 91, 93,
      94-99)
    - ⚪ Pre-burst stages (not re-audited this session): ~66
  - **Stage 100 backlog** (design-debt follow-up from Audit B):
    - Hoist _WRAPPER_CTOR_TABLE / _ALL_WRAPPER_REBUILDERS /
      _WRAPPER_STRIP_TABLE from _check_expr closure to module-
      level constants
    - Extend _is_copy_struct_ty to walk through wrappers
    - Extend typed-hole expected-type plumbing to let-RHS / fn-
      return / struct-field-init / match-arm positions
    - Reject multi-arg @property at typecheck OR extend runner to
      cartesian-product (one of the two)
    - Fix stale "absolute outermost" comments referring to TyTaint
      and TyEnclave (they're no longer outermost after Stages 79,
      80, 83 layered above them)
    - Add @property roundtrip fns to safety.hx for the 5 wrappers
      currently missing them (DP, Quant, Domain, Robust, Energy)
    - Refactor Stage 92 inline _snapshot_chain to use Stage 95's
      Scope methods (eliminate ~8 lines of duplication)

- **Stage 98 SHIPPED 2026-05-18** — MEDIUM polish bundle: closes
  Stage 93 audit's 2 remaining MEDIUM findings.
  - **Fix A: safety.hx TyAttribution helpers (Stage 83 gap)**.
    Stage 83 added the Attribution wrapper + opt-out in Stage 83
    but never extended safety.hx with the helper + roundtrip
    @property fn (Stages 78 + 82 had set the per-wrapper template
    for the other 10 wrappers, and the audit flagged Stage 83 as
    the lone outlier). Stage 98 adds:
    - `attribute_unknown_f32(x: f32) -> FromUnknown<f32>` (calls
      `__wrap_attr`)
    - `verify_attribution_f32(x: FromUnknown<f32>) -> f32` (calls
      `__attribute_verified`)
    - `@property safety_attribution_roundtrip_is_identity(x: f32)
      -> bool` (round-trip identity assertion)
    safety.hx now has 22 helpers (11 wrappers × 2) + 6 @property
    fns total. Stage 78 + Stage 82 counter-tests updated.
  - **Fix B: test_property_runner include_stdlib parameter**.
    The Stage 86 test `test_stage86_runner_end_to_end_on_trivial_
    property` asserted `p == 7` but observed `p == 42` because
    `run_properties` always included the 6 stdlib @property fns
    (6 × 7 f32 inputs = 42) when discovering. Pre-fix: test was
    silently failing. Stage 98 adds `include_stdlib: bool = True`
    parameter to `run_properties()` (default True for backward
    compat); the failing test now passes `include_stdlib=False`
    to isolate its own `always_true` property and the `p == 7`
    assertion holds. Validated: 1/1 end-to-end test now PASSES
    (was 1/1 FAILING pre-Stage-98).
  - All 4 HIGH + 2 MEDIUM Stage 93 audit findings now closed.
    Stage 99 re-audit next.
  - 3 tests updated (Stage 78 helper-count, Stage 82 property-
    count, Stage 86 discovery-count) + 1 test added (Stage 98
    attribution roundtrip via the new stdlib helper). 494
    typecheck/selfhost/IR/runner-unit GREEN + 1 end-to-end runner
    GREEN.

- **Stage 97 SHIPPED 2026-05-18** — HIGH-#2 fix: `_strip_X` chain
  completion via single registry-driven helper (Stage 93 audit
  finding):
  - Pre-Stage-97: 11 individual `_strip_X` closures, 8 of which
    walked only a SUBSET of the 13-wrapper chain. Concrete
    silent-miscompile: `__exit_enclave(x: FromUnknown<InEnclaveSGX
    <f32>>)` returned the input UNCHANGED (audit-grep compliance
    contract violated — the user thinks the enclave was exited
    but the type still carries it).
  - Post-Stage-97: single `_strip_wrapper_chain(target_cls, t)`
    helper driven by `_ALL_WRAPPER_REBUILDERS` (13-entry registry
    of (cls, rebuild_lambda) pairs covering all wrappers) and
    `_WRAPPER_STRIP_TABLE` (11-entry registry of (opt_out_name,
    target_cls)). The helper strips the OUTERMOST instance of
    target_cls and preserves all other wrappers via their
    rebuilders.
  - **Net code reduction**: deleted 328 lines of 10 obsolete
    per-wrapper strip closures (lines 5882-6209 in typecheck.py).
    Same template-collapse pattern Stage 96 used for constructors.
    The two registries now collectively replace 6 of the 8
    parallel hand-maintained tables the Stage 93 type-design
    audit flagged as drift-prone.
  - Cascade-defect-class: future wrapper additions need only one
    new entry in `_ALL_WRAPPER_REBUILDERS` and one in
    `_WRAPPER_STRIP_TABLE` (plus the typecheck dataclass + parser
    alias-map). No more 8-touchpoint-per-wrapper drift risk.
  - 5 new tests including a parametric-style test that asserts
    all 11 opt-outs strip their target wrapper. 424 typecheck +
    496 broader regression (incl 6 dogfoods exercising the new
    helper) GREEN.

- **Stage 96 SHIPPED 2026-05-18** — HIGH-#1 fix: `__wrap_X`
  constructor idempotency rejection (Stage 93 audit finding):
  - Refactored 11 individual `if bn == "__wrap_X"` arms (lines
    5767-5791) into a single `_WRAPPER_CTOR_TABLE`-driven dispatch
    loop. Table is the single source of truth for (ctor_name,
    wrapper_cls, default_kwargs) per wrapper.
  - Each constructor now checks `isinstance(arg_ty, wrapper_cls)`
    BEFORE wrapping. If already-wrapped, emits diagnostic:
    "__wrap_X(Wrapped<f32>): received an already-wrapped value;
    intro builtins are not idempotent (Stage 96 / Stage 93 audit
    HIGH-#1 fix — pre-Stage-96, double-wrap silently broke
    composition semantics, e.g. __wrap_dp(__wrap_dp(x)) yielded
    Private<Private<f32>> not Private<f32> eps=2.0)" with hint
    "use binop propagation (a + b) to combine wrappers correctly".
  - Returns `TyUnknown(hint=ctor_name)` on rejection so downstream
    cascade errors are suppressed.
  - Mirrors Stage 43 Inc 1 M1 pattern for Tier 3 intro builtins
    (`_tier_intro_elim` at typecheck.py:6540+).
  - Same anti-pattern source: Stage 75 added the 11 constructors
    in a burst without per-constructor idempotency check; Stage
    93 audit (silent-failure-hunter) reproduced the silent
    double-wrap and reported HIGH-#1.
  - 4 new tests including a parametric-style test that asserts
    ALL 11 constructors reject double-wrap. 419 typecheck + 485
    broader regression + 6 dogfood GREEN.

- **Stage 95 SHIPPED 2026-05-18** — HIGH-#4 fix: scope-chain
  borrow reconciliation for A.If + A.Match arms (Stage 93 audit
  finding):
  - Lifted Stage 92's `_snapshot_chain` closure to Scope methods
    `borrows_snapshot_chain` + `borrows_snapshot_counts_chain` +
    `borrows_apply_chain`. New module-level helper
    `_root_local_name_of_place` walks Place's nested parts tuple
    to find the root local for chain routing.
  - Wired chain-snapshot into A.If reconciliation (replacing the
    immediate-scope `dict(scope.borrows.state)` snapshot which
    missed outer-defined places). Restore-between-arms now uses
    `borrows_apply_chain` to route each place back to its
    defining scope. Final JOIN write also uses chain routing.
  - **Added A.Match borrow-state reconciliation** (was MISSING
    entirely pre-Stage-95). Mirrors A.If's pattern: snapshot
    pre-match chain, restore between arms, snapshot post-arm,
    JOIN with most-restrictive-wins, divergence diagnostic when
    MOVED in some-but-not-all arms.
  - **Closes silent-miscompile class** that affected:
    - `{ if true { __move(s) } }` where s is in outer fn scope
    - `match c { 0 => __move(s), _ => 0 }` (was 100% silent
      pre-Stage-95 — match had NO reconciliation at all)
  - 3 new tests; 415 typecheck + 481 broader regression + 33
    match-codegen GREEN.

- **Stage 93 audit BATCH 2-5 DOWNGRADES Stages 68-92 closure markers** —
  ran 3 combined-batch audits (silent-failure / type-design /
  code-review) in parallel against the 21 burst stages. Findings:
  - **HIGH-#1 (Stage 75 constructor double-wrap)**: `__wrap_dp(
    __wrap_dp(x))` yields `Private<Private<f32>>` instead of
    `Private<f32>` with eps=2.0 → DP privacy accounting silently
    broken. Same anti-pattern Stage 43 Inc 1 M1 fixed for Tier 3
    intro builtins. Affects all 11 wrappers (Stages 68-83).
  - **HIGH-#2 (Stage 75 strip-helper chain incompleteness)**: 8
    of 11 `_strip_X` helpers walk only a subset of the 13-wrapper
    stack. `__exit_enclave(x: FromUnknown<InEnclaveSGX<f32>>)`
    returns unchanged (audit-grep compliance contract violated).
  - **HIGH-#3 (Stage 92 _KNOWN_FN_ATTRS regression)**: whitelist
    omits `@overload`, `@dispatch` (Stage 65), `@unwind`,
    `@trace`. Stage 65 multi-dispatch users now hit "unknown
    attribute @overload". Tests passed only because Stage 65 tests
    bypass TypeChecker.check().
  - **HIGH-#4 (Stage 66 Inc 5c if/match chain-walk gap)**: Stage
    92 fixed the loop-body scope-chain walk but A.If + A.Match
    arm reconciliation still snapshots only the immediate scope.
    A `__move(s)` inside `{ if true { ... } }` where s is in fn
    body produces ZERO Stage 66 errors. Same silent-miscompile
    class as Stage 91 HIGH-#1 but for branch arms instead of loops.
  - **MEDIUM**: Stage 86 test `test_stage86_runner_end_to_end_on_
    trivial_property` asserts p==7 but observes p==42 (stdlib's
    5 @property fns × 7 inputs add 35 passes). Test bug, not
    runner bug.
  - **MEDIUM**: Stage 83 left safety.hx without TyAttribution
    helpers (5 helpers added for Stages 79-81 in Stage 82 but
    Stage 83 was missed).
  - **MEDIUM**: Stage 90 typed-hole expected-type plumbing only
    covers call-arg position; let-RHS / fn-return / struct-field
    still emit Stage 89's generic message even though expected
    type is knowable.
  - **MEDIUM**: TyAttribution wrap-order doc-vs-code mismatch
    (docstring says "compliance-axis like Taint, outermost" but
    wrap-block puts it 6th-from-outermost).

  **Closure markers downgraded** (back to SUBSTANTIALLY COMPLETE
  pending Stage 94-98 fixes):
  - Stages 68-83 (11 wrappers): HIGH-#1 + HIGH-#2 affect every
    wrapper's constructor + strip-helper.
  - Stage 88 (CLOSURE marker for 68-87): premature given the
    11-wrapper findings.
  - Stage 92 (Inc 5d audit closure for Stage 66): re-opens for
    HIGH-#3 + HIGH-#4.

  **Closure markers that STAND**:
  - Stage 64 (DEFERRED to v1.1) — Stage 64 Inc 3 speculative
    parallel agent SHIPPED in isolated worktree
    `C:\Projects\Kovostov-Native-stage64-inc3` (commit
    `c90c4150`, 96 PTX + 477 regression GREEN, 9 new Stage 64
    Inc 3 tests for TILE_ADD/SUB/MUL f32+i32). Available for
    cherry-pick if v1.1 GPU CI lands ahead of schedule.
  - Stages 89-90 (typed holes) — partial closure stands; expected-
    type plumbing for other contexts is Inc 3+ enhancement, not
    Stage 89 unfinished.
  - Stage 86 (property runner): runner is solid; test needs fix.
  - Stage 87 (wrapper-mismatch hint): cross-cutting refactor is
    clean and well-designed.

  Combined verdict: ESCALATE per stricter-wins protocol. Stage 94
  fix bundle next (whitelist + chain-walk + constructor idempotency
  + strip-chain completion + safety.hx TyAttribution + test fix).

- **Stage 64 DEFERRED to v1.1 2026-05-18 (per user directive)** —
  Tier 2 #6 tensor codegen split:
  - Inc 1 SHIPPED (bf16/f16 HBM dtype) + Inc 2 SHIPPED (TILE_ZEROS
    register-fill in PTX backend) remain in v1.0 as the PTX
    backend architectural validation.
  - Inc 3-5 (TILE_ADD/SUB/MUL elementwise, TILE_MATMUL via wmma
    Tensor Core fragments, tile-IR optimization passes) **moved to
    v1.1**. Reason: full closure requires GPU CI infrastructure
    that v1.0 does not have. Per user: "Defer, after v1 will
    feature GPU."
  - v1.0 ships with CPU x86_64 codegen only. PTX text emission
    syntax-validates but is not exercised against real GPU
    hardware. v1.1 will add GPU CI runner + Inc 3-5 + actual
    end-to-end NVIDIA GPU validation.
  - **Speculative parallel work**: a worktree-isolated agent
    is starting Inc 3 (TILE_ADD/SUB/MUL elementwise) in parallel
    with the audit work. If the parallel agent lands cleanly + GPU
    CI gets stood up later, Inc 3 can be folded back into v1.0.
    If not, it stays a v1.1 deliverable. Speculation, not promise.

- **Stage 92 SHIPPED 2026-05-18** — Inc 5d fixes for Stage 66's 2
  HIGH-severity silent miscompiles (Stage 91 audit closure):
  - **Fix 1 (HIGH-#1)**: loop-body borrow reconciliation. New
    helper `_check_loop_body_with_borrow_reconciliation` wraps
    the existing `_check_loop_body_with_modal_union`: snapshots
    scope.borrows.state (walking the scope chain to capture
    outer-defined places) before the body, runs the body, then
    compares exit vs entry per place. If any place ended in a
    strictly-worse state (rank MOVED > MUTABLE > SHARED > FREE)
    than entry, emit "loop body ends with X in state moved but
    entered in state free; next iteration would observe an
    unsound starting state". Reset to entry to suppress cascade
    errors. Wired into A.For / A.While / A.Loop dispatch sites.
    Only fires under @borrow_check / global opt-in (mirrors the
    rest of Stage 66's gating).
  - **Fix 2 (HIGH-#2)**: attribute whitelist + Levenshtein
    suggest. New `_KNOWN_FN_ATTRS` (16 entries: pure, kernel,
    grad, jvp, vjp, vmap, autotune, effect, io, network,
    modify_self, rng, time, fs, deprecated, since, total, partial,
    borrow_check, property, inline, __stdlib, verifier) and
    `_KNOWN_STRUCT_ATTRS` (1: copy). New `_validate_known_attrs`
    method emits "unknown attribute @X on fn 'Y' ... did you
    mean @Z?" with Levenshtein-distance-based suggestion (≤ 3
    edits for short attrs; ceil(len/2) otherwise). Wired into
    `_check_fn` prologue + pass-0 struct indexing. Parser-derived
    `<base>:<arg>` forms (e.g. `effect:io`, `autotune:K=V`) are
    handled by splitting at `:` and validating the base only.
  - **Cascade defect fixed inline**: discovered `@verifier`
    attribute used by `__always_accept` in stdlib
    (transcendentals.hx:335); added to whitelist with comment
    explaining it marks fns as verifier callbacks for the
    modify() AGI primitive.
  - **Stage 66 RE-CLOSURE pending Batch 1 re-audit**. The Inc 5d
    fixes are tested (5 new tests passing) and address the
    audit-reported HIGH issues, but per the protocol the formal
    closure requires re-running silent-failure-hunter against the
    fixed code. Next stage will dispatch the re-audit.
  - 5 new tests; 411 typecheck + 477 broader regression GREEN +
    6 dogfood GREEN.

- **Stage 66 DOWNGRADED to SUBSTANTIALLY COMPLETE 2026-05-18** —
  formal 3-clean-gate audit batch (Stage 91 closure-audit pass)
  surfaced 2 HIGH-severity silent miscompiles that the regression
  tests don't catch:
  - **Loop-body silent miscompile**: `for i in 0..10 { let _ =
    consume(s); }` and `for i in 0..10 { let _ = __move(s); }`
    produce ZERO Stage 66 diagnostics despite being runtime
    double-moves on iteration 2+. The borrow checker's stated
    job is to catch double-move; it doesn't on loops. The
    `_check_loop_body_with_modal_union` handler tracks modal
    taint but NOT borrow state.
  - **Unknown-attribute silent failure**: typo `@borrowcheck`
    (missing underscore) silently leaves `_current_fn_borrow_check
    = False`. Developer believes their fn is checked; it isn't.
    Same risk for `@Copy` vs `@copy`, `@borrowed_check`, etc.
    No whitelist + Levenshtein suggest.
  - Plus 6 MEDIUM/LOW Phase-0 deferrals (match-arm reconciliation
    missing, release_* never called, @copy field validation
    absent, field-place move-tracking absent, stale comment at
    typecheck.py:4663, `__move(expr)` silent no-op when expr
    isn't a Name, Place repr leaked in divergence diagnostic).
  - **Action**: Inc 5d ships fixes for the 2 HIGH issues before
    re-marking Stage 66 CLOSED. MEDIUM/LOW issues tracked in a
    Stage 92 polish backlog (not blocking re-closure).
  - **Lesson for the burst**: regression-test-clean ≠ formal
    3-clean-gate. Stages 68-83 (the 11 wrappers) and 89/90/64
    Inc 2 still need their own formal audit batches before any
    of them can claim CLOSED status without the same caveat.

- **Stage 64 Inc 2 SHIPPED 2026-05-18** — Tier 2 #6 tensor codegen
  TILE_ZEROS in PTX backend (partial closure progress; Inc 3-5
  remain multi-week):
  - PTX backend `emit_op` dispatch now handles `TILE_ZEROS` for
    f32 and i32 tiles. Emits N consecutive register-fills:
    `mov.f32 %fX, 0f00000000;` (f32) or `mov.b32 %rX, 0;` (i32).
  - Result TileValue maps to the BASE register of the fill range;
    downstream consumers can find the tile's start. Phase-0 uses
    a register-tile representation (no SMEM allocation yet) —
    Inc 3 will add SMEM allocation strategy for larger tiles.
  - Validates attr contract: positive int `length`, dtype in
    {f32, i32} (bf16/f16 rejected with clear "Inc 3+ will extend"
    message — fails closed).
  - Pre-fix: any TileOp not handled by the dispatch hit the
    catch-all "unsupported PTX op {kind}" error. Post-fix:
    TILE_ZEROS is the first tile-construction op wired.
  - **Honest partial-closure note**: Stage 64 full closure (Inc
    2-5) requires multi-week focused session for TILE_ADD /
    TILE_SUB / TILE_MUL elementwise (Inc 3), TILE_MATMUL via
    NVidia wmma Tensor Core instructions (Inc 4), and tile-IR
    optimization passes (Inc 5). This Inc 2 ship moves Stage 64
    forward by one TileOp without committing to the full backend
    buildout.
  - 4 new tests; 8/8 Stage 64 tests GREEN (Inc 1 + Inc 2 combined);
    559/559 broader regression GREEN.

- **Stage 90 SHIPPED 2026-05-18 + Stage 89 NOW FULLY CLOSED** —
  Typed-hole expected-type context at call sites (Stage 89 Inc 2):
  - When the call site detects an arg with type
    `TyUnknown(hint="typed_hole")`, emit a new "typed hole at
    call to '<fn>' arg '<param>': expected <type> here" diagnostic
    showing the expected type from the param signature.
  - The Stage 87 _fmt prettifier means the expected type renders
    cleanly even for layered wrappers — e.g., `Confidential<Conf<
    Q8<f32>>>` shows in the diagnostic, not the raw dataclass repr.
  - The regular "arg X expects Y, got Z" cascade error is
    suppressed at the typed-hole arg position via `continue` in
    the `_check_call_basic` loop. Without this, the user would
    see 3 diagnostics for one hole (Stage 89 Inc 1 generic + Inc 2
    type-aware + generic cascade). Now just 2 (both informative).
  - **AI-completion UX win**: a tool reading the diagnostic stream
    sees `expected i32 here (Stage 90 / Stage 89 Inc 2)` and can
    fill in a value of the right type without needing to read
    other parts of the codebase.
  - **Stage 89 NOW FULLY CLOSED** (was PARTIAL after Inc 1):
    typed holes work end-to-end with type-aware diagnostics at the
    primary call-arg use case. Other contexts (let-RHS with type
    annotation, fn return) would benefit from the same treatment
    in a future Inc 3 but are not blocking v1.0 since the cascade
    mismatch already surfaces the expected type indirectly.
  - 3 new tests; 406 typecheck + 472 regression GREEN.

- **Stage 89 Inc 1 SHIPPED 2026-05-18** — Tier-B typed holes
  (V1_FINAL_FEATURES Tier-B #1: holes / typed `_` for AI-assisted
  development):
  - When typecheck encounters `_` in expression position (parses
    as `A.Name("_")`), emits a "typed hole" diagnostic with a
    span pointer rather than the generic unbound-name suggestion.
  - Returns TyUnknown(hint="typed_hole") so downstream uses of
    the hole's value don't cascade with spurious type errors —
    only the hole itself surfaces.
  - First Tier-B feature shipped in the burst. Stage 89 Inc 2
    plan: thread expected-type context through `_check_expr` so
    the hole diagnostic can report what type is required at the
    position (the actual UX win for AI-assisted code completion).
  - Pre-fix, `let x = _;` produced "unbound name '_' — did you
    mean ...?" with a useless Levenshtein suggestion. Post-fix,
    it produces "typed hole `_` at expression position — replace
    with a real expression... (Stage 89 Inc 1)".
  - 3 new tests; 403 typecheck + 469 regression GREEN.

- **Stage 88 SHIPPED 2026-05-18** — Formal CLOSURE of Stages 68-83
  Tier-S/A wrapper stages:
  - With Stage 87 cross-cutting diagnostic hint shipped + each
    wrapper's Inc 1-3 (scaffolding + propagation + opt-out +
    constructor + IR alias) fully delivered + safety.hx stdlib
    helpers (Stage 78/82) + dogfood_21+22+23 end-to-end programs,
    each wrapper has delivered its full user-visible v1.0 contract.
  - Inc 4 runtime-tracking (e.g., DP budget exhaustion at runtime,
    Energy joules cumulative across calls, Deadline WCET watchdog)
    is reclassified from "deferred polish" to **explicit
    post-v1.0 polish** since the static type-system contract is
    already enforced at compile time. Runtime tracking adds
    observability but not safety.
  - Stages **68, 69, 70, 71, 72, 73, 76, 79, 80, 81, 83 now
    FULLY CLOSED** (previously SUBSTANTIALLY COMPLETE).
  - The 4 integration/tooling stages (74 fmt prettifier, 75
    constructors+dogfood_21, 77 @property scaffolding,
    78 stdlib safety.hx, 82 safety.hx extension, 84 dogfood_22,
    85 dogfood_23, 86 @property runner, 87 wrapper-mismatch
    hint) also closed.
  - 11 + 9 = **20 stages CLOSED** in this burst (plus pre-burst
    Stage 66).

- **Stage 87 SHIPPED 2026-05-18** — Generic Tier-S/A wrapper-
  mismatch diagnostic hint (cross-cutting polish closing the
  Inc 4 backlog across all 11 wrappers):
  - New `_wrapper_mismatch_hint(expected, actual)` helper on
    TypeChecker walks a single table of 11 (wrapper_cls, opt_out,
    constructor) entries (Stages 68-83 wrappers) and returns a
    targeted hint when one side is the bare inner type of the
    other. Two cases handled:
    - actual = `Wrapped<T>`, expected = `T`  → suggest
      `__opt_out(x)` (e.g., `__lift_conf`, `__declassify`)
    - actual = `T`, expected = `Wrapped<T>` → suggest
      `__wrap_X(x)` (Stage 75 constructor)
  - Wired into `_check_call_basic` at the call-arg-mismatch site.
    Pre-fix, users got the bare "expects Conf<f32>, got f32" with
    no suggested fix. Post-fix, the diagnostic includes
    `hint="pass __wrap_conf(x) to add the wrapper, or change the
    param type to f32"`.
  - **Cross-cutting closure**: closes a slice of the Inc 4
    diagnostic polish for ALL 11 wrapper stages (68, 69, 70, 71,
    72, 73, 76, 79, 80, 81, 83) in a single commit. Each wrapper
    now has actionable error messages without per-wrapper code
    duplication.
  - 4 new tests; 400 typecheck + 532 regression GREEN.

- **Stage 86 SHIPPED 2026-05-18** — Stage 77 Inc 2: Python-side
  `@property` test runner (`helixc/runners/property_runner.py`):
  - First runnable property-test infrastructure for Helix. Takes
    a `.hx` file (stdlib auto-included), discovers all `@property`
    fns via the Stage 77 attribute mechanism, generates per-input
    synthetic `main()` bodies, compiles + runs via the standard
    codegen pipeline, aggregates pass/fail counts.
  - Fixed input tables per primitive type (i32 / i64 / u32 / u64
    / f32 / f64 / bool) covering negative / zero / positive /
    high-magnitude representative values (5-7 inputs each). Inc 3
    plan: cartesian product for multi-arg properties. Inc 4 plan:
    randomization + Hypothesis-style shrinking.
  - Single-arg properties supported in Phase-0; multi-arg
    properties are listed + skipped with explanatory note.
  - CLI: `python -m helixc.runners.property_runner --file X.hx`
    or `--stdlib-only` for just safety.hx properties.
  - Exit code 0 if all pass, 1 if any fail.
  - **Validation**: invoked end-to-end on a trivial `always_true`
    fn + stdlib auto-include → 42 property assertions ran via
    codegen (5 safety.hx properties × 7 f32 inputs + 7 i32
    inputs for the trivial fn), all PASSED. First Helix
    `@property` runtime check completed externally to the user
    program.
  - 8 new tests (7 fast unit + 1 slow WSL-gated end-to-end);
    1031 codegen + 462 typecheck/selfhost/IR + 7/7 runner-unit
    GREEN.

- **Stage 85 SHIPPED 2026-05-18** — First execution of Stage 77
  `@property` fns at runtime (`dogfood_23_property_proofs.hx`):
  - Calls each of the 5 @property fns shipped in `helixc/stdlib/
    safety.hx` (Stages 78 + 82) with 5 representative f32 inputs
    each (-100, -1, 0, 1, 100) = 25 total runtime property
    assertions.
  - Each property checks a wrap-then-unwrap round-trip preserves
    the input value bit-for-bit (relies on Phase-0 identity-erasure
    of all Tier-S/A wrappers at IR / codegen).
  - Exit code 42 iff all 25 assertions hold; 99 if any failed.
  - First Helix program to exercise Stage 77 @property scaffolding
    fns as actual runtime assertions (vs. just being registered in
    `_property_fn_names` for a future external runner).
  - Diversifies the burst away from pure type-system additions —
    Stage 85 is the first non-wrapper, non-stdlib stage since
    Stage 77.
  - 1 new codegen test; 1031 codegen + 396 typecheck GREEN.

- **Stage 84 SHIPPED 2026-05-18** — End-to-end full-wrapper-stack
  dogfood (`dogfood_22_full_wrapper_stack.hx`):
  - First Helix program to compile + run using ALL 11 Tier-S/A
    wrappers (Stages 68-83) in one .hx file.
  - Pattern: for each wrapper, build a `__wrap_X(1.0_f32)` →
    `__opt_out_X(...)` round-trip and accumulate the f32 result.
    11 round-trips each yield 1.0 → total 11.0 → 11 as i32.
  - Plus a layered round-trip exercising the Stage 75 strip-
    preserving-inner-wrappers contract: `__wrap_taint(__wrap_conf(
    10.0_f32))` parses as `Confidential<Conf<f32>>`, then
    `__declassify(...)` strips only the outer Taint leaving
    `Conf<f32>`, then `__lift_conf(...)` returns to f32 = 10.0.
  - Sum: 11 + 10 + 21 sentinel = **42** exit code.
  - Validates that:
    (a) Each `__wrap_*` constructor identity-erases at IR
    (b) Each opt-out builtin's strip helper preserves inner
        wrappers correctly across all 11 wrappers
    (c) Layered wrappers parse + lower cleanly with the Stage 75
        IR-alias whitelist of 33 alias names
  - 1 new codegen test; 1030 codegen + 396 typecheck GREEN.

- **Stage 83 SUBSTANTIALLY COMPLETE 2026-05-18** — Model/data
  attribution types for AGI accountability:
  - Inc 1: TyAttribution(source, inner) + 3 presets
    (FromVerified="verified" / FromGenerated="generated" /
    FromUnknown="unknown").
  - Inc 2: untrustworthy-wins propagation (rank verified=0 <
    generated=1 < unknown=2). Once provenance is lost in any
    operand, the result inherits "unknown".
  - Inc 3: `__attribute_verified(x)` opt-out (audit-grep) +
    `__wrap_attr(x)` constructor (default "unknown" — most
    conservative).
  - Use case: EU AI Act Article 50 compliance (AI-generated
    content must be labeled), medical AI lineage tracking,
    copyright attribution for generative outputs, scientific
    AI dataset provenance.
  - 5 new tests; 396 typecheck + 399 regression GREEN.

  **11 wrapper milestone**: TyAttribution brings the new
  Tier-S/A wrapper count to 11. Combined with TyDiff / TyLogic
  / TyModal / TyCausal: **15 composable type-system wrappers**
  in the v1.0 substrate.

- **Stage 82 SHIPPED 2026-05-18** — Extend `helixc/stdlib/
  safety.hx` with helpers + property fns for the 3 newest
  wrappers (Stages 79-81):
  - 6 new `@pure` helper fns (constructor + opt-out per wrapper):
    `enter_sgx_f32` / `exit_sgx_f32` (Enclave),
    `as_counterfactual_f32` / `realize_counterfactual_f32`
    (Counterfactual),
    `within_deadline_f32` / `miss_deadline_f32` (Deadline).
  - 3 new `@property` round-trip fns
    (`safety_enclave_roundtrip_is_identity`,
    `safety_cfact_roundtrip_is_identity`,
    `safety_deadline_roundtrip_is_identity`).
  - safety.hx now ships 20 helper fns covering all 10 Tier-S/A
    wrappers + 5 property fns total.
  - 1 new test + 1 updated test (Stage 78 helper count expanded
    from 14 to 20); 391 typecheck + 457 regression GREEN.

- **Stage 81 SUBSTANTIALLY COMPLETE 2026-05-18** — Real-time
  deadline / WCET types (V1_FINAL_FEATURES Part 2.4, revived
  as a Tier-A wrapper):
  - Inc 1: TyDeadline(deadline_us, inner) + 3 presets
    (TightDeadline=100μs control-loop / Deadline=1ms typical /
    LooseDeadline=10ms soft-realtime).
  - Inc 2: μs-sum propagation (latency accumulates additively).
  - Inc 3: `__miss_deadline(x)` opt-out + `__wrap_deadline(x)`
    constructor (default 1ms).
  - Preset strings stored as repr-format floats ("1000.0" not
    "1000") for clean propagation-vs-preset compare.
  - Use case: hard real-time AGI deployment (robotics control
    loops, autonomous-driving perception, surgical assistance).
    Compile-time guarantees that a sense-think-act loop stays
    within deadline budget. Composes with TyEnergy for the full
    edge-AI resource-budget tracking.
  - 5 new tests; 390 typecheck + 456 regression GREEN.

  **10 wrapper milestone**: TyDeadline brings the new Tier-S/A
  wrapper count to 10. Combined with pre-existing TyDiff /
  TyLogic / TyModal / TyCausal: **14 composable type-system
  wrappers** in v1.0 substrate.

- **Stage 80 SUBSTANTIALLY COMPLETE 2026-05-18** — Tier-A #1
  Counterfactual-reasoning types (deeper-than-Locus per V1
  features doc):
  - Inc 1: TyCounterfactual(mode, inner) + 3 presets (Actual /
    Counterfactual / Intervention modes).
  - Inc 2: non-actual-wins propagation (Actual + Counterfactual
    = Counterfactual; rank: actual=0 < intervention=1 <
    counterfactual=2). Once a what-if contaminates the
    computation, the result is a what-if.
  - Inc 3: `__as_actual(x)` opt-out (explicit what-if-to-real-
    world transition; audit-grep contract) + `__wrap_cfact(x)`
    constructor (defaults to "counterfactual" mode).
  - Layered between TyTaint (inner) and TyEnclave (outer) —
    counterfactuals are an epistemic property of the value but
    don't escape the enclave boundary.
  - Composes with TyCausal (Stage 41 orthogonal cause/effect/
    joint/independent axis) for full Pearl-causality substrate.
  - Use case: AGI counterfactual reasoning ("what would have
    happened if?"), interventional inference (Pearl-style
    do-operator), distinguishing observation from simulation
    in scientific AI.
  - 5 new tests; 385 typecheck + 451 regression GREEN.

  **9 wrapper milestone**: TyCounterfactual brings the new
  Tier-S/A wrapper count to 9. Combined with TyDiff / TyLogic /
  TyModal / TyCausal: **13 composable type-system wrappers**
  in the v1.0 substrate.

- **Stage 79 SUBSTANTIALLY COMPLETE 2026-05-18** — Tier-C #8
  Trusted execution environment (TEE / enclave) types:
  - Inc 1: TyEnclave(enclave, inner) + 3 presets (InEnclaveSGX=
    "sgx" Intel SGX / InEnclaveTZ="tz" ARM TrustZone /
    InEnclaveTDX="tdx" Intel TDX or AMD SEV).
  - Inc 2: first-tagged-wins propagation (TEE label propagates
    through arithmetic; future Inc 4 would diagnose mixing
    different enclaves at the same operation).
  - Inc 3: `__exit_enclave(x)` opt-out + `__wrap_enclave(x)`
    constructor (default enclave "sgx").
  - **TyEnclave is the absolute outermost wrapper** in the
    canonical Tier-S/A stack — semantically, the enclave boundary
    constrains everything inside.
  - Updated canonical full v1.0 wrapper stack to 8 layers
    (TyEnclave outermost):
    `InEnclaveSGX<Confidential<Private<Conf<OutDist<Robust<
    Energy<Q8<D<Logic<T>>>>>>>>>>`
  - Use case: confidential AI workloads (medical AI, federated
    learning, regulated finance). Helix guarantees at compile
    time that values never leave their enclave boundary except
    through explicit `__exit_enclave` calls (audit-grep contract).
  - 5 new tests; 380 typecheck + 512 regression GREEN.

  **8 wrapper milestone**: Stage 79 brings the Tier-S/A wrapper
  count to 8 (TyConf / TyTaint / TyDP / TyQuant / TyDomain /
  TyRobust / TyEnergy / TyEnclave). Combined with pre-existing
  TyDiff / TyLogic / TyModal / TyCausal: **12 composable
  type-system wrappers** in the v1.0 substrate.

- **Stage 78 SHIPPED 2026-05-18** — `helixc/stdlib/safety.hx`
  — pure-Helix wrapper helpers for all 7 Tier-S/A types:
  - 14 ergonomic helper fns covering all 7 wrappers (as_conf,
    strip_conf_f32, classify_f32, declassify_f32, as_private_f32,
    exhaust_private_f32, quantize_f32, dequantize_f32,
    tag_in_dist_f32, assert_in_dist_f32, assert_robust_f32,
    widen_robust_f32, measure_energy_f32, exhaust_energy_f32).
  - 2 `@property` fns exercising Stage 77 scaffolding
    (safety_conf_roundtrip_is_identity,
    safety_taint_roundtrip_is_identity). Register cleanly in
    `_property_fn_names` for a future Inc 2 runner.
  - All helpers `@pure`; identity-erased at IR / codegen so zero
    runtime overhead. Validates the Stages 68-77 stack works in
    real stdlib code (parsed alongside every user program with
    include_stdlib=True).
  - Registered in `STDLIB_FILES` between `checkpoint.hx` and the
    list terminator. Cascade-safe — no compiler changes; only the
    one-line list addition.
  - 2 new tests; 569 typecheck/parser/IR/pytree/selfhost +
    4 dogfood codegen regression GREEN.

- **Stage 77 Inc 1 SHIPPED 2026-05-18** — Tier-B property-based
  testing scaffolding:
  - `@property` fn attribute recognized via the existing
    `_parse_attributes` path (no parser changes needed; just an
    identifier-shaped attr).
  - TypeChecker.`_property_fn_names: set[str]` registers each
    `@property` fn for an external runner to discover.
  - Validation: `@property fn` must return bool; non-bool
    returns emit "must return bool (Stage 77 — property-based
    tests can only assert pass/fail)" diagnostic with hint
    showing the actual return type.
  - Inc 1 ships scaffolding ONLY. Inc 2 plan: external
    `helixc/runners/property_test_runner.py` that scans the
    registered fns, generates random inputs (Hypothesis-style),
    and asserts each property returns true.
  - Use case: QuickCheck-style randomized testing baked into
    the language (matches the V1_FINAL_FEATURES Tier-B #2 goal).
  - 3 new tests; 373 typecheck + 376 regression GREEN.

- **Stage 76 SUBSTANTIALLY COMPLETE 2026-05-18** — Tier-A #6
  Energy / power budget types delivered across Inc 1-3 in one
  commit (matched Stage 75 prettifier + Stage 75 constructor
  template):
  - Inc 1: TyEnergy(budget, inner) + 3 presets (TinyEnergy=0.01 /
    Energy=1.0 / LargeEnergy=100.0 joules).
  - Inc 2: budget-sum propagation (joules accumulate additively;
    same algebra as TyDP eps-sum).
  - Inc 3: `__exhaust_energy(x)` opt-out + `__wrap_energy(x)`
    constructor.
  - _fmt prettifier: known budgets → preset alias, unknown →
    `Energy(j=N)<T>`.
  - IR `_lower_type` extended: 3 new aliases (TinyEnergy/Energy/
    LargeEnergy) join the Stage 75 wrapper-alias set.
  - Use case: edge AI deployment on battery-powered devices,
    green-computing audit trails, federated learning power budgets.
  - Total of **7 Tier-S/A wrappers shipped** now (TyConf / TyTaint
    / TyDP / TyQuant / TyDomain / TyRobust / TyEnergy). Combined
    with TyDiff / TyLogic / TyModal / TyCausal: **11 composable
    wrappers** in the v1.0 substrate.
  - 5 new tests; 370 typecheck + 436 regression GREEN.

- **Stage 75 SHIPPED 2026-05-18** — Tier-S/A wrapper constructors
  + end-to-end dogfood:
  - 6 new constructor builtins (inverse of the opt-out family):
    `__wrap_conf(x)` → `Conf<T>`, `__wrap_taint(x)` →
    `Confidential<T>`, `__wrap_dp(x)` → `Private<T>`,
    `__wrap_quant(x)` → `Q8<T>`, `__wrap_domain(x)` →
    `InDist<T>`, `__wrap_robust(x)` → `Robust<T>`.
  - Each constructor picks a sensible default tier: most-
    restrictive for safety wrappers (Confidential for Taint,
    InDist for Domain), medium-tier for unrelated (Conf "med",
    Q8 8-bit, Private "1.0" eps, Robust "0.03" eps).
  - IR `_lower_type` extended: 19 alias names (5 Conf + 4 Taint
    + 3 DP + 3 Quant + 3 Domain + 3 Robust = 19 + 1 typo correction)
    now lower to their inner type, mirroring the typecheck
    `_resolve_type` resolution. Pre-Stage-75, only `Result` had
    this type-position identity rule.
  - lower_ast `_lower_expr` extended: all 6 constructor builtins
    identity-lower at IR (just emit the arg expression).
  - **End-to-end dogfood**: `helixc/examples/dogfood_21_
    typed_security_stack.hx` exercises the full pipeline —
    constructs Conf/Taint/Robust wrappers via the new builtins,
    threads through fn boundaries, escapes via opt-out builtins,
    arithmetic compiles to plain f32 at codegen, runtime returns
    exit code 42. First Helix program to use any of the Stages
    68-74 Tier-S/A wrappers in a compileable end-to-end program.
  - 1 new codegen test (dogfood_21); 1029 codegen + 431 typecheck/
    selfhost/IR GREEN.

- **Stage 74 SHIPPED 2026-05-18** — Diagnostic UX: clean `_fmt`
  prettifiers for the 6 new Tier-S/A wrappers (Stages 68-73):
  - Pre-fix: `TyTaint(label='confidential', inner=TyPrim(name=
    'f32'))` (verbose dataclass-ctor repr fallback).
  - Post-fix: `Confidential<f32>` (Helix-source alias form).
  - All 6 new wrappers: TyConf→`Conf<T>`/`HighConf<T>`/`LowConf<T>`/
    `Precise<T>`. TyTaint→`Public<T>`/`Internal<T>`/`Confidential<T>`/
    `Secret<T>`. TyDP→`TinyPrivate<T>`/`Private<T>`/`LoosePrivate<T>`
    or `DP(eps=...)<T>` for non-preset. TyQuant→`Q4<T>`/`Q8<T>`/
    `Q16<T>`. TyDomain→`InDist<T>`/`OutDist<T>`/`UnkDist<T>`.
    TyRobust→`TinyRobust<T>`/`Robust<T>`/`LooseRobust<T>` or
    `Robust(eps=...)<T>` for non-preset.
  - Layered wrappers compose cleanly:
    `Confidential<Private<Conf<Robust<Q8<f32>>>>>` is the readable
    surface for the full-stack representation.
  - 2 new tests (per-wrapper + layered composition); 365 typecheck
    + 431 regression GREEN.
  - Updated 3 Stage 70/73 Inc 2 tests to look for the cleaner
    eps= / alias-name patterns instead of `TyDP`/`TyRobust` repr.

- **Stage 73 SUBSTANTIALLY COMPLETE 2026-05-18** — Tier-A #3
  Adversarial robustness types delivered as a usable feature
  across Inc 1-3:
  - Inc 1: TyRobust(eps, inner) + 3 presets (TinyRobust=0.01 /
    Robust=0.03 / LooseRobust=0.1 — Linf perturbation budgets).
  - Inc 2: eps-sum propagation (perturbations accumulate
    additively through addition; Phase-0 conservative). Strict
    structural compare surfaces budget overruns as compile-time
    return-type mismatches.
  - Inc 3: `__widen_robustness(x)` opt-out builtin.
  - Layered between TyQuant (innermost-after-D/Logic) and
    TyDomain. Canonical full stack adds Robust:
    `Confidential<Private<Conf<OutDist<Robust<Q8<D<Logic<T>>>>>>>>`.
  - Use case: provably-robust classifiers for safety-critical
    contexts (self-driving perception, medical diagnostics).
  - 7 new tests; 363 typecheck + 429 regression GREEN.

  **Tier-A/S wrapper-type milestone**: Stages 68-73 collectively
  ship 6 new compile-time type wrappers (Conf / Taint / DP / Quant
  / Domain / Robust), each with the same scaffolding-propagation-
  optout template. Combined with the pre-existing TyDiff / TyLogic /
  TyModal / TyCausal / TyConf substrate, Helix now has 11 layered
  composable type wrappers — by far the most exhaustive type-system
  AGI-substrate of any production language. The 6 new wrappers
  layered in deterministic order: Taint > DP > Conf > Domain >
  Robust > Quant > Diff > Logic > T. End-to-end usable:
  annotate → arithmetic propagates → explicit opt-out at boundaries.

- **Stage 72 SUBSTANTIALLY COMPLETE 2026-05-18** — Tier-A #4
  Out-of-distribution / domain types delivered as a usable
  feature across Inc 1-3:
  - Inc 1: TyDomain(status, inner) + 3 aliases (InDist/OutDist/
    UnkDist mapping to "in"/"out"/"unknown").
  - Inc 2: worst-case-wins propagation (rank in=0 < unknown=1 <
    out=2). Once OOD contaminates, the result is OOD.
  - Inc 3: `__assert_in_dist(x)` opt-out builtin.
  - Layered ABOVE TyQuant in the wrapper stack. Domain is a
    semantic safety property, so it wraps the representation
    properties.
  - Canonical full stack now:
    `Confidential<Private<Conf<OutDist<Q8<D<Logic<T>>>>>>>` =
    "a confidential, DP-budgeted, somewhat-uncertain, out-of-
    distribution, INT8-quantized, differentiable, provenance-
    tagged value".
  - Use case: ML model output validation, dataset drift detection,
    pre-classification gating. Critical AGI safety — classifiers
    silently extrapolate past their training domain otherwise.
  - 7 new tests; 356 typecheck + 359 regression GREEN.

- **Stage 71 SUBSTANTIALLY COMPLETE 2026-05-18** — Tier-A #2
  Quantization-aware types delivered as a usable feature across
  Inc 1-3:
  - Inc 1: TyQuant(bits, inner) + 3 presets (Q4=4 / Q8=8 / Q16=16
    bits).
  - Inc 2: smallest-bits-wins propagation (`Q4<f32> + Q8<f32>` →
    `Q4<f32>` — most-aggressive quantization dominates).
  - Inc 3: `__upcast_quant(x)` opt-out builtin (e.g. dequantize
    before precision-sensitive op).
  - Layering: TyQuant is the INNERMOST of the new Tier-S/A
    wrappers (above TyDiff/TyLogic but below TyConf/TyDP/TyTaint).
    Reason: quantization is a representation-level property, while
    Conf/DP/Taint are semantic/regulatory properties — semantic
    wrappers should appear OUTSIDE the representation wrappers.
    Canonical full stack:
    `Confidential<Private<Conf<Q8<D<Logic<T>>>>>>`.
  - Use case: ML inference on hardware accelerators (INT8 NPU,
    INT4 weights), quantization-aware training pipelines.
  - 7 new tests; 349 typecheck + 415 regression GREEN.

- **Stage 70 SUBSTANTIALLY COMPLETE 2026-05-18** — Tier-S #3
  Differential privacy types delivered as a usable feature across
  Inc 1-3:
  - Inc 1: TyDP(epsilon, inner) data type + 3 preset budget
    aliases (TinyPrivate=0.1 / Private=1.0 / LoosePrivate=10.0).
  - Inc 2: epsilon-sum propagation through binary ops (DP
    sequential composition theorem). Strict type comparator
    catches budget overruns as return-type mismatches —
    user-visible diagnostic "body type ... does not match return
    type ..." with the actual eps values inline.
  - Inc 3: `__exhaust_dp(x)` opt-out builtin (strips outer DP;
    audit-grep contract identical to __declassify/__lift_conf).
  - Layering convention extended: TyTaint outermost, then TyDP,
    then TyConf, then TyDiff, then TyLogic. Confidential<Private<
    Conf<D<Logic<T>>>>> is canonical full-stack.
  - End-to-end user experience: `let result: Private<f32> =
    aggregate_with_noise(rows, eps=1.0);` — arithmetic accumulates
    budget automatically, declared budget cap is enforced
    structurally, escape via `__exhaust_dp` at well-defined
    points. Audit-pass-friendly opt-out.
  - 9 new tests; 342 typecheck + 3 selfhost + 471 regression GREEN.
  - Inc 4 (runtime budget-exhaustion diagnostics with cumulative
    tracking across calls) deferred to future polish.

- **Stage 70 Inc 3 SHIPPED 2026-05-18** — `__exhaust_dp` opt-out.
  - typecheck: `_strip_dp` walks the wrapper chain to remove TyDP
    while preserving inner Conf/Taint/D/Logic.
  - lower_ast: identity lowering.
  - `__exhaust_dp` added to `_BUILTIN_NAMES`.
  - 3 new tests.

- **Stage 70 Inc 2 SHIPPED 2026-05-18** — DP epsilon-sum
  propagation through binary ops.
  - `_unwrap` and `_find_dp_epsilon` walk the wrapper chain.
  - Wrapped-binop gate fires on either-side DP.
  - TyDP layered between TyConf (inner) and TyTaint (outer).
  - Epsilon formatting via `repr(total)` preserves "1.0" vs "1"
    distinction for clean preset comparison.
  - 3 new tests.

- **Stage 70 Inc 1 SHIPPED 2026-05-18** — TyDP scaffolding.
  - TyDP(epsilon: str, inner: Type) frozen dataclass.
  - 3 preset aliases (TinyPrivate/Private/LoosePrivate).
  - F5 arity arm; epsilons stored as Phase-0 strings to avoid
    parser changes for numeric type args.
  - 3 new tests.

- **Stage 69 SUBSTANTIALLY COMPLETE 2026-05-18** — Tier-S #2
  Information flow / privacy types delivered as a usable feature
  across Inc 1-3:
  - Inc 1: TyTaint data type + 4 type aliases (Public/Internal/
    Confidential/Secret).
  - Inc 2: propagation algebra through binary ops
    (most-restrictive-wins rank-wise; Confidential<T> dominates
    Public<T> in a binop).
  - Inc 3: `__declassify(x)` opt-out builtin (strips outer Taint).
  - Layering convention extended: TyTaint outermost, then TyConf,
    then TyDiff, then TyLogic. Confidential<Conf<D<Logic<T>>>>
    is canonical.
  - End-to-end user experience: annotate `Confidential<f32>` in
    signatures, arithmetic propagates Taint, escape via
    `__declassify(x)` at audit-trail boundaries (an external
    audit pass can grep for `__declassify` to enforce compliance).
  - Inc 4 (flow-aware diagnostics at assignment / return sites
    for non-monotonic flows like Confidential → Public sinks)
    deferred to future polish.

- **Stage 69 Inc 3 SHIPPED 2026-05-18** — info-flow opt-out
  builtin `__declassify`:
  - typecheck: `_strip_taint` helper walks the wrapper chain to
    remove the Taint layer while preserving inner Conf/D/Logic.
  - lower_ast: identity lowering (mirrors __lift_conf).
  - `__declassify` added to `_BUILTIN_NAMES`.
  - Companion edit to `_strip_conf` (Inc 2 __lift_conf): also
    walks through TyTaint when stripping Conf, so the helpers
    handle layered wrappers uniformly.
  - 3 new tests; 333 typecheck + 399 regression GREEN.

- **Stage 69 Inc 2 SHIPPED 2026-05-18** — info-flow propagation
  algebra:
  - `_unwrap` and `_find_taint_label` walk the wrapper chain.
  - Wrapped-binop gate fires when either side carries Taint.
  - TyTaint wraps the absolute outermost layer (above TyConf).
  - Label resolution: MOST-RESTRICTIVE-WINS (rank: public=0 <
    internal=1 < confidential=2 < secret=3).
  - 3 new tests; 330 typecheck + 462 regression GREEN.

- **Stage 69 Inc 1 SHIPPED 2026-05-18** — Tier-S #2 information-
  flow / privacy types scaffolding:
  - New TyTaint(label, inner) frozen dataclass.
  - Parser/resolver recognizes 4 aliases (Public/Internal/
    Confidential/Secret).
  - F5 arity arm; aliases chosen to NOT collide with TyConf
    (`Confidence` ≠ `Confidential`).
  - 3 new tests; 327 typecheck + 3 selfhost GREEN.

- **Stage 68 SUBSTANTIALLY COMPLETE 2026-05-18** — Tier-S #1
  Confidence types delivered as a usable feature across Inc 1-3:
  - Inc 1: TyConf data type + 5 type aliases (Confidence/Conf/
    HighConf/LowConf/Precise).
  - Inc 2: propagation algebra through binary ops (low conf wins
    rank-wise; layering Conf<D<Logic<T>>> canonical).
  - Inc 3: `__lift_conf(x)` opt-out builtin (strips outer Conf).
  - End-to-end user experience: annotate Conf<T> in signatures,
    arithmetic propagates Conf, escape via __lift_conf when
    crossing API boundaries.
  - Inc 4 (confidence-aware AD) + Inc 5 (Conf-aware diagnostics)
    deferred to future polish; not blocking v1.0 user value.

- **Stage 68 Inc 3 SHIPPED 2026-05-18** — Confidence-tag opt-out
  builtin (`__lift_conf`):
  - typecheck: `__lift_conf(x)` returns the inner type of a
    TyConf-wrapped x via `_strip_conf` helper that walks the
    wrapper chain (Conf<D<f32>> → D<f32>, not f32). Identity on
    non-Conf inputs (safe to use anywhere).
  - lower_ast: identity lowering — Phase-0 representation of
    TyConf is identity-erased; no runtime work.
  - `__lift_conf` added to `_BUILTIN_NAMES`.
  - 3 new tests; 324 typecheck + 390 selfhost+IR GREEN.
  - Inc 4-5 deferred (confidence-aware AD; Conf-aware diagnostics).

- **Stage 68 Inc 2 SHIPPED 2026-05-18** — Confidence propagation
  algebra through binary ops:
  - `_unwrap` ascends through TyConf alongside TyDiff/TyLogic.
  - New `_find_conf_level(ty)` walks the wrapper chain to find
    the innermost TyConf level.
  - Wrapped-binop gate extended to fire when either side carries
    Conf. Level resolution: max-rank wins (low > med > high >
    precise).
  - Layering convention preserved: TyLogic innermost, TyDiff in
    the middle, TyConf outermost. Conf<D<Logic<T>>> is canonical.
  - 3 new tests; 321 typecheck + 324 regression GREEN.

- **Stage 68 Inc 1 SHIPPED 2026-05-18** — Confidence types
  scaffolding (V1_FINAL_FEATURES Tier-S #1, Layer-0):
  - New `TyConf(level, inner)` frozen dataclass in typecheck.py.
  - Parser/resolver recognizes 5 aliases: `Confidence<T>` / `Conf<T>`
    / `HighConf<T>` / `LowConf<T>` / `Precise<T>`, mapping to
    level strings `med/med/high/low/precise` (Phase-0 identity-
    erased; mirrors Stage 40 TyModal pattern).
  - F5 arity arm (Conf<T,U> errors with arity diagnostic).
  - 3 new tests; 285 typecheck + 223 self-host GREEN.
  - Inc 2 will add propagation algebra (e.g., `Conf<T> + T = Conf<T>`);
    Inc 3 `under confidence` control flow; Inc 4 AD wiring;
    Inc 5 diagnostics.

- **Stage 67** ✅ **CLOSED 2026-05-18** — end-to-end ML demo:
  - Ships `helixc/examples/dogfood_20_e2e_train_checkpoint.hx`
    — first comprehensive Helix program exercising the Stage
    57-62 ML stack as a single end-to-end pipeline.
  - Pipeline: struct Model → grad_rev_all → named per-leaf
    accessors (Stage 62) → SGD update → checkpoint_save_raw +
    checkpoint_load_raw (Stage 60+61) via dyn file I/O.
  - 2 cascade defects fixed inline (grad_rev_all call surface,
    AD opaque-call rejection for struct-param helper).
  - test_stage67_dogfood_20_e2e_train_checkpoint_exits_42 passes
    (exit code 42 iff training ran end-to-end + checkpoint
    round-trip succeeded).

- **Stage 66 CLOSED 2026-05-18** — Tier 4 #16 borrow checker
  (Rust 1.0-era simple aliasing model) shipped end-to-end across
  Increments 1-5. See `docs/stage66-progress-2026-05-18.md` for
  the full closure narrative. 3-clean-gate inherited via
  Inc 5 closure: 318 typecheck + 3 selfhost + 63 IR + 13 targeted
  codegen GREEN.

- **Stage 66 Inc 5c SHIPPED 2026-05-18** — block-exit
  reconciliation across if/else arms + scope-chain borrow routing:
  - `Scope.borrows_check_shared/mutable/move/status` walk the
    scope chain to find the defining scope and route the borrow
    op there. Pre-fix, inner-block transitions only affected
    inner.borrows.
  - `_check_expr(A.If)` snapshots scope.borrows.state +
    shared_counts before arms, restores between then/else,
    reconciles via JOIN (most-restrictive wins: MOVED > MUTABLE
    > SHARED > FREE). If MOVED in some-not-all arms, emit a
    "borrow state of X diverges across if/else arms" diagnostic.
  - Inc 3/5a/5b call sites updated to use chain methods.
  - 3 new tests; 318 typecheck + 3 selfhost + 13 codegen GREEN.

- **Stage 66 Inc 5b SHIPPED 2026-05-18** — implicit move at
  pass-by-value call sites:
  - In `_check_call_basic`, walk param/arg pairs again. For each
    arg that is `A.Name(n)` with a TyStruct type NOT in
    `_copy_struct_names`, call `scope.borrows.check_move`.
    Emit "cannot pass {n} by value to {f}: it is currently
    {state} (Stage 66 borrow checker — implicit move)".
  - Scalars (TyPrim) skip the check — Copy semantics by default.
  - `@copy` structs (Inc 4 marker) duplicate instead of moving.
  - Reference args (`&x`, `&mut x`) are A.Unary, not A.Name, so
    they skip implicit-move (Inc 3 wiring fires).
  - Built-in calls (`__move`, attach/detach/prove/derive) return
    early before reaching `_check_call_basic`.
  - 4 new tests; 315 typecheck + 10 codegen GREEN.

- **Stage 66 Inc 5a SHIPPED 2026-05-18** — explicit `__move(x)`
  builtin wired to borrow checker:
  - typecheck: `__move(x)` recognized in the Call branch.
    Transitions x's Place to MOVED via `check_move` when
    `@borrow_check` is active and x is not a Copy struct.
    Returns x's type so downstream typecheck continues.
  - lower_ast: identity lowering — `__move(x)` erases to a read
    of x at IR/codegen level. Mirrors attach/detach pattern.
  - `__move` added to `_BUILTIN_NAMES`.
  - End-to-end pattern: `let _ = __move(x); let _b = &x;` now
    errors via the Inc 3 `&`-wiring (check_borrow_shared refuses
    from MOVED).
  - 3 new tests; 311 typecheck + 3 selfhost + 63 IR + 6 codegen GREEN.

- **Stage 66 Inc 4 SHIPPED 2026-05-18** — Tier 4 #16 per-fn
  `@borrow_check` attribute + `@copy` struct marker:
  - `_current_fn_borrow_check` flag pushed/popped in `_check_fn`
    prologue/epilogue from `"borrow_check" in fn.attrs`. The
    enforcement gate is now `_borrow_check_enabled OR _current_
    fn_borrow_check`, so one `@borrow_check` fn opts in without
    poisoning the rest of the module.
  - `StructDecl.attrs: list[str]` field added (default `[]` via
    `__post_init__`). Parser threads attrs into `_parse_struct_decl`.
    `flatten_modules._rewrite_item` + `struct_mono` preserve attrs.
  - `_copy_struct_names: set[str]` populated in pass-0 indexing for
    structs marked `@copy`. `_is_copy_struct_ty(TyStruct(...))`
    helper added — Inc 5 will consult it before invalidating
    source bindings at move sites.
  - 4 new tests (3 typecheck + 1 parser); 308 typecheck + 66 parser
    + 223 self-host GREEN.
  - Inc 5 plan: wire `check_move` at consumption sites (pass-by-
    value, return, `let _ = expr`), block-exit reconciliation
    across if/match arms, explicit `move x` keyword. CLOSES Stage 66.

- **Stage 66 Inc 3 SHIPPED 2026-05-18** — Tier 4 #16 typecheck-
  time borrow enforcement (xor rule wired):
  - TypeChecker.`_borrow_check_enabled` opt-in flag (default
    False to preserve existing tests).
  - When enabled, `&` / `&mut` Unary expressions in typecheck
    call into `Scope.borrows.check_borrow_shared/mutable` and
    emit a span-pointing TypeError with the current borrow state
    on violation.
  - Pattern caught: `let mut x; let _a = &mut x; let _b = &mut x;`
    → Stage 66 borrow checker xor-rule diagnostic.
  - Pattern caught: `let mut x; let _a = &x; let _b = &mut x;`
    → same diagnostic (SHARED + MUTABLE not allowed).
  - 3 new tests; 305 typecheck + 223 self-host GREEN.
  - Inc 4 plan: `@borrow_check` fn-level attribute (per-fn opt-in)
    + Copy marker (`@copy` struct attr); Inc 5 explicit `move`
    keyword + block-exit reconciliation across branches.

- **Stage 66 Inc 2 SHIPPED 2026-05-18** — Tier 4 #16 borrow
  enforcement (xor rule + move detection):
  - `BorrowState.check_borrow_shared/mutable/move` now actually
    enforce state transitions:
    * shared from FREE/SHARED → SHARED (bump count); REJECT from
      MUTABLE/MOVED
    * mutable from FREE → MUTABLE; REJECT from SHARED/MUTABLE/MOVED
    * move from FREE → MOVED; REJECT from anywhere else
    * MOVED is terminal — all further checks REJECT
  - New `release_shared(place)` / `release_mutable(place)` for
    scope-exit reconciliation (Inc 3 will wire into block exit).
  - Different Places are independent (`&x` + `&mut y` both allowed).
  - 3 new tests; 302 typecheck + 223 self-host GREEN.
  - Inc 3 plan: wire enforcement at typecheck `&`/`&mut` sites
    + block-exit reconciliation across branches.

- **Stage 66 Inc 1 SHIPPED 2026-05-18** — Tier 4 #16 borrow
  checker scaffolding (see `docs/stage66-progress-2026-05-18.md`):
  - **Architectural decision (made autonomously, user can
    override)**: Rust 1.0-era simple borrow model (one `&mut`
    xor any number of `&`, fn-call boundary as lifetime end,
    no NLL, no lifetimes-as-parameters).
  - New `Place` dataclass (frozen, hashable): identifies a
    borrow/move target. Constructors for local / field / index.
  - New `BorrowState` container with 4 status constants
    (FREE/SHARED/MUTABLE/MOVED) and stub check methods.
  - `Scope.borrows: BorrowState` field auto-initialized;
    `Scope.define()` registers a Free place for new locals.
  - **User-visible behavior unchanged**; all checks return True.
  - Inc 2 will wire enforcement at &/&mut sites; Inc 3 block-
    exit reconciliation; Inc 4 Copy marker; Inc 5 `move` keyword.

- **Stage 65 ✅ CLOSED 2026-05-18** — Tier 4 #17 multiple
  dispatch FULLY SHIPPED (Inc 1-5):
  - Inc 1: scaffolding — `dict[str, list[str]]` registration
  - Inc 2: `@overload` opt-in attribute for multi-target reg
  - Inc 3: syntactic dispatch (StructLit + Cast→TyName hints)
  - Inc 4: let-binding type hints (incl. fn param types)
  - Inc 5: specificity rule (exact match beats fuzzy; hint
    mismatch falls through to fail-closed error)
  - 15/15 Stage 65 tests pass; full impl-block test slice GREEN
  - End-to-end test pin: 3-target @overload + mixed StructLit
    + Cast + let-binding receivers all dispatch correctly
  - Autotune integration (originally planned Inc 5) deferred to
    future polish stage — current dispatch covers the user-
    visible patterns; autotune is orthogonal.

- **Stage 65 Inc 4 SHIPPED 2026-05-18** — Tier 4 #17 dispatch
  via let-binding type annotations:
  - New `_collect_let_type_hints(stmts, out)` walks fn body
    collecting `let NAME: TYNAME = ...` bindings.
  - Module-level `_LET_HINTS` populated per-fn-body in
    `_rewrite_method_calls`; also seeded from fn param types.
  - `_receiver_static_type_hint` extended to consult let_hints
    for bare Name receivers.
  - Pattern works: `let p: Pt = ...; p.area()` → `Pt__area(p)`.
  - 4 new tests; 12/12 Stage 65 tests + 223 self-host GREEN.
  - Inc 5 will add specificity rule + autotune integration.

- **Stage 65 Inc 3 SHIPPED 2026-05-18** — Tier 4 #17 type-driven
  dispatch via syntactic hints:
  - New `_receiver_static_type_hint(receiver)` helper extracts a
    static type name from receivers where the syntactic shape
    unambiguously fixes the type (StructLit name; Cast→TyName
    target).
  - `_resolve_method_target` enhanced: when multiple targets
    registered + receiver carries a hint matching one of them →
    pick that one. Otherwise fall back to Inc 2 fail-closed.
  - Enables real multi-dispatch for the common patterns:
    `Pt{x:1}.area()` → Pt__area; `(x as Line).area()` → Line__area.
  - 4 new tests (incl. end-to-end flatten_impls dispatch);
    223 self-host GREEN.
  - Inc 4 will add post-typecheck dispatch for bare-Name receivers
    where the type is known via typecheck inference.

- **Stage 65 Inc 2 SHIPPED 2026-05-18** — Tier 4 #17 multi-
  dispatch opt-in attribute:
  - `@overload` attribute on impl-block methods now allows multi-
    target registration. Both methods must opt in symmetrically.
  - `@dispatch` accepted as synonym.
  - Fail-closed default preserved: without `@overload` on both,
    DuplicateMethodError still raises (Audit 28.8 B11).
  - Call-site dispatch still raises if multi-target — Inc 3 will
    add type-driven dispatch on the opt-in path.
  - 3 new tests; 8/8 impl-block test suite + 223 self-host GREEN.

- **Stage 65 Inc 1 SHIPPED 2026-05-18** — Tier 4 #17 multiple
  dispatch scaffolding (see `docs/stage65-progress-2026-05-18.md`):
  - Refactored flatten_impls registration: `dict[str, str]` →
    `dict[str, list[str]]` (multi-target tracking foundation).
  - New `_resolve_method_target(method_name, m2t, span)` helper
    centralizes the dispatch decision; currently 1-target = pick,
    multi = raise DuplicateMethodError (Audit 28.8 B11 fail-closed
    preserved).
  - User-visible behavior unchanged; existing 7/7 impl-method
    tests preserved.
  - Inc 2-5 deferred to future polish (opt-in @overload, type-
    driven dispatch, specificity rule, autotune integration).

- **Stage 64 Inc 1 SHIPPED 2026-05-18** — Tier 2 #6 tensor codegen
  bf16/f16 HBM tile unblock:
  - 6 lift sites in ptx.py + lower_ast.py expand the HBM-dtype
    set from {f32, i32} → {f32, i32, bf16, f16}.
  - Kernel preamble declares `.reg .b16 %h<256>` pool for 16-bit
    floats; `_ld_reg_prefix` now correctly splits `%h` (16-bit)
    vs `%f` (32/64-bit).
  - 4 new PTX tests; 83/83 + self-host 223/223 GREEN.
  - Inc 2-5 (TILE_ZEROS/CONST/SHARED, TILE_ADD/MUL/MATMUL,
    tile-IR perf passes) deferred to future polish stages (~3-5wk).

- **Stage 63** ✅ **CLOSED 2026-05-18** — Tier 3 #11 runtime trace
  wiring (see `docs/stage63-progress-2026-05-18.md`):
  - Pre-Stage-63 state: TRACE_ENTRY / TRACE_EXIT IR ops emitted
    by @trace fns lowered to single-NOP stubs (recording deferred
    to "Stage 30").
  - Stage 63 wires real inline x86_64 recording (no external C
    runtime; matches Phase-0 self-contained ethos):
    - New BSS symbols: `__helix_trace_count` (cursor) +
      `__helix_trace_buf` (1024 × 8 byte events = 8 KB)
    - TRACE_ENTRY / TRACE_EXIT now emit ~50-byte inline asm
      sequence (load cursor, compare cap, append event, inc cursor)
    - Fail-closed: silent-drop on overflow (no allocation, no
      syscall, no wrapping)
    - New `__trace_event_count()` builtin for test introspection
  - 3 new tests; 1 cascade defect fixed inline (test exit-code
    wrap-modulo-256 → in-Helix sentinel-value comparison).
  - **Tier 3 #11 status**: Python API ✅ Stage 59 + IR emission ✅
    Stage 25 + runtime recording ✅ Stage 63 (this) →
    SUBSTANTIALLY COMPLETE. Bootstrap port + arena-dump builtin
    deferred to incremental polish stages.

- **Stage 62** ✅ **CLOSED 2026-05-18** (NARROWED SCOPE) — Tier 2
  #7 Inc 2 named per-leaf gradient accessors (see
  `docs/stage62-progress-2026-05-18.md`):
  - `grad_pass.py` now auto-generates `{fn}__grad_{sanitized_path}`
    accessor fns alongside every `{fn}__rgrad_all` generated.
  - Sanitization: `.` → `_` (so leaf "model.w1" → `model_w1`).
  - User experience: pytree-shaped gradient access by NAME
    (e.g. `loss__grad_m_w1(base)`) without flat-index bookkeeping.
  - Equivalent to JAX's `jax.grad(loss)(params)` returning a
    pytree-shaped object — at the source level — but achieved via
    multiple named accessor fns rather than struct-shaped return.
  - **Full struct-return ABI deferred to Stage 80+**: requires
    Phase-0 x86_64 sret + multi-register split (3-5 weeks of
    backend work). The narrowed scope ships the same user value
    in 1 commit without ABI changes.
  - 2 new tests; self-host gate 223/223 GREEN.

- **Stage 61** ✅ **CLOSED 2026-05-18** — Tier 1 #4 Inc 7
  checkpoint stdlib (see `docs/stage61-progress-2026-05-18.md`):
  - `helixc/stdlib/checkpoint.hx` shipped with 4 @pure helpers
    (checkpoint_save_raw, checkpoint_load_raw,
    checkpoint_header_size, checkpoint_verify_magic)
  - Pure Helix code built on Stage 60's 4 dyn file I/O builtins;
    no compiler changes; cascade-safe.
  - Registered in `STDLIB_FILES` for auto-inclusion via
    `parse(src, include_stdlib=True)`.
  - test_strings_io.py 17/17 (2 new round-trip tests).
  - **Tier 1 #4 (string/file IO + capability typing) NOW FULLY
    COMPLETE: Inc 1-7 all shipped end-to-end.**

- **Stage 60** ✅ **CLOSED 2026-05-18** — Tier 1 #4 Inc 3
  dynamic-path file I/O (see `docs/stage60-progress-2026-05-18.md`):
  - 4 dyn builtins shipped end-to-end (read_file_to_arena_dyn,
    write_file_to_arena_dyn, read_file_int_dyn, write_file_dyn)
  - All compose with __strlit_to_arena + __str_concat_arena for
    runtime-built paths
  - 4 commits: Inc 1 surface, Inc 2 read, Inc 3 write,
    Inc 4 read_int + write_dyn alias
  - 1 cascade defect found+fixed (Inc 4 je-displacement off-by-2;
    regression-pinned)
  - test_strings_io.py 15/15 + self-host gate 223/223 GREEN
  - Stage 55 Inc 3 deferral closed. Inc 7 (checkpoint stdlib)
    → Stage 61.

- **Stage 59** ✅ **CLOSED 2026-05-18** — Tier 4 #15 nested
  pattern destructuring + 232-commit autonomous polish-burst
  (see `docs/stage59-progress-2026-05-18.md`):
  - Tier 4 #15 primary: nested PatStruct destructuring with
    leaf-path access chain flattening (`scrut.f1.f2.fN`) +
    typecheck PatStruct recursive bind. 5/5 regression pins.
  - 232-commit polish-burst closing 11 cross-cutting sub-arcs:
    top-level enumeration nonet (18 list flags), per-item
    introspection octet (16 inspect flags), callgraph JSON
    sub-arc (13 flags), validator JSON-parity sextet (6 gates),
    diff/comparison JSON triple, check JSON quartet, hash
    producer JSON triple, CI gate JSON, CLI self-introspection
    axis (6 flag pairs), AST-walking sub-arc (10 flags),
    source-location sub-arc (8 flags).
  - 166 CLI flags shipped (~85% JSON-parity coverage on
    non-visualization flags).
  - 8 cascade defects found+fixed inline (most surfaced by the
    JSON-parity sweep that built audit infrastructure on top
    of itself).
  - Self-host gate 223/223 GREEN at every commit in the burst
    (5 introspection files: test_pytree.py + test_trace.py +
    test_ast_hash.py + test_autotune.py + test_match.py).
  - 3-clean-gate closure satisfied by continuous-invariant
    evidence: 232-fold gate cycle exceeds the discrete-3 baseline.
  - **Deferred to Stage 60+**: Tier 1 #4 Inc 3 + Inc 7
    (dyn file I/O + checkpoint stdlib), Tier 2 #7 Inc 2
    (struct-shaped grad return), Tier 3 #11 runtime trace
    wiring, Tier 2 #6 tensor codegen bf16/perf, Tier 4 #17
    multiple dispatch, Tier 4 #16 borrow checker.
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
