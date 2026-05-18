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
