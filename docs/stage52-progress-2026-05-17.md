# Stage 52 Progress — 2026-05-17

## Stage Goal

Stage 52 ships **modal-origin taint-tracking** — a typecheck-side
pass that closes the Stage 40 closure gate-1 H1 documented
limitation: "F1 syntactic-only guard is bypassed by let-binding
and helper-fn indirection."

The AI-safety property at stake (per Stage 40):

> Cross-modal upgrade must go through an AUDITED epistemic-
> transition (confirm: Believed → Known; act_on: Goal → Known).
> Unwrap-rewrap (`into_X(from_Y(v))`) is a category mistake at
> the heart of many AI safety failures.

Stage 40 gate-1 added a syntactic guard at `into_X(from_Y(v))`
inline-form. Stage 52 closes the let-binding bypass + while-loop
Assign bypass + match-arm Assign bypass. Helper-fn indirection
remains the LAST bypass — deferred to Stage 53.

## Increment breakdown

### Inc 1 — Basic flat-dict taint tracking (commit c274059)

- New `_modal_origin_provenance: dict[str, str]` map: var_name
  → modal-kind ('known'/'believed'/'goal'/'uncertain') when
  bound to `from_X(...)`.
- Populated at Let-stmt when value is `Call(from_X, ...)`.
- Cleared at check() + _check_fn entry (per-fn boundary).
- Consulted at `into_Y(name)` site in the F1 launder guard:
  if `name`'s tracked source_kind != target_kind, emit launder
  diagnostic with same epistemic-upgrade hint table as inline
  form. New diagnostic tag: "via let-binding bypass".

Polarity flip: test_stage40_f1_known_limitation_let_bypass →
test_stage40_f1_let_bypass_closed_by_stage52_taint_tracking.

### Inc 2 — Scope-stack discipline + Assign-arm POPULATE (commit 2925121)

Gate-1 audits caught 1 CRITICAL + 5 HIGH. Fixed inline per F8
recommendation ("Inc 2 deferral premature — gate-1 surfaced the
defect already"):

- **C1 (CRITICAL) + HIGH-3**: Assign-arm POPULATE on `from_X(...)`
  RHS (in addition to the existing POP on opaque RHS). Closes
  `let mut r: i32 = 0; while ... { r = from_uncertain(u); }`
  silent launder.
- **F1e + HIGH-5**: scope-stack via `_modal_origin_let_block_scopes`
  parallel list. Selective restore at `_check_block` exit (and
  `_check_expr_in_block_scope`): inner-LET shadows drop their
  entry + restore outer's saved taint if present; inner-Assign
  mutations PROPAGATE upward (INVERTED from Result-provenance
  which DROPS on inner-Assign — modal-origin's AI-safety
  semantics say "any from_X introduction MUST surface").
- **H2**: rewrote stale "let-binding bypass deferred" comment at
  inline-form guard.
- **F4/M3**: stripped self-referential "Stage 52 Inc 1 catches
  this" from user-facing diagnostic.

### Inc 3 — Match-arm parallel-union (commit c9d8915)

Gate-1 HIGH-1: `match cond { true => r = from_uncertain(u),
false => r = 0 }; into_known(r)` was silent because arm 2's
`r = 0` POPPED arm 1's installed taint via the Inc 2 Assign-arm
pop branch. Sequential arm processing made the last arm "win".

Inc 3 implements PARALLEL UNION semantics:
- Snapshot `_modal_origin_provenance` before each arm.
- Restore to pre-match state between arms (each arm starts
  fresh from outer state).
- After all arms: UNION arm-result dicts into post-match
  state. Any name tainted by ANY arm propagates conservatively.
- Multi-kind union (arm1=Uncertain, arm2=Known) currently
  keeps FIRST arm's kind — refined diagnostic deferred to
  Inc 4 polish.

### Inc 4 — Closure audits + ship

**Note: see gate-N closure sections below for current state**
(this section captures the initial Inc 4 plan; gates 2-6 have
since landed many cascading-defect fixes — read the gate-N
appendix subsections for the live picture).

## Verification (post-gate-6)

- 80/80 Stage 40 modal tests pass (was 57 at Inc 3; +23
  regression pins added across gates 2-6, covering each
  closure-round fix path)
- Self-host cascade re-verified live: PASS G2..G11 byte-
  identical sha=`a6f1ee44`, smoke 4/4 PASS
- dogfood_16 + dogfood_17 still exit 42
- Original Inc 1 let-bypass: caught
- HIGH-3 while-loop Assign: caught (Inc 2)
- HIGH-1 match-arm parallel: caught (Inc 3)
- F1e false-positive (inner-let shadow leak): closed (Inc 2)
- Gate-3 NEW-HIGH-1/2/3/4/5: closed (restore-domain + A.If
  cleared-branch + A.Match cleared-arm + meta over-broad
  drop guard)
- Gate-4 HIGH-1: PatBind taint propagation (scrutinee Name);
  CRITICAL-1: documented Phase-0 limitation (loop-body
  multi-kind, same as if-no-else; deferred Inc 4 multi-kind
  diagnostic)
- Gate-5 HIGH-1: PatBind taint copy hoisted ABOVE guard
- Gate-6 CRITICAL-1/2/3: unified `_modal_origin_of_expr`
  helper closes 3 distinct silent miscompiles (Call-form
  scrutinee + let/Assign name-alias + PatOr-of-PatBind)
  via single source-of-truth; F1 dup dict removed
- Negative test (untainted match, same-kind alias): silent

## Deferred (Stage 52 Inc 4 / Stage 53)

- **HIGH-2** (Inc 4): recursive RHS yield detection — `let r =
  if cond { from_X(u) } else { ... }; into_Y(r)` requires
  walking the RHS subtree (Block / If / Match yielding from_X).
- **F2** (Stage 53): helper-fn indirection — `fn launder(x:
  i32) -> Known<i32> { into_known(x) }` called with from_X
  result. Inter-procedural taint propagation — different
  defect class from let-bypass.
- **Multi-kind union** (Inc 4 polish): when arm1=Uncertain and
  arm2=Known both taint a name, current implementation keeps
  FIRST arm's kind. Richer "could be either" diagnostic
  needs `Union[str, set[str]]` dict-value shape.
- **Member-access** (Stage 49+ pattern): `let p: Pair = Pair {
  x: from_uncertain(u) }; into_X(p.x)` — same defect class as
  Stage 48 F5 (struct-field provenance). Stage 53+ work.

## Lineage

- Stage 40 closure gate-1 H1 documented the let-binding bypass
  as a Phase-0 known limitation, deferred to "a future
  taint-tracking pass." That's Stage 52.
- Pattern infrastructure copied from Stage 46-48
  `_result_constructor_provenance` (which itself went through
  5+ HIGHs during its own closure cycles). Stage 52 reuses the
  proven flat-dict + per-fn clear + scope-stack approach with
  INVERTED restore semantics (modal-origin propagates on
  Assign; Result drops on Assign).
- Audit cascading-defect rhythm continues: Stage 52 gate-1
  alone caught 1 CRITICAL + 5 HIGH that Inc 1 missed.

### Gate-3 cascading-defect surface (2026-05-17 mid-afternoon)

Gate-3 audit returned 5 NEW HIGH on the gate-2 if-else +
multi-kind code:

- **NEW-HIGH-1** (silent miscompile): inner-let-shadow with
  non-tainted i32 outer let — restore loop iterated
  current.keys(), but the inner let's POP had already
  removed the name. Saved outer taint was never restored.
  Fixed: iterate `inner_modal_lets` directly and restore
  from `saved_modal_origin` per-name.
- **NEW-HIGH-2/3** (false positive): A.If union saw pre-if
  taint + arm results, no concept of "branch assigned
  without installing taint". Both-arms-clear or no-else +
  then-clear cases falsely propagated pre-if taint.
  Fixed: added `_modal_origin_assigns_block_scopes` parallel
  stack; A.If captures `branch_assigns[i]` per branch;
  union loop drops names where `name in assigns AND name
  not in arm_result`.
- **NEW-HIGH-4** (false positive): symmetric A.Match
  cleared-arm fix — same `cleared_names_match` drop logic.
- **NEW-HIGH-5** (meta): the gate-3 fixes must NOT over-
  broadly suppress real launders. Pinned dual-test
  (`...real_launder_still_fires_with_drop_path`): if-then
  where then-branch INSTALLS taint must still fire — the
  cleared-name drop only triggers on assigns that don't
  install taint.

All 5 NEW-HIGH have post-fix regression pins in
`test_stage40_modal.py` (5 tests, ~70 total Stage 40 now).
Stage 40 modal sweep 65→70 passing post-pins.

### Gate-4 closure (2026-05-17 evening)

Gate-4 audit returned 1 CRITICAL + 1 HIGH:

- **HIGH-1 (silent miscompile)** — PatBind in match-arm did
  not propagate scrutinee taint. `let r = from_uncertain(u);
  match r { x => into_known(x) }` silently passed because the
  pattern binding only wrote to the value scope, not to
  `_modal_origin_provenance`. Direct AI-safety bypass via
  trivial bind.
  Fixed: in A.Match arm processing (placed INSIDE the snapshot
  region per gate-4 MEDIUM-2 constraint), when scrutinee is a
  Name with tracked modal origin AND pattern is a top-level
  PatBind, copy taint to the bound name.
- **CRITICAL-1 (Phase-0 limitation, not closeable here)** —
  loop-body INSTALLs different modal kind than pre-loop. At
  the 0-iter case at runtime, the original kind persists; the
  static dict carries the new kind. into_X matching new kind
  silently passes when the runtime path has the OLD kind.
  Audit recommended "mirror A.If no-else union semantics" but
  this is wrong — union would DROP on multi-kind divergence
  (still silent). Verified independently that A.If no-else
  has the EXACT same Phase-0 limitation. Drop-on-conflict is
  the chosen design philosophy (gate-2 HIGH-C); the proper
  fix is the deferred Inc 4 multi-kind diagnostic ("could be
  X or Y, neither matches target"). Documented + pinned with
  current behavior (`test_stage52_gate4_critical_1_loop_body_
  phase0_limitation_documented` + parallel if-no-else test).
  When Inc 4 lands the multi-kind diagnostic, BOTH pinned
  tests must flip in lockstep.

Stage 40 sweep 70→73 passing post-pins (1 closable HIGH +
2 Phase-0 limitation pins).

### Gate-5 closure (2026-05-17 evening)

Gate-5 audit returned 1 HIGH + 1 MEDIUM:

- **HIGH-1 (silent miscompile)** — PatBind taint propagation
  ran AFTER guard check, not before. A guard expression that
  called a modal eliminator on the bound name (e.g. `match r
  { x if into_known(x) > 0 => 1, _ => 0 }`) consulted
  `_modal_origin_provenance['x']` BEFORE the PatBind copy
  installed taint, so the launder consult found NO entry and
  silently passed. Direct AI-safety bypass via the guard slot.
  Same defect class as gate-4 HIGH-1 but routed differently.
  Fixed: hoisted the snapshot + PatBind taint copy to BEFORE
  the guard check. Order now: bind → snapshot → PatBind taint
  → guard → arm body.
- **MEDIUM-1** — non-top-level PatBind (PatTuple/PatVariant/
  PatOr sub-binds) intentionally skipped (sub-binds receive
  value fragments, not whole scrutinee). Phase-0 has no
  modal-typed tuple/struct fields. Documented in code comment
  to prevent a future contributor from "fixing" it incorrectly.

Stage 40 sweep 73→74 passing post-gate-5-pin.

### Gate-6 closure (2026-05-17 evening)

Triple-parallel audit (silent-failure + type-design + code-review)
returned 3 NEW CRITICAL + 1 HIGH + 1 doc-drift:

- **silent-failure CRITICAL-1**: Call-form match scrutinee bypassed
  PatBind taint (Name-only check). Fixed: unified `_modal_origin_
  of_expr` helper handles A.Call too.
- **silent-failure CRITICAL-2**: let-alias `let s = r;` (and Assign-
  alias `s = r;`) dropped taint when r was tainted. Fixed: helper
  handles A.Name with provenance lookup.
- **silent-failure CRITICAL-3**: PatOr-of-same-PatBind bypassed
  taint copy. Fixed: detect PatOr where every alt is PatBind of
  same name, treat as top-level PatBind for copy.
- **type-design HIGH F1**: residual `_modal_elim_kind` local dict
  at line 4357 violated gate-2 F3 hoisting invariant. Fixed:
  deleted; replaced with module-level `_MODAL_ELIM_TO_KIND`.
- **code-review MEDIUM**: progress doc had stale Inc 4 section
  + "Verification (post-Inc-3)" claimed 57/57 tests. Fixed:
  rewrote with gate-N pointer + refreshed test count.

Plus LATENT bug surfaced by the gate-6 let-alias fix: double-pop
in `_check_expr_in_block_scope` clobbered `_last_modal_assigns_
popped` when arm body was a Block. Fixed: UNION semantics preserve
inner-block contribution. Plus cleared-vs-installed refinement:
only mark a name cleared if NO arm installed taint.

Stage 40 sweep 74→80 passing post-gate-6 (6 new pins: 4 positives
for 3 CRITICAL + 1 bonus Assign-alias variant, 1 negative for
same-kind round-trip, 1 grep-pin for F1 invariant).

### Gate-7 closure (2026-05-17 evening)

Triple-parallel returned 3 HIGH (silent-failure) + 1 HIGH (type-
design) + 1 IMPORTANT (code-review):

- **silent-failure HIGH-3**: if-no-else / match-arm where the
  identity arm preserves pre-state taint and another arm clears.
  At runtime the 0-iter / identity path leaks the launder.
  Fixed via `kept_somewhere` / `kept_somewhere_match` sets: any
  name preserved in any arm's result overrides cleared.
  SEMANTIC FLIP: NEW-HIGH-3 + NEW-HIGH-4 prior tests asserted
  DROP (gate-3 drop-on-conflict design); gate-7 audit correctly
  identified these as silently missing real-runtime launders.
  Aligned with the stage's AI-safety property "category-error
  launders MUST be caught".
- **silent-failure HIGH-1+2** (loops): A.For/A.While/A.Loop
  body opaque-clear with pre-loop preserved taint. DEFERRED to
  Stage 52 Inc 5 (requires new union helper for loops).
- **type-design HIGH-1**: `_last_modal_assigns_popped` not cleared
  at `_check_fn` entry — masked by always-precedes-read ordering
  but fragile. Fixed: defensive clear added.
- **code-review IMPORTANT**: stale primary docstring on NEW-HIGH-3
  test (still claimed DROP) — fixed.

5 new pins (gate-7 standalone HIGH-3 + 3 negatives + type-design
grep-pin). Stage 40 sweep 80→85.

### Gate-8 closure (2026-05-17 evening) — FIRST CLEAN

Triple-parallel returned ALL 3 LANES CLEAN:
- silent-failure: 0 CRITICAL, 0 HIGH, 0 MEDIUM. Cascade has
  converged after 7 rounds.
- type-design: 0 HIGH+; 1 MEDIUM polish (redundant
  `installed_names` check — strict subset of `kept_somewhere`,
  dropped); 2 LOW deferred (F2 Literal type alias, F3 dataclass
  refactor).
- code-review: 0 HIGH+; 1 IMPORTANT docstring drift (NEW-HIGH-3
  primary docstring) — fixed inline.

Polish landed: dropped redundant installed_names guard in both
A.If and A.Match unions; flipped NEW-HIGH-3 + NEW-HIGH-4 docstrings.

This was the 1st of 3 consecutive clean gates required by the
3-clean-gate closure protocol.

### Stage 52 Inc 5 (2026-05-17 evening): loop body union

Closes the gate-7 HIGH-1+2 deferred work. New helper
`_check_loop_body_with_modal_union` applies the kept_somewhere
union semantics to A.For/A.While/A.Loop body checking. Same
defect class as A.If no-else identity arm — the 0-iter case
preserves pre-loop taint, so into_X after loop fires if pre-
loop kind mismatches target. 3 new pins (while-body, for-body,
negative same-kind round-trip). Phase-0 multi-kind divergence
(gate-4 CRITICAL-1) still drops as documented.

### Gate-9 closure (2026-05-17 evening) — partial verdict

Triple-parallel:
- silent-failure: pending
- type-design: pending
- code-review: 0 CRITICAL, 0 HIGH on code surface. 3 IMPORTANT
  doc-drift items (this progress doc missing gate-6+ sections;
  ROADMAP Stage 53 entry stale; Stage 53 blueprint doc stale
  post-implementation). All 3 fixed inline.

## Stage 53 Inc 1 follow-on landing (2026-05-17 evening, commit 179678d)

Closes the LAST remaining modal-launder bypass — helper-function
indirection. This is the Stage 40 H1 "different defect class"
deferred from Stage 52.

Implementation (4 surgical edits + 1 hoist):
1. New `_fn_modal_return_kind: dict[str, str]` populated in
   `_register_fn` from `sig.ret` when TyModal.
2. `_modal_origin_of_expr` extended with `Call(user_fn, ...)`
   case → all 3 install sites inherit Stage 53 automatically.
3. Launder check at user-fn call site (analog of F1 into_X
   check): if fn returns modal kind AND any arg has DIFFERENT
   modal-origin → fire.
4. `_modal_upgrade_hint` hoisted to module-level
   `_MODAL_UPGRADE_HINT` (gate-2 F3 single-source-of-truth).

7 regression pins (5 Inc 1 + 2 Inc 2 verifying Assign + match-
scrutinee paths). Stage 40 sweep 90→95 at Stage 53 Inc 1+2; the
gate-9 doc fixes + Inc 5 loop body + Inc 6 recursive yield-from-
modal subsequently added more pins, bringing total to 100+ at
gate-10 / Inc 7 closure point. Diagnostic message:
"launders Uncertain<T> into Known<T> via helper-fn indirection".

## Stage 52 Inc 7 follow-on landing (2026-05-17 late evening)

Gate-10 silent-failure caught 1 NEW HIGH: Inc 6's recursive
`_modal_origin_of_expr` was wired to the user-fn launder check
(Stage 53) but NOT to the builtin into_X launder check. Inline
forms `into_known(match scrut { x => from_uncertain(u) })` (or
inline if/block) silently passed — asymmetric coverage between
the user-fn site and the builtin site.

Fix (Inc 7): replaced the 2 narrow syntactic guards at the
builtin into_X check (A.Call + A.Name) with a single unified
`_modal_origin_of_expr(args[0])` consult. Coverage now symmetric
across all 5 consult sites (builtin into_X + user-fn call +
Let-RHS + Assign-RHS + match-scrutinee).

3 new regression pins (inline match/if/block in into_known).
Diagnostic message form refined: "via yielded modal expression
(a match/if/block tail with from_X(...))".

Stage 52 surface now catches **17+ distinct modal-launder paths**
(up from 14 pre-Inc-7).

## Stage 50 retry: Exp A executed (2026-05-17 evening, commit 10cab73)

Per diagnostic plan, Exp A measured combined bootstrap source:
732,396 bytes. Well under 1 MB BUF_SIZE. H4 (SIGILL buffer
overflow recurrence) DEFINITIVELY RULED OUT. Top hypothesis
H1 (stack overflow from fixed 1024-byte Helix codegen prologue)
remains; Exp B (`ulimit -s unlimited`) is the next step for the
Stage 50 retry attempt.

## Inc 8–12 + gates 11–14 follow-on (2026-05-17 late evening)

After the Inc 7 closure point above, the cascading-defect rhythm
continued for 5 more increments + 4 more gates. Each Inc shipped
caught the next defect class:

- **Inc 8 (e9d3d6d, gate-11 silent-failure HIGH-1)**: A.UnsafeBlock
  arm in recursive `_modal_origin_of_expr`. Inc 6's helper handled
  Block/If/Match but missed UnsafeBlock — `into_known(unsafe {
  from_uncertain(u) })` silently passed.
- **Inc 9 (006df58, gate-12 type-design HIGH)**: F2 partial
  application of ModalKind Literal alias — Inc 8 retyped only
  `_MODAL_ELIM_TO_KIND`, leaving `_modal_origin_provenance`,
  `_fn_modal_return_kind`, helper return, and 4 locals still
  raw `str`. Inc 9 propagated ModalKind to all 7 sites + added
  runtime assertion at `_register_fn` populate (boundary guard
  for AST `TyModal.kind`).
- **Inc 10 (40a791d, gate-12 silent-failure CRITICAL-1)**: A.Cast
  arm. `into_known(from_uncertain(u) as i32)` silently passed —
  the 3-character `as T` annotation bypassed all category-error
  audits. Same defect class as Inc 8 (UnsafeBlock).
- **Inc 11 (9ab8123, proactive cascade-break)**: scanned all 32
  A.Expr nodes for wrapper-class gaps. Found A.Unary + A.Binary;
  closed both in same commit BEFORE gate-13 had to find them.
  Binary's value-merge semantics (one-sided propagate vs
  control-flow drop) intentionally DIFFERS from If/Match — see
  the long comment at the Binary arm for the 3-domain
  distinction (expression-tree merge vs control-flow vs
  statement-scope name-map).
- **Inc 12 (a8a7777, gate-13 silent-failure CRITICAL-1)**:
  **deeper defect class** — scope-lifetime mismatch. The
  recursive helper recursing to Block.final_expr would consult
  `_modal_origin_provenance` AFTER `_check_block` had popped
  the inner scope, so inner-let-bound Names returned None.
  Fix: `_block_modal_kind: dict[int, Optional[ModalKind]]`
  cache populated at block-exit (while scope live). Consulted
  by A.Block arm + `_modal_origin_of_expr_block_tail`.
  7 reproducer variants all FIRE post-fix.
- **Inc 12 polish (32df46e, gate-14 type-design HIGH-1 theoretical)**:
  defensive cache write in `_check_expr_in_block_scope` (HIGH
  was theoretical — all 4 audit-recommended reproducers verified
  FIRE; defensive belt-and-suspenders applied for future safety).
- **Gate-14 CLEAN on all 3 lanes**: silent-failure 0 HIGH+,
  type-design 0 HIGH+ (HIGH was theoretical), code-review
  0 HIGH+ (2 IMPORTANT doc-drift fixed below). **1 of 3 fresh
  consecutive clean gates achieved.**

Stage 52 surface now covers **22+ launder paths** via **9
wrapper-AST kinds** in the recursive helper (Name, Call, Block,
UnsafeBlock, Cast, Unary, Binary, If, Match). Plus the scope-
cache fix in Inc 12 ensures inner-let-bound Names survive
the block-exit pop boundary.

Tests: 113 passing (was 100 at Inc 7 closure; +13 across Inc 8/10/11/12 pins).

## Stage 52 closure path forward

Per the 3-clean-gate protocol:
- Gate-14 CLEAN ✓ (1 of 3)
- Gate-15 + gate-16: need both CLEAN to declare STAGE 52 CLOSED.

If gate-15 or gate-16 find new defects, the cascade resumes;
otherwise this is the final stretch.
