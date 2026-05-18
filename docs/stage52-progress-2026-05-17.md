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

Gate-2 audit sweep launched (3 lanes). Expecting cascading-
defect rhythm to surface 1-2 HIGHs in the new Inc 2+3 code.

## Verification (post-Inc-3)

- 57/57 Stage 40 modal tests pass (1 polarity flip from Inc 1
  + the rest unchanged)
- Self-host cascade re-verified live: PASS G2..G11 byte-
  identical sha=`a6f1ee44`, smoke 4/4 PASS
- dogfood_16 + dogfood_17 still exit 42
- Original Inc 1 let-bypass: caught
- HIGH-3 while-loop Assign: caught (Inc 2)
- HIGH-1 match-arm parallel: caught (Inc 3)
- F1e false-positive (inner-let shadow leak): closed (Inc 2)
- Negative test (untainted match): correctly silent

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
