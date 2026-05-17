# Stage 43 Inc 2 Gate-1 — Type-Design Audit
Date: 2026-05-17
Scope: git diff 7699f00..e474c17 (Stage 43 Inc 1: deferred-items cleanup sweep)
HEAD: d4a1e8b0da74b6c81f37ed61e8d08bc88d58616b

## Verdict
GATE CLEAN

No HIGH or MEDIUM finding at confidence >= 80. Two LOW findings recorded
at confidence 75-85 for hint-quality polish (non-blocking; recommended
for Stage 44 alongside item 1).

## Findings (HIGH / MEDIUM / LOW, with confidence 0-100)

### LOW-1 — Memory M1 hint omits family-specific transitions (conf 85)

`typecheck.py:3375-3384` (the `into_X` for TyMemTier arm). When
`into_working(WorkingMem<i32>)` is rejected, the hint reads:

> "intro builtins are not idempotent — double-wrapping changes the
> type's semantic meaning. If you mean to re-tag, unwrap first."

This is the ONLY one of the 5 families whose M1 hint omits a
pointer to its family-specific transition builtins. Compare:

- Frame:    "use a cross-frame transform (e.g. `world_to_robot`) to change frames, or unwrap first"
- Temporal: "use a temporal transition (`to_past`, `forecast`, `recall_past`, `actualize`) to change kind, or unwrap first"
- Modal:    "use a modal transition (`confirm`, `act_on`) to change epistemic kind, or unwrap first"
- Causal:   "use a causal transition (`propagate`, `aggregate`, `isolate`) to change kind, or unwrap first"
- Memory:   "If you mean to re-tag, unwrap first." (no transition mentioned)

Memory DOES have cross-tier transitions: `consolidate` (Episodic ->
Semantic, typecheck.py:4027) and `recall` (Semantic -> Working,
typecheck.py:4037). The Memory M1 hint should mention them for
symmetry — a contributor hitting this diagnostic with
`into_episodic(SemanticMem<T>)` deserves a pointer at `consolidate`
the same way a `into_world(RobotFrame<T>)` user gets `world_to_robot`.

**Severity**: LOW (functional behavior correct; hint pedagogy lopsided).
**Recommended fix**: Append " — use a tier transition (`consolidate`,
`recall`) to change tier, or unwrap first" to the Memory hint to
match the Frame/Temporal/Modal/Causal phrasing. Single-line change at
typecheck.py:3378-3381.

### LOW-2 — Temporal/Causal M1 hints list transitions inapplicable to all variants (conf 75)

`typecheck.py:3528-3531` (temporal) and `typecheck.py:3915-3918`
(causal). The hints list ALL transition builtins for the family, but
some variants have NO outgoing transitions:

- `into_eternal(Eternal<T>)` -> hint lists `to_past, forecast,
  recall_past, actualize`. NONE of these accept an `Eternal<T>`
  source (`_temporal_transitions` at typecheck.py:3566 covers only
  past/present/future). Eternal is by-design timeless and has no
  transitions out.
- `into_independent(Independent<T>)` -> hint lists `propagate,
  aggregate, isolate`. None accept `Independent<T>` as source; the
  Stage 41 `_causal_upgrade_hint` entries at typecheck.py:3882-3898
  explicitly frame Independent -> X as "experimentally falsified" /
  "category mistake".

This is the same lesson Stage 40 gate-3 LOW logged for the
`_modal_upgrade_hint` fallback: generic "use a transition" framing
mis-suggests a future feature when the source variant has no
transitions by design. The M1 hint inherits this misframing for the
Eternal and Independent variants.

**Severity**: LOW. The user's intent on `into_X(X<T>)` is "wrap" not
"transition", so the M1 message is still correctly diagnostic. The
hint-list noise only matters if a contributor follows the hint
literally.

**Recommended fix** (defer to Stage 44 alongside item 1): conditionally
omit or rephrase the transition-list portion of the hint when the
inner TyXxx's kind has no outgoing transition. For Eternal/
Independent, prefer "unwrap first" framing or "this variant has no
transitions by design — keep the value at its current kind". Adds
~10 lines per family arm; cosmetic.

## Items audited that found NO issue at >= 70 confidence

### Item 2 — autodiff name rename + alias

- **Q1 semantic accuracy of new name**: `_IDENTITY_AD_CHAIN_RULE_NAMES`
  accurately describes what the set IS (names whose AD chain rule
  evaluates to identity because the call lowers to identity at IR).
  The alternative `_IDENTITY_AD_WRAPPER_NAMES` would also work but
  emphasizes the cause (these are wrappers) rather than the effect
  (chain rule is identity). The chosen name foregrounds the AD
  semantics, which is what the caller's `_diff_call_chain_rule` and
  `_propagate` are reasoning about. Accept as-is.
- **Q2 alias soundness**: verified at runtime
  (`_FRAME_IDENTITY_AD_NAMES is _IDENTITY_AD_CHAIN_RULE_NAMES`
  returns True). It is a name binding to the SAME frozenset object,
  not a copy. The frozenset is immutable so mutation-propagation is
  moot, but the pinning test `test_stage43_item2_old_name_still_aliased`
  enforces `is` identity, locking out future drift. Sound.
- **Q3 alias necessity**: there ARE in-tree external callers that
  still use the old name —
  `helixc/tests/test_stage40_modal.py:475,480,484,485` (3 sites) and
  `helixc/tests/test_stage41_causal.py:290,296` (2 sites). The
  one-stage retention is therefore justified (those tests pass via
  the alias; verified by running both files — 80 tests pass). The
  Stage 44 task to drop the alias must rename those 5 test sites
  in the same commit.

### Item 3 — F5 arity diagnostics across 5 families

- **Q4 symmetry**: verified at typecheck.py:1190-1276. All 5 families
  emit `f"{ty.base}<T> takes 1 type argument, got {len(ty.args)}"`
  with identical structure. Format is consistent: same wording, same
  `<T>` placeholder convention, same "got N" suffix. The 5 arms are
  copy-paste-symmetric — diagnostic-vocabulary fragmentation NOT
  detected for the F5 surface.
- **Q5 4-variant coverage**: TyTemporal (past/present/future/eternal)
  and TyModal (known/believed/goal/uncertain) and TyCausal (cause/
  effect/joint/independent) all have 4 variants. Each F5 arm gates
  on `ty.base in <map>` where the map contains all 4, so all
  variants are uniformly covered. NOT a 3-vs-4 miss; alphabetical-
  list bias not detected. Verified at runtime via direct
  `into_eternal`, `into_goal`, `into_independent` double-wrap
  probes — all 3 produce the expected M1 diagnostic.
- **Q6 user-struct vs wrapper asymmetry**: user-defined generic
  structs use `f"generic type {ty.base!r} expects {N} arg(s), got
  {M}"` (typecheck.py:1301), while wrappers use `f"{base}<T> takes
  1 type argument, got {N}"`. Vocabulary differs ("expects" vs
  "takes", "arg(s)" vs "type argument", quoted vs angle-bracketed
  base name). This is a Phase-0-wide style inconsistency that
  PRE-EXISTS Stage 43 and is symmetric across all 5 wrappers vs
  all user-structs. Not introduced by this ship; out of scope for a
  gate-1 finding.

### Item 4 — M1 intro double-wrap rejection

- **Q7 family-tag vs string-name dispatch**: verified each intro arm
  matches via `isinstance(arg_tys[0], TyXxx)` — TyMemTier, TyFrame,
  TyTemporal, TyModal, TyCausal — not via string-name comparison.
  This is the correct discipline: type aliases or generic
  substitutions that resolve to the same Ty class will trigger M1
  identically. No string-name bug detected.
- **Q8 hint quality**: see LOW-1 (Memory) and LOW-2 (Eternal/
  Independent variants). Frame, Modal, Causal hints are family-
  specific and accurate. Memory hint is non-specific (LOW-1).
  Temporal/Causal hints list transitions inapplicable to one
  variant each (LOW-2). All hints are at least
  directionally correct.
- **Q9 legitimate same-family double-wrap**: argued in the test file
  docstring at `test_stage43_item4_self_wrap_to_unwrap_still_allowed`
  — `from_X(into_X(v))` continues to typecheck (verified). The M1
  rejection only fires on `into_X(into_X(v))` and
  `into_X(InnerCall returning TyXxx)`. `Working<Working<T>>` would
  be a "fact about a working-memory fact" — semantically incoherent
  in Phase-0 (the wrappers are flat semantic tags, not nested
  meta-types). Rejection is correct.

### Q10 — M1 ordering vs F1 launder guard

Traced `into_cause` arm (typecheck.py:3900-3983):

1. arity check (line 3901)
2. M1 same-family rejection (line 3911) — fires FIRST
3. inner_is_shadowed compute (line 3938)
4. F1 cross-causal launder guard (line 3945)
5. return TyCausal (line 3982)

Confirmed: `into_cause(into_cause(7))` returns TyUnknown at step 2
before reaching step 4, so the user gets the M1 "not idempotent"
diagnostic, not the F1 "launders" diagnostic. Order is correct
across all 5 families (Memory does not have F1 launder, so trivial;
Frame/Temporal have only the simpler `_X_intro` post-arity guard,
no launder; Modal/Causal both place M1 before F1 launder per same
pattern).

## Verification steps performed

- `git diff 7699f00 e474c17 --stat` and full diff inspection.
- Direct read of `typecheck.py:1187-1276` (F5 arms for 5 families)
  and `typecheck.py:3338-3990` (M1 arms + F1 launder arms for 5
  intro families).
- Runtime confirmation that `_FRAME_IDENTITY_AD_NAMES is
  _IDENTITY_AD_CHAIN_RULE_NAMES` evaluates to True.
- Runtime probes for the 4 "edge" variants the alphabetical-bias
  pattern would miss (`into_eternal`, `into_goal`,
  `into_independent`) — all 3 correctly emit M1 diagnostic.
- Runtime probe for cross-family double-wrap (`into_past(Known<i32>)`)
  — correctly accepted with 0 errors.
- Runtime probe for `into_cause(into_cause(c))` — confirms M1 fires
  (not F1 launder, which would have produced a different message).
- `pytest helixc/tests/test_stage43_cleanup.py -x -q` — 15/15 pass.
- `pytest helixc/tests/test_stage40_modal.py
  helixc/tests/test_stage41_causal.py --tb=no -q` — 80/80 pass
  (legacy import paths through the alias work).
- Grep audit for `_FRAME_IDENTITY_AD_NAMES` callers — confirmed
  5 in-tree test-file sites still use the old name (alias is
  load-bearing for the one-stage retention period).
- Diagnostic-format comparison across the 5 F5 arms (Q4) and against
  the user-defined-struct arm at typecheck.py:1301 (Q6).
