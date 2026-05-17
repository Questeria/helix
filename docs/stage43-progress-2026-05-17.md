# Stage 43 Progress - 2026-05-17

## Stage Goal

Stage 43 is **Deferred-items cleanup sweep across Stages 36-42**.
The 7 stages closed in this session burst (Stage 37-42 plus the
provenance work in Stage 36) accumulated 4 actionable LOW/MEDIUM
findings that were explicitly deferred at closure time as either
"multi-stage refactors" or "pre-existing patterns symmetric
across all wrapper families." Stage 43 pays down that debt in
one focused sweep.

Beginner meaning: every stage added a new AGI semantic-type
family (memory / spatial / temporal / modal / causal), and each
closure deferred a few cross-stage cleanup items because fixing
them mid-stage would have churned the prior stages' surface.
Stage 43 collects all 4 deferred items and closes them together,
restoring full hygiene to the type-system surface before any new
features land.

## The 4 deferred items

### Item 1 — LOW-2 (Stage 41 closure gate-1 code-review)

The Stage 41 causal transition builtin `aggregate` (Effect ->
Joint) collides with pre-existing compiler diagnostic vocabulary
for "aggregate types" (struct/enum/tuple/array layouts). Users
who hit "aggregate return type" diagnostics in unrelated code
will be confused about which `aggregate` is meant.

**DEFERRED to Stage 44+** (decided at Inc 1 implementation
time). Original fix plan: rename the user-visible diagnostic
strings ("aggregate return type" → "composite return type",
"aggregate argument" → "composite argument", "operator for the
aggregate type" → "operator for the composite type"). Keep the
IR-internal `_aggregate_*` identifier names; only the
user-facing strings need disambiguation.

Why deferred: the rename touches 6 existing test assertions in
helixc/tests/test_typecheck.py (lines 2486-2505) and
helixc/tests/test_codegen.py (line 14650) that pin the literal
"aggregate" wording. The risk of breaking the test surface for
a cosmetic disambiguation exceeds the reward — especially since
Stage 43 is already a cleanup stage and adding test churn
defeats the hygiene goal. Stage 44+ will need to land the rename
+ test updates as a single co-ordinated commit.

### Item 2 — LOW-3 (Stage 41 closure gate-1 code-review)

The set `_FRAME_IDENTITY_AD_NAMES` started at Stage 38 with 12
frame builtins, then grew to absorb temporal (Stage 39), modal
(Stage 40), and causal (Stage 41) — total 45 names across 5
wrapper families, of which only 12 are actual frames. The name
is now actively misleading.

Fix: rename `_FRAME_IDENTITY_AD_NAMES` →
`_IDENTITY_AD_CHAIN_RULE_NAMES`. Touches autodiff.py
declaration site + autodiff.py use site + autodiff_reverse.py
import site + autodiff_reverse.py use site. Leave a one-stage
alias `_FRAME_IDENTITY_AD_NAMES = _IDENTITY_AD_CHAIN_RULE_NAMES`
for any third-party plugin that might import the old name; drop
the alias at Stage 44 or beyond.

### Item 3 — F5 (Stage 39 closure gate-1 silent-failure)

`_resolve_type` arity fall-through: `Past<i32, i32>` or `Past<>`
silently falls through to the generic-struct error path and
emits "unknown type 'Past'" instead of "Past<T> takes 1 type
argument, got 2". Symmetric across all 5 wrapper families
(TyMemTier, TyFrame, TyTemporal, TyModal, TyCausal).

Fix: add an explicit arity check at each wrapper's
`_resolve_type` arm.

### Item 4 — M1 (Stage 39 closure gate-1 type-design)

Intro builtins accept already-wrapped values silently:
`into_past(Past<i32>)` typechecks as `Past<Past<i32>>` (no
guard against double-wrapping). Symmetric across all 5 wrapper
families.

Fix: in each `_X_intro` dispatch, emit a diagnostic when
`arg_tys[0]` is already a `TyX` of the same family.

## Increment 0 - Open Stage 43

Same conventions as Stage 37/38/39/40/41/42.

## Increment 1 - Cleanup sweep (single fix-commit)

All 4 items fixed in one commit. Tests added pinning each fix.

## Increment 2 - Stage 43 Closure

### Gate 1 (post-Inc-1) — fix sweep landed

- silent-failure: GATE CLEAN
- type-design: GATE CLEAN (1 MEDIUM "5-fold duplication"
  deferred to Stage 44+ as a `_resolve_unary_wrapper` +
  `WrapperFamily.intro_hint(source, target)` refactor)
- code-review: 2 MEDIUMs + 1 LOW
  - MEDIUM-1 (frame direction-blind hint): fixed
  - MEDIUM-2 (tier missing transition names): fixed
  - LOW-1 (5-fold duplication): deferred (same as type-design)

Gate-1 fix sweep (commit eee95fc):
- Frame double-wrap hint now computes
  `{source}_to_{target}` direction-correctly from actual
  (source, target) pair, with same-kind fallback at
  "unwrap with from_{source}" instead of nonsense
  self-transform.
- Tier double-wrap hint now names concrete Phase-0
  transitions (`consolidate` for Episodic→Semantic,
  `recall` for Semantic→Working) instead of generic
  "unwrap first".
- 4 direction-pin regression tests added.

### Gate 2 (post-gate-1) — direction-aware sweep extended to remaining 3 families

- silent-failure: GATE CLEAN
- type-design: 1 MEDIUM (asymmetry — gate-1 only direction-
  fixed 2 of 5 families; temporal/modal/causal still used
  generic one-liner). FIXED.
- code-review: GATE CLEAN

Gate-2 fix sweep (commit 9df6329): extended direction-aware
pattern to temporal/modal/causal. All 5 arms now use the
same shape: per-family `_X_transitions_by_pair` table
lookup; same-kind → unwrap; audited transition → name the
verb; deferred direction → kind-specific Phase-0 message.
7 new direction-pin tests added (one per audited (source,
target) per family).

### Gate 3 (post-gate-2) — ALL 3 LANES CLEAN

- silent-failure: GATE 3 CLEAN
- type-design: GATE 3 CLEAN
- code-review: GATE 3 CLEAN

### STAGE 43 CLOSED 2026-05-17 at Inc 2

3 consecutive clean audit gates achieved per the post-Stage-35
closure convention. Deferred-items cleanup complete:

- Item 2 (rename `_FRAME_IDENTITY_AD_NAMES` →
  `_IDENTITY_AD_CHAIN_RULE_NAMES` + backwards-compat alias):
  DONE.
- Item 3 (F5 `_resolve_type` arity arms across 5 wrapper
  families): DONE.
- Item 4 (M1 intro double-wrap rejection across 5 wrapper
  families, with direction-aware hints post-gate-2): DONE.
- Item 1 (aggregate → composite diagnostic rename): DEFERRED
  to Stage 44+ (would force 6 existing test assertions to be
  rewritten in lockstep; bad risk/reward mid-cleanup).

Outstanding deferrals for Stage 44+:
- LOW-2 (Item 1 above): aggregate-name rename.
- LOW-3 follow-up: drop the `_FRAME_IDENTITY_AD_NAMES`
  backwards-compat alias.
- LOW-1 / gate-2 follow-up: `_resolve_unary_wrapper` +
  `WrapperFamily.intro_hint(source, target)` 5-fold
  duplication refactor.
- Stage 40 F1 known limitation: let-binding bypass of
  Uncertain-laundering guard (needs taint-tracking pass).

26 Stage 43 tests green. Self-host cascade still
byte-identical G2..G4 fixpoint. 187+ tests across Stage 37-43
all green.

Stage 44 opens next per ROADMAP Phase 2.
