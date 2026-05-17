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

Fix: rename the user-visible diagnostic strings ("aggregate
return type" → "composite return type", "aggregate argument" →
"composite argument", "operator for the aggregate type" →
"operator for the composite type"). Keep the IR-internal
`_aggregate_*` identifier names; only the user-facing strings
need disambiguation.

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

## Increment 2 - Stage 43 Closure (3/3 clean gates)

Same protocol as Stage 35/36/37/38/39/40/41/42.
