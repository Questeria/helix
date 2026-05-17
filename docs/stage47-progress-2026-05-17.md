# Stage 47 Progress - 2026-05-17

## Stage Goal

Stage 47 is a **small consolidation stage** — drop the Stage 43
deferred backwards-compat alias `_FRAME_IDENTITY_AD_NAMES`, then
refresh ROADMAP to mark Tier 4 #14 Inc 1 done and sequence the
remaining Stage 47-50 picks.

Stage 43 LOW-3 deferred the alias removal "drop at Stage 44 or
beyond" to give any external importer one stage of grace period.
Three stages have elapsed since (44, 45, 46) — well past the
documented deprecation window. Time to drop it.

Beginner meaning: when we renamed a set of AD-related names at
Stage 43, we left both the old and new names working for one
stage so any external code wouldn't break. That grace period is
over; we're cleaning up the old name now.

## Increment 0 - Open Stage 47 (Convention Declaration)

Slim stage: small Python cleanup + ROADMAP refresh. No new
features; no audit-lane fan-out beyond a single-pass review.

## Increment 1 - Drop `_FRAME_IDENTITY_AD_NAMES` alias

`helixc/frontend/autodiff.py` has both:
- `_IDENTITY_AD_CHAIN_RULE_NAMES = frozenset({...})` (canonical
  since Stage 43)
- `_FRAME_IDENTITY_AD_NAMES = _IDENTITY_AD_CHAIN_RULE_NAMES`
  (alias added at Stage 43 LOW-3, "drop at Stage 44+")

Verify no internal use sites still import the old name (Stage 43
swept them to the new name); drop the alias line; rerun tests.

## Increment 2 - ROADMAP refresh for Stage 46

Update the "Current state" header with Stage 46 → CLOSED. Update
Tier 4 #14 ROADMAP entry:
- Before: "**Stack-passed overflow args.** ✅ DONE 2026-05-17
  (Stage 44)" + #14 description
- After: mark Tier 4 #14 partially done (Inc 1 Result<T,E>
  typecheck-side shipped at Stage 46; Inc 2 `?` operator + Inc 3
  runtime tag remain).

Re-sequence Stage 48-50 picks now that Stage 46 is done.

## Increment 3 - Stage 47 Closure (single-pass review)

Slim stage — no compiler change beyond a 2-line deletion. No
audit-lane fan-out needed; single-pass review confirms:
(a) the alias drop doesn't break any importer,
(b) the ROADMAP refresh is accurate vs Stage 46 closure.
