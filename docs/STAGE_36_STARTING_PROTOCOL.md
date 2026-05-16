# Stage 36 Starting Protocol

**Status**: Stage 36 has *not yet started*. Stage 35 CLOSED 2026-05-16
at restart 65 (3/3 clean gates). Stage 36 prep (Increment 84
post-closure audit + this protocol doc) is the on-ramp.

**Source**: This document promotes the Stage 36 starting protocol out
of `docs/stage35-progress-2026-05-15.md` Increment 82 (lines 4989-5001)
to a discoverable location, per Stage 36 prep backlog item.

---

## 1 — Preconditions before opening Stage 36

1. `git log` HEAD must include Increment 82 (`b8cafe7`) and the
   Increment 83 catch-up (`8177350`).
2. Stage 35 closure status (3/3 clean gates) must remain in effect —
   no Stage 35 audit-cycle restart is permitted under Stage 36 rules.
3. Stage 36 prep backlog (HANDOFF_FOR_CLAUDE.md) should be cleared or
   formally deferred. As of 2026-05-16 the live backlog is:
   - DONE — delete `_probe_stage29_*.py` (98d8e1f).
   - DONE — refresh "post-Stage-30" doc refs (98d8e1f).
   - DONE — promote Stage 36 starting protocol out of progress ledger
     (this file).
   - DONE — reconcile three-docs disagreement on Stage 36 first
     deliverable (see §3 below; HELIX_V1_FINAL_FEATURES.md Status
     section rewritten to reflect reconciled view).
   - DEFERRED — refresh stale `pytest-codegen-shard-1-of-2 / 1-of-4`
     timing JSONs in `.stage31-logs/` (unused by current 1-of-8
     sharding; cleanup-only).
   - DEFERRED — split `helixc/tests/test_codegen.py` (21,504 lines).
     Touches every Stage 1-35 codegen test; deserves its own
     scoped commit campaign, not a Stage 36 prep sweep.

## 2 — Campaign convention shift

Stage 35's audit-cleanup convention was "3 consecutive clean fresh
audit gates close the stage." That convention is **now closed** with
Stage 35.

Stage 36 declares its own protocol when it opens. Until then:

- No clean-gate counter is active.
- No audit campaign is in flight.
- The combined audit-and-fix anti-abbreviation discipline established
  at restart 62 (and validated across restarts 62-65) **remains the
  recommended default** for any audit dispatched during Stage 36.

## 3 — Reconciled view of Stage 36 deliverables

Three docs historically named different "first" candidates for Stage 36.
The disagreement is reconciled here:

### Candidates already shipped (NOT Stage 36 work)

| Candidate | Where it shipped | Evidence |
|---|---|---|
| Refinement types | Stage 31 (named alias erase) + Stage 34 (proof/refinement expansion) | `helixc/frontend/typecheck.py:47` Stage 31 named refinement; `docs/stage34-progress-2026-05-14.md` proof carries for refinement |
| Capability tokens / effect system | Stages 33-34 (effect-system extensions, capability checks) | Effect-system enforcement in IR; `docs/ROADMAP.md` Phase B Stage 33 + Stage 34 |

Their appearance as "Stage 36 candidates" came only from the obsolete
Phase B/C stage labels in `docs/HELIX_FINAL_PRODUCT_RESEARCH.md`,
which that document itself flags as obsolete in its stage-numbering
note (lines 833-836). The authoritative roadmap is `docs/ROADMAP.md`.

### Candidates still on the table for Stage 36

Per `docs/ROADMAP.md` Stage 36 section (lines 225-241), Stage 36 is
the Strategic AGI Features stage with four feature families:

1. **Provenance-typed neuro-symbolic primitives** (`D<Logic<T>>`).
   ROADMAP Tier 3 #10 strategic differentiator. Differentiable
   relational data with provenance semirings (Scallop / Lobster
   pattern). Estimated 4-6 weeks for MVP.
2. **Trace-based introspection** for `quote`/`modify`. ROADMAP Tier 3
   #11. Estimated 3-4 weeks.
3. **Verifier-gated self-modification with stronger proof objects**.
   ROADMAP Tier 3 #12 (Lean-4-style proof-carrying terms). Months,
   but bounded.
4. **Memory / knowledge types**. `HELIX_V1_FINAL_FEATURES.md` §2.6
   "Tiered memory" (`storage hot/warm/cold`, `remember`/`recall`).
   Estimated ~3000 LOC + runtime support.

### Recommended first deliverable (pending user confirmation)

**Provenance-typed neuro-symbolic primitives** is the recommended
first deliverable, on the following grounds:

- ROADMAP Tier 3 #10 explicitly identifies it as the strategic
  differentiator against tensor-only stacks (Mojo/JAX/Triton). It is
  the *reason* Stage 36 is called "Strategic AGI Features."
- It composes on top of existing infrastructure: typed refinements
  (Stage 31/34), reverse-mode AD (Stage 35), and the effect system —
  all already shipped.
- It can land an MVP without requiring runtime support (unlike tiered
  memory, which is ~3000 LOC and a new tier-managed allocator).
- It unblocks downstream Layer-1 work (Stage 40 `Knowledge<T> /
  Unknown<T> / Belief<T>`, Stage 42 world-modeling primitives).

**This recommendation is not autonomous-approved.** Stage 36 first
deliverable is a major architectural decision and is gated on
explicit user approval, by analogy with the documented Stage 29 hard
gate.

## 4 — When Stage 36 actually opens

User approval should specify:

1. Which first deliverable (the recommendation above, or another of
   the four candidates).
2. Which audit-campaign convention Stage 36 will run under
   (3-clean-gate, single-pass, or some new convention).
3. The Stage 36 progress ledger filename (suggested
   `docs/stage36-progress-YYYY-MM-DD.md`).

Until that approval lands, work on Stage 35 closure surfaces remains
the only permitted main-line activity, and Stage 36 prep backlog
items in §1 may continue.

---

*This protocol is a Stage 36 prep artifact. It does not itself
constitute opening Stage 36.*
