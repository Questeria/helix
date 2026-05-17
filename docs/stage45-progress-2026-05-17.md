# Stage 45 Progress - 2026-05-17

## Stage Goal

Stage 45 is a **ROADMAP status drift refresh + Stage 46+
sequencing**. After 10 stages closed in this session burst
(Stages 35-44), the ROADMAP source-of-truth document has
accumulated stale claims — features ROADMAP says are "to
do" that are actually done, features claimed as "in
progress" without current status, and ordering implied by
the historical Tier numbering that's no longer the best
sequence given what's shipped.

Stage 45 is a small hygiene stage: refresh ROADMAP, mark
done items as ✅ DONE, surface real outstanding work, name
the Stage 46+ sequence with concrete deliverables. The
refresh is itself the unit of work — no compiler change.

Beginner meaning: we cleaned up a lot in 10 stages; let's
update the planning doc so we know what to do next without
re-reading every per-stage progress doc to figure it out.

## Increment 0 - Open Stage 45 (Convention Declaration)

Slim stage: docs-only. No compiler changes; no audits
beyond a single-pass review that confirms the refresh is
accurate. No self-host gate required (no codegen surface
touched).

## Increment 1 - ROADMAP status drift refresh

Walk each Tier-1/2/3/4 item, cross-check against per-stage
progress docs from Stages 35-44, mark DONE/IN-PROGRESS/
DEFERRED with brief evidence.

Known status drift entering Stage 45:

- **Tier 1 #2 AD across user-defined function calls** —
  partially shipped; need to document current scope vs
  remaining.
- **Tier 1 #3 Multi-output reverse-mode AD** — Python-side
  ALREADY done at `differentiate_reverse(expr, param_names)`
  via per-param bucket dict; one walk produces all
  gradients. Bootstrap-side parser.hx:5402 still does
  N-walks (documented TODO). The ROADMAP entry needs to
  reflect this Python-done / bootstrap-pending split.
- **Tier 1 #5 Stack-passed overflow args** — ✅ DONE at
  Stage 44 (commit 3f3d25a). Already marked.
- **Tier 3 #10 Provenance-typed neuro-symbolic primitives**
  — ✅ DONE at Stage 36. Need to confirm marked.
- **Memory tier / spatial frame / temporal / modal / causal
  types** — these are Stages 37-41 deliverables that were
  NOT in the original ROADMAP. They map roughly to "memory
  /knowledge types as the language grows toward the broader
  vision" from Stage 36 plans. Need a new top-level
  ROADMAP entry summarizing the AGI semantic-type quintet.
- **Quintet cohesion (Stage 42)** + **deferred-items
  cleanup sweep (Stage 43)** — also need to be reflected
  in the ROADMAP's "current state" header / Phase 2
  narrative.

## Increment 2 - Stage 46+ concrete sequencing

Name the next 3-5 stages with concrete deliverables, sized
to single-stage scope. Avoid the "1 week" / "2-3 weeks"
ROADMAP estimates that don't decompose into stages.

Candidate Stage 46-50 picks (to be sequenced in Inc 2):

- Tier 1 #2 follow-through: extend AD across user-defined
  function calls to cover the currently-failing-closed
  opaque call shapes. Likely 1-2 stages.
- Tier 4 #14 Result<T,E> + `?` operator: parser change +
  TyResult wrapper + early-return lowering. Decomposes into
  3-4 stages (parser, typecheck, lower, dogfood).
- Stage 40 F1 let-binding bypass / Uncertain-laundering
  taint tracking: would need a new analysis pass that
  propagates Uncertain-origin through let-bindings and
  function calls. Likely 1-2 stages.
- Bootstrap `grad_rev_all` N-walk → single-walk: bootstrap
  parser.hx perf fix per ROADMAP Tier 1 #3 (bootstrap
  side). 1 stage.
- ROADMAP Tier 4 #15 Pattern matching enhancements: guards,
  or-patterns, nested destructuring. Required for richer
  AST-walking inside quote/splice. 2-3 stages.

## Increment 3 - Stage 45 Closure (single-pass review)

Slim stage — no compiler change means no audit-lane fan-out.
Single-pass review confirms (a) the ROADMAP refresh is
accurate vs the per-stage docs, (b) the Stage 46+ sequence
matches actual outstanding work.

### STAGE 45 CLOSED 2026-05-17 at Inc 3

Docs-only stage closure. ROADMAP refresh shipped:
- Tier 1 #3 multi-output reverse-mode AD marked ✅ DONE
  Python-side; bootstrap-side N-walk gap documented as
  Stage 46 work.
- Tier 1 #5 stack-passed overflow args marked ✅ DONE
  Stage 44.
- "Current state" header rewritten to summarize the
  11-stage burst with per-stage callouts.
- "Next-stage sequencing" section names concrete Stage
  46-50 deliverables with single-stage scope.

No codegen surface touched → self-host cascade trivially
preserved. No tests added/changed.

Stage 46 opens next: bootstrap `grad_rev_all` N-walk →
single-walk port. Closes the bootstrap side of Tier 1 #3.
