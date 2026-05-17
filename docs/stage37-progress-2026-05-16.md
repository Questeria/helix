# Stage 37 Progress - 2026-05-16

## Stage Goal

Stage 37 is the **Cognitive Substrate Capability Push**. Per
`docs/ROADMAP.md`, Stage 37's three feature families are:

1. **Continuous execution** — persistent runtime, infinite loops,
   checkpoint/restore.
2. **Tiered memory** — Working / Episodic / Semantic / Procedural
   memory types with consolidation, decay, and retrieval semantics.
3. **Theorem-prover integration** — Z3 bridge for refinement-type
   discharge.

**First deliverable (chosen by autonomous-loop steering with full
user autonomy mandate, 2026-05-16)**: **Tiered memory**, in direct
parallel to Stage 36's "provenance-typed primitives" playbook.

## Predecessor State

- Stage 24 (cycle ~10 months ago) shipped the type-level scaffolding:
  `TyMemTier` in `helixc/frontend/typecheck.py:232`.
- The `WorkingMem<T>` / `EpisodicMem<T>` / `SemanticMem<T>` /
  `ProceduralMem<T>` type wrappers parse and resolve in the
  TyGeneric arm of `_resolve_type` (typecheck.py:1074-1083).
- `consolidate(EpisodicMem<T>) -> SemanticMem<T>` and
  `recall(SemanticMem<T>) -> WorkingMem<T>` exist as
  typecheck-recognized builtins (typecheck.py:3087-3105) but are
  NOT lowered — any program using them fails at IR with "unknown
  function 'consolidate'" (matching the pre-Stage-36 status of
  `attach`/`detach` and `prove`/`unwrap_logic`).
- No runtime representation, no constructors, no eliminators, no
  cross-tier transitions at IR level.

## Increment 0 - Open Stage 37 (Convention Declaration)

Stage 37 opens here. Conventions:

1. **Audit campaign convention**: combined audit-and-fix per
   increment (the Stage 36 playbook). 3 consecutive clean audit
   gates close the stage (Stage 35 + Stage 36 precedent).
2. **Increment numbering**: starts at Stage 37 Inc 1.
3. **Progress ledger**: this file (`docs/stage37-progress-2026-05-16.md`).
4. **Pre-flight commitment**: every Stage 37 increment must pass
   `python scripts/stage33_selfhost_gate.py` before commit.
5. **Strategic discipline**: same as Stage 36 — production-quality
   primitives, mathematical verification, audit-hardened, dogfood-
   demonstrated.

## Increment 1 - WorkingMem<T> Constructor + Eliminator (planned)

Goal: make `WorkingMem<T>` a usable type in user code, not just an
annotation. Mirror the Stage 36 Inc 1 pattern for `Logic<T>`.

Scope:
- `into_working(value: T) -> WorkingMem<T>` builtin — wraps a value
  with the working-memory tier.
- `unwrap_working(m: WorkingMem<T>) -> T` builtin — strips the
  wrapper.
- Both lower to identity at IR (Phase-0: zero runtime overhead,
  tier lives purely in the type system).
- Phase-0 limitation: no actual memory-tier semantics yet (decay /
  consolidation / retrieval). The wrapper exists so user code can
  express "this value belongs in working memory" intent.

## Increment 2+ — Planned Sequence

- **Inc 2**: `into_episodic` + `unwrap_episodic` (matches Inc 1 for
  episodic tier).
- **Inc 3**: `into_semantic` + `unwrap_semantic`; wire up the
  existing `consolidate(Episodic) -> Semantic` and
  `recall(Semantic) -> Working` builtins at IR level (identity
  lowering, same as Stage 36's attach/detach).
- **Inc 4**: `into_procedural` + `unwrap_procedural`. Procedural
  memory is the highest-consolidation tier (learned skills,
  motor sequences) — at Phase-0 it's just a typecheck-level
  annotation.
- **Inc 5**: Tier-id arena side-table (parallels Stage 36 Inc 5
  for derivation arena). `tag_memory(value, tier_id)` returns a
  handle; `read_tier(handle)` recovers the tier-id. Lets user code
  observe runtime cross-tier transitions.
- **Inc 6**: Dogfood — `dogfood_10_memory_tiers.hx` showing a
  working-memory item promoted to episodic via tag + consolidated
  to semantic.

The audit-fix sweep (Stage 36 Inc 9 + Inc 11 + Inc 12 pattern)
will run as needed once Inc 1-5 ship.

## Strategic Significance

Stage 37 closes the "memory architecture" gap that Phase 2 of the
ROADMAP depends on. Without tiered memory, the language can express
"this value matters now" but can't express "this value should be
remembered" or "this value is procedural skill" — those distinctions
matter for AGI workloads where memory tiers control consolidation
timing, decay, retrieval, and learning-rate-by-tier.

## Increment 4 — STAGE 37 CLOSURE (3/3 clean gates) (2026-05-16)

Per the user direction "Do not forget 3 clean audits at the end of
each stage before moving on" + "you have permission to do whatever
you feel is best and move on to any next stages until everything is
finished" (full autonomy), Stage 37 closes via the same 3-clean-gate
convention used by Stage 35 (restart 65) and Stage 36 (Inc 16).

### Closure timeline

| Gate | Result | Findings | Fix-sweep commit |
|------|--------|----------|------------------|
| 1 (initial) | NOT CLEAN | 1 LOW (S37-CLEAN1-001: tier builtins absent from AD_KNOWN_PURE_CALLS) | ab54524 (10 tier names added) |
| 1 (re-audit) | CLEAN | 0 new findings; prior LOW verified closed | — |
| 2 | CLEAN | 0 findings | — |
| 3 (final) | CLEAN | 0 findings; cosmetic dogfood line-count comment noted as <80 confidence threshold | — |

**Counter advances**: 0/3 → 1/3 (after gate-1 fix-sweep) → 2/3 → 3/3.

### Stage 37 final scorecard

- **Increments shipped**: Inc 0 (convention) + Inc 1 (constructors +
  eliminators + cross-tier wiring) + Inc 2 (lifecycle dogfood) +
  Inc 3 (cross-tier mismatch coverage) + Inc 4 (closure).
- **Audit cycles**: 1 (closure gate sequence) — Stage 37 surface was
  small (~50 lines of typecheck + lowering) so the per-increment
  audit overhead Stage 36 required was unnecessary; closure gates
  served as the only formal audit pass.
- **Audit findings closed**: 1/1 (1 LOW).
- **Tests**: 23 in `helixc/tests/test_stage37_memory.py` + 1
  dogfood-runtime test in `test_reflection.py`.
- **Self-host gate**: PASS at every Stage 37 commit.
- **Total Stage 37 surface area**: 8 new typecheck-recognized
  builtins, 0 new IR opcodes (all 10 tier builtins lower as
  identity — matches Stage 36's Logic<T> attach/detach pattern),
  0 new stdlib files, 1 new dogfood program (memory-tier lifecycle).

### Strategic significance

Stage 37's first deliverable was the **AGI-shaped memory
architecture**: Working / Episodic / Semantic / Procedural memory
types with cross-tier consolidation and recall transitions. This
maps directly to the human-memory model used in cognitive science
(working memory = active focus, episodic = autobiographical events,
semantic = abstract knowledge, procedural = motor/skill memory).

A Helix program can now express:
- "this value belongs in working memory" (into_working)
- "this value should be remembered as an event" (into_episodic)
- "this event should consolidate into long-term knowledge"
  (consolidate)
- "I need to recall this knowledge for active use" (recall)
- All cross-tier transitions enforced by the typechecker (12
  wrong-pair tests prove the boundary checks)

The Phase-0 implementation lowers tiers as identity (zero runtime
overhead). Phase-1+ work will add tier-id arena side-tables for
runtime tracking, consolidation timing, decay semantics, and
retrieval-by-tier.

**STAGE 37 IS CLOSED.** Stage 37 feature families NOT yet shipped
(carryforward to Inc 5+ or new stages): continuous execution,
theorem-prover integration, tier-id arena side-table, real
consolidation/decay timing semantics.

The next stage opens next. Per ROADMAP Phase 2 (Stages 38-46
written IN HELIX), Stage 38 is "Spatial types + frames" — the
first stage to be implemented in Helix-itself rather than Python.
