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
