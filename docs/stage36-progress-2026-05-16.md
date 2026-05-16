# Stage 36 Progress - 2026-05-16

## Stage Goal

Stage 36 is the **Strategic AGI Features** stage. First deliverable
(user-approved 2026-05-16): **provenance-typed neuro-symbolic
primitives** — make `Logic<T>` and `D<Logic<T>>` real, not just
type-level scaffolding.

Beginner meaning: Helix should be able to tell you not just WHAT a
value is, but WHERE it came from (which facts justified it) and
WHETHER a gradient can flow through it. This is what makes Helix
strategically different from JAX, Mojo, and Triton — those stacks see
numbers, Helix sees numbers + their evidence trail + their gradient
trail simultaneously.

## Predecessor State

- Stage 24 (cycle ~10 months ago) shipped the type-level scaffolding:
  `TyLogic<T>` and `TyDiff<T>` in `helixc/frontend/typecheck.py:186-228`.
- 13 tests in `helixc/tests/test_provenance.py` pin the parsing +
  typecheck behavior of `Logic<T>` and `D<Logic<T>>` in signatures.
- The `provenance: Optional[str]` field on `TyLogic` exists but is
  always `None` — no current code populates it.
- No runtime representation, no constructors, no eliminators, no
  logical operations.

## Increment 0 - Open Stage 36 (Convention Declaration)

Stage 36 opens here. Conventions:

1. **Audit campaign convention**: combined-audit-and-fix discipline
   (the restart-62-65 pattern that worked through Stage 35 closure).
   No multi-restart clean-gate counter at the start; each MVP
   increment closes when its tests pass + a single combined audit
   returns clean.
2. **Increment numbering**: continues at Stage 36 Increment 1 (Stage
   35's count stopped at 84). Stage 36 uses its own counter starting
   at 1.
3. **Progress ledger**: this file
   (`docs/stage36-progress-2026-05-16.md`).
4. **Pre-flight commitment**: every Stage 36 increment must pass
   `python scripts/stage33_selfhost_gate.py` before commit. The
   self-host cascade is the load-bearing acceptance gate for any
   stdlib or typecheck change.

## Increment 1 - Logic<T> Constructor + Eliminator (in progress)

Goal: make `Logic<T>` a usable type in user code, not just an
annotation. Define `fact(value, source_id)` and `unwrap_logic(l)` as
stdlib functions so user code can construct and destructure `Logic`
values.

Scope:

- Stdlib functions:
  - `fn fact[T](value: T, source_id: i32) -> Logic<T>` — wraps a value
    with a provenance tag (an i32 identifier into a user-managed
    source table).
  - `fn unwrap_logic[T](l: Logic<T>) -> T` — extracts the value,
    discarding provenance.
  - `fn logic_source[T](l: Logic<T>) -> i32` — returns the provenance
    tag (so user code can do source attribution).
- Runtime representation: a `Logic<T>` is a struct
  `{ value: T, source: i32 }`. Provenance lives at runtime in the
  i32 source field.
- Phase-0 limitation (carried from Stage 24): provenance is a single
  i32 tag, not a lattice. The lattice/semiring upgrade is reserved for
  Increment 3.

Implementation path:

1. Add stdlib `.hx` file for the three functions (no Python
   typecheck change required — the existing TyLogic resolver in
   typecheck.py already maps `Logic<T>` to `TyLogic<T>`).
2. Add tests pinning the round-trip
   `unwrap_logic(fact(v, src)) == v`.
3. Add a test pinning `logic_source(fact(v, src)) == src`.
4. Run the full test suite + self-host gate before commit.
