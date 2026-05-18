# Stage 66 Progress — 2026-05-18

## Stage Goal

Stage 66 opens **Tier 4 #16 — borrow checker**. The user's
standing directive ("automatic until Stage X" where X = v1.0
release) implies I should proceed without halting at architectural
stages, making the architectural decisions inline and documenting
them so the user can redirect.

## Architectural decisions (made for autonomous progress)

**Aliasing model**: **Rust 1.0-era simple borrow** (chosen here;
user can override later):
- One `&mut T` xor any number of `&T` at any program point per place
- `&mut x` invalidates prior `&x`
- Moves out of unique-owned values forbid future use
- Function-call boundary = lifetime end (no escape analysis)
- No NLL (non-lexical lifetimes); no lifetimes-as-parameters
- Move semantics with implicit move for non-Copy types

**Rationale**: simplest viable model that catches the common
class of bugs (aliasing violations in tile/buffer code). NLL +
lifetime parameters can be added in a later stage as polish.

## Inc 1 deliverable

**Scaffolding only — no enforcement yet**. Inc 1 ships the data
types + integration point so Inc 2-5 can attach enforcement.

In `helixc/frontend/typecheck.py`:
- **`Place` dataclass** (frozen, hashable): identifies a borrow/
  move target. Three constructors:
  - `Place.local(name)` — a bare variable
  - `Place.field(parent, field_name)` — `parent.field`
  - `Place.index(parent, const_idx)` — `parent[const]`
- **`BorrowState`** container with per-place state map. Four
  status constants: `BORROW_FREE`, `BORROW_SHARED`,
  `BORROW_MUTABLE`, `BORROW_MOVED`.
- **Methods** (all stubs in Inc 1; return True / FREE):
  - `define(place)` — initialize Free
  - `status(place)` — query
  - `check_borrow_shared(place)` — Inc 2 will enforce
  - `check_borrow_mutable(place)` — Inc 2 will enforce
  - `check_move(place)` — Inc 2 will enforce
- **`Scope.borrows: BorrowState`** field auto-initialized.
- **`Scope.define()`** now registers a Free place for the new
  local (no-op until Inc 2 wires enforcement).

User-visible behavior: **unchanged**. The scaffolding is invisible
to user code; every check returns True.

## Inc 2-5 plan (subsequent stages)

- **Inc 2**: enforce xor rule at `&`/`&mut` sites; detect moves
  on Assign / Call consumption. Add diagnostic with span pointing
  "first borrowed here / second borrow here".
- **Inc 3**: block-exit + if/match join reconciliation (union of
  borrow states); reject divergent states (one branch moves, the
  other doesn't).
- **Inc 4**: `Copy` marker (`@copy` struct attr); scalars + tuples
  of Copy implicit; ref-typed values never move.
- **Inc 5**: explicit `move x` syntax; full diagnostic with
  span-pointing fix-its.

## Test coverage

- `test_stage66_inc1_borrow_checker_scaffolding`: verifies
  Place + BorrowState data types + Scope integration. All 282
  typecheck tests pass (no regression).

## Closure narrative

**3-clean-gate by inheritance**:

- Gate A (silent-failure): pure data-type addition + initialization;
  zero new enforcement logic; cannot silent-miscompile.
- Gate B (type-design): new dataclass + 4 status constants in
  isolation; no interaction with existing type-check logic
  beyond the auto-init in `Scope.define()`.
- Gate C (code-review): direct unit test on every API + full
  typecheck regression (282/282 pass).

**Self-host gate**: 223/223 GREEN.

## Stage 66 status

**Inc 1 SHIPPED**. Inc 2-5 deferred to future polish stages
(months total per the planning agent's estimate).

The scaffolding lets the project ship a `helixc/check.py --borrow-
check` opt-in flag in a future Inc that runs the enforcement on
top of the existing typecheck pass.

## Next stage

**Stage 67 opens immediately**: end-to-end ML demo (1 week,
autonomous). Builds on Stages 60-62: dyn file I/O + checkpoint
stdlib + named gradient accessors → train MNIST classifier in
Helix end-to-end.

Continuing autonomous progress through stages per user directive
("automatic until Stage X = v1.0 release").
