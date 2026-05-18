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

## Inc 2 SHIPPED (later same day)

`BorrowState.check_borrow_shared`, `check_borrow_mutable`, and
`check_move` are no longer stubs — they enforce the Rust 1.0-era
xor rule per place. Tests added: `test_stage66_inc2_*` (5 unit-
level tests). Closure: full typecheck regression GREEN; self-host
gate 223/223 GREEN.

## Inc 3 SHIPPED (later same day)

`_borrow_check_enabled` global flag + `_borrow_enforcement_enabled()`
gate added to `TypeChecker`. Wired at `&` / `&mut` Unary expr sites
(typecheck.py ~line 3878). When opt-in, double-`&mut` produces a
"Stage 66 borrow checker — xor rule violated" diagnostic.
Tests added: `test_stage66_inc3_*` (3 e2e tests). Closure: full
typecheck regression GREEN; self-host gate 223/223 GREEN.

## Inc 4 SHIPPED (later same day)

Per-fn `@borrow_check` attribute + `@copy` struct marker added.
- `_current_fn_borrow_check` flag pushed/popped in `_check_fn`
  prologue/epilogue; `"borrow_check" in fn.attrs` sets it.
- `_borrow_enforcement_enabled()` is now the OR of the global flag
  and the per-fn flag — so one annotated fn opts in without
  poisoning the rest of the module.
- `StructDecl.attrs: list[str]` field added (default `[]` via
  `__post_init__`). Parser threads attrs into `_parse_struct_decl`.
  `flatten_modules._rewrite_item` + `struct_mono` preserve attrs.
- `_copy_struct_names: set[str]` populated in pass-0 indexing for
  structs marked `@copy`. `_is_copy_struct_ty(TyStruct(...))`
  returns True for Copy structs — ready for Inc 5 move-wiring to
  consult before invalidating the source binding.
Tests added: `test_stage66_inc4_per_fn_borrow_check_attr_enables_only_for_that_fn`,
`test_stage66_inc4_global_flag_still_works_alongside_attr`,
`test_stage66_inc4_copy_struct_marker_registered`. Closure: full
typecheck regression GREEN (308/308); self-host gate 223/223 GREEN.

## Inc 5a SHIPPED (later same day)

Explicit `__move(x)` builtin recognized in typecheck (alongside
attach/detach/prove/derive) and identity-lowered to a read of x in
the IR. When `@borrow_check` is active and x is not a `@copy`
struct, the call transitions x's Place to MOVED via the scope
chain. Subsequent `&x` / `&mut x` then rejected by Inc 3 wiring.
Tests added: `test_stage66_inc5a_*` (3). Closure: 311 typecheck
+ 3 selfhost + 63 IR + 6 codegen GREEN.

## Inc 5b SHIPPED (later same day)

Implicit move detection at pass-by-value call sites — for every
`A.Name` arg passed to a user fn with a non-Copy struct type
under `@borrow_check`, the call transitions the place to MOVED.
Scalars + `@copy` structs + reference args skip the check.
Tests added: `test_stage66_inc5b_*` (4). Closure: 315 typecheck
+ 3 selfhost + 63 IR + 7 codegen GREEN.

## Inc 5c SHIPPED (later same day) — CLOSES Stage 66

Block-exit reconciliation across if/else arms + scope-chain
borrow routing:

- `Scope.borrows_check_shared/mutable/move/status` added —
  walk the scope chain to find the scope where the place was
  *originally defined* and route the operation there. Pre-fix,
  a check fired inside an inner block (e.g. if arm) affected
  only the inner scope's borrows, leaving the parent scope
  unaware of the transition.

- `_check_expr(A.If)` now snapshots `scope.borrows.state` +
  `shared_counts` before the arms, restores between then/else,
  and after both arms reconciles via JOIN:
  - For each place touched by any arm, the most-restrictive
    state wins (rank: MOVED=3 > MUTABLE=2 > SHARED=1 > FREE=0).
  - If the join state is MOVED but NOT all arms moved, emit a
    "borrow state of X diverges across if/else arms" diagnostic
    at the if expression's span.
  - Shared counts collapse to 1 for joined SHARED places (we
    cannot reconstruct counts soundly across arm divergence).
  - Apply unioned state to `scope.borrows` so the post-if code
    sees the conservative join.

- The no-else `if cond { ... }` form has an implicit identity
  arm that inherits the pre-if state — so a then-arm move
  produces a divergence diagnostic (since the implicit arm
  doesn't move).

- Inc 3 / 5a / 5b call sites updated to use the chain methods.

Tests added: `test_stage66_inc5c_*` (3):
- if_arms_diverge_on_move_diagnosed
- if_arms_agree_on_move_ok
- post_if_borrow_rejected_when_uniformly_moved

Closure: 318 typecheck + 3 selfhost + 63 IR + 13 codegen GREEN.

## Stage 66 CLOSED

All 5 increments shipped on 2026-05-18:
- Inc 1: Scaffolding (Place, BorrowState, Scope.borrows)
- Inc 2: Enforcement primitives (check_borrow_*, check_move)
- Inc 3: typecheck wiring at &/&mut sites + opt-in gate
- Inc 4: @borrow_check fn attr + @copy struct marker
- Inc 5: Move semantics (5a: explicit __move; 5b: implicit at
  pass-by-value; 5c: block-exit reconciliation + scope-chain)

Tier 4 #16 — borrow checker (Rust 1.0-era simple aliasing model)
— is shipped. Future polish (NLL, lifetime parameters, match-arm
reconciliation, escape analysis) can be added in later stages.

Total test additions across Stage 66: 20 tests.
Self-host gate: 3/3 GREEN at every commit.

## Next stage

**Stage 67 opens immediately**: end-to-end ML demo (1 week,
autonomous). Builds on Stages 60-62: dyn file I/O + checkpoint
stdlib + named gradient accessors → train MNIST classifier in
Helix end-to-end.

Continuing autonomous progress through stages per user directive
("automatic until Stage X = v1.0 release").
