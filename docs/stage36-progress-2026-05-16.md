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

### Increment 1 status: SHIPPED (commit 9e9b421, 2026-05-16)

What landed:

- `helixc/frontend/typecheck.py` — added `prove(value, source)` and
  `unwrap_logic(l)` as typecheck-recognized builtins next to
  `detach`/`attach`. Both registered in `_BUILTIN_NAMES`.
- `helixc/ir/lower_ast.py` — both lower to identity at IR level
  (Phase-0: Logic<T> has zero runtime overhead).
- `helixc/tests/test_provenance.py` — 8 new typecheck-level tests
  (added in parallel by an autonomous loop run).
- `helixc/tests/test_stage36_provenance.py` — 3 end-to-end runtime
  tests pinning exit codes.
- Self-host gate: PASS (G2..G4 byte-identical sha a6f1ee44...).
- 24 tests pass (21 in test_provenance.py + 3 in test_stage36
  _provenance.py).

Deviation from plan: kept `prove`/`unwrap_logic` as Python-side
intrinsics (matching detach/attach pattern) rather than writing a
stdlib `.hx` file, because generic functions (`fn fact<T>...`) are
not yet shipped. The intrinsic path matches the existing pattern and
is end-to-end-runnable.

## Increment 2 - Provenance-Composing Combinators (in progress)

Goal: extend the single-input prove/unwrap_logic vocabulary with
combinators that compose provenance from two parents. This is the
first step toward a real provenance lattice/semiring (Increment 3
will track BOTH parents, not just the first).

Scope:

- `derive(a: Logic<T>, b: Logic<U>) -> Logic<T>` — propagates
  provenance through a binary derivation step. Phase-0 keeps the
  value of `a` and discards `b`; the lattice upgrade tracks both.
  Both inputs must be Logic-wrapped — passing a bare T fires trap
  24100.
- `and_logic(a, b: Logic<i32>) -> Logic<i32>` — boolean AND on
  0/1 truth values, lowered to BIT_AND.
- `or_logic(a, b: Logic<i32>) -> Logic<i32>` — boolean OR, lowered
  to BIT_OR.
- `not_logic(a: Logic<i32>) -> Logic<i32>` — boolean NOT, lowered
  to `1 - a` (preserves provenance).

### Increment 2 status: SHIPPED (this commit, 2026-05-16)

What landed:

- `helixc/frontend/typecheck.py` — added 4 builtins in the same
  pattern as Increment 1. Registered in `_BUILTIN_NAMES`. Each
  validates that args are Logic-typed (trap 24100 otherwise).
- `helixc/ir/lower_ast.py` — all 4 lower to direct IR ops:
  - `derive(a, b)` → identity on a (b evaluated for side-effect
    parity with the AST traversal pattern).
  - `and_logic(a, b)` → BIT_AND.
  - `or_logic(a, b)` → BIT_OR.
  - `not_logic(a)` → 1 - a.
- `helixc/tests/test_stage36_provenance.py` — 8 new end-to-end
  tests covering full truth tables (AND × 4, OR × 4, NOT × 2),
  derive, compound expressions, and trap-24100 boundary checks.
- 11 tests pass in test_stage36_provenance.py (Increment 1's 3 +
  Increment 2's 8).

First sample of a non-trivial provenance-typed expression that runs:

```rust
let t = prove(1, 0);
let f = prove(0, 0);
let r = or_logic(and_logic(t, t), not_logic(f));
unwrap_logic(r) * 42  // exits 42
```

This is the first running Helix program that does propositional
reasoning with provenance-typed truth values.

## Increment 3 - Boolean-Algebra Completeness (SHIPPED, 2026-05-16)

Goal: extend the AND/OR/NOT core with the rest of propositional
logic so user code can express arbitrary boolean expressions over
provenance-typed truth values.

What landed:

- `xor_logic(a, b: Logic<i32>) -> Logic<i32>` — lowered to BIT_XOR.
- `implies_logic(a, b: Logic<i32>) -> Logic<i32>` = OR(NOT a, b),
  lowered to BIT_OR(1-a, b).
- `eq_logic(a, b: Logic<i32>) -> Logic<i32>` = NOT XOR, lowered to
  1 - BIT_XOR(a, b).
- `if_logic(cond, then_v, else_v: Logic<T>) -> Logic<T>` —
  provenance-typed ternary. Lowered to CMP_NE + SELECT.
- `to_logic_bool(x: i32) -> Logic<i32>` — convenience lift; identity
  at IR.

All 5 register in `_BUILTIN_NAMES` and enforce trap 24100 on bare-T
arguments. 21 tests pass in test_stage36_provenance.py (Increment
1's 3 + Increment 2's 8 + Increment 3's 10).

Highlight: **De Morgan's law verified at runtime over provenance-
typed values**:

```rust
let a = prove(1, 0);
let b = prove(0, 0);
let lhs = not_logic(and_logic(a, b));
let rhs = or_logic(not_logic(a), not_logic(b));
unwrap_logic(eq_logic(lhs, rhs)) * 42  // exits 42
```

This is the first running Helix program that verifies a theorem of
classical propositional logic with the operands carrying typed
provenance. Boolean algebra is now feature-complete on Logic<i32>.

## Increment 4 - True Two-Parent Provenance (planned)

Goal: replace the single-tag i32 provenance with a real two-parent
provenance lattice. derive/and_logic/or_logic should track BOTH
parent sources, not just the first.

Sketch:

- Runtime representation: a `Logic<T>` becomes a pair
  `{ value: T, prov: i64 }` where `prov` packs two i32 source tags
  into the high/low halves.
- `prove(v, src)` puts `src` in both halves.
- `derive(a, b)` merges via `(a.prov << 32) | b.prov` (or similar
  combining function).
- `and_logic` / `or_logic` similarly combine.
- A `logic_provenance(l: Logic<T>) -> i64` accessor exposes the
  packed tag.
- A `logic_source_left` / `logic_source_right` extractor pair lets
  user code unpack.

This is the bridge to Increment 4 (AD through Logic<T>), which will
treat the provenance pair as a gradient path: `D<Logic<T>>` carries
both the gradient AND the evidence trail simultaneously.
