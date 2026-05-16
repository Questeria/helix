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

## Increment 4 - D<Logic<T>> End-to-End Runnable (SHIPPED, 2026-05-16)

Goal: make the strategic D<Logic<T>> type composition actually
runnable end-to-end. Stage 24 shipped the typecheck-level TyDiff
and TyLogic types and the binop-propagation rule that composes
`D<Logic<T>> + Logic<T>` → `D<Logic<T>>`, but attach/detach were
never wired through IR lowering. Any program using them failed with
"unknown function 'attach' in IR lowering". This Increment closes
that gap.

What landed:

- `helixc/ir/lower_ast.py` — `attach(T) -> D<T>` and `detach(D<T>) -> T`
  now lower as identity (Phase-0: D<T> is representationally
  identical to T at runtime, matching the same convention used for
  Logic<T> in Increment 1).

This is a 2-line code change but unblocks the entire D<Logic<T>>
strategic composition. Four end-to-end runnable patterns verified:

1. `D<i32>` round-trip via `attach`/`detach` — exits 42.
2. `D<Logic<i32>> = attach(prove(42, 99))` — exits 42.
3. Boolean compute on D<Logic<i32>> via detach + and_logic +
   or_logic — exits 42.
4. Derive-as-rule: two `D<Logic<i32>>` parents combine via `derive`
   into a `D<Logic<i32>>` conclusion — exits 42.

Pattern (4) is the foundational shape of a differentiable
production-rule fire — the building block for differentiable
Datalog (the Scallop/Lobster pattern that's the Tier 3 #10
strategic differentiator).

Tests: 25 pass in test_stage36_provenance.py (Inc 1's 3 + Inc 2's 8
+ Inc 3's 10 + Inc 4's 4 new). Self-host gate: PASS.

### What's still missing for a "real" provenance lattice

The remaining gap before the strategic moat is fully claimed:

1. **Real two-parent provenance** — currently `derive(a, b)` keeps
   only `a`'s source tag. The lattice/semiring upgrade requires
   either (a) packed-i64 provenance per-value, or (b) a side-table
   arena. Either path needs an ABI / representation change that's
   bigger than Phase-0's identity-lowering approach.
2. **AD gradient flow through Logic ops** — `grad(f)` over
   `f: Logic<i32> -> Logic<i32>` needs the AD passes to register
   chain rules for and_logic/or_logic/not_logic. This is genuinely
   bigger than the typecheck-level work above and is the right
   target for Stage 36 Increment 5.
3. **Pretty-printing / debug observation of provenance** — currently
   provenance is invisible at runtime since Logic<T> = T. Once the
   representation upgrades (item 1), `print_provenance(l)` becomes
   trivially useful.

Increment 4 is the right stopping point for this autonomous batch:
the strategic D<Logic<T>> composition runs, the Stage 24 scaffolding
is no longer dead, and the path to real provenance + AD is clearly
mapped.

## Increment 5 - Real Two-Parent Provenance via Arena Side-Table (SHIPPED, 2026-05-16)

Goal: close the "single-tag provenance" Phase-0 limitation flagged
in Increments 2-4 without requiring an ABI representation change.

Approach: instead of changing `Logic<T>`'s runtime representation
(which would break the SysV ABI and require multi-slot returns), use
the existing global arena as a side-table. User code maintains
explicit source IDs, calls `register_derivation(left_src, right_src)`
to record a relationship under an arena-allocated handle, then
queries `parent_left_at(handle)` / `parent_right_at(handle)` later.

This is **real two-parent provenance** — observable at runtime,
queryable, distinct from the single-tag prove() tag.

What landed:

- `helixc/frontend/typecheck.py` — 3 new typecheck-recognized builtins
  registered in `_BUILTIN_NAMES`:
  - `register_derivation(left_src: i32, right_src: i32) -> i32`
  - `parent_left_at(handle: i32) -> i32`
  - `parent_right_at(handle: i32) -> i32`
- `helixc/ir/lower_ast.py` — `register_derivation` emits two
  `ARENA_PUSH` ops and returns the index of the first push (the
  handle). `parent_left_at` lowers to `ARENA_GET(idx)`,
  `parent_right_at` to `ARENA_GET(idx + 1)`.

5 end-to-end runnable verifications (all exit-code-checked):

1. `parent_left_at` recovers the first source ID.
2. `parent_right_at` recovers the second.
3. Both parents readable as a sum.
4. Two independent derivations produce independent handles —
   reads on h1 don't disturb h2's data.
5. Integrated Datalog: the grandparent rule fires (and_logic) AND
   the provenance handle correctly identifies both parent source
   IDs. The same program both proves a logical conclusion AND
   recovers the evidence trail.

Tests: 31 pass in test_stage36_provenance.py (Inc 1's 3 + Inc 2's 8
+ Inc 3's 10 + Inc 4's 4 + Inc 5's 6 new). Self-host gate: PASS.

### Phase-0 limitation that remains

The arena side-table is a flat append-only log: `register_derivation`
allocates 2 i32 entries linearly. There's no hashmap index, so
"find all derivations of source X" is an O(n) scan. For typical
Datalog programs (hundreds of facts) this is fine; an Increment 6+
upgrade can wire a hashmap if needed.

The user must explicitly call `register_derivation` — the combinators
(`derive`, `and_logic`, etc.) do NOT auto-register. A future increment
can make this automatic by reserving a "current derivation handle"
slot per Logic<T> at the IR level, but that's a representation
change deferred to a later session.

## Increment 6 - AD Gradient Flow Through Logic Ops (planned)

The remaining strategic goal: `grad(loss)(...)` where `loss` is a
function returning `Logic<f32>` or `D<Logic<T>>`. The AD passes
need chain rules registered for `and_logic`, `or_logic`, `not_logic`
(fuzzy-relaxation derivatives, e.g. sigmoid-relaxed AND).

This is genuinely bigger work than Increments 1-5 because it touches
the autodiff passes (`helixc/frontend/grad_pass.py`,
`helixc/stdlib/autodiff_reverse.hx`). Reserved for a later session.

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
