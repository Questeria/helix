# Stage 46 Progress - 2026-05-17

## Stage Goal

Stage 46 is **Result<T, E> typecheck-side scaffolding**
(Tier 4 #14 Inc 1 — the `?` operator + parser changes come
in Stage 47). First two-parameter wrapper family in the
Helix type system.

Beginner meaning: real programs need a way to say "this
function either succeeds with a T, or fails with an E"
without crashing on the failure case. Rust calls it
`Result<T, E>`; the broader ML world calls it
`Either<E, T>` or `Try<T>`. Helix needs this before any
non-trivial dataset I/O, network call, or parser-style
code can be written safely.

This stage builds the type-level scaffolding:
- `TyResult(ok_ty, err_ty)` dataclass (two inner types,
  not one).
- `Ok(v)` / `Err(e)` constructors as built-ins.
- `unwrap_ok` / `unwrap_err` / `is_ok` / `is_err` accessors.
- `map_ok(r, f)` / `map_err(r, f)` higher-order combinators.
- 8 type-system helper arms (compatible / shape /
  refinement walks / fmt / etc.) parallel to the existing
  5 single-inner wrapper families.

No parser change (no `?` operator). No IR change (all
identity-lowered at Phase-0; the ok/err discriminant
lives in the type system only — actual runtime tag goes
in Stage 48+).

## The 4 modes Result enables

- **Direct error return**: `fn parse_int(s: &str) -> Result<i32, ParseError>`.
- **Bubble-up via combinator**: `parse_int(s).map_ok(|n| n * 2)`.
- **Conditional fork**: `if is_ok(r) { ... } else { ... }`.
- **Future `?` operator** (Stage 47+): `let n = parse_int(s)?;`
  desugars to early-return on Err.

## Increment 0 - Open Stage 46 (Convention Declaration)

Same conventions as Stage 35-45. 3-clean-gate closure.

## Increment 1 - TyResult dataclass + intro/elim + helper arms

Mirrors the Stage 37-41 wrapper-family playbook EXCEPT for
the two-parameter generic. The 7 type-system helpers
(`_compatible`, `_refinement_shape_exact`,
`_erase_refinement`, `_contains_refinement`,
`_is_refinement_container`, `_contains_refined_function`,
`_contains_unknown_type`) need arms that walk BOTH inner
types.

Builtins (8 new):
- `Ok(v)` — produces `Result<typeof(v), TyVar>` (err type
  inferred from context).
- `Err(e)` — produces `Result<TyVar, typeof(e)>` (ok type
  inferred from context).
- `unwrap_ok(r)` / `unwrap_err(r)` — extract inner; panic
  if wrong variant (Phase-0; safety upgrade in Stage 47).
- `is_ok(r)` / `is_err(r)` — return bool.
- `map_ok(r, f)` / `map_err(r, f)` — apply f to the inner;
  passes the other variant through unchanged.

## Increment 2 - Identity IR lowering

Phase-0: all 8 builtins lower as identity. The Ok/Err
discriminant lives at the type level only — no runtime
tag. This is the same pattern Stages 37-41 used for the
5 single-inner wrapper families. Stage 47+ will introduce
a real runtime tag once `?` operator semantics need it.

## Increment 3 - Dogfood + tests

`helixc/examples/dogfood_16_result_basic.hx` — a basic
parse-or-default program that uses Ok/Err/is_ok/unwrap_ok
to demonstrate the type system catches mismatches.

## Increment 4 - Stage 46 Closure (3/3 clean gates)

Same protocol as Stage 35-45.

### Out of scope (Stage 47+)

- `?` operator desugaring (needs parser change).
- Real runtime Err tag (needs IR opcode for discriminated
  union or arena side-table).
- Auto-promote Result to panic at top-level if unhandled.
