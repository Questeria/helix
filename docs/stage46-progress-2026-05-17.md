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

### Gate 1 silent-failure fix sweep (commit b4ff52b)

Gate-1 audits found 1 CRITICAL + 2 HIGH + 1 MEDIUM
silent-failure findings. Initial Phase-0 design naively
stubbed runtime behavior — `is_ok` always returned 1,
`is_err` always returned 0, `map_err` was a runtime no-op.
The audit caught that these stubs silently miscompile any
real error-handling code: `if is_err(r) { panic("err") }`
ALWAYS took the else branch regardless of whether r was
actually Err.

The correct Phase-0 stance is to TYPECHECK-REJECT these
operations rather than silently lower them to misleading
defaults:

- **is_ok / is_err** — reject with kind-specific message.
  When operand has statically-determinable provenance
  (Ok or Err constructor), the error also says "is
  statically true/false; you can replace this call with
  the literal".
- **map_err** — reject with "no runtime semantics in
  Phase-0" hint pointing at explicit `Err(new_err)` at
  the call-site instead.
- **unwrap_err on Ok-inferred Result** — reject with
  "constructed via Ok() — this is an unconditional
  runtime panic" diag.
- **unwrap_ok on Err-inferred Result** — symmetric
  reject.

Surface table (post-gate-1):

| Builtin       | Phase-0 status | Stage 48+ plan |
|---------------|---------------|----------------|
| `Ok(v)`       | ✅ works       | unchanged      |
| `Err(e)`      | ✅ works       | unchanged      |
| `unwrap_ok`   | ✅ on Ok       | branch on tag  |
| `unwrap_err`  | ✅ on Err      | branch on tag  |
| `map_ok`      | ✅ works       | branch on tag  |
| `is_ok`       | ❌ rejected    | returns bool   |
| `is_err`      | ❌ rejected    | returns bool   |
| `map_err`     | ❌ rejected    | branch on tag  |

The 4 "rejected" surface elements work at the type system
level (Result<T,E> typechecks; the rejections happen at
call-site) but produce a typecheck error if the user
actually calls them in Phase-0. Stage 48+ will add the
runtime Ok/Err tag and unlock those four call sites.

### Gate 2 silent-failure G2-F1 fix (commit 50b542e)

CRITICAL G2-F1: gate-1 F4 caught the inference path
(`let r = Ok(7)`) via TyUnknown.hint, but the typed-let
path (`let r: Result<i32, i32> = Ok(7)`) overrides the
inferred type with the declared type at bind time, stripping
the hint. Post-gate-1, the typed-let wrong-arm call
typechecked clean and silently returned the Ok payload.

Fix: per-binding constructor-provenance tracking via a new
`self._result_constructor_provenance: dict[str, str]` map.
At let-binding time, if the RHS is a direct `Ok(...)` /
`Err(...)` call, record "ok" or "err" on the binding name.
At `unwrap_ok` / `unwrap_err` dispatch, check the map by
arg name; reject on wrong-arm.

The TyUnknown.hint check from gate-1 F4 is preserved as a
second line of defense for the inference path.

### Gate 3 silent-failure G3-F1 + G3-F2 fixes

G3-F1 HIGH: mutable reassignment left stale provenance.
`let mut r = Ok(7); r = Err(99); unwrap_ok(r)` silently
returned 99. Fix: at the `A.Assign` arm, if the target is
a Name in the provenance map, either overwrite (if RHS is
Ok/Err) or pop (if RHS is anything else — non-constructor
reassignment).

G3-F2 MEDIUM: `map_ok` / `map_err` lost provenance from
their source argument. `let r0 = Ok(7); let r = map_ok(r0,
999); unwrap_err(r)` silently returned 999. Fix: extend
the let-RHS matcher to propagate provenance through these
combinators (they only transform the value side, never the
variant tag).

G3-F3 MEDIUM (conditional RHS, e.g. `let r = if cond { Ok(7)
} else { Err(99) }`): explicitly deferred per the audit's
own recommendation. Cleaner Stage 48+ work once the runtime
tag enables real `if is_ok(r)` branching.

### STAGE 46 CLOSED 2026-05-17 at Inc 4 (3/3 clean audit gates)

Result<T, E> typecheck-side scaffolding shipped end-to-end
(Tier 4 #14 Inc 1). First two-parameter wrapper family in
the Helix type system.

Phase-0 surface (final, post-3-gate-closure):

| Builtin       | Phase-0  | Stage 47+ / 48+ plan |
|---------------|----------|----------------------|
| `Ok(v)`       | ✅ works  | unchanged            |
| `Err(e)`      | ✅ works  | unchanged            |
| `unwrap_ok`   | ✅ on Ok  | branch on runtime tag |
| `unwrap_err`  | ✅ on Err | branch on runtime tag |
| `map_ok`      | ✅ works  | branch on runtime tag |
| `is_ok`       | ❌ reject | enable on runtime tag |
| `is_err`      | ❌ reject | enable on runtime tag |
| `map_err`     | ❌ reject | enable on runtime tag |

The 4 "reject" builtins typecheck cleanly when first
defined; they only error at call-site. Stage 48+ will add
the runtime Ok/Err tag and unlock those four call surfaces.

Wrong-arm safety net: two-layer rejection (TyUnknown.hint
provenance for the inference path + `_result_constructor_
provenance` map for the typed-let / map_ok / map_err
propagated paths + Assign-arm invalidation for mutable
reassignment). Pre-Stage-46 Result builtin calls could
silently miscompile in 3 distinct patterns; post-closure
all 3 are typecheck-rejected.

26 Stage 46 tests green. Self-host cascade still
byte-identical G2..G4 fixpoint. 16 dogfood programs total.

Stage 47 opens next per ROADMAP Phase 2.

### Out of scope (Stage 47+)

- `?` operator desugaring (needs parser change).
- Real runtime Err tag (needs IR opcode for discriminated
  union or arena side-table).
- Auto-promote Result to panic at top-level if unhandled.
- G3-F3 conditional-RHS provenance (`if cond { Ok } else
  { Err }`) — explicitly deferred to Stage 48+ when the
  runtime tag enables real `if is_ok(r)` branching.
