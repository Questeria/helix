# Stage 48 Progress - 2026-05-17

## Stage Goal

Stage 48 is **Tier 4 #14 Inc 2 — the `?` propagation operator**
(parser + typecheck + IR lowering). Stage 46 shipped Result<T,
E> typecheck-side; this stage adds the `?` postfix operator
that lets functions chain Result-returning calls without
explicit unwrap-and-rebuild boilerplate.

Beginner meaning: Rust-style `?` operator. `let x = parse(s)?;`
means "if parse returned Ok, extract the inner; if it returned
Err, return that Err from this function immediately." Massive
quality-of-life win for any code path that touches I/O, parsing,
or any fallible operation.

## Phase-0 limitation

Stage 48 ships the **syntax and typecheck** but not the runtime
early-return semantics, because Phase-0 Result has no runtime
Ok/Err tag yet (Stage 49+ work). In Phase-0:
- `?` parses and typechecks correctly.
- `expr?` desugars to `__try(expr)` at the AST level.
- The typechecker enforces (1) operand is Result, (2) enclosing
  fn returns Result, (3) Err types are compatible.
- IR lowering treats `__try(r)` as `unwrap_ok(r)` — pulls the
  Ok inner.
- At runtime: every Result is shape-Ok (no tag), so the early-
  return branch never fires. `?` is identity-lowered.

This means real code can be **written** with `?` today, and the
type system catches structural mistakes (non-Result operand,
non-Result return type, Err-type mismatch). Once Stage 49+
adds the runtime tag, the lowering arm becomes a real
conditional branch and Phase-0 code starts behaving with full
error-propagation semantics WITHOUT source changes.

## Increment 0 - Open Stage 48

Same conventions as Stage 35-47. 3-clean-gate closure.

## Increment 1 - Parser: `expr?` postfix → `__try(expr)`

`helixc/frontend/parser.py` postfix-call loop gains a
`T.QUESTION` arm that desugars `expr?` to
`A.Call(callee=A.Name("__try"), args=[expr])`. The QUESTION
token already exists in the lexer (line 74 + line 460).

Reuses existing `Call` AST node rather than introducing a
dedicated `Try` node — every IR pass already handles Call, so
this avoids ~10 pass-handler additions.

## Increment 2 - Typecheck: `__try` builtin dispatch arm

`helixc/frontend/typecheck.py` gains:
- `__try` in `_BUILTIN_NAMES`.
- An `if bn == "__try"` arm in the call-dispatch loop that
  validates:
  1. Arity: exactly 1 operand.
  2. Operand is `Result<T, E1>`.
  3. Enclosing fn return type is `Result<U, E2>`.
  4. `E1` is `_compatible` with `E2` (Err types must match).
  5. Result type = operand's Ok inner.

Each failure mode emits a kind-specific diagnostic with a
remediation hint.

## Increment 3 - IR lowering: identity + Result fn-return-type

`helixc/ir/lower_ast.py`:
- `__try` added to the Result identity-lowering tuple
  (`Ok` / `Err` / `unwrap_ok` / `unwrap_err` / `__try`); all
  one-arg, all lower to the operand (Phase-0: no runtime tag,
  so every `__try` is observationally identical to
  `unwrap_ok`).
- `_lower_type` gains a `Result<T, E>` arm that lowers to the
  Ok inner. Needed because `?` only makes sense in a Result-
  returning function, which forces Result into the fn
  signature; without this arm, the fn return type wouldn't
  lower to a concrete TIR scalar.

## Increment 4 - Stage 48 Closure (3/3 clean gates)

Same protocol as Stage 35-47.

### Phase-0 vs Stage 49+ semantic upgrade

Once Stage 49 adds the runtime Ok/Err tag, `__try(r)` lowering
becomes:

```
if is_err(r) {
    return r;  // early-return up the call stack
}
// fall through: extract Ok inner and continue
unwrap_ok(r)
```

The Stage 48 typecheck guards (1)-(4) above will all still
apply; only the IR lowering changes.

### Out of scope (Stage 49+)

- Real runtime Ok/Err tag (IR opcode for discriminated union).
- The runtime `?` early-return branch.
- Auto-promotion to panic at top-level if Result is unhandled.
- `?` for Option<T> (Phase-0 has no Option type yet).
