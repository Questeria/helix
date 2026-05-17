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

### Gate-1 results (commit 722bfdb)

Silent-failure lane caught F1-F4 + 3 LOWs. Resolution:
- **F1 HIGH** (dynamic-Err `?` via call return): acknowledged as
  Phase-0 limitation. No static way to distinguish Ok-from-call
  vs Err-from-call until Stage 49 runtime tag.
- **F2 HIGH** (typed-let Err `?`): FIXED. Provenance map check at
  `__try` site rejects `name?` when the name was constructed via
  `Err(...)` directly. Same defect class as Stage 46 G2-F1.
- **F3 MEDIUM** (user-callable `__try`): deferred. Low-risk; the
  typecheck arm runs the same validation regardless of who wrote
  the call.
- **F4 MEDIUM** (brittle return_ty default): deferred to Stage 49.
- 3 LOWs: applied inline (shadow hint mentions `__-prefix`,
  symmetric `_compatible` note, TODO marker for future 2-param
  wrappers).

Type-design and code-review lanes returned CLEAN at gate-1.

### Gate-2 results (this commit)

Silent-failure lane caught 2 HIGH silent miscompiles + 1 LOW
that gate-1's per-name flat-dict provenance design didn't
guard against. Both reproducers verified end-to-end at exit
code 99:
- **F1 HIGH** (inner-block shadow leaks outer Err provenance):
  FIXED. `_check_block` now snapshots
  `_result_constructor_provenance` at entry and restores at
  exit. Inner-block `let r = Ok(5)` no longer overwrites the
  outer `r='err'` entry.
- **F2 HIGH** (member-access operand bypasses provenance check
  → `p.a?` on a struct-literal `Err(...)` field silently
  extracts Err as Ok): NOT FIXED in Stage 48; documented as
  **F5** deferred (same defect class as F1 dynamic-Err: aggregate
  field access is fundamentally dynamic from per-name
  provenance's perspective; Stage 49+ runtime tag fixes the
  whole class). New regression test
  `test_stage48_closure_gate2_f5_member_access_documented_as_phase0_defect`
  pins the current Phase-0 behavior so a future regression
  surfaces the right delta.
- **F3 LOW** (generic diagnostic for user-callable `__try`):
  deferred (same as F3 from gate-1).

Type-design lane: 1 HIGH + 4 MEDIUM + 2 LOW. Resolution:
- **H1 HIGH** (span attribution points to operand, not `?`
  token): deferred. Operand-span is acceptable for now;
  diagnostic quality matters more once Stage 49+ branching IR
  lands.
- **M2 MEDIUM** (constructor-provenance check duplicated, not
  factored into helper): deferred to Stage 49 as the right
  moment for the refactor (3rd consumer arrives with the
  runtime-tag-aware arm).
- **M3 MEDIUM** (`_compatible` Err-side alias gaps): no
  reproducer surfaced; flagged for Stage 49 audit prep.
- **M4 MEDIUM** (rename `__try` + structural `__-prefix`
  policy): deferred; rename touches 5 sites + comment claims.
  Acceptable cost later.
- **M5 MEDIUM/false-reject** (cross-fn provenance carry —
  param `r` falsely flagged after a prior fn's `let r =
  Ok(...)` set the dict): FIXED. `_check_fn` now clears
  `_result_constructor_provenance` at entry.
- **L6 LOW** (IR identity tuple lacks Stage 49 split marker):
  applied — `STAGE49_TODO` comments added at the `__try`
  identity-tuple line and the `_lower_type` Result arm.
- **L7 LOW** (no `desugar_origin` marker on synthesized Call):
  deferred; no concrete consumer needs it yet.

Code-review lane: CLEAN with 6 findings (0 CRITICAL / 1 HIGH /
3 MEDIUM / 2 LOW). Resolution:
- **H1 HIGH** (test_stage48_inc3_phase0_runtime_returns_ok_inner
  has limited regression coverage of the IR identity arm):
  partially addressed by the new F1+M5 regression tests which
  exercise the typecheck path more thoroughly. Structural-TIR
  assertion deferred.
- **M1 MEDIUM** (non-Result-operand diagnostic omits operand
  name): FIXED. When operand is `A.Name`, the diagnostic now
  reads `` `?` on 'x' requires a Result<T, E> operand, got i32 ``.
- **M2 MEDIUM** (test naming meta-named per closure cycle):
  deferred; the historical tag retains audit-trail value.
- **M3 MEDIUM** (5 missing test cases): 3 of 5 added (F1 inner-
  block shadow, M5 cross-fn carry, F5 member-access defer).
  The 2 remaining (while/for body `?`, 3-deep `?` chain)
  deferred — Stage 49 runtime-tag-tests will be the natural
  home.
- **L1 LOW** (TODO comment placement in _lower_type): applied.
- **L2 LOW** (`_compatible` inline-comment redundancy):
  deferred; comment is helpful in-place.

### Test summary

Stage 48 test count: 11 → 14 (added F1, M5, F5 regression
tests). Stage 46+48 combined: 41 tests pass. Full helixc
suite: 626 passed + 1 pre-existing unrelated failure
(`test_grad_rejects_opaque_call_in_loss`). Self-host cascade
G2..G4 byte-identical. dogfood_17 still exits 42.

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
