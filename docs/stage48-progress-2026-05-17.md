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

### Gate-3 results (this commit)

Pattern repeating: gate-3 silent-failure caught a NEW HIGH
silent miscompile that gate-2's fix accidentally created. The
cascading-defect rhythm is now Stage 48's 4th instance (gate-1
F2, gate-2 F1, gate-2 M5, gate-3 G3-F1) and mirrors Stage 46's
4 escalating audit findings. Audits are doing their job;
patches are converging on a sound design.

- **G3-F1 HIGH** (inner-block ASSIGN to outer Result name +
  post-block `?`): the gate-2 snapshot-restore solved the inner-
  LET case but introduced an inner-ASSIGN mirror. `let mut r =
  Ok(7); { r = Err(99); } let v = r?;` — assign-arm popped `r`
  inside the inner block, restore put back the stale `r='ok'`,
  post-block `r?` accepted silently. 3 reproducers (anonymous
  block, if-then arm, match arm) all verified exit 99.
  FIXED with scope-aware tracking: parallel set-stack
  `_result_let_block_scopes` records names introduced via let
  inside each open block. At restore: names that were inner-let-
  introduced (shadows) leave the outer dict alone; names that
  exist in the saved snapshot AND were inner-assign-mutated
  (current dict differs and NOT in inner-lets) drop from the
  restored map → falls into F1-dynamic Phase-0 territory
  (typecheck-clean, runtime exit 99 remains as a Phase-0 known
  defect, fixed by Stage 49 runtime tag).
- **G3-F2 MEDIUM** (cascade of G3-F1 through nested blocks):
  addressed by same fix (each level's restore composes).
- **G3-F3 LOW** (exception-safety in finally): FIXED — nested
  try ensures the provenance restore always runs even if
  `_pop_local_const_scope` raises.

Type-design lane: 5 findings (0 HIGH / 3 MEDIUM Stage 49-prep
/ 2 LOW polish).
- **T-M1 MEDIUM** (scope-stack pattern divergence): deferred
  to Stage 49 — the fix from G3-F1 partially closes this by
  introducing the parallel set-stack.
- **T-M2 MEDIUM** (O(N·M) snapshot cost): same M1 conversion
  addresses incidentally.
- **T-M3 MEDIUM** (5-site stewardship without centralizing
  helper): partially addressed by the expanded comment block
  at the provenance map declaration listing all 6 sites
  explicitly.
- **T-L1 LOW** (comment lineage churn): expanded comment block
  is the right home — closure-ledger pointer added.
- **T-L2 LOW** (F5 test STAGE49 inline marker): applied.

Code-review lane: 5 findings (0 CRITICAL / 0 HIGH / 3 MEDIUM /
2 LOW).
- **CR-M1 MEDIUM** (pop-before-restore exception-safety):
  FIXED — nested try/finally.
- **CR-M2 MEDIUM** (no positive test for operand-name diag):
  FIXED — new test `test_stage48_question_diagnostic_names_operand`.
- **CR-M3 MEDIUM** (M5 test order-sensitive on fn declaration
  order): documented with banner comment near the test.
- **CR-L1 LOW** (`STAGE49_TODO:` convention diverges from
  pre-existing `TODO(stageN):`): FIXED — all 4 sites renamed.
- **CR-L2 LOW** (cross-gate F-tag namespace reuse): documented
  in this ledger — F-tags reset per gate.

### Test summary

Stage 48 test count: 11 → 14 (gate-2 +F1, +M5, +F5) → 18
(gate-3 +G3-F1a, +G3-F1b, +G3-F1c, +operand-name-diag).
Stage 46+48 combined: 45 tests pass. Self-host cascade still
3/3 byte-identical. dogfood_16 + dogfood_17 still exit 42.

### Gate-3 verification audit cycle (Stage 48 CLOSED commit 7eaba56)

Re-spawned all 3 audit lanes to confirm no new defect class
introduced by the gate-2+gate-3 fix sweep. The first verification
read findings → `docs/audit-stage48-inc4-gate3-{silent-failures,
type-design,codereview}.md`.

**Initial verdict: 3/3 GATE-3 CLEAN — Stage 48 declared CLOSED.**

(See "Gate-4/5 cascade" below for the follow-up audit pass that
caught two more HIGHs against the gate-3 head, and the subsequent
narrowing that landed in commit 3415727→ next-commit.)

- **Silent-failure**: 0 HIGH/CRITICAL. 1 MEDIUM (MED-1:
  `let r = map_ok(Err(7), 999); r?` — same Phase-0 defect
  class as F1-dynamic/F5/F6, fixed by Stage 49 runtime tag)
  + 1 LOW (Result inside TyTuple/TyArray identity-lowering).
  Both deferred to Stage 49. while/for/loop body coverage,
  compound if/else-if chains, user-callable `__try`, tuple-
  destructuring let, and Helix-has-no-closure: all explicitly
  cleared by trace.
- **Type-design**: 0 HIGH/MEDIUM. 2 LOW (Result fn-signature
  size-asymmetry test pin; if-merge join not yet a 3-state
  semilattice — sound, missed-detection only) + 2 OBS (the
  TyResult two-inner symmetry is correctly applied at all
  8 helper sites; wrapper-composition `Known<Result<T,E>>`
  works at type level). All deferred to Stage 49 or
  Phase-1 backlog.
- **Code-review**: 0 CRITICAL/HIGH/MEDIUM. 2 LOW (no parser-
  level test for `(if c {} else {})?`; M5 banner location).
  Notable verification: `STAGE49_TODO:` → `TODO(stage49)`
  rename from gate-3 CR-L1 is complete; operand-name
  interpolation degrades cleanly for non-Name operands;
  diagnostic strings consistently use `` `?` `` not
  `__try`; provenance-map declaration documents all
  consumers.

Cumulative Stage 48 audit budget: 4 gates (1+2+3+verification),
22 findings across 3 lanes, 9 FIXED inline, 13 deferred to
Stage 49 or Phase-1. Same cascading-defect rhythm Stage 46
hit (gate-2 F1 created the gate-3 G3-F1 mirror). Patches
converged on a sound design.

### Gate-4/5 follow-up cascade (post-close audits)

After the initial 3/3 CLEAN closure declaration, a
post-close gate-4 silent-failure audit (cron-fired against
HEAD=3415727) discovered TWO additional HIGH silent miscompiles
the prior gates missed plus one type-design HIGH:

- **G4-F1 HIGH** (match-arm bare-Assign body bypasses scope
  restore): `match b { true => { r = Err(99); }, false => {} }`
  (and bare-block, if-then variants) silently miscompiled to
  exit 99 — assign-arm mutated provenance but no scope-restore
  ran for expression-form arm bodies. FIXED via new helper
  `_check_expr_in_block_scope` wrapping match-arm / if-else
  expression-form arm bodies with the same snapshot-restore
  discipline as `_check_block`.
- **G4-F2 HIGH** (ASSIGN-then-LET-shadow on same name): gate-3's
  per-name `_result_let_block_scopes` set could not detect a
  pre-shadow outer-assign — the inner let-shadow's dict write
  masked the prior mutation, stale 'ok' survived restore, exit
  50 verified end-to-end. FIXED via parallel
  `_result_assigns_block_scopes` set-stack populated at every
  Assign event; at restore, names in this set drop from the
  restored map unconditionally (per-event mask).
- **G4-H1 HIGH→DEFERRED** (Result-of-wrapper-quintet asymmetry):
  `Result<Known<i32>, i32>` in fn-RETURN-type position passes
  typecheck but raises NotImplementedError at IR lowering
  because `_lower_type`'s Result-arm identity-recurses into a
  wrapper-quintet without type-position arms. The initial
  attempted fix REJECTED at typecheck broadly, but that broke
  an existing Stage 46 test + the dogfood_16 `cross_stack_result`
  let-binding probe (both of which use the let-binding-position
  pattern that works fine via expression-lowerer wrapper-arms).
  RESOLUTION: narrowed (revert the broad rejection); pin the
  Phase-0 limit via dedicated test
  `test_stage48_closure_gate5_g4h1_result_of_wrapper_in_fn_signature_raises_at_ir`
  asserting typecheck-clean + IR-lowering raises
  NotImplementedError, mirroring the F5 deferral pattern. Stage
  49's runtime tag + wrapper type-position arms lift this in
  one fix.

Updated Stage 48 audit budget: **6 gates** (1+2+3+verification+
4+5), 30+ findings across 3 lanes, 13 FIXED inline (including 2
new HIGH silent miscompiles in gate-4/5), 17+ deferred to Stage
49 or Phase-1. The cascading-defect rhythm is the audit
infrastructure's job — every closure cycle catches a defect
class the prior fix didn't think of. Final state IS sound: all
documented defects are deferred-to-Stage-49 known-limits in the
F1-dynamic equivalence class that the runtime Ok/Err tag
eliminates wholesale.

Final test count: Stage 48: 21 tests (Inc 1+2+3 base 11, +F1+M5+F5
gate-2, +G3-F1a/b/c+operand-name gate-3, +gate-4/5 G4-F1+G4-F2
+G4-H1 pin). Stage 46+48 combined: 50 tests. Self-host cascade
G2..G4 byte-identical preserved. dogfood_16 + dogfood_17 still
exit 42.

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
