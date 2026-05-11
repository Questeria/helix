# Stage 28.8 pre-29 audit gate — Cycle 23 (Audit A: silent failures)

**Date:** 2026-05-11
**HEAD:** `4bdc800` ("Cycle 22 Audit C C0: delete dead visit_stmt shim in
struct_mono")
**Lens:** silent failures (Audit A)
**Streak counter at start:** 2/5 (cycle 22 silent-failures was clean)

---

## Scope

Strict-criterion read-only audit. Two priorities per the cycle-23 brief:

1. **Verify the cycle-22 C-fix** (`4bdc800`, deletion of the dead
   `visit_stmt` shim in `helixc/frontend/struct_mono.py:186-189`) did
   not introduce regressions to `collect_concrete_uses`.
2. **Re-scan everything cycle-22 covered** to confirm stability at the
   new HEAD.

Per the cycle-23 task brief, manufacturing of findings is forbidden,
and re-flagging of cycle 1-22 findings is forbidden.

---

## Priority 1 — C-fix regression check

### Diff inspection

`git diff bee36e6..4bdc800 -- helixc/` shows exactly one file changed
(`helixc/frontend/struct_mono.py`) and exactly one semantic change:
the four-line `visit_stmt(s)` function definition at lines 186-189 was
replaced by a four-line comment block. The replacement comment
records the deletion rationale and points future authors at
`_body_visitor.visit(stmt)` as the direct API. No other lines moved.

### Dead-code confirmation

AST parse of the post-fix file confirms:

```
DEF: visit_expr at line 177
CALL: visit_expr() at line 199
CALL: visit_expr() at line 205
```

No `visit_stmt` definition, no `visit_stmt` call. The function had
zero callers in the file before deletion (the prior cycle-22 Audit C
finding, conf 80) and has zero callers / zero definition after
deletion. No dangling references.

`grep visit_stmt helixc/` across the whole codebase returns no hits.
The shim was strictly local — its removal cannot affect any external
caller.

### Body-walk pipeline intactness

The two surviving call sites at `struct_mono.py:199` (`visit_expr(it
.body)` for FnDecl bodies) and `:205` (`visit_expr(it.value)` for
ConstDecl initializers) both invoke `_body_visitor.visit(...)`
through the retained `visit_expr` shim. `_BodyVisitor(ASTVisitor)` at
lines 133-173 is unchanged. Its five overrides (`visit_Cast`,
`visit_Name`, `visit_TileLit`, `visit_Let`, `visit_ConstStmt`) all
remain in place to walk type fields the default ASTVisitor skips, and
the cycle-22 verdict on each is unaffected.

`ASTVisitor.visit()` at `ast_walker.py:180-197` handles a `None`
argument explicitly (`if node is None: return None`), so passing a
fn body that happens to be absent (extern fn — guarded at line 198
with `if not it.is_extern:` anyway) cannot crash silently.

### Test regression check

`pytest helixc/tests/test_struct_mono.py` (39 tests) all pass at HEAD
`4bdc800`. Tests exercise generic-struct collection across signatures,
let-bindings, casts, name-generics, tile literals, const-decl
initializers, and the `_ty_key` dedup machinery — i.e. exactly the
body-walk surface that touches the deleted shim's neighborhood.

**Verdict for Priority 1: clean.** The C-fix is strictly dead-code
removal. The walking pipeline is byte-equivalent in behavior to the
pre-fix code.

---

## Priority 2 — Cycle-22 re-scan at HEAD `4bdc800`

For each cycle-22 target, the audit re-checked whether the diff
between `bee36e6` and `4bdc800` (the single `struct_mono.py` edit
above) could have invalidated the prior verdict.

### Target 1 — `helixc/frontend/ast_walker.py`

File unchanged in the diff. `_TYPE_FIELD_NAMES` /
`_NON_NODE_FIELD_NAMES` skip-lists, `_is_ast_node` filter, and the
`visit()` / `generic_visit()` dispatchers are byte-identical to
cycle-22 state. Verdict carries: clean.

### Target 2 — `_op_suffix` and `id(op)` table

`helixc/backend/x86_64.py` unchanged. Construction at
`x86_64.py:835-840`, fallback comment at lines 866-868, fallback at
line 875 all unchanged. Determinism regression test
`test_codegen_determinism.py` unchanged. Verdict carries: clean.

### Target 3 — `_FRESH_COUNTER` reset in `match_lower.py`

`helixc/frontend/match_lower.py` unchanged. `_FRESH_COUNTER[0] = 0`
reset at line 52 and the two pinning regression tests remain
authoritative. Verdict carries: clean.

### Target 4 — Four walker refactors

- `panic_pass.py`, `deprecated_pass.py`, `grad_pass.py` files all
  unchanged in the diff. Their cycle-22 verdicts carry directly.
- `struct_mono.py` — the only changed file — has its body-walk
  pipeline analyzed under Priority 1 above. The `_BodyVisitor` class
  and its five overrides are byte-identical to cycle-22 state. The
  only delta (deletion of the `visit_stmt` shim) is in dead code that
  was never on the live walking path. Cycle-22 verdict on the walker
  refactor carries: clean.

### Target 5 — isize/usize fixes (cycles 17-21)

All eight width-keyed sites enumerated in cycle 22
(`_is_i64_type`, `_is_u64_type`, `wide_widths`, `_INT_BITS`,
`_ptx_type_str`, `_DTYPE_SIZE`, `_DTYPE_PTX_LOAD`, `_ld_reg_prefix`)
live in `helixc/backend/x86_64.py`, `helixc/backend/const_fold.py`,
and `helixc/backend/ptx.py`. None of these files appear in the
`bee36e6..4bdc800` diff. The cycle-22 enumeration is therefore still
exact at HEAD `4bdc800`. Verdict carries: clean.

### Target 6 — Deferred `_rewrite_in_expr` + `_resolve_in_expr`

`helixc/frontend/grad_pass.py` unchanged. The cycle-22 status (known
v0.2 `ASTTransformer` work, not flaggable per task brief) carries
unchanged. Recorded, not flagged.

---

## Audit findings

**Cycle 23 silent-failures audit: CLEAN.**

| Severity   | Count |
|------------|-------|
| CRITICAL   | 0     |
| HIGH       | 0     |
| MEDIUM     | 0     |
| LOW        | 0     |
| **Total**  | **0** |

Key observations:

- The cycle-22 C-fix (`4bdc800`, deletion of the dead `visit_stmt`
  shim) is strictly dead-code removal. AST + grep confirm zero
  callers existed before and after; the body-walk pipeline through
  `visit_expr` → `_body_visitor.visit` is unchanged.
- All 39 `test_struct_mono.py` tests pass at HEAD `4bdc800`.
- The cycle-22 verdicts on Targets 1-6 (ast_walker, `_op_suffix`,
  `_FRESH_COUNTER`, four walker refactors, isize/usize coverage,
  deferred rewriter+resolver) all carry forward because the diff
  between `bee36e6` and `4bdc800` touches only one file
  (`struct_mono.py`) and only one block of dead code within it.
- No new silent-failure surfaces have been introduced since cycle 22.
- No previously-clean surface has regressed.

**Clean-cycle counter:** was 2/5 → **advances to 3/5.**

Two more consecutive clean cycles required to fire the Stage-29 gate.

---

## Out-of-scope per task instructions

- The Stage-29-class "centralize scalar-width predicate" refactor
  recommendation (carried since cycle-17 forward note) is not a
  cycle-23 finding per the no-re-flag rule.
- The v0.2 `ASTTransformer` base class for grad_pass rewriter +
  resolver is recorded as a known limitation per cycle-22, not a
  finding.
- Future-AST-author drift risk on `_TYPE_FIELD_NAMES` is recorded as
  a forward note (cycle 22 Target 1), not a finding.

---

## Files touched by this audit

None — read-only audit cycle. No production-code or test edits.
Only this doc.

## Cross-reference

- Cycle 22 silent-failures (declared CLEAN, advanced 1/5 → 2/5):
  `docs/audit-stage28-8-cycle22-silent-failures.md`.
- Cycle 22 Audit C C0 fix commit: `4bdc800` (deletion of dead
  `visit_stmt` shim).
- Cycle 22 HEAD: `bee36e6`; cycle 23 HEAD: `4bdc800`. Diff scope:
  `helixc/frontend/struct_mono.py` only.
- Test suite verification: `helixc/tests/test_struct_mono.py` (39
  tests passing).
