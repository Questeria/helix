# Stage 28.8 Pre-29 Audit Gate — Cycle 5, Audit C: General Code Quality Review

**Date**: 2026-05-11
**Commit**: HEAD `960303b` (read-only audit).
**Scope**: Cycle-4 fix-sweep commit range `b3504a2..960303b` — five commits closing C4-1..C4-5 + E1..E8 plus the audit-stage5-6 F9 / audit-stage7-8 F12 retroactive fix. Files touched: `helixc/backend/x86_64.py`, `helixc/bootstrap/parser.hx`, `helixc/frontend/{autodiff.py, monomorphize.py, typecheck.py}`, `helixc/tests/{test_codegen.py, test_const_fold.py}`, `docs/lang/trap-ids.md`, and three audit-doc files.
**Method**: Read every commit's diff in full (`git show 31c7912`, `db4055b`, `1b3aa94`, `a59e233`, `960303b`). Walk modified source files at HEAD. Cross-reference every trap-ids.md "Constant name" claim against actual source usage (not just declaration). Verify brace balance for the C4-2 parser.hx 8-arm dispatch insertion. Spot-check regression-test counts against the cycle-3 baseline (16 tests for similar-sized fix sweep).
**Reporting threshold**: confidence ≥ 80 (per cycle-5 audit-C prompt's strict criterion).

---

## Summary

| # | Severity | Confidence | Finding |
|---|----------|------------|---------|
| C5-1 | HIGH | 90 | `TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO` and `TRAP_CAST_MATRIX_RECURSION_DEPTH` defined module-level but never referenced — emit sites still use literal `"28802"` / `"28803"` strings. Audit-time invariant violation. |
| C5-2 | MEDIUM | 82 | Cycle-4 fix-sweep adds 13 fixes (C4-1..C4-5 + E1..E8) with **zero** regression tests. Cycle-3 baseline was 16 tests for comparable scope. |

**Two findings (1 HIGH, 1 MEDIUM). Cycle 5 NOT clean under the strict-zero rule.**

---

## Cycle 5 status

Per user directive 2026-05-10 (strict criterion): cycle counts CLEAN only when zero new findings of any severity at or above the audit threshold (≥ 80).

**Cycle 5 Audit C: NOT CLEAN — 2 findings (0 CRITICAL, 1 HIGH, 1 MEDIUM, 0 LOW).**

---

## Files reviewed

`helixc/backend/x86_64.py`, `helixc/bootstrap/parser.hx`, `helixc/frontend/autodiff.py`, `helixc/frontend/monomorphize.py`, `helixc/frontend/typecheck.py`, `helixc/tests/test_codegen.py`, `helixc/tests/test_const_fold.py`, `docs/lang/trap-ids.md`, plus the three persisted cycle-4 audit-doc files (`audit-stage28-8-cycle4-{codereview,silent-failures,type-design}.md`) for context.

---

## Finding C5-1 — Two new `TRAP_*` constants are dead; audit-time invariant violated (HIGH, confidence 90)

**Location**: `helixc/frontend/typecheck.py` lines 221-222 (defined) vs. lines 591, 596, 606, 611, 2100, 2113 (emit sites).

**Problem**: Commit `a59e233` (cycle-4 audit-C C4-1 fix) added three module-level trap-id constants to address the prior reviewer's HIGH finding that the trap-ids.md "Constant name" column referenced non-existent symbols. Two of the three constants — `TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO` and `TRAP_CAST_MATRIX_RECURSION_DEPTH` — are declared but **never read anywhere in the source tree**:

```python
# helixc/frontend/typecheck.py:221-222
TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO = 28802  # _resolve_size_expr
TRAP_CAST_MATRIX_RECURSION_DEPTH = 28803  # _check_cast_compat
```

Repo-wide grep for `TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO` returns one hit (line 221 — the definition). Same for `TRAP_CAST_MATRIX_RECURSION_DEPTH`. Every emit site uses the literal `"trap 28802"` / `"trap 28803"` substring in an f-string:

```python
# typecheck.py:589-591  (_resolve_size_expr negative-IntLit branch)
self.errors.append(TypeError_(
    f"array size must be > 0, got {expr.value} "
    f"(trap 28802)",                          # literal, not constant
    expr.span,
))
# typecheck.py:594-597  (_resolve_size_expr zero-IntLit branch)
# typecheck.py:604-607  (_resolve_size_expr Unary-NegIntLit negative branch)
# typecheck.py:609-612  (_resolve_size_expr Unary-NegIntLit zero branch)
# typecheck.py:2098-2104 (_check_cast_compat depth-exceeded structured emit)
# typecheck.py:2110-2117 (_check_cast_compat defensive depth-guard)
```

All six emit sites embed the trap number as a literal. The constants are dead.

**Why this is a violation**: `docs/lang/trap-ids.md` lines 94-96 declare two audit-time invariants. The first reads:

> Every `TRAP_*` constant must have at least one caller that actually emits it. Audit C1 cycle 1 found `@trace` reserved 25001 but never invoked it (now fixed in commit c418fb2).

The cycle-1 fix `c418fb2` was specifically called out as the precedent. The cycle-4 audit-C HIGH finding C4-1 noted this very invariant (cycle-4 doc lines 56-58) and proposed two fix paths: *"(a) add the three module-level constants and reference them from the emission sites, or (b) change the `Constant name` column for these rows to `(monomorphize)` / `(typecheck)`."* The cycle-4 fix-sweep chose option (a) but stopped after the declaration half. The "reference them from the emission sites" half never happened.

`TRAP_SHAPE_FOLD_ZERO_DIV` (the third constant, in `monomorphize.py:66`) scrapes by because it is referenced as a class attribute (`ShapeFoldError.trap_id`, line 76) — technically a "caller that emits it" in the read-from-source sense. The other two have no such backstop.

**Why this matters**:

1. The original cycle-4 HIGH finding (a trap-ids.md row claiming a symbol that doesn't exist) is only superficially closed. The symbol exists for `grep -r TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO helixc/` to find — but it grep-points to a line that imports nothing and is imported by nothing. The mnemonic-driven debugging UX is identical to having no constant at all, and *worse* than the cycle-4 reviewer's option (b) would have been (which would at least have signaled the column convention as `(typecheck)`).
2. The invariant grep in step 2 of the "How to add a new trap ID" protocol — *"Comment with the human-readable meaning + the namespace rationale"* — is one half of the contract; the unstated other half is that the constant participate in emission. Otherwise it is purely a documentation artifact and trap-ids.md is the canonical location for documentation, not a comment on a dead Python identifier.
3. Future cycle-N audits running the audit-time-invariants sweep on the registry will catch this — exactly as cycle-1 caught `TRAP_TRACE_OVERFLOW`. The fix-sweep that resolved a HIGH thus regressed into the same class of finding it was meant to close.

**Fix** (one of):

- **(a-complete)** Wire the constants into the emit sites: replace each `f"(trap 28802)"` with `f"(trap {TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO})"` (six sites for 28802, two for 28803). This is what cycle-3 cycle-2 audit-C C2-L2 did for the trace / pytree / panic / unsafe traps and what `autodiff.py:_ad_warn` does for `TRAP_AD_ASSUMED_ZERO`.
- **(b)** Delete the two dead declarations and revert the trap-ids.md rows' `Constant name` columns to the parenthesised `(typecheck)` convention used by rows 28603 / 28604 / 24001 / 24100 / 24200 / 28603 / 28604 / 10030 / 11001 / 16003.

Option (a) better matches the rest of the registry; option (b) is the smaller diff.

---

## Finding C5-2 — Cycle-4 fix-sweep landed 13 fixes with zero regression tests (MEDIUM, confidence 82)

**Location**: Commit `960303b` overall. The diff stat reports `tests/` unchanged for this commit; the only test edits in the b3504a2..960303b range are (i) the audit-stage5-6 F9 codegen test added by `db4055b` (predates the C4-* fixes — it's a retroactive close of a prior audit finding, not a cycle-4 regression test), and (ii) the cycle-4 `31c7912` const-fold dedup (which removes duplicate tests, not adds new ones).

**Problem**: The fix-sweep commit `960303b` documents 13 distinct findings closed (HIGH C4-1, C4-2; MEDIUM C4-3, C4-4, E1, E2, E3/C4-5; LOW E4, E6, E7, E8). Each is a real behavioral change:

- **C4-1**: `_inline_lets` adds three identity arms (`A.Path`, `A.Continue`, `A.TileLit`) — three new code paths in a hot AD pass.
- **C4-2**: `parser.hx` adds a 41-line, 8-arm secondary dispatch table at line 2334 that classifies eight new val_tag families (Binary, Unary, Index, Field, If, Match, Block, UnsafeBlock) as untracked-complex sentinels OR as proven-i32 bools (AST_LT/GT/EQ/NE/LE/GE). Each new arm is a new closure-capture trap-trigger trajectory.
- **C4-3**: `_inline_lets` now recurses through `A.If.cond` (previously dropped).
- **C4-4**: `_compatible` adds two structural arms (`TyTensor` and `TyTile`) with shape/dtype/device/layout/memspace comparison — type-system semantics change.
- **E1**: `_compatible.TyArray` upgrades raw `a.size == b.size` to a recursive `_compatible(a.size, b.size)` fallback.
- **E2**: Logic-binop tie-callback gate broadens from `inner_mismatch` only to `inner_mismatch or (l_is_logic != r_is_logic)`.
- **E3/C4-5**: New public function `monomorphize_safe` wraps `monomorphize` and catches `ShapeFoldError` into a diags list.
- **E6**: `_inline_lets.A.Call` arm now preserves caller turbofish generics on alias substitution.
- **E8**: `_inline_lets.A.Call` arm now walks `A.Field`-typed callees recursively.

The cycle-3 fix-sweep (b3504a2 ancestry) added 16 regression tests for a comparable set of findings — explicit per cycle-4 audit-C axis 3: *"15 new regression tests for cycle-3 findings + the C3-1 test committed earlier… Spot-check of 5 …shows both happy and error paths covered. PASS."*

The cycle-4 fix-sweep's commit message claims *"Tests: targeted suite (autodiff + typecheck + struct_mono) 172 pass."* — those are existing tests. Grepping for `C4-1` / `C4-2` / `C4-3` / `C4-4` / `C4-5` / `E1` / `E2` / `E6` / `E8` in `helixc/tests/` returns zero matches. Grepping for `monomorphize_safe` in `helixc/tests/` returns zero matches.

**Why this matters**:

1. **Regression baseline drift**. The fix-sweep is the documented sole vehicle for closing audit findings during this gate cycle. Cycle-3 set the bar at one-test-per-finding (occasionally more). Cycle-4 lands at zero. A future audit cycle that retroactively flags a regression in any of these 13 fixes will have no test to bisect against; the C3-3 / C3-6 / D6 / D7 cycle-3 tests pinpoint individual fixes by name, whereas cycle-4 fixes are co-mingled in the existing test-suite happy-path coverage at best.
2. **Bootstrap-language risk concentration**. C4-2 is a parser.hx change. Bootstrap-language changes have historically been the highest-risk modifications (per Audit-Stage5-6 F9, Audit-Stage7-8 F4/F12, the cycle-3 D2 fix itself). The new 8-arm dispatch covers eight val_tag families. None of the eight has a dedicated regression test asserting the val_tag → inferred_ty_tag mapping or the downstream trap-76003 firing. The cycle-3 D2 fix that this extends had a dedicated test (`test_codegen.py` D2 case asserting exit 132 on `let pi = get_pi(); let c = |y| y + pi; c(0)`); the natural cycle-4 C4-2 analogue (e.g. `let x = a < b; let c = |y| y + x; c(0)` for the AST_LT bool case, or `let p = foo.bar; let c = |y| y + p; c(0)` for the Field case) is absent.
3. **`monomorphize_safe` semantic drift uncovered**. The new wrapper's docstring asserts *"the caller should treat the diag as a typecheck error and abort the pipeline."* The only caller (`backend/x86_64.py:3025`) does not abort — it prints `warning: fn-mono: {d}` and continues to `grad_pass`, `typecheck`, `hash_cons`, etc. on a `prog` that the failed `Monomorphizer.run()` has mutated in-place (rewriting `item.body` for each fn before the exception). A regression test asserting end-to-end behavior on `fn f() -> [i32; 1/0]` would catch any drift here; none exists.

**Fix**: Add per-finding regression tests in the cycle-5 fix-sweep — one happy-path test per behavioral change. Suggested minimal coverage:

- `test_c4_1_inline_lets_no_warning_on_path_continue_tilelit`: assert `_inline_lets` returns the input unchanged for each of `A.Path`, `A.Continue`, `A.TileLit`, and no `_ad_warn` fires.
- `test_c4_2_*` (eight cases, one per new val_tag family): `let x = <expr>; let c = |y| y + x; c(0)` for each of Binary (e.g. `1+2`), Unary (`-1`), Index (`arr[0]`), Field (`obj.f`), If-expr, Match-expr, Block-expr, UnsafeBlock-expr.
- `test_c4_3_inline_lets_if_cond`: assert `let g = grad(loss); if g(x) > 0.0 { ... }` has `g` substituted in the cond after `_inline_lets`.
- `test_c4_4_compatible_tytensor`, `test_c4_4_compatible_tytile`: dtype-mismatch + shape-mismatch + device-mismatch cases.
- `test_e1_compatible_tyarray_size_via_compatible`: TyArray with TyUnknown size on one side does not false-positive.
- `test_e2_logic_wrap_asymmetric_warns`: `Logic<i32> + i32` (same inner, asymmetric wrap) now warns where pre-fix it was silent.
- `test_e3_monomorphize_safe_catches_shapefolderror`: `fn f() -> [i32; 1/0]` returns `(0, [diag])` with `"28801"` in `diag`.
- `test_e6_inline_lets_call_preserves_generics`: `let g = mk_grad; g::<f64>(x)` retains `::<f64>` after alias substitution.
- `test_e8_inline_lets_call_field_callee`: `let obj = make(); obj.method()` substitutes `obj` correctly.

11 tests total — comparable to the cycle-3 baseline.

---

## Seven axes checked + per-axis verdicts

1. **Code smells in cycle-4 fix-sweep additions** — `monomorphize_safe` has a docstring + rationale; the parser.hx C4-2 dispatch chain has inline AST_TAG → val_tag mapping comments; the autodiff.py C4-1 / C4-3 / E6 / E8 changes each have an audit-cycle-tagged comment block. No copy-paste between C4-1's three identity arms and pre-existing identity arms (each is a distinct three-line `if isinstance` / `return expr`). Magic number `12` (sentinel tag) is documented inline. PASS.
2. **API regressions** — `monomorphize` retained as-is; `monomorphize_safe` is purely additive (no caller of `monomorphize` was rewritten except the `backend/x86_64.py` main block, which is a developer-debug entry, not the production `helixc check` CLI). `_inline_lets` signature unchanged. `_compatible` signature unchanged. `_check_cast_compat` signature unchanged (`_depth` already cycle-3-introduced). PASS.
3. **Test coverage** — FAIL (see C5-2).
4. **Doc-source mismatch** — Two of three new constants are dead; see C5-1. The trap-ids.md "Last updated" header was correctly bumped to 2026-05-11. Row 76003 was correctly extended for the cycle-3 D2 Call-RHS case. The cycle-4 codereview doc itself was correctly persisted to repo. Net: PARTIAL FAIL (only the constant-name dead-link sub-axis fails; rest passes).
5. **Naming consistency** — `monomorphize_safe` matches the existing `_safe` convention used by `kernel_emit_safe` etc. `TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO` matches the `TRAP_*` ALL_CAPS convention. `ShapeFoldError.trap_id` matches the class-attribute precedent set by `DuplicateMethodError`. PASS.
6. **Dead code** — Two TRAP_* constants are dead; see C5-1. Otherwise no new dead code. The `monomorphize_safe` wrapper is technically reachable only via `backend/x86_64.py` main block, but that's a deliberate dev-driver wiring, not dead. PARTIAL FAIL (the dead-constant sub-axis fails; nothing else does).
7. **Brace balance for the C4-2 parser.hx 8-arm dispatch** — Outer chain (lines 2303–2374) opens and closes balance at 38/38 (comments stripped). Inner C4-2 chain (lines 2358–2374) opens and closes balance at 14/14. The 8 arms (`val_tag == 1`, `== 6`, `== 19`, `== 20`, `== 21`, `== 22`, `== 23`, plus the final `else { inferred_ty_tag = 12; }`) chain correctly via `} else { if val_tag == X {` and close via the `};};};};};};};` pattern (7 cascading closes, matching the 7 `else { if` openings). Per-bash-counter and per-pattern-counter both verify. PASS.

---

## What was checked and found below threshold

- **`monomorphize_safe` driver semantics — "warning" + continue, not abort.** The docstring on `helixc/frontend/monomorphize.py:691-702` says the caller "should treat the diag as a typecheck error and abort the pipeline." The only caller (`helixc/backend/x86_64.py:3025-3027`) prints `warning: fn-mono: {d}` and continues into `grad_pass`, `typecheck`, `hash_cons`, totality, lowering, and codegen on a `prog` that `Monomorphizer.run()` had begun mutating (rewriting `item.body` for each FnDecl before the exception). Subsequent passes may emit cascading misleading warnings instead of a single clean abort with the trap-28801 message. **Confidence 75**, below threshold. The blast radius is limited to the developer-debug entry point — `check.py` (the production CLI) does not call fn-mono at all, only `monomorphize_structs`. If the user runs `python -m helixc.backend.x86_64 foo.hx` on a source with `[T; 1/0]`, the trap diagnostic surfaces but the build continues; if subsequent passes happen to typecheck cleanly on the partially-monomorphized AST, the driver could write an `out.elf` despite the trap. Worth tracking but not at threshold.
- **`TRAP_SHAPE_FOLD_ZERO_DIV` near-dead.** Defined at `monomorphize.py:66`; only "use" is as a class-attribute initializer at `ShapeFoldError.trap_id` (line 76). Emit sites at lines 117, 125 still use literal `"trap 28801"`. Technically the invariant is satisfied (one read-from-source caller) but the same lint-pass that flags the other two would flag this if its threshold tightened. **Confidence 65**, below threshold.
- **E6 generics-preservation for `A.Path` callees.** When `cand` is `A.Path` (line 584-585), `new_callee = cand` drops `expr.callee.generics`. `A.Path` has no `generics` field in the AST, so there's nowhere to attach them — the existing behavior matches the pre-fix code for this sub-case. A future Path-with-generics extension would resurface this, but as-is it's not a regression. **Confidence 50**, below threshold.
- **`_compatible.TyTensor` device/layout comparison via raw `==`.** The new arm at lines 2230-2236 uses `a.device == b.device and a.layout == b.layout`. Both are `Optional[str]`, so raw `==` is fine. But this contrasts with the rest of the arm using `self._compatible(...)` for dtype + shape. If `device` or `layout` later becomes a structured type, the raw `==` will silently false-positive. Defensible for now (markers are strings). **Confidence 55**, below threshold.

---

## Open prior findings (not re-flagged this cycle)

Pre-cycle-1 audit-stage5-6, audit-stage7-8, audit-stage9-16: unchanged from cycle-1 baseline (see `audit-stage28-8-cycle1-codereview.md` lines 102-106). Cycle-1 / cycle-2 / cycle-3 / cycle-4 findings: all marked CLOSED by their respective fix-sweep commits except C5-1 (which reopens a sub-issue of cycle-4 C4-1).

---

## Verdict

**Cycle 5 Audit C: NOT CLEAN — 2 findings (0 CRITICAL, 1 HIGH, 1 MEDIUM, 0 LOW).**

Strict-zero rule per user directive 2026-05-10. Cycle counter does not advance.

Suggested fix order:

1. **C5-1** (HIGH, smallest blast radius): wire `TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO` and `TRAP_CAST_MATRIX_RECURSION_DEPTH` into the emit-site f-strings via the `f"(trap {TRAP_*})"` pattern. Eight sites total (six for 28802, two for 28803). Optionally do the same for `TRAP_SHAPE_FOLD_ZERO_DIV` at `monomorphize.py:117, 125` for symmetry. Closes the audit-time-invariant violation.
2. **C5-2** (MEDIUM): add the 11 regression tests listed under the C5-2 fix section to `helixc/tests/{test_autodiff.py, test_typecheck.py, test_codegen.py}`. Brings cycle-4 test-density to the cycle-3 baseline.

After fixes land, rerun cycles A (silent-failure) + B (type-design) + C (this audit) for cycle 6.
