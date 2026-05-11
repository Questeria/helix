# Stage 28.8 Pre-29 Audit Gate — Cycle 4, Audit C: General Code Quality Review

**Date**: 2026-05-11 (rerun)
**Commit**: HEAD `31c7912` (cycle-3 fix sweep + cycle-4 const-fold dedup).
**Scope**: All cycle-3 fix-sweep diffs (range `3779270..b3504a2`, 10 commits) plus the cycle-4 `31c7912` dedup commit. Files touched: `helixc/bootstrap/parser.hx`, `helixc/check.py`, `helixc/frontend/{autodiff.py, grad_pass.py, monomorphize.py, struct_mono.py, typecheck.py}`, all four test files, plus `docs/lang/trap-ids.md`.
**Method**: Read each commit's diff in full. Walk modified files at HEAD for context. Spot-check the 16 new regression tests for cycle-3 fixes. Cross-reference every trap-ids.md row added/touched by cycle 3 against actual source-side constant declarations.
**Reporting threshold**: confidence ≥ 75 (per user instructions to this audit run).

---

## Summary

| # | Severity | Confidence | Finding |
|---|----------|------------|---------|
| C4-1 | HIGH | 92 | trap-ids.md rows 28802/28803 name `TRAP_*` constants that do not exist in source |
| C4-2 | MEDIUM | 85 | trap-ids.md "Last updated" header stale (still cites cycle 2) |
| C4-3 | MEDIUM | 82 | `_inline_lets` catch-all double-prints `(trap 85001)` in every emitted warning |
| C4-4 | LOW | 78 | trap-ids.md row 76003 not updated to reflect cycle-3 D2 Call-RHS trigger |
| C4-5 | LOW | 76 | `_inline_lets` catch-all fires spuriously for `A.Path` / `A.Continue` / `A.TileLit` |

**Five findings of varying severity. Cycle does NOT count as clean under the strict-zero rule.**

---

## Cycle 4 status

Per user directive 2026-05-10 (strict criterion): cycle counts CLEAN only when zero new findings of ANY severity at or above the audit threshold.

**Cycle 4 Audit C: NOT CLEAN — 5 findings (0 CRITICAL, 1 HIGH, 2 MEDIUM, 2 LOW).**

The prior revision of this document (committed in `31c7912`) reached a CLEAN verdict by treating the trap-ids.md mismatch as below an 80 threshold and as scoped only to row 28801. This rerun reapplies the per-prompt 75 threshold and extends the cross-reference to all three new rows (28801/28802/28803). Rows 28802 and 28803 have no corresponding source identifier at all — not even an exception class. The doc-source mismatch is therefore a genuine HIGH-severity finding, and four additional sub-threshold-in-prior-revision items round up to ≥ 75.

---

## Files reviewed

`helixc/bootstrap/parser.hx`, `helixc/check.py`, `helixc/frontend/autodiff.py`, `helixc/frontend/grad_pass.py`, `helixc/frontend/monomorphize.py`, `helixc/frontend/struct_mono.py`, `helixc/frontend/typecheck.py`, `helixc/tests/test_autodiff.py`, `helixc/tests/test_autodiff_reverse.py`, `helixc/tests/test_codegen.py`, `helixc/tests/test_const_fold.py`, `helixc/tests/test_struct_mono.py`, `helixc/tests/test_typecheck.py`, and `docs/lang/trap-ids.md` — plus the two cycle-3 audit-finding docs (silent-failures + type-design) for context.

---

## Finding C4-1 — trap-ids.md "Constant name" column references non-existent symbols (HIGH, confidence 92)

**Location**: `docs/lang/trap-ids.md` rows 28801, 28802, 28803 (added in commit `a878709`).

**Problem**: The rows declare `Constant name` values that do not exist as module-level constants in the source:

| Trap | Constant claimed              | Source reality                                                                  |
|-----:|-------------------------------|---------------------------------------------------------------------------------|
| 28801 | `SHAPE_FOLD_ZERO_DIV`        | No such identifier. `monomorphize.py` raises a `ShapeFoldError` exception with the literal string `"trap 28801"`. The mnemonic is at least *informally* recoverable via the exception class name, but no symbol matches the column. |
| 28802 | `ARRAY_SIZE_NEGATIVE_OR_ZERO`| No such identifier anywhere in source. `typecheck.py:_resolve_size_expr` only emits the literal `"trap 28802"` string. |
| 28803 | `CAST_MATRIX_RECURSION_DEPTH`| No such identifier anywhere in source. `typecheck.py:_check_cast_compat` only emits the literal `"trap 28803"` string. |

A repo-wide grep for any of the three uppercase identifiers returns zero matches in source files.

**Why this is a violation**:

1. The registry's own "How to add a new trap ID" protocol (trap-ids.md lines 81-87) explicitly requires step 2: *"Add a `TRAP_*` constant at module-level. Comment with the human-readable meaning + the namespace rationale."*
2. The "Audit-time invariants" section (line 95) states: *"Every `TRAP_*` constant must have at least one caller that actually emits it."* The contrapositive — every named-constant claim in the table must correspond to a real source identifier — is implicit and is being violated.
3. Every other named-constant row in the same table (`TRAP_TRACE_OVERFLOW` 25001, `TRAP_TRACE_EQUIV_SHAPE_MISMATCH` 25002, `TRAP_PYTREE_*` 26001-26003, `TRAP_AUTOTUNE_OVERSIZED` 27001, `TRAP_PARAM_STRUCT_*` 28001-28002, `TRAP_PANIC_*` 28501-28502, `TRAP_UNSAFE_*` 28601-28602, `TRAP_DUPLICATE_METHOD_NAME` 74002, `TRAP_AD_ASSUMED_ZERO` 85001) has a real module-level definition. Auditors and contributors who follow the table to find the source-of-truth constant will hit dead links for these three new rows.

**Why this matters**:

* The cycle-3 commit `a878709` claims the constants exist ("28801 SHAPE_FOLD_ZERO_DIV — division or modulo by zero…"), which suggests the omission was an accidental oversight rather than a deliberate convention break.
* Future collision-detection greps (per the protocol's step 5: *"grep -r '<number>' helixc/"*) will work for the integer, but mnemonic-driven greps will not. The mnemonic is the constant's primary value.
* Cycle-3 audit C did not catch this; cycle-4 audit C (prior revision) flagged only row 28801 at confidence 75 and treated it as below threshold. Both passes missed 28802 and 28803.

**Fix**: Either (a) add the three module-level constants and reference them from the emission sites, **or** (b) change the `Constant name` column for these rows to `(monomorphize)` / `(typecheck)` mirroring the convention used by rows that explicitly do not name a constant (10030, 11001, 16003, 24001, 24100, 24200, 28603, 28604). Option (a) better matches the rest of the registry.

---

## Finding C4-2 — trap-ids.md "Last updated" header stale (MEDIUM, confidence 85)

**Location**: `docs/lang/trap-ids.md` line 3.

**Problem**: The header reads `**Last updated**: 2026-05-10 (Stage 28.8 cycle 2 audit C C2-L2 fix)`. Commit `a878709` on 2026-05-11 added three new rows (28801/28802/28803) without advancing this header.

**Why this matters**: This field is the at-a-glance recency signal for a registry that is otherwise unread by `git log` for casual viewers. A reader who cross-references the header to know when the registry was last touched will be off by a cycle for any future trap-collision investigation.

**Fix**: Update to `2026-05-11 (Stage 28.8 cycle 3 audit — added 28801/28802/28803)` or similar.

---

## Finding C4-3 — `_inline_lets` catch-all double-prints `(trap 85001)` in emitted warnings (MEDIUM, confidence 82)

**Location**: `helixc/frontend/autodiff.py` lines 679-686 (cycle-3 C3-5 catch-all) interacting with `_ad_warn` (lines 95-105).

**Problem**: The cycle-3 catch-all calls:

```python
_ad_warn(
    expr,
    f"_inline_lets fell through on Expr subtype "
    f"'{type(expr).__name__}' — let-bindings beyond this point "
    f"may not be substituted (trap 85001)",
)
```

But `_ad_warn` itself already appends `(trap {TRAP_AD_ASSUMED_ZERO})` (i.e. `(trap 85001)`) at the end of every message it records:

```python
_DIFF_WARNINGS.append(
    f"{line_col}AD: assumed 0 derivative for {kind} ({reason}) "
    f"(trap {TRAP_AD_ASSUMED_ZERO})"
)
```

The rendered warning therefore reads (literal example):

> `AD: assumed 0 derivative for Path (_inline_lets fell through on Expr subtype 'Path' — let-bindings beyond this point may not be substituted (trap 85001)) (trap 85001)`

The trap-id appears twice. Every other call site of `_ad_warn` (six of them — search for `_ad_warn(` in autodiff.py) passes a `reason` string that does NOT pre-embed the trap-id, deferring the trailing trap-id append to `_ad_warn`. The cycle-3 catch-all broke that convention.

**Why this matters**: Cosmetic warning quality — and slightly worse: any future regression test that asserts `len([w for w in warnings if "85001" in w]) == 1` would behave correctly on every other path but fail-by-double-count on the catch-all path. The convention also signals to readers of the code that emitter call-sites should not embed the trap-id; the catch-all is the only counterexample.

**Fix**: Remove the trailing `" (trap 85001)"` substring from the catch-all's `reason` argument. `_ad_warn` will then produce a single, correctly-formatted trap-id at message tail.

---

## Finding C4-4 — trap-ids.md row 76003 not updated for cycle-3 D2 Call-RHS-let trigger (LOW, confidence 78)

**Location**: `docs/lang/trap-ids.md` line 68, row 76003.

**Problem**: The row reads:

> closure capture of non-i32 local OR a local whose type can't be confirmed as i32 (untyped `let x = 3.14;` etc.) — Phase-0 loud failure. Triggers in BOTH typed-non-i32 case (e.g. `let pi: f64 = 3.14`) AND untyped-uninferrable case (`let pi = 3.14_f64` whose type wasn't tracked into var_type_tab).

Cycle 3 D2 (commit `3b321e6` in `helixc/bootstrap/parser.hx`) added a third trigger: untyped Call-RHS lets like `let pi = get_pi(); let c = |y| y + pi;`. The parser now registers Call-RHS as `inferred_ty_tag = 12` (untracked-call sentinel) so the capture-site probe traps.

The new regression test `test_codegen.py` D2 case asserts this behavior (exit 132 = SIGILL from trap 76003 on `let pi = get_pi(); let c = |y| y + pi; c(0)`). But the registry row only describes the literal-RHS uninferrable case. A user investigating why their `let pi = get_pi()` traps will not find a description that matches.

**Why this matters**: User-debugging UX and registry completeness. This is the second consecutive cycle the row has been incrementally extended in source without the doc tracking the change — cycle-3 audit C noted *"row 76003 still describes only non-i32 capture"* at confidence 72 and the cycle-3 D2 fix only partially closed it (by mentioning "untyped-uninferrable case" but only for literal RHS).

**Fix**: Append a clause naming the Call-RHS case to row 76003. Example replacement tail: *"…and untyped Call-RHS lets (e.g. `let pi = get_pi();`) — the parser registers tag 12 'untracked-call sentinel' that the capture probe treats as non-i32 per cycle-3 D2."*

---

## Finding C4-5 — `_inline_lets` catch-all warns on `A.Path` / `A.Continue` / `A.TileLit` (LOW, confidence 76)

**Location**: `helixc/frontend/autodiff.py` lines 679-686.

**Problem**: The cycle-3 C3-5 fix added explicit arms for 20+ `Expr` subtypes plus a catch-all `_ad_warn`. Three existing `Expr` subtypes are not handled by either the leading-literal short-circuit (line 478: `(A.IntLit, A.FloatLit, A.BoolLit, A.StrLit, A.CharLit)`) nor any of the explicit arms:

- `A.Path` (`foo::bar::baz`)
- `A.Continue`
- `A.TileLit`

Of these, `A.Continue` is common in `match` arms and loop bodies; `A.Path` appears as a call-callee or iter-source. `_inline_lets` recurses into `Match.arms[i].body`, `For.iter_expr`, `Loop.body`, etc., where these forms legitimately appear. The catch-all then emits a `(trap 85001)` warning with the message *"let-bindings beyond this point may not be substituted"* — but these node types carry no let-bindings to substitute through.

The catch-all's stated intent (per the inline comment) is *"warn loud so future AST extensions surface immediately rather than silently dropping let-bindings."* The three unhandled existing subtypes will trigger false alarms on normal AD compilation paths — diluting the catch-all's signal value for the case it was designed for.

**Why this matters**: False-positive AD warnings. Diagnostic signal-to-noise ratio. A user differentiating a function that contains `continue` (e.g. a `match` arm short-circuit) will see a confusing warning suggesting let-bindings may have been dropped, when no such loss actually occurred.

**Fix**: Add explicit identity arms for `A.Path`, `A.Continue`, `A.TileLit` (each just `return expr`) before the catch-all. The catch-all then only fires on future AST extensions, restoring its intended signal value.

---

## Seven axes checked + per-axis verdicts

1. **Code smells in cycle-3 additions** — No copy-paste; new helpers (`_widen_canon_name`, `ShapeFoldError`, `_fold_intlit_unary`, `_WIDEN_NAME_ALIASES`) have docstrings or inline rationale. Magic numbers (`8` for ref-cast depth, `12` for the new closure-capture sentinel tag) are inline-commented. The top-level `except Exception` at `check.py:_main_inner` wrapper is appropriate CLI-boundary usage. PASS.
2. **API regressions** — `_check_cast_compat` adds an optional `_depth: int = 0` keyword param (backward-compatible). `_ty_key` now raises `TypeError` on non-`A.TyNode` input; all existing callers pass AST nodes (verified). `monomorphize_structs` return shape unchanged. `_inline_lets` signature unchanged. PASS.
3. **Test coverage** — 15 new regression tests for cycle-3 findings + the C3-1 test committed earlier in `025d55e`. Spot-check of 5 (`test_c3_2_pointer_width_alias_silent`, `test_c3_3_main_clean_on_exception`, `test_c3_6_shape_fold_div_by_zero_traps_28801`, `test_d6_ty_key_raises_on_non_astnode`, `test_d7_deep_ref_cast_bounded`) shows both happy and error paths covered. PASS.
4. **Doc-source mismatch** — Three issues found: C4-1 (constant-name claims), C4-2 (stale Last-updated), C4-4 (row 76003 not extended for D2). FAIL.
5. **Naming consistency** — `_WIDEN_NAME_ALIASES` (ALL_CAPS), `_widen_canon_name` (snake_case underscore-private), `ShapeFoldError` (PascalCase, mirrors `ValueError` parent), `_fold_intlit_unary` (snake_case sibling of `_fold_intlit_arith`) all consistent. `tie_fired = [False]` mutable-flag-via-list matches existing `prov_violations`-list batching idiom. PASS.
6. **Dead code** — `_check_cast_compat._depth > 8` guard is currently unreachable (peel loop fully unwraps `TyRef` first; recursive call depth never exceeds 1) but is documented as explicit defense-in-depth for future non-Ref recursive arms. Defensible. Line 386 of `struct_mono.py` (`return ("?", type(t).__name__)`) is reachable only for `A.TyNode` subclasses without an explicit arm — kept as a forward-compatibility fallback. PASS.
7. **Bundled commit atomicity** — `74b72ec` bundles six typecheck fixes (C3-2 + D1 + D3 + D4 + D7 + D8); hunks are non-overlapping; each fix has its own comment, message paragraph, and regression test. `dccfc7e` (C3-6 + D5 + D9) stays within `monomorphize.py`; `2b15928` (C3-4 + D6 + C3-6 catch) stays within `struct_mono.py`. Bisectability not perfect but acceptable. PASS.

---

## What was checked and found below threshold

- D7 cosmetic side-effect: the rewritten `_check_cast_compat` peels matching `TyRef` wrappers iteratively before recursing, so when the unwrapped inner pair fails the matrix check, the trap-28604 diagnostic renders `_fmt(Foo)` instead of `_fmt(&Foo)` for the source `&Foo as &Bar` case — the `&` prefix is lost in the error message. The existing test `test_c2_6_ref_to_unrelated_ref_traps_28604` only asserts `"28604" in errs` so it still passes, but a user sees `source Foo cannot convert to Bar` instead of the pre-fix `source &Foo cannot convert to &Bar`. **Confidence 73**, below threshold.
- D3 wording: "array size must be >= 0, got -5" implies 0 would be allowed; the next elif disallows 0 separately with a different message. Strictly correct but mildly confusing. **Confidence 55**, below threshold.
- Tie-callback Logic-domain branch: when same-rank-tie fires inside Logic-domain mixing (e.g. `Logic<u32> + Logic<i32>` if their ranks happen to tie), the emitted warning lacks the `[Logic-domain]` tag because `_tie_cb` runs `_ad_warn_mixed_inner` without the Logic suffix and `tie_fired` then suppresses the outer Logic-tagged emit. Edge case (the rank table makes ties rare). **Confidence 65**, below threshold.

---

## Verdict

**Cycle 4 Audit C: NOT CLEAN — 5 findings (0 CRITICAL, 1 HIGH, 2 MEDIUM, 2 LOW).**

Strict-zero rule per user directive 2026-05-10. Cycle counter does not advance.

Suggested fix order (smallest blast radius first):

1. C4-2: bump the trap-ids.md "Last updated" header (one-line edit).
2. C4-4: extend row 76003 description to mention the Call-RHS tag-12 trigger (one-line edit).
3. C4-3: remove the redundant `(trap 85001)` from the catch-all's `reason` argument (one-line edit in `autodiff.py`).
4. C4-5: add identity arms for `A.Path`, `A.Continue`, `A.TileLit` in `_inline_lets` (3-6 lines).
5. C4-1: either define `TRAP_SHAPE_FOLD_ZERO_DIV` / `TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO` / `TRAP_CAST_MATRIX_RECURSION_DEPTH` constants at module level and reference them, or change the three rows' `Constant name` column to `(monomorphize)` / `(typecheck)`.

After fixes land, rerun cycles A (silent-failure) + B (type-design) + C (this audit) for cycle 5.
