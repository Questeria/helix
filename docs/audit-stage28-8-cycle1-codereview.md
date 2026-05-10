# Stage 28.8 Pre-29 Audit Gate — Cycle 1, Audit C: General Code Quality Review

**Date**: 2026-05-10
**Commit**: fc96595 (read-only)
**Scope**: All helixc Python source, focus on stages 22-28.7 (new files never audited). Prior 21 open findings tracked and not re-flagged.

---

## Summary

| # | Severity | Finding |
|---|----------|---------|
| C1-H1 | HIGH | `panic_pass._walk_exprs` uses wrong `A.If` field names (`then_branch`/`else_branch` instead of `then`/`else_`) — `panic()` inside `if/else` silently not detected |
| C1-M1 | MEDIUM | `deprecated_pass.emit_warnings` monkey-patches `A.Program` with undeclared `_deprecation_warnings` — second call overwrites first without indication |
| C1-M2 | MEDIUM | `struct_mono.collect_concrete_uses` skips fn bodies — `let p: Pt<i32>` local-type annotations never trigger instantiation, silently |
| C1-M3 | MEDIUM | `test_ffi.py` hardcodes `C:\` drive letter (pre-existing, open) |
| C1-L1 | LOW | `panic_pass._walk_exprs` passes `ExprStmt`/`Block` nodes to callback — statement nodes are not expressions |

**1 HIGH finding. Cycle 1 NOT clean.**

**Stop-the-line on C1-H1.**

---

## C1-H1 (HIGH) — panic_pass walker misses if/else branches

**File**: `helixc/frontend/panic_pass.py`, lines 60-61
**Category**: silent functional gap

`A.If` declares branches as `then: Block` and `else_: Optional[...]`. The walker looks for `"then_branch"` and `"else_branch"` instead — both `getattr` calls return `None`. `panic()` inside any `if/else` branch is silently skipped by `collect_panics` and `validate_panic_args`.

Contrast: `deprecated_pass._walk_call_sites` (line 93) and `unsafe_pass._walk` (lines 65-67) both correctly use `"then"` and `"else_"`.

**Reproducer**:
```helix
fn maybe_panic(cond: i32) -> i32 {
    if cond == 0 {
        panic("zero is bad");
    }
    1
}
```
`collect_panics(prog)` returns `[]`. Expected: one site.

**Fix**: Replace `"then_branch"` / `"else_branch"` with `"then"` / `"else_"` in panic_pass.py line 60-61. Add regression test for if-arm panic.

---

## C1-M1 (MEDIUM) — deprecated_pass monkey-patches A.Program

**File**: `helixc/frontend/deprecated_pass.py`, line 149

`prog._deprecation_warnings = out` sets an undeclared attribute on the Program dataclass. Second call silently overwrites first.

**Fix (preferred)**: Have `emit_warnings` only return the list; caller stores locally. Don't couple AST data model to analysis-pass output.

---

## C1-M2 (MEDIUM) — struct_mono skips fn bodies

**File**: `helixc/frontend/struct_mono.py`, lines 110-119

`collect_concrete_uses` only walks fn signatures (params, return, struct field tys), NOT fn bodies. So `let p: Pt<i32> = ...` in a body never triggers instantiation. Documented as Phase-0 limitation but the test suite never covers the gap. Idiomatic Helix `let p: Pt<f64> = ...` is a real-world footgun.

**Fix**: Extend `collect_concrete_uses` to walk `Let` stmt `ty` fields in fn bodies.

---

## C1-M3 (MEDIUM) — test_ffi.py hardcodes drive letter

**File**: `helixc/tests/test_ffi.py`, lines 26-28

`_PROJ_ROOT.replace("C:\\", "/mnt/c/")` unconditionally assumes drive C. Breaks on D: drive or Linux. Pre-existing, flagged in prior audits as open.

**Fix**: Derive WSL mount path from `pathlib.Path` drive component dynamically.

---

## C1-L1 (LOW) — panic_pass callback fires on Stmts

**File**: `helixc/frontend/panic_pass.py`, lines 46-47

`callback(node)` called when `hasattr(node, "span")`. `A.ExprStmt` and `A.Block` both have `span`. `_is_panic_call(ExprStmt)` returns False, so no false positives today. But future callback changes could misbehave.

**Fix**: Guard with `isinstance(node, A.Expr)` instead.

---

## What was checked and found clean

- `diagnostics.py` — render_caret, color-strip, did_you_mean all correct
- `trace_pass.py` — buffer overflow logic, trace_equiv comparison, is_traced gate all correct
- `pytree.py` — depth check fires correctly, unflatten defaults, all TyNode keys covered
- `autotune.py` — variant_count edge cases, mangled_variant_name determinism all correct
- `struct_mono.py` (except C1-M2) — arity check, mangling, dedup all correct
- `unsafe_pass.py` — context threading + restore correct
- `deprecated_pass.py` (except C1-M1) — attr parsing, call-site walk correct including if/else
- `check.py` — CLI parse, -O check, -l form variants, -W flags all correct
- @kernel + closures interaction — no conflict (kernels are top-level decls, closures are anonymous)
- 146 new test files use portable paths via `__file__` derivation

## Open prior findings (not re-flagged this cycle)

From audit-stage5-6 (7 open): F2, F4, F9, F10, F11, F12, F13.
From audit-stage7-8 (6 open): F4, F7, F9, F10, F11, F12.
From audit-stage9-16 (5 open): MEDIUM-2, MEDIUM-3, LOW-1, LOW-2, LOW-3.

These remain open — not re-examined this cycle.
