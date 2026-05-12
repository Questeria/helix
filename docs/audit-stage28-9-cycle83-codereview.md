# Audit Stage 28.9 cycle 83 — Code review

Scope: HEAD 42f4e11 (Stage 28.9 at 1/5 after cycle-80 reset; parallel Stage 28.10 commits independent).
Mode: STRICT READ-ONLY. No edits to source/tests were performed.
Inputs reviewed: helixc/tests/test_ir.py, helixc/tests/test_ffi.py, helixc/tests/test_totality.py, helixc/tests/test_deprecated.py, helixc/check.py; commit messages for cycles 77..82 cross-referenced against tree state at HEAD.

## Result: FAIL — 1 finding at conf >= 75%

## Findings

### CR-1 (HIGH, conf 90) — test_ffi.py runner drift: new regression test silently skipped

Cycle-79 commit d218e65 added `test_c76_1_ffi_call_routes_f32_args_to_xmm0` to `helixc/tests/test_ffi.py` (line 111) as a regression for the SysV float-class FFI routing fix. Cycle-81 commit 7b13010 strengthened the test for discrimination. The commit message of d218e65 claims "Heavy gate post-fix: 1509 passed (+1 from C76-1 FFI test)."

However, `test_ffi.py`'s `__main__` block (lines 181-203) uses a **static, hand-maintained list of three tests**:
```
tests = [
    ("test_extern_c_puts_hello", test_extern_c_puts_hello),
    ("test_extern_c_no_op_no_dynlink", test_extern_c_no_op_no_dynlink),
    ("test_extern_c_uses_dynlink", test_extern_c_uses_dynlink),
]
```
The new `test_c76_1_ffi_call_routes_f32_args_to_xmm0` was never added to this list. Other test files in the same suite (test_ir.py, test_totality.py) use dynamic `globals()` enumeration so newly added `test_*` functions are auto-discovered; test_deprecated.py routes through `pytest.main([__file__])`. test_ffi.py is the only one in the cycle-83 scope using a hard-coded list.

The official test runner `scripts/run_all_tests.sh` invokes each file as `python <test.py>` and parses the trailing `N passed, N failed` line, so under the documented heavy gate the new regression is silently skipped. Empirical verification at HEAD: `python helixc/tests/test_ffi.py` prints exactly `3 passed, 0 failed` — the +1 the commit message asserts is not actually present in the gate path the script runs. A regression to all-INT FFI routing would not be caught by the documented gate; the test only fires when pytest is invoked directly on the file (a separate, undocumented path).

This is genuine commit-message-vs-code drift in the scope explicitly requested (cycles 77..82 vs actual code), not a deferred-known item.

## Items verified clean

- test_ir.py: dynamic globals enumeration; cycle-77/79/81 regression `test_c76_f1_for_range_i64_increment_dtype_matches_iterator` IS exercised by the default runner.
- test_totality.py: dynamic globals enumeration; cycle-58 C57-1 mod/impl regressions exercised; no stale or duplicate cases.
- test_deprecated.py: `pytest.main([__file__])` discovers all tests; the cycles 59-74 regression suite is dense but each test asserts a distinct property (intra-mod alias, cross-mod non-capture, ImplBlock target mangling, StructLit name remap top-level-vs-mod, double-descent self-call count). No duplicates spotted.
- helixc/check.py: warning/error code paths consistent — AD-warning drain is universal via outer try/finally; totality `--strict` aborts with rc=1; deprecated `-Wdeprecated=error` promotes to rc=1; panic/unwind/trace/unsafe/autotune all fail-closed; effect-check classification delegated to effect_check.report_diagnostics with `--strict` gating on hard_count; `--check-only` short-circuit position (before lower) is intentional and documented in-line.
