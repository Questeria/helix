# Audit Stage 28.9 cycle 98 — Code review

Scope: `HEAD 1ff41ff` (cycle-97 fix-sweep at `3b065d2`).

Mode: STRICT READ-ONLY. No edits performed.

## Verdict

**PASS** — 0 findings at confidence >= 75%.

## Scope (narrow)

1. Cycle-97 comment/code drift in `x86_64.FnCompiler._is_float_type` + `_check_float_supported`.
2. Cycle-97 A.Loop `append_block` fix in `lower_ast.py`.
3. Discriminativeness of regression test `test_c96_loop_blocks_appended_to_fn_blocks`.
4. Other `new_block()` direct calls in `lower_ast.py` outside `begin_function`'s entry block (orphan-block hazards).

Prior C1-C97 + deferred-known: NOT re-flagged.

## Findings

None at conf >= 75%.

## Observations (informational, sub-threshold)

- **x86_64 float tuple aligned with typecheck.** Cycle-97 `_is_float_type` and `_check_float_supported` list `f16, bf16, f32, f64, fp8, mxfp4, nvfp4, ternary` — same 8-name set as `frontend/typecheck.py::_FLOAT_PRIM_NAMES` (line 387). Comment ("all 8 float-domain suffixes the lexer accepts") matches code. No drift.

- **A.Loop fix consistent with For/While idiom.** `lower_ast.py:1909-1910` now uses `append_block()` for header + body, matching For (1813-1815) and While (1873-1875). No remaining functional `new_block()` calls in `lower_ast.py` — the only match (line 1902) is the cycle-97 comment referencing the pre-fix code. Verified via `Grep new_block\(` over the file.

- **No other orphan-block hazards in lower_ast.py.** Sole legit `new_block()` caller is `tir.Builder.begin_function` (`tir.py:388`), which constructs the entry block and bakes it into `fn.blocks=[entry]` at function creation — not an orphan. All AST control-flow arms (If, For, While, Loop) use `append_block()`.

- **Regression test is discriminative.** `test_c96_loop_blocks_appended_to_fn_blocks` asserts two independent invariants: (a) `len(fn.blocks) >= 3` (entry + header + body) — pre-fix this would be 1 because orphaned blocks never reach `fn.blocks`; (b) every BR `target_block` attr references an id in `{b.id for b in fn.blocks}` — pre-fix the header→body and body→header BRs targeted ids absent from `fn.blocks`. Either check alone would catch the cycle-96 defect; both together also catch regressions where blocks are appended but BR targets get rewritten incorrectly. Test passes post-fix (verified locally, 0.58s).

## Counter

Cycle 98 PASS → 1/5.
