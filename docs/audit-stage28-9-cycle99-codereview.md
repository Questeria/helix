# Audit Stage 28.9 cycle 99 — Code review

Scope: `HEAD 32c66bf` (cycle-98 audit-clean snapshot, no code change since cycle-97).

Mode: STRICT READ-ONLY. No edits performed.

## Scope (narrow)

1. `helixc/check.py` — warning vs error code paths (cycle-85 verified; brief re-check at HEAD).
2. `helixc/tests/test_typecheck.py` — regression-test discrimination quality.
3. `helixc/tests/test_const_fold.py` — regression-test discrimination quality.
4. TODO/FIXME comments in the three files above with no tracking ID and older than 2026-05.

Prior C1-C98 findings + deferred-known: NOT re-flagged. Parallel stage 28.10/28.11 audits are
INDEPENDENT and not in scope.

## Verdict

**PASS** — 0 findings at confidence >= 75%.

## Observations (informational, sub-threshold)

- **check.py error/warning paths consistent.** Per-stage diagnostic output uses a uniform
  `   stage: N ERROR(s)` / `   stage: N warning(s)` shape, with the `--strict` flag
  uniformly promoting totality + effect-check warnings to rc=1. Error classification is
  tiered: `rc=0` clean, `rc=1` compile error / `--strict` abort / internal bug,
  `rc=2` bad invocation / env (FileNotFound, Permission, IsADirectory, NotADirectory,
  UnicodeDecodeError). The AD-warning drain is wrapped in `try/finally` (C3-3 fix) so
  exception exits still drain `_DIFF_WARNINGS`; the drain itself is wrapped in its own
  try/except so a drain failure does not mask the primary failure (C4-6). The env-error
  helper `_emit_env_error` strips a pre-existing `helixc:` prefix from the inner message
  (C8-2) to avoid `helixc: helixc:` double-prints. No drift since the cycle-85 review.

- **test_typecheck.py is discriminative across 95+ tests.** Each negative test asserts the
  presence of a specific trap-id substring (e.g. `"28604"`, `"24200"`, `"AD002"`,
  `"11001"`, `"74002"`, `"28801"`, `"28802"`, `"28803"`) AND/OR a specific human-readable
  phrase (`"unbound"`, `"does not match return type"`, `"requires effect"`,
  `"declared bool but value is i32"`). Each positive test asserts `errs == []`. The
  pair-shape would catch both silent-pass regressions (negative→empty) and false-positive
  regressions (positive→non-empty). Direct-AST-construction tests (e.g.
  `test_d3_array_size_zero_rejected`, `test_d7_deep_ref_cast_bounded`,
  `test_flatten_impls_rejects_same_name_methods`) exercise checker internals that the
  surface parser cannot reach, ensuring coverage of branches that source code may
  short-circuit upstream. Mock-based tests (`test_c3_3_main_clean_on_exception`,
  `test_c4_6_filenotfound_not_attributed_as_compiler_bug`, etc.) discriminate between
  rc=1 (compiler bug) and rc=2 (env error) and verify _DIFF_WARNINGS is drained on
  exception exits.

- **test_const_fold.py is discriminative across 35+ tests.** The dominant pattern is
  `count_ops(mod, OpKind.X) == 0` after fold — a regression that disables the fold pass
  for an op would produce `count_ops > 0` and fail. NaN/shift traps build IR directly
  (`test_stage17_nan_fold_traps_17001`, `test_stage19_shift_out_of_range_traps_17002`,
  `test_c85_1_shift_bound_uses_result_type_bitwidth`) to exercise the FoldError /
  ShiftFoldError raise paths reliably, bypassing parser-level rejection. The end-to-end
  `test_identity_forwarding_runs_correctly_across_blocks` runs the compiled binary via
  `compile_and_run` and asserts the exit code (42), catching SSA-forwarding miscompiles
  that op-count alone would miss. Two's-complement wrap, C-semantics div/mod, and isize/
  usize 64-bit canonicalization each have dedicated regression tests with explicit
  expected values.

- **No untracked TODO/FIXME in scope.** `Grep -P TODO|FIXME|XXX|HACK` returned zero
  matches in `helixc/check.py`, `helixc/tests/test_typecheck.py`, and
  `helixc/tests/test_const_fold.py`. The eight `TODO` matches elsewhere in `helixc/`
  (backend/x86_64.py:2610 stage30 marker, backend/ptx.py:332 PTX op-handler placeholder,
  frontend/ast_hash.py:392 generic-param recursion note, ir/tile_ir.py:154+219 tiling
  rules, tests/test_codegen.py:1867 negative-exponent skip note, tests/test_ptx.py:140+141
  assertion-strings checking that `// TODO:` does NOT appear in emitted PTX) are
  deferred-known and outside the narrow scope.

## Counter

Cycle 99 PASS → 2/5.
