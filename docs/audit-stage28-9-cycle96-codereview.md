# Audit Stage 28.9 cycle 96 — Code review

Scope: HEAD `56fa3df`

Mode: STRICT READ-ONLY. One Write (this doc). Zero edits.

## Narrow scope reviewed

- Cycle-95 fix-sweep diff vs cycle-93 (`git diff 85bece0 56fa3df`) over
  `helixc/frontend/typecheck.py`, `helixc/frontend/parser.py`,
  `helixc/tests/test_autotune.py`.
- `_FLOAT_PRIM_NAMES` docstring at typecheck.py:373-390.
- All consumers of `t.value`-then-split in `helixc/frontend/parser.py`.
- All production-side consumers of `int(...value)` / `float(...value)`
  across `helixc/`.
- Recently-touched tests under `helixc/tests/` (27 files since
  2026-05-09); cycle-92/94/95-tagged test names.

Prior C1-C95 + deferred-known NOT re-flagged. Parallel
Stage 28.10/28.11 NOT in scope.

## Findings at confidence ≥ 75%

None.

### Sub-threshold observations (NOT counted, < 75%)

- **`_FLOAT_PRIM_NAMES` "kept in sync with lexer suffix whitelist"
  is aspirational, not enforceable.** typecheck.py:385-386 asserts
  the rule in prose but no test cross-references lexer.py:338-341.
  If a future suffix is added to the lexer alone, both `IntLit_<new>`
  and `FloatLit_<new>` paths regress silently. However, the cycle-95
  comment is honest about the constraint (it names the lexer line
  numbers), and cycle-94's audit-T already enumerated this gap; this
  is meta-quality, not a defect. Confidence 60.
- **Cycle-95 commit message claims local verification of all 6
  float-suffix variants (fp8/mxfp4/nvfp4/ternary/f16/bf16) but the
  fix-sweep added zero per-suffix regression tests to
  `test_typecheck.py`.** Cycle-94 audit-T explicitly recommended
  `test_c94_intlit_with_fp8_suffix_rejected`, `...mxfp4_...`,
  `...nvfp4_...`. The only kind-coherence regressions are the
  cycle-92 `test_c92_f1_intlit_with_float_suffix_rejected` /
  `..._floatlit_with_int_suffix_rejected` pair, which only probe
  `f32` / `i32`. If `_FLOAT_PRIM_NAMES` regressed by dropping
  `fp8`/`mxfp4`/`nvfp4`/`ternary` again, the heavy gate would not
  catch it. Confidence 70 — partial coverage gap, not a live bug.

### F2 backstop completeness (Parser `t.value.split` consumers)

Audit ran a project-wide grep for `\.value\.split` and
`int\(.*\.value\)` / `float\(.*\.value\)` in `helixc/frontend/`:

- Production parser has **zero** remaining `t.value.split("_")[0]`
  re-parse patterns after the cycle-95 fix.
- All other parser sites that consume integer/float literal token
  data correctly read `t.int_value` / `t.float_value`
  (parser.py:1021, 1089, 1092).
- `helixc/ir/lower_ast.py:1038/1040` consume `IntLit.value` /
  `FloatLit.value` (already-parsed AST fields, not raw lexeme text)
  — same shape, no bug.

No other consumer is subject to the digit-separator collision.

### Test naming clarity

`test_c94_f2_autotune_int_digit_separators_preserved` vs
`test_c94_f2_autotune_int_suffix_still_stripped` — distinct concern
suffixes, no near-duplicate.

### `_FLOAT_PRIM_NAMES` docstring completeness

Mentions the 4 newly-added suffixes (`fp8`, `mxfp4`, `nvfp4`,
`ternary`), references lexer.py:338-341 as the source of truth,
explains the pre-fix defect (raw int bits in float slot). Docstring
is complete on its own terms.

## Result

**PASS** — 0 findings at confidence ≥ 75%.

Two sub-threshold observations (60 / 70) noted for context; neither
crosses the bar and both are partial-coverage rather than live
defects.

**No edits performed by this audit.** One Write (this doc) only.
