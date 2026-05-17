# Stage 48 Inc 4 Gate-3 Code-Review Audit

Reviewed surface:
- `helixc/frontend/parser.py:1006-1090` (postfix `?` arm)
- `helixc/frontend/typecheck.py` — `__try` dispatch (4516-4661),
  `_result_constructor_provenance` declaration + stewardship sites
  (590-628, 677, 2274, 2400-2462, 2517-2584, 4452-4485, 4628-4646),
  `_check_block` snapshot/restore (2371-2462), `_check_fn` clear
  (2263-2310)
- `helixc/ir/lower_ast.py:830-895` (`_lower_type` Result arm),
  `helixc/ir/lower_ast.py:2010-2117` (identity-lowered tuple +
  `__try` entry)
- `helixc/tests/test_stage48_try.py` (18 tests, full file)
- `docs/stage48-progress-2026-05-17.md` (full file)

## CRITICAL (90-100)
None.

## HIGH (80-89)
None.

## MEDIUM (filtered)
None at confidence >= 80 — see LOW for two genuine but
low-impact polish items.

## LOW (informational, conf 80-85)

**L1 — Test coverage gap: `?` on control-flow primary expressions**
(conf 82). `helixc/tests/test_stage48_try.py` (whole file). No
test exercises the parse-level interaction of `?` with the
parenthesised control-flow forms `(if c { Ok(1) } else { Err(2)
})?` or `({ Ok(1) })?`. The postfix loop at
`parser.py:1060-1075` handles these naturally (primary returns
the parenthesised expr, `?` arm fires), but absence of a pinned
regression means a future refactor of `_parse_primary` could
silently break the pattern. Adding one parse + typecheck assertion
is ~10 lines. Acceptable to defer to Stage 49 alongside
runtime-tag tests (already flagged in gate-2 CR-M3 "2 remaining
deferred").

**L2 — Banner-comment placement for order-sensitivity note**
(conf 80). `helixc/tests/test_stage48_try.py:419-428`. The
"Order-sensitive note on M5 test above" banner sits inside the
gate-3 G3-F1 section header rather than adjacent to the M5 test
it documents. Refers to "M5 test above" which is correct but
non-obvious; moving the comment block to immediately precede
`test_stage48_closure_gate2_m5_cross_fn_no_provenance_carry`
would make the constraint impossible to miss when reordering
tests. Pure polish; gate-3 CR-M3 already chose this placement.

## Findings filtered as below-threshold (logged for completeness)

- Naming `__try`: the M4 gate-2 deferral is sound. The double-
  underscore convention is consistent with `__arena_push`,
  `__strlen`, `__hash_i32` etc. All user-facing diagnostic
  strings (typecheck.py:4554, 4570, 4582, 4605, 4634) use
  `` `?` `` (the operator), never `__try`. The user mental
  model is preserved. (conf 70 — not a real issue)
- Operand-name interpolation degrades gracefully: the
  `isinstance(expr.args[0], A.Name)` guard at
  typecheck.py:4566 produces `""` for non-Name operands so
  `f()?` produces `` `?` requires a Result<T, E> operand... ``
  without a bogus interpolation. Correct. (conf 75 — verified
  clean, no issue)
- `STAGE49_TODO` → `TODO(stage49)` rename (gate-3 CR-L1)
  confirmed complete: 4 sites at lower_ast.py:866,
  lower_ast.py:2097, test_stage48_try.py:404, 410. No stale
  `STAGE49_TODO:` markers remain. The task prompt's claim that
  `STAGE49_TODO:` comments still exist is itself stale — they
  were renamed in this gate. (conf 88 — correctly applied)
- Cross-cutting provenance map: declaration site
  (typecheck.py:590-621) explicitly lists `unwrap_ok /
  unwrap_err / __try` as consumers and enumerates all 6
  stewardship sites. T-M3 "stewardship without centralizing
  helper" already partially addressed. (conf 80 — no further
  action)
- Doc accuracy: stage48-progress-2026-05-17.md sections G3-F1,
  G3-F2, G3-F3, CR-M1, CR-M2 match the implementation as
  landed (snapshot/restore with `_result_let_block_scopes`
  mutation-detection, nested try/finally, operand-name diag
  test). No drift. (conf 85)

CLEAN — gate-3 code-review surfaces no CRITICAL/HIGH issues
and only two LOW polish items both already partially
acknowledged in prior gates. The cascading-defect rhythm
(gate-1 F2 → gate-2 F1 + M5 → gate-3 G3-F1) has converged on
a sound scope-aware snapshot/restore design with exception-safe
finally semantics. Tests pin the regression surface.

3/3 GATE-3 CLEAN — Stage 48 ready to CLOSE.
