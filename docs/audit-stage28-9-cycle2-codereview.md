# Stage 28.9 Cycle 2 — Audit C (codereview)

**Date**: 2026-05-11
**HEAD**: post `dd2bc76` (Stage 28.9 cycle-1 audit-A F1 sticky overflow flag landed)
**Lens**: code review (Audit C)
**Strict criterion**: ZERO findings of ANY severity at confidence ≥ 80.

---

## Scope

Cycle-1 codereview returned 2 findings (stale `25001` comment + missing `trace_pass` regression test); both fixed in `9326fc7`. Cycle 1 also dispatched 4 more fixes (D1 rename `477f025`, F3 dep_tab `0888dfb`, F2 walker tuple-lit `50eeef0`, F1 overflow sticky flag `dd2bc76`). This cycle audits the integrated state.

Reviewed all 5 cycle-1 fix commits across 7 standard axes: correctness, security, project guidelines, error handling, testing, naming, platform compatibility.

---

## Findings

**None at confidence ≥ 80.**

| Severity | Count |
|---|---|
| CRITICAL | 0 |
| HIGH | 0 |
| MEDIUM | 0 |
| LOW | 0 |
| **Total** | **0** |

**Cycle 2 codereview: CLEAN.**

---

## Verification details

- **`_is_fn_traced` / `_current_fn_name` save-restore** in `lower_ast.py` (lines 485-491, 589-591) correctly brackets `_lower_fn_body` and restores before `end_function()`. Nested fn lowering (not currently reachable but defensively correct) would not corrupt state.
- **`dep_tab_add` returning 0** is checked at every call site (line 2768 in `kovc.hx`) and the 28702 diag is emitted exactly once per dropped name (severity-1, not severity-2), matching the Python pass's warning-only policy.
- **Sticky-overflow flag at `diag_state + 2 + cap * 4`** does not alias any valid entry slot. Entries occupy `diag_state + 2` through `diag_state + 2 + (cap-1)*4 + 3`, so the flag is safely beyond the last entry.
- **AST_TUPLE_LIT arms in both `walk_for_panic` and `walk_for_deprecated`** use an identical chain-walk pattern. `walk_for_unsafe` is intentionally a no-op stub (documented) and does not need the arm.
- **Python-side ASTVisitor** already covers `TupleLit.elems` via dataclass-field introspection (`elems` is not in `_NON_NODE_FIELD_NAMES` or `_TYPE_FIELD_NAMES`), so no explicit case is needed there.
- **`test_trace.py` `test_trap_25001_reserved`** tests both `TRAP_TRACE_OVERFLOW` and `TRAP_TRACE_EQUIV_SHAPE_MISMATCH` — the prior cycle-1 gap is closed.

Pending cycle 2 Audit A.
