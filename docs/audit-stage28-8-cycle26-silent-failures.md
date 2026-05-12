# Stage 28.8 Pre-29 Audit Gate — Cycle 26 (Audit A: silent failures)

**Date:** 2026-05-11
**HEAD:** `6db467f` ("Audit 28.8 cycle 23+: close C22-C (HIGH, match_lower
walker drift)")
**Lens:** silent failures (Audit A)
**Streak counter at start:** 3/5 (cycles 23 + 24 + 25 all CLEAN).

> Note: HEAD is identical to the cycle-25 audit HEAD. No commits have
> landed since `6db467f`, and `git status --short helixc/` is empty.
> The production-code surface is byte-identical to the cycle-25 state.

---

## Scope

Strict-criterion read-only stability re-pass. Per the cycle-26 brief, no
re-flagging of cycle 1-25 findings, no manufacturing of findings.

Diff vs. cycle-25 baseline (`6db467f..HEAD` in `helixc/`):

| Files changed | 0 |
| Lines added   | 0 |
| Lines removed | 0 |

No production code, no test code, and no doc has changed since cycle 25
declared CLEAN. The audit is therefore a confirmation that all
previously-cleared surfaces remain byte-identical.

---

## Stability verification

### Cycle 25 cleared surfaces — all unchanged at `6db467f`

| Surface                                          | Diff | Status   |
|--------------------------------------------------|------|----------|
| `frontend/match_lower.py` (6 walker arms)        | 0    | Clean    |
| `ir/passes/effect_check.py` (OP_EFFECTS + FFI)   | 0    | Clean    |
| `frontend/ast_walker.py` (visit/generic_visit)   | 0    | Clean    |
| `backend/x86_64.py` (_op_suffix, isize/usize)    | 0    | Clean    |
| `ir/passes/const_fold.py` (_INT_BITS)            | 0    | Clean    |
| `backend/ptx.py` (_ptx_type_str + width tables)  | 0    | Clean    |
| `frontend/struct_mono.py` (_BodyVisitor)         | 0    | Clean    |
| `frontend/panic_pass.py` (walker refactor)       | 0    | Clean    |
| `frontend/deprecated_pass.py` (walker refactor)  | 0    | Clean    |
| `frontend/grad_pass.py` (walker refactor)        | 0    | Clean    |

### Silent-failure smell sweep

Repeated the canonical smell sweep across `helixc/`:

- `except\s*:` (bare except) — **zero hits** (production + tests).
- `except\s+Exception\s*:\s*$` (catch-all, no body) — **zero hits**.
  All `except Exception:` instances in production code paths
  (`ir/lower_ast.py:2149`, `frontend/diagnostics.py:76`,
  `ir/passes/const_fold.py:{257,331,356,408}`) carry explicit fall-back
  semantics flagged and accepted in prior cycles (const-prop probe
  guards, diagnostics best-effort source-line lookup, bigint over-range
  guard). The 9 occurrences in `tests/test_match.py` and 2 in
  `tests/test_codegen.py` are negative-path assertions inside test
  helpers — out of production scope.
- `pass\s*#\s*(ignore|skip|TODO)` (silent stub) — **zero hits**.
- `.get(.., 0)` / `.get(.., None)` silent-decay sites — no new entries
  since the cycle-22 forward note on `ptx.py:350-353` (Stage 29 scalar-
  width predicate refactor target).

No new silent-failure smell exists anywhere in `helixc/`.

---

## Test verification

At HEAD `6db467f`:

| Suite                          | Tests | Status     |
|--------------------------------|-------|------------|
| `test_match.py`                | 25    | all pass   |
| `test_effect_check.py`         | 22    | all pass   |
| `test_codegen_determinism.py`  | 7     | all pass   |
| Combined (audit-scope)         | **54**| all pass   |

Every test exercising a previously-flagged silent-failure surface
remains green. Total wall time 22.62s.

---

## Audit findings

**Cycle 26 silent-failures audit: CLEAN.**

| Severity   | Count |
|------------|-------|
| CRITICAL   | 0     |
| HIGH       | 0     |
| MEDIUM     | 0     |
| LOW        | 0     |
| **Total**  | **0** |

Key observations:

- Zero production-code delta since the cycle-25 CLEAN baseline. No new
  commits have landed; the working tree is clean for `helixc/`.
- All cycle 1-25 cleared surfaces remain byte-identical.
- Silent-failure smell sweep returns zero new hits.
- 54 audit-scope tests pass.

**Clean-cycle counter:** was 3/5 → **advances to 4/5.**

One more consecutive clean cycle required to fire the Stage-29 gate.

---

## Out-of-scope per task instructions

Forward notes carried from prior cycles (not re-flagged):

- Effect-label docstring drift in `effect_check.py:15-22` (carry from
  cycle 24).
- Stage-29-class "centralize scalar-width predicate" refactor for
  `ptx.py:350-353` (carry from cycle 17).
- v0.2 `ASTTransformer` base class for `grad_pass._resolve_in_expr`,
  `grad_pass._rewrite_in_expr`, `match_lower._rewrite_expr` (carry from
  cycle 22, status Stage 28.8.2).
- Future-AST-author drift on hand-rolled walkers (carry from cycle 22).

---

## Files touched by this audit

None — read-only. Only this doc.

## Cross-reference

- Cycle 25 silent-failures (declared CLEAN, advanced 2/5 → 3/5):
  `docs/audit-stage28-8-cycle25-silent-failures.md`.
- Cycle 24 silent-failures (declared CLEAN, advanced 1/5 → 2/5):
  `docs/audit-stage28-8-cycle24-silent-failures.md`.
- Cycle 23 silent-failures (declared CLEAN, advanced 2/5 → 3/5 then
  reset under brief renumber): `docs/audit-stage28-8-cycle23-silent-failures.md`.
- Cycle 25 HEAD: `6db467f`; cycle 26 HEAD: `6db467f` (identical).
- Production-code delta scope vs. cycle-25 baseline: **none**.
- Test suite verification: 54 audit-scope tests pass.
