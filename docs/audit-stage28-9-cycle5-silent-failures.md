# Stage 28.9 Cycle 5 — Audit A (silent failures)

**Date:** 2026-05-11
**HEAD:** `f7e7b02` (one commit ahead of cycle 4 baseline `dd2bc76`)
**Lens:** silent failures (Audit A)
**Criterion:** ZERO findings of ANY severity at confidence >=75%.

## Scope

Read-only stability re-pass over the new surface only.
`git diff dd2bc76..f7e7b02 -- helixc/` is a single 7-line addition to
`helixc/frontend/match_lower.py` (C4-1 fix: `_rewrite_expr` now also
traverses `Assign.target` so `arr[match x { ... }] = v` desugars
before lowering). No other helixc/ file changed since cycle 4.

## Verification

### New surface (match_lower.py _rewrite_expr)

The new line `expr.target = _rewrite_expr(expr.target)` is symmetric
with the pre-existing `expr.value = _rewrite_expr(expr.value)` arm.
`Assign.target` is typed `"Expr"` (ast_nodes.py:329-333). The walker
itself has no `try/except`, no fallback values, no logging — unknown
nodes return unchanged via the trailing `return expr` (same behaviour
as every other miss in this file; not new, not a silent failure since
any persisting `Match` is trapped by lower_ast's "Match should not
reach _lower_expr" assertion downstream).

### Walker-coverage cross-check (defense-in-depth)

Enumerated all `Expr` subclasses with `Expr`/`Block` children and
diffed against `isinstance(expr, A.X)` arms in `_rewrite_expr`. After
the cycle-5 fix, only **TileLit** is uncovered. TileLit's
`shape: list[Expr]` and `memspace: Expr` are produced by
`_parse_tile_type` (parser.py:1258-1296), which is invoked only inside
the `tile<...>::zeros()/::ones()` primary-expression form. The grammar
restricts shape entries to type-position dim expressions and memspace
to a name marker — a `match` token cannot reach those positions. Not
exploitable; **not** flagged.

### Smell sweep (helixc/ Python, full surface)

- `except:` (bare) — 0 hits.
- `except Exception:` — 11 hits in 4 production files
  (`check.py` x5, `ir/passes/const_fold.py` x4, `ir/lower_ast.py` x1,
  `frontend/diagnostics.py` x1). All preexisting; identical to cycle 4
  CLEAN baseline. (`ir/lower_ast.py:2149` was triaged at cycle 27.)
- `pass # (ignore|skip|TODO|fixme)` — 0 hits.

## Findings

**None at confidence >=75%.**

| Severity | Count |
|---|---|
| CRITICAL | 0 |
| HIGH | 0 |
| MEDIUM | 0 |
| LOW | 0 |
| **Total** | **0** |

**Cycle 5 silent-failures audit: CLEAN.** The C4-1 closure is a pure
walker-coverage addition; it neither swallows errors nor masks any
downstream signal.

## Files touched by this audit

None — read-only. Only this doc.
