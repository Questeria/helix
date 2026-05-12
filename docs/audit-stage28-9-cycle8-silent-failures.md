# Stage 28.9 Cycle 8 — Audit A (silent failures)

**Date:** 2026-05-11
**HEAD:** `fdbcfc5` (post C7-1 TileLit walker arm)
**Lens:** silent failures (Audit A)
**Criterion:** ZERO findings at confidence >=75% (strict).

## Scope

Delta since cycle-7 baseline `f24cf15` is a single commit
`fdbcfc5` (the C7-1 fix). `git diff f24cf15..fdbcfc5 -- helixc/`
= +81 lines across `frontend/match_lower.py` (+13: TileLit arm)
and `tests/test_match.py` (+68: regression test). No other
production-pass surface changed. Prior cleared inventory is
re-verified, not re-flagged.

## Verification of the cycle-8 delta

**match_lower.py TileLit arm (lines 210–222).** Descends into
`shape` (list[Expr]) and `memspace` (Expr); `dtype` is `TyNode`
and `init` is `str`, so no Expr child is missed. Pattern mirrors
the cycle-23 C22-C arms (UnsafeBlock/Range/Modify/Break/Quote/
Splice) and the C4-1 Assign.target arm. Comment cites the loud
diagnostic chain: `_tile_shape_dims` gate emits the user-visible
error; the arm prevents the deeper Match-assertion from being
the failure mode. Failure remains loud and observable either
way — no silent fallback, no broad except, no default return.

**test_match.py +68.** Direct-AST regression that asserts no
`A.Match` node survives `lower_matches` when nested in
`TileLit.shape`. Walker traverses via `__dict__` so any future
Expr-child addition to TileLit that lacks a walker arm will
also trip the assertion. Test failure mode is an explicit
`AssertionError` with C7-1 context — well-surfaced.

## Smell re-sweep (helixc/, full surface)

- Bare `except:` — 0.
- `except Exception:` — 11 production hits (check.py x6,
  const_fold.py x4, lower_ast.py x1, diagnostics.py x1); all
  cleared in cycles 3–5, inventory unchanged.
- `pass # (ignore|skip|TODO|fixme)` — 0.
- New code in `match_lower._rewrite_expr` TileLit arm: no try/
  except, no None-coalesce, no fallback path.

## Findings

**None at confidence >=75%.**

| Severity | Count |
|---|---|
| CRITICAL | 0 |
| HIGH | 0 |
| MEDIUM | 0 |
| LOW | 0 |
| **Total** | **0** |

**Cycle 8 silent-failures audit: CLEAN.** C7-1 fix is a pure
walker extension with no error-handling surface; the cleared
inventory holds. Strict >=75 criterion met.

## Files touched by this audit

None — read-only. Only this doc.
