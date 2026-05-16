# Stage 35 Restart 61 — Lane B (Compiler/Backend/CLI) Audit Report

**HEAD**: `8f774a4` (Linter test additions before restart 61)
**Date**: 2026-05-16
**Lane**: B (Compiler/Backend/CLI)
**Discipline**: applied audit. Findings landed in commit `c697f3d`.
Lane doc landed retroactively in restart 62.

## Findings (1 MEDIUM + 3 LOW = 4 findings; mixed Family 4 + Family 5)

### B1 MEDIUM — `diagnostics.py _should_color` isatty guard (Family 4 — loud-fail)

`helixc/frontend/diagnostics.py _should_color` wrapped `stream.isatty()`
in a bare `except Exception` returning False. A stream subclass raising
NotImplementedError from isatty would be silently coerced to "no color"
instead of surfacing the loud-fail signal. Mirror of restart 47 B1
narrowing pattern.

**Fix**: narrow to `(AttributeError, OSError, ValueError)` after the
standard re-raise prelude
`(NotImplementedError, AssertionError, KeyboardInterrupt, SystemExit, MemoryError)`.

**Canary**: `test_stage35_restart61_diagnostics_isatty_narrowed_to_stream_failures`.

### B2 LOW — `check.py` argv parser double `-o` / empty `-o` (Family 5 — bookkeeping)

Pre-fix, double `-o` silently overwrote the first output path with no
warning; empty `-o ""` produced a confusing OSError on the atomic-write
layer.

**Fix**: reject duplicate `-o` with `error: check: duplicate -o flag` (rc=2);
reject empty `-o` argument similarly.

**Canaries**: `test_stage35_restart61_check_rejects_duplicate_dash_o`,
`test_stage35_restart61_check_rejects_empty_dash_o`.

### B3 LOW — `examples/run.py` `-h` / `--help` flag (Family 5 — bookkeeping)

The runner had no help discoverability. Mirror of restart 49 B2 four-CLI
help support.

**Fix**: add `-h` / `--help` flag with usage banner.

**Canary**: `test_stage35_restart61_examples_run_help_flag_works`.

### B4 LOW — `monomorphize.py _mangle_expr` dead try/except (Family 5 — bookkeeping)

The handler around `structural_hash` claimed it added "mangle-site
breadcrumb" but immediately re-raised without decoration — a no-op
that implied safety it did not provide.

**Fix**: remove the dead try/except.

**Canary**: `test_stage35_restart61_monomorphize_structural_hash_dead_try_removed`.

## CLEAN spot-checks (no findings)

- `helixc/backend/x86_64.py`, `ptx.py` — verified `except Exception:`
  sites all have proper re-raise preludes.
- `const_fold.py` — verified int-arith / float-arith / bitwise blocks
  have the re-raise prelude (restart 51 B4 sibling sweep).
- `lower_ast.py _resolve_monomorphized_struct_type` — restart 47 B1 still
  intact; `_lower_type` sentinel returning verified clean.
- Flag parity across check / x86 / ptx / autodiff_cli — verified.
- Banner accuracy + `-h`/`--help` support across all four CLIs —
  verified.

## Carry-forward to restart 62

None — Lane B clean for restart 62 (no new findings on top of c697f3d).
