# Stage 35 Restart 62 — Lane B (Compiler/Backend/CLI) Audit Report

**HEAD**: `c697f3d` (Fix Stage 35 sixty-first restart findings)
**Date**: 2026-05-16
**Lane**: B (Compiler/Backend/CLI)
**Discipline**: combined audit-and-fix.

## Findings: CLEAN

Lane B has been the consistently-clean lane since restart 58 (now five
consecutive clean windows: 58 catch-up, 59, 60, 61, 62). The
Compiler/Backend/CLI audit surface is approaching exhaustion under the
patterns established in restarts 46-61.

## CLEAN spot-checks

- `helixc/check.py` argv parser — restart 61 B2 closed double `-o` +
  empty `-o`. No new bad-invocation surfaces detected.
- `helixc/backend/x86_64.py`, `ptx.py` — `except Exception:` sites all
  have the re-raise prelude. Banner/help support verified.
- `helixc/ir/passes/const_fold.py` — int-arith / float-arith / bitwise
  blocks have re-raise prelude (restart 51 B4).
- `helixc/ir/lower_ast.py _resolve_monomorphized_struct_type` — restart
  47 B1 still intact.
- `helixc/frontend/diagnostics.py _should_color` — restart 61 B1 closed.
- `helixc/frontend/monomorphize.py _mangle_expr` — restart 61 B4 cleaned
  dead try/except.
- `helixc/frontend/autodiff_cli.py` — exit codes restart 49 B1, help
  banner restart 49 B2, hash flag restart 47 B6.
- Flag parity across check / x86 / ptx / autodiff_cli — restarts
  47-49 closed.
- check.py `-o /dev/null` corner case (Linux/WSL) — verified harmless;
  the atomic-write layer treats `/dev/null` as a valid path,
  `os.replace` is a no-op equivalent on null devices on most platforms.
  Not a real corner-case bug.
- PTX backend determinism — verified at restart 47 with the
  fixed-iteration-order kernel emission test
  (`test_stage35_emit_ptx_deterministic_kernel_order`).

## Carry-forward to restart 63

None.
