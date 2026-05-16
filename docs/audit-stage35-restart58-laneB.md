# Stage 35 Restart 58 — Lane B (Compiler/Backend/CLI) Audit Report

**HEAD**: `c8398d3`
**Date**: 2026-05-16
**Lane**: B (Compiler/Backend/CLI)
**Discipline**: read-only.

## Lane verdict

**CLEAN** — 0 findings (H=0 M=0 L=0) across all 6 audited families.

This is the third consecutive Lane B clean window (R55 = 0, R56 = 0,
R57 = bookkeeping only, R58 = 0). The restart 47–54 sweep covered the
practical regression surface for `except Exception` discipline,
atomic-write contracts, banner / parser flag parity, and CLI exit-code
conventions; nothing has regressed.

## Clean families swept

1. **Stale-artifact / bad-invocation cleanup**: all backend CLIs clean
   the output path before `sys.exit(2)` on unknown-flag errors;
   `helixc.check` error paths exit before the output-write block. No
   new CLI return paths since restart 47.
2. **Partial-write atomicity**: all five file-writer sites use the
   canonical `mkstemp + os.replace + on-failure cleanup` pattern
   (restart 46 B4/B5 + restart 47 B2 precedents). No new file writers
   in `helixc/**/*.py` or `examples/**/*.py` since restart 47.
3. **Backend flag-mismatch / parity**: parity verified across `check`,
   `backend.x86_64`, `backend.ptx`, and (where applicable)
   `frontend.autodiff_cli` for `-O0..-O3`, `--no-opt`, `-Wad`,
   `-Wdeprecated`, `-l<lib>`, `--no-color`/`--color`, `--hash`,
   `--hash-cons`, `--strict`, `--stdlib`/`--no-stdlib`, `-h`/`--help`.
4. **CLI exit code conventions**: all four CLIs match the documented
   `rc=2 bad invocation, rc=1 compile error, rc=0 clean` contract.
5. **Parser/typechecker/codegen silent fallbacks**: all `except Exception`
   blocks in scope are guarded with the `(NotImplementedError,
   AssertionError, KeyboardInterrupt, SystemExit, MemoryError): raise`
   precedent (restart 46/47/48/51/53/54 sweeps). The two
   `validate_kernel_tile_lowering` blocks in `check.py` are
   intentionally bare per the documented contract (restart 51 B2 +
   restart 53 B2 — NOT flagged).
6. **Bootstrap parser drift vs Python parser**: `frontend/parser.py` has
   had no commits introducing new metadata kinds since `e7c05bf`;
   `bootstrap/parser.hx` stays Stage-33 aligned.

## Bookkeeping note

This Lane B audit was performed as part of the restart 58 catch-up
sweep (Increment 77) on the same HEAD that c8398d3 produced. Lane B
returned no source-fixable findings, but the catch-up sweep still wrote
this stub doc + the Lane A doc + the Lane C doc + Increment 77 so the
abbreviated-source-only-commit anti-pattern (restarts 52/55/56/58) is
no longer cumulative.
