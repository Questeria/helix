# Lane B Audit Report — Stage 35 Restart 52

**HEAD**: `a4ad9a0 Fix Stage 35 fifty-first restart findings`
**Scope**: Compiler / backend / CLI. Read-only audit; fixes applied separately.

## Summary

Reviewed `helixc/check.py`, `helixc/backend/x86_64.py`, `helixc/backend/ptx.py`, `helixc/frontend/autodiff_cli.py`, `helixc/ir/lower_ast.py`, and all `helixc/ir/passes/*.py`. Restart 51 closed B1-B4 (autodiff_cli unknown-flag, check.py codegen re-raise sites, const_fold sibling sweep). Confirmed all of those remain in place. Found **0 new findings**.

## Clean families swept

- Stale-artifact cleanup: every bad-invocation early-return in `helixc.check` and the direct backend drivers clears stale `-o`. Clean.
- Partial-write atomicity: all file-writing sites use `tempfile.mkstemp + os.replace + BaseException cleanup`. Clean.
- Backend flag parity (`-O0..-O3`, `--no-opt`, `-l`, `--no-color`/`--color`, `--hash`/`--hash-cons`, `-Wad`, `-Wdeprecated`, `--strict`, `--stdlib`/`--no-stdlib`): all three CLI drivers agree. Clean.
- Silent-fallback exceptions: restart 51 B2/B3 + B4 sibling sweep closed `check.py:1849/1872/1834` and `const_fold.py:484/525/631`. `lower_ast.py:660 / 3097` narrowed. `autodiff_cli.py:63 / 145` re-raise. `ptx.py:1009 / 1055` re-raise. `validate_kernel_tile_lowering` intentionally keeps `except Exception` (codified by `test_stage35_emit_ptx_reports_tile_lowering_error_without_bug_label`). Clean.
- Help / banner completeness: all four CLIs (`check`, `backend.x86_64`, `backend.ptx`, `frontend.autodiff_cli`) print banners that enumerate every flag the parser accepts. Clean.
- Bootstrap parser drift: `helixc/_bootstrap/parser.hx` metadata kinds match the Python parser (no new kinds since Stage 33). Clean.
- Exit-code convention: bad invocation → rc=2, source/parse → rc=1, internal/runtime → rc=1 across all four CLIs. Clean.

---

LANE_B_TOTAL: 0 findings (H=0 M=0 L=0) | 7 clean families
