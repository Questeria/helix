# Stage 32 Audit Gate 2 - Silent Failure Review

Status: PASS

Silent-failure checks:

- Selector exact-file rules do not accidentally match suffixes such as
  `scripts/stage31_validate.py.bak`.
- Missing or malformed duration-weight files still fall back to stable hash
  sharding.
- Focused mode rejects positional paths outside `--mode focused`, so quick and
  full gates cannot silently ignore a user-supplied path list.
- Docs-only focused mode runs `git diff --check` instead of reporting a false
  pytest pass.
- Default focused mode ignores stale untracked audit docs while still seeing
  untracked source/test/tooling files.

Evidence:

- `helixc\tests\test_stage32_select_tests.py`
  - covers suffix matching, docs-only selection, broad fallback selection, JSON
    output, and untracked source/test/tooling discovery.
- `helixc\tests\test_stage31_validate.py`
  - covers focused-mode pytest execution, docs-only diff check, focused-mode
    path acceptance, and rejection of paths in non-focused modes.
- `python scripts\stage31_validate.py --mode focused --skip-snapshot`
  - selected current source/test changes, not stale audit docs.
- `bash scripts/run_all_tests.sh`
  - completed green after the audit fix.

Residual risk:

- The selector is a hand-maintained map. Future new compiler areas should add
  selector rules and tests in the same commit as the new area.
