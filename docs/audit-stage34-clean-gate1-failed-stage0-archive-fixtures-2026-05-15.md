# Stage 34 Clean Gate 1 Stage0 Archive Fixture Restart

Date: 2026-05-15
Stage: 34
Gate: Clean gate 1
Result: Failed, fixed, and reset to 0/3 clean gates

## Finding

Fresh archive reproducibility auditors on commit `b636256` found that a clean
`git archive HEAD` extraction failed the Stage 0 shell gate under WSL:

```text
FAIL 01-hello expected $'Hello\r' actual Hello
FAIL 02-comments-ws expected $'Kovostov\r' actual Kovostov
Results: 1 passed, 2 failed
```

The shell scripts were LF and parsed correctly, but the archive exported
`stage0/hex0/test/01-hello.expected` and
`stage0/hex0/test/02-comments-ws.expected` with CRLF bytes. The live checkout
looked clean, so the bug only appeared when the full shell gate was replayed
from the committed archive.

## Fix

`.gitattributes` now forces LF for the Stage 0 hex0 text fixtures:

- `stage0/hex0/test/*.expected text eol=lf`
- `stage0/hex0/test/*.hex0 text eol=lf`

A new regression test archives the candidate tree, verifies the Stage 0 fixture
bytes contain no carriage returns, and runs `stage0/hex0/run_tests.sh` from the
extracted archive. The quick validation list includes that regression so this
specific archive case is checked early.

## Verification

- Candidate archive fixture byte scan:
  `01-hello.expected CR=0`, `02-comments-ws.expected CR=0`, and all checked
  `.hex0` fixtures `CR=0`.
- Candidate archive Stage 0 shell gate:
  `PASS 01-hello`, `PASS 02-comments-ws`, `PASS 03-empty`,
  `Results: 3 passed, 0 failed`.
- Stage 0 archive regression:
  `python -m pytest -q helixc/tests/test_stage0_archive.py::test_stage0_hex0_archive_fixtures_are_lf_and_shell_gate_passes`:
  `1 passed`.
- `python scripts\stage31_validate.py --mode quick`: passed; snapshot check
  and compile returned `0`, and snapshot run returned `42`.

## Gate State

The clean-gate counter remains reset to `0/3`. A fresh clean gate should start
from the commit containing this fix set.
