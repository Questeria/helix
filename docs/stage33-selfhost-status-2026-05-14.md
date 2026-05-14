# Stage 33 Self-Host Status - 2026-05-14

Purpose: return from Stage 32 speed work to the central independence goal:
Helix should compile Helix, repeatedly and byte-identically, until the Python
compiler can become a historical reference instead of a required dependency.

## Baseline

Prior Stage 30 release-hardening evidence:

- `python scripts\selfhost_cascade.py --generations 10 --keep`
- Result: PASS
- G2..G11 stable SHA-256:
  `5a7367ad436e72ade3d8f96a9860e0d08b64528cbb15295e1a47076090667408`
- G2..G11 stable size: `277899` bytes
- Final-generation smoke cases: literal, call, and loop all returned `42`

Fresh Stage 33 baseline:

- `python scripts\selfhost_cascade.py --generations 3`
  - Result: PASS
  - G2..G4 stable SHA-256:
    `5a7367ad436e72ade3d8f96a9860e0d08b64528cbb15295e1a47076090667408`
  - G2..G4 stable size: `277899` bytes
  - Final-generation smoke cases: literal, call, and loop all returned `42`

## Slice 1 - Machine-Readable Cascade Reports

`scripts/selfhost_cascade.py` now accepts `--json-out <path>` and writes a
machine-readable cascade report with:

- schema id: `helix.selfhost_cascade.v0`
- seed compiler size and SHA-256
- every self-host generation's size and SHA-256
- stable/unstable decision
- stable hash and size when stable
- final-generation smoke results

Validation:

- `python -m pytest -q helixc\tests\test_selfhost_cascade.py`
  - Result: `3 passed`
- `python -m pytest -q helixc\tests\test_selfhost_cascade.py helixc\tests\test_stage32_select_tests.py`
  - Result: `14 passed`
- `python scripts\stage31_validate.py --mode focused --skip-snapshot scripts\selfhost_cascade.py helixc\tests\test_selfhost_cascade.py`
  - Result: `rc=0`
- `python scripts\selfhost_cascade.py --generations 3 --json-out .stage33-logs\selfhost-cascade-g3.json`
  - Result: PASS
  - JSON report confirms `stable: true`

## Next

The next Stage 33 slice should convert this report into a stricter gate:

1. Add a lightweight validator for cascade report JSON.
2. Fail closed if stable is false, expected smoke cases are missing, or the
   stable generation set drifts.
3. Use that validator before any self-host parity change is committed.
