# Stage 32 Audit Gate 3 - Integration Verification

Status: PASS

Integration checks:

- Stage 31/32 quick validation remains green.
- Focused mode works as a direct validator mode.
- Full sharded pytest, snapshot smoke, and stage0/hex0 all pass after Stage 32
  speed-tooling changes.
- Machine-readable shard timing and node-duration files are still produced by
  the full validator.

Final full-gate evidence:

- `bash scripts/run_all_tests.sh`
  - pytest gate `rc=0`
  - stage0/hex0 `rc=0`
  - total: all gates passed
  - snapshot smoke: `rc=42`
  - stage0/hex0: `3 passed, 0 failed`
  - parallel pytest group: about 17m55s on the audit-fix run

Timing note:

- The final full gate was much slower than the prior clean Stage 32 gates.
  Earlier Stage 32 gates reported parallel groups around 4m11s to 4m21s.
  The slow run still completed green and reinforced the need for Stage 32's
  telemetry and focused-mode workflow.

Residual risk:

- Wall-clock speed still depends heavily on machine load. Stage 32 improves
  observability and edit-loop speed, but it does not eliminate the need for the
  full gate.
