# Stage 35 Clean Gate 1 - Twenty-First Restart Findings

Date: 2026-05-15

Result: FAIL. Clean-gate count remains 0/3.

This audit was run after commit `c6273a9` (`Fix Stage 35 twentieth restart
findings`). Smoke checks and supporting regression slices were green, but audit
lanes found remaining Stage 35 issues. The gate did not count as clean.

## Smoke And Supporting Verification Before Fix

- Stdlib parser sweep: parsed 16 stdlib files.
- Stage 35 reverse-AD smoke: 8 passed.
- Stage 35 2D/accessor smoke: 5 passed.
- `emit_ptx` CLI smoke: 11 passed.
- Direct PTX smoke: 20 passed.
- Public-doc stale-claim scan: clear for the narrowed restart-21 scan.
- Tensor/accessor family slice: 45 passed.
- Full CLI tests: 156 passed.
- Full direct PTX tests: 72 passed.
- Full reverse-AD tests: 29 passed.
- Scoped test collection: 2,258 tests collected before this fix sweep added
  new regressions.

## Lane A - AD / NN / Runtime Correctness

- P2: 2D metadata was enforced only by direct `ti2d_*`/`tf2d_*` accessors.
  Higher-level 2D, matrix, and NN helpers still trusted caller-provided
  `rows`/`cols`, so a flat or undersized arena allocation could be treated as a
  matrix and read or write adjacent allocations.
- P2: `rev_backward` could mutate adjoints for earlier tape entries before
  discovering a later corrupt tape entry. A bad tape therefore left partial
  backward-state changes behind.
- P2: `__gelu` was listed as AD-known but had no forward-mode or reverse-mode
  analytic chain rule.
- P3: BCE public losses were runtime-tested but not AD-compatible because
  `bce_loss_scalar` used local clamp/log composition while `__clamp` remained
  opaque to AD.

## Lane B - PTX / Tile / Autotune CLI Parity

- No completed finding packet was received. The first PTX lane exceeded the
  practical wait window, was closed, and the replacement was closed while this
  fix sweep proceeded from Lane A and Lane C evidence. Restart 22 should include
  a fresh PTX lane instead of counting this as clean PTX evidence.

## Lane C - Documentation / Status Consistency

- P2: public status docs still contained the restart-20 test count after
  restart 21 verification had collected 2,258 tests.
- P2: `helix_website/HELIX_REFERENCE.md` still had a `1000+ tests` style
  public-count claim.
- P2: website reference/API contract surfaces still had stale `120-byte` /
  `120 B` / `sizeBytes: 120` hex0 claims even though the current hex0 binary is
  299 bytes.

## Fix Plan

- Require 2D shape metadata in the higher-level matrix and NN helpers that
  consume or write 2D buffers, and pin one cross-allocation mismatch case.
- Pre-validate the full reverse-AD tape before mutating any adjoints.
- Add forward and reverse AD chain rules for `__gelu`, and route BCE scalar
  loss through an AD-known `__bce` primitive with forward and reverse rules.
- Update public docs and website draft surfaces to the live restart-21 fix
  verification count and current 299-byte hex0 value.
