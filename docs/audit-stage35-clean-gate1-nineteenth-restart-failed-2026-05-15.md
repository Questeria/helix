# Stage 35 Clean Gate 1 - Nineteenth Restart Findings

Date: 2026-05-15

Result: FAIL. Clean-gate count remains 0/3.

This audit was run after commit `33a6b11` (`Fix Stage 35 eighteenth restart
findings`). Smoke checks and several supporting regression slices were green,
and the runtime lane was clean, but the PTX/CLI and documentation lanes found
remaining issues. The gate did not count as clean.

## Smoke And Supporting Verification Before Fix

- Reverse-AD smoke: 5 passed.
- Tensor/accessor smoke: 6 passed.
- PTX/CLI smoke: 7 passed.
- Docs scan for known stale public claims: clear at the start of restart 19.
- Stdlib parser sweep: parsed 16 stdlib files.
- Tensor/accessor family slice: 40 passed.
- Full reverse-AD tests: 29 passed.
- Full direct PTX tests: 70 passed.
- `emit_ptx` CLI slice before the fix: 18 passed.
- Non-`emit_ptx` CLI slice: 134 passed.

## Lane A - AD / NN / Runtime Correctness

- Clean. The lane found no provable P1/P2/P3 runtime issues in the scoped
  read-only pass. It rechecked `t2d_offset`, direct `ti2d_*` / `tf2d_*`
  accessors, 2D tensor/NN guards, and reverse-AD guard paths.

## Lane B - PTX / Tile / Autotune CLI Parity

- P2: `helixc.check --emit-ptx --strict` still lowered unrelated host-only AD
  helpers through the full-program path and surfaced `unresolved generic type
  D<...>` as a public compiler-bug diagnostic. The PTX artifact path should
  lower only kernel-reachable code while AD warnings remain policy-controlled.
- P3: `helixc.check --emit-ptx` with no source path printed the full help text
  on stdout instead of keeping artifact stdout empty and putting the invocation
  diagnostic on stderr.

## Lane C - Documentation / Status Consistency

- P2: root `README.md` still described Stage 28.9 and an old test-count
  snapshot as current status.
- P2: `QUICKSTART.md` still contained the old 266-test status table and
  unsupported absolute comparison claims.
- P2: website stats/reference files overclaimed current evidence with `3000+`
  tests, a 120-byte trust root, and shipped/self-hosted compiler wording.
- P3: the previous restart's docs-lane-clean wording was too broad if read as
  a full public-docs clean bill, since root and website surfaces were missed.

## Fix Plan

- Keep `--emit-ptx` stdout reserved for PTX artifacts by routing missing-path
  invocation errors to stderr.
- Scope public `helixc.check --emit-ptx` lowering to kernel-reachable AST in
  strict and non-strict modes, while retaining full parse/typecheck/totality
  validation before artifact lowering.
- Add strict host-AD and missing-path CLI regressions.
- Update public docs and website draft material to say Stage 35 is active,
  clean gates are `0/3`, the current compiler is Python-hosted `helixc`, hex0
  is 299 bytes, and test counts are dated snapshots with a scoped collection
  command.
