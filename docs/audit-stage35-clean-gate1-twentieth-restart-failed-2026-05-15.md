# Stage 35 Clean Gate 1 - Twentieth Restart Findings

Date: 2026-05-15

Result: FAIL. Clean-gate count remains 0/3.

This audit was run after commit `b776d2a` (`Fix Stage 35 nineteenth restart
findings`). Smoke checks and supporting regression slices were green, but all
three lanes found remaining issues. The gate did not count as clean.

## Smoke And Supporting Verification Before Fix

- Reverse-AD smoke: 5 passed.
- Tensor/accessor smoke: 6 passed.
- PTX/CLI smoke: 10 passed.
- Public-doc stale-claim scan: initially clear for the narrowed scan.
- Stdlib parser sweep: parsed 16 stdlib files.
- Tensor/accessor family slice: 40 passed.
- Full CLI tests: 155 passed.
- Full direct PTX tests: 70 passed.
- Full reverse-AD tests: 29 passed.
- Scoped test collection: 2,254 tests collected.

## Lane A - AD / NN / Runtime Correctness

- P2: direct 2D accessors still accepted row-index out-of-bounds accesses.
  `t2d_offset(start, cols, i, j)` only knew `cols`, so `i >= rows` could write
  into the next arena allocation. Existing tests only covered `j >= cols`.

## Lane B - PTX / Tile / Autotune CLI Parity

- P2: `helixc.check --emit-ptx --strict` filtered to kernel-reachable code
  before effect validation, so host-only `@pure` IO violations were missed.
- P2: direct `helixc.backend.ptx --strict` still lowered full host code and
  rejected unrelated host-only AD helpers with `unresolved generic type D`.
- P3: direct `helixc.backend.ptx` did not implement `-Wad=error`, so AD
  warning policy was not aligned with `helixc.check`.

## Lane C - Documentation / Status Consistency

- P2: the public-doc stale-claim scan was too narrow; website reference and
  contract files still contained old self-hosting, 120-byte, and `3000+` style
  claims.
- P2: website README and reference copy still presented self-hosting/current
  bootstrap claims too strongly for the live Stage 35 state.
- P3: `docs/PLAN.md` was marked historical but still used phase headings that
  looked current in stale-status scans.

## Fix Plan

- Add metadata-backed 2D tensor allocation so `t2d_offset` can validate rows
  as well as columns, and pin the row-OOB clobber case with a regression.
- Run strict full-program effect validation before PTX kernel filtering, while
  keeping artifact lowering restricted to kernel-reachable code.
- Mirror direct PTX CLI behavior with strict host-AD filtering and `-Wad=error`.
- Broaden public-doc cleanup across root docs, website drafts, API contracts,
  and the historical plan snapshot.
