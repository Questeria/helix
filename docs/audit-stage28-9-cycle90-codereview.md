# Audit Stage 28.9 cycle 90 — Code review

Scope: HEAD 94f7427 (Stage 28.9 cycle-89 fix-sweep: rename 4 duplicate test
defs to *_legacy_api). Narrow scope per cycle 90 prompt. Prior C1-C88 findings
and known-deferred items NOT re-flagged. Parallel Stages 28.10 / 28.11 NOT in
scope.

## Verdict

**FAIL** — 1 finding at conf >= 75 %.

## Findings

### C90-1 — Renamed `*_legacy_api` tests carry no docstring explaining the legacy/canonical split (conf 90, HIGH)

The cycle-89 fix renamed 4 duplicate test defs to `*_legacy_api` in
`helixc/tests/test_codegen.py`:

- `test_stdlib_vec_eq_legacy_api` (line 8382)
- `test_stdlib_vec_reverse_inplace_legacy_api` (line 8402)
- `test_stdlib_vec_first_legacy_api` (line 11167)
- `test_stdlib_vec_last_legacy_api` (line 11180)

Each kept its original functional docstring verbatim
(e.g. `"vec_eq returns 1 when all elements match, 0 on first divergence."`,
`"first([42,99]) = 42."`). The audit criterion specifically asked whether the
new docstrings "explain the legacy/canonical split" — none of the four do.

Concretely, the `_legacy_api` variants exercise the older
`vec_push(arena, idx, val)` / 3-arg `vec_eq(a, b, len)` surface, while the
canonical un-suffixed twins at lines 13494 / 11789 / 12785 / 12802 exercise
the newer `__arena_push` / 4-arg `vec_eq(a, na, b, nb)` surface (this fact is
captured only in the cycle-89 commit message, not in source). A future reader
encountering `test_stdlib_vec_first_legacy_api` next to `test_stdlib_vec_first`
has no in-source explanation of why both exist, what each covers, or whether
one is deprecated. The risk is a well-intentioned future cleanup re-collapsing
the pair on the assumption it is residual duplication — exactly what cycle-89
just paid 4 dead-coverage cases of debt to undo.

Suggested wording for each legacy docstring (one extra line above the
functional summary): `"Legacy-API surface: vec_push(arena, idx, val) +
3-arg vec_eq; the canonical __arena_push / 4-arg vec_eq counterpart lives at
<sibling line>."` Cheap, source-resident, prevents the re-collapse footgun.

## Negative results (asked, nothing found)

- **Other intra-file duplicate test defs across all `helixc/tests/test_*.py`**:
  scripted scan of 37 files finds **0** intra-file `def test_*` collisions.
  Cycle-89 was a complete sweep.
- **Near-duplicate test naming in `test_codegen.py`** (mergeable pairs other
  than the deliberate `_legacy_api` quartet): scripted scan of all 671 test
  names in the file finds only the 4 deliberate pairs above. No merge
  candidates.
- **`helixc/check.py` error-path consistency** (per cycle-85 sub-75 `-Wad`
  help-text note): out of scope this cycle per "already reviewed in
  cycle 85"; not re-flagged.

## Edits

None. Read-only audit. One `Write` to this document, no `Edit`.
