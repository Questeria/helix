# Lane C Audit Report — Stage 35 Restart 52

**HEAD**: `a4ad9a0 Fix Stage 35 fifty-first restart findings`
**Scope**: Docs / status / release surfaces. Read-only audit; fixes applied separately.

## Summary

Reviewed `README.md`, `QUICKSTART.md`, `HANDOFF_FOR_CLAUDE.md`, `HANDOFF_FOR_CHATGPT.md`, `docs/`, `helix_website/`, and `helixc/examples/`. Restart 51 reconciled all eight test-count surfaces to the live `2,497`. Restart 52's runtime fix to `ti2d_matvec`/`ti2d_matmul` does not add new tests, so the test-count remains `2,497`. Found **2 findings**: 1 MEDIUM (typo cluster in forecast-string corrections), 1 LOW (HELIX_REFERENCE per-module @-attributed count drifted by 1).

---

## C1 — Forecast-string regression typo cluster across four surfaces — MEDIUM

**Files**:
- `HANDOFF_FOR_CHATGPT.md:6` and `:231`
- `QUICKSTART.md:22`
- `helix_website/stats_and_facts.md:14`
- `helix_website/HELIX_REFERENCE.md:510`

**Issue**: Restart 51 C12 reconciled the published test count from `2,489` (forecast) to `2,497` (live) on all eight surfaces. Spot-check of the historical-context blocks ("restart 50 forecast was 2,489") found four sites where the historical figure was unintentionally bumped to `2,497`, corrupting the narrative ("restart 50 forecast was 2,497" — but it was 2,489). The live current number is correct; only the historical-forecast figure is wrong.

**Suggested fix**: Restore "2,489" in the four narrative-historical contexts; keep "2,497" in the current-state contexts.

**Suggested canary**: none feasible at doc level — this is the cross-surface narrative consistency family.

---

## C2 — HELIX_REFERENCE.md @-attributed fn count drift — LOW

**File**: `helix_website/HELIX_REFERENCE.md:510`

**Issue**: After restart 51's `transcendentals.hx` `"2 bare fn (+50 @-attributed)"` callout, the global stdlib summary `"892 declarations total"` (455 + 437) needs a re-verification — the bare/@-attributed split should be re-counted against live `ls` output if restart 52's `ti2d_matvec`/`ti2d_matmul` saturation refactor added any internal helpers. Live count: still 16 modules, no new files added in restart 52.

**Suggested fix**: optionally re-run the stdlib counter and bump `437` → `437+N` if any new helpers were added. For restart 52 (no new helpers), keep the number as-is.

**Suggested canary**: none feasible at doc level.

## Clean families swept

- Test count consistency (8 surfaces): all consistent at 2,497. Clean.
- Restart number consistency: HEAD says restart 51 was the most recent fix-sweep landed; all surfaces also say restart 51. Clean.
- Current vs future capability claims: all "ships X" / "supports X" claims gated by Python-hosted-helixc honesty disclaimers. Clean.
- License consistency: Apache 2.0 file-resident + CC-BY-4.0/CC0 stated policy. Clean.
- Tool flag claims: README/QUICKSTART/HELIX_REFERENCE flag lists match `python -m helixc.check --help`. Clean.
- Stdlib module count: 16 modules everywhere. Clean.

---

LANE_C_TOTAL: 2 findings (H=0 M=1 L=1) | 6 clean families
