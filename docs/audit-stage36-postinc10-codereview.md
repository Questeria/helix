# Stage 36 Post-Increment-10 Audit — Code-Review Lane

**Date**: 2026-05-16
**Auditor**: pr-review-toolkit:code-reviewer (Opus 4.7 1M)
**HEAD audited**: `821592f` (Inc 10 knowledge-graph dogfood) + `14e1fa4` (Inc 9 catch-up ledger)
**Baseline**: `a451591` (Inc 8 closure)
**Scope**: 16 commits, 18 files, +1978/−80 LOC (largest churn: `test_stage36_provenance.py` +844, `lower_ast.py` +240, `typecheck.py` +111)
**Status**: **0 HIGH + 0 MEDIUM + 2 LOW** (CLEAN)

## Verification steps run

1. Read `audit-stage36-postinc8-codereview.md` for prior lane structure.
2. Read `stage36-progress-2026-05-16.md` (Inc 9 + catch-up + Inc 10 ledger).
3. Read `dogfood_09_knowledge_graph.hx` end-to-end (70 LOC).
4. Read `test_reflection.py::test_dogfood_09_knowledge_graph`.
5. Diffed Inc 9 catch-up commit `e1ca1f9` (autodiff/typecheck/lower_ast/tests).
6. Inspected `register_derivation` lowering (`lower_ast.py:1980–2014`), `derive` lowering (`lower_ast.py:1855–1875`), and `_clamp_unit_f32` reuse across all fuzzy ops (`lower_ast.py:2134–2227`).
7. Cross-checked ROADMAP "9 dogfood programs" claim against `helixc/examples/dogfood_*.hx` on disk (= 9).
8. Verified `count * 21 * ev_ok = 42` algebra for all reachable `(count, ev_ok)` tuples.

## Exit-42 math (Inc 10 dogfood)

`count = unwrap_logic(gp_ac) + unwrap_logic(gp_ad)` ∈ {0,1,2} (each `unwrap_logic` of an `and_logic` over `prove(1,_)` operands is 0 or 1). `ev_ok` ∈ {0,1} by `if/else` construction. `count * 21 * ev_ok = 42` iff `count = 2 ∧ ev_ok = 1`. No degenerate path produces 42 by luck (the only factor combos giving 42 are 2·21·1; `count = 1` would need a non-existent `ev_ok = 2`). Math is sound.

## Findings

### C1 LOW (conf 82) — Doc/code drift in `dogfood_09_knowledge_graph.hx` header comment

**File**: `helixc/examples/dogfood_09_knowledge_graph.hx:5-7`

Header claims the program "Demonstrates derive() now being observably side-effectful (B2 fix)." The program never calls `derive()` — it uses `register_derivation(1, 2)` / `register_derivation(1, 3)` directly. The B2 fix (derive routes through `ARENA_PUSH_PAIR`) is not exercised by this dogfood. Misleads a reader looking to confirm B2 lands.

**Fix**: drop the "derive() now being observably side-effectful (B2 fix)" clause, or add a third rule that uses `derive(...)` to actually exercise B2.

### C2 LOW (conf 81) — `test_dogfood_09_knowledge_graph` only pins exit code, not program semantics

**File**: `helixc/tests/test_reflection.py:229-240`

The test asserts `compile_and_run(src) == 42`. As shown above, `count * 21 * ev_ok = 42` uniquely identifies `(2, 1)` — so the exit code does pin the correct semantic outcome (both rules fire AND all four `parent_*_at` lookups recover the right source IDs). However, the test does not pin: (a) the handle values `h_ac=1, h_ad=3` (which would catch a regression in 1-based handle math or `ARENA_PUSH_PAIR` cursor advancement), (b) cross-talk between h_ac and h_ad (`parent_left_at(h_ad)` must not read h_ac's slot). Matches the pattern of `test_dogfood_07/08` which also only assert exit-42, so this is consistent — flagging as a coverage gap, not a defect.

**Fix**: optional — add a second sibling test that exercises a wrong handle (`parent_left_at(99)`) and asserts the `-1` sentinel reaches exit, or add a third rule whose handle would be `5` to pin `ARENA_PUSH_PAIR` cursor stride.

## Items checked and clean

- **Inc 9 catch-up commit `e1ca1f9` minimality**: every change is scope-aligned to its audit finding ID. The 7 `return a or b` → `return None` edits (B1 silent-failure) are mechanical, identically-shaped, and each has a corresponding test in `test_stage36_provenance.py`. The `prove`-flatten rejection (B1 type-design) adds a single TypeError branch and updates the previously-wrong `test_prove_on_already_logic_is_idempotent` docstring + assertion to match the new contract. No drive-by edits; no stale TODOs; no commented-out code.
- **`AD_KNOWN_PURE_CALLS` consistency**: prior lane's C1 LOW finding (Inc 2/3/5 ops missing) was closed in commit `0e548f0` (per progress ledger). Re-verified: `derive`, all `*_logic`, `to_logic_bool`, `register_derivation`, `parent_*_at` are in the set or explicitly excluded.
- **Self-host gate stability**: progress ledger claims sha `a6f1ee44...` byte-identical before/after Inc 9 catch-up. No cascade log is checked into the repo (`.stage31-logs/` only holds Stage 31 logs), so the claim cannot be independently verified from artifacts alone; however, the catch-up commit's surface is typecheck/AD-only with zero codegen path — the byte-identical claim is mechanically plausible.
- **ROADMAP "8→9 dogfood programs" claim**: matches `ls helixc/examples/dogfood_*.hx` = 9 files (01–09). Consistent.
- **`_clamp_unit_f32` reuse**: A3 clamp helper applied to every fuzzy op (`fuzzy_and`, `fuzzy_or`, `fuzzy_not`, `fuzzy_xor`, `fuzzy_implies`) symmetrically. No op forgot to clamp.

## Verdict

No HIGH or MEDIUM findings. Inc 10 dogfood math is uniquely identifying. Inc 9 catch-up commits are minimal-scope and well-tested. The two LOW items are doc/coverage polish, not blockers — Stage 36 Inc 10 is clean from the code-review lane.
