# Stage 38 post-Inc-3 code review

**Reviewer**: code-review subagent (Claude Opus 4.7)
**Date**: 2026-05-16
**Scope**: commits `86c2ce4..b427f4f` (Stage 38 Inc 0/1/2/3 + Stage 37 backfill `a8ab17b`)
**Filter**: confidence >= 80 only (per system policy)
**Files reviewed**:
- `C:/Projects/Kovostov-Native/helixc/frontend/typecheck.py`
- `C:/Projects/Kovostov-Native/helixc/frontend/autodiff.py`
- `C:/Projects/Kovostov-Native/helixc/ir/lower_ast.py`
- `C:/Projects/Kovostov-Native/helixc/examples/dogfood_11_spatial_frames.hx`
- `C:/Projects/Kovostov-Native/helixc/examples/run.py`
- `C:/Projects/Kovostov-Native/helixc/tests/test_stage38_frames.py`
- `C:/Projects/Kovostov-Native/helixc/tests/test_reflection.py`
- `C:/Projects/Kovostov-Native/docs/stage38-progress-2026-05-16.md`

---

## HIGH (90-100)

None.

---

## IMPORTANT (80-89)

### S38-CR-001 — Ledger drift: Inc 1 still labelled "(planned)" and Inc 3 still in "Planned Sequence" after both landed

**Severity**: MEDIUM
**Confidence**: 92
**File**: `C:/Projects/Kovostov-Native/docs/stage38-progress-2026-05-16.md`
**Lines**: 37, 101-105

The ledger violates the Stage 37 ledger-shape convention in two places:

1. **Line 37** — `## Increment 1 - Frame Constructors + Eliminators (planned)` — still tagged "(planned)" despite Inc 1 having shipped in commit `86c2ce4` ("Stage 38 OPENS: Inc 0 + Inc 1"). Stage 37 ledger pattern was to keep the "(planned)" header AND add a `### Increment N status: SHIPPED (commit <sha>, <date>)` subsection (see `docs/stage37-progress-2026-05-16.md:81` for the template). Stage 38 Inc 1 has NO post-landing status note.

2. **Lines 101-105** — `## Increment 3+ — Planned Sequence` still lists `**Inc 3**: Dogfood — dogfood_11_spatial_frames.hx ...` even though Inc 3 landed in commit `b427f4f` ("Stage 38 Inc 3: spatial-frame lifecycle dogfood"). Inc 2 got a `## Increment 2 — Cross-Frame Transforms (LANDED)` section header but Inc 3 did not.

This is the same class of bug the Stage 36 closure ceremony was caught with (per `stage37-progress-2026-05-16.md:231` "Closure-gate hygiene retrospective"): the ledger does not reflect ground truth. Future audit gates that read this ledger as the source-of-truth for what shipped will mis-classify Inc 1 and Inc 3 as un-landed.

**Citation**:
```
$ grep -n "(planned)" docs/stage38-progress-2026-05-16.md
37:## Increment 1 - Frame Constructors + Eliminators (planned)
$ grep -n "Inc 3" docs/stage38-progress-2026-05-16.md
103:- **Inc 3**: Dogfood — `dogfood_11_spatial_frames.hx` showing a
```

**Recommended fix**: Promote Inc 3 to its own `## Increment 3 — Lifecycle Dogfood (LANDED)` section (matching the Inc 2 pattern), remove Inc 3 from the Planned Sequence list, and append a `### Increment 1 status: SHIPPED (commit 86c2ce4, 2026-05-16)` subsection under the Inc 1 header to match Stage 37's template.

---

### S38-CR-002 — Coverage gap: zero T-propagation tests for frame builtins (only i32 tested)

**Severity**: MEDIUM
**Confidence**: 85
**File**: `C:/Projects/Kovostov-Native/helixc/tests/test_stage38_frames.py`
**Lines**: all (every test uses `i32` only)

The Inc 1 constructors return `TyFrame(frame=..., inner=arg_tys[0])` and Inc 2 transforms return `TyFrame(frame=dst_frame, inner=arg_tys[0].inner)` — both rely on the inner type being propagated unchanged. ZERO test in `test_stage38_frames.py` exercises a non-`i32` inner type. Specifically:

- No `WorldFrame<f32>` test, despite the ledger and the TyFrame docstring explicitly motivating the feature with `(0.5, 0.3, 1.2)` float coordinates (typecheck.py:246).
- No test that `world_to_robot(WorldFrame<f32>)` returns `RobotFrame<f32>` (not e.g. `RobotFrame<i32>` due to a silent fall-through to a hardcoded default).
- No test for nested wrappers (e.g. `WorldFrame<Logic<i32>>` or `D<WorldFrame<f32>>`) that would exercise interaction with Stage 36 provenance / autodiff types.

A regression that, say, hardcoded `TyPrim("i32")` as the inner for cross-frame transforms would currently pass all 14 Stage 38 tests because every test uses i32 anyway. The Stage 37 test suite had the same gap (all tier tests use i32) but Stage 37 is closed; Stage 38 is still pre-closure and can fix this cheaply with one parametrised test.

**Citation**:
```
$ grep -E "Frame<" helixc/tests/test_stage38_frames.py | grep -v "i32"
(no output — all 12 occurrences are <i32>)
```

**Recommended fix**: Add one Inc 4 test `test_stage38_inc4_inner_type_preserved_across_transform` that asserts `world_to_robot(into_world(1.5f32))` typechecks to `RobotFrame<f32>` and `from_robot` returns `f32`.

---

### S38-CR-003 — Stage 37 backfill: `parent_at` post-closure M2 fix uses a misleading "pre-Inc-14" remediation label

**Severity**: LOW (cosmetic; user-visible wording)
**Confidence**: 82
**File**: `C:/Projects/Kovostov-Native/helixc/frontend/typecheck.py`
**Lines**: 3047-3049, 3054-3055

The Stage 37 post-closure fix at commit `a8ab17b` (in this Stage 38 range) added a strict-i32 family-standard remediation hint to `parent_at`. The inline comment at typecheck.py:3047-3049 admits the problem:

```
# Pre-Inc-14 the generic `parent_at` did not exist; using the
# `pre-Inc-14` label keeps the timeline accurate.
```

But the user-facing error string says `"...(pre-Inc-14 also accepted i64/u32/u64 but those silently truncated in downstream arena read ops)"`. If `parent_at` didn't exist pre-Inc-14, then pre-Inc-14 it accepted NOTHING — the hint is factually wrong, not just calibration-loose. A user reading this hint after passing `i64` to `parent_at` will read "pre-Inc-14 also accepted i64" and reasonably conclude there was a deprecation transition for `parent_at`. There wasn't.

Compare the same-commit `register_derivation3` block (typecheck.py:3014-3018) which uses `"pre-Inc-14"` correctly because `register_derivation3` DID exist at Inc 14 (it was the strict-i32 transition point for that builtin).

**Citation**:
```
$ grep -n "pre-Inc-14" helixc/frontend/typecheck.py
3015:                                t, "pre-Inc-14", "arena push ops")
3055:                                t, "pre-Inc-14", "arena read")
```

**Recommended fix**: Either (a) use a label that's accurate for `parent_at` (e.g. the actual increment when `parent_at` became strict-i32), or (b) extend `_strict_i32_truncation_hint` with an optional `omit_era: bool` parameter so the parent_at site can emit just `"(i64/u32/u64 would silently truncate in downstream arena read ops)"` without claiming a non-existent historical transition. This is Stage 37 backfill technical debt, not Stage 38 surface, but it shipped in the Stage 38 commit range.

---

### S38-CR-004 — Naming-convention drift: Stage 38 uses `from_*` eliminators, Stage 37 used `unwrap_*`

**Severity**: LOW
**Confidence**: 80
**Files**:
- `C:/Projects/Kovostov-Native/helixc/frontend/typecheck.py:1885-1887` (registration)
- `C:/Projects/Kovostov-Native/docs/stage38-progress-2026-05-16.md:55-57` (spec)

Stage 24/Stage 37 established the eliminator naming convention as `unwrap_<tag>`:
```
unwrap_working, unwrap_episodic, unwrap_semantic, unwrap_procedural
unwrap_logic   (Stage 36)
```

Stage 38 deviates to `from_<tag>` (`from_world`, `from_robot`, `from_camera`). The Stage 38 progress ledger does NOT call out or justify this naming pivot, unlike the Inc 2 cross-frame pivot which did get a "Naming pivot from the planned spec" subsection (lines 67-73).

Two consequences:
1. User-facing inconsistency — a user who learned `unwrap_working` for Stage 37 will hunt for `unwrap_world` and fail.
2. Future audit-gate convention drift will be hard to argue against ("Stage 38 did it") if not anchored with a stated rationale.

There IS an arguable rationale (`unwrap_world` reads awkwardly; `from_world` reads naturally as "extract from world frame") but it's not in the ledger.

**Citation**:
```
$ grep -E "^\s+\"(into|unwrap|from)_" helixc/frontend/typecheck.py | head -20
1883:        "into_working", "into_episodic", "into_semantic", "into_procedural",
1884:        "unwrap_working", "unwrap_episodic", "unwrap_semantic", "unwrap_procedural",
1886:        "into_world", "into_robot", "into_camera",
1887:        "from_world", "from_robot", "from_camera",
```

**Recommended fix**: Add a "Naming pivot" subsection to `## Increment 1` in the Stage 38 ledger explaining why `from_*` was chosen over `unwrap_*` (matches Inc 2's existing "Naming pivot from the planned spec" pattern). Optionally add `unwrap_world` / `unwrap_robot` / `unwrap_camera` as aliases for users coming from Stage 37 muscle memory.

---

## Items considered and dismissed (confidence < 80)

### Dogfood is "trivial" — only exercises 3 of 6 Inc 2 transforms

**Confidence**: 70 — dismissed.

`dogfood_11_spatial_frames.hx:37-46` uses `into_world` (Inc 1), `world_to_robot`, `robot_to_camera`, `camera_to_world` (3 of 6 Inc 2 transforms), and `from_world` (Inc 1) — but NOT `into_robot`, `into_camera`, `from_robot`, `from_camera`, `robot_to_world`, `camera_to_robot`, `world_to_camera`. This is a partial-coverage program (5 of 12 Stage 38 surface builtins). However, the comparable `dogfood_10_memory_tiers.hx` for Stage 37 has the same shape — partial-surface exercise of a real cyclic flow. The dogfood pattern is "demonstrate a realistic chain", not "exhaustive matrix coverage" (that's the test suite's job). Dismissed below threshold.

### No test pins Stage 38 builtins are in `AD_KNOWN_PURE_CALLS`

**Confidence**: 70 — dismissed.

The Stage 37 closure gate-1 LOW (S37-CLEAN1-001) finding was exactly "tier builtins absent from AD_KNOWN_PURE_CALLS" — and the Stage 38 ledger explicitly calls out preemptive registration to avoid recurrence (autodiff.py:84-86, ledger lines 61-62). But neither Stage 37 (post-fix) nor Stage 38 has a unit test pinning the AD-pure registration. The dogfood runtime test (`test_dogfood_11_spatial_frames` in test_reflection.py) implicitly exercises the path because `compile_and_run` calls `grad_pass`, so any frame builtin not in AD_KNOWN_PURE_CALLS would surface as a runtime failure. Adequate coverage via integration test. Dismissed below threshold.

### Cross-frame transform error message format vs. tier consolidate/recall errors

**Confidence**: 70 — dismissed.

Tier `consolidate`/`recall` errors say `"consolidate() requires EpisodicMem<T>, got X"` — no transform-name redundancy. Frame transforms say `"world_to_robot() requires WorldFrame<T>, got X"` — transform name is the same as `{bn}` so the format is implicitly identical. Family-standard format is met. Dismissed below threshold.

### `run.py` docstring says "5 in sequence" but DEMOS now has 11 entries

**Confidence**: 75 — dismissed.

Pre-existing drift (not Stage 38 specific — was already wrong at the start of Stage 38). Dismissed; not introduced by this commit range.

### Dogfood uses `@pure` decorator but Stage 37 dogfood does not

**Confidence**: 55 — dismissed.

Stage 37 dogfood_10 omits `@pure`; Stage 38 dogfood_11 adds it. This is a positive divergence (helps the IR optimizer) and not a convention violation. Dismissed.

---

## Convention-check summary

| Convention (Stage 36/37 playbook) | Stage 38 status |
|-----------------------------------|-----------------|
| Combined audit-and-fix per increment | Not yet — Inc 1/2/3 shipped without per-increment audit; closure gate sequence pending (matches ledger plan) |
| Ledger updated as increments land | **VIOLATED** (S38-CR-001) |
| Builtin naming consistent with prior stages | **DRIFT** (S38-CR-004 — `from_*` vs `unwrap_*`) |
| Identity-lowering pattern at IR | OK (matches Stage 37 pattern exactly) |
| Frame builtins in AD_KNOWN_PURE_CALLS | OK (preemptive, both Inc 1 and Inc 2) |
| Cross-mismatch coverage tests (all wrong-pair combinations) | OK (6/6 for Inc 1 eliminators, 12/12 for Inc 2 transforms) |
| Typecheck error format ("CapitalizedFrame<T>", transform name in message) | OK |
| T-propagation tested | **GAP** (S38-CR-002 — i32-only) |
| Stage 37 post-closure backfill (`a8ab17b`) | **LABEL BUG** (S38-CR-003) |
| Self-host gate green per commit | Not directly verified by reviewer; per ledger convention |

---

## 5-line summary

Stage 38 Inc 0-3 surface is structurally clean and mirrors the Stage 37 tier pattern faithfully (TyFrame + 6 + 6 builtins + identity-lowering + AD-pure registration + 14 tests covering happy path and all 18 wrong-source rejections). Four findings, no HIGHs: S38-CR-001 (MEDIUM/92) ledger ground-truth drift — Inc 1 still tagged "(planned)" and Inc 3 still listed in "Planned Sequence" though both landed; S38-CR-002 (MEDIUM/85) every test uses `i32` — zero coverage that `WorldFrame<f32>` etc. propagate the inner type correctly, despite the docstring motivating the feature with float coordinates; S38-CR-003 (LOW/82) Stage 37 backfill `parent_at` strict-i32 hint cites a "pre-Inc-14" era during which `parent_at` did not exist (per its own inline comment); S38-CR-004 (LOW/80) Stage 38 silently pivoted eliminator naming from Stage 37's `unwrap_*` to `from_*` without a "Naming pivot" rationale in the ledger (Inc 2 documented its naming pivot, Inc 1 did not). All four are fixable in a single Inc 4 ledger/test sweep before the 3-clean-gate closure sequence.
