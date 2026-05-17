# Stage 39 post-Inc-3 code review

**Reviewer**: code-review subagent (Claude Opus 4.7)
**Date**: 2026-05-17
**Scope**: commit `01b3b86` (Stage 39 Inc 0/1/2/3 — Temporal types) against base `9fcc621`
**Filter**: confidence >= 70 only (per closure-gate-1 spec)
**Files reviewed**:
- `C:/Projects/Kovostov-Native/helixc/frontend/typecheck.py`
- `C:/Projects/Kovostov-Native/helixc/frontend/autodiff.py`
- `C:/Projects/Kovostov-Native/helixc/ir/lower_ast.py`
- `C:/Projects/Kovostov-Native/helixc/examples/dogfood_12_temporal_lifecycle.hx`
- `C:/Projects/Kovostov-Native/helixc/examples/run.py`
- `C:/Projects/Kovostov-Native/helixc/tests/test_stage39_temporal.py`
- `C:/Projects/Kovostov-Native/helixc/tests/test_reflection.py`
- `C:/Projects/Kovostov-Native/docs/stage39-progress-2026-05-17.md`

Reference template: `docs/audit-stage38-postinc3-codereview.md` (S38 review).

---

## HIGH (90-100)

### S39-CR-001 — TyTemporal omitted from six refinement / compatibility surfaces (parallel to S38-CR H1+H2+H3, NOT fixed at commit time)

**Severity**: HIGH
**Confidence**: 95
**File**: `C:/Projects/Kovostov-Native/helixc/frontend/typecheck.py` (at commit `01b3b86`)
**Lines**: missing arms at 4848 (`_refinement_shape_exact` target/value), 5507 (`_refinement_shape_exact` a/b), 5552 (`_erase_refinement`), 5664 (`_contains_refinement`), 5683 (`_is_refinement_container` set), 5705 (`_contains_refined_function`), 6688/6690 (`_compatible`).

Stage 38 closure gate-1 identified exactly this symmetry gap for TyFrame and shipped explicit H1/H2/H3 fixes (see the inline comments at `typecheck.py:6684-6691`, `4846-4852`, `5505-5510`, `5550-5555`, `5662-5666`, `5681-5685`, `5702-5708` in the post-fix file). Stage 39 introduces `TyTemporal` with the same wrapper shape but mirrors the **pre-fix** Stage 38 surface — TyTemporal is missing from all six places that TyFrame is present in.

Concrete consequences (each is a real silent-acceptance or refinement-loss bug):

1. `_compatible(Past<T_generic>, Past<i32>)` falls through to dataclass `a == b`, which fails for any non-equal inner — generic / refined / shape-symbolic inners that should structurally unify are silently rejected. (Mirror of Stage 38 H1 at commit `b427f4f`.)
2. `_refinement_shape_exact` cannot see refinements under `Past<{x: f32 | x.is_finite()}>` — call-boundary conversion checks silently miss refinement-shape mismatches under temporal wrappers.
3. `_erase_refinement(Past<{...}>)` returns the input unchanged (dataclass falls through), so refined inners survive erasure and cause inconsistent downstream diagnostics.
4. `_contains_refinement(Past<{...}>)` returns False — `_join_branch_types` silently drops the refinement on temporal-wrapped values at branch joins.
5. `_is_refinement_container` set excludes TyTemporal — refinement-shape check is not even fired for temporal-wrapped joins.
6. `_contains_refined_function(Past<fn(...) -> ... where ...>)` returns False — refined-function detection breaks under temporal wrappers.

**Evidence**:
```
$ git show 01b3b86:helixc/frontend/typecheck.py | grep -nE "TyFrame|TyTemporal" | grep -v "^.*:\s*#"
242: class TyFrame(Type):
255: class TyTemporal(Type):    # NEW
1119: TyFrame(frame=..., inner=...)
1130: TyTemporal(kind=..., inner=...)   # _resolve_type
3233: TyFrame(frame=_frame_intro[bn], ...)
3311: TyTemporal(kind=_temporal_intro[bn], ...)   # into_*
3248: isinstance(arg_tys[0], TyFrame)
3327: isinstance(arg_tys[0], TyTemporal)   # from_*
3281: isinstance(arg_tys[0], TyFrame)
3356/3358: isinstance(arg_tys[0], TyTemporal) + TyTemporal(kind=dst_kind, ...)
4848: TyFrame _refinement_shape_exact       # NO TyTemporal counterpart
5507: TyFrame _refinement_shape_exact (a/b) # NO TyTemporal counterpart
5552/5553: TyFrame _erase_refinement         # NO TyTemporal counterpart
5664: TyFrame _contains_refinement           # NO TyTemporal counterpart
5683: TyFrame in _is_refinement_container set # NO TyTemporal counterpart
5705: TyFrame _contains_refined_function     # NO TyTemporal counterpart
6688/6690: TyFrame _compatible               # NO TyTemporal counterpart
6854: TyFrame in _fmt
6898: TyTemporal in _fmt
```

TyFrame: 12 distinct method/site insertions. TyTemporal at commit `01b3b86`: 7 insertions. The 5 missing-arm count matches exactly the 6 H1/H2/H3 closure-gate-1 fixes (one helper, `_refinement_shape_exact`, has two sites so the diff count is 6; the Stage 38 H2 fix counted it as one logical fix).

**Note**: a follow-up commit on the working tree (uncommitted at review time) is adding precisely these 6 arms with comments labelled `Stage 39 closure gate-1 type-design H1/H2/H3` — that work is the correct fix but it had not landed at `01b3b86`. This finding documents the gap as it existed at the reviewed commit, so the closure-gate-1 ledger records that Stage 39 shipped the symmetry gap and then patched it (mirroring Stage 38's own closure trajectory).

**Recommended fix**: Land the in-progress H1/H2/H3 TyTemporal arms (currently uncommitted). The 6 inserts are mechanical mirrors of the TyFrame arms — one `isinstance(...TyTemporal...)` per site, `.kind` instead of `.frame`. Confirm via test cases that already exist in the working tree (`test_stage39_h1_*`, `test_stage39_h3_*`).

---

## IMPORTANT (80-89)

### S39-CR-002 — No integration test for `dogfood_12_temporal_lifecycle.hx` in `test_reflection.py` (convention drift from Stage 37/38)

**Severity**: MEDIUM
**Confidence**: 88
**File**: `C:/Projects/Kovostov-Native/helixc/tests/test_reflection.py`
**Lines**: dogfood test sequence at 229 (`test_dogfood_11_spatial_frames`), 242 (`test_dogfood_10_memory_tiers`) — no `test_dogfood_12_temporal_lifecycle` peer.

Stage 37 shipped `dogfood_10_memory_tiers.hx` + matching `test_dogfood_10_memory_tiers` in test_reflection.py (line 242). Stage 38 shipped `dogfood_11_spatial_frames.hx` + matching `test_dogfood_11_spatial_frames` (line 229). Stage 39 ships `dogfood_12_temporal_lifecycle.hx` but **no `test_dogfood_12_temporal_lifecycle`** test was added.

`test_stage39_inc2_lifecycle_chain_round_trips` in test_stage39_temporal.py exercises an in-memory Present→Future→Present→Past chain in raw `src` form, but it does NOT load the actual `dogfood_12_temporal_lifecycle.hx` file. Consequences:

1. A regression that breaks the dogfood `.hx` file parsing/typechecking (e.g. an `@pure` decorator interaction, the multi-helper-fn call graph, the `obs1_ok * obs2_ok * obs3_ok * rec_ok * eternal_ok` witness collapse arithmetic) wouldn't be caught by Stage 39's own test suite.
2. Future audit-gate cadence that mechanically expects "every dogfood has a peer test" (per Stage 37/38 pattern) will flag this as drift.
3. The `python -m helixc.examples.run temporal` entry path itself is uncovered — a regression in `run.py` DEMOS dict resolution (key collision, file-path resolution) would only surface when a human runs the demo by hand.

**Citation**:
```
$ grep -nE "test_dogfood_(10|11|12)" helixc/tests/test_reflection.py
229:def test_dogfood_11_spatial_frames():
242:def test_dogfood_10_memory_tiers():
$ grep -rn "dogfood_12_temporal_lifecycle" helixc/tests
(no matches)
```

**Recommended fix**: Add one parallel test in `test_reflection.py`:

```python
def test_dogfood_12_temporal_lifecycle():
    proj_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    p = os.path.join(proj_root, "helixc", "examples",
                     "dogfood_12_temporal_lifecycle.hx")
    src = open(p).read()
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42
```

Mirrors `test_dogfood_11_spatial_frames` byte-for-byte modulo the file name and exit-code label.

---

### S39-CR-003 — Coverage gap: zero T-propagation tests for temporal builtins (only `i32` tested) — repeat of S38-CR-002

**Severity**: MEDIUM
**Confidence**: 86
**File**: `C:/Projects/Kovostov-Native/helixc/tests/test_stage39_temporal.py`
**Lines**: all (every test uses `i32` only)

The Inc 1 constructors return `TyTemporal(kind=..., inner=arg_tys[0])` and Inc 2 transitions return `TyTemporal(kind=dst_kind, inner=arg_tys[0].inner)` — both rely on the inner type being propagated unchanged. ZERO test in `test_stage39_temporal.py` exercises a non-`i32` inner type. Specifically:

- No `Past<f32>` / `Future<f32>` test, despite the temporal docstring at `typecheck.py:255` motivating the feature with "coordinate X" (float). The Stage 38 review (S38-CR-002) already flagged this for frames; Stage 39 had a chance to be the first non-broken cousin and didn't take it.
- No test that `forecast(into_present(1.5f32))` returns `Future<f32>` — a regression hardcoding `TyPrim("i32")` as inner would currently pass all Stage 39 tests because every test uses i32 anyway.
- No nested-wrapper test (e.g. `Past<Logic<i32>>`, `D<Future<f32>>`, `Past<WorldFrame<f32>>`) that would exercise interaction with Stage 36 provenance / autodiff types AND Stage 38 frame wrappers.

The Inc 2 transition path is especially fragile here: `TyTemporal(kind=dst_kind, inner=arg_tys[0].inner)` does a `.inner` extraction from the source TyTemporal — if some future refactor accidentally swapped to `arg_tys[0]` (whole wrapper) instead of `.inner`, every test still passes (because i32 == i32) but real `forecast(Present<f32>)` would return `Future<Present<f32>>` (doubly wrapped) and the bug stays silent until first f32-typed use site.

**Citation**:
```
$ grep -E "<f32>|<f64>|<u32>|<i64>|<bool>" helixc/tests/test_stage39_temporal.py
(no output — all 25 tests use Past/Present/Future/Eternal<i32> exclusively)
```

**Recommended fix**: Add one `test_stage39_inc4_inner_type_preserved_across_transition` that asserts the type of `forecast(into_present(1.5f32))` is `Future<f32>` and round-trips via `from_future` back to `f32`. One test, ~15 lines, closes the family-wide T-propagation hole that Stage 37 and Stage 38 both share.

---

### S39-CR-004 — `_FRAME_IDENTITY_AD_NAMES` is a misleading name post-Stage-39 (12 temporal builtins live in a set named for frames)

**Severity**: MEDIUM
**Confidence**: 82
**File**: `C:/Projects/Kovostov-Native/helixc/frontend/autodiff.py`
**Lines**: 172-187

Stage 38 introduced `_FRAME_IDENTITY_AD_NAMES` as a frame-specific frozenset (per its docstring at lines 162-171: "the 12 frame builtins are identity-lowered at IR..."). Stage 39 piles 12 temporal builtins into this same set with a justifying comment ("Reusing the frame-identity arm avoids a parallel set + duplicate test surface; the only structural difference is the wrapper tag, which the AD pass never inspects").

The decision to share the set IS defensible (the AD pass really doesn't inspect the wrapper tag, and the chain rule IS the same), but the NAME of the set still says `_FRAME_`. Consequences:

1. The docstring at lines 162-171 still says "the 12 frame builtins" — factually wrong post-commit (it's now 24 wrappers: 12 frame + 12 temporal). A reader inspecting the docstring will believe only frame ops have AD chain rules and conclude (incorrectly) that `grad(into_past(x))` will raise NotImplementedError.
2. `_FRAME_IDENTITY_AD_NAMES` is exported and used at `autodiff.py:1293` and `autodiff_reverse.py:684` — neither call site will look surprising, but a future engineer searching for "where is the AD chain rule for `into_past`?" will not naturally grep for `_FRAME_`.
3. Stage 40 (or later) adding another wrapper family (e.g. probabilistic `Stochastic<T>`) will hit a triple-misnamed set.

The Stage 38 review explicitly noted that "Tier ops (Stage 37) still raise; a future increment may backfill the same arm there once the symmetric question is settled" (autodiff.py:170). Stage 39 backfills temporal ops INTO the frame set but leaves the docstring and name pointing at frames only — the Stage 38 forward-looking comment was the chance to introduce a more general name (e.g. `_IDENTITY_WRAPPER_AD_NAMES`) and was not taken.

**Citation**:
```
$ grep -nE "_FRAME_IDENTITY_AD_NAMES|the 12 frame builtins" helixc/frontend/autodiff.py helixc/frontend/autodiff_reverse.py
helixc/frontend/autodiff.py:89:    # via _FRAME_IDENTITY_AD_NAMES — so `grad(use_frame)` now flows
helixc/frontend/autodiff.py:163:# the 12 frame builtins are identity-lowered at IR (the wrapper is
helixc/frontend/autodiff.py:172:_FRAME_IDENTITY_AD_NAMES = frozenset({
helixc/frontend/autodiff.py:1293:    if (call.callee.name in _FRAME_IDENTITY_AD_NAMES
helixc/frontend/autodiff_reverse.py:53:    _FRAME_IDENTITY_AD_NAMES,
helixc/frontend/autodiff_reverse.py:684:                and node.callee.name in _FRAME_IDENTITY_AD_NAMES
```

**Recommended fix**: Either (a) rename `_FRAME_IDENTITY_AD_NAMES` → `_IDENTITY_WRAPPER_AD_NAMES` (or `_AD_IDENTITY_CHAIN_RULE_NAMES`) plus update 2 import/use sites and the docstring, OR (b) at minimum update the docstring at lines 162-171 to say "frame + temporal" (24 names) and update the comment at line 89 to mention temporal. (a) is the cleaner long-term fix; (b) is the cheap one.

---

## LOW (70-79)

### S39-CR-005 — Stage 39 ledger uses no landed/shipped marker (silent convention deviation from Stage 37/38)

**Severity**: LOW
**Confidence**: 75
**File**: `C:/Projects/Kovostov-Native/docs/stage39-progress-2026-05-17.md`
**Lines**: 24, 31, 43, 58 (all `## Increment N` headers).

The Stage 38 review (S38-CR-001) flagged the OPPOSITE problem — Stage 38 ledger sections kept `(planned)` tags after landing. Stage 39 reacted by removing all status markers entirely: no `(planned)`, no `(LANDED)`, no `### Increment N status: SHIPPED (commit ..., date)` subsection (which was the Stage 37 template).

Effect:
- Future audit gates that grep `(planned)` or `(LANDED)` to enumerate shipped/unshipped increments will return empty — they can't distinguish Stage 39 Inc 0/1/2/3 from Stage 39 Inc 4 (the closure gate, which IS planned-not-shipped). The lack of any marker makes ground-truth derivation harder, not easier.
- No commit-sha anchor in the ledger pointing to `01b3b86`. The Stage 37 ledger pattern was `### Increment N status: SHIPPED (commit <sha>, <date>)` (per `docs/stage37-progress-2026-05-16.md:81`).

This is a course-correction overshoot, not a hard bug. The Stage 38 lesson was "update the marker when the increment lands", not "delete the marker".

**Recommended fix**: Append a `### Increment 1/2/3 status: SHIPPED (commit 01b3b86, 2026-05-17)` subsection under each landed increment header, and mark `## Increment 4 - Stage 39 Closure (3/3 clean gates)` with a `(planned)` tag. Restores ledger-grep ground-truth derivability.

---

### S39-CR-006 — Stage 39 ledger does not document the AD-set reuse decision (parallel to S38-CR-004 naming-pivot omission)

**Severity**: LOW
**Confidence**: 72
**File**: `C:/Projects/Kovostov-Native/docs/stage39-progress-2026-05-17.md`
**Lines**: 84-86 (the Inc 1+2 implementation summary autodiff bullet)

The Stage 39 ledger summarises the autodiff change as "12 names added to `AD_KNOWN_PURE_CALLS` and to `_FRAME_IDENTITY_AD_NAMES` (chain-rule is identity, same as frame wrappers)" — one sentence. But the actual decision encoded in `autodiff.py:178-186` is non-trivial: temporal joins frame in the SAME set rather than getting `_TEMPORAL_IDENTITY_AD_NAMES`, and the rationale ("avoids a parallel set + duplicate test surface; the only structural difference is the wrapper tag") only appears in the code comment.

Stage 38's review (S38-CR-004) flagged "Stage 38 silently pivoted eliminator naming from Stage 37's `unwrap_*` to `from_*` without a 'Naming pivot' rationale in the ledger". Stage 39 has an analogous undocumented design call (set-sharing vs duplication) that future maintainers will need to re-derive from code comments. The Inc 2 transition naming itself also pivots from a hypothetical `present_to_past` / `past_to_present` (which would mirror Stage 38's `world_to_robot`) to verb-form `to_past` / `forecast` / `recall_past` / `actualize` — and the ledger does NOT call out this naming choice either (Stage 38 Inc 2 did get a "Naming pivot from the planned spec" subsection, lines 67-73 in its ledger).

**Recommended fix**: Add a brief "AD chain-rule set: reused `_FRAME_IDENTITY_AD_NAMES` rather than a parallel temporal set, rationale: structural-tag-only wrapper, AD never inspects the tag (see autodiff.py:178-186)" subsection in the ledger, plus a "Naming pivot: verb-form (`to_past`, `forecast`, `recall_past`, `actualize`) chosen over noun-pair (`present_to_past`, ...) for AGI-domain readability" subsection in `## Increment 2`.

---

## Items considered and dismissed (confidence < 70)

### Run.py docstring still says "5 in sequence" but DEMOS has 12 entries

**Confidence**: 65 — dismissed.

Pre-existing drift already dismissed in Stage 38 review (was wrong at the start of Stage 38; still wrong now). Not introduced by Stage 39 commit. Dismissed at threshold.

### Dogfood `Eternal` arm is only an intro/elim sanity check (not a true lifecycle exercise)

**Confidence**: 65 — dismissed.

`dogfood_12_temporal_lifecycle.hx:84-85` only does `from_eternal(into_eternal(1))` — no transition exercise of Eternal (correctly, because Eternal has none in Inc 2). The witness shape (`eternal_ok` is a factor of the product) is structurally OK and matches the "Eternal doesn't transition" design. Dismissed below threshold.

### No test pins that temporal builtins are in `_FRAME_IDENTITY_AD_NAMES`

**Confidence**: 68 — dismissed.

`test_stage39_ad_pure_registration` pins `AD_KNOWN_PURE_CALLS` membership but not `_FRAME_IDENTITY_AD_NAMES`. The Stage 38 review (Items dismissed section) dismissed the equivalent for frames at conf 70 because the integration test path implicitly exercises it. For temporal, there is NO integration test (per S39-CR-002) — but no Stage 39 test exercises `grad(into_past(x))` either, so the symmetric reasoning still applies. Borderline; dismissed.

### Comment at `autodiff.py:170` ("Tier ops still raise") is now misleading post-Stage-39

**Confidence**: 68 — dismissed.

The comment says tier ops still raise; that's still factually true (tier ops are NOT in `_FRAME_IDENTITY_AD_NAMES`), but the comment was written when "frame" was the only post-Stage-37 family in the set. Now that temporal also lives there, the prose "a future increment may backfill the same arm there once the symmetric question is settled" reads oddly because that future increment (Stage 39 for temporal) HAS happened — but only for temporal, not for tier. Sub-threshold; partially folded into S39-CR-004's recommended docstring update.

### Cross-temporal transition error message format vs. tier consolidate/recall errors

**Confidence**: 65 — dismissed.

Tier `consolidate`/`recall` errors say `"consolidate() requires EpisodicMem<T>, got X"`. Frame transforms say `"world_to_robot() requires WorldFrame<T>, got X"`. Temporal transitions say `"to_past() requires Present<T>, got X"`. Family-standard format met. Dismissed.

### `_temporal_intro` / `_temporal_elim` / `_temporal_transitions` dicts are defined inside the call-typecheck loop on every visit

**Confidence**: 60 — dismissed.

Same shape as Stage 38's `_frame_intro` / `_frame_elim` / `_frame_transitions` dicts — defined inside the per-call loop rather than module-level. Allocates 3 small dicts per call site visited. Performance is negligible at typecheck scale and the Stage 38 pattern was accepted; consistency wins over micro-optimisation. Dismissed.

---

## Convention-check summary

| Convention (Stage 36/37/38 playbook) | Stage 39 status |
|---|---|
| Combined audit-and-fix per increment | Not yet — closure gate-1 in progress (matches ledger plan) |
| Ledger updated as increments land | **DRIFT** (S39-CR-005 — no landed/shipped marker on any header) |
| Builtin naming consistent with prior stages | OK — `into_*` / `from_*` matches Stage 38; transitions adopt verb-form (S39-CR-006 undocumented but defensible) |
| Identity-lowering pattern at IR | OK (matches Stage 37/38 pattern exactly) |
| Temporal builtins in `AD_KNOWN_PURE_CALLS` | OK (preemptive, Inc 1 + Inc 2) |
| Temporal builtins in `_FRAME_IDENTITY_AD_NAMES` AD chain rule set | OK functionally (S39-CR-004 — set name now misleading) |
| Cross-mismatch coverage tests (all wrong-pair combinations) | OK (12/12 for Inc 1 eliminators via `test_stage39_inc1_all_12_wrong_kind_combinations`, 4/4 per-transition rejections via `test_stage39_inc2_eternal_never_transitions` + 4 explicit Eternal-rejection diag tests) |
| Typecheck error format (`Capitalized<T>`, builtin name in message) | OK |
| T-propagation tested | **GAP** (S39-CR-003 — i32-only, same as Stage 37/38) |
| TyTemporal symmetry with TyFrame in 6 refinement/compat surfaces | **VIOLATED** (S39-CR-001 — fix in progress on working tree, not in commit) |
| Dogfood file has matching integration test in `test_reflection.py` | **VIOLATED** (S39-CR-002 — Stage 37/38 had one, Stage 39 does not) |

---

## 5-line summary

Stage 39 Inc 0-3 surface mirrors the Stage 37/38 tier+frame template faithfully on the **happy path** (TyTemporal + 8 intro/elim + 4 transitions + identity-lowering + AD-pure registration + 25 tests covering happy path and the full wrong-kind rejection matrix). One HIGH finding — S39-CR-001 (95) repeats the entire Stage 38 closure-gate-1 H1/H2/H3 symmetry-gap: TyTemporal is absent from `_compatible`, `_refinement_shape_exact` (×2 sites), `_erase_refinement`, `_contains_refinement`, `_is_refinement_container` set, and `_contains_refined_function` — exactly the 6 sites Stage 38 fixed for TyFrame; the fix is being written on the working tree concurrent with this review but had not landed at `01b3b86`. Three MEDIUMs: S39-CR-002 (88) no `test_dogfood_12_temporal_lifecycle` in test_reflection.py — Stage 37 + 38 both shipped this; S39-CR-003 (86) every test uses `i32` — `forecast(into_present(1.5f32))` is uncovered (repeat of S38-CR-002 for temporal); S39-CR-004 (82) `_FRAME_IDENTITY_AD_NAMES` is a misleading name post-Stage-39 (12 temporal builtins live in a frame-named set; docstring still says "the 12 frame builtins"). Two LOWs: S39-CR-005 (75) ledger overshoots the Stage 38 lesson by deleting all landed/shipped markers; S39-CR-006 (72) ledger doesn't document the AD-set reuse decision or the transition-name verb-form pivot. All six findings are fixable in a single Inc 4 sweep before the 3-clean-gate closure sequence.

---

**Verdict**: CLEAN-WITH-FIXES | 1 HIGH + 3 MEDIUM + 2 LOW | S39-CR-001 (TyTemporal refinement/compat symmetry gap) is the gate-1 blocker; fix in progress on working tree
