# Stage 40 Inc 4 closure gate-3 code review

**Reviewer**: code-review subagent (Claude Opus 4.7, 1M context)
**Date**: 2026-05-17
**Scope**: HEAD on `main` = `d914557` ("Stage 40 Inc 4 closure gate-2 fix sweep"). Stage 40 surface = `cb36bbc..d914557`. Gate-2 surface = `e8fb593..d914557`.
**Filter**: confidence >= 80 (gate-3 strictness — CLEAN target).
**Files reviewed**:
- `C:/Projects/Kovostov-Native/helixc/frontend/typecheck.py`
- `C:/Projects/Kovostov-Native/helixc/frontend/autodiff.py`
- `C:/Projects/Kovostov-Native/helixc/ir/lower_ast.py`
- `C:/Projects/Kovostov-Native/helixc/examples/dogfood_13_modal_lifecycle.hx`
- `C:/Projects/Kovostov-Native/helixc/tests/test_stage40_modal.py`

Reference template: `docs/audit-stage40-inc4-gate2-codereview.md` (gate-2 review). Cycle counter: 0/3.

---

**VERDICT: 0 HIGH, 1 MEDIUM, 0 LOW, 4 OBS**

---

## HIGH (90-100)

*(none)*

---

## MEDIUM (80-89)

### S40-CR-G3-001 — F1 launder guard does not check whether `from_X` was shadowed; emits a false-positive when a user fn shadows a modal eliminator

**Severity**: MEDIUM
**Confidence**: 82
**File**: `C:/Projects/Kovostov-Native/helixc/frontend/typecheck.py`
**Lines**: 3564-3585 (F1 cross-modal guard predicate).

The gate-2 H2 fix established `self._shadowed_builtin_names` and rewrites `bn` to a sentinel at the call-dispatch entry (line 2848-2849), so the OUTER call falls through to user-fn lookup. But the F1 cross-modal launder guard at lines 3564-3568 inspects `expr.args[0].callee.name` (the INNER call's syntactic name) without checking whether that name has been shadowed. As a result, when a user shadows a modal eliminator (e.g., `from_uncertain`, `from_known`), the F1 guard treats the call as if the builtin had stripped a modal wrapper — but the call actually dispatched to the user fn and never stripped anything.

Reproducible cascade:
```
fn from_uncertain(x: i32) -> i32 { x }
fn main() -> i32 {
    let k: Known<i32> = into_known(from_uncertain(42));
    from_known(k)
}
```
Expected errors: 1 shadow diagnostic at the fn-decl (the H2 promise: ONE diagnostic per shadow).
Actual errors: 2 — the shadow diagnostic AND a false-positive F1 "into_known(from_uncertain(...)) launders an Uncertain<T> into Known<T>" at the call site, even though no `Uncertain<T>` ever existed in the program (the inner call evaluated `42 -> i32` via the user fn).

The F1 false positive structurally misrepresents the bug: it claims an Uncertain<T> was laundered, when in fact the user's `from_uncertain` is a benign int→int passthrough. The guard's `not isinstance(arg_tys[0], TyUnknown)` clause (the gate-2 M1 fix) does NOT catch this case, because the user fn returns a valid `TyPrim("i32")`, not `TyUnknown`. Same issue applies symmetrically to all 4 shadowable `from_X` eliminators (`from_known`, `from_believed`, `from_goal`, `from_uncertain`).

**Why this matters for gate-3**: the H2 fix's stated promise (audit-stage40-inc4-gate2-codereview.md, line 49 onwards) is "the user sees ONE diagnostic, not 1 + N noise". The promise holds for the bare-name call-dispatch path (H2 fixed it explicitly), but the F1 guard still fires off-by-one extra diagnostic when the shadowed name appears as an INNER arg to a modal intro. This is a subtle interaction between the gate-2 H2 fix and the gate-2 F1 generalization that neither audit covered explicitly.

**Citation**:
```python
# typecheck.py:3564-3568 (HEAD d914557)
if (len(expr.args) >= 1
        and isinstance(expr.args[0], A.Call)
        and isinstance(expr.args[0].callee, A.Name)
        and expr.args[0].callee.name
            in _modal_elim_kind
        and not isinstance(arg_tys[0], TyUnknown)):
```

The predicate inspects the bare callee name but never asks whether `expr.args[0].callee.name in self._shadowed_builtin_names` — that single clause would close the false-positive symmetrically with the H2 fix's bare-name skip.

**Recommended fix** (1 line in the F1 predicate):
```python
if (len(expr.args) >= 1
        and isinstance(expr.args[0], A.Call)
        and isinstance(expr.args[0].callee, A.Name)
        and expr.args[0].callee.name in _modal_elim_kind
        and expr.args[0].callee.name
            not in self._shadowed_builtin_names    # <-- new
        and not isinstance(arg_tys[0], TyUnknown)):
```

Mechanically isomorphic to the bare-`bn` skip the H2 fix added; just applied to the inner callee name as well.

---

## LOW (70-79)

*(none above threshold; the four LOW items below documented as observations only.)*

---

## Observations (sub-threshold, informational)

### S40-CR-G3-OBS-1 — Gate-2 F1 tests pin diagnostic CONTENT inconsistently (some pin upgrade hints, some only the "launders" word)

**Confidence**: 72 — observation (test-strength gap).

Of the 6 new gate-2 F1 tests in `test_stage40_modal.py`:
- `test_stage40_gate2_f1_blocks_believed_to_known_laundering` (line 731): asserts `"launders" in str(e) and "confirm" in str(e)` — CONTENT pinned to the audited upgrade hint.
- `test_stage40_gate2_f1_blocks_goal_to_known_laundering` (line 748): asserts `"launders" and "act_on"` — CONTENT pinned.
- `test_stage40_gate2_f1_blocks_known_to_believed_downgrade` (line 763): asserts `"launders" and "no Known -> Believed"` — CONTENT pinned to the deferred-direction hint.
- `test_stage40_gate2_f1_blocks_known_to_uncertain_laundering` (line 778): asserts ONLY `"launders"`. A future copy-paste regression that swapped `source_kind`/`target_kind` labels in the diagnostic would NOT fail this test.
- `test_stage40_gate2_f1_allows_all_4_self_rewraps` (line 790): asserts `errs == []` — no content to pin (negative case).
- `test_stage40_gate2_f1_all_12_cross_modal_combinations_reject` (line 807): asserts ONLY `"launders"` across 12 combos. Same label-swap regression would not fail.

The audit-spec methodology question asked: "do they ALSO pin the diagnostic CONTENT (not just count), so future copy-paste regressions land a test failure?" Answer for the 4 named-direction tests: yes. Answer for the matrix sweep + 1 Known→Uncertain blanket test: partial — only the "launders" word is pinned, not the per-direction labels or kind-specific hints. A copy-paste bug that swapped `f"{bn}(from_{source_kind}(...))"` with `f"{bn}(from_{target_kind}(...))"` in the diagnostic body would pass all 6 tests (because each test only checks for `"launders"`, and the diagnostic body always contains that word).

Below MEDIUM threshold because (a) the four direction-specific tests DO pin enough content to catch the most-likely copy-paste regressions (mis-routing the hint dict, missing the deferred-direction branch), and (b) the matrix test catches the categorical regression (any of the 12 combos silently allowed). Documented for future test-strengthening pass.

### S40-CR-G3-OBS-2 — `test_stage40_gate2_medium1_typechecker_reentrancy_no_stale_shadows` validates `__init__` but does not exercise the `check()`-time reset

**Confidence**: 78 — observation (test coverage gap on the gate-2 MEDIUM-1 fix it claims to pin).

The test docstring claims to validate two reset paths: "Post-fix, `__init__` + `check()` both clear the set." The test creates TWO separate `TypeChecker` instances (`tc1 = TypeChecker(prog1)` and `tc2 = TypeChecker(prog2)`), so the `check()`-time reset at line 588 of `typecheck.py` is never exercised — only the `__init__` reset at line 551 is. A regression that removed the line-588 reset (`self._shadowed_builtin_names = set()` inside `check()`) but kept the `__init__` reset would NOT fail this test.

The proper re-entrancy test pattern is: use the SAME TypeChecker instance, call `check()` twice with different programs (or re-attach `tc.prog = prog2` and re-call `check()`). The gate-2 fix added both reset sites; the test only pins one. Below MEDIUM threshold because (a) the gate-2 fix is double-belted (defense in depth), and (b) standard usage patterns in `helixc.examples.run.typecheck()` create a fresh instance per program. But the `check()`-time reset was added explicitly for "LSP / REPL / test harness reuse" (line 547) — exactly the case the test doesn't cover.

### S40-CR-G3-OBS-3 — `test_stage40_f1_known_limitation_let_bypass` violates the gate-2 `test_stage40_gate2_*` naming convention

**Confidence**: 72 — observation (cosmetic naming inconsistency).

9 of the 10 new gate-2 tests use the `test_stage40_gate2_*` prefix. The exception is `test_stage40_f1_known_limitation_let_bypass` at line 886, which is missing the `gate2_` infix even though it was added by the gate-2 commit (`d914557`) and lives in the gate-2 section comment block ("Stage 40 closure gate-2 H2 + M1 + audit-trail backfills"). A consistent name would be `test_stage40_gate2_f1_known_limitation_let_bypass` — matches the surrounding tests and makes grep by gate trivial.

Functionally harmless (the test runs and asserts correctly); flagged only because the gate-3 spec asked verbatim about "test naming consistency" and this is the one outlier across the 10-test sweep.

### S40-CR-G3-OBS-4 — Gate-2 MEDIUM-1 fix comment claims to "mirror the cascade-suppression set discipline at lines 542-560" but the mirror is partial

**Confidence**: 72 — observation (comment-vs-code drift internal to the gate-2 fix).

Both `_seen_unbound` (cascade-suppression for unbound-name diagnostics, line 542) and `_shadowed_builtin_names` (shadow-dispatch suppression, line 551) are `set[str]` initialized in `__init__`. The comment at lines 549-550 says the shadow set "mirrors the cascade-suppression set discipline at lines 542-550". But the disciplines differ in one key respect:

- `_seen_unbound` resets per-iteration inside the Pass-2 saturation loop (line 666: `self._seen_unbound = set()`). Cascade suppression is iteration-local because each iteration retracts intermediate errors.
- `_shadowed_builtin_names` resets ONLY at `__init__` and at `check()` entry (line 588). It must NOT reset per saturation-loop iteration, because shadow names are determined at Pass 1 (`_register_fn`) and must persist across all body-check iterations.

The two reset cadences are deliberately opposite. The "mirrors the discipline" phrasing in the gate-2 comment elides this distinction and could mislead a future maintainer who wants to "make the disciplines consistent" by adding a per-iteration reset that would re-introduce the H2 bug. Recommended re-phrasing: "shares the cascade-suppression set IDIOM but with different reset cadence — initialized in `__init__` + cleared at `check()` entry only (NOT per Pass-2 iteration, since shadow names persist across body-check retries)".

Below LOW threshold because the comment is technically true (both ARE sets reset in `__init__`) and the actual code is correct; the drift is purely in the explanation.

---

## Items considered and dismissed (confidence < 70)

### `<<shadowed_builtin_skip>>` sentinel collision risk

**Confidence**: 95 (no finding — collision impossible).

The sentinel string `<<shadowed_builtin_skip>>` contains `<<` and `>>` which are not valid characters in any Helix identifier (the parser rejects them as identifier chars). All 17 `bn` comparisons in the builtin-dispatch block (`if bn == "detach"`, etc.) check against valid identifier literals. The sentinel cannot match any of them, and it cannot appear in `self.functions` (the user-fn registry) for the same reason. The `if bn in _modal_intro:` / `in _temporal_intro:` / etc. dict membership checks are also safe — the sentinel is not a key in any dispatch dict. Dismissed.

### `_shadowed_builtin_names` persistence

**Confidence**: 92 (no finding).

Grepped `_shadowed_builtin_names` across the repo: appears in `typecheck.py` (definition + 2 use sites + 2 reset sites) and `test_stage40_modal.py` (1 reentrancy test). No serialization layer (no `pickle`, `json.dumps`, `to_dict`, etc.) touches the TypeChecker instance state. The set is purely a per-instance runtime cache. No persistence risk. Dismissed.

### `_modal_upgrade_hint` lookup case-sensitivity

**Confidence**: 95 (no finding).

Both `source_kind` and `target_kind` come from dicts whose values are lowercase string literals (`"known"`, `"believed"`, `"goal"`, `"uncertain"`). The hint-dict keys are also lowercase tuples. The lookup `_modal_upgrade_hint.get((source_kind, target_kind))` is consistent. The `.capitalize()` is applied only at message-emission time (for the user-visible "Believed -> Known" rendering), not at lookup time. Dismissed.

### "Phase-0 has no Known -> Goal transition" message clunkiness

**Confidence**: 65 — dismissed at threshold.

The full hint text is: "Phase-0 has no Known -> Goal transition; if this direction is semantically meaningful, request a future-stage spec and keep the value in its current modal kind until then". Read in full it's a complete, actionable sentence. The first 6 words ("Phase-0 has no Known -> Goal transition") read clunky in isolation but the continuation gives the user (a) confirmation it's not their bug, (b) an escape hatch ("request a future-stage spec"), and (c) the immediate workaround ("keep the value in its current modal kind"). Dismissed as a matter of preference, not a defect.

### Comment-vs-code drift on the gate-2 F2-cascade comment

**Confidence**: 90 (no finding — gate-2 correctly updated the comment).

The original gate-1 F2 comment at `_register_fn` line 947 claimed "the diagnostic alone gates the typecheck pass" (gate-2 finding S40-CR-G2-001 flagged this as false). The gate-2 fix updated the comment at line 947 to read: "Continue registration so downstream code doesn't crash on a missing FunctionSig." — the false claim removed, the technical reason for continued registration preserved. No drift in the post-gate-2 surface. Dismissed.

### Dogfood `.hx` peer test in `test_reflection.py` (S40-CR-G2-004 LOW from gate-2)

**Confidence**: 78 — pre-existing residual (gate-2 LOW), documented deferred per the audit-context spec.

Carried over from gate-2 as "documented in progress doc, deferred" — the gate-3 spec explicitly tells me not to re-flag.

### `_FRAME_IDENTITY_AD_NAMES` docstring drift (S40-CR-G2-003 LOW from gate-2)

**Confidence**: 78 — pre-existing residual, documented deferred per the audit-context spec.

Same as above.

### Cross-modal verbatim style consistency with Stage 39 temporal arms

**Confidence**: 95 (no finding — same conclusion as gate-2 dismissal).

The Stage 40 modal arms remain byte-shape-identical to the Stage 39 temporal arms modulo the F1 launder guard insertion. The F1 guard is structurally distinct from anything in the temporal family (Stage 39 has no temporal-laundering guard because temporal kinds are factual, not epistemic), so its presence is correct. Dismissed.

---

## Convention-check summary

| Convention | Stage 40 HEAD (`d914557`) status |
|---|---|
| Builtin arm shape matches Stage 37/38/39 verbatim | OK |
| `_BUILTIN_NAMES` updated for all 10 new modal verbs | OK |
| Identity-lowering at IR (all 10 entries) | OK |
| `AD_KNOWN_PURE_CALLS` updated (10 entries) | OK |
| `_FRAME_IDENTITY_AD_NAMES` updated (10 entries) | OK functionally; docstring drift carried per gate-2 LOW residual |
| TyModal in all 6 refinement/compat surfaces | OK |
| Dogfood `.hx` peer test in `test_reflection.py` | VIOLATED (carried per gate-2 LOW residual) |
| F1/F2 gate-1+gate-2 fix sweep regression tests | OK (10 new gate-2 tests; pre/post = 36 -> 56) |
| Gate-2 H2 + M1 + F1 generalization + MEDIUM-1 fixes | OK functionally; subtle interaction flaw (S40-CR-G3-001) |
| Cascade noise on shadow / launder paths (post-gate-2) | OK on bare-name dispatch; FAILED on inner-arg shadow (S40-CR-G3-001) |
| `<<shadowed_builtin_skip>>` sentinel safety | OK (invalid-identifier sentinel; no collision possible) |
| Gate-2 audit doc on disk for ledger continuity | OK (3 docs on disk: silent-failures, type-design, codereview) |
| Test naming consistency (gate2_* prefix) | 9/10 OK; 1 outlier (OBS-3) |
| Test pins diagnostic content not just count | Mixed (OBS-1) |
| Test coverage of `check()`-time reset path | Missing (OBS-2) |

---

## 5-line summary

Gate-3 finds the gate-2 fix sweep substantially clean but exposes ONE MEDIUM (conf 82): the F1 cross-modal launder guard at typecheck.py:3564-3585 inspects `expr.args[0].callee.name` (the inner call's bare name) without checking `self._shadowed_builtin_names` — so when a user shadows a `from_X` eliminator, the F1 guard fires a false-positive "launders" diagnostic on top of the H2 fn-decl shadow error, partially defeating the H2 promise of "ONE diagnostic per shadow". Mechanically a 1-line fix: add `and expr.args[0].callee.name not in self._shadowed_builtin_names` to the F1 predicate, isomorphic to the bare-`bn` skip at the dispatch entry. Four observations document test-strength gaps in the gate-2 sweep (matrix test pins only the "launders" word not the per-direction labels; the reentrancy test exercises `__init__` reset but not `check()` reset; one test misses the `gate2_` naming convention; the MEDIUM-1 comment's "mirrors the cascade-suppression discipline" phrasing elides the deliberately-opposite reset cadence). Sentinel-collision risk is zero (the `<<...>>` token cannot match any valid Helix identifier). `_shadowed_builtin_names` is not persisted to any serialization layer. The 12 cross-modal hints are functionally correct and case-consistent; the "Phase-0 has no Known -> Goal transition" phrasing read clunky in isolation but the full hint sentence is complete and actionable (dismissed). Gate-3 verdict: ONE MEDIUM + FOUR OBS — not CLEAN; gate-3 fix sweep recommended to close the F1/H2 interaction, then re-run gate-3.
