# Stage 41 Inc 4 closure gate-1 code review

**Reviewer**: code-review subagent (Claude Opus 4.7, 1M context)
**Date**: 2026-05-17
**Scope**: HEAD on `main` = `7448bf5` ("Stage 41 Inc 1 + Inc 2 + Inc 3: causal/intent types shipped end-to-end"). Stage 41 surface = `5dd478a..7448bf5`.
**Filter**: confidence >= 80 (gate-1 strictness per Stage 40 closure protocol).
**Files reviewed**:
- `C:/Projects/Kovostov-Native/helixc/frontend/typecheck.py`
- `C:/Projects/Kovostov-Native/helixc/frontend/autodiff.py`
- `C:/Projects/Kovostov-Native/helixc/ir/lower_ast.py`
- `C:/Projects/Kovostov-Native/helixc/examples/dogfood_14_causal_lifecycle.hx`
- `C:/Projects/Kovostov-Native/helixc/examples/run.py`
- `C:/Projects/Kovostov-Native/helixc/tests/test_stage41_causal.py`

Reference template: `docs/audit-stage40-inc4-gate2-codereview.md` (the canonical example).
Cross-reference: `docs/audit-stage40-inc4-gate3-codereview.md` (the gate-3 finding this audit reproduces verbatim at the causal site).

Test surface: `pytest helixc/tests/test_stage41_causal.py` -> 23/23 pass at HEAD.

---

**VERDICT: 1 HIGH, 2 MEDIUM, 3 LOW, 1 OBS**

---

## HIGH (90-100)

### S41-CR-G1-001 — F1 cross-causal launder guard missing `inner_is_shadowed` clause; verbatim replay of Stage 40 gate-3 H1 finding

**Severity**: HIGH
**Confidence**: 95
**File**: `C:/Projects/Kovostov-Native/helixc/frontend/typecheck.py`
**Lines**: 3781-3786 (the Stage 41 F1 guard predicate).

The Stage 40 closure took THREE audit gates because gate-1 + gate-2 missed exactly this interaction: the F1 cross-modal launder guard inspected `expr.args[0].callee.name` (the INNER call's syntactic name) without checking `self._shadowed_builtin_names`. When a user shadowed a `from_X` eliminator, the H2 fn-decl shadow diagnostic AND a spurious F1 "launders" diagnostic both fired, violating H2's "1 + 0 noise" invariant. The Stage 40 gate-3 fix (commit `38f5598`) added an `inner_is_shadowed` precondition to the modal F1 guard at typecheck.py:3643-3656.

**Stage 41 did not carry that fix forward.** The Stage 41 causal F1 guard at lines 3781-3786 is byte-shape-identical to the PRE-gate-3 modal guard — same predicate without the `inner_is_shadowed` clause. Reproducible:

```
$ python -c "
from helixc.frontend.parser import parse
from helixc.frontend.typecheck import typecheck

src = '''
fn from_cause(x: i32) -> i32 { x }
fn main() -> i32 {
    let e: Effect<i32> = into_effect(42);
    let r = into_effect(from_cause(7));
    from_effect(r)
}
'''
prog = parse(src, include_stdlib=True)
for e in typecheck(prog): print(e)
"
```

Yields 3 errors:
1. `2:1: function 'from_cause' shadows a reserved builtin name ...` (H2 shadow diagnostic — correct, single source-of-truth)
2. `5:13: into_effect(from_cause(...)) launders a Cause<T> into Effect<T> with no causal-transition audit.` (**FALSE POSITIVE** — the user's `from_cause` is a benign int→int passthrough; no `Cause<T>` ever existed in the program; the inner call dispatched to the user fn and returned `i32`)
3. `6:5: from_effect() requires Effect<T>, got ?{into_effect}` (cascade noise from the TyUnknown returned by the false-positive F1 guard)

Control: same probe with `from_uncertain` shadowed (Stage 40 surface) yields exactly 1 diagnostic (the H2 shadow). The asymmetry between the modal and causal sites is empirically observed at HEAD.

**Why this is HIGH, not MEDIUM**: Stage 40 gate-3 found the equivalent issue at MEDIUM conf 82 because the modal launder text said "Uncertain<T>" — which could plausibly exist elsewhere in the program. The causal site is structurally worse — there are 4 shadowable `from_X` eliminators (`from_cause`, `from_effect`, `from_joint`, `from_independent`), each of which produces a false-positive when shadowed. The H2 invariant ("ONE diagnostic per shadow") is violated for every causal eliminator. The fix is also explicitly documented in code (lines 3630-3648 of the modal arm) AND in the Stage 40 gate-3 audit doc on disk — making the omission a clear copy-paste regression rather than a missed novel finding.

**Citation** (Stage 41 F1 guard, missing the `inner_is_shadowed` clause):
```python
# typecheck.py:3781-3786 (HEAD 7448bf5)
if (len(expr.args) >= 1
        and isinstance(expr.args[0], A.Call)
        and isinstance(expr.args[0].callee, A.Name)
        and expr.args[0].callee.name
            in _causal_elim_kind
        and not isinstance(arg_tys[0], TyUnknown)):
```

**Modal site (post-Stage-40-gate-3, lines 3643-3656)**:
```python
inner_is_shadowed = (
    len(expr.args) >= 1
    and isinstance(expr.args[0], A.Call)
    and isinstance(expr.args[0].callee, A.Name)
    and expr.args[0].callee.name
        in self._shadowed_builtin_names
)
if (len(expr.args) >= 1
        and isinstance(expr.args[0], A.Call)
        and isinstance(expr.args[0].callee, A.Name)
        and expr.args[0].callee.name
            in _modal_elim_kind
        and not isinstance(arg_tys[0], TyUnknown)
        and not inner_is_shadowed):       # <-- missing in causal
```

**Recommended fix** (verbatim mirror of the modal site):
```python
inner_is_shadowed = (
    len(expr.args) >= 1
    and isinstance(expr.args[0], A.Call)
    and isinstance(expr.args[0].callee, A.Name)
    and expr.args[0].callee.name
        in self._shadowed_builtin_names
)
if (len(expr.args) >= 1
        and isinstance(expr.args[0], A.Call)
        and isinstance(expr.args[0].callee, A.Name)
        and expr.args[0].callee.name
            in _causal_elim_kind
        and not isinstance(arg_tys[0], TyUnknown)
        and not inner_is_shadowed):
```

Plus a regression test pinning the parity:
```python
def test_stage41_gate1_f1_inner_shadow_no_cascade():
    """User shadowing from_X must produce ONE diagnostic (H2 shadow),
    not 1 + 1 spurious launder + 1 cascade arg-mismatch."""
    src = '''
fn from_cause(x: i32) -> i32 { x }
fn main() -> i32 {
    from_effect(into_effect(from_cause(7)))
}
'''
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    shadow_errs = [e for e in errs if "shadows" in str(e)]
    launder_errs = [e for e in errs if "launders" in str(e)]
    assert len(shadow_errs) == 1
    assert len(launder_errs) == 0, "F1 must skip when inner is shadowed"
```

This is the answer to bug-check #8 in the audit prompt — the `inner_is_shadowed` cascade-suppression was NOT applied at Stage 41, and the H2 cascade IS triggered for all 4 shadowable causal eliminators.

---

## MEDIUM (80-89)

### S41-CR-G1-002 — Cross-causal F1 launder test surface covers only 1 of 12 directions; matrix sweep missing

**Severity**: MEDIUM
**Confidence**: 88
**File**: `C:/Projects/Kovostov-Native/helixc/tests/test_stage41_causal.py`
**Lines**: 222-232 (single-direction F1 test); contrast with Stage 40 `test_stage40_gate2_f1_all_12_cross_modal_combinations_reject`.

The Stage 40 closure ledger explicitly added a 12-combo matrix test (`test_stage40_gate2_f1_all_12_cross_modal_combinations_reject` at `test_stage40_modal.py:807`) that sweeps every cross-modal `into_X(from_Y(...))` direction with `X != Y`. The motivation was preventing a categorical regression where a single hint-dict entry silently gets deleted and 1/12 directions silently allows laundering.

Stage 41 ships only `test_stage41_f1_blocks_cross_causal_laundering` at lines 222-232 covering exactly ONE direction (`into_cause(from_effect(...))`). The other 11 of 12 cross-causal directions are uncovered:

| | from_cause | from_effect | from_joint | from_independent |
|--|--|--|--|--|
| **into_cause**       | self-rewrap | **TESTED** | UNCOVERED | UNCOVERED |
| **into_effect**      | UNCOVERED   | self-rewrap | UNCOVERED | UNCOVERED |
| **into_joint**       | UNCOVERED   | UNCOVERED   | self-rewrap | UNCOVERED |
| **into_independent** | UNCOVERED   | UNCOVERED   | UNCOVERED | self-rewrap |

A regression that broke any of the 11 uncovered F1 directions (e.g., a stray `continue` in the kind-comparison loop, an off-by-one in the elim_kind dict, a typo in `target_kind != source_kind`) would not fail any test. The audit-spec methodology question 1d ("every laundering hint") returns "1 of 12 (8%) covered".

The 3 entries in `_causal_upgrade_hint` (`cause->effect`, `effect->joint`, `joint->independent`) are also untested at the diagnostic-content level — no test asserts that `into_effect(from_cause(...))` produces a hint mentioning `propagate`, or `into_joint(from_effect(...))` mentions `aggregate`, or `into_independent(from_joint(...))` mentions `isolate`. A regression that flipped the dict values (e.g., `("cause", "effect"): "use aggregate(...)"`) would pass the existing tests.

**Recommended fix**: Add a Stage-40-mirror matrix test:

```python
def test_stage41_f1_all_12_cross_causal_combinations_reject():
    """4 kinds x 3 wrong targets = 12 cross-causal launder
    combinations. All must reject with the "launders" diagnostic."""
    kinds = ["cause", "effect", "joint", "independent"]
    for src_k in kinds:
        for tgt_k in kinds:
            if src_k == tgt_k:
                continue
            src = (f"fn main() -> i32 {{ "
                   f"from_{tgt_k}(into_{tgt_k}("
                   f"from_{src_k}(into_{src_k}(42)))) }}")
            prog = parse(src, include_stdlib=True)
            errs = typecheck(prog)
            assert any("launders" in str(e) for e in errs), \
                f"into_{tgt_k}(from_{src_k}(...)) should launder-reject"

def test_stage41_f1_upgrade_hints_pin_transition_names():
    """The 3 forward-direction hints must name the audited transition."""
    cases = [
        ("cause", "effect", "propagate"),
        ("effect", "joint", "aggregate"),
        ("joint", "independent", "isolate"),
    ]
    for src_k, tgt_k, want_verb in cases:
        src = (f"fn main() -> i32 {{ "
               f"from_{tgt_k}(into_{tgt_k}("
               f"from_{src_k}(into_{src_k}(42)))) }}")
        prog = parse(src, include_stdlib=True)
        errs = typecheck(prog)
        joined = " ".join(str(e) for e in errs)
        assert "launders" in joined and want_verb in joined, \
            f"hint for {src_k}->{tgt_k} must mention {want_verb}"
```

Adding ~30 LoC closes both the coverage gap and the hint-content pinning gap.

---

### S41-CR-G1-003 — 9 of 12 cross-causal directions fall through to generic "Phase-0 has no X -> Y transition" hint without safety-anchored framing

**Severity**: MEDIUM
**Confidence**: 80
**File**: `C:/Projects/Kovostov-Native/helixc/frontend/typecheck.py`
**Lines**: 3757-3767 (`_causal_upgrade_hint` dict; 3 entries) and 3794-3804 (generic fallback hint).

`_causal_upgrade_hint` defines hints only for the 3 forward-chain transitions (`cause->effect`, `effect->joint`, `joint->independent`). The remaining 9 of 12 cross-causal directions fall through to the generic message:

> "Phase-0 has no `Source` -> `Target` transition; if this direction is semantically meaningful, request a future-stage spec and keep the value in its current causal kind until then"

This was exactly the Stage 40 closure gate-3 silent-failure LOW finding that triggered the special-cased `("uncertain", "known")`, `("uncertain", "believed")`, `("uncertain", "goal")` safety hints at lines 3603-3619: the generic "request a future-stage spec" framing **misleadingly suggests a future feature** when several directions are semantically incoherent (you cannot "un-aggregate" a Joint back to a Cause without losing information; you cannot manufacture causal independence by syntactic rewrap).

The reverse-direction causal moves are categorically Phase-0-deferred-or-incoherent:
- `effect -> cause`, `joint -> cause`, `independent -> cause` — running causation BACKWARDS through the type system; this is post-hoc rationalization, not Phase-0 audited.
- `joint -> effect`, `independent -> effect`, `independent -> joint` — going from "more entangled" to "less entangled" via syntactic rewrap manufactures an isolation property the runtime cannot verify.

The 6 forward-skip directions (`cause -> joint`, `cause -> independent`, `effect -> independent`) are not in the dict either — these CAN be reached legitimately via chained audited transitions (`propagate` + `aggregate`, etc.) but the user has no hint pointing at the chain.

The progress doc (`docs/stage41-progress-2026-05-17.md`, gate-1 fix sweep section) explicitly flagged this as a known finding. At HEAD `7448bf5` the fix is not yet landed.

**Recommended fix**: Add the 6 reverse-direction safety hints AND the 3 forward-chain hints:

```python
_causal_upgrade_hint = {
    # Forward audited (already present)
    ("cause", "effect"): "use `propagate(c)` ...",
    ("effect", "joint"): "use `aggregate(e)` ...",
    ("joint", "independent"): "use `isolate(j)` ...",

    # Forward-chain hints (currently fall-through)
    ("cause", "joint"):
        "chain `propagate(c)` then `aggregate(e)` — direct "
        "Cause -> Joint shortcut is not audited in Phase-0",
    ("cause", "independent"):
        "chain `propagate`, `aggregate`, `isolate` — direct "
        "Cause -> Independent shortcut is not audited in Phase-0",
    ("effect", "independent"):
        "chain `aggregate(e)` then `isolate(j)` — direct "
        "Effect -> Independent shortcut is not audited in Phase-0",

    # Reverse-direction safety hints (currently fall-through to
    # misleading "request a future-stage spec")
    ("effect", "cause"):
        "Phase-0 does not run causation backwards; an Effect cannot "
        "be re-cast as its own Cause via syntactic rewrap",
    ("joint", "cause"):
        "Phase-0 does not run causation backwards; pick the specific "
        "upstream Cause<T> at observation time, not after aggregation",
    ("independent", "cause"):
        "an Independent<T> by definition has no upstream Cause; the "
        "isolation hypothesis is incompatible with a Cause rewrap",
    ("joint", "effect"):
        "Phase-0 has no Joint -> Effect de-aggregation; the multi-"
        "source observation cannot be syntactically narrowed",
    ("independent", "effect"):
        "an Independent<T> has no upstream; it cannot be syntactically "
        "re-attached as an Effect<T> of unspecified causes",
    ("independent", "joint"):
        "an Independent<T> contradicts the multi-source Joint<T> "
        "premise; the isolation hypothesis must be re-evaluated",
}
```

Closes the silent-failure-class gap that Stage 40 gate-3 had to backfill.

---

## LOW (70-79)

### S41-CR-G1-004 — Dogfood cross-stack probe uses degenerate witness input `1`

**Severity**: LOW
**Confidence**: 78
**File**: `C:/Projects/Kovostov-Native/helixc/examples/dogfood_14_causal_lifecycle.hx`
**Lines**: 63, 68.

```hx
let cs: i32 = cross_stack_known_cause(1);
...
let cs_ok: i32 = if cs == 1  { 1 } else { 0 };
```

The cross-stack composition probe — the only test of `Known<Cause<i32>>` 5-stack composition in the dogfood — uses `1` as input. The check is `cs == 1`. A regression where the wrapper-erasure cascade silently dropped the inner value and the function returned the constant `1` (e.g., via a buggy identity lowering that hard-coded a literal) would still pass this gate. Stage 37/38/39/40 dogfoods all use non-degenerate witnesses (7, 11, 13, etc.) precisely to catch identity-laundering regressions.

The `causal_lifecycle` calls at lines 59-61 use 10, 14, 18 — non-degenerate. The asymmetry is purely the cross-stack probe.

**Recommended fix**: Change line 63 to `cross_stack_known_cause(7)` and line 68 to `if cs == 7 { 1 } else { 0 }`. Two-line edit.

(The progress doc identifies this as a known gate-1 LOW; this audit re-flags it for completeness against the audited HEAD.)

---

### S41-CR-G1-005 — `_FRAME_IDENTITY_AD_NAMES` docstring drift now factually wrong by ~3.75x (REPEAT of S40-CR-G2-003 + 11 more entries)

**Severity**: LOW
**Confidence**: 75
**File**: `C:/Projects/Kovostov-Native/helixc/frontend/autodiff.py`
**Lines**: 177-186 (docstring); 187-219 (the frozenset itself).

S39-CR-004 (MEDIUM conf 82) flagged this docstring drift in Stage 39. S40-CR-G2-003 (LOW conf 78) re-flagged it after Stage 40 added 10 entries. The docstring at lines 177-186 still says **"the 12 frame builtins are identity-lowered at IR"** — the actual set now contains:

- 12 frame builtins (original)
- 12 temporal builtins (Stage 39)
- 10 modal builtins (Stage 40)
- **11 causal builtins (Stage 41 adds)**

**Total: 45 names; the docstring describes 12 of them (~27%).** Same `_FRAME_IDENTITY_AD_NAMES` name pretends to be frame-only even though it's the universal wrapper-identity registry for 5 type families.

Stage 41 makes the misnaming strictly worse. Same recommended fix as S39-CR-004 / S40-CR-G2-003: rename to `_IDENTITY_WRAPPER_AD_NAMES` (2 import sites in `autodiff_reverse.py:53` + `autodiff.py:187` + the call site at `autodiff_reverse.py:684`) AND update the docstring header to "45 wrapper-identity builtins across frame (12) / temporal (12) / modal (10) / causal (11) families". Documented as deferred per the progress doc; flagged because the residual gets larger every stage.

---

### S41-CR-G1-006 — No `test_dogfood_14_causal_lifecycle` peer in `test_reflection.py` (REPEAT of S40-CR-G2-004 / S39-CR-002)

**Severity**: LOW
**Confidence**: 78
**File**: `C:/Projects/Kovostov-Native/helixc/tests/test_reflection.py`

Stage 39 missed adding `test_dogfood_12_temporal_lifecycle` and the closure trail backfilled it. Stage 40 then missed adding `test_dogfood_13_modal_lifecycle` (S40-CR-G2-004, never backfilled). Stage 41 now ships `dogfood_14_causal_lifecycle.hx` AND fails to add `test_dogfood_14_causal_lifecycle`:

```
$ grep -nE "test_dogfood_(11|12|13|14)" helixc/tests/test_reflection.py
229:def test_dogfood_11_spatial_frames():
242:def test_dogfood_12_temporal_lifecycle():
# no test_dogfood_13_*
# no test_dogfood_14_*
```

Consequences:
1. Regressions in parsing / typechecking `dogfood_14_causal_lifecycle.hx` (e.g., the `Known<Cause<i32>>` 5-stack composition at line 54) are NOT covered by the reflection test surface.
2. The `python -m helixc.examples.run causal` entry path through `run.py:104-109` (added in this stage) is uncovered.
3. The lesson Stage 39 wrote down is being lost; Stage 41 inherits Stage 40's failure to absorb it.

**Recommended fix**: Add `test_dogfood_13_modal_lifecycle` AND `test_dogfood_14_causal_lifecycle` to `test_reflection.py` immediately after the temporal entry at line 253. Byte-for-byte mirror of the existing pattern modulo the file name and demo key.

---

## Observations (sub-threshold, informational)

### S41-CR-G1-OBS-1 — Test pins are loose on diagnostic content (variants of OBS-1 from Stage 40 gate-3)

**Confidence**: 70 — observation (test-strength gap).

Of the new Stage 41 tests:
- `test_stage41_f1_blocks_cross_causal_laundering` (line 222): asserts ONLY `"launders" in str(e)`. A regression that swapped `source_kind`/`target_kind` labels in the message body would not fail.
- `test_stage41_inc1_from_cause_rejects_effect` (line 88): asserts ONLY `"Cause" in str(e)`. A regression that broke the lowercase-to-capitalized rendering in `_fmt` (e.g., returned "cause" instead of "Cause") would fail correctly, but a regression in the actual "requires Cause<T>" → "requires Effect<T>" label-swap would still pass IF the error message happened to include "Cause" anywhere (e.g., in a TyCausal repr that mentioned the inner).
- `test_stage41_inc1_all_12_wrong_kind_combinations`: pins only the expected kind label; doesn't pin "requires" or the GOT type formatting.

`test_stage41_inc2_propagate_rejects_effect_input` (line 198) is the gold-standard counter-example — it pins both `"Cause"` and `"propagate"`. The other tests should follow this pattern.

Below MEDIUM threshold because the existing pins catch the categorical regressions (a totally-empty errs list, or a totally-wrong kind label). Documented for future test-strengthening pass.

---

## Items considered and dismissed (confidence < 70)

### TyCausal arms duplicate TyModal arms structurally — extraction opportunity?

**Confidence**: 30 — dismissed.

The duplication IS the established Phase-0 preemptive-parallel-arm pattern (per Stage 37/38/39/40). Extraction to a generic wrapper-helper would (a) break grep-by-stage audits (the explicit per-stage comments are load-bearing for the closure protocol), (b) couple all wrapper families to a single registry making cross-family stage changes higher-risk, and (c) trade specific debuggability for shared cleverness. The 5-stack wrapper-identity comment in `autodiff.py` already covers the "shared identity chain rule" case. The duplication is intentional and correct for Phase-0. Audit-spec methodology question 2 answers: "preserve the duplication; flag the misnaming separately (S41-CR-G1-005)".

### Helper arms parity scan: TyModal vs TyCausal

**Confidence**: 95 (no finding — parity complete).

Verified TyCausal coverage at every site TyModal appears:

| Helper | TyModal line | TyCausal line |
|--|--|--|
| `_resolve_type` | 1230 | 1241 |
| `_contains_unknown_type` (tuple) | 5359 | 5359 (same tuple) |
| `_refinement_proof_carried` | 5427 | 5432 |
| `_refinement_shape_exact` | 6101 | 6105 |
| `_erase_refinement` | 6158 | 6161 |
| `_contains_refinement` | 6282 | 6285 |
| `_is_refinement_container` (tuple) | 6309 | 6309 (same tuple) |
| `_contains_refined_function` | 6339 | 6342 |
| `_compatible` (and arm) | 7341 | 7346 |
| `_compatible` (or arm) | 7343 | 7348 |
| `_fmt` | 7520 | 7524 |

All 11 sites parallel. The audit-spec question 1e ("every TyCausal helper arm covered") returns: complete at the code level. (Test coverage of these arms is partial — `test_stage41_h1_*`, `test_stage41_h3_*`, `test_stage41_f2_*` exercise 3 of the 11 helpers; the other 8 are covered indirectly via the round-trip + lifecycle tests but not directly pinned. Below threshold.)

### Naming consistency — `_causal_intro` vs `_modal_intro` etc.

**Confidence**: 92 (no finding — fully consistent).

`_causal_intro`, `_causal_elim_kind`, `_causal_transitions`, `_causal_upgrade_hint` mirror `_modal_intro`, `_modal_elim_kind`, `_modal_transitions`, `_modal_upgrade_hint` byte-for-byte. No naming drift.

### Documentation drift in TyCausal docstring + dogfood

**Confidence**: 90 (no finding — both correct).

`TyCausal` docstring at lines 281-298 references Stage 41 Inc 2 correctly and the Stage 40 quartet correctly. Dogfood docstring lines 1-39 reference Stage 41 Inc 1 + Inc 2 + Inc 3 correctly. No stage-number or increment drift.

### Arm dispatch ordering — could causal dispatch fire before user-fn lookup?

**Confidence**: 88 (no finding).

The causal dispatch arm (lines 3744-3861) sits inside the `if isinstance(expr.callee, A.Name):` block after the H2 shadow-skip rewrite (line 2898) and after the modal/temporal/frame/tier arms. Key disjointness across `_causal_intro` / `_modal_intro` / `_temporal_intro` / etc. is verified — no name overlap. Ordering safe.

### `_BUILTIN_NAMES` completeness

**Confidence**: 95 (no finding).

All 11 causal names (`into_cause`, `into_effect`, `into_joint`, `into_independent`, `from_cause`, `from_effect`, `from_joint`, `from_independent`, `propagate`, `aggregate`, `isolate`) present at typecheck.py:2037-2041. Match the registered arms in the dispatch dicts.

### IR identity lowering completeness

**Confidence**: 92 (no finding).

`lower_ast.py:2023-2033` lists all 11 names in the identity tuple. The `test_stage41_ir_identity_lowering_all_11` test exercises all 8 intro/elim + 3 transitions and pins exit code (7 for intro/elim chains, 11 for transition chains). Identity preservation verified end-to-end.

### AD pure-call + chain-rule registration completeness

**Confidence**: 95 (no finding).

`AD_KNOWN_PURE_CALLS` and `_FRAME_IDENTITY_AD_NAMES` both list all 11 causal names (autodiff.py:116-122 + 213-218). Pinned by `test_stage41_ad_pure_registration` and `test_stage41_ad_identity_chain_rule_registration` at the import-set level.

### `run.py` DEMOS entry for "causal"

**Confidence**: 90 (no finding).

`run.py:104-109` adds the "causal" demo key pointing at `dogfood_14_causal_lifecycle.hx` with exit-42 expectation. Consistent with the "modal" entry shape at 96-101. The lack of a peer test in `test_reflection.py` is captured as S41-CR-G1-006.

---

## Convention-check summary

| Convention (Stage 37/38/39/40 playbook) | Stage 41 HEAD (`7448bf5`) status |
|---|---|
| Builtin arm shape matches Stage 37/38/39/40 verbatim | OK (modulo the missing `inner_is_shadowed` clause; see S41-CR-G1-001) |
| `_BUILTIN_NAMES` updated for all 11 new causal verbs | OK |
| Identity-lowering at IR (all 11 entries) | OK |
| `AD_KNOWN_PURE_CALLS` updated (11 entries) | OK |
| `_FRAME_IDENTITY_AD_NAMES` updated (11 entries) | OK functionally; **docstring drift now ~3.75x worse** (S41-CR-G1-005) |
| TyCausal in all 9 refinement/compat surfaces + `_resolve_type` + `_fmt` | OK |
| Dogfood `.hx` peer test in `test_reflection.py` | **VIOLATED — REPEAT of S40-CR-G2-004** (S41-CR-G1-006) |
| F1 launder guard mirrors Stage 40 gate-3 `inner_is_shadowed` clause | **VIOLATED** (S41-CR-G1-001) |
| 12-combo cross-modal matrix test in regression suite | **MISSING** (S41-CR-G1-002) |
| Safety-anchored hints for incoherent transition directions | **MISSING for 9 of 12 directions** (S41-CR-G1-003) |
| Dogfood witness inputs non-degenerate | **VIOLATED for cross-stack probe** (S41-CR-G1-004) |
| Diagnostic noise minimised on shadow path | **FAILED — H2 invariant broken via F1 cascade** (consequence of S41-CR-G1-001) |
| Gate-1 audit doc on disk for ledger continuity | OK (this doc) |

---

## 5-line summary

Stage 41 Inc 0-3 ships clean preemptive helper parity (all 11 TyModal sites have parallel TyCausal arms; `_BUILTIN_NAMES`, `AD_KNOWN_PURE_CALLS`, `_FRAME_IDENTITY_AD_NAMES`, IR identity lowering, `_resolve_type`, `_fmt` all complete for the 11 new builtins) and clean dispatch logic (no off-by-one in kind matching, no swapped src/dst transitions, verbatim style consistency with Stage 40 modal arms). **One HIGH gate-1 finding**: S41-CR-G1-001 (conf 95) — the F1 cross-causal launder guard at typecheck.py:3781-3786 is byte-shape-identical to the PRE-gate-3 Stage 40 modal guard; it lacks the `inner_is_shadowed` clause that Stage 40 gate-3 (commit `38f5598`) added to suppress F1 false-positives when the user shadows a `from_X` eliminator. Empirically reproducible: shadowing `from_cause` produces 3 errors (H2 shadow + spurious F1 launder + cascade arg-mismatch); the equivalent modal probe produces 1. This is the verbatim replay of the gate-3 H1 finding the Stage 40 closure trail explicitly documented. **Two MEDIUM gate-1 findings**: S41-CR-G1-002 (conf 88) — 11 of 12 cross-causal F1 launder directions are uncovered (only `into_cause(from_effect(...))` is tested; Stage 40's 12-combo matrix has no Stage 41 mirror); S41-CR-G1-003 (conf 80) — 9 of 12 cross-causal hint directions fall through to the generic "request a future-stage spec" fallback that Stage 40 gate-3 specifically called out as misleading for semantically-incoherent directions (no safety-anchored framing for reverse-causal moves). **Three LOWs**: dogfood cross-stack probe uses degenerate witness `1` (S41-CR-G1-004); `_FRAME_IDENTITY_AD_NAMES` docstring drift now ~3.75x wrong (S41-CR-G1-005, REPEAT of S40-CR-G2-003 + 11 more entries); no `test_dogfood_14_*` peer in `test_reflection.py` (S41-CR-G1-006, REPEAT of S40-CR-G2-004 + the Stage 40 missing one still missing). **One observation**: tests pin diagnostic content loosely (mostly only the "launders" or `"Cause"` word; not source/target labels). Gate-1 target was discovery; **the HIGH finding alone disqualifies a gate-1 CLEAN verdict**. Fix sweep: ~5 LoC for the `inner_is_shadowed` clause; ~30 LoC for matrix + hint-content tests; ~12 LoC for safety-anchored hints; 2-line dogfood witness change; ~30 LoC for backfilled peer tests; documentation drift deferred per the project's multi-stage rename ledger.
