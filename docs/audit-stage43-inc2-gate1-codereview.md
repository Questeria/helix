# Stage 43 Inc 2 Gate-1 — Code Review
Date: 2026-05-17
Scope: git diff 7699f00..e474c17 (Stage 43 Inc 1: deferred-items cleanup sweep)
HEAD: f843fb65b971f062df3b4696789d69456d46ee38

## Verdict
GATE CLEAN

No HIGH or MEDIUM findings at confidence >= 80. One LOW-confidence-80 finding
is recorded below as a "Stage 44 follow-up" pre-flight checklist item, not a
blocker for Stage 43 closure.

## Findings (HIGH / MEDIUM / LOW, with confidence 0-100)

### LOW-1 — Two stale-name imports remain in pre-Stage-43 tests; will break when the alias is dropped at Stage 44 (conf 80)

**File / line**:
- `helixc/tests/test_stage40_modal.py:480` — `from helixc.frontend.autodiff import _FRAME_IDENTITY_AD_NAMES`
- `helixc/tests/test_stage40_modal.py:484-485` — `assert name in _FRAME_IDENTITY_AD_NAMES, f"{name} must be in _FRAME_IDENTITY_AD_NAMES"`
- `helixc/tests/test_stage41_causal.py:290` — `from helixc.frontend.autodiff import _FRAME_IDENTITY_AD_NAMES`
- `helixc/tests/test_stage41_causal.py:296` — `assert name in _FRAME_IDENTITY_AD_NAMES, ...`

**Explanation**: Stage 43 Inc 1 renamed `_FRAME_IDENTITY_AD_NAMES` to
`_IDENTITY_AD_CHAIN_RULE_NAMES` and added a one-stage backwards-compat alias
(`autodiff.py:229`: `_FRAME_IDENTITY_AD_NAMES = _IDENTITY_AD_CHAIN_RULE_NAMES`).
The comment at `autodiff.py:226-228` explicitly states "Drop this alias at
Stage 44 or beyond". The new `test_stage43_cleanup.py` correctly imports the
new name everywhere. However, the two pre-existing tests
`test_stage40_frame_identity_ad_registration` (modal) and
`test_stage41_ad_identity_chain_rule_registration` (causal) still import the
OLD name through the alias. Today they pass because the alias is the same
frozenset object (verified by `test_stage43_item2_old_name_still_aliased`).
The day the alias is removed at Stage 44, these two tests will fail at import
time with `ImportError: cannot import name '_FRAME_IDENTITY_AD_NAMES'` — a
silent landmine for the future maintainer.

**Concrete fix suggestion** (do at Stage 44, not Stage 43):
Before deleting line 229 of `autodiff.py`, run:

```
grep -rn "_FRAME_IDENTITY_AD_NAMES" helixc/tests/
```

and migrate the 2 hits in `test_stage40_modal.py` + `test_stage41_causal.py`
to `_IDENTITY_AD_CHAIN_RULE_NAMES` (5 substitutions total — 2 imports + 3
in-string assertion messages). Also update the assertion-message string
literals so the diagnostic still names the correct symbol. Optionally pin
this checklist by adding a short docstring TODO comment immediately above
line 229: `# TODO(stage44): migrate test_stage40_modal.py:480 and
# test_stage41_causal.py:290 to the new name before deleting this alias.`

This finding is NOT a Stage 43 blocker — the alias works as documented and
328/0 regression pass per the silent-failure lane confirms no current
breakage. Recorded so the Stage 44 alias-drop ticket has the call-site
inventory in hand.

## Other code-review concerns examined (no finding)

- **15-test distribution + naming** (ship state at e474c17): tests cleanly
  partition as item2=3 / item3=6 / item4=6, which lines up with the audit
  brief's "5/5/5 roughly" expectation. Item 3 has 6 tests because the tier
  arm gets both zero-args and two-args coverage (`WorkingMem<>` and
  `WorkingMem<i32, i32>`) while the other 4 families get the representative
  two-args case only — defensible since the arity check is structurally
  identical across families and tier was the first arm written.
  Item 4 has 6 because the 5-family negative cases are paired with one
  positive sanity test (`test_stage43_item4_self_wrap_to_unwrap_still_allowed`)
  that pins `from_X(into_X(v))` continues to typecheck clean — exactly the
  bread-and-butter usage pattern the new M1 guard must not regress. All
  names follow `test_stage43_<item>_*` and are discoverable by pytest
  defaults. No `xfail`, `skip`, `pytest.skip`, or `skipif` markers anywhere
  in the file (grepped).

- **Test isolation**: every test parses its own source string and calls
  `typecheck(prog)` on a fresh AST. No module-level mutation, no shared
  fixtures, no class-based grouping with `setUp`. Order-independent.

- **Diagnostic substring matching**: the substrings the tests assert on
  (`"WorkingMem<T> takes 1 type argument"`, `"not idempotent"`, plus the
  callee-name) all come straight from the f-strings in `typecheck.py` and
  read as stable feature contracts, not brittle implementation details
  ("not idempotent" is the AGI-vocabulary anchor word, callee-name is the
  fix's whole point). Reasonably tight — not so loose they'd match
  accidental matches, not so tight they'd break on copyedit. Good balance.

- **Working tree at HEAD**: `git status` returns "nothing to commit, working
  tree clean". The other two lane audit docs are already committed in
  `f843fb6 Stage 43 closure trail: gate-1 audit docs`. This audit doc will
  be the only new untracked file post-write.

- **Dead code / leftover prints / no-owner TODOs in the +160 diff lines**:
  zero prints, zero TODO/FIXME/XXX/HACK markers across both `autodiff.py`
  and `typecheck.py` per `grep -n 'print\\(|TODO|FIXME|XXX|HACK'`. (The
  `print()` matches in `typecheck.py:7808-7809` are pre-existing CLI
  error-rendering, untouched by this stage.)

- **Public API surface**: all renamed/added symbols are `_`-prefixed
  (`_IDENTITY_AD_CHAIN_RULE_NAMES`, `_FRAME_IDENTITY_AD_NAMES`,
  `_resolve_type` arms, M1 intro arms inside `TypeChecker._check_call`).
  No public-API break. The autodiff helpers are imported with leading
  underscore by `autodiff_reverse.py`, which is in-tree, so the rename's
  reach is fully observed by Stage 43 grep.

- **One-stage alias drop-deadline comment in autodiff.py**: PRESENT.
  `autodiff.py:187-191` says "The old name remains as a backwards-compat
  alias for one stage (Stage 44 will drop the alias)." and `autodiff.py:226-228`
  says "Drop this alias at Stage 44 or beyond." A future maintainer reading
  only `autodiff.py` (not the progress doc) gets the message. Concern #7
  from the audit brief is satisfied. (The orphan-imports finding above,
  LOW-1, is a separate, complementary concern about the migration checklist
  for that drop, not about whether the drop intent is documented.)

- **Test inheritance from prior gates**: `test_stage43_cleanup.py` imports
  only public-ish helpers (`parse`, `typecheck`) and the renamed private
  set under test. Zero imports of test-internals from `test_stage4{0,1,2}_*.py`
  or any other test module. Self-contained.

## Verification steps performed

- Read `helixc/tests/test_stage43_cleanup.py` in full (445 lines at HEAD,
  202 lines at ship-state e474c17). Enumerated all `def test_*` defs at
  e474c17 (15 tests) and at HEAD (24 tests; the extra 9 are the gate-1
  and gate-2 MEDIUM-fix backfill tests that landed AFTER the audit scope,
  so they're out-of-scope for this lane but confirm the audit-fix
  discipline is healthy).
- Read the full diff `git diff 7699f00 e474c17` for autodiff.py (18 +/-),
  autodiff_reverse.py (4 +/-), typecheck.py (138 +) — 343 insertions /
  19 deletions matched the stat header.
- Grepped tree-wide for `_FRAME_IDENTITY_AD_NAMES` and
  `_IDENTITY_AD_CHAIN_RULE_NAMES` — 2 stale import sites in
  `test_stage40_modal.py` + `test_stage41_causal.py` are the basis of
  LOW-1. All non-test source matches at HEAD use the new name; the alias
  line at `autodiff.py:229` is the only producer of the old name.
- Verified drop-deadline comment is present at `autodiff.py:187-191` and
  `autodiff.py:226-228` (concern #7).
- Ran `git status` → "nothing to commit, working tree clean" (concern #4).
- Grepped `xfail|skip|pytest\\.skip|skipif` in `test_stage43_cleanup.py`
  → no matches (concern #1 sub-bullet).
- Grepped `print\\(|TODO|FIXME|XXX|HACK` in `autodiff.py` + `typecheck.py`
  → only pre-existing CLI prints, no new findings (concern #5).
- Confirmed all `_`-prefixed symbol exports (concern #6).
- Confirmed `test_stage43_cleanup.py` imports only `parse`, `typecheck`,
  `_IDENTITY_AD_CHAIN_RULE_NAMES`, `_FRAME_IDENTITY_AD_NAMES` — no
  cross-test-module imports (concern #8).
- Cross-checked the other two lane audit docs
  (`audit-stage43-inc2-gate1-silent-failures.md` + `-type-design.md`) — both
  pre-existing on disk (committed in f843fb6), no overlap with LOW-1
  finding (silent-failure lane verified the alias is same-object;
  type-design lane scored the new name as semantically accurate). LOW-1 is
  a genuinely third-lane-only finding (call-site migration prep), not a
  rehash of either earlier lane's observations.
- Checked `/tmp/pytest_full.log`: still in flight per orchestrator (13
  lines logged, progress at `test_cli.py` collection phase, no
  `passed|failed` summary line yet). Per orchestrator instructions, did
  not block on the run; the silent-failure lane independently confirmed
  328/0 on a wider regression sweep and the orchestrator owns the
  full-suite verification.
