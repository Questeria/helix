# Stage 43 Inc 2 Gate-1 — Silent-Failure Audit
Date: 2026-05-17
Scope: git diff 7699f00..e474c17 (Stage 43 Inc 1: deferred-items cleanup sweep)
HEAD: d4a1e8b0da74b6c81f37ed61e8d08bc88d58616b

## Verdict
GATE CLEAN

No HIGH, MEDIUM, or LOW silent-failure findings at confidence >= 70 in the Stage 43 Inc 1
ship surface. The three deferred items (LOW-3 rename, F5 arity arms, M1 double-wrap
rejection) each closed their stated silent-failure window without opening a new one. The
test suite is adequate and the self-host cascade reproduces byte-identical G2..G4 with the
expected sha `a6f1ee44eb4418ba296954528d05564f5a37627dc38bb350b2308675d86b8986`.

## Findings (HIGH / MEDIUM / LOW, with confidence 0-100)

No findings at confidence >= 70.

### Sub-threshold observations (informational, not findings)

- **OBS-1 (conf 55, NOT a finding)**: Two pre-Stage-43 tests
  (`helixc/tests/test_stage40_modal.py:480-485` and
  `helixc/tests/test_stage41_causal.py:290-296`) still import the deprecated
  `_FRAME_IDENTITY_AD_NAMES` name. This is *intentional* — Stage 43's plan
  explicitly keeps the alias for one stage (drop at Stage 44+). The alias is a
  direct rebind to the same `frozenset` object (autodiff.py:229:
  `_FRAME_IDENTITY_AD_NAMES = _IDENTITY_AD_CHAIN_RULE_NAMES`), so identity is
  preserved (`x is y` returns True) and the membership tests in those legacy
  callers cannot diverge. No silent-failure risk; flagged only as a Stage 44
  cleanup pointer.

- **OBS-2 (conf 50, NOT a finding)**: The tier intro arm at typecheck.py:3368
  retains the original `bn in _tier_intro_elim and len(arg_tys) == 1` compound
  guard, while the four other family intro arms (frame at 3419, temporal at
  3513, modal at 3604, causal at 3900) follow the F1 "name first, arity
  second" pattern with an explicit "takes 1 argument, got N" diagnostic. This
  is a pre-existing inconsistency carried over from Stage 38's F1 fix that
  only retrofitted the frame arm (Stages 39/40/41 mirrored the F1 pattern;
  Stage 37's tier arm was never retrofitted). Stage 43 was scoped to F5
  (type-position arity) and M1 (double-wrap), not the call-position arity
  symmetry. `into_working(7, 8)` therefore falls through to the call catchall.
  Not introduced by Stage 43; out of scope. Worth flagging as a candidate for
  Stage 44+ deferred-cleanup.

## Verification steps performed

1. **Scope confirmed**: `git diff --stat 7699f00..e474c17` shows exactly the four
   advertised files (autodiff.py +18, autodiff_reverse.py +4, typecheck.py +138,
   test_stage43_cleanup.py NEW 202 lines, totaling +343/-19). No drift from the
   stated ship.

2. **Item 2 (rename) — backwards-compat alias is by reference, not copy**:
   - autodiff.py:229 contains `_FRAME_IDENTITY_AD_NAMES = _IDENTITY_AD_CHAIN_RULE_NAMES`.
     Direct rebind of the `frozenset` object. Both names point at the same
     object identity. (`frozenset` is immutable anyway, so any "mutating one
     mutates both" question is moot — but the identity equality
     `_FRAME_IDENTITY_AD_NAMES is _IDENTITY_AD_CHAIN_RULE_NAMES` is asserted by
     `test_stage43_item2_old_name_still_aliased`, which passed.)
   - Grep across `C:/Projects/Kovostov-Native/helixc/` for both names confirms
     all in-code use sites (autodiff.py:1337, autodiff_reverse.py:53 + 684) use
     the new name. Two pre-Stage-43 test files still import the old name — OK
     per the one-stage alias policy.
   - No tests, IR lowering, or codegen sites reference the old name.

3. **Item 3 (F5 arity) — all 19 wrapper-type names diagnose arity**:
   - I ran a parametric probe (`f(x: <Name><i32, i32>)`) over all 19 names
     across 5 families: WorkingMem, EpisodicMem, SemanticMem, ProceduralMem;
     WorldFrame, RobotFrame, CameraFrame; Past, Present, Future, Eternal;
     Known, Believed, Goal, Uncertain; Cause, Effect, Joint, Independent.
   - All 19 emit `"<Name><T> takes 1 type argument, got 2"`. No fall-through
     to the misleading "unknown type 'X'" message.
   - Edge cases: `Past<>` → "got 0"; `Past<i32, i32, i32>` → "got 3". Count is
     accurate (not off-by-one). The arity arm fires for any `len(ty.args) != 1`,
     not just `> 1`.
   - The arms still return `TyUnknown(hint=ty.base)` so downstream code does
     not segfault on the malformed type.

4. **Item 4 (M1 double-wrap) — all 19 intro builtins reject same-family**:
   - Parametric probe over all 19 intros (`into_working`/`into_episodic`/.../
     `into_independent`): each correctly rejects when the argument is the
     same-family wrapper type. All 19 emit `"<name>() received an
     already-wrapped <Family><i32>; intro builtins are not idempotent — ..."`.
   - **Cross-family acceptance**: `into_past(into_known(7))` typechecks clean
     to `Past<Known<i32>>` (verified, errors list empty). The M1 guard tests
     `isinstance(arg_tys[0], TyTemporal)`, not generic wrappers, so cross-
     family composition is preserved.
   - **Triple-nest same family**: `into_past(into_past(into_past(...)))` is
     caught at the first inner same-family wrap (correct behavior — the
     innermost `into_past` typechecks but the second one detects the
     `TyTemporal` and rejects). The diagnostic correctly points at the second
     `into_past`, not the third.
   - **Hint routing**: frame hint suggests `world_to_robot`/`world_to_camera`,
     temporal hint suggests `to_past`/`forecast`/`recall_past`/`actualize`,
     modal hint suggests `confirm`/`act_on`, causal hint suggests `propagate`/
     `aggregate`/`isolate`, tier hint says "unwrap first" (no canonical cross-
     tier rename builtin exists, so the hint correctly avoids suggesting one).
     All four family hints route family-appropriately.

5. **test_stage43_cleanup.py adequacy**:
   - 15 tests collected, 15 passed, 0 skipped, 0 xfailed.
   - Item 2: 3 tests (export, alias-identity, autodiff_reverse import).
   - Item 3: 6 tests — tier-0-args + tier-2-args + frame-2-args + temporal-
     2-args + modal-2-args + causal-2-args. **Mild gap (sub-threshold)**: no
     test pins the "got N" count for N != 2 except the tier-0-args case, but
     my parametric probe confirmed N is reported correctly for 0/2/3. Not
     promoted to finding.
   - Item 4: 6 tests — one per family + a positive sanity test
     (`from_known(into_known(42))` typechecks clean). The positive test is
     load-bearing: it confirms the guard does NOT regress single-wrap-then-
     unwrap, which is the bread-and-butter usage. Cross-family acceptance is
     not pinned by an explicit test, but the positive test plus the type
     `Past<Known<i32>>` round-tripping in my probe gives sufficient coverage.
   - All diagnostic assertions use substring matches (`"not idempotent" in
     str(e)` and `"<name>" in str(e)`) — specific enough to fail on regression
     (a missing-handler regression would emit a different message), stable
     enough not to false-flag on hint-text wordsmithing.

6. **Self-host gate**:
   - `python scripts/stage33_selfhost_gate.py` PASS.
   - G2 = G3 = G4 byte-identical, sha =
     `a6f1ee44eb4418ba296954528d05564f5a37627dc38bb350b2308675d86b8986`. This
     matches the expected sha `a6f1ee44eb44...` from the gate spec. The Stage
     43 typecheck.py changes are diagnostic-only (no change to the
     successful-typecheck path), so the self-hosted compilation pipeline is
     unaffected, as predicted.

7. **Regression sweep across the wrapper-family stages**:
   - `pytest test_stage36_provenance.py test_stage37_memory.py
     test_stage38_frames.py test_stage39_temporal.py test_stage40_modal.py
     test_stage41_causal.py test_stage43_cleanup.py` → 328 passed in 1577.64s
     (0:26:17), 0 failed, 0 skipped.

8. **Out-of-scope items respected**:
   - The `gate-2-audit-stash` (stash@{0}) was not popped, dropped, or applied.
   - Item 1 (aggregate → composite diagnostic rename) remains deferred to
     Stage 44+ per the progress note; nothing in the Stage 43 ship touches
     the `aggregate` diagnostic surface.
