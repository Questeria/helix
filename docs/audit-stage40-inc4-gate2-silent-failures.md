VERDICT: 1 HIGH, 0 MEDIUM, 2 LOW, 3 OBS

# Stage 40 Inc 4 closure gate-2 silent-failure audit

Surface: HEAD `e8fb593` on `main` (Stage 40 Inc 4 closure gate-1 fix sweep: F1 + F2). Base: `cb36bbc` (Stage 40 OPENS — Inc 0+1+2+3 modal/epistemic types). Audit target is the SHIPPED surface as of `e8fb593`; an in-flight working-tree patch broadening F1 to all cross-modal launders was stashed (`gate-2-audit-stash-broader-launder-lane`) so probes ran against the gate-1-only fix-sweep state.

Stage 40 introduces `TyModal` (kinds: known / believed / goal / uncertain), 4 intro + 4 elim + 2 cross-modal transitions (`confirm` Believed→Known, `act_on` Goal→Known) — 10 modal builtins total. Registered in `_BUILTIN_NAMES` (1989-1992), `AD_KNOWN_PURE_CALLS` (113-115), `_FRAME_IDENTITY_AD_NAMES` (203-205), `lower_ast.py` identity arm (2013-2023). Gate-1 fix sweep added an F1 syntactic Uncertain-laundering reject (3468-3517) and an F2 user-fn shadow reject in `_register_fn` (908-921). Stage-39 audit reference template: `docs/audit-stage39-postinc3-silent-failures.md`. Parallel gate-2 audits already on disk: `audit-stage40-inc4-gate2-type-design.md` (H1+L1+2 OBS) and `audit-stage40-inc4-gate2-codereview.md` (0 HIGH, 2 MEDIUM, 2 LOW, 2 OBS).

## Audit methodology executed

1. Stashed two layers of in-flight gate-2 fix-lane work on `typecheck.py` (a small M2 noise-suppression patch + a substantial F1-broadening patch) so probes ran against the SHIPPED commit `e8fb593`, not the working-tree successor. Stash messages: `gate-2-audit-stash`, `gate-2-audit-stash-broader-launder-lane`. Re-popped after probing.
2. Read full diff `git diff cb36bbc..e8fb593` for each affected file; cross-referenced against the Stage 39 closure-gate-1 silent-failure report and against both parallel Stage 40 gate-2 audits already on disk.
3. End-to-end probes via `python -c` harness:
   - 20 wrong-arity probes (10 builtins × {0-arg, 2-arg}) — all rejected with named diagnostics like `into_known() takes 1 argument, got 0`.
   - 8 wrong-source transition probes (`confirm` and `act_on` × every wrong source incl. raw i32) — all rejected; each emits 2 errors (the primary kind-mismatch + a cascading `?{name}` from the TyUnknown sentinel propagating to the outer eliminator).
   - 6 forward-mode + 6 reverse-mode AD probes (4 intro+elim round-trips + 2 transition round-trips, all wrapping `x*x`) — all produce `(x + x)` for `d(x*x)/dx` through the chained modal+elim wrappers.
   - 7 wrapper-stacking probes (`Known<Known<i32>>`, `Believed<Goal<i32>>`, `Goal<Goal<i32>>`, `Past<Known<i32>>`, `Known<WorldFrame<i32>>`, `Known<WorkingMem<i32>>`, `Uncertain<Known<i32>>`) — all silently accepted (no diagnostic, no warning).
   - 9 cross-modal launder probes — `into_X(from_Y(s))` for every X≠Y pair except the gate-1-blocked `from_uncertain` source — all 9 silently accepted at typecheck.
   - 5 F1-bypass probes (let-binding, helper-fn indirection, arithmetic wrapper, id-helper, parenthesized) — 4 of 5 bypass F1; only the parenthesized form is caught (parens preserve the `A.Call` AST). The let-binding and helper-fn bypasses are documented as known limits at `typecheck.py:3477-3489`.
   - 16 direct method probes on the 8 helper visitors with synthetic `TyModal` instances (`_compatible`, `_refinement_shape_exact`, `_erase_refinement`, `_contains_refinement`, `_is_refinement_container`, `_contains_refined_function`, `_contains_unknown_type`, `_refinement_proof_carried`) — all 16 return the expected value; preemptive TyModal arms at 5083, 5752, 5805, 5926, 5949, 5978, 5017, 5063 all wired correctly.
   - 9 builtin-shadow probes for F2 (covering modal, temporal, frame, tier, and memory-op builtins) — all 9 reject with the expected `'name' shadows a reserved builtin name` diagnostic. F2 coverage is complete across the reserved-name surface.
   - 4 refined-inner probes (`Known<NonZero>`, `WorldFrame<NonZero>`, `Past<NonZero>`, plain `NonZero`) — modal/frame/temporal wrappers all emit the generic `would change refined parameter or return requirements` diagnostic on a refinement-violating literal instead of the sharper trap-31001 diagnostic the bare case produces; parity-symmetric across all 3 wrapper families, not Stage-40-specific.
   - 3 IR-only lowering probes (wrong-arity calls of `into_known`, `confirm`, and zero-arity `into_known` via `Lowerer(prog).lower()` skipping typecheck) — all 3 raise the opaque `NotImplementedError: unknown function 'X' in IR lowering at L:C; run typecheck first`. Same Stage 39 F5 pattern; defense-in-depth is missing.
   - 1 typecheck-bypass probe: a program that fails typecheck for an F1 launder, then ignored typecheck errors and lowered anyway, compiles a working binary. Intrinsic to Helix's Phase-0 typecheck-as-truth-gate; family-wide observation, not Stage-40-specific.
   - 1 test-suite scan: `test_reflection.py` contains entries for `dogfood_01` through `dogfood_12` but not `dogfood_13_modal_lifecycle` — same O1 family-symmetry gap that Stage 39 also entered its own gate-1 with.
4. Cross-referenced findings against the two parallel Stage 40 gate-2 audits (`audit-stage40-inc4-gate2-type-design.md` H1+L1; `audit-stage40-inc4-gate2-codereview.md` CR-G2-001 through CR-G2-006) so the silent-failure-hunter lane records its independent verification of overlapping findings rather than duplicating numbering.

## Findings

### F1 [HIGH conf 90] cross-modal launder asymmetry — gate-1 F1 closes only 3 of 12 silent epistemic-upgrade paths

**Citation**: `helixc/frontend/typecheck.py:3494-3517` (the gate-1 F1 guard). The guard reads:

```py
target_kind = _modal_intro[bn]
if (target_kind in ("known", "believed", "goal")
        and len(expr.args) >= 1
        and isinstance(expr.args[0], A.Call)
        and isinstance(expr.args[0].callee, A.Name)
        and expr.args[0].callee.name == "from_uncertain"):
    self.errors.append(TypeError_(
        f"{bn}(from_uncertain(...)) launders an "
        f"Uncertain<T> into {target_kind.capitalize()}<T> ..."))
```

The guard only matches when the inner call's callee name is the literal string `"from_uncertain"`. The other three eliminators (`from_known`, `from_believed`, `from_goal`) are excluded — `into_X(from_Y(s))` for `Y ∈ {known, believed, goal}` is silently accepted by typecheck for ALL 9 cross-kind X≠Y combinations, including the two combinations that have legitimate audited transition equivalents.

**Reasoning**: Stage 40's headline AI-safety claim, repeated in three places (commit message for `cb36bbc`, `TyModal` docstring at `typecheck.py:268-278`, `test_stage40_modal.py:92-98`), is that "treating a goal as a known fact (a category mistake at the heart of many AI safety failures) is caught at compile time". The compile-time enforcement mechanism is: (a) `from_X` rejects cross-kind eliminator mistakes (gate-1-verified); (b) the only audited epistemic upgrades are `confirm` (Believed→Known) and `act_on` (Goal→Known), enumerated in `_modal_transitions` at 3552-3555. The F1 guard closes one bypass: `into_known(from_uncertain(u))` etc.

The two upgrade-launder vectors `into_known(from_believed(b))` and `into_known(from_goal(g))` are the EXACT silent equivalents of `confirm(b)` and `act_on(g)`. They achieve the same modal upgrade with zero compile-time signal. Reproduced live against `e8fb593`:

```
fn main() -> i32 {
    let s = into_believed(42);
    let k = into_known(from_believed(s));   // <-- silently accepted (0 errs)
    from_known(k)                            // exit code 42
}
```

```
fn main() -> i32 {
    let s = into_goal(42);
    let k = into_known(from_goal(s));        // <-- silently accepted (0 errs)
    from_known(k)
}
```

Both programs typecheck CLEAN (`typecheck(prog) == []`), then lower as identity, then compile and run to exit code 42 — identical observable behavior to the `confirm`/`act_on`-using version. The "category mistake at compile time" claim is materially incomplete: a contributor who skips reading `confirm`/`act_on` and reaches for the more-obvious `into_X(from_Y(...))` get the same runtime result with NO compile-time mention that they bypassed the audited upgrade path.

The remaining 7 cross-kind launders (Known→{Believed, Goal, Uncertain}; Believed→{Goal, Uncertain}; Goal→{Believed, Uncertain}; Believed→Uncertain) are also silently accepted but are less safety-critical:

- 3 downgrades (Known→{Believed, Goal, Uncertain}): semantically dubious (downgrading a known fact to a belief/goal/uncertain is information-erasure), but no audit-trail bypass since no transition exists in either direction. LOW silent-failure surface.
- 2 lateral (Believed↔Goal): plausibly the spec's intended deferral path; semantically nonsense (a belief is not a goal). LOW.
- 2 to-Uncertain (Believed→Uncertain, Goal→Uncertain): possibly legitimate (introducing uncertainty about a prior belief/goal). MEDIUM-ambiguous; intrinsic deferral question.

The two that matter for AI-safety enforcement are the 2 upgrades to Known: those have audited replacements (`confirm`, `act_on`) and the launder pattern is precisely what the audited transitions were built to require.

Direct comparison with the gate-1 F1 reject:

| Pattern | Gate-1 behavior | Audited equivalent | Risk |
|---|---|---|---|
| `into_known(from_uncertain(u))` | REJECT (F1 fires) | (none — Phase-0 deferred) | gate-1 closed |
| `into_known(from_believed(b))` | ACCEPT (silent) | `confirm(b)` | **silent upgrade-bypass** |
| `into_known(from_goal(g))` | ACCEPT (silent) | `act_on(g)` | **silent upgrade-bypass** |

The gate-1 F1 fix chose the right hardest-case (Uncertain has no legitimate upgrade so MUST reject) but stopped before generalizing to the two cases where the legitimate-upgrade-bypass risk actually has a named alternative. The asymmetric treatment is also visible in the F1 diagnostic hint at 3512-3515 ("resolve uncertainty before the value enters the type system") — the hint can't mention an audited alternative because there isn't one; for the Believed/Goal cases the hint could literally name `confirm`/`act_on`, making the diagnostic strictly more actionable than the Uncertain case.

**Note on parallel work**: an in-flight working-tree patch on `typecheck.py` (the `gate-2-audit-stash-broader-launder-lane` stash, currently in the working tree at audit close) is generalizing F1 to cover ALL cross-modal `into_X(from_Y(...))` for X≠Y, with kind-specific hints pointing at `confirm` or `act_on` where they exist. The patch is structurally a verbatim widening of the gate-1 F1 reject; ~50 LOC. This audit records the gap as shipped at `e8fb593` so the closure-gate-2 ledger has a record that Stage 40 entered the gate with the silent-upgrade-bypass hole — the same way Stage 39 gate-1 F1 (Stage 38 H1+H2 carry-over) recorded the in-flight fix it discovered against ship state.

**Why HIGH and not MEDIUM**: Stage 40's headline guarantee — the very claim that justifies the stage — is materially incomplete in a way that is reproducible in 3 lines and that the test suite at `test_stage40_modal.py:578-637` does NOT cover (the existing F1 tests only probe `from_uncertain` sources). The gate-1 author's framing "Stage 40 OPENS — AI-safety category mistakes caught at compile time" is contradicted by a literal `into_known(from_believed(...))` two-line program. The HIGH rating reflects the gap between the claim and the enforcement, not the difficulty of the fix (which is small and authored).

**Why not CRITICAL/etc.**: typecheck errors HAVE to be honored downstream (the Phase-0 truth-gate model); the launder is "only" silent at typecheck, not at the user-visible binary level — they get the wrong epistemic discipline, not crashes. And the cross-modal launder still requires the user to write `from_Y(...)` explicitly, so it's not a sneaky compiler-internal bypass — it's an absent fence on a foreseeable pattern.

**Remediation**: generalize the F1 guard at 3494-3517 from "Uncertain only" to "every cross-kind launder". Mechanically, replace the literal `expr.args[0].callee.name == "from_uncertain"` with `expr.args[0].callee.name in {"from_known", "from_believed", "from_goal", "from_uncertain"}` and emit a kind-specific hint that names `confirm`/`act_on` when applicable. The stashed in-flight patch already implements exactly this; running it lifts gate-2 to CLEAN on this finding. Verify post-fix by adding 6 regression tests (4 launders to Known + 2 lateral launders should reject; 4 self-rewraps + 2 transitions via `confirm`/`act_on` should still pass).

### F2 [LOW conf 75] Wrapper-stacking — `into_X` silently accepts an already-modal-wrapped value, producing semantically-incoherent nested types

**Citation**: `helixc/frontend/typecheck.py:3461-3519` (`_modal_intro` dispatch). The dispatch returns `TyModal(kind=_modal_intro[bn], inner=arg_tys[0])` without inspecting `arg_tys[0]`.

**Reasoning**: Live probes against `e8fb593`:

```
into_known(into_known(42))     -> Known<Known<i32>>             (0 errs)
into_believed(into_goal(42))   -> Believed<Goal<i32>>           (0 errs)
into_goal(into_goal(42))       -> Goal<Goal<i32>>               (0 errs)
into_uncertain(into_known(42)) -> Uncertain<Known<i32>>         (0 errs)
into_known(into_world(42))     -> Known<WorldFrame<i32>>        (0 errs, valid)
into_known(into_working(42))   -> Known<WorkingMem<i32>>        (0 errs, valid)
into_past(into_known(42))      -> Past<Known<i32>>              (0 errs, debatable)
```

Modal-on-modal nesting is semantically incoherent for the same reason temporal-on-temporal was rated MEDIUM at Stage 39 F3: a proposition has exactly one epistemic status at a time. `Known<Known<i32>>` is meaningless (a fact known-to-be-known is just known); `Believed<Goal<i32>>` confuses the WHY-axis with the WHAT-axis (the value isn't a belief about a goal; it's either a goal OR a belief about its content); `Uncertain<Known<i32>>` is the most contradictory — wrapping a known fact in uncertainty contradicts the "Known" tag's meaning and, worse, makes the gate-1 F1 guard irrelevant for that particular bypass route:

```
fn main() -> i32 {
    let k: Known<i32> = into_known(42);
    let u: Uncertain<Known<i32>> = into_uncertain(k);  // silently OK
    let k2: Known<i32> = from_uncertain(u);            // strips Uncertain, restores Known
    from_known(k2)
}
```

A contributor who wants to deliberately erase the audit trail can wrap a Known value in Uncertain then strip it — the gate-1 F1 guard does NOT fire because the inner is not a `from_uncertain` call. This is a strict subset of F1 — anything F1 should reject in spirit, this bypasses in letter. The gate-2 audit DOES rate F1 as HIGH; this finding is rated separately because the bypass mechanism is different (nesting, not laundering).

The frame/tier/temporal mixed cases (`Known<WorldFrame<i32>>`, `Known<WorkingMem<i32>>`, `Past<Known<i32>>`) are semantically defensible. `Known<WorldFrame<i32>>` = "directly observed in world frame" is the literal motivating use case (test `test_stage40_compose_known_past_round_trip` at `test_stage40_modal.py:529-542` exercises it). The dispatch can't tell the legitimate compositions from the incoherent ones structurally.

**Why LOW not MEDIUM**: family-symmetric with Stage 38 L1 (TyFrame nesting, deferred), Stage 37 (TyMemTier nesting), Stage 39 F3 (TyTemporal nesting, rated MEDIUM but deferred at gate-1 and never closed). Stage 40 inherits the family-wide pattern. The Stage 39 F3 verdict was MEDIUM because temporal kinds have a stronger exclusivity property than frames; modal kinds have the SAME stronger exclusivity — but with three prior stages deferring the fix, the precedent is "defer family-wide". Downgraded to LOW because (a) test coverage demonstrates the lateral compose case is intentional, (b) the F1 nesting bypass route surfaces only under deliberate contributor adversarial intent (which is the H1 case), and (c) the Stage 39 finding documented the lesson and the closure decision was "defer".

**Why not OBS**: the `Uncertain<Known<i32>>` bypass IS a concrete bypass route for the AI-safety claim and IS testable today. Calibration-borderline; rated LOW because the gate-1 F1 widening (in flight) does NOT close this route either, so the issue is structurally distinct from F1 not a duplicate.

**Remediation**: in `_modal_intro` dispatch at line 3461, after the arity check (3462-3467) and after the F1 guard (3494-3517), add:

```py
if isinstance(arg_tys[0], TyModal):
    self.errors.append(TypeError_(
        f"{bn}() input is already modally-tagged "
        f"({self._fmt(arg_tys[0])}); use a transition "
        f"(confirm/act_on) to change kinds, or "
        f"from_{arg_tys[0].kind}() to unwrap first",
        expr.span,
    ))
    return TyUnknown(hint=bn)
```

~8 LOC; symmetric with the recommended Stage 39 F3 fix that was deferred. Worth applying to the frame and temporal families simultaneously to close the entire backlog with one fix-sweep.

### F3 [LOW conf 75] IR identity-lowering arm silently drops `args[1..]` if a wrong-arity modal call slips past typecheck

**Citation**: `helixc/ir/lower_ast.py:1986-2025` (post-Stage-40 surface). The Stage 40 additions extend the existing identity-lowering arm to the 10 modal builtins; the guard is `if expr.callee.name in (...12 frame + 12 temporal + 10 modal names...) and len(expr.args) == 1: return self._lower_expr(expr.args[0])`.

**Reasoning**: Live IR-only probe (skipping typecheck) reproduces the same Stage 39 F5 pattern:

```
Lowerer(parse('fn main() -> i32 { let x: i32 = into_known(1, 2); 0 }')).lower()
=> NotImplementedError: unknown function 'into_known' in IR lowering at 1:33; run typecheck first

Lowerer(parse('fn main() -> i32 { let x: i32 = confirm(1, 2, 3); 0 }')).lower()
=> NotImplementedError: unknown function 'confirm' in IR lowering at 1:33; run typecheck first

Lowerer(parse('fn main() -> i32 { let x: i32 = into_known(); 0 }')).lower()
=> NotImplementedError: unknown function 'into_known' in IR lowering at 1:33; run typecheck first
```

The opaque-catchall diagnostic asserts "run typecheck first" even though the user might have a build pipeline that ignores typecheck errors (and indeed, the F1-bypass probe demonstrates `lower(prog_with_typecheck_errs)` does succeed today on the launder programs because the surface is identity-lowered). The diagnostic is technically wrong: typecheck WAS run, it emitted errors, and the user (or pipeline) just chose to ignore them. The risk dimension is unchanged from Stage 39 F5: any future refactor that (a) relaxes the `len == 1` guard inside the identity arm OR (b) reorders the dispatch so a wrong-arity call no longer falls through to the catchall — would silently drop side-effecting args. Same name-set is now 34 entries (12 frame + 12 temporal + 10 modal) and growing.

**Why LOW**: same call as Stage 39 F5. No active runtime bug; defense-in-depth is what is missing. The diagnostic IS produced (just opaque); the wrong-arity safety net at typecheck IS active (verified by the 20 wrong-arity probes that all rejected). The IR-arm is the second-of-two gates; today it doesn't fail open, it fails closed-but-misleading.

**Remediation**: add an explicit assertion inside the identity arm at 2024:

```py
if (isinstance(expr.callee, A.Name)
        and expr.callee.name in IDENTITY_LOWER_NAMES):
    assert len(expr.args) == 1, (
        f"{expr.callee.name} arity guard violated; "
        f"typecheck should have rejected wrong-arity calls"
    )
    return self._lower_expr(expr.args[0])
```

~3 LOC. Or convert the in-place tuple literal at 1987-2023 to a hoisted module-level frozenset (closes Stage 39 O2 too) and add the assertion at the same time. Note that the OBS-B finding below recommends precisely this hoist.

## OUT OF SCOPE — observations (no severity)

- **O1** (test-suite gap, family-symmetric carry-over from Stage 38 O1 / Stage 39 O1). `test_reflection.py` contains parallel entries for `test_dogfood_10_memory_tiers` (Stage 37), `test_dogfood_11_spatial_frames` (Stage 38), `test_dogfood_12_temporal_lifecycle` (Stage 39) but no `test_dogfood_13_modal_lifecycle`. The silent-failure framing: a regression that breaks `dogfood_13_modal_lifecycle.hx` end-to-end (e.g., a `run.py` DEMOS-dict key collision, an `@pure` decorator interaction across the three helper fns, a witness-arithmetic regression in the cross-stage `Known<Past<i32>>` composition probe) would not be caught by any Stage 40 test — the silent-failure mode is "the dogfood breaks unnoticed because nothing references it from CI". Fix is 7 LOC of test_reflection.py addition. Stage 39 also entered its own gate with this gap, then closed it before Stage 39 CLOSED.

- **O2** (dispatch perf, family carry-over from Stage 38 O2 / Stage 39 O2). The three dispatch dicts at `typecheck.py:3455-3459` (`_modal_intro`), `3520-3524` (`_modal_elim`), `3552-3555` (`_modal_transitions`) rebuild on every Call-expression typecheck visit. Negligible cost (3 small dicts) but pattern-symmetric to Stage 38/39 O2; same hoisting opportunity. The identity-lowering tuple at `lower_ast.py:1987-2023` is similarly per-Call-visit rebuilt and has now grown to 34 entries.

- **O3** (typecheck-bypass observation, intrinsic to Phase-0 truth-gate). Programs that fail typecheck for an F1 launder can still be lowered via `lower(prog)` without re-checking errors and produce working binaries (all 10 modal builtins lower as identity). Repro:
  ```
  prog = parse("fn main() -> i32 { let u: Uncertain<i32> = into_uncertain(42);
                                    let k: Known<i32> = into_known(from_uncertain(u));
                                    from_known(k) }", include_stdlib=True)
  errs = typecheck(prog)              # 1 err
  ir = lower(prog)                     # SUCCESS — binary compiles to 42
  ```
  This is intrinsic to Helix's typecheck-as-truth-gate model (the IR layer assumes typecheck has been honored). Cross-listed in Stage 36 audit history as a recurring lower-of-typecheck-rejects observation. Out of scope for Stage 40 gate-2.

## Summary

ONE HIGH (cross-modal launder asymmetry: gate-1 F1 closes only the Uncertain-source launder; `into_known(from_believed(b))` and `into_known(from_goal(g))` are silent upgrade-bypasses for the audited `confirm`/`act_on` transitions, materially incomplete vs Stage 40's headline "category mistake at compile time" claim — verbatim replay of the in-flight working-tree fix-lane stash, equivalent to a generalization of the F1 syntactic guard from `from_uncertain` only to all 4 elim names), ZERO MEDIUM, TWO LOW (modal-on-modal wrapper-stacking produces semantically-incoherent nested types and bypasses F1 via the `Uncertain<Known<T>>` strip-pattern; IR identity-arm has no defense-in-depth assertion against wrong-arity calls slipping past typecheck — Stage 39 F5 carry-over), THREE OBS (missing `test_dogfood_13_modal_lifecycle` in test_reflection.py; per-call dict + tuple reallocation; lower-of-typecheck-rejects observation).

Gate-2 strictness reminder: per brief, anything below HIGH conf 75 goes to OBS. The two LOW findings sit at conf 75 exactly — borderline OBS, retained as LOW because both have concrete reproducible behaviors (F2 has the `Uncertain<Known<T>>` bypass that the gate-1 F1 widening does NOT close even after landing; F3 carries forward a Stage 39 F5 pattern that has now widened to 34 names and remains uncovered by any test).

F1 is HIGH but the fix is already authored on the working tree (`gate-2-audit-stash-broader-launder-lane` stash) and structurally generalizes the gate-1 F1 reject — same pattern as Stage 39 audit's "F1 is HIGH but already being remediated by a parallel lane". All Stage 40 builtin-shadow probes pass; the F2 fix at `_register_fn` is complete and covers the entire reserved-name surface. All 20 wrong-arity probes reject cleanly; all 8 wrong-source transition probes reject cleanly (with a 2-error cascade from the TyUnknown sentinel — pre-existing pattern, family-wide, not Stage-40-specific). All 12 AD probes (6 forward + 6 reverse) produce the correct identity-chain gradient through chained modal+temporal+frame wrappers; the preemptive `_FRAME_IDENTITY_AD_NAMES` registration at `autodiff.py:203-205` and the parallel reverse-mode arm at `autodiff_reverse.py:683-687` are complete. All 16 method probes on the 8 refinement-traversal helpers return the expected value — the Stage 39 H1+H2 lesson WAS internalized; no symmetric helper-coverage gap exists.

**Verdict**: 1 HIGH (in-flight, fix authored) + 0 MEDIUM + 2 LOW + 3 OBS — gate-2 NOT CLEAN; F1 is the blocker but the fix is already in the working tree. Closing F1 brings gate-2 to 0 HIGH, 0 MEDIUM, 2 LOW, 3 OBS — which clears the CLEAN bar under the brief's "anything below HIGH conf 75 → OBS" calibration ONLY IF the two LOW findings are also explicitly accepted as deferred or downgraded to OBS by the gate-3 reviewer.
