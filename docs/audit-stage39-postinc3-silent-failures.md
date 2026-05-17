VERDICT: 1 HIGH, 2 MEDIUM, 3 LOW, 2 OBS

# Stage 39 Inc 3 silent-failure audit

Surface: commit `01b3b86` on `main` (Stage 39 OPENS — Inc 0+1+2+3 temporal types). Base: `9fcc621` (Stage 38 closed).
Files audited: `helixc/frontend/typecheck.py`, `helixc/frontend/autodiff.py`, `helixc/frontend/autodiff_reverse.py`, `helixc/ir/lower_ast.py`, `helixc/examples/dogfood_12_temporal_lifecycle.hx`, `helixc/examples/run.py`, `helixc/tests/test_stage39_temporal.py`.

Stage 39 introduces `TyTemporal` (kinds: past / present / future / eternal), 4 intro + 4 elim + 4 cross-temporal transition builtins (12 total), registers all 12 in `_BUILTIN_NAMES`, `AD_KNOWN_PURE_CALLS`, `_FRAME_IDENTITY_AD_NAMES`, and the lower-AST identity arm. It mirrors the Stage 38 (frame) playbook verbatim. The audit reference template is `docs/audit-stage38-postinc3-silent-failures.md`.

## Audit methodology executed

1. Stashed in-flight Inc-4 fix lane work (`git stash push` on `typecheck.py`, `autodiff.py`, `test_stage39_temporal.py`, `stage39-progress`) to ensure probes ran against the SHIPPED commit `01b3b86`, not the in-progress patch.
2. Read full diff `git diff 9fcc621..01b3b86` for each affected file; cross-referenced against the Stage 38 closure-gate-1 silent-failure report.
3. End-to-end probes via `python -c` harness:
   - 24 wrong-arity probes (12 builtins × {0-arg, 2-arg}) — all rejected.
   - 15 wrong-kind transition probes (each transition × every wrong source kind incl. Eternal and bare `i32`) — all rejected.
   - 6 transition-pipeline lowering probes — all lower as identity.
   - 6 forward-mode AD probes via `differentiate(...)` — all produce `x + x` for `x*x` wrapped.
   - 6 reverse-mode AD probes via `differentiate_reverse(...)` — all produce `x + x` for `x*x` wrapped.
   - 3 wrapper-stacking probes (`Past<Future<T>>`, `Eternal<Past<T>>`, `Past<WorldFrame<T>>`) — silently accepted.
   - 4 user-defined-fn shadowing probes — collisions silent.
   - 1 IR-only lowering probe (`Lowerer(prog).lower()` skipping typecheck) for wrong-arity safety net.
   - Direct method probes on `_compatible`, `_erase_refinement`, `_contains_refinement`, `_is_refinement_container`, `_refinement_proof_carried` with synthetic `TyTemporal` instances.
4. Cross-referenced findings against the two parallel Stage 39 audits already on disk (`audit-stage39-postinc3-type-design.md` H1+H2; `audit-stage39-postinc3-codereview.md` S39-CR-001) so the silent-failure-hunter lane records its independent verification of the same gap rather than duplicating numbering.

## Findings

### F1 [HIGH conf 92] `TyTemporal` is silently invisible to 6 of 7 refinement / compatibility visitor helpers

**Citation**: `helixc/frontend/typecheck.py` at commit `01b3b86`:
- `4848` `_refinement_proof_carried` (target/value_ty pair) — has TyMemTier and TyFrame arms; no TyTemporal arm.
- `5507` `_refinement_shape_exact` (a/b pair) — same gap.
- `5552` `_erase_refinement` — same gap.
- `5664` `_contains_refinement` — same gap.
- `5683` `_is_refinement_container` (tuple membership predicate) — `TyTemporal` missing from the `isinstance(..., (TyArray, ..., TyMemTier, TyFrame, TyTensor, TyTile))` tuple.
- `5705` `_contains_refined_function` — same gap.
- `6688/6690` `_compatible` (bilateral arm + unilateral rejection arm) — both absent for TyTemporal; `TyTemporal × anything` falls through to frozen-dataclass `a == b` at the catchall.

The only TyTemporal-aware visitor in commit `01b3b86` is `_fmt` at line 6858. Total `isinstance(_, TyTemporal)` sites in the typecheck file: 4 (call dispatch at 3327, 3356; resolver return at 1130; `_fmt` at 6858). TyFrame post-Stage-38-close has 8 isinstance sites.

**Reasoning**: This is a verbatim replay of the Stage 38 closure-gate-1 type-design H1+H2 finding (commit `1c6e047`) — the fixes were authored for TyFrame and never propagated back into the family template. Because every Stage 39 reasoning helper that walks through wrapper types lacks a TyTemporal arm, refinements UNDER a temporal wrapper are invisible to the refinement subsystem. Concrete silent / misleading-failure consequences I reproduced live against commit `01b3b86`:

1. **Refined-inner proof carry**: a function `fn id_past(p: Past<NonZero>) -> Past<NonZero>` that simply returns its argument produces — when called via `id_past(into_past(7))` — the diagnostic `"call to 'id_past': arg 'p' expects Past<NonZero>, got Past<i32>"`. The analogous WorldFrame test produces a different, refinement-aware diagnostic `"... type conversion from WorldFrame<i32> to WorldFrame<NonZero> would change refined parameter or return requirements in Stage 31"`. The Temporal variant skips the refinement-aware diagnostic path entirely because `_is_refinement_container` returns False for TyTemporal, so the H1 fix at typecheck.py:5430-5432 never fires.
2. **`_compatible` cross-kind silent fallthrough**: `TyTemporal × anything else` falls through to dataclass equality. For simple shapes this happens to reject (probed: `Past × Future`, `Past × i32`, `Past × WorldFrame` all rejected by dataclass equality). But the moment an inner is `TyVar`, `TySize`, or `TyRefined`, dataclass equality compares wrong — generic functions over `Past<T>` (a literal motivating use case per the `TyTemporal` docstring at typecheck.py:255-264) will silently fail to typecheck at every call site without the structural-recursion arm that every other wrapper has.
3. **`_erase_refinement(Past<NonZero>)` returns the wrapper unchanged** — the wrapper falls through to the trailing `return ty` so any pass that calls `_erase_refinement` to widen a refined type silently keeps the refinement under the temporal tag. The direct probe is unambiguous; the live observable effect depends on which call site invokes it first.

The silent-failure-specific framing (vs. the parallel type-design audit) is this: every one of these gaps either (a) produces a less-actionable diagnostic than the analogous frame case, or (b) lets the wrapper silently survive a transform that should have stripped its inner. Neither is loud; both compound across Phase-1 work where temporal types start carrying refined / generic inners.

**Note on parallel work**: a follow-up patch on the working tree (uncommitted at audit start, currently in the in-flight Inc 4 fix lane) is adding the missing arms. This audit records the gap as shipped at `01b3b86` so the closure-gate-1 ledger has a record that Stage 39 entered the gate with the same H1+H2 hole Stage 38 entered its own gate with.

**Remediation**: 6 verbatim TyFrame-pattern copies, ~9 LOC total. See `audit-stage39-postinc3-type-design.md` H2 for the exact patch text. Verify post-fix with the 3 canary probes named in that finding plus a `_compatible(Past<TyVar>, Past<i32>)` probe.

### F2 [MEDIUM conf 88] User functions named `into_past` / `forecast` / `actualize` / `recall_past` (etc.) are silently shadowed by the new builtins

**Citation**: `helixc/frontend/typecheck.py:3304-3367` (dispatch arms for all 12 Stage 39 builtins) all run BEFORE the user-function lookup. `_BUILTIN_NAMES` at 1913-1923 newly reserves 12 names. No reserved-name guard at fn-declaration time.

**Reasoning**: Verbatim Stage 38 F3 carry-over with widened surface. The 12 newly-reserved names include `forecast`, `actualize`, `recall_past`, `to_past`, and the 8 `into_X` / `from_X` constructors. Live probes:

```
fn forecast(x: i32) -> i32 { x * 2 }
fn main() -> i32 { forecast(10) }
=> typecheck error: "forecast() requires Present<T>, got i32"
```

The user's `forecast(x)` definition silently becomes dead code; the call resolves to the Stage 39 builtin, which rejects the bare `i32` with a temporal-typing diagnostic that gives the user no clue their `fn forecast` was shadowed. Same pattern for `actualize` (very plausible business-logic name in planning / scheduling / financial code), `recall_past` (cognition/memory code), and the 4 `into_*` / 4 `from_*` constructors.

The risk surface is higher than Stage 38's frame names: `world_to_robot` and `camera_to_world` are robotics-specific niche names, but `forecast` and `actualize` are general-purpose verbs that appear in many real codebases. An AGI-shaped codebase exercising temporal reasoning is precisely the place where users will write functions named `forecast(...)` as their primary domain logic.

**Why MEDIUM not LOW**: the Stage 38 closure-gate F3 (HIGH conf 85 in that audit, marked MEDIUM here because Stage 39 inherits it rather than newly introducing the pattern) deferred this. Stage 39 widens the silent-shadow surface by 12 highly-plausible names. The pattern is mechanically the same — but the names are worse.

**Remediation**: at `fn` declaration time, if `name in self._BUILTIN_NAMES`, emit a `TypeError_(... 'name' shadows compiler builtin)`. One ~5 LOC addition shared across all stages, would close Stage 36/37/38/39 builtin-shadow holes simultaneously. Cheap; high signal.

### F3 [MEDIUM conf 80] `into_X` silently accepts an already-wrapped TyTemporal, producing nonsense layered types

**Citation**: `helixc/frontend/typecheck.py:3304-3312` (`_temporal_intro` dispatch). The dispatch returns `TyTemporal(kind=..., inner=arg_tys[0])` without inspecting `arg_tys[0]`.

**Reasoning**: Live probes:

```
into_past(into_future(42))    -> typechecks as Past<Future<i32>>, no diagnostic
into_eternal(into_past(42))   -> typechecks as Eternal<Past<i32>>, no diagnostic
into_past(into_world(42))     -> typechecks as Past<WorldFrame<i32>>, no diagnostic
```

Each result is semantic nonsense for temporal reasoning — a fact cannot be "past inside future" and `Eternal<Past<T>>` contradicts the docstring's explicit "Eternal is timeless" property at typecheck.py:255-264. The frame-cross case `Past<WorldFrame<i32>>` is mostly harmless (a frame-tagged value in the past is plausible), but the same dispatch accepts the contradictory cases without distinction.

**Silent-failure framing**: the temporal-kind invariant the Stage 39 docstring announces ("a value tagged with a temporal kind" — implicitly: ONE kind) is violated silently. No diagnostic, no warning, no `TyUnknown` recovery. The bug only surfaces when a downstream transition tries to operate on the nested wrapper — at which point the user gets a confusing "requires Present<T>, got Past<Future<i32>>" diagnostic without explanation that the original `into_past(into_future(...))` was the actual mistake.

**Why MEDIUM not LOW**: this is family-symmetric with tier and frame (Stage 37/38 both accept layered wrappers), but TEMPORAL kinds have a stronger exclusivity property — a fact is in *exactly one* temporal kind at a time. The semantic invariant is intrinsic, not just a convention. Phase-1 AGI temporal reasoning code will hit this within a handful of complex pipelines.

**Why not HIGH**: no live runtime bug yet (everything lowers as identity). The corruption only surfaces when later code tries to unwrap, and at that point the diagnostic, while opaque, still rejects rather than silently miscomputes.

**Remediation**: in `_temporal_intro` dispatch, after the arity check, add:

```py
if isinstance(arg_tys[0], TyTemporal):
    self.errors.append(TypeError_(
        f"{bn}() input is already temporally-tagged "
        f"({self._fmt(arg_tys[0])}); use a transition "
        f"(to_past/forecast/recall_past/actualize) to change "
        f"kinds, or from_{arg_tys[0].kind}() to unwrap first",
        expr.span,
    ))
    return TyUnknown(hint=bn)
```

Also worth applying to `into_X` for frames (Stage 38 L1 deferred) at the same time. ~10 LOC for both families.

### F4 [LOW conf 75] Cross-temporal transition diagnostics omit the source's inner type (Stage 38 F6 carry-over)

**Citation**: `helixc/frontend/typecheck.py:3347-3365`. The diagnostic template reads:

```
to_past() requires Present<T>, got Past<i32>
forecast() requires Present<T>, got Eternal<i32>
```

The `<T>` is a literal placeholder. `_fmt` at line 6858 can already render the actual inner type when given a `TyTemporal` value, but the template doesn't use it.

**Reasoning**: This is the Stage 38 F6 finding propagated verbatim. UX impact: a user who sees `requires Present<T>, got Past<i32>` and is deep in a generic-parameter chain may waste cycles confirming `T` resolved correctly elsewhere; the actual error is the wrapper kind, not the inner type. Lower-severity because the diagnostic IS produced and IS actionable about the wrapper-kind mismatch, just not maximally clear about the inner.

**Remediation**: same cheap fix as Stage 38 F6 — render the source's inner type in the "requires" half when the source is a TyTemporal, e.g. `requires Present<{inner_fmt}>, got Past<i32>`. ~3 LOC per arm, applied at 3331-3335 (from_X) and 3361-3365 (transitions). Symmetric with the recommended Stage 38 fix.

### F5 [LOW conf 70] IR identity-lowering arm silently drops `args[1..]` if a wrong-arity call ever slips past typecheck

**Citation**: `helixc/ir/lower_ast.py:1986-2014` (post-Stage-39 surface). The Stage 39 additions extend the existing identity-lowering arm: the guard is `if expr.callee.name in (...12 frame names + 12 temporal names...) and len(expr.args) == 1: return self._lower_expr(expr.args[0])`.

**Reasoning**: Today, the typecheck arity check at typecheck.py:3304-3312 / 3319-3327 / 3348-3355 always rejects wrong-arity calls before they reach lowering (probed live — all 24 wrong-arity probes hit typecheck errors, none reach IR). Plus the lower-AST guard `len(expr.args) == 1` makes wrong-arity calls fall through to the generic `<unknown>` callee raise:

```
NotImplementedError: unknown function 'into_past' in IR lowering
at L:C; run typecheck first
```

— which is an opaque compiler-internal exception whose error text asserts "run typecheck first" even when typecheck DID run and emit a diagnostic (the user might have a build pipeline that ignores typecheck warnings). The "skipped-typecheck" probe reproduces this:

```
Lowerer(parse('fn main() -> i32 { let x: i32 = into_past(1, 2); 0 }')).lower()
=> NotImplementedError: unknown function 'into_past' in IR lowering at 1:33; run typecheck first
```

**Risk dimension**: any future refactor that (a) relaxes the `len == 1` guard inside the identity arm OR (b) reorders the dispatch so a wrong-arity call no longer falls through to the catchall — would silently drop side-effecting args. Same Stage 38 F4 pattern, widened by 12 names. No active runtime bug; defense-in-depth is what is missing.

**Remediation**: add an explicit assertion inside the identity arm asserting `len(expr.args) == 1, f"{callee_name} arity guard violated; typecheck should have rejected"` so future relaxation gets a deterministic crash rather than a silent arg-drop. ~2 LOC. Or convert the dispatch to `(name in IDENTITY_NAMES) and assert len(args) == 1`.

### F6 [LOW conf 65] Dogfood witness collapses to `raw_in == raw_out` arithmetic — temporal-typing semantics are entirely typecheck-time, runtime witness is dead

**Citation**: `helixc/examples/dogfood_12_temporal_lifecycle.hx:30-93`. Per-observation witness is `obs == raw_input` (line 70-72); `eternal_ok` is `from_eternal(into_eternal(1)) == 1` (line 84-85); `all_ok` is the 5-way product (line 89).

**Reasoning**: Stage 38 O1 verbatim carry-over. Because all 12 Stage 39 builtins lower as identity, the runtime witness collapses to "raw_in == raw_out" — true by construction of identity-lowering, independent of which transition was used. If `forecast`, `actualize`, `to_past`, `recall_past` were all aliased to a single identity function (or to one another), every per-observation witness still fires `1`. The runtime test does NOT validate that `forecast` is distinct from `actualize`, only that all four are individually identity.

The prose at lines 30-34 explicitly claims "Witness is collapse-resistant: each observation must round-trip exactly, AND the chain must be type-correct end-to-end" — the second half ("type-correct end-to-end") is only validated at typecheck (compile-time). The binary witness is silent on whether the right transition was used in the right slot.

**Why LOW**: This is intrinsic to Phase-0 design. Phase-1 transitions will gain real semantics (tick-counter increment, history-cone update, etc.) at which point the runtime witness will start biting. The typecheck pass IS the witness for Phase-0. Same call as Stage 38 O1 — observation, not active silent-failure of audited code.

**Remediation**: defer until Phase-1 transition math lands. At that point, the dogfood should be revised so that, e.g., `forecast` increments a tick counter that the witness can check. Out-of-scope for closure gate-1.

## OUT OF SCOPE — observations (no severity)

- **O1** (test-suite gap, already covered by `audit-stage39-postinc3-codereview.md` S39-CR-002). `test_reflection.py` has parallel `test_dogfood_10_memory_tiers` (Stage 37) and `test_dogfood_11_spatial_frames` (Stage 38) entries but no `test_dogfood_12_temporal_lifecycle`. From the silent-failure lens: a regression that breaks `dogfood_12_temporal_lifecycle.hx` end-to-end (e.g., a `run.py` DEMOS-dict key collision or path bug, an `@pure` decorator interaction across the four helper fns, a witness-arithmetic regression) would not be caught by any Stage 39 test — the silent-failure mode is "the dogfood breaks unnoticed because nothing references it from CI". Fix is 7 LOC of test_reflection.py addition. Cross-listed here because the code-review lane already filed it.

- **O2** (dispatch perf, family carry-over). The three dispatch arms at typecheck.py:3298-3370 rebuild `_temporal_intro` / `_temporal_elim` / `_temporal_transitions` dicts on every Call-expression typecheck visit. Negligible cost (3 small dicts) but pattern-symmetric to the Stage 38 O2 observation; same hoisting opportunity.

## Summary

ONE HIGH (TyTemporal silently invisible to 6 of 7 refinement / compatibility visitor helpers; produces less-actionable diagnostics for refined-inner cases and risks silent acceptance for generic-inner cases at any call site that crosses the structural-recursion arms — verbatim replay of Stage 38 H1+H2 already known and being patched by the in-flight Inc 4 lane), TWO MEDIUM (builtin shadowing of 12 highly-plausible user fn names; into_X silently accepts already-wrapped TyTemporal producing nonsense `Past<Future<T>>`-shaped types), THREE LOW (transition diagnostic omits source inner type; IR identity-arm has no defense-in-depth assertion; dogfood runtime witness is dead for Phase-0 transition semantics), TWO OBS (missing `test_dogfood_12_temporal_lifecycle` in test_reflection.py; per-call dict reallocation).

F1 is HIGH but already being remediated by a parallel lane (uncommitted at audit time, comment marker `Stage 39 closure gate-1 type-design H1/H2/H3 fix` present in the in-flight patch). All Stage-38-F1 (wrong-arity silent acceptance) and Stage-38-F2 (AD chain rule missing) holes are CLOSED in Stage 39 ship state — 24 wrong-arity probes all reject cleanly with named-arg diagnostics; 12 forward-mode AD probes and 12 reverse-mode AD probes all produce the correct identity-chain gradient (`x + x` for `x*x` wrapped). The Inc 0+1+2+3 ship is a faithful Stage 38 port that proactively closed the two Stage 38 closure-gate-1 silent-failure findings and inherited only the type-design symmetry gap that Stage 38's own gate-1 then patched.

**Verdict**: 1 HIGH (already in flight) + 2 MEDIUM + 3 LOW + 2 OBS — gate-1 NOT CLEAN; F1 is the blocker but the fix is already authored in the working tree.
