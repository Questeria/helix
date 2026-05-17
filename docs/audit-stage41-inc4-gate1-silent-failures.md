VERDICT: 2 HIGH, 0 MEDIUM, 2 LOW, 3 OBS

# Stage 41 Inc 0+1+2+3 gate-1 silent-failure audit

Surface: commit `7448bf5` (Stage 41 Inc 1+2+3: causal/intent types shipped end-to-end). Base: `5dd478a` (Stage 40 closure trail merged; Stage 40 audit ledger complete). Audit target is the SHIPPED surface at `7448bf5`; probes ran against the historical typecheck.py blob `ffa46be2c704e7295ea0747d473ba529f3eec3d2` extracted via `git show 7448bf5:helixc/frontend/typecheck.py`. The working tree at audit start is `4e74244` (Stage 42 OPENS) which, as documented in OBS-Z below, accidentally reverted commit `246c33f`'s gate-1 fixes — so the working-tree blob hash matches 7448bf5 again. Probe results are consistent with the documented shipped behavior.

Stage 41 introduces `TyCausal` (kinds: cause / effect / joint / independent), 4 intro + 4 elim + 3 cross-causal transitions (`propagate` Cause→Effect, `aggregate` Effect→Joint, `isolate` Joint→Independent) — 11 causal builtins total. Registered in `_BUILTIN_NAMES` (2037-2041), `AD_KNOWN_PURE_CALLS` (113-122 add-block), `_FRAME_IDENTITY_AD_NAMES` (210-219 add-block), `lower_ast.py` identity arm (2023-2033 add-block), `_resolve_type` causal_map (1229-1242 add-block). The Stage 41 F1 cross-causal launder guard is at typecheck.py 3781-3816 — structurally a copy of the Stage 40 gate-2 cross-modal F1 guard (typecheck.py 3650-3686) EXCEPT for one missing clause (the `inner_is_shadowed` cascade-suppression check added in Stage 40 closure gate-3 at 3643-3656). Stage 40 gate-2 audit reference template: `docs/audit-stage40-inc4-gate2-silent-failures.md`. Stage 41 is the 5th AGI semantic-type family — completes the 4-stack quintet (memory / spatial / temporal / modal / causal). Phase-0 thesis: TyCausal is identity-at-IR; causal kind discipline lives purely at the type system level.

## Audit methodology executed

1. Read full diff `git diff 5dd478a..7448bf5` for each affected file (typecheck.py +244, autodiff.py +13, lower_ast.py +12, examples/run.py +5, examples/dogfood_14_causal_lifecycle.hx +72 new, tests/test_stage41_causal.py +353 new). Cross-referenced against the Stage 40 closure-gate-2 silent-failure report (`audit-stage40-inc4-gate2-silent-failures.md`) for the same finding patterns.
2. Extracted typecheck.py at the audit-target commit (`git show 7448bf5:helixc/frontend/typecheck.py`) and probed against the historical blob to ensure the working-tree's later state did not contaminate findings. Confirmed via blob hash comparison: 7448bf5 → `ffa46be`; gate-1-fix commit 246c33f → `679f8f7`; Stage 41 CLOSED at 6f818e4 → `679f8f7`; HEAD (4e74244 Stage 42 OPENS) → `ffa46be` REVERTED. See OBS-Z.
3. End-to-end probes via `python -c` harness against the historical-blob-restored working tree:
   - 12 cross-causal direct launder probes (`into_X(from_Y(...))` for every X ≠ Y across {cause, effect, joint, independent}) — all 12 reject with F1 named-launder diagnostic + the cascaded TyUnknown propagation, identical to Stage 40 gate-2's modal pattern.
   - 32 cross-FAMILY launder probes (4 causal × 4 modal × 2 directions = 32) — all 32 silently accepted at typecheck. The 3-line `Cause -> Known -> Independent` modal-laundromat sequence (the most safety-critical case) compiles to a 406455-byte working ELF binary without a single diagnostic.
   - 28 cross-family launder probes against frame/temporal (4 causal × 4 temporal + 4 causal × 3 frame, with rewrap discipline) — all 28 silently accepted; family-wide carry-over confirming Stage 38/39/40 inherited the same gap.
   - 2 shadowed-builtin parity probes (modal vs causal F1 with user shadowing `from_known` and `from_cause`) — modal version produces 1 error (Stage 40 gate-3 fix at line 3656 fires `inner_is_shadowed`), causal version produces 2 errors (1 shadow + 1 launder; the cascade-suppression check was not mirrored at the causal F1 guard at 3781-3816). Direct asymmetry, reproduced live.
   - 3 wrapper-stacking probes (`Cause<Cause<i32>>`, `Effect<Cause<i32>>`, `Independent<Joint<i32>>`) — all 3 silently accepted; Stage 40 gate-2 F2 family-wide carry-over.
   - 2 multi-step launder probes through audited transitions (`into_cause(from_effect(propagate(c)))`, `into_independent(propagate(...))`) — both reject; the F1 guard syntactically inspects the inner CALLEE name and `propagate`'s return-type contract makes the type-check still fail on the outer kind-mismatch.
   - 2 bypass probes (let-binding `let r = from_cause(c); into_effect(r)` and `id`-helper indirection `into_effect(id(from_cause(c)))`) — both silently accepted; same Phase-0 deferral pattern documented in Stage 40 gate-1 H1.
   - 16 direct method probes on the 8 helper visitors with synthetic `TyCausal` instances (`_compatible`, `_refinement_shape_exact`, `_refinement_proof_carried`, `_erase_refinement`, `_contains_refinement`, `_is_refinement_container`, `_contains_refined_function`, `_contains_unknown_type`) plus `_fmt` (incl. bogus kind) — all 17 return expected values; preemptive TyCausal arms at typecheck.py 5302 (`_contains_unknown_type`), 5374 (`_refinement_proof_carried`), 6049 (`_refinement_shape_exact`), 6106 (`_erase_refinement`), 6230 (`_contains_refinement`), 6253 (`_is_refinement_container`), 6287 (`_contains_refined_function`), 7290 (`_compatible`), 7469 (`_fmt`) all wired correctly. `_fmt` falls through on unknown kinds (`Cause`/`Effect`/`Joint`/`Independent` map cleanly; bogus kind echoes the raw string `BAD_KIND<i32>` — defensible but no internal-corruption diagnostic).
   - 2 AD chain-rule probes (`d/dx(from_cause(into_cause(x*x)))` and `d/dx(propagate(into_cause(x*x)))`) — both produce `(x + x)` correctly through the AD_KNOWN_PURE + _FRAME_IDENTITY_AD_NAMES dispatch. Reverse-mode autodiff at autodiff_reverse.py:684 reads `_FRAME_IDENTITY_AD_NAMES` directly, so it picks up the Stage 41 additions automatically.
   - 1 test-suite scan: `test_reflection.py` contains entries through `dogfood_12_temporal_lifecycle` but NEITHER `dogfood_13_modal_lifecycle` NOR `dogfood_14_causal_lifecycle`. The Stage 40 gate-2 already entered with this gap as O1 — Stage 41 inherits it doubled.
4. Cross-referenced findings against the Stage 40 gate-2 silent-failure ledger so duplicated findings record their independent re-verification rather than restate the original. F1, F2 below are direct parity carry-overs.

## Findings

### F1 [HIGH conf 92] Causal F1 guard missing `inner_is_shadowed` parity — shadow + launder cascade is the EXACT cross-stage symmetric carry-over Stage 40 closed at gate-3

**Citation**: `helixc/frontend/typecheck.py:3781-3816` (the Stage 41 cross-causal F1 guard, code-inlined immediately after the `bn in _causal_intro` check). The guard reads:

```py
target_kind = _causal_intro[bn]
# Stage 40 F1 lesson applied preemptively: ...
if (len(expr.args) >= 1
        and isinstance(expr.args[0], A.Call)
        and isinstance(expr.args[0].callee, A.Name)
        and expr.args[0].callee.name
            in _causal_elim_kind
        and not isinstance(arg_tys[0], TyUnknown)):
    source_kind = _causal_elim_kind[
        expr.args[0].callee.name]
    if source_kind != target_kind:
        ...
        self.errors.append(TypeError_(
            f"{bn}(from_{source_kind}(...)) launders ..."))
        return TyUnknown(hint=bn)
```

The Stage 40 gate-3 H1 amendment added the `inner_is_shadowed` clause to the cross-modal F1 guard at typecheck.py:3643-3656:

```py
inner_is_shadowed = (
    len(expr.args) >= 1
    and isinstance(expr.args[0], A.Call)
    and isinstance(expr.args[0].callee, A.Name)
    and expr.args[0].callee.name
        in self._shadowed_builtin_names
)
if (...
        and not isinstance(arg_tys[0], TyUnknown)
        and not inner_is_shadowed):
```

This clause is MISSING from the Stage 41 causal F1 guard. The codebase has the comment "Stage 40 F1 lesson applied preemptively" at line 3776 of the causal guard, which advertises symmetric defense — but the symmetry stops at the cross-kind reject and does NOT carry over the gate-3-added `inner_is_shadowed` cascade-suppression check. The Stage 40 gate-3 closure trail at commit `ac34df9` was the last commit on disk before `7448bf5`; the Stage 41 author had the gate-3 lesson literally one commit upstream and inlined the gate-2 form rather than the gate-3 form.

**Reasoning**: Reproduced live against `7448bf5`:

```py
# SHADOW from_cause (user fn) + launder pattern
src = """
fn from_cause(x: i32) -> Effect<i32> { into_effect(x) }
fn make() -> Effect<Effect<i32>> {
    into_effect(from_cause(42))   // <- shadowed inner + launder-shaped outer
}
fn main() -> i32 {
    let r: Effect<Effect<i32>> = make();
    from_effect(from_effect(r))
}
"""
# Stage 41 causal version: 2 errs
#   shadow diagnostic at fn-decl + spurious launder diagnostic at the outer into_effect
# Stage 40 modal version (equivalent): 1 err  (the inner_is_shadowed clause suppresses the launder)
```

The asymmetry mirrors EXACTLY the Stage 40 gate-2 → gate-3 closure cycle:
- Stage 40 gate-2 audit `F1 [HIGH]` flagged the symmetric-but-incomplete cross-modal launder guard.
- Stage 40 gate-2 fix (committed before audit close) added the `inner_is_shadowed` clause.
- Stage 40 gate-3 H1 audit re-confirmed the fix held under shadow-inner adversarial probes.
- Stage 41 author hand-authored a parallel cross-causal F1 guard for the new family, copied the gate-2 fix-form, did NOT copy the gate-3 H1 amendment.

The `_shadowed_builtin_names` set is a member of `self` and includes all 8 causal elim/intro names by the time the body of `main` is visited (per `_register_fn` registration at 966). The fix is identical structurally — add 7 LOC mirror of 3643-3656 inside the causal F1 guard at 3781, and add the `and not inner_is_shadowed` conjunct at the analogous launder-condition line.

The user impact is exactly the same as the Stage 40 gate-2 H2 "1 + 0 noise" invariant the gate-3 fix preserved. A contributor who refactors `from_cause` into a user fn (most common shadow vector: porting a function from a sibling project) triggers BOTH the shadow diagnostic at the fn-decl site AND the spurious launder diagnostic on every call site that happens to be wrapped in `into_<otherkind>(...)`. The cascading noise distracts from the actual fix (rename the user fn) and, worse, the launder diagnostic suggests `propagate(c)` as the "right" path — pointing the user further from the actual root cause.

**Why HIGH and not MEDIUM**: this is a literal repeat of the Stage 40 gate-3 closure-blocker, in the exact pattern Stage 40's H1 documented as "the F1 guard inspects the INNER call's syntactic name without checking `_shadowed_builtin_names`". The Stage 41 implementation pre-existed the Stage 40 gate-3 fix being committed (commits ordered: gate-1-Stage-41 at 7448bf5 ↓ gate-2-Stage-40 at e8fb593 ↓ gate-3-Stage-40 at ac34df9 ↓ Stage 41 OPENS at e82a742) — actually NO, the Stage 41 OPENS commit is at `e82a742` (Stage 41 Inc 0 ledger) which is between the Stage 40 closure trail and 7448bf5; the Stage 40 gate-3 closure docs at `ac34df9` are LITERALLY the predecessor of `e82a742` on the same lineage. The Stage 41 author had the gate-3 fix in the same file when authoring 7448bf5 — and copied only the gate-2 form. Cross-stage symmetric carry-over with the fix already-in-the-same-file is the textbook gate-1 HIGH for AGI semantic-type compounding.

**Why not CRITICAL**: the broken case requires the user to ALSO have shadowed a `from_X` builtin, and the user already gets the shadow diagnostic at the fn-decl site. The "1 + 1 noise" outcome is not a silent failure — it's a noisy false-positive cascade. CRITICAL is reserved for silent acceptance of safety-violating programs. This is HIGH because it materially violates an invariant the Stage 40 audit ledger explicitly established and the fix is already-authored in the same file.

**Remediation**: lift the F1 launder-condition body at typecheck.py:3781-3816 to mirror the cross-modal F1 guard at 3643-3656. Mechanically: insert the `inner_is_shadowed = (...)` block at the analogous position (right before the launder-pattern `if` at 3781), and add `and not inner_is_shadowed` to the launder-condition's `if`. ~7 LOC. Add a regression test mirror of the Stage 40 gate-3 H1 test (`test_stage40_gate3_f1_inner_is_shadowed_suppresses_launder_cascade` — same pattern, swap modal→causal names). Verify: the post-fix test count for `test_stage41_causal.py` should be 23 passing instead of the as-shipped 22 (the test was actually added in the gate-1-closure commit 246c33f along with the fix; both then got REVERTED by the Stage 42 OPENS commit 4e74244 — see OBS-Z below).

### F2 [HIGH conf 88] Cross-family modal/causal launder is silent in ALL 32 directions — `Cause -> Known -> Independent` 3-line laundromat compiles to a working ELF

**Citation**: typecheck.py:3781-3816 (causal F1 guard inspects only `_causal_elim_kind`); typecheck.py:3650-3686 (modal F1 guard inspects only `_modal_elim_kind`). NEITHER guard inspects the OTHER family's elim names. Family-wide carry-over from Stage 37/38/39/40 — the cross-family launder surface has never been audited as a finding before, but Stage 41 is the first stage where the laundromat pattern composes through MORE THAN ONE prior family, and where the safety-critical exemplar (Cause → Known → Independent) becomes a 3-line program with no compile-time signal.

**Reasoning**: A systematic 32-probe sweep over the 4 causal × 4 modal × 2 directions (into_C(from_M(v)) and into_M(from_C(v))) gives:

```
Tested 32 cross-family launder combinations
Silent acceptances: 32/32
```

Including:
- `into_cause(from_uncertain(...))` — laundering an explicitly Uncertain value into a Causal claim. Stage 40's headline AI-safety property says Uncertain values must NOT be upgraded without an outside observation; the F1 guard for Stage 40 closes the within-modal upgrade route. Re-routing through Stage 41 causal kinds bypasses the modal guard entirely.
- `into_independent(from_known(k))` — asserting "this previously-known fact has NO upstream causal dependency" with zero audit trail. Stage 41's `isolate` transition exists for exactly this Joint → Independent epistemic step; the laundromat skips it.
- `into_known(from_cause(c))` — asserting "this proposition (previously a causal upstream) is now a directly-known fact". This is the EXACT category mistake Stage 40's docstring at `typecheck.py:268-278` calls "the heart of many AI safety failures" — and the Stage 41 surface ships with this route silently accepted.

The most damning concrete reproduction:

```py
fn main() -> i32 {
    let c: Cause<i32> = into_cause(42);
    let k: Known<i32> = into_known(from_cause(c));
    let i: Independent<i32> = into_independent(from_known(k));
    from_independent(i)
}
# typecheck: 0 errs
# IR-lowered + ELF compiled successfully (406455 bytes)
```

A 6-line program (4 expression lines) takes a value that the type system originally tagged as a Cause and ends with the type system asserting it has NO upstream — without using the audited `isolate` (Joint → Independent) transition AT ALL. The launderomat exploits modal Known<T> as an opaque carrier to strip the causal kind. Same pattern works in every direction.

The Stage 41 dogfood `dogfood_14_causal_lifecycle.hx` includes a `cross_stack_known_cause` helper at lines 51-58 that DOES use the legitimate composition `Known<Cause<i32>>` and unwraps in the right order (`from_cause(from_known(kc))`) — proving the legitimate composition is intentional and supported. But that's a different pattern: `Known<Cause<i32>>` keeps both kinds; the laundermat strips them. The type system has no syntactic way to distinguish "stripping while wrapping" from "stripping a wrap that already exists" because all 4 (modal) + 4 (causal) eliminators are typed `Kind<T> -> T`. The F1 guard depends on the immediate-syntactic-inner-call pattern; it must inspect ALL elim names across ALL families for cross-kind defense to be complete.

The same family-wide gap exists at the Stage 38/39/40 surface boundaries too — Stage 38 added 28 modal-temporal/frame silent vectors I separately verified; Stage 39 added another 28; Stage 40 added another 28. Stage 41 adds 28 (causal vs temporal+frame) + 32 (causal vs modal) = 60 NEW silent vectors. The total cross-family silent launder vector count is now ≥ 144 (Stage 38: 28, Stage 39: +28, Stage 40: +28, Stage 41: +60). None of these have ever appeared in a prior audit ledger as a finding.

**Why HIGH and not MEDIUM**: the safety-critical exemplar `Cause → Known → Independent` compiles to a working binary in 3 expression lines without a single diagnostic. The launderomat is the EXACT pattern Stage 40's F1 was authored to close within the modal family — the cross-family bypass is the same shape, same severity, same fix-pattern (extend the F1 guard's elim-name lookup to include other families' elim names). The gap has been latent since Stage 38 and has compounded silently with each new family. Stage 41 is the trigger to rate it HIGH because (a) the laundermat now ROUTES through THREE families to make a SINGLE-FAMILY safety claim — the new compounding makes the AGI semantic-type framework structurally vulnerable to "family X laundromat" attacks, and (b) the AI-safety claim in Stage 40's docstring is materially undermined by the Stage 41 cross-family hole even though Stage 40's own ledger never caught it.

**Why not CRITICAL**: the launder still requires the user to write `from_X(into_Y(...))` explicitly — it is not a sneaky compiler-internal bypass. And typecheck errors HAVE to be honored downstream per Phase-0 truth-gate (the same "lower-of-typecheck-rejects" observation from Stage 40 gate-2 O3). HIGH reflects the gap between the headline guarantee and the cross-family enforcement; the fix is small but cross-cutting.

**Remediation**: at each per-family F1 guard, change the inner-callee lookup from `expr.args[0].callee.name in _<family>_elim_kind` to `expr.args[0].callee.name in (_modal_elim_kind | _causal_elim_kind | _temporal_elim | _frame_elim_names)`. Compute the inner's source kind by querying the union map, then emit a kind+family-specific diagnostic. The cross-family diagnostic should hint that "kind cannot be changed by going through another family" — there is no audited cross-family transition in Phase-0 by design.

Alternative (simpler): hoist a single module-level `_ALL_ELIM_NAMES_BY_FAMILY: dict[str, str]` map that returns the family+kind for each elim name, then unify all four per-family F1 guards into one. ~30 LOC consolidation that closes 144 silent vectors. Defense-in-depth: also reject cross-family laundromat patterns through `let`-bindings and helper-fn indirection at a future stage (same Phase-0 known-limit caveat as the within-family F1 guard).

Add regression tests: 12 cross-family launder probes (3 prior families × 4 causal kinds, both directions) for Stage 41; symmetric additions for Stage 38/39/40 if the fix is back-applied. ~60 LOC of test code total.

### F3 [LOW conf 75] Wrapper-stacking — `into_X` silently accepts an already-causally-wrapped value, producing semantically-incoherent nested types

**Citation**: `helixc/frontend/typecheck.py:3768-3818` (`_causal_intro` dispatch). After the F1 guard, the dispatch returns `TyCausal(kind=target_kind, inner=arg_tys[0])` without inspecting whether `arg_tys[0]` is already a TyCausal.

**Reasoning**: Live probes against `7448bf5`:

```
into_cause(into_cause(42))           -> Cause<Cause<i32>>           (0 errs)
into_effect(into_cause(42))          -> Effect<Cause<i32>>          (0 errs)
into_independent(into_joint(42))     -> Independent<Joint<i32>>     (0 errs)
```

Same family-symmetric carry-over as Stage 40 F2 (modal-on-modal), Stage 39 F3 (temporal-on-temporal), Stage 38 L1 (frame-on-frame). Causal-on-causal nesting is semantically incoherent: a proposition has exactly one causal status at a time. `Cause<Cause<i32>>` is meaningless (a cause-of-a-cause is still a cause; the nesting doesn't track lineage); `Effect<Cause<i32>>` confuses the downstream/upstream axis (an effect-OF-a-cause is just an effect with the causal lineage erased into the type tag).

The `Effect<Cause<i32>>` case is particularly insidious because the dogfood DOES legitimately use `Known<Cause<i32>>` (the cross-family wrapper at dogfood lines 51-58). The dispatch can't tell `Known<Cause<i32>>` (legitimate: "I directly observed that this was a cause") from `Effect<Cause<i32>>` (incoherent: "this is an effect-of-a-cause", but the lineage is lost because Phase-0 ships no audited Cause→Effect-of-Cause transition) — they have isomorphic AST shapes. The F1 guard at 3781-3816 doesn't fire because the inner is `into_<kind>(...)`, not `from_<kind>(...)`.

**Why LOW not MEDIUM**: family-symmetric with Stage 38 L1, Stage 39 F3, Stage 40 F2 — all 3 prior families have this LOW finding deferred. The Stage 40 gate-2 F2 verdict notes: "Stage 40 inherits the family-wide pattern. ... downgraded to LOW because (a) test coverage demonstrates the lateral compose case is intentional, (b) the F1 nesting bypass route surfaces only under deliberate contributor adversarial intent, and (c) the Stage 39 finding documented the lesson and the closure decision was 'defer'." Stage 41 inherits the verdict exactly. Calibration-borderline LOW because no concrete bypass-route (analogous to Stage 40 F2's `Uncertain<Known<T>>` strip-pattern) was found for causal-on-causal at Stage 41 — the bypass requires either a cross-family laundromat (covered by F2) or a same-family helper-fn launder (covered by the Phase-0 known limits of the F1 syntactic guard).

**Why not OBS**: the `Effect<Cause<i32>>` case is concretely accepted and produces a value the type system claims is causally-classified-as-effect but whose inner type is causally-classified-as-cause — a contradiction. Reproducible today; not just a hypothetical.

**Remediation**: in `_causal_intro` dispatch at 3768, after the F1 guard at 3816, add:

```py
if isinstance(arg_tys[0], TyCausal):
    self.errors.append(TypeError_(
        f"{bn}() input is already causally-tagged "
        f"({self._fmt(arg_tys[0])}); use a transition "
        f"({_causal_transitions_named()}) to change "
        f"kinds, or from_{arg_tys[0].kind}() to unwrap first",
        expr.span,
    ))
    return TyUnknown(hint=bn)
```

~8 LOC. Symmetric with the deferred fixes for Stage 38/39/40 families. Worth applying to all 4 families simultaneously to close the entire backlog with one fix-sweep (Stage 38 L1 + Stage 39 F3 + Stage 40 F2 + Stage 41 F3 → one consolidated change).

### F4 [LOW conf 75] IR identity-lowering arm silently drops `args[1..]` if a wrong-arity causal call slips past typecheck

**Citation**: `helixc/ir/lower_ast.py:2020-2033` (post-Stage-41 surface). The Stage 41 additions extend the existing identity-lowering arm to the 11 causal builtins. The guard is `if expr.callee.name in (...12 frame + 12 temporal + 10 modal + 11 causal names = 45 total...) and len(expr.args) == 1: return self._lower_expr(expr.args[0])`.

**Reasoning**: Same Stage 39 F5 / Stage 40 F3 carry-over. IR-only probe (skipping typecheck) reproduces the pattern:

```
Lowerer(parse('fn main() -> i32 { let x: i32 = into_cause(1, 2); 0 }')).lower()
=> NotImplementedError: unknown function 'into_cause' in IR lowering at 1:33; run typecheck first

Lowerer(parse('fn main() -> i32 { let x: i32 = propagate(1, 2, 3); 0 }')).lower()
=> NotImplementedError: unknown function 'propagate' in IR lowering at 1:33; run typecheck first

Lowerer(parse('fn main() -> i32 { let x: i32 = into_cause(); 0 }')).lower()
=> NotImplementedError: unknown function 'into_cause' in IR lowering at 1:33; run typecheck first
```

The opaque catchall diagnostic asserts "run typecheck first" even though typecheck WAS run and emitted errors that the build pipeline ignored. Same Stage 40 F3 conclusion: no active runtime bug; defense-in-depth is what is missing. The 45-name identity-lower set is now large enough that a future refactor relaxing the `len == 1` guard, or reordering the dispatch, would silently drop side-effecting args.

**Why LOW**: same call as Stage 39 F5 and Stage 40 F3. Wrong-arity safety net at typecheck IS active (verified by the test suite's wrong-arity tests). The IR-arm is the second-of-two gates; today it fails closed-but-misleading, not fail-open.

**Remediation**: add an explicit assertion inside the identity arm at 2034, or hoist the 45-name tuple to a module-level frozenset and add the assertion at the same time. ~3 LOC. Stage 40 gate-2 O2 recommended this hoist; deferred each stage. Mechanically:

```py
IDENTITY_LOWER_NAMES = frozenset({...45 names...})
...
if (isinstance(expr.callee, A.Name)
        and expr.callee.name in IDENTITY_LOWER_NAMES):
    assert len(expr.args) == 1, (
        f"{expr.callee.name} arity guard violated; "
        f"typecheck should have rejected wrong-arity calls"
    )
    return self._lower_expr(expr.args[0])
```

## OUT OF SCOPE — observations (no severity)

- **O1** (test-suite gap, family-symmetric carry-over from Stage 38 O1 / Stage 39 O1 / Stage 40 O1). `test_reflection.py` contains parallel entries for `test_dogfood_10_memory_tiers` through `test_dogfood_12_temporal_lifecycle` but NEITHER `test_dogfood_13_modal_lifecycle` (the Stage 40 O1 that should have been closed before Stage 40 CLOSED) NOR `test_dogfood_14_causal_lifecycle` (the new Stage 41 gap). The silent-failure framing: a regression that breaks `dogfood_14_causal_lifecycle.hx` end-to-end (e.g., a `run.py` DEMOS-dict key collision on `"causal"`, a `@pure` decorator interaction across the two helper fns, a witness-arithmetic regression in the `Known<Cause<i32>>` cross-stack composition probe) would not be caught by any Stage 41 test — the dogfood breaks unnoticed because nothing references it from CI. Fix is ~7 LOC × 2 dogfoods = 14 LOC of test_reflection.py additions. Worth closing both simultaneously since Stage 40's O1 has now been outstanding for two stages.

- **O2** (dispatch perf, family carry-over from Stage 38 O2 / Stage 39 O2 / Stage 40 O2). The three dispatch dicts at `typecheck.py:3745-3767` (`_causal_intro`, `_causal_elim_kind`, `_causal_upgrade_hint`) and the transition dict at `3838-3842` (`_causal_transitions`) rebuild on every Call-expression typecheck visit. Negligible cost (4 small dicts) but pattern-symmetric to Stage 38/39/40 O2; same hoisting opportunity. The identity-lowering tuple at `lower_ast.py:2008-2033` is similarly per-Call-visit rebuilt and has now grown to 45 entries.

- **OBS-Z** (regression-in-shipped-state, NOT Stage-41-Inc-0+1+2+3 specific but worth flagging because it materially affects the as-tested working tree). Commit `4e74244` (Stage 42 OPENS) silently REVERTED commit `246c33f`'s typecheck.py changes — specifically the gate-1 fixes for the F1 inner_is_shadowed parity (the F1 of THIS audit) and the 5 reverse-direction safety-anchored hints for the Stage 41 `_causal_upgrade_hint` map. Verified via blob hash: 7448bf5 → `ffa46be`, 246c33f → `679f8f7`, 6f818e4 (Stage 41 CLOSED) → `679f8f7`, HEAD (4e74244) → `ffa46be` (REVERTED). The Stage 42 commit message says "no new type primitives; only dogfood + ledger" — yet the diff at typecheck.py shows a 43-line REVERSAL of the gate-1 fixes (the same 43 lines that 246c33f added). The test `test_stage41_gate1_f1_inner_is_shadowed_suppresses_launder_cascade` survives at HEAD (the test file was untouched by 4e74244) and now FAILS against the reverted typecheck.py. The reversion is most plausibly a stale-rebase-or-cherry-pick accident — the Stage 42 OPENS author likely operated on a worktree branched from 7448bf5 (pre-gate-1-fix) and let git "resolve" the conflict by taking the older base. This is OBS-grade for THIS audit (the audit target is 7448bf5, and the gate-1 fixes that 246c33f added are exactly what F1 here recommends) but is a CRITICAL finding for the working-tree state at HEAD: Stage 41 closure gate-1 fixes are silently undone in the live branch, and the test suite at HEAD is failing 1/23 as a result. A short follow-up commit re-applying 246c33f's typecheck.py changes would restore the working-tree to the Stage-41-CLOSED state; verification: `git diff 246c33f..HEAD -- helixc/frontend/typecheck.py` should be empty post-fix (currently shows a 43-line reverse-diff).

## Summary

TWO HIGH (F1: causal F1 guard missing `inner_is_shadowed` parity — direct cross-stage symmetric carry-over of the Stage 40 gate-3 H1 fix that was in the same file when 7448bf5 was authored; fix is 7 LOC mirror. F2: cross-family modal/causal launder is silent in all 32 direct directions — `Cause -> Known -> Independent` 3-line laundromat compiles to a working ELF binary, materially undermining both Stage 40's "category mistake at compile time" headline and Stage 41's "causal misattribution is caught at the type system" headline; fix requires extending each per-family F1 guard to inspect all families' elim names, ~30 LOC consolidation), ZERO MEDIUM, TWO LOW (causal-on-causal wrapper-stacking is the Stage 38/39/40 family-symmetric L1/F3/F2 carry-over; IR identity-arm has no defense-in-depth assertion against wrong-arity calls — Stage 39 F5 / Stage 40 F3 carry-over now widened to 45 names), THREE OBS (missing `test_dogfood_13` AND `test_dogfood_14` in test_reflection.py — Stage 40 O1 still open + new Stage 41 gap; per-Call dict reallocation — family carry-over; a CRITICAL-grade follow-up observation that Stage 42 OPENS silently reverted Stage 41 closure gate-1 typecheck.py fixes in the live branch).

Gate-1 strictness reminder: per Stage 40 gate-2's "anything below HIGH conf 75 → OBS" calibration, both LOW findings sit at conf 75 exactly — borderline OBS, retained as LOW because both have concrete reproducible behaviors and both are family-symmetric carry-overs that the Stage 40 ledger explicitly retained as LOW rather than collapsing to OBS.

F1 is the textbook gate-1 closure-blocker: the fix has literally already been authored (commit 246c33f, then accidentally reverted by 4e74244). Re-cherry-picking 246c33f's typecheck.py change closes F1 in 7 LOC. F2 is the broader and newer finding — it documents a structural gap that has been latent across Stage 38/39/40 too but became visibly safety-critical at Stage 41 with the Cause→Known→Independent 3-line repro. Closing F2 requires a small cross-family consolidation (~30 LOC) and ~60 LOC of new regression tests; not in the same fix-sweep as F1 but addressable in the gate-1 fix-sweep cycle.

All 12 cross-causal direct launders reject cleanly with named F1 diagnostics; all 16 helper-method probes return expected values (no symmetric helper-coverage gap as Stage 39 H1+H2 had); all 2 AD chain-rule probes produce the correct `(x + x)` identity-chain through causal wrappers and transitions; all 11 causal builtins are registered in `_BUILTIN_NAMES`, `AD_KNOWN_PURE_CALLS`, `_FRAME_IDENTITY_AD_NAMES`, and the IR identity-lowering tuple consistently — no surface inconsistency between passes; `_resolve_type` causal_map correctly resolves all 4 capitalized type names. The Stage 41 surface is structurally clean within the causal family; the silent-failure surface is at the family boundaries (F1: shadow + launder cascade across the inner-call-name discipline; F2: cross-family laundromat across the elim-name-set discipline).

**Verdict**: 2 HIGH + 0 MEDIUM + 2 LOW + 3 OBS — gate-1 NOT CLEAN. F1 has an in-history fix at 246c33f that needs to be re-applied (currently reverted in the live branch per OBS-Z). F2 requires a fresh ~30 LOC consolidation across the 4 per-family F1 guards plus ~60 LOC of regression tests. Closing both lifts gate-1 to 0 HIGH, 0 MEDIUM, 2 LOW, 3 OBS — which clears the CLEAN bar under the Stage 40 gate-2 calibration ONLY IF the two LOW findings are explicitly accepted as deferred or downgraded to OBS by the gate-2 reviewer (matching the Stage 40 gate-2 decision).
