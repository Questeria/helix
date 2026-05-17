# Stage 48 Inc 4 closure gate-4 type-design audit (verification)

**HEAD**: `3415727` (Stage 48 Inc 4 closure gate-3 G3-F1 scope-aware fix + M1/M2/L1 polish)
**Base**: `48db12d` (Stage 48 OPENS + Inc 1+2+3)
**Scope**: gate-4 verification re-audit of the type-design surface against the gate-3 patch. Specific focus per audit brief: (1) `_result_constructor_provenance` + `_result_let_block_scopes` lifecycle parity; (2) TyResult composition with the Stage 37-41 wrapper quintet; (3) `?` operator interaction with `unwrap_*`/`is_*` builtins; (4) diagnostic-style consistency with Stage 40/41 surface; (5) `TODO(stage49)` marker coverage.
**Date**: 2026-05-17
**Method**: read-only. Code review of `helixc/frontend/typecheck.py`, `helixc/ir/lower_ast.py` + targeted Python repro probes via `python -c` against the live compiler. Three runtime repros confirmed observed structural behavior.

## VERDICT: NOT CLEAN — 1 HIGH, 1 MEDIUM-HIGH, 2 MEDIUM, 2 LOW

## Summary table

| ID | Sev | Conf | Title |
|----|-----|------|-------|
| G4-H1 | HIGH | 95 | Stage 48 Result-arm in `_lower_type` recurses into Ok inner without the wrapper-quintet arms — `Result<Known<i32>, Err>` and friends pass typecheck but raise `NotImplementedError` at IR lowering. Composition family-break exposed (and partially introduced) by Stage 48's fn-return Result arm. |
| G4-H2 | MEDIUM-HIGH | 85 | Provenance lifecycle asymmetry: gate-2/gate-3 snapshot-restore lives ONLY in `_check_block`, but `Assign` mutations of the provenance dict happen in `_check_expr`. Expression-form `if`/`match` arms (`match b { true => r = Err(99), false => () }`) permanently mutate the outer dict, producing a false-reject on a post-construct `?`. Block-form arms are correct. Sound (no silent miscompile) but a structural lifecycle break — the snapshot/restore boundary is misplaced. |
| G4-M1 | MEDIUM | 80 | `_result_let_block_scopes` has no `check()`-entry clear (parallel to dict's `typecheck.py:677`) and no `_check_fn`-entry clear (parallel to dict's `typecheck.py:2274`). Push/pop balance in `_check_block` makes this a latent defense-in-depth gap rather than an observable defect — but the audit brief explicitly flagged "fn-entry-clear parity" as a HIGH trigger if divergent. |
| G4-M2 | MEDIUM | 75 | Stewardship comment block at `typecheck.py:599-619` lists 6 mutation sites but omits the 3 consumer (read) sites at `typecheck.py:4459, 4630, 5024` — the unwrap_ok/unwrap_err/__try arms that READ the provenance dict. A future refactor renaming the dict would miss those 3 sites if relying on the comment-listed inventory. |
| G4-L1 | LOW | 70 | `_result_let_block_scopes[-1].add(stmt.name)` at `typecheck.py:2517-2518` runs for every `Let` regardless of whether the binding is Result-typed. Harmless (the set is only consulted via `n not in inner_lets` and outer-key intersection), but conceptually couples the set's domain to "all locals" rather than "Result-tracked locals". |
| G4-L2 | LOW | 65 | `TODO(stage49)` markers are present at `lower_ast.py:866, 2097` and `test_stage48_try.py:404` but ABSENT at the typecheck-side provenance consumer sites (`typecheck.py:4459-4485, 4628-4646, 5021-5035`) and the snapshot/restore machinery (`typecheck.py:2400-2462`). All become obsolete or substantially restructured when the runtime tag lands. Inconsistent marker discipline. |

---

## G4-H1: Wrapper-quintet composition break at `_lower_type` Result arm (HIGH, conf 95)

**Location**: `helixc/ir/lower_ast.py:865-874` (Stage 48 Inc 3-added Result arm in `_lower_type`).

**Observed structural issue**: Stage 48 Inc 3 added a `Result<T,E>` arm to `_lower_type` to support `?` requiring a Result-returning fn signature. The arm returns `self._lower_type(ty.args[0])` — identity-lowering to the Ok inner. But `_lower_type`'s wrapper-quintet (`Known`/`Past`/`Cause`/`WorldFrame`/`Believed`/etc.) arms are NON-EXISTENT — the comment at lines 855-862 explicitly states those families are "handled by struct-mono ... and only surface in let-RHS expression positions". Stage 48's recursion through the Result arm into the Ok inner reaches the wrapper-quintet TyGeneric AST node and falls through to the `raise NotImplementedError` at line 880-882.

Live repro (run against HEAD `3415727`):

```python
src = """
fn ret_known() -> Result<Known<i32>, i32> {
    let k: Known<i32> = into_known(42);
    Ok(k)
}
fn main() -> i32 { from_known(unwrap_ok(ret_known())) }
"""
typecheck(parse(src)) == []          # PASSES — typecheck clean
lower(parse(src))                    # RAISES NotImplementedError:
                                     #   "unresolved generic type Known<...>
                                     #    reached IR lowering"
```

Same break in the `?` path:

```python
fn inner() -> Result<Known<i32>, i32> { Ok(into_known(42)) }
fn outer() -> Result<Known<i32>, i32> {
    let x = inner()?;                # the `?` adds nothing to lowering,
    Ok(x)                            # but the FN RETURN TYPE forces
}                                    # _lower_type into the quintet hole
```

Both repros: typecheck clean, lower raises.

**Why this is a structural-cohesion HIGH and not deferred polish**:

- Audit point 2 explicitly asks: "Stage 42 dogfood proves 4-deep wrapper stacks work; Stage 48 should preserve this — verify `Result<Known<Past<i32>>>`, `Known<Result<i32, Err>>` etc. are typecheck-sound, even if not yet test-pinned."
- Typecheck-sound: YES (`TyResult` + `TyModal` + `TyTemporal` all recurse through `_compatible` at `typecheck.py:8296`).
- IR-sound: **NO**. The asymmetry between typecheck (accepts) and lowering (raises) is the exact pattern Stage 39/40 H1/H2 audits flagged as silent-miscompile-adjacent.
- Stage 48 partially INTRODUCED this: pre-Stage-48 the Result arm at `lower_ast.py:865-874` did not exist, so the only way to reach Known<...> at lowering was via a let-RHS-position expression, where the expression-lowerer dispatches through the constructor/accessor identity arms that DO handle wrappers. Stage 48 added the type-position arm and immediately exposed the quintet hole.
- Gate-3 type-design lane (audit-stage48-inc4-gate3-type-design.md OBS-B) noted "Composition `Known<Result<T,E>>` and `Result<Known<T>, E>` both typecheck correctly" but did not exercise the IR-lowering side. Gate-3 verdict was OBS, not HIGH, because the audit lens stayed at the typecheck layer. This gate-4 verification finds the asymmetry one layer down.

**Recommended fix or design alternative** (pick one):

1. *Minimal, sound, narrowing*: tighten the typecheck Result-in-fn-signature check at `typecheck.py:1352-1363` to REJECT a Result whose Ok or Err side is a wrapper-quintet AST node (TyGeneric with base in {Known, Believed, Past, Future, Eternal, WorldFrame, RobotFrame, CameraFrame, Cause, Effect, Joint, Independent}). Emit a diagnostic naming the offending wrapper and pointing at the Stage 49+ wrapper-in-Result lowering work. This re-aligns typecheck with what lowering can actually consume. ~15 lines, no IR change.

2. *Extend lowering*: add wrapper-quintet identity arms to `_lower_type` (each one returns `self._lower_type(inner)`). ~30 lines, but couples the wrapper families to IR earlier than `struct_mono` would, creating a parallel-path-to-struct-mono risk (the two paths might diverge over time).

3. *Defer*: accept this as a Stage-48-known-defect deferred to Stage 49, but add a dedicated pinning test `test_stage48_phase0_result_of_wrapper_documented_as_phase0_defect` asserting current `pytest.raises(NotImplementedError)` behavior so a future regression surfaces the right delta. Same discipline as the F5 deferral pattern. ~20 lines.

Recommendation: **Option 1** today (typecheck rejection) — converts a silent IR-pass-only crash into a user-facing diagnostic that names the constraint. Option 3 alone is insufficient: the current "typecheck accepts → lowering raises" asymmetry violates the Helix invariant that typecheck-clean implies lowering-clean.

---

## G4-H2: Provenance snapshot/restore boundary misplaced — expression-form arms bypass it (MEDIUM-HIGH, conf 85)

**Location**: `helixc/frontend/typecheck.py:2400, 2451-2462` (`_check_block` snapshot/restore) + `5021-5035` (`Assign` arm provenance mutation) + `4936-4978` (`If`/`Match` dispatch through `_check_expr`).

**Observed structural issue**: gate-2 F1 + gate-3 G3-F1 build a snapshot/restore boundary inside `_check_block` so inner-block let-shadows and inner-block assigns don't pollute outer scope. But the provenance dict is MUTATED inside `_check_expr` (the `Assign` arm at line 5021-5035). `If`/`Match` arm bodies that are EXPRESSION-form (not `Block`-form) dispatch through `_check_expr` directly — no `_check_block`, no snapshot, no restore.

Live repro (run against HEAD `3415727`):

```python
fn helper(b: bool) -> Result<i32, i32> {
    let mut r: Result<i32, i32> = Ok(7);
    match b {
        true  => r = Err(99),   # expression-form arm, no block
        false => (),
    }
    let v: i32 = r?;
    Ok(v)
}
```

Result: typecheck rejects `r?` with the static "constructed via Err()" diagnostic — a FALSE REJECT, because only one of two arms set r to Err. The block-form mirror `true => { r = Err(99); }` is the gate-3 G3-F1c test and correctly accepts (F1-dynamic territory).

**Why this is structural cohesion and not just UX polish**: the GATE-3 G3-F1 fix's design narrative ("snapshot at block entry, scope-disambiguate at block exit") assumed `_check_block` was the sole control-flow boundary that wraps a mutation. The Assign-arm mutation site at line 5021 is invoked by ANY `_check_expr` call — including expression-form match/if arm bodies that don't go through `_check_block`. The lifecycle invariant stated in the dict's stewardship comment ("Invariant: this dict mirrors the names in scope inside the CURRENT function's body, restored across block boundaries") is silently violated when an arm body bypasses the block boundary.

This is one layer deeper than gate-3 G3-F1's defect class: G3-F1 was "inner-block assign + restore put back stale OK"; G4-H2 is "expression-form arm assign + NO restore at all, permanent mutation leaks past the arm and out of the if/match expression itself".

Severity tradeoff: this is **sound** (false-rejects can't cause silent miscompile — gate-3 explicitly preferred false-reject over silent miscompile), so it sits just below the gate-3-style HIGH bar. But the structural pattern (snapshot/restore at the wrong granularity) is a real type-design break that future gate-N escalation could exploit if/when expression-form arms gain richer control flow (e.g. `let` inside an arm body via a future block-expression sugar).

**Recommended fix or design alternative**:

- *Narrow*: at the `If`/`Match` arm-dispatch sites (`typecheck.py:4936-4978`), wrap each arm's `_check_expr` call in a snapshot/restore pair that mirrors `_check_block`'s logic. ~25 lines duplicated, or factor into a `_with_provenance_scope(callable)` helper.
- *Generalize*: move the snapshot/restore to a context manager and use it at every control-flow branching point (Block, If-arm, Match-arm, While body, For body, Loop body). The current `_check_block`-only protection is incomplete coverage of "things that look like a scope".
- *Defer*: same Stage 49 deferral as G3-F1 false-reject acceptance, but ADD a regression test that pins current behavior at the expression-form variant of G3-F1c so the asymmetry is visible in CI. The minimum to avoid silent regression.

Recommendation: deferral with regression-pinning is acceptable for Stage 48 (matches the gate-3 discipline). Stage 49 work should generalize the snapshot/restore to a context manager and apply uniformly across all control-flow branches.

---

## G4-M1: `_result_let_block_scopes` lifecycle parity gap (MEDIUM, conf 80)

**Location**: `helixc/frontend/typecheck.py:628` (declaration) vs `677` (`check()` clear) + `2274` (`_check_fn` clear).

**Observed structural issue**: the two parallel data structures `_result_constructor_provenance` (dict) and `_result_let_block_scopes` (list[set[str]]) have asymmetric lifecycle hygiene:

| Lifecycle event | `_result_constructor_provenance` | `_result_let_block_scopes` |
|---|---|---|
| Declaration (init) | `{}` at line 621 | `[]` at line 628 |
| `check()` entry | Cleared at line 677 | NOT cleared |
| `_check_fn` entry | Cleared at line 2274 | NOT cleared |
| `_check_block` entry | snapshot at line 2400 | push `set()` at line 2401 |
| `_check_block` exit | restore at line 2460 | pop at line 2451 |

Push/pop balance via `_check_block`'s try/finally MAKES the set-stack return to baseline (empty) at each `_check_block`'s exit. If every fn body is wrapped in a `_check_block` (which `_check_fn_body` does at line 2357 — fn body is always `A.Block`), the stack returns to empty at fn-exit. So in practice, the missing clears are defense-in-depth.

But:
- The audit brief explicitly singles out "fn-entry-clear parity, both updated in every let/assign/scope-exit site" as a HIGH trigger if divergent.
- The dict has 2 explicit defense-in-depth clears (check entry + fn entry) precisely BECAUSE the gate-2 M5 silent-failure showed that even a perfectly-paired in-fn lifecycle could leak across fn boundaries if a fn body's exception handling deviated from expectations. The set-stack has neither defense.
- Adding the two clears is 2 lines of code, zero behavioral change today, and forecloses an entire class of future regression.

**Recommended fix**: add `self._result_let_block_scopes = []` at line 677 (check entry) AND line 2274 (`_check_fn` entry), with a comment cross-referencing the gate-3 G3-F1 lineage and gate-2 M5 defense-in-depth rationale. Cost: 2 lines + 1 comment. Risk: zero.

---

## G4-M2: Stewardship comment omits the 3 consumer (read) sites (MEDIUM, conf 75)

**Location**: `helixc/frontend/typecheck.py:599-619` (stewardship comment block listing 6 sites).

**Observed structural issue**: the post-gate-3 stewardship comment at lines 599-619 lists 6 sites:

1. declaration
2. cleared at check() entry
3. cleared at _check_fn entry
4. snapshot + mutate-aware restore across _check_block
5. Let-stmt populates ...
6. Assign-stmt pops ...

All 6 are MUTATION sites. The 3 CONSUMER (read) sites are missing:

- `typecheck.py:4459-4460` — `unwrap_err` / `unwrap_ok` provenance check
- `typecheck.py:4630-4631` — `__try` provenance check (gate-1 F2 fix)
- `typecheck.py:5024` — `Assign`-arm consults dict before mutating

A future refactor (e.g. the gate-2 M2 "factor into a helper" deferral, which gate-2 ledger says will land in Stage 49 with the runtime-tag-aware arm as the 3rd consumer) needs to find ALL READ sites to rewire correctly. The comment-listed inventory is the natural map; omitting consumers leaves the refactor partially-blind.

**Recommended fix**: append to the stewardship comment:

```
Consumer (read) sites — these READ the dict to make typecheck decisions
and must be migrated in lockstep when the dict's shape changes:
  C1. unwrap_ok / unwrap_err static-provenance reject (line 4459-4485)
  C2. __try static-provenance reject (line 4628-4646, gate-1 F2 fix)
  C3. Assign-arm consults before mutating (line 5024-5035)
```

Cost: 6 lines of comment. Zero behavioral change. Closes the gate-2 M2 refactor-prep stewardship gap.

---

## G4-L1: `_result_let_block_scopes[-1].add(stmt.name)` runs for every Let (LOW, conf 70)

**Location**: `helixc/frontend/typecheck.py:2517-2518`.

**Observed structural issue**: the gate-3 G3-F1 fix added `if self._result_let_block_scopes: self._result_let_block_scopes[-1].add(stmt.name)` to the Let-stmt arm. It runs for EVERY let regardless of whether the let binds a Result-typed value. The block-exit logic at line 2454-2458 only consults `inner_lets` via `n not in inner_lets` where `n in saved_provenance`, so non-Result names in the set are inert.

Conceptually this couples the set's domain to "all locals introduced in this block" rather than "Result-tracked locals introduced in this block" — the latter is what the name `_result_let_block_scopes` implies.

**Recommended fix**: either:

- Rename to `_let_block_scopes` to match the actual domain (all let-bound names).
- Or narrow the recording to only Result-typed bindings: add the name only when `stmt.name in self._result_constructor_provenance` after the let-arm has finished its dict update. Saves O(non-Result-lets) set entries per block.

The first option is cleaner — the set's purpose at block-exit is "is this name an inner-shadow?", which is true for any let regardless of type. The name should reflect the actual semantics. Cost: rename, ~3 line touches.

---

## G4-L2: `TODO(stage49)` marker coverage incomplete (LOW, conf 65)

**Location**: gate-3 CR-L1 fix renamed `STAGE49_TODO:` to `TODO(stage49):` at 4 sites. Actual current inventory (verified by grep):

- `lower_ast.py:866` — Result arm in `_lower_type`
- `lower_ast.py:2097` — `__try` in identity tuple
- `tests/test_stage48_try.py:404` — F5 polarity-flip pin
- (gate-3 CR-L1 claimed 4 sites; only 3 found in source post-rename — possibly counting `tests/test_stage48_try.py:410` reference as the 4th)

ABSENT at sites that become obsolete or substantially restructured when the runtime tag lands:

- `typecheck.py:4459-4485` (unwrap_ok/unwrap_err static-provenance reject — runtime tag obsoletes the static path; the diagnostic becomes a soft warning at worst)
- `typecheck.py:4628-4646` (__try static-provenance reject — same: runtime tag enables true Err propagation, the static reject becomes either obsolete or a code-smell lint)
- `typecheck.py:2400-2462` (snapshot/restore machinery — Stage 49 runtime tag eliminates the need for compile-time provenance tracking entirely; this entire 60-line block goes away or becomes a debug-only path)
- `typecheck.py:677, 2274` (clears — same as above, removed when the dict is removed)
- `typecheck.py:5021-5035` (Assign-arm provenance update — same)

The TODO markers are an honest reminder system for "this site MUST be revisited when X lands". Omitting them at the 5 above categories means a future Stage-49 implementer reading typecheck.py won't be reminded which sites are Stage-48-tactical-Phase-0-only vs Stage-46-Phase-0-also-relevant-post-runtime-tag.

**Recommended fix**: add `# TODO(stage49): ...` comments at the 5 categories above with one-line notes on what the runtime tag changes for each site. ~10 comment-only lines. Lineage cross-reference to `docs/stage49-plan-2026-05-17.md` (which already exists per gate-3 stat).

---

## Items audited and ruled CLEAN

- **`?` interaction with `unwrap_*`/`is_*`** (audit point 3): is_ok/is_err are hard-rejected at typecheck for both static-Ok and dynamic operands (`typecheck.py:4691-4734`). The pattern `if is_ok(r) { unwrap_ok(r) }` doesn't compile at all in Phase-0, so it cannot interact with `?`. unwrap_ok consults the same provenance dict as `?` and emits parallel-shape diagnostics; no rule asymmetry.

- **TyResult composition with single-arg wrappers at typecheck** (audit point 2, typecheck layer only): `_compatible` arms at `typecheck.py:8260-8336` recurse through every wrapper correctly. `Result<Known<Past<i32>>, i32>` parses → resolves → typechecks. Verified by trace.

- **Diagnostic-style consistency** (audit point 4): Stage 48 `?` diagnostics use backticks for the operator + `!r` quoting for the name (`` `?` on 'x' requires a Result<T, E> operand, got i32 ``). Stage 40/41 laundering diagnostics use `bn(from_X(...)) launders a X<T>` style (lowercase fn + Capitalized kind). Both use TypeError_(span, hint=) construction and conform to the project's diagnostic invariant (span + kind-specific hint). The stylistic difference (operator-vs-builtin) is justified by the underlying surface (postfix operator vs builtin call) and matches Rust's `?` diagnostic style. Not a defect.

- **Gate-3 scope-aware restore logic correctness** for the block-form cases: traced inner-let-shadow, inner-assign-same-value, inner-assign-different-value, inner-non-Result-shadow, all give the correct dict at block exit. The bug class is only the expression-form arm bypass (G4-H2 above), not the block-form logic itself.

- **fn body always being `A.Block`**: confirmed at `helixc/frontend/ast_nodes.py:451` (`FnDecl.body: "Block"`), so `_check_fn_body`'s `_check_block` wrapping is always the symmetric push/pop point. G4-M1's defense-in-depth recommendation stands but the in-practice push/pop balance is solid.

---

## Cross-gate summary

Gate-1 type-design: CLEAN.
Gate-2 type-design: 1 HIGH (H1 span attribution, deferred) + 4 MEDIUM (deferred / fixed).
Gate-3 type-design: 0 HIGH, 3 MEDIUM (Stage 49-prep), 2 LOW (polish, applied).
**Gate-4 type-design: 1 HIGH (G4-H1, NEW composition break), 1 MEDIUM-HIGH (G4-H2, NEW), 2 MEDIUM (G4-M1, G4-M2), 2 LOW (G4-L1, G4-L2).**

The cascading-defect rhythm from gate-1 → gate-2 → gate-3 continues at gate-4: each verification gate has found a NEW HIGH that the prior gate's fix-set did not cover. The gate-4 HIGH (G4-H1, composition break) is at a different layer (IR-lowering, not typecheck-provenance) so it represents a genuinely new defect class, not a regression of the gate-1/2/3 fixes.

Stage 48's typecheck-side `?` story is sound. The composition break at lowering is a separate-axis structural-cohesion issue that Stage 48 partially introduced and that Stage 49 should address as part of the runtime-tag rollout (which forces a wholesale `_lower_type` Result-arm rewrite anyway, providing a natural moment to add the wrapper-quintet arms or to tighten typecheck rejection at the fn-signature site).

## Recommended gate-4 closure action

VERDICT: **NOT CLEAN**. Gate-4 closure should:

1. Decide on G4-H1 disposition: typecheck rejection (Option 1, recommended) before closing Stage 48, OR explicit defer with pinning test added (Option 3) and `TODO(stage49)` marker at `lower_ast.py:865-874` extended to name the wrapper-quintet recursion path as needing a Stage 49 fix.
2. Accept G4-H2 deferral with a regression-pinning test added (match the gate-3 F5/F6 deferral discipline). Stage 49 generalizes snapshot/restore to a context manager.
3. Apply G4-M1 (2 lines), G4-M2 (6 comment lines), G4-L2 (~10 comment lines) inline — all zero-behavior-change stewardship improvements.
4. Defer G4-L1 (rename) to Stage 49 alongside the gate-2 M2 helper factor-out.
