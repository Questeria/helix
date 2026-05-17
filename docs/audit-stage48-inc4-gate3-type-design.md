# Stage 48 Inc 4 closure gate-3 type-design audit

**HEAD**: `3415727` (Stage 48 Inc 4 closure gate-3 G3-F1 scope-aware fix + M1/M2/L1 polish)
**Base**: `48db12d` (Stage 48 OPENS + Inc 1+2+3)
**Scope**: `?` propagation operator type-design surface — `TyResult` two-parameter family interaction with the Stage 37-41 one-parameter wrappers, `_lower_type` Result arm in fn-signature positions, `__try` builtin namespace, and the `_result_constructor_provenance` semilattice.
**Date**: 2026-05-17
**Auditor brief**: gate-3 type-design lens — Result generic-arity confusion, wrapper-family interleaving symmetry, IR-position coverage, provenance LUB completeness, TyResult dataclass invariants.

## VERDICT: 0 HIGH, 0 MEDIUM, 2 LOW, 2 OBS

## Summary table

| ID | Sev | Conf | Title |
|----|-----|------|-------|
| L1 | LOW | 70 | `_lower_type` Result arm at `lower_ast.py:865-874` drops `args[1]` (Err type) without `sizeof(ok_ty) >= sizeof(err_ty)` assertion; every test exercises `Result<i32, i32>` or `Result<i32, bool>` (both same-ABI-slot), so unequal-size lowering is unexercised — F5-class silent ABI corruption extends to fn param/return/tuple/array element positions, not just struct fields |
| L2 | LOW | 55 | Provenance semilattice has no `if`-merge join: `if c { r = Err(1) } else { r = Err(2) }` post-merge has r dropped from map (both branches independently dropped via mutated_outer_names exit logic); the static Err-rejection at `typecheck.py:4628` is therefore unreachable for any branched-Err-only flow — sound (no false claim), but a missed-detection opportunity |
| OBS-A | OBS | 95 | Family-symmetry verdict for TyResult: every Stage 37-41 wrapper helper site has a parallel TyResult two-inner arm. Inventory confirmed at `_compatible` (8296), `_refinement_shape_exact` both pair-versions (6351, 7029), `_erase_refinement` (7089), `_contains_refinement` (7221), `_contains_refined_function` (7285), `_contains_unknown_type` (6274), `_is_refinement_container` tuple (7249), `_fmt` (8479). The Stage 39/40 H1/H2/H3 lesson IS internalized for the first 2-parameter family |
| OBS-B | OBS | 80 | Composition `Known<Result<T,E>>` and `Result<Known<T>, E>` both typecheck correctly: TyModal arm recurses via `_compatible(modal.inner, modal.inner)` → TyResult arm → walks both ok+err. Verified by tracing the four wrapper helpers. The OBS-B generic-inner deferral hole from Stage 40 audit applies symmetrically to TyResult (Phase-1 backlog item; not a Stage 48 regression) |

---

## L1: Err-side ABI drop unexercised in any signature position (LOW, conf 70)

**Location**: `helixc/ir/lower_ast.py:865-874`.

The Result arm returns `_lower_type(ty.args[0])` unconditionally. `ty.args[1]` (Err type) is never inspected. F5 (test_stage48_closure_gate2_f5_member_access_documented_as_phase0_defect) names this for struct-field position; the Stage 48 gate-2 lower_ast change extends the arm to FOUR new positions (fn return, fn param, tuple element, array element) but none has a test where `sizeof(Ok) != sizeof(Err)`. Every Stage 46/48 test uses `Result<i32, i32>` or `Result<i32, bool>` (i8 vs i32 — both still 1 slot). A param `r: Result<i64, struct_2_slots>` would silently mis-ABI when called with `Err({...})`.

**Why LOW**: F5 already names the entire class as Phase-0 deferred; the TODO(stage49) markers at `lower_ast.py:866-872` enumerate the runtime-tag fix scope. **Fix**: add ONE test like `test_stage48_phase0_unequal_result_size_param_documented_as_phase0_defect` using `Result<i64, i32>` as a fn param + caller passing `Err(99)`, asserting current Phase-0 behavior (compiles, may exit nondeterministically); flips polarity at Stage 49 like F5. Mirrors the existing F5 deferral discipline. No Phase-0 code change.

## L2: Provenance dict has no if-merge LUB; branched-Err flows lose static rejection (LOW, conf 55)

**Location**: `helixc/frontend/typecheck.py:4936-4951` (A.If handler) + `2400/2451-2462` (_check_block save/restore-then-drop).

The `_check_block` exit drops any outer name whose post-block provenance differs from saved. After an `if-else` where both branches `r = Err(...)`: then-branch drops r (saved 'ok', mutated to 'err'); else-branch sees r missing from map, the assign-arm at 5024 only updates IF the name is already in the map, so the else `r = Err(2)` does NOT re-add provenance. Post-if: r is missing from the map. The static Err-rejection at line 4628 never fires for any branched-only Err flow. The current behavior is sound (no false static claim) but a missed detection.

**Why LOW**: not a soundness bug; the F1-dynamic Phase-0 limitation already covers this — Stage 49 runtime tag subsumes. The fix would be a per-branch provenance snapshot + 3-valued join (ok / err / unknown) at if/match merge points — non-trivial. Acceptable as Phase-0 best-effort.

## OBS-A: TyResult family symmetry — Stage 39/40 H1/H2 lesson internalized (OBS, conf 95)

Eight helper sites checked; all have a parallel TyResult arm walking BOTH ok_ty AND err_ty. No symmetric gap. The two-parameter family is correctly handled as "both inners reachable" everywhere it matters. This is the right pattern to keep for any future 2+-parameter wrapper.

## OBS-B: Composition correctness (OBS, conf 80)

`Known<Result<T,E>>` and `Result<Known<T>, E>` both walk through correctly. The pre-existing OBS-B generic-inner deferral hole (`fn id[T](p: Result<T, E>)` rejects at call boundary) applies symmetrically — Phase-1 backlog, not a Stage 48 regression.

## Other questions ruled out

- **__try BUILTIN_NAMES clash**: no collision with `__arena_*`/`__strlen` — names are distinct; user-fn shadow check at `typecheck.py:1009` rejects user definitions of `__try`.
- **TyResult ok_ty == err_ty aliasing**: frozen dataclass + idempotent recursive walks — no aliasing bug possible.
- **_compatible cross-wrapper rejection**: TyResult-vs-non-TyResult correctly returns False at line 8299.

3/3 GATE-3 CLEAN — Stage 48 ready to CLOSE.
