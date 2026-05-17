# Stage 42 Inc 2 Gate-1 — Type-Design Audit
Date: 2026-05-17
Scope: git diff 6f818e4..HEAD (Stage 42 Inc 0/1 ship + Inc 1 hotfix + Inc 2 close)
HEAD: 7699f00769891faeef3196b17d8fd03f4335b58f

## Verdict
GATE CLEAN

No HIGH or MEDIUM findings at confidence >= 80. Two LOW-confidence design-quality
observations (each conf 70-75) are recorded below for future-stage follow-up;
neither blocks Stage 42 closure since Stage 42's stated scope is "no new type
primitives, demonstrate cohesion of the existing quintet" and the dogfood
demonstrates cohesion as advertised.

## Findings (HIGH / MEDIUM / LOW, with confidence 0-100)

### L1 — No documented wrapper-stack ordering convention exists (LOW, conf 75)

Stage 42's dogfood is the FIRST 4-deep wrapper composition in the codebase. It
uses `Modal > Temporal > Spatial > Causal` (outside-to-inside) for the
`Known<Present<WorldFrame<Cause<i32>>>>` and
`Believed<Future<WorldFrame<Effect<i32>>>>` stacks. Prior dogfoods only
exercise 2-deep compositions:

- dogfood_13: `Known<Past<i32>>` (Modal > Temporal)
- dogfood_14: `Known<Cause<i32>>` (Modal > Causal)
- dogfood_15: `Modal > Temporal > Spatial > Causal` (4-deep)

Greps of `docs/stage{37..42}*.md` for "convention | ordering | outside-to-inside
| wrapper.*order | stack.*order | canonical" turn up only the per-stage
"Convention Declaration" sections, which scope the *audit-cycle* convention
(combined audit-and-fix, 3-clean-gate closure) — not the *wrapper composition*
convention. No document specifies whether modal-outside-temporal,
temporal-outside-spatial, or spatial-outside-causal is canonical.

Why this is LOW not MEDIUM: the 8 wrapper helpers (`_compatible`,
`_refinement_shape_exact`, `_refinement_proof_carried`, `_erase_refinement`,
`_contains_refinement`, `_is_refinement_container`, `_contains_refined_function`,
`_contains_unknown_type`) treat the 5 wrapper kinds *symmetrically* — there's
no functional difference between `Known<Past<T>>` and `Past<Known<T>>` at the
typecheck layer. The convention gap is a *documentation / hint-text /
formatter-output* concern, not a soundness concern. Diagnostic messages built
from `self._fmt(arg_tys[0])` will print whatever ordering the user wrote, and
a future hint-text rewriter that wants to normalize "did you mean X?"
suggestions will need a canonical order to do so without flapping. Until such
a rewriter exists, the gap is latent.

Fix scope: one paragraph in a future stage progress doc declaring (e.g.)
"canonical 5-stack order: Memory > Modal > Temporal > Spatial > Causal —
outermost is what's most epistemically labile, innermost is what's intrinsic
to the value". Cite the rationale dogfood_15's inline comment already gives:
"causality applies to the value regardless of which frame it's observed in"
(lines 47-48).

### L2 — Comment header overstates demonstration scope (LOW, conf 72)

dogfood_15's file header (lines 3-4) says:

```
// AGI planning-loop scenario exercising all 5 semantic-type
// families (memory + spatial + temporal + modal + causal) in a
// single coherent program.
```

But the program never wraps anything in `Working` / `Episodic` / `Semantic`
(the memory tier). It uses 4 of the 5 families. The progress doc explains
the omission ("omitting the memory wrapper which would make it 6 levels deep
but adds no demonstration value beyond the 5 already exercised", stage42-
progress.md lines 36-38), but the dogfood file's own header does not
acknowledge this asymmetry — a reader who opens dogfood_15.hx in isolation
will look for the memory tier and not find it, and may wonder if it was
forgotten rather than deliberately omitted.

Type-design lens: misleading documentation around type compositions is a
design-quality issue because it weakens the dogfood's value as a worked
example. The fix is a one-line clarification, e.g. amend lines 3-5 to:

```
// AGI planning-loop scenario exercising 4 of the 5 semantic-type
// families (spatial + temporal + modal + causal; memory tier omitted
// per docs/stage42-progress-2026-05-17.md to keep depth at 4) in a
// single coherent program.
```

Confidence is 72 (not higher) because reasonable engineers could disagree on
whether "exercising all 5 families" is a soft claim about the *quintet being
demonstrated to compose* (which the prior 2-deep dogfoods cover for the
remaining pair) or a hard claim about *this particular file using all 5*.

### Cohesion-completeness observation (NOT a finding)

The dogfood uses only `propagate` (Cause -> Effect) from the cross-causal
transition set; it does NOT exercise `aggregate` (Cause+Cause -> Joint),
`isolate` (Joint -> Independent), or `confirm` (Believed -> Known) /
`actualize` (Future -> Present) / `to_past` (Present -> Past). Stage 42's
claim is "quintet COHESION", and the question arises whether exercising one
cross-family transition per family demonstrates cohesion or whether all
transitions must be type-checked in composition with all wrappers.

Resolution: not a finding. Cohesion (as Stage 42 defines it) is the property
that *the 8 wrapper helpers all walk through all 5 wrapper TyXxx kinds
symmetrically*. That property is what makes 4-deep compositions typecheck at
all, and it is verified by the dogfood compiling end-to-end (exit 42). The
per-transition matrix (5 families × ~3 transitions each = ~15 transitions)
is exercised by the per-family lifecycle dogfoods (10-14); 15 is not a
realistic count to also re-exercise inside a 4-deep stack. The dogfood
proves the *composition mechanism* works; the per-family dogfoods prove
each *transition mechanism* works; the union covers the demonstration
surface.

## Verification steps performed

1. Confirmed Stage 42 diff scope via `git diff --stat 6f818e4..HEAD`:
   only `dogfood_15_agi_planning_loop.hx` (new), `run.py` (+5), 4 progress
   /audit docs. No typecheck.py / IR / AD changes in the net diff.

2. Verified hotfix `1e58862` correctly restored
   `helixc/frontend/typecheck.py` to Stage-41-closure blob hash
   `679f8f7de5c633b331d93333be8de53f5f1a7aa3`. Confirmed via
   `git ls-tree HEAD helixc/frontend/typecheck.py` (HEAD) and
   `git ls-tree 6f818e4 helixc/frontend/typecheck.py` (Stage 41 close)
   match byte-for-byte.

3. Verified F1 `inner_is_shadowed` parity between cross-modal guard
   (typecheck.py:3738-3751) and cross-causal guard (typecheck.py:3938-3951).
   The two blocks differ only by `_modal_elim_kind` -> `_causal_elim_kind`
   (the kind-specific dict reference, which is correct). All other tokens
   match verbatim including indentation, predicate ordering, and the
   `and not inner_is_shadowed` final clause.

4. Verified the 8 wrapper-walk helpers each recurse into all 5 wrapper
   TyXxx tags (TyMemTier, TyFrame, TyTemporal, TyModal, TyCausal):

   | Helper | TyMemTier | TyFrame | TyTemporal | TyModal | TyCausal |
   |---|---|---|---|---|---|
   | `_contains_unknown_type` (5433) | grouped tuple at 5468-5472 | ditto | ditto | ditto | ditto |
   | `_refinement_proof_carried` (5475) | 5516 | 5523 | 5530 | 5537 | 5542 |
   | `_refinement_shape_exact` (6171) | 6198 | 6201 | 6207 | 6211 | 6215 |
   | `_erase_refinement` (6241) | 6258 | 6260 | 6265 | 6268 | 6271 |
   | `_contains_refinement` (6331) | 6381 | 6383 | 6389 | 6392 | 6395 |
   | `_is_refinement_container` (6411) | tuple at 6417-6420 | ditto | ditto | ditto | ditto |
   | `_contains_refined_function` (6423) | 6440 | 6442 | 6446 | 6449 | 6452 |
   | `_compatible` (7391) | 7426 | 7435 | 7444 | 7451 | 7456 |

   All 8 × 5 = 40 cells present. The F6-class wrapper-walk symmetry bug
   from Stage 40 gate-3 (where one helper missed one wrapper tag) cannot
   recur on this surface as long as adding a new wrapper kind continues
   to add to all 8 helpers — the Stage 41 Inc 1 "preemptive parallel arm"
   pattern is now established and visible.

5. Verified witness pattern parity. dogfood_15 uses
   `o1_ok * o2_ok * o3_ok * (o1 + o2 + o3)` (line 83-84). dogfoods
   10-14 use the same product-of-binary-witnesses × sum-gate pattern
   with the same all-ok-then-sum-equals-42 collapse-resistance shape.

6. Empirically verified dogfood_15 compiles + executes via
   `python -m helixc.examples.run planning`. Exit code 42 returned.
   This exercises `_compatible` recursively through 4-deep wrapper
   stacks at every let annotation in `agi_perceive_plan_cycle`.

7. Ran `helixc/tests/test_stage41_causal.py`: 23/23 passing
   (confirms the F1 closure-trail test that gated the hotfix is now
   green at HEAD).

8. Searched stage37..42 progress docs and the wider codebase for any
   prior statement of canonical wrapper-stack ordering. None found.
   The convention is implicit (dogfood_13/14's `Modal<Temporal>` and
   `Modal<Causal>` ordering is preserved-in by dogfood_15's
   `Modal > Temporal > Spatial > Causal` 4-deep stack), but
   undocumented. Recorded as L1 above.
