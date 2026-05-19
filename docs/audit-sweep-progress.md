# Helix v1.0 + v0 5-Clean-Gate Audit Sweep — Progress Log

Per user directive 2026-05-18: after v1.0 ships (Stage 108), run full
audit sweeps on EVERYTHING in v1.0 + v0 and fix any/all issues until
**5 audit cycles in a row come clean**.

## Protocol

### Audit cycle composition (per cycle, 5 auditors)
1. `pr-review-toolkit:silent-failure-hunter` — silent miscompiles,
   missing error returns, swallowed exceptions
2. `pr-review-toolkit:type-design-analyzer` — type-system design debt,
   inconsistent invariants
3. `pr-review-toolkit:code-reviewer` — general code-quality + convention
   violations
4. `pr-review-toolkit:silent-failure-hunter` (depth pass on different
   focus area than cycle pass 1)
5. `feature-dev:code-reviewer` — fresh-eyes cross-check

### "Clean" criterion (per cycle)
- Zero HIGH findings
- Zero MEDIUM findings that survive to MUST-FIX status
- LOW findings + design-debt-residuals logged but don't break the streak

### Stop condition
**5 consecutive cycles return clean.** Counter resets to 0 if any cycle
finds HIGH/MEDIUM that requires fix.

### Batching strategy
Stages batched by subsystem for parallel auditor dispatch:
- **Batch FE** — frontend (parser, typecheck, AST hash, monomorphize)
- **Batch IR** — IR (tir, tile_ir, passes/*)
- **Batch BE** — backends (x86_64, ptx)
- **Batch RT** — runtime (stdlib, bootstrap)
- **Batch TEST** — test infrastructure (scorecards, property runner)

## Scope (~108 stages total)

### Tier 1: Post-burst (already 3-clean per Stage 99 audit) — **34 stages**
Stages 66, 68-73, 75-78, 80-83, 86, 88, 92, 100, 101, 102, 103, 104,
105, 106, 107, 108. Already 3-clean → need 2 more clean cycles to hit 5.

### Tier 2: Pre-burst (never formally 3-clean-gate audited) — **~66 stages**
Stages 1-65, 67, plus Stage 64 Inc 1. Need 5 fresh consecutive clean
cycles.

## Cycle log (filled in as sweep progresses)

### Cycle 1: IN PROGRESS
Started 2026-05-18 post-v1.0 release (commit `00812ee`).

#### Batch FE (frontend) — 2026-05-18

Auditors dispatched in parallel:
1. silent-failure-hunter on parser.py + autodiff_reverse.py
2. type-design-analyzer on typecheck.py 1-6000
3. code-reviewer on typecheck.py 6000-12316
4. silent-failure-hunter (depth) on autodiff.py + autodiff_cli.py +
   grad_pass.py
5. feature-dev:code-reviewer on ast_hash + monomorphize + struct_mono
   + flatten_modules

**Total findings**: 7 HIGH + 9 MEDIUM = 16

**HIGH findings (must-fix before counter advances)**:
- HIGH-1 (Auditor 1) — parser.py `_parse_trait_decl` silent token
  swallow. **FIXED** in Cycle 1 fix batch 1.
- HIGH-1 (Auditor 2) — `_ALL_WRAPPER_CLS_NAMES` missing 5 AGI-quartet
  wrappers (TyMemTier/TyFrame/TyTemporal/TyModal/TyCausal). _is_copy_
  struct_ty + _strip_wrapper_chain under-count. **FIXED** in Cycle 1
  fix batch 1.
- HIGH-2 (Auditor 2) — `ModalKind` Literal vs hand-coded assert
  tuple — divergent source of truth. **DEFERRED to batch 2**.
- HIGH-3 (Auditor 2) — `TyPrim("size_N")` overloads primitive
  namespace with synthetic sizes. **DEFERRED to batch 2** (larger
  refactor, not blocking).
- HIGH-1 (Auditor 4) — `_args_are_unroll_safe` in autodiff.py admits
  any literal arg, not the recursion-driving one. **DEFERRED to
  batch 2**.
- HIGH-2 (Auditor 4) — `_inline_user_calls` doesn't traverse
  Modify/Quote/Splice/TileLit/Path. **DEFERRED to batch 2**.
- HIGH-3 (Auditor 4) — `_with_float_literal_suffix` misses dict/set
  containers (latent). **DEFERRED to batch 2** (latent, not
  triggered today).
- HIGH-4 (Auditor 4) — `_generate_grad_rev_all_fn` blanket
  `except Exception` silently falls back to scalar-only. **DEFERRED
  to batch 2**.
- HIGH-5 (Auditor 4) — `_reject_unsupported_grad_signature` blanket
  `except (ValueError, Exception)` parity issue. **DEFERRED to
  batch 2**.
- HIGH-1 (Auditor 5) — monomorphize.py `_walk_subst_expr` missing
  TileLit arm. Generic `tile<T, ...>` body silently unsubstituted.
  **FIXED** in Cycle 1 fix batch 1.

**MEDIUM findings (logged, will fix as time permits)**:
- Auditor 1 MED-1: parser try/finally invites future state corruption
- Auditor 2 MED-1: `Place.parts: tuple` untyped variant schema
- Auditor 2 MED-2: Magic-string borrow states (no Literal alias)
- Auditor 2 MED-3: `TySize` + `TyPrim("size_N")` dual representation
- Auditor 3 MED-1: `consolidate()` / `recall()` return arg type on
  error (should return TyUnknown for cascade-suppression parity)
- Auditor 3 MED-2: Causal launder check is syntactic-only while
  modal is unified (defect class parity)
- Auditor 3 MED-3: Dead branch in `_eval_refinement_predicate`
- Auditor 3 MED-4: `_compare_scalar` swallows unknown op
- Auditor 4 MED-1: `__powi` non-literal n silent zero derivative
- Auditor 4 MED-2: `_stage54_min_max_chain_rule` None handling
- Auditor 4 MED-3: AD unroll guard no warn at max_unroll
- Auditor 5 MED-1: flatten_modules `_rewrite_calls` guard mismatch

#### Cycle 1 fix batch 1 — 3 HIGH fixes shipped 2026-05-18

- typecheck.py: extend `_ALL_WRAPPER_CLS_NAMES` from 13 → 18 (add
  TyMemTier/TyFrame/TyTemporal/TyModal/TyCausal). `_is_copy_struct_ty`
  + `_strip_wrapper_chain` now correctly walk through AGI-quartet
  wrappers.
- parser.py: `_parse_trait_decl` non-fn token now raises ParseError
  instead of silent swallow.
- monomorphize.py: `_walk_subst_expr` now has TileLit arm so
  `tile<T, ...>` in generic fn body gets dtype + memspace +
  shape substituted.
- 2 new fix-verification tests (cycle1_high1_is_copy +
  cycle1_high1_parser_trait_swallow). Stage 100 hoist test updated
  for 18-entry table.
- 586 pins GREEN after batch 1.

**Batch FE verdict (so far)**: NOT CLEAN. 7 HIGH remaining (5 AD-
specific, 2 design-debt). Will continue in subsequent ticks.

| Batch | Auditor 1 | Auditor 2 | Auditor 3 | Auditor 4 | Auditor 5 | Verdict |
|-------|-----------|-----------|-----------|-----------|-----------|---------|
| FE    | 1H+1M     | 3H+3M     | 0H+4M     | 5H+3M     | 1H+1M     | NOT CLEAN (7H remain) |
| IR    | TBD       | TBD       | TBD       | TBD       | TBD       | TBD     |
| BE    | TBD       | TBD       | TBD       | TBD       | TBD       | TBD     |
| RT    | TBD       | TBD       | TBD       | TBD       | TBD       | TBD     |
| TEST  | TBD       | TBD       | TBD       | TBD       | TBD       | TBD     |

### Cycle 2: pending
### Cycle 3: pending
### Cycle 4: pending
### Cycle 5: pending

## Clean-streak counter

**Current: 0 / 5** (Batch FE has 7 HIGH remaining; counter cannot
advance until ALL HIGH fixed AND re-audit returns clean.)

## Findings log (cumulative)

See Cycle 1 / Batch FE section above for the 7 HIGH + 9 MEDIUM
findings from Auditors 1-5. Fix-shipped count so far: 3 HIGH +
0 MEDIUM. Remaining: 4 HIGH + 9 MEDIUM in frontend.

## Honest forecast

- Cycle 1 across ~108 stages will likely surface 10-30 findings (years
  of accumulated debt + 66 stages never seen formal audit).
- Each fix cycle resets the counter — realistically **multi-week work**,
  not multi-tick.
- Subagent budget: each batch-audit = 5 auditor dispatches.
  Cycle = ~5 batches × 5 auditors = 25 dispatches. Full sweep
  (5 cycles minimum, more if findings reset counter) = 125+
  dispatches. Could be 500+ if cycle 1 surfaces a lot.
- Per user: "Do not move on to v2 until I say so." → cron loop after
  v1.0 release stays on the audit sweep, not v2.0 work.
