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

### Cycle 1: NOT YET STARTED
Will fire on next cron tick after v1.0 announcement.

| Batch | Auditor 1 | Auditor 2 | Auditor 3 | Auditor 4 | Auditor 5 | Verdict |
|-------|-----------|-----------|-----------|-----------|-----------|---------|
| FE    | TBD       | TBD       | TBD       | TBD       | TBD       | TBD     |
| IR    | TBD       | TBD       | TBD       | TBD       | TBD       | TBD     |
| BE    | TBD       | TBD       | TBD       | TBD       | TBD       | TBD     |
| RT    | TBD       | TBD       | TBD       | TBD       | TBD       | TBD     |
| TEST  | TBD       | TBD       | TBD       | TBD       | TBD       | TBD     |

### Cycle 2: pending
### Cycle 3: pending
### Cycle 4: pending
### Cycle 5: pending

## Clean-streak counter

**Current: 0 / 5**

## Findings log (cumulative)

(populated as cycles run)

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
