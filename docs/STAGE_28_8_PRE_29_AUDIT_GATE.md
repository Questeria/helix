# Stage 28.8: Pre-29 Audit Gate

**Inserted**: 2026-05-10 per user directive.
**Rationale**: Run 5 consecutive clean audit cycles BEFORE dropping the helixc-Python reference (Stage 29). Bugs found during this gate get fixed against the gold-standard Python reference while it still exists. Without this stage, Stage 30 would have to distinguish "bug in self-host" from "bug in dropped Python" — impossible.

## Sequence

| Stage | Purpose | Reference exists? |
|-------|---------|------------------|
| 28.8 (NEW) | 5 clean audit cycles | YES — helixc-Python is the gold standard |
| 29 | Byte-identical verification | YES — used as oracle |
| 30 | 5 clean audit cycles on self-host alone | NO — Python is gone |

## Audit cycle definition

One cycle:
1. Spawn 3 specialist subagents in parallel:
   - `pr-review-toolkit:silent-failure-hunter` — silent corruption windows
   - `pr-review-toolkit:type-design-analyzer` — type soundness gaps
   - `feature-dev:code-reviewer` — general bug review
2. Each writes findings to `docs/audit-stage28-8-cycleN.md` (N = cycle number)
3. Review findings:
   - If **zero new findings** across all 3 audits → counter `count++`
   - If any new findings → fix them all, reset `count = 0`, run another cycle
4. Repeat until `count == 5`

## Initial state

Going in with **21 known open findings** from the prior 3 audits (docs/audit-stage{5-6,7-8,9-16}-*.md):

- From `audit-stage5-6-aggregates.md`: F2, F4, F9, F10, F11, F12, F13 (7 open)
- From `audit-stage7-8-typesystem.md`: F4, F7, F9, F10, F11, F12 (6 open)
- From `audit-stage9-16-codegen.md`: MEDIUM-2, MEDIUM-3, LOW-1, LOW-2, LOW-3 (5 open) + 3 untracked = 8 open

These count as **existing-finding cleanup** — they must be resolved before the cycle count starts incrementing.

## Coverage scope per cycle

Each audit cycle scopes to:
- **All Helix source**: `helixc/bootstrap/*.hx`, `helixc/frontend/*.py`, `helixc/ir/*.py`, `helixc/backend/*.py`, `helixc/stdlib/*.hx`
- **All stages 1-28.7** for regression coverage
- **Cross-stage interactions** (e.g., closure + reflection, generic + trait, AD + tile)

## Success criteria

5 consecutive cycles with **zero new findings of any severity** — CRITICAL, HIGH, MEDIUM, and LOW all count. **Per user directive 2026-05-10**: "For the gates to be clear there cannot be even medium or low issues, there has to be no issues at all." The earlier "MEDIUM/LOW may persist with deferred-to-v0.2 markers" relaxation is explicitly REVOKED. Every finding must be addressed in-cycle.

## Time budget

Indefinite. Don't rush — the value of this gate is rigor, not speed.

## Outputs

- `docs/audit-stage28-8-cycle{1..5}.md` — one findings doc per cycle
- `docs/STAGE_28_8_LOG.md` — running log of cycle outcomes (clean count, fixes applied)
