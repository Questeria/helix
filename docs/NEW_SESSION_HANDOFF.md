# New Session Handoff — 2026-05-08 03:23

**Read this first if you are a fresh Claude Code session opening on this repo.**

The previous session (still active) is running 3 background subagents and 2 external cron loops on this repo. This document tells you what NOT to touch and what you CAN work on without conflict.

## State at handoff

- Branch: `main` at commit `9eced83` (Stage 5 Iter A step 2: struct_table accessors)
- Working tree: clean
- 7 of 8 Stage 4 audit findings RESOLVED
  - #1 (`6aaec01`), #2 (`d1cdb6f`), #3 (`7f9db80`), #4 (`6aaec01`), #5 (`f8db565`), #6 (`0ba85b4`/`8e8bb44`/`c7e0f09`), #7 (`be751cb`)
  - #8 deferred — confirmed theoretical-only window (self-host has no narrow-type bindings; trap can't fire on real code)
- Stage 5 Iter A: 2 of ~7 sub-steps done

## Active background work (do NOT duplicate)

### Subagents in worktrees

1. `stage5-iter-a` worktree — finishing Stage 5 Iter A: `parse_struct_decl`, `parse_program` update, `parse_struct_lit`, AST_STRUCT_DECL codegen, end-to-end test. Touches `helixc/bootstrap/parser.hx` + `helixc/bootstrap/kovc.hx`.
2. `finding-8-investigation` worktree — deep investigation of Finding #8. Touches `helixc/bootstrap/kovc.hx` comparison arms (~lines 3767-4101).
3. Third subagent — writing `docs/APPROACH_A_DETAILED_PLAN.md`. Read-only on source, writes one new doc file.

### External cron loops (independent of any session)

- `helix-approach-a-loop` (12-min, ENABLED) — drives Stage 5 forward. Same scope as Subagent 1 above.
- `helix-overnight-loop` (hourly, DISABLED) — was committing stdlib churn against directive. Left disabled.

## Non-conflicting work for this fresh session

Pick ONE of these so you don't trip over the active subagents:

### A) Stage 6 (Enums) advance planning [RECOMMENDED if Stage 5 isn't merged yet]

- Read `docs/APPROACH_A_PLAN.md` Stage 6 section
- Read `helixc/frontend/parser.py` and `helixc/frontend/types.py` for how enums work in helixc-Python (the reference)
- Sketch in `docs/STAGE_6_ENUMS_DESIGN.md`:
  - AST tags to reserve (next free is 55+ since 54 = AST_STRUCT_DECL)
  - Discriminant + payload codegen approach
  - Variant construction syntax
  - `match` interaction (Stage 7 prereq)
- Commit the design doc only. Do NOT implement until Stage 5 lands.

### B) Helix-Python parser quirks reference doc

- Document the host parser's recursion-budget limit and the FLAT prefix-trap pattern
- File: `runtime/memory/semantic/helixc-python-parser-quirks.md` (in the Kovostov framework, not helixc-Native)
- Pull lessons from commits `be751cb` (Finding #7), `0ba85b4` / `8e8bb44` (Finding #6), `c7e0f09` (Finding #8 root cause)
- Helps every future stage avoid re-discovering this

### C) Code review the recent audit commits

- Review `6aaec01`, `be751cb`, `8e8bb44`, `c7e0f09` for type-design issues using the `pr-review-toolkit:type-design-analyzer` agent
- Goal: catch any soundness gaps before Stage 5 lands on top of them
- Output: review summary, no code changes unless severe

### D) Phase 0 stdlib cleanup audit

- The `helix-overnight-loop` was committing stdlib pieces (`28fd864`, `b8a1afd`, `03720ee`, `0f121fd`)
- Audit each: is it actually a Phase-0 prereq? Or premature?
- If not prereq: document for later removal (don't remove now — destructive)

## What NOT to do in this session

- Do NOT modify `helixc/bootstrap/kovc.hx` (Subagents 1, 2 are touching it)
- Do NOT modify `helixc/bootstrap/parser.hx` (Subagent 1 is touching it)
- Do NOT spawn a duplicate Stage 5 Iter A subagent
- Do NOT reenable `helix-overnight-loop` — it conflicts with the user's "no stdlib churn" directive
- Do NOT touch the `stage5-iter-a` or `finding-8-investigation` worktree branches

## User directives currently in force

1. "Do all the features and nice to haves, do not skip anything we want even if it takes longer I want it done right" (2026-05-07)
2. "Don't defer fixes — use an agent to fix them while moving on" (2026-05-08)
3. "No Kovostov AI assembly" — STOP point
4. "No stdlib churn unless prereq for current stage"

## TG update format

Single-line beginner English. Ordered next-list (top=first). ~55% complete (anchored to APPROACH_A_PLAN.md = 30 stages + 7 amendments).

```
python /c/Projects/Kovostov/runtime/lib/kovostov_telegram.py send --chat 8212106071 --msg "..."
```

## Heavy gate

```
cd /c/Projects/Kovostov-Native && python -m pytest helixc/tests/ -n auto --tb=short -q
```

`-n auto` enables pytest-xdist parallel — ~4× speedup. Confirmed working in this session.

## Trap-id convention

`AST_TAG * 1000 + sub_id`. See `helixc/bootstrap/kovc.hx` for existing IDs.

## Flat prefix-trap pattern (mandatory)

Use this shape, NOT the wrap-with-if-else shape, when adding traps:

```
let n_pre_trap = if cond { emit_trap_with_id(N) } else { 0 };
... existing body ...
total + n_pre_trap
```

Wrap-with-if-else strains the host parser's recursion budget and miscompiles unrelated programs. See `c7e0f09` for the full root-cause writeup.
