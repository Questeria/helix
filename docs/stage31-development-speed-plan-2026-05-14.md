# Stage 31 Development Speed Plan - 2026-05-14

Goal: increase Helix development speed without reducing correctness,
verification strength, or audit discipline.

## Main Rule

Only one lane owns a commit-producing code slice at a time. Other lanes may run
in parallel if they are read-only, documentation-only, or isolated to clearly
disjoint files/worktrees.

## Parallel Lanes

### 1. Main Implementation Lane

Purpose: produce the next safe commit.

Allowed:
- Edit the current slice files.
- Run focused tests, quick gate, broad gate, and final official gate.
- Commit and push only explicit paths.

Not allowed:
- Mix unrelated compiler-core changes into one commit.
- Start another core edit that depends on an unverified result.

### 2. Audit Lane

Purpose: use long test windows instead of idle waiting.

Allowed while tests run:
- Read current diffs.
- Review changed code for silent failures.
- Inspect logs and prior audit docs.
- Prepare audit findings.

Action rule:
- If audit finds a real issue, fix it in the main lane and reset the clean
  count.
- If an audit stalls, restart it instead of waiting indefinitely.

### 3. Future-Stage Prep Lane

Purpose: prepare the next move without touching active code.

Allowed while audits or full gates run:
- Read roadmap and stage docs.
- Identify dependency-free next slices.
- Draft tests and implementation notes.
- Write planning docs.

Examples:
- Map the next proof/refinement feature.
- Identify stdlib aliases or helper functions that do not touch compiler core.
- Build a file ownership map for future parallel workers.

### 4. Independent Implementation Lane

Purpose: safely work ahead when the next feature is truly independent.

Allowed only when one of these is true:
- Work happens in a separate git worktree/branch.
- Work touches disjoint files and does not depend on current uncommitted code.
- Work is docs/tooling/stdlib and cannot change the active compiler behavior.

Examples:
- Documentation and usage guides.
- Slow-test telemetry.
- Proof-artifact helper scripts.
- New stdlib definitions with isolated tests.

Avoid:
- Editing `helixc/frontend/typecheck.py`, parser, lowering, codegen, or
  bootstrap files in two lanes at the same time.

## Testing Speedups That Preserve Coverage

1. Keep stable sharding for codegen and non-codegen suites.
2. Print slowest shards after every full run.
3. Add a failed-shard rerun mode to distinguish real failures from flakes.
4. Add slow-test telemetry by node id, not just by shard.
5. Balance codegen shards by historical duration instead of hash count.
6. Keep full gates for commit boundaries, but use focused tests plus quick gate
   during edit loops.
7. Avoid deleting caches unless diagnosing cache corruption.
8. Keep WSL temp paths isolated or locked so concurrent runs cannot collide.

## Next Practical Improvements

1. Add a `--rerun-failed` or `--retry-failed-once` option to
   `scripts/stage31_validate.py`.
2. Save slow-shard summaries into a machine-readable JSON file.
3. Add duration-weighted shard assignment for `test_codegen.py`.
4. Add a changed-files-to-tests map for focused gates.
5. Use separate worktrees for truly independent stage work while the main lane
   runs final gates.

## Commit Discipline

Every speedup must be judged by:
- Does it keep the same tests or add more?
- Does it make failures clearer?
- Does it reduce repeated waiting?
- Does it avoid hiding real bugs?
- Can another AI understand and continue it from repo files?
