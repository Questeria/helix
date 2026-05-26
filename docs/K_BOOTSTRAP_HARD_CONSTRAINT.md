# Hard constraint — Helix must be fully self-hosting

**Stated:** 2026-05-26 (user directive)
**Scope:** binding on the entire K-bootstrap track and any post-K1 work
**Severity:** HARD — no exceptions, no partial-credit, no "Python keeps X forever"

## The rule

When the K-bootstrap track completes (currently targeted at v1.0 in
`scripts/helix_status.py` terms), **the project must contain zero
non-Helix runtime code**. Specifically:

- **No Python in the compiler.** `helixc/` (the Python implementation)
  must be deleted. K4 (cutover) is **mandatory**, not optional.
- **No Python in test infrastructure** for compiled programs.
  Test harnesses that exercise `kovc.hx` (the Helix-side compiler) must
  themselves be written in Helix. (Python may remain for harness
  bootstrapping until the trusted-seed work at K3 closes that gap.)
- **No Python in build scripts** or developer tooling that ships with
  the project. If a `.py` file is in the source tree at v1.0, it
  must be either (a) removed, (b) ported to Helix, or (c) clearly
  marked as ephemeral dev tooling that runs outside the published
  artifact.
- **No deferral of features to "Python helixc forever."** Earlier
  optimization plans suggested keeping GPU / MLIR / Tile ops in
  Python permanently while bootstrap handles CPU x86 only. **That
  plan is invalid under this constraint.** Every feature in the
  Python helixc must be ported to the bootstrap before v1.0.

## Why this matters

Self-hosting is the headline goal of the K-bootstrap track
(`scripts/helix_status.py`: "SELF-HOSTING ACHIEVED -- the headline
goal: a Helix compiler written in Helix, compiled in Helix, all the
way from raw binary with NO Python in the final product"). The
hard-constraint statement makes "fully in Helix" non-negotiable.

This means:

- The remaining-chunks estimate (`docs/K_BOOTSTRAP_FEATURE_MATRIX.md`)
  must include all ~25 GPU/MLIR/Tile/reflection rows, not just the
  CPU-relevant subset.
- Any plan that says "defer X to a future track that never closes"
  is rejected by this constraint.
- K5 (DDC) is also mandatory because it's part of the "trusted from
  first principles, no Python in the chain" story.

## Practical impact on the optimization plan

The session-2026-05-26 optimization plan said:

> Aggressive Phase-2 ordering means GPU/MLIR don't ship in the
> bootstrap. That's fine: Python helixc keeps those, and K4 (delete
> Python) is the only step that requires bootstrap-side parity for
> them. We can defer GPU/MLIR until after K3 lands and re-evaluate
> whether they actually need to be in the bootstrap at all.

**This is no longer valid.** GPU/MLIR/Tile must be ported. The
"re-evaluate whether they need to be in the bootstrap" decision is
already made: yes, they do.

Realistic timeline impact: instead of "~25–35 chunks to a
deletable-Python state" (the optimistic estimate), the real path is
closer to **~60–80 chunks** because the GPU/MLIR/Tile/reflection
work cannot be skipped.

## Verification

At v1.0 release:

- `find C:/Projects/Kovostov-Native -name "*.py" | wc -l` should
  return zero (or only files explicitly marked as ephemeral dev
  tooling per the rule above).
- The bootstrap must compile **itself** (lexer.hx + parser.hx +
  kovc.hx) via the existing self-host test chain plus all v3.0
  features that the Python helixc supported.
- The DDC (K5) check must pass: build the bootstrap two
  independent ways, confirm bit-identical output.

## Autonomous-loop stop criterion (user directive 2026-05-26)

The autonomous-worker loop (cron job `5091b305` at the time of
writing) must KEEP WORKING until the project reaches the
**Python-ready-to-delete** state, at which point a stability
gate of **5 consecutive clean audits** unlocks loop termination.

Specifically:

1. **Python-ready-to-delete** means:
   - All Category-1 syntax niceties shipped (K1.* parser/lexer
     completion to the level real Rust source parses).
   - All Category-2 semantic gaps closed: impl method dispatch,
     generic monomorphization, mixed-type binops, f16 literals
     (bit-accurate), reflection (quote/splice/modify/reflect_hash),
     tile ops (TILE_ZEROS/ADD/MUL/MATMUL), GPU backends (PTX +
     ROCm + Metal + WebGPU), MLIR migration path, trace events,
     field-store mutation, const-name resolution, macros.
   - K2 (parity harness) green: every test program goes through
     both Python helixc AND bootstrap kovc.hx; outputs are
     byte-identical.
   - K3 (trusted seed) shipped: a small hand-audited Helix
     binary that re-bootstraps the compiler from source.

2. **5 consecutive clean audits** at that state means:
   - Run the per-chunk 3-axis audit (silent-failure-hunter /
     type-design-analyzer / code-reviewer) AND the 5-clean
     end-of-phase audit (FE / IR / BE / RT / TEST).
   - **All 8 axes must come back HIGH-confidence clean.**
   - Repeat 5 times in succession, ideally across different
     ticks separated by at least one re-compilation of the
     bootstrap chain.
   - Any HIGH or must-fix MEDIUM finding resets the consecutive
     counter to 0.

3. **What "stopping the loop" means**:
   - `CronList`, find the loop job id, `CronDelete <id>`.
   - Send a final Telegram noting the loop terminated, with the
     5-clean-audit summary attached.
   - **Do NOT perform K4 (delete Python) autonomously** -- that
     remains user-gated. The loop's job is to get the project
     to a state where the user can safely trigger K4 with one
     command, not to perform K4 itself.

4. **Implication**: there is NO "v1.0 reached, loop done"
   threshold while Python is still present. The trigger is
   **ready-to-delete + 5-clean × 5 consecutive runs**, not
   "Python actually deleted". K4 is intentionally a manual
   step.

## References

- User directive: 2026-05-26 conversation (initial hard constraint)
- User directive: 2026-05-26 follow-up (5-clean-audit stop criterion)
- Stored in Kovostov semantic memory:
  `C:/Projects/Kovostov/runtime/memory/semantic/helix.md`
  (entries at `2026-05-26T06:26:38Z` and the 5-clean-audit
  follow-up at the next timestamp)
- Supersedes: optimization-plan deferral language re GPU/MLIR/Tile;
  the cron prompt's earlier "v1.0 reached" stop criterion
