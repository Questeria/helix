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

## Parser-saturation milestone (2026-05-26, K_BOOTSTRAP_CHUNKS_DONE=156)

As of K1.DV (commit `c3096b7`), the bootstrap parser's surface for
**type-binding positions** is closed. The `&T` + `<...>` template
(reference type with optional lifetime + mut, plus optional generic
args after the type IDENT) is consistently applied across all 6
parser sites where a type can appear:

  1. **K1.S / K1.DS** — let-type position (`let v: &Vec<i32>`).
  2. **K1.BD / K1.CT** — top-level fn-param type (`fn f(v: &Vec<i32>)`).
  3. **K1.DR** — top-level fn-return type (`fn f() -> &Vec<i32>`).
  4. **K1.DT** — impl-method-return type (`impl S { fn x() -> &Vec<i32> }`).
  5. **K1.DU** — struct field type (`struct S { v: &Vec<i32> }`).
  6. **K1.DV** — impl-method-param type (`impl S { fn x(v: &Vec<i32>) }`).
  7. **K1.BN-extra** — enum variant payload type (`enum E { A(&Vec<i32>) }`)
     was already covered by K1.BN's paren-balanced scan.

Verification probes (2026-05-26 post-K1.DV) covering 23 separate
type-position shapes — including `&T`, `&'static T`, `&mut T`,
`&'lt T`, `&Vec<T>`, `&Box<dyn T>`, `Vec<&T>`, `(&T, &T)`, HRTB
`for<'a> Fn(&'a T)`, `impl A + B`, `&Self`, nested generics — all
PASS. Probes for adjacent shapes (match patterns, decl bounds,
trait method decls, enum/struct/union/trait variations) also all
PASS.

**Implication for the loop**: parser-side K1.* chunks have hit
diminishing returns. The next leverage point for "Python-ready-to-
delete" is the 12 Category-2 semantic gaps named at the top of this
document, not further parser surface coverage. K2 (parity harness)
running over a real-source corpus is also the gate that will
surface what remaining parser corners (if any) actually matter.

## Pre-existing Category-2 carry-overs (discovered 2026-05-26 K2.D)

The K2 corpus expansion runs surfaced two existing failures that
predate the K2 phase and have been quietly broken throughout the
recent K1.* parser work:

1. **Bootstrap kovc i64-i64 subtraction silently miscompiles**
   (`test_bootstrap_kovc_full_pipeline_arithmetic` line ~2898,
   commit `6fb85215` dated 2026-05-07). Source `100_i64 - 58_i64`
   returns 100 instead of 42 — the subtraction is dropping and the
   left operand flows through. Pre-existing for ~3 weeks; not
   caused by recent work. Belongs to the **mixed-type binops**
   Category-2 bucket (same code path that traps on i64+i32).
   Treat as a dedicated multi-tick chunk; ship pure-i32 corpus
   items in the meantime.

2. **Python helixc char-literal IR-lowering** raises
   `NotImplementedError: char literal not yet supported in IR
   lowering at <pos>`. The bootstrap kovc accepts char literals
   (per K1.K). K2.D's `let c = 'A';` shape hit this gap; the
   corpus uses a let-shadowing variant instead. Python-side gap;
   future K2.* chunk re-introduces char-literal once Python helixc
   gains parity (or after K4 deletes Python).

3. **Python helixc match-block-arm requires explicit comma**
   between arms when an arm body is a brace block (`} _ =>`
   errors "expected RBRACE got IDENT '_'"). The bootstrap kovc
   accepts the comma-less form (per K1.AL). K2.D's corpus uses
   the comma-separated form for parity.

Items (2) and (3) are Python-side defects -- K2 surfaces them
because both compilers must accept every corpus item. They are
NOT bootstrap bugs. Item (1) IS a bootstrap bug that has been
dormant in test_codegen.py.

## References

- User directive: 2026-05-26 conversation (initial hard constraint)
- User directive: 2026-05-26 follow-up (5-clean-audit stop criterion)
- Stored in Kovostov semantic memory:
  `C:/Projects/Kovostov/runtime/memory/semantic/helix.md`
  (entries at `2026-05-26T06:26:38Z` and the 5-clean-audit
  follow-up at the next timestamp)
- Supersedes: optimization-plan deferral language re GPU/MLIR/Tile;
  the cron prompt's earlier "v1.0 reached" stop criterion
