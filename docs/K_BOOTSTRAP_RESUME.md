# K-Bootstrap autonomous-loop resume card

**Last paused:** 2026-05-26 ~20:03 local (after K2.D ship).
**Reason for pause:** user requested clean stop before computer restart.
**HEAD at pause:** `6918037` (K2.D — expand parity corpus from 25 to 40 items).
**Counter at pause:** `K_BOOTSTRAP_CHUNKS_DONE = 160` in `scripts/helix_status.py`.

## To resume after restart

Open Claude Code in this repo (`C:/Projects/Kovostov-Native` or
a Kovostov worktree pointed at it) and paste the loop command
below verbatim:

```
/loop 12m Helix K-bootstrap autonomous worker iteration. Repo: C:/Projects/Kovostov-Native, branch main — prefix `cd C:/Projects/Kovostov-Native` for git/pytest; use absolute C:/Projects/Kovostov-Native/... paths for file ops. Orient first: `git log --oneline -8`, `git status --short`; read docs/V3_HANDOFF.md (with the hard-constraint pointer at the top) and docs/K_BOOTSTRAP_HARD_CONSTRAINT.md (including the "Autonomous-loop stop criterion" section at the bottom).

**HARD CONSTRAINT (user directive 2026-05-26):** At v1.0 release the project must contain ZERO non-Helix runtime code. Python helixc must be deleted (K4 mandatory). GPU/MLIR/Tile ops MUST be ported to the bootstrap — they cannot stay in Python forever. No "defer to Python forever" strategy is acceptable for any subsystem.

**LOOP STOP CRITERION (user directive 2026-05-26 follow-up):** KEEP WORKING until the project reaches the Python-ready-to-delete state, then run 5 CONSECUTIVE CLEAN AUDITS as the stability gate. ONLY THEN stop the loop. Specifically:

1. Python-ready-to-delete = all Category-1 syntax niceties shipped + all Category-2 semantic gaps closed (impl method dispatch, generic monomorphization, mixed-type binops, f16 bit-accurate, reflection, tile ops, GPU backends, MLIR migration, trace events, field-store mutation, const-name resolution, macros) + K2 parity harness green + K3 trusted seed shipped.

2. 5 consecutive clean audits = per-chunk 3-axis (silent-failure / type-design / code-review) PLUS 5-clean end-of-phase (FE / IR / BE / RT / TEST). All 8 axes HIGH-confidence clean. Repeat 5x across separate ticks. ANY HIGH or must-fix MEDIUM resets the counter to 0.

3. Stop the loop = CronList -> CronDelete on this cron's id; send final Telegram with the 5-clean summary attached.

4. DO NOT perform K4 (delete Python) autonomously. K4 stays user-gated. The loop's job is to get the project to "ready to delete", not to delete.

Each tick, advance K-bootstrap by one coherent chunk: implement → test → commit → push → Telegram. See `scripts/helix_status.py` for live state. The K_BOOTSTRAP_CHUNKS_DONE constant in that file is the canonical chunk counter; bump it each commit.

Per-chunk DISCIPLINE:
- Per-chunk 3-axis audit when scope justifies (silent-failure-hunter / type-design-analyzer / code-reviewer). Small parser-level no-ops can skip the audit. Track consecutive-clean-audit counter once Python-ready-to-delete state is reached.
- Each chunk = one coherent commit. Commit message has the K1.* shape (concrete description, Co-Authored-By line).
- Push to origin/main after each commit (never force-push, never skip hooks).
- ALWAYS send a Telegram update after each commit: `cd C:/Projects/Kovostov-Native && MSG="$(python scripts/helix_status.py --note '<one plain-English sentence>' --commit $(git rev-parse --short HEAD))" && cd C:/Projects/Kovostov/runtime/lib && python kovostov_telegram.py send --chat 8212106071 --msg "$MSG"`.

HARD RULES (always):
- Claude subscription only — no external AI APIs, no direct Anthropic API.
- Never read C:/Projects/Neptune/api.env.
- Never force-push to main; never skip git hooks (--no-verify etc.).
- Never delete x86_64.py until test_codegen has been migrated AND a Telegram confirmation was sent.

If the tree is dirty when the tick fires, finish + commit that work before starting new chunks.

WSL FLAKE NOTE: the K-bootstrap self-host test chain runs through WSL and can flake when WSL is under load — same source can return rc=42 or rc=132 across runs. If a probe fails, re-run 2-3 times before concluding the code is broken. If baseline `fn main() -> i32 { 42 }` fails, WSL itself is degraded; skip the tick and let the next one retry.
```

(`12m` = fire every 12 minutes. Session-only — survives the
Claude session but dies on Claude exit. For a cloud-durable
schedule that survives Claude exits, use `/schedule` instead of
`/loop` with the same prompt body.)

## State snapshot at pause

**Recent commits (K2 phase):**
- `6918037` K2.D — expand parity corpus from 25 to 40 items (+ 3 carry-overs documented)
- `2d0ced6` K2.C — matrix-parity counter sync (honest-up the numbers)
- `3feee66` K2.B — expand parity corpus to 25 items
- `382c86a` K2.A — parity harness scaffold

**Matrix counter (re-tallied at K2.C):**
- `K_BOOTSTRAP_TOTAL_ROWS = 144`
- `K_BOOTSTRAP_PARITY_DONE = 126`
- 18 KOVC-MISSING rows = the Category-2 semantic gaps (see
  `docs/K_BOOTSTRAP_HARD_CONSTRAINT.md`).

**K2 corpus state (`helixc/tests/test_k2_parity.py`):**
- 40 items, all pass parity (Python helixc rc == bootstrap kovc rc).
- Size-guard ratchet at `>= 40`.

**Three pre-existing carry-overs documented in
`docs/K_BOOTSTRAP_HARD_CONSTRAINT.md` "Pre-existing Category-2
carry-overs":**
1. Bootstrap kovc `100_i64 - 58_i64` returns 100 not 42 — same-type
   i64 subtraction silently miscompiles. Dormant since commit
   `6fb85215` (2026-05-07). Belongs to mixed-type-binops Category-2
   bucket; needs dedicated multi-tick chunk.
2. Python helixc IR-lowering of char literals raises
   `NotImplementedError`. Bootstrap accepts char-lits (K1.K).
3. Python helixc match-block-arm requires inter-arm comma; bootstrap
   accepts the comma-less form (K1.AL).

## Next-tick recommendations (when loop resumes)

In order of leverage for Python-ready-to-delete:

1. **K1.E1 — i64-i64 subtraction codegen fix in kovc.hx.** This is
   the dormant bug K2.D surfaced. Likely 2-3 ticks: locate the i64
   binop emit path in `helixc/bootstrap/kovc.hx`, identify the
   missing/incorrect sub instruction, write a fix, verify the
   legacy `test_bootstrap_kovc_full_pipeline_arithmetic` assert
   passes AND K2 corpus stays green.

2. **K2.E corpus expansion** — 15-25 more items if K1.E1 looks too
   deep. Targets: multi-fn recursion, array indexing, larger struct
   fields, mixed-arity match. Lower leverage than (1) but always
   shippable.

3. **Other Category-2 starts** (after K1.E1 lands):
   - `__trace_event` runtime (small surface)
   - f16 bit-accurate literal (small AST tag addition)
   - mixed-type binops i64+i32 (the trap-on-mismatch path,
     distinct from the same-type broken path)
   - impl method dispatch (the biggest single leverage win;
     unblocks much real Helix code)

## To stop the loop cleanly mid-session (not via restart)

```
CronList                # find the K-bootstrap cron id
CronDelete <id>         # stop it
```

The cron used at pause time was `f9950fa2` (now deleted).
