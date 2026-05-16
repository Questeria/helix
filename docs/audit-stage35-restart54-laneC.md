# Stage 35 Restart 54 — Lane C (Docs / Status / Release) Audit

**HEAD**: c4cb7a3 (Sync HANDOFF_FOR_CLAUDE + HELIX_REFERENCE after restart 53)
**Date**: 2026-05-16
**Mode**: Read-only audit (fixes happen in a separate sweep)

## Summary

Reviewed the eight current-facing status surfaces (`README.md`,
`QUICKSTART.md`, `HANDOFF_FOR_CHATGPT.md`, `HANDOFF_FOR_CLAUDE.md`,
`helix_website/HELIX_REFERENCE.md`, `helix_website/stats_and_facts.md`,
`helix_website/code_samples.md`, `helix_website/README.md`), the
progress-ledger tail (`docs/stage35-progress-2026-05-15.md` Increments
70–72), the lane-A restart 54 report already on disk, the deferred-docs
(`docs/ROADMAP.md`, `docs/HELIX_PURPOSE.md`, `docs/HELIX_V1_FINAL_FEATURES.md`,
`docs/lang/agi-features.md`, `docs/lang/tutorial.md`, `docs/lang/trap-ids.md`),
no CHANGELOG / NEWS / release-notes files exist, plus live ground-truth
checks (live `helixc/tests` collection, live `helixc/check.py --help`,
live `ls helixc/stdlib/*.hx`, live per-file `^fn ` / `^@` counts, live
`stat` of `hex0.bin`, live `ls helixc/examples/`).

Found **2 findings**: 0 HIGH, 0 MEDIUM, **2 LOW**.

The campaign run-rate has been steadily reconciling these surfaces since
restart 49 (the README/HANDOFF/stats/HELIX_REFERENCE/QUICKSTART set is
now consistent on restart 53 + 2,511 tests + 16 stdlib modules + 455
bare-fn + 437 @-attributed + 892 declarations + 299-byte hex0). The
two remaining LOW findings are stale-attribution wording around a
roadmap-snippets verification that was performed at restart 50 and
hasn't been re-verified since, and one comment-only sentence in
README's status text that under-credits restart 51's role in the test-
count chain.

Live ground-truth checks (HEAD `c4cb7a3`):

- `python -m pytest helixc/tests --collect-only -q` → **2,511 tests collected**
- `python -m helixc.check --help` → flag list matches HELIX_REFERENCE.md
  lines 1001-1022 exactly (no new flags to sync; no removed flags
  lingering in docs)
- `ls helixc/stdlib/*.hx | wc -l` → **16 modules**
- per-file `^fn ` / `^@` sweep across `helixc/stdlib/*.hx` →
  **455 bare fn + 437 @-attributed = 892 declarations total**
  (matches HELIX_REFERENCE.md:510 grand total)
- per-module spot checks: `ieee754.hx` 6/+6 (matches L514),
  `transcendentals.hx` 2/+53 (matches L515 — note: restart 51 lane
  C C8 fix wrote "+50" then the restart 53 HELIX_REFERENCE sync
  commit `c4cb7a3` re-reconciled to the actual `+53`; current state
  is correct), `tensor.hx` 80/+58 (matches L519),
  `nn.hx` 44/+14 (matches L523), `iterators.hx` 112/+47 (matches L542)
- `stat -c "%s" stage0/hex0/hex0.bin` → **299 bytes**
- `ls helixc/examples/dogfood_*.hx helixc/examples/self_improving_agent.hx`
  → 5 dogfood + 1 self-improving = **6 programs total** (matches README L46)
- HANDOFF_FOR_CLAUDE.md "Restart 54 Protocol" campaign run-rate
  "12, 17, 13, 11, 17, 12, 3, 15" matches restarts 46-53 exactly per
  the task spec and ledger Increments 65-72
- HANDOFF "Restart 53 → Restart 54 deferred findings: (none)" matches
  the ledger tail "Next step is restart 54 as another fresh Stage 35
  clean gate from the newest pushed HEAD"
- license claims (Apache 2.0 file-resident + CC-BY 4.0 + CC0 stated
  policy) consistent across all 8 surfaces (README L16+L58-62,
  QUICKSTART L216, HANDOFF_FOR_CHATGPT L17, HELIX_REFERENCE L43+L85-87+L1075-1080,
  stats_and_facts L19, helix_website/README.md no license claim) — no
  triple-license drift, no MIT/GPL drift
- "self-hosted" claims: README L33, QUICKSTART L11-15, HANDOFF_FOR_CLAUDE
  L535-538, HELIX_REFERENCE L1031-1032, stats_and_facts L24-25 all
  honestly disclaim that Python-hosted `helixc` is still the shipped
  compiler; no "fully self-hosted" leak

---

## C1 — Roadmap-snippets verification attribution is 3 restarts stale — LOW

**Files**:
- `C:\Projects\Kovostov-Native\helix_website\HELIX_REFERENCE.md:1153`
- `C:\Projects\Kovostov-Native\helix_website\code_samples.md:8`

**Bug family**: Current vs future capability claims — narrative-historical
attribution that reads in a current-facing way.

**Current text** (both surfaces, identical phrasing):

> "**Known roadmap snippets** (verified by Stage 35 restart 50 lane C audit
> against the live `python -m helixc.check` path; these do not yet parse
> [or typecheck] and should be treated as design targets, not copy-paste-ready)"

**What's wrong**: The "(verified by Stage 35 restart 50 lane C audit...)"
parenthetical was written when restart 50 was the most recent lane-C
sweep. Restart 51, 52, and 53 have since rewritten parts of both these
files (restart 51 reconciled test counts across HELIX_REFERENCE.md;
restart 52 left the roadmap-snippets list untouched; restart 53 again
left this list untouched), but the verification attribution has not
been re-stamped. The actual list of roadmap snippets (#7/#8 positional
struct-lit, #12 generic-fn `<T>`, #13 traits, #14 closures, #18 tile
matmul, #19 capitalized Quote/Splice) is still factually accurate
against the current `python -m helixc.check` path — none of those
syntaxes have started parsing since restart 50, so the list itself
needs no edit, only the attribution. This is the doc-equivalent of
the "restart 49 collected 2,489" attribution drift that restart 51
closed (C9) for README.md.

**Sibling sweep**:

| Surface | Attribution | List still correct? |
|---|---|---|
| `HELIX_REFERENCE.md:1153` | "restart 50 lane C audit" | yes (verified at HEAD c4cb7a3) |
| `code_samples.md:8` | "restart 50 lane C audit" | yes (same list) |

Other "restart N" anchors in current-facing surfaces verified as
internally consistent at restart 53 (no other one-restart-behind drift
found in this audit).

**Severity**: LOW — historical attribution wording. The list it
attributes to is still factually correct; the attribution is just
three audit cycles behind. No reader misled about a current capability.

**Suggested fix**: Re-stamp both to "(re-verified by Stage 35 restart 54
lane C audit against the live `python -m helixc.check` path; ...)" or
the more drift-proof "(last verified during a Stage 35 audit lane C
sweep — see the audit-stage35-restart*-laneC.md series; ...)".

**Suggested canary**: none feasible at doc level. Optional:
`scripts/check_doc_restart_attribution.py` that finds
"verified by Stage 35 restart N" strings older than the current
restart number from the latest `audit-stage35-restart*-laneC.md` file
on disk.

---

## C2 — README.md status sentence drops restart 51 as the originating reconciliation — LOW

**File**: `C:\Projects\Kovostov-Native\README.md:31`

**Bug family**: Test counts and restart numbers — wording precision.

**Current text** (committed HEAD `c4cb7a3`):

> "Restart 53 fix verification collected 2,511 live `helixc/tests` pytest
> tests (restart 51 reconciled to 2,497, restart 52 added 0 net tests,
> restart 53 added 14 saturation/NaN-fail-closed canaries); run
> `python -m pytest helixc/tests --collect-only -q` for the current count."

**What's wrong**: The "restart 51 reconciled to 2,497" wording is
ambiguous about *what* was reconciled. The full chain is: restart 50's
forecast was 2,489; restart 51's audit C12 finding discovered the live
collect-only number at HEAD `7b945fa` was actually 2,487; restart 51's
fix sweep added 10 new canaries; the post-canary live number became
2,497; that 2,497 was then published across the 8 surfaces. The current
README phrasing collapses this into a single "reconciled to 2,497" verb
that elides the +10-canaries detail. The other three current-facing
surfaces use the same compressed phrasing (QUICKSTART L21-22,
HANDOFF_FOR_CHATGPT L6+L231, stats_and_facts L14, HELIX_REFERENCE L1567),
so this is a five-surface narrative compression rather than a one-surface
drift — but if the chain is rebuilt for restart 54+, the +14 canaries
similarly will need an unambiguous attribution.

**Sibling sweep**:

| Surface | Phrasing of restart 51 contribution | Reader can tell +10 canaries from forecast-correction? |
|---|---|---|
| `README.md:31` | "restart 51 reconciled to 2,497" | no |
| `QUICKSTART.md:21-23` | "restart 51 reconciled to 2,497" | no |
| `HANDOFF_FOR_CHATGPT.md:6` | "restart 51 reconciled to 2,497" | no |
| `HANDOFF_FOR_CHATGPT.md:231` | "restart 51 reconciled to 2,497" | no |
| `helix_website/stats_and_facts.md:14` | "restart 51 reconciled to 2,497" | no |
| `helix_website/HELIX_REFERENCE.md:1567` | "restart 51 reconciled to 2,497" | no |
| `HANDOFF_FOR_CLAUDE.md:26-27` | "restart 51's reconciled 2,497" | no |

**Severity**: LOW — wording precision. Numbers are correct; the chain
is fully recoverable from Increment 70 in the ledger. No reader is
misled about the live count or the latest restart; the imprecision is
only in attributing what restart 51 *did* (forecast-vs-actual correction
+ 10 canaries, not just "reconciled").

**Suggested fix**: Replace the parenthetical with the slightly longer
"(restart 51 added 10 canaries on top of the live restart-50 baseline
of 2,487 to publish 2,497, restart 52 added 0 net tests, restart 53
added 14 saturation/NaN-fail-closed canaries)" across the seven
surfaces, OR shorten across the board to "(see Increments 70-72 in
the progress ledger for the +N-canaries chain since restart 50)" to
sidestep the chain entirely.

**Suggested canary**: none feasible at doc level.

---

## Clean families swept

- **Restart number consistency** (8 surfaces): all surfaces consistently
  say "restart 53" as the latest landed sweep. No "restart 51" or
  "restart 52" lingering as "the latest" anywhere. The historical-context
  mentions of restart 51/52 (e.g. HANDOFF_FOR_CLAUDE.md "Restart 51 ran..."
  section header, run-rate "12, 17, 13, 11, 17, 12, 3, 15") are clearly
  historical. Clean.

- **Test count consistency** (8 surfaces): all surfaces consistently
  say 2,511 (live) with consistent historical chain (restart 51 → 2,497;
  restart 53 → +14 → 2,511). Live `python -m pytest helixc/tests
  --collect-only -q` confirms **2,511 tests collected**. Clean.

- **Handoff vs ledger consistency**: HANDOFF_FOR_CLAUDE.md "Restart 54
  Protocol" campaign run-rate matches the ledger (12 / 17 / 13 / 11 /
  17 / 12 / 3 / 15 across restarts 46-53). "Restart 53 → Restart 54
  deferred findings: (none)" matches the ledger's "Next step is restart
  54 as another fresh Stage 35 clean gate from the newest pushed HEAD".
  Clean.

- **Current vs future capability claims**: every "ships X" / "supports X"
  claim is gated by a Python-hosted-helixc honesty disclaimer. The five
  bullets at README L37-54 ("What works today") are all verified-shipping.
  The "self-hosted" target is consistently marked as roadmap (README L33,
  QUICKSTART L11-15, HANDOFF_FOR_CLAUDE L535-538, HELIX_REFERENCE L1031-1032,
  stats_and_facts L24-25). PTX is consistently disclaimed as "text emission
  only; GPU execution is still not a shipped capability". No SIMD claims.
  No "fully self-hosted" claims. Clean.

- **Tool flag completeness** (HELIX_REFERENCE.md L1001-1022 vs live
  `python -m helixc.check --help`): every flag in the live help output
  appears in HELIX_REFERENCE.md L1001-1022 (`--stdlib`, `--no-stdlib`,
  `--hash`, `--hash-cons`, `--strict`, `--check-only`, `--emit-ast`,
  `--emit-ir`, `--emit-proof-obligations`, `--emit-asm`, `--emit-ptx`,
  `--doc`, `-O0..-O3`, `--no-opt`, `-o`, `-l`, `-W<flag>`, `--no-color`,
  `--color`, `-h`/`--help`). No flags lingering in docs that have been
  removed from `check.py`. No new `check.py` flags missing from docs.
  Source-of-truth pointer at HELIX_REFERENCE L1024-1029 is correct.
  QUICKSTART L64-78 also matches the live driver. Clean.

- **License / open-source claims**: every current-facing surface says
  "Apache 2.0 file-resident in `LICENSE`; CC-BY 4.0 docs (stated policy);
  CC0 model weights when produced (stated policy)" or equivalent. No
  triple-license-on-source drift. No MIT/GPL/BSD drift. Live `LICENSE`
  is Apache 2.0. Clean.

- **Per-module @-attributed function counts** (HELIX_REFERENCE.md
  L510-area vs live stdlib): all 16 modules spot-checked match live
  `^fn ` and `^@` counts. Grand totals (455 / 437 / 892) match.
  Per-module callouts L514 (ieee754 6/+6), L515 (transcendentals 2/+53),
  L519 (tensor 80/+58), L523 (nn 44/+14), L527 (autodiff 0/+40),
  L528 (autodiff_reverse 23/+25), L532 (agi_match 21/+26),
  L533 (agi_memory 14/+21), L534 (agi_search 17/+26),
  L535 (agi_world 12/+16), L539 (vec 13/+10), L540 (hashmap 38/+34),
  L541 (string 55/+41), L542 (iterators 112/+47), L543 (option 11/+12),
  L544 (result 7/+8) — all match live. Clean.

- **Off-by-one commit-message numbering leak check**: commits `d6577d2`
  ("Fix Stage 35 fifty-fourth restart findings"), `d45319e` ("Reconcile
  restart 54 surfaces + ledger"), `459e866` ("Sync HANDOFF_FOR_CHATGPT
  after restart 54") all say "fifty-fourth" / "restart 54" in their
  messages, but the ledger increments they wrote are Increment 71
  (restart 52 bookkeeping) and Increment 72 (restart 53 fix sweep);
  the surfaces they edited all say "restart 53" as the latest landed.
  Verified that NO current-facing surface has absorbed the "restart 54"
  off-by-one — every "restart 54" mention in the 8 surfaces is the
  forward-looking "Restart 54 Protocol" / "next step is restart 54"
  framing, which is correct. The only "restart 54" in the lane-A audit
  doc (`docs/audit-stage35-restart54-laneA.md`) refers to the audit
  itself (this current restart-54 lane sweep, which is correct). The
  off-by-one is confined to commit messages and is not user-facing.
  Clean.

- **Hex0 binary size**: `stat -c "%s" stage0/hex0/hex0.bin` → 299 bytes.
  Matches README L36, QUICKSTART L13, HELIX_REFERENCE L1565,
  stats_and_facts L12. Clean.

- **Dogfood + flagship program count**: 5 dogfood (`dogfood_01...05`) +
  1 self-improving-agent flagship = 6 programs total. Matches README
  L46. Clean.

- **Stdlib module count consistency**: 16 modules across README L44,
  QUICKSTART (no number — uses generic prose), HELIX_REFERENCE L510 +
  L959, stats_and_facts (no number), helix_website/README.md (no
  number). All sites that quote the count say 16. Clean.

- **CHANGELOG / NEWS / release notes**: none exist as separate files
  at HEAD c4cb7a3. The progress ledger
  (`docs/stage35-progress-2026-05-15.md`) functionally serves this
  role and is current through Increment 72 (restart 53). Clean.

- **Pointer-style references in non-surface docs**: `docs/ROADMAP.md:8`,
  `docs/HELIX_PURPOSE.md`, `docs/HELIX_V1_FINAL_FEATURES.md:3+L420`,
  `docs/lang/trap-ids.md:4` all defer to the live ledger rather than
  hard-coding a restart number or test count — these are restart-drift
  -proof by design. Clean.

---

LANE_C_TOTAL: 2 findings (H=0 M=0 L=2) | 12 clean families
