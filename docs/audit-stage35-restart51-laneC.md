# Lane C Audit Report — Stage 35 Restart 51

**HEAD**: `7b945fa Record Stage 35 restart 50 lane audit reports`
**Scope**: Docs / status / release honesty. Read-only audit; fixes applied separately.

## Summary

Reviewed `helix_website/HELIX_REFERENCE.md`, `helix_website/stats_and_facts.md`, `helix_website/code_samples.md`, `helix_website/README.md`, `README.md`, `QUICKSTART.md`, `HANDOFF_FOR_CLAUDE.md`, `HANDOFF_FOR_CHATGPT.md`, `docs/ROADMAP.md`, `docs/HELIX_V1_FINAL_FEATURES.md`, `docs/HELIX_PURPOSE.md`, `docs/lang/agi-features.md`, `docs/lang/tutorial.md`, `docs/lang/trap-ids.md`, `docs/stage35-progress-2026-05-15.md` (tail/Increment 69), live `helixc/stdlib/*.hx` fn counts, `helixc/check.py` flag inventory.

Found **6 new issues**: 2 HIGH, 3 MEDIUM, 1 LOW. (Includes the C8 carry-forward from restart 50, now narrowed to two specific modules.)

Live ground-truth checks (committed HEAD `7b945fa`):

- `python -m pytest helixc/tests --collect-only -q` → **2,487 tests collected** (verified twice in this audit; live count is 2 below the "2,489" published in 8 surfaces)
- `python -m helixc.check --help` → flag list matches HELIX_REFERENCE.md's table (no new flags to sync)
- `ls helixc/stdlib/*.hx | wc -l` → **16 modules**
- `for f in helixc/stdlib/*.hx; do echo "$f: $(grep -c '^fn ' "$f")"; done` → totals **455 bare fn** (matches the line-510 headline)
- `stat -c "%s" stage0/hex0/hex0.bin` → **299 bytes**
- `ls helixc/examples/dogfood_*.hx` → **5 files** + `self_improving_agent.hx` → **6 programs total**

---

## C9 — `README.md` credits 2,489 tests to restart 49 (should be restart 50) — HIGH

**File**: `C:\Projects\Kovostov-Native\README.md:31`
**Bug family**: Test counts and restart numbers — current-status text attributes restart 50's number to restart 49.

**Current text** (committed HEAD `7b945fa`):

> "restart 50 is the latest recorded fix sweep in this status text. **Restart 49 fix verification collected 2,489 live helixc/tests pytest tests**; run `python -m pytest helixc/tests --collect-only -q` for the current count."

**What's wrong**: The two sentences contradict each other inside the same paragraph. Per `docs/stage35-progress-2026-05-15.md` Increment 69 (restart 50 fix sweep) the 2,489 collection happened *during restart 50*, not restart 49. Restart 49's collection was 2,479 (per the ledger one increment earlier and per `docs/audit-stage35-restart50-laneC.md:7`). The "restart 49 collected 2,489" wording is the same kind of one-restart-behind drift that restart 49 closed in other files; here it lingered because README's status sentence was rewritten in the restart-50 fix sweep but only the lead clause (`restart 50 is the latest`) was updated, not the body number attribution.

**Sibling sweep**:

| Surface | restart-number anchor | tests-count anchor | Consistent? |
|---|---|---|---|
| `README.md:31` | restart 50 | 2,489 / restart 49 | **NO — internal contradiction** |
| `QUICKSTART.md:20-23` | restart 50 | 2,489 / restart 50 | yes |
| `HANDOFF_FOR_CLAUDE.md:24` | restart 50 | 2,489 / restart 50 | yes |
| `HANDOFF_FOR_CHATGPT.md:6` | restart 49 (stale) | 2,489 / restart 49 | yes-internal, stale vs ledger — see C11 |
| `HANDOFF_FOR_CHATGPT.md:231` | restart 50 | 2,489 / restart 50 | yes — internal contradiction with line 6 |
| `helix_website/stats_and_facts.md:8` | restart 49 (stale) | (n/a) | stale — see C10 |
| `helix_website/stats_and_facts.md:14` | restart 50 | 2,489 / restart 50 | yes — internal contradiction with line 8 |
| `helix_website/HELIX_REFERENCE.md:961` | restart 50 | 2,489 / restart 50 | yes |
| `helix_website/HELIX_REFERENCE.md:1567` | restart 50 | 2,489 / restart 50 | yes |

**Severity**: HIGH — the README is the first-read file and the wrong restart attribution puts the public count on the wrong cycle of the audit ledger.

**Suggested fix**: Replace "Restart 49 fix verification collected 2,489 live `helixc/tests` pytest tests" with "Restart 50 fix verification collected 2,489 live `helixc/tests` pytest tests" (or with the 2,487 live-reconciled wording from C12 if both fixes land together).

**Suggested canary**: None (doc-only). Optional: a `scripts/check_doc_test_count.py` grep that confirms all current-facing surfaces cite the same `(restart-N, count)` pair.

---

## C10 — `stats_and_facts.md` snapshot prose says "restart 49" while table says restart 50 — MEDIUM

**File**: `C:\Projects\Kovostov-Native\helix_website\stats_and_facts.md:8`
**Bug family**: Handoff and progress-ledger consistency — within one file.

**Current text** (committed HEAD `7b945fa`):

> "Snapshot date: 2026-05-16. **Restart 49 is the latest recorded Stage 35 fix verification in this file**; use live `git log -1 --oneline` before publishing."

The table immediately below at line 14 correctly attributes 2,489 to "restart 50 fix verification":

> `| **pytest tests collected** | 2,489 | python -m pytest helixc/tests --collect-only -q during restart 50 fix verification |`

**What's wrong**: Snapshot preamble at line 8 is one restart behind the table at line 14, inside the same file. Restart 50 was the latest committed fix sweep at HEAD `7b945fa`, so both the prose and the table should say restart 50. This is a missed sibling of the wholesale "restart 50" sweep that updated the table row.

**Sibling sweep**: see C9 table above — `stats_and_facts.md` is one of two files (the other is `HANDOFF_FOR_CHATGPT.md`, see C11) whose preamble lags the body by one restart number.

**Severity**: MEDIUM — internal contradiction inside a "Stats and Facts" file that explicitly markets itself as the source-of-truth snapshot.

**Suggested fix**: `Snapshot date: 2026-05-16. Restart 50 is the latest recorded Stage 35 fix verification in this file; use live git log -1 --oneline before publishing.`

---

## C11 — `HANDOFF_FOR_CHATGPT.md` continuation pointer says "restart 49 is the latest" — MEDIUM

**File**: `C:\Projects\Kovostov-Native\HANDOFF_FOR_CHATGPT.md:6`
**Bug family**: Handoff and progress-ledger consistency.

**Current text** (committed HEAD `7b945fa`):

> "**Restart 49 is the latest recorded fix sweep in this file**; clean gates remain `0/3`, and live `helixc/tests` collection is 2,489."

But line 231 in the same file correctly attributes 2,489 to restart 50:

> "Stage 35 restart 50 fix verification collected 2,489 live `helixc/tests` tests"

**What's wrong**: Line 6 ("restart 49 is the latest") and line 231 ("restart 50 fix verification collected 2,489") contradict each other inside the same handoff. The 2,489 count is correctly attributed to restart 50 in the body but to restart 49 in the lead.

**Sibling sweep**: identical pattern to C10 (`stats_and_facts.md:8` vs line 14). Both files had the body number updated during the restart-50 fix sweep without updating the preamble pointer.

**Severity**: MEDIUM — handoff file consumed by the ChatGPT-side workflow; lead-vs-body mismatch undermines the "tail the ledger for newest truth" instruction in the same paragraph.

**Suggested fix**: `Restart 50 is the latest recorded fix sweep in this file; clean gates remain 0/3, and live helixc/tests collection is 2,489.`

---

## C12 — Live `collect-only` is 2,487; 8 surfaces say 2,489 — HIGH

**Files**:
- `C:\Projects\Kovostov-Native\README.md:31`
- `C:\Projects\Kovostov-Native\QUICKSTART.md:21`
- `C:\Projects\Kovostov-Native\HANDOFF_FOR_CLAUDE.md:24`
- `C:\Projects\Kovostov-Native\HANDOFF_FOR_CHATGPT.md:6, 231`
- `C:\Projects\Kovostov-Native\helix_website\stats_and_facts.md:14`
- `C:\Projects\Kovostov-Native\helix_website\HELIX_REFERENCE.md:961, 1567`

**Bug family**: Test counts and restart numbers — forecast-vs-actual drift.

**What's wrong**: All 8 current-facing surfaces publish "2,489 tests collected". Re-running the documented command at HEAD `7b945fa` returns a different number:

```
$ python -m pytest helixc/tests --collect-only -q
... 2487 tests collected in 90.64s (0:01:30)
$ python -m pytest helixc/tests --collect-only -q -p no:cacheprovider
... 2487 tests collected in 58.83s
```

Both invocations return **2,487**, deterministically. The ledger Increment 69 entry says "Result: 2,489 tests collected (was 2,479 + 10 net)" but Increment 68's prior wording was "in flight at commit time, expected 2,479 (was 2,466 + 13 new)" — suggesting the published counts are forecasts written from "old-count + N-new-tests-added" arithmetic rather than from re-running collect-only after the commit landed. Restart 50 added 10 tests on top of the 2,479 forecast; the actual delta was 8, which gives 2,487, not 2,489.

This is a published-vs-live contradiction on the most public number on every current-facing surface. Worse than C9/C10/C11 because the discrepancy is reproducible by anyone who runs the documented command — including the "rerun scoped pytest collection before publishing" guard text right next to the number in `stats_and_facts.md` and `HELIX_REFERENCE.md`. The guard fails the moment anyone uses it.

**Sibling sweep**: see C9 table. Every surface publishes 2,489; live is 2,487 across both invocations.

**Severity**: HIGH — the public number on README, QUICKSTART, stats_and_facts, and HELIX_REFERENCE is off by 2 from what the documented command actually returns.

**Suggested fix**: Sweep all 8 surfaces from 2,489 → 2,487. Also update `docs/stage35-progress-2026-05-15.md` Increment 69 to record the live-vs-forecast reconciliation ("forecast 2,489; live collect-only = 2,487; 2 forecasted tests were not added or were skip-marked"). Going forward, run collect-only *after* the commit lands rather than computing arithmetic from the diff. A `scripts/check_doc_test_count.py` canary that fails if any current-facing surface's number differs from live would prevent recurrence.

**Suggested canary**: doc-only; optional regression script per the suggested fix.

---

## C13 — Per-module fn callouts missing for `ieee754.hx` + `transcendentals.hx` (restart-50 C8 carry-forward) — LOW

**File**: `C:\Projects\Kovostov-Native\helix_website\HELIX_REFERENCE.md:514-515`
**Bug family**: Per-module stdlib fn-count convention (deferred from restart 50).

**What's wrong**: Restart 50's fix sweep updated 14 of 16 stdlib module entries with the standardized `"N bare fn (+M @-attributed)"` callout (matches the section-510 headline's bare/`@`-attributed convention). The two Numerics & IEEE 754 modules at lines 514-515 are still prose-only.

**Live counts** (verified by `grep -cE '^fn ' helixc/stdlib/*.hx` and `grep -cE '^@' helixc/stdlib/*.hx` against committed HEAD):

```
agi_match.hx         bare=21  at=26   doc=21 bare (+26)  ✓
agi_memory.hx        bare=14  at=21   doc=14 bare (+21)  ✓
agi_search.hx        bare=17  at=26   doc=17 bare (+26)  ✓
agi_world.hx         bare=12  at=16   doc=12 bare (+16)  ✓
autodiff.hx          bare=0   at=40   doc=0  bare (+40)  ✓
autodiff_reverse.hx  bare=23  at=25   doc=23 bare (+25)  ✓
hashmap.hx           bare=38  at=34   doc=38 bare (+34)  ✓
ieee754.hx           bare=6   at=6    doc=(prose only)   MISSING callout
iterators.hx         bare=112 at=47   doc=112 bare (+47) ✓
nn.hx                bare=44  at=14   doc=44 bare (+14)  ✓
option.hx            bare=11  at=12   doc=11 bare (+12)  ✓
result.hx            bare=7   at=8    doc=7  bare (+8)   ✓
string.hx            bare=55  at=41   doc=55 bare (+41)  ✓
tensor.hx            bare=80  at=58   doc=80 bare (+58)  ✓
transcendentals.hx   bare=2   at=52   doc=(prose only)   MISSING callout
vec.hx               bare=13  at=10   doc=13 bare (+10)  ✓
```

Grand totals: **455 bare fn** + **436 @-attributed** = **891 declarations** — matches the headline at line 510 exactly.

**Severity**: LOW — the headline aggregates are correct; only the two per-module callouts are missing the standardized parenthetical. Confirms C8's prediction that restart 51 should "standardize per-module callouts to live `grep -c '^fn '` output".

**Suggested fix** (exact replacement text):

Replace line 514:

> `- `ieee754.hx` — bit-pattern conversions: `__bits_of_f32`, `__f32_from_bits`, `__bits_hi_f64`, `__bits_lo_f64`, `__f64_pack`, `__f64_to_f32`, `__f32_to_f64`, `__f64_to_i32`, `f32_bits_zero`, `f32_bits_one`, `f32_bits_neg`.`

with:

> `- `ieee754.hx` — bit-pattern conversions: `__bits_of_f32`, `__f32_from_bits`, `__bits_hi_f64`, `__bits_lo_f64`, `__f64_pack`, `__f64_to_f32`, `__f32_to_f64`, `__f64_to_i32`, `f32_bits_zero`, `f32_bits_one`, `f32_bits_neg`. 6 bare fn (+0 @-attributed).`

Replace line 515:

> `- `transcendentals.hx` — math + helpers: ... `__sgd_step`/`__momentum_step_v`/`__adam_step` optimizer steps. All transcendentals participate in the autodiff chain rule.`

with:

> `- `transcendentals.hx` — math + helpers: ... `__sgd_step`/`__momentum_step_v`/`__adam_step` optimizer steps. All transcendentals participate in the autodiff chain rule. 2 bare fn (+50 @-attributed).`

**Suggested canary**: None (doc-only). A long-term canary would parse the section, sum the per-module callouts, and assert the result matches `grep -cE '^fn ' helixc/stdlib/*.hx`.

---

## C14 — `Increments 50-N+` open-ended ledger anchors disagree across HELIX_REFERENCE.md — MEDIUM

**File**: `C:\Projects\Kovostov-Native\helix_website\HELIX_REFERENCE.md:59` and `:1569`
**Bug family**: Stale historical anchors inside current-facing prose.

**Current text**:

- Line 59 says: "live count grows with each Stage 35 restart; see `docs/stage35-progress-2026-05-15.md` Increments **50-68+** for the open-ended ledger"
- Line 1569 says: "live count grows with each Stage 35 restart; see `docs/stage35-progress-2026-05-15.md` Increments **50-67+** for the open-ended ledger"

**What's wrong**: Two anchors for the same ledger range, off by one between the same file's Core-Philosophy section (line 59) and Stats-and-Numbers section (line 1569). Per `docs/stage35-progress-2026-05-15.md`, restart 50 is the **Increment 69** entry (visible at line 3720+ of the ledger). So both anchors are stale — the correct open-ended range as of restart 50 is "Increments 50-69+" (or simpler: drop the upper bound and say "Increments 50+").

The mismatch is mild on its own (each is only off by 1-2 from the live ledger end) but the inconsistency *between* the two sites in the same file is a tell that the two sites were not swept together.

**Sibling sweep**: `grep -n "Increments 50-" helix_website/HELIX_REFERENCE.md` returns exactly 2 hits (59, 1569); no other surfaces use this anchor.

**Severity**: MEDIUM — internal cross-section contradiction in the marketing source-of-truth file; off-by-one drift from the live ledger.

**Suggested fix**: Pick one convention and apply uniformly:

- Conservative: replace both with "Increments 50-69+" (matches restart-50 ledger entry).
- Simpler: replace both with "Increments 50+ (live, see the ledger tail)" — removes the upper bound entirely so it stops drifting per restart.

---

## Areas verified clean

- **License-triple wording**: README, QUICKSTART, HELIX_REFERENCE.md, stats_and_facts.md, HANDOFF_FOR_CHATGPT — all softened to Apache 2.0 file-resident; CC-BY 4.0 + CC0 stated policy. No new triple-license overclaims appeared. `docs/PLAN.md`, `docs/research-log.md`, `docs/decisions/2026-05-03-go.md` carry older un-softened wording but are historical/decision files explicitly outside the current-facing surface set.
- **Tool-flag completeness**: `python -m helixc.check --help` output matches the HELIX_REFERENCE.md flag table (lines 1004-1021) and the QUICKSTART.md flag list (lines 66-77) exactly. Documented flags: `--stdlib`, `--no-stdlib`, `--hash`, `--hash-cons`, `--strict`, `--check-only`, `--emit-ast`, `--emit-ir`, `--emit-asm`, `--emit-ptx`, `--emit-proof-obligations`, `--doc`, `--no-color`, `--color`, `--no-opt`, `-O0..-O3`, `-o`, `-l <libname>` and `-l<libname>`, `-W<flag>[=warn|error]`, `-h`/`--help`. `--dump-ast-hashes` correctly attributed to `autodiff_cli` (not `check`). No drift.
- **Dogfood-program count consistency**: README:46 says "6 programs total (5 dogfood + 1 self-improving-agent flagship)" and ROADMAP:17 says "5 dogfood programs/tests ... + a self-improving-agent flagship that composes them (6 programs total: 5 dogfood + 1 flagship)". Both now match the file glob `ls helixc/examples/dogfood_*.hx` (5 files) + `self_improving_agent.hx`. Restart 50 C2/C4 closed cleanly.
- **Stage-numbering note**: `docs/HELIX_V1_FINAL_FEATURES.md` line-3 disclaimer correctly tags sections 2.1-2.7's planning-era Stage 31-37 numbering as predating live numbering and defers to `docs/ROADMAP.md`. ROADMAP is internally consistent (Stage 35 = AI/ML Capability Push, in audit cleanup).
- **HELIX_PURPOSE.md**: dated 2026-05-13, explicitly labeled "policy doctrine, not a status snapshot," defers to README + `stage35-progress-2026-05-15.md` for status. Clean.
- **`agi-features.md`**: Implementation-status table (lines 198-211) and Roadmap (remaining work) table (lines 282-292) both pin to "Stage 35" with the const-fold/CSE/DCE/FDCE shipped disclaimer at the bottom (restart 50 C2 closed). No new staleness.
- **`code_samples.md`** and **`HELIX_REFERENCE.md` Code Samples Gallery preamble**: both carry the restart-50 known-roadmap-snippet list (#9, #10, #15, #16, #17, #18 in code_samples; matching list in HELIX_REFERENCE.md:1153+). Restart 50 C5/C6 fixes still in place.
- **`tutorial.md:6`** disclaimer ("Most examples here are fragment-level — for the loop / array / assignment samples in steps 5 and 6, wrap them in `fn main() -> i32 { ... }`...") — restart 50 C7 fix in place.
- **`trap-ids.md:1-11`** header (trap-ID set authoritative, line refs drift, regenerate via grep) — restart 50 C4 fix in place.
- **`HANDOFF_FOR_CLAUDE.md` Restart 50 Protocol** (lines 336-415) — restart-N-agnostic phrasing per restart 50 C3, no stale references to specific prior restarts at the protocol level.
- **Bootstrap-chain diagram** in HELIX_REFERENCE.md (lines 853-900): self-hosted compiler marked as "roadmap target"; production = Python-hosted `helixc` disclaimer present.
- **hex0 binary size = 299 bytes**: matches `stage0/hex0/hex0.bin` actual; all references across README, QUICKSTART, HELIX_REFERENCE, stats_and_facts are consistent.
- **Compiler-Architecture stdlib list** in HELIX_REFERENCE.md (lines 944-961): lists all 16 modules with single-line purpose tags. Restart 49 C8 fix clean.
- **Stage 35 progress ledger ↔ HANDOFF_FOR_CLAUDE consistency**: ledger Increment 69 = restart 50 fix sweep; handoff line 21 correctly points at Increment 69. Clean-gate count `0/3` matches across both.
- **Stdlib grand totals**: 455 bare fn + 436 `@`-attributed = 891 declarations, matches the HELIX_REFERENCE.md:510 headline and the live grep totals exactly.

---

## Lane verdict

**DIRTY** — 6 findings (2 HIGH, 3 MEDIUM, 1 LOW). All Lane C; no cross-lane spill.

The two HIGH findings are both in the "test counts and restart numbers" bug family:

- **C9**: README mis-attributes the 2,489 count to restart 49 instead of restart 50, contradicting the same paragraph's "restart 50 is the latest" lead.
- **C12**: All 8 current-facing surfaces publish 2,489 but the documented command returns 2,487 at HEAD `7b945fa` (verified twice). Forecast-vs-actual drift introduced when restart 50's count was computed as "prior + N-new" without re-running collect-only after the commit landed.

The three MEDIUM findings are sibling-class:

- **C10** + **C11**: snapshot preambles in `stats_and_facts.md:8` and `HANDOFF_FOR_CHATGPT.md:6` say "restart 49 is the latest" while the body of each file correctly says restart 50 — same kind of partial-sweep drift that the restart-50 fix introduced.
- **C14**: `Increments 50-N+` upper-bound anchor differs between HELIX_REFERENCE.md:59 (says "50-68+") and :1569 (says "50-67+"), and both are off-by-one to off-by-two from the live ledger end (Increment 69 = restart 50).

The one LOW finding closes restart-50's deferred C8: standardize the two missing per-module fn callouts (`ieee754.hx`, `transcendentals.hx`) to the same `N bare fn (+M @-attributed)` format the other 14 modules already use. Exact replacement text included in C13.

Recommend fixing all 6 in the restart-51 fix sweep. After that sweep, sweep `docs/audit-stage35-restart50-laneC.md:7` (which says live collection = 2,479 at the restart-50 commit window) and `docs/stage35-progress-2026-05-15.md` Increment 69 to record the live-vs-forecast reconciliation. Clean-gate counter remains `0/3` until a restart returns 0 findings across all three lanes on the same HEAD.
