# Stage 35 Restart 50 — Lane C (Docs / Status / Release) Audit

**Audit date**: 2026-05-16
**HEAD**: `f0ab654` (handoff-only) over `6c555a4` (restart-49 fix sweep)
**Mode**: STRICTLY READ-ONLY. Findings only; fixes belong in a separate sweep.
**Ground-truth checks**:
- `python -m pytest helixc/tests --collect-only -q` → **2,479 tests collected** (matches restart-49 expectation; no test-count drift)
- `python -m helixc.check --help` → flag list matches HELIX_REFERENCE.md's table (no new flags to sync)
- `ls helixc/stdlib/*.hx | wc -l` → **16 modules** (matches docs)
- `stat -c "%s" stage0/hex0/hex0.bin` → **299 bytes** (matches docs)
- `ls helixc/examples/dogfood_*.hx` → **5 files** + `self_improving_agent.hx` → **6 programs total**

## Summary

| Severity | Count |
|----------|-------|
| HIGH | 0 |
| MEDIUM | 2 |
| LOW | 3 |
| **Total** | **5** |

**Verdict**: **DIRTY** (5 findings; one is a residual sibling of an already-fixed bug, two are internal contradictions between docs the restart-49 sweep introduced.)

---

## Family 1 — "23+ silent-corruption" residual instance (restart-49 C7 sibling miss)

### C1 (MEDIUM)

**File**: `C:\Projects\Kovostov-Native\helix_website\HELIX_REFERENCE.md`
**Line**: 59
**Current text**:
> "23+ silent-corruption bugs were found and fixed during development; their repo-local audit docs include reproducers and status, and a future `/audits` website page should expose them publicly."

**What's wrong**: Restart 49 C7 reframed the OTHER instance of this claim (the stats block at line 1548, now reading "Dozens of silent-corruption defects (live count grows with each Stage 35 restart; see ... Increments 50-67+ ...)") but missed this Core-Philosophy-section sibling. The two phrasings now contradict each other inside the same file: line 59 anchors at 23, line 1548 says "dozens, growing with each restart." Per the Stage 35 progress ledger and the 50+ restart-level fix sweeps, the live count is well past 23, so the line-59 wording also understates by an order of magnitude.

**Sibling sweep**: `grep -n "silent-corruption" helix_website/HELIX_REFERENCE.md` returns exactly 3 hits (lines 57 heading, 59 prose, 1443 visual-identity callout, 1548 stats block). Only 1548 carries the reframed wording; 59 still has "23+ ... were found and fixed."

**Suggested fix**: replace "23+ silent-corruption bugs were found and fixed" with the same reframed wording used at line 1548 (e.g., "Dozens of silent-corruption defects have been found and disclosed during development; the live count grows with each Stage 35 restart"). Then re-check that lines 59 + 1548 + 1443 agree.

---

## Family 2 — Dogfood-program count: README vs ROADMAP contradiction

### C2 (MEDIUM)

**Files**:
- `C:\Projects\Kovostov-Native\README.md` line 46
- `C:\Projects\Kovostov-Native\docs\ROADMAP.md` line 17

**README.md line 46**: "6 dogfood programs running real ML in Helix-emitted binaries:" — and then lists 6 items where item #6 is "Self-improving agent (flagship, composes everything)." So README treats the flagship as one of the 6 dogfood programs.

**ROADMAP.md line 17**: "6 dogfood programs/tests running real gradient descent + a self-improving-agent flagship that composes them (see `helixc/examples/dogfood_*.hx` and `helixc/examples/self_improving_agent.hx`)" — this reads as "6 dogfood + 1 flagship = 7 things," contradicting the README.

**Ground truth**: `ls helixc/examples/dogfood_*.hx` → **5 files** (`dogfood_01_one_param.hx`, `dogfood_02_linreg.hx`, `dogfood_03_affine.hx`, `dogfood_04_xor_relu.hx`, `dogfood_05_binary_classifier.hx`). Plus `helixc/examples/self_improving_agent.hx`. Total: **6 programs (5 dogfood + 1 flagship)**.

Either interpretation could be correct, but they can't both be — README treats flagship as a dogfood (6 total, ALL dogfood), ROADMAP treats it as separate (6 dogfood + 1 flagship = 7). Reality is 5 dogfood + 1 flagship. ROADMAP's "6 dogfood programs" specifically miscounts the actual file glob.

**Note**: This is restart 49 C4's own change — the ledger says "C4: `docs/ROADMAP.md` line 17 corrected from '5 dogfood programs' to '6 dogfood programs + a self-improving-agent flagship'." That fix introduced the contradiction by changing "5 dogfood programs" (which was correct against `dogfood_*.hx`) to "6 dogfood programs + flagship" (which double-counts the flagship since README treats it as one of 6).

**Suggested fix**: pick one convention and apply it to BOTH README and ROADMAP. Option A — ROADMAP back to "5 dogfood programs/tests ... + a self-improving-agent flagship that composes them" (matches file glob). Option B — both files say "6 programs total: 5 dogfood + 1 self-improving-agent flagship that composes them" (explicit).

---

## Family 3 — HANDOFF_FOR_CLAUDE.md restart-50-protocol internal contradiction

### C3 (MEDIUM)

**File**: `C:\Projects\Kovostov-Native\HANDOFF_FOR_CLAUDE.md`
**Lines**: 300-303
**Current text**:
> "The bar continues to rise — restarts 46/47/48/49 each closed 12, 17, 13, 11 findings respectively, so 4 consecutive restarts have each found more issues than the prior."

**What's wrong**: The numbers given are 12 → 17 → 13 → 11, which is up-then-down-then-down, not monotonically rising. The conclusion "each found more issues than the prior" is directly contradicted by the data in the same sentence (13 < 17, 11 < 13). The trend is the opposite — findings are decreasing after a restart-47 peak.

**Suggested fix**: replace with the actual trend, e.g., "restarts 46/47/48/49 each closed 12, 17, 13, 11 findings respectively — the campaign has settled into a high-throughput run-rate after the restart-47 peak, but is not yet trending to zero. Restart 50 should continue the bug-family sweep until a single restart returns 0 findings on the same HEAD." This both states the data accurately and reframes the bar-rising language as "settled run-rate, not yet zero."

---

## Family 4 — HANDOFF_FOR_CLAUDE.md cross-references to prior restarts use stale advancement language

### C4 (LOW)

**File**: `C:\Projects\Kovostov-Native\HANDOFF_FOR_CLAUDE.md`
**Lines**: 367-375
**Current text** (excerpt):
> "If restart 48's audit returns 0 findings across all three lanes on the same HEAD (`4ba725f` or its newest descendant), the clean-gate counter advances to `1/3`. Restart 49 then starts from that same HEAD; restart 50 if 49 is also clean; three consecutive clean gates close Stage 35."
>
> "If all three lanes are clean on the same HEAD and support checks pass, restart 47 becomes clean gate `1/3`."

**What's wrong**: The handoff was rewritten for restart 50 (per commit `f0ab654`: "Add Claude handoff for Stage 35 restart 50") but this block still refers to "restart 48's audit," "restart 49 then starts," and "restart 47 becomes clean gate." These are stale references that should now say restart 50/51/52. Restart 49 has already happened (it's the handoff's anchor commit), so "Restart 49 then starts from that same HEAD" is incoherent in the restart-50 context.

The restart-50 protocol section above this block (lines 296-365) correctly references restart 50. Only the closing paragraphs at 367-375 carry the stale restart-48/49 wording.

**Suggested fix**: replace "restart 48" → "restart 50", "restart 49" → "restart 51", "restart 50" → "restart 52", and "restart 47" → "restart 50" in the lines-367-375 paragraphs. Also reconcile the duplicate "becomes clean gate 1/3" sentences (lines 367-370 and 372-375 say nearly the same thing twice with different restart numbers).

---

## Family 5 — HELIX_REFERENCE.md stdlib per-module function counts are stale

### C5 (LOW)

**File**: `C:\Projects\Kovostov-Native\helix_website\HELIX_REFERENCE.md`
**Lines**: 510-545 (Standard Library section, written at restart 48 C7)

**What's wrong**: Several per-module function counts in this section do not match ground truth. Counts using "~" hedge so the severity is LOW, but a few are off enough that "~" cannot bridge them:

| Module | REFERENCE.md says | Bare-fn ground truth | Bare + @-attributed | Off by |
|---|---|---|---|---|
| `tensor.hx` | ~113 functions | 80 | 138 | ambiguous (between bare and total) |
| `nn.hx` | ~75 functions | 44 | 58 | overstates by ~17 even vs total |
| `agi_match.hx` | ~31 functions | 21 | 47 | between bare and total |
| `agi_memory.hx` | ~27 functions | 14 | 35 | between bare and total |
| `agi_search.hx` | ~40 functions | 17 | 43 | matches total within hedge |
| `agi_world.hx` | ~19 functions | 12 | 28 | between bare and total |
| `hashmap.hx` | ~41 functions | 38 | 72 | matches bare within hedge |
| `option.hx` | ~5 functions | 11 | 23 | **understates by 6 even vs bare** |
| `result.hx` | ~7 functions | 7 | 15 | matches bare exactly |
| `iterators.hx` | ~112 functions | 112 | 159 | matches bare exactly |
| `string.hx` | ~55 functions | 55 | 96 | matches bare exactly |
| `vec.hx` | ~13 functions | 13 | 23 | matches bare exactly |
| `autodiff_reverse.hx` | ~35 functions | 23 | 48 | between bare and total |

The intro line at 510 says "~455 bare `fn` declarations (644 including `@attribute`-prefixed declarations)." The 455 bare-fn total matches ground truth exactly. The 644 attributed total does NOT match — actual bare + every `@`-prefixed line is 891; bare + single-line-attributed (`@pure\nfn`) is 700; no clean computation reaches 644.

The per-module counts mix two counting conventions (bare-only vs bare+attributed) inconsistently, and `option.hx` is off in either direction.

**Suggested fix**: re-run `for f in helixc/stdlib/*.hx; do echo "$f: bare=$(grep -c '^fn ' "$f"); attributed=$(grep -c '^@' "$f")"; done` and rewrite the per-module counts with one explicit convention. Either pick bare-fn only and drop the `@`-attributed total, or report both numbers explicitly per module. Also replace the 644 figure with the correct bare+attributed total (700 if single-`@`, 891 if every `@`-line).

---

## Areas verified clean

- **Test count consistency**: Live `pytest --collect-only` = 2,479. README, QUICKSTART, HANDOFF_FOR_CLAUDE (Suggested Verification block), HANDOFF_FOR_CHATGPT, stats_and_facts.md, HELIX_REFERENCE.md (twice) all say 2,479. No drift across the eight surfaces.
- **Restart-number references** in current-facing docs: README/QUICKSTART/HANDOFF_FOR_CHATGPT/stats_and_facts/HELIX_REFERENCE.md all say "restart 49." HANDOFF_FOR_CLAUDE is written FROM restart 49 and points at restart 50. Internally consistent (modulo C4 above).
- **License-triple wording**: README, QUICKSTART, HELIX_REFERENCE.md, stats_and_facts.md, HANDOFF_FOR_CHATGPT — all softened to Apache-2.0-file-resident, CC-BY-4.0 + CC0 stated-policy. No new triple-license overclaims appeared. `docs/PLAN.md` and `docs/research-log.md` carry the older un-softened phrasing but those are historical/research files, not current-facing.
- **Tool-flag completeness** (`helixc.check --help` ↔ HELIX_REFERENCE.md flag table): every flag in --help is in the table; no new flags since restart 47's sync. `-Wad=` and `-Wdeprecated=` both present per restart 46 B7 and restart 47 B5.
- **Stage-count cross-references**: `docs/HELIX_V1_FINAL_FEATURES.md` enumerates Stages 31-65 (35 distinct, actually consecutive — the "not strict consecutive sequence" hedge in HELIX_REFERENCE.md and helix_website/README.md is slightly loose but not actually wrong since `Stage 65+` opens the door to non-enumerated outliers; not worth flagging at restart-50 strictness).
- **Bootstrap-chain diagram** in HELIX_REFERENCE.md (lines 856-900): self-hosted compiler still marked as "roadmap target." Production-compiler-is-Python-hosted disclaimer present at lines 896-898.
- **"30 ready-to-use snippets" / sample count claims**: `code_samples.md` actually has 30 sections (`grep -cE '^## [0-9]+\.' code_samples.md` = 30). The "30 draft snippets" reference in `helix_website/README.md` line 11 is accurate.
- **HELIX_PURPOSE.md and HELIX_V1_FINAL_FEATURES.md scope claims**: both consistent with ROADMAP.md (live Stage 35 is AI/ML Capability Push; planning-era Stage 31-34 numbering is disclaimed). The Stage 65+ ceiling reference is consistent across these docs.
- **hex0 binary size = 299 bytes**: matches `stage0/hex0/hex0.bin` actual size. All `299-byte` / `299 bytes` references across README, QUICKSTART, HELIX_REFERENCE.md, stats_and_facts.md correctly cite this.
- **HELIX_REFERENCE.md Compiler-Architecture stdlib list (lines 944-961)**: lists all 16 modules with single-line purpose tags. Restart 49 C6 fix is clean.
- **helix_website/README.md `/learn` softening** (restart 48 C9): line 22 reads "Planned beginner tutorial sequence (lesson count and curriculum TBD; no shipped content yet)." No regression.
- **Stage 35 progress ledger ↔ handoff consistency**: ledger Increment 68 says "in flight at commit time, expected 2,479 (was 2,466 + 13 new)." Live collection confirms 2,479. Ledger's clean-gate count `0/3` matches HANDOFF_FOR_CLAUDE.md line 14 `0/3`.

---

## Lane verdict

**DIRTY** — 5 findings (0 HIGH, 2 MEDIUM, 3 LOW). The two MEDIUMs are both *new-since-restart-49* defects (C1 is a sibling restart 49 C7 missed; C2 is a contradiction restart 49 C4's change introduced). C3 is also new (restart 50 handoff drafted with the incoherent "bar continues to rise" framing). C4 + C5 are residual stale text not caught by restart 48/49.

Recommend fixing all 5 in restart 50's fix sweep before declaring any clean gate. Clean-gate counter remains `0/3`.
