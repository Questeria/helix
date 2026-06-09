# Independent Review Handoff — Helix

**For a fresh, strong, independent reviewer.** (Intended reviewer for this pass: **Claude Fable 5**,
released 2026‑06‑09 — but this brief is model‑agnostic and works for any capable reviewer, human or AI.)

**Why you're here:** the entire Helix project + its GPT‑2‑on‑Helix investor demo has just been through
**five consecutive independent adversarial audit rounds** by Claude **Opus 4.8**. The trust core
reproduced clean‑clone all five times and no HIGH / no trust‑or‑correctness break was found after round 1.
You are a *newer, stronger* model. **Your job is to find what Opus missed** — be maximally adversarial,
skeptical, and fail‑closed. Do **not** rubber‑stamp. Cite `file:line` evidence for every finding.

---

## 0. Your mission (one line)

Re‑review **all of Helix** — the from‑raw compiler, the bootstrap ladder, the trust chain, the GPU
emitters, **and** the GPT‑2‑on‑Helix demo — for **any** issue: correctness bugs, trust/honesty gaps,
overclaims, security holes, reproducibility gaps, dead/contradictory docs. Then fix what you safely can
(gated, invariants preserved) and **adjudicate the 2 still‑open items in §7**.

## 1. What Helix is (90 seconds)

A **from‑raw‑binary, self‑hosting compiler**: a 299‑byte hand‑typed `hex0` bootstraps a ladder
(`hex0 → hex1 → hex2 → catm → M0 → cc_amd64 → M2‑Planet → seed`) up to **`seed`** (an Apache‑2.0
C‑subset compiler), which compiles **`kovc`** (the Helix compiler written in Helix). The trust thesis:
*the whole toolchain is rebuildable and verifiable from 299 bytes you could type by hand*, and it
defends against Ken Thompson's "trusting trust" attack via **diverse double‑compile** (gcc, an
unrelated lineage, reproduces a key rung byte‑for‑byte).

On top of that sits the **investor demo**: real, unchanged **GPT‑2** (124M, and 774M/1.5B at scale)
runs inference + generation through `kovc`‑emitted GPU kernels (and a CPU no‑`ptxas` path), output
matched **token‑for‑token** to an independent numpy reference oracle, byte‑reproducible, with a signed
attestation and a **live chat server** — the "bring‑your‑weights **verified execution layer**" pitch.
Lead with **trust, not speed** (it is intentionally slow; ~10 s/token for XL).

## 2. Orient fast — repo + read‑first

- **Repo:** `C:/Projects/Kovostov-Native`, branch `main`, **HEAD `d4b8e32`**. Push remote is private
  (`github.com/Questeria/helix.git`).
- **Read these first** (source of truth, in order):
  1. `README.md` (§Status) and `QUICKSTART.md`
  2. `docs/CLEAN_REPRODUCTION.md` — the from‑raw reproduction method + "where it walls"
  3. `docs/TRUST_CHAIN_CLOSED.md` — the closed‑chain record + disclosed residuals
  4. `docs/CURRENT_HEAD_AUDIT_PACKET.md` — committed proof extract, pinned hashes, the reproducibility **tiers**
  5. `docs/TRUSTED_C_INVENTORY.md` — the trusted‑C surface (Category A from‑raw ladder vs Category B host harnesses)
  6. Demo: `docs/HELIX_GPT2_DEMO_RUNBOOK.md`, `docs/HELIX_GPT2_DEMO_EXECUTION_PLAN.md`, `docs/HELIX_GPT2_CHAT_BACKEND_PLAN.md`
  7. The guide: `docs/book/` (44 files, the full Helix manual) — useful for understanding the language/compiler
- The sibling dir `C:/Projects/Kovostov` is the **AGI framework that drives Claude Code** — *not* part of
  Helix. Ignore it for this review (and never read `C:/Projects/Neptune/api.env`).

## 3. Pinned trust anchors (these must reproduce)

| Anchor | Value | What it proves |
|---|---|---|
| `hex0` size | **299 bytes** | the hand‑typed root of the ladder |
| `seed.bin` sha256 | **`9837db12…`** | the from‑raw ladder rebuilt the seed |
| self‑host fixpoint `K2==K3==K4` | **`0992dddd…`** | kovc compiles itself to a stable fixpoint |
| gcc‑DDC `K1` | **`84363adb…`** | an unrelated compiler reproduces a rung byte‑for‑byte (anti‑trusting‑trust) |

**The one command (CPU‑only, ~1 min, runs on a clean clone):** `bash scripts/reproduce_trust.sh` →
must print `REPRODUCE_TRUST: PASS`. It deletes every pre‑built rung binary, rebuilds `hex0 → seed`,
runs the self‑host fixpoint gate + the gcc diverse‑double‑compile, and asserts all anchors. **Start by
cloning to a throwaway dir and running this yourself** — don't take the claim on faith.

## 4. Hard invariants — you MUST NOT break these (and please re‑verify they hold)

- **Exactly 1 committed `.py`**: `verification/oracle/oracle_train.py` (the capstone verification oracle).
  `git ls-files "*.py" | wc -l` must equal **1**. Do **not** commit a second `.py` (the GPT‑2 demo's
  numpy oracle/importer are *deliberately* uncommitted under gitignored `helix-llm/` — see §6 tiers).
- **Exactly 29 committed `.c/.h`** = **22 from‑raw ladder** (`stage0/*`, incl. `seed.c`, byte‑identical
  to tag `v1.3-release`) **+ 7 Category‑B host harnesses** (`helixc/runtime/*.c` — `cuda_launch.c`,
  `train_transformer.c`, `gpt2_infer.c`, `cpu_host.c`, `gpt2_tok.c`, `gpt2_pack.c`, `gpt2_serve_http.c`),
  all **OUTSIDE** the self‑host fixpoint (proof: `grep helixc/runtime scripts/gate_kovc.sh` → none).
  LOC totals in the docs are *informational/approximate*; the **COUNT (1 / 29) is the load‑bearing fence**.
- **Self‑host fixpoint is byte‑frozen:** `helixc/bootstrap/{kovc,lexer,parser}.hx`,
  `helixc/runtime/train_transformer.c`, and `stage0/helixc-bootstrap/seed.c` are byte‑identical to
  `v1.3-release` and must stay so. The fixpoint sha is `0992dddd`. Any edit that perturbs it is a
  trust break.
- **Claude‑subscription only** — no external AI APIs, no Anthropic API in any shipped/automated path.
- **Discipline:** never force‑push or skip git hooks; never ship red; **gate before commit**; builds are
  **STRICTLY SERIAL** (one `kovc`/GPU build at a time); preserve tags (`v0-pre-k4-full-with-python`,
  `v1.0`, `v1.1`, `v1.2-complete`, `v1.3-release`). The shipped toolchain stays from‑raw + Python‑free.
- **Push** (if you commit) via Windows‑native git, not WSL git (WSL git hangs on credentials).

## 5. What's already verified (don't blindly redo — try to BREAK it)

The committed **gates** are fail‑closed and were each confirmed:
- `scripts/reproduce_trust.sh` — from‑raw ladder + fixpoint + gcc‑DDC (reproduced clean‑clone **5×**).
- `scripts/gate_kovc.sh` — self‑host fixpoint `0992dddd` + corpus **109/0** + check_err **4/0** + PTX regression.
- `stage0/helixc-bootstrap/ddc_crosscheck.sh` — gcc vs M2‑seed `K1` byte‑identical (`84363adb`).
- `scripts/capstone_audit.sh` — a real 2‑layer transformer trains on `kovc`‑emitted GPU kernels within
  **0.0009%** of an independent numpy oracle (negative control catches a corrupted gradient).
- Demo gates: `gpt2_gpu_mvp.sh` (124M GPU), `gpt2_scale.sh` (Large/XL), `gpt2_cpu_parity.sh`
  (CPU no‑`ptxas` block‑0 + full‑logits argmax), `gpt2_pyfree.sh` (C tokenizer/importer bit‑exact),
  `helix_serve_gate.sh` (live server: served XL == offline oracle token‑for‑token 25/25, single‑flight,
  fixpoint clean), `gpt2_demo_attest.sh` (one‑command, fail‑closed, signed attestation).

**Be adversarial about the gates themselves:** does each *actually* fail‑closed? Could a wrong‑but‑
self‑consistent result slip through? Is any "PASS" greppable in two different strength regimes? (Round 4
found exactly such a gap — see §6.)

## 6. The 5 prior Opus audit rounds — findings ALREADY fixed (focus your energy elsewhere)

All committed; re‑verify they're genuinely resolved, but you needn't re‑discover them:
- **Round 1** (`81bf30d`): stale fence count (26→29), an `f64` oracle overclaim (oracle is **fp32**),
  "stub" comments on wired code, stale doc LOC.
- **Round 2** (`a3e327f`): live‑demo reproducibility (parameterized the hardcoded `/home/legoa` paths via
  `HELIX_SRC/WORK/XL_WEIGHTS` + documented obtaining XL weights); the CPU "ZERO trusted above the seed"
  claim scoped to **arithmetic** + a **Shared‑TCB** residual added; `helix_serve_gate.sh` G1 hardened;
  the SSE telemetry‑purity leak (`event: message`) closed.
- **Round 3** (`d0074c7`): **reproducibility tiering** (Tier A vs Tier B — see below); `QUICKSTART.md`
  rewritten to v1.3 reality; LOC reframed (COUNT is the invariant); GPU block‑0 metric relabeled
  *floored* max‑abs‑rel.
- **Round 4** (`d4b8e32`): live per‑layer timing made honest (`--timing 1` + real host‑clock label, no
  false "CUDA‑event"); `helix_serve_gate.sh` G1 now **hard‑requires PRIMARY mode** (rejects a
  helix‑vs‑helix fallback masquerading as oracle token‑for‑token); server **`SO_RCVTIMEO` + `MAX_CONN`**
  (slow‑loris / conn cap).

**Reproducibility tiers (important framing to keep honest):** **Tier A** = the from‑raw *trust core* is
FULLY third‑party‑reproducible from the committed repo alone (`reproduce_trust.sh` + the CI), no weights
or oracle needed. **Tier B** = the GPT‑2 *demo legs* additionally need external artifacts under
gitignored `helix-llm/`: the public GPT‑2 weights (HuggingFace `openai-community/gpt2[-xl]`, MIT →
convert via committed `gpt2_pack.c`) **and** an independent numpy oracle (kept uncommitted to preserve
the 1‑`.py` fence; a reviewer may supply their own). Only Tier A is repo‑only‑reproducible — the
materials must never claim otherwise.

## 7. STILL‑OPEN (round 5) — please adjudicate or fix

Both are **MEDIUM, real but contained** (loopback‑only / mock‑labeled), found by Opus round 5 but
**not yet fixed** (left for you / the owner). Verify, then fix or argue why not:
1. **`helixc/runtime/gpt2_serve_http.c` — missing `SO_SNDTIMEO`.** Round 4 added `SO_RCVTIMEO` (read
   side) but there's no send‑side timeout, and `handle_generate` holds the single‑flight GPU mutex for
   the whole ~195 s generation while `write_all()` blocks. A client that gets the `200` then **stops
   reading** fills the socket buffer → `write_all` blocks forever → the mutex is never released → every
   other request gets `409` permanently until that client disconnects. Contained to `127.0.0.1` +
   recoverable on disconnect, but a real single‑GPU starvation vector for the web‑exposed goal. **Fix:**
   set `SO_SNDTIMEO` on the accepted fd (mirror the `SO_RCVTIMEO` block) so a stalled write bails and
   releases the mutex. (The design doc also promised a `: ping` heartbeat that isn't implemented.)
2. **`demo/index.html` — mock chat speed vs reality.** The *mock* playground streams at
   `MOCK_SECONDS = n*1.3 + 0.8` (~1.3 s/token) and a code comment calls 1.3 s/token "honest XL‑style" —
   but **real XL is ~9.8 s/token** (`scripts/_gate_run.log`: `seconds:195.513, tok_per_s:0.102` for 20
   tokens). It is clearly PREVIEW/MOCK‑labeled (not fraud), but ~7.5× optimistic on a *trust‑not‑speed*
   demo. **Fix:** make the mock cadence reflect ~10 s/token (or relabel the comment "compressed for
   preview, not representative of live XL latency"), and add one line to the runbook + `serve_chat_demo.sh`
   banner: live XL ≈ 10 s/token (~3 min for 20 tokens) — the XL chat is the *verifiability* flex, not a
   speed flex; use the 124M MVP path for a snappy live demo.

## 8. Where to dig hardest (the convergence frontier + the under‑scrutinized parts)

The 5 Opus rounds **settled** trust/parity/overclaim (LENS 1/2/4 → PASS); the *fertile* surfaces were the
**live server** and the **completeness/fresh‑investor** lens. Push hardest on:
- **`gpt2_serve_http.c`** (660 lines, will be web‑exposed): concurrency/back‑pressure (see §7.1), CSRF on
  the state‑changing endpoints, any unbounded allocation, the worker‑pipe lifecycle, header parsing.
- **Telemetry honesty:** every SSE event must wrap *real* work (no decorative/fabricated events); the
  served output must be the real forward, not a replay. Verify against `gpt2_infer.c`'s `--serve` path.
- **Overclaim sweep** across `README`, `QUICKSTART`, `docs/HELIX_GPT2_DEMO_RUNBOOK.md`, the dashboard
  (`demo/dashboard.html`), and the emitted attestation — any claim an investor could check and find false;
  any inconsistency between the four surfaces.
- **BROADEN beyond the demo** (the prior campaign was demo‑focused): give the **compiler `kovc.hx`**, the
  **bootstrap ladder** (`stage0/*`), the **GPU PTX emitters**, and the **stdlib** a fresh deep read — the
  v1.3 trust *core* is well‑audited, but the compiler internals + the language semantics got less fresh
  adversarial scrutiny *this* campaign. Look for soundness gaps, missed‑edge codegen, and any place a
  claim in `docs/book/` overstates what the compiler actually does.
- **The honest residuals** (in `TRUST_CHAIN_CLOSED.md` / the runbook): are they *complete*? Shared TCB
  (OS/gcc/libc/CPU + the audited `seed.c`), PTX‑not‑SASS (closed `ptxas`/driver below PTX on the GPU
  path; the CPU path has no such boundary), fp32‑only, single GPU `sm_86`, the oracle shares the GPT‑2
  *spec* (independent implementation, not independent specification). Is anything *un*disclosed?

## 9. How to run the review

- **Reproduce the core** (Tier A): clone to a throwaway dir, `bash scripts/reproduce_trust.sh` → PASS.
- **Re‑run the gate suite** as needed (serial, on a CUDA host for the GPU ones): the `scripts/*.sh` above.
- **Environment notes (WSL/Windows):** in `wsl.exe bash -c '…'`, never assign a `/mnt` path to a
  variable (MSYS empties it) and never use `< /path` redirection (mangled) — write nontrivial scripts to
  a file, `tr -d '\r'`, then `bash /tmp/x.sh`; invoke `.sh` with `MSYS_NO_PATHCONV=1`. Build on WSL
  **ext4** (DrvFs is ~75× slower). Validate by **non‑empty output + sha**, not `rc==0`, for `kovc`/`seed`
  (they return the output byte‑count as exit status).
- If you're in Claude Code: a ready **5‑lens adversarial audit workflow** already exists at
  `…/workflows/scripts/demo-5-audit-finale-wf_4cfaf0bc-ae0.js` (trust‑chain / parity / live‑server /
  overclaim / completeness) — re‑running it under your model is a stronger, apples‑to‑apples pass than
  the Opus rounds. Or just drive the lenses yourself.

## 10. What to produce

A **calibrated, evidence‑cited findings report**: each finding as `{severity HIGH|MEDIUM|LOW, area,
issue, file:line evidence, fix}`, plus a clear verdict on (a) whether the from‑raw/trusting‑trust claim
holds, (b) whether the demo's parity/honesty claims hold, and (c) anything the Opus rounds missed. Then
**fix what you safely can** — preserving every §4 invariant, gating before commit, never shipping red —
and **adjudicate the 2 open items in §7**. If you find a genuine HIGH, that's the win; the prior reviewer
will be glad you caught it.

*— Handoff prepared by Claude Opus 4.8 (1M), HEAD `d4b8e32`, after 5 prior adversarial audit rounds.*
