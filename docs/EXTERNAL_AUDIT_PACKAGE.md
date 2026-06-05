# Helix — External Cross-Model Audit Package (independent review by ChatGPT, with read access)

**Purpose.** Helix was built *and* internally audited by Claude-family agents. Five independent
in-family adversarial audits passed — but same-family auditors can share a blind spot the builder
also had. This package hands the whole trust claim to a **different model lineage (ChatGPT)**, which
has **read-only access to the entire repository**, to close that gap. ChatGPT is asked to be
adversarial, not affirming, and to **verify against the actual files**, not just trust this summary.

**How to use.**
1. Confirm ChatGPT has read-only access to the Helix repo/folder (locally `C:/Projects/Kovostov-Native`,
   remote `github.com/Questeria/helix.git`, branch `main`, tip `660905a`).
2. Paste everything between `===PROMPT BEGINS===` and `===PROMPT ENDS===` into ChatGPT (strongest model,
   reasoning mode on).
3. Forward ChatGPT's findings back. **Any real Critical/Major finding → fix (gated by the universal
   build gate) → re-verify, BEFORE tagging `v1.3-release`.** This cross-model review is a release gate.

---

```
===PROMPT BEGINS===

ROLE
You are an independent, adversarial systems-and-security reviewer auditing the TRUST CHAIN of a
self-hosting compiler called "Helix." You are deliberately a DIFFERENT AI model from the one that
built and internally audited Helix; your value is to catch errors, overclaims, hidden assumptions,
and blind spots that the builder's own model family could have shared and missed. Do NOT flatter or
rubber-stamp. Default to skeptical. If after a genuine adversarial effort you find nothing wrong, say
so — but earn it. Reward yourself for finding a real problem, not for reassurance.

YOU HAVE READ-ONLY ACCESS TO THE ENTIRE REPOSITORY — USE IT.
Do NOT just trust this package's summary. OPEN THE ACTUAL FILES and verify each claim against the real
source, docs, scripts, and logs. This package tells you WHAT is claimed and WHERE to look; your job is
to confirm-or-break each claim against the repo.
 - VERIFY DIRECTLY whatever read access allows: count the committed Python files; read seed.c and the
   bootstrap source; read the build gate; check the trusted-C inventory against the real file list;
   read the language spec, the trust-record doc, and the audit logs for overclaims; recompute the
   SHA-256 of any committed FILE you can hash (e.g. seed.c against the committed seed.sha256, corpus
   programs) and compare.
 - WHAT READ-ONLY ACCESS LIKELY CANNOT DO: rebuild the from-raw ladder (hex0→seed→kovc), run the
   self-host fixpoint, or run the GPU gate — those need a WSL + CUDA + ptxas toolchain you don't have.
   For anything requiring a build/run, treat the stated result as a CLAIM, and give the EXACT commands
   a machine-equipped reproducer must run to confirm it.
 - LABEL EVERY finding [VERIFIED-FROM-REPO] / [CLAIM-COULDNT-BUILD] / [NEEDS-REPRODUCTION]. A file that
   is missing, mis-described, contradicts its claim, or an UNDISCLOSED file/dependency/overclaim you
   find = a finding.

KEY FILES TO READ (start here, but you may read anything in the repo):
 - stage0/helixc-bootstrap/seed.c  + seed.sha256        — the irreducible trust root + its hash pin
 - stage0/**                                            — the from-raw bootstrap ladder (hex0…M2-Planet)
 - helixc/bootstrap/kovc.hx, parser.hx, lexer.hx        — the Helix compiler source (self-hosted)
 - scripts/gate_kovc.sh                                 — the UNIVERSAL build gate (read its logic)
 - docs/TRUST_CHAIN_CLOSED.md                           — the owner's honest trust record + residuals
 - docs/HELIX_V1_LANGUAGE_SPEC.md                       — the as-built language spec (+ v1.3 deltas)
 - docs/TRUSTED_C_INVENTORY.md                          — the 24-file / 15600-LOC trusted-C inventory
 - docs/K_DDC_BROADENED.md                              — the diverse-double-compile story
 - docs/HELIX_V1_DEFINITION_OF_DONE.md, HELIX_V1_1_HARDENING.md, HELIX_COMPLETION.md, HELIX_V1_3.md
                                                        — the campaign charters / definitions of done
 - .stage33-logs/finale_audit1b_results.txt … finale_audit5b_results.txt  — the 5 finale audit logs
                                                        (the re-derived evidence; cross-check them)
 - .stage33-logs/hxcc_state.txt                         — the FULL campaign log: it openly records a
                                                          silent f16 bug the finale caught, a hash
                                                          FABRICATION by one builder agent (caught +
                                                          corrected pre-commit), and the orchestrator's
                                                          own mistakes. Read it to judge whether the
                                                          PROCESS is honest, or whether the warts hide
                                                          something worse.
 - verification/py_witness/v11_interp/interp.py         — the SECOND DDC witness (audit its independence)
 - verification/oracle/oracle_train.py                  — the ONE committed .py (the numpy oracle)

────────────────────────────────────────────────────────────────────────
THE CLAIMS TO VERIFY (against the repo)
────────────────────────────────────────────────────────────────────────
1. FROM A HAND-TYPED ROOT. hex0 (~299 hand-authored hex bytes) → hex1 → hex2 → catm → M0 → cc_amd64
   → M2-Planet → seed (Apache-2.0 C-subset compiler, seed.c) → kovc (the Helix compiler, in Helix).
   Each rung built ONLY by the prior; no pre-built binary trusted. Only seed.c + seed.sha256 committed
   (seed.bin gitignored, rebuilt). Claimed seed.bin SHA-256:
       9837db12752a22159ca75a533910bc0d7b9afb35df9b9963f256b7b1b915c9bb
   VERIFY: read seed.c (is it a plausible minimal compiler with no hidden network/eval/backdoor?);
   sha256 seed.c and compare to seed.sha256's intent; confirm seed.bin is gitignored not committed.
   (Rebuilding the ladder is [NEEDS-REPRODUCTION].)

2. SELF-HOST FIXED POINT. seed → K1 → K2 → K3 → K4 with K2==K3==K4 byte-identical. Claimed:
       K1 = 84363adb…346abba (697425 B); K2==K3==K4 = 0992dddd…4bd20f (698392 B).
   VERIFY: read how scripts/gate_kovc.sh computes + asserts the fixpoint (real cmp? escape hatches?).
   The actual byte-equality is [NEEDS-REPRODUCTION] — give the commands.

3. DIVERSE DOUBLE-COMPILE (anti "trusting-trust", per Thompson/Wheeler). gcc (zero shared lineage with
   the M2-Planet seed) compiles the same seed.c and its seed produces a byte-identical K1. SCOPE BOUND
   (disclosed): byte-identical DDC covers SEED→K1; the v1.1 language surface (generics/traits/closures/
   turbofish/wide-field/bf16) is cross-checked only BEHAVIORALLY by a second interpreter (residual R3).
   VERIFY: read docs/K_DDC_BROADENED.md + the DDC scripts; judge the logic. Re-running the DDC is
   [NEEDS-REPRODUCTION].

4. PYTHON-FREE FENCE. Exactly ONE committed .py: verification/oracle/oracle_train.py (a numpy oracle,
   never invoked by the compiler/runtime). VERIFY DIRECTLY: run `git ls-files "*.py"` → expect 1.
   NOTE — the repo ALSO contains gitignored .py under verification/py_witness/ (the fenced interpreter
   witnesses) + a gitignored seed.bin; the fence claim is about COMMITTED files. JUDGE FOR YOURSELF
   whether gitignoring the witnesses is a legitimate fence (they're auditor tools, not the toolchain)
   or a dodge — and whether ANY committed file in the actual compile/run path is Python.

5. REAL CAPABILITY (capstone). A ≥2-layer transformer trains end-to-end on kovc-EMITTED GPU kernels
   (PTX-as-text) on an RTX 3070 (sm_86), converging to ~0% loss difference vs the independent numpy
   oracle (bar was 2%); the oracle reads only shared INITIAL weights, not Helix's trajectory. VERIFY:
   read verification/oracle/oracle_train.py + the training harness — is the oracle genuinely
   independent, and is "trains a real neural network" honestly supported? Running it is
   [NEEDS-REPRODUCTION].

6. GPU PERFORMANCE (stated with its honest fraction, never "parity"). G1 4.56 TFLOP/s (≈56% cuBLAS-f32);
   G2 5.445 (≈67.5%); G3 TF32 5.35 (≈50-54% cuBLAS-TF32). Capstone end-to-end speedup 7.03–8.70× (the
   ≥10× target NOT met; Amdahl-bound). VERIFY: read the perf docs; check every number is paired with
   its scope; flag any bare "parity"/"beats cuBLAS". The measurements are [NEEDS-REPRODUCTION].

7. TRUSTED-C SURFACE. 24 committed .c/.h, 15600 LOC; seed.c (1368) = single irreducible root; the rest
   = vendored bootstrap ladder (compiled-from-raw) + GPU host launcher (cuda_launch.c / train_transformer.c,
   makes the closed NVIDIA driver/cuBLAS calls). VERIFY DIRECTLY: `git ls-files "*.c" "*.h"` → expect 24;
   confirm EVERY file is in docs/TRUSTED_C_INVENTORY.md (no hidden trusted C); spot-check LOC.

8. INTERNAL VERIFICATION (what you double-check). v1.3 shipped V1–V7; then a FINALE of 5 consecutive
   clean independent adversarial audits, each rebuilt-from-raw (lenses: silent-residual / type-
   correctness / DDC-independence / trusted-C-accuracy / overclaim-sweep). VERIFY: read the 5
   finale_audit*b_results.txt logs — are they genuine reproductions with re-derived numbers, or
   hand-waving? Do their hashes agree with each other and with claims 1–3?

────────────────────────────────────────────────────────────────────────
DISCLOSED RESIDUALS (audit for completeness + honesty — what's MISSING?)
────────────────────────────────────────────────────────────────────────
 R1. GPU 50–67% of cuBLAS, NOT parity.            R2. Capstone 7–8.7×, not ≥10× (Amdahl).
 R3. Byte-identical DDC = seed→K1 only; the v1.1 surface is BEHAVIORAL-only (a second zero-kovc-lineage
     interpreter, 44/44), NOT a byte-identical second-compiler reproduction.
 R4. f16 arithmetic computes (F16C) + is bit-exact-gated, but is NOT covered by the behavioral witness.
 R5. Hand-auditable hex0→PTX; below PTX relies on closed NVIDIA ptxas + driver (the one trusted-once GPU
     boundary). CPU path is all-the-way-down.
 R6. Only sm_86 (RTX 3070 Laptop) tested; no cross-arch/multi-vendor.
 R7. Documented-but-unenforced language bounds: borrows/aliasing, const/static, module privacy, match
     exhaustiveness; closure captures i32-only (wider fail closed at compile time).

────────────────────────────────────────────────────────────────────────
DIRECT-VERIFICATION CHECKLIST (do these against the repo)
────────────────────────────────────────────────────────────────────────
 1. `git ls-files "*.py"` == 1? Is any COMMITTED file on the compile/run path Python? (claim 4)
 2. Read seed.c end-to-end: minimal C-subset compiler? any network/eval/exec/backdoor/odd dependency?
 3. `git ls-files "*.c" "*.h"` == 24, and all present in TRUSTED_C_INVENTORY.md? Any hidden trusted C?
 4. interp.py: imports only sys+struct? Its only open() the .hx source? No subprocess/socket/reading any
    kovc binary/ELF/stdout? (the DDC-independence claim, R3) — is it genuinely an independent witness?
 5. gate_kovc.sh: does it actually enforce fixpoint byte-equality + the full corpus + the GPU-PTX
    regression, with NO weakened checks, skips, or escape hatches? Could a regression pass silently?
 6. The 5 finale_audit*b_results.txt: real re-derivations? Mutually consistent hashes? Any sign of
    copy-paste/fabrication? Cross-check against hxcc_state.txt's account of the f16 bug + the fabrication.
 7. Spec + TRUST_CHAIN_CLOSED.md: grep for "parity / proven / complete / closed / byte-identical / 100%"
    — each scoped + true? Any claim the code doesn't support?
 8. Read hxcc_state.txt: is the disclosed history (f16 bug, fabrication, orchestrator mistakes) the FULL
    story, or does it hint at something undisclosed?

────────────────────────────────────────────────────────────────────────
ADVERSARIAL TASKS
────────────────────────────────────────────────────────────────────────
A. TRUST-CHAIN LOGIC. Is hex0 → seed → kovc → fixpoint → DDC sound? Where could a compiler-level
   corruption survive? Stress the DDC: does gcc-compiles-seed.c → byte-identical-K1 really defeat
   trusting-trust here, or do both "independent" routes still share a libc/linker/assembler/OS/CPU/
   microcode? NAME the residual trusted computing base the DDC does NOT eliminate.
B. THE "FROM RAW" CLAIM. Is ~299 hand-typed bytes a credible root, or is trust merely PUSHED DOWN to
   the assembler/linker/kernel/hardware that runs hex0? Claimed vs. what a skeptic should accept.
C. DDC SCOPE GAP (R3). How much weaker is "a behavioral interpreter the same project wrote" than a
   byte-identical second compiler? Could interp.py and kovc share a bug? Is the framing honest?
D. OVERCLAIM / HONESTY SWEEP. Across claims 1–8 + R1–R7 + the actual docs: anything overstated,
   ambiguous, or quietly load-bearing? Is the residual list COMPLETE — what caveat is MISSING?
   Scrutinize the capstone "real neural network", the "~0% vs independent oracle", and the perf framing.
E. GPU TRUST BOUNDARY (R5). Is conceding closed ptxas/driver below PTX sufficient, or does it undercut
   the "from-raw, no-trusted-binary" thesis more than admitted?
F. PROCESS CREDIBILITY. The finale "caught a real bug + a fabrication, then passed." From the logs, does
   that genuinely raise confidence (verification that bites), or could it be theater? What distinguishes them?
G. REPRODUCTION PLAN. The checks you'd run WITH a build machine, in priority order — exact commands /
   files / hashes — that most efficiently confirm-or-break the trust claim.

────────────────────────────────────────────────────────────────────────
OUTPUT FORMAT
────────────────────────────────────────────────────────────────────────
1. ONE-PARAGRAPH VERDICT: is the trust story sound + honestly stated, given what you verified from the repo?
2. FINDINGS by severity — CRITICAL / MAJOR / MINOR. Each: the claim, the file/line evidence, why it's
   suspect, the fix-or-resolution, and a label [VERIFIED-FROM-REPO] / [CLAIM-COULDNT-BUILD] /
   [NEEDS-REPRODUCTION].
3. MISSING RESIDUALS: caveats the owner should disclose but didn't.
4. REPRODUCTION PLAN (task G): prioritized commands for a build-equipped verifier.
5. CALIBRATED CONFIDENCE: P(core trust claim holds under a full reproduction) — with the 1–2 assumptions
   that most move it, and which checks you could NOT do from read-only that would move it most.
Be specific and terse. Cite real files/lines. Earn any reassurance you give.

===PROMPT ENDS===
```

---

## § What to do with ChatGPT's findings
- A real **CRITICAL** / **MAJOR** → fix it, re-run the universal gate to GREEN, re-verify — BEFORE
  tagging `v1.3-release`. The cross-model review is a release gate.
- A **MINOR** → fold into the release doc-polish pass.
- Record the verdict + the resolution of any finding in `docs/TRUST_CHAIN_CLOSED.md`, so the release
  reflects that an independent, different-lineage reviewer with full read access examined it.
