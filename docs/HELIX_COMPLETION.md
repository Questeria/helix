# Helix Completion — vision-complete, then officially done

**Goal.** Complete Helix in accordance with its vision — an auditable language/substrate for
AGI and high-certainty computing, designed for AI to read and write — **before** building the
AI on it. When this is done, Helix is **FULLY COMPLETE**, and the trust chain is **reexamined
and officially announced closed** (concluded complete-for-now as of 2026-06-02).

**Status of the foundation so far (DONE):** v1.0 substrate (`HELIX_V1_DEFINITION_OF_DONE.md`,
tag `v1.0`), v1.1 hardening (`HELIX_V1_1_HARDENING.md`, tag `v1.1`, H1–H6 green), the from-raw
ladder fully self-hosts (H6 GREEN via mescc-tools, independently reproduced), and the seed is
independently cross-checked by a gcc-vs-M2-Planet diverse-double-compile (`SEED_DDC_CROSSCHECK.md`,
DC1–DC3 green, commit `72faee0`).

## Tracks (each GATED; the self-host fixpoint + the full corpus stay green throughout)

- **T1 — DDC BROADENING (verification).** Re-activate the original **Python `helixc`** (restorable
  at tag `v0-pre-k4-full-with-python`) as a *fenced, independent* DDC witness (NOT shipped — a
  verification oracle like the existing numpy oracle, consistent with the Python-free *shipped*
  toolchain) over a **broadened corpus** that exercises the codegen arms the self-host doesn't
  (today only ~15 of ~53, per `K_DDC_RESULT.md`): structs, enums, match, generics, traits,
  closures, guards, i64, etc. Assert Python-`helixc` and the from-raw `kovc` agree over the
  broadened corpus. Gate: a measurable arm-coverage target + byte/fixpoint agreement.

- **T2 — GPU FULL FUNCTION + PERFORMANCE.** From the capstone's *correct-but-naive* kernels to
  **full GPU function + performance at least at parity with (ideally exceeding) the standard
  (cuBLAS/cuDNN)** — via shared-memory tiled GEMM, thread-block barriers, Tensor Cores (`wmma`/
  `mma`), async copy, and the complete optimized transformer op set. Gate: a **measurable
  performance target** (set honestly by the scoping skeptic — defensible vs NVIDIA's hand-tuned
  libraries) **with correctness maintained** + broad op coverage. (The scoping workflow sets the
  exact target; "parity-or-better" is the aspiration, bounded by what a from-raw PTX emitter can
  achieve.)

- **T3 — POLISH (everything else for vision-completeness).** Packaged standard collections
  (`Vec<T>`, `HashMap`, …), rich string types, quality error messages/diagnostics, the deferred
  generics/traits edges (turbofish-on-enum, f64/i64 8-byte generic struct fields, bare
  non-turbofish generic call at non-i32, trait *default* methods, higher-order closures),
  syntactic sugar (`+=`/`for`), module-system semantics, and any unimplemented spec features.
  Each item gated by a corpus test; the language stays self-hosting throughout.

- **FINALE — 5 CONSECUTIVE CLEAN INDEPENDENT ADVERSARIAL AUDITS** certifying Helix is fully
  complete (T1–T3 green + the trust chain intact, reexamined). Each a fresh skeptic, distinct
  lens. Any real gap → fix (gated) → **reset the streak to 0**. 5-in-a-row clean → DONE.

## DONE — Helix FULLY COMPLETE

T1–T3 all green + 5 clean audits → update the v1.0 DoD / spec as needed; tag (e.g. `v2.0`);
**reexamine and OFFICIALLY ANNOUNCE the trust chain closed**; BIG confetti Telegram; update the
Kovostov workspace + goal; **STOP**. Then the AI-building phase begins (the user's call).

## Discipline (HARD)

Claude-subscription only; never read `C:/Projects/Neptune/api.env`; never force-push / never skip
hooks; **WSL for all build/run/test** (use `.sh` files, not inline `wsl bash -c` with `$vars`);
**never ship red; never fake; honest** — a finding that changes the plan is the loop WORKING. The
**shipped** toolchain stays from-raw-binary + Python-free; the Python `helixc` and the numpy oracle
are *fenced verification witnesses only*. Every compiler/seed change GATED (self-host fixpoint +
corpus) before commit; SERIAL on shared build artifacts (never two concurrent compiler/GPU builds).
Preserve tags `v0-pre-k4-full-with-python`, `v1.0`, `v1.1`. Workflows/agents/ultracode authorized
for hard multi-angle work; designs/audits read-only, edits serial, the gate is the only arbiter of
green. **The per-track concrete ordered/gated plans are produced by the scoping workflow
(`helix-completion-scope`) and appended once it lands.**
