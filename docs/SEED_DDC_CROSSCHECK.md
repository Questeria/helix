# Independent DDC cross-check of the C-compiler rung (gcc vs M2-Planet)

**Goal.** Independently corroborate the seed (the C-compiler rung) with a diverse
double-compile: build the seed from the **frozen** `seed.c` via **gcc** (an independent
lineage, zero M2-Planet ancestry) and via **M2-Planet** (the existing `seed.bin`), and
prove both produce a **byte-identical `K1`** from `k1src.hx`. Identical `K1` from two
independent compilers = Wheeler diverse-double-compile: M2-Planet injected nothing into
the seed (a trojan would have to live in `seed.c`'s visible source, or in *both* gcc and
M2-Planet identically).

**Why this, not hxcc (Option C).** The design skeptic showed hxcc-in-Helix is **not**
trust-independent of M2-Planet (kovc descends from the M2-Planet-built seed), so it is a
*weaker* DDC route than this one. gcc has no M2-Planet ancestry → genuine independence.
This is the honest, strongest-trust-per-effort path. (The hxcc charter
`HXCC_OWN_C_COMPILER.md` is **superseded**; its blueprint + skeptic findings are preserved
there and in `.stage33-logs/hxcc_state.txt`.)

**Honest scope.** This does **not** make M2-Planet self-host (H6 stays documented). It
**adds an independent witness** that the seed's behavior is reproducible by a compiler of
a different lineage — a real trust strengthening of the one rung M2-Planet currently
single-sources. gcc here is an **auditor** (a verification tool, like the fenced numpy
oracle kept "by design for independent verification"), **not** a shipped primary route;
the from-raw-binary ladder (`hex0..M2-Planet`) stays the trust root. Using another
language's toolchain as the *root* would trade the tiny hand-audited hex0 for a giant
trusted surface — so it stays strictly an auditor.

## Definition of DONE (5 clean audits certify)

**✅ STATUS (2026-06-02):** **DC1 GREEN** (gcc builds `seed_gcc`, self-test 42). **DC2 GREEN** —
`seed_gcc` and the M2-Planet seed both compile `k1src.hx` → **byte-identical `K1` = 600783 B**,
sha `a435b6ca…` (commit `116d2a0`). **DC3 GREEN** — the full fixpoint via the gcc-built seed reaches
`K2==K3==K4` = 606680 B, sha `03a456fe…`, identical to the M2 route (commit `72faee0`). **DC4** —
the reusable gates (`ddc_crosscheck.sh`, `ddc_fixpoint_gcc.sh`) are committed; PROVENANCE records the
independent corroboration. **DC5** (5 clean audits) — **folded into the Helix Completion final audits**
(`HELIX_COMPLETION.md`), which reexamine the whole trust chain before officially closing it.

- **DC1** — gcc compiles the **frozen** `seed.c` (no edits; libc decls supplied via
  `-include`, since seed.c omits `#include`s) → `seed_gcc`; no-arg self-test exits **42**.
  *(confirmed 2026-06-02: gcc 13.3, seed_gcc 46368 B, exit 42.)*
- **DC2** — `seed_gcc` compiles `k1src.hx` → `K1_gcc` **byte-identical** to `K1_m2`
  (the existing M2-Planet seed's K1). **The DDC anchor.**
- **DC3** — the gcc route reaches the same fixpoint: `K1_gcc → K2==K3==K4` byte-identical,
  and `K2_gcc == K2_m2`, **computed live** (never against a stale pinned byte-count).
- **DC4** — a reusable gate (`stage0/helixc-bootstrap/ddc_crosscheck.sh`) committed +
  documented; `stage0/M2-Planet/PROVENANCE.md` + the H6 framing re-evaluated **honestly**
  (the rung is now independently DDC-corroborated; M2-Planet still does not self-host, so
  H6's text is refined, **not** overclaimed as "green").
- **DC5** — **5 consecutive clean independent adversarial audits** (distinct lenses:
  lineage/independence of the witness; byte-identity + no-stale-pin; `seed.c`-not-edited +
  `-include` is honest; coverage / dark-arms of what `k1src.hx` exercises; trust-claim
  honesty / no overclaim). Any real gap → fix → **reset streak to 0**. 5-in-a-row → DONE.

## Discipline (HARD)

Claude-subscription only; never read `C:/Projects/Neptune/api.env`; WSL for all
build/run; never ship red; never fake; honest — a finding that changes the plan is the
loop WORKING. `seed.c` + the ladder stay FROZEN (gcc uses `-include`, zero edits). Preserve
tags `v0-pre-k4-full-with-python`, `v1.0`, `v1.1`. gcc is an auditor, never the shipped root.
