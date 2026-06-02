> ⚠️ **SUPERSEDED (2026-06-02).** The adversarial design skeptic showed hxcc-in-Helix is **not trust-independent of M2-Planet** (kovc descends from the M2-Planet-built seed), so it is a *weaker* diverse-double-compile route than using an independent compiler, and it does **not** make H6 green. Pivoted to **`SEED_DDC_CROSSCHECK.md`** — a gcc-vs-M2-Planet DDC of the seed (genuinely independent lineage). This doc + the design-workflow blueprint + the skeptic's 7 findings are retained as the design record.

# hxcc — our own C-compiler rung, written in Helix (Option C)

**Goal.** Replace our reliance on the borrowed GPL **M2-Planet** rung with **`hxcc`** — a
C compiler written in **Helix** (`.hx`), built by our own DDC-verified, self-hosting
**kovc**, that compiles our `stage0/helixc-bootstrap/seed.c` (the ~1368-line minimal-C
bridge) into a seed that drives the Helix self-host to the **same** `K2==K3==K4`
fixpoint. This makes **H6 genuinely green** (the C-compiler rung's trust now flows
through kovc's *passing* self-host + DDC, not M2-Planet's *failing* one) and gives us
our own compiler for every future rebuild.

## The honest trust claim (read this first — no overclaim allowed)

- **What Option C achieves:** hxcc is 100% our Helix code; its integrity derives from
  kovc (self-host fixpoint + DDC, both green). hxcc independently reproduces a
  fixpoint-equivalent seed → a real **diverse-double-compile** of the seed stage (a
  Thompson-backdoor defense). M2-Planet drops from *sole required trusted rung* to an
  *optional cross-check*.
- **What it does NOT, by itself, achieve:** it does **not** erase M2-Planet from the
  very first **cold** raw-bytes bootstrap. hxcc is Helix → needs kovc → kovc's first
  existence used M2-Planet (circularity). Truly deleting M2-Planet from the cold start
  additionally needs a C-subset compiler buildable by `cc_amd64` (an Option-B sibling)
  — a **documented stretch goal**, pursued only if cleanly achievable. We state this
  truthfully and **never fake a "M2-Planet removed from cold start" claim.** (Same
  discipline that made H6 honest in the first place.)

## Definition of DONE (all must hold; 5 clean audits certify)

- **D1 — hxcc compiles seed.c.** A C compiler in Helix, built by kovc, that compiles
  `stage0/helixc-bootstrap/seed.c` to a working seed. Supports exactly the C subset
  seed.c uses (int/char, pointers, arrays, the operators + control flow it needs,
  `calloc`/`sizeof`/`fopen`/`fgetc`/`exit` and whatever else the full feature audit
  finds). The C-subset is fully enumerated by the design phase, not guessed.
- **D2 — Faithfulness via the fixpoint (the core proof).** The seed produced by hxcc
  drives the Helix self-host to the **same `K2==K3==K4` byte-identical fixpoint** as the
  M2-Planet-built seed. Byte-identical seed is the strongest form; fixpoint-identity is
  the accepted form (hxcc's codegen may legitimately differ from M2-Planet's while
  producing an equivalent seed). hxcc-vs-M2-Planet producing the same fixpoint = a DDC
  of the seed stage.
- **D3 — Trust rooted in kovc.** Documented trust chain: hxcc = Helix source compiled by
  the DDC-verified self-hosting kovc; its integrity = kovc's (green).
- **D4 — M2-Planet demoted.** Canonical build: hxcc produces the seed; M2-Planet kept
  only as an independent cross-check. The cold-start circularity stated honestly;
  C-subset cold path is a documented stretch goal.
- **D5 — Gated, no regressions.** A new `hxcc` gate (kovc builds hxcc → hxcc compiles
  seed.c → seed → drives the kovc fixpoint) is the arbiter. The existing v1.0/v1.1 gates
  stay green. Ladder `hex0..M2-Planet` + `helixc/bootstrap` stays FROZEN except
  deliberately-gated integration. Never ship red.
- **D6 — Docs + H6 re-evaluation.** Charter + `stage0/M2-Planet/PROVENANCE.md` updated;
  H6 re-evaluated — flip to GREEN iff the kovc-rooted hxcc argument honestly satisfies
  "the C-compiler rung is trust-rooted in a passing self-host," else refine its docs.
- **D7 — 5 consecutive clean independent adversarial audits.** Distinct lenses
  (C-subset completeness; faithfulness-proof soundness; trust-claim honesty / no
  overclaim; fixpoint + DDC integrity; no-fakery / reproducibility). Any real gap → fix
  → **reset the streak to 0**. Only 5-in-a-row clean → DONE.

## Phase plan (the autonomous loop drives these)

1. **DESIGN** (workflow / parallel agents): enumerate seed.c's exact C subset; map
   M2-Planet's interface in our ladder (how seed.c is compiled to the seed today: the
   build script, the M1-assembly output, M0/hex2 assemble, the calling convention the
   seed expects); choose hxcc's codegen target (emit M1 to reuse M0/hex2, vs emit ELF
   directly like kovc); design hxcc's architecture in Helix; pin the faithfulness-proof
   + trust story + DoD. Output: an implementation blueprint doc.
2. **IMPLEMENT** (serial `.hx` edits, gated): build hxcc incrementally — trivial C first
   (`int main(){return 42;}`), then grow the subset (arrays, pointers, calloc, file IO,
   the operators/control-flow) until it compiles all of seed.c. Each increment gated by
   a growing C corpus.
3. **FAITHFULNESS** (the make-or-break gate): hxcc compiles seed.c → seed'; prove seed'
   drives the **same** kovc `K2==K3==K4` fixpoint as the M2-Planet seed.
4. **INTEGRATE**: wire hxcc into the canonical build as the seed-builder; demote
   M2-Planet to cross-check; update charter + PROVENANCE; re-evaluate H6.
5. **AUDIT**: 5 consecutive clean independent adversarial audits (fresh skeptics,
   distinct lenses). Any gap → fix → reset streak.
6. **DONE**: tag (e.g. `v1.2` or `hxcc-1.0`), big-confetti Telegram, STOP.

## Discipline (HARD)

Claude-subscription only (no external AI APIs); never read `C:/Projects/Neptune/api.env`;
never force-push / never skip hooks; **WSL for all build/run/test**; **never ship red;
never fake; honest** — a finding that changes the plan is the loop WORKING. Every kovc /
hxcc / seed-path change GATED before commit; SERIAL on shared build artifacts (never two
concurrent compiler builds). Ladder `hex0..M2-Planet` + `helixc/bootstrap` FROZEN except
deliberately-gated integration. Preserve tags `v0-pre-k4-full-with-python`, `v1.0`,
`v1.1`. Workflows/agents (and ultracode) authorized for hard multi-angle work; designs
read-only, edits serial, the gate is the only arbiter of green.
