# Helix v1.3 — "Honest-Completeness & Trust" (the release campaign)

**Goal:** make the language **silent-bug-free and every type first-class**, and **deepen the
trust** — then certify with 5 clean independent adversarial audits + a joint trust
re-verification, and tag **`v1.3-release`** as the finished release version (after which: the
investor demo). Builds on `v1.2-complete` (`140a231`).

**Discipline (unchanged from v1.2, HARD):** every commit passes the universal gate
(self-host fixpoint **K2==K3==K4 byte-identical** + the feature corpus + the GPU-PTX
regression); commit ONLY green; SERIAL builds; Python-free fence (exactly 1 committed `.py`);
DDC stays honest; **never ship red, never fake, fail-closed over silent-wrong**; from-raw
ladder unchanged. Build pattern: fast inner loop (seed→K1) foreground; full gate via the
detached runner + foreground-poll; never the Monitor tool; `timeout`-wrap GPU runs.

---

## 1. Definition of DONE — the item set (each gated; ship or honestly re-bound)

### P0 — the silent bug (the one thing that MUST be fixed first)
- **V1 — i64/u64 (and f64) wide struct fields read full 64-bit.** Today an i64/u64 wide
  struct-field READ silently truncates to low-32 (the *only* silent-wrong residual in the
  language; v1.2 M-3). **DoD:** a struct with an i64/u64 field holding a value > 2³² reads
  back EXACT (full 64-bit) through fresh K2; f64 wide fields read full-width; gated corpus
  tests (i64 > 2³², u64 > 2³², f64); the v1.2 M-3 "silent residual" is CLOSED. If a full fix
  proves infeasible, the residual must at minimum become **fail-closed (loud), not silent.**

### Types — first-class (retire the fail-closed bounds where feasible)
- **V2 — u64 ≥2³² literals. ✅ SHIPPED (2026-06-04).** **DoD:** a u64 literal > 2³² parses +
  computes correctly; gated test; the v1.2 L-2 bound becomes shipped. **Done:** the parser
  stores the u64 literal's source-text ref and codegen (kovc.hx tag 38) decodes it full-width
  via the i64 16-bit-limb path **UNSIGNED** (no sign extension) — the H5 wide-literal decode
  mirrored for u64. The lexer's L-2 over-range cap (+ its `check_u64_10digit_overflow` /
  `ref_byte_4294967295` helpers) is **RETIRED**, and the `L2_u64_over_2p32` fail-closed negative
  test is **retired** (a shipped feature must not assert fail-closed). Gated: `V2_u64_lit_over_2p32`
  (`5_000_000_000_u64 / 1e8 = 50` exact), `V2_u64_lit_near_max` (`2⁶⁴-1 > 2⁶³-1` unsigned → 42),
  `V2_u64_lit_div_max` (`(2⁶⁴-1)/(2⁶³-1) = 2` unsigned). Fixpoint K2==K3==K4 byte-identical
  (sha 28024fbf), GPU-PTX regression clean.
- **V3 — capturing closures as values/arguments.** Today a capturing closure passed by value
  traps (SIGSEGV). **DoD:** a real closure object (heap env + fn-pointer); a capturing
  closure passed as an argument + invoked reads its captured variables correctly; gated test;
  the v1.2 M-6 capturing bound becomes shipped.
- **V4 — bf16/f16 arithmetic.** Today bf16/f16 are storage-only (arith traps). **DoD:**
  bf16/f16 add/mul/convert compute correctly within the format precision vs an f32 reference
  (CPU at minimum); gated test; the v1.2 bf16/f16 bound becomes shipped. (Foundation for a
  future G4 bf16-`wmma`.)

### Trust — deepen (honest improvement; document any residual)
- **V5 — broaden the DDC toward the v1.1 surface.** Today 44/53 witness-reachable arms; the
  v1.1 surface (generics/traits/closures/turbofish/wide-field/bf16) is un-DDC'd by the frozen
  Python witness. **DoD:** materially increase independent DDC coverage of the v1.1 surface —
  via an extended/second independent witness OR a second-compiler behavioral cross-check over
  the v1.1-surface corpus — and HONESTLY document whatever residual remains (the bar is a real
  reduction of the 44/53 gap, not necessarily 53/53). gcc-seed-DDC + the fixpoint stay intact.
- **V6 — shrink the trusted-C surface.** Today the trusted-once C is the seed (`seed.c`,
  irreducible bootstrap root, stays) + the harness (`cuda_launch.c`, `train_transformer.c`,
  outside the fixpoint). **DoD:** port harness logic from C toward Helix where feasible
  (reduce the trusted non-seed C), OR precisely inventory + minimize the trusted-C boundary;
  document exactly what trusted C remains and why (the seed is the only irreducible root).

### Spec
- **V7 — update the language spec + the DoD** to reflect v1.3 (the now-first-class types, the
  closed silent residual, the broadened DDC, the trusted-C inventory). No silent overclaim.

## 2. FINALE — 5 consecutive clean INDEPENDENT adversarial audits
Same protocol as v1.2 §1.4: 5 fresh, context-isolated, adversarial auditors, each REPRODUCING
its claims, distinct lenses — incl. **(a) the i64/u64/f64 wide-field fix is real + no NEW
silent residual was introduced; (b) closures/bf16/f16/u64-lit are correct, not papered-over;
(c) the broadened DDC is genuinely more independent + the remaining residual is honest;
(d) the trusted-C inventory is accurate + nothing hidden; (e) the whole-repo overclaim sweep +
the v1.2 invariants (from-raw ladder byte-identical, fixpoint, fence) still hold.** Any real
gap → fix (gated) → RESET the streak to 0. 5-in-a-row → finale passed.

## 3. RELEASE
T1-style invariants + V1–V7 done + 5 clean audits → **JOINT trust re-verification with the
project owner** (present the full honest record; the owner makes the call) → tag
**`v1.3-release`** (the finished release version; NOT v2.0 — the stale `v2.0.0`–`v3.1.0` tags
are a superseded MLIR line). Update `TRUST_CHAIN_CLOSED.md` with the v1.3 state + the (smaller)
honest residual list. **Then:** the investor/capabilities demo (the user's phase).

**Honest expectation:** V1–V4 are concrete + gated. V5 (DDC breadth) and V6 (trusted-C) are
genuine-improvement-with-honest-residual items — the bar is real progress + truthful
documentation, not a number forced. Fail-closed beats silent at every step.
