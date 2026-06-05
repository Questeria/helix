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
- **V3 — capturing closures as values/arguments. ✅ SHIPPED (2026-06-04).** **DoD:** a real
  closure object (heap env + fn-pointer); a capturing closure passed as an argument + invoked
  reads its captured variables correctly; gated test; the v1.2 M-6 capturing bound becomes
  shipped. **Done:** a capturing closure compiles to a real CLOSURE OBJECT in the runtime
  arena — cells `[code_ptr, cap0, cap1, ...]` — and the closure VALUE is the object's
  env-index OR-ed with a tag bit (`0x40000000`). The tagged index is a small positive i32 that
  survives a by-value i32 param because the runtime arena is a low `.data` address (< 2³⁰).
  The synthesized `__closure_<id>` body takes the env-index as a hidden leading param
  (`__cenv`) and reads each capture from object cell `1+k` via `__arena_get(__cenv + 1 + k)`
  (parser.hx `mk_capture_read`). The indirect-call dispatch (kovc.hx `emit_closure_dispatch`)
  tag-tests the value: **bit-30-clear** = a non-capturing raw code pointer → env-less
  `call r11` (the M-6 path, byte-identical); **bit-30-set** = a capturing object → untag, load
  the code ptr from `arena[env]`, pass the env in `rdi`, shift the user args up one register,
  `call r11`. The by-name capturing path (`let c=|y| x+y; c(2)`) now ALSO uses the object
  (cl_var_tab registration retired for capturing closures, so `c(args)` flows through the same
  indirect dispatch — the old positional-capture injection is gone). **Capture semantics:
  CAPTURE-BY-VALUE AT CLOSURE-CREATION** (each captured local's value is snapshotted into the
  object when the `|...|` literal is evaluated; later mutation of the original does NOT change
  what the closure sees — this is by-value-at-creation, not Rust-style by-reference capture).
  The i32-only capture bound is retained (a non-i32 capture would be truncated in a 4-byte
  arena cell → fail-closed trap 76003, not silent). Gated: `V3_capture_arg`
  (`x=40; c=|y| x+y; apply(c,2) → 42`, a capturing closure passed by value + invoked, capture
  flows + is read — the charter probe, pre-fix a SIGSEGV), `V3_multi_capture` (a closure
  capturing 3 locals passed by value reads all of them → 42), `V3_modify_after`
  (capture-by-value-at-creation: modify the captured local after creation → closure still sees
  the old value → 42, not 1001). Fixpoint K2==K3==K4 byte-identical (sha 794790f9; the
  self-host source has no closure literals), GPU-PTX regression clean (x86-only fix).
- **V4 — bf16/f16 arithmetic. ✅ SHIPPED (2026-06-04).** **DoD:** bf16/f16 add/mul/convert
  compute correctly within the format precision vs an f32 reference (CPU at minimum); gated
  test; the v1.2 bf16/f16 bound becomes shipped. **Done:** bf16/f16 add+mul go
  **convert-op-convert** — the operands convert to f32, the op runs in f32 (`addss`/`mulss`),
  and the f32 result rounds back to the 16-bit float. **Rounding is ROUND-TO-NEAREST-EVEN**
  (the IEEE default) at every f32→bf16 site, made consistent across the three of them:
  (1) the **bf16 LITERAL** fold (`rne_f32_bits_to_bf16`, a host-side RNE that replaced the
  Stage-1.5 plain truncation), (2) a runtime **`as bf16` CAST** (`emit_round_f32_to_bf16`:
  add the round-half-to-even bias `0x7FFF + lsb`, mask `0xFFFF0000`), and (3) bf16
  **ARITHMETIC** round-back (`emit_bf16_binop` = `addss`/`mulss` then the same RNE). bf16→f32
  is the identity (bf16 is stored as the f32-valid top-16). bf16 needs only **SSE2**; **f16**
  uses the **F16C** ISA extension (`vcvtph2ps`/`vcvtps2ph` imm8=0 RNE — Ivy Bridge/Jaguar 2012+,
  the documented f16-arith hardware floor) and shares the same convert-op-convert structure +
  cast path. A 16-bit float mixed with a non-16-bit-float operand still **TRAPS** (2001/4001;
  fail-closed — no implicit widening). Gated, each a **BIT-EXACT** internal compare returning
  a 42/0 sentinel (RNE distinguished from truncation by the operands; the sentinel keeps the
  assertion exact while fitting the 8-bit exit byte): `V4_bf16_add` (`256.0 + 3.0`: f32 sum
  259.0 → RNE bf16 **260** not trunc-258 → `(c as i32)==260` → 42), `V4_bf16_mul`
  (`17.0 * 19.0`: f32 product 323.0 → RNE bf16 **324** not trunc-322 → 42), `V4_bf16_roundtrip`
  (`1.1_bf16` → RNE bf16 **1.1015625** not trunc-1.09375; bf16→f32 identity `== 1.1015625_f32`
  → 42).
  Fixpoint K2==K3==K4 byte-identical (the self-host source uses no bf16/f16 arithmetic),
  GPU-PTX regression clean (x86-only change). (Foundation for a future G4 bf16-`wmma`.)
  _Note: the convert-op-convert arith was found WRONG on first attempt (a `mov ecx,eax` vs
  `mov eax,ecx` register-direction typo in the RNE round, + the literal still truncating); the
  bit-exact gate caught it (it had only previously RED'd on the obsolete trap test), and it was
  fixed before ship — fail-closed beat silent._

- **V4 f16 GAP FIX — ✅ (2026-06-04, post Finale Audit 2).** The V4 ship above gated **bf16**
  bit-exactly but shipped **f16 arithmetic without an f16 fixture** — and Finale Audit 2 then
  caught that f16 same-type arith was **SILENTLY MISCOMPUTING with no trap**: the original V4
  wired `is_f16_expr` to fire on type tag **5**, but NOTHING produced tag 5 — `ty_ident_to_tag`
  (parser.hx, + its two twin inline resolvers for typed-params and return-types) had no
  `f16`→5 case, and `expr_type` mapped the f16 literal (AST tag 80) to **4 (bf16)**, not 5. So
  `is_f16_expr` was permanently 0 and `emit_f16_binop` (the F16C `vcvtph2ps`/`vcvtps2ph` path)
  was **UNREACHABLE DEAD CODE**; f16 arith mis-routed to the bf16 path (a half pattern misread
  as a bf16 top-16 → a tiny denormal → cast to ~0), returning a wrong value with no SIGILL.
  Repro: `100.0_f16 + 28.0_f16` (= 128 exact) → exit 0 (expected 42); zero F16C bytes in any
  emitted f16 binary. **Fix (Option A — make the claim TRUE):** map the `f16` ident + the f16
  literal to tag 5 (all three resolvers + `expr_type`), plus the matching `type_width_class`
  (f16 → 2 bytes) and a fail-closed f16-binding assign trap (8017). `emit_f16_binop` is now
  reached; the f16 literal already stored the IEEE-754 half via `f32_to_f16_bits`, exactly what
  `vcvtph2ps` widens. **Now gated** by two SHARP rows: `V4_f16_add` (`100+28` → 128 exact; the
  old silent path gave ~0) and `V4_f16_mul` (`7.0_f16 * 293.0_f16`: f32 product **2051** →
  **RNE** f16 **2052**, distinct from a truncating narrow's 2048 AND from the old ~0 — proving
  the F16C path is genuinely used, not coincidentally right; `vcvtph2ps`/`vcvtps2ph` bytes
  verified present in the emitted binary). f16 **mixed-operand still TRAPS** (unchanged). The
  corpus moved **107 → 109** (+2 f16 rows); fixpoint K2==K3==K4 byte-identical, GPU-PTX clean.
  _Note: the finale streak RESET to 0 on this gap; the fix lands GATED before the finale
  restarts. Twice now (V4 bf16 typo, then this f16 dead-code) the bit-exact gate — not the
  spec prose — caught a 16-bit-float defect: fail-closed/gated beats a confident claim._

> **V1–V4 "types first-class" group: ✅ COMPLETE (2026-06-04).** The four type-correctness
> items are all shipped + gated: V1 (i64/u64/f64 wide struct fields — the silent-truncation
> residual CLOSED), V2 (full-range u64 literals), V3 (capturing closures as values/args), V4
> (bf16/f16 arithmetic, RNE). Remaining v1.3 work is the TRUST-deepening group (V5 DDC breadth,
> V6 trusted-C) + V7 spec, then the 5-audit finale.

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

## 2. FINALE — 5 consecutive clean context-isolated, same-model-family adversarial reproductions
Same protocol as v1.2 §1.4: 5 fresh, context-isolated, **same-model-family (Claude)**
adversarial reproductions, each REPRODUCING its claims, distinct lenses — incl. **(a) the
i64/u64/f64 wide-field fix is real + no NEW silent residual was introduced; (b)
closures/bf16/f16/u64-lit are correct, not papered-over; (c) the broadened DDC is genuinely
more independent + the remaining residual is honest; (d) the trusted-C inventory is accurate
+ nothing hidden; (e) the whole-repo overclaim sweep + the v1.2 invariants (from-raw ladder
byte-identical, fixpoint, fence) still hold.** Any real gap → fix (gated) → RESET the streak
to 0. 5-in-a-row → finale passed. **Honest scope:** these 5 are same model lineage as the
build and share its blind spots (the monomorphic-dispatch ceiling — `docs/HELIX_COMPLETION.md`
~749/767), so each must be a re-verification producing a reproducible artifact, not a debate;
and a **different-lineage cross-model review (ChatGPT, read-only)** was since run and its
findings remediated (a doc/logic review, not an independent build reproduction).

## 3. RELEASE
T1-style invariants + V1–V7 done + 5 clean audits → **JOINT trust re-verification with the
project owner** (present the full honest record; the owner makes the call) → tag
**`v1.3-release`** (the finished release version; NOT v2.0 — the stale `v2.0.0`–`v3.1.0` tags
are a superseded MLIR line). Update `TRUST_CHAIN_CLOSED.md` with the v1.3 state + the (smaller)
honest residual list. **Then:** the investor/capabilities demo (the user's phase).

**Honest expectation:** V1–V4 are concrete + gated. V5 (DDC breadth) and V6 (trusted-C) are
genuine-improvement-with-honest-residual items — the bar is real progress + truthful
documentation, not a number forced. Fail-closed beats silent at every step.
