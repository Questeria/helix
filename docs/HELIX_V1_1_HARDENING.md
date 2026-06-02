# Helix v1.1 — Language & Toolchain Hardening (post-substrate)

**Context.** Helix **v1.0 (the substrate) is DONE** (`HELIX_V1_DEFINITION_OF_DONE.md`, commit
`efc2af9`, tag `v1.0`): self-hosting, GPU-executing, autodiff-correct, Python-free, raw-binary
trust-rooted, capstone-proven, 5/5 adversarial audits. v1.1 does **not** add scale — it closes
the **post-v1.0 completeness gaps** the spec §7 and the audits explicitly flagged, so that the
language is feature-complete and the toolchain is fully self-sufficient (no shell in the
build/test path) **before** the AI-building phase. Same discipline as v1.0: every
compiler/parser/lexer/seed change is **GATED** (`scripts/gate_kovc.sh`: self-host fixpoint
K2==K3==K4 byte-identical + GPU-PTX regression + the feature corpus) before commit; never ship
red; never fake; honest caveats stay visible.

---

## DEFINITION OF DONE — measurable criteria (all must hold)

| # | Criterion | Measurable acceptance test | Status |
|---|-----------|----------------------------|--------|
| **H1** | **Toolchain fully Helix (retire shell)** | the **K1-assembly** step (`assemble_k1.sh`) becomes a Helix-native concatenator, and the **test/gate corpus** runner (`feature_corpus.sh`/`gate_kovc.sh` orchestration) becomes a **Helix-native test runner** (built on the done process-exec builtins, task #12/T1). Remaining non-Helix is ONLY the OS-boundary trusted tools (the from-raw-binary ladder `build.sh` rungs, the WSL launcher, `gcc`/`ptxas`/CUDA for the GPU C launcher), each documented. A Helix program drives the self-host + corpus end-to-end. | ✅ **GREEN (2026-06-01)** — **concatenator** (`assemble_k1.hx` drives the build via a thin `assemble_k1.sh` wrapper; shell awk/printf/cat retired; fixpoint-green K2=`96c440d3`, `ea5552e`) **+ Helix test runner** (`test_runner.hx` compiles+runs+asserts all 35 corpus programs via `run_process`/`set_exec`, replacing the shell `check()` loop; 35/35 + negative-control validated, `e43cc47`). The build source-assembly + the test corpus-check are now Helix programs. The only remaining shell is the IRREDUCIBLE bootstrap launcher (`seed`→K1→K2 builds the first compiler; `run_process` is kovc-only so a Helix orchestrator cannot bootstrap K2 itself) + thin invocation = the documented OS-boundary trusted-tool layer (like `make` / WSL / the ladder `build.sh`). |
| **H2** | **Generics — real codegen** | `<T>` **monomorphizes** (no longer type-erased — spec §7 "the most significant gap"). A generics corpus (`Vec<T>`, `Option<T>`, a generic fn over ≥2 differing element types, a generic struct/impl) **compiles AND runs correctly** on the self-hosted compiler, gated by the fixpoint. | ✅ **GREEN (2026-06-01)** — `<T>` **monomorphizes** (no longer type-erased; spec §7's most-significant-gap closed). The full charter corpus compiles AND runs on the self-hosted fixpoint-gated compiler, now **permanently gated (43/43 in `gate_kovc.sh`, +8 H2 generics)**: generic **functions** over differing scalar types (i32/f32/i64); generic **struct fields** i32/f32 (`aefea02`); generic **impl methods** incl. **bare-`T` return at non-i32** mono'd per receiver (`gen_impl_t_single_f32`→5, `gen_pair_multi` i32+f32 in one program→12; approach-B monomorphize-alias, no new sb-slot, `3205836`); **`Vec<T>`** container over i32+f32 (`gen_vec_i32`→42, `gen_vec_f32`→5, incl. f32-through-i32-arena round-trip); **`Option<T>`** generic enum via bare construct→42 (`2b2e536`); **square-bracket generics** across struct/enum/impl decls (`8c492c8`+`2b2e536`). Fixpoint K2==K3==K4 byte-identical throughout. **Deferred (non-blocking, workarounds work, NOT charter-corpus items):** turbofish-on-enum `Opt::<T>::Some` parser hang (bare construct works), f64/i64 8-byte generic struct fields (4-byte works), bare non-turbofish generic call at non-i32 (turbofish works). |
| **H3** | **Traits + closures — real codegen** | trait-method dispatch and closure capture **codegen** (no longer erased); each corpus-proven (compile + run, gated). | ✅ **GREEN (2026-06-01)** — trait-method dispatch + closure capture codegen are real, now permanently gated (corpus 50/50, +7 H3 programs): trait dispatch single-impl (`t2_trait_impl`→42), polymorphic two-type (`t7_trait_poly`→42), two-type same-field (`t7b`→42) and differing-field (`t7c`→42), closure call (`t3`→42) and single/double capture (`t4`/`t8`→42). Probe-first also found + fixed a **latent multi-impl bug** (a 2nd impl method's `self.field` resolved against the 1st struct → 132; `parse_impl_method` now resets `var_struct_tab`; fixpoint byte-identical, `75f744d`). **Deferred (non-blocking, beyond DoD, workarounds exist):** trait DEFAULT methods (unimplemented — `parse_trait_decl` brace-skips the body; workaround: explicit method impls; a call to an un-overridden default traps ud2 at runtime, not a compiler crash), higher-order closures (closure-as-fn-arg — needs a proper probe). |
| **H4** | **Pattern guards** | match-arm `if`-guards are **enforced** at runtime (no longer erased); corpus-proven. | ✅ **GREEN (2026-06-01, `329d78b`)** — match-arm `if cond` guards are now **evaluated** (were parsed-and-discarded / always-true). Parser stores the guard in the arm node; `emit_one_match_arm` evaluates it after the pattern matches (binds active) and falls through to the next arm when false (`test eax,eax; je →fail_state`). Corpus-gated 53/53: `g1_guard_true`→1, `g2_guard_false`→0 (falls through), `g3_guard_chain`→2 (second guard wins). Fixpoint byte-identical (self-host uses if-cascades, never `match`). |
| **H5** | **i64 source literals ≥ 2³¹** | the lexer/AST carries 64-bit literals (mirror the f64 lo32+hi32 path; the seed has no i64 → i32 multi-word arithmetic). The full i64 range compiles correctly (`5_000_000_000_i64` no longer truncates). | ✅ **GREEN (2026-06-02)** — full 64-bit i64 literal range compiles correctly (`5_000_000_000_i64` no longer truncates), **gated 56/56 in `gate_kovc.sh`** (+3 H5: 3e9→30, **5e9→50 (> 2³²)**, 2.2e9→22). Fix mirrors the **f64 tag-34 text-ref path**: the parser stores the literal's byte_start/byte_len (not the lex-truncated i32 value), and codegen decodes the decimal digits into the full 64-bit value via **i32-multi-word 16-bit limbs** (all-positive arithmetic — no i64/shift/cast/signed-division — so the i32-only seed self-compiles it; 2 edits, no lexer change). Fixpoint K2==K3==K4 byte-identical (zero `_i64` literals in the self-host source → the new codegen path is never self-exercised). The stale lexer-i64-accumulator design was **unbuildable** on the i32-only seed — replaced by this codegen-decode. **Deferred (non-blocking, same pattern):** u64 literals ≥ 2³² (tag 38). |
| **H6** | **(trust) M2-Planet self-host fixpoint** | M2-Planet rebuilds M2-Planet byte-stably (the ladder's one open fixpoint, `stage0/M2-Planet/PROVENANCE.md`), **OR** a documented, honest decision to keep it built-once-and-audited (it is the trusted-once root the seed is built from). | ✅ **DOCUMENTED (2026-06-02)** — green-or-documented satisfied via the **built-once-and-audited** path. A bounded self-host attempt (`selfhost_probe.sh`; vendored `libc-full.M1` @ pin b8bb2a01) confirmed M2 self-compiles its own 12 sources (gen2.M1 = 2.2 MB) but the assembled gen2 binary SIGILLs (132) — a latent M1-emission/assemble-pairing gap **inside the vendored GPL M2-Planet/M0/hex2 toolchain**, deeper than the libc variant, deliberately out of scope (it would mean debugging upstream GPL codegen, not Helix). M2-Planet is kept as the **trusted-once root**: vendored at a pinned commit from community-audited sources, built only by prior rungs (no pre-built binary trusted), output verified end-to-end (compiles C, the result runs). The trust bearing on *our* code (seed → kovc) is the green, every-commit-gated **K2==K3==K4 fixpoint + DDC**. Full record: `stage0/M2-Planet/PROVENANCE.md`. |

**HELIX v1.1 — DONE** ⇔ H1–H5 green (H6 green-or-documented), each gated, with a per-criterion
corpus, and a short adversarial audit pass (lighter than v1.0's 5-round capstone gate — these are
lower-uncertainty engineering, but real).

---

## Recommended sequence

1. **H1 first** — bounded, high trust-value, builds directly on the done T1 process-exec
   builtins; completes the "fully-Helix toolchain" story (DoD #6 "final form"); gives the v1.1
   track an early measurable gated win.
2. **H2 generics** — the headline language capability + the biggest gap. The largest, riskiest
   item (real monomorphization in the self-hosted compiler); do it on its own with heavy gating.
3. **H3 / H4 / H5** — round out the language (traits, closures, guards, i64 literals).
4. **H6** — ladder fixpoint trust depth (or document the honest deferral).

## Out of scope for v1.1 (later tracks)
- **GPU performance** (tiled/shared-memory GEMM, fusion) — that is the first enabler of the
  *AI-building* phase (scaling), not a language gap. Tracked separately.
- **Scale / a real LM (Alt)** — the AI-building phase, after v1.1.

*Charter authored 2026-06-01, the day v1.0 was certified DONE. The status column is honest TODO
until each criterion is gated-green.*
