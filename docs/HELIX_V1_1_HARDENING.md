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
| **H2** | **Generics — real codegen** | `<T>` **monomorphizes** (no longer type-erased — spec §7 "the most significant gap"). A generics corpus (`Vec<T>`, `Option<T>`, a generic fn over ≥2 differing element types, a generic struct/impl) **compiles AND runs correctly** on the self-hosted compiler, gated by the fixpoint. | 🟡 **IN PROGRESS (2026-06-01)** — monomorphizing (no longer erased): generic **functions** over differing scalar types (i32/f32/i64), generic **struct fields** (i32/f32; `aefea02`), generic **impl methods** — including **bare-`T` return at non-i32**, now mono'd per receiver (`gen_impl_t_single_f32`→5; `gen_pair_multi`, i32 AND f32 in one program→12; approach-B monomorphize-alias, **no new sb-slot**, `3205836`) — and **`Option<T>`** (generic enum decl+construct+match via bare `Opt::Some`/`Opt::None`→42; `2b2e536`) all compile AND run, fixpoint-gated. **Square-bracket generics** across **struct/enum/impl** decls (`8c492c8`+`2b2e536`). **Remaining for green:** a `Vec<T>` growable container over differing types (the generic struct/impl mechanism is proven; a true `Vec<f32>` additionally stresses f32-in-i32-arena storage). **Deferred (non-blocking, workarounds work):** turbofish-on-enum `Opt::<T>::Some` parser hang (bare form works), f64/i64 8-byte generic fields (4-byte works), bare non-turbofish call at non-i32 (turbofish works). Probe-mapped in `.stage33-logs/v11_state.txt`. |
| **H3** | **Traits + closures — real codegen** | trait-method dispatch and closure capture **codegen** (no longer erased); each corpus-proven (compile + run, gated). | 🔴 TODO (parsed-but-erased) |
| **H4** | **Pattern guards** | match-arm `if`-guards are **enforced** at runtime (no longer erased); corpus-proven. | 🔴 TODO (parsed-but-erased) |
| **H5** | **i64 source literals ≥ 2³¹** | the lexer/AST carries 64-bit literals (mirror the f64 lo32+hi32 path; the seed has no i64 → i32 multi-word arithmetic). The full i64 range compiles correctly (`5_000_000_000_i64` no longer truncates). | 🔴 TODO (documented v1.0 limitation, task #23) |
| **H6** | **(trust) M2-Planet self-host fixpoint** | M2-Planet rebuilds M2-Planet byte-stably (the ladder's one open fixpoint, `stage0/M2-Planet/PROVENANCE.md`), **OR** a documented, honest decision to keep it built-once-and-audited (it is the trusted-once root the seed is built from). | 🟡 OPEN (deferred/honest in v1.0; revisit) |

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
