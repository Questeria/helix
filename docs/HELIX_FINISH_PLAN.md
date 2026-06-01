# Helix v1.0 — Comprehensive Finishing Plan

**Created:** 2026-05-31 · **Source:** 7-agent deep research (`HELIX_FINISH_RESEARCH.md`, ~1M tokens) verified live against the post-K4 tree · **Target:** `HELIX_V1_DEFINITION_OF_DONE.md` (8 criteria + the capstone transformer) · **Then:** 5 consecutive clean multi-agent audits → STOP.

This is the plan the autonomous finishing loop executes, tick by tick. It is dependency-ordered, each step a **bounded green commit** (never leave the tree red, never fake, fail-closed). Effort tags: **S** ≈ hours · **M** ≈ days · **L** ≈ 1–2 weeks · **XL** ≈ multi-week/multi-session.

---

## 0. Executive summary — the one critical path

```
P0 restore the floor (fix broken gate)  →  P1 GPU first-light (vector_add on the 3070, Helix-driven)
   →  P5 general GPU GEMM + tiled kernels  →  P6 transformer ops (fwd+bwd)  →  P5/P4 GPU autodiff
   →  capstone: ≥2-layer transformer trains on GPU within 2% of a PyTorch oracle  =  HELIX v1.0 DONE
   (in parallel, off the critical path: P2 de-Python harness · P3 feature corpus + feature-diverse DDC
    · P4 CPU autodiff completeness · P7 spec-doc truthfulness)
   →  freeze spec  →  5 consecutive clean multi-agent audits  →  STOP
```

**The single most important finding:** criterion #3 ("GPU executes", THE GATE) is **much closer than the DoD assumed**. A pure CUDA Driver-API launcher (`cuInit → cuModuleLoadData(ptx) → cuLaunchKernel → cuMemcpyDtoH`) was **written and run on this exact RTX 3070 during the research, and a vector_add PTX kernel executed correctly (c[7]=21.0 PASS)**. The whole downstream device path works *today*. What's missing is only the **wiring**: Helix emitting the `.ptx` file from a standalone driver-main, and triggering the launcher. So **GPU first-light is days, not weeks.**

**The honest hard part** is everything *after* first-light: growing a register-only naive 2×2 matmul into a real tiled GEMM with shared memory + Tensor-Cores, building the transformer op set (attention/layernorm/embedding) **with backward passes**, GPU autodiff, a training loop, and tuning numerics to within 2% of PyTorch. That is **multi-month real ML-systems engineering** — the bulk of the remaining work.

**And K4 left debris that must be cleaned first:** the live regression gate (`scripts/stage31_validate.py`) still invokes the **deleted** `helixc/tests/test_codegen.py`, so the CPU+PTX regression suite is **currently dark**; 2 DDC harnesses and 5 dev scripts import the deleted `helixc.*` and are dead; there is **no live self-host fixpoint test** post-K4. Restoring a green, Python-free verification floor (P0) is the precondition for trusting any subsequent claim.

---

## 1. State of the 8 criteria (live, post-K4)

| # | Criterion | Live status (verified 2026-05-31) | Phase |
|---|-----------|-----------------------------------|-------|
| 1 | Self-hosts (full language) | i32 fixpoint K2==K3 proven + 5×-audited pre-K4, but **the test was deleted at K4**; seed `_start` has no big-stack stub so K1'→K2 still needs external `ulimit`. No live fixpoint test. | P0, P3 |
| 2 | Feature-complete + runs (CPU) | ~140/144 parity & "core 100%" — **but measured vs the now-deleted Python oracle**; unverifiable today; many "FUNCTIONAL PARITY" rows (grad/closures/wrappers) are **vacuous**. No live compiles-AND-runs scorecard. | P3 |
| 3 | **GPU executes** ⭐ THE GATE | Real **text-only** PTX emitter (ptxas-grammar-valid through naive matmul). **Zero host launch path in Helix.** Launcher proven in C on the 3070 — wiring is the gap. | **P1 → P5** |
| 4 | Autodiff correct (CPU+GPU) | CPU: scalar `grad` intrinsic + hand-written per-op backward kernels; **4 tensor backward kernels missing** (matmul/softmax/layernorm/attention), forward-mode regressed (missing tanh/gelu/silu/softplus/log/powi chain rules), no finite-diff checker. GPU autodiff: 0%, hard-blocked by #3. | P4, P5 |
| 5 | Full-language trust (feature-diverse DDC) | i32 DDC proven (self-source + 21-program battery, 5×-audited). **Feature-diverse DDC blocked**: the only feature-capable 2nd route was the deleted Python compiler; the C seed is i32-only. | P3 (**Decision D1**) |
| 6 | Python-free + raw-binary | Reference compiler **deleted (K4)**; mint verified Python-free. Remaining: `assemble_k1.py`, 2 DDC harnesses, 5 dead dev scripts, `run_all_tests.sh`, and the **Helix-native test-runner port**. | P0, P2 (**Decision D2**) |
| 7 | Usable stdlib + toolchain | Strong CPU tensor/nn stack (`tensor.hx` 2236 L, `nn.hx` 1295 L). **Disconnected from the GPU path; no trainable transformer pieces; no data loader/tokenizer.** Helix test runner in progress (task #13). | P2, P6 |
| 8 | Design frozen (v1.0 spec) | Not frozen; `spec.md` is a v0.1 living draft with **stale dead-Python invocations** and an obsolete back-half. Legacy `v1.0.0–v3.1.0` tags collide with the DoD's forward "v1.0". | P7 (cheap now) + final freeze |

---

## 2. Build order — 9 phases (P0–P8)

### P0 — Restore the verification floor *(CRITICAL PATH, do first, S–M)*
K4 left the gate dark. Nothing downstream is trustworthy until this is green.
- **P0.1** Orient: build the seed under WSL (`stage0/helixc-bootstrap/build.sh`, 17 tests), confirm seed→kovc mint + `6*7→42` live. *(verify the floor still stands.)*
- **P0.2** Fix `scripts/stage31_validate.py` — it invokes the deleted `helixc/tests/test_codegen.py` (lines 7, 297). Re-point at what actually exists (the WSL seed self-test + a Helix-native check) so the regression gate runs again. **S.**
- **P0.3** Re-establish a **Python-free self-host fixpoint check** in Helix: a committed `helixc/bootstrap/seed_selfhost.hx` (or `runner.hx`) driver that reads the 3 frozen sources, mints K-next via `run_process`, runs it on `6*7`, and asserts a byte-identical `K2==K3`. This replaces the deleted `test_self_host_fixpoint.py` entirely in Helix. *(Process-exec builtins `run_process`/`set_exec` already landed — task #12.)* **M.**
- **Deliverable:** a single green `validate` entrypoint that runs with **zero Python in the toolchain path** and re-asserts the i32 self-host fixpoint. **Verify:** exit 0, byte-identical fixpoint, on a clean checkout under WSL.

### P1 — GPU first-light *(CRITICAL PATH, the GATE's smallest milestone, S–M; mostly de-risked)*
DoD #3 "first light" ≡ *a `vector_add` PTX kernel, emitted by kovc from a Helix `@kernel` source, runs on the RTX 3070 driven from the Helix toolchain, and the H2D→launch→D2H round-trip is verified against a CPU reference.*
- **P1.1** Commit `helixc/runtime/cuda_launch.c` — the **proven** Driver-API launcher (verbatim the sequence already run on this 3070), parameterized by `argv` = `.ptx` path + kernel name + buffer sizes. Build: `gcc cuda_launch.c -I/usr/local/cuda/include -L/usr/lib/wsl/lib -lcuda`. Treated like `ptxas`/the build ladder — a trusted tool **outside** the self-host fixpoint (see **Decision D2**). **S.**
- **P1.2** Land the **standalone Helix compiler-driver main** (the in-progress test-infra T1/T2, task #13): read a source path, parse, `emit_auto_for_ast_to_path`, write `out.ptx` via `write_file_to_arena`. This is the real long-pole of first-light and is **shared infra**, not GPU-specific. **M.**
- **P1.3** Drive the launcher from Helix. **Recommended (3a):** extend `run_process` to accept an argv list (small additive change to `emit_run_process_body`/dispatch, kovc.hx:4325/5412) so Helix can exec `cuda_launch out.ptx vecAdd …`. Fallback (3b): `write_file_to_arena` a fixed-path `run.sh`, `set_exec`, `run_process` it. **S.** *(3a touches a hot codegen path — full fixpoint + broad regression after.)*
- **P1.4** Bump the emitter target `sm_75 → sm_86` (kovc.hx:11476; the 3070 is Ampere) — or rely on PTX JIT forward-compat, but track the device for the capstone to avoid silent perf cliffs. **S.**
- **P1.5** End-to-end: kovc emits PTX → launcher runs on the 3070 → result checked vs CPU. Capture as the **criterion-#3 "first-light" artifact**, then `quince`/audit it. **Verify:** numeric match on hardware, reproducible under WSL.

### P2 — De-Python the harnesses + Helix-native test runner *(parallel, criterion #6, M)*
- **P2.1** Replace `assemble_k1.py` (the sole generator of `k1src.hx`/`k1input.hx`, both git-ignored): either a ~15-line POSIX `assemble_k1.sh`, or — preferred — add argv support to kovc's `_start` so a Helix driver concatenates the frozen sources itself. **S.**
- **P2.2** Convert the 2 DDC harnesses (`ddc_check.py`, `ddc_battery.py`) — both dead (import deleted `helixc.tests.test_codegen`) — into `ddc_check.sh`/`ddc_battery.sh` driving the **seed** + kovc binaries (they only assemble, run two binaries under `ulimit -s unlimited`, `cmp -s`, and check `6*7→42`; the deleted import was the only compiler call). Re-point Route B off Python (see **Decision D1**). **S–M.**
- **P2.3** Drop or port the 5 dead dev scripts (`proof_artifact_{gate,validate,key}.py`, `selfhost_cascade.py`, `mlir_audit_canaries.py`) — they test deleted Python internals; confirm they only guarded Python, then drop, and re-implement `selfhost_cascade` as a shell cascade over seed/kovc. Re-point `run_all_tests.sh` off pytest. **S.**
- **P2.4** Build the committed **Helix test runner** `helixc/tests/runner.hx` (lost with K4): inline-unrolled bridge first (~50 programs/file via strlit → `write_file_to_arena` → `run_process` kovc → compare rc). To reach the full corpus, add a **dynamic-path read builtin** (`read_file_dyn(arena_ptr)`, mirroring `emit_read_file_to_arena_body` kovc.hx:4077 but lifting the strlit-only trap) so one loop reads `(program, expected_rc)` from a manifest. **M (the load-bearing item).**

### P3 — Feature coverage + feature-diverse DDC *(parallel, criteria #1/#2/#5, M + Decision D1)*
- **P3.1** Close the seed big-stack gap (finishes #1's "no external ulimit"): port the 53-byte `emit_start_bigstack` mmap prologue (kovc.hx:1990) into `seed.c` codegen (seed.c:1173) so **K1' itself** carries the stub → K1'→K2 runs with no `ulimit`. **S, but changes every seed-emitted binary → mandatory full fixpoint + broad regression before commit.**
- **P3.2** Extract a **runnable feature corpus** into the repo as Helix data: lift the ~720 `(category,name,src,expected_rc)` tuples from git tag `v0-pre-k4-full-with-python` (parity_matrix 284 + k2_parity 285 + stdlib 72 + autodiff ~49), partition **REAL** (structs/enums/match/generics/tuples/arrays/int+float widths/tile/impl) vs **VACUOUS** (grad/closures/wrappers), feed P2.4's runner → a **live, Python-free compiles-AND-runs scorecard** for #2 (today there is none). **M.**
- **P3.3** Stand up feature-diverse DDC (#5) — **Decision D1**: (D1a) restore Python from the v0 tag as a **fenced offline oracle only** and byte-compare feature ELFs across the two routes (cheap, real signal, transitional), and/or (D1b) build a 2nd feature-capable diverse seed (**XL**, the durable Wheeler answer). Default: **D1a now**, D1b tracked. *(A within-kovc "diverse settings" shortcut does NOT satisfy Wheeler — shared codegen — and must not be counted.)*

### P4 — CPU autodiff completeness *(parallel, criterion #4 CPU half, M)*
- **P4.1** Build a **Python-free finite-difference gradient checker** in Helix first (now possible — T1 landed): per op, compare `(f(x+h)−f(x−h))/2h` to the analytic backward. This is the missing acceptance instrument for #4 and gates the rest. **S–M.**
- **P4.2** Add the 4 missing CPU backward kernels (f32, mirroring `dense_layer_f32_grad_*`): **matmul-backward** (dA=dC·Bᵀ, dB=Aᵀ·dC), **standalone softmax-backward** (Jacobian-vector), **layernorm-backward** (incl. γ/β), **attention-backward** (composes the prior three). **M.**
- **P4.3** Restore forward-mode chain rules in `differentiate` (parser.hx:10540–10653) for `__tanh/__gelu/__silu/__softplus/__log/__powi` (builtins already exist in `transcendentals.hx`; old Python diff'd them — pattern-copy). Optionally give reverse-mode a `Call` arm so it stops trapping (88001). **S.**
- **P4.4** Replace fail-closed-to-0 in `softmax_layer`/`layer_norm_f32`/`attention_softmax_f32` with a strict/status variant for the **training** path (silent zeroing corrupts gradients without signal — a real convergence risk against the 2% bar). **S.**

### P5 — GPU kernels toward the capstone *(CRITICAL PATH after P1, criterion #3 grow + #4 GPU half, L–XL)*
- **P5.1** General **tiled GEMM** f32: from the hardcoded register-only naive matmul to MxKxN with **shared-memory staging** (`.shared` + `cp.async`) and **barriers** (`bar.sync`) over HBM. Validate each against the CPU `tf2d_matmul` oracle. **L.**
- **P5.2** GPU elementwise (add/mul/scale), transpose, row-softmax, reductions — each validated vs the CPU `tensor.hx`/`nn.hx` oracle on the 3070. **M.**
- **P5.3** *(perf, can trail correctness)* `wmma.mma.sync` bf16 Tensor-Core path — the only way a real transformer is fast enough; currently just a deferred comment. **L.**
- **P5.4** GPU **backward** kernels (re-emit the P4.2 set as tile kernels) + GPU autodiff gradient-check on the 3070 (finishes #4's "and GPU"). **L.**

### P6 — Stdlib transformer ops + data path *(CRITICAL PATH after P5, criterion #7 + capstone pieces, XL)*
Build each **forward + backward**, reference op-set canonical (llm.c / nanoGPT): token **embedding + positional encoding** → **dense/linear** (GPU) → **LayerNorm backward** (fill the gap) → **scaled-dot-product → multi-head attention** (learnable Q/K/V, causal mask) → **MLP block** (4× + GELU) → **residuals** → **final LN + LM-head** → **cross-entropy** composed fwd+bwd → **Adam on GPU**. Plus the data path: **byte/text loader + char tokenizer + batch iterator** (today `mnist.hx` is header-parse only). **XL.**

### P7 — Spec-doc truthfulness *(parallel, cheap, criterion #8 partial — do NOW; freeze waits)*
- **P7.1** De-stale the 4 lang docs (`spec.md`, `tutorial.md`, `agi-features.md`, `HELIX_REFERENCE.md`): global-replace the **dead `python -m helixc.*` invocations** with the live seed→kovc mint path; delete `spec.md`'s obsolete 2026-05-04 "Historical Implementation Snapshot" back-half; fix the `return`-keyword inconsistency. **S.**
- **P7.2** Add `docs/lang/STABILITY.md` (the explicit v1.0 backward-compat commitment, semver-grounded, folding in the stable `trap-ids.md` registry). Reconcile the legacy `v1.0.0–v3.1.0` tags with a one-paragraph "these are legacy dev-cycle tags; language v1.0 is the DoD finish line" note. **S.**
- *(The **freeze itself** — `spec.md` normative + every sample green under kovc + frozen-surface table matching the corpus — is sequenced **last**, in P8: you cannot honestly promise "no breaking changes" until #1–#7 are done.)*

### P8 — Capstone + freeze + STOP *(CRITICAL PATH terminus, XL)*
- **P8.1** Assemble a ≥2-layer char-level transformer training loop, **pure Helix-native on the GPU** (zero Python/PyTorch in the loop), reusing P5/P6 ops.
- **P8.2** Build the **PyTorch oracle** on the same fixed dataset/config/seed (oracle only — outside the Helix toolchain).
- **P8.3** Iterate to **final training loss within 2% of the oracle + eval-metric parity**. *(Watch numerics: f32-as-i32-bits + Taylor-series transcendentals may drift; the 2% bar may force better range reduction / higher-order series.)*
- **P8.4** Freeze: promote `spec.md` to normative, frozen-surface table matches `parity_matrix`, every sample green under kovc, tag language **v1.0**.
- **P8.5** Run **5 consecutive clean multi-agent audits** (any red → reset to zero, honest), then **STOP**. ✅ HELIX v1.0 DONE.

---

## 3. Critical path vs parallel

- **Critical path (serial, gates the capstone):** P0 → P1 → P5 → P6 → P8. This is the long pole; everything about "Helix is done" routes through GPU execution growing into a converging transformer.
- **Parallelizable (do not gate the capstone, run as audits/regressions allow):** P2 (de-Python), P3 (feature corpus + DDC), P4 CPU half, P7 doc cleanup. The loop interleaves these on ticks where the critical-path step is blocked on a long compile/mint or awaiting an audit.

---

## 4. Blockers, decisions, risks

**Environment — NOT a blocker (verified):** WSL2 + RTX 3070 8 GB + `libcuda.so` (`/usr/lib/wsl/lib`) + `ptxas`/`nvcc` (CUDA 12.0/12.8) + `cuda.h` all present; **PTX executes on the device** (proven this session). The `nvidia-smi ERR!` telemetry fields are a cosmetic WSL passthrough quirk; compute is unaffected.

**Two architectural decisions to surface (don't block P0/P1; proceed on the recommended default, revisit at the named phase):**
- **D1 — feature-diverse DDC second route (bites at P3):** restore Python from the `v0-pre-k4` tag as a **fenced offline oracle** (D1a — cheap/transitional, but re-touches the retired route) **vs** build a 2nd independent feature-capable diverse seed (D1b — XL, durable, the principled answer). *Default: D1a now, D1b tracked.*
- **D2 — the C launcher at v1.0 freeze (bites at P8):** does "zero non-Helix" require retiring `cuda_launch.c` via **in-Helix dynamic linking / dlopen** (path b — a major new compiler subsystem: PT_INTERP/PT_DYNAMIC/GOT/PLT or a runtime loader) **vs** accepting it as a **ptxas-style trusted-tool boundary** outside the self-host fixpoint, like the build ladder? *Default: trusted-tool boundary now (clears the gate); schedule path-b before formal freeze if purity demands.*

**Top risks (and the discipline against each):**
- **Stale evidence trap** — the 140/144 parity, "30 passing PTX tests", and self-host fixpoint claims were all vs the **deleted** oracle/tests. They are **historical, not live**, until P0/P2/P3 re-establish them in Helix. Never re-cite them as current.
- **PTX-text ≠ GPU-ready** — grammar-valid text is not hardware-correct; only the end-to-end H2D→launch→D2H→compare loop (P1.5) earns the #3 claim.
- **Register-only tiles don't scale** — the existing naive matmul cannot carry the capstone; P5.1 (SMEM + tiling) is mandatory, not optional polish.
- **Hot-codegen regressions** — `run_process`-argv (P1.3) and the seed big-stack port (P3.1) touch the kovc.hx/seed.c hot path with a history of large-main/arena corruption bugs; every such change is gated by a **full self-host fixpoint + broad sequential regression** before commit.
- **Numerical drift vs the 2% bar** (P8.3) — f32-as-i32-bits + range-limited Taylor transcendentals may accumulate error; budget for better numerics.
- **Fail-closed-to-0 as a silent gradient bug** (P4.4) — zeroing on NaN mid-training corrupts convergence without signal.
- **Restored-Python trust hole** (D1a) — only ever as fenced/offline evidence; never back into the live toolchain.

---

## 5. The STOP gate (unchanged from the DoD)

**HELIX v1.0 DONE ⇔** all 8 criteria green **AND** the capstone transformer trains end-to-end on the GPU in pure Helix-native within **2%** of the PyTorch oracle — **THEN 5 consecutive clean multi-agent audits** (any red resets the count to zero, reported honestly). Only then does the loop STOP. Substrate-done is **not** AGI (open research, pursued *on* Helix afterward); frontier-scale/multi-GPU and apps (Alt) come after.

---

*Discipline (always): Claude-subscription only; never read `C:/Projects/Neptune/api.env`; never force-push/skip hooks; WSL for all build/run/test; commit only green; bounded honest steps; no backticks in commit/Telegram messages; the from-raw-binary ladder + `helixc/bootstrap` frozen except deliberately-gated chunks. The trust spine is done — this phase builds the engine on top of it.*
