# Plan: Helix → CUDA Obsolescence

**Status**: living document, 2026-05-25. Tracks what must land BEFORE Helix is "finished" (= K-bootstrap complete + v1.0 standardized) so that AFTER finish, the CUDA-replacement library work can proceed in pure self-hosted Helix.

The interpretation of "make Helix obsolete" used in this doc: Helix-in-Helix replaces the Python helixc reference implementation; the same Helix language then replaces the CUDA C++ kernel ecosystem. Two distinct obsolescence vectors, both real.

## 0. Strategic framing

Helix-the-language and CUDA-the-ecosystem are different targets. Replacing them requires:

1. **The language MUST be finished first** (Helix can target every GPU from one source).
2. **The library ecosystem MUST be written in finished-Helix** (so it benefits from autotune, multi-backend codegen, refinement types — and doesn't require re-porting).
3. **Adoption channels MUST be wired through existing frameworks** (PyTorch / JAX) so the user-facing migration is gradual, not big-bang.

Pre-finish work is dominated by (1). Post-finish work is dominated by (2) and (3).

## 1. Pre-finish work — what must land BEFORE v1.0

### 1.1 Language foundation (already shipped in Python helixc, K-bootstrap pending)

| Component | Python helixc | K-bootstrap | Status |
|---|---|---|---|
| Multi-backend codegen (x86_64, LLVM IR, PTX, ROCm/HIP, Metal MSL, WebGPU WGSL) | ✅ shipped v3.0 | ⚠️ only x86_64 in bootstrap; GPU backends MISSING | pre-finish: K-bootstrap parity track |
| MLIR migration path (Phase E) | ✅ shipped Stage 211-216 | ❌ not in bootstrap | pre-finish: bootstrap doesn't need MLIR for self-host, but Python helixc continues using it for GPU paths |
| Tile-IR (`tile<T, [N, M], mem>`) | ✅ shipped | ⚠️ parses as generic; no semantic | pre-finish: tile codegen not required for self-host; deferred to v1.x for GPU kernels |
| @autotune (variant expansion + benchmarking) | ✅ shipped | ⚠️ attribute parses; no expansion | pre-finish: kept in Python helixc track; bootstrap rejects the expansion |
| Refinement types via presburger (compile-time shape checks) | ✅ shipped | ⚠️ no shape system | pre-finish: kept in Python; bootstrap is type-erased |
| Effect system (@pure / @effect) | ✅ shipped | ⚠️ attribute parses; no validation | pre-finish: kept in Python; vacuously satisfied in bootstrap |

**Net pre-finish work for (1)**: complete the K-bootstrap parity track (currently ~140/143 ≈ 98%). The remaining real-code gaps are GPU codegen (PTX/ROCm/Metal/WebGPU), MLIR migration, tile-IR ops, and the @autotune-expand pass. Since the bootstrap source doesn't use these, they remain Python-helixc-only features — but the Python helixc IS Helix-the-language, so they ship as part of the language at v1.0.

### 1.2 Standardization (must happen before v1.0 commits to backward compat)

- **Language spec**: docs/HELIX_SPEC.md frozen at v1.0. Currently the language is defined by what the test suite covers. A spec document anchored to test-suite assertions would catch silent behavioral drift between Python helixc and the bootstrap.
- **ABI for `extern "C"` boundary**: defined precisely so Helix kernels can be linked against C runtimes (PyTorch / JAX integration).
- **Standard library scope**: what's `helixc-std` vs `helix-ml` vs `helix-gpu`. Three-tier layering: std is OS-portable (file I/O, arena, basic data types); ml is hardware-agnostic ML primitives; gpu is hardware-specific kernels.

### 1.3 Toolchain quality (pre-finish prerequisites for library work)

- **Profiler hooks**: `@trace` is already designed. Need wire-up to a binary-level profiler (perf / VTune / Nsight Systems). Pre-finish: the @trace runtime helper (`__trace_event`) lands in both backends.
- **Debugger support**: DWARF emission for debug builds. Currently the bootstrap produces stripped ELF. Pre-finish: DWARF support added (gives gdb / lldb integration).
- **LSP**: IDE integration for syntax highlighting, type info on hover, jump-to-def. Pre-finish: minimal LSP wrapping the existing parser. (Could be deferred to v1.1.)
- **Standardized error format**: caret-rendering + numeric trap-id ↔ description mapping. Pre-finish: `diagnostics` row in K-bootstrap matrix moved from "vacuous parity" to actual implementation in bootstrap.

### 1.4 Adoption-channel groundwork

These don't need to ship pre-finish but the DESIGN must be settled:

- **PyTorch backend**: Helix as a `torch.compile` target. Roughly: PyTorch's `inductor` codegen emits Helix tile-IR instead of Triton. Discussions with PyTorch core team should start at v0.9 (pre-1.0).
- **JAX backend**: lower XLA HLO through Helix's MLIR path. Less invasive than PyTorch since JAX already uses MLIR.
- **Standalone framework**: a `helix-train` API for users who want Helix without PyTorch/JAX dependency. Defer to v1.x.

## 2. At-finish work — what ships in v1.0

### 2.1 First Helix-in-Helix programs (proof the compiler works on real code)

The K-bootstrap track ends at K4 (Python helixc deleted). Immediately after K4:

1. **`helix-hello-world.hx`**: a GPU "hello" that compiles to PTX + ROCm + Metal + WebGPU from one source. Validates the multi-backend codegen path is real, not just compiler-internal.
2. **`vector-add.hx`**: parallel C = A + B for vector types. Validates the tile-IR's elementwise lowering.
3. **`gemm-i32.hx`**: integer matmul. Validates the tile-IR's contract lowering (vector.contract → wmma m16n16k16). Numerical correctness checked against a CPU reference.
4. **`gemm-f32.hx`**: float matmul with @autotune for BLOCK_SIZE, NUM_WARPS. Validates the autotune harness.

These four programs are the v1.0 acceptance criteria for "the compiler can self-host AND produce real GPU code".

### 2.2 Documentation freeze

- HELIX_SPEC.md → v1.0
- HELIX_TUTORIAL.md → covers all language features with examples
- HELIX_KERNEL_GUIDE.md → how to write GPU kernels in Helix (the eventual cuDNN-replacement entry point)

## 3. Post-finish work — the actual CUDA obsolescence

Now Helix is finished. The remaining work is writing CUDA-replacement libraries IN HELIX. This is community-scale work, ~5-10 engineer-years if done well; the @autotune machinery amortizes most of it.

### 3.1 Phase α — foundational kernels (target: v1.1)

| Kernel | Reference | Helix size estimate |
|---|---|---|
| GEMM (f16/f32/f64, batched + non-batched) | cuBLAS | 200-400 LOC + autotune sweep |
| Attention (FlashAttention-2) | xformers / FlashAttention repo | 500-800 LOC |
| Conv2D (im2col + GEMM + direct) | cuDNN | 600-1000 LOC |
| Softmax + LogSumExp | cuDNN | 100-200 LOC |
| LayerNorm + RMSNorm | cuDNN | 200-300 LOC |
| Reductions (sum, mean, max, argmax) | thrust | 300 LOC |
| Elementwise (relu, gelu, sigmoid, tanh, exp, log) | cuDNN | 100 LOC per op |

These are tractable IN HELIX because:
- `@autotune(BLOCK: [16, 32, 64], WARPS: [4, 8])` automates the per-hardware tuning.
- Refinement types (presburger) catch shape bugs at compile time.
- Multi-backend codegen produces NVIDIA + AMD + Apple + WebGPU binaries from the same source.
- Effect annotations (@pure on the inner loops) let the compiler vectorize/fuse aggressively.

### 3.2 Phase β — distributed-training primitives (target: v1.2)

- AllReduce (ring + tree topology)
- AllGather + ReduceScatter
- Pipeline-parallel scheduling
- Tensor-parallel matmul splits

These require:
- Cross-device memory model (Helix needs `device<T>` annotations finalized)
- Async kernel launch semantics
- NCCL-equivalent on-wire protocol (or wrap NCCL initially, replace later)

### 3.3 Phase γ — ecosystem integration (target: v1.3)

- PyTorch backend: lower torch.compile graphs through Helix
- JAX backend: lower XLA HLO through Helix's MLIR
- Numpy-style `helix-array` API for end-users not using PyTorch/JAX

### 3.4 Phase δ — production hardening (target: v2.0)

- Continuous benchmarking against cuBLAS/cuDNN on H100, MI300, M3
- Fuzz testing for numerical edge cases (denormals, NaN/Inf propagation)
- Memory leak detection
- Long-running stability (>30-day continuous training runs)

## 4. Adoption strategy

Time to displace CUDA from a real workload depends on adoption channels:

1. **Researchers writing kernels** (early adopters): Helix's kernel-writing ergonomics + multi-backend make it attractive for HPC/AI research labs that already write custom kernels. Target community: 6 months post-v1.0.
2. **PyTorch users via torch.compile backend**: zero-code-change migration. If Helix is a faster torch.compile backend than Inductor for some workloads, users migrate transparently. Target: 12-18 months post-v1.0.
3. **AI framework authors**: HuggingFace transformers, vLLM, SGLang. If the Helix kernels are competitive on H100 + cheaper-than-CUDA-on-AMD, framework authors pick it up. Target: 18-24 months post-v1.0.
4. **Hyperscalers**: AWS / GCP / Azure adopt Helix internally for cost-of-compute on non-NVIDIA hardware. Target: 24-36 months post-v1.0.

CUDA-the-monopoly is "obsolete" when (3) and (4) happen at scale.

## 5. Critical risks + mitigations

| Risk | Mitigation |
|---|---|
| NVIDIA evolves PTX faster than Helix can track | Helix targets MLIR's gpu dialect rather than raw PTX; LLVM-project upstream tracks PTX |
| Performance gap to hand-tuned cuDNN | @autotune sweep + community-submitted kernel libraries close the gap iteratively |
| AMD/Apple/WebGPU backend bit-rot | CI matrix runs all 4 backends on every commit (already designed in helixc/backend/llvm_parity.py) |
| K-bootstrap track stalls | Currently ~98% parity; remaining gaps are real-code chunks that I'm shipping one per cron tick. Worst case: parity stalls in 90%s; Python helixc remains usable indefinitely |
| Ecosystem inertia (CUDA libraries entrenched) | Hybrid mode: Helix kernels callable from CUDA programs (and vice versa) via the `extern "C"` ABI. Migration is incremental |

## 6. Concrete v0.x → v1.0 milestone checklist

Adopting K-bootstrap stage numbering:

- [x] v3.0 released (Python helixc with MLIR + 4 GPU backends)
- [x] K-bootstrap K0 (feature parity matrix)
- [x] K-bootstrap K1 (in progress, ~98% parity as of 2026-05-25)
- [ ] K-bootstrap K2 (parity harness — verify K1=K2=K3 fixpoint on bootstrap source)
- [ ] K-bootstrap K3 (trusted seed bootstrap — DDC chain from raw binary)
- [ ] K-bootstrap K4 (delete Python helixc, USER-GATED)
- [ ] K-bootstrap K5 (DDC + 5-clean-gate final audit)
- [ ] v1.0 standardization: HELIX_SPEC.md, ABI spec, std/ml/gpu library tiers settled
- [ ] First 4 Helix-in-Helix GPU programs (hello / vector-add / gemm-i32 / gemm-f32)
- [ ] Profiler + DWARF debug info
- [ ] PyTorch backend prototype (proof-of-concept on a single kernel)

After all of the above: tag v1.0. Then start Phase α library work.

## 7. What I'm doing each cron tick

Each autonomous worker tick advances K-bootstrap by one chunk. Cadence:
- Real-code chunks: implement a missing feature (e.g., K1.AL match block bodies, K1.AK print_str, K1.AJ PatStruct)
- Discovery batches: flip stale-matrix entries when probing reveals the bootstrap actually supports the feature

Once K1 closes (~100% parity), K2 builds a parity-gate harness, K3 builds the trusted-seed DDC chain, then K4 (user-gated) deletes Python helixc. K5 closes the v1.0 audit.

This plan document is committed alongside the cron-tick work to make the long-term goal legible — the K-bootstrap parity work is NOT the end goal, it is the prerequisite for the post-v1.0 library work that actually displaces CUDA.

---

*Plan authored 2026-05-25 in response to a user query about CUDA obsolescence timing. Update as v1.0 milestones close or new risks surface.*
