# Helix — Trust Chain Record (v1.2-complete)

**Status: the from-raw-binary trust chain is COMPLETE and independently verified.**
This document records the verified state of Helix at `v1.2-complete`. It is the honest
record on which the *formal, public* "trust chain closed" declaration rests — that
declaration is the project owner's to make (a deliberate joint re-examination), not an
automated one. Everything below was reproduced by independent adversarial auditors, and
every limitation is stated plainly.

Tag: `v1.2-complete` · Finalization commit: `291f0ec` · Fixpoint: `K2==K3==K4 = 9cc8f20b…`
(NOTE: the stale `v2.0.0`–`v3.1.0` git tags are from a **superseded MLIR exploration line**;
`v1.2-complete` is the current real head despite being numerically lower.)

---

## 1. What the trust chain IS (verified)

- **From a hand-typed root.** `hex0` (299 hand-authored hex bytes) → `hex1` → `hex2` →
  `catm` → `M0` → `cc_amd64` → `M2-Planet` → **`seed`** (our Apache-2.0 C-subset compiler) →
  **`kovc`** (the Helix compiler). Each rung is built **only by the prior rung** — no
  pre-built binary is trusted. An independent auditor **rebuilt the entire ladder from
  source and every rung reproduced its committed SHA byte-identically** (seed `9837db12…`).
- **Self-host fixpoint.** `seed → K1 → K2 → K3 → K4`, with **K2 == K3 == K4 byte-identical**
  (`9cc8f20b…`). Reproduced live by auditors.
- **Diverse double-compile (DDC).** `gcc` (independent lineage) and the frozen Python
  witness independently produce a byte-identical seed `K1` — Wheeler DDC against a
  trusting-trust attack. `gcc` is an **auditor**, never the shipped root.
- **Python-free shipped toolchain.** Exactly **1** committed `.py` in the repo
  (`verification/oracle/oracle_train.py`), a fenced verification witness never referenced
  by the toolchain. The compiler/runtime are Helix + a small hand-authored C subset.
- **Real capability.** A ≥2-layer transformer trains **end-to-end on Helix-emitted GPU
  kernels** (a TF32 Tensor-Core op-set) and converges to within **2% of an independent
  numpy oracle** — reproduced at **~0%** loss difference. The oracle was adversarially
  proven genuinely independent (it reads only the *shared initial* weights, never Helix's
  trajectory; f32-vs-f64 curves are close-but-not-bit-identical).
- **Verification.** **5 consecutive clean, independent, adversarial audits** (distinct
  lenses: trust chain/DDC, GPU perf, capstone correctness + oracle independence,
  language/codegen + bounds, overclaim/completeness). Each auditor *reproduced* its
  claims; the audits found no faked result and no undisclosed residual.

## 2. The honest boundaries (residuals — all disclosed)

These are real limits. They do not undermine the trust chain; they are stated so the
claim is precise, not inflated.

1. **GPU performance is a fraction of cuBLAS, not parity.** Same-GPU (RTX 3070 Laptop):
   G1 4.56 TFLOP/s (56% cuBLAS-f32), G2 5.445 (67.5%), G3 TF32 Tensor-Core 5.35–5.76
   (50–54% cuBLAS-TF32). Helix emits correct, reasonably-performant kernels; it does **not**
   beat NVIDIA's hand-tuned library, on this or any GPU. "Parity tier" is a label always
   paired with the explicit fraction.
2. **Capstone end-to-end speedup is 7.0–8.7×, not ≥10×.** The ≥10× target was an estimate;
   the honest measured ceiling is **Amdahl-bound** (GEMM is ~70% of the step and already the
   f32-SMEM cp.async tier; TF32 Tensor Cores are a confirmed **dead end** on this GPU at
   0.97× the f32 GEMM; a larger model cannot be baselined because the naive matmul cannot
   launch at scale). **Loss parity (the hard correctness gate) holds at ~0%.** The ≥10×
   clause is re-scoped to the measured ceiling, documented.
3. **DDC coverage is 44/53 witness-reachable arms.** The v1.1-surface (generics, traits,
   closures, turbofish, wide-field store, bf16) is **un-DDC'd** by the frozen Python witness
   — a named open residual, not full coverage.
4. **Documented language bounds.** One **silent** residual: **i64/u64 wide struct-field reads
   truncate to low-32** (a 64-bit-looking value silently truncated — flagged **v-next**).
   Everything else fails **loud**: f64 wide fields, bf16/f16 arithmetic, and u64 ≥2³²
   literals trap (SIGILL / compile error, no output); capturing-closure-as-argument traps
   (SIGSEGV). Borrows (`&mut` non-aliasing), `const`/`static`, module privacy, and match
   exhaustiveness are **documented design bounds**, not enforced.
5. **G4 (bf16 `wmma`) is a STRETCH and was not taken.** Not claimed done.
6. **Single hardware target.** Only `sm_86` (RTX 3070 Laptop) is tested. No cross-arch
   (sm_80/sm_90) or multi-vendor (AMD) validation.
7. **The GPU path is not all-the-way-down.** The chain is hand-auditable from `hex0`
   **to PTX**; below PTX it relies on NVIDIA's **closed `ptxas`** (PTX→SASS) and driver.
   The **CPU** path is all-the-way-down from raw binary; the **GPU** path is
   from-hex0-to-PTX-then-`ptxas`. This is the one trusted-once boundary on the GPU side,
   stated openly. _(v1.3 V6: the GPU **host launcher** — `helixc/runtime/cuda_launch.c` +
   `train_transformer.c` — is the C-FFI half of this same boundary: it makes the closed
   `libcuda` driver-API calls Helix cannot. Porting it to Helix would **move**, not close,
   this boundary, so it stays trusted-once C. This residual **STANDS** unchanged. Full
   trusted-C inventory + the V6 dead-C prune: `docs/TRUSTED_C_INVENTORY.md`.)_

## 3. Phase 2 (the user's to start — not auto-started)

- **Datacenter scaling (Runpod):** parameterize the PTX target (`sm_80`/`sm_90`, Hopper
  TMA/bigger mma), validate cross-arch correctness, optimize for absolute throughput, scale
  the capstone on A100/H100.
- **AMD / ROCm:** a genuinely separate backend (CDNA MFMA, rocBLAS reference, no native
  TF32) — currently unimplemented.
- **v-next codegen:** i64/u64 wide struct fields (close the one silent residual), capturing
  closures as values, bf16/f16 arithmetic, G4 bf16 `wmma`.
- **Broaden the DDC** to the v1.1 language surface.

## 4. Bottom line

From a hand-typed hex0 seed, with no trusted pre-built compiler, Helix self-hosts
byte-reproducibly, is independently cross-compiled, and trains a real neural network on its
own GPU kernels — verified by five independent adversarial reproductions, with every
limitation documented. **The trust chain is complete to PTX and the residuals are honest.**
The formal declaration that it is "closed" is reserved for the project owner's review.
