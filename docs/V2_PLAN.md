# Helix v2.0 Implementation Plan

User authorized 2026-05-19 after 5-clean-gate achieved on v1.0.

## Scope (from `docs/v2-research/findings-consolidated.md`)

### Phase B — Differentiators (~5 EM, mostly compiler work)
- **Stage 110** (B.1.a): GPU effect labels — `gpu.warp_sync`, `gpu.block_sync`, `gpu.grid_sync`, `gpu.smem_borrow` extending `_KNOWN_FN_ATTRS` + `_SUB_LABELS`
- **Stage 111** (B.1.b): Stdlib annotation — annotate wmma, cp.async, ld.matrix with sync obligations
- **Stage 112** (B.1.c): Effect propagation tests — verify obligation surfacing at call sites
- **Stage 113** (B.2.a): Scope-tagged borrows — extend `Place` with scope field (`'thread`/`'warp`/`'block`/`'grid`)
- **Stage 114** (B.2.b): Borrow check at scope boundaries — `BorrowState.check_borrow_*` consults scope
- **Stage 115** (B.2.c): Phase-typed SMEM — `Smem<f32, Producer>` typestate + `barrier_flip!` primitive
- **Stage 116** (B.2.d): `split_by_thread` view via Presburger injectivity proof
- **Stage 117** (B.3.a): TILE_MATMUL adjoint — emit backward via 3 wmma calls
- **Stage 118** (B.3.b): TILE_ADD adjoint
- **Stage 119** (B.3.c): TILE_REDUCE adjoint
- **Stage 120** (B.3.d): End-to-end MLP forward → backward generated test

### Phase C wedges (~6-7 EM)
- **Stage 121** (C.1): Info-flow typing on TyEnclave — non-coercibility
- **Stage 122** (C.3): Attestation-binding manifest emit (signed ProofObligation)
- **Stage 123** (Backend ROCm.1): tile-IR → AMDGPU/HIP text emit
- **Stage 124** (Backend ROCm.2): ROCm wmma analogs
- **Stage 125** (Backend Metal.1): tile-IR → MSL text emit
- **Stage 126** (Backend Metal.2): Metal Neural Accelerators (M5+)
- **Stage 127** (Backend WebGPU.1): tile-IR → WGSL text emit
- **Stage 128** (Backend WebGPU.2): Tile-loop matmul (no tensor cores)

### Phase A — Substrate (deferred to last for completeness)
- **Stage 129** (A.1): GPU CI scaffolding (mock-GPU validation, no real-HW yet)
- **Stage 130** (A.2): Tile-IR audit per backend — confirm 40 ops decompose cleanly

## Per-stage audit protocol

Each stage runs 3 clean audits before moving on:
1. silent-failure-hunter
2. type-design-analyzer
3. code-reviewer

If any audit surfaces HIGH or MUST-FIX MEDIUM → fix → re-audit until clean.

## End of v2.0 audit gate

5 consecutive clean cycles across all 5 batches (FE/IR/BE/RT/TEST), same protocol as v1.0.

## Stage tracking

| Stage | Title | Status |
|-------|-------|--------|
| 110 | GPU effect labels | PENDING |
| 111 | Stdlib sync annotation | PENDING |
| 112 | Effect propagation tests | PENDING |
| ... | ... | ... |
