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

Reconciled from `git log` as of 2026-05-19T17:53Z (HEAD `05d1bb9`).
"Shipped" = ship commit landed. "Audited" = 3-clean-audit returned
CLEAN or audit-fix landed. "Deferred" = stage exists in scope but
postponed per a later stage's findings (commits will pick them up
after the v2.0.0 gate).

| Stage | Title | Ship | Audit | Notes |
|-------|-------|------|-------|-------|
| 110 | GPU effect labels | `6887341` | `0199217` + `b32b8ab` (combined w/113 fix-batch) | closed |
| 111 | Stdlib sync annotation | `6d190a7` | `b32b8ab` (combined w/113) | closed |
| 112 | Effect propagation tests | `5ff242a` | inline (parser fix in ship) | closed |
| 113 | Scope-tagged borrows | `0199217` (partial) | `b32b8ab` | closed |
| 114 | Borrow check at scope boundaries | `8707029` | clean (no fix-batch) | closed |
| 115 | Phase-typed SMEM substrate | `353d674` | clean | closed |
| 116 | TyTile phase field | `d6f87f2` | `9a50e02` + `46d8da8` (R2) | closed |
| 117-119 | Tile-IR adjoint table | `0d658a1` | `90a7409` (AdjointRecord + TILE_RESHAPE + drift detector) | closed |
| 120 | End-to-end MLP forward → backward | — | — | **DEFERRED**: needs Stage 120 grad_pass; Stage 130 explicitly deferred |
| 121 | Info-flow typing on TyEnclave | `c866e76` | clean | closed |
| 122 | Attestation-binding manifest emit | `4e4e30b` | not yet | post-v2.0 audit candidate |
| 123 | ROCm/HIP backend substrate | `4786b12` | inline (`_check_rocm_lowering_coverage`) | substrate-closed; wmma stubs |
| 124 | ROCm wmma analogs | — | — | **DEFERRED**: explicit stub in Stage 130 audit matrix |
| 125 | Apple Metal MSL substrate | `d380fdd` | inline (`_check_metal_lowering_coverage`) | substrate-closed |
| 126 | Metal Neural Accelerators (M5+) | — | — | **DEFERRED**: requires M5 HW |
| 127 | WebGPU/WGSL substrate | `1a8eacd` | inline (`_check_webgpu_lowering_coverage`) | substrate-closed |
| 128 | WebGPU tile-loop matmul | — | — | **DEFERRED**: no tensor cores; runtime-only path |
| 129 | GPU CI scaffolding | `1159479` | not yet | post-v2.0 audit candidate |
| 130 | Cross-backend tile-IR audit | `05d1bb9` | **PENDING** (current HEAD, no audit yet) | drift detector self-passes; needs external 3-clean |

## Status notes

### 2026-05-19T17:53Z — concurrent-fire race + v2.0 substrate frontier

The scheduled-task cron loop is firing every 12 min with high
concurrency. In a single ~30-minute window today, fires shipped
Stages 117-119 audit-fix (`90a7409`) → 122 → 123 → 125 → 127 → 129
→ 130 — seven commits, all on `main`, none collided thanks to the
self-contained one-action-per-fire protocol.

One race did occur: this fire (started ~17:18Z on HEAD `0d658a1`)
spent ~10 min on a hung pytest re-running 4 test modules. Meanwhile
another fire shipped `90a7409` covering essentially the same audit
findings (AdjointRecord dataclass + TILE_RESHAPE + the partition
invariant test). When this fire's `git add` ran, the working tree
had been reset and the staged edits were no-ops. Loss = ~one fire's
worth of duplicated audit work; no data lost; main is consistent.

**Lessons (for the cron loop, not for plan content):**
- pytest can hang silently when launched via Bash `run_in_background`;
  prefer foreground with explicit timeout or `pytest -x` for fast-fail.
- Two fires holding the same audit subject WILL race. The race is
  benign because both produce semantically-equivalent diffs, but the
  loser wastes a fire.

**Frontier as of HEAD `05d1bb9`:**

1. Stage 130 needs its 3-clean audit before the v2.0.0 gate. It's
   the most-recent ship commit; protocol says "if most-recent commit
   is a stage shipping AND no audit yet → run 3-clean audit."
2. Stages 122 and 129 also have no explicit audit-fix or closure
   marker — may already be CLEAN (no commit needed) but worth a
   parallel sweep.
3. Stages 120, 124, 126, 128 are intentionally deferred per the
   plan + Stage 130's audit matrix. v2.0.0 ships without them; they
   pick up post-tag.

**Next-fire action:** dispatch 3-clean audit on Stage 130
(silent-failure-hunter + type-design-analyzer + code-reviewer in
parallel on `helixc/backend/tile_ir_audit.py` + its test file).
If clean → ship the v2.0.0 5-clean-gate. If not → ship audit-fix.
