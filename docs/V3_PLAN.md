# Helix v3.0 Implementation Plan

v3.0 — the industrialization release. Opened 2026-05-20, immediately
after v2.5.0 (the PTX register-allocation emitter wiring) shipped.

## Premise & authority

v3.0 replaces two home-grown subsystems with industry-standard
infrastructure:

- **MLIR migration** — the home-grown tile-IR (`helixc/ir/tile_ir.py`,
  ~520 lines, plus `tir.py` ~600) → MLIR dialects.
- **LLVM IR backend** — the hand-rolled x86_64 ELF emitter
  (`helixc/backend/x86_64.py`, ~5500 lines, + `elf_dyn.py`) → textual
  LLVM IR consumed by `opt` + `llc`.

A note of honesty carried forward from the v2.0 research and the v2.x
`V2_PLAN.md` "v3.0 horizon": that research recommended **deferring the
v3.0 rewrite until an anchor customer or a hard performance ceiling
forces it** — the home-grown stack was sufficient for Phase 0 → v2.5.
v3.0 nonetheless proceeds, under the user's explicit standing
authority ("go as far as v3.0 without my approval"). To keep that
proceeding responsible, v3.0 is structured so it is **reversible at
every stage until a single, clearly-marked cutover** (Stage 221) — see
"Migration strategy" below.

## Why now / why these two

- The home-grown tile-IR has no general pass infrastructure, no
  progressive-lowering framework, and a fixed op set. MLIR supplies
  all three, plus reusable upstream dialects (`linalg`, `vector`,
  `gpu`, `llvm`).
- The x86_64 emitter hand-rolls register allocation, instruction
  selection and ELF encoding for one target. LLVM supplies industrial
  regalloc + isel for every target it supports — and the v2.4/v2.5
  linear-scan allocator, while correct, is a fraction of what LLVM's
  allocators do.
- v2.x already retired the two smaller v3.0-deferred items — real-HW
  dispatch (v2.4 item 13) and emitted-kernel register allocation
  (v2.4 item 15 / v2.5 Edit B). What remains is genuinely the "big
  rewrite".

## Migration strategy — parallel-path, parity-gated, additive-first

The single most important constraint: v3.0 must never leave the
compiler broken between stages.

1. **Additive.** Each new subsystem (the LLVM backend, the MLIR path)
   is built as a NEW module ALONGSIDE the existing one. The tile-IR
   and `x86_64.py` keep working untouched until their replacement is
   proven.
2. **Parity-gated.** A replacement is "proven" only when a parity
   harness shows it produces output observably identical to the
   incumbent across the entire `helixc/tests/` program corpus.
3. **One cutover.** Only after a parity gate passes does the incumbent
   get retired — and that retirement (Stage 221) is the ONE
   destructive step, explicitly flagged as a user checkpoint.
4. **External tools are optional at rest.** MLIR and LLVM command-line
   tools (`mlir-opt`, `opt`, `llc`) may be absent on a given machine.
   Every stage ships a mock-validation path that needs no toolchain
   and gates real dispatch behind tool-detection — the exact pattern
   `helixc/backend/gpu_ci.py` already uses for real-HW GPU dispatch.

## Phases

v3.0 sequences the LOWER-risk half first.

### Phase D — LLVM IR backend (the x86_64 replacement)
A new host backend that consumes the existing host IR and emits
textual LLVM IR. Purely additive: `x86_64.py` is untouched. This phase
also stands up the LLVM toolchain integration that Phase E reuses —
MLIR's standard lowering target is the `llvm` dialect → LLVM IR.

### Phase E — MLIR migration (the tile-IR replacement)
A parallel MLIR path: tile-IR → MLIR, MLIR → the existing backends.
Higher risk — the tile-IR is consumed by all five backends, the
adjoint, and every IR pass — so it follows Phase D, reusing Phase D's
LLVM path as MLIR's lowering target and Phase D's parity harness.

### Phase F — Backend unification & cutover
The deferred v2.2 Item 2 (a shared backend Protocol/ABC across all
backends), then the single cutover that retires the incumbents.

## Stages

Numbers are a starting layout; like `V2_PLAN.md` they will grow and
split as work reveals detail. The 200+ range keeps clear of the v2.x
110–131 range.

### Phase D — LLVM IR backend
- **Stage 200** — LLVM IR emitter substrate. Confirm `x86_64.py`'s
  input IR + entry points; create `helixc/backend/llvm_ir.py`; emit
  textual LLVM IR for the scalar core (module header / target triple,
  `define`, integer constants, `add`/`sub`/`mul`, `ret`). Mock-
  validate the `.ll` shape.
- **Stage 201** — LLVM toolchain detection + dispatch. Detect
  `llvm-as` / `opt` / `llc` / `clang`; assemble emitted IR to an
  object behind tool-detection, mirroring `gpu_ci` real-HW dispatch.
- **Stage 202** — control flow: host-IR basic blocks → LLVM IR
  labels, `br`, `phi`.
- **Stage 203** — full scalar op set: comparisons, `select`, `neg`,
  unsigned / narrow-width parity with `x86_64.py`.
- **Stage 204** — memory & aggregates: loads/stores, structs, arrays,
  GEPs (`x86_64.py`'s largest surface — expect sub-stages).
- **Stage 205** — calls & ABI: function calls, calling convention,
  FFI parity.
- **Stage 206** — runtime & intrinsics: panic, bounds checks,
  overflow traps — whatever `x86_64.py` lowers specially.
- **Stage 207** — PARITY GATE. A harness runs every `helixc/tests/`
  program through both the x86_64 and the LLVM path and asserts
  identical observable behaviour.
- **Stage 208** — end-of-Phase-D 5-clean-gate.

### Phase E — MLIR migration
- **Stage 210** — MLIR dependency + dialect-strategy decision.
  Evaluate MLIR Python-binding availability; decide upstream dialects
  vs. a custom Helix dialect vs. hybrid; write the decision record.
- **Stage 211** — Helix MLIR dialect / mapping substrate.
- **Stage 212** — tile-IR → MLIR translation (parallel path).
- **Stage 213** — MLIR → backends: lower MLIR so the 4 GPU backends +
  the Phase-D LLVM backend consume it.
- **Stage 214** — progressive-lowering pass pipeline.
- **Stage 215** — PARITY GATE: MLIR path vs. the home-grown tile-IR
  path, across all backends.
- **Stage 216** — end-of-Phase-E 5-clean-gate.

### Phase F — Unification & cutover
- **Stage 220** — shared backend Protocol/ABC (deferred v2.2
  Item 2): one `Backend` interface across PTX/ROCm/Metal/WebGPU/LLVM.
- **Stage 221** — CUTOVER (destructive; user checkpoint). With parity
  gates 207 + 215 green, retire `x86_64.py` and the home-grown tile-IR
  behind a flag, then remove. Recommend explicit user confirmation
  here even under blanket authority — it is the one irreversible step.
- **Stage 222** — end-of-v3.0 5-clean-gate + tag `v3.0.0`.

## Per-stage audit protocol

Unchanged from v2.x. Each stage closes with a 3-clean audit
(silent-failure-hunter / type-design-analyzer / code-reviewer); any
HIGH or must-fix MEDIUM → fix → re-audit until clean. Each phase and
the v3.0 release close with a 5-clean-gate.

## Test-suite invocation note

The canonical test command is **`pytest helixc/tests/`** — 4031
tests, collects clean. A bare `pytest` from the repo root hits ~51
"import file mismatch" collection errors (it collects modules outside
`helixc/tests/`). v3.0 adds a `pytest.ini` pinning `testpaths` so the
bare command is correct too; the parity harnesses (Stages 207, 215)
depend on a clean, unambiguous full-suite run.

## Stage tracking

| Stage | Title | Ship | Audit | Notes |
|-------|-------|------|-------|-------|
| 200 | LLVM IR emitter substrate | — | — | next |
| 201–208 | Phase D — LLVM IR backend | — | — | planned |
| 210–216 | Phase E — MLIR migration | — | — | planned |
| 220–222 | Phase F — unification & cutover | — | — | planned |

## Status notes

- 2026-05-20 — v3.0 opened. v2.5.0 released (tag `v2.5.0`). This plan
  drafted as the v3.0 scoping pass; `pytest.ini` testpaths fix shipped
  alongside. Next: Stage 200 — the LLVM IR emitter substrate.
