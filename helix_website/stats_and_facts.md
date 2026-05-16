# Helix - Stats and Facts

Hard numbers and verifiable facts for the website. This file is a current
snapshot, not a permanent claim; rerun the listed commands before publishing.

## Current Snapshot

Snapshot date: 2026-05-16. **Stage 35 CLOSED at restart 65** (Increment 82 — three consecutive all-clean fresh audits on top of substantive HEAD `e441173`). Use live `git log -1 --oneline` before publishing.

| Stat | Value | Where it comes from |
|------|-------|---------------------|
| **hex0 binary size** | 299 bytes | `stage0/hex0/hex0.bin` |
| **Total bootstrap bytes you must initially audit** | 299 | The hand-encoded hex0 root |
| **pytest tests collected** | 2,556+ | `python -m pytest helixc/tests --collect-only -q` at restart 65 (Stage 35 closure). See Increments 70 onward in the progress ledger for the per-restart canary chain since restart 50; Increments 80 + 81 + 82 are the three consecutive clean-gate records that closed Stage 35. |
| **Clean audit gates** | **3/3 — Stage 35 CLOSED** | `docs/stage35-progress-2026-05-15.md` Increment 82 |
| **Current stage** | Stage 35 CLOSED; Stage 36 opens next | Stage 35 progress ledger Increment 82 |
| **Backend targets with tests** | x86-64 ELF, PTX text emission | `helixc/backend/` and tests |
| **Optimization passes** | const-fold, CSE, DCE, FDCE, hash-cons | `helixc/ir/passes/` and frontend hash-cons |
| **License** | Apache 2.0 (in `LICENSE`); CC-BY 4.0 and CC0 are stated policy, not yet file-resident | source / docs / future weights |

## Important Honesty Notes

- The current production compiler implementation is Python-hosted `helixc`.
- A self-hosted Helix compiler is still the bootstrap target, not the shipped
  replacement for Python yet.
- PTX support currently means text emission for covered kernels; GPU execution
  is not a finished public capability.
- Test counts change frequently because audit fixes add regression tests.

## Website-Safe Story Arcs

1. **"From 299 bytes to a compiler"** - the bootstrap journey.
2. **"How Helix differentiates"** - autodiff, effects, tensor shapes, and auditability.
3. **"Why no silent corruption"** - the trap-id and audit-driven regression philosophy.
4. **"Tile types that mean something"** - compile-time shape checking and PTX-oriented work.
5. **"Reflection with verifier gates"** - constrained self-modification as a design goal.
6. **"The bootstrap audit trail"** - every step should be inspectable and reproducible.
