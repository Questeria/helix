# Helix - Stats and Facts

Hard numbers and verifiable facts for the website. This file is a current
snapshot, not a permanent claim; rerun the listed commands before publishing.

## Current Snapshot

Snapshot date: 2026-05-15, Stage 35 restart 28 fix verification.

| Stat | Value | Where it comes from |
|------|-------|---------------------|
| **hex0 binary size** | 299 bytes | `stage0/hex0/hex0.bin` |
| **Total bootstrap bytes you must initially audit** | 299 | The hand-encoded hex0 root |
| **pytest tests collected** | 2,316 | `python -m pytest helixc/tests --collect-only -q -p no:cacheprovider` during restart 28 fix verification |
| **Clean audit gates** | 0/3 | `docs/stage35-progress-2026-05-15.md` |
| **Current stage** | Stage 35 audit cleanup | Stage 35 progress ledger |
| **Backend targets with tests** | x86-64 ELF, PTX text emission | `helixc/backend/` and tests |
| **Optimization passes** | const-fold, CSE, DCE, FDCE, hash-cons | `helixc/ir/passes/` and frontend hash-cons |
| **License** | Apache 2.0 / CC-BY 4.0 / CC0 | source / docs / future weights |

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
