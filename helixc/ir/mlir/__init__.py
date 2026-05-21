"""
helixc/ir/mlir/ — the v3.0 Phase E MLIR migration package.

Phase E (docs/V3_PLAN.md, Stages 210-216) migrates Helix's home-grown
Tensor IR (`helixc/ir/tir.py`) and Tile IR (`helixc/ir/tile_ir.py`) to
MLIR, per the ratified Stage 210 decision
(docs/V3_STAGE210_MLIR_DECISION.md): a HYBRID dialect strategy — a
small custom `helix` dialect over upstream MLIR dialects — built
MOCK-PATH-FIRST so the compiler imports and runs on a machine with no
MLIR bindings installed.

HARD RULE (Stage 210 decision, section 3.2): no module in this package
— or anywhere in `helixc/` — may `import mlir` at module top level.
Every MLIR import is lazy, inside a capability-probed code path, so
`import helixc.ir.mlir...` always succeeds even when the bindings are
absent.

License: Apache 2.0
"""
