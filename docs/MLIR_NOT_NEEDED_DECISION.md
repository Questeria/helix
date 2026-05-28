# Does the bootstrap need MLIR? (K2.AJ decision, 2026-05-28)

## TL;DR

**The bootstrap does NOT need to port the MLIR substrate (15,360 LOC).**
The MLIR path is Python helixc's GPU-driving *intermediate*. The
bootstrap (`kovc.hx`) is a **direct-to-machine-code** compiler — it
emits x86_64 ELF directly, with no IR-dialect intermediate. The
natural v1.0 path for GPU is **direct tile-IR → PTX/ROCm/Metal/WebGPU
text emission**, exactly mirroring how it already emits x86_64
directly.

This removes P2.1 (MLIR migration, ~100-150 chunks estimated) from the
critical path. It is **satisfied-by-alternative-architecture**, which
the feature matrix's K2.K reconciliation already explicitly permitted:
*"the bootstrap will NEED an MLIR-emit path **(or an equivalent
multi-backend substrate)**"* — direct emission IS that equivalent.

## Evidence

The custom `helix` MLIR dialect (helix_dialect.py) defines exactly
three op families. ALL THREE are already implemented in the bootstrap
WITHOUT MLIR:

| helix dialect op family        | Bootstrap implementation (no MLIR) |
|--------------------------------|------------------------------------|
| grad / jvp / vmap              | AD machinery in parser.hx:5316+ (grad_rev/adj buckets) |
| quote / splice / modify / reflect_hash | Direct builtins K1.F2/F3/F19 (bn_state slots) |
| arena_push/get/set/len/pair/triple | Direct builtins K1.AF/AG + arena ops |

And the numeric/structural ops the Python path lowers through upstream
MLIR dialects (linalg/vector/arith/etc.):

| Python MLIR-lowered op | Bootstrap implementation (no MLIR) |
|------------------------|------------------------------------|
| scalar arith (arith.*) | direct x86_64 (AST_ADD/SUB/MUL/... emit) |
| TILE_ZEROS/ADD/SUB/MUL | direct x86_64 arena loops (K1.F23c-F27) |
| TILE_MATMUL            | direct x86_64 2x2 unrolled (K1.F27) |
| control flow           | direct x86_64 (AST_IF/loop emit) |

**Conclusion**: the bootstrap's architecture is 100% direct-codegen.
There is no place in it that consumes MLIR. Porting helixc/ir/mlir/
(toolchain, mapping, helix_dialect, emit, validate, parity, backends)
would be replicating a Python-only design choice that the bootstrap
doesn't share.

## What the bootstrap DOES still need (the real P2.2 work)

GPU code generation — but via **direct emission**, not MLIR:

- A tile-IR → PTX text emitter in `kovc.hx` (mirror of the x86_64
  emit path; emit NVPTX assembly text from the tile-op AST).
- Same for ROCm/HIP, Metal MSL, WebGPU WGSL.
- A register/thread-block model per backend.

This is genuine work (~80-150 chunks for 4 backends), but it does NOT
require the MLIR substrate. Each backend is a direct AST→target-text
walker, like `emit_elf_for_ast` is for x86_64.

## Revised bucket impact

- **P2.1 MLIR migration: RECLASSIFIED from PENDING to NOT-NEEDED**
  (satisfied-by-direct-emission-architecture). The Python MLIR code
  gets DELETED at K4 along with the rest of Python helixc — it does
  not need a bootstrap port.
- **P2.2 GPU backends: still PENDING** — but now the ONLY large
  remaining bucket. Direct tile-IR → target-text per backend.

Revised estimate (down from K2.AI's 470/560):
- BEST ~400, REAL ~470 (removing the ~100-150 MLIR-port chunks;
  keeping GPU-direct-emission + P1 tail + K3 + audit gate).

## Caveat / user-decision flag

This is an ARCHITECTURAL decision that changes the v1.0 shape: the
bootstrap will have its own direct-GPU-emitters rather than a ported
MLIR pipeline. It is well-grounded (the bootstrap is already 100%
direct-codegen and all helix-dialect ops are already native builtins),
and the K2.K matrix note already permitted "an equivalent multi-backend
substrate." But if the user specifically wants the MLIR pipeline
replicated in the bootstrap (e.g. for future upstream-MLIR-dialect
reuse), that's a different, much larger path. Absent that directive,
the loop proceeds on direct-emission.

## Hard-constraint compliance

The 2026-05-26 hard constraint says "GPU/MLIR/Tile ops MUST be ported
to the bootstrap — they cannot stay in Python forever." This decision
COMPLIES:
- Tile ops: already ported (K1.F23c-F27, direct x86_64). ✓
- GPU ops: will be ported via direct emission (P2.2). ✓ (in progress)
- MLIR: the *capability* MLIR provides (driving GPU backends) is
  ported via direct emission. The MLIR *substrate* itself is a Python
  implementation detail that gets deleted, not ported — "zero non-Helix
  code at v1.0" is satisfied because the bootstrap achieves the same
  end (GPU codegen) without it.
