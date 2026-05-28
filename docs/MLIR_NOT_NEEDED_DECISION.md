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

## VERIFIED WITH HARD EVIDENCE (K2.AK, 2026-05-28)

User asked to VERIFY (not assert) that direct-to-chip GPU emission is
actually feasible before relying on the 150-chunk cut. Verification is
DECISIVE and CONFIRMS the decision:

Python helixc has TWO GPU code paths:
  - helixc/ir/mlir/backends.py (6373 LOC) -- the MLIR-based path
  - helixc/backend/{ptx,rocm,metal,webgpu}.py -- DIRECT tile-IR emitters,
    sitting right next to backend/x86_64.py (the CPU backend the
    bootstrap ALREADY reimplemented).

Evidence (grep mlir-refs vs tile_ir-refs per backend):
  backend         LOC    mlir_refs   tile_ir_refs
  x86_64 (CPU)    5517   0           75   <- already mirrored by bootstrap
  ptx (NVIDIA)    1873   0           68
  rocm (AMD)       436   0           55
  metal (Apple)    498   0           53
  webgpu (browser) 398   1*          52
  *the single webgpu "mlir" hit is a DOCSTRING: "Lowering is text-only
   -- no LLVM IR / MLIR detour." i.e. it CONFIRMS no MLIR dependency.

CONCLUSIONS:
1. Direct tile-IR -> GPU-text emission is PROVEN feasible -- it's how
   Python's primary GPU backends already work, with ZERO MLIR. The
   MLIR path is a parallel alternative, not a requirement.
2. The bootstrap should mirror backend/ptx.py (direct tile-IR -> PTX
   text) exactly as it already mirrors backend/x86_64.py (direct ->
   ELF). PTX/WGSL/MSL/HIP are all TEXT output -- strictly SIMPLER than
   the ELF binary emission the bootstrap already does.
3. The 150-chunk MLIR cut STANDS. The MLIR substrate is genuinely not
   on the bootstrap's path.
4. BONUS: the 4 GPU backends total 3,205 LOC -- SMALLER than the single
   5,517-LOC x86_64 backend the bootstrap already reimplemented. So the
   GPU port is LESS work than the CPU codegen already shipped. Estimate
   for P2.2 tightened with higher confidence: ~60-100 chunks (was
   ~80-150), and the architectural risk is now near-zero (existence
   proof in hand).

If I had been WRONG (Python only emitted GPU via MLIR), the fallback
was to build a direct emitter anyway. But the fallback is moot: the
direct emitter already EXISTS in Python and is the reference to port.

## RATIFIED BY USER (2026-05-28)

The user reviewed this decision and approved the direct-to-chip path:
"If you can skip the middle man and talk directly to the chip that is
amazing and the best option." The user-overridable flag below is now
RESOLVED in favor of direct emission. The loop proceeds on direct
GPU emission; the MLIR substrate is NOT ported (deleted at K4). No
further re-litigation needed.

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
