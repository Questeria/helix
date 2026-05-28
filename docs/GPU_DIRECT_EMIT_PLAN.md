# GPU direct-emission plan (P2.2) — Helix talks directly to the chip

## North-star goal (user directive, 2026-05-27)

> "Have Helix wherever possible talk directly to the chips."

Helix already talks **directly to the CPU**: the bootstrap's
`emit_elf_for_ast_to_path` (kovc.hx) emits x86_64 machine code into an
ELF, with **no LLVM and no MLIR** in between. This document is the plan
to do the same for the **GPU**: emit NVIDIA PTX (and then AMD ROCm,
Apple Metal, browser WebGPU) **directly from the bootstrap**, again with
no MLIR detour.

This is the general Helix design principle: *skip the middle man — the
compiler talks straight to the target ISA.* CPU = direct (done). GPU =
direct (this plan). Future accelerators (TPU/NPU/...) = direct, same
pattern.

## Why this is feasible (and SIMPLER than the CPU path)

PTX / ROCm-LLVM-IR-text / Metal MSL / WebGPU WGSL are all **text** target
formats. Emitting them is *strictly simpler* than the ELF binary the
bootstrap already emits:

| Aspect            | x86_64 (done)                  | GPU targets (this plan)        |
|-------------------|--------------------------------|--------------------------------|
| Output            | ELF **binary**                 | **ASCII text**                 |
| Headers/offsets   | ELF + program headers, p-offs  | none (a few `.directive` lines)|
| Relocations       | yes (patch_u64 fixups)         | none                           |
| Assembler/linker  | hand-rolled in kovc.hx         | none — driver JIT-compiles     |
| Emit primitive    | `emit_byte` / `emit_u32_le`    | `emit_ptx_byte` (ASCII)        |

The verified reference (see `docs/MLIR_NOT_NEEDED_DECISION.md`) is
Python's `helixc/backend/ptx.py` (1873 LOC, **0 MLIR refs**, 68 tile-IR
refs). It is a direct tile-IR→PTX-text walker sitting right next to
`backend/x86_64.py` (the CPU backend the bootstrap already mirrors). The
four GPU backends total 3,205 LOC — **smaller** than the single 5,517-LOC
x86_64 backend already reimplemented in the bootstrap. So the GPU port is
*less* work than the CPU codegen already shipped, with the architecture
de-risked by an existence proof.

## Output mechanism (already in the bootstrap)

```
emit_byte(b)                       -> __arena_push(b)   (low byte = one output byte)
write_file_to_arena(path, off, n)  -> flush n arena slots' low bytes to a file
```

PTX emission reuses this verbatim via `emit_ptx_byte` (an alias of
`emit_byte` named for clarity). `main()` / the test harness calls
`emit_ptx_for_ast_to_path(ast_root)` then `write_file_to_arena(".ptx",
start, total)`. Identical shape to the ELF path.

## Current state (what the bootstrap already has)

- **Parser (parser.hx, Stage 33)** already recognizes `@kernel` and
  `@autotune(...)` and stores summary metadata on `AST_FN_DECL`:
  - slot 14 = `is_kernel`
  - slot 15 = `is_autotune`
  - slot 16 = deduped autotune variant product (saturated at 17)
  - slot 17 = autotune parse-error kind
- **Codegen (kovc.hx)** had `autotune_pass` (validation only); full
  kernel codegen was "Python-only for now". **K1.M1 changes that** —
  `emit_ptx_for_ast_to_path` is the first GPU codegen in the bootstrap.
- Tile ops (`__tile_zeros/add/sub/mul/matmul`, K1.F23c–F27) are already
  bootstrap builtins on the **CPU** path (direct x86_64 arena loops).
  The GPU plan re-lowers the same tile-IR to PTX text.

## Per-chunk PTX plan (smallest-first)

| Chunk | Scope | Test |
|-------|-------|------|
| **K1.M1** ✅ | Detect `@kernel` (slot 14) → emit minimal empty-entry module (`.version`/`.target`/`.address_size` + `.visible .entry k() { ret; }`). Entry name hardcoded `k`. | `test_bootstrap_ptx_empty_kernel` — cat `.ptx`, exact-match the 74-byte text. |
| **K1.M2** | Extract the **real** kernel name from `AST_FN_DECL` slots 1/2 (name_start/name_len) + the source buffer; emit `.entry <name>()`. | kernel named `foo` → `.entry foo()`. |
| **K1.M3** | Multiple kernels in one module — walk emits one `.entry` per `is_kernel` fn (mirror `emit_module`'s loop). | 2-kernel source → 2 entries. |
| **K1.M4** | Kernel **params** → `.param .b64 param_0, ...` (one per fn param). | `@kernel fn k(a, b)` → 2 `.param` lines. |
| **K1.M5** | `.reg` file declarations + scalar body: `SCALAR_CONST_INT/FLOAT`, `SCALAR_ADD/SUB/MUL/NEG`, `THREAD_IDX` (`%tid.x`). Mirror ptx.py `emit_op` scalar branches. | scalar-arith kernel → `add.s32`/`mov.s32` text. |
| **K1.M6** | Tile elementwise: `TILE_ZEROS/ADD/SUB/MUL` → PTX loops (re-lower the same tile-IR the CPU path already handles). | tile-add kernel shape-check. |
| **K1.M7** | `TILE_MATMUL` → `wmma.mma.sync` (Tensor Cores), `TILE_INDEX_LOAD/STORE_HBM` → `ld.global/st.global`. | matmul kernel shape-check. |
| **K1.M8** | `main()` output-mode switch: AST has `@kernel` → emit `.ptx`; else emit ELF (today's behavior). Pure-CPU tests unaffected (no `@kernel`). | existing ELF tests stay green; a kernel program emits `.ptx`. |

After PTX, the sibling backends are the same walker against a different
text grammar (each is **smaller** than PTX):

- **K1.N\*** — ROCm/HIP (AMDGPU LLVM-IR text), `backend/rocm.py` (436 LOC).
- **K1.O\*** — Apple Metal MSL, `backend/metal.py` (498 LOC).
- **K1.P\*** — Browser WebGPU WGSL, `backend/webgpu.py` (398 LOC).

## Test strategy

- **Shape-check the emitted text** (cat the `.ptx`/`.wgsl`/... and assert
  the bytes) — exactly how Python's `ptx.py` validates when `ptxas` is
  unavailable. No GPU hardware required in CI.
- **ptxas round-trip** (optional, later) — if `ptxas` is on the box, pipe
  the emitted PTX through it to confirm it assembles to SASS. Gated on
  tool availability so CI without CUDA still passes.
- Self-host tests are **slow** (full Stage-30 bootstrap compile per test);
  run sequentially (`-p no:xdist`) per the regression policy.

## Safety notes (K1.F5d-j lesson)

- `emit_ptx_*` is **pure-additive codegen** in kovc.hx. It touches **no
  sb scratch slots** and **no parser state** — the sb-region collision
  hazard that broke K1.F5d-j does **not** apply here.
- K1.M8 is the only chunk that changes `main()`'s control flow; it gates
  PTX strictly on `is_kernel`, so the existing ELF tests (no `@kernel`)
  take the unchanged path. Run a sequential regression after K1.M8.

## Relationship to the hard constraint

The 2026-05-26 hard constraint ("zero non-Helix at v1.0; GPU/MLIR/Tile
ops MUST be ported, no defer-to-Python-forever") is satisfied by this
plan: the *capability* (driving the GPU) is ported to the bootstrap via
direct emission. The Python MLIR substrate is **deleted** at K4, not
ported — the bootstrap reaches the same end (GPU codegen) without it.

## K1.M21 parity finding (2026-05-28) — non-NVIDIA backends are scaffolds

After scaffolding all 4 GPU backends in the bootstrap (PTX full;
WebGPU/Metal/ROCm empty-kernel, M18–M20), a parity probe of the Python
reference backends revealed the honest state:

- **The Python non-NVIDIA backends (`webgpu.py`/`metal.py`/`rocm.py`) are
  SUBSTRATE + STUBS, not functional compilers.** They emit a valid module
  header + kernel entry, but for actual ops they emit `@@HELIX-STUB`
  parse-breaking tokens. Probe: `WgslEmitter` on a `tile<f32,[256],HBM>`
  kernel doing `a[i] = a[i]` emits `@@HELIX-STUB: TileOpKind.
  TILE_INDEX_LOAD_HBM status='stub' not wired` (+ the store) — no real
  WGSL load/store. The op-mapping tables are mostly `status="stub"`.
- **Only NVIDIA PTX is a real, functional GPU compiler** (both in Python
  and now the bootstrap, where it is ptxas-validated through matmul).
- **Empty-kernel byte-parity** (bootstrap == Python module skeleton) is
  achieved for all 4 (M18–M20). **Op-level byte-parity is infeasible**:
  Python's stub tokens reference tile-IR `TileOpKind` names, but the
  bootstrap lowers AST-direct (no tile-IR), so the two cannot byte-match
  on op kernels.

**Implications:**
1. For the **Python-deletion bucket**, the bootstrap's non-NVIDIA
   backends already match the Python backends' *functional* capability
   (both: substrate only, zero real non-NVIDIA ops). Nothing is lost by
   deleting the Python scaffolds.
2. The **north-star** ("real AI on ANY GPU, incl. non-NVIDIA") is
   *unrealized in Python too* — real WGSL/MSL/GCN op lowering is an
   **unbuilt from-scratch arc**, not a port. The bootstrap PTX is the
   only path that talks to a chip with real compute today.
3. To genuinely deliver the north-star for non-NVIDIA, the bootstrap must
   **EXCEED** Python: build AST-direct op lowering (params → storage
   buffers, global load/store, tile ops) per backend, mirroring the
   bootstrap PTX arc, validated by shape (+ a real validator like `naga`
   for WGSL if available — there is no Python byte-match oracle since
   Python stubs everything).

**Plan from here:** pursue real WGSL op lowering next (WebGPU is the most
portable target — runs on any GPU), starting with kernel params +
global load/store (the memory foundation). This is net-new capability
beyond the Python reference, and the highest-leverage way to make
"AI on any GPU" real rather than scaffolded.
