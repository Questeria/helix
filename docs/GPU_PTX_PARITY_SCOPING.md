# GPU PTX emitted-text parity — scoping (2026-05-30, counter 412→413)

Scoping pass for the build-order item "GPU (PTX emitted-text parity)". Goal:
determine whether the self-hosting bootstrap can reach PTX parity with the
Python reference, and decompose the remaining work. **Verify-before-claim**:
every finding below was confirmed by running both compilers, not assumed.

## Headline finding — further along than the gap map implied

The bootstrap **already has a working PTX emitter** (`emit_ptx_entry` at
`helixc/bootstrap/kovc.hx:11158`, ~800 emit_ptx_* refs). There are **30
passing bootstrap-vs-golden PTX tests** (`test_bootstrap_ptx_*` in
`helixc/tests/test_codegen.py`, K1.M1…M12+), covering: empty/named/multi
kernels, params, scalar const/add/mul/div/neg/cmp, let-chains, `thread_idx`,
global index, `if`, `while`, global load/store, **f32** load/copy/elementwise-
add, and tile zeros/add/sub/mul/matmul. Harness: `_kovc_self_host_emit_ptx`
(test_codegen.py:7473) drives P0→K1 then calls `emit_ptx_for_ast_to_path`
instead of the ELF emitter, returning PTX *text* (no GPU/ptxas needed — text
shape-check, same fallback Python uses). So GPU PTX is **not** a missing
backend and **not** a from-scratch port.

## The real gap — two non-overlapping kernel DIALECTS

The bootstrap and Python accept *different* `@kernel` surface syntax, so a
golden-fragment test passing does NOT prove cross-compiler parity:

| | Bootstrap dialect (today) | Python dialect (`helixc/backend/ptx.py`) |
|---|---|---|
| param type | bare scalar `a: f32` = "global f32 array" | `a: tile<f32, [256], HBM>` |
| dtype source | param `type_tag` (AST_PARAM slot 4) via `ty_ident_to_tag` | element of `tile<ELEM,…>` |
| return | `-> i32` / `-> f32` | unit only (`RuntimeError: non-unit returns not supported`) |
| body | expression OR stmt (`a[i]`, `out[i]=a[i]+b[i]`) | statements |
| BB labels | none | emits `BB0:` after the reg block |
| PTX version | `.version 8.0` | `.version 8.3` (PTX_VERSION) |

Direct emit-diff (bootstrap vs Python `emit_ptx`, ignoring the `.version`
line) on the SAME `tile<…>` source — the bootstrap parses `tile<f32,…>`
structurally but does **not** descend into `<f32>`, so it falls back to the
i32/u32 path:

```
 kernel: c[i] = a[i] + b[i]   (tile<f32,[256],HBM>)
 python : ld.global.f32 %f0 ; add.f32 %f2 ; st.global.f32       (+ BB0: label)
 bootstrap: ld.global.u32 %r1 ; add.s32 %r3 ; st.global.u32     (no BB0:)
 i32 tile: python ld.global.s32 / st.global.s32  vs  bootstrap .u32  (same-width, cosmetic)
```

## Decomposed parity chunks (for the bootstrap to match Python on Python's dialect)

1. **`tile<ELEM, [SHAPE], SPACE>` param parsing → element dtype** (parser.hx,
   typed-param branch ~13540-13586). The `<…>` is already consumed; capture
   the FIRST generic-arg IDENT and set `type_tag = ty_ident_to_tag(elem)`.
   This is the highest-value chunk: makes `tile<f32>` kernels emit `f32`
   ops (the realistic AI workload). HOT param path → full fixpoint + broad
   regression gate. **Verify by**: bootstrap emits `ld.global.f32` for a
   `tile<f32,…>` kernel (matching Python on the dtype axis).
2. **`BB0:` basic-block label** (kovc.hx PTX kernel emission). Python emits a
   block label after the reg block; bootstrap omits it. Updates the 30
   existing goldens (they predate BB0). Low value alone; bundle with (1)+(3).
3. **i32 load/store signedness `.u32`→`.s32`** (kovc.hx). Cosmetic for
   same-width loads (ptxas treats identically) but text-divergent from Python.
4. **Unit-return `@kernel`** — Python rejects `-> T` kernels; the bootstrap
   uses them. For parity on Python's dialect the bootstrap must accept
   unit-return kernels (minor).
5. **`.version` 8.0 → 8.3** to match `PTX_VERSION` (one string literal +
   golden updates).

Only when (1)-(5) land together does a `tile<…>` kernel emit byte-identical
PTX in both compilers — at which point a DIRECT bootstrap-vs-`emit_ptx`
parity corpus (the stdlib-parity pattern) replaces the golden-fragment tests.

## Not parity gaps (stubbed in Python too — do NOT port as parity)

`TILE_LOAD_GLOBAL/STORE_GLOBAL` (generic SMEM↔HBM), `LOAD_SHARED/STORE_SHARED`,
`TMA_LOAD/STORE`, `BARRIER_WAIT`, `SELECT`, `CALL`, `TILE_REDUCE/TRANSPOSE/
RESHAPE` are `status="stub"` in `helixc/backend/ptx.py` (`PTX_OP_LOWERING`)
and emit a loud `.error "HELIX-STUB"` directive. Both sides "agree" by
stubbing; nothing to reproduce until Python implements them.

## Pointers
- Bootstrap emitter: `helixc/bootstrap/kovc.hx` (`emit_ptx_entry` ~11158).
- Python: `helixc/backend/ptx.py` (1873 lines, `emit_ptx`, `PtxEmitter`);
  pipeline AST→`ir/lower_ast.py`(TIR)→`ir/tile_ir.py`(TileIR)→ptx.
- Harness: `_kovc_self_host_emit_ptx` (test_codegen.py:7473); Python `emit()`
  (test_ptx.py:36). Both text-only, no GPU. See `docs/GPU_DIRECT_EMIT_PLAN.md`.

## Progress update (2026-05-30)

- **Chunk 1 DONE** (commit d7737e5): `parse_fn_decl` extracts the element dtype
  from a `tile<ELEM,…>` param; `tile<f32>` kernels now emit `ld.global.f32`/
  `add.f32` matching Python. +2 regression tests (`ptx_tile_dtype_f32/i32`).
- **Chunks 2-5 = MEASURED COSMETIC, DEFERRED.** A direct unnormalized
  bootstrap-vs-Python `emit_ptx` diff (post-chunk-1) shows tile<> kernels now
  differ ONLY on: `.version` 8.0/8.3, a `BB0:` label, i32 `.u32`/`.s32`
  (same-width, ptxas-identical), and a trailing newline. **None affect SASS** —
  the bootstrap already emits valid, ptxas-accepted PTX. Byte-matching Python
  needs ~25 hand-edits across 21 inline goldens in `test_bootstrap_ptx_*`
  (high churn, cosmetic value), so per loop discipline it is DEFERRED to a
  single batch-regenerated finalization pass right before the GPU audit (then
  replace golden-fragment tests with a DIRECT bootstrap-vs-`emit_ptx` corpus).
  Chunk 4 (unit-return `@kernel`) already works. Verified emitter sites for the
  future pass are recorded in task #11. NOT correctness-blocking.
