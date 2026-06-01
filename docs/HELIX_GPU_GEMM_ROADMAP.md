# Helix GPU codegen roadmap — elementwise → matmul → tiled GEMM (P5)

**Date:** 2026-06-01 · From a code-architect design pass over the verified first-light baseline. All changes are **PTX-path-only** in `helixc/bootstrap/kovc.hx` (the `@kernel`→PTX path via `emit_ptx_for_ast_to_path:11440`, reachable only when `ast_has_kernel`=1); the x86-64 ELF path kovc self-hosts through is disjoint, so the self-host fixpoint is structurally unaffected — **but every change is still gated** (re-mint + gpu-corpus regression + K2==K3 fixpoint) before commit.

## Baseline (done)
Elementwise f32 kernels proven on the RTX 3070 (vector_add c[7]=21, vector_mul c[7]=98). Every param is emitted `.param .b64` + loaded `ld.param.u64` (`kovc.hx:11404-11414`, `10772-10785`); the kernels only worked because their `n: i32` is unused.

## Step A — scalar-param ABI fix (unblocks reading matrix dims)
- **Edit A1** `emit_ptx_entry` param-emission loop (`kovc.hx:~11397-11414`): branch on the param `type_tag` = `__arena_get(pcur+4)`. **tag==0 (i32 scalar) → `.param .u32 param_N`; all others (f32 array / pointer, which are 64-bit pointers) → keep `.param .b64`.**
- **Edit A2** `emit_ptx_expr` AST_VAR arm (`kovc.hx:~10365`): when `ptx_vtab_lookup` returns -1, fall back to `ptx_param_index(name)`; if it is an i32 param, emit `ld.param.u32 %rN, [param_idx]` into a fresh `%r` and return it (today an unbound var silently yields %r-1 → invalid PTX).
- `cuda_launch.c` needs **no** change (`&N` as a 4-byte int already matches `.param .u32`).
- **Gate:** new PTX for vector_add/mul (the unused `n` becomes `.param .u32`) — ptxas-accept + both still PASS; then K2==K3 fixpoint.
- **Prereq check:** confirm the parser sets AST_PARAM slot-4 type_tag=0 for `i32`, 1 for `f32` on @kernel params (same field `ptx_param_type:10748` already uses).

## Step B — naive matmul (first real compute kernel)
- **Thread decomposition trick:** launch `gridDim.x=M, blockDim.x=N` → `row=block_idx()`, `col=thread_idx()` — avoids the divide/modulo the emitter lacks.
- **Kernel** `helixc/examples/naive_matmul_kernel.hx`: `while k<K { acc = acc + A[row*K+k]*B[k*N+col]; k=k+1 } C[row*N+col]=acc`.
- **Edit B1** AST_INT arm (`kovc.hx:~10361`): f32-literal branch → `mov.f32 %fN, 0f00000000` (reuse `emit_ptx_mov_f_zero`).
- **Edit B2** `emit_ptx_assign` (`kovc.hx:~10690-10703`): branch on float flag `vtab+55` → `mov.f32 %f<rx>,%f<rv>` for f32 accumulator (today hardcoded `mov.s32`). **Risk:** the flag must be read immediately after the single RHS `emit_ptx_expr` with no emit in between (safe in current code).
- **Edit B3** register pool cap `emit_ptx_reg_block` (`kovc.hx:~10209`): 256 → 1024 (text-only; ptxas needs the virtual count, no runtime cost). Or start at 16×16 (fits 256) then upscale.
- Computed indices `row*K+k` already work (`emit_ptx_index_load` lowers the whole idx expr).
- `cuda_launch.c`: add a `matmul` mode (3 matrices, 6 args, MxN grid, cell-by-cell verify vs CPU `tf2d_matmul` `tensor.hx:1112`).
- **Milestone:** naive matmul correct on HW = the first real GPU compute kernel; adversarial audit before trusting.

## Step C — tiled GEMM with shared memory (throughput; ~350 lines)
New PTX constructs: `.shared .b32` decls, `st.shared`/`ld.shared.f32`, `bar.sync 0`. Cleanest surface: a `__tiled_matmul_smem(A,B,C,M,K,N,TILE)` intrinsic dispatched in `emit_ptx_call:11314` to a new `emit_ptx_tiled_matmul_smem`. Cooperative tile load → barrier → partial dot → barrier → next tile. The throughput tier the transformer matmuls need.

## Gating (every kovc.hx change)
Edit → `assemble_k1.py` → seed mints the new kovc (~10 min) → gpu_corpus regression (vector_add/mul PASS) + the new kernel on HW vs CPU oracle → K2==K3 byte-identical self-host fixpoint → commit only if all green → re-mint `_kovc_ptx_driver.bin`. Prefer PTX-only edits; never perturb the ELF path.
