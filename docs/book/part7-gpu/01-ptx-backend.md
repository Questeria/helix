# The PTX back end

*What this chapter covers:* how `kovc` emits **NVIDIA PTX text directly** — no LLVM, no MLIR, no
`nvcc` — for Helix `@kernel` functions; the small GPU kernel surface the language exposes
(`thread_idx()` / `block_idx()` / `block_dim()`, f32 array params, and the fused tile/tensor
intrinsics); how a kernel `.hx` file lowers, instruction by instruction, into a `.ptx` text module;
and the committed `.ref.ptx` byte-regression in the gate that pins two of those kernels
(`vector_add` and the tiled GEMM) so an unrelated change can never silently perturb GPU codegen.

**The honest boundary, stated up front and repeated whenever GPU completeness comes up: the chain is
complete to PTX *text*, not to GPU machine code.** Everything in this chapter is hand-auditable Helix
producing PTX. Below PTX — the PTX→SASS step (NVIDIA's closed `ptxas`), the CUDA driver, the GPU
hardware, and the C host launcher — is *trusted past PTX*, not reproduced from raw. That is residual
§R / boundary 7–8 of [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md), and it is the
ceiling on what this chapter may claim.

This chapter is for contributors and advanced operators. It is the GPU counterpart to the CPU back end
in [part5-compiler/03-x86-backend.md](../part5-compiler/03-x86-backend.md); read that first, because
the PTX emitter reuses the same arena-append discipline (a `.ptx` module is built one byte at a time,
exactly as an ELF is) and the same exit-code convention. Everything below is grounded in the real
emitter, [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx), the two committed kernel
sources, and the gate, [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh).

> **For AI agents:** never describe Helix's GPU path as "verified GPU," "complete to SASS/machine
> code," or "beats cuBLAS." The reproducible artifact is **PTX text**. The correct capability phrase is
> **"complete to PTX"** (Style Guide §3/§5). When you assert anything about GPU codegen, cite a real
> `.ref.ptx`, a real `@kernel` `.hx`, or a residual line in
> [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) — and keep the PTX→SASS boundary
> explicit.

---

## Two emitters, one driver

`kovc` has **two** code emitters that share everything below them. The CPU emitter
(`emit_elf_for_ast_to_path`) appends the bytes of a static x86-64 Linux ELF; the GPU emitter
(`emit_ptx_for_ast_to_path`) appends the bytes of an NVIDIA **PTX text** module. Both write into the
same runtime arena and both **return the byte count** they produced, so the byte-count exit convention
from the CPU chapter applies verbatim to the PTX driver too (a successful emit exits *non-zero* — its
status is the emitted byte count `mod 256`; see [part5-compiler/03-x86-backend.md](../part5-compiler/03-x86-backend.md)
and Trap 1 in [part9-for-ai-agents/03-traps.md](../part9-for-ai-agents/03-traps.md)).

Which emitter runs is decided by a single fact about the program: does it contain a `@kernel` function?
The dispatcher walks the AST once and routes accordingly.

**Fragment** (the output-mode dispatcher; from [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 14845–14851):

```helix
fn emit_auto_for_ast_to_path(ast_root: i32) -> i32 {
    if ast_has_kernel(ast_root) == 1 {
        emit_ptx_for_ast_to_path(ast_root)
    } else {
        emit_elf_for_ast_to_path(ast_root)
    }
}
```

`ast_has_kernel` walks the `AST_FN_LIST` (tag `15`) and checks each function's `is_kernel` flag —
slot `14` of an `AST_FN_DECL`, set by the parser when it sees the `@kernel` attribute. A pure-CPU
program has no kernel, so `emit_ptx_for_ast_to_path` returns `0` and nothing PTX is produced; a program
with one or more `@kernel`s emits a PTX module instead of an ELF. The source comment states the
design directly:

**Fragment** (the routing rationale; from [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 14839–14844):

```helix
// K1.M17: output-mode dispatcher -- emit GPU PTX when the program has a
// @kernel, else x86_64 ELF. The bridge from "two separate emitters" to a
// driver that picks the right target: a real compiler entry calls THIS.
// Both emitters write into the arena + return the byte count; the caller
// writes the file. Like a host toolchain routing .cu -> ptx vs .c -> elf,
// but with NO CUDA / NO MLIR -- direct Helix -> chip on either path.
```

> **For AI agents:** the GPU vs CPU choice is **structural**, not a flag. A `.hx` file with a `@kernel`
> compiles to PTX; one without compiles to ELF. There is no `--emit=ptx` switch — write a `@kernel`, or
> route through the PTX driver `main` used by the gate (`k1ptxdrv.hx`; see the regression section).

The byte-stream primitive is the same `__arena_push`-of-one-byte mechanism the ELF emitter uses; for
PTX it is `emit_ptx_byte`, and the whole `.ptx` text — every keyword, register name, and newline — is
spelled out as ASCII codes through it:

**Fragment** (the PTX byte primitive; from [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 10751–10754):

```helix
fn emit_ptx_byte(b: i32) -> i32 {
    __arena_push(b);
    0
}
```

There is no instruction encoder, no string library, and no template engine. A line like
`mov.u32 %r0, %tid.x;` is emitted as the literal byte sequence `109 111 118 ...` (`m`, `o`, `v`, …).
This is the same "the arena *is* the assembler" philosophy as the CPU back end — applied to *text*
output instead of binary.

---

## The kernel surface in Helix

The GPU surface a kernel author touches is deliberately tiny. A kernel is an ordinary Helix `fn`
annotated `@kernel`; inside it, a few **GPU intrinsics** are recognised by name and lowered to PTX
special registers, and array indexing on the f32 pointer parameters lowers to global loads/stores. The
language spec records the surface as `[impl]` (implemented), proven on hardware via the capstone:

**Fragment** (the GPU surface, per the spec; from [`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) lines 111 and 153):

```text
- **`@kernel` function**: GPU kernel, params may be `tile<…>` / `f32` arrays / `i32` scalars;
  emitted as **PTX** ...
- **GPU**: textual **PTX** for `@kernel` functions (one+ `.entry` per module; the C launcher loads
  the module and `cuLaunchKernel`s each). Scalar ops, `threadIdx.x`/`blockIdx.x`, the math intrinsics
  above.
```

The three index intrinsics are the entry point to every data-parallel kernel. `kovc` recognises them
by an exact byte-string name match and emits the corresponding PTX special register:

| Helix intrinsic | PTX special register | meaning |
|-----------------|----------------------|---------|
| `thread_idx()`  | `%tid.x`             | thread index within the block |
| `block_idx()`   | `%ctaid.x`           | block (CTA) index within the grid |
| `block_dim()`   | `%ntid.x`            | threads per block |

The recognition is a flat byte comparison — there is no symbol table for these; the compiler matches
the literal characters of the call name. The `thread_idx` matcher and its emitter are representative of
all three:

**Fragment** (recognising `thread_idx()` and lowering it to `%tid.x`; from [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 11707–11724 and 11730–11744):

```helix
fn ptx_name_is_thread_idx(name_s: i32, name_l: i32) -> i32 {
    if name_l != 10 {
        0
    } else {
        let mut ok: i32 = 1;
        if __arena_get(name_s + 0) != 116 { ok = 0; };   // t
        if __arena_get(name_s + 1) != 104 { ok = 0; };   // h
        if __arena_get(name_s + 2) != 114 { ok = 0; };   // r
        if __arena_get(name_s + 3) != 101 { ok = 0; };   // e
        if __arena_get(name_s + 4) != 97 { ok = 0; };    // a
        if __arena_get(name_s + 5) != 100 { ok = 0; };   // d
        if __arena_get(name_s + 6) != 95 { ok = 0; };    // _
        if __arena_get(name_s + 7) != 105 { ok = 0; };   // i
        if __arena_get(name_s + 8) != 100 { ok = 0; };   // d
        if __arena_get(name_s + 9) != 120 { ok = 0; };   // x
        ok
    }
}
```

The kernel-body expression walker, `emit_ptx_call`, dispatches a recognised call to its register emit:

**Fragment** (intrinsic dispatch in the kernel body; from [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 14580–14591):

```helix
    if ptx_name_is_thread_idx(name_s, name_l) == 1 {
        let r = ptx_alloc_reg(vtab);
        emit_ptx_tid_x(r);
        r
    } else { if ptx_name_is_block_idx(name_s, name_l) == 1 {
        let r = ptx_alloc_reg(vtab);
        emit_ptx_mov_ctaid_x(r);
        r
    } else { if ptx_name_is_block_dim(name_s, name_l) == 1 {
        let r = ptx_alloc_reg(vtab);
        emit_ptx_mov_ntid_x(r);
        r
    } else { ...
```

Beyond the index intrinsics, the same dispatcher recognises a set of **GPU math** intrinsics
(`__gpu_exp`, `__gpu_rsqrt`, `__gpu_i2f`) and the **tile/tensor** intrinsics (`__tile_zeros`,
`__tile_add`/`__tile_sub`/`__tile_mul`, `__tile_matmul`, and the fused
`__tiled_matmul_smem` discussed below) — each lowered by its own `emit_ptx_*` routine
([`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 14592–14629). These are the
building blocks the capstone's kernels are written from. This chapter focuses on the two kernels the
gate byte-regresses; the broader op set and the GEMM/transformer path are covered in the next chapter.

> **For AI agents:** the GPU intrinsics are matched by **exact name and exact length** (`thread_idx`
> is length 10; a misspelling or a different arity will *not* match and will fall through to the
> generic-call path). When you author or edit a kernel, use these names verbatim — do not invent
> `threadIdx_x()` or `gpu_exp()`; the spec names are the ones in
> [`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) and the matchers above.

---

## A kernel `.hx`, lowered: `vector_add`

The simplest committed kernel is an elementwise add, `c[i] = a[i] + b[i]` over f32 global arrays. It is
the GPU first-light kernel and the smaller of the two the gate pins. Quote it whole:

**Verified example** — [`helixc/examples/vector_add_kernel.hx`](../../../helixc/examples/vector_add_kernel.hx)
(a `@kernel`; the gate emits its PTX from the re-minted driver and asserts that PTX is **byte-identical**
to the committed [`helixc/examples/vector_add_kernel.ref.ptx`](../../../helixc/examples/vector_add_kernel.ref.ptx),
gate step `[3]`):

```helix
@kernel
fn vector_add(a: f32, b: f32, c: f32, n: i32) {
    let i = block_idx() * block_dim() + thread_idx();
    c[i] = a[i] + b[i]
}
```

Two design facts from the file's own header comment ([`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx)
is the emitter; the comment lives in the kernel source): the kernel is **deliberately non-generic** —
the PTX emitter has no monomorphize pass, so a generic `@kernel` would mis-lower — and the count
parameter `n` is present only so the host argument array lines up positionally; it is **unused** in the
body (with `N=256` and one block of 256 threads, every index is in-bounds, so no `if i < n` guard is
needed). The index expression `block_idx() * block_dim() + thread_idx()` is the canonical grid-stride
index, and it is exactly the pattern the emitter is built for.

Now read what `kovc` emits. The committed reference is 945 bytes; here it is verbatim:

**Fragment** (the committed reference PTX for `vector_add`; from [`helixc/examples/vector_add_kernel.ref.ptx`](../../../helixc/examples/vector_add_kernel.ref.ptx), all 35 lines):

```text
.version 8.3
.target sm_86
.address_size 64

.visible .entry vector_add(.param .b64 param_0, .param .b64 param_1, .param .b64 param_2, .param .u32 param_3)
{
    .reg .pred  %p<256>;
    .reg .b32   %r<256>;
    .reg .b64   %rd<256>;
    .reg .f32   %f<256>;
    .reg .b16   %h<256>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mul.lo.s32 %r2, %r0, %r1;
    mov.u32 %r3, %tid.x;
    add.s32 %r4, %r2, %r3;
    ld.param.u64 %rd0, [param_0];
    cvta.to.global.u64 %rd1, %rd0;
    mul.wide.s32 %rd2, %r4, 4;
    add.s64 %rd3, %rd1, %rd2;
    ld.global.f32 %f0, [%rd3];
    ld.param.u64 %rd4, [param_1];
    cvta.to.global.u64 %rd5, %rd4;
    mul.wide.s32 %rd6, %r4, 4;
    add.s64 %rd7, %rd5, %rd6;
    ld.global.f32 %f1, [%rd7];
    add.f32 %f2, %f0, %f1;
    ld.param.u64 %rd8, [param_2];
    cvta.to.global.u64 %rd9, %rd8;
    mul.wide.s32 %rd10, %r4, 4;
    add.s64 %rd11, %rd9, %rd10;
    st.global.f32 [%rd11], %f2;
    ret;
}
```

You can read the lowering off the Helix source line by line:

- **The module header** — `.version 8.3` / `.target sm_86` / `.address_size 64` — is emitted once per
  module by `emit_ptx_for_ast_to_path`. The `.version` is PTX ISA 8.3 (the TF32 Tensor-Core op-set
  needs it) and `.target sm_86` is the reference RTX 3070 Laptop GPU (compute capability 8.6). Both are
  *hardcoded* — there is no architecture flag — which is one of the residuals: a **single GPU target**
  (§R / boundary 6 of [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md)).
- **The `.entry` signature** carries one `.param` per Helix parameter: the three f32 array pointers as
  `.param .b64`, and `n` as `.param .u32`. The entry name is the kernel's real name, `vector_add`.
- **The register-file block** declares the five over-allocated PTX register pools
  (`%p`/`%r`/`%rd`/`%f`/`%h`, each `<256>`). Declaring a large pool is free — `ptxas` allocates only
  the registers actually used — so the emitter declares a fixed pool and never does live-range
  allocation. (This fixed `%r<256>` regfile, with no common-subexpression elimination, is one of the
  fingerprints that the PTX is *kovc's own* and not smuggled `nvcc` output;
  [`docs/HELIX_GPU_FIRSTLIGHT.md`](../../../docs/HELIX_GPU_FIRSTLIGHT.md).)
- **The index** `block_idx() * block_dim() + thread_idx()` becomes `mov.u32 %r0, %ctaid.x` /
  `mov.u32 %r1, %ntid.x` / `mul.lo.s32 %r2, %r0, %r1` / `mov.u32 %r3, %tid.x` /
  `add.s32 %r4, %r2, %r3` — the three intrinsics and the two arithmetic ops, in source order.
- **Each array access** lowers to the same four-instruction address computation, re-emitted *per
  access* with no CSE: load the pointer param (`ld.param.u64`), convert it to a global address
  (`cvta.to.global.u64`), scale the index by the element size (`mul.wide.s32 …, 4` — f32 is 4 bytes),
  and add (`add.s64`). Then `ld.global.f32` for a read, `st.global.f32` for the write. You can see this
  exact shape three times in the PTX: once for `a[i]`, once for `b[i]`, once for `c[i]`.

That per-access address computation is `emit_ptx_index_load` in the emitter; the comments in it spell
out each PTX line it appends:

**Fragment** (per-access global-address lowering; from [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 11512–11533):

```helix
fn emit_ptx_index_load(node: i32, vtab: i32) -> i32 {
    let base = __arena_get(node + 1);
    let idx_node = __arena_get(node + 2);
    let fn_idx = __arena_get(vtab + 53);
    let pidx = ptx_param_index(fn_idx, __arena_get(base + 1),
                               __arena_get(base + 2));
    let ri = emit_ptx_expr(idx_node, vtab);
    // "    ld.param.u64 %rd<rdb>, [param_<pidx>];\n"
    let rdb = ptx_alloc_rd(vtab);
    ...
    // "    cvta.to.global.u64 %rd<rdg>, %rd<rdb>;\n"
```

The mapping from a Helix index expression to that `ld.param → cvta.to.global → mul.wide.s32 ·,4 →
add.s64` quartet is the heart of the elementwise path, and matching it byte-for-byte against the
committed reference is what the gate's PTX regression proves.

---

## A fused kernel: the tiled GEMM

The second kernel the gate pins is the throughput-path matrix multiply: a shared-memory **tiled** GEMM,
`C = A*B`. Unlike `vector_add`, almost none of its PTX comes from a line-by-line walk of the body —
instead, a single **fused intrinsic** call, `__tiled_matmul_smem`, expands into the *entire* tiled
kernel. Quote the source whole:

**Verified example** — [`helixc/examples/tiled_matmul_kernel.hx`](../../../helixc/examples/tiled_matmul_kernel.hx)
(a `@kernel`; the gate emits its PTX and asserts byte-identity to the committed
[`helixc/examples/tiled_matmul_kernel.ref.ptx`](../../../helixc/examples/tiled_matmul_kernel.ref.ptx)
**and** checks instruction-class provenance in the output, gate step `[3]`):

```helix
@kernel
fn tiled_matmul(a: f32, b: f32, c: f32, mm: i32, kk: i32, nn: i32) {
    __tiled_matmul_smem(a, b, c, mm, kk, nn)
}
```

The kernel body is one call. The emitter behind it generates the cooperative GMEM→SMEM staging, the
barrier, the runtime k-tile loop, the register micro-tile FMA accumulate, and the epilogue store — the
whole shape that a register-only kernel cannot express. The source comment documents the fusion and the
tile geometry:

**Fragment** (what the fused intrinsic expands to; from [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 12245–12259):

```helix
// intrinsic __tiled_matmul_smem(a, b, c, M, K, N) lowers to an ENTIRE
// cooperative-staging tiled f32 matmul kernel: .shared tile decls +
// GMEM->SMEM cooperative load + bar.sync + a runtime k-tile loop + a
// register micro-tile FMA accumulate + an epilogue global store. This
// is the load-bearing new GPU capability (real PTX loops with a
// tid->shared-address mapping the register-tile model cannot express).
// ZERO parser change: __tiled_matmul_smem parses as AST_CALL today, the
// same path as __tile_matmul; only emit_ptx_call name-matching + this
// emitter are new. emit_ptx_entry/emit_ptx_reg_block are UNTOUCHED (the
// .shared decls are emitted at the top of this intrinsic ...
// Tile params for the RTX 3070 (sm_86), correctness-first:
//   BM=BN=64, BK=8, TM=TN=4, threadblock 16x16=256, grid=(N/BN, M/BM).
```

The emitted reference is far larger than `vector_add` — 485 lines — so it is not quoted in full here;
its head shows the new structural features the elementwise path never produces:

**Fragment** (head of the committed tiled GEMM reference; from [`helixc/examples/tiled_matmul_kernel.ref.ptx`](../../../helixc/examples/tiled_matmul_kernel.ref.ptx) lines 1–22):

```text
.version 8.3
.target sm_86
.address_size 64

.visible .entry tiled_matmul(.param .b64 param_0, .param .b64 param_1, .param .b64 param_2, .param .u32 param_3, .param .u32 param_4, .param .u32 param_5)
{
    .reg .pred  %p<256>;
    .reg .b32   %r<256>;
    .reg .b64   %rd<256>;
    .reg .f32   %f<256>;
    .reg .b16   %h<256>;

    .shared .align 16 .b8 smem_a0[2048];
    .shared .align 16 .b8 smem_a1[2048];
    .shared .align 16 .b8 smem_b0[2048];
    .shared .align 16 .b8 smem_b1[2048];
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %tid.y;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ctaid.y;
    ld.param.u32 %r4, [param_4];
    ld.param.u32 %r5, [param_5];
```

Three things in that head are specific to the tiled path and absent from `vector_add`:

- **`.shared` tile declarations.** Four ping-pong shared-memory buffers (`smem_a0`/`smem_a1`/`smem_b0`/
  `smem_b1`, each 2048 bytes, 16-byte aligned). The pairs implement a **double-buffered** stage: while
  one tile is consumed, the next is being loaded. `vector_add` has no `.shared` at all.
- **2-D indexing.** `%tid.y` and `%ctaid.y` appear (the threadblock is 16×16), where the elementwise
  kernel used only the `.x` dimension.
- **`u32` scalar params loaded into registers.** `kk` and `nn` (the K and N dimensions) are read with
  `ld.param.u32` because the tiled loop derives addresses from the runtime matrix dimensions — unlike
  `vector_add`, whose `n` is unused.

Deeper in the reference (not quoted) are the double-buffered async copies and the barrier: the gate's
provenance check requires the literal tokens `cp.async.cg.shared.global`, `cp.async.commit_group`,
`cp.async.wait_group`, `.shared`, and `bar.sync 0` to be present **in the emitted output** (gate step
`[3]`). That double-buffer `cp.async` form is why the reference was deliberately re-minted at the T2/G2
milestone — an *intentional* PTX change, re-committed with a recorded reason, which is the only
sanctioned way the reference is allowed to move (more on that next).

> **For AI agents:** `__tiled_matmul_smem` is a **whole-kernel fused intrinsic**, not a library call —
> one call in the body becomes hundreds of PTX lines. Do not expect to read its lowering by tracing the
> `.hx` body statement-by-statement (as you can with `vector_add`); the body *is* the single intrinsic.
> The geometry is fixed (`BM=BN=64, BK=8, TM=TN=4`) and the kernel requires `M%64==N%64==K%8==0` with
> **no boundary guard** — same constraint the source comment states.

---

## The committed `.ref.ptx` byte-regression in the gate

The two kernels above are the standing **compile-proof** for GPU codegen, and the mechanism is a
**byte regression against a committed reference**. The universal gate,
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh), re-mints the PTX driver from the *edited*
`kovc.hx`, emits each kernel's PTX, and byte-compares the result to the committed `.ref.ptx`. It is a
**pure text** check: it needs **no GPU and no `ptxas`** — it only emits PTX and compares bytes.

The gate is emphatic that a missing reference is **not** a benign "no GPU" skip — it is a real failure,
because the text regression has no anchor:

**Fragment** (the fail-closed posture of the PTX text regression; from [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) lines 28–34):

```bash
# FAIL-CLOSED vs FAIL-OPEN distinction (v1.3 audit-remediation A1): the GPU PTX
# REGRESSION is PURE TEXT -- emit a kernel's PTX from the (re-minted) driver and
# byte-cmp it to the COMMITTED reference. It needs NO GPU and NO ptxas. The ONLY
# legitimate "skip" in this gate is running a .ptx ON A GPU (kernel EXECUTION),
# which this gate never does. Therefore a MISSING committed reference is NOT a
# benign GPU-absent skip -- it means the text regression cannot run at all, so it
# is a REAL gate FAILURE (GATE_OK=0), never a WARN.
```

For `vector_add`, the comparison is a direct `cmp`:

**Fragment** (the `vector_add` PTX byte-regression; from [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) lines 143–149):

```bash
  chmod +x /tmp/newdrv.bin; cp "$Kern" /tmp/kernel_in.hx; rm -f /tmp/out.ptx
  timeout 30 /tmp/newdrv.bin >/dev/null 2>&1 || true
  if [ ! -s /tmp/out.ptx ]; then
    # The emit is pure text generation (no GPU); no /tmp/out.ptx = the emitter ran but produced nothing -> REAL FAILURE.
    echo "  FAIL: re-minted driver emitted no /tmp/out.ptx (PTX text emit failed -- not a GPU-execution skip)"; GATE_OK=0
  elif cmp -s /tmp/out.ptx /tmp/ref.ptx; then echo "  GPU PTX REGRESSION OK (PTX byte-identical pre/post fix)";
  else echo "  GPU PTX CHANGED -- inspect (x86-only fix should NOT alter PTX)"; GATE_OK=0; fi
```

For the tiled GEMM, the gate adds a **provenance** grep on the emitted output — it is not enough that
the bytes match; the output must carry the shared-memory + double-buffer instruction classes, so a
future change cannot quietly downgrade the kernel to a non-tiled form that happens to match a stale
reference:

**Fragment** (tiled PTX byte-regression + provenance; from [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) lines 176–185):

```bash
  if [ -s /tmp/out.ptx ]; then
    cp "$TREF" /tmp/tref.ptx
    if cmp -s /tmp/out.ptx /tmp/tref.ptx; then echo "  TILED PTX REGRESSION OK (matches committed tiled_matmul_kernel.ref.ptx)";
    else echo "  TILED PTX CHANGED -- re-mint+re-commit the tiled reference with a reason"; GATE_OK=0; fi
    if grep -q '\.shared' /tmp/out.ptx && grep -q 'bar\.sync 0' /tmp/out.ptx \
       && grep -q 'cp\.async\.cg\.shared\.global' /tmp/out.ptx \
       && grep -q 'cp\.async\.commit_group' /tmp/out.ptx \
       && grep -q 'cp\.async\.wait_group' /tmp/out.ptx; then echo "  TILED PROVENANCE OK (.shared + bar.sync + cp.async double-buffer in the OUTPUT)";
    else echo "  TILED PROVENANCE FAIL (missing .shared/bar.sync/cp.async in emitted PTX)"; GATE_OK=0; fi
  else echo "  FAIL: tiled kernel emitted no /tmp/out.ptx (PTX text emit failed -- not a GPU-execution skip)"; GATE_OK=0; fi
```

Two further properties make this a strong regression:

1. **The driver is re-minted from the edited compiler.** The gate first rebuilds the PTX driver
   (`seed.bin k1ptxdrv.hx /tmp/newdrv.bin`) from the *current* `kovc.hx`, then emits with it. So the
   regression catches any change in the actual self-hosting compiler's codegen, not a cached artifact.
2. **It guards both directions of intent.** A pure CPU (x86) fix is expected to leave PTX *unchanged*
   ("x86-only fix should NOT alter PTX"); a deliberate GPU change is expected to require a re-mint and
   re-commit of the reference *with a recorded reason*. The reference moving silently is a gate
   failure either way.

This PTX text regression sits alongside the rest of the gate's checks: the **self-host fixpoint**
(`seed → K1 → K2 → K3 → K4`, asserting K2 == K3 == K4 byte-identical and equal to the pinned
`0992dddd…`) and the **feature corpus** (the gate compiles and runs its programs through the freshly
minted K2 and asserts `109` pass with `0` regressions; gate step `[4]`, verdict at
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) lines 728–729). The gate prints the literal
token `GATE_PASS` only if **every** leg holds — fixpoint identical **and** both PTX references
byte-identical (with tiled provenance) **and** the corpus clean.

> **For AI agents:** the GPU proof the gate provides is **PTX text byte-identity + instruction-class
> provenance**, not on-hardware execution. Match the exact success token `GATE_PASS` (a bare `grep -q
> '^GATE_PASS'`), and read a `GPU PTX REGRESSION OK` / `TILED PTX REGRESSION OK` line as proof the
> *emitter* is unchanged — never as proof a kernel *ran*. On-hardware execution skips live in
> `scripts/capstone_audit.sh`, not in this gate (gate lines 28–34).

---

## The honest boundary: complete to PTX, not to SASS

Everything in this chapter — the `@kernel` surface, the intrinsic→special-register lowering, the
per-access global addressing, the fused tiled GEMM, and the gate's byte regression — is **hand-auditable
Helix producing PTX text**. That is the precise scope of the GPU claim, and it must not be inflated.
Stated plainly, and citing [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md):

- **Complete to PTX, not to GPU machine code.** The from-raw chain ends at **PTX text**. Below PTX, the
  trusted computing base is NVIDIA's **closed `ptxas`** (the PTX→SASS assembler), the **CUDA driver**,
  the **GPU hardware**, and the **C host launcher** (`helixc/runtime/cuda_launch.c` /
  `train_transformer.c`, the C-FFI half that makes the `libcuda` driver-API calls Helix cannot). None of
  these is reproduced from raw binary; they are **trusted-once** (§R / boundaries 7–8). The **CPU** path,
  by contrast, *is* all-the-way-down from the hand-typed root — that asymmetry is the whole point of
  saying "PTX, not SASS."
- **PTX, not SASS — what that means concretely.** `kovc` emits the `.ptx` text you saw above. It does
  **not** emit SASS (the actual GPU machine instructions). The PTX→SASS step is `ptxas`'s job, and
  `ptxas` is closed-source NVIDIA code outside the reproducible chain.
- **A fraction of cuBLAS, not parity.** On the reference RTX 3070 Laptop (sm_86), the emitted GEMMs run
  at roughly **50–67.5% of cuBLAS** (G1 ≈56% cuBLAS-f32, G2 ≈67.5%, the TF32 path ≈50–54% of
  cuBLAS-TF32). Helix emits correct, reasonably performant kernels; it does **not** beat NVIDIA's
  hand-tuned library, on this GPU or any other (residual 1).
- **End-to-end speedup 7.0–8.7×, not ≥10×.** The capstone's end-to-end GPU speedup is Amdahl-bound at
  **7.0–8.7×**, not the originally estimated ≥10× (residual 2). The hard correctness gate — loss parity
  against an independent numpy oracle — holds at **~0%** difference; that is the load-bearing result,
  and it is a *correctness* claim, not a throughput one.
- **Single hardware target.** Only **sm_86** is tested. There is no cross-architecture (sm_80/sm_90) or
  multi-vendor (AMD) validation (residual 6). The hardcoded `.target sm_86` you saw in both references
  is the honest reflection of that scope.

The full closed-state record and every residual is in
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md); the first on-hardware demonstration
of the elementwise path (and the negative controls proving the GPU genuinely JIT-executes kovc's PTX
*text*, not a hardcoded answer) is recorded in
[`docs/HELIX_GPU_FIRSTLIGHT.md`](../../../docs/HELIX_GPU_FIRSTLIGHT.md). What this chapter establishes is
the reproducible link in that chain: **a raw-binary-bootstrapped `kovc` emits the PTX, and the gate pins
that PTX byte-for-byte.**

> **For AI agents:** the ceiling on any statement you make about Helix's GPU capability is §R of
> [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md). Permitted: "kovc emits PTX text
> directly, byte-pinned by the gate," "complete to PTX," "~50–67.5% of cuBLAS on sm_86," "7.0–8.7×
> end-to-end," "loss parity at ~0%." **Not** permitted: "verified GPU," "complete to SASS/machine
> code," "beats cuBLAS," "≥10×," or any cross-arch / multi-vendor claim.

---

**Next:** [GEMM, tiling & the capstone](02-gemm-tiling-capstone.md): how the tiled GEMM
above is tuned into the f32-SMEM `cp.async` double-buffered performance tier, the broader GPU op set
(elementwise, softmax, layernorm, attention) the capstone is built from, and the end-to-end transformer
that trains on kovc-emitted PTX kernels to **loss parity** with an independent oracle — with the
performance fractions and the PTX-not-SASS boundary carried through. The on-hardware first-light record
is in [`docs/HELIX_GPU_FIRSTLIGHT.md`](../../../docs/HELIX_GPU_FIRSTLIGHT.md) and the trust record in
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md).
