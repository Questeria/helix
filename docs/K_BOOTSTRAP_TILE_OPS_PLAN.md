# K-bootstrap tile-ops port plan

**Status:** discovery (K1.F23-discovery, 2026-05-27)
**Audience:** future K1.F23, K1.F23b, ... implementation chunks
**Scope:** Phase-0 CPU baseline. GPU/HBM/async memory deferred.

## 1. Why this matters

The 2026-05-26 hard-constraint directive (`docs/K_BOOTSTRAP_HARD_CONSTRAINT.md`)
requires ZERO non-Helix runtime code at v1.0 — Python `helixc/` must be
deleted, including `helixc/ir/tile_ir.py` (521 lines) and
`helixc/ir/tile_adjoint.py` (290 lines). "Defer GPU/MLIR/Tile ops to
Python forever" is explicitly disallowed.

Tile ops are one of three Category-2 OPEN blockers (alongside GPU
backends and MLIR migration). The macros cluster (K1.F22*) is approaching
saturation; tile ops is the next highest-leverage track.

## 2. Python helixc tile-op surface (the thing being ported)

`helixc/ir/tile_ir.py` defines `class TileOpKind(Enum)` with the
following members (line numbers in tile_ir.py):

### 2a. Tile creation (port priority HIGH)
- `TILE_ZEROS` (line 58) — allocate + memset(0) a tile of given dtype/shape
- `TILE_CONST` (line 59) — embed a compile-time-known constant tile in .data

### 2b. Memory movement (port priority LOW — CPU-only baseline ignores)
- `TILE_LOAD_GLOBAL` / `TILE_STORE_GLOBAL` (HBM <-> SMEM/REG)
- `TILE_LOAD_SHARED` / `TILE_STORE_SHARED` (SMEM <-> REG)

The CPU baseline treats all memory as flat — these ops become no-ops or
identity copies. Full memspace differentiation lands with the GPU
backends port.

### 2c. Async memory (port priority DEFERRED)
- `TMA_LOAD`, `TMA_STORE`, `BARRIER_WAIT` (Hopper/Blackwell TMA)

Only meaningful on real GPU hardware. The bootstrap port skips these
for the Phase-0 CPU baseline.

### 2d. Compute on tiles (port priority HIGH)
- `TILE_ADD`, `TILE_SUB`, `TILE_MUL` — elementwise binary ops
- `TILE_MATMUL` — matrix multiply (the headline perf op)
- `TILE_REDUCE` — axis-reduce within tile (port priority MEDIUM)

### 2e. Layout transforms (port priority MEDIUM)
- `TILE_TRANSPOSE` — swap row/col of a 2D tile
- `TILE_RESHAPE` — reinterpret shape (no data movement)

### 2f. Scalar ops (already in bootstrap)
- `SCALAR_CONST_INT/FLOAT`, `SCALAR_ADD/SUB/MUL/NEG`, `SCALAR_CMP`,
  `SCALAR_SELECT`, `CALL`, `RETURN`. The bootstrap already covers these
  via AST_INT, AST_ADD/SUB/MUL/NEG, AST_LT/EQ/NE/LE/GE/GT, AST_IF
  (select-shaped), AST_CALL, AST_RET.

### 2g. GPU primitives (port priority DEFERRED)
- `THREAD_IDX`, `TILE_INDEX_LOAD_HBM`, `TILE_INDEX_STORE_HBM`

Tied to the GPU-backends port.

## 3. Bootstrap port — Phase-0 minimum-viable subset (5 ops)

For the Phase-0 CPU baseline:

1. `TILE_ZEROS<T, N, M>` — allocate N*M*sizeof(T) bytes on stack, memset 0.
2. `TILE_ADD<T, N, M>` — N*M-element loop, elementwise add into result.
3. `TILE_SUB<T, N, M>` — same loop, sub.
4. `TILE_MUL<T, N, M>` — same loop, mul (elementwise, NOT matmul).
5. `TILE_MATMUL<T, M, K, N>` — A:[M,K] @ B:[K,N] -> C:[M,N], triple
   nested loop, accumulate.

The bootstrap already parses `Tile<T, N, M>` types via K1.T (generic
parse as no-op) — this is the K1.F-discovery batch 9 finding. Tile-aware
codegen is the new work.

## 4. AST representation

Three options considered:

### Option A: dedicated AST tags per tile op
- `AST_TILE_ZEROS` (new tag)
- `AST_TILE_ADD`, `AST_TILE_SUB`, `AST_TILE_MUL`
- `AST_TILE_MATMUL`

Pros: explicit, the codegen layer can pattern-match cleanly.
Cons: needs 5 new tag numbers (currently AST tags 0-33 used; tag 34+
would be the new range).

### Option B: reuse AST_CALL with a tile-name convention
- `Tile::zeros()` parses as AST_CALL(name="Tile::zeros", ...)
- Codegen's `try_emit_builtin_call` adds tile-op branches matching the
  name and shape.

Pros: no new AST tags; reuses the existing builtin-call machinery.
Cons: name-based dispatch is fragile; type info (dtype + shape) is
encoded in turbofish args which the bootstrap doesn't fully type-check
in let-position.

### Option C: hybrid — keep types as no-ops, dispatch at call-site
- Tile values are just `i32` arena offsets in the bootstrap's type
  system (the bootstrap has no real tile types).
- `Tile::<i32, 4, 4>::zeros()` parses to AST_CALL(name="__tile_zeros",
  args=[shape constants]).
- The codegen for `__tile_zeros` builtin allocates the storage and
  returns its arena offset.
- Tile ops then operate on these offsets (which act like pointers).

This is the most bootstrap-friendly approach: keep the bootstrap's
flat-i32 type model, treat tiles as opaque arena offsets, and codegen
each tile op as a builtin that operates on those offsets.

**Decision: Option C** for Phase-0. Revisit when types become first-class
in the bootstrap (post-K3).

## 5. Codegen approach (CPU baseline)

For Tile<i32, N, M> = N*M i32 slots:

### `__tile_zeros(N, M) -> offset`
```
allocate(N*M*4 bytes) on stack OR arena
memset(allocated, 0, N*M*4)
return offset
```

### `__tile_add(a_off, b_off, dst_off, N, M)`
```
for i in 0..N*M {
    dst[i] = a[i] + b[i]
}
return dst_off (or 0)
```

### `__tile_matmul(a_off, b_off, dst_off, M, K, N)`
```
for i in 0..M {
    for j in 0..N {
        let mut acc: i32 = 0;
        for k in 0..K {
            acc = acc + a[i*K + k] * b[k*N + j];
        }
        dst[i*N + j] = acc;
    }
}
```

Each builtin emits inline assembly (the bootstrap's standard codegen
pattern). For small fixed-size tiles (N, M, K known at parse time),
the loop can be unrolled or kept as a runtime loop.

## 6. Chunk sequence

### K1.F23 — DISCOVERY FINDING — multi-fn arena access is broken
1. `K1.F23` (2026-05-27, this commit) — landed two isolation probes:
   - Probe A: `__arena_push`/`__arena_set`/`__arena_get` round-trip
     in main() directly. **PASSES.**
   - Probe B: same round-trip with `__arena_push` moved into a helper
     fn. **FAILS with SIGILL (rc=132).**
   This blocks the user-fn-based tile_zeros approach (Option C in §4)
   for Phase-0 -- the helper fn pattern is unusable.
   Hypotheses (to investigate in a follow-up chunk): arena_base patch-
   table tied to main()'s code section; fn_table_lookup off for non-
   main fns; bind_state pollution between the helper compile and
   main's compile.
   Pin test: `test_bootstrap_kovc_k1f23_arena_helper_fn_known_broken_
   self_host` asserts the current broken rc=132; flip to rc=7 once
   fixed.

### K1.F23b — investigation update (defect localized; fix pending)
The K1.F23 finding ("multi-fn arena access traps SIGILL") is **narrower
than initially thought**. K1.F23b probes (6 sub-probes total) localize:

  | Probe | Source shape | Result |
  |-------|--------------|--------|
  | C | plain multi-fn, no builtins | rc=7 ✓ works |
  | D | helper with `let` | rc=7 ✓ works |
  | E | helper with int param | rc=7 ✓ works |
  | F | helper calls `print_int` | rc=7 ✓ works |
  | G | helper calls `__arena_len` (read-only) | rc=0 ✓ works |
  | H | helper calls `__arena_set` (indexed write) | rc=7 ✓ works |
  | B (K1.F23) | helper calls `__arena_push` (cursor-bump write) | rc=132 ✗ BROKEN |

**Only `__arena_push` fails from helper context.** All other builtins
including the arena WRITE (`__arena_set`) work correctly. The defect
is specifically in the cursor-bump portion of `__arena_push` codegen
(kovc.hx ~line 4193): the `mov ecx, [rax]` -> bounds-check -> `mov
[rax+rcx*4+4], edx` -> `inc ecx` -> `mov [rax], ecx` sequence misfires
in helper-fn context.

Hypotheses (still open; require instruction-level instrumentation):
  - The relative-jump targets (`jb +7`, `jmp +12`) compute correctly
    in main's code section but mis-target in helper's code section.
  - The cursor-bump section's interaction with the helper's prologue/
    epilogue clobbers a register.
  - Some interaction between __arena_push's 37-byte body and the K2
    binary's per-fn code-region layout.

Pinned via `test_bootstrap_kovc_k1f23b_multi_fn_defect_probes_self_host`
(asserts the full findings dict; failure means defect pattern drifted).

### K1.F23b-fix — fix the localized __arena_push helper defect (BLOCKER)
- Capture the emitted bytes for both the main-context and helper-context
  `__arena_push(0)` calls; binary-diff to spot the divergence.
- Likely fix scope: small. The codegen byte sequence is fixed; what
  changes is the surrounding context. Most likely a register-clobber
  or jmp-offset bug.
- Without this, NO higher-level tile-op user-fn pattern using
  __arena_push in helpers works.

### K1.F23c — TILE_ZEROS builtin (FALLBACK plan if K1.F23b is hard)
Alternative path: implement `__tile_zeros` as a BUILTIN with inline
codegen (cursor-bump trick: read arena cursor, add N*M, write back; the
arena's BSS-zero init means new cells are already 0). This bypasses the
multi-fn defect entirely but adds ~30-50 bytes of inline asm.

### K1.F23d — Tile syntax sugar (gated on F23b or F23c)
- Wire `Tile::<T, N, M>::zeros()` parse-time rewrite to
  `AST_CALL(__tile_zeros, [N, M])` or the user-fn equivalent.
- Test: full Rust-style construction works.

### K1.F24 — TILE_ADD elementwise (1-2 chunks)
1. `K1.F24` — `__tile_add(a, b, dst, N, M)` builtin. Loops elementwise.
2. `K1.F24b` — wire `tile_add(a, b)` or `a + b` (with tile-typed
   operands) to the builtin call.

### K1.F25 — TILE_SUB, TILE_MUL (likely batched)
- Mirrors K1.F24 with sub/mul instead of add.

### K1.F26 — TILE_MATMUL (largest chunk)
- The triple-nested loop. Most complex builtin codegen so far.

### K1.F27 — TILE_LOAD/STORE (CPU-baseline = identity copy)
- For CPU baseline, these are memcpy ops between identical regions.
- The real per-memspace lowering lands with the GPU backends.

### Deferred to GPU-backends track
- `TILE_REDUCE`, `TILE_TRANSPOSE`, `TILE_RESHAPE`
- `THREAD_IDX`, `TILE_INDEX_LOAD_HBM`, `TILE_INDEX_STORE_HBM`
- `TMA_LOAD`, `TMA_STORE`, `BARRIER_WAIT`
- All memspace-distinct LOAD/STORE variants

## 7. Test plan

Each chunk gets one self-host probe:

- TILE_ZEROS: `let t = __tile_zeros(2, 2); __arena_get(t)` -> 0.
- TILE_ADD: 2x2 tile a={1,2,3,4}, b={5,6,7,8}, dst=a+b; verify each cell.
- TILE_MATMUL: 2x2 @ 2x2 -> 2x2 with known result.

K2 parity probes deferred — Python helixc's compile-and-run frontend
likely rejects raw `__tile_zeros` calls (frontend type-checks
strictly). Inverse-parity if so; document as such.

## 8. Risks

### R1. Stack overflow for large tiles
Phase-0 stores tiles on the stack. A 1024x1024 i32 tile is 4MB — well
past the default Linux 8MB stack limit when combined with other
locals. Mitigation: arena allocation instead of stack. The bootstrap's
arena is sized at 8MB+ already.

### R2. ABI for passing tiles to functions
Tiles-as-offsets pass cleanly as i32. No new ABI needed.

### R3. Type inference for builtin args
`__tile_zeros(4, 4)` — the bootstrap's existing AST_INT codegen handles
i32 constants. No new infrastructure needed.

### R4. Codegen size growth
Each tile op adds 50-200 bytes of inline x86_64 code. The bootstrap's
str_table cap was bumped to 64 entries in K3.O — should be sufficient
for now. Re-evaluate when tile-op chunks accumulate.

## 9. Deferred questions

1. Should tiles be heap-allocated (arena) or stack-allocated? Arena
   chosen above; revisit if performance becomes a concern.
2. How does the type system disambiguate `Tile<i32, 4, 4>` from
   `Tile<i64, 2, 8>` at codegen time? Phase-0 hardcodes i32 (the
   only-supported tile dtype); generic dtype is a future K1.F23z chunk
   that requires propagating the turbofish type through the AST.
3. Adjoints (tile_adjoint.py, 290 lines): defer to a post-Phase-0
   chunk. The bootstrap's panic_pass / unwind_pass / trace_pass pattern
   suggests an `adjoint_pass` is feasible but heavy work.
4. MLIR-translation path (the v3.0 Stage 212): defer entirely. The
   bootstrap's tile-op runtime is a separate code path from the MLIR
   migration on the Python side.

## 10. Counter accounting

K1.F23-discovery is one chunk (this document). K1.F23 through K1.F27
covers the 5-op minimum-viable subset — likely ~8 chunks total. Beyond
Phase-0, the GPU/Reduce/Transpose/Reshape work adds another ~20 chunks
spread across the GPU-backends track.

Realistic total tile-ops cost to Python-ready-to-delete state:
~25-30 chunks. Same order of magnitude as the GPU-backends and
MLIR-migration tracks individually.
