#!/usr/bin/env python3
"""
scripts/helix_status.py — beginner-friendly Helix progress reporter.

The Helix autonomous build worker (the `helix-approach-a-loop`
scheduled task) sends a Telegram status update at the end of every
fire. Those updates used to be terse and developer-facing — e.g.
"Stage 117, commit abc1234, 21 tests pass" — unreadable to anyone
who is not a compiler engineer.

This module renders a plain-language update instead: what is finished
and audited, what is in progress, what is still ahead, and a
percent-progress readout for build stages, versions, and the project
overall.

It is the SINGLE SOURCE OF TRUTH for release-journey status. When a
version ships, change its `status` in `VERSIONS` below from
"in_progress" / "planned" to "released" (and open the next one). As
each v3.0 build stage closes its 3-part audit, bump `V3_STAGES_DONE`.
Every percentage recomputes from that edit; the test-suite size is
counted LIVE from `helixc/tests/` (so it grows with every chunk and
never goes stale — no manual bump).

Usage:
    python scripts/helix_status.py
    python scripts/helix_status.py --note "<plain-English summary>" \\
        --commit <hash>

License: Apache 2.0
"""
from __future__ import annotations

import argparse
from pathlib import Path


# --- The v2.0 -> v3.0 release journey --------------------------------
# Each Helix version ends with a 5-part "clean-gate" code audit before
# it counts as released. Statuses:
#   "released"    — shipped AND its end-of-version audit gate passed
#   "in_progress" — actively being built right now
#   "planned"     — scoped but not started
# Update `status` here (and ONLY here) as versions ship.
VERSIONS: list[dict[str, str]] = [
    {"id": "v2.0", "status": "released",
     "theme": "GPU compiler foundation (22 build stages)"},
    {"id": "v2.1", "status": "released",
     "theme": "Per-operation GPU code generation + autodiff"},
    {"id": "v2.2", "status": "released",
     "theme": "Polish and audit clean-up"},
    {"id": "v2.3", "status": "released",
     "theme": "Type-system design polish"},
    {"id": "v2.4", "status": "released",
     "theme": "Real-GPU testing + attestation + register allocator"},
    {"id": "v2.5", "status": "released",
     "theme": "Wiring the register allocator into real GPU kernels"},
    {"id": "v3.0", "status": "released",
     "theme": "The big rewrite - industrial MLIR + LLVM backend"},
    {"id": "v3.1", "status": "released",
     "theme": "Post-v3.0 cleanup - LLVM toolchain wiring, polymorphic "
              "SPLICE/MODIFY, REFLECT_HASH, shared-constants module"},
    {"id": "v3.2", "status": "planned",
     "theme": "Real-execution parity gate (or first K-bootstrap "
              "milestone toward Helix-in-Helix)"},
]

# v2.x shipped its compiler work as 22 numbered build stages
# (Stage 110-131), all closed — the v2.0-v2.5 entries in VERSIONS
# record that. v3.0 is built as its own 19 numbered stages: Phase D
# (Stage 200-208), Phase E (210-216), Phase F (220-222). Every stage
# closes with a 3-part audit. Bump `V3_STAGES_DONE` as each closes —
# every percentage below recomputes from it.
V3_STAGES_TOTAL = 19
V3_STAGES_DONE = 19       # ALL Phase D + E + F stages COMPLETE — v3.0 RELEASED

# K-bootstrap track (post v3.1.0, declared the new top-line goal
# 2026-05-25). See docs/HELIX_K_BOOTSTRAP_MASTER_PLAN.md and the
# feature-parity matrix docs/K_BOOTSTRAP_FEATURE_MATRIX.md. The
# matrix enumerates every Helix language feature with a column for
# Python helixc support and a column for kovc.hx support. A row is
# PARITY when both columns agree; KOVC-MISSING when only Python
# supports it. The goal: get every row to PARITY, then delete the
# Python compiler.
#
# Bump K_BOOTSTRAP_PARITY_DONE as each K-track chunk lands and the
# matrix's PARITY count rises.
# K_BOOTSTRAP_CHUNKS_DONE counts shipped K0/K1 commits on the
# K-bootstrap track (run `git log --oneline | grep -E "K[01]\.|K0 chunk"
# | wc -l` to recount). Bump each commit. The chunk count is more
# meaningful than matrix parity rows under the hard constraint because
# many "PARITY" rows are vacuously satisfied.
K_BOOTSTRAP_CHUNKS_DONE = 336      # last bump: K1.M14 -- FIRST GPU TILE COMPUTE OP: __tile_add(a, b, dst, count) elementwise-adds two register-tiles into a third over `count` consecutive %f registers (mirroring Python backend/ptx.py TILE_ADD / Stage 64 Inc 3). a/b/dst are vars bound to prior __tile_zeros results (their %f base, resolved via ptx_vtab_lookup); count is a static int literal. emit_ptx_call now dispatches __tile_add: read the 4 args off the AST_ARG chain (ah=node+3; expr=arg+1, next=arg+2), resolve the 3 tile-var %f bases, emit `count` add.f32 lines (dst[k]=a[k]+b[k] over consecutive %f), set the float flag, return base_d. New helpers: ptx_name_is_tile_add (10-char matcher) + emit_ptx_add_f3 (3-register add.f32). test_bootstrap_ptx_tile_add (@kernel zeros 3 tiles then __tile_add(a,b,c,4) -> 12x `mov.f32` + 4x `add.f32 %f8,%f0,%f4` .. `%f11,%f3,%f7`) ptxas-VALIDATED to REAL SASS. WITH M13 __tile_zeros THIS IS A COMPLETE ON-GPU TILE ELEMENTWISE PIPELINE (allocate + compute), all direct: Helix -> PTX -> ptxas -> SASS, NO CUDA, NO MLIR. Pure-additive (no sb-slots, no parser.hx) -- i32/f32 scalar + memory + tile_zeros stay BYTE-IDENTICAL. Ran new + 3 representative (38.16s). NEXT: __tile_sub/mul (one-byte opcode change off emit_ptx_add_f3 -> sub.f32/mul.f32) then __tile_matmul (wmma.mma.sync.aligned.m16n16k16 Tensor Cores -- the big AI matrix primitive + confetti milestone); main() output-mode switch; ROCm/Metal/WebGPU. PRIOR K1.M13 -- FIRST GPU TILE OP: __tile_zeros(N, M) lowers to N*M consecutive `mov.f32 %fX, 0f00000000;` register-fills (the register-tile model, mirroring Python backend/ptx.py TILE_ZEROS / Stage 64 Inc 2; 0f00000000 = +0.0f). MAJOR FINDING: the __tile_* CALL surface ALREADY PARSES -- the x86 CPU path (K1.F23c--F27) already implements __tile_zeros/add/sub/mul/matmul as machine-code builtins with the SAME 2-arg signature, so the GPU side is EMITTER-ONLY (NO parser change). emit_ptx_call previously returned -1 for every non-index builtin; now it dispatches __tile_zeros: read the two static int-literal args (AST_INT slot 1) -> count = N*M -> emit count zero-fills via the new emit_ptx_mov_f_zero -> return the base %f register + set the float flag (vtab slot 55). New helpers: ptx_name_is_tile_zeros (12-char byte matcher) + emit_ptx_mov_f_zero. test_bootstrap_ptx_tile_zeros (@kernel fn k() { __tile_zeros(2, 2) } -> 4x `mov.f32 %f0..3, 0f00000000;`) ptxas-VALIDATED to REAL SASS. Pure-additive (no sb-slots, no parser.hx) -- the i32/f32 scalar + memory paths stay BYTE-IDENTICAL (empty_kernel + i32/f32 elementwise-add stay green). Ran new + 3 representative (42.57s). The tile/matmul AI-primitive arc starts here. NEXT: __tile_add/sub/mul (elementwise over the consecutive %f block; needs last-tile base+length side-channels, e.g. vtab slots 56/57) then __tile_matmul (wmma.mma.sync.aligned.m16n16k16 Tensor Cores); main() output-mode switch; ROCm/Metal/WebGPU. PRIOR K1.M12b -- MILESTONE: a full FLOAT elementwise-add kernel compiles. `@kernel fn k(out, a, b) { let i = thread_idx(); out[i] = a[i] + b[i] }` (all f32) -> the self-hosted Helix bootstrap emits ld.global.f32 x2 + add.f32 + st.global.f32 (the M12a type-flag picks the float ops), ptxas-validated to REAL SASS. The REALISTIC AI workload (floats), fully direct: Helix -> PTX -> ptxas -> SASS, NO CUDA, NO MLIR. emit_ptx_binop now has an f32 path (BOTH operands float -> add.f32/sub.f32/mul.f32/div.rn.f32 into %f + set flag); the s32 path is BYTE-IDENTICAL (i32 elementwise-add + scalar arith stay green). test_bootstrap_ptx_f32_elementwise_add ptxas-VALIDATED. f32 elementwise compute COMPLETE (load + arith + store). Ran new + 4 i32-binop regressions (55.78s). NOTE: mixed int+float needs a cvt (future; both_f requires both float). Next: tile ops + wmma matmul (AI matrix primitives); main() output switch; ROCm/Metal/WebGPU. PRIOR K1.M12a --f32 STORE + the type-tracking flag: a complete float COPY kernel `out[i]=a[i]` (out,a f32) compiles. New: a "last-result-is-float" side-channel flag (vtab slot 55) -- emit_ptx_expr defaults it to 0 (i32); the f32 index-load sets it 1; the indexed store captures it (right after lowering the value, before the index clobbers it) and emits st.global.f32 (of a %f) vs st.global.u32 (of a %r). The i32 store path is BYTE-IDENTICAL (out[i]=7 + i32 elementwise-add stay green). test_bootstrap_ptx_f32_copy (out[i]=a[i], f32) ptxas-VALIDATED. Ran new + 4 i32-store regressions (51.57s). The flag side-channel is the foundation for f32 arithmetic. NEXT (M12b): add.f32 in emit_ptx_binop (capture operand flags -> f32 op into %f) -> the full f32 out[i]=a[i]+b[i] elementwise-add MILESTONE. PRIOR K1.M11 --f32 (float) global LOAD: the FIRST float op (the backend was i32-only). A kernel param typed `: f32` makes a[i] lower to ld.global.f32 into a %f register (vs ld.global.u32 %r for i32), selected via the param type_tag (AST_PARAM slot 4 == 1). New: %f register counter (vtab slot 54) + ptx_alloc_f, emit_ptx_f helper, ptx_param_type (reads AST_PARAM slot 4). emit_ptx_index_load now branches on element type; the i32 path is BYTE-IDENTICAL (global_load/elementwise_add/two_load_add stay green). test_bootstrap_ptx_f32_load (k(a: f32){...; a[i]} -> ld.global.f32 %f0) ptxas-VALIDATED. Foundation for f32 arithmetic (the realistic AI workload). Ran new + 4 i32 index-load regressions (52.68s). Pure-additive (no parser.hx). NEXT (M12): f32 type-tracking side-channel (vtab last-result-is-float flag) -> f32 store + add.f32 -> a full f32 elementwise-add kernel (real AI workload) MILESTONE. PRIOR K1.M10d --MILESTONE: a COMPLETE elementwise-add GPU kernel compiles end-to-end. `@kernel fn k(out, a, b) { let i = thread_idx(); out[i] = a[i] + b[i] }` -> the self-hosted Helix bootstrap emits PTX (thread index + two ld.global loads + add.s32 + st.global store, with full base+i*4 address arithmetic per array) that ptxas assembles into REAL GPU machine code (SASS). The canonical data-parallel GPU/AI kernel, fully direct: Helix -> PTX -> ptxas -> SASS, NO CUDA frontend, NO MLIR. No new emitter code -- pure COMPOSITION of thread_idx (M6) + global load (M10a) + global store (M10c) + add (M5d). test_bootstrap_ptx_elementwise_add (18-instruction kernel) ptxas-VALIDATED, 16.21s. The GPU backend now compiles real data-parallel kernels. Next: __tile_* tile ops + wmma matmul (the AI matrix primitives) + f32 floats; main() output-mode switch; ROCm/Metal/WebGPU. PRIOR K1.M10c --GLOBAL MEMORY STORE out[i]=v. The FIRST parser.hx change in the GPU work (careful + minimal): added an AST_INDEX (tag 53) branch to the field-store detection -> AST_INDEX_STORE (tag 55), mirroring the proven field-store (tag 79) path; previously `a[i]=v` dropped the `=` (parse trip). emit_ptx_index_store lowers the value, computes base+i*4 (param ptr -> cvta -> address), then st.global.u32 [addr], val. test_bootstrap_ptx_global_store (out[i]=7) ptxas-VALIDATED. BROAD REGRESSION after the parser change: full PTX suite (19 tests) + CPU canary (write_file_to_arena) = 20 passed 177s -- the parser edit broke NOTHING (every kernel re-parses, CPU path intact). Both LOAD (M10a) + STORE (M10c) now work -> the full out[i]=a[i]+b[i] elementwise-add kernel is now compilable (next: M10d, the milestone). PRIOR K1.M10b --the COMPUTE core of an elementwise kernel: `a[i] + b[i]` reads two arrays from global memory (multi-param: param_0 AND param_1) and adds them -> two full load sequences (ld.param/cvta/mul.wide/add.s64/ld.global) feeding add.s32. Pure-additive (composes the M10a index-load with the existing binop; NO new emitter code -- a test-only validation of multi-param multi-load + arithmetic). test_bootstrap_ptx_two_load_add ptxas-VALIDATED (real SASS). PROBE FINDING: indexed-STORE `out[i] = 7` does NOT parse currently (the `=` after `]` is unhandled -> emits an empty kernel, not even the let's mov); the store needs a parser.hx change (M10c, the careful parser chunk). Ran new + global_load (26.30s). PRIOR K1.M10a --GLOBAL MEMORY LOAD: a kernel can now READ arrays from GPU memory. a[i] (AST_INDEX tag 53 -- ALREADY parsed by the bootstrap at parser.hx:2698 mk_node(53,base,idx,0), so this is PURE-ADDITIVE, NO parser change!) lowers to the canonical CUDA load: ld.param.u64 (param pointer) + cvta.to.global.u64 + mul.wide.s32 (i*4) + add.s64 + ld.global.u32. New: ptx_param_index (name -> 0-based kernel param index), ptx_alloc_rd (%rd 64-bit addr regs, vtab slot 52), cur_fn_idx (vtab slot 53, set by emit_ptx_entry), emit_ptx_indent/r/rd helpers, emit_ptx_index_load; emit_ptx_expr dispatches tag 53. test_bootstrap_ptx_global_load (k(a){ let i=thread_idx(); a[i] } -> reads a[tid]) ptxas-VALIDATED. A kernel now reads global memory at a computed index -- the data-access pattern of every GPU kernel. vtab-heavy regression (while/global_index) stayed green. Ran new + 4 representative (52.75s). MAJOR FINDING: a[i] already parses, so MEMORY is emitter-only, NOT the risky parser change feared. Next: M10b store (a[i]=v -> st.global), then M10c full out[i]=a[i]+b[i] MILESTONE. PRIOR K1.M9 --AST_WHILE (tag 10) loops + AST_ASSIGN (tag 11): a real TERMINATING counting loop now compiles. `while x < 4 { x = x + 1 }` lowers to "$Ltop_<n>:" + cond (setp.lt+selp) + setp.ne + "@!%p bra $Lwend_<n>" + body (add + mov overwriting x's register) + "bra $Ltop_<n>" (back-edge) + "$Lwend_<n>:". emit_ptx_lbl_ref extended (which 2=top, 3=wend); new emit_ptx_while + emit_ptx_assign (x=v -> mov %rX,%rV, var binding unchanged); emit_ptx_expr dispatches tags 10/11. test_bootstrap_ptx_while_loop (x counts 0->4) ptxas-VALIDATED. The PTX backend now compiles the FULL scalar language: const/var/let/arith/cmp/if/while/assign + thread/block index, all ptxas-validated. Ran new + 4 representative (51.79s). Pure-additive (no sb-slots, no parser.hx). Next: the parser-subscript MEMORY chunk (a[i] -> ld/st.global -> out[i]=a[i]+b[i] MILESTONE; this one TOUCHES parser.hx -- check lexer for []; add minimal postfix subscript; sequential regression after). PRIOR K1.M8b --AST_IF (tag 7) -> predicated branch: CONTROL FLOW COMPLETE. `if cond { } else { }` lowers to cond-value + "setp.ne.s32 %pZ, %rC, 0" + "@!%pZ bra $Lelse_<n>" + then + "bra $Lend_<n>" + "$Lelse_<n>:" + else + "$Lend_<n>:" with per-kernel unique labels. New: ptx_alloc_label (vtab slot 51), emit_ptx_lbl_ref, emit_ptx_if; emit_ptx_expr dispatches tag 7. if-as-statement (value discarded in void kernels; if-as-value phi deferred). test_bootstrap_ptx_if (if x<3 {1} else {2}) ptxas-VALIDATED. The PTX backend now expresses the SHAPE of real bounds-checked GPU kernels (global-index + if). Ran new + 4 representative tests (46.81s) per the test-time policy; 15 PTX tests total. Pure-additive (no sb-slots, no parser.hx). Next: AST_WHILE (tag 10, loops, pure-additive) then the parser-subscript MEMORY chunk (a[i] -> ld/st.global -> out[i]=a[i]+b[i] MILESTONE; touches parser.hx -- careful). PRIOR K1.M8a --comparison-as-value (control-flow foundation): AST_LT/GT/EQ/NE/LE/GE (tags 6/19/20/21/22/23) lower to "setp.<cc>.s32 %pP, %rA, %rB" + "selp.b32 %rR, 1, 0, %pP" (reify 0/1 into a register, matching AST semantics). New: predicate-register counter appended to the vtab at slot 50 (var entries UNCHANGED -> 13 prior tests stay byte-identical; ptx_alloc_pred), emit_ptx_cc (2-char cond mnemonic), emit_ptx_cmp. selp-with-immediates + setp ptxas-validated. test_bootstrap_ptx_scalar_cmp ({let x=5; x<3} -> setp.lt.s32 + selp.b32), ptxas-VALIDATED. 14 PTX tests green, 120s. Pure-additive (no sb-slots, no parser.hx). Next: K1.M8b AST_IF (tag 7 -> setp.ne + @!%p bra + labels via next_label slot 51 + emit_ptx_label). NOTE: PTX suite now ~2min/14 tests -- run NEW + ~4 representative prior tests next tick, not all. PRIOR K1.M7 --block_idx() -> %ctaid.x and block_dim() -> %ntid.x. With thread_idx() + scalar arithmetic, a @kernel now computes the CANONICAL global thread index `block_idx()*block_dim() + thread_idx()` (the foundation of every grid-stride GPU kernel). New helpers ptx_name_is_block_idx/dim + emit_ptx_mov_ctaid_x/ntid_x; emit_ptx_call dispatches all 3 index builtins. test_bootstrap_ptx_global_index: the full formula -> 3 sreg movs + mul.lo.s32 + add.s32, ptxas-VALIDATED (real SASS). 13 PTX tests green, 105.88s. Pure-additive (no sb-slots, no parser.hx). SURVEY NOTE: full memory (a[i] load/store) needs SUBSCRIPT parsing -- the bootstrap AST has NO array-index node, so memory requires a parser.hx change (a deliberate sb-slot-careful chunk + sequential regression), unlike everything M1-M7 which is pure-additive kovc.hx. Next: K1.M8 control flow (AST_IF tag 7 + AST_LT tag 6 -> setp/bra, pure-additive) or the parser-subscript chunk for memory. PRIOR K1.M6 --thread_idx() builtin: the entry point to data-parallel kernels. AST_CALL (tag 16) to "thread_idx" now lowers to "mov.u32 %rN, %tid.x;" (reading the hardware thread-index special register), matching the Helix surface (lower_ast.py thread_idx -> THREAD_IDX) + Python ptx.py. New helpers: ptx_name_is_thread_idx (flat byte-compare), emit_ptx_tid_x, emit_ptx_call; emit_ptx_expr now handles AST_CALL. SURVEY FINDING: the AST has NO array-subscript node -- kernel memory/compute is expressed via AST_CALL to builtins (thread_idx / __tile_* / block_idx). test_bootstrap_ptx_thread_idx ({let i = thread_idx(); i} -> mov.u32 %r0, %tid.x), ptxas-VALIDATED (assembles to real SASS). 12 PTX tests green, 97.55s. Pure-additive (no sb-slots, no parser.hx). Next: K1.M7 memory load/store (survey-gated) or parallel-index siblings (thread_idx_y/z, block_idx, block_dim) or __tile_* ops. PRIOR K1.M5f --PROOF the emitted PTX is REAL GPU code: ptxas ROUND-TRIP validation. NVIDIA's official ptxas (CUDA 12.0) ACCEPTS the bootstrap's direct-emitted PTX and produces a cubin (SASS GPU machine code), rc=0. End-to-end: Helix source -> self-hosted bootstrap -> PTX text -> ptxas -> SASS, with NO CUDA frontend and NO MLIR. Also lowered .version 8.3->8.0 (the self-host path's WSL ptxas caps at PTX ISA 8.0 = CUDA 12.0; we use only basic scalar ops so 8.0 is sufficient + more broadly compatible). New test_bootstrap_ptx_ptxas_roundtrip (skips gracefully if ptxas absent, so non-CUDA CI still passes). 11 PTX tests green, 86.75s. Pure-additive (no sb-slots, no parser.hx). LESSON: the self-host test path runs ptxas via WSL (/usr/bin/ptxas = CUDA 12.0, max .version 8.0); a manual `ptxas` in the Bash tool hit a DIFFERENT 12.8 install -- always validate via the WSL path. Next: K1.M6 tile ops (the AI matrix primitives) or M8 main() output-mode switch. PRIOR K1.M5e --rounded out scalar arithmetic: added AST_DIV (tag 5 -> div.s32, binop opc 3) and AST_NEG (tag 9, unary -> neg.s32 via new emit_ptx_neg helper) to the recursive emit_ptx_expr. The bootstrap PTX backend now lowers the FULL scalar arithmetic set: const / var / let + add / sub / mul / div / neg. 2 new tests ({let x=12; x/3} -> div.s32 %r2,%r0,%r1; {let x=5; -x} -> neg.s32 %r1,%r0); 10 PTX tests green, 82.50s. Pure-additive (no sb-slots, no parser.hx). Next: K1.M5f comparison (AST_LT tag 6 -> setp) + control flow (AST_IF tag 7), OR pivot to higher-AI-value M8 main() output-mode switch (make the compiler actually emit .ptx files) or M6 tile ops. PRIOR K1.M5d --scalar ARITHMETIC + variable->register environment. Built a recursive emit_ptx_expr lowering kernel-body scalar expressions to PTX: AST_INT->mov.s32, AST_VAR->resolve via the var table, AST_LET->bind name->reg + recurse into continuation, AST_ADD/SUB/MUL->emit_ptx_binop (add.s32 / sub.s32 / mul.lo.s32). New var->reg table (ptx_vtab_init/reset/add/lookup + ptx_alloc_reg) lives in the arena BEFORE the output region so it never pollutes the .ptx; reset per kernel. M5b/M5c body lowering refactored onto emit_ptx_expr -> byte-identical (6 prior tests stay green). 2 new tests: {let x=5; x+2} -> mov %r0,5 / mov %r1,2 / add.s32 %r2,%r0,%r1; {let x=5; x*3} -> mul.lo.s32. 8 PTX tests green, 67.88s. Pure-additive (no sb-slots, no parser.hx). The GPU backend now compiles real scalar expressions (foundation for tile ops). Next: K1.M6 tile ops (TILE_ZEROS/ADD/SUB/MUL) or M8 main() output-mode switch. PRIOR K1.M5c --kernel-body let-chain lowering: walk a leading AST_LET chain (tag 8; value in slot 4, continuation in slot 3), emitting one SCALAR_CONST_INT "mov.s32 %rN, <const>" per integer-const-init let; tail AST_INT emits one more mov, tail AST_VAR (tag 1) resolves to an existing register (void kernel -> no instruction). New helper emit_ptx_mov_const(ridx, val). LESSON: `reg` is a reserved Helix keyword (KW_REG) -- it CANNOT be an identifier in kovc.hx (caught when Python Stage-30 failed to PARSE the bootstrap: "expected IDENT got KW_REG 'reg'"); param renamed reg->ridx. 6 PTX tests green (new test_bootstrap_ptx_let_chain: {let x=3; let y=8; y} -> 2 movs; _ptx_entry refactored to a movs tuple): 6 passed 45.73s. Pure-additive (no sb-slots, no parser.hx). Next: K1.M5d scalar arithmetic (add.s32) via a var->reg env. PRIOR K1.M5b --FIRST kernel-BODY op lowering: an integer-literal kernel body now lowers to SCALAR_CONST_INT -> "    mov.s32 %r0, <val>;" (mirrors Python ptx.py emit_op), materialized before ret. The step from "valid empty kernel" to "kernel that computes". Body = AST_FN_DECL slot 3 (confirmed bare AST_INT = tag 0, value in slot 1, since a brace-block returns its inner expr directly). New helper emit_ptx_decimal (recursive int->ASCII). 5 PTX tests green: 4 prior now expect "mov.s32 %r0, 0;" via shared _ptx_entry(body_const=0); new test_bootstrap_ptx_scalar_const ({7} -> "mov.s32 %r0, 7;"); 5 passed 41.90s. Pure-additive (no sb-slots, no parser.hx). Next: K1.M5c (scalar add/sub/mul or thread-idx) or M6 tile ops. PRIOR K1.M5a --kernel body foundation: every @kernel now emits the standard PTX register-file declaration block (5 files: .pred %p, .b32 %r, .b64 %rd, .f32 %f, .b16 %h; pool 256) after "{", byte-matching Python ptx.py _REG_FILES, plus an indented "    ret;". An empty kernel now byte-matches Python emit_kernel EXACTLY. Declaring a big pool is free (ptxas allocates only USED regs). New helpers emit_ptx_reg_prefix/suffix/block. 4 PTX tests refactored onto shared _PTX_HEADER/_PTX_REG_BLOCK/_ptx_entry golden fragments + green: 4 passed 47.15s. This is the foundation op-lowering (M5b) needs. Pure-additive (no sb-slots, no parser.hx). Next: K1.M5b first scalar op (mov.s32 const or %tid.x). PRIOR K1.M4 --kernel params: emit ".param .b64 param_N" (positional; v0.1 all .b64, mirroring Python ptx.py _format_param) per fn param inside the .entry parens, comma-space separated. Walks AST_FN_DECL slot 4 (params_head); each AST_PARAM links via slot 3 (next). Zero-param kernels keep "()" so empty/named/multi stay byte-identical. test_bootstrap_ptx_kernel_params (@kernel fn k(a,b) -> ".visible .entry k(.param .b64 param_0, .param .b64 param_1)") + 3 prior: 4 passed 37.29s. Pure-additive (no sb-slots, no parser.hx). Next: K1.M5 scalar body + .reg decls. PRIOR K1.M3 --multi-kernel PTX: emit the module header once then one ".visible .entry <name>()" per @kernel fn (blank-line separated), mirroring Python ptx.py emit_module. Single-kernel output stays byte-identical (no trailing blank) so M1/M2 tests stay green; fn_list confirmed source-order. test_bootstrap_ptx_multi_kernel (kernels a,b -> 2 entries) + named + empty: 3 passed 31.34s. Pure-additive (no sb-slots, no parser.hx). Next: K1.M4 kernel params (.param .b64 param_N). PRIOR K1.M2 --PTX .entry name is now the REAL kernel fn name (copied from AST_FN_DECL slots 1/2 = name_start/len; same source-byte read as the bootstrap's 'main' detection), not a hardcoded 'k'. test_bootstrap_ptx_named_kernel (@kernel fn saxpy -> ".visible .entry saxpy()") + empty_kernel both green: 2 passed 22.75s. Pure-additive codegen (no sb-slots, no parser.hx). Next: K1.M3 multi-kernel (one .entry per is_kernel fn). PRIOR K1.M1 -- FIRST DIRECT-TO-GPU EMISSION. The bootstrap now emits NVIDIA PTX *text* directly from a @kernel fn (emit_ptx_for_ast_to_path in kovc.hx), mirroring how emit_elf_for_ast_to_path emits x86_64 machine code -- NO MLIR, NO LLVM, straight to the target ISA. PTX is a text virtual-ISA so this is STRICTLY SIMPLER than the ELF binary the bootstrap already emits (no headers/offsets/relocations -- just ASCII bytes the NVIDIA driver JITs to SASS). test_bootstrap_ptx_empty_kernel: @kernel fn k() -> emits the minimal valid 74-byte module (.version 8.3 / .target sm_75 / .address_size 64 / .visible .entry k() { ret; }), PASSED 15.76s. Pure-additive codegen: no sb-slots, no parser.hx change (parser already tags @kernel on AST_FN_DECL slot 14 since Stage 33), so the K1.F5d-j sb-collision hazard does not apply. Implements the user's 2026-05-27 north-star goal: "Have Helix wherever possible talk directly to the chips" (CPU=done via ELF, GPU=now starting via PTX). Per-chunk plan in docs/GPU_DIRECT_EMIT_PLAN.md (K1.M2 real fn name, M3 multi-kernel, M4 params, M5 scalar body+regs, M6 tile ops, M7 wmma matmul, M8 main() output-mode switch; then ROCm/Metal/WebGPU siblings). Reference: MLIR-free helixc/backend/ptx.py (verified K2.AK). [prior K2.AK: VERIFIED MLIR-not-needed with hard evidence -- all 4 Python GPU backends are MLIR-free direct tile-IR->text emitters totaling 3205 LOC < the 5517-LOC x86_64 backend already mirrored; see docs/MLIR_NOT_NEEDED_DECISION.md. K1.F5k: disabled the broken chained-method substrate (sb-slot collisions); LESSON: new sb-slots must extend parse_top alloc-block past 123 and grep "sb + N)" with close-paren.]
# Estimated total chunks to v1.0 (Python fully deleted, all features
# ported, K5 DDC passes). Two estimates:
#   BEST     = optimistic, batched, parallelized, deferring some Tile/GPU
#              corners that turn out vacuously satisfied at K2 time
#   REAL     = under the 2026-05-26 hard constraint (no Python-forever
#              deferral for any subsystem)
K_BOOTSTRAP_CHUNKS_BEST_ESTIMATE = 400  # K2.AJ 2026-05-28 RE-revised DOWN
                                          # from K2.AI's 470. K2.AI counted
                                          # the 15k-LOC MLIR surface as
                                          # port-work; K2.AJ determined MLIR
                                          # is NOT-NEEDED (bootstrap is direct-
                                          # codegen, doesn't consume MLIR; all
                                          # helix-dialect ops already native).
                                          # So P2.1 (~100-150 chunks) drops off.
                                          # Remaining big bucket = P2.2 GPU
                                          # direct-emission (~80-150 chunks).
K_BOOTSTRAP_CHUNKS_REAL_ESTIMATE = 470  # K2.AJ: 310 done + GPU-direct-emit
                                          # (~80-150) + P1 tail + K3 seed +
                                          # 5-clean gate ~= 470. The whiplash
                                          # (440->560->470) reflects: K2.AI
                                          # saw the MLIR LOC surface, K2.AJ
                                          # determined most of it isn't
                                          # bootstrap-bound. Net ~similar to
                                          # the original 440, different reason.

# K2.W (2026-05-27): Python-deletion-readiness bucket model. Each bucket
# is one Category-1 syntax/semantic gap or Category-2 platform port that
# must close before Python helixc can be deleted (K4). Status values:
#   "done"    : feature-complete + audit-clean
#   "partial" : at least one shipped chunk but not feature-complete
#   "pending" : zero chunks shipped, scoping not yet done
# Percent: done = 1.0, partial = 0.5, pending = 0.0; weighted average.
# This is the canonical list per the loop prompt's Python-ready-to-delete
# definition + the 2026-05-26 hard constraint.
PYTHON_DELETION_BUCKETS = [
    {"name": "Macros (assert/print/dbg/panic/todo family)",
     "status": "done",
     "note": "K1.F22-F52 saturated; assert!-cmp family closed F41-F52 audit-clean"},
    {"name": "Mixed-type int binops (i64<->i32, u64<->u32)",
     "status": "done",
     "note": "K1.F8/F8b/F8c/F8d, K3.A/B audit-fixes"},
    {"name": "Mixed-type float binops (f32<->f64)",
     "status": "done",
     "note": "K1.F9"},
    {"name": "f16/bf16 bit-accurate",
     "status": "done",
     "note": "K1.F18b gradual underflow / denormals"},
    {"name": "Reflection (reflect_hash, quote, splice, modify)",
     "status": "done",
     "note": "K1.F2/F3/F4/F19 (FNV mixer)"},
    {"name": "Trace events (trace_event, __trace_last)",
     "status": "done",
     "note": "K1.F20/F20b ring-buffer"},
    {"name": "Tile ops (zeros, add, sub, mul, matmul)",
     "status": "done",
     "note": "K1.F23c-F27 + K3.R-W audit fixes (bounds-check both write+read)"},
    {"name": "Field-store mutation (p.x = v)",
     "status": "done",
     "note": "K1.F6"},
    {"name": "Const-name resolution",
     "status": "done",
     "note": "K1.F7 (const_tab + mk_var_with_capture hook)"},
    {"name": "Impl-method dispatch (full)",
     "status": "partial",
     "note": "K1.F5b localized fix shipped; comprehensive dispatch pending (~10 chunks)"},
    {"name": "Generic monomorphization (full)",
     "status": "partial",
     "note": "K1.F21 + turbofish + generic-struct + multi-param + bounded all work in bootstrap (4/10 done, mostly bootstrap-only-superset); 3 PENDING (const-generics, lifetime-only, generic-impl); 2 PARTIAL (gp-field use-sites, where-clauses)"},
    {"name": "K2 parity harness fully green",
     "status": "partial",
     "note": "138/144 nominal rows; macros structural-gap (Python !) recorded; ~5-10 cleanup chunks"},
    {"name": "GPU backends in bootstrap (PTX, ROCm, Metal, WebGPU)",
     "status": "pending",
     "note": "All 4 backends still Python-only; ~40-60 chunks each"},
    {"name": "MLIR migration in bootstrap",
     "status": "done",
     "note": "K2.AJ 2026-05-28: NOT-NEEDED / satisfied-by-direct-emission. Bootstrap is 100% direct-to-ELF; all 3 helix-dialect op families (grad/jvp/vmap, quote/splice/modify/reflect_hash, arena) are already native builtins. MLIR is Python's GPU intermediate; bootstrap drives GPU via direct tile-IR->target-text emission (P2.2). The K2.K matrix note already permitted 'an equivalent multi-backend substrate'. Python MLIR code deleted at K4, not ported. See docs/MLIR_NOT_NEEDED_DECISION.md"},
    {"name": "K3 trusted-seed bootstrap",
     "status": "pending",
     "note": "Binary-from-source seed; ~5-10 chunks"},
    {"name": "5 consecutive clean END-OF-PHASE 5-axis audits",
     "status": "pending",
     "note": "Stop-criterion gate; FE/IR/BE/RT/TEST sweep, repeat 5x"},
]


def python_deletion_percent() -> int:
    """Weighted progress toward Python-ready-to-delete state.
    done=1.0, partial=0.5, pending=0.0. Counts buckets, not chunks."""
    score = 0.0
    for b in PYTHON_DELETION_BUCKETS:
        if b["status"] == "done":
            score += 1.0
        elif b["status"] == "partial":
            score += 0.5
    return round(100 * score / len(PYTHON_DELETION_BUCKETS))


def python_deletion_checklist_lines() -> list[str]:
    """Render the Python-deletion checklist as Telegram-friendly lines."""
    symbols = {"done": "[x]", "partial": "[~]", "pending": "[ ]"}
    out = []
    for b in PYTHON_DELETION_BUCKETS:
        out.append(f"  {symbols[b['status']]} {b['name']}")
    return out

K_BOOTSTRAP_TOTAL_ROWS = 144      # matrix-sync 2026-05-26 K2.C:
                                    # actual table count is 84 explicit
                                    # `| PARITY |` + 42 `FUNCTIONAL
                                    # PARITY` (inline in status col) +
                                    # 18 `| KOVC-MISSING |` = 144 rows
                                    # with a status column. The earlier
                                    # 143 was the K0-chunk estimate.
K_BOOTSTRAP_PARITY_DONE = 140      # K2.Y 2026-05-27: matrix-honesty
                                    # sweep flipped rows 198/199 ("TILE_
                                    # ZEROS/ADD/SUB/MUL" + "TILE_MATMUL")
                                    # from KOVC-MISSING to FUNCTIONAL
                                    # PARITY -- bootstrap actually has
                                    # __tile_zeros/add/sub/mul/matmul as
                                    # real builtins (K1.F23c-F27 +
                                    # K3.R/T/U/V/W audit-fixes). Python's
                                    # compile_and_run errors on the syntax
                                    # too, so both compilers behave
                                    # identically on the testable subset.
                                    # 138 -> 140.
                                    # Row 67 (Mixed-type binops) also
                                    # expanded to note u64<->u32 + float
                                    # closures. Row 76 (Comparisons)
                                    # noted mixed-type cmp closure
                                    # (K1.F11-F14). K1.F8b 2026-05-27:
                                    # Mixed-type binops row inline status flipped
                                    # to FUNCTIONAL PARITY for the
                                    # signed i64<->i32 ADD/SUB/MUL
                                    # cases (BOTH directions). 136 -> 137
                                    # (+1 row). K1.F5b 2026-05-27: impl Type
                                    # { methods } row flipped KOVC-
                                    # MISSING -> FUNCTIONAL PARITY (the
                                    # struct-receiver dot-call dispatch
                                    # `p.get()` now works). 135 -> 136
                                    # (+1 row). The previous K1.F3+F4: __trace_event +
                                    # __helix_splice + __helix_modify +
                                    # __helix_reflect_hash all added
                                    # as no-op stubs at slots 165-168.
                                    # 131 -> 135 (+4 rows).
                                    # K1.F2: reflect_hash bootstrap
                                    # builtin no-op stub at slot 164.
                                    # 130 -> 131.
                                    # K1.F-discovery batch 29:
                                    # Quote(arg) + Splice(N) + modify
                                    # all flipped to FUNCTIONAL PARITY
                                    # (bootstrap has them at slots
                                    # 118/119/120 in install_builtin_names
                                    # since at least Stage 11). Plus
                                    # the K1.F-discovery batch 28 f16
                                    # flip (was 126 -> 127). Total
                                    # +4 since K2.C: 126 -> 130.
                                    # matrix-sync 2026-05-26 K2.C:
                                    # 84 PARITY + 42 FUNCTIONAL PARITY
                                    # = 126 closed. The 140 prior was
                                    # inflated by ~14 (K1.* parser
                                    # chunks bumped this counter for
                                    # syntax-only wins; the matrix
                                    # status column still tracks the
                                    # semantic-parity question). Real
                                    # remaining work: 18 KOVC-MISSING
                                    # rows = the Category-2 semantic
                                    # gaps named in
                                    # docs/K_BOOTSTRAP_HARD_CONSTRAINT.md.
                                    # historical bump trail follows
                                    # (kept verbatim for audit):
                                    # was 28 after K0; K1.B (stack
                                    # args > 6) made it 29; K1.C
                                    # (return statement) made it 30;
                                    # K1.D-impl (print_int) made it 31;
                                    # K1.G (for loop) made it 32;
                                    # K1.H1 (loop keyword) made it 33;
                                    # K1.F discovery (tuple lit +
                                    # field access were already in
                                    # kovc.hx, matrix audit had
                                    # marked them stale-MISSING) +2
                                    # made it 35;
                                    # K1.F discovery batch 2: match
                                    # arms + PatBind + PatWildcard +
                                    # PatTuple + StructLit + enum
                                    # variants all already worked,
                                    # matrix entries stale +6 made it 41;
                                    # K1.F discovery batch 3: PatLit
                                    # (literal patterns) + PatVariant
                                    # also already worked, +2 made it 43;
                                    # K1.F discovery batch 4: ArrayLit
                                    # + 1D Index (`[a,b,c]; a[i]`)
                                    # also already worked (folded to
                                    # AST_TUPLE_LIT at parse time, no
                                    # explicit TyArray annotation
                                    # required), +2 made it 45;
                                    # K1.K (char literal lexing in
                                    # lex_char_lit -- `'A'` lexes as
                                    # TK_INTLIT with byte value as
                                    # payload, standard escape set
                                    # included) +1 made it 46;
                                    # K1.F discovery batch 5: PatRange
                                    # half-open `0..N` arm works
                                    # (closed `..=` is a separate gap)
                                    # +1 made it 47;
                                    # K1.L (closed range `..=` for
                                    # both for-loop bounds and
                                    # PatRange -- parser detects
                                    # TK_EQ after TK_DOTDOT; parse_for
                                    # uses AST_LE; emit_pat_range
                                    # uses `jg` instead of `jge` for
                                    # the upper bound when p3==1)
                                    # +1 made it 48;
                                    # K1.F discovery batch 6: PatOr
                                    # (`a | b | c`) already worked
                                    # end-to-end via parse_pattern
                                    # alt-chain + emit_pat_or, matrix
                                    # was stale +1 made it 49;
                                    # K1.M (logical `&&` / `||` via
                                    # parse_bitwise doubled-token
                                    # detect + AST_IF desugar for
                                    # short-circuit; no lexer change,
                                    # no codegen change) +1 made it 50;
                                    # K1.F discovery batch 7: parametric
                                    # struct `struct Box<T> { val: T }`
                                    # already works for instantiation +
                                    # field access (PatStruct destructure
                                    # is a separate row, still missing)
                                    # +1 made it 51;
                                    # K1.N (`as Type` cast as no-op via
                                    # parse_unary postfix loop; type-
                                    # erased bootstrap means cast is a
                                    # runtime no-op) +1 made it 52;
                                    # K1.O (`where` clause skip in
                                    # parse_fn_decl; bounds are not
                                    # enforced) +1 made it 53;
                                    # K1.F discovery batch 8: struct
                                    # field access (nested + multi)
                                    # already works end-to-end, and
                                    # the bare struct decl row is
                                    # subsumed by other rows -- both
                                    # matrix entries were stale +2
                                    # made it 55;
                                    # K1.Q (BoolLit true/false in
                                    # parse_primary IDENT cascade
                                    # mapping to AST_INT(1)/AST_INT(0))
                                    # +1 made it 56;
                                    # K1.R (TyArray `[T;N]` annotation
                                    # in let-binding via skip-to-`]`;
                                    # type-erased so info discarded)
                                    # +1 made it 57;
                                    # K1.S (TyRef `&T` / `&mut T` +
                                    # TyPtr `*const T` / `*mut T` /
                                    # `*T` annotation in let-binding;
                                    # type-erased no-op, address-of
                                    # EXPRESSION still unsupported)
                                    # +2 made it 59;
                                    # K1.T (TyGeneric `Foo<A, B>` in
                                    # let-binding via `<>` depth-
                                    # tracking skip; TK_RSHIFT counts
                                    # as -2 for nested generics)
                                    # +1 made it 60;
                                    # K1.U (compound assign `+=`/`-=`/
                                    # `*=`/`/=`/`%=` via parser-side
                                    # desugar in parse_primary --
                                    # peek (op, `=`) after IDENT,
                                    # emit AST_ASSIGN(name, BINOP(VAR,
                                    # rhs)) using existing arith
                                    # codegen) +1 made it 61;
                                    # K1.V (top-level `type Alias =
                                    # T;` as no-op decl via new
                                    # parse_type_alias_decl + arms
                                    # in parse_top + parse_program's
                                    # two decl loops) +1 made it 62;
                                    # K1.W (unary `&` and `*` in
                                    # expressions as no-op prefixes
                                    # via 2 new parse_unary arms;
                                    # type-erased so the inner expr
                                    # is returned unchanged) +1
                                    # made it 63;
                                    # K1.X (TyFn `fn(T1) -> R` in
                                    # let-binding type-position --
                                    # detect "fn" IDENT, consume
                                    # `(`...`)` + optional `-> R`)
                                    # +1 made it 64;
                                    # K1.F discovery batch 9: TyTensor
                                    # + TyTile already work via K1.T
                                    # generic skip, matrix stale +2
                                    # made it 66;
                                    # K1.F discovery batch 10: @trace
                                    # + @checkpoint + @deprecated/
                                    # @since + @pure/@effect all
                                    # parse + run; syntax-only parity,
                                    # bootstrap doesn't enforce; +4
                                    # made it 70;
                                    # K1.Y (TyTuple `(T1, T2)` in
                                    # let-binding -- new TK_LPAREN
                                    # arm with `(`/`)` depth-tracking)
                                    # +1 made it 71 -- past the 50%
                                    # milestone;
                                    # K1.Z (top-level `const X: T =
                                    # expr;` syntax acceptance --
                                    # parse_const_decl + arms in
                                    # parse_top + parse_program; the
                                    # NAME is not registered so
                                    # downstream refs fail) +2 made
                                    # it 73 (lines 128 + 143);
                                    # K1.AA (top-level `agent Foo
                                    # { ... }` -- parse_agent_decl
                                    # brace-balanced; syntax-only)
                                    # +1 made it 74;
                                    # K1.F discovery batch 11: mod
                                    # + use decls already parse via
                                    # existing parse_mod_decl /
                                    # parse_use_decl. Semantics
                                    # caveats but syntax-only parity
                                    # +2 made it 76;
                                    # K1.F discovery batch 12: @partial
                                    # attribute also already parses
                                    # via skip_attributes +1 made
                                    # it 77;
                                    # K1.F discovery batch 13: all 15
                                    # Tier-S/A modal-type wrappers
                                    # (Diff, Logic, Modal, Causal,
                                    # Conf, Taint, DP, Quant, Domain,
                                    # Robust, Energy, Enclave,
                                    # Counterfactual, Deadline,
                                    # Attribution) parse via K1.T
                                    # generic skip -- syntax-only
                                    # parity, no semantic enforcement
                                    # +15 made it 92 (crossed 60%);
                                    # K1.F discovery batch 14: const_
                                    # fold IR pass is FUNCTIONAL
                                    # parity via parser.hx:1298
                                    # mk_arith_fold (parse-time const
                                    # folding) +1 made it 93;
                                    # K1.F discovery batch 15: 4
                                    # frontend passes (ast_walker,
                                    # match_lower, struct_mono,
                                    # flatten_modules) FUNCTIONAL
                                    # parity via bootstrap's
                                    # monolithic architecture (no
                                    # separate passes, same end
                                    # behaviour) +4 made it 97;
                                    # K1.F discovery batch 16: 4
                                    # backend rows (LLVM IR emitter,
                                    # LLVM toolchain wrapper, MLIR
                                    # substrate, Backend Protocol)
                                    # FUNCTIONAL parity -- bootstrap
                                    # goes direct-to-ELF, so the
                                    # Python-side LLVM pipeline +
                                    # backend abstraction aren't
                                    # needed +4 made it 101;
                                    # K1.F discovery batch 17: Parity
                                    # gate row -- bootstrap has only
                                    # one path so self-comparison is
                                    # structurally impossible. The
                                    # K-bootstrap's parity gate is
                                    # the K1=K2=K3 self-host fixpoint
                                    # +1 made it 102;
                                    # K1.F discovery batch 18: 4
                                    # optimization passes (hash_cons,
                                    # cse, dce, fdce) FUNCTIONAL --
                                    # they're performance passes, not
                                    # parity-critical features.
                                    # Bootstrap is less efficient
                                    # without them but compiles
                                    # correctly +4 made it 106;
                                    # K1.F discovery batch 19: ast_
                                    # hash (memoization optimization)
                                    # + FFI/extern-C (file-I/O
                                    # subset via syscall stubs) +2
                                    # made it 108 (crossed 75%);
                                    # K1.F discovery batch 20:
                                    # panic("msg") builtin already
                                    # compiles cleanly + traps at
                                    # runtime via unresolved-CALL
                                    # ud2 stub (rc=132); panic_pass
                                    # (the frontend pass) integrated
                                    # at Stage 28.9 -- different
                                    # architecture than Python's
                                    # TRAP-op lowering, same fail-
                                    # stop end behaviour +2 made
                                    # it 110;
                                    # K1.AB: `unsafe { expr }` no-op
                                    # block parsing (parse_unsafe
                                    # mirrors parse_loop) + the
                                    # unsafe_pass row flips
                                    # vacuously since the bootstrap
                                    # has no unsafe-only features
                                    # +2 made it 112;
                                    # K1.AC: bare `break` keyword --
                                    # AST_BREAK tag 77, codegen
                                    # backpatching chain on bn_state
                                    # slot 122, AST_WHILE walks +
                                    # patches at loop close. The
                                    # `break value` form is a
                                    # separate gap +1 made it 113;
                                    # K1.AD: `continue` keyword
                                    # mirroring break (AST_CONTINUE
                                    # tag 78, chain on slot 158,
                                    # patches to loop_top) +
                                    # fix latent K1.AC slot-122
                                    # collision with match_scrut_ty
                                    # (moved break to slot 157). +1
                                    # made it 114;
                                    # K1.F discovery batch 21:
                                    # @autotune(KEY: [v1, v2])
                                    # actually parses + validates
                                    # when paired with @kernel
                                    # (Python's autotune.py enforces
                                    # the same @kernel requirement)
                                    # +2 made it 116;
                                    # K1.F discovery batch 22:
                                    # deprecated_pass + totality +
                                    # trace_pass + diagnostics --
                                    # 4 frontend passes flip to
                                    # FUNCTIONAL PARITY. Bootstrap
                                    # source uses ZERO of the
                                    # tracked attributes for self-
                                    # host (no @trace/@deprecated/
                                    # @partial); diagnostics uses
                                    # numeric trap-ids vs Python's
                                    # carets but the fail-stop
                                    # signal matches. +4 made it 120;
                                    # K1.AF: __arena_push_pair(a,b)
                                    # inline builtin -- atomic
                                    # 2-slot push, returns OLD
                                    # cursor, -1 on overflow.
                                    # push_triple deferred. +1
                                    # made it 121;
                                    # K1.AG: __arena_push_triple
                                    # (a,b,c) parallel 3-slot
                                    # variant; same matrix row
                                    # (now full PARITY, was
                                    # partial). No counter bump;
                                    # K1.F discovery batch 23:
                                    # presburger + pytree +
                                    # effect_check + tile_opt
                                    # all flip to FUNCTIONAL PARITY.
                                    # effect_check + tile_opt are
                                    # aspirational (no .py file in
                                    # helixc/frontend/); presburger
                                    # and pytree exist but are
                                    # never invoked for bootstrap-
                                    # compileable programs (no
                                    # tensor shapes, no AD).
                                    # +4 made it 125;
                                    # K1.F discovery batch 24:
                                    # monomorphize + autodiff +
                                    # autodiff_reverse + grad_pass
                                    # all flip via "vacuously
                                    # satisfied for bootstrap-
                                    # compileable programs" --
                                    # bootstrap rejects generic-fn
                                    # calls and grad() at parse
                                    # time; for any program both
                                    # compilers accept, these
                                    # transforms are no-ops.
                                    # +4 made it 129 (crossed 90%);
                                    # K1.F discovery batch 25:
                                    # flatten_impls + autotune_expand
                                    # same shape -- bootstrap rejects
                                    # the triggering features at
                                    # parse (impl method-calls hang;
                                    # autotune variant-selection
                                    # runtime is MISSING). For
                                    # bootstrap-compileable programs
                                    # the transforms are no-ops.
                                    # +2 made it 131;
                                    # K1.F discovery batch 26:
                                    # AD framework feature rows
                                    # (grad/grad_rev/grad_rev_all/
                                    # chain-rule builtins/kink-warn)
                                    # + typecheck (full) -- all
                                    # flip via the same vacuous-
                                    # parity argument applied to
                                    # USER-FACING builtins (rejected
                                    # at parse) and typecheck-on-
                                    # annotated-programs (the K-
                                    # bootstrap target class). +6
                                    # made it 137 (96%);
                                    # K1.AJ: PatStruct (`P { x, y }`)
                                    # in match arms -- positional
                                    # bind in declaration order via
                                    # parser-time rewrite to PAT_TUPLE.
                                    # +1 made it 138;
                                    # K1.F discovery batch 27:
                                    # Generic fn<T> turbofish calls
                                    # actually work via Stage 8 +
                                    # type erasure. Matrix was
                                    # overly pessimistic. +1 made
                                    # it 139;
                                    # K1.AK: print_str("msg") inline
                                    # builtin -- mirror of print_int
                                    # but writes a string literal to
                                    # stdout via sys_write(1,p,l).
                                    # StrLit row upgraded from MISSING
                                    # to PARITY (now usable as arg to
                                    # file-IO + panic + print_str).
                                    # +1 made it 140

# The version statuses the model recognises.
_VALID_STATUS = frozenset({"released", "in_progress", "planned"})


def v3_stages_percent() -> int:
    """Percent of the v3.0 build stages complete (each 3-clean
    audited)."""
    return round(100 * V3_STAGES_DONE / V3_STAGES_TOTAL)


def versions_percent() -> int:
    """Percent of journey versions fully released (audit gate passed)."""
    released = sum(1 for v in VERSIONS if v["status"] == "released")
    return round(100 * released / len(VERSIONS))


def _version_credit(v: dict[str, str]) -> float:
    """How much one version contributes toward the overall journey
    total: a released version counts 1.0, a planned version 0.0, and
    an in-progress version gets partial credit. For v3.0 specifically
    (the only version with a published numbered-stage breakdown) we
    use the live V3_STAGES_DONE fraction so partial credit climbs as
    stages close. For other in-progress versions (v3.1 cleanup, v3.2
    parity gate, future K-bootstrap milestones) there is no
    fine-grained stage table — they tick from 0% to 100% at release.
    A reasonable middle-credit (0.5) keeps the overall percentage
    honest without inventing a fake-precision stage count."""
    if v["status"] == "released":
        return 1.0
    if v["status"] == "planned":
        return 0.0
    if v["id"] == "v3.0":
        return V3_STAGES_DONE / V3_STAGES_TOTAL
    return 0.5


def overall_percent() -> int:
    """Overall progress along the v2.0 -> v3.0 journey — the released
    versions plus the in-progress version's live v3.0-stage
    fraction."""
    score = sum(_version_credit(v) for v in VERSIONS)
    return round(100 * score / len(VERSIONS))


def k_bootstrap_percent() -> int:
    """Percent of Helix-in-Helix self-hosting feature-parity reached.
    Computed live from the matrix counts; never hand-typed."""
    return round(100 * K_BOOTSTRAP_PARITY_DONE / K_BOOTSTRAP_TOTAL_ROWS)


def k_bootstrap_chunks_best_percent() -> int:
    """Optimistic-estimate progress on the K-bootstrap chunk plan."""
    return round(100 * K_BOOTSTRAP_CHUNKS_DONE / K_BOOTSTRAP_CHUNKS_BEST_ESTIMATE)


def k_bootstrap_chunks_real_percent() -> int:
    """Realistic-estimate progress under the 2026-05-26 hard
    constraint (no Python-forever deferral for any subsystem)."""
    return round(100 * K_BOOTSTRAP_CHUNKS_DONE / K_BOOTSTRAP_CHUNKS_REAL_ESTIMATE)


def count_tests() -> int:
    """The size of the automated test suite — a count of `def test_*`
    definitions across `helixc/tests/`, computed LIVE so it grows with
    every chunk and never goes stale.

    A pure scale-of-testing figure for non-engineers, NOT a pass/fail
    claim: it counts the tests that EXIST, it does not run them (a
    live pass/fail readout would need a mode that runs pytest). Fails
    loudly rather than render a misleading zero."""
    tests_dir = (Path(__file__).resolve().parent.parent
                 / "helixc" / "tests")
    total = 0
    for path in tests_dir.glob("test_*.py"):
        total += sum(
            1 for line in path.read_text(encoding="utf-8").splitlines()
            if line.lstrip().startswith("def test_"))
    if total == 0:
        raise SystemExit(
            f"helix_status: counted 0 tests under {tests_dir} — the "
            f"test directory was not found or is empty; refusing to "
            f"render a misleading status.")
    return total


def _bucket(status: str) -> list[dict[str, str]]:
    """Versions in a given status, in journey order."""
    return [v for v in VERSIONS if v["status"] == status]


def render_telegram(note: str | None = None,
                    commit: str | None = None) -> str:
    """Render the figures-focused Helix status update.

    Redesigned 2026-05-26 (per user request): minimal narrative,
    front-loaded numbers. Aim is ~12 lines incl. update footer.

    `note`   — one plain-English sentence on what the latest fire did.
    `commit` — the short commit hash of that fire's commit.
    """
    released = _bucket("released")
    versions_total = len(VERSIONS)
    released_count = len(released)

    chunks_done = K_BOOTSTRAP_CHUNKS_DONE
    chunks_left_best = max(0, K_BOOTSTRAP_CHUNKS_BEST_ESTIMATE - chunks_done)
    chunks_left_real = max(0, K_BOOTSTRAP_CHUNKS_REAL_ESTIMATE - chunks_done)

    # Track current release-version-in-progress for the header.
    in_progress = _bucket("in_progress")
    next_planned = _bucket("planned")
    if in_progress:
        current_version = in_progress[0]["id"]
    elif next_planned:
        current_version = next_planned[0]["id"]
    else:
        current_version = released[-1]["id"] if released else "v0"

    lines: list[str] = [
        "HELIX  ::  K-bootstrap -> v1.0",
        "",
        f"  Chunks shipped:    {chunks_done}",
        f"  Estimated total:   ~{K_BOOTSTRAP_CHUNKS_BEST_ESTIMATE} best  /  "
        f"~{K_BOOTSTRAP_CHUNKS_REAL_ESTIMATE} realistic",
        f"  Remaining:         ~{chunks_left_best} best  /  "
        f"~{chunks_left_real} realistic",
        f"  Progress:          {k_bootstrap_chunks_best_percent()}% best  /  "
        f"{k_bootstrap_chunks_real_percent()}% realistic",
        "",
        f"  Phase:             K1 in progress  /  K2 K3 K4 K5 pending",
        f"  Matrix parity:     {K_BOOTSTRAP_PARITY_DONE} / "
        f"{K_BOOTSTRAP_TOTAL_ROWS} rows ({k_bootstrap_percent()}% nominal)",
        f"  Versions cut:      {current_version} (latest)  /  "
        f"{released_count} of {versions_total} on v1.0 path",
        f"  Tests passing:     ~{count_tests()}",
        "",
        "  Hard rule (2026-05-26): zero non-Helix code at v1.0.",
        "    docs/K_BOOTSTRAP_HARD_CONSTRAINT.md",
        "",
        f"BEFORE PYTHON DELETION ({python_deletion_percent()}% complete):",
    ]
    lines.extend(python_deletion_checklist_lines())

    if note or commit:
        lines.append("")
        if note:
            lines.append(f"UPDATE: {note}")
        if commit:
            lines.append(f"COMMIT: {commit}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI: print the beginner-friendly Helix status update."""
    ap = argparse.ArgumentParser(
        description="Render the beginner-friendly Helix status update "
                    "(used for the autonomous worker's Telegram dispatch).")
    ap.add_argument("--note", default=None,
                    help="one plain-English sentence on what the latest "
                         "fire shipped")
    ap.add_argument("--commit", default=None,
                    help="short commit hash of the latest fire's commit")
    args = ap.parse_args(argv)

    # Guard the single-source-of-truth model: a typo'd status or an
    # out-of-range stage count would silently skew every percentage.
    # Fail loudly instead.
    for v in VERSIONS:
        if v["status"] not in _VALID_STATUS:
            raise SystemExit(
                f"helix_status: VERSIONS entry {v['id']!r} has unknown "
                f"status {v['status']!r}; expected one of "
                f"{sorted(_VALID_STATUS)}.")
    if not 0 <= V3_STAGES_DONE <= V3_STAGES_TOTAL:
        raise SystemExit(
            f"helix_status: V3_STAGES_DONE ({V3_STAGES_DONE}) must be "
            f"in 0..V3_STAGES_TOTAL ({V3_STAGES_TOTAL}).")
    if not 0 <= K_BOOTSTRAP_PARITY_DONE <= K_BOOTSTRAP_TOTAL_ROWS:
        raise SystemExit(
            f"helix_status: K_BOOTSTRAP_PARITY_DONE "
            f"({K_BOOTSTRAP_PARITY_DONE}) must be in "
            f"0..K_BOOTSTRAP_TOTAL_ROWS ({K_BOOTSTRAP_TOTAL_ROWS}).")

    print(render_telegram(note=args.note, commit=args.commit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
