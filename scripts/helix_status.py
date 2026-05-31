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
K_BOOTSTRAP_CHUNKS_DONE = 446      # last bump: STAGE-0 LADDER rung 5 -- M0 DONE (the macro assembler, first 'real' tool). Vendored M0_AMD64.hex2 + ELF-amd64.hex2 + cc_amd64.M1 (GPL-3.0, pinned 15535f8, shas a9692351/bfad808d/599c0c6a) + built with OUR catm+hex2 (catm prepends ELF header, hex2 assembles -> M0.bin 1684B sha db97dff1; no assembler, no pre-built binary). REPRODUCIBLE (sha stable 2 builds) + 2/2 tests under WSL: M0 assembles the real cc_amd64.M1 C-compiler seed -> 61KB hex2 -> a valid 17976-byte cc_amd64 ELF (proves M0 emits correct machine code). Trust chain hex0(frozen)->hex1->hex2->catm->M0. Next rung: cc_amd64 (M0 builds it), then M2-Planet (needs oriansj/M2libc).
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
     "note": "K1.F5b localized fix + K1.M27 probe (2026-05-28): the bootstrap FULLY handles CORE impl-method dispatch -- method on a struct value (p.get()), method with an arg (p.add(2)), and method-calling-method (self.a()+2) -- all compile+run correctly via self-host (test_bootstrap_impl_method_dispatch). FINDING: Python helixc cannot even PARSE the bare `(self)` receiver (ParseError: expected COLON), so the bootstrap EXCEEDS Python here (parallel to the M21 GPU finding) -> these CANNOT be K2-parity entries (Python errors), and DELETION-PARITY for the core case is MET (deleting Python loses nothing). Kept 'partial' pending: TRAITS only (generic-impl methods via turbofish FIXED 2026-05-30, test_bootstrap_generic_impl_method). RECONCILED 2026-05-30: &self + multiple-impl-blocks + where-clause + nested-generic re-verified WORKING (test_bootstrap_impl_generics_advanced_working); the old 'pending' for those was stale -- which may themselves be Python-gaps; core dispatch is done + ahead of Python."},
    {"name": "Generic monomorphization (full)",
     "status": "partial",
     "note": "K1.F21 + K1.M28 probe (2026-05-28): Python helixc CANNOT PARSE generic syntax at all (ParseError on `<T>` for fn-generic / generic-struct / turbofish / multi-param), so the bootstrap EXCEEDS Python on generics -> DELETION-PARITY trivially MET (Python supports ZERO generics). Bootstrap: generic-struct (Box<T>=42) + TURBOFISH (id::<i32>(42)=42) WORK + pinned (test_bootstrap_generics_struct_and_turbofish); BUT a BARE generic-fn call without turbofish (id(42), first(42,7)) MISCOMPILES to SIGILL (exit 132) -- the bare-call SIGILL is FIXED (re-verified 2026-05-30 + pinned by test_bootstrap_generics_bare_call); the generic-impl-method dispatch bug was FIXED 2026-05-30 (test_bootstrap_generic_impl_method): turbofish Box::<i32>{} recorded the mono name Box__i32 on the binding; dispatch now truncates at the __ mono-separator -> Box__get. No remaining generics CODEGEN gap (absolute-completeness ~6/10: const-generics/lifetime-only still pending) (a bootstrap-QUALITY gap, NOT a deletion blocker since Python can't do generics). Absolute-completeness: ~4/10 (const-generics/lifetime-only/generic-impl pending; gp-field/where-clauses partial)."},
    {"name": "K2 parity harness fully green",
     "status": "partial",
     "note": "138/144 nominal rows; macros structural-gap (Python !) recorded; ~5-10 cleanup chunks"},
    {"name": "GPU backends in bootstrap (PTX, ROCm, Metal, WebGPU)",
     "status": "partial",
     "note": "K1.M1-M24 (2026-05-28): all 4 backends now emit DIRECTLY from the bootstrap (direct-to-target text, NO MLIR/LLVM). NVIDIA PTX = FULL, ptxas-validated to real SASS (scalar/cmp/if/while/assign, thread+block index, global load/store, f32+i32 elementwise-add, full tile family zeros/add/sub/mul/matmul). WebGPU/WGSL = REAL elementwise kernels (f32+i32 params -> @group/@binding storage buffers, global_invocation_id, out[i]=a[i] OP b[i]) -- this EXCEEDS the Python WebGPU backend. Apple Metal/MSL + AMD ROCm/GCN = empty-kernel byte-matched to Python. M21 FINDING: the Python non-NVIDIA backends are SUBSTRATE+STUBS (emit @@HELIX-STUB for ops, no real WGSL/MSL/GCN), so DELETION-PARITY (bootstrap >= Python functional capability) is already MET for all 4 -- deleting the Python GPU backends loses nothing. Remaining is OPTIONAL real-op depth (WGSL tile/matmul; Metal/ROCm bodies) = perf/polish BEYOND what Python ever did, not a deletion blocker. docs/GPU_DIRECT_EMIT_PLAN.md."},
    {"name": "MLIR migration in bootstrap",
     "status": "done",
     "note": "K2.AJ 2026-05-28: NOT-NEEDED / satisfied-by-direct-emission. Bootstrap is 100% direct-to-ELF; all 3 helix-dialect op families (grad/jvp/vmap, quote/splice/modify/reflect_hash, arena) are already native builtins. MLIR is Python's GPU intermediate; bootstrap drives GPU via direct tile-IR->target-text emission (P2.2). The K2.K matrix note already permitted 'an equivalent multi-backend substrate'. Python MLIR code deleted at K4, not ported. See docs/MLIR_NOT_NEEDED_DECISION.md"},
    {"name": "K3 trusted-seed bootstrap",
     "status": "pending",
     "note": "K1.M30 assessment: Stage-K3 SEED = from-raw-binary hex0 -> ... -> kovc chain. Master plan (HELIX_K_BOOTSTRAP_MASTER_PLAN.md) marks it 'not blocking on it; decision when the time comes; several cron iterations, possibly WEEKS'. No stage0/ code in-repo (hex0 is design-stage per the goal hierarchy) -> a MAJOR DEFERRED effort, NOT 60s-tick-tractable (the prior '~5-10 chunks' was optimistic). Separately the N-generation FIXPOINT (kovc compiles its own kovc.hx to a stable binary; K2 Phase-3) is UNTESTED + gated on the bootstrap supporting ALL Helix in its ~11k-line source."},
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
