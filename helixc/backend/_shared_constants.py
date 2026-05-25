"""
helixc/backend/_shared_constants.py — backend-shared runtime layout.

Every Helix backend (`x86_64.py`, `llvm_ir.py`, future MLIR-driven
emitters) must agree on the runtime layout: how many reflection cells
the program gets, how big the arena is, how big the trace ring buffer
is, the SysV stack-arg ABI. Pre-v3.1 each backend hard-coded these
constants separately and kept them in sync via a parity-gate test
(`test_stage206r_num_cells_matches_x86_backend` et al.) — a drift
surface that grew with every new backend.

This module is the single source of truth. The constants here are the
public ABI of the runtime: any consumer (frontend lowering passes,
backends, runtime helpers, tests) reads from here. A cap bump (e.g.
arena from 8 MB to 16 MB) is one edit, not three.

DEPENDENCY HYGIENE: this module is a LEAF. It imports nothing from
`helixc.backend.*` to avoid a circular dependency — both backend
modules import from here at top level. Any future shared backend
helper that needs Python imports goes in a sibling module, not here.

Stability: these values are LOAD-BEARING ABI. Changing them requires:
1. A coordinated bump across every backend's emit pipeline.
2. A parity-gate re-run (Stage 207) to confirm structurally-identical
   output across backends after the bump.
3. A note in V3_HANDOFF.md (the cap bump is a user-observable
   capacity change).

v3.1 step 6a (this module's creation) replaces the per-backend private
copies with shared imports; the constants themselves are unchanged.
"""

# ---------------------------------------------------------------------------
# Reflection cells (verifier-gated self-modification primitives:
# QUOTE / SPLICE / MODIFY / REFLECT_HASH)
# ---------------------------------------------------------------------------
# Count of mutable i64 cells available to a program. Each cell is
# addressed by a compile-time-or-runtime i32 handle in [0, NUM_CELLS).
# The count is intentionally small (64) — programs that need more
# reflection state chain handles via MODIFY or use the arena.
HELIX_NUM_CELLS = 64

# Per-cell size in bytes. Cells store i64 values (or bit-patterns of
# f64s reinterpreted as i64, or low-bit-truncated i32 splices). The
# size is part of the helper-call ABI between op handlers and the
# backend-emitted helpers.
HELIX_CELL_SIZE = 8

# ---------------------------------------------------------------------------
# Arena allocator (bump-allocated i32 region for compile-time / runtime
# scratch — ASTs during self-host, IR ops, symbol tables, ELF output)
# ---------------------------------------------------------------------------
# Slot count. 2 097 152 i32 slots × 4 bytes = 8 MB BSS arena. Sized
# for self-host: the bootstrap source (lexer + parser + kovc, ~111 KB)
# lands as 111 K slots; tokens add ~30 K * 4 = 120 K; AST ~5 K * 5 =
# 25 K; ELF output ~30 K. Total ~290 K — well under 2 M, room for
# fn_table / patch_table / str_state. Lives in BSS so cap bumps do
# not inflate produced binary file size.
#
# Slot 0 is reserved for the cursor. Slots 1..HELIX_ARENA_CAP are
# user-allocatable; `__arena_push` returns the slot index, advancing
# the cursor. Bounds checks compare cursor against this constant.
HELIX_ARENA_CAP = 2097152

# ---------------------------------------------------------------------------
# Trace ring buffer (compile-time-zero-overhead @trace fn
# instrumentation)
# ---------------------------------------------------------------------------
# Event count. Each event is 8 bytes (4 fn_id + 4 kind+value). 1024
# events = 8 KB BSS overhead — enough for typical @trace-instrumented
# programs without truncation. Phase-0 fail-closed: when the buffer
# is full, subsequent events are silently dropped (no allocation, no
# syscall, no blocking).
HELIX_TRACE_CAP = 1024

# ---------------------------------------------------------------------------
# SysV ABI stack-arg layout (callee's view)
# ---------------------------------------------------------------------------
# Bytes above the saved-rbp slot where stack-passed args begin.
# Layout: [rbp + 0] = saved rbp, [rbp + 8] = return address,
# [rbp + 16] = first stack arg. The 16 = saved_rbp_size +
# return_addr_size; both 8 bytes on x86-64.
SYSV_STACK_ARG_BASE = 16

# Stride between consecutive stack args. Each stack arg occupies 8
# bytes regardless of its actual payload size (f32 pads to 8).
SYSV_STACK_ARG_STRIDE = 8

# Required stack alignment before a CALL instruction. SysV mandates
# rsp ≡ 0 (mod 16) at the call site so the callee can use SSE/AVX
# spills with the natural alignment.
SYSV_STACK_ALIGNMENT = 16
