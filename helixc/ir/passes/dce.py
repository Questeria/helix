"""
helixc/ir/passes/dce.py — dead code elimination on Tensor IR.

After constant folding, intermediate const ops whose results no longer
have any users are dead. DCE removes them.

A value is *live* if it is:
- Used as an operand to a non-removable op (RETURN, BR, COND_BR, CALL,
  STORE_VAR, STORE_ELEM, side-effecting calls), OR
- Used as an operand to another live op.

We compute liveness via a fixpoint reverse-walk, then drop ops whose
results are all dead AND that have no side effects.

Side-effecting op kinds (kept regardless of result use):
- RETURN, BR, COND_BR
- CALL (might have effects)
- STORE_VAR, STORE_ELEM
- ALLOC_VAR, ALLOC_ARRAY (we keep these even though their results aren't
  directly used — backend uses them for layout)
- MODIFY, SPLICE
- io.print, etc.

License: Apache 2.0
"""

from __future__ import annotations

from .. import tir


SIDE_EFFECT_KINDS = {
    tir.OpKind.RETURN,
    tir.OpKind.BR,
    tir.OpKind.COND_BR,
    tir.OpKind.CALL,
    tir.OpKind.STORE_VAR,
    tir.OpKind.STORE_ELEM,
    tir.OpKind.ALLOC_VAR,
    tir.OpKind.ALLOC_ARRAY,
    tir.OpKind.MODIFY,
    tir.OpKind.SPLICE,
    tir.OpKind.PRINT,
    # QUOTE has the side effect of reserving a reflection cell handle
    # for the binary's runtime; even if the i32 result isn't directly
    # used, downstream MODIFY/SPLICE may target the cell by index.
    tir.OpKind.QUOTE,
    # REFLECT_HASH is similar: it provides a stable testing handle that
    # downstream code may reach via cell indexing.
    tir.OpKind.REFLECT_HASH,
    # Arena ops mutate a global region — even if the result (slot index)
    # is unused, the push/set must still execute for downstream reads.
    tir.OpKind.ARENA_PUSH,
    tir.OpKind.ARENA_SET,
    # Stage 36 Inc 9 type-design A2: atomic-pair sibling of ARENA_PUSH.
    # Even if the returned slot index is unused, the two writes must
    # still execute so downstream parent_*_at reads see the values.
    tir.OpKind.ARENA_PUSH_PAIR,
    # Stage 36 Inc 14: atomic-triple sibling of ARENA_PUSH_PAIR — same
    # rationale (three writes must execute even when the result slot
    # index is unused).
    tir.OpKind.ARENA_PUSH_TRIPLE,
    # Stage 16 — HBM tile stores are observable side effects (the write
    # is what kernel launchers care about). Loads + thread_idx are pure
    # functions of their inputs (operands + the implicit %tid.x), so
    # standard liveness is correct for those.
    tir.OpKind.TILE_INDEX_STORE,
    # v2.x re-audit R8 (gate re-run #6 IR LOW): TENSOR_STORE writes to
    # an external file / host buffer (the store counterpart of the
    # external TENSOR_LOAD) — an observable side effect. It was
    # mislabeled pure in _KNOWN_PURE_OPKINDS; a TENSOR_STORE with an
    # unused result would have been DCE-dropped. Latent today (no
    # lowering path emits TENSOR_STORE) — reclassified before v3.0
    # wires it. The R7 hard-fail coverage check catches a MISSING
    # opcode, not a WRONG decision; this is the wrong-decision fix.
    tir.OpKind.TENSOR_STORE,
    # Stage 16.5 follow-up audit CRITICAL-1: FFI calls have side effects
    # by definition (puts, free, mutex_lock, etc.). DCE was silently
    # dropping void-return extern calls because their results were
    # never live. Adding here so liveness preserves them unconditionally.
    tir.OpKind.FFI_CALL,
    # Stage 28.5 — TRAP terminates the process. Even if the result slot
    # is unused, the trap MUST execute (it's the entire point of panic).
    tir.OpKind.TRAP,
    # Audit 28.8 cycle 13 C13-1 (HIGH): TRACE_ENTRY / TRACE_EXIT are
    # @trace prologue/epilogue ops with side effects (the runtime
    # records entry/exit events). TRACE_EXIT consumes the return value
    # as an operand so the runtime can log it; for unit-returning
    # traced fns, lower_ast.py synthesizes a `const_int(0)` whose sole
    # consumer is TRACE_EXIT. Without this entry, DCE's seed phase
    # never marks the const_int(0) live (TRACE_EXIT has no results, so
    # the spread phase has no live result to propagate from either),
    # the producer is dropped, and the backend (x86_64.py:2498)
    # KeyErrors when it tries to look up the slot of the now-deleted
    # operand on `-O2`.
    tir.OpKind.TRACE_ENTRY,
    tir.OpKind.TRACE_EXIT,
}


# Cycle 3 R1 fix batch 21 (IR HIGH-5): exhaustiveness guard against
# silent miscompilation when a new OpKind is added without explicit
# classification as pure vs effectful. Pre-fix: a new side-effecting
# opcode (e.g., future ATOMIC_RMW) would default to "pure" (since
# SIDE_EFFECT_KINDS is a denylist) and DCE would silently drop it
# when its result was unused. Stage 16.5 already caught FFI_CALL
# being missing once; this guard makes the same class of drift
# detectable.
#
# Strategy: positive allowlist of side-effecting opcodes (above) +
# explicit allowlist of pure opcodes (below). Anything in neither is a
# hard build error at module load (v2.x re-audit R7, gate re-run #5 IR
# MEDIUM — upgraded from a soft importwarning): an unclassified opcode
# would default via the SIDE_EFFECT_KINDS denylist to pure-and-
# droppable, so if it were in fact side-effecting DCE would silently
# drop it. Forcing the classification decision matches the sibling
# `tile_opt._check_kind_coverage` assert and is load-bearing for v3.0,
# which adds new IR ops.
_KNOWN_PURE_OPKINDS = {
    # Constants
    tir.OpKind.CONST_INT, tir.OpKind.CONST_FLOAT, tir.OpKind.CONST_BOOL,
    tir.OpKind.CONST_TENSOR,
    # Tensor creation (pure: shape + dtype -> new tensor value)
    tir.OpKind.TENSOR_ZEROS, tir.OpKind.TENSOR_ONES,
    tir.OpKind.TENSOR_FULL, tir.OpKind.TENSOR_RAND,
    tir.OpKind.TENSOR_LOAD,  # external read; pure (a dead load is droppable)
    # Arithmetic
    tir.OpKind.ADD, tir.OpKind.SUB, tir.OpKind.MUL, tir.OpKind.DIV,
    tir.OpKind.MOD, tir.OpKind.NEG, tir.OpKind.ABS,
    tir.OpKind.MAXIMUM, tir.OpKind.MINIMUM, tir.OpKind.POW,
    # Bitwise
    tir.OpKind.SHL, tir.OpKind.SHR, tir.OpKind.BIT_AND, tir.OpKind.BIT_OR,
    tir.OpKind.BIT_XOR, tir.OpKind.BIT_NOT,
    # Transcendentals
    tir.OpKind.EXP, tir.OpKind.LOG, tir.OpKind.SQRT, tir.OpKind.RECIP,
    tir.OpKind.RELU, tir.OpKind.GELU, tir.OpKind.SILU, tir.OpKind.TANH,
    tir.OpKind.SIGMOID,
    # Reductions
    tir.OpKind.REDUCE_SUM, tir.OpKind.REDUCE_MEAN, tir.OpKind.REDUCE_MAX,
    tir.OpKind.REDUCE_MIN, tir.OpKind.REDUCE_PROD,
    # Linear algebra
    tir.OpKind.MATMUL, tir.OpKind.CONV1D, tir.OpKind.CONV2D,
    # Shape ops
    tir.OpKind.RESHAPE, tir.OpKind.TRANSPOSE, tir.OpKind.BROADCAST,
    tir.OpKind.SLICE, tir.OpKind.CONCAT,
    # Casts
    tir.OpKind.CAST, tir.OpKind.BITCAST, tir.OpKind.QUANTIZE,
    tir.OpKind.DEQUANTIZE,
    # Control flow primitives (pure as values)
    tir.OpKind.SELECT, tir.OpKind.WHERE,
    # Comparisons
    tir.OpKind.CMP_EQ, tir.OpKind.CMP_NE, tir.OpKind.CMP_LT,
    tir.OpKind.CMP_LE, tir.OpKind.CMP_GT, tir.OpKind.CMP_GE,
    # Transforms
    tir.OpKind.GRAD, tir.OpKind.JVP, tir.OpKind.VMAP,
    # Memory loads (pure reads from typed slots)
    tir.OpKind.LOAD_VAR, tir.OpKind.LOAD_ELEM,
    # Arena pure reads
    tir.OpKind.ARENA_GET, tir.OpKind.ARENA_LEN,
    # String literal reads
    tir.OpKind.STR_BYTE, tir.OpKind.STR_PTR,
    # GPU
    tir.OpKind.THREAD_IDX, tir.OpKind.TILE_INDEX_LOAD,
    # Result<T,E> pack / unpack
    tir.OpKind.RESULT_PACK, tir.OpKind.RESULT_TAG, tir.OpKind.RESULT_PAYLOAD,
}


def _check_dce_kind_coverage() -> None:
    """Module-load coverage check: every tir.OpKind MUST be explicitly
    classified as side-effecting (SIDE_EFFECT_KINDS) or pure
    (_KNOWN_PURE_OPKINDS). v2.x re-audit R7 (gate re-run #5 IR MEDIUM):
    upgraded from a soft warning to a hard build error. An unclassified
    opcode defaults — via the SIDE_EFFECT_KINDS denylist — to pure-and-
    droppable; if it is in fact side-effecting, DCE silently drops it
    when its result is unused (the exact FFI_CALL drift that bit Stage
    16.5). A build error forces the decision, matching the sibling
    `tile_opt._check_kind_coverage` assert."""
    all_kinds = set(tir.OpKind)
    classified = SIDE_EFFECT_KINDS | _KNOWN_PURE_OPKINDS
    unclassified = all_kinds - classified
    if unclassified:
        names = sorted(k.name for k in unclassified)
        raise AssertionError(
            f"dce.py exhaustiveness: opcode(s) not classified as pure "
            f"or side-effecting: {names}. Classify each in dce.py — "
            f"SIDE_EFFECT_KINDS (if it must execute even when its "
            f"result is unused) or _KNOWN_PURE_OPKINDS (if it is "
            f"liveness-droppable). Leaving an opcode unclassified "
            f"risks a silent miscompile: a side-effecting op dropped "
            f"by DCE."
        )


_check_dce_kind_coverage()


def dce_module(module: tir.Module) -> int:
    """Run DCE on every function. Returns total ops removed."""
    if any(fn.attrs.get("kernel") for fn in module.functions.values()) \
            and not getattr(module, "_helix_kernel_tile_validated", False):
        setattr(module, "_helix_kernel_tile_validation_blocked_by_dce", True)
    total = 0
    for fn in module.functions.values():
        total += dce_function(fn)
    return total


def dce_function(fn: tir.FnIR) -> int:
    """Compute liveness, drop dead ops. Iterates to fixpoint."""
    removed_total = 0
    changed = True
    while changed:
        changed = False
        # Compute live value-ids
        live: set[int] = set()
        # Seed: operands of side-effecting ops are live
        for blk in fn.blocks:
            for op in blk.ops:
                if op.kind in SIDE_EFFECT_KINDS:
                    for o in op.operands:
                        live.add(o.id)
        # Function params are always live
        for p in fn.params:
            live.add(p.id)
        # Block params are always live
        for blk in fn.blocks:
            for p in blk.params:
                live.add(p.id)
        # Fixpoint: any op whose result is live -> its operands are live
        spread = True
        while spread:
            spread = False
            for blk in fn.blocks:
                for op in blk.ops:
                    if op.kind in SIDE_EFFECT_KINDS:
                        continue
                    if any(r.id in live for r in op.results):
                        for o in op.operands:
                            if o.id not in live:
                                live.add(o.id)
                                spread = True

        # Drop ops whose results are all dead AND op has no side effect
        for blk in fn.blocks:
            new_ops = []
            for op in blk.ops:
                if op.kind in SIDE_EFFECT_KINDS:
                    new_ops.append(op)
                    continue
                if not op.results:
                    new_ops.append(op)
                    continue
                if any(r.id in live for r in op.results):
                    new_ops.append(op)
                else:
                    removed_total += 1
                    changed = True
            blk.ops = new_ops
    return removed_total
