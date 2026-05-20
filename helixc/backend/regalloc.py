"""
helixc/backend/regalloc.py — linear-scan register allocation.

v2.4 item 15 (slice 1 of N): the backend-agnostic linear-scan core.

The v2.x GPU backends (ptx / rocm / metal / webgpu) currently emit
substrate-level kernel bodies — operand-less mnemonics and
HELIX-STUB-OPERANDS markers — because no register allocator maps the
tile-IR value SSA onto a finite physical register file. Item 15
closes that gap. This module is the foundation: a pure, backend-
agnostic linear-scan allocator (Poletto & Sarkar, "Linear Scan
Register Allocation", ACM TOPLAS 1999).

Slice 1 deliberately stops at the *algorithm*. It operates on
abstract register indices (0 .. num_registers-1) and abstract value
ids — it knows nothing about %r / %rd / VGPR / SGPR. Per-backend
register-class wiring (mapping a tile-IR value's dtype to a register
class + pool size, then threading the assignment into the emitters)
lands in subsequent item-15 slices.

Why linear-scan and not graph-colouring: linear-scan is O(n log n),
produces good-enough allocations, and is the standard choice for a
JIT-speed / single-pass compiler. Helix's tile-IR kernel bodies are
small (a few dozen values); graph-colouring's extra quality is not
worth its cost here.

License: Apache 2.0
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

from ..ir import tile_ir as ti


@dataclass(frozen=True)
class LiveInterval:
    """A value's live range over a linear instruction numbering.

    `vreg`  — abstract value id (e.g. a tile-IR SSA value index).
    `start` — index of the instruction that defines the value.
    `end`   — index of the last instruction that uses it.

    The interval is closed: [start, end]. A value used only at its
    definition point has start == end. `start > end` is malformed
    and rejected by `linear_scan`.
    """
    vreg: int
    start: int
    end: int


@dataclass(frozen=True)
class RegAllocResult:
    """Outcome of a linear-scan pass.

    `assignment` — vreg -> physical register index (0-based) for every
                   value that got a register.
    `spilled`    — vregs that did not fit the register file and must
                   be spilled to memory (stack / scratch / LDS — the
                   spill *mechanism* is a per-backend later slice).
    `num_registers` — the pool size the pass ran against (echoed so
                   callers can size their `.reg` declarations).

    Invariant: `assignment.keys()` and `spilled` are disjoint, and
    together cover every input vreg exactly once. `register_high_water`
    reports how many distinct physical registers were actually used —
    a backend emits exactly that many register declarations.

    v2.5 polish (item-15 type-design): frozen. An allocation result is
    an immutable fact once the pass returns it — `frozen=True` blocks a
    consumer from rebinding `assignment` / `spilled` (an aliasing-bug
    class). `linear_scan` builds the result by mutating the dict/set
    CONTENTS during the pass; freezing blocks attribute rebinding, not
    content mutation, so that construction is unaffected.
    """
    assignment: dict[int, int] = field(default_factory=dict)
    spilled: set[int] = field(default_factory=set)
    num_registers: int = 0

    @property
    def register_high_water(self) -> int:
        """Count of distinct physical registers used (0 if none)."""
        if not self.assignment:
            return 0
        return max(self.assignment.values()) + 1

    @property
    def spill_count(self) -> int:
        """Number of values that did not fit the register file."""
        return len(self.spilled)


def linear_scan(intervals: list[LiveInterval],
                num_registers: int) -> RegAllocResult:
    """Allocate `num_registers` physical registers to `intervals`.

    Poletto-Sarkar linear-scan:
      1. Sort intervals by increasing start point.
      2. Maintain `active` — intervals currently holding a register,
         kept sorted by increasing end point.
      3. For each interval, first expire every active interval that
         ends before this one starts (returning its register to the
         free pool), then either allocate a free register or — if
         the file is full — spill the interval whose end is furthest
         away (this one or an active one, whichever lives longer).

    Determinism: ties (equal start, or equal end) break by `vreg` so
    the result is byte-stable across runs — required for the codegen
    determinism the Helix test suite pins.

    Raises:
        ValueError: if `num_registers < 1`, or any interval has
            `start > end`, or two intervals share a `vreg`.
    """
    if num_registers < 1:
        raise ValueError(
            f"linear_scan: num_registers must be >= 1, got "
            f"{num_registers}"
        )
    seen_vregs: set[int] = set()
    for iv in intervals:
        if iv.start > iv.end:
            raise ValueError(
                f"linear_scan: LiveInterval for vreg {iv.vreg} is "
                f"malformed — start={iv.start} > end={iv.end}"
            )
        if iv.vreg in seen_vregs:
            raise ValueError(
                f"linear_scan: duplicate vreg {iv.vreg} — each value "
                f"must have exactly one live interval"
            )
        seen_vregs.add(iv.vreg)

    result = RegAllocResult(num_registers=num_registers)

    # Free physical-register pool. Allocating the lowest free index
    # keeps register_high_water minimal + the assignment deterministic.
    free_regs: list[int] = list(range(num_registers))
    # `active` holds (interval, reg_index), kept sorted by interval.end
    # then vreg (deterministic tie-break).
    active: list[tuple[LiveInterval, int]] = []

    ordered = sorted(intervals, key=lambda iv: (iv.start, iv.vreg))

    for iv in ordered:
        # --- ExpireOldIntervals: free registers whose interval ended
        #     strictly before this interval starts. ---
        still_active: list[tuple[LiveInterval, int]] = []
        for act_iv, act_reg in active:
            if act_iv.end < iv.start:
                free_regs.append(act_reg)
            else:
                still_active.append((act_iv, act_reg))
        active = still_active
        free_regs.sort()

        if len(active) >= num_registers:
            # --- SpillAtInterval: the file is full. Spill whichever
            #     of {this interval, furthest-ending active interval}
            #     lives longer. ---
            spill_iv, spill_reg = active[-1]  # furthest end
            if spill_iv.end > iv.end:
                # The active one outlives `iv` — evict it, give its
                # register to `iv`.
                result.spilled.add(spill_iv.vreg)
                result.assignment.pop(spill_iv.vreg, None)
                result.assignment[iv.vreg] = spill_reg
                active.pop()
                active.append((iv, spill_reg))
            else:
                # `iv` lives at least as long — spill `iv` itself.
                result.spilled.add(iv.vreg)
        else:
            # --- Allocate a free register. ---
            reg = free_regs.pop(0)
            result.assignment[iv.vreg] = reg
            active.append((iv, reg))

        # Keep `active` sorted by end point (deterministic tie-break).
        active.sort(key=lambda pair: (pair[0].end, pair[0].vreg))

    return result


# ============================================================================
# Liveness analysis (v2.4 item 15, slice 2)
# ============================================================================
def compute_live_intervals(fn: ti.TileFn) -> list[LiveInterval]:
    """Compute conservative live intervals for every TileValue in a
    tile-IR function — the liveness half of register allocation,
    producing the input `linear_scan` consumes.

    All ops across all blocks are flattened into one linear
    numbering (op 0, 1, 2, ...). Each value's interval spans
    [min, max] over every op index where it appears as a result OR
    an operand. Function params and every block's params are treated
    as live from index 0 (defined at entry — they hold inputs).

    CONSERVATIVE LINEAR APPROXIMATION. For a straight-line kernel
    body this is exact. For control flow (branches, loops) it
    over-approximates: a value touched in two blocks gets an
    interval spanning the gap between them, so a register is never
    freed too early — safe, just less tight. Taking min/max over
    BOTH defs and uses also makes a loop back-edge use (which can
    precede the linear def index) widen the interval rather than
    produce a malformed start>end. Precise per-block dataflow
    liveness is a later item-15 slice if branchy kernels need
    tighter allocation; Helix tile-IR kernel bodies are mostly
    straight-line, so the approximation costs little today.

    Returns the intervals sorted by `vreg` (deterministic — required
    for the codegen-determinism the Helix test suite pins).
    """
    appearances: dict[int, list[int]] = {}

    def _touch(value_id: int, idx: int) -> None:
        appearances.setdefault(value_id, []).append(idx)

    # Function params + every block's params hold inputs — live from
    # the entry instruction (index 0).
    for p in fn.params:
        _touch(p.id, 0)
    for blk in fn.blocks:
        for p in blk.params:
            _touch(p.id, 0)

    # Flatten every op across every block into one linear numbering.
    idx = 0
    for blk in fn.blocks:
        for op in blk.ops:
            for v in op.results:
                _touch(v.id, idx)
            for v in op.operands:
                _touch(v.id, idx)
            idx += 1

    intervals = [
        LiveInterval(vreg=vid, start=min(idxs), end=max(idxs))
        for vid, idxs in appearances.items()
    ]
    intervals.sort(key=lambda iv: iv.vreg)
    return intervals


def allocate_fn(fn: ti.TileFn, num_registers: int) -> RegAllocResult:
    """v2.4 item 15 — single-register-class register allocation for one
    tile-IR function: liveness analysis followed by linear-scan.

    Convenience composition of `compute_live_intervals` +
    `linear_scan`, for a backend (or test) with a single uniform
    register file. Backends with multiple register classes (int vs
    float vs predicate) use `allocate_by_class` instead.
    """
    return linear_scan(compute_live_intervals(fn), num_registers)


# ============================================================================
# Multi-class allocation (v2.4 item 15, slice 3)
# ============================================================================
class RegAssignment(NamedTuple):
    """A vreg's multi-class register placement: which register file
    (`reg_class`) and which index within it.

    v2.5 polish (item-15 type-design audit Finding 3): the
    `MultiClassResult.assignment` payload was a bare `tuple[str, int]`
    — positionally ambiguous, forcing callers to index `[0]`/`[1]`.
    A NamedTuple is a zero-cost, backward-compatible upgrade: it IS
    still a tuple (so `== ("%r", 3)` and `[0]`/`[1]` indexing keep
    working) but now also exposes `.reg_class` / `.index`.
    """
    reg_class: str
    index: int


@dataclass(frozen=True)
class MultiClassResult:
    """Outcome of a multi-register-class allocation pass.

    `assignment` — vreg -> RegAssignment(reg_class, index). Two vregs
                   in DIFFERENT classes may share a register index —
                   they name distinct physical files (e.g. PTX `%r0`
                   vs `%f0`), so that is correct, not a collision.
    `spilled`    — vregs that did not fit their class's register file.
    `skipped`    — vregs an optional `allocate_by_class(skip=...)`
                   predicate excluded from register allocation. A real
                   kernel function mixes scalar values (one register
                   each — register-allocated) with tile/tensor values
                   (memory-resident — held across many registers or in
                   shared memory). The v2.5 emitter-wiring caller
                   passes a `skip` predicate so the allocator runs over
                   a real kernel without the per-backend classifier
                   raising on the first non-scalar value. A skipped
                   vreg is neither assigned nor spilled — the emitter
                   places it by its own memory-resident mechanism.
    `per_class`  — the underlying single-class RegAllocResult for each
                   class key, so a backend can read each class's
                   `register_high_water` to size its `.reg` decls.

    Invariant: `assignment.keys()`, `spilled`, and `skipped` are
    pairwise disjoint and together cover every vreg with a live
    interval exactly once.

    v2.5 polish (item-15 type-design): frozen, parity with
    `RegAllocResult` — see that class's note.
    """
    assignment: dict[int, RegAssignment] = field(default_factory=dict)
    spilled: set[int] = field(default_factory=set)
    skipped: set[int] = field(default_factory=set)
    per_class: dict[str, RegAllocResult] = field(default_factory=dict)

    @property
    def spill_count(self) -> int:
        return len(self.spilled)


def _value_map(fn: ti.TileFn) -> dict[int, ti.TileValue]:
    """Build a vreg-id -> TileValue map over every value a function
    mentions (params, block params, op results + operands)."""
    vmap: dict[int, ti.TileValue] = {}
    for p in fn.params:
        vmap[p.id] = p
    for blk in fn.blocks:
        for p in blk.params:
            vmap[p.id] = p
        for op in blk.ops:
            for v in op.results:
                vmap[v.id] = v
            for v in op.operands:
                vmap[v.id] = v
    return vmap


def allocate_by_class(
    fn: ti.TileFn,
    classify: Callable[[ti.TileValue], str],
    class_pools: dict[str, int],
    skip: Callable[[ti.TileValue], bool] | None = None,
) -> MultiClassResult:
    """v2.4 item 15 (slice 3) — multi-register-class allocation.

    Real backends have several register files that values cannot
    share across: PTX has `%r` (b32 int) / `%rd` (b64) / `%f` (f32) /
    `%p` (predicate); AMDGCN has VGPR vs SGPR. A b32-int value and an
    f32 value are simultaneously live without contending — they live
    in different files. Running one linear-scan over a single pool
    would wrongly serialize them.

    This runs liveness once, partitions the live intervals by
    register class (via `classify` on each value), then runs an
    independent `linear_scan` per class against that class's pool
    size. Per-class register indices are namespaced by the class key
    in the merged `assignment`, so equal indices in different classes
    are not collisions.

    Args:
        fn: the tile-IR kernel function.
        classify: maps a TileValue to its register-class key. The
            per-backend dtype -> class mapping (slice 4+).
        class_pools: class key -> register-file size for that class.
        skip: optional predicate marking a value as NOT register-
            allocated. v2.5 item-15 emitter-wiring slice: a real
            kernel function mixes scalar values (one register each)
            with tile/tensor values (memory-resident — many registers
            or shared memory). The per-backend classifier
            (`ptx_register_class` / `rocm_register_class`) raises on a
            non-scalar value by design, so `allocate_by_class` cannot
            be pointed at a real kernel without first excluding those
            values. A caller passes `skip` (e.g.
            `lambda v: not isinstance(v.ty, tir.TIRScalar)`) to drop
            them: a skipped vreg lands in `MultiClassResult.skipped`,
            is neither assigned nor spilled, and `classify` is never
            called on it. When `skip` is None every value is
            classified (the slice-3 behaviour).

    Raises:
        ValueError: if `class_pools` is empty — a backend must supply
            at least one register-class pool size. An empty table is
            a backend-configuration error; surfaced once, up front,
            rather than as a vacuous empty allocation (which an empty
            kernel would otherwise produce, hiding the misconfig).
        ValueError: if `classify` returns a key absent from
            `class_pools` — a backend that forgot to size a class
            fails loudly here, not with a silent mis-allocation.
    """
    if not class_pools:
        raise ValueError(
            "allocate_by_class: class_pools is empty — a backend must "
            "supply at least one register-class pool size. An empty "
            "pool table is a backend-configuration error, surfaced "
            "here rather than as a vacuous empty allocation."
        )
    vmap = _value_map(fn)
    intervals = compute_live_intervals(fn)

    result = MultiClassResult()
    # Partition intervals by register class.
    by_class: dict[str, list[LiveInterval]] = {}
    for iv in intervals:
        value = vmap.get(iv.vreg)
        if value is None:
            # compute_live_intervals only produces vregs drawn from
            # the same walk _value_map does — this is unreachable, but
            # raise rather than silently drop a value from allocation.
            raise ValueError(
                f"allocate_by_class: vreg {iv.vreg} has a live "
                f"interval but no TileValue — liveness/value-map drift"
            )
        if skip is not None and skip(value):
            # Memory-resident (tile/tensor) value — excluded from
            # register allocation; classify is deliberately not called.
            result.skipped.add(iv.vreg)
            continue
        cls = classify(value)
        if cls not in class_pools:
            raise ValueError(
                f"allocate_by_class: classify() returned register "
                f"class {cls!r} for vreg {iv.vreg}, but class_pools "
                f"has no pool size for it. Known classes: "
                f"{sorted(class_pools)}."
            )
        by_class.setdefault(cls, []).append(iv)

    # Iterate class keys in sorted order for deterministic merge.
    for cls in sorted(by_class):
        class_result = linear_scan(by_class[cls], class_pools[cls])
        result.per_class[cls] = class_result
        for vreg, reg_idx in class_result.assignment.items():
            result.assignment[vreg] = RegAssignment(
                reg_class=cls, index=reg_idx)
        # `.update()` not `|=`: with the frozen dataclass `spilled |=`
        # desugars to `spilled = spilled.__ior__(...)` — an attribute
        # rebind that raises FrozenInstanceError. In-place update of
        # the set's CONTENTS is allowed and is what is intended.
        result.spilled.update(class_result.spilled)
    return result
