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

from dataclasses import dataclass, field

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


@dataclass
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
    """v2.4 item 15 — end-to-end register allocation for one tile-IR
    function: liveness analysis followed by linear-scan.

    Convenience composition of `compute_live_intervals` +
    `linear_scan`. Per-backend slices wrap this with a register-class
    model (dtype -> class + pool size) and thread the resulting
    `assignment` into operand emission.
    """
    return linear_scan(compute_live_intervals(fn), num_registers)
