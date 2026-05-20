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
