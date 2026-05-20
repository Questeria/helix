"""Tests for helixc.backend.regalloc — v2.4 item 15 (slice 1).

Backend-agnostic linear-scan register allocation core.
"""
from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from helixc.backend.regalloc import (
    LiveInterval,
    RegAllocResult,
    linear_scan,
)


def test_v24_empty_intervals_empty_result():
    """v2.4 item 15 — no intervals → empty assignment, no spills,
    zero high-water."""
    r = linear_scan([], num_registers=4)
    assert r.assignment == {}
    assert r.spilled == set()
    assert r.num_registers == 4
    assert r.register_high_water == 0
    assert r.spill_count == 0


def test_v24_num_registers_must_be_positive():
    """v2.4 item 15 — num_registers < 1 is rejected loudly."""
    with pytest.raises(ValueError, match="num_registers must be >= 1"):
        linear_scan([LiveInterval(0, 0, 1)], num_registers=0)
    with pytest.raises(ValueError, match="num_registers must be >= 1"):
        linear_scan([], num_registers=-3)


def test_v24_malformed_interval_rejected():
    """v2.4 item 15 — a LiveInterval with start > end is malformed
    and raises (not silently mis-allocated)."""
    with pytest.raises(ValueError, match="malformed"):
        linear_scan([LiveInterval(0, 5, 2)], num_registers=4)


def test_v24_duplicate_vreg_rejected():
    """v2.4 item 15 — each value must have exactly one live interval;
    a duplicate vreg raises."""
    with pytest.raises(ValueError, match="duplicate vreg"):
        linear_scan(
            [LiveInterval(7, 0, 2), LiveInterval(7, 3, 5)],
            num_registers=4,
        )


def test_v24_non_overlapping_intervals_reuse_one_register():
    """v2.4 item 15 — two values whose live ranges do not overlap
    share a single physical register (the whole point of RegAlloc:
    register reuse). high-water = 1 even with a larger pool."""
    r = linear_scan(
        [LiveInterval(0, 0, 2), LiveInterval(1, 3, 5)],
        num_registers=4,
    )
    assert r.spilled == set()
    assert r.assignment[0] == r.assignment[1]  # same register reused
    assert r.register_high_water == 1


def test_v24_overlapping_intervals_get_distinct_registers():
    """v2.4 item 15 — values live at the same time get distinct
    registers; with enough registers nothing spills."""
    r = linear_scan(
        [LiveInterval(0, 0, 5), LiveInterval(1, 1, 5),
         LiveInterval(2, 2, 5)],
        num_registers=3,
    )
    assert r.spilled == set()
    regs = {r.assignment[v] for v in (0, 1, 2)}
    assert len(regs) == 3  # all distinct
    assert r.register_high_water == 3


def test_v24_overflow_spills_excess_values():
    """v2.4 item 15 — when more values are simultaneously live than
    the register file holds, the excess spill. 3 overlapping values,
    2 registers → exactly 1 spill."""
    r = linear_scan(
        [LiveInterval(0, 0, 5), LiveInterval(1, 1, 5),
         LiveInterval(2, 2, 5)],
        num_registers=2,
    )
    assert r.spill_count == 1
    assert len(r.assignment) == 2
    # assignment and spilled partition the input exactly.
    assert set(r.assignment) | r.spilled == {0, 1, 2}
    assert set(r.assignment) & r.spilled == set()


def test_v24_spill_heuristic_evicts_longer_lived_interval():
    """v2.4 item 15 — Poletto-Sarkar spill heuristic: when the file
    is full, spill whichever lives LONGER. A short-lived value
    arriving while a long-lived value holds the only register must
    EVICT the long-lived one — the short value gets the register,
    the long value spills."""
    r = linear_scan(
        [LiveInterval(0, 0, 100),  # long-lived
         LiveInterval(1, 1, 2)],   # short-lived, arrives during 0
        num_registers=1,
    )
    # The short-lived value 1 wins the register; the long-lived 0
    # is the one spilled.
    assert r.assignment == {1: 0}
    assert r.spilled == {0}


def test_v24_spill_heuristic_keeps_register_when_new_interval_longer():
    """v2.4 item 15 — converse: if the newly-arriving interval lives
    at least as long as the furthest active one, the new interval
    itself spills (no pointless eviction churn)."""
    r = linear_scan(
        [LiveInterval(0, 0, 3),    # shorter
         LiveInterval(1, 1, 100)],  # longer, arrives during 0
        num_registers=1,
    )
    # Value 0 keeps the register; the longer-lived 1 spills.
    assert r.assignment == {0: 0}
    assert r.spilled == {1}


def test_v24_linear_scan_is_deterministic():
    """v2.4 item 15 — two runs on the same input produce byte-
    identical results. Required for the codegen-determinism the
    Helix test suite pins (a non-deterministic allocator would make
    emitted kernels differ run-to-run)."""
    intervals = [
        LiveInterval(3, 0, 8), LiveInterval(1, 0, 2),
        LiveInterval(4, 2, 9), LiveInterval(0, 1, 4),
        LiveInterval(2, 3, 6),
    ]
    a = linear_scan(intervals, num_registers=2)
    b = linear_scan(intervals, num_registers=2)
    assert a.assignment == b.assignment
    assert a.spilled == b.spilled


def test_v24_register_high_water_minimal():
    """v2.4 item 15 — the allocator hands out the lowest free
    register index, so register_high_water is the minimum register
    count a backend must declare. Sequential non-overlapping values
    in a 16-register pool still report high-water 1."""
    intervals = [LiveInterval(i, i * 2, i * 2 + 1) for i in range(8)]
    r = linear_scan(intervals, num_registers=16)
    assert r.spilled == set()
    assert r.register_high_water == 1  # full reuse — never 2+


def test_v24_regalloc_result_partition_invariant():
    """v2.4 item 15 — assignment.keys() and spilled always partition
    the input vregs: disjoint, and together exhaustive."""
    intervals = [LiveInterval(i, 0, 10) for i in range(10)]
    r = linear_scan(intervals, num_registers=4)
    all_vregs = {iv.vreg for iv in intervals}
    assert set(r.assignment) | r.spilled == all_vregs
    assert set(r.assignment) & r.spilled == set()
    assert len(r.assignment) == 4
    assert r.spill_count == 6
