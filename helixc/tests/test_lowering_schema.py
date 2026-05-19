"""Tests for helixc.backend._lowering_schema — v2.3 polish item 2.

Shared type-design module for backend lowering tables. Closes
cross-backend TypedDict + Literal status + Protocol observations
from the v2.1 + v2.2 5-clean-gate type-design auditors.
"""
from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from helixc.backend._lowering_schema import (
    BackendEmitter,
    HELIX_STUB_OPERANDS_TOKEN,
    HELIX_STUB_TOKEN,
    LoweringStatus,
    OpLowering,
    VALID_STATUSES,
    is_loud_stub_status,
)
from helixc.backend.metal import MslEmitter
from helixc.backend.ptx import PtxEmitter
from helixc.backend.rocm import HipEmitter
from helixc.backend.webgpu import WgslEmitter


def test_v23_valid_statuses_is_frozenset():
    """VALID_STATUSES is an immutable frozenset of exactly the
    canonical status values."""
    assert isinstance(VALID_STATUSES, frozenset)
    assert VALID_STATUSES == {"supported", "stub", "deferred", "skipped"}


def test_v23_valid_statuses_matches_literal_type_args():
    """The runtime frozenset and the static Literal must stay
    aligned — adding a new status to one requires adding to the
    other. This test catches drift."""
    from typing import get_args
    literal_args = set(get_args(LoweringStatus))
    assert literal_args == set(VALID_STATUSES)


def test_v23_oplowering_typeddict_shape():
    """OpLowering accepts the two required keys; mypy enforces
    types statically. Here we exercise the runtime shape."""
    entry: OpLowering = {"lowering": "test", "status": "supported"}
    assert entry["lowering"] == "test"
    assert entry["status"] == "supported"


def test_v23_helix_stub_token_substring():
    """HELIX_STUB_TOKEN is the substring all 4 backends emit on
    stub/deferred status — downstream tooling can use it as a single
    string-match for 'non-functional kernel' detection."""
    assert HELIX_STUB_TOKEN == "HELIX-STUB"
    assert isinstance(HELIX_STUB_TOKEN, str)


def test_v23_helix_stub_operands_token_substring():
    """HELIX_STUB_OPERANDS_TOKEN is the marker for status="supported"
    branches where operand-binding is not yet wired (RegAlloc is
    v2.3 item 15)."""
    assert HELIX_STUB_OPERANDS_TOKEN == "HELIX-STUB-OPERANDS"


def test_v23_backend_emitter_protocol_is_runtime_checkable():
    """BackendEmitter is `runtime_checkable` so `isinstance` works
    at runtime — useful for plugin registration + factory dispatch."""
    # Each of the 4 emitters satisfies the Protocol structurally
    # (they all have `emit_module(mod) -> str`).
    assert isinstance(PtxEmitter(), BackendEmitter)
    assert isinstance(HipEmitter(), BackendEmitter)
    assert isinstance(MslEmitter(), BackendEmitter)
    assert isinstance(WgslEmitter(), BackendEmitter)


def test_v23_backend_emitter_protocol_rejects_non_emitter():
    """`isinstance` against BackendEmitter rejects objects lacking
    the `emit_module` method."""
    class NotAnEmitter:
        def emit_something_else(self, x): return ""

    assert not isinstance(NotAnEmitter(), BackendEmitter)
    assert not isinstance("a string", BackendEmitter)
    assert not isinstance(42, BackendEmitter)


def test_v23_is_loud_stub_status_categorization():
    """`is_loud_stub_status` partitions the 4 status values:
    stub/deferred/skipped → True (emit directive)
    supported → False (real codegen)."""
    assert is_loud_stub_status("stub") is True
    assert is_loud_stub_status("deferred") is True
    assert is_loud_stub_status("skipped") is True
    assert is_loud_stub_status("supported") is False


def test_v23_is_loud_stub_status_rejects_invalid():
    """Unknown status string returns False (does not raise — the
    caller's `entry["status"]` reads through the table, so a
    drift-detector catches the unknown value at module load via
    VALID_STATUSES membership, not here)."""
    assert is_loud_stub_status("Supported") is False  # capital-S typo
    assert is_loud_stub_status("") is False
    assert is_loud_stub_status("unknown") is False


def test_v23_all_4_backends_use_valid_statuses():
    """Cross-backend invariant: every entry's status in every
    backend's _OP_LOWERING table must be a member of VALID_STATUSES.
    This test catches the typo-class bug VALID_STATUSES was built
    to prevent (e.g., `"Supported"` slipping in)."""
    from helixc.backend.metal import METAL_OP_LOWERING
    from helixc.backend.ptx import PTX_OP_LOWERING
    from helixc.backend.rocm import ROCM_OP_LOWERING
    from helixc.backend.webgpu import WEBGPU_OP_LOWERING

    for table_name, table in [
        ("PTX_OP_LOWERING", PTX_OP_LOWERING),
        ("ROCM_OP_LOWERING", ROCM_OP_LOWERING),
        ("METAL_OP_LOWERING", METAL_OP_LOWERING),
        ("WEBGPU_OP_LOWERING", WEBGPU_OP_LOWERING),
    ]:
        for kind, entry in table.items():
            assert entry["status"] in VALID_STATUSES, (
                f"{table_name}[{kind.name}].status="
                f"{entry['status']!r} not in {VALID_STATUSES}"
            )
