"""Unit tests for self-host cascade reporting helpers."""

from __future__ import annotations

import json

from scripts import selfhost_cascade


def test_selfhost_cascade_report_marks_identical_generations_stable():
    report = selfhost_cascade.cascade_report(
        generations_requested=3,
        seed_size=100,
        seed_sha="seed",
        generations=[
            {"exit_low_byte": 139, "size": 200, "sha": "same"},
            {"exit_low_byte": 139, "size": 200, "sha": "same"},
            {"exit_low_byte": 139, "size": 200, "sha": "same"},
        ],
        smoke=[{"name": "literal", "expected_exit": 42, "actual_exit": 42}],
    )

    assert report["schema"] == selfhost_cascade.CASCADE_REPORT_SCHEMA
    assert report["stable"] is True
    assert report["stable_sha256"] == "same"
    assert report["stable_size"] == 200
    assert report["self_host_generations"][0]["generation"] == 2
    assert report["smoke"] == [
        {"name": "literal", "expected_exit": 42, "actual_exit": 42}
    ]


def test_selfhost_cascade_report_marks_drift_unstable():
    report = selfhost_cascade.cascade_report(
        generations_requested=2,
        seed_size=100,
        seed_sha="seed",
        generations=[
            {"exit_low_byte": 139, "size": 200, "sha": "first"},
            {"exit_low_byte": 139, "size": 201, "sha": "second"},
        ],
    )

    assert report["stable"] is False
    assert report["stable_sha256"] is None
    assert report["stable_size"] is None


def test_selfhost_cascade_write_report_creates_parent_dirs(tmp_path):
    out = tmp_path / "nested" / "cascade.json"
    report = {"schema": selfhost_cascade.CASCADE_REPORT_SCHEMA}

    selfhost_cascade.write_report(str(out), report)

    assert json.loads(out.read_text(encoding="utf-8")) == report


# ============================================================================
# v2.x re-audit R3 (RT-M3): `run_smoke` previously recorded a fabricated
# `actual_exit` (a copy of `expected`), making selfhost_cascade_validate's
# `actual_exit == 42` cross-check tautological. The exit code is now
# parsed from the run's trailing `echo exit=$?` line by
# `_parse_smoke_exit_code` — these tests pin that parser, including the
# whole-line match that also closes the RT-L1 fragile-substring LOW
# (`exit=420` must not be mistaken for `exit=42`).
# ============================================================================
def test_parse_smoke_exit_code_basic():
    """The trailing `echo exit=$?` line is parsed to its integer."""
    assert selfhost_cascade._parse_smoke_exit_code("exit=42\n") == 42
    assert selfhost_cascade._parse_smoke_exit_code("exit=0\n") == 0


def test_parse_smoke_exit_code_with_preceding_output():
    """Program stdout before the echo line does not confuse the parse."""
    assert selfhost_cascade._parse_smoke_exit_code(
        "hello\nworld\nexit=7\n") == 7


def test_parse_smoke_exit_code_no_substring_false_match():
    """RT-M3 / RT-L1: a whole-line match — `exit=420` is parsed as 420,
    not mistaken for `exit=42` the way the prior `f"exit={expected}" in
    stdout` substring test would have."""
    assert selfhost_cascade._parse_smoke_exit_code("exit=420\n") == 420
    assert selfhost_cascade._parse_smoke_exit_code("exit=420\n") != 42


def test_parse_smoke_exit_code_takes_last_line():
    """A binary that prints its own `exit=` text cannot shadow the
    appended echo — the LAST `exit=N` line wins."""
    assert selfhost_cascade._parse_smoke_exit_code(
        "exit=1\nexit=42\n") == 42


def test_parse_smoke_exit_code_missing_line_is_none():
    """No `exit=N` line → None, so run_smoke's guard (`actual_exit !=
    expected`) fails the smoke case loudly instead of fabricating a
    passing `actual_exit`."""
    assert selfhost_cascade._parse_smoke_exit_code("garbage output\n") is None
    assert selfhost_cascade._parse_smoke_exit_code("") is None


def test_parse_smoke_exit_code_handles_crlf():
    """WSL stdout decoded with a trailing CR still parses."""
    assert selfhost_cascade._parse_smoke_exit_code("exit=42\r\n") == 42
