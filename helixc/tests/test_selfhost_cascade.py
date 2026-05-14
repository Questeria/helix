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
