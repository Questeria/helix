"""Tests for the Stage 33 self-host gate wrapper."""

from __future__ import annotations

from pathlib import Path

from scripts import stage33_selfhost_gate


def test_stage33_selfhost_gate_runs_cascade_then_validator(monkeypatch, tmp_path):
    calls = []

    def fake_run_command(cmd):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(stage33_selfhost_gate, "run_command", fake_run_command)

    rc = stage33_selfhost_gate.gate(
        generations=3,
        json_out=tmp_path / "report.json",
        prefix="/tmp/test_gate",
        keep=False,
        expect_stable_sha="abc",
    )

    assert rc == 0
    assert calls[0][1:] == [
        "scripts/selfhost_cascade.py",
        "--generations",
        "3",
        "--prefix",
        "/tmp/test_gate",
        "--json-out",
        str(tmp_path / "report.json"),
    ]
    assert calls[1][1:] == [
        "scripts/selfhost_cascade_validate.py",
        str(tmp_path / "report.json"),
        "--min-generations",
        "3",
        "--expect-stable-sha",
        "abc",
    ]


def test_stage33_selfhost_gate_stops_when_cascade_fails(monkeypatch, tmp_path):
    calls = []

    def fake_run_command(cmd):
        calls.append(cmd)
        return 9

    monkeypatch.setattr(stage33_selfhost_gate, "run_command", fake_run_command)

    rc = stage33_selfhost_gate.gate(
        generations=3,
        json_out=tmp_path / "report.json",
        prefix="/tmp/test_gate",
        keep=False,
        expect_stable_sha=None,
    )

    assert rc == 9
    assert len(calls) == 1


def test_stage33_selfhost_gate_rejects_too_few_generations(capsys):
    rc = stage33_selfhost_gate.main(["--generations", "1"])
    captured = capsys.readouterr()

    assert rc == 2
    assert "--generations must be at least 2" in captured.err


def test_stage33_selfhost_gate_default_report_path_is_repo_local():
    assert isinstance(stage33_selfhost_gate.DEFAULT_REPORT, Path)
    assert ".stage33-logs" in str(stage33_selfhost_gate.DEFAULT_REPORT)


def test_stage33_selfhost_gate_default_prefix_matches_canonical_cascade():
    args = stage33_selfhost_gate.parse_args([])

    assert args.prefix == "/tmp/helix_cascade"
