"""Tests for the Stage 31 validation runner."""

from __future__ import annotations

import os
import shlex
import subprocess

from scripts import stage31_validate


def test_stage31_validate_default_shards_are_capped(monkeypatch):
    monkeypatch.setattr(stage31_validate.os, "cpu_count", lambda: 128)
    assert stage31_validate.default_shards() == stage31_validate.MAX_SHARDS


def test_stage31_validate_default_shards_fallback_when_cpu_unknown(monkeypatch):
    monkeypatch.setattr(stage31_validate.os, "cpu_count", lambda: None)
    assert stage31_validate.default_shards() == 4


def test_stage31_validate_rejects_excessive_manual_shards(capsys):
    rc = stage31_validate.main([
        "--mode",
        "full",
        "--shards",
        str(stage31_validate.MAX_SHARDS + 1),
        "--skip-snapshot",
    ])
    captured = capsys.readouterr()
    assert rc == 2
    assert f"--shards must be <= {stage31_validate.MAX_SHARDS}" in captured.err
    assert "pytest" not in captured.out


def _bash_root() -> str:
    root = stage31_validate.ROOT
    if os.name == "nt":
        drive = root.drive.rstrip(":").lower()
        rest = "/".join(root.parts[1:])
        return f"/mnt/{drive}/{rest}"
    return str(root)


def _run_all_tests_with_shards(shards: str) -> subprocess.CompletedProcess[str]:
    command = (
        f"cd {shlex.quote(_bash_root())} && "
        f"HELIX_TEST_SHARDS={shlex.quote(shards)} "
        "bash scripts/run_all_tests.sh"
    )
    return subprocess.run(
        ["bash", "-lc", command],
        cwd=stage31_validate.ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )


def test_run_all_tests_rejects_excessive_manual_shards_before_gates():
    proc = _run_all_tests_with_shards(str(stage31_validate.MAX_SHARDS + 1))
    assert proc.returncode == 2
    assert f"HELIX_TEST_SHARDS must be an integer from 1 to {stage31_validate.MAX_SHARDS}" in proc.stderr
    assert "stage0/hex0" not in proc.stdout
    assert "pytest (stage31 sharded gate)" not in proc.stdout


def test_run_all_tests_rejects_zero_padded_excessive_manual_shards_before_gates():
    proc = _run_all_tests_with_shards(f"0{stage31_validate.MAX_SHARDS + 1}")
    assert proc.returncode == 2
    assert f"HELIX_TEST_SHARDS must be an integer from 1 to {stage31_validate.MAX_SHARDS}" in proc.stderr
    assert "value too great for base" not in proc.stderr
    assert "stage0/hex0" not in proc.stdout
    assert "pytest (stage31 sharded gate)" not in proc.stdout
