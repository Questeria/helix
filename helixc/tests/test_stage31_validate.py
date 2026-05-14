"""Tests for the Stage 31 validation runner."""

from __future__ import annotations

import os
import shlex
import subprocess
from types import SimpleNamespace

from scripts import stage31_validate


def test_stage31_validate_default_shards_are_capped(monkeypatch):
    monkeypatch.setattr(stage31_validate.os, "cpu_count", lambda: 128)
    assert stage31_validate.default_shards() == stage31_validate.MAX_SHARDS


def test_stage31_validate_default_shards_fallback_when_cpu_unknown(monkeypatch):
    monkeypatch.setattr(stage31_validate.os, "cpu_count", lambda: None)
    assert stage31_validate.default_shards() == 4


def test_stage31_validate_full_shards_non_codegen_suite(monkeypatch):
    calls = []
    retry_flags = []

    def fake_run_parallel(jobs, *, retry_failed_once=False):
        calls.extend(jobs)
        retry_flags.append(retry_failed_once)
        return 0

    monkeypatch.setattr(stage31_validate, "validation_env", lambda: {"TEST": "1"})
    monkeypatch.setattr(stage31_validate, "run_parallel", fake_run_parallel)

    rc = stage31_validate.full("python", shards=6)

    assert rc == 0
    assert retry_flags == [True]
    names = [name for name, _cmd, _env in calls]
    assert "pytest-no-codegen-shard-1-of-4" in names
    assert "pytest-no-codegen-shard-4-of-4" in names
    assert "pytest-codegen-shard-1-of-6" in names
    assert "pytest-codegen-shard-6-of-6" in names
    no_codegen_cmds = [
        cmd for name, cmd, _env in calls
        if name.startswith("pytest-no-codegen-shard")
    ]
    assert len(no_codegen_cmds) == stage31_validate.MAX_NO_CODEGEN_SHARDS
    assert all("scripts/pytest_shard.py" in cmd for cmd in no_codegen_cmds)
    assert all("--ignore=helixc/tests/test_codegen.py" in cmd for cmd in no_codegen_cmds)


def test_stage31_validate_can_disable_failed_shard_retry(monkeypatch):
    retry_flags = []

    def fake_run_parallel(jobs, *, retry_failed_once=False):
        retry_flags.append(retry_failed_once)
        return 0

    monkeypatch.setattr(stage31_validate, "validation_env", lambda: {})
    monkeypatch.setattr(stage31_validate, "run_parallel", fake_run_parallel)

    assert stage31_validate.full("python", shards=2, retry_failed_once=False) == 0
    assert retry_flags == [False]


def test_stage31_validate_retries_failed_parallel_shards_once(monkeypatch, capsys):
    jobs = [
        ("fast", ["python", "-c", "pass"], None),
        ("flaky", ["python", "-c", "pass"], {"ENV": "1"}),
    ]
    calls = []

    def fake_run_parallel_once(run_jobs):
        calls.append(run_jobs)
        if len(calls) == 1:
            return 1, [run_jobs[1]]
        return 0, []

    monkeypatch.setattr(stage31_validate, "run_parallel_once", fake_run_parallel_once)

    rc = stage31_validate.run_parallel(jobs, retry_failed_once=True)
    captured = capsys.readouterr()

    assert rc == 0
    assert [name for name, _cmd, _env in calls[1]] == ["flaky-retry1"]
    assert "failed shards recovered on retry: flaky" in captured.out


def test_stage31_validate_failed_retry_keeps_gate_red(monkeypatch, capsys):
    jobs = [("bad", ["python", "-c", "raise SystemExit(1)"], None)]
    calls = []

    def fake_run_parallel_once(run_jobs):
        calls.append(run_jobs)
        return 1, [run_jobs[0]]

    monkeypatch.setattr(stage31_validate, "run_parallel_once", fake_run_parallel_once)

    rc = stage31_validate.run_parallel(jobs, retry_failed_once=True)
    captured = capsys.readouterr()

    assert rc == 1
    assert len(calls) == 2
    assert "failed shards still failing after retry: bad-retry1" in captured.out


def test_stage31_validate_extracts_pytest_shard_summary(tmp_path):
    log = tmp_path / "pytest-shard.log"
    log.write_text(
        "$ python scripts/pytest_shard.py ...\n"
        "................................\n"
        "107 passed, 635 deselected in 705.14s (0:11:45)\n",
        encoding="utf-8",
    )

    assert stage31_validate.pytest_summary_from_log(log) == (
        705.14,
        "107 passed, 635 deselected in 705.14s (0:11:45)",
    )


def test_stage31_validate_ignores_logs_without_pytest_summary(tmp_path):
    log = tmp_path / "other.log"
    log.write_text("$ echo hello\nhello\n", encoding="utf-8")

    assert stage31_validate.pytest_summary_from_log(log) is None


def test_stage31_validate_prints_slowest_pytest_shards(tmp_path, capsys):
    slow = tmp_path / "slow.log"
    fast = tmp_path / "fast.log"
    slow.write_text("5 passed in 12.50s\n", encoding="utf-8")
    fast.write_text("5 passed in 1.25s\n", encoding="utf-8")

    stage31_validate.print_slowest_pytest_shards([
        ("slow", object(), object(), slow, ["python"], None),
        ("fast", object(), object(), fast, ["python"], None),
    ])
    captured = capsys.readouterr()

    assert "slowest pytest shards:" in captured.out
    assert "slow: 12.5s" in captured.out
    assert "fast: 1.2s" in captured.out


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


def test_snapshot_smoke_runs_modules_outside_repo_root(monkeypatch):
    calls = []

    def fake_run_logged(name, cmd, *, env=None, cwd=None):
        calls.append((name, cmd, env, cwd))
        return 0

    def fake_run(cmd, cwd=None, **_kwargs):
        calls.append(("snapshot-run", cmd, None, cwd))
        return SimpleNamespace(returncode=42)

    monkeypatch.setattr(stage31_validate, "run_logged", fake_run_logged)
    monkeypatch.setattr(stage31_validate.subprocess, "run", fake_run)

    rc = stage31_validate.snapshot_smoke("python")

    assert rc == 0
    module_calls = [call for call in calls if call[0].startswith("snapshot-")]
    check = next(call for call in module_calls if call[0] == "snapshot-check")
    compile_ = next(call for call in module_calls if call[0] == "snapshot-compile")
    assert check[3] != stage31_validate.ROOT
    assert compile_[3] != stage31_validate.ROOT
    assert check[3] == compile_[3]
    assert check[2]["PYTHONPATH"] == str(
        stage31_validate.ROOT / "HELIX_STAGE30_COMPILER_SNAPSHOT"
    )
