"""Archive reproducibility checks for the Stage 0 hex0 fixtures."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _win_to_wsl(path: Path) -> str:
    p = str(path.resolve()).replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:]
        if not rest.startswith("/"):
            rest = "/" + rest
        return f"/mnt/{drive}{rest}"
    return p


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_stage0_hex0_archive_fixtures_are_lf_and_shell_gate_passes(tmp_path):
    if not (ROOT / ".git").exists():
        pytest.skip("requires a git checkout")

    tree = _git("write-tree")
    archive_path = tmp_path / "candidate.tar"
    with archive_path.open("wb") as out:
        subprocess.run(
            ["git", "-C", str(ROOT), "archive", "--format=tar", tree],
            check=True,
            stdout=out,
        )

    archive_root = tmp_path / "tree"
    archive_root.mkdir()
    with tarfile.open(archive_path) as archive:
        archive.extractall(archive_root, filter="data")

    fixture_dir = archive_root / "stage0" / "hex0" / "test"
    for name in [
        "01-hello.expected",
        "02-comments-ws.expected",
        "03-empty.expected",
        "01-hello.hex0",
        "02-comments-ws.hex0",
        "03-empty.hex0",
    ]:
        data = (fixture_dir / name).read_bytes()
        assert b"\r" not in data, name

    stage0_dir = archive_root / "stage0" / "hex0"
    if os.name == "nt":
        quoted = shlex.quote(_win_to_wsl(stage0_dir))
        cmd = ["wsl", "--", "bash", "-lc", f"cd {quoted} && bash run_tests.sh"]
    else:
        cmd = ["bash", "-lc", "bash run_tests.sh"]
    result = subprocess.run(
        cmd,
        cwd=stage0_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0, result.stdout
