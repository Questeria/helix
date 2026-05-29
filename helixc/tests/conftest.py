"""Pytest fixtures for the Helix test suite.

WSL keepalive (2026-05-29): the bootstrap test harness runs every compiled
ELF via `wsl -e bash`. WSL shuts the VM down ~60s after the last command
(vmIdleTimeout); when a harness call then cold-starts the VM it can race and
return WSL_E_USER_NOT_FOUND (rc 1) or a spurious SIGILL (rc 132) on the first
calls. That surfaced as *flaky* parity / self-host failures on tiny programs
(different cases failing each run) — never a real codegen bug, but enough to
turn a green suite red and waste re-run cycles. A one-shot warmup is not
enough because the VM can idle-shutdown again mid-run. So we hold a persistent
background `wsl ... sleep` for the whole test session, keeping the VM warm so
harness `wsl` calls never cold-start. Everything is best-effort (wrapped in
try/except): if WSL is unavailable the suite behaves exactly as before.
"""
import subprocess

import pytest


@pytest.fixture(scope="session", autouse=True)
def _keep_wsl_warm():
    keeper = None
    try:
        # Persistent sleeper holds the WSL VM up for the whole session.
        keeper = subprocess.Popen(
            ["wsl", "-e", "bash", "-c", "sleep 86400"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Block briefly so the VM is actually up before the first test runs.
        subprocess.run(
            ["wsl", "-e", "bash", "-c", "true"],
            capture_output=True,
            timeout=60,
        )
    except Exception:
        pass
    yield
    if keeper is not None:
        try:
            keeper.terminate()
        except Exception:
            pass
