"""v2.x re-audit R1 regression tests — helixc/examples/run.py demo
runner success-exit-code logic (RT 5-clean-gate HIGH).

The fresh pre-v3.0 re-audit found `_run_one` checked success as
`code == 0`, but Helix Phase-0 demos signal success via distinctive
NON-zero exit codes (metacircular 40, symbolic 77, sat 1, the dogfood
demos 42, graddescent 43-44) — only mandelbrot, a stdout-rendering
demo, exits 0. So 18 of 19 demos were scored failed-on-success and
the demo-runner CI exit code was permanently, falsely red. These
tests pin the fix: a per-demo `_DEMO_EXIT_OK` table + a membership
check.

License: Apache 2.0
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from helixc.examples import run


def test_demo_exit_ok_covers_every_demo():
    """_DEMO_EXIT_OK and DEMOS must stay in lockstep — every demo
    needs a documented success exit code, none orphaned. (run.py also
    asserts this at module load; this re-pins it as a test.)"""
    assert set(run._DEMO_EXIT_OK) == set(run.DEMOS), (
        f"_DEMO_EXIT_OK {sorted(run._DEMO_EXIT_OK)} != "
        f"DEMOS {sorted(run.DEMOS)}")


def test_demo_success_codes_are_non_zero_except_mandelbrot():
    """v2.x re-audit R1 (RT HIGH): Helix Phase-0 demos signal success
    via NON-zero exit codes; only mandelbrot (stdout-rendering) exits
    0. Pins that the success codes are not all 0 — the prior
    `code == 0` check treated 0 as the sole success and reported 18 of
    19 demos failed-on-success."""
    assert run._DEMO_EXIT_OK["mandelbrot"] == (0,)
    for key, codes in run._DEMO_EXIT_OK.items():
        assert codes, f"demo {key!r} has an empty exit-code tuple"
        assert all(isinstance(c, int) for c in codes), key
        if key == "mandelbrot":
            continue
        assert 0 not in codes, (
            f"demo {key!r}: exit 0 is not its success code — Helix "
            f"Phase-0 demos succeed with non-zero codes")


def test_run_one_checks_documented_exit_code(monkeypatch):
    """v2.x re-audit R1 (RT HIGH): _run_one returns True iff the
    process exit code is in the demo's _DEMO_EXIT_OK tuple — NOT
    `code == 0`. _build_and_run is monkeypatched so no compile/WSL
    run happens."""
    # A correct run: metacircular exits 40 -> success.
    monkeypatch.setattr(run, "_build_and_run",
                        lambda *a, **k: ("", "", 40))
    assert run._run_one("metacircular") is True
    # Exit 0 is NOT success for metacircular — this was the bug.
    monkeypatch.setattr(run, "_build_and_run",
                        lambda *a, **k: ("", "", 0))
    assert run._run_one("metacircular") is False
    # A wrong non-zero code is also a failure.
    monkeypatch.setattr(run, "_build_and_run",
                        lambda *a, **k: ("", "", 39))
    assert run._run_one("metacircular") is False
    # graddescent accepts a RANGE — 43 or 44 both succeed.
    monkeypatch.setattr(run, "_build_and_run",
                        lambda *a, **k: ("", "", 44))
    assert run._run_one("graddescent") is True
    monkeypatch.setattr(run, "_build_and_run",
                        lambda *a, **k: ("", "", 43))
    assert run._run_one("graddescent") is True
    # mandelbrot is the one demo whose success code IS 0.
    monkeypatch.setattr(run, "_build_and_run",
                        lambda *a, **k: ("rendered\n", "", 0))
    assert run._run_one("mandelbrot") is True


def test_run_one_unknown_demo_returns_false():
    """An unknown demo key is a failure, not a crash."""
    assert run._run_one("no_such_demo") is False


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
