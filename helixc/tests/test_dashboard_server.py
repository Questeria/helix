"""Tests for helixc.examples.dashboard_server — v2.2 polish item 7
(RT M1 from v2.1 5-clean-gate): query-string typo / invalid-int
must surface HTTP 400 with diagnostic body, not silently coerce to
defaults."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import io


def _make_handler(path: str):
    """Construct a Handler instance with stubbed I/O so we can drive
    do_GET in isolation. Bypasses the BaseHTTPRequestHandler __init__
    socket setup."""
    from helixc.examples.dashboard_server import Handler

    h = Handler.__new__(Handler)
    h.path = path
    h.client_address = ("127.0.0.1", 0)
    h.request_version = "HTTP/1.1"
    h.command = "GET"
    h.headers = {}
    h.requestline = f"GET {path} HTTP/1.1"
    h.wfile = io.BytesIO()
    h.rfile = io.BytesIO()
    h._headers_buffer = []
    return h


def _drive(handler) -> tuple[int, bytes]:
    """Drive do_GET and return (status_code, body)."""
    # send_response/send_header/end_headers/wfile.write all write to
    # wfile in BaseHTTPRequestHandler — but send_response also writes
    # the status line via self.wfile.write directly. Capture both.
    handler.do_GET()
    raw = handler.wfile.getvalue()
    # Status line is the first line, e.g. b"HTTP/1.1 400 Bad Request\r\n"
    first_line, _, rest = raw.partition(b"\r\n")
    parts = first_line.split(b" ", 2)
    status = int(parts[1]) if len(parts) >= 2 else 0
    # Body is after the blank line that ends headers.
    _, _, body = rest.partition(b"\r\n\r\n")
    return status, body


def test_item7_unknown_query_key_returns_400():
    """Typo'd query key (e.g. `kihd` instead of `kind`) must surface
    HTTP 400 with a diagnostic body, not silently fall through to
    default `hillclimb`."""
    h = _make_handler("/run?kihd=floodfill")
    status, body = _drive(h)
    assert status == 400, f"expected 400, got {status}; body={body!r}"
    assert b"kihd" in body, f"diagnostic missing offending key: {body!r}"
    assert b"unknown query key" in body, (
        f"diagnostic missing reason: {body!r}"
    )


def test_item7_invalid_seed_returns_400():
    """`?seed=abc` must surface HTTP 400, not silently coerce to None."""
    h = _make_handler("/run?seed=abc")
    status, body = _drive(h)
    assert status == 400, f"expected 400, got {status}; body={body!r}"
    assert b"seed" in body, f"diagnostic missing 'seed': {body!r}"
    assert b"abc" in body, f"diagnostic missing offending value: {body!r}"


def test_item7_invalid_size_returns_400():
    """`?size=xl` must surface HTTP 400, not silently coerce to 10."""
    h = _make_handler("/run?size=xl")
    status, body = _drive(h)
    assert status == 400, f"expected 400, got {status}; body={body!r}"
    assert b"size" in body, f"diagnostic missing 'size': {body!r}"
    assert b"xl" in body, f"diagnostic missing offending value: {body!r}"


def test_item7_unknown_key_diagnostic_lists_allowed():
    """The 400 body should also surface the allowed-key list so the
    caller can self-correct without grepping the source."""
    h = _make_handler("/run?xyzzy=foo")
    status, body = _drive(h)
    assert status == 400
    for key in (b"kind", b"seed", b"maze", b"size"):
        assert key in body, f"allowed-key {key!r} missing from body: {body!r}"


# ============================================================================
# v2.x re-audit R3 (RT-M1/M2): a compile-time knob whose target @pure fn
# is absent from the chosen agent's source must be REJECTED, not silently
# dropped. Pre-fix, `maze` for the `nn` agent (no use_maze()) and `size`
# for `nn` (grid_n rewrite was gated to qlearn) no-op'd silently — the
# request was accepted and logged but the binary ignored the knob.
# ============================================================================
def _read_agent_src(hx: str) -> str:
    from helixc.examples.dashboard_server import EXAMPLES
    with open(os.path.join(EXAMPLES, hx), "r", encoding="utf-8") as f:
        return f.read()


def test_v2x_reaudit_r3_nn_maze_knob_rejected():
    """RT-M1: the `maze` knob must be REJECTED for the `nn` agent —
    dashboard_nn_agent.hx has no use_maze() to flip."""
    from helixc.examples.dashboard_server import _rewrite_knobs
    src = _read_agent_src("dashboard_nn_agent.hx")
    new_src, err = _rewrite_knobs(
        src, "dashboard_nn_agent.hx", "nn", None, True, None)
    assert new_src is None, "maze on nn must not silently rewrite"
    assert err is not None and "maze" in err and "nn" in err, err


def test_v2x_reaudit_r4_nn_size_knob_rejected():
    """RT HIGH-1 (R4): the `size` knob must be REJECTED for the `nn`
    agent. dashboard_nn_agent.hx HAS a grid_n(), but its grid_total()
    / goal_id() are HARDCODED, not derived from grid_n() — rewriting
    grid_n() alone would silently miscompile (a resized world with
    stale grid constants). The R3 fix wrongly honored it; R4
    restricts the knob to qlearn, whose constants derive from
    grid_n()."""
    from helixc.examples.dashboard_server import _rewrite_knobs
    src = _read_agent_src("dashboard_nn_agent.hx")
    new_src, err = _rewrite_knobs(
        src, "dashboard_nn_agent.hx", "nn", None, False, 15)
    assert new_src is None, "size on nn must not silently rewrite"
    assert err is not None and "size" in err and "nn" in err, err


def test_v2x_reaudit_r4_qlearn_size_knob_honored():
    """RT HIGH-1 control: the `size` knob IS honored for the `qlearn`
    agent, whose grid_total() / goal_id() derive from grid_n()."""
    from helixc.examples.dashboard_server import _rewrite_knobs
    src = _read_agent_src("dashboard_qlearn.hx")
    new_src, err = _rewrite_knobs(
        src, "dashboard_qlearn.hx", "qlearn", None, False, 15)
    assert err is None, f"size on qlearn must be honored, got: {err}"
    assert "@pure fn grid_n() -> i32 { 15 }" in new_src
    assert "@pure fn grid_n() -> i32 { 10 }" not in new_src


def test_v2x_reaudit_r3_qlearn_maze_knob_honored():
    """RT-M1 control: the `maze` knob still works for the `qlearn`
    agent — dashboard_qlearn.hx HAS a use_maze()."""
    from helixc.examples.dashboard_server import _rewrite_knobs
    src = _read_agent_src("dashboard_qlearn.hx")
    new_src, err = _rewrite_knobs(
        src, "dashboard_qlearn.hx", "qlearn", None, True, None)
    assert err is None, f"maze on qlearn must be honored, got: {err}"
    assert "@pure fn use_maze() -> i32 { 1 }" in new_src


# v2.x re-audit R4b (RT-M2): a knob target that occurs MORE THAN ONCE is
# rejected. `str.replace` rewrites every match; the pre-R4b presence-only
# check (`if target not in new_src`) would let a duplicated constant line
# be double-rewritten silently. Latent today (the real .hx files each
# define the constants once) — the guard catches a future regression.


def test_v2x_reaudit_r4b_duplicate_seed_target_rejected():
    """RT-M2: a source with two map_seed() definitions must be REJECTED,
    not have both definitions rewritten by str.replace."""
    from helixc.examples.dashboard_server import _rewrite_knobs
    dup_src = (
        "@pure fn map_seed() -> i32 { 12345 }\n"
        "@pure fn map_seed() -> i32 { 12345 }\n"
        "fn main() -> i32 { 0 }\n"
    )
    new_src, err = _rewrite_knobs(
        dup_src, "dup_agent.hx", "qlearn", 99, False, None)
    assert new_src is None, "a duplicated knob target must not be rewritten"
    assert err is not None and "ambiguous" in err and "seed" in err, err


def test_v2x_reaudit_r4b_duplicate_maze_target_rejected():
    """RT-M2: same guard for the `maze` knob."""
    from helixc.examples.dashboard_server import _rewrite_knobs
    dup_src = (
        "@pure fn use_maze() -> i32 { 0 }\n"
        "@pure fn use_maze() -> i32 { 0 }\n"
        "fn main() -> i32 { 0 }\n"
    )
    new_src, err = _rewrite_knobs(
        dup_src, "dup_agent.hx", "qlearn", None, True, None)
    assert new_src is None, "a duplicated knob target must not be rewritten"
    assert err is not None and "ambiguous" in err and "maze" in err, err


def test_v2x_reaudit_r4b_duplicate_grid_target_rejected():
    """RT-M2: same guard for the `size` knob."""
    from helixc.examples.dashboard_server import _rewrite_knobs
    dup_src = (
        "@pure fn grid_n() -> i32 { 10 }\n"
        "@pure fn grid_n() -> i32 { 10 }\n"
        "fn main() -> i32 { 0 }\n"
    )
    new_src, err = _rewrite_knobs(
        dup_src, "dup_agent.hx", "qlearn", None, False, 15)
    assert new_src is None, "a duplicated knob target must not be rewritten"
    assert err is not None and "ambiguous" in err and "size" in err, err
