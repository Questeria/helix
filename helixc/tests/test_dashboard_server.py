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
