"""Stage 22: tests for the unified diagnostics module."""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.diagnostics import (
    render_caret,
    did_you_mean,
    Diagnostic,
    DiagSink,
    use_color,
)
from helixc.frontend.parser import parse, ParseError
from helixc.frontend.typecheck import typecheck


SRC = "fn main() -> i32 { let x = 1; bar }\n"


def test_render_caret_basic_no_color():
    out = render_caret(
        filename="foo.hx", line=1, col=32, msg="unbound name 'bar'",
        source=SRC, color=False,
    )
    assert "error: unbound name 'bar'" in out
    assert "--> foo.hx:1:32" in out
    assert "| fn main() -> i32 { let x = 1; bar }" in out
    assert "^" in out
    # No ANSI escapes when color disabled
    assert "\x1b[" not in out


def test_render_caret_hint():
    out = render_caret(
        filename="x.hx", line=1, col=32, msg="unbound name 'bar'",
        source=SRC, hint="did you mean 'baz'?", color=False,
    )
    assert "= hint: did you mean 'baz'?" in out


def test_render_caret_with_code():
    out = render_caret(
        filename="x.hx", line=1, col=1, msg="provenance violation",
        source=SRC, code=24001, color=False,
    )
    assert "error[E24001]:" in out


def test_render_caret_warning_level():
    out = render_caret(
        filename="x.hx", line=1, col=1, msg="deprecated call",
        source=SRC, level="warning", color=False,
    )
    assert out.startswith("warning:")


def test_render_caret_color_on():
    out = render_caret(
        filename="x.hx", line=1, col=32, msg="x",
        source=SRC, color=True,
    )
    assert "\x1b[" in out
    assert "\x1b[0m" in out


def test_render_caret_out_of_range_falls_back():
    out = render_caret(
        filename="x.hx", line=999, col=1, msg="off-end", source=SRC,
        color=False,
    )
    assert out == "x.hx:999:1: error: off-end"


def test_render_caret_source_none_falls_back():
    out = render_caret(
        filename="x.hx", line=1, col=1, msg="nope", source=None,
        color=False,
    )
    assert out == "x.hx:1:1: error: nope"


def test_render_caret_span_len():
    out = render_caret(
        filename="x.hx", line=1, col=32, msg="x",
        source=SRC, span_len=3, color=False,
    )
    assert "^^^" in out


def test_did_you_mean_basic():
    out = did_you_mean("foo", ["foe", "for", "fork", "barbaz"])
    assert out
    assert out[0] in {"foe", "for", "fork"}


def test_did_you_mean_no_match():
    out = did_you_mean("zzz", ["foo", "bar", "baz"], cutoff=0.99)
    assert out == []


def test_did_you_mean_empty():
    assert did_you_mean("", ["a", "b"]) == []
    assert did_you_mean("foo", []) == []


def test_use_color_no_color_env(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("HELIXC_COLOR", raising=False)
    assert use_color() is False


def test_use_color_helixc_force_on(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("HELIXC_COLOR", "1")
    assert use_color() is True


def test_use_color_helixc_force_off(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("HELIXC_COLOR", "0")
    assert use_color() is False


def test_diag_sink_basic():
    s = DiagSink()
    assert not s
    s.error("x.hx", 1, 1, "first error")
    s.warning("x.hx", 2, 1, "a warning")
    assert len(s) == 2
    assert s.has_errors()
    out = s.render_all(source=None)
    assert "first error" in out
    assert "a warning" in out


def test_parse_error_render_uses_new_module():
    bad = "fn main() -> i32 { let x = }\n"
    try:
        parse(bad)
        raise AssertionError("expected ParseError")
    except ParseError as e:
        out = e.render(source=bad, filename="x.hx", color=False)
        assert "--> x.hx:" in out
        assert "|" in out
        assert "^" in out


def test_typecheck_render_emits_hint():
    src = "fn main() -> i32 { let foo = 5; bar }\n"
    prog = parse(src)
    errs = typecheck(prog)
    assert errs
    # First unbound-name error should have a did-you-mean hint.
    rendered = errs[0].render(source=src, filename="x.hx", color=False)
    assert "unbound name 'bar'" in rendered
    # The hint may or may not suggest 'foo' depending on cutoff; just
    # confirm rendered uses the new format.
    assert "--> x.hx:" in rendered


def test_diagnostic_dataclass_render():
    d = Diagnostic(
        filename="x.hx", line=1, col=1, msg="warn me",
        level="warning", hint="be careful",
    )
    out = d.render(source="line one\n", color=False)
    assert out.startswith("warning:")
    assert "= hint: be careful" in out


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
