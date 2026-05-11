"""Stage 28.7: tests for @deprecated + @since version gating."""

from __future__ import annotations

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.frontend import ast_nodes as A
from helixc.frontend.deprecated_pass import (
    deprecation_msg,
    since_marker,
    find_deprecated_decls,
    find_deprecation_call_sites,
    emit_warnings,
)
from helixc.check import main


def test_parse_deprecated_attribute():
    src = '@deprecated("use foo_v2") fn foo() -> i32 { 0 }'
    prog = parse(src)
    fn = prog.items[0]
    assert "deprecated" in fn.attrs
    assert deprecation_msg(fn) == "use foo_v2"


def test_parse_deprecated_no_msg():
    src = '@deprecated fn foo() -> i32 { 0 }'
    prog = parse(src)
    fn = prog.items[0]
    assert "deprecated" in fn.attrs
    assert deprecation_msg(fn) == ""


def test_parse_since():
    src = '@since("v0.3") fn new_api() -> i32 { 0 }'
    prog = parse(src)
    fn = prog.items[0]
    assert "since" in fn.attrs
    assert since_marker(fn) == "v0.3"


def test_not_deprecated():
    src = 'fn ok() -> i32 { 0 }'
    prog = parse(src)
    fn = prog.items[0]
    assert deprecation_msg(fn) is None


def test_find_deprecated_decls():
    src = """
@deprecated("old") fn old1() -> i32 { 0 }
fn ok() -> i32 { 0 }
@deprecated fn old2() -> i32 { 0 }
"""
    prog = parse(src)
    decls = find_deprecated_decls(prog)
    assert set(decls.keys()) == {"old1", "old2"}
    assert decls["old1"] == "old"
    assert decls["old2"] == ""


def test_find_call_sites_to_deprecated():
    src = """
@deprecated("use new_id") fn old_id(x: i32) -> i32 { x }
fn new_caller() -> i32 { old_id(5) }
fn another() -> i32 { old_id(3) + 1 }
fn safe() -> i32 { 42 }
"""
    prog = parse(src)
    sites = find_deprecation_call_sites(prog)
    assert len(sites) == 2
    names = {n for (n, _, _) in sites}
    assert names == {"old_id"}


def test_emit_warnings_messages():
    src = """
@deprecated("renamed to id2") fn id_old(x: i32) -> i32 { x }
fn user() -> i32 { id_old(5) }
"""
    prog = parse(src)
    out = emit_warnings(prog)
    assert len(out) == 1
    assert "renamed to id2" in out[0]
    assert "id_old" in out[0]


def test_emit_warnings_no_calls():
    src = """
@deprecated("x") fn dead() -> i32 { 0 }
fn safe() -> i32 { 42 }
"""
    prog = parse(src)
    out = emit_warnings(prog)
    assert out == []


def test_emit_warnings_returns_list():
    """Audit 28.8 C1-M1: emit_warnings should return its list, NOT
    monkey-patch `_deprecation_warnings` onto A.Program. Verify the
    return is the source of truth and multiple calls are idempotent."""
    src = """
@deprecated fn d() -> i32 { 0 }
fn u() -> i32 { d() }
"""
    prog = parse(src)
    first = emit_warnings(prog)
    assert isinstance(first, list)
    assert len(first) == 1
    # Re-invocation: same result. The pass no longer mutates prog.
    second = emit_warnings(prog)
    assert second == first
    # The monkey-patched attribute should NOT exist (caller-store model).
    assert not hasattr(prog, "_deprecation_warnings"), \
        "emit_warnings must not couple AST to pass output (Audit 28.8 C1-M1)"


def test_cli_deprecated_warning_logged(capsys, tmp_path):
    src = """
@deprecated("use new_api") fn old_api() -> i32 { 0 }
fn main() -> i32 { old_api() }
"""
    p = str(tmp_path / "in.hx")
    with open(p, "w") as f:
        f.write(src)
    rc = main([p])
    out = capsys.readouterr().out
    assert "deprecated:" in out
    assert "old_api" in out
    assert rc == 0  # warning doesn't fail


def test_cli_deprecated_error_promotion(capsys, tmp_path):
    src = """
@deprecated fn old() -> i32 { 0 }
fn main() -> i32 { old() }
"""
    p = str(tmp_path / "in.hx")
    with open(p, "w") as f:
        f.write(src)
    rc = main([p, "-Wdeprecated=error"])
    out = capsys.readouterr().out
    assert "ERROR" in out
    assert rc == 1


def test_cli_clean_no_deprecation(capsys, tmp_path):
    src = "fn main() -> i32 { 0 }\n"
    p = str(tmp_path / "in.hx")
    with open(p, "w") as f:
        f.write(src)
    rc = main([p])
    assert rc == 0
    out = capsys.readouterr().out
    assert "deprecated:" not in out


# ----------------------------------------------------------------------
# Audit-28.8 A5 regressions: the deprecated-call walker must recurse
# into `Index.indices`, `For.iter_expr`, `Range.start/end`, etc. Without
# these arms, deprecation warnings were silently lost when the offending
# call appeared in those positions.
# ----------------------------------------------------------------------
def test_find_call_sites_inside_index():
    """A5: `arr[deprecated_fn(2)]` — deprecated call inside Index.indices."""
    src = """
@deprecated("use new_fn") fn old_fn(x: i32) -> i32 { x }
fn main() -> i32 {
    let arr: [i32; 4] = [0, 0, 0, 0];
    arr[old_fn(2)]
}
"""
    prog = parse(src)
    sites = find_deprecation_call_sites(prog)
    names = {n for (n, _, _) in sites}
    assert "old_fn" in names, (
        f"old_fn(2) inside arr[..] should be detected; got names={names}"
    )


def test_find_call_sites_inside_range_end():
    """A5: `for i in 0..old_fn(10) { ... }` — deprecated call in Range.end."""
    src = """
@deprecated("use new_fn") fn old_fn(x: i32) -> i32 { x }
fn main() -> i32 {
    for i in 0..old_fn(10) {
        let unused: i32 = i;
    }
    0
}
"""
    prog = parse(src)
    sites = find_deprecation_call_sites(prog)
    names = {n for (n, _, _) in sites}
    assert "old_fn" in names, (
        f"old_fn(10) in Range.end should be detected; got names={names}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
