"""Stage 23: tests for the helixc check.py CLI."""

from __future__ import annotations

import os
import sys
import subprocess
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.check import parse_args, extract_doc_comments, main


def write_src(content: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".hx", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# ---- parse_args ----
def test_parse_args_no_flags():
    a, errs = parse_args(["foo.hx"])
    assert not errs
    assert a.path == "foo.hx"
    assert a.opt_level == 1
    assert a.output is None
    assert a.libs == []


def test_parse_args_opt_levels():
    for lvl in range(4):
        a, errs = parse_args([f"-O{lvl}", "foo.hx"])
        assert not errs
        assert a.opt_level == lvl


def test_parse_args_bad_opt_level():
    a, errs = parse_args(["-O9", "foo.hx"])
    assert errs


def test_parse_args_output():
    a, errs = parse_args(["-o", "out.bin", "foo.hx"])
    assert not errs
    assert a.output == "out.bin"


def test_parse_args_output_missing():
    a, errs = parse_args(["-o"])
    assert errs


def test_parse_args_lib_separate():
    a, errs = parse_args(["-l", "m", "-l", "c", "foo.hx"])
    assert not errs
    assert a.libs == ["m", "c"]


def test_parse_args_lib_attached():
    a, errs = parse_args(["-lm", "-lc", "foo.hx"])
    assert not errs
    assert a.libs == ["m", "c"]


def test_parse_args_warnings():
    a, errs = parse_args(["-Wdeprecated", "-Wfoo=error", "foo.hx"])
    assert not errs
    assert a.warnings["deprecated"] == "warn"
    assert a.warnings["foo"] == "error"


def test_parse_args_unknown_flag():
    a, errs = parse_args(["--unknown", "foo.hx"])
    assert errs


def test_parse_args_emit_flags():
    a, errs = parse_args(["--emit-ir", "foo.hx"])
    assert "--emit-ir" in a.flags

    a, errs = parse_args(["--emit-asm", "foo.hx"])
    assert "--emit-asm" in a.flags

    a, errs = parse_args(["--emit-ptx", "foo.hx"])
    assert "--emit-ptx" in a.flags


def test_parse_args_check_only():
    a, errs = parse_args(["--check-only", "foo.hx"])
    assert "--check-only" in a.flags


def test_parse_args_no_color():
    a, errs = parse_args(["--no-color", "foo.hx"])
    assert a.color is False


def test_parse_args_color_force():
    a, errs = parse_args(["--color", "foo.hx"])
    assert a.color is True


# ---- doc extraction ----
def test_doc_simple():
    src = """\
/// Adds two i32 values.
/// Returns their sum.
fn add(a: i32, b: i32) -> i32 { a + b }
"""
    md = extract_doc_comments(src)
    assert "## `fn add`" in md
    assert "Adds two i32 values." in md
    assert "Returns their sum." in md


def test_doc_struct():
    src = """\
/// A 2D point.
struct Pt { x: f64, y: f64 }
"""
    md = extract_doc_comments(src)
    assert "## `struct Pt`" in md


def test_doc_skip_no_doc_fn():
    src = "fn nodoc() -> i32 { 0 }\n"
    md = extract_doc_comments(src)
    assert "## `fn nodoc`" not in md


def test_doc_multiple_decls():
    src = """\
/// First.
fn one() -> i32 { 1 }

/// Second.
fn two() -> i32 { 2 }
"""
    md = extract_doc_comments(src)
    assert "## `fn one`" in md
    assert "## `fn two`" in md
    assert "First." in md
    assert "Second." in md


# ---- main() end-to-end ----
def test_main_check_only_clean(capsys):
    p = write_src("fn main() -> i32 { 1 + 2 }\n")
    try:
        rc = main([p, "--check-only"])
        assert rc == 0
        cap = capsys.readouterr()
        assert "check-only" in cap.out
    finally:
        os.remove(p)


def test_main_parse_error(capsys):
    p = write_src("fn main() -> i32 { let x = }\n")
    try:
        rc = main([p])
        assert rc == 1
        cap = capsys.readouterr()
        assert "PARSE ERROR" in cap.err
    finally:
        os.remove(p)


def test_main_emit_ir(capsys):
    p = write_src("fn main() -> i32 { 1 + 2 }\n")
    try:
        rc = main([p, "--emit-ir"])
        assert rc == 0
        cap = capsys.readouterr()
        assert "ir:" in cap.out
        assert "fn main" in cap.out
    finally:
        os.remove(p)


def test_main_emit_ast(capsys):
    p = write_src("fn main() -> i32 { 1 + 2 }\n")
    try:
        rc = main([p, "--emit-ast"])
        assert rc == 0
        cap = capsys.readouterr()
        assert "fn main" in cap.out
    finally:
        os.remove(p)


def test_main_doc(capsys):
    p = write_src("/// Identity\nfn id(x: i32) -> i32 { x }\n")
    try:
        rc = main([p, "--doc"])
        assert rc == 0
        cap = capsys.readouterr()
        assert "## `fn id`" in cap.out
    finally:
        os.remove(p)


def test_main_help(capsys):
    rc = main(["--help"])
    assert rc == 0


def test_main_missing_file(capsys):
    rc = main(["/nonexistent/path.hx"])
    assert rc == 2


def test_main_emit_asm(capsys):
    p = write_src("fn main() -> i32 { 42 }\n")
    try:
        rc = main([p, "--emit-asm"])
        assert rc == 0
        cap = capsys.readouterr()
        assert "asm:" in cap.out
    finally:
        os.remove(p)


def test_main_o_writes_file(tmp_path):
    src = "fn main() -> i32 { 5 }\n"
    src_path = str(tmp_path / "in.hx")
    with open(src_path, "w") as f:
        f.write(src)
    out_path = str(tmp_path / "out.bin")
    rc = main([src_path, "-o", out_path])
    assert rc == 0
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_main_emit_asm_traps_backend_error(monkeypatch, capsys, tmp_path):
    """Audit 28.8 A9: --emit-asm wraps the backend call in try/except.
    Internal compiler bugs (raising any Exception from
    compile_module_to_elf) should NOT leak a Python traceback — instead
    helixc should print `helixc: internal error: <type>: <msg>` and
    exit 1."""
    from helixc.backend import x86_64 as _be

    def _boom(_mod):
        raise RuntimeError("synthetic backend explosion")

    monkeypatch.setattr(_be, "compile_module_to_elf", _boom)

    src_path = str(tmp_path / "in.hx")
    with open(src_path, "w") as f:
        f.write("fn main() -> i32 { 1 }\n")

    rc = main([src_path, "--emit-asm"])
    cap = capsys.readouterr()
    assert rc == 1, "must exit 1 on backend internal error"
    assert "internal error" in cap.err
    assert "RuntimeError" in cap.err
    assert "synthetic backend explosion" in cap.err
    # No raw Python traceback should leak.
    assert "Traceback (most recent call last)" not in cap.err


def test_main_o_traps_backend_error(monkeypatch, capsys, tmp_path):
    """Audit 28.8 A9: -o path wraps the backend call in try/except so
    internal compiler bugs surface as clean diagnostics."""
    from helixc.backend import x86_64 as _be

    def _boom(_mod):
        raise ValueError("synthetic codegen failure")

    monkeypatch.setattr(_be, "compile_module_to_elf", _boom)

    src_path = str(tmp_path / "in.hx")
    out_path = str(tmp_path / "out.bin")
    with open(src_path, "w") as f:
        f.write("fn main() -> i32 { 1 }\n")

    rc = main([src_path, "-o", out_path])
    cap = capsys.readouterr()
    assert rc == 1
    assert "internal error" in cap.err
    assert "ValueError" in cap.err
    # The output file should NOT have been created.
    assert not os.path.exists(out_path)


def test_main_o_handles_oserror_on_write(monkeypatch, capsys, tmp_path):
    """Audit 28.8 A9: -o catches OSError on the file write so
    permission / disk-full failures get a clean diagnostic too."""
    import builtins as _bi
    real_open = _bi.open

    src_path = str(tmp_path / "in.hx")
    with open(src_path, "w") as f:
        f.write("fn main() -> i32 { 1 }\n")
    out_path = str(tmp_path / "noway" / "deep" / "out.bin")

    def _open_blocking(path, *args, **kwargs):
        # Only block the wb open for the output file.
        if str(path) == out_path and "b" in (args[0] if args else
                                              kwargs.get("mode", "")):
            raise PermissionError("synthetic permission denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(_bi, "open", _open_blocking)
    rc = main([src_path, "-o", out_path])
    cap = capsys.readouterr()
    assert rc == 1
    assert "cannot write output" in cap.err
    assert "synthetic permission denied" in cap.err


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
