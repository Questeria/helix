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


def _count_op_kinds(mod):
    """Helper: total number of ops per kind across all functions."""
    from collections import Counter
    c = Counter()
    for fn in mod.functions.values():
        for blk in fn.blocks:
            for op in blk.ops:
                c[op.kind.name] += 1
    return c


def test_o1_invokes_fold_module(monkeypatch, capsys, tmp_path):
    """Audit 28.8 A10: -O1 must invoke fold_module (constant folding) in
    addition to fdce_module. Pre-fix, only fdce ran. We monkeypatch
    fold_module to count invocations from check.py.main()."""
    from helixc.ir.passes import const_fold
    call_count = [0]
    real_fold = const_fold.fold_module

    def counted(mod):
        call_count[0] += 1
        return real_fold(mod)

    monkeypatch.setattr(const_fold, "fold_module", counted)
    # check.py imports fold_module by attribute access at call site, but
    # since we monkey at module-level we need to patch the binding the
    # check module uses too — the import happens at runtime inside main().
    src_path = str(tmp_path / "fold.hx")
    with open(src_path, "w") as f:
        f.write("fn main() -> i32 { 1 + 2 }\n")
    rc = main([src_path, "--emit-ir", "-O1"])
    assert rc == 0, "compile must succeed"
    assert call_count[0] == 1, "fold_module must be invoked at -O1"


def test_o0_skips_optimization(monkeypatch, capsys, tmp_path):
    """Audit 28.8 A10: -O0 disables all opt passes (fdce, fold, cse,
    dce). Pre-fix would still skip them; verify the discipline holds."""
    from helixc.ir.passes import const_fold, cse, dce, fdce
    fdce_called = [0]
    fold_called = [0]
    cse_called = [0]
    dce_called = [0]

    monkeypatch.setattr(
        fdce, "fdce_module",
        lambda m, **kw: fdce_called.__setitem__(0, fdce_called[0] + 1) or 0
    )
    monkeypatch.setattr(
        const_fold, "fold_module",
        lambda m: fold_called.__setitem__(0, fold_called[0] + 1) or 0
    )
    monkeypatch.setattr(
        cse, "cse_module",
        lambda m: cse_called.__setitem__(0, cse_called[0] + 1) or 0
    )
    monkeypatch.setattr(
        dce, "dce_module",
        lambda m: dce_called.__setitem__(0, dce_called[0] + 1) or 0
    )
    src_path = str(tmp_path / "noopt.hx")
    with open(src_path, "w") as f:
        f.write("fn main() -> i32 { 0 }\n")
    rc = main([src_path, "--emit-ir", "-O0"])
    assert rc == 0
    assert fdce_called[0] == 0, "fdce must NOT run at -O0"
    assert fold_called[0] == 0, "fold must NOT run at -O0"
    assert cse_called[0] == 0, "cse must NOT run at -O0"
    assert dce_called[0] == 0, "dce must NOT run at -O0"


def test_o2_invokes_cse_and_dce(monkeypatch, capsys, tmp_path):
    """Audit 28.8 A10: -O2 must invoke cse_module + dce_module (on top
    of -O1's fdce + fold). Pre-fix had a no-op try/import-pass
    placeholder; this guards the regression."""
    from helixc.ir.passes import cse, dce
    cse_calls = [0]
    dce_calls = [0]
    real_cse = cse.cse_module
    real_dce = dce.dce_module

    def cse_counted(mod):
        cse_calls[0] += 1
        return real_cse(mod)

    def dce_counted(mod):
        dce_calls[0] += 1
        return real_dce(mod)

    monkeypatch.setattr(cse, "cse_module", cse_counted)
    monkeypatch.setattr(dce, "dce_module", dce_counted)
    src_path = str(tmp_path / "o2.hx")
    with open(src_path, "w") as f:
        f.write("fn main() -> i32 { 1 + 2 + 3 }\n")
    rc = main([src_path, "--emit-ir", "-O2"])
    assert rc == 0
    assert cse_calls[0] == 1, "cse must run at -O2"
    assert dce_calls[0] == 1, "dce must run at -O2"


def test_o3_runs_at_least_o2_passes(monkeypatch, capsys, tmp_path):
    """Audit 28.8 A10: -O3 is currently alias-of-O2 (no aggressive
    layer yet). Verify cse + dce still run."""
    from helixc.ir.passes import cse, dce
    cse_calls = [0]
    dce_calls = [0]
    real_cse = cse.cse_module
    real_dce = dce.dce_module

    monkeypatch.setattr(cse, "cse_module",
                        lambda m: (cse_calls.__setitem__(0, cse_calls[0] + 1)
                                   or real_cse(m)))
    monkeypatch.setattr(dce, "dce_module",
                        lambda m: (dce_calls.__setitem__(0, dce_calls[0] + 1)
                                   or real_dce(m)))
    src_path = str(tmp_path / "o3.hx")
    with open(src_path, "w") as f:
        f.write("fn main() -> i32 { 1 }\n")
    rc = main([src_path, "--emit-ir", "-O3"])
    assert rc == 0
    assert cse_calls[0] == 1
    assert dce_calls[0] == 1


# ----------------------------------------------------------------------
# Audit 28.8 cycle 2 C2-1 — AD-warning drain must run on EVERY exit path
# ----------------------------------------------------------------------
def test_ad_drain_runs_on_default_no_emit(capsys, tmp_path):
    """C2-1: B13 mixed-inner widening warnings emit during typecheck.
    Pre-fix, the drain only ran inside the `--emit-*` / `-o` branch, so
    `python -m helixc.check loss.hx` (no flags) silently swallowed them.
    Verify the warning now appears on stderr for default invocation."""
    src_path = str(tmp_path / "loss.hx")
    with open(src_path, "w") as f:
        f.write(
            "fn loss(x: D<f64>, y: D<i32>) -> D<f64> { x + y }\n"
            "fn main() -> i32 { 0 }\n"
        )
    rc = main([src_path])
    cap = capsys.readouterr()
    # Compile must still succeed (B13 is a warning, not an error).
    assert rc == 0, f"unexpected rc={rc}; stderr={cap.err!r}"
    # The drain emits an `ad: N warning(s)` line on stdout AND the
    # actual warning lines on stderr.
    assert "ad:" in cap.out, (
        f"missing ad-warning header on stdout; stdout={cap.out!r}"
    )
    assert "24200" in cap.err or "AD002" in cap.err, (
        f"missing B13 widening warning on stderr; stderr={cap.err!r}"
    )


def test_ad_drain_runs_on_check_only(capsys, tmp_path):
    """C2-1: `--check-only` short-circuits before lowering. The drain
    must still run so CI lint sweeps see B13 warnings."""
    src_path = str(tmp_path / "loss.hx")
    with open(src_path, "w") as f:
        f.write(
            "fn loss(x: D<f64>, y: D<i32>) -> D<f64> { x + y }\n"
            "fn main() -> i32 { 0 }\n"
        )
    rc = main([src_path, "--check-only"])
    cap = capsys.readouterr()
    assert rc == 0
    assert "ad:" in cap.out
    assert "24200" in cap.err or "AD002" in cap.err, (
        f"missing B13 widening warning on stderr; stderr={cap.err!r}"
    )


def test_ad_drain_clears_state_between_compiles(capsys, tmp_path):
    """C2-1: stale `_DIFF_WARNINGS` from a prior compile must not
    leak into the next file's diagnostics. The outer `main` wrapper
    clears the channel at entry."""
    from helixc.frontend import autodiff
    # Seed the module-level list with a stale warning.
    autodiff._DIFF_WARNINGS.append("stale-from-prior-compile (trap 24200)")
    # Now compile a clean file (no D<...> at all).
    src_path = str(tmp_path / "clean.hx")
    with open(src_path, "w") as f:
        f.write("fn main() -> i32 { 1 + 2 }\n")
    rc = main([src_path])
    cap = capsys.readouterr()
    assert rc == 0
    # The stale warning must NOT surface against the clean file.
    assert "stale-from-prior-compile" not in cap.err, (
        f"stale warning leaked across compiles; stderr={cap.err!r}"
    )


def test_ad_drain_wad_error_promotes(capsys, tmp_path):
    """C2-1 + B5: `-Wad=error` promotes drained warnings to errors,
    flipping the exit code to 1 even on default invocation."""
    src_path = str(tmp_path / "loss.hx")
    with open(src_path, "w") as f:
        f.write(
            "fn loss(x: D<f64>, y: D<i32>) -> D<f64> { x + y }\n"
            "fn main() -> i32 { 0 }\n"
        )
    rc = main([src_path, "-Wad=error"])
    cap = capsys.readouterr()
    assert rc == 1, (
        f"-Wad=error must flip rc to 1 when warnings drain; "
        f"rc={rc} stdout={cap.out!r} stderr={cap.err!r}"
    )
    assert "ERROR" in cap.out or "ERROR" in cap.err


def test_ad_drain_subprocess_default(tmp_path):
    """C2-1: end-to-end subprocess test — the dominant user invocation
    `python -m helixc.check loss.hx` surfaces B13 warnings on stderr."""
    src_path = str(tmp_path / "loss.hx")
    with open(src_path, "w") as f:
        f.write(
            "fn loss(x: D<f64>, y: D<i32>) -> D<f64> { x + y }\n"
            "fn main() -> i32 { 0 }\n"
        )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.check", src_path],
        capture_output=True, text=True, check=False,
        cwd=os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))),
    )
    assert proc.returncode == 0, (
        f"compile failed: rc={proc.returncode} "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "24200" in proc.stderr or "AD002" in proc.stderr, (
        f"B13 warning missing from subprocess stderr; "
        f"stderr={proc.stderr!r}"
    )


# --- Stage 28.9 cycle 24 audit-R regression tests ---


def test_c23_1_fold_error_via_check_py_clean_diagnostic(capsys, tmp_path):
    """C23-1 regression (Critical, conf 92): a compile-time NaN fold
    (trap 17001) via check.py must render as a clean
    `helixc: const-fold error: [trap 17001] ...` diagnostic with
    rc=1 — NOT as 'helixc: internal error: FoldError: ...' followed by
    'this is a compiler bug — please file an issue.' Pre-cycle-24 fix,
    check.py invoked fold_module bare; FoldError bubbled to the outer
    main() Exception handler and got mis-labelled as a compiler bug."""
    src_path = str(tmp_path / "nan_fold.hx")
    with open(src_path, "w") as f:
        f.write(
            "fn main() -> f32 { "
            "let a = 1.0e200 * 1.0e200; "
            "let b = 1.0e200 * 1.0e200; "
            "a - b }\n"
        )
    rc = main([src_path, "--emit-ir", "-O1"])
    captured = capsys.readouterr()
    assert rc == 1, f"expected rc=1 for NaN fold, got {rc}"
    assert "const-fold error" in captured.err, (
        f"expected 'const-fold error' diagnostic; got stderr={captured.err!r}"
    )
    assert "17001" in captured.err, (
        f"expected trap 17001 reference; got stderr={captured.err!r}"
    )
    assert "compiler bug" not in captured.err, (
        "FoldError must NOT be mis-labelled as 'compiler bug' — that's "
        "the wrong message for a user-authored source error. "
        f"got stderr={captured.err!r}"
    )
    assert "internal error" not in captured.err, (
        "FoldError must NOT route through the 'internal error' handler; "
        f"got stderr={captured.err!r}"
    )


def test_c23_3_effect_check_via_check_py_flags_pure_violation(capsys, tmp_path):
    """C23-3 + C24-1 + C25-3 regression (Important, conf 88+):
    check.py's optimization pipeline runs effect_check after fold/cse/dce.
    Per the documented policy (effect_check.py docstring lines 296-303),
    a @pure violation (trap 19001) prints as `warning: effect-check: ...`
    by default and ONLY returns rc=1 under `--strict`. Cycle 24 wrongly
    treated all violations as hard errors, breaking
    helixc/examples/hello_world.hx; cycle 26 restored the documented
    warn-by-default policy and mirrored x86_64.py."""
    src_path = str(tmp_path / "pure_io.hx")
    with open(src_path, "w") as f:
        f.write(
            "@pure fn bad(x: i32) -> i32 { print_int(x); x }\n"
            "fn main() -> i32 { bad(42) }\n"
        )
    # Default: warn but emit IR, rc=0
    rc = main([src_path, "--emit-ir", "-O1"])
    captured = capsys.readouterr()
    assert rc == 0, (
        f"@pure violation must be a WARNING (rc=0) by default; "
        f"got rc={rc}. stderr={captured.err!r}"
    )
    assert "warning: effect-check:" in captured.err, (
        f"expected `warning: effect-check:` prefix; got stderr={captured.err!r}"
    )
    assert "19001" in captured.err, (
        f"expected trap 19001 reference; got stderr={captured.err!r}"
    )
    assert "bad" in captured.err, (
        f"expected mention of violating fn 'bad'; got stderr={captured.err!r}"
    )

    # --strict: turn the warning into a hard failure
    rc_strict = main([src_path, "--emit-ir", "-O1", "--strict"])
    captured_strict = capsys.readouterr()
    assert rc_strict == 1, (
        f"@pure violation under --strict must rc=1; got {rc_strict}. "
        f"stderr={captured_strict.err!r}"
    )
    assert "--strict aborts" in captured_strict.err, (
        f"expected --strict abort message; got stderr={captured_strict.err!r}"
    )


def test_c23_3_effect_check_via_check_py_clean_when_correct(capsys, tmp_path):
    """C23-3 regression-the-other-way: a program with correct effect
    declarations must STILL compile clean via check.py."""
    src_path = str(tmp_path / "pure_clean.hx")
    with open(src_path, "w") as f:
        f.write(
            "@pure fn double(x: i32) -> i32 { x + x }\n"
            "fn main() -> i32 { double(21) }\n"
        )
    rc = main([src_path, "--emit-ir", "-O1"])
    assert rc == 0, f"clean @pure program must compile rc=0, got {rc}"


def test_c25_3_19002_unused_effect_never_aborts(capsys, tmp_path):
    """C25-3 regression (HIGH conf 88) + C27-1 strengthening (MED 88):
    trap 19002 (declared unused effect) is documented as informational-
    only ("a code smell, not a correctness violation" — effect_check.py
    docstring 296-303). Even under --strict it must NOT cause rc=1.
    Pre-cycle-26 fix grouped 19001 and 19002 together as hard failures.

    Two assertions:
    (1) Unit-level: build the 19002 case directly in IR and verify
        check_module produces 19002 (not 19001).
    (2) Integration-level (cycle 28 C27-1 fix): invoke check.py main()
        with --strict on a program that produces ONLY 19002 (no 19001),
        and assert rc=0. Pre-fix, the C25-3 test claimed to verify
        --strict but never actually invoked check.py with the flag.
    """
    from helixc.ir import tir
    from helixc.ir.passes.effect_check import check_module
    # --- Part 1: unit-level classification ---
    mod = tir.Module()
    i32 = tir.TIRScalar("i32")
    blk = tir.Block(id=0)
    v_p = tir.Value(id=0, ty=i32)  # param
    v_r = tir.Value(id=1, ty=i32)
    blk.ops = [
        tir.Op(kind=tir.OpKind.ADD, operands=[v_p, v_p], results=[v_r]),
        tir.Op(kind=tir.OpKind.RETURN, operands=[v_r], results=[]),
    ]
    # @effect(io) declared but body has no PRINT/FFI — produces 19002.
    fn = tir.FnIR(name="declares_io_but_uses_none",
                  params=[v_p], return_ty=i32, blocks=[blk],
                  attrs={"effect:io": True})
    mod.functions["declares_io_but_uses_none"] = fn
    errs = check_module(mod)
    has_19002 = any("19002" in e for e in errs)
    has_19001 = any("19001" in e for e in errs)
    assert has_19002 and not has_19001, (
        f"expected exactly trap 19002 (declared unused effect); "
        f"got errs={errs}"
    )

    # --- Part 2: cycle 28 C27-1 — integration with check.py --strict ---
    # We need a surface program that produces ONLY 19002, NOT 19001.
    # That means: a fn with @effect(io) declared but with a body that
    # doesn't actually use io, AND that's not transitively called by
    # a non-declaring fn (which would trigger 19001).
    # The typecheck path catches surface-level effect mismatches at
    # the AST. To exercise the IR-only 19002 path, the fn must be
    # reachable so fdce doesn't drop it. Use a non-pure caller that
    # also declares the effect, so the caller doesn't trip 19001.
    src_path = str(tmp_path / "unused_eff_only.hx")
    with open(src_path, "w") as f:
        f.write(
            "@effect(io) fn unused_io_decl(x: i32) -> i32 { x + 1 }\n"
            "@effect(io) fn main() -> i32 { unused_io_decl(42) }\n"
        )
    # Even under --strict, 19002-only must NOT cause failure.
    rc = main([src_path, "--emit-ir", "-O1", "--strict"])
    captured = capsys.readouterr()
    assert rc == 0, (
        f"trap-19002-only program under --strict must rc=0 "
        f"(C27-1: 19002 is informational only). got rc={rc}. "
        f"stderr={captured.err!r}"
    )
    # 19002 may or may not appear depending on fdce removal — what
    # matters is no `warning: effect-check:` 19001 line appears AND
    # rc=0. If 19002 does appear, it should be on the informational
    # `   effect-check:` line.
    assert "warning: effect-check:" not in captured.err, (
        f"19002-only must not produce a hard `warning: effect-check:` "
        f"line; got stderr={captured.err!r}"
    )


def test_c24_1_hello_world_compiles_under_check_py(capsys, tmp_path):
    """C24-1 regression (Critical conf 95): the canonical example
    helixc/examples/hello_world.hx must compile clean via check.py
    despite main calling print_str (io effect) without explicit
    @effect(io). Pre-cycle-26, cycle-24's strict effect_check wiring
    rejected hello_world with `trap 19001`, breaking the canonical
    example through the developer CLI."""
    src_path = str(tmp_path / "hello.hx")
    with open(src_path, "w") as f:
        f.write(
            'fn main() -> i32 { print_str("Hello\\n"); 0 }\n'
        )
    rc = main([src_path, "--emit-ir", "-O1"])
    captured = capsys.readouterr()
    assert rc == 0, (
        f"hello-world shape (main with io, no @effect(io)) must rc=0 "
        f"by default; got rc={rc}. stderr={captured.err!r}"
    )


def test_c29_r3_real_hello_world_example_compiles(capsys):
    """C29-R3 regression (MEDIUM conf 85): the cycle-26 hello-world
    test uses a synthetic 1-liner. The REAL helixc/examples/hello_world.hx
    has 4 print_str calls + write_file + conditional branches — a
    structurally different program. A regression in the example file
    itself (syntax change, added @effect annotation, etc.) would not
    be caught by the synthetic test. This integration test invokes
    check.py on the actual file."""
    from pathlib import Path
    real_path = (
        Path(__file__).parent.parent / "examples" / "hello_world.hx"
    )
    assert real_path.is_file(), (
        f"expected helixc/examples/hello_world.hx to exist at {real_path}"
    )
    rc = main([str(real_path), "--emit-ir", "-O1"])
    captured = capsys.readouterr()
    assert rc == 0, (
        f"real hello_world.hx must compile clean via check.py; "
        f"got rc={rc}. stderr={captured.err!r}"
    )
    # Stage 28.9 cycle 32 audit-R C31-R3 fix (conf 85): the canonical
    # example DOES emit a `warning: effect-check:` line because main
    # has io effects without explicit @effect(io) — that's the
    # documented warn-by-default behavior (cycle 26 C24-1). The
    # regression guard here is NARROWER: assert the unknown-trap-id
    # fail-closed branch never fires for the canonical example. If
    # a future change makes effect_check emit an unrecognized trap-id
    # on this file, the test fails loudly. Regular warnings are
    # expected.
    assert "unknown trap-id" not in captured.err, (
        f"real hello_world.hx must not produce 'unknown trap-id' "
        f"diagnostics; got stderr={captured.err!r}"
    )
    # Also assert the example does NOT raise `helixc: const-fold error:`
    # or `helixc: internal error:` — both indicate compiler bugs vs
    # source-level diagnostics.
    assert "helixc: const-fold error:" not in captured.err
    assert "helixc: internal error:" not in captured.err
    assert "compiler bug" not in captured.err


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
