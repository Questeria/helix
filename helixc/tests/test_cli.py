"""Stage 23: tests for the helixc check.py CLI."""

from __future__ import annotations

import os
import json
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

    a, errs = parse_args(["--emit-proof-obligations", "foo.hx"])
    assert "--emit-proof-obligations" in a.flags


def test_parse_args_no_stdlib():
    a, errs = parse_args(["--no-stdlib", "foo.hx"])
    assert not errs
    assert "--no-stdlib" in a.flags


def test_parse_args_check_only():
    a, errs = parse_args(["--check-only", "foo.hx"])
    assert "--check-only" in a.flags


def test_emit_modes_are_mutually_exclusive(capsys, tmp_path):
    src_path = str(tmp_path / "emit_conflict.hx")
    with open(src_path, "w") as f:
        f.write("fn main() -> i32 { 0 }\n")
    rc = main([src_path, "--emit-proof-obligations", "--emit-ast"])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert "choose exactly one stdout-producing mode" in captured.err
    assert "--emit-proof-obligations" in captured.err
    assert "--emit-ast" in captured.err


def test_doc_and_emit_proof_obligations_are_mutually_exclusive(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "doc_conflict.hx")
    with open(src_path, "w") as f:
        f.write("/// Main docs\nfn main() -> i32 { 0 }\n")
    rc = main([src_path, "--emit-proof-obligations", "--doc"])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert "choose exactly one stdout-producing mode" in captured.err
    assert "--emit-proof-obligations" in captured.err
    assert "--doc" in captured.err


def test_c117_emit_ptx_uses_kernel_attrs(capsys):
    src = write_src("@kernel fn k() { }\nfn main() -> i32 { 42 }\n")
    rc = main([src, "--emit-ptx"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert ".visible .entry k" in captured.out, captured.out
    assert "no @kernel fns" not in captured.out


def test_c119_emit_ptx_rejects_no_kernel_modules(capsys):
    src = write_src("fn helper(x: i32) -> i32 { x + 1 }\n")
    try:
        rc = main([src, "--emit-ptx"])
        captured = capsys.readouterr()
    finally:
        if os.path.exists(src):
            os.remove(src)
    assert rc == 1, captured.out + captured.err
    assert "PTX emission requires at least one @kernel function" in captured.err
    assert "no @kernel fns" not in captured.out


def test_c119_emit_ptx_rejects_unsupported_hbm_float_dtype(capsys):
    src = write_src("""
    @kernel fn k(a: tile<f16, [256], HBM>) {
        let x = a[0];
        let y = x < 1.0_f16;
    }
    """)
    try:
        rc = main([src, "--emit-ptx"])
        captured = capsys.readouterr()
    finally:
        if os.path.exists(src):
            os.remove(src)
    assert rc == 1, captured.out + captured.err
    assert "@kernel HBM tile parameter dtype f16 is not supported" in captured.out
    assert "ld.global.f16" not in captured.out


def test_c119_emit_ptx_allows_folded_bool_constants(capsys):
    src = write_src("@kernel fn k() { let y = 1 < 2; }\n")
    try:
        rc = main([src, "--emit-ptx"])
        captured = capsys.readouterr()
    finally:
        if os.path.exists(src):
            os.remove(src)
    assert rc == 0, captured.err
    assert "mov.b32" in captured.out
    assert "unsupported PTX integer constant type bool" not in captured.err


def test_c119_emit_ptx_accepts_kernel_index_builtin(capsys):
    src = write_src("@kernel fn k() { let i = thread_idx(); }\n")
    try:
        rc = main([src, "--emit-ptx"])
        captured = capsys.readouterr()
    finally:
        if os.path.exists(src):
            os.remove(src)
    assert rc == 0, captured.err
    assert "%tid.x" in captured.out


def test_c119_emit_ptx_rejects_bare_kernel_index_builtin(capsys):
    src = write_src("@kernel fn k() { let i: i32 = thread_idx; }\n")
    try:
        rc = main([src, "--emit-ptx"])
        captured = capsys.readouterr()
    finally:
        if os.path.exists(src):
            os.remove(src)
    assert rc == 1, captured.out + captured.err
    assert "thread_idx must be called as thread_idx()" in captured.out
    assert "mov.b32" not in captured.out


def test_c119_emit_ptx_rejects_non_1d_hbm_params(capsys):
    src = write_src("@kernel fn k(a: tile<f32, [16, 16], HBM>) {}\n")
    try:
        rc = main([src, "--emit-ptx"])
        captured = capsys.readouterr()
    finally:
        if os.path.exists(src):
            os.remove(src)
    assert rc == 1, captured.out + captured.err
    assert "@kernel HBM tile parameters must be 1D" in captured.out
    assert "internal error" not in captured.err


def test_c119_emit_ptx_rejects_non_1d_extern_hbm_params(capsys):
    src = write_src('@kernel extern "C" fn k(a: tile<f32, [16, 16], HBM>);\n')
    try:
        rc = main([src, "--emit-ptx"])
        captured = capsys.readouterr()
    finally:
        if os.path.exists(src):
            os.remove(src)
    assert rc == 1, captured.out + captured.err
    assert "@kernel HBM tile parameters must be 1D" in captured.out
    assert "internal error" not in captured.err


def test_c119_emit_ptx_rejects_extern_only_kernels(capsys):
    src = write_src('@kernel extern "C" fn k(a: tile<f32, [16], HBM>);\n')
    try:
        rc = main([src, "--emit-ptx"])
        captured = capsys.readouterr()
    finally:
        if os.path.exists(src):
            os.remove(src)
    assert rc == 1, captured.out + captured.err
    assert "PTX emission requires at least one @kernel function" in captured.err
    assert ".visible .entry" not in captured.out


def test_c119_emit_ptx_rejects_non_unit_kernel_returns(capsys):
    src = write_src("@kernel fn k() -> i32 { 42 }\n")
    try:
        rc = main([src, "--emit-ptx"])
        captured = capsys.readouterr()
    finally:
        if os.path.exists(src):
            os.remove(src)
    assert rc == 1, captured.out + captured.err
    assert "@kernel functions must return ()" in captured.out
    assert "mov.b32" not in captured.out


def test_c119_emit_ptx_rejects_kernel_value_returns(capsys):
    src = write_src("@kernel fn k() { return 1; }\n")
    try:
        rc = main([src, "--emit-ptx"])
        captured = capsys.readouterr()
    finally:
        if os.path.exists(src):
            os.remove(src)
    assert rc == 1, captured.out + captured.err
    assert "@kernel functions cannot return a value" in captured.out
    assert "mov.b32" not in captured.out


def test_c117_backend_cli_aborts_on_struct_mono_errors(tmp_path):
    src_path = tmp_path / "bad_struct_mono.hx"
    out_path = tmp_path / "bad_struct_mono.bin"
    src_path.write_text(
        "struct Pt[T] { x: T }\n"
        "fn bad(p: Pt<i32, f64>) -> i32 { 0 }\n"
        "fn main() -> i32 { 42 }\n",
        encoding="utf-8",
    )
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.backend.x86_64",
         str(src_path), str(out_path), "--no-stdlib"],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "error: struct-mono:" in proc.stderr
    assert not out_path.exists(), "backend emitted a binary after struct-mono error"


def test_c117_backend_cli_panic_validation_no_traceback(tmp_path):
    src_path = tmp_path / "bad_panic.hx"
    out_path = tmp_path / "bad_panic.bin"
    src_path.write_text("fn main() -> i32 { panic(); 0 }\n", encoding="utf-8")
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.backend.x86_64",
         str(src_path), str(out_path), "--no-stdlib"],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "error: panic:" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert not out_path.exists(), "backend emitted a binary after panic validation error"


def test_c117_backend_cli_aborts_on_bad_compound_assignment(tmp_path):
    src_path = tmp_path / "bad_compound.hx"
    out_path = tmp_path / "bad_compound.bin"
    src_path.write_text(
        "fn main() -> i32 { let mut b: bool = true; b += false; 42 }\n",
        encoding="utf-8",
    )
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.backend.x86_64",
         str(src_path), str(out_path), "--no-stdlib"],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "operator '+' does not support operand type bool" in proc.stderr
    assert not out_path.exists(), (
        "backend emitted a binary after invalid compound assignment"
    )


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


def test_o1_invokes_backend_default_pass_order(monkeypatch, capsys, tmp_path):
    """Stage 31: check.py -O1 must mirror x86_64.py default pass order."""
    from helixc.ir.passes import const_fold, cse, dce, fdce
    calls = []
    real_fold = const_fold.fold_module
    real_cse = cse.cse_module
    real_dce = dce.dce_module
    real_fdce = fdce.fdce_module

    def counted(mod):
        calls.append("fold")
        return real_fold(mod)

    def cse_counted(mod):
        calls.append("cse")
        return real_cse(mod)

    def dce_counted(mod):
        calls.append("dce")
        return real_dce(mod)

    def fdce_counted(mod):
        calls.append("fdce")
        return real_fdce(mod)

    monkeypatch.setattr(const_fold, "fold_module", counted)
    monkeypatch.setattr(cse, "cse_module", cse_counted)
    monkeypatch.setattr(dce, "dce_module", dce_counted)
    monkeypatch.setattr(fdce, "fdce_module", fdce_counted)
    src_path = str(tmp_path / "fold.hx")
    with open(src_path, "w") as f:
        f.write("fn main() -> i32 { 1 + 2 }\n")
    rc = main([src_path, "--emit-ir", "-O1"])
    assert rc == 0, "compile must succeed"
    assert calls == ["fold", "cse", "dce", "fdce"]


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
    """Audit 28.8 A10: -O2 currently keeps the same cse/dce coverage as
    -O1. Pre-fix had a no-op try/import-pass placeholder; this guards the
    regression."""
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


def test_stage31_strict_without_emit_runs_effect_check(capsys, tmp_path):
    src_path = str(tmp_path / "strict_effect.hx")
    with open(src_path, "w") as f:
        f.write('fn main() -> i32 { print_str("Hello\\n"); 0 }\n')
    rc = main([src_path, "--strict"])
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert rc == 1, out
    assert "effect-check" in out
    assert "--strict aborts" in out
    assert "internal error" not in out


def test_stage31_refinement_violation_reports_clear_cli_diagnostic(capsys, tmp_path):
    src_path = str(tmp_path / "bad_refinement.hx")
    with open(src_path, "w") as f:
        f.write(
            "type Probability = f64 where 0.0 <= self <= 1.0;\n"
            "fn main() -> i32 {\n"
            "    let p: Probability = 1.2_f64;\n"
            "    42\n"
            "}\n"
        )
    rc = main([src_path, "--emit-ir", "-O0"])
    captured = capsys.readouterr()
    assert rc == 1, f"expected refinement violation to fail; stderr={captured.err!r}"
    out = captured.out + captured.err
    assert "refinement Probability violated" in out
    assert "1.2" in out
    assert "0.0 <= self <= 1.0" in out
    assert "Traceback" not in out


def test_stage31_valid_refinement_alias_erases_for_cli_ir(capsys, tmp_path):
    src_path = str(tmp_path / "valid_refinement.hx")
    with open(src_path, "w") as f:
        f.write(
            "type Probability = f64 where 0.0 <= self <= 1.0;\n"
            "fn main() -> i32 {\n"
            "    let p: Probability = 0.5_f64;\n"
            "    42\n"
            "}\n"
        )
    rc = main([src_path, "--emit-ir", "-O0"])
    captured = capsys.readouterr()
    assert rc == 0, (
        f"valid refined alias should typecheck and lower as its base type; "
        f"out={captured.out!r} err={captured.err!r}"
    )


def test_stage31_emit_proof_obligations_json_for_proved_refinement(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "proved_obligation.hx")
    with open(src_path, "w") as f:
        f.write(
            "type Probability = f64 where 0.0 <= self <= 1.0;\n"
            "fn main() -> i32 {\n"
            "    let p: Probability = 0.5_f64;\n"
            "    0\n"
            "}\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    artifact = json.loads(captured.out)
    assert artifact["schema"] == "helix.proof_obligations.v0"
    assert artifact["summary"]["typecheck_errors"] == 0
    assert artifact["summary"]["obligations"] == 1
    obligation = artifact["obligations"][0]
    assert obligation["kind"] == "refinement"
    assert obligation["context"] == "let 'p'"
    assert obligation["refinement"] == "Probability"
    assert obligation["predicate"] == "0.0 <= self <= 1.0"
    assert obligation["status"] == "proved"
    assert obligation["value"] == "0.5"
    assert "parse:" in captured.err


def test_stage31_emit_proof_obligations_json_for_failed_refinement(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "failed_obligation.hx")
    with open(src_path, "w") as f:
        f.write(
            "type Probability = f64 where 0.0 <= self <= 1.0;\n"
            "fn main() -> i32 {\n"
            "    let p: Probability = 1.2_f64;\n"
            "    0\n"
            "}\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] == 1
    assert len(artifact["typecheck_errors"]) == 1
    obligation = artifact["obligations"][0]
    assert obligation["refinement"] == "Probability"
    assert obligation["status"] == "failed"
    assert obligation["trap"] == "31001"
    assert obligation["value"] == "1.2"


def test_stage31_emit_proof_obligations_json_for_unproven_refinement(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "unproven_obligation.hx")
    with open(src_path, "w") as f:
        f.write(
            "type Probability = f64 where 0.0 <= self <= 1.0;\n"
            "fn use_raw(x: f64) -> i32 {\n"
            "    let p: Probability = x;\n"
            "    0\n"
            "}\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] == 1
    obligation = artifact["obligations"][0]
    assert obligation["refinement"] == "Probability"
    assert obligation["status"] == "unproven"
    assert "value" not in obligation
    assert "compile-time-proven value" in artifact["typecheck_errors"][0]


def test_stage31_emit_proof_obligations_includes_inherited_unproven_refinement(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "nested_unproven_obligation.hx")
    with open(src_path, "w") as f:
        f.write(
            "type Probability = f64 where 0.0 <= self <= 1.0;\n"
            "type Certain = Probability where self >= 0.9;\n"
            "fn use_raw(x: f64) -> i32 {\n"
            "    let c: Certain = x;\n"
            "    0\n"
            "}\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    observed = {
        (o["refinement"], o["predicate"], o["status"])
        for o in artifact["obligations"]
    }
    assert ("Certain", "self >= 0.9", "unproven") in observed
    assert ("Probability", "0.0 <= self <= 1.0", "unproven") in observed


def test_stage31_emit_proof_obligations_stdout_stays_json_with_ad_warning(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "proof_with_ad_warning.hx")
    with open(src_path, "w") as f:
        f.write(
            "fn loss(x: D<f64>, y: D<i32>) -> D<f64> { x + y }\n"
            "fn main() -> i32 { 0 }\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    artifact = json.loads(captured.out)
    assert artifact["schema"] == "helix.proof_obligations.v0"
    assert artifact["summary"]["typecheck_errors"] == 0
    assert "ad:" not in captured.out
    assert "ad:" in captured.err
    assert "24200" in captured.err or "AD002" in captured.err


def test_stage31_emit_proof_obligations_json_for_struct_mono_error(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "proof_struct_mono_error.hx")
    with open(src_path, "w") as f:
        f.write(
            "struct Pt[T] { x: T }\n"
            "fn bad(p: Pt<i32, f64>) -> i32 { 0 }\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert artifact["schema"] == "helix.proof_obligations.v0"
    assert artifact["summary"]["pipeline_errors"] == 1
    assert artifact["summary"]["typecheck_errors"] == 0
    assert artifact["pipeline_errors"][0]["phase"] == "struct-mono"
    assert "struct-mono" in artifact["pipeline_errors"][0]["message"]
    assert "struct-mono" in captured.err


def test_stage31_emit_proof_obligations_json_for_parse_error(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "proof_parse_error.hx")
    with open(src_path, "w") as f:
        f.write("fn main( -> i32 { 0 }\n")
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert artifact["summary"]["pipeline_errors"] == 1
    assert artifact["pipeline_errors"][0]["phase"] == "parse"
    assert "PARSE ERROR" in artifact["pipeline_errors"][0]["message"]
    assert "PARSE ERROR" in captured.err


def test_stage31_cli_default_stdlib_agi_safe_scalar_refinements(capsys, tmp_path):
    src_path = str(tmp_path / "stdlib_agi_scalars.hx")
    with open(src_path, "w") as f:
        f.write(
            "fn main() -> i32 {\n"
            "    let c: Confidence = 0.95_f64;\n"
            "    let p: Probability = 0.5_f64;\n"
            "    let d: DistanceMeters = 12.0_f64;\n"
            "    0\n"
            "}\n"
        )
    rc = main([src_path, "--check-only"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err


def test_stage31_cli_default_stdlib_confidence_violation(capsys, tmp_path):
    src_path = str(tmp_path / "stdlib_bad_confidence.hx")
    with open(src_path, "w") as f:
        f.write(
            "fn main() -> i32 {\n"
            "    let c: Confidence = 1.2_f64;\n"
            "    0\n"
            "}\n"
        )
    rc = main([src_path, "--check-only"])
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert rc == 1, out
    assert "refinement Confidence violated" in out
    assert "1.2" in out
    assert "0.0 <= self <= 1.0" in out


def test_stage31_cli_default_stdlib_probability_violation(capsys, tmp_path):
    src_path = str(tmp_path / "stdlib_bad_probability.hx")
    with open(src_path, "w") as f:
        f.write(
            "fn main() -> i32 {\n"
            "    let p: Probability = 1.2_f64;\n"
            "    0\n"
            "}\n"
        )
    rc = main([src_path, "--check-only"])
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert rc == 1, out
    assert "refinement Probability violated" in out
    assert "1.2" in out
    assert "0.0 <= self <= 1.0" in out


def test_stage31_cli_no_stdlib_requires_local_agi_scalar_aliases(capsys, tmp_path):
    src_path = str(tmp_path / "stdlib_scalars_disabled.hx")
    with open(src_path, "w") as f:
        f.write(
            "fn main() -> i32 {\n"
            "    let c: Confidence = 0.95_f64;\n"
            "    0\n"
            "}\n"
        )
    rc = main([src_path, "--check-only", "--no-stdlib"])
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert rc == 1, out
    assert "Confidence" in out
    assert "declare this type or import it before use" in out


def test_stage31_cli_checks_module_local_refinement_alias(capsys, tmp_path):
    src_path = str(tmp_path / "mod_refinement.hx")
    with open(src_path, "w") as f:
        f.write(
            "mod m {\n"
            "    type Probability = f64 where 0.0 <= self <= 1.0;\n"
            "    fn f() { let p: Probability = 1.2_f64; }\n"
            "}\n"
            "fn main() -> i32 { 0 }\n"
        )
    rc = main([src_path, "--emit-ir", "-O0"])
    captured = capsys.readouterr()
    assert rc == 1, (
        f"module-local refined alias violation should fail in CLI; "
        f"out={captured.out!r} err={captured.err!r}"
    )
    assert "refinement m__Probability violated" in (
        captured.out + captured.err)


def test_stage31_cli_emit_ir_runs_fn_mono_before_typecheck(capsys, tmp_path):
    src_path = str(tmp_path / "generic_alias_emit_ir.hx")
    with open(src_path, "w") as f:
        f.write(
            "type I = i32;\n"
            "fn id[T](x: T) -> T { x }\n"
            "fn main() -> i32 { id::<I>(42) }\n"
        )
    rc = main([src_path, "--emit-ir", "-O0"])
    captured = capsys.readouterr()
    assert rc == 0, (
        f"--emit-ir should mirror backend function-mono ordering; "
        f"out={captured.out!r} err={captured.err!r}"
    )
    assert "body type T does not match return type i32" not in (
        captured.out + captured.err)


def test_stage31_cli_emit_ir_includes_stdlib_by_default(capsys, tmp_path):
    src_path = str(tmp_path / "stdlib_default.hx")
    with open(src_path, "w") as f:
        f.write(
            "fn main() -> i32 {\n"
            "    let s = vec_new();\n"
            "    let n = vec_push(s, 0, 42);\n"
            "    vec_get(s, 0)\n"
            "}\n"
        )
    rc = main([src_path, "--emit-ir", "-O0"])
    captured = capsys.readouterr()
    assert rc == 0, (
        f"check.py should match backend's stdlib-by-default behavior; "
        f"out={captured.out!r} err={captured.err!r}"
    )

    rc_no_stdlib = main([src_path, "--emit-ir", "-O0", "--no-stdlib"])
    captured_no_stdlib = capsys.readouterr()
    assert rc_no_stdlib == 1
    no_stdlib_out = captured_no_stdlib.out + captured_no_stdlib.err
    assert "typecheck:" in no_stdlib_out
    assert "vec_" in no_stdlib_out


def test_stage31_cli_rejects_function_typed_call_before_codegen(capsys, tmp_path):
    src_path = str(tmp_path / "fn_typed_call.hx")
    with open(src_path, "w") as f:
        f.write(
            "type Positive = i32 where self > 0;\n"
            "struct Box[T] { v: T }\n"
            "type B = Box<Positive>;\n"
            "fn use_box(b: B) -> i32 { b.v }\n"
            "fn apply(fp: fn(B) -> i32, b: B) -> i32 { fp(b) }\n"
            "fn main(b: B) -> i32 { apply(use_box, b) }\n"
        )
    rc = main([src_path, "--no-stdlib", "--emit-ir", "-O0"])
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert rc == 1, out
    assert "function-typed calls are not supported by the Stage 31 backend" in out
    assert "internal error" not in out
    assert "compiler bug" not in out


def test_stage31_strict_o0_default_stdlib_prunes_unused(capsys, tmp_path):
    src_path = str(tmp_path / "strict_o0_stdlib.hx")
    with open(src_path, "w") as f:
        f.write("fn main() -> i32 { 42 }\n")
    rc = main([src_path, "--strict", "-O0"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err


def test_stage31_strict_o0_default_stdlib_checks_pub_without_main(capsys, tmp_path):
    src_path = str(tmp_path / "strict_o0_pub_no_main.hx")
    with open(src_path, "w") as f:
        f.write(
            '@pure pub fn bad() -> i32 { print_str("x"); 0 }\n'
        )
    rc = main([src_path, "--strict", "-O0"])
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert rc == 1, out
    assert "@pure function 'bad'" in out
    assert "actual effects {io}" in out


def test_stage31_strict_o0_default_stdlib_checks_private_without_main(capsys, tmp_path):
    src_path = str(tmp_path / "strict_o0_private_no_main.hx")
    with open(src_path, "w") as f:
        f.write(
            '@pure fn bad() -> i32 { print_str("x"); 0 }\n'
        )
    rc = main([src_path, "--strict", "-O0"])
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert rc == 1, out
    assert "@pure function 'bad'" in out
    assert "actual effects {io}" in out


def test_stage31_strict_default_stdlib_checks_private_dead_pure(capsys, tmp_path):
    src_path = str(tmp_path / "strict_dead_private_pure.hx")
    with open(src_path, "w") as f:
        f.write(
            '@pure fn bad() -> i32 { print_str("x"); 0 }\n'
            'fn main() -> i32 { 0 }\n'
        )
    rc = main([src_path, "--strict"])
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert rc == 1, out
    assert "@pure function 'bad'" in out
    assert "actual effects {io}" in out


def test_stage31_strict_default_stdlib_no_main_prunes_unused(capsys, tmp_path):
    src_path = str(tmp_path / "strict_default_stdlib_no_main.hx")
    with open(src_path, "w") as f:
        f.write("fn helper() -> i32 { 1 }\n")
    rc = main([src_path, "--strict"])
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert rc == 0, out
    assert "vec_push" not in out
    assert "effect-check warning" not in out


def test_stage31_emit_ir_o0_default_stdlib_prunes_unused_warnings(capsys, tmp_path):
    src_path = str(tmp_path / "emit_ir_o0_stdlib.hx")
    with open(src_path, "w") as f:
        f.write("fn main() -> i32 { 42 }\n")
    rc = main([src_path, "--emit-ir", "-O0"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    assert "warning: effect-check:" not in captured.err


def test_stage31_backend_strict_no_opt_default_stdlib_prunes_unused(tmp_path):
    src_path = tmp_path / "backend_strict_no_opt_stdlib.hx"
    out_path = tmp_path / "backend_strict_no_opt_stdlib.bin"
    src_path.write_text("fn main() -> i32 { 42 }\n", encoding="utf-8")
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.backend.x86_64",
         str(src_path), str(out_path), "--strict", "--no-opt"],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out_path.exists()


def test_stage31_backend_strict_default_stdlib_checks_private_dead_pure(tmp_path):
    src_path = tmp_path / "backend_strict_dead_private_pure.hx"
    out_path = tmp_path / "backend_strict_dead_private_pure.bin"
    src_path.write_text(
        '@pure fn bad() -> i32 { print_str("x"); 0 }\n'
        'fn main() -> i32 { 0 }\n',
        encoding="utf-8",
    )
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.backend.x86_64",
         str(src_path), str(out_path), "--strict"],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 1, out
    assert "@pure function 'bad'" in out
    assert "actual effects {io}" in out
    assert not out_path.exists()


def test_stage31_backend_no_opt_default_stdlib_prunes_unused_warnings(tmp_path):
    src_path = tmp_path / "backend_no_opt_stdlib.hx"
    out_path = tmp_path / "backend_no_opt_stdlib.bin"
    src_path.write_text("fn main() -> i32 { 42 }\n", encoding="utf-8")
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.backend.x86_64",
         str(src_path), str(out_path), "--no-opt"],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "warning: effect-check:" not in proc.stderr
    assert out_path.exists()


def test_stage31_check_py_lex_error_is_user_diagnostic(capsys, tmp_path):
    src_path = str(tmp_path / "bad_lex.hx")
    with open(src_path, "w") as f:
        f.write("fn main() -> i32 { # }\n")
    rc = main([src_path, "--check-only"])
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert rc == 1, out
    assert "LEX ERROR:" in out
    assert "unexpected character '#'" in out
    assert "internal error" not in out
    assert "compiler bug" not in out


def test_stage31_backend_lex_error_is_not_traceback(tmp_path):
    src_path = tmp_path / "bad_lex_backend.hx"
    out_path = tmp_path / "bad_lex_backend.bin"
    src_path.write_text("fn main() -> i32 { # }\n", encoding="utf-8")
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.backend.x86_64",
         str(src_path), str(out_path), "--no-stdlib"],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "error: lex:" in proc.stderr
    assert "unexpected character '#'" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert not out_path.exists()


def test_stage31_backend_flatten_error_is_not_traceback(tmp_path):
    src_path = tmp_path / "bad_flatten_backend.hx"
    out_path = tmp_path / "bad_flatten_backend.bin"
    src_path.write_text(
        "mod m { fn foo() -> i32 { 1 } }\n"
        "fn m__foo() -> i32 { 2 }\n"
        "fn main() -> i32 { m::foo() }\n",
        encoding="utf-8",
    )
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.backend.x86_64",
         str(src_path), str(out_path), "--no-stdlib"],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "error: mod-flatten:" in proc.stderr
    assert "trap 78001" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert not out_path.exists()


def test_stage31_check_only_strict_stays_frontend_only(capsys, tmp_path):
    src_path = str(tmp_path / "check_only_strict_effect.hx")
    with open(src_path, "w") as f:
        f.write(
            "@pure fn bad() -> i32 { print_str(\"x\"); 0 }\n"
            "fn main() -> i32 { bad() }\n"
        )
    rc = main([src_path, "--check-only", "--strict"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    assert "-- clean (check-only)" in captured.out


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
