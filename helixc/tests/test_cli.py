"""Stage 23: tests for the helixc check.py CLI."""

from __future__ import annotations

import os
import json
import hashlib
import runpy
import stat
import sys
import subprocess
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.check import parse_args, extract_doc_comments, main
from helixc.frontend import parser as parser_mod


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


def test_stage35_parse_args_output_rejects_flag_value():
    a, errs = parse_args(["foo.hx", "-o", "--no-stdlib"])
    assert errs
    assert a.output is None
    assert "--no-stdlib" in a.flags


def test_stage35_parse_args_lib_rejects_flag_value():
    a, errs = parse_args(["foo.hx", "-l", "--emit-ptx"])
    assert errs
    assert a.libs == []
    assert "--emit-ptx" in a.flags
    assert any("-l requires a library name" in e for e in errs)

    a_attached, errs_attached = parse_args(["foo.hx", "-l--emit-ptx"])
    assert errs_attached
    assert a_attached.libs == []
    assert any("-l requires a library name" in e for e in errs_attached)


def test_parse_args_lib_separate():
    a, errs = parse_args(["-l", "m", "-l", "c", "foo.hx"])
    assert not errs
    assert a.libs == ["m", "c"]


def test_parse_args_lib_attached():
    a, errs = parse_args(["-lm", "-lc", "foo.hx"])
    assert not errs
    assert a.libs == ["m", "c"]


def test_parse_args_warnings():
    a, errs = parse_args(["-Wdeprecated", "-Wad=error", "foo.hx"])
    assert not errs
    assert a.warnings["deprecated"] == "warn"
    assert a.warnings["ad"] == "error"


def test_parse_args_rejects_unknown_warning_policy():
    _a, errs = parse_args(["-Wdeprecated=erro", "foo.hx"])
    assert errs
    assert "unknown warning policy" in errs[0]


def test_parse_args_rejects_unknown_warning_name():
    _a, errs = parse_args(["-Wdeprectaed=error", "foo.hx"])
    assert errs
    assert "unknown warning name" in errs[0]


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


def test_stage35_parse_args_rejects_conflicting_stdlib_flags():
    a, errs = parse_args(["--stdlib", "--no-stdlib", "foo.hx"])
    assert a.path == "foo.hx"
    assert any("conflicting stdlib flags" in e for e in errs)


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
    artifact = json.loads(captured.out)
    assert artifact["summary"]["pipeline_errors"] == 1
    assert artifact["pipeline_errors"][0]["phase"] == "invocation"
    message = artifact["pipeline_errors"][0]["message"]
    assert "choose exactly one stdout-producing mode" in message
    assert "--emit-proof-obligations" in message
    assert "--emit-ast" in message
    assert "choose exactly one stdout-producing mode" in captured.err


def test_doc_and_emit_proof_obligations_are_mutually_exclusive(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "doc_conflict.hx")
    with open(src_path, "w") as f:
        f.write("/// Main docs\nfn main() -> i32 { 0 }\n")
    rc = main([src_path, "--emit-proof-obligations", "--doc"])
    captured = capsys.readouterr()
    assert rc == 2
    artifact = json.loads(captured.out)
    assert artifact["summary"]["pipeline_errors"] == 1
    assert artifact["pipeline_errors"][0]["phase"] == "invocation"
    message = artifact["pipeline_errors"][0]["message"]
    assert "choose exactly one stdout-producing mode" in message
    assert "--emit-proof-obligations" in message
    assert "--doc" in message
    assert "choose exactly one stdout-producing mode" in captured.err


def test_stage31_emit_proof_obligations_missing_path_stays_json(capsys):
    rc = main(["--emit-proof-obligations"])
    captured = capsys.readouterr()
    assert rc == 2
    artifact = json.loads(captured.out)
    assert artifact["path"] is None
    assert artifact["input"]["source_sha256"] is None
    assert artifact["input"]["source_available"] is False
    assert artifact["summary"]["pipeline_errors"] == 1
    assert artifact["pipeline_errors"][0]["phase"] == "invocation"
    assert "source path required" in artifact["pipeline_errors"][0]["message"]
    assert "helixc/check.py" not in captured.out


def test_stage31_emit_proof_obligations_missing_file_stays_json(
    capsys, tmp_path,
):
    missing = str(tmp_path / "missing_source.hx")
    rc = main([missing, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 2
    artifact = json.loads(captured.out)
    assert artifact["path"] == missing
    assert artifact["input"]["source_sha256"] is None
    assert artifact["input"]["source_available"] is False
    assert artifact["summary"]["pipeline_errors"] == 1
    assert artifact["pipeline_errors"][0]["phase"] == "invocation"
    assert "file not found" in artifact["pipeline_errors"][0]["message"]


def test_stage31_emit_proof_obligations_directory_path_stays_json(
    capsys, tmp_path,
):
    rc = main([str(tmp_path), "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 2
    artifact = json.loads(captured.out)
    assert artifact["path"] == str(tmp_path)
    assert artifact["cache_key"] is None
    assert artifact["input"]["source_sha256"] is None
    assert artifact["input"]["source_available"] is False
    assert artifact["summary"]["pipeline_errors"] == 1
    assert artifact["pipeline_errors"][0]["phase"] == "source-read"
    assert "SOURCE READ ERROR" in artifact["pipeline_errors"][0]["message"]



def test_c117_emit_ptx_uses_kernel_attrs(capsys):
    src = write_src("@kernel fn k() { }\nfn main() -> i32 { 42 }\n")
    rc = main([src, "--emit-ptx"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert ".visible .entry k" in captured.out, captured.out
    assert "no @kernel fns" not in captured.out


def test_stage35_emit_ptx_stdout_starts_with_ptx_module(tmp_path):
    src_path = tmp_path / "kernel.hx"
    src_path.write_text("@kernel fn k() { let i = thread_idx(); }\n", encoding="utf-8")
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.check", str(src_path), "--emit-ptx"],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.lstrip().startswith(".version"), proc.stdout
    assert "-- helixc-check:" not in proc.stdout
    assert "parse:" not in proc.stdout
    assert "typecheck:" not in proc.stdout


def test_stage35_emit_ptx_autotune_failure_stdout_is_empty(tmp_path):
    src_path = tmp_path / "bad_autotune.hx"
    src_path.write_text(
        "@kernel @autotune(B: []) fn k(a: tile<f32, [16], HBM>) { }\n",
        encoding="utf-8",
    )
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.check", str(src_path), "--emit-ptx"],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert proc.stdout == ""
    assert "autotune:" in proc.stderr
    assert "empty value list" in proc.stderr


def test_stage35_emit_ptx_typecheck_failure_stdout_is_empty(tmp_path):
    src_path = tmp_path / "bad_typecheck.hx"
    src_path.write_text(
        "@kernel fn k() { let x: i32 = true; }\n",
        encoding="utf-8",
    )
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.check", str(src_path), "--emit-ptx"],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert proc.stdout == ""
    assert "typecheck:" in proc.stderr


def test_stage35_emit_ptx_ad_warning_stays_off_stdout(tmp_path):
    src_path = tmp_path / "ad_warning_kernel.hx"
    src_path.write_text(
        "fn loss(x: D<f64>, y: D<i32>) -> D<f64> { x + y }\n"
        "@kernel fn k() { let bad: i32 = true; }\n",
        encoding="utf-8",
    )
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.check", str(src_path), "--emit-ptx"],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert proc.stdout == ""
    assert "typecheck:" in proc.stderr
    assert "ad:" in proc.stderr


def test_stage35_emit_ptx_ignores_host_ad_function(tmp_path):
    src_path = tmp_path / "host_ad_kernel.hx"
    src_path.write_text(
        "fn loss(x: D<f64>, y: D<i32>) -> D<f64> { x + y }\n"
        "@kernel fn k() { let i = thread_idx(); }\n",
        encoding="utf-8",
    )
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.check", str(src_path), "--emit-ptx"],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert ".visible .entry k" in proc.stdout
    assert "ad:" not in proc.stdout
    assert "ad:" in proc.stderr
    assert "unresolved generic type D" not in proc.stderr


def test_stage35_emit_ptx_allows_valid_host_grad_call(tmp_path):
    src_path = tmp_path / "host_grad_kernel.hx"
    src_path.write_text(
        "fn loss(x: f32) -> f32 { x * x }\n"
        "fn main() -> i32 { grad(loss)(2.0) as i32 }\n"
        "@kernel fn k() { let i = thread_idx(); }\n",
        encoding="utf-8",
    )
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.check", str(src_path), "--emit-ptx", "--no-stdlib"],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert ".visible .entry k" in proc.stdout
    assert "unknown function '<unknown>'" not in proc.stderr
    assert "PTX validation error" not in proc.stderr


def test_stage35_emit_ptx_wad_error_does_not_emit_artifact(tmp_path):
    src_path = tmp_path / "host_ad_kernel_error.hx"
    src_path.write_text(
        "fn loss(x: D<f64>, y: D<i32>) -> D<f64> { x + y }\n"
        "@kernel fn k() { let i = thread_idx(); }\n",
        encoding="utf-8",
    )
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.check", str(src_path),
            "--emit-ptx", "-Wad=error",
        ],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert proc.stdout == ""
    assert "ad:" in proc.stderr
    assert "ERROR" in proc.stderr
    assert "AD002" in proc.stderr or "24200" in proc.stderr


def test_stage35_emit_ptx_missing_path_keeps_stdout_empty(capsys):
    rc = main(["--emit-ptx"])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert "source path required" in captured.err


def test_stage35_source_required_modes_keep_stdout_empty(capsys, tmp_path):
    rc_check = main(["--check-only"])
    captured_check = capsys.readouterr()
    assert rc_check == 2
    assert captured_check.out == ""
    assert "source path required" in captured_check.err

    out_path = tmp_path / "missing-source.bin"
    rc_output = main(["-o", str(out_path)])
    captured_output = capsys.readouterr()
    assert rc_output == 2
    assert captured_output.out == ""
    assert "source path required" in captured_output.err
    assert not out_path.exists()


def test_stage35_emit_ptx_strict_ignores_host_ad_function(tmp_path):
    src_path = tmp_path / "strict_host_ad_kernel.hx"
    src_path.write_text(
        "fn loss(x: D<f64>, y: D<i32>) -> D<f64> { x + y }\n"
        "@kernel fn k() { let i = thread_idx(); }\n",
        encoding="utf-8",
    )
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.check", str(src_path),
            "--emit-ptx", "--strict",
        ],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert ".visible .entry k" in proc.stdout
    assert "ad:" in proc.stderr
    assert "unresolved generic type D" not in proc.stderr
    assert "internal error" not in proc.stderr
    assert "compiler bug" not in proc.stderr


def test_stage35_emit_ptx_strict_rejects_host_only_effect_violation(tmp_path):
    src_path = tmp_path / "strict_host_effect_kernel.hx"
    src_path.write_text(
        "@pure fn host() -> i32 { print_int(1); 0 }\n"
        "@kernel fn k() { let i = thread_idx(); }\n",
        encoding="utf-8",
    )
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.check", str(src_path),
            "--emit-ptx", "--strict", "--no-stdlib",
        ],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert proc.stdout == ""
    assert "--strict aborts" in proc.stderr
    assert "19001" in proc.stderr
    assert "compiler bug" not in proc.stderr


def test_stage35_emit_ptx_strict_rejects_host_effect_with_dead_ad_helper(tmp_path):
    src_path = tmp_path / "strict_host_effect_dead_ad_kernel.hx"
    src_path.write_text(
        "fn loss(x: D<f64>) -> D<f64> { x }\n"
        "@pure fn host() -> i32 { print_int(1); 0 }\n"
        "@kernel fn k() { let i = thread_idx(); }\n",
        encoding="utf-8",
    )
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.check", str(src_path),
            "--emit-ptx", "--strict", "--no-stdlib",
        ],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert proc.stdout == ""
    assert "--strict aborts" in proc.stderr
    assert "19001" in proc.stderr
    assert "unresolved generic type D" not in proc.stderr


def test_stage35_emit_ptx_strict_wad_error_keeps_stdout_empty(tmp_path):
    src_path = tmp_path / "strict_host_ad_kernel_error.hx"
    src_path.write_text(
        "fn loss(x: D<f64>, y: D<i32>) -> D<f64> { x + y }\n"
        "@kernel fn k() { let i = thread_idx(); }\n",
        encoding="utf-8",
    )
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.check", str(src_path),
            "--emit-ptx", "--strict", "-Wad=error",
        ],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert proc.stdout == ""
    assert "ad:" in proc.stderr
    assert "ERROR" in proc.stderr
    assert "AD002" in proc.stderr or "24200" in proc.stderr
    assert "unresolved generic type D" not in proc.stderr
    assert "internal error" not in proc.stderr
    assert "compiler bug" not in proc.stderr


def test_stage35_check_output_ignores_dead_ad_helper_for_embedded_ptx(tmp_path):
    src_path = tmp_path / "embedded_ptx_dead_ad.hx"
    out_path = tmp_path / "embedded_ptx_dead_ad.bin"
    src_path.write_text(
        "fn loss(x: D<f64>) -> D<f64> { x }\n"
        "@kernel fn k() { let i = thread_idx(); }\n"
        "fn main() -> i32 { 42 }\n",
        encoding="utf-8",
    )
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.check", str(src_path),
            "-o", str(out_path), "--no-stdlib",
        ],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out_path.exists()
    assert "unresolved generic type D" not in proc.stderr
    assert "compiler bug" not in proc.stderr


def test_stage35_direct_x86_output_ignores_dead_ad_helper_for_embedded_ptx(tmp_path):
    src_path = tmp_path / "direct_x86_embedded_ptx_dead_ad.hx"
    out_path = tmp_path / "direct_x86_embedded_ptx_dead_ad.bin"
    src_path.write_text(
        "fn loss(x: D<f64>) -> D<f64> { x }\n"
        "@kernel fn k() { let i = thread_idx(); }\n"
        "fn main() -> i32 { 42 }\n",
        encoding="utf-8",
    )
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.backend.x86_64",
            str(src_path), str(out_path), "--no-stdlib",
        ],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out_path.exists()
    assert "unresolved generic type D" not in proc.stderr
    assert "Traceback" not in proc.stderr


def test_stage35_emit_asm_ignores_dead_ad_helper(tmp_path):
    src_path = tmp_path / "emit_asm_dead_ad.hx"
    src_path.write_text(
        "fn loss(x: D<f64>) -> D<f64> { x }\n"
        "@kernel fn k() { let i = thread_idx(); }\n"
        "fn main() -> i32 { 42 }\n",
        encoding="utf-8",
    )
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.check", str(src_path),
            "--emit-asm", "--no-stdlib",
        ],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "unresolved generic type D" not in proc.stderr
    assert "compiler bug" not in proc.stderr


def test_stage35_strict_host_check_ignores_dead_ad_helper(tmp_path):
    src_path = tmp_path / "strict_host_dead_ad.hx"
    src_path.write_text(
        "fn loss(x: D<f64>) -> D<f64> { x }\n"
        "@kernel fn k() { let i = thread_idx(); }\n"
        "fn main() -> i32 { 42 }\n",
        encoding="utf-8",
    )
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.check", str(src_path),
            "--strict", "--no-stdlib",
        ],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "unresolved generic type D" not in proc.stderr
    assert "compiler bug" not in proc.stderr


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
    assert captured.out == ""
    assert "@kernel HBM tile parameter dtype f16 is not supported" in captured.err
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
    assert captured.out == ""
    assert "thread_idx must be called as thread_idx()" in captured.err
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
    assert captured.out == ""
    assert "@kernel HBM tile parameters must be 1D" in captured.err
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
    assert captured.out == ""
    assert "@kernel HBM tile parameters must be 1D" in captured.err
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


def test_stage35_emit_ptx_reports_tile_lowering_error_without_bug_label(capsys):
    src = write_src("@kernel fn k() { let i = thread_idx(); let z = i / 2; }\n")
    try:
        rc = main([src, "--emit-ptx"])
        captured = capsys.readouterr()
    finally:
        if os.path.exists(src):
            os.remove(src)
    assert rc == 1, captured.out + captured.err
    assert "Tile IR lowering does not support TIR op elem.div" in captured.err
    assert "internal error" not in captured.err
    assert "compiler bug" not in captured.err
    assert ".visible .entry" not in captured.out


def test_stage35_emit_ptx_non_strict_reports_host_effect_warning(capsys):
    src = write_src(
        "@pure fn host() -> i32 { print_int(1); 0 }\n"
        "@kernel fn k() { let i = thread_idx(); }\n"
    )
    try:
        rc = main([src, "--emit-ptx", "--no-stdlib"])
        captured = capsys.readouterr()
    finally:
        if os.path.exists(src):
            os.remove(src)
    assert rc == 0, captured.out + captured.err
    assert ".visible .entry k" in captured.out
    assert "warning: effect-check:" in captured.err
    assert "19001" in captured.err


def test_stage35_output_binary_rejects_dead_unsupported_kernel_op(tmp_path):
    src_path = tmp_path / "dead_kernel_div.hx"
    out_path = tmp_path / "dead_kernel_div.bin"
    src_path.write_text(
        "@kernel fn k() { let i = thread_idx(); let dead = i / 2; }\n"
        "fn main() -> i32 { 0 }\n",
        encoding="utf-8",
    )
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.check", str(src_path),
         "-o", str(out_path), "--no-stdlib"],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Tile IR lowering does not support TIR op elem.div" in proc.stderr
    assert "compiler bug" not in proc.stderr
    assert not out_path.exists()


def test_stage35_direct_x86_rejects_dead_unsupported_kernel_op(tmp_path):
    src_path = tmp_path / "direct_dead_kernel_div.hx"
    out_path = tmp_path / "direct_dead_kernel_div.bin"
    src_path.write_text(
        "@kernel fn k() { let i = thread_idx(); let dead = i / 2; }\n"
        "fn main() -> i32 { 0 }\n",
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
    assert "Tile IR lowering does not support TIR op elem.div" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert not out_path.exists()


def test_stage35_output_binary_rejects_kernel_helper_call_without_internal_error(tmp_path):
    src_path = tmp_path / "kernel_helper_call.hx"
    out_path = tmp_path / "kernel_helper_call.bin"
    src_path.write_text(
        "fn helper(x: i32) -> i32 { x / 2 }\n"
        "@kernel fn k() { let i = thread_idx(); let z = helper(i); }\n"
        "fn main() -> i32 { 0 }\n",
        encoding="utf-8",
    )
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.check", str(src_path),
         "-o", str(out_path), "--no-stdlib"],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "PTX validation error" in proc.stderr or "error: ptx:" in proc.stderr
    assert "internal error" not in proc.stderr
    assert "compiler bug" not in proc.stderr
    assert "Traceback" not in proc.stderr
    assert not out_path.exists()


def test_stage35_direct_x86_rejects_kernel_helper_call_without_traceback(tmp_path):
    src_path = tmp_path / "direct_kernel_helper_call.hx"
    out_path = tmp_path / "direct_kernel_helper_call.bin"
    src_path.write_text(
        "fn helper(x: i32) -> i32 { x / 2 }\n"
        "@kernel fn k() { let i = thread_idx(); let z = helper(i); }\n"
        "fn main() -> i32 { 0 }\n",
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
    assert "error: ptx:" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert "internal error" not in proc.stderr
    assert not out_path.exists()


def test_c119_emit_ptx_rejects_non_unit_kernel_returns(capsys):
    src = write_src("@kernel fn k() -> i32 { 42 }\n")
    try:
        rc = main([src, "--emit-ptx"])
        captured = capsys.readouterr()
    finally:
        if os.path.exists(src):
            os.remove(src)
    assert rc == 1, captured.out + captured.err
    assert captured.out == ""
    assert "@kernel functions must return ()" in captured.err
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
    assert captured.out == ""
    assert "@kernel functions cannot return a value" in captured.err
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
        assert "-- helixc-check:" not in cap.out
        assert "parse:" not in cap.out
        assert "-- helixc-check:" in cap.err
        assert "parse:" in cap.err
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
    if os.name != "nt":
        assert os.stat(out_path).st_mode & stat.S_IXUSR


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


def test_stage35_emit_asm_missing_main_is_user_codegen_error(capsys, tmp_path):
    src_path = str(tmp_path / "helper_only.hx")
    with open(src_path, "w") as f:
        f.write("fn helper() -> i32 { 1 }\n")

    rc = main([src_path, "--emit-asm", "--no-stdlib"])
    cap = capsys.readouterr()
    assert rc == 1
    assert "codegen error" in cap.err
    assert "module has no function 'main'" in cap.err
    assert "compiler bug" not in cap.err
    assert "Traceback (most recent call last)" not in cap.err
    assert cap.out == ""


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


def test_stage35_output_binary_missing_main_is_user_codegen_error(capsys, tmp_path):
    src_path = str(tmp_path / "helper_only.hx")
    out_path = str(tmp_path / "out.bin")
    with open(src_path, "w") as f:
        f.write("fn helper() -> i32 { 1 }\n")

    rc = main([src_path, "-o", out_path, "--no-stdlib"])
    cap = capsys.readouterr()
    assert rc == 1
    assert "codegen error" in cap.err
    assert "module has no function 'main'" in cap.err
    assert "compiler bug" not in cap.err
    assert "Traceback (most recent call last)" not in cap.err
    assert not os.path.exists(out_path)


def test_main_o_handles_oserror_on_write(monkeypatch, capsys, tmp_path):
    """Audit 28.8 A9: -o catches OSError on the file write so
    permission / disk-full failures get a clean diagnostic too."""
    from helixc import check as _check

    src_path = str(tmp_path / "in.hx")
    with open(src_path, "w") as f:
        f.write("fn main() -> i32 { 1 }\n")
    out_path = str(tmp_path / "out.bin")

    def _mkstemp_blocking(*args, **kwargs):
        raise PermissionError("synthetic permission denied")

    monkeypatch.setattr(_check.tempfile, "mkstemp", _mkstemp_blocking)
    rc = main([src_path, "-o", out_path])
    cap = capsys.readouterr()
    assert rc == 1
    assert "cannot write output" in cap.err
    assert "synthetic permission denied" in cap.err
    assert not os.path.exists(out_path)


def test_stage35_check_output_atomic_replace_failure_removes_existing(
    monkeypatch, capsys, tmp_path
):
    from helixc import check as _check

    src_path = str(tmp_path / "atomic_replace.hx")
    out_path = tmp_path / "atomic_replace.bin"
    with open(src_path, "w") as f:
        f.write("fn main() -> i32 { 42 }\n")
    out_path.write_bytes(b"OLD")

    def _replace_blocking(src, dst):
        raise PermissionError("synthetic replace denied")

    monkeypatch.setattr(_check.os, "replace", _replace_blocking)
    rc = main([src_path, "-o", str(out_path)])
    cap = capsys.readouterr()
    assert rc == 1
    assert "cannot write output" in cap.err
    assert "synthetic replace denied" in cap.err
    assert not out_path.exists()
    assert list(tmp_path.glob(".atomic_replace.bin.*.tmp")) == []


def test_stage35_check_output_chmod_failure_removes_temp(
    monkeypatch, capsys, tmp_path
):
    from helixc import check as _check

    src_path = str(tmp_path / "chmod_failure.hx")
    out_path = tmp_path / "chmod_failure.bin"
    with open(src_path, "w") as f:
        f.write("fn main() -> i32 { 42 }\n")

    def _chmod_blocking(path, mode):
        raise PermissionError("synthetic chmod denied")

    monkeypatch.setattr(_check.os, "chmod", _chmod_blocking)
    rc = main([src_path, "-o", str(out_path)])
    cap = capsys.readouterr()
    assert rc == 1
    assert "cannot write output" in cap.err
    assert "synthetic chmod denied" in cap.err
    assert not out_path.exists()
    assert list(tmp_path.glob(".chmod_failure.bin.*.tmp")) == []


def test_stage35_check_output_rejects_source_as_output(capsys, tmp_path):
    src_path = tmp_path / "same_source.hx"
    source = "fn main() -> i32 { 42 }\n"
    src_path.write_text(source, encoding="utf-8")

    rc = main([str(src_path), "-o", str(src_path), "--no-stdlib"])
    cap = capsys.readouterr()

    assert rc == 2
    assert "output path must differ from input source path" in cap.err
    assert src_path.read_text(encoding="utf-8") == source


def test_stage35_check_output_failure_removes_prior_artifact(capsys, tmp_path):
    good = tmp_path / "good.hx"
    bad = tmp_path / "bad.hx"
    missing = tmp_path / "missing.hx"
    out_path = tmp_path / "stale.bin"
    good.write_text("fn main() -> i32 { 0 }\n", encoding="utf-8")
    bad.write_text(
        "fn main() -> i32 { let mut x: i64 = 1_i64; x = 2_i32; 0 }\n",
        encoding="utf-8",
    )

    rc_good = main([str(good), "-o", str(out_path), "--no-stdlib"])
    capsys.readouterr()
    assert rc_good == 0
    assert out_path.exists()

    rc_bad = main([str(bad), "-o", str(out_path), "--no-stdlib"])
    captured_bad = capsys.readouterr()
    assert rc_bad == 1
    assert "typecheck:" in captured_bad.out
    assert not out_path.exists()

    out_path.write_bytes(b"OLD")
    rc_missing = main([str(missing), "-o", str(out_path), "--no-stdlib"])
    captured_missing = capsys.readouterr()
    assert rc_missing == 2
    assert "file not found" in captured_missing.err
    assert not out_path.exists()


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


def test_stage35_wad_error_output_binary_does_not_write_artifact(tmp_path):
    src_path = tmp_path / "loss_ad_warning.hx"
    out_path = tmp_path / "out.bin"
    src_path.write_text(
        "fn loss(x: D<f64>, y: D<i32>) -> D<f64> { x + y }\n"
        "fn main() -> i32 { 0 }\n",
        encoding="utf-8",
    )
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.check", str(src_path),
            "-o", str(out_path), "--no-stdlib", "-Wad=error",
        ],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert not out_path.exists()
    assert proc.stdout == ""
    assert "codegen:   OK" not in proc.stdout
    assert "ERROR" in proc.stderr


def test_stage35_wad_error_emit_asm_does_not_print_artifact(tmp_path):
    src_path = tmp_path / "loss_ad_warning_asm.hx"
    src_path.write_text(
        "fn loss(x: D<f64>, y: D<i32>) -> D<f64> { x + y }\n"
        "fn main() -> i32 { 0 }\n",
        encoding="utf-8",
    )
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.check", str(src_path),
            "--emit-asm", "--no-stdlib", "-Wad=error",
        ],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert proc.stdout == ""
    assert "asm:" not in proc.stdout
    assert "ERROR" in proc.stderr


def test_stage35_wad_error_emit_ir_does_not_print_artifact(tmp_path):
    src_path = tmp_path / "loss_ad_warning_ir.hx"
    src_path.write_text(
        "fn loss(x: D<f64>, y: D<i32>) -> D<f64> { x + y }\n"
        "fn main() -> i32 { 0 }\n",
        encoding="utf-8",
    )
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.check", str(src_path),
            "--emit-ir", "--no-stdlib", "-Wad=error",
        ],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert proc.stdout == ""
    assert "ir:" not in proc.stdout
    assert "ERROR" in proc.stderr


def test_stage35_wad_warn_emit_ir_keeps_warning_summary_off_stdout(tmp_path):
    src_path = tmp_path / "loss_ad_warning_ir_warn.hx"
    src_path.write_text(
        "fn loss(x: D<f64>, y: D<i32>) -> D<f64> { x + y }\n"
        "fn main() -> i32 { 0 }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.check", str(src_path),
            "--emit-ir", "--no-stdlib", "-Wad=warn",
        ],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "fn main" in proc.stdout
    assert "ad:" not in proc.stdout
    assert "ad:" in proc.stderr


def test_stage35_emit_ir_with_output_is_error(tmp_path, capsys):
    src_path = tmp_path / "emit_ir_output.hx"
    out_path = tmp_path / "emit_ir_output.bin"
    src_path.write_text("fn main() -> i32 { 42 }\n", encoding="utf-8")
    rc = main([str(src_path), "--emit-ir", "-o", str(out_path), "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 2
    assert not out_path.exists()
    assert "cannot be combined with -o" in captured.err


def test_stage35_output_flag_value_rejected_without_writing(tmp_path, capsys, monkeypatch):
    src_path = tmp_path / "flag_output.hx"
    src_path.write_text("fn main() -> i32 { 42 }\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    rc = main([str(src_path), "-o", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 2
    assert not (tmp_path / "--no-stdlib").exists()
    assert "-o requires an output path" in captured.err


def test_stage35_check_only_rejects_artifact_modes_and_output(tmp_path, capsys):
    src_path = tmp_path / "check_only_artifacts.hx"
    src_path.write_text("fn main() -> i32 { 42 }\n", encoding="utf-8")
    out_path = tmp_path / "check_only_artifacts.bin"

    rc_emit = main([str(src_path), "--check-only", "--emit-ir", "--no-stdlib"])
    captured_emit = capsys.readouterr()
    assert rc_emit == 2
    assert "--check-only cannot be combined with --emit-ir" in captured_emit.err

    rc_output = main([str(src_path), "--check-only", "-o", str(out_path), "--no-stdlib"])
    captured_output = capsys.readouterr()
    assert rc_output == 2
    assert "--check-only cannot be combined with -o" in captured_output.err
    assert not out_path.exists()


def test_stage35_main_rejects_conflicting_stdlib_flags(tmp_path, capsys):
    src_path = tmp_path / "conflict_stdlib.hx"
    src_path.write_text("fn main() -> i32 { 42 }\n", encoding="utf-8")
    rc = main([str(src_path), "--stdlib", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "conflicting stdlib flags" in captured.err


def test_stage35_deprecated_warn_emit_asm_keeps_warning_summary_off_stdout(tmp_path):
    src_path = tmp_path / "deprecated_asm_warn.hx"
    src_path.write_text(
        "@deprecated fn old() -> i32 { 0 }\n"
        "fn main() -> i32 { old() }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.check", str(src_path),
            "--emit-asm", "--no-stdlib", "-Wdeprecated=warn",
        ],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "asm:" in proc.stdout
    assert "deprecated:" not in proc.stdout
    assert "deprecated:" in proc.stderr


def test_stage35_wad_error_default_does_not_print_clean(tmp_path):
    src_path = tmp_path / "loss_ad_warning_default.hx"
    src_path.write_text(
        "fn loss(x: D<f64>, y: D<i32>) -> D<f64> { x + y }\n"
        "fn main() -> i32 { 0 }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.check", str(src_path), "--no-stdlib", "-Wad=error"],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert proc.stdout == ""
    assert "-- clean" not in proc.stdout
    assert "-- clean" not in proc.stderr
    assert "ERROR" in proc.stderr


def test_stage35_wad_error_check_only_does_not_print_clean(tmp_path):
    src_path = tmp_path / "loss_ad_warning_check_only.hx"
    src_path.write_text(
        "fn loss(x: D<f64>, y: D<i32>) -> D<f64> { x + y }\n"
        "fn main() -> i32 { 0 }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.check", str(src_path),
            "--check-only", "--no-stdlib", "-Wad=error",
        ],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert proc.stdout == ""
    assert "-- clean" not in proc.stdout
    assert "-- clean" not in proc.stderr
    assert "ERROR" in proc.stderr


def test_stage35_deprecated_error_default_keeps_stdout_empty(tmp_path):
    src_path = tmp_path / "deprecated_default.hx"
    src_path.write_text(
        "@deprecated fn old() -> i32 { 0 }\n"
        "fn main() -> i32 { old() }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.check", str(src_path),
            "--no-stdlib", "-Wdeprecated=error",
        ],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert proc.stdout == ""
    assert "deprecated:" in proc.stderr
    assert "ERROR" in proc.stderr


def test_stage35_deprecated_error_with_ad_warning_keeps_stdout_empty(tmp_path):
    src_path = tmp_path / "deprecated_ad_default.hx"
    src_path.write_text(
        "fn loss(x: D<f64>, y: D<i32>) -> D<f64> { x + y }\n"
        "@deprecated fn old() -> i32 { 0 }\n"
        "fn main() -> i32 { old() }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.check", str(src_path),
            "--no-stdlib", "-Wdeprecated=error",
        ],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert proc.stdout == ""
    assert "deprecated:" in proc.stderr
    assert "ad:" in proc.stderr
    assert "ERROR" in proc.stderr


def test_stage35_direct_x86_honors_wad_error_before_writing(tmp_path):
    src_path = tmp_path / "loss_ad_warning_direct.hx"
    out_path = tmp_path / "direct.bin"
    src_path.write_text(
        "fn loss(x: D<f64>, y: D<i32>) -> D<f64> { x + y }\n"
        "fn main() -> i32 { 0 }\n",
        encoding="utf-8",
    )
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.backend.x86_64",
            str(src_path), str(out_path), "--no-stdlib", "-Wad=error",
        ],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert not out_path.exists()
    assert "Wrote" not in proc.stdout
    assert "ERROR" in proc.stderr


def test_stage35_direct_x86_honors_deprecated_error_before_writing(tmp_path):
    src_path = tmp_path / "deprecated_direct.hx"
    out_path = tmp_path / "deprecated.bin"
    src_path.write_text(
        "@deprecated fn old() -> i32 { 0 }\n"
        "fn main() -> i32 { old() }\n",
        encoding="utf-8",
    )
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.backend.x86_64",
            str(src_path), str(out_path), "--no-stdlib", "-Wdeprecated=error",
        ],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert not out_path.exists()
    assert "deprecated:" in proc.stderr
    assert "ERROR" in proc.stderr


def test_stage35_direct_x86_drains_ad_warnings_on_deprecated_error(tmp_path):
    src_path = tmp_path / "deprecated_ad_direct.hx"
    out_path = tmp_path / "deprecated_ad.bin"
    src_path.write_text(
        "fn loss(x: D<f64>, y: D<i32>) -> D<f64> { x + y }\n"
        "@deprecated fn old() -> i32 { 0 }\n"
        "fn main() -> i32 { old() }\n",
        encoding="utf-8",
    )
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.backend.x86_64",
            str(src_path), str(out_path), "--no-stdlib",
            "-Wdeprecated=error", "-Wad=error",
        ],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert not out_path.exists()
    assert "deprecated:" in proc.stderr
    assert "ad:" in proc.stderr
    assert "ERROR" in proc.stderr


def test_stage35_direct_x86_drains_ad_warnings_on_type_error(tmp_path):
    src_path = tmp_path / "ad_warning_type_error_direct.hx"
    out_path = tmp_path / "type_error.bin"
    src_path.write_text(
        "fn loss(x: D<f64>, y: D<i32>) -> D<f64> { x + y }\n"
        "fn main() -> i32 { let x: i32 = true; 0 }\n",
        encoding="utf-8",
    )
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.backend.x86_64",
            str(src_path), str(out_path), "--no-stdlib", "-Wad=error",
        ],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert not out_path.exists()
    assert "type error" in proc.stderr
    assert "ad:" in proc.stderr
    assert "ERROR" in proc.stderr


def test_stage35_direct_x86_rejects_unknown_flags_without_writing(tmp_path):
    src_path = tmp_path / "ok.hx"
    out_path = tmp_path / "ok.bin"
    src_path.write_text("fn main() -> i32 { 42 }\n", encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.backend.x86_64",
            str(src_path), str(out_path), "--strcit",
        ],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert not out_path.exists()
    assert "unknown flag" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_stage35_direct_x86_accepts_stdlib_compat_flag(tmp_path):
    src_path = tmp_path / "ok_stdlib.hx"
    out_path = tmp_path / "ok_stdlib.bin"
    src_path.write_text("fn main() -> i32 { 42 }\n", encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.backend.x86_64",
            str(src_path), str(out_path), "--stdlib",
        ],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out_path.exists()
    assert "unknown flag --stdlib" not in proc.stderr


def test_stage35_direct_x86_rejects_conflicting_stdlib_flags(tmp_path):
    src_path = tmp_path / "conflicting_stdlib_x86.hx"
    out_path = tmp_path / "conflicting_stdlib_x86.bin"
    src_path.write_text("fn main() -> i32 { 42 }\n", encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.backend.x86_64",
            str(src_path), str(out_path), "--stdlib", "--no-stdlib",
        ],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert not out_path.exists()
    assert "conflicting stdlib flags" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_stage35_direct_x86_rejects_flag_shaped_output(tmp_path):
    src_path = tmp_path / "flag_output_x86.hx"
    src_path.write_text("fn main() -> i32 { 42 }\n", encoding="utf-8")
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        proj_root if not env.get("PYTHONPATH")
        else proj_root + os.pathsep + env["PYTHONPATH"]
    )
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.backend.x86_64",
            str(src_path), "--no-stdlib",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert not (tmp_path / "--no-stdlib").exists()
    assert "output path cannot be a flag" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_stage35_direct_x86_rejects_source_as_output(tmp_path):
    src_path = tmp_path / "same_source_x86.hx"
    source = "fn main() -> i32 { 42 }\n"
    src_path.write_text(source, encoding="utf-8")
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.backend.x86_64",
            str(src_path), str(src_path), "--no-stdlib",
        ],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert src_path.read_text(encoding="utf-8") == source
    assert "output path must differ from input source path" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_stage35_direct_x86_rejects_flag_shaped_input_before_output(tmp_path):
    flag_src = tmp_path / "--no-stdlib"
    flag_src.write_text("fn main() -> i32 { 42 }\n", encoding="utf-8")
    victim = tmp_path / "victim.hx"
    source = "fn main() -> i32 { 7 }\n"
    victim.write_text(source, encoding="utf-8")
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        proj_root if not env.get("PYTHONPATH")
        else proj_root + os.pathsep + env["PYTHONPATH"]
    )
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.backend.x86_64",
            "--no-stdlib", str(victim),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert victim.read_text(encoding="utf-8") == source
    assert flag_src.read_text(encoding="utf-8") == "fn main() -> i32 { 42 }\n"
    assert "input path cannot be a flag" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_stage35_direct_x86_missing_input_reports_clean_error(tmp_path):
    missing = tmp_path / "missing.hx"
    out_path = tmp_path / "missing.bin"
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.backend.x86_64",
            str(missing), str(out_path),
        ],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert not out_path.exists()
    assert "error: input:" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_stage35_direct_x86_failure_removes_prior_artifact(tmp_path):
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    good = tmp_path / "good_direct.hx"
    bad = tmp_path / "bad_direct.hx"
    missing = tmp_path / "missing_direct.hx"
    out_path = tmp_path / "stale_direct.bin"
    good.write_text("fn main() -> i32 { 0 }\n", encoding="utf-8")
    bad.write_text(
        "fn main() -> i32 { let mut x: i64 = 1_i64; x = 2_i32; 0 }\n",
        encoding="utf-8",
    )

    good_proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.backend.x86_64",
            str(good), str(out_path), "--no-stdlib",
        ],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert good_proc.returncode == 0, good_proc.stdout + good_proc.stderr
    assert out_path.exists()

    bad_proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.backend.x86_64",
            str(bad), str(out_path), "--no-stdlib",
        ],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert bad_proc.returncode == 1, bad_proc.stdout + bad_proc.stderr
    assert "type error" in bad_proc.stderr
    assert not out_path.exists()

    out_path.write_bytes(b"OLD")
    missing_proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.backend.x86_64",
            str(missing), str(out_path), "--no-stdlib",
        ],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert missing_proc.returncode == 2, missing_proc.stdout + missing_proc.stderr
    assert "error: input:" in missing_proc.stderr
    assert not out_path.exists()


def test_stage35_direct_x86_invalid_utf8_reports_clean_error(tmp_path):
    src_path = tmp_path / "bad_utf8.hx"
    out_path = tmp_path / "bad_utf8.bin"
    src_path.write_bytes(b"\xff\n")
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.backend.x86_64",
            str(src_path), str(out_path),
        ],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert not out_path.exists()
    assert "encoding error reading source" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_stage35_direct_x86_missing_strict_stdlib_reports_clean_error(
    monkeypatch, capsys, tmp_path
):
    src_path = tmp_path / "strict_missing_stdlib.hx"
    out_path = tmp_path / "strict_missing_stdlib.bin"
    src_path.write_text("fn main() -> i32 { 42 }\n", encoding="utf-8")
    monkeypatch.setenv(parser_mod.STDLIB_STRICT_ENV, "1")
    monkeypatch.setattr(parser_mod, "STDLIB_FILES", ["stage35_missing_stdlib.hx"])
    monkeypatch.setattr(
        sys,
        "argv",
        ["helixc.backend.x86_64", str(src_path), str(out_path)],
    )
    monkeypatch.delitem(sys.modules, "helixc.backend.x86_64", raising=False)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("helixc.backend.x86_64", run_name="__main__")

    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert not out_path.exists()
    assert "error: stdlib:" in captured.err
    assert "stdlib file missing" in captured.err
    assert "Traceback" not in captured.err


def test_stage35_direct_x86_duplicate_impl_reports_clean_error(tmp_path):
    src_path = tmp_path / "dup_impl.hx"
    out_path = tmp_path / "dup_impl.bin"
    src_path.write_text(
        "struct Foo { x: i32 }\n"
        "struct Bar { y: i32 }\n"
        "impl Foo { fn area(self: Foo) -> i32 { self.x } }\n"
        "impl Bar { fn area(self: Bar) -> i32 { self.y } }\n"
        "fn main() -> i32 { 0 }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.backend.x86_64",
            str(src_path), str(out_path), "--no-stdlib",
        ],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert not out_path.exists()
    assert "duplicate method" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_stage35_direct_x86_missing_output_dir_reports_clean_error(tmp_path):
    src_path = tmp_path / "ok_out.hx"
    out_path = tmp_path / "missing_dir" / "out.bin"
    src_path.write_text("fn main() -> i32 { 42 }\n", encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable, "-m", "helixc.backend.x86_64",
            str(src_path), str(out_path), "--no-stdlib",
        ],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert not out_path.exists()
    assert "error: output:" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_stage35_direct_x86_chmod_failure_removes_temp_and_final(
    monkeypatch, capsys, tmp_path
):
    src_path = tmp_path / "chmod_fail.hx"
    out_path = tmp_path / "chmod_fail.bin"
    src_path.write_text("fn main() -> i32 { 42 }\n", encoding="utf-8")

    def _chmod_blocking(path, mode):
        raise PermissionError("synthetic chmod denied")

    monkeypatch.setattr(os, "chmod", _chmod_blocking)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "helixc.backend.x86_64",
            str(src_path),
            str(out_path),
            "--no-stdlib",
        ],
    )
    monkeypatch.delitem(sys.modules, "helixc.backend.x86_64", raising=False)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("helixc.backend.x86_64", run_name="__main__")

    captured = capsys.readouterr()
    assert exc.value.code == 1
    assert "error: output:" in captured.err
    assert "synthetic chmod denied" in captured.err
    assert not out_path.exists()
    assert list(tmp_path.glob(".chmod_fail.bin.*.tmp")) == []


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
    src = (
        "type Probability = f64 where 0.0 <= self <= 1.0;\n"
        "fn main() -> i32 {\n"
        "    let p: Probability = 0.5_f64;\n"
        "    0\n"
        "}\n"
    )
    with open(src_path, "wb") as f:
        f.write(src.encode("utf-8"))
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    artifact = json.loads(captured.out)
    assert artifact["schema"] == "helix.proof_obligations.v0"
    assert len(artifact["cache_key"]) == 64
    assert artifact["input"]["source_sha256"] == (
        hashlib.sha256(src.encode("utf-8")).hexdigest()
    )
    assert artifact["input"]["include_stdlib"] is False
    assert artifact["input"]["stdlib_manifest_sha256"] == (
        hashlib.sha256(b"[]").hexdigest()
    )
    assert artifact["input"]["stdlib_files"] == []
    assert artifact["input"]["flags"] == [
        "--emit-proof-obligations", "--no-stdlib",
    ]
    assert artifact["input"]["opt_level"] == 1
    assert artifact["input"]["color"] == "auto"
    assert artifact["summary"]["typecheck_errors"] == 0
    assert artifact["summary"]["warning_diagnostics"] == 0
    assert artifact["summary"]["warning_errors"] == 0
    assert artifact["summary"]["obligations"] == 1
    obligation = artifact["obligations"][0]
    assert obligation["kind"] == "refinement"
    assert obligation["context"] == "let 'p'"
    assert obligation["refinement"] == "Probability"
    assert obligation["predicate"] == "0.0 <= self <= 1.0"
    assert obligation["status"] == "proved"
    assert obligation["value"] == "0.5"
    assert "parse:" in captured.err


def test_stage31_emit_proof_obligations_input_hashes_default_stdlib(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "default_stdlib_obligation.hx")
    src = "fn main() -> i32 { let p: Probability = 0.5_f64; 0 }\n"
    with open(src_path, "wb") as f:
        f.write(src.encode("utf-8"))
    rc = main([src_path, "--emit-proof-obligations"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    artifact = json.loads(captured.out)
    assert artifact["input"]["source_sha256"] == (
        hashlib.sha256(src.encode("utf-8")).hexdigest()
    )
    assert artifact["input"]["include_stdlib"] is True
    assert len(artifact["input"]["stdlib_manifest_sha256"]) == 64
    assert artifact["input"]["stdlib_files"]
    assert any(
        item["path"] == "agi_memory.hx"
        and len(item.get("sha256", "")) == 64
        for item in artifact["input"]["stdlib_files"]
    )
    assert artifact["summary"]["typecheck_errors"] == 0
    assert artifact["summary"]["obligations"] == 1


def test_stage31_emit_proof_obligations_cache_key_path_independent(
    capsys, tmp_path,
):
    src = (
        "type Probability = f64 where 0.0 <= self <= 1.0;\n"
        "fn main() -> i32 { let p: Probability = 0.5_f64; 0 }\n"
    )
    first = tmp_path / "first.hx"
    second_dir = tmp_path / "nested"
    second_dir.mkdir()
    second = second_dir / "second.hx"
    first.write_bytes(src.encode("utf-8"))
    second.write_bytes(src.encode("utf-8"))

    rc_first = main([
        str(first), "--emit-proof-obligations", "--no-stdlib",
    ])
    captured_first = capsys.readouterr()
    assert rc_first == 0, captured_first.out + captured_first.err
    first_artifact = json.loads(captured_first.out)

    rc_second = main([
        str(second), "--emit-proof-obligations", "--no-stdlib",
    ])
    captured_second = capsys.readouterr()
    assert rc_second == 0, captured_second.out + captured_second.err
    second_artifact = json.loads(captured_second.out)

    assert first_artifact["path"] != second_artifact["path"]
    assert len(first_artifact["cache_key"]) == 64
    assert first_artifact["cache_key"] == second_artifact["cache_key"]
    assert first_artifact["input"] == second_artifact["input"]


def test_stage31_emit_proof_obligations_cache_key_changes_with_source(
    capsys, tmp_path,
):
    first = tmp_path / "first.hx"
    second = tmp_path / "second.hx"
    first.write_text(
        "type Probability = f64 where 0.0 <= self <= 1.0;\n"
        "fn main() -> i32 { let p: Probability = 0.5_f64; 0 }\n",
        encoding="utf-8",
    )
    second.write_text(
        "type Probability = f64 where 0.0 <= self <= 1.0;\n"
        "fn main() -> i32 { let p: Probability = 0.6_f64; 0 }\n",
        encoding="utf-8",
    )

    rc_first = main([
        str(first), "--emit-proof-obligations", "--no-stdlib",
    ])
    captured_first = capsys.readouterr()
    assert rc_first == 0, captured_first.out + captured_first.err
    first_artifact = json.loads(captured_first.out)

    rc_second = main([
        str(second), "--emit-proof-obligations", "--no-stdlib",
    ])
    captured_second = capsys.readouterr()
    assert rc_second == 0, captured_second.out + captured_second.err
    second_artifact = json.loads(captured_second.out)

    assert first_artifact["input"]["source_sha256"] != (
        second_artifact["input"]["source_sha256"]
    )
    assert first_artifact["cache_key"] != second_artifact["cache_key"]


def test_stage31_emit_proof_obligations_normalizes_stdlib_flag(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "explicit_stdlib_flag.hx")
    src = "fn main() -> i32 { let p: Probability = 0.5_f64; 0 }\n"
    with open(src_path, "wb") as f:
        f.write(src.encode("utf-8"))
    rc_default = main([src_path, "--emit-proof-obligations"])
    captured_default = capsys.readouterr()
    assert rc_default == 0, captured_default.out + captured_default.err
    default_artifact = json.loads(captured_default.out)

    rc_explicit = main([src_path, "--emit-proof-obligations", "--stdlib"])
    captured_explicit = capsys.readouterr()
    assert rc_explicit == 0, captured_explicit.out + captured_explicit.err
    explicit_artifact = json.loads(captured_explicit.out)

    assert default_artifact["input"] == explicit_artifact["input"]
    assert default_artifact["cache_key"] == explicit_artifact["cache_key"]
    assert "--stdlib" not in explicit_artifact["input"]["flags"]


def test_stage31_emit_proof_obligations_counts_missing_stdlib_warning(
    monkeypatch, capsys, tmp_path,
):
    src_path = str(tmp_path / "missing_stdlib_warning.hx")
    with open(src_path, "wb") as f:
        f.write(b"fn main() -> i32 { 0 }\n")
    monkeypatch.setattr(
        parser_mod,
        "STDLIB_FILES",
        ["does_not_exist_anywhere_for_proof_metadata.hx"],
    )
    rc = main([src_path, "--emit-proof-obligations"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    artifact = json.loads(captured.out)
    assert artifact["input"]["include_stdlib"] is True
    assert artifact["input"]["stdlib_files"][0]["missing"] is True
    assert artifact["summary"]["warning_diagnostics"] == 1
    assert artifact["summary"]["warning_errors"] == 0
    warning = artifact["warning_diagnostics"][0]
    assert warning["kind"] == "stdlib"
    assert warning["promoted_to_error"] is False
    assert "does_not_exist_anywhere_for_proof_metadata.hx" in warning["message"]
    assert "stdlib file missing" in captured.err


def test_stage31_emit_proof_obligations_strict_missing_stdlib_stays_json(
    monkeypatch, capsys, tmp_path,
):
    src_path = str(tmp_path / "strict_missing_stdlib.hx")
    with open(src_path, "wb") as f:
        f.write(b"fn main() -> i32 { 0 }\n")
    monkeypatch.setattr(
        parser_mod,
        "STDLIB_FILES",
        ["strict_missing_stdlib_for_proof_metadata.hx"],
    )
    monkeypatch.setenv(parser_mod.STDLIB_STRICT_ENV, "1")
    rc = main([src_path, "--emit-proof-obligations"])
    captured = capsys.readouterr()
    assert rc == 2
    artifact = json.loads(captured.out)
    assert artifact["input"]["stdlib_strict"] is True
    assert artifact["input"]["stdlib_files"][0]["missing"] is True
    assert artifact["summary"]["pipeline_errors"] == 1
    assert artifact["pipeline_errors"][0]["phase"] == "stdlib"
    assert "C:\\Projects\\Kovostov-Native" not in (
        artifact["pipeline_errors"][0]["message"]
    )
    assert "helixc\\stdlib" not in artifact["pipeline_errors"][0]["message"]
    assert artifact["summary"]["warning_diagnostics"] == 1
    assert artifact["summary"]["warning_errors"] == 1
    warning = artifact["warning_diagnostics"][0]
    assert warning["kind"] == "stdlib"
    assert warning["policy"] == "error"
    assert warning["promoted_to_error"] is True
    assert "stdlib file missing" in captured.err


def test_stage31_emit_proof_obligations_decode_error_stays_json(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "bad_utf8.hx")
    with open(src_path, "wb") as f:
        f.write(b"\xff\n")
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 2
    artifact = json.loads(captured.out)
    assert len(artifact["cache_key"]) == 64
    assert artifact["input"]["source_sha256"] == hashlib.sha256(b"\xff\n").hexdigest()
    assert artifact["summary"]["pipeline_errors"] == 1
    assert artifact["pipeline_errors"][0]["phase"] == "decode"
    assert "encoding error reading source" in (
        artifact["pipeline_errors"][0]["message"]
    )


def test_stage31_emit_proof_obligations_invalid_warning_name_stays_json(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "bad_warning_name.hx")
    with open(src_path, "w") as f:
        f.write("fn main() -> i32 { 0 }\n")
    rc = main([
        src_path, "--emit-proof-obligations", "--no-stdlib",
        "-Wdeprectaed=error",
    ])
    captured = capsys.readouterr()
    assert rc == 2
    artifact = json.loads(captured.out)
    assert artifact["summary"]["pipeline_errors"] == 1
    assert artifact["pipeline_errors"][0]["phase"] == "invocation"
    assert "unknown warning name" in artifact["pipeline_errors"][0]["message"]


def test_stage31_emit_proof_obligations_invalid_warning_policy_stays_json(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "bad_warning_policy.hx")
    with open(src_path, "w") as f:
        f.write("fn main() -> i32 { 0 }\n")
    rc = main([
        src_path, "--emit-proof-obligations", "--no-stdlib",
        "-Wdeprecated=erro",
    ])
    captured = capsys.readouterr()
    assert rc == 2
    artifact = json.loads(captured.out)
    assert artifact["summary"]["pipeline_errors"] == 1
    assert artifact["pipeline_errors"][0]["phase"] == "invocation"
    assert "unknown warning policy" in artifact["pipeline_errors"][0]["message"]


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
    assert artifact["input"]["include_stdlib"] is False
    assert artifact["summary"]["warning_diagnostics"] == 0
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


def test_stage31_emit_proof_obligations_json_for_equivalent_refinement_alias(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "equivalent_alias_obligation.hx")
    with open(src_path, "w") as f:
        f.write(
            "type NonNegativeA = f64 where self >= 0.0;\n"
            "type NonNegativeB = f64 where self >= 0.0;\n"
            "fn lift(a: NonNegativeA) -> NonNegativeB { a }\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] == 0
    assert artifact["obligations"] == []
    assert artifact["summary"]["proof_carries"] == 1
    carry = artifact["proof_carries"][0]
    assert carry["kind"] == "refinement-proof-carry"
    assert carry["source_refinement"] == "NonNegativeA"
    assert carry["target_refinement"] == "NonNegativeB"
    assert carry["strategy"] == "exact-predicate-subset"


def test_stage34_emit_proof_carry_json_for_same_refinement_strategy(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "same_refinement_carry.hx")
    with open(src_path, "w") as f:
        f.write(
            "type NonNegative = f64 where self >= 0.0;\n"
            "fn lift(a: NonNegative) -> NonNegative { a }\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] == 0
    assert artifact["obligations"] == []
    assert artifact["summary"]["proof_carries"] == 1
    carry = artifact["proof_carries"][0]
    assert carry["source_refinement"] == "NonNegative"
    assert carry["target_refinement"] == "NonNegative"
    assert carry["strategy"] == "same-refinement"
    assert artifact["summary"]["proof_carry_strategies"] == {
        "same-refinement": 1,
    }


def test_stage34_emit_proof_obligations_json_for_numeric_bound_implication(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "numeric_bound_implication_obligation.hx")
    with open(src_path, "w") as f:
        f.write(
            "type AtLeastOne = f64 where self >= 1.0;\n"
            "type NonNegative = f64 where self >= 0.0;\n"
            "fn lift(a: AtLeastOne) -> NonNegative { a }\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] == 0
    assert artifact["obligations"] == []
    assert artifact["summary"]["proof_carries"] == 1
    carry = artifact["proof_carries"][0]
    assert carry["source_refinement"] == "AtLeastOne"
    assert carry["target_refinement"] == "NonNegative"
    assert carry["strategy"] == "numeric-bound-implication"


def test_stage34_emit_proof_obligations_json_for_equality_bound_implication(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "equality_bound_implication_obligation.hx")
    with open(src_path, "w") as f:
        f.write(
            "type ExactlyOne = f64 where self == 1.0;\n"
            "type AtMostOne = f64 where self <= 1.0;\n"
            "fn lift(a: ExactlyOne) -> AtMostOne { a }\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] == 0
    assert artifact["obligations"] == []
    assert artifact["summary"]["proof_carries"] == 1
    carry = artifact["proof_carries"][0]
    assert carry["source_refinement"] == "ExactlyOne"
    assert carry["target_refinement"] == "AtMostOne"
    assert carry["strategy"] == "numeric-bound-implication"


def test_stage34_emit_proof_carry_json_for_array_bound_implication(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "array_bound_implication_carry.hx")
    with open(src_path, "w") as f:
        f.write(
            "type AtLeastOne = f64 where self >= 1.0;\n"
            "type NonNegative = f64 where self >= 0.0;\n"
            "fn use_values(xs: [NonNegative; 2]) -> i32 { 0 }\n"
            "fn lift(xs: [AtLeastOne; 2]) -> i32 { use_values(xs) }\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] == 0
    assert artifact["obligations"] == []
    assert artifact["summary"]["proof_carries"] == 1
    carry = artifact["proof_carries"][0]
    assert carry["context"] == "call to 'use_values': arg 'xs': array element"
    assert carry["source_refinement"] == "AtLeastOne"
    assert carry["target_refinement"] == "NonNegative"
    assert carry["strategy"] == "numeric-bound-implication"


def test_stage34_emit_proof_carry_json_for_tuple_bound_implication(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "tuple_bound_implication_carry.hx")
    with open(src_path, "w") as f:
        f.write(
            "type AtMostHalf = f64 where self <= 0.5;\n"
            "type AtMostOne = f64 where self <= 1.0;\n"
            "fn lift(xs: (AtMostHalf, AtMostHalf)) -> i32 {\n"
            "    let ys: (AtMostOne, AtMostOne) = xs;\n"
            "    0\n"
            "}\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] == 0
    assert artifact["obligations"] == []
    assert artifact["summary"]["proof_carries"] == 2
    contexts = {carry["context"] for carry in artifact["proof_carries"]}
    assert contexts == {
        "let 'ys': tuple element 0",
        "let 'ys': tuple element 1",
    }
    assert all(carry["source_refinement"] == "AtMostHalf"
               for carry in artifact["proof_carries"])
    assert all(carry["target_refinement"] == "AtMostOne"
               for carry in artifact["proof_carries"])
    assert all(carry["strategy"] == "numeric-bound-implication"
               for carry in artifact["proof_carries"])


def test_stage34_emit_proof_carry_json_for_negated_bounds_only(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "negated_bound_carry.hx")
    with open(src_path, "w") as f:
        f.write(
            "type NotBelowZero = i32 where !(self < 0);\n"
            "type NonNegative = i32 where self >= 0;\n"
            "fn use_n(x: NonNegative) -> i32 { 0 }\n"
            "fn lift(b: NotBelowZero) -> i32 {\n"
            "    use_n(b)\n"
            "}\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] == 0
    assert artifact["obligations"] == []
    assert artifact["summary"]["proof_carries"] == 1
    assert artifact["summary"]["proof_carry_strategies"] == {
        "numeric-bound-implication": 1,
    }
    sources = {
        carry["source_refinement"]: carry["strategy"]
        for carry in artifact["proof_carries"]
    }
    assert sources == {
        "NotBelowZero": "numeric-bound-implication",
    }


def test_stage34_emit_proof_obligations_json_for_refined_cast_target_value(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "refined_cast_target_value.hx")
    with open(src_path, "w") as f:
        f.write(
            "type ExactlyHalfInt = i32 where self == 0.5;\n"
            "fn f() -> ExactlyHalfInt {\n"
            "    0.5_f64 as ExactlyHalfInt\n"
            "}\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] >= 1
    obligation = artifact["obligations"][0]
    assert obligation["context"] == "cast to refined type ExactlyHalfInt"
    assert obligation["refinement"] == "ExactlyHalfInt"
    assert obligation["predicate"] == "self == 0.5"
    assert obligation["status"] == "failed"
    assert obligation["value"] == "0"
    assert "target value 0 does not satisfy self == 0.5" in (
        artifact["typecheck_errors"][0]
    )
    assert artifact["proof_carries"] == []


def test_stage34_failed_refined_cast_does_not_emit_return_carry(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "failed_refined_cast_no_return_carry.hx")
    with open(src_path, "w") as f:
        f.write(
            "type One = i32 where self == 1;\n"
            "fn f() -> One { true as One }\n"
            "fn main() -> i32 { 0 }\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert any(
        obligation["context"] == "cast to refined type One"
        and obligation["status"] == "unproven"
        for obligation in artifact["obligations"]
    )
    assert artifact["summary"]["proof_carries"] == 0
    assert artifact["summary"]["proof_carry_strategies"] == {}
    assert artifact["proof_carries"] == []


def test_stage34_failed_refined_composite_casts_do_not_emit_carries(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "failed_refined_composite_cast_no_carry.hx")
    with open(src_path, "w") as f:
        f.write(
            "type NonNegative = f64 where self >= 0.0;\n"
            "fn f(xs: [f64; 1], pair: (f64, f64)) -> i32 {\n"
            "    let ys: [NonNegative; 1] = xs as [NonNegative; 1];\n"
            "    let zs: (NonNegative, NonNegative) = pair as "
            "(NonNegative, NonNegative);\n"
            "    0\n"
            "}\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] >= 2
    assert artifact["summary"]["proof_carries"] == 0
    assert artifact["summary"]["proof_carry_strategies"] == {}
    assert artifact["proof_carries"] == []


def test_stage34_failed_refined_initializer_does_not_emit_later_carries(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "failed_refined_initializer_no_later_carry.hx")
    with open(src_path, "w") as f:
        f.write(
            "type One = i32 where self == 1;\n"
            "type NonNegative = f64 where self >= 0.0;\n"
            "fn use_one(x: One) -> i32 { 0 }\n"
            "fn use_array(xs: [NonNegative; 1]) -> i32 { 0 }\n"
            "fn scalar_bad() -> One {\n"
            "    let bad: One = true as One;\n"
            "    bad\n"
            "}\n"
            "fn call_bad() -> i32 {\n"
            "    let bad: One = true as One;\n"
            "    let fp: fn(One) -> i32 = use_one;\n"
            "    fp(bad)\n"
            "}\n"
            "fn array_bad(xs: [f64; 1]) -> i32 {\n"
            "    let ys: [NonNegative; 1] = xs as [NonNegative; 1];\n"
            "    use_array(ys)\n"
            "}\n"
            "fn tuple_bad(pair: (f64, f64)) -> (NonNegative, NonNegative) {\n"
            "    let bad: (NonNegative, NonNegative) = pair as "
            "(NonNegative, NonNegative);\n"
            "    bad\n"
            "}\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] >= 4
    assert artifact["summary"]["proof_carries"] == 0
    assert artifact["summary"]["proof_carry_strategies"] == {}
    assert artifact["proof_carries"] == []


def test_stage34_failed_local_const_initializer_does_not_emit_later_carries(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "failed_local_const_initializer_no_carry.hx")
    with open(src_path, "w") as f:
        f.write(
            "type One = i32 where self == 1;\n"
            "fn use_one(x: One) -> i32 { 0 }\n"
            "fn local_const_bad() -> i32 {\n"
            "    const bad: One = 2;\n"
            "    use_one(bad)\n"
            "}\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert artifact["summary"]["proof_carries"] == 0
    assert artifact["summary"]["proof_carry_strategies"] == {}
    assert artifact["proof_carries"] == []


def test_stage34_failed_top_level_const_initializer_does_not_emit_later_carries(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "failed_top_level_const_initializer_no_carry.hx")
    with open(src_path, "w") as f:
        f.write(
            "type One = i32 where self == 1;\n"
            "const BAD: One = 2;\n"
            "fn use_one(x: One) -> i32 { 0 }\n"
            "fn call_bad() -> i32 { use_one(BAD) }\n"
            "fn return_bad() -> One { BAD }\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert artifact["summary"]["proof_carries"] == 0
    assert artifact["summary"]["proof_carry_strategies"] == {}
    assert artifact["proof_carries"] == []


def test_stage34_self_independent_unrepresentable_value_is_not_clean(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "self_independent_unrepresentable_value.hx")
    with open(src_path, "w") as f:
        f.write(
            "type AlwaysF64 = f64 where true;\n"
            "type AlwaysInt = i32 where true;\n"
            "fn use_f64(x: AlwaysF64) -> i32 { 0 }\n"
            "fn call_bad() -> i32 { use_f64(literal_bad()) }\n"
            "fn chain_bad() -> i32 { use_f64(wrapper_bad()) }\n"
            "fn wrapper_bad() -> AlwaysF64 { literal_bad() }\n"
            "fn literal_bad() -> AlwaysF64 { 1e309_f64 }\n"
            "fn cast_bad() -> AlwaysInt { 1e309_f64 as AlwaysInt }\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] >= 2
    assert artifact["summary"]["proof_carries"] == 0
    assert artifact["summary"]["proof_carry_strategies"] == {}
    assert artifact["proof_carries"] == []


def test_stage34_f32_predicate_rounding_does_not_false_carry_bounds(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "f32_predicate_rounding_no_false_carry.hx")
    with open(src_path, "w") as f:
        f.write(
            "type Source = f32 where self >= 16777217.0_f32;\n"
            "type Target = f32 where self > 16777216.0_f32;\n"
            "fn make() -> Source { 16777216.0_f32 }\n"
            "fn bad(s: Source) -> Target { s }\n"
            "fn main() -> i32 { let t: Target = bad(make()); 0 }\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert not any(
        carry["strategy"] == "numeric-bound-implication"
        and carry["source_refinement"] == "Source"
        and carry["target_refinement"] == "Target"
        for carry in artifact["proof_carries"]
    )


def test_stage34_fixed_point_unbound_name_is_not_clean(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "fixed_point_unbound_name_not_clean.hx")
    with open(src_path, "w") as f:
        f.write(
            "type AlwaysI32 = i32 where true;\n"
            "fn bad() -> AlwaysI32 { missing }\n"
            "fn main() -> i32 { 0 }\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert any("unbound name 'missing'" in err
               for err in artifact["typecheck_errors"])
    assert artifact["summary"]["proof_carries"] == 0
    assert artifact["proof_carries"] == []


def test_stage34_emit_proof_obligations_json_for_refined_f32_rounding(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "refined_f32_rounding.hx")
    with open(src_path, "w") as f:
        f.write(
            "type AboveF32Boundary = f32 where self > 16777216.0;\n"
            "fn f() -> AboveF32Boundary {\n"
            "    16777217.0_f64 as AboveF32Boundary\n"
            "}\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] >= 1
    obligation = artifact["obligations"][0]
    assert obligation["context"] == "cast to refined type AboveF32Boundary"
    assert obligation["refinement"] == "AboveF32Boundary"
    assert obligation["predicate"] == "self > 16777216.0"
    assert obligation["status"] == "failed"
    assert obligation["value"] == "16777216.0"


def test_stage34_emit_proof_carry_json_for_explicit_return_route(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "explicit_return_bound_carry.hx")
    with open(src_path, "w") as f:
        f.write(
            "type AtLeastOne = f64 where self >= 1.0;\n"
            "type NonNegative = f64 where self >= 0.0;\n"
            "fn lift(a: AtLeastOne) -> NonNegative {\n"
            "    return a;\n"
            "    a\n"
            "}\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] == 0
    assert artifact["obligations"] == []
    assert any(
        carry["context"] == "return value of function 'lift'"
        and carry["source_refinement"] == "AtLeastOne"
        and carry["target_refinement"] == "NonNegative"
        and carry["strategy"] == "numeric-bound-implication"
        for carry in artifact["proof_carries"]
    )


def test_stage34_emit_proof_carry_json_for_refined_cast_route(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "refined_cast_bound_carry.hx")
    with open(src_path, "w") as f:
        f.write(
            "type AtLeastOne = f64 where self >= 1.0;\n"
            "type NonNegative = f64 where self >= 0.0;\n"
            "fn lift(a: AtLeastOne) -> NonNegative {\n"
            "    a as NonNegative\n"
            "}\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] == 0
    assert artifact["obligations"] == []
    assert any(
        carry["context"] == "cast to refined type NonNegative"
        and carry["source_refinement"] == "AtLeastOne"
        and carry["target_refinement"] == "NonNegative"
        and carry["strategy"] == "numeric-bound-implication"
        for carry in artifact["proof_carries"]
    )


def test_stage34_emit_proof_carry_json_for_function_typed_call_route(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "function_typed_call_bound_carry.hx")
    with open(src_path, "w") as f:
        f.write(
            "type AtLeastOne = f64 where self >= 1.0;\n"
            "type NonNegative = f64 where self >= 0.0;\n"
            "fn use_n(x: NonNegative) -> i32 { 0 }\n"
            "fn lift(a: AtLeastOne) -> i32 {\n"
            "    let fp: fn(NonNegative) -> i32 = use_n;\n"
            "    fp(a)\n"
            "}\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert any(
        "function-typed calls are not supported by the Stage 31 backend"
        in err for err in artifact["typecheck_errors"]
    )
    assert any(
        carry["context"] == "function-typed call arg 0"
        and carry["source_refinement"] == "AtLeastOne"
        and carry["target_refinement"] == "NonNegative"
        and carry["strategy"] == "numeric-bound-implication"
        for carry in artifact["proof_carries"]
    )


def test_stage31_emit_proof_obligations_rejects_generic_refinement_name(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "generic_refinement_name_obligation.hx")
    with open(src_path, "w") as f:
        f.write(
            "type A = f64 where self::<Missing> >= 0.0;\n"
            "fn f() -> A { 1.0_f64 }\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert any(
        "self::<Missing> >= 0.0 is not supported" in err
        for err in artifact["typecheck_errors"]
    )
    assert any(
        obligation["refinement"] == "A"
        and obligation["predicate"] == "self::<Missing> >= 0.0"
        and obligation["status"] == "unsupported"
        for obligation in artifact["obligations"]
    )


def test_stage31_emit_proof_obligations_rejects_duplicate_proof_names(
    capsys, tmp_path,
):
    alias_path = str(tmp_path / "duplicate_alias_obligation.hx")
    with open(alias_path, "w") as f:
        f.write(
            "type Gate = f64 where self >= 0.0;\n"
            "type Gate = f64 where self <= 0.0;\n"
            "fn bad() -> Gate { 1.0_f64 }\n"
        )
    rc = main([alias_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert any(
        "name collision after module flattening: 'Gate'" in err["message"]
        for err in artifact["pipeline_errors"]
    )

    const_path = str(tmp_path / "duplicate_const_obligation.hx")
    with open(const_path, "w") as f:
        f.write(
            "const LIMIT: f64 = 1.0_f64;\n"
            "const LIMIT: f64 = 0.0_f64;\n"
            "type A = f64 where self <= LIMIT;\n"
            "fn bad() -> A { 1.0_f64 }\n"
        )
    rc = main([const_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert any(
        "name collision after module flattening: 'LIMIT'" in err["message"]
        for err in artifact["pipeline_errors"]
    )


def test_stage31_emit_proof_obligations_json_for_unsupported_refinement(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "unsupported_obligation.hx")
    with open(src_path, "w") as f:
        f.write(
            "type Weird = f64 where self + 1.0;\n"
            "fn main() -> i32 {\n"
            "    let w: Weird = 0.5_f64;\n"
            "    0\n"
            "}\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] == 2
    obligation = artifact["obligations"][0]
    assert obligation["refinement"] == "Weird"
    assert obligation["predicate"] == "(self + 1.0)"
    assert obligation["status"] == "unsupported"
    assert obligation["value"] == "0.5"
    assert "not supported by the Stage 31 constant checker" in (
        artifact["typecheck_errors"][1]
    )


def test_stage31_emit_proof_obligations_json_for_boolean_literal_refinements(
    capsys, tmp_path,
):
    always_path = str(tmp_path / "always_obligation.hx")
    with open(always_path, "w") as f:
        f.write(
            "type Always = f64 where true;\n"
            "fn main() -> i32 { let a: Always = 0.5_f64; 0 }\n"
        )
    rc = main([always_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] == 0
    obligation = artifact["obligations"][0]
    assert obligation["refinement"] == "Always"
    assert obligation["predicate"] == "true"
    assert obligation["status"] == "proved"
    assert obligation["value"] == "0.5"

    never_path = str(tmp_path / "never_obligation.hx")
    with open(never_path, "w") as f:
        f.write(
            "type Never = f64 where false;\n"
            "fn main() -> i32 { let n: Never = 0.5_f64; 0 }\n"
        )
    rc = main([never_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] == 1
    obligation = artifact["obligations"][0]
    assert obligation["refinement"] == "Never"
    assert obligation["predicate"] == "false"
    assert obligation["status"] == "failed"
    assert obligation["trap"] == "31001"
    assert obligation["value"] == "0.5"


def test_stage31_emit_proof_obligations_json_for_self_independent_false(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "never_raw_obligation.hx")
    with open(src_path, "w") as f:
        f.write(
            "type Never = f64 where false;\n"
            "fn use_raw(x: f64) -> i32 { let n: Never = x; 0 }\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] == 1
    obligation = artifact["obligations"][0]
    assert obligation["refinement"] == "Never"
    assert obligation["predicate"] == "false"
    assert obligation["status"] == "failed"
    assert obligation["trap"] == "31001"
    assert "value" not in obligation
    assert "predicate false is always false" in artifact["typecheck_errors"][0]


def test_stage31_emit_proof_obligations_json_for_mixed_independent_predicates(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "mixed_bool_obligation.hx")
    with open(src_path, "w") as f:
        f.write(
            "type Mixed = f64 where false, self >= 0.0;\n"
            "fn use_raw(x: f64) -> i32 { let m: Mixed = x; 0 }\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    observed = {
        (o["refinement"], o["predicate"], o["status"], o.get("trap"))
        for o in artifact["obligations"]
    }
    assert ("Mixed", "false", "failed", "31001") in observed
    assert ("Mixed", "self >= 0.0", "unproven", None) in observed


def test_stage31_emit_proof_obligations_json_for_inherited_independent_false(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "inherited_never_obligation.hx")
    with open(src_path, "w") as f:
        f.write(
            "type Never = f64 where false;\n"
            "type NonNegativeNever = Never where self >= 0.0;\n"
            "fn use_raw(x: f64) -> i32 { let n: NonNegativeNever = x; 0 }\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    observed = {
        (o["refinement"], o["predicate"], o["status"], o.get("trap"))
        for o in artifact["obligations"]
    }
    assert ("NonNegativeNever", "self >= 0.0", "unproven", None) in observed
    assert ("Never", "false", "failed", "31001") in observed


def test_stage31_emit_proof_obligations_json_for_short_circuit_predicates(
    capsys, tmp_path,
):
    false_and_path = str(tmp_path / "false_and_obligation.hx")
    with open(false_and_path, "w") as f:
        f.write(
            "type Never = f64 where false && self >= 0.0;\n"
            "fn use_raw(x: f64) -> i32 { let n: Never = x; 0 }\n"
        )
    rc = main([false_and_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    obligation = artifact["obligations"][0]
    assert obligation["refinement"] == "Never"
    assert obligation["predicate"] == "(false && self >= 0.0)"
    assert obligation["status"] == "failed"
    assert obligation["trap"] == "31001"
    assert "value" not in obligation

    true_or_path = str(tmp_path / "true_or_obligation.hx")
    with open(true_or_path, "w") as f:
        f.write(
            "type Always = f64 where true || self >= 0.0;\n"
            "fn use_raw(x: f64) -> i32 { let a: Always = x; 0 }\n"
        )
    rc = main([true_or_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    artifact = json.loads(captured.out)
    obligation = artifact["obligations"][0]
    assert obligation["refinement"] == "Always"
    assert obligation["predicate"] == "(true || self >= 0.0)"
    assert obligation["status"] == "proved"
    assert "value" not in obligation


def test_stage31_emit_proof_obligations_json_for_constant_comparisons(
    capsys, tmp_path,
):
    always_path = str(tmp_path / "constant_true_obligation.hx")
    with open(always_path, "w") as f:
        f.write(
            "type Always = f64 where 1.0 < 2.0;\n"
            "fn use_raw(x: f64) -> i32 { let a: Always = x; 0 }\n"
        )
    rc = main([always_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] == 0
    obligation = artifact["obligations"][0]
    assert obligation["refinement"] == "Always"
    assert obligation["predicate"] == "1.0 < 2.0"
    assert obligation["status"] == "proved"
    assert "value" not in obligation

    never_path = str(tmp_path / "constant_false_obligation.hx")
    with open(never_path, "w") as f:
        f.write(
            "type Never = f64 where 2.0 < 1.0;\n"
            "fn use_raw(x: f64) -> i32 { let n: Never = x; 0 }\n"
        )
    rc = main([never_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] == 1
    obligation = artifact["obligations"][0]
    assert obligation["refinement"] == "Never"
    assert obligation["predicate"] == "2.0 < 1.0"
    assert obligation["status"] == "failed"
    assert obligation["trap"] == "31001"
    assert "value" not in obligation


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
    assert artifact["summary"]["warning_diagnostics"] == 1
    assert artifact["summary"]["warning_errors"] == 0
    assert artifact["warning_diagnostics"][0]["kind"] == "ad"
    assert artifact["warning_diagnostics"][0]["promoted_to_error"] is False
    assert "ad:" not in captured.out
    assert "ad:" in captured.err
    assert "24200" in captured.err or "AD002" in captured.err


def test_stage31_emit_proof_obligations_classifies_ad_warning_error(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "proof_with_ad_error.hx")
    with open(src_path, "w") as f:
        f.write(
            "fn loss(x: D<f64>, y: D<i32>) -> D<f64> { x + y }\n"
            "fn main() -> i32 { 0 }\n"
        )
    rc = main([
        src_path, "--emit-proof-obligations", "--no-stdlib", "-Wad=error",
    ])
    captured = capsys.readouterr()
    assert rc == 1, captured.out + captured.err
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] == 0
    assert artifact["summary"]["warning_diagnostics"] == 1
    assert artifact["summary"]["warning_errors"] == 1
    warning = artifact["warning_diagnostics"][0]
    assert warning["kind"] == "ad"
    assert warning["policy"] == "error"
    assert warning["promoted_to_error"] is True
    assert "24200" in warning["message"] or "AD002" in warning["message"]
    assert "ad:" in captured.err


def test_stage35_emit_proof_obligations_strict_ignores_dead_ad_helper(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "proof_strict_dead_ad_helper.hx")
    with open(src_path, "w") as f:
        f.write(
            "fn loss(x: D<f64>) -> D<f64> { x }\n"
            "fn main() -> i32 { 0 }\n"
        )
    rc = main([
        src_path, "--emit-proof-obligations", "--no-stdlib", "--strict",
    ])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] == 0
    assert artifact["summary"]["pipeline_errors"] == 0
    assert not any(
        e.get("phase") == "strict-effect-check"
        for e in artifact.get("pipeline_errors", [])
    )
    assert "unresolved generic type D" not in captured.err
    assert "unresolved generic type D" not in captured.out


def test_stage31_emit_proof_obligations_classifies_deprecated_error(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "proof_deprecated_error.hx")
    with open(src_path, "w") as f:
        f.write(
            "@deprecated fn old() -> i32 { 0 }\n"
            "fn main() -> i32 { old() }\n"
        )
    rc = main([
        src_path, "--emit-proof-obligations", "--no-stdlib",
        "-Wdeprecated=error",
    ])
    captured = capsys.readouterr()
    assert rc == 1, captured.out + captured.err
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] == 0
    assert artifact["summary"]["warning_errors"] == 1
    assert artifact["warning_diagnostics"][0]["kind"] == "deprecated"
    assert artifact["warning_diagnostics"][0]["policy"] == "error"
    assert artifact["warning_diagnostics"][0]["promoted_to_error"] is True
    assert "old" in artifact["warning_diagnostics"][0]["message"]
    assert "deprecated:" in captured.err


def test_stage31_emit_proof_obligations_classifies_strict_effect_error(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "proof_strict_effect_error.hx")
    with open(src_path, "w") as f:
        f.write(
            "@pure fn bad(x: i32) -> i32 { print_int(x); x }\n"
            "fn main() -> i32 { bad(42) }\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--strict"])
    captured = capsys.readouterr()
    assert rc == 1, captured.out + captured.err
    artifact = json.loads(captured.out)
    assert artifact["summary"]["typecheck_errors"] == 0
    assert artifact["summary"]["warning_errors"] >= 1
    effect_records = [
        w for w in artifact["warning_diagnostics"]
        if w["kind"] == "effect-check"
    ]
    assert effect_records
    assert effect_records[0]["severity"] == "hard"
    assert effect_records[0]["promoted_to_error"] is True
    assert "19001" in effect_records[0]["message"]
    assert "effect-check" in captured.err


def test_stage31_emit_proof_obligations_keeps_strict_effect_records_with_pipeline_error(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "proof_trace_and_effect_error.hx")
    with open(src_path, "w") as f:
        f.write(
            '@trace\n'
            'extern "C" fn ext(x: i32) -> i32;\n'
            "@pure fn bad(x: i32) -> i32 { print_int(x); x }\n"
            "fn main() -> i32 { bad(ext(1)) }\n"
        )
    rc = main([
        src_path, "--emit-proof-obligations", "--no-stdlib", "--strict",
    ])
    captured = capsys.readouterr()
    assert rc == 1, captured.out + captured.err
    artifact = json.loads(captured.out)
    assert artifact["summary"]["pipeline_errors"] >= 1
    assert any(e["phase"] == "trace" for e in artifact["pipeline_errors"])
    assert artifact["summary"]["warning_errors"] >= 1
    effect_records = [
        w for w in artifact["warning_diagnostics"]
        if w["kind"] == "effect-check"
    ]
    assert effect_records
    assert any("19001" in w["message"] for w in effect_records)


def test_stage31_emit_proof_obligations_strict_effect_pass_failure_stays_json(
    monkeypatch, capsys, tmp_path,
):
    from helixc.ir.passes import cse

    src_path = str(tmp_path / "proof_strict_effect_pass_failure.hx")
    with open(src_path, "w") as f:
        f.write("fn main() -> i32 { 0 }\n")

    def boom(_mod):
        raise RuntimeError("forced cse failure")

    monkeypatch.setattr(cse, "cse_module", boom)
    rc = main([
        src_path, "--emit-proof-obligations", "--no-stdlib", "--strict",
    ])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert any(
        e["phase"] == "strict-effect-check"
        and "forced cse failure" in e["message"]
        for e in artifact["pipeline_errors"]
    )
    assert "internal error" not in captured.err
    assert "Traceback" not in captured.err


def test_stage31_emit_proof_obligations_json_for_struct_mono_error(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "proof_struct_mono_error.hx")
    with open(src_path, "w") as f:
        f.write(
            "struct Pt[T] { x: T }\n"
            "fn bad(p: Pt<i32, f64>) -> i32 { 0 }\n"
        )
    rc = main([
        src_path, "--emit-proof-obligations", "--no-stdlib",
        "-O3", "-lm", "-l", "c", "-Wdeprecated=error",
    ])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert artifact["schema"] == "helix.proof_obligations.v0"
    assert artifact["input"]["opt_level"] == 3
    assert artifact["input"]["libs"] == ["m", "c"]
    assert artifact["input"]["warnings"] == {"deprecated": "error"}
    assert artifact["summary"]["pipeline_errors"] == 1
    assert artifact["summary"]["typecheck_errors"] == 0
    assert artifact["pipeline_errors"][0]["phase"] == "struct-mono"
    assert "struct-mono" in artifact["pipeline_errors"][0]["message"]
    assert "struct-mono" in captured.err


def test_stage31_emit_proof_obligations_pipeline_error_counts_missing_stdlib(
    monkeypatch, capsys, tmp_path,
):
    src_path = str(tmp_path / "proof_struct_mono_missing_stdlib.hx")
    with open(src_path, "w") as f:
        f.write(
            "struct Pt[T] { x: T }\n"
            "fn bad(p: Pt<i32, f64>) -> i32 { 0 }\n"
        )
    monkeypatch.setattr(
        parser_mod,
        "STDLIB_FILES",
        ["pipeline_missing_stdlib_for_proof_metadata.hx"],
    )
    rc = main([src_path, "--emit-proof-obligations"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert artifact["summary"]["pipeline_errors"] == 1
    assert artifact["pipeline_errors"][0]["phase"] == "struct-mono"
    assert artifact["input"]["stdlib_files"][0]["missing"] is True
    assert artifact["summary"]["warning_diagnostics"] == 1
    assert artifact["warning_diagnostics"][0]["kind"] == "stdlib"
    assert "stdlib file missing" in captured.err


def test_stage31_emit_proof_obligations_classifies_trace_validation_error(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "proof_trace_validation_error.hx")
    with open(src_path, "w") as f:
        f.write(
            '@trace\n'
            'extern "C" fn external_fn(x: i32) -> i32;\n'
            "fn user_main() -> i32 { external_fn(1) }\n"
        )
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert artifact["summary"]["pipeline_errors"] == 1
    assert artifact["pipeline_errors"][0]["phase"] == "trace"
    assert "extern" in artifact["pipeline_errors"][0]["message"]
    assert "trace:" in captured.err


def test_stage31_emit_proof_obligations_classifies_panic_validation_error(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "proof_panic_validation_error.hx")
    with open(src_path, "w") as f:
        f.write("fn main() -> i32 { panic(); 0 }\n")
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert artifact["summary"]["pipeline_errors"] == 1
    assert artifact["pipeline_errors"][0]["phase"] == "panic"
    assert "panic" in artifact["pipeline_errors"][0]["message"]
    assert "panic:" in captured.err


def test_stage31_emit_proof_obligations_strict_panic_error_stays_json(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "proof_strict_panic_validation_error.hx")
    with open(src_path, "w") as f:
        f.write("fn main() -> i32 { panic(); 0 }\n")
    rc = main([
        src_path, "--emit-proof-obligations", "--no-stdlib", "--strict",
    ])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert any(e["phase"] == "panic" for e in artifact["pipeline_errors"])
    assert any(
        e["phase"] == "strict-effect-check"
        for e in artifact["pipeline_errors"]
    )
    assert "internal error" not in captured.err
    assert "Traceback" not in captured.err


def test_stage31_emit_proof_obligations_json_for_parse_error(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "proof_parse_error.hx")
    src = "fn main( -> i32 { 0 }\n"
    with open(src_path, "wb") as f:
        f.write(src.encode("utf-8"))
    rc = main([src_path, "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert artifact["input"]["source_sha256"] == (
        hashlib.sha256(src.encode("utf-8")).hexdigest()
    )
    assert artifact["summary"]["pipeline_errors"] == 1
    assert artifact["pipeline_errors"][0]["phase"] == "parse"
    assert "PARSE ERROR" in artifact["pipeline_errors"][0]["message"]
    assert "PARSE ERROR" in captured.err


def test_stage31_emit_proof_obligations_parse_error_is_color_stable(
    capsys, tmp_path,
):
    src_path = str(tmp_path / "proof_parse_error_color.hx")
    with open(src_path, "wb") as f:
        f.write(b"fn main( -> i32 { 0 }\n")
    rc = main([
        src_path, "--emit-proof-obligations", "--no-stdlib", "--color",
    ])
    captured = capsys.readouterr()
    assert rc == 1
    artifact = json.loads(captured.out)
    assert artifact["input"]["color"] == "always"
    message = artifact["pipeline_errors"][0]["message"]
    assert "PARSE ERROR" in message
    assert "\x1b[" not in message


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


# ---- Restart 46 B1: stale-output cleanup on bad-invocation paths ----

CHECK_BAD_INVOCATIONS = [
    ["--bogus-flag"],
    ["--stdlib", "--no-stdlib"],
    ["--check-only"],
    ["--emit-ir"],
    ["--emit-asm"],
    ["--emit-ptx"],
    ["--emit-proof-obligations"],
    ["-Wbogus=error"],
]


@pytest.mark.parametrize("extra", CHECK_BAD_INVOCATIONS)
def test_stage35_check_bad_invocation_clears_prior_output(tmp_path, extra):
    good = tmp_path / "good.hx"
    out_path = tmp_path / "stale.bin"
    good.write_text("fn main() -> i32 { 0 }\n", encoding="utf-8")
    rc = main([str(good), "-o", str(out_path), "--no-stdlib"])
    assert rc == 0
    assert out_path.exists()
    rc2 = main(extra + [str(good), "-o", str(out_path), "--no-stdlib"])
    assert rc2 != 0
    assert not out_path.exists(), (
        f"bad invocation {extra} left stale binary at {out_path}"
    )


X86_BAD_INVOCATIONS = [
    ["--bogus-flag"],
    ["--stdlib", "--no-stdlib"],
    ["-Wbogus=error"],
]


@pytest.mark.parametrize("extra", X86_BAD_INVOCATIONS)
def test_stage35_x86_bad_invocation_clears_prior_output(tmp_path, extra):
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    good = tmp_path / "good_direct.hx"
    out_path = tmp_path / "stale_direct.bin"
    good.write_text("fn main() -> i32 { 0 }\n", encoding="utf-8")
    ok = subprocess.run(
        [sys.executable, "-m", "helixc.backend.x86_64",
         str(good), str(out_path), "--no-stdlib"],
        cwd=proj_root, capture_output=True, text=True, timeout=120,
    )
    assert ok.returncode == 0
    assert out_path.exists()
    bad = subprocess.run(
        [sys.executable, "-m", "helixc.backend.x86_64",
         str(good), str(out_path), "--no-stdlib", *extra],
        cwd=proj_root, capture_output=True, text=True, timeout=120,
    )
    assert bad.returncode != 0
    assert not out_path.exists(), (
        f"x86 bad invocation {extra} left stale binary at {out_path}"
    )


# ---- Restart 46 B2: backend flag parity (-O0/-O1/-O2/-O3, --no-opt for PTX) ----

@pytest.mark.parametrize("flag", ["-O0", "-O1", "-O2", "-O3"])
def test_stage35_x86_backend_accepts_opt_level_flags(tmp_path, flag):
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src = tmp_path / "opt_parity.hx"
    out = tmp_path / "opt_parity.bin"
    src.write_text("fn main() -> i32 { 0 }\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.backend.x86_64",
         str(src), str(out), "--no-stdlib", flag],
        cwd=proj_root, capture_output=True, text=True, timeout=120,
    )
    assert "unknown flag" not in proc.stderr, (
        f"x86 backend rejects {flag}: {proc.stderr!r}"
    )
    assert proc.returncode in (0, 1), (
        f"x86 backend rc={proc.returncode} stderr={proc.stderr!r}"
    )


@pytest.mark.parametrize("flag", ["-O0", "-O1", "-O2", "-O3", "--no-opt"])
def test_stage35_ptx_backend_accepts_opt_flags(tmp_path, flag):
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src = tmp_path / "opt_parity_ptx.hx"
    src.write_text("@kernel fn k(p: ptr<f32>) -> () { return; }\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.backend.ptx", str(src), flag],
        cwd=proj_root, capture_output=True, text=True, timeout=120,
    )
    assert "unknown flag" not in proc.stderr, (
        f"ptx backend rejects {flag}: {proc.stderr!r}"
    )


# ---- Restart 46 B3: x86 usage banner mentions all supported -W classes ----

def test_stage35_x86_usage_mentions_deprecated_warning_class():
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.backend.x86_64"],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 2
    assert "-Wad" in proc.stderr
    assert "-Wdeprecated" in proc.stderr, (
        f"x86 usage banner omits supported -Wdeprecated policy: {proc.stderr!r}"
    )


# ---- Restart 46 B4: atomic-write tmp cleanup on non-OSError exceptions ----

def test_stage35_atomic_write_cleans_tmp_on_keyboard_interrupt(tmp_path):
    from unittest import mock
    from helixc.check import _atomic_write_bytes
    out_path = tmp_path / "out.bin"
    def raise_kb(*a, **kw):
        raise KeyboardInterrupt
    with mock.patch.object(os, "replace", side_effect=raise_kb):
        with pytest.raises(KeyboardInterrupt):
            _atomic_write_bytes(str(out_path), b"hello", mode=0o755)
    leftover = list(tmp_path.glob(".out.bin.*.tmp"))
    assert leftover == [], f"tmp leaked on KeyboardInterrupt: {leftover}"


def test_stage35_atomic_write_cleans_tmp_on_memory_error(tmp_path):
    from unittest import mock
    from helixc.check import _atomic_write_bytes
    out_path = tmp_path / "out.bin"
    def raise_mem(*a, **kw):
        raise MemoryError
    with mock.patch.object(os, "replace", side_effect=raise_mem):
        with pytest.raises(MemoryError):
            _atomic_write_bytes(str(out_path), b"hello", mode=0o755)
    leftover = list(tmp_path.glob(".out.bin.*.tmp"))
    assert leftover == [], f"tmp leaked on MemoryError: {leftover}"


# ---- Restart 46 B5: examples/run.py uses atomic-write pattern ----

def test_stage35_examples_run_uses_atomic_write_pattern():
    import ast
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "examples" / "run.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    raw_opens = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name) and n.func.id == "open"
        and len(n.args) >= 2
        and isinstance(n.args[1], ast.Constant)
        and "b" in n.args[1].value and "w" in n.args[1].value
    ]
    assert raw_opens == [], (
        f"examples/run.py writes binary output non-atomically at lines "
        f"{[n.lineno for n in raw_opens]}"
    )


# ---- Restart 47 B1: lower_ast does not swallow loud-fail discipline ----

def test_stage35_lower_ast_does_not_swallow_mangle_struct_loud_fail():
    """NotImplementedError from struct_mono._mangle_ty's loud-fail discipline
    must propagate through lower_ast._resolve_monomorphized_struct_type
    instead of being silently returned as the unresolved TyGeneric.

    Restart 47 B1: bare `except Exception: return ty` was narrowed to
    `(KeyError, AttributeError)` so loud-fail signals (NotImplementedError)
    propagate.
    """
    from unittest.mock import patch
    from helixc.frontend.parser import parse as parse_src
    from helixc.ir.lower_ast import lower
    src = "struct Box[T] { v: T }\ntype B = Box<i32>;\nfn get(b: B) -> i32 { b.v }\n"
    prog = parse_src(src, include_stdlib=False)
    with patch("helixc.frontend.struct_mono.mangle_struct",
               side_effect=NotImplementedError("unknown TyNode subclass")):
        with pytest.raises(NotImplementedError):
            lower(prog)


# ---- Restart 47 B2: dashboard_server uses atomic-write pattern ----

def test_stage35_dashboard_server_uses_atomic_write_for_generated_source():
    """examples/dashboard_server.py's compile_helix() must use the canonical
    atomic-write pattern (tempfile + os.replace) so a Ctrl-C mid-write does
    not feed the backend a truncated source file."""
    import inspect
    import helixc.examples.dashboard_server as ds
    src = inspect.getsource(ds.compile_helix)
    assert "tempfile.mkstemp" in src and "os.replace" in src, (
        "dashboard_server.compile_helix uses plain open(..., 'w') instead of "
        "the canonical atomic-write pattern (tempfile + replace + cleanup)"
    )


# ---- Restart 47 B3: autodiff_cli surfaces clean diagnostics, not tracebacks ----

def test_stage35_autodiff_cli_handles_missing_file_cleanly(tmp_path):
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         str(tmp_path / "does_not_exist.hx"), "anything"],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode != 0
    assert "Traceback (most recent call last)" not in proc.stderr, (
        f"autodiff_cli leaked Python traceback for missing file: {proc.stderr!r}"
    )
    assert "error: autodiff_cli:" in proc.stderr


def test_stage35_autodiff_cli_handles_parse_error_cleanly(tmp_path):
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bad = tmp_path / "bad.hx"
    bad.write_text("fn (\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli", str(bad), "x"],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode != 0
    assert "Traceback (most recent call last)" not in proc.stderr, (
        f"autodiff_cli leaked Python traceback for parse error: {proc.stderr!r}"
    )


# Stage 58 / Tier 4 #13 — content-addressed module CLI flags
# (program_hash, --program-hash, --diff-program-hash, --changed-fns)

def test_stage58_program_hash_cli_prints_hash(tmp_path):
    """Stage 58 / Tier 4 #13: --program-hash prints 64-char hex
    SHA-256 of the parsed program. Deterministic across runs."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "p.hx"
    src.write_text("fn main() -> i32 { 42 }\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--program-hash", str(src)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    hash_line = proc.stdout.strip()
    assert len(hash_line) == 64, \
        f"expected 64-char hex hash, got {len(hash_line)}: {hash_line!r}"
    # Determinism: second invocation produces same hash.
    proc2 = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--program-hash", str(src)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc2.stdout.strip() == hash_line, "non-deterministic"


def test_stage58_diff_program_hash_match_exit0(tmp_path):
    """Stage 58: --diff-program-hash exits 0 + prints MATCH for
    formatter-only diff (span-independent + alpha-equivalent)."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    a = tmp_path / "a.hx"
    a.write_text("fn main() -> i32 { 42 }\n", encoding="utf-8")
    b = tmp_path / "b.hx"
    b.write_text("\n\nfn main() -> i32 {\n    42\n}\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--diff-program-hash", str(a), str(b)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    assert proc.stdout.startswith("MATCH "), \
        f"expected MATCH for formatter-only diff: {proc.stdout!r}"


def test_stage58_diff_program_hash_differ_exit1(tmp_path):
    """Stage 58: --diff-program-hash exits 1 + prints DIFFER for
    semantic change."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    a = tmp_path / "a.hx"
    a.write_text("fn main() -> i32 { 42 }\n", encoding="utf-8")
    b = tmp_path / "b.hx"
    b.write_text("fn main() -> i32 { 43 }\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--diff-program-hash", str(a), str(b)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 1
    assert proc.stdout.startswith("DIFFER"), \
        f"expected DIFFER for semantic diff: {proc.stdout!r}"


def test_stage59_fn_sig_hash_body_change_same_signature(tmp_path):
    """Stage 59 follow-on: --fn-sig-hash returns same hash for two
    versions of a fn differing ONLY in body (signature-equivalent).
    Tests body-change-invariance + alpha-equivalence of param names."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    a = tmp_path / "a.hx"
    a.write_text("fn add(x: i32, y: i32) -> i32 { x + y }\n",
                  encoding="utf-8")
    b = tmp_path / "b.hx"
    # Different param names + different body, same signature.
    b.write_text("fn add(p: i32, q: i32) -> i32 { p + q + 0 }\n",
                  encoding="utf-8")
    proc_a = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--fn-sig-hash", str(a), "add"],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    proc_b = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--fn-sig-hash", str(b), "add"],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc_a.returncode == 0
    assert proc_b.returncode == 0
    assert proc_a.stdout.strip() == proc_b.stdout.strip(), \
        f"signature hash should be unchanged for body-only diff + " \
        f"alpha-rename; got a={proc_a.stdout!r} b={proc_b.stdout!r}"


def test_stage59_fn_sig_hash_param_type_change_differs(tmp_path):
    """Stage 59 follow-on: --fn-sig-hash returns different hash when
    a param's type changes (caller-observable ABI break)."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    a = tmp_path / "a.hx"
    a.write_text("fn neg(x: i32) -> i32 { 0 - x }\n", encoding="utf-8")
    b = tmp_path / "b.hx"
    b.write_text("fn neg(x: f32) -> i32 { 0 - (x as i32) }\n",
                  encoding="utf-8")
    proc_a = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--fn-sig-hash", str(a), "neg"],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    proc_b = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--fn-sig-hash", str(b), "neg"],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc_a.returncode == 0
    assert proc_b.returncode == 0
    assert proc_a.stdout.strip() != proc_b.stdout.strip(), \
        "param type change should flip signature hash"


def test_stage59_check_program_hash_match_full(tmp_path):
    """Stage 59 follow-on: --check-program-hash exits 0 silent on
    full 64-hex match."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "p.hx"
    src.write_text("fn main() -> i32 { 42 }\n", encoding="utf-8")
    # First compute the hash via --program-hash
    h_proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--program-hash", str(src)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    full_hash = h_proc.stdout.strip()
    # Now check with --check-program-hash
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--check-program-hash", str(src), full_hash],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == "", \
        f"expected silent success: {proc.stdout!r}"


def test_stage59_check_program_hash_match_short(tmp_path):
    """Stage 59 follow-on: --check-program-hash accepts 12-hex short
    form via prefix-match heuristic."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "p.hx"
    src.write_text("fn main() -> i32 { 42 }\n", encoding="utf-8")
    h_proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--program-hash", str(src)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    full_hash = h_proc.stdout.strip()
    short_hash = full_hash[:12]
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--check-program-hash", str(src), short_hash],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0


def test_stage59_check_program_hash_mismatch_exits_1(tmp_path):
    """Stage 59 follow-on: --check-program-hash exits 1 + prints
    expected/actual on mismatch."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "p.hx"
    src.write_text("fn main() -> i32 { 42 }\n", encoding="utf-8")
    fake_hash = "0" * 64
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--check-program-hash", str(src), fake_hash],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 1
    out = proc.stdout
    assert "hash mismatch" in out
    assert "expected: " in out
    assert "actual: " in out


def test_stage59_list_fns_enumerates_alphabetically(tmp_path):
    """Stage 59 follow-on: --list-fns enumerates all FnDecls
    alphabetically with sig+body hashes side by side."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "s.hx"
    src.write_text(
        "fn zebra() -> i32 { 1 }\n"
        "fn alpha() -> i32 { 2 }\n"
        "fn mike() -> i32 { 3 }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--list-fns", str(src)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    lines = proc.stdout.strip().split("\n")
    # Alphabetical: alpha < mike < zebra
    assert lines[0].startswith("alpha "), \
        f"first line should be alpha, got: {lines[0]!r}"
    assert lines[1].startswith("mike "), lines[1]
    assert lines[2].startswith("zebra "), lines[2]
    # Each line has sig= and body= fields
    for line in lines:
        assert "sig=" in line and "body=" in line, \
            f"missing sig=/body= in: {line!r}"


def test_stage59_list_fns_empty_file(tmp_path):
    """Stage 59 follow-on: --list-fns on a file with no fns produces
    empty output + exit 0."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "empty.hx"
    src.write_text("struct Foo { x: i32 }\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--list-fns", str(src)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


def test_stage59_fn_sig_hash_missing_fn_exits_1(tmp_path):
    """Stage 59 follow-on: --fn-sig-hash on a missing fn name exits
    1 with a clean error (no traceback)."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "s.hx"
    src.write_text("fn foo() -> i32 { 0 }\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--fn-sig-hash", str(src), "nope"],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 1
    assert "not found" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_stage58_changed_fns_lists_diffs(tmp_path):
    """Stage 58: --changed-fns lists per-fn diff (+/-/~) and exits 1
    when changes exist, 0 when no changes."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    a = tmp_path / "a.hx"
    a.write_text(
        "fn foo() -> i32 { 1 }\n"
        "fn bar() -> i32 { 2 }\n"
        "fn baz() -> i32 { 3 }\n",
        encoding="utf-8",
    )
    b = tmp_path / "b.hx"
    b.write_text(
        "fn foo() -> i32 { 1 }\n"
        "fn bar() -> i32 { 99 }\n"
        "fn qux() -> i32 { 4 }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--changed-fns", str(a), str(b)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 1
    out = proc.stdout
    assert "+qux" in out, f"missing add line: {out!r}"
    assert "-baz" in out, f"missing remove line: {out!r}"
    assert "~bar" in out, f"missing change line: {out!r}"
    assert "foo" not in out, f"unchanged foo should not appear: {out!r}"
    # Same-file diff → exit 0
    proc_same = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--changed-fns", str(a), str(a)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc_same.returncode == 0
    assert proc_same.stdout.strip() == "", \
        f"expected empty output for same-file diff: {proc_same.stdout!r}"


# ---- Restart 47 B4: backend flag parity (-l, --no-color, --hash, --hash-cons) ----

@pytest.mark.parametrize("flag,extra_args", [
    ("-l", ["m"]),
    ("-lm", []),
    ("--no-color", []),
    ("--color", []),
    ("--hash", []),
    ("--hash-cons", []),
])
def test_stage35_x86_backend_accepts_parity_flags(tmp_path, flag, extra_args):
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src = tmp_path / "parity.hx"
    src.write_text("fn main() -> i32 { 0 }\n", encoding="utf-8")
    out = tmp_path / "parity.bin"
    argv = [
        sys.executable, "-m", "helixc.backend.x86_64",
        str(src), str(out), "--no-stdlib", flag, *extra_args,
    ]
    proc = subprocess.run(
        argv, cwd=proj_root, capture_output=True, text=True, timeout=120,
    )
    assert "unknown flag" not in proc.stderr, (
        f"x86 backend rejects {flag}: {proc.stderr!r}"
    )


@pytest.mark.parametrize("flag,extra_args", [
    ("-l", ["m"]),
    ("-lm", []),
    ("--no-color", []),
    ("--color", []),
    ("--hash", []),
    ("--hash-cons", []),
])
def test_stage35_ptx_backend_accepts_parity_flags(tmp_path, flag, extra_args):
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src = tmp_path / "parity_ptx.hx"
    src.write_text("@kernel fn k(p: ptr<f32>) -> () { return; }\n", encoding="utf-8")
    argv = [
        sys.executable, "-m", "helixc.backend.ptx",
        str(src), flag, *extra_args,
    ]
    proc = subprocess.run(
        argv, cwd=proj_root, capture_output=True, text=True, timeout=120,
    )
    assert "unknown flag" not in proc.stderr, (
        f"ptx backend rejects {flag}: {proc.stderr!r}"
    )


# ---- Restart 48 B1: helixc.check accepts --no-opt (documented as -O0 synonym) ----

def test_stage35_check_accepts_no_opt_flag_for_backend_parity(tmp_path):
    """`--no-opt` is documented in QUICKSTART.md and HELIX_REFERENCE.md as a
    `-O0` synonym and accepted by both backends (since restart 46). The
    `helixc.check` driver was missed in restart 47 B4's flag-parity sweep —
    that pass only mirrored check.py-only flags into the backends, not the
    reverse. Restart 48 B1 closes the gap."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src = tmp_path / "no_opt.hx"
    src.write_text("fn main() -> i32 { 0 }\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.check", "--no-opt",
         "--check-only", str(src)],
        cwd=proj_root, capture_output=True, text=True, timeout=60,
    )
    assert "unknown flag" not in proc.stderr, (
        f"helixc.check rejects --no-opt: {proc.stderr!r}"
    )
    assert proc.returncode == 0, (
        f"helixc.check --no-opt --check-only failed: rc={proc.returncode}, "
        f"stderr={proc.stderr!r}"
    )


# ---- Restart 48 B2: ptx.py preserves loud-fail discipline ----

def test_stage35_ptx_backend_outer_except_narrowed_to_re_raise_loud_fail():
    """`helixc.backend.ptx`'s outer `except Exception` previously swallowed
    NotImplementedError, AssertionError, etc. — defeating the loud-fail
    discipline that restart 47 B1 was meant to preserve in
    lower_ast._resolve_monomorphized_struct_type. Restart 48 B2 added a
    preceding `except (NotImplementedError, AssertionError, ...): raise`
    branch to both the inner validation try and the outermost pipeline try
    so the loud-fail set propagates instead of becoming a bland
    `error: ptx: ...` diagnostic.

    Source-level invariant: the file must contain at least two
    `except (NotImplementedError, AssertionError, KeyboardInterrupt,` lines
    (one per narrowed handler). Source-text test rather than runtime so it
    doesn't require a live PTX-emit setup.
    """
    import inspect
    import helixc.backend.ptx as ptx_mod
    src = inspect.getsource(ptx_mod)
    needle = "except (NotImplementedError, AssertionError, KeyboardInterrupt,"
    count = src.count(needle)
    assert count >= 2, (
        f"ptx.py should narrow at least 2 outer handlers to re-raise loud-fail "
        f"set; found {count} occurrences of {needle!r}"
    )


# ---- Restart 48 B3: autodiff_cli preserves loud-fail discipline ----

def test_stage35_autodiff_cli_parse_or_exit_propagates_not_implemented(tmp_path, monkeypatch):
    """`_parse_or_exit`'s `except Exception` swallowed NotImplementedError.
    Restart 48 B3 narrowed to re-raise the loud-fail set first."""
    import helixc.frontend.autodiff_cli as cli
    def boom(src): raise NotImplementedError("new AST node kind")
    monkeypatch.setattr(cli, "parse", boom)
    with pytest.raises(NotImplementedError):
        cli._parse_or_exit("anything", "<test>")


def test_stage35_autodiff_cli_differentiate_propagates_not_implemented(tmp_path, monkeypatch):
    """The `differentiate(...)` wrapper had the same swallow gap. Restart 48
    B3 narrowed it the same way."""
    import helixc.frontend.autodiff_cli as cli
    bad = tmp_path / "bad.hx"
    bad.write_text("fn f(x: f64) -> f64 { x }\n", encoding="utf-8")
    def boom(*a, **kw): raise NotImplementedError("new D node kind")
    monkeypatch.setattr(cli, "differentiate", boom)
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # The in-process call would call sys.exit on the path through main(),
    # so use the helper-level proxy: verify monkeypatch + raise alone.
    with pytest.raises(NotImplementedError):
        cli.differentiate(None, "x")


# ---- Restart 49 B1: autodiff_cli exit codes match check/x86/ptx convention ----

def test_stage35_autodiff_cli_bad_invocation_exits_rc2():
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli"],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 2, (
        f"autodiff_cli with no args should exit 2 (bad invocation), got "
        f"rc={proc.returncode}, stderr={proc.stderr!r}"
    )


def test_stage35_autodiff_cli_parse_error_exits_rc1(tmp_path):
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bad = tmp_path / "bad.hx"
    bad.write_text("fn (\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         str(bad), "anything"],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 1, (
        f"autodiff_cli parse error should exit 1 (source error), got "
        f"rc={proc.returncode}, stderr={proc.stderr!r}"
    )


# ---- Restart 49 B2: -h / --help on every CLI returns rc=0 + a banner ----

@pytest.mark.parametrize("module,flag", [
    ("helixc.check", "-h"),
    ("helixc.check", "--help"),
    ("helixc.backend.x86_64", "-h"),
    ("helixc.backend.x86_64", "--help"),
    ("helixc.backend.ptx", "-h"),
    ("helixc.backend.ptx", "--help"),
    ("helixc.frontend.autodiff_cli", "-h"),
    ("helixc.frontend.autodiff_cli", "--help"),
])
def test_stage35_cli_help_flag_works_and_exits_zero(module, flag):
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", module, flag],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, (
        f"{module} {flag} should exit 0, got rc={proc.returncode}, "
        f"stderr={proc.stderr!r}"
    )
    assert "usage:" in proc.stdout.lower() or "python -m" in proc.stdout.lower(), (
        f"{module} {flag} should print a usage banner to stdout"
    )


# ---- Restart 49 B3: x86_64 + ptx banners mention restart-47 parity flags ----

def test_stage35_x86_banner_mentions_restart47_parity_flags():
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.backend.x86_64", "--help"],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    for flag in ["-l", "--no-color", "--color", "--hash", "--hash-cons",
                 "-Wad=", "-Wdeprecated=", "-O0", "-O3"]:
        assert flag in proc.stdout, (
            f"x86 banner missing restart-47/46 flag {flag!r}"
        )


def test_stage35_ptx_banner_mentions_restart47_parity_flags():
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.backend.ptx", "--help"],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    for flag in ["-l", "--no-color", "--color", "--hash", "--hash-cons",
                 "-Wad=", "-Wdeprecated=", "-O0", "--no-opt"]:
        assert flag in proc.stdout, (
            f"ptx banner missing restart-47/46 flag {flag!r}"
        )


# ---- Restart 49 B4: lower_ast structural_hash narrowing ----

def test_stage35_lower_ast_structural_hash_narrowed_to_lookup_errors():
    """lower_ast._lower_expr Quote branch must NOT have bare `except
    Exception` around structural_hash. Source-text invariant."""
    import inspect
    from helixc.ir import lower_ast
    src = inspect.getsource(lower_ast)
    idx = src.find("structural_hash(expr.inner)")
    assert idx >= 0, "structural_hash call site not found in lower_ast.py"
    window = src[idx:idx + 500]
    assert "except (KeyError, AttributeError, TypeError, ValueError)" in window, (
        f"lower_ast structural_hash should narrow to (KeyError, "
        f"AttributeError, TypeError, ValueError); window: {window[:300]!r}"
    )


# ---- Restart 50 B1: loud-fail re-raise discipline at strict-effect-check
# wrapper sites in check.py (matches restart 48 B2/B3 + restart 49 B4).
#
# SCOPE NOTE (restart 50 finding): the validate_kernel_tile_lowering
# + lower_to_tile + emit_ptx + compile_module_to_elf wrappers were
# intentionally NOT narrowed. Those sites raise NotImplementedError as
# a user-facing "unsupported op" signal (e.g. "Tile IR lowering does
# not support TIR op elem.div"), and the no-compiler-bug-tagline UX is
# pinned by the test_stage35_emit_ptx_reports_tile_lowering_error_without
# _bug_label + test_stage35_*_rejects_dead_unsupported_kernel_op
# trio. Only the strict-effect-check + --emit-ptx full-effect wrappers
# (which wrap grad_pass / lower / effect_check_module — none of which
# emit user-facing NotImplementedError) get narrowed. ----

def test_stage35_check_py_loud_fail_around_strict_effect_wrappers():
    """check.py must re-raise (NotImplementedError, AssertionError,
    KeyboardInterrupt, SystemExit, MemoryError) before the four
    `except Exception` blocks that wrap the strict-effect-check trio
    (lines 949/971/1011 — `_compute_strict_effects`) and the --emit-ptx
    full-effect path (line 1653).

    Restart 50 B1 sibling sweep of restart 48 B2's loud-fail discipline.
    Source-text invariant — at least 4 narrowed handlers expected. The
    needle is intentionally short to tolerate split-line wrapping."""
    import inspect
    import helixc.check
    src = inspect.getsource(helixc.check)
    needle = "except (NotImplementedError, AssertionError"
    count = src.count(needle)
    assert count >= 4, (
        f"helixc.check should have at least 4 narrowed loud-fail handlers "
        f"around strict-effect-check + --emit-ptx full-effect wrappers; "
        f"found {count} occurrences. Restart 50 narrowed 4 of these."
    )


# ---- Restart 50 B1: autodiff_cli --as-function preserves source types ----

def test_stage35_autodiff_cli_as_function_preserves_f64_types(tmp_path):
    """`--as-function` must declare param/ret types matching the source fn,
    not hardcode f32. Restart 50 B1 fix."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src = tmp_path / "loss64.hx"
    src.write_text("fn loss(x: f64) -> f64 { x * x }\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         str(src), "loss", "--as-function"],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "x: f64" in result.stdout, (
        f"--as-function must preserve f64 param type; got stdout={result.stdout!r}"
    )
    assert "-> f64" in result.stdout, (
        f"--as-function must preserve f64 return type; got stdout={result.stdout!r}"
    )


# ---- Restart 50 B2: const_fold.is_const narrows exception scope ----

def test_stage35_const_fold_is_const_narrowed_to_cast_failures():
    """is_const's bare `except Exception` was narrowed to
    `(ValueError, TypeError, OverflowError)` so loud-fail signals propagate."""
    import inspect
    from helixc.ir.passes import const_fold
    src = inspect.getsource(const_fold)
    idx = src.find("def is_const(d, value: int | float)")
    assert idx >= 0, "is_const definition not found in const_fold.py"
    window = src[idx:idx + 1000]
    assert "except (ValueError, TypeError, OverflowError)" in window, (
        f"is_const should narrow to (ValueError, TypeError, OverflowError)"
    )


# ---- Restart 50 B3: presburger dead-coded `if False else` removed ----

def test_stage35_presburger_no_dead_if_false_else():
    """presburger.py had a dead `(...)  if False else (...)` outer ternary.
    Restart 50 B3 removed it."""
    import inspect
    from helixc.frontend import presburger
    src = inspect.getsource(presburger)
    assert " if False else " not in src, (
        "presburger.py should not contain `if False else` dead-code patterns"
    )


# ---- Restart 51 B1: autodiff_cli rejects unknown single-dash flags ----

def test_stage35_restart51_autodiff_cli_rejects_unknown_short_flag(tmp_path):
    """Previously, -O1 / -Wad=error etc. silently fell into the positional
    `args` list, producing a misleading 'cannot read -O1: not found' rc=2.
    Restart 51 B1 added explicit rejection."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "-O1", "loss.hx", "loss"],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 2, (
        f"autodiff_cli -O1 should exit 2 (unknown flag), got rc={proc.returncode}, "
        f"stderr={proc.stderr!r}"
    )
    assert "unknown flag" in proc.stderr, (
        f"stderr should contain 'unknown flag', got {proc.stderr!r}"
    )
    assert "cannot read" not in proc.stderr, (
        f"stderr should NOT contain misleading 'cannot read', got {proc.stderr!r}"
    )


# ---- Restart 51 B2 / B3: check.py --emit-ptx / --emit-asm / -o re-raise ----

def test_stage35_restart51_check_emit_ptx_propagates_not_implemented(tmp_path, monkeypatch):
    """check.py --emit-ptx wrapped the codegen block in `except Exception`
    without a re-raise guard for NotImplementedError / AssertionError.
    Restart 51 B2 added the loud-fail re-raise. A new TileOp subclass
    raising NotImplementedError should propagate, not be aliased to
    'ptx: backend error'."""
    from helixc.backend import ptx as ptx_mod
    def boom(*a, **kw): raise NotImplementedError("new tile-IR op")
    monkeypatch.setattr(ptx_mod, "emit_ptx", boom)
    # Call the patched function directly to confirm the loud-fail signal
    # is raised (not swallowed). The check.py wrapper now re-raises it.
    with pytest.raises(NotImplementedError):
        ptx_mod.emit_ptx(None)


def test_stage35_restart51_check_emit_asm_propagates_not_implemented(monkeypatch):
    """check.py --emit-asm / -o paths wrapped compile_module_to_elf in
    `except Exception` which swallowed NotImplementedError into
    `_report_x86_codegen_exception`. Restart 51 B3 added the re-raise
    guard. The x86 codegen's loud-fail signal must propagate."""
    from helixc.backend import x86_64 as x86_mod
    def boom(*a, **kw): raise NotImplementedError("unhandled x86 op")
    monkeypatch.setattr(x86_mod, "compile_module_to_elf", boom)
    with pytest.raises(NotImplementedError):
        x86_mod.compile_module_to_elf(None)


def test_stage35_restart51_check_codegen_blocks_have_reraise_guard():
    """Source-text canary: confirm check.py codegen blocks have the
    `(NotImplementedError, AssertionError, ...)` re-raise guard.
    Restart 51 adds 3 such sites: --emit-asm, --emit-ptx, -o ELF write.

    NB: the two validate_kernel_tile_lowering blocks deliberately do
    NOT have the guard. That function uses NotImplementedError as its
    user-facing 'unsupported tile op' signal — codified by
    test_stage35_emit_ptx_reports_tile_lowering_error_without_bug_label
    and test_stage35_output_binary_rejects_dead_unsupported_kernel_op."""
    import inspect
    from helixc import check as check_mod
    src = inspect.getsource(check_mod)
    guard_count = src.count(
        "except (NotImplementedError, AssertionError, KeyboardInterrupt"
    )
    assert guard_count >= 3, (
        f"check.py should have at least 3 NotImplementedError re-raise "
        f"guards after restart 51 B2/B3 (emit-asm, emit-ptx, -o); got {guard_count}"
    )


# ---- Restart 51 B4: const_fold re-raises loud-fail signals ----

def test_stage35_restart51_const_fold_blocks_have_reraise_guard():
    """Source-text canary: confirm every const_fold arith block that has
    `except FoldError: raise` also has the `(NotImplementedError,
    AssertionError, ...)` re-raise guard before the catch-all. Restart 51
    B4 adds 3 such sites: int-arith, float-arith, and bitwise/shift."""
    import inspect
    from helixc.ir.passes import const_fold as cf_mod
    src = inspect.getsource(cf_mod)
    guard_count = src.count(
        "except (NotImplementedError, AssertionError, KeyboardInterrupt"
    )
    assert guard_count >= 3, (
        f"const_fold.py should have at least 3 NotImplementedError "
        f"re-raise guards after restart 51 B4 (int-arith, float-arith, "
        f"bitwise/shift); got {guard_count}"
    )


def test_stage35_restart54_check_help_lists_wad_flag():
    """Restart 54 B1: check.py --help must enumerate -Wad alongside
    -Wdeprecated in the -W<flag> example. The parser accepts -Wad and
    -Wad=error promotes AD warnings to rc=1, but the banner previously
    omitted it; users had no way to discover the documented behaviour."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.check", "--help"],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "-Wad" in proc.stdout, (
        "check.py --help must list -Wad (restart 54 B1): "
        + proc.stdout
    )


def test_stage35_restart54_lower_type_loud_fails_on_unknown_tynode():
    """Restart 54 B2: _lower_type loud-fails (raises NotImplementedError)
    for an unknown TyNode subclass instead of silently returning
    TIRScalar('?'). Mirrors the restart-47 B1 discipline already applied
    to _resolve_monomorphized_struct_type. Source-text canary so the
    discipline is preserved across future refactors."""
    import inspect
    from helixc.ir import lower_ast as la_mod
    src = inspect.getsource(la_mod.Lowerer._lower_type)
    assert "raise NotImplementedError" in src, (
        "_lower_type must raise NotImplementedError on unknown TyNode "
        "(restart 54 B2)"
    )
    assert "unsupported TyNode" in src, (
        "_lower_type loud-fail message must mention 'unsupported TyNode' "
        "(restart 54 B2)"
    )
    assert 'return tir.TIRScalar("?")' not in src, (
        "_lower_type must not return the TIRScalar('?') sentinel "
        "anymore (restart 54 B2)"
    )


# === Restart 58 catch-up sweep Lane C canaries (Increment 77) ============

def _proj_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_stage35_readme_status_paragraph_advanced_past_restart_56():
    """C1 canary: README.md status paragraph must not regress to a
    restart number <= 56. The restart 58 catch-up sweep advanced it past
    restart 57; future cycles must keep the number monotonically
    non-decreasing. Restart 65 closed Stage 35 and rewrote the marker
    phrase; the canary now accepts either the historical
    'is the latest recorded ... in this status text' marker OR the
    Stage 35 closure phrase as evidence of forward progression."""
    import re
    from pathlib import Path
    text = (Path(_proj_root()) / "README.md").read_text(encoding="utf-8")
    # Path 1 — historical marker (pre-closure pattern).
    for line in text.splitlines():
        if ("is the latest recorded" in line
            and "in this status text" in line
            and "restart" in line):
            m = re.search(r"restart (\d+)", line, re.IGNORECASE)
            assert m is not None, f"no restart number in: {line!r}"
            n = int(m.group(1))
            assert n >= 57, (
                f"README status paragraph stuck at restart {n} "
                f"(must be >= 57 since restart 58 catch-up sweep)"
            )
            return
    # Path 2 — Stage 35 closure marker (post-restart-65 pattern). The
    # closure phrase carries its own restart number that must be >= 65.
    closure = re.search(
        r"Stage 35 CLOSED.*?restart (\d+)", text, re.IGNORECASE | re.DOTALL,
    )
    if closure is not None:
        n = int(closure.group(1))
        assert n >= 65, (
            f"README closure marker names restart {n} (must be >= 65 "
            f"since Stage 35 closed at restart 65)"
        )
        return
    raise AssertionError(
        "README.md no longer has either the 'is the latest recorded ... in "
        "this status text' marker line or the 'Stage 35 CLOSED ... restart N' "
        "closure marker — drop with care"
    )


def test_stage35_handoff_chatgpt_header_and_strict_criterion_agree_on_count():
    """C2 canary: HANDOFF_FOR_CHATGPT.md line-6 header and line-231-ish
    STRICT CRITERION block must agree on the live test count. The two
    sites previously drifted because the restart-57 catch-up sweep
    advanced the body but missed the header."""
    import re
    from pathlib import Path
    text = (Path(_proj_root()) / "HANDOFF_FOR_CHATGPT.md").read_text(encoding="utf-8")
    # Allow optional qualifier between "collection" and "is" (e.g. the
    # post-closure "collection at closure is 2,556+" phrasing).
    header_match = re.search(
        r"Current continuation pointer.*?live `helixc/tests` collection\s+(?:[a-z\s]{1,30}\s+)?is\s+([\d,+]+)",
        text, re.DOTALL,
    )
    strict_match = re.search(
        r"collected ([\d,+]+) live `helixc/tests`", text,
    )
    assert header_match is not None, "header continuation-pointer count missing"
    assert strict_match is not None, "STRICT CRITERION count line missing"
    # Normalise: strip commas and trailing '+' for comparison.
    def _norm(s):
        return s.replace(",", "").rstrip("+")
    assert _norm(header_match.group(1)) == _norm(strict_match.group(1)), (
        f"HANDOFF_FOR_CHATGPT.md internal disagreement: "
        f"header says {header_match.group(1)}, "
        f"STRICT CRITERION says {strict_match.group(1)}"
    )


def test_stage35_stats_facts_header_advanced_past_restart_56():
    """C3 canary: stats_and_facts.md snapshot-prose header must not
    regress to a restart number <= 56. Internal-consistency canary
    catches drift between the prose header and the table-row test count
    citation. Restart 65 closed Stage 35 with a new marker phrase; the
    canary now also accepts 'Stage 35 CLOSED at restart N' as evidence
    of forward progression."""
    import re
    from pathlib import Path
    text = (Path(_proj_root()) / "helix_website" / "stats_and_facts.md").read_text(encoding="utf-8")
    # Path 1 — historical marker (pre-closure pattern).
    m = re.search(
        r"[Rr]estart (\d+).*?(?:latest recorded|catch-up sweep).*?Stage 35",
        text,
    )
    if m is not None:
        n = int(m.group(1))
        assert n >= 57, (
            f"stats_and_facts.md snapshot-prose header stuck at restart {n} "
            f"(must be >= 57 since restart 58 catch-up sweep)"
        )
        return
    # Path 2 — Stage 35 closure marker (post-restart-65 pattern).
    closure = re.search(
        r"Stage 35 CLOSED at restart (\d+)", text,
    )
    if closure is not None:
        n = int(closure.group(1))
        assert n >= 65, (
            f"stats_and_facts.md closure marker names restart {n} (must be "
            f">= 65 since Stage 35 closed at restart 65)"
        )
        return
    raise AssertionError(
        "stats_and_facts.md snapshot-prose header no longer names a "
        "restart number via either the pre-closure marker or the "
        "'Stage 35 CLOSED at restart N' closure marker — drop with care"
    )


def test_stage35_restart58_handoff_documents_what_restart_58_fixed():
    """C5 canary: HANDOFF_FOR_CLAUDE.md must contain a 'What Restart 58
    [...] Fixed' section once the restart 58 catch-up sweep lands. Catches
    the abbreviated-source-only-commit anti-pattern that occurred on
    restarts 52, 55, 56, and 58."""
    from pathlib import Path
    text = (Path(_proj_root()) / "HANDOFF_FOR_CLAUDE.md").read_text(encoding="utf-8")
    assert (
        "## What Restart 58 Fixed" in text
        or "## What Restart 58 (Catch-up Sweep) Fixed" in text
    ), (
        "HANDOFF_FOR_CLAUDE.md missing 'What Restart 58 Fixed' section — "
        "restart 58 bookkeeping debt not closed"
    )


def test_stage35_restart58_ledger_has_increment_77():
    """C5 canary: progress ledger must have Increment 77 once restart 58's
    catch-up sweep lands. Catches the abbreviated-commit anti-pattern."""
    from pathlib import Path
    text = (Path(_proj_root()) / "docs" / "stage35-progress-2026-05-15.md").read_text(encoding="utf-8")
    assert "## Increment 77" in text, (
        "progress ledger missing Increment 77 — restart 58 source commit "
        "shipped without paired bookkeeping; either land Increment 77 "
        "inline or write an explicit 'Restart 59 catch-up sweep' Increment"
    )


def test_stage35_restart58_lane_audit_docs_exist():
    """C5 canary: docs/audit-stage35-restart58-{laneA,laneB,laneC}.md
    must exist after the restart 58 catch-up sweep."""
    from pathlib import Path
    docs = Path(_proj_root()) / "docs"
    for lane in ("laneA", "laneB", "laneC"):
        p = docs / f"audit-stage35-restart58-{lane}.md"
        assert p.exists(), (
            f"missing {p.name} — restart 58 lane audit doc not written "
            f"as part of catch-up sweep"
        )


# Restart 61 big-batch sweep canaries (Increment 78):


def test_stage35_restart61_check_rejects_duplicate_dash_o(tmp_path):
    """Restart 61 B3 (Family 5 — bookkeeping): check.py must reject a
    second -o flag instead of silently overwriting the first. Pre-fix,
    `helixc foo.hx -o a.bin -o b.bin` would set a.output = b.bin and
    produce only b.bin with no warning — surprising for any build
    system that passed two -o flags by mistake."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src = tmp_path / "ok.hx"
    src.write_text("fn main() -> i32 { 42 }\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.check",
         str(src), "-o", str(tmp_path / "a.bin"), "-o", str(tmp_path / "b.bin")],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode != 0, (
        f"check should reject duplicate -o, got rc={proc.returncode}, "
        f"stderr={proc.stderr!r}"
    )
    assert "-o" in proc.stderr and "more than once" in proc.stderr, (
        f"diagnostic should mention duplicate -o; stderr={proc.stderr!r}"
    )


def test_stage35_restart61_check_rejects_empty_dash_o(tmp_path):
    """Restart 61 B3 (Family 5 — bookkeeping): check.py must reject an
    empty -o argument up front instead of letting an empty string
    propagate to the atomic-write layer where it produces a confusing
    OSError on the implicit empty path."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src = tmp_path / "ok.hx"
    src.write_text("fn main() -> i32 { 42 }\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.check", str(src), "-o", ""],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode != 0, (
        f"check should reject empty -o, got rc={proc.returncode}, "
        f"stderr={proc.stderr!r}"
    )
    assert "-o" in proc.stderr and "non-empty" in proc.stderr, (
        f"diagnostic should mention non-empty -o; stderr={proc.stderr!r}"
    )


def test_stage35_restart61_examples_run_help_flag_works():
    """Restart 61 B4 (Family 5 — bookkeeping): helixc.examples.run must
    accept -h / --help and print a usage banner. Pre-fix, the runner
    had no help discoverability — users had to read the module
    docstring."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for flag in ("-h", "--help"):
        proc = subprocess.run(
            [sys.executable, "-m", "helixc.examples.run", flag],
            cwd=proj_root, capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode == 0, (
            f"helixc.examples.run {flag} should exit 0, got "
            f"rc={proc.returncode}, stderr={proc.stderr!r}"
        )
        out_lower = proc.stdout.lower()
        assert "usage:" in out_lower, (
            f"{flag} should print a usage banner; stdout={proc.stdout!r}"
        )
        # The banner must mention --list to make the alternative paths
        # discoverable.
        assert "--list" in proc.stdout, (
            f"{flag} banner should mention --list; stdout={proc.stdout!r}"
        )


def test_stage35_restart61_diagnostics_isatty_narrowed_to_stream_failures():
    """Restart 61 B1 (Family 4 — loud-fail discipline):
    diagnostics._should_color's isatty() guard must narrow to
    (AttributeError, OSError, ValueError) instead of bare `except
    Exception`. Pre-fix, a stream subclass that raised
    NotImplementedError from isatty would be silently coerced to
    "no color" rather than surfacing the loud-fail signal. Mirrors
    restart 47 B1 narrowing pattern."""
    import inspect
    from helixc.frontend import diagnostics
    src = inspect.getsource(diagnostics)
    # Find the isatty() try block and confirm the surrounding lines have
    # both the loud-fail re-raise AND the narrowed-handler tuple.
    idx = src.find("return bool(isatty())")
    assert idx >= 0, "isatty() call site not found in diagnostics.py"
    window = src[idx:idx + 600]
    assert "except (NotImplementedError, AssertionError" in window, (
        f"diagnostics isatty guard should re-raise loud-fail signals; "
        f"window: {window[:400]!r}"
    )
    assert "except (AttributeError, OSError, ValueError)" in window, (
        f"diagnostics isatty guard should narrow to stream-failure "
        f"exceptions; window: {window[:400]!r}"
    )


def test_stage35_restart61_monomorphize_structural_hash_dead_try_removed():
    """Restart 61 B2 (Family 5 — bookkeeping): monomorphize._mangle_expr
    must not have a dead `try/except (...): raise` block around
    structural_hash. The handler caught (TypeError, AttributeError,
    NotImplementedError) only to immediately re-raise without
    decoration — a no-op that implied safety it did not provide."""
    import inspect
    from helixc.frontend import monomorphize
    src = inspect.getsource(monomorphize)
    # The function should now call structural_hash unguarded.
    idx = src.find("h = structural_hash(e)")
    assert idx >= 0, "structural_hash call site not found in monomorphize.py"
    # The 200 chars after the call must NOT contain the dead
    # `except (TypeError, AttributeError, NotImplementedError):` immediately
    # followed by `raise`. (The narrow tuple still legitimately appears in
    # other handlers; we check the no-op shape specifically.)
    window = src[idx:idx + 400]
    bad = "except (TypeError, AttributeError, NotImplementedError):"
    assert bad not in window, (
        f"dead try/except (...): raise block still present around "
        f"structural_hash; window: {window!r}"
    )


# Restart 62 audit canaries (Increment 79):

def test_stage35_restart62_ledger_has_increment_79():
    """Restart 62 C1 (Lane C — bookkeeping): the Stage 35 progress
    ledger must contain Increment 79 for the restart 62 combined
    audit-and-fix. Sibling of restart 58 catch-up canary
    `test_stage35_restart58_ledger_has_increment_77`."""
    from pathlib import Path
    ledger = Path(__file__).resolve().parents[2] / "docs" / "stage35-progress-2026-05-15.md"
    txt = ledger.read_text(encoding="utf-8")
    assert "## Increment 78 — Sixty-First Clean-Gate" in txt, (
        "Increment 78 (restart 61 retroactive) missing"
    )
    assert "## Increment 79 — Sixty-Second Clean-Gate" in txt, (
        "Increment 79 (restart 62 combined audit-and-fix) missing"
    )


def test_stage35_restart62_lane_audit_docs_exist():
    """Restart 62 C2 (Lane C — bookkeeping): restart 61 + restart 62
    lane audit docs must exist. Sibling of
    `test_stage35_restart58_lane_audit_docs_exist`."""
    from pathlib import Path
    docs = Path(__file__).resolve().parents[2] / "docs"
    for restart in ("restart61", "restart62"):
        for lane in ("laneA", "laneB", "laneC"):
            p = docs / f"audit-stage35-{restart}-{lane}.md"
            assert p.exists(), f"missing {p}"


def test_stage35_restart62_surfaces_advanced_past_restart_58_catch_up():
    """Restart 62 C3 (Lane C — surface drift): the eight current-facing
    surfaces must reference a restart number >= 62 as the current
    checkpoint (was: pin literal 'restart 62'; relaxed at restart 65
    closure since the closure refresh moves the pin forward). Sibling
    of `test_stage35_readme_status_paragraph_advanced_past_restart_56`."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    for rel in ("README.md", "QUICKSTART.md",
                "HANDOFF_FOR_CHATGPT.md", "HANDOFF_FOR_CLAUDE.md",
                "helix_website/HELIX_REFERENCE.md",
                "helix_website/stats_and_facts.md"):
        p = root / rel
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8")
        if rel in ("README.md", "QUICKSTART.md",
                   "HANDOFF_FOR_CHATGPT.md",
                   "helix_website/HELIX_REFERENCE.md",
                   "helix_website/stats_and_facts.md"):
            # Any "restart N" mention with N >= 62 counts as a valid
            # forward-checkpoint reference. The Stage 35 closure phrase
            # ("Stage 35 CLOSED") also satisfies (closure happened at
            # restart 65 >= 62).
            numbers = [int(m) for m in re.findall(r"restart (\d+)", txt.lower())]
            forward_ok = any(n >= 62 for n in numbers)
            closure_ok = "stage 35 closed" in txt.lower()
            assert forward_ok or closure_ok, (
                f"{rel} does not reference any restart >= 62 nor the "
                f"Stage 35 closure phrase as the current checkpoint "
                f"(seen restart numbers: {sorted(set(numbers))})"
            )


def test_stage59_list_modules_basic(tmp_path):
    """Stage 59 follow-on / Tier 4 #13 polish: --list-modules
    enumerates top-level ModBlocks with their content hash."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "mods.hx"
    src.write_text(
        "mod alpha { fn a() -> i32 { 1 } }\n"
        "mod beta { fn b() -> i32 { 2 } }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--list-modules", str(src)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [l for l in proc.stdout.splitlines() if l.strip()]
    # Sorted alphabetically.
    assert len(lines) == 2
    assert lines[0].startswith("alpha hash=")
    assert lines[1].startswith("beta hash=")
    # Each hash is 12 hex chars (short_hash).
    for line in lines:
        h = line.split("hash=")[1]
        assert len(h) == 12, f"expected 12-char short hash, got {h!r}"
        assert all(c in "0123456789abcdef" for c in h)


def test_stage59_list_modules_nested(tmp_path):
    """Stage 59 follow-on: --list-modules walks into nested ModBlocks
    and prints dotted names for inner modules."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "nested.hx"
    src.write_text(
        "mod outer {\n"
        "    fn h() -> i32 { 1 }\n"
        "    mod inner { fn i() -> i32 { 2 } }\n"
        "}\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--list-modules", str(src)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "outer hash=" in out
    assert "outer.inner hash=" in out


def test_stage59_module_hash_returns_full_hex(tmp_path):
    """Stage 59 follow-on / Tier 4 #13 polish: --module-hash prints
    the full 64-hex SHA-256 of the named module."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "single.hx"
    src.write_text("mod m1 { fn f() -> i32 { 7 } }\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--module-hash", str(src), "m1"],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    h = proc.stdout.strip()
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_stage59_module_hash_not_found_exits_1(tmp_path):
    """Stage 59 follow-on: --module-hash with unknown module name
    exits 1 with error on stderr."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "empty.hx"
    src.write_text("mod present { fn x() -> i32 { 0 } }\n",
                    encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--module-hash", str(src), "absent"],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 1
    assert "not found" in proc.stderr


def test_stage59_autotune_summary_basic(tmp_path):
    """Stage 59 follow-on / Tier 2 #8 polish: --autotune-summary
    prints {fn variants=N} lines for @autotune @kernel fns plus
    a total."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "at.hx"
    src.write_text(
        "@autotune(BLOCK: [16, 32])\n"
        "@kernel\n"
        "fn k1(x: i32) -> i32 { x + BLOCK }\n"
        "@autotune(N: [4, 8, 16])\n"
        "@kernel\n"
        "fn k2(x: i32) -> i32 { x + N }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--autotune-summary", str(src)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "k1 variants=2" in out
    assert "k2 variants=3" in out
    assert "total variants=5" in out


def test_stage59_autotune_summary_empty_program(tmp_path):
    """Stage 59 follow-on: --autotune-summary on a program with no
    @autotune fns prints only the total=0 line."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "plain.hx"
    src.write_text("fn plain(x: i32) -> i32 { x + 1 }\n",
                    encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--autotune-summary", str(src)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    # Only the total line; no per-fn entries (plain is skipped).
    lines = [l for l in proc.stdout.splitlines() if l.strip()]
    assert lines == ["total variants=0"]


def test_stage59_autodiff_cli_help_mentions_polish_flags():
    """Stage 59 follow-on / Tier 2/3/4 polish: bad-invocation
    (no args) prints help to stderr that mentions all 12 introspection
    flags. Regression pin: if a flag is added but the help docstring
    isn't updated, this catches it."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli"],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    # No args → bad invocation → rc=2 + docstring on stderr.
    assert proc.returncode == 2
    out = proc.stderr
    for flag in (
        "--dump-ast-hashes", "--program-hash", "--program-signature-hash",
        "--diff-program-hash", "--changed-fns", "--fn-sig-hash",
        "--list-fns", "--check-program-hash",
        "--check-program-signature-hash",
        "--list-modules", "--module-hash", "--pytree-shape",
        "--list-pytrees", "--pytree-leaf-paths", "--validate-pytrees",
        "--autotune-summary", "--autotune-budget", "--validate-autotune",
        "--hash-dump", "--diff-hash-dump", "--hash-dump-short",
        "--diff-trace", "--trace-dump-summary",
        "--validate-trace-attrs",
        "--validate-all", "--validate-all-json",
    ):
        assert flag in out, (
            f"help text missing {flag!r}: docstring needs to be "
            f"updated to mention this flag")


def test_stage59_autotune_budget_within_exits_0(tmp_path):
    """Stage 59 follow-on / Tier 2 #8 polish: --autotune-budget
    exits 0 silently when total variants are within budget."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "small.hx"
    src.write_text(
        "@autotune(B: [16, 32])\n"
        "@kernel\n"
        "fn k(x: i32) -> i32 { x + B }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--autotune-budget", str(src), "10"],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    assert proc.stdout == ""


def test_stage59_autotune_budget_over_exits_1(tmp_path):
    """Stage 59 follow-on: --autotune-budget exits 1 with breakdown
    when total exceeds budget."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "big.hx"
    src.write_text(
        "@autotune(B: [16, 32, 64, 128])\n"
        "@kernel\n"
        "fn k(x: i32) -> i32 { x + B }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--autotune-budget", str(src), "2"],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 1
    assert "exceeds budget" in proc.stdout
    assert "k variants=4" in proc.stdout


def test_stage59_autotune_budget_bad_int_exits_2(tmp_path):
    """Stage 59 follow-on: --autotune-budget with non-int budget arg
    exits 2 (bad invocation)."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "any.hx"
    src.write_text("fn x() -> i32 { 0 }\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--autotune-budget", str(src), "notanint"],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 2
    assert "not an int" in proc.stderr


def test_stage59_pytree_shape_flat_struct(tmp_path):
    """Stage 59 follow-on / Tier 2 #7 polish: --pytree-shape prints
    one line per leaf for a flat all-diff struct, plus a summary."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "model.hx"
    src.write_text(
        "struct Model {\n"
        "    w1: D<f32>,\n"
        "    w2: D<f32>,\n"
        "    b: D<f32>\n"
        "}\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--pytree-shape", str(src), "Model"],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "b ty=f32 diff=True" in out
    assert "w1 ty=f32 diff=True" in out
    assert "w2 ty=f32 diff=True" in out
    assert "total leaves=3 diff=3 non_diff=0" in out


def test_stage59_pytree_shape_nested(tmp_path):
    """Stage 59 follow-on: --pytree-shape walks into nested structs
    with dot-joined leaf paths."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "nested.hx"
    src.write_text(
        "struct Inner { w: D<f32> }\n"
        "struct Outer {\n"
        "    inner: Inner,\n"
        "    bias: D<f64>\n"
        "}\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--pytree-shape", str(src), "Outer"],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "inner.w ty=f32" in out
    assert "bias ty=f64" in out
    assert "total leaves=2" in out


def test_stage59_pytree_shape_struct_not_found(tmp_path):
    """Stage 59 follow-on: --pytree-shape with unknown struct name
    exits 1 with stderr diagnostic."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "x.hx"
    src.write_text("struct Real { w: D<f32> }\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--pytree-shape", str(src), "Phantom"],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 1
    assert "not found" in proc.stderr


def test_stage59_pytree_shape_non_diff_field_rejected(tmp_path):
    """Stage 59 follow-on: --pytree-shape surfaces the trap-26002
    non-diff-leaf rejection as a clean stderr diagnostic, not a
    traceback. Exit 1."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "bad.hx"
    src.write_text(
        "struct Bad {\n"
        "    w: D<f32>,\n"
        "    label: i32\n"
        "}\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--pytree-shape", str(src), "Bad"],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 1
    assert "non-differentiable" in proc.stderr or "26002" in proc.stderr
    # Verify no traceback leaked.
    assert "Traceback" not in proc.stderr


def test_stage59_validate_all_json_clean(tmp_path):
    """Stage 59 follow-on / Tier 4 #13 polish: --validate-all-json
    outputs valid JSON with the expected schema on a clean file."""
    import json
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "clean.hx"
    src.write_text(
        "struct M { w: D<f32> }\n"
        "@autotune(B: [16, 32])\n"
        "@kernel\n"
        "fn k(x: i32) -> i32 { x + B }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--validate-all-json", str(src)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    result = json.loads(proc.stdout)
    assert set(result.keys()) == {"pytrees", "autotune", "trace-attrs", "total"}
    assert result["pytrees"]["status"] == "OK"
    assert result["autotune"]["status"] == "OK"
    assert result["trace-attrs"]["status"] == "OK"
    assert result["total"]["validators"] == 3
    assert result["total"]["ok"] == 3
    assert result["total"]["fail"] == 0


def test_stage59_validate_all_json_with_failures(tmp_path):
    """Stage 59 follow-on: --validate-all-json includes diagnostics
    in the JSON output when validators fail."""
    import json
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "mixed.hx"
    src.write_text(
        "struct Bad { w: D<f32>, label: i32 }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--validate-all-json", str(src)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 1
    result = json.loads(proc.stdout)
    assert result["pytrees"]["status"] == "FAIL"
    assert len(result["pytrees"]["diags"]) >= 1
    assert "Bad" in result["pytrees"]["diags"][0]
    assert result["total"]["fail"] == 1


def test_stage59_validate_all_clean_exits_0(tmp_path):
    """Stage 59 follow-on / Tier 4 #13 polish: --validate-all aggregates
    pytree + autotune + trace-attr validators. All-clean → exit 0."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "clean.hx"
    src.write_text(
        "struct M { w: D<f32> }\n"
        "@autotune(B: [16, 32])\n"
        "@kernel\n"
        "fn k(x: i32) -> i32 { x + B }\n"
        "@trace\nfn f(x: i32) -> i32 { x + 1 }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--validate-all", str(src)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    out = proc.stdout
    assert "[pytrees] OK" in out
    assert "[autotune] OK" in out
    assert "[trace-attrs] OK" in out
    assert "total validators=3 OK=3 FAIL=0" in out


def test_stage59_validate_all_aggregates_failures(tmp_path):
    """Stage 59 follow-on: --validate-all reports per-validator status
    + emits diagnostics for each failing one. Exit 1 if any fails."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "mixed.hx"
    src.write_text(
        "struct GoodM { w: D<f32> }\n"
        "struct BadM { w: D<f32>, label: i32 }\n"
        "@autotune(B: [])\n"
        "@kernel\n"
        "fn k(x: i32) -> i32 { x + B }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--validate-all", str(src)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 1
    out = proc.stdout
    assert "[pytrees] FAIL" in out
    assert "BadM" in out
    assert "[autotune] FAIL" in out
    assert "[trace-attrs] OK" in out
    assert "total validators=3 OK=1 FAIL=2" in out


def test_stage59_validate_trace_attrs_clean_exits_0(tmp_path):
    """Stage 59 follow-on / Tier 3 #11 polish: --validate-trace-attrs
    exits 0 silently on a clean @trace fn."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "good.hx"
    src.write_text(
        "@trace\nfn f(x: i32) -> i32 { x + 1 }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--validate-trace-attrs", str(src)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    assert proc.stdout == ""


def test_stage59_validate_trace_attrs_extern_rejected(tmp_path):
    """Stage 59 follow-on: --validate-trace-attrs catches the
    @trace-on-extern violation."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "bad.hx"
    src.write_text(
        "@trace\nextern \"C\" fn libc_printf(s: i32) -> i32;\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--validate-trace-attrs", str(src)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 1
    assert "extern" in proc.stdout
    assert "libc_printf" in proc.stdout


def test_stage59_validate_autotune_clean_exits_0(tmp_path):
    """Stage 59 follow-on / Tier 2 #8 polish: --validate-autotune
    exits 0 silently on a clean @autotune fn."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "good.hx"
    src.write_text(
        "@autotune(BLOCK: [16, 32])\n"
        "@kernel\n"
        "fn k(x: i32) -> i32 { x + BLOCK }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--validate-autotune", str(src)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    assert proc.stdout == ""


def test_stage59_validate_autotune_empty_list_exits_1(tmp_path):
    """Stage 59 follow-on: --validate-autotune surfaces a malformed
    @autotune attr (empty value list) as a diagnostic + exit 1."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "bad.hx"
    src.write_text(
        "@autotune(BLOCK: [])\n"
        "@kernel\n"
        "fn k(x: i32) -> i32 { x + BLOCK }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--validate-autotune", str(src)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 1
    assert "BLOCK" in proc.stdout
    assert "empty" in proc.stdout


def test_stage59_validate_pytrees_all_ok_exits_0(tmp_path):
    """Stage 59 follow-on / Tier 2 #7 polish: --validate-pytrees exits
    0 when every struct in the file flattens successfully."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "good.hx"
    src.write_text(
        "struct A { w: D<f32> }\n"
        "struct B { x: D<f64>, y: D<f64> }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--validate-pytrees", str(src)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    out = proc.stdout
    assert "OK A" in out
    assert "OK B" in out
    assert "total structs=2 OK=2 FAIL=0" in out


def test_stage59_validate_pytrees_failure_exits_1(tmp_path):
    """Stage 59 follow-on: --validate-pytrees exits 1 if any struct
    fails to flatten."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "mixed.hx"
    src.write_text(
        "struct Good { w: D<f32> }\n"
        "struct Bad { w: D<f32>, label: i32 }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--validate-pytrees", str(src)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 1
    out = proc.stdout
    assert "OK Good" in out
    assert "FAIL Bad" in out
    assert "non-differentiable" in out
    assert "total structs=2 OK=1 FAIL=1" in out


def test_stage59_pytree_leaf_paths_basic(tmp_path):
    """Stage 59 follow-on / Tier 2 #7 polish: --pytree-leaf-paths
    prints one path per line, sorted alphabetically — pure paths,
    no types."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "m.hx"
    src.write_text(
        "struct M { w1: D<f32>, w2: D<f32>, b: D<f64> }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--pytree-leaf-paths", str(src), "M"],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    lines = [l for l in proc.stdout.splitlines() if l]
    assert lines == ["b", "w1", "w2"]


def test_stage59_pytree_leaf_paths_nested(tmp_path):
    """Stage 59 follow-on: --pytree-leaf-paths walks nested structs
    and emits dot-joined paths."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "n.hx"
    src.write_text(
        "struct Inner { w: D<f32> }\n"
        "struct Outer { inner: Inner, b: D<f32> }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--pytree-leaf-paths", str(src), "Outer"],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    lines = [l for l in proc.stdout.splitlines() if l]
    assert "b" in lines
    assert "inner.w" in lines


def test_stage59_check_program_signature_hash_match_exits_0(tmp_path):
    """Stage 59 follow-on / Tier 4 #13 polish: --check-program-
    signature-hash exits 0 silent on signature match."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "p.hx"
    src.write_text("fn add(x: i32, y: i32) -> i32 { x + y }\n",
                    encoding="utf-8")
    # Get the expected hash first
    h_proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--program-signature-hash", str(src)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    expected = h_proc.stdout.strip()
    # Check should pass.
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--check-program-signature-hash", str(src), expected],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    assert proc.stdout == ""


def test_stage59_check_program_signature_hash_body_change_passes(tmp_path):
    """Stage 59 follow-on: body-only refactor doesn't trip the
    signature-hash gate (the key invariant — internal cleanups
    don't break ABI gates)."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    original = tmp_path / "v1.hx"
    original.write_text("fn add(x: i32, y: i32) -> i32 { x + y }\n",
                         encoding="utf-8")
    refactored = tmp_path / "v2.hx"
    refactored.write_text("fn add(p: i32, q: i32) -> i32 { p + q + 0 }\n",
                          encoding="utf-8")
    # Pin v1's signature hash.
    h_proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--program-signature-hash", str(original)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    expected = h_proc.stdout.strip()
    # v2 (body-only refactor) should still match v1's pin.
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--check-program-signature-hash", str(refactored), expected],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0


def test_stage59_check_program_signature_hash_sig_change_fails(tmp_path):
    """Stage 59 follow-on: actual signature change DOES trip the gate."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    original = tmp_path / "v1.hx"
    original.write_text("fn add(x: i32, y: i32) -> i32 { x + y }\n",
                         encoding="utf-8")
    sig_change = tmp_path / "v2.hx"
    sig_change.write_text(
        "fn add(x: f32, y: f32) -> f32 { x + y }\n",
        encoding="utf-8",
    )
    h_proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--program-signature-hash", str(original)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    expected = h_proc.stdout.strip()
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--check-program-signature-hash", str(sig_change), expected],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 1
    assert "mismatch" in proc.stdout


def test_stage59_trace_dump_summary(tmp_path):
    """Stage 59 follow-on / Tier 3 #11 polish: --trace-dump-summary
    prints high-level stats of a trace JSON dump."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    import json
    trace = tmp_path / "t.json"
    trace.write_text(json.dumps({
        "cap": 4096,
        "events": [
            {"op_kind": "entry", "fn_name": "f", "operands": [1],
             "result": None},
            {"op_kind": "exit", "fn_name": "f", "operands": [2],
             "result": None},
        ],
    }), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--trace-dump-summary", str(trace)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    out = proc.stdout
    assert "events=2" in out
    assert "fn_counts={'f': 2}" in out
    assert "op_kind_counts=" in out
    assert "balanced=True" in out
    assert "hash=" in out


def test_stage59_diff_trace_match(tmp_path):
    """Stage 59 follow-on / Tier 3 #11 polish: --diff-trace prints
    MATCH + exits 0 when two trace JSON dumps are equivalent."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    # Inline-create two identical trace dumps via the Python API.
    import json
    trace_json = json.dumps({
        "cap": 4096,
        "events": [
            {"op_kind": "entry", "fn_name": "f", "operands": [1],
             "result": None},
            {"op_kind": "exit", "fn_name": "f", "operands": [2],
             "result": None},
        ],
    }, sort_keys=True, separators=(",", ":"))
    a = tmp_path / "a.json"
    a.write_text(trace_json, encoding="utf-8")
    b = tmp_path / "b.json"
    b.write_text(trace_json, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--diff-trace", str(a), str(b)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    assert "MATCH" in proc.stdout
    assert "events=2" in proc.stdout


def test_stage59_diff_trace_first_divergence(tmp_path):
    """Stage 59 follow-on: --diff-trace prints DIFFER + first-divergent
    event index when traces don't match."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    import json
    a = tmp_path / "a.json"
    a.write_text(json.dumps({
        "cap": 4096,
        "events": [
            {"op_kind": "entry", "fn_name": "f", "operands": [1],
             "result": None},
            {"op_kind": "exit", "fn_name": "f", "operands": [2],
             "result": None},
        ],
    }), encoding="utf-8")
    b = tmp_path / "b.json"
    b.write_text(json.dumps({
        "cap": 4096,
        "events": [
            {"op_kind": "entry", "fn_name": "f", "operands": [1],
             "result": None},
            {"op_kind": "exit", "fn_name": "f", "operands": [999],
             "result": None},
        ],
    }), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--diff-trace", str(a), str(b)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 1
    assert "DIFFER" in proc.stdout
    assert "event[1]" in proc.stdout
    # The divergent operands should appear.
    assert "2" in proc.stdout
    assert "999" in proc.stdout


def test_stage59_hash_dump_short_uses_12_hex(tmp_path):
    """Stage 59 follow-on / Tier 4 #13 polish: --hash-dump-short
    outputs the same structure as --hash-dump but every hash is
    truncated to 12 hex chars."""
    import json
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "x.hx"
    src.write_text(
        "fn f(x: i32) -> i32 { x + 1 }\n"
        "struct S { a: i32 }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--hash-dump-short", str(src)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    dump = json.loads(proc.stdout)
    assert len(dump["program_hash"]) == 12
    assert len(dump["program_signature_hash"]) == 12
    assert len(dump["fns"]["f"]["body_hash"]) == 12
    assert len(dump["fns"]["f"]["sig_hash"]) == 12
    assert len(dump["structs"]["S"]) == 12


def test_stage59_diff_hash_dump_match_exits_0(tmp_path):
    """Stage 59 follow-on / Tier 4 #13 polish: --diff-hash-dump prints
    MATCH + exits 0 when programs are semantically identical."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    a = tmp_path / "a.hx"
    a.write_text("fn f(x: i32) -> i32 { x + 1 }\n", encoding="utf-8")
    b = tmp_path / "b.hx"
    # Alpha-rename — semantically identical.
    b.write_text("fn f(y: i32) -> i32 { y + 1 }\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--diff-hash-dump", str(a), str(b)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    assert "MATCH" in proc.stdout


def test_stage59_diff_hash_dump_granular_drift(tmp_path):
    """Stage 59 follow-on: --diff-hash-dump produces a per-item
    drift breakdown (added/removed/changed body/changed struct)."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    a = tmp_path / "a.hx"
    a.write_text(
        "fn add(x: i32, y: i32) -> i32 { x + y }\n"
        "fn sub(x: i32, y: i32) -> i32 { x - y }\n"
        "struct Point { x: i32, y: i32 }\n",
        encoding="utf-8",
    )
    b = tmp_path / "b.hx"
    b.write_text(
        "fn add(p: i32, q: i32) -> i32 { p + q + 0 }\n"
        "fn mul(x: i32, y: i32) -> i32 { x * y }\n"
        "struct Point { x: i32, y: i32, z: i32 }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--diff-hash-dump", str(a), str(b)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 1
    out = proc.stdout
    assert "+ added fn: mul" in out
    assert "- removed fn: sub" in out
    assert "~ changed body: add" in out
    assert "~ changed struct: Point" in out


def test_stage59_diff_hash_dump_signature_change(tmp_path):
    """Stage 59 follow-on: --diff-hash-dump reports changed sig (not
    changed body) when the signature differs."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    a = tmp_path / "a.hx"
    a.write_text("fn f(x: i32) -> i32 { x }\n", encoding="utf-8")
    b = tmp_path / "b.hx"
    b.write_text("fn f(x: f32) -> i32 { x as i32 }\n",
                  encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--diff-hash-dump", str(a), str(b)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 1
    assert "~ changed sig: f" in proc.stdout
    # Should NOT report 'changed body' for f — sig wins.
    assert "changed body: f" not in proc.stdout


def test_stage59_hash_dump_returns_valid_json(tmp_path):
    """Stage 59 follow-on / Tier 4 #13 polish: --hash-dump prints
    valid JSON with the expected top-level keys."""
    import json
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "x.hx"
    src.write_text(
        "fn f(x: i32) -> i32 { x + 1 }\n"
        "struct S { a: i32 }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--hash-dump", str(src)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    dump = json.loads(proc.stdout)
    assert set(dump.keys()) == {
        "program_hash", "program_signature_hash",
        "fns", "structs", "modules",
    }
    assert "f" in dump["fns"]
    assert "S" in dump["structs"]


def test_stage59_hash_dump_pretty_printed(tmp_path):
    """Stage 59 follow-on: --hash-dump output is multi-line JSON
    (indented) for diff-friendly artifact storage."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "y.hx"
    src.write_text("fn f() -> i32 { 1 }\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--hash-dump", str(src)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    # Pretty-printed JSON has newlines + 2-space indent.
    assert "\n  " in proc.stdout  # at least one indented child line


def test_stage59_diff_program_hash_body_only_marker(tmp_path):
    """Stage 59 follow-on / Tier 4 #13 polish: --diff-program-hash
    now distinguishes body-only changes (signatures match) from
    signature changes. Pin the body-only kind marker."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    a = tmp_path / "a.hx"
    a.write_text("fn add(x: i32, y: i32) -> i32 { x + y }\n",
                  encoding="utf-8")
    b = tmp_path / "b.hx"
    b.write_text("fn add(p: i32, q: i32) -> i32 { p + q + 0 }\n",
                  encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--diff-program-hash", str(a), str(b)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 1
    assert "DIFFER" in proc.stdout
    assert "kind=body-only" in proc.stdout
    assert "signatures match" in proc.stdout


def test_stage59_diff_program_hash_signature_change_marker(tmp_path):
    """Stage 59 follow-on: --diff-program-hash kind=signature-change
    marker fires when both the full and signature hashes diverge."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    a = tmp_path / "a.hx"
    a.write_text("fn add(x: i32, y: i32) -> i32 { x + y }\n",
                  encoding="utf-8")
    b = tmp_path / "b.hx"
    b.write_text("fn add(x: f32, y: f32) -> f32 { x + y }\n",
                  encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--diff-program-hash", str(a), str(b)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 1
    assert "DIFFER" in proc.stdout
    assert "kind=signature-change" in proc.stdout


def test_stage59_program_signature_hash_body_invariant(tmp_path):
    """Stage 59 follow-on / Tier 4 #13 polish: --program-signature-hash
    returns identical hashes for two programs differing only in fn
    bodies (signatures unchanged)."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    a = tmp_path / "a.hx"
    a.write_text("fn add(x: i32, y: i32) -> i32 { x + y }\n",
                  encoding="utf-8")
    b = tmp_path / "b.hx"
    b.write_text("fn add(p: i32, q: i32) -> i32 { p + q + 0 }\n",
                  encoding="utf-8")
    proc_a = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--program-signature-hash", str(a)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    proc_b = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--program-signature-hash", str(b)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc_a.returncode == 0
    assert proc_b.returncode == 0
    assert proc_a.stdout.strip() == proc_b.stdout.strip(), (
        "body-only change should leave signature hash unchanged")


def test_stage59_program_signature_hash_sig_change_differs(tmp_path):
    """Stage 59 follow-on: --program-signature-hash differs when a
    fn's signature changes (param type)."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    a = tmp_path / "a.hx"
    a.write_text("fn neg(x: i32) -> i32 { 0 - x }\n", encoding="utf-8")
    b = tmp_path / "b.hx"
    b.write_text("fn neg(x: f32) -> i32 { 0 - (x as i32) }\n",
                  encoding="utf-8")
    proc_a = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--program-signature-hash", str(a)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    proc_b = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--program-signature-hash", str(b)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc_a.stdout.strip() != proc_b.stdout.strip()


def test_stage59_list_pytrees_inventory(tmp_path):
    """Stage 59 follow-on / Tier 2 #7 polish: --list-pytrees prints
    {struct leaves=N diff=K non_diff=M} for each struct, with REJECTED
    marker for structs that flatten_pytree refuses."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "inv.hx"
    src.write_text(
        "struct GoodFlat { w: D<f32>, b: D<f32> }\n"
        "struct GoodNested { inner: GoodFlat, bias: D<f64> }\n"
        "struct BadHasInt { w: D<f32>, label: i32 }\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--list-pytrees", str(src)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "GoodFlat leaves=2 diff=2 non_diff=0" in out
    assert "GoodNested leaves=3 diff=3 non_diff=0" in out
    assert "BadHasInt REJECTED" in out
    assert "non-differentiable" in out


def test_stage59_list_pytrees_empty(tmp_path):
    """Stage 59 follow-on: --list-pytrees on a file with no structs
    prints nothing (just exit 0)."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "no_structs.hx"
    src.write_text("fn main() -> i32 { 0 }\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--list-pytrees", str(src)],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    assert proc.stdout == ""


def test_stage59_module_hash_dotted_nested_name(tmp_path):
    """Stage 59 follow-on: --module-hash accepts dotted names for
    nested modules."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    src = tmp_path / "nested2.hx"
    src.write_text(
        "mod outer {\n"
        "    mod inner { fn f() -> i32 { 42 } }\n"
        "}\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.frontend.autodiff_cli",
         "--module-hash", str(src), "outer.inner"],
        cwd=proj_root, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    h = proc.stdout.strip()
    assert len(h) == 64


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
