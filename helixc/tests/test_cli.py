"""Stage 23: tests for the helixc check.py CLI."""

from __future__ import annotations

import os
import json
import hashlib
import runpy
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
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert not out_path.exists()
    assert "error: input:" in proc.stderr
    assert "Traceback" not in proc.stderr


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
    assert proc.returncode == 1, proc.stdout + proc.stderr
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
    assert exc.value.code == 1
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


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
