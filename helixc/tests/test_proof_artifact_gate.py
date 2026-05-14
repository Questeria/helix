"""Tests for the source-to-clean-proof gate helper."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from helixc.check import main as check_main
from scripts import proof_artifact_gate


def _write_clean_source(path: Path) -> None:
    path.write_text(
        "type Probability = f64 where 0.0 <= self <= 1.0;\n"
        "fn main() -> i32 { let p: Probability = 0.5_f64; 0 }\n",
        encoding="utf-8",
    )


def test_gate_accepts_clean_relative_source(capsys, monkeypatch, tmp_path):
    source = tmp_path / "input.hx"
    _write_clean_source(source)
    artifact_path = tmp_path / "artifacts" / "input.proof.json"
    monkeypatch.chdir(tmp_path)

    rc = proof_artifact_gate.main([
        "input.hx",
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    assert captured.out.strip() == "proof-artifact-gate: ok"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["path"] == str(source.resolve())
    assert artifact["summary"]["obligations"] == 1
    assert artifact["obligations"][0]["status"] == "proved"


def test_gate_rejects_unproven_obligation_and_writes_artifact(capsys, tmp_path):
    source = tmp_path / "unproven.hx"
    source.write_text(
        "type Probability = f64 where 0.0 <= self <= 1.0;\n"
        "fn use_raw(x: f64) -> i32 { let p: Probability = x; 0 }\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "unproven.proof.json"

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "is 'unproven', not 'proved'" in captured.err
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["obligations"][0]["status"] == "unproven"
    assert artifact["typecheck_errors"]


def test_gate_returns_bad_invocation_for_missing_source(capsys, tmp_path):
    source = tmp_path / "missing.hx"
    artifact_path = tmp_path / "missing.proof.json"

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert "input.source_sha256 is required for --require-clean" in captured.err
    assert "compiler exited 2" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["input"]["source_sha256"] is None
    assert artifact["cache_key"] is None


def test_gate_rejects_artifact_out_equal_to_source(capsys, tmp_path):
    source = tmp_path / "input.hx"
    _write_clean_source(source)
    original = source.read_text(encoding="utf-8")

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(source),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert "must not point to the source file" in captured.err
    assert source.read_text(encoding="utf-8") == original


def test_gate_returns_bad_invocation_for_artifact_write_error(capsys, tmp_path):
    source = tmp_path / "input.hx"
    _write_clean_source(source)
    artifact_dir = tmp_path / "artifact-dir"
    artifact_dir.mkdir()

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_dir),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert "could not write artifact" in captured.err
    assert "Traceback" not in captured.err


def test_gate_rejects_disallowed_output_args(capsys, tmp_path):
    source = tmp_path / "input.hx"
    _write_clean_source(source)
    output = tmp_path / "ignored.bin"

    rc = proof_artifact_gate.main([
        str(source),
        "--",
        "--no-stdlib",
        "-o",
        str(output),
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert "-o is not allowed" in captured.err
    assert not output.exists()


def test_gate_rejects_disallowed_debug_output_arg(capsys, tmp_path):
    source = tmp_path / "input.hx"
    _write_clean_source(source)

    rc = proof_artifact_gate.main([
        str(source),
        "--",
        "--no-stdlib",
        "--emit-ir",
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert "--emit-ir" in captured.err


def test_gate_rejects_invalid_json_stdout(capsys, monkeypatch, tmp_path):
    source = tmp_path / "input.hx"
    _write_clean_source(source)

    monkeypatch.setattr(
        proof_artifact_gate.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="not json",
            stderr="",
        ),
    )

    rc = proof_artifact_gate.main([str(source), "--", "--no-stdlib"])

    captured = capsys.readouterr()
    assert rc == 2
    assert "invalid proof artifact JSON" in captured.err


def test_gate_rejects_structurally_clean_but_partial_artifact(
    capsys, monkeypatch, tmp_path,
):
    source = tmp_path / "input.hx"
    _write_clean_source(source)
    rc = check_main([
        str(source),
        "--emit-proof-obligations",
        "--no-stdlib",
    ])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    artifact = json.loads(captured.out)
    artifact["obligations"] = []
    artifact["summary"]["obligations"] = 0
    raw = json.dumps(artifact)

    monkeypatch.setattr(
        proof_artifact_gate.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=raw,
            stderr="",
        ),
    )

    rc = proof_artifact_gate.main([str(source), "--", "--no-stdlib"])

    captured = capsys.readouterr()
    assert rc == 2
    assert "proof artifact obligations mismatch" in captured.err


def test_gate_rejects_structurally_clean_but_missing_typecheck_errors(
    capsys, monkeypatch, tmp_path,
):
    source = tmp_path / "bad.hx"
    source.write_text("fn main() -> i32 { true }\n", encoding="utf-8")
    rc = check_main([
        str(source),
        "--emit-proof-obligations",
        "--no-stdlib",
    ])
    captured = capsys.readouterr()
    assert rc == 1, captured.out + captured.err
    artifact = json.loads(captured.out)
    assert artifact["typecheck_errors"]
    artifact["typecheck_errors"] = []
    artifact["summary"]["typecheck_errors"] = 0
    raw = json.dumps(artifact)

    monkeypatch.setattr(
        proof_artifact_gate.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=raw,
            stderr="",
        ),
    )

    rc = proof_artifact_gate.main([str(source), "--", "--no-stdlib"])

    captured = capsys.readouterr()
    assert rc == 2
    assert "typecheck_errors mismatch" in captured.err


def test_gate_rejects_artifact_path_mismatch(capsys, monkeypatch, tmp_path):
    source = tmp_path / "input.hx"
    _write_clean_source(source)
    rc = check_main([
        str(source),
        "--emit-proof-obligations",
        "--no-stdlib",
    ])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    artifact = json.loads(captured.out)
    artifact["path"] = str(tmp_path / "other.hx")
    raw = json.dumps(artifact)

    monkeypatch.setattr(
        proof_artifact_gate.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=raw,
            stderr="",
        ),
    )

    rc = proof_artifact_gate.main([str(source), "--", "--no-stdlib"])

    captured = capsys.readouterr()
    assert rc == 2
    assert "proof artifact path mismatch" in captured.err


def test_gate_missing_requested_source_cannot_use_artifact_path(
    capsys, monkeypatch, tmp_path,
):
    real_source = tmp_path / "real.hx"
    _write_clean_source(real_source)
    rc = check_main([
        str(real_source),
        "--emit-proof-obligations",
        "--no-stdlib",
    ])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    raw = captured.out

    missing_source = tmp_path / "missing.hx"
    monkeypatch.setattr(
        proof_artifact_gate.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=raw,
            stderr="",
        ),
    )

    rc = proof_artifact_gate.main([str(missing_source), "--", "--no-stdlib"])

    captured = capsys.readouterr()
    assert rc == 2
    assert "requested source path is not a readable file" in captured.err
