"""Tests for the proof artifact cache-key helper script."""

from __future__ import annotations

import hashlib
import io
import json

from helixc.check import PROOF_SCHEMA, proof_cache_key
from scripts import proof_artifact_key


EMPTY_MANIFEST_SHA256 = (
    "e3b0c44298fc1c149afbf4c8996fb924"
    "27ae41e4649b934ca495991b7852b855"
)


def _artifact() -> dict[str, object]:
    input_metadata = {
        "source_sha256": "a" * 64,
        "include_stdlib": False,
        "stdlib_strict": False,
        "stdlib_manifest_sha256": EMPTY_MANIFEST_SHA256,
        "stdlib_files": [],
        "opt_level": 1,
        "flags": ["--emit-proof-obligations", "--no-stdlib"],
        "libs": [],
        "warnings": {},
        "color": "auto",
    }
    return {
        "schema": PROOF_SCHEMA,
        "cache_key": proof_cache_key(input_metadata),
        "path": "ignored.hx",
        "input": input_metadata,
        "summary": {},
        "obligations": [],
        "pipeline_errors": [],
        "typecheck_errors": [],
        "warning_diagnostics": [],
    }


def test_cache_key_for_artifact_matches_compiler_helper():
    artifact = _artifact()
    assert proof_artifact_key.cache_key_for_artifact(artifact) == (
        artifact["cache_key"]
    )
    canonical = json.dumps(
        {"schema": PROOF_SCHEMA, "input": artifact["input"]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert artifact["cache_key"] == hashlib.sha256(canonical).hexdigest()


def test_main_prints_cache_key(capsys, tmp_path):
    artifact = _artifact()
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    rc = proof_artifact_key.main([str(path)])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == artifact["cache_key"]


def test_main_accepts_powershell_utf16_redirected_json(capsys, tmp_path):
    artifact = _artifact()
    path = tmp_path / "artifact-utf16.json"
    path.write_text(json.dumps(artifact), encoding="utf-16")
    rc = proof_artifact_key.main(["--check", str(path)])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == artifact["cache_key"]


def test_main_accepts_utf16_json_on_stdin(monkeypatch, capsys):
    artifact = _artifact()
    raw = json.dumps(artifact).encode("utf-16")
    monkeypatch.setattr(
        proof_artifact_key.sys,
        "stdin",
        io.TextIOWrapper(io.BytesIO(raw), encoding="utf-16"),
    )
    rc = proof_artifact_key.main(["--check", "-"])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == artifact["cache_key"]


def test_main_check_rejects_mismatched_cache_key(capsys, tmp_path):
    artifact = _artifact()
    artifact["cache_key"] = "0" * 64
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    rc = proof_artifact_key.main(["--check", str(path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "cache_key mismatch" in captured.err


def test_cache_key_for_source_unavailable_artifact_is_null(capsys, tmp_path):
    artifact = _artifact()
    input_metadata = artifact["input"]
    assert isinstance(input_metadata, dict)
    input_metadata["source_sha256"] = None
    artifact["cache_key"] = None
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    rc = proof_artifact_key.main(["--check", str(path)])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == "null"
