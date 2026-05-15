"""Tests for the proof artifact validation helper."""

from __future__ import annotations

import hashlib
import json

from helixc.check import main as check_main, proof_cache_key
from scripts import proof_artifact_validate


EMPTY_MANIFEST_SHA256 = (
    "4f53cda18c2baa0c0354bb5f9a3ecbe5"
    "ed12ab4d8e11ba873c2f11161202b945"
)


def _manifest_sha256(files: list[dict[str, object]]) -> str:
    raw = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _real_artifact(
    capsys,
    tmp_path,
    src: str | None = None,
    *,
    expected_rc: int = 0,
):
    source = (
        src
        if src is not None
        else (
            "type Probability = f64 where 0.0 <= self <= 1.0;\n"
            "fn main() -> i32 { let p: Probability = 0.5_f64; 0 }\n"
        )
    )
    source_path = tmp_path / "input.hx"
    source_path.write_text(source, encoding="utf-8")
    rc = check_main([
        str(source_path), "--emit-proof-obligations", "--no-stdlib",
    ])
    captured = capsys.readouterr()
    assert rc == expected_rc, captured.out + captured.err
    artifact = json.loads(captured.out)
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    return source_path, artifact_path, artifact


def test_validate_real_artifact_with_source_passes(capsys, tmp_path):
    source_path, artifact_path, _artifact = _real_artifact(capsys, tmp_path)
    rc = proof_artifact_validate.main([
        str(artifact_path), "--source", str(source_path),
    ])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == "proof-artifact-validate: ok"


def test_require_clean_accepts_proved_artifact(capsys, tmp_path):
    source_path, artifact_path, _artifact = _real_artifact(capsys, tmp_path)
    rc = proof_artifact_validate.main([
        str(artifact_path), "--source", str(source_path), "--require-clean",
    ])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == "proof-artifact-validate: ok"


def test_require_clean_rejects_unproven_obligation(capsys, tmp_path):
    source_path, artifact_path, artifact = _real_artifact(
        capsys,
        tmp_path,
        src=(
            "type Probability = f64 where 0.0 <= self <= 1.0;\n"
            "fn use_raw(x: f64) -> i32 { let p: Probability = x; 0 }\n"
        ),
        expected_rc=1,
    )
    assert artifact["obligations"][0]["status"] == "unproven"
    assert artifact["typecheck_errors"]
    rc = proof_artifact_validate.main([
        str(artifact_path), "--source", str(source_path), "--require-clean",
    ])
    captured = capsys.readouterr()
    assert rc == 1
    assert "is 'unproven', not 'proved'" in captured.err
    assert "typecheck_errors must be empty" in captured.err


def test_require_clean_rejects_forged_clean_artifact_with_source(
    capsys, tmp_path,
):
    source_path, artifact_path, artifact = _real_artifact(
        capsys,
        tmp_path,
        src=(
            "type Probability = f64 where 0.0 <= self <= 1.0;\n"
            "fn use_raw(x: f64) -> i32 { let p: Probability = x; 0 }\n"
            "fn main() -> i32 { 0 }\n"
        ),
        expected_rc=1,
    )
    artifact["obligations"][0]["status"] = "proved"
    artifact["typecheck_errors"] = []
    artifact["summary"]["typecheck_errors"] = 0
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    rc = proof_artifact_validate.main([
        str(artifact_path), "--source", str(source_path), "--require-clean",
    ])
    captured = capsys.readouterr()
    assert rc == 1
    assert "proof artifact summary mismatch against recomputed source" in (
        captured.err
    )
    assert "proof artifact obligations mismatch against recomputed source" in (
        captured.err
    )
    assert "proof artifact typecheck_errors mismatch against recomputed source" in (
        captured.err
    )
    assert "recomputed proof run exited 1" in captured.err


def test_validate_accepts_unsupported_obligation_structurally(capsys, tmp_path):
    source_path, artifact_path, artifact = _real_artifact(
        capsys,
        tmp_path,
        src=(
            "type Weird = f64 where self + 1.0;\n"
            "fn main() -> i32 { let w: Weird = 0.5_f64; 0 }\n"
        ),
        expected_rc=1,
    )
    assert artifact["obligations"][0]["status"] == "unsupported"
    rc = proof_artifact_validate.main([
        str(artifact_path), "--source", str(source_path),
    ])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert captured.out.strip() == "proof-artifact-validate: ok"


def test_require_clean_rejects_unsupported_obligation(capsys, tmp_path):
    source_path, artifact_path, artifact = _real_artifact(
        capsys,
        tmp_path,
        src=(
            "type Weird = f64 where self + 1.0;\n"
            "fn main() -> i32 { let w: Weird = 0.5_f64; 0 }\n"
        ),
        expected_rc=1,
    )
    assert artifact["obligations"][0]["status"] == "unsupported"
    rc = proof_artifact_validate.main([
        str(artifact_path), "--source", str(source_path), "--require-clean",
    ])
    captured = capsys.readouterr()
    assert rc == 1
    assert "is 'unsupported', not 'proved'" in captured.err
    assert "typecheck_errors must be empty" in captured.err


def test_require_clean_rejects_pipeline_errors(capsys, tmp_path):
    _source_path, artifact_path, artifact = _real_artifact(capsys, tmp_path)
    artifact["pipeline_errors"] = [{"phase": "demo", "message": "not clean"}]
    artifact["summary"]["pipeline_errors"] = 1
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    rc = proof_artifact_validate.main([str(artifact_path), "--require-clean"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "pipeline_errors must be empty" in captured.err


def test_require_clean_rejects_promoted_warning(capsys, tmp_path):
    _source_path, artifact_path, artifact = _real_artifact(capsys, tmp_path)
    artifact["warning_diagnostics"] = [{
        "kind": "demo",
        "policy": "error",
        "message": "not clean",
        "promoted_to_error": True,
    }]
    artifact["summary"]["warning_diagnostics"] = 1
    artifact["summary"]["warning_errors"] = 1
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    rc = proof_artifact_validate.main([str(artifact_path), "--require-clean"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "promoted errors" in captured.err


def test_validate_rejects_source_hash_mismatch(capsys, tmp_path):
    source_path, artifact_path, _artifact = _real_artifact(capsys, tmp_path)
    source_path.write_text(
        "type Probability = f64 where 0.0 <= self <= 1.0;\n"
        "fn main() -> i32 { let p: Probability = 0.6_f64; 0 }\n",
        encoding="utf-8",
    )
    rc = proof_artifact_validate.main([
        str(artifact_path), "--source", str(source_path),
    ])
    captured = capsys.readouterr()
    assert rc == 1
    assert "source sha256 mismatch" in captured.err


def test_require_clean_rejects_forged_artifact_path_with_source(
    capsys, tmp_path,
):
    source_path, artifact_path, artifact = _real_artifact(capsys, tmp_path)
    artifact["path"] = str(tmp_path / "not-the-source.hx")
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    rc = proof_artifact_validate.main([
        str(artifact_path), "--source", str(source_path), "--require-clean",
    ])
    captured = capsys.readouterr()
    assert rc == 1
    assert "proof artifact path mismatch against provided source" in (
        captured.err
    )


def test_validate_rejects_forged_artifact_path_with_source_by_default(
    capsys, tmp_path,
):
    source_path, artifact_path, artifact = _real_artifact(capsys, tmp_path)
    artifact["path"] = str(tmp_path / "not-the-source.hx")
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    rc = proof_artifact_validate.main([
        str(artifact_path), "--source", str(source_path),
    ])
    captured = capsys.readouterr()
    assert rc == 1
    assert "proof artifact path mismatch against provided source" in (
        captured.err
    )


def test_validate_rejects_stale_artifact_path_by_default(capsys, tmp_path):
    source_path, artifact_path, _artifact = _real_artifact(capsys, tmp_path)
    source_path.write_text(
        "type Probability = f64 where 0.0 <= self <= 1.0;\n"
        "fn main() -> i32 { let p: Probability = 0.6_f64; 0 }\n",
        encoding="utf-8",
    )
    rc = proof_artifact_validate.main([str(artifact_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "source sha256 mismatch" in captured.err


def test_validate_resolves_relative_artifact_path_from_artifact_dir(
    capsys, monkeypatch, tmp_path,
):
    source_path = tmp_path / "input.hx"
    source_path.write_text(
        "type Probability = f64 where 0.0 <= self <= 1.0;\n"
        "fn main() -> i32 { let p: Probability = 0.5_f64; 0 }\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    rc = check_main(["input.hx", "--emit-proof-obligations", "--no-stdlib"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    artifact = json.loads(captured.out)
    assert artifact["path"] == "input.hx"

    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    source_path.write_text(
        "type Probability = f64 where 0.0 <= self <= 1.0;\n"
        "fn main() -> i32 { let p: Probability = 0.6_f64; 0 }\n",
        encoding="utf-8",
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)
    rc = proof_artifact_validate.main([str(artifact_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "source sha256 mismatch" in captured.err


def test_validate_rejects_missing_embedded_source_path(capsys, tmp_path):
    _source_path, artifact_path, artifact = _real_artifact(capsys, tmp_path)
    artifact["path"] = "missing-source.hx"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    rc = proof_artifact_validate.main([str(artifact_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "source path from artifact could not be resolved" in captured.err


def test_validate_rejects_cache_key_mismatch(capsys, tmp_path):
    _source_path, artifact_path, artifact = _real_artifact(capsys, tmp_path)
    artifact["cache_key"] = "0" * 64
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    rc = proof_artifact_validate.main([str(artifact_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "cache_key mismatch" in captured.err


def test_validate_rejects_summary_count_mismatch(capsys, tmp_path):
    _source_path, artifact_path, artifact = _real_artifact(capsys, tmp_path)
    artifact["summary"]["obligations"] = 99
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    rc = proof_artifact_validate.main([str(artifact_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "summary.obligations" in captured.err


def test_validate_checks_stage34_proof_carry_records(capsys, tmp_path):
    _source_path, artifact_path, artifact = _real_artifact(
        capsys,
        tmp_path,
        src=(
            "type AtLeastOne = f64 where self >= 1.0;\n"
            "type NonNegative = f64 where self >= 0.0;\n"
            "fn lift(a: AtLeastOne) -> NonNegative { a }\n"
        ),
    )
    assert artifact["summary"]["proof_carries"] == 1
    assert artifact["proof_carries"][0]["strategy"] == "numeric-bound-implication"
    assert artifact["summary"]["proof_carry_strategies"] == {
        "numeric-bound-implication": 1,
    }

    artifact["summary"]["proof_carries"] = 99
    artifact["summary"]["proof_carry_strategies"] = {
        "numeric-bound-implication": 2,
    }
    artifact["proof_carries"][0]["strategy"] = "made-up-proof"
    artifact["proof_carries"][0]["span"]["line"] = False
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    rc = proof_artifact_validate.main([str(artifact_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "summary.proof_carries" in captured.err
    assert "summary.proof_carry_strategies does not match" in captured.err
    assert "proof_carries[0].strategy" in captured.err
    assert "proof_carries[0].span.line must be an integer" in captured.err


def test_validate_rejects_boolean_integer_fields(capsys, tmp_path):
    _source_path, artifact_path, artifact = _real_artifact(capsys, tmp_path)
    artifact["summary"]["obligations"] = True
    artifact["summary"]["warning_errors"] = False
    artifact["input"]["opt_level"] = False
    artifact["obligations"][0]["span"]["line"] = False
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    rc = proof_artifact_validate.main([str(artifact_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "summary.obligations must be an integer" in captured.err
    assert "summary.warning_errors must be an integer" in captured.err
    assert "input.opt_level must be an integer" in captured.err
    assert "span.line must be an integer" in captured.err


def test_validate_rejects_malformed_diagnostic_sections(capsys, tmp_path):
    _source_path, artifact_path, artifact = _real_artifact(capsys, tmp_path)
    artifact["path"] = 123
    artifact["pipeline_errors"] = ["not an object"]
    artifact["typecheck_errors"] = [{"not": "a string"}]
    artifact["warning_diagnostics"] = ["not an object"]
    artifact["summary"]["pipeline_errors"] = 1
    artifact["summary"]["typecheck_errors"] = 1
    artifact["summary"]["warning_diagnostics"] = 1
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    rc = proof_artifact_validate.main([str(artifact_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "path must be a string or null" in captured.err
    assert "pipeline_errors[0] must be an object" in captured.err
    assert "typecheck_errors[0] must be a string" in captured.err
    assert "warning_diagnostics[0] must be an object" in captured.err


def test_validate_rejects_incomplete_input_metadata(capsys, tmp_path):
    _source_path, artifact_path, artifact = _real_artifact(capsys, tmp_path)
    source_sha = artifact["input"]["source_sha256"]
    minimal_input = {"source_sha256": source_sha}
    artifact["input"] = minimal_input
    artifact["cache_key"] = proof_cache_key(minimal_input)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    rc = proof_artifact_validate.main([str(artifact_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "input.include_stdlib is required" in captured.err
    assert "input.stdlib_manifest_sha256 is required" in captured.err
    assert "input.flags is required" in captured.err


def test_validate_rejects_bad_stdlib_manifest_hash(capsys, tmp_path):
    _source_path, artifact_path, artifact = _real_artifact(capsys, tmp_path)
    input_metadata = artifact["input"]
    assert isinstance(input_metadata, dict)
    input_metadata["include_stdlib"] = True
    input_metadata["stdlib_files"] = [
        {"path": "std.hx", "sha256": "a" * 64, "bytes": 10},
    ]
    input_metadata["stdlib_manifest_sha256"] = "0" * 64
    artifact["cache_key"] = proof_cache_key(input_metadata)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    rc = proof_artifact_validate.main([str(artifact_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "stdlib_manifest_sha256 does not match" in captured.err


def test_validate_rejects_malformed_missing_stdlib_entry(capsys, tmp_path):
    _source_path, artifact_path, artifact = _real_artifact(capsys, tmp_path)
    input_metadata = artifact["input"]
    assert isinstance(input_metadata, dict)
    input_metadata["include_stdlib"] = True
    input_metadata["stdlib_files"] = [
        {"path": "missing.hx", "missing": True, "sha256": "bad", "bytes": True},
    ]
    input_metadata["stdlib_manifest_sha256"] = _manifest_sha256(
        input_metadata["stdlib_files"]
    )
    artifact["cache_key"] = proof_cache_key(input_metadata)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    rc = proof_artifact_validate.main([str(artifact_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "sha256 must be absent when missing is true" in captured.err
    assert "bytes must be absent when missing is true" in captured.err


def test_validate_rejects_no_stdlib_artifact_with_stdlib_files(
    capsys, tmp_path,
):
    _source_path, artifact_path, artifact = _real_artifact(capsys, tmp_path)
    input_metadata = artifact["input"]
    assert isinstance(input_metadata, dict)
    input_metadata["include_stdlib"] = False
    input_metadata["stdlib_files"] = [
        {"path": "std.hx", "sha256": "a" * 64, "bytes": 10},
    ]
    input_metadata["stdlib_manifest_sha256"] = _manifest_sha256(
        input_metadata["stdlib_files"]
    )
    artifact["cache_key"] = proof_cache_key(input_metadata)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    rc = proof_artifact_validate.main([str(artifact_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "stdlib_files must be empty when include_stdlib is false" in captured.err


def test_validate_rejects_malformed_obligation(capsys, tmp_path):
    _source_path, artifact_path, artifact = _real_artifact(capsys, tmp_path)
    artifact["obligations"][0]["status"] = "maybe"
    artifact["obligations"][0]["span"].pop("line")
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    rc = proof_artifact_validate.main([str(artifact_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "status must be one of" in captured.err
    assert "span.line" in captured.err


def test_validate_source_unavailable_artifact_accepts_null_cache(capsys, tmp_path):
    artifact = {
        "schema": "helix.proof_obligations.v0",
        "cache_key": None,
        "path": None,
        "input": {
            "source_sha256": None,
            "source_available": False,
            "source_error": "source path is missing",
            "include_stdlib": False,
            "stdlib_strict": False,
            "stdlib_manifest_sha256": EMPTY_MANIFEST_SHA256,
            "stdlib_files": [],
            "opt_level": 1,
            "flags": ["--emit-proof-obligations", "--no-stdlib"],
            "libs": [],
            "warnings": {},
            "color": "auto",
        },
        "summary": {
            "obligations": 0,
            "pipeline_errors": 0,
            "typecheck_errors": 0,
            "warning_diagnostics": 0,
            "warning_errors": 0,
        },
        "obligations": [],
        "pipeline_errors": [],
        "typecheck_errors": [],
        "warning_diagnostics": [],
    }
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    rc = proof_artifact_validate.main([str(artifact_path)])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == "proof-artifact-validate: ok"


def test_validate_source_unavailable_requires_cache_key_field(capsys, tmp_path):
    artifact = {
        "schema": "helix.proof_obligations.v0",
        "path": None,
        "input": {
            "source_sha256": None,
            "source_available": False,
            "source_error": "source path is missing",
            "include_stdlib": False,
            "stdlib_strict": False,
            "stdlib_manifest_sha256": EMPTY_MANIFEST_SHA256,
            "stdlib_files": [],
            "opt_level": 1,
            "flags": ["--emit-proof-obligations", "--no-stdlib"],
            "libs": [],
            "warnings": {},
            "color": "auto",
        },
        "summary": {
            "obligations": 0,
            "pipeline_errors": 0,
            "typecheck_errors": 0,
            "warning_diagnostics": 0,
            "warning_errors": 0,
        },
        "obligations": [],
        "pipeline_errors": [],
        "typecheck_errors": [],
        "warning_diagnostics": [],
    }
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    rc = proof_artifact_validate.main([str(artifact_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "cache_key field is required" in captured.err


def test_validate_source_unavailable_rejects_proof_content(capsys, tmp_path):
    artifact = {
        "schema": "helix.proof_obligations.v0",
        "cache_key": None,
        "path": None,
        "input": {
            "source_sha256": None,
            "source_available": False,
            "source_error": "source path is missing",
            "include_stdlib": False,
            "stdlib_strict": False,
            "stdlib_manifest_sha256": EMPTY_MANIFEST_SHA256,
            "stdlib_files": [],
            "opt_level": 1,
            "flags": ["--emit-proof-obligations", "--no-stdlib"],
            "libs": [],
            "warnings": {},
            "color": "auto",
        },
        "summary": {
            "obligations": 1,
            "proof_carries": 1,
            "pipeline_errors": 0,
            "typecheck_errors": 1,
            "warning_diagnostics": 0,
            "warning_errors": 0,
            "proof_carry_strategies": {"same-refinement": 1},
        },
        "obligations": [{
            "kind": "refinement",
            "context": "let 'p'",
            "refinement": "Probability",
            "predicate": "0.0 <= self <= 1.0",
            "status": "proved",
            "span": {"line": 1, "col": 1},
        }],
        "proof_carries": [{
            "kind": "refinement-proof-carry",
            "context": "let 'p'",
            "source_refinement": "Probability",
            "target_refinement": "Probability",
            "strategy": "same-refinement",
            "span": {"line": 1, "col": 1},
        }],
        "pipeline_errors": [],
        "typecheck_errors": ["fake typecheck error"],
        "warning_diagnostics": [],
    }
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    rc = proof_artifact_validate.main([str(artifact_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "obligations must be empty" in captured.err
    assert "proof_carries must be empty" in captured.err
    assert "typecheck_errors must be empty" in captured.err


def test_require_clean_rejects_source_unavailable_artifact(capsys, tmp_path):
    artifact = {
        "schema": "helix.proof_obligations.v0",
        "cache_key": None,
        "path": None,
        "input": {
            "source_sha256": None,
            "source_available": False,
            "source_error": "source path is missing",
            "include_stdlib": False,
            "stdlib_strict": False,
            "stdlib_manifest_sha256": EMPTY_MANIFEST_SHA256,
            "stdlib_files": [],
            "opt_level": 1,
            "flags": ["--emit-proof-obligations", "--no-stdlib"],
            "libs": [],
            "warnings": {},
            "color": "auto",
        },
        "summary": {
            "obligations": 0,
            "pipeline_errors": 0,
            "typecheck_errors": 0,
            "warning_diagnostics": 0,
            "warning_errors": 0,
        },
        "obligations": [],
        "pipeline_errors": [],
        "typecheck_errors": [],
        "warning_diagnostics": [],
    }
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    rc = proof_artifact_validate.main([str(artifact_path), "--require-clean"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "input.source_sha256 is required for --require-clean" in captured.err
