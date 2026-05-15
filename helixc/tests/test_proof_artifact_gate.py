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


def test_gate_rejects_boolean_to_numeric_refined_cast(capsys, tmp_path):
    source = tmp_path / "bool_cast_refined.hx"
    source.write_text(
        "type Probability = f64 where 0.0 <= self <= 1.0;\n"
        "fn f() -> Probability { true as Probability }\n"
        "fn main() -> i32 { 0 }\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "bool_cast_refined.proof.json"

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert not any(
        obligation.get("status") == "proved"
        and obligation.get("value") == "true"
        for obligation in artifact["obligations"]
    )


def test_gate_rejects_impossible_refined_integer_alias(capsys, tmp_path):
    source = tmp_path / "bad_refined_u8.hx"
    source.write_text(
        "type Exactly300 = u8 where self == 300;\n"
        "fn f() -> Exactly300 {\n"
        "    return 300_u8;\n"
        "    300_u8\n"
        "}\n"
        "fn use_arr(xs: [Exactly300; 1]) -> i32 { 0 }\n"
        "fn main() -> i32 {\n"
        "    let xs: [Exactly300; 1] = [300_u8];\n"
        "    use_arr(xs)\n"
        "}\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "bad_refined_u8.proof.json"

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert not any(
        obligation.get("status") == "proved"
        and obligation.get("value") == "300"
        for obligation in artifact["obligations"]
    )


def test_gate_rejects_unrepresentable_suffixed_int_source_cast_to_refined(
    capsys, tmp_path,
):
    source = tmp_path / "bad_suffixed_int_refined_cast.hx"
    source.write_text(
        "type PositiveI64 = i64 where self > 0;\n"
        "fn f() -> PositiveI64 { 2147483648_i32 as PositiveI64 }\n"
        "fn main() -> i32 { 0 }\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "bad_suffixed_int_refined_cast.proof.json"

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert any(
        "cast to refined type PositiveI64" in error
        and "value is not representable" in error
        for error in artifact["typecheck_errors"]
    )


def test_gate_rejects_overflowing_refined_f32_literal(capsys, tmp_path):
    source = tmp_path / "overflow_refined_f32.hx"
    source.write_text(
        "type Huge = f32 where self > 3.5e38;\n"
        "fn f() -> Huge { 1e40_f32 }\n"
        "fn main() -> i32 { let h: Huge = f(); 0 }\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "overflow_refined_f32.proof.json"

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert not any(
        obligation.get("status") == "proved"
        and obligation.get("value") == "inf"
        for obligation in artifact["obligations"]
    )


def test_gate_rejects_nonfinite_refined_f32_literal(capsys, tmp_path):
    source = tmp_path / "nonfinite_refined_f32.hx"
    source.write_text(
        "type Huge = f32 where self > 3.5e38;\n"
        "fn f() -> Huge { 1e309_f32 }\n"
        "fn main() -> i32 { let h: Huge = f(); 0 }\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "nonfinite_refined_f32.proof.json"

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert not any(
        obligation.get("status") == "proved"
        and obligation.get("value") == "inf"
        for obligation in artifact["obligations"]
    )


def test_gate_rejects_nonfinite_refined_f64_literal(capsys, tmp_path):
    source = tmp_path / "nonfinite_refined_f64.hx"
    source.write_text(
        "type Huge = f64 where self > 1.0e308;\n"
        "fn f() -> Huge { 1e309_f64 }\n"
        "fn main() -> i32 { let h: Huge = f(); 0 }\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "nonfinite_refined_f64.proof.json"

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert not any(
        obligation.get("status") == "proved"
        and obligation.get("value") == "inf"
        for obligation in artifact["obligations"]
    )


def test_gate_rejects_self_independent_unrepresentable_value(capsys, tmp_path):
    source = tmp_path / "self_independent_unrepresentable_value.hx"
    source.write_text(
        "type AlwaysF64 = f64 where true;\n"
        "fn f() -> AlwaysF64 { 1e309_f64 }\n"
        "fn main() -> i32 { let h: AlwaysF64 = f(); 0 }\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "self_independent_unrepresentable_value.proof.json"

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert artifact["summary"]["proof_carries"] == 0


def test_gate_rejects_rounded_f32_predicate_literal(capsys, tmp_path):
    source = tmp_path / "rounded_f32_predicate_literal.hx"
    source.write_text(
        "type BelowRounded = f32 where self < 16777217.0_f32;\n"
        "fn f() -> BelowRounded { 16777216.0_f32 }\n"
        "fn main() -> i32 { let x: BelowRounded = f(); 0 }\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "rounded_f32_predicate_literal.proof.json"

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert not any(
        obligation.get("status") == "proved"
        and obligation.get("refinement") == "BelowRounded"
        for obligation in artifact["obligations"]
    )


def test_gate_rejects_unsuffixed_rounded_f32_predicate_literal(
    capsys, tmp_path,
):
    source = tmp_path / "unsuffixed_rounded_f32_predicate_literal.hx"
    source.write_text(
        "type BelowRounded = f32 where self < 16777217.0;\n"
        "fn f() -> BelowRounded { 16777216.0_f32 }\n"
        "fn main() -> i32 { let x: BelowRounded = f(); 0 }\n",
        encoding="utf-8",
    )
    artifact_path = (
        tmp_path / "unsuffixed_rounded_f32_predicate_literal.proof.json")

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert not any(
        obligation.get("status") == "proved"
        and obligation.get("refinement") == "BelowRounded"
        for obligation in artifact["obligations"]
    )


def test_gate_rejects_false_carry_from_rounded_f32_bounds(capsys, tmp_path):
    source = tmp_path / "rounded_f32_bound_false_carry.hx"
    source.write_text(
        "type Source = f32 where self >= 16777217.0_f32;\n"
        "type Target = f32 where self > 16777216.0_f32;\n"
        "fn make() -> Source { 16777216.0_f32 }\n"
        "fn bad(s: Source) -> Target { s }\n"
        "fn main() -> i32 { let t: Target = bad(make()); 0 }\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "rounded_f32_bound_false_carry.proof.json"

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert not any(
        carry.get("strategy") == "numeric-bound-implication"
        and carry.get("source_refinement") == "Source"
        and carry.get("target_refinement") == "Target"
        for carry in artifact["proof_carries"]
    )


def test_gate_rejects_false_carry_from_unsuffixed_f32_bounds(
    capsys, tmp_path,
):
    source = tmp_path / "unsuffixed_f32_bound_false_carry.hx"
    source.write_text(
        "type Source = f32 where self >= 16777217.0;\n"
        "type Target = f32 where self > 16777216.0;\n"
        "fn bad(s: Source) -> Target { s }\n"
        "fn main() -> i32 { 0 }\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "unsuffixed_f32_bound_false_carry.proof.json"

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert not any(
        carry.get("strategy") == "numeric-bound-implication"
        and carry.get("source_refinement") == "Source"
        and carry.get("target_refinement") == "Target"
        for carry in artifact["proof_carries"]
    )


def test_gate_rejects_raw_const_predicate_bound(capsys, tmp_path):
    source = tmp_path / "raw_const_predicate_bound.hx"
    source.write_text(
        "const LIMIT: f32 = 16777217.0_f32;\n"
        "type BelowLimit = f32 where self < LIMIT;\n"
        "fn f() -> BelowLimit { 16777216.0_f32 }\n"
        "fn main() -> i32 { let x: BelowLimit = f(); 0 }\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "raw_const_predicate_bound.proof.json"

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert not any(
        obligation.get("status") == "proved"
        and obligation.get("refinement") == "BelowLimit"
        for obligation in artifact["obligations"]
    )


def test_gate_rejects_nonfinite_predicate_arithmetic(capsys, tmp_path):
    source = tmp_path / "nonfinite_predicate_arithmetic.hx"
    source.write_text(
        "type Below = f64 where self < (1e308_f64 * 10.0_f64);\n"
        "fn f() -> Below { 0.0_f64 }\n"
        "fn main() -> i32 { let x: Below = f(); 0 }\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "nonfinite_predicate_arithmetic.proof.json"

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert not any(
        obligation.get("status") == "proved"
        and obligation.get("refinement") == "Below"
        for obligation in artifact["obligations"]
    )


def test_gate_rejects_f32_predicate_arithmetic_false_pass(capsys, tmp_path):
    source = tmp_path / "f32_predicate_arithmetic_false_pass.hx"
    source.write_text(
        "type Above = f32 where self + 1.0_f32 > 16777216.0_f32;\n"
        "fn f() -> Above { 16777216.0_f32 }\n"
        "fn main() -> i32 { let x: Above = f(); 0 }\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "f32_predicate_arithmetic_false_pass.proof.json"

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert not any(
        obligation.get("status") == "proved"
        and obligation.get("refinement") == "Above"
        for obligation in artifact["obligations"]
    )


def test_gate_rejects_float_affine_bound_carry_false_pass(capsys, tmp_path):
    source = tmp_path / "float_affine_bound_carry_false_pass.hx"
    source.write_text(
        "type Source = f32 where self >= 16777216.0_f32;\n"
        "type Target = f32 where self + 1.0_f32 > 16777216.0_f32;\n"
        "fn make() -> Source { 16777216.0_f32 }\n"
        "fn main() -> Target { make() }\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "float_affine_bound_carry_false_pass.proof.json"

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert not any(
        carry.get("strategy") == "numeric-bound-implication"
        and carry.get("source_refinement") == "Source"
        and carry.get("target_refinement") == "Target"
        for carry in artifact["proof_carries"]
    )


def test_gate_rejects_integer_division_predicate_false_pass(capsys, tmp_path):
    source = tmp_path / "integer_division_predicate_false_pass.hx"
    source.write_text(
        "type HalfPositive = i32 where self / 2 > 0;\n"
        "fn f() -> HalfPositive { 1 }\n"
        "fn main() -> i32 { let x: HalfPositive = f(); 0 }\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "integer_division_predicate_false_pass.proof.json"

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert not any(
        obligation.get("status") == "proved"
        and obligation.get("refinement") == "HalfPositive"
        for obligation in artifact["obligations"]
    )


def test_gate_rejects_refined_initializer_machine_semantics_false_pass(
    capsys, tmp_path,
):
    source = tmp_path / "refined_initializer_machine_semantics_false_pass.hx"
    source.write_text(
        "type Positive = i32 where self > 0;\n"
        "fn f() -> i32 { let x: Positive = -1_i32 % 2_i32; 0 }\n"
        "fn main() -> i32 { f() }\n",
        encoding="utf-8",
    )
    artifact_path = (
        tmp_path / "refined_initializer_machine_semantics_false_pass.proof.json")

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert not any(
        obligation.get("status") == "proved"
        and obligation.get("context") == "let 'x'"
        for obligation in artifact["obligations"]
    )


def test_gate_rejects_refined_cast_hidden_nonfinite_false_pass(
    capsys, tmp_path,
):
    source = tmp_path / "refined_cast_hidden_nonfinite_false_pass.hx"
    source.write_text(
        "type AlwaysF64 = f64 where true;\n"
        "fn f() -> AlwaysF64 { (1e309_f64 as f64) as AlwaysF64 }\n"
        "fn main() -> i32 { 0 }\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "refined_cast_hidden_nonfinite_false_pass.json"

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert any(
        "value is not representable after casting f64 to f64" in error
        for error in artifact["typecheck_errors"]
    )


def test_gate_rejects_refined_cast_hidden_nonfinite_arithmetic_false_pass(
    capsys, tmp_path,
):
    source = tmp_path / "refined_cast_hidden_nonfinite_arithmetic.hx"
    source.write_text(
        "type AlwaysF64 = f64 where true;\n"
        "fn f() -> AlwaysF64 { (1e309_f64 + 0.0_f64) as AlwaysF64 }\n"
        "fn main() -> i32 { 0 }\n",
        encoding="utf-8",
    )
    artifact_path = (
        tmp_path / "refined_cast_hidden_nonfinite_arithmetic.proof.json")

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert any(
        "value is not representable after casting f64 to f64" in error
        for error in artifact["typecheck_errors"]
    )


def test_gate_rejects_f32_overflow_to_f64_refinement_false_pass(
    capsys, tmp_path,
):
    source = tmp_path / "f32_overflow_to_f64_refinement_false_pass.hx"
    source.write_text(
        "type AlwaysF64 = f64 where true;\n"
        "fn f() -> AlwaysF64 { (3.4028235e38_f32 * 2.0_f32) as AlwaysF64 }\n"
        "fn main() -> i32 { 0 }\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "f32_overflow_to_f64_refinement_false_pass.json"

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert any(
        "value is not representable after casting f32 to f64" in error
        for error in artifact["typecheck_errors"]
    )


def test_gate_rejects_f32_overflow_hidden_by_primitive_cast_return_false_pass(
    capsys, tmp_path,
):
    source = tmp_path / "f32_overflow_hidden_by_primitive_cast_return.hx"
    source.write_text(
        "type AlwaysF64 = f64 where true;\n"
        "fn f() -> AlwaysF64 { (3.4028235e38_f32 * 2.0_f32) as f64 }\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "f32_overflow_hidden_by_primitive_cast_return.json"

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert any(
        "return value of function 'f'" in error
        and "requires a representable target value" in error
        for error in artifact["typecheck_errors"]
    )


def test_gate_rejects_f32_overflow_hidden_by_top_level_const_return_false_pass(
    capsys, tmp_path,
):
    source = tmp_path / "f32_overflow_hidden_by_const_return.hx"
    source.write_text(
        "type AlwaysF64 = f64 where true;\n"
        "const OVER: f64 = (3.4028235e38_f32 * 2.0_f32) as f64;\n"
        "fn f() -> AlwaysF64 { OVER }\n"
        "fn main() -> i32 { 0 }\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "f32_overflow_hidden_by_const_return.json"

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert any(
        "return value of function 'f'" in error
        and "requires a representable target value" in error
        for error in artifact["typecheck_errors"]
    )


def test_gate_rejects_unrepresentable_hidden_by_if_return_false_pass(
    capsys, tmp_path,
):
    source = tmp_path / "unrepresentable_hidden_by_if_return.hx"
    source.write_text(
        "type AlwaysF64 = f64 where true;\n"
        "fn f(b: bool) -> AlwaysF64 { if b { 1e309_f64 } else { 0.0_f64 } }\n"
        "fn main() -> i32 { 0 }\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "unrepresentable_hidden_by_if_return.proof.json"

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert any(
        "return value of function 'f'" in error
        and "requires a representable target value" in error
        for error in artifact["typecheck_errors"]
    )


def test_gate_rejects_unrepresentable_primitive_return_producer_false_pass(
    capsys, tmp_path,
):
    source = tmp_path / "unrepresentable_primitive_return_producer.hx"
    source.write_text(
        "type AlwaysF64 = f64 where true;\n"
        "fn raw_bad() -> f64 { 1e309_f64 }\n"
        "fn f() -> AlwaysF64 { raw_bad() }\n"
        "fn main() -> i32 { 0 }\n",
        encoding="utf-8",
    )
    artifact_path = (
        tmp_path / "unrepresentable_primitive_return_producer.proof.json"
    )

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert any(
        "return value of function 'f'" in error
        and "requires a representable target value" in error
        for error in artifact["typecheck_errors"]
    )


def test_gate_rejects_unrepresentable_refined_return_call_arg_false_pass(
    capsys, tmp_path,
):
    source = tmp_path / "unrepresentable_refined_return_call_arg.hx"
    source.write_text(
        "type AlwaysF64 = f64 where true;\n"
        "fn accept(x: f64) -> AlwaysF64 { x }\n"
        "fn f() -> AlwaysF64 { accept(1e309_f64) }\n"
        "fn main() -> i32 { 0 }\n",
        encoding="utf-8",
    )
    artifact_path = (
        tmp_path / "unrepresentable_refined_return_call_arg.proof.json"
    )

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert any(
        "call to 'accept': arg 'x'" in error
        and "requires a representable target value" in error
        for error in artifact["typecheck_errors"]
    )


def test_gate_rejects_unrepresentable_generic_call_arg_false_pass(
    capsys, tmp_path,
):
    source = tmp_path / "unrepresentable_generic_call_arg.hx"
    source.write_text(
        "type AlwaysF64 = f64 where true;\n"
        "fn id[T](x: T) -> T { x }\n"
        "fn accept(x: f64) -> AlwaysF64 { x }\n"
        "fn f() -> AlwaysF64 { accept(id(1e309_f64)) }\n"
        "fn main() -> i32 { 0 }\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "unrepresentable_generic_call_arg.proof.json"

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert any(
        "call to 'accept': arg 'x'" in error
        and "requires a representable target value" in error
        for error in artifact["typecheck_errors"]
    )


def test_gate_rejects_unrepresentable_generic_wrapper_false_pass(
    capsys, tmp_path,
):
    source = tmp_path / "unrepresentable_generic_wrapper.hx"
    source.write_text(
        "type AlwaysF64 = f64 where true;\n"
        "fn id[T](x: T) -> T { x }\n"
        "fn via[T](x: T) -> AlwaysF64 { x }\n"
        "fn f() -> AlwaysF64 { via(id(id(1e309_f64))) }\n"
        "fn main() -> i32 { 0 }\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "unrepresentable_generic_wrapper.proof.json"

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert any(
        "call to 'via': arg 'x'" in error
        and "requires a representable target value" in error
        for error in artifact["typecheck_errors"]
    )


def test_gate_rejects_unrepresentable_index_assignment_false_pass(
    capsys, tmp_path,
):
    source = tmp_path / "unrepresentable_index_assignment.hx"
    source.write_text(
        "type AlwaysF64 = f64 where true;\n"
        "fn f(b: bool) -> AlwaysF64 {\n"
        "    let mut xs = [0.0_f64];\n"
        "    xs[0] = if b { 1e309_f64 } else { 0.0_f64 };\n"
        "    xs[0]\n"
        "}\n"
        "fn main() -> i32 { 0 }\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "unrepresentable_index_assignment.proof.json"

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert any(
        "return value of function 'f'" in error
        and "requires a representable target value" in error
        for error in artifact["typecheck_errors"]
    )


def test_gate_rejects_unrepresentable_index_assignment_wrong_index_repair(
    capsys, tmp_path,
):
    source = tmp_path / "unrepresentable_index_wrong_repair.hx"
    source.write_text(
        "type AlwaysF64 = f64 where true;\n"
        "fn f(b: bool) -> AlwaysF64 {\n"
        "    let mut xs = [0.0_f64, 0.0_f64];\n"
        "    xs[0] = if b { 1e309_f64 } else { 0.0_f64 };\n"
        "    xs[1] = 0.0_f64;\n"
        "    xs[0]\n"
        "}\n"
        "fn main() -> i32 { 0 }\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "unrepresentable_index_wrong_repair.proof.json"

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "typecheck_errors must be empty" in captured.err
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["typecheck_errors"] >= 1
    assert any(
        "return value of function 'f'" in error
        and "requires a representable target value" in error
        for error in artifact["typecheck_errors"]
    )


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
    assert "pipeline_errors must be empty when input.source_sha256 is null" in (
        captured.err
    )
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


def test_gate_rejects_replay_libraries(capsys, tmp_path):
    source = tmp_path / "input.hx"
    _write_clean_source(source)
    artifact_path = tmp_path / "input.proof.json"

    rc = proof_artifact_gate.main([
        str(source),
        "--artifact-out",
        str(artifact_path),
        "--",
        "-l",
        "forgedlib",
        "--no-stdlib",
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert "libraries are not allowed in proof_artifact_gate" in captured.err
    assert "forgedlib" in captured.err
    assert not artifact_path.exists()


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
