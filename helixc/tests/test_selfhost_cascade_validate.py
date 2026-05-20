"""Tests for self-host cascade report validation."""

from __future__ import annotations

import json

from scripts import selfhost_cascade_validate


STABLE_SHA = "5a7367ad436e72ade3d8f96a9860e0d08b64528cbb15295e1a47076090667408"


def valid_report() -> dict[str, object]:
    stable_size = 277899
    return {
        "schema": "helix.selfhost_cascade.v0",
        "stable": True,
        "stable_sha256": STABLE_SHA,
        "stable_size": stable_size,
        "self_host_generations": [
            {
                "generation": 2,
                "exit_low_byte": stable_size & 0xFF,
                "size": stable_size,
                "sha256": STABLE_SHA,
            },
            {
                "generation": 3,
                "exit_low_byte": stable_size & 0xFF,
                "size": stable_size,
                "sha256": STABLE_SHA,
            },
        ],
        "smoke": [
            {"name": "literal", "expected_exit": 42, "actual_exit": 42},
            {"name": "call", "expected_exit": 42, "actual_exit": 42},
            {"name": "loop", "expected_exit": 42, "actual_exit": 42},
            {"name": "metadata_attrs", "expected_exit": 42, "actual_exit": 42},
        ],
    }


def test_selfhost_cascade_validate_accepts_stable_report():
    assert selfhost_cascade_validate.validate_report(valid_report()) == []


def test_selfhost_cascade_validate_rejects_unstable_report():
    report = valid_report()
    report["stable"] = False

    errors = selfhost_cascade_validate.validate_report(report)

    assert "stable must be true" in errors


def test_selfhost_cascade_validate_rejects_generation_drift():
    report = valid_report()
    report["self_host_generations"][1]["sha256"] = "0" * 64
    report["self_host_generations"][1]["exit_low_byte"] = 0

    errors = selfhost_cascade_validate.validate_report(report)

    assert "self_host_generations[1].sha256 does not match stable_sha256" in errors
    assert (
        "self_host_generations[1].exit_low_byte does not match stable_size low byte"
        in errors
    )


def test_selfhost_cascade_validate_rejects_generation_size_drift():
    report = valid_report()
    report["self_host_generations"][1]["size"] = 999999

    errors = selfhost_cascade_validate.validate_report(report)

    assert "self_host_generations[1].size does not match stable_size" in errors


def test_selfhost_cascade_validate_rejects_wrong_smoke_actual_exit():
    report = valid_report()
    report["smoke"][0]["actual_exit"] = 7

    errors = selfhost_cascade_validate.validate_report(report)

    assert "smoke literal.actual_exit must be 42" in errors


def test_selfhost_cascade_validate_rejects_missing_smoke():
    report = valid_report()
    report["smoke"] = report["smoke"][:2]

    errors = selfhost_cascade_validate.validate_report(report)

    assert "smoke missing loop" in errors


def test_selfhost_cascade_validate_rejects_expected_hash_mismatch():
    errors = selfhost_cascade_validate.validate_report(
        valid_report(),
        expect_stable_sha="0" * 64,
    )

    assert "stable_sha256 does not match expected hash" in errors


def test_selfhost_cascade_validate_cli_accepts_report(tmp_path, capsys):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(valid_report()), encoding="utf-8")

    rc = selfhost_cascade_validate.main([str(path)])
    captured = capsys.readouterr()

    assert rc == 0
    assert "selfhost-cascade-validate: ok" in captured.out


def test_selfhost_cascade_validate_cli_rejects_bad_json(tmp_path, capsys):
    path = tmp_path / "report.json"
    path.write_text("not json", encoding="utf-8")

    rc = selfhost_cascade_validate.main([str(path)])
    captured = capsys.readouterr()

    assert rc == 1
    assert "selfhost-cascade-validate:" in captured.err
