"""Tests for the Stage 32 focused-test selector."""

from __future__ import annotations

import json
from types import SimpleNamespace

from scripts import stage32_select_tests


def test_stage32_selector_maps_validation_tooling_to_tool_tests():
    selection = stage32_select_tests.select_tests_for_paths([
        "scripts/stage31_validate.py",
        "scripts/pytest_shard.py",
    ])

    assert selection.pytest_targets == [
        "helixc/tests/test_stage32_select_tests.py",
        "helixc/tests/test_stage31_validate.py",
        "helixc/tests/test_pytest_shard.py",
    ]
    assert selection.extra_commands == []


def test_stage32_selector_maps_test_files_to_themselves():
    selection = stage32_select_tests.select_tests_for_paths([
        "helixc/tests/test_codegen.py",
    ])

    assert selection.pytest_targets == ["helixc/tests/test_codegen.py"]


def test_stage32_selector_maps_typecheck_to_stage31_proof_regressions():
    selection = stage32_select_tests.select_tests_for_paths([
        "helixc/frontend/typecheck.py",
    ])

    assert selection.pytest_targets[0] == "helixc/tests/test_typecheck.py"
    assert (
        "helixc/tests/test_cli.py::test_stage31_emit_proof_obligations_json_for_equivalent_refinement_alias"
        in selection.pytest_targets
    )
    assert (
        "helixc/tests/test_typecheck.py::test_stage31_equivalent_refinement_aliases_carry_exact_proofs"
        in selection.pytest_targets
    )


def test_stage32_selector_maps_backend_to_codegen_and_determinism():
    selection = stage32_select_tests.select_tests_for_paths([
        "helixc/backend/x86_64.py",
    ])

    assert selection.pytest_targets == [
        "helixc/tests/test_codegen.py",
        "helixc/tests/test_codegen_determinism.py",
        "helixc/tests/test_cli.py::test_o1_invokes_backend_default_pass_order",
    ]


def test_stage32_selector_maps_selfhost_cascade_to_own_test():
    selection = stage32_select_tests.select_tests_for_paths([
        "scripts/selfhost_cascade.py",
        "scripts/selfhost_cascade_validate.py",
    ])

    assert selection.pytest_targets == [
        "helixc/tests/test_selfhost_cascade.py",
        "helixc/tests/test_selfhost_cascade_validate.py",
    ]
    assert selection.notes == []


def test_stage32_selector_maps_docs_only_to_diff_check():
    selection = stage32_select_tests.select_tests_for_paths([
        "docs/stage32-verification-speed-2026-05-14.md",
    ])

    assert selection.pytest_targets == []
    assert selection.extra_commands == ["git diff --check"]
    assert selection.notes == ["docs-only or non-code change"]


def test_stage32_selector_uses_broad_fallback_for_unknown_source():
    selection = stage32_select_tests.select_tests_for_paths([
        "helixc/frontend/new_pass.py",
    ])

    assert selection.pytest_targets == [
        "helixc/tests/test_cli.py::test_o1_invokes_backend_default_pass_order",
        "helixc/tests/test_codegen.py",
    ]
    assert selection.notes == ["broad fallback for helixc/frontend/new_pass.py"]


def test_stage32_selector_exact_file_rules_do_not_match_suffixes():
    selection = stage32_select_tests.select_tests_for_paths([
        "scripts/stage31_validate.py.bak",
    ])

    assert selection.pytest_targets == [
        "helixc/tests/test_cli.py::test_o1_invokes_backend_default_pass_order",
        "helixc/tests/test_codegen.py",
    ]
    assert selection.notes == ["broad fallback for scripts/stage31_validate.py.bak"]


def test_stage32_selector_cli_prints_plain_targets(capsys):
    rc = stage32_select_tests.main(["scripts/stage32_select_tests.py"])
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.out.splitlines() == ["helixc/tests/test_stage32_select_tests.py"]


def test_stage32_selector_cli_emits_json(capsys):
    rc = stage32_select_tests.main([
        "--json",
        "docs/ROADMAP.md",
    ])
    captured = capsys.readouterr()

    payload = json.loads(captured.out)
    assert rc == 0
    assert payload["schema"] == "helix.stage32.focused_test_selection.v0"
    assert payload["pytest_targets"] == []
    assert payload["extra_commands"] == ["git diff --check"]


def test_stage32_selector_git_paths_include_untracked(monkeypatch):
    def fake_run(cmd, **_kwargs):
        if cmd[1:] == ["diff", "--name-only", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="docs/ROADMAP.md\n", stderr="")
        if cmd[1:] == ["ls-files", "--others", "--exclude-standard"]:
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "scripts/stage32_select_tests.py\n"
                    "docs/audit-stage30-cycle114-codereview.md\n"
                ),
                stderr="",
            )
        raise AssertionError(cmd)

    monkeypatch.setattr(stage32_select_tests.subprocess, "run", fake_run)

    assert stage32_select_tests.changed_paths_from_git("HEAD") == [
        "docs/ROADMAP.md",
        "scripts/stage32_select_tests.py",
    ]
