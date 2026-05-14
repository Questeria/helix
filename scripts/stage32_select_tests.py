"""Select focused pytest targets for changed Helix files.

This is a speed helper, not a coverage replacement. It chooses a conservative
first-pass test set for the edit loop; full gates still decide commits.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


TOOLING_TESTS = [
    "helixc/tests/test_stage32_select_tests.py",
    "helixc/tests/test_stage31_validate.py",
    "helixc/tests/test_pytest_shard.py",
]

PROOF_QUICK_TESTS = [
    "helixc/tests/test_proof_artifact_key.py",
    "helixc/tests/test_proof_artifact_validate.py",
    "helixc/tests/test_proof_artifact_gate.py",
]

STAGE31_PROOF_REGRESSION_TARGETS = [
    "helixc/tests/test_cli.py::test_stage31_emit_proof_obligations_cache_key_path_independent",
    "helixc/tests/test_cli.py::test_stage31_emit_proof_obligations_json_for_unsupported_refinement",
    "helixc/tests/test_cli.py::test_stage31_emit_proof_obligations_json_for_equivalent_refinement_alias",
    "helixc/tests/test_cli.py::test_stage31_emit_proof_obligations_rejects_generic_refinement_name",
    "helixc/tests/test_cli.py::test_stage31_emit_proof_obligations_rejects_duplicate_proof_names",
    "helixc/tests/test_typecheck.py::test_stage31_equivalent_refinement_aliases_carry_exact_proofs",
    "helixc/tests/test_typecheck.py::test_stage31_unsupported_refinement_predicates_do_not_carry_by_name",
]

SOURCE_RULES: list[tuple[tuple[str, ...], list[str]]] = [
    (("scripts/stage32_select_tests.py",), ["helixc/tests/test_stage32_select_tests.py"]),
    (("scripts/stage31_validate.py", "scripts/pytest_shard.py"), TOOLING_TESTS),
    (("scripts/selfhost_cascade.py",), ["helixc/tests/test_selfhost_cascade.py"]),
    (("scripts/selfhost_cascade_validate.py",), ["helixc/tests/test_selfhost_cascade_validate.py"]),
    (("scripts/stage33_selfhost_gate.py",), ["helixc/tests/test_stage33_selfhost_gate.py"]),
    (("scripts/proof_artifact_key.py", "scripts/proof_artifact_validate.py"), PROOF_QUICK_TESTS),
    (("helixc/check.py",), PROOF_QUICK_TESTS + STAGE31_PROOF_REGRESSION_TARGETS),
    (("helixc/backend/x86_64.py", "helixc/backend/elf_dyn.py"), [
        "helixc/tests/test_codegen.py",
        "helixc/tests/test_codegen_determinism.py",
        "helixc/tests/test_cli.py::test_o1_invokes_backend_default_pass_order",
    ]),
    (("helixc/backend/ptx.py",), ["helixc/tests/test_ptx.py"]),
    (("helixc/frontend/lexer.py",), ["helixc/tests/test_lexer.py", "helixc/tests/test_parser.py"]),
    (("helixc/frontend/parser.py",), [
        "helixc/tests/test_parser.py",
        "helixc/tests/test_typecheck.py",
        "helixc/tests/test_codegen.py",
    ]),
    (("helixc/frontend/typecheck.py",), [
        "helixc/tests/test_typecheck.py",
        *STAGE31_PROOF_REGRESSION_TARGETS,
    ]),
    (("helixc/frontend/autodiff.py",), ["helixc/tests/test_autodiff.py"]),
    (("helixc/frontend/autodiff_reverse.py",), [
        "helixc/tests/test_autodiff_reverse.py",
        "helixc/tests/test_autodiff_parity.py",
    ]),
    (("helixc/frontend/autotune.py",), ["helixc/tests/test_autotune.py"]),
    (("helixc/frontend/hash_cons.py",), ["helixc/tests/test_hash_cons.py"]),
    (("helixc/frontend/match_lower.py",), ["helixc/tests/test_match.py"]),
    (("helixc/frontend/panic_pass.py",), ["helixc/tests/test_panic.py"]),
    (("helixc/frontend/presburger.py",), ["helixc/tests/test_presburger.py"]),
    (("helixc/frontend/pytree.py",), ["helixc/tests/test_pytree.py"]),
    (("helixc/frontend/trace_pass.py",), ["helixc/tests/test_trace.py"]),
    (("helixc/frontend/totality.py",), ["helixc/tests/test_totality.py"]),
    (("helixc/frontend/unsafe_pass.py",), ["helixc/tests/test_unsafe.py"]),
    (("helixc/frontend/ast_hash.py",), ["helixc/tests/test_ast_hash.py"]),
    (("helixc/frontend/ast_walker.py",), ["helixc/tests/test_ast_walker.py"]),
    (("helixc/frontend/ast_nodes.py",), [
        "helixc/tests/test_parser.py",
        "helixc/tests/test_typecheck.py",
        "helixc/tests/test_ir.py",
    ]),
    (("helixc/ir/lower_ast.py", "helixc/ir/tir.py"), [
        "helixc/tests/test_ir.py",
        "helixc/tests/test_codegen.py",
    ]),
    (("helixc/ir/passes/const_fold.py",), ["helixc/tests/test_const_fold.py"]),
    (("helixc/ir/passes/cse.py",), ["helixc/tests/test_cse.py"]),
    (("helixc/ir/passes/dce.py",), ["helixc/tests/test_dce.py"]),
    (("helixc/ir/passes/effect_check.py",), ["helixc/tests/test_effect_check.py"]),
    (("helixc/ir/passes/fdce.py",), ["helixc/tests/test_fdce.py"]),
    (("helixc/ir/tile_ir.py",), ["helixc/tests/test_tile_ir.py"]),
    (("helixc/bootstrap/lexer.hx",), ["helixc/tests/test_lexer.py", "helixc/tests/test_parser.py"]),
    (("helixc/bootstrap/parser.hx",), ["helixc/tests/test_parser.py", "helixc/tests/test_codegen.py"]),
    (("helixc/bootstrap/kovc.hx", "helixc/bootstrap/evaluator.hx"), [
        "helixc/tests/test_codegen.py",
        "helixc/tests/test_parser.py",
    ]),
    (("helixc/stdlib/",), ["helixc/tests/test_codegen.py"]),
]


@dataclass(frozen=True)
class Selection:
    pytest_targets: list[str]
    extra_commands: list[str]
    notes: list[str]


def normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    if len(normalized) >= 3 and normalized[1] == ":" and normalized[2] == "/":
        try:
            normalized = str(Path(normalized).resolve().relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            normalized = Path(normalized).name
    return normalized.lstrip("./").lower()


def add_unique(items: list[str], additions: list[str]) -> None:
    for item in additions:
        if item not in items:
            items.append(item)


def rule_matches(path: str, prefix: str) -> bool:
    if prefix.endswith("/"):
        return path.startswith(prefix)
    return path == prefix


def git_lines(args: list[str]) -> list[str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def include_untracked_path(path: str) -> bool:
    normalized = normalize_path(path)
    return normalized.startswith("helixc/") or normalized.startswith("scripts/")


def changed_paths_from_git(base: str) -> list[str]:
    paths: list[str] = []
    add_unique(paths, git_lines(["diff", "--name-only", base]))
    add_unique(paths, [
        path
        for path in git_lines(["ls-files", "--others", "--exclude-standard"])
        if include_untracked_path(path)
    ])
    return paths


def select_tests_for_paths(paths: list[str]) -> Selection:
    normalized = [normalize_path(path) for path in paths if path.strip()]
    pytest_targets: list[str] = []
    notes: list[str] = []
    extra_commands: list[str] = []

    for path in normalized:
        if path.startswith("helixc/tests/test_") and path.endswith(".py"):
            add_unique(pytest_targets, [path])
            continue

        matched = False
        for prefixes, targets in SOURCE_RULES:
            if any(rule_matches(path, prefix) for prefix in prefixes):
                add_unique(pytest_targets, targets)
                matched = True

        if matched:
            continue

        if path.endswith(".md") or path.startswith("docs/"):
            continue

        if path.startswith("helixc/") or path.startswith("scripts/"):
            add_unique(pytest_targets, [
                "helixc/tests/test_cli.py::test_o1_invokes_backend_default_pass_order",
                "helixc/tests/test_codegen.py",
            ])
            notes.append(f"broad fallback for {path}")

    if normalized and not pytest_targets:
        add_unique(extra_commands, ["git diff --check"])
        notes.append("docs-only or non-code change")

    return Selection(
        pytest_targets=pytest_targets,
        extra_commands=extra_commands,
        notes=notes,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        help="changed file paths; if omitted, git diff --name-only HEAD is used",
    )
    parser.add_argument("--base", default="HEAD", help="git diff base when paths are omitted")
    parser.add_argument("--json", action="store_true", help="emit machine-readable selection")
    parser.add_argument("--explain", action="store_true", help="print notes after pytest targets")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    paths = args.paths or changed_paths_from_git(args.base)
    selection = select_tests_for_paths(paths)

    if args.json:
        print(json.dumps({
            "schema": "helix.stage32.focused_test_selection.v0",
            "paths": [normalize_path(path) for path in paths],
            "pytest_targets": selection.pytest_targets,
            "extra_commands": selection.extra_commands,
            "notes": selection.notes,
        }, indent=2, sort_keys=True))
        return 0

    for target in selection.pytest_targets:
        print(target)
    if args.explain:
        for command in selection.extra_commands:
            print(f"# extra: {command}")
        for note in selection.notes:
            print(f"# note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
