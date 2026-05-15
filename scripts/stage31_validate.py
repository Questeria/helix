"""Stage 31 validation runner.

The runner speeds up validation by orchestrating the same checks with less
manual waiting:

- quick mode runs the recent high-signal regression tests.
- full mode runs sharded non-codegen and `test_codegen.py` workers in
  parallel.
- snapshot smoke verifies the copied Stage 30 compiler snapshot can check,
  compile, and run a tiny Helix program.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stage32_select_tests

LOG_DIR = ROOT / ".stage31-logs"
BIN_DIR = ROOT / ".stage31-bin"
MAX_SHARDS = 8
MAX_NO_CODEGEN_SHARDS = 4
PYTEST_SUMMARY_RE = re.compile(r"\bin (?P<seconds>[0-9]+(?:\.[0-9]+)?)s(?:\s|$)")
TIMING_SUMMARY_PATH = LOG_DIR / "pytest-shard-timings.json"
NODE_TIMING_SUMMARY_PATH = LOG_DIR / "pytest-node-durations.json"
NODE_DURATION_SCHEMA = "helix.pytest_node_durations.v0"


def default_shards() -> int:
    """Pick a conservative parallel default without reducing test coverage."""
    return min(MAX_SHARDS, max(1, os.cpu_count() or 4))


def no_codegen_shards_for(shards: int) -> int:
    """Shard the broad non-codegen suite without overloading WSL/IO."""
    return max(1, min(MAX_NO_CODEGEN_SHARDS, shards))


def validation_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if os.environ.get("WSL_DISTRO_NAME"):
        BIN_DIR.mkdir(exist_ok=True)
        shim = BIN_DIR / "wsl"
        shim.write_text(
            "#!/usr/bin/env bash\n"
            "if [[ \"${1:-}\" == \"--\" || \"${1:-}\" == \"-e\" ]]; then\n"
            "    shift\n"
            "fi\n"
            "exec \"$@\"\n",
            encoding="utf-8",
        )
        shim.chmod(0o755)
        env["PATH"] = (
            f"{BIN_DIR}:/usr/local/sbin:/usr/local/bin:"
            "/usr/sbin:/usr/bin:/sbin:/bin"
        )
    if extra:
        env.update(extra)
    return env


def run_logged(
    name: str,
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | str | None = None,
) -> int:
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT if cwd is None else cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log.write(proc.stdout)
    elapsed = time.monotonic() - started
    print(f"{name}: rc={proc.returncode} time={elapsed:.1f}s log={log_path}")
    return proc.returncode


ParallelJob = tuple[str, list[str], dict[str, str] | None]


def run_parallel_once(jobs: list[ParallelJob]) -> tuple[int, list[ParallelJob]]:
    LOG_DIR.mkdir(exist_ok=True)
    procs = []
    started = time.monotonic()
    for name, cmd, env in jobs:
        log_path = LOG_DIR / f"{name}.log"
        log = log_path.open("w", encoding="utf-8")
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        procs.append((name, proc, log, log_path, cmd, env))
    rc = 0
    failed: list[ParallelJob] = []
    for name, proc, log, log_path, cmd, env in procs:
        code = proc.wait()
        log.close()
        print(f"{name}: rc={code} log={log_path}")
        rc = rc or code
        if code:
            failed.append((name, cmd, env))
    elapsed = time.monotonic() - started
    print(f"parallel group time={elapsed:.1f}s")
    print_slowest_pytest_shards(procs)
    merge_pytest_node_duration_files(procs)
    return rc, failed


def run_parallel(
    jobs: list[ParallelJob], *, retry_failed_once: bool = False,
) -> int:
    rc, failed = run_parallel_once(jobs)
    if rc and retry_failed_once and failed:
        print("retrying failed shards once:")
        retry_jobs = [
            (f"{name}-retry1", cmd, env) for name, cmd, env in failed
        ]
        retry_rc, retry_failed = run_parallel_once(retry_jobs)
        if retry_rc == 0:
            recovered = ", ".join(name for name, _cmd, _env in failed)
            print(f"failed shards recovered on retry: {recovered}")
            return 0
        still_failed = ", ".join(name for name, _cmd, _env in retry_failed)
        print(f"failed shards still failing after retry: {still_failed}")
        return retry_rc
    return rc


def pytest_summary_from_log(log_path: Path) -> tuple[float, str] | None:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if " in " not in line:
            continue
        if not any(word in line for word in (" passed", " failed", " error")):
            continue
        match = PYTEST_SUMMARY_RE.search(line)
        if match is None:
            continue
        return float(match.group("seconds")), line.strip()
    return None


def print_slowest_pytest_shards(
    procs: list[tuple[str, subprocess.Popen[str], object, Path, list[str], dict[str, str] | None]],
) -> None:
    summaries = pytest_shard_summaries(procs)
    if not summaries:
        return
    write_pytest_timing_summary(summaries)
    print("slowest pytest shards:")
    for summary in summaries[:5]:
        print(
            f"  {summary['name']}: {summary['seconds']:.1f}s | "
            f"{summary['summary']}"
        )


def pytest_shard_summaries(
    procs: list[tuple[str, subprocess.Popen[str], object, Path, list[str], dict[str, str] | None]],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for name, _proc, _log, log_path, _cmd, _env in procs:
        summary = pytest_summary_from_log(log_path)
        if summary is None:
            continue
        seconds, line = summary
        summaries.append({
            "name": name,
            "seconds": seconds,
            "summary": line,
            "log": str(log_path),
        })
    return sorted(summaries, key=lambda item: item["seconds"], reverse=True)


def write_pytest_timing_summary(
    summaries: list[dict[str, object]],
    path: Path = TIMING_SUMMARY_PATH,
) -> None:
    payload = {
        "schema": "helix.stage31.pytest_shard_timings.v0",
        "generated_at_unix": time.time(),
        "shards": summaries,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def node_duration_path_for(job_name: str) -> Path:
    return LOG_DIR / f"{job_name}-node-durations.json"


def load_pytest_node_durations(path: Path) -> dict[str, float]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    raw_tests = payload.get("tests", {})
    if isinstance(raw_tests, dict):
        iterator = raw_tests.items()
    elif isinstance(raw_tests, list):
        iterator = (
            (entry.get("nodeid"), entry.get("seconds"))
            for entry in raw_tests
            if isinstance(entry, dict)
        )
    else:
        return {}
    durations: dict[str, float] = {}
    for nodeid, seconds in iterator:
        if not isinstance(nodeid, str):
            continue
        if not isinstance(seconds, (int, float)) or seconds < 0:
            continue
        durations[nodeid] = float(seconds)
    return durations


def write_pytest_node_durations(
    durations: dict[str, float],
    path: Path = NODE_TIMING_SUMMARY_PATH,
) -> None:
    payload = {
        "schema": NODE_DURATION_SCHEMA,
        "generated_at_unix": time.time(),
        "tests": [
            {"nodeid": nodeid, "seconds": seconds}
            for nodeid, seconds in sorted(durations.items())
        ],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def merge_pytest_node_duration_files(
    procs: list[tuple[str, subprocess.Popen[str], object, Path, list[str], dict[str, str] | None]],
    path: Path = NODE_TIMING_SUMMARY_PATH,
) -> None:
    durations = load_pytest_node_durations(path)
    updated = False
    for name, _proc, _log, _log_path, _cmd, _env in procs:
        shard_durations = load_pytest_node_durations(node_duration_path_for(name))
        if not shard_durations:
            continue
        durations.update(shard_durations)
        updated = True
    if updated:
        write_pytest_node_durations(durations, path)


def quick(py: str) -> int:
    return run_logged(
        "stage31-quick",
        [
            py,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "helixc/tests/test_typecheck.py::test_stage31_nonrecursive_aggregate_returns_rejected_before_lowering",
            "helixc/tests/test_typecheck.py::test_stage31_refined_array_call_and_return_checked",
            "helixc/tests/test_stage31_validate.py::test_snapshot_smoke_runs_modules_outside_repo_root",
            "helixc/tests/test_cli.py::test_o1_invokes_backend_default_pass_order",
            "helixc/tests/test_cli.py::test_o2_invokes_cse_and_dce",
            "helixc/tests/test_cli.py::test_o3_runs_at_least_o2_passes",
            "helixc/tests/test_cli.py::test_stage31_emit_proof_obligations_cache_key_path_independent",
            "helixc/tests/test_cli.py::test_stage31_emit_proof_obligations_json_for_unsupported_refinement",
            "helixc/tests/test_cli.py::test_stage31_emit_proof_obligations_json_for_equivalent_refinement_alias",
            "helixc/tests/test_cli.py::test_stage34_emit_proof_carry_json_for_same_refinement_strategy",
            "helixc/tests/test_cli.py::test_stage34_emit_proof_obligations_json_for_numeric_bound_implication",
            "helixc/tests/test_cli.py::test_stage34_emit_proof_obligations_json_for_equality_bound_implication",
            "helixc/tests/test_cli.py::test_stage34_emit_proof_carry_json_for_array_bound_implication",
            "helixc/tests/test_cli.py::test_stage34_emit_proof_carry_json_for_tuple_bound_implication",
            "helixc/tests/test_cli.py::test_stage34_emit_proof_carry_json_for_negated_bounds_only",
            "helixc/tests/test_cli.py::test_stage34_emit_proof_obligations_json_for_refined_cast_target_value",
            "helixc/tests/test_cli.py::test_stage34_failed_refined_cast_does_not_emit_return_carry",
            "helixc/tests/test_cli.py::test_stage34_failed_refined_composite_casts_do_not_emit_carries",
            "helixc/tests/test_cli.py::test_stage34_failed_refined_initializer_does_not_emit_later_carries",
            "helixc/tests/test_cli.py::test_stage34_failed_local_const_initializer_does_not_emit_later_carries",
            "helixc/tests/test_cli.py::test_stage34_failed_top_level_const_initializer_does_not_emit_later_carries",
            "helixc/tests/test_cli.py::test_stage34_self_independent_unrepresentable_value_is_not_clean",
            "helixc/tests/test_cli.py::test_stage34_f32_predicate_rounding_does_not_false_carry_bounds",
            "helixc/tests/test_cli.py::test_stage34_fixed_point_unbound_name_is_not_clean",
            "helixc/tests/test_cli.py::test_stage34_emit_proof_obligations_json_for_refined_f32_rounding",
            "helixc/tests/test_cli.py::test_stage34_emit_proof_carry_json_for_explicit_return_route",
            "helixc/tests/test_cli.py::test_stage34_emit_proof_carry_json_for_refined_cast_route",
            "helixc/tests/test_cli.py::test_stage34_emit_proof_carry_json_for_function_typed_call_route",
            "helixc/tests/test_cli.py::test_stage31_emit_proof_obligations_rejects_generic_refinement_name",
            "helixc/tests/test_cli.py::test_stage31_emit_proof_obligations_rejects_duplicate_proof_names",
            "helixc/tests/test_cli.py::test_stage31_emit_proof_obligations_json_for_boolean_literal_refinements",
            "helixc/tests/test_cli.py::test_stage31_emit_proof_obligations_json_for_self_independent_false",
            "helixc/tests/test_cli.py::test_stage31_emit_proof_obligations_json_for_mixed_independent_predicates",
            "helixc/tests/test_cli.py::test_stage31_emit_proof_obligations_json_for_inherited_independent_false",
            "helixc/tests/test_cli.py::test_stage31_emit_proof_obligations_json_for_short_circuit_predicates",
            "helixc/tests/test_cli.py::test_stage31_emit_proof_obligations_json_for_constant_comparisons",
            "helixc/tests/test_typecheck.py::test_stage31_boolean_literal_refinement_predicates_are_supported",
            "helixc/tests/test_typecheck.py::test_stage31_equivalent_refinement_aliases_carry_exact_proofs",
            "helixc/tests/test_typecheck.py::test_stage34_numeric_bound_implication_carries_proofs",
            "helixc/tests/test_typecheck.py::test_stage34_numeric_bound_implication_respects_strictness",
            "helixc/tests/test_typecheck.py::test_stage34_equality_refinement_implies_matching_bounds",
            "helixc/tests/test_typecheck.py::test_stage34_equality_refinement_keeps_strict_bounds_fail_closed",
            "helixc/tests/test_typecheck.py::test_stage34_compound_numeric_bounds_carry_proofs",
            "helixc/tests/test_typecheck.py::test_stage34_numeric_bounds_carry_through_array_and_tuple_proofs",
            "helixc/tests/test_typecheck.py::test_stage34_negated_comparison_refinements_are_supported",
            "helixc/tests/test_typecheck.py::test_stage34_negated_comparison_bounds_carry_proofs",
            "helixc/tests/test_typecheck.py::test_stage34_affine_numeric_bounds_fail_closed_for_fixed_width_numbers",
            "helixc/tests/test_typecheck.py::test_stage34_affine_numeric_bounds_keep_strictness",
            "helixc/tests/test_typecheck.py::test_stage34_named_constant_bounds_carry_proofs",
            "helixc/tests/test_typecheck.py::test_stage34_numeric_bound_implication_requires_same_erased_base",
            "helixc/tests/test_typecheck.py::test_stage34_refined_cast_checks_target_converted_value",
            "helixc/tests/test_typecheck.py::test_stage34_refined_cast_rejects_boolean_source_to_numeric_refinement",
            "helixc/tests/test_typecheck.py::test_stage34_refined_integer_alias_checks_base_width_before_proof",
            "helixc/tests/test_typecheck.py::test_stage34_refined_f32_checks_rounded_target_value",
            "helixc/tests/test_typecheck.py::test_stage34_refined_f32_rejects_overflow_before_proof",
            "helixc/tests/test_typecheck.py::test_stage34_refined_f32_rejects_nonfinite_literal_before_proof",
            "helixc/tests/test_typecheck.py::test_stage34_refined_f64_rejects_nonfinite_literal_before_proof",
            "helixc/tests/test_typecheck.py::test_stage34_refined_integer_cast_rejects_nonfinite_before_proof",
            "helixc/tests/test_typecheck.py::test_stage34_self_independent_refinement_rejects_unrepresentable_values",
            "helixc/tests/test_typecheck.py::test_stage34_unrepresentable_scalar_evidence_covers_value_surfaces",
            "helixc/tests/test_typecheck.py::test_stage34_unrepresentable_scalar_evidence_covers_index_assignment",
            "helixc/tests/test_typecheck.py::test_stage34_unrepresentable_primitive_return_producer_is_not_clean",
            "helixc/tests/test_typecheck.py::test_stage34_unrepresentable_scalar_evidence_rejects_refined_return_call_args",
            "helixc/tests/test_typecheck.py::test_stage34_unrepresentable_scalar_evidence_rejects_generic_call_args",
            "helixc/tests/test_typecheck.py::test_stage34_unrepresentable_scalar_evidence_rejects_generic_wrappers",
            "helixc/tests/test_typecheck.py::test_stage34_fixed_point_preserves_unbound_name_errors",
            "helixc/tests/test_typecheck.py::test_stage34_refinement_predicate_float_literals_use_target_suffix",
            "helixc/tests/test_typecheck.py::test_stage34_numeric_bound_carry_uses_represented_predicate_literals",
            "helixc/tests/test_typecheck.py::test_stage34_numeric_bound_carry_uses_represented_f64_predicates",
            "helixc/tests/test_typecheck.py::test_stage34_const_predicate_uses_declared_scalar_representation",
            "helixc/tests/test_typecheck.py::test_stage34_predicate_arithmetic_rejects_nonfinite_results",
            "helixc/tests/test_typecheck.py::test_stage34_f32_predicate_arithmetic_rounds_each_operation",
            "helixc/tests/test_typecheck.py::test_stage34_float_affine_bound_carry_fails_closed",
            "helixc/tests/test_typecheck.py::test_stage34_integer_predicate_arithmetic_uses_machine_semantics",
            "helixc/tests/test_typecheck.py::test_stage34_refined_initializers_use_source_machine_semantics",
            "helixc/tests/test_typecheck.py::test_stage34_fixed_point_preserves_unknown_type_errors",
            "helixc/tests/test_typecheck.py::test_stage31_unsupported_refinement_predicates_do_not_carry_by_name",
            "helixc/tests/test_typecheck.py::test_stage31_generic_qualified_refinement_names_are_unsupported",
            "helixc/tests/test_typecheck.py::test_stage31_duplicate_refinement_names_fail_closed",
            "helixc/tests/test_typecheck.py::test_stage31_self_independent_false_refinement_rejects_unknown_values",
            "helixc/tests/test_typecheck.py::test_stage31_mixed_self_independent_refinements_do_not_downgrade",
            "helixc/tests/test_typecheck.py::test_stage31_boolean_short_circuit_refinements_are_decisive",
            "helixc/tests/test_typecheck.py::test_stage31_constant_comparison_refinement_predicates_are_supported",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_real_artifact_with_source_passes",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_rejects_source_hash_mismatch",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_rejects_stale_artifact_path_by_default",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_rejects_malformed_diagnostic_sections",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_rejects_missing_embedded_source_path",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_resolves_relative_artifact_path_from_artifact_dir",
            "helixc/tests/test_proof_artifact_validate.py::test_require_clean_uses_embedded_relative_source_path",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_rejects_cache_key_mismatch",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_rejects_forged_artifact_path_with_source_by_default",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_rejects_null_artifact_path_with_source_by_default",
            "helixc/tests/test_proof_artifact_validate.py::test_require_clean_rejects_forged_artifact_path_with_source",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_checks_stage34_proof_carry_records",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_rejects_missing_proof_carry_metadata_by_default",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_rejects_erased_proof_carries_with_source_by_default",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_rejects_erased_proof_carries_from_embedded_source_path",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_rejects_forged_clean_status_with_source_by_default",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_rejects_forged_input_and_cache_with_source_by_default",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_rejects_unsafe_replay_flags_without_side_effect",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_rejects_replay_libs",
            "helixc/tests/test_proof_artifact_validate.py::test_require_clean_rejects_forged_clean_artifact_with_source",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_rejects_boolean_integer_fields",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_rejects_bad_stdlib_manifest_hash",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_rejects_malformed_missing_stdlib_entry",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_source_unavailable_rejects_proof_content",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_source_unavailable_rejects_unsafe_flags",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_source_unavailable_rejects_impossible_input_metadata",
            "helixc/tests/test_proof_artifact_validate.py::test_require_clean_accepts_proved_artifact",
            "helixc/tests/test_proof_artifact_validate.py::test_require_clean_rejects_unproven_obligation",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_accepts_unsupported_obligation_structurally",
            "helixc/tests/test_proof_artifact_validate.py::test_require_clean_rejects_unsupported_obligation",
            "helixc/tests/test_proof_artifact_validate.py::test_require_clean_rejects_pipeline_errors",
            "helixc/tests/test_proof_artifact_validate.py::test_require_clean_rejects_promoted_warning",
            "helixc/tests/test_proof_artifact_validate.py::test_require_clean_rejects_source_unavailable_artifact",
            "helixc/tests/test_proof_artifact_gate.py",
        ],
        env=validation_env(),
    )


def focused(py: str, paths: list[str], *, base: str = "HEAD") -> int:
    changed_paths = paths or stage32_select_tests.changed_paths_from_git(base)
    selection = stage32_select_tests.select_tests_for_paths(changed_paths)

    if changed_paths:
        print("focused: changed paths:")
        for path in changed_paths:
            print(f"  {path}")
    else:
        print("focused: no changed paths")

    if selection.pytest_targets:
        print("focused: pytest targets:")
        for target in selection.pytest_targets:
            print(f"  {target}")
        return run_logged(
            "stage32-focused",
            [
                py,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                *selection.pytest_targets,
            ],
            env=validation_env(),
        )

    if "git diff --check" in selection.extra_commands:
        print("focused: no pytest targets; running git diff --check")
        return run_logged(
            "stage32-focused-diff-check",
            ["git", "diff", "--check"],
            env=validation_env(),
        )

    print("focused: nothing selected")
    return 0


def full(py: str, shards: int, *, retry_failed_once: bool = True) -> int:
    no_codegen_shards = no_codegen_shards_for(shards)
    print(f"full: no-codegen shards={no_codegen_shards} codegen shards={shards}")
    env = validation_env()
    jobs: list[tuple[str, list[str], dict[str, str] | None]] = []
    weights_path = str(NODE_TIMING_SUMMARY_PATH)
    for index in range(no_codegen_shards):
        name = f"pytest-no-codegen-shard-{index + 1}-of-{no_codegen_shards}"
        jobs.append(
            (
                name,
                [
                    py,
                    "scripts/pytest_shard.py",
                    "--total",
                    str(no_codegen_shards),
                    "--index",
                    str(index),
                    "--weights",
                    weights_path,
                    "--durations-out",
                    str(node_duration_path_for(name)),
                    "helixc/tests",
                    "--ignore=helixc/tests/test_codegen.py",
                ],
                env,
            )
        )
    for index in range(shards):
        name = f"pytest-codegen-shard-{index + 1}-of-{shards}"
        jobs.append(
            (
                name,
                [
                    py,
                    "scripts/pytest_shard.py",
                    "--total",
                    str(shards),
                    "--index",
                    str(index),
                    "--weights",
                    weights_path,
                    "--durations-out",
                    str(node_duration_path_for(name)),
                    "helixc/tests/test_codegen.py",
                ],
                env,
            )
        )
    return run_parallel(jobs, retry_failed_once=retry_failed_once)


def snapshot_smoke(py: str) -> int:
    snapshot = ROOT / "HELIX_STAGE30_COMPILER_SNAPSHOT"
    if not snapshot.exists():
        print(
            "snapshot-smoke: missing HELIX_STAGE30_COMPILER_SNAPSHOT; "
            "use --skip-snapshot to skip this gate explicitly",
            file=sys.stderr,
        )
        return 1
    scratch = (Path("/mnt/c/Projects/Helix-Scratch")
               if os.environ.get("WSL_DISTRO_NAME")
               else Path("C:/Projects/Helix-Scratch"))
    scratch.mkdir(parents=True, exist_ok=True)
    src = scratch / "stage31_validate_hello.hx"
    out = scratch / "stage31_validate_hello.bin"
    src.write_text("fn main() -> i32 { 42 }\n", encoding="utf-8")
    env = validation_env({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(snapshot),
    })
    rc = run_logged(
        "snapshot-check",
        [py, "-m", "helixc.check", "--check-only", "--strict", str(src)],
        env=env,
        cwd=scratch,
    )
    if rc:
        return rc
    rc = run_logged(
        "snapshot-compile",
        [
            py,
            "-m",
            "helixc.backend.x86_64",
            str(src),
            str(out),
            "--no-stdlib",
        ],
        env=env,
        cwd=scratch,
    )
    if rc:
        return rc
    run_script = (
        "chmod +x /mnt/c/Projects/Helix-Scratch/stage31_validate_hello.bin "
        "&& /mnt/c/Projects/Helix-Scratch/stage31_validate_hello.bin"
    )
    if os.environ.get("WSL_DISTRO_NAME"):
        run_cmd = ["bash", "-lc", run_script]
    else:
        run_cmd = ["wsl", "--", "bash", "-lc", run_script]
    run = subprocess.run(run_cmd, cwd=str(ROOT))
    print(f"snapshot-run: rc={run.returncode}")
    return 0 if run.returncode == 42 else 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("focused", "quick", "full"),
        default="quick",
        help=(
            "focused selects tests from changed files; quick runs recent "
            "regressions; full runs the broad suite"
        ),
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=default_shards(),
        help="codegen shard count for full mode (default: min(cpu_count, 8))",
    )
    parser.add_argument(
        "--no-retry-failed",
        action="store_true",
        help="do not rerun failed parallel shards once before returning failure",
    )
    parser.add_argument("--skip-snapshot", action="store_true")
    parser.add_argument(
        "--base",
        default="HEAD",
        help="git diff base for focused mode when no paths are supplied",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="changed paths for focused mode; ignored by quick/full",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.shards < 1:
        print("--shards must be >= 1", file=sys.stderr)
        return 2
    if args.shards > MAX_SHARDS:
        print(f"--shards must be <= {MAX_SHARDS}", file=sys.stderr)
        return 2
    py = sys.executable
    if args.paths and args.mode != "focused":
        print("paths are only accepted with --mode focused", file=sys.stderr)
        return 2
    if args.mode == "focused":
        rc = focused(py, args.paths, base=args.base)
    elif args.mode == "quick":
        rc = quick(py)
    else:
        rc = full(py, args.shards, retry_failed_once=not args.no_retry_failed)
    if rc:
        return rc
    if not args.skip_snapshot:
        return snapshot_smoke(py)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
