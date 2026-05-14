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
LOG_DIR = ROOT / ".stage31-logs"
BIN_DIR = ROOT / ".stage31-bin"
MAX_SHARDS = 8
MAX_NO_CODEGEN_SHARDS = 4
PYTEST_SUMMARY_RE = re.compile(r"\bin (?P<seconds>[0-9]+(?:\.[0-9]+)?)s(?:\s|$)")
TIMING_SUMMARY_PATH = LOG_DIR / "pytest-shard-timings.json"


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
            "helixc/tests/test_typecheck.py::test_stage31_unsupported_refinement_predicates_do_not_carry_by_name",
            "helixc/tests/test_typecheck.py::test_stage31_generic_qualified_refinement_names_are_unsupported",
            "helixc/tests/test_typecheck.py::test_stage31_duplicate_refinement_names_fail_closed",
            "helixc/tests/test_typecheck.py::test_stage31_self_independent_false_refinement_rejects_unknown_values",
            "helixc/tests/test_typecheck.py::test_stage31_mixed_self_independent_refinements_do_not_downgrade",
            "helixc/tests/test_typecheck.py::test_stage31_boolean_short_circuit_refinements_are_decisive",
            "helixc/tests/test_typecheck.py::test_stage31_constant_comparison_refinement_predicates_are_supported",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_real_artifact_with_source_passes",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_rejects_malformed_diagnostic_sections",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_rejects_missing_embedded_source_path",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_resolves_relative_artifact_path_from_artifact_dir",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_rejects_boolean_integer_fields",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_rejects_bad_stdlib_manifest_hash",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_rejects_malformed_missing_stdlib_entry",
            "helixc/tests/test_proof_artifact_validate.py::test_validate_source_unavailable_rejects_proof_content",
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


def full(py: str, shards: int, *, retry_failed_once: bool = True) -> int:
    no_codegen_shards = no_codegen_shards_for(shards)
    print(f"full: no-codegen shards={no_codegen_shards} codegen shards={shards}")
    env = validation_env()
    jobs: list[tuple[str, list[str], dict[str, str] | None]] = []
    for index in range(no_codegen_shards):
        jobs.append(
            (
                f"pytest-no-codegen-shard-{index + 1}-of-{no_codegen_shards}",
                [
                    py,
                    "scripts/pytest_shard.py",
                    "--total",
                    str(no_codegen_shards),
                    "--index",
                    str(index),
                    "helixc/tests",
                    "--ignore=helixc/tests/test_codegen.py",
                ],
                env,
            )
        )
    for index in range(shards):
        jobs.append(
            (
                f"pytest-codegen-shard-{index + 1}-of-{shards}",
                [
                    py,
                    "scripts/pytest_shard.py",
                    "--total",
                    str(shards),
                    "--index",
                    str(index),
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
        choices=("quick", "full"),
        default="quick",
        help="quick runs recent regressions; full runs the broad suite",
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
    rc = (
        quick(py) if args.mode == "quick"
        else full(py, args.shards, retry_failed_once=not args.no_retry_failed)
    )
    if rc:
        return rc
    if not args.skip_snapshot:
        return snapshot_smoke(py)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
