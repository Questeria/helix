"""Compile Helix proof obligations and enforce the clean proof-artifact gate."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helixc.check import main as check_main, parse_args as parse_check_args
from scripts.proof_artifact_validate import clean_policy_errors, validate_artifact


DISALLOWED_PROOF_GATE_FLAGS = {
    "--doc",
    "--emit-asm",
    "--emit-ast",
    "--emit-ir",
    "--emit-ptx",
    "--hash",
    "--hash-cons",
    "--help",
    "-h",
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    compiler_args: list[str] = []
    gate_argv = argv
    if "--" in argv:
        split_at = argv.index("--")
        gate_argv = argv[:split_at]
        compiler_args = argv[split_at + 1:]

    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Pass extra helixc.check flags after '--', for example: "
            "proof_artifact_gate.py input.hx -- --no-stdlib --strict"
        ),
    )
    parser.add_argument("source", help="Helix source file to compile and prove")
    parser.add_argument(
        "--artifact-out",
        help="optional path to write the emitted proof artifact JSON",
    )
    args = parser.parse_args(gate_argv)
    args.compiler_args = compiler_args
    return args


def _write_artifact(path: Path, raw_json: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw_json, encoding="utf-8")


def _same_path(left: Path, right: Path) -> bool:
    if left == right:
        return True
    try:
        return left.samefile(right)
    except OSError:
        return False


def _load_stdout_artifact(raw_json: str) -> dict[str, object]:
    artifact = json.loads(raw_json)
    if not isinstance(artifact, dict):
        raise ValueError("proof artifact must be a JSON object")
    return artifact


def _validate_compiler_args(source: Path, compiler_args: list[str]) -> list[str]:
    parsed, errors = parse_check_args([
        str(source),
        "--emit-proof-obligations",
        *compiler_args,
    ])
    if parsed.output is not None:
        errors.append("-o is not allowed in proof_artifact_gate")
    disallowed = sorted(parsed.flags & DISALLOWED_PROOF_GATE_FLAGS)
    if disallowed:
        errors.append(
            "output/debug flags are not allowed in proof_artifact_gate: "
            + ", ".join(disallowed)
        )
    return errors


def _expected_artifact(source: Path, compiler_args: list[str]) -> tuple[int, dict[str, object]]:
    out = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = check_main([
            str(source),
            "--emit-proof-obligations",
            *compiler_args,
        ])
    return rc, _load_stdout_artifact(out.getvalue())


def _obligation_completeness_errors(
    artifact: dict[str, object],
    *,
    source: Path,
    compiler_args: list[str],
) -> list[str]:
    _parsed, errors = parse_check_args([
        str(source),
        "--emit-proof-obligations",
        *compiler_args,
    ])
    if errors:
        return [f"could not parse proof args for completeness check: {errors}"]
    try:
        expected_rc, expected = _expected_artifact(source, compiler_args)
    except Exception as e:
        return [f"could not recompute proof artifact: {e}"]

    errors = []
    for field in (
        "schema",
        "cache_key",
        "path",
        "input",
        "summary",
        "obligations",
        "pipeline_errors",
        "typecheck_errors",
        "warning_diagnostics",
    ):
        if artifact.get(field) != expected.get(field):
            errors.append(f"proof artifact {field} mismatch against recomputed artifact")
    if expected_rc != 0:
        errors.append(
            f"recomputed proof run exited {expected_rc}, expected a clean proof run"
        )
    return errors


def run_gate(args: argparse.Namespace) -> int:
    source = Path(args.source).resolve()
    artifact_out = Path(args.artifact_out).resolve() if args.artifact_out else None
    for error in _validate_compiler_args(source, args.compiler_args):
        print(f"proof-artifact-gate: {error}", file=sys.stderr)
        return 2
    if artifact_out is not None and _same_path(artifact_out, source):
        print(
            "proof-artifact-gate: --artifact-out must not point to the source file",
            file=sys.stderr,
        )
        return 2

    cmd = [
        sys.executable,
        "-m",
        "helixc.check",
        str(source),
        "--emit-proof-obligations",
        *args.compiler_args,
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if artifact_out is not None:
        try:
            _write_artifact(artifact_out, proc.stdout)
        except OSError as e:
            print(f"proof-artifact-gate: could not write artifact: {e}", file=sys.stderr)
            return 2

    try:
        artifact = _load_stdout_artifact(proc.stdout)
    except Exception as e:
        print(f"proof-artifact-gate: invalid proof artifact JSON: {e}", file=sys.stderr)
        return 2

    artifact_dir = artifact_out.parent if artifact_out is not None else None
    source_for_validation = str(source) if source.is_file() else None
    structural_errors = validate_artifact(
        artifact,
        source_path=source_for_validation,
        artifact_dir=artifact_dir,
    )
    if structural_errors:
        for error in structural_errors:
            print(f"proof-artifact-gate: {error}", file=sys.stderr)
        return 2

    policy_errors = clean_policy_errors(artifact)
    completeness_errors: list[str] = []
    if not source.is_file():
        completeness_errors.append(
            "requested source path is not a readable file"
        )
    elif not policy_errors and proc.returncode == 0:
        completeness_errors = _obligation_completeness_errors(
            artifact,
            source=source,
            compiler_args=args.compiler_args,
        )

    for error in completeness_errors:
        print(f"proof-artifact-gate: {error}", file=sys.stderr)
    if policy_errors:
        for error in policy_errors:
            print(f"proof-artifact-gate: {error}", file=sys.stderr)

    if proc.returncode == 2:
        print("proof-artifact-gate: compiler exited 2", file=sys.stderr)
        return 2
    if completeness_errors:
        return 2
    if policy_errors:
        return 1
    if proc.returncode:
        print(
            f"proof-artifact-gate: compiler exited {proc.returncode}",
            file=sys.stderr,
        )
        return proc.returncode

    print("proof-artifact-gate: ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    return run_gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
