"""Validate a Helix proof-obligation artifact."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helixc.check import PROOF_SCHEMA, main as check_main
from scripts.proof_artifact_key import cache_key_for_artifact, load_artifact


SHA256_RE = re.compile(r"[0-9a-f]{64}")
OBLIGATION_STATUSES = {"proved", "failed", "unproven", "unsupported"}
PROOF_CARRY_STRATEGIES = {
    "same-refinement",
    "exact-predicate-subset",
    "numeric-bound-implication",
}
SUMMARY_COUNTS = {
    "obligations": "obligations",
    "pipeline_errors": "pipeline_errors",
    "typecheck_errors": "typecheck_errors",
    "warning_diagnostics": "warning_diagnostics",
}
INPUT_COLORS = {"auto", "always", "never"}


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _is_json_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _list_field(artifact: dict[str, object], name: str, errors: list[str]) -> list[object]:
    value = artifact.get(name)
    if not isinstance(value, list):
        errors.append(f"{name} must be a list")
        return []
    return value


def _validate_string_list(
    value: object,
    *,
    name: str,
    errors: list[str],
) -> None:
    if not isinstance(value, list):
        errors.append(f"{name} must be a list")
        return
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            errors.append(f"{name}[{idx}] must be a string")


def _validate_input_metadata(
    input_metadata: dict[str, object],
    *,
    errors: list[str],
) -> None:
    required = [
        "source_sha256",
        "include_stdlib",
        "stdlib_strict",
        "stdlib_manifest_sha256",
        "stdlib_files",
        "opt_level",
        "flags",
        "libs",
        "warnings",
        "color",
    ]
    for field in required:
        if field not in input_metadata:
            errors.append(f"input.{field} is required")

    if not isinstance(input_metadata.get("include_stdlib"), bool):
        errors.append("input.include_stdlib must be a boolean")
    if not isinstance(input_metadata.get("stdlib_strict"), bool):
        errors.append("input.stdlib_strict must be a boolean")
    if not _is_sha256(input_metadata.get("stdlib_manifest_sha256")):
        errors.append(
            "input.stdlib_manifest_sha256 must be a lowercase SHA-256 hex digest"
        )
    if not _is_json_int(input_metadata.get("opt_level")):
        errors.append("input.opt_level must be an integer")
    if input_metadata.get("color") not in INPUT_COLORS:
        errors.append(f"input.color must be one of {sorted(INPUT_COLORS)}")

    _validate_string_list(input_metadata.get("flags"), name="input.flags", errors=errors)
    _validate_string_list(input_metadata.get("libs"), name="input.libs", errors=errors)

    warnings = input_metadata.get("warnings")
    if not isinstance(warnings, dict):
        errors.append("input.warnings must be an object")
    else:
        for key, value in warnings.items():
            if not isinstance(key, str):
                errors.append("input.warnings keys must be strings")
            if not isinstance(value, str):
                errors.append(f"input.warnings[{key!r}] must be a string")

    stdlib_files = input_metadata.get("stdlib_files")
    if not isinstance(stdlib_files, list):
        errors.append("input.stdlib_files must be a list")
    else:
        for idx, entry in enumerate(stdlib_files):
            if not isinstance(entry, dict):
                errors.append(f"input.stdlib_files[{idx}] must be an object")
                continue
            if not isinstance(entry.get("path"), str):
                errors.append(f"input.stdlib_files[{idx}].path must be a string")
            if entry.get("missing") is True:
                if "sha256" in entry:
                    errors.append(
                        f"input.stdlib_files[{idx}].sha256 must be absent "
                        f"when missing is true"
                    )
                if "bytes" in entry:
                    errors.append(
                        f"input.stdlib_files[{idx}].bytes must be absent "
                        f"when missing is true"
                    )
                continue
            if "missing" in entry and entry.get("missing") is not False:
                errors.append(f"input.stdlib_files[{idx}].missing must be boolean")
            if not _is_sha256(entry.get("sha256")):
                errors.append(
                    f"input.stdlib_files[{idx}].sha256 must be a lowercase "
                    f"SHA-256 hex digest"
                )
            if not _is_json_int(entry.get("bytes")):
                errors.append(f"input.stdlib_files[{idx}].bytes must be an integer")
        manifest_src = json.dumps(
            stdlib_files,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        expected_manifest = hashlib.sha256(manifest_src).hexdigest()
        if input_metadata.get("stdlib_manifest_sha256") != expected_manifest:
            errors.append(
                "input.stdlib_manifest_sha256 does not match input.stdlib_files"
            )

    if input_metadata.get("include_stdlib") is False and stdlib_files != []:
        errors.append("input.stdlib_files must be empty when include_stdlib is false")


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_artifact(
    artifact: dict[str, object],
    *,
    source_path: str | None = None,
    artifact_dir: str | Path | None = None,
) -> list[str]:
    errors: list[str] = []

    if artifact.get("schema") != PROOF_SCHEMA:
        errors.append(f"schema must be {PROOF_SCHEMA!r}")
    if "path" not in artifact:
        errors.append("path field is required")
    path_value = artifact.get("path")
    if path_value is not None and not isinstance(path_value, str):
        errors.append("path must be a string or null")

    input_metadata = artifact.get("input")
    if not isinstance(input_metadata, dict):
        errors.append("input must be an object")
        input_metadata = {}
    else:
        _validate_input_metadata(input_metadata, errors=errors)

    expected_key: str | None = None
    try:
        expected_key = cache_key_for_artifact(artifact)
    except Exception as e:
        errors.append(str(e))
    if "cache_key" not in artifact:
        errors.append("cache_key field is required")
    observed_key = artifact.get("cache_key")
    if observed_key != expected_key:
        errors.append(
            f"cache_key mismatch: artifact={observed_key!r} expected={expected_key!r}"
        )

    source_hash = input_metadata.get("source_sha256")
    if source_hash is None:
        if observed_key is not None:
            errors.append("cache_key must be null when source_sha256 is null")
        if input_metadata.get("source_available") is not False:
            errors.append("source_available must be false when source_sha256 is null")
        if not isinstance(input_metadata.get("source_error"), str):
            errors.append("input.source_error must be a string when source_sha256 is null")
    elif not _is_sha256(source_hash):
        errors.append("input.source_sha256 must be a lowercase SHA-256 hex digest")

    summary = artifact.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        summary = {}

    lists = {
        field_name: _list_field(artifact, field_name, errors)
        for field_name in SUMMARY_COUNTS.values()
    }
    if "proof_carries" not in artifact:
        errors.append("proof_carries field is required")
        proof_carries_value = []
    else:
        proof_carries_value = artifact.get("proof_carries")
    if "proof_carries" in artifact and not isinstance(proof_carries_value, list):
        errors.append("proof_carries must be a list")
        proof_carries = []
    else:
        proof_carries = list(proof_carries_value) \
            if isinstance(proof_carries_value, list) else []
    lists["proof_carries"] = proof_carries
    for summary_name, field_name in SUMMARY_COUNTS.items():
        observed_count = summary.get(summary_name)
        expected_count = len(lists[field_name])
        if not _is_json_int(observed_count):
            errors.append(f"summary.{summary_name} must be an integer")
        elif observed_count != expected_count:
            errors.append(
                f"summary.{summary_name}={observed_count!r} "
                f"but {field_name} has {expected_count} entries"
            )

    observed_carries = summary.get("proof_carries")
    if observed_carries is not None:
        if not _is_json_int(observed_carries):
            errors.append("summary.proof_carries must be an integer")
        elif observed_carries != len(proof_carries):
            errors.append(
                f"summary.proof_carries={observed_carries!r} "
                f"but proof_carries has {len(proof_carries)} entries"
            )
    else:
        errors.append("summary.proof_carries must be an integer")

    expected_strategies: dict[str, int] = {}
    for carry in proof_carries:
        if isinstance(carry, dict):
            strategy = carry.get("strategy")
            if isinstance(strategy, str):
                expected_strategies[strategy] = (
                    expected_strategies.get(strategy, 0) + 1
                )
    observed_strategies = summary.get("proof_carry_strategies")
    if observed_strategies is not None:
        if not isinstance(observed_strategies, dict):
            errors.append("summary.proof_carry_strategies must be an object")
        else:
            normalized: dict[str, int] = {}
            for key, value in observed_strategies.items():
                if not isinstance(key, str):
                    errors.append(
                        "summary.proof_carry_strategies keys must be strings"
                    )
                    continue
                if not _is_json_int(value):
                    errors.append(
                        f"summary.proof_carry_strategies.{key} "
                        "must be an integer"
                    )
                    continue
                normalized[key] = value
            if normalized != expected_strategies:
                errors.append(
                    "summary.proof_carry_strategies does not match "
                    "proof_carries"
                )
    else:
        errors.append("summary.proof_carry_strategies must be an object")

    warning_errors = sum(
        1 for warning in lists["warning_diagnostics"]
        if isinstance(warning, dict) and warning.get("promoted_to_error")
    )
    observed_warning_errors = summary.get("warning_errors")
    if not _is_json_int(observed_warning_errors):
        errors.append("summary.warning_errors must be an integer")
    elif observed_warning_errors != warning_errors:
        errors.append(
            f"summary.warning_errors={observed_warning_errors!r} "
            f"but warning_diagnostics has {warning_errors} promoted errors"
        )

    for idx, obligation in enumerate(lists["obligations"]):
        if not isinstance(obligation, dict):
            errors.append(f"obligations[{idx}] must be an object")
            continue
        for field in ("kind", "context", "refinement", "predicate", "status"):
            if not isinstance(obligation.get(field), str):
                errors.append(f"obligations[{idx}].{field} must be a string")
        if obligation.get("status") not in OBLIGATION_STATUSES:
            errors.append(
                f"obligations[{idx}].status must be one of "
                f"{sorted(OBLIGATION_STATUSES)}"
            )
        span = obligation.get("span")
        if not isinstance(span, dict):
            errors.append(f"obligations[{idx}].span must be an object")
        else:
            if not _is_json_int(span.get("line")):
                errors.append(f"obligations[{idx}].span.line must be an integer")
            if not _is_json_int(span.get("col")):
                errors.append(f"obligations[{idx}].span.col must be an integer")

    for idx, carry in enumerate(lists["proof_carries"]):
        if not isinstance(carry, dict):
            errors.append(f"proof_carries[{idx}] must be an object")
            continue
        for field in (
            "kind",
            "context",
            "source_refinement",
            "target_refinement",
            "strategy",
        ):
            if not isinstance(carry.get(field), str):
                errors.append(f"proof_carries[{idx}].{field} must be a string")
        if carry.get("kind") != "refinement-proof-carry":
            errors.append(
                f"proof_carries[{idx}].kind must be 'refinement-proof-carry'"
            )
        if carry.get("strategy") not in PROOF_CARRY_STRATEGIES:
            errors.append(
                f"proof_carries[{idx}].strategy must be one of "
                f"{sorted(PROOF_CARRY_STRATEGIES)}"
            )
        span = carry.get("span")
        if not isinstance(span, dict):
            errors.append(f"proof_carries[{idx}].span must be an object")
        else:
            if not _is_json_int(span.get("line")):
                errors.append(
                    f"proof_carries[{idx}].span.line must be an integer"
                )
            if not _is_json_int(span.get("col")):
                errors.append(
                    f"proof_carries[{idx}].span.col must be an integer"
                )

    for idx, pipeline_error in enumerate(lists["pipeline_errors"]):
        if not isinstance(pipeline_error, dict):
            errors.append(f"pipeline_errors[{idx}] must be an object")
            continue
        if not isinstance(pipeline_error.get("phase"), str):
            errors.append(f"pipeline_errors[{idx}].phase must be a string")
        if not isinstance(pipeline_error.get("message"), str):
            errors.append(f"pipeline_errors[{idx}].message must be a string")

    for idx, typecheck_error in enumerate(lists["typecheck_errors"]):
        if not isinstance(typecheck_error, str):
            errors.append(f"typecheck_errors[{idx}] must be a string")

    for idx, warning in enumerate(lists["warning_diagnostics"]):
        if not isinstance(warning, dict):
            errors.append(f"warning_diagnostics[{idx}] must be an object")
            continue
        for field in ("kind", "policy", "message"):
            if not isinstance(warning.get(field), str):
                errors.append(f"warning_diagnostics[{idx}].{field} must be a string")
        if not isinstance(warning.get("promoted_to_error"), bool):
            errors.append(
                f"warning_diagnostics[{idx}].promoted_to_error must be a boolean"
            )

    if source_hash is None:
        if lists["obligations"]:
            errors.append(
                "obligations must be empty when input.source_sha256 is null"
            )
        if proof_carries:
            errors.append(
                "proof_carries must be empty when input.source_sha256 is null"
            )
        if lists["typecheck_errors"]:
            errors.append(
                "typecheck_errors must be empty when input.source_sha256 is null"
            )

    source_to_check = source_path
    if source_to_check is None and isinstance(path_value, str):
        artifact_source = Path(path_value)
        candidates: list[Path] = []
        if artifact_source.is_absolute():
            candidates.append(artifact_source)
        else:
            if artifact_dir is not None:
                candidates.append(Path(artifact_dir) / artifact_source)
            candidates.append(artifact_source)
        for candidate in candidates:
            if candidate.is_file():
                source_to_check = str(candidate)
                break
        if source_to_check is None and source_hash is not None:
            errors.append(
                "source path from artifact could not be resolved: "
                + ", ".join(str(candidate) for candidate in candidates)
            )
    elif source_to_check is None and source_hash is not None:
        errors.append(
            "source path is required to verify artifacts with source_sha256"
        )

    if source_hash is not None and not isinstance(path_value, str):
        errors.append("path must be a string when input.source_sha256 is present")

    if source_path is not None and isinstance(path_value, str):
        artifact_source = Path(path_value)
        if not artifact_source.is_absolute() and artifact_dir is not None:
            artifact_source = Path(artifact_dir) / artifact_source
        if artifact_source.resolve() != Path(source_path).resolve():
            errors.append("proof artifact path mismatch against provided source")
    elif source_path is not None:
        errors.append("proof artifact path mismatch against provided source")

    if source_to_check is not None:
        if source_hash is None:
            errors.append("cannot verify source path when input.source_sha256 is null")
        else:
            try:
                actual = _source_sha256(Path(source_to_check))
            except OSError as e:
                errors.append(f"source could not be read: {e}")
            else:
                if actual != source_hash:
                    errors.append(
                        f"source sha256 mismatch: artifact={source_hash!r} "
                        f"actual={actual!r}"
                    )

    return errors


def clean_policy_errors(artifact: dict[str, object]) -> list[str]:
    """Return proof-gate failures for a structurally valid artifact."""
    errors: list[str] = []

    input_metadata = artifact.get("input")
    if isinstance(input_metadata, dict) and input_metadata.get("source_sha256") is None:
        errors.append("input.source_sha256 is required for --require-clean")

    obligations = artifact.get("obligations")
    if isinstance(obligations, list):
        for idx, obligation in enumerate(obligations):
            if not isinstance(obligation, dict):
                continue
            status = obligation.get("status")
            if status != "proved":
                refinement = obligation.get("refinement", "<unknown>")
                errors.append(
                    f"obligations[{idx}] for {refinement!r} is {status!r}, "
                    "not 'proved'"
                )

    pipeline_errors = artifact.get("pipeline_errors")
    if isinstance(pipeline_errors, list) and pipeline_errors:
        errors.append(f"pipeline_errors must be empty, found {len(pipeline_errors)}")

    typecheck_errors = artifact.get("typecheck_errors")
    if isinstance(typecheck_errors, list) and typecheck_errors:
        errors.append(f"typecheck_errors must be empty, found {len(typecheck_errors)}")

    warning_diagnostics = artifact.get("warning_diagnostics")
    if isinstance(warning_diagnostics, list):
        promoted = [
            warning for warning in warning_diagnostics
            if isinstance(warning, dict) and warning.get("promoted_to_error")
        ]
        if promoted:
            errors.append(
                "warning_diagnostics must not contain promoted errors, "
                f"found {len(promoted)}"
            )

    return errors


def _proof_args_from_artifact(
    artifact: dict[str, object],
    source_path: Path,
) -> tuple[list[str], list[str]]:
    input_metadata = artifact.get("input")
    if not isinstance(input_metadata, dict):
        return [], ["input must be an object"]

    errors: list[str] = []
    args: list[str] = [str(source_path)]

    opt_level = input_metadata.get("opt_level")
    if _is_json_int(opt_level):
        if opt_level != 1:
            args.append(f"-O{opt_level}")
    else:
        errors.append("input.opt_level must be an integer")

    flags = input_metadata.get("flags")
    if isinstance(flags, list) and all(isinstance(flag, str) for flag in flags):
        args.extend(flags)
        if "--emit-proof-obligations" not in flags:
            args.append("--emit-proof-obligations")
    else:
        errors.append("input.flags must be a list of strings")

    libs = input_metadata.get("libs")
    if isinstance(libs, list) and all(isinstance(lib, str) for lib in libs):
        for lib in libs:
            args.extend(["-l", lib])
    else:
        errors.append("input.libs must be a list of strings")

    warnings = input_metadata.get("warnings")
    if isinstance(warnings, dict):
        for key in sorted(warnings):
            value = warnings[key]
            if isinstance(key, str) and isinstance(value, str):
                args.append(f"-W{key}={value}")
            else:
                errors.append("input.warnings must map strings to strings")
    else:
        errors.append("input.warnings must be an object")

    color = input_metadata.get("color")
    if color == "always":
        args.append("--color")
    elif color == "never":
        args.append("--no-color")
    elif color != "auto":
        errors.append(f"input.color must be one of {sorted(INPUT_COLORS)}")

    return args, errors


def recomputed_clean_errors(
    artifact: dict[str, object],
    *,
    source_path: str | None,
) -> list[str]:
    if source_path is None:
        return [
            "--source is required with --require-clean so the proof artifact "
            "can be recomputed"
        ]

    source = Path(source_path)
    if not source.is_file():
        return [f"source path is not a readable file: {source}"]

    args, arg_errors = _proof_args_from_artifact(artifact, source)
    if arg_errors:
        return arg_errors

    out = io.StringIO()
    err = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = check_main(args)
        recomputed = json.loads(out.getvalue())
    except Exception as e:
        return [f"could not recompute proof artifact from source: {e}"]
    if not isinstance(recomputed, dict):
        return ["recomputed proof artifact must be a JSON object"]

    errors: list[str] = []
    for field in (
        "schema",
        "cache_key",
        "path",
        "input",
        "summary",
        "obligations",
        "proof_carries",
        "pipeline_errors",
        "typecheck_errors",
        "warning_diagnostics",
    ):
        if artifact.get(field) != recomputed.get(field):
            errors.append(
                f"proof artifact {field} mismatch against recomputed source"
            )
    if rc != 0:
        errors.append(
            f"recomputed proof run exited {rc}, expected a clean proof run"
        )
    return errors


def recomputed_source_artifact_errors(
    artifact: dict[str, object],
    *,
    source_path: str | None,
) -> list[str]:
    if source_path is None:
        return []

    source = Path(source_path)
    if not source.is_file():
        return [f"source path is not a readable file: {source}"]

    args, arg_errors = _proof_args_from_artifact(artifact, source)
    if arg_errors:
        return arg_errors

    out = io.StringIO()
    err = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            check_main(args)
        recomputed = json.loads(out.getvalue())
    except Exception as e:
        return [f"could not recompute proof artifact from source: {e}"]
    if not isinstance(recomputed, dict):
        return ["recomputed proof artifact must be a JSON object"]

    errors: list[str] = []
    for field in (
        "summary",
        "obligations",
        "proof_carries",
        "pipeline_errors",
        "typecheck_errors",
        "warning_diagnostics",
    ):
        if artifact.get(field) != recomputed.get(field):
            errors.append(
                f"proof artifact {field} mismatch against recomputed source"
            )
    return errors


def source_path_for_metadata_recompute(
    artifact: dict[str, object],
    *,
    explicit_source: str | None,
    artifact_dir: str | Path | None = None,
) -> str | None:
    if explicit_source is not None:
        return explicit_source
    input_metadata = artifact.get("input")
    if (not isinstance(input_metadata, dict)
            or input_metadata.get("source_sha256") is None):
        return None
    path_value = artifact.get("path")
    if not isinstance(path_value, str):
        return None
    artifact_source = Path(path_value)
    candidates: list[Path] = []
    if artifact_source.is_absolute():
        candidates.append(artifact_source)
    else:
        if artifact_dir is not None:
            candidates.append(Path(artifact_dir) / artifact_source)
        candidates.append(artifact_source)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact",
        nargs="?",
        default="-",
        help="proof artifact JSON path, or '-' for stdin",
    )
    parser.add_argument(
        "--source",
        help="optional source path whose SHA-256 must match artifact input",
    )
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help=(
            "also require no pipeline/typecheck errors, no promoted warnings, "
            "and every proof obligation to be proved"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        artifact = load_artifact(args.artifact)
    except Exception as e:
        print(f"proof-artifact-validate: {e}", file=sys.stderr)
        return 2

    artifact_dir = None
    if args.artifact != "-":
        artifact_dir = Path(args.artifact).resolve().parent
    errors = validate_artifact(
        artifact,
        source_path=args.source,
        artifact_dir=artifact_dir,
    )
    metadata_source = source_path_for_metadata_recompute(
        artifact,
        explicit_source=args.source,
        artifact_dir=artifact_dir,
    )
    if not errors and metadata_source is not None:
        errors.extend(recomputed_source_artifact_errors(
            artifact,
            source_path=metadata_source,
        ))
    if args.require_clean:
        errors.extend(clean_policy_errors(artifact))
        errors.extend(recomputed_clean_errors(
            artifact,
            source_path=args.source,
        ))
    if errors:
        for error in errors:
            print(f"proof-artifact-validate: {error}", file=sys.stderr)
        return 1

    print("proof-artifact-validate: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
