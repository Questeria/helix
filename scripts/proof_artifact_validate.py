"""Validate a Helix proof-obligation artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helixc.check import PROOF_SCHEMA
from scripts.proof_artifact_key import cache_key_for_artifact, load_artifact


SHA256_RE = re.compile(r"[0-9a-f]{64}")
OBLIGATION_STATUSES = {"proved", "failed", "unproven"}
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
    if errors:
        for error in errors:
            print(f"proof-artifact-validate: {error}", file=sys.stderr)
        return 1

    print("proof-artifact-validate: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
