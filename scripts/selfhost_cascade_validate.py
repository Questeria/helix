"""Validate a machine-readable Helix self-host cascade report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.selfhost_cascade import CASCADE_REPORT_SCHEMA


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_SMOKE = {
    "literal": 42,
    "call": 42,
    "loop": 42,
    "metadata_attrs": 42,
}


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.match(value) is not None


def validate_report(
    report: dict[str, Any],
    *,
    min_generations: int = 2,
    expect_stable_sha: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != CASCADE_REPORT_SCHEMA:
        errors.append(f"schema must be {CASCADE_REPORT_SCHEMA}")

    generations = report.get("self_host_generations")
    if not isinstance(generations, list):
        errors.append("self_host_generations must be a list")
        generations = []
    elif len(generations) < min_generations:
        errors.append(
            f"self_host_generations must contain at least {min_generations} entries"
        )

    stable_sha = report.get("stable_sha256")
    stable_size = report.get("stable_size")
    if report.get("stable") is not True:
        errors.append("stable must be true")
    if not is_sha256(stable_sha):
        errors.append("stable_sha256 must be a lowercase SHA-256 hex digest")
    if not isinstance(stable_size, int) or stable_size <= 0:
        errors.append("stable_size must be a positive integer")
    if expect_stable_sha is not None and stable_sha != expect_stable_sha:
        errors.append("stable_sha256 does not match expected hash")

    for idx, entry in enumerate(generations):
        if not isinstance(entry, dict):
            errors.append(f"self_host_generations[{idx}] must be an object")
            continue
        if entry.get("sha256") != stable_sha:
            errors.append(f"self_host_generations[{idx}].sha256 does not match stable_sha256")
        if entry.get("size") != stable_size:
            errors.append(f"self_host_generations[{idx}].size does not match stable_size")
        exit_low_byte = entry.get("exit_low_byte")
        if isinstance(stable_size, int) and exit_low_byte != (stable_size & 0xFF):
            errors.append(
                f"self_host_generations[{idx}].exit_low_byte does not match stable_size low byte"
            )

    smoke = report.get("smoke")
    if not isinstance(smoke, list):
        errors.append("smoke must be a list")
        smoke = []
    smoke_by_name = {
        entry.get("name"): entry
        for entry in smoke
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }
    for name, expected in REQUIRED_SMOKE.items():
        entry = smoke_by_name.get(name)
        if entry is None:
            errors.append(f"smoke missing {name}")
            continue
        if entry.get("expected_exit") != expected:
            errors.append(f"smoke {name}.expected_exit must be {expected}")
        if entry.get("actual_exit") != expected:
            errors.append(f"smoke {name}.actual_exit must be {expected}")

    return errors


def load_report(path: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise ValueError("report must be a JSON object")
    return payload


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="cascade report JSON path")
    parser.add_argument("--min-generations", type=int, default=2)
    parser.add_argument("--expect-stable-sha")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.min_generations < 1:
        print("--min-generations must be >= 1", file=sys.stderr)
        return 2
    try:
        report = load_report(args.report)
    except ValueError as exc:
        print(f"selfhost-cascade-validate: {exc}", file=sys.stderr)
        return 1
    errors = validate_report(
        report,
        min_generations=args.min_generations,
        expect_stable_sha=args.expect_stable_sha,
    )
    if errors:
        for error in errors:
            print(f"selfhost-cascade-validate: {error}", file=sys.stderr)
        return 1
    print("selfhost-cascade-validate: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
