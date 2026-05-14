"""Run a stable shard of a pytest collection.

This keeps verification strength the same: pytest still collects the normal
tests, then this plugin selects the shard whose node-id hash maps to
`--index`. If historical duration weights are supplied, it assigns collected
tests greedily by runtime while preserving one-shard-per-test coverage.
Use it to split large files such as `test_codegen.py` across parallel workers
without adding a pytest dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import pytest


NODE_DURATION_SCHEMA = "helix.pytest_node_durations.v0"


def stable_bucket(nodeid: str, total: int) -> int:
    digest = hashlib.sha256(nodeid.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % total


def load_duration_weights(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    weights: dict[str, float] = {}
    raw_tests: Any = payload.get("tests", {})
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
    for nodeid, seconds in iterator:
        if not isinstance(nodeid, str):
            continue
        if not isinstance(seconds, (int, float)) or seconds <= 0:
            continue
        weights[nodeid] = float(seconds)
    return weights


def assign_nodeids_to_buckets(
    nodeids: list[str], total: int, weights: dict[str, float],
) -> dict[str, int]:
    if not weights:
        return {nodeid: stable_bucket(nodeid, total) for nodeid in nodeids}
    loads = [0.0 for _ in range(total)]
    assignment: dict[str, int] = {}
    ordered = sorted(
        nodeids,
        key=lambda nodeid: (
            -weights.get(nodeid, 1.0),
            stable_bucket(nodeid, total),
            nodeid,
        ),
    )
    for nodeid in ordered:
        bucket = min(range(total), key=lambda idx: (loads[idx], idx))
        assignment[nodeid] = bucket
        loads[bucket] += weights.get(nodeid, 1.0)
    return assignment


def write_node_durations(path: str | None, durations: dict[str, float]) -> None:
    if not path:
        return
    payload = {
        "schema": NODE_DURATION_SCHEMA,
        "generated_at_unix": time.time(),
        "tests": [
            {"nodeid": nodeid, "seconds": seconds}
            for nodeid, seconds in sorted(durations.items())
        ],
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class ShardPlugin:
    def __init__(
        self, total: int, index: int, *,
        weights: dict[str, float] | None = None,
        durations_out: str | None = None,
    ) -> None:
        self.total = total
        self.index = index
        self.weights = weights or {}
        self.durations_out = durations_out
        self.durations: dict[str, float] = {}

    def pytest_collection_modifyitems(self, config, items):  # type: ignore[no-untyped-def]
        selected = []
        deselected = []
        assignment = assign_nodeids_to_buckets(
            [item.nodeid for item in items],
            self.total,
            self.weights,
        )
        for item in items:
            bucket = assignment[item.nodeid]
            if bucket == self.index:
                selected.append(item)
            else:
                deselected.append(item)
        if deselected:
            config.hook.pytest_deselected(items=deselected)
        items[:] = selected

    def pytest_runtest_logreport(self, report):  # type: ignore[no-untyped-def]
        if report.when == "call":
            self.durations[report.nodeid] = float(report.duration)

    def pytest_sessionfinish(self, session, exitstatus):  # type: ignore[no-untyped-def]
        write_node_durations(self.durations_out, self.durations)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument(
        "--weights",
        help="optional prior node-duration JSON for weighted shard assignment",
    )
    parser.add_argument(
        "--durations-out",
        help="optional path to write node-duration JSON for this shard",
    )
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.total < 1:
        parser.error("--total must be >= 1")
    if args.index < 0 or args.index >= args.total:
        parser.error("--index must satisfy 0 <= index < total")
    if args.pytest_args and args.pytest_args[0] == "--":
        args.pytest_args = args.pytest_args[1:]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    pytest_args = ["-q", "-p", "no:cacheprovider", *args.pytest_args]
    weights = load_duration_weights(args.weights)
    return pytest.main(
        pytest_args,
        plugins=[
            ShardPlugin(
                args.total,
                args.index,
                weights=weights,
                durations_out=args.durations_out,
            )
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
