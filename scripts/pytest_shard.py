"""Run a stable hash shard of a pytest collection.

This keeps verification strength the same: pytest still collects the normal
tests, then this plugin selects the shard whose node-id hash maps to
`--index`. Use it to split large files such as `test_codegen.py` across
parallel workers without adding a pytest dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import sys

import pytest


class ShardPlugin:
    def __init__(self, total: int, index: int) -> None:
        self.total = total
        self.index = index

    def pytest_collection_modifyitems(self, config, items):  # type: ignore[no-untyped-def]
        selected = []
        deselected = []
        for item in items:
            digest = hashlib.sha256(item.nodeid.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big") % self.total
            if bucket == self.index:
                selected.append(item)
            else:
                deselected.append(item)
        if deselected:
            config.hook.pytest_deselected(items=deselected)
        items[:] = selected


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--index", type=int, required=True)
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
    return pytest.main(
        pytest_args,
        plugins=[ShardPlugin(args.total, args.index)],
    )


if __name__ == "__main__":
    raise SystemExit(main())
