"""Run and validate the Helix self-host cascade as one gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / ".stage33-logs" / "selfhost-cascade-latest.json"
DEFAULT_PREFIX = "/tmp/helix_cascade"


def run_command(cmd: list[str]) -> int:
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    return proc.returncode


def gate(
    *,
    generations: int,
    json_out: Path,
    prefix: str,
    keep: bool,
    expect_stable_sha: str | None,
) -> int:
    cascade_cmd = [
        sys.executable,
        "scripts/selfhost_cascade.py",
        "--generations",
        str(generations),
        "--prefix",
        prefix,
        "--json-out",
        str(json_out),
    ]
    if keep:
        cascade_cmd.append("--keep")
    rc = run_command(cascade_cmd)
    if rc:
        return rc

    validate_cmd = [
        sys.executable,
        "scripts/selfhost_cascade_validate.py",
        str(json_out),
        "--min-generations",
        str(generations),
    ]
    if expect_stable_sha:
        validate_cmd.extend(["--expect-stable-sha", expect_stable_sha])
    return run_command(validate_cmd)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--expect-stable-sha")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.generations < 2:
        print("--generations must be at least 2", file=sys.stderr)
        return 2
    return gate(
        generations=args.generations,
        json_out=args.json_out,
        prefix=args.prefix,
        keep=args.keep,
        expect_stable_sha=args.expect_stable_sha,
    )


if __name__ == "__main__":
    raise SystemExit(main())
