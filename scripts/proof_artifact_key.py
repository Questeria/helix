"""Compute or verify a Helix proof artifact cache key."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helixc.check import PROOF_SCHEMA, proof_cache_key


def decode_artifact_bytes(raw_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw_bytes.decode("utf-8")


def load_artifact(path: str) -> dict[str, object]:
    if path == "-":
        if isinstance(sys.stdin, io.TextIOBase):
            raw_bytes = sys.stdin.buffer.read()
        else:
            raw = sys.stdin.read()
            raw_bytes = raw if isinstance(raw, bytes) else raw.encode("utf-8")
        raw = decode_artifact_bytes(raw_bytes)
    else:
        raw_bytes = Path(path).read_bytes()
        raw = decode_artifact_bytes(raw_bytes)
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("artifact must be a JSON object")
    return data


def cache_key_for_artifact(artifact: dict[str, object]) -> str | None:
    schema = artifact.get("schema")
    if not isinstance(schema, str):
        raise ValueError("artifact is missing string field: schema")
    if schema != PROOF_SCHEMA:
        raise ValueError(f"unsupported proof schema: {schema}")
    input_metadata = artifact.get("input")
    if not isinstance(input_metadata, dict):
        raise ValueError("artifact is missing object field: input")
    return proof_cache_key(input_metadata, schema=schema)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact",
        nargs="?",
        default="-",
        help="proof artifact JSON path, or '-' for stdin",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify artifact.cache_key matches the recomputed key",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        artifact = load_artifact(args.artifact)
        key = cache_key_for_artifact(artifact)
    except Exception as e:
        print(f"proof-artifact-key: {e}", file=sys.stderr)
        return 2

    if args.check:
        observed = artifact.get("cache_key")
        if observed != key:
            print(
                f"proof-artifact-key: cache_key mismatch: "
                f"artifact={observed!r} expected={key!r}",
                file=sys.stderr,
            )
            return 1
    print(key if key is not None else "null")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
