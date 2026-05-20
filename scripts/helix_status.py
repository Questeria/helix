#!/usr/bin/env python3
"""
scripts/helix_status.py — beginner-friendly Helix progress reporter.

The Helix autonomous build worker (the `helix-approach-a-loop`
scheduled task) sends a Telegram status update at the end of every
fire. Those updates used to be terse and developer-facing — e.g.
"Stage 117, commit abc1234, 21 tests pass" — unreadable to anyone
who is not a compiler engineer.

This module renders a plain-language update instead: what is finished
and audited, what is in progress, what is still ahead, and a
percent-progress readout for build stages, versions, and the project
overall.

It is the SINGLE SOURCE OF TRUTH for release-journey status. When a
version ships, change its `status` in `VERSIONS` below from
"in_progress" / "planned" to "released" (and open the next one) —
every percentage recomputes from that one edit. The constants
(`STAGES_DONE`, `TESTS_PASSING`) are likewise updated in place.

Usage:
    python scripts/helix_status.py
    python scripts/helix_status.py --note "<plain-English summary>" \\
        --commit <hash>

License: Apache 2.0
"""
from __future__ import annotations

import argparse


# --- The v2.0 -> v3.0 release journey --------------------------------
# Each Helix version ends with a 5-part "clean-gate" code audit before
# it counts as released. Statuses:
#   "released"    — shipped AND its end-of-version audit gate passed
#   "in_progress" — actively being built right now
#   "planned"     — scoped but not started
# Update `status` here (and ONLY here) as versions ship.
VERSIONS: list[dict[str, str]] = [
    {"id": "v2.0", "status": "released",
     "theme": "GPU compiler foundation (22 build stages)"},
    {"id": "v2.1", "status": "released",
     "theme": "Per-operation GPU code generation + autodiff"},
    {"id": "v2.2", "status": "released",
     "theme": "Polish and audit clean-up"},
    {"id": "v2.3", "status": "released",
     "theme": "Type-system design polish"},
    {"id": "v2.4", "status": "released",
     "theme": "Real-GPU testing + attestation + register allocator"},
    {"id": "v2.5", "status": "released",
     "theme": "Wiring the register allocator into real GPU kernels"},
    {"id": "v3.0", "status": "in_progress",
     "theme": "The big rewrite - industrial MLIR + LLVM backend"},
]

# v2.x shipped its compiler work as 22 numbered build stages
# (Stage 110 through Stage 131); every one closed with a 3-part audit
# (silent-failure + type-design + code-review). All 22 are done.
STAGES_TOTAL = 22
STAGES_DONE = 22

# Size of the automated test suite (`helixc/tests/`) — a
# scale-of-testing signal for non-engineers. Bump as the suite grows.
# Deliberately NOT a live pass/fail claim: this module renders stable
# facts, and a hardcoded "all passing" would read false during any
# transient regression. Live pass/fail belongs in a future mode that
# actually runs pytest.
TESTS_TOTAL = 4013

# Weight a version contributes toward the overall journey total.
_WEIGHT = {"released": 1.0, "in_progress": 0.5, "planned": 0.0}

_VALID_STATUS = frozenset(_WEIGHT)


def stages_percent() -> int:
    """Percent of the v2.x build stages complete (each 3-clean audited)."""
    return round(100 * STAGES_DONE / STAGES_TOTAL)


def versions_percent() -> int:
    """Percent of journey versions fully released (audit gate passed)."""
    released = sum(1 for v in VERSIONS if v["status"] == "released")
    return round(100 * released / len(VERSIONS))


def overall_percent() -> int:
    """Overall progress along the v2.0 -> v3.0 journey. A version that
    is in progress counts as half — honest partial credit, not a guess."""
    score = sum(_WEIGHT[v["status"]] for v in VERSIONS)
    return round(100 * score / len(VERSIONS))


def _bucket(status: str) -> list[dict[str, str]]:
    """Versions in a given status, in journey order."""
    return [v for v in VERSIONS if v["status"] == status]


def render_telegram(note: str | None = None,
                    commit: str | None = None) -> str:
    """Render the beginner-friendly Helix status update.

    `note`   — one plain-English sentence on what the latest fire did.
    `commit` — the short commit hash of that fire's commit.
    Both are optional; the per-fire footer is omitted if neither is set.
    """
    released = _bucket("released")
    in_progress = _bucket("in_progress")
    planned = _bucket("planned")

    lines: list[str] = [
        "HELIX COMPILER  -  BUILD UPDATE",
        "================================",
        "",
        "Helix is a new programming language and the compiler that "
        "builds and runs it. The work is split into small numbered "
        '"stages", which are grouped into "versions" (v2.0, v2.1, and '
        "so on). Every version must pass a thorough multi-part code "
        "audit before it counts as done.",
        "",
        "DONE & FULLY AUDITED",
    ]
    for v in released:
        lines.append(f"  - {v['id']}   {v['theme']}")

    if in_progress:
        lines += ["", "IN PROGRESS"]
        for v in in_progress:
            lines.append(f"  - {v['id']}   {v['theme']}")

    if planned:
        lines += ["", "STILL AHEAD"]
        for v in planned:
            lines.append(f"  - {v['id']}   {v['theme']}")

    lines += [
        "",
        "PROGRESS",
        f"  - Build stages (v2.x):  {STAGES_DONE} / {STAGES_TOTAL} "
        f"done  ({stages_percent()}%) - each one 3-part audited",
        f"  - Versions released:    {len(released)} / {len(VERSIONS)}"
        f"         ({versions_percent()}%)",
        f"  - Overall toward v3.0:  about {overall_percent()}%",
        f"  - Test coverage:        ~{TESTS_TOTAL} automated tests "
        f"guard the code",
    ]

    if note or commit:
        lines.append("")
        if note:
            lines.append(f"THIS UPDATE: {note}")
        if commit:
            lines.append(f"  commit {commit}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI: print the beginner-friendly Helix status update."""
    ap = argparse.ArgumentParser(
        description="Render the beginner-friendly Helix status update "
                    "(used for the autonomous worker's Telegram dispatch).")
    ap.add_argument("--note", default=None,
                    help="one plain-English sentence on what the latest "
                         "fire shipped")
    ap.add_argument("--commit", default=None,
                    help="short commit hash of the latest fire's commit")
    args = ap.parse_args(argv)

    # Guard the single-source-of-truth model: a typo'd status would
    # silently skew every percentage. Fail loudly instead.
    for v in VERSIONS:
        if v["status"] not in _VALID_STATUS:
            raise SystemExit(
                f"helix_status: VERSIONS entry {v['id']!r} has unknown "
                f"status {v['status']!r}; expected one of "
                f"{sorted(_VALID_STATUS)}.")

    print(render_telegram(note=args.note, commit=args.commit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
