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
"in_progress" / "planned" to "released" (and open the next one). As
each v3.0 build stage closes its 3-part audit, bump `V3_STAGES_DONE`.
Every percentage recomputes from that edit; the test-suite size is
counted LIVE from `helixc/tests/` (so it grows with every chunk and
never goes stale — no manual bump).

Usage:
    python scripts/helix_status.py
    python scripts/helix_status.py --note "<plain-English summary>" \\
        --commit <hash>

License: Apache 2.0
"""
from __future__ import annotations

import argparse
from pathlib import Path


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
    {"id": "v3.0", "status": "released",
     "theme": "The big rewrite - industrial MLIR + LLVM backend"},
    {"id": "v3.1", "status": "released",
     "theme": "Post-v3.0 cleanup - LLVM toolchain wiring, polymorphic "
              "SPLICE/MODIFY, REFLECT_HASH, shared-constants module"},
    {"id": "v3.2", "status": "planned",
     "theme": "Real-execution parity gate (or first K-bootstrap "
              "milestone toward Helix-in-Helix)"},
]

# v2.x shipped its compiler work as 22 numbered build stages
# (Stage 110-131), all closed — the v2.0-v2.5 entries in VERSIONS
# record that. v3.0 is built as its own 19 numbered stages: Phase D
# (Stage 200-208), Phase E (210-216), Phase F (220-222). Every stage
# closes with a 3-part audit. Bump `V3_STAGES_DONE` as each closes —
# every percentage below recomputes from it.
V3_STAGES_TOTAL = 19
V3_STAGES_DONE = 19       # ALL Phase D + E + F stages COMPLETE — v3.0 RELEASED

# K-bootstrap track (post v3.1.0, declared the new top-line goal
# 2026-05-25). See docs/HELIX_K_BOOTSTRAP_MASTER_PLAN.md and the
# feature-parity matrix docs/K_BOOTSTRAP_FEATURE_MATRIX.md. The
# matrix enumerates every Helix language feature with a column for
# Python helixc support and a column for kovc.hx support. A row is
# PARITY when both columns agree; KOVC-MISSING when only Python
# supports it. The goal: get every row to PARITY, then delete the
# Python compiler.
#
# Bump K_BOOTSTRAP_PARITY_DONE as each K-track chunk lands and the
# matrix's PARITY count rises.
K_BOOTSTRAP_TOTAL_ROWS = 143      # matrix total (28 PARITY + 115
                                    # KOVC-MISSING at K0 chunk 2 close)
K_BOOTSTRAP_PARITY_DONE = 35       # was 28 after K0; K1.B (stack
                                    # args > 6) made it 29; K1.C
                                    # (return statement) made it 30;
                                    # K1.D-impl (print_int) made it 31;
                                    # K1.G (for loop) made it 32;
                                    # K1.H1 (loop keyword) made it 33;
                                    # K1.F discovery (tuple lit +
                                    # field access were already in
                                    # kovc.hx, matrix audit had
                                    # marked them stale-MISSING) +2
                                    # made it 35

# The version statuses the model recognises.
_VALID_STATUS = frozenset({"released", "in_progress", "planned"})


def v3_stages_percent() -> int:
    """Percent of the v3.0 build stages complete (each 3-clean
    audited)."""
    return round(100 * V3_STAGES_DONE / V3_STAGES_TOTAL)


def versions_percent() -> int:
    """Percent of journey versions fully released (audit gate passed)."""
    released = sum(1 for v in VERSIONS if v["status"] == "released")
    return round(100 * released / len(VERSIONS))


def _version_credit(v: dict[str, str]) -> float:
    """How much one version contributes toward the overall journey
    total: a released version counts 1.0, a planned version 0.0, and
    an in-progress version gets partial credit. For v3.0 specifically
    (the only version with a published numbered-stage breakdown) we
    use the live V3_STAGES_DONE fraction so partial credit climbs as
    stages close. For other in-progress versions (v3.1 cleanup, v3.2
    parity gate, future K-bootstrap milestones) there is no
    fine-grained stage table — they tick from 0% to 100% at release.
    A reasonable middle-credit (0.5) keeps the overall percentage
    honest without inventing a fake-precision stage count."""
    if v["status"] == "released":
        return 1.0
    if v["status"] == "planned":
        return 0.0
    if v["id"] == "v3.0":
        return V3_STAGES_DONE / V3_STAGES_TOTAL
    return 0.5


def overall_percent() -> int:
    """Overall progress along the v2.0 -> v3.0 journey — the released
    versions plus the in-progress version's live v3.0-stage
    fraction."""
    score = sum(_version_credit(v) for v in VERSIONS)
    return round(100 * score / len(VERSIONS))


def k_bootstrap_percent() -> int:
    """Percent of Helix-in-Helix self-hosting feature-parity reached.
    Computed live from the matrix counts; never hand-typed."""
    return round(100 * K_BOOTSTRAP_PARITY_DONE / K_BOOTSTRAP_TOTAL_ROWS)


def count_tests() -> int:
    """The size of the automated test suite — a count of `def test_*`
    definitions across `helixc/tests/`, computed LIVE so it grows with
    every chunk and never goes stale.

    A pure scale-of-testing figure for non-engineers, NOT a pass/fail
    claim: it counts the tests that EXIST, it does not run them (a
    live pass/fail readout would need a mode that runs pytest). Fails
    loudly rather than render a misleading zero."""
    tests_dir = (Path(__file__).resolve().parent.parent
                 / "helixc" / "tests")
    total = 0
    for path in tests_dir.glob("test_*.py"):
        total += sum(
            1 for line in path.read_text(encoding="utf-8").splitlines()
            if line.lstrip().startswith("def test_"))
    if total == 0:
        raise SystemExit(
            f"helix_status: counted 0 tests under {tests_dir} — the "
            f"test directory was not found or is empty; refusing to "
            f"render a misleading status.")
    return total


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
        "builds and runs it. The current top-line goal is "
        "SELF-HOSTING: get the Helix compiler written in Helix, "
        "compiled in Helix, all the way from raw binary -- no Python "
        "in the final product. We track two things: the released "
        'versions (grouped into "v2.0", "v3.0", and so on), and the '
        "feature-parity matrix that measures how close the Helix-"
        "side compiler is to the Python compiler. Every version "
        "ends with a thorough multi-part code audit before it counts "
        "as done.",
        "",
        "RELEASED VERSIONS",
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
        "SELF-HOSTING PROGRESS (Helix-in-Helix)",
        f"  - Feature-parity rows: {K_BOOTSTRAP_PARITY_DONE} / "
        f"{K_BOOTSTRAP_TOTAL_ROWS} done   "
        f"({k_bootstrap_percent()}%)",
        "    Each row is a Helix language feature; PARITY means the "
        "Helix-",
        "    side compiler handles it the same way the Python compiler "
        "does.",
        "    Track plan: K0 survey -> K1 ports -> K2 parity harness ->",
        "    K3 trusted seed -> K4 delete Python (gated) -> K5 final "
        "audits.",
        "",
        "PROGRESS",
        f"  - v3.0 build stages:   {V3_STAGES_DONE} / "
        f"{V3_STAGES_TOTAL} done   ({v3_stages_percent()}%) - each "
        f"one 3-part audited",
        f"  - Versions released:   {len(released)} / {len(VERSIONS)}"
        f"         ({versions_percent()}%)",
        f"  - Test coverage:       ~{count_tests()} automated tests "
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

    # Guard the single-source-of-truth model: a typo'd status or an
    # out-of-range stage count would silently skew every percentage.
    # Fail loudly instead.
    for v in VERSIONS:
        if v["status"] not in _VALID_STATUS:
            raise SystemExit(
                f"helix_status: VERSIONS entry {v['id']!r} has unknown "
                f"status {v['status']!r}; expected one of "
                f"{sorted(_VALID_STATUS)}.")
    if not 0 <= V3_STAGES_DONE <= V3_STAGES_TOTAL:
        raise SystemExit(
            f"helix_status: V3_STAGES_DONE ({V3_STAGES_DONE}) must be "
            f"in 0..V3_STAGES_TOTAL ({V3_STAGES_TOTAL}).")
    if not 0 <= K_BOOTSTRAP_PARITY_DONE <= K_BOOTSTRAP_TOTAL_ROWS:
        raise SystemExit(
            f"helix_status: K_BOOTSTRAP_PARITY_DONE "
            f"({K_BOOTSTRAP_PARITY_DONE}) must be in "
            f"0..K_BOOTSTRAP_TOTAL_ROWS ({K_BOOTSTRAP_TOTAL_ROWS}).")

    print(render_telegram(note=args.note, commit=args.commit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
