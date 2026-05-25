"""Tests for scripts/helix_status.py — the beginner-friendly Helix
progress reporter used for the autonomous worker's Telegram updates.

The script is the single source of truth for release-journey status;
these tests pin that its percentages are computed (never hand-typed)
and that the rendered update names every section a non-engineer needs.
"""
from __future__ import annotations

import importlib.util
import os

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_HS_PATH = os.path.join(_REPO, "scripts", "helix_status.py")

# Import by explicit file path — no sys.path pollution, no collision
# with the other top-level modules in scripts/.
_spec = importlib.util.spec_from_file_location("helix_status", _HS_PATH)
assert _spec is not None and _spec.loader is not None
hs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hs)


def test_helix_status_version_model_is_consistent():
    """Every VERSIONS entry has exactly the three required fields and a
    known status. The model is the single source of truth, so a typo'd
    status would silently skew every percentage — pin it shut."""
    assert hs.VERSIONS, "VERSIONS must not be empty"
    for v in hs.VERSIONS:
        assert set(v) == {"id", "status", "theme"}, v
        assert v["status"] in {"released", "in_progress", "planned"}, v
        assert v["id"] and v["theme"]


def test_helix_status_percentages_are_computed_from_the_model():
    """The progress numbers are derived from VERSIONS / the v3.0 stage
    counts — never hand-typed — so they cannot drift from the model.

    v3.1.0 update: the credit model now has THREE cases for an
    in-progress version. v3.0 (the only version with a published
    numbered-stage breakdown) earns `V3_STAGES_DONE / V3_STAGES_TOTAL`.
    Other in-progress versions (v3.1 cleanup track, v3.2 parity gate,
    future K-bootstrap milestones) earn a flat 0.5 — there is no
    fine-grained stage table for them so a fake-precision number would
    be dishonest."""
    released = sum(1 for v in hs.VERSIONS if v["status"] == "released")
    total = len(hs.VERSIONS)
    assert hs.versions_percent() == round(100 * released / total)
    assert hs.v3_stages_percent() == round(
        100 * hs.V3_STAGES_DONE / hs.V3_STAGES_TOTAL)
    # Replicate the production credit model exactly.
    def _credit(v: dict[str, str]) -> float:
        if v["status"] == "released":
            return 1.0
        if v["status"] == "planned":
            return 0.0
        if v["id"] == "v3.0":
            return hs.V3_STAGES_DONE / hs.V3_STAGES_TOTAL
        return 0.5
    score = sum(_credit(v) for v in hs.VERSIONS)
    assert hs.overall_percent() == round(100 * score / total)
    # All three must be valid percentages.
    for p in (hs.v3_stages_percent(), hs.versions_percent(),
              hs.overall_percent()):
        assert 0 <= p <= 100


def test_helix_status_overall_tracks_v3_stage_progress():
    """The overall % MOVES with v3.0 stage progress — it is not frozen
    while v3.0 is in progress. (The bug this reporter was fixed for: a
    flat 0.5 in-progress weight pinned 'about 93%' constant for the
    whole of v3.0, so the Telegram update never reflected real
    progress.) Post v3.0.0 release, the in-progress weight no longer
    contributes — but the partial-credit semantics for an unfinished
    in-progress version must still be exercised, so this test pokes
    v3.0 back to "in_progress" for the duration of the assertions.

    v3.1.0 update: with v3.1 released and v3.2 planned, the journey
    is no longer "v3.0 is the last entry" — so the assertion
    `overall_percent() == 100` only holds if EVERY version is
    released. This test re-poses to compare ordering of base /
    upper / lower without anchoring on 100."""
    original_done = hs.V3_STAGES_DONE
    original_v3_status = next(
        v["status"] for v in hs.VERSIONS if v["id"] == "v3.0")
    v3_entry = next(v for v in hs.VERSIONS if v["id"] == "v3.0")
    try:
        # Restore the "v3.0 in_progress" world for the partial-credit
        # test to be meaningful — and pin V3_STAGES_DONE to a
        # middle value so `base` isn't already at the boundary.
        v3_entry["status"] = "in_progress"
        hs.V3_STAGES_DONE = hs.V3_STAGES_TOTAL // 2
        base = hs.overall_percent()
        hs.V3_STAGES_DONE = hs.V3_STAGES_TOTAL      # v3.0 fully done
        upper = hs.overall_percent()
        hs.V3_STAGES_DONE = 0                       # v3.0 not started
        lower = hs.overall_percent()
        # Strict ordering — overall tracks v3.0 progress monotonically.
        assert lower < base < upper, (lower, base, upper)
    finally:
        hs.V3_STAGES_DONE = original_done
        v3_entry["status"] = original_v3_status


def test_helix_status_counts_are_sane():
    """V3_STAGES_DONE never exceeds V3_STAGES_TOTAL; the test-suite
    size is a positive integer (a beginner-facing scale signal).
    Stage 222 5-clean-gate fix: `TESTS_TOTAL` was retired in commit
    1c827d8 (2026-05-20) in favour of the live `count_tests()`
    function — the test was latent-failing through 11 v3.0 stage
    closures until the v3.0.0-tag prep flushed the bookkeeping."""
    assert 0 <= hs.V3_STAGES_DONE <= hs.V3_STAGES_TOTAL
    count = hs.count_tests()
    assert isinstance(count, int) and count > 0


def test_helix_status_telegram_message_is_beginner_friendly():
    """The rendered update names every section a non-expert needs:
    what is done + audited, what is in progress (if any), what is
    left (if any), and the progress numbers — plus a plain-language
    explanation of the jargon (stages / versions). Each bucket
    section renders conditionally."""
    msg = hs.render_telegram()
    # Plain-language framing for a non-engineer.
    assert "programming language" in msg
    assert "stages" in msg and "versions" in msg
    # The status buckets all render conditionally.
    assert "DONE & FULLY AUDITED" in msg
    if any(v["status"] == "in_progress" for v in hs.VERSIONS):
        assert "IN PROGRESS" in msg
    if any(v["status"] == "planned" for v in hs.VERSIONS):
        assert "STILL AHEAD" in msg
    # The progress numbers the user asked for.
    assert "PROGRESS" in msg
    assert f"{hs.v3_stages_percent()}%" in msg
    assert f"{hs.versions_percent()}%" in msg
    assert f"{hs.overall_percent()}%" in msg
    # The first released and the final planned version both appear.
    assert "v2.0" in msg and "v3.0" in msg


def test_helix_status_telegram_includes_fire_detail_when_given():
    """`note` and `commit` are per-fire specifics the worker passes in;
    they appear when supplied and are omitted cleanly when not."""
    plain = hs.render_telegram()
    assert "THIS UPDATE" not in plain

    rich = hs.render_telegram(note="added a plain-English status tool",
                              commit="abc1234")
    assert "THIS UPDATE: added a plain-English status tool" in rich
    assert "abc1234" in rich


def test_helix_status_main_prints_and_exits_zero(capsys):
    """The CLI entry point renders the update, includes the passed-in
    fire detail, and returns 0."""
    rc = hs.main(["--note", "shipped the widget", "--commit", "deadbee"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "HELIX COMPILER" in out
    assert "shipped the widget" in out
    assert "deadbee" in out
