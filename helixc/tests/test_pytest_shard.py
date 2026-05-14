"""Unit tests for the lightweight pytest sharding helper."""

from __future__ import annotations

import json

from scripts import pytest_shard


def test_pytest_shard_uses_hash_without_weights():
    nodeids = ["test_a.py::test_one", "test_b.py::test_two"]

    assigned = pytest_shard.assign_nodeids_to_buckets(nodeids, 3, {})

    assert assigned == {
        nodeid: pytest_shard.stable_bucket(nodeid, 3)
        for nodeid in nodeids
    }


def test_pytest_shard_balances_known_duration_weights():
    nodeids = ["heavy", "medium", "small"]
    weights = {"heavy": 10.0, "medium": 5.0, "small": 1.0}

    assigned = pytest_shard.assign_nodeids_to_buckets(nodeids, 2, weights)

    assert assigned["heavy"] == 0
    assert assigned["medium"] == 1
    assert assigned["small"] == 1


def test_pytest_shard_loads_and_writes_duration_json(tmp_path):
    path = tmp_path / "durations.json"

    pytest_shard.write_node_durations(
        str(path),
        {"b::test_two": 2.0, "a::test_one": 1.0},
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == pytest_shard.NODE_DURATION_SCHEMA
    assert pytest_shard.load_duration_weights(str(path)) == {
        "a::test_one": 1.0,
        "b::test_two": 2.0,
    }


def test_pytest_shard_ignores_missing_or_bad_weight_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")

    assert pytest_shard.load_duration_weights(str(tmp_path / "missing.json")) == {}
    assert pytest_shard.load_duration_weights(str(bad)) == {}
