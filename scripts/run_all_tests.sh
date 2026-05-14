#!/usr/bin/env bash
# Run every test suite. From project root.
set -u
cd "$(dirname "$0")/.."

SHARDS="${HELIX_TEST_SHARDS:-}"
MAX_SHARDS=8

if [[ -n "$SHARDS" ]]; then
    if ! [[ "$SHARDS" =~ ^[0-9]+$ ]]; then
        echo "HELIX_TEST_SHARDS must be an integer from 1 to $MAX_SHARDS" >&2
        exit 2
    fi
    SHARDS_DEC=$((10#$SHARDS))
    if (( SHARDS_DEC < 1 || SHARDS_DEC > MAX_SHARDS )); then
        echo "HELIX_TEST_SHARDS must be an integer from 1 to $MAX_SHARDS" >&2
        exit 2
    fi
    SHARDS="$SHARDS_DEC"
fi

ensure_python_with_pytest() {
    local candidate="${PYTHON:-python}"
    if "$candidate" -c "import pytest" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        return 0
    fi

    local venv_py=".stage31-venv/bin/python"
    if [[ ! -x "$venv_py" ]]; then
        echo "python pytest unavailable; creating .stage31-venv"
        if ! "$candidate" -m venv .stage31-venv; then
            echo "failed to create .stage31-venv with $candidate"
            return 1
        fi
    fi
    if ! "$venv_py" -c "import pytest" >/dev/null 2>&1; then
        echo "installing pytest into .stage31-venv"
        if ! "$venv_py" -m pip install -q pytest; then
            echo "failed to install pytest into .stage31-venv"
            return 1
        fi
    fi
    PYTHON_BIN="$venv_py"
    return 0
}

echo "pytest (stage31 sharded gate):"
SHARD_ARGS=()
if [[ -n "$SHARDS" ]]; then
    SHARD_ARGS=(--shards "$SHARDS")
fi

if ! ensure_python_with_pytest; then
    PYTEST_RC=1
elif "$PYTHON_BIN" scripts/stage31_validate.py \
        --mode full \
        "${SHARD_ARGS[@]}"; then
    PYTEST_RC=0
else
    PYTEST_RC=$?
fi

echo
echo "stage0/hex0:"
if [[ -f stage0/hex0/run_tests.sh ]]; then
    if (cd stage0/hex0 && bash run_tests.sh); then
        HEX0_RC=0
    else
        HEX0_RC=$?
    fi
else
    echo "stage0/hex0/run_tests.sh not found"
    HEX0_RC=1
fi

echo
echo "============================="
echo "pytest gate rc: $PYTEST_RC"
echo "stage0/hex0 rc: $HEX0_RC"

if [[ "$PYTEST_RC" -eq 0 && "$HEX0_RC" -eq 0 ]]; then
    echo "TOTAL: all gates passed"
    exit 0
fi

echo "TOTAL: one or more gates failed"
exit 1
