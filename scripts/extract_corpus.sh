#!/usr/bin/env bash
# Extract the 28 generated corpus programs (the gen <<'EOF' heredocs in feature_corpus.sh)
# into byte-exact committed files under stage0/helixc-bootstrap/corpus/, so the Helix
# test_runner.hx can read_file_to_arena each. Reads the actual feature_corpus.sh -> exact.
set -u
ROOT=/mnt/c/Projects/Kovostov-Native
SRC=$ROOT/scripts/feature_corpus.sh
DIR=$ROOT/stage0/helixc-bootstrap/corpus
mkdir -p "$DIR"
awk -v dir="$DIR" '
  /^gen [A-Za-z0-9_]+\.hx <</ { name=$2; out=dir"/"name; cap=1; next }
  cap && $0=="EOF" { cap=0; close(out); next }
  cap { print > out }
' "$SRC"
echo "extracted $(ls "$DIR"/*.hx 2>/dev/null | wc -l) corpus programs to $DIR"
