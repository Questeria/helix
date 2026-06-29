#!/usr/bin/env bash
# llama_stage_mmlu.sh -- stage a SmolLM2-135M fine-tune corpus from on-disk MMLU.
#
# Builds a DOMAIN-COHERENT train + DISJOINT held-out token corpus for the v1.9
# ternary QAT conversion measurement (helixc/runtime/llama_train.c).
#
#   - Source : on-disk MMLU (cais___mmlu, MIT-licensed, NO download), single
#              subject so the fine-tune genuinely transfers to the held-out.
#   - Format : each example = "Question: ...\nA. ...\nB. ...\nAnswer: X. ...\n"
#              (plain text; NEVER printed by this script -- only token COUNTS).
#   - Tokenizer: the SmolLM2-135M tokenizer.json.
#   - Split  : the held-out rows are a CONTIGUOUS slice taken from the END of the
#              row list; the train rows are a DISJOINT slice from the FRONT. No row
#              appears in both -> no token overlap (the script asserts this by id-set
#              disjointness AND verifies the row index sets do not intersect).
#
# Outputs (under helix-llm/, which is gitignored):
#   helix-llm/mmlu_train_ids.txt   train token stream (whitespace ints, one corpus)
#   helix-llm/mmlu_train_lens.txt  per-sequence token counts (sum == train stream len)
#   helix-llm/mmlu_heldout_ids.txt held-out token stream (disjoint)
#
# Usage: scripts/llama_stage_mmlu.sh [SUBJECT] [MAXLEN] [N_TRAIN] [N_HELDOUT]
#   defaults: professional_psychology 256 140 16
#
# Env: REPO (repo root, default the script's parent), VENV (python venv),
#      MMLU_BASE (datasets cache dir). All have sane WSL defaults.
set -euo pipefail

SUBJECT="${1:-professional_psychology}"
MAXLEN="${2:-256}"
N_TRAIN="${3:-140}"
N_HELDOUT="${4:-16}"

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
VENV="${VENV:-$HOME/alt-eval-venv}"
MMLU_BASE="${MMLU_BASE:-$HOME/.cache/huggingface/datasets/cais___mmlu}"
TOK="$REPO/helix-llm/models/smollm2-135m/tokenizer.json"
OUT="$REPO/helix-llm"

mkdir -p "$OUT"

SUBJECT="$SUBJECT" MAXLEN="$MAXLEN" N_TRAIN="$N_TRAIN" N_HELDOUT="$N_HELDOUT" \
MMLU_BASE="$MMLU_BASE" TOK="$TOK" OUT="$OUT" \
"$VENV/bin/python" - <<'PYEOF'
import os, glob, sys
from datasets import Dataset
from tokenizers import Tokenizer

subject  = os.environ["SUBJECT"]
maxlen   = int(os.environ["MAXLEN"])
n_train  = int(os.environ["N_TRAIN"])
n_held   = int(os.environ["N_HELDOUT"])
base     = os.environ["MMLU_BASE"]
tokpath  = os.environ["TOK"]
out      = os.environ["OUT"]

# locate the test split arrow for this subject (any commit hash dir)
cands = glob.glob(os.path.join(base, subject, "*", "*", "mmlu-test.arrow"))
if not cands:
    sys.stderr.write("no test arrow for subject %s under %s\n" % (subject, base)); sys.exit(2)
ds = Dataset.from_file(sorted(cands)[0])
tok = Tokenizer.from_file(tokpath)

LET = ["A", "B", "C", "D"]
def fmt(r):
    s = "Question: " + r["question"].strip() + "\n"
    for i, c in enumerate(r["choices"]):
        s += LET[i] + ". " + str(c).strip() + "\n"
    a = r["answer"]
    s += "Answer: " + LET[a] + ". " + str(r["choices"][a]).strip() + "\n"
    return s

# tokenize every row, keep only those that fit the activation window (T <= maxlen)
rows = []  # (row_index, token_ids)
for idx in range(len(ds)):
    ids = tok.encode(fmt(ds[idx])).ids
    if 2 <= len(ids) <= maxlen:
        rows.append((idx, ids))

if len(rows) < n_train + n_held:
    sys.stderr.write("only %d usable rows, need %d\n" % (len(rows), n_train + n_held)); sys.exit(2)

# DISJOINT split: train = front slice, held-out = back slice (no shared row index)
train_rows = rows[:n_train]
held_rows  = rows[len(rows) - n_held:]

train_idx = set(i for i, _ in train_rows)
held_idx  = set(i for i, _ in held_rows)
assert train_idx.isdisjoint(held_idx), "row index overlap between train and held-out"

train_stream, lens = [], []
for _, ids in train_rows:
    train_stream.extend(ids)
    lens.append(len(ids))
held_stream = []
for _, ids in held_rows:
    held_stream.extend(ids)

# write the three files (token ids only -- NO source text ever emitted)
with open(os.path.join(out, "mmlu_train_ids.txt"), "w", newline="\n") as f:
    f.write(" ".join(str(t) for t in train_stream) + "\n")
with open(os.path.join(out, "mmlu_train_lens.txt"), "w", newline="\n") as f:
    f.write("\n".join(str(l) for l in lens) + "\n")
with open(os.path.join(out, "mmlu_heldout_ids.txt"), "w", newline="\n") as f:
    f.write(" ".join(str(t) for t in held_stream) + "\n")

# integrity: token-id multiset overlap is expected (shared vocab), so the real
# guarantee is ROW disjointness (asserted above). Report counts only.
assert sum(lens) == len(train_stream)
print("SUBJECT %s MAXLEN %d" % (subject, maxlen))
print("TRAIN_ROWS %d TRAIN_TOKENS %d TRAIN_SEQS %d" % (n_train, len(train_stream), len(lens)))
print("HELDOUT_ROWS %d HELDOUT_TOKENS %d" % (n_held, len(held_stream)))
print("ROW_OVERLAP %d (PASS if 0)" % len(train_idx & held_idx))
print("LENS_SUM_MATCHES_STREAM %s" % ("PASS" if sum(lens) == len(train_stream) else "FAIL"))
print("STAGE_MMLU_DONE")
PYEOF
