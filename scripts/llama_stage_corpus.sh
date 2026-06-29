#!/usr/bin/env bash
# llama_stage_corpus.sh -- stage a small OPEN, PUBLIC-DOMAIN held-out + train corpus for the
# SmolLM2-135M trainer's perplexity eval (v1.9 step5a) + multi-sequence QAT fine-tune (step5b).
#
# PROVENANCE (fully open, public domain -- NOT model-generated, NOT downloaded at runtime):
#   The text embedded below is excerpted from Project Gutenberg public-domain works:
#     - "Pride and Prejudice" by Jane Austen (first published 1813; PG eBook #1342)
#     - "A Tale of Two Cities" by Charles Dickens (first published 1859; PG eBook #98)
#     - "The Adventures of Sherlock Holmes" by A. Conan Doyle (1892; PG eBook #1661)
#   All three are in the US public domain (published well before 1929). The excerpts are typed
#   inline here (no network access) so the corpus is self-contained + reproducible. Project
#   Gutenberg texts carry no copyright on the public-domain work itself.
#
# TOKENIZER: the local SmolLM2 tokenizer (helix-llm/models/smollm2-135m/tokenizer.json), Apache-2.0,
#   via the confirmed-working alt-eval-venv one-liner. Token ids are written space-separated (the
#   format read_ids_file() in llama_train.c parses with fscanf %d).
#
# OUTPUTS (under helix-llm/, which is gitignored for ids -- the COMMITTED artifact is THIS script
#   that regenerates them, per the task's "commit the script not the ids" guidance):
#   helix-llm/ppl_heldout_ids.txt   -- one long held-out token stream (for --ppl)
#   helix-llm/train_corpus_ids.txt  -- concatenated train sequences (for --train-corpus)
#   helix-llm/train_corpus_lens.txt -- per-sequence token counts (one per line)
#
# Run as a FILE under WSL (CRLF stripped):
#   wsl.exe bash -lc "tr -d '\r' < /mnt/c/Projects/Kovostov-Native/scripts/llama_stage_corpus.sh > /tmp/stage.sh && bash /tmp/stage.sh"
set -u
ROOT="${HELIX_SRC:-}"; if [ -z "$ROOT" ]; then ROOT="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"; fi
[ -d "$ROOT/helix-llm" ] || ROOT="/mnt/c/Projects/Kovostov-Native"
cd "$ROOT" || { echo "FATAL: no repo root"; exit 9; }
PY="${ALT_PY:-$HOME/alt-eval-venv/bin/python}"
TOK="$ROOT/helix-llm/models/smollm2-135m/tokenizer.json"
OUTD="$ROOT/helix-llm"
[ -x "$PY" ] || { echo "FATAL: tokenizer venv python not found at $PY"; exit 9; }
[ -f "$TOK" ] || { echo "FATAL: tokenizer.json not found at $TOK"; exit 9; }

# ---- HELD-OUT text (public domain): Pride & Prejudice opening + A Tale of Two Cities opening. ----
HELDOUT_TXT=$(cat <<'PD_HELDOUT'
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters. "My dear Mr. Bennet," said his lady to him one day, "have you heard that Netherfield Park is let at last?" Mr. Bennet replied that he had not. "But it is," returned she; "for Mrs. Long has just been here, and she told me all about it." Mr. Bennet made no answer. "Do you not want to know who has taken it?" cried his wife impatiently. "You want to tell me, and I have no objection to hearing it." This was invitation enough. "Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a young man of large fortune from the north of England; that he came down on Monday in a chaise and four to see the place, and was so much delighted with it that he agreed with Mr. Morris immediately; that he is to take possession before Michaelmas, and some of his servants are to be in the house by the end of next week."
It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way - in short, the period was so far like the present period, that some of its noisiest authorities insisted on its being received, for good or for evil, in the superlative degree of comparison only. There were a king with a large jaw and a queen with a plain face, on the throne of England; there were a king with a large jaw and a queen with a fair face, on the throne of France. In both countries it was clearer than crystal to the lords of the State preserves of loaves and fishes, that things in general were settled for ever.
PD_HELDOUT
)

# ---- TRAIN text (public domain): 3 short sequences (P&P, Sherlock Holmes, Tale of Two Cities). ----
TRAIN_SEQ1=$(cat <<'PD_S1'
Mr. Bennet was so odd a mixture of quick parts, sarcastic humour, reserve, and caprice, that the experience of three and twenty years had been insufficient to make his wife understand his character. Her mind was less difficult to develop. She was a woman of mean understanding, little information, and uncertain temper. When she was discontented, she fancied herself nervous. The business of her life was to get her daughters married; its solace was visiting and news.
PD_S1
)
TRAIN_SEQ2=$(cat <<'PD_S2'
To Sherlock Holmes she is always the woman. I have seldom heard him mention her under any other name. In his eyes she eclipses and predominates the whole of her sex. It was not that he felt any emotion akin to love for Irene Adler. All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind. He was, I take it, the most perfect reasoning and observing machine that the world has seen.
PD_S2
)
TRAIN_SEQ3=$(cat <<'PD_S3'
There were a king with a large jaw and a queen with a plain face on the throne of England. It was the year of Our Lord one thousand seven hundred and seventy-five. Spiritual revelations were conceded to England at that favoured period, as at this. France, less favoured on the whole as to matters spiritual than her sister of the shield and trident, rolled with exceeding smoothness down hill, making paper money and spending it.
PD_S3
)

export TOK HELDOUT_TXT TRAIN_SEQ1 TRAIN_SEQ2 TRAIN_SEQ3 OUTD
"$PY" - <<'PYEOF'
import os
from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.environ["TOK"])
outd = os.environ["OUTD"]

def enc(s):
    return tok.encode(s).ids

# held-out: one long stream
held = enc(os.environ["HELDOUT_TXT"])
with open(os.path.join(outd, "ppl_heldout_ids.txt"), "w", newline="\n") as f:
    f.write(" ".join(str(i) for i in held) + "\n")

# train: 3 sequences -> concatenated stream + lengths
seqs = [enc(os.environ["TRAIN_SEQ1"]),
        enc(os.environ["TRAIN_SEQ2"]),
        enc(os.environ["TRAIN_SEQ3"])]
flat = [i for s in seqs for i in s]
with open(os.path.join(outd, "train_corpus_ids.txt"), "w", newline="\n") as f:
    f.write(" ".join(str(i) for i in flat) + "\n")
with open(os.path.join(outd, "train_corpus_lens.txt"), "w", newline="\n") as f:
    for s in seqs:
        f.write(str(len(s)) + "\n")

print("[stage] held-out tokens : %d" % len(held))
print("[stage] train sequences : %d  (lens: %s)  total %d tokens" %
      (len(seqs), ",".join(str(len(s)) for s in seqs), len(flat)))
print("[stage] wrote ppl_heldout_ids.txt, train_corpus_ids.txt, train_corpus_lens.txt under %s" % outd)
PYEOF
echo "LLAMA_STAGE_CORPUS_DONE"
