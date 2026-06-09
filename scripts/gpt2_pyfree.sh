#!/usr/bin/env bash
# gpt2_pyfree.sh -- FAIL-CLOSED gate proving the GPT-2-on-Helix demo's PRODUCTION data
# path is Python-free. The two offline steps that used to need a Python interpreter --
#   (1) tokenization (text<->token-ids) and
#   (2) weight import (safetensors -> the flat HXGW .weights file)
# -- are re-authored as Category-B HOST TOOLS in C (helixc/runtime/gpt2_tok.c +
# helixc/runtime/gpt2_pack.c): offline, OUTSIDE the self-host fixpoint, ZERO arithmetic
# on the compute-trust path (exactly like helixc/runtime/cpu_host.c). With these, the
# demo runs with ZERO Python installed.
#
# The independent numpy/Python oracle (helix-llm/tools/gpt2_numpy_ref.py + gpt2_import.py)
# STAYS Python ON PURPOSE -- it is the cross-check verifier and its independence is the
# whole point. THIS GATE uses it to generate the REAL reference encode/decode pairs and
# the reference .weights, so the parity is measured, never assumed. (Running the *gate*
# needs python3+regex+numpy; running the *demo* does not -- that is precisely the claim.)
#
#   MSYS_NO_PATHCONV=1 wsl.exe bash -c "bash /mnt/c/Projects/Kovostov-Native/scripts/gpt2_pyfree.sh"
#
# Checks (each fail-closed; a printed FAIL is never exit 0):
#   [T1] tokenizer ENCODE parity vs the Python oracle over a varied test set (ASCII,
#        leading/trailing/internal spaces, punctuation, contractions, numbers, unicode
#        letters/numbers, newlines/tabs, emoji) -- bit-exact id sequences.
#   [T2] tokenizer DECODE parity: feed the oracle's ids to the C decoder; bytes identical.
#   [T3] pinned demo prompt: "The capital of France is" -> ids 464 3139 286 4881 318.
#   [T4] hero decode: the demo's generated ids decode to the exact hero sentence.
#   [P1] importer byte-identity: gpt2_pack output sha256 == the Python importer's .weights
#        (streamed; RAM bounded), 64-byte HXGW header + flat un-transposed fp32 body.
# Prints GPT2_PYFREE_PASS / GPT2_PYFREE_FAIL and propagates to the process exit status.
#
# LANGUAGE: C (gcc), not Helix. Honest rationale (reported in the deliverable): Helix's
# read_file_to_arena has a ~1 MB read buffer with a ud2 truncation trap, and GPT-2's
# vocab.json (~1.04 MB) + merges.txt (~0.46 MB) exceed it; Helix has no regex for the
# pretokenizer; and the project's own precedent (cpu_host.c) is a Category-B C host tool.
# So C is the pragmatic, aligned choice -- the Python *interpreter* dependency is what is
# being eliminated, and a from-raw-buildable C host tool does that with no fence cost to
# the self-host fixpoint. (The Unicode \p{L}/\p{N}/\s tables gpt2_tok.c uses are a
# committed generated DATA file, gpt2_unicode_ranges.inc -- a .inc, NOT a .c/.h, so it is
# outside the .c/.h fence, like a .hx.)
#
# ext4-mirror pattern for speed (DrvFs gcc is slow); model files read fenced from helix-llm/.
# License: Apache 2.0.
set -u

# ---- paths (NEVER assign a /mnt path to a variable -- MSYS empties it; use literals) ----
ROOT=/mnt/c/Projects/Kovostov-Native
SRC=/mnt/c/Projects/Kovostov-Native/helixc/runtime          # the two committed C host tools (+ the .inc)
TOOLS=/mnt/c/Projects/Kovostov-Native/helix-llm/tools       # fenced Python oracle/importer (verifier)
MODEL=/mnt/c/Projects/Kovostov-Native/helix-llm/models/gpt2 # fenced model files (vocab/merges/safetensors/config)
BD=$HOME/gpt2pf/bs                                           # ext4 build dir (fast)
WORK=$HOME/gpt2pf/work                                       # ext4 scratch (refs, outputs)
OK=1
say(){ echo "[pyfree] $*"; }
bad(){ echo "[pyfree] *** FAIL: $*" >&2; OK=0; }

echo "============================================================"
echo " GPT-2-on-Helix demo: Python-free PRODUCTION data path gate"
echo " (tokenizer + importer in C; numpy/Python oracle stays the verifier)"
echo "============================================================"

# ---- [0] inputs present ----
VOCAB=$MODEL/vocab.json
MERGES=$MODEL/merges.txt
SAFE=$MODEL/model.safetensors
CFG=$MODEL/config.json
for f in "$VOCAB" "$MERGES" "$SAFE" "$CFG" "$SRC/gpt2_tok.c" "$SRC/gpt2_pack.c" "$SRC/gpt2_unicode_ranges.inc" "$TOOLS/gpt2_numpy_ref.py" "$TOOLS/gpt2_import.py"; do
  [ -s "$f" ] || bad "missing input: $f"
done
command -v gcc     >/dev/null 2>&1 || bad "gcc not found (needed to build the C host tools)"
command -v python3 >/dev/null 2>&1 || bad "python3 not found (needed to run the VERIFIER oracle; the demo itself needs no python)"
[ "$OK" = "1" ] || { echo "GPT2_PYFREE_FAIL"; exit 1; }
mkdir -p "$BD" "$WORK"

# ---- [1] build the two C host tools (gcc, ext4) -- STRICTLY SERIAL ----
say "[1] build gpt2_tok.c + gpt2_pack.c (gcc -O2, ext4)"
tr -d '\r' < "$SRC/gpt2_tok.c"  > "$BD/gpt2_tok.c"
cp "$SRC/gpt2_unicode_ranges.inc" "$BD/gpt2_unicode_ranges.inc"
tr -d '\r' < "$SRC/gpt2_pack.c" > "$BD/gpt2_pack.c"
gcc "$BD/gpt2_tok.c"  -O2 -o "$BD/gpt2_tok"  2>"$WORK/tok_gcc.log"  || { bad "gpt2_tok build failed"; sed 's/^/    /' "$WORK/tok_gcc.log" >&2; }
gcc "$BD/gpt2_pack.c" -O2 -o "$BD/gpt2_pack" 2>"$WORK/pack_gcc.log" || { bad "gpt2_pack build failed"; sed 's/^/    /' "$WORK/pack_gcc.log" >&2; }
[ -x "$BD/gpt2_tok" ]  && say "    built gpt2_tok"  || bad "no gpt2_tok binary"
[ -x "$BD/gpt2_pack" ] && say "    built gpt2_pack" || bad "no gpt2_pack binary"
[ "$OK" = "1" ] || { echo "GPT2_PYFREE_FAIL"; exit 1; }

# ---- [2] generate the REAL reference from the Python oracle (varied test set) ----
say "[2] generate reference encode/decode pairs from the Python oracle (real, not assumed)"
python3 - "$WORK" "$ROOT" <<'PY' || bad "oracle reference generation failed"
import sys, os, base64
work, root = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(root, "helix-llm", "tools"))
import gpt2_numpy_ref as ref
md = os.path.join(root, "helix-llm", "models", "gpt2")
tok = ref.BPE(os.path.join(md, "vocab.json"), os.path.join(md, "merges.txt"))
tests = [
    "", " ", "  ", "   leading triple space",
    "The capital of France is",
    "The capital of France is the capital of the French Republic, and the capital of the French Republic is the capital of the French",
    "Hello, world!", "hello world", " hello", "hello ",
    "don't can't won't I'm we're they've he'll she'd",
    "It's a test's of 'quotes' and don't.",
    "Numbers: 0 1 2 42 1234567890 3.14159 -7", "12345", "Mixed123abc456DEF",
    "Punctuation!!! ??? ... ,,, ;;; :::",
    "Tabs\tand\nnewlines\r\nhere", "\n\n\nleading newlines",
    "trailing spaces and tab \t   ", "CamelCase snake_case kebab-case",
    "email@example.com http://url.test/path?q=1&r=2",
    "emoji test \U0001F600\U0001F680❤ done",
    "accents: café naïve résumé Zürich",
    "unicode letters: Ελληνικά Русский 日本語 中文 한국어",
    "unicode numbers: ٠١٢ 一二三",
    "math: ∀x∈ℝ, x²≥0",
    "mixed ws:  a   b\t\tc\n\nd", "a"*50, " " + "x"*40,
    "The quick brown fox jumps over the lazy dog. 0123456789!",
    "Rare: \U0001F1FA\U0001F1F8 flags and ½ ⅓ fractions",
    "Repeated merges: aaaa bbbb cccc the the the and and and",
    "Quotes mix: \"double\" 'single' `back` (paren) [brack] {brace}",
    "ALLCAPSWORD lowercaseword MiXeDcAsE",
    "C++ is fun; Python3 too. node.js & rust-lang!",
    "\t\t\ttabs then text", "end with apostrophe'", "'leading apostrophe s",
    "multiple    internal     spaces      collapse?",
]
fb = open(os.path.join(work, "tests.b64"), "w")
fe = open(os.path.join(work, "py_enc.txt"), "w")
fd = open(os.path.join(work, "py_dec.b64"), "w")
for t in tests:
    fb.write(base64.b64encode(t.encode("utf-8")).decode("ascii") + "\n")
    ids = tok.encode(t)
    fe.write(" ".join(str(i) for i in ids) + "\n")
    fd.write(base64.b64encode(tok.decode(ids).encode("utf-8")).decode("ascii") + "\n")
fb.close(); fe.close(); fd.close()
print("[pyfree]     oracle produced %d reference pairs" % len(tests))
PY
[ "$OK" = "1" ] || { echo "GPT2_PYFREE_FAIL"; exit 1; }

# ---- [T1]/[T2] tokenizer encode + decode parity over the varied test set ----
say "[T1/T2] tokenizer encode + decode parity vs oracle (bit-exact)"
mapfile -t B64   < "$WORK/tests.b64"
mapfile -t PYENC < "$WORK/py_enc.txt"
mapfile -t PYDEC < "$WORK/py_dec.b64"
NTEST=0; ENCFAIL=0; DECFAIL=0
for idx in "${!B64[@]}"; do
  NTEST=$((NTEST+1))
  printf '%s' "${B64[$idx]}" | base64 -d > "$WORK/in.txt"
  cids=$("$BD/gpt2_tok" "$VOCAB" "$MERGES" encode-file "$WORK/in.txt")
  if [ "$cids" != "${PYENC[$idx]}" ]; then
    ENCFAIL=$((ENCFAIL+1))
    echo "    ENC MISMATCH #$idx in_b64=${B64[$idx]}"; echo "      py: ${PYENC[$idx]}"; echo "      c : $cids"
  fi
  if [ -n "${PYENC[$idx]}" ]; then cdec=$("$BD/gpt2_tok" "$VOCAB" "$MERGES" decode ${PYENC[$idx]} | base64 -w0)
  else cdec=$(printf '' | base64 -w0); fi
  if [ "$cdec" != "${PYDEC[$idx]}" ]; then
    DECFAIL=$((DECFAIL+1)); echo "    DEC MISMATCH #$idx py_b64=${PYDEC[$idx]} c_b64=$cdec"
  fi
done
if [ "$ENCFAIL" = "0" ]; then say "    ENCODE parity: $NTEST/$NTEST bit-exact"; else bad "ENCODE parity: $ENCFAIL/$NTEST mismatched"; fi
if [ "$DECFAIL" = "0" ]; then say "    DECODE parity: $NTEST/$NTEST bit-exact"; else bad "DECODE parity: $DECFAIL/$NTEST mismatched"; fi

# ---- [T3] pinned demo prompt ----
say "[T3] pinned prompt parity"
PIN=$(printf '%s' "The capital of France is" | "$BD/gpt2_tok" "$VOCAB" "$MERGES" encode)
if [ "$PIN" = "464 3139 286 4881 318" ]; then say "    'The capital of France is' -> $PIN  (matches pinned)"; else bad "pinned prompt -> '$PIN' (want '464 3139 286 4881 318')"; fi

# ---- [T4] hero decode ----
say "[T4] hero decode parity"
HERO_IDS="464 3139 286 4881 318 262 3139 286 262 4141 2066 11 290 262 3139 286 262 4141 2066 318 262 3139 286 262 4141"
HERO_WANT="The capital of France is the capital of the French Republic, and the capital of the French Republic is the capital of the French"
HERO_GOT=$("$BD/gpt2_tok" "$VOCAB" "$MERGES" decode $HERO_IDS)
if [ "$HERO_GOT" = "$HERO_WANT" ]; then say "    hero ids decode to the exact hero sentence"; else bad "hero decode mismatch: got '$HERO_GOT'"; fi

# ---- [P1] importer byte-identity vs the Python importer ----
say "[P1] importer byte-identity vs the Python importer (.weights sha256)"
rm -f "$WORK/py.weights" "$WORK/c.weights"
( cd "$TOOLS" && python3 gpt2_import.py --out "$WORK/py.weights" ) > "$WORK/py_import.log" 2>&1 || { bad "python importer failed"; tail -5 "$WORK/py_import.log" >&2; }
if [ -s "$WORK/py.weights" ]; then
  "$BD/gpt2_pack" "$SAFE" "$CFG" "$WORK/c.weights" > "$WORK/c_pack.log" 2>&1 || { bad "gpt2_pack run failed"; tail -5 "$WORK/c_pack.log" >&2; }
  if [ -s "$WORK/c.weights" ]; then
    PYSHA=$(sha256sum "$WORK/py.weights" | cut -d' ' -f1)
    CSHA=$(sha256sum "$WORK/c.weights"  | cut -d' ' -f1)
    PYSZ=$(stat -c%s "$WORK/py.weights"); CSZ=$(stat -c%s "$WORK/c.weights")
    say "    python .weights: $PYSHA ($PYSZ B)"
    say "    C      .weights: $CSHA ($CSZ B)"
    if [ "$PYSHA" = "$CSHA" ] && [ "$PYSZ" = "$CSZ" ]; then say "    BYTE-IDENTICAL importer output"; else bad "importer output NOT byte-identical"; cmp "$WORK/py.weights" "$WORK/c.weights" 2>&1 | head -2 >&2; fi
    # bonus: corroborate against the committed gpt2_124M.weights if present (non-fatal note)
    if [ -s "$MODEL/gpt2_124M.weights" ]; then
      COMM=$(sha256sum "$MODEL/gpt2_124M.weights" | cut -d' ' -f1)
      if [ "$COMM" = "$CSHA" ]; then say "    (also == committed gpt2_124M.weights $COMM)"; else say "    (note: committed gpt2_124M.weights sha $COMM differs from this run -- regenerate if stale)"; fi
    fi
  else bad "gpt2_pack produced no output"; fi
else bad "python importer produced no reference .weights"; fi

# ---- verdict ----
echo "============================================================"
if [ "$OK" = "1" ]; then
  echo "GPT2_PYFREE_PASS"
  echo "  tokenizer (encode+decode) + importer are bit-exact vs the Python oracle;"
  echo "  the demo's production data path runs with ZERO Python installed."
  exit 0
else
  echo "GPT2_PYFREE_FAIL"
  exit 1
fi
