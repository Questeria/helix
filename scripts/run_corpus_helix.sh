#!/usr/bin/env bash
# H1 sub-item 2 full validation: the Helix test_runner.hx covers the full 35-program corpus
# and matches feature_corpus.sh. Uses a K2 CACHE (dev-opt #1: build once, reuse) keyed by the
# compiler-source sha -- no redundant ~4min rebuild when the compiler is unchanged.
set -u
ROOT=/mnt/c/Projects/Kovostov-Native
BS=$ROOT/stage0/helixc-bootstrap
SRC=$ROOT/helixc/bootstrap
CACHE=$ROOT/.stage33-logs/cache
mkdir -p "$CACHE"
cd "$BS" || { echo "FATAL: no bootstrap dir"; exit 9; }

echo "=== [1] extract the 28 generated corpus programs ==="
bash $ROOT/.stage33-logs/extract_corpus.sh

echo "=== [2] K2 (cache build-or-reuse, keyed by compiler-source sha) ==="
SHA=$(cat "$SRC/lexer.hx" "$SRC/parser.hx" "$SRC/kovc.hx" "$BS/seed.c" 2>/dev/null | sha256sum | cut -c1-16)
K2C="$CACHE/K2_$SHA.bin"
if [ -x "$K2C" ]; then
  echo "  K2 CACHE HIT ($SHA)"; cp "$K2C" /tmp/K2.bin; chmod +x /tmp/K2.bin
else
  echo "  K2 CACHE MISS ($SHA) -- building once + caching"
  bash assemble_k1.sh
  t0=$SECONDS; timeout 400 ./seed.bin k1src.hx /tmp/K1.bin; echo "    seed->K1 rc=$? ($((SECONDS-t0))s)"
  chmod +x /tmp/K1.bin; cp k1input.hx /tmp/k1_in.hx; timeout 60 /tmp/K1.bin
  [ -s /tmp/k1_out.bin ] || { echo "FATAL: K2 not built"; echo "H1_RUNNER_FAIL"; exit 8; }
  cp /tmp/k1_out.bin /tmp/K2.bin; chmod +x /tmp/K2.bin
  cp /tmp/K2.bin "$K2C"; echo "    cached -> K2_$SHA.bin"
fi
echo "  K2 = $(stat -c%s /tmp/K2.bin) bytes"

echo "=== [3] compile + run the FULL Helix test runner (35 programs) ==="
cp test_runner.hx /tmp/k2_in.hx; rm -f /tmp/k2_out.bin
timeout 60 /tmp/K2.bin
if [ ! -s /tmp/k2_out.bin ]; then echo "RUNNER COMPILE-FAIL"; echo "H1_RUNNER_FAIL"; exit 7; fi
cp /tmp/k2_out.bin /tmp/runner.bin; chmod +x /tmp/runner.bin
timeout 180 /tmp/runner.bin; rc=$?
echo "  Helix runner exit (= #failures over 35) = $rc"

echo "=== [4] negative control: run_one must DETECT a wrong expected (exit42 asserted 99) ==="
cp neg_probe.hx /tmp/k2_in.hx; rm -f /tmp/k2_out.bin
timeout 60 /tmp/K2.bin
if [ ! -s /tmp/k2_out.bin ]; then echo "NEG_PROBE COMPILE-FAIL"; echo "H1_RUNNER_FAIL"; exit 7; fi
cp /tmp/k2_out.bin /tmp/neg.bin; chmod +x /tmp/neg.bin
timeout 30 /tmp/neg.bin; nrc=$?
echo "  neg_probe exit = $nrc (expect 1 = run_one correctly flagged the 42-vs-99 mismatch)"

echo "=== VERDICT ==="
if [ "$rc" = "0" ] && [ "$nrc" = "1" ]; then echo "H1_RUNNER_PASS (35/35 pass AND negative control detects failures)"; else echo "H1_RUNNER_FAIL (runner=$rc neg=$nrc)"; fi
