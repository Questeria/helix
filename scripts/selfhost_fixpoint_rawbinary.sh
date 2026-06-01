#!/usr/bin/env bash
# Python-FREE raw-binary self-host fixpoint proof (Helix v1.0 DoD criterion #1).
#
# Chain:  seed.bin  ->  K1  ->  K2  ->  K3  ->  K4   on the FULL kovc.hx source.
#   seed.bin = the from-raw-binary trusted seed (hex0..M2-Planet -> seed.c -> seed.bin),
#              a Python-free compiler. It compiles the full bootstrap source (lexer.hx
#              + parser.hx + kovc.hx + driver, ~1.5 MB) into K1, the self-hosted kovc.
#   K1 = seed.bin(k1src.hx);  K2 = K1(k1input.hx);  K3 = K2(k1input.hx);  K4 = K3(k1input.hx)
# FIXPOINT (criterion #1): K2 == K3 == K4 byte-identical.
#
# NO EXTERNAL ulimit (proven 2026-06-01): the seed compiles the 1.5 MB source on the
# DEFAULT 8 MB stack (seed_rc=0, byte-identical K1=595754 B), and K2+ are kovc-emitted
# and carry emit_start_bigstack() (kovc.hx:~1990, a 512 MiB mmap'd stack). The full
# chain runs on the default stack -- this runner does NOT raise it; a green run here
# proves DoD #1's "no external ulimit" requirement. (Earlier defensiveness raised it;
# removed once measured unnecessary.)
#
# The ONLY non-Helix step is assemble_k1.py concatenating the FROZEN bootstrap sources;
# that source-assembly helper is tracked for de-Python under DoD #6. The COMPILE chain
# proven here uses ZERO Python.
set -u
echo "stack soft limit (NOT raised): $(ulimit -s) KB"
HERE=/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap
cd "$HERE" || { echo "FATAL: no bootstrap dir"; exit 90; }
chmod +x seed.bin 2>/dev/null

echo "=== Python-free raw-binary self-host fixpoint ==="
echo "date: $(date -u +%FT%TZ)"
echo "kovc.hx sha: $(sha256sum ../../helixc/bootstrap/kovc.hx | awk '{print $1}')"
echo "[1/5] regenerate k1src.hx/k1input.hx from the CURRENT frozen sources (concatenator only)"
bash assemble_k1.sh || { echo "FATAL: assemble"; exit 89; }
ls -l k1src.hx k1input.hx | awk '{print "  "$5" "$9}'

run_gen () { # <label> <binary> <expected-out>
  local label="$1" bin="$2" out="$3" t0=$SECONDS rc
  "$bin"; rc=$?
  if [ ! -s "$out" ]; then echo "FATAL: $label produced empty $out (exit=$rc)"; exit 80; fi
  echo "  $label exit=$rc (low byte of out size, cosmetic) elapsed=$((SECONDS-t0))s"
}

echo "[2/5] K1 = seed.bin(k1src.hx)"
t0=$SECONDS
./seed.bin k1src.hx /tmp/K1.bin; src_rc=$?
if [ ! -s /tmp/K1.bin ]; then echo "FATAL: K1 empty (seed exit=$src_rc)"; exit 91; fi
chmod +x /tmp/K1.bin
echo "  seed->K1 exit=$src_rc elapsed=$((SECONDS-t0))s"

echo "[3/5] K2 = K1(k1input.hx)   [K1 reads /tmp/k1_in.hx -> /tmp/k1_out.bin]"
cp k1input.hx /tmp/k1_in.hx
run_gen "K1->K2" /tmp/K1.bin /tmp/k1_out.bin
cp /tmp/k1_out.bin /tmp/K2.bin; chmod +x /tmp/K2.bin

echo "[4/5] K3 = K2(k1input.hx)   [K2 reads /tmp/k2_in.hx -> /tmp/k2_out.bin]"
cp k1input.hx /tmp/k2_in.hx
run_gen "K2->K3" /tmp/K2.bin /tmp/k2_out.bin
cp /tmp/k2_out.bin /tmp/K3.bin; chmod +x /tmp/K3.bin

echo "[5/5] K4 = K3(k1input.hx)   [K3 reads /tmp/k2_in.hx -> /tmp/k2_out.bin]"
run_gen "K3->K4" /tmp/K3.bin /tmp/k2_out.bin
cp /tmp/k2_out.bin /tmp/K4.bin

echo "=== sizes + sha256 ==="
for k in K1 K2 K3 K4; do
  printf "  %s %s bytes  %s\n" "$k" "$(stat -c%s /tmp/$k.bin)" "$(sha256sum /tmp/$k.bin | awk '{print $1}')"
done

echo "=== Helix-native byte-identity check (selfhost_bytecmp.hx, seed-compiled) ==="
# The load-bearing equality assertion done by a HELIX program built by the raw-binary
# seed (read_file_to_arena + arena byte-compare), cross-checked against bash cmp.
./seed.bin selfhost_bytecmp.hx /tmp/bytecmp.bin && chmod +x /tmp/bytecmp.bin
helix_eq () { cp "$1" /tmp/cmp_a; cp "$2" /tmp/cmp_b; /tmp/bytecmp.bin; echo $?; }
hx23=$(helix_eq /tmp/K2.bin /tmp/K3.bin)
hx34=$(helix_eq /tmp/K3.bin /tmp/K4.bin)
echo "  Helix-checker: K2 vs K3 -> $hx23  K3 vs K4 -> $hx34  (0 = identical)"

echo "=== verdict (cmp AND Helix-native checker must agree) ==="
v=0
if cmp -s /tmp/K2.bin /tmp/K3.bin && [ "$hx23" = "0" ]; then echo "  K2 == K3  IDENTICAL (cmp + Helix agree)"; else echo "  K2 vs K3  MISMATCH"; v=1; fi
if cmp -s /tmp/K3.bin /tmp/K4.bin && [ "$hx34" = "0" ]; then echo "  K3 == K4  IDENTICAL (cmp + Helix agree)"; else echo "  K3 vs K4  MISMATCH"; v=1; fi
if [ $v -eq 0 ]; then
  echo "RESULT: PYTHON-FREE FULL-SOURCE SELF-HOST FIXPOINT PASS (K2==K3==K4 byte-identical; Helix-native check + cmp agree)"
else
  echo "RESULT: FIXPOINT FAIL"
fi
echo "=== done $(date -u +%FT%TZ) ==="
exit $v
