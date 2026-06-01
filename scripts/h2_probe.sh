#!/usr/bin/env bash
# H2 generics PROBE -- capture CURRENT behavior of the 4 generic probes via the cached K2.
# Probe-first (dev-opt #20): see what already passes vs fails before editing the compiler.
set -u
ROOT=/mnt/c/Projects/Kovostov-Native
BS=$ROOT/stage0/helixc-bootstrap
SRC=$ROOT/helixc/bootstrap
CACHE=$ROOT/.stage33-logs/cache
GEN=$BS/corpus_gen
cd "$BS" || { echo FATAL; exit 9; }
mkdir -p "$CACHE"
SHA=$(cat "$SRC/lexer.hx" "$SRC/parser.hx" "$SRC/kovc.hx" "$BS/seed.c" 2>/dev/null | sha256sum | cut -c1-16)
K2C="$CACHE/K2_$SHA.bin"
if [ -x "$K2C" ]; then echo "K2 CACHE HIT ($SHA)"; cp "$K2C" /tmp/K2.bin; chmod +x /tmp/K2.bin
else
  echo "K2 CACHE MISS ($SHA) -- building once"
  bash assemble_k1.sh >/dev/null 2>&1
  timeout 400 ./seed.bin k1src.hx /tmp/K1.bin; chmod +x /tmp/K1.bin
  cp k1input.hx /tmp/k1_in.hx; timeout 60 /tmp/K1.bin
  [ -s /tmp/k1_out.bin ] || { echo "FATAL: K2"; exit 8; }
  cp /tmp/k1_out.bin /tmp/K2.bin; chmod +x /tmp/K2.bin; cp /tmp/K2.bin "$K2C"
fi
probe() {  # <file> <correct-expected> <note>
  cp "$GEN/$1" /tmp/k2_in.hx; rm -f /tmp/k2_out.bin
  timeout 30 /tmp/K2.bin >/dev/null 2>&1
  if [ ! -s /tmp/k2_out.bin ]; then echo "  $1: COMPILE-FAIL  (want $2; $3)"; return; fi
  chmod +x /tmp/k2_out.bin; timeout 10 /tmp/k2_out.bin; local rc=$?
  if [ "$rc" = "$2" ]; then echo "  $1: exit=$rc == $2  PASS   ($3)"; else echo "  $1: exit=$rc != $2  FAIL   ($3)"; fi
}
echo "=== H2 generics PROBE (current behavior, before any edit) ==="
probe gen_id_i32.hx    42 "baseline turbofish i32 -- expect PASS"
probe gen_add_f32.hx    5 "generic fn param arithmetic f32 -- expect PASS (param tag flows)"
probe gen_body_local.hx 5 "generic fn T-typed local -- d.1, expect FAIL (body shared)"
probe gen_box_f32.hx    5 "generic struct f32 field math -- d.2, expect FAIL (field scalar-erased)"
probe gen_body_op.hx    4 "generic fn: T-local y drives y+y -- SHARP d.1, FAIL if body-clone erases the local"
probe gen_two_types.hx 42 "one generic fn at BOTH i32 and f32 -- expect PASS (two mono clones)"
probe gen_box_i32.hx    5 "generic struct i32 fields -- CONTROL, expect PASS (isolates the bug to non-i32 fields)"
probe gen_bare_f32.hx   3 "bare generic call id(3.0f32) no turbofish -- d.3, expect FAIL (defaults i32)"
