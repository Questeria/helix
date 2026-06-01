#!/usr/bin/env bash
# H2 probe2: generic impl method + generic enum, via the cached (post-d.2) K2.
set -u
ROOT=/mnt/c/Projects/Kovostov-Native
BS=$ROOT/stage0/helixc-bootstrap; SRC=$ROOT/helixc/bootstrap; CACHE=$ROOT/.stage33-logs/cache; GEN=$BS/corpus_gen
cd "$BS" || { echo FATAL; exit 9; }
SHA=$(cat "$SRC/lexer.hx" "$SRC/parser.hx" "$SRC/kovc.hx" "$BS/seed.c" 2>/dev/null | sha256sum | cut -c1-16)
K2C="$CACHE/K2_$SHA.bin"
if [ -x "$K2C" ]; then echo "K2 CACHE HIT ($SHA)"; cp "$K2C" /tmp/K2.bin; chmod +x /tmp/K2.bin
else echo "K2 CACHE MISS ($SHA) -- building"; bash assemble_k1.sh >/dev/null 2>&1; timeout 400 ./seed.bin k1src.hx /tmp/K1.bin; chmod +x /tmp/K1.bin; cp k1input.hx /tmp/k1_in.hx; timeout 60 /tmp/K1.bin; [ -s /tmp/k1_out.bin ] || { echo FATAL K2; exit 8; }; cp /tmp/k1_out.bin /tmp/K2.bin; chmod +x /tmp/K2.bin; cp /tmp/K2.bin "$K2C"; fi
pr() { cp "$GEN/$1" /tmp/k2_in.hx; rm -f /tmp/k2_out.bin; timeout 30 /tmp/K2.bin >/dev/null 2>&1
  if [ ! -s /tmp/k2_out.bin ]; then echo "  $1: COMPILE-FAIL (want $2; $3)"; return; fi
  chmod +x /tmp/k2_out.bin; timeout 10 /tmp/k2_out.bin; local rc=$?
  [ "$rc" = "$2" ] && echo "  $1: $rc == $2 PASS ($3)" || echo "  $1: $rc != $2 FAIL ($3)"; }
echo "=== H2 probe2 (impl + enum) ==="
pr gen_impl_f32.hx    5 "generic struct + impl method over T returning f32"
pr gen_option_i32.hx 42 "generic enum Opt<T> Some(T)/None + match"
