#!/usr/bin/env bash
# about.sh -- Helix easter egg (undocumented by design).
#
# The "about Helix" message is itself a Helix program: helixc/examples/about.hx.
# If a from-raw-built Helix compiler (kovc) is cached, this COMPILES + RUNS that
# Helix program -- so the plaque is rendered by the very compiler you can rebuild
# from a 299-byte raw-binary root. Otherwise it reads the message straight from
# the (auditable) Helix source. Either way the message is the same bytes.
#
#   bash scripts/about.sh                  # instant: compile-via-cache, else print source
#   HELIX_ABOUT_BUILD=1 bash scripts/about.sh   # build kovc from raw first (slow on /mnt drives)
set -u
ROOT="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"
[ -d "$ROOT/helixc/examples" ] || ROOT="/mnt/c/Projects/Kovostov-Native"
BS="$ROOT/stage0/helixc-bootstrap"; SRC="$ROOT/helixc/bootstrap"
ABOUT="$ROOT/helixc/examples/about.hx"
CACHE="$ROOT/.stage33-logs/cache"

print_from_source() {
  # The message lives verbatim in the Helix source -- read it directly.
  grep -oE 'print_str\("[^"]*"\)' "$ABOUT" 2>/dev/null \
    | sed -e 's/^print_str("//' -e 's/")$//' -e 's/\\n$//' | sed '/^$/d'
}

# Build-or-reuse the from-raw Helix compiler (K2), keyed by compiler-source sha.
SHA=$(cat "$SRC/lexer.hx" "$SRC/parser.hx" "$SRC/kovc.hx" "$BS/seed.c" 2>/dev/null | sha256sum 2>/dev/null | cut -c1-16)
K2C="$CACHE/K2_$SHA.bin"
if [ -n "$SHA" ] && [ -x "$K2C" ]; then
  cp "$K2C" /tmp/K2.bin && chmod +x /tmp/K2.bin
elif [ "${HELIX_ABOUT_BUILD:-0}" = "1" ] && [ -x "$BS/seed.bin" ] && [ -f "$BS/assemble_k1.sh" ]; then
  mkdir -p "$CACHE" 2>/dev/null
  ( cd "$BS" \
    && bash assemble_k1.sh >/dev/null 2>&1 \
    && timeout 900 ./seed.bin k1src.hx /tmp/K1.bin >/dev/null 2>&1 && chmod +x /tmp/K1.bin \
    && cp k1input.hx /tmp/k1_in.hx && timeout 120 /tmp/K1.bin >/dev/null 2>&1 )
  if [ -s /tmp/k1_out.bin ]; then
    cp /tmp/k1_out.bin /tmp/K2.bin && chmod +x /tmp/K2.bin
    [ -n "$SHA" ] && cp /tmp/K2.bin "$K2C" 2>/dev/null
  fi
fi

# Render: compile + run via the from-raw kovc when available; else read the source.
if [ -x /tmp/K2.bin ] && [ -f "$ABOUT" ]; then
  cp "$ABOUT" /tmp/k2_in.hx; rm -f /tmp/k2_out.bin
  timeout 60 /tmp/K2.bin >/dev/null 2>&1
  if [ -s /tmp/k2_out.bin ]; then chmod +x /tmp/k2_out.bin; /tmp/k2_out.bin; exit 0; fi
fi
print_from_source
