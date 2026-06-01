#!/usr/bin/env bash
# assemble_k1.sh -- Helix v1.1 H1 (retire shell): THIN wrapper. The K1 source-assembly
# LOGIC (strip_demo + concat + driver-main) is now the Helix-native assemble_k1.hx,
# compiled by the FROZEN raw-binary seed and run -- replacing the former shell awk/printf/
# cat. Output (k1src.hx / k1input.hx / k1ptxdrv.hx) is byte-identical to the old shell
# version (gated by the full self-host fixpoint K2==K3==K4 + 35-corpus). The old shell
# concatenation logic is preserved in git history.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
./seed.bin assemble_k1.hx /tmp/asm_k1.bin || { echo "FATAL: seed could not compile assemble_k1.hx" >&2; exit 7; }
chmod +x /tmp/asm_k1.bin
/tmp/asm_k1.bin || { echo "FATAL: assemble_k1.hx run failed" >&2; exit 8; }
echo "assembled (Helix concatenator): k1src.hx k1input.hx k1ptxdrv.hx"
