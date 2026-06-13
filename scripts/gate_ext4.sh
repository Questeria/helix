#!/usr/bin/env bash
# gate_ext4.sh (v2) -- run the FULL gate on a WSL ext4 MIRROR, with a reproduce_trust-style
# PATH REWRITE so gate_kovc.sh's hardcoded /mnt/c paths (BS/EX) AND any /mnt/c-hardcoded .hx
# point at the mirror. This moves the gate's heavy I/O (the assemble_k1 source-regen WRITE +
# the K-generation) off the /mnt/c DrvFs bridge -- which hangs in D-state (uninterruptible
# disk-wait) under this machine's sustained-write load -- onto fast, stable ext4. Only a single
# rsync READ burst touches /mnt/c. Results are PATH-INDEPENDENT (identical sources -> identical
# fixpoint, the same property reproduce_trust.sh's [0] rewrite relies on), so a green gate on
# the mirror is valid for the /mnt/c source. COMMIT still happens on /mnt/c (git source-of-truth).
#
#   bash scripts/gate_ext4.sh
#
# v1 was BROKEN (it rsynced but gate_kovc kept its hardcoded /mnt/c paths -> ran on /mnt/c +
# hung). v2 adds the path rewrite. NEVER run concurrently with a workflow.
set -u
SRC=/mnt/c/Projects/Kovostov-Native
MIRROR="$HOME/helix-ext4"
mkdir -p "$MIRROR"
echo "[gate_ext4] rsync $SRC -> $MIRROR (one /mnt/c read burst; excl .git/venvs/node_modules/website/helix-llm)"
rsync -a --delete \
  --exclude='.git' --exclude='*venv*' --exclude='node_modules' \
  --exclude='website' --exclude='helix_website' --exclude='helix-llm' \
  "$SRC/" "$MIRROR/" || { echo "[gate_ext4] rsync FAILED"; exit 2; }
echo "[gate_ext4] path-rewrite $SRC -> $MIRROR across mirror files (reproduce_trust [0] style)"
mapfile -t HC < <(grep -rlI "$SRC" "$MIRROR" 2>/dev/null || true)
if [ "${#HC[@]}" -gt 0 ]; then
  printf '%s\n' "${HC[@]}" | xargs sed -i "s#$SRC#$MIRROR#g"
  echo "[gate_ext4] rewrote ${#HC[@]} file(s) (incl gate_kovc.sh BS/EX)"
fi
echo "[gate_ext4] running the FULL gate ON EXT4 (all heavy I/O on ext4 now)"
( cd "$MIRROR" && bash scripts/gate_kovc.sh ); rc=$?
echo "[gate_ext4] GATE_RC=$rc  (mirror: $MIRROR)  -- if GATE_PASS, commit on the /mnt/c SOURCE"
