#!/usr/bin/env bash
# gpu_receipt_check.sh (v1.5 #4): a SUCCINCT VERIFIABLE-INFERENCE RECEIPT for the kovc-emitted
# ternary_matmul, verified on a real CUDA device (RTX 3070, sm_86) PLUS an independent host-side checker.
#
# MECHANISM: exact-integer FREIVALDS over F_p (p=2^31-1) made non-interactive via FIAT-SHAMIR. The runner
# emits a receipt binding SHA-256(W) || SHA-256(X) || SHA-256(C) for one ternary matmul C = W*X (ternary
# weights {-1,0,+1} x small-int activations) computed by the GENUINE #2-certified kernel. An INDEPENDENT
# checker (receipt_check) re-derives the challenge vectors r_i = SHA-256(H_W||H_X||H_C||i||j) mod p ITSELF
# (never from the receipt), recomputes X*r / W*(X*r) / C*r mod p in O(MK+KN+MN), and asserts W*(X*r)==C*r
# for all rounds. Cost O(n^2) << the O(n^3) matmul -> FASTER THAN RE-EXECUTION; the checker never re-runs
# the kernel. A range guard (|C| < p/2) makes mod-p equality <=> integer equality, so the soundness is
# EXACT (no tolerance): if C != W*X then the checker accepts with probability <= (1/p)^t = 2^-62 (t=2),
# plus the SHA-256 collision advantage. SHA-256 is from-scratch (FIPS-180-4), KAT-gated before use.
#
# HONEST SCOPE (state plainly, do NOT overstate -- the #2 audit lesson): this verifies ONE ternary/int
# matmul output of the S0 exact-int leg. It is NOT zero-knowledge (W,X,C are revealed to the checker), NOT
# a succinct SNARK (only Freivalds is sublinear), and does NOT cover the f32 SmolLM2/GPT-2 token-for-token
# path (Freivalds needs a tolerance over f32 -> forgery-soundness would be false; that f32 leg is the
# explicitly-deferred SECOND increment = a re-derivable hash-chained transcript). matmul only; one model+input.
#
# #2 -> #4 bridge: #2 (gpu_matmul_witness_check.sh) certifies the emitted kernel IS matmul for the shape;
# #4 (this) certifies THIS run's committed C equals matmul(W,X) without re-running.
#
# Negative controls (each run AS A FILE, token-gated on '-> RECEIPT_PASS/FAIL', non-vacuity-guarded, the
# committed kernel/receipt never edited): NC1 forged-output (self-consistent C[0]+1 -> Freivalds teeth),
# NC2 tampered-hash-binding (flip H_C), NC3 echo-not-trusted (forged C + doctored self-agreeing echo ->
# still rejected, proves r is checker-derived), NC4 tampered-weight-commitment (flip seed_W), NC5 vacuity
# (truncated receipt fail-closed), NC6 kernel-corruption (drop-term ternary kernel from /tmp).
#
# Run under WSL (CUDA 12.0+12.8): bash scripts/gpu_receipt_check.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
EX="$ROOT/helixc/examples"
KERN="$EX/ternary_matmul_kernel.hx"
BS="$ROOT/stage0/helixc-bootstrap"
M=16; K=16; N=16
OK=1
say(){ echo "[receipt] $*"; }
bad(){ echo "[receipt] *** FAIL: $*" >&2; OK=0; }
emit(){ cp "$1" /tmp/kernel_in.hx; rm -f /tmp/out.ptx; "$DRV" >/dev/null 2>&1 || true; }
vchk(){ "$CL" x x 0 receipt_check "$1" 2>&1 | sed -n 's/.*-> \(RECEIPT_PASS\|RECEIPT_FAIL\)$/\1/p' | tail -1; }
# nc <label> <receiptpath> : the receipt MUST be rejected
nc(){ local v; v=$(vchk "$2"); if [ "$v" = RECEIPT_FAIL ]; then say "    [$1] receipt_check=RECEIPT_FAIL (expected FAIL)  OK"; else bad "[$1] receipt_check=$v but expected RECEIPT_FAIL"; fi; }

echo "============================================================"
echo " Helix v1.5 #4: ternary_matmul verifiable-inference receipt (Freivalds + Fiat-Shamir)"
echo "============================================================"

# --- [A] obtain the PTX driver (reuse the fast_iter ext4 cache -- #4 is HOST-SIDE only, NO kovc edit, so
#         the cdcf8673 driver is valid; else mint from raw) + emit the ternary_matmul PTX ---
say "[A] obtain PTX driver + emit ternary_matmul PTX"
CACHE="$HOME/.helix_fastiter"
CUR=$(cat "$ROOT/helixc/bootstrap/kovc.hx" "$ROOT/helixc/bootstrap/parser.hx" 2>/dev/null | sha256sum | cut -d' ' -f1)
if [ -s "$CACHE/newdrv.bin" ] && [ "$(cat "$CACHE/compiler.sha" 2>/dev/null)" = "$CUR" ]; then
  DRV="$CACHE/newdrv.bin"; say "    reusing cached driver (compiler unchanged -- no #4 kovc edit)"
else
  say "    minting driver from raw (compiler changed / no cache; ~4min)"
  ( cd "$BS" && bash assemble_k1.sh >/tmp/rc_asm.log 2>&1 )
  chmod +x "$BS/seed.bin" 2>/dev/null || true
  ( cd "$BS" && ./seed.bin k1ptxdrv.hx /tmp/rc_newdrv.bin >/tmp/rc_drv.log 2>&1 ) || true
  if [ ! -s /tmp/rc_newdrv.bin ]; then bad "PTX driver not built"; echo "RECEIPT_CHECK_FAIL"; exit 1; fi
  chmod +x /tmp/rc_newdrv.bin; DRV=/tmp/rc_newdrv.bin
  mkdir -p "$CACHE"; cp "$DRV" "$CACHE/newdrv.bin" 2>/dev/null && echo "$CUR" > "$CACHE/compiler.sha"
fi
emit "$KERN"
if [ ! -s /tmp/out.ptx ]; then bad "ternary_matmul PTX not emitted"; echo "RECEIPT_CHECK_FAIL"; exit 1; fi
cp /tmp/out.ptx /tmp/rc_tern.ptx
say "    emitted ternary_matmul PTX ($(wc -c < /tmp/rc_tern.ptx) B)"

# --- [B] build the launcher (committed cuda_launch.c, -lcuda -lcublas -lm) ---
say "[B] build cuda_launch.c launcher"
gcc "$ROOT/helixc/runtime/cuda_launch.c" -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcuda -lcublas -lm -o /tmp/rc_cl >/tmp/rc_gcc.log 2>&1
if [ ! -s /tmp/rc_cl ]; then bad "launcher build failed:"; tail -6 /tmp/rc_gcc.log >&2; echo "RECEIPT_CHECK_FAIL"; exit 1; fi
CL=/tmp/rc_cl

# --- [C] SHA-256 NIST KAT self-test -- the hash MUST be correct before any binding is trusted ---
say "[C] SHA-256 self-test (3 NIST KAT vectors)"
ST=$("$CL" x x 0 sha256_selftest 2>&1 | sed -n 's/.*-> \(PASS\|FAIL\)$/\1/p' | tail -1)
if [ "$ST" = PASS ]; then say "    sha256_selftest=PASS"; else bad "sha256_selftest=$ST (the from-scratch SHA-256 is wrong -> all bindings void)"; echo "RECEIPT_CHECK_FAIL"; exit 1; fi

# --- [D] POSITIVE: emit a genuine receipt from the real kernel, the independent checker ACCEPTS ---
say "[D] positive: genuine receipt verifies (RECEIPT_PASS)"
"$CL" /tmp/rc_tern.ptx ternary_matmul "$N" receipt_emit "$M" "$K" "$N" /tmp/rc_rcpt.txt >/tmp/rc_emit.log 2>&1
if [ ! -s /tmp/rc_rcpt.txt ] || [ ! -s /tmp/rc_rcpt.txt.cbytes ]; then bad "genuine receipt not emitted"; tail -4 /tmp/rc_emit.log >&2; echo "RECEIPT_CHECK_FAIL"; exit 1; fi
GEN_HC=$(grep '^H_C=' /tmp/rc_rcpt.txt)
VP=$(vchk /tmp/rc_rcpt.txt)
if [ "$VP" = RECEIPT_PASS ]; then say "    genuine receipt_check=RECEIPT_PASS  OK"; else bad "genuine receipt_check=$VP (expected RECEIPT_PASS)"; fi

# --- [E] NEGATIVE CONTROLS (each must be REJECTED) ---
say "[E] negative controls (each must be REJECTED)"

# NC1 forged-output (Freivalds teeth): a SELF-CONSISTENT forged receipt (C[0]+1; H_C/cbytes/echo all match
#     the forged C) -> CHECK1/CHECK3 pass, but C != W*X so the Freivalds leg MUST reject.
"$CL" /tmp/rc_tern.ptx ternary_matmul "$N" receipt_emit "$M" "$K" "$N" /tmp/rc_nc1.txt mutate >/tmp/rc_nc1emit.log 2>&1
if [ ! -s /tmp/rc_nc1.txt ]; then bad "NC1 receipt not emitted"; else
  if [ "$(grep '^H_C=' /tmp/rc_nc1.txt)" = "$GEN_HC" ]; then bad "NC1 vacuous -- mutate did not change C (H_C identical to genuine)"; else nc "NC1 forged-output" /tmp/rc_nc1.txt; fi
fi

# NC2 tampered-hash-binding: flip the first hex char of H_C (cbytes genuine) -> CHECK3 (recomputed C hash
#     != receipt H_C) MUST reject.
cp /tmp/rc_rcpt.txt /tmp/rc_nc2.txt; cp /tmp/rc_rcpt.txt.cbytes /tmp/rc_nc2.txt.cbytes
sed -i 's/^H_C=./H_C=z/' /tmp/rc_nc2.txt
if [ "$(grep '^H_C=' /tmp/rc_nc2.txt)" = "$GEN_HC" ]; then bad "NC2 vacuous -- H_C flip did not change the line"; else nc "NC2 tampered-hash-bind" /tmp/rc_nc2.txt; fi

# NC3 echo-not-trusted: take the forged NC1 receipt + DOCTOR the round echoes to fake self-agreement
#     (lhs=1 rhs=1). The checker IGNORES the echo, re-derives r, and Freivalds still rejects -> proves r
#     is checker-derived (closes the runner-chosen-challenge hole).
cp /tmp/rc_nc1.txt /tmp/rc_nc3.txt; cp /tmp/rc_nc1.txt.cbytes /tmp/rc_nc3.txt.cbytes
sed -i 's/^round \([0-9]*\) lhs=.*/round \1 lhs=1 rhs=1/' /tmp/rc_nc3.txt
if grep -q '^round 0 lhs=1 rhs=1' /tmp/rc_nc3.txt; then nc "NC3 echo-not-trusted" /tmp/rc_nc3.txt; else bad "NC3 vacuous -- round-echo doctor did not apply"; fi

# NC4 tampered-weight-commitment: flip seed_W -> the checker regenerates a different W whose hash != H_W
#     -> CHECK1 MUST reject (model-identity binding).
cp /tmp/rc_rcpt.txt /tmp/rc_nc4.txt; cp /tmp/rc_rcpt.txt.cbytes /tmp/rc_nc4.txt.cbytes
sed -i 's/^seed_W=0/seed_W=5/' /tmp/rc_nc4.txt
if grep -q '^seed_W=5' /tmp/rc_nc4.txt; then nc "NC4 tampered-weight" /tmp/rc_nc4.txt; else bad "NC4 vacuous -- seed_W flip did not apply"; fi

# NC5 vacuity: a truncated receipt (header only, no commitments) MUST fail-closed, not silently pass.
printf 'HELIX_RECEIPT_V1\nshape M=16 K=16 N=16\n' > /tmp/rc_nc5.txt; : > /tmp/rc_nc5.txt.cbytes
nc "NC5 vacuity" /tmp/rc_nc5.txt

# NC6 kernel-corruption: re-emit a DROP-TERM ternary_matmul from /tmp (committed kernel NEVER edited),
#     emit a receipt over the WRONG C -> Freivalds MUST reject (ties the receipt to the genuine compute).
sed 's/let mut t = 1;/let mut t = 2;/' "$KERN" > /tmp/rc_badkern.hx
if cmp -s /tmp/rc_badkern.hx "$KERN"; then bad "NC6 vacuous -- drop-term sed did not match the committed kernel"; else
  emit /tmp/rc_badkern.hx
  if [ ! -s /tmp/out.ptx ]; then bad "NC6 corrupted kernel emitted no PTX"; else
    cp /tmp/out.ptx /tmp/rc_bad.ptx
    "$CL" /tmp/rc_bad.ptx ternary_matmul "$N" receipt_emit "$M" "$K" "$N" /tmp/rc_nc6.txt >/tmp/rc_nc6emit.log 2>&1
    if [ ! -s /tmp/rc_nc6.txt ]; then bad "NC6 receipt not emitted"; else
      if [ "$(grep '^H_C=' /tmp/rc_nc6.txt)" = "$GEN_HC" ]; then bad "NC6 vacuous -- corrupted kernel produced the SAME C as genuine"; else nc "NC6 kernel-corruption" /tmp/rc_nc6.txt; fi
    fi
  fi
fi
# restore the genuine PTX in /tmp (NC6 overwrote /tmp/out.ptx; the committed kernel was never touched)
emit "$KERN" >/dev/null 2>&1 || true

echo "------------------------------------------------------------"
if [ "$OK" = "1" ]; then echo "RECEIPT_CHECK_PASS"; exit 0; else echo "RECEIPT_CHECK_FAIL"; exit 1; fi
