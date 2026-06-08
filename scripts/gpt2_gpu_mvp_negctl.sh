#!/usr/bin/env bash
# Negative controls for the GPT-2 GPU MVP gates (scripts/gpt2_gpu_mvp.sh): prove the gates are
# LOAD-BEARING -- a deliberately-broken expectation MUST produce FAIL + a nonzero exit. If a broken
# variant PASSES, the parity claim is void. Reuses the ext4-minted driver/PTX + the built launcher.
# Run as a FILE under WSL (bash -c loops mangle the loop var here; files do not).
set -u
WORK=/home/legoa/gpt2_ext4/Kovostov-Native
BS=$WORK/stage0/helixc-bootstrap
EXMNT=/mnt/c/Projects/Kovostov-Native/helixc/examples
WEIGHTS=/home/legoa/gpt2_ext4/gpt2_124M.weights
REFDIR=/mnt/c/Projects/Kovostov-Native/helix-llm/ref
FAILS=0; CHECKS=0

echo "=== rebuild driver + PTX + launcher in this session (fast, ext4) ==="
cd "$BS"
( ulimit -s unlimited; ./seed.bin k1ptxdrv.hx /tmp/nc_drv.bin ) >/tmp/nc_mint.log 2>&1; chmod +x /tmp/nc_drv.bin
: > /tmp/nc_ki.hx
for k in tiled_matmul tiled_matmul_abt gpu_softmax_causal gpu_layernorm_fwd_eps gpu_add_bias_rowbcast gpu_gelu_stable vector_add gpu_scale_rt; do
  tr -d '\r' < "$EXMNT/${k}_kernel.hx" >> /tmp/nc_ki.hx; printf '\n' >> /tmp/nc_ki.hx
done
cp /tmp/nc_ki.hx /tmp/kernel_in.hx
rm -f /tmp/out.ptx; ( ulimit -s unlimited; /tmp/nc_drv.bin ) >/tmp/nc_emit.log 2>&1; cp /tmp/out.ptx /tmp/nc.ptx
cp /mnt/c/Projects/Kovostov-Native/helixc/runtime/gpt2_infer.c "$WORK/helixc/runtime/gpt2_infer.c"
cd "$WORK/helixc/runtime"
gcc gpt2_infer.c -O2 -I/usr/local/cuda/include -L/usr/lib/wsl/lib -lcuda -lm -o /tmp/nc_gi 2>/tmp/nc_gcc.log
echo "  launcher $(stat -c%s /tmp/nc_gi) B; ptx $(grep -c '\.entry' /tmp/nc.ptx) entries"
printf '464 3139 286 4881 318\n' > /tmp/nc_ids.txt

# ---- NC-A: wrong expected argmax -> GATE 3 must FAIL closed ----
echo "=== NC-A: --logits with a WRONG expected argmax (999999) -> expect FAIL + nonzero exit ==="
echo 999999 > /tmp/nc_bad_am.txt
/tmp/nc_gi /tmp/nc.ptx "$WEIGHTS" --logits "$REFDIR/ref_logits_last.bin" /tmp/nc_bad_am.txt /tmp/nc_ids.txt >/tmp/nc_a.log 2>&1; ax=$?
grep -E 'argmax=|GPT2_LOGITS' /tmp/nc_a.log | sed 's/^/    /'
CHECKS=$((CHECKS+1))
if grep -q '^GPT2_LOGITS_PARITY_FAIL' /tmp/nc_a.log && [ "$ax" -ne 0 ]; then echo "    NC-A OK (failed closed, exit=$ax)"; else echo "    NC-A BROKEN (did not fail closed; exit=$ax)"; FAILS=$((FAILS+1)); fi

# ---- NC-B: wrong oracle gen ids -> GATE 4 must FAIL closed ----
echo "=== NC-B: --generate with WRONG oracle gen ids -> expect FAIL + nonzero exit ==="
printf '464 3139 286 4881 318 1 2 3\n' > /tmp/nc_bad_gen.txt
/tmp/nc_gi /tmp/nc.ptx "$WEIGHTS" --generate 3 /tmp/nc_ids.txt /tmp/nc_bad_gen.txt >/tmp/nc_b.log 2>&1; bx=$?
grep -E 'TOKEN_|GPT2_GENERATE' /tmp/nc_b.log | sed 's/^/    /'
CHECKS=$((CHECKS+1))
if grep -q '^GPT2_GENERATE_MATCH_FAIL' /tmp/nc_b.log && [ "$bx" -ne 0 ]; then echo "    NC-B OK (failed closed, exit=$bx)"; else echo "    NC-B BROKEN (did not fail closed; exit=$bx)"; FAILS=$((FAILS+1)); fi

# ---- NC-C (control): the REAL expectations still PASS (the gate is not stuck-failing) ----
echo "=== NC-C: --logits with the REAL oracle argmax -> expect PASS (sanity that gate is not stuck-fail) ==="
/tmp/nc_gi /tmp/nc.ptx "$WEIGHTS" --logits "$REFDIR/ref_logits_last.bin" "$REFDIR/ref_argmax.txt" /tmp/nc_ids.txt >/tmp/nc_c.log 2>&1; cx=$?
grep -E 'argmax=|GPT2_LOGITS' /tmp/nc_c.log | sed 's/^/    /'
CHECKS=$((CHECKS+1))
if grep -q '^GPT2_LOGITS_PARITY_PASS' /tmp/nc_c.log && [ "$cx" -eq 0 ]; then echo "    NC-C OK (clean pipeline passes, exit=$cx)"; else echo "    NC-C BROKEN (clean pipeline did not pass; exit=$cx)"; FAILS=$((FAILS+1)); fi

echo "=================== NEG-CTRL VERDICT ==================="
echo "  $((CHECKS-FAILS))/$CHECKS controls behaved correctly"
if [ "$FAILS" -eq 0 ]; then echo "GPT2_MVP_NEGCTL_PASS"; exit 0; else echo "GPT2_MVP_NEGCTL_FAIL"; exit 1; fi
