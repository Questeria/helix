#!/usr/bin/env bash
# P2 gate: GPT-2 124M CPU BLOCK-0 HIDDEN PARITY through kovc-compiled pure Helix,
# with NO ptxas / NO GPU boundary. The purest-trust artifact.
#
# Modeled on scripts/gpt2_gpu_parity.sh but with ZERO GPU: mint the self-hosted
# kovc FRESH from the 299-byte raw seed (seed -> K1 -> K2, the full-language
# self-host compiler), compile the pure-Helix op-dispatch ELF
# (helixc/runtime/gpt2_cpu_ops.hx) with that kovc, build the CUDA-FREE
# byte-movement harness (helixc/runtime/cpu_host.c), run GPT-2 block 0 for the
# canonical prompt, and compare the post-block-0 hidden [5,768] to
# helix-llm/ref/ref_block0.npy at max_abs<1e-3 AND mean_abs<1e-4.
# (The .hx + .c are the COMMITTED tracked sources; only the model weights +
#  the numpy-oracle ref remain fenced under helix-llm/.)
#
#   bash scripts/gpt2_cpu_parity.sh
#
# Emits GPT2_CPU_BLOCK0_PARITY_PASS / GPT2_CPU_BLOCK0_PARITY_FAIL and propagates
# the verdict to the PROCESS EXIT STATUS (fail-closed: a printed FAIL is never
# exit 0). ALL ARITHMETIC stays in the kovc-emitted Helix ELF; the C harness does
# only byte-movement (mmap weights, embedding gather, head pack/scatter, GEMM
# N-tiling, per-op file staging). STRICTLY SERIAL (one kovc/build at a time).
#
# Speed: the build mints kovc from the raw seed (seed->K1->K2). Set CPU_BUILD_DIR
# to an ext4 path (default ~/gpt2cpu/bs) to avoid DrvFs 1-byte-write slowness;
# the seed+sources are copied there. The trust anchor is unchanged (same seed.bin).
set -u
ROOT=/mnt/c/Projects/Kovostov-Native
BS=$ROOT/stage0/helixc-bootstrap
# CANONICAL committed CPU-path sources now live under helixc/runtime/ (mirroring the
# committed GPU launcher helixc/runtime/gpt2_infer.c). The model weights + the numpy
# oracle reference stay fenced under helix-llm/ (large/independent-verification only).
SRC=$ROOT/helixc/runtime
WEIGHTS=$ROOT/helix-llm/models/gpt2/gpt2_124M.weights
REF=$ROOT/helix-llm/ref/ref_block0.npy
BD=${CPU_BUILD_DIR:-$HOME/gpt2cpu/bs}
WD=${CPU_WORK_DIR:-$HOME/gpt2cpu/work}
OK=1
echo "=================== GPT-2 CPU (no-ptxas) BLOCK-0 PARITY  $(date -u +%H:%M:%S) ==================="

# ---- [0] inputs present ----
[ -s "$WEIGHTS" ] || { echo "PARITY FAIL: missing weight file $WEIGHTS (run the P1 importer)"; OK=0; }
[ -s "$REF" ]     || { echo "PARITY FAIL: missing parity target $REF (run gpt2_numpy_ref.py)"; OK=0; }
[ -s "$SRC/gpt2_cpu_ops.hx" ] || { echo "PARITY FAIL: missing $SRC/gpt2_cpu_ops.hx"; OK=0; }
[ -s "$SRC/cpu_host.c" ]      || { echo "PARITY FAIL: missing $SRC/cpu_host.c"; OK=0; }
echo "  weights $( [ -s "$WEIGHTS" ] && stat -c%s "$WEIGHTS" || echo MISSING ) B ; ref $( [ -s "$REF" ] && stat -c%s "$REF" || echo MISSING ) B"

# ---- [1] mirror seed + bootstrap sources to the ext4 build dir + assemble ----
echo "=== [1] mirror seed + sources to ext4 ($BD) and assemble k1src/k1input ==="
mkdir -p "$BD/drivers"
cp "$BS/seed.bin" "$BD/seed.bin" && chmod +x "$BD/seed.bin"
cp "$ROOT/helixc/bootstrap/lexer.hx"  "$BD/lexer.hx"
cp "$ROOT/helixc/bootstrap/parser.hx" "$BD/parser.hx"
cp "$ROOT/helixc/bootstrap/kovc.hx"   "$BD/kovc.hx"
cp "$BS/drivers/driver_k1src.hx"   "$BD/drivers/"
cp "$BS/drivers/driver_k1input.hx" "$BD/drivers/"
SEEDSHA=$(sha256sum "$BD/seed.bin" | cut -c1-16)
echo "  seed sha=$SEEDSHA (trust anchor; expect 9837db12752a2215...)"

# ext4-path assemble variant (reads/writes ext4; output k1src/k1input are gitignored artifacts).
cat > "$BD/assemble_ext4.hx" <<HXEOF
fn find_keep_len(base: i32, len: i32) -> i32 {
    let mut demo_at = 0 - 1;
    let mut i = 1;
    while i + 8 <= len {
        if __arena_get(base + i - 1) == 10 {
            let mut ok = 1;
            if __arena_get(base + i) != 47 { ok = 0; }
            if __arena_get(base + i + 1) != 47 { ok = 0; }
            if __arena_get(base + i + 2) != 32 { ok = 0; }
            if __arena_get(base + i + 3) != 68 { ok = 0; }
            if __arena_get(base + i + 4) != 101 { ok = 0; }
            if __arena_get(base + i + 5) != 109 { ok = 0; }
            if __arena_get(base + i + 6) != 111 { ok = 0; }
            if __arena_get(base + i + 7) != 58 { ok = 0; }
            if ok == 1 { if demo_at < 0 { demo_at = i; } }
        }
        i = i + 1;
    }
    let mut j = demo_at - 2;
    let mut prev_nl = 0 - 1;
    while j >= 0 {
        if __arena_get(base + j) == 10 { prev_nl = j; j = 0 - 1; } else { j = j - 1; }
    }
    prev_nl + 1
}
fn append_stripped(base: i32, len: i32) -> i32 {
    let mut i = 0;
    while i < len {
        let b = __arena_get(base + i);
        if b != 13 { __arena_push(b); }
        i = i + 1;
    }
    0
}
fn main() -> i32 {
    let lex_base = __arena_len();
    let lex_len = read_file_to_arena("$BD/lexer.hx");
    let par_base = __arena_len();
    let par_len = read_file_to_arena("$BD/parser.hx");
    let kov_base = __arena_len();
    let kov_len = read_file_to_arena("$BD/kovc.hx");
    let lex_keep = find_keep_len(lex_base, lex_len);
    let kov_keep = find_keep_len(kov_base, kov_len);
    let d1_base = __arena_len();
    let d1_len = read_file_to_arena("$BD/drivers/driver_k1src.hx");
    let o1_base = __arena_len();
    append_stripped(lex_base, lex_keep);
    append_stripped(par_base, par_len);
    append_stripped(kov_base, kov_keep);
    append_stripped(d1_base, d1_len);
    let o1_len = __arena_len() - o1_base;
    write_file_to_arena("$BD/k1src.hx", o1_base, o1_len);
    let d2_base = __arena_len();
    let d2_len = read_file_to_arena("$BD/drivers/driver_k1input.hx");
    let o2_base = __arena_len();
    append_stripped(lex_base, lex_keep);
    append_stripped(par_base, par_len);
    append_stripped(kov_base, kov_keep);
    append_stripped(d2_base, d2_len);
    let o2_len = __arena_len() - o2_base;
    write_file_to_arena("$BD/k1input.hx", o2_base, o2_len);
    0
}
HXEOF
( cd "$BD" && ./seed.bin assemble_ext4.hx asm.bin && chmod +x asm.bin && ./asm.bin )
if [ ! -s "$BD/k1src.hx" ] || [ ! -s "$BD/k1input.hx" ]; then echo "  ASSEMBLE FAIL (no k1src/k1input)"; OK=0; fi
echo "  k1src=$(stat -c%s "$BD/k1src.hx" 2>/dev/null) k1input=$(stat -c%s "$BD/k1input.hx" 2>/dev/null)"

# ---- [2] seed -> K1 -> K2 (the full-language self-hosted kovc) ----
echo "=== [2] mint kovc from the raw seed: seed -> K1 -> K2 ==="
rm -f "$BD/K1.bin" /tmp/gpc_k1_out.bin
( cd "$BD" && ulimit -s unlimited; timeout 1200 ./seed.bin k1src.hx K1.bin ) ; r1=$?
chmod +x "$BD/K1.bin" 2>/dev/null
if [ "$r1" -ne 0 ] || [ ! -s "$BD/K1.bin" ]; then echo "  K1 FAIL (seed->K1 rc=$r1)"; OK=0; fi
echo "  K1=$(stat -c%s "$BD/K1.bin" 2>/dev/null) B"
# K1 reads /tmp/k1_in.hx -> writes /tmp/k1_out.bin (baked paths inside k1input.hx).
cp "$BD/k1input.hx" /tmp/k1_in.hx; rm -f /tmp/k1_out.bin
( cd "$BD" && ulimit -s unlimited; timeout 300 ./K1.bin ) ; r2=$?
if [ ! -s /tmp/k1_out.bin ]; then echo "  K2 FAIL (K1->K2 rc=$r2, no /tmp/k1_out.bin)"; OK=0; fi
cp /tmp/k1_out.bin "$BD/K2.bin" 2>/dev/null && chmod +x "$BD/K2.bin"
echo "  K2=$(stat -c%s "$BD/K2.bin" 2>/dev/null) B sha=$(sha256sum "$BD/K2.bin" 2>/dev/null | cut -c1-16)"

# ---- [3] compile the pure-Helix op ELF with the seed-minted kovc (K2) ----
echo "=== [3] compile gpt2_cpu_ops.hx -> ops ELF via the seed-minted kovc (K2, baked-path) ==="
mkdir -p "$WD"
# K2 compiles the file staged at /tmp/k2_in.hx -> /tmp/k2_out.bin.
tr -d '\r' < "$SRC/gpt2_cpu_ops.hx" > /tmp/k2_in.hx
rm -f /tmp/k2_out.bin
( cd "$BD" && ulimit -s unlimited; timeout 120 ./K2.bin ) ; r3=$?
if [ ! -s /tmp/k2_out.bin ]; then echo "  OP COMPILE FAIL (K2 rc=$r3, no /tmp/k2_out.bin)"; OK=0; fi
cp /tmp/k2_out.bin "$WD/gpt2_cpu_ops.bin" 2>/dev/null && chmod +x "$WD/gpt2_cpu_ops.bin"
echo "  ops ELF=$(stat -c%s "$WD/gpt2_cpu_ops.bin" 2>/dev/null) B"

# ---- [4] build the CUDA-free byte-movement harness ----
echo "=== [4] build cpu_host.c (CUDA-free; byte-movement only) ==="
rm -f "$WD/cpu_host"
gcc "$SRC/cpu_host.c" -O2 -lm -o "$WD/cpu_host" 2>/tmp/gpc_gcc.log \
  || { echo "  GCC FAIL"; sed 's/^/    /' /tmp/gpc_gcc.log; OK=0; }
[ -x "$WD/cpu_host" ] && echo "  built $WD/cpu_host" || { echo "  no harness binary"; OK=0; }

# ---- [5] run BLOCK-0 parity (stale-artifact guard: rm dump before, require fresh) ----
echo "=== [5] run CPU block-0 parity (canonical prompt; max_abs<1e-3 AND mean_abs<1e-4) ==="
mkdir -p /tmp/gpc
rm -f /tmp/gpc/helix_block0.bin
if [ -x "$WD/cpu_host" ] && [ -s "$WD/gpt2_cpu_ops.bin" ] && [ -s "$WEIGHTS" ] && [ -s "$REF" ]; then
  "$WD/cpu_host" "$WEIGHTS" "$WD/gpt2_cpu_ops.bin" --block0 "$REF" > /tmp/gpc_run.log 2>&1 ; prc=$?
  sed 's/^/    /' /tmp/gpc_run.log
  if [ ! -s /tmp/gpc/helix_block0.bin ]; then echo "  PARITY FAIL: run left no /tmp/gpc/helix_block0.bin (rc=$prc)"; OK=0;
  else echo "  fresh artifact: /tmp/gpc/helix_block0.bin ($(stat -c%s /tmp/gpc/helix_block0.bin) B)"; fi
  if ! grep -q '^GPT2_CPU_BLOCK0_PARITY_PASS' /tmp/gpc_run.log; then echo "  PARITY FAIL (no PASS line; rc=$prc)"; OK=0; fi
else
  echo "  PARITY FAIL: prerequisites missing"; OK=0
fi

echo "=================================================================="
if [ "$OK" = "1" ]; then echo "GPT2_CPU_BLOCK0_PARITY_PASS"; exit 0; else echo "GPT2_CPU_BLOCK0_PARITY_FAIL"; exit 1; fi
