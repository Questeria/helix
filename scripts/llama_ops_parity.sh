#!/usr/bin/env bash
# llama_ops_parity.sh -- G-L0 kernel parity gate for the Llama-arch leg (docs/HELIX_LLAMA_PLAN.md
# section 6.1). Run as a FILE under WSL (CRLF-stripped to /tmp first):
#   wsl.exe bash -lc "tr -d '\r' < /mnt/c/Projects/Kovostov-Native/scripts/llama_ops_parity.sh > /tmp/llama_ops_parity.sh && bash /tmp/llama_ops_parity.sh"
#
# FAIL-CLOSED per-kernel parity for the 3 NEW Llama kernels (committed, UNVERIFIED on real HW):
#   gpu_rmsnorm_fwd_eps   (helixc/examples/gpu_rmsnorm_fwd_eps_kernel.hx)  -- RMSNorm, eps 1e-5 baked
#   gpu_rope_rot          (helixc/examples/gpu_rope_rot_kernel.hx)         -- HF rotate_half RoPE, in-place
#   gpu_silu_mul          (helixc/examples/gpu_silu_mul_kernel.hx)         -- SwiGLU gate y=u*silu(g)
#
# It REUSES THE EXACT GPT-2 PRECEDENT MACHINERY (gpu_elementwise_corpus.sh / gpu_reduction_corpus.sh):
#   from-raw kovc  ->  emit PTX  ->  ptxas sm_86  ->  cuLaunchKernel with host I/O  ->  compare max-abs.
# The from-raw kovc is the seed-built K1 PTX driver (seed 9837db12 -> assemble_k1 -> k1ptxdrv).
#
# REFERENCE TRUTH = the INDEPENDENT numpy oracle helix-llm/tools/llama_ops_numpy_ref.py (uncommitted,
# like the GPT-2 oracle). A small Python driver IMPORTS the oracle's PINNED ops verbatim (rmsnorm /
# rope_tables / rope_rot / silu_mul) -- the oracle is NOT modified -- generates the random inputs at
# SmolLM2-135M dims, computes the reference output, then re-reads the GPU output and reports max-abs
# error. fp32 tol = 1e-4 (same order as the GPT-2 ops). The GPU launcher is a SCRATCH C shim written
# to /tmp (uncommitted, like the oracle -- NO committed .c/.h touched, so the fence count is unchanged):
# it mirrors cuda_launch.c's cuInit/cuModuleLoad/cuMemAlloc/cuLaunchKernel/cuMemcpyDtoH idiom exactly,
# but is data-driven (reads inputs from a .bin, writes the GPU result to a .bin) so the oracle is the
# SOLE source of reference truth.
#
# SmolLM2-135M dims (docs/HELIX_LLAMA_PLAN.md section 1; G-L0 needs NO model download -- random inputs):
#   d_model 576, n_q_heads 9, n_kv_heads 3, head_dim 64, d_ff 1536, rms_eps 1e-5, rope_theta 1e5, seq S.
#   rmsnorm : rows=S, cols=d_model=576
#   rope    : rows = S * n_q_heads (each q head a packed [S, head_dim] block), half = head_dim/2 = 32
#   silu_mul: n = S * d_ff (one SwiGLU intermediate row per position)
#
# NEG-CONTROLS (comparator must have teeth): MUTATE perturbs one GPU-output cell pre-compare -> MUST FAIL.
#
# Prints LLAMA_GL0_PASS / LLAMA_GL0_FAIL and a per-kernel table. Reference box: RTX 3070 (sm_86). SERIAL.
set -u
T0=$(date +%s)
ROOT="${HELIX_SRC:-}"; if [ -z "$ROOT" ]; then ROOT="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"; fi
[ -d "$ROOT/helixc/examples" ] || ROOT="/mnt/c/Projects/Kovostov-Native"
cd "$ROOT" || { echo "FATAL: no repo root"; exit 9; }
EX="$ROOT/helixc/examples"
ORACLE_DIR="$ROOT/helix-llm/tools"
OUT="$ROOT/.m1probe"
mkdir -p "$OUT"
HB="$ROOT/stage0/helixc-bootstrap"
WORK="${HELIX_WORK:-$HOME/gpt2_ext4/Kovostov-Native}"   # fast ext4 build mirror (proven serve_chat_demo.sh pattern)
BS_W="$WORK/stage0/helixc-bootstrap"
DRV="/tmp/llama_kovc_drv.bin"
PTXAS="${PTXAS:-/usr/local/cuda-12.8/bin/ptxas}"   # kovc emits .version 8.3 (TF32 mma); 12.0 ptxas rejects 8.3
[ -x "$PTXAS" ] || PTXAS="/usr/local/cuda/bin/ptxas"
REMINT="${REMINT:-1}"
S="${S:-7}"                     # small seq len (G-L0 is per-kernel random-input parity, not a model run)
DM="${DM:-576}"                 # d_model (SmolLM2-135M)
NQH="${NQH:-9}"                 # n_q_heads
HD="${HD:-64}"                  # head_dim
DFF="${DFF:-1536}"              # d_ff
THETA="${THETA:-100000.0}"      # rope_theta (SmolLM2)
TOL="${TOL:-1e-4}"
RC=0
WRK=/tmp/llama_gl0; mkdir -p "$WRK"

echo "=================== HELIX LLAMA G-L0 KERNEL PARITY  $(date -u +%H:%M:%S) ==================="
echo "  root=$ROOT  ptxas=$PTXAS  dims: S=$S d_model=$DM n_q_heads=$NQH head_dim=$HD d_ff=$DFF rope_theta=$THETA tol=$TOL"

# Oracle must self-test green before we trust it as the reference.
echo "=== [0] oracle self-test (llama_ops_numpy_ref.py) ==="
if python3 "$ORACLE_DIR/llama_ops_numpy_ref.py" 2>&1 | tee "$OUT/llama_oracle_selftest.log" | grep -q '^LLAMA_OPS_REF_SELFTEST: PASS'; then
  echo "  oracle LLAMA_OPS_REF_SELFTEST: PASS"
else
  echo "  FAIL: oracle self-test did not PASS -- refusing to gate against an unverified reference"; RC=9
  echo "LLAMA_GL0_FAIL"; exit "$RC"
fi

# =================== [A]/[B] from-raw kovc: mint the K1 PTX driver via the PROVEN ext4 + path-rewrite ===================
# serve_chat_demo.sh's EXACT mint. In-place on /mnt/c without ulimit/ext4/timeout hangs the assemble step.
if [ "$REMINT" = "1" ]; then
  echo "=== [A] regenerate k1ptxdrv.hx from current kovc.hx (assemble_k1; ext4 mirror + path-rewrite) ==="
  mkdir -p "$BS_W" "$WORK/helixc/bootstrap"
  cp -r "$ROOT/stage0/helixc-bootstrap/." "$BS_W"/
  cp "$ROOT/helixc/bootstrap/lexer.hx" "$ROOT/helixc/bootstrap/parser.hx" "$ROOT/helixc/bootstrap/kovc.hx" "$WORK/helixc/bootstrap/"
  sed -i "s#/mnt/c/Projects/Kovostov-Native/#$WORK/#g" "$BS_W/assemble_k1.hx"
  _seedsha=$(sha256sum "$BS_W/seed.bin" 2>/dev/null | cut -c1-8)
  [ "$_seedsha" = "9837db12" ] || { echo "FATAL: ext4 seed sha $_seedsha != 9837db12"; echo "LLAMA_GL0_FAIL"; exit 7; }
  cd "$BS_W" || { echo "FATAL: no $BS_W"; echo "LLAMA_GL0_FAIL"; exit 7; }
  rm -f /tmp/asm_k1_ll.bin "$DRV" /tmp/out.ptx
  ( ulimit -s unlimited; timeout 600 ./seed.bin assemble_k1.hx /tmp/asm_k1_ll.bin ) || { echo "FATAL assemble_k1 (seed emit)"; echo "LLAMA_GL0_FAIL"; exit 7; }
  chmod +x /tmp/asm_k1_ll.bin
  ( ulimit -s unlimited; timeout 600 /tmp/asm_k1_ll.bin ) || { echo "FATAL assemble_k1 (concat run)"; echo "LLAMA_GL0_FAIL"; exit 7; }
  echo "  k1ptxdrv.hx regenerated ($(stat -c%s "$BS_W/k1ptxdrv.hx" 2>/dev/null) bytes)"
  echo "=== [B] seed -> K1 PTX driver build (from-raw kovc; the slow ~3min step) ==="
  TB=$(date +%s)
  ( ulimit -s unlimited; timeout 1200 ./seed.bin k1ptxdrv.hx "$DRV" ) || { echo "FATAL: seed could not compile k1ptxdrv.hx"; echo "LLAMA_GL0_FAIL"; exit 6; }
  chmod +x "$DRV"
  echo "  from-raw kovc PTX driver built in $(( $(date +%s) - TB ))s ($(stat -c%s "$DRV") bytes)"
fi
[ -x "$DRV" ] || { echo "FATAL: no PTX driver (run with REMINT=1)"; echo "LLAMA_GL0_FAIL"; exit 7; }

# =================== [1] emit ONE combined PTX with the 3 Llama kernels ===================
echo "=== [1] emit combined.ptx (gpu_rmsnorm_fwd_eps + gpu_rope_rot + gpu_silu_mul) via from-raw kovc ==="
: > /tmp/kernel_in.hx
RMSN_OK=0; ROPE_OK=0; SILU_OK=0
for k in gpu_rmsnorm_fwd_eps_kernel gpu_rope_rot_kernel gpu_silu_mul_kernel; do
  [ -f "$EX/${k}.hx" ] || { echo "  FATAL: missing kernel source $EX/${k}.hx"; RC=6; }
  tr -d '\r' < "$EX/${k}.hx" >> /tmp/kernel_in.hx; echo "" >> /tmp/kernel_in.hx
done
rm -f /tmp/out.ptx
"$DRV" >/dev/null 2>&1 || true
if [ ! -s /tmp/out.ptx ]; then
  echo "  FATAL: from-raw kovc emitted NO /tmp/out.ptx (compile of the 3 kernels failed)"; echo "LLAMA_GL0_FAIL"; exit 6
fi
cp /tmp/out.ptx "$OUT/llama_ops_combined.ptx"
PTX_SHA=$(sha256sum /tmp/out.ptx | cut -c1-12)
NENT=$(grep -c '\.entry' /tmp/out.ptx)
echo "  emitted $(stat -c%s /tmp/out.ptx) bytes, $NENT .entry kernels, sha256=$PTX_SHA -> $OUT/llama_ops_combined.ptx"
# Per-kernel "compiled via kovc?" = its .entry is present in the from-raw-kovc-emitted PTX.
grep -q '\.entry gpu_rmsnorm_fwd_eps' /tmp/out.ptx && { RMSN_OK=1; echo "  .entry gpu_rmsnorm_fwd_eps PRESENT (compiled via from-raw kovc)"; } || echo "  MISSING .entry gpu_rmsnorm_fwd_eps"
grep -q '\.entry gpu_rope_rot'        /tmp/out.ptx && { ROPE_OK=1; echo "  .entry gpu_rope_rot PRESENT (compiled via from-raw kovc)"; }        || echo "  MISSING .entry gpu_rope_rot"
grep -q '\.entry gpu_silu_mul'        /tmp/out.ptx && { SILU_OK=1; echo "  .entry gpu_silu_mul PRESENT (compiled via from-raw kovc)"; }        || echo "  MISSING .entry gpu_silu_mul"

echo "=== [2] PTX provenance (grep the OUTPUT, never source; informational -- ptxas[3] is the gate) ==="
grep -q '\.target sm_86'     /tmp/out.ptx && echo "  .target sm_86 PRESENT"                         || echo "  NOTE: no '.target sm_86' literal (ptxas -arch=sm_86 in [3] is the real acceptance gate)"
grep -q 'rsqrt\.approx\.f32' /tmp/out.ptx && echo "  rsqrt.approx.f32 PRESENT (rmsnorm 1/sqrt)"     || echo "  NOTE: no rsqrt.approx.f32 in OUTPUT (rmsnorm uses __gpu_rsqrt)"
grep -q 'ex2\.approx\.f32'   /tmp/out.ptx && echo "  ex2.approx.f32 PRESENT (silu_mul exp)"         || echo "  NOTE: no ex2.approx.f32 in OUTPUT (silu_mul uses __gpu_exp)"

# =================== [3] ptxas acceptance at sm_86 (per the gate; 12.8 ptxas for .version 8.3) ===================
echo "=== [3] ptxas acceptance (sm_86) ==="
PTXAS_RMSN=0; PTXAS_ROPE=0; PTXAS_SILU=0
if "$PTXAS" -arch=sm_86 -v /tmp/out.ptx -o "$OUT/llama_ops_combined.cubin" 2>&1 | tee "$OUT/llama_ptxas.log"; then
  echo "  PTXAS_ACCEPT (sm_86, $PTXAS) -- all 3 kernels in one module"
  PTXAS_RMSN=1; PTXAS_ROPE=1; PTXAS_SILU=1
else
  echo "  PTXAS_REJECT (sm_86) -- see $OUT/llama_ptxas.log"; RC=2
fi

# =================== [4] build the SCRATCH GPU launcher (uncommitted /tmp shim; mirrors cuda_launch.c) ===================
echo "=== [4] build scratch GPU launcher /tmp/llama_cl (cuLaunch idiom from cuda_launch.c; uncommitted) ==="
cat > "$WRK/llama_cl.c" <<'CEOF'
/* llama_cl.c -- SCRATCH (uncommitted, /tmp) data-driven GPU launcher for G-L0 Llama-kernel parity.
 * Mirrors helixc/runtime/cuda_launch.c's cuInit/cuModuleLoad/cuModuleGetFunction/cuMemAlloc/
 * cuLaunchKernel/cuMemcpyDtoH idiom EXACTLY, but reads ALL host I/O from binary files so the
 * numpy oracle (llama_ops_numpy_ref.py) is the sole source of reference truth. NO comparison
 * is done here (the oracle owns the reference); this shim only runs the kernel and dumps output.
 *
 *   llama_cl <module.ptx> <op:rmsnorm|rope|silu_mul> <indir> <outpath> [mutate]
 *
 * File layout (little-endian f32, written by the Python driver) per op:
 *   rmsnorm : indir/dims.txt = "rows cols"; indir/x.bin [rows*cols]; indir/w.bin [cols]
 *             -> outpath y.bin [rows*cols]   (grid=rows, block=1; kernel gpu_rmsnorm_fwd_eps(x,y,w,cols))
 *   rope    : indir/dims.txt = "rows half"; indir/q.bin [rows*2half]; indir/cos.bin [rows*half]; indir/sin.bin [rows*half]
 *             -> outpath q.bin [rows*2half]  (IN-PLACE; grid=rows, block=1; gpu_rope_rot(q,cos,sin,half))
 *   silu_mul: indir/dims.txt = "n"; indir/g.bin [n]; indir/u.bin [n]
 *             -> outpath y.bin [n]           (grid=n, block=1; gpu_silu_mul(g,u,y,n))
 * mutate (optional): perturb out[0] by +1e3 after D2H so the oracle compare MUST FAIL (comparator teeth).
 */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int check(CUresult r, const char* what){ if(r!=CUDA_SUCCESS){ const char* m=0; cuGetErrorString(r,&m); fprintf(stderr,"CUDA error in %s: %s (%d)\n", what, m?m:"?", (int)r); return 1; } return 0; }
#define CK(call,what) do{ if(check((call),(what))) return 2; }while(0)
static char* slurp(const char* p, size_t* n){ FILE* f=fopen(p,"rb"); if(!f){ fprintf(stderr,"open %s\n",p); return 0; } fseek(f,0,SEEK_END); long s=ftell(f); fseek(f,0,SEEK_SET); char* b=(char*)malloc(s+1); if(fread(b,1,s,f)!=(size_t)s){ fclose(f); free(b); return 0; } b[s]=0; fclose(f); if(n)*n=(size_t)s; return b; }
static float* rd(const char* dir,const char* name,size_t nf){ char p[1024]; snprintf(p,sizeof p,"%s/%s",dir,name); size_t bs=0; float* b=(float*)slurp(p,&bs); if(!b){ return 0; } if(bs<nf*sizeof(float)){ fprintf(stderr,"%s too small: %zu < %zu\n",p,bs,nf*sizeof(float)); free(b); return 0; } return b; }
static int wr(const char* path,const float* x,size_t nf){ FILE* f=fopen(path,"wb"); if(!f){ fprintf(stderr,"open out %s\n",path); return 1; } if(fwrite(x,sizeof(float),nf,f)!=nf){ fclose(f); return 1; } fclose(f); return 0; }

int main(int argc,char** argv){
  if(argc<5){ fprintf(stderr,"usage: %s <ptx> <op> <indir> <outpath> [mutate]\n",argv[0]); return 2; }
  const char* ptxp=argv[1]; const char* op=argv[2]; const char* indir=argv[3]; const char* outp=argv[4];
  int mutate=(argc>5 && strcmp(argv[5],"mutate")==0);
  char dimp[1024]; snprintf(dimp,sizeof dimp,"%s/dims.txt",indir);
  char* dt=slurp(dimp,0); if(!dt){ fprintf(stderr,"no dims.txt\n"); return 2; }
  size_t fptx=0; char* ptx=slurp(ptxp,&fptx); if(!ptx){ fprintf(stderr,"no ptx\n"); return 2; }
  CK(cuInit(0),"cuInit"); CUdevice dev; CK(cuDeviceGet(&dev,0),"dev"); CUcontext ctx; CK(cuCtxCreate(&ctx,0,dev),"ctx");
  CUmodule mod; CK(cuModuleLoadData(&mod,ptx),"module load");

  if(strcmp(op,"rmsnorm")==0){
    int rows=0,cols=0; if(sscanf(dt,"%d %d",&rows,&cols)!=2){ fprintf(stderr,"rmsnorm dims\n"); return 2; }
    size_t ne=(size_t)rows*cols;
    float* hx=rd(indir,"x.bin",ne); float* hw=rd(indir,"w.bin",(size_t)cols); float* hy=(float*)malloc(ne*sizeof(float));
    if(!hx||!hw||!hy) return 2;
    CUfunction fn; CK(cuModuleGetFunction(&fn,mod,"gpu_rmsnorm_fwd_eps"),"get rmsnorm");
    CUdeviceptr dx,dy,dw; CK(cuMemAlloc(&dx,ne*sizeof(float)),"a x"); CK(cuMemAlloc(&dy,ne*sizeof(float)),"a y"); CK(cuMemAlloc(&dw,(size_t)cols*sizeof(float)),"a w");
    CK(cuMemcpyHtoD(dx,hx,ne*sizeof(float)),"h2d x"); CK(cuMemcpyHtoD(dw,hw,(size_t)cols*sizeof(float)),"h2d w");
    void* a[]={ &dx,&dy,&dw,&cols };
    CK(cuLaunchKernel(fn,(unsigned)rows,1,1, 1,1,1, 0,0, a,0),"launch rmsnorm");
    CK(cuCtxSynchronize(),"sync");
    CK(cuMemcpyDtoH(hy,dy,ne*sizeof(float)),"d2h y");
    if(mutate && ne>0) hy[0]+=1.0e3f;
    if(wr(outp,hy,ne)) return 2;
    printf("llama_cl rmsnorm rows=%d cols=%d -> %s%s\n",rows,cols,outp,mutate?" [MUTATED]":"");
  } else if(strcmp(op,"rope")==0){
    int rows=0,half=0; if(sscanf(dt,"%d %d",&rows,&half)!=2){ fprintf(stderr,"rope dims\n"); return 2; }
    size_t hd=(size_t)(half+half); size_t ne=(size_t)rows*hd; size_t nt=(size_t)rows*half;
    float* hq=rd(indir,"q.bin",ne); float* hc=rd(indir,"cos.bin",nt); float* hs=rd(indir,"sin.bin",nt);
    if(!hq||!hc||!hs) return 2;
    CUfunction fn; CK(cuModuleGetFunction(&fn,mod,"gpu_rope_rot"),"get rope");
    CUdeviceptr dq,dc,ds; CK(cuMemAlloc(&dq,ne*sizeof(float)),"a q"); CK(cuMemAlloc(&dc,nt*sizeof(float)),"a c"); CK(cuMemAlloc(&ds,nt*sizeof(float)),"a s");
    CK(cuMemcpyHtoD(dq,hq,ne*sizeof(float)),"h2d q"); CK(cuMemcpyHtoD(dc,hc,nt*sizeof(float)),"h2d c"); CK(cuMemcpyHtoD(ds,hs,nt*sizeof(float)),"h2d s");
    void* a[]={ &dq,&dc,&ds,&half };
    CK(cuLaunchKernel(fn,(unsigned)rows,1,1, 1,1,1, 0,0, a,0),"launch rope");
    CK(cuCtxSynchronize(),"sync");
    CK(cuMemcpyDtoH(hq,dq,ne*sizeof(float)),"d2h q");
    if(mutate && ne>0) hq[0]+=1.0e3f;
    if(wr(outp,hq,ne)) return 2;
    printf("llama_cl rope rows=%d half=%d (head_dim=%zu) -> %s%s\n",rows,half,hd,outp,mutate?" [MUTATED]":"");
  } else if(strcmp(op,"silu_mul")==0){
    int n=0; if(sscanf(dt,"%d",&n)!=1){ fprintf(stderr,"silu dims\n"); return 2; }
    size_t ne=(size_t)n;
    float* hg=rd(indir,"g.bin",ne); float* hu=rd(indir,"u.bin",ne); float* hy=(float*)malloc(ne*sizeof(float));
    if(!hg||!hu||!hy) return 2;
    CUfunction fn; CK(cuModuleGetFunction(&fn,mod,"gpu_silu_mul"),"get silu_mul");
    CUdeviceptr dg,du,dy; CK(cuMemAlloc(&dg,ne*sizeof(float)),"a g"); CK(cuMemAlloc(&du,ne*sizeof(float)),"a u"); CK(cuMemAlloc(&dy,ne*sizeof(float)),"a y");
    CK(cuMemcpyHtoD(dg,hg,ne*sizeof(float)),"h2d g"); CK(cuMemcpyHtoD(du,hu,ne*sizeof(float)),"h2d u");
    void* a[]={ &dg,&du,&dy,&n };
    CK(cuLaunchKernel(fn,(unsigned)n,1,1, 1,1,1, 0,0, a,0),"launch silu_mul");
    CK(cuCtxSynchronize(),"sync");
    CK(cuMemcpyDtoH(hy,dy,ne*sizeof(float)),"d2h y");
    if(mutate && ne>0) hy[0]+=1.0e3f;
    if(wr(outp,hy,ne)) return 2;
    printf("llama_cl silu_mul n=%d -> %s%s\n",n,outp,mutate?" [MUTATED]":"");
  } else { fprintf(stderr,"unknown op %s\n",op); return 2; }
  cuModuleUnload(mod); cuCtxDestroy(ctx);
  return 0;
}
CEOF
gcc "$WRK/llama_cl.c" -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcuda -o /tmp/llama_cl 2>"$OUT/llama_cl_gcc.log" \
  || { echo "  FATAL gcc llama_cl:"; sed 's/^/    /' "$OUT/llama_cl_gcc.log"; echo "LLAMA_GL0_FAIL"; exit 2; }
echo "  built /tmp/llama_cl (scratch launcher)"

# =================== [5] per-kernel parity vs the numpy oracle (fail-closed, tol $TOL) ===================
# The Python driver generates random inputs, computes the reference with the ORACLE's pinned ops, and
# compares the GPU output the launcher dumped. Returns 0 PASS / nonzero FAIL; prints a max-abs line.
cat > "$WRK/parity_driver.py" <<PYEOF
import sys, os, numpy as np
sys.path.insert(0, "$ORACLE_DIR")
import llama_ops_numpy_ref as ref   # the INDEPENDENT oracle; pinned ops imported verbatim, NOT modified

op   = sys.argv[1]
stage= sys.argv[2]           # "write" (gen inputs) or "check" (compare GPU out)
wdir = sys.argv[3]
tol  = float(sys.argv[4])
S    = int(os.environ.get("S","7"))
DM   = int(os.environ.get("DM","576"))
NQH  = int(os.environ.get("NQH","9"))
HD   = int(os.environ.get("HD","64"))
DFF  = int(os.environ.get("DFF","1536"))
THETA= float(os.environ.get("THETA","100000.0"))
rng  = np.random.default_rng(20260609)

def wbin(name,a): a.astype(np.float32).ravel().tofile(os.path.join(wdir,name))
def rbin(name,shape):
    a = np.fromfile(os.path.join(wdir,name), dtype=np.float32)
    return a.reshape(shape)

if op == "rmsnorm":
    rows, cols = S, DM
    if stage == "write":
        x = (rng.standard_normal((rows,cols)).astype(np.float32) * 4.0)
        w = (rng.standard_normal(cols).astype(np.float32))
        wbin("x.bin", x); wbin("w.bin", w)
        open(os.path.join(wdir,"dims.txt"),"w").write(f"{rows} {cols}\n")
        np.save(os.path.join(wdir,"_ref.npy"), ref.rmsnorm(x, w, eps=1e-5))
        print(f"  rmsnorm inputs written rows={rows} cols={cols}")
    else:
        r   = np.load(os.path.join(wdir,"_ref.npy"))
        got = rbin("y.bin",(rows,cols))
        err = float(np.max(np.abs(got.astype(np.float64)-r.astype(np.float64))))
        ok  = np.all(np.isfinite(got)) and err <= tol
        print(f"  rmsnorm max-abs err = {err:.3e} (tol {tol:.0e}) -> {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)

elif op == "rope":
    rows, half = S*NQH, HD//2
    if stage == "write":
        q = (rng.standard_normal((rows, HD)).astype(np.float32))
        # positions: each q head is a packed [S, head_dim] block -> positions 0..S-1 repeat per head.
        pos = np.tile(np.arange(S), NQH).astype(np.float32)
        cos_t, sin_t = ref.rope_tables(pos, HD, THETA)        # [rows, half], oracle's pinned HF inv_freq
        wbin("q.bin", q); wbin("cos.bin", cos_t); wbin("sin.bin", sin_t)
        open(os.path.join(wdir,"dims.txt"),"w").write(f"{rows} {half}\n")
        np.save(os.path.join(wdir,"_ref.npy"), ref.rope_rot(q, cos_t, sin_t))
        print(f"  rope inputs written rows={rows} half={half} head_dim={HD}")
    else:
        r   = np.load(os.path.join(wdir,"_ref.npy"))
        got = rbin("q.bin",(rows,HD))
        err = float(np.max(np.abs(got.astype(np.float64)-r.astype(np.float64))))
        ok  = np.all(np.isfinite(got)) and err <= tol
        print(f"  rope max-abs err = {err:.3e} (tol {tol:.0e}) -> {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)

elif op == "silu_mul":
    n = S*DFF
    if stage == "write":
        g = (rng.standard_normal(n).astype(np.float32) * 5.0)   # *5 exercises the overflow-safe sigmoid
        u = (rng.standard_normal(n).astype(np.float32))
        wbin("g.bin", g); wbin("u.bin", u)
        open(os.path.join(wdir,"dims.txt"),"w").write(f"{n}\n")
        np.save(os.path.join(wdir,"_ref.npy"), ref.silu_mul(g, u))
        print(f"  silu_mul inputs written n={n}")
    else:
        r   = np.load(os.path.join(wdir,"_ref.npy"))
        got = rbin("y.bin",(n,))
        err = float(np.max(np.abs(got.astype(np.float64)-r.astype(np.float64))))
        ok  = np.all(np.isfinite(got)) and err <= tol
        print(f"  silu_mul max-abs err = {err:.3e} (tol {tol:.0e}) -> {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)
else:
    print("unknown op"); sys.exit(2)
PYEOF

# run_kernel <op> <launcher-op> <out-bin-name>  : write inputs -> launch -> check parity ; sets <OP>_PARITY
RMSN_PAR=0; ROPE_PAR=0; SILU_PAR=0
RMSN_ERR="n/a"; ROPE_ERR="n/a"; SILU_ERR="n/a"
run_kernel(){
  local op="$1" clop="$2" outbin="$3" var="$4"
  local d="$WRK/$op"; rm -rf "$d"; mkdir -p "$d"
  echo "  -- $op --"
  python3 "$WRK/parity_driver.py" "$op" write "$d" "$TOL" 2>&1 | tee -a "$OUT/llama_parity.log" || { echo "  $op: input-gen FAIL"; RC=1; return; }
  if ! timeout 120 /tmp/llama_cl "$OUT/llama_ops_combined.ptx" "$op" "$d" "$d/$outbin" 2>&1 | tee -a "$OUT/llama_parity.log"; then
    echo "  $op: GPU launch FAIL (see $OUT/llama_parity.log)"; RC=1; return
  fi
  local perr
  perr=$(python3 "$WRK/parity_driver.py" "$op" check "$d" "$TOL" 2>&1); echo "$perr" | tee -a "$OUT/llama_parity.log"
  local pc=${PIPESTATUS[0]}
  # capture the max-abs number for the summary table
  local errnum; errnum=$(echo "$perr" | grep -o 'err = [0-9.eE+-]*' | head -1 | sed 's/err = //')
  eval "${var}_ERR=\"${errnum:-?}\""
  if echo "$perr" | grep -q 'PASS' && [ "$pc" = "0" ]; then eval "${var}_PAR=1"; fi
  # NEG-CONTROL: mutate one cell -> the oracle compare MUST FAIL (comparator teeth)
  rm -f "$d/$outbin"
  timeout 120 /tmp/llama_cl "$OUT/llama_ops_combined.ptx" "$op" "$d" "$d/$outbin" mutate >/dev/null 2>&1 || true
  if python3 "$WRK/parity_driver.py" "$op" check "$d" "$TOL" >/dev/null 2>&1; then
    echo "  NEG-CONTROL($op) FAIL: mutated GPU output still PASSED the oracle compare (no teeth)"; RC=4
  else
    echo "  NEG-CONTROL($op) OK: mutated output correctly FAILED the oracle compare"
  fi
}

if [ "$RC" = "0" ]; then
  echo "=== [5] per-kernel parity vs numpy oracle (fail-closed, tol $TOL) ==="
  run_kernel rmsnorm  gpu_rmsnorm_fwd_eps y.bin RMSN
  run_kernel rope     gpu_rope_rot        q.bin ROPE
  run_kernel silu_mul gpu_silu_mul        y.bin SILU
else
  echo "=== [5] SKIPPED parity (an earlier leg already FAILED rc=$RC) ==="
fi

passmark(){ [ "$1" = "1" ] && echo PASS || echo FAIL; }
echo "=================== HELIX LLAMA G-L0 VERDICT (wall $(( $(date +%s) - T0 ))s) ==================="
printf "  %-22s | %-15s | %-16s | %-18s | %s\n" "kernel" "compiled (kovc)" "ptxas sm_86" "max-abs err vs oracle" "verdict"
printf "  %-22s | %-15s | %-16s | %-18s | %s\n" "gpu_rmsnorm_fwd_eps" "$(passmark $RMSN_OK)" "$(passmark $PTXAS_RMSN)" "$RMSN_ERR" "$(passmark $RMSN_PAR)"
printf "  %-22s | %-15s | %-16s | %-18s | %s\n" "gpu_rope_rot"        "$(passmark $ROPE_OK)" "$(passmark $PTXAS_ROPE)" "$ROPE_ERR" "$(passmark $ROPE_PAR)"
printf "  %-22s | %-15s | %-16s | %-18s | %s\n" "gpu_silu_mul"        "$(passmark $SILU_OK)" "$(passmark $PTXAS_SILU)" "$SILU_ERR" "$(passmark $SILU_PAR)"
echo "  (PTX sha256=$PTX_SHA ; oracle=helix-llm/tools/llama_ops_numpy_ref.py ; ptxas=$PTXAS)"
# PASS iff: all 3 compiled via kovc, all 3 ptxas-accepted sm_86, all 3 parity-passed vs oracle,
# AND no comparator-teeth breach or other failure (RC must be 0).
if [ "$RMSN_OK$ROPE_OK$SILU_OK" = "111" ] && [ "$PTXAS_RMSN$PTXAS_ROPE$PTXAS_SILU" = "111" ] \
   && [ "$RMSN_PAR$ROPE_PAR$SILU_PAR" = "111" ] && [ "$RC" = "0" ]; then
  echo "LLAMA_GL0_PASS"; exit 0
else
  echo "LLAMA_GL0_FAIL (rc=$RC)"; exit 1
fi
