#!/usr/bin/env bash
# gemv_ops_parity.sh -- G-KV0 kernel parity gate for the 3 KV-CACHE DECODE kernels
# (gpu_gemv_abt, gpu_gemv_ab, gpu_softmax_row). Same fail-closed pattern as
# scripts/llama_ops_parity.sh: from-raw kovc -> PTX -> ptxas sm_86 -> data-driven
# scratch launcher -> parity vs the INDEPENDENT numpy oracle's pinned ops + a mutate
# negative control per kernel. Run as a FILE under WSL:
#   wsl.exe bash -lc "tr -d '\r' < /mnt/c/Projects/Kovostov-Native/scripts/gemv_ops_parity.sh > /tmp/gop.sh && bash /tmp/gop.sh"
set -u
set -o pipefail
T0=$(date +%s)
ROOT="${HELIX_SRC:-}"; if [ -z "$ROOT" ]; then ROOT="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"; fi
[ -d "$ROOT/helixc/examples" ] || ROOT="/mnt/c/Projects/Kovostov-Native"
EX="$ROOT/helixc/examples"
ORACLE_DIR="$ROOT/helix-llm/tools"
OUT="$ROOT/.m1probe"; mkdir -p "$OUT"
DRV="${LLAMA_DRV:-$HOME/gpt2_ext4/llama_kovc_drv.bin}"
PTXAS="${PTXAS:-/usr/local/cuda-12.8/bin/ptxas}"; [ -x "$PTXAS" ] || PTXAS="/usr/local/cuda/bin/ptxas"
TOL="${TOL:-1e-4}"
RC=0
WRK=/tmp/gemv_gl0; mkdir -p "$WRK"

echo "=================== HELIX GEMV/DECODE KERNEL PARITY (G-KV0)  $(date -u +%H:%M:%S) ==================="
echo "  root=$ROOT  ptxas=$PTXAS  tol=$TOL"

echo "=== [0] oracle self-test (the 3 decode ops are pinned there) ==="
if python3 "$ORACLE_DIR/llama_ops_numpy_ref.py" 2>&1 | grep -q '^LLAMA_OPS_REF_SELFTEST: PASS'; then
  echo "  oracle selftest PASS"
else
  echo "  FATAL: oracle selftest failed"; echo "GEMV_GL0_FAIL"; exit 9
fi

echo "=== [1] from-raw kovc -> 3-kernel PTX -> ptxas sm_86 ==="
[ -x "$DRV" ] || { echo "FATAL: no cached from-raw driver $DRV (run llama_ops_parity.sh REMINT=1 first)"; echo "GEMV_GL0_FAIL"; exit 7; }
: > /tmp/kernel_in.hx
for k in gpu_gemv_abt gpu_gemv_ab gpu_softmax_row; do
  tr -d '\r' < "$EX/${k}_kernel.hx" >> /tmp/kernel_in.hx; echo "" >> /tmp/kernel_in.hx
done
rm -f /tmp/out.ptx
"$DRV" >/dev/null 2>&1 || true
[ -s /tmp/out.ptx ] || { echo "FATAL: kovc emitted no PTX"; echo "GEMV_GL0_FAIL"; exit 6; }
cp /tmp/out.ptx "$WRK/gemv.ptx"
NENT=$(grep -c '\.entry' "$WRK/gemv.ptx")
echo "  PTX $(stat -c%s "$WRK/gemv.ptx") B, $NENT .entry (want 3)"
[ "$NENT" = "3" ] || { echo "FATAL kernel count"; echo "GEMV_GL0_FAIL"; exit 6; }
"$PTXAS" -arch=sm_86 "$WRK/gemv.ptx" -o "$WRK/gemv.cubin" 2>"$OUT/gemv_ptxas.log" \
  && echo "  PTXAS_ACCEPT (sm_86)" || { echo "  PTXAS_REJECT"; cat "$OUT/gemv_ptxas.log"; echo "GEMV_GL0_FAIL"; exit 5; }

echo "=== [2] scratch data-driven launcher (uncommitted /tmp shim; cuda_launch.c idiom) ==="
cat > "$WRK/gemv_cl.c" <<'CEOF'
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int check(CUresult r,const char* w){ if(r!=CUDA_SUCCESS){ const char* m=0; cuGetErrorString(r,&m); fprintf(stderr,"CUDA %s: %s\n",w,m?m:"?"); return 1;} return 0; }
#define CK(c,w) do{ if(check((c),(w))) return 2; }while(0)
static char* slurp(const char* p,size_t* n){ FILE* f=fopen(p,"rb"); if(!f) return 0; fseek(f,0,SEEK_END); long s=ftell(f); fseek(f,0,SEEK_SET); char* b=malloc(s+1); if(fread(b,1,s,f)!=(size_t)s){fclose(f);free(b);return 0;} b[s]=0; fclose(f); if(n)*n=s; return b; }
static float* rd(const char* d,const char* n,size_t nf){ char p[1024]; snprintf(p,sizeof p,"%s/%s",d,n); size_t bs=0; float* b=(float*)slurp(p,&bs); if(!b||bs<nf*4){ fprintf(stderr,"bad %s\n",p); return 0;} return b; }
static int wr(const char* p,const float* x,size_t nf){ FILE* f=fopen(p,"wb"); if(!f) return 1; size_t k=fwrite(x,4,nf,f); fclose(f); return k!=nf; }
int main(int argc,char** argv){
  if(argc<5){ fprintf(stderr,"usage: %s <ptx> <op> <indir> <out> [mutate]\n",argv[0]); return 2; }
  const char* op=argv[2]; const char* ind=argv[3]; const char* outp=argv[4];
  int mutate=(argc>5 && !strcmp(argv[5],"mutate"));
  char dp[1024]; snprintf(dp,sizeof dp,"%s/dims.txt",ind);
  char* dt=slurp(dp,0); if(!dt) return 2;
  char* ptx=slurp(argv[1],0); if(!ptx) return 2;
  CK(cuInit(0),"init"); CUdevice dev; CK(cuDeviceGet(&dev,0),"dev"); CUcontext ctx; CK(cuCtxCreate(&ctx,0,dev),"ctx");
  CUmodule mod; CK(cuModuleLoadData(&mod,ptx),"mod");
  if(!strcmp(op,"gemv_abt")){
    int N=0,K=0; if(sscanf(dt,"%d %d",&N,&K)!=2) return 2;
    float* hx=rd(ind,"x.bin",(size_t)K); float* hw=rd(ind,"w.bin",(size_t)N*K); float* hy=malloc((size_t)N*4);
    if(!hx||!hw||!hy) return 2;
    CUfunction fn; CK(cuModuleGetFunction(&fn,mod,"gpu_gemv_abt"),"get");
    CUdeviceptr dx,dw,dy; CK(cuMemAlloc(&dx,(size_t)K*4),"ax"); CK(cuMemAlloc(&dw,(size_t)N*K*4),"aw"); CK(cuMemAlloc(&dy,(size_t)N*4),"ay");
    CK(cuMemcpyHtoD(dx,hx,(size_t)K*4),"hx"); CK(cuMemcpyHtoD(dw,hw,(size_t)N*K*4),"hw");
    void* a[]={ &dx,&dw,&dy,&K };
    CK(cuLaunchKernel(fn,(unsigned)N,1,1, 1,1,1, 0,0, a,0),"launch"); CK(cuCtxSynchronize(),"sync");
    CK(cuMemcpyDtoH(hy,dy,(size_t)N*4),"d2h");
    if(mutate) hy[0]+=1.0e3f;
    if(wr(outp,hy,(size_t)N)) return 2;
    printf("gemv_cl gemv_abt N=%d K=%d -> %s%s\n",N,K,outp,mutate?" [MUTATED]":"");
  } else if(!strcmp(op,"gemv_ab")){
    int T=0,N=0; if(sscanf(dt,"%d %d",&T,&N)!=2) return 2;
    float* hp=rd(ind,"p.bin",(size_t)T); float* hm=rd(ind,"m.bin",(size_t)T*N); float* hy=malloc((size_t)N*4);
    if(!hp||!hm||!hy) return 2;
    CUfunction fn; CK(cuModuleGetFunction(&fn,mod,"gpu_gemv_ab"),"get");
    CUdeviceptr dp_,dm,dy; CK(cuMemAlloc(&dp_,(size_t)T*4),"ap"); CK(cuMemAlloc(&dm,(size_t)T*N*4),"am"); CK(cuMemAlloc(&dy,(size_t)N*4),"ay");
    CK(cuMemcpyHtoD(dp_,hp,(size_t)T*4),"hp"); CK(cuMemcpyHtoD(dm,hm,(size_t)T*N*4),"hm");
    void* a[]={ &dp_,&dm,&dy,&T,&N };
    CK(cuLaunchKernel(fn,(unsigned)N,1,1, 1,1,1, 0,0, a,0),"launch"); CK(cuCtxSynchronize(),"sync");
    CK(cuMemcpyDtoH(hy,dy,(size_t)N*4),"d2h");
    if(mutate) hy[0]+=1.0e3f;
    if(wr(outp,hy,(size_t)N)) return 2;
    printf("gemv_cl gemv_ab T=%d N=%d -> %s%s\n",T,N,outp,mutate?" [MUTATED]":"");
  } else if(!strcmp(op,"softmax_row")){
    int R=0,C=0; if(sscanf(dt,"%d %d",&R,&C)!=2) return 2;
    size_t ne=(size_t)R*C;
    float* hx=rd(ind,"x.bin",ne); float* hy=malloc(ne*4);
    if(!hx||!hy) return 2;
    CUfunction fn; CK(cuModuleGetFunction(&fn,mod,"gpu_softmax_row"),"get");
    CUdeviceptr dx,dy; CK(cuMemAlloc(&dx,ne*4),"ax"); CK(cuMemAlloc(&dy,ne*4),"ay");
    CK(cuMemcpyHtoD(dx,hx,ne*4),"hx");
    void* a[]={ &dx,&dy,&C };
    CK(cuLaunchKernel(fn,(unsigned)R,1,1, 1,1,1, 0,0, a,0),"launch"); CK(cuCtxSynchronize(),"sync");
    CK(cuMemcpyDtoH(hy,dy,ne*4),"d2h");
    if(mutate) hy[0]+=1.0e3f;
    if(wr(outp,hy,ne)) return 2;
    printf("gemv_cl softmax_row R=%d C=%d -> %s%s\n",R,C,outp,mutate?" [MUTATED]":"");
  } else return 2;
  cuModuleUnload(mod); cuCtxDestroy(ctx); return 0;
}
CEOF
gcc "$WRK/gemv_cl.c" -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcuda -o /tmp/gemv_cl 2>"$OUT/gemv_cl_gcc.log" \
  || { echo "FATAL gcc gemv_cl"; cat "$OUT/gemv_cl_gcc.log"; echo "GEMV_GL0_FAIL"; exit 4; }
echo "  built /tmp/gemv_cl"

echo "=== [3] parity driver (oracle ops imported verbatim; write inputs / check outputs) ==="
cat > "$WRK/gemv_driver.py" <<PYEOF
import sys, os, numpy as np
sys.path.insert(0, "$ORACLE_DIR")
import llama_ops_numpy_ref as ref
op, stage, wdir = sys.argv[1], sys.argv[2], sys.argv[3]
tol = float(sys.argv[4])
rng = np.random.default_rng(20260610)
def w(name, arr): arr.astype("<f4").tofile(os.path.join(wdir, name))
def r(name): return np.fromfile(os.path.join(wdir, name), dtype="<f4")
def dims(s): open(os.path.join(wdir, "dims.txt"), "w").write(s)
if op == "gemv_abt":
    N, K = 113, 960          # deliberately NOT %64 -- the whole point of the GEMV form
    if stage == "write":
        x = rng.standard_normal((1, K), dtype=np.float32); W = rng.standard_normal((N, K), dtype=np.float32)
        w("x.bin", x); w("w.bin", W); dims("%d %d" % (N, K)); np.save(os.path.join(wdir, "ref.npy"), ref.gemv_abt(x, W))
    else:
        got = r("out.bin")[:N]; want = np.load(os.path.join(wdir, "ref.npy")).ravel()
        err = float(np.max(np.abs(got - want)))
        ok = np.all(np.isfinite(got)) and err <= tol
        print("gemv_abt max-abs err = %.3e (tol %g) -> %s" % (err, tol, "PASS" if ok else "FAIL")); sys.exit(0 if ok else 1)
elif op == "gemv_ab":
    T, N = 37, 64
    if stage == "write":
        p = rng.standard_normal((1, T), dtype=np.float32); M = rng.standard_normal((T, N), dtype=np.float32)
        w("p.bin", p); w("m.bin", M); dims("%d %d" % (T, N)); np.save(os.path.join(wdir, "ref.npy"), ref.gemv_ab(p, M))
    else:
        got = r("out.bin")[:N]; want = np.load(os.path.join(wdir, "ref.npy")).ravel()
        err = float(np.max(np.abs(got - want)))
        ok = np.all(np.isfinite(got)) and err <= tol
        print("gemv_ab max-abs err = %.3e (tol %g) -> %s" % (err, tol, "PASS" if ok else "FAIL")); sys.exit(0 if ok else 1)
elif op == "softmax_row":
    R, C = 3, 91             # multi-row + odd cols
    if stage == "write":
        x = (rng.standard_normal((R, C)) * 4).astype(np.float32)
        w("x.bin", x); dims("%d %d" % (R, C)); np.save(os.path.join(wdir, "ref.npy"), ref.softmax_row(x))
    else:
        got = r("out.bin")[:R*C].reshape(R, C); want = np.load(os.path.join(wdir, "ref.npy"))
        err = float(np.max(np.abs(got - want)))
        ok = np.all(np.isfinite(got)) and err <= tol
        print("softmax_row max-abs err = %.3e (tol %g) -> %s" % (err, tol, "PASS" if ok else "FAIL")); sys.exit(0 if ok else 1)
PYEOF

run_op () {  # $1 = op
  local op="$1" d="$WRK/$1"; mkdir -p "$d"
  python3 "$WRK/gemv_driver.py" "$op" write "$d" "$TOL" || { echo "  $op: input gen FAIL"; return 1; }
  /tmp/gemv_cl "$WRK/gemv.ptx" "$op" "$d" "$d/out.bin" || { echo "  $op: launch FAIL"; return 1; }
  python3 "$WRK/gemv_driver.py" "$op" check "$d" "$TOL" || return 1
  # negative control: mutated output MUST fail
  /tmp/gemv_cl "$WRK/gemv.ptx" "$op" "$d" "$d/out.bin" mutate || { echo "  $op: mutate launch FAIL"; return 1; }
  if python3 "$WRK/gemv_driver.py" "$op" check "$d" "$TOL" >/dev/null 2>&1; then
    echo "  NEG-CONTROL($op) FAIL: mutated output PASSED (no teeth)"; return 1
  fi
  echo "  NEG-CONTROL($op) OK: mutated output correctly FAILED"
  # restore the clean output for the record
  /tmp/gemv_cl "$WRK/gemv.ptx" "$op" "$d" "$d/out.bin" >/dev/null 2>&1
  return 0
}

G1=FAIL; G2=FAIL; G3=FAIL
run_op gemv_abt    && G1=PASS || RC=1
run_op gemv_ab     && G2=PASS || RC=1
run_op softmax_row && G3=PASS || RC=1

echo "=================== G-KV0 VERDICT (wall $(( $(date +%s) - T0 ))s) ==================="
printf "  %-18s %s\n" "gpu_gemv_abt"    "$G1"
printf "  %-18s %s\n" "gpu_gemv_ab"     "$G2"
printf "  %-18s %s\n" "gpu_softmax_row" "$G3"
if [ "$G1$G2$G3" = "PASSPASSPASS" ] && [ "$RC" = "0" ]; then
  echo "GEMV_GL0_PASS"; exit 0
else
  echo "GEMV_GL0_FAIL (rc=$RC)"; exit 1
fi
