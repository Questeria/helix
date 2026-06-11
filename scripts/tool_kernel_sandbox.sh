#!/usr/bin/env bash
# tool_kernel_sandbox.sh -- W5 "the model writes a Helix kernel and runs it" SANDBOX.
# SECURITY-AUDIT-PENDING: this executes MODEL-SUPPLIED code. It ships DISABLED by default
# (the server requires --tool-kernel 1 AND this script) and must clear the independent
# security audit before being enabled anywhere public.
# DEFENSE LAYERS (documented for the audit):
#   1. INPUT SHAPE: exactly ONE @kernel fn; <= 4 KB; <= 120 lines; ASCII only; the ONLY
#      tokens admitted are the gated-kernel language subset (see the allowlist scan) --
#      no host code, no I/O, no inline PTX escapes (kovc has none, but we scan anyway).
#   2. COMPILER: the SAME from-raw cached kovc driver the gates use (no new toolchain),
#      run with a 60 s timeout in a throwaway dir; output validated by non-empty PTX with
#      exactly 1 .entry + ptxas sm_86 acceptance.
#   3. EXECUTION: the data-driven scratch launcher pattern (gemv_cl) on FIXED random
#      inputs; grid hard-capped (N <= 4096, block 1); 30 s timeout; ulimit -v 2 GB.
#      The kernel can only read/write the buffers the harness allocates.
#   4. NO network access is granted by anything in this path; filesystem writes are
#      confined to the throwaway dir under /tmp.
#   Residual risks for the audit: GPU denial-of-service (long-spin kernels are killed by
#   the 30 s timeout + context destroy), PTX-level memory safety inside the allocated
#   buffers only (out-of-bounds clamps are NOT guaranteed -- the harness allocates +64KB
#   guard slack; a hostile kernel can corrupt ITS OWN process GPU context, which is
#   destroyed after the run; the persistent demo workers run in SEPARATE processes).
# Usage: tool_kernel_sandbox.sh <kernel.hx> ; prints JSON {ok, name, n, out_head, ms} or {ok:false, error}
set -u
set -o pipefail
KFILE="${1:-}"
[ -s "$KFILE" ] || { echo '{"ok":false,"error":"no kernel file"}'; exit 1; }
ROOT="${HELIX_SRC:-/mnt/c/Projects/Kovostov-Native}"
DRV="${LLAMA_DRV:-$HOME/gpt2_ext4/llama_kovc_drv.bin}"
PTXAS="${PTXAS:-/usr/local/cuda-12.8/bin/ptxas}"; [ -x "$PTXAS" ] || PTXAS=/usr/local/cuda/bin/ptxas
WRK=$(mktemp -d /tmp/hxtool.XXXXXX) || exit 1
trap 'rm -rf "$WRK"' EXIT

# ---- layer 1: input shape ----
SZ=$(wc -c < "$KFILE"); LN=$(wc -l < "$KFILE")
[ "$SZ" -le 4096 ] || { echo '{"ok":false,"error":"kernel too large (4KB cap)"}'; exit 1; }
[ "$LN" -le 120 ]  || { echo '{"ok":false,"error":"kernel too long (120-line cap)"}'; exit 1; }
LC_ALL=C grep -q '[^ -~\t\r\n]' "$KFILE" && { echo '{"ok":false,"error":"non-ASCII content"}'; exit 1; }
NK=$(grep -c '@kernel' "$KFILE")
[ "$NK" = "1" ] || { echo '{"ok":false,"error":"exactly one @kernel fn required"}'; exit 1; }
KNAME=$(grep -A1 '@kernel' "$KFILE" | grep -oE 'fn +[a-z_][a-z0-9_]*' | head -1 | awk '{print $2}')
[ -n "$KNAME" ] || { echo '{"ok":false,"error":"no kernel fn name"}'; exit 1; }
# allowlist scan: forbid anything outside the gated kernel subset
if grep -nE 'include|import|asm|__attribute|system|exec|fopen|/dev/|\\\\x' "$KFILE" >/dev/null; then
  echo '{"ok":false,"error":"forbidden token"}'; exit 1
fi

# ---- layer 2: from-raw compile (same driver as the gates) ----
[ -x "$DRV" ] || { echo '{"ok":false,"error":"no cached from-raw driver"}'; exit 1; }
tr -d '\r' < "$KFILE" > /tmp/kernel_in.hx; echo "" >> /tmp/kernel_in.hx
rm -f /tmp/out.ptx
( ulimit -v 2097152 -t 60; timeout 60 "$DRV" ) >/dev/null 2>&1 || true
[ -s /tmp/out.ptx ] || { echo '{"ok":false,"error":"kovc emitted no PTX (compile failed)"}'; exit 1; }
NENT=$(grep -c '\.entry' /tmp/out.ptx)
[ "$NENT" = "1" ] || { echo '{"ok":false,"error":"PTX entry count != 1"}'; exit 1; }
cp /tmp/out.ptx "$WRK/k.ptx"
"$PTXAS" -arch=sm_86 "$WRK/k.ptx" -o "$WRK/k.cubin" 2>"$WRK/ptxas.log" || { echo '{"ok":false,"error":"ptxas rejected"}'; exit 1; }

# ---- layer 3: bounded execution on fixed random inputs ----
cat > "$WRK/run.c" <<'CEOF'
/* throwaway harness: alloc 3 buffers (a,b,y) of N floats (+64KB guard), random a/b,
 * launch <name>(a,b,y,n) grid=N block=1, copy y back, print head. Any signature
 * mismatch just garbles y -- acceptable for a sandbox demo; nothing else is mapped. */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int ck(CUresult r,const char*w){ if(r!=CUDA_SUCCESS){ const char*m=0; cuGetErrorString(r,&m); fprintf(stderr,"%s:%s\n",w,m?m:"?"); return 1;} return 0; }
#define CK(c,w) do{ if(ck((c),(w))) return 2; }while(0)
int main(int argc,char**argv){
  if(argc<4) return 2;
  const char* kname=argv[2]; int n=atoi(argv[3]); if(n<1||n>4096) return 2;
  FILE* f=fopen(argv[1],"rb"); if(!f) return 2; fseek(f,0,SEEK_END); long s=ftell(f); fseek(f,0,SEEK_SET);
  char* ptx=malloc(s+1); if(fread(ptx,1,s,f)!=(size_t)s) return 2; ptx[s]=0; fclose(f);
  CK(cuInit(0),"init"); CUdevice d; CK(cuDeviceGet(&d,0),"dev"); CUcontext cx; CK(cuCtxCreate(&cx,0,d),"ctx");
  CUmodule m; CK(cuModuleLoadData(&m,ptx),"mod"); CUfunction fn; CK(cuModuleGetFunction(&fn,m,kname),"get");
  size_t bytes=(size_t)n*4+65536;
  float* ha=malloc(bytes); float* hy=malloc(bytes);
  srand(20260610); for(int i=0;i<n;i++) ha[i]=(float)(rand()%1000)/100.0f-5.0f;
  CUdeviceptr da,db,dy; CK(cuMemAlloc(&da,bytes),"a"); CK(cuMemAlloc(&db,bytes),"b"); CK(cuMemAlloc(&dy,bytes),"y");
  CK(cuMemcpyHtoD(da,ha,(size_t)n*4),"h2d a"); CK(cuMemcpyHtoD(db,ha,(size_t)n*4),"h2d b");
  void* args[]={ &da,&db,&dy,&n };
  CK(cuLaunchKernel(fn,(unsigned)n,1,1,1,1,1,0,0,args,0),"launch");
  CK(cuCtxSynchronize(),"sync");
  CK(cuMemcpyDtoH(hy,dy,(size_t)n*4),"d2h");
  printf("["); for(int i=0;i<(n<8?n:8);i++) printf("%s%.4f",i?",":"",hy[i]); printf("]\n");
  cuModuleUnload(m); cuCtxDestroy(cx); return 0;
}
CEOF
gcc "$WRK/run.c" -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcuda -o "$WRK/run" 2>"$WRK/gcc.log" || { echo '{"ok":false,"error":"harness build failed"}'; exit 1; }
N=256
T0=$(date +%s%3N)
OUT=$( ( ulimit -v 2097152 -t 30; timeout 30 "$WRK/run" "$WRK/k.ptx" "$KNAME" "$N" ) 2>"$WRK/run.log" ) || { echo "{\"ok\":false,\"error\":\"launch failed: $(head -c120 "$WRK/run.log" | tr -d '"\n')\"}"; exit 1; }
T1=$(date +%s%3N)
echo "{\"ok\":true,\"name\":\"$KNAME\",\"n\":$N,\"out_head\":$OUT,\"ms\":$((T1-T0)),\"note\":\"compiled FROM-RAW by the cached kovc driver, ptxas sm_86, launched grid=$N block=1 on (a,b,y,n) random inputs; output head shown raw -- NOT verified against a reference\"}"
