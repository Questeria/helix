/* helixc/runtime/cuda_launch.c -- Helix GPU first-light launcher.
 *
 * DoD criterion #3 (docs/HELIX_V1_DEFINITION_OF_DONE.md) -- "GPU executes", THE GATE.
 *
 * A tiny, generic CUDA Driver-API host launcher. It loads a PTX module as TEXT
 * (exactly the form kovc's emit_ptx_* path produces, or nvcc -ptx for testing),
 * gets a vector_add-class kernel  f(const float* a, const float* b, float* c, int n),
 * runs it on the GPU, copies the result back, and verifies c[i] == a[i] + b[i].
 *
 * WHY C, AND WHY OUTSIDE THE SELF-HOST FIXPOINT (Decision D2, HELIX_FINISH_PLAN.md):
 * libcuda.so is a DYNAMIC shared library; the Helix compiler emits a static,
 * syscall-only, single-PT_LOAD ELF with no dynamic linker, so it cannot link or
 * call libcuda today. This launcher is therefore a trusted-tool boundary, exactly
 * like ptxas or the from-raw-binary build ladder: it sits OUTSIDE the self-host
 * fixpoint. Helix emits the PTX (already real); this C shim is what talks to the
 * driver. In-Helix dynamic linking (path-b) is scheduled to retire this shim
 * before the formal v1.0 freeze, keeping the toolchain pure at the limit.
 *
 * Build (WSL): gcc cuda_launch.c -I/usr/local/cuda/include -L/usr/lib/wsl/lib -lcuda -o cuda_launch
 * Usage:       cuda_launch <module.ptx> <kernel_name> [N]
 * Exit:        0 = launched + numerically verified; 1 = result mismatch; 2 = usage/driver error.
 */

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int check(CUresult r, const char* what) {
    if (r != CUDA_SUCCESS) {
        const char* msg = 0;
        cuGetErrorString(r, &msg);
        fprintf(stderr, "CUDA error in %s: %s (%d)\n", what, msg ? msg : "?", (int)r);
        return 1;
    }
    return 0;
}
#define CK(call, what) do { if (check((call), (what))) return 2; } while (0)

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <module.ptx> <kernel_name> [N] [op:add|mul|sub|reverse]\n", argv[0]);
        return 2;
    }
    const char* ptx_path = argv[1];
    const char* kname    = argv[2];
    int N = (argc > 3) ? atoi(argv[3]) : 256;
    if (N <= 0) N = 256;
    /* op selects the CPU reference the GPU result is checked against, so this one
     * launcher verifies a growing kernel corpus (vector_add/mul/sub) over the same
     * f32 inputs a[i]=i, b[i]=2*i. Default add (back-compat with the first-light run). */
    const char* op = (argc > 4) ? argv[4] : "add";
    int is_exp = (strcmp(op, "exp") == 0);    /* c[i]=e^a[i] via __gpu_exp; small inputs, tol-checked */
    int is_relu = (strcmp(op, "relu") == 0);  /* c[i]=max(a[i],0); negative inputs exercise the float compare */

    /* slurp the PTX text (NUL-terminated; cuModuleLoadData wants a C string) */
    FILE* f = fopen(ptx_path, "rb");
    if (!f) { fprintf(stderr, "cannot open ptx: %s\n", ptx_path); return 2; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return 2; }
    char* ptx = (char*)malloc((size_t)sz + 1);
    if (!ptx) { fclose(f); return 2; }
    if (fread(ptx, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(ptx); return 2; }
    ptx[sz] = 0; fclose(f);

    CK(cuInit(0), "cuInit");
    CUdevice dev; CK(cuDeviceGet(&dev, 0), "cuDeviceGet");
    char gpu[256]; gpu[0] = 0; cuDeviceGetName(gpu, (int)sizeof gpu, dev);
    CUcontext ctx; CK(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate");

    CUmodule mod; CK(cuModuleLoadData(&mod, ptx), "cuModuleLoadData");
    CUfunction fn; CK(cuModuleGetFunction(&fn, mod, kname), "cuModuleGetFunction");

    /* matmul mode: cuda_launch <ptx> <kernel> <Nvec-ignored> matmul <M> <K> <N>.
     * One thread per output cell -- gridDim.x=M (row=block_idx), blockDim.x=N
     * (col=thread_idx). Verifies EVERY M*N cell of C against a CPU reference in the
     * SAME accumulation order the kernel uses (acc=a[row*K]*b[col], then += k=1..K-1),
     * with integer-valued inputs (a[i]=i%7, b[i]=i%5) so all sums are exact in f32. */
    if (strcmp(op, "matmul") == 0) {
        int Md = (argc > 5) ? atoi(argv[5]) : 16;
        int Kd = (argc > 6) ? atoi(argv[6]) : 16;
        int Nd = (argc > 7) ? atoi(argv[7]) : 16;
        size_t aN = (size_t)Md * Kd, bN = (size_t)Kd * Nd, cN = (size_t)Md * Nd;
        float* hA = (float*)malloc(aN * sizeof(float));
        float* hB = (float*)malloc(bN * sizeof(float));
        float* hC = (float*)malloc(cN * sizeof(float));
        if (!hA || !hB || !hC) return 2;
        for (size_t i = 0; i < aN; i++) hA[i] = (float)(i % 7);
        for (size_t i = 0; i < bN; i++) hB[i] = (float)(i % 5);
        CUdeviceptr dA, dB, dC;
        CK(cuMemAlloc(&dA, aN * sizeof(float)), "cuMemAlloc A");
        CK(cuMemAlloc(&dB, bN * sizeof(float)), "cuMemAlloc B");
        CK(cuMemAlloc(&dC, cN * sizeof(float)), "cuMemAlloc C");
        CK(cuMemcpyHtoD(dA, hA, aN * sizeof(float)), "cuMemcpyHtoD A");
        CK(cuMemcpyHtoD(dB, hB, bN * sizeof(float)), "cuMemcpyHtoD B");
        void* margs[] = { &dA, &dB, &dC, &Md, &Kd, &Nd };
        CK(cuLaunchKernel(fn, Md, 1, 1, Nd, 1, 1, 0, 0, margs, 0), "cuLaunchKernel matmul");
        CK(cuCtxSynchronize(), "cuCtxSynchronize");
        CK(cuMemcpyDtoH(hC, dC, cN * sizeof(float)), "cuMemcpyDtoH C");
        int mbad = 0;
        for (int r = 0; r < Md; r++) {
            for (int cc = 0; cc < Nd; cc++) {
                float ref = hA[r * Kd] * hB[cc];
                for (int t = 1; t < Kd; t++) ref += hA[r * Kd + t] * hB[t * Nd + cc];
                float got = hC[r * Nd + cc];
                if (got != ref) { if (mbad < 4) fprintf(stderr, "matmul mismatch C[%d,%d]=%g ref %g\n", r, cc, got, ref); mbad++; }
            }
        }
        printf("GPU [%s] naive_matmul %dx%dx%d over %d cells: C[1,1]=%g, %d bad -> %s\n",
               gpu, Md, Kd, Nd, Md * Nd, hC[1 * Nd + 1], mbad, mbad ? "FAIL" : "PASS");
        cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hA); free(hB); free(hC); free(ptx);
        return mbad ? 1 : 0;
    }

    size_t bytes = (size_t)N * sizeof(float);
    float* ha = (float*)malloc(bytes);
    float* hb = (float*)malloc(bytes);
    float* hc = (float*)malloc(bytes);
    if (!ha || !hb || !hc) { return 2; }
    for (int i = 0; i < N; i++) {
        ha[i] = is_exp ? (float)((i % 8) - 4) : is_relu ? (float)(i - 128) : (float)i;
        hb[i] = (float)(2 * i); hc[i] = -1.0f;
    }

    CUdeviceptr da, db, dc;
    CK(cuMemAlloc(&da, bytes), "cuMemAlloc a");
    CK(cuMemAlloc(&db, bytes), "cuMemAlloc b");
    CK(cuMemAlloc(&dc, bytes), "cuMemAlloc c");
    CK(cuMemcpyHtoD(da, ha, bytes), "cuMemcpyHtoD a");
    CK(cuMemcpyHtoD(db, hb, bytes), "cuMemcpyHtoD b");

    void* args[] = { &da, &db, &dc, &N };
    int tpb = 256;
    int bpg = (N + tpb - 1) / tpb;
    CK(cuLaunchKernel(fn, bpg, 1, 1, tpb, 1, 1, 0, 0, args, 0), "cuLaunchKernel");
    CK(cuCtxSynchronize(), "cuCtxSynchronize");
    CK(cuMemcpyDtoH(hc, dc, bytes), "cuMemcpyDtoH c");

    int is_mul = (strcmp(op, "mul") == 0);
    int is_sub = (strcmp(op, "sub") == 0);
    int is_rev = (strcmp(op, "reverse") == 0);  /* c[i]=a[N-1-i]: exercises an i32 scalar param read in the index */
    int bad = 0;
    for (int i = 0; i < N; i++) {
        float want = is_exp ? expf(ha[i]) : is_relu ? (ha[i] > 0.0f ? ha[i] : 0.0f)
                   : is_rev ? ha[N - 1 - i] : is_mul ? ha[i] * hb[i] : is_sub ? ha[i] - hb[i] : ha[i] + hb[i];
        int bad_i;
        if (is_exp) { float d = hc[i] - want; if (d < 0) d = -d; float aw = want < 0 ? -want : want; bad_i = (d > 1.0e-3f * (aw + 1.0e-6f)); }
        else bad_i = (hc[i] != want);
        if (bad_i) { if (bad < 4) fprintf(stderr, "mismatch c[%d]=%g want %g\n", i, hc[i], want); bad++; }
    }
    float want7 = is_exp ? expf(ha[7]) : is_relu ? (ha[7] > 0.0f ? ha[7] : 0.0f)
                : is_rev ? ha[N - 1 - 7] : is_mul ? ha[7] * hb[7] : is_sub ? ha[7] - hb[7] : ha[7] + hb[7];
    printf("GPU [%s] kernel '%s' op=%s over %d elems: c[7]=%g (want %g) -> %s\n",
           gpu, kname, op, N, hc[7], want7, bad ? "FAIL" : "PASS");

    cuMemFree(da); cuMemFree(db); cuMemFree(dc);
    cuModuleUnload(mod); cuCtxDestroy(ctx);
    free(ha); free(hb); free(hc); free(ptx);
    return bad ? 1 : 0;
}
