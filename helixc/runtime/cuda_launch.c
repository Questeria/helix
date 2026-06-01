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

    /* attention mode: cuda_launch <combined.ptx> gpu_qkt <Nignored> attention <S> <d>.
     * combined.ptx carries 3 entries (gpu_qkt, gpu_softmax, naive_matmul -- concat the
     * 3 kernel .hx files, emit once). Single-head scaled dot-product attention:
     *   scores = (1/sqrt(d)) * Q @ K^T   (gpu_qkt bakes 0.25 = 1/sqrt(16), so d MUST be 16)
     *   attn   = softmax(scores) row-wise
     *   out    = attn @ V                (naive_matmul, A=attn[S,S] B=V[S,d])
     * mod is already loaded from argv[1]; fetch all 3 entries from it (argv[2]=gpu_qkt
     * satisfies the generic get above). Verify out vs a 3-stage CPU reference AND
     * (independent) that each GPU attn row sums to ~1. Integer-valued inputs so QK^T is
     * exact; 0.25*int is exact; the only error is softmax ex2.approx (tol 1e-3). */
    if (strcmp(op, "attention") == 0) {
        int S = (argc > 5) ? atoi(argv[5]) : 4;
        int d = (argc > 6) ? atoi(argv[6]) : 16;
        CUfunction f_qkt, f_sm, f_mm;
        CK(cuModuleGetFunction(&f_qkt, mod, "gpu_qkt"), "get gpu_qkt");
        CK(cuModuleGetFunction(&f_sm, mod, "gpu_softmax"), "get gpu_softmax");
        CK(cuModuleGetFunction(&f_mm, mod, "naive_matmul"), "get naive_matmul");
        size_t sd = (size_t)S * d, ssz = (size_t)S * S;
        float* hQ = (float*)malloc(sd * sizeof(float));
        float* hK = (float*)malloc(sd * sizeof(float));
        float* hV = (float*)malloc(sd * sizeof(float));
        float* hO = (float*)malloc(sd * sizeof(float));
        float* hA = (float*)malloc(ssz * sizeof(float));
        float* sc = (float*)malloc(ssz * sizeof(float));
        float* at = (float*)malloc(ssz * sizeof(float));
        if (!hQ || !hK || !hV || !hO || !hA || !sc || !at) return 2;
        for (size_t i = 0; i < sd; i++) {
            hQ[i] = (float)((int)(i % 7) - 3);
            hK[i] = (float)((int)(i % 5) - 2);
            hV[i] = (float)((int)(i % 9) - 4);
        }
        CUdeviceptr dQ, dK, dV, dS, dA, dO;
        CK(cuMemAlloc(&dQ, sd * sizeof(float)), "alloc Q");
        CK(cuMemAlloc(&dK, sd * sizeof(float)), "alloc K");
        CK(cuMemAlloc(&dV, sd * sizeof(float)), "alloc V");
        CK(cuMemAlloc(&dS, ssz * sizeof(float)), "alloc scores");
        CK(cuMemAlloc(&dA, ssz * sizeof(float)), "alloc attn");
        CK(cuMemAlloc(&dO, sd * sizeof(float)), "alloc out");
        CK(cuMemcpyHtoD(dQ, hQ, sd * sizeof(float)), "H2D Q");
        CK(cuMemcpyHtoD(dK, hK, sd * sizeof(float)), "H2D K");
        CK(cuMemcpyHtoD(dV, hV, sd * sizeof(float)), "H2D V");
        /* stage 1: scores = 0.25 * Q@K^T. gridDim=S, blockDim=S. */
        void* a1[] = { &dQ, &dK, &dS, &S, &d };
        CK(cuLaunchKernel(f_qkt, S, 1, 1, S, 1, 1, 0, 0, a1, 0), "launch qkt");
        CK(cuCtxSynchronize(), "sync qkt");
        /* stage 2: attn = softmax(scores) row-wise. gridDim=S, blockDim=1, cols=S. */
        void* a2[] = { &dS, &dA, &S, &S };
        CK(cuLaunchKernel(f_sm, S, 1, 1, 1, 1, 1, 0, 0, a2, 0), "launch softmax");
        CK(cuCtxSynchronize(), "sync softmax");
        /* stage 3: out = attn @ V. naive_matmul(a,b,c,M,K,N): M=S,K=S,N=d. gridDim=S, blockDim=d. */
        void* a3[] = { &dA, &dV, &dO, &S, &S, &d };
        CK(cuLaunchKernel(f_mm, S, 1, 1, d, 1, 1, 0, 0, a3, 0), "launch matmul");
        CK(cuCtxSynchronize(), "sync matmul");
        CK(cuMemcpyDtoH(hO, dO, sd * sizeof(float)), "D2H out");
        CK(cuMemcpyDtoH(hA, dA, ssz * sizeof(float)), "D2H attn");
        /* CPU reference: scores -> row softmax -> out. */
        float scale = 1.0f / sqrtf((float)d);
        for (int i = 0; i < S; i++) {
            for (int j = 0; j < S; j++) {
                float dot = 0.0f;
                for (int t = 0; t < d; t++) dot += hQ[i * d + t] * hK[j * d + t];
                sc[i * S + j] = scale * dot;
            }
            float mx = sc[i * S]; for (int j = 1; j < S; j++) if (sc[i * S + j] > mx) mx = sc[i * S + j];
            float sm = 0.0f; for (int j = 0; j < S; j++) sm += expf(sc[i * S + j] - mx);
            for (int j = 0; j < S; j++) at[i * S + j] = expf(sc[i * S + j] - mx) / sm;
        }
        int abad = 0; float ref0 = 0.0f; float maxrs = 0.0f;
        for (int i = 0; i < S; i++) {
            for (int kk = 0; kk < d; kk++) {
                float ref = 0.0f; for (int j = 0; j < S; j++) ref += at[i * S + j] * hV[j * d + kk];
                if (i == 0 && kk == 0) ref0 = ref;
                float got = hO[i * d + kk];
                float e = got - ref; if (e < 0) e = -e;
                if (isnan(got) || e > 1.0e-3f) { if (abad < 4) fprintf(stderr, "attention mismatch out[%d,%d]=%g ref %g\n", i, kk, got, ref); abad++; }
            }
        }
        /* independent cross-check: each GPU attn row sums to ~1. */
        for (int i = 0; i < S; i++) {
            float rs = 0.0f; for (int j = 0; j < S; j++) rs += hA[i * S + j];
            float e = rs - 1.0f; if (e < 0) e = -e; if (e > maxrs) maxrs = e;
            if (isnan(rs) || e > 1.0e-3f) { if (abad < 4) fprintf(stderr, "attention attn row %d sum %g (want 1)\n", i, rs); abad++; }
        }
        printf("GPU [%s] attention S=%d d=%d: out[0,0]=%g ref %g, max|attn_rowsum-1|=%g, %d bad -> %s\n",
               gpu, S, d, hO[0], ref0, maxrs, abad, abad ? "FAIL" : "PASS");
        cuMemFree(dQ); cuMemFree(dK); cuMemFree(dV); cuMemFree(dS); cuMemFree(dA); cuMemFree(dO);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hQ); free(hK); free(hV); free(hO); free(hA); free(sc); free(at); free(ptx);
        return abad ? 1 : 0;
    }

    /* ce_softmax_grad mode: cuda_launch <ptx> gpu_ce_softmax_grad <Nignored> ce_softmax_grad <rows> <cols>.
     * softmax-cross-entropy backward: dlogits[r,c]=softmax(logits[r,:])[c]-onehot(tgt[r])[c]. Targets
     * passed as f32 (class index as float). Verify cell-by-cell vs a CPU softmax-minus-onehot ref
     * (tol 1e-4) AND the INDEPENDENT conservation property that each grad row sums to ~0 (a missing
     * or misplaced onehot breaks this even when every cell formula looks plausible). */
    if (strcmp(op, "ce_softmax_grad") == 0) {
        int rows = (argc > 5) ? atoi(argv[5]) : 8;
        int cols = (argc > 6) ? atoi(argv[6]) : 16;
        size_t ne = (size_t)rows * cols;
        float* hL = (float*)malloc(ne * sizeof(float));
        float* hG = (float*)malloc(ne * sizeof(float));
        float* htg = (float*)malloc((size_t)rows * sizeof(float));
        if (!hL || !hG || !htg) return 2;
        for (size_t i = 0; i < ne; i++) hL[i] = (float)((int)((i * 7 + 3) % 13) - 6);
        for (int r = 0; r < rows; r++) htg[r] = (float)((r * 5 + 2) % cols);
        CUdeviceptr dL, dT, dG;
        CK(cuMemAlloc(&dL, ne * sizeof(float)), "alloc L");
        CK(cuMemAlloc(&dT, (size_t)rows * sizeof(float)), "alloc T");
        CK(cuMemAlloc(&dG, ne * sizeof(float)), "alloc G");
        CK(cuMemcpyHtoD(dL, hL, ne * sizeof(float)), "H2D L");
        CK(cuMemcpyHtoD(dT, htg, (size_t)rows * sizeof(float)), "H2D T");
        void* cargs[] = { &dL, &dT, &dG, &rows, &cols };
        CK(cuLaunchKernel(fn, rows, 1, 1, 1, 1, 1, 0, 0, cargs, 0), "launch ce_grad");
        CK(cuCtxSynchronize(), "sync ce_grad");
        CK(cuMemcpyDtoH(hG, dG, ne * sizeof(float)), "D2H G");
        int cbad = 0; float g00 = 0.0f, ref00 = 0.0f, maxrs = 0.0f;
        for (int r = 0; r < rows; r++) {
            float mx = hL[r * cols]; for (int c = 1; c < cols; c++) if (hL[r * cols + c] > mx) mx = hL[r * cols + c];
            float sm = 0.0f; for (int c = 0; c < cols; c++) sm += expf(hL[r * cols + c] - mx);
            int tgt = (int)htg[r];
            float rs = 0.0f;
            for (int c = 0; c < cols; c++) {
                float p = expf(hL[r * cols + c] - mx) / sm;
                float ref = p - ((c == tgt) ? 1.0f : 0.0f);
                float got = hG[r * cols + c];
                if (r == 0 && c == 0) { g00 = got; ref00 = ref; }
                rs += got;
                float e = got - ref; if (e < 0) e = -e;
                if (isnan(got) || e > 1.0e-4f) { if (cbad < 4) fprintf(stderr, "ce_grad mismatch g[%d,%d]=%g ref %g\n", r, c, got, ref); cbad++; }
            }
            float ers = rs < 0 ? -rs : rs; if (ers > maxrs) maxrs = ers;
            if (isnan(rs) || ers > 1.0e-3f) { if (cbad < 4) fprintf(stderr, "ce_grad row %d sum %g (want 0)\n", r, rs); cbad++; }
        }
        printf("GPU [%s] ce_softmax_grad %dx%d: g[0,0]=%g ref %g, max|row_sum|=%g, %d bad -> %s\n",
               gpu, rows, cols, g00, ref00, maxrs, cbad, cbad ? "FAIL" : "PASS");
        cuMemFree(dL); cuMemFree(dT); cuMemFree(dG);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hL); free(hG); free(htg); free(ptx);
        return cbad ? 1 : 0;
    }

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

    /* matmul_atb mode: cuda_launch <ptx> gpu_matmul_atb <Nignored> matmul_atb <M> <K> <N>.
     * C[K,N] = A[M,K]^T @ B[M,N], i.e. C[i,j]=sum_t A[t,i]*B[t,j] -- the dW=X^T@dY backward
     * workhorse (the 3rd matmul variant). gridDim=K, blockDim=N. Integer inputs so the result
     * is exact in f32; verify every K*N cell vs a CPU A^T@B in the kernel's accumulation order. */
    if (strcmp(op, "matmul_atb") == 0) {
        int M = (argc > 5) ? atoi(argv[5]) : 16;
        int K = (argc > 6) ? atoi(argv[6]) : 16;
        int Nn = (argc > 7) ? atoi(argv[7]) : 16;
        size_t aN = (size_t)M * K, bN = (size_t)M * Nn, cN = (size_t)K * Nn;
        float* hA = (float*)malloc(aN * sizeof(float));
        float* hB = (float*)malloc(bN * sizeof(float));
        float* hC = (float*)malloc(cN * sizeof(float));
        if (!hA || !hB || !hC) return 2;
        for (size_t i = 0; i < aN; i++) hA[i] = (float)(i % 7);
        for (size_t i = 0; i < bN; i++) hB[i] = (float)(i % 5);
        CUdeviceptr dA, dB, dC;
        CK(cuMemAlloc(&dA, aN * sizeof(float)), "alloc A");
        CK(cuMemAlloc(&dB, bN * sizeof(float)), "alloc B");
        CK(cuMemAlloc(&dC, cN * sizeof(float)), "alloc C");
        CK(cuMemcpyHtoD(dA, hA, aN * sizeof(float)), "H2D A");
        CK(cuMemcpyHtoD(dB, hB, bN * sizeof(float)), "H2D B");
        void* targs[] = { &dA, &dB, &dC, &M, &K, &Nn };
        CK(cuLaunchKernel(fn, K, 1, 1, Nn, 1, 1, 0, 0, targs, 0), "launch matmul_atb");
        CK(cuCtxSynchronize(), "sync matmul_atb");
        CK(cuMemcpyDtoH(hC, dC, cN * sizeof(float)), "D2H C");
        int tbad = 0;
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < Nn; j++) {
                float ref = hA[0 * K + i] * hB[0 * Nn + j];
                for (int t = 1; t < M; t++) ref += hA[t * K + i] * hB[t * Nn + j];
                float got = hC[i * Nn + j];
                if (got != ref) { if (tbad < 4) fprintf(stderr, "matmul_atb mismatch C[%d,%d]=%g ref %g\n", i, j, got, ref); tbad++; }
            }
        }
        printf("GPU [%s] matmul_atb (A^T@B) %dx%dx%d over %d cells: C[1,1]=%g, %d bad -> %s\n",
               gpu, M, K, Nn, K * Nn, hC[1 * Nn + 1], tbad, tbad ? "FAIL" : "PASS");
        cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hA); free(hB); free(hC); free(ptx);
        return tbad ? 1 : 0;
    }

    /* softmax mode: cuda_launch <ptx> <kernel> <Nignored> softmax <rows> <cols>.
     * One thread per row (gridDim.x=rows, blockDim.x=1). Non-constant inputs so each
     * row differs; verify every cell vs a CPU max-subtract softmax (tol 1e-3 for
     * ex2.approx) AND that each row sums to ~1. */
    if (strcmp(op, "softmax") == 0) {
        int rows = (argc > 5) ? atoi(argv[5]) : 8;
        int cols = (argc > 6) ? atoi(argv[6]) : 16;
        size_t ne = (size_t)rows * cols;
        float* hx = (float*)malloc(ne * sizeof(float));
        float* hy = (float*)malloc(ne * sizeof(float));
        if (!hx || !hy) return 2;
        for (size_t i = 0; i < ne; i++) { hx[i] = (float)((int)((i * 7 + 3) % 13) - 6); hy[i] = -1.0f; }
        CUdeviceptr dx, dy;
        CK(cuMemAlloc(&dx, ne * sizeof(float)), "cuMemAlloc x");
        CK(cuMemAlloc(&dy, ne * sizeof(float)), "cuMemAlloc y");
        CK(cuMemcpyHtoD(dx, hx, ne * sizeof(float)), "cuMemcpyHtoD x");
        void* sargs[] = { &dx, &dy, &rows, &cols };
        CK(cuLaunchKernel(fn, rows, 1, 1, 1, 1, 1, 0, 0, sargs, 0), "cuLaunchKernel softmax");
        CK(cuCtxSynchronize(), "cuCtxSynchronize");
        CK(cuMemcpyDtoH(hy, dy, ne * sizeof(float)), "cuMemcpyDtoH y");
        int sbad = 0;
        for (int r = 0; r < rows; r++) {
            float mx = hx[r * cols];
            for (int j = 1; j < cols; j++) if (hx[r * cols + j] > mx) mx = hx[r * cols + j];
            float sm = 0.0f;
            for (int kk = 0; kk < cols; kk++) sm += expf(hx[r * cols + kk] - mx);
            float rowsum = 0.0f;
            for (int c = 0; c < cols; c++) {
                float ref = expf(hx[r * cols + c] - mx) / sm;
                float got = hy[r * cols + c];
                rowsum += got;
                float d = got - ref; if (d < 0) d = -d;
                if (isnan(got) || d > 1.0e-3f) { if (sbad < 4) fprintf(stderr, "softmax mismatch y[%d,%d]=%g ref %g\n", r, c, got, ref); sbad++; }
            }
            float ds = rowsum - 1.0f; if (ds < 0) ds = -ds;
            if (isnan(rowsum) || ds > 1.0e-3f) { if (sbad < 4) fprintf(stderr, "softmax row %d sum %g (want 1)\n", r, rowsum); sbad++; }
        }
        float r0s = 0.0f; for (int c = 0; c < cols; c++) r0s += hy[c];
        printf("GPU [%s] softmax %dx%d: row0 sum=%g (want 1), %d bad -> %s\n",
               gpu, rows, cols, r0s, sbad, sbad ? "FAIL" : "PASS");
        cuMemFree(dx); cuMemFree(dy);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hx); free(hy); free(ptx);
        return sbad ? 1 : 0;
    }

    /* layernorm mode: cuda_launch <ptx> <kernel> <Nignored> layernorm <rows> <cols>.
     * TWO passes. pass 0: gamma=1,beta=0 -> verify each cell vs a CPU layernorm AND
     * that each row of y has mean~0, var~1. pass 1: non-trivial per-column gamma/beta
     * -> verify each cell vs an affine-aware reference gamma[c]*norm+beta[c], which
     * exercises the kernel's gamma[t]/beta[t] load+store path (a kernel that ignored
     * gamma/beta would pass pass 0 but mismatch pass 1). All checks NaN-guarded; the
     * real kernel must pass BOTH passes. */
    if (strcmp(op, "layernorm") == 0) {
        int rows = (argc > 5) ? atoi(argv[5]) : 8;
        int cols = (argc > 6) ? atoi(argv[6]) : 16;
        size_t ne = (size_t)rows * cols;
        float* hx = (float*)malloc(ne * sizeof(float));
        float* hy = (float*)malloc(ne * sizeof(float));
        float* hg = (float*)malloc((size_t)cols * sizeof(float));
        float* hbe = (float*)malloc((size_t)cols * sizeof(float));
        if (!hx || !hy || !hg || !hbe) return 2;
        for (size_t i = 0; i < ne; i++) { hx[i] = (float)((int)((i * 7 + 3) % 13) - 6); }
        CUdeviceptr dx, dy, dg, db;
        CK(cuMemAlloc(&dx, ne * sizeof(float)), "cuMemAlloc x");
        CK(cuMemAlloc(&dy, ne * sizeof(float)), "cuMemAlloc y");
        CK(cuMemAlloc(&dg, (size_t)cols * sizeof(float)), "cuMemAlloc g");
        CK(cuMemAlloc(&db, (size_t)cols * sizeof(float)), "cuMemAlloc b");
        CK(cuMemcpyHtoD(dx, hx, ne * sizeof(float)), "cuMemcpyHtoD x");
        void* largs[] = { &dx, &dy, &dg, &db, &cols };
        int lbad = 0;
        float r0m = 0.0f;
        for (int pass = 0; pass < 2; pass++) {
            for (int c = 0; c < cols; c++) {
                if (pass == 0) { hg[c] = 1.0f; hbe[c] = 0.0f; }
                else { hg[c] = 1.0f + 0.25f * (float)(c % 4); hbe[c] = 0.5f * (float)((c % 3) - 1); }
            }
            for (size_t i = 0; i < ne; i++) hy[i] = -7.0f;
            CK(cuMemcpyHtoD(dy, hy, ne * sizeof(float)), "cuMemcpyHtoD y");
            CK(cuMemcpyHtoD(dg, hg, (size_t)cols * sizeof(float)), "cuMemcpyHtoD g");
            CK(cuMemcpyHtoD(db, hbe, (size_t)cols * sizeof(float)), "cuMemcpyHtoD b");
            CK(cuLaunchKernel(fn, rows, 1, 1, 1, 1, 1, 0, 0, largs, 0), "cuLaunchKernel layernorm");
            CK(cuCtxSynchronize(), "cuCtxSynchronize");
            CK(cuMemcpyDtoH(hy, dy, ne * sizeof(float)), "cuMemcpyDtoH y");
            for (int r = 0; r < rows; r++) {
                float mean = 0.0f; for (int c = 0; c < cols; c++) mean += hx[r * cols + c]; mean /= (float)cols;
                float v = 0.0f; for (int c = 0; c < cols; c++) { float d = hx[r * cols + c] - mean; v += d * d; } v /= (float)cols;
                float inv = 1.0f / sqrtf(v);
                for (int c = 0; c < cols; c++) {
                    float norm = (hx[r * cols + c] - mean) * inv;
                    float ref = hg[c] * norm + hbe[c];
                    float got = hy[r * cols + c];
                    float d = got - ref; if (d < 0) d = -d;
                    if (isnan(got) || d > 1.0e-3f) { if (lbad < 4) fprintf(stderr, "layernorm[g/b pass %d] mismatch y[%d,%d]=%g ref %g\n", pass, r, c, got, ref); lbad++; }
                }
                if (pass == 0) {
                    float ym = 0.0f; for (int c = 0; c < cols; c++) ym += hy[r * cols + c]; ym /= (float)cols;
                    float yv = 0.0f; for (int c = 0; c < cols; c++) { float dd = hy[r * cols + c] - ym; yv += dd * dd; } yv /= (float)cols;
                    float dm = ym < 0 ? -ym : ym;
                    float dv = (yv - 1.0f) < 0 ? -(yv - 1.0f) : (yv - 1.0f);
                    if (isnan(ym) || dm > 1.0e-2f) { if (lbad < 4) fprintf(stderr, "layernorm row %d mean %g (want 0)\n", r, ym); lbad++; }
                    if (isnan(yv) || dv > 1.0e-2f) { if (lbad < 4) fprintf(stderr, "layernorm row %d var %g (want 1)\n", r, yv); lbad++; }
                    if (r == 0) r0m = ym;
                }
            }
        }
        printf("GPU [%s] layernorm %dx%d (gamma/beta affine-checked): row0 mean=%g (want 0), %d bad -> %s\n",
               gpu, rows, cols, r0m, lbad, lbad ? "FAIL" : "PASS");
        cuMemFree(dx); cuMemFree(dy); cuMemFree(dg); cuMemFree(db);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hx); free(hy); free(hg); free(hbe);
        return lbad ? 1 : 0;
    }

    /* gelu mode: cuda_launch <ptx> gpu_gelu <N> gelu. y=0.5*x*(1+tanh(0.7978846*
     * (x+0.044715*x^3))), the tanh GELU. CPU ref mirrors stdlib __gelu exactly.
     * Inputs in [-3,3] so e^(2z) never saturates. Combines f32 literals + __gpu_exp. */
    if (strcmp(op, "gelu") == 0) {
        size_t ne = (size_t)N;
        float* hx = (float*)malloc(ne * sizeof(float));
        float* hy = (float*)malloc(ne * sizeof(float));
        if (!hx || !hy) return 2;
        for (size_t i = 0; i < ne; i++) { hx[i] = (float)((int)(i % 61) - 30) * 0.1f; hy[i] = -7.0f; }
        CUdeviceptr dx, dy;
        CK(cuMemAlloc(&dx, ne * sizeof(float)), "cuMemAlloc x");
        CK(cuMemAlloc(&dy, ne * sizeof(float)), "cuMemAlloc y");
        CK(cuMemcpyHtoD(dx, hx, ne * sizeof(float)), "cuMemcpyHtoD x");
        CK(cuMemcpyHtoD(dy, hy, ne * sizeof(float)), "cuMemcpyHtoD y");
        void* gargs[] = { &dx, &dy, &N };
        CK(cuLaunchKernel(fn, N, 1, 1, 1, 1, 1, 0, 0, gargs, 0), "cuLaunchKernel gelu");
        CK(cuCtxSynchronize(), "cuCtxSynchronize");
        CK(cuMemcpyDtoH(hy, dy, ne * sizeof(float)), "cuMemcpyDtoH y");
        int gbad = 0; float gref0 = 0.0f;
        for (size_t i = 0; i < ne; i++) {
            float xx = hx[i];
            float x3 = xx * xx * xx;
            float inner = 0.7978846f * (xx + 0.044715f * x3);
            float e2 = expf(2.0f * inner);
            float th = (e2 - 1.0f) / (e2 + 1.0f);
            float ref = 0.5f * xx * (1.0f + th);
            if (i == 0) gref0 = ref;
            float got = hy[i];
            float d = got - ref; if (d < 0) d = -d;
            if (isnan(got) || d > 1.0e-3f) { if (gbad < 4) fprintf(stderr, "gelu mismatch y[%zu]=%g ref %g (x=%g)\n", i, got, ref, xx); gbad++; }
        }
        printf("GPU [%s] gelu N=%d (tanh approx, f32 literals + exp): y[0]=%g ref %g, %d bad -> %s\n",
               gpu, N, hy[0], gref0, gbad, gbad ? "FAIL" : "PASS");
        cuMemFree(dx); cuMemFree(dy);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hx); free(hy);
        return gbad ? 1 : 0;
    }

    /* affine probe: cuda_launch <ptx> gpu_affine <N> affine. y[i]=0.5*x[i]+0.25,
     * validating the f32-LITERAL PTX emitter (0.5=0f3F000000, 0.25=0f3E800000). */
    if (strcmp(op, "affine") == 0) {
        size_t ne = (size_t)N;
        float* hx = (float*)malloc(ne * sizeof(float));
        float* hy = (float*)malloc(ne * sizeof(float));
        if (!hx || !hy) return 2;
        for (size_t i = 0; i < ne; i++) { hx[i] = (float)((int)((i * 7 + 3) % 13) - 6); hy[i] = -7.0f; }
        CUdeviceptr dx, dy;
        CK(cuMemAlloc(&dx, ne * sizeof(float)), "cuMemAlloc x");
        CK(cuMemAlloc(&dy, ne * sizeof(float)), "cuMemAlloc y");
        CK(cuMemcpyHtoD(dx, hx, ne * sizeof(float)), "cuMemcpyHtoD x");
        CK(cuMemcpyHtoD(dy, hy, ne * sizeof(float)), "cuMemcpyHtoD y");
        void* aargs[] = { &dx, &dy, &N };
        CK(cuLaunchKernel(fn, N, 1, 1, 1, 1, 1, 0, 0, aargs, 0), "cuLaunchKernel affine");
        CK(cuCtxSynchronize(), "cuCtxSynchronize");
        CK(cuMemcpyDtoH(hy, dy, ne * sizeof(float)), "cuMemcpyDtoH y");
        int abad = 0;
        for (size_t i = 0; i < ne; i++) {
            float ref = 0.5f * hx[i] + 0.25f;
            float got = hy[i];
            float d = got - ref; if (d < 0) d = -d;
            if (isnan(got) || d > 1.0e-4f) { if (abad < 4) fprintf(stderr, "affine mismatch y[%zu]=%g ref %g\n", i, got, ref); abad++; }
        }
        printf("GPU [%s] affine N=%d (y=0.5x+0.25, f32 literals): y[0]=%g ref %g, %d bad -> %s\n",
               gpu, N, hy[0], 0.5f * hx[0] + 0.25f, abad, abad ? "FAIL" : "PASS");
        cuMemFree(dx); cuMemFree(dy);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hx); free(hy);
        return abad ? 1 : 0;
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
        if (is_exp) { float d = hc[i] - want; if (d < 0) d = -d; float aw = want < 0 ? -want : want; bad_i = (isnan(hc[i]) || d > 1.0e-3f * (aw + 1.0e-6f)); }
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
