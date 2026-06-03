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
#include <cublas_v2.h>   /* G1: fenced cuBLAS f32 GEMM oracle (gemm_perf mode only); link -lcublas */
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

/* L for one layernorm row with gamma and a perturbation delta at column cc -- the
 * forward used by the dx finite-difference. y[k]=gamma[k]*(x[k]-mean)/std + beta[k];
 * beta cancels in the central difference so it is omitted. L = sum_k dy[k]*y[k]. */
static float ln_rowL(const float* xr, const float* dyr, const float* gam, int cols, int cc, float delta) {
    float mean = 0.0f;
    for (int k = 0; k < cols; k++) mean += xr[k] + (k == cc ? delta : 0.0f);
    mean /= (float)cols;
    float var = 0.0f;
    for (int k = 0; k < cols; k++) { float v = xr[k] + (k == cc ? delta : 0.0f) - mean; var += v * v; }
    var /= (float)cols;
    float istd = 1.0f / sqrtf(var);
    float L = 0.0f;
    for (int k = 0; k < cols; k++) { float xv = xr[k] + (k == cc ? delta : 0.0f); L += dyr[k] * gam[k] * (xv - mean) * istd; }
    return L;
}

/* CPU single-head attention forward: out = softmax(0.25*Q@K^T) @ V, all [S,d] row-major.
 * Used by the attn_backward finite-difference (perturb an input, recompute out, L=sum(dOut*out)).
 * S is bounded by 64 here so the SxS scratch fits the static buffers. */
static void attn_forward_cpu(const float* Q, const float* K, const float* V, float* out, int S, int d) {
    static float sc[4096]; static float at[4096];
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) { float dot = 0.0f; for (int t = 0; t < d; t++) dot += Q[i * d + t] * K[j * d + t]; sc[i * S + j] = 0.25f * dot; }
        float mx = sc[i * S]; for (int j = 1; j < S; j++) if (sc[i * S + j] > mx) mx = sc[i * S + j];
        float sm = 0.0f; for (int j = 0; j < S; j++) sm += expf(sc[i * S + j] - mx);
        for (int j = 0; j < S; j++) at[i * S + j] = expf(sc[i * S + j] - mx) / sm;
    }
    for (int i = 0; i < S; i++) for (int t = 0; t < d; t++) { float o = 0.0f; for (int j = 0; j < S; j++) o += at[i * S + j] * V[j * d + t]; out[i * d + t] = o; }
}

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

    /* attn_backward mode: cuda_launch <combined.ptx> gpu_qkt <Nignored> attn_backward <S> <d>.
     * combined.ptx has 7 entries (gpu_qkt, gpu_softmax, naive_matmul, gpu_matmul_abt,
     * gpu_matmul_atb, gpu_softmax_backward, gpu_scale_inplace). Single-head attention forward
     * (scores=0.25*Q@K^T -> softmax -> out=attn@V) + backward given dOut: dV=attn^T@dOut,
     * d_attn=dOut@V^T, d_scores=softmax_bwd(attn,d_attn), dQ=0.25*d_scores@K, dK=0.25*d_scores^T@Q.
     * Verify dQ/dK/dV vs a central finite-difference of L=sum(dOut*out) wrt Q/K/V (INDEPENDENT;
     * uses only the forward; tol 2e-2). mod already loaded from argv[1]. */
    if (strcmp(op, "attn_backward") == 0) {
        int S = (argc > 5) ? atoi(argv[5]) : 8;
        int d = (argc > 6) ? atoi(argv[6]) : 16;
        CUfunction f_qkt, f_sm, f_mm, f_abt, f_atb, f_smb, f_sc;
        CK(cuModuleGetFunction(&f_qkt, mod, "gpu_qkt"), "get qkt");
        CK(cuModuleGetFunction(&f_sm, mod, "gpu_softmax"), "get softmax");
        CK(cuModuleGetFunction(&f_mm, mod, "naive_matmul"), "get matmul");
        CK(cuModuleGetFunction(&f_abt, mod, "gpu_matmul_abt"), "get abt");
        CK(cuModuleGetFunction(&f_atb, mod, "gpu_matmul_atb"), "get atb");
        CK(cuModuleGetFunction(&f_smb, mod, "gpu_softmax_backward"), "get sm_bwd");
        CK(cuModuleGetFunction(&f_sc, mod, "gpu_scale_inplace"), "get scale");
        size_t sd = (size_t)S * d, ss = (size_t)S * S;
        float* hQ = (float*)malloc(sd * sizeof(float));
        float* hK = (float*)malloc(sd * sizeof(float));
        float* hV = (float*)malloc(sd * sizeof(float));
        float* hdO = (float*)malloc(sd * sizeof(float));
        float* hdV = (float*)malloc(sd * sizeof(float));
        float* hdQ = (float*)malloc(sd * sizeof(float));
        float* hdK = (float*)malloc(sd * sizeof(float));
        float* tmp = (float*)malloc(sd * sizeof(float));
        float* ob = (float*)malloc(sd * sizeof(float));
        if (!hQ || !hK || !hV || !hdO || !hdV || !hdQ || !hdK || !tmp || !ob) return 2;
        for (size_t i = 0; i < sd; i++) {
            hQ[i] = (float)((int)(i % 7) - 3) * 0.1f;
            hK[i] = (float)((int)(i % 5) - 2) * 0.1f;
            hV[i] = (float)((int)(i % 9) - 4) * 0.1f;
            hdO[i] = (float)((int)(i % 6) - 2) * 0.1f;
        }
        CUdeviceptr gQ, gK, gV, gdO, gSc, gAt, gDat, gDsc, gOut, gDV, gDQ, gDK;
        CK(cuMemAlloc(&gQ, sd * sizeof(float)), "Q"); CK(cuMemAlloc(&gK, sd * sizeof(float)), "K");
        CK(cuMemAlloc(&gV, sd * sizeof(float)), "V"); CK(cuMemAlloc(&gdO, sd * sizeof(float)), "dO");
        CK(cuMemAlloc(&gSc, ss * sizeof(float)), "Sc"); CK(cuMemAlloc(&gAt, ss * sizeof(float)), "At");
        CK(cuMemAlloc(&gDat, ss * sizeof(float)), "Dat"); CK(cuMemAlloc(&gDsc, ss * sizeof(float)), "Dsc");
        CK(cuMemAlloc(&gOut, sd * sizeof(float)), "Out"); CK(cuMemAlloc(&gDV, sd * sizeof(float)), "DV");
        CK(cuMemAlloc(&gDQ, sd * sizeof(float)), "DQ"); CK(cuMemAlloc(&gDK, sd * sizeof(float)), "DK");
        CK(cuMemcpyHtoD(gQ, hQ, sd * sizeof(float)), "h2d Q"); CK(cuMemcpyHtoD(gK, hK, sd * sizeof(float)), "h2d K");
        CK(cuMemcpyHtoD(gV, hV, sd * sizeof(float)), "h2d V"); CK(cuMemcpyHtoD(gdO, hdO, sd * sizeof(float)), "h2d dO");
        int Si = S, di = d, sdi = (int)sd;
        /* FORWARD */
        void* fa[] = { &gQ, &gK, &gSc, &Si, &di }; CK(cuLaunchKernel(f_qkt, S, 1, 1, S, 1, 1, 0, 0, fa, 0), "qkt"); CK(cuCtxSynchronize(), "s");
        void* fb[] = { &gSc, &gAt, &Si, &Si }; CK(cuLaunchKernel(f_sm, S, 1, 1, 1, 1, 1, 0, 0, fb, 0), "sm"); CK(cuCtxSynchronize(), "s");
        void* fc[] = { &gAt, &gV, &gOut, &Si, &Si, &di }; CK(cuLaunchKernel(f_mm, S, 1, 1, d, 1, 1, 0, 0, fc, 0), "mm"); CK(cuCtxSynchronize(), "s");
        /* BACKWARD */
        void* b1[] = { &gAt, &gdO, &gDV, &Si, &Si, &di }; CK(cuLaunchKernel(f_atb, S, 1, 1, d, 1, 1, 0, 0, b1, 0), "dV"); CK(cuCtxSynchronize(), "s");
        void* b2[] = { &gdO, &gV, &gDat, &Si, &di, &Si }; CK(cuLaunchKernel(f_abt, S, 1, 1, S, 1, 1, 0, 0, b2, 0), "dAttn"); CK(cuCtxSynchronize(), "s");
        void* b3[] = { &gAt, &gDat, &gDsc, &Si, &Si }; CK(cuLaunchKernel(f_smb, S, 1, 1, 1, 1, 1, 0, 0, b3, 0), "dScores"); CK(cuCtxSynchronize(), "s");
        void* b4[] = { &gDsc, &gK, &gDQ, &Si, &Si, &di }; CK(cuLaunchKernel(f_mm, S, 1, 1, d, 1, 1, 0, 0, b4, 0), "dQ"); CK(cuCtxSynchronize(), "s");
        void* s1[] = { &gDQ, &sdi }; CK(cuLaunchKernel(f_sc, sdi, 1, 1, 1, 1, 1, 0, 0, s1, 0), "scaleQ"); CK(cuCtxSynchronize(), "s");
        void* b5[] = { &gDsc, &gQ, &gDK, &Si, &Si, &di }; CK(cuLaunchKernel(f_atb, S, 1, 1, d, 1, 1, 0, 0, b5, 0), "dK"); CK(cuCtxSynchronize(), "s");
        void* s2[] = { &gDK, &sdi }; CK(cuLaunchKernel(f_sc, sdi, 1, 1, 1, 1, 1, 0, 0, s2, 0), "scaleK"); CK(cuCtxSynchronize(), "s");
        CK(cuMemcpyDtoH(hdV, gDV, sd * sizeof(float)), "d2h dV");
        CK(cuMemcpyDtoH(hdQ, gDQ, sd * sizeof(float)), "d2h dQ");
        CK(cuMemcpyDtoH(hdK, gDK, sd * sizeof(float)), "d2h dK");
        /* finite-difference of L=sum(dOut*out) wrt Q (->dQ), K (->dK), V (->dV). */
        int bad = 0; float maxe = 0.0f, q0 = hdQ[0], q0r = 0.0f, h = 1.0e-3f;
        for (int pass = 0; pass < 3; pass++) {
            float* base = (pass == 0) ? hQ : (pass == 1) ? hK : hV;
            float* got = (pass == 0) ? hdQ : (pass == 1) ? hdK : hdV;
            for (size_t i = 0; i < sd; i++) {
                for (size_t k = 0; k < sd; k++) tmp[k] = base[k];
                tmp[i] = base[i] + h;
                attn_forward_cpu(pass == 0 ? tmp : hQ, pass == 1 ? tmp : hK, pass == 2 ? tmp : hV, ob, S, d);
                float Lp = 0.0f; for (size_t k = 0; k < sd; k++) Lp += hdO[k] * ob[k];
                tmp[i] = base[i] - h;
                attn_forward_cpu(pass == 0 ? tmp : hQ, pass == 1 ? tmp : hK, pass == 2 ? tmp : hV, ob, S, d);
                float Lm = 0.0f; for (size_t k = 0; k < sd; k++) Lm += hdO[k] * ob[k];
                float fd = (Lp - Lm) / (2.0f * h);
                if (pass == 0 && i == 0) q0r = fd;
                float e = got[i] - fd; if (e < 0) e = -e; if (e > maxe) maxe = e;
                float af = fd < 0 ? -fd : fd;
                /* magnitude-aware: an absolute floor 1e-3 (ignores finite-diff/f32 noise on
                 * near-zero grads) AND a 5% relative bound (catches multiplicative errors like a
                 * wrong backward scale on the small dQ/dK gradients -- an audit found an
                 * absolute-only 2e-2 tol missed a 2x scale error here). */
                if (isnan(got[i]) || (e > 1.0e-3f && e > 0.05f * af)) { if (bad < 6) fprintf(stderr, "attn_bwd pass %d [%zu] got %g fd %g\n", pass, i, got[i], fd); bad++; }
            }
        }
        printf("GPU [%s] attn_backward S=%d d=%d: dQ[0]=%g fd %g, max|grad-fd|=%g, %d bad -> %s\n",
               gpu, S, d, q0, q0r, maxe, bad, bad ? "FAIL" : "PASS");
        cuMemFree(gQ); cuMemFree(gK); cuMemFree(gV); cuMemFree(gdO); cuMemFree(gSc); cuMemFree(gAt);
        cuMemFree(gDat); cuMemFree(gDsc); cuMemFree(gOut); cuMemFree(gDV); cuMemFree(gDQ); cuMemFree(gDK);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hQ); free(hK); free(hV); free(hdO); free(hdV); free(hdQ); free(hdK); free(tmp); free(ob); free(ptx);
        return bad ? 1 : 0;
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

    /* gemm_perf mode (T2/M1 correctness + T2/G1 perf): cuda_launch <ptx> tiled_matmul
     *   <Nignored> gemm_perf <M> <K> <N> [mutate].
     * Launches the kovc SMEM-tiled GEMM (emit_ptx_tiled_matmul_smem) on the device.
     * The launch geometry is read from the emitted PTX-implied tile params via the
     * BM/BN/BK constants below (kept in sync with kovc.hx's emit_ptx_tiled_matmul_smem).
     *
     * G1 (this chunk):
     *  (a) FENCED cuBLAS ORACLE (cublasSgemm, column-major -> swapped operands for a
     *      row-major C=A*B), forced to CUBLAS_PEDANTIC_MATH so it is a TRUE-f32 oracle
     *      (no TF32 tensor-core contamination), and VALIDATED vs the CPU oracle FIRST so
     *      the oracle itself is trusted before it judges the kovc kernel. Link with
     *      -lcublas -L/usr/local/cuda/lib64.
     *  (b) CORRECTNESS: every M*N cell of the kovc result vs the CPU oracle (integer
     *      inputs a[i]=i%7,b[i]=i%5 -> sums < 2^24 are EXACT in f32, == compare) AND vs
     *      the cuBLAS oracle cell-by-cell within tol (1e-3 rel; integer-exact so it is
     *      really 0). A mismatch -> exit 1.
     *  (c) TIMING via cuEvent: ~5 warmup + ~50 timed KERNEL-ONLY launches, report
     *      min/median/max ms (the laptop throttles), TFLOP/s = 2*M*N*K / median_seconds,
     *      for BOTH the kovc kernel and the (pedantic, true-f32) cuBLAS reference, plus
     *      the kovc/cuBLAS ratio. The >=3 TFLOP/s G1 bar is checked by the orchestrator
     *      (scripts/gpu_perf_corpus.sh) which parses the "MEDIAN-TFLOPS" line below.
     * The optional "mutate" arg perturbs one C cell pre-compare to prove the comparator
     * has teeth (comparator negative control -> must FAIL). The barrier-removal negative
     * control (a bar.sync-stripped PTX must mis-compute, proving .shared/bar.sync are
     * load-bearing) is driven by the orchestrator, which re-runs THIS mode on a PTX with
     * the bar.sync lines deleted and asserts it FAILs.
     * Requires M%BM==N%BN==K%BK==0 (asserted) so the launch covers all of C. */
    if (strcmp(op, "gemm_perf") == 0) {
        int Md = (argc > 5) ? atoi(argv[5]) : 64;
        int Kd = (argc > 6) ? atoi(argv[6]) : 64;
        int Nd = (argc > 7) ? atoi(argv[7]) : 64;
        int mutate = (argc > 8 && strcmp(argv[8], "mutate") == 0);
        /* Tile params -- KEEP IN SYNC with kovc.hx emit_ptx_tiled_matmul_smem. The block
         * is (BN/TN)x(BM/TM) threads; grid is (N/BN, M/BM). Override via env for the G1
         * upscale experiments without recompiling the launcher. */
        int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
        { const char* e;
          if ((e = getenv("GEMM_BM"))) BM = atoi(e);
          if ((e = getenv("GEMM_BN"))) BN = atoi(e);
          if ((e = getenv("GEMM_BK"))) BK = atoi(e);
          if ((e = getenv("GEMM_TM"))) TM = atoi(e);
          if ((e = getenv("GEMM_TN"))) TN = atoi(e); }
        int bdimx = BN / TN, bdimy = BM / TM;   /* threads per block in x/y */
        /* tile-divisibility assert: billed cells == computed cells, no boundary code. */
        if (Md % BM != 0 || Nd % BN != 0 || Kd % BK != 0) {
            fprintf(stderr, "gemm_perf: dims must satisfy M%%%d==0 (%d), N%%%d==0 (%d), K%%%d==0 (%d)\n", BM, Md, BN, Nd, BK, Kd);
            return 2;
        }
        size_t aN = (size_t)Md * Kd, bN = (size_t)Kd * Nd, cN = (size_t)Md * Nd;
        float* hA = (float*)malloc(aN * sizeof(float));
        float* hB = (float*)malloc(bN * sizeof(float));
        float* hC = (float*)malloc(cN * sizeof(float));
        float* hR = (float*)malloc(cN * sizeof(float));   /* cuBLAS result */
        if (!hA || !hB || !hC || !hR) return 2;
        for (size_t i = 0; i < aN; i++) hA[i] = (float)(i % 7);
        for (size_t i = 0; i < bN; i++) hB[i] = (float)(i % 5);
        for (size_t i = 0; i < cN; i++) hC[i] = -1.0f;
        CUdeviceptr dA, dB, dC, dR;
        CK(cuMemAlloc(&dA, aN * sizeof(float)), "alloc A");
        CK(cuMemAlloc(&dB, bN * sizeof(float)), "alloc B");
        CK(cuMemAlloc(&dC, cN * sizeof(float)), "alloc C");
        CK(cuMemAlloc(&dR, cN * sizeof(float)), "alloc R(cublas)");
        CK(cuMemcpyHtoD(dA, hA, aN * sizeof(float)), "H2D A");
        CK(cuMemcpyHtoD(dB, hB, bN * sizeof(float)), "H2D B");
        CK(cuMemcpyHtoD(dC, hC, cN * sizeof(float)), "H2D C");   /* sentinel so an unwritten cell shows */
        void* gargs[] = { &dA, &dB, &dC, &Md, &Kd, &Nd };
        unsigned gx = (unsigned)(Nd / BN), gy = (unsigned)(Md / BM);
        CK(cuLaunchKernel(fn, gx, gy, 1, (unsigned)bdimx, (unsigned)bdimy, 1, 0, 0, gargs, 0), "launch tiled_matmul");
        CK(cuCtxSynchronize(), "sync tiled_matmul");
        CK(cuMemcpyDtoH(hC, dC, cN * sizeof(float)), "D2H C");
        if (mutate) hC[0] += 1.0f;   /* comparator negative control: must trip the compare */

        /* --- fenced cuBLAS oracle: true-f32 (pedantic), validated vs CPU FIRST --- */
        cublasHandle_t cbh;
        if (cublasCreate(&cbh) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasCreate failed\n"); return 2; }
        cublasSetMathMode(cbh, CUBLAS_PEDANTIC_MATH);   /* TRUE f32 -- no TF32 contamination */
        float alpha = 1.0f, beta = 0.0f;
        /* row-major C(MxN)=A*B  <=>  column-major C^T(NxM)=B^T*A^T:
         *   Sgemm(N,N, N,M,K, B(ld N), A(ld K), C(ld N)). */
        if (cublasSgemm(cbh, CUBLAS_OP_N, CUBLAS_OP_N, Nd, Md, Kd, &alpha,
                        (const float*)dB, Nd, (const float*)dA, Kd, &beta, (float*)dR, Nd) != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cublasSgemm failed\n"); return 2;
        }
        CK(cuCtxSynchronize(), "sync cublas");
        CK(cuMemcpyDtoH(hR, dR, cN * sizeof(float)), "D2H R");

        /* The CPU-oracle triple-loops (1)+(2) are O(M*N*K) on ONE host thread -- at
         * 2048^3 that is ~17 GFLOP twice (minutes). They establish the chain-of-trust
         * CPU<-cuBLAS<-kovc and are run in FULL at the smaller corpus sizes; above a cell
         * budget they are SKIPPED and large-N correctness rests on (3) kovc-vs-cuBLAS
         * (GPU-fast) plus the already-trusted oracle. Override the budget via GEMM_CPU_MNK
         * (set 0 to force-skip). This is an honest, reported relaxation -- not a hidden one. */
        double mnk = (double)Md * (double)Nd * (double)Kd;
        double cpu_budget = 6.0e8;   /* ~840^3; covers 64..512 cubed in full */
        { const char* e; if ((e = getenv("GEMM_CPU_MNK"))) cpu_budget = atof(e); }
        int cpu_oracle_ran = (mnk <= cpu_budget);
        /* (1) trust the cuBLAS oracle: cuBLAS == CPU oracle, cell-by-cell. */
        int obad = 0;
        if (cpu_oracle_ran) {
            for (int r = 0; r < Md; r++) {
                for (int cc = 0; cc < Nd; cc++) {
                    float ref = 0.0f;
                    for (int t = 0; t < Kd; t++) ref += hA[r * Kd + t] * hB[t * Nd + cc];
                    float got = hR[r * Nd + cc];
                    float e = got - ref; if (e < 0) e = -e;
                    float aref = ref < 0 ? -ref : ref;
                    if (isnan(got) || (e > 1.0e-3f && e > 1.0e-3f * aref)) { if (obad < 6) fprintf(stderr, "cuBLAS-vs-CPU mismatch [%d,%d]=%g ref %g\n", r, cc, got, ref); obad++; }
                }
            }
        }
        /* (2) kovc kernel == CPU oracle (integer-exact, == compare). */
        int gbad = 0;
        if (cpu_oracle_ran) {
            for (int r = 0; r < Md; r++) {
                for (int cc = 0; cc < Nd; cc++) {
                    float ref = 0.0f;
                    for (int t = 0; t < Kd; t++) ref += hA[r * Kd + t] * hB[t * Nd + cc];
                    float got = hC[r * Nd + cc];
                    if (got != ref) { if (gbad < 6) fprintf(stderr, "gemm_perf(CPU) mismatch C[%d,%d]=%g ref %g\n", r, cc, got, ref); gbad++; }
                }
            }
        }
        /* (3) kovc kernel vs cuBLAS oracle, cell-by-cell within tol (1e-3 f32) -- O(M*N), GPU-trusted. */
        int cbad = 0;
        for (int r = 0; r < Md; r++) {
            for (int cc = 0; cc < Nd; cc++) {
                float got = hC[r * Nd + cc], refb = hR[r * Nd + cc];
                float e = got - refb; if (e < 0) e = -e;
                float ar = refb < 0 ? -refb : refb;
                if (isnan(got) || (e > 1.0e-3f && e > 1.0e-3f * ar)) { if (cbad < 6) fprintf(stderr, "gemm_perf(cuBLAS) mismatch C[%d,%d]=%g cublas %g\n", r, cc, got, refb); cbad++; }
            }
        }

        /* --- TIMING (kernel-only) via cuEvent: warmup + timed launches, min/med/max --- */
        double flop = 2.0 * (double)Md * (double)Nd * (double)Kd;
        double kov_med = 0.0, blas_med = 0.0;
        if (!mutate) {
            int WARM = 5, ITERS = 50;
            float* ts = (float*)malloc((size_t)ITERS * sizeof(float));
            CUevent e0, e1; CK(cuEventCreate(&e0, 0), "evt0"); CK(cuEventCreate(&e1, 0), "evt1");
            /* kovc kernel timing */
            for (int w = 0; w < WARM; w++) CK(cuLaunchKernel(fn, gx, gy, 1, (unsigned)bdimx, (unsigned)bdimy, 1, 0, 0, gargs, 0), "warm kov");
            CK(cuCtxSynchronize(), "sync warm kov");
            for (int it = 0; it < ITERS; it++) {
                CK(cuEventRecord(e0, 0), "rec0");
                CK(cuLaunchKernel(fn, gx, gy, 1, (unsigned)bdimx, (unsigned)bdimy, 1, 0, 0, gargs, 0), "time kov");
                CK(cuEventRecord(e1, 0), "rec1"); CK(cuEventSynchronize(e1), "evsync");
                CK(cuEventElapsedTime(&ts[it], e0, e1), "elapsed");
            }
            /* sort to get min/median/max */
            for (int i = 0; i < ITERS; i++) for (int j = i + 1; j < ITERS; j++) if (ts[j] < ts[i]) { float t = ts[i]; ts[i] = ts[j]; ts[j] = t; }
            float kmin = ts[0], kmax = ts[ITERS - 1], kmed = ts[ITERS / 2];
            kov_med = (double)kmed;
            /* cuBLAS (pedantic, true-f32) timing -- same protocol */
            for (int w = 0; w < WARM; w++) cublasSgemm(cbh, CUBLAS_OP_N, CUBLAS_OP_N, Nd, Md, Kd, &alpha, (const float*)dB, Nd, (const float*)dA, Kd, &beta, (float*)dR, Nd);
            CK(cuCtxSynchronize(), "sync warm blas");
            for (int it = 0; it < ITERS; it++) {
                CK(cuEventRecord(e0, 0), "brec0");
                cublasSgemm(cbh, CUBLAS_OP_N, CUBLAS_OP_N, Nd, Md, Kd, &alpha, (const float*)dB, Nd, (const float*)dA, Kd, &beta, (float*)dR, Nd);
                CK(cuEventRecord(e1, 0), "brec1"); CK(cuEventSynchronize(e1), "bevsync");
                CK(cuEventElapsedTime(&ts[it], e0, e1), "belapsed");
            }
            for (int i = 0; i < ITERS; i++) for (int j = i + 1; j < ITERS; j++) if (ts[j] < ts[i]) { float t = ts[i]; ts[i] = ts[j]; ts[j] = t; }
            float bmin = ts[0], bmax = ts[ITERS - 1], bmed = ts[ITERS / 2];
            blas_med = (double)bmed;
            double kov_tf = flop / (kov_med * 1.0e-3) / 1.0e12;
            double blas_tf = flop / (blas_med * 1.0e-3) / 1.0e12;
            printf("GPU [%s] TIMING kovc   %dx%dx%d: min=%.4f med=%.4f max=%.4f ms\n", gpu, Md, Kd, Nd, kmin, kmed, kmax);
            printf("GPU [%s] TIMING cuBLAS %dx%dx%d (pedantic f32): min=%.4f med=%.4f max=%.4f ms\n", gpu, Md, Kd, Nd, bmin, bmed, bmax);
            printf("MEDIAN-TFLOPS kovc=%.3f cublas=%.3f ratio=%.1f%%\n", kov_tf, blas_tf, 100.0 * kov_tf / blas_tf);
            cuEventDestroy(e0); cuEventDestroy(e1); free(ts);
        }
        cublasDestroy(cbh);

        int totbad = gbad + obad + cbad;
        printf("GPU [%s] tiled_matmul (SMEM %dx%d/BK%d/%dx%d) %dx%dx%d over %d cells%s: C[1,1]=%g, kovc-vs-CPU=%d oracle(cuBLAS-vs-CPU)=%d kovc-vs-cuBLAS=%d [CPU-oracle:%s] -> %s\n",
               gpu, BM, BN, BK, TM, TN, Md, Kd, Nd, Md * Nd, mutate ? " [MUTATED]" : "", hC[1 * Nd + 1], gbad, obad, cbad,
               cpu_oracle_ran ? "ran" : "skipped(large-N; kovc-vs-cuBLAS only)", totbad ? "FAIL" : "PASS");
        cuMemFree(dA); cuMemFree(dB); cuMemFree(dC); cuMemFree(dR);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hA); free(hB); free(hC); free(hR); free(ptx);
        return totbad ? 1 : 0;
    }

    /* ===================================================================== *
     * gemm_tf32 mode (T3/G3): cuda_launch <ptx> tf32_matmul 0 gemm_tf32 <M> <K> <N> [mutate]
     *
     * Correctness + perf harness for the kovc TF32 Tensor-Core (mma.sync) GEMM. The kernel
     * is warp-collaborative: ONE warp (32 threads = one block) computes one 16x8 output tile
     * via mma.sync.aligned.m16n8k8.row.col.f32.tf32, looping K/8 times. Grid = (N/8, M/16),
     * block = 32. Requires M%16==0, N%8==0, K%8==0.
     *
     * ORACLE DESIGN (skeptic-corrected, the load-bearing part):
     *  - PRIMARY correctness: judge the kovc TF32 kernel against cublasGemmEx(
     *    CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP) at a TIGHT ~2e-3 rel
     *    tol. Two correct TF32 GEMMs differ only by accumulation order (both truncate the
     *    SAME 10-bit mantissa), so they agree to ~1e-3; a dropped/mis-indexed fragment
     *    stands out sharply. This is the test that has TEETH for a reduced-precision kernel.
     *  - META-ANCHOR (retained): pedantic-f32 cublasSgemm == CPU triple-loop, exact-ish
     *    (1e-3), proves the cuBLAS references themselves are sane (chain-of-trust CPU<-cuBLAS).
     *  - DISTINCT-per-element inputs (NOT i%7/i%5 uniform): every one of the 32 lanes'
     *    fragments is individually observable, so a fragment-permutation bug cannot hide
     *    behind periodic/low-rank input. Inputs are small bounded values to keep the
     *    accumulation noise tiny but each (r,c) element is distinct.
     *  - kovc-TF32 vs CPU-f32 deviation is reported for CONTEXT only (NOT a gate -- f32 ref
     *    vs a tf32 kernel is the self-contradictory tol the skeptic warned against).
     * ===================================================================== */
    if (strcmp(op, "gemm_tf32") == 0) {
        int Md = (argc > 5) ? atoi(argv[5]) : 16;
        int Kd = (argc > 6) ? atoi(argv[6]) : 8;
        int Nd = (argc > 7) ? atoi(argv[7]) : 32;
        int mutate = (argc > 8 && strcmp(argv[8], "mutate") == 0);
        /* TF32_NB / TF32_WP MUST equal the `nb` / `wp` constants in emit_ptx_tf32_matmul_mma
         * (kovc.hx): a block = (32, WP, 1) of WP warps, each warp a 16 x (8*NB) strip, so the
         * block covers 16 x (8*NB*WP) and grid=(N/(8*NB*WP), M/16). */
        const int TF32_NB = 4;
        const int TF32_WP = 4;
        const int NTILE = 8 * TF32_NB * TF32_WP;   /* = 128: the N-span one block covers */
        if (Md % 16 != 0 || Nd % NTILE != 0 || Kd % 8 != 0) {
            fprintf(stderr, "gemm_tf32: dims must satisfy M%%16==0 (%d), N%%%d==0 (%d), K%%8==0 (%d)\n", Md, NTILE, Nd, Kd);
            return 2;
        }
        size_t aN = (size_t)Md * Kd, bN = (size_t)Kd * Nd, cN = (size_t)Md * Nd;
        float* hA = (float*)malloc(aN * sizeof(float));
        float* hB = (float*)malloc(bN * sizeof(float));
        float* hC = (float*)malloc(cN * sizeof(float));   /* kovc TF32 result */
        float* hR = (float*)malloc(cN * sizeof(float));   /* cuBLAS-TF32 result (primary ref) */
        float* hF = (float*)malloc(cN * sizeof(float));   /* cuBLAS-f32-pedantic (meta-anchor) */
        if (!hA || !hB || !hC || !hR || !hF) return 2;
        /* DISTINCT per-element inputs: a quasi-random but deterministic bounded spread so
         * every element differs from its neighbours (exposes any lane/fragment swap), yet
         * stays small (|elem|<~7.8, |prod|<~61, K-partial sums < ~1e3) so TF32 truncation
         * noise is low. NOTE the (int) cast BEFORE the -125/-120: the modulo is computed in
         * size_t (unsigned), so subtracting in size_t would UNDERFLOW to ~1.8e19 for the
         * lower half of values (-> garbage ~1.15e18 floats); cast to int first. */
        for (size_t i = 0; i < aN; i++) hA[i] = (float)((int)((i * 131 + 7) % 251) - 125) * 0.0625f;
        for (size_t i = 0; i < bN; i++) hB[i] = (float)((int)((i * 97 + 13) % 241) - 120) * 0.0625f;
        for (size_t i = 0; i < cN; i++) hC[i] = -123456.0f;   /* sentinel: unwritten cell shows */
        CUdeviceptr dA, dB, dC, dR;
        CK(cuMemAlloc(&dA, aN * sizeof(float)), "alloc A");
        CK(cuMemAlloc(&dB, bN * sizeof(float)), "alloc B");
        CK(cuMemAlloc(&dC, cN * sizeof(float)), "alloc C");
        CK(cuMemAlloc(&dR, cN * sizeof(float)), "alloc R(cublas)");
        CK(cuMemcpyHtoD(dA, hA, aN * sizeof(float)), "H2D A");
        CK(cuMemcpyHtoD(dB, hB, bN * sizeof(float)), "H2D B");
        CK(cuMemcpyHtoD(dC, hC, cN * sizeof(float)), "H2D C");
        /* launch: WP warps/block, each a 16 x (8*NB) strip. grid=(N/(8*NB*WP), M/16),
         * block=(32, WP, 1). */
        void* gargs[] = { &dA, &dB, &dC, &Md, &Kd, &Nd };
        unsigned gx = (unsigned)(Nd / NTILE), gy = (unsigned)(Md / 16);
        CK(cuLaunchKernel(fn, gx, gy, 1, 32, TF32_WP, 1, 0, 0, gargs, 0), "launch tf32_matmul");
        CK(cuCtxSynchronize(), "sync tf32_matmul");
        CK(cuMemcpyDtoH(hC, dC, cN * sizeof(float)), "D2H C");
        /* comparator negative control: perturb one cell so the tol compare MUST trip. The
         * delta is SCALED to exceed the magnitude-aware tol (skeptic §5.7): a bare +1.0f
         * could be < TF32_REL*|hC[0]| on a large-magnitude cell -> vacuous. */
        if (mutate) { float d = 0.05f * fabsf(hC[0]); hC[0] += (d > 1.0f ? d : 1.0f); }

        cublasHandle_t cbh;
        if (cublasCreate(&cbh) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasCreate failed\n"); return 2; }
        float alpha = 1.0f, beta = 0.0f;
        /* (A) cuBLAS-TF32 reference (PRIMARY): cublasGemmEx COMPUTE_32F_FAST_TF32 + TENSOR_OP.
         * row-major C(MxN)=A*B <=> col-major C^T=B^T*A^T: GemmEx(N,N, N,M,K, B(ld N), A(ld K), R(ld N)). */
        cublasSetMathMode(cbh, CUBLAS_TF32_TENSOR_OP_MATH);
        if (cublasGemmEx(cbh, CUBLAS_OP_N, CUBLAS_OP_N, Nd, Md, Kd, &alpha,
                         (const void*)dB, CUDA_R_32F, Nd, (const void*)dA, CUDA_R_32F, Kd, &beta,
                         (void*)dR, CUDA_R_32F, Nd, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP)
            != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasGemmEx TF32 failed\n"); return 2; }
        CK(cuCtxSynchronize(), "sync cublas tf32");
        CK(cuMemcpyDtoH(hR, dR, cN * sizeof(float)), "D2H R(tf32)");
        /* (B) cuBLAS-f32 pedantic (META-ANCHOR): true f32, no TF32 contamination. */
        cublasSetMathMode(cbh, CUBLAS_PEDANTIC_MATH);
        if (cublasSgemm(cbh, CUBLAS_OP_N, CUBLAS_OP_N, Nd, Md, Kd, &alpha,
                        (const float*)dB, Nd, (const float*)dA, Kd, &beta, (float*)dR, Nd) != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cublasSgemm failed\n"); return 2;
        }
        CK(cuCtxSynchronize(), "sync cublas f32");
        CK(cuMemcpyDtoH(hF, dR, cN * sizeof(float)), "D2H F(f32)");

        /* The two CPU triple-loops below are O(M*N*K). At small corpus sizes (<=128^3) they
         * run in full and give the harness its teeth. At a large PERF size (e.g. 2048^3) a
         * CPU triple-loop is ~17e9 scalar MACs = MINUTES at 0% GPU -- the "hang" the prior
         * agent hit was NOT a kernel illegal-access but this CPU reference. So gate them on a
         * work cap: above it, skip the CPU loops and rely on the PRIMARY GPU-side kovc-vs-
         * cuBLAS-TF32 compare (line below, only O(M*N) cells -- cheap even at 2048^2), which is
         * itself a sound oracle (cuBLAS-TF32 was validated == CPU at the small sizes). */
        double cpu_work = (double)Md * (double)Nd * (double)Kd;
        int big = (cpu_work > 4.0e8);   /* ~735^3; 128^3=2.1e6 runs, 512^3=1.34e8 runs, 2048^3 skips */
        /* META-ANCHOR check: pedantic-f32 cuBLAS == CPU triple-loop (cell-by-cell, ~1e-3).
         * Proves the references are correct; runs in full at the small G3 sizes (skipped when big). */
        int obad = 0;
        if (!big) for (int r = 0; r < Md; r++) for (int cc = 0; cc < Nd; cc++) {
            float ref = 0.0f;
            for (int t = 0; t < Kd; t++) ref += hA[r * Kd + t] * hB[t * Nd + cc];
            float got = hF[r * Nd + cc];
            float e = got - ref; if (e < 0) e = -e;
            float aref = ref < 0 ? -ref : ref;
            if (isnan(got) || (e > 1.0e-3f && e > 1.0e-3f * aref)) { if (obad < 6) fprintf(stderr, "f32-cuBLAS-vs-CPU mismatch [%d,%d]=%g ref %g\n", r, cc, got, ref); obad++; }
        }
        /* PRIMARY (ALWAYS, even when big): kovc TF32 kernel vs cuBLAS-TF32, tight ~2e-3 RELATIVE
         * (magnitude-aware). O(M*N) only -- this is the real correctness gate at every size. */
        const float TF32_REL = 2.0e-3f, TF32_ABS = 2.0e-3f;
        int tbad = 0; float maxrel = 0.0f;
        for (int r = 0; r < Md; r++) for (int cc = 0; cc < Nd; cc++) {
            float got = hC[r * Nd + cc], refb = hR[r * Nd + cc];
            float e = got - refb; if (e < 0) e = -e;
            float ar = refb < 0 ? -refb : refb;
            float rel = e / (ar > 1.0f ? ar : 1.0f); if (rel > maxrel) maxrel = rel;
            if (isnan(got) || (e > TF32_ABS && e > TF32_REL * ar)) { if (tbad < 6) fprintf(stderr, "tf32 kovc-vs-cuBLAS-TF32 mismatch C[%d,%d]=%g tf32ref %g (rel %g)\n", r, cc, got, refb, rel); tbad++; }
        }
        /* CONTEXT only: kovc TF32 vs CPU-f32 max relative deviation (NOT a gate; -1 = skipped when big). */
        float ctx_maxrel = big ? -1.0f : 0.0f;
        if (!big) for (int r = 0; r < Md; r++) for (int cc = 0; cc < Nd; cc++) {
            float ref = 0.0f; for (int t = 0; t < Kd; t++) ref += hA[r * Kd + t] * hB[t * Nd + cc];
            float got = hC[r * Nd + cc]; float e = got - ref; if (e < 0) e = -e;
            float ar = ref < 0 ? -ref : ref; float rel = e / (ar > 1.0f ? ar : 1.0f);
            if (rel > ctx_maxrel) ctx_maxrel = rel;
        }

        /* --- TIMING (kernel-only) via cuEvent: warmup + median-of-50, min/med/max. The
         * absolute perf GATE is DEFERRED this phase (correctness-first); set via G1_MIN_TFLOPS
         * by the corpus. We still report the number for the orchestrator. --- */
        double flop = 2.0 * (double)Md * (double)Nd * (double)Kd;
        if (!mutate) {
            int WARM = 5, ITERS = 50;
            float* ts = (float*)malloc((size_t)ITERS * sizeof(float));
            CUevent e0, e1; CK(cuEventCreate(&e0, 0), "evt0"); CK(cuEventCreate(&e1, 0), "evt1");
            for (int w = 0; w < WARM; w++) CK(cuLaunchKernel(fn, gx, gy, 1, 32, TF32_WP, 1, 0, 0, gargs, 0), "warm kov");
            CK(cuCtxSynchronize(), "sync warm kov");
            for (int it = 0; it < ITERS; it++) {
                CK(cuEventRecord(e0, 0), "rec0");
                CK(cuLaunchKernel(fn, gx, gy, 1, 32, TF32_WP, 1, 0, 0, gargs, 0), "time kov");
                CK(cuEventRecord(e1, 0), "rec1"); CK(cuEventSynchronize(e1), "evsync");
                CK(cuEventElapsedTime(&ts[it], e0, e1), "elapsed");
            }
            for (int i = 0; i < ITERS; i++) for (int j = i + 1; j < ITERS; j++) if (ts[j] < ts[i]) { float t = ts[i]; ts[i] = ts[j]; ts[j] = t; }
            float kmin = ts[0], kmax = ts[ITERS - 1], kmed = ts[ITERS / 2];
            double kov_tf = flop / (kmed * 1.0e-3) / 1.0e12;
            printf("GPU [%s] TIMING tf32_kovc %dx%dx%d: min=%.4f med=%.4f max=%.4f ms\n", gpu, Md, Kd, Nd, kmin, kmed, kmax);
            printf("MEDIAN-TFLOPS-TF32 kovc=%.3f\n", kov_tf);
            cuEventDestroy(e0); cuEventDestroy(e1); free(ts);
        }
        cublasDestroy(cbh);

        int totbad = tbad + obad;
        printf("GPU [%s] tf32_matmul (mma m16n8k8) %dx%dx%d over %d cells%s: C[1,1]=%g tf32ref=%g, kovc-vs-cuBLAS-TF32=%d(maxrel=%.2e tol=%.0e) anchor(f32cuBLAS-vs-CPU)=%d [ctx:kovc-vs-CPUf32 maxrel=%.2e] -> %s\n",
               gpu, Md, Kd, Nd, Md * Nd, mutate ? " [MUTATED]" : "", hC[1 * Nd + 1], hR[1 * Nd + 1], tbad, maxrel, (double)TF32_REL, obad, ctx_maxrel, totbad ? "FAIL" : "PASS");
        cuMemFree(dA); cuMemFree(dB); cuMemFree(dC); cuMemFree(dR);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hA); free(hB); free(hC); free(hR); free(hF); free(ptx);
        return totbad ? 1 : 0;
    }

    /* matmul_abt mode: cuda_launch <ptx> gpu_matmul_abt <Nignored> matmul_abt <M> <K> <N>.
     * C[M,N] = A[M,K] @ B[N,K]^T, i.e. C[i,j]=sum_t A[i,t]*B[j,t] -- the UNSCALED A@B^T used
     * for d_attn=dOut@V^T. gridDim=M, blockDim=N. Integer inputs so exact; verify every cell. */
    if (strcmp(op, "matmul_abt") == 0) {
        int M = (argc > 5) ? atoi(argv[5]) : 16;
        int K = (argc > 6) ? atoi(argv[6]) : 16;
        int Nn = (argc > 7) ? atoi(argv[7]) : 16;
        size_t aN = (size_t)M * K, bN = (size_t)Nn * K, cN = (size_t)M * Nn;
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
        void* uargs[] = { &dA, &dB, &dC, &M, &K, &Nn };
        CK(cuLaunchKernel(fn, M, 1, 1, Nn, 1, 1, 0, 0, uargs, 0), "launch matmul_abt");
        CK(cuCtxSynchronize(), "sync matmul_abt");
        CK(cuMemcpyDtoH(hC, dC, cN * sizeof(float)), "D2H C");
        int tbad = 0;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < Nn; j++) {
                float ref = hA[i * K] * hB[j * K];
                for (int t = 1; t < K; t++) ref += hA[i * K + t] * hB[j * K + t];
                float got = hC[i * Nn + j];
                if (got != ref) { if (tbad < 4) fprintf(stderr, "matmul_abt mismatch C[%d,%d]=%g ref %g\n", i, j, got, ref); tbad++; }
            }
        }
        printf("GPU [%s] matmul_abt (A@B^T) %dx%dx%d over %d cells: C[1,1]=%g, %d bad -> %s\n",
               gpu, M, K, Nn, M * Nn, hC[1 * Nn + 1], tbad, tbad ? "FAIL" : "PASS");
        cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hA); free(hB); free(hC); free(ptx);
        return tbad ? 1 : 0;
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

    /* gemm_abt mode (T2/M4): cuda_launch <combined.ptx> tiled_matmul_abt 0 gemm_abt <M> <K> <N> [mutate].
     * The SMEM-tiled A@B^T (emit_ptx_tiled_matmul_t mode 0) vs (a) a CPU A@B^T oracle for
     * CORRECTNESS and (b) the naive non-tiled gpu_matmul_abt for the FASTER-THAN-NAIVE bar.
     * combined.ptx must carry BOTH kernels (concat tiled_matmul_abt_kernel.hx +
     * gpu_matmul_abt_kernel.hx, emit once). A[M,K] (a[i*K+t]), B[N,K] (b[j*K+t]) transposed,
     * C[M,N] (c[i*N+j]); C[i,j]=sum_t A[i,t]*B[j,t]. Tiled: grid=(N/64,M/64) block=(16,16);
     * naive: grid.x=M block.x=N (so the speedup compare needs N<=1024). Integer inputs
     * (a[i]=i%7,b[i]=i%5) keep every cell sum < 2^24 for K up to ~7e5 -> EXACT == compare.
     * Requires M%64==0, N%64==0, K%8==0 (tiled, asserted). */
    if (strcmp(op, "gemm_abt") == 0) {
        int M = (argc > 5) ? atoi(argv[5]) : 64;
        int K = (argc > 6) ? atoi(argv[6]) : 64;
        int Nn = (argc > 7) ? atoi(argv[7]) : 64;
        int mutate = (argc > 8 && strcmp(argv[8], "mutate") == 0);
        /* "corr" (8th arg): correctness-only -- still measure+report the tiled-vs-naive speedup
         * but do NOT gate on it (used for tiny sizes like 64^3 where there is too little work to
         * amortize the tiling overhead so the naive kernel can match/beat it; the faster-than-
         * naive GATE is asserted at the large/perf sizes). */
        int corr_only = (argc > 8 && strcmp(argv[8], "corr") == 0);
        if (M % 64 != 0 || Nn % 64 != 0 || K % 8 != 0) {
            fprintf(stderr, "gemm_abt: dims must satisfy M%%64==0 (%d), N%%64==0 (%d), K%%8==0 (%d)\n", M, Nn, K);
            return 2;
        }
        if (Nn > 1024) { fprintf(stderr, "gemm_abt: N<=1024 (naive baseline blockDim.x=N); got %d\n", Nn); return 2; }
        CUfunction f_naive; CK(cuModuleGetFunction(&f_naive, mod, "gpu_matmul_abt"), "get gpu_matmul_abt");
        size_t aN = (size_t)M * K, bN = (size_t)Nn * K, cN = (size_t)M * Nn;
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
        { float neg = -1.0f; for (size_t i = 0; i < cN; i++) hC[i] = neg; }
        CK(cuMemcpyHtoD(dC, hC, cN * sizeof(float)), "H2D C(sentinel)");
        void* gargs[] = { &dA, &dB, &dC, &M, &K, &Nn };
        unsigned gx = (unsigned)(Nn / 64), gy = (unsigned)(M / 64);
        CK(cuLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0, 0, gargs, 0), "launch tiled_matmul_abt");
        CK(cuCtxSynchronize(), "sync tiled_matmul_abt");
        CK(cuMemcpyDtoH(hC, dC, cN * sizeof(float)), "D2H C");
        if (mutate) hC[0] += 1.0f;
        /* CORRECTNESS: cell-by-cell vs CPU A@B^T (integer-exact). Work-capped: above ~735^3
         * the O(M*N*K) CPU loop is skipped (sizes here stay <=512 so it runs in full). */
        double cpu_work = (double)M * (double)Nn * (double)K;
        int big = (cpu_work > 4.0e8);
        int cbad = 0;
        if (!big) for (int i = 0; i < M; i++) for (int j = 0; j < Nn; j++) {
            float ref = hA[i * K] * hB[j * K];
            for (int t = 1; t < K; t++) ref += hA[i * K + t] * hB[j * K + t];
            float got = hC[i * Nn + j];
            if (got != ref) { if (cbad < 6) fprintf(stderr, "gemm_abt mismatch C[%d,%d]=%g ref %g\n", i, j, got, ref); cbad++; }
        }
        /* FASTER-THAN-NAIVE: time the tiled kernel AND the naive gpu_matmul_abt, kernel-only,
         * cuEvent median-of-30. The naive run also writes dC -> validate it agrees with the
         * tiled result (GPU-side, O(M*N)) so the baseline is a REAL same-answer comparison. */
        double speedup = 0.0; int nbad = 0;
        if (!mutate) {
            int WARM = 3, ITERS = 30;
            float* ts = (float*)malloc((size_t)ITERS * sizeof(float));
            CUevent e0, e1; CK(cuEventCreate(&e0, 0), "evt0"); CK(cuEventCreate(&e1, 0), "evt1");
            float* hCn = (float*)malloc(cN * sizeof(float));
            void* nargs[] = { &dA, &dB, &dC, &M, &K, &Nn };
            /* tiled timing */
            for (int w = 0; w < WARM; w++) CK(cuLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0, 0, gargs, 0), "warm tiled");
            CK(cuCtxSynchronize(), "sync warm tiled");
            for (int it = 0; it < ITERS; it++) { CK(cuEventRecord(e0,0),"r0"); CK(cuLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0, 0, gargs, 0),"t"); CK(cuEventRecord(e1,0),"r1"); CK(cuEventSynchronize(e1),"s"); CK(cuEventElapsedTime(&ts[it],e0,e1),"e"); }
            for (int i = 0; i < ITERS; i++) for (int j = i+1; j < ITERS; j++) if (ts[j] < ts[i]) { float t=ts[i]; ts[i]=ts[j]; ts[j]=t; }
            float t_tiled = ts[ITERS/2];
            /* naive timing (grid.x=M, block.x=N) */
            for (int w = 0; w < WARM; w++) CK(cuLaunchKernel(f_naive, (unsigned)M, 1, 1, (unsigned)Nn, 1, 1, 0, 0, nargs, 0), "warm naive");
            CK(cuCtxSynchronize(), "sync warm naive");
            for (int it = 0; it < ITERS; it++) { CK(cuEventRecord(e0,0),"r0"); CK(cuLaunchKernel(f_naive,(unsigned)M,1,1,(unsigned)Nn,1,1,0,0,nargs,0),"t"); CK(cuEventRecord(e1,0),"r1"); CK(cuEventSynchronize(e1),"s"); CK(cuEventElapsedTime(&ts[it],e0,e1),"e"); }
            for (int i = 0; i < ITERS; i++) for (int j = i+1; j < ITERS; j++) if (ts[j] < ts[i]) { float t=ts[i]; ts[i]=ts[j]; ts[j]=t; }
            float t_naive = ts[ITERS/2];
            CK(cuMemcpyDtoH(hCn, dC, cN * sizeof(float)), "D2H Cn(naive)");
            for (int i = 0; i < M; i++) for (int j = 0; j < Nn; j++) if (hCn[i*Nn+j] != hC[i*Nn+j]) { if (nbad < 6) fprintf(stderr, "gemm_abt tiled-vs-naive mismatch C[%d,%d] tiled=%g naive=%g\n", i, j, hC[i*Nn+j], hCn[i*Nn+j]); nbad++; }
            speedup = (t_naive > 0.0f) ? (double)t_naive / (double)t_tiled : 0.0;
            printf("GPU [%s] SPEEDUP-ABT %dx%dx%d: tiled med=%.4f ms  naive med=%.4f ms  speedup=%.2fx\n", gpu, M, K, Nn, t_tiled, t_naive, speedup);
            cuEventDestroy(e0); cuEventDestroy(e1); free(ts); free(hCn);
        }
        int faster = (mutate || corr_only) ? 1 : (speedup > 1.0);
        int totbad = cbad + nbad + (faster ? 0 : 1);
        printf("GPU [%s] tiled_matmul_abt (A@B^T, SMEM 64x64/BK8/4x4) %dx%dx%d over %d cells%s: C[1,1]=%g, vs-CPU=%d vs-naive=%d [CPU:%s] faster-than-naive=%s -> %s\n",
               gpu, M, K, Nn, M * Nn, mutate ? " [MUTATED]" : "", hC[1 * Nn + 1], cbad, nbad,
               big ? "skipped(large)" : "ran", mutate ? "n/a" : (corr_only ? "not-gated(corr)" : (faster ? "YES" : "NO")), totbad ? "FAIL" : "PASS");
        cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hA); free(hB); free(hC); free(ptx);
        return totbad ? 1 : 0;
    }

    /* gemm_atb mode (T2/M4): cuda_launch <combined.ptx> tiled_matmul_atb 0 gemm_atb <M> <K> <N> [mutate].
     * The SMEM-tiled A^T@B (emit_ptx_tiled_matmul_t mode 1) vs (a) a CPU A^T@B oracle for
     * CORRECTNESS and (b) the naive non-tiled gpu_matmul_atb for the FASTER-THAN-NAIVE bar.
     * combined.ptx carries BOTH kernels (concat tiled_matmul_atb_kernel.hx + gpu_matmul_atb_kernel.hx).
     * A[M,K] (a[t*K+i], t=contraction), B[M,N] (b[t*N+j]), C[K,N] (c[i*N+j]); C[i,j]=sum_t A[t,i]*B[t,j],
     * contraction length = M. Tiled: grid=(N/64,K/64) block=(16,16); naive: grid.x=K block.x=N (N<=1024).
     * Integer inputs keep cell sums < 2^24 for M up to ~7e5 -> EXACT == compare.
     * Requires K%64==0, N%64==0, M%8==0 (tiled, asserted). */
    if (strcmp(op, "gemm_atb") == 0) {
        int M = (argc > 5) ? atoi(argv[5]) : 64;
        int K = (argc > 6) ? atoi(argv[6]) : 64;
        int Nn = (argc > 7) ? atoi(argv[7]) : 64;
        int mutate = (argc > 8 && strcmp(argv[8], "mutate") == 0);
        int corr_only = (argc > 8 && strcmp(argv[8], "corr") == 0);   /* correctness-only; speedup reported not gated */
        if (K % 64 != 0 || Nn % 64 != 0 || M % 8 != 0) {
            fprintf(stderr, "gemm_atb: dims must satisfy K%%64==0 (%d), N%%64==0 (%d), M%%8==0 (%d)\n", K, Nn, M);
            return 2;
        }
        if (Nn > 1024) { fprintf(stderr, "gemm_atb: N<=1024 (naive baseline blockDim.x=N); got %d\n", Nn); return 2; }
        CUfunction f_naive; CK(cuModuleGetFunction(&f_naive, mod, "gpu_matmul_atb"), "get gpu_matmul_atb");
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
        { float neg = -1.0f; for (size_t i = 0; i < cN; i++) hC[i] = neg; }
        CK(cuMemcpyHtoD(dC, hC, cN * sizeof(float)), "H2D C(sentinel)");
        void* gargs[] = { &dA, &dB, &dC, &M, &K, &Nn };
        unsigned gx = (unsigned)(Nn / 64), gy = (unsigned)(K / 64);
        CK(cuLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0, 0, gargs, 0), "launch tiled_matmul_atb");
        CK(cuCtxSynchronize(), "sync tiled_matmul_atb");
        CK(cuMemcpyDtoH(hC, dC, cN * sizeof(float)), "D2H C");
        if (mutate) hC[0] += 1.0f;
        double cpu_work = (double)K * (double)Nn * (double)M;
        int big = (cpu_work > 4.0e8);
        int cbad = 0;
        if (!big) for (int i = 0; i < K; i++) for (int j = 0; j < Nn; j++) {
            float ref = hA[0 * K + i] * hB[0 * Nn + j];
            for (int t = 1; t < M; t++) ref += hA[t * K + i] * hB[t * Nn + j];
            float got = hC[i * Nn + j];
            if (got != ref) { if (cbad < 6) fprintf(stderr, "gemm_atb mismatch C[%d,%d]=%g ref %g\n", i, j, got, ref); cbad++; }
        }
        double speedup = 0.0; int nbad = 0;
        if (!mutate) {
            int WARM = 3, ITERS = 30;
            float* ts = (float*)malloc((size_t)ITERS * sizeof(float));
            CUevent e0, e1; CK(cuEventCreate(&e0, 0), "evt0"); CK(cuEventCreate(&e1, 0), "evt1");
            float* hCn = (float*)malloc(cN * sizeof(float));
            void* nargs[] = { &dA, &dB, &dC, &M, &K, &Nn };
            for (int w = 0; w < WARM; w++) CK(cuLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0, 0, gargs, 0), "warm tiled");
            CK(cuCtxSynchronize(), "sync warm tiled");
            for (int it = 0; it < ITERS; it++) { CK(cuEventRecord(e0,0),"r0"); CK(cuLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0, 0, gargs, 0),"t"); CK(cuEventRecord(e1,0),"r1"); CK(cuEventSynchronize(e1),"s"); CK(cuEventElapsedTime(&ts[it],e0,e1),"e"); }
            for (int i = 0; i < ITERS; i++) for (int j = i+1; j < ITERS; j++) if (ts[j] < ts[i]) { float t=ts[i]; ts[i]=ts[j]; ts[j]=t; }
            float t_tiled = ts[ITERS/2];
            for (int w = 0; w < WARM; w++) CK(cuLaunchKernel(f_naive, (unsigned)K, 1, 1, (unsigned)Nn, 1, 1, 0, 0, nargs, 0), "warm naive");
            CK(cuCtxSynchronize(), "sync warm naive");
            for (int it = 0; it < ITERS; it++) { CK(cuEventRecord(e0,0),"r0"); CK(cuLaunchKernel(f_naive,(unsigned)K,1,1,(unsigned)Nn,1,1,0,0,nargs,0),"t"); CK(cuEventRecord(e1,0),"r1"); CK(cuEventSynchronize(e1),"s"); CK(cuEventElapsedTime(&ts[it],e0,e1),"e"); }
            for (int i = 0; i < ITERS; i++) for (int j = i+1; j < ITERS; j++) if (ts[j] < ts[i]) { float t=ts[i]; ts[i]=ts[j]; ts[j]=t; }
            float t_naive = ts[ITERS/2];
            CK(cuMemcpyDtoH(hCn, dC, cN * sizeof(float)), "D2H Cn(naive)");
            for (int i = 0; i < K; i++) for (int j = 0; j < Nn; j++) if (hCn[i*Nn+j] != hC[i*Nn+j]) { if (nbad < 6) fprintf(stderr, "gemm_atb tiled-vs-naive mismatch C[%d,%d] tiled=%g naive=%g\n", i, j, hC[i*Nn+j], hCn[i*Nn+j]); nbad++; }
            speedup = (t_naive > 0.0f) ? (double)t_naive / (double)t_tiled : 0.0;
            printf("GPU [%s] SPEEDUP-ATB %dx%dx%d: tiled med=%.4f ms  naive med=%.4f ms  speedup=%.2fx\n", gpu, M, K, Nn, t_tiled, t_naive, speedup);
            cuEventDestroy(e0); cuEventDestroy(e1); free(ts); free(hCn);
        }
        int faster = (mutate || corr_only) ? 1 : (speedup > 1.0);
        int totbad = cbad + nbad + (faster ? 0 : 1);
        printf("GPU [%s] tiled_matmul_atb (A^T@B, SMEM 64x64/BK8/4x4) %dx%dx%d over %d cells%s: C[1,1]=%g, vs-CPU=%d vs-naive=%d [CPU:%s] faster-than-naive=%s -> %s\n",
               gpu, M, K, Nn, K * Nn, mutate ? " [MUTATED]" : "", hC[1 * Nn + 1], cbad, nbad,
               big ? "skipped(large)" : "ran", mutate ? "n/a" : (corr_only ? "not-gated(corr)" : (faster ? "YES" : "NO")), totbad ? "FAIL" : "PASS");
        cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hA); free(hB); free(hC); free(ptx);
        return totbad ? 1 : 0;
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

    /* softmax_perf mode (T2/M4): cuda_launch <combined.ptx> softmax_blockred <Nignored>
     *   softmax_perf <rows> <cols> [mutate].
     * Correctness + faster-than-naive harness for the WARP/BLOCK-REDUCTION row softmax.
     * combined.ptx carries BOTH the kovc block-reduction kernel (argv[2], here
     * softmax_blockred) AND the naive one-thread-per-row gpu_softmax (the baseline).
     *  (a) CORRECTNESS: every cell of the block-reduction y vs a CPU stable-softmax
     *      reference (expf, max-subtracted); report maxrel + a per-row sum~1 check. The
     *      tol accounts HONESTLY for the kernel's ex2.approx exp (sm_86 ~2^-22) -- tol
     *      SM_TOL (default 1e-3, the same the naive softmax mode uses).
     *  (b) TIMING (kernel-only, cuEvent): block-reduction grid=(rows,1,1) block=(256,1,1)
     *      vs naive grid=(rows,1,1) block=(1,1,1); report medians + speedup-vs-naive.
     * The optional "mutate" perturbs one y cell pre-compare (comparator negative control
     * -> must FAIL). The bar.sync-strip neg-control is driven by the corpus script. */
    if (strcmp(op, "softmax_perf") == 0) {
        int rows = (argc > 5) ? atoi(argv[5]) : 64;
        int cols = (argc > 6) ? atoi(argv[6]) : 256;
        int mutate = (argc > 7 && strcmp(argv[7], "mutate") == 0);
        float SM_TOL = 1.0e-3f; { const char* e; if ((e = getenv("SM_TOL"))) SM_TOL = (float)atof(e); }
        CUfunction f_naive;
        CK(cuModuleGetFunction(&f_naive, mod, "gpu_softmax"), "get gpu_softmax (naive baseline)");
        size_t ne = (size_t)rows * cols;
        float* hx = (float*)malloc(ne * sizeof(float));
        float* hy = (float*)malloc(ne * sizeof(float));
        if (!hx || !hy) return 2;
        for (size_t i = 0; i < ne; i++) { hx[i] = (float)((int)((i * 7 + 3) % 13) - 6); hy[i] = -1.0f; }
        CUdeviceptr dx, dy;
        CK(cuMemAlloc(&dx, ne * sizeof(float)), "alloc x");
        CK(cuMemAlloc(&dy, ne * sizeof(float)), "alloc y");
        CK(cuMemcpyHtoD(dx, hx, ne * sizeof(float)), "H2D x");
        CK(cuMemcpyHtoD(dy, hy, ne * sizeof(float)), "H2D y(sentinel)");
        void* sargs[] = { &dx, &dy, &rows, &cols };
        /* block-reduction launch: ONE block of 256 threads per row. */
        CK(cuLaunchKernel(fn, rows, 1, 1, 256, 1, 1, 0, 0, sargs, 0), "launch softmax_blockred");
        CK(cuCtxSynchronize(), "sync softmax_blockred");
        CK(cuMemcpyDtoH(hy, dy, ne * sizeof(float)), "D2H y");
        if (mutate) hy[0] += 1.0f;   /* comparator negative control */
        /* CPU stable-softmax reference: maxrel + per-row sum~1. */
        int sbad = 0; float maxrel = 0.0f, maxsumerr = 0.0f;
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
                float ar = ref < 0 ? -ref : ref; float rel = d / (ar > 1e-6f ? ar : 1e-6f);
                if (rel > maxrel) maxrel = rel;
                if (isnan(got) || (d > SM_TOL && rel > SM_TOL)) { if (sbad < 6) fprintf(stderr, "softmax_perf mismatch y[%d,%d]=%g ref %g (rel %.2e)\n", r, c, got, ref, rel); sbad++; }
            }
            float ds = rowsum - 1.0f; if (ds < 0) ds = -ds; if (ds > maxsumerr) maxsumerr = ds;
            if (isnan(rowsum) || ds > 1.0e-3f) { if (sbad < 6) fprintf(stderr, "softmax_perf row %d sum %g (want 1)\n", r, rowsum); sbad++; }
        }
        /* TIMING (kernel-only): block-reduction vs naive. */
        double br_med = 0.0, nv_med = 0.0;
        if (!mutate) {
            int WARM = 5, ITERS = 50;
            float* ts = (float*)malloc((size_t)ITERS * sizeof(float));
            CUevent e0, e1; CK(cuEventCreate(&e0, 0), "evt0"); CK(cuEventCreate(&e1, 0), "evt1");
            for (int w = 0; w < WARM; w++) CK(cuLaunchKernel(fn, rows, 1, 1, 256, 1, 1, 0, 0, sargs, 0), "warm br");
            CK(cuCtxSynchronize(), "sync warm br");
            for (int it = 0; it < ITERS; it++) { CK(cuEventRecord(e0, 0), "r0"); CK(cuLaunchKernel(fn, rows, 1, 1, 256, 1, 1, 0, 0, sargs, 0), "time br"); CK(cuEventRecord(e1, 0), "r1"); CK(cuEventSynchronize(e1), "es"); CK(cuEventElapsedTime(&ts[it], e0, e1), "el"); }
            for (int i = 0; i < ITERS; i++) for (int j = i + 1; j < ITERS; j++) if (ts[j] < ts[i]) { float t = ts[i]; ts[i] = ts[j]; ts[j] = t; }
            br_med = (double)ts[ITERS / 2];
            for (int w = 0; w < WARM; w++) CK(cuLaunchKernel(f_naive, rows, 1, 1, 1, 1, 1, 0, 0, sargs, 0), "warm nv");
            CK(cuCtxSynchronize(), "sync warm nv");
            for (int it = 0; it < ITERS; it++) { CK(cuEventRecord(e0, 0), "n0"); CK(cuLaunchKernel(f_naive, rows, 1, 1, 1, 1, 1, 0, 0, sargs, 0), "time nv"); CK(cuEventRecord(e1, 0), "n1"); CK(cuEventSynchronize(e1), "nes"); CK(cuEventElapsedTime(&ts[it], e0, e1), "nel"); }
            for (int i = 0; i < ITERS; i++) for (int j = i + 1; j < ITERS; j++) if (ts[j] < ts[i]) { float t = ts[i]; ts[i] = ts[j]; ts[j] = t; }
            nv_med = (double)ts[ITERS / 2];
            cuEventDestroy(e0); cuEventDestroy(e1); free(ts);
            printf("GPU [%s] TIMING softmax %dx%d: blockred med=%.4f ms  naive med=%.4f ms  SPEEDUP=%.2fx\n", gpu, rows, cols, br_med, nv_med, nv_med / br_med);
        }
        printf("GPU [%s] softmax_perf (block-reduction 256t/row) %dx%d%s: maxrel=%.2e (tol=%.0e) max|rowsum-1|=%.2e, %d bad -> %s\n",
               gpu, rows, cols, mutate ? " [MUTATED]" : "", maxrel, SM_TOL, maxsumerr, sbad, sbad ? "FAIL" : "PASS");
        cuMemFree(dx); cuMemFree(dy);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hx); free(hy); free(ptx);
        return sbad ? 1 : 0;
    }

    /* gelu_backward mode: cuda_launch <ptx> gpu_gelu_backward <N> gelu_backward.
     * dx = dy * gelu'(x). INDEPENDENT ref = central finite-difference of the GELU FORWARD
     * (uses tanhf, does NOT share the kernel's analytic derivative): gp_fd=(gelu(x+h)-gelu(x-h))/2h,
     * h=1e-3, dx_ref=dy*gp_fd, tol 2e-2; plus a tighter analytic gelu' check at tol 1e-3.
     * Inputs in [-3,3] so e^2z never saturates. */
    if (strcmp(op, "gelu_backward") == 0) {
        size_t ne = (size_t)N;
        float* hx = (float*)malloc(ne * sizeof(float));
        float* hdy = (float*)malloc(ne * sizeof(float));
        float* hdx = (float*)malloc(ne * sizeof(float));
        if (!hx || !hdy || !hdx) return 2;
        for (size_t i = 0; i < ne; i++) { hx[i] = (float)((int)(i % 61) - 30) * 0.1f; hdy[i] = (float)((int)(i % 7) - 3); }
        CUdeviceptr dxb, ddy, ddx;
        CK(cuMemAlloc(&dxb, ne * sizeof(float)), "alloc x");
        CK(cuMemAlloc(&ddy, ne * sizeof(float)), "alloc dy");
        CK(cuMemAlloc(&ddx, ne * sizeof(float)), "alloc dx");
        CK(cuMemcpyHtoD(dxb, hx, ne * sizeof(float)), "H2D x");
        CK(cuMemcpyHtoD(ddy, hdy, ne * sizeof(float)), "H2D dy");
        void* gargs[] = { &dxb, &ddy, &ddx, &N };
        CK(cuLaunchKernel(fn, N, 1, 1, 1, 1, 1, 0, 0, gargs, 0), "launch gelu_bwd");
        CK(cuCtxSynchronize(), "sync gelu_bwd");
        CK(cuMemcpyDtoH(hdx, ddx, ne * sizeof(float)), "D2H dx");
        int gbad = 0; float maxfd = 0.0f, g0 = 0.0f, r0 = 0.0f;
        for (size_t i = 0; i < ne; i++) {
            float xx = hx[i];
            float inn = 0.7978846f * (xx + 0.044715f * xx * xx * xx);
            float e2 = expf(2.0f * inn); float th = (e2 - 1.0f) / (e2 + 1.0f);
            float idv = 0.7978846f * (1.0f + 0.134145f * xx * xx);
            float gp_a = 0.5f * (1.0f + th) + 0.5f * xx * (1.0f - th * th) * idv;
            float ref_a = hdy[i] * gp_a;
            float h = 1.0e-3f, xp = xx + h, xm = xx - h;
            float gpx = 0.5f * xp * (1.0f + tanhf(0.7978846f * (xp + 0.044715f * xp * xp * xp)));
            float gmx = 0.5f * xm * (1.0f + tanhf(0.7978846f * (xm + 0.044715f * xm * xm * xm)));
            float ref_fd = hdy[i] * (gpx - gmx) / (2.0f * h);
            float got = hdx[i];
            if (i == 0) { g0 = got; r0 = ref_fd; }
            float ea = got - ref_a; if (ea < 0) ea = -ea;
            float ef = got - ref_fd; if (ef < 0) ef = -ef; if (ef > maxfd) maxfd = ef;
            if (isnan(got) || ea > 1.0e-3f || ef > 2.0e-2f) { if (gbad < 4) fprintf(stderr, "gelu_bwd mismatch dx[%zu]=%g ref_a %g ref_fd %g (x=%g)\n", i, got, ref_a, ref_fd, xx); gbad++; }
        }
        printf("GPU [%s] gelu_backward N=%d: dx[0]=%g ref_fd %g, max|dx-fd|=%g, %d bad -> %s\n",
               gpu, N, g0, r0, maxfd, gbad, gbad ? "FAIL" : "PASS");
        cuMemFree(dxb); cuMemFree(ddy); cuMemFree(ddx);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hx); free(hdy); free(hdx); free(ptx);
        return gbad ? 1 : 0;
    }

    /* softmax_backward mode: cuda_launch <ptx> gpu_softmax_backward <N> softmax_backward <rows> <cols>.
     * dA[i,j]=P[i,j]*(dP[i,j]-sum_k dP[i,k]*P[i,k]). P is a valid CPU softmax, dP a varied upstream
     * grad. Verify per-cell vs CPU (1e-4) AND the INDEPENDENT conservation that each dA row sums
     * to ~0 (sum_j dA = dot - 1*dot = 0 since sum_j P = 1). */
    if (strcmp(op, "softmax_backward") == 0) {
        int rows = (argc > 5) ? atoi(argv[5]) : 8;
        int cols = (argc > 6) ? atoi(argv[6]) : 16;
        size_t ne = (size_t)rows * cols;
        float* hP = (float*)malloc(ne * sizeof(float));
        float* hdP = (float*)malloc(ne * sizeof(float));
        float* hdA = (float*)malloc(ne * sizeof(float));
        if (!hP || !hdP || !hdA) return 2;
        for (int r = 0; r < rows; r++) {
            float mx = -1.0e30f;
            for (int c = 0; c < cols; c++) { float z = (float)((int)(((r * cols + c) * 7 + 3) % 13) - 6); hP[r * cols + c] = z; if (z > mx) mx = z; }
            float sm = 0.0f; for (int c = 0; c < cols; c++) sm += expf(hP[r * cols + c] - mx);
            for (int c = 0; c < cols; c++) hP[r * cols + c] = expf(hP[r * cols + c] - mx) / sm;
            for (int c = 0; c < cols; c++) hdP[r * cols + c] = (float)((int)(((r * cols + c) * 5 + 1) % 7) - 3);
        }
        CUdeviceptr dP, ddP, ddA;
        CK(cuMemAlloc(&dP, ne * sizeof(float)), "alloc P");
        CK(cuMemAlloc(&ddP, ne * sizeof(float)), "alloc dP");
        CK(cuMemAlloc(&ddA, ne * sizeof(float)), "alloc dA");
        CK(cuMemcpyHtoD(dP, hP, ne * sizeof(float)), "H2D P");
        CK(cuMemcpyHtoD(ddP, hdP, ne * sizeof(float)), "H2D dP");
        void* sargs[] = { &dP, &ddP, &ddA, &rows, &cols };
        CK(cuLaunchKernel(fn, rows, 1, 1, 1, 1, 1, 0, 0, sargs, 0), "launch softmax_bwd");
        CK(cuCtxSynchronize(), "sync softmax_bwd");
        CK(cuMemcpyDtoH(hdA, ddA, ne * sizeof(float)), "D2H dA");
        int sbad = 0; float maxrs = 0.0f, a0 = 0.0f, r0 = 0.0f;
        for (int r = 0; r < rows; r++) {
            float dot = 0.0f; for (int c = 0; c < cols; c++) dot += hdP[r * cols + c] * hP[r * cols + c];
            float rs = 0.0f;
            for (int c = 0; c < cols; c++) {
                float ref = hP[r * cols + c] * (hdP[r * cols + c] - dot);
                float got = hdA[r * cols + c];
                if (r == 0 && c == 0) { a0 = got; r0 = ref; }
                rs += got;
                float e = got - ref; if (e < 0) e = -e;
                if (isnan(got) || e > 1.0e-4f) { if (sbad < 4) fprintf(stderr, "softmax_bwd mismatch dA[%d,%d]=%g ref %g\n", r, c, got, ref); sbad++; }
            }
            float ers = rs < 0 ? -rs : rs; if (ers > maxrs) maxrs = ers;
            if (isnan(rs) || ers > 1.0e-3f) { if (sbad < 4) fprintf(stderr, "softmax_bwd row %d sum %g (want 0)\n", r, rs); sbad++; }
        }
        printf("GPU [%s] softmax_backward %dx%d: dA[0,0]=%g ref %g, max|row_sum|=%g, %d bad -> %s\n",
               gpu, rows, cols, a0, r0, maxrs, sbad, sbad ? "FAIL" : "PASS");
        cuMemFree(dP); cuMemFree(ddP); cuMemFree(ddA);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hP); free(hdP); free(hdA); free(ptx);
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

    /* layernorm_perf mode (T2/M4): cuda_launch <combined.ptx> layernorm_blockred <Nignored>
     *   layernorm_perf <rows> <cols> [mutate].
     * Correctness + faster-than-naive harness for the WARP/BLOCK-REDUCTION row LayerNorm.
     * combined.ptx carries BOTH the kovc block-reduction kernel (argv[2]) AND the naive
     * one-thread-per-row gpu_layernorm (baseline). Non-trivial per-column gamma/beta (so a
     * kernel that ignored gamma/beta would mismatch). CORRECTNESS: every cell vs a CPU
     * affine layernorm reference gamma[c]*(x-mean)/sqrt(var)+beta[c]; report maxrel. The tol
     * (LN_TOL, default 1e-3) accounts HONESTLY for rsqrt.approx. TIMING: block-reduction
     * grid=(rows,1,1) block=(256,1,1) vs naive block=(1,1,1); speedup-vs-naive. The "mutate"
     * arg perturbs one y cell (comparator neg-control -> must FAIL). */
    if (strcmp(op, "layernorm_perf") == 0) {
        int rows = (argc > 5) ? atoi(argv[5]) : 64;
        int cols = (argc > 6) ? atoi(argv[6]) : 256;
        int mutate = (argc > 7 && strcmp(argv[7], "mutate") == 0);
        float LN_TOL = 1.0e-3f; { const char* e; if ((e = getenv("LN_TOL"))) LN_TOL = (float)atof(e); }
        CUfunction f_naive;
        CK(cuModuleGetFunction(&f_naive, mod, "gpu_layernorm"), "get gpu_layernorm (naive baseline)");
        size_t ne = (size_t)rows * cols;
        float* hx = (float*)malloc(ne * sizeof(float));
        float* hy = (float*)malloc(ne * sizeof(float));
        float* hg = (float*)malloc((size_t)cols * sizeof(float));
        float* hbe = (float*)malloc((size_t)cols * sizeof(float));
        if (!hx || !hy || !hg || !hbe) return 2;
        for (size_t i = 0; i < ne; i++) hx[i] = (float)((int)((i * 7 + 3) % 13) - 6);
        for (int c = 0; c < cols; c++) { hg[c] = 1.0f + 0.25f * (float)(c % 4); hbe[c] = 0.5f * (float)((c % 3) - 1); }
        CUdeviceptr dx, dy, dg, db;
        CK(cuMemAlloc(&dx, ne * sizeof(float)), "alloc x");
        CK(cuMemAlloc(&dy, ne * sizeof(float)), "alloc y");
        CK(cuMemAlloc(&dg, (size_t)cols * sizeof(float)), "alloc g");
        CK(cuMemAlloc(&db, (size_t)cols * sizeof(float)), "alloc b");
        for (size_t i = 0; i < ne; i++) hy[i] = -7.0f;
        CK(cuMemcpyHtoD(dx, hx, ne * sizeof(float)), "H2D x");
        CK(cuMemcpyHtoD(dy, hy, ne * sizeof(float)), "H2D y(sentinel)");
        CK(cuMemcpyHtoD(dg, hg, (size_t)cols * sizeof(float)), "H2D g");
        CK(cuMemcpyHtoD(db, hbe, (size_t)cols * sizeof(float)), "H2D b");
        void* largs[] = { &dx, &dy, &dg, &db, &cols };
        CK(cuLaunchKernel(fn, rows, 1, 1, 256, 1, 1, 0, 0, largs, 0), "launch layernorm_blockred");
        CK(cuCtxSynchronize(), "sync layernorm_blockred");
        CK(cuMemcpyDtoH(hy, dy, ne * sizeof(float)), "D2H y");
        if (mutate) hy[0] += 1.0f;
        /* CPU affine layernorm reference: maxrel. */
        int lbad = 0; float maxrel = 0.0f;
        for (int r = 0; r < rows; r++) {
            float mean = 0.0f; for (int c = 0; c < cols; c++) mean += hx[r * cols + c]; mean /= (float)cols;
            float v = 0.0f; for (int c = 0; c < cols; c++) { float d = hx[r * cols + c] - mean; v += d * d; } v /= (float)cols;
            float inv = 1.0f / sqrtf(v);
            for (int c = 0; c < cols; c++) {
                float norm = (hx[r * cols + c] - mean) * inv;
                float ref = hg[c] * norm + hbe[c];
                float got = hy[r * cols + c];
                float d = got - ref; if (d < 0) d = -d;
                float ar = ref < 0 ? -ref : ref; float rel = d / (ar > 1e-6f ? ar : 1e-6f);
                if (rel > maxrel) maxrel = rel;
                if (isnan(got) || (d > LN_TOL && rel > LN_TOL)) { if (lbad < 6) fprintf(stderr, "layernorm_perf mismatch y[%d,%d]=%g ref %g (rel %.2e)\n", r, c, got, ref, rel); lbad++; }
            }
        }
        /* TIMING (kernel-only): block-reduction vs naive. */
        double br_med = 0.0, nv_med = 0.0;
        if (!mutate) {
            int WARM = 5, ITERS = 50;
            float* ts = (float*)malloc((size_t)ITERS * sizeof(float));
            CUevent e0, e1; CK(cuEventCreate(&e0, 0), "evt0"); CK(cuEventCreate(&e1, 0), "evt1");
            for (int w = 0; w < WARM; w++) CK(cuLaunchKernel(fn, rows, 1, 1, 256, 1, 1, 0, 0, largs, 0), "warm br");
            CK(cuCtxSynchronize(), "sync warm br");
            for (int it = 0; it < ITERS; it++) { CK(cuEventRecord(e0, 0), "r0"); CK(cuLaunchKernel(fn, rows, 1, 1, 256, 1, 1, 0, 0, largs, 0), "time br"); CK(cuEventRecord(e1, 0), "r1"); CK(cuEventSynchronize(e1), "es"); CK(cuEventElapsedTime(&ts[it], e0, e1), "el"); }
            for (int i = 0; i < ITERS; i++) for (int j = i + 1; j < ITERS; j++) if (ts[j] < ts[i]) { float t = ts[i]; ts[i] = ts[j]; ts[j] = t; }
            br_med = (double)ts[ITERS / 2];
            for (int w = 0; w < WARM; w++) CK(cuLaunchKernel(f_naive, rows, 1, 1, 1, 1, 1, 0, 0, largs, 0), "warm nv");
            CK(cuCtxSynchronize(), "sync warm nv");
            for (int it = 0; it < ITERS; it++) { CK(cuEventRecord(e0, 0), "n0"); CK(cuLaunchKernel(f_naive, rows, 1, 1, 1, 1, 1, 0, 0, largs, 0), "time nv"); CK(cuEventRecord(e1, 0), "n1"); CK(cuEventSynchronize(e1), "nes"); CK(cuEventElapsedTime(&ts[it], e0, e1), "nel"); }
            for (int i = 0; i < ITERS; i++) for (int j = i + 1; j < ITERS; j++) if (ts[j] < ts[i]) { float t = ts[i]; ts[i] = ts[j]; ts[j] = t; }
            nv_med = (double)ts[ITERS / 2];
            cuEventDestroy(e0); cuEventDestroy(e1); free(ts);
            printf("GPU [%s] TIMING layernorm %dx%d: blockred med=%.4f ms  naive med=%.4f ms  SPEEDUP=%.2fx\n", gpu, rows, cols, br_med, nv_med, nv_med / br_med);
        }
        printf("GPU [%s] layernorm_perf (block-reduction 256t/row, affine) %dx%d%s: maxrel=%.2e (tol=%.0e), %d bad -> %s\n",
               gpu, rows, cols, mutate ? " [MUTATED]" : "", maxrel, LN_TOL, lbad, lbad ? "FAIL" : "PASS");
        cuMemFree(dx); cuMemFree(dy); cuMemFree(dg); cuMemFree(db);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hx); free(hy); free(hg); free(hbe);
        return lbad ? 1 : 0;
    }

    /* gelu mode: cuda_launch <ptx> gpu_gelu <N> gelu. y=0.5*x*(1+tanh(0.7978846*
     * (x+0.044715*x^3))), the tanh GELU. CPU ref mirrors stdlib __gelu exactly.
     * Inputs in [-3,3] so e^(2z) never saturates. Combines f32 literals + __gpu_exp. */
    if (strcmp(op, "gelu") == 0) {
        /* optional "mutate" (argv[5], after <ptx> <kname> <N> <op>) = comparator-teeth control. */
        int mutate = (argc > 5 && strcmp(argv[5], "mutate") == 0);
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
        if (mutate) hy[0] += 1.0f;   /* comparator negative control -> must FAIL */
        int gbad = 0; float gref0 = 0.0f, maxrel = 0.0f, maxabs = 0.0f;
        for (size_t i = 0; i < ne; i++) {
            float xx = hx[i];
            float x3 = xx * xx * xx;
            float inner = 0.7978846f * (xx + 0.044715f * x3);
            float e2 = expf(2.0f * inner);
            float th = (e2 - 1.0f) / (e2 + 1.0f);
            float ref = 0.5f * xx * (1.0f + th);
            if (i == 0) gref0 = ref;
            float got = hy[i];
            float d = got - ref; if (d < 0) d = -d; if (d > maxabs) maxabs = d;
            float ar = ref < 0 ? -ref : ref; float rel = d / (ar > 1e-6f ? ar : 1e-6f);
            if (rel > maxrel) maxrel = rel;
            if (isnan(got) || d > 1.0e-3f) { if (gbad < 4) fprintf(stderr, "gelu mismatch y[%zu]=%g ref %g (x=%g)\n", i, got, ref, xx); gbad++; }
        }
        /* THROUGHPUT (kernel-only, cuEvent): elementwise GELU is MEMORY-BOUND
         * (read N + write N f32 = 8N bytes). NO naive/tiled pair exists -- this
         * IS the elementwise form -- so we report GB/s honestly, never a fake
         * speedup. Gate rests on correctness (gbad==0). */
        if (!mutate) {
            int WARM = 5, ITERS = 50;
            float* ts = (float*)malloc((size_t)ITERS * sizeof(float));
            CUevent e0, e1; CK(cuEventCreate(&e0, 0), "evt0"); CK(cuEventCreate(&e1, 0), "evt1");
            for (int w = 0; w < WARM; w++) CK(cuLaunchKernel(fn, N, 1, 1, 1, 1, 1, 0, 0, gargs, 0), "warm gelu");
            CK(cuCtxSynchronize(), "sync warm gelu");
            for (int it = 0; it < ITERS; it++) { CK(cuEventRecord(e0, 0), "r0"); CK(cuLaunchKernel(fn, N, 1, 1, 1, 1, 1, 0, 0, gargs, 0), "time gelu"); CK(cuEventRecord(e1, 0), "r1"); CK(cuEventSynchronize(e1), "es"); CK(cuEventElapsedTime(&ts[it], e0, e1), "el"); }
            for (int i = 0; i < ITERS; i++) for (int j = i + 1; j < ITERS; j++) if (ts[j] < ts[i]) { float t = ts[i]; ts[i] = ts[j]; ts[j] = t; }
            double med = (double)ts[ITERS / 2];
            double gbps = (8.0 * (double)ne) / (med * 1.0e-3) / 1.0e9;
            cuEventDestroy(e0); cuEventDestroy(e1); free(ts);
            printf("GPU [%s] THROUGHPUT gelu N=%d: med=%.4f ms  %.1f GB/s (8N B, memory-bound elementwise)\n", gpu, N, med, gbps);
        }
        printf("GPU [%s] gelu N=%d (tanh approx, f32 literals + exp)%s: y[0]=%g ref %g, maxrel=%.2e maxabs=%.2e (tol 1e-3), %d bad -> %s\n",
               gpu, N, mutate ? " [MUTATED]" : "", hy[0], gref0, maxrel, maxabs, gbad, gbad ? "FAIL" : "PASS");
        cuMemFree(dx); cuMemFree(dy);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hx); free(hy);
        return gbad ? 1 : 0;
    }

    /* adam mode: cuda_launch <ptx> gpu_adam <N> adam. One in-place Adam step on w (with g,m,v).
     * bc1,bc2 (bias-correction) passed as 1-elem arrays; here bc1=10, bc2=1000 (step t=1). Verify
     * nm,nv (tol 1e-5, exact arithmetic) and new_w (tol 1e-4, rsqrt.approx) vs an independent CPU
     * Adam step with the same baked literals. */
    if (strcmp(op, "adam") == 0) {
        /* optional "mutate" (argv[5], after <ptx> <kname> <N> <op>) = comparator-teeth
         * control. Re-runs are non-idempotent (Adam mutates w,m,v in place), so timing
         * re-uploads the pristine host arrays before each launch. */
        int mutate = (argc > 5 && strcmp(argv[5], "mutate") == 0);
        size_t ne = (size_t)N;
        float* hw = (float*)malloc(ne * sizeof(float));
        float* hg = (float*)malloc(ne * sizeof(float));
        float* hm = (float*)malloc(ne * sizeof(float));
        float* hv = (float*)malloc(ne * sizeof(float));
        float* hw0 = (float*)malloc(ne * sizeof(float));
        float* hm0 = (float*)malloc(ne * sizeof(float));
        float* hv0 = (float*)malloc(ne * sizeof(float));
        if (!hw || !hg || !hm || !hv || !hw0 || !hm0 || !hv0) return 2;
        for (size_t i = 0; i < ne; i++) {
            hw[i] = hw0[i] = (float)((int)(i % 5) - 2);
            hg[i] = (float)((int)(i % 7) - 3) * 0.1f;
            hm[i] = hm0[i] = (float)((int)(i % 3) - 1) * 0.05f;
            hv[i] = hv0[i] = (float)(i % 4) * 0.01f;
        }
        float hbc1 = 10.0f, hbc2 = 1000.0f;
        CUdeviceptr dw, dg, dm, dv, dbc1, dbc2;
        CK(cuMemAlloc(&dw, ne * sizeof(float)), "alloc w");
        CK(cuMemAlloc(&dg, ne * sizeof(float)), "alloc g");
        CK(cuMemAlloc(&dm, ne * sizeof(float)), "alloc m");
        CK(cuMemAlloc(&dv, ne * sizeof(float)), "alloc v");
        CK(cuMemAlloc(&dbc1, sizeof(float)), "alloc bc1");
        CK(cuMemAlloc(&dbc2, sizeof(float)), "alloc bc2");
        CK(cuMemcpyHtoD(dw, hw, ne * sizeof(float)), "H2D w");
        CK(cuMemcpyHtoD(dg, hg, ne * sizeof(float)), "H2D g");
        CK(cuMemcpyHtoD(dm, hm, ne * sizeof(float)), "H2D m");
        CK(cuMemcpyHtoD(dv, hv, ne * sizeof(float)), "H2D v");
        CK(cuMemcpyHtoD(dbc1, &hbc1, sizeof(float)), "H2D bc1");
        CK(cuMemcpyHtoD(dbc2, &hbc2, sizeof(float)), "H2D bc2");
        void* aargs[] = { &dw, &dg, &dm, &dv, &dbc1, &dbc2 };
        CK(cuLaunchKernel(fn, N, 1, 1, 1, 1, 1, 0, 0, aargs, 0), "launch adam");
        CK(cuCtxSynchronize(), "sync adam");
        CK(cuMemcpyDtoH(hw, dw, ne * sizeof(float)), "D2H w");
        CK(cuMemcpyDtoH(hm, dm, ne * sizeof(float)), "D2H m");
        CK(cuMemcpyDtoH(hv, dv, ne * sizeof(float)), "D2H v");
        if (mutate) hw[0] += 1.0f;   /* comparator negative control -> must FAIL */
        int abad = 0; float w0 = 0.0f, w0ref = 0.0f, maxrel = 0.0f;
        for (size_t i = 0; i < ne; i++) {
            float nm = 0.9f * hm0[i] + 0.1f * hg[i];
            float nv = 0.999f * hv0[i] + 0.001f * hg[i] * hg[i];
            float mh = nm * hbc1, vh = nv * hbc2;
            float nw = hw0[i] - 0.001f * mh / sqrtf(vh + 1.0e-8f);
            if (i == 0) { w0 = hw[i]; w0ref = nw; }
            float em = hm[i] - nm; if (em < 0) em = -em;
            float ev = hv[i] - nv; if (ev < 0) ev = -ev;
            float ew = hw[i] - nw; if (ew < 0) ew = -ew;
            float aw = nw < 0 ? -nw : nw; float rw = ew / (aw > 1e-6f ? aw : 1e-6f);
            if (rw > maxrel) maxrel = rw;
            if (isnan(hw[i]) || em > 1.0e-5f || ev > 1.0e-5f || ew > 1.0e-4f) { if (abad < 4) fprintf(stderr, "adam mismatch i=%zu w=%g(ref %g) m=%g(ref %g) v=%g(ref %g)\n", i, hw[i], nw, hm[i], nm, hv[i], nv); abad++; }
        }
        /* THROUGHPUT (kernel-only, cuEvent): the Adam step is MEMORY-BOUND --
         * reads w,g,m,v + writes w,m,v = 7 array touches = 28N bytes. NO
         * naive/tiled pair (this IS the elementwise form), so report GB/s
         * honestly; gate rests on correctness (abad==0). Re-upload w,m,v each
         * launch (in-place mutation). */
        if (!mutate) {
            int WARM = 5, ITERS = 50;
            float* ts = (float*)malloc((size_t)ITERS * sizeof(float));
            CUevent e0, e1; CK(cuEventCreate(&e0, 0), "evt0"); CK(cuEventCreate(&e1, 0), "evt1");
            for (int w = 0; w < WARM; w++) {
                CK(cuMemcpyHtoD(dw, hw0, ne * sizeof(float)), "re-w"); CK(cuMemcpyHtoD(dm, hm0, ne * sizeof(float)), "re-m"); CK(cuMemcpyHtoD(dv, hv0, ne * sizeof(float)), "re-v");
                CK(cuLaunchKernel(fn, N, 1, 1, 1, 1, 1, 0, 0, aargs, 0), "warm adam");
            }
            CK(cuCtxSynchronize(), "sync warm adam");
            for (int it = 0; it < ITERS; it++) {
                CK(cuMemcpyHtoD(dw, hw0, ne * sizeof(float)), "re-w"); CK(cuMemcpyHtoD(dm, hm0, ne * sizeof(float)), "re-m"); CK(cuMemcpyHtoD(dv, hv0, ne * sizeof(float)), "re-v");
                CK(cuCtxSynchronize(), "pre-time sync");
                CK(cuEventRecord(e0, 0), "r0"); CK(cuLaunchKernel(fn, N, 1, 1, 1, 1, 1, 0, 0, aargs, 0), "time adam"); CK(cuEventRecord(e1, 0), "r1"); CK(cuEventSynchronize(e1), "es"); CK(cuEventElapsedTime(&ts[it], e0, e1), "el");
            }
            for (int i = 0; i < ITERS; i++) for (int j = i + 1; j < ITERS; j++) if (ts[j] < ts[i]) { float t = ts[i]; ts[i] = ts[j]; ts[j] = t; }
            double med = (double)ts[ITERS / 2];
            double gbps = (28.0 * (double)ne) / (med * 1.0e-3) / 1.0e9;
            cuEventDestroy(e0); cuEventDestroy(e1); free(ts);
            printf("GPU [%s] THROUGHPUT adam N=%d: med=%.4f ms  %.1f GB/s (28N B, memory-bound elementwise)\n", gpu, N, med, gbps);
        }
        printf("GPU [%s] adam N=%d%s: w[0]=%g ref %g, maxrel(w)=%.2e (tol m/v 1e-5, w 1e-4), %d bad -> %s\n",
               gpu, N, mutate ? " [MUTATED]" : "", w0, w0ref, maxrel, abad, abad ? "FAIL" : "PASS");
        cuMemFree(dw); cuMemFree(dg); cuMemFree(dm); cuMemFree(dv); cuMemFree(dbc1); cuMemFree(dbc2);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hw); free(hg); free(hm); free(hv); free(hw0); free(hm0); free(hv0); free(ptx);
        return abad ? 1 : 0;
    }

    /* scale mode: cuda_launch <ptx> gpu_scale_inplace <N> scale. a[i]=0.25*a[i] in place. */
    if (strcmp(op, "scale") == 0) {
        size_t ne = (size_t)N;
        float* ha = (float*)malloc(ne * sizeof(float));
        float* ha0 = (float*)malloc(ne * sizeof(float));
        if (!ha || !ha0) return 2;
        for (size_t i = 0; i < ne; i++) ha[i] = ha0[i] = (float)((int)(i % 11) - 5);
        CUdeviceptr da;
        CK(cuMemAlloc(&da, ne * sizeof(float)), "alloc a");
        CK(cuMemcpyHtoD(da, ha, ne * sizeof(float)), "H2D a");
        void* scargs[] = { &da, &N };
        CK(cuLaunchKernel(fn, N, 1, 1, 1, 1, 1, 0, 0, scargs, 0), "launch scale");
        CK(cuCtxSynchronize(), "sync scale");
        CK(cuMemcpyDtoH(ha, da, ne * sizeof(float)), "D2H a");
        int scbad = 0;
        for (size_t i = 0; i < ne; i++) {
            float ref = 0.25f * ha0[i];
            float e = ha[i] - ref; if (e < 0) e = -e;
            if (isnan(ha[i]) || e > 1.0e-6f) { if (scbad < 4) fprintf(stderr, "scale mismatch a[%zu]=%g ref %g\n", i, ha[i], ref); scbad++; }
        }
        printf("GPU [%s] scale N=%d: a[1]=%g ref %g, %d bad -> %s\n",
               gpu, N, ha[1], 0.25f * ha0[1], scbad, scbad ? "FAIL" : "PASS");
        cuMemFree(da);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(ha); free(ha0); free(ptx);
        return scbad ? 1 : 0;
    }

    /* layernorm_save mode: cuda_launch <ptx> gpu_layernorm_fwd_save <N> layernorm_save <rows> <cols>.
     * Forward layernorm with non-trivial gamma/beta, ALSO writing ist[row]=1/sqrt(var). Verify y
     * per-cell vs CPU affine layernorm (1e-3) AND ist[row]=1/sqrt(var_row) (1e-3, independent). */
    if (strcmp(op, "layernorm_save") == 0) {
        int rows = (argc > 5) ? atoi(argv[5]) : 8;
        int cols = (argc > 6) ? atoi(argv[6]) : 16;
        size_t ne = (size_t)rows * cols;
        float* hx = (float*)malloc(ne * sizeof(float));
        float* hy = (float*)malloc(ne * sizeof(float));
        float* hg = (float*)malloc((size_t)cols * sizeof(float));
        float* hb = (float*)malloc((size_t)cols * sizeof(float));
        float* hist = (float*)malloc((size_t)rows * sizeof(float));
        if (!hx || !hy || !hg || !hb || !hist) return 2;
        for (size_t i = 0; i < ne; i++) hx[i] = (float)((int)((i * 7 + 3) % 13) - 6);
        for (int c = 0; c < cols; c++) { hg[c] = 1.0f + 0.25f * (float)(c % 4); hb[c] = 0.5f * (float)((c % 3) - 1); }
        CUdeviceptr dx, dy, dg, db, dist;
        CK(cuMemAlloc(&dx, ne * sizeof(float)), "alloc x");
        CK(cuMemAlloc(&dy, ne * sizeof(float)), "alloc y");
        CK(cuMemAlloc(&dg, (size_t)cols * sizeof(float)), "alloc g");
        CK(cuMemAlloc(&db, (size_t)cols * sizeof(float)), "alloc b");
        CK(cuMemAlloc(&dist, (size_t)rows * sizeof(float)), "alloc ist");
        CK(cuMemcpyHtoD(dx, hx, ne * sizeof(float)), "H2D x");
        CK(cuMemcpyHtoD(dg, hg, (size_t)cols * sizeof(float)), "H2D g");
        CK(cuMemcpyHtoD(db, hb, (size_t)cols * sizeof(float)), "H2D b");
        void* largs[] = { &dx, &dy, &dg, &db, &dist, &cols };
        CK(cuLaunchKernel(fn, rows, 1, 1, 1, 1, 1, 0, 0, largs, 0), "launch ln_save");
        CK(cuCtxSynchronize(), "sync ln_save");
        CK(cuMemcpyDtoH(hy, dy, ne * sizeof(float)), "D2H y");
        CK(cuMemcpyDtoH(hist, dist, (size_t)rows * sizeof(float)), "D2H ist");
        int lbad = 0; float i0 = 0.0f, i0ref = 0.0f;
        for (int r = 0; r < rows; r++) {
            float mean = 0.0f; for (int c = 0; c < cols; c++) mean += hx[r * cols + c]; mean /= (float)cols;
            float var = 0.0f; for (int c = 0; c < cols; c++) { float d = hx[r * cols + c] - mean; var += d * d; } var /= (float)cols;
            float invref = 1.0f / sqrtf(var);
            if (r == 0) { i0 = hist[r]; i0ref = invref; }
            float ei = hist[r] - invref; if (ei < 0) ei = -ei;
            if (isnan(hist[r]) || ei > 1.0e-3f) { if (lbad < 4) fprintf(stderr, "ln_save ist[%d]=%g ref %g\n", r, hist[r], invref); lbad++; }
            for (int c = 0; c < cols; c++) {
                float ref = hg[c] * ((hx[r * cols + c] - mean) * invref) + hb[c];
                float got = hy[r * cols + c];
                float e = got - ref; if (e < 0) e = -e;
                if (isnan(got) || e > 1.0e-3f) { if (lbad < 4) fprintf(stderr, "ln_save y[%d,%d]=%g ref %g\n", r, c, got, ref); lbad++; }
            }
        }
        printf("GPU [%s] layernorm_save %dx%d: ist[0]=%g ref %g, %d bad -> %s\n",
               gpu, rows, cols, i0, i0ref, lbad, lbad ? "FAIL" : "PASS");
        cuMemFree(dx); cuMemFree(dy); cuMemFree(dg); cuMemFree(db); cuMemFree(dist);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hx); free(hy); free(hg); free(hb); free(hist); free(ptx);
        return lbad ? 1 : 0;
    }

    /* layernorm_bwd_dx mode: cuda_launch <ptx> gpu_layernorm_backward_dx <N> layernorm_bwd_dx <rows> <cols>.
     * dx = layernorm input gradient. INDEPENDENT ref = central finite-difference of the layernorm
     * FORWARD: dx_fd[s,c]=(L(x+h e_sc)-L(x-h e_sc))/2h, L=sum_k dy*y, h=1e-3, tol 2e-2; plus the
     * conservation law sum_c dx[s,:] ~ 0 (tol 1e-3). gamma non-trivial; ist computed on CPU. */
    if (strcmp(op, "layernorm_bwd_dx") == 0) {
        int rows = (argc > 5) ? atoi(argv[5]) : 8;
        int cols = (argc > 6) ? atoi(argv[6]) : 16;
        size_t ne = (size_t)rows * cols;
        float* hx = (float*)malloc(ne * sizeof(float));
        float* hdy = (float*)malloc(ne * sizeof(float));
        float* hg = (float*)malloc((size_t)cols * sizeof(float));
        float* hist = (float*)malloc((size_t)rows * sizeof(float));
        float* hdx = (float*)malloc(ne * sizeof(float));
        if (!hx || !hdy || !hg || !hist || !hdx) return 2;
        for (size_t i = 0; i < ne; i++) { hx[i] = (float)((int)((i * 7 + 3) % 13) - 6); hdy[i] = (float)((int)((i * 5 + 1) % 7) - 3) * 0.5f; }
        for (int c = 0; c < cols; c++) hg[c] = 1.0f + 0.25f * (float)(c % 4);
        for (int r = 0; r < rows; r++) { float m = 0.0f; for (int c = 0; c < cols; c++) m += hx[r * cols + c]; m /= (float)cols; float v = 0.0f; for (int c = 0; c < cols; c++) { float d = hx[r * cols + c] - m; v += d * d; } v /= (float)cols; hist[r] = 1.0f / sqrtf(v); }
        CUdeviceptr dx_, ddy, dg, dist, ddx;
        CK(cuMemAlloc(&dx_, ne * sizeof(float)), "alloc x");
        CK(cuMemAlloc(&ddy, ne * sizeof(float)), "alloc dy");
        CK(cuMemAlloc(&dg, (size_t)cols * sizeof(float)), "alloc g");
        CK(cuMemAlloc(&dist, (size_t)rows * sizeof(float)), "alloc ist");
        CK(cuMemAlloc(&ddx, ne * sizeof(float)), "alloc dx");
        CK(cuMemcpyHtoD(dx_, hx, ne * sizeof(float)), "H2D x");
        CK(cuMemcpyHtoD(ddy, hdy, ne * sizeof(float)), "H2D dy");
        CK(cuMemcpyHtoD(dg, hg, (size_t)cols * sizeof(float)), "H2D g");
        CK(cuMemcpyHtoD(dist, hist, (size_t)rows * sizeof(float)), "H2D ist");
        void* ax[] = { &dx_, &ddy, &dg, &dist, &ddx, &cols };
        CK(cuLaunchKernel(fn, rows, 1, 1, 1, 1, 1, 0, 0, ax, 0), "launch ln_bwd_dx");
        CK(cuCtxSynchronize(), "sync ln_bwd_dx");
        CK(cuMemcpyDtoH(hdx, ddx, ne * sizeof(float)), "D2H dx");
        int bad = 0; float maxfd = 0.0f, maxrs = 0.0f, d0 = 0.0f, r0 = 0.0f, h = 1.0e-3f;
        for (int s = 0; s < rows; s++) {
            float rs = 0.0f;
            for (int c = 0; c < cols; c++) {
                float fd = (ln_rowL(&hx[s * cols], &hdy[s * cols], hg, cols, c, h) - ln_rowL(&hx[s * cols], &hdy[s * cols], hg, cols, c, -h)) / (2.0f * h);
                float got = hdx[s * cols + c];
                if (s == 0 && c == 0) { d0 = got; r0 = fd; }
                rs += got;
                float e = got - fd; if (e < 0) e = -e; if (e > maxfd) maxfd = e;
                if (isnan(got) || e > 2.0e-2f) { if (bad < 4) fprintf(stderr, "ln_bwd_dx mismatch dx[%d,%d]=%g fd %g\n", s, c, got, fd); bad++; }
            }
            float ers = rs < 0 ? -rs : rs; if (ers > maxrs) maxrs = ers;
            if (isnan(rs) || ers > 1.0e-3f) { if (bad < 4) fprintf(stderr, "ln_bwd_dx row %d sum %g (want 0)\n", s, rs); bad++; }
        }
        printf("GPU [%s] layernorm_bwd_dx %dx%d: dx[0,0]=%g fd %g, max|dx-fd|=%g, max|row_sum|=%g, %d bad -> %s\n",
               gpu, rows, cols, d0, r0, maxfd, maxrs, bad, bad ? "FAIL" : "PASS");
        cuMemFree(dx_); cuMemFree(ddy); cuMemFree(dg); cuMemFree(dist); cuMemFree(ddx);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hx); free(hdy); free(hg); free(hist); free(hdx); free(ptx);
        return bad ? 1 : 0;
    }

    /* layernorm_bwd_dgb mode: cuda_launch <ptx> gpu_layernorm_backward_dgb <N> layernorm_bwd_dgb <rows> <cols>.
     * dgamma=dgb[c], dbeta=dgb[cols+c]. Verify per-cell vs a CPU reduction dgamma[c]=sum_s
     * dy[s,c]*xhat[s,c], dbeta[c]=sum_s dy[s,c] (xhat=(x-mean_s)*ist[s]), tol 1e-3. */
    if (strcmp(op, "layernorm_bwd_dgb") == 0) {
        int rows = (argc > 5) ? atoi(argv[5]) : 8;
        int cols = (argc > 6) ? atoi(argv[6]) : 16;
        size_t ne = (size_t)rows * cols;
        float* hx = (float*)malloc(ne * sizeof(float));
        float* hdy = (float*)malloc(ne * sizeof(float));
        float* hist = (float*)malloc((size_t)rows * sizeof(float));
        float* hdgb = (float*)malloc((size_t)(2 * cols) * sizeof(float));
        if (!hx || !hdy || !hist || !hdgb) return 2;
        for (size_t i = 0; i < ne; i++) { hx[i] = (float)((int)((i * 7 + 3) % 13) - 6); hdy[i] = (float)((int)((i * 5 + 1) % 7) - 3) * 0.5f; }
        for (int r = 0; r < rows; r++) { float m = 0.0f; for (int c = 0; c < cols; c++) m += hx[r * cols + c]; m /= (float)cols; float v = 0.0f; for (int c = 0; c < cols; c++) { float d = hx[r * cols + c] - m; v += d * d; } v /= (float)cols; hist[r] = 1.0f / sqrtf(v); }
        CUdeviceptr dx_, ddy, dist, ddgb;
        CK(cuMemAlloc(&dx_, ne * sizeof(float)), "alloc x");
        CK(cuMemAlloc(&ddy, ne * sizeof(float)), "alloc dy");
        CK(cuMemAlloc(&dist, (size_t)rows * sizeof(float)), "alloc ist");
        CK(cuMemAlloc(&ddgb, (size_t)(2 * cols) * sizeof(float)), "alloc dgb");
        CK(cuMemcpyHtoD(dx_, hx, ne * sizeof(float)), "H2D x");
        CK(cuMemcpyHtoD(ddy, hdy, ne * sizeof(float)), "H2D dy");
        CK(cuMemcpyHtoD(dist, hist, (size_t)rows * sizeof(float)), "H2D ist");
        void* ag[] = { &dx_, &ddy, &dist, &ddgb, &rows, &cols };
        CK(cuLaunchKernel(fn, cols, 1, 1, 1, 1, 1, 0, 0, ag, 0), "launch ln_bwd_dgb");
        CK(cuCtxSynchronize(), "sync ln_bwd_dgb");
        CK(cuMemcpyDtoH(hdgb, ddgb, (size_t)(2 * cols) * sizeof(float)), "D2H dgb");
        int bad = 0; float maxe = 0.0f, g0 = 0.0f, g0r = 0.0f;
        for (int c = 0; c < cols; c++) {
            float dgr = 0.0f, dbr = 0.0f;
            for (int s = 0; s < rows; s++) {
                float m = 0.0f; for (int k = 0; k < cols; k++) m += hx[s * cols + k]; m /= (float)cols;
                float xh = (hx[s * cols + c] - m) * hist[s];
                dgr += hdy[s * cols + c] * xh;
                dbr += hdy[s * cols + c];
            }
            float gg = hdgb[c], gb = hdgb[cols + c];
            if (c == 0) { g0 = gg; g0r = dgr; }
            float eg = gg - dgr; if (eg < 0) eg = -eg; if (eg > maxe) maxe = eg;
            float eb = gb - dbr; if (eb < 0) eb = -eb; if (eb > maxe) maxe = eb;
            if (isnan(gg) || isnan(gb) || eg > 1.0e-3f || eb > 1.0e-3f) { if (bad < 4) fprintf(stderr, "ln_bwd_dgb c=%d dgamma=%g(ref %g) dbeta=%g(ref %g)\n", c, gg, dgr, gb, dbr); bad++; }
        }
        printf("GPU [%s] layernorm_bwd_dgb %dx%d: dgamma[0]=%g ref %g, max|err|=%g, %d bad -> %s\n",
               gpu, rows, cols, g0, g0r, maxe, bad, bad ? "FAIL" : "PASS");
        cuMemFree(dx_); cuMemFree(ddy); cuMemFree(dist); cuMemFree(ddgb);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hx); free(hdy); free(hist); free(hdgb); free(ptx);
        return bad ? 1 : 0;
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
