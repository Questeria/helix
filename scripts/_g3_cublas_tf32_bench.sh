#!/usr/bin/env bash
# STEP 2: standalone cuBLAS-TF32 GEMM TFLOP/s measurement at 2048^3 (RTX 3070 Laptop).
# Pure cuBLAS (no kovc kernel) so the measurement cannot be lost to codegen issues.
# warmup=10 + median-of-50, min/med/max. Tees result to /mnt/c durable files.
set -u
WORK=/tmp/g3bench
mkdir -p "$WORK"
cat > "$WORK/bench.c" <<'CEOF'
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
static int cmpf(const void*a,const void*b){float x=*(const float*)a,y=*(const float*)b;return (x>y)-(x<y);}
int main(int argc,char**argv){
  int N = argc>1?atoi(argv[1]):2048;
  int M=N,K=N;
  size_t aN=(size_t)M*K,bN=(size_t)K*N,cN=(size_t)M*N;
  float *hA=malloc(aN*4),*hB=malloc(bN*4);
  for(size_t i=0;i<aN;i++)hA[i]=(float)((i%251)-125)*0.01f;
  for(size_t i=0;i<bN;i++)hB[i]=(float)((i%241)-120)*0.01f;
  float *dA,*dB,*dC; cudaMalloc(&dA,aN*4);cudaMalloc(&dB,bN*4);cudaMalloc(&dC,cN*4);
  cudaMemcpy(dA,hA,aN*4,cudaMemcpyHostToDevice);
  cudaMemcpy(dB,hB,bN*4,cudaMemcpyHostToDevice);
  cublasHandle_t h; cublasCreate(&h);
  cublasSetMathMode(h,CUBLAS_TF32_TENSOR_OP_MATH);
  float al=1.0f,be=0.0f;
  /* row-major C=A*B  <=>  col-major C^T = B^T * A^T : GemmEx(N,N, N,M,K, B,A,C) */
  #define GEMM() cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&al,dB,CUDA_R_32F,N,dA,CUDA_R_32F,K,&be,dC,CUDA_R_32F,N,CUBLAS_COMPUTE_32F_FAST_TF32,CUBLAS_GEMM_DEFAULT_TENSOR_OP)
  cublasStatus_t st=GEMM(); if(st!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"GemmEx fail %d\n",st);return 2;}
  cudaDeviceSynchronize();
  int WARM=10,IT=50; float*ts=malloc(IT*4);
  cudaEvent_t e0,e1; cudaEventCreate(&e0);cudaEventCreate(&e1);
  for(int w=0;w<WARM;w++)GEMM();
  cudaDeviceSynchronize();
  for(int it=0;it<IT;it++){
    cudaEventRecord(e0,0); GEMM(); cudaEventRecord(e1,0); cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ts[it],e0,e1);
  }
  qsort(ts,IT,4,cmpf);
  double flop=2.0*(double)M*(double)N*(double)K;
  float tmin=ts[0],tmed=ts[IT/2],tmax=ts[IT-1];
  double tf_min=flop/(tmax*1e-3)/1e12; /* slowest time -> min TFLOPs */
  double tf_med=flop/(tmed*1e-3)/1e12;
  double tf_max=flop/(tmin*1e-3)/1e12; /* fastest time -> max TFLOPs */
  printf("CUBLAS_TF32 N=%d  ms[min=%.4f med=%.4f max=%.4f]  TFLOPs[min=%.3f med=%.3f max=%.3f]\n",
         N,tmin,tmed,tmax,tf_min,tf_med,tf_max);
  cublasDestroy(h); return 0;
}
CEOF
echo "=== compiling cuBLAS-TF32 bench ==="
gcc "$WORK/bench.c" -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcudart -lcublas -lm -o "$WORK/bench" \
  || { echo "FATAL gcc"; exit 2; }
echo "=== running @2048^3 (warmup 10 + median-of-50) ==="
"$WORK/bench" 2048 2>&1 | tee /mnt/c/Projects/Kovostov-Native/.m1probe/_g3_cublas_tf32_2048.txt
