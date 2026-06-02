#!/usr/bin/env bash
set -u
echo "=== ptxas 12.0 (bare) ==="; /usr/bin/ptxas --version 2>&1 | head -3
echo "=== ptxas 12.8 ==="; /usr/local/cuda/bin/ptxas --version 2>&1 | head -3
echo "=== nvidia-smi ==="; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1
echo "=== cublas header tf32 symbols ==="; grep -c 'CUBLAS_COMPUTE_32F_FAST_TF32\|CUBLAS_GEMM_DEFAULT_TENSOR_OP\|CUBLAS_TF32_TENSOR_OP_MATH\|CUBLAS_PEDANTIC_MATH' /usr/local/cuda/include/cublas_api.h
echo "=== driver bin present? ==="; ls -la /mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/_kovc_ptx_driver.bin 2>&1
echo "=== seed bin present? ==="; ls -la /mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/seed.bin 2>&1
echo "=== gcc + libcublas ==="; which gcc; ls /usr/local/cuda/lib64/libcublas.so 2>&1 | head -1
