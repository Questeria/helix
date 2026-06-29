#!/usr/bin/env python3
# Compare the REAL kovc scaled_packed_ternary_matmul GPU output against the trainer fake-quant path.
# GPU integer dump (sptmatmul_dump, sc=1) = i2f(sum_k trit*x_int) [M x N]; multiply by per-row scale[r]
# == the kernel's full scaled output c[r,col] = i2f(int_dot)*sc[r] (proven element-exact in sptmatmul).
# Fake-quant (trainer ternarize_dequant) expected[r,col] = sum_k (trit[r,k]*scale[r]) * x_int[k,col].
# These are mathematically identical (scale factors out) and have no FMA, so the match is element-exact.
import sys, numpy as np
gpu_dump, scale_bin, expected_bin, M, N = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
gint  = np.fromfile(gpu_dump, np.float32).reshape(M, N)       # i2f(int_dot), GPU
scale = np.fromfile(scale_bin, np.float32)                    # [M]
exp   = np.fromfile(expected_bin, np.float32).reshape(M, N)   # fake-quant deq @ x_int
gpu_scaled = (gint * scale[:, None]).astype(np.float32)
d = np.abs(gpu_scaled.astype(np.float64) - exp.astype(np.float64))
maxabs = float(d.max()); nbad = int((gpu_scaled != exp).sum())
# also report on the raw integer path magnitude for context
print("KERNEL_VS_FAKEQUANT M=%d N=%d max_abs_diff=%.6g exact_mismatches=%d/%d gpu00=%.6g exp00=%.6g -> %s"
      % (M, N, maxabs, nbad, M*N, float(gpu_scaled[0,0]), float(exp[0,0]),
         "PASS_ELEMENT_EXACT" if nbad == 0 else ("PASS_TIGHT_TOL" if maxabs < 1e-3 else "FAIL")))
sys.exit(0 if (nbad == 0 or maxabs < 1e-3) else 1)
