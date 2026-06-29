// GPU knowledge-distillation (KD) softmax gradient (v1.9 ternary QAT, HX_KD path).
// The KD analogue of gpu_ce_softmax_grad: the top-of-net gradient when the loss is the
// cross-entropy between a FIXED teacher distribution and the student softmax, instead of
// hard-label CE. The fused form is the SAME shape as the CE grad but the subtracted target
// is the FULL teacher distribution (one float per vocab cell), not a one-hot:
//   dlogits[r,c] = ( softmax(logits[r,:]/T)[c] - teach[r,c] ) * (T*T)
// where logits = the STUDENT (ternary-QAT) logits, teach[r,:] = the precomputed teacher
// softmax row (already at temperature T, summing to 1), and T = the distillation temperature.
// The T*T factor restores the gradient magnitude when logits are softened by 1/T (the standard
// Hinton-distillation scaling). For T=1 this reduces to (student_softmax - teach), exactly
// mirroring (student_softmax - onehot) of the CE kernel.
//
// CRITICAL emitter constraint: the kovc PTX emitter declares EVERY @kernel param as .param .b64
// and has no scalar-f32 ABI -- the ONLY scalar params it lowers correctly are i32 (rows, cols)
// and the f32 ones must be DEVICE POINTERS (cf. gpu_scale_rt's `s: f32` which is a 1-elem buffer).
// So the temperature is passed as a 1-ELEMENT DEVICE BUFFER `tparm` (tparm[0]=T, tparm[1]=T*T),
// NOT a scalar -- reading tparm[0]/tparm[1] is a normal global load. (A scalar f32 temp param
// silently mis-decodes to garbage -> the grad collapses to 0; this pointer form is the fix.)
//
// One thread per row (gridDim.x=rows, blockDim.x=1). Three serial passes mirror
// gpu_ce_softmax_grad: (1) row max of logits/T, (2) exp+sum, (3) normalize, subtract the teacher
// cell, scale by T*T. `teach` is a full [rows,cols] f32 buffer (the host precomputes it once from
// the fp teacher forward and reuses it across epochs -- the teacher is fixed). s init via x-x (=0).
//
// Correctness: if the student logits EQUAL the teacher logits (so the student softmax equals
// teach), every dlogits cell is (teach-teach)*T*T = 0 -> the KD grad vanishes at the teacher
// (the host KD self-check exploits exactly this). Each grad row also sums to ~0 (both the student
// softmax and the teacher row sum to 1), the same conservation property the CE kernel has.
// Launch: cuda_launch out.ptx gpu_kd_softmax_grad <rows> kd_softmax_grad <rows> <cols> <tparm>
@kernel
fn gpu_kd_softmax_grad(logits: f32, teach: f32, dlogits: f32, rows: i32, cols: i32, tparm: f32) {
    let row = block_idx() * block_dim() + thread_idx();
    let base = row * cols;
    let invT = (logits[base] - logits[base] + 1.0) / tparm[0];
    let t2 = tparm[1];
    let mut m = logits[base] * invT;
    let mut j = 1;
    while j < cols {
        let lj = logits[base + j] * invT;
        if lj > m { m = lj };
        j = j + 1
    };
    let mut s = logits[base] - logits[base];
    let mut k = 0;
    while k < cols {
        s = s + __gpu_exp(logits[base + k] * invT - m);
        k = k + 1
    };
    let mut t = 0;
    while t < cols {
        let p = __gpu_exp(logits[base + t] * invT - m) / s;
        dlogits[base + t] = (p - teach[base + t]) * t2;
        t = t + 1
    }
}
