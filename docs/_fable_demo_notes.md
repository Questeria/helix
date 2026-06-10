# Fable demo-complete run notes (2026-06-09, branch fable/demo-complete)

One lesson per entry; why it mattered.

1. **Run the full-model numpy oracle BEFORE touching any C.** The SmolLM2 oracle's first
   greedy run produced coherent English, which validated every pinned convention at once
   (BF16 widening, rotate_half RoPE @ theta=1e5, GQA 9/3, SwiGLU, RMSNorm, tied head,
   no-BOS tokenizer) for ~3 minutes of work. Any convention error would have cost a GPU
   debug session instead.

2. **bf16 -> f32 is a bit-shift, not arithmetic.** u32 = (u32)u16 << 16. This keeps the
   importer's "byte movement only" trust stance intact for BF16 checkpoints; numpy's
   (u16.astype(u32)<<16).view(f32) is the same operation, so importer and oracle agree
   bit-exactly on the widened weights (verified: 5 tensors byte-identical).

3. **Keep HF Linear weights [out,in] untransposed and use the A.Bt GEMM everywhere.**
   The verified tiled_matmul_abt kernel makes the natural HF layout directly consumable --
   the importer stays a pure byte-mover and no transpose code exists anywhere.

4. **The v2 weight header is self-describing (arch/NKV/theta/eps in bytes 40..55).** The
   worker peeks the header and sets ALL dims before any setup, so one binary serves both
   architectures and the gpt2 v1 path is provably untouched (regression: repacked 124M
   sha-identical c661e224).

5. **SmolLM2's dims are all %64** (DM 576, KVD 192, DFF 1536, NV 49152), so the tiled
   GEMM's M%64==N%64 constraint is satisfied with zero padding logic beyond GPT-2's Spad.
   Checked BEFORE writing code; a non-aligned model would have needed pad-columns design.

6. **Forward-declare before use in single-file C.** upload_layer_ll + read_ids_file were
   defined after their first use sites -> implicit-decl errors. Caught by compiling both
   worker configs immediately after the edit, before any GPU time.

7. **The rmsnorm kernel bakes eps=1e-5; fail closed on any other config eps.** The peek
   asserts header eps == 1e-5 and refuses to run otherwise -- silently-wrong numerics are
   worse than a refusal (the gate would catch it, but hours later).

8. **GPU output dumps + oracle-side comparison beats C-side comparators.** G-L1/G-L2
   keep the C worker dump-only; the comparison logic lives in the readable python oracle
   (compare-block0/compare-logits). Less trusted-C surface, easier to audit.
