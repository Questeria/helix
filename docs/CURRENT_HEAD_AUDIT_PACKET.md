# Helix — Current-Head Audit Packet (committed proof extract)

**Purpose.** A self-contained, **committed** record of the v1.3 trust results at the current head —
exact commands, pinned hashes, environment, and verbatim verdict lines — so a clean reader can see
them **without** relying on the gitignored process logs under `.stage33-logs/`. Companion to
`docs/CLEAN_REPRODUCTION.md` (method) and `docs/TRUST_CHAIN_CLOSED.md` (record).

> This is **our process evidence committed to the tree**, NOT an external/independent reproduction.
> An independent operator running a clean clone + publishing logs is the open residual that moves
> confidence past ~0.9 (see Residuals below). Every number here is reproducible by the commands shown.

- **Head:** the v1.3 final-convergence-pass head (this commit; prior verified tip `828480a`).
- **Date:** 2026-06-06.
- **Environment:** WSL2 (Linux 6.6 WSL2, x86_64); NVIDIA RTX 3070 Laptop GPU (sm_86), driver 596.21 /
  CUDA 13.2 runtime + 12.x `ptxas`; `gcc` (gnu89) as the independent DDC lineage. Build executed on a
  WSL-native **ext4 mirror** of the committed tree for speed (DrvFs per-syscall latency, ~75x); output
  is byte-identical (same fixpoint + driver SHA) — see `CLEAN_REPRODUCTION.md` "Where it walls".

## Static fence (committed-tree facts)

| Check | Command | Result |
|-------|---------|--------|
| Exactly 1 committed `.py` | `git ls-files "*.py" \| wc -l` | **1** — `verification/oracle/oracle_train.py` |
| 24 committed `.c`/`.h`, 15 604 LOC | `git ls-files "*.c" "*.h" \| wc -l` ; `\| xargs wc -l` | **24 / 15 604** |
| `seed.bin` gitignored + pinned | `git check-ignore` ; `sha256sum` vs `seed.sha256` | ignored; `9837db12…` == `seed.sha256` |

## The three result-bearing legs (verbatim verdict lines)

**1. Self-host fixpoint + corpus + PTX + diagnostics** — `bash scripts/gate_kovc.sh`
```
FIXPOINT OK (K2==K3==K4 byte-identical AND == pinned known-good)
  0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f
GPU PTX REGRESSION OK
CORPUS: 109 passed, 0 failed
CHECK_ERR: 4 passed, 0 failed
GATE_PASS
```

**2. gcc diverse-double-compile (seed→K1)** — `bash stage0/helixc-bootstrap/ddc_crosscheck.sh`
```
seed_gcc no-arg self-test exit=42 (want 42)
K1_m2  sha256=84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba
K1_gcc sha256=84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba
DDC_ANCHOR_OK -- gcc (independent lineage) reproduces the M2-Planet seed's K1 byte-for-byte AND == pinned known-good.
```

**3. GPU capstone (real transformer, kovc-emitted kernels)** — `bash scripts/capstone_audit.sh current-head`
```
[1] GATE_PASS (0992dddd... ; CORPUS: 109 passed, 0 failed)
[4a] backward finite-diff: PASS        (SAMPLED spot-check: 6 tensors x <=5 indices each)
[4b] train K=500: start loss 62.350887 -> final 0.415819   (fresh artifacts written this run)
[5] worst-case relative diff = 0.00000876 over 22 rows  (bar = 0.02)
    (oracle computes its OWN curve from shared init weights, then compares vs Helix's loss_curve.csv)
[6] NC-PERTURB ok (corrupted gpu_gelu_backward caught by finite-diff)
CAPSTONE_AUDIT_PASS
```

## Pinned hashes (release anchors)

| Artifact | SHA-256 | Size |
|----------|---------|------|
| `seed.bin` (gitignored; == `seed.sha256`) | `9837db12752a22159ca75a533910bc0d7b9afb35df9b9963f256b7b1b915c9bb` | 62 467 B |
| `K1` (seed→K1; gcc-DDC pinned) | `84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba` | 697 425 B |
| self-host fixpoint `K2==K3==K4` | `0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f` | 698 392 B |

## Honest residuals (NOT closed by this packet)

1. **Independent-operator reproduction is still open** — these are our process logs committed to the
   tree, not a third-party clean-clone run. That external step is what moves confidence > ~0.9.
2. **Shared TCB** — OS / kernel / filesystem / shell / coreutils / gcc / libc / binutils / loader /
   CPU+microcode / RAM, and the audited `seed.c` source, remain trusted (`TRUST_CHAIN_CLOSED.md`).
3. **V5 v1.1-surface behavioral DDC** — a *manually-reconciled behavioral* audit; its witness is
   gitignored + not clean-checkout reproducible (`K_DDC_BROADENED.md` honest-scope caveat). The
   byte-identical, hash-pinned, one-command DDC is the separate seed→K1 `ddc_crosscheck.sh` (leg 2).
4. **Path portability** — the fixpoint layer's `assemble_k1.hx` hardcodes the canonical path; a
   noncanonical checkout needs the documented path rewrite (`CLEAN_REPRODUCTION.md` "Where it walls").
