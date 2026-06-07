# Helix — Current-Head Audit Packet (committed proof extract)

**Purpose.** A self-contained, **committed** record of the v1.3 trust results at the current head —
exact commands, pinned hashes, environment, and verbatim verdict lines — so a clean reader can see
them **without** relying on the gitignored process logs under `.stage33-logs/`. Companion to
`docs/CLEAN_REPRODUCTION.md` (method) and `docs/TRUST_CHAIN_CLOSED.md` (record).

> This is **our process evidence committed to the tree**, NOT an external/independent reproduction.
> An independent operator running a clean clone + publishing logs is the open residual that moves
> confidence past ~0.9 (see Residuals below). Every number here is reproducible by the commands shown.

- **Head:** the v1.3 final-convergence-pass line on `main` (verify the live tip: `git rev-parse HEAD`).
  The trust results below are **byte-stable across these final-pass commits** — `kovc.hx` + `seed.c` are
  unchanged, so the fixpoint/K1/seed SHAs do not move; only docs + verification wrappers changed.
- **Date:** 2026-06-06.
- **Environment:** WSL2 (Linux 6.6 WSL2, x86_64); NVIDIA RTX 3070 Laptop GPU (sm_86), driver 596.21 /
  CUDA 13.2 runtime + 12.x `ptxas`; `gcc` (gnu89) as the independent DDC lineage. Build executed on a
  WSL-native **ext4 mirror** of the committed tree for speed (DrvFs per-syscall latency, ~75x); output
  is byte-identical (same fixpoint + driver SHA) — see `CLEAN_REPRODUCTION.md` "Where it walls".

## Static fence (committed-tree facts)

| Check | Command | Result |
|-------|---------|--------|
| Exactly 1 committed `.py` | `git ls-files "*.py" \| wc -l` | **1** — `verification/oracle/oracle_train.py` |
| 24 committed `.c`/`.h`, 15 605 LOC | `git ls-files "*.c" "*.h" \| wc -l` ; `\| xargs wc -l` | **24 / 15 605** |
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

## Push-button reproduction (v1.3 Path A) — anyone can re-run this

The whole trust core above is reproducible by **one committed command on a clean checkout**:
`bash scripts/reproduce_trust.sh` — it deletes every pre-built rung binary, rebuilds the entire
`hex0 → seed` ladder (each rung self-verifying its `.sha256`), runs the self-host fixpoint + the gcc
diverse-double-compile, and asserts the three pinned anchors above, exiting nonzero on any mismatch.
It applies the disclosed `/mnt/c → checkout` path rewrite automatically, so it runs at any path.
**Verified PASS on a fresh clean clone, CPU-only, ~1 min.** `.github/workflows/trust-reproduce.yml`
runs it on a **clean GitHub `ubuntu-latest` runner** (a different machine, fresh clone, zero local
state) on every push/PR + weekly — so the byte-identical trust core is reproducible push-button by any
third party who forks the repo or runs the script locally. (The GPU capstone stays a separate CUDA-host
step, `scripts/capstone_audit.sh`.)

## Honest residuals (status after v1.3 Path A)

1. **Fully-independent THIRD-PARTY reproduction** — a clean-clone reproduction now exists **committed +
   push-button** (`scripts/reproduce_trust.sh` + the `trust-reproduce.yml` CI on a clean different-machine
   runner; see above). What remains is a run by an operator who is *not the author* (a genuine outside
   party / lab) — that final increment is the last step past ~0.9. The mechanism for it is now in place:
   anyone can fork the repo or clone it and run the one command.
2. **Shared TCB** — OS / kernel / filesystem / shell / coreutils / gcc / libc / binutils / loader /
   CPU+microcode / RAM, and the audited `seed.c` source, remain trusted (`TRUST_CHAIN_CLOSED.md`).
3. **V5 v1.1-surface behavioral DDC** — a *manually-reconciled behavioral* audit; its witness is
   gitignored + not clean-checkout reproducible (`K_DDC_BROADENED.md` honest-scope caveat). The
   byte-identical, hash-pinned, one-command DDC is the separate seed→K1 `ddc_crosscheck.sh` (leg 2).
4. **Path portability** — the fixpoint layer's `assemble_k1.hx` hardcodes the canonical path; a
   noncanonical checkout needs the documented path rewrite — now applied **automatically** by
   `scripts/reproduce_trust.sh` step [0], so the CI + any clone build at their own path. A native
   parameterization of the concatenator remains a possible future cleanup.
5. **Tracked stage0 rung binaries are REFERENCE artifacts** — the committed `stage0/*/*.bin`
   (hex0…M2-Planet) are *convenience/reference* copies. "No trusted pre-built binary" means each rung
   must be **rebuilt from source and compared** to its committed `.bin`/`.sha256` (the ladder rebuild,
   `CLEAN_REPRODUCTION.md` Step 2 / `stage0/<rung>/build.sh`). Trust rests on that rebuild, not on the
   committed binaries. **RESOLVED push-button:** `scripts/reproduce_trust.sh` deletes all pre-built
   rung binaries first and rebuilds the whole ladder from `hex0`, and `trust-reproduce.yml` runs that on
   a clean runner — so rebuild-and-compare is now one command, not a manual ritual.
