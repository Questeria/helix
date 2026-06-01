# Helix v1.0 — De-Language Plan (Criterion #6: Python-free / no other language)

**Goal (DoD #6 + user hard constraint).** v1.0 DoD #6 requires the **live toolchain**
(compiler, build, test runner) to be **Helix/seed/ladder only — zero `.py`**, reproducible
from hex0. The user's broader constraint is stricter: **NO Python OR OTHER LANGUAGE** may
remain in the project at v1.0 (so `.c`, `.sh` also count, eventually). This plan inventories
the non-Helix in the live tree and sequences its removal.

> **✅ EXECUTED 2026-06-01 — Python purge complete (criterion #6 GREEN).** Removed the
> 93-file `HELIX_STAGE30_COMPILER_SNAPSHOT/` (archived pre-self-host Python compiler;
> preserved in tag `v0-pre-k4-full-with-python`), the 4 dead `helixc.*`-importing dev
> scripts (`proof_artifact_{gate,key,validate}`, `mlir_audit_canaries`), the dev status
> reporter (`helix_status.py`), and the 2 non-load-bearing hex0 audit aids (`encode.py`,
> `hex0_reference.py`; `hex0.bin` SHA byte-identical after). The **only** `.py` remaining
> is the numpy oracle, relocated to a fenced `verification/oracle/` (decision 4, an
> independent verification reference outside the Helix tree). The toolchain self-hosts
> **byte-identically Python-free** (post-purge gate: fixpoint `96c440d3` K2==K3==K4 + GPU
> PTX byte-identical + corpus 35/35). Remaining non-Helix is by DECLARED trusted-tool
> boundary: the C GPU launcher (decision 3) and shell build-orchestration — a Helix
> concatenator/test-runner is post-v1.0 (#13). See `HELIX_V1_DEFINITION_OF_DONE.md`,
> "v1.0 SCOPE DECISIONS".

## Trust-root boundary (EXEMPT — intentionally not Helix)

The from-raw-binary trust root is the bootstrap seed; you cannot bootstrap Helix from Helix.
EXEMPT (documented, permanent): `stage0/hex0..hex2`, `stage0/catm`, `stage0/M0`, `stage0/M1`,
`stage0/cc_amd64`, `stage0/M2-Planet` + `M2libc` (vendored GPL), and **`stage0/helixc-bootstrap/seed.c`**
(the Apache-2.0 seed compiler: built by the ladder, mints K1, after which K1+ are pure Helix).
`HELIX_STAGE30_COMPILER_SNAPSHOT/` is an archived pre-K4 snapshot (not live) — also out of scope.

## Inventory (live tree, non-trust-root) — 2026-06-01

| File | Lang | Role | Critical? | Target |
|------|------|------|-----------|--------|
| `stage0/helixc-bootstrap/assemble_k1.py` | py | source concatenator (mints k1*.hx) | **YES** | port → shell, then Helix |
| `stage0/helixc-bootstrap/ddc_battery.py` | py | DDC vs the DELETED Python compiler | no (dead post-K4) | **DELETE** |
| `stage0/helixc-bootstrap/ddc_check.py` | py | DDC wheel check vs deleted Python | no (dead) | **DELETE** |
| `scripts/selfhost_cascade.py` | py | multi-gen self-host via deleted Python backend | no (dead) | **DELETE** |
| `scripts/selfhost_cascade_validate.py` | py | cascade report validator | no (dead) | **DELETE** |
| `scripts/stage33_selfhost_gate.py` | py | orchestrates the dead cascade/DDC | no (dead) | **DELETE** |
| `scripts/helix_status.py` | py | release-progress / status reporter | no (dev) | delete (dev-only) |
| `scripts/mlir_audit_canaries.py` | py | v3+ proof-loop canaries | no (v3 infra) | defer (v3, not v1.0) |
| `scripts/proof_artifact_{gate,key,validate}.py` | py | v3+ proof-artifact infra | no (v3 infra) | defer (v3) |
| `helixc/runtime/oracle_train.py` | py (numpy) | FENCED-OFFLINE capstone audit oracle (D1) | audit-only | **USER DECISION** (keep as documented offline-audit exception, or port to Helix numeric — huge) |
| `helixc/runtime/cuda_launch.c` | C | GPU first-light launcher (CUDA Driver API) | YES (#3) | **BLOCKED on Helix FFI** |
| `helixc/runtime/train_transformer.c` | C | capstone GPU training launcher (CUDA Driver API) | YES (capstone) | **BLOCKED on Helix FFI** |
| `scripts/*.sh` (gate_kovc, feature_corpus, selfhost_fixpoint_rawbinary, gpu_corpus, run_all_tests, build.sh, run_tests.sh) | sh | build/test/gate harnesses | YES | interim-exception → Helix test-runner (#7/#13) |

## De-language phases (easiest first)

- **P1 — DELETE dead Python (this campaign).** `ddc_battery.py`, `ddc_check.py`,
  `selfhost_cascade.py`, `selfhost_cascade_validate.py`, `stage33_selfhost_gate.py` all import
  the K4-deleted Python compiler (`helixc.backend`/`helixc.frontend`) — they cannot run; the live
  proof is `scripts/selfhost_fixpoint_rawbinary.sh` (Python-free). Zero build impact. **Done first.**
  Then the dev-only `helix_status.py`.
- **P2 — ✅ DONE (2026-06-01)**: ported `assemble_k1.py` → `assemble_k1.sh`, **byte-identical** output (sha-verified) + gated (fixpoint `96c440d3` unchanged, corpus 17/17); all 4 callers rewired; `.py` deleted. The live toolchain is now Python-free. (Final form: a Helix concatenator.) Original note — (the one *used* build-helper Python: a pure concatenator —
  `lexer.hx + parser.hx + kovc.hx` minus the `// Demo:` tail + a driver `main`, written to
  `k1src.hx`/`k1input.hx`/`k1ptxdrv.hx`). Port → a shell `assemble_k1.sh` (interim), then a Helix
  concatenator (read_file_to_arena + scan the SEP marker + write_file_to_arena, like
  `selfhost_bytecmp.hx`). Rewire `gate_kovc.sh`/`feature_corpus.sh`/`selfhost_fixpoint_rawbinary.sh`.
  Then the **live toolchain is Python-free** (modulo the oracle + C launchers).
- **P3 — the BIG blocker: Helix FFI for the C CUDA launchers.** `cuda_launch.c` +
  `train_transformer.c` call the CUDA Driver API in `libcuda.so` (a dynamic library). Helix today
  emits **static, syscall-only, single-PT_LOAD ELFs with NO dynamic linker and NO FFI/extern**
  (verified: no `extern`/`dlopen`/`libcuda`/FFI anywhere in kovc.hx/lexer.hx/parser.hx). To make
  the launchers Helix, Helix needs an `extern "C"` / dynamic-linking mechanism (ELF relocations,
  GOT/PLT, dynamic-linker entry) — est. **~3-4 weeks**. **USER DECISION (D2):** implement Helix FFI,
  OR accept the trusted C launcher as a **documented v1.0 exception** (like the from-raw-binary
  ladder + `ptxas` — a trusted tool outside the self-host fixpoint).
- **P4 — port the `.sh` harnesses to a Helix test-runner** (overlaps #7/#13). Shell is "another
  language"; the final test-runner should be Helix (`run_process`/`set_exec` builtins exist).
  Interim: the `.sh` gates are a documented exception.

## The two v1.0-blocking USER DECISIONS (do not block grinding the rest)

1. **CUDA FFI vs documented C-launcher exception** (P3). The biggest remaining #6 item.
2. **The numpy capstone oracle** (`oracle_train.py`): keep as a fenced-offline audit reference
   (documented exception, like the ladder), or port to Helix numeric (huge)? It is NOT in the
   training loop or the live compiler/build/test path — it is an independent verification reference.

## Honest #6 status

K4 deleted the Python *compiler* (the hardest part). Remaining: P1 (dead Python, trivial),
P2 (assemble_k1, easy), P3 (FFI / C launchers — the real blocker, user decision), P4 (.sh →
Helix test-runner). The trust root (ladder + seed.c) is permanently exempt. With the documented
exceptions (C launchers + oracle accepted like the ladder), #6 is reachable after P1+P2; without
them (full FFI port), it is a multi-week effort.
