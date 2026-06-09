# Helix — Trusted-C Inventory (v1.3 item V6)

**Purpose.** A precise, verified inventory of **every** committed `.c`/`.h` in the repo:
its path, LOC, role, whether it is **in or out of the self-host fixpoint**, why it is trusted,
whether it is **on a build path or dead**, and whether it is **portable or irreducible**. This is
the V6 deliverable of `docs/HELIX_V1_3.md` §1 ("shrink the trusted-C surface"). It **extends**
(does not duplicate) the existing trust docs:

- `docs/TRUST_CHAIN_CLOSED.md` — the verified trust-chain record + the 7 honest residuals.
- `docs/K3_TRUSTED_SEED_SCOPING.md` — scoping of the trusted seed (the irreducible root).
- `docs/SEED_DDC_CROSSCHECK.md` — the gcc-vs-M2-Planet diverse-double-compile of the seed rung.
- `stage0/MESCC_TOOLS_PROVENANCE.md` / `stage0/M2-Planet/PROVENANCE.md` — vendored-source provenance.

Every claim below was verified against the **live tree** (`git ls-files "*.c" "*.h"`) and the
**actual build scripts** (which files are truly `-f`'d / `gcc`'d), not assumed.

---

## 0. Headline

- **Committed C/H after V6: 24 files, 15 605 LOC** (was 30 files / 16 308 LOC; V6 pruned 6 dead
  files / 708 LOC — see §4).
- **Post-v1.3 (GPT-2 inference demos) addendum:** at HEAD the committed C/H is **29 files / 19 158 LOC**.
  The v1.3 V6 trusted toolchain detailed below (24 files / 15 605 LOC — Category A's from-raw ladder
  and the `seed.c` trust root) is **UNCHANGED**, and the self-host fixpoint stays `0992dddd`. The
  GPT-2-on-Helix demos added **five Category-B host tools** — `helixc/runtime/gpt2_infer.c`
  (a CUDA-FFI forward-only launcher like `train_transformer.c`, outside the self-host fixpoint,
  ptxas-boundary; it now also carries the **ADDITIVE, forward-only `--serve` mode** for the live chat
  demo: a 4th `main()` branch that does `device_init`/`alloc_buffers`/`setup_head` ONCE then loops on
  stdin request frames running the **unchanged** `forward_full`, plus a tiny printf-only telemetry emit
  module whose hooks read only host-scope values — the numeric path is **byte-identical** to `--generate`
  and the fixpoint is structurally untouched, proven token-for-token by `scripts/helix_serve_gate.sh` G1),
  `helixc/runtime/cpu_host.c` (579 LOC, the CPU **no-ptxas** demo launcher — a
  CUDA-FREE byte-movement harness, outside the self-host fixpoint, **ZERO arithmetic on the trust path**;
  all math lives in the kovc-compiled `helixc/runtime/gpt2_cpu_ops.hx`, which is a `.hx` and so does NOT
  count against the `.c`/`.h` fence), the two **Python-free-data-path** offline tools —
  `helixc/runtime/gpt2_tok.c` (byte-level BPE tokenizer: encode/decode, hand-written GPT-2
  pretokenizer, no regex lib; byte↔id bookkeeping only, **ZERO arithmetic on the trust path**; it now
  also links into the `--serve` worker via `GPT2_TOK_LIB` — its four entrypoints are exposed and the
  pure-bookkeeping `decode_one`/`decode_range` helpers added — so the live server tokenizes in-process,
  Python-free) and `helixc/runtime/gpt2_pack.c` (345 LOC, safetensors→`.weights` importer: byte-movement
  only, **ZERO arithmetic on the trust path**) — and the **NEW** `helixc/runtime/gpt2_serve_http.c`
  (549 LOC, the dependency-light, **NO-Python** C HTTP+SSE server for the live chat demo: POSIX sockets +
  libc only, no third-party deps; serves `demo/` static files [GET `/`→`index.html`, `/dashboard.html`,
  assets with correct MIME, rejects `..` traversal] + `GET /api/health` + the `POST /api/generate`
  `text/event-stream` bridge that spawns ONE persistent `gpt2_infer --serve` worker over two pipes and
  re-frames each worker JSON line as one SSE event [single-flight: a concurrent generation gets 409] +
  an honest `POST /api/verify` that degrades to `UNAVAILABLE` and never fakes a verdict; **HTTP/byte-pump
  only, ZERO arithmetic on the trust path, OUTSIDE the self-host fixpoint** — `gate_kovc.sh` never
  compiles it — exactly the classification of `cpu_host.c`/`gpt2_tok.c`/`gpt2_pack.c`) — and grew
  `cuda_launch.c` by 273 LOC (GPU kernel verify modes).
  So Category B is **7 files** at HEAD (was 6; from the original 2 / 2 388). Nothing in Category A or the
  trust root changed. The tokenizer's Unicode `\p{L}`/`\p{N}`/`\s` range tables are a generated DATA file,
  `helixc/runtime/gpt2_unicode_ranges.inc` — a `.inc`, NOT a `.c`/`.h`, so outside the fence (like a `.hx`).
- The trusted-C surface is **two disjoint categories**:
  - **A. The from-raw bootstrap ladder** (`stage0/*`, 22 files / 13 217 LOC) — trusted **source**
    that is **compiled from raw** by the `hex0 → hex1 → hex2 → catm → M0 → cc_amd64 → M2-Planet`
    chain. **Not** trusted *binaries*. The **seed** (`seed.c`, 1368 LOC) is the **single
    irreducible trust root**; the rest are the vendored bootstrap compilers/assemblers/linker that
    carry the chain. Independently corroborated by gcc-DDC (`SEED_DDC_CROSSCHECK.md`) and the
    self-host fixpoint.
  - **B. The runtime/GPU harness** (`helixc/runtime/`, 2 files / 2 388 LOC) — `cuda_launch.c`
    (1923) + `train_transformer.c` (465). **Outside** the self-host fixpoint. The genuinely
    reducible-looking target — but every line is either a **CUDA driver-API host call** (a closed
    C-FFI boundary Helix cannot cross) or a CPU oracle that **independently verifies** the
    Helix-emitted kernels (and so must *not* be written in Helix).

- **What V6 changed:** a real, honest **shrink** — 6 dead duplicate `.c`/`.h` (708 LOC) removed
  (§4). The 3 `.c` are byte-identical to the single canonical `M2-Planet/M2libc/bootstrappable.c`;
  the 3 `.h` are identical to each other and there is **no** canonical `bootstrappable.h` anywhere
  (no build ever needed one). Proven safe by **rebuilding all 3 mescc-tools (M1, blood-elf,
  hex2-linker) from their `build.sh` with the 6 files removed** — each built a valid ELF and passed
  its in-script capability self-check (§4) — plus the full main gate re-run GREEN after the prune
  (§4). **No harness port was undertaken** (none is cheap + gateable; see §5).

- **What V6 did NOT change (honest, by design):** `seed.c` is untouched (irreducible root); the
  closed-`ptxas`/closed-driver GPU boundary (`TRUST_CHAIN_CLOSED.md` residual #7) **STANDS** —
  pruning/porting harness C does not and cannot close it.

---

## 1. Category A — the from-raw bootstrap ladder (`stage0/*`)

These are trusted **source**, each compiled by the **prior rung** (no pre-built binary trusted).
The build wiring is: `M2-Planet/build.sh` builds `M2.bin` from `cc_amd64`; then `M1`, `blood-elf`,
`hex2-linker`, and `helixc-bootstrap` (the seed) are each built by `M2.bin` — and **all four of
those use `../M2-Planet/M2libc/`** for the bootstrap libc (verified: `MLIB=../M2-Planet/M2libc` in
each `build.sh`). M2 runs with `--expand-includes` off, so every `#include` in these sources is a
**no-op**; the real translation unit is the explicit ordered list of `-f` files in each script.

| Path | LOC | Role | Fixpoint | On build path? | Trusted-why | Portable? |
|---|---:|---|---|---|---|---|
| `stage0/helixc-bootstrap/seed.c` | 1368 | **The seed** — our Apache-2.0 C-subset compiler; rung 8; compiles `kovc` (the Helix compiler) | root of it | **YES** — `helixc-bootstrap/build.sh -f seed.c` | **IRREDUCIBLE trust ROOT.** Built from raw by the ladder; corroborated by gcc-DDC (byte-identical K1) + the self-host fixpoint | **NO — irreducible.** Any change must re-pass the full gate incl gcc-DDC + fixpoint; **out of V6 scope** |
| `stage0/M2-Planet/M2-Planet/cc.c` | 397 | M2-Planet driver `main()` | n/a (builds the seed) | **YES** — `M2-Planet/build.sh` | Vendored GPL-3.0 @ pinned commit; compiled from raw (cc_amd64 → M2) | reducible only by replacing the whole vendored compiler — not a V6 target |
| `stage0/M2-Planet/M2-Planet/cc_core.c` | 3485 | M2-Planet core codegen | n/a | **YES** | as above | as above |
| `stage0/M2-Planet/M2-Planet/cc_emit.c` | 1372 | M2-Planet emit | n/a | **YES** | as above | as above |
| `stage0/M2-Planet/M2-Planet/cc_macro.c` | 1145 | M2-Planet preprocessor/macros | n/a | **YES** | as above | as above |
| `stage0/M2-Planet/M2-Planet/cc_types.c` | 947 | M2-Planet types | n/a | **YES** | as above | as above |
| `stage0/M2-Planet/M2-Planet/cc_reader.c` | 482 | M2-Planet reader/lexer | n/a | **YES** | as above | as above |
| `stage0/M2-Planet/M2-Planet/cc_strings.c` | 226 | M2-Planet string pool | n/a | **YES** | as above | as above |
| `stage0/M2-Planet/M2-Planet/cc.h` | 184 | M2-Planet shared header | n/a | **YES** — `-f`'d as a unit | as above | as above |
| `stage0/M2-Planet/M2-Planet/cc_globals.c` | 73 | M2-Planet globals | n/a | **YES** | as above | as above |
| `stage0/M2-Planet/M2libc/bootstrap.c` | 275 | M2libc syscalls/stdio (FILE*, fopen, fgetc, malloc…) | n/a | **YES** — `-f`'d by M2-Planet/M1/blood-elf/hex2/seed builds | Vendored GPL-3.0/public-domain M2libc @ pinned commit; from raw | the bootstrap libc — irreducible at this layer |
| `stage0/M2-Planet/M2libc/bootstrappable.c` | 200 | M2libc helpers (require/match/in_set/strtoint/int2str) | n/a | **YES** — **the one canonical copy**; `-f`'d by **all** tool builds | as above | as above |
| `stage0/M2-Planet/M2libc/amd64/linux/bootstrap.c` | 78 | amd64/linux syscall stubs (read/write/open/close/exit) | n/a | **YES** | as above | as above |
| `stage0/M1/M1.c` | 933 | the mescc-tools **M1 macro assembler** (= upstream `M1-macro.c`) | n/a (assembles M2's large self-output) | **YES** — `M1/build.sh -f M1.c` | Vendored GPL-3.0 mescc-tools @ pinned commit; built by M2 (from raw) | aux verification tool for the M2 rung — not a V6 target |
| `stage0/M1/stringify.c` | 104 | `stringify()`/`LittleEndian()` for M1 (no header) | n/a | **YES** — `M1/build.sh -f stringify.c` | as above | as above |
| `stage0/blood-elf/blood-elf.c` | 580 | the mescc-tools **blood-elf** debug-symbol footer generator | n/a | **YES** — `blood-elf/build.sh -f blood-elf.c` | as above | as above |
| `stage0/blood-elf/stringify.c` | 104 | `stringify()`/`LittleEndian()` for blood-elf | n/a | **YES** — `blood-elf/build.sh -f stringify.c` | as above (byte-identical to `M1/stringify.c`, but each tool's own unit) | as above |
| `stage0/hex2-linker/hex2_linker.c` | 583 | flag-driven **hex2 linker** core (defines globals + link passes) | n/a | **YES** — `hex2-linker/build.sh -f hex2_linker.c` | as above | as above |
| `stage0/hex2-linker/hex2_word.c` | 370 | hex2 word/shift-register output passes | n/a | **YES** — `-f hex2_word.c` | as above | as above |
| `stage0/hex2-linker/hex2.c` | 204 | hex2 `main()` + CLI flag parsing | n/a | **YES** — `-f hex2.c` | as above | as above |
| `stage0/hex2-linker/hex2.h` | 57 | hex2 constants/struct header | n/a | **provenance source only** — the build **re-expresses** its `#define` block as a generated `hex2_defs.m2.h` enum (M2 can't expand object-like value-macros under `--bootstrap-mode`); `#include`d by `hex2_globals.h` but that is a no-op under M2 | as above. **KEPT** in V6: it is the documented byte-for-byte upstream origin the generated header is mechanically derived from | kept for provenance fidelity |
| `stage0/hex2-linker/hex2_globals.h` | 50 | hex2 prototypes + extern globals | n/a | **YES** — `-f hex2_globals.h` (forward decls inside the one combined TU) | as above | as above |

**Category A total: 22 files, 13 217 LOC** (1 seed + 9 M2-Planet `cc*` + 3 M2-Planet/M2libc + 2 M1
+ 2 blood-elf + 5 hex2). Verified: `git ls-files "stage0/*.c" "stage0/*.h" | xargs wc -l` → 13 217.
Note there is **no** committed `bootstrappable.h` after V6 — the M2-Planet M2libc subset never
vendored one (the build needs none).

> **The seed is the only irreducible root.** Everything else in Category A is a vendored bootstrap
> compiler/assembler/linker built **from raw** by the prior rung — they exist so the chain can
> reach the seed and so M2-Planet has a self-host path (`MESCC_TOOLS_PROVENANCE.md` "H6"). The
> trust does not rest on any of their *binaries*; it rests on (a) `hex0`'s 299 hand-authored bytes
> and (b) the seed source, both re-derivable from raw, with the seed independently DDC-corroborated.

---

## 2. Category B — the runtime/GPU harness (`helixc/runtime/`)

**Outside the self-host fixpoint** (the gate, `scripts/gate_kovc.sh`, never compiles these — it
runs the pure-x86 `seed → K1 → K2 → K3 → K4` fixpoint + a ptxas-free PTX byte-diff + the feature
corpus). The two **v1.3 V6** files below are compiled **only** by the GPU/capstone scripts
(`scripts/gpu_*.sh`, `scripts/capstone_audit.sh`, and the `.stage33-logs/_g3_*`/`m6_*` probes) via
`gcc … -lcuda -lcublas -lm`. (The four **post-v1.3** Category-B host tools — `helixc/runtime/gpt2_infer.c`,
`helixc/runtime/cpu_host.c`, `helixc/runtime/gpt2_tok.c` and `helixc/runtime/gpt2_pack.c` — are itemized
in the Headline addendum above and §2a; all are outside the fixpoint, and the last three are **CUDA-FREE**
with **zero arithmetic on the trust path**.)

| Path | LOC | Role | Fixpoint | On build path? | Trusted-why | Portable? |
|---|---:|---|---|---|---|---|
| `helixc/runtime/cuda_launch.c` | 1923 | Multi-mode **GPU correctness + perf harness**: loads a kovc-emitted PTX module and drives vector_add / attention / GEMM / TF32-Tensor-Core kernels; times them (cuEvent); checks each against a **CPU oracle** and **cuBLAS**. | **OUT** | GPU scripts only (`gcc -lcuda -lcublas`), **never** in `gate_kovc.sh` | Trusted-once **host** launcher; the math it judges is all kovc-emitted PTX | **IRREDUCIBLE** as a host launcher (see §3) |
| `helixc/runtime/train_transformer.c` | 465 | The **capstone training-loop host**: a 2-layer transformer trained end-to-end on kovc-emitted GPU kernels; gradient check = a **sampled finite-difference spot-check** (6 gradient tensors × ≤5 sampled indices each vs analytic backprop — `verify` mode, NOT exhaustive); 2% loss-parity vs an independent numpy oracle. | **OUT** | capstone/`m6_*` scripts only (`gcc -lcuda`), **never** in `gate_kovc.sh` | Trusted-once **host** launcher; all math is kovc-emitted PTX | **IRREDUCIBLE** as a host launcher (see §3) |

**Category B (v1.3 V6) total: 2 files, 2 388 LOC.** At HEAD, with the four post-v1.3 demo host tools
(`gpt2_infer.c` 667 + `cpu_host.c` 579 + `gpt2_tok.c` 659 + `gpt2_pack.c` 345) and `cuda_launch.c`'s
+273 growth, **Category B = 6 files / 4 911 LOC** (see Headline addendum + §2a).

### 2a. Post-v1.3 Category-B addendum — the GPT-2 demo host tools

Four GPT-2-on-Helix demo host tools were added after v1.3. All are **outside the self-host fixpoint**
(`gate_kovc.sh` never compiles them) and **trusted-once host tools** — none performs any arithmetic on
the compute-trust path. The first two are the forward-pass launchers (the model's arithmetic is all
kovc-emitted); the last two are the **offline data-path** tools that make the demo's PRODUCTION path
**Python-free** (they do only string↔token-id bookkeeping and byte-movement — the demo's trust claim is
the exact token-id sequence + the from-raw toolchain that executes it, NOT this host-side rendering;
see `docs/HELIX_GPT2_DEMO_RUNBOOK.md` residual #1).

| Path | LOC | Role | Fixpoint | On build path? | Trusted-why | Portable? |
|---|---:|---|---|---|---|---|
| `helixc/runtime/gpt2_infer.c` | 667 | **GPU** forward-only GPT-2 demo launcher (CUDA-FFI), the `train_transformer.c` twin minus the training loop. | **OUT** | GPU demo scripts only (`gcc -lcuda`), **never** in `gate_kovc.sh` | Trusted-once **host** launcher; all math is kovc-emitted PTX | **IRREDUCIBLE** as a host launcher (closed `ptxas`/driver boundary, §3) |
| `helixc/runtime/cpu_host.c` | 579 | **CPU no-ptxas** forward-only GPT-2 demo launcher: a **CUDA-FREE** byte-movement harness (mmap the P1 weights, host embedding gather, multi-head pack/scatter, GEMM N-tiling, per-op `/tmp/gpc` file staging into the 25 MB Helix arena). **ZERO arithmetic on the trust path** — every layernorm/softmax/matmul/GELU/residual runs inside the kovc-compiled `gpt2_cpu_ops.hx` ELF. Gated by `scripts/gpt2_cpu_parity.sh` (block-0 parity vs the numpy oracle, fail-closed). | **OUT** | `scripts/gpt2_cpu_parity.sh` only (`gcc -O2 -lm`, **no** `-lcuda`), **never** in `gate_kovc.sh` | Trusted-once **host** launcher that does **no math**; all arithmetic is in the kovc-from-raw Helix ELF | **IRREDUCIBLE** as a host launcher (mmap/file-staging glue), but does **NOT** rely on `ptxas`/driver — it is CUDA-free |
| `helixc/runtime/gpt2_tok.c` | 659 | **Offline byte-level BPE tokenizer** (encode + decode): the 256-entry byte↔unicode map, the GPT-2 pretokenization split **hand-written (NO regex lib)**, merges.txt rank-ordered merges, vocab.json id map. Replaces the Python tokenizer on the demo's production path. **ZERO arithmetic on the trust path** (string↔token-id bookkeeping only). Gated by `scripts/gpt2_pyfree.sh` (bit-exact encode/decode parity vs the Python oracle + the pinned demo prompt + the hero decode, fail-closed). | **OUT** | `scripts/gpt2_pyfree.sh` only (`gcc -O2`), **never** in `gate_kovc.sh` | Trusted-once **host** tool that does **no math**; the trust claim is the token-id sequence + the from-raw toolchain, not the host string rendering — and parity is bit-exact-gated vs the independent Python oracle | **IRREDUCIBLE** as a host tool (no compute-trust role); Helix-native is blocked today (1 MB read-buffer ud2 trap < the ~1.04 MB vocab.json; no regex) |
| `helixc/runtime/gpt2_pack.c` | 345 | **Offline safetensors→`.weights` importer**: parse the JSON header (hand-rolled), stream the F32 tensors in the same `build_order` as `gpt2_import.py`, UN-transposed, write the 64-byte HXGW header + flat fp32 body. Replaces the Python importer on the demo's production path. **ZERO arithmetic on the trust path** (byte-movement only; streamed to keep RAM bounded). Gated by `scripts/gpt2_pyfree.sh` (output sha256 **byte-identical** to the Python importer's `.weights`, fail-closed). | **OUT** | `scripts/gpt2_pyfree.sh` only (`gcc -O2`), **never** in `gate_kovc.sh` | Trusted-once **host** tool that does **no math**; byte-identity to the independent Python importer is sha-gated | **IRREDUCIBLE** as a host tool (no compute-trust role) |

> Companion source: `helixc/runtime/gpt2_cpu_ops.hx` (the pure-Helix op ELF holding **all** CPU-path
> arithmetic) is a `.hx`, so it is **not** counted by the `.c`/`.h` fence — freely committable, no fence
> impact. Design doc: `helixc/runtime/README_CPU_PATH.md`. Likewise the tokenizer's Unicode
> `\p{L}`/`\p{N}`/`\s` range tables live in `helixc/runtime/gpt2_unicode_ranges.inc` — a generated DATA
> `.inc` (bit-exact with Python's `regex`), NOT a `.c`/`.h`, so **no fence impact** (like a `.hx`).

---

## 3. Why Category B is irreducible (the CUDA-driver C-FFI boundary)

The harness is the genuinely *reducible-looking* C — but it is **irreducible as a host launcher**,
for three independent reasons, each verified by reading the source:

1. **It IS the closed-driver C-FFI boundary.** Almost every line is a CUDA **driver-API** call —
   `cuInit`, `cuDeviceGet`, `cuCtxCreate`, `cuModuleLoadData`, `cuModuleGetFunction`,
   `cuLaunchKernel`, `cuMemAlloc`, `cuMemcpyHtoD`/`DtoH`, `cuCtxSynchronize`, `cuEventElapsedTime`,
   plus `cublasGemmEx`. **Helix emits PTX but cannot call `libcuda`** (there is no Helix→C FFI to
   the driver). A "port" would have to re-expose every one of these as a Helix builtin — i.e.
   *move* the C-FFI boundary, not remove it. This is exactly residual #7 of
   `TRUST_CHAIN_CLOSED.md`: hand-auditable `hex0 → PTX`, then NVIDIA's **closed `ptxas` + driver**.
   Porting the host launcher to Helix **does not close that boundary**.

2. **The CPU oracles must stay C — porting them would defeat their purpose.** The small pure-CPU
   helpers in the harness (`cuda_launch.c`: `ln_rowL`, `attn_forward_cpu`, the GEMM/TF32 CPU
   references; `train_transformer.c`: `ce_loss`, the finite-difference verifier) exist to
   **independently verify the kovc-emitted kernels**. An independent oracle that checks the
   language under test **must not be written in that same language** (the same principle that keeps
   the numpy oracle fenced "by design for independent verification" — `TRUST_CHAIN_CLOSED.md`).
   Rewriting them in Helix would make them no longer independent.

3. **No cheap, gateable port exists.** Even a tiny pure helper (e.g. the xorshift weight init, or
   `ce_loss`) is **load-bearing for the audited capstone**: `train_transformer.c` writes the exact
   init weights to `init_weights.bin`, which the numpy oracle reads to prove the 2% loss-parity.
   Splitting any such helper into a separately-compiled Helix object linked into the C launcher
   would (a) create a **new, un-gated, fragile link path**, and (b) require a **full capstone
   re-train** (a multi-minute GPU run) to re-validate the byte-for-byte audit — explicitly the
   **high-risk large harness port the V6 charter says NOT to undertake**. The honest call is to
   **inventory it as irreducible** and leave it.

---

## 4. What V6 pruned — the honest shrink (DEAD vendored C)

**6 files / 708 LOC removed** — dead duplicates, **provably off every build path**. The 3 `.c`
are **byte-identical** (SHA-256 `efe16699…`) to the single canonical
`stage0/M2-Planet/M2libc/bootstrappable.c` — which is the only `bootstrappable.c` that survives in
the tree. The 3 `.h` are **byte-identical to each other** (SHA-256 `81b0e0e9…`); there is **no**
canonical `bootstrappable.h` in the repo (the M2-Planet M2libc subset never vendored one), so the
`.h` was a duplicate of nothing the build needs — its only consumers are the `#include` lines in
`M1.c`/`blood-elf.c`/`hex2.h`, which are no-ops under M2 (`--expand-includes` off). After the prune,
`git ls-files "*bootstrappable.h"` returns nothing and all 3 tools still build (§ below) — which
proves no `bootstrappable.h` was ever needed.

| Removed | LOC | What it duplicated | Evidence it was dead |
|---|---:|---|---|
| `stage0/M1/M2libc/bootstrappable.c` | 200 | canonical `M2-Planet/M2libc/bootstrappable.c` (SHA-256 `efe16699…`) | not `-f`'d by any script; the only `#include` is a no-op under M2 |
| `stage0/M1/M2libc/bootstrappable.h` | 36 | the other deleted `.h` (SHA-256 `81b0e0e9…`); no canonical `.h` exists | `#include`'d only (no-op under M2); never `-f`'d as a unit |
| `stage0/blood-elf/M2libc/bootstrappable.c` | 200 | same canonical `.c` | not on any build path |
| `stage0/blood-elf/M2libc/bootstrappable.h` | 36 | same `81b0e0e9…` `.h` | `#include`'d only; never a `-f` unit |
| `stage0/hex2-linker/M2libc/bootstrappable.c` | 200 | same canonical `.c` | not on any build path |
| `stage0/hex2-linker/M2libc/bootstrappable.h` | 36 | same `81b0e0e9…` `.h` | `#include`'d only; never a `-f` unit |

**How "dead" was proven (three independent static ways):**
1. **Current scripts:** all three mescc-tool `build.sh` (`M1`, `blood-elf`, `hex2-linker`) set
   `MLIB=../M2-Planet/M2libc` and `-f "$MLIB/bootstrappable.c"` — they reach for the **M2-Planet**
   copy, never the local one. Grep of every tool source + `*.sh` confirms the only references to the
   per-tool-local copies are the `#include "M2libc/bootstrappable.h"` lines (M1.c:23, blood-elf.c:25,
   hex2.h:25), which are no-ops under M2 (`--expand-includes` off; M1/build.sh:18 documents this).
2. **Full git history:** `git log -S "M1/M2libc/bootstrappable" -- "*.sh"` (and the blood-elf /
   hex2-linker equivalents) is **empty** — the local copies were **never** wired into any build/test
   script at any commit.
3. **No per-tool `run_tests.sh`** exists for M1/blood-elf/hex2-linker that could compile them.

**Proof the prune is safe (verified by build, 2026-06-04):** with the 6 files removed from the
working tree, all 3 mescc-tools were rebuilt **serially from their `build.sh`**; each compiled a
valid x86-64 ELF and **passed its in-script capability self-check** (the scripts `set -euo pipefail`
and `exit` nonzero on any failure — all three exited 0). Durable logs:
`.stage33-logs/v6_{M1,blood-elf,hex2-linker}_build.txt`.

| Tool | Rebuilt ELF | bytes | In-script capability check (passed) |
|---|---|---:|---|
| `M1.bin` | ELF 64-bit LSB exec | 32822 | assembles `amd64_defs.M1`+`libc-core.M1` via flag-driven CLI → 181-byte hex2 |
| `blood-elf.bin` | ELF 64-bit LSB exec | 25023 | emits a 2704-byte debug footer (`--64 --little-endian --entry -f -o`) |
| `hex2.bin` | ELF 64-bit LSB exec | 36666 | links a runnable ELF (`--base-address/--architecture/--little-endian`) that exits 0 |

(These are the genuine post-prune build outputs; the prune was not separately byte-diffed against a
pre-prune build, but removing files that no build path reads cannot change any artifact, and the
green capability self-checks confirm the tools are fully functional without the 6 files.) **Standalone
buildability is preserved**: each tool builds against the byte-identical canonical
`M2-Planet/M2libc/bootstrappable.c`, which is what the scripts already did.
`stage0/MESCC_TOOLS_PROVENANCE.md` was updated to record the removal + the build-against-M2-Planet
fact. The full universal gate (`scripts/gate_kovc.sh`) was re-run **GREEN** after the prune (the
verdict sha is recorded in the V6 commit message / `.stage33-logs/hxcc_state.txt`): self-host
fixpoint `K2==K3==K4` byte-identical + GPU-PTX byte-diff + the feature corpus, none of which touch
the pruned files.

> **Note — no `.py` added, no `kovc.hx` touched.** The V6 prune removes only dead vendored C +
> updates docs. The Python fence is unchanged: exactly **1** committed `.py`
> (`verification/oracle/oracle_train.py`).

---

## 5. What V6 did NOT do, and why (honest scope)

- **No harness port.** §3 establishes there is no small, clearly-portable, gateable pure-logic
  helper in `cuda_launch.c` / `train_transformer.c`: the file IS the CUDA-driver C-FFI host, the
  pure helpers are independent oracles that must stay non-Helix, and any split would need a full
  capstone re-train (high-risk, out of scope per the charter). The reducible-looking surface is
  irreducible **as a host launcher**.
- **`seed.c` untouched.** The irreducible trust root. Any edit must re-pass gcc-DDC + the fixpoint;
  high-risk, out of V6 scope. Its **existence** (a small, from-raw, DDC-corroborated C-subset
  compiler) is the trust anchor — **not** a flaw.

---

## 6. The precise trusted-C boundary (the bottom line)

After V6, the trusted C is **24 files / 15 605 LOC**, and the boundary is:

- **The seed (`seed.c`, 1368 LOC) is the single irreducible trust ROOT** — compiled from raw,
  independently DDC-corroborated (gcc vs M2-Planet → byte-identical K1) + self-host-fixpoint-stable.
- **The rest of Category A (the vendored bootstrap ladder, ~11 849 LOC) is COMPILED-FROM-RAW**, not
  trusted binaries — each rung built only by the prior rung from `hex0`'s 299 hand-authored bytes.
- **Category B (the harness, 2 388 LOC) is the CUDA-driver C-FFI host**, outside the self-host
  fixpoint, irreducible as a launcher; below PTX it relies on NVIDIA's **closed `ptxas` + driver**
  (`TRUST_CHAIN_CLOSED.md` residual #7 — **STANDS**; V6 does not and cannot close it).

The apparent surface shrank honestly by 708 LOC of dead duplicate C, with zero loss of unique
trusted content — proven by rebuilding all 3 mescc-tools green (ELF + capability self-check) without
the 6 files, plus the full main gate GREEN after the prune.

---

## v1.3 V6 record

- **Inventory:** all 30→24 committed `.c`/`.h` precisely classified (path / LOC / role / in-or-out
  of the fixpoint / on-build-path-or-dead / trusted-why / portable-or-irreducible), each claim
  verified against the live tree **and** the build scripts (not guessed).
- **Pruned:** 6 dead duplicate `M2libc/bootstrappable.{c,h}` (708 LOC) from `M1/`, `blood-elf/`,
  `hex2-linker/` (3 `.c` byte-identical to canonical `M2-Planet/M2libc/bootstrappable.c`; 3 `.h`
  identical to each other, no canonical `.h` exists). Proven dead (current scripts + full git
  history + no run_tests + the only references are no-op `#include`s under M2). Proven safe: all 3
  mescc-tools rebuild GREEN from `build.sh` without the files (valid ELF + in-script capability
  self-check, each exit 0); the full main gate GREEN after the prune.
- **Ported:** nothing — no cheap, gateable harness port exists (§3, §5); the high-risk large port
  was correctly **not** undertaken.
- **Irreducible remainder + why:** `seed.c` (root) + the from-raw vendored ladder (compiled from
  raw) + the CUDA-driver harness (closed C-FFI boundary). The closed-`ptxas`/driver GPU residual
  STANDS.
- **Fence:** `git ls-files "*.py"` == **1** (`verification/oracle/oracle_train.py`).
