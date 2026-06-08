# Appendix A — Glossary

*What this appendix covers: an alphabetical reference of every load-bearing term *Helix: The
Complete Guide* uses — the compiler and language names (`kovc`, `seed`, Helix, `.hx`), the rungs of
the from-raw ladder (`hex0` … `M2-Planet`), the trust anchors (the self-host fixpoint, `K1`–`K4`,
the gcc-DDC, trusting-trust, the TCB), the verification machinery (the gate, the feature corpus,
`check_err`, the capstone), the GPU vocabulary (PTX, SASS, `ptxas`), the language internals
(monomorphization, turbofish, the arena), and the two operator gotchas (the byte-count exit,
DrvFs/ext4). Each entry is one to three sentences, consistent with how the book actually uses the
term. It is a lookup table, not a narrative — for the full treatment of any term it cross-references
the chapter that owns it.*

This appendix is bound by the same terminology table the whole book is
([Style Guide §3](../STYLE_GUIDE.md)): the left-column terms there are authoritative, and the
definitions below match them. Where a term has an honest boundary (the GPU path, the DDC's reach),
the boundary is stated here too and cited to [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md),
the honesty ceiling for the book. Where this page and a repo source disagree, the source wins and
this appendix is the bug.

> **For AI agents:** these terms map to real files and real output tokens, not to English
> descriptions. When you script against Helix, key off the exact strings an entry cites (e.g.
> `GATE_PASS`, `DDC_ANCHOR_OK`, the pinned hashes `9837db12…` / `84363adb…` / `0992dddd…`) and
> dereference the linked path before acting. The authoritative term table is
> [Style Guide §3](../STYLE_GUIDE.md); the authoritative trust ceiling is
> [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R.

---

## A

**the arena.** The single global, append-only memory model that Helix's collections and tensors
are built on: one contiguous run of `i32` slots, BSS-zeroed at load and grown only by appending. A
"value" living in the arena is named by an integer **start index** (plus a length or shape), and the
four arena builtins `__arena_len` / `__arena_get` / `__arena_set` / `__arena_push` are the
foundation the gate proves directly. See
[Tensors, collections & I/O §1](../part4-stdlib/03-tensors-collections-io.md#1-the-arena-memory-model-and-the-honest-status-of-these-modules).

## B

**byte-count exit (the byte-count exit convention).** The fact that `kovc` and the self-hosted
`K`-binaries (and the seed-minted PTX driver) **return their output byte-count as the process exit
status** — a single 8-bit value, i.e. `bytes mod 256`. The 698 392-byte self-compile therefore
exits with status **24** (`698392 mod 256 = 24`) *on success*, so a non-zero exit from a `kovc`
self-compile leg is **expected, not a failure**; those legs are judged by non-empty output + the
pinned SHA, never by `rc == 0`. Only the C-compiled legs (`seed → K1`, the mint, the launcher, the
oracle) assert `rc == 0`. See
[Troubleshooting §1](../part2-setup-build/05-troubleshooting.md) and
[Part IX — Traps](../part9-for-ai-agents/03-traps.md).

## C

**the capstone.** The real-capability proof at the top of the chain: a ≥2-layer transformer trained
**end-to-end on `kovc`-emitted GPU (PTX) kernels**, whose loss curve matches an *independent* numpy
oracle to within the 2% bar (reproduced at ~0%, worst-case `0.00000876`). It exercises the whole
substrate at once — the full language, GPU execution, autodiff numerics, the tensor stack, and the
Python-free loop — and is verified on a CUDA host by
[`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh), separately from the gate. See
[What Helix is §"The high-level shape"](../part1-orientation/01-what-is-helix.md) and
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §1.

**`catm`.** Rung 4 of the from-raw ladder: a tiny file **concatenator** (it replaces `cat` / shell
redirection in the build), built by `hex2`. Its committed binary is 299 bytes (`911d19bf…`) — the
same *size* as `hex0` but a different program. See
[The MESCC-lineage rungs to seed](../part6-bootstrap/02-mescc-rungs-to-seed.md) and
[Appendix C §C.2](C-pinned-hashes.md#c2-the-from-raw-ladder-rung-hashes).

**`cc_amd64`.** Rung 6 of the from-raw ladder: a **minimal C-subset compiler** (C → `M1` assembly),
built by `M0` + `catm` + `hex2`. Committed binary `ea0054d1…`, 17 976 bytes. See
[The MESCC-lineage rungs to seed](../part6-bootstrap/02-mescc-rungs-to-seed.md).

**`check_err` (the negative diagnostics).** The gate's negative-corpus pass: **four** malformed
programs that must each (1) make `K2` exit non-zero at compile time, (2) write **no** output ELF, and
(3) emit a `path:line:col: parse error` diagnostic at the *exact* hand-computed line and column. The
gate driver calls them with `chk_err <fixture> <line> <col>` and prints
`CHECK_ERR: 4 passed, 0 failed` on success; any miss fails the gate
([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4b]`). It is the proof that `kovc`
*rejects* bad input precisely, not just that it accepts good input. See
[Front end §"check_err"](../part5-compiler/01-front-end.md).

## D

**diverse double-compile (DDC).** The general technique behind the **gcc-DDC**: compile the same
source with two compilers of *independent lineage* and require a byte-identical result, so a backdoor
present in one lineage but not the other is exposed. A DDC catches only a divergent backdoor; it says
nothing about anything **both** compilers share (see *TCB*, *trusting-trust*). See *gcc-DDC* and
[Appendix F §F.2–F.3](F-tcb.md#f2-the-shared-tcb--what-no-ddc-can-retire).

**DrvFs / 9p.** The Windows-mounted filesystem WSL exposes under `/mnt/c/...`. Because `seed`/`kovc`
do byte-at-a-time file I/O, a build on DrvFs pays a 9p round-trip per syscall and runs roughly **75×
slower** than on native **ext4** — the build's *output* is identical, only its wall-time changes.
See *ext4* and [Troubleshooting §2](../part2-setup-build/05-troubleshooting.md).

## E

**ext4.** The WSL-native Linux filesystem. The fix for the DrvFs/9p slowdown is to mirror the
committed tree onto WSL-native **ext4** and build there (keeping the `/mnt/c` checkout for git);
the v1.3 reproduction did exactly this and produced the *same* fixpoint `0992dddd…`. See *DrvFs / 9p*
and [Troubleshooting §2](../part2-setup-build/05-troubleshooting.md).

## F

**the feature corpus.** The **109-program** suite of small Helix programs the gate compiles **and
runs** with the freshly self-hosted `K2`, checking each one's exit code against an expected value via
the `chk` helper (the gate prints `CORPUS: 109 passed, 0 failed`). It is assembled from three places
— committed programs in [`helixc/examples/`](../../../helixc/examples/), programs generated inline by
the gate at run time, and committed fixtures under
[`stage0/helixc-bootstrap/corpus_gen/`](../../../stage0/helixc-bootstrap/corpus_gen/) — so there is no
single `corpus/` folder holding all 109. It is the standing compile-and-run proof for the language
surface. See [Appendix D §D.4](D-file-directory-map.md#d4-scripts--the-gate-and-the-reproduction-drivers)
and [Appendix E §E.1](E-example-index.md#e1-how-to-read-this-index-and-what-gate-asserted-means).

**the from-raw ladder.** The hand-typed-root build chain
`hex0 → hex1 → hex2 → catm → M0 → cc_amd64 → M2-Planet → seed → kovc`, in which **each rung is built
only by the rung before it** and there is no trusted pre-built compiler anywhere. The committed rung
binaries are reference copies; the one-command reproduction deletes them and rebuilds from raw, each
rung self-verifying its committed `.sha256`. See
[Build from raw](../part2-setup-build/02-build-from-raw.md),
[The MESCC-lineage rungs to seed](../part6-bootstrap/02-mescc-rungs-to-seed.md), and
[Appendix C §C.2](C-pinned-hashes.md#c2-the-from-raw-ladder-rung-hashes).

## G

**the gate.** [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh), the universal gate every
compiler change must pass: the self-host fixpoint + the 109-program feature corpus + the PTX **text**
regression + the `check_err` negative diagnostics. It prints the literal token `GATE_PASS` and exits
`0` on success; its four sub-anchors are the verbatim lines
`FIXPOINT OK (K2==K3==K4 byte-identical AND == pinned known-good)`, `GPU PTX REGRESSION OK`,
`CORPUS: 109 passed, 0 failed`, and `CHECK_ERR: 4 passed, 0 failed`. See
[Reproduce & verify the trust chain](../part2-setup-build/04-reproduce-verify-trust.md).

**gcc-DDC.** The `gcc` **diverse-double-compile** of the `seed → K1` rung: `gcc` (an independent
compiler lineage with **zero M2-Planet ancestry**) and the from-raw `seed` both compile `k1src.hx`
into a **byte-identical** `K1` (pinned `84363adb…`), a Wheeler defense against a trusting-trust
attack on that one rung. `gcc` is used **only as an auditor**, never as the shipped root; the runner
([`stage0/helixc-bootstrap/ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh))
prints `DDC_ANCHOR_OK`. The byte-identical DDC covers the `seed → K1` surface only — the broader
language surface is cross-checked behaviorally, not byte-identically
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §1, §2). See
[Appendix F §F.3](F-tcb.md#f3-what-the-gcc-ddc-narrows--and-what-it-does-not).

## H

**Helix.** The language (capital **H**): a statically typed, from-raw-binary, self-hosting language
with GPU code generation, oriented toward AGI and high-certainty computing — integer/float widths
down to `bf16`/`f16`, structs, enums, `match`, generics, traits, closures, source-level autodiff
(`grad`), and a tile/tensor surface for GPU kernels. Its whole reason for existing is auditable
trust. See [What Helix is](../part1-orientation/01-what-is-helix.md).

**`helixc`.** The **historical** Python-hosted frontend that originally bootstrapped the language. It
is **not** in the shipped compile/run path and was removed from the toolchain when `kovc` took over —
when the book means the real compiler it always says `kovc`. (Not to be confused with the directory
`helixc/`, which holds the Helix-native compiler sources, stdlib, and examples.) See
[What Helix is §"Python-free"](../part1-orientation/01-what-is-helix.md).

**`helixrt`.** The Helix runtime. See [Style Guide §3](../STYLE_GUIDE.md).

**`hex0`.** The bottom rung and **root of trust**: a **299-byte, hand-authored** raw-binary program
(typed as hex in `stage0/hex0/hex0.hex`, turned into the binary with the audit-only `xxd -r -p`, no
assembler), pinned at `cc1d1741…` and frozen. It is the only rung whose bytes are not produced by
another program; nothing below it is trusted. See
[hex0 and the raw-binary root](../part6-bootstrap/01-hex0-raw-root.md) and
[Appendix C §C.2](C-pinned-hashes.md#c2-the-from-raw-ladder-rung-hashes).

**`hex1` / `hex2`.** Rungs 2 and 3 of the ladder. `hex1` (built by `hex0`; `c264a212…`, 622 B) adds
single-character labels; `hex2` (built by `hex1`; `6c69c7e6…`, 1 519 B) adds long labels and absolute
addresses — effectively a linker. See
[The MESCC-lineage rungs to seed](../part6-bootstrap/02-mescc-rungs-to-seed.md).

**`.hx`.** The Helix source-file extension. The 98 example programs under
[`helixc/examples/`](../../../helixc/examples/) and the 21 stdlib modules under
[`helixc/stdlib/`](../../../helixc/stdlib/) are all `.hx` files. See
[Appendix E](E-example-index.md).

## K

**K1 / K2 / K3 / K4.** The four stages of the self-host fixpoint. `K1` is `kovc` as built by the
`seed`; `K2` is `kovc` compiling its own source with `K1`; `K3` and `K4` repeat the step. Reaching
**K2 == K3 == K4 byte-for-byte** (pinned `0992dddd…`) is the fixpoint — the same convergence test a
self-hosted C compiler uses (stage2 == stage3). (`K1` separately is the `seed → K1` artifact the
gcc-DDC pins at `84363adb…`.) See *the self-host fixpoint* and
[Appendix C §C.1](C-pinned-hashes.md#c1-the-three-release-anchors).

**`kovc`.** The Helix compiler — the from-scratch compiler **written in Helix**
([`helixc/bootstrap/{lexer,parser,kovc}.hx`](../../../helixc/bootstrap/)) that emits x86-64 ELF
directly (no external assembler or linker) and also emits the GPU PTX. It is the shipped,
self-hosting compiler the whole capability claim rests on; written lowercase and code-formatted.
Contrast *`helixc`* (the retired Python frontend). See
[What Helix is](../part1-orientation/01-what-is-helix.md) and
[Front end](../part5-compiler/01-front-end.md).

## M

**`M0`.** Rung 5 of the from-raw ladder: a **macro assembler** (`M1` assembly → `hex2` input), built
by `catm` + `hex2`. Committed binary `db97dff1…`, 1 684 B. See
[The MESCC-lineage rungs to seed](../part6-bootstrap/02-mescc-rungs-to-seed.md).

**`M2-Planet`.** Rung 7 and the **last vendored rung**: a full, self-hosting C compiler (built by
`cc_amd64` + `catm` + `M0` + `hex2`) capable of building the `seed`; committed binary `724b9e2d…`,
200 561 B. It ships alongside its `M2libc`. See
[The MESCC-lineage rungs to seed](../part6-bootstrap/02-mescc-rungs-to-seed.md) and
[Appendix D §D.3](D-file-directory-map.md#d3-stage0--the-from-raw-ladder).

**MESCC.** The bootstrap lineage the lower rungs (`hex1 … M2-Planet`) are **vendored** from — the
community-audited `mescc-tools` / M2-Planet family that climbs from raw bytes to a C-subset compiler.
The vendored provenance is recorded in
[`stage0/MESCC_TOOLS_PROVENANCE.md`](../../../stage0/MESCC_TOOLS_PROVENANCE.md). The ladder uses this
lineage *only* up to `M2-Planet`; from the `seed` upward the code is Helix's own. See
[The MESCC-lineage rungs to seed](../part6-bootstrap/02-mescc-rungs-to-seed.md).

**monomorphization.** How `kovc` makes generics emit **real per-type machine code**: for each
distinct concrete type a generic is instantiated with, it synthesises a separate, fully-typed copy of
the function (or struct method) and emits that — no type erasure, no boxing, no runtime type tag. It
is a parser-side pass that runs before codegen. What ships is precisely **turbofish-directed**
monomorphization (see *turbofish*); inferring a non-`i32` type argument *without* a turbofish is the
named residual. See
[Generics, traits & closures](../part3-language/04-generics-traits-closures.md).

## P

**PTX.** NVIDIA's **Parallel Thread Execution** virtual ISA — the assembly-like text `kovc` emits for
the GPU, with no LLVM in the path. **"Complete to PTX"** is the precise capability claim for the GPU
side: the hand-auditable from-raw chain ends at PTX *text*, byte-verified by the gate against
committed `.ref.ptx` references — never "complete to GPU machine code." See *SASS*, *`ptxas`*, and
[Appendix F §F.4](F-tcb.md#f4-the-gpu-side-tcb--complete-to-ptx-not-to-gpu-machine-code).

**`ptxas`.** NVIDIA's **closed** PTX → SASS assembler. It sits *below* Helix's from-raw boundary: the
chain is hand-auditable from `hex0` to PTX, and `ptxas` (plus the CUDA driver, the GPU hardware, and
the C host launcher) is **trusted-once** past PTX. The gate's GPU leg is a pure-text PTX regression
that invokes **no** `ptxas` and needs no GPU; `ptxas` is exercised only by the separate capstone
audit on the reference `sm_86` box. See *PTX*, *SASS*, and
[Appendix F §F.4](F-tcb.md#f4-the-gpu-side-tcb--complete-to-ptx-not-to-gpu-machine-code).

## S

**SASS.** NVIDIA's actual GPU **machine code**, produced from PTX by the closed `ptxas`. Helix does
**not** reach SASS from raw — the precise, honest phrasing throughout the book is **PTX-not-SASS**:
below PTX, correctness rests on `ptxas`, the driver, and the silicon, none of which Helix authored or
audited. See *PTX*, *`ptxas`*, and
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 residual 8.

**the seed (`seed`).** The project's own Apache-2.0 **C-subset compiler** (source `seed.c`) — rung 8,
the last rung of the from-raw ladder and the **first original artifact** in the chain. It is built by
`M2-Planet` + `catm` + `M0` + `hex2`, pinned at `9837db12…` (62 467 B), and it is what builds `kovc`.
Call it the `seed`, never "the bootstrap compiler." See
[seed to kovc: the self-host fixpoint](../part6-bootstrap/03-seed-to-kovc-fixpoint.md) and
[Appendix C §C.1](C-pinned-hashes.md#c1-the-three-release-anchors).

**`seed.c`.** The **human-readable source** of the `seed` — an Apache-2.0 C-subset compiler of about
1,368 lines at
[`stage0/helixc-bootstrap/seed.c`](../../../stage0/helixc-bootstrap/seed.c). It is committed (the
produced `seed.bin` is gitignored and must be re-derived from raw), and it is part of the **shared
TCB**: auditable one line at a time, but *trusted-by-reading*, not proven — a backdoor living
identically in `seed.c`'s visible source would be invisible to the DDC by construction. See *TCB* and
[Appendix F §F.2](F-tcb.md#f2-the-shared-tcb--what-no-ddc-can-retire).

**the self-host fixpoint.** `seed → K1 → K2 → K3 → K4` with **K2 == K3 == K4 byte-identical** (pinned
`0992dddd…`) — machine-checkable proof that `kovc`'s shipped binary faithfully corresponds to its
source, with no drift. Say "the self-host fixpoint," not "stage2==stage3." See *K1 / K2 / K3 / K4*,
[Trust at a glance](../part1-orientation/04-trust-at-a-glance.md), and
[Appendix C §C.1](C-pinned-hashes.md#c1-the-three-release-anchors).

## T

**TCB (trusted computing base).** Everything Helix still **trusts** after the trust chain is closed —
what closure deliberately leaves outside the audited-from-raw spine. The book splits it in two: the
**shared TCB** beneath both DDC compilers (host OS/kernel, filesystem, shell + coreutils,
`gcc`/libc/binutils/loader, CPU + microcode, RAM, and the `seed.c` source — *no* DDC retires it), and
the **GPU-side TCB past PTX** (the closed `ptxas`, the CUDA driver, the GPU hardware, and the C host
launcher). It is the difference between "closed" and "absolute." See
[Appendix F](F-tcb.md) and [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R.

**trusting-trust.** Ken Thompson's 1984 attack (*Reflections on Trusting Trust*): a compiler can be
made to insert a backdoor into the programs it compiles — including into future copies of itself — so
that the backdoor survives even though it appears in **no** readable source. It is the threat Helix's
from-raw root, self-host fixpoint, and gcc-DDC are built to answer. A diverse double-compile is the
direct defense, but only against a backdoor that *diverges* between the two lineages (see *TCB*). See
[What Helix is §"Why Helix exists"](../part1-orientation/01-what-is-helix.md).

**turbofish.** The explicit type-argument syntax `::<…>` (also `::[…]`) at a generic use site —
`id::<i32>(…)`, `Box::<f32>{…}`, `Opt::<i32>::Some(…)`. It directs monomorphization (names which
concrete instance you mean) and disambiguates the type argument. In shipping `kovc` a bare non-`i32`
scalar generic defaults `T → i32`, so the turbofish is **required** whenever the concrete type is not
`i32`. See *monomorphization* and
[Generics, traits & closures §"Turbofish and disambiguation"](../part3-language/04-generics-traits-closures.md#turbofish-and-disambiguation).

## V

**`v1.3-release`.** The git tag this book documents, **declared trust-chain-closed on 2026-06-07**.
"Closed," in the project's narrow sense, means the from-raw chain is reproducible from a hand-typed
root, self-hosting, defended against a trusting-trust attack at the `seed → K1` rung, demonstrably
capable, and Python-free — with the residuals disclosed in full. (The numerically-higher `v2.0.0`–
`v3.1.0` tags are a **superseded MLIR exploration line**, not the current head; trust the `v1.3`
line.) See
[What Helix is §"What `v1.3-release` means"](../part1-orientation/01-what-is-helix.md) and
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md).

---

## A note on the honest boundaries baked into these terms

Three definitions above carry the book's standing residuals, and they are worth restating in one
place because over-reading any one of them is the easiest way to overclaim
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R):

- **PTX, not SASS.** "Complete to PTX" is the GPU claim; below PTX the closed `ptxas` + driver +
  hardware + C launcher are trusted-once, on a **single** target (`sm_86`). GPU kernel performance is
  a **fraction of cuBLAS** (~50–67.5% on the reference box), and the end-to-end capstone speedup is
  **7.0–8.7×** (Amdahl-bound), **not** ≥10×. The load-bearing capstone number is **loss parity ≈0%**
  (correctness), not the speedup.
- **The gcc-DDC covers `seed → K1` only.** It is byte-identical and hash-pinned for that one rung; the
  broader v1.1 language surface is cross-checked **behaviorally** against a second, zero-lineage
  interpreter (not byte-identical, and its witness is out-of-tree / not clean-checkout reproducible).
- **The shared TCB stands.** No DDC retires the OS, kernel, filesystem, shell/coreutils,
  `gcc`/libc/binutils/loader, CPU + microcode, RAM, or the `seed.c` source. **External third-party
  reproduction on independent hardware remains the one open increment.**

> **For AI agents:** never assert *less* trust than this appendix lists (do not drop a residual) and
> never assert *more* than [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R
> supports. Do not write "beats cuBLAS," "complete to GPU machine code," "fully verified GPU," or
> "AGI" as an achieved state. When in doubt, undersell and cite the residual.

---

**Next:** [Appendix B — Command reference](B-command-reference.md) — the exact commands for building
from raw, running the gate, reproducing the trust chain, and the gcc-DDC, each quoted verbatim from its
real script. The authoritative command sources it distills are
[`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh),
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh), and
[Part IX — Recipes](../part9-for-ai-agents/04-recipes.md).
