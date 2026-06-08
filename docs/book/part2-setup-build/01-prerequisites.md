# Prerequisites & environment

*What this chapter covers:* the machine, operating system, and small set of command-line
tools you need to build Helix from raw and run the produced programs — what is required, what
is optional (the GPU capstone only), what is **not** needed at all (no Python for the
toolchain), and the one filesystem choice on Windows that changes a build from seconds to
half an hour.

Helix's whole point is honest, reproducible trust (see
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md)), and the environment is
the first place that trust is either earned or quietly broken. A reproduction is only
meaningful if the tools doing the reproducing are ordinary, named, and auditable. So this
chapter is deliberately minimal: the authoritative tool list is the one the project's own CI
installs on a clean machine before it rebuilds the entire chain from a 299-byte hand-authored
root.

---

## 1. The machine: x86-64 Linux, or WSL2 on Windows

The from-raw ladder produces and runs **x86-64 Linux ELF** binaries. The root of the chain,
`hex0`, is a hand-encoded x86-64 Linux ELF you can audit one byte at a time
([`stage0/hex0/`](../../../stage0/hex0/)); every rung above it is an x86-64 Linux executable
built only by the rung below. So you need an x86-64 Linux environment, in one of two forms:

- **Native x86-64 Linux** — any reasonably current distribution. The project's reference CI
  runs on `ubuntu-latest` (see
  [`.github/workflows/trust-reproduce.yml`](../../../.github/workflows/trust-reproduce.yml)).
- **WSL2 on Windows** — the Windows Subsystem for Linux, version 2, which is a real Linux
  kernel. This is the project's primary development environment; the clean-reproduction
  record was produced "under WSL2 (Linux 6.6.114.1-microsoft-standard-WSL2, x86_64)"
  ([`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md)).

The [`QUICKSTART.md`](../../../QUICKSTART.md) states the same requirement plainly:

> **WSL2 + Linux** on Windows, or any Linux (for the from-raw build + running the produced ELFs)

> **Note:** "x86-64" is not incidental — the entire CPU trust spine is x86-64-specific, from
> the hand-authored `hex0` bytes through the `seed` to `kovc`'s ELF back end. There is no
> cross-architecture build. (The GPU side targets a single NVIDIA GPU architecture, `sm_86`;
> that residual is covered in §5 and in
> [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 residual 6.)

> **For AI agents:** if you are on Windows, run **every** build and gate command *inside* WSL,
> not in PowerShell or `cmd`. The scripts are `bash` scripts that call Linux tools and produce
> Linux ELFs. The one Windows-side exception is pushing commits to GitHub — see §4.

---

## 2. The authoritative tool list (from the CI workflow)

The single source of truth for "what must be installed" is the one step in the reference CI
that provisions a clean runner. [`.github/workflows/trust-reproduce.yml`](../../../.github/workflows/trust-reproduce.yml)
installs exactly this, and then runs the whole from-raw reproduction on top of it:

```yaml
      - name: Install audit tools (xxd / binutils / gcc / file)
        run: |
          sudo apt-get update
          sudo apt-get install -y xxd binutils gcc file coreutils
```

That `apt-get install` line is the **authoritative minimal set** for reproducing the trust
core. On a Debian/Ubuntu system you can run it verbatim:

```bash
sudo apt-get update
sudo apt-get install -y xxd binutils gcc file coreutils
```

Plus `bash` itself, which is already present on any Linux/WSL2 system (the scripts require
`bash` 4+; see the header of
[`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh)). Here is what each piece
is for, grounded in where it is actually used:

| Tool (apt package) | Provides | Used for |
|--------------------|----------|----------|
| `bash` (≥ 4) | the shell | every `scripts/*.sh` and `stage0/<rung>/build.sh` |
| `coreutils` | `sha256sum`, `stat`, `cp`, `rm`, `cmp`, `tr`, `cut`, … | hashing every rung against its committed `.sha256`; byte-comparing the self-host fixpoint; driving and checking builds |
| `xxd` | hex↔binary conversion | turning the hand-authored `hex0.hex` into the raw `hex0.bin` ELF — `xxd -r -p`, "an audit-only tool — no assembler involved" ([`stage0/hex0/build.sh`](../../../stage0/hex0/build.sh)) |
| `binutils` | `objdump` (and the binutils suite) | disassembling rung binaries for inspection, e.g. `objdump -D -b binary -m i386:x86-64` in [`stage0/hex0/build.sh`](../../../stage0/hex0/build.sh) |
| `file` | file-type identification | ELF sanity checks — every rung's `build.sh` asserts `file "$OUT" \| grep -q "ELF 64-bit LSB executable"` |
| `gcc` | the GNU C compiler | the **diverse-double-compile auditor** only — it builds the `seed` a second, independent way to defend against a trusting-trust attack (see §3 and Part VIII). `gcc` is an *auditor*, never the shipped root. |

The `reproduce_trust.sh` header lists the same dependencies in prose, which is a useful
cross-check:

> Tools required: bash 4+, coreutils (sha256sum/stat), xxd, file, objdump (binutils), gcc, grep, sed.

`grep` and `sed` are part of the same base userland and are present on every Linux/WSL2
system; the path-rewrite step of `reproduce_trust.sh` (§4) uses them.

> **For AI agents:** treat the `apt-get install -y xxd binutils gcc file coreutils` line from
> [`.github/workflows/trust-reproduce.yml`](../../../.github/workflows/trust-reproduce.yml) as
> the canonical dependency manifest for the CPU trust core. Do not add packages it does not
> list, and do not assume a package is present just because a tool name is common — if a
> `command -v` check is cheap, run it before relying on a tool.

### The CPU core needs nothing else

That is the complete list for the part of Helix that is reproduced on a clean machine: the
from-raw ladder, the self-host fixpoint, and the `gcc` diverse-double-compile. No assembler
(`as`, `nasm`), no linker beyond what the ladder builds itself, and — importantly — no Python.
See §6.

---

## 3. For the GPU capstone: a CUDA toolkit and an NVIDIA GPU

Everything in §2 is enough to reproduce the **CPU trust spine**. The **capstone** — a
≥2-layer transformer trained end-to-end on `kovc`-emitted GPU (PTX) kernels — is the one part
that needs real GPU hardware. It is verified **separately**, on a CUDA host, not in the
CPU-only CI. The CI workflow says so directly:

> CPU-only by design: GitHub-hosted runners have no GPU, so the transformer CAPSTONE (perf + 2%
> oracle parity) is NOT run here -- it is verified on a CUDA host via scripts/capstone_audit.sh

For the capstone you additionally need:

- **An NVIDIA GPU.** The reference hardware is an **RTX 3070 Laptop GPU (`sm_86`)**
  ([`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md)). This is the single
  validated target — see the residual in §5.
- **A CUDA toolkit**, providing `ptxas` (the PTX→SASS assembler) and the CUDA headers and
  driver library that the C host launcher links against. The capstone audit script invokes
  the toolkit at the canonical CUDA install path. From
  [`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh), the launcher is built
  with:

```bash
gcc train_transformer.c -O2 -I/usr/local/cuda/include -L/usr/lib/wsl/lib -lcuda -lm -o /tmp/train
```

  and the gate pins the PTX assembler at `${PTXAS:-/usr/local/cuda/bin/ptxas}`
  ([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh)). The reference reproduction ran
  with "`/usr/local/cuda` (RTX 3070)", driver 596.21 and a CUDA-12.8 `ptxas`
  ([`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md)).

The [`QUICKSTART.md`](../../../QUICKSTART.md) summarizes the same split — the auditor `gcc`
is always needed, the CUDA stack only for the capstone:

> **gcc** (the diverse-double-compile auditor) + a CUDA toolchain & RTX-class GPU (for the capstone)

> **Residual:** the chain is **complete to PTX, not to GPU machine code.** `kovc` emits PTX,
> byte-verified against committed references; below PTX, NVIDIA's closed `ptxas`, the CUDA
> driver / `libcuda`, the GPU hardware, and the C host launcher are **trusted, not reproduced
> from raw**. The CPU path is all-the-way-down from the hand-authored root; the GPU path is
> from-`hex0`-to-PTX-then-`ptxas`. See
> [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 residuals 7–8. Do not
> describe Helix as "complete to GPU machine code."

> **For AI agents:** the GPU and CPU verifications are different jobs with different
> requirements. Reproducing the CPU trust core (`bash scripts/reproduce_trust.sh`) needs **no
> GPU**; only the capstone (`bash scripts/capstone_audit.sh`) needs a CUDA GPU. On a machine
> without an NVIDIA GPU, run the CPU core and report the capstone as *not attempted here*, not
> as failed.

---

## 4. The one environment lesson that matters most: build on ext4, not `/mnt/c`

This is the single most consequential setup choice on Windows, and it is a *performance* trap,
not a correctness one. The bytes you produce are identical either way; the **wall-clock time**
differs by roughly **75×**.

WSL2 exposes your Windows drives under `/mnt/c` via a network-style filesystem (DrvFs / 9p).
Helix's `seed` and `kovc` do their I/O the simplest possible way: millions of tiny,
byte-at-a-time reads and writes. On a WSL-native **ext4** filesystem those are cheap; on
`/mnt/c` each one pays a 9p round-trip. [`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md)
records the measured effect directly:

> the mirror exists only to avoid `/mnt/c` DrvFs per-syscall latency, which had inflated build
> wall-time ~75×

In practical terms: an assemble step that takes about **7.5 seconds** on ext4 takes about
**15 minutes** on `/mnt/c`; the full gate that finishes in well under a minute of compute on
ext4 can stretch toward **half an hour**. The fix is to mirror the working tree onto the
WSL-native filesystem and build *there*:

- Keep the git checkout wherever you like (a `/mnt/c` checkout is fine for editing and for
  committing), but **copy the tree to an ext4 path under your WSL home** (e.g. `~/helix`) and
  run the build and gate from that copy.
- The reproduction record confirms this mirror is purely about speed and changes nothing
  about the result: building on "a WSL-native ext4 mirror of the current-head tree
  (byte-identical committed sources)" produces "the **same** fixpoint `0992dddd…` and the same
  seed-minted PTX driver" ([`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md)).

Two related Windows-environment gotchas follow from this split checkout, both grounded in the
reproduction record:

**Path rewrite for a non-canonical checkout.** A few build inputs — notably the fixpoint
concatenator `assemble_k1.hx` — hardcode the canonical absolute path
`/mnt/c/Projects/Kovostov-Native`. A checkout at any other path (including your ext4 mirror)
must rewrite that path first. You do **not** do this by hand: step `[0]` of
[`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) performs it automatically
as "a pure mechanical path swap":

```bash
say "[0] path rewrite  /mnt/c/Projects/Kovostov-Native -> $ROOT"
mapfile -t HCFILES < <(grep -rlI '/mnt/c/Projects/Kovostov-Native' . 2>/dev/null || true)
if [ "${#HCFILES[@]}" -gt 0 ]; then
  printf '%s\n' "${HCFILES[@]}" | xargs sed -i "s#/mnt/c/Projects/Kovostov-Native#$ROOT#g"
  say "    rewrote ${#HCFILES[@]} file(s)"
fi
```

This is the disclosed portability caveat ("Where it walls" in
[`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md)): the from-raw ladder uses
relative `../` paths and is fully portable, but the self-host fixpoint layer currently assumes
the canonical path until that rewrite runs. It is build hygiene, not a trust gap — the bytes
produced are identical. The mechanics of the build itself are covered in the next chapter.

**Pushing commits: use the host (Windows) git, not WSL git.** WSL's git has no GitHub
credentials configured, so a `git push` from inside WSL will hang waiting for an auth it cannot
provide. Push from the Windows side, where the Git Credential Manager is set up. The standard
flow is therefore: build and verify in WSL on ext4; commit and push from the `/mnt/c` checkout
via host git.

**CRLF/LF line-ending churn.** With `core.autocrlf` enabled on Windows, many files can show as
modified on a line-ending basis only, which clutters `git status` and risks staging unrelated
EOL changes. Stage only the files you intend to change, and verify the *content* diffstat while
ignoring carriage-return-only differences:

```bash
git diff --cached --stat --ignore-cr-at-eol
```

> **For AI agents:** four hard rules for a Windows/WSL2 session, all load-bearing:
> 1. **Build on ext4.** Never run the ladder or the gate from `/mnt/c` — it is ~75× slower
>    (assemble ~15 min vs ~7.5 s; the full gate ~30 min vs seconds). Mirror the tree to a
>    WSL-native path and build there.
> 2. **Let the script rewrite the path.** A non-canonical checkout needs the
>    `/mnt/c/Projects/Kovostov-Native` rewrite; `scripts/reproduce_trust.sh` step `[0]` does it
>    for you. Do not skip it on a mirrored or cloned tree.
> 3. **Push via host git**, not WSL git — WSL git has no credentials and will hang on push.
> 4. **Check the real diff** with `git diff --cached --stat --ignore-cr-at-eol`, and stage only
>    intended files, so CRLF/LF churn never sneaks into a commit.

---

## 5. The success-signal trap: `kovc` returns its output byte-count as the exit status

This is not an install requirement, but it is the most important thing to understand about the
environment *before* you run anything — because the very first time you check whether a build
"succeeded," the naive check is wrong.

`kovc` (and the self-hosted K-binaries built from it) report the **size of the file they just
wrote** as the process exit status. Because an exit status is a single byte, that is the output
size **mod 256** — which is **nonzero on success** for essentially every real output. The
canonical example: the self-compile output is 698392 bytes, and `698392 mod 256 = 24`, so a
*successful* self-compile exits with status **24**, not 0.

[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) documents this in the fixpoint leg and
spells out the correct validation:

> the kovc self-compile legs (K1->K2->K3->K4) do NOT [assert rc==0], because kovc returns its
> OUTPUT BYTE COUNT as the process exit status (rc = size mod 256 -> 24 for the 698392-byte
> self-compile, i.e. NONZERO ON SUCCESS) -- those legs are validated by non-empty output + the
> byte-identical and pinned-SHA fixpoint below, never by rc.

So the rule for validating a `kovc`/seed compile leg is: **output file exists, is non-empty,
and matches the expected SHA** — never `rc == 0`. The gate's own corpus runner follows exactly
this: it `rm`s the output first, runs the compile, and then checks `[ -s /tmp/k2_out.bin ]`
(non-empty) before running the produced binary.

The `rc == 0` convention *is* meaningful, but only for the **C-compiled** binaries in the
chain. The `seed` is compiled by M2-Planet (it is the ladder's C-subset compiler) and exits 0
on a clean run; `gcc` itself exits 0 normally. So in
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) the `seed -> K1` leg — a pure x86 C
build — *does* assert `rc == 0`, while the `kovc` self-compile legs deliberately do not.

A second, related freshness rule the scripts enforce, and which you should too: **make
freshness checks filesystem-agnostic.** Remove the expected output *before* a build and require
it to be non-empty *after*; do **not** compare a `/tmp` marker's mtime against an artifact on
`/mnt/c`. Cross-filesystem mtimes between WSL ext4 (`/tmp`) and DrvFs (`/mnt/c`) are
unreliable and will false-flag a freshly written file as stale. The capstone audit's stale-
artifact guard makes the reasoning explicit:

> FRESHNESS via rm-before + non-empty-after (filesystem-agnostic). We DELETED these above, so any
> non-empty file present after the run was necessarily written THIS run.

> **For AI agents:** this is the single most common way to misread a Helix build.
> - **Never** assert `rc == 0` for a `kovc` / `seed`-output (K-binary) compile leg — success is
>   `output exists && non-empty && SHA matches` (e.g. exit **24** for a 698392-byte output, since
>   `698392 mod 256 = 24`). `rc == 0` is meaningful only for the C-compiled binaries (the
>   `seed`-build leg and `gcc` itself).
> - Check freshness by `rm`-ing the output **before** the build and requiring non-empty
>   **after** — do **not** compare a `/tmp` marker mtime against a `/mnt/c` artifact
>   (cross-filesystem mtime is unreliable and false-flags fresh files as stale).
> - Key overall success off the literal token **`GATE_PASS`** printed by
>   [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh), and the pinned anchors `seed
>   9837db12…`, fixpoint `0992dddd…`, gcc-DDC `K1 84363adb…`.

The single validated GPU target is the **RTX 3070 Laptop (`sm_86`)**; there is no cross-arch
or AMD validation ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2
residual 6). If you only have a different NVIDIA architecture or no GPU at all, you can still
reproduce the entire CPU trust core; the capstone is what requires the reference-class
hardware.

---

## 6. What you do *not* need

It is worth being explicit about the tools you might *expect* to need and do not, because
their absence is part of the trust story, not an oversight.

- **No Python — for the toolchain.** The **shipped** Helix toolchain (`hex0 → seed → kovc`) is
  **Python-free**. The repository's *sole* committed `.py` file is a fenced numpy audit oracle
  used only to cross-check the capstone — it is never on the compile/run path. The fence is
  machine-checked: step `[1]` of [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh)
  asserts "exactly 1 committed .py, 24 committed .c/.h":

```bash
NPY=$(git ls-files "*.py" | wc -l | tr -d ' ')
NCH=$(git ls-files "*.c" "*.h" | wc -l | tr -d ' ')
if [ "$NPY" = "1" ]; then say "    committed .py = 1 ($(git ls-files '*.py'))"; else bad "committed .py = $NPY (want 1)"; fi
if [ "$NCH" = "24" ]; then say "    committed .c/.h = 24"; else bad "committed .c/.h = $NCH (want 24)"; fi
```

  That one file is `verification/oracle/oracle_train.py`. So you need Python **only if you run
  the GPU capstone** (whose audit invokes `python3` for the oracle in
  [`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh)); the from-raw `kovc`
  toolchain itself needs no Python interpreter at all.

  > **Note:** the `helixc` you may see referenced elsewhere is the **historical,
  > Python-hosted** compiler frontend, retained for reference only. It is **not** in the
  > shipped compile/run path. The shipped compiler is `kovc`, written in Helix. Do not install
  > Python expecting it to be required to build Helix.

- **No assembler or external linker for the ladder.** The from-raw ladder uses **no** `nasm`,
  `as`, `gcc`, or `ld` to produce its root. As [`stage0/hex0/build.sh`](../../../stage0/hex0/build.sh)
  states: "NOT used: nasm, as, gcc, ld, clang. The bytes in `hex0.hex` are the source of
  truth." Each higher rung is linked by tools the ladder itself built (`catm`, `M0`, `hex2`),
  not by a system toolchain — see [`stage0/helixc-bootstrap/build.sh`](../../../stage0/helixc-bootstrap/build.sh)
  and the next chapter.

- **No pre-built compiler.** Nothing in the chain trusts a pre-existing compiler binary. The
  committed rung binaries are reference copies; the reproduction **deletes them first** and
  rebuilds each rung from the one below
  ([`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) step `[2]`). This is the
  point of the whole exercise, and it is covered in detail in
  [Build from raw](02-build-from-raw.md).

- **No network access** for the core reproduction beyond installing the §2 tools. Once the
  packages are present, `bash scripts/reproduce_trust.sh` runs offline on a clean checkout.

> **For AI agents:** do not install Python to build the CPU trust core, and do not reach for the
> historical Python `helixc` — the shipped compiler is `kovc`. Python (`python3`) is needed
> **only** for the GPU capstone's numpy oracle
> ([`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh)). The committed-file fence
> is exact: **1** `.py`, **24** `.c`/`.h`; if those counts differ on your checkout, the static
> fence in [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) step `[1]` will
> fail and you should treat the tree as suspect.

---

## Checklist

Before moving on, you should have:

- [ ] An **x86-64 Linux** environment — native, or **WSL2** on Windows.
- [ ] The CI's minimal tool set installed:
      `sudo apt-get install -y xxd binutils gcc file coreutils` (plus `bash` ≥ 4).
- [ ] On Windows: a plan to build on a **WSL-native ext4** path (not `/mnt/c`), to **push via
      host git**, and to check diffs with `--ignore-cr-at-eol`.
- [ ] *(Capstone only)* an **NVIDIA GPU** (reference: RTX 3070 Laptop, `sm_86`) and a **CUDA
      toolkit** at `/usr/local/cuda`, plus `python3` for the numpy oracle.
- [ ] The mindset that a `kovc` compile "succeeds" on **non-empty output + matching SHA**, not
      on `rc == 0`.

---

**Next:** [Build from raw](02-build-from-raw.md) — running the `hex0 → … → seed → kovc` ladder
on the environment you just set up, one rung at a time, and watching each rung reproduce its
committed hash.
