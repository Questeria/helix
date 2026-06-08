# Troubleshooting and gotchas

*What this chapter covers:* the small set of traps that bite almost everyone who builds,
reproduces, or scripts Helix — each as **symptom → cause → fix**, grounded in the exact
behaviour of the real build scripts. If something "fails" in a way that looks alarming, the odds
are good it is one of the items below and is actually working as designed.

The authority for everything here is the scripts themselves —
[`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh),
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh), and
[`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh) — together with the two trust
records, [`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md) and
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md). Where a chapter and a script ever
disagree, the script wins.

> **For AI agents:** every "fix" in this chapter is already implemented in the cited scripts. If
> you are *driving* an existing script, you do not need to apply these fixes — they are baked in.
> You need this chapter to (a) avoid re-introducing the same bugs when you write a *new* wrapper,
> and (b) correctly interpret output that *looks* like a failure but is not. The deepest of these
> traps — the nonzero-exit-on-success one — has a dedicated treatment in
> [Part IX — Traps](../part9-for-ai-agents/03-traps.md); read it before you assert any
> pass/fail verdict on a `kovc` or `seed` leg.

---

## 1. `kovc` "fails" with a nonzero exit code — but the build succeeded

**Symptom.** You run a self-hosted Helix compiler binary (a `K`-binary) or the PTX driver and
your shell reports a nonzero exit status. A naive wrapper that checks `rc == 0` declares the
step a failure, even though the output file was written correctly.

```text
  K1->K2 run rc=24
```

**Cause.** `kovc` (and the self-hosted `K1`/`K2`/`K3`/`K4` binaries, and the seed-minted PTX
driver) **returns its output byte-count as the process exit status**. The process exit status is
an 8-bit value, so what you actually see is `bytes mod 256`. The self-compile output is
**698392 bytes**, and `698392 mod 256 = 24`, so a *successful* self-compile exits with status
**24** — nonzero. This is success, not an error. The gate's own comment states it directly
([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh), step `[2]`):

```text
The seed->K1 leg (a C-compiled binary) also asserts rc==0;
the kovc self-compile legs (K1->K2->K3->K4) do NOT, because kovc returns its OUTPUT BYTE
COUNT as the process exit status (rc = size mod 256 -> 24 for the 698392-byte self-compile,
i.e. NONZERO ON SUCCESS) -- those legs are validated by non-empty output + the byte-identical
and pinned-SHA fixpoint below, never by rc.
```

The same convention is documented for the GPU driver in
[`docs/HELIX_GPU_FIRSTLIGHT.md`](../../../docs/HELIX_GPU_FIRSTLIGHT.md): "the driver's exit code is the
emitted byte-count mod 256 … that is success, not an error."

**Fix.** Validate a `kovc`/`seed`-mint/driver leg by **output-exists + non-empty + SHA**, never
by `rc == 0`. That is exactly what the gate does — after each `K`-generation it `rm`s the target
first and then checks the file is non-empty, and the fixpoint is judged by the byte-identical,
pinned-SHA comparison of K2/K3/K4, not by any exit code:

```bash
rm -f /tmp/k1_out.bin
timeout 240 /tmp/K1.bin; rc=$?; echo "  K1->K2 run rc=$rc (kovc returns output-byte-count as exit status -> nonzero on success; validated by non-empty + SHA, NOT rc==0)"
if [ ! -s /tmp/k1_out.bin ]; then
  echo "  FIXPOINT FAIL: K2 generation produced empty/missing /tmp/k1_out.bin (K1 run rc=$rc -- kovc.hx did not self-compile)"; GATE_OK=0; FIX_OK=0
else
  cp /tmp/k1_out.bin /tmp/K2.bin; chmod +x /tmp/K2.bin
fi
```

> **For AI agents:** the *only* legs where `rc == 0` is meaningful are the **C-compiled** ones —
> the `seed → K1` leg (`./seed.bin` is built by `gcc`/M2-Planet and exits 0 on success), the
> `seed → newdrv` mint, the `gcc`-built launcher (`/tmp/train`), and the numpy oracle. On any
> `kovc` self-compile leg or any `K`-binary run, a nonzero exit is expected. Decide pass/fail by
> the produced file (`[ -s "$out" ]`) and its SHA against the pinned anchor
> `0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f`, not by the exit code. The
> driver's exit code is likewise its PTX byte-count mod 256 — ignore it; check `/tmp/out.ptx`
> exists and `cmp`s byte-identical to the committed `.ref.ptx`.

---

## 2. The build is glacially slow on `/mnt/c` (DrvFs / 9p)

**Symptom.** A build that should take seconds takes many minutes. Assembling K1 crawls; the full
gate runs for ~30 minutes instead of completing quickly. The machine is not under load and
nothing is obviously wrong — it is just *slow*.

**Cause.** You are building on a Windows-mounted path under WSL — `/mnt/c/...`, which is the
DrvFs / 9p filesystem. The `seed` and `kovc` do their file I/O one byte at a time
(`read_file_to_arena` reads "1 byte/slot", per
[`docs/HELIX_V1_STDLIB.md`](../../../docs/HELIX_V1_STDLIB.md)), so a single compile performs **millions
of tiny byte-level read/write syscalls**. On native ext4 each is cheap; on `/mnt/c` each one
pays a 9p round-trip to the Windows filesystem. The measured penalty is roughly **75×**: the
clean-reproduction record notes that `/mnt/c` DrvFs "per-syscall latency … had inflated build
wall-time ~75×" ([`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md)), turning a
~7.5-second assemble into ~15 minutes and a seconds-scale gate into ~30 minutes.

**Fix.** Build on **WSL-native ext4**, not on `/mnt/c`. Mirror the committed tree into your WSL
home (ext4) and build there; keep the canonical `/mnt/c` checkout for git operations and commit
from it. This is precisely what the v1.3 final-pass reproduction did — it re-ran the three
result-bearing legs "on a **WSL-native ext4 mirror of the current-head tree** … the mirror
exists only to avoid `/mnt/c` DrvFs per-syscall latency"
([`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md)) — and it produced the *same*
fixpoint `0992dddd…` and the same seed-minted PTX driver. The build's **output is
deterministic**; only its *wall-time* depends on the filesystem.

> **Warning:** the ext4 mirror changes the build's *location*, which interacts with the
> hardcoded-path gotcha in §3 — the fixpoint layer assumes a specific absolute path, so a mirror
> at a different path needs the path rewrite. The byte-for-byte output is identical either way;
> what is documented here is *output*-determinism, not run-from-arbitrary-directory portability
> (see [`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md), "Where it walls").

---

## 3. A checkout at a non-canonical path: the fixpoint reads/writes the wrong directory

**Symptom.** You clone or copy the repo to some path other than
`/mnt/c/Projects/Kovostov-Native` and try to run the gate. The from-raw ladder builds fine, but
the self-host fixpoint behaves strangely: `assemble_k1.sh` returns success yet the regenerated
`k1src.hx` / `k1input.hx` / `k1ptxdrv.hx` never appear in *your* checkout — or, if the canonical
directory is absent entirely, the step fails to read its inputs.

**Cause.** [`stage0/helixc-bootstrap/assemble_k1.hx`](../../../stage0/helixc-bootstrap/assemble_k1.hx)
— the fixpoint concatenator — **hardcodes the canonical absolute path**
`/mnt/c/Projects/Kovostov-Native/...` for both the files it reads (lexer/parser/kovc/drivers) and
the files it writes (the `k1*.hx` outputs): **9 such paths** in that one file. Run from a clone,
its `read_file_to_arena("/mnt/c/Projects/Kovostov-Native/...")` reads from — and writes to — the
**canonical** directory, not the clone. On a machine where the canonical directory does not
exist, those reads fail and the gate's step `[0]`, hence the whole self-host fixpoint, cannot
run. This is the disclosed portability caveat in
[`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md) ("Where it walls"). Note this is a
**build-hygiene limitation, not a trust gap** — the bytes produced are identical; only the
*location* the build assumes is fixed. (The from-raw `stage0/*/build.sh` rungs use relative
`../` paths and *are* fully path-portable; only the fixpoint layer is path-locked.)

**Fix.** Rewrite the hardcoded path to your checkout root before building. The one-command
reproduction does this for you automatically — [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh)
step `[0]` is a pure mechanical path swap:

```bash
say "[0] path rewrite  /mnt/c/Projects/Kovostov-Native -> $ROOT"
mapfile -t HCFILES < <(grep -rlI '/mnt/c/Projects/Kovostov-Native' . 2>/dev/null || true)
if [ "${#HCFILES[@]}" -gt 0 ]; then
  printf '%s\n' "${HCFILES[@]}" | xargs sed -i "s#/mnt/c/Projects/Kovostov-Native#$ROOT#g"
  say "    rewrote ${#HCFILES[@]} file(s)"
fi
```

So the simplest fix is: **run `bash scripts/reproduce_trust.sh`**, which rewrites the paths and
then drives the ladder + fixpoint + gcc-DDC end to end. If you must invoke
`stage0/helixc-bootstrap/assemble_k1.sh` directly from a non-canonical checkout, perform the same
`sed` rewrite first.

> **Note:** `reproduce_trust.sh` deliberately *modifies the working tree* (the path rewrite plus
> the rung-binary rebuilds). Its own header says so: "intended for a CLEAN CHECKOUT (CI runner or
> a throwaway clone) … do not run on a tree you want pristine." Run it on a throwaway clone, not
> on a checkout you intend to commit from.

---

## 4. `git push` hangs forever under WSL

**Symptom.** `git push` from inside WSL hangs with no progress and never completes (or eventually
times out). Read-only git operations such as `git status` and `git log` work fine.

**Cause.** The WSL git installation **has no GitHub credentials**. Windows git is wired to the
Windows credential manager; the Linux git inside WSL is not, so a push that needs authentication
blocks waiting for credentials that never arrive.

**Fix.** Push via the **host (Windows) git**, which has the credential manager configured. Keep
using WSL for the build (it needs the Linux toolchain and ext4 speed), but do the push from the
Windows side against the same `/mnt/c` checkout. The discipline note in
[`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md) reflects this split — builds run under
WSL, and the canonical tree lives on `/mnt/c` so Windows tooling can operate on it.

> **For AI agents:** treat **build** and **push** as different execution contexts. Run builds and
> the gate under WSL (`wsl.exe -e bash -lc '...'`, with `/mnt/c` only *inside* the quoted string);
> run `git push` from the Windows host git. Do not attempt to push from inside the WSL shell — it
> will block, not error, and you will appear hung.

---

## 5. Half the tree shows as "modified" on Windows (CRLF/LF churn)

**Symptom.** `git status` lists many files as modified that you never touched. A `git diff` shows
no semantic change — only line endings differ. This is noisy and risks committing an accidental
whole-file EOL rewrite.

**Cause.** **CRLF/LF line-ending churn.** On Windows, editors and git's autocrlf can flip line
endings, so files differ from the committed copies by EOL only. This matters for Helix beyond
mere noise: the gate `cmp`s the kovc-emitted PTX **byte-for-byte** against committed `.ref.ptx`
references, so an EOL flip on a reference file would break the byte-identical match. That is why
the repo pins line endings for the byte-sensitive files in
[`.gitattributes`](../../../.gitattributes):

```text
*.sh text eol=lf
...
helixc/examples/*.ref.ptx text eol=lf
```

**Fix.** Stage only the files you actually intend to change, and verify the *content* diffstat
while **ignoring EOL-only differences**:

```bash
git diff --cached --stat --ignore-cr-at-eol
```

If that diffstat is empty for a file you did not mean to touch, the only change is line endings —
unstage it. Stage intended files explicitly (`git add <path>`) rather than `git add -A`, so an
ambient EOL flip elsewhere in the tree does not ride along into your commit.

---

## 6. A freshness check false-flags a freshly built artifact as stale

**Symptom.** You write a guard meant to ensure a build actually produced a *new* artifact (not a
leftover from a previous run), and it reports the brand-new file as **STALE** even though the
build just wrote it. The capstone audit hit exactly this and had to correct it.

**Cause.** The guard compared an **mtime across filesystems** — a marker in `/tmp` (WSL ext4)
against an artifact under `/mnt/c` (DrvFs). Cross-filesystem mtime comparison is unreliable here:
DrvFs has coarse mtime granularity and clock skew relative to ext4, so an `-nt` ("newer than")
test against a `/tmp` marker reads freshly-written `/mnt/c` files as older than the marker. The
corrected script documents the failure mode in
[`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh), step `[4b]`:

```text
The earlier mtime "-nt
marker" test was unreliable here: the marker lived in /tmp [WSL ext4] while the artifacts live in
$RT under /mnt/c [DrvFs], whose coarse mtime + clock skew made freshly-written files read as STALE.
```

**Fix.** Make the freshness check **filesystem-agnostic**: `rm` the output **before** the build,
then require it **non-empty after**. Because you deleted it first, any non-empty file present
afterward was necessarily written by this run — no mtime comparison needed. That is exactly the
corrected pattern in [`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh):

```bash
rm -f "$RT/loss_curve.csv" "$RT/init_weights.bin"
# FRESHNESS via rm-before + non-empty-after (filesystem-agnostic). We DELETED these above, so any
# non-empty file present after the run was necessarily written THIS run.
...
for art in loss_curve.csv init_weights.bin; do
  if [ ! -s "$RT/$art" ]; then echo "  AUDIT FAIL: train left no/empty $art this run (rc=$trc)"; OK=0;
  else echo "  fresh artifact: $art ($(stat -c%s "$RT/$art") B, written this run)"; fi
done
```

> **For AI agents:** never assert freshness by comparing a `/tmp` marker's mtime to a `/mnt/c`
> artifact — it will lie. Use the documented idiom: delete the expected output first, run, then
> require it non-empty (`[ -s "$out" ]`). The same `rm`-before/non-empty-after pattern is what
> keeps a *stale* `/tmp` output from being copied into a later `K`-generation and producing a
> false fixpoint match (see [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[2]`).

---

## 7. The GPU capstone "fails" on a machine with no CUDA GPU

**Symptom.** You try to reproduce the GPU capstone — the transformer trained on `kovc`-emitted
kernels — on a CPU-only box (for example a GitHub-hosted CI runner) and it does not produce the
`CAPSTONE_AUDIT_PASS` line.

**Cause.** The capstone **requires a real CUDA GPU**. [`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh)
trains on the reference RTX 3070, links against `libcuda` (`gcc train_transformer.c … -lcuda …`),
and runs the GPU finite-difference check. None of that can run without a GPU. By design, the CI
reproduction is **CPU-only**: [`.github/workflows/trust-reproduce.yml`](../../../.github/workflows/trust-reproduce.yml)
runs the from-raw ladder, the self-host fixpoint, and the gcc-DDC on a stock `ubuntu-latest`
runner, and explicitly *excludes* the capstone:

```text
CPU-only by design: GitHub-hosted runners have no GPU, so the transformer CAPSTONE (perf + 2%
oracle parity) is NOT run here -- it is verified on a CUDA host via scripts/capstone_audit.sh
(a self-hosted GPU runner could add it later as a separate job).
```

`reproduce_trust.sh` says the same in its header: "The GPU capstone is verified SEPARATELY by
scripts/capstone_audit.sh on a CUDA host (no GPU here)."

**Fix.** Split the two reproductions by where they belong. On any x86-64 Linux (CPU-only is
fine), run the trust *core*:

```bash
bash scripts/reproduce_trust.sh
```

On a **CUDA host** (the reference box is an RTX 3070 Laptop, `sm_86`, CUDA-12.8 `ptxas`), run the
capstone separately:

```bash
bash scripts/capstone_audit.sh
```

> **Residual:** the capstone is a **single hardware target** (`sm_86`); there is no cross-arch
> (`sm_80`/`sm_90`) or AMD validation, and the chain is **complete to PTX, not to GPU machine
> code** — below PTX it trusts NVIDIA's closed `ptxas`, the CUDA driver, the GPU hardware, and the
> C host launcher. GPU performance is a *fraction* of cuBLAS (~50–67.5% on the reference box), not
> parity, and the end-to-end capstone speedup is **7.0–8.7×** (Amdahl-bound), not ≥10×. The hard
> gate — loss parity to the independent numpy oracle — holds to ≈0% (worst-case ~`0.0000088`,
> well under the 2% bar). Stated precisely in
> [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) (residuals 6–8).

---

## Quick reference

| Symptom | Root cause | Fix |
|---------|-----------|-----|
| `kovc`/`K`-binary/driver exits nonzero (e.g. `24`) | exit status **is** output-byte-count mod 256 — nonzero on success | validate by output-exists + non-empty + SHA, not `rc==0`; `rc==0` is meaningful only on C-compiled legs (`seed`, the mint, `gcc` launcher, oracle) |
| Build takes minutes instead of seconds | building on `/mnt/c` DrvFs/9p → byte-level syscalls ~75× slower | build on WSL-native ext4; commit from `/mnt/c` |
| Fixpoint reads/writes the wrong dir from a clone | `assemble_k1.hx` hardcodes 9 canonical absolute paths | run `bash scripts/reproduce_trust.sh` (step `[0]` rewrites the path) |
| `git push` hangs under WSL | WSL git has no GitHub credentials | push from the Windows host git |
| Many files show "modified" (EOL only) | CRLF/LF churn on Windows | stage intended files only; `git diff --cached --stat --ignore-cr-at-eol` |
| Fresh artifact reported STALE | cross-filesystem mtime (`/tmp` ext4 vs `/mnt/c` DrvFs) is unreliable | `rm` before the build, require non-empty after (filesystem-agnostic) |
| Capstone won't pass on CI | capstone needs a real CUDA GPU; CI is CPU-only | run the core with `reproduce_trust.sh`; run `capstone_audit.sh` on a CUDA host |

---

**Next:** this concludes Part II — Setup & Build. For the dedicated operator manual — driving
Helix as an AI agent, the non-negotiable invariants, the full trap list, and copy-paste recipes —
continue to [Part IX — Driving Helix](../part9-for-ai-agents/01-driving-helix.md).
