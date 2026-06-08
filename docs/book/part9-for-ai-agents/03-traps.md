# For AI agents: the traps

*What this chapter covers:* the specific, concrete traps that cost real time when an AI operator
drives Helix — each as **symptom → why → fix**, grounded in the exact scripts that handle it.

The Helix trust scripts are written *defensively*: nearly every trap below is one the maintainers
already hit, diagnosed, and hard-coded a guard against. That is good news for you — the fix is
usually "do what the script already does," and the script is the citation. This chapter exists so
you recognise the symptom before you waste a build cycle on it, and so that when you write your own
wrapper around these tools you reproduce the *discipline*, not just the command.

Read [Non-negotiables](02-non-negotiables.md) first; this chapter assumes you already know the
pinned anchors (`seed = 9837db12…`, fixpoint `K2==K3==K4 = 0992dddd…`, gcc-DDC `K1 = 84363adb…`)
and that the gate signals success with the literal token `GATE_PASS`. When in doubt, the repo
source wins over this chapter — if a script disagrees with anything here, the script is right and
this prose is the bug.

---

## Trap 1 — `kovc` exits non-zero *on success*

**Symptom.** You run a `kovc`/seed compile leg, it produces a correct output binary, and the
process exit status is something like `24` (or `81`, `225`, …). Your wrapper sees a non-zero return
code, concludes the compile failed, and aborts — or worse, marks a *passing* build as broken.

**Why.** `kovc` (and every self-hosted **K-binary** in the fixpoint — K1, K2, K3, K4) returns its
**output byte-count** as the process exit status. The exit status is a single byte, so what you
observe is `size mod 256`. The self-compiled `kovc` output is **698392 bytes**, and `698392 mod 256
= 24` — so a *successful* self-compile exits `24`. This is by design, not a bug; the compiler has no
separate "I succeeded" status code, it just reports how many bytes it wrote.

The gate spells this out in a comment block right above the fixpoint legs in
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[2]`:

```bash
# the kovc self-compile legs (K1->K2->K3->K4) do NOT, because kovc returns its OUTPUT BYTE
# COUNT as the process exit status (rc = size mod 256 -> 24 for the 698392-byte self-compile,
# i.e. NONZERO ON SUCCESS) -- those legs are validated by non-empty output + the byte-identical
# and pinned-SHA fixpoint below, never by rc.
```

and the K1→K2 leg log line says it in one breath
([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh), step `[2]`):

```bash
timeout 240 /tmp/K1.bin; rc=$?; echo "  K1->K2 run rc=$rc (kovc returns output-byte-count as exit status -> nonzero on success; validated by non-empty + SHA, NOT rc==0)"
```

**Fix.** **Do not assert `rc == 0` on any `kovc`/K-binary/seed-self-compile leg.** Validate success
the way the gate does: the output file **exists**, is **non-empty**, and (where you have an anchor)
matches the expected **SHA-256**. Concretely, the gate's K2 validation is:

```bash
if [ ! -s /tmp/k1_out.bin ]; then
  echo "  FIXPOINT FAIL: K2 generation produced empty/missing /tmp/k1_out.bin (K1 run rc=$rc -- kovc.hx did not self-compile)"; GATE_OK=0; FIX_OK=0
else
  cp /tmp/k1_out.bin /tmp/K2.bin; chmod +x /tmp/K2.bin
fi
```

Note what the gate checks (`[ ! -s … ]`, i.e. non-empty) and what it deliberately *omits* (any
test on `rc`).

The one place `rc == 0` **is** meaningful is a **C-compiled** binary leg. The `seed → K1` leg is
checked for `rc == 0` because the `seed` itself is a C-compiled binary (its `seed.c` built by
M2-Planet/gcc), and a C-compiled program exits 0 on success; the gate asserts both `rc` and
non-empty for that leg only:

```bash
timeout 1200 ./seed.bin k1src.hx /tmp/K1.bin; rc=$?; echo "  seed->K1 rc=$rc ($((SECONDS-t0))s)"
if [ "$rc" -ne 0 ] || [ ! -s /tmp/K1.bin ]; then
  echo "  FIXPOINT FAIL: K1 generation rc/empty (seed->K1 rc=$rc, /tmp/K1.bin empty/missing)"; GATE_OK=0; FIX_OK=0
fi
```

The clean line to draw: the **`gcc`-compiled** auditors in the diverse-double-compile *do* check
`rc`/exit — e.g. the gcc-built seed's self-test must exit `42`
([`stage0/helixc-bootstrap/ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh)):

```bash
/tmp/seed_gcc; stc=$?; echo "  seed_gcc no-arg self-test exit=$stc (want 42)"
if [ "$stc" -ne 42 ]; then echo "  DDC_FAIL (gcc-seed self-test exit=$stc != 42 -- gcc-built seed misbehaves)"; exit 2; fi
```

…but the two *K1 generations* in that same script — both produced by **Helix-built seed compilers**
— are explicitly **not** rc-checked, for exactly the byte-count reason
([`stage0/helixc-bootstrap/ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh)):

```bash
# Both K1 outputs are produced by Helix-built seed
# compilers (M2-seed and gcc-seed BOTH run the Helix seed program), which exit NONZERO on
# success (output byte-count); success is the NON-EMPTY assert below, NOT rc==0.
```

> **For AI agents:** never branch on `rc == 0` for a `kovc`/seed/K-binary compile. Validate with
> output-exists **+** non-empty **+** (where pinned) SHA-256. Reserve `rc`/exit-code assertions for
> C-compiled binaries: the `seed → K1` leg (`rc == 0`) and the gcc-built-seed self-test (`exit 42`).
> A non-zero status from a `kovc` self-compile is the *expected* success signal, not an error.

---

## Trap 2 — builds are ~75× slower on `/mnt/c`

**Symptom.** A build that should take seconds takes many minutes. Assembling the fixpoint sources
crawls (~15 min instead of ~7.5 s); a full gate run drags on for ~30 minutes; you start adding
generous `timeout`s or assume something is hung.

**Why.** The `seed` and `kovc` do their I/O with **millions of tiny byte-level `fgetc`/`fputc`
syscalls**. On a Windows checkout, the repo lives on `/mnt/c`, which WSL exposes through the
**DrvFs / 9p** filesystem bridge. Every one of those byte-level syscalls pays a 9p round-trip across
the WSL↔Windows boundary. The constant factor is brutal: roughly **75×** slower than native ext4.
[`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md) records this directly — the result-bearing
legs were re-run on

> a **WSL-native ext4 mirror of the current-head tree** (byte-identical committed sources; the
> mirror exists only to avoid `/mnt/c` DrvFs per-syscall latency, which had inflated build
> wall-time ~75×).

The same doc notes the coarse DrvFs `mtime` it causes — which is the root of Trap 6, below.

**Fix.** **Build on WSL-native ext4, not on `/mnt/c`.** Mirror the tree into the Linux filesystem
(e.g. under `~`), build there, and commit from the `/mnt/c` checkout. The bytes produced are
identical — [`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md) confirms the ext4 mirror
"produces the **same** fixpoint `0992dddd…` and the same seed-minted PTX driver" — so this is purely
a wall-clock fix, with no effect on the trust result.

> **For AI agents:** if a `seed`/`kovc` build is taking minutes where the docs say seconds, suspect
> `/mnt/c` DrvFs latency, not a hang, before you raise a timeout. Build on ext4 and commit from
> `/mnt/c`. Moving to ext4 is an **output-determinism-preserving** path change (same SHAs), **not**
> the run-from-arbitrary-path portability that Trap 3 is about.

---

## Trap 3 — the build hard-codes an absolute path

**Symptom.** You check the repo out at some path that is **not** the canonical
`/mnt/c/Projects/Kovostov-Native`, run the gate, and step `[0]` (source regeneration) silently does
the wrong thing: `assemble_k1.sh` returns `rc = 0` but reads from / writes to the **canonical** dir,
not your checkout — so the `k1*.hx` you expected never appear where you are, or the build fails
outright if the canonical dir is absent.

**Why.** `assemble_k1.hx` (the fixpoint concatenator) **hard-codes the canonical absolute path** for
both its reads and its writes — nine such paths. From
[`stage0/helixc-bootstrap/assemble_k1.hx`](../../../stage0/helixc-bootstrap/assemble_k1.hx):

```text
let lex_len = read_file_to_arena("/mnt/c/Projects/Kovostov-Native/helixc/bootstrap/lexer.hx");
...
write_file_to_arena("/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/k1src.hx", o1_base, o1_len);
```

[`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md) ("Where it walls") documents this exact
failure mode: running `assemble_k1.sh` *inside a clone* "returns rc=0 but **reads from and writes to
the CANONICAL dir, not the clone**." The from-raw ladder itself is path-independent (the
`stage0/*/build.sh` rungs use relative `../` paths and clone cleanly), so the wall is specifically at
the **fixpoint layer**, because of these hard-coded strings. This is a disclosed build-hygiene
portability caveat, **not** a trust gap — the bytes are identical; only the location the build
assumes is fixed.

**Fix.** A checkout at any other path needs a **path rewrite** before the fixpoint will run there,
and [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) does this automatically as
its step `[0]`:

```bash
# --- [0] disclosed path rewrite -----------------------------------------------------------------
# assemble_k1.hx (the fixpoint concatenator) + a few scripts hardcode the original absolute build
# path; rewrite it to THIS checkout so the build runs at any path. This is the disclosed portability
# caveat (docs/CLEAN_REPRODUCTION.md "Where it walls"); the rewrite is a pure mechanical path swap.
say "[0] path rewrite  /mnt/c/Projects/Kovostov-Native -> $ROOT"
mapfile -t HCFILES < <(grep -rlI '/mnt/c/Projects/Kovostov-Native' . 2>/dev/null || true)
if [ "${#HCFILES[@]}" -gt 0 ]; then
  printf '%s\n' "${HCFILES[@]}" | xargs sed -i "s#/mnt/c/Projects/Kovostov-Native#$ROOT#g"
  say "    rewrote ${#HCFILES[@]} file(s)"
fi
```

So if you drive the **whole** reproduction through `bash scripts/reproduce_trust.sh`, the rewrite is
handled for you. If you instead invoke `scripts/gate_kovc.sh` directly from a non-canonical path, you
must do the equivalent rewrite first, or build at the canonical path.

> **For AI agents:** to reproduce from a non-canonical checkout, run the whole flow via
> `bash scripts/reproduce_trust.sh` (it path-rewrites at step `[0]`). If you call `gate_kovc.sh`
> directly off-canonical, the fixpoint will read/write the *canonical* dir — rewrite
> `/mnt/c/Projects/Kovostov-Native` → your repo root in the matching files first, exactly as step
> `[0]` does. Note that step `[0]` **modifies the working tree** (it `sed -i`s real files); only run
> it on a throwaway/CI clone, not on a tree you want pristine.

---

## Trap 4 — `git push` from inside WSL hangs

**Symptom.** You stage and commit fine, then `git push` from the WSL shell hangs indefinitely (or
fails on an auth prompt that never resolves).

**Why.** The WSL-side `git` has **no GitHub credentials** — the credential helper that holds your
token lives on the **Windows** side (Git Credential Manager). A push from WSL has nothing to
authenticate with, so it blocks waiting for input that, in an automated context, never comes.

**Fix.** **Push via the host (Windows) `git`**, which is wired to the credential manager. Do the
build under WSL/ext4 as usual, but run the `push` from the Windows side. (The
[`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md) discipline note keeps WSL invocations to
`.sh` files and crosses the boundary deliberately for exactly this kind of reason.)

> **For AI agents:** keep `git push` on the **host** git, not the WSL git. If a WSL push blocks with
> no output, do not wait on it or feed it credentials — switch to the Windows-side push.

---

## Trap 5 — every file shows as modified (CRLF/LF only)

**Symptom.** `git status` lights up with files that look modified, but the content is unchanged —
the entire "diff" is line-ending churn. You risk committing a tree-wide CRLF↔LF reflow, which (among
other harm) can break the **byte-identical** PTX text regression and other hash-pinned comparisons.

**Why.** On Windows, Git's `autocrlf` and editors can rewrite line endings, so files flip between
CRLF and LF and register as changed even though nothing semantic moved. This matters acutely here
because the trust chain depends on **byte-exact** artifacts. The repo pins the line endings of the
files that must stay byte-stable in
[`.gitattributes`](../../../.gitattributes):

```text
*.sh text eol=lf
stage0/hex0/test/*.expected text eol=lf
stage0/hex0/test/*.hex0 text eol=lf
helixc/examples/*.ref.ptx text eol=lf
```

The comment on the `.ref.ptx` line says why in one sentence — the gate "cmp's it BYTE-for-byte
against the kovc-emitted PTX (LF from WSL). Pin LF so git's autocrlf cannot corrupt the
byte-identical match." The gate echoes the same `.gitattributes eol=lf` reliance in
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[1]`.

**Fix.** **Stage only the files you actually intend to change**, and verify the staged content diff
while ignoring end-of-line noise:

```bash
git diff --cached --stat --ignore-cr-at-eol
```

If that diffstat shows files you did not mean to touch, do not commit — unstage them. The goal is
that your commit carries the intended content change and **nothing else**, so that no incidental EOL
reflow can perturb a hash-pinned artifact.

> **For AI agents:** before committing on Windows, run `git diff --cached --stat --ignore-cr-at-eol`
> and confirm the staged set is exactly the files you meant to change. Stage files individually; do
> **not** `git add -A` a tree that shows EOL-only modifications, or you may reflow a byte-pinned file
> (`*.ref.ptx`, `*.sh`) and break the gate's byte-identical comparisons.

---

## Trap 6 — freshness checks via cross-filesystem `mtime`

**Symptom.** You add a "is this artifact fresh?" guard by comparing a `/tmp` marker's mtime against
a build output under `/mnt/c` (e.g. `marker -nt artifact`). Freshly-written files get **false-flagged
as STALE**, so a genuinely-good run fails its own freshness check.

**Why.** Cross-filesystem mtime comparison is unreliable here. The marker lives on WSL **ext4**
while the artifact lives under `/mnt/c` on **DrvFs**, whose coarse mtime resolution and clock skew
mean a file written *after* the marker can read as *older*. [`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md)
records that this bug was caught and corrected — a "`/tmp`-vs-`/mnt/c` **cross-filesystem `mtime`
test** was false-flagging freshly-written artifacts as STALE."

**Fix.** Make freshness **filesystem-agnostic**: **`rm` the output *before* the build, then require
it non-empty *after*.** If you deleted it first, any non-empty file present afterward was necessarily
written this run — no mtime needed.
[`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh) step `[4b]` is the canonical
pattern:

```bash
rm -f "$RT/loss_curve.csv" "$RT/init_weights.bin"
# FRESHNESS via rm-before + non-empty-after (filesystem-agnostic). We DELETED these above, so any
# non-empty file present after the run was necessarily written THIS run. (The earlier mtime "-nt
# marker" test was unreliable here: the marker lived in /tmp [WSL ext4] while the artifacts live in
# $RT under /mnt/c [DrvFs], whose coarse mtime + clock skew made freshly-written files read as STALE.)
```

then, after the run:

```bash
for art in loss_curve.csv init_weights.bin; do
  if [ ! -s "$RT/$art" ]; then echo "  AUDIT FAIL: train left no/empty $art this run (rc=$trc)"; OK=0;
  else echo "  fresh artifact: $art ($(stat -c%s "$RT/$art") B, written this run)"; fi
done
```

This same rm-before / non-empty-after discipline is what hardens the gate's fixpoint against a
**stale `/tmp`** output being copied into a later K-generation and producing a false fixpoint match
(see [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[2]`, where each K-generation
`rm -f`s its output file before the run).

> **For AI agents:** never gate freshness on `mtime` across a `/tmp`(ext4)↔`/mnt/c`(DrvFs) boundary —
> it false-flags fresh files as stale. Use **`rm` the artifact before the build, assert non-empty
> after**. This both fixes the false-stale and prevents a stale artifact from producing a false
> pass.

---

## Trap 7 — running the GPU capstone where there is no GPU

**Symptom.** You try to reproduce the **whole** trust story — including the transformer capstone — on
a machine with no CUDA GPU (a CI runner, a laptop without an NVIDIA card), and the capstone legs fail
or skip. Or, conversely, you expect `scripts/reproduce_trust.sh` to *include* the capstone and are
surprised it does not.

**Why.** The capstone is the real-capability proof — a ≥2-layer transformer trained **end-to-end on
`kovc`-emitted PTX kernels**, checked for loss parity against an independent numpy oracle — and it
**requires a real CUDA GPU**. The CPU-only reproduction and the GPU capstone are **deliberately
separate**. [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) runs the
**ladder + self-host fixpoint + gcc-DDC** and says so in its header:

```bash
# The GPU capstone is verified SEPARATELY by scripts/capstone_audit.sh on a CUDA host (no GPU here).
```

and the CI workflow [`.github/workflows/trust-reproduce.yml`](../../../.github/workflows/trust-reproduce.yml)
is **CPU-only by design**:

```yaml
# CPU-only by design: GitHub-hosted runners have no GPU, so the transformer CAPSTONE (perf + 2%
# oracle parity) is NOT run here -- it is verified on a CUDA host via scripts/capstone_audit.sh
# (a self-hosted GPU runner could add it later as a separate job).
```

**Fix.** Run the right script on the right hardware. On any x86-64 Linux (incl. a CI runner), the
CPU trust core reproduces with:

```bash
bash scripts/reproduce_trust.sh
```

The capstone runs **separately, on a CUDA host**, via:

```bash
bash scripts/capstone_audit.sh
```

Keep the boundary honest in any claim you make. The trust chain is **complete to PTX, not to GPU
machine code**: below PTX it trusts NVIDIA's closed `ptxas`, the CUDA driver, the GPU hardware
(reference RTX 3070 Laptop, sm_86), and the C host launcher — see
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) and the "trusted computing base" section
of [`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md). And the capstone's headline is **loss
parity** (≈0% worst-case vs the numpy oracle, the hard gate), **not** GPU-perf parity: kovc's GEMM is
a *fraction* of cuBLAS (~50–67.5% on the reference box), and the end-to-end speedup is **7.0–8.7×**
(Amdahl-bound), not ≥10×.

> **For AI agents:** do **not** expect or assert capstone results from a CPU-only run.
> `scripts/reproduce_trust.sh` (and the `trust-reproduce` CI job) cover ladder + fixpoint + gcc-DDC
> *only*; the capstone is `scripts/capstone_audit.sh` on a CUDA host. When you report the capstone,
> the load-bearing claim is **loss parity ≈0% vs the numpy oracle** — never "beats cuBLAS" or "fully
> verified GPU." The chain is complete to PTX; everything below PTX is in the trusted computing base.

---

## The pattern behind all seven

Step back and the traps share one shape: **a naïve success/freshness signal that is wrong in this
environment**, and a guard that replaces it with something *byte-grounded*.

- Don't trust **`rc == 0`** from a `kovc` compile → trust **non-empty output + SHA**.
- Don't trust **`mtime`** across filesystems → trust **rm-before + non-empty-after**.
- Don't trust **wall-clock intuition** on `/mnt/c` → build on **ext4**.
- Don't trust a tree-wide **`git add`** on Windows → trust an **EOL-ignoring diffstat** of exactly
  the files you meant.
- Don't trust the build to **find its own path** → **path-rewrite** (or run at the canonical path).
- Don't trust **WSL git** to push, or a **CPU runner** to do GPU work → use the **host git** and a
  **CUDA host** respectively.

In every case the scripts already encode the right answer, fail **closed** on the wrong one, and say
*why* in a comment. When you script Helix, mirror that discipline: prefer the byte-exact check the
script uses over the convenient signal your environment hands you.

---

**Next:** [Recipes](04-recipes.md) — copy-paste, end-to-end flows for the common operator tasks
(build from raw, run the gate, reproduce the trust core, compile and run a `.hx` program), with the
guards from this chapter already baked in.
