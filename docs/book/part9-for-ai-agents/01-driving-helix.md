# For AI agents: driving Helix

*What this chapter covers: the mental model and the canonical commands an AI agent uses to operate Helix end to end — orient (check the static fence, run the gate), the build/verify loop, where the artifacts live (the K-binaries, `/tmp`, the gate verdict), how to compile and run a `.hx` program, and how to reproduce the whole trust chain.*

This is the operator manual for an LLM-driven agent. The rest of the book explains Helix to a human reader; this part tells *you*, the agent, exactly what to run, exactly what tokens to match, and exactly what not to do. Where this chapter and a repo source ever disagree, the **repo source wins** and the chapter is the bug — flag it, do not silently follow stale prose.

Two companion chapters carry the rules and the patterns: read [part9-for-ai-agents/02-non-negotiables.md](02-non-negotiables.md) before you act, and reach for [part9-for-ai-agents/04-recipes.md](04-recipes.md) for copy-paste flows. The traps that bite a scripting agent specifically live in [part9-for-ai-agents/03-traps.md](03-traps.md).

---

## The mental model

Helix is a from-raw-binary, self-hosting compiler with GPU code generation. Four facts shape everything you do as an operator:

1. **There is no trusted pre-built compiler.** The toolchain is a *ladder* of small rungs, each built **only by the rung below it**, rooted in a 299-byte hand-authored ELF (`hex0`). The ladder climbs `hex0 → hex1 → hex2 → catm → M0 → cc_amd64 → M2-Planet → seed → kovc`. The `seed` is an Apache-2.0 C-subset compiler; `kovc` is the Helix compiler *written in Helix*. The committed rung binaries are reference copies — reproduction **deletes them and rebuilds from raw**.

2. **`kovc` reproduces itself byte-for-byte.** The *self-host fixpoint* is `seed → K1 → K2 → K3 → K4` with **K2 == K3 == K4 byte-identical**, pinned to `0992dddd…`. K1 through K4 are successive generations of the Helix compiler; that they converge to a fixed point is the proof `kovc` is a stable self-compiler.

3. **Trust is pinned to hashes, not prose.** Three anchors are the ground truth an honest run must reproduce, pinned in [scripts/reproduce_trust.sh](../../../scripts/reproduce_trust.sh):

   ```bash
   SEED_SHA=9837db12752a22159ca75a533910bc0d7b9afb35df9b9963f256b7b1b915c9bb
   K1_SHA=84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba
   FIX_SHA=0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f
   ```

   `seed` is the from-raw C-subset compiler hash, `K1` is the gcc-diverse-double-compile anchor, and `fix` is the self-host fixpoint (`K2==K3==K4`).

4. **The chain is complete to PTX, not to GPU machine code.** The CPU path is all-the-way-down from raw binary; the GPU path is auditable from `hex0` **to PTX text**, after which it trusts NVIDIA's closed `ptxas`, the CUDA driver, the GPU hardware, and the C host launcher. State this boundary honestly; never claim "fully verified GPU." See [docs/TRUST_CHAIN_CLOSED.md](../../../docs/TRUST_CHAIN_CLOSED.md) §R and residual 8.

> **For AI agents:** key every decision off the literal anchor strings above (`9837db12…` / `84363adb…` / `0992dddd…`) and off output **tokens** the scripts emit (`GATE_PASS`, `DDC_ANCHOR_OK`, `CAPSTONE_AUDIT_PASS`), never off an English paraphrase of "it passed". Dereference the cited paths before acting; do not trust your memory of them.

### Where you run

The build and verify scripts are bash scripts that expect a **Linux x86-64** environment (a CI runner, or WSL2 on the reference machine). They use `set -u` / `set -uo pipefail`, GNU coreutils (`sha256sum`, `stat`), `xxd`, `file`, `objdump`, and `gcc`. The canonical reference checkout lives at `/mnt/c/Projects/Kovostov-Native` and a few build files hardcode that path — see [Residual: path-locked fixpoint](#residual-the-fixpoint-layer-is-path-locked) below before you run from anywhere else.

---

## Step 1 — orient: read the fence, then run the gate

Before you build or change anything, establish where you are. Orientation is two cheap checks: the **static fence** (a structural invariant of the committed tree) and the **gate** (the dynamic proof the toolchain still self-hosts).

### The static fence

The fence asserts the committed tree carries the *source* of a Python-free toolchain: **exactly one** committed `.py` (a fenced numpy audit oracle, never on the compile/run path) and **24** committed `.c`/`.h` files (the small hand-authored C subset). This is step `[1]` of [scripts/reproduce_trust.sh](../../../scripts/reproduce_trust.sh):

```bash
NPY=$(git ls-files "*.py" | wc -l | tr -d ' ')
NCH=$(git ls-files "*.c" "*.h" | wc -l | tr -d ' ')
if [ "$NPY" = "1" ]; then say "    committed .py = 1 ($(git ls-files '*.py'))"; else bad "committed .py = $NPY (want 1)"; fi
if [ "$NCH" = "24" ]; then say "    committed .c/.h = 24"; else bad "committed .c/.h = $NCH (want 24)"; fi
```

If the fence is wrong — more than one `.py`, or a count other than 24 — **stop and investigate** before touching the build. A broken fence means the tree is not the trusted shape, and any downstream "pass" is suspect.

> **For AI agents:** the single committed `.py` is `verification/oracle/oracle_train.py` and it is an **audit oracle**, never on the compile or run path. Do not "fix" the toolchain by adding Python, and do not treat the oracle as part of the compiler. Adding a second committed `.py` breaks the fence.

### The gate

The gate is [scripts/gate_kovc.sh](../../../scripts/gate_kovc.sh) — the universal verification gate. One run proves, from the current compiler sources, that:

- the **self-host fixpoint** still holds (`seed → K1 → K2 → K3 → K4`, K2==K3==K4 byte-identical **and** equal to the pinned `0992dddd…`);
- the **GPU PTX text regression** holds (a re-minted PTX driver emits PTX byte-identical to the committed `*.ref.ptx` references — this leg is pure text, **needs no GPU and no `ptxas`**);
- the **feature corpus** passes (**109 programs**, compiled and run through the fresh K2, each asserted against its expected exit code);
- the **negative diagnostics** pass (**4** malformed programs that must produce a `path:line:col` parse error and write no ELF).

Run it and match the verdict token:

```bash
bash scripts/gate_kovc.sh
```

It prints exactly one of `GATE_PASS` or `GATE_FAIL` on its own line, and exits `0` only on pass (the verdict is propagated to the process exit status). The decisive lines you parse, quoted from the gate:

```text
  FIXPOINT OK (K2==K3==K4 byte-identical AND == pinned known-good)
  CORPUS: 109 passed, 0 failed
  CHECK_ERR: 4 passed, 0 failed
GATE_PASS
```

> **For AI agents:** confirm success with `grep -q '^GATE_PASS'` against the gate's output — anchored at line start, the exact token — exactly as [scripts/reproduce_trust.sh](../../../scripts/reproduce_trust.sh) step `[3]` does. Do **not** infer pass from "no errors" or from the exit code alone; match the token. A printed `GATE_FAIL` must never be read as success.

The gate is **fail-closed by design**. A missing committed PTX reference, an empty `/tmp` output, or a fixpoint that is self-consistent but `!=` the pinned hash are all **real failures** (`GATE_OK=0`), not benign skips. The only legitimate skip anywhere in the discipline is running a `.ptx` *on a GPU*, and the gate never does that — GPU-hardware execution lives in the capstone audit, not here.

---

## Step 2 — where the artifacts live

You will reason about a small, fixed set of build artifacts. Most live in `/tmp` (regenerated every run); a few are committed references in the tree. Knowing these by path lets you inspect state between steps.

| Artifact | Path | What it is |
|----------|------|------------|
| `seed` binary | `stage0/helixc-bootstrap/seed.bin` | the from-raw C-subset compiler; **gitignored + untracked** (rebuilt from raw). Its hash must equal `9837db12…`. |
| K1 | `/tmp/K1.bin` | `seed` compiling `k1src.hx`; the gcc-DDC anchor (`84363adb…`). |
| K2 / K3 / K4 | `/tmp/K2.bin`, `/tmp/K3.bin`, `/tmp/K4.bin` | successive self-compilations; **byte-identical** at the fixpoint (`0992dddd…`). K2 is the compiler the corpus runs through. |
| PTX driver | `/tmp/newdrv.bin` | the PTX-emitting `kovc`, freshly minted from the raw-binary `seed` (`./seed.bin k1ptxdrv.hx /tmp/newdrv.bin`). |
| emitted PTX | `/tmp/out.ptx` | the PTX a driver run produces; diffed against a committed `*.ref.ptx`. |
| PTX references | `helixc/examples/vector_add_kernel.ref.ptx`, `helixc/examples/tiled_matmul_kernel.ref.ptx` | committed sm_86 baselines the PTX text regression anchors to. |
| gate verdict | the gate's stdout | the `GATE_PASS` / `GATE_FAIL` token (and, in the detached-runner flow, `.stage33-logs/gate_verdict.txt`). |
| capstone verdict | the audit's stdout | `CAPSTONE_AUDIT_PASS` / `CAPSTONE_AUDIT_FAIL`. |

The compiler **source** lives elsewhere: the Helix-in-Helix compiler is `helixc/bootstrap/{lexer,parser,kovc}.hx`, and the gate regenerates the self-compile inputs (`k1src.hx`, `k1input.hx`, `k1ptxdrv.hx`) from those committed `.hx` sources via `assemble_k1.sh` at step `[0]` before it runs.

> **For AI agents:** the `/tmp/K*.bin` and `/tmp/*.ptx` files are **scratch**, rewritten on every run. The gate and the capstone audit `rm -f` each expected output **before** the run and assert it is non-empty **after**, so a stale `/tmp` file can never false-pass. If *you* re-use one of these paths, you must do the same `rm`-before / check-after, or you risk reading a previous run's artifact. Never assume a `/tmp/K2.bin` you find is the current one.

---

## Step 3 — the build/verify loop

If you change a compiler source (`helixc/bootstrap/{lexer,parser,kovc}.hx`), the loop is: **regenerate → re-mint → verify**, all driven by the gate. You do not invoke the individual rungs by hand for a routine compiler change; the gate orchestrates them.

1. **Edit** the source under `helixc/bootstrap/`.
2. **Run the gate** (`bash scripts/gate_kovc.sh`). It re-assembles the self-compile inputs from your edited sources (step `[0]`), re-derives the fixpoint from `seed`, re-mints the PTX driver, and re-runs the corpus and diagnostics.
3. **Read the verdict.** `GATE_PASS` means: the fixpoint still holds and the corpus has no regressions.

A critical, non-obvious invariant about the fixpoint hash:

> **For AI agents:** a legitimate compiler change **moves** the pinned fixpoint hash (the self-host source itself changed), but **K2 == K3 == K4 must stay byte-identical**. The gate pins `0992dddd…` for the *current released* compiler; if you change `kovc.hx` you will need to re-derive and re-pin that anchor with a recorded reason — the byte-identical 3-way equality is the load-bearing property, the pinned value additionally rejects a consistent-but-wrong output. Do not silently change a pinned hash; treat a hash change as a deliberate, documented event. Many features (closures, traits, wide fields) **do not** appear in the self-host source, so they keep the fixpoint byte-identical — see the per-feature notes in [scripts/gate_kovc.sh](../../../scripts/gate_kovc.sh).

### A note on `kovc`'s exit-status convention

When `kovc` self-compiles, it returns its **output byte count** as the process exit status (`rc = size mod 256`), so a successful self-compile exits **non-zero**. The gate validates those legs by **non-empty output + the byte-identical/pinned SHA**, never by `rc == 0`. Quoted from [scripts/gate_kovc.sh](../../../scripts/gate_kovc.sh) step `[2]`:

```bash
timeout 240 /tmp/K1.bin; rc=$?; echo "  K1->K2 run rc=$rc (kovc returns output-byte-count as exit status -> nonzero on success; validated by non-empty + SHA, NOT rc==0)"
```

> **For AI agents:** do **not** treat a non-zero exit from a `kovc` self-compile leg (K1→K2, K2→K3, K3→K4) as failure. The `seed → K1` leg is a C-compiled binary and **does** exit 0 on success, so `rc != 0` there *is* a failure. Only the `kovc`-self-compile legs use the byte-count convention.

---

## Step 4 — compile and run a `.hx` program

To compile a single Helix program from raw-rooted trust, drive the `seed` compiler directly. Its command line is `./seed.bin <source.hx> <output.bin>`, used verbatim by the gate (`./seed.bin k1src.hx /tmp/K1.bin`) and the capstone (`./seed.bin k1ptxdrv.hx /tmp/newdrv.bin`). The output is a standalone x86-64 ELF you run directly; verify by its **exit code**.

The simplest complete program in the tree is [helixc/examples/exit42.hx](../../../helixc/examples/exit42.hx):

**Verified example** — [helixc/examples/exit42.hx](../../../helixc/examples/exit42.hx) (compiles to a Linux ELF; `$? == 42`). This program is part of the gate's feature corpus, where it is compiled through the fresh K2 and asserted to exit `42` (`chk "$EX/exit42.hx" 42`):

```helix
// First end-to-end Helix program: compiles, links, runs, exits with status 42.
//
// Demonstrates: function decl, return, integer literal.
// Verification: $? == 42 after running the produced ELF.

fn main() -> i32 {
    42
}
```

The corpus uses the **K2** compiler to compile-and-run each program, and that is the pattern to mirror for checking a program against an expected exit code. The K-binaries read a **hardcoded** input path `/tmp/k2_in.hx` and write `/tmp/k2_out.bin` — so you stage the source into that path, run K2, then run the produced ELF. Quoted from the gate's corpus runner `chk()`:

```bash
chk() { local f="$1" exp="$2" b; b=$(basename "$1")
  [ -f "$f" ] || { echo "  MISSING $b"; fail=$((fail+1)); return; }
  cp "$f" /tmp/k2_in.hx; rm -f /tmp/k2_out.bin; timeout 30 /tmp/K2.bin >/dev/null 2>&1
  [ -s /tmp/k2_out.bin ] || { echo "  COMPILE-FAIL $b"; fail=$((fail+1)); return; }
  chmod +x /tmp/k2_out.bin; timeout 10 /tmp/k2_out.bin; local rc=$?
  if [ "$rc" = "$exp" ]; then echo "  PASS $b ($rc)"; pass=$((pass+1)); else echo "  FAIL $b ($rc!=$exp)"; fi
}
```

The three steps that matter: **(a)** `cp <src> /tmp/k2_in.hx` and `rm -f /tmp/k2_out.bin` (stage input, clear stale output); **(b)** run `/tmp/K2.bin` and assert `/tmp/k2_out.bin` is non-empty (compile succeeded); **(c)** `chmod +x /tmp/k2_out.bin` and run it, reading `$?` as the result. The deeper mechanics of `kovc`'s I/O paths and command line are the subject of the **Using `kovc`** chapter ([part2-setup-build/03-using-kovc.md](../part2-setup-build/03-using-kovc.md)); this chapter gives you the operator-level pattern the verification scripts actually use.

> **For AI agents:** to compile an *arbitrary* `.hx` program reproducibly, prefer the explicit-argument `seed` form `./seed.bin <in.hx> <out.bin>` — it takes its paths on the command line, so there is no hidden hardcoded I/O path to surprise you. The K-binary `/tmp/k2_in.hx → /tmp/k2_out.bin` convention is the gate's internal corpus harness; reuse it only when you replicate its `rm`-before / non-empty-after discipline. Verify a program by its **exit code**, the way every corpus row does — not by parsing stdout.

> **Note:** a **Verified example** in this book is a complete program (with an `fn main`) that was compiled — and run, where it has a defined exit code — before the chapter shipped, and cites the path it came from. A **Fragment** is a partial snippet and is not claimed to run on its own.

---

## Step 5 — reproduce the whole trust chain

When you need to establish trust from scratch — not just "the gate passes now" but "a clean checkout reproduces every pinned anchor from raw" — run the **one-command reproduction**. This is the canonical, push-button proof:

```bash
bash scripts/reproduce_trust.sh
```

On a CPU-only Linux host with no local state it does four things, in order, and exits `0` only if **every** check matches the pinned anchors:

- **`[1]` static fence** — exactly 1 committed `.py`, 24 committed `.c`/`.h`.
- **`[2]` from-raw ladder** — **deletes** every pre-built rung binary, then rebuilds `hex0 → … → seed` using *only the prior rung* (hex0 from hand-authored hex via `xxd`), each rung self-verifying its committed `.sha256`; the final `seed.bin` must equal `9837db12…`.
- **`[3]` self-host fixpoint** — runs the gate; asserts `GATE_PASS`, fixpoint `0992dddd…`, corpus `109/0`, check_err `4/0`.
- **`[4]` gcc diverse-double-compile** — `gcc` (zero M2-Planet ancestry) and the from-raw `seed` both produce a **byte-identical** K1 (`84363adb…`), a Wheeler trusting-trust defense.

The final verdict line is `REPRODUCE_TRUST: PASS` or `REPRODUCE_TRUST: FAIL`.

> **Warning:** `reproduce_trust.sh` is for a **clean checkout** (a CI runner or a throwaway clone). It **modifies the working tree** — it deletes the pre-built rung binaries and performs a disclosed `/mnt/c/Projects/Kovostov-Native → $ROOT` path rewrite (step `[0]`). Do **not** run it on a tree you want pristine. The header of [scripts/reproduce_trust.sh](../../../scripts/reproduce_trust.sh) says so explicitly.

This same script runs on a **fresh GitHub `ubuntu-latest` runner** — a different machine, a fresh clone, zero local state — via [.github/workflows/trust-reproduce.yml](../../../.github/workflows/trust-reproduce.yml), whose entire reproduce step is one line:

```yaml
      - name: Reproduce from-raw trust core (ladder + self-host fixpoint + gcc-DDC)
        run: bash scripts/reproduce_trust.sh
```

> **For AI agents:** `reproduce_trust.sh` is **CPU-only by design** — it never touches a GPU. The GPU capstone is verified **separately** by [scripts/capstone_audit.sh](../../../scripts/capstone_audit.sh) on a CUDA host. Do not expect, or claim, GPU results from a `reproduce_trust.sh` run.

### The GPU capstone (CUDA host only)

The real-capability proof is the *capstone*: a 2-layer transformer trained **end-to-end on `kovc`-emitted GPU (PTX) kernels**, whose loss curve matches an independent numpy oracle to within 2% (reproduced at ≈0%). One audit round is [scripts/capstone_audit.sh](../../../scripts/capstone_audit.sh):

```bash
bash scripts/capstone_audit.sh [round-label]
```

It re-runs the gate (minting `/tmp/newdrv.bin` fresh from the raw-binary `seed`), emits a single PTX module of **15 kovc-emitted transformer kernels**, runs the built-in finite-difference gradient check (`verify` mode), trains 500 Adam steps on the GPU, compares the loss curve against the oracle, and runs negative controls. It emits a final `CAPSTONE_AUDIT_PASS` or `CAPSTONE_AUDIT_FAIL` and propagates the verdict to the exit status.

Two operator-critical constraints from its header:

> **For AI agents:** the capstone audit is **GPU-serial** — *never invoke two of these at once*. It also **neutralizes the ambient environment** first: it unsets every `HX_*` variable (`HX_S`, `HX_D`, `HX_OPT`, `HX_TF32`, …) because a stray value would silently change the model dims or launch geometry and produce a different run masquerading as the v1.0 audit. If you script around it, do not set `HX_*` vars expecting them to carry into the audit; it clears them on purpose.

When you report capstone results, report them **honestly with their fraction**. GPU performance is a *fraction* of cuBLAS (~50–67.5% on the reference RTX 3070 Laptop, sm_86), **not parity**; the end-to-end speedup is **7.0–8.7×** (Amdahl-bound), **not ≥10×**. Loss parity — the hard correctness gate — holds at ≈0%. These numbers and their bounds are recorded in [docs/TRUST_CHAIN_CLOSED.md](../../../docs/TRUST_CHAIN_CLOSED.md) §2; treat that file as the **ceiling** on what you may claim, and never exceed it.

---

## Residual: the fixpoint layer is path-locked

A clean checkout reproduces the **from-raw ladder → seed** (the root of trust, including the gcc-DDC anchor) **from any directory** — the `stage0/*/build.sh` scripts use relative `../` paths. The **self-host fixpoint + GPU gate + capstone**, however, currently require the checkout to live at the canonical `/mnt/c/Projects/Kovostov-Native` path, because `assemble_k1.hx` hardcodes absolute paths for its reads and writes. `reproduce_trust.sh` handles this with a disclosed mechanical path rewrite (step `[0]`); a bare gate run from a non-canonical directory will read from / write to the canonical dir instead of your checkout.

This is a build-hygiene portability limitation, **not** a trust gap — the bytes produced are identical; only the *location* the build assumes is fixed. It is documented in full in [docs/CLEAN_REPRODUCTION.md](../../../docs/CLEAN_REPRODUCTION.md) ("Where it walls") and as a residual in [docs/TRUST_CHAIN_CLOSED.md](../../../docs/TRUST_CHAIN_CLOSED.md).

> **For AI agents:** if you must build outside `/mnt/c/Projects/Kovostov-Native`, drive [scripts/reproduce_trust.sh](../../../scripts/reproduce_trust.sh) (which rewrites the path mechanically for you) rather than running the gate by hand — otherwise the fixpoint legs will silently operate on the canonical directory, not your checkout. There is one more open increment, stated plainly in [docs/TRUST_CHAIN_CLOSED.md](../../../docs/TRUST_CHAIN_CLOSED.md) residual 10: reproduction by a fully independent third party on independent hardware. The CI workflow makes it push-button, but you should not represent same-machine evidence as independent reproduction.

---

## The canonical command set

The complete operator toolkit, every command quoted from the real scripts:

```bash
# Orient: static fence (committed-tree shape)
git ls-files "*.py" | wc -l          # expect 1  (verification/oracle/oracle_train.py)
git ls-files "*.c" "*.h" | wc -l     # expect 24

# Verify the toolchain self-hosts now (fixpoint + corpus + PTX text + diagnostics)
bash scripts/gate_kovc.sh            # match: ^GATE_PASS   (corpus 109/0, check_err 4/0)

# Compile + run one program from raw-rooted trust (explicit args)
./seed.bin <source.hx> <output.bin>  # then: chmod +x <output.bin>; ./<output.bin>; echo $?

# Reproduce the whole trust CORE from a clean checkout (modifies the tree; CPU-only)
bash scripts/reproduce_trust.sh      # match: REPRODUCE_TRUST: PASS

# Verify the GPU capstone (CUDA host only; GPU-serial — never two at once)
bash scripts/capstone_audit.sh [round-label]   # match: CAPSTONE_AUDIT_PASS
```

And the four ground-truth tokens/anchors to match against:

```text
GATE_PASS                  # gate_kovc.sh success
REPRODUCE_TRUST: PASS      # reproduce_trust.sh success
DDC_ANCHOR_OK              # gcc diverse-double-compile success
CAPSTONE_AUDIT_PASS        # capstone_audit.sh success

seed = 9837db12…           # from-raw C-subset compiler hash
K1   = 84363adb…           # gcc-DDC anchor
fix  = 0992dddd…           # self-host fixpoint (K2==K3==K4)
```

---

**Next:** [part9-for-ai-agents/02-non-negotiables.md](02-non-negotiables.md) — the hard rules: fail-closed verification, never fake an audit, never overclaim past the residuals, and the exact tokens and hashes you must never invent.
