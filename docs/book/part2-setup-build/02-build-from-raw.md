# Building from raw: the hex0 to seed to kovc ladder

*What this chapter covers:* how Helix is built with **no trusted pre-built compiler** — starting
from 299 bytes you can audit by hand, climbing one rung at a time to the `seed` C-subset compiler,
and finally minting `kovc` (the Helix compiler written in Helix). You will see the real per-rung
commands, the pinned rung hashes every rung self-verifies, why each rung is *deleted and rebuilt*
rather than trusted, and the one path-rewrite caveat you must know before you run it elsewhere.

This is the practical "do the build" chapter. The deeper *why* of each rung lives in Part VI; the
trusting-trust defense lives in Part VIII. Here we walk the ladder.

---

## The idea: trust nothing you did not rebuild

A normal compiler bootstraps from another compiler you already have — `gcc` from a previous `gcc`,
and so on back to a binary someone handed you. That binary is opaque. Ken Thompson's *Reflections
on Trusting Trust* showed a compiler binary can hide a self-perpetuating backdoor that never appears
in any source. Helix's answer is to refuse the opaque binary entirely: the **root of trust** is a
file of hand-authored hexadecimal you can read byte by byte, and *every* binary above it is rebuilt,
on your machine, by the rung directly below it.

That chain is **the from-raw ladder**:

```text
[299 hand-authored hex bytes]
       │  xxd -r -p   (audit-only; no assembler)
       ▼
   hex0       ── hex chars (stdin) → bytes (stdout); skips ws + ; / # comments
       │
       ▼
   hex1       ── adds single-char labels
       │
       ▼
   hex2       ── adds long labels + absolute addresses (acts as a linker)
       │
       ▼
   catm       ── file concatenation (catm OUT in1 in2 …); replaces cat/shell redirect
       │
       ▼
   M0         ── macro assembler: M1 assembly → hex2
       │
       ▼
   cc_amd64   ── minimal C compiler: C subset → M1
       │
       ▼
   M2-Planet  ── full self-hosting C compiler (last vendored rung)
       │
       ▼
   seed       ── the Apache-2.0 C-subset compiler WE wrote (seed.c); mints kovc
       │
       ▼
   kovc       ── the Helix compiler, written in Helix; self-hosts (K2==K3==K4)
```

Two authorship facts matter for trust, and the repo states them plainly in
[`stage0/README.md`](../../../stage0/README.md):

- **`hex0` is fully hand-authored from raw bytes** — "the literal 'raw binary as starting point'
  hard constraint." It is **frozen**; any change to it is a flag-the-user event.
- **`hex1` through `M2-Planet` are vendored** (audited sources from `oriansj/stage0-posix-amd64`,
  `oriansj/M2-Planet`, and `oriansj/M2libc` at pinned commits) — but **"no pre-built binary is ever
  trusted; every rung is rebuilt by the rung below it, reproducibly, and byte-audited against its
  annotated source."**
- **`seed` is the first original work** — `seed.c`, written in the M2-Planet C-subset under
  Apache-2.0 to avoid GPL-3.0 contagion. It is what replaces the historical Python frontend as the
  minter of `kovc`.

> **For AI agents:** "no pre-built binary is trusted" is operational, not rhetorical.
> [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) step `[2]` literally `rm -f`s
> every committed rung binary *before* rebuilding. If you script a verification, delete the rung
> `.bin`s first; a build that reuses an existing `*.bin` proves nothing.

---

## The per-rung contract

Every rung in `stage0/` ships, per [`stage0/README.md`](../../../stage0/README.md), the same four
artifacts:

- the **source** in auditable text form (annotated hex, or M0/M1 macro source, or C source),
- the **binary** (`.bin`),
- a **SHA-256 hash file** (`.sha256`),
- a **`build.sh`** that produces the binary from source *using only the previous stage's tools*.

And every `build.sh` obeys the same contract:

1. Build the rung's binary using **only the rung below it** (and earlier rungs already built).
2. Sanity-check it is a real `ELF 64-bit LSB executable` (via `file`).
3. Dump a disassembly for audit (`objdump`, audit-only).
4. **Verify the SHA-256 against the committed `.sha256`** — and *fail the build* on mismatch.
5. Run the rung's behavioral tests (`run_tests.sh`).

Step 4 is the reproducibility gate. Here is the exact check, copied from
[`stage0/hex1/build.sh`](../../../stage0/hex1/build.sh) — it is the same shape in every rung:

```bash
# 4. SHA-256 reproducibility check
NEW=$(sha256sum "$OUT")
if [ -f hex1.sha256 ]; then
    if [ "$(cat hex1.sha256)" = "$NEW" ]; then
        echo "SHA-256: ${NEW%% *}  (matches hex1.sha256)"
    else
        echo "ERROR: SHA-256 mismatch (non-reproducible build)" >&2
        echo "  recorded: $(cat hex1.sha256)" >&2
        echo "  rebuilt:  $NEW" >&2
        exit 1
    fi
else
    echo "$NEW" > hex1.sha256
    echo "SHA-256 recorded: ${NEW%% *}"
fi
```

> **For AI agents:** a rung `build.sh` is **fail-closed** on the hash. A non-zero exit from any
> `stage0/<rung>/build.sh` means the rebuilt bytes diverged from the committed `.sha256` — treat it
> as a hard failure, never a warning. Do not delete or regenerate a committed `.sha256` to make a
> build "pass"; the `else` branch above only records a hash when none exists, and in the committed
> tree they all exist.

The pinned hashes the ladder must reproduce (from the committed `.sha256` files, and matching the
clean-room re-derivation in [`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md)):

| # | Rung | Built by | Bytes | committed `.bin` SHA-256 |
|---|------|----------|-------|---------------------------|
| 1 | `hex0` | `xxd -r -p` of `hex0.hex` (raw) | 299 | `cc1d1741…e417c125` |
| 2 | `hex1` | hex0 | 622 | `c264a212…f45f1cf8` |
| 3 | `hex2` | hex1 | 1519 | `6c69c7e6…a82b33bc` |
| 4 | `catm` | hex2 | 299 | `911d19bf…07e83ce7` |
| 5 | `M0` | catm + hex2 | 1684 | `db97dff1…f1ec2bd1` |
| 6 | `cc_amd64` | M0 + catm + hex2 | 17976 | `ea0054d1…242baaa9` |
| 7 | `M2-Planet` | cc_amd64 + catm + M0 + hex2 | 200561 | `724b9e2d…c860a91925` |
| 8 | `seed` | M2-Planet + catm + M0 + hex2 | 62467 | `9837db12…b915c9bb` |

The full pinned `seed` hash — the gate of the whole ladder — is
`9837db12752a22159ca75a533910bc0d7b9afb35df9b9963f256b7b1b915c9bb`. It is hardcoded as the release
anchor `SEED_SHA` in [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh).

---

## Rung 1 — hex0, from hand-authored hex

`hex0` is the only "raw" rung. There is no compiler under it; its bytes *are* the source. The build
strips comments and whitespace from the annotated `hex0.hex` and runs the result through `xxd -r -p`
— a hex-to-binary converter, explicitly **audit-only, no assembler involved**. From
[`stage0/hex0/build.sh`](../../../stage0/hex0/build.sh):

```bash
OUT=hex0.bin
SRC=hex0.hex

# 1. Hex source -> binary
grep -v '^;' "$SRC" | sed 's/;.*//' | tr -d '[:space:]' | xxd -r -p > "$OUT"
chmod +x "$OUT"
```

The header of that script lists the only tools it uses, and the tools it pointedly does **not**:

```bash
# Tools used (all permitted by project's hard constraint):
#   xxd      — audit-only, hex<->bin only
#   file     — audit-only, header sniffing
#   objdump  — audit-only, disassembly
#   sha256sum — integrity
#
# NOT used: nasm, as, gcc, ld, clang. The bytes in hex0.hex are the source of truth.
```

That is the crux of the raw-binary claim: between `hex0.hex` and `hex0.bin` there is no assembler,
no linker, no other compiler — only a deterministic hex decode. Run it:

```bash
cd stage0/hex0 && bash build.sh
```

It builds `hex0.bin` (299 bytes), confirms it is an `ELF 64-bit LSB executable`, writes
`disasm.txt`, and verifies the SHA-256 against `hex0.sha256` (`cc1d1741…e417c125`).

> **Note:** the ladder targets **linux-x86_64 ELF**. On Windows this runs under WSL2, as the build
> records throughout. See [Prerequisites](01-prerequisites.md) for the toolchain
> (`xxd`, `gcc`, `objdump`, `file`, `sha256sum`).

---

## Rungs 2–3 — hex1 and hex2, built only by the rung below

From here up, each rung is built by feeding its vendored source through the binary you produced one
step earlier. `hex1` is decoded by **`hex0`**; the command is just the prior rung consuming the next
source. From [`stage0/hex1/build.sh`](../../../stage0/hex1/build.sh):

```bash
HEX0=../hex0/hex0.bin
SRC=hex1_AMD64.hex0
OUT=hex1.bin

# 1. Build: hex0 decodes the hex1 source (hex pairs + '#'/';' comments) -> hex1.bin
"$HEX0" < "$SRC" > "$OUT"
```

`hex1` adds single-character labels; `hex2` adds long labels and absolute addresses (it acts as a
linker). `hex2` is in turn built **only by `hex1`**, from
[`stage0/hex2/build.sh`](../../../stage0/hex2/build.sh):

```bash
HEX1=../hex1/hex1.bin
SRC=hex2_AMD64.hex1
OUT=hex2.bin

# 1. Build: hex1 assembles the hex2 source (hex + single-char labels) -> hex2.bin
"$HEX1" "$SRC" "$OUT"
```

Each then runs the same `file`/`objdump`/`sha256sum` checks. `hex1.bin` must match
`c264a212…f45f1cf8`; `hex2.bin` must match `6c69c7e6…a82b33bc`.

Notice the invocation *form* changes as the rungs gain capability: `hex0` reads stdin and writes
stdout (`< SRC > OUT`); `hex1` and `hex2` take filename arguments (`SRC OUT`). Use the exact form
from each rung's `build.sh` — do not assume one calling convention across the ladder.

---

## Rung 4 — catm, the shell-free concatenator

`catm` replaces `cat` and shell redirection so the higher rungs never depend on shell plumbing for
correctness. It is built **only by `hex2`**. From [`stage0/catm/build.sh`](../../../stage0/catm/build.sh):

```bash
HEX2=../hex2/hex2.bin
SRC=catm_AMD64.hex2
OUT=catm.bin

# 1. Build: hex2 assembles the catm source -> catm.bin
"$HEX2" "$SRC" "$OUT"
```

It is 299 bytes and must match `911d19bf…07e83ce7`. From here, builds use `catm OUT in1 in2 …` to
stitch sources and ELF headers together instead of `cat a b > c`.

---

## Rung 5 — M0, the macro assembler

`M0` turns M1 assembly (mnemonics, named registers, macros) into `hex2`. This is the first rung that
needs *two* prior rungs together: `catm` to assemble the input, then `hex2` to produce the binary.
The build follows the canonical mescc-tools phase-3 sequence — `catm` prepends the ELF header to the
M0 program, then `hex2` assembles the result. From [`stage0/M0/build.sh`](../../../stage0/M0/build.sh):

```bash
CATM=../catm/catm.bin
HEX2=../hex2/hex2.bin
OUT=M0.bin

# 1. Build (canonical mescc-tools phase-3): catm prepends the ELF header to the
#    M0 program, then hex2 assembles the result.
T=$(mktemp -d)
"$CATM" "$T/M0.hex2" ELF-amd64.hex2 M0_AMD64.hex2
"$HEX2" "$T/M0.hex2" "$OUT"
rm -rf "$T"
```

`M0.bin` is 1684 bytes and must match `db97dff1…f1ec2bd1`.

---

## Rung 6 — cc_amd64, the minimal C compiler

`cc_amd64` compiles a small C subset to M1, and it is what bootstraps the full C compiler above it.
It is built by **`M0` + `catm` + `hex2`** (mescc-tools phase-4). From
[`stage0/cc_amd64/build.sh`](../../../stage0/cc_amd64/build.sh):

```bash
M0=../M0/M0.bin
CATM=../catm/catm.bin
HEX2=../hex2/hex2.bin
OUT=cc_amd64.bin

# 1. Build (mescc-tools phase-4): M0 assembles cc_amd64.M1 -> hex2, catm
#    prepends the ELF header, hex2 assembles the binary.
T=$(mktemp -d)
"$M0"   cc_amd64.M1 "$T/cc.hex2"
"$CATM" "$T/cc_full.hex2" ELF-amd64.hex2 "$T/cc.hex2"
"$HEX2" "$T/cc_full.hex2" "$OUT"
rm -rf "$T"
```

`cc_amd64.bin` is 17976 bytes and must match `ea0054d1…242baaa9`.

---

## Rung 7 — M2-Planet, the full C compiler

`M2-Planet` is the full self-hosting C compiler and the **last vendored rung**. It is built by
**`cc_amd64` + `catm` + `M0` + `hex2`** (mescc-tools-mini-kaem phase-5). The build `catm`s the
bootstrap libc plus all the M2-Planet C sources into one translation unit, compiles it with
`cc_amd64`, then assembles and links it. From
[`stage0/M2-Planet/build.sh`](../../../stage0/M2-Planet/build.sh):

```bash
# 1. Build (mescc-tools-mini-kaem.kaem Phase-5, exact source order):
#    catm concatenates the bootstrap libc + all M2-Planet sources into one .c,
#    cc_amd64 compiles it to M1, catm prepends cc_amd64's defs, M0 assembles to
#    hex2, catm prepends the ELF header, hex2 links the binary.
T=$(mktemp -d)
"$CATM" "$T/M2-0.c" \
    M2libc/amd64/linux/bootstrap.c \
    M2libc/bootstrap.c \
    M2-Planet/cc.h \
    M2libc/bootstrappable.c \
    M2-Planet/cc_globals.c \
    M2-Planet/cc_reader.c \
    M2-Planet/cc_strings.c \
    M2-Planet/cc_types.c \
    M2-Planet/cc_emit.c \
    M2-Planet/cc_core.c \
    M2-Planet/cc_macro.c \
    M2-Planet/cc.c
"$CC"   "$T/M2-0.c"      "$T/M2-0.M1"
"$CATM" "$T/M2-0-0.M1"   "$DEFS" "$LIBC" "$T/M2-0.M1"
"$M0"   "$T/M2-0-0.M1"   "$T/M2-0.hex2"
"$CATM" "$T/M2-0-0.hex2" "$ELF" "$T/M2-0.hex2"
"$HEX2" "$T/M2-0-0.hex2" "$OUT"
rm -rf "$T"
```

`M2.bin` is 200561 bytes and must match `724b9e2d…c860a91925`.

> **Note (honest residual):** `stage0/README.md` records that M2-Planet's *core capability* is
> tested (compiles C → runs → correct exit), but its **own** self-host fixpoint (M2 rebuilding M2
> byte-stably) "is investigated but not yet holding — see `M2-Planet/PROVENANCE.md`; left open and
> honest, not faked." This does not affect the ladder: M2-Planet is used to build the `seed`, and
> the trusting-trust gap created by trusting M2-Planet is closed independently by the **gcc-DDC**
> (covered below and in Part VIII), not by M2-Planet self-hosting.

---

## Rung 8 — seed, the first original artifact

`seed` is where the vendored ladder ends and Helix's own code begins. `seed.c` is the Apache-2.0
C-subset compiler the project wrote to mint `kovc` from the frozen Helix sources — written in the
M2-Planet C-subset so it can be built by rung 7 while keeping a clean Apache-2.0 license boundary
away from the GPL-3.0 vendored tools. It is built by **`M2-Planet` + `catm` + `M0` + `hex2`**. From
[`stage0/helixc-bootstrap/build.sh`](../../../stage0/helixc-bootstrap/build.sh):

```bash
# 1. M2-Planet compiles seed.c (+ the bootstrap libc) to M1 assembly.
T=$(mktemp -d)
"$M2" --architecture amd64 \
    -f "$LIB/amd64/linux/bootstrap.c" \
    -f "$LIB/bootstrap.c" \
    -f "$LIB/bootstrappable.c" \
    -f seed.c \
    --bootstrap-mode -o "$T/seed.M1"

# 2. Assemble M2's output with M2libc's amd64 defs (lesson 30: M2 output pairs
#    with M2libc/amd64 defs, NOT cc_amd64's), then link to a self-contained ELF.
"$CATM" "$T/seed-0.M1"   "$MDEFS" "$MLIBC" "$T/seed.M1"
"$M0"   "$T/seed-0.M1"   "$T/seed.hex2"
"$CATM" "$T/seed-0.hex2" "$MELF" "$T/seed.hex2"
"$HEX2" "$T/seed-0.hex2" "$OUT"
rm -rf "$T"
```

`seed.bin` is 62467 bytes and must match the full pinned anchor
`9837db12752a22159ca75a533910bc0d7b9afb35df9b9963f256b7b1b915c9bb`. It passes 17/17 of its own
tests. With this rung, the from-raw root of trust is complete: 299 hand-typed bytes have, through
seven deterministic rebuilds, produced a working C-subset compiler whose every intermediate byte is
pinned.

> **Note:** `seed.bin` is **gitignored and not tracked** — the committed tree carries the *source*
> (`seed.c`) but not the pre-built binary, so the seed *must* be re-derived from raw. The fence step
> in [`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md) confirms this:
> `git ls-files stage0/helixc-bootstrap/seed.bin` is empty, and `git check-ignore` points at
> `stage0/helixc-bootstrap/.gitignore`.

---

## From seed to kovc: the self-host fixpoint

The `seed` is not the product — `kovc` is. The seed's job is to compile the frozen Helix compiler
sources (`helixc/bootstrap/{lexer,parser,kovc}.hx`) into the first native Helix compiler, **K1**,
and from there the chain proves `kovc` reproduces *itself*:

```text
seed → K1 → K2 → K3 → K4,   with  K2 == K3 == K4  (byte-identical)
```

That equality is **the self-host fixpoint**. Once K2, K3, and K4 are byte-for-byte identical, you
have a fixed point: the Helix compiler, compiling its own source, lands on exactly the same bytes
forever after. The pinned values, from [`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md):

| Stage | Built by | SHA-256 |
|-------|----------|---------|
| **K1** | `seed` → K1 | `84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba` |
| **K2** | K1 → K2 | `0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f` |
| **K3** | K2 → K3 | == K2 (byte-identical) |
| **K4** | K3 → K4 | == K3 (byte-identical) |

The fixpoint is verified by **the gate**,
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh), which also runs the 109-program feature
corpus, the GPU-PTX text regression, and the negative-diagnostic corpus, and prints `GATE_PASS` on
success. Building and verifying `kovc` end-to-end (using `kovc`, running the gate) is the subject of
the next two chapters; here the point is only that the *output* of the from-raw ladder, the `seed`,
is what makes that self-host fixpoint reachable from hand-typed bytes.

> **For AI agents:** the fixpoint anchors are exact tokens, not prose. Key off the pinned strings —
> K1 `84363adb…`, the fixpoint `0992dddd…` — and the literal `GATE_PASS` line, exactly as
> [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) does with `grep -q '^GATE_PASS'`.

---

## Why delete, rebuild, and compare

It is worth restating the discipline, because it is the whole point. A build that *reuses* a
committed `*.bin` proves nothing — the binary could be the very backdoored artifact you are trying
to rule out. So the clean reproduction **removes the pre-built outputs first**, then rebuilds, then
compares the rebuilt bytes to the committed hashes. [`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md)
frames the claim precisely: not "the numbers were once produced," but "a clean checkout reproduces
them today, from committed source, with the pre-built outputs removed first."

The automated path does exactly this. [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh)
step `[2]` deletes every rung binary, then loops the rungs in order, building each from the prior and
checking the seed against the pinned anchor:

```bash
rm -f stage0/hex0/hex0.bin stage0/hex1/hex1.bin stage0/hex2/hex2.bin stage0/catm/catm.bin \
      stage0/M0/M0.bin stage0/cc_amd64/cc_amd64.bin stage0/M2-Planet/M2.bin \
      stage0/helixc-bootstrap/seed.bin
LADDER_OK=1
for rung in hex0 hex1 hex2 catm M0 cc_amd64 M2-Planet helixc-bootstrap; do
  if ( cd "stage0/$rung" && bash build.sh ) >"/tmp/rt_${rung}.log" 2>&1; then
    say "    rung $rung : build + self-verify OK"
  else
    bad "rung $rung build/verify failed (tail of /tmp/rt_${rung}.log):"; tail -8 "/tmp/rt_${rung}.log" >&2
    LADDER_OK=0; break
  fi
done
if [ "$LADDER_OK" = "1" ] && [ -s stage0/helixc-bootstrap/seed.bin ]; then
  GOT=$(sha256sum stage0/helixc-bootstrap/seed.bin | cut -d' ' -f1)
  if [ "$GOT" = "$SEED_SHA" ]; then say "    seed.bin == pinned ($SEED_SHA)"; else bad "seed.bin $GOT != pinned $SEED_SHA"; fi
else
  bad "ladder did not produce seed.bin"
fi
```

A clean re-derivation of this exact ladder is recorded in
[`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md) Step 2: every rung **MATCH**, seed
`9837db12` **MATCH**, `LADDER_ALL_GREEN` in about 3m44s on a single serial builder. Notably, the
same document reports the ladder also rebuilds byte-identically inside a fresh `git clone` to a
*different* directory — the `stage0/*/build.sh` scripts use relative `../` paths, so the from-raw
ladder is path-portable.

---

## The one-command reproduction

You do not have to walk the rungs by hand. The whole from-raw core — fence, ladder, self-host
fixpoint, and the gcc diverse-double-compile — runs from a single command:

```bash
bash scripts/reproduce_trust.sh
```

It exits 0 **only if** every check matches the pinned anchors (`seed=9837db12…`, `K1=84363adb…`,
fixpoint `0992dddd…`), printing `REPRODUCE_TRUST: PASS`. The script's own header states its scope:
it is **CPU-only** and runnable on any x86-64 Linux (including a CI runner) with no local state; the
GPU capstone is verified separately by `scripts/capstone_audit.sh` on a CUDA host. Reproducing and
reading that verdict in full is the job of [Reproduce & verify the trust chain](04-reproduce-verify-trust.md).

> **Warning:** the script header is explicit that it **modifies the working tree** (the path rewrite
> below, plus the rung-binary rebuilds): *"do not run on a tree you want pristine."* Run it on a
> clean checkout or a throwaway clone, not on a working directory with edits you care about.

---

## The path-rewrite caveat (read before running elsewhere)

There is one honest portability limitation, and it sits *above* the ladder, in the self-host
fixpoint layer — not in the from-raw ladder itself.

The fixpoint's source concatenator, `stage0/helixc-bootstrap/assemble_k1.hx`, **hardcodes absolute
`/mnt/c/Projects/Kovostov-Native/...` paths** for the files it reads and writes.
[`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md) ("Where it walls") documents the
consequence precisely: the **from-raw ladder → seed** reproduces from *any* directory (its scripts
use relative paths), but the **self-host fixpoint + gate + capstone** currently require the checkout
to live at the canonical `/mnt/c/Projects/Kovostov-Native` path, because that file hardcodes it. The
trust doc is careful to call this what it is:

> This is a build-hygiene portability limitation … **not** a trust gap in the chain — the bytes
> produced are identical; only the *location* the build assumes is fixed.

`scripts/reproduce_trust.sh` handles this for you. Its step `[0]` performs a **pure mechanical path
swap** — rewriting `/mnt/c/Projects/Kovostov-Native` to wherever the checkout actually lives — so the
build runs at any path:

```bash
say "[0] path rewrite  /mnt/c/Projects/Kovostov-Native -> $ROOT"
mapfile -t HCFILES < <(grep -rlI '/mnt/c/Projects/Kovostov-Native' . 2>/dev/null || true)
if [ "${#HCFILES[@]}" -gt 0 ]; then
  printf '%s\n' "${HCFILES[@]}" | xargs sed -i "s#/mnt/c/Projects/Kovostov-Native#$ROOT#g"
  say "    rewrote ${#HCFILES[@]} file(s)"
fi
```

This is *output-determinism* under a documented, controlled rewrite — the same fixpoint
`0992dddd…` results — and is explicitly **not** the same as run-from-arbitrary-checkout portability,
which the caveat above spells out. The one open increment, per the trust docs, is **external
third-party reproduction on independent hardware**.

> **For AI agents:** if you build the fixpoint outside `/mnt/c/Projects/Kovostov-Native`, either run
> the whole thing through `scripts/reproduce_trust.sh` (which does the rewrite for you) or apply the
> same `sed` swap first. Running `assemble_k1.sh` from a non-canonical directory **silently reads
> from and writes to the canonical dir** (it returns rc=0 but produces no files in your checkout),
> exactly as [`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md) records — so a
> "success" there can be a false positive. Do not treat a non-canonical fixpoint run as verified.

---

## What you have after the ladder

When `scripts/reproduce_trust.sh` prints `REPRODUCE_TRUST: PASS`, you have established, from a clean
checkout and 299 hand-authored bytes:

- the **from-raw ladder** `hex0 → hex1 → hex2 → catm → M0 → cc_amd64 → M2-Planet → seed`, each rung
  built only by the rung below it and self-verifying its committed `.sha256`, with the seed at the
  pinned `9837db12…`;
- the **self-host fixpoint** `seed → K1 → K2 → K3 → K4` with `K2 == K3 == K4` at `0992dddd…` (K1 at
  `84363adb…`), the corpus at 109/0, and `GATE_PASS`;
- the **gcc diverse-double-compile**, in which `gcc` — a toolchain with zero M2-Planet ancestry —
  and the from-raw `seed` both produce a byte-identical K1 (`84363adb…`), the trusting-trust defense.

The chain is closed on the CPU side, all the way down from raw binary. For the calibrated statement
of what it does **not** yet cover — the GPU path is complete *to PTX, not to GPU machine code*; GPU
performance is a fraction of cuBLAS, not parity; a single hardware target; and external third-party
reproduction as the one remaining increment — see
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md). This book does not claim past
those residuals.

---

**Next:** [Using kovc](03-using-kovc.md) — now that the ladder has minted a Helix compiler, how to
compile and run your own `.hx` programs with `kovc`.
