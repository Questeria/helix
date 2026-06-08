# The MESCC-lineage rungs to `seed`

*What this chapter covers:* the seven build rungs that climb from `hex0`'s 299
hand-authored bytes up to the `seed` — `hex1`, `hex2`, `catm`, `M0`, `cc_amd64`,
`M2-Planet` (the MESCC bootstrap lineage), then the `seed` itself, compiled from
our own `seed.c`. For each rung: what it is, what it does, the rule that it is
built **only by the rung below it** and self-verifies its committed `.sha256`,
and the pinned per-rung hash and byte size. This is the middle of
[the from-raw ladder](../part2-setup-build/02-build-from-raw.md); the bottom
(`hex0`) is the previous chapter, and the top (`seed → kovc`) is the next.

---

## Where we are on the ladder

The previous chapter established the root: `hex0`, 299 bytes you can audit one
byte at a time, hand-authored and frozen. It is the only rung whose bytes are not
produced by another program — everything above it is *built*, never trusted
pre-built. This chapter walks the chain from there to the point where original
Helix work begins:

```text
[299 hand-encoded bytes]                  ← previous chapter
       │
       ▼
   hex0   ── hex chars → bytes (frozen root)
       │
       ▼
   hex1   ── hex0 builds it; adds single-char labels
       │
       ▼
   hex2   ── hex1 builds it; long labels + absolute addresses (a linker)
       │
       ▼
   catm   ── hex2 builds it; file concatenation (replaces cat/shell redirect)
       │
       ▼
   M0     ── catm+hex2 build it; macro assembler (M1 assembly → hex2)
       │
       ▼
   cc_amd64 ── M0+catm+hex2 build it; minimal C-subset compiler (C → M1)
       │
       ▼
   M2-Planet ── cc_amd64+catm+M0+hex2 build it; full self-hosting C compiler
       │
       ▼
   seed   ── M2-Planet builds it; the Apache-2.0 C-subset compiler that compiles Helix
       │
       ▼
   kovc   ── the Helix compiler the seed mints                   ← next chapter
```

This is the diagram in [`stage0/README.md`](../../../stage0/README.md), the
canonical map of the chain. Rungs `hex1` through `M2-Planet` are **vendored**:
their *source* is pulled from canonical, community-audited upstreams
(`oriansj/stage0-posix-amd64`, `oriansj/M2-Planet`, `oriansj/M2libc`) at pinned
commits, and each rung is **rebuilt by the rung below it** — no upstream-built
binary is ever trusted. The `seed` (rung 8) is the first **original** rung: our
own `seed.c`, Apache-2.0, written in the C subset `M2-Planet` accepts.

> **For AI agents:** the canonical one-command rebuild of this whole span is
> [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) step `[2]`.
> It `rm -f`s every pre-built rung binary, then loops `for rung in hex0 hex1
> hex2 catm M0 cc_amd64 M2-Planet helixc-bootstrap` calling each
> `stage0/<rung>/build.sh`, and asserts the final `seed.bin` SHA equals the
> pinned `SEED_SHA=9837db12…`. Drive that script; do not hand-run rungs out of
> order — each `build.sh` reads the binary of the rung below via a relative
> `../` path and will fail if that rung has not been built.

---

## The two invariants every rung obeys

Before the per-rung detail, the two properties that make this a *trust* ladder
rather than just a build. Both are enforced mechanically inside each rung's
`stage0/<rung>/build.sh`, so they hold on every rebuild, not just the once.

**Invariant 1 — built only by the prior rung.** Each `build.sh` names the
binaries it consumes by relative path to lower rungs and uses *nothing else* — no
system assembler, no system compiler, no pre-built copy of itself. For example,
[`stage0/hex2/build.sh`](../../../stage0/hex2/build.sh) opens with exactly one
tool reference:

**Fragment** (from [`stage0/hex2/build.sh`](../../../stage0/hex2/build.sh); shows
the prior-rung-only rule — `hex2` is built solely by `../hex1/hex1.bin`):

```bash
HEX1=../hex1/hex1.bin
SRC=hex2_AMD64.hex1
OUT=hex2.bin

[ -x "$HEX1" ] || chmod +x "$HEX1"

# 1. Build: hex1 assembles the hex2 source (hex + single-char labels) -> hex2.bin
"$HEX1" "$SRC" "$OUT"
```

**Invariant 2 — self-verifies its committed `.sha256`.** After building, each
`build.sh` re-derives the SHA-256 of the output it just produced and compares it
to the rung's committed `.sha256` file; a mismatch is a hard, non-zero exit. This
is the line that turns "it built" into "it built *the exact committed bytes*":

**Fragment** (from [`stage0/hex2/build.sh`](../../../stage0/hex2/build.sh); the
reproducibility gate — identical in shape across every rung's `build.sh`):

```bash
# 4. SHA-256 reproducibility check
NEW=$(sha256sum "$OUT")
if [ -f hex2.sha256 ]; then
    if [ "$(cat hex2.sha256)" = "$NEW" ]; then
        echo "SHA-256: ${NEW%% *}  (matches hex2.sha256)"
    else
        echo "ERROR: SHA-256 mismatch (non-reproducible build)" >&2
        echo "  recorded: $(cat hex2.sha256)" >&2
        echo "  rebuilt:  $NEW" >&2
        exit 1
    fi
else
    echo "$NEW" > hex2.sha256
    echo "SHA-256 recorded: ${NEW%% *}"
fi
```

Each `build.sh` also runs an ELF sanity check (`file … | grep -q "ELF 64-bit LSB
executable"`), writes an audit-only disassembly via `objdump`, and finishes with
`bash run_tests.sh`. The chain targets **linux-x86_64 ELF**, run under WSL2 on
Windows.

> **For AI agents:** the literal failure token to watch for is `SHA-256 mismatch
> (non-reproducible build)` on stderr followed by `exit 1`. A rung that prints
> `SHA-256: <hash>  (matches <rung>.sha256)` reproduced byte-for-byte. Never
> treat a clean ELF check or passing test as sufficient — the `.sha256` match is
> the load-bearing assertion.

---

## The pinned per-rung anchors

Every binary in this span is pinned: a fixed byte count and a fixed SHA-256,
committed in `stage0/<rung>/<rung>.sha256`. These values are also tabulated in
[`stage0/README.md`](../../../stage0/README.md) and re-derived from scratch in
[`docs/CLEAN_REPRODUCTION.md`](../../CLEAN_REPRODUCTION.md) step 2, where a clean
checkout (pre-built binaries removed first) reproduced each one byte-identically.

| # | Rung | Built by | Bytes | SHA-256 |
|---|------|----------|-------|---------|
| 2 | `hex1` | `hex0` | 622 | `c264a212d2b0e1f1bcf34217ed7876bb9324bd7e29cd902bb1cad4d9f45f1cf8` |
| 3 | `hex2` | `hex1` | 1519 | `6c69c7e60df220e884de4fc3bdf7137352b7b3c25a1fb7000ef7f7dea82b33bc` |
| 4 | `catm` | `hex2` | 299 | `911d19bff7be2bc4657b312b19c29ad98cbaad2fed141a016fa0104e07e83ce7` |
| 5 | `M0` | `catm`+`hex2` | 1684 | `db97dff12dbbc1f547b5fb58fe70267ac9a99d43d5879d8bbf578f31f1ec2bd1` |
| 6 | `cc_amd64` | `M0`+`catm`+`hex2` | 17976 | `ea0054d18301701b4c11a486ace94ff2045356c9fac9f616af339051242baaa9` |
| 7 | `M2-Planet` | `cc_amd64`+`catm`+`M0`+`hex2` | 200561 | `724b9e2d60050c4308fd9c8780b5d83338a5a9d0784e8d5290e161c860a91925` |
| 8 | `seed` | `M2-Planet`+`catm`+`M0`+`hex2` | 62467 | `9837db12752a22159ca75a533910bc0d7b9afb35df9b9963f256b7b1b915c9bb` |

(`hex0`, rung 1, is 299 bytes, SHA `cc1d1741…`; it is the previous chapter's
subject and the root of this table.) The `seed` hash `9837db12…` is one of the
three release anchors hard-coded as `SEED_SHA` in
[`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) — the value an
independent run must reproduce for the ladder leg to pass.

---

## Rung 2 — `hex1`: hex with single-character labels

`hex0` decodes hex digit pairs into bytes and skips whitespace and `;`/`#`
comments — enough to type a program in hex, but with no way to refer to an
address symbolically. `hex1` adds exactly one capability on top: **single-character
labels**, so a jump target can be named rather than counted out by hand.

Its source, `hex1_AMD64.hex0`, is GPL-3.0, vendored from
`oriansj/stage0-posix-amd64`. It is built by feeding that source through `hex0` on
stdin:

**Fragment** (from [`stage0/hex1/build.sh`](../../../stage0/hex1/build.sh); `hex0`
decodes the `hex1` source on stdin):

```bash
HEX0=../hex0/hex0.bin
SRC=hex1_AMD64.hex0
OUT=hex1.bin

# 1. Build: hex0 decodes the hex1 source (hex pairs + '#'/';' comments) -> hex1.bin
"$HEX0" < "$SRC" > "$OUT"
```

Result: `hex1.bin`, **622 bytes**, SHA `c264a212…`, verified against
`hex1.sha256`, 2/2 tests pass.

---

## Rung 3 — `hex2`: long labels, absolute addresses, a linker

`hex2` is the first rung that does real linking. It extends `hex1` with
**multi-character labels and absolute-address resolution** — it computes where
labels land and patches the references — which makes it, in effect, the chain's
first linker. Every rung above it that produces an ELF leans on `hex2` to stitch
the final binary together.

It is built by `hex1` (see the Fragment under Invariant 1 above). Source
`hex2_AMD64.hex1`, GPL-3.0, vendored. Result: `hex2.bin`, **1519 bytes**, SHA
`6c69c7e6…`, 2/2 tests pass.

> **For AI agents:** this rung-3 `hex2` is the **positional** linker (fixed
> argument order, no flags). It is a different program from the flag-driven
> `hex2` in `stage0/hex2-linker/` (the mescc-tools linker discussed below), which
> takes `--base-address` / `--architecture`. Do not conflate them — the
> rung-3 `hex2.bin` is the one in the trust ladder; the flag-driven one is an
> auxiliary verifier.

---

## Rung 4 — `catm`: concatenation without a shell

The upper rungs assemble a binary in stages — prepend an ELF header, append a
debug footer, stitch units together — and they cannot assume a Unix shell with
`cat` or `>` redirection is available *inside the trusted chain*. `catm`
("cat multiple") is the rung that provides that: `catm OUT in1 in2 …` writes the
concatenation of the inputs to `OUT`. It replaces `cat` / shell redirection so the
build steps above it stay inside the bootstrapped toolchain.

It is built by `hex2` from the vendored `catm_AMD64.hex2` (GPL-3.0). Result:
`catm.bin`, **299 bytes** (the same size as `hex0`, coincidentally), SHA
`911d19bf…`, 2/2 tests pass. From here up, `catm` shows up in nearly every
`build.sh` as the tool that prepends ELF headers and concatenates source units.

---

## Rung 5 — `M0`: the macro assembler

`M0` is the **macro assembler**: it turns M1 assembly — mnemonics, named
registers, macros — into the `hex2` input format. It is the rung that lets the
chain stop hand-encoding instruction bytes and start writing assembly. Its build
is the first to use *two* lower rungs together: `catm` prepends an ELF header to
the `M0` program, then `hex2` assembles the result:

**Fragment** (from [`stage0/M0/build.sh`](../../../stage0/M0/build.sh); the
canonical mescc-tools phase-3 recipe — `catm` then `hex2`):

```bash
CATM=../catm/catm.bin
HEX2=../hex2/hex2.bin
OUT=M0.bin

# 1. Build (canonical mescc-tools phase-3): catm prepends the ELF header to the
#    M0 program, then hex2 assembles the result.
T=$(mktemp -d)
"$CATM" "$T/M0.hex2" ELF-amd64.hex2 M0_AMD64.hex2
"$HEX2" "$T/M0.hex2" "$OUT"
```

Source `M0_AMD64.hex2` (GPL-3.0, vendored). Result: `M0.bin`, **1684 bytes**, SHA
`db97dff1…`, 2/2 tests pass.

---

## Rung 6 — `cc_amd64`: a minimal C compiler

`cc_amd64` is the first compiler in the chain that accepts **C** — a small C
subset — and emits M1 assembly. It is the rung that bootstraps the real C compiler
above it: its only job in this ladder is to compile `M2-Planet` into existence.
Its build runs the mescc-tools phase-4 recipe across three lower rungs — `M0`
assembles `cc_amd64.M1` to `hex2`, `catm` prepends the ELF header, `hex2` links:

**Fragment** (from [`stage0/cc_amd64/build.sh`](../../../stage0/cc_amd64/build.sh);
phase-4 — `M0` → `catm` → `hex2`):

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
```

Source `cc_amd64.M1` (GPL-3.0, vendored). Result: `cc_amd64.bin`, **17976
bytes**, SHA `ea0054d1…`, 2/2 tests pass.

> **Note:** `cc_amd64` ships only the minimal `libc-core.M1`; it is a deliberately
> weak compiler. It is strong enough to compile `M2-Planet`'s C and no more.
> Anything that needs richer libc — `FILE*`, `fopen`, `fgetc` — is built with
> `M2-Planet`'s own M2libc, not `cc_amd64`.

---

## Rung 7 — `M2-Planet`: the full self-hosting C compiler

`M2-Planet` (`M2.bin`) is the last vendored rung and the **full C compiler**: from
here the toolchain compiles real C — structs, function pointers — against the
M2libc standard library. It is the compiler in whose C subset we wrote our own
`seed`. Its source is `oriansj/M2-Planet` @ `761c2af5`, with M2libc @ `b8bb2a01`,
both GPL-3.0; see [`stage0/M2-Planet/PROVENANCE.md`](../../../stage0/M2-Planet/PROVENANCE.md)
for the per-file source SHAs.

Building it is the most involved recipe in the span (the mescc-tools-mini-kaem
phase-5 order): `catm` concatenates the bootstrap libc and all twelve M2-Planet
sources into one `.c`; `cc_amd64` compiles that to M1; `catm` prepends
`cc_amd64`'s paired defs; `M0` assembles to `hex2`; `catm` prepends the ELF
header; `hex2` links:

**Fragment** (from [`stage0/M2-Planet/build.sh`](../../../stage0/M2-Planet/build.sh);
the rung-build pipeline — `cc_amd64` → `catm` → `M0` → `catm` → `hex2`, source
list elided):

```bash
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
```

Result: `M2.bin`, **200561 bytes**, SHA `724b9e2d…`, verified against
`M2.sha256`, 2/2 tests pass. M2's core capability is verified end to end: it
compiles ordinary C and the result runs with the correct exit code (note the M1
it emits is assembled with **M2libc's** amd64 defs, not `cc_amd64`'s — M2's output
uses a different calling convention).

### The honest `M2-Planet` self-host note

A self-hosting C compiler should be able to recompile its own sources and produce
a byte-stable second generation. `M2-Planet` upstream does self-host — but **not
through this ladder's reduced positional `M0`/`hex2`**. This is documented
plainly in
[`stage0/M2-Planet/PROVENANCE.md`](../../../stage0/M2-Planet/PROVENANCE.md) and
[`stage0/MESCC_TOOLS_PROVENANCE.md`](../../../stage0/MESCC_TOOLS_PROVENANCE.md),
and it matters for an honest reading of the chain (the "H6" finding):

- Our rung-5 `M0` **corrupts exactly one instruction** — a
  `lea_rax,[rdi+DWORD]` — when it assembles M2's ~2.2 MB self-output (it drops the
  opcode and writes a garbage displacement, producing an illegal byte that
  `SIGILL`s at runtime). The M1 text M2 emits is correct; the fault is a
  large-input bug in the **vendored reduced stage0 assembler**, not in M2 and not
  in Helix.
- Upstream `M2-Planet` self-hosts via a **different** toolchain — the
  **mescc-tools** family: the `M1` macro-assembler, `blood-elf` (debug-symbol
  footer), and the **flag-driven** `hex2` linker (`--base-address`,
  `--architecture`). Those tools were vendored (mescc-tools `5adfbf33`, tag
  `Release_1.7.0`), **built with the `seed`'s `M2.bin`**, and used to run the
  self-host: `M2 → gen2 → gen3` with `gen2 == gen3` byte-identical at both source
  and binary. Gates G1–G4 pass and the result is a working compiler, independently
  reproduced three times.

The crucial honesty point — stated exactly as the trust docs state it — is the
**scope** of that result:

> **Residual:** the mescc-tools (`M1` / `blood-elf` / flag-driven `hex2`) are
> **auxiliary verifiers built by the `seed`'s `M2.bin`**, *not* new rungs of the
> trust ladder. The main `hex0 … M2-Planet` ladder is unchanged. `M2-Planet`
> remains the **trusted-once root** the `seed` is built from — the correct
> Reflections-on-Trusting-Trust position that *some* root must be trusted once,
> and M2-Planet is the strongest available one: vendored from canonical
> community-audited sources at a pinned commit, built only by prior rungs, with
> its output verified end to end. The trust that actually bears on *our*
> Apache-2.0 code (`seed → kovc`) is the **self-host fixpoint** (`K2==K3==K4`) and
> the **gcc diverse-double-compile** of the `seed` — both green and gated, and
> both the subject of later parts. Chasing the vendored-toolchain `SIGILL` is
> deliberately out of scope; it would mean debugging GPL upstream codegen, not
> Helix. (Sources: `stage0/M2-Planet/PROVENANCE.md`,
> `stage0/MESCC_TOOLS_PROVENANCE.md`,
> [`docs/TRUST_CHAIN_CLOSED.md`](../../TRUST_CHAIN_CLOSED.md).)

In short: the ladder *does* reach `M2-Planet` purely by building each rung from
the one below, and `M2-Planet`'s self-host fixpoint holds under a fixed functional
assembler — but that fixpoint is a *verification* run on tools the `seed` itself
builds, layered on top of a trusted-once `M2-Planet`, not a claim that the
positional rung-5/rung-3 tools can rebuild M2.

---

## Rung 8 — `seed`: the Apache-2.0 C-subset compiler that compiles Helix

`M2-Planet` is the top of the vendored span. The next rung is the first thing in
the whole ladder that is **ours**: the **`seed`**, source
[`stage0/helixc-bootstrap/seed.c`](../../../stage0/helixc-bootstrap/seed.c),
**Apache-2.0**, written in the M2-Planet C subset so the ladder can compile it
with no external toolchain. The SPDX header states its identity directly:

**Fragment** (from [`stage0/helixc-bootstrap/seed.c`](../../../stage0/helixc-bootstrap/seed.c);
the file's identity header):

```text
/* SPDX-License-Identifier: Apache-2.0
 * helixc-bootstrap seed -- the trusted Helix-subset bootstrap compiler.
 *
 * The first ORIGINAL rung of the Kovostov-Native ladder (everything below it is
 * hand-authored hex0 or vendored stage0/M2-Planet sources). A small C program in
 * the M2-Planet C subset, compiled by our stage0 ladder with NO external
 * toolchain. Its job: compile the tiny Helix subset that helixc
 * (helixc/bootstrap/{kovc,parser,lexer}.hx) is written in, minting the first
 * helixc WITHOUT Python -- replacing Python as the K1 minter.
 */
```

**What the `seed` is for.** `kovc` — the Helix compiler — is written in Helix and
self-hosts, but historically its *first* build (call it K1) was minted by a
Python reference compiler. Python was the last untrusted link in the chain. The
`seed` removes it: it is a small C compiler for the *tiny Helix subset that
`kovc.hx` is itself written in* — i32-only, one global arena, `while` +
`if`-as-expression + recursion, and six intrinsics (`__arena_push/get/set/len`,
`read_file_to_arena`, `write_file_to_arena`); no structs, enums, generics,
`match`, or closures. That subset is small enough to compile with a from-raw C
compiler, and because `kovc` emits fully self-contained ELFs there is **no
separate helix-libc to write** — the `seed` is the only original artifact needed
to bridge from the raw ladder into Helix. (See
[`stage0/helixc-bootstrap/README.md`](../../../stage0/helixc-bootstrap/README.md)
and `docs/K_TASK0_HELIX_SUBSET_FINDINGS.md` for the subset spec.)

**How the `seed` is built.** `M2-Planet` compiles `seed.c` (plus the bootstrap
libc) to M1, then the same `catm` / `M0` / `hex2` triple from the rungs below
assembles and links it to a self-contained ELF:

**Fragment** (from [`stage0/helixc-bootstrap/build.sh`](../../../stage0/helixc-bootstrap/build.sh);
`M2-Planet` compiles `seed.c`, then `catm`+`M0`+`hex2` link it):

```bash
M2=../M2-Planet/M2.bin
# ...
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
```

Note one subtlety the build script encodes: M2's *output* pairs with
**M2libc's** amd64 defs (`$LIB/amd64/amd64_defs.M1`, `libc-core.M1`,
`ELF-amd64.hex2`), **not** `cc_amd64`'s defs — the same calling-convention pairing
seen when building M2 itself. This is what the comment "lesson 30" refers to;
using the wrong defs here produces a binary that faults at runtime.

Result: `seed.bin`, **62467 bytes**, SHA `9837db12…`, and **17/17 tests pass**.
This is rung 8 of the table above, and the byte-for-byte target of
[`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) step `[2]`'s
final assertion (`seed.bin == pinned`).

> **For AI agents:** unlike the lower rungs, the committed tree does **not**
> contain `seed.bin` — it is gitignored
> (`stage0/helixc-bootstrap/.gitignore:3`). The committed artifact is `seed.c`
> plus `seed.sha256`. A clean checkout therefore *must* re-derive the seed from
> raw; you cannot find a pre-built `seed.bin` to trust.
> [`docs/CLEAN_REPRODUCTION.md`](../../CLEAN_REPRODUCTION.md) step 1 records this
> fence (`git ls-files stage0/helixc-bootstrap/seed.bin` is empty).

**What the `seed` ultimately compiles.** The `seed` exists to mint `kovc`, and
`kovc` is what compiles ordinary Helix programs end to end. The smallest example
the gate proves is the same one the style guide uses as its reference — a complete
program compiled and run by [the gate](../../../scripts/gate_kovc.sh):

**Verified example** — [`helixc/examples/exit42.hx`](../../../helixc/examples/exit42.hx)
(the gate compiles and runs it via the freshly self-hosted `kovc`; asserted exit
code `42`):

```helix
// First end-to-end Helix program: compiles, links, runs, exits with status 42.
//
// Demonstrates: function decl, return, integer literal.
// Verification: $? == 42 after running the produced ELF.

fn main() -> i32 {
    42
}
```

The gate asserts this with the line `chk "$EX/exit42.hx" 42` in
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) (the feature corpus, step
`[4]`); that corpus stands at 109 passed, 0 failed, reproduced at the v1.3
release. So the chain end to end is: 299 hand-typed `hex0` bytes → `hex1` →
`hex2` → `catm` → `M0` → `cc_amd64` → `M2-Planet` → `seed` → `kovc` → a running
Helix program — with no external compiler and no Python anywhere in the trust
path.

---

## Recap of the span

| Rung | Role | Built by | Bytes | SHA-256 prefix |
|------|------|----------|-------|----------------|
| `hex1` | hex + single-char labels | `hex0` | 622 | `c264a212…` |
| `hex2` | long labels / linker | `hex1` | 1519 | `6c69c7e6…` |
| `catm` | concatenation | `hex2` | 299 | `911d19bf…` |
| `M0` | macro assembler | `catm`+`hex2` | 1684 | `db97dff1…` |
| `cc_amd64` | minimal C compiler | `M0`+`catm`+`hex2` | 17976 | `ea0054d1…` |
| `M2-Planet` | full C compiler | `cc_amd64`+`catm`+`M0`+`hex2` | 200561 | `724b9e2d…` |
| `seed` | Apache-2.0 C-subset Helix compiler | `M2-Planet`+`catm`+`M0`+`hex2` | 62467 | `9837db12…` |

Every row was built only by the rungs to its right, self-verified its committed
`.sha256`, and reproduced byte-identically from a clean checkout in
[`docs/CLEAN_REPRODUCTION.md`](../../CLEAN_REPRODUCTION.md). `hex1` through
`M2-Planet` are vendored GPL-3.0 sources, statically separable from the
Apache-2.0 `seed`; `M2-Planet` is the trusted-once root, honestly bounded as
above.

**Next:** [seed to kovc: the self-host fixpoint](03-seed-to-kovc-fixpoint.md)
— how the `seed` mints `kovc` (K1), `kovc` recompiles itself to the byte-identical
`K2==K3==K4` fixpoint, and the gcc diverse-double-compile closes the
trusting-trust gap on the `seed→K1` surface.
