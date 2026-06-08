# `seed` to `kovc`: the self-host fixpoint

*What this chapter covers:* the exact mechanism by which the `seed` C-subset compiler builds the
first Helix-in-Helix compiler `K1`, and how the `seed → K1 → K2 → K3 → K4` chain converges to a
**byte-identical** self-host fixpoint pinned at `0992dddd…`. It opens the lid on the parts that
[Trust at a glance](../part1-orientation/04-trust-at-a-glance.md) summarized: how `K1`'s source is
*assembled* (the `assemble_k1.hx` concatenation plus the three tiny driver-mains), how each leg of
the fixpoint runs and is validated, why the validation deliberately ignores the process exit code,
and how `gcc` independently double-compiles the `seed → K1` rung to the same `84363adb…`. This is
the rung where the from-raw ladder hands off to a compiler written in Helix; the two chapters before
this one carried the ladder up to `seed`.

The grounding sources are the bootstrap directory scripts and the gate's fixpoint legs:
[`stage0/helixc-bootstrap/assemble_k1.hx`](../../../stage0/helixc-bootstrap/assemble_k1.hx),
[`stage0/helixc-bootstrap/drivers/`](../../../stage0/helixc-bootstrap/drivers/),
[`scripts/selfhost_fixpoint_rawbinary.sh`](../../../scripts/selfhost_fixpoint_rawbinary.sh),
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[2]`, and
[`stage0/helixc-bootstrap/ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh).
Where this chapter and a repo source disagree, the source wins.

---

## The handoff: from a C compiler to a Helix compiler

By the end of the previous chapter, the from-raw ladder has produced **`seed`** — the Apache-2.0
C-subset compiler at [`stage0/helixc-bootstrap/`](../../../stage0/helixc-bootstrap/), source
`seed.c` (58 811 bytes of C), built up the ladder through M2-Planet, and pinned at
`9837db12…` (62 467 bytes). `seed` is a *compiler for a C subset*. The Helix compiler `kovc` is
written in **Helix**, across three source files in
[`helixc/bootstrap/`](../../../helixc/bootstrap/): `lexer.hx`, `parser.hx`, and `kovc.hx`.

So there is a gap to cross. `seed` cannot read three separate `.hx` files and link them — it
compiles **one** Helix source file into one x86-64 ELF. The first Helix-in-Helix compiler, **`K1`**,
must therefore be presented to `seed` as a *single* `.hx` file. Producing that single file from the
three committed compiler sources is the job of the **concatenator**, and it is the first link in the
fixpoint.

> **For AI agents:** the three files in `helixc/bootstrap/` (`lexer.hx`, `parser.hx`, `kovc.hx`) are
> the **only** compiler source you edit. The single-file `k1src.hx` / `k1input.hx` / `k1ptxdrv.hx`
> are *generated build artifacts* (gitignored), regenerated from those three on every gate run. Never
> hand-edit a `k1*.hx`; edit the three sources and re-run [`assemble_k1.sh`](../../../stage0/helixc-bootstrap/assemble_k1.sh).

---

## Building `K1`'s source: the concatenation

`K1`'s source is not written; it is **assembled**. The wrapper
[`stage0/helixc-bootstrap/assemble_k1.sh`](../../../stage0/helixc-bootstrap/assemble_k1.sh) is a thin
shell shim — the actual concatenation logic is itself a Helix program,
[`assemble_k1.hx`](../../../stage0/helixc-bootstrap/assemble_k1.hx), compiled by the frozen
raw-binary `seed` and run. (The de-shelled history: the assembly logic was a `.py`, then a `.sh`,
and was finally rewritten in Helix so that the toolchain is Python-free; the output is gated
byte-identical to the old shell version.)

**Fragment** — the wrapper that compiles and runs the Helix concatenator
([`stage0/helixc-bootstrap/assemble_k1.sh`](../../../stage0/helixc-bootstrap/assemble_k1.sh),
the whole script minus its header comment):

```bash
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
./seed.bin assemble_k1.hx /tmp/asm_k1.bin || { echo "FATAL: seed could not compile assemble_k1.hx" >&2; exit 7; }
chmod +x /tmp/asm_k1.bin
/tmp/asm_k1.bin || { echo "FATAL: assemble_k1.hx run failed" >&2; exit 8; }
echo "assembled (Helix concatenator): k1src.hx k1input.hx k1ptxdrv.hx"
```

The concatenator produces **three** single-file sources. Each is the *same* three compiler bodies
glued together, differing **only** in a tiny driver-`main` appended at the tail:

```text
k1src.hx     = strip_demo(lexer.hx) + parser.hx + strip_demo(kovc.hx) + driver_k1src.hx
k1input.hx   = strip_demo(lexer.hx) + parser.hx + strip_demo(kovc.hx) + driver_k1input.hx
k1ptxdrv.hx  = strip_demo(lexer.hx) + parser.hx + strip_demo(kovc.hx) + driver_k1ptxdrv.hx
```

Two mechanical details matter, because they are what make the concatenation *byte-deterministic*
(and therefore the fixpoint reproducible):

1. **`strip_demo`** trims the trailing `// Demo:` block off `lexer.hx` and `kovc.hx` (each source has
   a standalone demo `main` that must not collide with the real driver-`main`). `parser.hx` is taken
   whole. The keep-length is computed as the byte offset of the dashes line (`// ----`) that
   immediately precedes the `// Demo:` line.
2. **CR stripping.** The committed sources are CRLF; the concatenator strips every `0x0D` so the
   output is pure LF. A stray carriage return would change bytes and break byte-identity.

**Fragment** — the CR-stripping append and the demo-boundary finder, the two load-bearing helpers
from [`stage0/helixc-bootstrap/assemble_k1.hx`](../../../stage0/helixc-bootstrap/assemble_k1.hx):

```helix
// append bytes [base, base+len) to the arena tail, skipping CR (0x0D = 13).
fn append_stripped(base: i32, len: i32) -> i32 {
    let mut i = 0;
    while i < len {
        let b = __arena_get(base + i);
        if b != 13 { __arena_push(b); }
        i = i + 1;
    }
    0
}
```

The `main` of the concatenator reads the three frozen sources into the arena once, computes the
two keep-lengths, then for each variant appends `lexer` (stripped to keep-length) + `parser` (whole)
+ `kovc` (stripped to keep-length) + the variant's driver, and writes the result. The three
driver-mains are the **only** difference between the three outputs:

| Generated file | Appended driver | Reads | Writes |
|----------------|-----------------|-------|--------|
| `k1src.hx` | [`drivers/driver_k1src.hx`](../../../stage0/helixc-bootstrap/drivers/driver_k1src.hx) | `/tmp/k1_in.hx` | `/tmp/k1_out.bin` |
| `k1input.hx` | [`drivers/driver_k1input.hx`](../../../stage0/helixc-bootstrap/drivers/driver_k1input.hx) | `/tmp/k2_in.hx` | `/tmp/k2_out.bin` |
| `k1ptxdrv.hx` | [`drivers/driver_k1ptxdrv.hx`](../../../stage0/helixc-bootstrap/drivers/driver_k1ptxdrv.hx) | `/tmp/kernel_in.hx` | `/tmp/out.ptx` (PTX) |

The first two drive the x86 self-host fixpoint (this chapter). The third re-mints the GPU PTX
driver and belongs to the PTX-regression leg covered in
[Part VII — The PTX back end](../part7-gpu/01-ptx-backend.md).

> **For AI agents:** the hard-coded I/O paths are not incidental — they are **compiled into the
> binary**. `K1`'s read path `/tmp/k1_in.hx` and write path `/tmp/k1_out.bin` are literal strings in
> `k1src.hx` (the appended driver), e.g. `k1src.hx:33481` / `k1src.hx:33496`. A leg's runner *must*
> stage the input at the exact path the binary will read and `rm` the exact output path it will
> write. The gate does this per leg.

---

## The driver-main: what a compiler generation actually does

Each driver-`main` is a five-line program: read a Helix source into the arena, lex it, parse it,
optionally report a parse diagnostic, otherwise emit an ELF and write it. Because the lexer, parser,
and `kovc` codegen bodies are concatenated *above* it, the driver-`main` has the whole compiler in
scope.

**Fragment** — the K1 driver-main, appended to `k1src.hx`
([`stage0/helixc-bootstrap/drivers/driver_k1src.hx`](../../../stage0/helixc-bootstrap/drivers/driver_k1src.hx),
the whole file):

```helix
fn main() -> i32 {
    let src_start = __arena_len();
    let src_len = read_file_to_arena("/tmp/k1_in.hx");
    let tok_base = __arena_len();
    lex(src_start, src_len);
    let ast_root = parse_top(tok_base);
    // H-3: compile-time file:line:col diagnostic on a parse error.
    // Clean input (e.g. the self-host source) hits no AST_ERR, so
    // err_off < 0 and the normal emit path runs byte-identically.
    let err_off = find_first_err_offset(ast_root);
    if err_off >= 0 {
        print_str("/tmp/k1_in.hx");
        report_parse_diag(src_start, err_off);
        1
    } else {
        let total = emit_elf_for_ast_to_path(ast_root);
        let elf_start = __arena_len() - total;
        write_file_to_arena("/tmp/k1_out.bin", elf_start, total)
    }
}
```

The [`driver_k1input.hx`](../../../stage0/helixc-bootstrap/drivers/driver_k1input.hx) is identical
except it reads `/tmp/k2_in.hx` and writes `/tmp/k2_out.bin`. That comment about the diagnostic path
is itself part of the fixpoint argument: the self-host source is **clean** (it contains no
`AST_ERR`), so `err_off < 0`, the `else` branch always runs, and the byte-for-byte emit path is the
one exercised on every leg. The error path exists for malformed user programs and is checked
separately by the gate's `CHECK_ERR` corpus.

The last expression of the success branch — `write_file_to_arena(...)` — is the key to the next
section. It returns the **number of bytes written**, and in Helix the last expression of `main` is
the process exit status.

---

## The fixpoint: `seed → K1 → K2 → K3 → K4`

With the three sources assembled, the fixpoint runs as a four-leg chain. The canonical Python-free
runner is [`scripts/selfhost_fixpoint_rawbinary.sh`](../../../scripts/selfhost_fixpoint_rawbinary.sh);
the universal gate runs the same chain inline as step `[2]`. The chain:

```text
K1 = seed.bin(k1src.hx)        # seed compiles K1's single-file source -> K1 (the Helix-in-Helix compiler)
K2 = K1(k1input.hx)            # K1 compiles the self-host source -> K2
K3 = K2(k1input.hx)            # K2 compiles the SAME source       -> K3
K4 = K3(k1input.hx)            # K3 compiles the SAME source       -> K4
FIXPOINT: K2 == K3 == K4       # byte-identical
```

A subtle but important point: `seed` compiles **`k1src.hx`** (whose driver reads `/tmp/k1_in.hx`),
while the three self-compilation legs each compile **`k1input.hx`** (whose driver reads
`/tmp/k2_in.hx`). The compiler *bodies* in the two files are identical — same `lexer.hx`, same
`parser.hx`, same `kovc.hx`. Only the appended driver differs (which temp paths it wires up). That is
why the fixpoint is taken over `K2`, `K3`, `K4` rather than `K1`: `K1` is built from a
*different driver tail* and compiles a different input file, so it is not expected to be byte-equal to
`K2`. From `K2` onward, the compiler is compiling exactly the source it was itself built from, so a
true fixpoint must reproduce itself bit-for-bit. (`K1` *is* still pinned and checked — by the gcc-DDC
below — just not for self-equality.)

The chain is heavy on the first leg and cheap thereafter. The single-file `k1src.hx` is roughly
1.74 MB of Helix, and `seed` (the slow C-subset compiler) compiling it into `K1` dominates the
wall-clock — on the order of minutes. Once `K1` exists, each subsequent leg is a fast `kovc`-class
self-compile.

> **Note:** an early concern that the seed needed a raised stack to compile the 1.5 MB source proved
> unfounded. [`scripts/selfhost_fixpoint_rawbinary.sh`](../../../scripts/selfhost_fixpoint_rawbinary.sh)
> deliberately does **not** raise `ulimit -s`; the seed compiles the full source on the default 8 MB
> stack, and the `kovc`-emitted generations carry their own large `mmap`'d stack. A green run proves
> the "no external `ulimit`" property.

---

## Validation: why the exit code is ignored, and what is checked instead

This is the part most likely to trip up a naive runner, and it is documented as
[Trap 1 in Part IX](../part9-for-ai-agents/03-traps.md#trap-1--kovc-exits-non-zero-on-success):
**`kovc` (and the seed running the Helix program) exits non-zero on success.** Recall the driver's
last expression is `write_file_to_arena(...)`, which returns the output byte-count. The process exit
status is a single byte, so a leg's `rc` is `size mod 256`. The self-compiled `kovc` is **698 392
bytes**, and `698392 mod 256 = 24` — so a *successful* self-compile leg exits **24**, not 0.

A runner that asserts `rc == 0` would treat every successful generation as a failure. The fixpoint
legs therefore validate by **output existence + non-empty**, never by `rc`. The seed→K1 leg is the
one exception worth noting carefully: in the standalone DDC script the seed is still running a
*Helix* program (so it too returns a byte-count and is validated by non-empty); the gate's step `[2]`
additionally rc-checks the seed→K1 leg because in that context it treats the seed invocation as a
build step — but the three `kovc` self-compile legs are explicitly **not** rc-checked.

**Fragment** — the gate's own rationale, verbatim from
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[2]` (the comment block that gates the
fixpoint):

```bash
# The seed->K1 leg (a C-compiled binary) also asserts rc==0;
# the kovc self-compile legs (K1->K2->K3->K4) do NOT, because kovc returns its OUTPUT BYTE
# COUNT as the process exit status (rc = size mod 256 -> 24 for the 698392-byte self-compile,
# i.e. NONZERO ON SUCCESS) -- those legs are validated by non-empty output + the byte-identical
# and pinned-SHA fixpoint below, never by rc.
```

The runner is also hardened against a *stale-`/tmp`* false match. Every K-generation `rm -f`s its
expected output **before** the run and asserts the output is non-empty **after**. This matters
because `K3` and `K4` write the *same* output path (`/tmp/k2_out.bin`), so without the `rm` a failed
`K3` run could leave `K2`'s stale output behind and produce a spurious "byte-identical" result. A
`FIX_OK` flag gates the comparison: if any generation fails or produces empty output, the script
sets `GATE_OK=0` and **skips** the `cmp`/`sha` entirely, so a stale file can never be promoted into a
later `K`.

**Fragment** — the K1→K2 generation leg with its rm-before / non-empty-after discipline, from
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[2]`:

```bash
# K2 generation: K1.bin reads /tmp/k1_in.hx -> writes /tmp/k1_out.bin. rm output first; check rc + non-empty.
if [ "$FIX_OK" = "1" ]; then
  chmod +x /tmp/K1.bin; cp k1input.hx /tmp/k1_in.hx; rm -f /tmp/k1_out.bin
  timeout 240 /tmp/K1.bin; rc=$?; echo "  K1->K2 run rc=$rc (kovc returns output-byte-count as exit status -> nonzero on success; validated by non-empty + SHA, NOT rc==0)"
  if [ ! -s /tmp/k1_out.bin ]; then
    echo "  FIXPOINT FAIL: K2 generation produced empty/missing /tmp/k1_out.bin (K1 run rc=$rc -- kovc.hx did not self-compile)"; GATE_OK=0; FIX_OK=0
  else
    cp /tmp/k1_out.bin /tmp/K2.bin; chmod +x /tmp/K2.bin
  fi
fi
```

---

## The fixpoint assertion: byte-identical **and** pinned

Three-way equality alone is necessary but not sufficient. A deterministic *partial* write — say, a
compiler bug that truncated every output identically — would satisfy `K2 == K3 == K4` while being
flatly wrong. So the gate asserts **both** that the three are byte-identical **and** that they equal
the pinned known-good hash. That second clause is what catches a consistent-but-wrong output.

**Fragment** — the fixpoint comparison, verbatim from
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[2]`:

```bash
  EXPECT_FIX=0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f
  if [ "$S2" = "$S3" ] && [ "$S3" = "$S4" ] && cmp -s /tmp/K2.bin /tmp/K3.bin && cmp -s /tmp/K3.bin /tmp/K4.bin; then
    if [ "$S2" = "$EXPECT_FIX" ]; then
      echo "  FIXPOINT OK (K2==K3==K4 byte-identical AND == pinned known-good)"
    else
      echo "  FIXPOINT FAIL (K2==K3==K4 self-consistent but != pinned known-good $EXPECT_FIX -- toolchain drifted)"; GATE_OK=0
    fi
  else
    echo "  FIXPOINT FAIL (K2/K3/K4 differ)"; GATE_OK=0
  fi
```

Note the equality is belt-and-suspenders: it requires the SHA-256 strings to match **and** a raw
`cmp -s` byte-diff to pass, so a SHA collision alone could not slip through. The canonical
Python-free runner adds a *third* independent check — a Helix-native byte-comparator
(`selfhost_bytecmp.hx`, itself compiled by the seed) must also report `K2 == K3 == K4`, and its
verdict must agree with `cmp`.

The pinned fixpoint value is one of the three anchors recorded in
[Appendix C — Pinned hashes & anchors](../appendices/C-pinned-hashes.md) and in
[Trust at a glance](../part1-orientation/04-trust-at-a-glance.md#the-pinned-hashes):

| Artifact | SHA-256 | Size |
|----------|---------|------|
| `seed.bin` | `9837db12752a22159ca75a533910bc0d7b9afb35df9b9963f256b7b1b915c9bb` | 62 467 B |
| `K1` (`seed→K1`) | `84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba` | 697 425 B |
| fixpoint `K2==K3==K4` | `0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f` | 698 392 B |

The re-derived values from a clean checkout are recorded in
[`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md) Step 3, which reports the gate's
verbatim verdict `FIXPOINT OK`, `K1 84363adb…`, `K2==K3==K4 0992dddd…`, corpus 109/0, and
`GATE_PASS`.

> **For AI agents:** match the literal success token. The gate emits
> `FIXPOINT OK (K2==K3==K4 byte-identical AND == pinned known-good)` on success and `GATE_PASS` as
> the overall verdict. A `K2==K3==K4` that is self-consistent but `!= 0992dddd…` is a **FAILURE**
> (toolchain drift), not a pass — the pinned-hash clause exists precisely to reject it. Always
> compare full hashes, never a prefix.

---

## The `gcc` diverse-double-compile of `seed → K1`

The from-raw ladder establishes that no *pre-built* compiler is trusted, but it still trusts
M2-Planet — the rung that built `seed`. Ken Thompson's trusting-trust attack lives exactly there: a
backdoor in the building compiler could survive recompilation from clean source. Helix answers it at
the `seed → K1` rung with Wheeler's **diverse double-compile (gcc-DDC)**: build the seed a *second,
independent way* — with `gcc`, a toolchain with **zero M2-Planet ancestry** — from the *frozen*
`seed.c`, and assert that the gcc-built seed compiles the same `k1src.hx` into a **byte-identical**
`K1`. If M2-Planet had injected a trojan into the seed, the gcc-built seed would have to carry the
*same* trojan (or it would live visibly in `seed.c`) for the two `K1` binaries to match. They match,
to the byte.

The driver is
[`stage0/helixc-bootstrap/ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh). It
builds the gcc-seed from the unedited `seed.c` (headers supplied via `-include`, so `seed.c` is not
touched), self-tests it (the gcc-seed must exit `42` on its no-arg self-test — this leg *is*
rc-checked, because it is a genuine C-compiled binary), then runs both seeds over a freshly
regenerated `k1src.hx`.

**Fragment** — the gcc build of the seed and its self-test, from
[`stage0/helixc-bootstrap/ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh)
step `[1]`:

```bash
INC="-include stdio.h -include stdlib.h -include unistd.h -include string.h"
rm -f /tmp/seed_gcc
gcc -std=gnu89 -w $INC -o /tmp/seed_gcc seed.c 2>/tmp/gccerr || { echo "  gcc build FAIL:"; head -8 /tmp/gccerr; exit 1; }
if [ ! -s /tmp/seed_gcc ]; then echo "  gcc build FAIL (no /tmp/seed_gcc)"; exit 1; fi
chmod +x /tmp/seed_gcc
echo "  seed_gcc = $(stat -c%s /tmp/seed_gcc) bytes"
/tmp/seed_gcc; stc=$?; echo "  seed_gcc no-arg self-test exit=$stc (want 42)"
if [ "$stc" -ne 42 ]; then echo "  DDC_FAIL (gcc-seed self-test exit=$stc != 42 -- gcc-built seed misbehaves)"; exit 2; fi
```

Both seeds are *Helix-built compilers running the Helix `K1` program*, so each emits a byte-count
exit status — they are validated by non-empty output, not `rc`, exactly as in the fixpoint. The
script regenerates `k1src.hx` first (never trusting a possibly-stale ignored copy), `rm`s both
outputs, runs both seeds, and asserts non-empty before comparing:

**Fragment** — the two K1 generations and the anchor comparison, from
[`stage0/helixc-bootstrap/ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh)
steps `[2]`–`[3]`:

```bash
rm -f /tmp/K1_m2.bin /tmp/K1_gcc.bin
t0=$SECONDS; ./seed.bin    k1src.hx /tmp/K1_m2.bin;  echo "  M2-seed  -> K1_m2  exit=$? $((SECONDS-t0))s ($(stat -c%s /tmp/K1_m2.bin 2>/dev/null) bytes)"
t0=$SECONDS; /tmp/seed_gcc k1src.hx /tmp/K1_gcc.bin; echo "  gcc-seed -> K1_gcc exit=$? $((SECONDS-t0))s ($(stat -c%s /tmp/K1_gcc.bin 2>/dev/null) bytes)"
...
sm=$(sha256sum /tmp/K1_m2.bin  | cut -d' ' -f1)
sg=$(sha256sum /tmp/K1_gcc.bin | cut -d' ' -f1)
EXPECT_K1=84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba   # pinned known-good K1 (release-proof anchor)
if [ "$sm" = "$sg" ] && [ "$sm" = "$EXPECT_K1" ]; then
  echo "  DDC_ANCHOR_OK -- gcc (independent lineage) reproduces the M2-Planet seed's K1 byte-for-byte AND == pinned known-good."
  echo "  => The seed's behavior is independently double-compiled; identical K1 implies identical K2==K3==K4."
```

The clean-checkout run records `K1_m2` and `K1_gcc` both at **697 425 bytes**, both
`84363adb…`, with the M2-seed taking ~288 s and the gcc-seed ~1 s
([`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md) Step 4). The success token is
**`DDC_ANCHOR_OK`**, and like the fixpoint it requires *both* `K1_m2 == K1_gcc` *and* equality to the
pinned `84363adb…`.

The anchor's closing line — "identical K1 implies identical K2==K3==K4" — is the bridge between the
two halves of this chapter. Once the *seed's behavior* is independently double-compiled (identical
`K1` from two independent lineages), the deterministic self-host chain on top of it produces the
identical `0992dddd…` fixpoint regardless of which seed started it. A companion script,
[`stage0/helixc-bootstrap/ddc_fixpoint_gcc.sh`](../../../stage0/helixc-bootstrap/ddc_fixpoint_gcc.sh),
makes that explicit (belt-and-suspenders): it runs the *full* fixpoint via the **gcc-built** seed and
confirms it reaches the same `K2 == K3 == K4 == 0992dddd…`.

> **For AI agents:** three independent fail-closed checks anchor this rung — `DDC_ANCHOR_OK`
> ([`ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh)), the gate's
> `FIXPOINT OK` / `GATE_PASS` ([`gate_kovc.sh`](../../../scripts/gate_kovc.sh)), and the gcc-route
> fixpoint ([`ddc_fixpoint_gcc.sh`](../../../stage0/helixc-bootstrap/ddc_fixpoint_gcc.sh)). A
> *mismatch* in any of them is a real finding and the scripts now exit non-zero on it — never a
> benign skip. A `cmp` divergence in `ddc_crosscheck.sh` reports `DDC_ANCHOR_DIFF` and exits `3`.

---

## What this rung does and does not establish

Two honest scope notes, kept strictly separate to avoid overclaim and detailed in
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md):

- **`gcc` is an auditor, never the shipped root.** The shipped chain's root is the hand-typed `hex0`;
  `gcc` appears *only* as an independent second witness for this one `seed → K1` cross-check. The
  byte-identical, hash-pinned DDC covers the **`seed → K1` surface only** — it does not, by itself,
  cover the broader v1.1 language surface (generics, traits, closures, wide fields, bf16). That wider
  surface is cross-checked **behaviorally** against a second zero-lineage interpreter, and that
  witness is *out-of-tree* (gitignored, not clean-checkout reproducible) —
  see [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) and the residuals in
  [Trust at a glance](../part1-orientation/04-trust-at-a-glance.md#honest-residuals--what-closed-does-not-cover).

- **The shared trusted computing base remains.** A diverse double-compile only catches a backdoor
  that one of the two compilers carries and the other does not. It says nothing about anything *both*
  share: the host OS and kernel, the shell and coreutils, the `gcc`/libc/binutils/loader, the CPU and
  microcode, and the human-readable `seed.c` itself (auditable line by line, but trusted-by-reading).
  No DDC retires this substrate. The x86 spine — ladder → `seed` → fixpoint → gcc-DDC — is closed; it
  is **complete to PTX, not to GPU machine code**, which the GPU chapters take up next.

---

**Next:** with `kovc` self-hosting and the seed rung double-compiled,
[Part VII — The PTX back end](../part7-gpu/01-ptx-backend.md) picks up the third concatenated
driver — `k1ptxdrv.hx` / the
[`driver_k1ptxdrv.hx`](../../../stage0/helixc-bootstrap/drivers/driver_k1ptxdrv.hx) main — and follows
`kovc`'s PTX back end from emission through the byte-identical PTX regression to the capstone, with
the honest performance and PTX-boundary residuals stated in full.
