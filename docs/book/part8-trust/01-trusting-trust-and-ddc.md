# The trusting-trust problem & the gcc-DDC

*What this chapter covers:* Ken Thompson's trusting-trust attack from first principles — why you cannot
audit your way out of it by reading source — and exactly how Helix defends against it with Wheeler's
**diverse double-compile (gcc-DDC)** applied to the `seed→K1` rung: two compilers of independent
lineage (the from-raw `seed` and `gcc`, which has **zero M2-Planet ancestry**) must produce a
**byte-identical** `K1` (`84363adb…`). It goes deeper than the summaries in
[Trust at a glance](../part1-orientation/04-trust-at-a-glance.md) and the fixpoint mechanics in
[Part VI ch03](../part6-bootstrap/03-seed-to-kovc-fixpoint.md): the *threat model* (what the attack
can and cannot reach), the *mechanism* (the script that runs it, step by step), and the *honest scope*
(what one byte-identical rung does and does not establish). The grounding sources are
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §1–§2 and
[`stage0/helixc-bootstrap/ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh).
Where this chapter and a repo source disagree, the source wins.

---

## The attack: a compiler that lies about itself

When you build a compiler, you compile its source with some *other* compiler. That other compiler was
itself built by yet another. The chain runs back further than anyone alive has inspected. Ken
Thompson's 1984 Turing Award lecture, *Reflections on Trusting Trust*, turned this regress into a
concrete, devastating attack — and the whole reason Helix exists is to push the chain of trust back as
far as it honestly can.

The attack works in two moves. Start with a compiler `C` and a target program `P` (Thompson's example
was the Unix `login` program). The attacker wants `C` to insert a backdoor into `P`.

1. **Miscompile the target.** Teach `C` to recognize when it is compiling `P`'s source, and to emit a
   backdoored binary instead of the faithful one — for example, a `login` that also accepts a secret
   master password. This first move is detectable: anyone who reads `C`'s source sees the special-case
   that names `P`.

2. **Make the lie self-reproducing — and then erase the evidence.** Teach `C` to *also* recognize when
   it is compiling its **own** source, and to re-insert *both* special-cases into the compiler it
   produces. Now compile `C` once with the malicious compiler to get a malicious binary, and **delete
   the special-case logic from the source**. The clean source compiles `login` correctly when you read
   it — but the *binary* compiler you are actually running carries the backdoor and the
   self-reproduction, and stamps both into every compiler it builds.

The result is a fixpoint of a lie. The compiler **source is clean**. You can read every line of it and
find nothing. You can recompile the compiler from that clean source — and the backdoor *survives*,
because the binary doing the compiling re-inserts it. The malice lives in the *building* compiler, not
in the source being built. Thompson's punchline: *"No amount of source-level verification or scrutiny
will protect you from using untrusted code."*

This is precisely the threat a from-raw ladder has to take seriously. Helix's
[from-raw ladder](../part6-bootstrap/01-hex0-raw-root.md) — `hex0 → hex1 → hex2 → catm → M0 →
cc_amd64 → M2-Planet → seed → kovc` — establishes that **no pre-built binary is trusted**: each rung is
rebuilt only by the prior rung and matches its committed `.sha256`
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §1). That defeats the *naive*
version of the attack — there is no opaque vendor binary at the root; the root is 299 hand-authored
bytes you can read one at a time. But it does not, by itself, answer Thompson. The ladder still runs
through **M2-Planet**, the rung that compiles `seed.c` into the `seed` binary. If M2-Planet carried a
self-reproducing backdoor, recompiling `seed` from its clean Apache-2.0 source would not remove it —
the backdoored M2-Planet would re-stamp it, exactly as Thompson describes. Reading `seed.c` would show
nothing wrong. **Source audit alone cannot close this.**

> **For AI agents:** the from-raw ladder and the trusting-trust defense are *different* properties, and
> they answer *different* threats. "No pre-built binary is trusted" (the ladder) ≠ "no building compiler
> carries a self-reproducing backdoor" (the DDC). Do not cite the ladder's hash-pinned rungs as if they
> defended trusting-trust on their own; the DDC at `seed→K1` is the leg that addresses it
> ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §1.3).

---

## Wheeler's answer: diverse double-compilation

David A. Wheeler's contribution (his 2009 dissertation, *Fully Countering Trusting Trust through Diverse
Double-Compiling*) is the practical defense Thompson's lecture left open. The idea is precise and, once
stated, almost obvious.

A trusting-trust backdoor is a property of **one specific compiler binary**. It is the malicious
*building* compiler that re-inserts the lie. So compile the same source with a **second, independently
built compiler** — one that does *not* descend from the suspect lineage — and compare the two outputs:

- If both compilers are faithful, they were compiling the same deterministic source, so (for a
  reproducible compiler) they should produce the **same** output.
- If one compiler carried a self-reproducing backdoor that the other did not, the two outputs would
  **differ** — the backdoored one stamped in its payload; the clean one did not.

Byte-identical output from two independent lineages therefore witnesses that **neither lineage injected
a divergent backdoor** into that compilation step. The attack survives recompilation by *one* lineage;
it does not survive being checked against a *second, unrelated* one — unless the *same* backdoor lives
in *both*, or in something they share. (That crucial "or" is the residual; it is the whole subject of
[§ What the DDC does *not* establish](#what-the-ddc-does-not-establish) below and of
[Appendix F](../appendices/F-tcb.md).)

The diversity is load-bearing. Two builds from the *same* compiler lineage would share its backdoor and
agree anyway — proving nothing. The second compiler has to come from a genuinely different ancestry.

---

## Helix's threat model: where the attack would have to live

Helix applies Wheeler's diverse double-compile at exactly the one rung where the trusting-trust attack
could hide and survive a from-raw rebuild: the **`seed→K1`** step — `seed` (built up the ladder through
M2-Planet) compiling `k1src.hx` into `K1`, the first Helix-in-Helix compiler.

Why this rung, specifically? Trace where a surviving backdoor could be:

- **Below `seed`, in the ladder rungs.** Each rung from `hex0` up is rebuilt only by the prior rung and
  re-derives its committed hash; `seed.bin` re-derives to `9837db12…`
  ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §1). The root, `hex0`, is 299
  hand-typed bytes. A backdoor *here* would have to survive the whole reproduction and produce the same
  pinned hashes — and the DDC below catches the case that matters (a divergence between lineages at the
  seed).
- **In `seed.c`'s visible source.** Auditable one line at a time. This is "trusted-by-reading" — and the
  DDC cannot help with a backdoor written *identically* into the source both lineages compile (see the
  residual). But a *self-reproducing* one in the M2-Planet *binary* would not need to appear in
  `seed.c` at all — which is exactly the Thompson scenario.
- **In M2-Planet, the rung that built `seed`.** This is the live trusting-trust suspect. A
  self-reproducing backdoor in the M2-Planet binary would survive recompiling `seed` from its clean
  source and would be invisible in `seed.c`.

So the defense builds the seed a **second, independent way** and checks that the two seeds compile the
*same* `K1`. The second lineage is **`gcc`** — chosen precisely because it has **zero M2-Planet
ancestry**. If M2-Planet had injected a trojan into `seed`, the `gcc`-built seed would have to carry the
*same* trojan (or it would have to live visibly in `seed.c`) for the two `K1` binaries to match
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §1.3, and the per-rung treatment in
[Part VI ch03](../part6-bootstrap/03-seed-to-kovc-fixpoint.md#the-gcc-diverse-double-compile-of-seed--k1)).

One scope rule is fixed and non-negotiable, and the book repeats it everywhere the DDC appears:

> **`gcc` is an auditor, never the shipped root.** The shipped chain's root is the hand-typed `hex0`.
> `gcc` appears *only* as an independent second witness for this one `seed→K1` cross-check. Nothing
> Helix ships descends from `gcc`. (Terminology: **gcc-DDC**, per the
> [Style Guide](../STYLE_GUIDE.md).)

---

## The mechanism, step by step

The driver is
[`stage0/helixc-bootstrap/ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh). Its
header states the claim it is built to discharge, verbatim:

**Fragment** — the script's own statement of the DDC claim, from the header of
[`stage0/helixc-bootstrap/ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh):

```bash
# Then assert BOTH seeds compile the SAME k1src.hx into a BYTE-IDENTICAL K1.
# K1 identical from two independent compilers = Wheeler DDC: M2-Planet injected
# nothing into the seed (a trojan would have to live in seed.c's visible source,
# or in BOTH gcc and M2-Planet identically). seed.c is NOT edited (headers via -include).
```

Read that last clause carefully: **`seed.c` is not edited.** The standard C headers `gcc` wants are
supplied on the command line via `-include`, so the *bytes* of the seed source both lineages compile
are identical. Editing `seed.c` for the `gcc` build would break the diversity argument (you would be
comparing two *different* sources). The script runs in three stages.

### [1] Build the seed a second way — with `gcc`

**Fragment** — the `gcc` build of the seed and its self-test, from
[`stage0/helixc-bootstrap/ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh) step
`[1]`:

```bash
INC="-include stdio.h -include stdlib.h -include unistd.h -include string.h"

echo "=== [1] gcc builds the seed from FROZEN seed.c (no edits) ==="
# rm-before (v1.3 audit-remediation 4b): no stale gcc-seed on a failed build. This is a
# C-COMPILED binary leg, so rc==0 IS a valid success assertion (kept).
rm -f /tmp/seed_gcc
gcc -std=gnu89 -w $INC -o /tmp/seed_gcc seed.c 2>/tmp/gccerr || { echo "  gcc build FAIL:"; head -8 /tmp/gccerr; exit 1; }
if [ ! -s /tmp/seed_gcc ]; then echo "  gcc build FAIL (no /tmp/seed_gcc)"; exit 1; fi
chmod +x /tmp/seed_gcc
echo "  seed_gcc = $(stat -c%s /tmp/seed_gcc) bytes"
/tmp/seed_gcc; stc=$?; echo "  seed_gcc no-arg self-test exit=$stc (want 42)"
if [ "$stc" -ne 42 ]; then echo "  DDC_FAIL (gcc-seed self-test exit=$stc != 42 -- gcc-built seed misbehaves)"; exit 2; fi
```

Two details matter for an operator. First, the `rm -f /tmp/seed_gcc` *before* the build is deliberate:
a failed build must not leave a stale binary from a previous run that a later step would silently
mistake for a good one. Second, the exit-code discipline here is the **opposite** of the fixpoint legs.
The `gcc`-seed is a genuine **C-compiled binary**, so `rc==0` *is* a valid success assertion — and the
script additionally runs a no-arg **self-test** that must exit `42` (the seed's built-in sanity check)
before it is trusted to compile anything. (Contrast the `kovc`-class legs, which return their
*output byte-count* as the exit status and so are validated by non-empty output rather than `rc` — the
trap documented in [Part IX ch03](../part9-for-ai-agents/03-traps.md) and the
[fixpoint chapter](../part6-bootstrap/03-seed-to-kovc-fixpoint.md#validation-why-the-exit-code-is-ignored-and-what-is-checked-instead).)

### [2] Compile the *same* `k1src.hx` with *both* seeds

The input is regenerated from committed source first — never trusting a possibly-stale gitignored copy
— and both output binaries are removed before they are written, so a silent failure cannot leave a
stale file that produces a false "match."

**Fragment** — input regeneration and the two `K1` generations, from
[`stage0/helixc-bootstrap/ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh) step
`[2]`:

```bash
echo "  [2.0] regenerating k1src.hx/k1input.hx/k1ptxdrv.hx from committed source via assemble_k1.sh"
rm -f k1src.hx k1input.hx k1ptxdrv.hx
bash assemble_k1.sh >/dev/null 2>&1
if [ ! -s k1src.hx ]; then echo "  DDC_FAIL (k1src.hx missing/empty after assemble_k1.sh)"; exit 2; fi
chmod +x seed.bin 2>/dev/null
# ...
rm -f /tmp/K1_m2.bin /tmp/K1_gcc.bin
t0=$SECONDS; ./seed.bin    k1src.hx /tmp/K1_m2.bin;  echo "  M2-seed  -> K1_m2  exit=$? $((SECONDS-t0))s ($(stat -c%s /tmp/K1_m2.bin 2>/dev/null) bytes)"
t0=$SECONDS; /tmp/seed_gcc k1src.hx /tmp/K1_gcc.bin; echo "  gcc-seed -> K1_gcc exit=$? $((SECONDS-t0))s ($(stat -c%s /tmp/K1_gcc.bin 2>/dev/null) bytes)"
```

The regeneration step matters to the *evidence*, not just hygiene. `k1src.hx` is the ~1.74 MB
single-file source that the [concatenator](../part6-bootstrap/03-seed-to-kovc-fixpoint.md#building-k1s-source-the-concatenation)
assembles from the three real compiler sources (`lexer.hx`, `parser.hx`, `kovc.hx`). A missing or empty
`k1src.hx` would make *both* `K1` outputs empty — and "two empty files are byte-identical" would be a
vacuous, dangerous pass. Regenerating from committed source and asserting non-empty closes that hole. A
later non-empty guard (before the hash compare) does the same for each `K1` output: a failed generation
must never leave a stale binary that fakes a match.

### [3] Assert byte-identical — *and* equal to the pinned anchor

**Fragment** — the anchor comparison, verbatim from
[`stage0/helixc-bootstrap/ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh) step
`[3]`:

```bash
echo "=== [3] DDC ANCHOR: K1_gcc == K1_m2 byte-identical? ==="
if [ ! -s /tmp/K1_gcc.bin ] || [ ! -s /tmp/K1_m2.bin ]; then echo "  DDC_FAIL (a K1 is empty -- build error)"; exit 2; fi
sm=$(sha256sum /tmp/K1_m2.bin  | cut -d' ' -f1)
sg=$(sha256sum /tmp/K1_gcc.bin | cut -d' ' -f1)
echo "  K1_m2  sha256=$sm"
echo "  K1_gcc sha256=$sg"
EXPECT_K1=84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba   # pinned known-good K1 (release-proof anchor)
if [ "$sm" = "$sg" ] && [ "$sm" = "$EXPECT_K1" ]; then
  echo "  DDC_ANCHOR_OK -- gcc (independent lineage) reproduces the M2-Planet seed's K1 byte-for-byte AND == pinned known-good."
  echo "  => The seed's behavior is independently double-compiled; identical K1 implies identical K2==K3==K4."
elif [ "$sm" = "$sg" ]; then
  echo "  DDC_FAIL (K1 self-consistent = $sm but != pinned known-good $EXPECT_K1 -- toolchain drifted from the release anchor)"; exit 2
else
  echo "  DDC_ANCHOR_DIFF -- the two K1 differ. A REAL finding to investigate (seed.c non-determinism/portability, or a compiler-semantics gap)."
  cmp /tmp/K1_m2.bin /tmp/K1_gcc.bin 2>&1 | head -2
  # v1.3 audit-remediation A2: a DDC anchor MISMATCH is a real finding -- FAIL CLOSED
  # (previously this branch printed the finding but fell off the end -> exit 0, masking it).
  exit 3
fi
```

The success condition is deliberately **two clauses, ANDed**:

1. `K1_m2 == K1_gcc` — the two independent lineages agree byte-for-byte. This is the Wheeler
   diverse-double-compile proper.
2. `K1_m2 == 84363adb…` — and that agreed value equals the **pinned known-good anchor**.

The second clause is not redundant. Two faithful compilers agreeing on a *correct* `K1` and two
backdoored compilers agreeing on a *backdoored* `K1` would both satisfy clause 1 alone. Pinning the hash
turns "the two lineages are mutually consistent" into "the two lineages are mutually consistent **at the
known-good value reviewed at release**," which is the claim the trust record actually rests on. The
short form of that anchor is `K1 = 84363adb…` (full: 64 hex chars, 697 425 bytes;
[Appendix C](../appendices/C-pinned-hashes.md)).

Equally important is what happens when the clauses *fail*. Each failure branch **exits non-zero** —
`DDC_FAIL` (drift) exits `2`, a genuine byte divergence reports `DDC_ANCHOR_DIFF` and exits `3`. This is
fail-closed by design: an earlier revision printed the divergence finding but then *fell off the end of
the script* and exited `0`, masking it. A DDC that can silently pass on a real divergence is worse than
none. The success token an operator (or this book's automation) matches is the literal string
**`DDC_ANCHOR_OK`**.

> **For AI agents:** the success token is the exact string `DDC_ANCHOR_OK`, emitted only when *both*
> `K1_m2 == K1_gcc` *and* `== 84363adb…`. A `DDC_ANCHOR_DIFF` (exit `3`) or `DDC_FAIL` (exit `2`) is a
> **real finding**, never a benign skip — do not treat a non-zero exit here as noise. Always compare the
> full 64-character hash, never a prefix.

---

## Belt-and-suspenders: the full gcc-route fixpoint

The anchor's closing line — *"identical K1 implies identical K2==K3==K4"* — is a claim worth proving
rather than asserting. Because the [self-host fixpoint](../part6-bootstrap/03-seed-to-kovc-fixpoint.md)
`seed → K1 → K2 → K3 → K4` is deterministic, an identical `K1` *must* drive the chain to the identical
`0992dddd…` fixpoint regardless of which seed minted it. A companion script makes that explicit instead
of leaving it to inference.

**Fragment** — the gcc-route full fixpoint's purpose and method, from the header and body of
[`stage0/helixc-bootstrap/ddc_fixpoint_gcc.sh`](../../../stage0/helixc-bootstrap/ddc_fixpoint_gcc.sh):

```bash
# DC3 -- run the FULL Python-free self-host fixpoint via the GCC-built seed (the
# independent route). Reuses the PROVEN scripts/selfhost_fixpoint_rawbinary.sh
# verbatim, swapping only the seed binary (./seed.bin -> /tmp/seed_gcc). Because
# K1_gcc == K1_m2 byte-identical (DC2), this MUST reach the same K2==K3==K4; the
# explicit run is the belt-and-suspenders evidence for the audit. seed.c FROZEN.
# ...
sed 's|\./seed\.bin |/tmp/seed_gcc |g' ../../scripts/selfhost_fixpoint_rawbinary.sh > /tmp/fixpoint_gcc.sh
```

It reuses the *proven* fixpoint runner verbatim, swapping only the seed binary, and propagates that
runner's verdict (fail-closed): an earlier revision fell off the end and exited `0` even if the
gcc-route fixpoint failed, which this version fixes by exiting with the inner runner's return code. The
result is a second, independent confirmation: not only does `gcc` reproduce `K1` byte-for-byte, the
*entire* self-host chain reaches the same pinned `K2==K3==K4` when bootstrapped from the gcc-built seed.

---

## How the DDC is wired into the one-command reproduction

The gcc-DDC is not a manual ritual; it is the fourth leg of the committed, push-button reproduction.

**Fragment** — the DDC leg of the one-command reproduction, verbatim from
[`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) step `[4]`:

```bash
# --- [4] gcc diverse-double-compile -------------------------------------------------------------
say "[4] gcc diverse-double-compile (stage0/helixc-bootstrap/ddc_crosscheck.sh)"
bash stage0/helixc-bootstrap/ddc_crosscheck.sh >/tmp/rt_ddc.log 2>&1 || true
if grep -q 'DDC_ANCHOR_OK' /tmp/rt_ddc.log; then say "    DDC_ANCHOR_OK"; else bad "DDC not OK (tail):"; tail -15 /tmp/rt_ddc.log >&2; fi
if grep -q "$K1_SHA" /tmp/rt_ddc.log; then say "    K1 byte-identical (gcc == M2-seed) == pinned ($K1_SHA)"; else bad "K1 != pinned $K1_SHA"; fi
```

So `bash scripts/reproduce_trust.sh` — the single command that rebuilds the whole ladder from raw,
runs the self-host fixpoint, and asserts every pinned anchor — runs the gcc-DDC as step `[4]` and
fails the whole reproduction unless both `DDC_ANCHOR_OK` appears *and* the pinned `K1_SHA` is found in
its log. Because that same script is the body of the CI job on a clean `ubuntu-latest` runner
([`.github/workflows/trust-reproduce.yml`](../../../.github/workflows/trust-reproduce.yml)), the
gcc-DDC is re-run, green, on a different machine from a fresh clone on every push — push-button for any
third party. The clean-checkout run records both `K1_m2` and `K1_gcc` at 697 425 bytes, both
`84363adb…`, with the M2-seed compile taking ~288 s and the `gcc`-seed ~1 s
([`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md) Step 4).

> **For AI agents:** the gcc-DDC is **clean-checkout reproducible** — one command, byte-identical,
> hash-pinned. This is the property that distinguishes it from the broader behavioral cross-check below
> (which is *not* clean-checkout reproducible). When you cite "the DDC is reproducible by anyone," you
> mean *this* one (`ddc_crosscheck.sh`, `DDC_ANCHOR_OK`, `84363adb…`), not the v1.1-surface behavioral
> witness.

---

## What the DDC anchors, in miniature

The whole point of double-compiling the `seed→K1` rung is that the compiler at the top of that rung —
`kovc` — is the same compiler that compiles ordinary Helix programs. The smallest end-to-end witness of
that is the first program the gate compiles and runs.

**Verified example** — [`helixc/examples/exit42.hx`](../../../helixc/examples/exit42.hx) (compiled and
run by the gate's feature corpus; the produced ELF exits with status `42` — the gate asserts it with the
corpus line `chk "$EX/exit42.hx" 42` in [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh)):

```helix
// First end-to-end Helix program: compiles, links, runs, exits with status 42.
//
// Demonstrates: function decl, return, integer literal.
// Verification: $? == 42 after running the produced ELF.

fn main() -> i32 {
    42
}
```

That this program runs and exits `42` is the trusting-trust defense made concrete: the compiler that
produced its ELF was minted by a `K1` that **two independent compiler lineages** — the from-raw `seed`
and `gcc` with no M2-Planet ancestry — produced byte-for-byte identically. A trusting-trust backdoor in
M2-Planet would have had to live, identically, in `gcc` too (or visibly in `seed.c`) to escape the
comparison. It does not.

---

## What the DDC does *not* establish

Helix's value proposition is *calibrated* honesty, so the gcc-DDC's limits are stated as plainly as its
strength. There are two, and the book keeps them strictly separate to avoid overclaim. Both are detailed
in [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 and enumerated in
[Appendix F](../appendices/F-tcb.md).

### 1. It says nothing about what both lineages *share* (the shared TCB)

A diverse double-compile only catches a backdoor that **one** of the two compilers carries and the
other does not. It is silent about anything **both** sides share. So the gcc-vs-M2-Planet DDC still
trusts, untouched, the entire substrate beneath both compilers — named bluntly in
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 residual 3:

> A backdoor identical in both `gcc` builds, or living in any of these shared layers, is **invisible** to
> DDC by construction.

That shared substrate is: the host OS and kernel; the filesystem; the shell and coreutils (`bash`,
`cmp`, `sha256sum`, `cp`, `rm` — note that a corrupted `cmp` or `sha256sum` would defeat the *comparison
itself*); the shared `gcc`/libc/binutils/loader used to build and link both seed variants; the CPU and
microcode; the RAM; and the human-readable `seed.c` source itself — auditable one line at a time, but
**trusted-by-reading, not proven**. This is the classic Wheeler **shared-substrate residual**. The
honest framing, verbatim from the trust record: *the DDC narrows the compiler-backdoor surface; it does
not erase the shared TCB.* The full enumeration is [Appendix F](../appendices/F-tcb.md#f2-the-shared-tcb--what-no-ddc-can-retire).

> **For AI agents:** do **not** describe the gcc-DDC as proving the seed is backdoor-free. It proves the
> `seed→K1` step is not carrying a backdoor that *diverges* between the two lineages. When asked "what
> does the DDC guarantee," answer with the **divergence** framing, never "the seed is safe." The shared
> TCB is explicitly out of its reach ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md)
> §2 residual 3).

### 2. The byte-identical DDC covers `seed→K1` only — the wider surface is *behavioral* and *out-of-tree*

The byte-identical, hash-pinned gcc-DDC covers the **`seed→K1` surface only**. It does not, by itself,
double-compile the broader **v1.1 language surface** — generics/monomorphization, traits, closures,
turbofish, wide-field stores, bf16. Those are real `kovc` codegen arms that the `seed→K1` byte-identity
does not, on its own, independently re-derive.

That wider surface is cross-checked by a **different, weaker form** of assurance, and the distinction is
load-bearing. A second, **zero-`kovc`-lineage tree-walking interpreter** re-derives each program's exit
value purely from the source semantics, and the from-raw `kovc`'s actual ELF exit must agree. The V5
broadening reports **44/44** v1.1-surface programs agreeing (`kovc == interp == expected`), with zero
disagreements, covering **44 of 53** witness-reachable codegen arms; f16 same-type arithmetic is **not
yet** cross-checked by this witness
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 residual 3,
[`docs/K_DDC_BROADENED.md`](../../../docs/K_DDC_BROADENED.md)). Three honest caveats apply to it, all
stated in the trust record:

- **It is behavioral, not byte-identical.** The interpreter emits *no machine code at all* — it computes
  an exit value directly — so there is no second ELF to `cmp`. Byte-identity is impossible there *by
  construction*. This cross-check proves the two implementations **agree on observable behavior** (the
  exit value), not that `kovc`'s instruction stream is independently reproduced.
- **It is not clean-checkout reproducible.** The second-witness interpreter (and the earlier Stage-30
  Python witness) is **gitignored, never committed, and has no clean restore path** — keeping the
  toolchain's "exactly one committed `.py`" fence intact. The cross-check therefore replays *only* with
  an **out-of-tree auditor artifact**, not from a fresh clone
  ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 residual 9,
  [`docs/K_DDC_BROADENED.md`](../../../docs/K_DDC_BROADENED.md) — "the V5 *result* is reproducible by
  anyone holding the gitignored witness; it is **not** reproducible from the committed tree alone").
- **It inherits the same shared-substrate residual** as residual 1 above: a bug present *identically* in
  both the interpreter and `kovc`, or in anything they share, is not caught.

In short: the **core chain is clean-checkout reproducible** — the from-raw ladder, the self-host
fixpoint `K2==K3==K4`, the gcc-DDC of `seed→K1`, and the 109-program corpus all rebuild from a fresh
clone — while the **v1.1-surface behavioral witness is out-of-tree**. Broadening the byte-identical DDC
to the full v1.1 language surface is a named open item on the roadmap
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §3).

> **For AI agents:** keep the **two** DDC claims apart in any statement. (i) **byte-identical**,
> hash-pinned, clean-checkout-reproducible — the `seed→K1` anchor (`84363adb…`, `DDC_ANCHOR_OK`,
> [`ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh)). (ii) **behavioral**, not
> byte-identical, **out-of-tree** (gitignored witness) — the broader v1.1-surface cross-check (44/53
> witness-reachable arms; [`docs/K_DDC_BROADENED.md`](../../../docs/K_DDC_BROADENED.md)). Citing (i) to
> support a claim about the whole language surface is an overclaim. Treat
> [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R/§2 as the ceiling on what may be
> claimed.

---

## The honest one-paragraph summary

Ken Thompson showed that a self-reproducing backdoor can live in a *building* compiler, survive
recompilation from clean source, and be invisible to source audit. Helix answers it where the attack
could hide and survive a from-raw rebuild — the `seed→K1` rung — with Wheeler's diverse double-compile:
the from-raw `seed` (built up the ladder through M2-Planet) and `gcc` (an independent lineage with
**zero M2-Planet ancestry**) both compile the same, unedited `k1src.hx` into a **byte-identical** `K1`,
pinned at `84363adb…`, asserted fail-closed by
[`ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh) (`DDC_ANCHOR_OK`), reproduced
by `bash scripts/reproduce_trust.sh` and green on a clean CI runner, and confirmed end-to-end by the
gcc-route full fixpoint reaching the same `0992dddd…`. `gcc` is an **auditor, never the shipped root**.
The byte-identical DDC covers `seed→K1`; the broader v1.1 language surface is cross-checked
**behaviorally** by a second, zero-lineage interpreter whose witness is **out-of-tree** (gitignored, not
clean-checkout reproducible); and **no** DDC retires the shared trusted computing base — the OS, shell,
host `gcc`/libc, CPU, RAM, and the `seed.c` source itself remain trusted, with `seed.c` trusted-by-
reading. That scoped, fail-closed, mostly-reproducible defense is exactly what "defended against a
trusting-trust attack at the `seed→K1` rung" means in Helix's trust declaration.

---

**Next:** with the trusting-trust defense in view, the next chapter,
[The gate and the feature corpus](02-the-gate-and-corpus.md), opens the other half of the
verification story — how
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) ties the self-host fixpoint, the 109-program
feature corpus, the PTX text regression, and the negative diagnostics into the single `GATE_PASS`
verdict that every reproduction keys off. The honest residuals and the full trusted computing base are
collected in [Appendix F — The trusted computing base](../appendices/F-tcb.md).
