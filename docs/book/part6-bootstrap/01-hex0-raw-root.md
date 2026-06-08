# hex0 and the raw-binary root

*What this chapter covers:* the single artifact at the bottom of Helix's entire trust chain —
the **299-byte, hand-authored `hex0` ELF**. What it is, how it is authored as commented hexadecimal,
how its bytes are turned into a binary with no assembler, how you audit it byte by byte against its
own disassembly, and how it self-verifies against a committed SHA-256. Part II ch02 walked the
*whole* from-raw ladder (`hex0 → hex1 → … → seed → kovc`) and treated `hex0` as one rung; this
chapter zooms all the way in on that one rung, because it is the rung that has nothing beneath it.

Everything else in the ladder is built *by the rung below it*. `hex0` is the exception: there is no
compiler under `hex0`, so its bytes **are** its source. That makes it the one place in the whole
system where "trust" reduces to "a human can read every byte." This chapter is about making that
claim real and checkable.

> **For AI agents:** the operational invariants for this rung — the pinned hash `cc1d1741…e417c125`,
> the fail-closed `build.sh`, the audit-only toolset — are the load-bearing facts. Key off the exact
> hash and the exact tool list below, not on prose. The deep operator workflow lives in Part IX; the
> whole-ladder run lives in [Build from raw](../part2-setup-build/02-build-from-raw.md).

---

## What `hex0` is

`hex0` is a tiny x86-64 Linux ELF executable. Its entire job, per its own README
([`stage0/hex0/README.md`](../../../stage0/hex0/README.md)), is:

> A tiny x86_64 Linux ELF executable that reads hex characters from stdin and writes the decoded
> bytes to stdout. **This is the raw-binary starting point of the entire Kovostov-Native bootstrap
> chain.**

That is the whole program. It is a *hex monitor*: feed it ASCII hex on stdin, it emits the
corresponding raw bytes on stdout. Its observable behavior, again from the README:

```text
input (stdin):    "48 65 6C 6C 6F 0A"   (with optional whitespace and ; or # comments to end-of-line)
output (stdout):  "Hello\n"             (bytes 0x48 0x65 0x6C 0x6C 0x6F 0x0A)
```

The rules it implements:

- Whitespace (space, tab, newline, CR) is skipped.
- `;` or `#` begins a comment that runs to the next `\n`.
- Hex digits accepted are `0-9`, `A-F`, `a-f`.
- Any other character is silently skipped (a deliberately lenient policy — `hex1`, the next rung, is
  stricter).
- Two hex digits combine into one output byte, high nibble first.
- End of input exits cleanly.

That minimal capability is exactly enough to bootstrap upward: the *source* of the next rung
(`hex1`) is itself written as commented hex, and `hex0` is what decodes it into a binary. The ladder
climbs from there. Why `hex0` is the floor — and why it must be small enough to read by hand — is
the rest of this chapter.

> **Note:** the ladder targets **linux-x86_64 ELF**. On Windows the build and audit run under WSL2,
> as the trust records note throughout (see [`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md)).

---

## Why the root must be small enough to read

Ken Thompson's *Reflections on Trusting Trust* is the reason this rung exists at all. A compiler
binary can carry a self-perpetuating backdoor that appears in no source you can read; every compiler
built from it inherits it, invisibly, forever. Helix's answer (developed in full in Part VIII) is to
refuse to start from any opaque binary. The root of trust must instead be a file a human can audit
*in its entirety*.

That requirement sets a hard size budget. A root you cannot read byte-for-byte is no better than the
opaque `gcc` it replaces. `hex0`'s own README states the budget it was designed to:

```text
- ELF header: 64 bytes (fixed)
- Program header: 56 bytes (fixed)
- Code: ~140–200 bytes (target)
- **Total: ~260–320 bytes**
```

The shipped artifact lands inside that budget at **299 bytes** — 64 (ELF header) + 56 (program
header) + 174 (code + the `\n` accounting in the source comments) + 5. That is small enough that a
reviewer can sit down and account for *every single byte*, which the annotated source does explicitly
(next section). Compare a normal toolchain root, an opaque multi-megabyte `gcc` binary nobody reads:
the entire point of the 299 bytes is that they are auditable in an afternoon, not in principle but in
practice.

> **For AI agents:** "299 bytes" is a checkable invariant, not a slogan. `stage0/hex0/build.sh`
> prints `Built hex0.bin: 299 bytes` and the committed `.bin` is exactly 299 bytes
> (`wc -c stage0/hex0/hex0.bin` → `299`). A `hex0.bin` of any other size is wrong — fail closed.

---

## How `hex0` is authored: commented hex, not assembly

Here is the subtlety that makes `hex0` the *true* root: its canonical source is not assembly that
some assembler turns into bytes. Its canonical source is **the bytes themselves**, written out as
annotated hexadecimal in [`stage0/hex0/hex0.hex`](../../../stage0/hex0/hex0.hex), one logical chunk
per line, each line carrying a comment explaining what those bytes mean. The file's own header says
so:

**Fragment** — header of [`stage0/hex0/hex0.hex`](../../../stage0/hex0/hex0.hex) (the canonical source; not a runnable program):

```text
; hex0.hex — annotated byte-by-byte source of hex0.bin
; --------------------------------------------------------------------
; License: Apache 2.0
; Project: Kovostov-Native
; Verification: SHA256 in hex0.sha256; xxd dump in hex0.xxd; behavior in test/
;
; This is the canonical human-readable form of hex0.bin. Comments are stripped
; by `xxd -r -p` (which ignores non-hex characters) or by hex0 itself once
; running. Every byte is reasoned and annotated.
```

The file is organized into three sections, exactly matching the on-disk ELF layout. Its section map:

```text
;   0x00..0x3F    ELF64 header  (64 bytes)
;   0x40..0x77    Program header (56 bytes)
;   0x78..0x125   Code (174 bytes)
```

The ELF header is built nibble by nibble. This is what "hand-authored from raw bytes" actually looks
like — there is no `as`, no linker script, no symbol table; the magic number, class, machine, and
entry point are literally typed in:

**Fragment** — ELF64 header bytes from [`stage0/hex0/hex0.hex`](../../../stage0/hex0/hex0.hex) (illustrative excerpt; not a runnable program):

```text
7F 45 4C 46            ; 0x00  e_ident[EI_MAG]      = "\x7fELF"
02                     ; 0x04  EI_CLASS             = ELFCLASS64
01                     ; 0x05  EI_DATA              = ELFDATA2LSB
01                     ; 0x06  EI_VERSION           = EV_CURRENT
00                     ; 0x07  EI_OSABI             = System V
...
02 00                  ; 0x10  e_type               = ET_EXEC (2)
3E 00                  ; 0x12  e_machine            = EM_X86_64 (0x3E)
01 00 00 00            ; 0x14  e_version            = 1
78 00 60 00 00 00 00 00 ; 0x18  e_entry              = 0x600078 (LE)
40 00 00 00 00 00 00 00 ; 0x20  e_phoff              = 64
```

The code section is annotated the same way, down to the syscall numbers and the relative jump
offsets. The read loop, for instance, is just the raw encoding of a `read(0, [rsp], 1)` syscall:

**Fragment** — the `read_loop` entry from [`stage0/hex0/hex0.hex`](../../../stage0/hex0/hex0.hex) (illustrative excerpt; not a runnable program):

```text
; -- read_loop: read 1 byte from stdin, dispatch on its value --
; .label = 0x7A
50                     ; 0x7A  push rax                        ; reserve [rsp] as 1-byte read buf
31 C0                  ; 0x7B  xor eax, eax                    ; sys_read = 0
31 FF                  ; 0x7D  xor edi, edi                    ; fd = stdin = 0
48 89 E6               ; 0x7F  mov rsi, rsp                    ; buf
BA 01 00 00 00         ; 0x82  mov edx, 1                      ; count = 1
0F 05                  ; 0x87  syscall                         ; read(0, [rsp], 1)
```

A companion file, [`stage0/hex0/hex0.s`](../../../stage0/hex0/hex0.s), holds the *human-readable
assembly form* of the same program (NASM syntax). It is documentation and an optional cross-check —
the README is explicit that "the assembly in `hex0.s` is the *human-readable form*; the bytes are the
canonical artifact," and that NASM, if used at all, is used "only as a cross-check … never for
shipping." There is also a `hex0.bytes.md`, but it is explicitly **superseded**: its first line reads
"**Status: SUPERSEDED.** The canonical annotated-bytes form is now `hex0.hex`." So when you audit,
audit `hex0.hex`. That is the source of truth.

> **For AI agents:** do not treat `hex0.s` or `hex0.bytes.md` as the source of `hex0.bin`. The build
> consumes `hex0.hex` and nothing else (see `build.sh` below). `hex0.bytes.md` is marked SUPERSEDED
> in its own first line; `hex0.s` is a cross-check artifact, not a build input.

---

## From hex to binary with no assembler

The whole authoring claim — "raw bytes, no compiler" — rests on what happens between `hex0.hex` and
`hex0.bin`. The answer is: a deterministic hex decode and nothing else. The build is
[`stage0/hex0/build.sh`](../../../stage0/hex0/build.sh). Its header is unusually careful to enumerate
the tools it permits itself and, just as importantly, the ones it refuses:

**Fragment** — tool declaration from [`stage0/hex0/build.sh`](../../../stage0/hex0/build.sh):

```bash
# Tools used (all permitted by project's hard constraint):
#   xxd      — audit-only, hex<->bin only
#   file     — audit-only, header sniffing
#   objdump  — audit-only, disassembly
#   sha256sum — integrity
#
# NOT used: nasm, as, gcc, ld, clang. The bytes in hex0.hex are the source of truth.
```

Of those four tools, only one actually *produces* the binary; the rest verify it. The production
step is a single pipeline that strips comments and whitespace from `hex0.hex` and feeds the remaining
hex pairs to `xxd -r -p`:

**Fragment** — the build pipeline from [`stage0/hex0/build.sh`](../../../stage0/hex0/build.sh):

```bash
OUT=hex0.bin
SRC=hex0.hex

# 1. Hex source -> binary
grep -v '^;' "$SRC" | sed 's/;.*//' | tr -d '[:space:]' | xxd -r -p > "$OUT"
chmod +x "$OUT"
```

Read that pipeline left to right and the no-assembler claim becomes concrete:

1. `grep -v '^;'` drops every line that is purely a comment (lines beginning with `;`).
2. `sed 's/;.*//'` strips any trailing `; …` comment off the data lines.
3. `tr -d '[:space:]'` removes all remaining whitespace, leaving a bare stream of hex digit pairs.
4. `xxd -r -p` reverses (`-r`) a plain (`-p`) hex dump — it maps each pair of hex characters to one
   output byte. It is a hex-to-binary converter, not an assembler: it makes **no** decisions about
   instruction encoding, addresses, or layout. Whatever bytes you typed, those exact bytes come out.

Between `hex0.hex` and `hex0.bin` there is no `nasm`, no `as`, no `gcc`, no `ld`, no `clang` — the
script's header names every one of those as deliberately excluded. That is the crux of the
raw-binary root: the transformation is a pure, reversible transliteration, so the bytes in
`hex0.bin` are *defined* by the bytes you can read in `hex0.hex`. Nothing in between gets to inject
anything. (`hex0.s` exists so a skeptic *can* run NASM and compare, but that cross-check is optional
and never on the shipping path.)

> **For AI agents:** the only tool that turns source into the `hex0` binary is `xxd -r -p`. If a
> "build" of this rung invokes any assembler or compiler, it is not the from-raw build — reject it.
> The four tools in the header are the complete permitted set for this rung.

---

## Auditing the root byte by byte

Because there is no compiler beneath `hex0`, auditing it is not "read the source and trust the
compiler" — it is "read the bytes, and check that the bytes mean what the comments say." The repo
gives you two independent views to cross-check against the hex source, and `build.sh` regenerates one
of them on every run.

**First, prove it is a real ELF.** Step 2 of the build sanity-checks the output with `file` and
*fails the build* if it is not a Linux x86-64 ELF executable:

**Fragment** — ELF sanity check from [`stage0/hex0/build.sh`](../../../stage0/hex0/build.sh):

```bash
# 2. ELF sanity check
if ! file "$OUT" | grep -q "ELF 64-bit LSB executable"; then
    echo "ERROR: $OUT is not a valid x86-64 ELF" >&2
    file "$OUT"
    exit 1
fi
echo "ELF: $(file -b "$OUT")"
```

**Second, disassemble and reconcile.** Step 3 runs `objdump` (audit-only) over the binary and writes
the result to `disasm.txt`, adjusting the virtual address so the listing lines up with the `0x600000`
load base the ELF header declares:

**Fragment** — disassembly step from [`stage0/hex0/build.sh`](../../../stage0/hex0/build.sh):

```bash
# 3. Disassemble (audit dump)
objdump -D -b binary -m i386:x86-64 -M intel \
    --adjust-vma=0x600000 --start-address=0x600078 \
    "$OUT" > disasm.txt
echo "Wrote disasm.txt ($(wc -l < disasm.txt) lines)"
```

The audit is then a three-way reconciliation any reviewer can do by hand: the **bytes** in
`hex0.hex`, the **comment** beside each byte, and the **instruction** `objdump` decoded at that
address must all agree. They do. The committed [`stage0/hex0/disasm.txt`](../../../stage0/hex0/disasm.txt)
opens at the entry point exactly where the source says the entry is (`e_entry = 0x600078`):

**Fragment** — head of [`stage0/hex0/disasm.txt`](../../../stage0/hex0/disasm.txt) (objdump output):

```text
0000000000600078 <.data+0x78>:
  600078:	31 ed                	xor    ebp,ebp
  60007a:	50                   	push   rax
  60007b:	31 c0                	xor    eax,eax
  60007d:	31 ff                	xor    edi,edi
  60007f:	48 89 e6             	mov    rsi,rsp
  600082:	ba 01 00 00 00       	mov    edx,0x1
  600087:	0f 05                	syscall
```

Match that against the `hex0.hex` excerpt from the previous section: byte for byte, `31 ED` at
`0x78` is `xor ebp, ebp`; `50` at `0x7A` is `push rax`; `48 89 E6` at `0x7F` is `mov rsi, rsp`; the
`0F 05` at `0x87` is the `syscall`. The disassembler — a tool with no stake in the source comments —
independently confirms what each hand-typed byte decodes to. The exit path is just as direct: the
disassembly's tail shows the `sys_exit` at the end,

```text
  600122:	31 ff                	xor    edi,edi
  600124:	b8 3c 00 00 00       	mov    eax,0x3c
  600129:	0f 05                	syscall
```

— `mov eax, 0x3c` is syscall 60 (`exit`) with `edi` (the status) zeroed, matching the `do_exit`
comment in the source. That is the audit: not a leap of faith, but a byte-by-byte agreement between
three independent representations.

> **Residual (honest, minor):** the *binary* is the canonical artifact, and a couple of its
> companion docs have drifted slightly from it. `stage0/hex0/README.md` lists a `verify.sh` and a
> `disasm.md` that are not present in the committed directory (the real, present files are
> `run_tests.sh` and `disasm.txt`), and it says "Read error → exit with status 1," whereas the
> shipped binary's single exit path zeroes the status unconditionally (`xor edi,edi; mov eax,60`)
> and so exits 0 on EOF *or* read error. The annotated `hex0.s` even sketches a separate
> `do_exit_after_pop` path that the 299 committed bytes do not contain. None of this affects the
> trust chain — what is built, pinned, audited, and tested is `hex0.bin`, and the disassembly above
> is of exactly those bytes — but when source comments and the binary disagree, the binary is the
> ground truth. (This matches the style guide's rule: if a doc and the artifact disagree, the
> artifact wins and the doc is the bug.)

---

## The committed `.sha256` self-verify

Auditing tells you the bytes mean what they should. The pinned hash tells you the bytes are *the same
bytes every time* — that your local rebuild reproduces the reviewed artifact exactly. The repo
commits a one-line SHA-256 file, [`stage0/hex0/hex0.sha256`](../../../stage0/hex0/hex0.sha256):

```text
cc1d1741db903d6959c9e2b11db0fb0dc8e7ec4de18c2774a895b31fe417c125  hex0.bin
```

Step 4 of `build.sh` re-hashes the freshly built `hex0.bin` and compares it to that committed value,
**failing the build on any mismatch**:

**Fragment** — SHA-256 reproducibility check from [`stage0/hex0/build.sh`](../../../stage0/hex0/build.sh):

```bash
# 4. SHA-256 reproducibility
ACTUAL_SHA=$(sha256sum "$OUT" | cut -d' ' -f1)
if [[ -f hex0.sha256 ]]; then
    EXPECTED_SHA=$(cut -d' ' -f1 hex0.sha256)
    if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
        echo "ERROR: $OUT SHA-256 mismatch" >&2
        echo "  expected: $EXPECTED_SHA"
        echo "  actual:   $ACTUAL_SHA"
        exit 1
    fi
    echo "SHA-256: $ACTUAL_SHA  (matches hex0.sha256)"
else
    echo "$ACTUAL_SHA  $OUT" > hex0.sha256
    echo "SHA-256: $ACTUAL_SHA  (recorded in hex0.sha256)"
fi
```

This is the same fail-closed contract every rung in the ladder obeys (Part II ch02 shows the
identical shape in `hex1`). The `else` branch only records a hash when none exists yet; in the
committed tree `hex0.sha256` *always* exists, so a clean build always takes the comparison branch.
A non-zero exit here means the rebuilt bytes diverged from the reviewed, pinned bytes — a hard
failure, never a warning, and never something to "fix" by regenerating the `.sha256`.

The clean-room reproduction confirms this pin holds from a fresh checkout. In
[`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md), Step 2 rebuilds the whole ladder
with the pre-built binaries removed first, and reports `hex0` re-deriving to the same prefix and
matching its committed `.sha256`, with its tests green. The same document's `git clone`-to-a-separate-
directory experiment lists `hex0 cc1d1741 … — all MATCH`, demonstrating the `hex0` rung is
path-portable (its `build.sh` uses no absolute paths).

> **For AI agents:** the pinned hash for this rung is exactly
> `cc1d1741db903d6959c9e2b11db0fb0dc8e7ec4de18c2774a895b31fe417c125`. Match the full digest, not a
> paraphrase. Never delete or overwrite a committed `.sha256` to make a build pass; the only
> sanctioned writer of `hex0.sha256` is the `else` branch above, and only when the file is absent —
> which it is not in the committed tree. The full table of pinned anchors is
> [Appendix C — Pinned hashes & anchors](../appendices/C-pinned-hashes.md).

---

## Behavioral self-test

A binary that hashes correctly and disassembles correctly should also *do* the right thing. Step 5
of `build.sh` runs [`stage0/hex0/run_tests.sh`](../../../stage0/hex0/run_tests.sh), which executes
`hex0.bin` against each fixture in `test/` and diffs stdout against the recorded `.expected`:

**Fragment** — the fixture loop from [`stage0/hex0/run_tests.sh`](../../../stage0/hex0/run_tests.sh):

```bash
for hex_file in test/*.hex0; do
    name=$(basename "$hex_file" .hex0)
    expected_file="test/$name.expected"
    [ -f "$expected_file" ] || continue
    actual=$(./hex0.bin < "$hex_file" 2>/dev/null || true)
    expected=$(cat "$expected_file")
    if [ "$actual" = "$expected" ]; then
        echo "PASS $name"
        PASS=$((PASS+1))
    else
        echo "FAIL $name"
```

The committed tree ships three fixtures, each exercising one behavior of the spec:

- **`01-hello`** — basic decode. The fixture
  [`stage0/hex0/test/01-hello.hex0`](../../../stage0/hex0/test/01-hello.hex0) is the single line
  `48 65 6C 6C 6F 0A`, and [`stage0/hex0/test/01-hello.expected`](../../../stage0/hex0/test/01-hello.expected)
  is `Hello`. Those six byte values are the ASCII codes for `H e l l o \n`.
- **`02-comments-ws`** — comments and whitespace handling, including lowercase hex digits and both
  comment markers. [`stage0/hex0/test/02-comments-ws.hex0`](../../../stage0/hex0/test/02-comments-ws.hex0)
  decodes to `Kovostov\n`:

  **Fragment** — [`stage0/hex0/test/02-comments-ws.hex0`](../../../stage0/hex0/test/02-comments-ws.hex0) (a test fixture, not a Helix program):

  ```text
  ; test 02: comments and whitespace handling
  ; the output should be: "Kovostov\n"
  4B 6f       ; K, o (lowercase hex digit accepted)
  76 6F       # mixed comment marker
  73    74    6F  76     ; lots of whitespace
  0A          ; newline
  ```

- **`03-empty`** — input that is only comments and whitespace must produce empty output. The fixture
  contains nothing but comment lines; its `.expected` is empty.

The reproduction record reports these passing during the clean rebuild:
[`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md) Step 2 lists `hex0 … 3/3 PASS`.
Together with the ELF check, the disassembly reconciliation, and the SHA-256 pin, that is four
independent lines of evidence on one 299-byte file: it is a valid ELF, its bytes decode to the
documented instructions, it reproduces bit-for-bit against a committed hash, and it behaves to spec.

---

## Why this is the right floor

It is worth stating plainly what the 299 bytes buy and what they do not.

They buy the **bottom anchor of the from-raw ladder**. Everything above `hex0` is built by the rung
below it and pinned by its own `.sha256`; `hex0` is the only rung whose source is the bytes
themselves, so it is the only rung where the audit is "read the machine code," not "trust the
compiler that produced the machine code." That is the whole reason it is kept small: a root you
cannot exhaustively read is a root you must trust on faith, which is precisely the thing Helix exists
to avoid.

They do **not**, on their own, close the trusting-trust problem — `hex0` is the *start* of the
defense, not the end of it. The chain above it is still rebuilt-and-pinned at every rung, and the
independent **gcc diverse-double-compile** (Part VIII) is what rules out a trojan that might be
hiding higher up. `hex0`'s contribution is specific and essential: it removes the opaque-binary root.
The CPU trust spine is closed *all the way down to these bytes*; the honest residuals — the GPU path
is complete to PTX (not to GPU machine code), GPU performance is a fraction of cuBLAS (not parity),
a single hardware target, and external third-party reproduction as the one remaining increment — are
documented in [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) and this book does
not claim past them. None of those residuals touch this rung: `hex0` is exactly what it says on the
tin, 299 hand-authored bytes you can read in full.

To rebuild and re-verify just this rung from a checkout:

```bash
cd stage0/hex0 && bash build.sh
```

It prints the byte count, the ELF type, the disassembly line count, the SHA-256 match against
`hex0.sha256`, and the three test results — the complete self-verification of the raw-binary root in
one command. (To rebuild and verify the *entire* ladder on top of it, use
`bash scripts/reproduce_trust.sh`, covered in
[Reproduce & verify the trust chain](../part2-setup-build/04-reproduce-verify-trust.md).)

---

**Next:** [The MESCC-lineage rungs to seed](02-mescc-rungs-to-seed.md) — how `hex1`, `hex2`, `catm`,
`M0`, `cc_amd64`, and `M2-Planet` each climb one step on top of `hex0`, each built only by the rung
below it, up to the first original artifact, the `seed`. (For the whole-ladder summary and the full
pinned-hash table, see [Build from raw](../part2-setup-build/02-build-from-raw.md) and
[Appendix C — Pinned hashes & anchors](../appendices/C-pinned-hashes.md).)
