# Appendix C — Pinned hashes & anchors

*What this appendix covers: the single authoritative table of every pinned hash and anchor in the
Helix trust chain — the from-raw ladder rung hashes (`hex0 … seed`), the `seed → K1` anchor, and
the self-host fixpoint `K2 == K3 == K4` — with, for each one, its full SHA-256, its size, and the
exact script line that pins it. Every hash below is copied verbatim from the committed `.sha256`
files and the assertion lines in the gate / reproduction / DDC scripts; this page is a lookup
table, not a narrative — for what the anchors mean and how to reproduce them, it cross-references
the Part I and Part II chapters.*

This appendix exists so a reader or an agent can answer one question quickly and exactly: **"what
is the pinned value of X, and where is it asserted?"** It does not re-explain the from-raw ladder
(see [Build from raw](../part2-setup-build/02-build-from-raw.md)), the self-host fixpoint or the
gcc-DDC (see [Trust at a glance](../part1-orientation/04-trust-at-a-glance.md) and
[Reproduce & verify the trust chain](../part2-setup-build/04-reproduce-verify-trust.md)), or the
repository layout (see [Appendix D — File & directory map](D-file-directory-map.md)). It pins the
numbers those chapters refer to.

Every value here was taken from the tree at tag `v1.3-release`: the per-rung hashes from the
committed `stage0/<rung>/*.sha256` files, and the three release anchors from the assertion lines
of [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh),
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh), and
[`stage0/helixc-bootstrap/ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh).
The verbatim verdict lines they produce are recorded in
[`docs/CURRENT_HEAD_AUDIT_PACKET.md`](../../../docs/CURRENT_HEAD_AUDIT_PACKET.md).

> **For AI agents:** these hashes are the load-bearing strings of the whole project — key off them,
> not off English descriptions. The three release anchors are
> `9837db12752a22159ca75a533910bc0d7b9afb35df9b9963f256b7b1b915c9bb` (seed),
> `84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba` (K1), and
> `0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f` (fixpoint). When this appendix
> and a script disagree, **the script wins and this page is the bug** — dereference the cited line
> (`scripts/reproduce_trust.sh` lines 29–31, `scripts/gate_kovc.sh:117`,
> `stage0/helixc-bootstrap/ddc_crosscheck.sh:52`) and flag the mismatch. Do not transcribe a hash
> from memory; copy it from the source.

---

## C.1 The three release anchors

These are the values an independent run must reproduce for the trust core to be considered intact.
All three are declared together near the top of
[`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) (lines 29–31) and re-asserted at
their point of use. The sizes are the committed byte counts recorded in the audit packet
([`docs/CURRENT_HEAD_AUDIT_PACKET.md`](../../../docs/CURRENT_HEAD_AUDIT_PACKET.md), "Pinned hashes").

| Anchor | SHA-256 | Size (bytes) | Variable | Pinned in |
|---|---|---|---|---|
| `seed.bin` | `9837db12752a22159ca75a533910bc0d7b9afb35df9b9963f256b7b1b915c9bb` | 62 467 | `SEED_SHA` | [`scripts/reproduce_trust.sh:29`](../../../scripts/reproduce_trust.sh) **and** the committed [`stage0/helixc-bootstrap/seed.sha256`](../../../stage0/helixc-bootstrap/seed.sha256) |
| `K1` (`seed → K1`) | `84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba` | 697 425 | `K1_SHA` / `EXPECT_K1` | [`scripts/reproduce_trust.sh:30`](../../../scripts/reproduce_trust.sh) **and** [`stage0/helixc-bootstrap/ddc_crosscheck.sh:52`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh) |
| self-host fixpoint `K2 == K3 == K4` | `0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f` | 698 392 | `FIX_SHA` / `EXPECT_FIX` | [`scripts/reproduce_trust.sh:31`](../../../scripts/reproduce_trust.sh) **and** [`scripts/gate_kovc.sh:117`](../../../scripts/gate_kovc.sh) |

The exact declaration block, copied verbatim from
[`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) (lines 28–31):

```bash
# Pinned release anchors (the values an independent run must reproduce):
SEED_SHA=9837db12752a22159ca75a533910bc0d7b9afb35df9b9963f256b7b1b915c9bb
K1_SHA=84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba
FIX_SHA=0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f
```

> **Note:** only `seed.bin` has a committed hash *file* ([`seed.sha256`](../../../stage0/helixc-bootstrap/seed.sha256)).
> `K1` and the fixpoint are pinned **inline** in the scripts (as `EXPECT_K1` and `EXPECT_FIX`), not
> as separate `.sha256` files — their outputs (`K1`, `K2`, `K3`, `K4`) are regenerated build
> artifacts, not committed binaries. The `seed.bin` binary itself is **gitignored and not tracked**;
> the committed tree carries `seed.c` plus `seed.sha256`, and the seed must be re-derived from raw
> (see [Appendix D §D.3](D-file-directory-map.md) and
> [`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md)).

---

## C.2 The from-raw ladder rung hashes

The from-raw ladder is `hex0 → hex1 → hex2 → catm → M0 → cc_amd64 → M2-Planet → seed`; each rung is
built **only by the rung below it** and self-verifies its own committed `.sha256` inside its
`build.sh`. The eight committed hash files pin every rung. The SHA-256 values below are copied
verbatim from each `stage0/<rung>/*.sha256` file. The byte sizes are the per-rung counts recorded
in [Build from raw §"The rungs"](../part2-setup-build/02-build-from-raw.md) (the `Bytes` column of
its rung table); the `.sha256` files themselves record only the hash and filename, not the size.

| # | Rung | `.sha256` file | SHA-256 | Size (bytes) |
|---|---|---|---|---|
| 1 | `hex0` | [`stage0/hex0/hex0.sha256`](../../../stage0/hex0/hex0.sha256) | `cc1d1741db903d6959c9e2b11db0fb0dc8e7ec4de18c2774a895b31fe417c125` | 299 |
| 2 | `hex1` | [`stage0/hex1/hex1.sha256`](../../../stage0/hex1/hex1.sha256) | `c264a212d2b0e1f1bcf34217ed7876bb9324bd7e29cd902bb1cad4d9f45f1cf8` | 622 |
| 3 | `hex2` | [`stage0/hex2/hex2.sha256`](../../../stage0/hex2/hex2.sha256) | `6c69c7e60df220e884de4fc3bdf7137352b7b3c25a1fb7000ef7f7dea82b33bc` | 1519 |
| 4 | `catm` | [`stage0/catm/catm.sha256`](../../../stage0/catm/catm.sha256) | `911d19bff7be2bc4657b312b19c29ad98cbaad2fed141a016fa0104e07e83ce7` | 299 |
| 5 | `M0` | [`stage0/M0/M0.sha256`](../../../stage0/M0/M0.sha256) | `db97dff12dbbc1f547b5fb58fe70267ac9a99d43d5879d8bbf578f31f1ec2bd1` | 1684 |
| 6 | `cc_amd64` | [`stage0/cc_amd64/cc_amd64.sha256`](../../../stage0/cc_amd64/cc_amd64.sha256) | `ea0054d18301701b4c11a486ace94ff2045356c9fac9f616af339051242baaa9` | 17976 |
| 7 | `M2-Planet` | [`stage0/M2-Planet/M2.sha256`](../../../stage0/M2-Planet/M2.sha256) | `724b9e2d60050c4308fd9c8780b5d83338a5a9d0784e8d5290e161c860a91925` | 200561 |
| 8 | `seed` | [`stage0/helixc-bootstrap/seed.sha256`](../../../stage0/helixc-bootstrap/seed.sha256) | `9837db12752a22159ca75a533910bc0d7b9afb35df9b9963f256b7b1b915c9bb` | 62467 |

A few things worth noting from the values themselves:

- **`hex0` is the hand-authored root.** Its 299 bytes are typed by hand as hex in
  `stage0/hex0/hex0.hex`; `build.sh` turns that text into the binary with `xxd -r -p` (an
  audit-only hex↔binary tool — no assembler), then checks the result against
  [`hex0.sha256`](../../../stage0/hex0/hex0.sha256) (`cc1d1741…`). This is the bottom of the chain:
  nothing below it is trusted.
- **`catm` and `hex0` are both 299 bytes** but are different programs with different hashes
  (`911d19bf…` vs `cc1d1741…`) — the equal size is a coincidence, not an error.
- **Rung 8 (`seed`) is anchor row 1 of §C.1.** The ladder's final rung *is* the `seed`, so its
  pinned hash `9837db12…` appears both here (as the last rung) and above (as the first release
  anchor). It is the only rung whose hash is pinned in **two** committed places: its `seed.sha256`
  file and the `SEED_SHA` line of `reproduce_trust.sh`.

### How a rung pins itself

Every rung's `build.sh` recomputes the binary's SHA-256 and compares it to the committed
`.sha256`, failing closed on a mismatch. This is the shape used by all eight rungs.

**Fragment** (excerpt of [`stage0/hex0/build.sh`](../../../stage0/hex0/build.sh) lines 48–58, the
SHA-256 reproducibility check; not a complete script):

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
```

> **For AI agents:** to re-pin the whole ladder from raw in one command, run
> `bash scripts/reproduce_trust.sh` (CPU-only, ~1 min). Its step `[2]` **deletes** every pre-built
> rung binary first, then rebuilds `hex0 → … → seed`, each rung self-verifying its committed
> `.sha256`, and asserts `seed.bin == $SEED_SHA` (`scripts/reproduce_trust.sh:73–78`). The committed
> `stage0/*/*.bin` files are reference copies only; trust rests on the rebuild-and-compare, never on
> a pre-built binary (see [`docs/CURRENT_HEAD_AUDIT_PACKET.md`](../../../docs/CURRENT_HEAD_AUDIT_PACKET.md)
> residual 5). The script **modifies the working tree** — run it on a throwaway clone.

---

## C.3 Where each anchor is asserted — exact lines

The three release anchors are not just declared; each is *checked* at the point where the
corresponding artifact is produced. This section pins the exact assertion site for each.

### The seed (`9837db12…`, 62 467 B)

- **Committed hash file:** [`stage0/helixc-bootstrap/seed.sha256`](../../../stage0/helixc-bootstrap/seed.sha256)
  contains, verbatim:

  ```text
  9837db12752a22159ca75a533910bc0d7b9afb35df9b9963f256b7b1b915c9bb  seed.bin
  ```

- **Reproduction assertion:** after rebuilding the ladder,
  [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) compares the freshly built
  `seed.bin` to `SEED_SHA` (lines 74–75):

  ```bash
  GOT=$(sha256sum stage0/helixc-bootstrap/seed.bin | cut -d' ' -f1)
  if [ "$GOT" = "$SEED_SHA" ]; then say "    seed.bin == pinned ($SEED_SHA)"; else bad "seed.bin $GOT != pinned $SEED_SHA"; fi
  ```

The size **62 467 B** is the committed byte count recorded in the audit packet's "Pinned hashes"
table ([`docs/CURRENT_HEAD_AUDIT_PACKET.md`](../../../docs/CURRENT_HEAD_AUDIT_PACKET.md)) and in
[Build from raw](../part2-setup-build/02-build-from-raw.md); the reproduction script asserts the
**hash**, which fixes the bytes.

### The gcc-DDC K1 (`84363adb…`, 697 425 B)

`K1` is the output of `seed` compiling the assembled compiler source. The gcc diverse-double-compile
([`stage0/helixc-bootstrap/ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh))
builds the seed two independent ways — the from-raw `seed` and `gcc` (zero M2-Planet ancestry) — and
asserts both produce a **byte-identical K1** that **also** equals the pinned `EXPECT_K1`. The pin is
declared at line 52 and asserted at line 53:

```bash
EXPECT_K1=84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba   # pinned known-good K1 (release-proof anchor)
if [ "$sm" = "$sg" ] && [ "$sm" = "$EXPECT_K1" ]; then
  echo "  DDC_ANCHOR_OK -- gcc (independent lineage) reproduces the M2-Planet seed's K1 byte-for-byte AND == pinned known-good."
```

Here `$sm` is the SHA-256 of `K1` from the M2-derived seed and `$sg` is the SHA-256 of `K1` from
the gcc-built seed (lines 48–49). The script also **fails closed** if the two K1 are
self-consistent but drift off `EXPECT_K1` (line 56–57), and if they differ at all (line 59–63). The
same value is the `K1_SHA` anchor in `reproduce_trust.sh:30`, where step `[4]` greps the DDC log for
it ([`scripts/reproduce_trust.sh:92`](../../../scripts/reproduce_trust.sh)).

The size **697 425 B** is the committed byte count in the audit packet's "Pinned hashes" table. The
verbatim DDC verdict lines (both K1 SHAs identical and `DDC_ANCHOR_OK`) are recorded in
[`docs/CURRENT_HEAD_AUDIT_PACKET.md`](../../../docs/CURRENT_HEAD_AUDIT_PACKET.md), "The three
result-bearing legs", leg 2.

> **For AI agents:** the DDC's success token is the literal `DDC_ANCHOR_OK`. The seed's no-arg
> self-test must exit `42` first (`ddc_crosscheck.sh:22–23`, "want 42"). Match the exact token and
> the exact K1 hash; a self-consistent-but-unpinned K1 is a **failure** (`exit 2`), not a pass.

### The self-host fixpoint (`0992dddd…`, 698 392 B)

The fixpoint is `seed → K1 → K2 → K3 → K4` with **K2 == K3 == K4 byte-identical** *and* equal to the
pinned known-good. The gate ([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh)) generates the
four stages, then declares and checks the pin at lines 117–123:

```bash
  EXPECT_FIX=0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f
  if [ "$S2" = "$S3" ] && [ "$S3" = "$S4" ] && cmp -s /tmp/K2.bin /tmp/K3.bin && cmp -s /tmp/K3.bin /tmp/K4.bin; then
    if [ "$S2" = "$EXPECT_FIX" ]; then
      echo "  FIXPOINT OK (K2==K3==K4 byte-identical AND == pinned known-good)"
    else
      echo "  FIXPOINT FAIL (K2==K3==K4 self-consistent but != pinned known-good $EXPECT_FIX -- toolchain drifted)"; GATE_OK=0
```

Note the two-part check: the **three-way byte-identical** equality (`S2==S3==S4` plus `cmp -s`) is
the fundamental self-host proof; the pinned `EXPECT_FIX` additionally rejects a consistent-but-wrong
output (e.g. a deterministic partial write) that three-way equality alone could miss
([`scripts/gate_kovc.sh:113–116`](../../../scripts/gate_kovc.sh)). On success the gate prints the
verbatim line `FIXPOINT OK (K2==K3==K4 byte-identical AND == pinned known-good)`.

The same value is the `FIX_SHA` anchor in `reproduce_trust.sh:31`; step `[3]` of the reproduction
greps the gate log for the exact `FIXPOINT OK (...)` line and for the corpus/diagnostics counts
([`scripts/reproduce_trust.sh:84–86`](../../../scripts/reproduce_trust.sh)).

The size **698 392 B** is the committed byte count in the audit packet; it also appears in
[`scripts/gate_kovc.sh:58`](../../../scripts/gate_kovc.sh) as the documented self-compile output size
("the 698392-byte self-compile"), which is why the kovc self-compile legs return a nonzero exit
status equal to the output byte-count mod 256 (`698392 mod 256 = 24`) — those legs are validated by
the non-empty output and the pinned SHA, **never** by `rc == 0`.

> **For AI agents:** the gate's overall success token is `GATE_PASS` (exit `0`); the fixpoint
> sub-anchor is the verbatim line `FIXPOINT OK (K2==K3==K4 byte-identical AND == pinned known-good)`.
> The kovc self-compile legs (`K1→K2`, `K2→K3`, `K3→K4`) **exit nonzero on success** (the exit code
> is the output byte-count mod 256, i.e. `24`); do **not** treat their nonzero `rc` as a failure —
> success is non-empty output + the pinned SHA. The seed→K1 leg is a C-compiled binary and *does*
> assert `rc == 0`. See [`scripts/gate_kovc.sh:53–129`](../../../scripts/gate_kovc.sh).

---

## C.4 The verbatim verdict lines

For convenience, here are the verdict lines the committed audit packet records at `v1.3-release`,
copied from [`docs/CURRENT_HEAD_AUDIT_PACKET.md`](../../../docs/CURRENT_HEAD_AUDIT_PACKET.md), "The
three result-bearing legs". These are the strings an agent should match (not paraphrase) when
checking a run.

From the gate (`bash scripts/gate_kovc.sh`):

```text
FIXPOINT OK (K2==K3==K4 byte-identical AND == pinned known-good)
  0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f
GPU PTX REGRESSION OK
CORPUS: 109 passed, 0 failed
CHECK_ERR: 4 passed, 0 failed
GATE_PASS
```

From the gcc diverse-double-compile (`bash stage0/helixc-bootstrap/ddc_crosscheck.sh`):

```text
seed_gcc no-arg self-test exit=42 (want 42)
K1_m2  sha256=84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba
K1_gcc sha256=84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba
DDC_ANCHOR_OK -- gcc (independent lineage) reproduces the M2-Planet seed's K1 byte-for-byte AND == pinned known-good.
```

The one-command reproduction that re-derives **all** of the above from a clean checkout prints
`REPRODUCE_TRUST: PASS` on success and exits nonzero on any mismatch
([`scripts/reproduce_trust.sh:96–104`](../../../scripts/reproduce_trust.sh)).

> **Note:** these results are **byte-stable** across the v1.3 final-pass commits — `kovc.hx` and
> `seed.c` are unchanged, so the seed / K1 / fixpoint SHAs do not move; only docs and verification
> wrappers changed ([`docs/CURRENT_HEAD_AUDIT_PACKET.md`](../../../docs/CURRENT_HEAD_AUDIT_PACKET.md),
> "Head"). To confirm the live tip you are auditing against, run `git rev-parse HEAD`.

---

## C.5 What these anchors do *not* cover

The pinned hashes above lock the **CPU** trust core — the from-raw ladder, the `seed → K1` surface
(byte-identically double-compiled by gcc), and the self-host fixpoint. They are reproducible
push-button on a clean checkout. They deliberately do **not** extend to the GPU path or to the
broader behavioral surface, and this appendix would overclaim if it implied otherwise.

> **Residual:** there is **no pinned hash for any GPU artifact** here. The GPU path is **complete to
> PTX, not to SASS** (machine code); the gate's GPU leg is a **pure-text PTX regression** — it
> byte-compares re-emitted PTX against committed `.ref.ptx` references, needing no GPU and no
> `ptxas` ([`scripts/gate_kovc.sh` steps `[1]`/`[3]`](../../../scripts/gate_kovc.sh)). Below PTX,
> NVIDIA's closed `ptxas`, the CUDA driver, the GPU hardware, and the C host launcher are trusted;
> the reference target is a single GPU (`sm_86`). GPU kernel performance is a **fraction of cuBLAS**
> (~50–67.5% on that box) and the end-to-end capstone speedup is **7.0–8.7×** (Amdahl-bound), not
> ≥10×; loss parity (the hard gate) holds at ~0%. The GPU capstone is verified **separately** on a
> CUDA host by [`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh), not by these
> anchors. See [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R for every
> residual.

> **Residual:** the **byte-identical, hash-pinned** DDC covers the **`seed → K1`** surface only. The
> broader v1.1 language surface is cross-checked **behaviorally** (a manually-reconciled audit whose
> witness is out-of-tree and not clean-checkout reproducible), not by a second byte-identical
> compiler. **External third-party reproduction on independent hardware remains the one open
> increment** past ~0.9 confidence
> ([`docs/CURRENT_HEAD_AUDIT_PACKET.md`](../../../docs/CURRENT_HEAD_AUDIT_PACKET.md), "Honest
> residuals"; [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md)).

---

**Next:** **Appendix D — File & directory map** — a navigational map of the whole
Kovostov-Native repository (where the scripts and `.sha256` files pinned above actually live),
followed by **Appendix E — Example index**.
