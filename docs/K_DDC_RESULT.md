# Diverse-Double-Compile (DDC) result — the seed faithfully reproduces kovc

**Date:** 2026-05-30 · **Repo:** Kovostov-Native @ branch main · **Harness:**
`stage0/helixc-bootstrap/ddc_check.py`

## Claim (precise, scoped)

The `helixc-bootstrap` **seed** — a small Apache-2.0 C compiler written in the
M2-Planet C subset and built **only** by our from-raw-binary stage-0 ladder
(`hex0` → `hex1` → `hex2` → `catm`/`M0` → `cc_amd64` → `M2-Planet` → seed) —
mints a first `helixc` (K1') that is **behaviorally byte-identical** to the one
the existing **Python reference compiler** mints. Therefore **Python is no longer
required in the bootstrap trust chain**: the first `helixc` can be produced from
299 hand-typed `hex0` bytes with no Python anywhere.

This is the bootstrap-trust precondition for K4 (deleting the Python compiler).
It does **not** by itself claim the whole project is Python-free — the Python
compiler is still the test-suite reference and build tooling; full v1.0
Python-deletion additionally requires the Helix-native test infrastructure to
subsume that role. K4 is **user-gated**.

## Method — Wheeler's diverse double-compiling

Two **independent** compilers build kovc from the same source, then each built
compiler compiles the SAME input; the two outputs are compared. Independent
codegen differences between the two *builders* wash out at the self-hosting
fixpoint, so a byte-identical match proves semantic equivalence — and, per
Wheeler (2009) / Thompson (1984), defeats a "trusting trust" trojan: a malicious
seed could only pass if it reproduced kovc exactly, i.e. injected nothing.

```
source S = k1src.hx   (lexer_no_main + parser_body + kovc_lib + driver_main, 1 495 577 bytes)
input  BIG = k1input.hx (the same 1.5 MB compiler source; what the fixpoint compiles)

Route A (seed):    S --seed (C, stage0 ladder)----> K1'   ; K1'(BIG)  -> K2_seed
Route B (Python):  S --python reference compiler--> K1py  ; K1py(BIG) -> K2_python
```

`K1'` and `K1py` differ in bytes (587 092 vs 955 807) — expected, different
compilers built kovc. The test is whether they *behave* identically.

## Result (reproduced by `python stage0/helixc-bootstrap/ddc_check.py`)

```
[B] K1py     = 955807 bytes ; K1py(BIG)  -> K2_python = 593572 bytes
[A] K1'      = 587092 bytes ; K1'(BIG)   -> K2_seed   = 593572 bytes

K2_seed vs K2_python : IDENTICAL  (593572 vs 593572 bytes)
K2_seed compiles 6*7 -> exit 42   (the seed-built compiler WORKS)
```

**PASS.** The seed route and the Python route converge byte-for-byte at the
self-hosting fixpoint.

## What this proves, and what it does not

- **Proves:** the seed-built compiler is *behaviorally identical* to the
  Python-built compiler on the canonical DDC input (the compiler's own 1.5 MB
  source, which exercises the whole language the bootstrap uses). A trojan in
  the C seed (independent of Python) cannot survive this match. Python is
  redundant for *minting helixc*.
- **Does not prove:** that kovc is bug-free (DDC proves equivalence to the
  reference, not absolute correctness — if the Python reference had a bug the
  seed would faithfully reproduce it, which is exactly what "replace Python"
  needs). It also does not port the Python *test/build* tooling — a separate
  track before full v1.0 Python-deletion.

## Reproduce from scratch

```
# 1. build the seed with the stage0 ladder (17 regression tests)
wsl -e bash -c "cd /mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap && bash build.sh"
# 2. assemble the 1.5 MB self-source from the FROZEN helixc/bootstrap/*.hx
python stage0/helixc-bootstrap/assemble_k1.py
# 3. seed builds K1' (slow, O(n^2) lookups, ~10 min)
wsl -e bash -c "cd /mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap && ./seed.bin k1src.hx /tmp/K1prime"
# 4. run the DDC (NO argv -- Git Bash mangles /tmp/...): builds the Python route, runs both, compares
python stage0/helixc-bootstrap/ddc_check.py
```

Vendor pins (frozen): stage0-posix-amd64 @ `15535f88e25825f01a0de275b6d45f77e618bd6b`,
M2-Planet @ `761c2af5eee5bc2c27945b0ec896be26b8f5939b`,
M2libc @ `b8bb2a0159a7376716a396ec6f6bc29dd27857b5`.
