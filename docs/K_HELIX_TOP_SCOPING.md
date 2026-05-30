# K-bootstrap: the Helix top — scoping & decision (helix-libc + helixc-bootstrap)

**Status:** CHECKPOINT — the vendored from-raw-binary ladder is complete; the next
rung is the first original work. This doc is the decision material; nothing
original gets written until the user picks a strategy.

**Date:** 2026-05-30. HEAD `3e7545a`, counter 435.

## Where we are

The trusted-seed ladder is built and verified, every rung byte-exact +
reproducible + tested under WSL, each built only by the rung below it:

```
[299 hand-authored bytes] hex0 → hex1 → hex2 → catm → M0 → cc_amd64 → M2-Planet
```

`M2-Planet` (rung 7) is a **full, self-hosting C compiler** with the M2libc
standard library. It compiles ordinary C and the result runs. That is the last
rung we can vendor — there is no upstream "Helix compiler in C" to borrow.

## The gap

Our product compiler is `helixc` = `helixc/bootstrap/kovc.hx` (+ `parser.hx`,
`lexer.hx`) — written **in Helix**, feature-complete, and self-hosting (K2==K3
byte-identical). Today its first stage (K1) is minted by the **Python** reference
compiler (`_compile_src_to_elf`). That Python is:

1. the remaining **trust hole** (an unaudited ~10k-line minter at the root), and
2. the thing the hard constraint says must be **deleted** (K4; end state is fully
   Helix, no Python anywhere).

So we need a bridge from `M2-Planet` (compiles C) to `helixc` (compiles Helix)
that does **not** route through Python. That bridge is `helixc-bootstrap`.

```
M2-Planet ──compiles──▶ helixc-bootstrap ──compiles──▶ helixc(K1′) ──compiles──▶ helixc(K2′)
   (C compiler)          (the bridge, in C)              (kovc.hx)        fixpoint: K2′==K1′?
```

`helixc-bootstrap` replaces Python as the K1 minter. Once K1′ is produced without
Python and the fixpoint holds, Python is deletable (K4) and "Python-deletion-ready"
is met.

## The two original artifacts

### A. `helixc-bootstrap` — the bridge compiler (the hard part)

A program **M2-Planet can compile** (so: written in the M2 C subset) that can
compile Helix source into a working ELF. It does **not** need to be fast or
optimizing — only correct enough to compile `kovc.hx` (or a reduced Helix that
in turn compiles `kovc.hx`). Strategy options below.

### B. `helix-libc` — the runtime surface

Scope TBD and **must be mapped first** (task 0 below): `kovc.hx` currently emits
largely self-contained ELFs (its own `_start` big-stack stub, raw syscalls, arena
allocator, builtins `run_process`/`set_exec`/`read_file_to_arena`/…). So the
runtime that *Helix programs* need may already be emitted inline. `helix-libc` is
more likely the small C runtime that **`helixc-bootstrap` itself** links against
(arena, file IO, process spawn) on top of M2libc — to be sized once strategy A is
chosen.

## Strategy options for `helixc-bootstrap` (A)

| # | Strategy | Trust | Effort | Notes |
|---|---|---|---|---|
| **1** | **Minimal Helix-subset seed in M2 C.** Hand-write a small C compiler that handles exactly the Helix subset `kovc.hx`'s own source uses (no optimization, direct codegen). It compiles `kovc.hx` → helixc; helixc then recompiles itself (fixpoint). | High — audit a human-sized C seed + the existing Helix product | Large but bounded | Classic seed bootstrap, same shape as cc_amd64→M2-Planet. **Recommended.** |
| **2** | **Tiered seed (C → tiny-Helix → fuller-Helix → kovc).** Several small Helix bootstrap stages, each compiling the next; only the first is in C. | High, most granular/auditable | Largest (more rungs) | Most faithful to the "many tiny verifiable steps" philosophy; slowest. |
| **3** | **Reduce-then-seed.** First refactor `kovc.hx` to a smaller Helix subset (`kovc-lite`) that is cheaper to seed in C; seed compiles `kovc-lite`; `kovc-lite` compiles full `kovc`. | High | Medium — but touches the FROZEN compiler | Shrinks the C seed at the cost of editing kovc.hx (currently frozen; would need its own fixpoint+gate). |
| **4** | **Transpile kovc→C automatically, vendor the C, M2-compile it.** | **LOW — REJECT.** The generated C is produced by the existing/Python compiler, so trust roots back to Python (circular); machine-generated C is not human-auditable. | Small | Defeats the entire from-raw-binary trust goal. Listed only to rule out. |

DDC (diverse double-compiling, Wheeler) sits on top of whichever we pick: once
`helixc-bootstrap` mints K1′, compare K1′-built helixc against the Python-built
K1 — identical fixpoint output from two independent roots is the strongest
possible trust evidence, and is the natural retirement ceremony for Python.

## Recommendation

**Strategy 1** (minimal Helix-subset seed in M2 C), with the explicit option to
fall back to **3** if the subset `kovc.hx` uses proves too large to seed directly.
Sequence:

0. **Map the surface** (read-only, no original code): enumerate exactly which
   Helix constructs `kovc.hx`/`parser.hx`/`lexer.hx` use, and what `kovc` emits
   inline vs. would link — this sizes both the seed and `helix-libc`. (Safe to do
   now; does not cross the checkpoint.)
1. Write `helixc-bootstrap` in M2 C for that subset, M2-compile it, test on small
   Helix programs.
2. Seed → compile `kovc.hx` → K1′; fixpoint K1′→K2′; DDC vs the Python K1.
3. Delete Python (K4, user-gated) → "Python-deletion-ready" → 5 clean audits → STOP.

## Decision needed from the user

1. **Which strategy** (1 / 2 / 3)? (4 is ruled out.)
2. **OK to start with task 0** (read-only surface-mapping) now, while the rest
   waits? It writes no original compiler code and crosses no line.
3. Any constraint on the seed's size/shape (e.g., must itself stay within a
   subset that's trivially re-auditable)?

Until this is answered, the loop does **bounded, non-crossing hardening only**
(M2-Planet self-host fixpoint) and will not write `helixc-bootstrap` or
`helix-libc`.
