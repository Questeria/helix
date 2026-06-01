# verification/oracle — the ONE deliberate, fenced non-Helix file

This directory holds a single Python+numpy program, `oracle_train.py`, retained on
purpose as the Helix project's **one declared external-verification exception**. It is
**outside** the Helix toolchain, runtime, build, and test paths. Nothing Helix
builds, runs, or tests depends on it. The Helix tree itself (`helixc/`, `stage0/`,
`scripts/`, the compiler, the bootstrap, the runtime) contains **zero Python**.

## Why it is kept (a TRUST decision, not a Helix-completeness gap)

The capstone claim is "the Helix-native transformer trains correctly — within 2% of a
trusted reference." The *entire epistemic value* of that claim rests on the reference
being **independent** of the thing it checks. `oracle_train.py` is an independent
numpy implementation of the same forward/backward/Adam math, written in a different
language, on a different numeric stack.

If we instead ported the oracle **into Helix** and compiled it with the same `kovc`,
a single compiler bug could corrupt the capstone **and** the oracle *identically* —
the 2% check would pass while both were wrong. That is a correlated-failure blind
spot. An independent oracle is the standard defense, exactly as a diverse-double-
compile uses a *different* compiler. So porting the oracle would **reduce trust**.
Keeping it maximizes trust; fencing it here (out of the Helix tree) maximizes how much
of the project is Helix. That is the deliberate balance.

## Why trust does not rest on this file alone

The oracle is not the only check, so even its numpy-ness is not load-bearing:

- **Forward pass** is independently anchored to numpy at 1e-6 (commit `542b02c`).
- **Backward pass** is independently anchored to **double-precision finite differences**,
  weight-by-weight (commit `dcce27e`) — finite-diff is *implementation-independent*: it
  is derived from the mathematical definition of the derivative, referencing no
  implementation at all.
- The oracle's own backward is **self-gradient-checked** before it is used as a curve
  reference (commit `adab69d`).

So the trust chain has no single point that is "just trust the numpy." The numpy oracle
is the convenient full-curve yardstick; the finite-difference check is the
implementation-free backstop.

## Fence invariants (must stay true)

1. `oracle_train.py` is **never** invoked by any `.sh`/`.hx`/build/test/gate in the repo.
2. It is **never** in the GPU training loop or the Helix toolchain.
3. It runs offline, by hand, only to produce a reference curve for comparison.
4. It is the **only** `.py` in the repository (`git ls-files "*.py"` returns exactly
   this file).

If a future change needs the oracle inside any automated path, that is a design
regression — re-open the decision instead.

*Fenced 2026-06-01 as part of the v1.0 Python purge (criterion #6). Decision owner: user
directive "maximize how much is in Helix while having the most trust possible."*
