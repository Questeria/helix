# verification/oracle — the ONE deliberate, fenced non-Helix file

This directory holds a single Python+numpy program, `oracle_train.py`, retained on
purpose as the Helix project's **one declared external-verification exception**. It is
**outside** the Helix *toolchain* — the compiler, the runtime, and the shipped from-raw
build never build, run, or depend on it. It IS invoked by the **verification harness**
(`scripts/capstone_audit.sh`) as the independent reference oracle — that is its entire
purpose. The Python-free fence is about the **toolchain**, not the verification harness:
the shipped Helix tree (`helixc/`, `stage0/`, the compiler, the bootstrap, the runtime)
contains **zero Python**; this single `.py` lives only in the audit/verification path.

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
  via a sampled spot-check — 6 gradient tensors (dW_lm, dW2[1], dWo[1], dWq[0], dW1[0],
  dWv[1]) × up to 5 sampled indices each, vs analytic backprop (NOT exhaustive;
  `helixc/runtime/train_transformer.c` ~404-435) (commit `dcce27e`) — finite-diff is
  *implementation-independent*: it is derived from the mathematical definition of the
  derivative, referencing no implementation at all.
- The oracle's own backward is **self-gradient-checked** before it is used as a curve
  reference (commit `adab69d`).

So the trust chain has no single point that is "just trust the numpy." The numpy oracle
is the convenient full-curve yardstick; the finite-difference check is the
implementation-free backstop.

## Fence invariants (must stay true)

1. `oracle_train.py` is invoked **ONLY** by the verification harness
   (`scripts/capstone_audit.sh`) as the independent reference oracle. It is **never**
   invoked by / part of the **compiler**, the **runtime**, or the shipped **from-raw
   toolchain** (no `.hx`, no bootstrap rung, no build of the shipped product, and not the
   `scripts/gate_kovc.sh` self-host/corpus gate reach it).
2. It is **never** in the GPU training loop or the Helix toolchain — it runs only as the
   offline reference inside the audit harness.
3. It runs offline, by hand, only to produce a reference curve for comparison.
4. It is the **only** `.py` in the repository (`git ls-files "*.py"` returns exactly
   this file).

If a future change needs the oracle inside any automated path, that is a design
regression — re-open the decision instead.

*Fenced 2026-06-01 as part of the v1.0 Python purge (criterion #6). Decision owner: user
directive "maximize how much is in Helix while having the most trust possible."*
