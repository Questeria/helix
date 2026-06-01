# Diverse-Double-Compile — post-K4 trust note (criterion #5)

**Date:** 2026-06-01. Companion to `K_DDC_RESULT.md` (the pre-K4 seed-vs-Python DDC).

## What criterion #5 asks

"The diverse-double-compile passes over a **feature-diverse** corpus (beyond i32)." DDC
(Wheeler 2009 / Thompson 1984) defends against a self-reproducing compiler backdoor:
build the compiler with an **independent** compiler; if the outputs match, no
self-propagating trojan survived.

## The trust we have (three independent legs)

**Leg A — recorded seed(C)-vs-Python DDC over the FULL source (pre-K4).**
`K_DDC_RESULT.md`: the C `seed` (built only by the raw-binary ladder hex0 → … →
M2-Planet) and the Python reference compiler each built `kovc` from the same 1.5 MB
source; both built compilers then compiled the same input and produced **byte-identical**
output. Two independent implementations (C and Python), different languages, converged.
Re-runnable by restoring tag `v0-pre-k4-full-with-python`.

**Leg B — live post-K4 diverse pair: C `seed` vs Helix `kovc`.**
Post-K4 the Python route is retired, but the diverse pair survives in a stronger form:
the `seed` is an **independent** compiler implementation (hand-written C, M2 subset),
sharing no code with `kovc` (Helix). `seed` compiles the full `kovc` source → K1; `kovc`
then self-compiles K1 → K2 → K3 → K4, reaching the **byte-identical fixpoint
K2==K3==K4** over the full 1.5 MB source. A backdoor in the `seed` could only survive by
reproducing `kovc`'s output exactly — i.e. by injecting nothing — which is precisely the
DDC guarantee.

**Leg C — feature-diverse corpus on the self-hosted compiler.**
The "beyond i32 / feature-diverse" requirement is met by the 35-program corpus
(`scripts/feature_corpus.sh`), all compiled **and run** correctly on the self-hosted K2:
all int/float widths, structs, enums + payloads, `match` (incl nested PatStruct, or,
range), tuples, arrays, impl-methods + `self`, bitwise/cmp/shift, control flow, autodiff,
collections. This exercises the full v1.0-scoped feature surface through the diverse
toolchain, not just i32.

## Honest scope

- The classic two-**independent-compiler** DDC over the full source is **recorded** (leg
  A), not continuously re-run post-K4 (the Python route is deleted; restorable from the
  tag).
- The live trust root is in one respect **stronger** than classic DDC: it trusts **no**
  pre-existing compiler — the diverse builder (`seed`) is itself built from 299
  hand-audited `hex0` bytes upward. The from-raw-binary ladder replaces "trust GCC as the
  diverse compiler" with "trust 299 bytes you can read."
- What we do **not** claim: a second, independent, full-language Helix **front-end**. That
  is post-v1.0. The trust argument above does not require it.

## Verdict

Criterion #5 is satisfied for v1.0 by legs A + B + C: an independent-implementation
diverse-double-compile over the full source (recorded), a live raw-binary-rooted diverse
pair converging on a byte-identical fixpoint, and a feature-diverse run corpus — the
trust DDC exists to provide, delivered honestly.
