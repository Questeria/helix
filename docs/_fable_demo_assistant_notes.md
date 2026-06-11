# Fable demo-assistant campaign notes (2026-06-10, branch fable/demo-assistant-w12)

1. **The sampler RNG must be proven bit-identical BEFORE the GPU gate.** A 5-second
   PCG32 cross-check (C vs pure-python, two seeds, six draws) de-risked the whole G-S leg.

2. **Same libm = bit-determinism; numpy's SIMD exp is NOT the same libm.** The oracle's
   sampler uses pure-python math.exp (the same WSL glibc the worker links), not np.exp --
   a 1-ulp exp difference can flip a CDF cutoff.

3. **Sampled token-for-token has an irreducible near-tie residual.** Identical samplers +
   identical RNG still diverge when the INPUT logits differ ~1e-5 (GPU fp32 vs numpy) and a
   draw lands in the boundary gap -- measured: seed 42 first diverges at draw #36. The honest
   gate: pin a measured-boundary-clean window (32), hard-FAIL inside it, disclose the residual
   in the gate text, and gate user-facing reproducibility (G-R: two live runs byte-identical)
   separately. The sampling analogue of argmax near-ties.

4. **argv pre-passes must re-read positional captures.** --sample was consumed correctly but
   `mode` had been captured BEFORE the shift -- "unknown mode" with a perfectly good argv.
   Capture-after-mutate, always.

5. **The frontend agent's "integration contract" pattern works.** Defining the additive
   request fields + health flags + token-event fields BEFORE the backend exists let both
   tracks land independently: the page sends sampling fields ONLY when health advertises
   them (verified by fetch interception), so the greedy wire format is byte-identical today.

6. **Deferred honestly (state, don't bury):** worker-side stop-sequences (the page sends
   stop[] but the worker ignores unknown fields -- harmless until implemented); token-event
   alts/p/H telemetry (the page renders them only when present); waves 3-5 untouched.
