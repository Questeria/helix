# Helix v1.6 — Tier-3 envelope τ calibration

**Purpose.** The v1.6 receipt's Tier-3 check accepts a run iff `max_abs(logits − f32_oracle) ≤ τ`
**and** the argmax matches the oracle. For τ to be a *meaningful* faithfulness bound (not a
self-fulfilling per-run value), it must be **pre-declared and calibrated** against the model's own
NVFP4 quantization error — the TAO pattern (acceptance regions), as flagged in the Definition-of-Done
Risk #5 ("calibrate τ against the oracle's own quant error, documented, never hand-tuned to pass").

## Method (empirical / TAO)

For each model: run the NVFP4-quantized worker and the trusted **fp32 numpy oracle** on a small,
**diverse** calibration prompt set; measure the per-prompt maximum absolute logit deviation
`max_abs = max_i |logit_worker[i] − logit_oracle[i]|`; then declare

> **τ_model = 2.0 × max(max_abs over the calibration set).**

The 2× factor is a deliberate, uniform safety margin covering prompt-to-prompt variation and unseen
inputs — chosen *before* seeing whether any particular run passes, i.e. **not tuned to pass**. It is a
conservative envelope on the NVFP4 quant error, not a claim of a tight analytic bound (the DoD prefers
empirical-calibrated over loose analytic-Lipschitz). The receipt records this τ in `tier3_tau` and its
provenance in `tier3_tau_prov`; the independent checker enforces `max_abs ≤ τ`.

## Calibration data (2026-06-15, RTX 3070 Laptop 8 GB, NVFP4 + per-layer streaming)

Prompts (Qwen3 tokenizer): **near-tie** = "Hello world is a test in"; **capital-of-France** =
"The capital of France is"; **France-repetition** = "The capital of France is Paris. The capital of
France is" (an in-context-repetition / induction prompt — the worst case observed).

### Qwen3-8B  (36 layers, untied head)
| prompt | tokens | max_abs (worker vs fp32 oracle) | argmax match |
|---|---|---|---|
| near-tie | 6 | 2.488 | yes (279) |
| capital-of-France | 5 | 3.264 | yes (12095 " Paris") |
| France-repetition | 12 | **4.022** | yes (12095 " Paris") |

**max = 4.022 → τ_8B = 8.044.**

### Qwen3-32B  (64 layers, untied head) — the headline model
| prompt | tokens | max_abs (worker vs fp32 oracle) | argmax match |
|---|---|---|---|
| near-tie | 6 | 2.974 | no — top-2 near-tie (279 vs 15473), 4-bit flips the order; DoD Risk #7 |
| France-repetition | 12 | **5.610** | yes (12095 " Paris") |

**max = 5.610 → τ_32B = 11.220.**

> **N is small for 32B (N=2)** because a 32B forward on the 8 GB card takes ~13.5 min (per-layer
> streaming + host dequant). This is disclosed honestly; the structured-repetition prompt is the worst
> case for both models, so it anchors the max, and the 2× margin is conservative. Expanding the
> calibration set is straightforward future work (and far cheaper once v1.7's GPU dequant lands).

## What this τ does and does not claim

- **Does:** bound the maximum logit deviation of a faithful NVFP4-streamed run from the fp32 reference;
  give the Tier-3 envelope **independent teeth** — a corrupted/drifted run whose `max_abs` exceeds τ is
  rejected (negative-control proven in `scripts/gpu_qwen3_receipt_check.sh`).
- **Does not:** guarantee exact next-token (argmax) agreement — 4-bit quantization reorders logits that
  are within the quant noise of each other (top-1 margin ≲ ~1.7 logits on 32B), so greedy argmax can
  flip on genuine near-ties (DoD Risk #7). The receipt reports `tier3_argmax_match` honestly; the
  argmax sub-gate passes only when the top-1 is decisive.
- **Is not cryptographic.** Tier-3 is an empirical envelope (TAO/CommitLLM-style acceptance region),
  never a soundness proof. Tier-1 exact Freivalds remains deferred (f32 GEMM).
