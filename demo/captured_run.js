/* ============================================================
   demo/captured_run.js -- the REAL captured run that powers the
   guided run's REPLAY (backup) mode when no live backend exists
   (e.g. featured on a public website with no GPU).

   PROVENANCE (nothing here is invented):
   Captured live on 2026-06-11 from the running Helix demo backend
   (scripts/serve_chat_demo.sh) on an RTX 3070, the SmolLM2-360M-Instruct
   worker (Llama architecture, 2024). The exact request:
     POST /api/generate  prompt = the ChatML-wrapped chat turn for
       "What is the capital of France?", model = smollm2-360m-instruct.
   The reply, "The capital of France is Paris.", and its 8 token ids were
   then re-derived by the INDEPENDENT numpy oracle (helix-llm/tools/
   llama_numpy_ref.py, original public weights, shares no code with Helix)
   via POST /api/verify and matched the GPU token-for-token: 8 / 8.

   VERBATIM from that capture:
     - tokenize event (the real ChatML token ids/strings the model read)
     - every token event (id / string / logit / context_len) -- logits REAL for all
     - done event (seconds 2.252, tok_per_s 3.553, text, gen_ids, nonfinite 0)
     - the live verify verdict (PASS 8 / 8 vs the numpy oracle)
     - the event histogram (8 forward_begin / 256 layer_begin / 4352 op /
       256 layer_end / 16 head / 8 token = 4900 SSE events; 32 layers x 17 ops)
   The replay's layer/op sweep re-walks this real structure; its on-screen
   pacing is time-compressed and the page says so. Real pacing was
   ~0.28 s/token (2.252 s / 8 tokens) -- a 360M model is far
   faster than the 1.5B GPT-2-XL leg.

   GPT-2-XL is still a real, supported model in this demo (switch to it in
   live mode); this default replay simply leads with the modern, capable
   instruction-tuned chat model.
   ============================================================ */
window.HELIX_CAPTURED_RUN = {
  schema_version: 1,
  kind: "captured-replay",
  captured_from: "live POST /api/generate on serve_chat_demo.sh (RTX 3070), SmolLM2-360M-Instruct; verified 8/8 vs the independent numpy oracle via /api/verify",
  model: "smollm2-360m-instruct",
  params: "360M",
  arch: "llama",
  n_layer: 32,
  n_head: 15,
  n_kv_head: 5,
  d_model: 960,
  d_ff: 2560,
  n_vocab: 49152,
  device: "RTX 3070",
  sm: "sm_86",
  precision: "fp32",
  build: "forward-only",
  ptx_bytes: 57633,
  ptx_entries: 14,
  chat: true,
  prompt: "What is the capital of France?",
  reply: "The capital of France is Paris.",
  n_gen: 8,

  /* the REAL ChatML tokenization the instruct model read (system + user turn
     + the assistant cue). Chat models wrap your message in this template. */
  tokenize: {
    ids: [1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 3511, 308, 34519, 28, 7018, 411, 407, 19712, 8182, 2, 198, 1, 4093, 198, 1780, 314, 260, 3575, 282, 4649, 47, 2, 198, 1, 520, 9531, 198],
    strings: ["<|im_start|>", "system", "\n", "You", " are", " a", " helpful", " AI", " assistant", " named", " Sm", "ol", "LM", ",", " trained", " by", " H", "ugging", " Face", "<|im_end|>", "\n", "<|im_start|>", "user", "\n", "What", " is", " the", " capital", " of", " France", "?", "<|im_end|>", "\n", "<|im_start|>", "ass", "istant", "\n"],
    n_prompt: 37,
    s_pad: 64
  },

  /* every generated token, verbatim -- id / string / logit / context_len.
     Logits are REAL for all tokens (the live stream retained them). The final
     <|im_end|> (id 2) is the chat stop token; the page hides it from the reply. */
  tokens: [
    { id: 504, string: "The", logit: 16.19659, context_len: 38 },
    { id: 3575, string: " capital", logit: 23.05727, context_len: 39 },
    { id: 282, string: " of", logit: 27.43804, context_len: 40 },
    { id: 4649, string: " France", logit: 24.46700, context_len: 41 },
    { id: 314, string: " is", logit: 24.18748, context_len: 42 },
    { id: 7042, string: " Paris", logit: 24.32419, context_len: 43 },
    { id: 30, string: ".", logit: 19.15237, context_len: 44 },
    { id: 2, string: "<|im_end|>", logit: 18.48693, context_len: 45 }
  ],

  done: {
    n_prompt: 37,
    n_gen: 8,
    n_total: 45,
    seconds: 2.252,
    tok_per_s: 3.553,
    text: "The capital of France is Paris.<|im_end|>",
    gen_ids: [504, 3575, 282, 4649, 314, 7042, 30, 2],
    nonfinite: 0
  },

  /* the live token-for-token verdict that was actually computed for THIS run
     by the independent numpy oracle (not invented, not a placeholder). */
  verify: {
    verdict: "PASS",
    matched: 8,
    total: 8,
    oracle: "numpy fp32 (smollm2-360m-instruct)"
  },

  event_histogram: {
    hello: 1, tokenize: 1, forward_begin: 8, embed: 1,
    layer_begin: 256, op: 4352, layer_end: 256, head: 16, token: 8, done: 1
  },
  events: null
};
