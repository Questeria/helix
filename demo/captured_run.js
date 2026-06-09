/* ============================================================
   demo/captured_run.js — the REAL captured run that powers the
   chat page's REPLAY (backup) mode when no live backend exists
   (e.g. featured on a public website with no GPU).

   PROVENANCE (nothing here is invented):
   Every value below is transcribed from the committed log of the
   fail-closed serve gate, scripts/_gate_run.log
   ("HELIX SERVE GATE ... HELIX_SERVE_GATE_PASS", G1 capture:
   POST /api/generate {prompt:"The capital of France is", n_gen:20}
   against the live GPT-2-XL worker on the RTX 3070; served ids
   matched the offline oracle token-for-token, 25/25).

   WHAT IS VERBATIM from that capture:
     - tokenize event (ids / strings / n_prompt / s_pad)
     - token events for steps 0-3 (id / string / logit / context_len)
     - done event (seconds 195.513, tok_per_s 0.102, text, gen_ids,
       nonfinite 0)
     - gen-id -> string pairs for ALL 20 tokens (each id's piece is
       pinned by the captured samples + the captured done.text)
     - ptx_bytes 44019 (8 seed-minted .entry kernels)
     - the event histogram (1 hello / 1 tokenize / 20 forward_begin /
       20 embed / 960 layer_begin / 16320 op / 960 layer_end /
       40 head / 20 token / 1 done = 18343 SSE events)
   WHAT WAS NOT RETAINED in the log (and is therefore NOT shown as
   data during replay):
     - per-token logits beyond step 3  -> null  (UI renders "—")
     - per-layer host-clock ms         -> 0     (untimed, never faked)
   The replay's layer/op sweep re-walks the run's real structure
   (48 layers x 17 ops per token step — exactly the histogram's
   counts); its on-screen pacing is time-compressed and the page
   says so. The REAL pacing was ~9.8 s/token (195.513 s / 20 tok).

   To swap in a verbatim full-stream capture later: re-run
   scripts/helix_serve_gate.sh, keep the raw SSE body, and emit it
   here as {events:[...]} — the page prefers `events` when present.
   ============================================================ */
window.HELIX_CAPTURED_RUN = {
  schema_version: 1,
  kind: "captured-replay",
  captured_from: "scripts/_gate_run.log — HELIX SERVE GATE (G1), real served GPT-2-XL run, token-for-token vs oracle 25/25",
  model: "gpt2-xl",
  params: "1.5B",
  n_layer: 48,
  n_head: 25,
  d_model: 1600,
  d_ff: 6400,
  n_vocab: 50257,
  device: "RTX 3070",
  sm: "sm_86",
  precision: "fp32",
  build: "forward-only",
  ptx_bytes: 44019,
  prompt: "The capital of France is",
  n_gen: 20,

  /* verbatim: {"_ev":"tokenize","seq":1,...} */
  tokenize: {
    ids: [464, 3139, 286, 4881, 318],
    strings: ["The", " capital", " of", " France", " is"],
    n_prompt: 5,
    s_pad: 64
  },

  /* steps 0-3 verbatim (incl. logits); steps 4-19: id + string + context_len
     are captured (gen_ids + done.text + the pinned id<->piece pairs),
     logit was not retained -> null (rendered as "—", never invented). */
  tokens: [
    { id: 262,  string: " the",     logit: 8.79165,  context_len: 6  },
    { id: 1748, string: " city",    logit: 9.49102,  context_len: 7  },
    { id: 286,  string: " of",      logit: 12.81686, context_len: 8  },
    { id: 6342, string: " Paris",   logit: 12.99207, context_len: 9  },
    { id: 13,   string: ".",        logit: null,     context_len: 10 },
    { id: 632,  string: " It",      logit: null,     context_len: 11 },
    { id: 318,  string: " is",      logit: null,     context_len: 12 },
    { id: 262,  string: " the",     logit: null,     context_len: 13 },
    { id: 3139, string: " capital", logit: null,     context_len: 14 },
    { id: 286,  string: " of",      logit: null,     context_len: 15 },
    { id: 4881, string: " France",  logit: null,     context_len: 16 },
    { id: 290,  string: " and",     logit: null,     context_len: 17 },
    { id: 262,  string: " the",     logit: null,     context_len: 18 },
    { id: 4387, string: " largest", logit: null,     context_len: 19 },
    { id: 1748, string: " city",    logit: null,     context_len: 20 },
    { id: 287,  string: " in",      logit: null,     context_len: 21 },
    { id: 4881, string: " France",  logit: null,     context_len: 22 },
    { id: 13,   string: ".",        logit: null,     context_len: 23 },
    { id: 632,  string: " It",      logit: null,     context_len: 24 },
    { id: 318,  string: " is",      logit: null,     context_len: 25 }
  ],

  /* verbatim: {"_ev":"done","seq":18342,...} */
  done: {
    n_prompt: 5,
    n_gen: 20,
    n_total: 25,
    seconds: 195.513,
    tok_per_s: 0.102,
    text: " the city of Paris. It is the capital of France and the largest city in France. It is",
    gen_ids: [262,1748,286,6342,13,632,318,262,3139,286,4881,290,262,4387,1748,287,4881,13,632,318],
    nonfinite: 0
  },

  /* verbatim histogram of the captured SSE stream (18343 events) */
  event_histogram: {
    hello: 1, tokenize: 1, forward_begin: 20, embed: 20,
    layer_begin: 960, op: 16320, layer_end: 960, head: 40, token: 20, done: 1
  },

  /* optional verbatim full event stream — absent in this capture;
     when present the replay uses it directly instead of re-walking
     the structure. */
  events: null
};
