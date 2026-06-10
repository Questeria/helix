# Live-coding run-of-show — "Verifiable execution for AI" (Anthropic talk)

**The one-line purpose:** prove the stack beneath an AI can be *audited instead of trusted* —
and that the verification has real teeth (it catches a real bug, live).

**Total: ~10 minutes.** All commands run under WSL in `C:/Projects/Kovostov-Native`.
The live loop is `bash scripts/talk_demo.sh gate` (REHEARSED: ~8 s per cycle once prewarmed; the full green->plant->RED->restore->green sequence verified on this machine 2026-06-09).

## Before the talk (once, ~5 min)
1. Stop anything on the GPU (the chat demo): check `bash scripts/talk_demo.sh status` → "GPU: free".
2. `bash scripts/talk_demo.sh prewarm` — mints the from-raw compiler driver (cached on ext4,
   survives WSL restarts) + runs a green baseline gate. After this the live gate loop is ~8 s.
3. Optional bookend: have the live chat demo on a second machine/window (`bash
   scripts/serve_chat_demo.sh`, http://127.0.0.1:8848/?source=sse) — but do NOT leave it
   running on the talk GPU (serial-GPU rule).
4. Have a fallback: a recording of one full green→red→green loop.

## The beats
1. **Frame (30 s).** "Every AI you run today sits on a stack you can't audit. This compiler
   rebuilds from 299 hand-typed bytes — and it runs real models. I'm going to code a piece
   of a modern LLM live and *prove* it ran correctly."
2. **Show the kernel (1 min).** Open `helixc/examples/gpu_silu_mul_kernel.hx` — the SwiGLU
   gate inside every Llama-family model, ~10 lines of Helix.
3. **Green baseline (30 s).** `bash scripts/talk_demo.sh gate` → the table: compiled by the
   from-raw kovc, ptxas sm_86, max-abs vs the independent oracle ~e-06, `LLAMA_GL0_PASS`.
4. **Plant the bug (2 min).** `bash scripts/talk_demo.sh plant` (or live-type the `let mut
   amag = gi` form). Tell them it's a REAL bug this project hit: the compiler aliases the
   mutable onto `gi` and clobbers it. **It compiles. ptxas accepts it. It runs.**
5. **The catch (1 min).** `gate` → `silu_mul max-abs err = 1.9e+00 -> FAIL`,
   `LLAMA_GL0_FAIL`, and the NEG-CONTROL line showing the comparator has teeth. "Compile
   passed, the GPU ran it — only *verification against an independent reference* caught it."
6. **Fix live (1 min).** `restore` (or live-edit back) → `gate` → green at 1.9e-06.
7. **The reveal (2 min).** "These three kernels aren't a toy: they're the ops of SmolLM2,
   a 2024 Llama-architecture model — which runs on this stack **token-for-token identical**
   to an independent reference (25/25), through a compiler you can rebuild from 299 bytes."
   Show `scripts/_llama_model_gate.log`'s verdict block (or re-run `llama_model_gate.sh`,
   72 s, if time allows).
8. **Close (30 s).** "Interpretability asks what a model is thinking. This makes what a
   model is *doing* — every operation — observable, reproducible, and checkable, from the
   first byte. That gate that caught my bug? The same mechanism gates an AI's actions."

## Honesty guardrails (this audience will check)
- Say "verifiable execution", never "fastest" (live XL is ~10 s/token by design; GPU is
  ~50–67 % of cuBLAS). Verified to PTX, not SASS (ptxas + driver are the trusted-once
  boundary). fp32. SmolLM2-135M is a small modern-architecture model — "modern architecture,
  verifiably executed", NOT "modern capability".
- If something fails live: that IS the demo — fail-closed beats silently-wrong. Diagnose
  honestly or fall back to the recording.
