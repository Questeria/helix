# Fable demo-agentic run notes (2026-06-10, branch fable/demo-agentic)

One lesson per entry; why it mattered.

1. **Oracle-first paid off again.** Running the numpy oracle's greedy CHAT generation on
   SmolLM2-360M-Instruct (templated prompt, special tokens) BEFORE any C work produced a real
   assistant answer ("The capital of France is Paris.<|im_end|>") — validating dims (32L/960/15:5),
   the ChatML template, special-token encoding, and eos semantics in one cheap run.

2. **The instruct model needed ZERO kernel/arch changes.** 360M-Instruct's head_dim is 64 (same
   attention scale), all dims %64, GQA group 3, theta 1e5, eps 1e-5 (kernel-baked), tied
   embeddings, same 49152 vocab. The entire new capability is: special-token tokenization +
   eos-stop + the template. Check the config FIRST — the answer to "how hard is this model?"
   is in config.json, not in the code.

3. **Pin the eos convention explicitly.** "Append eos then stop" (not "stop before emitting")
   must match EXACTLY between oracle and worker or token-for-token fails on length. One line
   in each, same convention, gate-verified.

4. **--generate sized buffers assuming a ~5-token prompt.** Templated chat prompts are ~37
   ids; the fixed `Tmax = 5 + Ngen + 4` would overflow Spad. Found by REASONING about the
   change (not by a crash) — pre-read the ids file for the real count. Old assumptions about
   prompt size hide in buffer sizing.

5. **The git-bash heredoc strips one backslash level.** `\\n` in a python heredoc became `\n`
   (a literal newline) inside string literals — broke a C fprintf, a JS regex, and a bash
   multi-line anchor THREE separate times. For patch scripts: construct backslashes with
   chr(92), avoid backslashes in anchors, and re-parse after every write.

6. **CRLF files break multi-line LF anchors silently.** gpt2_serve_http.c and demo/index.html
   are CRLF; llama_model_gate.sh and gpt2_infer.c are LF. Normalize to LF in memory, patch,
   restore the file's own convention on write. Check `raw.count(b'\r\n')` BEFORE patching.

7. **Shared ref/ filenames are a cross-model collision.** The 360M gate run overwrote the
   135M refs the serve smoke depended on. Regenerate refs per model inside any gate that
   compares more than one model (the smoke now does), or scope filenames by model.

8. **The C tokenizer needed only an ENCODE-side change for specials.** The decode table
   already contains <|im_start|>/<|im_end|> (they're ordinary vocab entries); only BPE encode
   can't produce them. A wrapper that splits on special literals and BPEs the gaps —
   leaving encode_bytes itself untouched — kept the gpt2 path provably unchanged
   (specials are opt-in per worker via HX_SPECIALS).

9. **The smoke's instruct leg proves THREE things at once over real HTTP:** C-tokenizer
   special parity (the templated prompt's 37 ids match the oracle), numerical parity (the
   continuation ids match), and the eos-stop (the stream ends at id 2). One leg, the whole
   chat contract.
