# Style Guide — *Helix: The Complete Guide*

This guide is binding for every chapter author of *Helix: The Complete Guide*. It exists so
the book reads as one voice and so every claim in it is true, grounded, and reproducible.

Helix is a from-raw-binary self-hosting compiler and language whose entire reason for existing
is honest, auditable trust (see [`docs/TRUST_CHAIN_CLOSED.md`](../TRUST_CHAIN_CLOSED.md)). A
book about that system that overclaims, paraphrases a command wrong, or invents an API would
defeat its own subject. Treat the rules below as load-bearing, not stylistic preference.

---

## 1. Audience — write for two readers at once

Every chapter is read by **two** audiences, and you must serve both:

1. **A human developer** — someone evaluating Helix, building it, or learning the language.
   They want narrative, motivation, worked examples, and a clear "why."
2. **An AI operator (agent)** — an LLM-driven agent that will *drive* Helix: invoke the build,
   run the gate, compile `.hx` programs, and reason about the trust chain. It wants exact
   commands, exact paths, exact invariants, and explicit do-not-do rules.

Write the main prose for the human. Where the AI operator needs *different or extra* guidance —
a non-negotiable invariant, a trap that only matters when scripting, an exact string to match —
add a blockquote callout:

```markdown
> **For AI agents:** the gate prints the literal token `GATE_PASS` on success. Match that exact
> token, not a paraphrase — `grep -q '^GATE_PASS'`, as in `scripts/reproduce_trust.sh`.
```

Render it as a real Markdown blockquote starting with the bold label `> **For AI agents:**`.

- Use the callout **only when the AI-specific guidance differs** from the human guidance. Do not
  duplicate the prose with "and agents should also…". If the advice is identical for both, write
  it once in the main text.
- Keep AI callouts imperative and checkable: an exact command, an exact token, an exact path, or
  an explicit prohibition.
- Part IX ("For AI Agents") is the *dedicated* operator manual. In Parts I–VIII, AI callouts are
  spot guidance; the deep operator material lives in Part IX and should be cross-linked, not
  re-explained.

> **For AI agents:** when you act on this book, prefer commands and paths quoted verbatim from a
> chapter over anything you infer. If a chapter and a repo source disagree, the repo source wins
> and the chapter is the bug — flag it, do not silently follow stale prose.

---

## 2. Markdown conventions

**Headings.** One `#` H1 per file (the chapter title). Sections are `##`, subsections `###`.
Do not skip levels. Sentence-case headings ("Build from raw", not "Build From Raw").

**Fenced code blocks — always tagged.** Every fence carries a language tag:

- ` ```helix ` for Helix (`.hx`) source.
- ` ```bash ` for shell commands, build invocations, and `scripts/*.sh` excerpts.
- ` ```text ` for program output, hashes, REPL/console transcripts, and diagrams.

Never leave a fence untagged.

**Verified examples vs fragments.** Mark every code block's status with a bold label on the line
immediately above the fence:

- **Verified example** — a *complete, compile-checked* program. A Helix "Verified example" MUST
  be a complete program with an `fn main`, MUST have been compiled (and run, where it has a
  defined exit code) before the chapter ships, and MUST cite the source path it was taken from or
  added to. State the observed result (e.g. exit code) in prose or a following ` ```text ` block.
- **Fragment** — a partial snippet illustrating syntax or one function. It is *not* expected to
  compile on its own. Never imply a fragment is runnable.

Example of the convention:

**Verified example** — `helixc/examples/exit42.hx` (compiles to a Linux ELF; `$? == 42`):

```helix
// First end-to-end Helix program: compiles, links, runs, exits with status 42.
fn main() -> i32 {
    42
}
```

For a partial snippet, label it instead:

**Fragment** (illustrates an `@pure` annotation; not a complete program):

```helix
@pure
fn loss(w: f32) -> f32 {
    let d = w - 7.0;
    d * d
}
```

**Bash blocks.** Copy commands **exactly** from the real scripts — do not paraphrase syntax,
flags, or filenames. The source of truth for build/verify commands is, in order:
[`scripts/reproduce_trust.sh`](../../scripts/reproduce_trust.sh),
[`scripts/gate_kovc.sh`](../../scripts/gate_kovc.sh), the per-rung
`stage0/<rung>/build.sh`, and [`.github/workflows/trust-reproduce.yml`](../../.github/workflows/trust-reproduce.yml).
If you show a one-command reproduction, show the real one: `bash scripts/reproduce_trust.sh`.

**Callouts.** Beyond the AI callout, use blockquotes for emphasis with a bold lead label:
`> **Note:**`, `> **Warning:**`, `> **Residual:**` (for an honest limitation). Keep them short.

---

## 3. Terminology — use these terms exactly

Consistency matters because an AI operator pattern-matches on them. Use the left column verbatim;
do not coin synonyms.

| Term | Meaning (use consistently) |
|------|----------------------------|
| **`kovc`** | The Helix compiler — the from-scratch compiler *written in Helix* (`helixc/bootstrap/{lexer,parser,kovc}.hx`) that emits x86-64 ELF directly. Lowercase, code-formatted. |
| **`seed`** | The Apache-2.0 **C-subset** compiler (`stage0/helixc-bootstrap/`, source `seed.c`) built by the raw ladder; it builds `kovc`. Not "the bootstrap compiler" — call it the `seed`. |
| **the from-raw ladder** | The hand-typed-root build chain: `hex0` (299 hand-authored hex bytes) → `hex1` → `hex2` → `catm` → `M0` → `cc_amd64` → `M2-Planet` → `seed` → `kovc`. Each rung is built **only by the prior rung**; no trusted pre-built compiler. |
| **the self-host fixpoint** | `seed → K1 → K2 → K3 → K4` with **K2 == K3 == K4 byte-identical** (pinned `0992dddd…`). The proof that `kovc` reproduces itself exactly. Say "self-host fixpoint," not "stage2==stage3." |
| **gcc-DDC** | The `gcc` **diverse-double-compile** of the `seed→K1` rung: `gcc` (zero M2-Planet ancestry) and the from-raw `seed` both produce a byte-identical K1 (`84363adb…`) — a Wheeler trusting-trust defense. `gcc` is an **auditor**, never the shipped root. |
| **the gate** | [`scripts/gate_kovc.sh`](../../scripts/gate_kovc.sh), the universal gate: self-host fixpoint + 109-program feature corpus + PTX text regression + negative diagnostics. Prints `GATE_PASS`. |
| **the capstone** | The real-capability proof: a ≥2-layer transformer trained **end-to-end on `kovc`-emitted GPU (PTX) kernels**, converging to within 2% (reproduced at ~0%) of an independent numpy oracle. |

Other fixed usages: **Helix** (the language; capital H), **`.hx`** (source extension),
**`helixc`** (the *historical* Python-hosted frontend — call it that explicitly; it is **not** in
the shipped compile/run path), **`helixrt`** (the runtime). "Complete **to PTX**" is the precise
capability claim — never "complete to GPU machine code" (see §5).

> **For AI agents:** these terms map to real files and real output tokens. When you script, key
> off the exact strings here (e.g. `GATE_PASS`, the pinned hashes `9837db12…` / `0992dddd…` /
> `84363adb…`) rather than English descriptions, and dereference the cited paths before acting.

---

## 4. Citations — link real repo paths

Ground every claim in real source, and cite it so a reader (or agent) can verify.

- Cite repo files by their path **relative to the repo root**, as a Markdown link. From a Stage-1
  chapter at `docs/book/partN-.../NN-....md`, the repo root is three levels up, so link like
  `` [scripts/reproduce_trust.sh](../../../scripts/reproduce_trust.sh) ``. From this guide
  (`docs/book/STYLE_GUIDE.md`) it is two levels up: `` [scripts/...](../../scripts/...) ``. Always
  show the repo-root-relative path as the **link text** so it reads correctly even unlinked. (Both
  forms here are shown as inline code, not live links, because the correct prefix depends on the
  citing file's depth.)
- When you quote code or a command, quote it **verbatim** and cite the exact file (and, where it
  aids verification, the line region, e.g. "`scripts/gate_kovc.sh` step `[1]`").
- Cite the canonical trust records by name where a trust claim is made:
  [`docs/TRUST_CHAIN_CLOSED.md`](../TRUST_CHAIN_CLOSED.md) (verified state + every residual) and
  [`docs/CLEAN_REPRODUCTION.md`](../CLEAN_REPRODUCTION.md) (rebuild from a clean checkout).
- Do not cite a file you have not opened. Do not cite line numbers you have not confirmed.

---

## 5. The honesty rule (non-negotiable)

This is the most important section. Helix's whole value proposition is calibrated honesty; the
book must match it.

1. **Ground every claim in real source.** If you cannot point to a file, a command, or a pinned
   hash that backs a statement, do not write the statement. Read more, or omit it.
2. **Verify every "Verified example."** Compile it (and run it if it has a defined exit code)
   before the chapter ships. If you cannot verify it, downgrade it to a **Fragment** or remove it.
3. **Never invent.** No invented APIs, flags, file paths, hashes, numbers, or behavior. If you are
   unsure whether a flag exists, read the source or omit it. Hallucination is the one unforgivable
   error in this book.
4. **State residuals where they matter, and cite them.** Match the calibrated tone of the trust
   docs. In particular, when relevant, state plainly (and cite
   [`docs/TRUST_CHAIN_CLOSED.md`](../TRUST_CHAIN_CLOSED.md)):
   - The chain is **complete to PTX, not to GPU machine code** — below PTX it trusts NVIDIA's
     closed `ptxas` + CUDA driver + GPU hardware + the C host launcher (the CPU path is
     all-the-way-down from raw binary; the GPU path is from-hex0-to-PTX-then-`ptxas`).
   - **GPU performance is a fraction of cuBLAS, not parity** (~50–67.5% on the reference RTX 3070
     Laptop, sm_86); end-to-end capstone speedup is **7.0–8.7×** (Amdahl-bound), not ≥10×. Loss
     parity (the hard gate) holds at ~0%.
   - **Single hardware target** (sm_86); no cross-arch / AMD validation.
   - The byte-identical, hash-pinned DDC covers the **`seed→K1`** surface; the broader v1.1
     language surface is cross-checked **behaviorally** (and that witness is out-of-tree, not
     clean-checkout reproducible). **External third-party reproduction on independent hardware
     remains the one open increment.**
5. **No overclaim, ever.** Pair every performance label with its honest fraction. Do not write
   "beats cuBLAS," "fully verified GPU," or "AGI" as an achieved state. When in doubt, undersell.

> **For AI agents:** if asked to add or update an example, do not assert it works until you have
> actually compiled/run it via the real toolchain. An unverifiable claim must be removed, not
> hedged. Treat `docs/TRUST_CHAIN_CLOSED.md` §R (residuals) as the ceiling on what the book may
> claim — never exceed it.
