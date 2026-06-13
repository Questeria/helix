# Helix v1.5 — Definition of Done (DRAFT, 2026-06-13)

> **STATUS: DRAFT charter.** This document is the falsifiable finish line for the **v1.5 slate**.
> It is a planning artifact — *nothing here is built yet.* Each component is DONE only when its
> measurable acceptance test passes its stated gate, honestly, with no lowered bar.

## Version baseline (read first)

- **v1.4 is ALREADY SHIPPED** (tag `v1.4` @ commit `2a5ab86`, 2026-06-10): "verifiable execution
  generalizes to a modern model + Helix goes public" — the SmolLM2-135M (Llama: GQA/RoPE/SwiGLU/
  RMSNorm) leg verified token-for-token vs an independent numpy oracle, plus the public 299bytes.com
  site. HEAD `a992ad3` (branch `main`) = that release + demo/site commits.
- This slate is therefore the **NEXT version = v1.5** (working label; final v1.5-vs-v2.0 is the
  owner's call). **Preserve ALL existing tags** (`v0-pre-k4-full-with-python`, `v1.0`, `v1.1`,
  `v1.2-complete`, `v1.3-release`, `v1.4`); tag new work `v1.5-*` only — never reuse or move a tag.
- **Hardware/precision baseline:** single RTX 3070 Laptop, `sm_86`, **fp32 only**; complete to
  **PTX text, not SASS** (below PTX trusts NVIDIA `ptxas`/driver). bf16/f16 *scalar* arith ships
  (v1.3 V4); there is **no low-precision tile/GEMM path and no FP4/ternary type** yet.

## The v1.5 thesis

Push Helix's one unique edge — **verifiable computation rooted in a 299-byte-rebuildable
toolchain** — in two directions at once: **to the frontier** (verifiable *low-precision* execution:
ternary and FP4, where silent numerical divergence hides) and **deeper** (from *empirical*
token-for-token matching toward *certified* equivalence and *succinct re-derivable receipts*). Each
is a genuine "no one has done this" at the from-scratch-trust × ML intersection.

## The universal gate (every code commit must pass it, before commit)

Inherited from v1.0–v1.4, unchanged:
1. **Self-host fixpoint byte-identical** — `seed → K1 → K2 → K3 → K4`, `K2==K3==K4` and `== pinned`
   (`scripts/gate_kovc.sh`). A feature the self-host source does not *use* keeps the pinned SHA
   (how bf16/f16/closures landed); a feature the source uses MOVES the SHA but must stay 3-way
   byte-identical.
2. **GPU PTX text regressions byte-identical** — the committed `vector_add_kernel.ref.ptx` and
   `tiled_matmul_kernel.ref.ptx` re-emit byte-for-byte (or are re-minted + re-committed *with a
   reason* if a change is intentional). ptxas-free, GPU-free, pure text.
3. **Feature corpus green** — the 109-program corpus (growing) all pass on the self-hosted K2;
   new components ADD rows, never weaken existing ones.
4. **gcc diverse-double-compile** — K1 byte-identical from gcc and the from-raw seed (`reproduce_trust.sh` leg [4]).
5. **Python-free fence** — exactly **1** committed `.py` (the fenced numpy oracle); new C/H host
   files counted + justified as zero-arithmetic, outside the fixpoint.

**Plus, per component, an EXECUTION-correctness gate** (this is the new v1.5 surface): the
component's kernel(s), run on the RTX 3070, match an **independent numpy oracle** within a stated
numeric tolerance (or token-for-token for inference), **with a load-bearing negative control** (a
deliberately corrupted kernel/weight is CAUGHT). Modeled on `scripts/capstone_audit.sh`. SERIAL
builds; commit ONLY green; never ship red; never fake.

## The components — per-component Definition of Done (sequenced by tractability)

| # | Component | Measurable DONE test | Gate | Honest residual / scope |
|---|-----------|----------------------|------|--------------------------|
| **S0** | **Ternary (BitNet b1.58, {-1,0,+1}) first-class type + verified ternary matmul** | A `ternary` element type exists in the type system (lexer ident + parser + `expr_type`/`ty_ident_to_tag` tag + codegen), with a packed representation; a `kovc`-emitted ternary matmul kernel runs on the RTX 3070 and matches an independent numpy ternary-matmul oracle to a stated tolerance (target: exact for integer-accumulated ternary·activation), with a corrupted-kernel negative control caught; >=3 new ternary corpus rows green. | Universal gate (fixpoint byte-identical — self-host source doesn't use ternary; PTX regressions byte-identical; corpus +ternary rows) **+** the ternary-matmul oracle execution gate. | No new HW needed (add/sub/int on sm_86). Claim is **trust, not speed**; a small ternary model demo, NOT modern capability. |
| **S1** | **fp16/bf16 TILE/GEMM compute path** (the dequant/compute target FP4 needs) | A `kovc`-emitted fp16 (and/or bf16) tiled matmul kernel runs on the RTX 3070 and matches the numpy oracle within fp16 tolerance, negative control caught; corpus rows added. (bf16/f16 *scalar* arith already ships — this is the *tensor* path.) | Universal gate + the fp16/bf16 GEMM oracle execution gate. | fp16 accum precision stated; no Tensor-Core MMA required for correctness (perf later). |
| **S2** | **MXFP4 first-class storage type + verified dequant→matmul** | An `mxfp4` tensor type (OCP: E2M1 4-bit elements + one shared **E8M0** 8-bit scale per 32-block); pack/unpack + block-scale; a dequant→(fp16/bf16)→matmul path whose result matches a numpy MXFP4 oracle within stated tolerance, negative control caught; the 4-bit storage footprint is measured + reported. | Universal gate + the MXFP4 dequant→matmul oracle execution gate. | On `sm_86` this is **storage + dequant** (memory win), NOT native FP4 Tensor-Core throughput (that needs Blackwell). Never imply FP4 speed parity. |
| **S3** | **NVFP4 first-class storage type + verified dequant** | An `nvfp4` tensor type (E2M1 + **FP8 E4M3** micro-scale per 16-block + FP32 per-tensor); pack/unpack + two-level scale; verified dequant vs a numpy NVFP4 oracle within tolerance, negative control caught. | Universal gate + the NVFP4 dequant oracle execution gate. | **Native FP4 MMA needs Blackwell (sm_100/sm_120)** — NOT available on this box, so the *speed* leg is explicitly DEFERRED and labeled; only the format + verified dequant land here. |
| **#2** | **Certified / translation-validated ML kernels + verifiable autodiff** | A per-compile pass emits, for a target kernel (start: matmul / softmax / layernorm), a machine-checkable **equivalence witness** that the emitted kernel computes its spec within a verified numerical envelope — *beyond* empirical token-match — plus a check that the backward kernel is the derivative of the forward (extend the existing finite-difference check toward a certificate). | Universal gate + the witness-checker runs green on the target kernel(s); negative control (a wrong kernel) is REJECTED by the witness. | Full formal PTX+IEEE-FP semantics is multi-week research; v1.5 DONE = the FIRST witness pass + its falsifiable checker on a named kernel set, honestly scoped. |
| **#4** | **Succinct / ZK verifiable-inference receipts** | A receipt format + an independent checker such that "this model on this input produced this output" is verifiable **without re-running the full model and without trusting the runner**, faster than re-execution OR with a re-derivable transcript; checker rejects a forged receipt. | Universal gate + the receipt checker green on a real inference + a forgery negative control. | Full ZK-SNARK proving is expensive + an active field (EZKL/Modulus/Giza); v1.5 DONE = the first re-derivable/succinct receipt increment with a fast independent checker; full ZK is the labeled stretch. |
| **#3** | **Verifiable PTX→SASS (kill the last trusted closed binary)** | A from-scratch, verifiable PTX→SASS path (or verified SASS validator) for the emitted `sm_86` kernel subset, so the hex0→…→PTX→**SASS** chain has no trusted closed `ptxas` in it for that subset; output verified against a reference. | Universal gate + the SASS path/validator green on the kernel subset, with a negative control. | SASS is proprietary/reverse-engineered + arch-specific — **the hardest**. v1.5 DONE = the first verifiable increment on a named kernel subset; full coverage is multi-week+ and labeled. |

## Honest completion policy

- **Realistically completable this session:** S0 (ternary) is the first shippable win; S1 (fp16/bf16
  GEMM) is plausible. S2/S3/#2/#4/#3 are research-grade — expect **honest gated partial progress**,
  not necessarily full completion overnight.
- **NEVER fake "done."** A component is DONE only when its row's test passes its gate with a
  negative control. Partial progress is logged honestly here + in the tracker each tick.
- **v1.5 is DONE** only when every component row is green + several consecutive clean independent
  adversarial audits. At that point: finalize this doc, tag `v1.5-complete`, update
  `docs/TRUST_CHAIN_CLOSED.md` honestly, and announce. Until then the self-paced loop keeps going.
- **The v1.5-complete BAR (fixed 2026-06-13, owner-deferred to the loop):** "every row green" means
  each row's OWN stated test passes its gate + negative control — and for the three research-grade
  rows that test IS a *first increment*, not full coverage: **#2** = a per-compile equivalence
  witness + its falsifiable checker on ONE named kernel (start: matmul), a wrong kernel REJECTED;
  **#4** = ONE re-derivable/succinct inference receipt + a fast independent checker, a forged receipt
  REJECTED; **#3** = a verifiable PTX->SASS path (or SASS validator) on ONE named sm_86 kernel,
  output verified + a negative control. So the stopping condition is: **S0-S3 fully green + #2/#4/#3
  each landed as a labeled first increment + the audit streak** — NOT full formal/ZK/SASS coverage
  (those are explicitly post-v1.5). v1.6 (verifiable inference of a much larger model on
  should-be-impossible hardware: quantize + layer-stream/offload + verified-correct + a receipt) and
  v1.7 (speed: same-or-better capability, made fast) build directly on this — #4's receipts + #2's
  certified kernels are exactly what make a streamed-70B run on an 8GB laptop GPU *provable* in v1.6.

## Discipline (carried, non-negotiable)

Claude-subscription only; never read `C:/Projects/Neptune/api.env`; from-raw + Python-free toolchain
(exactly 1 committed `.py`); full-gate before every CODE commit; commit only green; SERIAL builds;
never force-push or skip hooks; preserve all existing tags; DDC-clean; honest always. The 299bytes
*product* and the *company* are out of scope for this doc — this is the Helix substrate roadmap only.

*Baseline grounded 2026-06-13 against: HEAD a992ad3; `git ls-files "*.py"` == 1
(`verification/oracle/oracle_train.py`); `scripts/reproduce_trust.sh` + `scripts/gate_kovc.sh`
(CPU/x86, ptxas-free); `docs/HELIX_V1_DEFINITION_OF_DONE.md` (template); `docs/TRUST_CHAIN_CLOSED.md`;
the bf16/f16 dtype machinery in `helixc/bootstrap/kovc.hx`. v1.4 tag annotation read live.*

---

## S0 ternary — implementation plan (synthesized from the read-only probe `wf_8c943229-120`, 2026-06-13)

Three read-only agents mapped the type-system, the PTX/tile emitter, and the gate/oracle/corpus
surfaces. Decisions + two gate-able increments:

**Design decisions**
- **Type tag = 12** (next free; 0–11 taken, 12+ reserved — namespace comment `kovc.hx:1632`). Ident
  provisionally `t2` (confirm no collision in `ty_ident_to_tag`, `parser.hx:1658`).
- **Scalar domain = i32.** {-1,0,+1} ⊂ i32, so ternary scalar values + arithmetic fall through to the
  existing i32 codegen — NO new x86 scalar binop cascade, NO out-of-domain trap. Ternary's
  distinctness is (a) packed storage and (b) the GPU **integer-accumulate** matmul (the BitNet
  kernel) — those are the substantive new work, not scalar arithmetic.
- **GPU path = a NEW self-contained fused intrinsic `__ternary_matmul_smem`**, NOT a dtype threaded
  through the f32 emitters. Mandatory: the committed `vector_add_kernel.ref.ptx` +
  `tiled_matmul_kernel.ref.ptx` are byte-compared by the gate, so the shared f32 emitters
  (`emit_ptx_fma`, `emit_ptx_ld_shared`, `emit_ptx_gload_f32`, the stride-4 literals) must NOT change.
  The new intrinsic is a copy of `emit_ptx_tiled_matmul_smem` (`kovc.hx:12969`) with int32
  accumulators (existing `add.s32` @12387 + a tiny new `sub.s32`/neg), a new int shared/global load,
  ≤6 args, reusing the dtype-agnostic `cp.async.cg16` (16-byte) staging.

**Increment 1 — CPU only (no GPU / oracle / .ref.ptx): the first gate-able win**
- Edits: `t2`→12 arm in `ty_ident_to_tag` (`parser.hx:1658`); reverse arm in `ty_tag_push_name`
  (`parser.hx:1201`); a `type_width_class` arm for tag 12 (`kovc.hx:1914`, width 4 = i32-like).
- Corpus: add `stage0/helixc-bootstrap/corpus_gen/trit_dot.hx` (a {-1,0,+1} dot product,
  runtime-accumulated, full-i32-compared → 42/0 sentinel; template = `corpus_gen/V4_bf16_add.hx`).
  Wire `chk "$GENC/trit_dot.hx" 42` into `gate_kovc.sh` leg [4] (~line 615).
- **Lockstep count bumps (THREE places or reproduction breaks):** `gate_kovc.sh:728` numeric floor
  109→110; the expect-string `gate_kovc.sh:616`; the exact literal `CORPUS: 109 passed, 0 failed` at
  `reproduce_trust.sh:123` → 110. Dated bump-log comment (existing style).
- Gate: K2==K3==K4 byte-identical (self-host source never uses `t2`, so 3-way identical, but the
  pinned `EXPECT_FIX` @`gate_kovc.sh:117` + `FIX_SHA` @`reproduce_trust.sh:69` MOVE → re-mint +
  re-commit the pinned SHA *with a reason*, the V1–V4 pattern) + vector_add/tiled PTX byte-identical +
  corpus 110/0. Commit LOCALLY.

**Increment 2 — GPU + oracle (completes S0's DoD)**
- New `__ternary_matmul_smem` emitter + name-branch in `emit_ptx_call` (`kovc.hx:14628`);
  `helixc/examples/ternary_matmul_kernel.hx` + freshly-minted committed `.ref.ptx` (eol=lf auto) + a
  provenance grep; ≥2 more corpus rows.
- Independent numpy ternary-matmul oracle: fold into the single committed `oracle_train.py`
  (env-gated, byte-identical default) OR runtime-generate a fenced `/tmp` witness — keep
  `committed .py == 1`.
- GPU-vs-oracle execution check on the RTX 3070 with a **load-bearing negative control** (corrupt a
  weight, assert not-a-no-op, assert the compare then FAILS), modeled on `capstone_audit.sh`
  NC-PERTURB; **exact-int** compare (not within-2%) with a ≥N-element vacuous-pass guard.
- Avoid a new host `.c` (the `committed .c/.h == 29` fence) — reuse `cuda_launch.c`'s positional
  `void*[]` arg convention.

**Risks carried:** keep ternary OUT of the self-host source (corpus-only) or the fixpoint story
changes; the corpus count is triple-booked (bump all three in lockstep); mint the `.ref.ptx` only
once the emitter is frozen; declare nothing "done" until the GPU-vs-oracle check + its negative
control are green.
