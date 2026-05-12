# Audit Stage 28.9 cycle 96 — Type design

Scope: HEAD `56fa3df`. Narrow rotation per cycle-96 brief:

- `helixc/ir/lower_ast.py` — For / While / Loop block CFG construction
- `helixc/frontend/struct_mono.py` — `TyGeneric` param-arity check
- `helixc/backend/x86_64.py` — stack frame layout invariants

Read-only audit (Read / Grep / Glob / Bash). No code edits performed.

Prior-cycle findings C1..C95 and known deferred items are intentionally
NOT re-flagged. Stage 28.10 / 28.11 audits run independently.

## Verdict: FAIL — 1 finding at confidence ≥ 75 %

### F1 — `A.Loop` lowering builds orphan basic blocks (conf 90)

**File:** `helixc/ir/lower_ast.py:1895–1909` (in `_lower_expr`, `A.Loop`
arm).

**Type / kind:** CFG-construction invariant violation; cross-pass type
contract break between `IRBuilder` and the backend's block-iteration
contract on `FnIR.blocks`.

**Observation.** Every other CFG-building site in this lowerer
(`If/Else` at 1733–1735; `For` at 1813–1815; `While` at 1873–1875) calls
`self.builder.append_block()`, which (per `tir.py:400–406`) BOTH allocates
a fresh `Block` AND appends it to `self.current_fn.blocks`. The
`A.Loop` arm instead calls `self.builder.new_block()` for both
`header_blk` and `body_blk` (lines 1901–1902). `new_block()`
(`tir.py:379–382`) only mints a `Block` object — it does not register
it on the current function.

**Concrete consequence.** The Loop header and body blocks are
*orphans*: they exist as Python objects but never appear in
`fn.blocks`. Every downstream consumer iterates `fn.blocks`:

- Backend slot pre-allocation (`x86_64.py:908, 917, 925, 934`) skips the
  orphan blocks' params and op results, so SSA values produced inside
  the loop body have no stack slot.
- Backend block-emission loop (`x86_64.py:993`) never emits labels for
  the orphan blocks, so the BR to `header_blk.id` at line 1903 resolves
  in `x86_64.py:1835` via `next((b for b in self.fn.blocks if b.id == target_id), None)` →
  `None`, which raises `ValueError("BR to unknown block <id>")` at
  line 1837.

Net result: any program containing `loop { … }` aborts compilation in
the backend with an opaque `BR to unknown block` error rather than
either codegen'ing a real infinite loop or producing a typechecker-side
diagnostic. The error message points at the BR site, not the Loop
construct, making the failure mode hard to diagnose from a user
perspective.

**Why the test suite didn't catch it.** `grep -n 'loop\s*{' helixc/tests/`
finds exactly one hit, `test_parser.py:510`, which only exercises the
parser. There is no codegen or IR-roundtrip test for `A.Loop`, and the
prior `0066b58e` cycle that introduced this header→body→header skeleton
landed without one.

**Why it's a type-design finding (not silent-failures / not codegen).**
The `IRBuilder` API exposes two block-minting methods with overlapping
shapes but a load-bearing semantic difference (registered vs. orphan).
Their type signatures (`new_block() -> Block` vs.
`append_block() -> Block`) do not encode that difference — both return
the same `Block` type with no marker distinguishing
"attached-to-current-fn" from "free-floating." The For/While arms above
prove the contract that the lowerer expects: blocks used as branch
targets in the current fn must come from `append_block()`. The Loop arm
violates that unwritten contract; the IR-layer types don't help the
caller notice.

**Severity.** HIGH — feature is fully broken; any `loop { body }`
source produces a backend crash. Suggested fix (out of audit scope, for
maintainer reference): swap `self.builder.new_block()` to
`self.builder.append_block()` on both lines.

**Confidence breakdown.** 90 %: code is unambiguous; backend ValueError
path verified by reading; no test masks it; blame shows the
`new_block()` calls were introduced in `0066b58e` (2026-05-04) without
an accompanying append. The remaining 10 % covers the possibility
that some later pass (which I didn't enumerate exhaustively) re-walks
orphan blocks through the `IRBuilder`'s `current_block` chain — a search
of `helixc/` for `current_block` did not turn up such a pass, so the
reservation is precautionary.

## Files audited cleanly (no new findings ≥ 75 %)

- `helixc/frontend/struct_mono.py` — `_ty_key` covers the relevant
  `TyNode` arms (TyName / TyGeneric / TyTuple / TyArray / TyRef / TyPtr
  / TyFn / TyTensor / TyTile) post the cycle-77 + cycle-2 A13 fixes;
  the fall-through at 298–303 hard-raises rather than silently
  collapsing keys. `instantiate()` checks generics arity at line
  312–316 and feeds mismatches to `diags` in `monomorphize_structs`.
  Nested generic args go through `monomorphize.substitute_ty`, whose
  walker matches the `_ty_key` arms. No arity check is missing from a
  surface I could exercise.

- `helixc/backend/x86_64.py` stack-frame invariants — every slot is
  8 bytes; `_alloc_slot` / `_alloc_var` / `_alloc_array` are the sole
  mutators of `next_slot`; frame size at line 939 is
  `(-next_slot + 15) & ~15`, i.e. always non-negative and 16-byte
  aligned; the prologue at 944–948 emits `push rbp; mov rbp, rsp; sub
  rsp, frame_size` and the epilogue at 1823–1826 restores via
  `mov rsp, rbp; pop rbp; ret`. The pre-allocation pass at 907–937
  walks `fn.blocks` exhaustively, so under the (separately-flagged) F1
  caveat — that Loop blocks are not in `fn.blocks` and therefore never
  visited — every SSA value in registered blocks gets a slot before
  emission. No layout invariant in the backend itself is violated.

## No edits performed

This audit only reads source. The single Write call produced this
document.
