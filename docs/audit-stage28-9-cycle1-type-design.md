# Audit Stage 28.9 cycle 1 — Type design

**Scope.** Stage 28.9 (validation passes ported into kovc.hx). Files
under audit: `helixc/bootstrap/parser.hx`, `helixc/bootstrap/kovc.hx`,
with cross-reference to `helixc/frontend/{parser.py, panic_pass.py,
deprecated_pass.py, trace_pass.py, check.py}`.

**Criterion.** Pass = ZERO findings at confidence ≥ 75%.

## Result: FAIL (1 finding ≥ 75%)

### Type: `diag_arena` entry (kovc.hx:2052-2154)

#### Finding D1 — slot-2 contract violated at every emit site

**Confidence: 82%.**

Slot 2 of each entry is documented (kovc.hx:2063-2066) as
`src_byte_start (the source-byte offset, for line/col reconstruction)`.
The accessor `diag_get_src_offset` (kovc.hx:2134) and the
`emit_trap_with_id(first_code)` consumer downstream all rest on the
invariant "slot 2 is a source-text byte index."

Every emit site passes an **AST node arena index** in this slot, not a
source-byte offset:

- `panic_pass`: `diag_emit(diag_state, 28501, 2, idx, 1|2)` — `idx`
  is the AST_CALL node's arena offset (kovc.hx:2247, 2254).
- `unwind_pass`: `diag_emit(diag_state, 28502, 2, fn_idx, fn_name_s)`
  — `fn_idx` is the AST_FN_DECL arena offset (kovc.hx:2414). The
  byte_start lives in `aux` instead.
- `trace_pass`: same shape — slot 2 = `fn_idx`, aux = byte_start
  (kovc.hx:2460).
- `deprecated_pass`: `diag_emit(diag_state, 28701, 1, idx, p1)` —
  `idx` is the AST_CALL arena offset; p1 = callee byte_start
  (kovc.hx:2564).

The two namespaces are disjoint (AST indices reference the bytecode
arena starting at ~0 and growing; source byte offsets index the
program-text buffer). A future "driver prints line/col" pass that
reads `diag_get_src_offset` will produce gibberish locations. The
working byte-start is consistently stashed in `aux` (slot 3), so a
correct contract should either (a) rename slot 2 to `ast_node_idx`
and slot 3 to `src_byte_start` (swapping the doc), or (b) swap the
arguments at every emit site to pass true byte_start in slot 2.

**Encapsulation: 6/10.** Accessors are pure and well-named but
silently let callers violate the slot-2 contract.

**Invariant Expression: 4/10.** The header docstring states one
contract; every emit site implements another. Self-documenting
fails by construction.

**Invariant Usefulness: 8/10** (if expressed honestly).
Code/severity/byte-offset/aux is a sensible 4-tuple.

**Invariant Enforcement: 3/10.** No compile-time or runtime check
that the arg labelled `src_byte_start` is in the source-text range.

**Recommended fix (minimal).** Either rename the slot in the
docstring + accessor (`diag_get_ast_node_idx`) and add a separate
slot 4 if a byte_start is wanted later, or change every emit site to
pass `__arena_get(fn_idx + 1)` / equivalent byte_start in slot 2 and
move the AST idx into `aux`. The bigger fix is one-line per emit
site.

---

## Other observations (confidence < 75%, NOT findings)

- **OBS-A (≈70%).** `unwind_pass` / `trace_pass` walk every entry in
  the fn_list including monomorphized clones, which inherit slot
  11/10 from their template (parser.hx:4113-4124). A
  `@unwind fn f<T>(...)` with N instantiations emits N+1 diags
  (template + N clones); the Python pass loops over `prog.items`
  pre-mono and emits exactly 1. `deprecated_pass` correctly guards
  with `if is_generic == 0` (kovc.hx:2676); the two reserved-attribute
  passes do not. Borderline: behavior is still "correct" in that no
  diag is missed, but error counts diverge from the Python oracle.

- **OBS-B (≈55%).** Slot inventory in the parser.hx header (lines
  37-42) still documents AST_FN_DECL as ending at slot 8
  (`is_checkpoint`). Slots 9-11 added by Stage 28.9 are not
  reflected. Doc-drift, not an invariant violation.

- **OBS-C (≈60%).** `diag_emit` overflow path calls
  `emit_trap_with_id(28999)` (kovc.hx:2112) which writes 7 raw code
  bytes via `emit_byte` into the SAME shared arena that holds
  validation-phase state. Validation runs before `elf_start` is
  captured (kovc.hx:6045), so these orphan bytes never enter the
  produced ELF, but they DO bump `__arena_len` and shift the layout
  of state allocated after diag_state (e.g. fn_type_state). Triggers
  only on > 64 diags; not flagged at threshold.

- **OBS-D (≈55%).** `dep_tab` (kovc.hx:2509-2523) silently drops at
  cap=16. Matches the Python "no cap, never fail" warning-only
  convention well enough that this is sub-threshold. The hard-trap
  approach of `diag_arena` (cap 64 → trap 28999) and the silent-drop
  approach of `dep_tab` are deliberately asymmetric: deprecation
  itself is a warning channel, so dropping is consistent with the
  pass's severity stance.

## Focus item resolution

1. **Slot indices 9/10/11 collision check.** No collision. All 6
   AST_FN_DECL allocation sites (parser.hx:1860, 4116, 4818, 5179,
   5481, 5712) push exactly 8 trailing slots, landing slots 9/10/11
   consistently. The capture-and-clear of sb+75..77 in
   `parse_fn_decl` happens BEFORE `mk_node(14, ...)` and the
   subsequent 8 pushes are contiguous. Pass.

2. **`diag_arena` data layout consistency.** Format is consistent
   (4-slot entry, count+cap header) but slot 2's documented contract
   is violated at every emit site — see Finding D1.

3. **Severity-2 → ud2 trap distinction.** Correctly fail-fast vs
   silent-pass: `diag_arena_error_count` counts only severity == 2
   (kovc.hx:2148-2153); driver emits `emit_trap_with_id(first_code)`
   into main's prologue iff > 0 (kovc.hx:6105-6113). Severity-1
   warnings accumulate in arena but do not trap. Pass.

4. **Bootstrap vs Python parser-attribute parity.** Bootstrap
   captures `@deprecated` / `@trace` / `@unwind` as boolean flags;
   Python stores them in `attrs: list[str]` with optional
   `"deprecated:<msg>"` payload. Bootstrap does NOT preserve the
   `@deprecated("msg")` message arg (consumed by the generic paren-
   skip at parser.hx:3704-3715). This is explicitly documented as a
   Phase-0 limitation (kovc.hx:2480-2486) so it is by-design, not a
   defect. Pass.

5. **`dep_tab` cap-overflow alignment.** Bootstrap silently drops at
   16 (kovc.hx:2511-2515); Python uses an unbounded `dict[str, str]`
   and never overflows. The two stances are not numerically equal
   but converge on "deprecation must not break the build," so this
   is convention-consistent. Sub-threshold.

## Verdict

**One ≥ 75% finding (D1).** Recommend cycle 2 to address slot-2
contract before promoting Stage 28.9.
