# Audit Stage 28.9 cycle 4 — Type design

**Scope.** Stability re-pass over the cycle-2/3 audit surface in
`helixc/bootstrap/kovc.hx`. HEAD unchanged at `dd2bc76` (no commits since
cycle 2). Cycle 3 was CLEAN. Re-verified the three type contracts.

**Criterion.** Pass = ZERO findings at confidence >=75%.

## Result: PASS (0 findings >=75%)

### Type: `diag_arena` entry (kovc.hx:2052-2213)

D1 rename still consistent. Header doc (2064), `ast_node_idx` parameter
in `diag_emit` (2150), and accessor `diag_get_ast_node_idx` (2193) agree.
All emit sites (2306/2313/2489/2535/2649/2773) pass AST arena indices.
No `diag_get_src_offset` callers. No drift.

### Type: `diag_arena` overflow flag (kovc.hx:2120-2179, 6215-6239)

Sticky 0|1 invariant at `diag_state + 2 + cap*4` holds. Zero-init via
`__arena_push(0)` (2122); set-only in `diag_emit` overflow branch (2156);
read via `diag_arena_overflowed` (2176) with fresh `cap` re-read. Codegen
gate at 6226 still prioritizes overflow over error count. No external
mutators introduced.

### Type: AST_TUPLE_LIT walker contract (kovc.hx:2395-2410, 2714-2727)

Tag-50 arms in `walk_for_panic` and `walk_for_deprecated` remain byte-
identical. Chain traversal of AST_TUPLE_CONS (slot 1 = child_expr,
slot 2 = next) with `cur != 0` terminator matches parser contract. p1
(arity) correctly not walked.

## Stability

No cycle-1/2/3 findings re-flagged. No new commits. Two consecutive
CLEAN type-design passes (cycles 3, 4) at HEAD `dd2bc76`.

Relevant file: `C:/Projects/Kovostov-Native/helixc/bootstrap/kovc.hx`
