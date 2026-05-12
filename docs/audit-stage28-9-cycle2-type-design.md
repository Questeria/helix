# Audit Stage 28.9 cycle 2 — Type design

**Scope.** Verify cycle-1 D1 fix (`477f025`) and the new type-level surface
introduced by F1 sticky overflow flag (`dd2bc76`) and F2 tag-50 walker arms
(`50eeef0`) in `helixc/bootstrap/kovc.hx`. HEAD = `dd2bc76`.

**Criterion.** Pass = ZERO findings at confidence >=75%.

## Result: PASS (0 findings >=75%)

### Type: `diag_arena` entry (kovc.hx:2052-2213)

D1 rename is clean. Header doc, parameter `ast_node_idx`, and accessor
`diag_get_ast_node_idx` now agree with all 5 emit sites (lines
2306/2313/2489/2535/2649/2773). No dead `diag_get_src_offset` callers
linger. Encapsulation 8/10, Expression 9/10, Usefulness 8/10,
Enforcement 7/10.

### Type: `diag_arena` overflow flag (kovc.hx:2120-2179, 6215-6239)

New invariant: slot at `diag_state + 2 + cap*4` is a sticky 0|1. Set
inside `diag_emit` overflow branch; read by `diag_arena_overflowed`;
queried in `emit_elf_for_ast_to_path` AFTER validation completes. Cap
is read fresh on each access so a future `cap` change does not break
addressing. Slot is zero-initialized by `__arena_push(0)` at
`diag_arena_init`. No external mutators. Encapsulation 9/10,
Expression 9/10, Usefulness 9/10, Enforcement 8/10.

### Type: AST_TUPLE_LIT walker contract (kovc.hx:2395-2410, 2714-2727)

Tag-50 arm reads `p2` as head of AST_TUPLE_CONS chain (verified
against parser.hx:3119-3146: slot 1=child_expr, slot 2=next_cons, 0
terminator). Loop guard `cur != 0` matches the chain's null
terminator. p1 (arity) correctly NOT walked. Mirrored verbatim in
both walkers — no drift. Encapsulation 8/10, Expression 9/10,
Usefulness 9/10, Enforcement 7/10.

## No prior cycle-1 findings re-flagged.

Relevant file: `C:/Projects/Kovostov-Native/helixc/bootstrap/kovc.hx`
