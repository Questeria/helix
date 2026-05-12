# Stage 30 cycle-4 audit findings

**Date**: 2026-05-12
**HEAD**: `07e6535` (cycle-3 fix landed)

## Verdicts:
- silent-failure: **CLEAN**
- type-design: 1 MEDIUM (conf 82) — TK_RBRACE catch-all conflates empty-block vs syntax-error
- code-review: **CLEAN**

## Status: NOT CLEAN — 1 MEDIUM finding

---

## MEDIUM finding: TK_RBRACE catch-all over-broad (parser.hx:3788-3798)

**Confidence**: 82

**Description**: Stage 29.2 fix changed parse_primary's catch-all to
return `AST_INT(0)` for TK_RBRACE (tag 6) instead of `AST_ERR(6)`.
This addresses the legitimate empty-block case `else {}`, but the
return type conflates two semantically distinct outcomes:

1. Valid empty-block primary (`else {}` body, want unit value 0)
2. Truncated source landing at `}` (e.g., `fn foo() -> i32 { let x = }`,
   want loud trap)

Both now produce `AST_INT(0)` indistinguishably.

**Why it matters**: For truncated/malformed sources, pre-Stage-29.2
behavior was SIGILL at runtime (loud failure). Post-Stage-29.2 the
parser silently produces `let x = 0`, then parse_let's unconditional
cursor advance consumes the function body's `}`, desyncing the cursor
and cascading into either a misleading downstream trap or a silently-
wrong binary.

This was flagged in cycle-1 as M2 (conf 78-88). Has been re-flagged
in cycle-2 (no action), cycle-3 (no action), cycle-4 (this report).

**Suggested fix** (substantial refactor):

Option A — Two entry points:
```
fn parse_primary_empty_ok(tok_base, sb) -> i32 {
    // Used by block-body parsers. Empty `}` returns AST_INT(0).
    let t = peek_tag(...);
    if t == 6 { mk_node(0, 0, 0, 0) }
    else { parse_primary(tok_base, sb) }
}
```
Update if/else/while/closure body parsers to call this variant.
parse_primary itself reverts to AST_ERR(6) for TK_RBRACE.

Option B — Context flag in sb:
Add scratch slot `parse_expr_empty_ok`. Set by block-body parsers
before calling parse_expr; read by parse_primary catch-all.

Option A is preferable (compile-time visible, no shared mutable state).

## Counter status

cycle-4: NOT CLEAN (1 MEDIUM). Cumulative: cycle-2 CLEAN, cycle-3
1 LOW (deferred), cycle-4 1 MEDIUM. Need to address M2 properly
before counter can increment toward 5.

This is the same finding flagged in cycles 1, 2, 3 — each time
deferred. Time to actually fix it.
