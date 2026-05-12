# Stage 30 cycle-1 audit findings

**Date**: 2026-05-12
**HEAD**: `ca8c9ce` (Stage 29 FULLY COMPLETE)
**Audit scope**: Stage 29 self-host changes (parser.hx return-removal + kovc.hx cap bumps + parse_primary catch-all + test_codegen.py)

## Status: NOT CLEAN — 2 HIGH + 3 MEDIUM findings

3 parallel audits (silent-failure, type-design, code-review) all flagged
the same HIGH issue. Convergent finding strengthens confidence.

---

## HIGH findings

### H1: `early_err` sentinel set but never returned (parser.hx:3295-3302)

**Confidence**: 95 (3 audits agreed)

**Description**: Stage 29 rewrote 3 `return mk_node(99, NNN, 0, 0);` in
parse_primary's nt==16 branch as sentinel assignments:
```
let mut early_err: i32 = 0 - 1;
if ta_bad_token == 1 { early_err = mk_node(99, 62033, 0, 0); };
if early_err == (0 - 1) { if post_loop_t != 17 { ... }; };
if early_err == (0 - 1) { if ta_count != gp_count_pre { ... }; };
```

But `early_err` is never read or returned afterwards. Execution falls
through to the mangle/lookup/struct-lit body code regardless.

**Why it matters**: Regresses the Stage 28.11 INC-3b cycle-3 fix that
introduced trap-ids 62032 (arity mismatch) and 62033 (bad token). For
malformed `Pt<>` (zero args), `Pt<i32,i32>` (extra args), or
`Pt<+>` (bad token), the parser now silently proceeds with corrupt
state instead of trapping. Also leaks 4 arena slots per orphan
mk_node(99,...) call.

The author's own commit message (8e325cb) flagged this:
> NOTE: the sentinel value isn't checked at the end yet... this is
> benign for well-formed input but should be wired up for the trap path
> in a follow-on cycle.

**Suggested fix**: Wrap the post-sentinel mangle/struct-lit body in
an outer if/else:
```
if early_err != (0 - 1) {
    early_err
} else {
    // existing 3303-3506 mangle/struct-lit body
}
```

### H2: bind_state cap (512) > prologue stack reservation (128 slots)

**Confidence**: 85-90 (2 audits flagged)

**File**: `kovc.hx:983-993, 739-745, 1043-1057`

**Description**: Stage 29.1 bumped bind_state cap to 512 entries
(comment: "parse_primary has ~200 bindings/fn at peak"). But
emit_prologue still reserves 1024 stack bytes = 128 simultaneously-live
slots. bind_alloc_offset traps with 10030 when offset >= 1024.

The 4:1 ratio assumes LIFO bind_pop recycles slots. For programs with
many simultaneously-live bindings (no scope nesting), 129+ slots will
hit the codegen-time trap.

**Suggested fix**: Either (a) bump emit_prologue to 4096 bytes (matches
512 cap × 8 = 4096), OR (b) reduce bind_state cap back to 128 entries
to coordinate with prologue, OR (c) update comments to clarify the
512-entry cap is cumulative push count (not simultaneous offsets) and
document that bind_alloc_offset's 1024-byte ceiling is the actual
constraint.

Note: concurrent Stage 28.9 cycle-109/110 agent has been working on
this exact issue (in-flight changes to kovc.hx). May resolve
independently.

---

## MEDIUM findings

### M1: Test doesn't verify self-host fixed-point (K2 byte-identical to K1)

**Confidence**: 88

**File**: `test_codegen.py:4767-4927`

**Description**: Test docstring claims "closes the bootstrap loop:
kovc.hx compiles itself... including itself again". But the actual
test only verifies K3 = 42 for input `fn main() -> i32 { 6 * 7 }`
(11 chars). It does NOT verify:
- K2 byte-identical to K1
- K2 can compile the bootstrap source itself
- Larger inputs that exercise closures, generics, patterns, etc.

**Why it matters**: True self-host is fixed-point of compilation. The
current test passes if K2 is grossly miscompiled in any way that
happens not to affect `6 * 7`. Stage 30's "5 clean audits" has weak
ground truth without this.

**Suggested fix**: Add K2 → bootstrap-source → K2' assertion (cmp -s
should pass).

### M2: TK_RBRACE catch-all over-broad (parser.hx:3790-3794)

**Confidence**: 78-88

**Description**: Stage 29.2's TK_RBRACE → AST_INT(0) catch-all fires on
ANY primary-position `}`, not only empty blocks. Truncated sources like
`fn foo() -> i32 { let x = }` now silently compile to `let x = 0` instead
of trapping. Diagnostic quality regression for malformed inputs.

**Suggested fix**: Track empty-body context via scratch slot; only
emit AST_INT(0) in that context. Other TK_RBRACE catches should still
trap. (Lower priority — pre-existing behavior was also poor.)

### M3: Stale comments in test (test_codegen.py:4898-4912)

**Confidence**: 80

**Description**: Two adjacent comments contradict each other on K2's
exit behavior. The Stage 29.1 comment claims "K2 may SIGILL"; the
Stage 29 final comment says "K2 now exits cleanly".

**Suggested fix**: Collapse comments to reflect the final state only.

---

## Next steps

1. Apply H1 fix (wrap early_err sentinel) — mechanical
2. Investigate H2 (bind cap vs prologue) — may resolve via concurrent
   cycle-110 work
3. Apply M3 comment cleanup
4. Optionally: add K1≡K2 byte-identical check (M1) for stronger
   ground truth in Stage 30 cycle-2+

After fix-sweep + heavy gate green, dispatch Stage 30 cycle-2 audits.

Counter status: 0/5 clean cycles. NEED TO FIX H1+H2 BEFORE counter
increment.
