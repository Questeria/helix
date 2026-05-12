# Audit Stage 28.9 cycle 101 — Type design

## Scope

HEAD `fbfa211`. Stage 28.9 narrow type-design audit of cycle-100
fix-sweep (`caf203f` — unsigned int cmp). Rotation: parser tuple/
array literals + fdce function-graph reachability. Parallel Stage
28.10/28.11/28.13.1 INDEPENDENT — out of scope. Deferred-known
(cycle-57 deferred u64/usize backend-arith dispatch gap; later
cycles' deferred items) NOT re-flagged.

Read-only audit. No edits to source. ONE write — this document.

## Verdict

**PASS** — 0 findings at conf ≥ 75%.

## Verification points

### V1 — `_is_unsigned_int_type` covers the right CMP set (PASS)

`helixc/backend/x86_64.py:1033-1045` matches `TIRScalar.name in
{u8, u16, u32, u64, usize}`. The cmp dispatch at lines 1633-1660
selects this set when either operand is unsigned-int. Not a
strict "inverse" of `_is_i64_type` — the cmp dispatch treats
signed-int as the default (else) branch, so the unsigned
predicate only needs positive coverage of the u-family. `char`
(potentially u32-shaped) routes through the signed default; for
all valid Unicode scalar values (≤ 0x10FFFF) signed and unsigned
cmp agree, so no observable miscompile — below 75% as a finding.

### V2 — `unsigned_int_cmp_setters` covers 6 CMP opcodes (PASS)

Lines 1583-1590: EQ→sete, NE→setne, LT→setb, LE→setbe, GT→seta,
GE→setae. Six entries, mapping all six `OpKind.CMP_*` members
declared in `helixc/ir/tir.py`. Matches the keys of
`int_cmp_setters` and `float_cmp_setters` 1:1.

### V3 — Other backend codegen sites needing u64/usize dispatch

The cycle-100 fix only patched the CMP dispatch. ADD/SUB/MUL/DIV/
MOD/BIT_AND/BIT_OR/BIT_XOR/SHL/SHR/BIT_NOT/NEG (lines 1318, 1343,
1368, 1393, 1408, 1428, 1443, 1458, 1473, 1488, 1502, 1515) still
gate the 64-bit emit on `_is_i64_type(op.results[0].ty)` alone,
so u64/usize results fall through to the 32-bit truncating else.
This defect class was already flagged in
`docs/audit-stage28-9-cycle57-type-design.md` as deferred (the
cycle-57 recommendation was a `_is_64bit_int_type` predicate
sweep) — **deferred-known, NOT re-flagged** per scope.

### V4 — `helixc/frontend/parser.py` tuple/array literal parsing (PASS)

Tuple literal at lines 1173-1200: empty `()` → `TupleLit(elems=
[])`; single-expr `(e)` → unwraps to `e`; comma-separated → loop
with explicit `i == last_i` progress guard that raises
`ParseError`. Array literal at lines 1202-1212: comma-loop
exits cleanly on either `RBRACK` or `_match(T.COMMA)` failure;
every body iteration calls `_parse_expr()` which advances or
raises. No infinite-loop or silent-empty hazard at conf ≥ 75%.

### V5 — `helixc/ir/passes/fdce.py` function-graph reachability (PASS)

Worklist fixed-point at lines 67-84. Roots: `entry_fn` + every
fn with `attrs.is_pub` or `attrs.kernel`. Edges: CALL.target
attr, MODIFY.verifier_fn attr, QUOTE.ast_pretty identifier scan
(intersected with `all_fn_names`). The `if n in live: continue`
guard absorbs duplicate root additions and worklist re-entries.
`if entry_fn not in module.functions: return 0` short-circuits
the empty-module-by-accident case (guard documented at module
docstring). No reachability gap at conf ≥ 75%.

## Conclusion

PASS — 0 findings at conf ≥ 75%. Cycle-100 cmp fix is locally
correct and complete for cmp. The arith-dispatch parallel gap is
deferred-known from cycle-57 and out of scope per audit rules.
Stage 28.9 counter advances 0 → 1.

No edits made to source. This document is the only file written.
