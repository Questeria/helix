# Audit Stage 28.9 cycle 104 — Silent failures

**Date:** 2026-05-12
**HEAD:** `31e1725` (`Stage 28.9 cycle-103 audits: 3/3 CLEAN, counter 0/5 → 1/5`)
**Counter at start:** 1/5 (after cycle-103 advanced from cycle-102 fix-sweep).
**Bar:** ZERO new findings at confidence ≥ 75. CRITICAL or HIGH only — silent corruption, sign confusion, off-by-one in pointer arithmetic, missing nullability checks, swallowed errors.

**Scope (per prompt):**
- Backend u64 arithmetic promotion: `helixc/backend/x86_64.py` `_is_64bit_int_type` helper at line 1033 and ADD/SUB/MUL dispatch at 1329/1359/1387.
- Bootstrap parser generic structs + named struct-lit: `helixc/bootstrap/parser.hx`
  - Stage 28.13.1 (commit `30c4bc0`): `peek_named_struct_lit` helper at line 236, named-mode branch in `parse_primary` (`nt == 5` path) at lines 3532-3654.
  - Stage 28.13.2 (commit `4b938d2`): named-mode branch in `parse_primary` (`nt == 16` generic-mono path) at lines 3390-3465.
  - Stage 28.13.1 cycle-2 fix (commit `fbfa211`): asymmetric test probes.
  - Stage 28.11 INC-3b prior surface (commits `e33463a`, `7123f09`, `1ff41ff`, `549a68e`, `2e7f836`) covered by cycle-5 and cycle-12 silent-failure audits — re-flagging FORBIDDEN.
- Cross-stage interactions: backend u64 path × frontend generic-mono struct construction.

**Prompt-scope note:** the prompt names `helixc/frontend/parser_*.py` for parser work, but the file structure is `helixc/frontend/parser.py` (single file) and the active recent struct/named-lit work has been in the bootstrap `helixc/bootstrap/parser.hx`. Scope adjusted accordingly — bootstrap is where Stage 28.11/28.13 commits landed.

**Mode:** STRICT READ-ONLY. No source edits. Single Write of this audit doc. Source files only read/grepped. No scorecard run.

---

## Methodology

1. **Read cycle-103 silent-failures and codereview docs** to identify the deferred class (cycle-101 codereview F2: DIV/MOD/SHR signed-vs-unsigned; sibling BIT_AND/BIT_OR/BIT_XOR/SHL/BIT_NOT/NEG still `_is_i64_type`-only). Re-flagging FORBIDDEN per prompt.
2. **Read cycle-5 + cycle-12 silent-failures audits** (Stage 28.11 INC-3b zones A-D, INC-3a helpers/reader/writer) to identify OBS-1 through OBS-7 and the cycle-5 scenario matrix. These cover all generic-struct paths and gp_marker encoding. Re-flagging FORBIDDEN.
3. **Read the cycle-102 backend delta in isolation** (`_is_64bit_int_type` helper + ADD/SUB/MUL switch) — already covered by cycle-103 to CLEAN verdict, re-audited at the union-with-frontend-deltas surface.
4. **Read the Stage 28.13.1 and 28.13.2 named-mode branches** (`git show 30c4bc0 4b938d2 fbfa211 -- helixc/bootstrap/parser.hx`) and the live `parse_primary` body at lines 3390-3654.
5. **Read the `peek_named_struct_lit` helper** at line 236 and verified EOF-sentinel-safety on the `c+1` peek.
6. **Cross-stage trace**: backend u64 arithmetic emit path × frontend generic-struct construction lowering to confirm no shared state.

Concretely cross-referenced:
- `helixc/backend/x86_64.py:1019-1056` (predicate family).
- `helixc/backend/x86_64.py:1315-1403` (ADD/SUB/MUL emit sites).
- `helixc/bootstrap/parser.hx:236-245` (peek helper).
- `helixc/bootstrap/parser.hx:3380-3497` (generic named-mode + positional generic-mono).
- `helixc/bootstrap/parser.hx:3505-3658` (non-generic struct-lit including named-mode).
- `helixc/bootstrap/lexer.hx:672` (TK_EOF sentinel emission).
- `helixc/tests/test_codegen.py:2637-2660, 2843-2860` (asymmetric named-mode probes).

---

## Defect-class scenarios traced

### Backend cycle-102 surface (re-verified against cycle-103 CLEAN)

| Scenario | Path | Loud or silent? |
|----------|------|-----------------|
| u64 ADD result captured | `_is_64bit_int_type` true → 64-bit path; REX.W ADD | Correct |
| usize MUL result captured | union predicate true (isize/usize alias of i64/u64) | Correct |
| u64 × u64 overflow high-half | `imul` low-half identical to `mul` low-half under 2's complement | Correct (cycle-103 ruled) |
| Tensor/Tile type into ADD | both predicates guard `isinstance(ty, tir.TIRScalar)`; union returns False | Correct (cycle-103 ruled) |
| DIV/MOD/SHR signed-vs-unsigned | `_is_i64_type`-only gate, signed `idiv`/`sar` unconditional | DEFERRED (cycle-101 codereview F2); no re-flag |
| BIT_AND/BIT_OR/BIT_XOR/SHL/BIT_NOT/NEG 64-bit width gate | `_is_i64_type`-only at lines 1453/1468/1483/1498/1527/1540 | DEFERRED class (cycle-103 codereview classifies as deferred sibling); no re-flag |

### Frontend Stage 28.13.1 / 28.13.2 named-mode surface

| Scenario | Path | Loud or silent? | Trap |
|----------|------|-----------------|------|
| `Pt { x: 10, y: 32 }` (forward) | peek IDENT+COLON → named-mode; struct_tab_field_lookup maps → temp[0]=10, temp[1]=32 | Correct | n/a |
| `Pt { y: 32, x: 10 }.x` (reverse) | lookup("y")=1 → temp[1]=32; lookup("x")=0 → temp[0]=10 | Correct | n/a |
| `Pt { x: 1, x: 2 }` (dup) | second `x` lookup f_idx=0; temp[0] already 1 (≠ -1) → `named_err = 50042` | Loud | 50042 |
| `Pt { foo: 1 }` (unknown field) | struct_tab_field_lookup returns -1 → `named_err = 50041` | Loud | 50041 |
| `Pt { x: 10 }` (missing field, arity-2 Pt) | post-loop validation scan finds temp[1] == -1 → 50040 | Loud | 50040 |
| `Pt {}` (empty body, arity > 0) | empty fast-path at pt_first==6 traps 50040 BEFORE reaching named-mode | Loud | 50040 |
| `Pt { x: 10, }` (trailing comma) | after pair, ct=13 → consume, nt2=6 (`}`) → keep_n=0, no extra parse | Correct | n/a |
| `Pt<i32> { x: 10, y: 32 }` (generic named) | nt==16 branch: mono'd struct entry built; peek_named_struct_lit gates to named-mode keyed by `mono_s_idx`/`arity_m` | Correct | n/a |
| Generic named dup field | symmetric with non-generic: temp[f_idx] != -1 → 50042 | Loud | 50042 |
| Generic named missing field | symmetric validation scan → 50040 | Loud | 50040 |
| `Pt { x: 10, 32 }` (mixed positional/named) | iter 2: cur is INT (tag 1), not IDENT; `tok_p2/p3` of INT give digit-lexeme; `struct_tab_field_lookup` byte_eq mismatch (no field starts with digit) → f_idx=-1 → 50041 | Loud | 50041 (mis-attributed but loud — see OBS-A below) |
| `peek_named_struct_lit` at EOF after `{` | If `{` is last non-EOF token, cur=EOF (tag 0), t1!=2 → returns 0; falls through to positional which then errors on missing value | Correct | n/a |
| `peek_named_struct_lit` `c+1` past end | TK_EOF sentinel at lexer.hx:672 always appended; reading past EOF reads next arena slot (raw memory). c reaches EOF only when t1=0 returns early — c+1 never read in that case | Correct (early-return guard) | n/a |
| Named-mode parse_expr returns trap node | trap propagates to temp[f_idx]; not -1; validation scan passes; TUPLE_CONS chain built with trap inside → downstream traps loudly at codegen | Loud (delayed) | inherited trap id |

### Cross-stage: backend × frontend

| Scenario | Path | Loud or silent? |
|----------|------|-----------------|
| `Pair<u64> { a: 100, b: 200 }` construction | parser produces TUPLE_CONS(100, TUPLE_CONS(200, 0)) with u64 inferred at typecheck; backend stores via slot-mov (32-bit-mov-pair, NOT in cycle-102 delta) | OOS — not part of cycle-102 delta; backend tuple-field u64 store is a separate surface not modified by cycle-102 |
| u64 ADD on a struct-field-loaded operand | LOAD_FIELD → slot, then OpKind.ADD with u64 result → `_is_64bit_int_type` true → REX.W ADD | Correct |
| `_is_64bit_int_type` called on a generic-mono'd struct's field type | field type at typecheck is concrete (i64/u64/isize/usize) post-monomorphization; predicate sees a `TIRScalar`, dispatches correctly | Correct |

---

## Findings table

| Severity | Count |
|---|---|
| CRITICAL | 0 |
| HIGH (conf ≥ 75) | 0 |
| MEDIUM (conf 60-74) | 0 surfaced (see OBS-A, OBS-B below — both below 75 bar and recorded for transparency only) |
| LOW (<60) | 0 surfaced |

---

## Sub-threshold observations (NOT findings — for transparency only, below 75 bar)

### OBS-A: Named-mode loop has no per-iteration IDENT-followed-by-COLON check (conf ~50)

**Location:** parser.hx:3411-3437 (generic), 3555-3582 (non-generic).

**Pattern:** `peek_named_struct_lit` runs ONCE at entry to the named-mode branch. Inside the per-pair loop, each iteration unconditionally reads `tok_p2(tok_base, cur)` / `tok_p3(tok_base, cur)` as if `cur` were an IDENT, then unconditionally consumes two tokens (field-name + `:`). If the user writes `Pt { x: 10, 99 }` (positional-named mix), iteration 2 reads INT-token bytes as the field-name lexeme, consumes the INT and the next token (which should have been `:`), then calls `parse_expr` from after that. Eventually `struct_tab_field_lookup` mismatches (no field name starts with a digit) and traps 50041.

**Why OBS not finding:** The trap IS surfaced (50041 "unknown field"). The mis-attribution is cosmetic — the user sees "unknown field" when the actual diagnostic should be "expected field-name IDENT". No silent miscompile, no OOB (`byte_eq` returns 0 on length mismatch with all current fields, none of which have digit-prefixed names — Helix grammar disallows). Cursor desync produces noisy cascading errors downstream, not silent acceptance. Below the 75 bar for HIGH-class silent-failure. Same defect class would be a polish/diagnostic-precedence improvement, not a silent-failure fix.

### OBS-B: Named-mode error path leaves `}` unconsumed (conf ~45)

**Location:** parser.hx:3438-3439 (generic), 3583-3584 (non-generic).

**Pattern:** When `named_err != 0` (50041 or 50042), the loop exits via `keep_n = 0` and the branch returns `mk_node(99, named_err, 0, 0)` WITHOUT consuming the closing `}` or any remaining field tokens. The non-error path consumes `}` at lines 3441 / 3586. So an error-path return leaves the surrounding parser cursor on a stale `}` or mid-field-list state.

**Why OBS not finding:** The originating trap (50041/50042) IS surfaced as an AST_ERR node propagating to codegen. The stale cursor produces cascading downstream parse errors (more AST_ERR nodes), not silent acceptance. No miscompile path. Cosmetic recoverability concern only, below 75 bar.

---

## Re-flagging-guard check

Findings considered and explicitly NOT re-flagged because they are documented elsewhere:

- **Cycle-101 codereview F2 (DIV/MOD/SHR signed-vs-unsigned, deferred).** Per cycle-102 commit body and cycle-103 codereview classification. Not part of cycle-102 delta.
- **Sibling BIT_AND/BIT_OR/BIT_XOR/SHL/BIT_NOT/NEG `_is_i64_type`-only 64-bit gates** (parser.hx:1453, 1468, 1483, 1498, 1527, 1540). Cycle-103 codereview "Sub-threshold observations" classifies these as part of the cycle-101 codereview F2 deferred class. Re-flagging FORBIDDEN.
- **Cycle-101 silent-failures F1 (A.StrLit IR lowering gap, deferred).** Per cycle-102 commit body. Not part of cycle-102 delta.
- **OBS-1 through OBS-7 of Stage 28.11 INC-3b cycle-5 silent-failures audit** (struct_tab/struct_gp_tab cap, lookup misses, recursive type-args, etc.). Out of cycle-104 fresh-surface scope and already documented.
- **Stage 28.11 INC-3a cycle-12 silent-failures audit zones** (gp_marker encoding, writer/reader symmetry). Already CLEAN.
- **Sub-threshold cycle-103 silent-failures observation** (SUB/MUL u64 regression-test coverage gap at conf ~55). Same evidence base; no new evidence to clear the 75 bar.

---

## Verdict

**CLEAN** — 0 findings at confidence ≥ 75 within cycle-104 scope.

Counter advances: 1/5 → 2/5.

---

## Cross-references

- Cycle-103 silent-failures audit: `docs/audit-stage28-9-cycle103-silent-failures.md` (CLEAN; cycle-102 delta verified).
- Cycle-103 codereview audit: `docs/audit-stage28-9-cycle103-codereview.md` (PASS; deferred class enumerated).
- Cycle-102 commit: `26dfa82` (closed cycle-101 silent-failures F2 + codereview F1/F2-CMP; deferred 2).
- Stage 28.11 INC-3b cycle-5 silent-failures: `docs/audit-stage28-11-cycle5-silent-failures.md` (CLEAN with 7 OBS; covers generic-mono use-site).
- Stage 28.11 INC-3a cycle-12 silent-failures: `docs/audit-stage28-11-cycle12-silent-failures.md` (CLEAN; gp_marker encoding).
- Stage 28.13.1 commit: `30c4bc0` (named struct-lit, non-generic).
- Stage 28.13.1 cycle-2 fix: `fbfa211` (asymmetric probes).
- Stage 28.13.2 commit: `4b938d2` (named struct-lit, generic-mono).
- TK_EOF sentinel guarantee: `helixc/bootstrap/lexer.hx:672`.
