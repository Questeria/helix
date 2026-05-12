# Audit Stage 28.9 cycle 104 — Type design

## Header

- **Date**: 2026-05-12
- **HEAD**: `31e1725` ("Stage 28.9 cycle-103 audits: 3/3 CLEAN,
  counter 0/5 → 1/5")
- **Counter at start**: 1/5 (cycle-103 CLEAN advanced it from 0)
- **Scope**: cycle-104 narrow type-design audit covering four
  rotated surfaces:
  - `_is_i64_type` / `_is_u64_type` / `_is_64bit_int_type`
    predicate family in `helixc/backend/x86_64.py:1019-1042`
  - generic struct monomorphization type bookkeeping in
    `helixc/bootstrap/parser.hx` — `struct_gp_tab` table
    (lines 297-362), `gp_marker_encode` /
    `gp_marker_is` helpers (lines 292-296)
  - named struct-lit type resolution for the Stage 28.13.1
    (non-generic) and Stage 28.13.2 (generic-mono) branches in
    `helixc/bootstrap/parser.hx:3390-3497` and `3532-3612`
  - type-set soundness for `ADD` / `SUB` / `MUL` / `DIV` / `MOD`
    / `SHR` / `SHL` / cmp dispatch paths in
    `helixc/backend/x86_64.py` (lines 1318-1540, 1660-1693)
- **Bar**: PASS = ZERO new findings at confidence ≥ 75%.
  Re-flagging cycle 1-103 findings is FORBIDDEN. DIV/MOD/SHR
  signed-vs-unsigned mismatch was explicitly DEFERRED in
  cycle-101 (and re-affirmed in cycle-102/103) — NOT a new finding.
- **Mode**: read-only on `helixc/`. Only write is this document.
- **Source delta since cycle-103 audit**: NONE. HEAD `31e1725`
  added cycle-103 audit docs only. Source state == cycle-102
  fix-sweep state (`26dfa82`).

## Methodology

Five verification points covering the rotated surface. Cycle-103
already verified the post-cycle-102 ADD/SUB/MUL widening (V1-V4 in
that doc); cycle-104 extends to (a) the parser.hx generic-struct
metadata layer that cycle-101 deferred as "Stage 28.13.1
INDEPENDENT — out of scope" and (b) the Stage 28.13.1/2 named
struct-lit branches that landed pre-cycle-102 but were unaudited.

1. **V1 — `_is_64bit_int_type` post-cycle-103 stability**:
   confirm the helper and its three call sites
   (`ADD@1329`, `SUB@1359`, `MUL@1387`) are unchanged since
   cycle-103's PASS, and that the type-set
   `{i64, isize, u64, usize}` remains closed against the
   frontend primitive enumeration in `lower_ast._PRIMITIVE_TYPE_NAMES`
   and `typecheck.PRIMITIVES`.
2. **V2 — `struct_gp_tab` type bookkeeping invariants**: examine
   the (struct_idx, gp_count, gp_names_head) triple stored at
   stride 3, the 0-on-miss-or-0-count sentinel, the cap-8
   overflow path, and the writer's `gp_count > 0` guard at
   parse_struct_decl exit.
3. **V3 — `gp_marker_encode` / `gp_marker_is` invariants**: confirm
   the `200 + gp_idx` encoding stays disjoint from struct_tab
   indices (cap 8 ≪ 200), that `gp_marker_decode` was
   intentionally NOT exposed (cycle-7 type-design C1 rationale),
   and that the 7 deferred raw-200 sites at lines 4156-4280
   remain Stage-8-internal.
4. **V4 — named struct-lit type-resolution invariants**:
   the Stage 28.13.1 non-generic branch and Stage 28.13.2
   generic-mono branch both allocate `arity` arena slots with
   sentinel -1, fill via `struct_tab_field_lookup`, validate
   complete fill, then rebuild a positional TUPLE_CONS chain.
   Check: (a) sentinel value -1 vs valid AST node indices is
   unambiguous (AST node indices are arena offsets ≥ 0); (b)
   `temp_base = __arena_len()` captured BEFORE sentinel pushes
   stays valid across parse_expr / mk_node arena growth; (c)
   field lookup's stride-3 layout works identically on mono'd
   struct_tab entries per INC-3b.2 cloning.
5. **V5 — Type-set soundness across the full arith dispatch
   matrix**: for each op kind in {ADD, SUB, MUL, DIV, MOD,
   BIT_AND, BIT_OR, BIT_XOR, SHL, SHR, BIT_NOT, NEG, CMP_*,
   SELECT, BR, RETURN}, classify the 64-bit gate as (a)
   correctly using `_is_64bit_int_type` (cycle-102-fixed), (b)
   using the inline `_is_i64_type(...) or _is_u64_type(...)`
   pattern (cycle-100/Stage-16.5), (c) deferred-known
   `_is_i64_type`-only fallthrough (cycle-57 + cycle-101 V3 +
   cycle-103 V2 list), or (d) appropriately signedness-specific.
   No bucket (e) "net-new defect" should appear.

## Findings table

| ID | Severity | Confidence | Topic | Disposition |
|----|----------|------------|-------|-------------|
| —  | —        | —          | —     | No findings at conf ≥ 75 |

## Verification points (detailed)

### V1 — `_is_64bit_int_type` post-cycle-103 stability (PASS)

Source unchanged since cycle-103 audit
(`git diff 26dfa82..31e1725 -- helixc/backend/x86_64.py` is
empty — cycle-103 wrote docs only). Helper at
`x86_64.py:1033-1042` still returns
`self._is_i64_type(ty) or self._is_u64_type(ty)`, with the three
arithmetic call sites at ADD@1329, SUB@1359, MUL@1387 still
gated on it. Type-set closure against `_PRIMITIVE_TYPE_NAMES`
(`lower_ast.py:356-362`) and `PRIMITIVES` (`typecheck.py:336-343`)
re-verified — the 64-bit integer subset of `_NUMERIC_INT_PRIMS`
(`typecheck.py:2138-2141`) is exactly `{i64, isize, u64, usize}`,
matching the helper's set per `_is_i64_type` ∪ `_is_u64_type`.
No regression from cycle-103. PASS at conf ≥ 75.

### V2 — `struct_gp_tab` type bookkeeping invariants (PASS)

Region anchored at `sb+78` (base) / `sb+79` (count). Stride 3,
cap 8 (declared at `parser.hx:312`). Entry layout:
(struct_idx, gp_count, gp_names_head). Three operations
audited:

- **Writer (`struct_gp_tab_add` @310-323)**: overflow guard
  `if count >= 8 { 0 - 1 }` returns -1 on cap exceeded, never
  silently. Otherwise writes 3 slots and bumps count. The
  count-bump uses direct `__arena_set(sb + 79, ...)` rather
  than going through a setter — consistent with the existing
  table-update idiom in `struct_tab_add` (@149-...) and
  `enum_tab_add` (@1056-1071). No defect.

- **Lookup (`struct_gp_tab_lookup` @329-344)**: linear scan;
  returns gp_count on hit OR 0 on miss. The comment at lines
  325-328 explicitly documents the sentinel conflation: "0 is
  a meaningful '0 generic params' value that has the same
  effect as miss." Critically, the writer at
  `parser.hx:6820-6824` guards `struct_gp_tab_add` with
  `if gp_count_now > 0`, so the table NEVER stores a
  zero-gp_count entry. Therefore lookup's 0-return
  unambiguously means "not in the table" — i.e., non-generic
  struct. The "0 on hit with gp_count=0" branch is unreachable
  by construction. Invariant: writer guard ⇒ lookup sentinel
  disjoint from any real value. **Sound.**

- **Names-head retrieval (`struct_gp_tab_names_head` @347-362)**:
  same linear scan, returns the gp_names_head pointer on hit
  or 0 on miss. Caller at the use-site (parse_primary
  generic-mono branch) checks gp_count > 0 BEFORE retrieving
  the names head, so the 0-on-miss path is gated by a prior
  lookup. **Sound.**

The struct_gp_tab type design — keying by struct_idx, gating
add by gp_count > 0, dual-fn read (count + names_head) so
callers can decide before walking the chain — is internally
consistent. No type-design defect at conf ≥ 75. PASS.

### V3 — `gp_marker_encode` / `gp_marker_is` invariants (PASS)

The encoding `200 + gp_idx` (helper `gp_marker_encode @293`) lives
in a numeric region distinct from struct_tab indices (cap 8, range
0..7) and the field-type encoding (uses small positive ints for
TIRScalar tags and -1 for "unknown"). The cycle-5 polish commit's
inline-arithmetic guard pattern is preserved: callers compose
`gp_marker_is(v) == 1` with `v - gp_marker_base()` rather than
calling a free `gp_marker_decode`. Cycle-7 type-design C1
(MED conf 90) explicitly rejected the free decode helper because
it would be a partial function (defined only for
v ≥ gp_marker_base()) with no available trap primitive — the
inline pattern keeps the guard visible to reviewers.

Invariant — struct_tab cap (currently 8) must remain below
`gp_marker_base()` (200) — re-checked: `struct_tab_count >= 8`
guard at parser.hx:312/491/584/618/635/848/968/1059 traps
overflow at well below 200. The 192-slot gap between struct_tab's
max idx (7) and the gp marker range (200..) is substantial. No
risk of collision in current Phase-0.

The 7 deferred raw-200 sites at parser.hx:4156-4280 (Stage-8
monomorphize_pass) are intentionally NOT migrated per the
cycle-71 narrow-scope discipline documented at parser.hx:262-291.
These sites are deferred-known stylistic drift (free arithmetic
on `200 + X` instead of helper composition); they correctly
mirror the same encoding, so no type-mismatch risk. Below 75 as
a finding (stylistic, not type-design defect — the encoding is
correct at both forms). PASS at conf ≥ 75.

### V4 — Named struct-lit type-resolution invariants (PASS)

Both branches (Stage 28.13.1 non-generic @3532-3612; Stage 28.13.2
generic-mono @3390-3465) share the same algorithm with different
keys (`s_idx` + `arity` vs. `mono_s_idx` + `arity_m`).

- **Sentinel disambiguation**: temp slots initialized to `0 - 1`
  (= -1). Valid AST node indices stored later are
  `__arena_len()`-style positive offsets returned by `mk_node`.
  AST node 0 is the conventional "no-such-node" sentinel at the
  caller level but parse_expr never returns 0 for a successfully
  parsed expression (it always allocates via mk_node, which
  returns the new node's arena index ≥ 1). The
  `__arena_get(temp_base + f_idx) != 0 - 1` check correctly
  distinguishes filled vs unfilled slots. **Sound.**

- **temp_base lifetime**: `temp_base = __arena_len()` is captured
  BEFORE the sentinel pushes (lines 3403/3546). parse_expr calls
  during the loop push new arena entries (AST nodes), but the
  temp slots at absolute positions `temp_base..temp_base+arity-1`
  remain valid because arena is append-only — slots aren't moved.
  `__arena_set(temp_base + f_idx, fval)` correctly addresses the
  temp slot regardless of how much parse_expr grew the arena.
  **Sound.**

- **Field-lookup stride compatibility**: `struct_tab_field_lookup`
  @1028-1051 reads the entry's `fields_ptr` and walks pairs at
  stride 3 (name_s, name_l, ty). INC-3b.2 clones the fields region
  with the same stride-3 layout per parser.hx:3308-3334. So
  lookup on `mono_s_idx` returns the same positional indices as
  lookup on the original `s_idx`. The Stage 28.13.2 branch correctly
  reuses the helper unchanged. **Sound.**

- **Validation completeness**: after the field-fill loop, a
  second pass at lines 3442-3449 / 3588-3595 walks
  `temp_base..temp_base+arity` checking for any -1 slot. If any
  remains, trap 50040 (missing field — arity-mismatch class).
  Combined with the in-loop traps (50041 unknown field, 50042
  duplicate field), every error path leads to a trap node,
  never silent miscompile. **Sound.**

- **TUPLE_CONS chain build**: lines 3453-3461 / 3600-3608
  iterate `temp_base..temp_base+arity` in POSITIONAL order
  (not insertion order), writing each value into a new
  TUPLE_CONS (tag 51) node and linking via `tail + 2 = next`.
  This guarantees the downstream codegen receives values in
  declaration order regardless of named-mode parse order — the
  cycle-1 fix's asymmetric probe tests
  (`test_codegen.py:2658-2671`) lock this invariant.
  **Sound.**

One sub-75 observation: when an error path (`named_err != 0`)
triggers, the trap-node return at lines 3439 / 3584 does NOT
consume the closing `}` token — the cursor remains at whatever
position the in-loop break left it (typically just after the
comma or after the bad field). The surrounding parser then sees
the unconsumed `}` and cascades into more errors. This is
behaviorally lossy for error recovery (the trap id is correct
but the cursor state is asymmetric with the success path's
`cur_advance(sb); // consume } ` at 3441/3586). Below 75 because
(a) cascade-bounded — no silent miscompile, (b) error recovery
is not a Phase-0 goal, (c) the trap id correctly identifies
root cause. Mentioned for follow-up if error-recovery becomes a
deliverable.

PASS at conf ≥ 75.

### V5 — Type-set soundness across the full arith dispatch matrix (PASS)

Re-classification of every integer-dispatch site in
`helixc/backend/x86_64.py`:

**Bucket (a) — `_is_64bit_int_type` (cycle-102-fixed)**:
- ADD @ 1329 ✓
- SUB @ 1359 ✓
- MUL @ 1387 ✓

**Bucket (b) — Inline `_is_i64_type(...) or _is_u64_type(...)`
(cycle-100/Stage-16.5-fixed)**:
- cmp dispatch use_64 @ 1672-1676 ✓
- CALL-arg register-load @ 1872 ✓
- CALL-res-store @ 1891 ✓

**Bucket (c) — Deferred-known `_is_i64_type`-only
(cycle-57/cycle-101 V3/cycle-103 V2 list)**:
- CONST_INT @ 1198
- BITCAST `wide` @ 1234-1236
- CAST `from_is_i64`/`to_is_i64` @ 1253-1254 (signed-cast path
  — the u64→f64 case is a cycle-57 cast-matrix gap, distinct
  defect class but in the same deferred bucket)
- DIV @ 1418
- MOD @ 1433
- BIT_AND @ 1453
- BIT_OR @ 1468
- BIT_XOR @ 1483
- SHL @ 1498
- SHR @ 1513
- BIT_NOT @ 1527
- NEG @ 1540
- SELECT `is_i64` @ 1716
- BR `operand_ty` @ 1945
- RETURN `op.operands[0].ty` @ 1917
- FFI_CALL `_is_i64_type(arg.ty)` @ 1803 (this site does NOT
  also OR with `_is_u64_type` because u64 args go through the
  separate `_is_u64_type` branch one level up — verified at
  line 1872 — so the 32-bit fallthrough here is gated by
  exclusion, not by oversight)
- COND_BR `cond_slot` @ 1953 — the cond value is always i32-or-
  smaller (boolean), not 64-bit, so no _is_64bit_int gate needed
- function-prologue parameter spill @ 986 — i64/isize get the
  8-byte slot move; u64/usize handled separately via FFI layer.
  Latent cycle-57-deferred for the bare-IR `let p: u64 = ...`
  path; tracked in the same deferred-known sweep.

**Bucket (d) — Appropriately signedness-specific**:
- CMP_* setter table selection @ 1682 (signed/unsigned via
  `_is_unsigned_int_type` predicate; cycle-100-fixed)
- DIV/MOD `idiv` vs `div` dispatch (cycle-57 noted as
  signed-vs-unsigned mismatch — DEFERRED per cycle-101 V3 and
  reaffirmed cycle-102 commit msg)

**Bucket (e) — Net-new defect**: NONE.

No site appears in bucket (e). Every `_is_i64_type`-only site
is either appropriately signedness-specific or in the
deferred-known set explicitly enumerated by cycle-57's
"<75 Notes" section (lines 142-165) and reaffirmed by cycle-101
V3 (lines 39-49) and cycle-103 V2 (lines 119-129). Per cycle-104
scope, re-flagging these is FORBIDDEN.

The type-set soundness invariant holds for the cycle-102-fixed
trio (ADD/SUB/MUL): the helper `_is_64bit_int_type` correctly
routes the four-element set `{i64, isize, u64, usize}` to the
64-bit emit path. PASS at conf ≥ 75.

## Verdict

**PASS** — 0 new findings at confidence ≥ 75.

- V1 confirms `_is_64bit_int_type` and its three call sites are
  unchanged since cycle-103's PASS and remain type-set-closed.
- V2 confirms `struct_gp_tab`'s sentinel-conflation is sound
  because the writer guards `gp_count > 0`, making the
  0-on-zero-count branch unreachable.
- V3 confirms `gp_marker_encode` / `gp_marker_is` invariants
  (struct_tab cap 8 ≪ gp marker base 200) and the
  cycle-7-deliberate decode-helper omission.
- V4 confirms Stage 28.13.1/2 named struct-lit invariants
  (sentinel disambiguation, temp_base lifetime, stride-3 layout
  reuse, validation completeness, positional-order
  TUPLE_CONS build).
- V5 confirms the full arith dispatch matrix: every site falls
  into one of buckets (a)-(d); no net-new bucket (e) defects.

Stage 28.9 audit-gate counter advances **1 → 2**.

## Cross-reference to cycles 101-103

- **Cycle 101** (`fbfa211`,
  `docs/audit-stage28-9-cycle101-type-design.md`): PASS, 0
  findings. V3 enumerated the wider `_is_i64_type`-only
  fallthrough sites as deferred-known from cycle-57. Stage
  28.10/28.11/28.13.1 explicitly OUT OF SCOPE.
- **Cycle 102** (`26dfa82`): fix-sweep landing
  `_is_64bit_int_type` helper + ADD/SUB/MUL widening + two
  ELF-byte regression tests. Source delta locked.
- **Cycle 103** (`31e1725`,
  `docs/audit-stage28-9-cycle103-type-design.md`): PASS, 0
  findings on the cycle-102 delta. Counter 0 → 1.
- **Cycle 104** (this doc): adds parser.hx `struct_gp_tab` +
  `gp_marker_*` + Stage 28.13.1/2 named struct-lit to the
  audit surface (cycle-101 had marked 28.13.1 as
  "INDEPENDENT — out of scope"; cycles 102/103 narrowed to
  x86_64 backend deltas). All four new surfaces PASS. Counter
  1 → 2.

Deferred items unchanged:
- A.StrLit IR lowering gap (silent-failures F1)
- DIV/MOD/SHR signed-vs-unsigned (codereview F2 / cycle-101 V3
  / cycle-57 cast-matrix gap)
- Bitwise/shift/NEG sibling width-gate class (cycle-101 V3 +
  cycle-103 V2)
- 7 raw-200 Stage-8 sites at parser.hx:4156-4280 (cycle-7
  narrow-scope discipline)

No prior-cycle findings re-surface; no edits to source
performed; this document is the only file written.
