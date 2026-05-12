VERDICT: CLEAN

# Stage 28.11 INC-3a CYCLE-12 code-review audit

**Audit surface**: `helixc/bootstrap/parser.hx` zones at ~line 222
(`gp_marker_base / gp_marker_encode / gp_marker_is` helpers),
~line 1705 (reader-side guard in parse_unary chained-field path),
~line 6257 (writer-side `parse_struct_decl` field-type encoding).
**Commit range**: `2e7f836..c95b13f` (4 commits, 199 diff lines on parser.hx).
**Methodology**:
1. `git diff 2e7f836^..HEAD -- helixc/bootstrap/parser.hx` (full inspection).
2. Invariant verification: re-read cap proofs for `struct_tab` (parser.hx:880)
   and `gp_tab` (parser.hx:193-205); confirmed `200 + max_gp_idx == 203 <<
   struct_tab cap 8`.
3. Writer/reader symmetry: traced encode site (parser.hx:6286) and decode
   guard site (parser.hx:1729), confirmed both source the literal 200 from
   `gp_marker_base()` — single point of truth.
4. Style/naming consistency: grepped peer table-helper conventions
   (`gp_tab_*`, `struct_tab_*`, `mr_tab_*`); helpers conform.
5. Test coverage: one cross-zone probe at test_codegen.py:2729 exercises
   both encoder and decoder via `struct Pt<T> { x: T, y: T }`; field-access
   exit value 42 verifies scalar-shape semantics.
6. Regression suite: `python -m pytest helixc/tests/ -q` → **1550 passed,
   1 skipped, 0 failed in 771.80s**. Clean.

**Confidence threshold**: HIGH/MED ≥ 75; below 75 → LOW / OBS / OOS.

## Summary

Zero findings at HIGH or MED severity. The three zones under review form
a tight, internally-symmetric increment. The helper extraction at zone 1
correctly centralizes the 200-boundary literal that the cycle-5 sweep was
chartered to eliminate from the active INC-3a sites. The cycle-7 decision
to omit `gp_marker_decode` (and to require call-site composition with
`gp_marker_is`) is well-justified given the bootstrap's lack of a trap
primitive and the SF-1/SF-2/SF-3 silent-failure precedent.

The writer encodes via `gp_marker_encode(gp_idx)` whenever
`gp_tab_lookup(sb, t_s, t_l) >= 0`; the reader decodes via
`gp_marker_is(f_struct_idx) == 1` and treats the 200+ slot as scalar
(p3 == 0, struct chain reset). The encode/decode pair share
`gp_marker_base()` as the single source of the boundary literal — a
future bump (INC-3b or later cap raise) needs only the one-line change.

## Invariant verification (executable code)

1. **Writer/reader use the same boundary primitive.**
   Writer (parser.hx:6286) → `gp_marker_encode(gp_idx)` →
   `gp_marker_base() + gp_idx`.
   Reader (parser.hx:1729) → `gp_marker_is(f_struct_idx) == 0` →
   `if f_struct_idx >= gp_marker_base() { 1 } else { 0 }`.
   Both call paths route through `gp_marker_base()`. Symmetric by
   construction; cannot drift.

2. **Cap-vs-boundary headroom.** `struct_tab_add` (parser.hx:880) caps
   `count >= 8` returning `-1`; valid `struct_idx` ∈ [0, 7].
   `gp_tab_add` (parser.hx:195) caps `count >= 4`; valid `gp_idx` ∈ [0, 3].
   Therefore `gp_marker_encode` returns values in [200, 203], leaving a
   192-slot gap above struct_tab's maximum. Holds with substantial
   headroom; documented in the cycle-5 invariant block as requiring
   re-verification at any future cap bump.

3. **Encode precondition.** `gp_marker_encode(gp_idx)` is unconditionally
   called only inside the `if gp_idx >= 0` branch at parser.hx:6285. So
   the function only fires on a successful `gp_tab_lookup` hit (gp_idx ∈
   [0, 3]), never on a miss (-1). No risk of encoding -1 → 199 which
   would alias to a near-boundary scalar.

4. **Reader-side decode is guarded.** The cycle-7 doc-block enforces that
   `v - gp_marker_base()` MUST be call-site-composed under
   `if gp_marker_is(v) == 1 { ... }`. The current INC-3a reader at
   parser.hx:1729 does not need to decode (it only needs to detect the
   marker and switch to scalar shape) — so the discipline is not tested
   yet. INC-3b's use-site reader will be the first composition test.

5. **Existing scalar-field behavior preserved.** When `gp_tab_lookup`
   misses (non-generic field), writer falls through to
   `struct_tab_lookup_idx` exactly as in pre-INC-3a code. Non-generic
   structs are codegen-identical to pre-INC-3a; verified by the
   pre-existing `struct Pt { x: i32, y: i32 }` probes
   (test_codegen.py:2654-2670) passing in the regression run.

6. **Reader's `else` branch on the new `gp_marker_is` split mirrors the
   pre-existing `f_struct_idx < 0` scalar branch** at parser.hx:1746-1747:
   both emit `mk_node(52, prim, f_idx, 0)` and reset
   `cur_struct_idx = 0 - 1`. The 200+ generic-typed branch is therefore
   provably equivalent to a scalar-field branch from codegen's
   perspective — exactly what the INC-3a contract demands ("treated as
   scalar pending INC-3b monomorphization").

## Style / naming consistency

- Helper function names follow the project's `gp_tab_*` / `struct_tab_*`
  / `mr_tab_*` patterns: snake_case, single-purpose, returning `i32`.
  `gp_marker_*` reads as a natural sibling to `gp_tab_*`.
- Function bodies match the bootstrap's "expression-as-body" style for
  small accessors (cf. `gp_tab_base`, `mr_tab_base`, etc.). No `let mut`
  scaffolding where unnecessary.
- The `gp_marker_is` predicate returns 0/1 i32 (boolean-as-i32),
  consistent with `byte_eq`, `var_struct_tab_lookup`'s `>= 0` callers,
  and other peer predicates.

## Error-handling parity

- The writer at parser.hx:6285 does not check for `struct_tab_lookup_idx`
  miss either (it accepts the `-1` and writes it into the field's
  `f_struct_idx` slot — established pre-INC-3a behavior). The new code
  preserves this convention.
- `gp_tab_lookup` failure → -1 → falls through to `struct_tab_lookup_idx`
  unchanged. Failure paths are no worse than pre-INC-3a.

## Test coverage

- Cross-zone end-to-end probe at test_codegen.py:2729:
  `struct Pt<T> { x: T, y: T } fn main() -> i32 { let p = Pt { 10, 32 };
  p.x + p.y } == 42`. This exercises the writer (encodes both `x` and `y`
  field types as 200+0) AND the reader (decodes via `gp_marker_is` at
  field-access time) AND the codegen scalar-shape contract (4-byte
  i32 reads at p3 == 0).
- Non-generic struct probes at test_codegen.py:2654-2670 (Stage 5 Iter D
  nested-struct field access) lock the negative path (`gp_tab_lookup`
  miss → fall-through preserves pre-3a semantics).
- Full pytest suite: 1550 passed, 1 skipped, 0 failed (771.80s).
  Includes all stage-5 struct, stage-8 generic-fn, stage-28.11 INC-1/2/3a
  test surfaces.

## Findings

None at HIGH (90-100) or MED (80-89) confidence.

## OBS — sub-threshold observations (executable code)

The following observations did not meet the 75-confidence bar and are
recorded for transparency without action:

- **OBS-1 (conf 35)**: writer-side `gp_tab_lookup(sb, t_s, t_l)` runs
  unconditionally on every field declaration, even for non-generic
  structs where `gp_tab_count == 0`. The lookup short-circuits at line
  211 via `while i < count` so the cost is one __arena_get + one
  comparison per non-generic field — negligible and not a defect.
  Mirrors `parse_fn_decl`'s existing pattern at parser.hx:5458-5459 so
  consistency wins over micro-optimization.

- **OBS-2 (conf 45)**: the reader-side new branch at parser.hx:1737-1744
  duplicates the body of the pre-existing `f_struct_idx < 0` branch at
  parser.hx:1746-1747 (both emit `mk_node(52, prim, f_idx, 0)` and reset
  `cur_struct_idx = 0 - 1`). The duplication is intentional and
  documented (the new branch carries a distinct doc-comment explaining
  the 200+ marker case) — and a refactor to merge the two scalar
  branches into a single early-merge would force the comment ordering
  and lose clarity. Not flagged.

## OUT OF SCOPE — doc-class observation (informational only)

- The reader-side comment at parser.hx:1717-1721 describes the
  pre-INC-3a-fix bug as if it could have triggered, but the writer never
  wrote 200+ for struct fields pre-INC-3a (the diff shows the old writer
  unconditionally called `struct_tab_lookup_idx`). The bug class
  described is hypothetical-counterfactual ("had INC-2 landed the writer
  but not the reader"). This is a documentation-clarity nit, not a
  defect; out of scope per strict cycle-12 doc-exclusion rule.

## Cycle counter

- Cycle 10: CLEAN (90s, type-design).
- Cycle 11: CLEAN (75s, silent-failures).
- Cycle 12 code-review: CLEAN (this report).
- Counter advances 2/5 → 3/5. Need 2 more clean cycles to close
  INC-3a's audit gate.
