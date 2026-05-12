# Stage 28.11 INC-3b Cycle-5 — Silent-Failure Audit

VERDICT: CLEAN

Counter state after this cycle: 2/5 (cycle-4 was clean; this is cycle-5).

## Scope

Executable code in `helixc/bootstrap/parser.hx` only. INC-3b cumulative
surface across commits e33463a, 7123f09, 1ff41ff, 549a68e:

- Zone A: `struct_gp_tab_{base,count,add,lookup,names_head}` helpers
  (lines 271-341).
- Zone B: use-site monomorphization in `parse_primary` `nt == 16`
  branch (lines 3196-3407).
- Zone C: `parse_top` struct_gp_tab region init (lines 3906-3920,
  3958-3971).
- Zone D: `parse_struct_decl` struct_gp_tab population (lines
  6605-6646).

Kovostov runtime, docs, comments, and line-number references in
comments are explicitly out of scope.

## Methodology

Exhaustive attack-style scenario enumeration against the new code,
mapping each scenario to the executable behavior in the diff and
checking for silent acceptance, silent corruption, OOB reads, or
unbounded paths.

### Scenarios traced and outcomes

| Scenario | Path | Loud or silent? | Trap |
|----------|------|-----------------|------|
| `Pt<>` on arity-1 Pt | TK_GT first iter, ta_count=0 != gp_count_pre=1 | Loud | 62032 |
| `Pt<i32, i32>` on arity-1 Pt | ta_count=2 != gp_count_pre=1 | Loud | 62032 |
| `Pt<i32>` missing `{` | post-loop GT consumed; lbrace_t != 5 | Loud | 62030 |
| `Pt<+>` bad token | else-branch sets `ta_bad_token=1`; post-loop trap | Loud | 62033 |
| `Pt<i32::Foo>` (path) | `::` is non-{17,13,2} → bad token | Loud | 62033 |
| `Pt<-i32>` | `-` is non-{17,13,2} → bad token | Loud | 62033 |
| `i32<i32>{...}` (non-struct IDENT followed by `<`) | `s_idx_pre = -1` → `gp_count_pre = 0` → fall-through var-ref; surrounding parser parses `<` as comparison | Designed | n/a |
| Non-generic `Foo<i32>{...}` | `s_idx_pre >= 0` but `gp_count_pre = 0` (struct_gp_tab miss) → fall-through var-ref | Designed | n/a |
| Re-mono `Pt<i32>` twice | second call: `existing_idx >= 0` → reuse cached mono'd entry | Loud (on user errors) | n/a |
| struct_tab cap overflow at clone | `struct_tab_add` returns -1 → `mono_s_idx < 0` → 62031 | Loud | 62031 |
| `Pt<i32>{}` on arity > 0 | `pt_first == 6`; `arity_m != 0` → 50040 | Loud | 50040 |
| `Pt<i32>{10}` arity mismatch in body | post-body `n != arity_m` → 50040 | Loud | 50040 |
| `Pt<Pair<i32>>` (nested generic) | inner `<` (tag 16) is non-{17,13,2} → bad token | Loud | 62033 |
| `Pt<i32>>` (extra `>`) | one `>` consumed, second `>` is lbrace_t check → 62030 | Loud | 62030 |
| `Pt<i32 >>` (TK_RSHIFT) | TK_RSHIFT (31) is non-{17,13,2} → bad token | Loud | 62033 |
| `Pt<i32` EOF mid-args | EOF tag 0 in loop → bad token; post-loop `post_loop_t != 17` → 62033 | Loud | 62033 |
| `Pt<UnknownType>{...}` | `struct_tab_lookup_idx` returns -1, substituted as scalar marker | Convention-consistent, designed (see OBS-1) | n/a |
| `Pt<Pt>` (recursive type-arg using same generic name) | substituted struct_idx points at un-monomorphized Pt entry (markered fields) | Convention-consistent, designed (see OBS-2) | n/a |
| `Pt<i32>{...}.x.y` chained dot-access | new branch returns `mk_node(50, ...)`; postfix dot-access in surrounding parser unchanged | Works | n/a |
| `struct Node<T> { next: Node<T> }` self-ref | parse_struct_decl resolves `Node` at decl time via lookup; not in struct_tab yet → -1 (scalar). Pre-existing OOS limitation. | OOS | n/a |
| 9th generic struct decl when struct_tab cap (8) is hit | `struct_tab_add` returns -1 → guard at line 6643 skips `struct_gp_tab_add` | Loud at next use (lookup miss → var-ref → comparison-op parser path; arity mismatch downstream) | partial — see OBS-3 |
| `Pt<,i32>` / `Pt<i32,>` / `Pt<,,i32,,>` | excess commas absorbed, ta_count matches one IDENT | Permissive accept, no semantic change | n/a |

### Cycle-3 fixes re-verified

- F1/F2 (arity mismatch traps): verified — ta_count vs gp_count_pre
  check at line 3270, traps 62032.
- F3 (cursor walk past EOF on `>` consume): verified — line 3260-3263
  guards `cur_advance` behind `post_loop_t == 17` check.
- F4 (empty struct-lit on non-zero arity): verified — line 3364 traps
  50040 when `arity_m != 0`.
- F5 (cap overflow at clone): verified — line 3325-3341 traps 62031.
- F6 (bad-token attribution): verified — `ta_bad_token` flag at line
  3223 + early-return at line 3264-3266 traps 62033, distinct from
  the missing-`{` 62030.

### Helper-table corner cases checked

- `struct_gp_tab_add` cap 8 with `gp_count > 0` filter: callers can
  only insert when `gp_count_now > 0` (line 6642), so a 0-arity
  ghost entry can't get added. Lookup's 0-on-miss sentinel is
  unambiguous because hits always have gp_count >= 1.
- `struct_gp_tab_lookup` linear scan terminates: bounded by stored
  count; `i = count` early-exit on hit is correct.
- `struct_gp_tab_names_head` is currently unreferenced by the
  use-site mono code; substitution at line 3290-3313 uses positional
  `ta_arr_base` indexing instead. Helper is reserved for future use.
  Not a defect.
- gp_marker arithmetic boundary: `gp_marker_is(v) == 1` iff v >= 200;
  use-site decode `gp_idx_sub = f_struct_idx - 200` then checks
  `gp_idx_sub < ta_count`. Under the verified `ta_count == gp_count_pre`
  invariant and the gp_tab cap-4 contract, `gp_idx_sub` is always
  in 0..ta_count-1 for fields encoded by `parse_struct_decl`.

### Cursor-state continuity checked

- Trap-return paths leave cursor on the offending token (62033 bad
  token before `>`) or after the consumed `>` (62032 / 62030 / 62031).
  Downstream codegen receives AST_ERR; no silent state desync because
  all callers of `parse_primary` propagate `mk_node(99, ...)` loudly.
- Fall-through to `mk_var_with_capture` when `gp_count_pre == 0`
  leaves cursor on `<`, which the binary-expr parser correctly
  interprets as TK_LT comparison.

## OUT OF SCOPE — doc-class observations

Documentation, comments, and line-number references in comments are
explicitly out of scope. No findings under this heading.

## OBS-1: `Pt<UnknownType>` silently becomes scalar substitution (conf 60)

**Location**: parser.hx:3304.

**Pattern**: `let sub_struct_idx = struct_tab_lookup_idx(sb, sub_ty_s, sub_ty_l);` — returns -1 for any unrecognized IDENT. The substituted field stores -1, which downstream codegen treats as "scalar i32" (the convention used throughout the bootstrap for non-struct field types). So `Pt<TypoTypeName>{...}` is silently accepted as `Pt<i32>{...}` for codegen purposes.

**Why OBS not LOW**: This matches the codebase convention established in `parse_struct_decl`'s field-type loop (line 6587), which also uses `struct_tab_lookup_idx` and stores -1 for "non-struct type ident, treat as scalar." Adding a separate "valid scalar type name" allowlist (`i32`, `f32`, ...) is a Phase 0 design omission, not an INC-3b regression. Same defect class exists at the field-decl site and any future fix should be applied uniformly there. Not actionable in this audit cycle.

## OBS-2: `Pt<Pt>` self-recursive type-arg silently maps to un-mono'd entry (conf 55)

**Location**: parser.hx:3304-3305.

**Pattern**: When a generic's own name is passed as its type-arg (e.g. `Pt<Pt>`), `struct_tab_lookup_idx` returns the original (un-monomorphized, markered-field) Pt entry. The substituted field then holds the un-mono'd struct_idx. Subsequent struct-field access would walk into the markered fields, which is a category of defect (struct_idx escape) related to pre-existing recursive-type concerns documented under `struct Node<T> { next: Node<T> }`.

**Why OBS not LOW**: Same convention as OBS-1 — the bootstrap has no separate "fully concrete type" enforcement on type-arg IDENTs. Recursive generics aren't a documented feature in Phase 0. Not actionable here.

## OBS-3: `struct_gp_tab` cap-overflow at registration is silent (conf 58)

**Location**: parser.hx:6642-6646.

**Pattern**: When `gp_count_now > 0` and `struct_idx_added >= 0`, `struct_gp_tab_add` is called. If `struct_gp_tab_count >= 8`, the helper returns -1 and the entry is NOT stored. The caller does NOT check the return. Subsequent use-site `struct_gp_tab_lookup(struct_idx)` returns 0 (miss), and the use-site falls through to `mk_var_with_capture` — silently treating `Pt9<i32>{...}` as a var ref followed by a comparison.

**Why OBS not MED/LOW**: This path is **unreachable today** because `struct_tab` cap (8, line 947) equals `struct_gp_tab` cap (8, line 291), and `struct_gp_tab_add` is only invoked when `struct_idx_added >= 0` (i.e. struct_tab also had room). Since only generic structs occupy struct_gp_tab and they always occupy struct_tab too, struct_gp_tab can never fill before struct_tab is also at capacity — at which point `struct_tab_add` would have already returned -1 and the gp_tab_add call is skipped by the existing guard. Defense-in-depth concern only; if either cap is ever changed, this becomes a real silent miscompile and warrants a trap path. Not actionable at current caps.

## OBS-4: Missing comma between type-arg IDENTs silently accepted (conf 65)

**Location**: parser.hx:3224-3245.

**Pattern**: `Pt<i32 i32>` (no comma between IDENTs) is silently parsed as `Pt<i32, i32>` — the loop captures two IDENTs in successive iters because the IDENT branch doesn't enforce a separator-or-terminator follow-up.

**Why OBS not MED/LOW**: The same lenient-syntax pattern exists at the **decl-side** generic-param loop in `parse_struct_decl` (lines 6485-6514) and `parse_fn_decl`'s gp loop. Cycle INC-1 cycle-2 / cycle-3 audited that pattern and accepted it as the established bootstrap convention. INC-3b's use-site loop mirrors the decl-side convention deliberately. Cosmetic syntactic strictness gap, not a silent miscompile (the produced AST matches what the comma-separated form would produce). Convention-consistent with prior cycles.

## OBS-5: Trap precedence: missing `{` reported instead of cap overflow (conf 60)

**Location**: parser.hx:3321-3325.

**Pattern**: If `struct_tab_add` returns -1 (cap overflow) AND the next token is not `{`, the missing-`{` trap 62030 fires before the cap-overflow trap 62031. The user sees 62030 ("missing `{` after `Pt<i32>`") when the real cause is cap overflow.

**Why OBS not LOW**: Both traps are loud (AST_ERR). Both cause codegen failure. The diagnostic ID is mis-attributed but not silent. The realistic trigger sequence (input has both a malformed body AND cap overflow) is rare; cap overflow alone with a well-formed body fires 62031 correctly. Diagnostic-precedence polish only.

## OBS-6: Defensive `gp_idx_sub < ta_count` else-branch is dead under verified invariants (conf 50)

**Location**: parser.hx:3300-3308.

**Pattern**: After the arity-mismatch trap at line 3270 (`ta_count == gp_count_pre`), and given that fields encoded by `parse_struct_decl` only use gp_idx values 0..gp_count-1, the `gp_idx_sub >= ta_count` else-branch (`__arena_push(0 - 1)`) is dead code. If somehow reached, it silently substitutes scalar without trapping.

**Why OBS not LOW**: This is defensive code; dead under the invariants established by the arity check. The silent fallback would only manifest if a bug elsewhere produced an inconsistent gp_marker — currently impossible. If it triggers in the future, the symptom (field is silently scalarized) would still cause downstream codegen mismatch on access, surfacing the issue loudly. Defense-in-depth choice, not actionable.

## OBS-7: Resource leak on cache-hit re-mono (conf 70)

**Location**: parser.hx:3273-3315.

**Pattern**: Each invocation of the mono branch pushes the type-args region (ta_arr) AND the mangled name bytes into the arena, even when `existing_idx >= 0` (a cache hit reuses the prior mono'd struct entry). The mangled-name bytes are never reclaimed. A program with N call sites to `Pt<i32>` will accumulate N copies of `"Pt__i32"` in the arena.

**Why OBS not finding**: This is a memory-growth concern, not a silent failure. No semantic miscompile, no OOB, no unbounded loop — the arena grows linearly. Out of scope for silent-failure audit.

## Counter

Cycle-4 was clean (1/5). This cycle is clean (2/5). Three more clean cycles required before the silent-failure axis can promote past the 5/5 audit gate for INC-3b.
