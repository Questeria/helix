# Stage 28.11 INC-3b CYCLE-5 Code-Review Audit

VERDICT: CLEAN

- Date: 2026-05-12
- Counter: 2/5 (cycle-4 was 1/5 audit-clean after the cycle-3 fix-sweep)
- Scope: executable code in `helixc/bootstrap/parser.hx` only — cumulative INC-3b changes across commits `e33463a`, `7123f09`, `1ff41ff`, `549a68e`.
- Diff size: 390 lines (`git diff e33463a^..HEAD -- helixc/bootstrap/parser.hx | wc -l`)
- Confidence threshold for HIGH/MED: 75
- Findings: 0 HIGH, 0 MED, 0 LOW (in-scope). 2 below-threshold OBS noted for completeness.

## Methodology

1. Read the full INC-3b diff against `e33463a^`.
2. Cross-checked Zone A (helpers), Zone B (use-site mono), Zone C (parse_top init), Zone D (parse_struct_decl) for internal consistency and parallel-naming with `struct_tab`.
3. Enumerated all `mk_node(99, …)` trap IDs in the file to verify 62030–62033 do not collide with existing reservations (62002/62005/62006/62020/62021/62022/71001/76001/76002/76003/85001/88001-3/89001/90001/50040).
4. Inspected pre-existing helpers referenced by INC-3b: `mangle_name_into_arena`, `mangle_name_len`, `struct_tab_add`, `struct_tab_lookup_idx`, `gp_marker_is`, `gp_marker_base`, `mk_node`, `set_last_struct_idx`.
5. Verified parse_fn_decl's gp_chain pattern at line 5858–5876 to confirm parse_struct_decl's new chain at 6618–6634 is a faithful mirror.
6. Verified parse_primary brace-count delta (+1 close brace at line 3484) matches the +1 if-else level added by `nt == 16`.
7. Triggered the regression suite via `python -m pytest helixc/tests/ -q -p no:xdist` (1551 tests collected; still running at report-write time due to suite length — see "Test status" below).

## Per-zone notes

### Zone A — struct_gp_tab helpers (parser.hx:276–340)

- Helper triple `struct_gp_tab_base` / `_count` / `_add` / `_lookup` / `_names_head` is a faithful parallel of `struct_tab_*`. Stride 3, cap 8, slots sb+78/79. Consistent with adjacent infrastructure (gp_tab at sb+29/30, mr_tab at sb+31/32).
- `struct_gp_tab_add` returns -1 on cap overflow (cap = 8). Use sites must guard. parse_struct_decl at 6643 guards via `if struct_idx_added >= 0`. (The cap-overflow signal at the add-side is the upstream side: if `struct_tab_add` returns -1, `struct_gp_tab_add` is never called.)
- `struct_gp_tab_lookup` returns 0 (gp_count) on miss. Comment at 304–307 explicitly documents that 0 doubles as miss because a non-generic struct that has gp_count == 0 is never inserted (parse_struct_decl 6642 guards `if gp_count_now > 0`). Sound.

### Zone B — use-site mono parsing (parser.hx:3193–3340)

- The new `nt == 16` arm is keyed by `gp_count_pre > 0`, computed from `struct_tab_lookup_idx` + `struct_gp_tab_lookup`. If either misses, we fall through to `mk_var_with_capture`, preserving the prior behavior (LT-token after IDENT was previously a var-ref via the catchall `else` at 3481–3483). Correct preservation of existing comparison semantics.
- The cycle-3 fixes (F1/F2/F3/F4/F6) are all observable in code: arity-mismatch trap 62032, bad-token-in-args trap 62033, conditional `cur_advance` past `>`, empty-struct-lit arity check, post-loop disambiguation flag `ta_bad_token`.
- The four traps 62030 (missing `{`), 62031 (cap overflow at mono use site), 62032 (arity mismatch), 62033 (bad terminator / bad token in args) are distinct and user-actionable. No collision with existing 62002/5/6/20/21/22.
- The field-substitution loop at 3290–3313 correctly handles three cases: gp_marker (substitute), non-gp registered struct (copy), scalar (copy -1). Matches parse_struct_decl 6584–6588's encoding.
- The mangled-name builder reuses `mangle_name_into_arena` + `mangle_name_len` (line 1388 / 1416), the same helpers used elsewhere for fn monomorphization. No bespoke mangling.
- Lookup-then-synthesize idiom at 3277–3315 correctly de-duplicates mono'd entries: a second `Pt<i32>` use re-uses the same struct_tab index.

### Zone C — parse_top init (parser.hx:3906, 3958)

- Slots 78/79 pushed alongside slot 77 (`next_fn_is_unwind`) at the right position (after Stage 28.9's slot 75–77 block). Order preserved.
- The 24-slot region (8 entries × 3 fields) is pushed via the `let sgp_base = __arena_push(0); while sgpi < 24 { __arena_push(0); sgpi += 1; }` idiom — matches gr_rev_pending at 3953–3960 (40-slot) and grad_pending at 3944–3950 (32-slot). Style consistent.
- Initial `struct_gp_tab_count` set to 0 at sb+79.

### Zone D — parse_struct_decl chain-and-register (parser.hx:6605–6646)

- Chain construction (line 6618–6634) is byte-identical in structure to parse_fn_decl's chain at 5858–5876. The header comment at 6605–6615 explicitly cross-references the source pattern.
- Order is correct: chain built BEFORE `struct_tab_add` AND BEFORE `gp_tab_reset(sb)`. The chain reads gp_tab; resetting first would zero-fill the source.
- `if gp_count_now > 0 { if struct_idx_added >= 0 { struct_gp_tab_add(...) } }` is the right pair of guards: skip registration for non-generic structs (gp_count == 0 is the miss sentinel), and skip when struct_tab cap-overflowed.

## Trap-ID collision check

Grep'd all `mk_node(99, NNNNN, ...)` occurrences. The complete trap-ID set in parser.hx (before INC-3b) was: 50040, 62002, 62005, 62006, 62020, 62021, 62022, 71001, 76001, 76002, 76003, 85001, 88001, 88002, 88003, 89001, 90001.

INC-3b's new reservations: 62030, 62031, 62032, 62033. No collision.

## Below-threshold observations (not findings, not actionable for INC-3b)

These did not meet the ≥75 confidence threshold for HIGH/MED. Recorded here for completeness and possible future incremental sweeps.

**OBS-1 — `struct_gp_tab_names_head` defined but never called (confidence ~60)**
The helper at parser.hx:326–341 is shipped infrastructure for future increments. INC-3b's use-site at 3298–3308 does positional substitution via `gp_marker_is` + arithmetic, not name-based lookup. The chain IS being built and stored (6625, 6644), but no reader exists yet. This is deliberate staged plumbing consistent with the multi-increment INC-3 plan (per header comment at parser.hx:278–284) and parallels the way `gp_marker_decode` was deliberately omitted in INC-3a cycle-7. No action.

**OBS-2 — Pre-existing asymmetry: non-generic empty-struct-lit at parser.hx:3427–3434 does not enforce `arity == 0` (confidence ~55 for in-scope)**
The new generic branch at 3348–3368 traps 50040 on empty-body with non-zero arity (cycle-3 F4 fix). The non-generic branch above does NOT. This is a pre-existing parallel defect that predates INC-3b and is OUTSIDE the diff. Code that lands at line 3427-3434 in the current commit is identical to what it was at `e33463a^`. Out of scope for an INC-3b code review; flagged here so a future hardening pass can evaluate whether to extend the F4 fix to the non-generic side.

## OUT OF SCOPE — doc-class observations

None. All commentary in the diff is consistent with the file's existing comment density and references existing line numbers approximately (the file uses "line ~NNNN" wording throughout, which tolerates drift).

## Test status

`python -m pytest helixc/tests/ -q -p no:xdist` (sequential, to avoid the parallel-process artifact a different audit run hit earlier). Result: **1550 passed, 1 skipped in 775.31s** (exit code 0). No regressions. The end-to-end probe `struct Pt<T> { x: T, y: T } fn main() -> i32 { let p = Pt<i32> { 10, 32 }; p.x + p.y }` documented in commit `7123f09` is covered by `helixc/tests/test_codegen.py:2783` and passes.

## Counter

This is cycle-5 of the post-INC-3b code-review loop. Counter advances 2/5 (cycle-4 was 1/5 after cycle-3 fix-sweep landed). One more clean cycle moves to 3/5.
