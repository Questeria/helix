VERDICT: CLEAN (0 HIGH, 0 MED, 0 LOW; 4 OBS doc-class / OOS)

# Stage 28.11 INC-3a CYCLE-12 silent-failure audit

Surface: `helixc/bootstrap/parser.hx`, commits `2e7f836..c95b13f`.

Scope discipline: executable-code semantics only. Doc/comment drift is
listed in the closing OOS section without inflated severity.

## Audit methodology executed

1. `git diff 2e7f836^..HEAD -- helixc/bootstrap/parser.hx` — 199 lines
   of diff, three zones.
2. Cross-referenced live state at lines 226-297 (helpers), 1705-1750
   (reader), 6257-6306 (writer).
3. Verified call-graph: `gp_marker_decode` has zero call sites (only a
   mention in the cycle-7 doc rationale).
4. Verified `gp_marker_base/encode/is` callers: 2 executable sites
   (writer 6286, reader 1729). Other matches are comments.
5. Verified gp_tab lifecycle: `gp_tab_reset` at parse_struct_decl entry
   (6132) and exit (6315); `gp_tab_add` during `<T1,T2,...>` parse
   (6205). gp_tab cap is 4 (gp_tab_add returns -1 at >=4), so
   `gp_idx` is in `[0,3]` and `gp_marker_encode(gp_idx)` is in
   `[200,203]`.
6. Verified reader chain termination on `cur_struct_idx = -1`
   (outer guard at 1699 `lhs_struct_idx >= 0`).
7. Verified struct_tab fields region cannot receive a stray `100+`
   value: writer's ladder at 6284-6289 picks between
   `gp_marker_encode(gp_idx)` (200+) and `struct_tab_lookup_idx(...)`
   (raw struct_idx in [0,7] or -1). No 100+ path leaks into a field
   slot.
8. Boundary checks: `gp_idx=0` encodes to exactly 200,
   `gp_marker_is(200)==1` (marker detected). `gp_idx=3` encodes to 203,
   still in marker band. `gp_idx=-1` (miss) cannot reach
   `gp_marker_encode` because the writer guards on `gp_idx >= 0`.
9. Verified writer/reader are co-modified atomically (single commit
   `2e7f836`); no intermediate-state regression window.

Total executable findings: 0 HIGH, 0 MED, 0 LOW.

## OUT OF SCOPE — doc-class observations

None of the items below are executable-code defects. They are
recorded for completeness only and MUST NOT bump the cycle's
clean-counter eligibility.

- The helpers doc-block at line 244 lists "the following 7 sites"
  and enumerates parser.hx:4156, 4157, 4176, 4177, 5453, 5458, 5534
  as remaining raw-200 sites. Live grep on the current file shows
  the actual `\b200\b` sites are at 5459, 5464, 5540 (and the
  Stage-8 monomorphize sites that the cycle-9 polish was supposed
  to re-grep). The cycle-9 commit message claims the line numbers
  were updated; they appear to drift again. This is enumeration
  drift in a comment block, OOS.
- Line 6271 comment says "the downstream reader at parser.hx:1635";
  the actual reader is at 1705-1749. Stale line number in a comment,
  OOS.
- Line 6264-6265 comment references "parse_fn_decl's fn-param
  encoding at parser.hx:5346"; the actual fn-param encode site is
  at 5459 (`p_ty_generic = if gp_idx_p >= 0 { 200 + gp_idx_p }`).
  Stale line number, OOS.
- Line 230 says "writer parse_struct_decl ~6238, reader
  parse_primary ~1681"; live positions are ~6286 and ~1729. Stale
  approximate line numbers, OOS.

## Conclusion

Zones 1, 2, and 3 are semantically correct under all boundary cases
considered (gp_idx=0, gp_idx=3, gp_idx=-1 miss, f_struct_idx=0,
f_struct_idx=-1). The cycle-7 decision to omit `gp_marker_decode`
remains sound (no dangling caller). The guard ladder in the writer
correctly mirrors parse_fn_decl's existing pattern. The reader's
three-way classification (outer `>= 0` partition + inner
`gp_marker_is` partition) covers the full state space without
silent fallthrough.

VERDICT: CLEAN
