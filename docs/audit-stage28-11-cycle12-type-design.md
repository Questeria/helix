VERDICT: CLEAN

# Stage 28.11 INC-3a CYCLE-12 type-design audit

**Audit surface**: `helixc/bootstrap/parser.hx` zones at ~line 222 (helpers),
~line 1705 (reader), ~line 6257 (writer).
**Commit range**: `2e7f836..c95b13f`
**Methodology**: diff inspection (199 lines), invariant verification (cap
proofs for `struct_tab` and `gp_tab`), writer/reader symmetry check, helper-
composition usage scan across the file.

## Summary

Zero findings at HIGH or MED severity. The cycle-5 helper extraction
(`gp_marker_base / gp_marker_encode / gp_marker_is`) is clean and the
cycle-7 decision to omit `gp_marker_decode` is sound: the helper would be
a partial function with no trap primitive available, so call-site
composition `gp_marker_is(v) == 1 { let gp_idx = v - gp_marker_base(); }`
is the right discipline.

## Invariant verification

1. **Writer/reader symmetric.** Writer at parser.hx:6286 calls
   `gp_marker_encode(gp_idx)` which is `gp_marker_base() + gp_idx`. Reader
   at parser.hx:1729 calls `gp_marker_is(v) == 0` which guards
   `v >= gp_marker_base()`. Same boundary literal (200) sourced from a
   single function — symmetric by construction.

2. **Partition-boundary cap holds.** `struct_tab_add` at parser.hx:880
   hard-caps `count >= 8` returning `0 - 1`. Therefore valid struct_idx
   inhabits [0, 7]. `gp_tab_add` at parser.hx:195 hard-caps count at 4,
   so encoded marker values inhabit [200, 203]. The 192-slot gap between
   max real struct_idx (7) and `gp_marker_base()` (200) is preserved at
   all live storage sites.

3. **Helper signatures clean.** All three helpers are pure unary/nullary
   `i32 -> i32` with no `sb` smuggling. No partial functions left
   dangling — `gp_marker_encode` is only called when `gp_idx >= 0` is
   guarded at parser.hx:6285.

4. **`gp_marker_is` and `gp_marker_encode` mutually consistent.**
   `gp_marker_is(gp_marker_encode(k)) == 1` for all `k >= 0`, since
   `gp_marker_encode(k) = 200 + k >= 200` iff `k >= 0`. Verified by
   symbolic reduction against the helper bodies at parser.hx:271-275.

5. **No co-mingled encoding collision.** `struct_tab_field_struct_idx`
   at parser.hx:1107 returns exactly the third slot of the field triple
   written by `parse_struct_decl`. No intermediate transform; raw passthrough.
   The marker range [200, 203] never collides with the live encoding
   regions (struct_idx [0,7], scalar sentinel -1).

6. **Reader's gp-marker branch isolates downstream propagation.**
   parser.hx:1742-1743 emits `mk_node(52, prim, f_idx, 0)` (p3=0, 4-byte
   read) and `cur_struct_idx = 0 - 1`, preventing chained dot access on
   a generic-typed field at INC-3a stage. This matches the documented
   "scalar pending INC-3b monomorphization" semantics and matches the
   existing `struct Pt<T> { x: T, y: T }` probe exit-42 behavior.

7. **No dead-code paths.** Both branches of the reader's gp_marker_is
   check are reachable (writer can produce both 200+ and < 200 values
   in the f_struct_idx slot depending on whether `gp_tab_lookup`
   returns >= 0).

## Active-surface raw-200 sites

Two INC-3a active sites use helpers (1729 reader, 6286 writer). Seven
deferred Stage-8 / parse_fn_decl raw-200 sites (parser.hx:4162-4163,
4182-4183, 5459, 5464, 5540) remain open-coded per the documented
cycle-71 narrow-scope discipline. These are out of INC-3a's active
surface per the audit charter and the in-source cycle-7 docblock.

## OUT OF SCOPE — doc-class observation

The cycle-7 docblock at parser.hx:251-258 lists deferred raw-200 sites
with stale line numbers ("5346", "5453", "5458") versus current
positions (5319, 5459, 5464, 5540). Cycle-9 already updated some
references; the remaining drift is comment-only and explicitly excluded
from this audit's scope per the charter (docstring line-number drift
not flagged).

## Decision

**Counter advances 2/5 -> 3/5.** Three more clean cycles needed before
INC-3a closes.
