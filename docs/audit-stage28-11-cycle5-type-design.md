VERDICT: CLEAN

# Stage 28.11 INC-3b CYCLE-5 type-design audit

**Audit surface**: `helixc/bootstrap/parser.hx` zones at ~line 276-340
(struct_gp_tab helpers), ~line 3193-3401 (use-site mono in parse_primary),
~line 3906-3973 (parse_top region init), ~line 6605-6646 (parse_struct_decl
writer).
**Commit range**: `e33463a..549a68e` (HEAD `32c66bf`).
**Methodology**: diff inspection (~390 lines), 8 invariant verifications,
control-flow trace of the parallel-table writer/reader pair, trap-id
disjointness scan, slot-allocation arithmetic verification.

## Summary

Zero findings at HIGH or MED severity. The INC-3b parallel-table design
(`struct_gp_tab` keyed by struct_idx, parallel to struct_tab) is sound:
helpers are symmetric, the cap is structurally guaranteed by an upstream
cap rather than only checked locally, all five named traps are reachable
and disjoint from the rest of the parser's trap-id space, and the
slot-allocation arithmetic in parse_top correctly lands the new
(base, count) pair at sb+78/79.

## Invariant verification

1. **INV-1 cap=8 honored both sides.** `struct_gp_tab_add` at
   parser.hx:289-302 hard-caps `count >= 8` returning `0 - 1`. Reader
   helpers `struct_gp_tab_lookup` (3193:308-323) and `_names_head`
   (326-341) both terminate their walk at `i < count`, so they cannot
   read past the live region. Furthermore the cap is structurally
   unreachable: each `struct_gp_tab_add` call site (only one, at
   parser.hx:6644) is guarded by `gp_count_now > 0`, so only generic
   struct decls register; and the upstream `struct_tab_add` cap (also
   8, parser.hx:947) ensures at most 8 generic decls ever succeed.
   `struct_gp_tab.count <= n_generic_decls <= struct_tab.count <= 8`.

2. **INV-2 struct_idx references a real struct entry.** The single
   writer at parser.hx:6644 passes `struct_idx_added` directly from
   the immediately-preceding `struct_tab_add` return (parser.hx:6635),
   gated by `struct_idx_added >= 0` (line 6643). So every entry's
   key references a struct_tab slot that succeeded. Mono'd clones at
   parser.hx:3314 also pass through `struct_tab_add` and are NOT
   re-registered in struct_gp_tab (they are non-generic by
   construction), so they cannot violate this invariant either.

3. **INV-3 gp_count >= 1.** Writer gates on `gp_count_now > 0`
   (parser.hx:6642) before calling `struct_gp_tab_add`. Therefore
   every entry has gp_count >= 1; `struct_gp_tab_lookup` returning 0
   means "no entry / non-generic struct," not "generic with 0 params"
   — a sound conflation by design (304-307 comment captures it).

4. **INV-4 200-marker partition preserved.** `gp_marker_base() = 200`
   (parser.hx:271). Real struct_idx inhabits [0, 7] under the 8-cap;
   gp_marker values inhabit [200, 203] under the 4-cap on gp_tab
   (parser.hx:195). INC-3b adds struct_gp_tab but stores entries
   whose three fields are (struct_idx, gp_count, gp_names_head) —
   the struct_idx field reuses the same [0, 7] range as the original
   table, gp_count is small (<= 4), and gp_names_head is an AST
   arena address. None of these stored values approach the 200
   boundary in their interpretation domain. Use-site substitution
   at parser.hx:3298-3299 decodes via `gp_marker_is(v) == 1` then
   `f_struct_idx - gp_marker_base()` — symmetric with the writer.

5. **INV-5 writer/reader field-offset alignment.** Writer (289-302)
   stores entry+0=struct_idx, entry+1=gp_count, entry+2=gp_names_head.
   `struct_gp_tab_lookup` reads entry+1 after matching entry+0
   (parser.hx:315-316). `struct_gp_tab_names_head` reads entry+2
   after matching entry+0 (333-334). Stride is `count * 3` in all
   three helpers. **Aligned.**

6. **INV-6 clone-with-substitution semantics correct for single-level
   mono.** parser.hx:3290-3313 walks each field of the original
   generic struct. For each: copy f_name_s and f_name_l verbatim;
   for f_struct_idx, if `gp_marker_is(f_struct_idx) == 1`, decode
   `gp_idx_sub = f_struct_idx - gp_marker_base()`, look up the
   corresponding type-arg from `ta_arr_base + gp_idx_sub * 2`, and
   substitute with `struct_tab_lookup_idx(sub_ty_s, sub_ty_l)`
   (which yields a struct_idx for nested-struct type-args, or -1
   for scalar like `i32`); otherwise copy f_struct_idx as-is. The
   `gp_idx_sub < ta_count` bounds check at parser.hx:3300 prevents
   out-of-range type-arg references — though that check is in fact
   structurally redundant given the prior arity guard at parser.hx:
   3270 (trap 62032), it is harmless defense-in-depth.

7. **INV-7 trap-id disjointness.** New trap-ids 62030 (missing `{`
   after `Pt<i32>`), 62031 (struct_tab cap overflow at mono use
   site), 62032 (type-args arity mismatch), 62033 (bad token /
   bad terminator in type-args). All four scanned against every
   `mk_node(99, NNNNN, ...)` site in parser.hx — no collisions
   with the existing 50040, 62002, 62005, 62006, 62020-62022,
   71001, 76001-76003, 85001, 88001-88003, 89001, 90001 set.
   The 62030-62033 block fits in the 6202x/6203x band cycle-3
   carved out for INC-3b traps.

8. **INV-8 naming distinction `gp_*` vs `struct_gp_*` meaningful.**
   `gp_tab` (parser.hx:188+) is the per-decl scratch table that
   accumulates generic-param NAMES during a single struct or fn
   decl parse, capped at 4, reset between decls. `struct_gp_tab`
   (this audit) is the PERSISTENT table keyed by struct_idx
   recording which structs were declared generic and what their
   gp-name chain is — survives across decls so use-sites can look
   it up. The two registers serve distinct lifecycles. Naming is
   precise, not tautological.

## Control-flow trace: type-args parsing → mono cloning

Trace of parser.hx:3206-3401 with `gp_count_pre > 0`:

1. **Line 3206-3209**: struct_tab_lookup_idx + struct_gp_tab_lookup
   compose. Disambiguates `Pt<...>` (generic) from `var < 5`
   (comparison) cleanly. Non-struct IDENT or non-generic struct
   IDENT falls through to mk_var_with_capture at 3406.

2. **Line 3212-3246**: type-arg parse loop. ta_arr_base is captured
   from `__arena_len()` BEFORE any pushes (line 3213); the loop
   body only calls cur_advance and __arena_push (no AST node
   allocation), so ta_arr is contiguous (i.e. ta_arr_base + i*2
   addresses entry i correctly). Bad tokens (operators, literals,
   nested `<`) set `ta_bad_token = 1` and exit without consuming,
   surfaced as trap 62033 after the loop. EOF mid-list is also
   caught (cycle-3 F3 fix at parser.hx:3260-3263 — only consume
   `>` if cursor actually points at one).

3. **Line 3260-3272**: post-loop trap-emission cascade — bad
   token, bad terminator, then arity mismatch. Each emits a
   distinct trap id; the ORDER is well-defined (bad terminator
   takes precedence over arity, since a bad terminator implies
   we never had a valid arg list to count).

4. **Line 3274-3315**: name mangling + lookup-or-create. The
   existing_idx fast-path skips re-cloning when the same mono
   instance has been parsed before (e.g. two `Pt<i32>` uses share
   one struct_tab slot). The clone path at 3286-3314 reads
   orig_fields_ptr (set by parse_struct_decl at parser.hx:6592 —
   arena address, immutable once written) and writes new_fields_ptr
   AFTER ta_arr in the arena, so source and destination regions do
   not overlap. struct_tab_add at 3314 may return -1 if the 8-cap
   is exceeded — caught at 3325 by trap 62031.

5. **Line 3321-3400**: body parse. lbrace_t guard at 3322 emits
   62030 if `{` missing. mono_s_idx < 0 guard at 3325 emits 62031.
   Empty body `Pt<i32>{}` correctly arity-checks against arity_m
   (cycle-3 F4 fix, parser.hx:3362-3368). Non-empty body parses
   positional values into AST_TUPLE_LIT, matches against arity_m,
   trap 50040 on mismatch.

## Slot-allocation arithmetic (parse_top, parser.hx:3838-3973)

Verified 80 sequential `__arena_push(0)` calls produce slots 0..79.
Slot 78 receives sgp_base (parser.hx:3972), slot 79 receives 0
initial count (parser.hx:3973). No prior site reads or writes
slots 78/79 — sole users are `struct_gp_tab_base` (286),
`struct_gp_tab_count` (287), and `struct_gp_tab_add`'s count
update (299). Region is 24 slots = 8 entries x 3 fields, matching
the cap declared in struct_gp_tab_add at line 291.

## Observations (not findings)

- **Defensive-redundant cap check.** `struct_gp_tab_add` returns
  -1 on its own 8-cap overflow, but the caller at parser.hx:6644
  does not check that return value. This is benign because the
  upstream struct_tab cap (also 8) provides a structural guarantee
  that struct_gp_tab.count cannot exceed n_generic_decls <= 8.
  The unchecked return is consistent with the cycle-4 design
  discipline of "cap = structurally enforced once, locally redundant
  checks are harmless." No remediation needed.

- **Single-level monomorphization scope.** The clone at
  parser.hx:3290-3313 substitutes top-level gp_markers but does
  NOT recursively monomorphize a field whose declared type is
  another generic struct (e.g. `struct Wrap<T> { p: Pair<T> }`
  + use `Wrap<i32>` would copy `p`'s f_struct_idx pointing at
  the orig (un-monomorphized) `Pair` slot rather than creating
  a fresh `Pair<i32>` clone). This is a documented INC-3b scope
  decision per the audit charter framing (`Pt<i32>` is the
  target). Future INC-3c will need recursive substitution; no
  finding at this stage.

- **Trap node propagation does not reset cursor.** When 62030,
  62031, 62032, or 62033 returns, the cursor may be positioned
  mid-`<TY1,TY2>...{` rather than after a clean recovery point.
  This is consistent with the file's wider trap-then-propagate
  discipline (e.g. line 6713, 6716 in parse_pattern do the same)
  and is not a finding.

## OUT OF SCOPE — doc-class observation

The parse_top region-init comments at parser.hx:3909-3915 are
accurate to the line. No drift. The struct_gp_tab helpers'
docstring at parser.hx:276-285 references "line ~5475" for
parse_fn_decl's gp_chain_head pattern — actual current location
of that pattern is approximately parser.hx:5500-5550 (the file
has drifted modestly since INC-3a). Comment-only; per audit
charter, doc-class drift is not flagged or counted.

## Decision

**Counter advances 1/5 -> 2/5.** Four more clean cycles before
INC-3b closes.
