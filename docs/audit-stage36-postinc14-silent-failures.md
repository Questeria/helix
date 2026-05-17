# Stage 36 post-Inc-14 — Silent-Failure Audit

## Scope

Diff audited: `git diff HEAD~3..HEAD` on `main` (commits `13784a9`,
`6894348`, `e7c3552`) plus `abef645` (Inc 12 catch-up tests, referenced
by the user but outside HEAD~3..HEAD) and the upstream Inc 12 code
landing `4742128` for the integer-Logic AD guard verification. Files:

- `helixc/backend/x86_64.py` — ARENA_PUSH_TRIPLE codegen
- `helixc/frontend/{autodiff,parser,typecheck}.py`
- `helixc/ir/lower_ast.py`, `helixc/ir/passes/{dce,effect_check}.py`,
  `helixc/ir/tir.py`
- `helixc/stdlib/provenance.hx`
- `helixc/tests/test_stage36_provenance.py`

## Summary

**2 findings (1 HIGH + 1 MEDIUM)**. The new ARENA_PUSH_TRIPLE codegen
is byte-tight and the DCE/effect-check registrations are correct, but
`parent_at(handle, slot)` admits two silent-failure surfaces that the
two-parent `parent_left_at` / `parent_right_at` accessors do NOT have,
and `helixc/stdlib/provenance.hx` exports observation helpers that
silently mis-read three-parent records.

The Inc 12 integer-Logic AD guard is sound (both forward and reverse
fail loudly; one minor coverage gap noted as LOW).

---

## Findings

### H1 — `parent_at(handle, slot)` silently reads cross-record arena slots (HIGH, conf 90)

- **File:** `helixc/ir/lower_ast.py:2141-2167`, `helixc/frontend/typecheck.py:2977-2994`
- **Description:** `parent_at(h, slot)` lowers to `_safe_arena_get(h-1+slot, 0)`.
  The bounds check inside `_safe_arena_get` (lower_ast.py:2051) gates
  only on `[0, arena_len)` — it has no notion of which arena slot
  *belongs* to handle `h`. Phase-0 explicitly does not track per-handle
  arity, and the Inc 14 comment at `lower_ast.py:2147-2152` openly
  acknowledges the failure mode ("`parent_at(h, 2)` … for two-parent
  handles … reads into whatever happens to live at slot N+2, which may
  be another derivation's slot"). That comment is documenting a bug,
  not a feature: there is no language-level signal to the user that
  the returned i32 is a value from a *different* derivation.

- **Hidden errors / failure modes:**

  1. `parent_at(h, 2)` on a 2-parent handle returns the *next*
     derivation's `left` value (or the previous one's `right`,
     depending on layout). Indistinguishable from a legitimate i32
     source id.
  2. `parent_at(h, slot)` with a dynamic `slot < 0` shifts the
     effective index *back* into a previous record. For any
     `h >= -slot + 1`, eff_idx is non-negative and the OOB sentinel
     is bypassed. The user observes a sibling record's data.
  3. `parent_at(0, slot)` with `slot >= 1` defeats the null-handle
     sentinel invariant. `parent_left_at(0)` is guaranteed to return
     `-1` because `_safe_arena_get(-1, 0)` triggers the `eff_idx >= 0`
     gate. But `parent_at(0, 1)` computes eff_idx = 0, passes the
     bounds check, and returns `arena[0]` — usually the first
     registered derivation's `left` source id. A reflexive
     `if has_evidence(h) { parent_at(h, k) ... }` is NOT safe at
     `h == 0` because `has_evidence` short-circuits at `h <= 0` but
     a user composing other predicates may not.

- **Reproducer (sketch):**
  ```helix
  let h1 = register_derivation(11, 22);     // arena slots 0,1 = 11,22; handle 1
  let h2 = register_derivation(33, 44);     // arena slots 2,3 = 33,44; handle 3
  print_int(parent_at(h1, 2));              // prints 33, NOT -1
  print_int(parent_at(0, 1));               // prints 22, NOT -1
  print_int(parent_at(h2, -1));             // prints 22, NOT -1
  ```
  The test file `test_stage36_provenance.py:2069-2089` only exercises
  the *intended* `slot in {0,1,2}` access pattern; no negative test
  covers the silent cross-record read.

- **User impact:** Provenance-tracking user code (Helix's own KG
  reasoner dogfood, Stage 36's whole point) gets wrong "parent" answers
  with no panic, no `-1`, no log line. Debugging this requires
  reverse-engineering arena layout from `helixc/backend/x86_64.py`.

- **Suggested fix (pick one):**
  - **Tightest:** stamp a per-record arity word at push time
    (`ARENA_PUSH_PAIR` writes 3 slots, `ARENA_PUSH_TRIPLE` writes 4,
    layout becomes `[arity | l | m? | r]`). `parent_at` reads the
    arity slot and returns `-1` when `slot >= arity`. Costs +1 word
    per record.
  - **Cheap, partial:** require `slot` to be an `IntLit` at lowering
    time, statically bounds-check `0 <= slot <= 2`, and raise a
    typecheck error otherwise. Doesn't help with cross-record
    confusion when the user mixes 2-parent and 3-parent handles, but
    closes the `parent_at(h, -1)` and `parent_at(h, 9999)` paths.
  - **Documentation-only is NOT sufficient** — the comment at
    `lower_ast.py:2147-2152` already documents the hazard and the
    silent failure still ships.

---

### M1 — `provenance.hx` helpers silently mis-read three-parent records (MEDIUM, conf 88)

- **File:** `helixc/stdlib/provenance.hx:1-61` (entire file)
- **Description:** `evidence_left(h)` aliases `parent_left_at(h)`
  (reads slot `h-1`) and `evidence_right(h)` aliases
  `parent_right_at(h)` (reads slot `h`). For a handle returned by
  `register_derivation3(L, M, R)` the arena layout is
  `[h-1]=L, [h]=M, [h+1]=R`. So `evidence_right(h)` on a three-parent
  handle returns `M`, the *middle*, not the *right*. The function
  name lies about the value it returns.

  `trace_evidence(handle)` prints `"L=<left> R=<right>"` and runs the
  same `parent_right_at` underneath, so the diagnostic line itself is
  silently wrong for any handle produced by `register_derivation3` —
  exactly the path Inc 14 just shipped.

  `has_evidence(handle)` returns 1 (true) for triple handles as long
  as `parent_left_at(handle) != -1`. That's fine for the predicate
  itself, but it composes with the mislabelled `evidence_right`: a
  caller pattern of "if `has_evidence(h)` then `evidence_right(h)`"
  reliably returns the *middle* parent of a 3-parent record without
  any indication.

  Inc 14 introduces `register_derivation3` + `parent_at` and Inc 13
  ships `provenance.hx`, but `provenance.hx` was NOT updated in the
  same series. The two increments shipped within hours of each other;
  the helpers are now stale on arrival.

- **Reproducer:**
  ```helix
  let h = register_derivation3(7, 11, 13);
  trace_evidence(h);  // prints "h=1 L=7 R=11\n"  — wrong: R should be 13
  print_int(evidence_right(h));  // 11, not 13
  ```

- **User impact:** Debug/observation code lies to the developer. A
  user inspecting "why did the KG reasoner derive this?" gets a
  plausible-looking but wrong middle-as-right datum, and may chase a
  nonexistent provenance bug for hours. Worse than `H1` because the
  helpers are explicitly *for* introspection, so wrong output is the
  most damaging possible failure mode.

- **Suggested fix:**
  - Add `evidence_middle(h: i32) -> i32 { parent_at(h, 1) }` and
    redefine `evidence_right(h)` to `parent_at(h, 2)` for triple
    handles. But this requires distinguishing the two record types
    at runtime — which Phase-0 does not — so the cleanest fix is the
    per-record-arity word from `H1`. Until then:
  - At minimum, rename `evidence_right` to `evidence_slot1` (or just
    delete it pending the arity-word landing) and update
    `trace_evidence` to print `"slot[0]=<v> slot[1]=<v>"` rather than
    `"L=<v> R=<v>"`. Names that don't promise structure can't lie
    about it.
  - Add a `trace_evidence3(h)` printing all three slots, gated on
    the caller knowing the handle is 3-parent.

---

## Concerns ruled out

The following items I checked and judged clean:

- **Codegen byte arithmetic.** Triple's `jmp +0x18` and bounds
  `cmp ecx, CAP-2 / jb` are correct. The in_bounds block measures
  exactly 24 bytes (4+5+5+3+2+3+2), matching the jump displacement.
  The bounds inequality `cursor < CAP-2 ⇒ cursor+3 <= CAP` is right.
  No off-by-one.
- **Sentinel collision -1 → 0.** ARENA_PUSH_TRIPLE returns -1 on
  overflow; lower_ast then `ADD 1` makes the user-visible handle 0
  (null sentinel). Same fail-closed pattern as ARENA_PUSH_PAIR. The
  intermediate -1 is never bound to a user-visible name; it's a TIR
  value passed directly to ADD. No path lets the user observe it.
- **DCE / effect_check registrations.** Both `dce.SIDE_EFFECT_KINDS`
  (line 62) and `effect_check.OP_EFFECTS` (line 99) include
  ARENA_PUSH_TRIPLE with the same shape as ARENA_PUSH_PAIR. A `@pure`
  function calling `register_derivation3` will fail effect_check.
- **AD purity registration.** `parent_at` is correctly added to
  `AD_KNOWN_PURE_CALLS` (autodiff.py:82) — it's an arena READ, no
  mutation. `register_derivation3` is correctly *omitted* (mirrors
  `register_derivation`'s post-Inc-9-B2 status), so AD let-erasure
  won't drop a side-effecting push.
- **stdlib parse-time failure.** `parser._merge_stdlib` (parser.py:1611)
  prints `"helixc: stdlib file missing: <path>"` to stderr and only
  raises under `HELIXC_STDLIB_STRICT=1`. Adding `provenance.hx` to
  `STDLIB_FILES` inherits the existing behavior — not a regression.
  A parse *error* (vs. missing file) propagates via the normal Parser
  exception, which is clearly attributed to `provenance.hx` since the
  Lexer is fed `stdlib_path`. Acceptable.
- **Inc 12 AD integer-Logic guard.** `_diff_call_chain_rule`
  (autodiff.py:1070) and `_propagate` (autodiff_reverse.py:197) both
  test `isinstance(node.callee, A.Name) and name in
  AD_INTEGER_VALUED_LOGIC`. Method-call shapes (`x.and_logic(y)`)
  would bypass both, but the entire AD pass treats non-Name callees
  as opaque elsewhere, so the integer-Logic op via method syntax
  would hit the existing opaque-call NotImplementedError. Symmetric
  and tight. Inc 12 catch-up tests (`test_stage36_provenance.py`)
  cover the canonical bypass attempts (forward, reverse, no-twin,
  let-erasure positive control).

## Note on coverage

LOW: there is no test exercising `parent_at` with a *dynamic*
(non-literal) `slot` argument that crosses record boundaries. All
existing tests pass literals 0/1/2. Recommend adding one once `H1`
is fixed, to pin the new bounds semantics. Conf 70.
