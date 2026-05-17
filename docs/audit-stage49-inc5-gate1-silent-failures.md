# Stage 49 Inc 5 — Gate-1 Silent-Failure Audit

**Date:** 2026-05-17
**HEAD:** `47d8f66` ("Stage 49 Inc 4: real `?` early-return branch IR")
**Range audited:** `7eaba56..47d8f66` (4 commits: Inc 1, Inc 2, Inc 3, Inc 4)
**Lens:** silent failures (gate-1 of Stage 49 closure)

> Stage 49 is a multi-touchpoint change (TIR opcodes + IR lowering +
> typecheck + backend codegen) of the exact shape that Stages 46–48
> shipped silent miscompiles in every closure cycle. Mission: hunt
> for places where the compiler produces plausible-looking output
> that is semantically wrong, especially at the layering seam where
> the Phase-0 static-provenance machinery now coexists with the new
> runtime Ok/Err tag.

---

## Scope

Read-only audit over:

- `helixc/ir/tir.py` (+30) — new `RESULT_PACK` / `RESULT_TAG` /
  `RESULT_PAYLOAD` opcodes + convention block.
- `helixc/ir/lower_ast.py` (+242) — `_lower_type` Result arm (now
  returns packed i64); dedicated lowering arm for `Ok`, `Err`,
  `unwrap_ok`, `unwrap_err` (with Inc 1.5 runtime wrong-arm trap),
  `__try` (real conditional early-return), `is_ok` / `is_err`,
  `map_ok` / `map_err` (SELECT on tag).
- `helixc/backend/x86_64.py` (+58) — three new emit arms for the
  packed-tag ops; `shl_rax_imm8` / `shr_rax_imm8` / `mov_eax_eax`
  asm helpers.
- `helixc/frontend/typecheck.py` (+381) — provenance dict
  steward-comment refresh + dual `_result_let_block_scopes` /
  `_result_assigns_block_scopes` stack restore (gate-5 G4-F2
  carry-forward); `_check_expr_in_block_scope` helper; lifted
  rejection of `is_ok` / `is_err` / `map_err` / static-Err `?`.
- `helixc/tests/test_stage49_runtime_tag.py` (+738) — 38 collected
  tests across Inc 1, 1.5, 2, 3, 4 (the in-source comment header
  claims 72 cases; only 38 functions are present — informational).

Diff stats:

| Patch         | Files | +Lines | -Lines |
|---------------|-------|--------|--------|
| Stage 49 full | 12    | 2508   | 268    |

Test posture: `pytest helixc/tests/test_stage49_runtime_tag.py -q`
passes all 38 cases (no skipped). dogfood_17_try_operator.hx
end-to-end exit code 42 is regression-covered by Inc 4's dedicated
test (`test_stage49_inc4_dogfood_17_still_exits_42`).

---

## Defect-pattern checklist (against the 9 patterns in the brief)

| # | Pattern                                              | Result        |
|---|------------------------------------------------------|---------------|
| 1 | Static-Ok pathway preservation                       | runtime-OK; not bit-identical (pack-then-unpack), but exit-42 invariant intact (Inc 4 regression test passes) |
| 2 | Calling-convention CALL/RETURN agree on packed-i64   | clean — `_is_64bit_int_type` arm in CALL/RETURN/spill all route i64 through full rax/rdi |
| 3 | Wrong-arm detection (static + runtime layering)      | clean — static-provenance reject + Inc 1.5 runtime TRAP coexist; no override or conflict |
| 4 | `__try` Err-propagation correctness                  | clean for return-value width AND control flow; **but see SF1-F1: missed TRACE_EXIT for @trace fns** |
| 5 | `map_err` arm-skip                                   | clean — SELECT correctly preserves r_packed on the Ok arm |
| 6 | `is_ok` / `is_err` const-folding consistency         | clean — no folding, always consults RESULT_TAG; static and runtime views are forced to agree |
| 7 | Cascade-integrity packed-tag stability               | clean — backend uses `shl_rax_imm8(32)` / `shr_rax_imm8(32)` deterministically; logical (zero-extending) shift; little-endian rax store/load consistent across pack/extract |
| 8 | Empty-payload edge case (payload=0)                  | clean — `Ok(0)` packs as `0x00000000_00000000`, `Err(0)` packs as `0x00000001_00000000`; tag bit cleanly differentiates; no sentinel collision found |
| 9 | `map_ok` SELECT upgrade — both-arms-evaluated        | **see SF1-M2: both arms always evaluated by IR SELECT; side effects in new_val fire unconditionally** |

Additional patterns probed beyond the brief:

| #  | Pattern                                              | Result        |
|----|------------------------------------------------------|---------------|
| 10 | Result inner-type width — i32 only?                  | **see SF1-F2: no typecheck guard; `Ok(5G_i64)` typechecks clean and silently truncates** |
| 11 | `__try` inside If/Match arm (control-flow nesting)   | clean — ok_blk becomes current_block, BR-to-merge fires from ok_blk; err_blk's RETURN properly terminates |
| 12 | Multiple `?` in one expression (`f()? + g()?`)       | clean — each `?` allocates its own err_blk/ok_blk pair |
| 13 | Negative i32 payload round-trip                      | clean — zero-extending `mov ecx, [rbp+...]` + `mov eax, [rbp+packed]` correctly preserves sign bit on i32 reinterpret |
| 14 | `_check_block` snapshot/restore with dual stacks     | clean structure; **see SF1-L4 pre-try-leak parity** |

Detailed findings below.

---

## Findings

### SF1-F1 — HIGH (confidence 90): `__try` Err arm emits RETURN without TRACE_EXIT, silently corrupts trace stream for `@trace` functions using `?`

**Location:** `helixc/ir/lower_ast.py:2186-2187`
(`__try` Err arm — `self.builder.switch_to(err_blk); self.builder.emit(tir.OpKind.RETURN, packed)`).

**Defect class:** same family as Audit 28.8 cycle 2 C2-2 fix
(`A.Return` arm at `lower_ast.py:3718-3734`). Pre-C2-2, only the
fall-through return at the end of `_lower_fn_body` emitted
TRACE_EXIT, so an explicit early `return X` in a `@trace`'d fn
produced an unbalanced ENTRY-without-EXIT pair. C2-2 added the
TRACE_EXIT emission to the `A.Return` arm. Stage 49 Inc 4
introduces a SECOND early-return site — `__try`'s Err arm — that
emits `RETURN` directly via `self.builder.emit(tir.OpKind.RETURN,
packed)`, bypassing both the `A.Return` arm AND the C2-2 fix.

When the enclosing fn is `@trace`-annotated and `?` propagates
an Err at runtime, the trace stream gets:

```
TRACE_ENTRY(fn_name="helper", ...)
... user ops ...
RETURN packed     ← no preceding TRACE_EXIT
```

Concrete trace:

```helix
@trace
fn helper(r: Result<i32, i32>) -> Result<i32, i32> {
    let v: i32 = r?;       // <-- if r is Err, propagates without TRACE_EXIT
    Ok(v + 1)              // <-- this path's RETURN goes through A.Return,
                           //     gets correct TRACE_EXIT
}
```

**Hidden errors:** any unbalanced ENTRY/EXIT pair in the trace
ring buffer when consumed by a runtime that pairs entries by
position. The Stage 25 / Audit 28.8 A7 backend currently stubs
TRACE_ENTRY / TRACE_EXIT (no-ops), so the corruption is invisible
at Phase-0 — but as the C2-2 fix comment notes: "would corrupt the
buffer the moment Stage 30 runtime exists."

**User impact:** any user writing `@trace fn foo() -> Result<...>`
with `?` propagation will silently corrupt their trace stream once
the Stage 30+ runtime lands. Diagnosis would require deep buffer
forensics — the source code looks fine, the typecheck passes, the
test exit codes match. Worst-case symptoms: post-trace analytics
miscounting fn calls, dropped EXIT records, or downstream parser
crashes on the ENTRY-with-no-EXIT.

**Repro path:** static — read `lower_ast.py:2186-2187` and compare
to the C2-2 site at `lower_ast.py:3726-3733`. The `_is_fn_traced`
+ TRACE_EXIT-before-RETURN pattern is missing.

**Concrete fix sketch:** insert before line 2187 in the err-arm
switch:

```python
self.builder.switch_to(err_blk)
# Mirror the A.Return arm's C2-2 fix: emit TRACE_EXIT before
# the RETURN if the enclosing fn is @trace'd.
if self._is_fn_traced:
    self.builder.emit(
        tir.OpKind.TRACE_EXIT, packed,
        attrs={"fn_name": self._current_fn_name or "<unknown>"},
    )
self.builder.emit(tir.OpKind.RETURN, packed)
```

Add a regression test in `test_stage49_runtime_tag.py` that
inspects the IR for a `@trace` Result-returning fn with `?`:
assert that every block ending in `RETURN` has a `TRACE_EXIT` as
the immediately-preceding op.

**Severity rationale:** HIGH (90) because:
- Silent and unobservable at Phase-0 (backend stubs the ops).
- Becomes corruption the moment Stage 30 runtime lands.
- Same defect class as a previously-fixed defect (C2-2) — the
  fix discipline exists but wasn't carried forward to the new
  early-return site.
- Trivial to fix (3 lines).
- The combination "@trace + ?" is rare enough today that the
  bug may sit dormant for many cycles; the longer it sits, the
  more painful the downstream debug.

**Confidence:** 90.

---

### SF1-F2 — HIGH (confidence 85): no typecheck guard enforces the i32-payload constraint; non-i32 inner types silently truncate at RESULT_PACK / RESULT_PAYLOAD

**Location:**
- `helixc/ir/lower_ast.py:2061-2077` (Ok / Err arms — `payload = self._lower_expr(...)` with no width check).
- `helixc/ir/lower_ast.py:2143-2145` (unwrap_ok ok-block payload extract — hardcoded `result_ty=tir.TIRScalar("i32")`).
- `helixc/backend/x86_64.py:2199-2214` (RESULT_PACK uses `mov_ecx_mem_rbp` which is a 4-byte load regardless of payload's declared type).
- `helixc/frontend/typecheck.py:4540-4561` (Ok / Err typecheck arms accept ANY inner type — no `_compatible(arg_tys[0], TyPrim("i32"))` check).

**Defect class:** spec-vs-implementation drift. The Inc 1
convention block at `tir.py:299` is explicit: "Wider payloads
(Result<i64, ...>, Result<f64, ...>) remain out of scope until
Stage 50+; the i32 payload constraint is enforced via
constructor/accessor type arms below (and **still by the
typecheck arms that require T and E to be i32 for this stage**)."
The lowering arm's comment at `lower_ast.py:881-883` repeats:
"the i32 payload constraint is enforced via constructor/accessor
type arms below (and still by the typecheck arms that require T
and E to be i32 for this stage)."

**No such typecheck guard exists.** Grep for "i32" in
typecheck.py's Ok/Err/Result arms turns up only commentary and
test reproducers; no `_compatible(... TyPrim("i32"))` check.

**Minimal repro (static, no end-to-end run required — the
miscompile is visible in IR/asm):**

```helix
fn make() -> Result<i64, i32> {
    Ok(5_000_000_000i64)   // 0x12A05F200 — high bit lives at bit 32
}
fn main() -> i32 {
    let r: Result<i64, i32> = make();
    let v: i64 = unwrap_ok(r);
    (v >> 32) as i32        // intended: 0x1 (== 1); actual: 0
}
```

Trace:
1. Typecheck: `Ok(5_000_000_000i64)` returns
   `TyResult(ok_ty=TyPrim("i64"), err_ty=TyUnknown)`. Fn return
   type `Result<i64, i32>` is compatible. Clean.
2. `_lower_type(Result<i64, i32>)` returns
   `TIRScalar("i64")` — packed-i64 representation (correct for
   the packed-tag layout; misleading for a >32-bit payload).
3. Ok arm lowers: `payload = _lower_expr(5_000_000_000i64)` →
   a Value of type `i64` (8-byte slot). Then
   `self.builder.emit(RESULT_PACK, tag, payload,
   result_ty=TIRScalar("i64"))`.
4. Backend RESULT_PACK at `x86_64.py:2207-2214`:
   `self.asm.mov_ecx_mem_rbp(payload_slot)` — this is the 4-byte
   `mov ecx, dword ptr [rbp+disp8]` (encoding 0x8B 0x4D ..., per
   `asm:209-214`). **It reads only the low 4 bytes of the 8-byte
   payload slot. The high 4 bytes — including the `0x1` that
   represents 4-billion-and-change — are discarded.**
5. The 4-billion-high bits become the tag (which was 0 for Ok);
   pre-OR mask intersects with payload's truncated low half. The
   resulting packed value is `0x00000000_2A05F200` — Ok(733_517_824),
   NOT Ok(5_000_000_000).
6. `unwrap_ok` arm at `lower_ast.py:2143` always emits
   `result_ty=TIRScalar("i32")`. Even though typecheck says
   unwrap_ok returns `inner.ok_ty == TyPrim("i64")`, the IR
   produces an i32 — a 4-byte store into an 8-byte i64 slot.
   The high 4 bytes are stale (likely 0 from prior `mov rax`
   stores, but no guarantee).

**Hidden errors:**
- Any `Result<i64, ...>`, `Result<u64, ...>`, `Result<f64, ...>`,
  `Result<isize, ...>`, `Result<usize, ...>` silently truncates.
- Symmetric truncation on the Err side: `Result<..., i64>` Err
  payload also truncated by RESULT_PACK's 4-byte payload load.
- Any pointer-typed Result inner (u64-shaped TyPtr): pointer
  truncated to its low 32 bits — segfault or wild-pointer write
  on dereference.
- Struct-typed inner (if struct-mono produces a TyStruct that
  resolves to a wider TIR type): same 4-byte truncation.

**User impact:** any user writing `Result<i64, ...>` /
`Result<f64, ...>` / `Result<*T, ...>` etc. gets silent data loss
on the constructor path AND silent truncation on the unwrap path.
The two truncations compound. The source code looks fine, the
typecheck passes (intentionally, per the F4 inference policy that
makes TyUnknown universally compatible), the tests pass (because
no Stage 49 test uses non-i32 inners), but every byte above the
low 32 vanishes. This is exactly the cycle-100/102/106/108 defect
family — silent width truncation at a layering seam — that the
backend already has 4+ commits fixing for u64 vs i64.

**Concrete fix sketch:** add a typecheck guard at the Ok/Err
arms in `typecheck.py:4540-4561` rejecting non-i32 inner types
for this stage. Minimal patch:

```python
if bn == "Ok":
    if len(arg_tys) != 1:
        self.errors.append(...)
        return TyUnknown(hint=bn)
    # Stage 49 Inc 1: backend RESULT_PACK reads payload as 4 bytes.
    # Reject wider payloads until Stage 50+ wires a width-aware
    # pack/unpack path.
    if not (isinstance(arg_tys[0], TyPrim)
            and arg_tys[0].name in ("i32", "u32", "bool")):
        self.errors.append(TypeError_(
            f"Ok() inner type must be i32-width in Stage 49 "
            f"(got {self._fmt(arg_tys[0])}); wider payloads are "
            f"Stage 50+ work",
            expr.span,
            hint="constrain the Ok payload to i32 / u32 / bool, "
            "or wait for the width-generic packed representation",
        ))
        return TyUnknown(hint=bn)
    return TyResult(ok_ty=arg_tys[0], err_ty=TyUnknown(hint="Err inferred"))
```

Mirror for Err. Also add a fail-loud assert in the backend
RESULT_PACK arm: `assert self._is_i32_type(op.operands[1].ty)`
so a future bypass of the typecheck (e.g., synthetic IR) loud-
fails rather than silently truncating.

Alternative (deeper fix): make `_lower_type(Result<T, E>)` return
the i64 representation only when both T and E fit in 32 bits; for
wider inners, return a TIRTuple or a wider representation, and
update RESULT_PACK / RESULT_PAYLOAD to dispatch on operand width.

**Severity rationale:** HIGH (85) because:
- Silent data loss on production code that uses standard Rust-
  idiomatic Result types over i64/f64/pointers.
- The two comment-strings in tir.py and lower_ast.py both LIE
  about what's enforced — high false-confidence vector.
- Trivial reproducer (5-line program).
- Confidence is 85, not 95, because the audit could not find a
  user-facing surface that currently triggers this (the dogfood
  programs all use Result<i32, i32>). It is a latent silent
  miscompile waiting for the first non-i32 Result user.
- Same defect family as cycle-100/102/106/108 width-truncation
  silent miscompiles that have been HIGH-fixed multiple times.

**Confidence:** 85.

---

### SF1-M2 — MEDIUM (confidence 82): `map_ok` / `map_err` IR SELECT lowering evaluates BOTH arms unconditionally; user side effects in `new_val` always fire

**Location:** `helixc/ir/lower_ast.py:2293-2322`
(`map_ok` / `map_err` lowering — `new_val = self._lower_expr(expr.args[1])`
followed by `SELECT` on a tag comparison).

**Defect class:** strict-eval semantics differ from idiomatic
Rust-style `map_ok(|x| compute(x))` closure semantics. In Helix,
`map_ok(r, expr)` takes a VALUE not a closure — both `r` and
`expr` are fully lowered into IR before the SELECT runs, so the
IR for `expr` is unconditionally executed at runtime even when
the SELECT picks the other arm.

**Minimal repro (IR-level, no execution needed):**

```helix
fn side_effecting() -> i32 {
    // suppose this logs to stderr or mutates a global
    99
}

fn main() -> i32 {
    let r: Result<i32, i32> = Err(7);    // <-- Err arm
    let r2 = map_ok(r, side_effecting()); // <-- side_effecting() runs!
    unwrap_err(r2)                        // returns 7 (Err passed through)
                                          // but side_effecting() already ran
}
```

The IR lowering at `lower_ast.py:2296-2297`:
```
r_packed = self._lower_expr(expr.args[0])  # lowers r
new_val = self._lower_expr(expr.args[1])   # lowers side_effecting() — emits CALL
```

Both `_lower_expr` calls emit ops into the CURRENT block
unconditionally. The subsequent SELECT only picks a value — it
does not skip side-effects of the unchosen arm. The backend
SELECT at `x86_64.py:2433-2491` confirms this: it loads BOTH
operand slots before the branch, then `je` chooses which load's
result is final. The branch is value-merging, not eval-gating.

**Hidden errors:**
- Logging side effects fire on the wrong arm (`map_ok(err_result,
  log_and_compute())`).
- Mutating side effects (`map_ok(err_result, counter_incr())`)
  run when they shouldn't.
- FFI calls (`map_ok(err_result, syscall_open(path))`) run
  unconditionally — open file descriptors that the user's mental
  model says only open on Ok.
- Panic'ing computations (`map_ok(ok_result, unwrap_ok(other))`)
  panic on the err arm of `r` even when `r` is Err — the user
  expected `unwrap_ok(other)` to only run after confirming `r`
  is Ok.

**User impact:** users coming from Rust / Swift / Haskell where
`map` over a sum type is closure-lazy will write code that
silently runs side effects on both arms. Diagnosis requires
either reading the lowering source or careful runtime tracing —
the source-level `map_ok(r, expr)` reads as if `expr` only
matters on the Ok arm.

**Severity rationale:** MEDIUM (82), not HIGH, because:
- It's a semantic surprise, not a wrong-arm crossover. The
  TAG-matching SELECT correctly picks the right packed value;
  unwrap_ok / unwrap_err on the result return the correct
  payload. The dogfood test pattern `unwrap_ok(map_ok(Ok(7),
  99)) == 99` works as expected.
- Stage 49 inherits the Helix design choice that `map_ok` /
  `map_err` take values not closures. With value-arg semantics,
  eager eval is the only sound choice.
- The defect is a documentation / language-design surprise more
  than a compiler-correctness defect. But it IS a silent
  behavioral difference from user expectations, hence MEDIUM.

**Concrete fix sketch:** (a) document the eager-eval semantics
in the Ok / Err / map_ok / map_err typecheck arm comments
explicitly; (b) optionally, add a typecheck warning when the
2nd arg of map_ok / map_err is a Call expression with at-least-
maybe-side-effects (i.e., not a pure literal or pure Name); (c)
long-term, accept a closure form `map_ok(r, |x| expr)` that the
lowering routes through a conditional-branch wrapper instead of
SELECT. Option (a) is the minimum closure deliverable for Stage
49.

**Confidence:** 82. The lowering is sound; the user-model
mismatch is the defect.

---

### SF1-M3 — MEDIUM (confidence 80): `__try` Err arm emits RETURN via raw opcode, bypassing every future hook on `A.Return`

**Location:** `helixc/ir/lower_ast.py:2186-2187` (same site as
SF1-F1, broader concern).

**Defect class:** SF1-F1 names TRACE_EXIT as the immediate
observable miss. The broader concern: the `A.Return` arm at
`lower_ast.py:3700-3735` is the canonical "explicit early return"
hook site, and it carries 35 lines of comments documenting that
it's the right insertion point for return-side cross-cutting
logic (TRACE_EXIT today, but presumably any future per-return
hook — destructor calls, ARC release, audit logging, RAII drop,
mut-borrow release tracking, etc.). The `__try` arm bypasses
this entire machinery, emitting `RETURN packed` as if it were
an internal codegen helper.

**Hidden errors (future-proofing concern):**
- Any future per-return hook added to `A.Return` will not fire
  for `__try` Err propagation.
- The IR pass invariant "every RETURN is preceded by the
  cross-cutting epilogue ops" is silently broken at this site.

**User impact:** zero today (TRACE_EXIT is the only such hook,
and the C2-2 fix only covers `A.Return`). HIGH-latent the
moment ANY new return-side hook lands.

**Concrete fix sketch:** factor the cross-cutting return logic
into a helper `_emit_return_with_epilogue(value)` and call it
from BOTH `A.Return` and the `__try` Err arm. Co-located with
the SF1-F1 fix.

**Severity rationale:** MEDIUM (80) for forward-defense. The
immediate observable miss is SF1-F1 (HIGH). This finding
generalizes that fix.

**Confidence:** 80.

---

### SF1-M4 — MEDIUM (confidence 78): typecheck does not propagate provenance for `let r = some_fn()` even though `some_fn` returns `Result<T, E>` with statically-known constructor

**Location:** `helixc/frontend/typecheck.py:2704-2724` (the
opaque-RHS pop branch in the Let-stmt provenance handler).

**Defect class:** asymmetric coverage. The Let-stmt provenance
writer at lines 2669-2703 handles two cases: (a) direct
constructor `let r = Ok(7)`, (b) map_ok / map_err composition
`let r = map_ok(r0, ...)`. Any other RHS — including a CALL to
a fn that statically only returns Ok (or only Err) — pops the
provenance.

Stage 49 Inc 4 makes `?` propagation real, so the user can now
write:

```helix
fn always_ok() -> Result<i32, i32> { Ok(42) }
fn caller() -> i32 {
    let r: Result<i32, i32> = always_ok();
    unwrap_err(r)  // accepts! prov dict popped because RHS was a Call
                   // not directly Ok/Err. Runtime: TRAP fires (Inc 1.5)
                   // — but the typecheck-time diagnostic is missed.
}
```

The static-provenance reject at `typecheck.py:4602-4625` would
fire if the prov dict carried "ok" for `r`, but the opaque-RHS
pop at line 2723 cleared it. Inc 1.5's runtime trap catches the
runtime case (good!), but a typecheck-time diagnostic would be
strictly better UX.

**Hidden errors:** typecheck accepts a guaranteed-runtime-panic
program when the panic-causing call's variant is statically
derivable.

**User impact:** users get a runtime panic with the Inc 1.5
deterministic error message instead of a compile-time error.
This is strictly better than silently miscompiling (pre-Inc-1.5
behavior), but worse than the gate-3 G3-F2 map_ok / map_err
provenance-propagation experience for direct-constructor sources.

**Severity rationale:** MEDIUM (78), not HIGH, because:
- Inc 1.5 runtime trap is the safety net (silent miscompile is
  prevented).
- Static-derivation of fn return provenance requires a small
  intraprocedural analysis (scan fn body for a single-Ok / single-
  Err return shape) that did not previously exist.
- Real cost: missed early-diagnosis on a common Phase-0 pattern
  (helper fn returning `Ok(...)` unconditionally).

**Concrete fix sketch:** add a `_fn_constructor_return_provenance`
dict alongside `_result_constructor_provenance`, populated when
typechecking a fn body that has exactly one RETURN site (or all
RETURN sites with the same Ok/Err shape). Let-RHS-call against a
fn in this dict copies the provenance. Stage 50+ scope —
documented here so the Inc 4 typecheck arm comment doesn't claim
"the rest" is sound.

**Confidence:** 78.

---

### SF1-L4 — LOW (confidence 85): `_check_block` and `_check_expr_in_block_scope` double-append before try, leak on (unrealistic) exception

**Location:**
- `helixc/frontend/typecheck.py:2475-2478` (`_check_block` —
  `saved_provenance = dict(...)` then `_result_let_block_scopes.append(set())`
  then `_result_assigns_block_scopes.append(set())` then `try:`).
- `helixc/frontend/typecheck.py:2581-2584`
  (`_check_expr_in_block_scope` — same pattern).

**Defect class:** parity with gate-4 G4-L4. The new
`_result_assigns_block_scopes.append(set())` introduced at gate-5
G4-F2 adds a SECOND pre-try .append. If the first .append
succeeds but the second raises (`MemoryError` only realistic
scenario), the first stack carries a leaked frame into subsequent
processing. The dict snapshot `saved_provenance = dict(...)` could
also raise (also MemoryError-only).

**Hidden errors:** none in practice — `set().append()` doesn't
raise outside catastrophic memory exhaustion. Discipline parity
note only.

**User impact:** nil.

**Concrete fix sketch:** move the snapshot + both appends inside
the try, with corresponding cleanup in finally guarded on
whether each push happened. Or accept the catastrophic-only
hazard and move on. The Stage 48 G4-L4 / G4-L5 disposition was
"acceptable as discipline-parity LOW"; same here.

**Confidence:** 85 (hazard is real but unreachable in practice).

---

## Patterns confirmed clean

| Pattern                                                  | Verification |
|----------------------------------------------------------|--------------|
| `Ok(v)` static-Ok round-trip                              | `test_stage49_inc1_ok_round_trip_exits_42` passes; pack-then-unpack value-identical for i32 |
| dogfood_17 exit 42                                        | `test_stage49_inc1_dogfood_17_still_exits_42` + `test_stage49_inc4_dogfood_17_still_exits_42` |
| Result-returning fn returns packed-i64 in rax              | `_is_64bit_int_type` arm in RETURN (line 2830) routes via `mov rax, [rbp+slot]` |
| CALL receives packed-i64 in rax                            | `_is_64bit_int_type` arm in CALL return (line 2669) routes via `mov [rbp+slot], rax` |
| Result-typed param spill (callee side)                     | `_is_64bit_int_type` arm in param-spill (line 1148) uses INT_SPILLS_64 (full 8-byte) |
| Result-typed arg pass (caller side)                        | `_is_64bit_int_type` arm in CALL arg shuffle (line 2640) uses INT_REGS_64 (full 8-byte) |
| Tag-bit cleanly distinguishes Ok(0) from Err(0)            | static trace through pack: Ok(0) = 0x0000000000000000; Err(0) = 0x0000000100000000 |
| Negative i32 payload preserved                             | `mov ecx, [rbp+slot]` zero-extends → `or rax, rcx` preserves; payload-extract `mov eax, [rbp+slot]` preserves low-32 bits incl. sign bit |
| `__try` allocates distinct err/ok blocks per occurrence    | `append_block` called twice per `__try`; no aliasing |
| `__try` Err arm RETURN type-matches enclosing fn return ty | typecheck guard at `typecheck.py:4720-4732` requires enclosing fn return type to be Result<U, E2>; lowering uses `RETURN packed` (i64); RETURN's `_is_64bit_int_type` arm handles i64 |
| `is_ok` / `is_err` runtime vs static consistency           | typecheck returns generic TyPrim("bool"); IR lowers to RESULT_TAG + CMP_EQ; no fold, so runtime is single source of truth |
| `map_ok` / `map_err` tag bit preserved                     | SELECT picks between new_packed (tag=expected) and r_packed (untouched original); both packed-i64 paths through `_is_64bit_int_type` |
| `map_ok` / `map_err` Ok-side / Err-side passthrough        | `test_stage49_inc3_map_*_preserves_*_status` verifies tag preserved through SELECT |
| Inc 1.5 wrong-arm runtime trap on unwrap_ok / unwrap_err   | `test_stage49_inc1_5_unwrap_*_panics` cases verify non-zero exit + TRAP message |
| Static-provenance reject for `unwrap_ok(Err-named)` etc.   | typecheck.py:4602-4625 reject path retained from Stage 46 |
| Gate-5 G4-F2 ASSIGN-then-LET-shadow mitigation             | dual let/assigns stacks with per-event mask at restore — sound on inspection; gate-5 carried forward intact into Stage 49 |
| `_check_expr_in_block_scope` wraps match-arm body / guard / if-else expr-form arm | typecheck.py:5020 + 5039 + 5055 — three sites wrapped per G4-F1 / G4-M3 |
| `__try` inside If/Match arm (control-flow nesting)         | ok_blk becomes current_block; outer BR fires from ok_blk; err_blk's RETURN terminates that arm cleanly |
| Multiple `?` in one expression                             | each `?` allocates own err/ok pair (gate-1 type-design L1 polish, lowering.py:2156-2163) |

---

## Verdict

**NOT CLEAN — 2 HIGH findings.**

Gate-1 of Stage 49 found:

- **SF1-F1 (HIGH 90)**: `__try` Err arm misses TRACE_EXIT for
  `@trace` fns — same defect class as the existing C2-2 fix for
  `A.Return`. Silent today (backend stubs trace ops); becomes
  buffer corruption the moment Stage 30 runtime lands.
- **SF1-F2 (HIGH 85)**: no typecheck guard enforces the i32-only
  payload constraint that Inc 1's design explicitly assumes.
  Wider Result inner types (`Result<i64, ...>`, `Result<u64,
  ...>`, `Result<f64, ...>`, pointer-typed inners) silently
  truncate at RESULT_PACK / RESULT_PAYLOAD. Cycle-100/102/106/108
  width-truncation family resurfaced at a new layering seam.
- **SF1-M2 (MEDIUM 82)**: `map_ok` / `map_err` eager-eval both
  arms (IR SELECT semantics) — user code with side-effecting
  `new_val` runs unconditionally. Documentation / design surprise,
  not a wrong-arm crossover.
- **SF1-M3 (MEDIUM 80)**: `__try` Err arm bypasses every future
  hook on `A.Return`. SF1-F1 generalized; refactor request.
- **SF1-M4 (MEDIUM 78)**: missed static-provenance propagation
  for opaque fn calls returning statically-known Ok / Err shape.
  Runtime safety net (Inc 1.5 trap) catches it, but a typecheck-
  time diagnostic would be strictly better.
- **SF1-L4 (LOW 85)**: pre-try-leak parity with gate-4 G4-L4.
  Discipline-only.

Both HIGH findings have minimal-diff fix sketches above. SF1-F1
is a 3-line addition mirroring the existing C2-2 fix. SF1-F2 is
a 6-line typecheck guard plus a backend assert. Neither requires
architectural rework.

The cascading-defect rhythm continues: every Stage closure gate
finds a HIGH at a NEWLY-introduced layering seam. Stage 49 Inc 1
introduces packed-tag width as a new seam; Inc 4 introduces a
second early-return site as a new seam. Each surfaced a defect
of a class that was previously fixed elsewhere — the discipline
exists, the carry-forward to new sites was missed.

---

## Closure-protocol forward note

- SF1-F1 + SF1-F2 should land together as gate-2 (one commit each,
  or a fused commit with both regression tests). Both are
  minimal-diff.
- SF1-M2 should land as a Stage 50 design RFC item: either
  document the eager-eval semantics in user-facing language docs,
  or add a closure-form `map_ok(r, |x| expr)`. Not a Stage 49
  closure blocker.
- SF1-M4 is a Stage 50 typecheck-precision opportunity.
- SF1-L4 is acceptable as discipline-parity LOW (same disposition
  as Stage 48 G4-L4 / G4-L5).
- The Stage 48 G4-H1 type-design deferral (`Result<Known<...>, E>`
  fn-return-type position raises NotImplementedError) is still
  open; not in Stage 49's diff but flagged here for the gate-2
  scope decision.
