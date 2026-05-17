# Stage 49 Inc 5 closure gate-1 type-design audit

**HEAD**: `db26e1c` (Stage 49 closure gate-1: Inc 1.5 wrong-arm tag-check + type-design polish)
**Base**: `7eaba56` (Stage 48 CLOSED 2026-05-17 at Inc 4)
**Scope**: type-design lane of the Stage 49 closure gate-1 audit. Covers the Inc 1 (TIR opcodes) + Inc 2 (`is_*`) + Inc 3 (`map_ok`/`map_err`) + Inc 4 (`?` early-return) + Inc 1.5 (runtime wrong-arm tag-check) surface as a single rolled-up delta against the Stage 48 closed baseline. Specific focus per audit brief: (1) tag-width design extensibility; (2) endianness/signedness symmetry; (3) Result-typed fn return ABI integrity; (4) wrapper-quintet × packed-Result interactions; (5) provenance + runtime-tag layering; (6) `is_ok`/`is_err` bool propagation; (7) `map_*` parametric correctness; (8) `?` return-type constraint enforcement; (9) catch-all conflations.
**Date**: 2026-05-17
**Method**: read-only. Code review + 11 live Python repros against the live compiler (`parse → typecheck → lower → compile_module_to_elf`). Probes cover composition (`Result<Result<...>>`, `Result<i64,i32>`), aggregate-of-Result (struct field, tuple element, array element), wrapper composition (`Known<Result<...>>`), runtime tag-check firing, `?` chaining, and ABI cross-check at call boundaries.

## VERDICT: NOT CLEAN — 2 CRITICAL, 1 HIGH, 1 MEDIUM-HIGH, 3 MEDIUM, 3 LOW

## Summary table

| ID | Sev | Conf | Title |
|----|-----|------|-------|
| TD1-C1 | CRITICAL | 98 | Silent payload truncation for non-i32 Ok/Err inners. `Result<i64, i32>`, `Result<f64, i32>`, and (worst) `Result<Result<i32,i32>, i32>` ALL typecheck-clean, IR-lower-clean, and backend-compile-clean — and silently discard the upper 32 bits of any wider-than-i32 payload at `RESULT_PACK` codegen. The Inc 1 documentation says the i32 constraint is "enforced via constructor/accessor type arms ... and still by the typecheck arms that require T and E to be i32" but no such enforcement exists in either layer. |
| TD1-C2 | CRITICAL | 95 | Nested Result silent miscompile. `Result<Result<T,E1>, E2>` typechecks AND lowers AND backend-compiles; the outer `Ok(inner_result)` emits `RESULT_PACK(0, packed_i64)` where the inner packed-i64 is silently truncated to its low 32 bits at codegen (the inner tag bit lives in the high 32 and is destroyed). Subsequent `is_err(unwrap_ok(outer))` reads garbage. Stage 49 newly enabled this surface by lifting the `is_*` typecheck reject — pre-Stage-49 the nested case was unreachable. |
| TD1-H1 | HIGH | 92 | Aggregate-of-Result asymmetry. `struct Holder { r: Result<i32, i32> }`, `[Result<i32,i32>; N]`, and `(Result<i32,i32>, i32)` ALL typecheck and IR-lower cleanly — the IR uses `ALLOC_ARRAY dtype=i64` and `STORE_ELEM`/`LOAD_ELEM` of the packed i64. The x86_64 backend's `_check_array_elem_size_supported` (cycle-16 C16-1) raises `NotImplementedError` on i64 element arrays. Same typecheck-clean → IR-clean → backend-raises asymmetry pattern as Stage 48 G4-H1, one layer deeper. Stage 49 introduced this exposure (pre-Stage-49 Result-in-struct lowered as i32-element). |
| TD1-MH1 | MEDIUM-HIGH | 85 | Static-provenance reject for `unwrap_ok` / `unwrap_err` (typecheck.py C1 consumer sites) is now layered ON TOP of the Inc 1.5 runtime tag-check, contradicting the explicit retirement plan in the comment at typecheck.py:644-654 ("when Inc 1.5 lands, delete the C1 static-prov reject"). Either the typecheck reject is now dead code that needs to be removed per plan, or the plan is stale and the comment needs to be updated. Either way, the documented invariant ("runtime tag covers it" → typecheck reject removed) is broken by the committed code shape. The `__try` consumer C2 was correctly lifted by Inc 4. The asymmetry is in `unwrap_*` only. |
| TD1-M1 | MEDIUM | 80 | Tag-width extensibility lock-in. The i64-packed encoding (tag in high 32, payload in low 32) embeds the i32-payload assumption into the SysV ABI: a future `Result<i64, i64>` cannot inflate the return value beyond rax without an ABI break (callers compiled against today's i64 return cannot consume a {rdx, rax} pair). The tir.py M2 comment block correctly reserves tag values 0/1 EXCLUSIVELY for Result and correctly names Option<T> as needing its own opcode family — but does NOT name the payload-width extensibility constraint, which is a separate axis. Stage 50+ work cannot extend payload width without a new opcode family (e.g. `RESULT_PACK_WIDE` with `{tag, ok_payload, err_payload}` slot triples). |
| TD1-M2 | MEDIUM | 78 | Stale identity-tuple back-reference comment in lower_ast.py:2263-2273 says `is_ok` / `is_err` / `map_err` "remain typecheck-rejected (Stage 46 F1/F2) until Inc 2/3 of Stage 49 wire their lowering". Inc 2 + Inc 3 ALREADY shipped (the dedicated arms above are live). The comment is misleading to future readers; if someone greps for the rejection text they will be directed here and find the comment but no actual rejection. |
| TD1-M3 | MEDIUM | 75 | Dead asm helper `mov_eax_eax` added in Inc 1 (x86_64.py:932-938) but never invoked by any compile_op arm. Either keep it with a comment naming the deferred site that will use it, or remove until needed. The RESULT_PAYLOAD arm uses `mov_eax_mem_rbp` + `mov_mem_rbp_eax` instead — `mov_eax_eax` was apparently a design alternative that wasn't taken. |
| TD1-L1 | LOW | 70 | `RESULT_PACK` opcode declared in tir.py:337 as `(tag i32, payload i32) -> packed i64`, but the opcode shape declaration is COMMENT-ONLY — there is no schema enforcement at IR construction. Any caller (including buggy future lowering arms) can pass an i64 payload via `builder.emit(OpKind.RESULT_PACK, tag, i64_value, result_ty=...)` and the IR will accept it. The backend then silently truncates. A small schema-checker pass that asserts operand types match the opcode contract at IR-validator time would catch TD1-C1, TD1-C2, and TD1-H1 at validate-time rather than backend-codegen-time. |
| TD1-L2 | LOW | 68 | The Inc 1.5 panic-block synthetic `BR` terminator (lower_ast.py:2129-2140) is defensively sound but uses a sentinel `const_int(0)` as the branch arg. The `BR` op convention elsewhere in the codebase uses zero-arg branches for non-block-param-passing branches. The sentinel produces a wasted `const.int` op that DCE may or may not eliminate. Cosmetic IR shape issue. |
| TD1-L3 | LOW | 65 | Tag-value reservation comment in tir.py:329-336 names Option<T> as a future family but does not name other natural discriminated-union families (Either<L,R>, Result3<A,B,C> for ternary, sum types from enum lowering). The reservation policy "tag 0 = Ok, tag 1 = Err exclusively for Result" is correct but the policy SCOPE is under-specified. A user-defined enum lowered to a tagged-union (Stage 50+) would face the same naming-collision risk. |

---

## TD1-C1: Silent payload truncation for non-i32 Ok/Err inners (CRITICAL, conf 98)

**Location**: `helixc/ir/lower_ast.py:865-884` (`_lower_type` Result arm) + `helixc/ir/lower_ast.py:2058-2077` (`Ok` / `Err` constructor lowering) + `helixc/backend/x86_64.py:2200-2217` (`RESULT_PACK` codegen) + `helixc/frontend/typecheck.py:4540-4561` (`Ok` / `Err` typecheck arms — NO i32 constraint).

**Observed structural issue**: the tir.py convention block at lines 302-339 documents:

> Phase-0 (Stages 46-48) lowered Result identity to its Ok inner with no runtime tag. Stage 49 introduces a 2-slot packed representation: a Result<T, E> at the IR level is a single i64 where ... the low 32 bits hold the payload (Ok-inner OR Err-inner, both currently constrained to i32 for Inc 1). Wider payloads are deferred to Stage 50+.

The comment at lower_ast.py:879-883 reinforces:

> the i32 payload constraint is enforced via constructor/accessor type arms below (and still by the typecheck arms that require T and E to be i32 for this stage).

**Neither layer enforces the constraint.** `_lower_type` (line 865-884) unconditionally returns `TIRScalar("i64")` for `Result<T, E>` regardless of T and E. The typecheck `Ok` / `Err` arms (line 4540-4561) accept ANY argument type via `TyResult(ok_ty=arg_tys[0], err_ty=TyUnknown(...))` — no width check, no kind check.

Live repro (run against HEAD `db26e1c`):

```python
src = """
fn make(x: i64) -> Result<i64, i32> { Ok(x) }
fn use_i64(r: Result<i64, i32>) -> i64 { unwrap_ok(r) }
fn main() -> i32 { 0 }
"""
typecheck(parse(src)) == []            # PASSES
lower(parse(src))                       # IR-lower OK
compile_module_to_elf(lower(parse(src))) # BACKEND OK, 4899 bytes
# At runtime, make(0x100000007) returns Ok(0x07) — top 32 bits silently dropped
```

The IR for `make` is:

```
v0:i64 = (param x)
v2:i32 = const.int(0)        # tag
v3:i64 = result.pack(v2, v0)  # operand v0 is i64, opcode says i32!
return(v3)
```

Backend codegen for `RESULT_PACK` at x86_64.py:2201-2216:

```python
self.asm.mov_eax_mem_rbp(tag_slot)     # 32-bit load of tag
self.asm.shl_rax_imm8(32)
self.asm.mov_ecx_mem_rbp(payload_slot) # 32-bit load — silently TRUNCATES i64 payload
self.asm.or_rax_rcx()
```

The `mov ecx, [rbp+payload_slot]` reads exactly 4 bytes, discarding the upper 32 of the i64 payload. `unwrap_ok` symmetrically reads only the low 32 from the packed slot as i32, then the fn's RETURN places i32-in-i64-slot — the top 32 of the caller's rax is whatever happened to be there. **Silent miscompile**.

**Why this is CRITICAL not HIGH**:

- The audit brief Section 1 explicitly asks: "i32 tag (4 bytes) + i32 payload (4 bytes) packed into i64. Is this design future-proof or does it lock out wider payloads? Out-of-scope per Stage 49 plan, but is the encoding extensible without breaking ABI?" The design IS lock-out (TD1-M1 covers extensibility). But the SAFETY question — "what happens TODAY if a user writes `Result<i64, i32>`" — is silent miscompile, not a typecheck reject.
- This is the worst kind of Phase-0 limit: not documented to the USER (no compiler diagnostic), not pinned by a `pytest.raises` test, not even loud-fail at IR or codegen. A user who reads the Stage 46 docs and sees "Result<T, E> is a runtime-tagged sum type" will reasonably try `Result<i64, i32>` and get wrong answers at runtime with no warning.
- Comparable to the cycle-16 C16-1 issue (LOAD_ELEM/STORE_ELEM silent-i64-truncation) that was promoted to HIGH with a `NotImplementedError` loud-fail at the backend. The Result analog has no loud-fail and no diagnostic.
- Stage 49 newly enabled this exposure: pre-Stage-49 Result was identity-lowered to its Ok-inner type, so `Result<i64, i32>` would have lowered to `i64` (and the Err side was unreachable since there was no runtime tag). Now Result has a definite shape (packed i64) that imposes an implicit i32 payload assumption — but the assumption is not policed.

**Recommended fix** (priority order):

1. *Tight, narrow, immediate*: add a typecheck arm at the `Ok` / `Err` construction sites (typecheck.py:4540-4561) and at the `Result<T, E>` resolution site (search `TyResult(ok_ty=...)`) that asserts both T and E are an i32-kind type. Emit a diagnostic naming the unsupported width and pointing at Stage 50+ for the wide-payload work. ~20 lines, no IR or backend change. Mirrors the C16-1 loud-fail discipline but at the typecheck layer (better UX — caught at the source line).
2. *Defense in depth*: also add a `_check_payload_i32` assert at the IR `_lower_type` Result arm (lower_ast.py:865) that raises `NotImplementedError` if either inner lowers to non-i32. Catches any future typecheck regression that lets a wide payload through.
3. *Loud-fail at backend*: extend `_check_array_elem_size_supported` style to a `_check_result_payload_width_supported` helper invoked at `RESULT_PACK` codegen. Lowest priority — by then the typecheck reject should have fired.

Recommendation: **fix 1 today, before closing Stage 49.** This is silent miscompile territory that does not fit the "deferred Phase-0 limit with a pin test" pattern (which requires the limit to be DIAGNOSED, not silent).

---

## TD1-C2: Nested Result silent miscompile (CRITICAL, conf 95)

**Location**: same surface as TD1-C1 (`_lower_type` + `Ok`/`Err` typecheck arms) but specifically the case where the Ok or Err inner is itself a `TyResult`.

**Observed structural issue**: `Result<Result<T, E1>, E2>` is the worst sub-case of TD1-C1 because the inner Result has a meaningful HIGH-32-bit tag value that gets silently destroyed by the outer pack.

Live repro:

```python
src = """
fn foo() -> Result<Result<i32, i32>, i32> {
    Ok(Err(99))         # inner Err — meaningful tag bit (1) in high 32
}
fn main() -> i32 {
    let r = foo();
    let inner = unwrap_ok(r);    # inner is i32 by RESULT_PAYLOAD's declared return
    if is_err(inner) { 1 } else { 0 }
}
"""
typecheck(parse(src)) == []                # PASSES
mod = lower(parse(src))                     # IR-lower OK
compile_module_to_elf(mod)                  # BACKEND OK
# Runtime answer: arbitrary (depends on adjacent stack bytes for the
# "tag" read of an i32-typed value). Logically:
#   - foo() emits Ok(Err(99)):
#       inner = result.pack(1_tag, 99) = 0x00000001_00000063 (i64)
#       outer = result.pack(0_tag, inner) — payload is i64 but RESULT_PACK
#               reads only low 32: 0x00000063. So outer = 0x00000000_00000063
#   - The inner Err's tag (bit 32) is LOST.
#   - unwrap_ok(outer) returns 0x63 = 99 (i32).
#   - is_err on that i32 reads RESULT_TAG via `mov rax, [v_slot]; shr rax, 32`
#     — reads 8 stack bytes from a 4-byte-typed slot, so the "high 32" is
#     whatever happened to live in the next stack slot.
```

**Why this is CRITICAL and a separate finding from TD1-C1**:

- TD1-C1 covers wide-scalar payloads (i64, f64). TD1-C2 covers composite payloads where the truncated bits CARRY SEMANTIC INFORMATION (the inner discriminator). A user testing `Result<i64, i32>` might notice their payload values are getting truncated. A user testing `Result<Result<...>, ...>` gets type-correct results everywhere but semantically nonsense answers — much harder to diagnose.
- Pre-Stage-49 this case was unreachable: `is_err` / `is_ok` were typecheck-rejected for ALL Result-typed operands. Stage 49 Inc 2 lifted the rejection broadly without distinguishing flat-Result from nested-Result.
- The Result-of-Result pattern is a natural error-aggregation pattern (`fn parse_then_eval() -> Result<Result<Value, EvalErr>, ParseErr>`) that real users WILL write. Phase-0 should reject this at typecheck rather than silently miscompile.

**Recommended fix**: same as TD1-C1 fix-1 (typecheck-side i32-kind constraint at `Ok` / `Err` construction). The fix subsumes TD1-C2.

If fix-1 is deemed too narrow (e.g. the team wants to allow `Result<Tensor<...>, i32>` for the AGI surface), then the constraint should be specifically "no TyResult, no TyFn, no TyTensor, no aggregate-of-anything" — anything that lowers to multi-slot or to a non-i32 scalar. That narrower constraint still covers TD1-C1 (which is about scalar widths) and TD1-C2 (about composite payloads).

---

## TD1-H1: Aggregate-of-Result × i64 element silent asymmetry (HIGH, conf 92)

**Location**: `helixc/ir/lower_ast.py:865-884` (`_lower_type` Result-to-i64) + `helixc/ir/lower_ast.py:1385-1422` (struct-lit Let arm + `ALLOC_ARRAY dtype=elem_ty`) + `helixc/backend/x86_64.py:1402-1422` (`_check_array_elem_size_supported`).

**Observed structural issue**: any aggregate (struct, tuple, array) containing a Result-typed field typechecks AND IR-lowers cleanly — the IR uses `ALLOC_ARRAY dtype=TIRScalar("i64")` because the field's lowered TIR type is i64 (per Stage 49's Result-arm rewrite). The x86_64 backend's `_check_array_elem_size_supported` (audit 28.8 cycle 16 C16-1) explicitly raises `NotImplementedError` on i64 array elements with the message "would silently truncate to 32 bits — see audit-stage28-8 cycle 16 C16-1". The aggregate-of-Result thus hits a HARD ERROR at backend codegen with a misleading diagnostic ("array elements" — the user's source code mentions no array).

Live repros (against HEAD `db26e1c`):

```python
# (a) struct-with-Result-field
src = """
struct Holder { r: Result<i32, i32> }
fn main() -> i32 {
    let h: Holder = Holder { r: Ok(42) };
    unwrap_ok(h.r)
}
"""
# typecheck: clean; lower: clean; compile_module_to_elf: NotImplementedError
#   "x86_64 backend LOAD_ELEM/STORE_ELEM does not yet support i64 array
#    elements ... see audit-stage28-8 cycle 16 C16-1"

# (b) array-of-Result
src = """
fn main() -> i32 {
    let xs: [Result<i32, i32>; 2] = [Ok(1), Ok(2)];
    unwrap_ok(xs[0])
}
"""
# Same outcome — typecheck clean, lower clean, backend raises.

# (c) tuple-containing-Result
src = """
fn main() -> i32 {
    let t: (Result<i32, i32>, i32) = (Ok(7), 99);
    unwrap_ok(t.0)
}
"""
# Different surface — tuple is stored as homogeneous array, so the
# i32 / i64 mix raises `TypeError: array literal element type mismatch
# after typecheck: first element is i64, later element is i32` at the
# tuple-lit Let arm BEFORE reaching codegen. Same defect class — typecheck
# accepts a shape that no later layer can handle.
```

**Why this is HIGH and not CRITICAL**:

- The user gets a HARD ERROR (NotImplementedError / TypeError), not silent miscompile. The error message is misleading (mentions "array elements" / "literal element type mismatch" rather than "Result-in-aggregate") but it IS visible.
- Stage 49 newly introduced this exposure: pre-Stage-49 the Result-in-struct field lowered to i32 (Phase-0 identity), backend was happy. Now Result lowers to i64, backend rejects.
- Mirrors the Stage 48 G4-H1 pattern (typecheck-clean → IR-clean → loud-fail one layer later) — accepted there as a deferred Phase-0 limit with a pinning test. Same disposition is reasonable here IF the limit is pinned.

**Recommended fix** (pick one):

1. *Narrow typecheck rejection*: extend the `_check_aggregate_field_supported` or equivalent (the typecheck-side struct/tuple/array field-type validator) to reject `TyResult` in aggregate-field position. Emit a diagnostic naming the unsupported composition and pointing at Stage 50+ for aggregate-of-Result lowering. ~10 lines.
2. *Pin the limit*: add a pinning test for each of (a)/(b)/(c) above asserting current `pytest.raises(NotImplementedError, match="...")` behaviour. Same discipline as Stage 48's G4-H1 fn-signature-position pin. ~30 lines of tests.
3. *Fix the backend*: extend the i64 LOAD_ELEM / STORE_ELEM path to handle 8-byte loads/stores. ~50 lines of codegen + tests. This is the only path that ACTUALLY enables aggregate-of-Result; the other two just make the rejection visible.

Recommendation: **fix 1 today** (typecheck rejection converts a misleading backend error into a clear source-level diagnostic). Fix 3 lands as part of the wider-payload work in Stage 50+ alongside TD1-C1's typecheck constraint relaxation.

---

## TD1-MH1: Static-provenance reject layered on top of Inc 1.5 runtime tag-check (MEDIUM-HIGH, conf 85)

**Location**: `helixc/frontend/typecheck.py:4595-4625` (`unwrap_ok` / `unwrap_err` C1 consumer static-prov reject) + `helixc/frontend/typecheck.py:638-654` (retirement-plan comment).

**Observed structural issue**: the typecheck stewardship comment at lines 638-654 explicitly states the retirement plan for the C1 consumer site:

> TODO(stage49-inc1.5): the SPECIFIC trigger for retirement is the runtime tag-check addition to unwrap_ok / unwrap_err (panic-on-wrong-arm). Once that lands, consumer site C1 becomes redundant (runtime tag covers it). C2 (__try) is already redundant post-Inc-4 — kept only for completeness; the typecheck reject was lifted in commit 47d8f66. ... Retirement plan: when Inc 1.5 lands, delete the C1 static-prov reject ...

Inc 1.5 landed in commit `db26e1c`. The runtime tag-check IS live at `lower_ast.py:2084-2145`. The C1 typecheck reject at `typecheck.py:4595-4625` was NOT deleted. The asymmetry with the `__try` consumer C2 (which WAS lifted in Inc 4 per the comment) is now stark: `__try` allows static-Err and propagates at runtime, but `unwrap_*` rejects static-Err at typecheck.

This is a structural-stewardship break in two ways:

1. *Plan-vs-code divergence*: the comment names a specific trigger ("when Inc 1.5 lands") and a specific action ("delete the C1 static-prov reject"). The trigger fired; the action didn't. Either the action is now overdue, or the plan needs to be updated to "keep C1 as a defense-in-depth quality-of-life diagnostic alongside the runtime check".
2. *Asymmetric eliminator policy*: post-Inc-1.5, `unwrap_ok(Err(7))` is a typecheck error (early, source-line diagnostic); `Err(7)?` is a typecheck-clean runtime propagation. Both are now sound at runtime. The asymmetry in user experience is defensible (eliminator vs propagator) but it is NOT what the stewardship comment said would happen.

**Why this is MEDIUM-HIGH and not HIGH**:

- No silent miscompile. Both paths produce correct runtime behaviour. The defect is purely the documented-invariant-vs-code drift.
- The retirement is a SAFE refactor — deleting C1 leaves the runtime check as the sole guard, which is exactly the "runtime tag obsoletes static-prov machinery" story the comment narrates.
- Keeping C1 is ALSO defensible (early diagnostic is friendlier than runtime panic). But the decision was apparently not made consciously — the gate-1 silent-failure cascade that added Inc 1.5 also added comment maintenance but did not execute the documented retirement plan.

**Recommended fix** (the team picks one):

A. *Execute the retirement plan*: delete the C1 reject at typecheck.py:4595-4625 (replace with a comment saying "runtime tag-check covers this; was the Phase-0 static-prov reject, removed per stage49-inc1.5 plan"). Update the stewardship comment at 638-654 to reflect that C1 is gone. Also collapse mutation sites 4-6 (the snapshot/restore/assigns-stack machinery) since the dict is no longer consumed by C1. C2 was already lifted, C3 (Assign-arm consult) is the only remaining consumer.

B. *Keep C1, update the comment*: revise the stewardship comment at 638-654 to say "retained as defense-in-depth quality-of-life diagnostic — earlier diagnosis than runtime panic; runtime tag is the soundness layer". This makes the kept-code intentional rather than an oversight.

Recommendation: **B today, A in Stage 50** (when more of the Phase-0 surface lifts and the dict has more consumer sites to collapse). The IMMEDIATE need is to make the kept-code state intentional; the wholesale collapse is a Stage-50+ refactor that needs its own gate cycle.

---

## TD1-M1: Tag-width / payload-width extensibility lock-in (MEDIUM, conf 80)

**Location**: `helixc/ir/tir.py:302-339` (RESULT_PACK convention block) + `helixc/backend/x86_64.py:2200-2237` (codegen).

**Observed structural issue**: the i64-packed-in-rax encoding choice is hardcoded into both the IR opcode signature (`RESULT_PACK(tag, payload) -> i64`) and the SysV ABI (Result-returning fn returns in rax). Extending to wider payloads (`Result<i64, i32>`, `Result<f64, i32>`, `Result<i64, i64>`) requires either:

- A `{rax: tag, rdx: payload}` split — ABI break, every caller must be re-emitted.
- A `{tag, payload}` struct-by-reference return — ABI break in a different direction.
- A new opcode family (e.g. `RESULT_PACK_WIDE` with payload-width metadata) — preserves caller ABI for narrow Results, but the type-level decision of which family to use must happen at typecheck time, requiring T/E width discrimination.

The tir.py M2 comment block at lines 329-336 correctly reserves tag values 0/1 EXCLUSIVELY for Result and correctly names Option<T> as needing its own opcode family. It does NOT name the payload-width axis as a separate extensibility constraint.

**Why this is MEDIUM and not just LOW polish**:

- The audit brief Section 1 explicitly asks about future-proofing. The current encoding is NOT future-proof for wider payloads.
- The asymmetry with the tag-value reservation is structural: tag-values are documented as "exclusively Result's, do not share with Option"; payload-width is undocumented, so a Stage 50 implementer might naively add a `_lower_type` arm that returns `TIRScalar("i128")` for wide Result without realizing the entire backend codegen pipeline assumes 8-byte returns.
- The fix is documentation, not code change. Low cost, prevents a future Stage-50 misstep.

**Recommended fix**: extend the tir.py M2 comment block (lines 329-336) with a parallel "Payload-width reservation policy" sub-block:

```
Payload-width reservation policy (Stage 49 gate-1 type-design M1):
the (tag i32, payload i32) -> i64 shape EMBEDS the i32 payload
assumption into the SysV ABI (return in rax). Wider payloads
(Result<i64, ...>, Result<f64, ...>) require either an ABI break
or a new opcode family (e.g. RESULT_PACK_WIDE with payload-width
metadata). Stage 50+ work. TD1-C1 + TD1-C2 cover the typecheck-
side enforcement of the current i32-only constraint.
```

Cost: 8 lines of comment. Zero code change.

---

## TD1-M2: Stale identity-tuple back-reference comment (MEDIUM, conf 78)

**Location**: `helixc/ir/lower_ast.py:2263-2273` (in the identity-call tuple comment).

**Observed structural issue**: the comment reads:

> Stage 46 Inc 1 — Result<T,E> constructors + value-preserving accessors USED to live here as identity-lowered ops. Stage 49 Inc 1 split them into their own arm above that emits real RESULT_PACK / RESULT_PAYLOAD IR with packed-i64 representation. `Ok`, `Err`, `unwrap_ok`, `unwrap_err`, and `__try` are all handled by that arm now. `is_ok` / `is_err` / `map_err` remain typecheck-rejected (Stage 46 F1/F2) until Inc 2/3 of Stage 49 wire their lowering.

The last sentence is stale: Inc 2 + Inc 3 ALREADY shipped (commits `2c3253c` and `0868eae`). The dedicated `is_ok` / `is_err` arm is at lower_ast.py:2201-2214, the `map_ok` / `map_err` arm is at lower_ast.py:2293-2322. The "typecheck-rejected" claim was true for Stage 48-and-earlier but is false post-Inc-2/3.

**Why this is MEDIUM and not LOW**:

- A future reader greps for the rejection text and is misdirected to this comment, which says "Inc 2/3 will lift them" — but Inc 2/3 ALREADY lifted them. The misdirection wastes audit cycles.
- The stewardship comment in typecheck.py:638-654 was updated to reflect Inc 4's `__try` lift (in commit `db26e1c`). The lower_ast.py comment was not similarly updated.

**Recommended fix**: replace the last sentence with:

```
`is_ok` / `is_err` / `map_ok` / `map_err` are also handled by
dedicated arms above (Stage 49 Inc 2 wired is_ok/is_err to
RESULT_TAG + CMP_EQ; Inc 3 wired map_ok/map_err to
RESULT_TAG-driven SELECT on packed-i64).
```

Cost: 4 lines of comment touch. Zero behavioural change.

---

## TD1-M3: Dead asm helper `mov_eax_eax` (MEDIUM, conf 75)

**Location**: `helixc/backend/x86_64.py:932-938` (helper added in Inc 1 commit `a08f21a`).

**Observed structural issue**: the `mov_eax_eax` helper was added in the Inc 1 commit with a comment explaining it's intended for "extract the low-32 payload from a packed Result i64". But the actual `RESULT_PAYLOAD` codegen at x86_64.py:2228-2237 uses `mov_eax_mem_rbp` + `mov_mem_rbp_eax` instead — loading from memory and storing back to memory, without going through `mov_eax_eax`. The helper is unreferenced.

`grep -n mov_eax_eax helixc/backend/x86_64.py` returns one match (the definition) and zero callers.

**Why this is MEDIUM and not LOW**:

- Dead code in a backend is a maintenance hazard — a future reader assumes the helper IS used somewhere and treats it as a stable interface. Removing it later requires verifying no plugin / no test references it.
- The Inc 1 commit message explicitly names the intended use site, but the use site was apparently refactored away during development. The orphan helper is the residue.

**Recommended fix** (pick one):

A. Remove `mov_eax_eax` (3 lines deleted). Lowest maintenance burden.
B. Use it in `RESULT_PAYLOAD` codegen instead of the memory-mediated mov pair. Saves one stack round-trip per `RESULT_PAYLOAD`. Performance polish, not correctness.

Recommendation: A unless someone measures B as a meaningful perf win.

---

## TD1-L1: `RESULT_PACK` opcode shape is comment-only, not schema-enforced (LOW, conf 70)

**Location**: `helixc/ir/tir.py:337-339` (opcode declarations).

**Observed structural issue**: the IR opcode contract for `RESULT_PACK` says `(tag i32, payload i32) -> packed i64`, but the contract is documented in a COMMENT block, not in a machine-checked schema. Any caller of `builder.emit(OpKind.RESULT_PACK, ...)` can pass i64 operands and the IR will accept them — which is exactly what the Stage 49 lowering DOES for `Ok(Err(...))` (per TD1-C2 repro: `result.pack(0, v2:i64)`).

A small IR-validator pass that walks the opcode-to-operand-type map and asserts shape conformance at validate-time would catch TD1-C1, TD1-C2, and the worst of TD1-H1 at IR-validate rather than backend-codegen — earlier diagnosis, better error messages.

**Why this is LOW**:

- TD1-C1 + TD1-C2 already cover the user-visible defect (silent miscompile). The IR-validator schema is a defense-in-depth that catches the COMPILER bug (wrong lowering) rather than the USER bug.
- Helix has no general IR-validator pass today. Adding one for just RESULT_* would be inconsistent with the rest of the IR (ARENA_PUSH_PAIR etc. also have comment-only contracts).
- Best landed as part of a wider IR-validator pass in a future stage, not as a Stage 49 closure item.

**Recommended fix**: defer. Track as a stage-50+ "tighten IR contracts" item. If the validator lands, add a `RESULT_PACK` shape entry; until then, TD1-C1 fix at the typecheck layer is the binding constraint.

---

## TD1-L2: Inc 1.5 panic-block synthetic `BR` uses a sentinel arg (LOW, conf 68)

**Location**: `helixc/ir/lower_ast.py:2129-2140` (Inc 1.5 wrong-arm panic block).

**Observed structural issue**: the panic-block terminator is:

```python
sentinel_zero = self.builder.const_int(0, "i32")
self.builder.emit(
    tir.OpKind.BR, sentinel_zero,
    attrs={"target_block": ok_blk.id})
```

The `BR` opcode elsewhere in the codebase is used with zero operands (unconditional branch with no block-param passing) when the target block has no parameters. `ok_blk` here has no parameters (it's just the payload-extract block). The sentinel `const_int(0)` produces an op that DCE may or may not eliminate, and conceptually confuses the BR's "branch arguments" semantics — there is no parameter receiving this 0.

The comment at lines 2129-2136 explains the synthetic BR is there to give the panic block a well-formed terminator (since TRAP performs sys_exit). The synthetic IS sound; the sentinel arg is cosmetic noise.

**Why this is LOW**:

- No correctness impact (the TRAP doesn't return at runtime, so the BR is unreachable).
- Cosmetic / IR-shape quality issue.

**Recommended fix**: emit `tir.OpKind.BR` with no operands (matches other zero-arg BR sites). Drop the `sentinel_zero` const. 3 lines cleaner.

---

## TD1-L3: Tag-value reservation comment under-specifies the policy SCOPE (LOW, conf 65)

**Location**: `helixc/ir/tir.py:329-336` (M2 polish comment from Inc 1.5).

**Observed structural issue**: the comment names Option<T> as a future family that MUST have its own opcode family. But discriminated-union families are open-ended: Either<L, R>, Result3<A, B, C> (ternary), user-defined enum-with-payloads (Stage 50+ if the enum-lowering pipeline shares an arena-tagged discriminator with Result), even pattern-matched-once-per-arm switch lowerings could collide if they reuse `RESULT_TAG` as a "low-cost tag-extract" primitive.

The reservation says "DO NOT reuse RESULT_TAG to query an Option discriminator". The implied generalization is "DO NOT reuse RESULT_TAG for ANY non-Result discriminator". The implication is correct but not stated.

**Why this is LOW**:

- Stage 50+ work; no immediate impact.
- A Stage-50 implementer reading the comment will plausibly extrapolate the policy to other families. Risk of misextrapolation is low.

**Recommended fix**: append one sentence:

```
This policy generalizes: ANY new discriminated-union family
(Option<T>, Either<L, R>, user-defined enum-with-payloads, etc.)
MUST get its own opcode family. RESULT_PACK / RESULT_TAG /
RESULT_PAYLOAD are reserved for Result<T, E> exclusively.
```

Cost: 4 lines of comment.

---

## Items audited and ruled CLEAN

- **`is_ok` / `is_err` return type propagation** (audit point 6): the typecheck arm at typecheck.py:4793-4816 returns `TyPrim("bool")`. Downstream `if is_ok(r) { ... }` works correctly; the cond is bool, the if-result is the join of arms. IR-level the lowering emits `cmp.eq` with `result_ty=TIRScalar("bool")` (matches the bool convention). Verified by trace + live repro.

- **`?` operator return-type constraint** (audit point 8): the `__try` typecheck arm at typecheck.py:4720-4732 strictly requires `self._current_return_ty` to be `TyResult`. Live repro `fn main() -> i32 { helper()?; ... }` cleanly emits the diagnostic: `` `?` used in function 'main' whose return type is i32, not Result<T, E> ``. The Err-type compatibility check at lines 4743-4755 also fires correctly. No bypass.

- **Result-typed fn return ABI at typecheck cross-checks** (audit point 3): `_compatible(TyResult, TyPrim("i64"))` returns False via the `TyResult or TyResult` arm at typecheck.py:8400. Live repros confirm:
  - `let r: i64 = make_result();` → typecheck error "declared i64 but value is Result<i32, i32>"
  - `fn other(x: i64); other(make_result());` → typecheck error "expects i64, got Result<i32, i32>"
  The TIR-level i64 representation does NOT silently match the surface i64 type. ABI cross-check is sound at the typecheck layer.

- **Endianness / signedness symmetry** (audit point 2): pack `(tag << 32) | (payload & 0xFFFFFFFF)` is bit-pattern symmetric with unpack `tag = packed >> 32 (logical)`, `payload = packed & 0xFFFFFFFF`. For tag values 0/1 (the reserved range) signed-vs-unsigned shift is observably equivalent. For hypothetical tag = -1 (high bit set in i32), the `mov eax, [tag_slot]` zero-extends rax, then `shl rax, 32` puts the i32 bits in the high half; `shr rax, 32` (unsigned) recovers the bit pattern unchanged. Bit-pattern round-trip is preserved. The reservation policy (TD1-L3) keeps Option-style negative tags from sharing the space. Sound.

- **Provenance + runtime-tag layering correctness** (audit point 5): the IR lowering at lower_ast.py:2080-2145 (unwrap_ok/unwrap_err post-Inc-1.5) ALWAYS emits the RESULT_TAG + CMP_NE + COND_BR + TRAP sequence regardless of typecheck-side static-provenance verdict. If a malicious test injects a runtime Err where typecheck-provenance said Ok, the runtime check FIRES. Verified by `test_stage49_inc1_5_unwrap_ok_on_dynamic_err_panics` and friends in test_stage49_runtime_tag.py. The static-prov layer is now redundant-with-runtime but not bypassed — the runtime is the sole soundness layer.

- **`map_ok` / `map_err` parametric correctness** (audit point 7): the typecheck arms at typecheck.py:4817-4865 correctly bind the new ok-type / err-type from the second argument's value type:
  - `map_ok(r: Result<T, E>, new_v: U) -> Result<U, E>`
  - `map_err(r: Result<T, E>, new_e: F) -> Result<T, F>`
  Note: the second argument is a VALUE not a function (despite the `map_*` name suggesting Rust-style closure). This is a documented Phase-0 surface decision; the name is slightly misleading but is consistent with the rest of the codebase and the test pinning. NOT a defect.

- **Catch-all `RESULT_TAG`-style catch-alls** (audit point 9): the new `is_ok`/`is_err` arm uses a small `expr.callee.name in ("is_ok", "is_err")` tuple that's narrow and well-bounded. The `map_ok`/`map_err` arm similarly. The big identity-tuple at lower_ast.py:2215-2274 was correctly NARROWED by Stage 49 — `Ok`/`Err`/`unwrap_ok`/`unwrap_err`/`__try` were REMOVED from the tuple and given dedicated arms above. No conflation. (The stale comment at lines 2263-2273 is TD1-M2; the actual TUPLE is clean.)

- **Inc 4 `?` chained-call err-block construction**: each `?` allocates a fresh err_blk + ok_blk pair, so `f()? + g()?` produces 4 fresh blocks (no aliasing). Live repro confirms IR shape. The L1 polish comment at lower_ast.py:2156-2163 is accurate.

---

## Cross-gate summary

This is the first type-design gate for Stage 49. There is no prior Stage 49 type-design audit to compare against.

Comparison to Stage 48 closure cascade:
- Stage 48 gate-1: CLEAN (1 audit, 0 HIGH+).
- Stage 48 gate-2: 1 HIGH (deferred) + 4 MEDIUM.
- Stage 48 gate-3: 0 HIGH, 3 MEDIUM, 2 LOW.
- Stage 48 gate-4: 1 HIGH (G4-H1 composition break), 1 MEDIUM-HIGH, 2 MEDIUM, 2 LOW.
- Stage 48 gate-5: closure cascade fixed G4-H1 via narrowing.
- **Stage 49 gate-1 (this audit): 2 CRITICAL, 1 HIGH, 1 MEDIUM-HIGH, 3 MEDIUM, 3 LOW.**

Stage 49's gate-1 finding density (2 CRITICAL silent miscompiles) is markedly higher than Stage 48's gate-1 (CLEAN). The two CRITICAL findings are NEW defect classes introduced by Stage 49's packed-i64 representation: the prior identity-lowering happened to be safe for non-i32 payloads (it lowered Result<i64,...> to i64 directly), while the packed encoding adds a width-truncation point that no layer polices.

The remaining HIGH (TD1-H1) is a Stage-48-G4-H1-pattern asymmetry one layer deeper (typecheck-clean → IR-clean → BACKEND-raises, vs G4-H1's typecheck-clean → IR-raises). The Stage 48 gate-4 cascade recognized this pattern and added pinning tests for the G4-H1 case — Stage 49 should adopt the same discipline for the aggregate-of-Result cases.

---

## Recommended gate-1 closure action

VERDICT: **NOT CLEAN — 4 HIGH+ findings (2 CRITICAL, 1 HIGH, 1 MEDIUM-HIGH).**

Gate-1 closure should:

1. **TD1-C1 + TD1-C2 (CRITICAL): fix before closing Stage 49.** Add the typecheck-side i32-payload constraint at `Ok`/`Err` construction arms. These are silent miscompiles; the F5/G4-H1 "pin a Phase-0 limit with pytest.raises" pattern does not apply because the limit is currently SILENT (no exception raised). Once a typecheck reject is in place, pinning tests can be added.

2. **TD1-H1 (HIGH): pick fix 1 (narrow typecheck rejection)** today. Aggregate-of-Result becomes a typecheck diagnostic, not a misleading backend error. Adds 1 pinning test per aggregate flavor (struct, tuple, array).

3. **TD1-MH1 (MEDIUM-HIGH): pick fix B (intentional kept-code, comment update)** today. Decide the C1 retirement timing explicitly rather than letting the plan-vs-code drift accumulate.

4. **TD1-M1 / TD1-M2 / TD1-M3 (MEDIUM)**: apply inline. ~25 lines total (8 + 4 + 3-line-delete + a handful of cross-references). All zero-behaviour-change.

5. **TD1-L1 (LOW)**: defer to Stage 50+ IR-validator work.

6. **TD1-L2 / TD1-L3 (LOW)**: apply inline. ~7 lines total.

Total expected gate-1 closure delta: ~60 lines of code + ~20 lines of tests. The TD1-C1 / TD1-C2 typecheck constraint is the load-bearing fix; everything else is stewardship and asymmetry polish.

VERDICT: NOT CLEAN — 4 HIGH+ findings
