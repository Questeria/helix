# Stage 4 Follow-up Audit: Silent-Corruption Sweep

**Date**: 2026-05-07
**Scope**: helixc/bootstrap/kovc.hx (read-only)
**Trigger**: post bf16 silent-corruption sweep + trap-id sweep — looking for
remaining silent windows the prior sweeps missed.

**Method**: traced every emit_ast_code arm for narrow-type / wide-type /
mixed-signedness fall-throughs; checked dispatch matrices for completeness;
inspected every place where store width is selected per-binding without a
matching value-type check.

**Result**: 8 findings (5 HIGH, 3 MEDIUM, 0 LOW). The bf16/trap-id sweeps
correctly cover binary arithmetic ops (ADD/SUB/MUL/DIV/MOD) and the comparison
binop body, but several non-arithmetic paths (NEG, SHL/SHR, ASSIGN, FN return,
CALL arity, VAR read) still have silent windows.

---

## Finding 1: AST_FN_DECL body-vs-ret-ty trap only checks 8-byte width class

**Location**: helixc/bootstrap/kovc.hx:4537-4544
**Severity**: HIGH
**Category**: silent-corruption

**Description**:
The post-body trap fires only when `body_is_8b != ret_wants_8b` (where 8b means
i64 or u64). Every other return-type / body-type mismatch escapes the trap:

- `fn f() -> u8 { some_i32 }`     — both 4-byte. No trap. Caller of u8-typed
  fn sees only `al`, but the convention is "low byte of eax is the u8 value".
  An i32 body that produces 257 returns u8 = 1; the language does not surface
  the truncation.
- `fn f() -> i32 { some_f32 }`    — bit pattern in eax interpreted as i32.
  Silent garbage.
- `fn f() -> f32 { some_i32 }`    — caller reads xmm0 / eax expecting f32.
  Silent garbage.
- `fn f() -> f64 { some_i32 }`    — both fail the 8b check (body_is_8b=0,
  ret_wants_8b=0). Caller reads f64; rax/xmm0 high half undefined. Silent.
- `fn f() -> bf16 { some_i32 }`   — same problem.
- `fn f() -> i32 { some_bf16 }`   — same problem.

The comment at line 4527-4530 acknowledges that "Full expr_type comparison
still produces false positives in the existing bootstrap source" — meaning
the trap was deliberately weakened to ship Stage 1. After the bf16 sweep
those false positives may now be tractable.

**Reproducer**:
```
fn f() -> u8 { 257 }
fn main() -> i32 {
    let x: u8 = f();   // expect 257 -> u8 truncates to 1, but the
                       // language pretends 257 was returned and the
                       // type system is silent about the loss.
    x as i32           // observable: 1, not 257 — but no diagnostic.
}
```

**Recommended fix**:
Replace the 8b/8b check with a full `expr_type(fn_body, ...) != fn_ret_ty`
check, and trap on any mismatch. If false positives exist in the current
bootstrap source, document them and fix the source; do not weaken the trap.

**Trap-id reservation**:
14002 (body type ≠ ret type, narrow / float / bf16). Reuse 14001 if a single
unified id is preferred; current 14001 only fires on 8b vs ≠8b.

---

## Finding 2: AST_NEG falls through to 32-bit `neg eax` for u64

**Location**: helixc/bootstrap/kovc.hx:3437-3465
**Severity**: HIGH
**Category**: silent-corruption

**Description**:
The dispatch checks `is_f64_expr`, `is_i64_expr`, `is_f32_expr`, `is_bf16_expr`,
then falls through to `emit_ast_neg_suffix()` (2 bytes: F7 D8 = `neg eax`).
**`is_u64_expr` is never checked.** For a u64 operand, the high 32 bits in
rax remain untouched while `neg eax` two's-complements only the low 32 bits.

Result: `neg(0x00000001_FFFFFFFF_u64)` produces `0x00000001_00000001` (low
flipped, high preserved) instead of either `0xFFFFFFFE_00000001` (proper
64-bit two's complement of (2^33 - 1)) or a trap. This mirrors exactly the
i64 pre-fix bug that motivated the i64 dispatch.

The Stage 1.5 batch added i64 dispatch but did not extend to u64; the
Stage 2.4b "u64 added" comments (e.g., AST_BNOT at line 3490-3502) were
applied to most binops but missed AST_NEG.

**Reproducer**:
```
fn main() -> u64 {
    let x: u64 = 4294967296_u64;   // 2^32 (high half = 1, low half = 0)
    let y = 0_u64 - x;             // semantically -x in u64 = 2^64 - 2^32
    y                              // expected: 0xFFFFFFFF_00000000
                                   // actual:   0xFFFFFFFF_00000000? — depends
}
```
A unary-minus reproducer requires `let y = -x` to parse; if the surface
disallows that, do `0_u64 - x` via AST_SUB which is properly dispatched.
Direct AST_NEG exposure: `let y = -1_u64;` if literals + neg work.

**Recommended fix**:
Add an `is_u64_expr` arm before the catch-all in the AST_NEG dispatch:
```
} else { if is_u64_expr(p1, bind_state, bn_state) == 1 {
    emit_neg_rax_64()           // REX.W neg rax, same as i64
} else {
    emit_ast_neg_suffix()
}}
```

**Trap-id reservation**:
N/A — fix is to dispatch to existing `emit_neg_rax_64()`. If u64 NEG should
not be permitted, reserve 9002.

---

## Finding 3: AST_VAR has no unbound-name guard

**Location**: helixc/bootstrap/kovc.hx:3941-3962
**Severity**: HIGH
**Category**: silent-corruption / safety

**Description**:
`bind_lookup` returns 0 when a name is unbound (sentinel; bind_alloc_offset
starts at 8, never returns 0). AST_VAR's codegen unconditionally emits a load
at `[rbp + 0]` for an unbound name — which is the saved rbp slot of the
current frame. The user's program silently loads the previous frame's rbp
value into eax/rax.

The comment at lines 875-883 claims AST_VAR has an "audit-10 guard" that
emits "the integer-zero placeholder", but the actual guard exists ONLY in
AST_ASSIGN at line 4061 (`if off == 0 { n_val }`). AST_VAR (line 3941-3962)
goes straight to `emit_mov_eax_local(off)` with no off-zero check.

This affects:
- typo'd variable names (`let foo = 5; bar + 1;` — `bar` is unbound, returns
  saved rbp).
- variables falling out of scope incorrectly.
- the `bind_push_typed` cap-overflow path documented at line 875-883 (which
  is the case the comment was actually written for).

**Reproducer**:
```
fn main() -> i32 {
    let foo = 42;
    bar              // unbound; emits mov eax, [rbp+0] = saved rbp
}
```
The output of this program is non-deterministic (depends on rbp). Self-host
of the bootstrap is unaffected because the parser only emits AST_VAR for
identifiers it has seen in let-bindings — but this is a defense-in-depth
issue; the language should not silently corrupt on undefined references.

**Recommended fix**:
Mirror AST_ASSIGN's audit-10 guard:
```
let off = bind_lookup(bind_state, p1, p2);
if off == 0 {
    emit_trap_with_id(1001)   // unbound AST_VAR read
} else {
    let ty = bind_lookup_type(bind_state, p1, p2);
    ... existing dispatch ...
}
```

**Trap-id reservation**:
1001 (AST_VAR tag = 1, sub_id 001).

---

## Finding 4: AST_ASSIGN narrow-bind-ty ignores value type — silent truncation/garbage write

**Location**: helixc/bootstrap/kovc.hx:4060-4108
**Severity**: HIGH
**Category**: silent-corruption

**Description**:
The trap matrix has these arms:
- `val_i64=1, bind_ty=3` — store 8B (correct path)
- `val_i64=1, bind_ty≠3` — trap 8001
- `val_u64=1, bind_ty=9` — store 8B (correct path)
- `val_u64=1, bind_ty≠9` — trap 8002
- `bind_ty=2` (f64) — store 8B unconditionally (no value-type check)
- `bind_ty=3, val=i32` — trap 8003
- `bind_ty=9, val=i32` — trap 8004
- `bind_ty∈{7,8,10,11}` (narrow) — store 1B/2B unconditionally (no value-type check)
- else — store 4B unconditionally

Cases that escape the trap:
1. **f64 binding, f32 / i32 / u32 / u8 / u16 / i8 / i16 / bf16 value**:
   Falls through to `emit_mov_local_rax_64(off)` which writes 8 bytes. The
   high 32 bits of rax are whatever was last there (for an f32/i32 value
   in eax, the high bits are zero from the implicit zero-extension; for
   bf16-bits-in-eax, also zero). Net effect: f64 slot holds 0x00000000_BITS,
   which is the f32 bit pattern interpreted as a denormal f64 — silent
   garbage at math time.
2. **u8 binding, f32 / f64 / bf16 / u32 value**:
   Falls through to `emit_mov_local_al(off)` which stores the low byte of
   eax. For an f32 in eax, that's the low byte of the bit pattern — silent
   garbage that happens to be a valid u8.
3. **u16 / i8 / i16 binding, f32 / f64 / bf16 / u32 value**:
   Same — silent truncation of bit pattern.
4. **i32 / u32 / f32 binding, f64 value**:
   Falls through to `emit_mov_local_eax(off)` which stores low 32 bits. The
   user wrote an f64; the i32/f32 slot holds the low 32 of the f64 bit pattern.
   Silent garbage.
5. **f32 / u32 binding, bf16 value**:
   Falls through to 4B store; bf16 bits ARE i32-shaped so this produces a
   bit pattern whose semantics depend on the target type. Subtle — depending
   on whether bf16-bit-pattern-as-f32 is intentional, this could be benign
   or a bug.

**Reproducer**:
```
fn main() -> i32 {
    let mut x: u8 = 0;
    x = 1.5_f32;     // bind_ty=7 (u8), val_ty=1 (f32). No trap.
                     // emit_mov_local_al(off) stores low byte of f32 bits
                     // (1.5_f32 = 0x3FC00000, low byte = 0x00).
    x as i32         // = 0
}
```

**Recommended fix**:
Extend the trap matrix to compare val_ty (via expr_type) against bind_ty
for ALL bind_ty arms, not just bind_ty in {3, 9}:
```
let val_ty = expr_type(p3, bind_state, bn_state);
let mismatch = if val_ty == bind_ty { 0 } else { 1 };
let n_store = if mismatch == 1 {
    emit_trap_with_id(8005 + bind_ty)   // distinct id per bind_ty arm
} else { ... existing width-correct store ... };
```

**Trap-id reservation**:
8005 (val/bind type mismatch — generic). Optionally 8005..8016 for per-bind-ty
distinct ids (8005 = i32 bind, 8006 = f32 bind, 8007 = f64 bind, 8009 = i64,
8010 = u8, 8011 = u16, 8012 = u32, 8013 = u64, 8014 = i8, 8015 = i16,
8016 = bf16). The current 8001-8004 cover only the val=i64 / val=u64 pairs.

---

## Finding 5: AST_SHL / AST_SHR don't dispatch u64 / f32 / f64 / bf16

**Location**: helixc/bootstrap/kovc.hx:3598-3619
**Severity**: HIGH
**Category**: silent-corruption

**Description**:
Both AST_SHL and AST_SHR check only `is_i64_expr(p1, ...)` to choose between
`emit_shl_rax_cl_64()` (REX.W shl) and `emit_shl_eax_cl()` (32-bit shl).

Cases that fall through to the 32-bit shift silently:
1. **u64 << i32**: `is_i64_expr(u64) = 0`. Falls to 32-bit shl. The high 32
   bits of the u64 are NOT shifted. `0x00000001_FFFFFFFF_u64 << 1` should be
   `0x00000003_FFFFFFFE` but produces `0x00000001_FFFFFFFE` (high preserved).
2. **f32 / f64 / bf16 << i32**: bit pattern shifted as integer. Almost
   certainly a user error, but no trap.

**Reproducer**:
```
fn main() -> u64 {
    let x: u64 = 4294967295_u64;     // 0x00000000_FFFFFFFF
    x << 1_i32                        // expect 0x00000001_FFFFFFFE
                                      // actual 0x00000000_FFFFFFFE (low only)
}
```

**Recommended fix**:
Add `is_u64_expr` to the wide-shift arm; trap on float/bf16:
```
let l_i64 = is_i64_expr(p1, bind_state, bn_state);
let l_u64 = is_u64_expr(p1, bind_state, bn_state);
let l_f32 = is_f32_expr(p1, bind_state, bn_state);
let l_f64 = is_f64_expr(p1, bind_state, bn_state);
let l_bf  = is_bf16_expr(p1, bind_state, bn_state);
let na = if l_f32 == 1 { emit_trap_with_id(32040) }
        else { if l_f64 == 1 { emit_trap_with_id(32010) }
        else { if l_bf == 1 { emit_trap_with_id(32001) }
        else { if l_i64 == 1 { emit_shl_rax_cl_64() }
        else { if l_u64 == 1 { emit_shl_rax_cl_64() }   // unsigned shl is bit-identical to signed shl
        else { emit_shl_eax_cl() }}}}};
```
For SHR, the unsigned dispatch should use `shr` (logical) not `sar`
(arithmetic) — currently AST_SHR uses sar for both signed and unsigned.
That is a separate signedness bug worth flagging:

**Sub-finding (HIGH/MEDIUM)**: AST_SHR at line 3618 always uses
`emit_sar_eax_cl()` / `emit_sar_rax_cl_64()` (arithmetic shift). For u32/u64
the correct semantics is logical (zero-fill). Falls through silently — high
bits sign-extend instead of zero-fill, producing wrong results for u32/u64
values >= 0x80000000.

**Trap-id reservation**:
- 32001 (AST_SHL bf16), 32010 (AST_SHL f64), 32040 (AST_SHL f32)
- 33001 (AST_SHR bf16), 33010 (AST_SHR f64), 33040 (AST_SHR f32)
- For the signedness sub-finding: a new pair `emit_shr_eax_cl()` /
  `emit_shr_rax_cl_64()` (logical shifts: D3 E8 / 48 D3 E8).

---

## Finding 6: AST_CALL arity-mismatch (arg_count != pp_count, both ≤ 6) not trapped

**Location**: helixc/bootstrap/kovc.hx:4151-4214
**Severity**: MEDIUM
**Category**: silent-corruption

**Description**:
The per-arg type check (line 4161-4168) runs only while `arg_count < pp_count`.
The arity check (line 4181) only fires when `arg_count > 6`. Cases that escape:

- **arg_count < pp_count** (caller passes fewer args than declared):
  e.g., `fn f(a: i32, b: i32, c: i32) -> i32 { ... }; f(1, 2)`. Pass 1
  emits 2 pushes. Pass 2 pops 2 values (rdi, rsi). Register rdx (arg 2 of f)
  is NEVER set — the callee reads garbage from rdx. **Silent corruption.**
- **arg_count > pp_count but ≤ 6** (caller passes more args than declared):
  e.g., `fn f(a: i32) -> i32 { ... }; f(1, 2, 3)`. Pass 1 pushes 3 values.
  Pass 2 pops 3 into rdi, rsi, rdx. Callee sees only rdi (= 1). Extra args
  wasted but no immediate corruption. Still warrants a diagnostic.

The Stage 1.7 comment block (line 4141-4145) acknowledges that builtins skip
the per-arg check (pp_count=0 for them). But user-defined fns SHOULD have
exact arity matching enforced.

**Reproducer**:
```
fn add3(a: i32, b: i32, c: i32) -> i32 {
    a + b + c
}
fn main() -> i32 {
    add3(10, 20)     // 1 missing arg — c reads garbage from rdx.
                     // No trap fires; output is non-deterministic.
}
```

**Recommended fix**:
After the pass-1 loop, before the `> 6` check, add:
```
if arg_count != pp_count {
    if pp_count > 0 {                    // skip for builtins
        emit_trap_with_id(16003);
    };
};
```
Skip the trap when `pp_count == 0` to preserve the builtin-call path
(builtins are not in fn_type_table, so pp_count=0 means "skip the check").

**Trap-id reservation**:
16003 (AST_CALL arity mismatch with declared signature). 16001 = arg-type
mismatch, 16002 = arity > 6, 16003 = arity ≠ declared.

---

## Finding 7: AST_TUPLE_LIT / AST_TUPLE_FIELD wrap disp8 silently for arity > 15

**Location**:
- AST_TUPLE_LIT: helixc/bootstrap/kovc.hx:2972-2999
- AST_TUPLE_FIELD: helixc/bootstrap/kovc.hx:2928-2933
**Severity**: MEDIUM
**Category**: silent-corruption

**Description**:
AST_TUPLE_LIT emits `mov [rsp+disp8], eax` (opcode `89 44 24 disp8`) for each
element with disp8 = `off` where off increments by 8. For the 17th element,
off = 128 — `emit_byte(128)` pushes the byte value 128, which CPU interprets
as signed disp8 = -128. The store goes to `[rsp - 128]` (BELOW the alloca'd
region) — silently corrupts whatever lives there (red-zone, parent frame's
saved rbp, etc.).

AST_TUPLE_FIELD similarly: `mov eax, [rax+disp8]` with `off = p2 * 8`. For
field idx 16, off = 128 → reads `[rax - 128]` — garbage.

The comment at line 2970-2971 acknowledges "Limitation: arity must be ≤ 15
because slot offsets (i*8) need to fit in signed disp8. Tuples beyond that
are deferred." But there is no parser-side or codegen-side trap. The Stage 4
test suite covers up to 16 elements (commit 5088ba2 mentions "10-element +
16-element"), but with 16 elements off ranges 0..120 — still in disp8. 17+
silently breaks.

**Reproducer**:
```
fn main() -> i32 {
    let big = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];
    big[16]
    // off=128 wraps to -128; reads [rax-128] = garbage.
}
```

**Recommended fix**:
Either (a) trap when `arity > 15` / `p2 > 15` until Stage 5 implements a
disp32 store/load, or (b) implement disp32 emission when the offset would
overflow disp8.

For (a):
- AST_TUPLE_LIT at line 2972: add `if arity > 15 { emit_trap_with_id(50001) ; ... }` early-return.
- AST_TUPLE_FIELD at line 2928: add `if p2 > 15 { emit_trap_with_id(52001) }`.

**Trap-id reservation**:
- 50001 (AST_TUPLE_LIT arity > 15 — disp8 overflow)
- 52001 (AST_TUPLE_FIELD field idx > 15 — disp8 overflow)

Note: AST_INDEX (tag 53) does NOT have this issue — it computes the offset
in ecx via `imul ecx, ecx, 8` and uses `add rax, rcx` (REX.W), so disp is
not used. Only the static-disp paths (TUPLE_LIT store and TUPLE_FIELD load)
are affected.

---

## Finding 8: Comparison ops (LT/GT/EQ/NE/LE/GE) — narrow-vs-narrow type mismatch slips through

**Location**: helixc/bootstrap/kovc.hx:3620-3917 (each AST_LT/GT/EQ/NE/LE/GE arm)
**Severity**: MEDIUM
**Category**: silent-corruption / safety

**Description**:
The comparison cascade traps on:
- bf16 (any side)
- f64-vs-non-f64 (mixed)
- i64-vs-non-i64 (mixed)
- u64-vs-non-u64 (mixed)
- f32-vs-non-f32 (mixed)
- u32-vs-non-u32 (mixed)

But the final fall-through is `emit_lt_eax_ecx()` / `emit_gt_eax_ecx()` /
etc., which is **signed 32-bit** comparison. Cases that escape:

- `u8 < i32`: l_u32=0 (u8 is tag 7, not 6), r_u32=0. Falls to signed 32-bit
  cmp. For u8 in [0, 255] and i32 negative, signed compare reports negative
  < u8 — semantically defensible if u8 widens to i32, but the type system
  permits the mismatch silently (no trap).
- `u8 < u16`: both tag != 6 (u32). Falls to signed compare. Both fit in
  non-negative i32, semantically benign — but mixing types should not be
  allowed without a cast.
- `i8 < u8`: signed compare on the zero-extended u8 vs sign-extended i8.
  Semantically wrong if i8 is negative (signed compare sees i8 as more
  negative than u8 = 0 always; correct), so accidentally OK. But the type
  mismatch should still be diagnosed.
- `u8 < u8`: semantically OK (both non-negative i32 after movzx, signed and
  unsigned compare agree). Not a bug, just a not-yet-trapped same-type
  narrow path.

The PROPER fix is to require both operands to have the same tag. Currently
the cascade only enforces "if one side is wide-class W, the other must also
be wide-class W"; narrow types share the i32 fall-through bucket and can
mix freely without a trap.

This is MEDIUM rather than HIGH because in most practical cases (u8 < u8,
u16 < u16, i8 < i8) the signed 32-bit comparison happens to produce correct
results. Mixing narrow and wider types (u8 < i32) silently degrades to
implicit widening — a type-soundness gap, not a bit-pattern corruption.

**Reproducer**:
```
fn main() -> i32 {
    let a: u8 = 200_u8;
    let b: i32 = -50_i32;
    if a < b { 1 } else { 0 }
    // Expected (with type checking): trap or compile error.
    // Actual: signed 32-bit cmp(200, -50) = false. Returns 0.
    // The user might expect "u8 widening to i32 = 200, compare > -50 = false"
    // — coincidentally correct, but the compiler accepted unrelated types.
}
```

**Recommended fix**:
After the existing wide-class trap matrix, add a final "narrow-type-mismatch"
guard that compares `expr_type(p1) == expr_type(p2)` using the full tag.
If they differ AND the wide-class checks all said "no wide", emit a trap.

Alternatively, refactor the cascade to dispatch on the joint (l_ty, r_ty)
tuple via a helper `arith_dispatch_narrow(l_ty, r_ty, op)` that emits the
correct cmp (signed for i8/i16/i32, unsigned for u8/u16/u32) and traps on
all narrow-type mismatches.

**Trap-id reservation**:
- 6052 (AST_LT narrow-type mismatch)
- 19052 (AST_GT)
- 20052 (AST_EQ)
- 21052 (AST_NE)
- 22052 (AST_LE)
- 23052 (AST_GE)

---

## What was checked but found OK (no new finding)

- **AST_BNOT for narrow types**: 32-bit `not eax` is fine because narrow
  load/store re-truncates. Acknowledged in line 3482-3488 ("AUDIT VERIFIED
  2026-05-07").
- **AST_NOT for f32 -0.0 / NaN**: language-policy, not memory safety
  (line 3520-3529). Same accepted-corner-case as f64.
- **AST_INTLIT_U64 wrap > 2^32**: documented as KNOWN GAP at line 2897-2903
  with explicit pointer to docs/STAGE_24B_NOTES.md. Not a NEW finding.
- **AST_TUPLE_FIELD / AST_INDEX 4-byte load**: documented at line 2968 as
  "high 4 bytes of slot unused" — known Phase-0 limitation. Tuples with
  i64/u64/f64 elements silently truncate, but expr_type returns i32 so
  downstream ops would trap on type mismatch.
- **Parser silently skips `: T` annotation** (parser.hx:564-568): documented
  as Phase-0 design choice. Bind_state takes the binding type from val_ty,
  so any "declared type ≠ value type" is invisible. Not a new sweep miss.
- **emit_u16_le lacks negative-value workaround** (kovc.hx:32-36): all
  current callers pass small positive constants. Latent only.
- **AST_NEG for u32, u8, u16**: 32-bit `neg eax` produces correct mod-2^32
  / mod-2^N two's complement after subsequent narrow re-truncation. OK.
- **AST_BAND / AST_BOR / AST_BXOR**: dispatch i64 correctly (line 3562,
  3578, 3595) and trap on l_i64 != r_i64. The unsigned/narrow types fall
  through to 32-bit op, which is correct after narrow re-truncation.
- **AST_WHILE cond width**: REX.W test for i64/u64 (line 4243-4244). OK.
- **AST_IF cond width**: same pattern. OK.
- **fn_table_lookup miss → ud2 patch** (audit-11 fix, line 4602-4610). OK.
- **bind_push_typed cap-overflow**: silently skips, returns -1 (line 884-898).
  AST_VAR's missing guard (Finding 3) is the consequence — if AST_VAR's
  guard is added per Finding 3, the cap overflow becomes a loud trap.

---

## Summary

| # | Severity | Finding |
|---|----------|---------|
| 1 | HIGH | AST_FN_DECL body-vs-ret-ty trap only checks 8-byte width |
| 2 | HIGH | AST_NEG falls through to 32-bit `neg eax` for u64 |
| 3 | HIGH | AST_VAR has no unbound-name guard (reads saved rbp) |
| 4 | HIGH | AST_ASSIGN narrow-bind-ty ignores value type |
| 5 | HIGH | AST_SHL / AST_SHR don't dispatch u64 / f32 / f64 / bf16 (+ SHR signedness sub-finding) |
| 6 | MEDIUM | AST_CALL arity mismatch (arg_count != pp_count) not trapped |
| 7 | MEDIUM | AST_TUPLE_LIT / AST_TUPLE_FIELD disp8 wrap for arity > 15 |
| 8 | MEDIUM | Comparison ops narrow-type mismatch slips through |

5 HIGH, 3 MEDIUM, 0 LOW. Recommend addressing Findings 2, 3, 5 first
(localized fixes; existing helpers cover them). Finding 1 and Finding 4
require slightly broader refactors but no new helpers. Findings 6, 7, 8
are MEDIUM and can be deferred or addressed alongside related Stage 5
type-system work.

The bf16 sweep + trap-id sweep correctly cover binop arithmetic and
comparison wide-class dispatch. The remaining silent windows are in
unary ops (NEG), shifts (SHL/SHR), assignment narrow-bind cases, function
boundaries (FN body/ret + CALL arity), variable reads (VAR unbound), and
disp8-bounded composite types (tuples > 15).
