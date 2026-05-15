# Stage 28.9 Cycle 111 — Silent-Failure Audit

**Date**: 2026-05-12
**HEAD**: `f9425a0` on `main` (per prompt; actual git HEAD has one further
commit `07e6535` "Stage 30 cycle-3 MEDIUM fix: add regression test for trap
62033" — out of scope for cycle-111 silent-failure rotation, called out
under sub-threshold observation S-7 below)
**Counter at start**: 1/5 (per prompt — cycle-110 fix-sweep landed cleanly
and the f9425a0 Stage-30 cycle-2 effort closed C109-CR-F3 / C109-SF-F1)
**Auditor**: silent-failure-hunter (Stage 28.9 cycle 111 rotation)
**Mode**: STRICT READ-ONLY — only this report file is written.

## Prior cycle context

Cycle-109 silent-failures audit (commit `ae43303`) returned **1 CRITICAL +
4 HIGH** in the silent-failures lane plus 4 sibling HIGH findings from
codereview / type-design lanes, for **8 closures** total. Two fix-sweeps
landed:

- **`9c451e6`** — "Stage 28.9 cycle-110 fix-sweep: 3 cycle-109 findings
  (F2/F3/F4)" — closed F2 (A.Range silent return-None loud-fail), F3
  (cast<u32, f64/f32> zero-extend arm), F4 (BIT_AND/OR/XOR/SHL/BIT_NOT/
  NEG `_is_i64_type` → `_is_64bit_int_type` width-gate promotion).
- **`1aecbae` + `f9425a0`** — "Stage 30 cycle-2" pair — closed C109-SF-F1
  (`emit_prologue` 1024 → 4096 + `bind_alloc_offset` trap threshold 1024
  → 4096) and C109-CR-F3 (`patch_table_add` / `bind_push_typed` /
  `fn_table_add` cap-exceeded paths got `emit_trap_with_id` calls with
  ids 10031 / 10032 / 10033). The early_err sentinel wire-up in
  `parser.hx` for traps 62032 / 62033 landed alongside.

## Rotation surfaces — verification log

The cycle-111 prompt named 7 numbered rotation surfaces. Each is verified
in turn; findings (if any) escalate to the **Findings** section below.

### Surface 1 — cycle-110 fix-sweep diff (closures F2/F3/F4)

**Diff under inspection**:
`git -C C:/Projects/Kovostov-Native diff 8ea039e..9c451e6 -- helixc/ir/
lower_ast.py helixc/backend/x86_64.py helixc/tests/test_ir.py`

**F2 closure (A.Range non-For-iter loud-fail)** —
`helixc/ir/lower_ast.py:2006-2020`. Pre-fix `if isinstance(expr, A.Range):
return None` silently dropped; caller's `or const_int(0)` substituted 0.
Post-fix: `raise NotImplementedError(...)` with `expr.span.line:col` in
message. **Mechanically correct.** The arm placement is immediately
after the A.Return arm (line 2005's `return None` is the A.Return arm's
intentional sentinel for control-terminates, not a fallthrough; verified
at lines 1998-2005 — emits `tir.OpKind.TRACE_EXIT` / `builder.ret(v)`
before returning). No new silent arm above it was introduced in this
diff. ✓

**F3 closure (cast unsigned-source → float)** —
`helixc/backend/x86_64.py:1329-1357`. Two new arms added before the
existing i32→f64 / i32→f32 arms. Both gate on
`_is_unsigned_int_type(from_ty) and not from_is_i64 and to_is_f64/
to_is_float`. The `not from_is_i64` clause **explicitly excludes u64 and
usize sources**. This is consistent with the in-source commit comment at
line 1330 ("when the source is an unsigned int (u8/u16/u32)") and with
the commit-message scoping ("`cast<u32, f64/f32>` uses signed-int
conversion").

**However**: the cycle-109 F3 finding explicitly listed `cast<u64, f64>`
and `cast<u64, f32>` as part of the same defect class (citation:
`docs/audit-stage28-9-cycle109-silent-failures.md` lines 296-306 and the
F3 summary at line 458-460 — "F3 (HIGH conf 88): `cast<u32, f64>` /
`cast<u32, f32>` / `cast<u64, f64>` / `cast<u64, f32>` use signed
`cvtsi2sd`/`cvtsi2ss` opcodes for unsigned-int sources"). The cycle-110
commit message scoped the fix down to the u32 sources WITHOUT
acknowledging the u64 / usize part of the original finding remains open.
**This is a finding** — see **F1** below.

**F4 closure (BIT_AND/OR/XOR/SHL/BIT_NOT/NEG width-gate promotion)** —
`helixc/backend/x86_64.py:1550-1645`. Each of the six emit sites flipped
its `_is_i64_type` predicate to `_is_64bit_int_type`. Predicate semantics
verified at lines 1037-1060: `_is_i64_type` matches `i64`/`isize` only;
`_is_64bit_int_type` matches `i64`/`isize` + `u64`/`usize`. ISA spec:
the AND/OR/XOR/SHL/NOT/NEG opcodes are sign-agnostic — only operand
width matters. **Mechanically correct.**

The promotion was NOT extended to the adjacent `SHR` arm at lines
1610-1624, which still uses `_is_i64_type` (line 1614). This is **NOT a
silent regression introduced by cycle-110** — the deferred-known
disposition was preserved (the cycle-109 prompt explicitly said
"SHR signed-vs-unsigned deferred but BIT_* not"; the cycle-110 commit
message reiterated "SHR remains signed-only (sar) for now; cycle-101
finding F2 deferred-known"). But because SHR's wide-gate predicate is
ALSO `_is_i64_type`, `u64 >> N` and `usize >> N` ALSO **silently
truncate to 32 bits**, not just use the wrong arithmetic-vs-logical
shift opcode. This is the **width** half of the SHR deferred-known —
the cycle-101 deferred-known framing was about the sign half (sar
vs shr opcode choice), and the cycle-110 sweep's pattern of promoting
adjacent `_is_i64_type` predicates to `_is_64bit_int_type` would
naturally have included SHR's width-gate. The cycle-110 commit message
folds both halves into "deferred." This is **F2** below — a HIGH-class
finding for the unaddressed sibling.

**Regression tests** — `test_ir.py:596-665, 697-830` covering F2/F3/F4
closures. All tests use byte-pattern assertions (`assert b"\xF2\x48\x0F\
x2A\xC0" in elf`) which are sufficiently discriminative for the
specific opcode forms. The c110 "NEG sibling" test at line 814-830
ACTUALLY exercises the SUB path (`0_u64 - a` lowers to SUB, not NEG)
— this is a **test-quality gap** flagged as sub-threshold observation
**S-2** below (the NEG codegen arm at line 1641 IS promoted, but no
test exercises it). Not a silent failure in production code.

### Surface 2 — Stage 30 cycle-2 kovc.hx diff (prologue + cap-exceeded traps)

**Diff under inspection**:
`git -C C:/Projects/Kovostov-Native diff 8ea039e..1aecbae --
helixc/bootstrap/kovc.hx` (the f9425a0 commit didn't modify kovc.hx).

**Prologue bump (emit_prologue 1024 → 4096)** —
`helixc/bootstrap/kovc.hx:752-756`. The `sub rsp, IMM` literal at
`emit_u32_le(4096)` (line 755) matches the matching comment-table at
line 731 (`48 81 EC 00 10 00 00   sub rsp, 4096`). The literal change is
mechanically consistent.

**bind_alloc_offset trap threshold (1024 → 4096)** —
`kovc.hx:1078-1080`. `if off >= 4096 { emit_trap_with_id(10030); };`.
Matches the prologue allocation size. ✓

**Stale 1024 magic numbers** — `grep -n 1024 helixc/bootstrap/kovc.hx`
yields six matches at lines 733, 736, 742, 1075, 4751, 4752. ALL of these
are in **comments**, not in code. The comments at 4751-4752 are stale
documentation that reads "exhausts the 1024-byte prologue more quickly"
+ "1024 - 64*8 = 512 bytes" — these reference the OLD prologue size and
the OLD bind_state cap. They are not load-bearing for behavior. Sub-
threshold observation **S-4** below.

**Trap-id collision check** — new traps 10031 / 10032 / 10033 added in
this diff. Existing in-source emit_trap_with_id call sites enumerated
via `grep -nE "^\s*emit_trap_with_id\([0-9]+\)" kovc.hx`: 10030 (one
site, bind_alloc_offset), 10031 (one site, patch_table_add), 10032
(one site, bind_push_typed), 10033 (one site, fn_table_add). The other
trap id ranges (2xxx, 8xxx, 9xxx, 14xxx, 16xxx, 19xxx-24xxx, 26xxx,
27xxx, 28999, 31xxx, 42xxx, 52xxx, 60030, 62xxx, 81002, 99001) do not
overlap with 10031-10033. **No collisions.** ✓

For the parser.hx sentinel-pattern traps 62032 / 62033 (added by Stage
30 cycle-2 commits fe7042f + f9425a0): these are `mk_node(99, ID, 0,
0)` AST_ERR sentinels stored in `early_err`, NOT `emit_trap_with_id`
calls. Codegen for AST_ERR (tag 99) lowers to `emit_trap_with_id(id)`
via the AST_ERR arm. The ids 62032 / 62033 are within the 62xxx range
used elsewhere; `grep` shows 62001 (line 4592), 62002 (cited in test_
codegen.py), 62005, 62006 (cited in test_codegen.py). 62032 and 62033
do not collide with these. ✓

**Stale `64` references to former bind_state cap** — the comment at
line 4752 references `64*8 = 512` budget. The fact that bind_state cap
is now 512 (Stage 29.1 bump) and the prologue is 4096 means this stale
comment misleads about the actual headroom. Sub-threshold **S-4**.

### Surface 3 — DIV / MOD / SHR adjacent silent gaps post cycle-110 promotion

The cycle-110 F4 fix promoted six emit sites to `_is_64bit_int_type`.
The adjacent SHR / DIV / MOD sites were left at `_is_i64_type`. The
deferred-known framing in the cycle-110 commit message covers the
**sign** half of these (signed sar vs unsigned shr, signed idiv vs
unsigned div). The **width** half is also unaddressed:

- **SHR** at `x86_64.py:1610-1624` — `if self._is_i64_type(op.results[0].
  ty)` at line 1614. For `u64 >> N`, the result_ty is `u64` and
  `_is_i64_type(u64) = False`. Falls through to the 32-bit `sar eax, cl`
  path: load only low 32 bits via `mov_eax_mem_rbp`, signed-shift,
  store via `mov_mem_rbp_eax`. Result: high 32 bits of the u64 are
  silently zero (because the store is `mov [rbp+disp], eax` which
  zero-extends in the destination's mental model — actually, it writes
  4 bytes and leaves the next 4 bytes of the 8-byte slot stale, which
  is a SEPARATE concern). The bug class is parallel to the F4 closure
  for BIT_AND/OR/XOR/SHL.
- **DIV** at `x86_64.py:1499-1523` — line 1513 uses `_is_i64_type`. For
  `u64 / u64`, falls through to `_emit_idiv_guarded` (line 1522) which
  is the signed 32-bit `idiv` path. Both width-truncation AND signed-
  vs-unsigned division semantics are wrong.
- **MOD** at `x86_64.py:1524-1539` — line 1528 uses `_is_i64_type`.
  Same shape as DIV.

These are pre-existing deferred-known. **The cycle-110 fix did not
introduce any NEW silent gap in SHR / DIV / MOD; it left the existing
ones unaddressed.** The disposition matches the prompt's stated
expectation ("cycle-110's BIT_*/SHL/BIT_NOT/NEG promotion exposed any
NEW silent gaps in these adjacent sites" — answer: NO new gaps; the
existing deferred-known scope is preserved). But the width half of
SHR specifically (not the sign half) is **adjacent enough to the F4
sweep** that its omission is a reasonable cycle-111 finding —
documented as **F2** below.

### Surface 4 — patch_table_add / bind_push_typed / fn_table_add loud-fail wire-up

Each `*_add` function's cap-exceeded branch now contains:

```
let top = __arena_get(...);
if top >= CAP {
    emit_trap_with_id(ID);
    0 - 1
} else { ...write entry... 0 }
```

at lines 1033-1036 (`bind_push_typed`, id 10032, cap 512), 1569-1571
(`fn_table_add`, id 10033, cap 512), 1636-1639 (`patch_table_add`, id
10031, cap 16384).

**Important semantic check**: `emit_trap_with_id` is a **codegen
helper**, not a Helix `panic!` / `abort` primitive. Its body
(`kovc.hx:262-267`):

```
fn emit_trap_with_id(id: i32) -> i32 {
    emit_byte(0xB8);                           // mov eax, imm32
    emit_u32_le(id);                           // 4 bytes of imm32
    emit_byte(0x0F); emit_byte(0x0B);          // ud2
    7
}
```

It **EMITS 7 bytes of machine code into the output binary** at the
current `__arena_len()` position. It does NOT halt the compile. So
when `patch_table_add`'s 16385th call fires the trap, the 7 trap-bytes
get appended to the in-progress code stream at exactly the position
where the caller is mid-emission of an instruction sequence (typically
right after `emit_call_rel32_placeholder` at `kovc.hx:6080`). The
caller still gets `0 - 1` returned (which it still discards), so the
PATCH itself is still silently dropped. The fix achieves "runtime
SIGILL with eax=10031 IF the trap bytes happen to be reached during
execution of the output binary."

**Is this a regression**? Pre-cycle-2 the cap-exceeded path was a pure
silent return-of-`-1`. Post-cycle-2 it's "silent return-of-`-1` PLUS
runtime trap-bytes injected at codegen-cursor position." The trap
bytes may or may not be reached at output-binary runtime depending on
control flow around the injection point — but injecting them is
strictly LESS silent than not injecting them. The trap bytes also do
NOT desynchronize relocation accounting because relocation offsets are
tracked via absolute `__arena_len()` positions (verified at `kovc.hx:
620, 628, 641, 716-718`), not via caller-tracked byte counts. So a
disp_slot captured BEFORE the trap remains valid; one captured AFTER
the trap reflects the post-trap arena position.

This implementation matches the cycle-109 codereview F3 recommendation
verbatim ("convert silent return-of-`-1` to an `emit_trap_with_id(N)`
call inside each `*_add` function, with a distinct trap ID. (See
`bind_alloc_offset` at line 1052-1054 for the existing pattern — trap
ID 10030.)"). The `bind_alloc_offset` precedent has been in place
since the Stage 5-6 audit Finding #11 without observed corruption,
so the convention is established. **NO finding here.** ✓

Adjacent concern: the `bind_state` cap (512) and the `bind_alloc_
offset` trap threshold (off >= 4096) interact in a subtle way. The
bind_state cap allows 512 simultaneous bindings (bind_push_typed
permits 511, 512 — the 513th traps via 10032). But bind_alloc_offset
traps when the NEXT offset request is at off >= 4096; with offsets
incrementing by 8 from initial 8 (verified at `bind_reset` line
1138-1141 `__arena_set(state, 8)`), the 512th call has off = 8 +
511*8 = 4096 which TRIPS the 10030 trap. So bind_alloc_offset's trap
fires ONE binding earlier than bind_push_typed's. This is documented
under sub-threshold **S-3** below — not a silent failure (both traps
are loud) but an inconsistency that the cycle-3 type-design audit
hint also raised.

### Surface 5 — monomorphize / iter_fn_decls walkers

No diff in this surface across the cycle-110 fix-sweep + Stage 30
cycle-2 commits. `git diff 8ea039e..f9425a0 -- helixc/frontend/`
shows zero changes. The cycle-60..63 sweep is the most recent
maintenance; pre/post `flatten_modules` invariant documented at
`ast_walker.py:223-249`. **No new silent gaps.** ✓

### Surface 6 — typecheck._resolve_type sibling normalization

`PRIMITIVES` at `typecheck.py:336-351` and `_resolve_type` at
`typecheck.py:514-545` are unmodified across the cycle-110 fix-sweep
and Stage 30 cycle-2 commits. The cycle-105 F105-1 fix (textual `"()"`
→ TyUnit) is still in place at lines 519-520. No new silent
fallthrough paths introduced. ✓

### Surface 7 — IR lowering bottom-of-_lower_expr catch-all

`lower_ast.py:2281` is the bottom `return None` (catch-all). The
deferred-known A.StrLit is the only known silent fallthrough left.
The cycle-110 F2 fix added `raise NotImplementedError` for A.Range at
lines 2006-2020 — strictly above the catch-all in dispatch order, so
it cannot reintroduce silent A.Range dropping. The intervening arms
(A.Assign at 2021, A.MemberAccess at 2200-2212, A.Cast at 2213-2221,
A.Quote at 2224-2251, A.Splice at 2252-2257, A.Modify at 2258-2280)
are all unchanged. **No new silent-fold arm was introduced above the
catch-all.** ✓

## Findings

### F1 (HIGH, conf 86) — `cast<u64, f64>` / `cast<u64, f32>` still uses signed `cvtsi2sd` / `cvtsi2ss` after cycle-110 F3 partial fix

**File**: `helixc/backend/x86_64.py`
**Lines**: 1318-1323 (i64 → f64 arm — catches u64 → f64 because
`from_is_i64 = _is_64bit_int_type(u64) = True`) and 1365-1369 (i32 →
f32 arm — catches u64 → f32 because the prior arms don't claim u64
when to_is_f64 is false).

**Defect class**: silent miscompile / partial fix not acknowledged.

**Trace for u64 → f64**:
- `from_ty = u64` → `from_is_i64 = _is_64bit_int_type(u64) = True`,
  `from_is_float = False`, `_is_unsigned_int_type(u64) = True`.
- `to_ty = f64` → `to_is_f64 = True`, `to_is_float = True`,
  `to_is_i64 = False`.
- Line 1293 (i64→i32): requires `not to_is_float` → False. Skip.
- Line 1305 (cycle-108 unsigned widen): requires `not from_is_i64` →
  False. Skip.
- Line 1311 (i32→i64): requires `not from_is_i64` → False. Skip.
- Line 1318 (i64→f64): `from_is_i64 and to_is_f64` → **TRUE. FIRES.**
  Emits `mov rax, [src]` + `cvtsi2sd xmm0, rax` (F2 48 0F 2A C0).
- The cycle-110 F3 unsigned arms at 1340-1357 are NEVER REACHED for u64
  because they gate on `not from_is_i64`, which is False for u64.

**Concrete miscompile**: `let x: u64 = 0xFFFFFFFFFFFFFFFF_u64; x as f64`
loads `rax = -1` (signed 64-bit interpretation of 0xFFFF...). `cvtsi2sd`
produces `-1.0`. **Correct value**: ~`1.844674e19`.

**Trace for u64 → f32**:
- `from_ty = u64` → `from_is_i64 = True`, etc.
- `to_ty = f32` → `to_is_f64 = False`, `to_is_float = True`,
  `to_is_i64 = False`.
- Lines 1293, 1305, 1311 skip (same reasons as f64 trace).
- Line 1318 (i64→f64): `to_is_f64 = False`. Skip.
- Line 1325 (i64→i64): `to_is_i64 = False`. Skip.
- Lines 1340, 1349 (cycle-110 unsigned-int→float): `not from_is_i64` =
  False. Skip.
- Line 1359 (i32→f64): `to_is_f64 = False`. Skip.
- Line 1365 (i32→f32): `not from_is_float and to_is_float` → True.
  **FIRES.** Emits `mov eax, [src]` + `cvtsi2ss xmm0, eax`.

This is even worse than u64→f64: `mov eax` loads only the LOW 32 bits
of the u64, then signed-converts those to f32. So
`0xFFFFFFFFFFFFFFFF_u64 as f32` evaluates the signed-32 interpretation
of the low half (`0xFFFFFFFF` = -1), produces `-1.0_f32`. **Correct
value**: ~`1.844674e19_f32` (which doesn't even fit precisely in f32,
but should be the nearest representable, ~`1.8446744e19`, not `-1.0`).

**Reachability**: any `as f32` / `as f64` cast from a u64 or usize
source with the high bit (bit 63) set. The cycle-109 silent-failures
F3 finding (`docs/audit-stage28-9-cycle109-silent-failures.md` lines
296-323) explicitly enumerated this case and recommended "add explicit
unsigned-int→float arms BEFORE the i32→f-arm / i64→f-arm dispatch,
gated on `_is_unsigned_int_type(from_ty)`. The x86 sequence for u32→f64
is `mov eax, [src]` ... The u64→f64 case requires a longer
'split high/low + scale' sequence; alternatively emit a runtime helper
call." The cycle-110 fix added arms for the u8/u16/u32 sub-case (the
"easy" case where zero-extending into rax suffices) and **dropped the
u64/usize sub-case without acknowledging it in the commit message**
or the in-source comment.

The cycle-110 commit message body says:
> F3 (HIGH conf 88) — cast<u32, f64/f32> uses signed-int conversion:

— note the title scopes down to u32. The actual cycle-109 F3 finding
header was:
> F3 — `cast<u32, f64>` / `cast<u32, f32>` / `cast<u64, f64>` /
>      `cast<u64, f32>` use signed `cvtsi2sd` / `cvtsi2ss` opcodes
>      for unsigned-int sources

— note all four. The scope narrowing happened silently in the cycle-110
fix-sweep title; the in-source comment at lines 1329-1339 specifically
limits the new arms to "u8/u16/u32" sources. No deferred-known
disposition is recorded for the u64/usize remainder.

**Recommended fix**: add explicit `from_is_i64 and self._is_u64_type(
from_ty) and to_is_float` arms above the existing line-1318 i64→f64
arm. The x86 sequence for u64→f64 (full unsigned 64→f64) requires a
multi-instruction sequence because there is no single-instruction
unsigned-64-bit-to-double opcode (x86 AVX-512 VCVTUSI2SD does this in
one op, but baseline x86_64 lacks it). The canonical fallback is the
"high-bit branch" pattern:

```
; rax = u64 source
mov rax, [src]
test rax, rax
js .negative_branch
cvtsi2sd xmm0, rax           ; high bit clear, signed conversion == unsigned
jmp .done
.negative_branch:
mov rcx, rax
shr rax, 1
and rcx, 1
or rax, rcx
cvtsi2sd xmm0, rax
addsd xmm0, xmm0
.done:
```

Or equivalently call out to a runtime helper. A regression test of the
form `u64(0xFFFFFFFFFFFFFFFF) as f64` exit-code-checked against the
correct ~1.844e19 value would discriminate.

**Confidence**: 86. The CAST dispatch is explicit (verified line-by-
line); the cycle-109 finding's enumeration is direct; the cycle-110
fix's omission is documented in its own commit message and in-source
comment text. Slightly below cycle-109's original conf 88 because the
disposition could plausibly be re-classified as a fresh deferred-known
("u64-to-float needs multi-instruction sequence; deferred to Stage 30")
— but the cycle-110 fix did not record such a deferral. Above the HIGH
bar of 75.

---

### F2 (HIGH, conf 80) — `SHR` u64/usize width-gate silently truncates to 32 bits (parallel sibling of cycle-110 F4 closure)

**File**: `helixc/backend/x86_64.py`
**Lines**: 1610-1624 (SHR emit arm).

**Defect class**: silent miscompile / cycle-110 F4-sibling unaddressed.

```python
if op.kind == tir.OpKind.SHR:
    l_slot = self._slot_of(op.operands[0])
    r_slot = self._slot_of(op.operands[1])
    res_slot = self._slot_of(op.results[0])
    if self._is_i64_type(op.results[0].ty):       # ← still _is_i64_type
        self.asm.mov_rax_mem_rbp(l_slot)
        self.asm.mov_rcx_mem_rbp(r_slot)
        self.asm.sar_rax_cl()
        self.asm.mov_mem_rbp_rax(res_slot)
    else:
        self.asm.mov_eax_mem_rbp(l_slot)
        self.asm.mov_ecx_mem_rbp(r_slot)
        self.asm.sar_eax_cl()
        self.asm.mov_mem_rbp_eax(res_slot)
    return
```

**Trace for `u64 >> N`**:
- `op.results[0].ty = u64` → `_is_i64_type(u64) = False` (predicate
  matches only `i64`/`isize`).
- Falls to the else branch: 32-bit load + 32-bit sar + 32-bit store.
- `mov eax, [src]` reads only the low 32 bits of the u64.
- `sar eax, cl` does signed-32 arithmetic right shift.
- `mov [dst], eax` writes 4 bytes; the upper 4 bytes of the 8-byte slot
  are NOT touched by this store (they retain whatever was previously in
  the slot, which for a fresh let is the prologue-zeroed stack — see
  `emit_prologue` at `kovc.hx:752-756` which `sub rsp, 4096` without
  zeroing; the stack is technically uninitialized memory).

Two compounding bugs:
1. **Width truncation** — high 32 bits of u64 are discarded before the
   shift.
2. **Sign-extension via SAR** — `sar` (arithmetic right shift) preserves
   the source's sign bit. For `u32` re-interpretation of the low 32
   bits, the sign bit is bit 31 of the original u64. `u64 >> 1` for a
   high-bit-set value gets `sar_eax_cl(1)` which preserves bit 31 as
   the new bit 31 — fine for u32 semantics in the low half, but wrong
   for u64 semantics (which would have placed the shifted-in 0 at bit
   31 and the original bit 63 at bit 62).

Both bugs are silent: no compile error, no trap, the output binary
just produces a wrong number at runtime.

**Reachability**: any `u64 >> N` or `usize >> N` expression. K3 self-
host does not currently use u64 shifts in the bootstrap (verified by
`grep -n ">>" kovc.hx lexer.hx parser.hx` — no occurrences of `u64 >>`
or `usize >>` in source). External user code using u64 bitwise ops
(common in crypto, hashing, encoding) is the reachability path.

**Why this is a cycle-111 finding rather than a pre-existing deferred-
known**: the cycle-101 deferred-known for SHR was about the **sign**
half — sar (signed arithmetic) vs shr (logical) opcode choice. The
**width** half is parallel to cycle-110 F4's promotion of BIT_AND /
BIT_OR / BIT_XOR / SHL / BIT_NOT / NEG from `_is_i64_type` to
`_is_64bit_int_type`. The cycle-110 fix-sweep enumerated 6 emit sites
in the same `if op.kind == tir.OpKind.XXX:` cascade and stopped exactly
where SHR begins. The proximity makes the omission feel like an
oversight rather than a deliberate deferral. The cycle-110 commit
message frames it as "SHR remains signed-only (sar) for now; cycle-101
finding F2 deferred-known" — which conflates the sign half with the
width half. Treating the width half as included in the deferred-known
is defensible (cycle-101 framing) but inconsistent with the F4 sweep's
pattern. Confidence reflects this ambiguity.

**Recommended fix**: extend the SHR arm's width predicate to
`_is_64bit_int_type` exactly like F4 did for SHL. The sign half (sar
vs shr opcode choice for unsigned types) can remain deferred under the
cycle-101 framing — the fix only addresses width.

```python
if op.kind == tir.OpKind.SHR:
    l_slot = self._slot_of(op.operands[0])
    r_slot = self._slot_of(op.operands[1])
    res_slot = self._slot_of(op.results[0])
    if self._is_64bit_int_type(op.results[0].ty):
        self.asm.mov_rax_mem_rbp(l_slot)
        self.asm.mov_rcx_mem_rbp(r_slot)
        self.asm.sar_rax_cl()       # still SAR for now; SHR variant
        self.asm.mov_mem_rbp_rax(res_slot)
    else:
        self.asm.mov_eax_mem_rbp(l_slot)
        self.asm.mov_ecx_mem_rbp(r_slot)
        self.asm.sar_eax_cl()
        self.asm.mov_mem_rbp_eax(res_slot)
    return
```

Regression test pattern: `let x: u64 = 0x100000000_u64; (x >> 1)` should
return 0x80000000 (= 2^31), NOT 0 (which is what the 32-bit truncation
yields). Byte-pattern discriminator: REX.W-prefixed `sar rax, cl` (48
D3 F8) instead of `sar eax, cl` (D3 F8) for u64 result type.

**Confidence**: 80. Below F1's 86 because the cycle-101 deferred-known
framing arguably covers this; above the HIGH bar of 75 because the
cycle-110 commit message's "SHR remains signed-only" is a categorical
description that doesn't acknowledge the width half separately.

---

## Sub-threshold observations (NOT findings — below HIGH/CRITICAL bar)

### S-1 (conf 65) — patch_table_add / bind_push_typed / fn_table_add trap-bytes inject mid-emission, may not be reached at output runtime

The `emit_trap_with_id(10031/10032/10033)` calls inside the cap-exceeded
branches of `patch_table_add`, `bind_push_typed`, `fn_table_add` emit 7
bytes of `mov eax, IMM32; ud2` into the output binary at the current
`__arena_len()` position. The caller is typically mid-emission of a
function body (e.g., `patch_table_add` is called right after
`emit_call_rel32_placeholder` at `kovc.hx:6080`). The injected trap
bytes therefore land between meaningful instructions, where their
execution depends on the surrounding function's control flow. Some
trap-byte positions will be unreachable (e.g., if a `ret` precedes
them); others will execute when the function is called.

This implementation matches the cycle-109 codereview F3 recommendation
verbatim ("convert silent return-of-`-1` to an `emit_trap_with_id(N)`
call inside each `*_add` function") and matches the existing
`bind_alloc_offset` precedent (since Stage 5-6 audit Finding #11). Both
the recommendation and the precedent treat the trap-byte-injection as
"better than pure silent skip" — i.e., the fix achieves "loud-fail at
output-binary runtime IF the trap bytes are reached, else silent
drop" rather than "loud-fail at compile time." The latter is not
achievable in the bootstrap kovc.hx because there is no `abort()` /
`panic!()` primitive available to the bootstrap source.

Below the HIGH bar because:
- The convention is established (4 cycles of `bind_alloc_offset` use).
- The cap-overflow cases are unreachable in current self-host (K3 uses
  ~6800 patches vs 16384 cap; ~200 bindings/fn peak vs 512 cap; 472
  fns vs 512 cap — all well under the threshold).
- The pure-silent-skip pre-state was strictly worse.
- Relocation accounting is arena-position-based, not byte-count-based,
  so the trap-byte injection doesn't desynchronize patches.

Recorded for documentation.

### S-2 (conf 70) — `test_c110_neg_u64_via_sub_emits_64bit_form` exercises SUB path, not NEG path

`helixc/tests/test_ir.py:814-830` is named "NEG sibling" but the source
under test is `0_u64 - a` which lowers to `tir.OpKind.SUB`, not
`tir.OpKind.NEG`. The IR lowering of unary `-x` (which emits NEG —
verified at `lower_ast.py:1177-1178` `if expr.op == "-": return self.
builder.emit(tir.OpKind.NEG, inner, result_ty=inner.ty)`) is NOT
exercised. The cycle-110 F4 fix DID promote NEG's width-gate to
`_is_64bit_int_type` (line 1641), and that production code path is
correct, but the regression coverage is missing.

A direct test would be `let x: i64 = ...; -x`. Helix may not support
`-x` for u64 (unary minus typically requires signed types), so the
i64 case is the natural test. The test docstring itself acknowledges
"The unary `-` on unsigned types is rare in source code (lowering
uses SUB for unsigned-domain 'negation')" but doesn't note that this
means the test exercises SUB rather than NEG.

Test-quality gap, not a production silent failure. Sub-threshold.

### S-3 (conf 72) — bind_alloc_offset trap threshold off-by-one inconsistent with bind_state cap

`bind_alloc_offset` traps at `off >= 4096`. With offsets advancing from
initial 8 in steps of 8, the trap fires on the 512th allocation request
(when `off = 8 + 511*8 = 4096`). But `bind_state` cap allows 512
simultaneous bindings (bind_push_typed traps at `top >= 512`, i.e.,
on the 513th push).

So:
- 511 simultaneous bindings: both caps pass.
- 512 simultaneous bindings: bind_alloc_offset TRAPS (offset 4096) but
  proceeds to emit `mov [rbp-4096], rax` (which writes to bytes
  [rbp-4096, rbp-4089) — the bottom 8 bytes of the 4096-byte frame,
  WITHIN frame, legal). bind_push_typed succeeds.
- 513 simultaneous bindings: bind_alloc_offset TRAPS (offset 4104,
  writes to bytes [rbp-4104, rbp-4097) — past the frame bottom by 8
  bytes, corrupting parent frame). bind_push_typed TRAPS.

The frame can technically hold 512 slots (offsets 8 through 4096) but
the `off >= 4096` check forbids the 512th slot. The cap-vs-budget
inconsistency wastes one binding's worth of headroom and makes
bind_alloc_offset's trap fire before bind_push_typed's. Sub-threshold
because both traps ARE loud, no silent failure, and the off-by-one
costs only one binding.

This was also flagged by the Stage 30 cycle-3 type-design audit
according to the commit message at `07e6535` ("The LOW finding
involves concurrent agent's cycle-110 work on emit_prologue/
bind_alloc_offset coordination"). Cross-references support cycle-111
treating it as sub-threshold (it's a LOW-class observation, not HIGH).

### S-4 (conf 70) — stale "1024-byte prologue" and "64*8 = 512" comments in kovc.hx

`kovc.hx:1052` ("blowing past the 512-byte prologue") and lines
4751-4752 ("exhausts the 1024-byte prologue more quickly. For Phase-0
the headroom (1024 - 64*8 = 512 bytes = 64 extra slots)") reference
the OLD prologue size (1024 bytes) and the OLD bind_state cap (64
entries). The actual values are now 4096 / 512. The stale comments
mislead future maintainers about the headroom math.

The cycle-2 commit's NEW comment at lines 740-749 documents the bump
correctly, but the in-code older comments at 1052 and 4751-4752 were
not refreshed.

Documentation drift, not a code defect. Sub-threshold.

### S-5 (conf 60) — F11 test no longer exercises the +SIGILL boundary direction post-cycle-110 fix

`helixc/tests/test_codegen.py:3630-3660` (the F11 regression test for
bind_alloc_offset trap) was rewritten in the cycle-110 fix. Pre-fix it
tested `140 chained lets → SIGILL=132` (positive trap-fire direction).
Post-fix it only tests `256 chained lets → exit 42` (positive success
direction within new budget). The new test discriminates pre/post
cycle-110 because 256*8 = 2048 > 1024 (old budget would have trapped)
but fits in 4096 (new budget passes), so the test would have FAILED
under pre-cycle-110 code. But the +SIGILL direction at 525+ lets is
no longer exercised. The test docstring acknowledges: "directly
testing the new boundary (525+ lets) currently runs into a pre-
existing bootstrap output-emission issue at very large source sizes
(bisected: crash above ~362 lets, unrelated to cycle-110)."

The "crash above ~362 lets" upstream issue is the actual silent
failure (whatever it is), but it's "unrelated to cycle-110" per the
test author's bisect. Investigating this upstream issue would be a
fresh audit task; for cycle-111 it's recorded as sub-threshold.

### S-6 (conf 60) — `cast<i64, f32>` / `cast<u64, f32>` use signed 32-bit cvtsi2ss after truncation (pre-existing)

The CAST dispatch has an arm for `i64 → f64` at line 1318 but NO arm for
`i64 → f32`. Trace for `i64 as f32`:
- `from_is_i64 = True`, `to_is_f64 = False`, `to_is_float = True`.
- Lines 1293, 1305, 1311 skip.
- Line 1318: `to_is_f64 = False`. Skip.
- Line 1325: `to_is_i64 = False`. Skip.
- Lines 1340, 1349: `not from_is_i64 = False`. Skip.
- Line 1359: `to_is_f64 = False`. Skip.
- Line 1365: `not from_is_float and to_is_float` → True. **Fires.**
  Emits `mov eax, [src]; cvtsi2ss xmm0, eax` — 32-bit truncation
  before signed conversion.

So `(0x100000000_i64) as f32` becomes 0 instead of ~4.29e9. Pre-existing,
not introduced by cycle-110. Recorded as sub-threshold because it's a
pre-existing gap; the cycle-110 F3 fix did not claim to address it.
This is closely related to F1's u64→f32 case (which also takes this
arm).

### S-7 (conf 85) — `07e6535` "Stage 30 cycle-3 MEDIUM fix" landed one commit after stated HEAD

The prompt states HEAD = f9425a0 with counter 1/5. The actual `git
rev-parse HEAD` reports `07e6535065b6849579cfb6edadc2e8dd8200b833` —
one commit further. That commit ("add regression test for trap 62033")
adds a test in `test_codegen.py` exercising the `Pt<+>{...}` trap-
62033 path (TK_PLUS at type-arg position). Test-only addition, no
production code change. The audit follows the prompt's stated HEAD of
f9425a0 for the rotation surface review. Recorded for transparency —
not a finding.

---

## Summary

The cycle-110 fix-sweep + Stage 30 cycle-2 effort closed 4 of the 5
silent-failures findings from cycle-109 (F1 emit_prologue + bind_alloc_
offset, F2 A.Range loud-fail, F3 cast u32→float, F4 BIT_*/SHL/NEG/NOT
width-gate). The F3 closure is **partial**: it covers u8/u16/u32 sources
but silently drops the u64/usize sources from the cycle-109 finding's
original enumeration. This omission is the cycle-111 F1 finding (HIGH
conf 86). The adjacent SHR width-gate (parallel to F4's BIT_*/SHL
sweep) was deferred under the cycle-101 sign-half framing, but the
width half is plausibly fix-sweep scope; this is cycle-111 F2 (HIGH
conf 80).

The `emit_trap_with_id(10031/10032/10033)` wire-ups in `patch_table_add`
/ `bind_push_typed` / `fn_table_add` follow the established
`bind_alloc_offset` pattern faithfully — the loud-fail traps are
emitted at codegen time INTO the output binary, achieving "runtime
SIGILL if reached" rather than "compile-time abort." This is the same
convention the cycle-109 codereview recommended; not a regression.

No new silent-failure regressions were introduced by the cycle-110 +
Stage 30 cycle-2 diffs. The two findings are **partial-fix gaps**
(cycle-110 didn't close the full cycle-109 scope) rather than fresh
regressions.

## Findings table

| ID | Severity | Confidence | Surface | Summary |
|----|----------|------------|---------|---------|
| F1 | HIGH | 86 | 1 | `cast<u64, f64>` / `cast<u64, f32>` still uses signed `cvtsi2sd`/`cvtsi2ss` after cycle-110 F3 partial fix (u32 sub-case fixed, u64 sub-case silently dropped from scope) |
| F2 | HIGH | 80 | 3 | `SHR u64` / `SHR usize` width-gate predicate `_is_i64_type` silently truncates to 32 bits; parallel sibling of cycle-110 F4 BIT_*/SHL/NEG/NOT promotion |

| Sub-threshold | Confidence | Brief |
|---------------|------------|-------|
| S-1 | 65 | `emit_trap_with_id` inside `*_add` injects bytes mid-emission; matches established `bind_alloc_offset` convention |
| S-2 | 70 | `test_c110_neg_u64_via_sub_emits_64bit_form` exercises SUB, not NEG |
| S-3 | 72 | `bind_alloc_offset` trap threshold off-by-one vs `bind_state` cap (cross-ref Stage 30 cycle-3 type-design LOW) |
| S-4 | 70 | Stale "1024-byte prologue" and "64*8 = 512" comments in kovc.hx |
| S-5 | 60 | F11 test no longer exercises +SIGILL boundary direction (upstream large-source issue) |
| S-6 | 60 | `cast<i64, f32>` / `cast<u64, f32>` pre-existing truncation-before-signed-conversion |
| S-7 | 85 | Actual git HEAD is 07e6535, one commit after prompt's stated f9425a0 (test-only addition) |

## Verdict

**FINDINGS** — 2 HIGH findings at conf ≥75 (F1 at conf 86, F2 at conf 80).

The bar for PASS is "ZERO new HIGH / CRITICAL findings at confidence ≥75%."
The cycle-111 audit identifies two HIGH findings above this bar. Per the
counter rule ("if PASS → 1/5 → 2/5; if FINDINGS → counter resets to 0/5"),
the counter **resets from 1/5 to 0/5**.

Both findings are **partial-fix gaps** from cycle-110, not fresh
regressions. F1 is the u64/usize remainder of cycle-109 F3 that the
cycle-110 fix-sweep silently scoped out. F2 is the SHR sibling of
cycle-110 F4's BIT_*/SHL/NEG/NOT width-gate promotion sweep that was
deferred under the cycle-101 sign-half framing. Closing them is
mechanically straightforward (extend predicates, add regression tests
for byte-patterns) and would tee up a cycle-112 PASS attempt.

## Cross-references

- **Cycle-109 silent-failures audit**: `docs/audit-stage28-9-cycle109-silent-failures.md`
  - F1 (CRITICAL conf 90) — `emit_prologue` 1024-byte vs Stage 29.1 bind_state cap 512 — **closed by Stage 30 cycle-2 `1aecbae`**
  - F2 (HIGH conf 92) — A.Range silent return None — **closed by cycle-110 `9c451e6`**
  - F3 (HIGH conf 88) — `cast<u32/u64, f64/f32>` signed-int conversion — **partial close by cycle-110 (u32 only)**; u64 remainder is cycle-111 **F1**
  - F4 (HIGH conf 92) — BIT_AND/OR/XOR/SHL/BIT_NOT/NEG width-gate — **closed by cycle-110**; SHR sibling is cycle-111 **F2**
- **Cycle-109 codereview audit**: `docs/audit-stage28-9-cycle109-codereview.md`
  - F1 (HIGH conf 88) — `test_c107_mut_u64_local_uses_64bit_load_store` vacuous — **closed by `test_c109_mut_u64_load_store_byte_identical_to_i64`** at test_ir.py:596
  - F2 (HIGH conf 85) — `test_c107_call_return_u64_caller_stores_full_rax` coverage gap — **closed by `test_c109_call_return_u64_caller_stores_full_rax`** at test_ir.py:633
  - F3 (HIGH conf 82) — `*_add` silent-failure pattern — **closed by Stage 30 cycle-2 emit_trap_with_id wire-ups** at kovc.hx:1034/1570/1637
- **Cycle-109 type-design audit**: `docs/audit-stage28-9-cycle109-type-design.md`
  - F109-1 (HIGH conf 80) — prologue/cap mismatch — **closed by Stage 30 cycle-2**
- **Cycle-110 fix-sweep commit**: `9c451e6` "Stage 28.9 cycle-110 fix-sweep: 3 cycle-109 findings (F2/F3/F4)"
- **Stage 30 cycle-1 audit**: `docs/audit-stage30-cycle1-findings.md` (2 HIGH + 3 MEDIUM)
- **Stage 30 cycle-2 commits**: `fe7042f` (parser.hx early_err wire-up), `1aecbae` (kovc.hx prologue + trap-id wire-up), `f9425a0` (test_codegen.py 62032 regression assertions)
- **Stage 30 cycle-3 commit**: `07e6535` (test_codegen.py 62033 regression test addition) — one commit beyond the prompt's stated HEAD, recorded as sub-threshold S-7

## No code edits made

This audit is STRICTLY READ-ONLY. The only file written by this auditor
is the present report at `docs/audit-stage28-9-cycle111-silent-failures.md`.
No `Edit` or `Write` operations were performed on any source file, test
file, or other documentation file. No tests were run (the audit is
static-only). Verification of findings F1 and F2 was performed by
manual control-flow tracing through the dispatch cascades in
`helixc/backend/x86_64.py` against the corresponding test sources in
`helixc/tests/test_ir.py`, cross-referenced against the cycle-109
silent-failures audit's enumeration of expected fix scope.
