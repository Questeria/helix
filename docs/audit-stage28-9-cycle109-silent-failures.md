# Audit Stage 28.9 cycle 109 — Silent failures

**Date**: 2026-05-12.

**HEAD**: `c89432e` (`🎉 STAGE 29.1: bumped patch_table + bind_state caps
→ K3 exits 42!`).

**Mode**: STRICT READ-ONLY. No source edits performed. This document
is the sole write of the audit. Source files only read/grepped via
the read-only tooling.

**Rotation (per prompt — avoid recently-audited surfaces):**

- Cycle-108 fix-sweep diff (`git diff b0e18c7..a600616`) — verify the 8
  fixed sites (`_is_i64_type → _is_64bit_int_type` at CALL int-arg,
  CALL return, RETURN, SELECT, BR, LOAD_VAR, STORE_VAR, CAST + new
  unsigned-widening arm + loud-fail arms for CharLit/StructLit/
  TileLit) introduce no new bugs themselves.
- Stage 29.1 cap-bump diff (`git diff a600616..c89432e`) — patch_table
  4096→16384, bind_state 64→512. Check for cascading layout / slot
  overlap consequences.
- `BIT_AND` / `BIT_OR` / `BIT_XOR` / `SHL` / `BIT_NOT` / `NEG` emit
  paths in `helixc/backend/x86_64.py` (per prompt explicitly in scope:
  "DIV/MOD/SHR signed-vs-unsigned deferred but BIT_* not").
- `COND_BR` emit path in `helixc/backend/x86_64.py`.
- `monomorphize` / `iter_fn_decls` walkers (defense-in-depth
  post cycle-65 sweep).
- `typecheck._resolve_type` sibling normalization (post cycle-106
  PRIMITIVES `"()"` drop).
- IR lowering bottom-of-`_lower_expr` `return None` remaining for
  `A.StrLit` / `A.ArrayLit` / `A.Range` (per cycle-108 F8 covering
  CharLit/StructLit/TileLit — what's still missing?).

**Deferred-known list (NOT re-flagged):**

monomorphize `_mangle_ty` / hash_cons `_ast_equal` silent catchalls;
typecheck/struct_mono pre-flatten in `check.py`;
`autotune.collect_autotuned_fns` missing `iter_fn_decls`;
`struct_mono.mangle_struct` collision;
`A.StrLit` IR-lowering gap (cycle-101 F1 deferred);
DIV/MOD signed-vs-unsigned emit (cycle-101 deferred);
SHR signed-vs-unsigned emit (cycle-101 deferred);
raw-200 enumeration in parser.hx Stage-8 monomorphize_pass
(cycle-7 deferred);
Stage 29 K2 SIGILL probes (untracked `_probe_stage29_*.py`);
DT_BIND_NOW unused constant in `elf_dyn.py`;
evaluator.hx tag table covers only Stage-3 evaluator subset;
all prior cycle findings already addressed.

---

## Cycle-108 fix-sweep verification (regressions from the sweep itself)

Verified at HEAD `c89432e`:

- `x86_64.py:1290-1291` (`from_is_i64 = _is_64bit_int_type(...)`),
  `x86_64.py:1782` (SELECT `is_i64`), `x86_64.py:1882` (CALL int-arg),
  `x86_64.py:1899` (CALL return), `x86_64.py:2008` (RETURN),
  `x86_64.py:2045` (BR block-param), `x86_64.py:2087` (LOAD_VAR),
  `x86_64.py:2107` (STORE_VAR) — all now route u64/usize through the
  `_is_64bit_int_type` width gate. Each fix-site reads as expected;
  the dispatch order in CAST correctly places the new
  unsigned-widening arm (line 1305-1309) BEFORE the i32→i64 movsxd
  arm (line 1311-1316), so u32→u64 zero-extends rather than
  sign-extends. PASS.
- `lower_ast.py:1061-1077` adds explicit `raise NotImplementedError`
  arms for `A.CharLit`, `A.StructLit` (in expr position), `A.TileLit`
  (in expr position). All three arms read `expr.span.line`/`.col` —
  `A.Expr`'s base dataclass declares `span: Span` as required
  (`ast_nodes.py:99-101`), so the access cannot raise AttributeError
  on a well-formed AST. PASS.
- Stage 29.1 cap bumps in `kovc.hx`: `patch_table` 4096→16384 (line
  1583, while-loop bound 12288→49152 at line 1586; cap-check 4096→
  16384 at line 1598); `bind_state` 64→512 (line 988 while-loop bound
  256→2048; line 1013 cap-check 64→512). Arithmetic relations
  (entries × slots-per-entry; cap × bytes-per-binding) are
  self-consistent. PASS at the bound-arithmetic level — but see
  Finding F1 below for the consequence at the prologue layer.

The cycle-108 sweep itself introduces no new silent failures within
the IR/backend predicates it touched. **F1 below is the cascading
consequence of the Stage 29.1 cap bump alone, not of the cycle-108
fix sweep.**

---

## Findings

### F1 — Stage 29.1 `bind_state` cap bump (64→512) outruns the fixed 1024-byte prologue, silently corrupting parent frame at runtime (CRITICAL, conf 90)

**File**: `helixc/bootstrap/kovc.hx`, `bind_init` at lines 978-993,
`bind_alloc_offset` at lines 1043-1057, `emit_prologue` at lines
739-745.

```hx
// bind_init (post cap bump)
let mut i: i32 = 0;
while i < 2048 {                        // 512 entries * 4 slots
    __arena_push(0);
    i = i + 1;
}

// bind_alloc_offset (UNCHANGED in stage 29.1)
let off = __arena_get(state);
if off >= 1024 {
    emit_trap_with_id(10030);
};
__arena_set(state, off + 8);
off

// emit_prologue (UNCHANGED in stage 29.1)
//   48 81 EC 00 04 00 00   sub rsp, 1024
emit_byte(0x48); emit_byte(0x81); emit_byte(0xEC);
emit_u32_le(1024);
```

The Stage 29.1 commit bumped `bind_state`'s entry cap from 64 to 512
(line 988 / 1013) WITHOUT bumping `emit_prologue`'s `sub rsp` constant
(line 743) or `bind_alloc_offset`'s 1024-byte trap threshold (line
1052). The pre-bump arithmetic was:

- 64 bindings × 8-byte slot = 512 bytes of stack
- `bind_alloc_offset` trap fires at 1024 bytes (128 bindings)
- `emit_prologue` allocates 1024 bytes (128 slots)
- Headroom margin: 64 extra slots (per comment at line 4713)

The post-bump arithmetic is:

- 512 bindings × 8-byte slot = **4096 bytes of stack needed**
- `bind_alloc_offset` trap fires at the same 1024 bytes (still 128
  bindings — UNCHANGED)
- `emit_prologue` still allocates 1024 bytes — UNCHANGED
- **The cap-bump claims "Empirically the bootstrap source needs ~200
  bindings per fn at peak" (line 986) — but 200 × 8 = 1600 bytes >
  1024 byte prologue, so any fn with >128 simultaneously-live
  bindings is now reachable at the bind_state layer but blows the
  prologue at the codegen layer.**

The defect mode is worse than a clean overflow:
`bind_alloc_offset` (line 1052-1054) emits a `trap_with_id(10030)`
instruction **inline at the current emit cursor** AND then continues
(`__arena_set(state, off + 8); off`), returning the out-of-range
offset to its caller. The caller then emits `mov [rbp - off], rax`
with `off >= 1024`, writing past the prologue's stack allocation
into the parent frame's saved `rbp` / saved return address / red
zone. The codegen *continues normally* after emitting the trap
instruction — so the emitted binary contains both a trap *and*
ABI-violating writes, interleaved in the wrong basic block (the trap
fires whichever way control reaches the spot, but the wrong-frame
writes are emitted regardless of whether the trap is reached). The
combination is "silent failure plus loud failure plus wrong-frame
corruption" — symptoms include:

1. K2 (kovc.hx-by-K1 self-compile) emits trap instructions sprinkled
   through `emit_ast_code`'s body when the inner Helix code has >128
   live let-bindings. The cycle-108 commit comment "K2 currently
   returns 0 bytes from emit_elf_for_ast_to_path" is consistent with
   trap-instruction-mid-emit being how the codegen sees a failed
   prologue.
2. Stage 29.1's claim "K3 exits 42" passes ONLY because the test
   source happens to stay under 128 live bindings per fn. The
   bootstrap source itself has `emit_ast_code` with 334 lexical lets
   spread across nested if-branches; bind_pop reclaims sequentially
   but the worst-case live depth is unmeasured and the comment
   claims ~200 (i.e. > 128).
3. Any helixc test that exercises a fn with ≥128 live let-bindings
   silently miscompiles via wrong-frame writes (the runtime trap
   only fires the FIRST time the offending fn is called; the
   wrong-frame writes happen even on the success path of branches
   that don't reach the trap-instruction point).

**Reachability**: The Stage 29.1 commit specifically un-skips
`test_bootstrap_kovc_self_host_loop` (`test_codegen.py:4793` —
diff at `a600616..c89432e -- helixc/tests/test_codegen.py`). The
self-host loop puts kovc.hx (which contains `emit_ast_code`'s 334
lets) on the critical path. The Stage 29.1 commit message confirms
the patch_table cap-overflow class of bug ("emit count jumped from
~1500 to ~6800 patches, overflowing the 4096 cap and dropping ~2700
patches") — exactly the kind of cap-and-consumer asymmetry that
this finding flags for the bind_state side.

**Recommended fix**: bump `emit_prologue` `sub rsp` constant from
1024 to at least 4096 (or, better, parameterize on the per-fn live-
binding-count emitted by a frame-size-resolver pre-pass) AND bump
`bind_alloc_offset`'s 1024 trap threshold to match. Alternatively
shrink the `bind_state` cap-bump back to 128 entries with an
explicit fail-loud spill path for sources hitting it.

**Confidence**: 90. The arithmetic is exact (512 × 8 > 1024); the
codegen-time trap-but-continue behavior in `bind_alloc_offset` is
exactly as `emit_trap_with_id(10030)` reads (the trap emit is
unconditional within the if-branch but does not early-return); the
1024-byte prologue is hard-coded at line 743 with no Stage-29.1
update.

---

### F2 — IR lowering `A.Range` arm silently returns None, caller substitutes `const_int(0)` (HIGH, conf 92)

**File**: `helixc/ir/lower_ast.py`, line 2006-2007.

```python
if isinstance(expr, A.Range):
    return None
```

Cycle-108's F8 fix added loud-fail `raise NotImplementedError` arms
for `A.CharLit`, `A.StructLit`, `A.TileLit`. It also left the bottom
catch-all `return None` at line 2268 untouched (for the deferred-
known A.StrLit case). But the **explicit** silent-return arm for
`A.Range` at line 2006-2007 is even more clearly wrong: it is an
explicit `return None` arm rather than a catch-all, so it's
unambiguous that lowering knows about A.Range and has no
implementation.

Typecheck at `typecheck.py:1706-1711` accepts `A.Range` in any
expression position and returns `TyUnknown(hint="range")`. So
`let r = 0..10;` typechecks. The IR lowering for the let-stmt at
`lower_ast.py:1031-1033` (LetStmt path) calls `_lower_expr(stmt.value)`
which returns None for the Range expression, and then substitutes
`self.builder.const_int(0)`. The binding `r` is now bound to the
constant 0 — silent miscompile of the Range value (which the user
may have intended as some iterator handle / typed pair).

A `for i in 0..10 { ... }` use is safe because the For arm at
`lower_ast.py:1820` short-circuits the A.Range inspection BEFORE
calling _lower_expr. Same special-case pattern as the StructLit
let-stmt short-circuit at line 848. But any non-For-iter use of
A.Range (e.g. `let r = 0..n; for i in r {}`, or `f(0..10)` as a
call arg) routes through `_lower_expr`'s explicit return-None and
silently folds to 0.

**Reachability**: `A.Range` constructors are reachable from the
parser (range expression syntax `a..b`, parsed at parser.py).
Cycle-108 F8 closed the let-stmt-typed CharLit/StructLit/TileLit
silent-fold class explicitly. A.Range remains as the **explicit**
(not catch-all) silent-return arm, two lines above the loud-fail
arms cycle-108 already added. This is the same defect class as F8
but with an even clearer fix (the silent arm is explicit, not
catch-all).

**Recommended fix**: replace lines 2006-2007 with

```python
if isinstance(expr, A.Range):
    raise NotImplementedError(
        f"range expression in non-For-iter position not yet "
        f"supported in IR lowering at "
        f"{expr.span.line}:{expr.span.col}")
```

mirroring the cycle-108-pattern for CharLit/StructLit/TileLit.

**Confidence**: 92. The silent-return is explicit and visible; the
cycle-108 fix introduced the loud-fail pattern *in the same file
two screens above* but missed this sibling arm; `typecheck.py:1706`
confirms the path is reachable.

---

### F3 — `cast<u32, f64>` / `cast<u32, f32>` silently use signed-int conversion path (HIGH, conf 88)

**File**: `helixc/backend/x86_64.py`, CAST emit at lines 1273-1359;
relevant arms at lines 1330-1334 (i32→f64) and lines 1336-1340
(i32→f32).

```python
# i32 -> f64
if not from_is_float and to_is_f64:
    self.asm.mov_eax_mem_rbp(src_slot)
    self.asm.cvtsi2sd_xmm0_eax()           # SIGNED conversion
    self.asm.movsd_mem_rbp_xmm0(res_slot)
    return
# i32 -> f32
if not from_is_float and to_is_float:
    self.asm.mov_eax_mem_rbp(src_slot)
    self.asm.cvtsi2ss_xmm0_eax()           # SIGNED conversion
    self.asm.movss_mem_rbp_xmm0(res_slot)
    return
```

Cycle-108 F7 fix added the predicate-extension to `_is_64bit_int_type`
for u64/usize, plus an unsigned-widening arm before the i32→i64
movsxd arm. But the int→float arms at lines 1330-1340 still use
`cvtsi2sd` / `cvtsi2ss` — **signed** 32-bit-int-to-float opcodes.
For an `as f64` cast from a u32 source, the dispatch hits these
arms (from_is_float=False, from_is_i64=False since u32 isn't
64-bit, to_is_f64=True). x86 has no single-instruction unsigned-
int→float conversion; the canonical sequence is `mov eax, [src]`
(implicit zero-extend to rax) + `cvtsi2sd xmm0, rax` (REX.W).

For a u32 with the high bit set (e.g. 0xFFFFFFFF = u32::MAX), the
signed reading is -1, so `cvtsi2sd_xmm0_eax` produces -1.0. The
correct unsigned value is ~4.29e9. **Silent miscompile.**

Symmetric defects for `cast<u64, f64>` / `cast<u64, f32>`:
Post-cycle-108, dispatch routes to the i64→f64 arm at lines
1318-1323, which emits `cvtsi2sd xmm0, rax` (REX.W signed
conversion). For u64 with bit 63 set (e.g. 0x8000000000000000),
this reads as -2^63 and produces ~-9.22e18; correct unsigned
value is ~9.22e18. The cycle-108 commit comment at line 1287-1288
acknowledges the *pre-fix* `cast<u64, f64>` defect ("fell through
to the i32->float arms — silent low-32-only signed-int read") but
the post-fix dispatch still uses *signed* int→float opcodes —
the bug class is moved one rung up the dispatch table, not
eliminated.

**Reachability**: any `as f32` / `as f64` cast from a u8/u16/u32/
u64/usize source. The typecheck-hint at `typecheck.py:1799`
("use a bitcast through a u64 intermediate") routes users into
the CAST path. Any helixc-Python test that runtime-exit-codes the
result of a u32→f64 cast with high-bit-set source would surface
this; the existing byte-pattern tests do not.

**Recommended fix**: add explicit unsigned-int→float arms BEFORE
the i32→f-arm / i64→f-arm dispatch, gated on
`_is_unsigned_int_type(from_ty)`. The x86 sequence for u32→f64 is
`mov eax, [src]` (zero-extends to rax) + `cvtsi2sd xmm0, rax`
(REX.W signed-64 conversion on a guaranteed-non-negative 64-bit
value gives the unsigned interpretation). The u64→f64 case
requires a longer "split high/low + scale" sequence; alternatively
emit a runtime helper call. Regression test: `u32(0xFFFFFFFF) as
f64` and check the returned f64 against ~4.29e9 (not -1.0).

**Confidence**: 88. The CAST dispatch table is explicit at lines
1273-1359; `cvtsi2sd_xmm0_eax` is signed by x86 ISA spec; cycle-108
F7 did NOT touch these arms (the diff at `b0e18c7..a600616 --
helixc/backend/x86_64.py` shows the int→float arms unchanged).

---

### F4 — `BIT_AND` / `BIT_OR` / `BIT_XOR` / `SHL` / `BIT_NOT` / `NEG` u64/usize width-gates silently truncate to 32 bits (HIGH, conf 92)

**File**: `helixc/backend/x86_64.py`, BIT_AND at lines 1515-1529,
BIT_OR at lines 1530-1544, BIT_XOR at lines 1545-1559, SHL at lines
1560-1574, BIT_NOT at lines 1590-1601, NEG at lines 1602-1643.

```python
if op.kind == tir.OpKind.BIT_AND:
    ...
    if self._is_i64_type(op.results[0].ty):
        self.asm.mov_rax_mem_rbp(l_slot)
        self.asm.mov_rcx_mem_rbp(r_slot)
        self.asm.and_rax_rcx()
        self.asm.mov_mem_rbp_rax(res_slot)
    else:
        self.asm.mov_eax_mem_rbp(l_slot)
        ...
        self.asm.and_eax_ecx()
        self.asm.mov_mem_rbp_eax(res_slot)
    return
```

Every one of BIT_AND / BIT_OR / BIT_XOR / SHL / BIT_NOT / NEG uses
`_is_i64_type` (which only matches i64/isize) as the wide-int gate.
For a u64/usize result, the else-branch fires — 32-bit load via
`mov eax/ecx, [...]`, 32-bit op via `and_eax_ecx`/`or_eax_ecx`/
`shl_eax_cl`/`not_eax`/`neg_eax`, and 32-bit store via
`mov [...], eax`. The high 4 bytes of both operand slots are
dropped at load time; the result slot's high 4 bytes are left
stale.

This is the EXACT defect class that cycle-100 (cmp), cycle-101
(setcc), cycle-102 (ADD/SUB/MUL), cycle-106 (CONST_INT, param-spill,
BITCAST classifier), and cycle-108 (CALL int-arg, CALL return,
RETURN, SELECT, BR, LOAD_VAR, STORE_VAR, CAST) all closed for
their respective sibling sites. The cycle-107 audit's deferred-
known list explicitly listed BIT_* / SHL / BIT_NOT / NEG as
"deferred and not re-flagged"; the current cycle-109 prompt
explicitly re-includes them in scope ("DIV/MOD/SHR signed-vs-
unsigned being deferred but BIT_* not").

**Reachability**: any `u64` or `usize` typed value participating in
a bitwise/shift/negation op. `fn mask_u64(x: u64) -> u64 { x &
0xFFFF_FFFF_0000_0000_u64 }` — the AND mask itself is u64, ADD
result is u64 (cycle-102-correct), but the BIT_AND emit truncates
both operands to 32 bits at load time. The 0xFFFF_FFFF_0000_0000_u64
literal's load through `mov ecx, [r_slot]` reads only the LOW 32
bits (= 0x0000_0000), so the result becomes `x & 0` = 0. **Silent
miscompile** — the user-visible result is 0, not a high-half-masked
value.

Sibling existence: each of these ops also has a deferred-known
unsigned-cmp issue for the shifts (cycle-101's deferred SHR signed-
vs-unsigned); the AND/OR/XOR don't have a signed/unsigned semantic
difference, so a pure width-gate fix closes them. SHL and BIT_NOT
also don't have signed/unsigned ambiguity at the ISA level (left-
shift is identical for signed/unsigned modulo overflow behaviour).
NEG on unsigned is a defined two's-complement op.

**Recommended fix**: at each of the 6 ops, replace
`self._is_i64_type(op.results[0].ty)` (or `_is_i64_type(ty)` for
NEG/BIT_NOT) with `self._is_64bit_int_type(...)`. The text edit is
one-line each, mirroring the cycle-108 F1-F7 fix pattern exactly.
Regression tests: `fn mask_u64(x: u64) -> u64 { x &
0xFFFF_FFFF_0000_0000_u64 }` checked against an exit-code low/high
half pair.

**Confidence**: 92. The defect is uniform across all six op kinds;
the predicate name-equality miss is exactly the same one cycle-100
through cycle-108 have closed in batches; the test surface is the
same byte-pattern + runtime-exit-code combination that has caught
this defect class at every prior cycle.

---

## Sub-threshold observations (informational, NOT counted toward verdict)

**O1** (MEDIUM, conf 70): The Stage 29.1 `patch_table_add` cap-check
returns `0 - 1` to its caller (line 1599) when the 16384 cap is hit.
ALL call sites of `patch_table_add` (~20 sites grepped in `kovc.hx`)
discard the return value. The cap-bumped 16384 vs measured ~6800 gives
2.4x headroom so the cap is unlikely to be hit in current bootstrap;
but the pattern of "cap-check returns -1, caller ignores" is preserved
verbatim from pre-cycle-29.1 code. This is a deferred-known pattern,
not a new regression — but it is the same silent-failure class as F1's
"cap-but-continue" defect.

**O2** (MEDIUM, conf 65): The `bind_push_typed` cap-check at
`kovc.hx:1013` returns `0 - 1` to its caller when cap (now 512) is
hit. Two-of-three `bind_push_typed` call sites grepped in kovc.hx
discard the return value. Same silent-failure pattern as O1. Mitigated
by F1's CRITICAL: the prologue runs out 4x before the bind_state cap
in practice.

**O3** (LOW, conf 55): The cycle-108 `_lower_expr` arms for CharLit /
StructLit / TileLit raise `NotImplementedError` — but the LetStmt path
at `lower_ast.py:1031-1033` (cycle-108's caller-side) catches
`NotImplementedError` only at the `_lower_expr` dispatch level. If
NotImplementedError propagates out of `_lower_stmt`, the helixc-Python
driver does NOT have a top-level handler that converts it to a clean
typecheck-error-style diagnostic. So the user gets a Python traceback
instead of a `helixc: error: char literal not yet supported at
line:col` message. Not a silent failure (loud-fail-as-traceback is
better than silent-substitution), just suboptimal UX.

---

## Summary

**4 findings at confidence ≥ 80%**, of three related root classes:

- **F1** (CRITICAL conf 90): Stage 29.1 cap-bump asymmetry — the
  `bind_state` cap was raised 64→512 but `bind_alloc_offset`'s 1024-
  byte trap threshold and `emit_prologue`'s 1024-byte `sub rsp` were
  not updated. Any fn with >128 live let-bindings silently corrupts
  parent frame via wrong-disp writes; `bind_alloc_offset`'s
  "trap-but-continue" codegen pattern emits inline trap instructions
  AND wrong-frame writes. The cap-bump commit message claims ~200
  bindings/fn at peak — i.e. unconditionally beyond the prologue.

- **F2** (HIGH conf 92): `A.Range` IR-lowering arm at
  `lower_ast.py:2006-2007` is an explicit silent `return None` arm,
  TWO LINES from the cycle-108 NotImplementedError pattern for
  StructLit/TileLit. Caller substitutes `const_int(0)`; non-For-iter
  Range expressions silently fold to 0.

- **F3** (HIGH conf 88): `cast<u32, f64>` / `cast<u32, f32>` /
  `cast<u64, f64>` / `cast<u64, f32>` use signed `cvtsi2sd`/
  `cvtsi2ss` opcodes for unsigned-int sources. u32 high-bit-set
  values silently cast to negative floats. Cycle-108 F7's predicate-
  extension and unsigned-widening arm closed the u32→u64 case but
  did NOT touch the int→float arms, where the same signed/unsigned
  asymmetry exists.

- **F4** (HIGH conf 92): BIT_AND / BIT_OR / BIT_XOR / SHL / BIT_NOT
  / NEG all use `_is_i64_type` (i64/isize only) as the wide-int
  gate — same defect class as F1-F7 of cycle-107. Was deferred-
  known at cycle-107 but explicitly in scope at cycle-109 per the
  prompt's "BIT_* not deferred" clarification. Each is a one-line
  `_is_i64_type → _is_64bit_int_type` swap.

The cycle-108 fix-sweep itself is correct (verification section
above); F1-F4 are sibling sites the cycle-108 sweep did not touch
(F2/F3/F4) or cascading consequences of the Stage 29.1 cap bump
(F1) that landed AFTER cycle-108's commit.

---

## Findings table

| Severity | Count |
|---|---|
| CRITICAL (conf ≥ 90) | 1 (F1) |
| HIGH (conf ≥ 75)     | 3 (F2 92, F3 88, F4 92) |
| MEDIUM (60-74)       | 0 |
| LOW (<60)            | 0 |
| Sub-threshold (info) | 3 (O1, O2, O3) |

---

## Verdict

**Verdict: FINDINGS** — 4 HIGH/CRITICAL items (1 CRITICAL + 3 HIGH at
confidence ≥ 80). Counter remains at 0/5.

---

## Cross-references

- Cycle-108 fix-sweep commit `a600616`: closed CALL int-arg / CALL
  return / RETURN / SELECT / BR / LOAD_VAR / STORE_VAR / CAST u64
  width-gate (cycle-107 F1-F7) and added explicit loud-fail arms for
  CharLit / StructLit / TileLit (cycle-107 F8). F2 of this cycle is
  the A.Range sibling that the cycle-108 catch-all did not address;
  F3 is the unsigned-int→float dispatch arm that the cycle-108 F7
  predicate-extension did not address; F4 is the BIT_*/SHL/BIT_NOT/
  NEG sibling sites the cycle-108 sweep did not touch.
- Stage 29.1 cap-bump commit `c89432e`: bumped `patch_table` 4096→
  16384 and `bind_state` 64→512. F1 of this audit is the cascading
  prologue-layer consequence the cap-bump did not address.
- Source-of-truth files: `helixc/backend/x86_64.py` (predicates at
  lines 1037-1075, CAST 1273-1359, BIT_AND 1515-1529, BIT_OR 1530-
  1544, BIT_XOR 1545-1559, SHL 1560-1574, BIT_NOT 1590-1601, NEG
  1602-1643); `helixc/ir/lower_ast.py:2006-2007` (A.Range silent
  return None); `helixc/bootstrap/kovc.hx` (`bind_init` 978-993,
  `bind_alloc_offset` 1043-1057, `emit_prologue` 739-745,
  `patch_table_init` 1582-1591, `patch_table_add` 1593-1609).
