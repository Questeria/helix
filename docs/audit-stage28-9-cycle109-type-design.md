# Audit Stage 28.9 cycle 109 — Type design

## Header

- **Date**: 2026-05-12
- **HEAD**: `c89432e` ("🎉 STAGE 29.1: bumped patch_table +
  bind_state caps → K3 exits 42!").
- **Prior fix-sweep**: `a600616` ("Stage 28.9 cycle-108 fix-sweep:
  8 cycle-107 silent-failure findings F1-F8") closing the
  cycle-107 CALL/RETURN/SELECT/BR/LOAD_VAR/STORE_VAR/CAST
  `_is_64bit_int_type` sweep plus CharLit/StructLit/TileLit
  loud-fail IR lowering arms.
- **Counter at start**: 0/5.
- **Mode**: STRICT READ-ONLY on `helixc/`. The only file written
  by this audit is this document. No Edit calls.
- **Scope**: cycle-109 type-design rotation surfaces — seven
  verification points covering the cycle-108 fix-sweep delta,
  the Stage 29.1 cap bumps, and rotation surfaces explicitly
  flagged in the cycle-109 brief:
  1. Post-cycle-108 `_is_64bit_int_type` predicate asymmetry
     vs sibling sites still on `_is_i64_type` (BIT_*, SHL, SHR,
     NEG, BIT_NOT, DIV, MOD).
  2. New unsigned-widening CAST arm (`x86_64.py:1297-1309`)
     ordering and predicate coverage.
  3. LoadVar / StoreVar 8-byte path vs `_alloc_slot` /
     `_alloc_var` slot-size consistency.
  4. CharLit / StructLit / TileLit loud-fail arms — type-design
     subclass coverage vs bottom catchall `return None`.
  5. kovc.hx patch_table + bind_state cap bumps (Stage 29.1):
     consistency across reader / writer / sibling cap sites.
  6. TyUnit / TyPrim("()") normalization (cycle-106 follow-up).
  7. Parser.hx named-mode generic branches (lines 3217-3504,
     3505-3658).
- **Deferred-known and NOT re-flagged** (cycle-107 list,
  unchanged):
  - monomorphize `_mangle_ty` / hash_cons `_ast_equal` silent
    catchalls
  - typecheck/struct_mono pre-flatten in `check.py`
  - `autotune.collect_autotuned_fns` missing `iter_fn_decls`
  - `struct_mono.mangle_struct` collision
  - DT_BIND_NOW unused constant
  - raw-200 enumeration in parser.hx (cycle-7 deferred)
  - `_is_i64_type` sibling emit-site enumeration
    (BIT_AND / BIT_OR / BIT_XOR / SHL / SHR / BIT_NOT / NEG /
    DIV / MOD) — cycle-101 deferred. **NB:** the cycle-108
    sweep closed SELECT / BR / RETURN / COND_BR / FFI_CALL /
    CALL / LOAD_VAR / STORE_VAR / CAST. The cross-site
    INTERACTION between cycle-108-promoted and still-deferred
    sites is a new behaviour pattern; see V1 for the type-design
    observation (not flagged as fresh — it is the same defect
    class as cycle-101).
  - `A.StrLit` IR lowering gap (cycle-95 deferred).
- **Bar**: PASS = ZERO new HIGH findings at confidence ≥ 75%.
  MED observations recorded but do not count toward verdict.
- **Source delta since cycle-107**:
  - `a600616` cycle-108 fix-sweep: `helixc/backend/x86_64.py`
    (+82 / -11), `helixc/ir/lower_ast.py` (+33), and regression
    tests `helixc/tests/test_ir.py` (+188).
  - `c89432e` Stage 29.1: `helixc/bootstrap/kovc.hx` cap bumps
    (patch_table 4096→16384, bind_state 64→512) plus test
    un-skip in `helixc/tests/test_codegen.py`.

## Methodology

Seven verification points across the cycle-108 fix-sweep delta
+ Stage 29.1 cap bumps + a rotation through type-design surfaces
not recently audited. Each point is a targeted re-walk: read the
relevant module(s), trace every dispatch path through the
surface language, cross-check against the deferred-known list
to ensure candidate defects are fresh, and verify any new
predicate / arm against the rest of the type-class universe.

## V1 — `_is_64bit_int_type` cycle-108 sweep asymmetry (PASS, with HIGH observation)

### Sweep coverage post cycle-108

The cycle-108 fix-sweep extended the 8-byte emit path from
`_is_i64_type` (i64/isize) to `_is_64bit_int_type` (i64/isize +
u64/usize) at the following sites:

| Site | Line | Status |
|------|------|--------|
| Param spill | 1004 | Cycle-108 F1 sibling (already on `_is_64bit_int_type`) |
| CONST_INT | 1222 | Cycle-106 (already promoted) |
| BITCAST | 1263-1265 | Cycle-106 (already promoted) |
| ADD | 1395 | Cycle-102 (already promoted) |
| SUB | 1425 | Cycle-102 (already promoted) |
| MUL | 1453 | Cycle-102 (already promoted) |
| CAST from_is_i64 / to_is_i64 | 1290-1291 | Cycle-108 F7 (promoted) |
| CAST unsigned-widening arm | 1305-1309 | Cycle-108 F7 (new arm) |
| SELECT | 1786 | Cycle-108 F4 (promoted) |
| BR block-param copy | 2042 | Cycle-108 F5 (promoted) |
| CALL int-arg | 1848 | Cycle-108 F1 (promoted) |
| CALL return store | 1861 | Cycle-108 F2 (promoted) |
| RETURN | 1962 | Cycle-108 F3 (promoted) |
| LOAD_VAR | 2087 | Cycle-108 F6 (promoted) |
| STORE_VAR | 2105 (sibling of 2087) | Cycle-108 F6 (promoted) |

### Sites still on `_is_i64_type`

The following remain gated on `_is_i64_type` (i64/isize only),
so a u64/usize operand silently falls through to the 32-bit
truncating else branch:

| Site | Line | Predicate |
|------|------|-----------|
| DIV | 1484 | `_is_i64_type(op.results[0].ty)` |
| MOD | 1499 | `_is_i64_type(op.results[0].ty)` |
| BIT_AND | 1519 | `_is_i64_type(op.results[0].ty)` |
| BIT_OR | 1534 | `_is_i64_type(op.results[0].ty)` |
| BIT_XOR | 1549 | `_is_i64_type(op.results[0].ty)` |
| SHL | 1564 | `_is_i64_type(op.results[0].ty)` |
| SHR | 1579 | `_is_i64_type(op.results[0].ty)` |
| BIT_NOT | 1593 | `_is_i64_type(op.results[0].ty)` |
| NEG | 1606 | `_is_i64_type(ty)` (operand-typed) |

All nine sites are in the cycle-101 deferred-known list (which
the cycle-107 audit re-stated verbatim). The rotation prompt
explicitly asks: "after cycle-108 swept ... does the predicate
now have asymmetric treatment of signed-vs-unsigned 64-bit ints
in remaining sites (BIT_*, SHL, SHR, NEG, DIV, MOD)?" — the
answer is yes, by construction, because the cycle-108 sweep
was scoped to F1-F7 sites only.

### Cross-site interaction (HIGH observation, NOT flagged as fresh)

The cycle-108 sweep creates an interesting NEW observable
behaviour pattern in code that mixes promoted and unpromoted
sites on the same u64 value:

```helix
fn f(x: u64, y: u64) -> u64 {
    let z = x & y;       // BIT_AND — 32-bit emit (deferred site)
    z                    // RETURN — 64-bit emit (cycle-108 F3)
}
```

Pre cycle-108: BIT_AND emitted 32-bit `mov eax + and + mov`
writing low 4 bytes of `z`'s slot. RETURN emitted 32-bit `mov
eax, [z]` reading low 4 bytes. Whole pipeline truncated to 32
bits but CONSISTENTLY: high 4 bytes were never read or written.

Post cycle-108: BIT_AND still emits 32-bit, writing only low 4
bytes of `z`'s slot. The high 4 bytes are STALE STACK GARBAGE
(no zero-init at `_alloc_slot` / prologue `sub rsp, frame`).
RETURN now emits 64-bit `mov rax, [z]` reading all 8 bytes.
**Caller observes `(stale_high_garbage << 32) | (low32 & low32)`
— a non-deterministic miscompile.**

The same pattern applies to: CALL int-arg on a u64 result of
BIT_*/SHL/SHR/NEG/BIT_NOT/DIV/MOD; LOAD_VAR/STORE_VAR on a `let
mut x: u64` after a BIT_* op writes it; BR block-param copy
where one arm's u64 was computed via BIT_*; SELECT result
flowing into the new 64-bit SELECT arm.

This is the **same defect class** as the cycle-101 deferred-known
list — the sites themselves are unchanged. But the **observable
manifestation** has shifted from deterministic-truncation to
non-deterministic-garbage in the high half. Pre cycle-108 a
property test that printed `z` as decimal would see the same
truncated value every run; post cycle-108 it would see a
different value each run depending on what was on the stack
before the function call.

**Severity assessment**: this is observably worse (harder to
debug, breaks property tests, reduces compile-time-stability
of the produced binary), BUT it is mechanically the same
deferred-known defect class. Audit discipline says deferred-known
items are not re-flagged.

**Recommended next-cycle fix-sweep** (recorded here, not
prescribed): a full `_is_64bit_int_type` promotion at DIV, MOD,
BIT_AND, BIT_OR, BIT_XOR, SHL, SHR, BIT_NOT, NEG. ADD/SUB/MUL/
CAST have already been promoted with a sign-agnostic rationale
(machine opcodes for ADD/SUB/IMUL low-half are signed-vs-unsigned
identical at the bit level). The same rationale applies to
BIT_AND, BIT_OR, BIT_XOR (sign-agnostic bitwise). SHL is
sign-agnostic. **SHR is NOT** sign-agnostic — the current 32-bit
emit uses `sar` (arithmetic shift), correct for signed-32 but
wrong for u32 SHR (which needs `shr`). This is a strict superset
of the deferred-known and was noted as a defense-in-depth gap in
cycle-101.

Confidence as a present-miscompile vector: ~85%. Confidence as
a HIGH fresh finding: ~50% (downgraded by the deferred-known
disposition — the rotation prompt specifically asks the question
but the underlying defect class is on the deferred list). Per
audit-discipline, **not flagged as F1**; recorded as a HIGH
type-design observation O109-1.

### V1 verdict: PASS (with HIGH observation O109-1)

The cycle-108 sweep is internally consistent at the sites it
touched. The interaction-pattern between promoted and unpromoted
sites is observably worse than pre-cycle-108 but mechanically
unchanged (same site list as cycle-101 deferred-known).

## V2 — Unsigned-widening CAST arm ordering and coverage (PASS)

### Arm placement

The new arms at `x86_64.py:1305-1309` (unsigned-widening
zero-extend) and `1311-1316` (signed-widening sign-extend) are
in the order specified by the comment ("Must fire BEFORE the
i32→i64 movsxd arm below"). Verified by direct line read.

### Predicate dispatch matrix

For every reachable source-target pair under the Phase-0 surface
language:

| From | To | Arm fired | Emission | Correct? |
|------|----|----|----------|----------|
| u8 / u16 / u32 | u64 / i64 | 1305 (unsigned) | `mov eax + mov rax` (zero-extend) | ✓ |
| i8 / i16 / i32 | u64 / i64 | 1311 (signed-widen) | `mov eax + movsxd rax, eax` | ✓ |
| bool | u64 / i64 | 1311 (signed-widen) | sign-extend | by coincidence (bool ∈ {0,1}, high bit = 0) |
| char | u64 / i64 | 1311 (signed-widen) | sign-extend | by coincidence (valid Unicode scalar ≤ 0x10FFFF, high bit = 0) |
| u64 / usize | u64 / i64 / usize | 1325 | 8-byte mov-copy | ✓ |
| i64 / isize | u64 / i64 / isize | 1325 | 8-byte mov-copy | ✓ |
| u64 / usize / i64 / isize | i32 / u32 / etc. | 1293 (i64→i32 truncate) | 4-byte mov | ✓ (low 32 bits) |

The `bool → u64` and `char → u64` cases route through the
sign-extend arm rather than the zero-extend arm. The result is
identical in practice because:
- bool slot's low 32 bits ∈ {0, 1} (CONST_BOOL at line 1232
  writes `mov_eax_imm32(1 or 0)` to the slot); high bit of the
  32-bit value is 0, so movsxd zero-extends.
- char values are valid Unicode scalars ≤ 0x10FFFF (21 bits);
  high bit of the 32-bit slot value is 0, so movsxd zero-extends.

These two cases are **design-wise asymmetric** (a bool/char
*should* zero-extend, semantically) but **practically correct**.
Recorded as observation O109-2; below the HIGH bar.

### `_is_unsigned_int_type` predicate set

`x86_64.py:1072-1074`: `{u8, u16, u32, u64, usize}`. Excludes
bool and char. Consistent with cycle-100 CMP fix scope; the
omission is the design choice that makes O109-2 a question
rather than a finding.

### Slot pre-state vs widening semantics

For source slots, the 4-byte load `mov eax, [src_slot]`
implicitly zero-extends to rax on x86-64. The HIGH 4 bytes of
the source slot are NOT read by this instruction. So even if
the source slot's high 4 bytes are stack garbage (which they
are for u8/u16/u32 since CONST_INT / param-spill / 32-bit ops
only write the low 4 bytes), the zero-extension into rax is
hardware-guaranteed. The store `mov [res_slot], rax` then
writes all 8 bytes of the destination with the correctly
zero-extended value. **Correct under all stack-pre-state
conditions.**

### V2 verdict: PASS

The new unsigned-widening CAST arm is correctly ordered (1305
before 1311), its predicate `_is_unsigned_int_type` covers the
needed source types, and the zero-extension semantics are
hardware-guaranteed regardless of slot pre-state. bool/char
falling through to the sign-extend arm is practically correct
under the Phase-0 surface (recorded as O109-2).

## V3 — LoadVar/StoreVar 8-byte slot allocation (PASS)

### `_alloc_slot` and `_alloc_var` slot sizes

`x86_64.py:907-911` (`_alloc_slot`): `self.next_slot -= 8`.
Every SSA value's slot is 8 bytes regardless of TIR type.

`x86_64.py:888-893` (`_alloc_var`): `self.next_slot -= 8`. Every
mutable-variable slot is 8 bytes regardless of declared type.

So a `let mut x: u64` and a `let mut x: i32` both allocate 8
bytes. The cycle-108 F6 fix (LOAD_VAR/STORE_VAR on u64/usize
takes the 8-byte path) is therefore safe: the slot has the
required 8 bytes of space, no risk of overlap with adjacent
slots.

### Slot overlap risk

`next_slot` starts at 0 and decrements by 8 per allocation. No
slot ever overlaps with another. The cycle-108 F6 8-byte
load/store on a u64 var slot reads/writes the slot's full 8
bytes without touching neighbouring slots. **No silent clobber.**

### `_alloc_array` element-size consistency

`x86_64.py:895-905` (`_alloc_array`): `length * elem_size` with
`elem_size = 8` default. Hard-coded to 8 bytes per element.
Consistent with the value-slot size. No array/scalar slot-size
mismatch.

### Stack-frame alignment

`frame_size = (-self.next_slot + 15) & ~15` (line 949): the
frame is rounded up to a 16-byte boundary. SysV ABI requires
16-byte stack alignment at function call boundaries. The
prologue's `sub rsp, frame_size` preserves this. No alignment
hazard introduced by the 8-byte slot scheme.

### V3 verdict: PASS

Slot allocation is uniformly 8 bytes per value / var / array
element. The cycle-108 F6 8-byte load/store path is safe by
construction. No silent overlap or alignment hazard.

## V4 — CharLit / StructLit / TileLit loud-fail arm coverage (PASS)

### Cycle-108 F8 arms

`lower_ast.py:1061-1077`: explicit `A.CharLit`, `A.StructLit`,
`A.TileLit` arms each raise `NotImplementedError` with a
position-bearing diagnostic. Placed near the TOP of `_lower_expr`
(line ~1037 onwards), BEFORE the `A.Name` / `A.Path` / `A.Binary`
/ etc. branches. So any reaching of `_lower_expr` with these
nodes hits the loud-fail before any other dispatch.

### A.Expr subclass enumeration vs `_lower_expr` arms

Enumerating `A.Expr` subclasses from `frontend/ast_nodes.py`:

```
IntLit, FloatLit, StrLit, CharLit, BoolLit, Name, Path, Unary,
Binary, Call, Index, Field, TupleLit, ArrayLit, StructLit,
Block, UnsafeBlock, If, Match, For, While, Loop, Break,
Continue, Return, Range, Assign, Cast, TileLit, Quote, Splice,
Modify
```

Coverage in `_lower_expr` (grep `isinstance(expr, A.\w+)`):

```
IntLit ✓ FloatLit ✓ BoolLit ✓ CharLit ✓ (loud-fail) StructLit
✓ (loud-fail) TileLit ✓ (loud-fail) Name ✓ Path ✓ Binary ✓
Unary ✓ Call ✓ If ✓ UnsafeBlock ✓ Block ✓ For ✓ While ✓ Break
✓ (loud-fail) Continue ✓ (loud-fail) Loop ✓ Match ✓ Return ✓
Range ✓ Assign ✓ TupleLit ✓ ArrayLit ✓ Index ✓ Field ✓ Cast ✓
Quote ✓ Splice ✓ Modify ✓
```

**Only `A.StrLit` is missing.** It hits the bottom catch-all
`return None` at line 2268. Cycle-108 commit message explicitly
notes: "The bottom `return None` is preserved for A.StrLit
(deferred-known under cycle-101 F1 contract)." This is the
chosen design for the deferred-known item; the silent-failure
audit (not this type-design audit) tracks it.

### Type-design subclass appropriateness

`A.CharLit` / `A.StructLit` / `A.TileLit` are the right subclasses
to loud-fail because:
- They have constructor positions (let-RHS / call-arg / if-arm
  / return / assign-RHS) that the parser accepts.
- The typechecker passes them through without erroring.
- Pre-fix the catchall `return None` produced a TIR value of
  None which got coerced to `const_int(0)` by the surrounding
  arms, silently miscompiling the program.

`A.StrLit` is in the same defect class but is explicitly held
under the cycle-95 / cycle-101 F1 deferred-known contract — its
proper fix requires implementing string-literal IR lowering,
not just a loud-fail arm.

### V4 verdict: PASS

The cycle-108 F8 loud-fail arms are placed at the right
subclasses (CharLit / StructLit / TileLit) and at the right
position in `_lower_expr` dispatch. The bottom catch-all
preservation for A.StrLit matches the deferred-known contract.

## V5 — Stage 29.1 kovc.hx cap bumps (PASS, with HIGH observation)

### patch_table cap bump

Bumped from 4096 → 16384 entries (12288 → 49152 arena slots,
stride 3).

**Writer site**: `patch_table_add` at line 1593-1607. The cap
check at line 1597 was bumped to `if top >= 16384` — consistent
with the new cap. Returns -1 on overflow (loud-ish: callers
that ignore the return value would still silently corrupt, but
the cap is now 16384 which is well above the measured 6800
patches).

**Reader sites**: 13 call sites of `patch_table_add` were
greppe — all pass the same `patch_state` handle and stride-3
entry shape. None hard-code the old 4096 cap.

**The backpatch reader loop** at lines 6548-6571 iterates `pi < patch_top` where `patch_top = __arena_get(patch_state)`. Uses the ACTUAL count, not the cap. Consistent with the bumped cap.

**Initializer**: `patch_table_init` at line 1582-1591 pushes
`49152` arena slots (= `16384 * 3`). Matches the new cap.

**Comment / docstring drift**: line 1594 still says "patch_table_init allocates 16384 entries; without this guard, a source with > 16384 CALL+LEA patches would silently corrupt adjacent arena memory." Both numbers updated consistently. The header comment at lines 1574-1581 documents the rationale. **No drift.**

### bind_state cap bump

Bumped from 64 → 512 entries (256 → 2048 arena slots, stride 4).

**Writer site**: `bind_push_typed` at line 1009-1025. The cap
check at line 1013 was bumped to `if top >= 512` — consistent
with the new cap. Returns -1 on overflow (loud-ish on the
caller side; bind_push_typed callers ignore the return value
in most cases, so failed pushes silently lose the binding —
but at cap 512 with measured peak ~200, this is unreachable
under the Phase-0 surface).

**Initializer**: `bind_init` at line 978-993 pushes 2048 arena
slots (= 512 * 4). Matches the new cap.

**Reader sites**: `bind_lookup` (line 1059) and `bind_lookup_type`
(line 1087) iterate `top - 1 down to 0`. Use the ACTUAL count
`top = __arena_get(state + 1)`, not the cap. Consistent.

### CAP COORDINATION INVARIANT VIOLATION (HIGH observation O109-3)

The kovc.hx comments at lines 733-738 explicitly document a
cap-coordination invariant:

> // Audit fix (cycle 1, polish #14): bumped from 512 → 1024
> // to match the bind_state cap (64 entries) with 2× margin.
> // Previously 512 was "just enough" for 64 × 8-byte slots —
> // any future cap bump would silently corrupt the saved
> // rbp/return-address. 1024 gives 128 slots; future Phase-1
> // should derive this from bind_state cap dynamically rather
> // than hard-coding.

This invariant was: `prologue_frame_bytes ≥ bind_state_cap *
8`. Pre Stage 29.1: 1024 ≥ 64 * 8 = 512 ✓ (2× margin). Post
Stage 29.1: 1024 ≥ 512 * 8 = 4096 ✗ (4× SHORTFALL).

The actual STACK-OFFSET cap is enforced separately at
`bind_alloc_offset` (line 1043-1057) which traps with id 10030
when `off >= 1024`. So if a function tries to allocate more
than 128 simultaneously-live slots, the COMPILED BINARY contains
a ud2 sequence with trap id 10030 at the over-cap site.

**Critical type-design observation**: the trap is emitted in the
OUTPUT binary at compile time, but `bind_alloc_offset` does NOT
short-circuit — it sets `__arena_set(state, off + 8)` and returns
the over-cap offset. The downstream codegen then uses this
returned offset to emit a `mov [rbp - over_cap_offset], ...` that
writes past the saved-rbp boundary INTO THE PARENT FRAME. So the
produced binary's behaviour is:

1. Run reaches the over-cap function.
2. The function's prologue runs (`sub rsp, 1024`).
3. Codegen emitted both a trap-id 10030 ud2 AND the over-cap
   memory stores.
4. **Depending on which comes first in the function body**,
   either the trap fires first (SIGILL, loud), or the
   memory-store-into-saved-rbp executes first (silent corruption).

Looking at `bind_alloc_offset`'s body: line 1053 calls
`emit_trap_with_id(10030)` BEFORE line 1055 `__arena_set(state,
off + 8)`. So in the OUTPUT binary, the trap precedes any code
that uses the over-cap offset (assuming the codegen always
emits-then-uses, which is the standard pattern). So the trap
should always fire BEFORE any over-cap memory store. Loud
failure dominates.

**BUT**: the trap is in the output binary at the FIRST
over-cap allocation point. Subsequent allocations in the same
function would re-trap. The CALLER of `bind_alloc_offset`
proceeds with the over-cap offset and emits the memory-store,
which would be UNREACHABLE (the trap-id 10030 ud2 fires first).
So the binary is correctly fail-loud.

**The invariant violation remains** in the sense that the
documented invariant (`bind_state_cap * 8 ≤ prologue_frame`) is
now violated by a factor of 4. The runtime trap saves the binary
from silent corruption, but the documented design intent has
shifted from "compile-time impossibility" to "runtime trap".
This is a TYPE-DESIGN regression: a compile-time-static cap
relationship was broken without updating the corresponding
prologue allocation.

**Reachability under bootstrap**: per the Stage 29.1 commit, K3
exits 42 — i.e., at least one self-host path produces a working
binary. So under the bootstrap's heaviest function (parse_primary
with ~200 bindings/fn per the commit), the LIFO `bind_pop`
roll-back keeps SIMULTANEOUSLY-LIVE offsets ≤ 128. The
binding-table (`bind_state`'s 512 entries) records the names but
nested scopes reuse offsets. So the cap bump is consistent with
how bindings flow in practice; it is the *documented invariant
about caps* that broke, not necessarily the produced binary.

**Severity**: HIGH (the documented invariant explicitly warned
"any future cap bump would silently corrupt the saved
rbp/return-address" and Stage 29.1 did exactly that bump without
updating the prologue). Confidence 80%. The loud-trap mechanism
in `bind_alloc_offset` mitigates the silent-corruption risk but
the cap-coordination contract is broken.

**Flagged as F109-1** below.

### offset-arithmetic precondition check (PASS sub-component)

Both caps are `16384` and `512` — `16384 = 2^14`, `512 = 2^9`.
Both are powers of 2. Any offset arithmetic assuming
power-of-2 (e.g., bitmask % cap) would still work. No
precondition broken on that side.

### V5 verdict: PASS for patch_table; HIGH F109-1 for bind_state cap-coordination

The patch_table cap bump is consistent across init / write /
read / cap-check / commentary. The bind_state ENTRY-table cap
bump is similarly consistent on its own, BUT the cap-coordination
invariant with `emit_prologue`'s 1024-byte allocation is broken
(documented invariant warned about this exact regression).

## V6 — TyUnit / TyPrim("()") normalization (PASS)

Re-grep'd the entire `helixc/` tree for `TyPrim("()")` and
`TyName("()")`:

```
TyPrim('()')  → tests/test_ir.py:393  (docstring describing pre-fix defect)
TyName('()')  → frontend/typecheck.py:344-345  (comments only)
```

No new producers introduced by cycle-108 or Stage 29.1. The
cycle-106 fix and cycle-107 V1 audit findings are unchanged.

### V6 verdict: PASS

No new TyPrim("()") construction site. The cycle-106 fix
remains complete.

## V7 — Parser.hx named-mode generic branches (PASS, unchanged)

The two branches at lines 3217-3504 (generic mode) and
3505-3658 (non-generic mode) were not touched by cycle-108 or
Stage 29.1 (the kovc.hx delta is at lines 980-995 / 1009-1014
/ 1572-1607, all far from the parser.hx surface). The cycle-107
V4 audit on these branches concluded "PASS — structurally
consistent under Stage 28.13; the mono-clone preserves the
fields-region layout exactly". That conclusion is unchanged
because the source is unchanged.

### V7 verdict: PASS (no source delta since cycle-107 audit)

## Findings table

| ID | Severity | Confidence | Topic | Disposition |
|----|----------|------------|-------|-------------|
| F109-1 | HIGH | 80% | bind_state cap-coordination invariant violated by Stage 29.1: cap bumped 64→512 (factor of 8) without bumping `emit_prologue`'s 1024-byte allocation. Documented invariant at kovc.hx:733-738 warned "any future cap bump would silently corrupt the saved rbp/return-address. 1024 gives 128 slots; future Phase-1 should derive this from bind_state cap dynamically rather than hard-coding." Stage 29.1 broke the cap-coordination contract by a factor of 4 (1024 vs 512×8=4096). Runtime trap-id 10030 at `bind_alloc_offset` (line 1052-1053) saves the binary from silent corruption — but the type-design contract was that the cap should be COMPILE-TIME-STATIC ≤ prologue_frame, not enforced via runtime trap. K3 exits 42 under Stage 29.1 because bootstrap functions use LIFO `bind_pop` to reuse offsets; the 512 cap allows 512 distinct *names* but bind_alloc_offset still caps simultaneous LIVE offsets at 128. The cap-coordination invariant is in name-vs-offset asymmetry. Recommended fix: bump `emit_prologue`'s `sub rsp, 1024` to `sub rsp, 4096` (or derive dynamically from `bind_state` cap as the original audit-fix comment suggested) AND raise the trap-cap in `bind_alloc_offset` to match. Or document explicitly that the bind_state cap and the stack-offset cap are now intentionally decoupled. |

## Observations (not flagged; below the HIGH bar)

| ID | Severity | Confidence | Topic |
|----|----------|------------|-------|
| O109-1 | HIGH-OBS | 85% | Post-cycle-108 cross-site interaction between promoted (`_is_64bit_int_type`) and unpromoted (`_is_i64_type`) emit sites converts deterministic-u64-truncation (pre cycle-108) to non-deterministic-stack-garbage in high half (post cycle-108). The site list (DIV / MOD / BIT_AND / BIT_OR / BIT_XOR / SHL / SHR / BIT_NOT / NEG) is the cycle-101 deferred-known. The cycle-108 partial sweep makes the manifestation observably worse but does not change the site list. Recommended: next-cycle full-sweep promotion (sign-agnostic predicates for the bitwise / shift family; SHR needs unsigned `shr` opcode dispatch for u-family — strict superset of just `_is_64bit_int_type` promotion). |
| O109-2 | LOW-OBS | 60% | `cast<bool, u64>` and `cast<char, u64>` route through the sign-extend arm (1311) rather than the unsigned-widening zero-extend arm (1305) because `_is_unsigned_int_type` does not include bool or char. Practically correct under the Phase-0 surface (bool ∈ {0,1}, valid char ≤ 0x10FFFF — high bit always 0 so movsxd zero-extends in practice). Design-wise asymmetric. Defense-in-depth fix: add bool and char to `_is_unsigned_int_type`, OR explicitly route them through the zero-extend arm. Below 75% as a finding — no observable miscompile under Phase-0 surface. |

## Verdict

**Verdict: FINDINGS** — 1 HIGH (F109-1) at confidence 80%.
Counter resets 0/5 → 0/5 per audit-gate discipline (a HIGH
finding at conf ≥ 75% blocks counter advance).

- V1 PASS-with-observation: cycle-108 sweep is internally
  consistent at promoted sites; cross-site interaction with
  cycle-101 deferred-known is observably worse but not a fresh
  finding (same site list). Recorded as O109-1.
- V2 PASS: new unsigned-widening CAST arm is correctly ordered
  and covers the needed predicates; bool/char fallthrough to
  sign-extend is practically correct (O109-2).
- V3 PASS: slot allocation is uniformly 8 bytes; no overlap
  or alignment hazard for the cycle-108 F6 LOAD_VAR / STORE_VAR
  8-byte path.
- V4 PASS: cycle-108 F8 CharLit / StructLit / TileLit loud-fail
  arms are placed at the right subclasses and the right
  dispatch position; A.StrLit remains the deferred-known silent
  catch-all.
- V5 FINDINGS: patch_table cap bump is internally consistent;
  bind_state cap bump broke the documented cap-coordination
  invariant with `emit_prologue` (F109-1). Runtime trap-id 10030
  is the safety net but the type-design contract is regressed.
- V6 PASS: no new TyPrim("()") construction site.
- V7 PASS: parser.hx named-mode generic branches unchanged
  since cycle-107 audit.

**Stage 28.9 audit-gate counter unchanged at 0/5** (HIGH finding
blocks advance). Cycle-110 should fix F109-1 (either bump
`emit_prologue`'s frame, derive it dynamically from bind_state
cap, or formally decouple bind_state from bind_alloc_offset
caps in documentation) before re-attempting counter advance.

No edits to source performed. This document is the only file
written by cycle-109.

## Cross-reference to cycles 101-108

- **Cycle 101**: PASS, 0 findings. _is_i64_type sibling sites
  noted as deferred.
- **Cycle 102**: fix-sweep, 4 cycle-101 findings (ADD/SUB/MUL
  u64).
- **Cycle 103**: PASS, 0 findings.
- **Cycle 104**: PASS, 0 findings.
- **Cycle 105**: FAIL, 1 finding F105-1 (f64↔f32 CAST silent).
- **Cycle 106**: fix-sweep, 4+ cycle-105 findings + cross-cut.
- **Cycle 107**: PASS, 0 findings on cycle-106 delta. Counter
  0 → 1.
- **Cycle 107 silent-failures**: FAIL, 8 findings F1-F8.
- **Cycle 108**: fix-sweep, all 8 cycle-107 silent-failure
  findings closed. CALL/RETURN/SELECT/BR/LOAD_VAR/STORE_VAR/
  CAST promoted to `_is_64bit_int_type`. CharLit/StructLit/
  TileLit loud-fail arms added.
- **Cycle 109** (this doc): FINDINGS, 1 HIGH F109-1 on Stage
  29.1 bind_state cap-coordination invariant. Counter unchanged
  at 0/5.
