# Audit Stage 28.9 cycle 111 — Code review

- **Date**: 2026-05-12
- **HEAD**: `07e6535` ("Stage 30 cycle-3 MEDIUM fix: add regression test for trap 62033")
- **Audit-target HEAD per task brief**: `f9425a0` ("Stage 30 cycle-2 IMPORTANT fix: regression tests for traps 62032/62033"). The repository has since advanced by one Stage 30 cycle-3 follow-up commit (`07e6535`, adding the trap 62033 regression test). This audit verifies the full range `9c451e6..07e6535` so the verdict reflects what is actually on `main` at audit time. The cycle-3 commit is a strict supplement to cycle-2 (one extra assertion, no production code change), so the cycle-2 verdict and the cycle-3-extended verdict are identical with one extra positive observation.
- **Counter at start**: 0/5 per `f9425a0`'s commit message ("Cycle-2 IS NOW CLEAN. Counter: 1/5"). Per the cycle-110 fix-sweep commit `9c451e6`'s post-state ("Cycle: cycle 109 FAIL → reset to 0/5. Cycle 111 next."), counter at the start of cycle 111 is 0/5; the cycle-2 1/5 referenced a separate Stage 30 counter, not the Stage 28.9 cycle-level counter this codereview is gating.
- **Scope** (per task brief rotation surfaces 1–7):
  1. **Cycle-110 fix-sweep `9c451e6`** — review (a) u32→float zero-extend arm placement in the CAST emit if-chain; (b) BIT_*/SHL/BIT_NOT/NEG promotion consistency across the six emit functions; (c) A.Range NotImplementedError message informativeness; (d) the 11 new regression tests for true discriminativity (same defect class as cycle-109 F1/F2).
  2. **Stage 30 cycle-2 `1aecbae`** — kovc.hx prologue bump comments + the `kovc.hx:1050` stale `1024` reference disposition.
  3. **Stage 30 cycle-2 `f9425a0`** — regression tests for traps 62032/62033: (a) cross-OS-environment exit-132 stability; (b) reusability of the test pattern; (c) trap-id discrimination (62032 vs 62033 confused under exit-status-only check).
  4. **Cycle-110 regression test cross-check** — pick 3 of 11 new tests, evaluate discriminativity via mental-revert.
  5. **kovc.hx trap-id collision audit** — verify 10031/10032/10033 (cycle-110) and 62032/62033 (Stage 30 cycle-2) don't collide with existing trap-ids; numbering-convention check.
  6. **Test-suite hygiene** — orphan tests, `pytest.skip`, `xfail` added by cycle-110 fix-sweep.
  7. **Docstring/comment drift** — spot-check 2–3 cycle-110 new tests for docstring-vs-implementation match (cycle-109 F1/F2 precedent).
- **Bar**: confidence ≥ 75, severity HIGH (75–89) or CRITICAL (90–100) only. Below-bar items are recorded as sub-threshold observations.
- **Deferred-known list** (carried forward, NOT re-flagged):
  - All cycle-109 deferred-known items.
  - `BIT_AND` / `BIT_OR` / `BIT_XOR` / `SHL` / `BIT_NOT` / `NEG` `_is_i64_type` sibling emit paths — **now PROMOTED by cycle-110 F4**, removed from deferred list.
  - `SHR` `_is_i64_type`-only — explicitly retained per cycle-110 commit message and cycle-101 F2 deferred-known disposition (`sar` is signed-only; an unsigned variant `shr` was not implemented this cycle).
  - DIV / MOD signed-vs-unsigned emit — still deferred-known.
  - `A.StrLit` IR-lowering catch-all `return None` — still deferred-known per cycle-101 F1.
  - `CAST f32↔i64` / `f64↔i64` widening arms — still missing per cycle-109 sub-threshold observation.
  - `patch_table_add` / `bind_push_typed` / `fn_table_add` silent-failure pattern — **now PROMOTED to loud-fail (trap 10031/10032/10033) by cycle-110**, removed from deferred list (cycle-109 F3 closed).
  - `bind_alloc_offset` trap threshold off-by-one (margin collapse: cap effectively 511 of 512 simultaneously-live slots) — Stage 30 cycle-3 LOW-conf-78 deferred. Recorded sub-threshold below.
  - `bind_pop` comment at kovc.hx:1052-1053 references "512-byte prologue allocation" — stale per the 4096-byte bump. Cycle-2 LOW deferred per `1aecbae` commit message ("LOW: stale 'blowing past the 512-byte prologue' reference at kovc.hx:1050 ... the comment hasn't caught up").

---

## Methodology

Read-only inspection. No source mutation, no test execution, no scorecard run. Single Write of this document. Cross-referenced files:

- `helixc/backend/x86_64.py:1037-1074` (predicate helpers `_is_i64_type`, `_is_u64_type`, `_is_64bit_int_type`, `_is_unsigned_int_type`).
- `helixc/backend/x86_64.py:1273-1392` (CAST cascade — cycle-108 F7 + cycle-110 F3 arm placement).
- `helixc/backend/x86_64.py:1538-1678` (BIT_AND / BIT_OR / BIT_XOR / SHL / SHR / BIT_NOT / NEG emit — cycle-110 F4 promotion).
- `helixc/ir/lower_ast.py:1042-1077` (cycle-108 F8 loud-fail arms — sibling pattern reference).
- `helixc/ir/lower_ast.py:2006-2020` (cycle-110 F2 A.Range loud-fail arm).
- `helixc/ir/lower_ast.py:1173-1188` (unary lowering — `-` → NEG, `~` → BIT_NOT, `!` → CMP_EQ).
- `helixc/tests/test_ir.py:589-830` (11 new cycle-110 regression tests: c109-F1/F2 strengthening + c110 SF-F2/F3/F4 sweep).
- `helixc/bootstrap/kovc.hx:721-758` (`emit_prologue` — cycle-110 1024→4096 bump).
- `helixc/bootstrap/kovc.hx:983-1046` (`bind_init` / `bind_push` / `bind_push_typed` — cycle-110 cap-check + loud-trap 10032).
- `helixc/bootstrap/kovc.hx:1064-1083` (`bind_alloc_offset` — cycle-110 1024→4096 trap threshold).
- `helixc/bootstrap/kovc.hx:1048-1062` (`bind_pop` — comment stale per 1aecbae commit message).
- `helixc/bootstrap/kovc.hx:1560-1580` (`fn_table_add` — cycle-110 loud-trap 10033).
- `helixc/bootstrap/kovc.hx:1626-1640` (`patch_table_add` — cycle-110 loud-trap 10031).
- `helixc/bootstrap/parser.hx:3240-3320` (cycle-2 H1 sentinel pattern wiring — traps 62032/62033 mk_node payloads).
- `helixc/tests/test_codegen.py:2889-2912` (cycle-2 + cycle-3 regression tests for traps 62032/62033).
- `helixc/tests/test_codegen.py:3583-3620` (`bind_alloc_offset` F11 test — many_lets→132 dropped, fewer_lets bumped 64→256).
- `helixc/tests/test_codegen.py:4895-4925` (Stage 29 K2 SIGILL comment cleanup at 1aecbae).
- Full grep `emit_trap_with_id` across `helixc/bootstrap/kovc.hx` — 88 occurrences, full id-list enumerated below for collision audit.

---

## Findings table

| Severity | Count |
|----------|-------|
| CRITICAL (90–100) | 0 |
| HIGH (80–89)      | 1 |
| MED (50–79)       | 0 reportable (below bar) |
| LOW (<50)         | 0 reportable (below bar) |

---

## Findings

### F1 (HIGH conf 82) — `test_c110_neg_u64_via_sub_emits_64bit_form` exercises SUB (already cycle-102), not NEG; the cycle-110 F4 NEG predicate promotion has zero regression coverage

**File**: `helixc/tests/test_ir.py:814-830`
**Production code**: `helixc/backend/x86_64.py:1637-1645` (the F4 fix — NEG predicate flip `_is_i64_type` → `_is_64bit_int_type`).
**IR lowering**: `helixc/ir/lower_ast.py:1173-1178` (unary `-` lowers to `tir.OpKind.NEG`, NOT to SUB).

**Defect class**: vacuous test / docstring-vs-assertion drift / missing-regression-coverage. Direct repeat of cycle-109 F1/F2 precedent.

The cycle-110 fix-sweep promotes six emit arms from `_is_i64_type` to `_is_64bit_int_type`: BIT_AND, BIT_OR, BIT_XOR, SHL, BIT_NOT, **NEG**. The commit message claims "Regression tests added (test_ir.py) — discriminative byte-pattern assertions" and 11 tests follow. Five of the six new u64 bitwise/unary tests are discriminative; **NEG is the exception**.

The test `test_c110_neg_u64_via_sub_emits_64bit_form` at test_ir.py:814-830 uses the source:

```
fn neg_u64(a: u64) -> u64 { 0_u64 - a }
fn main() -> i32 { 0 }
```

and asserts `b"\x48\x29\xc8"` (= `sub rax, rcx`) is present in the ELF. The docstring states:

> u64 negation via `0_u64 - a` must use the 64-bit SUB path (48 29 C8 sub rax, rcx). The unary `-` on unsigned types is rare in source code (lowering uses SUB for unsigned-domain "negation"); this still exercises the u64 wide path that F4 promoted.

**Two errors in this docstring**:

1. **Misrepresentation of lowering**: the docstring claims "lowering uses SUB for unsigned-domain 'negation'". This is false. The actual lowering at `lower_ast.py:1173-1178` emits `tir.OpKind.NEG` for every `A.Unary` with `op == "-"`, regardless of operand sign:
   ```python
   if isinstance(expr, A.Unary):
       inner = self._lower_expr(expr.operand)
       if inner is None: return None
       if expr.op == "-":
           return self.builder.emit(tir.OpKind.NEG, inner, result_ty=inner.ty)
   ```
   The actual test source `0_u64 - a` is `A.BinaryOp(op="-", left=A.IntLit(0_u64), right=A.Name("a"))`, lowering to `SUB`, not `NEG`.

2. **Misrepresentation of what F4 promoted**: SUB is NOT in the cycle-110 F4 set. SUB was promoted to `_is_64bit_int_type` by **cycle-102** (per the cycle-102 commit message "ADD/SUB/MUL u64" and the cycle-109 audit's deferred-known cross-check). The test's discriminator (`48 29 C8`) fires regardless of whether cycle-110 F4 NEG promotion is applied or reverted, because the SUB emission has been on the wide path since cycle-102.

**Trace with NEG predicate reverted** (line 1641 `_is_64bit_int_type` → `_is_i64_type`):

- CONST_INT `0_u64`: cycle-106 path, emits `mov rax, imm64` (48 B8 00...00) + `mov [rbp+slot], rax` (48 89 45).
- `a` param-spill (cycle-106): `mov [rbp+a_slot], rdi` (48 89 7D).
- LOAD_VAR a (cycle-108 F6, still applied): `mov rax, [rbp+a_slot]` (48 8B 45).
- SUB u64 (cycle-102, untouched by cycle-110): `mov rax, [l]; mov rcx, [r]; sub rax, rcx; mov [res], rax` — emits `48 29 C8`.
- RETURN u64 (cycle-108 F3, still applied): `mov rax, [res]` (48 8B 45) + `ret`.

The byte sequence `48 29 C8` is unambiguously produced by the SUB u64 emit at `x86_64.py:1450-1456` (cycle-102 commit). **No NEG opcode is emitted at all** because the source has no unary `-`. Therefore reverting the NEG predicate at line 1641 leaves the test green.

**Why this matters**: NEG u64 IS reachable from real source. `fn neg_u64(a: u64) -> u64 { -a }` lowers to NEG u64, and the typecheck does not reject unary `-` on u64 (no rejection rule found in `helixc/check.py`). The bootstrap `kovc.hx:5253-5264` itself contains the codegen comment "u64 NEG semantically = 2^64 - x; REX.W neg rax computes that" — confirming u64 NEG is part of the language surface. Pre-cycle-110 the Python helixc backend would silently truncate u64 NEG to 32-bit `neg eax`, miscompiling `-x_u64` for any `x` with bits above the low 32 set. The cycle-110 fix closes this, but the regression test does not pin it.

**A future revert** of `_is_64bit_int_type` to `_is_i64_type` at `x86_64.py:1641` would not be caught by any test. The cycle-110 NEG promotion is structurally a HIGH-conf production fix with **zero** regression coverage — the same defect class the cycle-109 codereview F1/F2 flagged.

**Suggested fix**: replace the body of `test_c110_neg_u64_via_sub_emits_64bit_form` with a true NEG-exercising source:

```python
def test_c110_neg_u64_emits_64bit_form():
    """C109-SF-F4 NEG sibling: `-a` where a: u64 must emit `neg rax`
    (48 F7 D8). Pre-fix `_is_i64_type` excluded u64 so unary `-` on
    u64 silently truncated to `neg eax` (F7 D8 no REX)."""
    from helixc.backend.x86_64 import compile_module_to_elf
    src = """
    fn neg_u64(a: u64) -> u64 { -a }
    fn main() -> i32 { 0 }
    """
    elf = compile_module_to_elf(lower_src(src))
    assert b"\x48\xf7\xd8" in elf, (...)
```

The byte sequence `48 F7 D8` (`neg rax`) is unique to the NEG-u64 path and not emitted by SUB, ADD, MUL, or any other arm.

**Confidence**: 82. Mechanical: SUB is on `_is_64bit_int_type` since cycle-102 (line 1450); NEG predicate flip at line 1641 is the cycle-110 contribution; the test uses a source that emits SUB, not NEG. Conf below 85 because the test, while not pinning NEG specifically, does exercise the SUB wide path and would catch a hypothetical revert of cycle-102's SUB promotion — so the test isn't strictly orphan; it's just mis-labelled. Conf above 80 because the cycle-109 codereview explicitly flagged docstring-vs-assertion drift as HIGH (F2 at conf 85), and this is a stricter case: the docstring contains technically false claims about the lowering pipeline (claiming lowering uses SUB for unary negation when it actually uses NEG).

---

## Sub-threshold observations (NOT findings — below the 75-confidence bar)

- **F11 `many_lets` SIGILL test was DROPPED by cycle-110, replaced by one-sided coverage** (conf ~70). At `test_codegen.py:3583-3614` the cycle-110 1aecbae diff removed the assertion `assert compile_and_exec(many_lets) == 132` (140 chained lets → trap 10030 SIGILL) and bumped the `fewer_lets` test from 64 to 256 lets. The remaining test only verifies that 256 lets *fit* under the new budget; it does not exercise the new 4096-byte trap path. The asymmetric coverage IS discriminative for a downward revert of the prologue (256 lets would trap under the old 1024-byte budget) but does NOT catch an in-place revert of the `if off >= 4096` check itself (removing the trap leaves the test green; the trap would simply not fire). Documented in the cycle-110 commit message: "directly testing the new boundary (525+ lets) currently runs into a pre-existing bootstrap output-emission issue at very large source sizes (bisected: crash above ~362 lets, unrelated to cycle-110)". Acknowledged constraint, not a regression-introducing decision. Sub-threshold.

- **`bind_alloc_offset` off-by-one** (conf ~72, Stage 30 cycle-3 LOW deferred). The trap fires at `off >= 4096`. The reserved frame `sub rsp, 4096` covers `[rbp-4096, rbp-1]` inclusive — 4096 bytes — supporting 512 slots at offsets 8, 16, ..., 4096. The 512th slot (off=4096) is `[rbp-4096, rbp-4089]`, fully within budget. Current trap collapses the effective cap to 511 simultaneously-live bindings. Margin loss: 1 of 512 slots (~0.2%). Cycle-3 commit acknowledged this as a LOW conf 78 deferred item. Sub-threshold for cycle 111 by symmetry.

- **`bind_pop` comment at `kovc.hx:1052-1053` is stale** (conf ~85 PASS-as-doc-only). The comment reads "blowing past the 512-byte prologue allocation; emit_mov_local_eax(-560) writes into the parent frame's saved rbp/return-address". After cycle-110 the prologue is 4096 bytes; the example `-560` byte-offset is no longer the failure boundary. The `1aecbae` commit message explicitly flagged this as the "LOW: stale 'blowing past the 512-byte prologue' reference at kovc.hx:1050" and deferred. Doc-only — no behavioral consequence — sub-threshold per cycle-109 convention (doc-only stale references are not HIGH unless they affect a reader's ability to debug a real failure mode).

- **Cycle-2/cycle-3 trap-62032 vs trap-62033 tests are non-discriminative on trap-id** (conf ~72). The 3 regression tests at `test_codegen.py:2889-2912` assert `compile_and_exec(...) == 132` (SIGILL exit code). They distinguish "trap fires" from "trap doesn't fire" but not 62032 from 62033 or from any other SIGILL-causing trap (e.g., 28999 cap-overflow, 10030 prologue overrun). A future regression that re-routes trap 62033 → 62032 (or any other id) would leave the tests green. Cycle-3 acknowledged this concern at conf 78 ("the cycle-2 regression tests covered trap 62032 (arity-mismatch) twice but never exercised the 62033 (bad-token-in-args) path of the same sentinel mechanism") and resolved it partially by adding a third test for the bad-token path; but the underlying trap-id discrimination is still missing. Sub-threshold because: (i) the practical regression class to catch is the H1 sentinel-vs-return wire-up, which exit-status=132 does catch; (ii) full trap-id discrimination would require inspecting the trapping instruction's preceding `mov eax, imm32`, which the current test harness doesn't expose; (iii) cycle-3 commit already accepted this as a known limit.

- **Cross-environment exit-132 stability** (conf ~80 PASS). The cycle-2/cycle-3 tests assume exit code 132 = SIGILL. This is correct on Linux (128 + signal_num, SIGILL=4). The test harness runs the binary via `wsl -e bash -c ...` so the runtime environment is consistent Linux glibc. The task brief asked whether other OSes / glibc versions could exit with 11/6/9. SIGILL specifically (the `ud2` instruction at `emit_trap_with_id`) reliably produces SIGILL on Linux x86-64; the wait()-status-to-exit-code translation `128 + signum` is glibc convention adopted by bash. Inside WSL the same applies. The tests are NOT cross-OS portable (Windows native or macOS would behave differently) but the test harness is WSL-only and the assertion is correct for that environment. Not a finding.

- **Trap-id collision audit clean** (conf ~95 PASS). Full enumeration of `emit_trap_with_id` calls in `helixc/bootstrap/kovc.hx` yields the following ids: 1001, 2001/10/11/20/21/30/31/40/41, 3001/10/11/20/21/30/31/40/41, 4001/10/11/20/21/30/31/40/41, 5001/10/11/20/21/30/31/40/41/50/51, 6001/10/11/20/21/30/31/40/41/50/51, 8001-8016, 9001, 14001/02, 16001/02/03, 19001/10/11/20/21/30/31/40/41/50/51, 20001/10/11/20/21/30/31/40/41, 21001/10/11/20/21/30/31/40/41, 22001/10/11/20/21/30/31/40/41/50/51, 23001/10/11/20/21/30/31/40/41/50/51, 24001/10/11/20/21/30/31/40/41/50/51, 26001, 27002, 28020/21, 28999, 29020/21, 30020/21, 31001, 32001/10/40, 33001/10/40, 42002, 52001, 60030, 62001, 81002, 99001, **and the cycle-110 additions 10030 (pre-existing), 10031 (new — patch_table_add), 10032 (new — bind_push_typed), 10033 (new — fn_table_add)**. The new 10031/10032/10033 are sequential after pre-existing 10030 and do not collide. The 62032/62033 added by Stage 30 cycle-2 are `mk_node(99, N, ...)` AST_ERR payloads in `parser.hx`, NOT direct `emit_trap_with_id(N)` calls — they ride the existing AST_ERR codegen path that emits `mov eax, imm32; ud2`. No direct collision with kovc.hx trap ids. Numbering convention: there is no strict `AST_TAG * 1000 + sub_id` convention as the task brief speculated; rather the codebase uses scheme-by-namespace (low 1000s for AST-tag-paired traps; 10030+ for cap-overflow class; 60xxx for parser; 99xxx for placeholder). 10031/10032/10033 fit the 10030 cap-overflow namespace correctly. PASS.

- **Test-suite hygiene clean** (conf ~95 PASS). Diff `c89432e..07e6535` (entire cycle-110 + Stage 30 cycle 1-3 range) introduces zero new `pytest.skip` / `pytest.xfail` / `@pytest.mark.skip` markers. The two existing `pytest.skip` lines at `test_codegen.py:4829` and `:14121` predate cycle-110. No orphan test files added. The 11 new tests at `test_ir.py:589-830` are all properly registered in the `test_*` naming scheme picked up by the local `main()` runner at line 833. PASS.

- **CAST cycle-110 F3 arm placement vs cycle-108 F7 unsigned-widening arm** (conf ~90 PASS). The cycle-108 F7 arm at `x86_64.py:1305-1309` handles `u32/u16/u8 → u64/i64` (zero-extend). The cycle-110 F3 arms at `x86_64.py:1340-1357` handle `u32/u16/u8 → f64/f32` (zero-extend via eax then REX.W cvtsi2sd/ss). The placement is correct: F7 fires when `to_is_i64`; F3 fires when `to_is_f64` or `to_is_float`. Both arms guard `not from_is_i64 and self._is_unsigned_int_type(from_ty)` so they exclude u64/usize sources (which route through the `from_is_i64 and to_is_f64` arm at line 1318 or fall to the `i64→i64` 8-byte mov-copy at 1325). The F3 arms fire BEFORE the i32→f64/f32 arms at lines 1359/1365, so u32 with the high bit set zero-extends to a positive 64-bit signed integer before conversion, producing the correct unsigned-interpretation float. PASS.

- **BIT_*/SHL/BIT_NOT/NEG promotion symmetry** (conf ~92 PASS). The six arms at x86_64.py:1545 (BIT_AND), 1570 (BIT_OR), 1585 (BIT_XOR), 1599 (SHL), 1628 (BIT_NOT), 1641 (NEG) all now use `self._is_64bit_int_type(...)` consistently. SHR at 1614 remains `self._is_i64_type(...)` per cycle-101 F2 deferred-known (the `sar` instruction is sign-arithmetic; a signedness-correct dispatch would need a separate `shr rax, cl` arm). The promotion symmetry is intentional and matches the cycle-110 commit message's carve-out. PASS.

- **A.Range NotImplementedError message informativeness** (conf ~90 PASS). The cycle-110 F2 arm at `lower_ast.py:2017-2020` raises with `f"range expression in non-For-iter position not yet supported in IR lowering at {expr.span.line}:{expr.span.col}"`. The message includes:
  1. **What construct**: "range expression in non-For-iter position".
  2. **Why it's failing**: "not yet supported in IR lowering" (implementation-gap, not user-error).
  3. **Where it's in the source**: `{line}:{col}` span coords (Span at `ast_nodes.py:19-21` has `line: int` and `col: int`).
  4. **Sibling pattern**: matches cycle-108 F8's `lower_ast.py:1061-1077` style verbatim (same span-anchored format).
  PASS. Downstream debugging will surface the line:col immediately.

- **F1 strengthening test `test_c109_mut_u64_load_store_byte_identical_to_i64` IS discriminative** (conf ~92 PASS). Compares byte-equality between u64 and i64 versions of the same fn. Pre-cycle-108 F6 revert: u64 LOAD_VAR/STORE_VAR fall to 32-bit (no 48 prefix); i64 stays on 64-bit (with 48 prefix). Byte-inequality. Post-fix: both produce the same predicate-routed emission, byte-equal. The ELF emission path at `compile_module_to_elf` does not include a symbol table or string table, so fn names ("body_u64") are identical between the two sources and do not introduce spurious diffs. CONST_INT i64/u64 both go through `_is_64bit_int_type` (cycle-106). PASS as a true strengthening of the cycle-109 F1 finding.

- **F2 strengthening test `test_c109_call_return_u64_caller_stores_full_rax` IS discriminative** (conf ~88 PASS). Same byte-equality structure as F1 strengthening. Pre-cycle-108 F2 revert: u64 CALL-return caller-store falls to `mov [rbp+y_slot], eax` (no 48); i64 stays on `mov [rbp+y_slot], rax` (48 89 45). Byte-inequality. Post-fix: byte-equality. PASS.

- **5 of 6 cycle-110 F4 bitwise/unary tests are discriminative** (conf ~90 PASS). The byte sequences `48 21 C8`, `48 09 C8`, `48 31 C8`, `48 D3 E0`, `48 F7 D0` for BIT_AND / BIT_OR / BIT_XOR / SHL / BIT_NOT respectively are REX.W-prefixed 64-bit-register variants. The 32-bit non-REX variants (`21 C8`, `09 C8`, `31 C8`, `D3 E0`, `F7 D0`) would emit pre-fix with NO 48 prefix. The asserted byte sequences are not co-emitted by any other path in the small test source (prologue, param-spill, RETURN). Each test is independently discriminative for its respective arm. PASS. (The NEG sibling — F1 of this audit — is the exception.)

- **u32→f32 / u32→f64 byte-pattern assertions are discriminative** (conf ~88 PASS). `F2 48 0F 2A C0` (cvtsi2sd rax) and `F3 48 0F 2A C0` (cvtsi2ss rax) are 5-byte sequences uniquely produced by the cycle-110 F3 unsigned-widening arms at lines 1340-1357. Pre-fix the test source `x as f64` for u32 source would emit `F2 0F 2A C0` (4 bytes, no REX.W) via the i32→f64 arm at line 1359. The 48 byte differentiates. PASS.

- **A.Range value-position loud-fail test is discriminative** (conf ~90 PASS). Pre-fix `if isinstance(expr, A.Range): return None` would silently return; the let-rhs caller would substitute `const_int(0)`; the test would NOT raise NotImplementedError; the `try/except` block would reach the `raise AssertionError(...)` at the end. Post-fix the test catches the NotImplementedError and asserts "range" appears in the message. Discriminative. PASS.

- **Stage 30 cycle-2 H1 wiring discriminative against the H1 regression class** (conf ~80 PASS). The 3 trap-62032/62033 tests at `test_codegen.py:2889-2912` distinguish "sentinel set + wired up" (exit 132) from "sentinel set but NOT returned" (the H1 regression — would compile the malformed-arity struct as if well-formed, falling through to a struct-lit emission with truncated/extended field count; the exit code would be the value of `p.x` or a different SIGILL from cascading state corruption, but not reliably 132). The test catches the H1 wire-up regression class but not the specific trap-id used (sub-threshold above).

- **Cycle-2 commit message vs cycle-2 IMPORTANT scope** (conf ~75 PASS). `1aecbae` commit explicitly says "NOT addressed in this cycle: IMPORTANT: missing regression test for traps 62032/62033 (would require new test infrastructure to assert AST_ERR; deferred to Stage 30 cycle-3+)". The subsequent `f9425a0` (cycle-2 IMPORTANT fix) then DID add the tests within the same cycle-2 boundary — using the existing `compile_and_exec` exit-code infrastructure rather than the heavier "assert AST_ERR" inspection the earlier commit speculated would be needed. This is faithful: the test gap was closed within the cycle, the commit-message inconsistency is between two consecutive commits of the same cycle and the later one wins. The "deferred to cycle-3+" mention in `1aecbae` is then explicitly overridden by `f9425a0`. No drift, just iterative refinement. PASS.

- **Stage 30 cycle-2 H1 commit message accuracy** (conf ~85 PASS, examined from `1aecbae`'s prose). Cycle-2 commit message claims:
  - "kovc.hx:983 — updated NUM_BINDINGS_CAP comment from 64 to 512" — verified at lines 983-984.
  - "kovc.hx:1012-1021 — updated cap-check comment from 64-entry to 512-entry, added Stage 28.9 cycle-110 (C109-CR-F3) trap_10032 note" — verified at lines 1012-1022.
  - "test_codegen.py:4912-4923 — removed stale Stage 29.1 'K2 SIGILLs' comment" — verified in the diff (the `# Stage 29.1 ... accept that K2 may not exit cleanly` block at the old lines 4898-4903 is deleted).
  All three claims correct. PASS.

---

## Cross-stage cuts examined

### Rotation surface 1 — Cycle-110 fix-sweep `9c451e6` mechanical correctness

| F | Site (post-fix line) | Change | Verification |
|---|---------------------|--------|--------------|
| F2 (A.Range) | `lower_ast.py:2006-2020` | `return None` → `raise NotImplementedError(...)` | Sibling pattern matches cycle-108 F8 (CharLit/StructLit/TileLit) verbatim. Span coords included. Lower-time loud-fail. PASS conf ≥90. |
| F3 (u32→f64/f32) | `x86_64.py:1340-1357` | Added 2 new arms (cvtsi2sd-from-rax, cvtsi2ss-from-rax) | Placement BEFORE i32→f64/f32 arms; guard `_is_unsigned_int_type(from_ty) and not from_is_i64` correctly excludes u64/usize/i32/i16/i8-signed; uses implicit zero-extend via eax→rax. PASS conf ≥90. |
| F4 (BIT_AND) | `x86_64.py:1545` | `_is_i64_type` → `_is_64bit_int_type` | Mechanical 1-line flip. PASS conf ≥92. |
| F4 (BIT_OR) | `x86_64.py:1570` | same | same |
| F4 (BIT_XOR) | `x86_64.py:1585` | same | same |
| F4 (SHL) | `x86_64.py:1599` | same | same |
| F4 (BIT_NOT) | `x86_64.py:1628` | same | same |
| F4 (NEG) | `x86_64.py:1641` | same | Mechanically correct but lacks direct regression coverage — see F1 finding above. |

SHR at line 1614 correctly retained as `_is_i64_type` per cycle-101 F2 deferred-known (sar is sign-arithmetic; unsigned variant deferred).

### Rotation surface 2 — Stage 30 cycle-2 `1aecbae` kovc.hx comment cleanup

Verified at lines:
- `kovc.hx:721-758` (`emit_prologue` comment) — bumped 1024→4096, includes cycle-110 cross-reference. Faithful to the production change (prologue allocation also bumped to 4096 at line 754). PASS.
- `kovc.hx:983-986` (NUM_BINDINGS_CAP doc comment) — updated from 64 to 512. Faithful to bind_init's 512-entry allocation at line 1001. PASS.
- `kovc.hx:1012-1022` (bind_push_typed cap-check pre-comment) — updated from 64-entry to 512-entry, adds cycle-110 trap_10032 note. Faithful to lines 1033-1034 (`if top >= 512 { emit_trap_with_id(10032); 0 - 1 }`). PASS.
- `kovc.hx:1064-1083` (`bind_alloc_offset`) — threshold 1024→4096 with cycle-110 cross-reference comment. Faithful to line 1078 `if off >= 4096`. PASS.
- `kovc.hx:1048-1062` (`bind_pop` comment) — STALE: still references "blowing past the 512-byte prologue allocation; emit_mov_local_eax(-560)". Explicitly NOT addressed by `1aecbae` (commit message acknowledges this as a LOW deferred item). Sub-threshold.

### Rotation surface 3 — Stage 30 cycle-2 `f9425a0` regression tests for traps 62032/62033

Two assertions at `test_codegen.py:2889-2900`:
1. `Pt<>{ 10, 32 }` (zero type-args for arity-1 `struct Pt<T>`) → triggers trap 62032 (arity-mismatch) via the post-loop `if ta_count != gp_count_pre` branch at `parser.hx:3301`.
2. `Pt<i32, i64>{ 10, 32 }` (extra type-args) → same trap 62032 via same branch.

Both source bodies have `let p = ...; p.x` which would, in the absence of the trap, attempt to read field `.x` from a malformed struct, producing either a struct-lit emission with arity-mismatch (silently truncated to 1 field at `parser.hx`'s struct_tab_lookup_idx path) or cascading state corruption. The H1 fix's `if early_err != (0-1) { early_err } else { ... }` dispatch at `parser.hx:3303-3305` short-circuits this and returns the AST_ERR(99, 62032, 0, 0) node, which codegen at `kovc.hx:6395-6409` emits as `mov eax, 62032; ud2` (SIGILL → exit 132).

Both assertions check `== 132`. Cross-environment exit-status stability is reliable on WSL/Linux (sub-threshold above).

Trap-id discrimination (62032 vs 62033) is sub-threshold concern documented above. Cycle-3 commit `07e6535` adds a third assertion at `:2907-2912` for `Pt<+>` (bad-token-in-args → trap 62033). The triplet now covers both id paths, though still not byte-discriminative.

Reusability of pattern: future cycles can add malformed-syntax assertions by:
1. Crafting a source string that triggers the desired AST_ERR mk_node payload.
2. Asserting `compile_and_exec(src) == 132`.

Plausible to extend; the pattern reuses existing harness without new infrastructure.

### Rotation surface 4 — Cycle-110 regression test cross-check

Spot-checked 3 of the 11 new tests via mental revert:

1. **`test_c110_bit_and_u64_emits_64bit_form`** (test_ir.py:740-755): Source `fn and_u64(a: u64, b: u64) -> u64 { a & b }`. Asserts `b"\x48\x21\xc8"`. Revert: change line 1545 `_is_64bit_int_type` → `_is_i64_type`. Result: u64 BIT_AND falls to `mov eax, [l]; mov ecx, [r]; and eax, ecx; mov [res], eax` (bytes: 8B 45, 8B 4D, 21 C8, 89 45 — no 48 prefix on the AND opcode). Asserted `48 21 C8` not in ELF. Test fails. **Discriminative**. PASS.

2. **`test_c110_cast_u32_to_f64_uses_zero_extend_path`** (test_ir.py:697-720): Source `fn u32_to_f64(x: u32) -> f64 { x as f64 }`. Asserts `b"\xF2\x48\x0F\x2A\xC0"`. Revert: remove the cycle-110 F3 arm at lines 1340-1348. Result: u32 source falls to the i32→f64 arm at line 1359, emitting `F2 0F 2A C0` (4 bytes, no 48 REX.W). Asserted 5-byte sequence not in ELF. Test fails. **Discriminative**. PASS.

3. **`test_c110_neg_u64_via_sub_emits_64bit_form`** (test_ir.py:814-830): Source `fn neg_u64(a: u64) -> u64 { 0_u64 - a }`. Asserts `b"\x48\x29\xc8"`. Revert: change NEG predicate at line 1641 `_is_64bit_int_type` → `_is_i64_type`. Result: the source has no unary `-`, so the NEG arm is never reached. SUB u64 emit at line 1450 is unchanged by the revert (it uses its own `_is_64bit_int_type` predicate, on `_is_64bit_int_type` since cycle-102). The `48 29 C8` sub-opcode is still emitted. Test passes. **NOT discriminative for NEG**. F1 finding above.

3 of 3 spot-checks confirm: 2 discriminative, 1 vacuous (which became F1 of this audit).

### Rotation surface 5 — kovc.hx trap-id collision audit

Full grep `emit_trap_with_id\(` across `helixc/bootstrap/kovc.hx` yields the following IDs (88 occurrences total):

- AST-tag-paired traps (low 1000s): 1001, 2001-2041, 3001-3041, 4001-4041, 5001-5051, 6001-6051, 8001-8016, 9001, 14001-14002, 16001-16003, 19001-23051, 24001-24051, 26001, 27002, 28020-28021, 28999, 29020-29021, 30020-30021, 31001, 32001-32040, 33001-33040, 42002, 52001, 60030.
- Parser-error namespace: 62001, 81002.
- Placeholder: 99001.
- Cap-overflow namespace: **10030 (pre-existing, bind_alloc_offset), 10031 (new, patch_table_add), 10032 (new, bind_push_typed), 10033 (new, fn_table_add)**.

The new cycle-110 IDs are sequential after 10030 and do not collide. The 62032/62033 from Stage 30 cycle-2 are NOT `emit_trap_with_id` calls — they're `mk_node(99, N, 0, 0)` AST_ERR payloads in `parser.hx`, eventually emitted by the AST_ERR codegen path. No collision with kovc.hx trap IDs.

Numbering convention: there is no strict `AST_TAG * 1000 + sub_id` convention as the task brief speculated. The codebase uses scheme-by-namespace:
- Low 1000s = AST-tag-paired (AST_ADD=2 → 2xxx, AST_SUB=3 → 3xxx, etc.).
- 8001-8016 = AST_OPCODE-class.
- 10030+ = cap-overflow class.
- 60xxx = parser.
- 99xxx = placeholder.

10031/10032/10033 fit the 10030 cap-overflow namespace correctly. PASS.

### Rotation surface 6 — Test-suite hygiene

Diff `c89432e..07e6535` (full cycle-110 + Stage 30 cycles 1-3 range):
- 0 new `pytest.skip` / `pytest.xfail` / `@pytest.mark.skip` markers.
- 0 orphan test files added.
- 11 new tests in test_ir.py:589-830, all properly named `test_*` and picked up by the file's local `main()` runner at line 833.
- 3 new assertions inside `test_bootstrap_kovc_full_pipeline_arithmetic` at test_codegen.py:2889-2912.
- 1 modified test (`fewer_lets` bumped from 64 to 256 chained lets at test_codegen.py:3611) plus 1 deleted test branch (`many_lets ... == 132` SIGILL assertion).

The deletion is documented in `1aecbae`'s diff as a temporary measure due to a pre-existing upstream bootstrap-large-source bug; recorded as sub-threshold above. PASS for hygiene.

### Rotation surface 7 — Docstring/comment drift in cycle-110 tests

Spot-checked 2-3 cycle-110 tests for docstring accuracy:

| Test | Source matches docstring? | Production change matches docstring? | Disposition |
|------|---------------------------|--------------------------------------|-------------|
| `test_c109_mut_u64_load_store_byte_identical_to_i64` | ✓ | ✓ (compares u64 vs i64 byte-identity post-F6) | PASS |
| `test_c109_call_return_u64_caller_stores_full_rax` | ✓ | ✓ (compares u64 vs i64 byte-identity post-F2) | PASS |
| `test_c110_range_in_value_position_raises_loud` | ✓ (source `let r = 0..10; 0`; docstring example `let r = 0..10; r` is illustrative, not literal) | ✓ | PASS |
| `test_c110_cast_u32_to_f64_uses_zero_extend_path` | ✓ | ✓ | PASS |
| `test_c110_cast_u32_to_f32_uses_zero_extend_path` | ✓ | ✓ | PASS |
| `test_c110_bit_and_u64_emits_64bit_form` | ✓ | ✓ | PASS |
| `test_c110_bit_or_u64_emits_64bit_form` | ✓ | ✓ | PASS |
| `test_c110_bit_xor_u64_emits_64bit_form` | ✓ | ✓ | PASS |
| `test_c110_shl_u64_emits_64bit_form` | ✓ | ✓ | PASS |
| `test_c110_bit_not_u64_emits_64bit_form` | ✓ | ✓ | PASS |
| `test_c110_neg_u64_via_sub_emits_64bit_form` | ✓ for SUB; **✗ for NEG** | **✗** — docstring claims to pin NEG; source exercises SUB; the docstring's claim "lowering uses SUB for unsigned-domain 'negation'" is factually false. | **F1 above (HIGH conf 82)** |

10 of 11 tests have docstring-accurate match. 1 has docstring-vs-implementation drift — F1 above.

---

## Positive observations (no finding)

- **Cycle-110 mechanically correct for all 8 closures** (F2 A.Range loud-fail + F3 unsigned-int→float arms + F4 BIT_*/SHL/BIT_NOT/NEG promotions). Five of the six F4 arms have discriminative byte-pattern coverage; the NEG arm is mechanically correct but lacks direct test (F1 finding).
- **F3 unsigned-int→float arms correctly placed BEFORE the signed i32→f64/f32 arms** (line 1340 < line 1359), and the guard correctly excludes u64/usize via `not from_is_i64`. Zero-extend semantics correct: `mov eax, [src]` implicitly zero-extends to rax, then REX.W `cvtsi2sd/ss xmm0, rax` treats the zero-extended 64-bit value as signed (which equals the original unsigned interpretation since the high bit of the 64-bit value is now 0).
- **A.Range loud-fail message is informative and faithful to the sibling cycle-108 F8 pattern**, including span coordinates.
- **Stage 29.1 cap-bump silent-failure pattern CLOSED** by cycle-110: `patch_table_add` / `bind_push_typed` / `fn_table_add` now emit `emit_trap_with_id(10031/10032/10033)` respectively at the cap-exceeded path. Cycle-109 F3 closed.
- **Prologue/bind_alloc_offset coordination CLOSED** by cycle-110: prologue allocation bumped 1024→4096 (matching the bind_state cap of 512 × 8 = 4096 bytes), and `bind_alloc_offset`'s trap threshold bumped 1024→4096 in lockstep. The cycle-109 CRITICAL conf 90 silent-failure (513-of-512 binding writing past the 1024-byte prologue into saved rbp / return address) is closed.
- **Stage 30 cycle-2 H1 wire-up CLOSED** by `fe7042f`: `parser.hx:3303-3305`'s outer `if early_err != (0-1) { early_err } else { ... }` dispatch restores the cycle-3 silent-failure trap semantics. Regression tests at `test_codegen.py:2889-2912` pin this wire-up against a future re-revert.
- **Cycle-3 trap-62033 test ADDED** by `07e6535`, closing the cycle-3 MEDIUM conf 78 finding. The triplet of `Pt<>`, `Pt<i32, i64>`, `Pt<+>` now covers both trap-id paths through the sentinel mechanism (62032 arity-mismatch via two sources, 62033 bad-token via one source).
- **Cycle-110 cap-bumps consistent across all sites**: `kovc.hx:739-758` (prologue) + `kovc.hx:983-1001` (bind_init) + `kovc.hx:1012-1046` (bind_push_typed) + `kovc.hx:1064-1083` (bind_alloc_offset). Internal consistency: 512 entries × 8 bytes = 4096 bytes. The off-by-one margin loss at `bind_alloc_offset` (effective cap 511 of 512) is a sub-threshold LOW deferred to cycle-3.
- **No new deferred-known item silently introduced** by the cycle-110 + Stage 30 cycle-2/3 diffs. The cycle-110 SHR deferred-known is explicitly noted in the commit message; the cycle-3 bind_alloc_offset off-by-one is in the cycle-3 commit message.
- **F1-strengthening (mut u64 load/store) and F2-strengthening (CALL-return u64 caller-store) tests** correctly use byte-identity between u64 and i64 emissions as the discriminator — closing the cycle-109 F1 and F2 vacuous-test concerns at the higher byte-equality bar.
- **Trap-id 10030/10031/10032/10033 sequential namespace is clean and conventional** for the cap-overflow class.
- **Commit-message accuracy** for `9c451e6`, `1aecbae`, `f9425a0`, `07e6535`: each commit's diff-verifiable claims have been verified above; all correct.

---

## Verdict

**Verdict: FINDINGS** — 1 HIGH finding.

| # | Severity | Conf | Title |
|---|----------|------|-------|
| F1 | HIGH | 82 | `test_c110_neg_u64_via_sub_emits_64bit_form` exercises SUB (cycle-102), not NEG; cycle-110 F4 NEG predicate promotion has zero regression coverage |

The cycle-110 fix-sweep is mechanically correct for all 8 closures (F2 A.Range loud-fail, F3 u32→f64/f32 cast, F4 six bitwise/unary predicate promotions). The Stage 30 cycle-2 H1 wire-up and comment cleanup are faithful to the audit findings they address. The cycle-3 trap-62033 supplement closes the cycle-3 MEDIUM. **However**:

- **1 of the 6 cycle-110 F4 regression tests is vacuous** for its claimed production change. `test_c110_neg_u64_via_sub_emits_64bit_form` uses a source (`0_u64 - a`) that lowers to SUB (already on the wide path since cycle-102), not NEG, and asserts a byte pattern (`48 29 C8` = `sub rax, rcx`) that SUB emits regardless of whether the cycle-110 NEG predicate flip at `x86_64.py:1641` is applied or reverted. The docstring contains a factually incorrect claim about the lowering pipeline ("lowering uses SUB for unsigned-domain 'negation'" — actual lowering at `lower_ast.py:1178` uses `tir.OpKind.NEG`). This is structurally the same defect class as cycle-109 F1/F2 (vacuous tests / docstring-vs-assertion drift).

The cycle-110 NEG u64 production fix IS a real correctness improvement (NEG u64 was silently truncating to 32-bit pre-cycle-110, and `kovc.hx:5253-5264` confirms u64 NEG is part of the language surface) — but a future revert of the predicate flip would not be caught by any test. A trivial fix (replace the test body with `fn neg_u64(a: u64) -> u64 { -a }` asserting `48 F7 D8` = `neg rax`) would close this gap.

Counter at start was 0/5 — verdict FINDINGS holds counter at 0/5.

---

## Cross-reference

- **cycle 107** (`docs/audit-stage28-9-cycle107-codereview.md`, HEAD `6af8a46`): CLEAN.
- **cycle 108** (commit `a600616`): fix-sweep closed all 8 cycle-107 findings.
- **Stage 29.1** (commit `c89432e`): cap-bumps unblocking K3=42.
- **cycle 109** (`docs/audit-stage28-9-cycle109-codereview.md`, HEAD `c89432e`): codereview FINDINGS (3 HIGH).
- **cycle 110** (commit `9c451e6`): fix-sweep closing cycle-109 silent-failures F2/F3/F4 (3 of 8). F1 (bind_state cap prologue overrun) closed concurrently by the same cycle-110 work in kovc.hx.
- **Stage 30 cycle-1 audit** (commit `c8d579d`, NOT CLEAN — 2 HIGH + 3 MEDIUM).
- **Stage 30 cycle-2 H1 fix** (commit `fe7042f`): wire up the `early_err` sentinel return path in parser.hx.
- **Stage 30 cycle-2 polish** (commit `1aecbae`): kovc.hx/test_codegen.py comment cleanups.
- **Stage 30 cycle-2 IMPORTANT fix** (commit `f9425a0`): regression tests for traps 62032/62033.
- **Stage 30 cycle-3 MEDIUM fix** (commit `07e6535`): regression test for trap 62033 specifically (the bad-token path of the sentinel mechanism).
- **cycle 111** (this doc, HEAD `07e6535`): codereview **FINDINGS** (1 HIGH), counter 0/5 (no advancement).

---

## No code edits made

Strict read-only audit. One Write of this doc only. No Edit calls, no source mutation, no test run.
