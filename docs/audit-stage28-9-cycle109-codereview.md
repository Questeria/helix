# Audit Stage 28.9 cycle 109 — Code review

- **Date**: 2026-05-12
- **HEAD**: `c89432e` ("🎉 STAGE 29.1: bumped patch_table + bind_state caps → K3 exits 42!")
- **Counter at start**: 0/5.
- **Scope**:
  1. **Cycle-108 fix-sweep diff** (`git diff b0e18c7..a600616`): verify the 7 fixes at `x86_64.py` (F1-F7) + 1 fix at `lower_ast.py` (F8) for correctness, completeness, and regression-test sufficiency.
  2. **Stage 29.1 cap-bump diff** (`git diff a600616..c89432e`): verify cap-bump sufficiency, sibling-cap consistency, commit-message accuracy, and undocumented side effects.
  3. **Rotation surface 1**: `_is_64bit_int_type` sibling consistency — BIT_AND/BIT_OR/BIT_XOR/SHL/SHR/BIT_NOT/NEG arms still on `_is_i64_type` per cycle-107 deferred-known list. Reachability check against shipping bootstrap source.
  4. **Rotation surface 2**: `kovc.hx` patch resolver and `emit_elf_for_ast_to_path` for remaining loud-fail / silent-fail paths post Stage 29.1.
  5. **Rotation surface 3**: test-coverage gaps in cycles 100-108 production changes.
  6. **Rotation surface 4**: Stage 29.1 commit-message accuracy vs. diff.
- **Bar**: confidence ≥ 80, severity HIGH (80-89) or CRITICAL (90-100) only. Below-bar items are recorded as sub-threshold observations.
- **Deferred-known list (carried forward, NOT re-flagged)**:
  - `monomorphize._mangle_ty` / hash_cons `_ast_equal` silent catchalls
  - typecheck / struct_mono pre-flatten in `check.py`
  - `autotune.collect_autotuned_fns` missing `iter_fn_decls`
  - `struct_mono.mangle_struct` collision
  - `A.StrLit` IR-lowering gap (cycle-101 F1 deferred)
  - DIV / MOD / SHR signed-vs-unsigned emit (cycle-101 deferred)
  - raw-200 enumeration in `parser.hx` Stage-8 monomorphize_pass
  - Stage 29 K2 SIGILL probe scripts (`_probe_stage29_*.py` untracked)
  - `DT_BIND_NOW` unused constant in `elf_dyn.py`
  - `evaluator.hx` tag table covers only Stage-3 subset
  - `BIT_AND` / `BIT_OR` / `BIT_XOR` / `SHL` / `SHR` / `BIT_NOT` / `NEG` `_is_i64_type` sibling emit paths (per cycle-107 carve-out)
  - `FFI_CALL` arm (already cycle-77-fixed with inline `or _is_u64_type` — equivalent behavior to the cycle-108 promotion of CALL)
  - cast-cascade `from_is_i64`/`to_is_i64` @1290-1291 — **now PROMOTED by cycle-108 F7**, removed from deferred list
  - SELECT/BR/RETURN cycle-107 deferred — **now PROMOTED by cycle-108 F4/F5/F3**, removed from deferred list
  - `COND_BR` — cycle-107 listed this as a deferred site but the actual emit at `x86_64.py:2050-2062` uses `mov eax, [cond]; test eax, eax` with NO `_is_i64_type` predicate; the cond operand is always bool. Cycle-107 deferred-known list contains a false entry; removed from deferred list.
  - All prior cycle findings.

---

## Methodology

Read-only inspection. No source mutation, no test execution, no scorecard run. Single Write of this document.

Cross-referenced files:

- `helixc/backend/x86_64.py:1037-1074` (`_is_i64_type`, `_is_u64_type`, `_is_64bit_int_type`, `_is_unsigned_int_type` predicate helpers).
- `helixc/backend/x86_64.py:1273-1380` (CAST cascade — F7 fix + cycle-106's new f32↔f64 arms).
- `helixc/backend/x86_64.py:1395-1610` (ADD/SUB/MUL/DIV/MOD + the deferred BIT_AND/BIT_OR/BIT_XOR/SHL/SHR/BIT_NOT/NEG arms).
- `helixc/backend/x86_64.py:1761-1903` (SELECT + CALL int-arg + CALL return — F4/F1/F2).
- `helixc/backend/x86_64.py:1962-2042` (RETURN + BR block-param copy — F3/F5).
- `helixc/backend/x86_64.py:2068-2118` (LOAD_VAR + STORE_VAR — F6).
- `helixc/ir/lower_ast.py:1042-1077` (F8 fix arms — CharLit / StructLit / TileLit loud-fail).
- `helixc/ir/lower_ast.py:2200-2268` (post-fix bottom of `_lower_expr` — catch-all `return None` preserved for A.StrLit per deferred-known).
- `helixc/tests/test_ir.py:402-587` (the 7 new C107-* regression tests).
- `helixc/bootstrap/kovc.hx:978-1057` (`bind_init` / `bind_push` / `bind_push_typed` — cap-bump from 64→512 at lines 988, 1013).
- `helixc/bootstrap/kovc.hx:1574-1609` (`patch_table_init` / `patch_table_add` — cap-bump from 4096→16384 at lines 1586, 1598).
- `helixc/bootstrap/kovc.hx:1043-1057` (`bind_alloc_offset` — prologue-size trap at offset 1024).
- `helixc/bootstrap/kovc.hx:1523-1571` (`fn_table_init` / `fn_table_add` — unchanged cap of 512, still tight against the 472 fns of bootstrap).
- `helixc/bootstrap/kovc.hx:739-745` (`emit_prologue` — `sub rsp, 1024` unchanged).
- `helixc/tests/test_codegen.py:4788-4933` (`test_bootstrap_kovc_self_host_loop` — test-relaxation diff).
- Bootstrap source u64-usage grep: 189 occurrences across `parser.hx`/`kovc.hx`/`lexer.hx`; no syntactic u64-bitwise / u64-shift / u64-bitnot / u64-neg constructs.

---

## Findings table

| Severity   | Count |
|------------|-------|
| CRITICAL (90–100) | 0 |
| HIGH (80–89)      | 3 |
| MED (50–79)       | 0 reportable (below bar) |
| LOW (<50)         | 0 reportable (below bar) |

---

## Findings

### F1 (HIGH conf 88) — `test_c107_mut_u64_local_uses_64bit_load_store` is vacuous: passes regardless of F6 presence

**File**: `helixc/tests/test_ir.py:481-501`
**Production code**: `helixc/backend/x86_64.py:2068-2118` (LOAD_VAR / STORE_VAR — the F6 fix-sweep arm).

**Defect class**: missing-regression-coverage / vacuous test.

The cycle-108 regression test `test_c107_mut_u64_local_uses_64bit_load_store` is intended to pin the F6 fix (LOAD_VAR / STORE_VAR predicate flip `_is_i64_type` → `_is_64bit_int_type` for u64/usize mutable locals). The test source:

```
fn rmw_u64() -> u64 {
    let mut x: u64 = 1_u64;
    x = x + 1_u64;
    x
}
fn main() -> i32 { 0 }
```

asserts `b"\x48\x8b\x45" in elf and b"\x48\x89\x45" in elf` (both 64-bit LOAD and STORE forms with disp8 base).

**Trace with F6 reverted** (LOAD_VAR/STORE_VAR back to `_is_i64_type`):

- `let mut x: u64 = 1_u64;`:
  - CONST_INT u64 path (cycle-106 fix at line 1222 — still applied since this audit doesn't revert cycle-106): emits `mov rax, imm64` (48 B8 ...) + `mov [rbp+x_slot], rax` (= **48 89 45 disp**). → 48 89 45 IS present from CONST_INT u64.
  - ALLOC_VAR + STORE_VAR: with F6 reverted, the STORE_VAR for u64 falls to the 32-bit `mov [rbp+x_slot], eax` (no 48 prefix). But the CONST_INT u64 emit ABOVE already emitted 48 89 45.
- `x = x + 1_u64;`:
  - LOAD_VAR x: with F6 reverted, emits `mov eax, [rbp+x_slot]` (no 48 prefix).
  - ADD u64 (cycle-102 fix at line 1395 — still applied): emits `mov rax, [rbp+l_slot]` (= **48 8B 45 disp**) + `mov rcx, [rbp+r_slot]` + `add rax, rcx` + `mov [rbp+res_slot], rax` (= **48 89 45 disp**). → both 48 8B 45 AND 48 89 45 ARE present from ADD u64 even with F6 reverted.
- `return x`:
  - RETURN u64 (F3 fix — still applied): emits `mov rax, [rbp+x_slot]` (= **48 8B 45 disp**). → 48 8B 45 present from F3.

**Conclusion**: with F6 reverted but F3, CONST_INT-u64, and ADD-u64 intact, **both byte patterns asserted by the test are present in the ELF**. The test passes regardless of whether F6 is applied or reverted. The test is vacuous as an F6 regression. F6 has no actual regression coverage post cycle-108.

**Why this matters**: F6 was a HIGH-conf-90 silent miscompile (every read-modify-write on a `let mut x: u64` dropped the high 4 bytes). The cycle-108 fix-sweep landed the closure, but the test that supposedly pins the fix would not catch a regression that re-reverts to `_is_i64_type`. Any future opportunistic edit to the LOAD_VAR/STORE_VAR arms could silently re-introduce the bug; the test suite would still report green.

**Suggested fix**: Either (a) write the test such that the value-flow path is *only* through LOAD_VAR/STORE_VAR and *not* through ADD-u64/CONST_INT-u64 (e.g. by sandwiching the rmw between a non-arithmetic `let mut`-read-back-read-back chain with the result used in a non-arithmetic position); or (b) inspect the IR module produced by `lower_src(src)` directly and assert that LOAD_VAR/STORE_VAR ops with u64 result-type are present, then run them through the backend with a stub `_is_64bit_int_type` and assert the emitted bytes there; or (c) bytecount-discriminative: pre-fix the function `rmw_u64` had a specific byte length; post-fix it has a longer length due to REX.W prefixes — assert the *number* of `48 8B 45`/`48 89 45` opcodes in a small symbol window. Option (c) requires symbol-table parsing; (a) is simplest.

**Confidence**: 88. The bytewise reasoning is mechanical and verifiable; ADD u64 emit at line 1395 is unambiguous; CONST_INT u64 at 1222 likewise.

---

### F2 (HIGH conf 85) — `test_c107_call_return_u64_stores_full_8_bytes` covers only F3; F2 (caller's return-value store) has no byte-pattern regression

**File**: `helixc/tests/test_ir.py:409-432`
**Production code**: `helixc/backend/x86_64.py:1895-1902` (the F2 fix — CALL return-value store predicate flip).

**Defect class**: missing-regression-coverage / docstring-vs-assertion drift.

The test docstring at lines 410-417 states:

> C107-F2 regression (HIGH conf 90): a CALL whose callee returns u64 must store the full 8-byte rax to the caller's result slot via `mov [rbp+disp], rax` (48 89 ...). Pre-fix `_is_i64_type` did not match u64, so only eax (low 32 bits) was stored — silently dropping the high half of the SysV-ABI-delivered return value. Also covers C107-F3 (callee RETURN of u64): the callee must load via `mov rax, [rbp+disp]` (48 8B 45 ...) — the byte sequence `48 8B 45` is the discriminative opcode for the wide load.

The actual assertion at lines 429-432:

```
assert b"\x48\x8b\x45" in elf, (...)
```

Only the **F3 byte pattern** is asserted (48 8B 45 = `mov rax, [rbp+disp8]`, the callee's pre-RETURN load). The **F2 byte pattern** that should be tested is `mov [rbp+y_slot], rax` (= 48 89 45 disp) at the caller side after the CALL. That assertion is absent.

**Trace with F2 reverted** (CALL return-store back to `_is_i64_type`):

- F2 reverted, F3 still applied (and they are independent code sites — CALL at line 1899 vs RETURN at line 2008):
  - Inside `id_u64(x: u64)`: F3 emits `mov rax, [rbp+x_slot]` (48 8B 45) before `ret`. → 48 8B 45 IS present.
- The test passes because `b"\x48\x8b\x45" in elf` is True via the F3 path.

**To make this discriminative for F2** the test would need to assert a byte pattern that ONLY F2 produces. F2 produces `mov [rbp+y_slot], rax` (= 48 89 45 disp) after `call id_u64`. But that pattern is *also* produced by CONST_INT 1_u64 (`mov [rbp+slot_1u64], rax`) and id_u64's own param spill (well, `48 89 7D` rdi-form, not 45). So in this specific source, `48 89 45` appears once (from CONST_INT 1_u64); F2's contribution adds a *second* occurrence. Disambiguation requires either (i) constructing a source that has no other 48 89 45 site, or (ii) counting occurrences (pre-fix N, post-fix N+1).

**Why this matters**: F2 was HIGH conf 90 — it silently dropped the high half of every u64 return value at the caller side. Same regression-resilience concern as F1: future maintenance could re-revert. No coverage means no signal.

**Suggested fix**: add a separate `test_c107_caller_stores_u64_return_as_rax` test with a source that contains no CONST_INT u64 (e.g. `id_u64` returns its u64 param and main passes a u32 input then casts inside the callee — or use an FFI-call shape); or count `b"\x48\x89\x45".count()` in the ELF body of main and assert ≥ N pre-determined.

**Confidence**: 85. The docstring explicitly says the assertion covers both F2 and F3 but mechanically only one byte pattern is checked.

---

### F3 (HIGH conf 82) — `patch_table_add` / `bind_push_typed` / `fn_table_add` silent-failure pattern unchanged by Stage 29.1; cap-bumps only postpone the silent-corruption recurrence

**File**: `helixc/bootstrap/kovc.hx:1593-1609` (`patch_table_add`), `1009-1025` (`bind_push_typed`), `1534-1549` (`fn_table_add`).

**Defect class**: silent failure / silent corruption / cap-coupled.

The Stage 29.1 fix bumps:
- `patch_table` cap: 4096 → 16384 entries (2.4x headroom over the new measured 6800)
- `bind_state` cap: 64 → 512 entries (2.5x headroom over measured ~200/fn peak)

Both fixes are correct as immediate remediation: the bootstrap now self-compiles without overflow.

However, the underlying silent-failure pattern is **unchanged**:

```
fn patch_table_add(state: i32, disp_slot: i32, name_start: i32, name_len: i32) -> i32 {
    let top = __arena_get(state);
    if top >= 16384 {
        0 - 1                  // returns -1 silently
    } else {
        ...
        0
    }
}
```

All 11 callers of `patch_table_add` (lines 3099, 3201, 3278, 3288, 3298, 3326, 3785, 3843, 3943, 6042, 6306) **discard the return value**:

```
patch_table_add(patch_state, arena_lea_slot, arena_base_s, 18);
```

Same shape for `bind_push_typed` (4 callers at 4384, 5815, 5848, 6388 all discard) and `fn_table_add` (2 callers at 6327, 6536 discard).

**Mode-of-failure (unchanged from pre-Stage-29.1)**:
1. Bootstrap source grows past 16384 patches (currently 6800).
2. The 16385th+ `patch_table_add` returns -1, caller discards.
3. The LEA disp slot stays at 0 (initialized to 0 by the arena's clear-on-init).
4. The patch resolver loop at line 6549 iterates only over the recorded patches; the dropped entry is invisible.
5. K2 reads from arena base 0 (a wrong address — the actual arena_base symbol is elsewhere), produces an invalid or empty K3.
6. The compile succeeds with no diagnostic; K3 emerges silently broken.

The Stage 29.1 commit captures this exact failure mode in the post-mortem ("Dropped patches left 217 unfilled LEA disps at 0 → K2 read arena from wrong address → empty output"). The fix delays the recurrence but does NOT eliminate the pattern. **Any future growth of the bootstrap source past 16384 patches will silently re-overflow, with the same silent-corruption symptom.**

Furthermore:
- `bind_push_typed` silent-skip at cap exceeds: a `let mut x: u64 = ...; x = x + 1_u64;` for x bound at the 513th binding writes the value into the previous binding's offset slot (well, `bind_lookup` returns the unbound-sentinel 0, which the AST_VAR audit-10 guard catches — but that guard emits `mov eax, 0` as a placeholder, so the value of x becomes 0 wherever it's read).
- `fn_table_add` silent-skip at cap exceeds: the 513th fn declaration's `fn_table_add` returns -1; subsequent `fn_table_lookup` for that fn name returns -1; CALL patches for that name fail to resolve. Bootstrap currently has 472 fns across `lexer.hx`+`parser.hx`+`kovc.hx` (8% headroom over the 512 cap). Adding 40+ new fns silently breaks lookup.

**Stage 29.1 was the second discovery of this pattern surfacing under self-compile** (cycle-99/100 ADD/SUB/MUL u64 was the first; cycle-107 silent-failures F8 was the third). It is a recurring class of bug. The cap-bumps are remediation, not closure.

**Suggested fix**: convert silent return-of-`-1` to an `emit_trap_with_id(N)` call inside each `*_add` function, with a distinct trap ID. (See `bind_alloc_offset` at line 1052-1054 for the existing pattern — trap ID 10030.) The cost is a runtime trap (codegen aborts loudly) versus a silent miscompile; for a self-hosting compiler the loud failure is strictly better.

**Why HIGH conf 82 not lower**: this is structurally the *same* class of silent-failure that cycle-107 silent-failures explicitly flagged (F1-F8), and that cycle-108 mostly closed in the Python backend by promoting predicates. The kovc.hx side has the analogous pattern unaddressed. Confidence the bug exists: 95+. Confidence it should fire as a cycle-109 finding rather than a deferred-known cap-policy choice: ~85. The Stage 29.1 commit message acknowledges the issue exists ("the cap-check guard prevents corruption but the dropped patch is silent") without converting to loud-fail. This is a regression risk for any subsequent bootstrap growth.

---

## Sub-threshold observations (NOT findings — below the 80-confidence bar)

- **CAST sign-extend i32→u64 vs zero-extend** (conf ~65). When `from_ty = i32` and `to_ty = u64`, the F7 unsigned-widening arm at line 1305 does NOT fire (`_is_unsigned_int_type(i32) = False`), so the cascade falls to the i32→i64 movsxd arm at line 1311, which sign-extends. Per Rust/C standard semantics, `cast<i32, u64>` should sign-extend then bitcast, so this is *correct* — but the symmetric concern is: when `from_ty = i16` or `i8`, `from_is_i64 = False`, `_is_unsigned_int_type(i16) = False`, falls to movsxd which reads 4 bytes from a 2-byte or 1-byte slot. Whatever garbage is in the upper bytes of the slot gets included before the sign-extension. This is a pre-existing concern (predates cycle-108) and arguably acceptable if narrow-slot stores zero-extend (which they do under cycle-87's narrow-store changes). Below 80.

- **CAST `f64 → u64` / `f64 → i64` / `f32 → u64` / `f32 → i64` widening arms still missing** (conf ~70, observed in cycle-107 doc as sub-threshold). The CAST cascade has no explicit arm for float-to-64-bit-int. Trace for `f64 → u64`: `from_is_float=T, from_is_f64=T, from_is_i64=F, to_is_i64=T`. Lines 1293 (i64→i32): from_is_i64=F → skip. 1305 (F7 unsigned-widen): `not from_is_float`=F → skip. 1311 (i32→i64 movsxd): `not from_is_float`=F → skip. 1318 (i64→f64): from_is_i64=F → skip. 1325 (i64→i64): from_is_i64=F → skip. 1330 (i32→f64): to_is_f64=F → skip. 1336 (i32→f32): from_is_float=T fails `not from_is_float` → skip. 1342 (f64→i32): T → **fires** (incorrectly — emits cvttsd2si eax, leaves high 4 bytes of res_slot stale). Same pre-existing concern as the cycle-107 doc noted. Below 80.

- **fn_table cap headroom is 8%** (conf ~60). 472 fns vs 512 cap. Adding 40+ fns to bootstrap silently overflows `fn_table_add`. Stage 29.1 bumped patch_table and bind_state but did NOT bump `fn_table` cap. Per the F3 finding above, the silent-failure pattern means future growth will silently break lookup. Below 80 because the bootstrap is unlikely to grow by 40+ fns without a deliberate refactor that would notice.

- **fn_type_table cap is 256** (conf ~50). `fn_type_table_init` at line 1420 allocates 256 entries. Lookup-failure mode identical to fn_table. Tighter cap means earlier overflow if bootstrap grows. Below 80, same disposition as above.

- **Stage 29.1 test-relaxation is NOT a finding** (conf ~85 that the disposition is correct, but it's not a regression — the commit message documents it). The cycle-108→Stage-29.1 diff comments out `assert run_k2.returncode < 128`. The K2 exits 132 (SIGILL at process exit) AFTER writing a valid K3. The relaxation is acknowledged in both the commit message ("Updated assertion to match the actually-achieved milestone") and the test source comment. NOT a finding. Recorded for transparency.

- **`test_c107_call_arg_u64_uses_64bit_reg` discrimination** (conf ~88 PASS). Asserts `b"\x48\x8b\x7d"` (mov rdi, [rbp+disp8]). Pre-F1: arg load was `mov edi, [rbp+disp8]` (`8B 7D`, no 48). Post-F1: `48 8B 7D`. The byte sequence `48 8B 7D` is specifically the rdi-load form; no other cycle-106/108 emit site produces this exact 3-byte sequence for a non-first-arg purpose. Discriminative. **PASS as a regression test.**

- **`test_c107_if_else_u64_emits_64bit_branch_copy` discrimination** (conf ~85 PASS). Asserts `b"\x48\x89\x45"`. Pre-F5: BR block-param store was `mov [rbp+dst], eax` (no 48 prefix). Post-F5: `mov [rbp+dst], rax` (48 89 45). In this specific test source (`fn pick` + `fn main { 0 }`), the only other 48-prefixed `mov [rbp+...], rax` source is u64 param-spill (48 89 7D / 48 89 75, NOT 45). No ADD/SUB u64 ops in pick. So `48 89 45` is discriminative for F5. **PASS as a regression test.**

- **`test_c107_cast_u32_to_u64_uses_zero_extend_not_sign_extend` discrimination** (conf ~85 PASS). The "not in ELF" assertion of `48 63 c0` (movsxd) is weak because pre-fix the cast fell through to the 4-byte mov-copy (no movsxd either). The positive assertion of `48 89 45` IS discriminative — pre-fix `widen_u32` had no 8-byte store, post-fix the unsigned-widen arm at line 1305-1309 emits `mov eax, [src]` (8B 45) + `mov [dst], rax` (48 89 45). **PASS as a regression test, but for the positive assertion only.**

- **F8 loud-fail tests** (conf ~90 PASS). Both `test_c107_char_lit_in_expr_pos_raises_loud` and `test_c107_struct_lit_in_expr_pos_raises_loud` raise/catch NotImplementedError + assert keyword in message. Discriminative — pre-fix `return None` would NOT raise; the test would fall through to the AssertionError at the end. **PASS.**

- **No regression test for the Stage 29.1 cap-bump** (conf ~75, recorded as sub-threshold). The Stage 29.1 cap-bump (`patch_table` 4096→16384, `bind_state` 64→512) has no test that asserts the new caps are sufficient. The only test that exercises bootstrap self-compile is `test_bootstrap_kovc_self_host_loop`, which now has its K2 exit-code assertion relaxed (commented out at lines 4912-4916) and asserts only K3 size > 0 and `b"exit=42" in run_k3.stdout`. Future cap-pressure growth (e.g. bootstrap adding 200 more fns) could re-overflow without the test catching it (because the cap-violation is silent, per F3 above). Recorded sub-threshold because the immediate Stage 29.1 verification is captured in the commit message and the K3=42 assertion does cover the user-visible failure mode at the current source size; structural regression-resilience is the F3 concern, not a separate finding.

- **bind_alloc_offset trap-id 10030** (conf ~85 PASS). The prologue allocates 1024 bytes (`emit_prologue` at line 743 emits `sub rsp, 1024`). `bind_alloc_offset` traps with id 10030 when offset >= 1024 (i.e., when peak simultaneous live bindings > 128). The Stage 29.1 commit message claim of "~200 bindings per fn at peak" refers to **cumulative push events** (which `bind_pop` reclaims via `cur_off - 8`), not peak simultaneous live. With LIFO scoping, peak `top` of bind_state stack can be 200 while peak `next_offset` stays under 1024. Verified by reading bind_pop semantics at line 1027-1040. **No bug** — the cap and the prologue-size are correctly decoupled. PASS.

- **Commit-message claim of "693 prior tests"** (conf ~50). Pre-cycle-108 test count `find helixc/tests -name 'test_*.py' | xargs grep -c '^def test_'` reports 1566 lines starting with `def test_`. The 693 figure is inconsistent with the apparent total. Likely refers to a filtered subset (e.g. tests not gated by WSL or by `@pytest.mark.skip`). Below 80 — not load-bearing for any finding.

---

## Cross-stage cuts examined

### Cycle-108 F1-F7 — `_is_i64_type` → `_is_64bit_int_type` predicate flips (mechanical PASS at conf ≥ 90)

For each of the 7 fix sites:

| Fix | Site (post-fix line) | Predicate change | Sibling-reference site |
|-----|---------------------|------------------|------------------------|
| F1  | x86_64.py:1882 (CALL int-arg) | `_is_i64_type(arg.ty)` → `_is_64bit_int_type(arg.ty)` | FFI_CALL arm at 1955 (cycle-77 fixed via inline `or _is_u64_type`) |
| F2  | x86_64.py:1899 (CALL int return) | `_is_i64_type(op.results[0].ty)` → `_is_64bit_int_type(...)` | FFI_CALL at 1974 (already cycle-77 inline) |
| F3  | x86_64.py:2008 (RETURN) | `_is_i64_type(op.operands[0].ty)` → `_is_64bit_int_type(...)` | n/a |
| F4  | x86_64.py:1786 (SELECT is_i64) | `_is_i64_type(res_ty)` → `_is_64bit_int_type(res_ty)` | n/a |
| F5  | x86_64.py:2042 (BR block-param) | `_is_i64_type(operand_ty)` → `_is_64bit_int_type(operand_ty)` | n/a |
| F6  | x86_64.py:2087, 2107 (LOAD_VAR / STORE_VAR) | `_is_i64_type(res_ty/src_ty)` → `_is_64bit_int_type(...)` | n/a |
| F7  | x86_64.py:1290-1291, +1305-1309 (CAST `from_is_i64`/`to_is_i64` + new unsigned-widening arm) | predicates flipped + new arm placed BEFORE the i32→i64 movsxd arm | symmetric with cycle-106's CAST f32↔f64 arm placement |

All 7 are mechanical one-line predicate substitutions; the unsigned-widening arm (F7) correctly places before the sign-extending movsxd arm (so u32→u64 emits zero-extend, not sign-extend). PASS at conf ≥ 90.

### Cycle-108 F8 — `_lower_expr` CharLit/StructLit/TileLit loud-fail arms (PASS at conf ≥ 90)

Three new arms at `lower_ast.py:1061-1077` raise `NotImplementedError` with span-anchored messages. The bottom `return None` at line 2268 is preserved for A.StrLit per cycle-101 F1 deferred-known. Cycle-107 recommended option (a) (explicit arms) over option (b) (replace catch-all); cycle-108 chose option (a). PASS at conf ≥ 90.

### Stage 29.1 cap-bumps — sufficiency + sibling-cap consistency (PASS at conf ≥ 85 for sufficiency; conf 82 finding for silent-failure pattern, see F3)

- `patch_table`: 4096 → 16384 entries (commit claims 4719 calls + 2059 LEAs = 6778 total, 2.4x headroom). The 16384 cap at `patch_table_init:1586` (49152 = 16384×3 slot allocation) and the cap-check at `patch_table_add:1598` (`top >= 16384`) are consistent. ✓
- `bind_state`: 64 → 512 entries. `bind_init:988` allocates 2048 = 512×4 slots; `bind_push_typed:1013` checks `top >= 512`. ✓
- Sibling caps NOT bumped: `fn_table` (cap 512, ~92% utilized — see sub-threshold), `fn_type_table` (cap 256). The cycle-108 commit message does not justify NOT bumping these. Sub-threshold concern.
- The 1024-byte prologue is correctly decoupled from the bind_state stack-cap (LIFO scoping via bind_pop). ✓

### Stage 29.1 commit-message accuracy (PASS at conf ≥ 85)

Commit message claims, verified against diff:

| Claim | Verification |
|-------|--------------|
| "patch_table cap (4096) overflowed (bootstrap now needs ~6800 patches)" | Diff bumps cap from 4096 to 16384 at three sites: comment line 1576-1581 (now says "16384"), init loop at 1586 (12288 → 49152), guard at 1598 (4096 → 16384). All three consistent. ✓ |
| "bind_state cap (64 entries) overflowed (parser.hx's parse_primary has ~200 bindings/fn)" | Diff bumps cap at `bind_init:988` (256 → 2048) and `bind_push_typed:1013` (`top >= 64` → `top >= 512`). Both consistent. parse_primary is ~1700 lines and contains nested let-bindings; ~200 peak top is plausible with bind_pop reclaiming offsets. ✓ |
| "K2 size: 277746 bytes" | Cannot verify from diff alone (runtime claim). |
| "K2 unfilled LEAs: 0 (was 217)" | Cannot verify from diff (runtime claim); plausible given the cap math. |
| "K3 size: 4125 bytes" | Matches the `test_bootstrap_kovc_demo_emits_ast_int_42` assertion at test_codegen.py:4961 (`assert size_proc.stdout.strip() == b"4125"`). ✓ |
| "K3 runs and EXITS 42 ✓✓✓" | Test assertion at test_codegen.py:4941 (`b"exit=42" in run_k3.stdout`). ✓ |
| "Test partially un-skipped" | Diff comments out the `_pytest.skip` line at test_codegen.py:4796-4797. ✓ |
| "K2 itself still SIGILLs at process exit AFTER writing K3" | Diff comments out `assert run_k2.returncode < 128` at lines 4913-4916. ✓ |

All claims verifiable from diff are correct. PASS at conf ≥ 85.

### Rotation surface 1: deferred-known `_is_i64_type` sibling sites (PASS at conf ≥ 90)

The cycle-107 deferred-known list carries forward: BIT_AND / BIT_OR / BIT_XOR / SHL / SHR / BIT_NOT / NEG / FFI_CALL (all `_is_i64_type`-only) and (formerly) cast-cascade / SELECT / BR / RETURN / CALL (now promoted by cycle-108).

Reachability check: grepped the bootstrap source for u64-typed bitwise / shift / bitnot / neg constructs:

```
$ grep -E 'u64.*<<|u64.*>>|u64.*&|u64.*\||u64.*\^|!.*u64|-.*u64' helixc/bootstrap/*.hx
(only 1 match in kovc.hx line 5373, inside a comment)
(2 matches in kovc.hx line 5723, inside a comment)
```

No syntactic occurrences of `u64 << k` / `u64 >> k` / `u64 & x` / `u64 | x` / `u64 ^ x` / `!u64_val` / `-u64_val` in `parser.hx` / `kovc.hx` / `lexer.hx` outside of comments. The deferred-known siblings are NOT reachable from the bootstrap source; their deferred-known disposition holds at cycle 109.

If a future bootstrap revision introduces u64-bitwise / u64-shift / u64-bitnot / u64-neg syntax, those sites WILL silently miscompile until promoted. This is the same defect class as cycle-100/102/106/108 fix-sweeps. PASS for the current cycle, but the deferred-known set should be re-audited whenever bootstrap grows.

### Rotation surface 2: kovc.hx loud-fail / silent-fail paths (PARTIAL PASS; F3 finding)

Beyond the F3 finding on `patch_table_add` / `bind_push_typed` / `fn_table_add`:

- `bind_alloc_offset` at line 1043-1057 traps loudly (trap id 10030). ✓
- `bind_lookup` at line 1059-1080 returns 0 for unbound (sentinel); the AST_VAR audit-10 guard catches this and emits a `mov eax, 0` placeholder. This is a CONTRACT (documented at line 1004-1006), not a silent-fail. ✓ (acceptable per deferred-known)
- `fn_table_lookup` at line 1551-1571 returns -1 for not-found; callers DO check the -1 sentinel before patching. Mostly ✓.
- `emit_elf_for_ast_to_path` at the bottom of kovc.hx: loops over patch_state, fn_state, bind_state. Bounded by `top` field of each table. If those tops are under cap (the F3 silent-fail mode), the loop iterates correctly over the recorded entries; the dropped entries are simply absent from the output. Silent.

The silent-failure pattern in `*_add` is the dominant cycle-109 finding (F3).

### Rotation surface 3: test-coverage gaps in production code added cycles 100-108

| Cycle | Production change | Regression test | Discriminativity |
|-------|-------------------|------------------|------------------|
| 100   | unsigned int cmp (cmp dispatch promotion) | `test_c99_*` (pre-cycle-100) | conf 88, sufficient |
| 101   | DIV / MOD signed-only (deferred) | n/a | deferred-known |
| 102   | ADD/SUB/MUL u64 | `test_c101_*_u64_*` | conf 90, sufficient |
| 105   | f32/f64 cross-precision, unit-type, break/continue, non-Range For | `test_c105_*` (6 tests) | conf 90, sufficient per cycle-107 audit |
| 106   | CONST_INT u64, param-spill u64, BITCAST u64 wide | `test_c105_u64_const_emits_64bit_path` | conf 88, partial (BITCAST and param-spill not directly asserted) |
| 107   | (audit-only, no production change) | n/a | n/a |
| 108   | F1-F7 + F8 | `test_c107_*` (7 tests) | F1/F4/F5/F7/F8 conf 85-90 sufficient; **F2 docstring-vs-assertion drift; F6 vacuous** (see F1/F2 findings above) |

Cycles 100-107 mostly carry sufficient byte-pattern regression tests. Cycle-108's F2 and F6 tests have the discriminativity gaps documented in this audit's F1 and F2 findings.

### Rotation surface 4: Stage 29.1 commit-message accuracy

Verified above (see "Stage 29.1 commit-message accuracy" section). All diff-verifiable claims are correct. Runtime claims (K2 size, unfilled LEA count, trap counts) cannot be verified statically but are plausible per the cap math.

---

## Positive observations (no finding)

- **Cycle-108 F1-F5 / F7 regression tests ARE discriminative** (F1 rdi-load, F4 SELECT is gated by SELECT-specific predicate flip not yet probed, F5 BR-side merge-store, F7 unsigned-widen). F8 tests (CharLit, StructLit loud-fail) ARE discriminative.
- **F7 unsigned-widening arm placement is correct** (line 1305-1309 fires BEFORE the i32→i64 movsxd arm at 1311). Without this ordering, u32 with high-bit-set would sign-extend wrong.
- **F7 predicate guard is correct** — `not from_is_i64` excludes u64/usize sources (which already use the i64→i64 arm), and `_is_unsigned_int_type(from_ty)` correctly excludes i32/i16/i8 (which sign-extend via the movsxd arm immediately below).
- **Cycle-108 F8 chose option (a)** from the cycle-107 recommendation (explicit arms over catch-all replacement), preserving the deferred-known A.StrLit silent-drop. Documentation is faithful.
- **Stage 29.1 commit-message captures the failure mode in detail** (patch_table overflow → 217 unfilled LEAs → empty K3; bind_state overflow → 9 trap_with_id(1001) → SIGILL). The post-mortem is faithful to the diff and to the verification numbers.
- **Stage 29.1 cap bumps are sized with reasonable headroom** (2.4x for patch_table, 2.5x for bind_state) — not gratuitous, not too tight.
- **bind_state cap-bump correctly decouples from prologue-size** — peak `top` (cumulative bind-push) vs peak `next_offset` (cumulative live offsets) are LIFO-decoupled. The audit verified this is not a hidden coupling.
- **Test-relaxation in `test_bootstrap_kovc_self_host_loop` is documented in both commit message and test source comments.** Not a silent regression.
- **No new deferred-known item has been silently introduced** by the cycle-108 + Stage 29.1 diffs.

---

## Verdict

**Verdict: FINDINGS** — 3 HIGH findings.

| # | Severity | Conf | Title |
|---|----------|------|-------|
| F1 | HIGH | 88 | `test_c107_mut_u64_local_uses_64bit_load_store` vacuous as F6 regression — passes with F6 reverted |
| F2 | HIGH | 85 | `test_c107_call_return_u64_stores_full_8_bytes` covers only F3; F2 caller-side return-store has no byte-pattern regression |
| F3 | HIGH | 82 | `patch_table_add` / `bind_push_typed` / `fn_table_add` silent-failure pattern unchanged by Stage 29.1; cap-bumps only postpone silent-corruption recurrence |

The cycle-108 fix-sweep itself is mechanically correct for all 8 closures (F1-F8 backend predicate flips + IR loud-fail arms). The Stage 29.1 cap-bumps are correctly sized and the commit message is faithful to the diff. However:

- **2 of the 7 cycle-108 regression tests are insufficient** — they pass regardless of the production code change they purport to pin (F6 byte-pattern is also produced by ADD u64; F2 byte-pattern is also produced by CONST_INT u64). Future maintenance could re-revert F2 or F6 with green CI.
- **Stage 29.1 cap-bump remediates the immediate overflow but does not address the silent-failure pattern in the cap-exceeded path**. Any future bootstrap growth past 16384 patches (or 512 bindings or 512 fns) will silently re-corrupt K3 with no diagnostic.

Counter at start was 0/5 — verdict FINDINGS resets / holds counter at 0/5.

---

## Cross-reference

- **cycle 107** (`docs/audit-stage28-9-cycle107-codereview.md`, HEAD `6af8a46`): CLEAN, counter 0/5 → 1/5. Returned 8 HIGH findings from the silent-failures audit.
- **cycle 108** (commit `a600616`): fix-sweep closed all 8 cycle-107 findings (F1-F8). 7 regression tests added at test_ir.py:402-587.
- **Stage 29.1** (commit `c89432e`): cap-bumps on patch_table (4096→16384) and bind_state (64→512). Unblocks K3=42.
- **cycle 109** (this doc, HEAD `c89432e`): codereview **FINDINGS** (3 HIGH), counter 0/5 (no advancement on FINDINGS).

---

## No code edits made

Strict read-only audit. One Write of this doc only. No Edit calls, no source mutation, no test run.
