# Audit Stage 28.9 cycle 105 — Code review

- **Date**: 2026-05-12
- **HEAD**: `77e4b85` ("Stage 28.9 cycle-104 audits: 3/3 CLEAN, counter 1/5 -> 2/5")
- **Counter at start**: 2/5 (cycle-104 CLEAN)
- **Scope**: `git diff 31e1725..HEAD -- helixc/`
  - Empty for committed history. The 31e1725..77e4b85 diff is pure-markdown (the three cycle-104 audit docs in `docs/`). No `helixc/` source change between cycle-104 baseline and cycle-105 HEAD.
  - **Working-tree** carries one unstaged kovc.hx delta (lines 6549-6565) labeled `STAGE 29 DEBUG (variant 4)`. Tracked as a sub-threshold observation below; not a finding.
  - Per cycle-105 prompt the focus rotates to areas not deeply probed in cycles 101-104: `helixc/backend/x86_64.py` non-arithmetic emit paths (prologue/epilogue, calling convention, SysV register allocation, spill slots), `helixc/backend/elf*.py` (ELF section construction, relocations, symbol tables), `helixc/stdlib/*.hx`, `helixc/tests/`, and `helixc/bootstrap/*.hx`.
- **Bar**: only report findings at confidence ≥ 80. Re-flagging C1–C104 findings is FORBIDDEN. Specifically excluded from re-flag per cycle-104 carve-outs:
  - cycle-101 codereview F2 tail (DIV / MOD / SHR / BIT_AND / BIT_OR / BIT_XOR / SHL / BIT_NOT / NEG signed-vs-unsigned / u64-width gate);
  - cycle-101 silent-failures F1 (A.StrLit IR lowering gap);
  - cycle-57 / cycle-104 enumerated `_is_i64_type`-only fallthrough siblings: CONST_INT@1198, BITCAST@1234-1236, prologue spill@986, cast matrix `from_is_i64`/`to_is_i64` @1253-1254, DIV @1418, MOD @1433, BIT_AND @1453, BIT_OR @1468, BIT_XOR @1483, SHL @1498, SHR @1513, BIT_NOT @1527, NEG @1540, SELECT `is_i64` @1716, BR `operand_ty` @1945, RETURN `op.operands[0].ty` @1917, FFI_CALL @1803 (gated by exclusion at 1872), COND_BR @1953 (i32 cond), function-prologue parameter spill @986;
  - cycle-16 deferred-known `_alloc_array` `elem_size` unused parameter;
  - cycle-82 silent-failures `0x400000` / `0x401000` / `4096` ELF constants hard-coded in `kovc.hx` (documented Phase-0 design);
  - cycle-103 sub-threshold inline-`or _is_u64_type` style drift at 1672-1676 / 1872 / 1891;
  - cycle-7 / cycle-104 deferred-known 7-site (now 10-site post Stage-28.11 INC-3b expansion) raw-200 enumeration in `parser.hx` Stage-8 monomorphize_pass.

---

## Methodology

Read-only inspection. No source mutation, no test runs, no scorecard. One Write for this doc.

Because no `helixc/` source has moved between cycle-103 and cycle-105 HEAD (cycle-104 was itself a no-source-delta audit), this cycle re-walks the cycle-105-scoped surfaces under freshly chosen dimensions that prior cycles' code-review passes have not explicitly probed:

1. **SysV ABI stack-alignment chain end-to-end**: kernel → `_start` → entry_fn prologue → inline `call_rel32` / `call_qword_ptr_rip_rel_ffi` sites → MODIFY's nested verifier call → write_file_to_arena's callee-saved spill block (`push rbx; push r12; push r13; push r14; sub rsp, 16`).
2. **ELF dynamic-link tables algebraic invariants**: `n_dyn_entries = len(needed_libs) + 12` assert vs. the explicit appends; SYSV `.hash` chain construction for n_syms ∈ {1, 2, 3, ≥4}; `.rela.plt` size assertion (`SIZE_RELA * len(imports)`); `DT_BIND_NOW` constant usage.
3. **`emit_elf_dyn` interp/phdrs ordering invariant**: every `assert len(out) == layout.X_offset` chain across ehdr → phdrs → interp → pad → code → dynstr → dynsym → hash → rela.plt → dynamic → got.plt.
4. **CALL/FFI_CALL arg spill width-gate symmetry**: cycle-77/79 fixed the FFI float-arg/return arms; cycle-103/104 dispositioned the inline `or _is_u64_type` style drift. Re-walk the cycle-77 sibling — the regular CALL path return @1816 — to confirm it remains in the deferred-known set, not in a fresh bucket.
5. **`elf_dyn._imports_set` cache coherence**: confirm `DynLinkInfo.add_import` is idempotent and that `compile_module_to_elf` (line 3171) reads `_imports_set` rather than searching `imports[]` linearly.
6. **Bootstrap `parser.hx` magic-number drift vs frontend**: the cycle-7-vintage comment block at lines 263-280 enumerates 7 raw-200 sites at "parser.hx:4156-4280" (Stage-8 monomorphize_pass). Confirm whether those sites still match the comment's line numbers post Stage 28.11 / 28.13.1 / 28.13.2 inflations.
7. **Bootstrap `kovc.hx` ELF magic numbers vs Python frontend**: confirm `kovc.hx` emit_elf_header (lines 100-115) and emit_program_header (lines 118-130) hard-coded constants (e_type=2, e_machine=62/0x3E, e_phoff=64, e_phentsize=56, e_phnum=1, p_flags=7, ELF_BASE=0x400000, total_filesz=4096+code_size, p_align=4096) remain in lock-step with `helixc/backend/x86_64.py:emit_elf` (lines 2992-3022).
8. **`evaluator.hx` AST-tag table vs current parser.hx**: lines 14-16 of evaluator.hx claim "tag table mirrors helixc/bootstrap/parser.hx: 0 INT, 1 VAR, 2 ADD, 3 SUB, 4 MUL, 5 DIV, 6 LT, 7 IF, 8 LET, 9 NEG, 99 ERR". Confirm these 10 tags match the current parser's `mk_node(N, ...)` sites for the in-scope subset.
9. **`test_codegen.py` accidental skip patterns**: grep for `pytest.skip` / `_SkipTest` / `xfail` markers to confirm no test is silently no-op'd outside the documented Stage 29 K2 SIGILL marker at line 4792.
10. **`test_ir.py` cycle-102 regression discrimination**: re-verify `test_c100_unsigned_cmp_emits_setb_not_setl` and `test_c102_u64_add_emits_64bit_path` byte-pattern uniqueness against the minimal ELF output (entry stub + cmp_u32/add_u64 + main).
11. **Stdlib correctness shallow scan**: walk `helixc/stdlib/*.hx` for an obvious shape problem (file-level magic-number duplication, missing `@pure` on side-effect-free helpers, type-arg drift since Stage 28.11 INC-3b.2).

Concretely cross-referenced:

- `helixc/backend/x86_64.py:139-164` (Asm push/pop/sub_rsp).
- `helixc/backend/x86_64.py:885-901` (`_alloc_array`/`_alloc_slot`).
- `helixc/backend/x86_64.py:906-998` (FnCompiler.compile prologue).
- `helixc/backend/x86_64.py:1764-1820` (CALL emit).
- `helixc/backend/x86_64.py:1824-1894` (FFI_CALL emit).
- `helixc/backend/x86_64.py:1909-1927` (RETURN epilogue).
- `helixc/backend/x86_64.py:2368-2499` (write_file_to_arena callee-saved spill).
- `helixc/backend/x86_64.py:2804-2872` (MODIFY verifier call site).
- `helixc/backend/x86_64.py:2968-3028` (emit_elf static path).
- `helixc/backend/x86_64.py:3034-3190` (compile_module_to_elf).
- `helixc/backend/elf_dyn.py:203-481` (plan_layout + emit_elf_dyn).
- `helixc/bootstrap/evaluator.hx:14-138` (AST-tag table + eval_ast).
- `helixc/bootstrap/parser.hx:226-244` (peek_named_struct_lit), `262-296` (gp_marker_* helpers + raw-200 comment block), `3217-3504` (Stage 28.13.2 named-mode generic branch), `3505-3658` (Stage 28.13.1 named-mode non-generic branch), `4639-4665` / `5796-5941` / `6000-6017` (deferred raw-200 sites).
- `helixc/bootstrap/kovc.hx:80-137` (emit_elf_header/program_header constants).
- `helixc/tests/test_codegen.py:4767-4810` (Stage 29 self-host skip), `:14054-14096` (manual runner skip-types tuple).
- `helixc/tests/test_ir.py:232-279` (cycle-102 regression tests).

---

## Findings table

| Severity   | Count |
|------------|-------|
| CRITICAL (90–100) | 0 |
| HIGH (80–89)      | 0 |
| MED (50–79)       | 0 reportable (below bar) |
| LOW (<50)         | 0 reportable (below bar) |

**Sub-threshold observations** (NOT findings — recorded for transparency only):

- **Cycle-105 has no committed source delta** (conf ~95 that this is not a finding): cycles 104 → 105 are pure markdown (three cycle-104 audit docs). Per audit-gate counter rules, this means the 2/5 advance from cycle-104 is the only baseline change. There is no commit-introduced bug for cycle-105 to surface; the audit's job is to re-walk fresh dimensions and confirm no carry-over defect has been missed.
- **Unstaged `kovc.hx` Stage-29-debug delta** (conf ~70, below threshold): the working tree carries one unstaged change at `helixc/bootstrap/kovc.hx:6549-6565` labeled `STAGE 29 DEBUG (variant 4)`. The patch replaces the three trailing `0x90` NOP bytes that pad an unreachable-branch `ud2` sequence with debug-marker bytes (`top_lo`, `top_hi`, `tl`). Since `0F 0B` (ud2) halts execution at the trap, the trailing three bytes are unreachable regardless of value — semantically a no-op. Instruction-length invariant preserved (5 bytes in, 5 bytes out: `0F 0B XX XX XX`). However, this is debug instrumentation polluting a production source file; it should be reverted, properly gated, or committed as a documented temporary instrumentation if Stage 29 work is still in flight. Below 80 — the user's known Stage-29 SIGILL investigation explains the temporary state, and the audit-gate counter logic specifically scopes to committed history.
- **`parser.hx:265-280` raw-200 site line-number drift** (conf ~70, below threshold): the cycle-7-vintage comment block at parser.hx:265-280 enumerates 7 deferred raw-200 sites at lines 4156, 4157, 4176, 4177, 5453, 5458, 5534. Post Stage 28.11 / 28.13.1 / 28.13.2 the actual sites have drifted to 4639, 4640, 4659, 4660, 5796, 5932, 5936, 5941, 6000, 6017 (now 10 sites, not 7 — the Stage-28.11 INC-3b cycle additions added the extra three). Cycle-104 type-design ALREADY referenced these line numbers as `4156-4280` and dispositioned the issue as deferred-known under cycle-71 narrow-scope discipline (cycle-104 type-design line 165). The comment-block drift itself is a downstream documentation artifact of the deferred-known site set; both the existence and the count are dispositioned. Below 80; calling out the line-number drift specifically would re-flag a cycle-7-deferred-known item which cycle-104 type-design explicitly re-affirmed.
- **`elf_dyn.DT_BIND_NOW` unused constant** (conf ~60, below threshold): `helixc/backend/elf_dyn.py:83` defines `DT_BIND_NOW = 24` but the constant is never used. Grep confirms zero references. The BIND_NOW semantic is conveyed via `DT_FLAGS = DF_BIND_NOW` (line 341) and `DT_FLAGS_1 = DF_1_NOW` (line 342); both are sufficient under the ELF spec for the dynamic linker to honor BIND_NOW. The standalone `DT_BIND_NOW` tag is redundant in the presence of either flag form. Dead-constant; cosmetic. Below 80.
- **`evaluator.hx:14-16` AST-tag table coverage** (conf ~65, below threshold): the comment lists 10 tags (0-9 + 99) and the evaluator handles exactly that subset. The parser today emits ~30+ tags (50=tuple-lit, 51=tuple-cons, 52=field-access, 13=function-call, 16=typed-call, 18=param-decl, 19=GT, 20=EQ, 21=NE, 22=LE, 23=GE, 24=MOD, 26=BIT_NOT, 28-30=BAND/BOR/BXOR, 31=BIT_NEG, 32-33=cmp variants, 50-52=struct-related, 99=ERR, etc.). The evaluator's tag-table comment scopes itself to the Stage-3 evaluator subset ("AST tag table mirrors helixc/bootstrap/parser.hx" — `mirrors` interpreted as `mirrors the subset the evaluator cares about`). The evaluator was the Stage-3 tree-walker; production codegen runs through `kovc.hx`, not this evaluator. So this is not stale-per-se; it is correctly scoped to the evaluator's reach. Below 80.
- **`test_codegen.py` skip-handling** (conf ~85 the implementation is CORRECT, below 80 as a finding): the one `pytest.skip()` call at `test_codegen.py:4792` is wired through both the pytest collector and the manual `__main__` runner (lines 14054-14068) via the `Skipped` outcome class. The cycle-104-era comment block at lines 4774-4792 thoroughly documents the Stage 29 K2 SIGILL state and the rationale for re-skipping. The grep for `pytest.skip` / `skipif` / `@pytest.mark.skip` returns no other skip sites in the codegen suite. No accidental coverage gap. PASS at conf ≥ 75 — not a finding.
- **Cycle-102 regression-test discrimination re-confirmed** (conf ~90 the tests discriminate, below 80 as a finding): `test_c100_unsigned_cmp_emits_setb_not_setl` (test_ir.py:232) asserts presence of `0F 92 C0` (setb al). The minimal compiled module emits one cmp arm in `cmp_u32`; no other emit site in the entry stub or in `main` produces this opcode triple. `test_c102_u64_add_emits_64bit_path` (test_ir.py:256) asserts presence of `48 01 C8` (rex.W add rax, rcx). The minimal module's only ADD path is u64; the entry stub uses `mov eax, imm32; syscall`, not rex.W ADD. Both asserts discriminate the pre-fix vs post-fix code paths. PASS at conf ≥ 75 — not a finding.
- **Stage 29 K2 SIGILL probe scripts placement** (conf ~85 the placement is CORRECT, below 80 as a finding): `_probe_stage29_capture.py`, `_probe_stage29_diff.py`, `_probe_stage29_main_trace.py` are placed in `helixc/tests/` with a leading underscore — pytest's default-collection rule excludes underscore-prefixed test files, so these diagnostic probes do not pollute the test run. They are currently untracked (git status shows them as `??`); whether to track them or move to a dedicated `helixc/tools/` directory is a follow-on housekeeping question, not a defect. Below 80.

---

## Cross-stage cuts examined

### SysV ABI stack-alignment chain (PASS at conf ≥ 80)

The alignment chain holds end-to-end:

- **Kernel → `_start`**: kernel ABI ensures rsp = 0 mod 16 at `_start` entry.
- **`_start` → entry_fn**: `call_rel32(entry_fn)` is emitted at compile_module_to_elf:3063. Pre-call rsp = 0 mod 16; post-call (entry_fn entry) rsp = 8 mod 16 ✓ (function-entry invariant).
- **entry_fn prologue**: `push rbp` (-8 → rsp = 0 mod 16); `mov rbp, rsp`; `sub rsp, frame_size` where `frame_size = (-self.next_slot + 15) & ~15` is 16-aligned (line 939). rsp inside fn = 0 mod 16 ✓.
- **entry_fn → callee via `call_rel32`** (line 1808 etc.): pre-call rsp = 0 mod 16, post-call (callee entry) rsp = 8 mod 16 ✓.
- **entry_fn → libc via `call_qword_ptr_rip_rel_ffi`** (line 1878, 3070): pre-call rsp = 0 mod 16; post-call rsp = 8 mod 16 ✓. The `mov_edi_eax` between the user-fn return and the libc exit-call at lines 3064-3070 doesn't touch rsp.
- **MODIFY → verifier `call_rel32`** (line 2851): the MODIFY emit path uses ECX/EDI/ESI/imm-loads only between the prologue setup and the call — no push/pop. rsp stays 0 mod 16 inside the function; call-site requirement met.
- **write_file_to_arena callee-saved spill** (lines 2370-2499): `push rbx; push r12; push r13; push r14` = 4 × 8 = 32 bytes (still 0 mod 16 post-push); `sub rsp, 16` keeps 0 mod 16. The intervening `syscall` instructions have no alignment requirement. Symmetric tear-down at lines 2495-2499 restores alignment. ✓
- **entry stub fallback raw `sys_exit`** (line 3074-3075): `mov eax, 60; syscall` — syscalls have no alignment requirement; the fact that rsp was 0 mod 16 here is incidental.

No alignment violation on any path probed. PASS.

### ELF dynamic-link tables algebraic invariants (PASS at conf ≥ 80)

- **`n_dyn_entries` assert**: `n_dyn_entries = len(needed_libs) + 12`; the explicit appends in `plan_layout` are: 1 per needed_lib (= len(needed_libs)) + DT_HASH + DT_STRTAB + DT_SYMTAB + DT_STRSZ + DT_SYMENT + DT_PLTGOT + DT_PLTRELSZ + DT_PLTREL + DT_JMPREL + DT_FLAGS + DT_FLAGS_1 + DT_NULL (12 non-NEEDED entries). Total matches `n_dyn_entries`. ✓
- **SYSV `.hash` chain construction** for the four shapes:
  - n_syms = 1 (UND only, no imports): bucket = [0], chain = [0]. Length = 1 = n_syms. ✓ (Unreachable in practice: `emit_elf_dyn` is only called when `dyn.has_imports()`.)
  - n_syms = 2 (1 import): bucket = [1], chain = [0, 0]. bucket[0]=1 → sym 1, chain[1]=0 (terminate). ✓
  - n_syms = 3 (2 imports): bucket = [1], chain = [0, 2, 0]. bucket[0]=1 → sym 1, chain[1]=2 → sym 2, chain[2]=0. ✓
  - n_syms ≥ 4 (3+ imports): chain[0]=0, chain[i] = i+1 for i ∈ [1, n_syms-2], chain[n_syms-1] = 0. Single-bucket chain walks all syms. ✓
- **`.rela.plt` size**: `rela_plt_size = SIZE_RELA * len(imports)`; `assert len(rela_plt_bytes) == rela_plt_size` enforces it. ✓
- **`_imports_set` cache**: `DynLinkInfo.add_import` is idempotent via the dict guard; `compile_module_to_elf:3171` reads `_imports_set[fx.symbol]` — O(1) lookup. ✓

### `emit_elf_dyn` file-layout ordering (PASS at conf ≥ 80)

The chain of `assert len(out) == layout.X_offset` at emit_elf_dyn:459, 463, 468, 470, 472, 474, 476, 478, 480 enforces the file layout invariant byte-for-byte:

```
ehdr (64B) → phdrs (n*56B) → interp_str → zero-pad → code → 8B-align-pad
→ dynstr → dynsym → hash → rela.plt → dynamic → got.plt
```

If any region's emitted size disagrees with the layout-planned offset, the assert fires immediately rather than producing a silently-corrupt ELF. ✓ Defensive assertion pattern is the same shape used by `emit_elf` (single-PT_LOAD path) at compile_module_to_elf:3025 where `pad_size = CODE_OFFSET - len(ehdr) - len(phdr)` derives the pad from declared sizes.

### Bootstrap-frontend ELF-constant lock-step (PASS at conf ≥ 80)

`kovc.hx` open-codes:

- `e_type = 2` (ET_EXEC), `e_machine = 62` (EM_X86_64 = 0x3E), `e_version = 1`, `e_ehsize = 64`, `e_phentsize = 56`, `e_phnum = 1`, `e_phoff = 64` — all match `emit_elf` byte-for-byte (x86_64.py:3000-3011).
- `p_type = 1` (PT_LOAD), `p_flags = 7` (R|W|X), `ELF_BASE = 0x400000`, `total_filesz = 4096 + code_size` (i.e. CODE_OFFSET + len(code)), `p_align = 4096 = 0x1000` (= CODE_OFFSET) — all match emit_elf at x86_64.py:3014-3022.
- `e_entry = 0x401000 = ELF_BASE + CODE_OFFSET = ELF_BASE + ENTRY_OFFSET` — matches x86_64.py:2987 `code_vaddr = ELF_BASE + entry_offset` with default `entry_offset = ENTRY_OFFSET = 0x1000`.

No drift in the Phase-0 single-PT_LOAD shape. Cycle-82 silent-failures (line 91) already dispositioned the open-coding as a documented Phase-0 design choice (no SoT extraction yet). PASS — not a finding.

---

## Positive observations (no finding)

- **No source drift since cycle-103**: the 26dfa82 → 77e4b85 delta is committed-pure-markdown (three cycle-103 audit docs + three cycle-104 audit docs). The discipline of separating audit-doc commits from source commits preserves audit-gate integrity through two consecutive no-source cycles.
- **Cycle-104 sub-threshold notes carry forward correctly**: every below-80 observation cycle-104 disposed (test placement, local imports, edge-case coverage of SUB/MUL, sibling 64-bit width-gate sites, struct-mono ↔ backend predicate, PAT_OR drain ↔ bn_state threading) remains accurately classified in cycle-105. The deferred-known carve-outs are stable across two clean cycles.
- **Cycle-102 helper `_is_64bit_int_type`** continues to correctly route the four-element set {i64, isize, u64, usize} to the 64-bit ADD/SUB/MUL paths at x86_64.py:1329, 1359, 1387. Helper body remains a thin disjunction of `_is_i64_type` ∨ `_is_u64_type`, so any future widening of either sub-predicate flows through automatically.
- **Cycle-100 helper `_is_unsigned_int_type`** + the cmp dispatch at x86_64.py:1672-1676 remains the correct routing for setb/setbe/seta/setae vs. signed setl/setle/setg/setge. Cycle-103 V2 already dispositioned the inline `or _is_u64_type` style drift at 1672-1676 / 1872 / 1891 as below 75; cycle-105 re-walks and confirms.
- **`emit_elf_dyn` defensive offset asserts** form a runtime sanity check on every emitted ELF — the bootstrap-cycle 28.11 didn't add any new region to the `.dynamic` shape, so the assert chain remains complete and load-bearing.
- **Stage 29 K2 SIGILL probe scripts** (`_probe_stage29_capture.py` / `_probe_stage29_diff.py` / `_probe_stage29_main_trace.py`) are correctly placed in `helixc/tests/` with a leading underscore — pytest's default-collection rule excludes underscore-prefixed test files, so these diagnostic probes do not pollute the test run.

---

## Verdict

**PASS** — zero findings at confidence ≥ 80 within the cycle-105 scope.

No committed source delta since cycle-103 baseline (two consecutive no-source cycles). Re-walked dimensions (stack-alignment chain end-to-end, ELF dynamic-link table invariants, file-layout ordering asserts, bootstrap-frontend ELF-constant lock-step) all hold. The cycle-101 codereview F2 tail, cycle-101 silent-failures F1, the full cycle-57 / cycle-104 `_is_i64_type`-only sibling enumeration, cycle-16 `_alloc_array` `elem_size`, cycle-82 ELF-constant open-coding, cycle-7 raw-200 deferred-known sites in parser.hx, and cycle-103 inline-`or _is_u64_type` style drift remain stable in the deferred-known set and are not re-flagged per cycle-105 scope rules.

Stage 28.9 audit-gate counter advances **2/5 → 3/5**.

---

## Cross-reference

- **cycle 101** (`docs/audit-stage28-9-cycle101-codereview.md`): FAIL — F1 (missing regression test) + F2 (DIV/MOD/SHR signed-only).
- **cycle 102** (commit `26dfa82`): closed 3 of 4 cycle-101 findings; deferred 2 (DIV/MOD/SHR + A.StrLit).
- **cycle 103** (`docs/audit-stage28-9-cycle103-codereview.md`): PASS, 0 findings. Audit-gate counter 0/5 → 1/5.
- **cycle 104** (`docs/audit-stage28-9-cycle104-codereview.md`, HEAD `31e1725`): PASS, 0 findings. Audit-gate counter 1/5 → 2/5.
- **cycle 105** (this doc, HEAD `77e4b85`): PASS, 0 findings. Audit-gate counter 2/5 → 3/5.

---

## No code edits made

Strict read-only audit. One Write of this doc only. No Edit calls, no source mutation, no test run.
