# Audit Stage 28.9 cycle 99 — Silent failures

Scope: HEAD 32c66bf (Stage 28.9 cycle-98 audit clean, 1/5 after cycle-96 reset).

No code change since cycle-97 fix (3b065d2). Deferred-known carve-outs
(monomorphize._mangle_ty / hash_cons._ast_equal silent catchalls;
typecheck/struct_mono pre-flatten in check.py;
autotune.collect_autotuned_fns missing iter_fn_decls;
struct_mono.mangle_struct collision) NOT re-flagged.

Rotated fresh scope:
- helixc/frontend/parser.py: rare-path rules NOT in autotune/literal flow
- helixc/ir/passes/const_fold.py: NaN/Inf semantics in fold paths
- helixc/backend/x86_64.py: integer comparison emit (CMP_LT/GT/EQ)

Prior C1–C98 findings + deferred-known NOT re-flagged. Independent Stage
28.10 / 28.11 cycle activity ignored (scope-narrowed to Stage 28.9).

---

## Verdict: FAIL — 2 findings at conf >= 75%

### F1 (HIGH, conf 92%) — backend cmp emits signed setcc and 32-bit cmp for u64/usize operands

File: `helixc/backend/x86_64.py` lines 1607–1616.

The integer comparison dispatch chooses the 64-bit path only when
`_is_i64_type` returns true (i.e. operand type name in `{"i64", "isize"}`).
For `u64` / `usize` operands the branch falls through to the 32-bit `else`
path, which (a) emits `mov eax, [..]` truncating the high 32 bits of the
64-bit value into oblivion, and (b) selects from `int_cmp_setters` whose
LT/GT/LE/GE entries are signed `setl/setle/setg/setge`. So
`let a: u64 = 0x1_0000_0000_u64; if a < 1_u64 { ... }` silently miscompiles
to "compare low-32-bits-of-a (= 0) against 1, signed-less-than → true",
the inverse of the IEEE/Rust semantics this codebase otherwise honors.
The companion test at `tests/test_codegen.py:3132` (comment claims "REX.W
cmp + setb") only exercises values 5 and 10 — both fit signed-32 and the
sign of the signed-cmp accidentally matches the unsigned-cmp here, so the
test passes despite the bug. `_is_u64_type` already exists and is wired
into arg-load / result-spill at lines 1797 and 1816 — the omission at the
cmp site is asymmetric with the rest of the backend and matches the
cycle-19 C18-1 "isize alias miss" defect class one step over.

### F2 (HIGH, conf 88%) — backend cmp emits signed setcc for u32/u8/u16 operands

File: `helixc/backend/x86_64.py` lines 1612–1616.

The fall-through `else` clause uses signed setcc (`setl/setle/setg/setge`)
unconditionally, with no `_is_unsigned`-style discrimination. There is no
`_is_unsigned` helper anywhere in the backend (grep `is_unsigned` returns
zero hits). So `let a: u32 = 0xFFFF_FFFF_u32; if a < 1_u32 { ... }` emits
`cmp eax, ecx ; setl al` and returns 1 (because the 32-bit signed view of
`0xFFFFFFFF` is `-1`, which is less than `1`), but the correct unsigned
answer is 0 (`4_294_967_295 < 1` is false). Same defect class as F1, but
in the 32-bit path; affects every `u8`/`u16`/`u32` operand whose high bit
is set. No upstream typecheck guard rejects unsigned comparisons (a grep
for `CMP_LT_U` / `_unsigned` returns no IR-level opcode split). Parser
emits a single `CMP_LT` op for `<` regardless of operand signedness
(`helixc/ir/lower_ast.py:1105`), so the backend is the sole place where
signedness has to be honored, and it is not.

---

## No code edits made

This audit is strict read-only. ONE Write of this doc only — no Edit
calls, no source mutation, no test run. Surfaces F1 and F2 to the
Stage 28.9 fix-sweep for the next cycle.
