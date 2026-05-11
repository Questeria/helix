# Stage 28.8 pre-29 audit gate — Cycle 21 (Audit A: silent failures)

**Date:** 2026-05-11
**HEAD:** `bee36e6` ("Audit 28.8 cycle 21 fix-sweep: close C20-1 (HIGH,
PTX backend isize/usize silent 32-bit)")
**Lens:** silent failures (Audit A)
**Streak counter at start:** 0/5 (reset by C20-1 in cycle 20 Audit B)

---

## Scope

Strict-criterion read-only audit. Two questions:

1. **Verification.** Did the cycle-21 fix-sweep correctly close C20-1?
   Specifically: do the 4 PTX width-keyed tables now consistently treat
   `isize` / `usize` as 64-bit aliases of `i64` / `u64`, matching the
   canon established by `typecheck.py` ↔ `x86_64` backend ↔
   `const_fold._INT_BITS`?

2. **New-finding sweep.** Is there a **7th** instance of the
   "silent width narrowing" defect class anywhere else in the
   codebase — i.e., any other width-keyed dict / set / tuple / branch
   that disagrees with the canon by omitting `isize` / `usize`?

The prior six HIGH findings of this defect class are: **C13-1, C16-1,
C18-1, C18-B / C18-C, C19-1, C20-1.** Each one extended a different
width-keyed site to include `isize` / `usize` (or to enforce
narrow-but-loud failure). The C20-1 fix-sweep closed `helixc/backend/
ptx.py`'s 4 tables. After C20-1, the patched sites are:

| Site | File:line | Status at HEAD bee36e6 |
|------|-----------|------------------------|
| `_is_i64_type` classifier | `helixc/backend/x86_64.py:1005-1011` | Includes `isize` |
| `_is_u64_type` classifier | `helixc/backend/x86_64.py:1013-1017` | Includes `usize` |
| `_check_array_elem_size_supported` wide-widths set | `helixc/backend/x86_64.py:1042` | Includes `isize`, `usize` |
| `_INT_BITS` (const-fold wrap) | `helixc/ir/passes/const_fold.py:43-56` | `isize: 64`, `usize: 64` |
| `_ptx_type_str` mapping | `helixc/backend/ptx.py:166-172` | `isize: .b64`, `usize: .b64` |
| `_DTYPE_SIZE` | `helixc/backend/ptx.py:340-342` | `isize: 8`, `usize: 8` |
| `_DTYPE_PTX_LOAD` | `helixc/backend/ptx.py:343-347` | `isize: s64`, `usize: u64` |
| `_ld_reg_prefix` | `helixc/backend/ptx.py:355-363` | `isize`, `usize` ∈ rd-pool |

For the new-finding sweep, the methodology is the cycle-19 adversarial
rotation: enumerate every width-keyed site reachable from the
canonical pipeline, and verify each one is consistent with the canon.

---

## Verification of cycle-21 fix-sweep (closes C20-1)

### `helixc/backend/ptx.py` — all 4 tables checked

**`_ptx_type_str` (line 173):** `mapping.get(ty.name, ".b32")`. The
mapping at lines 166-172 now contains `"isize": ".b64", "usize":
".b64"`. The 32-bit default is now reachable only for genuinely
unknown scalar names (a TIRScalar with a non-canonical name, which
typecheck rejects upstream). Consistent with canon.

**`_DTYPE_SIZE` (line 340):** Now contains `"isize": 8, "usize": 8`
alongside `"i64": 8, "u64": 8`. The `_dtype_size` accessor's `.get(
dtype, 4)` default at line 350 is now reachable only for unknown
dtype names. Consistent with canon.

**`_DTYPE_PTX_LOAD` (line 343):** Now contains `"isize": "s64",
"usize": "u64"` alongside `"i64": "s64", "u64": "u64"`. The accessor's
`.get(dtype, "u32")` default at line 353 is now reachable only for
unknown dtype names. Consistent with canon.

**`_ld_reg_prefix` (lines 355-363):** The 64-bit `rd` pool branch is
now `if dtype in ("i64", "u64", "isize", "usize")`. The fall-through
to `"r"` (32-bit pool) is now reachable only for unknown dtype names.
Consistent with canon.

**Regression test pin:** `helixc/tests/test_ptx.py:237-262`
(`test_c20_1_isize_usize_treated_as_64_bit_in_ptx`) directly asserts
each of the 4 sites. Test run confirms 23/23 PTX tests pass at
HEAD bee36e6 (was 22 → +1).

**C20-1 verdict:** correctly and completely closed.

---

## New-finding sweep — adversarial rotation

Enumerated all width-keyed sites in `helixc/`. Each is classified as
**clean** (matches canon), **gated-unreachable** (default branch
exists but no production path can reach it), **out-of-class** (not a
width table), or **finding** (a 7th instance).

| Site | File:line | Classification |
|------|-----------|----------------|
| `_WIDEN_NAME_ALIASES` | `typecheck.py:225-228` | clean (defines canon) |
| `_WIDEN_RANK` | `typecheck.py:235-254` | clean (`isize: 40 == i64: 40`, `usize: 41 == u64: 41`) |
| `PRIMITIVES` set | `typecheck.py:336-343` | out-of-class (membership only) |
| `_INT_BOUNDS` | `typecheck.py:1807-1818` | clean (`isize`, `usize` keyed at 64-bit ranges) |
| `_INT_BITS` (const-fold) | `const_fold.py:43-56` | clean (closed by C19-1) |
| `_PRIMITIVE_TYPE_NAMES` | `lower_ast.py:356-362` | out-of-class (membership only) |
| `NUMERIC_FOR_AD` | `autodiff.py:73-79` | out-of-class (membership only) |
| `_is_i64_type` | `x86_64.py:1005-1011` | clean (closed by C18-1) |
| `_is_u64_type` | `x86_64.py:1013-1017` | clean (closed by C18-1) |
| `_check_array_elem_size_supported` | `x86_64.py:1030-1050` | clean (closed by C16-1, extended for isize/usize) |
| `_ptx_type_str` | `ptx.py:157-176` | clean (closed by C20-1) |
| `_DTYPE_SIZE` | `ptx.py:340-342` | clean (closed by C20-1) |
| `_DTYPE_PTX_LOAD` | `ptx.py:343-347` | clean (closed by C20-1) |
| `_ld_reg_prefix` | `ptx.py:355-363` | clean (closed by C20-1) |
| `tile_ir` width tables | `ir/tile_ir.py` | none exist (no width keying) |
| `kovc` (bootstrap Helix tests) | `kovc/**` | no isize/usize refs |

### Borderline cases reviewed and explicitly cleared

**`typecheck.py:1849` — `_suggest_wider_int`.** Iterates over
`("i32", "i64")` only; does NOT consider `isize` / `usize`. This is
the **diagnostic-hint** path: when an integer literal exceeds the
declared type's range, the typecheck error has already been raised
(line 1842); `_suggest_wider_int` only generates the optional `hint:
use \`<type>\` instead` text. Returning `None` (no hint) when isize
would have fit is a hint-quality issue, not a silent-narrowing
miscompile — the error itself is loud. **Not a finding.** (Stage-29
cosmetic improvement, not in the silent-failure defect class.)

**`typecheck.py:1634` — `loop_var_ty = iter_ty if iter_ty is not None
else TyPrim("i64")`.** Default loop-variable type for `for i in
range_expr` when range type can't be inferred. Hard-coded i64 (not
i32), which is the WIDE direction — overshoot, not narrowing. **Not
a finding.**

**`x86_64.py:2717` — `value_kind = op.attrs.get("value_kind",
"i32")`.** Default `value_kind` for `Modify` op verifier-call ABI.
The `value_kind` value space is `("i32", "f32")` — a 2-way
domain-discriminator, not a scalar-width table. Reachable values
come from the IR lowering of `Modify`, which only emits `"i32"` or
`"f32"`. The `i32` default is for backward-compat with old IR
modules that lack the attribute. **Not in defect class.**

**`lower_ast.py:1037-1038` — `IntLit` un-suffixed lowering.** Lowers
to `const_int(expr.value, expr.type_suffix or "i32")` regardless of
contextual `let x: isize = ...` declared type. **This is forward note
F-20-T-D-4 from cycle 20 Audit B** (the type-design lens). Cycle 20
explicitly classified it as "reachable but currently masked by the
fact that typical isize-typed bindings use the `_isize` suffix" and
recorded it as a forward note rather than a finding. Per the strict
re-flag rule, forward notes from prior cycles are **not** re-cited
as new findings — they remain on the Stage-29 backlog. **Not a
cycle-21 finding.**

**`lower_ast.py:353-355` — generic-type-param TIR lowering.** "Generics
silently lower to TIRScalar('T') with i32-sized ABI today; this is
correct for i32 type args and silently wrong for i64+." Documented
HBS limitation called out in the source comment, predates the
silent-width-narrowing defect class, and is a Stage-29-class lift
(monomorphization layer, not the scalar-width predicate). **Not in
the C13/16/18/19/20 defect class** — different layer, different
fix. Pre-existing acknowledged limitation, not a regression. Not a
cycle-21 finding.

**`ptx.py:152-155` — `_format_param` all-`.b64` hard-code.** "We
treat all params as `.b64` (pointer-like)." Overshoot (always 64-bit),
not narrowing. **Not in defect class.** Recorded as forward note 3
in cycle 20 Audit B.

**`ptx.py:181-237` — PTX `SCALAR_*` ops unconditionally 32-bit.**
"`SCALAR_CONST_INT` emits `mov.b32`; `SCALAR_ADD` emits `add.s32`;
etc." This is the same defect class (silent narrowing on 64-bit
operands) but at the **scalar-op layer**, not the dtype-table layer.
Cycle 20 Audit B forward note 2 already recorded it. It is **gated
unreachable** in production: the HBM-tile dtype allowlist at
`lower_ast.py:511` (the f32/i32/f16/bf16 set) plus the kernel-param
.b64 hard-code mean no isize/usize-typed value reaches these scalar
ops today. The window opens only if (a) the HBM-tile dtype allowlist
is widened, or (b) `emit_device_func` upgrades from stub bodies.
**Not currently reachable → not a finding.** Same standing as the
cycle-20 forward note F-20-1 (which became C20-1 only when one of
the gating conditions threatened to open).

### Empty-default `dict.get` audit (PTX backend)

The cycle-20 Audit B's key insight was that PTX width-table sites
used `.get(key, default)` rather than `[key]` — silent rather than
loud on miss. Re-scanned all `.get(` calls in `ptx.py` and `x86_64.
py` for width-domain keys:

- `ptx.py:173 mapping.get(ty.name, ".b32")` — closed by C20-1.
- `ptx.py:350 self._DTYPE_SIZE.get(dtype, 4)` — closed by C20-1.
- `ptx.py:353 self._DTYPE_PTX_LOAD.get(dtype, "u32")` — closed by
  C20-1.
- `ptx.py:72 self.next_reg_by_prefix.get(prefix, 0)` — register-pool
  counter, not width-keyed. Out of class.
- `ptx.py:183, 242, 250, 271-272, 288, 313 op.attrs.get(...)` —
  attribute defaults, not width-keyed dispatch. Out of class.
- `ptx.py:192-228 self.reg_map.get(...)` — register-map lookups for
  operands, not width-keyed dispatch. Out of class.
- `ptx.py:289, 314 self.hbm_param_map.get(name)` — name lookup, not
  width-keyed. Out of class.

No additional width-keyed `.get(..., default)` sites identified.

---

## Verdict

**Cycle 21 silent-failures audit: CLEAN.**

| Severity   | Count |
|------------|-------|
| CRITICAL   | 0     |
| HIGH       | 0     |
| MEDIUM     | 0     |
| LOW        | 0     |
| **Total**  | **0** |

- C20-1 closed at HEAD `bee36e6`; all 4 PTX width tables now treat
  `isize` / `usize` as 64-bit aliases. Regression test
  `test_c20_1_isize_usize_treated_as_64_bit_in_ptx` pins the contract.
- Adversarial rotation found no 7th instance of the silent-width-
  narrowing defect class. All width-keyed sites enumerated by the
  cycle-19 rotation remain consistent with the canon at HEAD bee36e6.
- Borderline cases (`_suggest_wider_int`, `loop_var_ty` default,
  `value_kind` default) reviewed and classified as out-of-class.
- Pre-existing forward notes from cycle-20 Audit B (PTX scalar-op
  narrowness, IntLit context-insensitivity, generic-param TIR ABI)
  remain on Stage-29 backlog; per the strict re-flag rule they are
  not re-cited as cycle-21 findings.

**Clean-cycle counter:** was 0/5 → **advances to 1/5** (cycle 21 is
the first clean cycle of the new strict-criterion streak, post-C20-1
reset).

Four more consecutive clean cycles required to fire the Stage-29
gate. Any new finding before then resets the counter.

---

## Out-of-scope per task instructions

The "centralize scalar-width predicate" recommendation (carry from
cycle-17 / -18 / -19 / -20 forward notes) is **explicitly out of
scope** per the cycle-21 prompt. Six consecutive HIGH findings of
the same defect class (C13-1, C16-1, C18-1, C18-B/C, C19-1, C20-1)
makes the refactor overdue, but the cycle-21 prompt instructs: "do
NOT pursue the 'centralize scalar-width predicate' refactor
recommendation (that's a Stage-29-class refactor out of scope)."
Recording compliance with that instruction. Forward note remains
open on the Stage-29 backlog where cycle 20 left it.

---

## Files touched by this audit

None — this is a read-only audit cycle. No production-code or test
edits. Only this doc.

## Cross-reference

- Cycle 20 silent-failures (Audit A, declared CLEAN with forward
  notes): `docs/audit-stage28-8-cycle20-silent-failures.md`.
- Cycle 20 type-design (Audit B, surfaced C20-1 via re-
  classification of cycle-18/19 forward note 2):
  `docs/audit-stage28-8-cycle20-type-design.md`.
- Cycle 21 fix-sweep commit (closes C20-1): `bee36e6`.
- Files touched by cycle-21 fix-sweep: `helixc/backend/ptx.py:157-
  172, 334-363`, `helixc/tests/test_ptx.py:237-262`.
- The 6 prior HIGH findings of this defect class:
  - C13-1 (cycle 13): closed in fix-sweep.
  - C16-1 (cycle 16): `_check_array_elem_size_supported` narrow-but-
    loud gate.
  - C18-1 (cycle 18 Audit A): x86_64 `_is_i64_type` / `_is_u64_type`
    extended.
  - C18-B / C18-C (cycle 18 Audit B): adjacent x86_64 width sites.
  - C19-1 (cycle 19): `const_fold._INT_BITS` extended.
  - C20-1 (cycle 20 Audit B): PTX 4 width tables extended (closed
    by cycle-21 fix-sweep).
