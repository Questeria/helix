# Stage 35 Restart 58 — Lane A (Runtime/Safety) Audit Report

**HEAD**: `c8398d3` (Fix Stage 35 fifty-eighth restart findings — actually
the title-drift +1 source-only commit for the carry-forward NaN-skip
family; restart number per the ledger is 58)
**Date**: 2026-05-16
**Lane**: A (Runtime/Safety)
**Discipline**: read-only audit. Findings landed in the restart 58
catch-up sweep (Increment 77).

## Carry-forward family check: tf1d NaN-skip siblings

The c8398d3 commit closed 3 of the 4 carry-forward siblings logged by
restart 57 for restart 58's Lane A:

- `tf1d_dot` (tensor.hx ~562) — NaN-skip applied via `if prod == prod`. CLEAN.
- `tf1d_l1_norm` (tensor.hx ~1170) — NaN-skip applied via `if v == v`. CLEAN.
- `tf1d_max_abs` (tensor.hx ~1344) — NaN-skip applied via `if v == v`. CLEAN.
- `tf1d_sum_in_range` (tensor.hx ~1445) — **NOT FIXED in c8398d3**.
  Captured below as A1.

## Findings (1 HIGH + 5 MEDIUM + 1 LOW)

### A1 HIGH — `tf1d_sum_in_range` NaN-skip (missed carry-forward sibling)

- **File**: `helixc/stdlib/tensor.hx` ~1445 `tf1d_sum_in_range`.
- **Bug**: bare `total = total + __f32_from_bits(__arena_get(start + i));`
  in the inner loop. Single NaN slot poisons the whole partial sum
  (NaN + anything = NaN), breaking any windowed reduction caller.
- **Family**: direct sibling of restart 57 A1 `tf1d_sum` and the three
  c8398d3 siblings (`tf1d_dot`, `tf1d_l1_norm`, `tf1d_max_abs`).
- **Severity**: HIGH — explicit carry-forward classification per restart
  57's HANDOFF.
- **Fix**: `let v = ...; if v == v { total = total + v; }` pattern,
  identical to `tf1d_sum`.
- **Canary**: `test_stage35_restart58_tf1d_sum_in_range_nan_skip_fails_closed`
  in `helixc/tests/test_codegen.py`.

### A2 MEDIUM — `vec_map_abs` returns INT32_MIN for INT32_MIN input

- **File**: `helixc/stdlib/iterators.hx` ~221 `vec_map_abs`.
- **Bug**: `if v < 0 { __arena_push(0 - v); }`. When `v == INT32_MIN`,
  `0 - v` wraps back to INT32_MIN, silently breaking the `|x| >= 0`
  postcondition.
- **Family**: direct sibling of `vec_map_neg` / `vec_negate_inplace`
  (restart 51 A5) and `ti1d_max_abs` / `vec_max_abs` (restart 56 A2/A3).
- **Severity**: MEDIUM — companion of those HIGH-class wrap bugs;
  rated MEDIUM because most callers follow up with max/saturate.
- **Fix**: special-case INT32_MIN to saturate to INT32_MAX.
- **Canary**: `test_stage35_restart58_vec_map_abs_saturates_on_int32_min`.

### A3 MEDIUM — `tf1d_dot_with_offset` NaN poisons whole product

- **File**: `helixc/stdlib/tensor.hx` ~1528 `tf1d_dot_with_offset`.
- **Bug**: bare `total = total + av * bv;`. Identical mechanism to the
  `tf1d_dot` fix that c8398d3 applied; the offset variant was missed.
- **Family**: direct sibling of `tf1d_dot` (c8398d3 A1).
- **Severity**: MEDIUM — sibling of an A1 HIGH; rated MEDIUM because the
  offset form is less commonly called.
- **Fix**: `let prod = av * bv; if prod == prod { total = total + prod; }`.
- **Canary**: `test_stage35_restart58_tf1d_dot_with_offset_nan_skip_fails_closed`.

### A4 MEDIUM — `tf2d_matvec` / `tf2d_matmul` per-cell NaN poisoning

- **File**: `helixc/stdlib/tensor.hx` ~627 `tf2d_matvec`, ~1005 `tf2d_matmul`.
- **Bug**: bare `acc = acc + wv * xv` (matvec) / `acc = acc + av * bv`
  (matmul). One NaN in W (resp. A) poisons a whole output row; one NaN
  in x (resp. B) poisons every output cell.
- **Family**: extends the dot-product NaN-skip discipline from `tf1d_dot`
  to the 2D layer; integer twins `ti2d_matvec` / `ti2d_matmul` already
  have i64 saturation (restart 52 A1).
- **Severity**: MEDIUM — one bad input only poisons one output cell, not
  the entire result.
- **Fix**: `let prod = ... * ...; if prod == prod { acc = acc + prod; };`.
- **Canary**: `test_stage35_restart58_tf2d_matvec_nan_skip_per_cell`
  (matmul shares the same idiom).

### A5 MEDIUM — `tf2d_row_sum` / `tf2d_col_sum` / `tf2d_trace` NaN poisoning

- **File**: `helixc/stdlib/tensor.hx` ~1464 (row_sum), ~1488 (col_sum),
  ~1591 (trace).
- **Bug**: all three use the same `acc = acc + __f32_from_bits(...)`
  pattern as the fixed `tf1d_sum`. One NaN poisons the row (resp.
  column, the whole trace).
- **Family**: same as restart 57 A1 `tf1d_sum`; the 2D variants were
  missed by the previous sweep.
- **Severity**: MEDIUM — bounded to per-row/col output, not whole result.
- **Fix**: NaN-skip pattern identical to A1.
- **Canary**: `test_stage35_restart58_tf2d_row_sum_nan_skip`.

### A6 MEDIUM — `mse_loss_f32` / `mae_loss_f32` NaN poisons loss

- **File**: `helixc/stdlib/nn.hx` ~305 `mse_loss_f32`, ~1088
  `mae_loss_f32`.
- **Bug**: bare `total = total + d * d` (mse) / `total = total + __abs(...)`
  (mae). One NaN slot makes the whole batch loss NaN, breaking the
  "garbage in one slot, partial loss in output" convention.
- **Family**: extends restart 57 A1 to the loss helpers.
- **Severity**: MEDIUM — training loops do see NaN as a fail-loud
  signal, but the codebase convention is NaN-skip discipline.
- **Fix**: NaN-skip with divisor held at `n` (matches `tf1d_sum`).
- **Canary**: `test_stage35_restart58_mse_loss_f32_nan_skip`.

### A7 LOW — `tf1d_max/min/argmax/argmax_in_range/argmin/argmax_rows_f32` NaN-at-index-0

- **Files**:
  - `helixc/stdlib/tensor.hx` ~874 (`tf1d_max`), ~889 (`tf1d_min`),
    ~904 (`tf1d_argmax`), ~1231 (`tf1d_argmin`), ~1423 (`tf1d_argmax_in_range`).
  - `helixc/stdlib/nn.hx` ~928 (`argmax_rows_f32`).
- **Bug**: bare-init `best = arena_get(start)` freezes the running best
  to NaN if index 0 (resp. lo, col 0) holds NaN — `v > NaN` is false
  per IEEE-754, so every subsequent slot is silently skipped.
- **Family**: the bare-init NaN-at-zero idiom shared across all f32
  max/min/argmax-style reductions. Partially shielded by downstream
  fail-closed callers (e.g. `softmax_layer` catches `tf1d_max`'s
  contribution), but the argmax* surface is user-facing.
- **Severity**: LOW — NaN at index 0 is rare in practice; downstream
  shielding catches the most common path.
- **Fix**: replace bare-init with the `seen = 0` sentinel pattern that
  adopts the first non-NaN slot. All-NaN input falls back to the
  pre-existing "empty input" return value (0.0 for max/min; 0 / lo
  for argmax indices).
- **Canary**: `test_stage35_restart58_tf1d_argmax_skips_nan_at_index_0`
  (covers max + argmax as a family canary).

## Sibling sweep tables

| Finding | Audited-clean siblings (verified at HEAD c8398d3) |
|---|---|
| A1 (`tf1d_sum_in_range`) | `tf1d_sum` (r57), `tf1d_dot` (c8398d3), `tf1d_l1_norm` (c8398d3), `tf1d_max_abs` (c8398d3) |
| A2 (`vec_map_abs`) | `vec_map_neg` (r51 A5), `vec_negate_inplace` (r51 A5), `ti1d_max_abs` (r56 A2), `vec_max_abs` (r56 A3) |
| A3 (`tf1d_dot_with_offset`) | `tf1d_dot` (c8398d3) |
| A4 (`tf2d_matvec`/`matmul`) | `tf1d_dot` (c8398d3); `ti2d_matvec`/`ti2d_matmul` are integer with i64 saturation (r52 A1) |
| A5 (`tf2d_row_sum`/`col_sum`/`trace`) | `tf1d_sum` (r57); `tf2d_norm_frobenius_sq`, `tf2d_max_abs` delegate to fixed primitives |
| A6 (`mse_loss_f32`/`mae_loss_f32`) | `tf1d_sum`/`tf1d_dot`/`tf1d_l1_norm` (r57 + c8398d3) |
| A7 (`tf1d_max/min/argmax*`/`argmax_rows_f32`) | `softmax_layer` (r48 A2), `attention_softmax_f32` (r53 A8) |

Magic-bearing validators (13) and arena-span call sites (14) re-counted,
matches restart 47 baseline. Stale-state surfaces (`wm_clear`,
`hashmap_clear`, `bindings_rewind`) re-verified clean.

## Lane verdict

**1 HIGH + 5 MEDIUM + 1 LOW = 7 findings**, all closed in the restart
58 catch-up sweep (Increment 77).
