# Stage 35 Restart 50 — Lane A (Runtime / stdlib safety) Audit Report

**Auditor:** Lane A
**Scope:** `helixc/stdlib/*.hx` (16 modules); `helixc/runtime/*` (no such directory exists)
**HEAD:** 6c555a4 (restart-49 fix commit) + f0ab654 (handoff doc)
**Mode:** READ-ONLY (no source modifications performed)
**Prior baseline:** Restart 47 fix at 4ba725f. Diff window 4ba725f..HEAD touched only `helixc/stdlib/autodiff.hx` and `helixc/stdlib/nn.hx` on the stdlib side.

---

## Summary

**Total findings: 0** (CLEAN).
- HIGH:    0
- MEDIUM:  0
- LOW:     0

**Verdict: CLEAN.**

After a focused sibling-by-sibling sweep of the bug families enumerated in the brief, Lane A finds nothing new to fix at restart 50 in the runtime/stdlib safety layer. The restart-48 and restart-49 fix commits (the two commits since the restart-47 baseline this lane is gated against) successfully closed every bug family they targeted, and the changes did not introduce any new typed handles, new transcendental call sites without fail-closed guards, new rewind/clear surfaces, new magic constants, or new integer-arithmetic-before-`__arena_*` sites that bypass the existing overflow-guard pattern.

The report below documents the sibling sweeps performed so the next-restart auditor can see exactly what surface was examined and re-use the sweep tables when new code arrives.

---

## Bug family 1 — New typed-handle validators introduced after restart 47

**Result: NO new validators introduced; all 13 pre-existing validators remain protected by `arena_span_in_tensor_payload`.**

Sibling sweep — all `fn *_ok` / `fn *_valid` validators in stdlib:

| Validator | File:line | Magic | `arena_span_in_tensor_payload` call | Status |
|---|---|---|---|---|
| `tree_node_ok` | agi_match.hx:39 | 7107001 | line 45, span = 6 | safe |
| `bindings_storage_ok` | agi_match.hx:248 | 7008001 | line 254, span = 67 | safe |
| `wmt_ok` | agi_world.hx:67 | 6006001 | line 79, span = total+4 | safe |
| `wml_ok` | agi_world.hx:136 | 6007001 | line 142, span = 5 | safe |
| `wm_ok` | agi_memory.hx:41 | 4004001 | line 47, span = wm_slot_count()+2 | safe |
| `ep_ok` | agi_memory.hx:217 | 5005001 | line 223, span = ep_slot_count()+2 | safe |
| `hashmap_ok` | hashmap.hx:54 | 7007001 | line 64, span = data_len+3 | safe |
| `bfs_ok` | agi_search.hx:45 | 6106101 | line 52, span = bfs_slot_count()+2 | safe |
| `visited_ok` | agi_search.hx:131 | 6206201 | line 138, span = visited_slot_count()+2 | safe |
| `pq_ok` | agi_search.hx:264 | 6306301 | line 271, span = pq_slot_count()+2 | safe |
| `rev_tape_valid` | autodiff_reverse.hx:94 | 3003001 | lines 113, 121, span = 5+cap*4 | safe |
| `rev_adj_cap` | autodiff_reverse.hx:287 | (inherits owner) | line 318, span = 6+cap+snapshot_total | safe |
| `t1d_slice_ok` (tensor.hx:90) and `t2d_shape_ok` (tensor.hx:244) are the *defining* validators for tensor payloads — they don't need an `arena_span_in_tensor_payload` self-check.

Diff `git diff 4ba725f..HEAD -- helixc/stdlib/` shows changes confined to `autodiff.hx` and `nn.hx`. Neither file added a new typed-handle struct or validator. No new magic constants. No new payload layouts.

---

## Bug family 2 — New `__sqrt` / `__log` / division sites that bypass the fail-closed pattern

**Result: NO new unguarded division sites. Restart 48 A1/A2/A3 closed the three sites the diff did add.**

Sibling sweep — every `__sqrt`, `__log`, and divisive-arithmetic site in stdlib (after restart 49):

| Site | File:line | Divisor | Fail-closed | Notes |
|---|---|---|---|---|
| `clip_grad_norm_f32` | nn.hx:172, 175 | `norm` after `norm_sq > 0` guard | yes | NaN `norm_sq` passes the `<= 0` check; documented garbage-in/garbage-out. |
| `adam_f32_step` | nn.hx:231, 237 | `raw_denom = __sqrt(next_v) + eps` | yes (restart 47 A1) | clamps `next_v >= 0`, then checks `raw_denom <= 0`. |
| `__adam_step` (scalar) | transcendentals.hx:503 | `raw_denom` | yes (restart 47 A2) | clamps `v >= 0`, then checks `raw_denom <= 0`. |
| `layer_norm_f32` | nn.hx:572, 586 | `denom = __sqrt(var + safe_eps)` | yes (restart 47 A3) | clamps `safe_eps >= 0`, writes 0 to outputs if `denom <= 0`. |
| `softmax_layer` | nn.hx:681 | `sum_e` | yes (restart 48 A2) | writes 1/n if `sum_e <= 0` or NaN. |
| `dense_classifier_sgd_step_f32` | nn.hx:822 | `sum_e` | yes (restart 48 A2 sibling) | no-op step if `sum_e <= 0` or NaN. |
| `attention_softmax_f32` | agi_search.hx:542 | `__sqrt((d as f32))` for `1.0 / __sqrt(d)` | safe by construction | `d <= 0` returns at line 531 so `d >= 1`, `__sqrt(>=1) > 0`. |
| `attention_dot` | agi_search.hx:636 | `total_w` | yes | wrapped in `if total_w > 0`. |
| `d_div_v` / `d_div_dx` | autodiff.hx:38–43 | `b_v`, `b_v*b_v` | yes (restart 48 A1) | returns 0 if `b_v == 0`. |
| `d_sqrt_dx` | autodiff.hx:66–68 | `2 * __sqrt_f64(a_v)` | yes (restart 47 A4) | returns 0 if `a_v <= 0`. |
| `d_log_dx` | autodiff.hx:86–88 | `a_v` | yes (restart 47 A5) | returns 0 if `a_v <= 0`. |
| `d_recip_v` / `d_recip_dx` | autodiff.hx:92–98 | `a_v`, `a_v*a_v` | yes (restart 47 A6/A7) | returns 0 if `a_v == 0`. |
| `mse_loss_f32_grad` | nn.hx:274 | `n` (i32 cast to f32) | yes | `n <= 0` returns 0 early. |
| `softmax_ce_grad_f32` | nn.hx:732 | `rows` (i32 cast to f32) | yes | `rows <= 0` returns 0 early. |
| `ce_loss_batch_f32` | nn.hx:957 | `rows` (i32 cast to f32) | yes | `rows <= 0` returns 0.0 early. |
| `ti1d_mean` | tensor.hx:597 | `n` | yes | `n <= 0` returns 0. |
| `tf1d_mean` | tensor.hx:774 | `n` (cast to f32) | yes | `n <= 0` returns 0.0. |
| `dropout_f32` | nn.hx:631 | `keep_prob` | yes | guarded `<= 0` and `>= 1` branches. |
| `__sigmoid` | transcendentals.hx:270 | `1 + __exp(-x)` | safe by construction | `__exp` Taylor returns ≥ 0, so denominator ≥ 1. |
| `__sigmoid_f64` | transcendentals.hx:206 | `1 + __exp_f64(-x)` | safe by construction | same as above. |
| `hashmap_load_factor_x100` | hashmap.hx:412 | `cap` | yes | `cap == 0` returns 0; `cap < 0` reaches `0 * 100 / negative = 0` via `hashmap_size` returning 0. |
| `hashmap_avg_value_x100` | hashmap.hx:634 | `n as i64` | yes | `n == 0` returns 0 early. |

All 22 divisive sites either fail-closed at the singularity or are guarded by a structural invariant that prevents the divisor from being zero. The three new sites introduced by restart 48 (`d_div_*`, `softmax_layer` `sum_e`, `dense_classifier_sgd_step_f32` `sum_e`) all received the matching fail-closed pattern in the same commit.

No new `__log` call sites since restart 47 (only autodiff.hx:85 `d_log_v` which uses `__log_f64`, guarded). No new `__sqrt` call sites.

---

## Bug family 3 — NaN-eps handling

**Result: NO new load-bearing reason to make NaN inputs fail-closed beyond what's already in place. Garbage-in/garbage-out remains the explicit documented behavior.**

Sibling sweep — every site that combines an `eps` parameter with NaN-propagating math:

| Site | NaN safety | Documented behavior |
|---|---|---|
| `adam_f32_step` (nn.hx:206) | clamps `next_v < 0` but not NaN `next_v`; `raw_denom <= 0` is false for NaN | NaN propagates to `w_i`. GIGO. |
| `__adam_step` (transcendentals.hx:501) | same as adam_f32_step | GIGO. |
| `layer_norm_f32` (nn.hx:556) | clamps `eps < 0` but not NaN `eps`; `denom <= 0` is false for NaN | NaN propagates to outputs. GIGO. |
| `clip_grad_norm_f32` (nn.hx:165) | `norm_sq <= 0` is false for NaN; `norm > target` is false for NaN | clipping silently skipped, original NaN gradient preserved. GIGO. |
| `softmax_layer` (nn.hx:645) | `sum_e != sum_e` (NaN check) added in restart 48 A2 | NEW: writes max-entropy distribution if NaN. |
| `dense_classifier_sgd_step_f32` (nn.hx:758) | `sum_e != sum_e` added in restart 48 A2 sibling | NEW: no-op step if NaN. |
| `__bce` (transcendentals.hx:439) | `__clamp(p, 0.000001, 0.999999)` — NaN p stays NaN through `__min`/`__max` since NaN comparisons return false | NaN propagates to output. GIGO. |

The `softmax_layer` and `dense_classifier_sgd_step_f32` upgrades from restart 48 explicitly added NaN handling because those sites are at the *boundary* between input poison and downstream consumers — a NaN logit otherwise becomes a NaN probability that poisons every weight in the subsequent backward pass. No analogous load-bearing boundary site has been newly introduced since restart 47 that lacks the NaN guard.

If the brief later decides to upgrade `clip_grad_norm_f32` or `adam_f32_step` from GIGO to fail-closed at NaN, that would be a *deliberate policy choice* rather than a defect — flagging it here purely for visibility, not as a finding.

---

## Bug family 4 — New rewind / restore / reset / clear surfaces

**Result: NO new rewind/clear surfaces. The three pre-existing ones remain clean.**

Sibling sweep — every `fn *(rewind|restore|reset|clear|undo|reload|reinit)*`:

| Function | File:line | Validator-gated | Verified clean (restart) |
|---|---|---|---|
| `wm_clear` | agi_memory.hx:89 | `wm_ok(start)` | restart 47 |
| `hashmap_clear` | hashmap.hx:183 | `hashmap_ok(start, cap)` | restart 47 |
| `bindings_rewind` | agi_match.hx:293 | `bindings_storage_ok(b)` + count bounds | restart 47 |

`rev_push` (autodiff_reverse.hx:170) actively *poisons* the tape footer when an adjoint buffer has been allocated — the deliberate one-shot seal that prevents reflective extension after a backward pass. That's intended state-clearing, not a regression surface; it falls within the rev_tape invariant.

No new functions matching the rewind/clear/reset/undo/restore/reload/reinit pattern were added since restart 47.

---

## Bug family 5 — Magic-constant collisions

**Result: 13 magic constants, all unique. NO collisions.**

Whole-class sweep (regex `_magic\(\)\s*->\s*i32\s*\{\s*[0-9]+\s*\}` across stdlib):

| Value     | Constant         | File |
|-----------|------------------|------|
| 1001001   | t1d_magic        | tensor.hx |
| 2002001   | t2d_magic        | tensor.hx |
| 3003001   | rev_tape_magic   | autodiff_reverse.hx |
| 4004001   | wm_magic         | agi_memory.hx |
| 5005001   | ep_magic         | agi_memory.hx |
| 6006001   | wmt_magic        | agi_world.hx |
| 6007001   | wml_magic        | agi_world.hx |
| 6106101   | bfs_magic        | agi_search.hx |
| 6206201   | visited_magic    | agi_search.hx |
| 6306301   | pq_magic         | agi_search.hx |
| 7007001   | hashmap_magic    | hashmap.hx |
| 7008001   | bindings_magic   | agi_match.hx |
| 7107001   | tree_node_magic  | agi_match.hx |

All 13 distinct (Python set verification: `len(set(values)) == 13`). The restart-46 distinction `tree_node_magic = 7107001` vs `hashmap_magic = 7007001` remains intact. No new typed structures appeared since restart 47, so no new collision risk.

---

## Bug family 6 — Overflow guards on integer arithmetic preceding `__arena_*` bounds checks

**Result: 14 `arena_span_in_tensor_payload` call sites, all preceded by `2147483647 - x` overflow checks. NO new sites that bypass the pattern.**

Sibling sweep — every `arena_span_in_tensor_payload` invocation:

| Site | Span-len expression | Overflow guard | Status |
|---|---|---|---|
| wm_ok | wm_slot_count()+2 | start + wm_slot_count() bounded above | safe |
| ep_ok | ep_slot_count()+2 | start + ep_slot_count() bounded above | safe |
| tree_node_ok | 6 (constant) | off + 4 ≤ 2147483647 check | safe |
| bindings_storage_ok | 67 (constant) | b + 65 ≤ 2147483647 check | safe |
| wmt_ok | total+4 | total ≤ 2147483647 - wmt - 2 check; total ≤ 2147483647 - 4 | safe |
| wml_ok | 5 (constant) | wml + 3 ≤ 2147483647 check | safe |
| bfs_ok | bfs_slot_count()+2 | bounded by bfs_slot_count() constant + arena_len check | safe |
| visited_ok | visited_slot_count()+2 | bounded by visited_slot_count() | safe |
| pq_ok | pq_slot_count()+2 | bounded by pq_slot_count() | safe |
| rev_tape_valid (no-adj branch) | 5+cap*4 | cap ≤ (2147483647 - tape - 3) / 4 then cap ≤ (2147483647 - 5) / 4; tape ≤ 2147483647 - (5+cap*4) | safe |
| rev_tape_valid (with-adj branch) | 5+cap*4 | same | safe |
| rev_adj_cap | 6+cap+snapshot_total | cnt ≤ (2147483647 - snapshot_start - 1) / 4; adj_start - 4 ≤ 2147483647 - adj_span_len | safe |
| hashmap_ok | data_len+3 | data_len ≤ 2147483647 - start; data_len ≤ 2147483647 - 3 | safe |
| t2d_offset / t2d_shape_ok | n (no span check; structural) | n ≤ 2147483647 - start | safe |

No new sites since restart 47 (no diff to any file containing these). The 14-site coverage from restart 47 remains current.

---

## Areas verified clean (sweeps that found nothing)

1. **Diff scope minimality** — only autodiff.hx (+10 lines) and nn.hx (+62 lines) changed on the stdlib side since restart 47. Every change is accompanied by a `// Restart 48 ...` comment that explains the fail-closed precedent it followed. Restart 49 added no stdlib changes; its work was confined to Python-side files (`backend/x86_64.py`, `backend/ptx.py`, `check.py`, `ir/lower_ast.py`, `frontend/autodiff_cli.py`) tightening CLI banners and narrowing exception scopes.

2. **`tf1d_max` NaN behavior** — returns the first slot's value if every later comparison fails. With a NaN first slot, returns NaN. This is documented GIGO and the downstream `softmax_layer` now handles a NaN `sum_e` correctly.

3. **`__exp` saturation** — caps `2^k` scale at ±48 (transcendentals.hx:53), so for any finite input `__exp(x)` returns a finite value in roughly [2^-48, 2^48]. Cannot produce +Inf from finite input. Combined with the max-subtract trick in softmax, `sum_e` for n≤2^48 finite inputs cannot exceed n*1.0; cannot produce +Inf.

4. **`bindings_rewind`** — checks `count > cnt`, `count < 0`, `count > 32`. Cannot over-rewind or write outside the 32-slot table.

5. **`rev_push` poison-on-adjoint-allocated** — explicitly writes 0 to the tape footer when called after `rev_alloc_adjoints`. This intentionally invalidates the tape so subsequent `rev_tape_valid` checks fail. The pattern matches the one-shot-seal invariant documented in autodiff_reverse.hx's module header.

6. **`__sigmoid`, `__sigmoid_f64`** — denominator is `1 + __exp(-x)` which is provably ≥ 1, so division is always safe.

7. **`__bce`** — clamps p to (0.000001, 0.999999) before `__log_stable`, so log input is always in the well-defined range.

8. **`hashmap_load_factor_x100`, `hashmap_avg_value_x100`** — `cap == 0` and `n == 0` guards prevent division by zero. Negative `cap` is safe because `hashmap_ok` returns 0 → `hashmap_size` returns 0 → guarded by the n==0/cap==0 branches.

9. **`attention_softmax_f32`** — `d <= 0` returns t2d_error at line 531, so `__sqrt((d as f32))` is on `d >= 1`, never zero.

10. **`attention_dot`** — `total_w > 0` gate prevents division by zero.

11. **`iterators.hx`, `vec.hx`, `string.hx`** — no magic-bearing handles, no validators required. Out of scope for the typed-handle bug family.

12. **Restart-48 / restart-49 regression tests** — `test_codegen.py` (+75 lines) added `test_stage35_d_div_fail_closed_at_zero_denominator`, `test_stage35_softmax_layer_fail_closed_on_degenerate_sum_e`, `test_stage35_tanh_layer_does_not_nan_at_saturation_boundary` plus prior tests pinning the restart 47 work. The whole-class regression coverage for magic uniqueness, arena_span checks, fail-closed division remains intact.

---

## Notes for next-restart auditor

- Diff window for restart 51 should be `git diff 6c555a4..HEAD -- helixc/stdlib/`. If empty, the stdlib has not changed and most families below can be reaffirmed by reference rather than re-swept.
- If a new `fn *_magic()` appears, re-run the magic-collision sweep (Python set check).
- If a new `*_ok` / `*_valid` validator appears, verify it ends with `arena_span_in_tensor_payload(handle - K, span_len) != 0`.
- If a new `__sqrt` / `__log` / `/` site appears in nn.hx or autodiff.hx, verify it follows the `if denom <= 0 || denom != denom` precedent established by restart 47/48.
- The Python-side narrowing pattern (restart 48 B2, restart 49 B4) is `except (NotImplementedError, AssertionError, KeyboardInterrupt, SystemExit, MemoryError): raise` before `except Exception`. New `except Exception` blocks in critical paths should match this.
