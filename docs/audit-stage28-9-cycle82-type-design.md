# Audit Stage 28.9 cycle 82 — Type design

**Scope.** HEAD `7b13010` (Stage 28.9 cycle-81 fix-sweep: 2 cycle-80 findings
on test discrimination). Strict read-only mode. ONE Write to this file; no
Edits. Prior C1–C81 findings + deferred-known set NOT re-flagged.

**Criterion.** 0 findings at conf >= 75%.

## Result: PASS — 0 findings at conf >= 75%

## Verification of cycle-81 fix surface — strict-inequality robustness

The cycle-81 fix at `helixc/tests/test_ffi.py:111–179` replaces the original
"byte pattern in elf" assertions with a two-program comparison:

```python
float_load  = elf_float.count(b"\xf3\x0f\x10")     # movss xmm0, [rbp-N]
float_store = elf_float.count(b"\xf3\x0f\x11")     # movss [rbp-N], xmm0
int_load    = elf_int.count(b"\xf3\x0f\x10")
int_store   = elf_int.count(b"\xf3\x0f\x11")
assert float_load  > int_load
assert float_store > int_store
```

The audit question: **can `float_load == int_load` (or `float_store ==
int_store`) ever hold post-fix, e.g. via DCE removing the FFI movss?**

### DCE cannot strip the movss bytes (conf 95)

`helixc/ir/passes/dce.py:32–81` defines `SIDE_EFFECT_KINDS`. Line 64 lists
`tir.OpKind.FFI_CALL` explicitly (Stage 16.5 follow-up audit CRITICAL-1 fix).
Liveness seeding (line 100–105) marks every operand of a side-effecting op
live, and side-effecting ops themselves are unconditionally retained
(line 131–132 `if op.kind in SIDE_EFFECT_KINDS: new_ops.append(op); continue`).

Therefore in the float-FFI program:
- `FFI_CALL(cosf, [CONST_FLOAT 1.5_f32]) -> f32 result` is retained.
- The arg operand (the CONST_FLOAT) is seeded live, so its STORE-to-slot is
  retained.
- The `op.results[0]` slot is referenced unconditionally by
  `x86_64.py:1779–1794` whenever `op.results` is non-empty; for an `f32`
  return that branch is `movss_mem_rbp_xmm0` (`F3 0F 11`).
- The arg-routing branch at `x86_64.py:1758–1766` emits
  `_movss_load_xmmN(0, arg_slot)` (`F3 0F 10`) unconditionally for float
  args.

No pass in `helixc/ir/passes/` removes ops inside an FFI_CALL operand chain
once seeded. `fdce.py:27–60` removes whole functions only; `caller` is
called from `main` so it is reachable. There is no inliner pass. The
backend's emit path does not gate the `movss` emission on liveness — it is
strictly type-driven.

So `float_load >= 1` and `float_store >= 1`. In the int-FFI control
program (`puts(s: *const u8) -> i32`, no float types anywhere), no
`_movss_*` call path is reached: arg routing takes the `INT_REGS` branch
(`x86_64.py:1772–1775`) and the i32 return takes `mov_mem_rbp_eax`
(`x86_64.py:1794`). Net: `int_load = int_store = 0` from instruction
emission.

### Spurious-substring residual risk (conf < 50, not flagged)

A pedantic concern is that `F3 0F 10` or `F3 0F 11` could appear as a
substring across instruction boundaries (e.g. an imm32 tail of `0xF3`
followed by a `0x0F 0x10`-prefixed next instruction). For this to break
strict inequality the int-FFI elf would need at least one such spurious
match in a region that the float-FFI elf does not. Both elves share
caller/main/dispatcher boilerplate and the `puts` route uses only
mov/lea/call opcodes — none of which begin with `0F 10` or `0F 11`.
Constant immediates in this program are small (the int `0` return,
slot offsets ≤ 256). The probability of a same-position spurious match
in `int_elf` that exceeds `float_elf`'s deterministic ≥ 1 movss-load and
≥ 1 movss-store is negligible across the regression-test horizon. Sub-75
confidence — not flagged.

## Rotation: tir.py — Module/Function/Block invariants over Op result lists

### Op.results / Block.ops freshness (conf 90)

`tir.py:305–316`: `Op` is an unfrozen dataclass with
`results: list[Value] = field(default_factory=list)` and same for
`operands` and `attrs`. The `field(default_factory=...)` form constructs a
fresh list/dict per `Op` instance — no shared-reference aliasing risk.
`IRBuilder.emit` (`tir.py:418–430`) passes `list(operands)` and `attrs or
{}`, so even caller-side aliasing is broken on construction. Independent
ops cannot share a mutable `results` or `operands` list.

### Value identity vs type-carry consistency (conf 85)

`tir.py:107–119`: `Value` is an unfrozen dataclass with custom `__hash__`
and `__eq__` keyed on `id` alone (not `ty`, not `name_hint`). This is the
SSA-identity contract — two `Value`s with the same id but different types
would compare equal. The contract relies on `IRBuilder.new_value`
(`tir.py:374–377`) being the single id source, which it is
(`module.next_value_id += 1`). No alternate constructor path is exposed.

### Block.params vs FnIR.params disambiguation (conf 85)

`FnIR.params: list[Value]` (`tir.py:345`) is the function-argument SSA
values, while `Block.params: list[Value]` (`tir.py:337`) is the
block-parameter list used in place of phi-nodes (Cranelift CLIF / SIL
pattern). DCE liveness seeds BOTH unconditionally (`dce.py:106–112`), so
a function param entering a sub-block as a block-param survives both
seeds. No collision: both go into the same `live` set keyed by `Value.id`.

### Module.next_value_id / next_block_id monotonicity (conf 80)

`tir.py:355–360`: `next_value_id` and `next_block_id` are int counters on
the `Module`. `IRBuilder.new_value` (`tir.py:374–377`) and `new_block`
(`tir.py:379–382`) increment after read. No reset path exists in the
module class. Two `IRBuilder` instances pointed at the same module would
share the counters (correct) — and the constructor (`tir.py:367–371`)
takes the module by reference, so this is the intended contract.

No invariant violations found.

## Rotation: typecheck.py — type-equality semantics

The cycle-82 brief specified `helixc/frontend/types.py`; that file does not
exist. Type definitions live in `helixc/frontend/typecheck.py:31–191`
(twenty `Ty*` classes plus the `Type` base). Treated as the type-equality
target.

### Frozen-dataclass structural `__eq__` (conf 90)

All twenty `Ty*` subclasses are `@dataclass(frozen=True)` — auto-generated
`__eq__` performs field-wise structural comparison. There is no manual
`__eq__` override or `types_equal` / `_type_eq` helper anywhere in
`typecheck.py` or the wider `helixc/frontend/` tree (grep
`types_equal|_eq_type|_type_eq` returns zero matches), so equality is
governed entirely by the dataclass machinery.

### Field-list audit for hidden equality-distorting state (conf 85)

- `TyPrim(name)` — pure nominal. `TyPrim("i32") == TyPrim("i32")`. OK.
- `TyTensor(dtype, shape, device, layout)` — `device` and `layout` are
  `Optional[str]` and PART of equality. `TyTensor(.., device=None) !=
  TyTensor(.., device="cpu")`. This is the intended contract (device
  affects ABI), and `lower_ast` paths set both fields consistently.
- `TyRef(inner, is_mut)` — `is_mut` is a field, so `&T != &mut T`
  (correct).
- `TyPtr(inner, is_mut)` — same shape as `TyRef`, equality correctly
  distinguishes `*const T` from `*mut T`.
- `TyLogic(inner, provenance)` — `provenance: Optional[str]` is a field
  in equality. Phase-0 sets it `None` at every construction site
  (per docstring lines 156–161); two `TyLogic[i32]` values constructed
  in Phase-0 always equal. A future pass that sets a non-`None`
  provenance would create equality asymmetry with the `None` baseline,
  but no such pass exists today. Not a current bug; flagged here only
  as design-time provenance.
- `TyDiff(inner)` — single field. `D<f32> != f32` (correct — TyDiff
  carries gradient tracking). Auto-eq.
- `TyMemTier(tier, inner)` — `tier: str` participates. Cross-tier
  values rightly inequal.
- `TySkill(inner, task)` — `task: str = ""` is part of equality, so
  `Skill<F>` with task `"a"` != with task `"b"`. Intended (skill registry
  is task-indexed).
- `TyTensor.shape: tuple[Type, ...]` — tuple of `Type`, element-wise
  hashed by `__eq__`. Tuple ordering is significant so shape `(M, N)`
  != `(N, M)`. OK.
- `TyTuple.elems: tuple[Type, ...]` and `TyArray(elem, size)` — same
  tuple/structural treatment. OK.
- `TyFn(params, ret)` — `params: tuple[Type, ...]` (positional). Two
  functions with permuted param tuples are inequal. Correct ABI-shape
  semantics.

No type permits two structurally-distinct instances to compare equal,
and none over-distinguishes within Phase-0's construction discipline.

### Hashability invariant (conf 80)

`@dataclass(frozen=True)` makes each class hashable via the synthesized
`__hash__`. `TyTensor.shape: tuple[Type, ...]` is hashable iff every
element is hashable; the shape elements are themselves `Type`-subclass
instances which are frozen dataclasses — recursively hashable. Same for
`TyTuple.elems`. No hash/eq contract violation.

## Summary

**0 findings at confidence >= 75%.** Cycle-81's strict-inequality test
discrimination is robust against the DCE corner case raised in scope:
FFI_CALL is pinned in `dce.py:SIDE_EFFECT_KINDS`, the backend emits
`movss` load/store bytes unconditionally for float-typed FFI operands and
returns, and no inliner / fdce / cse / const-fold pass can degrade those
counts. tir.py's Module/Function/Block invariants over Op result lists are
sound — fresh per-instance lists via `field(default_factory=list)`, SSA
identity via `Value.id`, monotonic counters on Module. typecheck.py's
twenty `Ty*` classes uniformly use frozen-dataclass structural equality
with no hidden equality-distorting fields.

**No edits performed.** This file is the sole Write of the audit.
