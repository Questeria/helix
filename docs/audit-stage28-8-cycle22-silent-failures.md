# Stage 28.8 pre-29 audit gate — Cycle 22 (Audit A: silent failures)

**Date:** 2026-05-11
**HEAD:** `bee36e6` ("Audit 28.8 cycle 21 fix-sweep: close C20-1 (HIGH,
PTX backend isize/usize silent 32-bit)")
**Lens:** silent failures (Audit A)
**Streak counter at start:** 1/5 (cycle 21 was first clean of streak)

> **Note.** A prior write of `docs/audit-stage28-8-cycle22-silent-
> failures.md` existed under a different cycle-22 task brief (Stage
> 28.8.1 / 28.8.2 deliverables audit). The current cycle-22 task
> brief explicitly rotates to the **least-covered surface** (effect_
> check.py interior, frontend/parser.py error paths, backend/elf_dyn.
> py) and requires a fresh-eyes re-audit. This doc replaces the
> earlier draft per the current brief; the earlier doc's content
> (walker library, op-suffix, fresh-counter, walker refactors,
> isize/usize fix verification) is preserved in the git history at
> the prior write (not in this commit because read-only) — those
> targets all returned CLEAN there and are not re-checked here.

---

## Scope

Strict-criterion read-only audit. The 6-cycle "silent width
narrowing" defect class (C13-1, C16-1, C18-1, C18-B/C, C19-1, C20-1)
is closed per cycle-21 Audit A's exhaustive sweep. The cycle-22 task
brief directs rotation to a least-covered surface to verify
stability across a different defect-class window.

Targets (explicit in the cycle-22 brief):

1. **`helixc/ir/passes/effect_check.py`** — IR-level
   effect/capability verifier interior. Does the
   `OP_EFFECTS` / `callees()` classification cover every
   side-effecting OpKind reachable from current frontend?
2. **`helixc/frontend/parser.py`** — error paths and lenient-
   recovery branches. Any silent token-skip that could mask a
   real syntax error?
3. **`helixc/backend/elf_dyn.py`** — ELF emitter for the dyn-link
   path (Stage 16.5 FFI binaries). Layout assertions, error
   modes, fallback defaults.

The bar is **zero findings of any severity**. Pre-existing forward
notes from cycles 1-21 are not re-cited (per the strict re-flag
rule). Cycle-21 forward note **F-21-3** (bootstrap kovc test
`test_bootstrap_kovc_full_pipeline_arithmetic` fails at HEAD
`bee36e6` and at parent `5a1e406` with identical drift) is **noted
but explicitly NOT re-flagged** — pre-existing, unchanged,
orthogonal to the cycle-22 rotation surfaces.

---

## Target 3 — `helixc/backend/elf_dyn.py` (481 lines)

### Error paths

| Site | Behavior | Classification |
|------|----------|----------------|
| `plan_layout` line 216-219 | `raise RuntimeError("phdrs+interp ... exceed code offset ...")` if PHDR table + interp string would overflow into the 0x1000 code-offset gap | Loud (raises). Clean. |
| `plan_layout` line 326 | `assert len(rela_plt_bytes) == rela_plt_size` | Loud (AssertionError on mismatch). Clean. |
| `plan_layout` line 344 | `assert len(dyn_entries) == n_dyn_entries` | Loud. Clean. |
| `emit_elf_dyn` lines 422, 453, 459, 463, 468, 470, 472, 474, 476, 478, 480 | 11 `assert len(...) == ...` checks pinning every region boundary as the file is assembled | Loud. Defense-in-depth: any layout-planner bug surfaces immediately as AssertionError with the byte counts attached. |

### `.get(...)` / silent-default scan

`elf_dyn.py` uses **zero `dict.get(..., default)` calls** in either
`plan_layout` or `emit_elf_dyn`. All key lookups go through
explicit subscripting on `list` indices (`lib_offsets[...]`,
`sym_name_offsets[...]`) and through enum-keyed constants (`DT_*`,
`PT_*`, `PF_*`, `STB_*`, `STT_*`, `R_X86_64_JUMP_SLOT`,
`SIZE_EHDR`, `SIZE_PHDR`, `SIZE_SYM`, `SIZE_RELA`, `SIZE_DYN`). No
fallback-default branches exist that could silently miscompile
layout.

### Width-keyed defect-class re-check

`elf_dyn.py` has no width-keyed tables (no isize/usize-style
narrowing surface). All integer fields use explicit `struct.pack`
format codes (`"<Q"` for u64, `"<I"` for u32, `"<H"` for u16, `"<q"`
for i64). Format characters are hard-coded against the ELF64 spec
— no ambiguous width selection. Out of the cycle-13-through-21
defect class.

### `DynLinkInfo.add_import` idempotency

Line 132-139: idempotent insert into `_imports_set` keyed by symbol
name, returning the GOT slot index. Race-free (single-threaded
codegen). No silent collision: same name → same slot; different
names → distinct slots. Clean.

**Verdict for `elf_dyn.py`: clean.** Every layout invariant is
asserted in-line; no silent fall-throughs; defect-class out-of-
scope. Fully loud.

---

## Target 2 — `helixc/frontend/parser.py` (1621 lines)

### Exception-handling sites

Only one `try/except` on a non-control-flow exception:

- **Line 367-376** (`_parse_autotune_int`): wraps `int(s, base)`
  parsing in `try / except ValueError: raise ParseError`. The
  catch is narrow (single exception type) and re-raises a
  user-facing `ParseError` with the original token's
  line/column attached. Loud. Clean.

The other two `try` blocks (`_parse_tensor_type` line 707,
`_parse_tile_type` line 736, `_peek_struct_lit_start` line 1306) are
`try / finally` for context-state push/pop — not exception handlers.
No silent suppression.

### Silent-skip / silent-recover sites

Three `continue` statements in the attribute parser
(`_parse_attributes` lines 226, 283, 292) skip to the next
attribute after processing the current one. Each is a normal
loop-iteration tail, not an error-recovery branch.

#### 2a. `_parse_string_attr_arg` (lines 378-392) — lenient token skip

```python
self._eat(T.LPAREN)
msg: "str | None" = None
if self._at(T.STRING):
    t = self._peek()
    msg = t.string_value or ""
    self.i += 1
# Skip any other tokens to RPAREN (lenient)
while not self._at(T.RPAREN) and self._peek().kind != T.EOF:
    self.i += 1
self._eat(T.RPAREN)
```

Used by `@deprecated("msg")` / `@since("v0.3")` attribute parsing.
A user that writes `@deprecated("hi", junk_tokens, 42)` will have
the trailing `junk_tokens, 42` silently swallowed with no warning;
the `msg` will be just `"hi"`. The lenient comment at line 388 is
intentional and called out in the source.

**Classification.** This is an attribute-payload parser, not a
lower-level grammar production. The lenient mode is documented in
the source as "(caller's job to handle)". The downstream effect of
swallowing extras is a recorded attribute set that lacks the
trailing tokens — but since `@deprecated` and `@since` consume
exactly one optional message string, the extras are
syntactically-spurious payload that the typechecker would have no
use for anyway. **Not a silent miscompile.** A user
typo-introducing the extras may be confusing if they expected
multiple args, but the attribute spec for `@deprecated`/`@since`
is single-string-arg-only by design. **Recorded forward note
F-22-1, NOT a finding** — promotion to a finding would require
the attribute syntax to admit multi-arg forms in a way that
silently dropped data.

#### 2b. `_builtin_kind_to_name` (lines 663-674) — `mapping.get(kind)`

`mapping.get(kind)` returns `None` for non-builtin kinds. The
caller at line 678 checks `if builtin is not None:` and falls
through to the user-IDENT branch when the lookup misses. Loud
relative to its contract (the contract IS "return None on
non-builtin"). Clean.

#### 2c. `_merge_stdlib` (lines 1582-1607) — missing-file lenient mode

When a stdlib file is missing AND `HELIXC_STDLIB_STRICT` is unset,
the merger prints to stderr (`helixc: stdlib file missing: ...`)
and continues. This is documented at lines 1552-1555 and is the
default lenient-mode for backward compatibility — but the
diagnostic is emitted to stderr, so it is **not silent**. The
companion `HELIXC_STDLIB_STRICT=1` mode upgrades to
`FileNotFoundError`. Clean.

### `extern "C"` parsing (lines 394-436)

Loud: rejects any ABI other than `"C"` with a ParseError; requires
`;` terminator; copies `attrs` into the FnDecl untouched. No
silent path.

### Token-stream-end behavior

Line 70-72 `_peek` returns the last token (typically EOF) on
out-of-range indexes. Every `_eat` either matches the expected
kind or raises ParseError. Token stream cannot silently terminate
mid-parse.

**Verdict for `parser.py` error paths: clean.** One lenient
attribute-payload site (`_parse_string_attr_arg`) recorded as
forward note F-22-1, classified as documented-and-bounded
behavior, not a silent failure.

---

## Target 1 — `helixc/ir/passes/effect_check.py` (228 lines)

### Background: the cycle-22 rotation surface

The IR-level effect verifier is the **authoritative effect
checker** per its own docstring (lines 2-13): "The frontend
typecheck checks effects at AST level for direct named callees,
but it can be bypassed by indirect calls, calls to unresolved
names, special ops like MODIFY/SPLICE/PRINT that aren't surface-
level 'calls'. This IR pass runs AFTER lowering and is the
authoritative effect checker." The frontend typecheck (`typecheck.
py:1004-1035 _check_call_effects`) explicitly defers to this
module: "the IR-level effect_check pass — that's the soundness
layer; this surface check only flags directly-declared effects"
(typecheck.py:1011-1013).

The cycle-22 rotation surface is therefore: does the IR-level
verifier cover **every side-effecting OpKind reachable from the
current frontend**, or are there OpKinds whose effects are
silently ignored?

### Step 1 — enumerate the OP_EFFECTS table

```python
OP_EFFECTS: dict[tir.OpKind, frozenset[str]] = {
    tir.OpKind.PRINT:  frozenset({"io"}),
    tir.OpKind.MODIFY: frozenset({"modify_self"}),
    tir.OpKind.SPLICE: frozenset({"modify_self"}),
    tir.OpKind.TRAP:   frozenset({"io"}),
}
```

Four entries. Plus `callees()` at lines 123-139 iterates
`OpKind.CALL` and `OpKind.MODIFY` (the latter for verifier_fn
propagation).

### Step 2 — enumerate every OpKind, cross-check against the
DCE side-effects canon

The DCE pass (`helixc/ir/passes/dce.py:32-87`) maintains its own
`SIDE_EFFECT_KINDS` set, which the cycle-14 audit established as
the authoritative "what executes-for-side-effects" set per
"explicit, justified, defect-history-backed" criteria (cycle-14
type-design audit confirmed the per-entry justification). At
HEAD `bee36e6` the DCE canon includes 19 entries; OP_EFFECTS
covers only 4 of them. The gap:

| OpKind | In DCE side-effects | In OP_EFFECTS | In callees()? | Effect-label? |
|--------|---------------------|---------------|---------------|---------------|
| `RETURN` | yes | no | no | n/a (control-flow) |
| `BR` | yes | no | no | n/a (control-flow) |
| `COND_BR` | yes | no | no | n/a (control-flow) |
| `CALL` | yes | no | **yes** (via callees) | inherited transitively |
| `STORE_VAR` | yes | no | no | local mutation — n/a |
| `STORE_ELEM` | yes | no | no | local mutation — n/a |
| `ALLOC_VAR` | yes | no | no | local mutation — n/a |
| `ALLOC_ARRAY` | yes | no | no | local mutation — n/a |
| `MODIFY` | yes | **yes** (`modify_self`) | yes (verifier_fn) | covered |
| `SPLICE` | yes | **yes** (`modify_self`) | no | covered |
| `PRINT` | yes | **yes** (`io`) | no | covered |
| `QUOTE` | yes | no | no | reflection — gap (see C22-2) |
| `REFLECT_HASH` | yes | no | no | reflection — gap (see C22-2) |
| `ARENA_PUSH` | yes | no | no | global mutation — gap (see C22-3) |
| `ARENA_SET` | yes | no | no | global mutation — gap (see C22-3) |
| `TILE_INDEX_STORE` | yes | no | no | HBM write — gap (see C22-4) |
| **`FFI_CALL`** | **yes** | **no** | **no** | **arbitrary extern effects — gap (C22-1)** |
| `TRAP` | yes | **yes** (`io`) | no | covered |
| `TRACE_ENTRY` | yes | no | no | runtime event — gap (see C22-5) |
| `TRACE_EXIT` | yes | no | no | runtime event — gap (see C22-5) |

Seven OpKinds are listed in the DCE side-effects canon but absent
from `OP_EFFECTS`, AND not iterated by `callees()` to inherit
transitive effects. The cycle-14 audit confirmed each member's
presence in DCE was justified by a real defect history (most
notably Stage 16.5's FFI_CALL audit CRITICAL-1, where DCE had been
silently dropping void-return FFI calls). The corresponding
analysis of OP_EFFECTS has never been done.

### Step 3 — reachability triage of each gap

**FFI_CALL — REACHABLE under current frontend.** lower_ast.py
line 1720-1723 routes every extern "C" function call through
`OpKind.FFI_CALL` (NOT `OpKind.CALL`). A user-written
`@pure fn f() { puts(...); }` has:

- Frontend typecheck (`typecheck.py:_check_call_effects`): looks
  up `puts` in `self.functions`; gets `sig.effects = frozenset()`
  because extern "C" decls default to "no declared effects"
  (typecheck.py:459-468 reads `@effect(...)` attrs from the
  extern decl; no implicit effect is added). Empty `sig.effects`
  → no error raised at AST layer.
- IR effect_check (`effect_check.py`): `own_op_effects(f)`
  iterates ops, checks `op.kind in OP_EFFECTS` — `FFI_CALL` is
  NOT in OP_EFFECTS → contributes nothing. `callees(f)` iterates
  ops, only matches `CALL` and `MODIFY` — `FFI_CALL` is NOT
  iterated → contributes no callee name. Closure for `f` =
  frozenset(). `is_pure_decl(f)` is True, closure is empty → no
  error raised at IR layer.
- Compilation proceeds. The compiled binary writes to stdout via
  libc `puts`, fulfilling no effect annotation.

The contract that `@pure` means "closure must be empty" is
silently broken. Downstream optimizations that assume purity
(CSE, hoisting, reordering, GVN) may now safely reorder /
duplicate / hoist / eliminate FFI-calling fns — the verifier has
signed off that there are no side effects to preserve. The
`compute_closure` test at `test_effect_check.py:71-91`
(`test_unknown_callee_treated_as_unknown_effect`) is a synthetic
`OpKind.CALL` with `target="extern_unknown"` — it does NOT
exercise the actual extern path, which goes through `FFI_CALL`.

**ARENA_PUSH / ARENA_SET — REACHABLE.** Arena ops mutate the
global bump arena. Bootstrap kovc relies on these for parser
scratch storage. A `@pure fn` that internally calls
`__arena_push(7)` (a builtin recognized at `typecheck.py:1046`)
would silently pass effect_check. Same FFI_CALL pattern: missing
from OP_EFFECTS, missing from callees().

**TILE_INDEX_STORE — gated.** Only emitted inside `@kernel`-
attributed fns (lower_ast device-fn lowering). Kernel fns
carry the `kernel` attribute (already in META_ATTRS, so it's
not treated as a declared effect). A kernel fn can't be `@pure`
in practice today because typecheck rejects `@pure + @kernel`
implicitly (kernel needs hbm side effects). **Gated-
unreachable from @pure** — but the verifier doesn't enforce
the gating; it's enforced by the kernel-launch pipeline.
LOW severity at current frontend, but the gap is in the same
class as FFI_CALL.

**QUOTE / REFLECT_HASH — gated.** Stage-6 reflection ops. A
`@pure fn` calling `quote(...)` should logically be banned
because QUOTE reserves a runtime reflection cell handle. Today
the only emitters are inside `unsafe { }` blocks for the
reflection probe, so production reachability from `@pure` is
unclear. LOW severity; needs explicit reflection-purity
policy decision.

**TRACE_ENTRY / TRACE_EXIT — gated.** Only emitted in
`@trace`-attributed fns. `@trace` is not a META_ATTR — it's
treated as a declared effect at typecheck.py:467 because
`"trace"` isn't in the recognized set there (and the cycle-1
"unknown effect" branch would assign it to `effects`). So
`@pure @trace fn ...` is caught at typecheck time
("cannot be both @pure and have @effect"). **Gated-unreachable
from `@pure`**, but the verifier still doesn't catch them in
the closure if `@trace` happens to be added without `@pure`.

### Step 4 — finding C22-1 (HIGH, FFI_CALL effect-classification gap)

**Severity: HIGH**

**Reproducer (constructed mentally, NOT run; the source-level
chain is laid out above):**

```helix
extern "C" fn puts(s: *const u8) -> i32;

@pure fn shout(p: *const u8) -> i32 {
    puts(p)        // FFI_CALL — silently passes effect_check.
}

fn main() -> i32 { shout("hi\0".as_ptr()) }
```

The AST-layer `_check_call_effects` sees an empty `sig.effects`
on `puts` and raises nothing. The IR-layer `effect_check.py`
sees a body whose only effectful op is `FFI_CALL`, not in
OP_EFFECTS and not in callees() — closure is empty, `@pure`
contract passes vacuously. The `puts` call writes to stdout at
runtime — a clear `"io"` effect that the verifier failed to
detect. No test in `test_effect_check.py` pins this scenario
(the unknown-callee test uses synthetic `OpKind.CALL`, not
`OpKind.FFI_CALL`, so it does NOT exercise the gap).

**Defect family.** Same OpKind-omission pattern as the Stage
16.5 audit CRITICAL-1 fix that added `FFI_CALL` to DCE's
`SIDE_EFFECT_KINDS` set (visible at `dce.py:60-64`). The DCE
audit closed the dead-code-elim window; the effect_check
window has remained open since.

**Why it survived prior cycles.** Cycle 14 explicitly
enumerated the DCE side-effects table (silent-failures doc
lines 195-210) and confirmed every entry was justified by
defect history. Cycle 14 did NOT cross-check this against
`OP_EFFECTS` in effect_check.py. The two tables encode
different (but adjacent) contracts:

- `SIDE_EFFECT_KINDS` (DCE) = "ops whose execution must be
  preserved regardless of result-liveness."
- `OP_EFFECTS` (effect_check) = "ops whose execution
  contributes to the fn's effect closure."

There is no automated coherence check linking the two. A
maintainer who adds a new effectful OpKind correctly to
SIDE_EFFECT_KINDS (because the DCE-failure mode is visible
via byte diff on the produced ELF) can silently forget the
adjacent OP_EFFECTS update — and effect_check.py degrades
silently. The FFI_CALL audit CRITICAL-1 fix landed exactly
this asymmetry: DCE got patched, OP_EFFECTS did not.

**Recommended fix (out-of-scope per cycle-22 read-only
constraint).** Add to `OP_EFFECTS`:

```python
tir.OpKind.FFI_CALL: frozenset({"unknown"}),
```

Or to `callees()` line 127-132: also iterate `FFI_CALL` and add
its `attrs["target"]` to the callee set so the closure picks
up "unknown" via the existing unknown-callee branch (line
163-165). Either approach restores the @pure-vs-FFI gate.
Regression test should mirror `test_unknown_callee_treated_as_
unknown_effect` but use `OpKind.FFI_CALL` instead of `CALL` to
pin the actual extern path. Companion fix recommended for
ARENA_PUSH / ARENA_SET to close C22-3.

**Impact.** Silent soundness violation of `@pure` for the entire
FFI surface (Stage 16.5 onward). Any downstream pass that uses
`is_pure_decl` to guide pure-call optimizations (CSE, GVN,
hoisting, memoization) may now safely transform FFI-calling
`@pure` fns in ways that violate observable semantics. Note
that today's pass pipeline doesn't (yet) consult `is_pure_decl`
for CSE/GVN gating — but the contract is the verifier's
soundness signature, not the optimizer's current consumption.
A user reading "effect_check passed" and concluding "@pure
holds" is reading correctly per the published contract; the
verifier is wrong.

### Step 5 — companion gaps

**C22-2 (LOW, reflection ops missing from OP_EFFECTS):**
`QUOTE` and `REFLECT_HASH` produce reflection-cell side effects
(cell handle reservation) that downstream `MODIFY`/`SPLICE` may
reference by index. The DCE canon recognizes this (lines 47-50
of `dce.py`); OP_EFFECTS does not. Reachability under `@pure`
is gated-unreachable in practice (reflection emission today is
inside unsafe blocks), but the gap is the same defect class as
C22-1 and would open if the surface syntax for `quote(...)`
ever became callable from `@pure`-allowed contexts.

**C22-3 (HIGH, arena ops missing from OP_EFFECTS):**
`ARENA_PUSH` and `ARENA_SET` mutate the global bump arena. They
are reachable from `@pure` via the `__arena_push` / `__arena_set`
builtins listed in `typecheck.py:1046`. Same OpKind-omission
pattern as C22-1. Same recommended fix path. The "unknown"
effect bucket would suffice for Phase-0; a future `"arena"`
effect label could be added to typecheck.py:467 in parallel.

**C22-4 (LOW, TILE_INDEX_STORE missing from OP_EFFECTS):**
HBM-store side effect emitted only inside `@kernel` fns. Gated-
unreachable from `@pure` today by the implicit kernel-vs-pure
exclusion at lower_ast, but the verifier doesn't enforce.

**C22-5 (LOW, TRACE_ENTRY / TRACE_EXIT missing from
OP_EFFECTS):** Trace-runtime events. Gated by the implicit
`@trace`-vs-`@pure` rejection at typecheck.py:467, but verifier
itself is unaware.

**Aggregation policy.** C22-1 and C22-3 are reachable from
current surface syntax and warrant the HIGH classification.
C22-2 / C22-4 / C22-5 are companions in the same defect
family but currently gated-unreachable. Per the cycle-22 strict
bar (zero findings of any severity), all five are reported but
only C22-1 / C22-3 are HIGH; the others are LOW gated-
unreachable companions retained for documentation completeness
and to inform the consolidated fix-sweep.

---

## Audit findings

**Cycle 22 silent-failures audit: NOT CLEAN.**

| Severity   | Count |
|------------|-------|
| CRITICAL   | 0     |
| HIGH       | 2     |
| MEDIUM     | 0     |
| LOW        | 3     |
| **Total**  | **5** |

### Findings

- **C22-1 (HIGH).** `effect_check.OP_EFFECTS` and
  `effect_check.callees()` omit `OpKind.FFI_CALL`. A `@pure`
  Helix fn that calls an extern "C" function (e.g. libc `puts`)
  silently passes the IR-level effect verifier. The
  @pure-contract soundness window has been open since Stage
  16.5 landed FFI_CALL. Reachable from surface syntax. Same
  OpKind-omission family as Stage 16.5 DCE CRITICAL-1.
  Recommended fix: add `FFI_CALL: frozenset({"unknown"})` to
  OP_EFFECTS (and/or iterate FFI_CALL in `callees()` to pick
  up the target name into the unknown-callee branch). Plus a
  regression test patterned on
  `test_unknown_callee_treated_as_unknown_effect` but using
  `OpKind.FFI_CALL`.

- **C22-3 (HIGH).** `effect_check.OP_EFFECTS` omits
  `OpKind.ARENA_PUSH` and `OpKind.ARENA_SET`. Both mutate the
  global bump arena and are reachable from `@pure` via the
  `__arena_push` / `__arena_set` builtins (`typecheck.py:1046`).
  Same defect family as C22-1.

- **C22-2 (LOW).** `OP_EFFECTS` omits `OpKind.QUOTE` and
  `OpKind.REFLECT_HASH`. Reflection-cell handle reservation
  unmodeled. Gated-unreachable from `@pure` today
  (reflection emission is unsafe-blocked); fix on the
  consolidated sweep.

- **C22-4 (LOW).** `OP_EFFECTS` omits `OpKind.TILE_INDEX_STORE`.
  HBM-write side effect. Gated-unreachable from `@pure` today
  by the implicit kernel-vs-pure exclusion.

- **C22-5 (LOW).** `OP_EFFECTS` omits `OpKind.TRACE_ENTRY` and
  `OpKind.TRACE_EXIT`. Runtime trace events. Gated by the
  implicit `@trace`-vs-`@pure` rejection at typecheck.py:467
  but verifier-unaware.

### Companion clean targets

- **`elf_dyn.py` (Target 3): CLEAN.** Eleven explicit
  `assert len(...) == ...` boundary checks plus one explicit
  `raise RuntimeError` overflow check. Zero `.get(..., default)`
  silent-fallback sites. No width-keyed tables (out of cycle-
  13-through-21 defect class).
- **`parser.py` error paths (Target 2): CLEAN.** Only one
  `except` site (`_parse_autotune_int` line 375-376) — narrow,
  re-raises ParseError. One documented lenient-skip site
  (`_parse_string_attr_arg`) recorded as forward note F-22-1.
  Stdlib-missing-file mode emits to stderr (not silent). All
  primary parse productions raise ParseError on
  unexpected-token.

### Clean-cycle counter

**Counter reset: 1/5 → 0/5.** C22-1 and C22-3 are real silent
soundness failures of the `@pure` contract. The strict-criterion
bar is "zero findings of any severity"; cycle 22 has 2 HIGH and
3 LOW findings.

The streak of clean cycles is broken. The next clean-cycle
sequence begins after a fix-sweep that closes C22-1 + C22-3
(and ideally C22-2 / C22-4 / C22-5 in the same sweep to avoid
the same incremental-defect-class pattern that produced six
consecutive HIGH findings in the width-narrowing family).

---

## Out-of-scope per task instructions

- Per the cycle-22 task brief, this is a read-only audit. No
  production-code or test edits.
- The "centralize scalar-width predicate" Stage-29 refactor
  recommendation (carried since cycle-17 forward notes) is
  unrelated to this cycle's surface; not re-cited.
- Cycle-21 forward note F-21-3 (`test_bootstrap_kovc_full_
  pipeline_arithmetic` failing at HEAD `bee36e6` with the same
  drift `100 - 50 - 8 → 132` as at parent `5a1e406`) is noted
  but **explicitly not re-flagged** per the cycle-22 brief —
  pre-existing and orthogonal to the rotation surfaces.
- C22-2 / C22-4 / C22-5 are companions in the same defect
  family as C22-1 / C22-3 but classified LOW because of
  current-frontend gating; they would become HIGH if a future
  change (e.g. widening unsafe-block requirements on QUOTE /
  promoting `@trace` into a META_ATTR) opens reachability.

---

## Files touched by this audit

None — read-only audit cycle. No production-code or test edits.
Only this doc.

## Cross-reference

- Cycle 21 silent-failures (declared CLEAN, advanced to 1/5):
  `docs/audit-stage28-8-cycle21-silent-failures.md`.
- Stage 16.5 FFI_CALL DCE audit CRITICAL-1 (sibling-pattern fix
  to OP_EFFECTS gap, closed for DCE only):
  `helixc/ir/passes/dce.py:60-64` (comment block) — landed in
  the Stage-16.5 follow-up audit; was NOT propagated to
  `effect_check.OP_EFFECTS`.
- Effect-check soundness-layer claim:
  `helixc/frontend/typecheck.py:1011-1013` (AST-level check
  explicitly defers to IR-level).
- Synthetic-CALL test that does NOT exercise the FFI path:
  `helixc/tests/test_effect_check.py:71-91`
  (`test_unknown_callee_treated_as_unknown_effect`).
- Cycle 14 enumeration of DCE side-effects (did not cross-check
  OP_EFFECTS): `docs/audit-stage28-8-cycle14-silent-failures.md`
  lines 195-210.
- Cycle-21 forward note F-21-3 (noted, not re-flagged):
  `docs/audit-stage28-8-cycle21-codereview.md:491-499`.
