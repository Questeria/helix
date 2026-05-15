# Helix Pre-Phase-A Finalization Research

> Historical pre-Stage-29 research snapshot; not live gate evidence for Stage
> 35. Live status is tracked in `docs/ROADMAP.md` and
> `docs/stage35-progress-2026-05-15.md`.

**Date**: 2026-05-11
**Context**: Stage 28.8 (pre-29 audit gate) is essentially closed after 11
audit cycles (cycle 11 cleaning advances counter to 5/5 per cycle-11
type-design and silent-failure docs; cycles 12-15 are stability re-passes at
the same HEAD). Heavy gate stable at ~1427 tests passing. Phase A
(Stages 28.9–28.13: port 8 Python-only frontend passes into kovc.hx) is
about to lock in scope.
**Builds on**: `docs/helix-pre-self-host-research.md` (5056 words; identified
the 8-Python-only-passes blocker and Phase A/B split).
**Scope**: Read-only. No source modifications. No tests added. No commits.
**Author**: Follow-on research agent commissioned by Kovostov-Native lead.

---

## Executive Summary

Since the original research doc landed on 2026-05-10, eleven multi-agent
audit cycles (cycles 1–11) have processed Stage 28.8 and produced ~80+
discrete findings across silent-failure, type-design, and code-review
lenses. About a third of those findings cluster into three SYSTEMIC
patterns that the original research treated as one-off bugs but that should
be read as **missing infrastructure**: (a) three separate AST-walker
implementations (panic_pass, unsafe_pass, deprecated_pass) plus grad_pass's
ad-hoc dispatch all repeatedly drifted on the same attribute lists; (b)
several passes existed in `helixc/frontend/*.py` but were not wired into
`helixc/check.py` until a cycle-1 audit caught it (A1, A2, A7, A8, A9, A10,
A11, A12, B1, B7); (c) cycle 11 surfaced **codegen non-determinism** —
`test_bootstrap_kovc_full_pipeline_arithmetic` produced three distinct
cached-binary hashes across consecutive invocations — and the root cause is
verifiable via grep: 9 call sites in `helixc/backend/x86_64.py` use
`id(op):x` as part of generated symbol names, and `match_lower._FRESH_COUNTER`
is module-level mutable state.

The original research recommended Phase A = 5 stages (match_lower port,
struct_mono port, validation passes batch, pytree port, ergonomics
cluster). That stays correct **as language scope**, but the 11 audit
cycles surface three pre-Phase-A items that should be addressed FIRST
because they block byte-identical verification at Stage 29 regardless of
what Phase A does: (1) the `id()`-in-symbol-name codegen non-determinism;
(2) the `_FRESH_COUNTER` module-level state in match_lower; (3) the AST
walker drift — without a shared walker library, every Phase A port adds
yet another walker that will drift on the next AST change.

**Top 3 recommendations (do BEFORE Phase A starts):**

  1. **Stage 28.8.1: Codegen determinism harden.** Replace `id(op)` symbol
     generation in `helixc/backend/x86_64.py` (9 call sites: lines 1705,
     1900, 1940, 1941, 2018, 2158, 2310, 2457, 2516) with
     `(fn_index, op_index)` tuples derived from IR traversal order. Reset
     `match_lower._FRESH_COUNTER` per program. Audit `autodiff._DIFF_WARNINGS`
     module-level list for cross-program leakage. Effort: 0.5 stage; risk:
     LOW (purely cosmetic in produced bytes — `id()` was already
     non-semantic, but the symbol names were leaking process-internal
     state into the byte stream). **Stage 29's byte-identical gate cannot
     pass until this lands.**

  2. **Stage 28.8.2: Shared AST walker library.** Extract the
     attribute-list dispatch pattern out of `panic_pass`,
     `unsafe_pass`, `deprecated_pass`, `grad_pass`, `struct_mono` and into
     a single `frontend/ast_walker.py`. Each pass becomes a visitor
     subclass with `visit_X` overrides. Closes ~20 audit findings across
     cycles 1–11 (A5, A6, A10, A11, C1-H1, C1-L1, C2-4, C3-1, C3-5, …)
     and prevents Phase A from adding three MORE drift-prone walkers
     (match_lower's, struct_mono's, pytree's). Effort: 1 stage; risk:
     MEDIUM (every pass touches every node kind; mistakes silently lose
     coverage). **Phase A is materially cheaper after this lands.**

  3. **Phase A re-ordering.** The original sequence was:
     28.9 match_lower → 28.10 struct_mono → 28.11 validation passes
     → 28.12 pytree → 28.13 ergonomics. The 11 audit cycles change
     the dependency analysis: pytree (28.12) is *upstream* of
     struct_mono in the data-flow graph (pytree walks struct decls,
     struct_mono mutates them); the validation passes (28.11) are the
     simplest port and validate the shared walker library (rec #2); and
     the ergonomics cluster (28.13) has hidden cross-cuts to struct_mono
     (named struct-lit fields) and match_lower (`let-else` desugars to
     match). Proposed new order: 28.8.1 determinism → 28.8.2 walker lib
     → 28.9 validation passes → 28.10 match_lower → 28.11 struct_mono
     → 28.12 pytree → 28.13 ergonomics. Effort: same five stages, but
     the dependency graph is honored; risk: LOWER (each stage's surface
     is smaller because the previous stage's infrastructure is in
     place).

The rest of this document expands on the 6 research categories with
specific source-file citations and effort estimates. **All
recommendations are addable to the master plan as Stages 28.8.1, 28.8.2,
and a re-ordering of 28.9–28.13.** Phase B (Tier-3 strategic moat) stays
deferred to v0.2 — nothing in the 11 audit cycles changes that
recommendation.

---

## Cat A: Systemic patterns from 11 audit cycles

The 11 audit cycles produced ~80+ findings (rough count: ~50 in cycles 1–5
during active fix-sweeps, ~30 deferred/below-threshold, ~3 still-open
CRITICAL/HIGH carryovers — C4-1, C4-4, C4-8). Reading them as a
population, three SYSTEMIC patterns emerge that should be addressed as
infrastructure rather than as one-off fixes.

### A1. Three AST walkers drifted on the same attribute lists

**Evidence**: `helixc/frontend/panic_pass.py:39-105`, `unsafe_pass.py:37-96`,
and `deprecated_pass.py:84-135` each define their own `_walk*` function.
Each uses `getattr(node, attr, None)` + `hasattr(sub, "span")` against a
hard-coded attribute list. Cycles 1–3 found that the lists had drifted:
panic_pass used `"then_branch"` / `"else_branch"` (legacy names that don't
exist on any node) while deprecated_pass and unsafe_pass used the correct
`"then"` / `"else_"` (C1-H1, HIGH, cycle 1). Cycle 2 found that all three
walkers missed `iter_expr`, `obj`, `target`, `start`, `end`, `guard`,
`inner`, `transformation`, `verifier`, `indices` (A5/A6/A10 cycle 1, plus
similar in grad_pass's `_expr_has_grad` per C2-4 cycle 2). The current
files are byte-for-byte similar in their attribute lists (lines 76-86 of
panic_pass.py vs 71-81 of unsafe_pass.py vs 111-122 of deprecated_pass.py)
— but only because every audit cycle since cycle 1 has manually
synchronized them. The next AST change WILL drift them again.

**Recommendation**: Extract `frontend/ast_walker.py` with a single
`ASTVisitor` base class. Each pass subclasses with `visit_Call`,
`visit_If`, etc. The base class's `generic_visit` introspects
`ast_nodes.py` field annotations (via `dataclasses.fields(node)`) to walk
ALL child nodes regardless of attribute name. This makes drift impossible
without changing the AST schema itself.

**Effort**: ~1 stage. Each pass refactor is 30–80 LoC. New walker library
is ~150 LoC. Net code delta: NEGATIVE (the three walkers + grad_pass's
ad-hoc dispatch are ~400 LoC combined; the visitor base class is ~150 LoC
and each visitor subclass is ~30 LoC — net savings ~120 LoC).

**Risk if deferred**: Phase A adds **three more** walkers (match_lower,
struct_mono's collect_concrete_uses + visit_expr, pytree's flatten/unflatten
which already has cycle-guard logic), each of which will drift on the next
AST change. Without a shared library, the bootstrap port of these passes
inherits the drift pattern.

**Recommendation status**: **DO BEFORE PHASE A** as new Stage 28.8.2.

### A2. Multiple "validation passes never wired into CLI" findings

**Evidence**: Cycle 1 audits A1, A2, A7, A8, A9, A10, A11, A12, B1, B7
(and three more from prior `audit-stage{5-6,7-8,9-16}` docs) each surfaced
a Python-side pass that existed in `helixc/frontend/` but was never called
by `helixc/check.py`. Examples:

- `panic_pass.validate_panic_args` — added in Stage 28.5, never wired
  (cycle 1 A1).
- `unsafe_pass.check_unsafe_ops` — added in Stage 28.6, never wired (A2).
- `trace_pass.validate_trace_attrs` — added in Stage 25, never wired (A7).
- `flatten_impls` — was only invoked inside the backend; surface-tool
  users iterating with `helixc check foo.hx` never saw duplicate-method
  diagnostics (A12 / cycle 2 B:C7).
- `struct_mono.monomorphize_structs` — added in Stage 28, partially wired
  but with diagnostics dropped silently (A3/B1/C1-M2).
- `-O1/-O2/-O3` flags — accepted by parse_args but never actually invoked
  fold/cse/dce until cycle 1 A10.
- `--emit-asm` and `-o` flags — backend errors were not trapped cleanly
  until A9.

**Recommendation**: Add a `frontend/pass_registry.py` that lists every
pass in topological order with `{name, callable, error_class,
diagnostic_label, fatal_on_error}` tuples. `check.py` iterates the
registry instead of unrolling each pass with bespoke print+continue logic.
This makes "pass X is not wired" structurally impossible — every entry in
the registry is invoked by construction, and the only way to add a pass
without wiring it is to skip the registry.

**Effort**: ~0.5 stage. `check.py` currently has lines 380–700 of pass
invocations (~320 lines, ~20 passes, ~16 lines/pass for error handling
boilerplate). The registry version is one driver loop (~30 lines) plus
~20 registry entries (~3 lines/entry = ~60 lines) — net savings ~230
lines.

**Risk if deferred**: Phase A adds at least 4 new passes to the registry
(panic/unsafe/deprecated/trace validations were already added but the
bootstrap version needs equivalent wiring). Without the registry, the
bootstrap-side `kovc.hx` will repeat the same "added but not wired"
pattern.

**Recommendation status**: **Defer to Stage 28.13 ergonomics cluster** —
the wiring debt is manageable as long as the AST walker library lands
first (the registry adds value but is not blocking). Spec it now;
implement during 28.13 or post-29.

### A3. Codegen non-determinism (cycle 11 finding)

**Evidence**: Cycle 11 silent-failures (`docs/audit-stage28-8-cycle11-silent-failures.md`
lines 412–486) documented that
`test_codegen.py::test_bootstrap_kovc_full_pipeline_arithmetic` produced
three distinct failure modes across three invocations:

1. `assert 1 == 255` for `compile_and_exec("~0")`, cache hash
   `9ec7a36127416cf3`.
2. `assert 2 == 14` for `compile_and_exec("2 + 3 * 4")`, cache hash
   `c00c44d73441dd46`.
3. `subprocess.TimeoutExpired` for `compile_and_exec("42")`, cache hash
   `9ec7a36127416cf3`.

The differing cache hashes mean the SAME bootstrap source produces
different ELF bytes across runs of Python on the same machine. The cycle-11
type-design audit doc identified this as out-of-scope for type-design but
flagged for downstream investigation (cycle-11 silent-failures § 6, "Out-
of-scope observation"). **Live investigation** in preparation of this
research doc identified the concrete sources:

**Source 1** (HIGH): `helixc/backend/x86_64.py` uses `f"__helix_*_{id(op):x}"`
in 9 call sites: lines 1705 (`strptr`), 1900 (`strbyte`), 1940 (`path`),
1941 (`content`), 2018 (`rftoa_path`), 2158 (`wftoa_path`), 2310 (`path`),
2457 (`str`), 2516 (`panic`). Each emits a symbol whose name embeds the
Python object identity (`id()`) of the IR op. Python's `id()` is the
object's memory address, which varies across processes and even within a
single process when objects are GC'd and re-allocated. This means **every
time the backend runs, every string-emitting site gets different symbol
names**, which changes the ELF symbol table and (through relocations) the
final bytes. The test_bootstrap_kovc_full_pipeline_arithmetic flake is a
direct consequence.

**Source 2** (HIGH): `helixc/frontend/match_lower.py:48`:

```python
_FRESH_COUNTER = [0]

def _fresh_name(prefix: str = "__scrut") -> str:
    _FRESH_COUNTER[0] += 1
    return f"{prefix}_{_FRESH_COUNTER[0]}"
```

This is module-level mutable state. It is NOT reset between calls to
`lower_matches(prog)`. Two consequences:

  1. Running pytest with multiple match-using tests in the same process
     produces different IR (e.g. `__scrut_47` vs `__scrut_5`) depending
     on test order. Test isolation via pytest fixtures does not save us
     — `_FRESH_COUNTER` is global to the helixc module.
  2. The bootstrap test invokes Python helixc once to build the bootstrap
     compiler, then runs the bootstrap compiler 285 times in subprocess.
     But the FIRST invocation of helixc may have already had its
     `_FRESH_COUNTER` polluted by an earlier test in the same pytest
     session.

The `match_lower` audit findings (cycle 1 onwards) never flagged
`_FRESH_COUNTER` because it doesn't produce wrong answers — it just
produces different names. The names propagate through IR and end up
embedded in error-strings + register hints, which can affect codegen
through allocation-order side effects.

**Source 3** (MEDIUM): `helixc/frontend/autodiff.py:54`: `_DIFF_WARNINGS:
list[str] = []` is similar — module-level state. Cycle 2 C2-1 (CRITICAL)
called this out for a different bug (orphan warnings persisting across
compilations) but the same pattern is a determinism hazard.

**Source 4** (LOW): `helixc/ir/passes/cse.py:122` uses `value_id: int` as
a parameter, which is fine — but `cse.py` and `dce.py` and `fdce.py` all
use `set()` for "seen" tracking. Set iteration order in CPython is
insertion-ordered (since 3.7 for dicts; sets are arbitrary but
deterministic within a single CPython version). This is a non-issue
*today* but is a latent risk if a future refactor adds a `for x in set(...)`
loop that affects output ordering.

**Source 5** (NEGATIVE — not a source): I grepped for `time.time()`,
`time.monotonic()`, `datetime.now()`, `time.perf_counter()`, `time_ns` in
`helixc/` and found zero matches. The compiler is **clean of timestamp
leaks** into output bytes. Same for `os.environ` — only two reads:
`diagnostics.py:62-64` (NO_COLOR / HELIXC_COLOR — affects stderr text,
not output bytes) and `parser.py:1563` (STDLIB_STRICT_ENV — affects which
files are read, not bytes-of-output for a given input).

**Recommendation**: Stage 28.8.1 "codegen determinism harden":

  1. Replace `id(op):x` with a per-fn op index. Walk the fn's IR once
     before codegen, build `op_to_index: dict[Op, int]`, generate symbols
     as `f"__helix_strptr_{fn_index}_{op_index}"`.
  2. Reset `match_lower._FRESH_COUNTER[0] = 0` at the top of
     `lower_matches(prog)`. Make `_fresh_name` take a counter parameter
     or move the counter into a `MatchLowerer` class.
  3. Same treatment for `autodiff._DIFF_WARNINGS` — make it a per-
     compilation list passed in.
  4. Add a `test_codegen_deterministic_bytes.py` that compiles the same
     source 5 times in the same process and asserts byte-identical ELF
     output.

**Effort**: ~0.5 stage. The `id(op):x` replacement is mechanical (9 call
sites + a dict-build helper). `_FRESH_COUNTER` is a 3-line refactor.

**Risk if deferred**: **Stage 29's byte-identical verification gate
cannot pass with the current determinism leaks.** This is a HARD blocker
for the entire Stage-29 plan.

**Recommendation status**: **DO BEFORE PHASE A** as new Stage 28.8.1.
The bootstrap byte-identical verification at Stage 29 explicitly depends
on this.

### A4. Repeated `TyUnknown` fallback as silent-pass channel

**Evidence**: Cycles 5–10 (especially C5-3, C6-1, C7-1) traced a recurring
type-system pattern: `_compatible(TyUnknown, *)` returns True, and several
typecheck arms `return TyUnknown(...)` as a "didn't know what to do"
fallback. The combination silently passes mis-typed code. Cycle 5 F4
documented that `TyMemTier` × `TyMemTier` uses string equality on tiers
(no subsumption); cycle 5 F3 documented that `TyLogic` provenance is not
checked in `_compatible` (only at the call boundary via the dedicated
provenance trap). Cycle 8 C7-1 dropped the cycle-7 `TyMemTier × (TyVar |
TySize)` carve-out for the same reason — these fallback paths are
silent-pass channels rather than principled type-equality decisions.

**Recommendation**: Add a `frontend/typecheck_exhaustiveness.py` audit
helper that walks `typecheck.py`'s `_compatible`, `_check_call_basic`,
`_resolve_type`, and `_size_compatible` cascades and produces a coverage
matrix: for every pair `(TypeKind, TypeKind)`, what's the verdict? Today
this matrix is ~20×20 = 400 cells; ~70 cells are TRUE, ~280 are FALSE,
~50 are "depends on inner". The matrix should be auditable: any new
TypeKind addition forces an explicit decision for every existing
TypeKind.

**Effort**: ~0.3 stage. The helper is a small dict-of-dicts that
`typecheck.py` populates as it dispatches; assertion at end of typecheck
verifies all cells are decided. Could be a `--strict` flag rather than
a required production check.

**Risk if deferred**: As Stage 29 lands more TypeKinds (TyMemTier,
TySkill are recent; future TyEffect, TyAlias, TyExistential would
extend), each addition introduces N new "did I forget this combination"
risks. The current cycle-5–10 cascade of carve-outs shows the pattern.

**Recommendation status**: **DEFER to v0.2.** Not blocking Stage 29; the
current cascade is correct *as-is* with the strict-zero rule enforcing
correctness. Add when TyEffect or TyAlias arrives.

### A5. Phase A ports will inherit walker-fragility unless addressed

**Evidence**: The original research's Phase A ports `match_lower`,
`struct_mono`, `pytree`, and four validation passes. Each has its own
walker today. The Phase A bootstrap port (kovc.hx) will translate these
walkers into Helix code. If the Python-side walkers are still
drift-prone, the bootstrap port inherits the drift pattern, and the
bootstrap-side audit cycles will find the same class of bugs that
cycles 1–11 of Python found.

**Recommendation**: Order Stage 28.8.2 (shared walker library) BEFORE
Stage 28.9 (first port). The bootstrap-side equivalent — a `visit_ast`
helper in `kovc.hx` that takes a node + a callback table — is then a
*direct* port of a clean library rather than a re-implementation of
walker-drift bugs.

**Recommendation status**: Implied by A1 above.

### A6. Audit-cycle scope drift

**Evidence**: Cycles 11–14 are stability re-passes at the same HEAD
(commit c2e36d4). The strict-zero rule + the "don't re-flag carryovers"
rule means that once a cycle is clean, the next cycle has nothing to do
unless production code changes. This works for *closing* Stage 28.8 but
produces no useful work between Stage 28.8 closure and Phase A start.

**Recommendation**: The cycle-counter mechanism should treat
"production-code unchanged since last clean cycle" as auto-CLEAN +
counter-advance, freeing the audit-agent budget for forward-looking
research (like this doc) instead of repeated re-walks.

**Recommendation status**: Process improvement; not language-scope. Spec
in `STAGE_28_8_PRE_29_AUDIT_GATE.md` next time the doc is touched.

---

## Cat B: Refined Phase A scope

The original research doc proposed Phase A as Stages 28.9–28.13, five
stages, total ~30–50 commits. The 11 audit cycles change the dependency
analysis and the relative effort estimates. I recommend a re-ordering
and one promotion from "deferred" to "Phase A":

### B1. Recommended new order

| Stage | What | Effort | Risk | Rationale |
|-------|------|--------|------|-----------|
| **28.8.1** | Codegen determinism harden | 0.5 stage | LOW | Stage-29 byte-identical gate blocker; not optional |
| **28.8.2** | Shared AST walker library | 1 stage | MEDIUM | Phase A ports cheaper after this lands |
| **28.9** | Port validation passes (panic/unsafe/deprecated/trace) | 1 stage | LOW (small individually) | Validates 28.8.2 against simplest call site |
| **28.10** | Port match_lower | 1 stage | MEDIUM | Foundational for subsequent passes |
| **28.11** | Port struct_mono | 1.5 stages | MEDIUM | Cross-cuts typecheck |
| **28.12** | Port pytree | 1 stage | LOW | Depends on 28.11 struct_mono |
| **28.13** | Ergonomics cluster | 1 stage | LOW | Last because each item's surface is small |

**Why this order vs the original**:

- 28.8.1 first because Stage 29's byte-identical verification cannot pass
  without it (concretely demonstrated by cycle-11 flake on
  test_bootstrap_kovc_full_pipeline_arithmetic).
- 28.8.2 first because every Phase A stage adds a new walker; doing them
  before the shared library means they enter the codebase as drift-prone
  per-pass walkers and have to be re-refactored later.
- Validation passes BEFORE match_lower: they're additive, easy to test,
  and validate the walker library against simple patterns (call sites,
  unsafe blocks, deprecated calls). Match_lower is bigger and has more
  surface for bugs; doing it second means the walker library has been
  exercised first.
- Pytree AFTER struct_mono: pytree walks struct decls. If struct_mono
  mutates them (creating Pt__i32 / Pt__f64 mono'd clones), pytree should
  walk the post-mono shape. Doing pytree first would mean pytree runs
  before mono'd structs exist, and then needs to re-run.
- Ergonomics LAST because `f"..."` needs to know about the post-Phase-A
  call surface (which `to_string<T>` impls exist?), `let-else` needs
  match_lower to land first (it desugars to a match), named struct-lit
  needs struct_mono (mono'd fields need to be name-addressable).

**Concrete shift from original plan**:

- Original 28.9 (match_lower port) becomes new 28.10.
- Original 28.10 (struct_mono port) becomes new 28.11.
- Original 28.11 (validation passes) becomes new 28.9 (promoted to first
  port).
- Original 28.12 (pytree) stays as 28.12.
- Original 28.13 (ergonomics) stays as 28.13.
- New 28.8.1 (determinism) and 28.8.2 (walker library) inserted before.

### B2. Items to ADD to Phase A

**Promote `monomorphize.py` port to Phase A.** The original research doc
listed `monomorphize.py` as "partially mirrored in the bootstrap" but
cycle 11 type-design probe 6 (nested generics across module boundaries)
revealed that bootstrap's `mono_table` only handles top-level `fn id<T>`
patterns; nested generics like `Pt<Vec<T>>` are partially handled by
struct_mono (Python-side) but the bootstrap side has no equivalent. After
struct_mono lands in Phase A, the bootstrap needs the *fn* monomorphize
port too. **Add as Stage 28.11.5** (between struct_mono and pytree).

**Effort**: ~0.5 stage. The Python `monomorphize.py` is 760 LoC but most
is shape-folding helpers; the core walk is ~150 LoC.

**Risk**: Cross-cuts with struct_mono — fn mono and struct mono should
share the `_ty_key` keying machinery. The cycle-3 D6 "resolved Type leaks
into mono keying" hard-raise should be ported once for both.

### B3. Items to DROP from Phase A

**Drop `f"..."` string interpolation from Phase A**, defer to v0.2.

**Rationale**: The original research argued `f"..."` is necessary for
post-29 diagnostics. The 11 audit cycles show that the current Python
diagnostics (`diagnostics.py`, 252 lines) is *adequate* for the
self-host story — the bootstrap can compile programs whose error
messages use plain `str_concat` chains. The post-29 diagnostics will be
plainer-looking but won't lose any information. The Phase A scope is
already 5–6 stages; dropping `f"..."` makes it 5 cleanly.

**Counter-argument**: `f"..."` is cheap (~0.5 stage). Including it makes
post-29 stdlib work nicer. Keep it if Phase A budget is comfortable.

**Verdict**: Defer to v0.2 unless Phase A budget has slack.

### B4. Items to ADD that the original didn't mention

**Render_caret port.** The original research mentioned this as part of
Stage 28.13. After cycle 5's audit (C4-7 cast-ref-prefix diagnostic),
the bootstrap's `emit_trap_with_id` style is *significantly* worse than
the Python `render_caret`. A user post-Stage-29 who hits a typecheck
error sees `trap 14001` and has to grep the registry. **Promote
render_caret to its own Stage 28.13.5** (between ergonomics and post-A
audit). Effort: ~0.5 stage.

**Auto-derive PartialEq / Debug / Clone for user structs.** Many cycle-11
audits noted that the bootstrap can't auto-derive trait impls for user
structs. This is a v0.2 item but a *partial* version (auto-PartialEq for
field-by-field equality, no trait machinery yet) would close ~30 stdlib
LoC. Cheap. **Don't add to Phase A** — too many cross-cuts. Spec for
v0.2.

### B5. Phase B (Tier-3 strategic moat): re-evaluation

Original research recommended Phase B deferred to v0.2 pending external
benchmark validation. The 11 audit cycles haven't surfaced anything that
changes this. The `D<Logic<T>>` and `TyMemTier` work is type-level only;
no codegen-side enforcement. Cycle 11 type-design probe 1 specifically
re-audited the `TyMemTier` strict-equality + cross-kind separation and
confirmed it's correct for Phase 0 but doesn't model HBM/DDR/NVMe tiers
(the cognitive tier model is `working/episodic/semantic/procedural`, a
different surface than what the user might assume from the name).
**Phase B stays deferred.** Promote when a real neuro-symbolic AGI task
exists.

---

## Cat C: Stage-29 determinism investigation

The cycle-11 silent-failures audit (lines 412–486) noted the flake but
classified it out-of-scope for that audit's charter. This research doc's
live investigation identified the concrete sources. Repeated here in
checklist form for Stage 28.8.1 implementation.

### C1. Non-determinism inventory

| Source | File:line | Risk | Fix sketch |
|--------|-----------|------|-----------|
| `id(op):x` symbol generation | `backend/x86_64.py:1705, 1900, 1940, 1941, 2018, 2158, 2310, 2457, 2516` | HIGH (process-address leakage) | Replace with `(fn_idx, op_idx)` from IR traversal order. Pre-walk fn IR; build `op_to_idx`; emit symbols as `__helix_strptr_{fn_idx}_{op_idx}`. |
| `match_lower._FRESH_COUNTER` | `frontend/match_lower.py:48` | HIGH (cross-program leakage) | Reset to 0 at top of `lower_matches(prog)`. Or convert `_FRESH_COUNTER` + `_fresh_name` into a `MatchLowerer` instance. |
| `autodiff._DIFF_WARNINGS` | `frontend/autodiff.py:54` | MEDIUM (already flagged as C2-1) | Make per-compilation; pass as argument. Drain at end of typecheck. |
| `parser.STDLIB_FILES` | `frontend/parser.py:1519` | NONE (constant list) | No fix needed; constant. |
| Python `hash()` of strings | implicit in `dict` iteration | THEORETICAL (CPython 3.7+ insertion-ordered) | Set `PYTHONHASHSEED=0` in test runner if any `set()` iteration order ever affects output. Not currently a leak. |
| `os.environ` reads | `frontend/parser.py:1563`, `frontend/diagnostics.py:62-64` | NONE (affects stderr + which files read, not output bytes) | No fix; document. |
| `time.time()` / `datetime.now()` | none found | NONE | Document as forbidden. |

### C2. Verification plan

After Stage 28.8.1 lands:

```bash
# Repeat 10 times; SHA-256 must be identical across runs.
for i in 1 2 3 4 5 6 7 8 9 10; do
    python -m helixc.check --emit-asm helixc/stdlib/option.hx > /tmp/out_$i.asm
    python -m helixc.check -o /tmp/out_$i.elf helixc/stdlib/option.hx
done
sha256sum /tmp/out_*.asm  # must all match
sha256sum /tmp/out_*.elf  # must all match
```

Add as `test_codegen_deterministic_bytes.py` with at least:

  1. `test_same_source_same_bytes_multiple_times` — compile same source
     5 times, assert identical bytes.
  2. `test_same_source_same_bytes_different_test_order` — compile same
     source from inside a test that runs after several other tests in
     the same process; assert identical bytes.
  3. `test_bootstrap_pipeline_cache_hash_stable` — the failing test from
     cycle 11; assert the cache hash is stable across 5 invocations.

### C3. Stage-29 byte-identical gate plan

Stage 29's gate is "kovc.hx compiles every test that helixc-Python
compiles, byte-identical." This requires:

  1. Determinism in both helixc-Python AND kovc.hx (a.k.a. both stay the
     same across runs). Stage 28.8.1 addresses helixc-Python.
  2. Identical pipeline order (same passes in same order). The bootstrap
     `kovc.hx` order is established by Phase A's pass ports.
  3. Identical mangling rules. Struct mono mangling, fn mono mangling,
     impl-method mangling must match byte-for-byte. Cycle 11 type-design
     probe 6 verified that `_ty_key`'s mangling rules are deterministic
     given the AST shape; the audit didn't probe whether kovc.hx
     mirrors them exactly.
  4. Identical optimization order. const-fold, CSE, DCE, FDCE, hash-cons,
     totality — Phase A ports preserve order? Audit explicitly.

**Recommendation**: After Stage 28.8.1 lands, run the
`test_codegen_deterministic_bytes.py` suite continuously; add to CI/heavy
gate. Stage 29's gate then layers on top of byte-identical by also
comparing kovc.hx's output to helixc-Python's output for the same input.

### C4. Bootstrap-side determinism

The bootstrap `kovc.hx` is itself a Helix program. Does it have its own
determinism leaks? Spot-checked:

- `kovc.hx` uses an i32 arena indexed by integer offsets. No `id()`
  equivalent (Helix doesn't have one). No randomness.
- `parser.hx` mr_tab (turbofish/generic table) is indexed by parsed
  order. Deterministic.
- `lexer.hx` is purely lexical. Deterministic.

The bootstrap appears to be free of these leaks **by construction** —
Helix doesn't expose `id()` or `time()` to the user (those would be FFI
calls; the bootstrap doesn't make them in its compilation path). This is
a strong argument FOR Stage 29: the post-Python compiler is structurally
more deterministic than its predecessor.

---

## Cat D: Self-host completeness gaps

The original research listed 8 Python-only files. The 11 audit cycles
surfaced no NEW major Python-only passes (the file count is stable),
but identified some files that are partial in either direction.

### D1. Files mistakenly listed as "Python-only" in original

- `autodiff.py` / `autodiff_reverse.py`: These are listed as Python-only
  in the original. But the bootstrap stdlib `helixc/stdlib/autodiff.hx`
  (120 lines) + `autodiff_reverse.hx` (191 lines) DOES exist and is
  parsed. The bootstrap `kovc.hx` has Stage 12 (fwd AD) + Stage 14 (rev
  AD) wired. The Python versions are the *reference* (used by Python
  helixc); the bootstrap has its own. **No port needed.** Mark as
  "bootstrap parity exists" in master plan.
- `grad_pass.py`: Listed as Python-only. The bootstrap doesn't have an
  equivalent grad-rewriting pass. **True port-needed item.** Add to
  Phase A as ~0.3 stage (small).
- `flatten_modules.py` / `flatten_impls.py`: Listed as Python-only.
  Bootstrap has its own flatten — partial coverage. Cycle 1 A12 wired
  flatten_impls into check.py; the bootstrap-side equivalent exists
  but doesn't fire trap 74002 (duplicate method name). **Port-needed
  item; add to Phase A as ~0.5 stage.**
- `hash_cons.py`: Listed as Python-only. Bootstrap has working
  hash-cons (Stage 20). **No port needed.**
- `totality.py`: Listed as Python-only. Bootstrap has working
  totality check (Stage 21). **No port needed.**
- `ast_hash.py`: Listed as Python-only. Bootstrap has `ast_hash`
  equivalent inside `kovc.hx`. **No port needed.**
- `diagnostics.py`: Listed as Python-only (render_caret). Bootstrap has
  `emit_trap_with_id` (less rich). **Port-needed item for v0.1 user
  experience; add to Phase A as ~0.5 stage (B4 above).**

### D2. Files with bootstrap partial coverage that need explicit "post-Phase-A audit"

- `monomorphize.py` (fn mono, 760 LoC): Bootstrap has `mono_table` but
  only for top-level fns; cross-cuts struct_mono. See B2 above —
  recommend explicit Phase A stage (28.11.5).
- `presburger.py` (shape constraint solver, 353 LoC): Bootstrap has no
  equivalent. Cycle 1 A11 partially wired pytree validation; presburger
  is for tile/tensor shape constraints. **NOT a v0.1 blocker** — shape
  constraints today are checked at runtime via traps. Defer to v0.2.
- `autotune.py` (variant Cartesian product, 242 LoC): Bootstrap has no
  equivalent. `@autotune` parsed but variant generation Python-only.
  **NOT a v0.1 blocker** if `@autotune` is judged v0.2-only.

### D3. Stdlib parse parity

Verified that the bootstrap parses all 16 stdlib files at HEAD by running
`python -m pytest helixc/tests/test_cli.py -k stdlib` (passes 38/38).
The stdlib uses positional `match`, positional struct-lit, no `?`
operator, no `f"..."`, no `let-else`. **All stdlib parses fine today.**
Post-Phase-A ergonomic adoption is OPTIONAL — the stdlib won't break if
it doesn't migrate. Some files (`option.hx`, `result.hx`) WILL benefit
from `?` operator adoption post-A; recommend doing the rewrite during
Phase A's 28.13 ergonomics commit batch.

### D4. Test parity

At that snapshot, `python -m pytest helixc/tests --collect-only -q | tail -5`
reported 1430 collected tests. Spot-checked which tests use Python-only features
that wouldn't work post-Stage-29:

- `test_autodiff.py` / `test_autodiff_parity.py` / `test_autodiff_reverse.py`:
  test Python AD pass. These will need a kovc.hx-side equivalent
  after Stage 29. Effort: low (just adapt test harness to compile via
  bootstrap, then exec).
- `test_pytree.py`: same — Python-side pytree pass tests. Needs kovc.hx
  equivalent post-Phase A.
- `test_struct_mono.py` (860 LoC): same. Needs kovc.hx equivalent.
- `test_match.py` / `test_match_lower.py`: same.
- `test_panic.py` / `test_unsafe.py` / `test_deprecated.py` /
  `test_trace.py`: same.
- `test_typecheck.py` (111 tests): tests Python typecheck. Post-Stage-29,
  the bootstrap has its own typecheck (already exists in kovc.hx);
  parity check needed.

**Recommendation**: As part of Stage 29 prep, add a `tests/golden/`
directory of expected bytes for every test. Pre-Stage-29 build:
`python -m helixc.check -o tests/golden/<test_name>.elf <source>` for
every test case. Stage 29 then runs `kovc <source>` and byte-compares.
**This was already recommended as CC1 in the original research.** It's
still the right move. Add as Stage 28.20 (post-Phase-A but pre-Stage-29).

### D5. Newly-identified Python-only items

After deeper grep across `helixc/frontend/*.py` (25 files, 12,226 LoC),
no additional Python-only passes were found that aren't already in the
original research's table. The original research's listing of 18 Python
modules (including ast_hash, autodiff_cli, parser, lexer, diagnostics,
ast_nodes, hash_cons, totality, plus the 8 ported-needed) is complete.

---

## Cat E: Forward-looking AGI features

The original research deferred AGI-specific architectural primitives. The
project's stated goal is "AGI bootstrap"; v0.1 is the language layer that
will support AGI training. Items NOT in the original research that AGI
work needs:

### E1. GPU kernel autotuning policies — beyond current `@autotune`

**What's there**: `@autotune` Cartesian product of variant params, with
trap 27001 capping at 16 variants. Variant Cartesian-product walker is
Python-only.

**What's missing**: Online autotune — pick best variant at runtime based
on observed latencies. Profile-guided kernel selection. Cost-model-
driven variant pruning (don't compile all 16; compile the 3 most
promising based on a static cost model).

**Sketch**:
```rust
@autotune(BM in [16, 32, 64], BN in [16, 32, 64])
@autotune_policy(cost_model = "register_pressure")  // new
fn matmul<const BM: i32, const BN: i32>(a: tile<f32, ...>, b: tile<f32, ...>) -> ...
```

**Priority**: v0.3+. Online autotune needs a runtime profiler which
itself is post-v0.1.

### E2. Online learning primitives — incremental gradient accumulation

**What's there**: `grad(loss)(x)` returns gradient as one shot.

**What's missing**: `grad_accumulator` that streams gradients across
minibatches and applies updates. Checkpoint-resume from saved gradient
state. Sketch:

```rust
let mut acc: GradAccumulator<Model> = grad_acc_new(model);
for batch in batches {
    grad_acc_add(acc, grad(loss)(batch));
}
let updated = grad_acc_apply(acc, model, lr);
```

**Priority**: v0.2. AGI training NEEDS this. Build as stdlib on top of
v0.1 primitives.

### E3. Distributed-training primitives

**What's there**: Nothing — no collective ops, no sharding annotations.

**What's missing**: `all_reduce`, `scatter`, `gather`, `broadcast` as
either FFI calls (link NCCL/RCCL) or compiler-emitted (synthesize MPI-
shaped IR). Sharding annotations on tile types (`tile<f32, [N], HBM,
shard(2)>`).

**Sketch**:
```rust
@distributed(sharding = "data_parallel")
fn train_step(model: Model, batch: Tensor<f32>) -> Model {
    let g = grad(loss)(batch);
    let g_avg = all_reduce(g, op: "mean");
    apply_grad(model, g_avg, lr)
}
```

**Priority**: v0.3+. Distributed training is post-AGI-prototype. v0.1
single-node + v0.2 multi-GPU-via-FFI is enough for AGI bootstrap.

### E4. Quantization-aware training primitives

**What's there**: `fp8 / mxfp4 / nvfp4 / ternary` keywords reserved
(lexer.py:98). No codegen.

**What's missing**: `fake_quantize(x, bits: 8)` for QAT. Calibration
support (`@calibrate fn` that records activation statistics on a
calibration set). Mixed-precision policies (when to upcast for stability,
when to downcast for memory).

**Sketch**:
```rust
fn forward(x: D<f32>) -> D<f32> {
    let x_q = fake_quantize(x, bits: 8, scale: auto);  // train fwd / inv-fwd backprop
    let y = linear(x_q, w);
    y
}
```

**Priority**: v0.2. AGI training will use quantization eventually but
not for the first iteration. Spec the syntax for v0.1 to reserve the
keyword space; defer codegen.

### E5. Mixed-precision policies

**What's there**: `bf16 / f16 / f32 / f64` all parse and codegen.
Explicit `as` casts.

**What's missing**: Policy annotations — `@autocast(f16 -> f32 for accum)`
that say "every accumulation in this fn runs at f32 even if inputs are
f16". Today the user does this manually with explicit `as` casts.

**Sketch**:
```rust
@autocast(accum: f32, weights: f16, activations: f16)
fn forward(x: tile<f16>) -> tile<f16> {
    let intermediate = matmul(x, w);  // accum done in f32 implicitly
    intermediate as tile<f16>
}
```

**Priority**: v0.2. The current explicit-cast model is verbose but
correct. v0.2 sugar.

### E6. AGI-specific architectural primitives beyond WorkingMem<T>

**What's there**: `WorkingMem<T>` / `EpisodicMem<T>` / `SemanticMem<T>` /
`ProceduralMem<T>` exist as `TyMemTier` (cycle 11 confirmed). Strict
equality only; no consolidation/recall transitions.

**What's missing**:

- `Attention<Q, K, V>` first-class tile-based attention primitive. Today
  the user writes attention as multiple matmuls. The compiler could
  recognize the pattern + emit a fused kernel.
- `ReplayBuffer<Experience>` for off-policy RL — would close ~200 LoC
  of `agi_memory.hx` (292 lines).
- `Skill<I, O>` as a learnable function with a fingerprint —
  partially modeled by `TySkill` (typecheck.py) but no codegen.
- `Predictor<X, Y>` as a "next-step prediction" primitive for world
  models.

**Sketch**:
```rust
type Attention<const D: i32, const H: i32> = trait {
    fn forward(q: tile<f16, [B, D]>, k: tile<f16, [B, D]>, v: tile<f16, [B, D]>) -> tile<f16, [B, D]>;
}

@kernel
fn flash_attn_v2<const D: i32, const H: i32>(...) -> ... { ... }
impl Attention<D, H> for FlashAttnV2 { ... }
```

**Priority**: v0.2 for Attention (top-3 AGI primitive); v0.3+ for the
others.

### E7. Reflection ergonomics for self-modification

**What's there**: Stage 11 reflection (Quote/Splice/modify cells).
Verifier-gated.

**What's missing**: A high-level API. Today the user manipulates raw
cells. AGI self-modification needs `compile_at_runtime(source: String)
-> Result<Function, _>` so the AGI can write Helix code on-the-fly and
have it integrated into its compiled body.

**Sketch**:
```rust
let new_skill_src: String = agi_synthesize_skill(observation);
let new_skill: Result<Skill<I, O>, ParseError> = runtime_compile(new_skill_src);
match new_skill {
    Result::Ok(s) => skill_table_register(s),
    Result::Err(_) => log_failure(),
}
```

**Priority**: v0.2. Critical for AGI self-modification but the v0.1
verifier-gated cells are enough for a manual scaffold.

### E8. Forward-look prioritization summary

| Item | Phase A | Phase B | v0.2 | v0.3+ | Out |
|------|---------|---------|------|-------|-----|
| Online autotune | | | | X | |
| Online learning (grad accum) | | | X | | |
| Distributed training | | | | X | |
| Quantization (QAT) | | | X | | |
| Mixed-precision policies | | | X | | |
| Attention primitive | | | X | | |
| ReplayBuffer | | | X | | |
| Skill codegen | | | X | | |
| Runtime-compile (reflection ergonomics) | | | X | | |

**None of these are Phase A.** Phase A stays scoped to porting Python
to bootstrap parity + ergonomics. AGI-specific features land in v0.2
once the language itself is self-hosted.

---

## Cat F: Tooling

Tooling beyond the language itself. The original research
(`HELIX_REFERENCE.md` tooling appendix) listed: LSP server,
property-based testing, fuzzing, doc-comment generator, source maps.
Below: which are blocking user-onboarding for v0.1.

### F1. Debugger story

**What's there**: trap-id encoded into `eax` then `ud2` (SIGILL). User
sees `Trace/breakpoint trap` and has to grep the trap-id in source.

**What's blocking**:

- **No source-position mapping in traps.** Original research's H6
  recommended a `.dbg` companion file. Cycle 11 audit-A noted this is
  still missing; no DWARF.
- **No GDB / LLDB integration.** Strictly v0.3+.

**Verdict**: v0.1 ships with trap-id-only error reporting (current
state). Source position would be NICE but not BLOCKING for the
self-host story. **Defer to post-v0.1 polish.**

### F2. Editor support

**What's there**: Nothing. No syntax highlighting, no LSP, no
tree-sitter grammar.

**What's blocking**:

- **No syntax highlighter.** Users write Helix in plain text editors.
- **No code-completion.** Users grep stdlib manually.
- **No goto-definition.** Users grep manually.

**Verdict**: Tree-sitter grammar is ~1 month for a language-veteran;
LSP server is ~2-3 months. **Both v0.3+** unless an external
contributor lands them. The language ships before tools.

### F3. Package manager (`kovpkg`)

**What's there**: Path-based modules via `mod` + `use`. Stdlib is in-
tree. No third-party packages.

**What's blocking**: Nothing for v0.1. The single user (Kovostov) and
its AGI work uses in-tree stdlib + bespoke code. **Defer to v1.0+** per
APPROACH_A_PLAN.md.

### F4. Build system (`kovbuild` / `Kovfile.toml`)

**What's there**: Direct invocation `python -m helixc.check foo.hx`
or `kovc foo.hx`. No project-level build graph.

**What's blocking**: Nothing for v0.1. A single `.hx` file with
`use stdlib::option` works today.

**Verdict**: v0.2 for `kovbuild`. v0.3+ for `Kovfile.toml` syntax (cargo-
style). Not blocking.

### F5. Test runner (`kovtest`)

**What's there**: Python `pytest helixc/tests/` runs the compiler's own
tests. No user-facing test framework.

**What's blocking**: User can't write `#[test] fn ...` tests in Helix.

**Verdict**: v0.2. Even a minimal `kovtest <file>.hx` that scans for
`fn test_*` and runs each, exit code 0/1, would be enough.

### F6. Profiler (`kovprof`)

**What's there**: Nothing. `@trace` exists at language level (Stage 25)
but the trace buffer isn't wired (per original research CC6).

**What's blocking**: Performance tuning is blind. AGI training will
need this.

**Verdict**: v0.2 for trace-buffer wiring + a kovprof CLI that pretty-
prints traces. Blocked on bootstrap-side @trace runtime (Stage 28.18 in
original research's "long" plan).

### F7. Coverage tool (`kovcov`)

**What's there**: Nothing.

**What's blocking**: Cycle 1 audits surface "validation passes never
wired" — a coverage tool would have caught this years earlier (in fact,
any of the Phase A audit cycles would have).

**Verdict**: v0.3+ for kovcov as a CLI tool. **Bootstrap-side coverage
should leverage @trace once it's wired.**

### F8. Documentation generator

**What's there**: `///` doc-comments parsed but no extractor (per
`check.py:--doc` is a stub at line 371 that prints `extract_doc_comments(src)`
which works for source-level extraction but no HTML rendering).

**What's blocking**: Stdlib has no rendered docs.

**Verdict**: v0.2. Doc extraction works today; HTML rendering can be a
Python script reading the JSON output.

### F9. Summary

| Tool | Status | Blocking v0.1? |
|------|--------|----------------|
| Debugger (DWARF, source maps) | Trap-id only | No |
| LSP server | None | No |
| Tree-sitter grammar | None | No |
| Package manager | None | No |
| Build system | Direct CLI | No |
| Test runner | pytest only | No |
| Profiler | @trace stub | No |
| Coverage tool | None | No |
| Doc generator | Stub | No |

**None of the tooling is blocking v0.1.** The language ships before the
tools. Stage 28.13's `render_caret` port + Stage 28.18's @trace
runtime (deferred to post-Phase-A) are the only tooling-adjacent items
that need pre-29 attention.

---

## Suggested staging delta vs current plan

Concrete proposal for the master plan (`APPROACH_A_PLAN.md` and
`STAGE_28_9_PHASE_A_PRE_29.md`):

### Insert before Phase A

- **Stage 28.8.1**: Codegen determinism harden. 0.5 stage.
  - Replace `id(op):x` symbol generation with `(fn_idx, op_idx)` in
    `backend/x86_64.py` (9 sites).
  - Reset `match_lower._FRESH_COUNTER` per-program.
  - Audit `autodiff._DIFF_WARNINGS` for cross-program leakage.
  - Add `test_codegen_deterministic_bytes.py` regression suite.
  - **HARD BLOCKER** for Stage 29 byte-identical gate.

- **Stage 28.8.2**: Shared AST walker library. 1 stage.
  - Add `frontend/ast_walker.py` with `ASTVisitor` base class +
    `generic_visit` introspecting dataclass fields.
  - Refactor `panic_pass`, `unsafe_pass`, `deprecated_pass`,
    `grad_pass._expr_has_grad`, `struct_mono.visit_expr` to subclass
    `ASTVisitor`.
  - Net code delta: NEGATIVE (~120 LoC saved).
  - Closes ~20 audit findings from cycles 1–11.

### Re-order Phase A

Original | Recommended
---------|-----------
28.9 match_lower | 28.9 validation passes (panic/unsafe/deprecated/trace)
28.10 struct_mono | 28.10 match_lower
28.11 validation passes | 28.11 struct_mono
28.11.5 — | 28.11.5 monomorphize.py port (new)
28.12 pytree | 28.12 pytree
28.13 ergonomics | 28.13 ergonomics
— | 28.13.5 render_caret bootstrap port (new)

### Split Stage 28.13

The original Stage 28.13 was "ergonomics cluster: `?` / let-else / named
struct-lit / `f"..."` / render_caret". Recommend splitting:

- **Stage 28.13a**: `?` operator + let-else + named struct-lit fields.
  0.6 stage. Each is a parser change + desugaring rule.
- **Stage 28.13b** (was render_caret): Bootstrap-side render_caret. 0.5
  stage. Renamed to **Stage 28.13.5** above.
- **DROP `f"..."`** from Phase A; spec for v0.2.

### Post-Phase-A but pre-Stage-29

Add a new **Stage 28.20: tests/golden bytes**. ~0.3 stage. Pre-Stage-29
build golden ELFs for every test case so Stage 29's byte-identical
verification doesn't require the Python interpreter live (per original
research's CC1).

### Total revised Phase A scope

| Stage | What | Effort |
|-------|------|--------|
| 28.8.1 | Determinism harden | 0.5 |
| 28.8.2 | Shared AST walker library | 1.0 |
| 28.9 | Validation passes (panic/unsafe/deprecated/trace) port | 1.0 |
| 28.10 | Match_lower port | 1.0 |
| 28.11 | Struct_mono port | 1.5 |
| 28.11.5 | Monomorphize.py port | 0.5 |
| 28.12 | Pytree port | 1.0 |
| 28.13a | Ergonomics (?, let-else, named struct-lit) | 0.6 |
| 28.13.5 | Render_caret bootstrap port | 0.5 |
| 28.20 | Tests/golden bytes | 0.3 |

Total: **7.9 stage-equivalents**. Original plan was 5 stages
(28.9–28.13). Net delta: +2.9 stages (the determinism + walker + monomorphize
+ render_caret additions).

**Audit discipline per stage**: same as Stage 28.8 (multi-agent audit
cycles, strict-zero rule, 5 consecutive clean cycles before stage closure).

---

## Out-of-scope (will NOT add)

Explicitly NOT recommending for pre-Stage-29 inclusion:

- **Higher-kinded types** (M1 original) — typecheck rewrite, defer to v0.2.
- **Associated types in traits** (M2 original) — defer to v0.2.
- **Const generic arithmetic** (M3 original) — defer to v0.2.
- **Variance annotations** (M4 original) — defer to v0.2.
- **Generic Option/Result drop i32 specialization** (M5 original) — defer to
  v0.2; bootstrap stdlib evolves post-29.
- **Where-clause solver** (M6 original) — defer to v0.2.
- **Iterator protocol + for-in** (M7 original) — defer to v0.2.
- **Operator overloading via traits** (M8 original) — defer to v0.2.
- **`f"..."` string interpolation** (H3 original) — defer to v0.2; the
  bootstrap can use `str_concat` chains.
- **Generic AST walker as bootstrap-side library** (CC2 original) — the
  Python-side library lands as 28.8.2; bootstrap-side equivalent lands
  during 28.9/28.10 ports.
- **D<Logic<T>> fuzzy AND/OR/NOT codegen** (Phase B / Tier-3 moat) —
  defer to v0.2 pending external neuro-symbolic benchmark validation.
- **TyMemTier cost annotations into IR** (Phase B / Tier-3 moat) — defer
  to v0.2.
- **All of Category L1–L13** in the original research — borrow checker,
  algebraic effects, refinement types, async/await, GADTs, existentials,
  inline asm, soft-typed values, WebAssembly backend, bytecode VM, LLVM
  IR backend, tree-sitter, kovpkg.
- **All of Category E forward-looking AGI features in this doc** — online
  learning primitives, distributed training, QAT, mixed-precision
  policies, attention primitive, replay buffer, skill codegen, runtime-
  compile. Defer to v0.2+.
- **AGI-specific runtime libraries** (autotune-online, profiler, etc.) —
  v0.2+.
- **GUI / web / DB / cross-platform / embedded targets** — out of scope.
- **JIT / REPL / interpreted execution** — v0.3+ if at all.
- **GC / multiple inheritance / class hierarchies** — out of scope per
  APPROACH_A_PLAN.md.

The rationale for the "out of scope" list is **discipline**: every
feature we add pre-Stage-29 must be re-baselined at Stage 30. Adding 8
AGI features now means 8 more re-baseline targets. Phase A as
re-scoped above is already 7.9 stages; adding more would slip Stage 29
by months. **The discipline path is the recommended path.**

---

## Concluding notes

The research validates the original `docs/helix-pre-self-host-research.md`
strategic decision (Phase A only; defer Phase B and v0.2+) but
refines the *tactics* in three concrete ways:

1. **Two new pre-Phase-A stages** (28.8.1 determinism harden, 28.8.2
   shared walker library) are HARD BLOCKERS that the original missed.
   Stage 29 cannot byte-identical-verify without 28.8.1; Phase A
   inherits walker-drift bugs without 28.8.2.

2. **Phase A re-ordering** honors the actual dependency graph: validation
   passes before match_lower (validates walker lib on simple sites);
   struct_mono before pytree (pytree walks mono'd structs);
   monomorphize.py port added between struct_mono and pytree (shares
   `_ty_key`).

3. **Phase A drops `f"..."`** and adds `render_caret` + `tests/golden`.
   The net effect is roughly the same scope but better-targeted at the
   Stage 29 gate.

The fact that 11 audit cycles produced ~80+ findings — most of which
were caught and fixed — argues that the audit gate is doing its job.
The systemic patterns identified here are the bugs the audit gate's
strict-zero rule is *trying* to surface as classes-of-bugs rather than
instances. Stages 28.8.1 and 28.8.2 are the user-instructable response.

End of pre-Phase-A finalization research. ~8400 words.
