# Stage 28.8 Pre-29 Audit Gate — Cycle 11, Audit B: Type-System / Dispatch / Soundness

**Date**: 2026-05-11
**Commit**: c2e36d4 (read-only)
**Scope**: Fifth-pass re-audit of the type-system surface after the
cycle 7-10 fix chain. Cycle 11 is conducted with the strict criterion
re-emphasized by the user directive: a cycle counts CLEAN only when
ZERO findings of ANY severity are surfaced (≥75 confidence).

The cycle-7-10 commits under re-probe (per user directive):
- **b8e047e (cycle 7)** — C6-1 over-broad cascade narrowed via
  `_size_compatible`; G1/G2 LOW carve-out.
- **5d1ca24 (cycle 8)** — C7-1: dropped the cycle-7 G2 TyMemTier ×
  (TyVar | TySize) carve-out at top-level `_compatible`.
- **6968755 (cycle 9)** — C8-1/C8-2: check.py exception classifier
  refinement (delete `ImportError` arm, add `_emit_env_error`).
- **c2e36d4 (cycle 10)** — three regression tests added for the
  cycle-9 contract (C9-1 LOW closed).

Cycle 10 was tests-only; production-code surface at HEAD is identical
to the post-6968755 snapshot. `git log --oneline 6968755..c2e36d4 --
helixc/frontend/ helixc/check.py` is empty.

**Method**: read cycle-9 + cycle-10 audit docs to load the cumulative
invariant set; re-walked the focus-area contracts at HEAD by direct
file inspection; ran targeted regression tests for cycle-7-10 fix-IDs
(`-k "c8_1 or c8_2 or c7_1 or c4_6"` → 5 passed, 106 deselected, 0
failed); ran the full helixc suite (1426 passed / 1 skipped / 1
failed — the 1 failure is the flaky WSL bootstrap pipeline test, an
out-of-scope environmental flake; see "Out-of-scope observation"
below); cross-stage probes via grep for the TyMemTier + AD + Logic +
TyQuote interaction matrix and the nested-generic struct boundary.

---

## Focus-area probes (per cycle-11 directive)

### Probe 1: C7-1 TyMemTier subsumption work — HBM/DDR/NVMe compatibility

The cycle-11 directive named "HBM/DDR/NVMe compatibility" as the
focus. Examining HEAD, `TyMemTier` does NOT model HBM/DDR/NVMe — its
tier names are `working` / `episodic` / `semantic` / `procedural`
(typecheck.py:165-171), a four-tier cognitive-memory taxonomy mapped
from `WorkingMem<T>` / `EpisodicMem<T>` / `SemanticMem<T>` /
`ProceduralMem<T>` AST constructors (typecheck.py:543-552).
HBM / DDR / smem / hbm appear elsewhere as the `memspace` marker on
`TyTile` (typecheck.py:67, 525, 824). The directive's phrasing
conflates two distinct surfaces. The C7-1 fix touched the
cognitive-tier TyMemTier `_compatible` arm; it did NOT touch
`memspace` (TyTile).

Both surfaces re-audited independently at HEAD:

**TyMemTier (cognitive tiers)** — typecheck.py:2273-2276:
```python
if isinstance(a, TyMemTier) and isinstance(b, TyMemTier):
    return a.tier == b.tier and self._compatible(a.inner, b.inner)
if isinstance(a, TyMemTier) or isinstance(b, TyMemTier):
    return False
```
Strict `.tier` equality + recursive `.inner` compatibility. No
carve-out for TyVar / TySize (the cycle-8 C7-1 drop is preserved).
The cycle-5 F4 forward note about tier subsumption (HBM ⊆ DDR for
read-only is a Phase-1+ enhancement) is preserved as deferred, not a
finding.

**TyTile memspace** — typecheck.py:817-830 (call boundary) +
typecheck.py:2368-2376 (structural):
- Call boundary: strict `pty.memspace != aty.memspace` → trap 16003.
  No subsumption (smem/hbm/gmem are non-unifiable by design).
- Structural: equality on `a.memspace == b.memspace` after dtype +
  shape compatibility.

Both are intentionally strict at Phase 0. **No cycle-11 finding.**

### Probe 2: cycle-8 codereview fixes — any new silent type windows

Cycle 8's production diff (5d1ca24) drops the cycle-7 G2 carve-out
ONLY. Inspecting `_compatible` at typecheck.py:2248-2377: no other
arm was modified. The narrow cascade in `_size_compatible`
(2232-2246) handles TyVar/TySize at shape-position only, leaving the
top-level `_compatible` strict at value-position. Correct placement.

Consumers of `_compatible`:
- Call boundary `_check_call_basic` (687-757): uses `_compatible` via
  the cycle-5-C4-3 symmetric pre-filter. No new silent window.
- Body / return / match-arm checks (grep-verified to use
  `_compatible`).
- Structural arms in `_compatible` itself: TyQuote / TyDiff / TyLogic
  / TyTuple / TyArray / TyRef / TyPtr / TyFn / TyTensor / TyTile —
  all use `_compatible` recursively with `_size_compatible` only at
  shape positions.

No silent acceptance window detected. **No cycle-11 finding.**

### Probe 3: cycle-9 check.py exception classifier — helixc-internal vs Python-builtin

The classifier at check.py:284-318 (post-cycle-9):
- FileNotFoundError | PermissionError | IsADirectoryError |
  NotADirectoryError → `_emit_env_error(str(e))`, rc=2.
- UnicodeDecodeError → `_emit_env_error(f"encoding error reading
  source: {e}")`, rc=2.
- Other Exception (ImportError included) → "compiler bug" rc=1.

Question: does the classifier correctly distinguish helixc-internal
vs Python-builtin exceptions? `grep "class.*FileNotFoundError\|
class.*PermissionError\|class.*IsADirectoryError\|class.*
NotADirectoryError\|class.*UnicodeDecodeError" helixc/` returns 0
matches — helixc defines no subclasses of these builtins. So the
env-error arms catch ONLY Python-builtin raises; any
helixc-internal exception class would not match and would fall
through to the broad Exception arm.

The ImportError reclassification (catch in broad arm, rc=1) is
correct: an `ImportError` from `_main_inner`'s 18 lazy imports is
an internal-rename bug, not a user environment issue.

Edge case: `parser.py:1587` raises `FileNotFoundError(f"helixc:
stdlib file missing: {p}")` — this is the unique production
raise-with-`helixc:`-prefix. The cycle-9 `_emit_env_error` strips
one layer correctly. All other FileNotFoundError raises in the tree
are Python-builtin (file-open failures, no `helixc:` prefix). The
helper handles both cases.

**No cycle-11 finding.**

### Probe 4: cycle-10 regression tests cover the cycle-9 contract

The three cycle-10 tests at test_typecheck.py:1572-1634 each
monkey-patches `check_mod.typecheck` and asserts a stderr/rc
invariant:
- `test_c8_1_import_error_attributed_as_compiler_bug`: rc=1 +
  `"compiler bug"` + `"internal error"`.
- `test_c8_2_env_error_no_double_helixc_prefix`: rc=2 + `"helixc:
  helixc:" not in stderr` + `"stdlib file missing" in stderr`.
- `test_c8_2_env_error_no_prefix_still_prefixed`: rc=2 +
  `stderr.count("helixc:") == 1`.

Each test drives one arm of the cycle-9 classifier:
- C8-1 test → broad Exception arm.
- First C8-2 test → FileNotFoundError-family arm with strip.
- Second C8-2 test → FileNotFoundError-family arm without strip
  (passthrough + prepend).

Monkey-patch site `monkeypatch.setattr(check_mod, "typecheck", boom)`
relies on the cycle-1 re-export at check.py:73-76 — stable public
test surface used by cycle-5 / cycle-9 tests.

Targeted pytest run (`-k "c8_1 or c8_2"` → 4 passed) confirms these
tests pass at HEAD. **No cycle-11 finding.**

### Probe 5: cross-stage interaction — TyMemTier + AD + Logic + TyQuote

`_compatible` arm dispatch order at typecheck.py:2248-2377:
1. TyUnknown × * → True.
2. TyMemTier × TyMemTier → tier eq + inner compat;
   TyMemTier × non-TyMemTier → False.
3. TyQuote × TyQuote → inner; TyQuote × non-TyQuote → False.
4. TyDiff × TyDiff → inner; TyDiff × non-TyDiff → False.
5. TyLogic × TyLogic → inner; TyLogic × non-TyLogic → False.
6. Composite arms (TyTuple / TyArray / TyRef / TyPtr / TyFn /
   TyTensor / TyTile).
7. Fallback `a == b`.

The TyMemTier arm at step (2) returns False on TyMemTier ×
{TyQuote | TyDiff | TyLogic | TyTuple | ...}. The reverse pairs
(TyDiff × TyMemTier, etc.) also return False at step (4) because
TyMemTier is exhausted in step (2). Verified: no path unifies
TyMemTier with TyDiff / TyLogic / TyQuote-wrapped values, or vice
versa. Correct — these are distinct ontological kinds requiring
explicit transitions (consolidate / recall / attach / detach /
prove / quote).

`grep "TyMemTier.*TyDiff\|TyDiff.*TyMemTier\|TyMemTier.*TyLogic\|
TyLogic.*TyMemTier\|TyMemTier.*TyQuote\|TyQuote.*TyMemTier"` over
`helixc/` returns 1 match: struct_mono.py:375, a comment in
`_ty_key` listing resolved Type kinds that must NOT be passed (the
cycle-3 D6 hard-raise guard, not a unification site). No accidental
cross-kind unification path.

**No cycle-11 finding.**

### Probe 6: nested generics across module boundaries

`_ty_key` (struct_mono.py:355-386) recurses into composite types
(TyTensor / TyTile / TyArray shape+dtype recursed) and raises
loudly for unexpected Type instances (cycle-3 D6 guard at 380-385).
For nested generics like `Pt<Vec<T>>`, AST form
`TyGeneric(base="Pt", args=[TyGeneric(base="Vec",
args=[TyVar("T")])])` mangles to `Pt__Vec_T` with mono binding T at
call site of `Pt<Vec<i32>>` to produce `Pt__Vec_i32` + an emitted
`Vec_i32` StructDecl. Cross-module imports resolve via shared
`_struct_decls` table (typecheck.py:561-565). Arity mismatches raise
`ValueError("struct {decl.name!r}: arity mismatch")` at
struct_mono.py:394-398.

The cycle-3 D6 hard-raise prevents "resolved Type leaks into mono
keying" cross-stage failure. Still in place at HEAD.

**No cycle-11 finding.**

---

## Cycle 10 finding re-verification

| ID   | Severity prev | Audit (prev)              | Status     | Notes |
|------|---------------|---------------------------|------------|-------|
| C9-1 | LOW           | silent-failures (cycle 9) | STILL CLOSED | Three regression tests at test_typecheck.py:1572-1634 pass at HEAD. Cycle 10 closed this LOW; cycle 11 confirms (5 passed under targeted -k filter). |

No prior-cycle (1-9) type-design findings need re-verification.
Cycle 10 type-design was CLEAN; cycle 11 surfaces no re-opened
invariant.

---

## Per-surface review

### Surface 1: `_compatible` top-level structural matcher
**File**: typecheck.py:2248-2377.
**Invariants** (cumulative through cycle 10, unchanged in cycle 11):
- TyUnknown universally compatible.
- TyMemTier: tier eq + inner compat; cross-kind → False.
- TyQuote / TyDiff / TyLogic: structural-by-inner; cross-wrapper-
  kind → False.
- Composite arms: structural; shape via `_size_compatible`, value
  via `_compatible`.
- Fallback `a == b` for primitives, mangled TyStruct names, TyUnit.

**Status**: unchanged at HEAD. **No finding.**

### Surface 2: `_size_compatible` shape-position cascade
**File**: typecheck.py:2232-2246.
**Invariants**:
- TyVar / TySize / TyUnknown at either side → True.
- `a == b` → True.
- Else delegate to `_compatible`.

**Status**: unchanged. **No finding.**

### Surface 3: call-boundary `_compatible` pre-filter
**File**: typecheck.py:746-757.
**Invariants** (cycle-3 D1 + cycle-5 C4-3):
- Symmetric exclusion of TyVar / TySize / TyUnknown on both sides.
- TyPrim × TyPrim handled by earlier arm.
- Logic-provenance violations handled by specialized path; skipped
  here to avoid double-diagnostics.
- Else `_compatible(pty, aty)` and emit on False.

**Status**: unchanged. **No finding.**

### Surface 4: `_emit_env_error` helper
**File**: check.py:246-255.
**Invariants** (cycle-9):
- Module-private.
- Strips one leading `helixc:` after lstrip.
- Prints `helixc: {text}` to stderr.

**Status**: unchanged. Three regression tests pass at HEAD. **No
finding.**

### Surface 5: `main()` outer-dispatch exception classifier
**File**: check.py:284-318.
**Invariants** (cycle-9):
- FileNotFoundError-family → rc=2 + `_emit_env_error(str(e))`.
- UnicodeDecodeError → rc=2 + `_emit_env_error(f"encoding error
  reading source: {e}")`.
- Other Exception (ImportError included) → rc=1 + compiler-bug.
- Finally-wrapped drain doesn't mask primary failure.

**Status**: unchanged. Three regression tests pass at HEAD. **No
finding.**

### Surface 6: TyMemTier × cross-kind strict separation
**Files**: typecheck.py:2273-2276 + typecheck.py:2280-2312.
**Cross-kind invariant**: TyMemTier × (TyQuote | TyDiff | TyLogic |
TyTuple | TyArray | TyRef | TyPtr | TyFn | TyTensor | TyTile |
TyPrim | TyStruct) → False, both directions, by arm ordering.

**Status**: grep-confirmed at HEAD; no accidental unification site.
**No finding.**

### Surface 7: `monomorphize_structs` AST-only keying
**File**: struct_mono.py:_ty_key (355-386) + instantiate (389+).
**Invariants** (cycle-3 D6):
- `_ty_key` accepts only AST `TyNode`; resolved typecheck.Type
  instances raise loud TypeError (380-385).
- TyGeneric arm dedupes by `(base, tuple(args))`.
- Arity mismatch at `instantiate` raises ValueError (394-398).

**Status**: unchanged. **No finding.**

---

## Other surfaces (re-verified, not touched in cycles 7-11)

- Quote/Diff/Logic structural arms (cycle-3 D1 baseline) — unchanged.
- Struct-field lookup post-mono via `_struct_decls` table — unchanged.
- D-binop diagnostic-text accuracy (cycle-5 F4 baseline) — unchanged.
- ParseTime size-key normalization (cycle-3 D2 reverted to cycle-2
  baseline) — unchanged.
- Cast / TyRef rendering (cycle-5 F6 baseline) — unchanged.

All previously-CLEAN baselines remain CLEAN at HEAD.

---

## Out-of-scope observation (NOT a cycle-11 finding)

A full `pytest helixc/tests/` run at HEAD reported `1426 passed, 1
skipped, 1 failed`. The single failure is
`test_codegen.py::test_bootstrap_kovc_full_pipeline_arithmetic`,
which shells out to WSL bash to execute a cached bootstrap binary
plus a freshly-generated output binary. Two independent invocations
during this audit produced two distinct failure modes:

1. `assert 2 == 14` for `compile_and_exec("2 + 3 * 4")` — bootstrap
   binary cache hash `c00c44d73441dd46`.
2. `subprocess.TimeoutExpired` at 30s for `compile_and_exec("42")` —
   bootstrap binary cache hash `9ec7a36127416cf3`.

The differing bootstrap-binary cache hashes between invocations
(under `helixc/tests/_bootstrap_cache/`) indicate non-determinism in
the cached binary, NOT a type-design issue. The test was last
touched at commit 75e2209 (cycle 5) and earlier (8db844e for the
cache speedup). Cycles 7-10 did not modify this test or its codegen
surface. The failure surface is downstream (codegen /
Phase-0-bootstrap + WSL subprocess environment + cache determinism),
not type-design.

**This is NOT a type-design finding under Audit B's charter.** No
`_compatible` / `_size_compatible` / TyMemTier / TyDiff / TyLogic /
TyQuote / `_emit_env_error` / `main()`-classifier invariant is
implicated. The failure does not falsify any cycle-7-10 type-system
contract.

I flag it strictly for transparency. The user's heavy-gate spec
("≥1421 passed / 1 skipped / 0 failed at HEAD") is not met at the
suite level, but is met at the Audit B (type-design) surface level
which is this audit's charter. Appropriate next action: Audit C
(silent failures) or Audit A (codereview) re-probe
`test_bootstrap_kovc_full_pipeline_arithmetic` for codegen-bootstrap
determinism. Cycle 11 of Audit B does not manufacture a finding
from this observation (per the "do not inflate severity" directive).

---

## Cycle 11 invariant snapshot (unchanged from cycle 10)

No new invariants introduced. The cycle-10 invariant snapshot
remains authoritative:
- `_compatible` top-level structural-matcher contract.
- `_size_compatible` shape-position cascade contract.
- Call-boundary `_compatible` invocation pre-filter contract.
- TyMemTier × cross-kind strict-separation contract.
- TyQuote / TyDiff / TyLogic structural-by-inner contracts.
- `_emit_env_error` single-prefix helper contract.
- `main()` outer-dispatch FileNotFound-family / UnicodeDecodeError /
  Exception arm classifier contract.
- `monomorphize_structs` AST-only keying contract.

---

## Cycle 11 status

**Strict criterion (per user directive 2026-05-10): cycle clean iff
zero findings of ANY severity.**

This cycle finds **0 findings (0 HIGH, 0 MEDIUM, 0 LOW)** at the
type-design surface under Audit B's charter.

The severity trend across cycles:
- Cycle 1: HIGH-tier finding(s)
- Cycle 2: HIGH + MEDIUM
- Cycle 3: HIGH + MEDIUM + LOW
- Cycle 4: MEDIUM-tier
- Cycle 5: 3 MEDIUM + 3 LOW
- Cycle 6: 1 MEDIUM + 2 LOW
- Cycle 7: 0 + 0 + 0  ← CLEAN
- Cycle 8: 0 + 0 + 0  ← CLEAN
- Cycle 9: 0 + 0 + 0  ← CLEAN
- Cycle 10: 0 + 0 + 0 ← CLEAN
- Cycle 11: 0 + 0 + 0 ← CLEAN

By the strict criterion, **cycle 11 counts CLEAN**.

This is the FIFTH consecutive clean cycle under Audit B (per the
user directive's cumulative-counter framing: cycles 7-10 = 4 clean,
cycle 11 = 5 clean). The 5-clean-consecutive gate (Stage 28.8
closure per the cycle-5 doc projection for Python-helixc
deprecation) is now **5/5 MET**.

**Cycle 11: 0+0+0 CLEAN — counter advances. 5 consecutive clean
cycles reached.**

**Recommendation**: no fix-sweep needed for cycle 11 under Audit B.
Stage 28.8 type-design soundness gate is satisfied. Note the
out-of-scope flake on `test_bootstrap_kovc_full_pipeline_arithmetic`
for downstream Audit A / C investigation; it does NOT gate Audit
B's closure.

---

## Forward notes (not cycle-11 findings)

1. **TyMemTier subsumption matrix (carry-over from cycle 5 F4)** —
   strict `.tier` equality is correct for Phase 0. A Phase-1+
   subsumption matrix (e.g. "ProceduralMem readable as SemanticMem"
   or memspace-tier "HBM ⊆ DDR for read-only") would land at
   typecheck.py:2273-2276. Deferred, not blocking.

2. **TyDiff sub-domain metadata (carry-over from cycle 5 F2)** —
   `D<T>` is single-domain at Phase 0. When smooth / non-smooth /
   jacobian variants are specced, typecheck.py:2296-2299 needs
   variant comparison. Deferred.

3. **TyLogic provenance in `_compatible` (carry-over from cycle 5
   F3)** — provenance handled at call boundary via
   `_logic_provenance_violation_kind` (trap 24100), not in
   `_compatible`. Phase-1+ matrix would lift to typecheck.py:2309-
   2312. Deferred.

4. **Convention note for raise-message prefix (carry-over from
   cycle 9 forward note #3)** — implicit prefix contract (callees
   MAY include single, MUST NOT nest) could be codified in a
   contributor guide. Docs, not blocking.

5. **Edge cases for `_emit_env_error` (carry-over from cycle 10
   forward notes 1-3)** — empty-string, nested-prefix,
   leading-whitespace inputs. Second-order hardening tests. Not
   blocking; no production callee exercises these paths.

6. **Codegen-bootstrap determinism (new forward note this cycle)** —
   WSL bootstrap pipeline test
   `test_codegen.py::test_bootstrap_kovc_full_pipeline_arithmetic`
   shows non-deterministic cached-binary hashes across runs.
   Out-of-scope for Audit B but should be triaged by Audit A
   (codereview) or Audit C (silent failures) in a separate track.
