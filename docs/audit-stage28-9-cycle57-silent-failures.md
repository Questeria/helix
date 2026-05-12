# Audit Stage 28.9 cycle 57 — Silent failures

**Scope.** Read-only HEAD `2f3dcbc`. Adversarial second pass against
the rotated focus (backend/, ir/passes/, frontend/totality.py +
effect_check). Prior C1–C56 not re-flagged.
**Criterion.** 0 findings at conf >=75%.

## Result: 1 finding at >=75% — FAIL

## Finding C57-1 — totality check silently skips fns inside `mod {}` blocks at `helixc check`

**Severity:** HIGH. **Confidence:** 88.
**Location:** `helixc/frontend/totality.py` lines 34-72 (in
`check_totality`); `helixc/check.py` lines 438-459 (pipeline order).

**Issue.** `check_totality` iterates only `prog.items` filtered for
`isinstance(item, A.FnDecl)`. It does NOT recurse into
`A.ModBlock.items` or `A.ImplBlock.methods`. In the backend driver
`helixc/backend/x86_64.py:3104,3107,3173` this is harmless because
`flatten_modules` and `flatten_impls` run *before* `check_totality`,
lifting every nested FnDecl to top level. In the `helixc check`
surface tool, however, only `flatten_impls(prog)` runs before
totality (`check.py:441-442`) — `flatten_modules` is **never**
invoked anywhere in `check.py`. A non-`@partial` recursive function
declared inside `mod m { ... }` therefore evades the totality
check on the user's primary iteration command and the user sees
`totality:  OK` while a real non-termination bug ships.

**Hidden errors.** This is the same item-walker defect class that
`deprecated_pass.py:113-152` was explicitly hardened against
(see the C57-5 / "match_lower C16-1" comment at lines 119-125 of
`deprecated_pass.py`, which calls out ModBlock/ImplBlock recursion
as a fix). Totality was the sibling that didn't get the symmetric
patch. Probe inputs that should fail but pass:
`mod m { fn loops() -> i32 { loops() } }`,
`mod outer { mod inner { fn diverges(n: i32) -> i32 { diverges(n) } } }`.
Both compile-clean under `helixc check`.

**Impact.** Trap 21001 contract ("non-`@partial` recursive fn without
strictly-decreasing arg") is silently bypassed for any module-scoped
fn at the `check` entrypoint. The backend `python -m helixc.backend.
x86_64` driver does flag it (mod_flattening runs first), so users
who only iterate via `helixc check` get a divergent, asymmetric
diagnostic — a CI that gates on `check` exit code rather than full
codegen accepts the buggy program. The cycle-56 audit explicitly
scanned `helixc/frontend/` and concluded "no NEW silent-failure
pattern at confidence >=75" — this miss matches the cycle-56 audit
note "secondary scan" depth limit; rotating to the
totality+effect_check focus surfaces it.

**Recommendation.** Mirror the `deprecated_pass.find_deprecation_
call_sites` pattern: introduce an inner `scan_items` helper in
`check_totality` that recurses into `A.ImplBlock.methods` and
`A.ModBlock.items` to collect every `FnDecl`, then build the same
direct-recursion check over the flattened set. Alternatively (and
simpler), wire `flatten_modules(prog)` into `check.py` before
totality at line ~447, matching the backend pipeline order at
`x86_64.py:3104`. Either fix re-enables trap 21001 for module-
scoped fns. Add a regression test under `tests/test_totality.py`:
non-`@partial` recursive fn nested in `mod m { ... }` must appear
in the `check_totality(parse(src))` failures list.

**Example (current — buggy):**
```python
# helixc/check.py (excerpt around line 440)
flatten_impls(prog)            # impl methods lifted, OK
# (no flatten_modules call)
fails = check_totality(prog)   # never sees `mod m { fn rec()... }`
```

**Example (corrected):**
```python
# helixc/check.py — add before totality
from .frontend.flatten_modules import flatten_modules
flatten_modules(prog)
flatten_impls(prog)
fails = check_totality(prog)   # now sees lifted m__rec(...)
```
Or fix the walker (preferred — independent of caller order):
```python
# helixc/frontend/totality.py — replace the top-of-check_totality loop
def _collect_fns(items, out):
    for it in items:
        if isinstance(it, A.FnDecl):
            out[it.name] = it
        elif isinstance(it, A.ImplBlock):
            for m in it.methods:
                if isinstance(m, A.FnDecl):
                    out[m.name] = m
        elif isinstance(it, A.ModBlock):
            _collect_fns(it.items, out)
fns: dict[str, A.FnDecl] = {}
_collect_fns(prog.items, fns)
```

## Notes (<75)

- `helixc/backend/ptx.py:331-332`: `_emit_op` falls through to
  `// TODO: {op.kind.value}` for unhandled tile-op kinds with no
  raise. Documented as STUB at lines 12-18; only reachable via
  `helixc check --emit-ptx` or `python -m helixc.backend.ptx`,
  and the `check.py` wrapper at lines 750-755 catches `Exception`
  with a clean diagnostic. The silent-comment path produces
  ptxas-rejected output rather than wrong codegen, so the failure
  is "loud at link time, quiet at emit time" — partial loudness.
  Would prefer `raise NotImplementedError(f"PTX: unhandled
  {op.kind.name} at {op.span}")` so the user sees the gap at
  emit-time with a stack trace. Confidence ~65.
- `helixc/ir/passes/cse.py:90` `_op_hash` falls through to
  `result_ty_key = None` when `op.results` is empty (e.g. RETURN
  filtered above by `PURE_KINDS`). Currently dead because PURE_KINDS
  excludes every result-less op kind, but a future PURE_KINDS
  addition of a side-effect-free result-less op would silently
  collide on the `(kind, operands, attrs, attrs_complex, None)`
  key across distinct functions' blocks. Confidence ~50.
- `helixc/ir/passes/fdce.py:30-31`: `if entry_fn not in module.
  functions: return 0` silently no-ops without any diagnostic. The
  docstring (line 17-18) explicitly calls this out as intentional
  ("we don't want to silently empty the module"), but the caller
  in `x86_64.py:3204` doesn't distinguish "0 dead fns" from "skip
  entirely because main was renamed". A user whose `entry_fn` was
  re-mangled by an extension pass would silently get the
  unmonomorphized full module shipped — undocumented at the call
  site. Confidence ~55.
- `helixc/ir/passes/effect_check.py:299-302`: `<indirect>` callee
  → `closure[n].add("unknown")` with no diagnostic. A non-pure fn
  taking a fn-pointer arg and calling it gets "unknown" in its
  closure; the user must declare an effect they can't name. Cycle-
  16 audit-C C1 partially addressed `<indirect-ffi>` but the
  generic indirect path still surfaces only through the
  declared-vs-actual mismatch — never as "indirect call requires
  effect annotation". Confidence ~50.
- `helixc/frontend/totality.py:111-128`: `_arg_strictly_decreases`
  returns False when `param_idx >= len(call.args)`. A recursive
  call with fewer args than the fn signature (e.g. via a
  default-value future feature) silently fails the decrease check
  for that param but doesn't distinguish "couldn't prove" from
  "syntactically impossible". Latent — Helix has no defaults
  today. Confidence ~35.
