VERDICT: 1 HIGH, 2 MEDIUM, 3 LOW, 2 OBS

# Stage 38 Inc 3 silent-failure audit

Surface: commits `86c2ce4..b427f4f` on `main` (HEAD `b427f4f`).
Files touched: `helixc/frontend/typecheck.py`, `helixc/frontend/autodiff.py`,
`helixc/ir/lower_ast.py`, `helixc/examples/dogfood_11_spatial_frames.hx`,
`helixc/tests/test_stage38_frames.py`.

Stage 38 adds `TyFrame` (WorldFrame/RobotFrame/CameraFrame), six
constructors/eliminators (`into_*`/`from_*`), six cross-frame transforms,
and registers all twelve names in `_BUILTIN_NAMES`, `AD_KNOWN_PURE_CALLS`,
and the lower-AST identity arm.

## Audit methodology executed

1. Captured full surface diff (662 lines) and read every chunk.
2. Cross-referenced live state at typecheck.py:3177-3235 (frame
   builtin dispatch arms), lower_ast.py:1986-2002 (identity-lowering
   arm), autodiff.py:84-94 (pure-set registration).
3. End-to-end probes (parse → typecheck → lower) against
   `python -c` harness for each suspected silent path.
4. Compared against the analogous Stage 37 tier-ops surface to
   distinguish newly-introduced regressions from inherited patterns.

## Findings

### F1 [HIGH conf 95] Wrong-arity calls to any of the 12 new builtins typecheck silently

**Citation**: `helixc/frontend/typecheck.py:3188`, `3196`, `3223`
(every dispatch arm gates on `bn in <dict> and len(arg_tys) == 1`),
and the dispatch-chain catchall at `helixc/frontend/typecheck.py:3302`
(`return TyUnknown(hint="call")`).

**Reasoning**: All three new dispatch arms use the pattern
`if bn in _frame_intro and len(arg_tys) == 1`. When the name matches
but the arity does not (0 args, 2 args, ...), control falls through
every subsequent arm, then past the user-function lookup at line 3283
(the name is not in `self.functions`), and finally hits
`return TyUnknown(hint="call")` at line 3302 — **no diagnostic emitted**.

Live probe:
```
into_world()              -> typecheck errors: []
into_world(1, 2)          -> typecheck errors: []
world_to_robot()          -> typecheck errors: []
world_to_robot(1, 2)      -> typecheck errors: []
from_world()              -> typecheck errors: []
```

Downstream, IR lowering raises `NotImplementedError: unknown function
'into_world' in IR lowering at L:C; run typecheck first` — a compiler-
internal exception whose message asserts typecheck has been skipped,
even though it ran and certified the program valid. Users see a
confusing internal traceback instead of a "wrong arg count" diagnostic.

The same shape exists in Stage 37 for tier ops, but Stage 38 widens
the silent surface to **12 additional names** (the 6 frame intro/elim
+ 6 transforms). Every one is independently exposed.

**Remediation**: gate on `bn in <dict>` first, then check arity
separately and emit a "expected 1 arg, got N" `TypeError_` on
mismatch. Same pattern as `learn_to` at typecheck.py:3256-3264.

### F2 [MEDIUM conf 90] AD-pure registration of frame builtins does not actually make `grad()` over them succeed

**Citation**: `helixc/frontend/autodiff.py:84-94`, dispatch in
`_diff_call_chain_rule` at `autodiff.py:1086-1252`, catchall raise at
`autodiff.py:1052-1056`.

**Reasoning**: The diff registers all 12 frame names in
`AD_KNOWN_PURE_CALLS` with the comment "preemptively AD-pure". But
`AD_KNOWN_PURE_CALLS` governs ONLY `_is_ad_erasable_expr`
(autodiff.py:664) — i.e., let-erasability when the value is unused.
It does NOT add a chain rule. The forward-mode AD catchall at
autodiff.py:1052-1056 raises `NotImplementedError: forward-mode AD
does not support opaque call 'into_world'; add a chain rule or
inline a differentiable helper` the moment any frame call is
actually used in a differentiated expression.

Live probe (a `@pure` function using `into_world`/`from_world`
attempted as `differentiate(fn.body, 'x')`):
```
NotImplementedError: forward-mode AD does not support opaque call
'into_world'; add a chain rule or inline a differentiable helper
```

The comment "Added preemptively so the Stage 37 closure gate-1 LOW
finding (S37-CLEAN1-001) doesn't recur for frames" implies the
registration suffices to flow gradients through the wrapper. It does
not. A user who reads the autodiff.py annotation and tries
`grad(use_frame)(2.0)` hits a hard raise that reads as an internal
limitation rather than a missing chain rule for an identity wrapper.

The same gap exists for the tier ops at autodiff.py:81-83 — but
Stage 38 doubles down on a misleading invariant without verifying it
end-to-end. The user's prompt asks "could any of them get
differentiated and silently produce a 0 grad?" The answer is NO
(they raise loudly), so this is MEDIUM not HIGH — but the
documentation/intent mismatch is a real silent failure of the
*claim*, even if not of the runtime.

**Remediation options**: (a) add an identity chain rule arm for the
12 frame names in `_diff_call_chain_rule` (literally
`return _diff(call.args[0], var)`); (b) revise the in-source
comments to scope the registration to "let-erasability only — these
calls still fail AD at use sites until a chain rule lands"; (c) the
arena-side note at line 105-111 (parent_*_at) is a cleaner model —
adopt that phrasing.

### F3 [MEDIUM conf 85] User functions named `into_world` / `world_to_robot` / etc. are silently shadowed by builtins

**Citation**: `helixc/frontend/typecheck.py:3188` and the surrounding
dispatch arms run BEFORE the user-function lookup at line 3283.

**Reasoning**: Builtin name resolution wins over user-function
resolution. There is no reserved-name check at function declaration.
Live probe:
```
fn into_world(x: i32) -> i32 { x + 1 }
fn main() -> i32 { into_world(41) }
```
Typecheck reports `main: body type WorldFrame<i32> does not match
return type i32` — i.e., the call `into_world(41)` was resolved to
the builtin, not the user fn, with no diagnostic that the user's
declaration was shadowed.

This is a pre-existing project-wide pattern (true for every Stage 36
/ 37 builtin), but Stage 38 newly reserves 12 plausible names
without diagnosing collisions. `world_to_robot` and `camera_to_world`
are particularly likely to collide with real user code — robotics
codebases use exactly those phrases as method names.

**Remediation**: at `fn` declaration time, if `name in
self._BUILTIN_NAMES`, emit a `TypeError_(... shadows compiler
builtin)` so users don't write a dead function.

### F4 [LOW conf 85] Identity-lowering arm drops `args[1..]` if a wrong-arity call slips past typecheck

**Citation**: `helixc/ir/lower_ast.py:1986-2002`.

**Reasoning**: The Stage 38 additions extend the identity-lowering
arm that returns `self._lower_expr(expr.args[0])`. The guard
`len(expr.args) == 1` protects the well-typed case. Combined with
F1, a 2-arg call like `into_world(payload, side_effect_with_arena())`
typechecks silently and then either (a) raises at lowering as
"unknown function" (current behaviour — `into_world` falls through
because arity guard fails), or (b) if F1 is fixed in isolation but
this guard ever loosens, would silently drop `args[1]`. The arm has
no `else: raise` defense.

This is a latent risk, not an active bug — confidence on
"exploitable today" is low. Listed as LOW because the order-of-fix
matters: any future change that swaps the identity-lowering arm
order or relaxes the `== 1` predicate would silently lose effects.

**Remediation**: add a defensive `assert len(expr.args) == 1, f"{name}
arity guard violated; typecheck should have rejected"` inside the
identity arm, or convert the gate to a strict equality with an
explicit fallthrough raise.

### F5 [LOW conf 70] `from_*` eliminator returns the inner Type on mismatch instead of `TyUnknown`

**Citation**: `helixc/frontend/typecheck.py:3201-3207`.

**Reasoning**: When `from_world(x)` is called with a non-WorldFrame
argument, the typecheck emits a diagnostic and then returns
`TyUnknown(hint=bn)`. Contrast with the parallel `consolidate` arm
at typecheck.py:3240-3245 and `recall` at 3250-3255, which return
`arg_tys[0]` (the actual argument's type) on mismatch. The result is
that downstream type inference sees a fresh `TyUnknown("from_world")`
which can mask follow-on type errors (e.g. a `let x: i32 =
from_world(robot_value)` would record two errors instead of one
useful "robot supplied where world expected" plus the genuine
mismatch on the let).

Live probe shows the diagnostic chain still produces both errors,
but the second error description ("declared X but value is i32" vs.
"declared X but value is unknown") shifts toward less actionable
text. Lower-severity because the user still sees A useful error;
just not the most useful one.

**Remediation**: return `arg_tys[0].inner` on `TyFrame` mismatch
(at least the inner type is recoverable), and `TyUnknown` only when
the input is not a TyFrame at all. Mirrors the tier-ops error-path
pragma.

### F6 [LOW conf 60] Cross-frame transform error message omits the inner type T

**Citation**: `helixc/frontend/typecheck.py:3229-3234`.

**Reasoning**: The error reads `world_to_robot() requires
WorldFrame<T>, got RobotFrame<i32>`. The `<T>` is a placeholder; the
diagnostic does not tell the user the actual inner type their
RobotFrame held, which matters when debugging deeper inference
chains (e.g., did `into_robot(some_call())` infer the inner type
they intended?). Compare with the consolidated `_fmt` output for
`TyFrame` at typecheck.py:6695-6699 — it CAN render `RobotFrame<i32>`
fully. The error template just doesn't surface the source's inner
type because the message says "requires WorldFrame<T>" (literal `T`
placeholder).

This is a UX nit, not a silent failure of correctness — but a user
who sees "requires WorldFrame<T>" might think the issue is generic
type inference, when in fact the inner type already resolved fine.

**Remediation**: include the actual inner type in the expected text,
e.g. `requires WorldFrame<{self._fmt(arg_tys[0].inner) if
isinstance(arg_tys[0], TyFrame) else 'T'}>`. Cheap, low-risk.

## OUT OF SCOPE — observations (no severity)

- **O1**. The dogfood `dogfood_11_spatial_frames.hx` claims to be
  "collapse-resistant: each observation must round-trip exactly".
  But the entire `cycle_through_frames` body lowers to literally
  `return(v0)` (verified by IR dump). The only frame-stack behaviour
  the runtime witness validates is "the type-checker accepted the
  let-binding annotations" — there is no runtime check that the
  cross-frame transforms preserved any structure beyond what
  identity-lowering already guarantees. The 42-witness fires equally
  if all 6 transforms were aliased to the same identity function.
  This isn't a silent failure of audited code; it's a dogfood-design
  observation: the runtime witness is weaker than the prose claim.
  Phase-1+, when transforms gain real matrix math, the witness will
  bite. For Phase-0 the typecheck pass IS the witness.

- **O2**. The three dispatch arms at typecheck.py:3183-3235 rebuild
  the `_frame_intro` / `_frame_elim` / `_frame_transforms` dicts on
  every Call-expression typecheck visit. Negligible perf cost (3
  small dicts) but easy to hoist to class-level constants. Stylistic.

## Summary

One HIGH (arity-mismatch silent acceptance — 12 newly-exposed names),
two MEDIUMs (AD claim/reality mismatch, builtin shadowing of user
fns), three LOWs (identity-lowering defense, mismatch-path return
type, error-message inner-T omission), two observations. The HIGH
finding is a direct integration regression — every new builtin
needs arity validation before the dispatch dict lookup.
