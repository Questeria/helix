# Stage 35 Restart 54 — Lane B (Compiler / Backend / CLI) Audit

**HEAD**: c4cb7a3 (Sync HANDOFF_FOR_CLAUDE + HELIX_REFERENCE after restart 53)
**Date**: 2026-05-16
**Mode**: Read-only audit (fixes happen in a separate sweep)

## Summary

Reviewed `helixc/check.py`, `helixc/backend/x86_64.py`, `helixc/backend/ptx.py`,
`helixc/frontend/autodiff_cli.py`, `helixc/ir/lower_ast.py`, all of
`helixc/ir/passes/*.py`, `helixc/frontend/parser.py` dispatcher tails
(`_parse_type`, `_parse_primary`, `_parse_named_type`, `_parse_pattern`),
`helixc/frontend/diagnostics.py`, the `--help` text of all four CLIs vs
their actual flag parsers, and the bootstrap-vs-Python parser metadata
alignment (restart 47 baseline). Restart 51 closed B1–B4 (autodiff_cli
unknown-flag, check.py codegen re-raise sites, const_fold sibling sweep);
restart 53 closed B1/B2 (x86_64 driver re-raise + tile-lowering comment).
All of those remain in place.

Found **2 findings**: 0 HIGH, 1 MEDIUM, 1 LOW.

The MEDIUM is a fresh `check.py --help` documentation drift — `-Wad` is a
parser-accepted, behaviour-honoured flag (controls the AD-warning policy)
but is not enumerated in the `-W<flag>` example line of `--help`, while
both backend banners (`backend/x86_64.py`, `backend/ptx.py`) DO enumerate
both `-Wad=warn|error` and `-Wdeprecated=warn|error`. Users reading
check.py's banner cannot discover the AD policy escape hatch.

The LOW is an `_lower_type` fallthrough sentinel `tir.TIRScalar("?")`
that violates the loud-fail discipline that restart 47 B1 installed for
`_resolve_monomorphized_struct_type`. Practical reachability is shielded
by upstream guards (typecheck unknown-name error, the function-typed-call
loud-fail at `lower_ast.py:2525,2536,2540`, and the lack of any way to
construct an fn-typed value in source), but a future TyNode subclass
(e.g. the AGI refinement / confidence / tier types already declared in
`typecheck.py`) would silently lower to TIRScalar("?") for ABI sizing
instead of forcing explicit dispatch — the exact pattern restart 47
fixed two layers up.

## Findings

### B1 (MEDIUM): `check.py --help` omits `-Wad` from the `-W<flag>` example line — drift with backend banners

- **File:function:line**:
  - `helixc/check.py:43-44` (the `--help` docstring's `-W<flag>` line:
    `Warning policy (e.g. -Wdeprecated, -Wdeprecated=error)`)
  - `helixc/check.py:220` (parser side: `_KNOWN_WARNING_NAMES = frozenset({"ad", "deprecated"})`)
  - `helixc/check.py:403,441` (behavioural side: `a.warnings.get("ad", "warn")` controls
    AD-warning policy and `-Wad=error` promotion)
  - `helixc/backend/x86_64.py:3999-4007` (the x86 banner lists
    `[-Wad=warn|error] [-Wdeprecated=warn|error]`)
  - `helixc/backend/ptx.py:806-813` (the PTX banner lists
    `[-Wad=warn|error] [-Wdeprecated=warn|error]`)
- **Bug family**: silent flag drift between parser, behaviour layer, and
  banner. The parser accepts `-Wad=warn|error`, the AD-warning drain reads
  the policy from `a.warnings["ad"]`, and `-Wad=error` actually promotes
  AD warnings to errors with rc=1 — but `check.py --help` does NOT mention
  `-Wad` in the example. Backend banners DO mention it. So a user reading
  `check.py --help` cannot discover the escape hatch even though it exists
  and is honoured.
- **Sibling sweep**: I diffed all four CLIs' help/banner output against
  their actual flag parsers.

  | CLI | Parses `-Wad` | Honours `-Wad=error` | Lists `-Wad` in banner |
  |---|---|---|---|
  | `helixc.check` | yes (`_KNOWN_WARNING_NAMES`) | yes (`check.py:403,441`) | **NO** (only `-Wdeprecated` shown) |
  | `helixc.backend.x86_64` | yes (`x86_64.py:4065`) | yes (`x86_64.py:4127-4133`) | yes |
  | `helixc.backend.ptx` | yes (`ptx.py:834-858`) | yes (`ptx.py:887,1130+`) | yes |
  | `helixc.frontend.autodiff_cli` | n/a (no `-W`) | n/a | n/a |

  All other long flags are mirrored consistently: `-O0..-O3, --no-opt,
  --stdlib, --no-stdlib, --strict, --no-color, --color, --hash, --hash-cons,
  -l, --help, -h` — all in both parser and banner for both backends and for
  check.py. Only `-Wad` is missing from check.py's banner.

  Verified `--help` output live at HEAD c4cb7a3:
  `python -m helixc.check --help | grep -- '-W'`
  → `-W<flag>              Warning policy (e.g. -Wdeprecated, -Wdeprecated=error)`
  (no `-Wad`).
- **Suggested fix**: extend `check.py:43-44` to
  `Warning policy (e.g. -Wad, -Wad=error, -Wdeprecated, -Wdeprecated=error)`,
  matching the backend banners.
- **Suggested regression name**:
  `test_stage35_restart54_check_help_lists_wad_flag` — assert
  `"-Wad"` is a substring of the `--help` output stream.
- **Severity**: MEDIUM (silent missing-doc on a behaviour-honoured flag;
  no miscompile, no wrong rc, but the documented-vs-actual flag drift
  is exactly the family restart 47/49 swept for the backend banners).

### B2 (LOW): `_lower_type` falls through to `tir.TIRScalar("?")` instead of loud-failing on unknown TyNode subclass

- **File:function:line**:
  - `helixc/ir/lower_ast.py:809-847` (`_lower_type` — the trailing
    `return tir.TIRScalar("?")` at line 847)
- **Bug family**: AST-visitor catch-all returning a sentinel instead of
  raising. This is the same family the restart-47 B1 fix narrowed for
  `_resolve_monomorphized_struct_type` (which the comment at lines
  647-656 cites as the precedent: "Promote to loud-fail so future
  additions force explicit dispatch here"). `_lower_type` covers
  `TyName, TyTuple, TyTensor, TyTile, TyArray, TyRef, TyPtr, TyGeneric`
  but not `TyFn` (declared at `ast_nodes.py:66-69` and produced by
  `parser.py:786-798` for `fn(T1, T2) -> R` type annotations). A user
  writing `fn h(cb: fn(i32) -> i32) -> i32 { 0 }` produces an
  `A.TyFn`-typed parameter; `_lower_type` silently returns
  `tir.TIRScalar("?")` for that parameter's TIR type. The same is
  true of any future TyNode subclass (the AGI refinement/confidence/
  tier types already declared in `typecheck.py:46,196,232` are
  candidates).
- **Sibling sweep**: I checked the three dispatchers in `lower_ast.py`:

  | Dispatcher | Last branch | Loud-fail on unknown? |
  |---|---|---|
  | `_lower_type` (809-847) | `return tir.TIRScalar("?")` | **NO** — sentinel fallthrough |
  | `_lower_stmt` (1222-1561) | `if ConstStmt: ... return` | Implicit None (no explicit error) but every stmt kind is enumerated |
  | `_lower_expr` (1562-3142) | `return None` (3142) | Caller does `... or const_int(0)` — same sentinel pattern but every expr kind is enumerated |
  | `_lower_dim` (849-858) | `return tir.DimDyn()` | Documented Stage-16 behaviour for non-static shape exprs; not a bug |

  Also checked `parser.py` dispatcher tails (`_parse_type:623`,
  `_parse_primary:1088`, `_parse_named_type:678`, `_parse_pattern:1374`)
  — all four end in either an explicit `raise ParseError` or an
  enumerated fallthrough that handles every kind. Parser is clean.

  The `_lower_type` site is the ONLY AST-visitor in `lower_ast.py`
  that returns a sentinel for a known but unhandled subclass (TyFn).
- **Practical reachability**:
  - `lower_ast.py:2521-2528` loud-fails a CALL whose callee is an
    fn-typed local: `function-typed calls are not supported by the
    Stage 31 backend`.
  - `lower_ast.py:2535-2539` loud-fails the const-fn-alias call path.
  - There is currently no source form to construct an fn-typed VALUE
    other than naming a function directly (which the `_const_fn_aliases`
    path captures separately, NOT via `_lower_type`).
  - So an fn-typed parameter that the user receives but never calls is
    the only path that reaches `_lower_type` with `A.TyFn` today — and
    that param's TIRType silently becomes `TIRScalar("?")` for ABI
    sizing in the function signature lowering at `_lower_fn_body`
    (line 891). Codegen sizing routines at `backend/x86_64.py:1097-1144`
    only match known scalar names (i32/i64/f32/f64/etc.) — unknown
    "?" silently falls past every branch.
  - This is upstream-shielded enough that I rate it LOW, not MEDIUM:
    there is no known user source program that hits it today. But the
    discipline mismatch is real and is the exact pattern restart 47
    B1 named: a future TyNode subclass would silently lower to "?"
    instead of forcing explicit dispatch.
- **Suggested fix**: replace `return tir.TIRScalar("?")` at line 847 with
  `raise NotImplementedError(f"unsupported TyNode subclass {type(ty).__name__} in IR lowering: {ty!r}")`.
  Add a TyFn case if function-typed parameters are intended to be
  supported (most likely they should lower to `TIRScalar("u64")` as a
  closure pointer placeholder, mirroring the TyPtr → u64 lowering at
  line 842).
- **Suggested regression name**:
  `test_stage35_restart54_lower_type_loud_fails_on_unknown_tynode` —
  construct a synthetic A.TyNode-but-not-one-of-known via a stub class
  and assert NotImplementedError is raised; OR
  `test_stage35_restart54_lower_type_handles_fn_type` — write
  `fn h(cb: fn(i32) -> i32) -> i32 { 0 }`, assert the lowered TIR
  param type is NOT `TIRScalar("?")`.
- **Severity**: LOW (discipline / future-proofing; no known reachable
  miscompile today, but the catch-all pattern is exactly the one
  restart 47 B1 explicitly named as a loud-fail target).

## Clean families swept

- **Silent-fallback `except Exception`** (in-scope files):
  `helixc/check.py:561,977,1020,1665,1848,1874,1895` — all guarded by an
  explicit `(NotImplementedError, AssertionError, KeyboardInterrupt,
  SystemExit, MemoryError): raise` clause above the catch-all.
  `helixc/check.py:1718,1752` — intentionally bare per documented
  `validate_kernel_tile_lowering` contract (NIE is the user-facing
  signal); restart 51 B2 + restart 53 B2 comments preserved.
  `helixc/check.py:587` is the finally-drain wrapper, not a primary
  handler. `helixc/backend/x86_64.py:4325,4349` — same bare-by-design
  pattern with the restart-53 B2 comment.
  `helixc/backend/x86_64.py:4401` — guarded (restart 53 B1).
  `helixc/backend/ptx.py:1009,1055` — both guarded (restart 48 B2).
  `helixc/frontend/autodiff_cli.py:63,145` — both guarded (restart 48 B3).
  `helixc/ir/lower_ast.py:660,3097` — narrowed to
  `(KeyError, AttributeError[, TypeError, ValueError])` (restart 47 B1
  + restart 49 B4).
  `helixc/ir/passes/const_fold.py:493,530,636` — all three preceded by
  the `FoldError: raise` + loud-fail re-raise (restart 51 B4 + earlier
  cycle 21 C20-R1). Clean.
- **Stale-artifact cleanup on bad invocation**:
  `helixc/backend/x86_64.py:_bad_invocation_cleanup_output` (lines
  3978-3991) clears the output path BEFORE the `sys.exit(2)` on
  unknown-flag, conflicting-stdlib-flags, input-path-as-flag,
  output-path-as-flag, and input==output. Every check.py error path
  via `_emit_env_error` or `parse_args` errors returns BEFORE the
  `if a.output: ... _atomic_write_bytes` block at line 1898. No
  partial output left on bad invocation. Clean.
- **Partial writes / atomic writes**:
  `helixc/check.py:_atomic_write_bytes` (lines 458-483) uses
  `tempfile.mkstemp + os.replace + BaseException cleanup`.
  `helixc/backend/x86_64.py:_atomic_write_output` (lines 4090-4117)
  same pattern.
  `helixc/examples/dashboard_server.py` (line 79-86) same pattern.
  `helixc/examples/run.py` (lines 92-112) same pattern (restart 46 B5).
  No new file-writers in this audit window. Clean.
- **Backend / flag mismatch**: diffed `python -m helixc.check --help`,
  `python -m helixc.backend.x86_64 --help`, `python -m helixc.backend.ptx --help`,
  `python -m helixc.frontend.autodiff_cli --help` against their actual
  parsers (`_KNOWN_LONG_FLAGS`, `allowed_flags`, `_opt_flag_set`,
  `_parity_passthrough_flags`).

  | Flag | check.py | x86_64 | ptx | autodiff_cli |
  |---|---|---|---|---|
  | `-O0`/`-O1`/`-O2`/`-O3` | parse+act | parse+act | parse+act | n/a |
  | `--no-opt` | parse+act (restart 48 B1) | parse+act | parse+act | n/a |
  | `-l <lib>`, `-l<lib>` | parse+act | parse+passthrough | parse+passthrough | n/a |
  | `--strict` | parse+act | parse+act | parse+act | n/a |
  | `--stdlib`/`--no-stdlib` | parse+act | parse+act | parse+act | n/a |
  | `--no-color`/`--color` | parse+act | parse+passthrough | parse+passthrough | n/a |
  | `--hash`/`--hash-cons` | parse+act | parse+passthrough | parse+passthrough | n/a |
  | `-Wad=warn|error` | parse+act, **not in banner** (see B1) | parse+act + in banner | parse+act + in banner | n/a |
  | `-Wdeprecated=warn|error` | parse+act + in banner | parse+act + in banner | parse+act + in banner | n/a |
  | `-h`/`--help` | parse+act | parse+act | parse+act | parse+act (restart 49 B2) |

  Output-/emit-only flags (`-o`, `--emit-*`, `--doc`, `--check-only`,
  `--emit-proof-obligations`) are check.py-only by design: backends
  always emit. Single check.py-side documented-but-unbannered flag is
  `-Wad` (finding B1). Otherwise clean.
- **Parser / typechecker / codegen silent fallbacks**: see B2 — the only
  finding. Parser dispatcher tails (`parser.py:_parse_type`,
  `_parse_primary`, `_parse_named_type`, `_parse_pattern`) all
  loud-fail or are exhaustive. `lower_ast.py:_lower_stmt` and
  `_lower_expr` dispatchers enumerate every kind (the `return None`
  at the end of `_lower_expr` is the documented fallthrough for
  no-op expressions; callers handle None). `_lower_dim` returning
  DimDyn() is Stage-16 documented. No new untouched silent-fallback
  AST-visitor sites apart from B2.
- **Bootstrap parser drift vs Python parser**: `helixc/bootstrap/parser.hx`
  (7647 lines) was Stage-33 aligned. `helixc/frontend/parser.py` has
  had no changes since the e7c05bf commit ("Add AGI scalar refinements
  to stdlib" — bumps refinement support but introduces NO new
  metadata/attribute kinds in the parser, which still routes
  attributes through the generic `_parse_attributes` open-set
  capture at `parser.py:254-327`). No new attribute kinds, no new
  TyNode kinds, no new ExprNode kinds since restart 47's alignment
  check. Clean.
- **Exit-code convention**: verified live with HEAD c4cb7a3 against
  `--help`-documented contract `0 = clean / 1 = compile error / 2 =
  bad invocation`.
  - `python -m helixc.check /tmp/probe_x.hx` → rc=0
  - `python -m helixc.check -Wmadeup=warn /tmp/probe_x.hx` → rc=2
    with `helixc: unknown warning name: madeup`
  - `python -m helixc.check -O5 /tmp/probe_x.hx` → rc=2 with
    `helixc: unknown opt level: -O5`
  - `python -m helixc.frontend.autodiff_cli /tmp/probe_x.hx main` (no
    params) → rc=1 with `error: function 'main' has no parameters`
    (this is a SOURCE error, not bad-invocation; rc=1 is correct
    per the restart-49 B1 convention).
  - `python -m helixc.frontend.autodiff_cli -O1 /tmp/probe_x.hx loss`
    → rc=2 with `error: autodiff_cli: unknown flag -O1` (restart 51
    B1, still in place).
  - The charter note about `rc=3 on internal error` is at variance
    with the documented `--help` contract (`0/1/2` only) and with the
    actual implementation (`check.py:573` sets `rc=1` for internal
    errors; backends `sys.exit(1)` for the same). The current
    convention is consistent across all four CLIs: internal errors
    map to `rc=1`. No drift. (If the charter's `rc=3` was the
    intended contract, it would be a wider refactor — not in scope
    for this audit.) Clean for the actually-implemented `0/1/2`
    convention.

---

LANE_B_TOTAL: 2 findings (H=0 M=1 L=1) | 7 clean families
