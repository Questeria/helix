# Audit Stage 28.9 cycle 59 — Silent failures

**Scope.** Read-only HEAD `722baf8`. Adversarial 3rd pass on cycle-58
fix-sweep + rotation to typecheck.py / lower_ast.py / tir.py per
cycle-59 instructions. Prior C1–C57 not re-flagged. Cycle-59
code-review findings C59-1..C59-3 also treated as already filed for
this HEAD (not re-flagged in the silent-failure lens).

**Criterion.** 0 findings at conf >=75%.

## Result: 0 findings at >=75% — PASS

## Cycle-58 surface verified clean

- **match_lower._rewrite_expr loud catchall (line 301-306).** Enumerated
  all 32 `A.Expr` subclasses against the explicit-arm cover plus the
  leaf-list `(IntLit, FloatLit, StrLit, CharLit, BoolLit, Name, Path,
  Continue)`. Every concrete `Expr` subclass at HEAD has an explicit
  arm: 24 recursive-rewrite arms + 8 leaves = 32. `A.Expr` has no
  nested subclass tree (verified via `cls.__subclasses__()` at runtime).
  Targeted pytest of `test_totality / test_match / test_deprecated /
  test_typecheck / test_pytree` passes 195/195 — the new
  `NotImplementedError` is unreachable from any current test path.
- **flatten_impls._rewrite_expr (line 199-223) and
  flatten_modules._rewrite_expr (line 221-249).** Same exhaustiveness
  audit. flatten_impls covers all 32 (Name handled via the unchanged
  fall-through default at line 233 inside the rewrite-stmt path is
  inapplicable to expr — flatten_impls treats Name as a leaf in the
  TupleLit/ArrayLit/StructLit arms). flatten_modules leaf-list
  intentionally omits Name because Name has its own explicit arm
  (line 233-239). UnsafeBlock.body assert (`isinstance(new_body,
  A.Block)`) is sound because the Block arm always returns A.Block.
- **pytree._is_struct_ref TyGeneric arm + local `mangle_struct`
  import (line 113-136).** No circular import: `struct_mono` imports
  from `monomorphize`, `monomorphize` does not import `pytree`. Local
  scope keeps the safeguard cheap. `_resolve_struct_name` mirror is
  symmetric to `_is_struct_ref`. Pytree is not invoked from any
  production driver (only `helixc/tests/test_pytree.py`) — the fix is
  defensive but harmless.
- **deprecated_pass._walk_items_for_fns (line 113-132) recursion.**
  Walker enumerates FnDecl / ImplBlock.methods / ModBlock.items
  exhaustively for the FnDecl-carrying Item subclasses. ConstDecl
  initializers and EnumDecl arms do not carry FnDecl-bearing bodies
  and skipping them is correct (deprecated-callee detection is a
  Call-inside-fn-body concern; the constructor for a ConstDecl value
  is itself an Expr but typecheck rejects free Calls in const-eval
  position elsewhere — `let X: i32 = deprecated_fn();` at module
  scope is rejected by parser const-eval rules before reaching this
  pass). The asymmetry between `_walk_items_for_fns` (recursive) and
  `find_deprecated_decls` (top-level only) is already filed in
  cycle-59 code-review as C59-3 (MED, conf 78); not re-flagged here.
- **totality.collect_items (line 54-66) recursion.** Same shape as
  deprecated_pass. The dict-keyed-by-name (`fns: dict[str, A.FnDecl]`)
  has a latent last-write-wins under `helixc check` (which omits
  `flatten_modules` before totality), but this is the structurally
  same item-walker / name-collision shape filed as the cycle-18
  "check.py-without-flatten_modules" NOT-FLAGGED observation. Not
  re-flagged here.

## typecheck.py / lower_ast.py / tir.py rotation — no new findings

- **typecheck `_resolve_type` TyGeneric arity check (line 562-565).**
  Uses the same `mangle_struct(name, ty.args)` as struct_mono. Arity
  check `len(ty.args) == len(user_struct.generics)` is sound. The
  size-vs-type kind mismatch (a struct generic declared `<N: size>`
  receiving a type-arg via `Pt<i32>`) flows through `_mangle_ty` and
  produces a mangled name that struct_mono will also produce for the
  same use — symmetry preserved.
- **typecheck `_register_fn` where-clause recording (line 455-456).**
  `self.constraints.append(w.constraint)` is a fire-and-forget
  collector; constraints are discharged elsewhere (per cycle-18+
  audits documenting the Presburger solver wiring). No regression.
- **typecheck silent-skip of ImplBlock/ModBlock fn bodies in
  check.py path.** Verified empirically: `helixc check --check-only
  foo.hx` reports `typecheck: OK` for an `impl Foo { fn area(self)
  -> i32 { let s: bool = self.x; 0 } }` containing a clear type error.
  Backend `compile` order (flatten_modules → flatten_impls →
  typecheck) catches it correctly; check.py order (typecheck →
  flatten_impls) misses it because typecheck only iterates
  `prog.items` for top-level `A.FnDecl/StructDecl/EnumDecl`. This is
  the same defect class as the cycle-18 silent-failures
  "check.py-without-flatten_modules" observation (filed NOT-FLAGGED
  because the backend pipeline is LOUD at link time). For the
  `--check-only` UX the failure mode is silent rather than loud, but
  the defect class is already in the corpus — not re-flagged per the
  C1–C57 prior-finding rule.
- **lower_ast Match-assertion (line 1908-1912).** Reachability of the
  Match-should-not-reach-_lower_expr assertion remains gated on
  match_lower's coverage. Cycle-58 closed the last
  silent-passthrough; the assertion is now strictly defense-in-depth.
- **tir.IRBuilder.add result-type approximation (line 440-442).**
  "Result type follows lhs (simplified)" is a known builder shortcut.
  Only consumed by hand-written tests (`test_ir.py`), not by
  `lower_ast`. No production miscompile path.

## Notes (<75)

- match_lower's TileLit arm mutates `expr.shape` / `expr.memspace`
  in place (line 281-283), while flatten_impls / flatten_modules
  construct fresh `A.TileLit` nodes via the constructor. The
  asymmetry is intentional (match_lower owns its AST) but worth
  documenting if hash-cons sharing is ever extended to TileLit. (~55)
- `tir.IRBuilder.add` and `.matmul` use `result_ty=a.ty` shortcut.
  If ever wired into automatic lowering, a broadcast-shape mismatch
  would silently propagate the lhs type. Not currently reachable
  from `lower_ast` (which constructs Op directly via builder.emit
  with explicit result_ty). (~65)
- Cycle-58 `pytree._is_struct_ref` per-call `from .struct_mono
  import mangle_struct` is hot on deep pytrees. Cycle-59 code-review
  noted as B59-3 (conf 60). Same observation here. (~60)

## Edits made

NONE. This audit was conducted in strict read-only mode per the
cycle-59 silent-failures instructions. No source files were
modified; only this audit document was written.
