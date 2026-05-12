# Audit Stage 28.9 cycle 56 — Type design

**Scope.** Read-only at HEAD `5d58d3d` (cycle-55 fix-sweep:
`_fn_table_sig` extended from `{name}:{body_hash}` to
`{name}/{arity}/{sorted_attrs}/{body_hash}`; two except clauses
extended with `NotImplementedError`). Prior C1-C54 dispositions
not re-flagged.

**Criterion.** Pass = ZERO findings at confidence >=75%.

## Result: PASS (0 findings >=75%)

### HEAD delta since cycle 54

`5d58d3d` makes three invariant-strengthening edits to
`helixc/frontend/autodiff.py`:

- **C54-AD1 (`autodiff.py:153`).** `_fn_table_sig` now folds
  `",".join(sorted(fn.attrs))` into the per-entry sig. The
  dimension `_inline_user_calls:361` (`"pure" not in fn.attrs`)
  is now captured.
- **C54-AD2 (`autodiff.py:154`).** `len(fn.params)` (arity) is
  also folded in. Closes the de-Bruijn body-hash collision
  documented in the commit message — `fn g(x,y)=x` and
  `fn g(x)=x` now hash to distinct sig entries despite identical
  body hashes.
- **C54-AD3 (`autodiff.py:146,184`).** Both `structural_hash`
  call sites in this module now catch `NotImplementedError`
  alongside `(TypeError, ValueError, AttributeError)` — aligns
  with the cycle-35 `ast_hash._hash_into` loud-fail discipline
  (`ast_hash.py:496-501`).

### Dimension-coverage verification

Read of `_inline_user_calls` (`autodiff.py:309-406`) and its
helper `_is_inferably_pure` (`autodiff.py:214-306`) confirms
that the only `FnDecl` fields the inliner consults are:
`fn.attrs` (purity gate, line 361), `fn.params` (arity gate +
substitution map, lines 364/367), `fn.body` (inlined material,
line 370), and `fn.name` (cycle key in `visiting` set, line
377). `fn.return_ty`, `fn.generics`, `fn.where_clauses`,
`fn.is_pub`, `fn.is_extern`, `fn.extern_abi` are NOT read —
omitting them from the sig is sound. The new four-field tuple
(`name`, `arity`, `attrs`, `body_hash`) is a tight cover of
the inliner's actual input surface.

The cycle-55 regression tests
(`test_autodiff.py:724-789`) exercise each of the three new
dimensions plus the NIE sentinel path.

### Cross-frontend identity layers (secondary scan)

- `ast_hash.py` — `FnDecl` arm (lines 373-432) emits
  `name`, sorted `attrs`, `is_pub`, `is_extern`, `extern_abi`,
  `Param.ty` via `_ty_repr`, `Param.is_mut`, `return_ty`,
  generics (`name`+`kind`), where-clauses (constraint via
  `_expr_canon`). Strictly STRONGER than what
  `_fn_table_sig` needs (which is correct — the autodiff
  cache only cares about inliner-visible dimensions). No
  missing-dimension defect.
- `hash_cons.py::_ast_equal` (lines 163+) mirrors
  `_hash_into` arm-by-arm — `IntLit`/`FloatLit` compare
  `type_suffix`, `Name` compares `generics` via `_ty_equal`,
  trap-20001 disambiguator preserved. No new drift.
- `cse.py::_op_hash` (lines 76-91) covers `kind`,
  `operand_ids`, primitive attrs, complex-attr repr, and
  result-type repr. Same defect-class hardened in cycles
  18/21/22; current arms cover all `PURE_KINDS` entries.

### Stability

No prior-cycle findings re-surface. The cycle-55 delta is
strictly invariant-strengthening on the autodiff memoization
key — each new dimension matches a previously-uncovered axis
that `_inline_user_calls` reads.

## Notes (<75)

- **Delimiter-injection in `_fn_table_sig` (conf ~35).** The
  sig string format `{name}/{arity}/{attrs}/{body_hash}` joined
  per-entry with `|` and the attrs-part joined with `,` admits
  reserved-character injection from attr payloads. Parser
  produces attrs of the form `f"{attr_name}:{msg}"` where
  `msg` is an arbitrary string literal
  (`parser.py:288-291,378-392`) — e.g. `@deprecated("a/b|c")`
  yields attr `"deprecated:a/b|c"`. The `/`, `,`, and `|`
  characters all appear in valid attr payloads. Constructing a
  CLEAN string-level collision is hard because: (a) `body_hash`
  is a fixed-format 64-hex-char SHA-256 digest or the
  `<unhashable:NNNN>` sentinel — neither contains `/` or `|`;
  (b) `fn.name` is an ident (no separators); (c) `arity` is a
  base-10 integer; (d) sorted-join scrambles most attr
  permutations into different orderings. I was unable to
  construct a working collision pair within the parser's
  attr-construction grammar, but the fact that the
  delimiter-set overlaps the attr-payload alphabet is a latent
  hazard — a future @attribute syntax that admits non-string
  args, or a recipe-attr scheme that injects synthetic attrs
  with crafted separators, could produce collisions. Pragmatic
  fix would be to repr each attr (`repr` quotes embedded
  separators) or to use length-prefixing as `ast_hash._emit`
  already does. Below 75 because no current-grammar exploit
  exists.

- **`_fn_table_sig` ignores fn-set composition (conf ~30).**
  `_inline_user_calls` only inlines callees present in
  `fn_table` AND NOT in the local `visiting` set. The sig
  encodes each fn's contents but the sig of an empty fn_table
  is the empty string, same as a fn_table containing only fns
  whose attrs/body are all empty. Theoretical degenerate case;
  no plausible user input reaches it because real FnDecls have
  a name string.

- **`autodiff_reverse.differentiate_reverse` uses no cache
  (conf 60, observational).** `autodiff_reverse.py:54-76`
  passes `fn_table` to `_inline_user_calls` but does not
  memoize — every call re-walks. Symmetric with forward-mode's
  pre-cycle-53 state. Not a correctness defect, but if the
  forward-mode cache key is now considered sound a similar
  memo could be added here cheaply. Out-of-scope for cycle-56.

Files: `C:/Projects/Kovostov-Native/helixc/frontend/autodiff.py`,
`C:/Projects/Kovostov-Native/helixc/frontend/ast_hash.py`,
`C:/Projects/Kovostov-Native/helixc/frontend/hash_cons.py`,
`C:/Projects/Kovostov-Native/helixc/frontend/parser.py`,
`C:/Projects/Kovostov-Native/helixc/frontend/ast_nodes.py`,
`C:/Projects/Kovostov-Native/helixc/ir/passes/cse.py`,
`C:/Projects/Kovostov-Native/helixc/tests/test_autodiff.py`.
