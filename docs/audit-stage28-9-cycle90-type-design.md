# Audit Stage 28.9 cycle 90 — Type design

Scope: HEAD 94f7427c22250b73b42af491ded15e353332444f

Mode: STRICT READ-ONLY. Read/Grep/Glob/Bash only. ONE Write (this doc). NO Edit.

## Files audited

- `helixc/ir/passes/fdce.py` — function-level DCE result-type invariants
- `helixc/frontend/grad_pass.py` — `@grad` walker totality over `Expr` subclasses
- `helixc/backend/elf_dyn.py` — static-ELF type-level layout invariants
  (audited as the closest extant match for the requested `backend/elf.py`,
  which does not exist at HEAD; the backend directory contains only
  `__init__.py`, `elf_dyn.py`, `ptx.py`, `x86_64.py`)

## Method

Per-file inspection of type annotations, dataclass field shapes, walker
dispatch totality versus the current `helixc/frontend/ast_nodes.py`
`Expr` subclass set, and arithmetic / size invariants enforced by
`assert` at construction time. Deferred-known findings from C1–C88
(notably bespoke walker drift risk vs. ASTVisitor — Stage 28.8.2
documented design choice — and Phase-0 W^X-relaxed PT_LOAD posture in
`elf_dyn`) intentionally NOT re-flagged per scope rules.

## Findings

### fdce.py

- `fdce_module(module, entry_fn="main") -> int` is fully typed; the
  return value is the count of dropped function names, computed as
  `len(dead)`. No type drift.
- Call-graph dict typed `dict[str, set[str]]`. Edge collection coerces
  every potential target through `isinstance(target, str)` /
  `isinstance(vfn, str)` / `isinstance(pretty, str)` guards, so
  non-string `attrs` values cannot silently leak into the worklist.
- Live-set / worklist both `set[str]` / `list[str]`. Final dead-fn
  iteration uses `module.functions` keys, dropping via `del`. No
  invariant violation observed.

### grad_pass.py

- `_rewrite_in_expr(expr, fn_by_name, new_fns) -> tuple[A.Expr, int]`.
  Dispatch arms cover every `Expr` subtype enumerated in
  `helixc/frontend/ast_nodes.py` at HEAD: `Call`, `Binary`, `Unary`,
  `Block`, `If`, `Match`, `Cast`, `Assign`, `Index`, `While`, `For`,
  `Loop`, `UnsafeBlock`, `Field`, `Return`, `Break`, `Range`,
  `TupleLit`, `ArrayLit`, `StructLit`, `Quote`, `Splice`, `Modify`.
  Leaf nodes (`Name`, `IntLit`, `FloatLit`, `StrLit`, `BoolLit`, etc.)
  correctly fall through the final `return (expr, count)` since they
  cannot contain `grad(...)` sub-trees.
- `_resolve_in_expr` mirrors the same coverage set.
- `_GradCallFinder(ASTVisitor)` short-circuits via `found` flag — the
  introspection-based base class drift-proofs the predicate path
  against new `Expr` subtypes, as documented in the Stage 28.8.2 note.
- `_extract_param_idx_from_args` raises `ValueError` on
  multi-param ambiguity and on non-`IntLit` indices; the typed `int`
  return is sound (range-checked against `len(target.params)`).

### elf_dyn.py

- `DynLayout` dataclass has every offset/vaddr/size field annotated
  `int` and every byte buffer annotated `bytes`. `plan_layout` is the
  sole constructor and populates all fields in one return; partial
  initialization is structurally impossible.
- Pre-code-layout guard `if interp_offset + interp_size > CODE_OFFSET:
  raise RuntimeError(...)` catches the only way the static phdr region
  could overrun the 0x1000 code anchor.
- `.dynamic` size accounting: `n_dyn_entries = len(dyn.needed_libs) +
  12` is sealed by `assert len(dyn_entries) == n_dyn_entries`; the
  twelve fixed entries (HASH, STRTAB, SYMTAB, STRSZ, SYMENT, PLTGOT,
  PLTRELSZ, PLTREL, JMPREL, FLAGS, FLAGS_1, NULL) match the appended
  set. Resilient to future `needed_libs` growth.
- `rela_plt` size: `SIZE_RELA * len(dyn.imports)` reserved up front and
  re-asserted after population (`assert len(rela_plt_bytes) ==
  rela_plt_size`).
- ELF / phdr emission re-asserts `len(ehdr) == SIZE_EHDR`,
  `len(phdrs) == ph_size`, and each region boundary
  (`len(out) == layout.<region>_offset`) on the way out, so any layout
  drift fails loudly at emission time rather than producing a corrupt
  binary.

## Verdict

PASS — 0 findings at confidence >= 75%.

Result-type invariants in `fdce.py`, walker totality in `grad_pass.py`
versus the current `Expr` subclass set, and dataclass / assertion
discipline in `elf_dyn.py` are all sound at this HEAD. No edits made;
this is the single Write permitted by scope.
