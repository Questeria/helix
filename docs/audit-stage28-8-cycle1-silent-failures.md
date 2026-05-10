# Stage 28.8 Cycle 1 — Silent-Failure Audit

**Date**: 2026-05-10
**Commit**: fc96595 (read-only audit, isolated worktree)
**Scope**: All Helix source — `helixc/bootstrap/*.hx`, `helixc/frontend/*.py`,
`helixc/ir/*.py`, `helixc/backend/*.py`, `helixc/stdlib/*.hx`.
**Trigger**: pre-Stage-29 audit gate — Cycle 1 of 5. Hunts silent corruption
windows the prior 3 audits missed, plus new windows introduced by stages
22-28.7 (which have never been audited before).
**Method**:
1. Traced the data flow of each new stage 22-28.7 module from
   parse → typecheck → lower → codegen and looked for places where:
   - a value falls through to a "safe" default (0, None, TyUnknown, AST_INT(0))
     without diagnostic
   - an `except Exception: pass` / `except TypeError: pass` swallows error
     state and continues
   - a pass advertises a trap-id but is never wired into the CLI dispatch
   - a walker iterates a fixed attr-name list and silently skips AST nodes
     whose attrs are spelled differently
2. Verified that the 21 still-open prior-audit findings remain real.
3. Cross-checked CLI driver `helixc/check.py` against module-level
   `validate_*` / `check_*` / `emit_warnings` entry points to identify
   "dead validation passes" — passes that exist but are never invoked.

**Result**: **9 new findings (4 CRITICAL, 5 HIGH, 4 MEDIUM, 0 LOW)** —
a *deeply* failed clean cycle. The dominant pattern across stages 22-28.7
is **dead validation**: stage authors built complete `validate_X` /
`check_X_ops` / `emit_warnings` helpers but neglected to wire them into
`helixc/check.py`, so the user-facing CLI silently bypasses every
phase-0 safety check that was supposed to come with each stage. Two
finding (Stage 28.5 panic and Stage 28.6 unsafe) are CRITICAL because
the entire feature is non-functional from the user's perspective —
`panic("msg")` emits no trap with no message; `*ptr` outside an unsafe
block emits no diagnostic.

A second dominant pattern is **walker/visitor skew**: three different
modules (panic_pass, deprecated_pass, unsafe_pass) each rolled their
own AST walker with a different fixed attr list, and each list has
distinct gaps. panic_pass uses `then_branch`/`else_branch` for `If`
recursion but the AST node has `then`/`else_` — so panic calls inside
if-branches are silently invisible to `validate_panic_args`.

---

## CRITICAL FINDINGS

### Finding 1: `panic("msg")` is non-functional — no codegen anywhere

**Location**:
- helixc/frontend/panic_pass.py (entire module: utility helpers only)
- helixc/frontend/typecheck.py:646-648 (registers "panic" as builtin to suppress unbound-name)
- helixc/ir/lower_ast.py (no panic handling at all)
- helixc/backend/x86_64.py (no panic / 28501 reference)
- helixc/check.py (never invokes panic_pass)
**Severity**: CRITICAL
**Category**: missing feature surfaced as silent dispatch
**Stage**: 28.5

**Description**:
Stage 28.5 advertised `panic("msg")` as a builtin that emits trap 28501
with the message string in `.rodata` for the runtime to print before
aborting. The actual implementation:
1. Adds `"panic"` to `_BUILTIN_NAMES` so the typechecker doesn't fire
   "unbound name 'panic'" (line 646-648 of typecheck.py).
2. Provides `collect_panics()`, `validate_panic_args()`,
   `find_unwind_attrs()`, `validate_unwind()` Python-side helper
   functions in panic_pass.py — for static analysis only.
3. **Has no lowering rule and no codegen** for `panic(...)` calls.

Search for `panic` / `28501` in lower_ast.py, x86_64.py, and the
bootstrap `kovc.hx` returns 0 hits. A user-written `panic("oh no")`
falls through the generic Call lowering, which expects a fn_table
entry for "panic". Since "panic" is a builtin (not in fn_table), the
backend either emits ud2 (silent crash with no message) or raises a
Python exception at lowering time.

Additionally, `validate_panic_args()` and `validate_unwind()` are
NEVER invoked by `helixc/check.py`. They exist but are dead code.
This means even the static validation that DOES exist (`panic(42)`
non-string arg, `panic("a", "b")` too many args, `@unwind` not
supported) is silently skipped — user gets neither runtime trap 28501
NOR compile-time validation. Two-layer silent failure.

**Reproducer**:
```
fn die() -> i32 {
    panic("everything is wrong");
    0
}

fn main() -> i32 {
    die()
}
```
Expected: program prints "everything is wrong" and exits with trap 28501.
Actual: depending on lowering, either silent ud2 (no message) or a
Python traceback at compile time. The trap-id 28501 reservation is
documented but never emitted.

**Recommendation**:
1. Add a lowering rule in `helixc/ir/lower_ast.py` for `Call(callee=Name("panic"), args=[StrLit(msg)])` → emit a CONST_STR op for `msg` + a TRAP op with id=28501, message-pointer attr.
2. Add corresponding backend emission in `helixc/backend/x86_64.py`: emit `.rodata` bytes for the message, emit `mov eax, 28501; ud2` (or use the existing trap-id machinery).
3. Wire `validate_panic_args(prog)` and `validate_unwind(prog)` into `helixc/check.py` after typecheck, before lowering. Make their diagnostics fail the build (return code 1) when present.

**Trap-id reservation**: 28501 (already reserved per panic_pass.py).

---

### Finding 2: Stage 28.6 unsafe gate is non-functional — `*ptr` outside unsafe is silently accepted

**Location**:
- helixc/frontend/unsafe_pass.py:125-135 (check_unsafe_ops)
- helixc/check.py (never invokes unsafe_pass)
- helixc/ir/lower_ast.py (no UnsafeBlock handling — treats it as plain Block)
**Severity**: CRITICAL
**Category**: missing feature surfaced as silent dispatch
**Stage**: 28.6

**Description**:
Stage 28.6 advertised `unsafe { ... }` as an explicit capability boundary:
raw-pointer ops (deref, arithmetic), FFI calls without effect-check, and
untyped memcpy may only appear inside an `unsafe` block; outside, they
trap 28601.

The actual implementation:
1. Parser accepts `unsafe { ... }` (AST node A.UnsafeBlock exists).
2. `unsafe_pass.py` provides `find_unsafe_blocks`, `find_raw_ptr_ops`,
   `check_unsafe_ops` — Python-side static analysis.
3. **`check_unsafe_ops` is NEVER called by `helixc/check.py`** — search
   the CLI driver for `unsafe_pass` / `check_unsafe` / `find_raw_ptr`
   returns 0 hits.
4. **`lower_ast.py` has no UnsafeBlock handling** — search for
   "UnsafeBlock" returns 0 hits. Codegen treats `unsafe { *p }`
   identically to `{ *p }`.

Consequences:
- `*ptr` outside an unsafe block is silently accepted. Trap 28601 never
  fires.
- The "deref vs ptr-value" distinction the audit prompt asked about
  IS correctly handled in `_is_raw_ptr_op` (checks `Unary` with op `*`,
  not bare Name), but since the entire pass is unwired, the distinction
  is academic.

Additionally, even if `check_unsafe_ops` were wired up, it only catches
the syntactic `*ptr` form. Other raw-pointer operations are silently
missed:
- **Pointer arithmetic** (`p + 1`, `p - q`) — `Binary` on pointer
  operands. The comment at unsafe_pass.py:108-109 says "we don't have
  type info here" and returns False unconditionally.
- **Index into raw ptr** (`p[i]`) — `Index` node, not checked.
- **Cast to/from ptr** (`x as *i32`) — `Cast` node, not checked.
- **Field deref through ptr** (`(*p).x`) — only the Unary deref portion
  is checked; field access on a deref'd pointer is not surfaced.

**Reproducer**:
```
fn read_unsafe(p: *i32) -> i32 {
    *p          // should trap 28601 — but doesn't, even outside unsafe.
}
```
Expected: compile-time error "raw-pointer deref outside unsafe block".
Actual: silent acceptance; the binary derefs `p` and either returns the
i32 or SEGVs at runtime if `p` is invalid.

**Recommendation**:
1. Wire `check_unsafe_ops(prog)` into `helixc/check.py` after typecheck.
   Treat its output as errors (return code 1).
2. Extend `_is_raw_ptr_op` to integrate with typecheck's type info so
   it can recognize ptr-arithmetic, ptr-index, and ptr-cast operations.
3. Add a lowering-side trap emission: when the lowering pass sees a
   raw-ptr op that was NOT inside an UnsafeBlock context, emit a
   `TRAP 28601` op. This catches the case where the Python validation
   was skipped (e.g., direct user invocation of `lower()` without
   running `check_unsafe_ops` first).

**Trap-id reservation**: 28601 (already reserved per unsafe_pass.py).

---

### Finding 3: Stage 28 `monomorphize_structs` does not walk fn bodies — body-level uses of `Pt<i32>` silently never instantiated

**Location**:
- helixc/frontend/struct_mono.py:59-120 (`collect_concrete_uses`)
- helixc/frontend/struct_mono.py:199-205 (mono pass appends but does NOT rewrite uses)
**Severity**: CRITICAL
**Category**: silent miscompile / missing instantiation
**Stage**: 28

**Description**:
`collect_concrete_uses` at line 67-68 documents: "Phase-0 collection is
conservative: it inspects function-signature types only (params + return
+ field tys of non-generic structs). Body-walking can be added
incrementally without breaking the API."

The walker iterates `prog.items` and visits ONLY:
- `FnDecl.params[*].ty` (line 113)
- `FnDecl.return_ty` (line 115)
- `StructDecl.fields[*].ty` of non-generic structs (line 118)

It does NOT visit:
- Function bodies (struct literals, calls with type args, casts, etc.)
- Let-binding type annotations
- Type aliases that resolve to generic structs
- ImplBlock target types or method signatures

**Consequence**: a user-written
```
struct Pt<T> { x: T, y: T }
fn main() -> i32 {
    let p = Pt::<i32> { x: 1, y: 2 };   // body-level use, not collected
    p.x
}
```
is silently broken. `collect_concrete_uses` returns `[]`. No `Pt__i32`
mono'd struct is emitted. The `Pt<i32>` reference at the call site is
left as `TyGeneric("Pt", [TyName("i32")])` — codegen sees this and
either emits ud2 / produces a Python traceback / silently produces the
generic Pt's slots (which still have `T` field types).

Additionally, even when collection IS triggered (via signature use),
the comment at line 199-204 explicitly says: "Uses are *not* rewritten
in this Phase-0 pass — the typechecker can lookup mangled names
directly via the new structs."

The "the typechecker can lookup mangled names" claim is false: nothing
in `typecheck.py` reads `Pt__i32` when the source code says `Pt<i32>`.
TyGeneric is resolved via the fixed mapping at typecheck.py:400-418
(D<>, Logic<>, WorkingMem<>, etc.); `Pt<i32>` returns
`TyUnknown(hint="generic Pt")` (line 418).

**Reproducer**:
```
struct Pair<T> { a: T, b: T }
fn main() -> i32 {
    let p = Pair::<i32> { a: 10, b: 32 };
    p.a + p.b               // expected: 42
                            // actual: silent miscompile — TyUnknown
                            //         flows through; either ud2 or
                            //         wrong-bit-pattern arithmetic
}
```

**Recommendation**:
1. Extend `collect_concrete_uses` with a `visit_expr` companion that
   walks fn bodies, looking for `StructLit` with parametric base
   names, `Cast` with TyGeneric target, etc.
2. Implement the use-site rewrite — `TyGeneric("Pt", [i32])` →
   `TyName("Pt__i32")` — at every site where the typechecker resolves
   types. This is the documented "incrementally without breaking the
   API" extension.
3. Until done, emit a diagnostic (trap 28001) for ANY remaining
   `TyGeneric` that names a user-defined generic struct at the end of
   the mono pass. Right now line 209-217's `find_uninstantiated` only
   detects fully-unused generics; it doesn't catch partially-used
   ones where signature mentions `Pt<i32>` but body uses `Pt<f64>`.

**Trap-id reservation**: 28001 (already reserved per struct_mono.py).

---

### Finding 4: Trap-id 24001 double-claim — kovc.hx emits 24001 for AST_MOD bf16, Stage 24 docs reserve 24001 for provenance violation

**Location**:
- helixc/bootstrap/kovc.hx:4220-4221 (`emit_trap_with_id(24001)` for bf16 MOD)
- helixc/frontend/typecheck.py:145 (docstring: "Trap 24001 emitted if a non-Logic value...")
- helixc/tests/test_provenance.py:147-155 (`RESERVED = 24001`)
**Severity**: CRITICAL
**Category**: trap-id collision / ambiguous diagnostic
**Stage**: 24 (vs bootstrap pre-24)

**Description**:
The bootstrap's trap-id convention is `AST_tag * 1000 + sub_id`. AST_MOD
is tag 24, so kovc.hx:4220 emits `24001` (`AST_MOD * 1000 + 1`) for
"bf16 operand in MOD". This trap-id is HARDCODED into the bootstrap
compiler and fires when the user writes `bf16 % anything`.

Stage 24 (TyLogic / provenance) reserved trap 24001 for "non-Logic value
passed where a Logic-typed parameter is required" (typecheck.py:145).
The reservation is documented in test_provenance.py:147-155 with an
explicit `RESERVED = 24001` constant.

This is a trap-id collision. Two distinct runtime conditions claim the
same id:
1. bf16 MOD (already implemented, fires routinely in bootstrap)
2. provenance violation (reserved but not implemented)

When Stage 24's provenance check is implemented, the user seeing
"trap 24001" at runtime will not be able to distinguish between
"you used bf16 with %" and "you violated provenance typing".

This is silent in the sense that it's a documentation/intent collision
rather than a runtime miscompile, but it IS a "silent failure" in the
debugging sense: a user staring at trap 24001 in a stack trace cannot
tell from the trap-id alone which condition fired. Without a
distinguishing source-line annotation, the diagnostic is ambiguous.

**Reproducer**:
```
fn modbf(a: bf16, b: bf16) -> bf16 {
    a % b           // trap 24001 fires per kovc.hx:4220
                    // — but is this "bf16 not supported in MOD" or
                    //   "non-Logic value in provenance context"?
}
```

**Recommendation**:
1. Reassign Stage 24's provenance trap to an unused id in the 24-namespace
   (e.g., 24100, 24200) — separate from `AST_MOD * 1000 + N` collisions.
   Update typecheck.py:145 docstring and test_provenance.py constant.
2. Document the trap-id namespace: `T * 1000 + N` is reserved for bootstrap
   AST-codegen traps where T = AST node tag. Stages-level reservations
   should pick a different prefix (e.g., 25xxx for stage 25, but 28xxx
   is also fine — 28501, 28601, 28701 are all in stage-numbered ranges
   that don't collide with AST tags).
3. Add a meta-document listing all reserved trap-id ranges so future
   stages can avoid collision.

**Trap-id reservation**: needs reassignment.

---

## HIGH FINDINGS

### Finding 5: Stage 28.7 deprecated-call walker silently misses `Index.indices`, `MatchArm.guard`, `For.iter_expr`, `Range.start/end`

**Location**: helixc/frontend/deprecated_pass.py:78-110 (`_walk_call_sites`)
**Severity**: HIGH
**Category**: silent miss / false-clean reports
**Stage**: 28.7

**Description**:
The deprecated-call walker at line 91-96 iterates a fixed set of "scalar
sub-expr" attrs and a fixed set of "sequence" attrs (line 97). Both
lists are incomplete relative to the AST schema in `ast_nodes.py`:

**Scalar-attr list (line 91-93)** includes: `expr, left, right, operand,
cond, then, else_, value, scrutinee, callee, init, rhs, body, then_branch,
else_branch`.

**Missing**:
- `obj` (Field.obj) — never recursed; `obj.field` where `obj` is `deprecated_fn()` is missed.
- `target` (Assign.target) — never recursed.
- `target_ty` (Cast.target_ty) — never recursed (typically a TyNode, but if it embeds an expr...).
- `iter_expr` (For.iter_expr) — never recursed; `for x in deprecated_fn() {...}` is missed.
- `start, end` (Range.start, Range.end) — never recursed.
- `guard` (MatchArm.guard) — never recursed; `match x { y if deprecated_fn(y) => ... }` is missed.
- `inner` (TyRef.inner, TyPtr.inner — not exprs but...)
- `name` (StructLit fields' second tuple element is the expr — line 102 handles via "fields" seq, OK).

**Sequence-attr list (line 97)** includes: `args, stmts, fields, elems, arms`.

**Missing**:
- `indices` (Index.indices) — never iterated; `arr[deprecated_fn()]` is missed.

Additionally:
- Line 109-110: `except TypeError: pass` silently swallows any error
  during sequence iteration. Hides genuine bugs.
- Line 95: `if sub is not None and hasattr(sub, "span")` — AST nodes
  WITHOUT a `span` attribute (rare but possible for synthesized nodes)
  are silently skipped.

**Consequence**: `find_deprecation_call_sites(prog)` returns an
incomplete list. Users get warnings only for calls in specific
"sanctioned" positions (binop operands, args, block stmts) and miss
calls in indexing brackets, match guards, for-loop iterables, range
bounds.

**Reproducer**:
```
@deprecated("use new_fn")
fn old_fn(x: i32) -> i32 { x }

fn main() -> i32 {
    let arr: [i32; 4] = [0, 0, 0, 0];
    let v = arr[old_fn(2)];          // ← deprecated call in Index.indices
    for i in 0..old_fn(10) {         // ← deprecated call in Range.end
        v;
    }
    v
}
```
Expected: 2 deprecation warnings.
Actual: 0 warnings. The user is silently using deprecated APIs.

**Recommendation**:
1. Replace the hand-rolled walker with a generic AST walker that uses
   `dataclasses.fields()` reflection to iterate all sub-nodes.
2. Or, at minimum, extend the attr lists to cover all sub-expr-bearing
   attrs (audit `ast_nodes.py` to enumerate them).
3. Replace `except TypeError: pass` with `except TypeError as e: raise`
   so silent walker errors surface.
4. Same fix applies to `panic_pass._walk_exprs` and
   `unsafe_pass._walk` (Findings 6 and 8).

---

### Finding 6: Stage 28.5 panic walker uses `then_branch`/`else_branch` but AST has `then`/`else_` — panics inside if-branches are invisible

**Location**: helixc/frontend/panic_pass.py:60-65 (`_walk_exprs` attr list)
**Severity**: HIGH
**Category**: silent miss / false-clean validation
**Stage**: 28.5

**Description**:
panic_pass.py:60-65 lists the scalar-attr names: `"left", "right",
"operand", "cond", "then_branch", "else_branch", "value", "scrutinee",
"callee", "init", "rhs", "body"`.

BUT `ast_nodes.py:217-220` defines `If` with attrs `cond`, `then`,
`else_`. There are NO `then_branch` / `else_branch` attrs on `If`.

So the walker NEVER recurses into the `then` block or the `else_` block
of an `If` node. Panic calls inside if-branches are silently invisible
to `collect_panics()` and `validate_panic_args()`.

The `body` attr IS in the list, but `If` doesn't have a `body` attr —
that's `While`, `Loop`, `For`. So `If` falls into the gap completely.

Additionally `iter_expr` (For.iter_expr) is also missing, so a panic
in a for-loop iterable is missed.

**Reproducer**:
```
fn die_conditionally(flag: i32) -> i32 {
    if flag > 0 {
        panic(42)            // ← panic(non-string) inside if.then
                             //   validate_panic_args should diag
                             //   "arg must be a string literal".
                             //   Actual: silently accepted, no diag.
    } else {
        panic("ok")          // ← legitimate panic inside if.else_
                             //   not reported by collect_panics.
    }
}
```
Expected: validate_panic_args returns 1 diagnostic for `panic(42)`.
Actual: returns 0 diagnostics. Silent acceptance of malformed panic.

**Recommendation**:
Same as Finding 5: fix the attr list to match `ast_nodes.py`. Specifically
add `then`, `else_`, `iter_expr` to the scalar-attr list at line 60-62.

Or — strongly recommended — share a single AST walker across panic_pass,
deprecated_pass, and unsafe_pass. The fact that each pass rolled its own
walker with DIFFERENT gaps is a maintenance bug waiting to compound.

---

### Finding 7: Stage 25 `@trace` attribute is non-functional — codegen never emits entry/exit events

**Location**:
- helixc/frontend/trace_pass.py (entire module: simulator + helpers, no wiring)
- helixc/ir/lower_ast.py (no trace handling)
- helixc/backend/x86_64.py (no trace handling)
- helixc/check.py (never invokes trace_pass)
**Severity**: HIGH
**Category**: missing feature surfaced as silent dispatch
**Stage**: 25

**Description**:
Stage 25 advertised `@trace fn f(...) { ... }` as a fn-level attribute
that causes codegen to emit trace-log calls into the prologue (ENTRY)
and epilogue (EXIT) so a runtime trace buffer captures each invocation's
args + return value.

The actual implementation:
1. Parser accepts `@trace` (existing `_parse_attributes` path).
2. trace_pass.py defines `TraceBuffer`, `is_traced(fn)`,
   `traced_fn_names(prog)`, `validate_trace_attrs(prog)`, plus a
   `trace_equiv` predicate — all Python-side simulation.
3. **No codegen wiring**: searches for "@trace", "trace_buffer", and
   "TRAP_TRACE_OVERFLOW" in `lower_ast.py`, `x86_64.py`, and
   `kovc.hx` return 0 hits.
4. **`validate_trace_attrs` is never invoked by `helixc/check.py`** —
   so `@trace` on `extern "C"` fn (which the validator rejects) is
   silently accepted.

The prompt asked: "can trace buffer overflow silently when @trace fn
is recursive?" The Python `TraceBuffer.push` correctly raises
OverflowError when capacity is exceeded — loud failure. But since
nothing in the compiler/runtime ever calls `TraceBuffer.push`, the
overflow scenario is moot. The bigger problem is that no events are
emitted at all.

**Reproducer**:
```
@trace
fn ping(x: i32) -> i32 {
    x + 1
}

fn main() -> i32 {
    ping(41)            // expected: trace buffer records (entry, ping, [41])
                        //                                (exit, ping, [42])
                        // actual: nothing recorded; @trace silently ignored.
}
```

**Recommendation**:
1. Add lowering for `@trace`: prologue inserts a CONST_STR for the fn name
   + a TRACE_ENTRY op; epilogue inserts a TRACE_EXIT op with the return
   value.
2. Add backend emission: emit `mov rdi, fn_name_ptr; mov rsi, args; call __trace_entry`
   in the prologue and similar in the epilogue.
3. Wire `validate_trace_attrs(prog)` into `helixc/check.py` so
   `@trace` on extern fns is caught.
4. Implement the runtime trace buffer with the documented 25001 overflow
   trap.

**Trap-id reservation**: 25001 (already reserved per trace_pass.py).

---

### Finding 8: `parser.parse(include_stdlib=True)` silently drops stdlib StructDecl / TraitDecl / ImplBlock / ConstDecl items

**Location**: helixc/frontend/parser.py:1531-1540 (stdlib merge loop)
**Severity**: HIGH
**Category**: silent feature loss
**Stage**: (regression — affects everything that includes stdlib)

**Description**:
`parser.parse(source, include_stdlib=True)` at line 1525-1540 walks the
configured `stdlib_files` list (line 1520: 16 files including `vec.hx`,
`hashmap.hx`, `tensor.hx`, `nn.hx`, `agi_memory.hx`, etc.) and merges
items into `user_prog.items`.

The merge logic at line 1532-1540 ONLY merges items of type
`ast.FnDecl` (line 1533) or `ast.EnumDecl` (line 1537). All other item
types are silently DROPPED:
- `StructDecl` — stdlib's `struct Vec<T> { ... }` is silently lost.
- `TraitDecl` — stdlib's trait definitions silently lost.
- `ImplBlock` — stdlib's `impl T for X { ... }` silently lost.
- `ConstDecl` — stdlib's compile-time constants silently lost.
- `TypeAlias` — stdlib's type aliases silently lost.
- `ModuleDecl` / `ModBlock` / `UseDecl` — silently lost.

**Consequence**: A user writes
```
fn main() -> i32 {
    let v: Vec<i32> = Vec::new();
    v.len() as i32
}
```
with `include_stdlib=True`. The parser:
1. Parses user code → `let v: Vec<i32> = ...` references TyGeneric("Vec", [i32]).
2. Loads `stdlib/vec.hx` → `struct Vec<T> { ... }`, `impl Vec<T> { fn new(...) ... }`.
3. **Drops** `StructDecl(Vec)` and `ImplBlock(Vec)` during merge.
4. Adds `Vec::new` fn-decl (if defined as plain fn, not impl method).
5. User code references `Vec<i32>`, which is now unknown → silent
   TyUnknown propagation.

Additionally at line 1527-1528, files that don't exist on disk are
silently skipped (`continue`). A partial Helix install missing `vec.hx`
gives the user NO indication. Programs that previously compiled with
the full stdlib silently fail.

**Reproducer**: any user program importing `Vec<T>` or `HashMap<K,V>` or
`Result<T, E>` (struct definitions) will silently misbehave.

**Recommendation**:
1. Change the merge logic to handle ALL item types — at minimum:
   StructDecl, TraitDecl, ImplBlock, ConstDecl, TypeAlias.
2. Track stdlib-item-name conflicts the same way (user takes precedence).
3. For missing stdlib files, emit a warning to stderr or raise an
   explicit error if `include_stdlib=True` was passed and a file in
   the configured list is missing.

---

### Finding 9: `check.py` --emit-asm and `-o` paths have no try/except — internal compiler errors leak Python tracebacks instead of clean error messages

**Location**: helixc/check.py:354-363 (--emit-asm) + 383-393 (`-o`)
**Severity**: HIGH
**Category**: poor error UX / inconsistent with --emit-ptx
**Stage**: 23

**Description**:
The `--emit-ptx` path at check.py:375-380 correctly wraps the backend
call in try/except:
```python
try:
    ptx = emit_ptx(tile_mod)
    print(ptx)
except Exception as e:
    print(f"   ptx: backend error: {e}", file=sys.stderr)
    return 1
```

The `--emit-asm` (line 354-363) and `-o` (line 383-393) paths invoke
`compile_module_to_elf(mod)` with NO try/except. If the backend raises
(any internal compiler error — type-mismatch leaking through, IR shape
unexpected, codegen NotImplementedError, struct-pack overflow on a
NaN const, etc.), the user gets a Python traceback dumped to stderr.
Exit code is 1, but no clean compile-error message.

This is inconsistent UX. A clean error message would say
"helixc: codegen error at fn 'foo' line 12: <reason>" and exit 1.

**Consequence**: every IR or backend bug is presented to the user as
a Python crash. Users have no clear signal that "this is a compiler
bug, not your code's bug". Worse, it discourages users from filing
bug reports because the traceback looks scary.

Additionally, `--emit-asm` at line 356 prints "{len(elf)} bytes of ELF
(use objdump -d for asm)" even when codegen produced nonsense bytes —
the user thinks the asm is valid when it's actually corrupt.

**Recommendation**:
1. Wrap each backend call in try/except. Translate exception class to
   a clean error message + exit 1.
2. Distinguish "compile error (user's fault)" vs "internal compiler
   error (helixc's fault)". The former gets a CompileError diagnostic
   render; the latter gets "helixc: internal error: <type>: <msg>"
   with a hint to file a bug.
3. For `-o`, verify the file write actually completes (currently
   silent on permission errors at write-time since `with open(...)`
   commits on close, not on write).

---

## MEDIUM FINDINGS

### Finding 10: `check.py` -O2 / -O3 silently runs ONLY fdce — const_fold / cse / dce are documented but not invoked

**Location**: helixc/check.py:329-339
**Severity**: MEDIUM
**Category**: silent feature loss / misleading help text
**Stage**: 23

**Description**:
The `_print_help()` docstring at check.py:30-32 says:
```
-O0 / -O1 / -O2 / -O3
                    Optimization level (0=none, 1=fold, 2=+cse+dce,
                    3=+aggressive). Default -O1.
```

The actual implementation at line 329-339:
```python
mod = lower(prog)
if a.opt_level >= 1:
    fdce_module(mod)
if a.opt_level >= 2:
    try:
        from .ir.passes import dce as _dce_mod  # noqa: F401
        # Pass is invoked elsewhere; placeholder here for shape.
    except ImportError:
        pass
```

Observations:
1. `-O1` runs ONLY `fdce_module` (function-level dead code), NOT
   `fold_module` (constant folding) as the help text claims.
2. `-O2` runs the same as `-O1` plus a `try/import dce; pass` block
   that imports the module but does NOT invoke `dce_module()`. The
   comment "Pass is invoked elsewhere" is misleading — it's not
   invoked anywhere in the user-flag path.
3. `-O3` is treated identically to `-O2`.
4. `cse_module` is never invoked at any opt level.

The CLI user thus has NO way to invoke `const_fold` / `cse` / `dce`
from `helixc/check.py`. Those passes ARE wired into the `examples/run.py`
pipeline (line 78-81) and the `x86_64.compile_module_to_elf` internal
path (line 2998-3004), but those aren't reachable from `--emit-ir` /
`--emit-asm` flags.

Additionally, the `try/except ImportError: pass` at line 335-339 is a
classic "empty except" — even if it WERE supposed to do something, the
import silently failing would mask the real problem.

**Consequence**: a user invoking `python -m helixc.check -O2 --emit-ir
loss.hx` sees IR that is NOT folded/cse'd/dce'd, but the help text led
them to expect those optimizations to have run. Comparing IR across
opt-levels is meaningless — they're all the same.

**Recommendation**:
1. Wire `fold_module`, `cse_module`, `dce_module` into check.py at the
   appropriate opt-level branches.
2. Remove the misleading `try/except ImportError: pass` placeholder
   block.
3. Either remove the help-text mention of `-O2 / -O3` or fix the
   implementation to match.

---

### Finding 11: `pytree._unflatten` silently defaults missing-path gradients to 0.0 with no diagnostic

**Location**: helixc/frontend/pytree.py:162-174 (`_unflatten`)
**Severity**: MEDIUM
**Category**: silent gradient zeroing
**Stage**: 26

**Description**:
The prompt asked: "does pytree flatten miss any leaf type and silently
zero its gradient?"

`flatten_pytree` correctly raises `ValueError` for unknown leaf types
(line 119-122). Good — loud failure.

But `_unflatten` at line 167-168:
```python
if is_pytree_leaf(f.ty):
    out[f.name] = grads.get(path, 0.0)        # ← silent default
```
silently substitutes `0.0` for any path that is NOT present in
`grads_by_path`. There is NO diagnostic if the gradient producer (e.g.,
the AD pass) failed to populate an expected leaf.

This means if the AD pass has a bug where some leaves are silently
skipped (e.g., off-by-one in path naming, e.g., `model.layer1.w` vs
`model.layer_1.w`), the corresponding gradients become 0.0 — the
gradient descent stalls on those parameters with no error.

Similarly at line 173: non-leaf, non-struct fields silently get
`None`. If the user's pytree has an unexpected field type (e.g., a
tuple), the gradient passes through as `None` without diagnostic.

The Python-side `validate_pytree(decl, struct_decls)` at line 177-185
runs flatten as validation — but is NEVER called from `helixc/check.py`.
Dead validation.

**Consequence**: a buggy AD pass silently produces zero gradients for
some leaves. Training stalls; user has no signal to debug.

**Reproducer**: hypothetical — requires a downstream AD bug — but the
silent-default-zero pattern is the enabler.

**Recommendation**:
1. Change `_unflatten` to raise `ValueError` when a leaf path is missing
   from `grads_by_path` (after the AD pass should have populated it).
   Alternative: take an explicit `default=0.0` parameter that the
   caller must pass; default to raising.
2. Wire `validate_pytree` into the typechecker / AD pass so the schema
   is checked before the runtime path traversal.

---

### Finding 12: Stage 27 `parse_autotune_attrs` silently swallows malformed `autotune:` attrs

**Location**: helixc/frontend/autotune.py:40-56 (`parse_autotune_attrs`)
**Severity**: MEDIUM
**Category**: silent miss / no diagnostic
**Stage**: 27

**Description**:
`parse_autotune_attrs` decodes attrs of the form `"autotune:KEY=v1,v2,..."`
back into `{KEY: [v1, v2, ...]}`. The dispatch:
- Line 49-50: if `"="` not in body, `continue` silently (no diag).
- Line 52-55: if any value is non-integer, `except ValueError: continue`
  silently drops the entire key.

Consequence: a user writing `@autotune(BLOCK_SIZE: [16, "fast", 32])`
(typo: string in int list) gets the entire BLOCK_SIZE entry silently
dropped. `validate_autotune` then reports "no parameters parsed"
(line 130-133) — but the underlying cause (the typo) is masked.

Worse: `validate_autotune` is itself NEVER invoked from `check.py`, so
even the secondary diagnostic is dead.

Additionally, `autotune_variants` does not dedupe values. `@autotune(X:
[1, 1, 2])` (typo: duplicate value) generates 3 cfgs, of which the
first two mangle to the SAME variant name. The second registration
silently overwrites the first (or is dropped depending on the variant-
table impl).

**Reproducer**:
```
@autotune(BLOCK_SIZE: [16, "fast", 32])
@kernel
fn k() { ... }
```
Expected: compile error "autotune values must be integers".
Actual: BLOCK_SIZE silently absent; if not the only param, the @autotune
is silently incomplete.

**Recommendation**:
1. Replace `except ValueError: continue` with explicit diagnostic
   collection.
2. Add dedup to `autotune_variants`.
3. Wire `validate_autotune` into `check.py`.

---

### Finding 13: Stage 28 `_ty_key` collapses all `TyFn` to `("?", "TyFn")` — distinct fn types silently dedupe to one instantiation

**Location**: helixc/frontend/struct_mono.py:123-137 (`_ty_key`)
**Severity**: MEDIUM
**Category**: silent dedup collapse / wrong codegen
**Stage**: 28

**Description**:
`_ty_key` converts a TyNode to a hashable key for dedup. It handles
`TyName`, `TyGeneric`, `TyTuple`, `TyArray`, `TyRef`, `TyPtr` explicitly.
Other TyNode kinds (TyFn, TyTensor, TyTile, TyMemTier) fall through to:
```python
return ("?", type(t).__name__)        # line 137
```

This means ALL `TyFn` instances have the same key `("?", "TyFn")`,
regardless of their actual parameter and return types. Two distinct
parametric struct instantiations
- `Pt<fn(i32) -> i32>`
- `Pt<fn(f32) -> f32>`
silently dedupe to a single mono'd struct. The struct's field type
substitutes `T → fn(i32) -> i32` in one instance and `T → fn(f32) -> f32`
in another, but only one is emitted.

Same problem for `TyTensor` (any two tensor types collapse), `TyTile`,
`TyMemTier` (any two memory-tier types collapse).

**Reproducer**:
```
struct Holder<T> { x: T }
fn use_pair() -> i32 {
    let a: Holder<fn(i32) -> i32> = Holder { x: add_one };
    let b: Holder<fn(f32) -> f32> = Holder { x: floor_f };
    // Mono pass emits ONE Holder__fn — silently wrong for one of them.
    0
}
```
Codegen sees both `Holder<...>` references → looks up `Holder__fn` →
finds only one. Either a's or b's field-type slot has the wrong width
(8 bytes for i32-fn-pointer vs 8 for f32-fn-pointer — actually same
width here, but the codegen-side type-tag dispatch differs).

**Recommendation**:
Implement proper `_ty_key` for TyFn (encode params + ret), TyTensor
(encode dtype + shape), TyTile, TyMemTier (encode tier).

---

## What was checked but found OK (no new finding)

- **Stage 22 (pretty errors) — color escape leak**: `use_color()` honors
  `NO_COLOR` env, `HELIXC_COLOR={0,1}` env, falls back to `isatty()`.
  When stderr is redirected to a file, isatty is False, color is False,
  no escapes emitted. Correct. The only minor edge case: `HELIXC_COLOR`
  set to non-{"0","1"} value falls through to isatty (e.g.,
  `HELIXC_COLOR=true` doesn't force-enable). Trivial UX nit, not silent
  corruption.
- **Stage 22 `render_caret` source-line bounds check** at line 162:
  out-of-range line gracefully falls back to `file:line:col: level: msg`
  format. No silent miss.
- **Stage 23 `compile_module_to_elf` failure mid-write**: line 387
  `with open(...) as f: f.write(elf)` — if `compile_module_to_elf` raises
  BEFORE `open()`, no file is created. If `write()` raises mid-byte
  (rare for in-memory bytes), the partial file remains. Acceptable for
  Phase-0.
- **Stage 24 TyLogic resolution** at typecheck.py:404-406: correctly
  handles `Logic<T>` → `TyLogic(inner=...)`. The pass ordering concern
  the audit prompt raised (does the new pass break struct/enum lookup
  elsewhere?) is unfounded — TyLogic is handled BEFORE the tier-map
  fallthrough and the user-type fallthrough. No ordering hazard.
- **Stage 25 TraceBuffer.push** at trace_pass.py:65-69: correctly
  raises OverflowError on cap exceeded. Loud failure mode (if anything
  ever called it).
- **Stage 26 `flatten_pytree` unknown-leaf**: correctly raises ValueError
  (line 119-122) — loud, not silent. The walker's depth cap (line
  90-92) also raises ValueError loudly.
- **Stage 27 `validate_autotune`** correctly diagnoses `@autotune` without
  `@kernel`, empty value lists, and >16 variants. The diagnostics ARE
  generated; they're just never displayed because `validate_autotune`
  isn't wired into check.py (Finding 12).
- **Stage 28 `instantiate` arity-mismatch** at struct_mono.py:145-149:
  correctly raises ValueError. Loud, not silent.
- **Stage 28.6 `_is_raw_ptr_op` deref-vs-value distinction**: correctly
  fires only on `Unary` with op `"*"`, not on bare `Name(p)`. The
  prompt's specific concern is correctly handled. But see Finding 2 for
  the upstream wiring problem.
- **Stage 28.7 deprecated walker walks `Call.callee`**: line 91-93's
  attr list includes `callee`, so `deprecated_fn(...)` IS detected when
  the call is the direct expression. Only the missed contexts
  (Index.indices, Range bounds, etc.) are problematic — see Finding 5.

---

## Status of 21 still-open prior-audit findings

Re-reviewed each open finding against current source (commit fc96595).
Verdicts:

### audit-stage5-6-aggregates.md (7 open)
| # | Description | Still valid? |
|---|---|---|
| F2 | PAT_VARIANT cross-enum match — enum_idx p3 ignored | YES — kovc.hx:3383-3394 still reads only p1/p2 |
| F4 | Unknown struct field name eats `.` and IDENT silently | YES — parser.hx:1454-1477 unchanged |
| F9 | emit_variant_subpats / emit_tuple_subpats disp8 wrap at idx > 15 | YES — kovc.hx:3294-3334 unchanged |
| F10 | `__enum_payload` non-INTLIT idx silently uses 0 | YES — parser.hx:2667-2680 unchanged |
| F11 | bind_alloc_offset cap-check missing | YES — kovc.hx:1037-1041 unchanged |
| F12 | Struct fn-call arg identity sentinel-15 collapse | YES — kovc.hx:4924-4935 unchanged |
| F13 | last_enum_idx dead infra | YES — parser.hx:174-178 + writes at 2486, 2541 still no readers |

### audit-stage7-8-typesystem.md (6 open)
| # | Description | Still valid? |
|---|---|---|
| F4 | mr_tab cap-32 overflow no 71001 trap | YES — parser.hx:242-258 unchanged |
| F7 | PAT_LIT 32-bit cmp on wide scrut | YES — kovc.hx:3338-3366 unchanged |
| F9 | clone_with_rewrite only handles AST_CALL at root | YES — parser.hx:3539-3558 unchanged |
| F10 | Mono clone discards is_checkpoint | YES — parser.hx:3666-3671 unchanged |
| F11 | self.method() in impl bodies doesn't work | YES — parser.hx:1378-1439 unchanged |
| F12 | PAT_VARIANT sub-pat disp8 wrap at idx > 15 | YES — same as F9 of stage 5-6 doc; both still open |

### audit-stage9-16-codegen.md (5+3 open)
**Note**: this audit doc is REFERENCED by `STAGE_28_8_PRE_29_AUDIT_GATE.md` but
**does not exist** in the repo. I searched `docs/` and the only audit docs
present are: `audit-stage4-followup.md`, `audit-stage5-6-aggregates.md`,
`audit-stage7-8-typesystem.md`. Cannot verify the 5+3 stage-9-16 findings
without the source doc. Recommend either (a) restoring the missing doc
from git history or (b) re-running the stage 9-16 audit as part of this
gate to refresh those findings.

**Validity estimate**: 13 of the 13 prior open findings from the two
existing audit docs remain valid concerns. The 8 prior findings from
the missing `audit-stage9-16-codegen.md` cannot be assessed. Net: 13
confirmed + 8 unknown = 21 still-open, matching the gate doc's count
modulo the missing-doc caveat.

---

## Summary

| #  | Severity  | Stage | Finding |
|----|-----------|-------|---------|
| 1  | CRITICAL  | 28.5  | `panic("msg")` is non-functional — no codegen |
| 2  | CRITICAL  | 28.6  | unsafe gate non-functional — `*ptr` outside unsafe silently OK |
| 3  | CRITICAL  | 28    | `monomorphize_structs` does not walk fn bodies — body-level uses silently uninstantiated |
| 4  | CRITICAL  | 24    | Trap-id 24001 double-claim — bf16 MOD vs provenance violation |
| 5  | HIGH      | 28.7  | Deprecated-call walker misses Index/MatchArm.guard/For/Range |
| 6  | HIGH      | 28.5  | Panic walker uses `then_branch`/`else_branch` but AST has `then`/`else_` |
| 7  | HIGH      | 25    | `@trace` is non-functional — codegen never emits entry/exit |
| 8  | HIGH      | (parser) | stdlib StructDecl / TraitDecl / ImplBlock / ConstDecl silently dropped |
| 9  | HIGH      | 23    | `--emit-asm` and `-o` no try/except — internal errors leak tracebacks |
| 10 | MEDIUM    | 23    | `-O2 / -O3` silently runs only fdce, not fold/cse/dce |
| 11 | MEDIUM    | 26    | `pytree._unflatten` silently zeros missing-path gradients |
| 12 | MEDIUM    | 27    | `parse_autotune_attrs` silently swallows malformed attrs |
| 13 | MEDIUM    | 28    | `_ty_key` collapses all TyFn / TyTensor / TyTile to one key |

**Total: 13 new findings (4 CRITICAL, 5 HIGH, 4 MEDIUM, 0 LOW)**.

This is far from a clean cycle. **NOT counted toward the 5-clean target.**

### Stop-the-line recommendation: **YES**.

Specifically, the following CRITICAL issues block any further progress
on stages 22-28.7 hardening:

1. **Findings 1 and 2 (panic + unsafe non-functional)**: these are not
   "edge case silent failures" — they're "the entire feature doesn't
   work". Stage 28.5 and Stage 28.6 should be considered NOT LANDED until
   codegen wiring + CLI dispatch is implemented. The trap-ids 28501 and
   28601 are reserved but never emitted; the validation passes exist but
   are never called.

2. **Finding 3 (Stage 28 parametric struct body-uses)**: a user writing
   `Pt::<i32> { x: 1, y: 2 }` in a fn body gets silent miscompile.
   This is the most common way to instantiate a parametric struct;
   the Phase-0 limitation (signature-only walk) makes the entire stage
   28 feature non-usable for typical workflows.

3. **Finding 4 (trap-id 24001 collision)**: cheap to fix (reassign Stage
   24's trap to an unused id) but must happen before Stage 24's
   provenance check is actually wired up. Otherwise the user-facing
   diagnostics become ambiguous and a debugging black hole.

The HIGH findings (5-9) compound the silent-failure landscape but are
mechanical fixes. They should be batched into a single sweep — Findings
5, 6, and 8 in particular share root cause (hand-rolled AST walkers
with inconsistent attr lists; the right fix is a single reflection-based
walker shared across panic_pass / deprecated_pass / unsafe_pass /
totality / etc.).

### Cycle 1 status

**Cycle 1 does NOT count toward the 5-clean target.** With 4 CRITICAL +
5 HIGH new findings, this cycle clearly fails the "zero new
HIGH/CRITICAL" criterion. The clean-count remains at 0.

### Estimated remaining open findings

13 confirmed-valid + 8 unverifiable (from missing audit-stage9-16
doc) + 13 new from this cycle = **34 open findings** going into the
fix pass.

After fixes: another audit cycle is required. The fix budget should
prioritize CRITICAL (1-4) and the walker unification (5, 6) since
those affect ALL three new-stage validators.

### Methodology validation

This audit identified 13 new findings in stages 22-28.7 alone — stages
that landed less than a week before this audit and had only ad-hoc
review. The pre-Stage-29 gate's existence is vindicated; without it,
the helixc-Python reference would have been dropped (Stage 29) before
panic/trace/unsafe were ever wired, and the self-host (Stage 30) would
have inherited the same non-functional features with no oracle to
diagnose against.

Recommend continuing with cycles 2-5 after each fix batch, with the
budget split between:
- Fixing the new findings (1-13) — estimated 2-3 cycles to clear all
  CRITICALs + HIGHs.
- Re-validating the 13 still-valid prior findings — each fix should
  land WITH a regression test in `helixc/tests/test_codegen.py` (per
  the resolution pattern documented in audit-stage5-6 status).
- Restoring or re-running `audit-stage9-16-codegen.md` to assess the 8
  unverifiable findings.
