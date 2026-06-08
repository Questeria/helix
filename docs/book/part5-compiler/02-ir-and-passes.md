# IR & lowering passes (monomorphization)

*What this chapter covers:* the internal representation `kovc` lowers through — an arena-encoded
AST, not a separate SSA IR — and the lowering passes that run over it, with **monomorphization** as
the centerpiece: how a generic `fn id[T](…)` becomes one concrete, fully-typed function per requested
type, via the request table `mr_tab`, the type-tag encoder `ty_ident_to_tag`, the name mangler
`mangle_name_into_arena`, and the `monomorphize_pass` clone synthesizer; how the codegen side then
skips templates and resolves calls by backpatch; and why every one of these passes must be
deterministic so the self-host fixpoint `K2 == K3 == K4` stays byte-identical.

This is the second chapter of Part V and is written for a **contributor** — someone reading or
changing `kovc`'s sources. It builds on the front-end chapter (lexer, parser, the arena and the
AST node encoding) and feeds the back-end chapter (x86-64 ELF emission). It is the compiler-internals
companion to Part III's [Generics, traits & closures](../part3-language/04-generics-traits-closures.md),
which gives the *language* view of the same feature with the user-facing syntax and the gate-proven
exit codes; this chapter does **not** repeat that material — it shows the machinery underneath it.

> **For AI agents:** the identifier names in this chapter (`mr_tab`, `ty_ident_to_tag`,
> `mangle_name_into_arena`, `monomorphize_pass`, `clone_with_rewrite`) are real `kovc` source
> symbols. The request-recording and clone-synthesis halves live in
> [`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx); the codegen halves
> (template-skip, the type-driven op cascade, the `__i32` backpatch fallback) live in
> [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx). Cite each at its real file; do not
> assume a symbol lives in `kovc.hx` just because it concerns codegen.

---

## The IR: an arena-encoded AST, not a separate SSA

`kovc` has no separate optimizing IR in the LLVM/SSA sense. Its single internal representation is the
**AST itself**, encoded as fixed-width node records inside the compiler's `__arena` (the same flat
`i32`-slot heap the front-end chapter introduces). A node is a small run of arena slots: slot 0 is a
**tag** (an integer naming the node kind — `AST_FN_LIST` is 15, `AST_FN_DECL` is the function-decl
tag, `AST_ADD` is 2, `AST_PARAM` is 18, and so on), and the following slots are that kind's payload
(`p1`, `p2`, …) plus link fields for sibling/child chains. Functions hang off an `AST_FN_LIST` spine,
each list cell pointing at one `AST_FN_DECL`; statements and expressions are child sub-trees.

Because the IR *is* the AST, "lowering" in `kovc` means **rewriting and appending AST nodes** before
the back end walks the tree and emits bytes — there is no lifting to a register IR and lowering back.
The `AST_FN_DECL` record is the hub everything keys off; its slots, as the codegen pre-pass documents
them:

**Fragment** (the `AST_FN_DECL` slot layout, as the codegen pre-pass reads it; excerpt of
[`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx)):

```helix
            // AST_FN_DECL: p1 = name_start, p2 = name_len,
            //              p4 (slot 4) = params_head, p5 (slot 5) = ret_ty,
            //              p6 (slot 6) = is_generic flag (Stage 8).
```

Two of those slots are the spine of this chapter: **slot 5** (`ret_ty`, the return type tag) and
**slot 6** (the *is-generic* flag — `1` marks a template that must not be emitted as-is). Parameters
are an `AST_PARAM` chain off slot 4; each `AST_PARAM` carries its own type tag in its slot 4 and a
`next` link in slot 3. A generic parameter is not a real type tag — it is encoded as a **generic
marker** in the `200…` range (`200 + gp_idx`), which the monomorphizer later substitutes for a
concrete tag. The pre-pass that registers each function's signature into `fn_type_table` says so
directly:

**Fragment** (why generic templates are skipped during signature registration; excerpt of
[`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx)):

```helix
            // Stage 8: skip generic-template fn decls — their concrete
            // mono'd clones (synthesized in the mono-pass below) are the
            // ones registered + emitted. Generic templates carry param
            // type tags in the 200..203 range (gp_idx markers) which
            // can't represent through 4-bit fn_type_table packing.
            let fn_is_generic = __arena_get(fn_idx + 6);
            if fn_is_generic == 1 {
                // skip this fn entirely
            } else {
```

---

## The lowering pipeline, end to end

The full path from source text to ELF bytes is a fixed sequence. The parse-side passes run inside the
parser; the emit-side passes run inside `kovc`'s `emit_elf_for_ast_to_path`:

1. **Lex + parse** build the AST in the arena. *During* parsing, each turbofish use site records a
   monomorphization request into `mr_tab` (see the next section) — so by the time parsing finishes,
   the request table is already populated.
2. **`monomorphize_pass`** walks `mr_tab` and appends one concrete `AST_FN_DECL` clone per request
   to the tail of the `AST_FN_LIST`. This is the centerpiece pass.
3. **`grad_pass`** (autodiff) appends differentiated function clones for any `grad`-requested loss
   functions — same append-to-tail shape as monomorphization.
4. **Validation passes** — `panic_pass`, `unwind_pass`, `trace_pass`, `deprecated_pass`,
   `unsafe_pass`, `autotune_pass` — walk every non-template `AST_FN_DECL` body and collect
   diagnostics into a diag arena; they emit no code, only fail-closed errors.
5. **Codegen** (`emit_elf_for_ast_to_path`): a signature **pre-pass** registers each non-template
   function in `fn_type_table`; then the emitter walks the `AST_FN_LIST`, **skips** templates
   (slot 6 == 1), and emits each concrete function with type-driven instruction selection.
6. **Backpatch** resolves every `call` placeholder against the `fn_table`, with a documented
   `__i32`-mangled fallback for bare generic calls.

The parse-side ordering — monomorphization first, then grad — is fixed in the parser:

**Fragment** (the parse-side pass sequence; excerpt of
[`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx)):

```helix
    // Stage 8: monomorphization pass. Walk mr_tab; for each registered
    // (orig_name, mangled_name, pack_lo) entry, find the original
    // AST_FN_DECL template in the fn_list (matching by name) and build
    // a concrete clone with type substitution applied to params + ret.
    // ...
    monomorphize_pass(sb, head);
    // Stage 12: grad pass. Walk grad_pending; for each registered
    // (loss_name, mang_name) entry, find the loss fn in the fn_list,
    // ...
    grad_pass(sb, head);
```

> **For AI agents:** these passes are **append-only over the AST** — they add concrete clones to the
> end of the function list and never mutate the original template node in place. That append-only,
> name-keyed shape is what makes the lowering reproducible. If you add a pass, append; do not reorder
> existing nodes or depend on iteration order beyond the linked-list spine.

---

## Monomorphization, step by step

Monomorphization is the H2 work of the v1.1 hardening — making `<T>` produce *real per-type codegen*
instead of being parsed-and-erased. The design that guided it is
[`docs/HELIX_V1_1_H2_GENERICS.md`](../../../docs/HELIX_V1_1_H2_GENERICS.md); the gate-true capability
is **turbofish-directed monomorphization** (an explicit `id::<f32>(…)` / `Box::<f32>{…}` round-trips;
bare non-`i32` inference does not) — see
[`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §2 (the generics row) and
the residuals in Part III ch04. Here is the machinery, in the order data flows through it.

### 1. Encode the concrete type as a tag — `ty_ident_to_tag`

Every concrete scalar type is reduced to a small integer **type tag** by `ty_ident_to_tag`. This
single function is the type vocabulary the whole pipeline shares: `i32 → 0`, `f32 → 1`, `f64 → 2`,
`i64 → 3`, plus the widths added by later audits. It works directly on the raw identifier bytes in
the arena (there is no string type in the bootstrap), branching on length and ASCII byte values:

**Fragment** (the type-tag vocabulary; excerpt of
[`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx)):

```helix
@pure
fn ty_ident_to_tag(ty_s: i32, ty_l: i32) -> i32 {
    if ty_l == 3 {
        let b0 = __arena_get(ty_s);
        let b1 = __arena_get(ty_s + 1);
        let b2 = __arena_get(ty_s + 2);
        if b0 == 102 {
            if b1 == 54 { if b2 == 52 { 2 } else { 0 } }                              // f64
            else { if b1 == 51 { if b2 == 50 { 1 } else { 0 } }                       // f32
            else { if b1 == 49 { if b2 == 54 { 5 } else { 0 } } else { 0 } } }        // v1.3 f16 GAP FIX: f16 (f-1-6) -> tag 5 (reaches emit_f16_binop F16C path)
        } else { if b0 == 105 {
            if b1 == 54 { if b2 == 52 { 3 } else { 0 } }                              // i64
            else { if b1 == 51 { if b2 == 50 { 0 } else { 0 } }                       // i32
            else { if b1 == 49 { if b2 == 54 { 11 } else { 0 } } else { 0 } } }       // i16
```

Note the `@pure` annotation and the all-positive integer arithmetic: this function must behave
identically every run, and it avoids any operation the `i32`-only `seed` cannot self-compile. The
in-source audit note above this function (in `parser.hx`) records why the *missing* tags were a real
bug: before they were added, `u8`/`u16`/`i8`/`i16`/`bf16` all silently mapped to `0` (i32), so
`id::<u8>(…)` and `id::<i32>(…)` produced the **same** packed key and the request table de-duplicated
them into one wrong instantiation. Getting the tag vocabulary exhaustive is a correctness
prerequisite for the dedup logic below.

### 2. Mangle the instance name — `mangle_name_into_arena`

A concrete instance needs a unique name. `mangle_name_into_arena` builds it directly into the arena:
the original function name, then `__`, then each type-argument identifier separated by `_` — so
`id` at `f32` becomes `id__f32`, and a two-arg `pair` at `i32, f64` becomes `pair__i32_f64`:

**Fragment** (name mangling into the arena; excerpt of
[`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx)):

```helix
fn mangle_name_into_arena(orig_s: i32, orig_l: i32, ta_arr_base: i32, ta_count: i32) -> i32 {
    let start = __arena_len();
    // Copy orig name.
    let mut i: i32 = 0;
    while i < orig_l {
        __arena_push(__arena_get(orig_s + i));
        i = i + 1;
    }
    // Append "__".
    __arena_push(95);
    __arena_push(95);
    // Append each type-arg ident, separated by '_'.
    let mut j: i32 = 0;
    while j < ta_count {
        if j > 0 { __arena_push(95); };  // '_' separator
        let ts = __arena_get(ta_arr_base + j * 2);
        let tl = __arena_get(ta_arr_base + j * 2 + 1);
        let mut bb: i32 = 0;
        while bb < tl {
            __arena_push(__arena_get(ts + bb));
            bb = bb + 1;
        }
        j = j + 1;
    }
    start
}
```

The mangled name is the link between the use site (which emits a call to `id__f32`) and the synthesized
clone (which is registered under `id__f32`). The exact byte sequence — `__` joiner, single-`_`
argument separator — is a contract the backpatch fallback later relies on.

### 3. Record the request — `mr_tab`

Each distinct instantiation is one row of the **monomorph-request table**, `mr_tab`. A row is six
arena slots: original name (start, len), mangled name (start, len), a packed `pack_lo`, and a reserved
slot. `pack_lo` cleverly folds *two* values into one `i32` to stay within the bootstrap's six-integer
argument limit — the low 3 bits are the argument **count**, the upper bits are the 4-bit-per-arg
**packed tag vector**:

**Fragment** (the `mr_tab` row encoding, from the table's header comment; excerpt of
[`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx)):

```helix
//   slot 4: pack_lo = type_args_packed * 8 + type_args_count
//           (low 3 bits = count, upper = 4-bit-per-arg packed tags)
//   slot 5: reserved (currently 0)
// Combining packed+count into a single i32 avoids the 7-arg fn limit
// (SysV bootstrap supports 6 int params).
```

Requests are added with `mr_tab_add` and **de-duplicated** with `mr_tab_lookup` keyed on
`(orig_name, pack_lo)` — so a program that calls `id::<f32>(…)` five times produces exactly one
`id__f32` clone. The table caps at 32 entries and returns `-1` on overflow (a real bound, audited):

**Fragment** (`mr_tab_add` with its cap and de-dup contract; excerpt of
[`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx)):

```helix
fn mr_tab_add(sb: i32, orig_s: i32, orig_l: i32, mang_s: i32, mang_l: i32, pack_lo: i32) -> i32 {
    let count = mr_tab_count(sb);
    if count >= 32 {
        0 - 1
    } else {
        let base = mr_tab_base(sb);
        let entry = base + count * 6;
        __arena_set(entry, orig_s);
        __arena_set(entry + 1, orig_l);
        __arena_set(entry + 2, mang_s);
        __arena_set(entry + 3, mang_l);
        __arena_set(entry + 4, pack_lo);
        __arena_set(entry + 5, 0);
        __arena_set(sb + 32, count + 1);
        count
    }
}
```

### 4. Synthesize the clones — `monomorphize_pass`

After parsing, `monomorphize_pass` does two things. First it **synthesizes a default `i32`
instantiation** for any generic template that lacks one, so a bare `id(42)` resolves (this default is
the origin of the bare-call `i32` behavior documented as a residual in Part III ch04). Then it walks
the `mr_tab` rows, finds the matching template by name, and appends a concrete `AST_FN_DECL` clone for
each. The substitution is the heart of it: a parameter whose type is a generic marker (`>= 200`) has
its `200 + gp_idx` decoded and the `g_idx`-th 4-bit slot pulled out of the packed tag vector:

**Fragment** (parameter type substitution inside the clone loop; excerpt of
[`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx)):

```helix
                    let p_ty_raw = __arena_get(t_p_cur + 4);
                    // Substitute generic markers.
                    let new_p_ty = if p_ty_raw >= 200 {
                        let g_idx = p_ty_raw - 200;
                        // Extract g_idx-th 4-bit slot from packed.
                        let mut shifted: i32 = packed;
                        let mut sk: i32 = 0;
                        while sk < g_idx { shifted = shifted / 16; sk = sk + 1; }
                        shifted - (shifted / 16) * 16
                    } else { p_ty_raw };
```

The return type (`AST_FN_DECL` slot 5) gets the same `>= 200` substitution. The body is handled by
`clone_with_rewrite`, which deep-copies the body subtree and rewrites `AST_CALL` names whose prefix
matches a generic-parameter name (so a `T::eq(…)` typed-call in a template body becomes `i32__eq` /
`f32__eq` in the concrete clone), sharing leaf nodes that do not depend on the type parameter:

**Fragment** (the body clone-and-rewrite step; excerpt of
[`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx)):

```helix
                // Stage 8.5C: deep-clone-with-rewrite of the template body.
                // For generic fns whose body uses `T::eq(...)` typed-calls, the
                // call's mangled name contains the gp name as prefix (e.g.
                // "T__eq"). The clone walks the body subtree, copying nodes,
                // and rewrites AST_CALL names whose prefix matches a gp name
                // to use the concrete type's name (e.g. "i32__eq"). For
                // non-call subtrees the clone shares leaf nodes (AST_VAR/INT)
                // since they don't depend on gp.
                let tpl_gp_head = __arena_get(tpl_idx + 7);
                let cloned_body = if tpl_gp_head == 0 {
                    tpl_body
                } else {
                    clone_with_rewrite(tpl_body, tpl_gp_head, packed)
                };
```

The clone also propagates the template's attribute slots (`@checkpoint`, `@deprecated`, `@trace`,
`@unwind`) so the validation and autodiff passes observe them on the concrete instance, not just the
template — a real bug fixed in the H2 work (a `@checkpoint fn step[T](…)` that lost its checkpoint
marker on monomorphization made the reverse-mode AD pass grow memory linearly instead of `sqrt(N)`).

---

## The codegen side: skip the template, select by type, backpatch the call

Once the clones exist, the back end (`emit_elf_for_ast_to_path` in `kovc.hx`) does three monomorphization-relevant things.

**It skips templates.** Walking the function list to emit code, any `AST_FN_DECL` with slot 6 == 1 is
emitted as nothing — only the concrete clones get bytes:

**Fragment** (the emit-side template skip; excerpt of
[`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx)):

```helix
        // Stage 8: skip emission of generic-template fn decls (slot 6 == 1).
        // Their concrete clones are appended to the same fn_list by the
        // mono pass and will be emitted normally on a later iteration.
        let mut cur_list: i32 = ast_root;
        while cur_list != 0 {
            let fn_idx = __arena_get(cur_list + 1);
            let fn_is_generic = __arena_get(fn_idx + 6);
            if fn_is_generic == 1 {
                // skip — emit nothing for the template
            } else {
```

**It selects instructions by operand type.** This is *why* monomorphization needed "~zero new
op-emission" ([`docs/HELIX_V1_1_H2_GENERICS.md`](../../../docs/HELIX_V1_1_H2_GENERICS.md), "Honest
scope"): the existing arithmetic cascade already chooses the machine instruction from the operands'
type tags. Once a clone has concrete parameter types, an `a + b` in its body lowers correctly with no
generic-specific code. The `AST_ADD` dispatch (tag 2) is a five-way cascade — `i64`/`u64` → 64-bit
`add rax, rcx`; `f64` → `addsd`; `f32` → `addss`; `i32` → 32-bit `add eax, ecx`; mixed →
fail-closed trap. The `f32` leg, verbatim:

**Fragment** (the `f32`/`i32` leg of the type-driven `AST_ADD` cascade; excerpt of
[`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx)):

```helix
                        if l_f == 1 {
                            if r_f == 1 { emit_addss() } else { emit_trap_with_id(2040) }
                        } else {
                            if r_f == 1 { emit_trap_with_id(2041) } else { emit_add_eax_ecx() }
                        }
```

So the `f32` clone of a generic adder emits `addss` and the `i32` clone emits `add eax, ecx`, and a
type *mismatch* traps loudly (`2040`/`2041`) rather than emitting an integer add over float bits.
This type-driven discipline is the whole reason a single template can specialize correctly across
scalar types.

**It resolves calls by backpatch, with an `__i32` fallback.** A call site emitted before its target's
address is known leaves a 5-byte `call` placeholder; after all functions are laid down, the backpatch
loop resolves each by name against `fn_table`. On a miss it retries the **`__i32`-mangled** name —
this is what lets a bare `id(42)` resolve to the default `id__i32` clone the mono pass synthesized:

**Fragment** (the K1.F21 `__i32` backpatch fallback; excerpt of
[`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx)):

```helix
            // K1.F21 (2026-05-27): on lookup miss, try the i32-monomorphized
            // mangled name `<target>__i32`. Closes matrix-row-137 limitation:
            // bare call `id(42)` -- where mono produced `id__i32` from a
            // turbofish call elsewhere -- previously SIGILL'd (returns
            // rc=132); now resolves to the i32 mono clone. Non-i32 T bare
            // calls still trap (the fallback assumes T=i32 default).
```

If even the fallback misses, the loop overwrites the placeholder with `ud2` (SIGILL) — an immediate,
loud failure rather than a wild jump. That fail-closed posture is the same discipline the whole
project is built on: the bare non-`i32` generic case yields a documented bound, never a silent-wrong
result.

---

## A monomorphization round-trip, end to end (verified)

Putting the pieces together: the following committed program declares a generic `Box[T]`, instantiates
it at `f32` with a turbofish, and calls a method on the instance. The parser records a request keyed
on `(Box, f32)`, mangles a concrete struct/method name, the mono pass appends a concrete clone, the
back end skips the `[T]` template and emits the clone, and the call backpatches to the mangled
instance — and the program exits with the method's value.

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/gen_concrete_on_mono.hx`](../../../stage0/helixc-bootstrap/corpus_gen/gen_concrete_on_mono.hx)
(the gate compiles + runs it on the self-hosted K2 and asserts exit `7`:
`chk "$GENC/gen_concrete_on_mono.hx" 7`, [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`):

```helix
struct Box[T] { v: T }
impl<T> Box<T> { fn area(self) -> i32 { 7 } }
fn main() -> i32 {
    let a = Box::<f32>{ v: 5.0 };
    a.area()
}
```

The exit code `7` is the gate's witness that the `Box::<f32>` instantiation routed through the
monomorphization machinery and the mangled `area` clone was emitted and called. This is the same
fixture the language spec cites for "the concrete type round-trips"
([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §2, generics row). For the
per-type *arithmetic* correctness this pass enables — an `f32` clone doing float math while an `i32`
clone does integer math, in one program — see the gate-proven `gen_pair_multi.hx` (→ `12`) and the
`gen_vec_i32.hx`/`gen_vec_f32.hx` pair quoted in Part III
[ch04](../part3-language/04-generics-traits-closures.md); this chapter does not duplicate them.

The opposite boundary is also gated, as a *bound-proving* row:
[`M5_bare_generic_bound.hx`](../../../stage0/helixc-bootstrap/corpus_gen/M5_bare_generic_bound.hx)
asserts exit `0` (`chk "$GENC/M5_bare_generic_bound.hx" 0`) — a **bare** non-`i32` generic resolves to
the `i32` default and yields `0`, **not** `3`. Were type inference ever added, that row would start
returning `3` and the gate would fail, deliberately signalling the bound changed. The full program and
the residual it locks are in Part III ch04.

---

## Determinism is the fixpoint constraint

Every pass in this chapter must be **deterministic**, and that is not a style preference — it is a
trust requirement. The self-host fixpoint is `seed → K1 → K2 → K3 → K4` with **K2 == K3 == K4
byte-identical** (`0992dddd…`), pinned in the gate
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §1). For that to hold, `kovc`
compiling its own source must produce the *exact same bytes* every time. A pass that iterated a hash
map in nondeterministic order, or whose output depended on uninitialized arena memory, would break
byte-identity and collapse the fixpoint. The passes here avoid that by construction: they walk
**linked-list spines** in source order, key lookups on byte-equal names, fold packed integers with
plain positive arithmetic, and append (never reorder).

Monomorphization has a second, sharper relationship to the fixpoint, and it is the reason the H2 work
could be staged safely. The monomorphizer **only ever fires for generic templates**, and the
self-host sources — [`helixc/bootstrap/{lexer,parser,kovc}.hx`](../../../helixc/bootstrap/) — contain
**zero** turbofish and **zero** generic `fn`/`struct` declarations
([`docs/HELIX_V1_1_H2_GENERICS.md`](../../../docs/HELIX_V1_1_H2_GENERICS.md), "Fixpoint-safety map").
So when `kovc` compiles itself, the request table is empty, `monomorphize_pass` synthesizes nothing,
and the emitted bytes are unchanged. The generic-function and generic-`impl` work is therefore
**fixpoint-safe by construction** and could be gated by the feature corpus alone while `K2 == K3 == K4`
stayed byte-identical throughout.

> **Residual.** That fixpoint-safety-by-construction argument covers the generic-*only* code paths.
> It does **not** cover generic *struct fields* (gap **d.2** in the H2 design): carrying a scalar tag
> through a struct field touches the *non-generic* struct codegen the self-host source does use, so
> that piece was put through the **full** self-host fixpoint gate, not just a probe
> ([`docs/HELIX_V1_1_H2_GENERICS.md`](../../../docs/HELIX_V1_1_H2_GENERICS.md), "Fixpoint-safety map";
> [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §1). And honest about what
> monomorphization is *not*: it is **turbofish-directed**, not full inference — a bare non-`i32`
> generic defaults to `i32` (locked by `M5_bare_generic_bound`), and 8-byte generic struct fields
> (`f64`/`i64` *as a type parameter*) are deferred while the 4-byte case is gated (Part III ch04).

> **For AI agents:** before changing any pass in `parser.hx`/`kovc.hx`, treat byte-identity as the
> acceptance test. The standing proof is the gate, [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh),
> which reproduces the fixpoint (`K2 == K3 == K4`) **and** runs the 109-program feature corpus; it
> prints `GATE_PASS` only if both hold. A change that touches code paths the self-host source uses
> (e.g. generic struct fields, the `AST_ADD` cascade) MUST pass the full gate, not just the corpus —
> if it perturbs the pinned `0992dddd…` fixpoint, it is wrong, no matter how clean the diff looks.

---

**Next:** [The x86-64 ELF back end](03-x86-backend.md) — how `kovc` turns the lowered AST into a
runnable ELF: instruction encoding, the `_start` stub, the `fn_table`/`patch_table` backpatch
machinery these passes feed, and the BSS-allocated arena.
