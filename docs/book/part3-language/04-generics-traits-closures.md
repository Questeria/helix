# Generics, traits & closures

*What this chapter covers:* how Helix gives you generic functions, structs and `impl` methods that
emit **real per-type machine code** by *monomorphization* (the v1.1 **H2** hardening — what the
language spec calls the single biggest gap, now closed for explicitly-instantiated generics); how
**turbofish** (`id::<i32>(…)`, `Box::<f32>{…}`) directs that instantiation and disambiguates a type
argument; how `trait`/`impl` dispatch works, including **default methods**; and how **closures**
(non-capturing fn-pointers and v1.3 capturing closure objects) compile and capture. Every runnable
program here is quoted verbatim from a `.hx` file that the gate compiles **and runs** on the
self-hosted compiler, with its asserted exit code cited. The honest residuals — what is *not* yet
inferred, checked, or widened — are stated as plainly as the capabilities.

This is the fourth chapter of Part III. It builds on the type system (the type tags `i32`/`f32`/
`f64`/`i64` that monomorphization substitutes are introduced in the Types chapter) and on functions,
`impl` methods and `match` (the Functions, control flow & pattern matching chapter). For the
compiler-internals view of how `kovc` lexes and parses these forms, see Part V; this chapter is the
*language* view.

---

## The shape of the feature: monomorphization, not erasure

A generic in Helix is written with a type parameter in square brackets on the declaration and
applied with a **turbofish** (angle brackets after `::`) at the use site:

**Fragment** (illustrates the two syntactic halves; excerpt of
[`stage0/helixc-bootstrap/corpus_gen/gen_id_i32.hx`](../../../stage0/helixc-bootstrap/corpus_gen/gen_id_i32.hx)):

```helix
fn id[T](x: T) -> T { x }
fn main() -> i32 { id::<i32>(42) }
```

The declaration `fn id[T](…)` is a **template**: it is never emitted as code by itself. The use site
`id::<i32>(42)` is an **instantiation request**: it says "I need `id` specialised so that `T` is
`i32`." `kovc` resolves this by *monomorphization* — for each distinct concrete type a template is
asked for, it synthesises a separate, fully-typed copy of the function and emits that. There is no
type-erased single body, no boxing, and no runtime type tag: the f32 copy of a generic uses SSE
float instructions, the i32 copy uses integer instructions, and they are *different functions in the
binary*.

This is the headline of the **H2** hardening. The v1.0 language carried `<T>` only as
*parsed-and-erased* syntax — the spec called it "the most significant gap"
([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §7;
[`docs/HELIX_V1_1_HARDENING.md`](../../../docs/HELIX_V1_1_HARDENING.md), criterion H2). v1.1 made `<T>`
produce real codegen; the design that guided the work is
[`docs/HELIX_V1_1_H2_GENERICS.md`](../../../docs/HELIX_V1_1_H2_GENERICS.md).

> **For AI agents:** the precise, gate-true capability is **turbofish-directed monomorphization**.
> An *explicit* instantiation (`id::<i32>`, `Box::<f32>{…}`, `Opt::<i32>::Some(…)`) emits a concrete
> mangled instance and round-trips. A **bare** call that would require *inferring* a non-`i32` type
> argument is **not** inferred — it defaults `T → i32` (see "The honest residuals" below). Do not
> generate generic Helix code that relies on inferring a differing element type without a turbofish;
> it will compile to the wrong thing fail-closed, not error. This is grounded in
> [`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §2 (generics row) and §7.

---

## How `kovc` monomorphizes, concretely

Monomorphization in `kovc` is a **parser-side pass that runs before codegen**, not a backend trick.
It is worth understanding mechanically, because the way it works is exactly *why* it is safe for the
self-host fixpoint (the AI callout at the end of this section).

The pipeline, grounded in [`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx) and
[`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx):

1. **Record a request at each turbofish use site.** When the parser sees `id::<f32>(…)` (or the
   `id::[f32](…)` bracket form), it computes a **mangled name** — `id` plus a per-type suffix, e.g.
   `id__f32` — via `mangle_name_into_arena` (`parser.hx`), packs the concrete type arguments into a
   small 4-bit-per-arg tag vector via `ty_ident_to_tag` (`parser.hx`, where `i32→0`, `f32→1`,
   `f64→2`, `i64→3`, …), and stores the triple in a **monomorph-request table**, `mr_tab`
   (`mr_tab_add`, `parser.hx`). Duplicate requests for the same name+types are de-duplicated
   (`mr_tab_lookup`).

2. **Mark the template.** A generic `fn id[T](…)` declaration is flagged as a template (its
   `AST_FN_DECL` carries a "is generic" slot, slot 6 == 1 — described in
   [`docs/HELIX_V1_1_H2_GENERICS.md`](../../../docs/HELIX_V1_1_H2_GENERICS.md)).

3. **Synthesise one concrete clone per request.** The `monomorphize_pass`
   ([`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx), with the in-source comment
   "*For each `mr_tab` entry, find the matching generic fn template … and synthesise a concrete
   clone*") walks the request table, finds the template, and appends a new function node to the end
   of the program's function list with the **parameter type tags and the return type substituted**
   to the requested concrete types. It also synthesises a default `i32` instantiation for each
   template so a plain `id(42)` resolves (this default is the source of the bare-call `i32` behaviour
   discussed under the residuals).

4. **Skip the template, emit the clones.** Codegen in
   [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) **skips** any function still
   marked as a template (it emits nothing for the un-instantiated `id[T]`), and emits each concrete
   clone, registering it in the `fn_table` under its mangled name.

5. **Resolve calls by backpatch.** A call site emitted as `id::<f32>(…)` is recorded as a pending
   call to `id__f32`; the backpatch loop in `kovc.hx` patches the real address once the clone has
   been laid down, with a documented `__i32` end-suffix fallback for the bare-call/default case.

The single most important property of monomorphization for op-selection is that **once a clone has
concrete parameter types, the existing type-driven backend does the rest**. `kovc` already chooses
the instruction by operand type — an add lowers to `addss` for `f32`, integer `add` for `i32`,
`addsd` for `f64`, and a REX.W 64-bit add for `i64`
([`docs/HELIX_V1_1_H2_GENERICS.md`](../../../docs/HELIX_V1_1_H2_GENERICS.md), citing `kovc.hx`'s `AST_ADD`
emission). So an `f32` clone of a generic that does `a + b` *automatically* gets float arithmetic,
with no new backend code. As the H2 design puts it, the backend needed "~zero new op-emission"; the
work was the parser-side clone-and-substitute.

> **For AI agents:** monomorphization is **fixpoint-safe by construction for the generic-only
> paths**, and this is load-bearing for trust. The clone-and-substitute machinery is reached **only**
> for functions that are generic templates. The self-host source —
> [`helixc/bootstrap/{lexer,parser,kovc}.hx`](../../../helixc/bootstrap/) — contains **zero**
> turbofish and **zero** generic fn/struct declarations
> ([`docs/HELIX_V1_1_H2_GENERICS.md`](../../../docs/HELIX_V1_1_H2_GENERICS.md), "Fixpoint-safety map"), so
> when `kovc` compiles itself the monomorphizer never fires and the emitted bytes are unchanged.
> That is why the generic-function and generic-impl work could be gated by the feature corpus while
> the self-host fixpoint `K2 == K3 == K4` stayed byte-identical throughout
> ([`docs/HELIX_V1_1_HARDENING.md`](../../../docs/HELIX_V1_1_HARDENING.md), H2). The one part that is **not**
> fixpoint-safe-by-construction is generic *struct fields* (it touches struct codegen the self-host
> source does use), which is why that work was put through the **full** fixpoint gate, not just a
> probe — see the next section.

---

## Generic functions over differing scalar types

The first thing monomorphization buys you is a single generic function used at more than one concrete
scalar type, each correct. The H2 probe corpus (recorded in
[`docs/HELIX_V1_1_H2_GENERICS.md`](../../../docs/HELIX_V1_1_H2_GENERICS.md)) established that generic function
*bodies* are typed correctly per instantiation: an identity at `i32` returns its integer argument; a
generic `add2::<f32>(a, b)` emits `addss` and computes the float sum; one generic function used at
**both** `i32` and `f32` in the same program produces two correct clones.

**Fragment** (a generic function instantiated at two scalar types; excerpt of
[`stage0/helixc-bootstrap/corpus_gen/gen_two_types.hx`](../../../stage0/helixc-bootstrap/corpus_gen/gen_two_types.hx)):

```helix
fn id[T](x: T) -> T { x }
fn main() -> i32 {
    let a = id::<i32>(40);
    let b: f32 = id::<f32>(2.0_f32);
    a + (b as i32)
}
```

> **Note:** `gen_two_types.hx`, `gen_id_i32.hx` and `gen_add_f32.hx` are the H2 **probe** fixtures
> that *characterised* this behaviour; their results are recorded in
> [`docs/HELIX_V1_1_H2_GENERICS.md`](../../../docs/HELIX_V1_1_H2_GENERICS.md) but they are not themselves
> rows in the standing gate. They are shown here as **Fragments**, illustrating the syntax. The
> *gate-proven* demonstrations of the same capability are the generic-`impl` and container programs
> below, which the gate compiles and runs with asserted exit codes.

The generic-`impl` programs the gate *does* assert show the same per-type correctness through a
method. The clearest is a generic `Box[T]` whose method returns the bare type parameter `T`,
instantiated at `f32`:

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/gen_impl_t_single_f32.hx`](../../../stage0/helixc-bootstrap/corpus_gen/gen_impl_t_single_f32.hx)
(the gate compiles + runs it on the self-hosted K2 and asserts exit `5`:
`chk "$GENC/gen_impl_t_single_f32.hx" 5`, [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`):

```helix
struct Box[T] { v: T }
impl<T> Box<T> { fn get(self) -> T { self.v } }
fn main() -> i32 {
    let a = Box::<f32>{ v: 5.0 };
    let s: f32 = a.get();
    s as i32
}
```

Here `Box::<f32>` is the turbofish on a struct constructor, and `get(self) -> T` returns the bare
type parameter — so the `f32` instantiation must read an `f32`-typed field and return it as `f32`.
The program exits `5`, which is the gate's proof that the `f32` field and the `T`-typed return were
monomorphized correctly rather than being read back as `i32` bits. A sibling fixture,
[`gen_impl_ret_f32.hx`](../../../stage0/helixc-bootstrap/corpus_gen/gen_impl_ret_f32.hx)
(`chk … 5`), pins the same value through a method declared with a concrete `-> f32` return.

---

## Generic structs and containers: `Box<T>`, `Pair<T>`, `Vec<T>`

Carrying a type parameter through a *struct field* is the harder half of H2, because the field
storage and the field load/store had to learn the element's scalar type rather than treating every
field as an `i32`-shaped slot. The H2 design singled this out as the part that is **not**
fixpoint-safe by construction (it touches non-generic struct codegen the self-host source uses), so
it was gated through the full self-host fixpoint
([`docs/HELIX_V1_1_H2_GENERICS.md`](../../../docs/HELIX_V1_1_H2_GENERICS.md), gap **d.2**;
[`docs/HELIX_V1_1_HARDENING.md`](../../../docs/HELIX_V1_1_HARDENING.md), H2). The payoff is generic
containers over more than one element type.

A `Pair[T]` with two `T` fields and two methods, instantiated at **both** `i32` and `f32` in one
program, exercises generic struct fields *and* generic methods at two types simultaneously:

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/gen_pair_multi.hx`](../../../stage0/helixc-bootstrap/corpus_gen/gen_pair_multi.hx)
(gate asserts exit `12`: `chk "$GENC/gen_pair_multi.hx" 12`,
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`):

```helix
struct Pair[T] { a: T, b: T }
impl<T> Pair<T> {
    fn first(self) -> T { self.a }
    fn second(self) -> T { self.b }
}
fn main() -> i32 {
    let pi = Pair::<i32>{ a: 3, b: 4 };
    let pf = Pair::<f32>{ a: 2.0, b: 3.0 };
    let si: i32 = pi.first() + pi.second();
    let sf: f32 = pf.first() + pf.second();
    si + (sf as i32)
}
```

The `i32` pair sums to `7` and the `f32` pair to `5.0`; `7 + 5 == 12`. The exit code is the gate's
witness that the two instantiations did **not** collapse onto one shape — the `f32` fields were read
through float arithmetic and the `i32` fields through integer arithmetic.

The charter's headline container is `Vec<T>`. The committed fixtures back it with an arena-backed
vector — `kovc`'s runtime heap is the *arena* (one `i32` slot per element; see the Builtins section
of the spec), and `Vec[T]` is a thin generic view over an arena region:

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/gen_vec_i32.hx`](../../../stage0/helixc-bootstrap/corpus_gen/gen_vec_i32.hx)
(gate asserts exit `42`: `chk "$GENC/gen_vec_i32.hx" 42`,
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`):

```helix
struct Vec[T] { base: i32, len: i32 }
impl<T> Vec<T> {
    fn at(self, i: i32) -> T { __arena_get(self.base + i) }
}
fn main() -> i32 {
    let b = __arena_len();
    __arena_push(10);
    __arena_push(20);
    __arena_push(12);
    let v = Vec::<i32>{ base: b, len: 3 };
    v.at(0) + v.at(1) + v.at(2)
}
```

`10 + 20 + 12 == 42`. The `f32` instantiation of the *same* `Vec[T]` is a separate gated fixture,
[`gen_vec_f32.hx`](../../../stage0/helixc-bootstrap/corpus_gen/gen_vec_f32.hx) (`chk … 5`), which
pushes `2.0` and `3.0` and reads them back through `Vec::<f32>`'s `at` (the H2 record notes this
includes an "f32-through-i32-arena round-trip" — the float bits survive the `i32`-slot arena because
`at` returns `T`):

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/gen_vec_f32.hx`](../../../stage0/helixc-bootstrap/corpus_gen/gen_vec_f32.hx)
(gate asserts exit `5`: `chk "$GENC/gen_vec_f32.hx" 5`,
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`):

```helix
struct Vec[T] { base: i32, len: i32 }
impl<T> Vec<T> {
    fn at(self, i: i32) -> T { __arena_get(self.base + i) }
}
fn main() -> i32 {
    let b = __arena_len();
    let x: f32 = 2.0;
    let y: f32 = 3.0;
    __arena_push(x);
    __arena_push(y);
    let v = Vec::<f32>{ base: b, len: 2 };
    let s: f32 = v.at(0) + v.at(1);
    s as i32
}
```

> **Note (square-bracket vs angle-bracket).** The two notations you see — `struct Vec[T]` on the
> declaration and `Vec::<i32>` at the use site — are both real and both gated. The H2 record notes
> "square-bracket generics across struct/enum/impl decls" were landed alongside the angle-bracket
> turbofish ([`docs/HELIX_V1_1_HARDENING.md`](../../../docs/HELIX_V1_1_HARDENING.md), H2). The fixtures use
> `[T]` on declarations and `::<T>` at call/construct sites; follow that pairing.

> **Residual (field width).** Generic struct fields are gated for **4-byte** element types (`i32`,
> `f32`). 8-byte generic struct fields (`f64`/`i64` *as a type parameter*) are listed as deferred,
> non-blocking, with the 4-byte case working
> ([`docs/HELIX_V1_1_HARDENING.md`](../../../docs/HELIX_V1_1_HARDENING.md), H2 "Deferred"). Note this is a
> distinct, narrower thing from a **non-generic** struct's `i64`/`u64`/`f64` field, which v1.3 (V1)
> made full-64-bit ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §2.1).

---

## Turbofish and disambiguation

The **turbofish** is the `::<…>` (or `::[…]`) that supplies a type argument explicitly. It does two
jobs in Helix, and both are concrete consequences of monomorphization:

1. **It selects the instantiation.** `id::<i32>` and `id::<f32>` are requests for two *different*
   emitted functions; the turbofish is how you name which one you mean. On a struct it is written on
   the constructor, `Box::<f32>{ v: 5.0 }` (see `gen_impl_t_single_f32.hx` above).

2. **It disambiguates a type argument that could not otherwise be inferred.** Because Helix infers
   a bare generic call as `i32` (the default instantiation the monomorphizer synthesises), the
   turbofish is *required* whenever the concrete type is not `i32`. That a missing turbofish at a
   non-`i32` type does **not** error but defaults to `i32` is a documented, **fail-closed-by-test**
   bound — see the next section.

The turbofish also works on enum constructors. v1.3 fixed `Opt::<i32>::Some(payload)` /
`Opt::<i32>::None` (which previously mis-routed and hung the compiler); it now routes to the same
construct path as the bare form, and is gated:
[`M4_turbofish_enum.hx`](../../../stage0/helixc-bootstrap/corpus_gen/M4_turbofish_enum.hx) → `42`
and [`gen_option_i32.hx`](../../../stage0/helixc-bootstrap/corpus_gen/gen_option_i32.hx) → `42`
([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`;
[`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §2).

> **For AI agents:** the gate-true rule is simple and worth keying off exactly. **For any non-`i32`
> scalar generic, write the turbofish.** `add2::<f32>(2.0_f32, 3.0_f32)` is supported; `add2(2.0,
> 3.0)` is not (it instantiates at `i32`). On enum constructors use either the bare `Opt::Some(x)` or
> the turbofish `Opt::<i32>::Some(x)` — both are gated; do not assume the turbofish-on-enum form is
> broken (the old hang is fixed).

---

## Traits and `impl`

A `trait` declares a set of method signatures; an `impl Trait for Type` provides them; and a method
call dispatches to the implementation for the receiver's static type. Because Helix resolves the
receiver type at compile time, trait dispatch is **static** — there is no vtable and no dynamic
dispatch object; the call goes straight to the right `impl` method. This codegen is the v1.1 **H3**
hardening ([`docs/HELIX_V1_1_HARDENING.md`](../../../docs/HELIX_V1_1_HARDENING.md), H3).

A single trait with one implementing struct:

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/t2_trait_impl.hx`](../../../stage0/helixc-bootstrap/corpus_gen/t2_trait_impl.hx)
(gate asserts exit `42`: `chk "$GENC/t2_trait_impl.hx" 42`,
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`):

```helix
trait Greet { fn hello(self) -> i32 }
struct P { x: i32 }
impl Greet for P { fn hello(self) -> i32 { self.x + 5 } }
fn main() -> i32 {
    let p = P { x: 37 };
    p.hello()
}
```

`37 + 5 == 42`. The polymorphic case — one trait, two implementing structs, each with its own method
body — is also gated. It is the proof that dispatch picks the *correct* `impl` per receiver type:

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/t7_trait_poly.hx`](../../../stage0/helixc-bootstrap/corpus_gen/t7_trait_poly.hx)
(gate asserts exit `42`: `chk "$GENC/t7_trait_poly.hx" 42`,
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`):

```helix
trait Shape { fn area(self) -> i32 }
struct Sq { s: i32 }
struct Rec { w: i32, h: i32 }
impl Shape for Sq { fn area(self) -> i32 { self.s * self.s } }
impl Shape for Rec { fn area(self) -> i32 { self.w * self.h } }
fn main() -> i32 {
    let sq = Sq { s: 5 };
    let r = Rec { w: 17, h: 1 };
    sq.area() + r.area()
}
```

`25 + 17 == 42`. The H3 work also caught and fixed a latent multi-`impl` bug where a second `impl`
method's `self.field` resolved against the *first* struct's layout; the fixtures
[`t7b_trait_2types.hx`](../../../stage0/helixc-bootstrap/corpus_gen/t7b_trait_2types.hx) (two types,
same field name) and [`t7c_difffields.hx`](../../../stage0/helixc-bootstrap/corpus_gen/t7c_difffields.hx)
(two types, *different* field names) are the gated regression for it (both → `42`,
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`;
[`docs/HELIX_V1_1_HARDENING.md`](../../../docs/HELIX_V1_1_HARDENING.md), H3).

### Default methods

A trait method may carry a **default body**. A type that implements the trait but does not override
that method dispatches to the default; a type that *does* override it uses its own. This was the last
of the v1.1 HIGH items and is gated (it had been deferred at H3's first landing, then implemented).
The default-used case:

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/t1_trait_default.hx`](../../../stage0/helixc-bootstrap/corpus_gen/t1_trait_default.hx)
(gate asserts exit `42`: `chk "$GENC/t1_trait_default.hx" 42`,
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`):

```helix
trait Greet { fn hello(self) -> i32 { 42 } }
struct P { x: i32 }
impl Greet for P {}
fn main() -> i32 {
    let p = P { x: 1 };
    p.hello()
}
```

The `impl Greet for P {}` is empty, so `p.hello()` dispatches to the trait's default body and returns
`42`. The default-**and**-override case proves override wins:

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/t5_trait_default_mix.hx`](../../../stage0/helixc-bootstrap/corpus_gen/t5_trait_default_mix.hx)
(gate asserts exit `42`: `chk "$GENC/t5_trait_default_mix.hx" 42`,
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`):

```helix
trait Greet { fn hello(self) -> i32 { 10 } }
struct A { x: i32 }
struct B { x: i32 }
impl Greet for A {}
impl Greet for B { fn hello(self) -> i32 { 32 } }
fn main() -> i32 {
    let a = A { x: 0 };
    let b = B { x: 0 };
    a.hello() + b.hello()
}
```

`A` takes the default (`10`); `B` overrides (`32`); `10 + 32 == 42`. Mechanically, the parser stores
each default-bodied method's token range and, for an `impl` that does not override the method,
re-parses that range as a concrete method of the implementing type (so `self.field`/`self.method()`
resolve against the concrete type), with an explicit override taking precedence
([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §8 H-4 record;
[`docs/HELIX_V1_1_HARDENING.md`](../../../docs/HELIX_V1_1_HARDENING.md), H3 "Deferred" — which is where it was
*before* this landed).

> **Residual (traits as an abstraction).** Traits give you method dispatch and default bodies; they
> are **not** a checked abstraction. There are no trait *bounds* on generics, no `dyn`/trait objects,
> and trait conformance is not type-checked as a constraint — the spec lists "no traits as a checked
> abstraction" among the honest residuals
> ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §7). Use traits to share method
> shape and supply defaults; do not write `fn f<T: Trait>(…)` expecting the bound to be enforced.

---

## Closures

A closure is written `|params| body`. Helix compiles closures in two forms, distinguished by whether
they capture anything from the enclosing scope. This is the closure half of **H3**, extended by v1.3
(**V3**) to make capturing closures first-class values
([`docs/HELIX_V1_1_HARDENING.md`](../../../docs/HELIX_V1_1_HARDENING.md), H3;
[`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §2.2).

A **non-capturing** closure compiles to a raw function pointer:

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/t3_closure_call.hx`](../../../stage0/helixc-bootstrap/corpus_gen/t3_closure_call.hx)
(gate asserts exit `42`: `chk "$GENC/t3_closure_call.hx" 42`,
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`):

```helix
fn main() -> i32 {
    let f = |x: i32| x + 1;
    f(41)
}
```

A **capturing** closure reads one or more bindings from its environment. The single-capture case:

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/t4_closure_capture.hx`](../../../stage0/helixc-bootstrap/corpus_gen/t4_closure_capture.hx)
(gate asserts exit `42`: `chk "$GENC/t4_closure_capture.hx" 42`,
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`):

```helix
fn main() -> i32 {
    let n = 40;
    let f = |x: i32| x + n;
    f(2)
}
```

`f` captures `n == 40`; `2 + 40 == 42`. Two captures work the same way:

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/t8_closure_two_caps.hx`](../../../stage0/helixc-bootstrap/corpus_gen/t8_closure_two_caps.hx)
(gate asserts exit `42`: `chk "$GENC/t8_closure_two_caps.hx" 42`,
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`):

```helix
fn main() -> i32 {
    let a = 30;
    let b = 10;
    let f = |x: i32| x + a + b;
    f(2)
}
```

### Capture semantics and passing closures as values

Under the hood, a capturing closure compiles to a real **closure object** in the runtime arena — the
cells `[code_ptr, cap0, cap1, …]` — and its runtime value is the object's arena index OR-ed with a
tag bit (`0x40000000`). Because that tagged index is a small positive `i32` (the arena lives at a low
`.data` address), a capturing closure **survives a by-value `i32` parameter**, so it can be passed as
an argument and invoked. The indirect-call dispatch tag-tests the value: bit-30 clear is a
non-capturing raw code pointer (an env-less call), bit-30 set is a capturing object (load the code
pointer from the arena, pass the env, call). This is the v1.3 V3 design
([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §2.2).

Two semantic points matter when you write closures:

- **Capture is by value, at creation.** Each captured local's value is *snapshotted* into the
  closure object when the `|…|` literal is evaluated. Mutating the original binding afterward does
  **not** change what the closure sees — this is gated by
  [`V3_modify_after.hx`](../../../stage0/helixc-bootstrap/corpus_gen/V3_modify_after.hx)
  ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §2.2). This is *not* Rust-style
  by-reference capture.
- **Passing a capturing closure by value works**, gated by
  [`V3_capture_arg.hx`](../../../stage0/helixc-bootstrap/corpus_gen/V3_capture_arg.hx) → `42`
  (a capturing closure passed to a higher-order fn and invoked;
  [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`).

> **Residual (capture width).** Captures are **`i32`-only**. A capture wider than `i32` (e.g. an
> `i64`/`f64`) would not fit a 4-byte arena cell, so it **fails closed at runtime with trap 76003** —
> it is *not* silently truncated ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md)
> §2.2; [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2.4). This fail-loud posture — no
> silent-wrong result — is the trust discipline the whole project is built on.

---

## The honest residuals

Helix's value is calibrated honesty, so the limits here are stated as plainly as the capabilities.
Each is locked by a real corpus row that proves the *exact* boundary
([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §7;
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2.4).

- **Generic type inference is not done; turbofish is required for non-`i32`.** What ships is
  *turbofish-directed* monomorphization. General monomorphization that **infers** a differing element
  type *without* an explicit turbofish is the residual. A bare non-`i32` scalar generic defaults
  `T → i32` rather than inferring — and this is **proved as a bound**, not hidden:

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/M5_bare_generic_bound.hx`](../../../stage0/helixc-bootstrap/corpus_gen/M5_bare_generic_bound.hx)
(a *negative/bound-proving* row: the gate asserts exit `0`, i.e. the bare form yields `0`, **not**
`3`: `chk "$GENC/M5_bare_generic_bound.hx" 0`, [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`):

```helix
fn id[T](x: T) -> T { x }
fn main() -> i32 {
    let r: f32 = id(3.0_f32);   // BARE generic at f32 -> T defaults to i32 -> 0
    r as i32                     // documented bound: yields 0, NOT 3
}
```

  The supported idiom for the same intent is the turbofish `id::<f32>(3.0_f32)`. Were inference ever
  added, this row would start returning `3` and the test would fail — deliberately signalling the
  bound has changed.

- **8-byte generic struct fields are deferred.** `f64`/`i64` *as a generic struct field's type
  parameter* is non-blocking-deferred; the 4-byte (`i32`/`f32`) case is gated and works
  ([`docs/HELIX_V1_1_HARDENING.md`](../../../docs/HELIX_V1_1_HARDENING.md), H2 "Deferred").
- **Traits are not a checked abstraction.** Dispatch and default methods work; trait *bounds*, trait
  objects, and conformance checking do not
  ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §7).
- **Closure captures are `i32`-only and fail-closed if wider** (trap 76003), as above.

None of these is a silent-wrong path: the bare-generic case is locked by a bound test, and the
width limits trap loudly. That is the line the gate holds — the v1.1 generics, traits and closures
work landed with the self-host fixpoint `K2 == K3 == K4` byte-identical throughout
([`docs/HELIX_V1_1_HARDENING.md`](../../../docs/HELIX_V1_1_HARDENING.md), H2/H3), and the corpus rows quoted
above are part of the standing 109-program gate
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §1).

---

**Next:** [Autodiff & the AGI-oriented features](05-autodiff-agi-features.md) — how `grad` and the
forward-mode derivative work, and which of the more ambitious type-system features are real versus
designed.
