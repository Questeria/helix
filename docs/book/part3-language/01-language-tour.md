# The language tour

*What this chapter covers: a fast, example-first feel for Helix — values and literals, `fn`
with last-expression return, `let` / `let mut` bindings, the `if` / `while` / `match` control
forms, and structs and enums at a glance — built entirely around real, gate-proven programs
from [`helixc/examples/`](../../../helixc/examples/). This is the on-ramp to Part III; the depth
(the full type system, generics, autodiff) comes in the chapters after it.*

---

## How to read this tour

Helix is a statically typed, expression-oriented language that compiles — via the from-raw
self-hosting compiler `kovc` — straight to a static x86-64 Linux ELF, with no LLVM and no
external assembler or linker. If you already write Rust, C, or Swift, most of this chapter will
feel familiar; the goal here is to get the *shape* of the language into your head quickly, not to
be exhaustive.

Every complete program below is quoted **verbatim** from a real file under
[`helixc/examples/`](../../../helixc/examples/), and almost all of them are pinned by the gate
([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh)), which compiles and runs each one on the
freshly self-hosted compiler and asserts its exact exit code. Where that is true, the example is
marked **Verified example** and the gate's assertion line is cited. Partial snippets are marked
**Fragment** and are excerpts of a cited file — they are not standalone programs.

This chapter does **not** re-explain how to compile and run a `.hx` file — that mechanics
(staging the source, the fixed `/tmp` paths, the exit-code convention) is covered in
[part2-setup-build/03-using-kovc.md](../part2-setup-build/03-using-kovc.md). Read it once and the
"compiles to exit *N*" claims here will be concrete for you.

> **For AI agents:** every "Verified example" in this chapter is a row in
> [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]` (the `chk "<file>" <exit>`
> calls) or an inline corpus program generated there. Treat the cited `chk` line as the ground
> truth for a program's exit code; do not infer exit codes from the prose. To reproduce, follow
> the compile-and-run loop quoted in
> [part2-setup-build/03-using-kovc.md](../part2-setup-build/03-using-kovc.md), not a guessed CLI.

---

## The smallest program

Every runnable Helix program is a `fn main() -> i32`, and the `i32` it returns becomes the
process exit code. The smallest one does nothing but return a number:

**Verified example** — [`helixc/examples/exit42.hx`](../../../helixc/examples/exit42.hx)
(compiles to a Linux ELF; runs and exits `42` — gate row `chk "$EX/exit42.hx" 42`,
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`):

```helix
// First end-to-end Helix program: compiles, links, runs, exits with status 42.
//
// Demonstrates: function decl, return, integer literal.
// Verification: $? == 42 after running the produced ELF.

fn main() -> i32 {
    42
}
```

Two language facts are already visible here, and both hold everywhere in Helix:

1. **The last expression of a block is its value.** There is no `return` keyword in this program;
   `42` is the final expression of `main`'s body, so it *is* what `main` returns. Helix does have
   `return` for early exit, but a tail expression never needs it. This block-value rule is the
   single most important thing to internalize about Helix's feel — it is in the as-built spec as
   "the trailing expression is the block's value"
   ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §4).
2. **Literals are typed.** `42` is an `i32` (the default integer type), which matches `main`'s
   declared `-> i32`. We will meet more literal forms in a moment.

> **Note — the 8-bit exit code.** A Linux exit status is one byte, so a returned `i32` surfaces
> **modulo 256**. That is why so many examples are engineered to return small sentinels like `42`
> or `69`: they survive the truncation and are unmistakable. The convention is explained in full
> in [part2-setup-build/03-using-kovc.md](../part2-setup-build/03-using-kovc.md).

---

## Values, `let`, and a little real arithmetic

Bindings use `let`. A binding is **immutable by default**; you opt into mutation with `let mut`.
The next example introduces local bindings and arithmetic by doing an actual (small) computation —
multiplying two 2×2 matrices with scalar `let`s and returning the trace of the product:

**Verified example** — [`helixc/examples/matmul_2x2.hx`](../../../helixc/examples/matmul_2x2.hx)
(compiles to a Linux ELF; runs and exits `69` — gate row `chk "$EX/matmul_2x2.hx" 69`,
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`):

```helix
// matmul_2x2.hx — 2x2 matrix multiplication via scalar ops.
//
// We don't have arrays + indexing yet, so this version operates on individual
// elements. The trace of A*B is computed and returned.
//
//   A = [[1, 2], [3, 4]]
//   B = [[5, 6], [7, 8]]
//   A*B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
//       = [[19, 22], [43, 50]]
//   trace(A*B) = 19 + 50 = 69

fn main() -> i32 {
    // We're capped at 3 args per fn in the v0.1 codegen, so call with
    // partial args via a wrapper. Actually let's compute directly inline:
    //   A = [[1, 2], [3, 4]]
    //   B = [[5, 6], [7, 8]]
    let a00 = 1; let a01 = 2; let a10 = 3; let a11 = 4;
    let b00 = 5; let b01 = 6; let b10 = 7; let b11 = 8;
    let c00 = a00 * b00 + a01 * b10;
    let c11 = a10 * b01 + a11 * b11;
    c00 + c11   // 19 + 50 = 69
}
```

Things to read off this program:

- **`let name = expr;`** introduces a binding; several can sit on one line separated by `;`. The
  trailing `c00 + c11` (no `;`, no `return`) is the block value — `19 + 50 = 69`.
- **Arithmetic is the usual `+ - * / %`**, and operators are **left-associative**: the spec pins
  this with corpus rows showing `10 - 3 - 2 → 5` (not 9) and `100 / 5 / 2 → 10` (not 40)
  ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §4). Multiplication
  binds tighter than addition, as you would expect, so `a00 * b00 + a01 * b10` is `(a00*b00) +
  (a01*b10)`.
- The in-source comments ("we don't have arrays + indexing yet", "capped at 3 args per fn") are
  *period notes* from when the file was written. The shipped `kovc` today has array indexing and
  many-argument functions — both are in the gate corpus. The program still compiles and returns
  `69` exactly as quoted; treat those comments as history, not current limits (this is also noted
  in [part2-setup-build/03-using-kovc.md](../part2-setup-build/03-using-kovc.md)).

### Mutation with `let mut`

When you need a cell you can reassign, use `let mut`. The clearest small example is the loop
program in the next section; here is the binding-and-reassignment shape on its own.

**Fragment** (excerpt of the gate's `while_sum` corpus program,
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`; not the whole program):

```helix
let mut s = 0; let mut i = 0;
// ... s and i are reassigned in the loop below with `s = s + i;` etc.
```

`let mut s = 0;` creates a mutable binding; plain assignment `s = s + i;` updates it. (An
immutable `let` cannot be reassigned.) Assignment is the proven core form — Helix does *not* have
compound-assignment operators (`+=`, `-=`, …) as a first-class control form; where you see `+=`
in older example files it is a parser desugar to `s = s + …`, exercised by the gate's
`M2_compound_assign` row, not a primitive
([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §1, §7).

---

## Literals at a glance

You have already seen `i32` integer literals. Helix's literal surface, as actually implemented by
`kovc` ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §1, §2), in
brief:

- **Integers** are `i32` by default; a width/sign suffix selects another type — e.g.
  `42_i64`, `0_u8`, `5_000_000_000_i64`. Underscores are legal digit separators, and `0x` / `0b`
  / `0o` give hex / binary / octal.
- **Floats** are `f32` by default (`3.0`), with suffixes `_f64`, `_bf16`, `_f16`. The Helix float
  story (IEEE-754, SSE codegen, and the 16-bit float types) is a Part III topic of its own.
- **`bool`** is `true` / `false` — but note Helix has *no* implicit int-to-bool coercion, so an
  `if` always takes an explicit comparison (see the control-flow section).
- **Casts** between numeric types are explicit with `as`, e.g. `(x_new + 39.0) as i32` in the next
  example.

A small complete program showing the typed-suffix forms, taken from the gate's own inline corpus:

**Verified example** — the gate's `i64_basic` corpus program
([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`; generated and run by the gate,
which asserts exit `42` with `chk "$CD/i64_basic.hx" 42`):

```helix
fn main() -> i32 { let x: i64 = 42_i64; x as i32 }
```

This one program shows three things at once: an **explicit type annotation** on a `let`
(`let x: i64 = …`), a **typed integer literal** (`42_i64`), and an **`as` cast** back to `i32`
for the return. It compiles and exits `42`.

---

## Functions

A function is `fn name(params) -> ReturnType { body }`. Parameters are typed; the body is a single
(block) expression whose trailing expression is the return value. Recursion is fully supported.

The cleanest small function example does one step of gradient descent — and incidentally shows a
real Helix differentiator, the source-level `grad` operator — while staying a complete, runnable
program:

**Verified example** — [`helixc/examples/gradient_descent.hx`](../../../helixc/examples/gradient_descent.hx)
(compiles to a Linux ELF; runs and exits `42` — gate row `chk "$EX/gradient_descent.hx" 42`,
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`):

```helix
// gradient_descent.hx — one step of gradient descent in Helix.
//
// We minimize loss(x) = (x - 3)^2  via gradient descent.
// d(loss)/dx = 2*(x - 3)
//
// Starting at x = 0, with learning rate 0.5:
//   gradient = 2*(0 - 3) = -6
//   step     = -0.5 * (-6) = 3
//   x_new    = 0 + 3 = 3   (= optimum, since loss is minimized at x=3)
//
// We verify: x_new should be 3. Add 39 for exit code 42.

fn loss(x: f32) -> f32 {
    let diff = x - 3.0;
    diff * diff
}

fn main() -> i32 {
    let x = 0.0;
    let lr = 0.5;
    // grad(loss)(x) = d(loss)/dx evaluated at x
    let g = grad(loss)(x);
    let step = lr * g;
    let x_new = x - step;
    // x_new should be 3.0
    (x_new + 39.0) as i32
}
```

What to notice:

- **Two functions.** `loss` takes an `f32` and returns an `f32`; its body is `diff * diff` (the
  trailing expression, again no `return`). `main` calls into it.
- **Floats and casts.** `x`, `lr`, and the intermediates are `f32`; the final `(x_new + 39.0) as
  i32` casts the float result to an integer for the exit code. The arithmetic here is real IEEE-754
  SSE codegen, not interpreted — `f32` and `f64` are both proven
  ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §2).
- **`grad(loss)` is source-level autodiff.** `grad(loss)` is the derivative of the named function
  `loss`; applying it at `x` gives `d(loss)/dx = 2*(x-3)`, which at `x=0` is `-6`. The full step
  works out to `x_new = 3.0`, and `3 + 39 = 42`. This forward-mode scalar `grad` on a named `@pure`
  function is a real, corpus-proven feature
  ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §5); the autodiff
  story has its own Part III chapter, so we leave it at this taste.

> **For AI agents:** `grad(f)` here is the source-level derivative of a *named* scalar `@pure`
> function `f` — not a general tensor-valued AD API. Its proven scope is `f32`/`f64` scalar
> parameters ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §5). Do
> not generalize beyond that surface; the broader AD design targets are flagged honestly in the
> spec and the autodiff chapter.

---

## Control flow: `if`, `while`, `match`

### `if` is an expression

`if`/`else` is an **expression** that yields the value of whichever arm is taken, so it composes
directly into a `let` or a return position. There is no `else if` keyword — you nest another `if`
in the `else` arm ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §4).
And because Helix has no implicit int-to-bool conversion, the condition is always an explicit
comparison. We will see `if`-as-expression used inside `match` arms in the recursion example below.

### `while` and `break`

The proven loop form is `while`. Here is a complete, gate-pinned `while`-loop that sums `0..5`:

**Verified example** — the gate's `while_sum` corpus program
([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`; generated and run by the gate,
which asserts exit `10` with `chk "$CD/while_sum.hx" 10`):

```helix
fn main() -> i32 { let mut s = 0; let mut i = 0; while i < 5 { s = s + i; i = i + 1; } s }
```

Read it left to right: two `let mut` cells, a `while i < 5 { … }` loop whose body reassigns both,
and the trailing `s` as the block value. `0 + 1 + 2 + 3 + 4 = 10`. The companion `while_break`
corpus row shows `break` exiting a loop early
([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`, `chk "$CD/while_break.hx" 7`).

> **Note — `for` is a desugar, not a core form.** You will see `for i in 0 .. n { … }` in some
> example files (it reads nicely), but in the as-built `kovc` a `for` loop is a **parser desugar**
> to a `while` with a counter, exercised by the gate's `M1_for_loop` row — it is not a first-class
> control form ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §4, §7).
> The same is true of `&&` / `||`, which are nested-`if` desugars rather than operators. When you
> want the proven core, reach for `while` and nested `if`.

### `match`

`match` is an expression with comma-separated `pattern => body` arms. Patterns can bind a name,
match a literal or a range, destructure a tuple or struct, or pull a payload out of an enum
variant, and arms can be combined with the or-pattern `A | B`
([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §4). The next two
sections put `match` to work on real enums and structs.

> **Residual — guards and exhaustiveness are not enforced.** A `match` arm may be *written* with
> an `if`-guard, but `kovc` does **not** enforce it (every matching arm body runs regardless), and
> there is **no** exhaustiveness check — a non-exhaustive `match` is accepted and runs the covered
> arms. Both bounds are stated plainly and locked by `*_bound` corpus rows in
> [`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §7. The pattern-matching
> chapter in Part III covers this in detail.

---

## Structs and enums at a glance

### Structs and `enum`-as-tag, with `match`

Structs group named, typed fields; enums are sum types whose variants you discriminate with
`match`. This example models a tiny shape calculator: an `enum Kind` for the shape kind, a `struct
Shape` carrying the kind plus dimensions, and `match s.kind { … }` to dispatch.

**Verified example** — [`helixc/examples/hbs_sample_enum_struct.hx`](../../../helixc/examples/hbs_sample_enum_struct.hx)
(compiles to a Linux ELF; runs and exits `129` — gate row
`chk "$EX/hbs_sample_enum_struct.hx" 129`,
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`):

```helix
enum Kind { Circle, Square, Rectangle }

struct Shape {
    kind: Kind,
    a: i32,
    b: i32,
}

@total
fn area_squared(s: Shape) -> i32 {
    // We return area^2 to keep everything in i32 (no f32 sqrt needed).
    // Circle: π*r^2 ≈ 3 * r^2 (cheap approximation)
    // Square: a^2
    // Rectangle: a*b
    match s.kind {
        Kind::Circle => 3 * s.a * s.a,
        Kind::Square => s.a * s.a,
        Kind::Rectangle => s.a * s.b,
        _ => 0,
    }
}
```

(The file continues with a `perimeter` function and a `main` that builds three `Shape` values and
sums their areas and perimeters to `129`; the excerpt above is the struct/enum/`match` core. The
full program is the cited file.) What it shows:

- **`enum Kind { Circle, Square, Rectangle }`** — three tag-only variants.
- **`struct Shape { kind: Kind, a: i32, b: i32 }`** — named fields, including a field whose type is
  the user enum. Field access is `s.kind`, `s.a`, `s.b`.
- **`match s.kind { Kind::Circle => …, … , _ => 0 }`** — variant patterns written as
  `EnumName::Variant`, with `_` as the wildcard arm. Each arm is an expression; the matched arm's
  value is the `match` value.
- **`@total`** is an attribute on the function. Attributes (like `@pure`, `@kernel`) annotate
  functions; the effect/attribute system is its own Part III chapter.

### Enums that carry payloads

Enum variants can carry data, and `match` pulls the payload out by binding it. This example
constructs `Maybe::Some(40)` and `Maybe::Some(2)`, extracts the inner `i32`s with `match`, and sums
them:

**Verified example** — [`helixc/examples/hbs_sample_option.hx`](../../../helixc/examples/hbs_sample_option.hx)
(compiles to a Linux ELF; runs and exits `42` — gate row `chk "$EX/hbs_sample_option.hx" 42`,
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`):

```helix
enum Maybe { None, Some(i32) }
enum Pair { Empty, Cons(i32, i32) }

fn main() -> i32 {
    // Compute Some(40) + Some(2) by extracting payloads and summing.
    let m1 = Maybe::Some(40);
    let m2 = Maybe::Some(2);

    let v1 = match m1 {
        Maybe::Some(x) => x,
        Maybe::None => 0,
    };
    let v2 = match m2 {
        Maybe::Some(x) => x,
        Maybe::None => 0,
    };
    let total1 = v1 + v2;     // 42

    // Pair::Cons unpacking: a + b
    let p = Pair::Cons(15, 25);
    let total2 = match p {
        Pair::Cons(a, b) => a + b,    // 40
        Pair::Empty => 0,
    };

    // Mix: total1 (42) is an i32; total2 is 40. Final answer = 42 (we
    // pick the option-extraction result as the demo).
    let r = total1;
    r
}
```

The shape to take away:

- **Payload variants** are declared with parentheses: `Some(i32)`, `Cons(i32, i32)`. Tag-only
  variants (`None`, `Empty`) sit alongside them as sentinels.
- **Construction** is `Maybe::Some(40)`, `Pair::Cons(15, 25)`.
- **Extraction** is a `match` arm that binds the payload: `Maybe::Some(x) => x` binds `x` to the
  inner value; `Pair::Cons(a, b) => a + b` binds both fields. This payload-bearing-enum-plus-`match`
  pattern is proven ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md)
  §2–§4).

> **Note — `Result` is just an `enum`.** Helix does not bake in an `Option`/`Result` type; the
> `Maybe` above is an ordinary user enum, and the shipped answer for `Result<T, E>` is likewise a
> user-defined `enum Result { Ok(i32), Err(i32) }` matched the same way — proven by the gate's
> `result_inline` row ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md)
> §7, decision 2). Nothing in this section is compiler magic; it is the plain enum + `match` surface.

### Enums + recursion + `if` together

To see the pieces compose, here is a recursive function over a payload enum. It computes `5! = 120`
by stepping a little `State` machine — and it uses `match`, payload binding, an `if`/`else`
expression, `let`, and a recursive self-call all at once:

**Verified example** — [`helixc/examples/hbs_sample_recursion.hx`](../../../helixc/examples/hbs_sample_recursion.hx)
(compiles to a Linux ELF; runs and exits `120` — gate row `chk "$EX/hbs_sample_recursion.hx" 120`,
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`):

```helix
enum State { Done, Continue(i32) }

@total
fn step(s: State, acc: i32) -> i32 {
    match s {
        State::Done => acc,
        State::Continue(n) => {
            if n <= 1 {
                acc
            } else {
                let next = State::Continue(n - 1);
                step(next, acc * n)
            }
        }
    }
}

@total
fn factorial(n: i32) -> i32 {
    let init = State::Continue(n);
    step(init, 1)
}

fn main() -> i32 {
    // 5! = 120, but exit code is 8-bit → 120 fits.
    // Use 5 to keep the result < 256.
    factorial(5)
}
```

(The excerpt above drops the file's explanatory comments inside `step`; the logic is verbatim. The
full program is the cited file.) Walking it:

- **`match s { State::Done => acc, State::Continue(n) => { … } }`** dispatches on the enum; the
  `Continue(n)` arm binds the payload `n`.
- **The arm body is a block** `{ … }` whose value is an **`if`/`else` expression** — note the two
  arms (`acc`, and the recursive `step(next, acc * n)`) are the block's value with no `return`.
- **Recursion**: `step` calls itself; `factorial` seeds the machine and calls `step`. `factorial(5)`
  reduces to `120`, which fits in the 8-bit exit code.

That single program is a good snapshot of Helix's everyday feel: data modeled with an enum,
behavior expressed by matching on it, control by expression-valued `if`, and composition by
ordinary recursion.

> **Note — struct *destructuring* in patterns.** Beyond enum payloads, `match` can destructure
> structs by field — `Point { x, y }`, a literal-field test `Point { x: 0, y }`, nested
> `Outer { inside: Inner { value }, tag }`, and the ignore-rest `Point { .. }`. All four are
> proven by [`helixc/examples/dogfood_18_pat_struct_showcase.hx`](../../../helixc/examples/dogfood_18_pat_struct_showcase.hx)
> (gate row `chk "$EX/dogfood_18_pat_struct_showcase.hx" 42`,
> [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`). The pattern-matching chapter
> in Part III walks through it.

---

## The overall feel

Pulling the tour together, here is the Helix you have now seen, all of it grounded in programs the
gate compiles and runs:

- **Expression-oriented.** Blocks yield their trailing expression; `if` and `match` are
  expressions; `return` is for early exit only. This is the rule that makes Helix read the way it
  does.
- **Statically typed, with explicit conversions.** Literals carry a default type (`i32`, `f32`),
  suffixes select others, type annotations go on `let`, and numeric conversions are explicit with
  `as` — there are no implicit numeric coercions and no implicit int-to-bool.
- **Familiar core control flow.** `let` / `let mut`, `if`/`else` (nest for `else if`), `while`
  (with `break`), and `match` with rich patterns. The conveniences `for`, `+=`, and `&&`/`||` exist
  as **parser desugars** over that proven core, not as primitives
  ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §7).
- **Data with structs and enums.** Named-field structs, sum-type enums (tag-only and
  payload-bearing), and `match` to take them apart — with `Option`/`Result` being ordinary user
  enums rather than built-ins.
- **A genuinely AGI-oriented edge, previewed.** Source-level `grad`, the `@pure`/`@total`
  attribute surface, and (beyond this tour) shape-typed tensors and a GPU `@kernel` path are what
  set Helix apart from a plain systems language. Those are the subject of the chapters ahead.

Everything here compiles with `kovc` straight to a static Linux ELF and is pinned by the gate, so
the language you just toured is the language that is actually shipped — no aspirational syntax
crept in. Where a feature is partial or unenforced (match guards, exhaustiveness, the desugared
conveniences), this tour said so and pointed at the residual in
[`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §7.

> **For AI agents:** when you need a known-good Helix program to compile or adapt, start from a
> file in [`helixc/examples/`](../../../helixc/examples/) that appears as a `chk` row in
> [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]` — those are compile-and-run
> proven with a pinned exit code. Prefer the proven core forms (`while`, nested `if`, plain
> assignment) over the desugared conveniences when emitting new code, and never assert a feature
> the spec marks `[erased]` / `[unsupported]` actually has its semantics.

---

**Next:** [Types: widths, structs, and enums](02-types.md) — the full type surface (the signed
and unsigned integer widths, including 64-bit literals beyond 2³¹/2³², the float widths down to
`bf16`/`f16`, `bool`, `struct`, and `enum`), and how integer literals are typed and widened, each
grounded against the as-built spec and the proven corpus.
