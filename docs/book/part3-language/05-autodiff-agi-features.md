# Autodiff & the AGI-oriented features

*What this chapter covers:* the one form of automatic differentiation the shipping compiler
actually proves — forward-mode scalar `grad` — and then, with scrupulous honesty about the
boundary, the *library-level* autodiff modules and the *AGI-oriented* constructs (provenance
tracking, the safety/confidence wrappers, the memory-tier and differentiable-type surface)
that live in the repository as **real committed source written against an earlier frontend**
but are **not** part of the gate-proven, self-hosting `kovc` today. This is the highest
overclaim-risk material in the book, so the chapter is organised around a single hard line:
**what `kovc` compiles and the gate runs** versus **what is designed, scaffolded, or
identity-erased**. Read [Part I — "Trust at a glance"](../part1-orientation/04-trust-at-a-glance.md)
and [Part IX — "Non-negotiables"](../part9-for-ai-agents/02-non-negotiables.md) first if you
have not; the honesty discipline there is the discipline here.

> **For AI agents:** the load-bearing fact of this chapter is the boundary, not the feature
> list. Exactly **one** autodiff construct has a standing compile-proof in the gate:
> forward-mode `grad`, via [`helixc/examples/gradient_descent.hx`](../../../helixc/examples/gradient_descent.hx)
> (`scripts/gate_kovc.sh` asserts exit **42**). Everything else in this chapter —
> `grad_rev`/`grad_rev_all`, the `autodiff.hx` / `autodiff_reverse.hx` modules, and **every**
> AGI/safety/provenance construct — is either `[impl]` (in `kovc` but not gate-asserted) or
> **not in the shipping compiler at all** (written for the deleted Python `helixc` frontend).
> Do not present any of it as a working, verified capability. When in doubt, treat it as
> design intent and cite the residual.

---

## 1. The honesty boundary, stated up front

Helix's reason for existing is auditable trust (see
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md)). That makes a chapter about
"AGI features" dangerous: it would be very easy to read the repository's ambition as the
repository's achievement. They are not the same thing, and the project's own completion
record says so in as many words. The Definition of Done draws a scope boundary before it
lists a single criterion:

> **Scope boundary (read this first).** This defines completion of the **substrate**.
> It does **not** define AGI. AGI is open research — no language, however complete,
> makes it achievable; it is *not* a Helix milestone.
>
> — [`docs/HELIX_V1_DEFINITION_OF_DONE.md`](../../../docs/HELIX_V1_DEFINITION_OF_DONE.md)

and again, explicitly, under "OUT OF SCOPE":

> - **AGI itself** — open research; not gated by the language. Pursued *on* Helix, after v1.0.
>
> — [`docs/HELIX_V1_DEFINITION_OF_DONE.md`](../../../docs/HELIX_V1_DEFINITION_OF_DONE.md)

So when this chapter says "AGI-oriented features," it means *language and library surface
designed with AGI workloads in mind* — not a claim that Helix does AGI, or that the features
below are finished. Three tiers will recur:

- **Gate-proven** — compiled **and** run by the universal gate
  ([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh)), with an asserted exit code. In
  this chapter that is exactly one thing: forward-mode `grad`.
- **`[impl]`** — implemented in the shipping `kovc` codegen or parser but **not** asserted by
  the gate. It exists; it is not independently proven here. (The spec's honesty legend, in
  [`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md), uses the same `[impl]`
  marker.)
- **Design-stage** — real committed `.hx` source that the **current** self-hosting compiler
  does **not** compile, because it depends on intrinsics that only ever existed in the
  **deleted** Python `helixc` frontend, or because its semantics are deliberately
  identity-erased (a compile-time-only metadata channel). This is most of the AGI surface.

> **For AI agents:** "the deleted Python `helixc` frontend" is a real, load-bearing
> distinction. The Python reference compiler was removed at K4 (see
> [`docs/HELIX_V1_DEFINITION_OF_DONE.md`](../../../docs/HELIX_V1_DEFINITION_OF_DONE.md), scope
> decision 6 and criterion #6). `kovc` — the from-`seed` self-hosting compiler — is the only
> compiler now. A feature that worked in the Python frontend is **not** a feature of Helix
> today unless it also lives in `helixc/bootstrap/{lexer,parser,kovc}.hx`. Verify with a grep
> over those three files before claiming any intrinsic exists.

---

## 2. Forward-mode `grad` — the one proven autodiff path

The shipping compiler implements a single, narrow, *real* automatic-differentiation feature:
a forward-mode derivative of a named scalar function, spelled `grad(f)(x)`. The spec lists it
honestly:

> **Autodiff**: `grad(f, idx)` — forward-mode derivative of a named fn [impl;
> gradient_descent.hx is corpus-proven]. (The capstone uses hand-written verified backward
> kernels, not the `grad` keyword on GPU — see DoD #4.)
>
> — [`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §5

and the stdlib reference pins its status as `[corpus-proven]` against one program:

> | `grad(f, idx)` | forward-mode CPU autodiff | [corpus-proven] (`gradient_descent`) |
>
> — [`docs/HELIX_V1_STDLIB.md`](../../../docs/HELIX_V1_STDLIB.md) (d)

That program is the proof. Here it is in full.

**Verified example** — [`helixc/examples/gradient_descent.hx`](../../../helixc/examples/gradient_descent.hx)
(compiled + run by the gate via the self-hosted K2; `scripts/gate_kovc.sh` asserts exit `42`):

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

The gate runs this with the line

```bash
chk "$EX/gradient_descent.hx" 42
```

in [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`, where `chk` compiles
the file with the freshly self-hosted K2 compiler and asserts the produced binary exits with
the second argument. Exit `42` means `x_new` came out to exactly `3.0` — i.e. `grad(loss)`
returned the correct derivative `2*(0 - 3) = -6` and one descent step landed on the optimum.

### How it works (and what it does not do)

`grad` is a compiler **rewrite**, not a runtime tape and not a function value you can store.
The parser recognises the `grad(IDENT)(args)` form for an *inlinable* named function and
synthesises the derivative at compile time. The result is ordinary `f32` arithmetic in the
emitted ELF — there is no autodiff machinery at run time, which is exactly why a from-raw,
GC-free target can support it at all.

Two honest limits follow directly from the spec text above and from the program itself:

- **Forward mode, scalar.** `grad(loss)(x)` computes a single directional derivative of a
  scalar `f32`→`f32` function. There is no tensor `grad`, no vector-Jacobian product, and no
  `vmap`/`jit`-style transform stack. The design notes call those out as future work (next
  section).
- **Not the capstone's GPU autodiff.** The end-to-end transformer capstone does **not** use
  the `grad` keyword to differentiate GPU kernels. Its backward pass is a set of
  **hand-written PTX kernels**, each checked against a finite-difference reference; that is a
  *stronger* guarantee than an auto-emitted-but-unverified gradient, and it is the path the
  Definition of Done actually certifies (criterion #4). The `grad` keyword *emitting* GPU
  code is named as post-v1.0 ergonomics, not a shipped capability —
  [`docs/HELIX_V1_DEFINITION_OF_DONE.md`](../../../docs/HELIX_V1_DEFINITION_OF_DONE.md) #4. The
  capstone op set and its honest residuals (PTX-not-SASS, a fraction of cuBLAS, 7.0–8.7×
  end-to-end) belong to the GPU chapters and to
  [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md); they are not autodiff claims.

> **For AI agents:** `grad` is a parse-time rewrite over a named, inlinable scalar `f32` fn.
> If you need a derivative inside a Helix program that the gate will accept, this is the only
> sanctioned route, and `gradient_descent.hx` is the template to copy. Do not reach for the
> `autodiff.hx` library functions below — they will not compile under `kovc` (§4).

### `grad_rev` / `grad_rev_all` — present in the parser, not gate-asserted

The parser also recognises a reverse-mode-shaped meta-call, `grad_rev_all(f)(args).field`,
which the design notes describe as returning a per-parameter gradient. Two honesty points:

1. It is **`[impl]`, not gate-proven.** No row in `scripts/gate_kovc.sh` exercises
   `grad_rev`/`grad_rev_all`; the only autodiff row is `gradient_descent.hx` above.
2. Its implementation is **forward-mode underneath.** The parser's own comment is explicit
   that the "reverse-mode" surface is synthesised one parameter at a time on a forward-mode
   basis:

**Fragment** (excerpt of [`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx),
the comment over the `grad_rev` handling; not a complete program):

```helix
// Built by parse_primary's grad_rev_all branch when it encounters the postfix
// `.IDENT` form; consumed at end of parse_program by grad_rev_pass to
// synthesize the per-param derivative fn decls (forward-mode based — single
// param at a time, since the field selects which partial we want).
```

So the honest statement is: Helix has a working, gate-proven **forward-mode scalar** `grad`,
plus a parser-level `grad_rev_all` surface that is implemented on the same forward-mode
mechanism and is not independently proven by the gate. The design intent of a true
tape-based reverse mode is discussed next — as intent.

---

## 3. The differentiable surface as *designed* (`D<T>`, transforms)

The design document [`docs/lang/agi-features.md`](../../../docs/lang/agi-features.md) sets out a
broader differentiability vision: a type-level "differentiable T," `D<T>`, that propagates
gradient flow through the type checker so gradient bugs fail at compile time. It is careful to
date and bound itself — it describes "Stage 35" behaviour, which predates the v1.0 freeze, and
it marks the scalar surface as the part that works:

> `D<T>` means "differentiable T". The compiler propagates gradient flow at type-check time.
> Stage 35 exposes scalar `grad`, `grad_rev`, and `grad_rev_all` compiler rewrites for
> `f32`/`f64`; broader tensor gradients and pytree leaf expansion are still being wired.
>
> — [`docs/lang/agi-features.md`](../../../docs/lang/agi-features.md) §5

and its own live status table lists the remaining transform surface as future work:

> | 9. Composable transforms (`grad`/`grad_rev`) | Stage 35 scalar surface | … | scalar
> forward/reverse AD with fail-closed opaque calls; `vmap`/`jit` remain future work |
>
> — [`docs/lang/agi-features.md`](../../../docs/lang/agi-features.md), "Implementation status"

Read this against §2's hard fact: the *scalar `grad` rewrite* is the proven part; `D<T>` as a
gradient-tracking **type** is design-stage. The authoritative v1.0 spec does not list `D<T>`
among the implemented types at all — its type table tops out at the concrete numeric, struct,
enum, tuple, array, generic, and closure forms, with references/pointers `[unsupported]`
([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §2). The `agi_demo.hx`
example that exercises `D<f32>` is itself labelled non-running by its own header (§5 below).

> **Warning:** `agi-features.md` is a **design/aspiration** document from before the v1.0
> freeze. Where it and [`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md)
> disagree about what is implemented, the spec wins — it is the as-built reference cited by
> Definition-of-Done criterion #8. Treat `D<T>` gradient-typing, tensor gradients, pytrees,
> `vmap`, and `jit` as designed/planned, not as things Helix does.

---

## 4. The library autodiff modules — real source, not in the gate

The repository ships two hand-written autodiff **libraries** in
[`helixc/stdlib/`](../../../helixc/stdlib/). They are genuine, committed, carefully written
Helix source — and they are **not** compiled or run by the gate, and the **current**
self-hosting `kovc` does **not** compile them, because they depend on math intrinsics
(`__exp_f64`, `__sigmoid_f64`, `__sqrt_f64`, `__log_stable_f64`, `__sin_f64`, `__cos_f64`, …)
that exist only in the deleted Python frontend, not in `helixc/bootstrap/`. Documenting them
honestly means: this is **design-stage** library code, valuable as a record of the intended
API and numerics, not a proven capability of the shipping toolchain.

> **For AI agents:** none of the `d_*` (forward) or `rev_*` (reverse) functions below appear
> in `scripts/gate_kovc.sh`, and `autodiff.hx`'s math-intrinsic dependencies are absent from
> `helixc/bootstrap/kovc.hx`. Do not import these modules expecting them to compile with the
> shipped `kovc`. The only gate-accepted autodiff is the `grad` rewrite of §2.

### 4.1 Forward mode by dual numbers — `autodiff.hx`

[`helixc/stdlib/autodiff.hx`](../../../helixc/stdlib/autodiff.hx) implements dual-number
forward-mode AD as a set of `@pure` helpers. Because Helix struct fields lower to i32 slots
(an `f64` field would truncate — see the module header), it represents each dual number as two
explicit `f64`s, value and derivative, and exposes a `<op>_v` / `<op>_dx` pair per operation
instead of a multi-return tuple. The module header documents the convention precisely:

**Fragment** (header of [`helixc/stdlib/autodiff.hx`](../../../helixc/stdlib/autodiff.hx);
documentation, not a runnable program):

```helix
// Convention: every dual op takes pairs (a_v, a_dx, b_v, b_dx, ...) and
// returns a tuple (val, dx) — but since Helix doesn't have multi-return
// yet, we expose two functions per op: `<op>_v` returns the value,
// `<op>_dx` returns the derivative.
//
// To compute df/dx for f(x), seed dx=1.0_f64 for x and dx=0.0_f64 for
// constants:
//   x_v = 3.0_f64, x_dx = 1.0_f64
//   f(x) = x*x + 2*x + 1
//   v = mul_v(x_v, x_dx, x_v, x_dx);  dx = mul_dx(x_v, x_dx, x_v, x_dx)
//   ... etc.
```

The product rule is the canonical case, and the module implements exactly the textbook form:

**Fragment** (the multiply rule from
[`helixc/stdlib/autodiff.hx`](../../../helixc/stdlib/autodiff.hx); excerpt):

```helix
// d/dx (a*b) = a'b + a b'  (product rule)
@pure fn d_mul_v(a_v: f64, a_dx: f64, b_v: f64, b_dx: f64) -> f64 { a_v * b_v }
@pure fn d_mul_dx(a_v: f64, a_dx: f64, b_v: f64, b_dx: f64) -> f64 {
    a_dx * b_v + a_v * b_dx
}
```

What is genuinely instructive here — and worth carrying over even though the module is
design-stage — is its **fail-closed numerics discipline**, which matches the audit posture of
the rest of the project. The division and reciprocal rules refuse to silently produce a wrong
gradient at a singularity; they fall back to a sentinel, and the comment explains the trap they
close:

**Fragment** (the `_checked` division variants from
[`helixc/stdlib/autodiff.hx`](../../../helixc/stdlib/autodiff.hx); excerpt):

```helix
// Post-fix: _checked variants accept caller-supplied sentinel that returns
// on singular input. Pattern mirrors __powi_checked from batch 17.
// Recommended sentinel: NaN (via 0.0_f64 / 0.0_f64) to surface the
// singularity through any downstream IEEE 754 arithmetic. Caller pattern:
//   let nan = 0.0_f64 / 0.0_f64;
//   let f = d_div_v_checked(a, da, b, db, nan);
//   if f != f { /* singularity hit (NaN != NaN per IEEE 754) */ }
@pure fn d_div_v_checked(a_v: f64, a_dx: f64, b_v: f64, b_dx: f64, sentinel: f64) -> f64 {
    if b_v == 0.0_f64 { sentinel } else { a_v / b_v }
}
```

The module also covers `exp`, `sigmoid`, `sqrt`, `ln`, `sin`, `cos`, `relu`, `abs`, and
constant-scaling rules, each with the same `_v`/`_dx` split and the same fail-closed-at-singular
treatment (e.g. `d_sqrt_dx` returns `0.0` at `a_v <= 0.0` rather than dividing by zero). Again:
this is a faithful record of an intended forward-mode library and a good model for numerics
discipline — not a gate-proven module.

### 4.2 Reverse mode by tape — `autodiff_reverse.hx`

[`helixc/stdlib/autodiff_reverse.hx`](../../../helixc/stdlib/autodiff_reverse.hx) is the more
ambitious module: a genuine **tape-based reverse-mode** engine built entirely on the i32
arena. It records each operation during the forward pass and walks the tape backward to
accumulate adjoints — the right asymptotics for neural-network backprop (one backward pass for
a scalar loss regardless of parameter count), as its header explains:

**Fragment** (header of
[`helixc/stdlib/autodiff_reverse.hx`](../../../helixc/stdlib/autodiff_reverse.hx); excerpt):

```helix
// Phase 2.1 step 2: reverse-mode AD. Records operations during the
// forward pass into a tape; backward pass walks the tape in reverse,
// propagating gradients. Unlike forward mode (O(N) per parameter),
// reverse mode is O(1) backward pass per OUTPUT — exactly what NN
// backprop needs (millions of params, single scalar loss).
```

It exposes a small, clean API — `rev_tape_new`, `rev_leaf`, `rev_add`/`rev_sub`/`rev_mul`/
`rev_neg`, `rev_alloc_adjoints`, `rev_seed`, `rev_backward`, `rev_grad` — and the backward walk
implements the standard adjoint rules:

**Fragment** (the adjoint rules documented over `rev_backward` in
[`helixc/stdlib/autodiff_reverse.hx`](../../../helixc/stdlib/autodiff_reverse.hx); excerpt):

```helix
// Walk tape in reverse, propagating adjoints.
// For each tape entry of kind K with inputs (a, b) and adjoint adj[i]:
//   K = leaf:  no propagation.
//   K = add:   adj[a] += adj[i]; adj[b] += adj[i]
//   K = sub:   adj[a] += adj[i]; adj[b] -= adj[i]
//   K = mul:   adj[a] += adj[i] * value(b); adj[b] += adj[i] * value(a)
//   K = neg:   adj[a] -= adj[i]
```

Two honest qualifications dominate any reading of this file:

- **It computes on i32 values, not floats.** The tape stores an i32 forward value per node and
  the adjoints are i32; the recent hardening in the file is about `i64` intermediates and
  INT32 **saturation** so a recorded value or an adjoint update cannot silently wrap (see the
  `rev_add`/`rev_mul`/`rev_backward` saturation logic). It is an integer-arithmetic reverse-mode
  engine — a correctness and memory-safety study (it carries elaborate tape/adjoint guard words
  to reject forged buffers), not a float training engine.
- **It is not in the gate, and not how the capstone trains.** No row of `scripts/gate_kovc.sh`
  exercises `rev_*`. The capstone's reverse pass is the hand-written, finite-difference-checked
  PTX kernels (§2), which is what the Definition of Done certifies. So
  `autodiff_reverse.hx` is best read as design-stage infrastructure with unusually rigorous
  memory-safety guards, not a proven training path.

> **For AI agents:** if a task needs a *proven* gradient under the shipping toolchain, there is
> exactly one: forward-mode scalar `grad` (§2). The reverse-mode library is real source but
> unproven by the gate and integer-valued. Do not cite `autodiff_reverse.hx` as evidence that
> "Helix does backprop"; the capstone's verified PTX kernels are that evidence, and they are
> not this module.

---

## 5. The AGI-oriented type-system features — designed, scaffolded, identity-erased

[`docs/lang/agi-features.md`](../../../docs/lang/agi-features.md) and
[`docs/HELIX_V1_FINAL_FEATURES.md`](../../../docs/HELIX_V1_FINAL_FEATURES.md) lay out an ambitious
"AGI substrate" surface: reflection (`quote`/`splice`/`modify`), effect/capability types,
memory-tier types (`WorkingMem`/`EpisodicMem`/`SemanticMem`/`ProceduralMem`), differentiable
types, tile-as-first-class types, agent/`society` primitives, and auto-curriculum primitives.
These are genuinely interesting designs and are worth reading as a roadmap. But the chapter's
job is to say plainly which of them the **shipping** compiler does, and the answer for the AGI
surface is: **almost none, as runtime behaviour.** The design doc itself is candid about this
at nearly every entry. A few representative admissions, quoted:

> Current Stage 35 behavior is narrower: `quote` has stub semantics and returns a
> stable AST hash. Real runtime AST inspection and real `splice` execution are
> future work.
>
> — [`docs/lang/agi-features.md`](../../../docs/lang/agi-features.md) §1 (reflection)

> Current Stage 35 behavior is a scaffold: the verifier value controls
> accept/reject, but real AST rewrite/commit semantics are future work.
>
> — [`docs/lang/agi-features.md`](../../../docs/lang/agi-features.md) §2 (verifier-gated `modify`)

> Current Stage 35 behavior is type-level only: `learn_to` can return
> `Skill<F>`, while the runtime registry and task-selection semantics remain
> future work.
>
> — [`docs/lang/agi-features.md`](../../../docs/lang/agi-features.md) §8 (auto-curriculum)

And the authoritative spec confirms the cut from the other direction: in
[`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md), `module`, `trait`,
`const`/`static` are "parsed-erased or unsupported (no real semantics)" (§3), and "reflection
stubs return 0" (§5). The combined honest reading is: the AGI type surface is, at most,
**parse-and-typecheck-level scaffolding with identity/stub runtime behaviour**, and several
pieces are not in the shipping compiler at all.

### 5.1 The showcase example labels itself non-running

The clearest single piece of evidence is the demo the design doc points to. Its own header
states it will not run — it exercises the *type checker*, not codegen:

**Fragment** (header of [`helixc/examples/agi_demo.hx`](../../../helixc/examples/agi_demo.hx);
the program's own disclaimer — excerpt, not a runnable claim):

```helix
// agi_demo.hx — demonstration of Helix's compile-time AGI features
//
// This program won't run (it uses types we haven't added codegen for yet),
// but it WILL compile through the type checker. The type checker catches
// 4 different classes of bugs that no other language catches:
//
// 1. Shape mismatches (Presburger constraint solver)
// 2. Effect-capability violations (@pure / @io / etc.)
// 3. Silent gradient loss (D<T> type wrapper)
// 4. Cross-tier memory confusion (WorkingMem / EpisodicMem / SemanticMem)
```

Note the careful wording — *"won't run … but it WILL compile through the type checker"* — and
note that it targets the **Python frontend's** type checker (it uses `D<T>`, `WorkingMem<T>`,
`learn_to`, and Presburger `where N % 16 == 0` shape constraints, none of which are in the
shipping `kovc` per §2/§3 of the spec). It is a design illustration. It is **not** a
gate-proven program, and this book must not present it as one.

> **For AI agents:** `agi_demo.hx` is the canonical trap. It is committed, it is named in
> `agi-features.md`, and it explicitly says it does not run. Do not cite it (or
> `agi_substrate_demo.hx`, `self_improving_agent.hx`, or the `dogfood_06`/`07`/`15`/`21`/`22`/
> `23` examples) as evidence of a working capability — they are written against intrinsics
> (`prove`, `Logic<T>`, `fuzzy_and`, `__wrap_conf`, `register_derivation`, …) that are absent
> from `helixc/bootstrap/`. The gate corpus list in `scripts/gate_kovc.sh` step `[4]` is the
> authoritative set of programs that actually compile-and-run.

### 5.2 Provenance — `Logic<T>` is identity-erased; `provenance.hx` is design-stage

Provenance — tracking where a value came from, through the type system — is the design doc's
"strategic differentiator." The committed module is
[`helixc/stdlib/provenance.hx`](../../../helixc/stdlib/provenance.hx). Its header is the most
honest possible description of what provenance actually *is* at runtime in the Phase-0 design,
and it is essential reading:

**Fragment** (header of [`helixc/stdlib/provenance.hx`](../../../helixc/stdlib/provenance.hx);
excerpt):

```helix
// Phase-0 reminder: `Logic<T> = T` at runtime; the source tag passed
// to `prove()` is discarded at IR lowering. The only runtime-observable
// provenance is the arena side-table populated by `register_derivation`
// (Inc 5), the Inc 9 B2 arena auto-push from `derive`, and the Inc 14
// `register_derivation3` triple-push.
```

So even in the design as written, `Logic<T>` carries **no** runtime payload — it is the value
itself, `T`, and the provenance tag handed to `prove()` is thrown away at IR lowering. The only
thing that survives is an arena side-table maintained by helper primitives. And those
primitives — `register_derivation`, `parent_left_at`, `parent_at`, `prove`, `and_logic`,
`fuzzy_and`, `unwrap_logic` — are **not present in the shipping `kovc`**; they belonged to the
Python frontend. Consequently `provenance.hx` (and the `dogfood_06_provenance_datalog.hx` /
`dogfood_07_provenance_sgd.hx` examples that use `prove`/`fuzzy_and`/`grad_rev`) is
**design-stage**: a real, committed record of the intended provenance API, not something the
current gate compiles or runs.

The module's own body is, fittingly, about *honesty of observation* — it documents a Phase-0
"sharp edge" where a positional slot read can silently return the wrong parent for a
three-parent handle, and adds explicitly-named aliases to avoid it:

**Fragment** (a `has_evidence` honesty caveat from
[`helixc/stdlib/provenance.hx`](../../../helixc/stdlib/provenance.hx); excerpt):

```helix
// This is a NECESSARY-BUT-NOT-SUFFICIENT predicate for
// the handle to refer to a real `register_derivation*` call — the
// Phase-0 arena has no per-handle tag, so a slot whose value happens
// to be non-(-1) for any reason will pass this check.
```

That is the right spirit — but it is documentation of a *designed* mechanism, not a proven one.

### 5.3 Safety / confidence wrappers — compile-time-only, identity-erased, not in the gate

[`helixc/stdlib/safety.hx`](../../../helixc/stdlib/safety.hx) provides the most concrete
slice of the "uncertainty as a typed value" vision: eleven Tier-S/A wrapper types — `Conf<T>`
(confidence), `Confidential<T>` (information-flow taint), `Private<T>` (differential-privacy
budget), `Q8<T>` (quantization), `InDist<T>` (out-of-distribution), `Robust<T>` (adversarial
robustness), `Energy<T>`, `InEnclaveSGX<T>`, `Counterfactual<T>`, `Deadline<T>`, and
`FromUnknown<T>` (attribution) — each with a constructor helper and an opt-out helper. The
module is honest about their nature in its header:

**Fragment** (header of [`helixc/stdlib/safety.hx`](../../../helixc/stdlib/safety.hx);
excerpt):

```helix
// The wrappers are identity-erased at IR / codegen, so these helpers
// have zero runtime overhead — they exist purely as compile-time
// metadata channels.
```

A representative wrapper pair shows the shape — a `@pure` constructor that calls a `__wrap_*`
builtin and a `@pure` opt-out that calls the matching `__lift_*`/strip builtin:

**Fragment** (the confidence wrapper helpers from
[`helixc/stdlib/safety.hx`](../../../helixc/stdlib/safety.hx); excerpt):

```helix
@pure
fn as_conf(x: f32) -> Conf<f32> {
    __wrap_conf(x)
}

@pure
fn strip_conf_f32(x: Conf<f32>) -> f32 {
    __lift_conf(x)
}
```

The module also carries `@property` round-trip assertions stating that wrap-then-strip is the
identity for every wrapper (because the wrappers are identity-erased) — a genuinely nice piece
of design that anticipates property-based testing. But the honest status is the same as
provenance:

- The wrapper **types** (`Conf<T>`, `Confidential<T>`, …) and their `__wrap_*` / strip
  **builtins** are **not in the shipping `kovc`** — they were frontend features. A grep over
  `helixc/bootstrap/` finds none of them.
- The wrappers are, by design, **compile-time-only metadata** — even where implemented, they
  are erased at IR with zero runtime effect. The design intent (per
  [`docs/HELIX_V1_FINAL_FEATURES.md`](../../../docs/HELIX_V1_FINAL_FEATURES.md), Part 1) is that a
  *future* type checker would make illegal flows **fail to compile**; that enforcement is the
  point, and it is the part that is not delivered.
- Neither `safety.hx` nor the `dogfood_21`/`dogfood_22`/`dogfood_23` programs that exercise the
  wrappers appear in the gate corpus. They are design-stage.

> **Residual:** the safety/confidence/provenance constructs are a **designed** surface for
> typed uncertainty, info-flow, and evidence. As shipped, they are not in the self-hosting
> compiler, and where they ever existed they were identity-erased (no runtime semantics) and
> ungated. The civilization-scale framing in
> [`docs/HELIX_V1_FINAL_FEATURES.md`](../../../docs/HELIX_V1_FINAL_FEATURES.md) is an explicit
> *roadmap* ("Design doc and forward roadmap," its own status line says), not a statement of
> current capability.

### 5.4 Memory-tier, agent, and curriculum primitives — type-level at most

The remaining AGI primitives follow the same pattern and need only brief, honest treatment:

- **Memory-tier types** (`WorkingMem`/`EpisodicMem`/`SemanticMem`/`ProceduralMem`) are
  described as "type-level plus selected builtins" in the design doc, with the cross-tier
  operators (`consolidate`, `recall`, `retrieve`) and their invariants marked future work
  ([`docs/lang/agi-features.md`](../../../docs/lang/agi-features.md) §4). In the shipping spec, the
  underlying `module`/generic machinery is parsed-erased.
- **Agent / `society` primitives** are listed as **parsing only** — `agent Foo { fn ...; }`
  parses, but `society::dispatch` semantics are explicitly named as remaining work
  ([`docs/lang/agi-features.md`](../../../docs/lang/agi-features.md) §7 and the status table).
- **Auto-curriculum** (`learn_to`) is "type-level only," with the runtime registry and
  task-selection as future work (§8, quoted in §5 above).

None of these have gate rows; none should be presented as working. They are design surface.

---

## 6. Summary — the one line to remember

Helix has a small, **real, gate-proven** autodiff capability: **forward-mode scalar `grad`**,
proven by [`helixc/examples/gradient_descent.hx`](../../../helixc/examples/gradient_descent.hx)
(exit `42`). It has a `grad_rev_all` parser surface that is forward-mode underneath and
ungated. It has two genuinely written autodiff **libraries**
([`autodiff.hx`](../../../helixc/stdlib/autodiff.hx) forward, dual-number;
[`autodiff_reverse.hx`](../../../helixc/stdlib/autodiff_reverse.hx) reverse, tape-on-arena,
integer-valued) that are **design-stage** — real committed source, not compiled by the current
`kovc`, not in the gate. And it has an ambitious **AGI-oriented** type surface — provenance,
safety/confidence wrappers, memory tiers, agents, curriculum — that is, at most,
parse-and-typecheck scaffolding with identity-erased or stub runtime behaviour, much of it not
in the shipping compiler at all, and **all** of it downstream of the project's explicit
declaration that **AGI is out of scope as a Helix milestone**
([`docs/HELIX_V1_DEFINITION_OF_DONE.md`](../../../docs/HELIX_V1_DEFINITION_OF_DONE.md)). The honest
achievement is the substrate; the AGI features are the roadmap built *on* it, not a capability
the substrate already has.

> **For AI agents:** when asked "does Helix do autodiff / provenance / typed safety / AGI?",
> answer with the boundary: yes to forward-mode scalar `grad` (one gate-proven program); the
> rest is design-stage or out of scope, per `docs/HELIX_V1_DEFINITION_OF_DONE.md` and the
> `[impl]`/erased markers in `docs/HELIX_V1_LANGUAGE_SPEC.md`. Never assert a working AGI
> capability for Helix.

---

**Next:** this concludes Part III — The Helix Language. **Part IV — The Standard Library**
*(planned)* picks up the proven counterpart to this chapter's designed surface: it inventories
the real compiler-builtin standard library (arena, I/O, math, and the capstone-proven
tensor/ML op set) with its honest `[capstone-proven]` / `[corpus-proven]` / `[impl]` status.
Until those chapters ship, [`docs/HELIX_V1_STDLIB.md`](../../../docs/HELIX_V1_STDLIB.md) is the
authoritative builtin-stdlib reference, and [Part IX — "Driving Helix"](../part9-for-ai-agents/01-driving-helix.md)
is the operator manual for compiling and running Helix programs end to end.
