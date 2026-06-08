# Functions, control flow & pattern matching

*What this chapter covers:* how to declare and call functions in Helix (parameters — including
the more-than-six-argument case — the last-expression return rule, and the exit-code
convention); the control-flow forms `if`/`else` (an expression), `while` with `break` and
`continue`, and the `for` desugar; and `match` with its pattern set and the now-enforced
`if`-guards (the v1.1 "H4" hardening). Every complete program below is quoted verbatim from a
real committed `.hx` file that **the gate compiles and runs** ([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh),
step `[4]`, "FEATURE CORPUS"), and where the gate asserts an exit code this chapter cites the
`chk` line that does so.

This chapter assumes the language shape introduced in [the ten-minute tour](../part1-orientation/02-ten-minute-tour.md)
and the type vocabulary of the preceding Part III chapters; it does not re-explain types,
structs, or enums except where control flow touches them.

> **For AI agents:** the authoritative grammar lives in the self-hosted compiler's front end
> ([`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx)) and the as-built spec
> ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md)). When a feature
> here is marked gate-proven, the proof is a `chk "<path>" <exit>` row in
> [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh); key off that row, not off prose.

---

## Functions

A Helix function is `fn name(params) -> ReturnType { body }`. The body is a **single block
expression**, and the value of that expression is the function's result — there is no separate
`return` statement required for the normal case (early `return` exists and is covered below).
The canonical first program is a function that does nothing but yield a literal:

**Verified example** — [`helixc/examples/exit42.hx`](../../../helixc/examples/exit42.hx)
(the gate compiles this to a Linux ELF and asserts the process exits `42`:
`chk "$EX/exit42.hx" 42` in [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh)):

```helix
// First end-to-end Helix program: compiles, links, runs, exits with status 42.
//
// Demonstrates: function decl, return, integer literal.
// Verification: $? == 42 after running the produced ELF.

fn main() -> i32 {
    42
}
```

The `42` on its own line is not a typo for `return 42;` — it *is* the function's value. This is
the **last-expression return** rule: a block `{ s1; s2; tail }` evaluates each statement in
order and the block's value is the trailing expression `tail` (the one with no `;` after it).
In the parser this is the `AST_SEQ` node — *"Evaluate first (discard), then second (return its
value)"* (the AST tag legend in [`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx),
tag 13). A function body is just such a sequence, so its result is its last expression.

A slightly longer body shows the rule with intermediate `let` bindings, all of which are
discarded except the final expression:

**Verified example** — [`helixc/examples/matmul_2x2.hx`](../../../helixc/examples/matmul_2x2.hx)
(gate-checked to exit `69`: `chk "$EX/matmul_2x2.hx" 69`):

```helix
fn main() -> i32 {
    let a00 = 1; let a01 = 2; let a10 = 3; let a11 = 4;
    let b00 = 5; let b01 = 6; let b10 = 7; let b11 = 8;
    let c00 = a00 * b00 + a01 * b10;
    let c11 = a10 * b01 + a11 * b11;
    c00 + c11   // 19 + 50 = 69
}
```

The body runs eight `let` statements, then the trailing expression `c00 + c11` becomes the
function's value (`69`), which becomes the exit code.

### Parameters and types

Parameters are written `name: Type`, comma-separated. The default return type when `-> Ret` is
omitted is `i32` (the spec records this for `fn` items:
[`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §3). Recursion is
supported; a function may call itself or other functions declared anywhere at top level.

A small function with a typed parameter, called from another function, appears inside this
gate fixture (early-`return` is discussed in the control-flow section; here, note only the
`fn pick(k: i32) -> i32` signature and the `pick(1)` / `pick(2)` call sites):

**Fragment** (excerpt of [`stage0/helixc-bootstrap/corpus_gen/arm_early_return.hx`](../../../stage0/helixc-bootstrap/corpus_gen/arm_early_return.hx);
the full program is gate-checked to exit `42` via `chk "$GENC/arm_early_return.hx" 42`):

```helix
fn pick(k: i32) -> i32 {
    // ...
    0   // tail (unreached for k in {1,2})
}
fn main() -> i32 {
    // pick(1) takes the FIRST early return -> 42 ; pick(2) -> 7.
    pick(1) + (pick(2) - 7)   // 42 + 0 = 42
}
```

Float parameters route through the SSE path — `fn loss(x: f32) -> f32` is the parameter type the
autodiff example relies on (the parser records the parameter's type tag so an `f32` param
propagates to floating-point codegen: `AST_PARAM` slot `p4`,
[`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx) tag 18):

**Verified example** — [`helixc/examples/gradient_descent.hx`](../../../helixc/examples/gradient_descent.hx)
(gate-checked to exit `42`: `chk "$EX/gradient_descent.hx" 42`):

```helix
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

Note the trailing `(x_new + 39.0) as i32`: the function returns `i32`, the body's last
expression is an `f32` cast to `i32` with `as`, and that integer is the value. (Autodiff —
`grad` — is its own subject, covered in the later Part III autodiff chapter; it appears here
only to exhibit a real `f32` function signature.)

### More than six arguments

The System-V AMD64 ABI that Helix's CPU backend targets passes the first six integer arguments
in registers and the rest on the stack
([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §6, "6 int args in
registers"). Functions with **more than six parameters** are supported: the caller pushes
arguments seven and beyond on the stack and the callee reads them back from the frame. The gate
proves this directly with functions of 8, 9, and 11 parameters, generated inline in
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) (step `[4]`, the `gen f8_args.hx` /
`f9_args.hx` / `f11_args.hx` heredocs):

**Verified example** — `f11_args.hx`, generated and run by the gate (the `gen f11_args.hx`
heredoc in [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh); asserted to exit `66` by
`chk "$CD/f11_args.hx" 66`):

```helix
fn f11(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32, h: i32, i: i32, j: i32, k: i32) -> i32 { a+b+c+d+e+f+g+h+i+j+k }
fn main() -> i32 { f11(1,2,3,4,5,6,7,8,9,10,11) }
```

`f11(1,2,…,11)` sums to `66`; each argument is distinct, so a dropped or clobbered
seventh-and-beyond argument would change the sum and fail the assertion. The gate's own comment
records why this matters: before this support landed, *"`kovc` dropped params 7+ (callee bound
only rdi..r9, args 7+ trapped ud2 → SIGILL rc 132)"* — the corpus rows
`f8_args.hx`→36, `f9_args.hx`→45, `f11_args.hx`→66 lock the fix
([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh): `chk "$CD/f8_args.hx" 36; chk
"$CD/f9_args.hx" 45; chk "$CD/f11_args.hx" 66`).

> **Note:** an older comment inside [`helixc/examples/matmul_2x2.hx`](../../../helixc/examples/matmul_2x2.hx)
> mentions a "3 args per fn" cap — that is a stale note from the early v0.1 codegen, written
> when the example was authored, *not* a current limit. The current self-hosted `kovc` passes
> eleven arguments in the gate. When the comment in a source file and a gate row disagree, the
> gate row is the truth.

### The exit-code convention

A `main` that returns `i32` has its value used as the process exit status — which on Linux is an
**8-bit** value (`0`–`255`). This is why the corpus programs target small numbers like `42`,
`69`, and `66`: they fit in the exit byte, so the test harness can read the answer back with
`$?`. The gate's `chk` helper does exactly that — it runs the compiled binary and compares its
exit status to the expected number
([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh), the `chk()` function:
`timeout 10 /tmp/k2_out.bin; local rc=$?; if [ "$rc" = "$exp" ]`).

There is a second, sharper exit-code fact that matters the moment you script the compiler
itself, as opposed to a program it produces:

> **For AI agents:** the **`kovc` compiler binary** returns its **output byte count** (modulo
> 256) as its process exit status — so it exits **non-zero on success**. This is the *compiler's*
> convention, not the convention of programs it compiles (those return their `main` value, as
> above). The gate validates a self-compile by the output being **non-empty** plus the pinned
> SHA, **never** by `rc == 0` — its own comment: *"kovc returns its OUTPUT BYTE COUNT as the
> process exit status (rc = size mod 256 → 24 for the 698392-byte self-compile, i.e. NONZERO ON
> SUCCESS)"* ([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh), step `[2]`). Do not test
> a `kovc` invocation with `rc == 0`. This is **Trap 1** in the operator manual — see
> [Part IX — Traps](../part9-for-ai-agents/03-traps.md#trap-1--kovc-exits-non-zero-on-success).

Keeping these two conventions distinct is the single easiest way to misread a Helix build: a
*compiled program* answers through its `main` exit code; the *compiler* answers through
output-exists-and-matches-its-hash.

---

## Control flow

### `if` / `else` is an expression

`if` in Helix is an **expression**: it yields the value of whichever arm runs, so it can sit on
the right of a `let` or be a function's tail. The form is `if cond { a } else { b }`. The
condition must be an explicit boolean-valued expression (a comparison such as `a > b`); there is
no implicit integer-to-bool coercion ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md)
§4, "an **expression** yielding the taken arm's value").

The gate's smallest `if` programs are generated inline and assert which arm ran by its value:

**Verified example** — `cmp_ge.hx`, generated and run by the gate (the `gen cmp_ge.hx` heredoc
in [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh); asserted to exit `1` by
`chk "$CD/cmp_ge.hx" 1`):

```helix
fn main() -> i32 { if 5 >= 5 { 1 } else { 0 } }
```

The whole `if 5 >= 5 { 1 } else { 0 }` is `main`'s tail expression; `5 >= 5` is true, so the
expression's value is `1`, which is the exit code. Companion rows exercise the rest of the
comparison set the same way — `cmp_ne.hx` (`5 != 3` → `1`), `cmp_le.hx` (`3 <= 5` → `1`), each
with its own `chk` assertion ([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh):
`chk "$CD/cmp_ne.hx" 1; chk "$CD/cmp_ge.hx" 1; chk "$CD/cmp_le.hx" 1`).

In the parser an `if` builds an `AST_IF` node with three slots — *"p1 = cond, p2 = then, p3 =
else"* ([`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx), tag 7). Two
behaviors are worth knowing:

- **A missing `else` yields `0`.** If you write `if c { a }` with no `else`, the parser supplies
  `AST_INT(0)` as the else branch ([`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx),
  the `else_e = mk_node(0, 0, 0, 0)` fallback in the `if`-expression case). So an `else`-less
  `if` used in value position evaluates to `0` when the condition is false.
- **There is no `else if` keyword — you nest.** Helix has no distinct `else if` token; an
  `else if` is parsed as `else { if … }`, i.e. a nested `if` in the else arm
  ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §4, "No `else if`
  keyword — nest in the `else` arm"; the parser detects a following `if` after `else` and recurses,
  [`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx) around the `is_elif` branch).
  In source you may still *write* `else if` and it parses, because it is identical to a nested
  `if` after `else`.

A real nested-`if` cascade (here as early returns; the shape is the same in value position) is
the body of `pick` in the early-return fixture quoted below.

### `while`, `break`, and `continue`

The looping primitive is `while cond { body }`. A `while` loop is a **statement-shaped**
expression: it always evaluates to `0` (the parser's AST legend: *"AST_WHILE … Always returns
0"*, [`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx) tag 10) — you loop for
effect (mutating `let mut` bindings), not for a value. The two canonical loop fixtures are
generated by the gate:

**Verified example** — `while_sum.hx`, generated and run by the gate (the `gen while_sum.hx`
heredoc in [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh); asserted to exit `10` by
`chk "$CD/while_sum.hx" 10`):

```helix
fn main() -> i32 { let mut s = 0; let mut i = 0; while i < 5 { s = s + i; i = i + 1; } s }
```

This sums `0+1+2+3+4 = 10`. Note the pieces a counting loop needs: `let mut` for the mutable
accumulator and counter, the `while i < 5` condition, the in-body reassignment `i = i + 1`, and
the trailing `s` that becomes the function's value *after* the loop (the loop itself is `0`).

`break` exits the innermost loop early:

**Verified example** — `while_break.hx`, generated and run by the gate (the `gen while_break.hx`
heredoc in [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh); asserted to exit `7` by
`chk "$CD/while_break.hx" 7`):

```helix
fn main() -> i32 { let mut i = 0; while i < 100 { i = i + 1; if i >= 7 { break; } } i }
```

The condition `i < 100` would run far longer, but `break` fires when `i` reaches `7`, so the
function returns `7`. `break` is `AST_BREAK` (tag 77); the parser also accepts the optional
value form `break <expr>` so that `let x = loop { break 42; }` can evaluate to `42`
([`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx), `parse_break`).

`continue` restarts the innermost loop. It is exercised by a gate fixture that skips selected
iterations:

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/arm_continue.hx`](../../../stage0/helixc-bootstrap/corpus_gen/arm_continue.hx)
(gate-checked to exit `42`: `chk "$GENC/arm_continue.hx" 42`):

```helix
fn main() -> i32 {
    // `continue` in a while loop: sum 1..=10 but SKIP 3 and 7 via continue.
    // 1+2+4+5+6+8+9+10 = 45 ; then +... no: sum = 55 - 3 - 7 = 45. Adjust to 42:
    // skip 3, 7, AND 13 (out of range) -> 55 - 10 = 45 ; subtract 3 more below.
    let mut i: i32 = 0;
    let mut acc: i32 = 0;
    while i < 10 {
        i = i + 1;
        if i == 3 { continue; }   // skip 3
        if i == 7 { continue; }   // skip 7
        acc = acc + i;
    }
    // acc = (1+2+4+5+6+8+9+10) = 45 ; 45 - 3 = 42
    acc - 3
}
```

The two `continue`s skip adding `3` and `7` to `acc`, leaving `45`, and the tail `acc - 3`
yields `42`. Notice that the counter `i = i + 1` is incremented at the **top** of the body,
before any `continue` — that placement is deliberate, and it is the key to the `for` caveat
below.

> **For AI agents:** `continue` is sound in a hand-written `while` (and in `loop { }`) because
> you control where the increment sits. In a `for` loop it is **not** safe — see the next
> section. The compiler's own note: in a desugared `for`, *"continue skips the auto-increment and
> risks an infinite loop — use plain `while` if continue is needed"*
> ([`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx), `parse_continue`).

### `for` (a parser desugar over ranges)

Helix has a `for` loop over integer ranges, but it is honestly a **parser desugar**, not a
distinct control form: the as-built spec lists `for` among the "parser desugars (to `while` /
… )" rather than as core surface ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md)
§7). `for v in start..end { body }` is rewritten by `parse_for` into an `AST_LET_MUT` for the
loop variable, an `AST_WHILE` with the bound as its condition, and an `AST_SEQ` of (your body
then the auto-increment) — using only existing AST tags, no new codegen
([`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx), `parse_for`). The
exclusive form `..` lowers the condition to `AST_LT` (tag 6); the inclusive form `..=` lowers it
to `AST_LE` (tag 22), so the body also runs at `v == end`.

The gate proves all three range shapes in one program:

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/M1_for_loop.hx`](../../../stage0/helixc-bootstrap/corpus_gen/M1_for_loop.hx)
(gate-checked to exit `42`: `chk "$GENC/M1_for_loop.hx" 42`):

```helix
fn main() -> i32 {
    let mut sum_a = 0;
    for i in 0..9 { sum_a = sum_a + i; }        // exclusive: 0..8 -> 36

    let mut sum_b = 0;
    for j in 1..=5 { sum_b = sum_b + j; }       // inclusive: 1..5 -> 15

    let lo = 2;
    let hi = 6;
    let mut sum_c = 0;
    for k in lo..hi { sum_c = sum_c + k; }      // var bounds: 2..5 -> 14

    sum_a - sum_b + sum_c + 7                    // 36 - 15 + 14 + 7 = 42
}
```

`0..9` runs `i` over `0..8` (exclusive of `9`) summing to `36`; `1..=5` runs `j` over `1..5`
*inclusive* summing to `15`; `lo..hi` with `lo = 2, hi = 6` runs `2..5` summing to `14`. The
range bounds may themselves be expressions or variables (the `lo..hi` case). A broken `for`
(wrong bound, missing increment, or inclusive-treated-as-exclusive) would change one of these
sums and move the exit code off `42` — which is exactly what the assertion catches.

Because the desugar appends the auto-increment **after** your body inside the loop's sequence,
`continue` inside a `for` would jump past the increment and spin forever; that is the precise
reason the spec and the parser steer you to a plain `while` when you need `continue`.

> **Note:** the self-hosted compiler's *own* source uses plain `while` everywhere and never uses
> `for` — which is why promoting `for` to a gate-proven feature leaves the self-host fixpoint
> byte-identical: the desugar is exercised only by `M1_for_loop.hx` compiled through the gate's
> `K2`, not by the compiler compiling itself ([the fixture's own header comment](../../../stage0/helixc-bootstrap/corpus_gen/M1_for_loop.hx)).

### Early `return`

Beyond last-expression return, a function may exit early with `return e;`. The early-return
fixture exercises returning from inside an `if` and from inside a loop:

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/arm_early_return.hx`](../../../stage0/helixc-bootstrap/corpus_gen/arm_early_return.hx)
(gate-checked to exit `42`: `chk "$GENC/arm_early_return.hx" 42`):

```helix
fn pick(k: i32) -> i32 {
    // early `return` from inside control flow (before the function's tail).
    if k == 1 { return 42; }
    if k == 2 { return 7; }
    let mut i: i32 = 0;
    while i < 100 {
        if i == 5 { return 99; }   // early return out of a loop
        i = i + 1;
    }
    0   // tail (unreached for k in {1,2})
}
fn main() -> i32 {
    // pick(1) takes the FIRST early return -> 42 ; pick(2) -> 7.
    pick(1) + (pick(2) - 7)   // 42 + 0 = 42
}
```

`pick(1)` hits `return 42;` immediately; `pick(2)` hits `return 7;`; the `0` tail and the
`return 99;` inside the loop are unreached for those inputs. `main`'s tail `pick(1) + (pick(2) -
7)` is `42 + 0 = 42`.

---

## Pattern matching with `match`

`match` is an **expression** that tests a scrutinee against a series of `pat => body` arms,
comma-separated, and yields the body of the first arm whose pattern matches. The full pattern
set the self-hosted compiler supports is enumerated in the spec
([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §4): a binding `x`,
the wildcard `_`, a literal `42`, a range `a..b`, a tuple `(a, b)`, a struct pattern
(`P { x, y }`, `P { x: 0, y }`, nested, or `P { .. }`), an enum variant `E::V(x)`, and an
or-pattern `A | B`.

The cleanest payload-and-binding example matches over user-defined enums:

**Verified example** — [`helixc/examples/hbs_sample_option.hx`](../../../helixc/examples/hbs_sample_option.hx)
(gate-checked to exit `42`: `chk "$EX/hbs_sample_option.hx" 42`):

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

Each `match` is used in value position — `let v1 = match m1 { … };` — and the arm
`Maybe::Some(x) => x` both matches the `Some` variant and binds its payload to `x`. (`Ok`/`Err`/
`Result` are *not* compiler builtins in Helix; they are ordinary user-defined enums of exactly
this shape — see [`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §7,
"v1.0 SCOPE DECISIONS", and the `result_inline.hx` corpus row.)

Or-patterns, range patterns, and a wildcard catch-all are each pinned by their own generated
fixture:

**Verified example** — `match_or.hx`, generated and run by the gate (the `gen match_or.hx`
heredoc in [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh); asserted to exit `10` by
`chk "$CD/match_or.hx" 10`):

```helix
fn main() -> i32 { let x = 2; match x { 1 | 2 | 3 => 10, _ => 0 } }
```

**Verified example** — `match_range.hx`, generated and run by the gate (the `gen match_range.hx`
heredoc in [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh); asserted to exit `1` by
`chk "$CD/match_range.hx" 1`):

```helix
fn main() -> i32 { let x = 5; match x { 1..10 => 1, _ => 0 } }
```

In `match_or`, `x = 2` matches `1 | 2 | 3` and the arm yields `10`; in `match_range`, `x = 5`
falls in the range `1..10` and yields `1`. The `_` wildcard is the catch-all default arm.

Struct destructuring in patterns is exercised end-to-end by a dedicated showcase — flat
destructuring, a literal field test, nested destructuring, and the ignore-rest `..` form, all in
one program:

**Verified example** — [`helixc/examples/dogfood_18_pat_struct_showcase.hx`](../../../helixc/examples/dogfood_18_pat_struct_showcase.hx)
(gate-checked to exit `42`: `chk "$EX/dogfood_18_pat_struct_showcase.hx" 42`). The four match
shapes it proves are:

```helix
// 1. Flat destructuring: bind both fields by name.
fn check_flat() -> i32 {
    let p = Point { x: 10, y: 7 };
    match p {
        Point { x, y } => x + y,
    }
}

// 2. Literal field test: first arm matches only when x==0.
fn check_literal_arm() -> i32 {
    let p = Point { x: 0, y: 17 };
    match p {
        Point { x: 0, y } => y - 5,
        Point { .. } => 0,
    }
}

// 3. Nested destructuring: bind through two levels of struct.
fn check_nested() -> i32 {
    let o = Outer { inside: Inner { value: 13 }, tag: 4 };
    match o {
        Outer { inside: Inner { value }, tag } => value - tag - 9,
        _ => 0,
    }
}

// 4. Ignore-rest: don't bind any field, just match the shape.
fn check_ignore_rest() -> i32 {
    let p = Point { x: 99, y: 999 };
    match p {
        Point { .. } => 42,
    }
}
```

(The above is quoted from the body of
[`helixc/examples/dogfood_18_pat_struct_showcase.hx`](../../../helixc/examples/dogfood_18_pat_struct_showcase.hx);
its `main` combines the four results so the program exits `42` only if every pattern shape
evaluates correctly.) `Point { x: 0, y }` shows a **literal field test** — the arm matches only
when `x` is `0` — alongside a bound field `y`; `Outer { inside: Inner { value }, tag }` shows
**nested** patterns; `Point { .. }` shows the **ignore-rest** form.

### Pattern guards — the H4 hardening

A `match` arm may carry an **`if`-guard**: `pat if cond => body`. The guard is an extra boolean
condition, evaluated *after* the pattern structurally matches (with the pattern's bindings in
scope); if the guard is false, the arm is skipped and matching falls through to the next arm.

This is worth a careful, honest note, because the behavior changed. The **frozen v1.0 language
surface** (the spec's §4 table) still records guards as parsed-but-not-enforced — *"Guards `pat
if cond =>` are parsed but NOT enforced — every matching arm body runs regardless of the
guard"* ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §4, marked
`[erased]`). That was true at v1.0 freeze. The **v1.1 hardening line "H4"** then closed the gap:
match-arm guards are now **evaluated**. The hardening record states it plainly: *"match-arm `if
cond` guards are now **evaluated** (were parsed-and-discarded / always-true). Parser stores the
guard in the arm node; `emit_one_match_arm` evaluates it after the pattern matches … and falls
through to the next arm when false"* ([`docs/HELIX_V1_1_HARDENING.md`](../../../docs/HELIX_V1_1_HARDENING.md),
criterion **H4**, GREEN 2026-06-01).

You can see the change in the parser itself: the guard, formerly parsed and discarded, is now
**captured** into the match arm's slot 4 so codegen can evaluate it
([`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx), `parse_match_expr` — the
`H4 (2026-06-01): CAPTURE the guard (was parse-and-discard) into arm slot 4` comment, and the
`__arena_push(first_guard); // arm slot 4 = guard expr (0 = none)` lines).

H4 is locked by three gate fixtures — a guard that passes, one that fails (so the arm is
skipped), and a chain where the first guard fails and the second wins:

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/g1_guard_true.hx`](../../../stage0/helixc-bootstrap/corpus_gen/g1_guard_true.hx)
(gate-checked to exit `1`: `chk "$GENC/g1_guard_true.hx" 1`):

```helix
fn main() -> i32 {
    let x = 7;
    match x {
        n if n > 5 => 1,
        _ => 0
    }
}
```

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/g2_guard_false.hx`](../../../stage0/helixc-bootstrap/corpus_gen/g2_guard_false.hx)
(gate-checked to exit `0`: `chk "$GENC/g2_guard_false.hx" 0`):

```helix
fn main() -> i32 {
    let x = 3;
    match x {
        n if n > 5 => 1,
        _ => 0
    }
}
```

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/g3_guard_chain.hx`](../../../stage0/helixc-bootstrap/corpus_gen/g3_guard_chain.hx)
(gate-checked to exit `2`: `chk "$GENC/g3_guard_chain.hx" 2`):

```helix
fn main() -> i32 {
    let x = 7;
    match x {
        n if n > 100 => 1,
        n if n > 5 => 2,
        _ => 0
    }
}
```

These three are the whole proof. In `g1`, the pattern `n` binds `7` and the guard `n > 5` is
true, so the arm fires and the program exits `1`. In `g2`, `x` is `3`: the pattern `n` would
match, but the guard `n > 5` is **false**, so that arm is *skipped* and matching falls through to
`_ => 0` — the program exits `0`. (Under the old, erased behavior the first arm would have fired
regardless and the program would have exited `1`; the gate's expected `0` is precisely what
distinguishes enforced guards from erased ones.) In `g3`, `n > 100` fails, the chain falls
through to `n > 5` which succeeds, and the program exits `2` — the *second* guard wins.

> **For AI agents:** the three rows `chk "$GENC/g1_guard_true.hx" 1; chk "$GENC/g2_guard_false.hx"
> 0; chk "$GENC/g3_guard_chain.hx" 2` ([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh),
> the "H4 pattern guards corpus" block) are the live contract that guards are evaluated. The
> `[erased]` mark in the frozen-surface spec §4 predates H4 and is superseded by it; trust the
> gate row. If you are checking whether a guard is enforced, the `g2`→`0` row is the
> discriminator — an erased guard would exit `1`.

### What `match` does *not* check (honest residuals)

Two limits are real and documented, so do not rely on the compiler to catch them:

- **No exhaustiveness checking.** A `match` that omits arms is accepted and simply runs the arms
  it has; there is no "non-exhaustive patterns" error. This is an *unenforced-by-design* bound,
  locked by a corpus row that proves `kovc` accepts a non-exhaustive payload-enum match
  ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §7, "Match
  exhaustiveness"; the `L3_nonexhaustive_bound` row in
  [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh)).
- **Guards are not used for exhaustiveness reasoning.** Because there is no exhaustiveness check
  at all, a set of guarded arms with no catch-all is likewise accepted as-is. Provide a `_`
  arm (as `g1`/`g2`/`g3` do) when you need a defined result for every input.

These are stated so the picture is precise: `match` *binds and dispatches* (including enforced
guards), but it does **not** prove totality. For the full list of by-design unenforced bounds
(borrows, `const`/`static`, module privacy, match exhaustiveness, bare non-i32 generics) see
[`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §7, each locked by a
`*_bound` corpus row.

---

## Summary

- A function is `fn name(params) -> Ret { body }`; the body's **last expression** is its value
  (no `return` needed for the normal case), and `return e;` exits early.
- Parameters beyond the sixth are passed on the stack and work correctly — proven by the gate's
  8-, 9-, and 11-argument fixtures.
- `main`'s `i32` value becomes the **8-bit process exit code**; the *compiler binary* `kovc`,
  by contrast, exits non-zero on success (output byte count) — never test it with `rc == 0`.
- `if`/`else` is an expression (missing `else` ⇒ `0`; `else if` is a nested `if`); `while` loops
  for effect and always evaluates to `0`, with `break` and `continue`; `for` over `..`/`..=`
  ranges is a parser desugar to `while` (and `continue` is unsafe inside it).
- `match` is an expression over a rich pattern set (binding, wildcard, literal, range, tuple,
  struct destructure, enum variant, or-pattern); its **`if`-guards are enforced** as of the v1.1
  H4 hardening (gate rows `g1`/`g2`/`g3`). It does **not** check exhaustiveness.

---

**Next:** [Generics, traits & closures](04-generics-traits-closures.md) — how `<T>` turbofish
monomorphization, trait-method dispatch, and capturing closures-as-values work in the
self-hosted compiler, with the honest residuals of each.
