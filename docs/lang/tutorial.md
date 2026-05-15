# Helix tutorial — getting started

This guide walks through the core of Helix in 10 short steps, ending with
a program that uses every unique compile-time AGI feature.

Every example here parses, type-checks, and (where applicable) compiles
through the current `helixc` toolchain. Examples run through the same
test suite that ships with the repo.

---

## 1. Hello, exit code

The simplest Helix program returns an integer that becomes the process
exit code.

```kov
fn main() -> i32 {
    42
}
```

Compile and run:

```bash
python -m helixc.backend.x86_64 hello.hx hello.bin
./hello.bin
echo $?     # prints: 42
```

---

## 2. Variables, mutation

```kov
fn main() -> i32 {
    let x = 10;            // immutable
    let mut y = 20;        // mutable
    y += 12;
    x + y                  // last expression is the block value
}
```

`let` is immutable by default. `let mut` makes a cell that can be
re-assigned. Compound operators (`+=`, `-=`, `*=`, `/=`, `%=`) work on
mutable variables.

---

## 3. Numeric types

| Type | Bits | Notes |
|---|---|---|
| `i32` `i64` `i8` `i16` | signed | default integer is `i32` |
| `u32` `u64` etc. | unsigned | parsed but not yet codegen'd |
| `bf16` `f16` `f32` `f64` | floats | default float is `f32` |
| `fp8` `mxfp4` `nvfp4` | low-precision | parsed; codegen TBD |
| `ternary` | {-1, 0, +1} | parsed; codegen TBD |
| `bool` | true/false |  |

Cast between types with `as`:

```kov
let x: f32 = 3.14;
let y: i32 = x as i32;     // = 3
```

---

## 4. Functions and recursion

```kov
fn fib(n: i32) -> i32 {
    if n < 2 {
        n
    } else {
        fib(n - 1) + fib(n - 2)
    }
}

fn main() -> i32 {
    fib(9)                 // = 34
}
```

Up to 6 arguments via System V ABI (`rdi/rsi/rdx/rcx/r8/r9`).

---

## 5. Loops

```kov
fn main() -> i32 {
    let mut total = 0;
    for i in 0 .. 10 {
        total += i;
    }
    total                  // = 45
}
```

`while` is also available:

```kov
let mut n = 5;
let mut result = 1;
while n > 1 {
    result *= n;
    n -= 1;
}
// result = 120
```

---

## 6. Arrays

```kov
fn main() -> i32 {
    let xs = [10, 20, 12];
    xs[0] + xs[1] + xs[2]   // = 42
}
```

Mutate elements:

```kov
let xs = [0, 0, 0];
xs[0] = 10;
xs[1] += 32;
```

---

## 7. Tensor types with shape parameters

This is where Helix starts to diverge from C-likes:

```kov
fn matmul[N: size, M: size, P: size](
    a: tensor<f32, [N, M]>,
    b: tensor<f32, [M, P]>,
) -> tensor<f32, [N, P]>
where
    N % 16 == 0,
    M % 16 == 0,
    P % 16 == 0,
{
    a    // (placeholder — tensor codegen lands in Phase 4)
}
```

`N`, `M`, `P` are *size variables* — compile-time integers.
The `where` clause adds Presburger constraints. The compiler
**rejects calls** that can't satisfy the shapes:

```kov
fn caller(x: tensor<f32, [4, 5]>) {
    matmul(x, x);   // error: shape constraint violated (5 == 4)
}
```

---

## 8. Differentiable types `D<T>`

When a value participates in gradient computation, wrap its type in `D<T>`:

```kov
@pure
fn loss(x: D<f32>, y: D<f32>) -> D<f32> {
    let diff = x - y;
    diff * diff
}
```

The compiler propagates D-ness through operations. Returning a non-D
from a D-typed function body is a compile error (you'd silently lose
the gradient).

To leave the differentiable world explicitly:

```kov
let plain = detach(x);     // D<f32> -> f32
let back = attach(plain);  // f32 -> D<f32>
```

---

## 9. Effect/capability types

Functions declare what they're allowed to do. The compiler enforces:

```kov
@pure
fn safe(x: i32) -> i32 { x * x }       // no I/O, no mutation, deterministic

@io
fn read_config() -> i32 { 0 }          // can read files

@io
fn driver() -> i32 {
    let cfg = read_config();           // OK: caller has @io
    safe(cfg)                          // OK: pure callable from anywhere
}

@pure
fn would_be_bad() -> i32 {
    read_config()                      // ERROR: @pure can't call @io
}
```

Available effect names (extensible): `io`, `network`, `modify_self`,
`rng`, `time`, `fs`. A function with `@pure` cannot have any effects.

---

## 10. Memory-tier types

Cognitive-architecture distinction at the type level:

```kov
fn process(
    perception: WorkingMem<i32>,        // current task
    experience: EpisodicMem<i32>,       // a single timestamped event
) -> SemanticMem<i32> {
    let knowledge = consolidate(experience);   // Episodic -> Semantic
    knowledge
}

fn use_knowledge(s: SemanticMem<i32>) -> WorkingMem<i32> {
    recall(s)                                  // Semantic -> Working
}
```

Cross-tier transitions require the explicit operators `consolidate`
and `recall`. Direct assignment between tiers is a compile error.

---

## All-in-one

The single function below combines several AGI-oriented type-level Helix
features:

```kov
@pure
fn agi_step[N: size](
    sensory: WorkingMem<tensor<f32, [N]>>,
    weights: D<tensor<f32, [N, N]>>,
) -> WorkingMem<tensor<f32, [N]>>
where N % 16 == 0
{
    sensory
}
```

In one signature:
- shape constraint (`N % 16 == 0`)
- gradient-tracked weights (`D<...>`)
- working-memory inputs/outputs
- `@pure` (no I/O, no self-modification)

This combination is an intended Helix differentiator at the type level.

---

## Where to go from here

- Run `python helixc/tests/test_typecheck.py` to see ~38 examples.
- Read `helixc/examples/*.hx` for working programs.
- See `docs/lang/spec.md` for the full language reference.
- See `docs/lang/agi-features.md` for the AGI-specific feature deep-dive.
