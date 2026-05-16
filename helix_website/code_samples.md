# Helix Code Samples — Ready for Website

30 code samples, each self-contained, each demonstrating a specific feature. Use these directly on the website (Features page, Learn tutorial, Playground examples).

Each sample shows: source code · expected output · feature demonstrated.

---

## 1. Hello, 42

The canonical first program.

```rust
fn main() -> i32 { 42 }
```

**Expected output:** exit code 42  
**Demonstrates:** function declaration, return type, integer literal

---

## 2. Variables and arithmetic

```rust
fn main() -> i32 {
    let x = 5;
    let y = x * x;
    y - 8
}
```

**Expected output:** 17  
**Demonstrates:** `let` bindings, immutability, expression-as-value

---

## 3. Mutable state

```rust
fn main() -> i32 {
    let mut counter = 0;
    counter = counter + 1;
    counter = counter + 41;
    counter
}
```

**Expected output:** 42  
**Demonstrates:** `let mut`, assignment

---

## 4. If-else as expression

```rust
fn classify(n: i32) -> i32 {
    if n < 0 { 1 }
    else if n == 0 { 2 }
    else { 3 }
}

fn main() -> i32 { classify(0) }
```

**Expected output:** 2  
**Demonstrates:** if/else if/else as expression

---

## 5. Recursion

```rust
fn fib(n: i32) -> i32 {
    if n < 2 { n }
    else { fib(n - 1) + fib(n - 2) }
}

fn main() -> i32 { fib(10) }
```

**Expected output:** 55  
**Demonstrates:** recursive functions, totality (compiler proves termination via structural recursion)

---

## 6. While loops

```rust
fn main() -> i32 {
    let mut sum = 0;
    let mut i = 1;
    while i <= 10 {
        sum = sum + i;
        i = i + 1;
    }
    sum
}
```

**Expected output:** 55  
**Demonstrates:** mutable counter, while loop

---

## 7. Tuples

```rust
fn main() -> i32 {
    let t = (10, 20, 30);
    t.0 + t.1 + t.2
}
```

**Expected output:** 60  
**Demonstrates:** tuple literals, positional field access

---

## 8. Arrays

```rust
fn main() -> i32 {
    let arr = [1, 2, 3, 4, 5];
    arr[0] + arr[2] + arr[4]
}
```

**Expected output:** 9  
**Demonstrates:** array literals, indexing

---

## 9. Structs (basic)

```rust
struct Pt { x: i32, y: i32 }

fn area(p: Pt) -> i32 {
    p.x * p.y
}

fn main() -> i32 {
    area(Pt { 6, 7 })
}
```

**Expected output:** 42  
**Demonstrates:** struct declaration, struct literal, by-value pass to function, named field access

---

## 10. Structs (nested)

```rust
struct Pt { x: i32, y: i32 }
struct Line { from: Pt, to: Pt }

fn main() -> i32 {
    let l = Line { Pt { 10, 0 }, Pt { 0, 32 } };
    l.from.x + l.to.y
}
```

**Expected output:** 42  
**Demonstrates:** nested struct, chained field access

---

## 11. Enums (unit variants)

```rust
enum Color { Red, Green, Blue }

fn main() -> i32 {
    let c = Color::Green;
    c
}
```

**Expected output:** 1  
**Demonstrates:** enum declaration, variant construction, discriminant value

---

## 12. Enums with payloads

```rust
enum Maybe { None, Some(i32) }

fn extract(m: Maybe) -> i32 {
    match m {
        Maybe::None    => 0,
        Maybe::Some(v) => v,
    }
}

fn main() -> i32 {
    extract(Maybe::Some(42))
}
```

**Expected output:** 42  
**Demonstrates:** payload variant, pattern matching with binding

---

## 13. Pattern matching with ranges

```rust
fn classify(n: i32) -> i32 {
    match n {
        0       => 100,
        1..10   => 200,
        10..=20 => 300,
        _       => 999,
    }
}

fn main() -> i32 { classify(15) }
```

**Expected output:** 300  
**Demonstrates:** literal pattern, exclusive range, inclusive range, wildcard

---

## 14. Tuple patterns

```rust
fn main() -> i32 {
    let p = (1, 2);
    match p {
        (0, _) => 100,
        (1, y) => y,
        _      => 0,
    }
}
```

**Expected output:** 2  
**Demonstrates:** tuple destructure pattern, binding pattern

---

## 15. Generic functions

```rust
fn id<T>(x: T) -> T { x }

fn main() -> i32 {
    id::<i32>(42)
}
```

**Expected output:** 42  
**Demonstrates:** generics, monomorphization, turbofish

---

## 16. Traits

```rust
trait Eq {
    fn eq(self, other: Self) -> i32;
}

impl Eq for i32 {
    fn eq(self, other: i32) -> i32 {
        if self == other { 1 } else { 0 }
    }
}

fn main() -> i32 {
    let a = 5;
    let b = 5;
    a.eq(b)
}
```

**Expected output:** 1  
**Demonstrates:** trait declaration, impl block, method-call sugar, Self type

---

## 17. Bounded generic

```rust
trait Eq { fn eq(self, other: Self) -> i32; }

impl Eq for i32 {
    fn eq(self, other: i32) -> i32 {
        if self == other { 1 } else { 0 }
    }
}

fn cmp<T: Eq>(a: T, b: T) -> i32 {
    T::eq(a, b)
}

fn main() -> i32 {
    cmp::<i32>(5, 5)
}
```

**Expected output:** 1  
**Demonstrates:** trait bound, generic with constraint

---

## 18. Closures

```rust
fn main() -> i32 {
    let a = 10;
    let c = |x| x + a;
    c(5)
}
```

**Expected output:** 15  
**Demonstrates:** closure literal, free-variable capture

---

## 19. Modules

```rust
mod geometry {
    fn rect_area(w: i32, h: i32) -> i32 {
        w * h
    }
}

fn main() -> i32 {
    geometry::rect_area(6, 7)
}
```

**Expected output:** 42  
**Demonstrates:** module declaration, path expression

---

## 20. Module with `use`

```rust
mod math {
    fn double(n: i32) -> i32 { n * 2 }
}

use math::double;

fn main() -> i32 {
    double(21)
}
```

**Expected output:** 42  
**Demonstrates:** `use` decl, alias to mangled name

---

## 21. Forward-mode autodiff

```rust
fn loss(x: f64) -> f64 { x * x + 3.0_f64 * x }

fn main() -> f64 {
    grad(loss)(2.0_f64)
}
```

**Expected output:** 7.0_f64  
**Demonstrates:** automatic differentiation, `grad(f)(x)` syntax

**Math:** `f(x) = x² + 3x`, so `f'(x) = 2x + 3`, and `f'(2) = 7`.

---

## 22. Reverse-mode autodiff

```rust
fn loss(x: f64, y: f64) -> f64 { x * y + x * x }

struct Grad { dx: f64, dy: f64 }

fn main() -> f64 {
    let g = grad_rev_all(loss)(2.0_f64, 3.0_f64);
    g.dx
}
```

**Expected output:** 7.0_f64  
**Demonstrates:** reverse-mode AD, multi-output struct return

**Math:** `∂L/∂x = y + 2x = 3 + 4 = 7`.

---

## 23. AD across user-defined functions

```rust
fn helper(x: f64) -> f64 { x * x }
fn loss(x: f64) -> f64 { helper(x) + x }

fn main() -> f64 {
    grad(loss)(3.0_f64)
}
```

**Expected output:** 7.0_f64  
**Demonstrates:** AD inlines user-defined helper functions automatically

---

## 24. @checkpoint rematerialization

```rust
@checkpoint
fn deep_block(x: f64) -> f64 { x * x * x * x * x }

fn loss(x: f64) -> f64 { deep_block(x) + x }

fn main() -> f64 {
    grad_rev_all(loss)(2.0_f64).dx
}
```

**Expected output:** 81.0_f64  
**Demonstrates:** `@checkpoint` attribute trades memory for compute in reverse-mode AD

**Math:** `d/dx (x⁵ + x) = 5x⁴ + 1`, at `x=2` gives `5·16 + 1 = 81`.

---

## 25. Tile literal and access

```rust
fn main() -> f32 {
    let t = tile<f32, [4, 4], REG>::ones();
    t.get(2, 3)
}
```

**Expected output:** 1.0_f32  
**Demonstrates:** tile type, shape parameter, memspace, init constructor

---

## 26. Tile matmul

```rust
fn main() -> f32 {
    let a = tile<f32, [4, 4], REG>::ones();
    let b = tile<f32, [4, 4], REG>::ones();
    let c = tile_matmul(a, b);
    c.get(2, 3)
}
```

**Expected output:** 4.0_f32  
**Demonstrates:** tile_matmul (sum of 4 ones × ones = 4)

---

## 27. Reflection: Quote and Splice

```rust
fn main() -> i32 {
    let h = Quote(1 + 2);
    Splice(h)
}
```

**Expected output:** 3  
**Demonstrates:** reflection cell allocation, value retrieval

---

## 28. Reflection: modify with verifier

```rust
fn always_true(_: i32) -> i32 { 1 }

fn main() -> i32 {
    let h = Quote(0);
    modify(h, 42, always_true(0));
    Splice(h)
}
```

**Expected output:** 42  
**Demonstrates:** verifier-gated cell mutation

---

## 29. Reflection: failed verifier

```rust
fn always_false(_: i32) -> i32 { 0 }

fn main() -> i32 {
    let h = Quote(0);
    modify(h, 42, always_false(0));
    Splice(h)
}
```

**Expected output:** 0  
**Demonstrates:** verifier failure preserves original cell value

---

## 30. FFI: calling libc

```rust
extern "C" fn puts(s: *const u8) -> i32;

fn main() -> i32 {
    let msg: *const u8 = "hello\n\0".as_ptr();
    puts(msg)
}
```

**Expected output:** prints "hello", returns 6  
**Demonstrates:** `extern "C"`, pointer types, FFI call

---

## Suggested grouping for the website

### "Getting started" block (Hero + Learn intro)
- Sample #1: Hello, 42
- Sample #2: Variables
- Sample #5: Recursion (fib)

### "Real language" block (Features page)
- Sample #9: Structs
- Sample #11: Enums
- Sample #13: Match with ranges
- Sample #15: Generics
- Sample #16: Traits
- Sample #18: Closures

### "ML-first" block (the showcase)
- Sample #21: grad(loss)(x) → 7
- Sample #22: grad_rev_all → multi-gradient
- Sample #24: @checkpoint
- Sample #26: tile_matmul

### "Bootstrap + Reflection" block (the moats)
- Sample #27: Quote/Splice
- Sample #28: Verifier-gated modify
- Sample #30: FFI puts hello
