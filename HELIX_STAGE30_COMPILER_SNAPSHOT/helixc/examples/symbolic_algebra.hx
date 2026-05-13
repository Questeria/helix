// Symbolic algebra engine: differentiate and simplify expressions
// represented as a recursive enum, then evaluate the result.
//
// We work with a small algebra over integer literals and a single
// variable x:
//
//   Expr = Lit(i32)
//        | Var
//        | Add(Expr, Expr)
//        | Mul(Expr, Expr)
//        | Pow(Expr, i32)        // Pow takes a literal exponent only
//
// The engine implements:
//   diff(e)        — symbolic differentiation w.r.t. x, by structural
//                    pattern match on `e`. d/dx of x is 1, d/dx of a
//                    constant is 0, sum/product/power rules.
//   simplify(e)    — fold trivial identities: 0+e=e, e+0=e, 0*e=0,
//                    1*e=e, e*1=e, x^0=1, x^1=x.
//   eval_at(e, x)  — substitute a value for x and evaluate to i32.
//
// Demonstration: differentiate x^3 + 2*x, simplify, evaluate at x=5.
//   d/dx(x^3 + 2*x) = 3*x^2 + 2.
//   At x=5: 3*25 + 2 = 77.
//
// Exercises: recursive enum (arena-indirect), pattern matching with
// payload binders, recursive helpers, multi-stage rewrites.

// Tag values in the arena: slot 0 of a node holds the tag, slots 1..3
// hold payload arena indices (or literal i32 for tag-1 / tag-4).
//   0 = Lit(value)
//   1 = Var
//   2 = Add(l, r)
//   3 = Mul(l, r)
//   4 = Pow(base, n)        // n is a literal i32 in the slot

// ---------------------------------------------------------------
// Builders
// ---------------------------------------------------------------
fn mk_lit(v: i32) -> i32 {
    let i = __arena_push(0);
    __arena_push(v);
    __arena_push(0);
    __arena_push(0);
    i
}

fn mk_var() -> i32 {
    let i = __arena_push(1);
    __arena_push(0);
    __arena_push(0);
    __arena_push(0);
    i
}

fn mk_add(l: i32, r: i32) -> i32 {
    let i = __arena_push(2);
    __arena_push(l);
    __arena_push(r);
    __arena_push(0);
    i
}

fn mk_mul(l: i32, r: i32) -> i32 {
    let i = __arena_push(3);
    __arena_push(l);
    __arena_push(r);
    __arena_push(0);
    i
}

fn mk_pow(base: i32, n: i32) -> i32 {
    let i = __arena_push(4);
    __arena_push(base);
    __arena_push(n);
    __arena_push(0);
    i
}

// Read tag of node at index `idx`.
fn tag_of(idx: i32) -> i32 { __arena_get(idx) }
fn p1_of(idx: i32) -> i32 { __arena_get(idx + 1) }
fn p2_of(idx: i32) -> i32 { __arena_get(idx + 2) }

// ---------------------------------------------------------------
// diff: d/dx(e) by structural recursion. Returns the arena index
// of a fresh expression representing the derivative.
//
//   d/dx(c)         = 0
//   d/dx(x)         = 1
//   d/dx(a + b)     = a' + b'
//   d/dx(a * b)     = a'*b + a*b'
//   d/dx(a^n)       = n * a^(n-1) * a'
// ---------------------------------------------------------------
fn diff(e: i32) -> i32 {
    let t = tag_of(e);
    if t == 0 {
        // Lit: derivative is 0.
        mk_lit(0)
    } else { if t == 1 {
        // Var (x): derivative is 1.
        mk_lit(1)
    } else { if t == 2 {
        // Add(l, r): l' + r'
        let dl = diff(p1_of(e));
        let dr = diff(p2_of(e));
        mk_add(dl, dr)
    } else { if t == 3 {
        // Mul(l, r): l' * r + l * r'
        let l = p1_of(e);
        let r = p2_of(e);
        let dl = diff(l);
        let dr = diff(r);
        let term1 = mk_mul(dl, r);
        let term2 = mk_mul(l, dr);
        mk_add(term1, term2)
    } else { if t == 4 {
        // Pow(base, n): n * base^(n-1) * base'
        let base = p1_of(e);
        let n = p2_of(e);
        let n_minus_one = n - 1;
        let inner_pow = if n_minus_one == 0 { mk_lit(1) }
                        else { mk_pow(base, n_minus_one) };
        let n_lit = mk_lit(n);
        let coef = mk_mul(n_lit, inner_pow);
        let base_d = diff(base);
        mk_mul(coef, base_d)
    } else {
        mk_lit(0)
    }}}}}
}

// ---------------------------------------------------------------
// simplify: a one-pass bottom-up rewriter that collapses trivial
// identities. Returns the arena index of a (possibly new) node.
// ---------------------------------------------------------------
fn simplify(e: i32) -> i32 {
    let t = tag_of(e);
    if t == 2 {
        // Add(l, r): simplify children, then check for 0 + r or l + 0.
        let l = simplify(p1_of(e));
        let r = simplify(p2_of(e));
        if tag_of(l) == 0 {
            if p1_of(l) == 0 { r }
            else { mk_add(l, r) }
        } else { if tag_of(r) == 0 {
            if p1_of(r) == 0 { l }
            else { mk_add(l, r) }
        } else { mk_add(l, r) }}
    } else { if t == 3 {
        // Mul(l, r): handle 0*x, x*0, 1*x, x*1
        let l = simplify(p1_of(e));
        let r = simplify(p2_of(e));
        if tag_of(l) == 0 {
            let lv = p1_of(l);
            if lv == 0 { mk_lit(0) }
            else { if lv == 1 { r } else { mk_mul(l, r) }}
        } else { if tag_of(r) == 0 {
            let rv = p1_of(r);
            if rv == 0 { mk_lit(0) }
            else { if rv == 1 { l } else { mk_mul(l, r) }}
        } else { mk_mul(l, r) }}
    } else { if t == 4 {
        // Pow(base, n): n=0 -> 1, n=1 -> base
        let base = simplify(p1_of(e));
        let n = p2_of(e);
        if n == 0 { mk_lit(1) }
        else { if n == 1 { base } else { mk_pow(base, n) }}
    } else {
        e
    }}}
}

// ---------------------------------------------------------------
// eval_at: substitute x_value for Var, evaluate the expression to i32.
// ---------------------------------------------------------------
fn eval_at(e: i32, x_value: i32) -> i32 {
    let t = tag_of(e);
    if t == 0 {
        // Lit
        p1_of(e)
    } else { if t == 1 {
        // Var
        x_value
    } else { if t == 2 {
        // Add
        eval_at(p1_of(e), x_value) + eval_at(p2_of(e), x_value)
    } else { if t == 3 {
        // Mul
        eval_at(p1_of(e), x_value) * eval_at(p2_of(e), x_value)
    } else { if t == 4 {
        // Pow: integer power via repeated multiplication.
        let base_v = eval_at(p1_of(e), x_value);
        let n = p2_of(e);
        let mut result: i32 = 1;
        let mut i: i32 = 0;
        while i < n {
            result = result * base_v;
            i = i + 1;
        }
        result
    } else { 0 }}}}}
}

// ---------------------------------------------------------------
// main: build x^3 + 2*x, differentiate, simplify, evaluate at x=5.
// d/dx(x^3 + 2*x) = 3*x^2 + 2. At x=5 that's 3*25 + 2 = 77.
// ---------------------------------------------------------------
fn main() -> i32 {
    let x = mk_var();
    let x_cubed = mk_pow(x, 3);
    let two = mk_lit(2);
    let two_x_var = mk_var();   // a fresh var node; same Var meaning
    let two_x = mk_mul(two, two_x_var);
    let f = mk_add(x_cubed, two_x);
    // f = x^3 + 2*x
    // f' = 3*x^2 + 2
    let df = diff(f);
    let df_simplified = simplify(df);
    // Evaluate the simplified derivative at x = 5.
    eval_at(df_simplified, 5)
}
