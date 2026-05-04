// hbs_sample_constant_fold.hx
//
// HBS dogfood: a real compiler pass written in Helix. Constant
// folding over the recursive Expr AST. This is the SAME shape that
// const-fold takes in the Python helixc/ir/passes/const_fold.py —
// proving Helix can host its own compiler passes.
//
// simplify(e):
//   Lit(n)             → Lit(n)
//   Add(Lit(a), Lit(b))→ Lit(a+b)
//   Mul(Lit(a), Lit(b))→ Lit(a*b)
//   Add(l, r)          → Add(simplify(l), simplify(r)) [recursing]
//   Mul(l, r)          → Mul(simplify(l), simplify(r))
//   Neg(Lit(n))        → Lit(-n)
//   Neg(x)             → Neg(simplify(x))
//
// The pass is functional: it returns a fresh Expr (arena index) for
// any folded sub-tree, leaving the original tree intact.

enum Expr {
    Lit(i32),
    Add(Expr, Expr),
    Mul(Expr, Expr),
    Neg(Expr),
}

// Helper: try to fold (Add a b) when both are Lit, else return Add.
@partial
fn fold_add(ls: Expr, rs: Expr) -> Expr {
    match ls {
        Expr::Lit(a) => {
            let r2 = rs;
            match r2 {
                Expr::Lit(b) => Expr::Lit(a + b),
                Expr::Add(_, _) => Expr::Add(ls, rs),
                Expr::Mul(_, _) => Expr::Add(ls, rs),
                Expr::Neg(_) => Expr::Add(ls, rs),
            }
        },
        Expr::Add(_, _) => Expr::Add(ls, rs),
        Expr::Mul(_, _) => Expr::Add(ls, rs),
        Expr::Neg(_) => Expr::Add(ls, rs),
    }
}

@partial
fn fold_mul(ls: Expr, rs: Expr) -> Expr {
    match ls {
        Expr::Lit(a) => {
            let r2 = rs;
            match r2 {
                Expr::Lit(b) => Expr::Lit(a * b),
                Expr::Add(_, _) => Expr::Mul(ls, rs),
                Expr::Mul(_, _) => Expr::Mul(ls, rs),
                Expr::Neg(_) => Expr::Mul(ls, rs),
            }
        },
        Expr::Add(_, _) => Expr::Mul(ls, rs),
        Expr::Mul(_, _) => Expr::Mul(ls, rs),
        Expr::Neg(_) => Expr::Mul(ls, rs),
    }
}

@partial
fn simplify(e: Expr) -> Expr {
    match e {
        Expr::Lit(n) => Expr::Lit(n),
        Expr::Add(l, r) => fold_add(simplify(l), simplify(r)),
        Expr::Mul(l, r) => fold_mul(simplify(l), simplify(r)),
        Expr::Neg(x) => {
            let xs = simplify(x);
            match xs {
                Expr::Lit(n) => Expr::Lit(0 - n),
                Expr::Add(_, _) => Expr::Neg(xs),
                Expr::Mul(_, _) => Expr::Neg(xs),
                Expr::Neg(_) => Expr::Neg(xs),
            }
        },
    }
}

@partial
fn eval(e: Expr) -> i32 {
    match e {
        Expr::Lit(x) => x,
        Expr::Add(l, r) => eval(l) + eval(r),
        Expr::Mul(l, r) => eval(l) * eval(r),
        Expr::Neg(x) => 0 - eval(x),
    }
}

// Detector: is this Expr a Lit? Returns 1 if yes, 0 otherwise.
// Used to verify that simplify actually collapsed a constant sub-tree.
@total
fn is_lit(e: Expr) -> i32 {
    match e {
        Expr::Lit(_) => 1,
        _ => 0,
    }
}

fn main() -> i32 {
    // Build  ((3 + 4) * 6) — a fully-constant expression.
    let three = Expr::Lit(3);
    let four = Expr::Lit(4);
    let sum = Expr::Add(three, four);          // not yet a Lit
    let six = Expr::Lit(6);
    let prod = Expr::Mul(sum, six);            // not yet a Lit

    // simplify should fold the whole thing into Lit(42).
    let folded = simplify(prod);

    // Two checks:
    //   1. eval should still produce 42 (semantics preserved)
    //   2. folded should be a Lit (not Mul)
    let val = eval(folded);
    let was_folded = is_lit(folded);

    // Compose: 42 if both checks pass; otherwise some lower number.
    if was_folded == 1 {
        val      // 42 if folding worked
    } else {
        0        // simplify didn't reduce to a literal
    }
}
