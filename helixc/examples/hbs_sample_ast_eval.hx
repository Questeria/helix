// hbs_sample_ast_eval.hx
//
// HBS dogfood: real recursive-enum AST evaluator using the new
// arena-indirected enum support.  Mirrors the shape a self-hosted
// compiler's expression AST would have.
//
// Expr is a recursive sum type:
//   - Const(i32)              — integer literal
//   - Add(Expr, Expr)         — binary plus
//   - Mul(Expr, Expr)         — binary times
//   - Neg(Expr)               — unary negation
//
// Each constructor allocates [tag, payload0, payload1, ...] in the
// global arena and returns the start index. eval() walks the tree
// recursively, dispatching on the tag.

enum Expr {
    Const(i32),
    Add(Expr, Expr),
    Mul(Expr, Expr),
    Neg(Expr),
}

@partial
fn eval(e: Expr) -> i32 {
    match e {
        Expr::Const(x) => x,
        Expr::Add(l, r) => eval(l) + eval(r),
        Expr::Mul(l, r) => eval(l) * eval(r),
        Expr::Neg(x) => 0 - eval(x),
    }
}

fn main() -> i32 {
    // Compute (3 + 4) * 6 = 42
    let three = Expr::Const(3);
    let four = Expr::Const(4);
    let sum = Expr::Add(three, four);
    let six = Expr::Const(6);
    let prod = Expr::Mul(sum, six);
    eval(prod)
}
