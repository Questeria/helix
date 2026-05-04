// hbs_sample_helix_ast.hx
//
// HBS dogfood: a Helix-side AST for a tiny calculator language with
// let-bindings and references. Demonstrates the pattern a self-hosted
// compiler would use to represent its OWN AST as Helix values.
//
// Grammar:
//   expr = i32_literal
//        | var_id              // references a let-binding
//        | binop(expr, expr)
//        | let(name_id, expr_value, expr_body)
//
// Name resolution uses arena-stored (name_id, value) pairs as a
// stack: when we enter a let, we __arena_push the binding; when we
// look up a variable, we scan back from the top.
//
// LIMITATION: name_ids are i32 (not strings) since runtime strings
// are not yet a thing. The user picks integer ids.

enum Expr {
    Lit(i32),
    Var(i32),                 // i32 = name id
    Add(Expr, Expr),
    Mul(Expr, Expr),
    Let(i32, Expr, Expr),     // (name_id, value, body)
}

// Stack-based env: pushes (name_id, value) pairs into the arena.
// Returns -1 if not found.
@partial
fn env_lookup(start: i32, top: i32, name: i32) -> i32 {
    if top <= start {
        0 - 1
    } else {
        let pi = top - 2;
        let k = __arena_get(pi);
        if k == name {
            __arena_get(pi + 1)
        } else {
            env_lookup(start, pi, name)
        }
    }
}

@partial
fn eval_expr(e: Expr, env_start: i32) -> i32 {
    match e {
        Expr::Lit(x) => x,
        Expr::Var(id) => env_lookup(env_start, __arena_len(), id),
        Expr::Add(l, r) => eval_expr(l, env_start) + eval_expr(r, env_start),
        Expr::Mul(l, r) => eval_expr(l, env_start) * eval_expr(r, env_start),
        Expr::Let(name, value_expr, body_expr) => {
            let v = eval_expr(value_expr, env_start);
            __arena_push(name);
            __arena_push(v);
            eval_expr(body_expr, env_start)
            // We don't pop the binding because the arena is monotonic;
            // for nested lets that's wrong, but for top-level eval the
            // env_start gets reset.
        }
    }
}

fn main() -> i32 {
    // Build:  let x = 3 in (x * x + x * 4) — should eval to 9 + 12 = 21
    // Actually, let's compute (let x = 6 in x * 7) = 42.
    let env_start = __arena_len();
    let three = Expr::Lit(7);
    let var_x = Expr::Var(1);          // name id 1 = "x"
    let prod = Expr::Mul(var_x, three);
    let six = Expr::Lit(6);
    let let_x_eq_6_in_x_times_7 = Expr::Let(1, six, prod);
    eval_expr(let_x_eq_6_in_x_times_7, env_start)
}
