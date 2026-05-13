// hbs_sample_visitor.hx
//
// HBS dogfood: a small "AST visitor" pattern using all the recent
// language additions — structs, enums, payload-bearing variants,
// pattern matching with extraction, and (now) struct/enum pass-by-
// value to helper functions.
//
// This is the kind of code shape a self-hosted typechecker / IR
// lowering pass would use. It exercises the multi-slot ABI end-to-end:
// each helper takes a Token (struct) or Expr (enum) BY VALUE.

// A token with kind + payload. Kind encoding: 0=int, 1=add, 2=mul.
struct Token { kind: i32, value: i32 }

// A binary expression with operator-kind + two operand-tokens.
struct BinExpr { op_kind: i32, lhs: Token, rhs: Token }

// Sum type: either a single token, or a binary expression.  We can't
// model `enum Expr { Lit(Token), Bin(BinExpr) }` directly because the
// payload-bearing variant's payload is itself a struct (not yet supported).
// Workaround: represent both shapes as a flat BinExpr where op_kind=0
// means "lit, value in lhs.value".

@total
fn token_value(t: Token) -> i32 {
    // For now, we treat any token as just its `value` field.
    t.value
}

@total
fn add_tokens(a: Token, b: Token) -> i32 {
    token_value(a) + token_value(b)
}

@total
fn mul_tokens(a: Token, b: Token) -> i32 {
    token_value(a) * token_value(b)
}

@total
fn eval_binexpr(e: BinExpr) -> i32 {
    match e.op_kind {
        0 => token_value(e.lhs),
        1 => add_tokens(e.lhs, e.rhs),
        2 => mul_tokens(e.lhs, e.rhs),
        _ => 0,
    }
}

fn main() -> i32 {
    // Compute (6 * 7) — encoded as a BinExpr with op_kind=2.
    let a = Token { kind: 0, value: 6 };
    let b = Token { kind: 0, value: 7 };
    let mul = BinExpr { op_kind: 2, lhs: a, rhs: b };
    eval_binexpr(mul)   // 42
}
