// hbs_sample_tree_eval.hx
//
// HBS dogfood: tiny "AST" evaluator. Demonstrates enum constants and
// match-dispatch over node kinds. The expression -((1 + 2) * 7),
// negated again to keep the exit code positive, is evaluated inline
// in main: each node's kind and literal value is a scalar local, and
// each node's value is computed by a `match` over its Op kind.
//
// Node kinds:
//   Op::Const   the node's value is a literal i32
//   Op::Add     sum of two child node values
//   Op::Mul     product of two child node values
//   Op::Neg     negation of one child node value
//
// An earlier draft passed the tree as an [i32; 16] array to accessor
// functions (node_kind / node_lhs / node_rhs / node_val). Indexing an
// array-typed function parameter is not yet supported by the backend,
// so the evaluator is inlined with scalar locals instead. When that
// feature lands, this example can be rewritten to showcase it.

enum Op { Const, Add, Mul, Neg }

fn main() -> i32 {
    // Encode the expression  -((1 + 2) * 7)  as four nodes:
    //   id 0: Const 1
    //   id 1: Const 2
    //   id 2: Add(0, 1)         = 3
    //   id 3: Const 7
    //   The full expression evaluates to -(3 * 7) = -21
    //   We negate again with another Neg node id 4 → +21 to keep the
    //   exit code positive.
    //   id 4: Mul(2, 3)         = 21
    //   Result: 21 (positive — fits the i32 exit code without two's-
    //   complement wrap).
    let kind0: i32 = 0;     // Op::Const
    let val0: i32 = 1;
    let kind1: i32 = 0;
    let val1: i32 = 2;
    let kind2: i32 = 1;     // Op::Add
    let kind3: i32 = 0;
    let val3: i32 = 7;
    let kind4: i32 = 2;     // Op::Mul

    // Inline evaluator: build values for each node id in dependency order.
    let v0 = match kind0 {
        Op::Const => val0,
        _ => 0,
    };
    let v1 = match kind1 {
        Op::Const => val1,
        _ => 0,
    };
    let v2 = match kind2 {
        Op::Add => v0 + v1,
        Op::Mul => v0 * v1,
        Op::Neg => 0 - v0,
        _ => 0,
    };
    let v3 = match kind3 {
        Op::Const => val3,
        _ => 0,
    };
    let v4 = match kind4 {
        Op::Add => v2 + v3,
        Op::Mul => v2 * v3,
        Op::Neg => 0 - v2,
        _ => 0,
    };
    v4
}
