// hbs_sample_tree_eval.hx
//
// HBS dogfood: tiny "AST" evaluator. The tree is a flat array indexed
// by node-id; each node is encoded as 4 i32 slots (kind, lhs_id,
// rhs_id, value).  This sidesteps the lack of payload-bearing enums
// and recursive struct types.
//
// Node kinds:
//   Op::Const   value  = the literal i32, lhs/rhs unused
//   Op::Add     lhs    = id of left operand, rhs = id of right
//   Op::Mul     lhs/rhs same
//   Op::Neg     lhs    = id of operand, rhs unused
//
// The eval routine looks up node by id, dispatches on kind, recurses.
//
// Demonstrates: enum constants, match dispatch, recursive function +
// totality checker (since each recursive call passes a smaller node id),
// struct field access through chained `tree.value` style is still TBD —
// here we use mutable globals via a fixed-size i32 array indexed by id.

enum Op { Const, Add, Mul, Neg }

@total
fn node_kind(arr: [i32; 16], id: i32) -> i32 { arr[4 * id] }

@total
fn node_lhs(arr: [i32; 16], id: i32) -> i32 { arr[4 * id + 1] }

@total
fn node_rhs(arr: [i32; 16], id: i32) -> i32 { arr[4 * id + 2] }

@total
fn node_val(arr: [i32; 16], id: i32) -> i32 { arr[4 * id + 3] }

// Note: we intentionally don't call `eval` here. The evaluator is
// inlined into main below to keep totality + codegen straightforward
// for now. A future tick can add struct-of-array passing once we have
// pass-by-reference for arrays.

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
