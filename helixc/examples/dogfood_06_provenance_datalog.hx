// dogfood_06_provenance_datalog.hx — Stage 36 Increment 4 dogfood.
//
// Datalog-shaped propositional reasoning over provenance-typed truth
// values. This is the first dogfood program that demonstrates the
// Tier 3 #10 strategic primitives shipped in Stage 36 Increments 1-4
// (commits 9e9b421, c142ce0, fb36c51, de0b60b).
//
// What it computes:
//
//   Facts (with source tags):
//     parent(alice, bob)  = TRUE  (source 1: birth registry)
//     parent(bob,  carol) = TRUE  (source 2: census)
//
//   Rule:
//     grandparent(X, Z) <- parent(X, Y) AND parent(Y, Z)
//
//   Query:
//     grandparent(alice, carol) ?
//
//   Plus a tautology check on each variable to verify the boolean
//   algebra is sound: (P OR NOT P) = TRUE for every truth value.
//
// Why this matters: in JAX/Mojo/Triton, this kind of relational
// reasoning would be either (a) impossible (no type for it) or
// (b) a separate tensor-encoding pipeline. In Helix, each value
// carries its provenance tag in the type, every combinator
// enforces the Logic<T> boundary at trap-24100, and the same
// language that runs the rule can later differentiate through it
// (the planned Increment 6).
//
// Exit code: 42 if the grandparent rule fires AND the tautology
// holds for both P=0 and P=1. Anything else means a soundness bug
// in the provenance algebra.

fn main() -> i32 {
    // Facts — wrap raw 0/1 truth values with their source tags.
    let p_alice_bob: Logic<i32>  = prove(1, 1);   // src 1
    let p_bob_carol: Logic<i32>  = prove(1, 2);   // src 2

    // Rule: grandparent(alice, carol) <- parent(alice, bob) AND parent(bob, carol)
    let grandparent_holds: Logic<i32> = and_logic(p_alice_bob, p_bob_carol);

    // Derive a conclusion that carries provenance from both parents
    // (Phase-0 single-tag: keeps the first parent's tag; the lattice
    // upgrade in Increment 5 will track both).
    let conclusion: Logic<i32> = derive(p_alice_bob, p_bob_carol);

    // Verify rule fires and conclusion matches: both should be 1.
    let rule_fired: Logic<i32> = eq_logic(grandparent_holds, conclusion);

    // Tautology check: P OR NOT P is always TRUE.
    let p_true: Logic<i32>  = prove(1, 100);
    let p_false: Logic<i32> = prove(0, 200);
    let taut_p_true: Logic<i32>  = or_logic(p_true,  not_logic(p_true));
    let taut_p_false: Logic<i32> = or_logic(p_false, not_logic(p_false));
    let tautology_holds: Logic<i32> = and_logic(taut_p_true, taut_p_false);

    // Final witness: rule fires AND tautology holds.
    let witness: Logic<i32> = and_logic(rule_fired, tautology_holds);

    // Exit 42 on success, 0 on failure.
    if unwrap_logic(witness) == 1 {
        42
    } else {
        0
    }
}
