// dogfood_09_knowledge_graph.hx - Stage 36 Increment 10 dogfood.
//
// Knowledge-graph reasoner: 3 facts + 2 rules + provenance recovery.
// First dogfood that uses the Inc 9 audit-clean primitives in a small
// real-shaped chained-rule scenario. Demonstrates derive() now being
// observably side-effectful (B2 fix) and register_derivation's
// 1-based handles (A2 fix) returning sane provenance trails.
//
// Knowledge base (with source tags):
//   1: parent(alice, bob)  = TRUE  (birth registry)
//   2: parent(bob, carol)  = TRUE  (birth registry)
//   3: parent(bob, dave)   = TRUE  (birth registry)
//
// Rules (chained):
//   R1: grandparent(alice, carol) <- parent(alice, bob) AND parent(bob, carol)
//   R2: grandparent(alice, dave)  <- parent(alice, bob) AND parent(bob, dave)
//
// Both rules fire (all antecedents TRUE), so the engine produces
// two derived facts with provenance trails (10 <- {1, 2} and
// 11 <- {1, 3}). The provenance handles are stored and queried
// via parent_left_at / parent_right_at to recover the evidence.
//
// Exit code 42 iff:
//   - both grandparent atoms compute to TRUE (count = 2)
//   - count * 21 = 42

fn main() -> i32 {
    // ---------- Facts ----------
    let p_alice_bob: Logic<i32>  = prove(1, 1);  // parent(alice, bob)
    let p_bob_carol: Logic<i32>  = prove(1, 2);  // parent(bob, carol)
    let p_bob_dave: Logic<i32>   = prove(1, 3);  // parent(bob, dave)

    // ---------- Rule R1: grandparent(alice, carol) ----------
    // Compute the truth value via and_logic.
    let gp_ac: Logic<i32> = and_logic(p_alice_bob, p_bob_carol);

    // Register the provenance: derived fact 10 has parents 1, 2.
    // (Inc 9 A2 fix: the handle is 1-based and the two pushes are
    // atomic via ARENA_PUSH_PAIR.)
    let h_ac: i32 = register_derivation(1, 2);

    // ---------- Rule R2: grandparent(alice, dave) ----------
    let gp_ad: Logic<i32> = and_logic(p_alice_bob, p_bob_dave);
    let h_ad: i32 = register_derivation(1, 3);

    // ---------- Count the firing rules ----------
    let count: i32 = unwrap_logic(gp_ac) + unwrap_logic(gp_ad);

    // ---------- Recover the evidence trail (verify provenance) ----------
    // grandparent(alice, carol) should trace back to source IDs (1, 2)
    let lp_ac: i32 = parent_left_at(h_ac);
    let rp_ac: i32 = parent_right_at(h_ac);
    // grandparent(alice, dave) should trace back to (1, 3)
    let lp_ad: i32 = parent_left_at(h_ad);
    let rp_ad: i32 = parent_right_at(h_ad);

    // Evidence check: all four lookups must succeed and match.
    let ev_ok: i32 = if lp_ac == 1 {
        if rp_ac == 2 {
            if lp_ad == 1 {
                if rp_ad == 3 { 1 } else { 0 }
            } else { 0 }
        } else { 0 }
    } else { 0 };

    // ---------- Final witness ----------
    // count = 2, ev_ok = 1, so count * 21 * ev_ok = 42.
    count * 21 * ev_ok
}
