// dogfood_09_knowledge_graph.hx - Stage 36 Increment 10 dogfood.
//
// Knowledge-graph reasoner: 3 facts + 2 rules + provenance recovery.
// First dogfood that uses the Inc 9 audit-clean primitives in a small
// real-shaped chained-rule scenario. Exercises register_derivation's
// 1-based handles (Inc 9 A2 fix), the ARENA_PUSH_PAIR atomicity
// (Inc 9 B2 fix), and parent_*_at evidence-trail recovery.
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
// two derived facts with provenance trails. The provenance handles
// are 1-based and each register_derivation pushes 2 arena entries
// via ARENA_PUSH_PAIR (Inc 9 A2 fix), so:
//   h_ac = 1 (refers to arena[0]; arena[0]=1, arena[1]=2)
//   h_ad = 3 (refers to arena[2]; arena[2]=1, arena[3]=3)
//
// Exit code 42 iff ALL of the following hold independently (Stage 36
// Inc 11 post-Inc-10 audit M1 strengthening — the original
// count * 21 * ev_ok witness collapsed firings into a sum and would
// still hit 42 under an AND-lowered-as-OR regression OR a 0-based
// handle regression):
//   - gp_ac unwraps to exactly 1 (R1 fired)
//   - gp_ad unwraps to exactly 1 (R2 fired)
//   - h_ac == 1 (1-based handle for slot 0)
//   - h_ad == 2 (1-based handle for slot 1)
//   - all four parent_*_at lookups return the expected source IDs
// Any single regression collapses the product to 0; exit 42 requires
// every invariant to pass.

fn main() -> i32 {
    // ---------- Facts ----------
    let p_alice_bob: Logic<i32>  = prove(1, 1);  // parent(alice, bob)
    let p_bob_carol: Logic<i32>  = prove(1, 2);  // parent(bob, carol)
    let p_bob_dave: Logic<i32>   = prove(1, 3);  // parent(bob, dave)

    // ---------- Rule R1: grandparent(alice, carol) ----------
    let gp_ac: Logic<i32> = and_logic(p_alice_bob, p_bob_carol);
    let h_ac: i32 = register_derivation(1, 2);

    // ---------- Rule R2: grandparent(alice, dave) ----------
    let gp_ad: Logic<i32> = and_logic(p_alice_bob, p_bob_dave);
    let h_ad: i32 = register_derivation(1, 3);

    // ---------- Per-rule truth-value witnesses ----------
    // Each one must be exactly 1; an OR-instead-of-AND regression
    // would still pass `> 0` checks but fails exact equality.
    let gp_ac_ok: i32 = if unwrap_logic(gp_ac) == 1 { 1 } else { 0 };
    let gp_ad_ok: i32 = if unwrap_logic(gp_ad) == 1 { 1 } else { 0 };

    // ---------- Per-handle 1-based-encoding witnesses ----------
    // Each register_derivation pushes 2 arena entries via
    // ARENA_PUSH_PAIR, so handles step by 2:
    //   h_ac = 1 (arena slot 0)
    //   h_ad = 3 (arena slot 2)
    // A 0-based regression would give h_ac=0, h_ad=2 — both fail.
    let h_ac_ok: i32 = if h_ac == 1 { 1 } else { 0 };
    let h_ad_ok: i32 = if h_ad == 3 { 1 } else { 0 };

    // ---------- Per-lookup evidence-trail witnesses ----------
    let lp_ac: i32 = parent_left_at(h_ac);
    let rp_ac: i32 = parent_right_at(h_ac);
    let lp_ad: i32 = parent_left_at(h_ad);
    let rp_ad: i32 = parent_right_at(h_ad);
    let lp_ac_ok: i32 = if lp_ac == 1 { 1 } else { 0 };
    let rp_ac_ok: i32 = if rp_ac == 2 { 1 } else { 0 };
    let lp_ad_ok: i32 = if lp_ad == 1 { 1 } else { 0 };
    let rp_ad_ok: i32 = if rp_ad == 3 { 1 } else { 0 };

    // ---------- Final witness ----------
    // Product of 8 independent binary witnesses; exit 42 requires
    // every one to be 1. Any single failure → 0.
    let all_ok: i32 = gp_ac_ok * gp_ad_ok
        * h_ac_ok * h_ad_ok
        * lp_ac_ok * rp_ac_ok * lp_ad_ok * rp_ad_ok;
    all_ok * 42
}
