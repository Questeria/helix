// dogfood_13_modal_lifecycle.hx — Stage 40 Increment 3 dogfood.
//
// Modal/epistemic lifecycle reasoner: a proposition flows through
// the AGI decision loop's natural modal upgrades — from goal to
// known (agent achieves a goal), from believed to known (a
// hypothesis is observationally confirmed). First dogfood that
// exercises the Stage 40 Inc 1 + Inc 2 modal-typing primitives in
// an AGI-shaped scenario.
//
// What this dogfood demonstrates:
//   1. A goal enters via into_goal (the planner sets a target —
//      "make the robot reach coordinate X").
//   2. act_on — the agent executes; what was desired is now
//      observed-true. Goal becomes Known.
//   3. A hypothesis enters via into_believed (the inference
//      engine produces a candidate fact — "the door is open").
//   4. confirm — direct observation upgrades the belief to a
//      known fact (Believed -> Known).
//   5. from_known unwraps the result.
//   6. Uncertain<T> intro/elim sanity check (no transition in
//      Stage 40 Inc 2 — Uncertain values gate info-gathering
//      actions; promotion to Believed/Known requires more spec).
//
// Real-world parallels:
//   - Robot planning: Goal(reach_pose) -> act_on -> Known(at_pose).
//   - Knowledge agents: Believed(door_open) -> confirm via
//     observation -> Known(door_open).
//   - Information-seeking: Uncertain(weather) gates the "check
//     weather sensor" action (no auto-upgrade without observation).
//
// Exit code 42 iff THREE independent propositions all complete
// their modal upgrades correctly. Witness is collapse-resistant:
// each modal upgrade must round-trip exactly, AND the chain must
// be type-correct end-to-end (any wrong-kind transition call
// would have failed at typecheck before this binary was even
// produced).

@pure
fn achieve_goal(raw: i32) -> i32 {
    // Step 1: the planner sets a goal.
    let want: Goal<i32> = into_goal(raw);
    // Step 2: agent executes; goal becomes known fact.
    let done: Known<i32> = act_on(want);
    // Step 3: unwrap for the witness.
    from_known(done)
}

@pure
fn observe_belief(raw: i32) -> i32 {
    // Step 1: inference engine produces a hypothesis.
    let hyp: Believed<i32> = into_believed(raw);
    // Step 2: direct observation confirms it.
    let fact: Known<i32> = confirm(hyp);
    // Step 3: unwrap.
    from_known(fact)
}

@pure
fn gate_uncertain(raw: i32) -> i32 {
    // Uncertain values don't transition in Stage 40 Inc 2 — they
    // gate info-gathering actions. The dogfood just exercises
    // intro/elim sanity (no auto-upgrade allowed).
    let q: Uncertain<i32> = into_uncertain(raw);
    from_uncertain(q)
}

fn main() -> i32 {
    // Goal-achievement loop on 3 independent propositions.
    let g1: i32 = achieve_goal(10);
    let g2: i32 = achieve_goal(14);
    let g3: i32 = achieve_goal(18);

    // Belief-confirmation loop on the same 3 (different path).
    let b1: i32 = observe_belief(10);
    let b2: i32 = observe_belief(14);
    let b3: i32 = observe_belief(18);

    // Uncertain sanity check on a fourth value.
    let u: i32 = gate_uncertain(7);

    // Cross-stage composition probe: a Known<Past<i32>> = "I
    // directly observed this past fact" — Stage 40 modal kinds
    // compose with Stage 39 temporal kinds at the type level.
    let kp: Known<Past<i32>> = into_known(into_past(1));
    let cross: i32 = from_past(from_known(kp));

    // Per-proposition binary witnesses.
    let g1_ok: i32 = if g1 == 10 { 1 } else { 0 };
    let g2_ok: i32 = if g2 == 14 { 1 } else { 0 };
    let g3_ok: i32 = if g3 == 18 { 1 } else { 0 };
    let b1_ok: i32 = if b1 == 10 { 1 } else { 0 };
    let b2_ok: i32 = if b2 == 14 { 1 } else { 0 };
    let b3_ok: i32 = if b3 == 18 { 1 } else { 0 };
    let u_ok:  i32 = if u  == 7  { 1 } else { 0 };
    let cross_ok: i32 = if cross == 1 { 1 } else { 0 };

    // Product of 8 binary witnesses; any single regression
    // collapses to 0 -> final exit code 0 not 42.
    let all_ok: i32 =
        g1_ok * g2_ok * g3_ok *
        b1_ok * b2_ok * b3_ok *
        u_ok * cross_ok;

    // Sum of the achieved goals (g1+g2+g3 = 42).
    all_ok * (g1 + g2 + g3)
}
