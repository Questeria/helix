// dogfood_12_temporal_lifecycle.hx — Stage 39 Increment 3 dogfood.
//
// Temporal-kind lifecycle reasoner: an observation flows through every
// meaningful temporal kind via the Stage 39 transition primitives,
// returning to a raw value intact. First dogfood that exercises the
// Stage 39 Inc 1 + Inc 2 temporal-typing primitives in an AGI-shaped
// scenario.
//
// What this dogfood demonstrates:
//   1. An observation enters via into_present (the AGI sees a fact
//      happen NOW — current sensor reading).
//   2. forecast — project current state into the future (the planner
//      asks "what will this value be next tick?"; Phase-0 the wrapper
//      just retags, real predictive math is Phase-1+).
//   3. actualize — the predicted future arrives and is now the present
//      (the planner's projection becomes a fresh observation when
//      time advances).
//   4. to_past — present recedes into past as time advances (the
//      observation is now history, immutable record).
//   5. from_past unwraps the result.
//
// Real-world parallels:
//   - Robot perception: observed_now -> predicted_next_tick ->
//     observed_next_tick -> historical_log.
//   - Knowledge agents: belief_now -> belief_after_inference ->
//     confirmed_belief -> belief_history.
//   - Planning loops: state_now -> projected_state -> realized_state
//     -> consumed_state.
//
// Exit code 42 iff THREE independent observations cycle through all
// four temporal kinds correctly. Witness is collapse-resistant: each
// observation must round-trip exactly, AND the chain must be type-
// correct end-to-end (any wrong-kind transition call would have
// failed at typecheck before this binary was even produced).

@pure
fn cycle_through_time(raw: i32) -> i32 {
    // Step 1: observation enters as Present (current sensor reading).
    let now: Present<i32> = into_present(raw);
    // Step 2: planner projects into Future (the prediction).
    let pred: Future<i32> = forecast(now);
    // Step 3: time advances, prediction becomes Present (the future
    // has arrived; the projected value is now an observation again).
    let realized: Present<i32> = actualize(pred);
    // Step 4: time advances again, Present recedes into Past (the
    // observation joins history; immutable record).
    let was: Past<i32> = to_past(realized);
    // Step 5: unwrap back to raw i32 for the witness.
    from_past(was)
}

@pure
fn recall_from_history(raw: i32) -> i32 {
    // Independent path exercising recall_past — bring a historical
    // observation back into current focus for reasoning (the AGI
    // pulls up a past observation to deliberate about it now).
    let hist: Past<i32> = into_past(raw);
    let focused: Present<i32> = recall_past(hist);
    from_present(focused)
}

fn main() -> i32 {
    // Three independent observations flow through the full lifecycle.
    let obs1: i32 = cycle_through_time(10);
    let obs2: i32 = cycle_through_time(14);
    let obs3: i32 = cycle_through_time(18);

    // Per-observation binary witnesses for the full Present -> Future
    // -> Present -> Past chain.
    let obs1_ok: i32 = if obs1 == 10 { 1 } else { 0 };
    let obs2_ok: i32 = if obs2 == 14 { 1 } else { 0 };
    let obs3_ok: i32 = if obs3 == 18 { 1 } else { 0 };

    // Independent recall_past witness on a fourth value — the
    // historical-recall direction is exercised separately so a chain-
    // path regression and a recall-path regression collapse the
    // witness independently.
    let rec: i32 = recall_from_history(7);
    let rec_ok: i32 = if rec == 7 { 1 } else { 0 };

    // Sanity-check the Eternal kind via plain intro/elim — Eternal
    // values are timeless (mathematical facts, physical constants)
    // and have no transitions in Stage 39 Inc 2.
    let eternal_truth: i32 = from_eternal(into_eternal(1));
    let eternal_ok: i32 = if eternal_truth == 1 { 1 } else { 0 };

    // Product of 5 binary witnesses; any single regression collapses
    // to 0 -> final exit code 0 not 42.
    let all_ok: i32 = obs1_ok * obs2_ok * obs3_ok * rec_ok * eternal_ok;

    // Sum of recalled observations: 10 + 14 + 18 = 42.
    all_ok * (obs1 + obs2 + obs3)
}
