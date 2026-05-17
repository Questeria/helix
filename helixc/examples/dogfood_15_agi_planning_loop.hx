// dogfood_15_agi_planning_loop.hx — Stage 42 Increment 1 dogfood.
//
// AGI planning-loop scenario exercising all 5 semantic-type
// families (memory + spatial + temporal + modal + causal) in a
// single coherent program. Demonstrates that the quintet
// completed at Stage 41 composes orthogonally end-to-end:
// values flow through 4-deep wrapper stacks with zero runtime
// overhead (Phase-0 identity-lowered) and the typechecker
// catches category mistakes at every layer.
//
// Scenario (robot perception + planning loop):
//   1. observe — sensor reading enters as
//      Known<Present<WorldFrame<i32>>> (directly observed, now,
//      in world frame).
//   2. mark_as_cause — the observation is identified as a
//      causal input → Known<Present<WorldFrame<Cause<i32>>>>.
//   3. plan — planner forecasts the downstream effect of the
//      cause; epistemic status drops from Known to Believed
//      (inferred) and temporal kind drops from Present to
//      Future → Believed<Future<WorldFrame<Cause<i32>>>>.
//      (We keep the WorldFrame and Cause layers; only the
//      modal+temporal layers shift.)
//   4. unwrap_to_witness — strip all 4 wrappers in reverse
//      order; the inner i32 must survive every transition.
//
// Real-world parallel: every AGI agent's perception-plan-act
// loop carries values through exactly this kind of multi-axis
// wrapper transition. Stage 42 proves Helix can express the
// loop at the type level without representational drift.
//
// Exit code 42 iff THREE independent observations cycle through
// the full 4-deep wrapper stack correctly. The witness is
// collapse-resistant: any wrong-kind transition or any identity-
// lowering bug at any of the 4 wrapper layers collapses the
// product to 0.

@pure
fn agi_perceive_plan_cycle(raw: i32) -> i32 {
    // Step 1: observe — directly observed, current, in world
    // frame.
    let obs: Known<Present<WorldFrame<i32>>> =
        into_known(into_present(into_world(raw)));

    // Step 2: identify the observation as a causal input.
    // (Unwrap WorldFrame, attach Cause, re-wrap WorldFrame —
    // the causal kind lives "below" the spatial frame in the
    // wrapper stack since causality applies to the value
    // regardless of which frame it's observed in.)
    let wf: WorldFrame<i32> = from_present(from_known(obs));
    let c_raw: i32 = from_world(wf);
    let cause_in_world: Known<Present<WorldFrame<Cause<i32>>>> =
        into_known(into_present(into_world(into_cause(c_raw))));

    // Step 3: plan — forecast the future state if this cause
    // propagates. Drop Known to Believed (inferred), Present
    // to Future (predicted), Cause to Effect (propagated).
    // The WorldFrame stays (spatial axis is invariant across
    // the planning step).
    let wf2: WorldFrame<Cause<i32>> =
        from_present(from_known(cause_in_world));
    let c: Cause<i32> = from_world(wf2);
    let eff_val: Effect<i32> = propagate(c);
    let plan: Believed<Future<WorldFrame<Effect<i32>>>> =
        into_believed(into_future(into_world(eff_val)));

    // Step 4: unwrap to the bare i32 witness through all 4
    // layers (modal → temporal → spatial → causal).
    let wf3: WorldFrame<Effect<i32>> =
        from_future(from_believed(plan));
    let eff: Effect<i32> = from_world(wf3);
    from_effect(eff)
}

fn main() -> i32 {
    let o1: i32 = agi_perceive_plan_cycle(10);
    let o2: i32 = agi_perceive_plan_cycle(14);
    let o3: i32 = agi_perceive_plan_cycle(18);

    let o1_ok: i32 = if o1 == 10 { 1 } else { 0 };
    let o2_ok: i32 = if o2 == 14 { 1 } else { 0 };
    let o3_ok: i32 = if o3 == 18 { 1 } else { 0 };

    let all_ok: i32 = o1_ok * o2_ok * o3_ok;
    all_ok * (o1 + o2 + o3)
}
