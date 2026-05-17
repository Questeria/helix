// dogfood_14_causal_lifecycle.hx — Stage 41 Increment 3 dogfood.
//
// Causal/intent lifecycle reasoner: a proposition flows through
// the AGI causal-decision loop's natural transitions — Cause to
// Effect (the cause is enacted), Effect to Joint (other causes
// also contributed), Joint to Independent (experiment confirms
// no upstream matters). First dogfood that exercises the Stage 41
// Inc 1 + Inc 2 causal-typing primitives in an AGI-shaped
// scenario.
//
// What this dogfood demonstrates:
//   1. A proposition enters as Cause (the planner identifies an
//      upstream input).
//   2. propagate — the cause is enacted; what was an upstream
//      input is now a downstream observation (Cause -> Effect).
//   3. aggregate — other causes also contributed; the single-
//      source effect becomes a multi-source joint observation
//      (Effect -> Joint).
//   4. isolate — experiment confirms no upstream actually
//      mattered; the multi-cause observation collapses to causal
//      independence (Joint -> Independent).
//   5. from_independent unwraps the result.
//   6. Plus a cross-stack composition probe: `Known<Cause<i32>>`
//      = "I directly observed that this was a cause" — the
//      5-stack quintet (memory / spatial / temporal / modal /
//      causal) composes orthogonally at the type level.
//
// Real-world parallels:
//   - Robot planning: Cause(motor_torque) -> Effect(joint_angle)
//     -> Joint(end_effector_position depends on every joint) ->
//     Independent(after experiment, only this joint matters).
//   - Knowledge agents: Cause(query_intent) -> Effect(retrieved
//     answer) -> Joint(answer also depends on context) ->
//     Independent(context-free answer for this query class).
//
// Exit code 42 iff THREE independent propositions complete their
// causal lifecycles correctly AND the cross-stack composition
// probe round-trips.

@pure
fn causal_lifecycle(raw: i32) -> i32 {
    let c: Cause<i32> = into_cause(raw);
    let e: Effect<i32> = propagate(c);
    let j: Joint<i32> = aggregate(e);
    let i: Independent<i32> = isolate(j);
    from_independent(i)
}

@pure
fn cross_stack_known_cause(raw: i32) -> i32 {
    // `Known<Cause<i32>>` — "I directly observed that this was
    // a cause." The 5-stack quintet composes orthogonally.
    let c: Cause<i32> = into_cause(raw);
    let kc: Known<Cause<i32>> = into_known(c);
    from_cause(from_known(kc))
}

fn main() -> i32 {
    let p1: i32 = causal_lifecycle(10);
    let p2: i32 = causal_lifecycle(14);
    let p3: i32 = causal_lifecycle(18);

    // gate-1 LOW-1 fix: non-degenerate value (7) so an identity-
    // laundering bug that mapped any input to 1 wouldn't silently
    // pass the cross-stack probe.
    let cs: i32 = cross_stack_known_cause(7);

    let p1_ok: i32 = if p1 == 10 { 1 } else { 0 };
    let p2_ok: i32 = if p2 == 14 { 1 } else { 0 };
    let p3_ok: i32 = if p3 == 18 { 1 } else { 0 };
    let cs_ok: i32 = if cs == 7  { 1 } else { 0 };

    let all_ok: i32 = p1_ok * p2_ok * p3_ok * cs_ok;
    all_ok * (p1 + p2 + p3)
}
