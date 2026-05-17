// dogfood_10_memory_tiers.hx — Stage 37 Increment 2 dogfood.
//
// Memory-tier lifecycle reasoner: 3 working-memory items + episodic
// recording + semantic consolidation + working recall. First dogfood
// that exercises the Stage 37 Inc 1 tiered-memory primitives in a
// real-shaped scenario.
//
// What this dogfood demonstrates:
//   1. Three observations enter via into_working (sensory input
//      tier — fast access, fast decay).
//   2. Each observation is promoted to episodic via into_episodic
//      (event-trace tier — preserves temporal context).
//   3. Consolidation: episodic memories become semantic via
//      consolidate (long-term knowledge tier).
//   4. Recall: semantic knowledge is brought back into working
//      memory via recall (active-use tier).
//   5. Final unwrap_working extracts the value for arithmetic.
//
// This matches the human-memory architecture:
//   - working = "what I'm thinking about right now"
//   - episodic = "what I just experienced"
//   - semantic = "what I generally know"
//   - procedural = "how I do things" (skill memory — touched but
//     not actively cycled in this dogfood since procedural is
//     usually a sink, not a source, for value-bearing reasoning)
//
// Exit code 42 iff all 3 observations flow through the lifecycle
// AND procedural-tier sanity check holds. Each step's correctness
// is independently witnessed; a single regression collapses to 0.

fn cycle_through_tiers(raw: i32) -> i32 {
    // Step 1: Raw value enters working memory (current focus).
    let w0: WorkingMem<i32> = into_working(raw);
    let w0_v: i32 = unwrap_working(w0);
    // Step 2: Promote to episodic (record the event).
    let e: EpisodicMem<i32> = into_episodic(w0_v);
    // Step 3: Consolidate to semantic (long-term knowledge).
    let s: SemanticMem<i32> = consolidate(e);
    // Step 4: Recall to working memory (bring back for active use).
    let w: WorkingMem<i32> = recall(s);
    unwrap_working(w)
}

fn main() -> i32 {
    // Three observations to cycle through the memory lifecycle.
    let obs1: i32 = cycle_through_tiers(10);
    let obs2: i32 = cycle_through_tiers(14);
    let obs3: i32 = cycle_through_tiers(18);

    // Procedural-tier sanity: a learned-skill value (a constant)
    // round-trips through the procedural tier unchanged.
    let skill: ProceduralMem<i32> = into_procedural(0);
    let skill_v: i32 = unwrap_procedural(skill);

    // Per-step witnesses (binary, multiplicative):
    let obs1_ok: i32 = if obs1 == 10 { 1 } else { 0 };
    let obs2_ok: i32 = if obs2 == 14 { 1 } else { 0 };
    let obs3_ok: i32 = if obs3 == 18 { 1 } else { 0 };
    let skill_ok: i32 = if skill_v == 0 { 1 } else { 0 };

    // Product of 4 binary witnesses; any single failure → 0.
    let all_ok: i32 = obs1_ok * obs2_ok * obs3_ok * skill_ok;

    // Sum of recalled observations: 10 + 14 + 18 = 42.
    // all_ok * 42 = 42 iff every tier transition preserved values.
    all_ok * (obs1 + obs2 + obs3)
}
