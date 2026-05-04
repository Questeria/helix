// agi_demo.hx — demonstration of Helix's compile-time AGI features
//
// This program won't run (it uses types we haven't added codegen for yet),
// but it WILL compile through the type checker. The type checker catches
// 4 different classes of bugs that no other language catches:
//
// 1. Shape mismatches (Presburger constraint solver)
// 2. Effect-capability violations (@pure / @io / etc.)
// 3. Silent gradient loss (D<T> type wrapper)
// 4. Cross-tier memory confusion (WorkingMem / EpisodicMem / SemanticMem)

module helix::demo

// ---------------------------------------------------------------------
// 1. SHAPE CHECKING
//    matmul has size-typed parameters; the inner dim must match.
// ---------------------------------------------------------------------
// Note: bodies are placeholders — full tensor codegen is Phase 4 work.
// This file demonstrates *type-system* features only.
@pure
fn matmul_sig[N: size, M: size, P: size](
    a: tensor<f32, [N, M]>,
    b: tensor<f32, [M, P]>,
) -> i32
where N % 16 == 0, M % 16 == 0, P % 16 == 0,
{
    0   // body returns i32 instead of a real tensor for now
}

// Caller with matching shapes — typechecks
@pure
fn good_caller(
    x: tensor<f32, [16, 32]>,
    y: tensor<f32, [32, 16]>,
) -> i32 {
    matmul_sig(x, y)
}

// ---------------------------------------------------------------------
// 2. EFFECT/CAPABILITY TRACKING
//    @pure functions cannot call effectful ones. Effects propagate.
// ---------------------------------------------------------------------
@pure
fn safe_calculation(x: i32) -> i32 {
    x * x + 1   // pure; no effects
}

@io
fn read_config() -> i32 {
    42   // imagine this reads a file
}

@io
fn read_and_compute() -> i32 {
    let cfg = read_config();    // OK: caller has @io capability
    safe_calculation(cfg)        // OK: pure callable from anywhere
}

// ---------------------------------------------------------------------
// 3. DIFFERENTIABILITY (D<T>)
//    Functions involved in gradient computation declare D-types.
//    The compiler refuses silent gradient loss.
// ---------------------------------------------------------------------
@pure
fn loss(x: D<f32>, y: D<f32>) -> D<f32> {
    let diff = x - y;
    diff * diff       // D propagates: D<f32> * D<f32> -> D<f32>
}

@pure
fn loss_with_constant(x: D<f32>) -> D<f32> {
    x * x + x         // mixing D<f32> with itself gives D<f32>
}

// ---------------------------------------------------------------------
// 4. MEMORY TIERS
//    Working / Episodic / Semantic / Procedural memory are separate types.
//    Cross-tier transitions are explicit (consolidate, recall, retrieve).
// ---------------------------------------------------------------------
fn current_belief(b: WorkingMem<i32>) -> WorkingMem<i32> {
    b   // can pass working memory through working-typed parameters
}

fn store_event(e: EpisodicMem<i32>) -> EpisodicMem<i32> {
    e   // episodic memory cannot be silently used as semantic
}

fn known_fact(s: SemanticMem<i32>) -> SemanticMem<i32> {
    s
}

// ---------------------------------------------------------------------
// 5. ALL COMBINED
//    A function with full Helix-flavor type discipline.
// ---------------------------------------------------------------------
@pure
fn agi_step[N: size](
    sensory: WorkingMem<tensor<f32, [N]>>,
    weights: D<tensor<f32, [N, N]>>,
) -> WorkingMem<tensor<f32, [N]>>
where N % 16 == 0,
{
    sensory
}

// ---------------------------------------------------------------------
// What this program demonstrates:
// - Compile-time shape constraints with Presburger reasoning
// - Capability-based safety: @pure cannot accidentally call @io
// - Type-level gradient flow tracking (catches silent grad-loss bugs)
// - Memory-tier distinction at the type level
//
// No language today (Mojo, Triton, Julia, JAX, Rust, C++) has all four.
// ---------------------------------------------------------------------
