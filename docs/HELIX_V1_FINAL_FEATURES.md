# Helix v1 Final Features — Compiler-Level Design

**Status**: Design doc — implementation queued for Stages 31+ after Stage 28.9-30 (bootstrap completion) lands.

**User directive (2026-05-11)**: "Anything that should be in Helix or is better in Helix do it for Helix compilation. Also do research on what other features like these we should add for the final Helix."

**Purpose update (2026-05-13)**: Helix is not limited to Kovostov. Kovostov is the first flagship system built on Helix; Helix itself should become a dominant open language for AGI development and for high-certainty systems across science, medicine, genomics, physics, mathematics, robotics, climate, energy, infrastructure, education, and any future field where auditable computation can reduce uncertainty for humanity.

This document captures features that **must** be added at the Helix language / compiler level (cannot be expressed as a library on top), plus a research catalog of features beyond the Locus-spec that should be considered for the final Helix.

The final product should be ambitious: not merely a better Python or Rust for AI, but a language where uncertainty, evidence, provenance, proof obligations, resource limits, sensor trust, and self-modification safety are first-class. Helix should make it harder for powerful systems to silently guess, silently corrupt, silently overclaim, or silently act outside their verified authority.

---

## Part 1 — Layer Discipline (the decision principle)

> **Anything that needs to FAIL TO COMPILE must be in the Helix compiler. Anything that just needs to WORK CORRECTLY at runtime can be a library.**

Rust's precedent: `core` (the unsafe-allowed primitives) → `std` (written in Rust on top of core) → ecosystem crates. Helix gets the same three layers:

- **Layer 0 — Helix-core (compiler-built-in)**: types/effects/runtime model that can't be a library
- **Layer 1 — Helix-std (in Helix-itself)**: compositional types built on Layer 0 primitives
- **Layer 2 — Helix programs / domain SDKs**: application logic for AGI, science, medicine, engineering, education, and industry

### Civilization-scale product goals

- **Uncertainty as a typed value**: confidence, probability, intervals, evidence quality, and out-of-distribution status should propagate through programs instead of being comments or dashboard text.
- **Proof-carrying computation**: critical functions should carry machine-checkable obligations and certificates where practical, especially in medicine, infrastructure, robotics, and scientific claims.
- **Causal and scientific reasoning**: support experiments, interventions, counterfactuals, units, dimensions, and provenance so research code can distinguish correlation, causation, assumption, and measurement.
- **Reproducible discovery**: binaries, datasets, model weights, transformations, and conclusions should be traceable and rebuildable.
- **Safe self-improvement**: reflection and code modification must stay verifier-gated, staged, reviewable, and reversible.
- **Human benefit as a design constraint**: features should be justified by the human problem they help solve, not by novelty alone.

---

## Part 2 — Layer 0 (Must Be Added to Helix Compilation)

These are the foundational features that **must** be in the type checker / IR / runtime. All are confirmed adds.

### 2.1 — Refinement types (Stage 31)

```helix
type Probability = Float where 0.0 <= self <= 1.0
type Temperature[Celsius] = Float where -273.15 <= self
type Distance[meters] = Float where self >= 0.0
type SafeSpeed[m/s] = Float where {
    self <= 5.0  if near(human),
    self <= 30.0 if on_road,
    self == 0.0  if obstacle_within(0.5.meters),
}
```

**Why must be Layer 0**: No library can refuse compilation on `let conf: Probability = 1.5`. The type checker needs to invoke an SMT solver (Z3 or similar) to discharge proof obligations.

**Compiler changes**:
- Extend `TyNode` with `TyRefine(base, predicate)`
- Typecheck: gather constraints at each value-use site, discharge via SMT bridge
- IR lowering: refinement vanishes (zero-cost), but a `@runtime_check` annotation can opt-in to dynamic checks at type boundaries
- New trap-id 31001-31999 for refinement violations

**Estimated effort**: Medium-large. ~2000 LOC in `typecheck.py` + new `refinement_solver.py`.

### 2.2 — Confidence types (Stage 32 — depends on 31)

```helix
let x: Float?0.9 = sensor.read()
let y: Float?0.8 = sensor.read()
let z = x * y    // z: Float?(0.9 * 0.8 = 0.72) — propagates automatically

if (x > threshold) under confidence:
    strong: do_action()
    weak: ask_human()
    unknown: defer()
```

**Why must be Layer 0**: "Cannot pass `Float?0.5` where `Float?0.9` required" needs the type checker. The propagation rules through arithmetic need to be **built into** the typecheck pass so they're consistent.

**Compiler changes**:
- New `TyConf(base, conf_expr)` — confidence is a refinement-typed `Float ∈ [0, 1]`
- Operator typing: `*`, `+`, `min/max` have known propagation rules
- New `under confidence` control flow construct
- Effect-system integration: `Cannot perform Motor action with Float?0.3 input` (capability + refinement combined)

**Estimated effort**: Medium. Builds on refinement types. ~800 LOC.

### 2.3 — Effect-system extensions (Stage 33)

Helix already has `@pure`, `@effect(io/ffi/arena/trace/reflect/tile_io)`. Add Locus-spec effects:

```helix
function learn(observation, label) -> updates: Learning
function predict(input) -> output with confidence: Inference
function explore(world) -> observations: Exploration
function ask_human(q) -> Answer: requires Human, async
function move_motor(cmd) -> (): requires Motor
```

Plus capability tokens:
```helix
function autonomous_robot(motor: cap Motor, sensor: cap Sensor) {
    // motor is a capability — no global access
}
```

**Why must be Layer 0**: New effect classes need to flow through `effect_check.compute_closure`. Capability passing changes the function signature ABI.

**Compiler changes**:
- Extend `OP_EFFECTS` and `META_ATTRS` 
- Add `cap T` type constructor (Phase-0 lowering: opaque token, runtime = ptr)
- New trap-ids 33001-33999

**Estimated effort**: Small (builds on existing effect system). ~400 LOC.

### 2.4 — Deadline / real-time types (obsolete Stage 34 label)

Stage-numbering note: this section predates the live roadmap. `docs/ROADMAP.md`
is authoritative for current stage numbers; current Stage 34 is Proof And
Refinement Expansion. Deadline and WCET work remains important, but its older
Stage 34 label is obsolete.

```helix
function emergency_brake() requires {
    deadline: 50.ms,
    no_blocking_io,
    no_dynamic_allocation,
}

function process_image(img: Image) -> Result deadline: 33.ms  // 30fps
```

**Why must be Layer 0**: WCET (worst-case execution time) analysis is a whole-program compiler pass. Cannot be a library.

**Compiler changes**:
- New `@deadline(N.ms)` attribute
- WCET analysis pass after IR lowering
- Refuses to compile if WCET > deadline
- Trap-ids 34001-34999
- Integration with no-GC discipline (Helix already has no GC)

**Estimated effort**: Large. WCET analysis is non-trivial. ~3000 LOC.

### 2.5 — Continuous execution model (future post-Stage-35 roadmap item)

```helix
continuous program robot_brain {
    state world_model: WorldModel
    state self_model: SelfModel
    
    on sensor_update(reading) {
        world_model.update(reading)
    }
    on time_passes(every: 100.ms) {
        self_model.consolidate()
    }
    on novel_observation(obs) {
        ask_human(formulate_question(obs))
    }
}
```

**Why must be Layer 0**: Runtime model change. No `main()` / exit concept. Event-loop scheduling baked into emission.

**Compiler changes**:
- New top-level item `continuous program`
- Event handlers compile to message-pump callbacks
- State persists across event firings
- Lowering to native event loop (Linux epoll for the x86_64 target)
- Trap-ids 35001-35999

**Estimated effort**: Large — fundamental runtime change. ~2500 LOC.

### 2.6 — Tiered memory (Stage 36)

```helix
storage hot: RAM, expires: 2.hours, detail: Full
storage warm: SSD, expires: 30.days, detail: Summary
storage cold: HDD, expires: Never, detail: Essence

remember(observation)        // automatic tier placement
recall(query) -> Known<T>    // automatic tier traversal, conf decreases with tier
```

**Why must be Layer 0**: Compiler-managed placement decisions. Memory lifetime tracking. Tier-traversal cost analysis.

**Compiler changes**:
- New `storage` top-level declaration
- New `remember`/`recall` primitives
- Lifetime analysis extension
- Runtime: tier-managed allocator
- Trap-ids 36001-36999

**Estimated effort**: Large. ~3000 LOC. Requires runtime support.

### 2.7 — Theorem-prover bridge (Stage 37)

Z3 (or similar SMT solver) integrated with refinement-types, deadline analysis, separation-logic obligations.

```helix
theorem brake_response_bounded {
    forall input: SensorReading,
    response_time(emergency_brake(input)) < 50.ms
}
// compiler discharges via SMT or refuses to compile
```

**Compiler changes**:
- `theorem` top-level item
- SMT export of Helix types/constraints
- Round-trip proof certificate cache (so re-compilation doesn't re-prove)
- Trap-ids 37001-37999

**Estimated effort**: Medium (SMT bridge is well-trodden ground). ~1500 LOC + Z3 dep (FFI'd).

---

## Part 3 — Layer 1 (Built IN HELIX after Self-Hosting)

These compose on top of Layer 0 — they get written in `.hx` files in `helixc/stdlib/`, not in the Python compiler:

| Layer-1 feature | Built from | Stages |
|---|---|---|
| **Spatial types + frames** (`Position<Frame>`, transform safety) | refinement + phantom generics (Helix already has) | 38 |
| **Temporal types** (Timestamp, Duration, history queries) | refinement-bounded i64 + std combinators | 39 |
| **Knowledge<T>, Unknown<T>, Belief<T>** | confidence + temporal + provenance struct | 40 |
| **Self-model primitives** (self.confidence, self.knowledge) | reflection (Helix already has quote/splice) | 41 |
| **World modeling primitives** (WorldModel, SemanticMap, ConfidenceMap, Pattern) | spatial + temporal + knowledge | 42 |
| **Sensor fusion** (`fuse { camera, lidar, imu } into ...`) | algebraic effects + spatial alignment | 43 |
| **Active learning primitives** (`novelty(obs)`, curiosity) | Learning effect + Knowledge<T> | 44 |
| **Communication / actors** | algebraic effects + FFI | 45 |
| **Vector store / knowledge graph types** | tiered memory + Knowledge<T> | 46 |

---

## Part 4 — Layer 0 Research: Beyond Locus

Locus is one design. Other PL and AI-systems research suggests additional features worth adding for a true AGI substrate. Categorized by priority.

### 4.1 — Tier S (Critical additions beyond Locus)

#### Probabilistic programming primitives

```helix
let coin: Bool = sample(Bernoulli(0.6))
observe(measured_x = sample(Normal(coin, 0.1)))
let posterior = condition(coin | measurements)
```

Why critical: Locus mentions `Belief<T>` but doesn't formalize sample/observe. Without these, Bayesian AI reasoning is library-only and loses compile-time check that observe-statements are consistent with their priors.

#### Information flow / privacy types

```helix
type Public = Visibility::Public
type Confidential<owner> = Visibility::Restricted<owner>

let patient_record: PatientData @ Confidential<Patient>
let public_summary: Summary @ Public = derive_summary(patient_record)  // COMPILE ERROR
// patient_record contains Confidential data; public_summary cannot flow it out
```

Why critical: Helix is an AGI substrate; medical/legal/regulated AI needs **provable** info-flow. Refinement types check value invariants; flow types check value *origin*. Different axis, both required for safety.

#### Differential privacy types

```helix
type Private<eps> = T with privacy_budget: eps
let aggregate: Float @ Private<0.1> = mean_with_noise(individual_records, eps=0.1)
// Compiler tracks budget exhaustion; cannot re-query when budget = 0
```

Critical for medical AI, federated learning, anything regulated.

#### Linear / affine types

```helix
linear type FileHandle
fn process(h: FileHandle) -> () {
    read(h)   // OK
    read(h)   // COMPILE ERROR — handle consumed
}
```

Locus mentions Rust-style ownership but doesn't go to linear types. For AGI: critical for **single-use cryptographic keys**, **one-shot RPC tokens**, **resource budgets** (energy, bandwidth, compute quota).

#### Algebraic effects with handlers (Koka / Frank / Eff style)

Generalize Locus's effect system. Instead of `requires Motor`, allow user-defined effects + handlers:

```helix
effect Choice { fn pick<T>(options: List<T>) -> T }

with Choice::pick handled_by {
    case (opts, k) => k(opts[0])    // deterministic: pick first
} run plan_action()
```

Why critical: more flexible than capability tokens. Effect handlers are the modern way to do dependency injection, async, retry, transactions, and probabilistic programming. Helix already has effect-tagged functions; adding handlers is the natural evolution.

#### Type-level computation for tensor shapes

```helix
fn matmul<N: Nat, M: Nat, K: Nat>(
    a: Tensor[f32, [N, M]],
    b: Tensor[f32, [M, K]],
) -> Tensor[f32, [N, K]]
```

Helix's `TyArray.size` is already Expr-valued (cycle-49 mono-hashing fix exercised this). Going further to **type-level Nat arithmetic** (`Add<N, M>`, `Mul<N, K>`) lets the compiler enforce shape correctness without runtime checks. Critical for tensor-heavy AI.

#### GADTs (Generalized Algebraic Data Types)

```helix
enum Term<T> {
    case IntLit(i32): Term<i32>
    case BoolLit(bool): Term<bool>
    case If<T>(Term<bool>, Term<T>, Term<T>): Term<T>
    case Add(Term<i32>, Term<i32>): Term<i32>
}
fn eval<T>(t: Term<T>) -> T { ... }
```

Why critical: Helix's quote/splice (reflection) currently returns opaque AST handles. GADTs let reflected ASTs be **typed** — `Term<i32>` vs `Term<bool>`. Massive upgrade to metaprogramming safety. Necessary for the AGI self-modification story.

### 4.2 — Tier A (Strong additions)

#### Counterfactual / causal types (deeper than Locus)

```helix
fn what_if<X>(intervention: Intervention<X>, outcome: Outcome<Y>) -> CounterfactualDist<Y>
```

Locus distinguishes `causes` / `correlates` / `preceded`. Deeper: types that represent **interventional distributions** (Pearl's do-calculus). Critical for: medical decision-making, AI explanation, scientific discovery.

#### Quantization-aware types

```helix
type Q8 = i8 with quantization { scale: f32, zero_point: i8 }
fn quantize(x: f32) -> Q8 { ... }
fn dequantize(x: Q8) -> f32 { ... }
```

For efficient inference. AI deployment on edge needs this.

#### Adversarial robustness types

```helix
fn classify(img: RobustWithin<eps=0.03>(Image)) -> Class
// the classifier is provably-robust to perturbations within eps
```

#### Out-of-distribution / data-domain types

```helix
type InDist<dataset> = T with origin: dataset_id
fn classify(x: InDist<ImageNet>(Image)) -> ImageNetClass
// rejects inputs flagged out-of-distribution
```

#### Higher-kinded types

```helix
trait Functor<F<_>> { fn map<A, B>(self: F<A>, f: fn(A) -> B) -> F<B> }
impl Functor for Option<_> { ... }
impl Functor for Confidence<_> { ... }
impl Functor for Known<_> { ... }
```

Enables compositional stdlib. Currently Helix is monomorphized; lifting to HKT enables generic combinators.

#### Energy / power budget types

```helix
fn predict(x: Input) -> Output budget: 50.mJ
```

For edge AI on battery devices.

#### Path-dependent types / first-class modules

```helix
module Sensor { type Reading = ...; fn read() -> Reading }
fn process(s: Sensor) { let r: s.Reading = s.read(); ... }
```

For modular AGI architecture composition.

### 4.3 — Tier B (Practical additions)

- **Holes** (typed `_`) — for AI-assisted development; the compiler tells you what type is required
- **Property-based testing** built into the language (Helix would have a `property` keyword like Idris/Haskell QuickCheck)
- **Row polymorphism** — structural records for ML feature engineering
- **Operator overloading** (limited) — Helix already has it for tensor `@`; generalize
- **View patterns** — match by transformation
- **Pattern synonyms** — abstract over pattern matching
- **Mutation testing** built-in — confidence in test quality

### 4.4 — Tier C (Future / nice-to-have)

- **Region-based memory** (ML kit, Cyclone) — bounded lifetimes beyond Rust ownership
- **Session types** — typed protocols (HTTP request-response)
- **CRDT types** — conflict-free replicated data, distributed AI
- **Quantitative type theory** (Idris 2) — track exact number of uses (0/1/many)
- **Gradual typing** — mix dynamic and static parts, useful for ML notebook integration
- **Macros 2.0 (hygienic)** — Helix has quote/splice; upgrade to hygiene-tracking
- **Time-travel debugging** — replay continuous-program state
- **Trusted execution types** — `InEnclave<T>` for SGX/TEE

---

## Part 5 — Sequencing Plan

```
CURRENT (May 2026):
  Stage 28.9 — bootstrap port (audit cycles 51+)
  Stage 28.10-28.13 — port remaining frontend passes
  Stage 29 — byte-identical self-hosting verification
  Stage 30 — 5 clean audits on self-host alone
  → v0.1 RELEASE (self-hosting Helix as it is today)

PHASE 1 (Stages 31-37, written in Python helixc → ported through bootstrap):
  Stage 31 — Refinement types + SMT bridge (foundation for everything else)
  Stage 32 — Confidence types (depends on 31)
  Stage 33 — Effect-system extensions (Learning, Asking, Motor, Sensor, capabilities)
  Stage 34 — Proof And Refinement Expansion (see docs/ROADMAP.md; replaces the
              older Deadline + WCET label)
  Stage 35 — AI/ML Capability Push (see docs/ROADMAP.md)
  Stage 36 — Strategic AGI features (see docs/ROADMAP.md)
  Stage 37+ — Continuous execution, tiered memory, theorem-prover integration
              sequencing will be assigned after the Stage 35 closeout.

PHASE 2 (Stages 38-46, written IN HELIX in helixc/stdlib/):
  Stage 38 — Spatial types + frames
  Stage 39 — Temporal types
  Stage 40 — Knowledge<T> / Unknown<T> / Belief<T>
  Stage 41 — Self-model primitives
  Stage 42 — World modeling primitives
  Stage 43 — Sensor fusion
  Stage 44 — Active learning primitives
  Stage 45 — Communication / actors
  Stage 46 — Vector store + knowledge graph types

PHASE 3 (Stages 47+, written IN HELIX):
  Stage 47 — Probabilistic programming primitives (Tier S)
  Stage 48 — Information flow + privacy types (Tier S)
  Stage 49 — Differential privacy budget tracking (Tier S)
  Stage 50 — Linear / affine types (Tier S)
  Stage 51 — Algebraic effects + handlers (Tier S)
  Stage 52 — Type-level Nat arithmetic for tensor shapes (Tier S)
  Stage 53 — GADTs for typed reflection (Tier S)
  
  Stage 54 — Counterfactual / causal types (Tier A)
  Stage 55 — Quantization-aware types (Tier A)
  Stage 56 — Adversarial robustness types (Tier A)
  Stage 57 — Out-of-distribution types (Tier A)
  Stage 58 — Higher-kinded types (Tier A)
  Stage 59 — Energy / power budget types (Tier A)
  Stage 60 — Path-dependent / first-class modules (Tier A)
  
  Stage 61 — Holes + type-directed search (Tier B)
  Stage 62 — Property-based testing built-in (Tier B)
  Stage 63 — Row polymorphism (Tier B)
  Stage 64 — Mutation testing (Tier B)
  
  Stage 65+ — Tier C as needed: region memory, session types, CRDTs, QTT, etc.

  Stage X — v1.0 RELEASE (target ~2027)
```

---

## Part 6 — Discipline Notes

1. **Every feature must justify itself by which AGI outcome it enables.** Same as Locus's design principle. No bloat.

2. **Layer 0 features have to survive the audit-driven 5-clean criterion** for each stage, same as Stage 28.9. No shortcuts.

3. **Layer 1+ in Helix-itself, not Python.** Once self-hosting, the language must extend itself. If we write `Knowledge<T>` in Python helixc, we've defeated the self-hosting goal.

4. **Provenance preserved.** Every value's audit trail (where it came from, when, with what confidence) must flow through the type system, not get stripped at type-erasure boundaries.

5. **No GC, ever.** Bootstrap from raw binary precludes GC; AGI-substrate latency precludes GC; ownership + region/linear types replace it. Already true in Phase-0 Helix.

6. **Theorem-prover integration is the gate-keeper.** Stage 31+ assumes Z3 (or equivalent) is reachable. Theorems must produce machine-checkable certificates so re-compilation doesn't re-prove from scratch.

7. **No dynamic dispatch in core type-checking.** All proof obligations must be statically discharged or explicitly downgraded with `@runtime_check`. Predictable timing.

8. **Standard library is part of the spec.** Not an afterthought. Layer 1 features (spatial, temporal, knowledge) are as much "Helix" as Layer 0.

---

## Status

**Active**: Stage 28.9 audit cycles 51+ (this document drafted at cycle 53 fix-sweep, heavy gate green at 1514/1/0).

**Next major milestone**: Stage 30 — 5 clean audits on self-host alone → v0.1 release.

**Then**: Stage 31 — Refinement types (Phase 1 begins).

---

*This document is itself part of the bootstrap discipline: written down so it survives across sessions, audit cycles, and (eventually) reimplementations.*
