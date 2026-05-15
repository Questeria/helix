# Helix Final Product Research Blueprint

**Status**: research-backed north star for post-Stage-30 Helix.
**Date**: 2026-05-13.
**Scope**: what Helix should fully become, what it must dream toward, and what
specific capabilities are needed for Helix to become a language for future
human development and prosperity.

## Executive Thesis

Helix should become a self-hosted, deterministic, proof-aware, AI-native,
multi-target language for high-certainty civilization-scale software.

Its central job is not only to make programs run. Its job is to make important
systems less able to hide uncertainty, provenance loss, unsafe authority,
resource misuse, timing failure, memory unsafety, and unverifiable claims.

No programming language can guarantee that humanity is saved. Helix should be
designed around the humbler and stronger target: reduce the ways powerful
software can silently fail, silently overclaim, silently act, or silently drift
away from human benefit.

If Helix succeeds, it should become the language people and AIs choose when:

- A wrong answer can harm a patient, vehicle, robot, lab, model, city, or
  civilization-scale decision.
- A claim needs evidence, not vibes.
- A system learns over time and must remember what it knows, what it believes,
  and what it does not know.
- A program acts on the physical world and must prove it has the authority,
  confidence, timing, and spatial safety to act.
- A compiler, model, or agent improves itself and must pass verifier gates
  before its changes become real.

## The Helix Promise

Helix should make the following sentence true:

> If uncertainty, authority, time, memory, hardware, provenance, or proof affects
> the safety of an action or claim, Helix makes that fact visible to the compiler
> and to the human or AI reviewing the system.

This is the DNA-level purpose. Helix should not just be a syntax. It should be a
substrate for reliable intelligence.

## Civilizational Design Laws

These laws should guide every future feature decision.

1. **If uncertainty can affect action, the compiler must see it.**
   Confidence, intervals, distributions, evidence quality, out-of-distribution
   status, and unknown values cannot be informal comments.

2. **If code can act on the world, authority must be typed.**
   Motor control, network access, money movement, medical recommendation,
   model training, memory mutation, and self-modification need explicit
   effects and capability tokens.

3. **If a claim matters, provenance must survive.**
   Scientific, medical, legal, safety, and AGI self-model claims need source,
   timestamp, transformation history, confidence, and audit trail.

4. **If timing matters, timing must be part of the program.**
   Real-time robotics, vehicles, industrial systems, and medical devices need
   deadline/resource obligations checked by the compiler where possible.

5. **If memory persists, memory must have type, tier, owner, and decay.**
   Long-running AI systems need memory that can be hot, warm, or cold; exact,
   summarized, or reconstructed; fresh, stale, or expired.

6. **If hardware matters, hardware constraints must be declared.**
   CPU, GPU, FPGA, accelerator, memory, energy, and real-time requirements
   should be visible to the compiler.

7. **If code changes code, verification and rollback are mandatory.**
   Self-improvement must be staged, reviewed, deterministic, reversible, and
   tied to proof or test evidence.

8. **If Python or outside tooling is used, Helix remains the source of truth for
   core behavior.**
   Python can remain useful for tests, scripts, research tooling, and migration.
   It should not remain the long-term authority for Helix compiler semantics.

9. **If a feature cannot explain which human outcome it helps, it is not yet a
   core feature.**
   Ambition is good. Unjustified complexity is not.

## What Research Says Helix Should Learn From

Helix should not copy one existing system. It should combine the strongest
lessons from many systems into one language stack.

| Area | Systems studied | Lesson for Helix |
| --- | --- | --- |
| Verified compilers | [CompCert](https://compcert.org/), [CakeML](https://cakeml.org/) | Verified compiler cores are possible, but scope must be controlled. Helix should grow a verified core and use translation validation for wider backends. |
| Verified kernels | [seL4](https://sel4.systems/Verification/) | Full-system verification is possible when specs, implementation, and proof are treated as one product. Helix should emit proof obligations and certification artifacts. |
| Program verification | [Dafny](https://dafny.org/), [F*](https://www.fstar-lang.org/), [Lean](https://lean-lang.org/) | Specifications and proofs must be first-class workflow, not rare expert-only ceremony. |
| Refinement types | [Liquid Haskell](https://ucsd-progsys.github.io/liquidhaskell/) | SMT-backed refinements can catch real errors while staying close to normal programming, but decidability boundaries must be explicit. |
| Memory and ownership | [Rust ownership](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html) | No-GC safety can be practical if ownership and lifetimes are language-level ideas. Helix should extend ownership to epistemic and tiered memory. |
| Capabilities | [CHERI](https://www.cl.cam.ac.uk/research/security/ctsrd/cheri/) | Language-level authority works best when hardware can also enforce capability boundaries. Helix should be future-ready for capability hardware. |
| Actor concurrency | [Pony](https://www.ponylang.io/), [Erlang](https://www.erlang.org/) | Isolated actors, capabilities, and supervision help build reliable concurrent systems. Helix should make actor isolation and effect safety native. |
| AI compiler stacks | [JAX](https://docs.jax.dev/en/latest/), [Triton](https://triton-lang.org/main/index.html), [MLIR](https://mlir.llvm.org/), [Enzyme](https://enzyme.mit.edu/) | AI languages need composable transformations, domain IRs, autodiff, and hardware-specialized kernels. Helix should own these ideas instead of bolting them on. |
| Probabilistic and causal programming | [Stan](https://mc-stan.org/), [Pyro](https://pyro.ai/), [DoWhy](https://www.pywhy.org/dowhy/) | Uncertainty and causality need first-class modeling. Helix should distinguish observed evidence, belief, intervention, counterfactual, and causal claim. |
| Robotics frames | [ROS 2 tf2](https://docs.ros.org/en/rolling/Concepts/Intermediate/About-Tf2.html) | Frame transforms are a real source of robot bugs. Helix should make spatial frame mismatches compile errors. |
| Temporal specs | [TLA+](https://lamport.azurewebsites.net/tla/tla.html) | Concurrent and distributed systems need temporal reasoning beyond unit tests. Helix should include temporal properties and model checking hooks. |
| AI risk management | [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) | AI systems need governance, mapping, measurement, and management artifacts. Helix should emit risk and audit artifacts from the build. |
| Medical AI lifecycle | [FDA GMLP](https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles), [IEC 62304](https://www.iso.org/standard/38421.html) | Medical systems require lifecycle discipline, data provenance, risk controls, and change management. Helix should make regulated evidence easier to produce. |
| Automotive safety | [ISO 26262](https://www.iso.org/standard/68383.html) | Safety-critical software needs hazard analysis, functional safety, traceability, and verification evidence. Helix should support safety cases as build artifacts. |
| Provenance and FAIR data | [W3C PROV](https://www.w3.org/TR/prov-overview/), [FAIR principles](https://www.go-fair.org/fair-principles/) | Data and knowledge must be findable, accessible, interoperable, reusable, and traceable. Helix knowledge types should preserve provenance by default. |
| Reproducible systems | [Reproducible Builds](https://reproducible-builds.org/), [Software Heritage](https://www.softwareheritage.org/) | A binary should be tied to source, toolchain, data, and history. Helix should treat reproducible builds as a core safety feature. |

## Minimum Final Product Shape

The final Helix product should be organized as five mutually reinforcing layers.

### Layer 0: Helix Core

This is the part that must be in the compiler and core language because it needs
to make code fail to compile.

Required Layer 0 capabilities:

- Refinement types.
- Confidence and uncertainty types.
- Knowledge, evidence, belief, and unknown types.
- Effect system and capability tokens.
- Spatial types with frames, units, and relationships.
- Temporal types with validity, freshness, histories, deadlines, and causality.
- Ownership, lifetime, and no-GC memory model.
- Tiered memory model with confidence decay.
- Resource and energy types.
- Deterministic concurrency model.
- Continuous program model.
- Hardware and target types.
- Proof obligation generation.
- Safe FFI contracts.
- Self-modification gates.
- Information-flow, privacy, and consent types.

### Layer 1: Helix Runtime

This is the long-running substrate that makes Helix programs live in time.

Required runtime capabilities:

- Deterministic scheduler.
- Real-time scheduling profile.
- Actor runtime with isolated state and explicit messages.
- Continuous program lifecycle.
- Hot/warm/cold memory tiers.
- Provenance/audit event stream.
- Snapshot and rollback.
- Sensor fusion runtime.
- Knowledge graph runtime.
- Vector store runtime.
- Secure capability registry.
- Build artifact and proof artifact registry.
- Reproducible package/build resolver.

### Layer 2: Helix Standard Library

This is reusable Helix code built on Layer 0. It should be written in Helix as
soon as self-hosting can support it safely.

Required standard library families:

- Math, units, intervals, distributions, and linear algebra.
- Tensor, autodiff, optimizer, and neural network primitives.
- Probabilistic programming and causal inference primitives.
- Spatial geometry, transforms, maps, and collision reasoning.
- Temporal histories, event streams, windows, clocks, and causal graphs.
- Data structures with explicit memory tier and complexity behavior.
- Cryptographic hashing and reproducibility utilities.
- Domain error types and result/evidence wrappers.
- Safe FFI wrappers for C, Python, CUDA, ROCm, Vulkan, and OS syscalls.

### Layer 3: Domain SDKs

These are not all compiler features, but Helix should ship official high-quality
SDKs because the language mission depends on them.

Required domain SDKs:

- AGI/AI research SDK.
- Robotics SDK.
- Scientific computing SDK.
- Medical/clinical decision-support SDK.
- Genomics and bioinformatics SDK.
- Physics/simulation SDK.
- Climate/geospatial SDK.
- Education and personalized learning SDK.
- Infrastructure and safety-critical systems SDK.
- Distributed multi-agent systems SDK.

### Layer 4: Tooling and Certification

Helix should ship with tools that make safety and truth easier than drift and
guesswork.

Required tooling:

- Compiler diagnostics that explain proof failures in beginner-friendly terms.
- Theorem/proof assistant bridge.
- SMT solver bridge.
- Model checker bridge.
- Time-travel debugger.
- Confidence/provenance debugger.
- Spatial frame visualizer.
- Temporal event graph viewer.
- Safety-case generator.
- Medical/regulatory evidence pack generator.
- Deterministic build and binary comparison tool.
- Self-host cascade tool.
- Package verifier and supply-chain checker.

## Foundational Type System Requirements

The type system should be the center of Helix. Every other capability depends on
what the compiler can see.

### Refinement Types

Refinement types let Helix say a value is not merely a `Float`, but a `Float`
inside a valid range or satisfying a safety rule.

Examples:

```helix
type Confidence = Float where 0.0 <= self <= 1.0
type Probability = Float where 0.0 <= self <= 1.0
type Celsius = Float where self >= -273.15
type DistanceMeters = Float where self >= 0.0
type NonEmptyText = Text where length(self) > 0
```

Required compiler behavior:

- Generate proof obligations when assigning, returning, or passing refined
  values.
- Discharge simple arithmetic obligations with an SMT solver.
- Emit clear compile errors when proof fails.
- Allow runtime checks at trust boundaries.
- Preserve proof artifacts for audit.

Acceptance gate:

- `let p: Probability = 1.2` fails compilation.
- `let p: Probability = clamp(x, 0.0, 1.0)` compiles only if `clamp` has a
  proven postcondition.

### Confidence Types

Every value that comes from uncertain observation, model output, memory
reconstruction, sensor fusion, or inference should be able to carry confidence.

Examples:

```helix
let reading: Temperature?0.93 = sensor.read_temperature()
let diagnosis: Diagnosis?0.71 = model.predict(patient)
let position: Position@World?0.88 = fuse(camera, lidar, imu)
```

Required compiler behavior:

- Treat confidence as a refinement-checked value in `[0, 1]`.
- Propagate confidence through arithmetic and domain operations.
- Reject low-confidence values where high confidence is required unless the
  program explicitly upgrades, checks, asks a human, or chooses a fallback.
- Require confidence-aware branching for actions that depend on uncertain
  predicates.

Required design work:

- Define propagation rules for independent evidence.
- Define propagation rules for correlated evidence.
- Define how confidence differs from probability.
- Define interval confidence and distribution confidence.
- Define how confidence decays through memory tiers and time.

Acceptance gate:

- A motor action requiring `Position?0.95` rejects `Position?0.60`.
- A medical recommendation cannot hide the confidence of the diagnosis that
  produced it.

### Knowledge, Evidence, Belief, and Unknown

Helix should distinguish facts, beliefs, missing knowledge, and evidence.

Core shapes:

```helix
type Known<T> = {
    value: T,
    confidence: Confidence,
    source: Provenance,
    observed_at: Timestamp,
    valid_until: Timestamp?,
    location: Position?,
    causal_chain: KnowledgeRef[],
}

type Unknown<T> = {
    last_known: Known<T>?,
    why_unknown: Reason,
    can_resolve_by: Action[],
}

type Belief<T> = {
    prior: Distribution<T>,
    evidence: Evidence[],
    posterior: Distribution<T>,
}
```

Required compiler behavior:

- Reject `Unknown<T>` where `Known<T>` is required.
- Reject a `Belief<T>` being silently treated as a `Known<T>`.
- Preserve provenance through transformations by default.
- Require explicit downgrade when provenance is intentionally summarized.
- Require explicit evidence update for belief revision.

Acceptance gate:

- A clinical conclusion cannot be compiled without a provenance trail.
- A scientific claim cannot accidentally erase which data and transformations
  produced it.

### Spatial Types

Space must be a first-class dimension, not a tuple convention.

Examples:

```helix
type WorldFrame = Frame absolute
type CameraFrame = Frame relative_to camera
type RobotFrame = Frame relative_to robot

let obs: Position@CameraFrame = camera.detect()
let world_obs: Position@WorldFrame = obs.transform_to(WorldFrame)
```

Required compiler behavior:

- Reject operations between positions in different frames unless transformed.
- Track units such as meters, millimeters, radians, and degrees.
- Track geometric relationships such as `contains`, `near`, `inside`, and
  `clear_of`.
- Combine spatial reasoning with confidence and refinement types.

Acceptance gate:

- `Position@CameraFrame + Position@WorldFrame` fails compilation.
- Fast robot motion cannot accept a target that may be inside a human safety
  zone.

### Temporal Types

Time must be a first-class dimension.

Examples:

```helix
type Timestamp = Moment in Time[UTC]
type Duration = Interval in Time
type Fresh<T> = T with expires_at: Timestamp
```

Required compiler behavior:

- Track when values were observed.
- Track when values expire.
- Reject stale data in contexts requiring fresh data.
- Distinguish causation, correlation, and sequence.
- Support deadline and resource obligations.

Acceptance gate:

- A robot cannot use a stale obstacle map for a fast motion plan.
- A scientific conclusion cannot compile as causal when the program only proved
  temporal precedence or correlation.

### Effect and Capability Types

Functions should declare what they can do, not only what they accept and return.

Examples:

```helix
function read_sensor() -> Reading requires IO
function predict(input: Input) -> Output?Confidence pure
function move_motor(cmd: MotorCommand) -> () requires Motor
function learn(obs: Observation, label: Label) -> Update requires Learning
function ask_human(q: Question) -> Answer requires Human, Async
```

Required compiler behavior:

- Track effects through the call graph.
- Reject effectful calls from pure contexts.
- Require explicit capabilities for dangerous actions.
- Log high-impact effects into provenance/audit streams.
- Support policy rules such as "no network inside clinical decision core" or
  "no blocking IO inside emergency braking".

Acceptance gate:

- A pure prediction function cannot mutate a model.
- A function without `Motor` capability cannot move a robot.

### Information-Flow, Privacy, and Consent Types

If Helix serves medicine, genomics, education, and AGI memory, it must know when
data is private, consent-limited, sensitive, or allowed to leave a boundary.

Examples:

```helix
type PatientData = Record tagged Private, Medical, ConsentScoped
type Genome = Sequence tagged Private, Identifying, ConsentScoped
type PublicFinding = Text tagged Public
```

Required compiler behavior:

- Reject private data flowing into public outputs without an approved
  de-identification proof.
- Track consent scopes and expiration.
- Prevent training on data whose consent does not allow training.
- Track jurisdiction/policy constraints as compile-time or build-time
  obligations.

Acceptance gate:

- A model training job cannot compile if its dataset includes patient data not
  consented for training.
- A report cannot export genome-identifying information through an unapproved
  output channel.

## Memory and Continuous Execution Requirements

Helix should support programs that live continuously, learn over time, and do
not depend on stop-the-world garbage collection.

### Continuous Programs

Helix should support programs that do not conceptually start, run, and exit.
They should exist, observe, remember, act, and consolidate.

Example:

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
}
```

Required runtime behavior:

- Deterministic scheduling option.
- Real-time scheduling option.
- State snapshotting.
- Hot/warm/cold memory migration.
- Event-sourced audit trail.
- Recovery after crash.

Acceptance gate:

- A continuous program can run, snapshot, restart, and prove that durable
  knowledge state matches the logged event stream.

### Tiered Memory

AI memory cannot just be heap allocation. It needs fidelity, confidence, owner,
and lifetime.

Required tiers:

- **Hot**: current detailed working memory.
- **Warm**: summarized persistent memory.
- **Cold**: durable core memory, identity, long-term facts, and major
  provenance.

Required behavior:

- Recent observations default to hot.
- Consolidated patterns move to warm.
- Durable identity and verified facts move to cold.
- Confidence changes when memory is summarized or reconstructed.
- Forgotten or expired memory must be explicit, not silent.

Acceptance gate:

- A recalled warm memory is not typed the same as an exact hot observation.
- A program must acknowledge lower confidence when using reconstructed memory.

### No Garbage Collection

Helix should avoid stop-the-world garbage collection in the core runtime because
real-time AGI, robotics, vehicles, industrial systems, and medical devices
cannot accept unpredictable pauses.

Required model:

- Ownership and borrowing for normal values.
- Regions/arenas for scoped allocation.
- Deterministic destructors.
- Compiler-proven cleanup.
- Tiered memory migration for continuous state.
- No hidden global mutable ownership.

Acceptance gate:

- A real-time function can require `no_dynamic_allocation`.
- A deadline-critical function cannot call code that may allocate or block.

## Compilation and Hardware Requirements

Helix should compile from one source into multiple hardware forms while
preserving correctness evidence as much as possible.

### Multi-Level Compiler Architecture

The compiler should be organized into explicit levels:

1. Parsed AST.
2. Typed AST with effects, refinements, confidence, space, time, and provenance.
3. High-level Helix IR.
4. Verification IR.
5. Optimization IR.
6. Backend-specific IRs.
7. Binary/kernel/bitstream output.

Required validation:

- Deterministic build hashes.
- Translation validation between optimization stages.
- Proof artifact emission.
- Whole-program checks for deadlines, memory, and effects.

### Backends

Priority order:

1. Current x86-64 ELF backend.
2. Self-hosted Helix compiler path.
3. LLVM IR for wider CPU support.
4. PTX/CUDA for NVIDIA GPUs.
5. ROCm or MLIR GPU path for AMD GPUs.
6. Vulkan compute for cross-vendor GPU.
7. RISC-V for open hardware.
8. FPGA Verilog/VHDL for low-latency pipelines.
9. ASIC-oriented bytecode or hardware description for future AI silicon.

Acceptance gate:

- A small tensor kernel compiles to CPU and GPU with the same typed shape
  contract and deterministic test output.

### Global Optimization

Helix should optimize whole programs, not isolated functions, because the major
targets are AI systems, proof-carrying systems, and real-time systems.

Required optimization families:

- Constant folding.
- Dead code elimination.
- Common subexpression elimination.
- Operation fusion.
- Shape specialization.
- Memory layout specialization.
- Confidence-aware precision selection.
- Hardware target selection.
- Effect-aware parallelization.
- Deadline-aware scheduling.

Acceptance gate:

- Optimization cannot change observable effects, confidence/provenance
  obligations, or proof obligations.

## AI and AGI Requirements

Helix should be designed for AI systems to use and extend, not only for human
developers.

### Native Tensor and Autodiff Stack

Required features:

- Shape-typed tensors.
- Device-typed tensors.
- Dtype-typed tensors.
- Compile-time shape checking.
- Forward and reverse autodiff.
- Differentiable function contracts.
- Fused kernels.
- Verified shape transforms.
- Confidence-aware model output.
- Provenance-aware training data.

Acceptance gate:

- A shape mismatch fails before runtime.
- A trained model artifact records source data, training code, hyperparameters,
  build hash, and evaluation evidence.

### Active Learning

Helix should treat learning as an effect and as a first-class system behavior.

Required primitives:

- `predict`: pure inference.
- `learn`: modifies model state.
- `explore`: seeks new evidence.
- `ask_human`: requests human input.
- `novelty`: detects surprise.
- `curriculum`: orders learning tasks.

Acceptance gate:

- A pure function cannot secretly learn.
- A system can explain why it asked a human a question.

### Self-Model

AGI-grade systems need to reason about their own knowledge, uncertainty,
capabilities, limits, and history.

Required primitives:

- `self.knowledge`
- `self.uncertainty`
- `self.capabilities`
- `self.history`
- `self.constraints`
- `self.confidence_in(claim)`

Acceptance gate:

- A system can return an explicit `Unknown` value with resolution options
  instead of fabricating an answer.

### Safe Self-Improvement

Helix currently has a verifier-gated reflective-cell scaffold. The final system
should expand it into a complete self-improvement discipline with runtime AST
handles, real splice execution, and source rewrite/commit semantics.

Required gates:

- Proposed change is represented as data.
- Tests are generated and run.
- Proof obligations are generated and checked where possible.
- Self-host cascade passes.
- Binary determinism check passes.
- Rollback plan exists.
- Human approval can be required by policy.
- Provenance of the change is recorded.

Acceptance gate:

- A compiler change cannot become the active compiler unless it passes the
  configured verifier gate.

## Domain Requirements

Helix should be broad, but each domain must justify what it needs from the core.

### Medicine

Needed:

- Confidence types.
- Knowledge provenance.
- Patient timeline types.
- Consent and privacy types.
- Clinical effect restrictions.
- Uncertainty-aware diagnosis.
- Medical dosage refinements.
- Audit evidence packs.

Demo gate:

- A dosage function rejects unsafe ranges and produces an audit trail explaining
  inputs, sources, confidence, and contraindications.

### Genomics and Biology

Needed:

- Sequence types.
- Provenance for samples and transformations.
- Consent-scoped data.
- Probabilistic variant interpretation.
- Causal experiment representation.
- Reproducible pipeline artifacts.

Demo gate:

- A genomic analysis pipeline can prove which sample, reference, tool version,
  and transformation produced a finding.

### Robotics and Autonomous Systems

Needed:

- Spatial frames.
- Sensor fusion.
- Confidence-aware perception.
- Temporal freshness.
- Real-time deadlines.
- Motor capabilities.
- Actor isolation.
- Safety zones as refinements.

Demo gate:

- A robot motion plan cannot compile or run unless frame, freshness, confidence,
  capability, and safety-zone obligations pass.

### Scientific Discovery

Needed:

- Units and dimensions.
- Causal vs correlational claims.
- Experiment types.
- Hypothesis locking.
- Provenance and reproducibility.
- Statistical uncertainty.
- Proof obligations for math-heavy transformations.

Demo gate:

- A scientific report generated from Helix includes data provenance,
  transformation code, uncertainty, assumptions, and reproducible build hashes.

### Climate, Energy, and Infrastructure

Needed:

- Spatial/geographic types.
- Temporal model histories.
- Sensor fusion.
- Confidence intervals.
- Scenario modeling.
- Energy/resource-aware computation.
- Long-term reproducibility.

Demo gate:

- A climate or grid simulation can report confidence bounds, source data,
  model versions, and resource costs.

### Education

Needed:

- Student knowledge-state model.
- Privacy/consent types.
- Temporal progress tracking.
- Confidence-aware tutoring.
- Active learning and curriculum primitives.

Demo gate:

- A tutor can explain what it thinks the student knows, why it believes that,
  and what evidence would update the belief.

## Certification and Trust Requirements

Helix should produce artifacts that people, institutions, auditors, and future
AI systems can inspect.

Every critical build should be able to emit:

- Source hash.
- Compiler hash.
- Runtime hash.
- Dependency hashes.
- Input data hashes.
- Build environment description.
- Target hardware assumptions.
- Proof obligations.
- Proof results.
- Test results.
- Confidence/provenance summaries.
- Safety case.
- Known limitations.
- Human approvals, if required.

Acceptance gate:

- A critical program can be rebuilt and compared at the binary level.
- The build can explain what changed since the last certified artifact.

## Phased Roadmap From Current Helix

Stage 30 and Stage 30.1 have proven that Helix can reach a stable self-host
fixed point for the bootstrap driver. The next work should move from "can Helix
stand up?" to "can Helix become a language of high-certainty intelligence?"

### Phase A: Write the Constitution

Goal:

- Finalize this blueprint into the project docs.
- Convert it into acceptance tests and stage gates.
- Decide which features are Layer 0, Layer 1, Layer 2, or tooling.

Exit gate:

- The roadmap maps every major feature to a testable compiler/runtime/tooling
  milestone.

### Phase B: Proof and Constraint Core

Stage-numbering note: this research blueprint predates the live roadmap. The
authoritative current stage labels are in `docs/ROADMAP.md`; current Stage 35 is
the AI/ML Capability Push in audit cleanup. The temporal/deadline/resource slice
listed below is still important, but its older stage label is obsolete.

Likely stages:

- Stage 31: refinement types and proof obligations.
- Stage 32: confidence types.
- Stage 33: expanded effects and capabilities.
- Stage 34: temporal/deadline/resource types.

Exit gate:

- Programs can fail compilation for unsafe value ranges, low confidence,
  missing capability, stale data, or missed deadlines.

### Phase C: Space, Time, Knowledge, and Memory

Likely stages:

- Spatial frames and units.
- Knowledge/evidence/belief/unknown types.
- Provenance-carrying values.
- Tiered memory.
- Continuous program runtime.
- Actor runtime.

Exit gate:

- A robot/world-model demo compiles with spatial, temporal, confidence, effect,
  and provenance checks.

### Phase D: AI-Native Stack

Likely stages:

- Shape-typed tensors.
- Device-typed tensors.
- Autodiff stabilization in self-hosted Helix.
- Model training provenance.
- Active learning primitives.
- Self-model primitives.

Exit gate:

- A small model can be trained, evaluated, provenance-tracked, and reproduced
  through Helix artifacts.

### Phase E: Multi-Target and Hardware

Likely stages:

- LLVM IR backend.
- GPU backend.
- MLIR/Triton-inspired kernel path.
- RISC-V backend.
- FPGA experiment.
- Hardware requirement declarations.

Exit gate:

- One source program compiles to at least two hardware targets with matching
  typed contracts and validated outputs.

### Phase F: Domain Demonstrators

Likely demos:

- Medical dosage safety demo.
- Robotics safety-zone demo.
- Scientific reproducibility demo.
- Genomics provenance demo.
- Climate uncertainty demo.
- Education knowledge-state demo.

Exit gate:

- Each demo proves a different reason Helix must exist.

### Phase G: Public Trust Stack

Likely work:

- Reproducible packages.
- Safety-case generator.
- Proof artifact viewer.
- Audit dashboard.
- Supply-chain checker.
- Long-horizon self-host cascade automation.

Exit gate:

- A third party can rebuild a Helix release, inspect proof/test artifacts, and
  confirm deterministic binary outputs.

## Specific Near-Term Work After This Blueprint

The next engineering moves should be small, testable, and aligned with the
larger mission.

1. Add a Stage 31 design appendix for refinement syntax and proof obligations.
2. Add parser support for simple `where` refinements.
3. Add typechecker representation for refined scalar types.
4. Add constant-only refinement proof checks before SMT integration.
5. Add compile-fail tests for invalid `Probability`, `Confidence`, and
   non-negative distance values.
6. Add proof obligation output artifacts.
7. Add an SMT bridge behind a feature flag or optional test path.
8. Move the first refinement examples into Helix source tests.
9. Keep the 10-generation self-host cascade as a release hardening gate.

## Open Research Questions

These must be answered carefully before the final system is trusted.

- What confidence calculus is sound enough for Helix core, and what should stay
  as domain-specific library policy?
- How should Helix represent dependence between evidence sources?
- Where is the decidability boundary for refinement predicates?
- Which proof obligations should be SMT-solved, which should be theorem-proved,
  and which should require runtime checks?
- What proof certificate format can survive long-term self-hosting?
- How should Helix verify an optimized backend without proving every backend
  from scratch?
- How should FFI preserve ownership, effects, confidence, and provenance across
  unsafe boundaries?
- What is the smallest continuous runtime that can support real-time systems
  without becoming too complex to verify?
- How should medical, genomic, and education consent rules be represented
  without hard-coding one jurisdiction's laws into the core language?
- How can Helix stay usable for humans while being structurally regular enough
  for AI systems to extend?

## Primary Failure Modes To Avoid

1. **False certainty**: confidence types that look rigorous but hide weak
   assumptions.
2. **Proof theater**: proofs that verify toy properties while real danger moves
   through FFI, runtime, or data.
3. **Overbuilt core**: too many features in Layer 0 before the compiler can
   support them cleanly.
4. **Python drift**: tooling becomes the real compiler authority after
   self-hosting.
5. **Backend unsoundness**: optimizations or hardware backends break semantics.
6. **Certification fantasy**: ignoring the evidence and process needed for
   medical, automotive, and infrastructure acceptance.
7. **Unusable language**: correctness features become so hard that nobody uses
   them.
8. **Unsafe self-improvement**: recursive changes bypass verifier gates.
9. **Provenance loss**: data pipelines silently erase evidence.
10. **Human-purpose drift**: the language becomes powerful without making
    benefit, consent, and accountability easier.

## Definition of Done for the Final Dream

Helix is not "done" when it has a parser, compiler, and standard library. Helix
is meaningfully complete when all of the following are true:

- The compiler is self-hosted and reproducible.
- The core language is documented by a stable spec.
- Critical subsets have machine-checkable semantics or translation validation.
- Refinement, confidence, effect, temporal, spatial, and provenance types work
  together.
- Continuous programs can run with deterministic memory and timing behavior.
- AI models can be trained, evaluated, and audited with provenance.
- Real-world action requires typed authority and confidence.
- Critical builds emit proof, test, provenance, and reproducibility artifacts.
- Self-improvement is verifier-gated and rollback-capable.
- Domain demonstrations prove usefulness in medicine, science, robotics,
  climate/infrastructure, education, and AGI research.
- A motivated third party can rebuild Helix and verify that the release binary
  matches.

## Final North Star

Helix should become a language where intelligence is not allowed to pretend it
knows, where action is not allowed to hide its authority, where memory is not
allowed to erase its origin, where optimization is not allowed to break truth,
and where self-improvement is not allowed to bypass verification.

That is the dream worth building toward: not just faster code, but a more honest
substrate for future human and machine civilization.
