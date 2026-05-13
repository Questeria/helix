# Helix Purpose

**Status**: living purpose statement.
**Date**: 2026-05-13.

Helix is not an internal language for Kovostov only. Kovostov is the first
flagship system built on Helix; Helix itself is meant to become a dominant open
language for AGI development and for any domain where auditable, reproducible,
high-certainty computation can improve human life.

## Mission

Helix exists to reduce uncertainty as much as software honestly can.

That means making uncertainty, evidence, provenance, assumptions, resource
limits, proof obligations, effects, and authority visible to the compiler and to
the human or AI systems using the compiler.

The long-term goal is for Helix to support:

- AGI and AI research
- scientific discovery
- medicine and clinical decision support
- genome research and synthetic biology
- physics and simulation
- mathematics and theorem-guided computation
- robotics and real-world actuation
- climate, energy, and infrastructure systems
- secure public-interest software
- education and tools that help humans understand complex systems

## Product North Star

Helix should be the language people reach for when silent uncertainty is
unacceptable.

It should make it natural to ask:

- What do we know?
- How do we know it?
- What is uncertain?
- What assumptions are being made?
- What can be proven at compile time?
- What must be checked at runtime?
- What authority does this program have to act?
- What evidence trail will remain after it acts?

## Design Consequences

- **Kovostov is the first flagship user, not the boundary.**
- **Helix core behavior should become Helix-derived after self-hosting.**
- **Python may remain as tooling and reference scaffolding, not the long-term source of truth.**
- **Anything that must fail to compile belongs in the compiler/type system.**
- **Anything that must merely work correctly at runtime can be a Helix library.**
- **Every major feature should justify which human or AGI capability it improves.**

## Dream-Big Feature Directions

- Typed uncertainty: confidence, intervals, probability, evidence quality, and out-of-distribution markers.
- Proof-carrying programs: machine-checkable certificates for critical claims.
- Causal programming: interventions, counterfactuals, assumptions, and measurement provenance.
- Scientific units and dimensions: physics/math errors caught before runtime.
- Medical and genomic safety types: consent, lineage, privacy, and clinical uncertainty as first-class constraints.
- Knowledge provenance: every belief can trace source, time, confidence, and transformation history.
- Safe self-modification: reflection and code updates gated by verifier functions and rollback plans.
- Reproducible computation: source, binary, data, model, and proof artifacts tied together by hashes.
- Resource-aware intelligence: compute, energy, bandwidth, memory, and time budgets visible in types/effects.
- Human-benefit gates: powerful actions require explicit alignment with declared human-purpose constraints.

Helix should become a language for building systems that can learn, reason,
prove, act, explain, and improve without hiding the uncertainty that remains.
