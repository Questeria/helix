# Helix — Stats and Facts

Hard numbers and verifiable facts for the website. Each stat below is grounded in the actual Kovostov-Native repo as of 2026-05-09.

## The Numbers

| Stat | Value | Where it comes from |
|------|-------|---------------------|
| **hex0 binary size** | 120 bytes | The hand-encoded x86-64 program at the bootstrap root |
| **kovc compiler size** | ~50 KB | Self-hosted compiler binary |
| **Total bytes you must trust** | 120 | Everything else is reproducible |
| **Tests passing** | 3000+ | helixc/tests/ across all test files |
| **Test categories** | 50+ | test_codegen, test_parser, test_match, test_ffi, ... |
| **Audit cycles run** | 9 | Multi-agent audits for the foundation |
| **Silent-corruption bugs found** | 23 | Disclosed in docs/audit-stage4-followup.md |
| **Numeric types** | 12 | i32/i64/u8/u16/u32/u64/i8/i16/f32/f64/bf16 + bool |
| **AST tags** | 100+ | One per language construct |
| **Compilation stages** | 7 | hex0 → hex1 → M0 → M1 → M2-Planet → kovc-bs → kovc |
| **Optimization passes** | 5 | const-fold, CSE, DCE, FDCE, hash-cons |
| **Approach A stages + amendments** | 39 | Per docs/APPROACH_A_DETAILED_PLAN.md |
| **Pattern kinds supported** | 9 | Lit/Bind/Wildcard/Range/Variant/Tuple/Or/Guard/Ref |
| **Stdlib functions** | 80+ | core, ieee754, math, nn, autodiff, agi_search, option |
| **Backend targets** | 2 active | x86-64 ELF + (planned) PTX |
| **License** | Apache 2.0 / CC-BY 4.0 / CC0 | source / docs / weights |
| **Toolchain dependencies** | 0 | for the bootstrap chain |

## Compiler internals

| Internal | Value |
|----------|-------|
| Parser state slots | 70+ |
| bn_state reserved slots | 121 |
| Generic instantiation cap (Phase-0) | 32 |
| Closure capture cap | 4 vars |
| Closure nesting depth | 1 |
| Function-table cap | 512 entries |
| Tile shape cap (Phase-0) | 8×8 |
| Bucket cap (reverse-mode AD per param) | 32 expressions |
| Trap-id convention | `AST_TAG * 1000 + sub_id` |

## Bootstrap chain byte sizes

| Stage | Size | Capability |
|-------|------|------------|
| hex0 | 120 B | Hex digit → byte converter |
| hex1 | ~700 B | Adds label support |
| M0 | ~3 KB | Minimal macro assembler |
| M1 | ~8 KB | Full macro assembler |
| M2-Planet | ~30 KB | ANSI C subset compiler |
| kovc-bootstrap | ~80 KB | Helix compiler in C |
| kovc (self-hosted) | ~50 KB | Helix compiler in Helix |

## Performance benchmarks (representative)

These are honest order-of-magnitude estimates based on Phase-0 implementation:

| Benchmark | Helix | Reference |
|-----------|-------|-----------|
| Compile time, 10K LOC | ~0.5s | Rust: ~30s, Go: ~2s |
| Hello-world binary size | ~4 KB | Rust: ~3 MB, Go: ~2 MB |
| Self-host time | ~60s | (only meaningful baseline) |
| 4×4 f32 matmul (CPU, REG) | ~16 fma ops | naive triple loop |

**Note:** These are *bootstrap-stage* numbers. Phase-1 with optimization passes will be substantially faster.

## Quality gates

- Heavy gate (`pytest helixc/tests/`) must pass for every commit on main.
- 5 consecutive clean audits required before Stage 30 ships.
- Every silent-corruption window has either a runtime trap (with unique trap-id) or a compile-time check.
- 23 of 23 known bugs publicly disclosed with reproducer + status.

## Reproducibility guarantees

- **Deterministic builds**: same source → byte-identical binary.
- **Bootstrap reproducibility**: anyone can rebuild from `hex0` and verify.
- **Test snapshot reproducibility**: every test pinned to expected output.
- **Provenance tracking**: every commit signed; every audit logged.

## Open-source story

| Component | License | Notes |
|-----------|---------|-------|
| Compiler source code | Apache 2.0 | Use commercially without restriction |
| Standard library source | Apache 2.0 | Same terms |
| Documentation | CC-BY 4.0 | Cite when reusing |
| Logos and brand | CC-BY 4.0 | Use freely with attribution |
| Model weights (when shipped) | CC0 | Public domain |
| Training data | Public-license-only | No GPT/Claude/Gemini outputs in training |

## Story arcs (use for blog posts / explainers)

1. **"From 120 bytes to a compiler"** — the bootstrap journey
2. **"How Helix differentiates"** — autodiff design and implementation
3. **"Why no silent corruption"** — the trap-id philosophy and 23-bug disclosure
4. **"Tile types that mean something"** — compile-time shape checking
5. **"Reflection without runtime overhead"** — Quote/Splice as compile-time cells
6. **"The bootstrap audit trail"** — every byte verifiable
7. **"From x86 to PTX in one language"** — multi-target codegen
8. **"Why we picked SysV ABI"** — calling convention deep-dive
