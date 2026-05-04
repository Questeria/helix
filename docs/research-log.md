# Kovostov-Native Research Log

Daily / per-session notes. Append-only. Every meaningful learning, dead end, or pivot.

---

## 2026-05-03 — Project initialization

**Session 1.** Project created at `C:/Projects/Kovostov-Native/`.

### Hard constraints established (user-set)
1. Raw binary as bootstrap start (no assembler, no compiler dependency for shipped artifacts).
2. End goal: open-source AGI named Kovostov.
3. Public training data only.

### Decisions taken (see `decisions/2026-05-03-go.md`)
- Project name: Kovostov-Native (sibling to `C:/Projects/Kovostov`, the Claude-shell framework that has been the cognitive scaffold during planning)
- AI / system name: **Kovostov**
- Language name: **Kov** (`.kov` extension)
- Compiler: **kovc**
- Bootstrap target: linux-x86_64 ELF (via WSL2 on Windows; ELF dramatically simpler than PE)
- Final runtime target for AI: Windows-native (CUDA Driver API → RTX 5090)
- License: Apache 2.0 (code) / CC-BY 4.0 (docs) / CC0 (weights)

### Deep-research findings absorbed
Five parallel agents earlier produced reports on:
1. AI-language survey — closest spiritual ancestor: Dex; secondary: Triton, Tinygrad, Futhark
2. No-LLVM backend feasibility — yes, PTX text emission is feasible; QBE-style typed SSA with block parameters is the recommended IR; ~12–24 months for x86 + PTX backend with AI assistance
3. Type system + autodiff — Futhark size types + opt-in refinements; AD as compiler pass after primary opt (Enzyme rule)
4. AI compiler optimizations — two-level IR (Tensor IR + Tile IR); top-5 must-haves: fusion, tile codegen, autotune, layout selection, memory planning
5. 2024–2026 SoTA — tile is THE primitive; element type is `(format, block, scale)` triple; AD has retreated from compiler-pass to library; design FOR AI-author from day one

These findings are crystallized in `PLAN.md`. Original agent reports archived in `docs/research/` (to be added).

### Phase 0 starting point
The first artifact will be **the hex0 seed monitor**: a tiny program (~150–250 bytes of hand-encoded x86-64 ELF) that reads hex characters from stdin, skips whitespace, pairs digits, and writes bytes to stdout. Every byte will be annotated.

Once hex0 works, every subsequent stage feeds higher-level text into the previous stage's binary, producing the next stage's binary. The chain ends with a self-hosted kovc.

### Open questions (this session)
- Should we adopt the existing M2-Planet seed verbatim (proven, audited, ~12 months saved) or write our own from scratch (purer, +months of effort)? Working assumption: write our own hex0 from scratch; consider adopting M2-Planet's later stages (M0/M1) since the marginal purity gain shrinks at each stage.
- Should the hex0 seed target Linux ELF or Windows PE? Working assumption: Linux ELF (via WSL2). PE is achievable but costs ~3× the bytes for the header alone.
- License for trained weights: CC0 vs Apache 2.0 vs OpenRAIL? Working assumption: CC0 for maximum freedom.

### Next actions (this session, continuing now)
1. Write Apache 2.0 LICENSE
2. `git init` + first commit
3. Spawn parallel research agents for: (a) ELF64 minimal-header byte layout, (b) Linux x86_64 syscall numbers and calling convention, (c) the M2-Planet bootstrap chain study
4. Begin hex0 design — annotated bytes in `stage0/hex0.annotated.md` and the binary file itself
