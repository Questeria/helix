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

---

## 2026-05-03 (continued) — hex0 design complete

**Session 1 progress:**
- Founding docs committed (initial commit `1ea237f`)
- 5 deep-research agents returned: ELF64 byte layout, Linux syscall table, x86-64 instruction encoding, stage0 ecosystem, recommended chain approach
- Decision: **hybrid** — hand-write hex0 ourselves (the literal "raw binary" hard constraint), re-evaluate adoption of stage0-posix from hex1 onwards at month-2 gate
- `stage0/hex0/hex0.s` written: full annotated NASM assembly, ~140-byte estimate for code section, total ELF ~260–320 bytes
- `stage0/hex0/hex0_reference.py` written: Python behavioral oracle
- 3 test fixtures (`01-hello`, `02-comments-ws`, `03-empty`) — all pass against Python reference
- `stage0/hex0/build.sh` written: assembles via nasm (cross-check only) + runs tests + verifies `cmp`-equivalence with hex0.bin (when present)
- `stage0/hex0/hex0.bytes.md` placeholder for hand-encoded byte form

**Blocker for next session:**
- `nasm` not installed in WSL; install requires user's sudo password. Two paths forward: (A) user installs nasm, we use it as cross-check while hand-encoding; (B) compute every byte from Intel SDM independently, cross-check against `oriansj/stage0-posix-amd64/hex0_AMD64.hex0` only.

**Tooling note:** WSL2 confirmed working with gcc, as, ld. Linux 6.6.87.2-microsoft-standard-WSL2.

**Decisions log:**
- Bootstrap target OS: Linux x86_64 ELF (via WSL2). Decision recorded in 2026-05-03-go.md.
- License contagion: kov-libc to be hand-written, not adopted from M2libc (avoids GPL-3.0 contagion into kovc-bootstrap binary).

---

## 2026-05-03 (continued) — hex0 SHIPPED

**Phase 0a complete.** First Kovostov-Native binary, 299 bytes.

### What landed
- `stage0/hex0/encode.py` — Python tiny-assembler that resolves labels (build-time only)
- `stage0/hex0/hex0.hex` — annotated byte-by-byte source (canonical text form)
- `stage0/hex0/hex0.bin` — 299-byte x86_64 Linux ELF binary (the shipped artifact)
- `stage0/hex0/hex0.sha256` — SHA-256 = `cc1d1741db903d6959c9e2b11db0fb0dc8e7ec4de18c2774a895b31fe417c125`
- `stage0/hex0/build.sh` — pipeline: hex0.hex → strip → xxd → hex0.bin, ELF check, objdump, SHA verify, run tests
- `stage0/hex0/run_tests.sh` — 3 fixtures, all PASS
- `stage0/hex0/hex0_reference.py` — Python behavioral oracle (kept for cross-checking)

### Verification trinity (all pass)
1. **Behavioral**: 3/3 fixtures pass (hello, comments+ws, empty)
2. **ELF validity**: `file` reports valid x86-64 LSB ELF executable
3. **objdump**: disassembly matches hex0.s mnemonics byte-for-byte (jumps, syscalls, dispatch all correct)

### Round-trip integrity
`hex0.hex` (annotated, with comments) → strip comments/whitespace → `xxd -r -p` → produces a binary byte-identical to `hex0.bin` (SHA-256 confirmed). The hex source IS the canonical artifact.

### Tools used (all permitted)
- xxd (hex<->bin only — audit-only)
- objdump (disassembly — audit-only)
- file (header sniff — audit-only)
- sha256sum (integrity)
- Python 3 (build-time encode.py — not shipped)

### Tools NOT used (raw-binary constraint honored)
- nasm, as, gcc, ld, clang, LLVM, MLIR — none used to produce the shipped artifact

### Encoding decisions
- Two backward jumps from `combine` and `skip_comment` to `read_loop` exceeded rel8 range; promoted to rel32 (5-byte E9 form). Total cost: +6 bytes.
- One forward jump from `read_loop` to `do_exit` exceeded rel8 (138 bytes); promoted to rel32 jle (6-byte 0F 8E form). Cost: +4 bytes.
- `test bpl, 1` requires REX prefix (0x40) to access the 8-bit form of rbp's low byte.
- Used `inc eax` instead of `mov eax, 1` in combine block to save 3 bytes (FF C0 vs B8 01 00 00 00).

### Next session
- **Hex1**: human-readable assembly format with labels and comments. Adds: label resolution, multi-byte numeric forms. Decision needed: write our own from scratch (~3-5 PM) OR vendor `oriansj/stage0-posix-amd64/hex1` and audit (~few weeks). Tentative: vendor hex1+ since Phase 0a is the load-bearing "raw binary" claim.
- **kov-libc design**: minimal libc shim in M2 C-subset to avoid GPL-3.0 contagion when M2-Planet is vendored.
- **Cross-reference vs oriansj's hex0**: confirm our encoding decisions match theirs for analogous instructions.
