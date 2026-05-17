# Stage 44 Progress - 2026-05-17

## Stage Goal

Stage 44 is **Stack-passed overflow float args** per ROADMAP
Tier 1 #5 — the smallest of the Tier-1 ML blockers. SysV
x86-64 ABI: the first 8 float args go in `xmm0..xmm7`; the 9th
and later go on the caller's stack. Pre-Stage-44, the backend
raised `NotImplementedError("v0.1 supports up to 8 float args
via xmm0..xmm7")` on the 9th float arg, blocking real ML code
(hit during XOR perceptron dogfooding per ROADMAP note).

Beginner meaning: neural networks have lots of parameters.
Many gradient-descent kernels take a dozen+ floats. The
compiler used to refuse to compile any function with 9 or more
float arguments, so real ML kernels couldn't even build.
Stage 44 makes the compiler do what the SysV ABI actually says
to do: put the overflow args on the stack.

## Increment 0 - Open Stage 44

Same conventions as Stage 35-43: 3-clean-gate closure, self-
host gate green before every commit, Phase-0 Python-side
implementation.

## Increment 1 - Caller-side stack-arg shuffle

`helixc/backend/x86_64.py` CALL arm:
- Pre-pass count of overflow float args
  (`max(0, float_count - 8)`).
- 16-byte-aligned stack allocation
  (`((overflow * 8 + 15) // 16) * 16`).
- `sub rsp, stack_alloc` before the reg-arg shuffle.
- Bit-blit each overflow arg from `[rbp+arg_slot]` to
  `[rsp + 8*overflow_idx]` via `rax` (f64, 8 bytes) or `eax`
  (f32, 4 bytes — avoids leaking adjacent stack bytes).
- Reg-arg shuffle continues unchanged for the first 8 floats
  (xmm0..xmm7) and first 6 ints (rdi..r9).
- After CALL, `add rsp, stack_alloc` restores the frame.

2 new emit helpers in `Assembler`:
- `mov_mem_rsp_rax(disp)` — 8-byte store to `[rsp+disp]`.
- `mov_mem_rsp_eax(disp)` — 4-byte store to `[rsp+disp]`.

## Increment 2 - Callee-side prologue load

`helixc/backend/x86_64.py` function prologue:
- When `xmm_idx >= 8`, load each overflow float param from
  `[rbp + 16 + 8*stack_param_idx]` (above saved rbp + return
  address) into `xmm0` (scratch), then store to the local
  frame slot via the existing `_movss/_movsd_store_xmmN`.
- `stack_param_idx` advances independently of `xmm_idx`.
- Reg-pass for the first 8 floats unchanged.

## Increment 3 - Tests + regression coverage

`helixc/tests/test_stage44_stack_overflow_args.py` — 7 tests:

1. `sum9` — 9 f32 args, sum = 45 (one overflow).
2. `sum10` — 2 overflow (sum = 55).
3. `sum12` — 4 overflow (sum = 78).
4. `check` — overflow preserves register args (8-on-regs minus
   1-on-stack = 7).
5. `sum9_f64` — f64 path (8-byte stack slots).
6. Distinct-arg-values sanity (10/20/.../90 = 450 mod 256 =
   194).
7. Position pin — 9th arg in isolation (8 zeros + 42 = 42),
   catches any indexing mistake in the overflow path.

All 7 green on the first end-to-end probe.

## Increment 4 - Stage 44 Closure

### Gate 1 (post-Inc-3) — 5 silent-failure + 2 type-design fixes

Gate-1 audits returned across 3 lanes:
- silent-failure (5 findings): F1 HIGH FFI_CALL parity, F2
  HIGH mixed int+float guard, F3 MEDIUM store-loop asserts,
  F4 MEDIUM alignment tripwire, F5 LOW sub-byte float doc.
  All actionable fixed.
- type-design (2 MEDIUM + 2 LOW): module-level SYSV_*
  constants extracted; xmm_idx dual meaning documented;
  cosmetic LOW deferred.
- code-review: GATE CLEAN (1 LOW dead-store stylistic
  below threshold).

Gate-1 fix sweep (commit 2da92b3):
- FFI_CALL now mirrors CALL for 9+ float args (parity).
- Mixed int+float overflow rejects BEFORE any sub_rsp
  (front-load guard).
- `assert _overflow_idx == overflow_float_count` tripwire
  after store loop.
- `assert stack_alloc % SYSV_STACK_ALIGNMENT == 0`
  tripwire before sub_rsp.
- Module-level `SYSV_STACK_ARG_BASE` / `STRIDE` /
  `ALIGNMENT` named constants replace the hard-coded
  16/8 magic numbers across all 3 sites (caller CALL,
  caller FFI_CALL, callee prologue).
- 3 new backfill tests pinning FFI_CALL 9-float compile,
  mixed-overflow clean-rejection, and SYSV_* constant
  public API.

10 Stage 44 tests now green (was 7 pre-gate-1).

### Gate 2 (post-gate-1) — ALL 3 LANES CLEAN

- silent-failure: GATE 2 CLEAN
- type-design: GATE 2 CLEAN
- code-review: GATE 2 CLEAN

### Gate 3 (post-gate-2) — ALL 3 LANES CLEAN

- silent-failure: GATE 3 CLEAN (3 LOW doc-polish items
  noted below threshold)
- type-design: GATE 3 CLEAN
- code-review: GATE 3 CLEAN

### STAGE 44 CLOSED 2026-05-17 at Inc 4 (3/3 clean audit gates)

3 consecutive clean audit gates achieved. ROADMAP Tier 1 #5
(stack-passed overflow float args) shipped end-to-end:
- 9th+ float args correctly pass via the SysV stack convention
  for both internal CALL and FFI_CALL.
- f32 (4-byte) and f64 (8-byte) payloads both correct.
- Caller-side stack allocation 16-aligned and tripwire-asserted.
- Callee-side prologue loads from [rbp + SYSV_STACK_ARG_BASE
  + idx * SYSV_STACK_ARG_STRIDE] correctly.
- Mixed int+float overflow rejects cleanly before any rsp
  mutation.

10 Stage 44 tests + 627+ tests across the codegen surface all
green. Self-host cascade still byte-identical G2..G4 fixpoint.

### Known limitations / future work

- **Int overflow** (>6 int args) still raises
  `NotImplementedError`. The infrastructure here (pre-pass
  count, aligned stack allocation, post-call restore) is the
  same shape — a future stage can extend it for int args
  symmetrically. Out of scope for Stage 44; few existing
  signatures need it.
- **Mixed int+float overflow** — the current pre-pass counts
  only float overflow. If a future signature needs both, the
  allocation math + indexing need a joint pass. Defer until
  hit.
- **xmm8..xmm15** are unused in SysV arg-passing; not a
  Stage-44 concern, but available scratch regs for future
  passes.
- **Gate-3 LOW polish items** (deferred):
  - FFI_CALL accounting assert lacks descriptive message
    (CALL twin has one).
  - FFI_CALL skip-arm dual-meaning comment missing (CALL
    twin has one).
  - Callee prologue xmm0 scratch register undocumented
    invariant.

Stage 45 opens next per ROADMAP Phase 2.
