# Stage 50 retry — root-cause diagnostic plan (2026-05-17)

Source: parallel exploration agent during Stage 52 gate-7 closure.

## Headline finding

The Stage 50 cascade-break failure mode is **SIGSEGV (rc=139),
NOT SIGILL (rc=132)** as the initial Stage 50 abort note implied.

The historical SIGILL bug (documented in
`docs/BOOTSTRAP_CASCADE_BUG.md`) was the read-buffer overflow at
the 256 KB boundary — fixed at Stage 50 prep by bumping BUF_SIZE
to 1 MB and adding a truncation sentinel.

The Stage 50 retry blocker is a different bug: G2 builds
successfully, byte-stable across G2..G11, but each generated
binary segfaults when invoked as a compiler.

## Top hypothesis (probability-ranked)

**H1 (highest)**: Stack overflow in the self-hosted binary at
runtime. The Helix codegen allocates a fixed 1024-byte stack
frame per fn invocation regardless of actual let-slot usage.
For deeply recursive fns (`propagate_adj`, `simplify`,
`parse_expr_basic`, `inline_user_calls`), a modest 1000-level
recursion burns 1 MB stack × N — trivially exhausting the
Linux default 8 MB stack.

**H2**: Uninitialized scratch region — the Stage 50 Inc 1 plan
added sb+88..sb+121 (32 new slots) but if any path in
`propagate_adj_multi` or `bucket_array_append` reads before
write, an arena-index of 0 corrupts the root AST nodes.

**H3**: n=1 bridge swap reads `bucket_array_sum(sb, 0)` with
zero-initialized multi-bucket head slots → writes to slot 0+1
= overwrites the first AST_FN node → segfault on next deref.

**H4 (least likely)**: SIGILL recurrence — the combined
bootstrap source has exceeded the 1 MB buffer. Not consistent
with the rc=139 evidence (would be rc=132), but worth ruling
out first since it's a 5-minute check.

## Cheapest bisect experiments

**Exp A (5 min)**: After `source = bootstrap_source(...)` in
`scripts/selfhost_cascade.py`, add
`print(f"source bytes: {len(source.encode())}")`. If under 1 MB,
rule out H4.

**Exp B (15 min)**: In WSL, `ulimit -s unlimited` then invoke
the G2 binary manually. If it stops segfaulting, **H1 confirmed**.
Cheapest permanent fix: add a `sys_prlimit64` call at binary
startup to lift the stack limit.

**Exp C (30 min)**: Add `ud2 / exit 99` assertions after every
`bucket_array_sum` and `param_idx_of` call that the returned
index is >0 before dereferencing. If this surfaces a trap exit
instead of segfault → exact callsite found.

**Exp D**: `objdump -d g2.bin | grep -n ud2` — if a ud2
instruction appears immediately before a fn epilogue without a
preceding jump-over, original cascade bug pattern recurred.

## Path forward

1. Run Exp A → almost certainly rules out H4.
2. Run Exp B → if confirmed H1, the permanent fix is per-fn
   stack-frame sizing (codegen change at `kovc.hx::emit_prologue`
   to compute actual let-slot needs instead of fixed 0x400).
3. If H1 not confirmed by Exp B, escalate to Exp C+D to nail
   the arena-corruption hypothesis.

## Files essential to investigation

- `docs/BOOTSTRAP_CASCADE_BUG.md` — historical SIGILL root-cause
- `docs/stage50-plan-2026-05-17.md` — Stage 50 Inc 1/2 slot
  layout + helper contracts
- `scripts/selfhost_cascade.py` — driver (lines 58-82 = seed
  source assembly, 124-158 = generation invocation + exit codes)
- `helixc/backend/x86_64.py:3220-3361` — Python-side
  `read_file_to_arena` with BUF_SIZE = 0x100000
- `helixc/bootstrap/kovc.hx:3128-3236` — Helix-side
  `emit_read_file_to_arena_body` (the 4 emit_u32_le(1048576)
  must stay in lock-step with backend)
- `helixc/bootstrap/parser.hx`:
  - 4100-4224: scratch-block init (sb slot allocations)
  - 5406-5574: bucket/propagate_adj/sum_bucket/diff_reverse_one
  - 5973-6120: grad_rev_pass caller loop
- `helixc/tests/_tmp/selfhost_cascade.out` — the actual logged
  failure (rc=139 SIGSEGV, G2 byte-stable at 277899 bytes)

## When to revisit

Stage 50 retry should run Exp A + Exp B first. If H1 confirmed,
the per-fn stack-frame sizing change is a self-contained
optimization that unblocks Stage 50 + reduces seed-binary
memory footprint generally. If H1 not confirmed, switch to
Exp C to localize the arena-corruption hypothesis.

Estimated time: Exp A+B together ≤30 min. If H1 confirmed,
codegen fix ≤2h. If H1 not confirmed, Exp C+D another ~2h to
localize then patch.

## Lineage

- 2026-05-17: Stage 50 Inc 1+2 landed (commits f4e94fc, 76b7735),
  gate-1 audit caught cascade-break, Inc 1+2 reverted (commit
  f678aa3), Stage 50 ABORTED.
- 2026-05-17 (evening): parallel exploration during Stage 52
  gate-7 closure produced this diagnostic plan. Stage 50 retry
  unblocks on Exp A+B execution.
