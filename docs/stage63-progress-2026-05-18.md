# Stage 63 Progress — 2026-05-18

## Stage Goal

Stage 63 ships **Tier 3 #11 — runtime trace wiring**. The
Python-side trace introspection API was shipped at Stage 59
(`trace_hash`, `trace_size`, `trace_op_counts`, etc.), and the
IR-level TRACE_ENTRY / TRACE_EXIT ops were emitted at Stage 25
(at `@trace` fn prologue/epilogue). What was MISSING: real runtime
recording — the x86_64 backend emitted single-NOP stubs for both
ops, so @trace fns ran without observable side effects.

This stage wires the real runtime recording inline (no external C
runtime library), keeping the Phase-0 fully-self-contained ethos.

## Deliverable

**Inline trace event recording** in the x86_64 binary's BSS:

- New BSS symbols:
  - `__helix_trace_count` (4 bytes, i32 cursor)
  - `__helix_trace_buf` (HELIX_TRACE_CAP × 8 bytes; 8 KB default)
- Each trace event is 8 bytes: 4 bytes `fn_id` + 4 bytes `kind`
  (0 = entry, 1 = exit).
- TRACE_ENTRY / TRACE_EXIT now emit inline assembly (~50 bytes each)
  that appends an event to the buffer when count < HELIX_TRACE_CAP.
- Fail-closed: when the buffer is full, subsequent events are
  silently dropped (deterministic; no allocation, no syscall, no
  wrapping that would lose old context).
- New builtin: `__trace_event_count() -> i32` reads the cursor for
  test introspection.

## Increment breakdown

**Inc 1 (single commit, ~150 LOC)** — establishes the surface
end-to-end. Future Incs may extend:
- Inc 2: bootstrap-side port (kovc.hx emits same asm pattern)
- Inc 3: arena-side trace dump builtin (`__trace_dump_to_arena()`
  for consumption by Python via the existing
  `trace_from_canonical_json` round-trip)
- Inc 4: lift the 1024-event cap to a build-time constant

### Inc 1 deliverables

1. `helixc/backend/x86_64.py`:
   - New constant `HELIX_TRACE_CAP = 1024`
   - 2 new BSS symbols (`__helix_trace_count`, `__helix_trace_buf`)
     defined alongside `__helix_arena_base`
   - `_intern_trace_fn_id(fn_name)` — assigns stable i32 per fn
   - `_emit_trace_event(fn_id, kind)` — emits ~50-byte inline asm
     sequence (load cursor, compare, jge skip, store event, inc
     cursor, store back)
   - TRACE_ENTRY / TRACE_EXIT branches now call `_emit_trace_event`
     instead of emitting NOP
   - New `trace_event_count` PRINT-op _kind variant that loads
     cursor for tests
2. `helixc/ir/lower_ast.py`: intercept `__trace_event_count()`
   builtin call → PRINT op with `_kind=trace_event_count`
3. `helixc/frontend/typecheck.py`: add `__trace_event_count` to
   builtin whitelist

## Test coverage

3 new end-to-end tests:
- `test_stage63_inc1_trace_event_count_zero_when_no_trace_fn` —
  baseline: no @trace calls → counter stays at 0 (BSS-zeroed)
- `test_stage63_inc1_trace_event_count_increments_on_traced_calls`
  — 3 calls to @trace fn → counter = 6 (3 × {entry, exit})
- `test_stage63_inc1_trace_buffer_caps_at_HELIX_TRACE_CAP` — 600
  calls (1200 events) → counter caps at 1024

## Closure narrative

**3-clean-gate**:

- Gate A (silent-failure): the inline trace asm is fully isolated
  (writes only to BSS-allocated `__helix_trace_*` symbols; no
  register state preserved beyond the local sequence). End-to-end
  tests verify exact event counts under 3 scenarios (no calls,
  steady state, overflow). Cannot silent-miscompile because the
  counter is observable.
- Gate B (type-design): the new builtin `__trace_event_count()`
  has signature `() → i32`, matching the existing
  `__arena_len()` pattern. No new type-design surface.
- Gate C (code-review): the asm sequence mirrors the existing
  `__helix_arena_base` BSS allocation + RIP-relative-load pattern
  used by the arena ops. No new infrastructure.

**Cascade defects in this stage**: 1 (test exit-code wrap-modulo-256
— sentinel-value-comparison fix inline before commit).

**Test counts at closure**:
- test_strings_io.py: 20/20 (3 new Stage 63 tests; previously 17)
- self-host gate: 223/223 GREEN

## Tier 3 #11 status after Stage 63

**Tier 3 #11 (Trace-based introspection) substantially complete**:
- Python-side trace API ✅ Stage 59 (trace_hash, trace_size,
  trace_op_counts, trace_fn_counts, trace_is_balanced,
  trace_equiv_modulo, trace_to_canonical_json,
  trace_from_canonical_json)
- IR-level TRACE_ENTRY / TRACE_EXIT op emission at @trace fn
  prologue/epilogue ✅ Stage 25
- **Runtime recording (inline x86_64 asm) ✅ Stage 63 (this)**
- Bootstrap port + arena-side dump builtin (Inc 2-4) → deferred
  to future incremental polish stages (~1 week each)

The 80% case ships today: `@trace` fns now genuinely record
entry/exit events at runtime; tests can verify exact counts via
`__trace_event_count()`.

## Next stage

**Stage 64 opens immediately**: Tier 2 #6 — tensor codegen bf16
+ perf passes. Per multi-week-scope agent's Inc 1 recommendation:
"drop bf16 from the HBM-dtype rejection set in
`_require_supported_hbm_dtype` (ptx.py:254-259) and add a `%h`
register pool so bf16 tile params can round-trip."

Stage 64 is multi-week (4-6 weeks per planning agent estimate)
but each Inc is small and self-contained. Stage 65 (multiple
dispatch) follows. Stage 66 (borrow checker) is the first
STOP-FOR-USER gate.
