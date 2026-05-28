# Category-2 next-phase plan (K2.X, 2026-05-27)

The macro saturation phase (K1.F22–F52, 30+ chunks) closed cleanly with
two consecutive 3-of-3 audit-clean signals (K3.Y, K3.Z). The remaining
gaps before Python-ready-to-delete (per `PYTHON_DELETION_BUCKETS` in
`scripts/helix_status.py`) are the 3 PARTIAL + 4 PENDING buckets. This
doc scopes them in tractable-first order so the dynamic-self-paced
loop can pick up the next-bucket-first ticks without per-tick scoping
overhead.

Per-bucket status as of 2026-05-27 (K2.X authored):
- 9 DONE buckets (macros, mixed-type int/float binops, f16, reflection,
  trace events, tile ops, field-store, const-name, ...)
- 3 PARTIAL: impl-method dispatch, generic monomorphization, K2 parity
- 4 PENDING: GPU backends, MLIR migration, K3 trusted seed, 5-clean gate

Overall: 66% (weighted) toward Python-deletion-readiness.

---

## Recommended execution order

### Phase 1 — Tractable PARTIALs (~25 chunks total)

These extend existing localized fixes and have well-understood shapes.
Highest velocity-per-chunk because the substrate is already in place.

#### P1.1 — K2 parity harness fully green (~5–10 chunks)
**Current**: 138/144 matrix-parity rows; macro-structural-gap recorded
(Python errors at `!`). The 6 remaining non-macro gap rows likely need
either small bootstrap-side fixes or matrix-row-honesty corrections.

**Approach**: read `docs/HELIX_K_BOOTSTRAP_FEATURE_MATRIX.md` for the 6
non-PARITY rows. For each:
- If bootstrap is missing a small feature → ship a K1.* chunk.
- If matrix is stale (feature actually works) → ship a K0.* matrix
  correction chunk.
- If feature is bootstrap-only (Python doesn't support) → mark
  `BOOTSTRAP-ONLY` in matrix; doesn't count against parity goal.

**Per-chunk**: 1 row at a time. Test via existing parity-harness if
appropriate; bootstrap-only self-host test otherwise.

#### P1.2 — Impl-method dispatch (full) (~10 chunks)
**Current** (post-K1.F5c 2026-05-27): K1.F5b shipped struct-VAR receiver
dispatch. K1.F5c extended to struct-LITERAL receiver via the
`last_struct_idx(sb)` side-channel. Remaining 9 gaps probed this tick:

**Gap probe results (K2.Z 2026-05-27, post-K1.F5c shipped 09bdc3e):**
- ✅ #5 Struct-literal receiver `P { x: 1 }.method()` — DONE (K1.F5c).
- ❌ #1+#2 Chained / non-let-bound receiver — `a.chain1().chain2()`
  returns rc=132 SIGILL. Substrate gap: prim after K1.F5b synthesis
  is AST_CALL (tag 16); the postfix-loop method-call PRE-CHECK at
  parser.hx:2416 only matches AST_VAR (1) and AST_TUPLE_LIT (50, K1.F5c).
  Fix needs a `last_call_ret_struct_idx` side-channel SET when the
  mangled-name resolves to a fn whose return type is a struct. That
  requires a fn-name → return-type lookup table (currently absent;
  fn-return tracking only happens implicitly for the body-vs-ret-ty
  trap K1.E1). Lift estimate: ~3 chunks (build fn-ret-tab, wire write
  at fn-decl time, wire read at K1.F5b synthesis).
- ❌ #6 `&self` receiver — `impl P { fn read(&self) -> i32 { ... } }`
  timed out (likely infinite-loop in parser or codegen). Substrate
  gap: the `&self` syntax probably parses (K1.CR lifetime + ref skip),
  but the method-call site doesn't take a reference of the receiver
  before passing as first arg. Lift estimate: ~2 chunks (parse, then
  ref-of-Self codegen which is a no-op since bootstrap doesn't have
  refs — pass-by-value).
- ❌ #7+#8 `Self::make() -> Self` static method + Self return — rc=139
  SIGSEGV. `P::make()` syntax not detected at the call site; `Self`
  return type not resolved. Lift: ~3 chunks.
- ⏳ #3 Trait dispatch — needs trait infrastructure (separate bucket).
- ⏳ #4 Generic-impl monomorphization — overlaps with P1.3 generic.
- ⏳ #9/#10 Default impls — needs trait infra (overlaps with #3).

**STRUCTURAL FINDING (K2.AA 2026-05-28 update):** the impl-block
self-receiver syntax (`impl P { fn read(self) ... }` and `&self`)
is BOOTSTRAP-ONLY territory — Python helixc raises `ParseError:
expected COLON (got RPAREN)` at the position right after `self`.
This parallels the macro / tile-op structural findings: the bootstrap
will be a feature-SUPERSET of Python for the self-receiver shape.
Practical implications:
  - K2 parity harness cannot test this shape (Python fails first).
  - The K1.F5h2-i chunks below close the bootstrap codegen gap as
    bootstrap-only self-host tests, NOT K2 parity probes.
  - Parity-testable impl methods continue to use the K1.F5b mangled-
    fn pattern (`fn P__read(p: P)`) — this is what
    test_impl_method_call_dispatch already uses and what works on
    both compilers.

**ADDITIONAL FINDING (K2.AB 2026-05-28): struct-by-value RETURN ABI is the
deeper blocker.** K1.F5g shipped the parse-side dispatch for chained
methods (AST_CALL prim_tag routing) but probe showed even single-call
struct returns fail: `let b: P = a.chain1(); b.v` returns rc=132. The
bootstrap's AST_CALL codegen at kovc.hx:8541 + the fn_type_table system
treats return values as i32 (in eax) by default. Struct returns are
8-byte pointers that need rax (full 64-bit).

Closing the chained-method path requires extending kovc.hx with:
1. A fn_ret_type_table parallel to fn_type_table that tracks each fn's
   return type tag (including 100+struct_idx encoding).
2. AST_CALL codegen reads the table; for struct returns, expects rax
   (REX.W) instead of eax. Already works because SysV ABI puts return
   in rax always — but kovc.hx may zero-extend/sign-extend from eax
   afterward, truncating the 64-bit struct-pointer.
3. The let-binding type-stamp for `let b: P = a.chain1()` must use
   var_struct_tab_add to register `b` as struct-typed.

Multi-chunk K1.F5g2+ arc; needs careful investigation of where exactly
the rax-vs-eax truncation happens. Probably looks like K1.E1's i64-bug
debug pattern: emit trap-id probes, capture stderr, isolate the bad
codegen site.

**Reordered P1.2 chunk plan (post-probe):**
1. K1.F5c ✅ — struct-literal receiver. SHIPPED 09bdc3e.
2. K1.F5d — fn-name→ret-ty table substrate.
3. K1.F5e — wire fn-ret-tab write at fn-decl time; populate from
   parse_fn_decl's return-type slot.
4. K1.F5f — wire fn-ret-tab read at K1.F5b synthesis; set
   last_call_ret_struct_idx if return is a struct.
5. K1.F5g — extend method-call PRE-CHECK to match AST_CALL prim_tag
   when last_call_ret_struct_idx >= 0. Closes #1 + #2 together.
6. K1.F5h — &self parse-side: allow `&self` as first impl-method param.
7. K1.F5i — &self codegen: pass-by-value since bootstrap doesn't have
   refs. Closes #6.
8. K1.F5j — `Self::method(args)` static call detection at parse_primary.
9. K1.F5k — `Self` type resolution in impl-method return position
   (substitute the impl's target struct name).
10. (Defer #3/#4/#9/#10 to bucket reorganization — these need trait
    infra which is its own multi-chunk arc.)

**Test pattern**: bootstrap-only self-host tests since several of
these shapes also push Python's frontend hard (likely K2-incompatible).

#### P1.3 — Generic monomorphization (full) (~10 chunks)
**Current**: K1.F21 generic-bare-call name-resolution fallback ships
the simplest case. Comprehensive monomorphization needs more.

**Probe findings (K2.AC 2026-05-28)**: items 4 and 5 already work
in bootstrap. Both are bootstrap-only superset features (Python
errors at parse-time on `fn f<A, B>` and `fn f<T: Trait>` — same
structural pattern as macros, tile ops, impl-block-self).

**Gaps to close**:
1. Turbofish `f::<T>(args)` already works (K1.F-discovery batch 27). [DONE]
2. Generic struct instantiation `let p = Pair::<i32>{a, b}` — currently
   uses K1.DJ turbofish + generic-param scalar marker. [DONE]
3. Generic-param-typed FIELD instantiation (Stage 28.11 INC-3a marks
   200+ but INC-3b's use-site monomorphization is incomplete). [PARTIAL]
4. Multi-param generics `fn f<A, B>(...)` — `pair<i32,i32>(42, 7)`
   returns 42 in bootstrap; Python ParseError. **BOOTSTRAP-ONLY DONE.**
5. Bounded generics `fn f<T: Trait>(x: T)` — `add<i32>(42, 0)` returns
   42 in bootstrap; Python ParseError. **BOOTSTRAP-ONLY DONE.**
6. Generic-fn calling generic-fn — `wrap<U>(x) calls id<T>(x)` returns
   42 in bootstrap; Python ParseError. **BOOTSTRAP-ONLY DONE.**
7. Where-clause monomorphization — K1.O parses, doesn't enforce. [PARTIAL]
8. Const-generic params `fn f<const N: i32>()` — bootstrap rc=132
   (SIGILL) on `arr<42>()`. Python ParseError. **PENDING** — bootstrap
   has separate codegen gap.
9. Lifetime-only generics `fn f<'a>(x: &'a T)` — currently parses as
   K1.CR/CS skip; bootstrap rc=132 per earlier probe. [PENDING]
10. Generic-impl monomorphization (overlaps with P1.2.4). [PENDING]

**Status (K2.AD update)**: **5 of 10** items now DONE (1, 2, 4, 5, 6);
2 PARTIAL (3, 7); 3 PENDING (8, 9, 10). The bucket is **MORE THAN
HALF CLOSED** — 5 items work in bootstrap as superset features that
Python's frontend rejects at parse, parallel to the macros / tile
ops / impl-block-self structural patterns.

The 5 remaining items are advanced shapes (const-gen, lifetime-only,
generic-impl, gp-field use-sites, where-clause enforcement). For a
v1.0 deletion-ready bootstrap, items 7-10 are likely not blocking
real-world usage. Item 3 (gp-field use-sites Stage 28.11 INC-3b) is
the highest-impact remaining sub-gap.

**Substrate**: extends `struct_tab` Stage 28.11 INC-3a marker model
(200+ markers for generic-params); INC-3b is the open use-site work.

---

### Phase 2 — Large PENDINGs (~100+ chunks total)

These are multi-chunk arcs with per-arc audits. Each starts with a
scope/feasibility chunk before any code lands.

#### P2.1 — MLIR migration in bootstrap (~30–50 chunks)
**Current**: helixc/ir/mlir/ Python files implement Stage 211-216 of
v3.0 (toolchain, mapping, helix_dialect, validate, emit, parity-gate).
Bootstrap needs to express the same IR + emission logic in Helix.

**Approach** (after MLIR Stage 211-216 chunks are reviewed):
1. Port helix_dialect.py (the small custom dialect) — simplest entry.
2. Port mapping.py (TileIR -> MLIR-dialect-op map).
3. Port emit.py (text emission of `module { func.func ... }`).
4. Port validate.py (MOCK-path validator + real-path validator).
5. Port toolchain.py (mlir-translate wrapper) — needs syscall plumbing.
6. Bootstrap-mock-validator parity vs Python-mock-validator.
7. Real-mlir-translate path (when binding-available).
8. Per-chunk 3-axis audit; per-stage 5-axis audit at 211/212/213/214/215.

**Substrate**: text-emission is straightforward; the dialect data
model needs careful design (Helix doesn't have Python's dict/list
ergonomics — will use arena-backed tables).

#### P2.2 — GPU backends in bootstrap (~40–60 chunks × 4 backends)
**Current**: Python helixc/backend/ has ptx.py / rocm.py / metal.py /
webgpu.py. Each is a tile-IR -> target-source emitter.

**Approach**: port one backend at a time, smallest first:
1. **PTX** (closest to architectural baseline) — start here.
   - Tile-IR -> NVPTX text emission.
   - Register allocator (already has Python prototype in
     helixc/backend/ptx_alloc.py).
   - Per-op emitters (load, store, arith, branch, sync, barrier).
   - PTX assembler wrapper or text-only emission for v1.0.
   - ~40 chunks expected.
2. **ROCm/HIP** — closely mirrors PTX. Reuse infrastructure.
   ~30-40 chunks.
3. **Metal MSL** — different surface (binding-table model).
   ~40-60 chunks.
4. **WebGPU WGSL** — also distinct (browser-friendly).
   ~40-60 chunks.

**Note**: at autonomy 5, we ship one backend, audit it, then move to
the next. Total realistic budget: ~150-220 chunks across all 4.

#### P2.3 — K3 trusted-seed bootstrap (~5–10 chunks)
**Current**: K-bootstrap binary is produced by Python compiler →
bootstrap. K3 needs a "trusted seed" — a binary checked into the repo
that compiles `kovc.hx` to itself without needing Python.

**Approach**:
1. Build `kovc.exe` from current `kovc.hx` via Python.
2. Stage that binary as `bootstrap/kovc_seed_v1.bin` (checksummed).
3. Use seed to compile `kovc.hx` and verify byte-identical output to
   the Python-built version (fixpoint check 1).
4. Use seed to compile `kovc.hx` → kovc' → kovc'' (fixpoint check 2;
   the N-generation DDC).
5. Document seed-rotation policy (regenerate when kovc.hx changes).

**Substrate**: needs DDC (diverse double-compiling) discipline. The
K5 milestone formalizes this; K3 is the prerequisite.

#### P2.4 — 5 consecutive clean END-OF-PHASE 5-axis audits
**Current**: 0 of 5. Not yet applicable (we're not at Python-ready-
to-delete state).

**Approach**: once all other buckets are done, run a 5-axis audit
(FE / IR / BE / RT / TEST). All 8 axes (5 phase + 3 holistic) must
return HIGH-confidence clean. Repeat 5 times across separate ticks.
Any HIGH or must-fix-MEDIUM resets to 0.

Per-axis subagents:
- FE: parser.hx + lexer.hx coherence
- IR: tile_ir / mlir_emit / lower
- BE: codegen (x86_64 + GPU + MLIR backends)
- RT: arena, trace, panic, exit
- TEST: test_codegen + test_k2_parity + test_self_host coverage

This is the stop-criterion gate. Once it passes 5 times consecutively
(and Python-ready-to-delete state is reached), the loop sends a final
Telegram and omits ScheduleWakeup, then halts.

---

## Per-tick discipline reminder

Each future tick:
1. Orient (`git status`, `git log -3`, counter check).
2. Pick the next chunk per this plan's order: P1.1 → P1.2 → P1.3 → P2.1
   → P2.2 → P2.3 → P2.4.
3. Within a phase, the smallest tractable chunk first; batch true
   mechanical siblings; otherwise one chunk per tick.
4. Test scope: pytest -k new + 1-2 closest siblings.
5. Audit: per-batch on family closure or per-10-chunks.
6. Commit + push + (Telegram if policy hits) + ScheduleWakeup.

When a new structural finding (like the macro/Python-! finding) surfaces
mid-bucket, capture it in semantic memory or this doc, then continue.

When all 16 buckets in `PYTHON_DELETION_BUCKETS` are DONE and the 5-clean
gate is satisfied, the loop sends the final Telegram and halts.
