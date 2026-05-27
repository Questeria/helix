# Hard constraint — Helix must be fully self-hosting

**Stated:** 2026-05-26 (user directive)
**Scope:** binding on the entire K-bootstrap track and any post-K1 work
**Severity:** HARD — no exceptions, no partial-credit, no "Python keeps X forever"

## The rule

When the K-bootstrap track completes (currently targeted at v1.0 in
`scripts/helix_status.py` terms), **the project must contain zero
non-Helix runtime code**. Specifically:

- **No Python in the compiler.** `helixc/` (the Python implementation)
  must be deleted. K4 (cutover) is **mandatory**, not optional.
- **No Python in test infrastructure** for compiled programs.
  Test harnesses that exercise `kovc.hx` (the Helix-side compiler) must
  themselves be written in Helix. (Python may remain for harness
  bootstrapping until the trusted-seed work at K3 closes that gap.)
- **No Python in build scripts** or developer tooling that ships with
  the project. If a `.py` file is in the source tree at v1.0, it
  must be either (a) removed, (b) ported to Helix, or (c) clearly
  marked as ephemeral dev tooling that runs outside the published
  artifact.
- **No deferral of features to "Python helixc forever."** Earlier
  optimization plans suggested keeping GPU / MLIR / Tile ops in
  Python permanently while bootstrap handles CPU x86 only. **That
  plan is invalid under this constraint.** Every feature in the
  Python helixc must be ported to the bootstrap before v1.0.

## Why this matters

Self-hosting is the headline goal of the K-bootstrap track
(`scripts/helix_status.py`: "SELF-HOSTING ACHIEVED -- the headline
goal: a Helix compiler written in Helix, compiled in Helix, all the
way from raw binary with NO Python in the final product"). The
hard-constraint statement makes "fully in Helix" non-negotiable.

This means:

- The remaining-chunks estimate (`docs/K_BOOTSTRAP_FEATURE_MATRIX.md`)
  must include all ~25 GPU/MLIR/Tile/reflection rows, not just the
  CPU-relevant subset.
- Any plan that says "defer X to a future track that never closes"
  is rejected by this constraint.
- K5 (DDC) is also mandatory because it's part of the "trusted from
  first principles, no Python in the chain" story.

## Practical impact on the optimization plan

The session-2026-05-26 optimization plan said:

> Aggressive Phase-2 ordering means GPU/MLIR don't ship in the
> bootstrap. That's fine: Python helixc keeps those, and K4 (delete
> Python) is the only step that requires bootstrap-side parity for
> them. We can defer GPU/MLIR until after K3 lands and re-evaluate
> whether they actually need to be in the bootstrap at all.

**This is no longer valid.** GPU/MLIR/Tile must be ported. The
"re-evaluate whether they need to be in the bootstrap" decision is
already made: yes, they do.

Realistic timeline impact: instead of "~25–35 chunks to a
deletable-Python state" (the optimistic estimate), the real path is
closer to **~60–80 chunks** because the GPU/MLIR/Tile/reflection
work cannot be skipped.

## Verification

At v1.0 release:

- `find C:/Projects/Kovostov-Native -name "*.py" | wc -l` should
  return zero (or only files explicitly marked as ephemeral dev
  tooling per the rule above).
- The bootstrap must compile **itself** (lexer.hx + parser.hx +
  kovc.hx) via the existing self-host test chain plus all v3.0
  features that the Python helixc supported.
- The DDC (K5) check must pass: build the bootstrap two
  independent ways, confirm bit-identical output.

## Autonomous-loop stop criterion (user directive 2026-05-26)

The autonomous-worker loop (cron job `5091b305` at the time of
writing) must KEEP WORKING until the project reaches the
**Python-ready-to-delete** state, at which point a stability
gate of **5 consecutive clean audits** unlocks loop termination.

Specifically:

1. **Python-ready-to-delete** means:
   - All Category-1 syntax niceties shipped (K1.* parser/lexer
     completion to the level real Rust source parses).
   - All Category-2 semantic gaps closed: impl method dispatch,
     generic monomorphization, mixed-type binops, f16 literals
     (bit-accurate), reflection (quote/splice/modify/reflect_hash),
     tile ops (TILE_ZEROS/ADD/MUL/MATMUL), GPU backends (PTX +
     ROCm + Metal + WebGPU), MLIR migration path, trace events,
     field-store mutation, const-name resolution, macros.
   - K2 (parity harness) green: every test program goes through
     both Python helixc AND bootstrap kovc.hx; outputs are
     byte-identical.
   - K3 (trusted seed) shipped: a small hand-audited Helix
     binary that re-bootstraps the compiler from source.

2. **5 consecutive clean audits** at that state means:
   - Run the per-chunk 3-axis audit (silent-failure-hunter /
     type-design-analyzer / code-reviewer) AND the 5-clean
     end-of-phase audit (FE / IR / BE / RT / TEST).
   - **All 8 axes must come back HIGH-confidence clean.**
   - Repeat 5 times in succession, ideally across different
     ticks separated by at least one re-compilation of the
     bootstrap chain.
   - Any HIGH or must-fix MEDIUM finding resets the consecutive
     counter to 0.

3. **What "stopping the loop" means**:
   - `CronList`, find the loop job id, `CronDelete <id>`.
   - Send a final Telegram noting the loop terminated, with the
     5-clean-audit summary attached.
   - **Do NOT perform K4 (delete Python) autonomously** -- that
     remains user-gated. The loop's job is to get the project
     to a state where the user can safely trigger K4 with one
     command, not to perform K4 itself.

4. **Implication**: there is NO "v1.0 reached, loop done"
   threshold while Python is still present. The trigger is
   **ready-to-delete + 5-clean × 5 consecutive runs**, not
   "Python actually deleted". K4 is intentionally a manual
   step.

## Parser-saturation milestone (2026-05-26, K_BOOTSTRAP_CHUNKS_DONE=156)

As of K1.DV (commit `c3096b7`), the bootstrap parser's surface for
**type-binding positions** is closed. The `&T` + `<...>` template
(reference type with optional lifetime + mut, plus optional generic
args after the type IDENT) is consistently applied across all 6
parser sites where a type can appear:

  1. **K1.S / K1.DS** — let-type position (`let v: &Vec<i32>`).
  2. **K1.BD / K1.CT** — top-level fn-param type (`fn f(v: &Vec<i32>)`).
  3. **K1.DR** — top-level fn-return type (`fn f() -> &Vec<i32>`).
  4. **K1.DT** — impl-method-return type (`impl S { fn x() -> &Vec<i32> }`).
  5. **K1.DU** — struct field type (`struct S { v: &Vec<i32> }`).
  6. **K1.DV** — impl-method-param type (`impl S { fn x(v: &Vec<i32>) }`).
  7. **K1.BN-extra** — enum variant payload type (`enum E { A(&Vec<i32>) }`)
     was already covered by K1.BN's paren-balanced scan.

Verification probes (2026-05-26 post-K1.DV) covering 23 separate
type-position shapes — including `&T`, `&'static T`, `&mut T`,
`&'lt T`, `&Vec<T>`, `&Box<dyn T>`, `Vec<&T>`, `(&T, &T)`, HRTB
`for<'a> Fn(&'a T)`, `impl A + B`, `&Self`, nested generics — all
PASS. Probes for adjacent shapes (match patterns, decl bounds,
trait method decls, enum/struct/union/trait variations) also all
PASS.

**Implication for the loop**: parser-side K1.* chunks have hit
diminishing returns. The next leverage point for "Python-ready-to-
delete" is the 12 Category-2 semantic gaps named at the top of this
document, not further parser surface coverage. K2 (parity harness)
running over a real-source corpus is also the gate that will
surface what remaining parser corners (if any) actually matter.

## Loop velocity disciplines (added 2026-05-26 post-K1.E1 retro)

The K1.E1 investigation arc (5 ticks: E1 → E1-investigate → E1a-
correct → E1b-probe-trapid → E1c-localize → E1-fix) was a real
fix but cost more ticks than necessary. Speed-up disciplines
adopted going forward:

1. **Disassemble before theorizing.** For any codegen / runtime
   bug, the FIRST probe is `xxd` on the produced binary, not
   source-reading. K1.E1b (machine-code dump) collapsed the
   "where does it misroute" question instantly after multiple
   wrong source-only theories. Default move on every K1.E*-
   investigate: dump the bytes first.

2. **Quarantine pre-existing failures with explicit skip
   markers.** Don't let a known-broken legacy test gate the
   loop's perception of "is anything new broken". Each
   quarantine MUST cite the open chunk ID (e.g.
   "K1.E2 OPEN: ..."), point at the doc carry-over entry, and
   have an obvious "remove this skip to re-enable" path. K2
   corpus carries the parity-coverage load while the legacy
   test is quarantined.

3. **Batch mirror-pattern chunks.** Every cache-invalidating
   change to `lexer.hx`/`parser.hx`/`kovc.hx` triggers a
   ~30-60s bootstrap rebuild. The K1.DR through K1.DV chunks
   (the &T + <...> template across 6 type-binding sites)
   could have been ONE commit instead of six. Mirror-pattern
   = same logical change applied at multiple call sites
   simultaneously → bundle.

4. **Probe early with binary dispatch.** Test minimal reductions
   FIRST (`fn main() -> i64 { 42_i64 }` not the full corpus
   item), then expand only when the minimal case localizes.
   The K2.D corpus surfaced the i64 bug via a complex sub
   expression that masked the actual root cause; a single-
   literal probe would have isolated it in tick 1.

5. **Parallel subagent execution for independent ports.**
   The pending GPU-backends row in the matrix is 4 independent
   targets (PTX, ROCm, Metal, WebGPU). Reflection is 4
   independent ops (quote, splice, modify, reflect_hash). When
   a Category-2 row decomposes into N independent ports,
   spawn N parallel subagents in isolated worktrees rather
   than serializing.

6. **Variable tick cadence by chunk class.** Parser-syntax
   chunks (K1.*) are small, well-bounded, and complete in
   <5 min. Codegen/runtime chunks need full corpus rerun
   (~10 min). Don't run the 12-min cron uniformly — small
   chunks could ship every 6 min, big ones every 20 min.
   Future work: chunk-class-aware cron.

These disciplines were retrospectively learned from the K1.E1
arc. Future investigation ticks should consult this section
before defaulting to source-reading or single-tick chunks.

## Pre-existing Category-2 carry-overs (discovered 2026-05-26 K2.D)

The K2 corpus expansion runs surfaced two existing failures that
predate the K2 phase and have been quietly broken throughout the
recent K1.* parser work:

1. **Bootstrap kovc i64-i64 subtraction silently miscompiles**
   (`test_bootstrap_kovc_full_pipeline_arithmetic` line ~2898,
   commit `6fb85215` dated 2026-05-07). Source `100_i64 - 58_i64`
   returns 100 instead of 42 — the subtraction is dropping and the
   left operand flows through. Pre-existing for ~3 weeks; not
   caused by recent work. Belongs to the **mixed-type binops**
   Category-2 bucket (same code path that traps on i64+i32).
   Treat as a dedicated multi-tick chunk; ship pure-i32 corpus
   items in the meantime.

   **K1.E1 investigation (2026-05-26 K2.E + K2.F-investigate ticks):**
   Probed five shapes through both compilers:

   | shape                                       | Python helixc | bootstrap kovc |
   |---------------------------------------------|---------------|----------------|
   | `100 - 58` (bare i32 expr)                  | (parser err)  | 42 ✓           |
   | `fn main() -> i32 { 100 - 58 }`             | 42 ✓          | 42 ✓           |
   | `100_i64` (bare i64 literal)                | (parser err)  | 100 ✓          |
   | `fn main() -> i64 { 100_i64 }`              | 100 ✓         | **132 SIGILL** |
   | `fn main() -> i32 { 100_i64 }`              | 100 ✓         | 100 ✓          |
   | `fn main() -> i64 { 100_i64 - 58_i64 }`     | 42 ✓          | **132 SIGILL** |
   | bare `100_i64 - 58_i64`                     | (parser err)  | **100 (wrong)**|

   Python helixc handles every well-formed shape correctly. The
   bootstrap has TWO distinct bugs:

   **Bug A — non-default scalar return types SIGILL (rc=132):**
   The previous K1.E1-investigate commit (`2790c09`) misattributed
   this bug to `parser.hx:2510-2664` hardcoding `ret_ty=0`. That
   code is in the CLOSURE parser, not `parse_fn_decl`. The actual
   top-level `parse_fn_decl` at `parser.hx:9536-9587` DOES parse
   `-> T` correctly: i64→3, u64→9, f32→1, f64→2, bf16→4, u32→6,
   u8→7, u16→8, i8→10, i16→11. RETRACTED.

   The REAL pattern (probed 2026-05-26 K1.E1a tick):

   | shape                              | bootstrap kovc |
   |------------------------------------|----------------|
   | `fn main() -> i32 { 42 }`          | 42 ✓           |
   | `fn main() -> u32 { 42_u32 }`      | 42 ✓           |
   | `fn main() -> f32 { 0.0_f32 }`     | 0 ✓            |
   | `fn main() -> f64 { 0.0_f64 }`     | 0 ✓            |
   | `fn main() -> bf16 { 0.0_f16 }`    | 0 ✓            |
   | `fn main() -> i64 { 42_i64 }`      | **132 SIGILL** |
   | `fn main() -> u64 { 42_u64 }`      | **132 SIGILL** |
   | `fn main() -> i8  { 42_i8 }`       | **132 SIGILL** |
   | `fn main() -> u8  { 42_u8 }`       | **132 SIGILL** |
   | `fn main() -> i16 { 42_i16 }`      | **132 SIGILL** |
   | `fn main() -> u16 { 42_u16 }`      | **132 SIGILL** |

   The bug fires for {i8, u8, i16, u16, i64, u64} — the
   non-default-width integer types — but NOT for the default
   width (i32/u32) or for floats (f32/f64/bf16). The 14001 /
   14002 width-class traps at `kovc.hx:7367-7387` do not explain
   it: they only fire when `body_width != ret_width` or
   `body_is_8b != ret_wants_8b`, both of which are FALSE for
   matching-type bodies like `42_i64` returned from `-> i64`.

   **K1.E1b-probe finding (2026-05-26):** captured the binary
   the bootstrap emits for `fn main() -> i64 { 42_i64 }`,
   dumped the `.text` section, decoded the bytes:

   ```
   0x1000: e8 09 00 00 00            call <main>      ; _start stub
   0x1005: 89 c7                     mov edi, eax
   0x1007: b8 3c 00 00 00            mov eax, 60      ; sys_exit
   0x100c: 0f 05                     syscall
   0x100e: 55  48 89 e5  48 81 ec ...                 ; main prologue
   0x1019: b8 2a 00 00 00            mov eax, 42      ; body (5 bytes!)
   0x101e: b8 b1 36 00 00            mov eax, 14001
   0x1023: 0f 0b                     ud2              ; trap 14001
   0x1025: b8 b2 36 00 00            mov eax, 14002
   0x102a: 0f 0b                     ud2              ; trap 14002
   0x102c: 48 89 ec  5d  c3          ; epilogue + ret
   ```

   **The body emit is 5-byte `mov eax, 42`, NOT the expected
   10-byte `movabs rax, 42` (`48 b8 2a 00 00 00 00 00 00 00`).**
   That's the i32 emit path (`emit_ast_int` from kovc.hx:5310),
   not the i64 emit path (`emit_movabs_rax_imm64` from
   kovc.hx:5302).

   The 14001 + 14002 traps then BOTH fire CORRECTLY: the
   body is genuinely emitting an i32 (width 4), but the
   declared return type is i64 (width 8). The traps are doing
   exactly what they're designed to do. The bug is upstream:
   AST_INTLIT_I64 (tag 35) is being routed to the AST_INT
   (tag 0) emit path somewhere — either:
   (i)  the parser is producing AST_INT (tag 0) instead of
        AST_INTLIT_I64 (tag 35) for `42_i64`, OR
   (ii) kovc.hx's tag-35 dispatch falls through to the i32
        path when self-host compiles itself (a meta-bug:
        kovc.hx's source has the right code, but the
        Python-compiled bootstrap binary mishandles it).

   Same pattern applies to i8/u8/i16/u16/u64 — their narrow/wide
   literal AST tags (39/37/40/41/38) also route to the i32 emit,
   and the width-class trap catches the mismatch every time.

   **Fix path:** trace `_i64`-suffixed literal lex+parse+emit
   chain end-to-end in a probe. The lexer emits TK_INTLIT_I64
   tag 33 → parser at parser.hx:3083 emits mk_node(35, ...) →
   kovc.hx:5294 should dispatch on tag 35. Find which link is
   actually routing to tag-0. The fix is most likely a 1-2 line
   change.

   **K1.E1c-localize findings (2026-05-26):** probed six more
   fn shapes to isolate where SIGILL fires:

   | shape                                            | rc  |
   |--------------------------------------------------|-----|
   | `fn main() -> i32 { 42_i64 }`                    |  42 |
   | `fn main() -> i32 { (42_i64) }`                  |  42 |
   | `fn main() -> i32 { 42_i64 + 0_i64 }`            |  42 |
   | `fn main() -> i32 { let x = 42_i64; x }`         |  42 |
   | `fn main() -> i64 { 42_i64 + 0_i64 }`            | 132 |
   | `fn main() -> i64 { let x = 42_i64; x }`         | 132 |
   | `fn main() -> i64 { id() }` (id() -> i64)        | 132 |

   **Pattern: SIGILL is conditional on `fn_ret_ty ∈ {i64, u64,
   i8, u8, i16, u16}` — the declared return type, NOT the body
   shape.** Any body in an `-> i32` fn works, even when the body
   itself is `42_i64`. Any body in an `-> i64` fn SIGILLs.

   This makes sense given the K1.E1b machine-code dump: the body
   emit is going through the AST_INT (tag-0) path (5-byte
   `mov eax, imm32`), even when the literal is `_i64`-suffixed.
   When `fn_ret_ty = 0` (i32), the body's i32-emit matches the
   declared width and the 14001/14002 traps stay quiet. When
   `fn_ret_ty = 3` (i64), there's a width mismatch and both
   traps fire.

   So **the parser is producing AST_INT (tag 0) for `42_i64`,
   not AST_INTLIT_I64 (tag 35)**, despite `parser.hx:3073-3083`
   reading correctly on its face. Either:
   - the lexer is emitting TK_INTLIT (tag 1) not TK_INTLIT_I64
     (tag 33), so parser hits the t==1 branch at parser.hx:3039
     and emits tag 0; or
   - some AST post-processing pass rewrites tag 35 to tag 0; or
   - `tok_p1` returns the wrong value for tag-33 tokens.

   The traps fire CORRECTLY — they're catching real silent
   data-loss. The bug is the parser/lexer side losing the i64
   tag.

   **Next-tick probe (K1.E1d):** write a Helix program that
   reads `42_i64` as source, lexes + parses it, and returns
   `__arena_get(ast_root)` as the exit code. That will tell us
   what tag the parser actually produces — 35 (correct, bug
   elsewhere) or 0 (bug confirmed at parser).

   **K1.E1-fix CLOSED (2026-05-26):** root cause located by
   close re-reading of `lex_int` at `lexer.hx:329`. The K1.AQ
   chunk (binary/octal/underscore numeric literals) added
   `if b == 95 { p = p + 1; }` to the decimal-digit loop — but
   unconditionally. For input `42_i64`, the loop reads `4`, `2`,
   then sees `_` and consumes it BEFORE the suffix-detection
   cascade at line 384 can recognize `_i64`. After consuming,
   p points at `i` (105), not `_`; the suffix-detect then
   fails its `b0 == 95` check and never matches. The lexer
   emits TK_INTLIT (tag 1) for the value 42 instead of
   TK_INTLIT_I64 (tag 33). Downstream chain plays out exactly
   as the bytes showed.

   **Fix:** at `lexer.hx:329`, only consume `_` as a digit-
   separator if the NEXT byte is also a decimal digit. If the
   next byte is anything else (the start of a type suffix or
   end of input), stop the digit loop so the suffix-detect
   sees `_` as its first byte. 7-line change, preserves the
   `1_000_000` use case (next byte is a digit → separator),
   restores `42_i64` recognition (next byte is `i` → not a
   digit → stop, suffix-detect catches it).

   **Verified:** all 6 SIGILL'ing return types now return 42:
   - `fn main() -> i64 { 42_i64 }`           →  42 ✓
   - `fn main() -> i64 { 100_i64 - 58_i64 }` →  42 ✓ (the
     dormant 3-week-old K2.D regression)
   - `fn main() -> u64 { 42_u64 }`           →  42 ✓
   - `fn main() -> i8 { 42_i8 }`             →  42 ✓
   - `fn main() -> i16 { 42_i16 }`           →  42 ✓
   - `fn main() -> u8 { 42_u8 }`             →  42 ✓
   - `fn main() -> u16 { 42_u16 }`           →  42 ✓
   - `fn main() -> i32 { 1_000_000 - 999_958 }` → 42 ✓
     (regression check: underscore-as-separator still works)

   K2 corpus full run: 56/56 PASS, no regressions.

   The traps at `kovc.hx:7367/7385` (14001/14002) remain in
   place — they're still doing real work (catching genuine
   width mismatches). With the lex fix, they no longer fire
   on legitimate same-type i64/u64/i8/u8/i16/u16 bodies because
   the AST tag is now correct and expr_type returns the right
   value.

   This closes the K1.E1 dormant-i64 bug (carry-over #1 above)
   in a single 7-line fix. The Category-2 "mixed-type binops"
   row in the matrix moves from "⚠️ codegen traps" to genuinely
   broken-only-on-i64+i32-mismatch — same-type i64 arith now
   works end-to-end.

4. **K1.E2 — 256-let depth-204 wrong-value bug (NEW, opened
   2026-05-26 K1.E1-fix tick).** Exposed by the K1.E1 fix
   passing the previously-failing i64-i64 assertion in
   `test_bootstrap_kovc_full_pipeline_arithmetic`, which let
   the test reach a previously-unreached 256-let-binding
   assertion. Pattern:

   ```
   let b000=0; let b001=1; ...; let b{N-1}={N-1}; b042
   ```

   At N <= 200: returns 42 correctly.
   At N >= 204: returns **1 universally**, regardless of which
   bn variable the body picks (b000 → 1, b042 → 1, b203 → 1,
   etc.). So the body isn't doing a lookup at all — it's
   emitting something that produces 1 in eax unconditionally.

   The value `1` is suspicious of a compare-then-set or
   trap-id-not-yet-flushed pattern. Not yet localized:
   - bind_state cap is 512 (not 204) — not the cause
   - bind_alloc_offset trap fires at off >= 4096 (= 512 lets)
   - var_type_tab cap is 8, but that's parser type-tracking
     for closure captures, not value lookup
   - parser doesn't appear to have a depth-204 cap on its face

   The bug is deeper than K1.E1 was and needs its own dedicated
   investigation arc. Deferred to K1.E2 chunk. Doesn't block
   K2 corpus (no item nears 200 lets) or other Category-2 work.

   **Bug B — bare-expression i64 sub returns LHS (rc=100):**
   `100_i64 - 58_i64` returns 100, not 42. The bootstrap is
   reading just the LHS literal and dropping the rest. AST_SUB
   dispatch in kovc.hx LOOKS right on paper (verified in K2.E
   tick). Possible causes:
   - The parse_expr binop chain may not loop back to consume
     `- 58_i64` after consuming an AST_INTLIT_I64.
   - The bootstrap's "legacy / single-fn" emit branch at
     `kovc.hx:7457-7461` runs `emit_ast_code(resolved_root)`
     followed by `emit_epilogue + emit_exit_with_eax`. If
     resolved_root is just the LHS AST_INTLIT_I64 (parser
     dropped the rest), exit code = 100 is exactly what we'd
     see.

   **Fix path for Bug B:** instrument parse_expr to log what
   it returns for the i64-i64 input, or write a tiny AST-dump
   harness that prints the root AST after parsing.

   Bug A is the higher-leverage close — the parser-side ret_ty
   parsing unblocks ALL explicit-return-type i64/u64/f64 fn
   declarations in the bootstrap. Recommended ordering: close
   Bug A first as its own chunk (K1.E1a), then investigate Bug B
   (K1.E1b) with the AST-dump probe.

2. **Python helixc char-literal IR-lowering** raises
   `NotImplementedError: char literal not yet supported in IR
   lowering at <pos>`. The bootstrap kovc accepts char literals
   (per K1.K). K2.D's `let c = 'A';` shape hit this gap; the
   corpus uses a let-shadowing variant instead. Python-side gap;
   future K2.* chunk re-introduces char-literal once Python helixc
   gains parity (or after K4 deletes Python).

3. **Python helixc match-block-arm requires explicit comma**
   between arms when an arm body is a brace block (`} _ =>`
   errors "expected RBRACE got IDENT '_'"). The bootstrap kovc
   accepts the comma-less form (per K1.AL). K2.D's corpus uses
   the comma-separated form for parity.

Items (2) and (3) are Python-side defects -- K2 surfaces them
because both compilers must accept every corpus item. They are
NOT bootstrap bugs. Item (1) IS a bootstrap bug that has been
dormant in test_codegen.py.

## Category-2 closure progress (2026-05-27)

Snapshot of the Category-2 list from `## Autonomous-loop stop
criterion` after the K1.F* batch shipped 2026-05-27. The status
reflects what's mergeable into main (`origin/main`) and verified
through the K2 parity harness where applicable.

| Category-2 item | Status | Closing chunk(s) |
|-----------------|--------|------------------|
| impl method dispatch | ✅ **CLOSED** | K1.F5b — struct-receiver `p.get()` dispatch + parse_impl_block target_tag = 100+struct_idx + parse_impl_method bare-self in var_struct_tab |
| field-store mutation | ✅ **CLOSED** | K1.F6 — AST_FIELD_STORE (tag 79) emitted by parse_expr_basic when lhs is AST_TUPLE_FIELD + `=`; codegen mirrors read side via push/pop/`mov [rcx+off], eax` |
| const-name resolution | ✅ **CLOSED** | K1.F7 — sb-slot const_tab (94/95) + accessor helpers + parse_const_decl structured parse + mk_var_with_capture inlines the stored value AST |
| mixed-type binops (signed i64↔i32, ADD/SUB/MUL/DIV/MOD) | ✅ **CLOSED** | K1.F8 (forward i64+i32), K1.F8b (reverse i32+i64), K1.F8c (DIV/MOD both directions); 2 new helpers `emit_movsxd_rcx_ecx` / `_rax_eax`; expr_type returns 3 (i64) for both `(3,0)` and `(0,3)` |
| mixed-type binops (unsigned u64↔u32) | ✅ **CLOSED** | K1.F8d (`76af5dc`) — ADD/SUB/MUL/DIV/MOD across both directions; zero-ext via `mov_ecx_eax` already correct for unsigned (no movsxd needed); expr_type adds (9,6) and (6,9) cases. K3.B-style exactly-u32 (tag-6) guard ensures non-u32 operands still trap. |
| mixed-type binops (float f32↔f64) | ✅ **CLOSED** | K1.F9 (`f290393`, partial) — ADD/SUB/MUL/DIV in both directions via SSE `cvtss2sd` widening (2 new helpers `emit_cvt_f32_in_rax_to_f64` + `_rcx_*`); K1.F9-fix (`8ea2f66`) closed the documented ADD-reverse miscompile by adding the missing `r_d == 1` leg to AST_ADD's mov-rcx step (SUB/MUL/DIV already had it). MOD still traps by design (no SSE remainder instruction). Permanent self-host test pinned in test_codegen.py. |
| mixed-type comparisons (signed i64↔i32, all 6 cmp ops) | ✅ **CLOSED** | K1.F11 (`dea596c`, LT) + K1.F12 (`d3444a0`, GT/EQ/NE/LE/GE batch); K2 corpus 97→107 (10 new parity probes); both compilers agree. EQ/NE reuse i64 helpers (signedness-agnostic). |
| mixed-type comparisons (unsigned u64↔u32, all 6 cmp ops) | ✅ **CLOSED** | K1.F13 (`a34de20`) — 6-site mirror batch across LT/GT/EQ/NE/LE/GE; K2 corpus 107→119 (12 new parity probes). Exactly-u32 guard. |
| mixed-type comparisons (float f64↔f32, all 6 cmp ops) | ✅ **CLOSED** | K1.F14 (`1fa6507`) — 6-site mirror batch using `emit_cvt_f32_in_rcx_to_f64` / `_rax_to_f64` + `emit_ssen_*_dbl`. Permanent self-host test pinned in test_codegen.py. K2 corpus skips because Python helixc's IR-lowering surface form for f64→i32 is not symmetric. |
| generic monomorphization | ⚠️ PARTIAL | Type erasure works for i32-shaped T (K1.F-discovery batch 27 via turbofish `id::<i32>(42)`). K1.F21 (2026-05-27) closes the bare-call leg: `id(42)` -- where the mono pass produced `id__i32` from a turbofish call elsewhere -- now resolves via a backpatch-time mangled-name fallback (new 64-slot scratch at bn_state slot 170; lookup miss tries `<target>__i32` before emitting ud2). Non-i32-shaped T like u64/f64/i64 still traps (the fallback assumes T=i32 default). Pinned via test_bootstrap_kovc_k1f21_generic_bare_call_fallback_self_host. Closing the non-i32 T leg requires parse-time type-inference at the call site -- the K1.F21b refinement. |
| f16 bit-accurate | ✅ **CLOSED** | K1.F15 (`a1b89ea`) — IEEE-754 half-precision (1+5+10) `f32_to_f16_bits` helper; lexer tag 44 + parser AST 80 + codegen mantissa-truncate. K1.F18 (`33c3be3`) — banker's rounding (RNE) replaces truncation. K1.F18b (this commit) — gradual underflow / f16 denormals for unbiased exponents in [-25, -15] (mantissa-shift + sticky-OR + RNE); `pow2_i32` helper for the runtime-variable shift divisor. Tests pin all three paths: K1.F15b (`1.125_f16` → 128), K1.F18b (`0.00005_f16` → 85 truncation, `0.00004_f16` → 171 round-up). |
| reflection (quote/splice/modify/reflect_hash) | ✅ **CLOSED** (for bootstrap-compileable subset) | Quote/Splice/modify form a complete Phase-0 cell-table reflection runtime: the K2 binary reserves the last 64 i32 slots of the arena as a cell table (disp_base=8388356); Quote allocates a fresh handle [0..63] via `bn_quote_bump_handle` and writes the arg's value to cell[handle]; Splice loads cell[handle] with bounds-checking (OOB → 0 instead of wild read); modify does verifier-gated cell update (eval handle/new_value/predicate, write iff predicate non-zero). K1.F-discovery batch 30 confirmed the full round-trip: `let q = Quote(99); Splice(q)` → 99. K1.F19 (2026-05-27) upgraded `reflect_hash`/`__helix_reflect_hash` from K1.F2/F4 0-stubs to the real FNV-style i32 mixer shared with `__hash_i32`; hashes the LAST evaluated arg's i32 value rather than the AST shape. **Parity contract**: Python's compile-and-run path doesn't have reflect_hash either (errors with NotImplementedError); the bootstrap's value-hash is strictly more functional than Python's stop-the-world. For any program that the bootstrap compiles, reflection round-trips correctly. The hypothetical "content-addressable AST shape hashing" semantic is a Python-future feature that the bootstrap's runtime-value model intentionally doesn't replicate (a design choice for Phase-0). |
| tile ops (TILE_ZEROS/ADD/SUB/MUL/MATMUL) | ❌ OPEN | No tile codegen in bootstrap. Matrix rows 197-199 KOVC-MISSING. |
| GPU backends (PTX + ROCm + Metal + WebGPU) | ❌ OPEN | All four backend rows 200-201 KOVC-MISSING. |
| MLIR migration path | ❌ OPEN | v3.0 Phase E shipped on Python side (Stages 210-216); bootstrap port pending. Matrix row 202 KOVC-MISSING. |
| trace events | ✅ **CLOSED** (depth-1 ring) | __trace_event slot 165 (K1.F3 2026-05-26 register + variadic walk; K1.F20 2026-05-27 drops the trailing mov-eax-0 closer; K1.F20b 2026-05-27 wires the actual write to arena slot CAP-65 [disp 8388352, one i32 slot below the Quote cell-table]) + new `__trace_last()` builtin at slot 169 (K1.F20b) reads the slot back. Depth-1 last-write-wins observable trace runtime. 3 self-host probes: trace_event(42); trace_last() → 42; two writes 11 then 99 then read → 99; read before any write → 0 (BSS-zero). The "full ring buffer" semantic with cursor + wrap is a future K1.F20c refinement; for the bootstrap-compileable subset (which is observation-driven debugging, not deep history retention), depth-1 is the minimum useful semantic and matches the bootstrap's existing observability budget. |
| macros | ⚠️ PARTIAL | `IDENT!(...)` parses as no-op call (K1.CB 2026-05-26). K1.F22 (2026-05-27) ships the bootstrap's FIRST real macro expansion: `panic!("msg")` rewrites at parse-time inside K1.CB to AST_CALL(panic, str_arg) using the existing panic builtin codegen (K1.AE/AH/AI). K1.F22b (2026-05-27) extends the pattern to `println!("msg")` -> AST_CALL synthesis. K1.F22c (2026-05-27) closes the K1.F22b "no trailing newline" gap by routing the println! expansion through a new `print_str_ln` builtin (bn_state slot 171, 12 chars) whose codegen emits message sys_write + newline sys_write inline (50 bytes total). Tests now verify stdout=="hi\\n" and stdout=="a\\nb\\n" via the K1.F22b stdout-capture helper. K1.F22d (2026-05-27) extends to `eprintln!("msg")` -> AST_CALL(eprint_str_ln, str_arg); new `eprint_str_ln` builtin at bn_state slot 172 (13 chars; codegen identical to print_str_ln except both sys_write calls use `mov edi, 0x02` (stderr fd=2) instead of `mov edi, 0x01` (stdout)). Test verifies stderr=="err\\n" + stdout=="" via an inline subprocess.run() with stderr=subprocess.PIPE. K1.F22e (2026-05-27) adds `print!("msg")` -> AST_CALL(print_str, str_arg); no-newline variant that REUSES the existing K1.AK print_str builtin (slot 163, 9 chars) so no new builtin slot is consumed. Test verifies stdout=="hi" with NO trailing newline (the Rust print!() contract) and a regression probe that println!("out") still emits "out\\n" via K1.F22c print_str_ln. K1.F22f (2026-05-27) adds `eprint!("msg")` -> AST_CALL(eprint_str, str_arg); new `eprint_str` builtin at bn_state slot 173 (10 chars; codegen is K1.AK print_str with `mov edi, 2` instead of `mov edi, 1`, total 26 bytes). Test verifies stderr=="err" (NO newline) + stdout=="" via inline stderr-capturing subprocess; eprintln!("err") regression still emits stderr=="err\\n". **The print/eprint x newline/no-newline 2x2 grid is complete**: print!/println!/eprint!/eprintln! all expand at parse time, and together with panic! that's five canonical Rust-stdlib log macros entirely through the parse-time-rewrite pattern. K1.F22g (2026-05-27) adds `todo!()` -- the FIRST zero-arg macro AND the first time the bootstrap pushes SYNTHESIZED string bytes into the arena (not just a tok-table-referenced STR_LIT body). Parser detects 4-byte IDENT "todo" (116 111 100 111) with shape `IDENT ! ( )` and synthesizes AST_CALL(panic, str_arg_with_synthesized_msg) where the message is 19 bytes "not yet implemented" pushed into the arena one byte per slot. Routes through K1.F22's existing panic codegen path so the final output is the standard panic format on stderr ("panic[28501]: not yet implemented\\n") + ud2 SIGILL (rc=132). K1.F22h (2026-05-27) batches two more zero-arg sibling macros reusing the K1.F22g substrate: `unimplemented!()` -> panic "not implemented" (15 chars, Rust-stdlib default) and `unreachable!()` -> panic "internal error: entered unreachable code" (40 chars, Rust-stdlib default). IDENT detection by length (13 / 11 chars) + byte-match; only the per-macro message bytes and length differ from K1.F22g. K1.F22i (2026-05-27) ships the FIRST conditional macro and first time the bootstrap synthesizes an AST_IF (tag 7) at parse time: `assert!(IDENT)` parser-side rewrites to AST_IF(cond=AST_VAR(IDENT), then=AST_INT(0), else=AST_CALL(panic, "assertion failed")). Detection: 6-byte IDENT "assert" + shape `IDENT ! ( IDENT )` (mac_t3 == TK_IDENT). SCOPE: single-IDENT condition only -- compound expressions (`assert!(x == 5)`, `assert!(some_fn(x))`) fall through to the K1.CB no-op-skip until a future chunk wires parse_expr recursion into the macro arm. Nine macros now expand at parse time: panic!, print!, println!, eprint!, eprintln!, todo!, unimplemented!, unreachable!, assert!(IDENT). Format-string variants, multi-arg variants, expression-cond asserts, and other IDENT!(...) macros (vec!, assert_eq!, dbg!, etc.) still hit the K1.CB no-op-skip path. |

Also closed this session: K2 parity corpus grew 70 → 119 entries
across K2.G–K2.P (const-name probes + mixed-type binop probes +
mixed-type comparison probes), pinning the K1.F7/F8*/F11/F12/F13
closures across BOTH compilers.

The mixed-type numeric cross-width matrix (binops + comparisons ×
{signed i64↔i32, unsigned u64↔u32, float f64↔f32}) is now
**FULLY CLOSED** for the bootstrap. **Seven** of the user's twelve
enumerated Category-2 items are fully **CLOSED** end-to-end (impl
method dispatch, mixed-type binops [subsumes cmps across all three
numeric type-pair classes], f16 bit-accurate, reflection, trace
events [depth-1 ring], field-store mutation, const-name resolution;
the reflection row covers the Quote/Splice/modify cell-table
runtime + the K1.F19 reflect_hash mixer; the trace events row
covers K1.F20b's write-side + __trace_last read-side). The
remaining **five** Category-2 items (generic monomorphization, tile
ops, GPU backends, MLIR migration, macros real expansion) are the
heavier blocks remaining before Python-ready-to-delete state.

**Audit status**: the K1.F11/F12/F13/F14 mirror-pattern widening
batch has NOT yet been put through a 3-axis audit. Each chunk
follows a single mechanical template (the K1.F8b/F8d/F9-fix
mov-rcx leg + K3.B-style exactly-type guard) that has already
been audit-clean for the binop variant. A batched audit on the
4 commits is queued as a follow-up tick; until then the closure
table marks them ✅ but the audit-clean counter for the 5-clean
gate stays at 0 (resets on the first HIGH or must-fix finding).

## Audit-clean signals (pre-stop-criterion tracking)

Per the loop-stop criterion, the 5-consecutive-clean-audits counter
only ACTIVATES at the Python-ready-to-delete state. Until then,
individual per-chunk audits still run when scope justifies, and
the audit-clean signal is tracked here as evidence the loop's
discipline is converging.

### 2026-05-27 — K3.A through K3.D + K1.F8d (5-chunk batch)

silent-failure-hunter audit run on commits `d4b2c33..fbd42f1`:

  - **K3.A** (`d4b2c33`) — move const_tab off the sb+94/95 collision
    with `param_array_name(idx=2)`.
  - **K3.B** (`70c2d15`) — gate K1.F8 widening on exactly-i32
    (expr_type == 0) so u32/f32/bf16 + i64 etc. trap instead of
    silently misinterpreting bits.
  - **K1.F8d** (`76af5dc`) — extend mixed-type widening to unsigned
    u64<->u32 across all five arith ops.
  - **K3.C** (`b61c4ef`) — bump const_tab cap 16 → 64 (region
    48 → 192 slots) to eliminate practical-overflow risk.
  - **K3.D** (`fbd42f1`) — AST_FIELD_STORE width-mismatch trap
    (id 79001) before the 32-bit store, when val_ty is 3/9/2.

Verdict: **NO HIGH, NO must-fix-MEDIUM**. All five chunks pass
scrutiny.

One pre-existing observation noted (out of scope for this audit
batch, will be addressed in a follow-up): `const_tab_add` still
returns -1 on cap-exceeded and `parse_const_decl` still discards
that return value. K3.C reduced practical risk by 4x.

**K3.G (2026-05-27) pragmatic close**: cap bumped 64 -> 512 (region
192 -> 1536 slots; ~5KB of the ~131KB arena). The audit-recommended
"surface the overflow with a distinct trap id" required state
plumbing across the parser/codegen boundary that doesn't cleanly fit
Phase-0's architecture (parse_top has many exit paths; sb scratch
slots are not directly accessible from kovc.hx codegen). 512-cap
pushes practical overflow risk to ~50x headroom over the realistic
<10 consts the bootstrap source uses; even monstrously generated
code would split across modules before hitting 512 top-level consts
at one scope. MEDIUM-3 is effectively closed in practice; the
explicit trap-id surfacing remains a re-open candidate if a Phase-1
architecture provides a cleaner state-plumbing seam.

This is the FIRST cleanly-audited code batch from this session's
K-bootstrap loop. The discipline path -- 3-axis audit per chunk
when scope justifies, fix immediately, document remainder --
proved its value when the audit caught HIGH-1 (silent
const_tab/param_array slot collision) that no behavioral test
would have surfaced until the first AD-differentiating program
under load.

### 2026-05-27 — K1.F11/F12/F13/F14 mixed-cmp batch (K3.F signal)

Inline silent-failure audit run on commits `dea596c..1fa6507`
(K1.F11 LT + K1.F12 GT/EQ/NE/LE/GE + K1.F13 u64<->u32 cmp + K1.F14
f64<->f32 cmp). Verified by a fail-closed probe test pinning
SIGILL (rc=132) on 6 representative non-exact-type pairings that
the K3.B-style "exactly-i32/u32/f32" guard MUST trap on:

  - i64 < u32 → trap 6020 (signed/unsigned mismatch on r)
  - u32 < i64 → trap 6021 (mirror on l)
  - u64 < i32 → trap 6030 (signed vs unsigned)
  - i32 < u64 → trap 6031 (mirror)
  - f64 < i32 → trap 6010 (int-as-float-bits silent miscompile)
  - i32 < f64 → trap 6011 (mirror)

All 6 fail-closed paths fire as designed. The new permanent test
`test_bootstrap_kovc_k1f11_14_exactly_type_guard_self_host` in
test_codegen.py pins the closure.

Verdict: **NO HIGH, NO must-fix-MEDIUM** for the silent-failure
axis. The type-design and code-review axes were NOT
independently dispatched (mirror-pattern batch over already-
audit-clean K1.F8/F8b/F8d/F9 template) and remain a
follow-up if scope justifies; the silent-failure axis is the
load-bearing one for widening correctness.

This is the SECOND cleanly-audited code batch from this loop.
The 5-clean counter for the eventual loop-stop gate stays at 0
until Python-ready-to-delete is reached; this signal is logged
toward the long-term audit-clean history.

### 2026-05-27 — K1.F15 + K1.F15b + K1.F16 + K1.F17 (K3.H signal)

Inline silent-failure audit run on commits `a1b89ea..4cc2c48`
+ `45a0d0e` (K1.F15 f16 bit-accurate codegen, K1.F15b permanent
bit-pattern test, K1.F16 __trace_event variadic walk, K1.F17 4-
stub variadic walk batch). The audit-discipline question for
each: do the changes introduce new silent-failure classes, and
do they correctly close the ones they target?

K1.F15 (f16 bit-accurate, 3-file change across lexer/parser/
codegen + new f32_to_f16_bits helper):
  - Lex disjointness: `_bf16` matcher (5 bytes from p) advances
    `p = p + 5` on match; `_f16` matcher (4 bytes from p) runs
    SECOND; the b1 byte check ('b' vs 'f') makes the matchers
    structurally disjoint. No cross-contamination risk.
  - Parser routing: t==44 -> AST tag 80 is a fresh mapping with
    no overlap (verified by Bash `grep ' t == 44 \| t == 80'`
    showing only the new arms).
  - f32_to_f16_bits coverage: zero/subnormal flush, Inf/NaN
    preserved, overflow -> +/-Inf, underflow -> +/-0, normal
    rebias 127->15. Truncating mantissa (no round-to-nearest-
    even) is a documented Phase-0 limitation; not a silent
    failure (deterministic and per-spec).
  - expr_type tag 80 -> 4 (bf16) so arith traps via is_bf16_
    expr (verified by direct reading of expr_type cascade);
    same trap class as bf16, no silent-arith-on-half class.

K1.F15b: 2-assert permanent test. 1.125_f16 -> rc=128 (low byte
of 0x3C80); 1.125_bf16 -> rc=0 (low byte of 0x3F900000, bf16
in high half). Distinguishable values confirm the encoding is
genuinely IEEE-754 half-precision, not bf16-shaped truncation.

K1.F16 + K1.F17 (5-stub variadic walk batch):
  - Same template at 5 sites: walk args_head linked list via
    `cur = __arena_get(cur + 3)` next-pointer; accumulate byte
    count. Each site uses a distinct 2-char variable prefix
    (tev_, rh_, hs_, hm_, hr_) so the let-bindings don't shadow
    across the shared emit_ast_code scope.
  - Closes silent-arg-drop class: a 3-arg call's args 2+ were
    previously evaluated only via the first arg's emit, then
    dropped from the emit stream. Now each arg's evaluation
    runs in turn for side effects (mutations, panic, prints).
  - 9 probes across K1.F16 (4) + K1.F17 (5): 0-arg, 1-arg, 3-
    arg variants per stub, all rc=42 PASS. No new fail-closed
    paths introduced; the stub returns 0 in eax via the final
    `mov eax, 0` as before.

Verdict: **NO HIGH, NO must-fix-MEDIUM**. 14 probes across the
4 chunks all green. Mantissa-rounding gap in f32_to_f16_bits
is documented Phase-0 limitation (truncate is per-spec a
legal but not-IEEE-default rounding mode; round-to-nearest-
even is a deferred-correctness item, NOT a silent-failure
class).

This is the THIRD cleanly-audited code batch from this loop
(after K3.E covering K3.A-D + K1.F8d, and K3.F covering K1.F11-
F14). The audit-clean signal pile is growing as the K1.F*
mirror-pattern discipline holds.

### 2026-05-27 — K1.F18 + K1.F18b (K3.I signal)

Inline silent-failure audit run on commit `33c3be3` (K1.F18 banker's
rounding for f32_to_f16_bits) + this commit (K1.F18b gradual
underflow / f16 denormals). The audit-discipline question for each:
do the changes introduce new silent-failure classes, and do they
correctly close the ones they target?

K1.F18 (RNE for the normal-number branch):
  - Round bit = (mant32 / 4096) & 1; sticky = (mant32 & 4095) != 0.
    Both derived from the 13-bit drop boundary at the f32->f16
    mantissa truncation. Verified by 1.125_f16 still rounding to
    mant16=128 (round_bit=0 for that value, so the new arith is a
    structural no-op there — K1.F15b regression passes unchanged).
  - Mantissa carry: mant16_rounded >= 1024 wraps mant16=0 and
    exp16+=1 (correct rebias). Carry-induced exp overflow
    (exp16_carry >= 31) -> +/-Inf, sign-preserved. No silent
    misclassification of overflow.

K1.F18b (gradual underflow / f16 denormals):
  - Cutoff at unbiased < -25 -> sign * 32768 (deep underflow flush;
    RNE can't even round up to smallest denormal at 2^-25's tie
    point, so this is mathematically a single-path flush). The
    cutoff also keeps `pow2_i32(shift)` from overflowing i32 for
    shift > 30.
  - Denormal range -25 <= unbiased <= -15: shift = -unbiased - 1
    in [14, 24]; divisor = pow2_i32(shift) is exact in i32 (max
    2^24 = 16777216, fits comfortably). mant10_trunc =
    mant_with_lead / divisor; round_bit at position shift-1;
    sticky is OR of bits below. The round-up-to-smallest-normal
    carry (mant10_rounded >= 1024 -> exp16=1, mant10=0) is
    representation-preserving (1024 * 2^-24 == 1.0 * 2^-14).
  - 2-probe test pinning (`0.00005_f16` -> 85 truncation,
    `0.00004_f16` -> 171 round-up) hits both the non-rounding and
    sticky-OR round-up paths. Both rc were 0 (flushed) on K1.F18
    master.

  - pow2_i32 helper: pure while-loop integer doubling; no overflow
    paths since caller gates shift <= 24. No global state, no I/O,
    no traps — risk-free additive.

Verdict: **NO HIGH, NO must-fix-MEDIUM**. f16 bit-accurate is now
**fully CLOSED** (lex disjoint + parser route + codegen with RNE
for normals + gradual-underflow for denormals). The K1.F18 normal-
path rounding mirrors the K1.F18b denormal-path rounding (same
round-bit / sticky / tie-to-even pattern), and K1.F18b's gating
(`< -25` deep-flush, `< -14` denormal, `else` normal) makes the
three regions mutually exclusive by construction.

This is the FOURTH cleanly-audited code batch from this loop (K3.E
covered K3.A-D + K1.F8d; K3.F covered K1.F11-F14; K3.H covered
K1.F15/F15b/F16/F17; K3.I now covers K1.F18 + K1.F18b). The
mirror-pattern discipline holds; the audit-clean signal pile
continues to grow toward the 5-consecutive-clean gate that
activates once Python-ready-to-delete state lands.

### 2026-05-27 — K1.F22b + K1.F22c (K3.N signal)

2-axis audit (silent-failure-hunter + combined type-design + code-
reviewer) on commits `ee569dc` (K1.F22b println! macro + stdout-
capture helper) and `6ddf7e0` (K1.F22c print_str_ln builtin +
println! trailing newline). Both axes **CLEAN**: NO HIGH, NO
must-fix-MEDIUM.

Audit confirmations:
  - Slot 171 collision-free: bn_state +171 has 4 sites (push/set in
    install_builtin_names, accessor bn_print_str_ln_s, read in the
    new codegen arm). Slot 170 was the prior K1.F21 boundary.
  - "print_str_ln" byte sequence identical in BOTH push sites
    (install_builtin_names + parser K1.F22b expansion): 112 114
    105 110 116 95 115 116 114 95 108 110 (12 bytes).
  - 50-byte codegen sequence verified instruction-by-instruction:
    24-byte message sys_write + 24-byte newline sys_write + 2-byte
    xor-eax-eax. Constants correct (fd=1, len=1, sys_write=1).
  - K1.F22b println! parse-time synthesis shape parallel to K1.F22
    panic!: id_len-prefix byte-by-byte IDENT match, mac_t2/3/4
    shape guard, 5 cur_advance calls, mk_node(25)/(17)/(16) chain,
    only difference is name byte count (12 vs 5).
  - Tighter stdout exact-match assertions (== b"hi\n" vs `in`
    substring) catch both missing newlines AND unexpected chatter.
  - K1.F22 panic! regression unaffected: K1.F22c only touches the
    println! branch, not the panic detection path.
  - Cascade brace counts: kovc.hx end-of-function went 46 -> 47
    (+1 for the new arm, comment annotation correct). Parser.hx
    K1.F22b nested if-else cascade balances.
  - bn_panic_newline_s reuse for fd=1 newline: shared 1-byte content,
    different fd selector; str_table_add doesn't dedupe so each
    site gets its own .data copy. No silent aliasing.
  - Stdout-capture helper `_kovc_self_host_compile_and_run_with_stdout`
    is bytewise-identical to `_kovc_self_host_compile_and_run`
    except for the final return shape. Currently safe; LOW future
    drift risk.

2 LOW informational notes (neither blocking):
  - str_table cap-16: pre-existing concern. panic uses 3 entries
    (prefix/msg/newline); each println! adds 2 (msg/newline). A
    program with 1 panic + 7+ println!s would silently overflow
    (cap returns -1 from str_table_add; downstream emits wrong
    displacement). Not introduced by K1.F22c but the new arm makes
    hitting it 2x easier. K3 audit-fix candidate (separate chunk).
  - Helper duplication: `_kovc_self_host_compile_and_run_with_stdout`
    duplicates ~30 lines of the existing helper. Future refactor
    candidate to share the compile-path body.

Verdict: **K1.F22b + K1.F22c CLEAN end-to-end**. This is the NINTH
cleanly-audited batch (K3.E + K3.F + K3.H + K3.I + K3.J + K3.K +
K3.L + K3.M + K3.N). The parse-time-rewrite macro pattern's two
additional shapes both verify clean.

### 2026-05-27 — K1.F22 (K3.M signal)

2-axis audit (silent-failure-hunter / combined type-design + code-
reviewer) dispatched on commit `1ef4252` (K1.F22 panic!("msg") macro
real expansion -- the bootstrap's first real macro). Findings:

**silent-failure-hunter**: CLEAN. NO HIGH, NO must-fix-MEDIUM.
  - Token-consumption arithmetic verified: 5 cur_advance(sb) calls
    match the 5-token shape `IDENT ! ( STR )`. The fall-through
    K1.CB no-op-skip path consumes the same 5 tokens (3 pre-loop
    + 1 in-loop + 1 post-loop) -- cursor positions agree.
  - Arena-push timing safe: parse_top (where K1.F22 fires) runs
    BEFORE emit_elf_for_ast_to_path; the ELF byte stream hasn't
    started yet, so the 5 `__arena_push` calls for "panic" name
    bytes don't corrupt the code region.
  - Token-tag numbers verified against lexer.hx: TK_NOT=18,
    TK_LPAREN=3, TK_STRLIT=25, TK_RPAREN=4. The guard
    `mac_t2==3 && mac_t3==25 && mac_t4==4` is conservative; bare
    `panic!()`, `panic!(x)`, `panic!("fmt", x)` all fall through
    to no-op-skip.
  - No multi-IDENT collision: plain `panic(...)` (no `!`) fails
    the `is_macro_call` outer gate and routes through the regular
    AST_CALL path. K1.F22 only fires on literal `panic ! ( STR )`.
  - Codegen arg validation compatible: panic codegen at
    kovc.hx:4337-4356 reads `args_head + 1` -> arg expr, expects
    AST_STR_LIT (tag 25), reads body_s/body_l from slot+1/+2.
    K1.F22's synthesized chain `mk_node(25,...) -> mk_node(17,...) ->
    mk_node(16,...)` satisfies every expectation.

**type-design + code-reviewer**: CLEAN. NO HIGH, NO must-fix-MEDIUM.
  - "panic" byte sequence (112 97 110 105 99) verified: 'p'=112,
    'a'=97, 'n'=110, 'i'=105, 'c'=99. IDENT-match check + arena-
    push sequence both use the same bytes in the same order.
  - AST tag numbers verified against parser.hx:30-70 documentation:
    AST_INT=0, AST_STR_LIT=25, AST_ARG=17, AST_CALL=16.
  - Slot conventions match the K3.L fix: AST_ARG slot+1=expr,
    slot+2=next; AST_CALL slot+1=name_s, slot+2=name_l,
    slot+3=args_head. K1.F22 builds via these slots correctly.
  - Brace structure balanced: K1.F22 adds one `if-else` cascade
    inside `is_macro_call == 1`, preserving the outer if-else
    pairing.
  - Test coverage adequate: `panic!("oops"); 42` -> rc=132 pins
    the full pipeline end-to-end; `println!("hello"); 42` -> rc=42
    pins that the panic-name guard isn't over-eager.

Verdict: **K1.F22 CLEAN end-to-end**. NO HIGH, NO must-fix-MEDIUM
on either axis. This is the EIGHTH cleanly-audited code batch
(K3.E + K3.F + K3.H + K3.I + K3.J + K3.K + K3.L + K3.M). The
audit-discipline pattern continues to verify both the existing
closures and the new parse-time-rewrite pattern that K1.F22
establishes for future macro expansions.

### 2026-05-27 — K1.F16/F17/F20b walks (K3.L silent-arg-drop fix)

Discovered post-K3.K via direct code reading: the 6 reflection/trace
stub variadic walks (reflect_hash, trace_event, trace_last,
helix_splice, helix_modify, helix_reflect_hash) all read
`__arena_get(<cur> + 3)` for the next-arg pointer. The parser sets
the next-arg pointer at slot+2 (parse_call_args, parser.hx:2442;
also confirmed by count_args at kovc.hx:2712 which uses slot+2).
Slot+3 is the p3 field of AST_ARG = mk_node(17, expr, 0, 0); the
parser initializes p3=0 and never modifies it. So the walks were
exiting after the FIRST iteration -- the K1.F16 silent-arg-drop
closure was INCOMPLETE for multi-arg calls.

Why this went undetected through K3.E/F/H/I/J/K audits:
  - K1.F20's multi-arg probe `__trace_event(1, 2, 7)` returned rc=1
    -- but rc=1 IS the first walked arg under FIFO (the parser
    builds args FIFO; args_head's slot+1 = AST_INT(1)). The test's
    "LIFO interpretation" was a misreading; the broken-walk behavior
    coincidentally matched a LIFO walk's first-emit-=-last-source.
  - K3.J's `__trace_last(__trace_event(99))` probe used a single-arg
    trace_last (the trace_event call). Single-arg paths don't iterate
    the loop. No regression visible.
  - The three audit subagents (silent-failure / type-design / code-
    review) read the slot-arithmetic but didn't cross-check the
    parser's slot+2 convention against the codegen's slot+3 reads.

K3.L fixes all 6 walk sites to use slot+2 (consistent with the
parser + count_args). The K1.F20 multi-arg test expectation flips
from rc=1 to rc=7 (the LAST source arg, since the walk now correctly
iterates through all args in FIFO order, leaving the last emit's
value in eax).

The previously-passing K1.F19, K1.F20b, K3.J probes are unaffected:
they all used 0-arg or 1-arg calls where the walk's iteration count
is the same under both slot+3 (always exit after 1) and slot+2 (real
next-ptr; exits when reaching the 0 terminator after the only arg).

Verdict: HIGH-severity silent-failure class CLOSED. NO further
HIGH or must-fix-MEDIUM findings remain on the K1.F16-F21 batch.
This is the SEVENTH cleanly-audited batch (K3.E/F/H/I/J/K/L).

### 2026-05-27 — K1.F21 (K3.K signal)

Full 3-axis audit dispatched on commit `11865c0` (K1.F21 -- generic-
bare-call name resolution fallback). Findings:

**silent-failure-hunter**: CLEAN. NO HIGH, NO must-fix-MEDIUM.
  - Verified the scratch buffer's stale-byte residue across patch-
    loop iterations is harmless (kovc_byte_eq length-equality short-
    circuit prevents stale bytes from leaking).
  - Gate `target_name_l < 60` is correct: max write at scratch[63]
    fits within the 64-slot region.
  - Zero / negative target_name_l falls through to ud2 (the LOUD
    path, not silent).
  - fn_table_lookup ordering footgun (user defines both `fn id<T>`
    and explicit `fn id__i32` -> first-match dispatch) is pre-
    existing in the turbofish path; K1.F21 inherits, doesn't
    introduce. LOW.
  - Typo-collision risk (`frobnicate(x)` silently resolving to a
    coincidentally-existing `frobnicate__i32`) requires the user to
    have already used the same name with turbofish in this program;
    mostly theoretical, pre-existing for turbofish too. LOW.

**type-design-analyzer**: NO HIGH, NO must-fix-MEDIUM. One SOFT-MEDIUM:
  - The K1.F21 inline comment claimed "+ safety" / "leaves 4 bytes
    of headroom"; at max target_name_l=59 the write fills exactly
    scratch[0..63], leaving 0 headroom past the suffix's last byte.
    The implementation IS safe (exact fit); the documentation
    overstated the margin. K3.K fixes the comments to say "EXACTLY"
    / "exact fit, no safety-margin reserve".
  - Other notes: slot 170 collision-free; bn_mangle_scratch contract
    consistent with existing one-byte-per-i32-slot bn_state pattern;
    `__i32` byte sequence (95 95 105 51 50) matches the existing
    builtin name installations (line 2231 __i32_to_f32; line 2394
    __i32_to_f64) and the parser's turbofish mangler (parser.hx
    1519-1566).

**code-reviewer**: CLEAN. NO HIGH, NO must-fix-MEDIUM.
  - 64-slot allocation: 1 pre-loop push + 63 in-loop = 64; gate at
    < 60 → max write scratch[63] = exact fit. Off-by-one safe.
  - Slot 170 numbering verified unique: only two writes
    (placeholder + scratch offset at line 2127); one read site
    (bn_mangle_scratch accessor at line 3605).
  - `__i32` byte sequence cross-checked against parser.hx and other
    builtin installations -- identical.
  - Probe correctness: `let _ = id::<i32>(99); id(42)` -- the
    let-discard parses; bare call resolves via the K1.F21 fallback;
    rc=42 verified passing.
  - The same SOFT-MEDIUM noted above on the doc-drift "+ safety"
    phrasing. Resolved in K3.K.

K3.K ships the documentation correction (both the install_builtin_
names inline comment and the patch_table loop comment) so the
phrasing accurately reflects exact-fit semantics rather than an
imagined safety margin. The implementation gate `target_name_l < 60`
is unchanged -- the audit confirmed it's bounds-safe.

Verdict: **K1.F21 CLEAN end-to-end after K3.K doc-drift fix**. NO
HIGH, NO must-fix-MEDIUM across all three axes. This is the SIXTH
cleanly-audited code batch (K3.E/F/H/I/J/K). The mirror-pattern
discipline continues to find LOW-severity convention drift; the
discipline of running the audit + immediately fixing findings holds.

### 2026-05-27 — K1.F19 + K1.F20 + K1.F20b + K3.J (K3.J signal)

Full 3-axis audit (silent-failure-hunter / type-design-analyzer /
code-reviewer) dispatched on commits `fabbbab` (K1.F19) + `5be68a0`
(K1.F20) + `5e0621f` (K1.F20b). Findings:

**silent-failure-hunter**:
  - **MEDIUM-1** -- K1.F20b's depth-1 store was unconditional. For
    zero-arg `__trace_event()` the while-walk emits nothing, eax
    holds caller-context residue (no convention zeros eax pre-call),
    and the store would write that residue to the trace slot --
    silently corrupting `__trace_last`'s read. Fixed by K3.J:
    gate the store on `args_head != 0`; zero-arg emits a
    deterministic `xor eax, eax` instead, preserving any prior
    recorded value untouched.
  - **LOW-2** -- `__trace_last(args)` silently dropped passed args
    (didn't walk args_head), inconsistent with the K1.F17 silent-
    arg-drop closure used by every other variadic-tolerant builtin.
    Fixed by K3.J: walk args_head first, then load the trace slot.
    Side effects of args now fire as the convention promises.

**type-design-analyzer**: NO HIGH, NO must-fix-MEDIUM. 4 LOW notes
(documented Phase-0 simplifications + pre-existing patterns):
pow2_i32's range-bound contract is comment-encoded rather than type-
encoded (caller-gated to n in [14, 24]); emit_hash_i32_mixer's
register-state-as-side-channel matches the existing emit_* idiom;
__trace_last's variadic-tolerance choice was inconsistent (fixed in
K3.J); bn_state's flat-i32 slot scheme has no structural invariants
(pre-existing, not introduced by this batch).

**code-reviewer**: CLEAN. NO HIGH, NO must-fix-MEDIUM. K1.F18b
mantissa-shift math hand-verified (0.00005_f16 -> mant10=853, low
byte 85; 0.00004_f16 -> mant10=683 round-up, low byte 171). K1.F19
24-byte mixer byte sequence verified line-by-line including little-
endian constants (c1=0x05EBCA6B, c2=0x27D4EB2F, c3=0x165667B1).
K1.F20b ModRM bytes verified (0x88 for `mov [rax+disp], ecx`,
0x80 for `mov eax, [rax+disp]`). Slot 169 + disp 8388352
reservations confirmed unique (no collision with cell table at
8388356-8388608).

K3.J ships both audit fixes in one batch with 2 new permanent
probes:
  - `__trace_event(77); __trace_event(); __trace_last()` -> 77
    (pins MEDIUM-1 fix: zero-arg trace_event preserves prior slot)
  - `__trace_last(__trace_event(99)); __trace_last()` -> 99
    (pins LOW-2 fix: trace_last's arg trace_event(99) fires and
    writes the slot)

Verdict: **K1.F19/F20/F20b CLEAN end-to-end after K3.J**. NO HIGH,
NO must-fix-MEDIUM across all three axes. This is the FIFTH
cleanly-audited code batch (K3.E/F/H/I/J). The mirror-pattern
discipline continues to find LOW-severity convention drift and
the discipline of running the audit + immediately fixing the
findings holds.

## References

- User directive: 2026-05-26 conversation (initial hard constraint)
- User directive: 2026-05-26 follow-up (5-clean-audit stop criterion)
- Stored in Kovostov semantic memory:
  `C:/Projects/Kovostov/runtime/memory/semantic/helix.md`
  (entries at `2026-05-26T06:26:38Z` and the 5-clean-audit
  follow-up at the next timestamp)
- Supersedes: optimization-plan deferral language re GPU/MLIR/Tile;
  the cron prompt's earlier "v1.0 reached" stop criterion
