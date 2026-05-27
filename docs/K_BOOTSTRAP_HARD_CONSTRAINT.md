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
| mixed-type binops (unsigned u64↔u32) | ⏳ **WIP** | K1.F8d staged twice (this and prior tick); reverted both times because WSL was unreliable during validation. Code design is a copy-paste of K1.F8b/F8c (zero-ext via mov_ecx_eax already correct for unsigned, no movsxd needed; expr_type adds (9,6) and (6,9) cases). Ships next WSL-clean tick. |
| generic monomorphization | ❌ OPEN | Type erasure works for i32-shaped T (K1.F-discovery batch 27 via turbofish); full monomorphization for non-i32 T is the gap. |
| f16 bit-accurate | ❌ OPEN | _f16 lexes as bf16-shaped (K1.BH 2026-05-26); IEEE-754 half-precision bit pattern not yet emitted. |
| reflection (quote/splice/modify/reflect_hash) | ⚠️ STUB | Builtins registered at bn_state slots 118-120 + 164-168 (K1.F2/F3/F4 2026-05-26 + 2026-05-27); real semantics (writing reflection cells, hash computation) still pending. |
| tile ops (TILE_ZEROS/ADD/SUB/MUL/MATMUL) | ❌ OPEN | No tile codegen in bootstrap. Matrix rows 197-199 KOVC-MISSING. |
| GPU backends (PTX + ROCm + Metal + WebGPU) | ❌ OPEN | All four backend rows 200-201 KOVC-MISSING. |
| MLIR migration path | ❌ OPEN | v3.0 Phase E shipped on Python side (Stages 210-216); bootstrap port pending. Matrix row 202 KOVC-MISSING. |
| trace events | ⚠️ STUB | __trace_event slot 165 (K1.F3 2026-05-27, variadic-tolerant no-op stub); real trace-arena impl pending. |
| macros | ⚠️ PARSER-ONLY | `IDENT!(...)` parses as no-op call (K1.CB 2026-05-26); no macro expansion. |

Also closed this session: K2 parity corpus grew 70 → 85 entries
across K2.G (const-name probes) + K2.H (mixed-type binop probes),
pinning the K1.F7/F8/F8b/F8c closures across BOTH compilers.

Three of the user's enumerated Category-2 items are now fully
**CLOSED** end-to-end and parity-pinned (impl method dispatch,
field-store mutation, const-name resolution). Mixed-type binops
is mostly closed (signed i64↔i32 for all five arith ops); the
unsigned u64↔u32 leg lands the next WSL-clean tick.

## References

- User directive: 2026-05-26 conversation (initial hard constraint)
- User directive: 2026-05-26 follow-up (5-clean-audit stop criterion)
- Stored in Kovostov semantic memory:
  `C:/Projects/Kovostov/runtime/memory/semantic/helix.md`
  (entries at `2026-05-26T06:26:38Z` and the 5-clean-audit
  follow-up at the next timestamp)
- Supersedes: optimization-plan deferral language re GPU/MLIR/Tile;
  the cron prompt's earlier "v1.0 reached" stop criterion
