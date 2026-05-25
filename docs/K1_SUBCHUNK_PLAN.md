# K1 Sub-Chunk Plan

**Status:** scoping doc · **Date:** 2026-05-25 ·
**Parent:** [`HELIX_K_BOOTSTRAP_MASTER_PLAN.md`](HELIX_K_BOOTSTRAP_MASTER_PLAN.md)
· **Driver:** [`K_BOOTSTRAP_FEATURE_MATRIX.md`](K_BOOTSTRAP_FEATURE_MATRIX.md) §18

The K-bootstrap master plan staged the work as K0 (survey) → K1
(first ports) → K2 (parity harness) → K3 (seed) → K4 (cutover) →
K5 (DDC + 5-clean). K0 is complete (chunks 1 and 2, commits
`2a8c3bc` and `86d3f80`). This document scopes K1 into atomic
sub-chunks small enough to execute one per cron tick.

The matrix §18 "priority order" was directionally right but
under-scoped — each row turned out to need lexer + parser +
codegen + tests, not just codegen. This plan re-decomposes.

## Discipline

Each sub-chunk:
- One coherent commit.
- Touches `lexer.hx` / `parser.hx` / `kovc.hx` / tests as needed.
- 3-axis audit (silent-failure-hunter / type-design-analyzer /
  code-reviewer) on the diff.
- Tests via the existing `test_bootstrap_kovc_*` harness in
  `helixc/tests/test_codegen.py` — write a small `.hx` program
  that exercises the new feature, compile it through both
  Python `helixc` and (after this chunk lands) kovc, run, assert
  identical output.
- Commit + push + Telegram.

## Sub-chunks (in dependency order)

### K1.A — sub_rsp_imm32 / add_rsp_imm32 helpers (foundation)

**Goal:** add the two x86 instruction helpers `emit_sub_rsp_imm32(n)`
and `emit_add_rsp_imm32(n)`. They are the prerequisite for stack-
args > 6 (K1.B), for `unsafe`-block stack alloca (later), and for
the future closure / alloca primitives.

**Encoding:** `sub rsp, imm32` = `48 81 EC <imm32-LE>` (7 bytes);
`add rsp, imm32` = `48 81 C4 <imm32-LE>` (7 bytes).

**Files:** `kovc.hx` (add 2 helpers near the existing
`emit_add_rax_rcx_64` family at line ~300).

**Tests:** none yet — these are leaf helpers; K1.B exercises them.

**Estimated size:** ~30 lines of code, 1 commit.

### K1.B — SysV stack args > 6 in `kovc.hx` AST_CALL arm

**Goal:** replace the `if arg_count > 6 { trap 16002 }` at
`kovc.hx:6122` with a working SysV stack-args implementation.

**Architecture:** caller-cleanup pattern (matches `x86_64.py`):
1. Pass 1: push all args in source order (existing).
2. `stack_alloc = align_to_16((arg_count - 6) * 8)`.
3. `sub rsp, stack_alloc`.
4. For each arg i in 6..arg_count-1: `mov rax, [rsp+S]; mov [rsp+D], rax`
   where `S = stack_alloc + (arg_count-1-i)*8`, `D = (i-6)*8`.
5. Load args 0..5 into rdi/rsi/rdx/rcx/r8/r9 via `mov reg, [rsp+offset]`.
6. `call`.
7. `add rsp, stack_alloc + arg_count*8` (cleanup pushed args + alloca).

**New helpers needed in `kovc.hx`** (build on K1.A):
- `emit_mov_rdi_rsp_disp32(disp)` — 6 bytes (`48 8B BC 24 <d>`)
- `emit_mov_rsi_rsp_disp32(disp)` — 6 bytes (`48 8B B4 24 <d>`)
- `emit_mov_rdx_rsp_disp32(disp)` — 6 bytes (`48 8B 94 24 <d>`)
- `emit_mov_rcx_rsp_disp32(disp)` — 6 bytes (`48 8B 8C 24 <d>`)
- `emit_mov_r8_rsp_disp32(disp)`  — 7 bytes (`4C 8B 84 24 <d>`)
- `emit_mov_r9_rsp_disp32(disp)`  — 7 bytes (`4C 8B 8C 24 <d>`)
- `emit_mov_rax_rsp_disp32(disp)` — 6 bytes (`48 8B 84 24 <d>`)
- `emit_mov_rsp_disp32_rax(disp)` — 6 bytes (`48 89 84 24 <d>`)

**Tests:** add a `.hx` test fn that takes 8 i32 args, returns their
sum. Compile through Python helixc + via kovc + assert identical
ELF behavior (exit code = expected sum).

**Estimated size:** ~150 lines (8 helpers + pass-2 rewrite) + ~50
lines of test, 1 commit.

### K1.C — `return` statement (lexer + parser + codegen)

**Status (2026-05-25): FIRST ATTEMPT REVERTED.** The attempt
broke 24 bootstrap tests; tree restored to pre-K1.C state.
Lesson + redo plan below.

**Goal:** support the `return expr;` form. Today the bootstrap uses
the implicit-last-expression form only.

**Design:**
- lexer.hx: no changes — keywords are post-lex-classified in the
  parser via byte_eq against state-slot-installed bytes.
- parser.hx: install "return" keyword bytes in `install_keywords`,
  add accessors `kw_return_s/n`, add `parse_return` top-level fn,
  add a new arm in `parse_primary`'s IDENT dispatch.
- kovc.hx: in the codegen dispatch, add an arm for AST_RET (tag
  43): emit value into rax, then emit_epilogue + emit_ret (same
  bytes the fn-end currently emits at kovc.hx:6806).

**LESSON FROM FIRST ATTEMPT (2026-05-25):**

The first K1.C attempt added all the pieces above, ran the
bootstrap tests, and 24 of 25 tests failed with
`ParseError: expected expression (got KW_ELSE 'else')`. Tree
reverted via `git checkout`. Root cause analysis:

1. `parse_primary` has a NESTED if-cascade across token types
   (t == 2 = IDENT, t == 3, t == 4, ...). Inside each token-
   type arm, there's a further sub-cascade over keyword bytes
   (let / if / while / match / catchall).
2. The closing `}}}}` braces at `parser.hx:3708-3711` close
   sub-cascades for SPECIFIC token-type arms, with explanatory
   comments ("Stage 28.11 INC-3b.2: extra `}` closes the new
   nt==16 branch" etc.). The very-end `}}}}}}}}}}}}}}}}` at
   `parser.hx:3848` closes the OUTER token-type if-cascade.
3. The first K1.C insertion correctly added the new arm in
   parse_primary, and adjusted the closing-brace count at
   line 3848 by +2. But that's the WRONG closing location for
   the new arm — the match-arm's else-block closes at line
   ~3708, not at line 3848.
4. Net effect: the IDENT sub-cascade had unbalanced braces;
   subsequent token-type arms (t == 3 etc.) parsed in the
   wrong scope; Python helixc choked on the resulting
   ill-structured source.

**REDO PLAN (next K1.C attempt):**

Instead of touching parse_primary's brace cascade, factor the
keyword dispatch in a way that doesn't require new braces:

- Option (i): add a `parse_keyword_or_ident(tok_base, sb, id_s, id_l)`
  top-level fn that takes the IDENT bytes and dispatches to
  parse_let / parse_if / parse_match / parse_return /
  fall-through-to-var-ref. Replace the existing inline keyword
  cascade with one call to this. The brace surface is owned by
  the new fn, not by parse_primary. Cleaner but a much larger
  edit.
- Option (ii): add the new arm AFTER closing the existing
  keyword sub-cascade — but BEFORE the token-type cascade
  continues. Requires identifying the exact `}` that closes
  the IDENT sub-cascade and inserting the new arm there with
  matching braces.
- Option (iii): split the work — first add the AST_RET codegen
  + parse_return fn as DEAD CODE (no caller). Then in a
  follow-up chunk, do option (i) or (ii) carefully with
  per-line brace counts.

For autonomous-cron-driving safety, **option (iii)** is the
preferred next attempt. The dead-code chunk lands safely; the
parse_primary chunk gets its own audit-pre-commit verification.

**Tests:** after the wire-up, a fn that returns early from
inside an if.

**Estimated size:** dead-code chunk ~50 lines; wire-up chunk
~30 lines + tests. Two commits.

### K1.D — `print_int` builtin

**Goal:** kovc-compiled programs can call `print_int(n)` to write an
i32 to stdout.

**Design:** mirror the Python helixc backend's `__helix_print_int`
helper: convert i32 to ASCII via a digit-loop, then `write(1, buf, len)`
syscall.

- kovc.hx: in `try_emit_builtin_call`, recognize a call to fn name
  "print_int" with one arg, and emit the ASCII-conversion + syscall
  sequence inline (~50 bytes of code).

**Tests:** `fn main() -> i32 { print_int(42); 0 }` — verify stdout
contains "42".

**Estimated size:** ~100 lines, 1 commit.

### K1.E — functional string literals (codegen)

**Goal:** AST_STR_LIT (tag 25, already parsed) becomes usable. Today
its codegen emits `mov eax, 0` — useless except as the first arg to
file builtins.

**Design:** strings live in a `.rodata`-style section appended to the
ELF (or in the existing arena, addressed by a runtime pointer). Two
sub-options:
- (a) **Arena-resident strings**: at codegen time, emit the string
  bytes into the arena, get back an arena slot index; return
  `(slot_index, length)` as a 16-byte pair (low 32 = slot, high 32
  = length). Compatible with kovc.hx's i32-only scalar world.
- (b) **ELF .rodata**: emit the bytes into a separate ELF section,
  return a `(ptr_va, length)` virtual-address pair. Closer to how
  Python helixc does it.

Pick (a) — arena-resident — for simplicity and consistency with the
rest of kovc.hx.

**Subtle bit:** Helix strings are normally fat-pointers
`(ptr: i64, len: i64)`. Kovc.hx uses i32 throughout; the v3.0
matrix-§3 noted bootstrap's strings are limited. Carrying the (slot,
len) i32 pair is a deliberate simplification — matches how
`read_file_to_arena` already returns a count.

**Tests:** a fn that takes a string, returns its length via a future
`str_len` builtin (or just returns a hardcoded value compared to
the string). Phase 1: just emit the bytes correctly; phase 2 (later
chunk): add `str_len`, `str_eq`, etc.

**Estimated size:** ~120 lines, 1 commit. Defers `str_*` builtins
to follow-up chunks.

### K1.F — tuple literal + `.field` access codegen

**Goal:** AST_TUPLE_LIT (tag 50) + AST_TUPLE_FIELD (tag 52) already
parse. Today both codegen-trap (`99001`). Make them work.

**Design:** tuples as N-slot stack regions. A tuple expression
allocates N stack slots, evaluates each element into rax then
`mov [slot], rax`. Tuple field access reads `[base_slot + N*8]`.

**Tests:** `fn main() -> i32 { let t = (10, 20, 30); t.0 + t.2 }`
should exit with 40.

**Estimated size:** ~150 lines, 1 commit.

### K1.G — `for x in Range` (lexer + parser + codegen)

**Goal:** support `for x in 0..10 { ... }`.

**Design:** desugar `for x in a..b { body }` into:
```
{
  let mut x = a;
  while x < b {
    body;
    x = x + 1;
  }
}
```

at the parser level. No new codegen — the `while` + `let mut`
arms already work.

**Lexer additions:** TK_FOR, TK_IN, TK_DOTDOT.
**Parser:** AST_FOR → desugars during parsing (no new tag, just
synthesize while + let).

**Tests:** `for i in 0..5 { sum = sum + i }` exits with 10.

**Estimated size:** ~100 lines, 1 commit.

### K1.H — `break` / `continue` / `loop`

**Goal:** loop-control statements.

**Design:**
- `loop { body }` desugars to `while 1 { body }`.
- `break` / `continue` require labeled jump targets — same patch-
  table pattern as `return` (K1.C option b). The innermost loop's
  start + end labels live in `bind_state`.

**Tests:** `for i in 0..10 { if i == 5 { break } }` exits with 5
result.

**Estimated size:** ~120 lines, 1 commit.

### K1.I — `Cast` (`expr as T`)

**Goal:** `let x: i64 = (42 as i64)` etc.

**Design:**
- Lexer: `as` keyword → TK_AS.
- Parser: AST_CAST (tag 44) carries inner expr + target type tag.
- Codegen: based on (src_ty, dst_ty), emit appropriate widen /
  narrow / int↔float conversion instructions.

**Tests:** `let x = (1.5_f32 as i32); x` exits with 1.

**Estimated size:** ~100 lines, 1 commit.

### K1.J — `const` declarations (top-level)

**Goal:** `const X: i32 = 42;` at module level.

**Design:**
- Lexer: TK_CONST.
- Parser: AST_CONST_DECL (tag 45). Eagerly evaluate the RHS at
  parse time (it must be a constant expression).
- Codegen: a const-name lookup table; identifiers that match
  resolve to their pre-evaluated values.

**Tests:** `const N: i32 = 5; fn main() -> i32 { N * 2 }` exits
with 10.

**Estimated size:** ~80 lines, 1 commit.

## After K1.A-J: status check

When all 10 sub-chunks land, the matrix's "PARITY" count moves
from ~28 to ~38 rows; KOVC-MISSING drops from ~115 to ~105. About
9% gap closure.

K1 continues iteratively from there. The remaining big-ticket
items per the matrix:
- All 8 patterns (PatLit / PatBind / PatWildcard / PatTuple /
  PatOr / PatRange / PatVariant / PatStruct) — requires Match.
- Structs (basic, non-generic, then parametric).
- Enums.
- AGI metaprogramming (quote / splice / modify / reflect_hash).
- Type-system wrappers (15 of them).
- AD framework.
- Tile / tensor / GPU.

Each is a multi-chunk effort. The pace through K1 calibrates how
long the whole port takes.

## What this plan does NOT cover

- **K2 (parity harness)** — separate, starts in parallel after a
  few K1 ports land.
- **K3 (trusted seed)** — separate, can start any time.
- **K4 (cutover)** — gated on K1/K2 being complete + user TG
  confirmation.

## Decision: this iteration's chunk

This document IS this iteration's chunk — a scoping deliverable.
Next cron tick picks up K1.A (the smallest, foundation sub-chunk:
add two x86 instruction helpers, no behavior change yet, sets up
K1.B).
