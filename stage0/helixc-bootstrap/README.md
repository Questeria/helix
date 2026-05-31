# stage0/helixc-bootstrap — the trusted Helix-subset seed compiler

**Apache-2.0. This is the first ORIGINAL rung of the ladder** — everything below
it is hand-authored `hex0` or vendored stage0/M2-Planet sources; from here up is
our own code. Kept statically separable from the GPL-3.0 vendored trees (we only
*build* with M2-Planet; none of its source is copied here).

## Why this exists

`helixc` (the real compiler, `helixc/bootstrap/kovc.hx` + `parser.hx` + `lexer.hx`)
is written in Helix and self-hosts — but today its *first* build (K1) is minted
by a **Python** reference compiler. Python is the last untrusted link and the
thing the project's hard constraint says must be deleted (K4).

This **seed** replaces Python. It is a small C program — written in the M2-Planet
C subset so our stage-0 ladder (`hex0 → … → cc_amd64 → M2-Planet`) can compile it
with **no external toolchain** — that compiles the *tiny Helix subset* `kovc.hx`
is itself written in. That lets us mint the first `helixc` from raw binary, with
no Python anywhere in the trust chain.

Why a *seed* and not a port of the whole compiler: per
`../../docs/K_TASK0_HELIX_SUBSET_FINDINGS.md`, the compiler self-hosts in an
astonishingly small subset — **i32-only, one global arena, `while` +
`if`-as-expression + recursion, six intrinsics** (`__arena_push/get/set/len`,
`read_file_to_arena`, `write_file_to_arena`); zero structs/enums/generics/match/
closures. So the seed only has to be a compiler for *that* subset, not all of
Helix. And because `kovc` emits a fully self-contained ELF, **there is no
separate helix-libc to write** — the seed is the only original artifact.

## The plan (Option A, user-approved 2026-05-30)

```
M2-Planet ──builds──▶ seed ──compiles──▶ helixc (K1′) ──compiles──▶ helixc (K2′)
  (rung 7)            (this)              (kovc.hx)        fixpoint: K2′ == K1′
```
Then **diverse double-compile**: compare the seed-built helixc against the
Python-built helixc at the self-hosting fixpoint; a byte-for-byte match retires
Python with proof.

## Build & test (under WSL)

```
wsl -e bash -c "cd /mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap && bash build.sh"
```
`build.sh` runs M2-Planet over `seed.c` → M1 → (catm + M0 + hex2) → `seed.bin`
(a self-contained ELF), then `run_tests.sh`. The source `seed.c` is the
committed artifact; `seed.bin` is a build output (git-ignored) until the seed is
complete, at which point its final `.bin` + `.sha256` get pinned like every
other rung.

## Increments

- **0 — DONE:** project + build-pipeline proof + the global-arena core
  (`calloc`'d int buffer + push/get/set/len; self-test sums to 42).
- **1 — DONE:** lexer. Tokenizes the Helix subset into stride-4 token records
  (tag, val, start, len): `//` + nested `/* */` comments, identifiers +
  keywords (fn/let/mut/if/else/while/return), i32 decimal + `0x` hex literals,
  string literals, the full operator/punctuation set, and skips `@attr`.
  Self-test lexes `fn main() -> i32 { let x = 41; x + 1 }` and asserts the
  17-token stream.
- **2a — DONE:** expression parser. AST nodes in a stride-5 int pool
  ({kind, a, b, c, next}); full precedence ladder (`||` < `&&` < `|` < `^` < `&`
  < `==`/`!=` < rel < `+`/`-` < `*`/`/`/`%`), unary minus via `0 - x`, parens,
  and calls with `next`-chained args. Self-test asserts precedence, parens
  override, and a call AST.
- **2b — DONE:** full parser. Statements (`let`/`let mut`, assignment, `while`,
  `if` as both statement and value), blocks (stmts chained via `next`, last bare
  expr = the block's value), `fn` with params, and `parse_program` (list of
  fns). `if` is reachable from `parse_primary`, so it works as an expression too.
  Self-test parses whole functions and asserts the let-mut/assign/tail-expr and
  while/if-expression AST shapes. **The parser is complete.**
- **next (3):** x86-64 ELF codegen — compile `fn main() -> i32 { 42 }` to a
  self-contained ELF that runs and exits 42; then grow codegen across the subset
  + the 6 intrinsics until it compiles `kovc.hx`.

## M2-Planet C-subset notes (learned, so we don't re-hit them)

- **No global array definitions** in `--bootstrap-mode` — use a global pointer
  `calloc`'d at startup (as M2-Planet's own source does). `sizeof(T)` and
  `calloc(n, sizeof(T))` are supported.
