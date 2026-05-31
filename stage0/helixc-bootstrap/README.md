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
- **3a — DONE:** minimal x86-64 ELF codegen + the compile pipeline. The seed now
  has two modes: no args → run the front-end self-tests (exit 42); `seed in.hx
  out.bin` → read the file, lex, parse, **emit a self-contained ELF** (ELF64
  header + one PT_LOAD R|W|X at 0x400000, code at 0x1000, `_start` calls main +
  sys_exit). Codegen handles a tail integer literal (`mov eax, imm32`). Verified
  end-to-end: the seed compiles `fn main() -> i32 { 42 }` → the output ELF runs
  and exits 42. **First runnable binary emitted by our own seed.**
- **3b — DONE:** integer expression codegen. `cg_bin` evaluates left→rax (push),
  right→rcx (pop), then emits the op: `+ - * / %`, bitwise `& | ^`, the six
  comparisons (cmp + setcc + movzx), and `&& ||` (non-short-circuit, sound for
  the side-effect-free subset). Verified: `6*7`→42, `2+3*4`→14 (precedence),
  `5>3`→1.
- **3c — DONE:** local variables. A per-function name→slot table (matched by
  source span); the prologue reserves a 16-aligned frame; `let`/assignment store
  to `[rbp - 8*(slot+1)]`, `ND_VAR` reads from it. `cg_stmt` runs the block's
  statements before the tail expression. Verified: `let mut x=41; x=x+1; x`→42,
  `let a=6; let b=7; a*b`→42.
- **3d — DONE:** control flow. `cg_while` (cond at top, `jz` past the loop,
  `jmp` back) and `cg_if` (cond, `jz` else, then-arm, `jmp` end, else-arm) with
  rel32 jump backpatching; `if` works as statement and value (both arms leave
  their result in eax); `collect_locals` now recurses into nested blocks. A
  `cg_block` helper runs a block's statements then its tail. Verified: while-sum
  0..8→36, `if x>5 {42} else {7}`→42, factorial-7 loop→5040 (exit 176).
- **3e — DONE:** function calls + params + multi-function. `codegen` emits every
  function, records each one's offset, and backpatches all call sites (forward
  refs resolved by name) plus the `_start`→`main` call. `cg_call` evaluates args,
  pushes them, pops into the SysV registers (rdi, rsi, …), and emits the call;
  `cg_fn` spills incoming param registers to their stack slots. Verified:
  `add(40,2)`→42, `sq(6)+6`→42, and **`fib(10)`→55 (recursion)**.
- **3f-arena — DONE:** the four arena intrinsics. A 128 MiB BSS arena per
  compiled program (reserved via `p_memsz`) at vaddr `0x1400000`; `__arena_push`/
  `get`/`set`/`len` emit inline against `r11` (cursor at base, slots at
  `base+4+i*4`). Functionally equal to kovc's RIP-relative arena (DDC washes out
  the difference at the fixpoint). Verified: push/get/set/len → 55.
- **3f-read — DONE:** string literals + `read_file_to_arena`. String literals
  (`ND_STR`) are emitted as NUL-terminated data after the code, referenced by a
  `mov rax, imm64` that's backpatched to the string's vaddr. `read_file_to_arena`
  emits open → read-into-1 MiB-stack-buffer → push each byte as one i32 into the
  arena → return the count (matching kovc's contract; the lexer reads
  `__arena_get(i)`). Verified: reads a 3-byte file → len 3, first byte 'A'(65) → 68.
- **3f-write — DONE:** `write_file_to_arena(path, off, len)` — opens the path
  (truncating), writes `len` bytes (the low byte of each arena slot off+k) one
  per `write(2)`, closes, returns the count. Matches kovc's contract (one byte
  per i32 slot). Verified: the seed compiles a program that pushes "Hi!" into the
  arena and writes it to disk → the file contains exactly "Hi!".
- **SUBSET COMPLETE.** The seed now implements the entire Helix self-hosting
  subset: lexer, parser, full codegen (literals, arithmetic, comparisons,
  bitwise, locals, while/if, calls, recursion), string literals, and all six
  intrinsics (arena + file I/O).
- **next (inc 4):** scale the pools and compile the real compiler —
  `lexer.hx` → `parser.hx` → `kovc.hx` — then the self-hosting fixpoint + DDC.

## M2-Planet C-subset notes (learned, so we don't re-hit them)

- **No global array definitions** in `--bootstrap-mode` — use a global pointer
  `calloc`'d at startup (as M2-Planet's own source does). `sizeof(T)` and
  `calloc(n, sizeof(T))` are supported.
