# Task 0 findings — the Helix subset the compiler uses, and the helix-libc surface

**Status:** read-only analysis complete (user-greenlit; no original code written). This
data decides Option A vs B for the helix-top bridge. **It decisively favors A.**

**Date:** 2026-05-30. HEAD c7d032e. Method: two parallel read-only agents over
`helixc/bootstrap/{lexer,parser,kovc,evaluator}.hx` (29,532 lines total).

## Finding 1 — the compiler writes itself in a *tiny* Helix subset

Despite the language's size, the compiler's OWN source uses almost none of it. The
self-hosting subset is **integer-only, single-arena, `while` + `if`-expression +
recursion, six intrinsics**. Concretely:

**USED (the seed must support these):**
- `fn` with `i32` params/return (i32 is ~97% of all annotations); direct calls; deep mutual recursion (incl. recursive `parse_expr`).
- `let` / `let mut` (typed or inferred-to-i32); plain assignment `x = expr;`.
- `if` / `else` **as both statement and value-producing expression** (implicit tail-return; cascades nested up to ~50 deep — this replaces `match`).
- `while` (the ONLY loop form; exit via a `mut keep = 1; … keep = 0;` sentinel — no break/continue).
- Integer operators: `+ - * / %`, `== != < <= > >=`, `& | ^`, `&& ||`, parens, `0x..` hex, the `0 - x` negation idiom.
- `//` comments (and ideally nested `/* */`); string literals **only** as the path arg of the two file intrinsics.
- ONE global `i32` arena = the entire heap; everything (source bytes, tokens, AST, symbol tables, output ELF bytes) is hand-encoded as integer offsets into it.

**NOT USED (the seed can skip entirely — verified absent from compiler logic):**
structs, enums/tagged-unions, generics, traits, impls, type aliases, modules/`use`,
`const`/`static`, closures/nested-fns, **`match` + all pattern kinds**, `for`, `loop`,
`break`/`continue`, ranges, indexing `a[i]`, field access `.f`, method calls, compound
assignment, shift ops, unary minus/not, casts, **Result/Option/`?`/panic**, string
escapes, and all 60+ float/GPU/autodiff/reflection builtins. **The seed needs only `i32`
as a real type.**

**The six intrinsics the seed MUST implement** (everything else the source merely
*recognizes and emits code for* — the seed never runs those):
| Intrinsic | Calls | Role | Difficulty |
|---|---|---|---|
| `__arena_push(v)->i32` | 2094 | append to global i32 arena, return index | MEDIUM |
| `__arena_get(i)->i32` | 1798 | arena load | EASY |
| `__arena_set(i,v)` | 726 | arena store | EASY |
| `__arena_len()->i32` | 102 | bump cursor | EASY |
| `read_file_to_arena(path)` | 1 | open/read a file (path = string literal) | HARD (syscalls) |
| `write_file_to_arena(path,off,len)` | 1 | open/write/chmod | HARD (syscalls) |

**Hardest 5 seed features** (these dominate complexity): (1) the two file-I/O intrinsics
via raw syscalls; (2) `if`/block as a value-producing expression with implicit
tail-return; (3) the large global i32 arena with correct push-returns-old-index
semantics; (4) `while` + mutable locals + correct nested scoping/shadowing; (5) tolerating
the source's deep mutual-recursion. **Overall seed-size verdict: SMALL** (borderline
small/medium) — "a few thousand lines of tiny C," dominated by repetition, not feature
breadth.

**Optional trim (a tiny Option-C sliver):** ~50–90 lines of `f32`/`f64`/wide-int *self-test*
functions are the only place non-i32 leaks into the source. Stripping them lets the seed be
strictly i32-only (no IEEE float parsing/arith) and keeps it firmly SMALL. This is a small,
optional simplification — NOT a compiler refactor.

## Finding 2 — there is essentially **no helix-libc to write**

`kovc` emits a **fully self-contained static ELF**: its own `_start` (the 512 MiB-mmap
big-stack stub), direct Linux syscalls, **no external libc, no dynamic linker**. A compiled
Helix program needs nothing but the Linux kernel.

Every runtime primitive is **emitted inline by kovc** — the arena ops, `print_int`/`print_str`,
`panic`, `read_file_to_arena`/`write_file_to_arena`, `run_process`/`set_exec`, SSE float ops,
tile ops, prologue/epilogue, the BSS arena region. Syscalls used: `read(0) write(1) open(2)
close(3) mmap(9) fork(57) execve(59) exit(60) wait4(61) chmod(90)`.

**helix-libc verdict: NONE for the bootstrap.** (Only general, non-bootstrap programs would
ever want a libc — for malloc/free, hardware transcendentals, or networking — and even those
have pure-Helix stdlib coverage in `helixc/stdlib/`.) So the helix-top is really just **one
artifact: the seed**. There is no separate runtime to author.

## Decision: **Option A** (one small C seed), with the optional i32-only trim

The data collapses the A-vs-B tradeoff:
- The seed is **SMALL**, so A's "one C seed compiles kovc.hx directly" is tractable and a
  single, focused audit target.
- **B is now pointless.** B's whole rationale was "the subset is too big for one seed, so
  tier it / grow in Helix." But the subset is *already* at the floor — there is nothing to
  grow. Tiering would mean writing throwaway intermediate Helix compilers to bridge a
  tiny-C language up to a barely-bigger one. Pure overhead.
- **helix-libc is a non-task** (Finding 2), so the only original work is the seed itself.
- Purity is preserved: the non-Helix footprint is just one small, archived C seed — B would
  not meaningfully shrink that, because the subset floor is already minimal.

On top of A goes the **diverse-double-compile check**: build helixc via the seed AND via the
existing route, compare the self-hosting fixpoint byte-for-byte; a match retires Python with
proof.

## The seed spec (what to build, when greenlit)

A small C program (M2-Planet C subset, so our stage-0 ladder compiles it) that:
1. Lexes the Helix subset above (`//` comments, idents, i32 + `0x` literals, the operator
   set, string literals for paths, `@attr` skipped).
2. Recursive-descent parses: `fn`, `let`/`let mut`, assignment, `if/else` (stmt + expr),
   `while`, blocks-as-expressions/implicit-return, calls.
3. Emits code via the SAME six intrinsics + i32 model — simplest correct path (it only has
   to compile `kovc.hx`, not be fast).
4. Implements the six intrinsics (arena = one big `int32_t*` bump buffer; the two file
   intrinsics via `open/read/write/close/chmod`).
5. Compiles `kovc.hx` → helixc (K1'), then fixpoint K1'→K2', then DDC vs the Python K1.

## Next step

Awaiting user green-light to begin writing the Option-A seed (the first original,
Apache-2.0 code of the helix top). Until then, no original code is written.
