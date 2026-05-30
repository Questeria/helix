/* SPDX-License-Identifier: Apache-2.0
 * helixc-bootstrap seed -- the trusted Helix-subset bootstrap compiler.
 *
 * This is the first ORIGINAL rung of the Kovostov-Native ladder (everything
 * below it is hand-authored hex0 or vendored stage0/M2-Planet sources). It is
 * a small C program, written in the M2-Planet C subset so our stage0 ladder
 * (hex0 -> ... -> cc_amd64 -> M2-Planet) can compile it WITHOUT any external
 * toolchain. Its job: compile the Helix self-hosting subset that helixc
 * (helixc/bootstrap/kovc.hx + parser.hx + lexer.hx) is itself written in, so it
 * can mint the first helixc WITHOUT Python -- replacing Python as the K1 minter.
 *
 * Apache-2.0, deliberately kept statically separable from the GPL-3.0 vendored
 * stage0/M2-Planet + M2libc trees (we only BUILD with those; none of their
 * source is copied into this file).
 *
 * Design (from docs/K_TASK0_HELIX_SUBSET_FINDINGS.md): i32-only value model;
 * one global integer arena is the entire heap; the Helix subset is
 * while + if-as-expression + recursion + six intrinsics. So this compiler is
 * built the same way: a single global int arena, plain functions, while loops.
 *
 * INCREMENT 0 (this commit): project + build-pipeline proof + the arena core.
 *   Proves (a) our own Apache-2.0 C compiles + runs through M2-Planet, and
 *   (b) the global-arena primitive (the heart of every later stage) works.
 *   Later increments add: lexer, recursive-descent parser, x86-64 ELF codegen.
 */

/* ----- the global arena: one flat int buffer, bump-allocated, never freed -----
 * M2-Planet's --bootstrap-mode forbids global ARRAY definitions, so (exactly as
 * M2-Planet's own source does) the arena is a global POINTER calloc'd at start.
 * Sized small for increment 0; grows to multi-MB once it must hold kovc.hx. */
int* ARENA;
int ARENA_LEN;

int arena_init() {
    ARENA = calloc(4096, sizeof(int));
    ARENA_LEN = 0;
    return 0;
}

/* append v; return the index it landed at (matches Helix __arena_push) */
int arena_push(int v) {
    int idx;
    idx = ARENA_LEN;
    ARENA[idx] = v;
    ARENA_LEN = ARENA_LEN + 1;
    return idx;
}

int arena_get(int i) {
    return ARENA[i];
}

int arena_set(int i, int v) {
    ARENA[i] = v;
    return 0;
}

int arena_len() {
    return ARENA_LEN;
}

/* ----- increment-0 self-test: exercise the arena, return 42 -----
 * Pushes 6, 7, 29; sums them back via a while loop (-> 42); mutates one slot
 * to prove arena_set. Returns the sum as the process exit code. This is the
 * smallest program that proves the build pipeline AND the core data structure.
 */
int main() {
    int i;
    int sum;

    arena_init();
    arena_push(6);
    arena_push(7);
    arena_push(28);
    arena_set(2, 29);   /* 28 -> 29, proves arena_set: now 6 + 7 + 29 */

    i = 0;
    sum = 0;
    while (i < arena_len()) {
        sum = sum + arena_get(i);
        i = i + 1;
    }
    return sum;   /* 6 + 7 + 29 = 42 */
}
