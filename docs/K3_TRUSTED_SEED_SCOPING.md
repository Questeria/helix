# K3 trusted seed — scoping (2026-05-30, counter 424)

> **v1.3 V6 cross-ref (2026-06-04):** the **full trusted-C inventory** (every committed
> `.c`/`.h`: role / LOC / in-or-out of the fixpoint / on-build-path-or-dead / portable-or-
> irreducible), the V6 **dead-C prune** (6 dead duplicate `M2libc/bootstrappable.{c,h}` removed —
> the 3 `.c` byte-identical to the canonical `M2-Planet/M2libc/bootstrappable.c`, the 3 `.h`
> identical to each other with no canonical `.h` in the tree; proven safe by rebuilding all 3
> mescc-tools GREEN from their `build.sh` without the files plus the full main gate GREEN after the
> prune), and the precise trusted-C boundary now live in **`docs/TRUSTED_C_INVENTORY.md`**. This
> scoping doc remains the deep-dive on the seed (the irreducible root) specifically;
> `TRUSTED_C_INVENTORY.md` is the repo-wide C surface.

Scoping the GATING build-order item "K3 trusted seed" (the trust root that lets
the Python compiler `helixc/` be deleted). Three read-only agents mapped the
Python build path, the intent/roadmap, and the bootstrap's self-rebuild gaps;
every load-bearing claim below was **independently re-verified** by reading the
actual code (lesson 17/19).

## Headline — the bootstrap ALREADY self-hosts; Python's only trust-chain role is minting K1

`helixc/tests/test_self_host_fixpoint.py` proves it, and I read it directly:
- **P0 (Python) builds K1 only** — `k1_elf = _compile_src_to_elf(k1_driver)`
  (fixpoint line 57). `_compile_src_to_elf` (test_codegen.py:7121) runs the
  **Python `helixc` package** (parse → flatten → monomorphize → grad → lower →
  fold/cse/dce → `compile_module_to_elf`, backend `x86_64.py:4784`). This is the
  entire Python trust-root surface.
- **K1 (a bootstrap BINARY) self-compiles the FULL ~1.43 MB driver → K2**
  (fixpoint lines 61-64): K1 is merely *run* over `lexer(stripped)+parser+kovc(
  stripped)+main` (673 fns). No Python.
- **K2 == K3 BYTE-IDENTICAL** (`test_self_host_fixpoint_byte_identical`, lines
  79-116; `cmp -s` → IDENTICAL; 566 963 bytes). The bootstrap reproduces itself
  exactly.

So "delete Python" does **not** require re-implementing the Python backend: a
bootstrap binary already performs the equivalent `lex → parse_top →
emit_elf_for_ast_to_path` on the full source. Python only mints the *first*
link. The remaining work is (1) make that self-compile run **unattended**, and
(2) establish a **trust root** that does not begin with the Python-built K1.

## Two senses of "trusted seed" (the task term)

`docs/HELIX_K_BOOTSTRAP_MASTER_PLAN.md:24-39` defines the final product; the
phrase "trusted seed" is condition (3): *a documented small hand-assembled
stage-0 (hex-blob/ELF) recompiles the whole chain; no Python in the trust
chain*, DDC-validated (Diverse Double-Compiling, dwheeler.com/trusting-trust).
Build order K0→K5 (master plan §5): K0 survey ✔, **K1 feature-catchup (NOW)**,
K2 parity harness, **K3 seed (PENDING)**, K4 cutover (user-gated), K5 DDC + 5
clean audits. `scripts/helix_status.py` PYTHON_DELETION_BUCKETS lists **K3
trusted-seed bootstrap** + the **5-clean-audit gate** as the two `pending`
gating buckets.

Decompose "trusted seed" into:
- **(A) The full stage-0 trust root (K3-proper)** — a `hex0`/GNU-Mes/minimal-C
  stage-0 that bootstraps from raw binary with zero Python. `stage0/hex0/` is
  **design-stage only** (READMEs, no working chain). helix_status.py calls this
  "a MAJOR DEFERRED effort, NOT 60s-tick-tractable." Needs DDC (≥2 independent
  paths → bit-identical `kovc`). **Defer — this is a multi-session structural
  build, likely user-steered.**
- **(B) The near-term Helix self-rebuild seed driver** — a `.hx` program
  (compiled by the bootstrap) that reads the 3 source files, assembles +
  compiles them, and self-validates by running the result, with **no Python in
  the loop**. This is **~80% supported today** and is the tractable path that
  *advances* toward (A). Gaps below.

## What is already CLOSED (do NOT re-investigate)

- **Arena cap** — a prior "full driver = ~130% of cap → overflow" finding is
  STALE. The cap was rescaled `2097152 → 6291456` i32 slots (24 MB BSS;
  `kovc.hx:3984` `helix_arena_cap()`, mirrored `_shared_constants.py:63`; commit
  `d14286e`). Post-rescale margin ~2.3×; the byte-identical fixpoint over the
  full driver passes, so the full source+tokens+AST+~566 KB ELF fit in one
  arena.
- **Read buffer** — `read_file_to_arena`'s per-call buffer is **4 MB**
  (`kovc.hx` emit body immediates `4194304`; the inline "1M" comments are
  stale), ~3.5× headroom over the largest file (parser.hx 852 KB). Commit
  `0ee8824`. Multi-file **concatenation is free**: `read_file_to_arena` appends
  at the live arena cursor, so three sequential reads leave the files contiguous
  — no concat primitive needed.
- **Process builtins** — `run_process` (slot 179), `set_exec` (slot 180),
  `read_file_to_arena`, `write_file_to_arena` all EXIST (strlit-path only). The
  chain `write_file_to_arena → set_exec → run_process` is proven end-to-end
  (test_codegen.py ~8340).

## THE blocker — big-stack entry stub ("bug #1")

The emitted compiler **overflows the default 8 MB stack** when it self-compiles
the full 1.43 MB source: the bootstrap parser is deeply recursive
(`parse_primary` ~1241 lets; `parse_expr`/`parse_let` recurse one host frame per
statement — task 15). The emitted ELF's `_start` jumps straight to `0x401000`
and every fn prologue is just `push rbp; mov rbp,rsp; sub rsp,4096` on the
kernel-default stack — there is **no `mmap`/`setrlimit` big-stack setup anywhere
in kovc.hx** (grep: none). So the fixpoint tests only pass under an **external
`ulimit -s unlimited`** (fixpoint line 21), and the canonical
`test_bootstrap_kovc_self_host_loop` (test_codegen.py:6725) is **SKIPPED**
(6776-6777: "Unskip once the entry-stub big-stack fix lands so no external
ulimit [is needed]"). A standalone seed run by the user (no pytest, no ulimit
wrapper) would **SIGSEGV**.

This is the #1 tractable compiler chunk: an `_start` entry stub that **`mmap`s a
large stack and switches `rsp`** before calling `main`. It (a) unblocks
**unattended** full self-compile, (b) lets the skipped canonical self-host test
be **unskipped** (a concrete green-test acceptance criterion), (c) **subsumes
task 15** for practical depths (a big stack absorbs the parser recursion), and
(d) is a prerequisite for seed (B). Fixpoint-safe-by-construction is NOT
automatic here — the entry stub changes EVERY emitted binary's prologue path —
so it needs the full gate (fixpoint 2 + broad parity 403).

## Decomposed chunks (priority order)

1. **Big-stack entry stub** (kovc.hx `_start`/entry emission). Emit `mmap`
   (9, addr=0 len=~256-512 MB prot=RW flags=MAP_PRIVATE|ANON|GROWSDOWN|STACK) +
   set `rsp` to the new top, then `call main` (preserve the exit-code path:
   `mov edi,eax; mov eax,60; syscall`). MODERATE x86 codegen, HOT (every
   binary). Gate: fixpoint 2 + broad 403; then **unskip
   test_bootstrap_kovc_self_host_loop** as the acceptance test. This is the next
   tick's primary.
2. **Helix self-rebuild seed driver** (`.hx`, e.g. `helixc/bootstrap/seed.hx` +
   a test). Reads lexer/parser/kovc (append-concat), compiles, set_exec +
   run_process the result, branch on exit. Two easy decisions: **order the
   driver `main` FIRST** in the concatenation so `resolve_program_root`
   (kovc.hx:9524, picks the first `main`) selects it and the two demo `main`s
   (lexer.hx:1082, kovc.hx:12120) become dead code — **avoids needing an
   in-Helix `rsplit`/`strstr` (which the bootstrap lacks)**; and **bake literal
   paths** (write/set_exec/run_process are strlit-only). SMALL once chunk 1
   lands.
3. **(Complementary) task 15** — iterativize `parse_expr`/`parse_let`. Lower
   priority after chunk 1 (the big stack handles the depth); still valuable for
   robustness + to drop the stack requirement.
4. **(A) full stage-0 trust root + DDC** — `hex0`/Mes/C. MAJOR deferred,
   user-steered; the real "K3-proper". Not loop-tractable in 60 s ticks.

## Pointers
- Fixpoint (read directly): `helixc/tests/test_self_host_fixpoint.py` (P0→K1
  line 57; K1→K2 line 62; K2==K3 byte-identical lines 79-116; ulimit line 21).
- Skipped canonical self-host: `test_codegen.py:6725` (skip reason 6776-6777).
- `_compile_src_to_elf` test_codegen.py:7121; `_self_host_driver` 7139; the
  driver `k1_main` 7160-7171. Backend `x86_64.py:4784`.
- Builtins: `read_file_to_arena` kovc.hx ~4033/dispatch ~5318; `write_file_to_arena`
  ~4157; `run_process` ~4281 (slot 179); `set_exec` ~4338 (slot 180);
  `resolve_program_root` ~9524; `emit_elf_for_ast_to_path` 9574; entry stub /
  `emit_prologue` ~932.
- Arena cap `kovc.hx:3984` = `_shared_constants.py:63` (must stay equal).
- Intent: `docs/HELIX_K_BOOTSTRAP_MASTER_PLAN.md` (§5 K0-K5),
  `docs/K_BOOTSTRAP_HARD_CONSTRAINT.md` (acceptance criteria),
  `scripts/helix_status.py` (PYTHON_DELETION_BUCKETS).

## Progress update (2026-05-30)

- **Chunk 1 (big-stack entry stub) DONE** (commit ae3ea64, task 16): every
  bootstrap-emitted binary mmaps a 512 MiB stack at `_start`, so the self-compile
  runs without external `ulimit`.
- **Chunk 2 (Helix self-rebuild seed driver) DONE** (test
  `test_bootstrap_seed_driver_self_rebuild`). **Two corrections to the chunk-2
  sketch above, found by probing (the sketch was optimistic):**
  1. The seed is NOT a standalone main. To call `lex`/`parse_top`/
     `emit_elf_for_ast_to_path` it must INCLUDE the full compiler source —
     `seed = [lexer + parser + kovc + seed_main]` — else those calls are
     unresolved and the binary `ud2`s (SIGILL 132).
  2. It must be built by K1 (the BOOTSTRAP), NOT the Python reference compiler:
     `seed_main` uses `run_process`/`set_exec`, which exist only in `kovc.hx`
     (the Python backend raises `NotImplementedError: unknown function
     'set_exec'`). K1 needs `ulimit` to build the 1.43 MB seed (it is Python-
     built, no stub), but the seed BINARY carries the chunk-1 stub so it RUNS
     without `ulimit`.
  Verified end-to-end with NO external `ulimit`: the 606 KB bootstrap-built seed
  reads the 3 source files (free append-concat), prepends a driver main placed
  FIRST (so `resolve_program_root` picks it; the demos become dead code — no
  in-Helix `rsplit` needed), compiles them into a 593 KB K-next, then has K-next
  compile `6*7` and runs the result → exit 42.
- REMAINING toward the trust root: (a) the full stage-0 (hex0/Mes/C) — the major
  deferred, user-steered effort; (b) optionally promote the seed driver from a
  generated test to a committed `helixc/bootstrap/seed.hx`; (c) task 17 (Python-
  backend stub) to unskip the canonical self-host loop.
