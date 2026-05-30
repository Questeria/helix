# Test-infra port — scoping (2026-05-30, counter 415→416)

Scoping the build-order item "test-infra port": replace the Python pytest
harness so no Python remains at v1.0 (hard constraint: end state FULLY in
Helix). 93 files / ~116k lines in `helixc/tests/`. Two read-only agents mapped
it; the load-bearing blocker below was **independently re-verified** by grep.

## Headline finding — BLOCKED on compiler capability, not corpus porting

A test runner must **compile a `.hx` program → RUN the executable → read its
exit code → loop over a corpus**. The bootstrap compiler (`kovc.hx`) emits a
**5-syscall world only**: `read(0)`, `write(1)`, `open(2)`, `close(3)`,
`exit(60)` (18 `0F 05` sites, all verified). It has **NO `execve`(59),
`fork`(57)/`clone`(56), or `wait4`(61)** — so a compiled Helix program
**cannot launch another program or capture its exit status**. It also has **no
argv, no env, no stdin**, and file paths must be **compile-time string
literals** (the `read_file_to_arena`/`write_file_to_arena` builtins `ud2`-trap a
non-`AST_STR_LIT` first arg; 64 strlit slots cap). How the Python harness runs
an ELF today: emit ELF → write to file → `subprocess.run(["wsl","-e","bash",
"-c","chmod +x {p} && {p}"])` → read `result.returncode`. A Helix runner must
replicate the exec+wait, which today's bootstrap cannot emit.

**=> The test-infra port is a TWO-PHASE effort, gated on Phase T1.**

## Phase T1 — process-execution builtins in the bootstrap (PREREQUISITE)

New `kovc.hx` builtins (each: new builtin-name slot + dispatch + x86 syscall
emission; full fixpoint + broad-regression gate; verify with a tiny probe):
1. **`run_process(path_strlit) -> i32`** (MVP): `fork`(57)/`clone`(56) +
   `execve`(59) + `wait4`(61), returning the child's decoded exit code. Mirror
   the existing `read_file_to_arena` syscall-emission style (inline
   `emit_byte(0x0F); emit_byte(0x05)`, no wrapper helper exists). Smallest
   useful unit; unblocks "run one fixed binary, check rc".
2. **Dynamic file path** for read/write: a variant taking a runtime
   arena-pointer path (not a strlit), so the runner can open per-iteration
   files. OR (3).
3. **argv access** in the `_start` stub (`kovc.hx:9538`, which today ignores
   `[rsp]`=argc/`[rsp+8]`=argv) so a corpus dir/file can be passed in.
T1 is genuine compiler work (~3-5 chunks). NOTE: process-exec is likely also
useful for K3-seed self-validation (running the seed binary) — T1 is
foundational, not test-only.

CONSTRAINT TENSION (flag for the user): "fully in Helix" implies the runner is
a Helix program needing T1. A pragmatic alternative is a ~10-line POSIX `sh`
script as exec glue (drive the Helix `kovc` binary + run outputs), but `sh` is
"another language" — so the principled path is T1. Defaulting to T1.

## Phase T2 — the Helix-native runner + corpus extraction (after T1)

- **Runner** (~400-700 LoH `.hx`): read a corpus manifest, for each entry
  compile (invoke the in-tree `kovc` / `run_process` on a prebuilt `kovc`
  binary) → run → compare rc → tally pass/fail → exit nonzero on any fail.
  `_self_host_driver` (test_codegen.py:7139) is the Python template to mirror.
- **Corpus**: ~720 programs are ALREADY clean data-driven tuples
  `(name, src, expected_rc)` — trivially extractable: `test_parity_matrix.py`
  (284), `test_k2_parity.py` (285), `test_bootstrap_stdlib_parity.py` (72),
  `test_bootstrap_autodiff_parity.py` (~49). ~2,300-3,000 more are INLINE in
  `test_codegen.py` (1,335 `def test_`) + stage files — a one-time scripted
  lift (`compile_and_run(<str>) == <int>` is a regular shape) into the same
  table form. Programs are individually small.

## Port-vs-drop split (confirms the ~72k-DROP hypothesis)

- **PORT** (behavior/parity corpora that RUN ELFs): ~19 files, ~48.7k lines
  (but `test_codegen.py` is 34k of that and mostly repetitive small asserts +
  comments — the *programs* are tiny). These define behavior to preserve.
- **DROP** (moot post-Python-deletion): ~72 files, ~67k lines — tests of
  Python compiler INTERNALS (`test_typecheck.py` 10k, `test_cli.py` 11k,
  `test_llvm_ir.py` 6.5k, `test_mlir_*` ~11k, `test_ptx.py`, `test_parser.py`,
  regalloc/cse/dce/hash_cons/...) + Python CI/proof meta (`test_proof_*`,
  dashboard). They test Python that won't exist.

## Decomposition (tasks created)

- **T1** (gating, compiler): process-exec builtins (`run_process` + dynamic
  path/argv). Each a fixpoint-gated `.hx` chunk.
- **T2** (after T1): Helix runner `.hx` + scripted corpus extraction to a
  manifest. Replaces pytest for the PORT bucket; DROP bucket deleted with
  Python.

## Pointers
- Harness: `test_codegen.py` `compile_and_run` (51-99), `_self_host_driver`
  (7139), `_kovc_self_host_compile_and_run` (7189). Backend seam
  `_codegen_backend.py`.
- Bootstrap syscalls: `read_file_to_arena` `kovc.hx:3998`, `write_file_to_arena`
  `kovc.hx:4122`, `_start`/exit stub `kovc.hx:9538`. NO exec/fork/wait.
