# Stage 60 Progress — 2026-05-18

## Stage Goal

Stage 60 ships **Tier 1 #4 Inc 3 — dynamic-path file I/O**, the
last remaining piece of Tier 1 #4 (string/file IO + capability
typing) that was deferred at Stage 55 closure. Inc 7 (checkpoint
stdlib) is the other deferred inc — pure Helix code that builds
on the 4 dyn builtins shipped here, will land at Stage 61.

The gap before Stage 60: file I/O builtins required string-literal
paths. Runtime-built paths (e.g. user-supplied checkpoint
directory + epoch number) couldn't open files.

## Increment breakdown

### Inc 1 — Surface (commit 52e65c0)

- 4 dyn builtins added to typecheck whitelist
- 4 intercept blocks in `lower_ast.py` emit PRINT op with operands
  (no static "path" attr)
- x86_64 backend stub raises NotImplementedError with clear
  "Inc 2 will wire" message — fail-closed surface preserves
  the silent-miscompile discipline

### Inc 2 — read_file_to_arena_dyn end-to-end (commit 0a55624)

- ~110 lines of x86_64 assembly
- Stack frame: 1MB read buffer + 4KB path scratch + 8 fd save
- Path-copy loop: arena[path_start..path_start+path_len)
  → path_scratch[0..path_len], null-terminate at [len]
- PATH_MAX overflow trap (ud2 if path_len > 4095)
- sys_open(path_scratch, O_RDONLY) + sys_read into read_buffer
  + truncation sentinel + per-byte arena push + sys_close
- End-to-end round-trip test: Python writes file → Helix
  builds path at runtime via __strlit_to_arena → dyn-reads
  → returns byte count = 10 for "hello dyn\n"

### Inc 3 — write_file_to_arena_dyn end-to-end (commit 83025fa)

- ~140 lines of x86_64 assembly
- Same path-copy preamble as Inc 2
- Stack frame: 4KB path scratch + 16 bytes (1-byte write buf + fd)
- Saves callee-saved regs (rbx, r12-r14) for write-loop state
- sys_open with O_WRONLY|O_CREAT|O_TRUNC mode 0644
- Per-byte sys_write loop (mirrors static write_file_to_arena)
- Error path: returns 0 on open failure
- Round-trip test: Helix writes "hello\n" via runtime path,
  Python verifies file exists + content matches

### Inc 4 — read_file_int_dyn + write_file_dyn alias + close (commit d579d87)

- read_file_int_dyn: ~95 lines, reads first 4 bytes as i32 LE
- write_file_dyn aliased to write_file_to_arena_dyn backend
  (semantically equivalent for dyn paths; naming distinction
  preserved for symmetry with static)
- Cascade defect #1 found+fixed: je-displacement off-by-2 in
  read_file_int_dyn error path; regression-pinned
- 2 round-trip tests

## Closure narrative

**3-clean-gate satisfaction**:

- Gate A (silent-failure): every test_strings_io test runs
  end-to-end via WSL — actual file I/O works, no silent miscompile
  observed. 15/15 GREEN at closure.
- Gate B (type-design): 4 builtins typecheck via existing
  whitelist mechanism; effect labels inherit io.read_file /
  io.write_file from `_SUB_LABELS` (Stage 55 Inc 4 infrastructure
  applies automatically). No new type-design surface.
- Gate C (code-review): each Inc's x86_64 assembly mirrors the
  static-path variant's already-audited pattern; differences are
  isolated to (a) path-copy preamble (b) lea rsp+disp32 vs lea
  rip+sym. Same disp8/disp32 fixup discipline.

**Cascade defects in this stage**: 1 (Inc 4 je-displacement
off-by-2; caught by test, fixed inline before commit).

**Test counts**:
- test_strings_io.py: 15/15 (added 4 new dyn tests, removed 2
  Inc 1 stub-error tests that became obsolete once Inc 2/3/4
  shipped real implementations)
- Self-host gate (5 introspection files): 223/223 GREEN at
  every commit

## Next stage

**Stage 61 opens immediately**: Tier 1 #4 Inc 7 — checkpoint
stdlib. Pure Helix code (`helixc/stdlib/checkpoint.hx`) built
on top of the 4 dyn builtins shipped here. Estimated 1 day
(stdlib-only, no codegen changes).

After Stage 61 closes, Stages 62-65 proceed autonomously
(struct-shaped grad return → runtime trace wiring → tensor
codegen bf16/perf → multiple dispatch). Stage 66 (borrow
checker) is the first STOP-FOR-USER gate.
