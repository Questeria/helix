# Helix Stage 30.1 Compiler Snapshot

This is a frozen copy of the Helix compiler tree for isolated testing. It was
created so another AI or developer can compile and run Helix programs without
editing the live `C:\Projects\Kovostov-Native` worktree.

## What This Snapshot Is

- A working Python-hosted `helixc` compiler package.
- The Helix-written bootstrap compiler sources under `helixc\bootstrap`.
- A Stage 30.1 self-host cascade evidence document.
- A beginner-friendly usage guide for isolated experiments.

## What This Snapshot Is Not

- It is not the active development worktree.
- It is not yet a fully Helix-only compiler.
- It should not be patched unless the user explicitly asks to modify the
  snapshot itself.

## Start Here

Read `AI_USAGE_GUIDE.md` first. It explains how to set `PYTHONPATH`, write test
programs in `C:\Projects\Helix-Scratch`, check them, compile them, and run the
resulting Linux ELF binaries through WSL.

Useful files:

- `AI_USAGE_GUIDE.md`
- `docs\stage30-1-selfhost-cascade.md`
- `helixc\bootstrap\kovc.hx`
- `helixc\bootstrap\lexer.hx`
- `helixc\bootstrap\parser.hx`

## Safe Testing Rule

Write experiments outside this snapshot, preferably in:

```text
C:\Projects\Helix-Scratch
```

Use this snapshot as the compiler source only.
