# Helix — Build and Run Quickstart

This is the fastest path from a fresh checkout to a running Helix program.

## Prerequisites

- **Python 3.10+** (frontend, IR, optimization passes, code generator)
- **WSL2 + Linux** on Windows, or any Linux (for running the produced ELFs)
- (Optional) `nasm` / `as` for cross-checking emitted machine code

No other dependencies for the current Python-hosted `helixc` compiler. The
repository also contains the live 299-byte hand-encoded ELF
(`stage0/hex0/hex0.bin`) that serves as the audited bootstrap floor; the later
bootstrap links and self-hosted compiler remain roadmap targets until they can
rebuild the compiler reproducibly.

## Build status

This is an early in-development language. Stage 35 is currently in audit
cleanup, and clean gates remain `0/3` in the Stage 35 progress ledger. Restart
51 fix verification collected 2,497 live `helixc/tests` pytest tests (the
restart-50 ledger forecast 2,489; restart 51 reconciled to the actual live
count). Run `python -m pytest helixc/tests --collect-only -q -p no:cacheprovider`
for the current count.

Working today:

| Layer | Status |
|---|---|
| Lexer/parser/AST | working and covered by pytest |
| Type checker | working, with refinements/effects/shapes under active audit |
| Presburger constraint solver | working |
| Tensor IR | working |
| Tile IR | data structures plus PTX-oriented lowering paths under Stage 35 audit |
| Const folding / CSE / DCE / FDCE | working and integrated |
| Forward and reverse autodiff | working for the covered symbolic/runtime paths |
| x86-64 backend | works for scalars, control flow, arrays, floats, arena-backed tensors |
| PTX backend | text emission for covered kernels; GPU execution is still not a shipped capability |
| stage0 hex0 monitor | working 299-byte binary fixture |

## Compile and run a Helix program

```bash
# 1. Write a .hx file
cat > hello.hx <<'EOF'
fn fib(n: i32) -> i32 {
    if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
}

fn main() -> i32 {
    fib(9)
}
EOF

# 2. Compile to a Linux ELF
python -m helixc.backend.x86_64 hello.hx hello.bin

# 3. Run it (Linux/WSL)
chmod +x hello.bin
./hello.bin
echo $?     # prints: 34   (Fibonacci(9))
```

CLI flags for `python -m helixc.backend.x86_64` (the `python -m helixc.check`
driver accepts the same set plus the `--emit-*`, `--check-only`, `--doc`,
and `-o` modes — see `python -m helixc.check --help` for the canonical
list):
- `--strict` — make totality/effect warnings hard errors
- `--no-opt` or `-O0` — disable optimization passes (const-fold + CSE + DCE + FDCE)
- `-O1` (default) / `-O2` / `-O3` — optimization level
- `--stdlib` (default) / `--no-stdlib` — bundle (or skip) `helixc/stdlib/*.hx`
- `-Wad=warn|error` / `-Wdeprecated=warn|error` — warning policy
- `-l <libname>` / `-l<libname>` — mark external library (FFI prerequisite;
  no-op for backends that don't link)
- `--no-color` / `--color` — disable / force ANSI escapes (also: `NO_COLOR` env)
- `--hash` / `--hash-cons` — structural hash / dedup helpers (no-op in
  backends; meaningful in `helixc.check`)

Run with no arguments to see the full banner. `python -m helixc.check --help`
is the canonical source of truth for accepted flags.

## Type-check only (no codegen)

```bash
python -m helixc.frontend.typecheck hello.hx
```

If there are type errors, you get Rust-style messages with source-line
context:

```
error: call to 'matmul': shape constraint violated (-1 == 0)
   --> hello.hx:3:5
    |
  3 |     matmul(x, z)
    |     ^
```

## Symbolic autodiff

```bash
cat > loss.hx <<'EOF'
fn loss(x: f32) -> f32 { x * x }
fn cubic(x: f32) -> f32 { x * x * x }
fn linear(x: f32, y: f32) -> f32 { 3.0 * x + 5.0 * y }
EOF

python -m helixc.frontend.autodiff_cli loss.hx loss
# d(loss)/d(x) = (x + x)

python -m helixc.frontend.autodiff_cli loss.hx cubic
# d(cubic)/d(x) = (((x + x) * x) + (x * x))

python -m helixc.frontend.autodiff_cli loss.hx linear x
# d(linear)/d(x) = 3

python -m helixc.frontend.autodiff_cli loss.hx linear y
# d(linear)/d(y) = 5
```

### Round-trip: generate a derivative function, then compile and run it

The CLI's `--as-function` flag emits a full Helix function definition,
ready to paste into another file:

```bash
$ cat my_loss.hx
fn loss(x: f32) -> f32 {
    let pred = x * 2.0 + 3.0;
    let target = 7.0;
    let diff = pred - target;
    diff * diff
}

$ python -m helixc.frontend.autodiff_cli my_loss.hx loss --as-function
fn loss__grad(x: f32) -> f32 {
    ((2 * (((x * 2) + 3) - 7)) + ((((x * 2) + 3) - 7) * 2))
}
```

Paste that into your file, compile, and you have a working
`loss__grad(x)` function.

Helix's autodiff path is built around compile-time symbolic AST manipulation:
the result is another Helix function you can read, edit, optimize, or
hand-tune.

## Run the test suite

```bash
bash scripts/run_all_tests.sh
```

The gate infrastructure still uses historical `stage31` log names, but it is
the current full-suite smoke gate used during Stage 35 audit cleanup. You
should see something like:

```
pytest (current sharded gate; historical stage31 log names):
pytest-no-codegen: rc=0 log=.stage31-logs/pytest-no-codegen.log
pytest-codegen-shard-1-of-4: rc=0 log=.stage31-logs/pytest-codegen-shard-1-of-4.log
pytest-codegen-shard-2-of-4: rc=0 log=.stage31-logs/pytest-codegen-shard-2-of-4.log
pytest-codegen-shard-3-of-4: rc=0 log=.stage31-logs/pytest-codegen-shard-3-of-4.log
pytest-codegen-shard-4-of-4: rc=0 log=.stage31-logs/pytest-codegen-shard-4-of-4.log
snapshot-check: rc=0 log=.stage31-logs/snapshot-check.log
snapshot-compile: rc=0 log=.stage31-logs/snapshot-compile.log
snapshot-run: rc=42

stage0/hex0:
PASS 03-empty
Results: 3 passed, 0 failed

=============================
pytest gate rc: 0
stage0/hex0 rc: 0
TOTAL: all gates passed
```

## Project layout

```
Kovostov-Native/
├── stage0/hex0/        # Hand-encoded raw-binary ELF (the bootstrap floor)
├── helixc/
│   ├── frontend/       # lexer, parser, AST, typecheck, presburger, autodiff
│   ├── ir/             # Tensor IR, Tile IR, lowering passes
│   │   └── passes/     # const_fold, dce
│   ├── backend/        # x86_64 (works), ptx (text-emit stub)
│   ├── examples/       # working .hx programs
│   └── tests/          # pytest suite; audits keep adding regressions
├── docs/
│   ├── PLAN.md
│   ├── lang/
│   │   ├── spec.md          # formal language reference
│   │   ├── tutorial.md      # 10-step beginner guide
│   │   └── agi-features.md  # what makes Helix different
│   └── research-log.md      # day-by-day implementation log
└── scripts/run_all_tests.sh
```

## What makes Helix different

Helix is being built to combine:
1. **Compile-time tensor shape checking** via Presburger arithmetic — catches matmul dimension bugs before code runs.
2. **Effect/capability typing** — `@pure` cannot accidentally call `@io`.
3. **Differentiable types `D<T>`** — gradient flow tracked at the type level.
4. **Memory-tier types** — `WorkingMem` / `EpisodicMem` / `SemanticMem` / `ProceduralMem` distinguished, transitions explicit.
5. **Reflection primitives** — `quote { ... }`, `splice`, `modify` (verifier-gated).
6. **Agent declarations** — society-of-mind cognitive architecture in the type system.
7. **Symbolic autodiff** — derivatives computed at compile time, not at runtime.

See `docs/lang/agi-features.md` for the deep dive.

## License

Apache 2.0 (code, in `LICENSE`); CC-BY 4.0 (docs, stated policy); CC0 (model weights when produced, stated policy).
