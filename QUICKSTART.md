# Helix — Build and Run Quickstart

This is the fastest path from a fresh checkout to a running Helix program.

## Prerequisites

- **Python 3.10+** (frontend, IR, optimization passes, code generator)
- **WSL2 + Linux** on Windows, or any Linux (for running the produced ELFs)
- (Optional) `nasm` / `as` for cross-checking emitted machine code

No other dependencies. Helix bootstraps from a 299-byte hand-encoded
ELF (`stage0/hex0/hex0.bin`) and builds itself up.

## Build status

This is an early in-development language. Working today:

| Layer | Status | Test count |
|---|---|---|
| Lexer | working | 42 tests |
| Parser | working | 46 tests |
| AST | working | included |
| Type checker | working — 9 classes of bugs caught at compile time | 43 tests |
| Presburger constraint solver | working | 24 tests |
| Tensor IR | working | 13 tests |
| Tile IR | data structures only | 7 tests |
| Const folding | working + integrated | 9 tests |
| Dead code elimination | working + integrated | 6 tests |
| Forward-mode autodiff | working as CLI tool | 13 tests |
| x86-64 backend | works for scalars, control flow, arrays, floats | 52 e2e tests |
| PTX backend | text emission only (no GPU codegen yet) | 8 tests |
| stage0 hex0 monitor | working binary | 3 fixture tests |

**Total: 263 Python tests + 3 hex0 fixtures = 266 tests, all passing.**

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

CLI flags:
- `--strict` — make totality/effect warnings hard errors
- `--no-opt` — disable constant folding + DCE

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

Helix is the only AI language that does autodiff at compile time as
plain symbolic AST manipulation — the result is just another Helix
function you can read, edit, optimize, or hand-tune.

## Run the test suite

```bash
bash scripts/run_all_tests.sh
```

You should see something like:

```
pytest (stage31 sharded gate):
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
│   └── tests/          # 263 tests
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

Helix is the only language with all of:
1. **Compile-time tensor shape checking** via Presburger arithmetic — catches matmul dimension bugs before code runs.
2. **Effect/capability typing** — `@pure` cannot accidentally call `@io`.
3. **Differentiable types `D<T>`** — gradient flow tracked at the type level.
4. **Memory-tier types** — `WorkingMem` / `EpisodicMem` / `SemanticMem` / `ProceduralMem` distinguished, transitions explicit.
5. **Reflection primitives** — `quote { ... }`, `splice`, `modify` (verifier-gated).
6. **Agent declarations** — society-of-mind cognitive architecture in the type system.
7. **Symbolic autodiff** — derivatives computed at compile time, not at runtime.

See `docs/lang/agi-features.md` for the deep dive.

## License

Apache 2.0 (code), CC-BY 4.0 (docs), CC0 (model weights when produced).
