# Helix Stage 30 Compiler Snapshot Guide

This folder is a read-only working snapshot of the Helix compiler from commit
`ecda9da` (`Add Stage 30.1 self-host cascade gate`).

Use this folder for experiments that must not edit the live project in
`C:\Projects\Kovostov-Native`.

## What This Snapshot Is

Helix Stage 30 is partially self-hosted:

- The practical compiler driver is still Python-hosted in `helixc`.
- The Helix-written bootstrap compiler source is in `helixc\bootstrap`.
- Stage 30.1 proved the bootstrap path can rebuild itself for 10 Helix-to-Helix
  generations with byte-identical outputs from `G2` through `G11`.

This is not yet a fully Helix-only compiler. Treat the Python compiler as the
reference driver and the bootstrap files as the Helix self-host core.

## Do Not Touch The Live Repo

If you are an AI testing Helix, follow these rules:

- Do not edit files under `C:\Projects\Kovostov-Native\helixc`.
- Do not run `git` commands in `C:\Projects\Kovostov-Native`.
- Write your test programs and binaries in a separate scratch folder.
- Use this snapshot folder as the compiler source.
- Set `PYTHONDONTWRITEBYTECODE=1` so Python avoids writing `__pycache__`.

Recommended scratch folder:

```powershell
mkdir C:\Projects\Helix-Scratch -Force
cd C:\Projects\Helix-Scratch
```

## Environment Setup

Run these commands before using the snapshot compiler:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = 'C:\Projects\Kovostov-Native\HELIX_STAGE30_COMPILER_SNAPSHOT'
cd C:\Projects\Helix-Scratch
```

## Smallest Test Program

Create `hello.hx` in the scratch folder:

```powershell
@'
fn main() -> i32 {
    42
}
'@ | Set-Content -Path C:\Projects\Helix-Scratch\hello.hx
```

Check it:

```powershell
python -m helixc.check --check-only --strict C:\Projects\Helix-Scratch\hello.hx
```

Compile it:

```powershell
python -m helixc.backend.x86_64 C:\Projects\Helix-Scratch\hello.hx C:\Projects\Helix-Scratch\hello.bin --no-stdlib
```

Run it with WSL:

```powershell
wsl -- bash -lc "chmod +x /mnt/c/Projects/Helix-Scratch/hello.bin && /mnt/c/Projects/Helix-Scratch/hello.bin"
Write-Output "EXIT:$LASTEXITCODE"
```

Expected result:

```text
EXIT:42
```

## Useful Compiler Commands

Check only:

```powershell
python -m helixc.check --check-only --strict C:\Projects\Helix-Scratch\hello.hx
```

Print Tensor IR:

```powershell
python -m helixc.check --emit-ir C:\Projects\Helix-Scratch\hello.hx
```

Print x86_64 assembly/bytes view:

```powershell
python -m helixc.check --emit-asm C:\Projects\Helix-Scratch\hello.hx
```

Compile to ELF binary through the main check CLI:

```powershell
python -m helixc.check -O1 -o C:\Projects\Helix-Scratch\hello.bin C:\Projects\Helix-Scratch\hello.hx
```

Compile directly through the x86_64 backend:

```powershell
python -m helixc.backend.x86_64 C:\Projects\Helix-Scratch\hello.hx C:\Projects\Helix-Scratch\hello.bin --no-stdlib
```

Use default stdlib:

```powershell
python -m helixc.backend.x86_64 C:\Projects\Helix-Scratch\program.hx C:\Projects\Helix-Scratch\program.bin
```

Avoid stdlib:

```powershell
python -m helixc.backend.x86_64 C:\Projects\Helix-Scratch\program.hx C:\Projects\Helix-Scratch\program.bin --no-stdlib
```

## Stage 30 Self-Host Cascade

This reproduces the Stage 30.1 self-host stability check from the snapshot:

```powershell
cd C:\Projects\Kovostov-Native\HELIX_STAGE30_COMPILER_SNAPSHOT
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = 'C:\Projects\Kovostov-Native\HELIX_STAGE30_COMPILER_SNAPSHOT'
python scripts\selfhost_cascade.py --generations 10 --keep
```

Expected Stage 30.1 stable result:

```text
cascade: PASS G2..G11 are byte-identical
G2..G11 sha256 = 5a7367ad436e72ade3d8f96a9860e0d08b64528cbb15295e1a47076090667408
G2..G11 size   = 277899 bytes
smoke: PASS final generation compiled and ran all smoke programs
```

The script writes temporary binaries under WSL `/tmp` by default. If you use
`--keep`, those binaries remain there for inspection.

## Where The Important Files Are

- Python reference compiler package: `helixc`
- Check CLI: `helixc\check.py`
- x86_64 backend: `helixc\backend\x86_64.py`
- Helix bootstrap compiler: `helixc\bootstrap\kovc.hx`
- Helix bootstrap lexer: `helixc\bootstrap\lexer.hx`
- Helix bootstrap parser: `helixc\bootstrap\parser.hx`
- Standard library: `helixc\stdlib`
- Stage 30 cascade script: `scripts\selfhost_cascade.py`
- Stage 30 cascade report: `docs\stage30-1-selfhost-cascade.md`

## What To Test

Start with small programs that return an integer exit code:

```hx
fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn main() -> i32 {
    add(20, 22)
}
```

Then try:

- arithmetic
- function calls
- `if` expressions
- `while` loops
- simple structs
- simple enums and matches
- small stdlib helper calls

When a program compiles but behaves unexpectedly, save:

- the `.hx` source
- the exact command used
- the compiler stdout/stderr
- the binary exit code

Do not patch the snapshot or live compiler unless explicitly asked.
