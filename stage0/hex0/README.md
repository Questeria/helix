# hex0 — the Kovostov-Native seed monitor

A tiny x86_64 Linux ELF executable that reads hex characters from stdin and writes the decoded bytes to stdout. **This is the raw-binary starting point of the entire Kovostov-Native bootstrap chain.**

## Behavior

```
input (stdin):    "48 65 6C 6C 6F 0A"   (with optional whitespace and ; or # comments to end-of-line)
output (stdout):  "Hello\n"             (bytes 0x48 0x65 0x6C 0x6C 0x6F 0x0A)
```

- Whitespace (space, tab, newline, CR) is skipped.
- Comments: `;` or `#` starts a comment that runs to the next `\n`.
- Hex digits accepted: `0-9`, `A-F`, `a-f`.
- Other characters: silently skipped (lenient policy; hex1 will be stricter).
- Pairs of hex digits combine into one output byte (high nibble first).
- EOF on stdin → exit cleanly with status 0.
- Read error → exit with status 1.

## Files

| File | Role |
|---|---|
| `hex0.s` | annotated assembly (NASM syntax) — the human-readable spec |
| `hex0.bytes.md` | byte-by-byte annotation matching `hex0.bin` |
| `hex0.bin` | the compiled binary — produced by hand-typing bytes from `hex0.bytes.md` |
| `hex0.sha256` | SHA-256 of `hex0.bin` for integrity |
| `build.sh` | (re)produces and verifies `hex0.bin` |
| `verify.sh` | runs the test suite and cross-checks against nasm-assembled reference |
| `test/*.hex0` | input fixtures |
| `test/*.expected` | expected stdout for each fixture |
| `disasm.md` | objdump-based audit, instruction-by-instruction |

## Authorship

Every byte in `hex0.bin` is hand-reasoned and annotated in `hex0.bytes.md`. The assembly in `hex0.s` is the *human-readable form*; the bytes are the canonical artifact. We use `nasm` only as a cross-check (assemble `hex0.s` and compare bytes — they must be byte-identical). After verification, `nasm` is never used for shipping.

## Approximate size budget

- ELF header: 64 bytes (fixed)
- Program header: 56 bytes (fixed)
- Code: ~140–200 bytes (target)
- **Total: ~260–320 bytes**

For comparison, `oriansj/stage0-posix-amd64/hex0_AMD64.hex0` is 229 bytes but reads from argv files. Our stdin/stdout-only variant should fit similar space.
