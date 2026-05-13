# Stage 30.1 Self-Host Cascade

Date: 2026-05-13

Purpose: release-hardening check after Stage 30 reached the 5/5 clean audit gate.

Command:

```powershell
python scripts\selfhost_cascade.py --generations 10 --keep
```

Result: PASS.

The seed compiler `G1` was built by the Python reference path from the Helix
bootstrap source. Then `G1` compiled the same Helix compiler source to `G2`,
`G2` compiled it to `G3`, and so on through `G11`.

Stable self-host hash:

```text
G2..G11 sha256 = 5a7367ad436e72ade3d8f96a9860e0d08b64528cbb15295e1a47076090667408
G2..G11 size   = 277899 bytes
```

The compiler process exit low byte was `139` for each generation. This is the
compiler returning the low byte of the written ELF size, not evidence of a
crash; each generation produced a non-empty next compiler and the hashes were
identical.

Final-generation smoke checks:

```text
literal -> exit 42
call    -> exit 42
loop    -> exit 42
```

Interpretation: Helix reached a stable self-host fixed point for this bootstrap
driver. This does not prove every possible compiler bug is impossible, but it is
strong evidence that the self-hosted compiler rebuilds itself deterministically
across at least ten Helix-to-Helix generations.
