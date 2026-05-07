# Stage 2.4b — Correction Notes

## Commit 7e88c94 message factual error

The commit message for 7e88c94 ("Stage 2.4b u64 DIV/MOD/comparisons")
contains an inaccurate description:

> "u64 > u64 emits emit_gt_rax_rcx_64_u (setae)"

This is wrong. `emit_gt_rax_rcx_64_u` calls `emit_cmp_setX_64(0x97)`,
and opcode `0x97` is **seta** (set if above; CF=0 AND ZF=0), not `setae`.

Correct mapping (documented at `kovc.hx` near `emit_lt_rax_rcx_64_u`):

| Helper                   | setcc opcode | Mnemonic | Semantics                          |
|--------------------------|--------------|----------|------------------------------------|
| `emit_lt_rax_rcx_64_u`   | `0x92`       | `setb`   | unsigned <  (CF=1)                 |
| `emit_gt_rax_rcx_64_u`   | `0x97`       | `seta`   | unsigned >  (CF=0 AND ZF=0)        |
| `emit_le_rax_rcx_64_u`   | `0x96`       | `setbe`  | unsigned <= (CF=1 OR ZF=1)         |
| `emit_ge_rax_rcx_64_u`   | `0x93`       | `setae`  | unsigned >= (CF=0)                 |

The commit message confused GT (`>`) with GE (`>=`). The code itself is
correct; only the commit message description is wrong. The signed 32-bit
and 64-bit GT helpers (`emit_gt_eax_ecx`, `emit_gt_rax_rcx_64`) similarly
use `0x9F` (setg), not `setge`.

This note supersedes the commit message for historical record purposes.
