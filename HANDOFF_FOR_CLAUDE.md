# Helix Handoff for Claude

Date: 2026-05-29
Repo: `C:\Projects\Kovostov-Native`
Remote: `https://github.com/Questeria/helix.git`
Branch: `main`

This handoff **supersedes the 2026-05-23 MLIR-audit takeover** — that batch is
obsolete: MLIR is now slated for **deletion**, not hardening
(`docs\MLIR_NOT_NEEDED_DECISION.md` — the bootstrap is 100% direct-codegen).
Treat live git state as truth if it differs from this file.

## Current thrust: the self-hosting K-bootstrap parity campaign

The project moved from per-stage MLIR work to a **self-hosting parity campaign**
driving the Helix-native bootstrap compiler toward **Python-deletion-ready**.
The bootstrap compiler is `helixc/bootstrap/{lexer,parser,kovc}.hx` — a
from-scratch Helix compiler written *in Helix* that emits x86-64 ELF directly
(no assembler / linker / libc). The Python `helixc/` is the reference oracle and
the current seed.

**Achieved:**
- **Self-host fixpoint (byte-identical):** Python builds K1 from the bootstrap
  source; K1 compiles the source → K2; K2 → K3; **K2 == K3 byte-for-byte**.
  Locked by `helixc/tests/test_self_host_fixpoint.py`.
- **Core CPU-language parity: 100%.** Measured by
  `helixc/tests/test_parity_matrix.py` — a data-driven Python-vs-bootstrap
  corpus, now **277 cases**. Traits, nested structures, and type-aliases are all
  at parity.
- 3 parity bugs fixed this campaign: string-escape decode, typed-`self` struct
  receiver, u8/u16 literal-suffix (`14002` width-trap family).
- Test infra hardened: a WSL keepalive conftest + corpus retry-on-any-mismatch
  kill the cold-start `rc=1` / SIGILL flakes that produced false reds.

**In progress / next build:**
- **Generics** are the sole remaining CPU-language gap — all generic fn/struct
  programs SIGILL in the bootstrap while Python compiles them; same `14002`
  width-trap family (generic-param index `200+k` not folded). Tracked as xfail
  in `KNOWN_PARITY_GAPS`; fix in progress.

## Remaining scope to Python-deletion (synthesized 2-agent assessment, 2026-05-29)

The CPU language is ~99% (generics the last gap). The **bulk** to actually
delete Python ("K4", user-gated) is the non-language subsystems:

| Item | Bootstrap status | Effort |
|---|---|---|
| Autodiff / grad passes (~4,650 lines) | none | XL |
| Type checker (`typecheck.py` 12,900 lines) | heuristic/partial only | XL |
| Python test harness (~115k lines pytest) | no Helix runner yet | XL |
| GPU backends (PTX/ROCm/Metal/WebGPU completion) | PTX partial, others skeleton | L ×4 |
| Opt passes + monomorphize | none / partial hack | M–L |
| **K3 trusted seed** (checked-in Python-free seed binary) | cascade works, no frozen seed | M — GATING |
| MLIR subsystem (15,360 lines) | DELETE, not port | S |

## Stop condition (user directive 2026-05-29)

The autonomous loop runs until **BOTH**: (1) Python-deletion-ready (all of the
above), AND (2) **5 consecutive multi-agent clean audits** (any finding resets
the streak to 0). **Never delete Python autonomously** — K4 is user-gated.

## Live state (read these — numbers go stale fast)

```
git -C C:\Projects\Kovostov-Native log --oneline -8
python scripts\helix_status.py            # K-bootstrap chunk counter (~392)
helixc\tests\test_parity_matrix.py        # parity corpus (277 cases; gaps as xfail)
helixc\tests\test_self_host_fixpoint.py   # K2==K3 byte-identical fixpoint
```
HEAD at this handoff: `5335304` (README status update). Counter ~392.

## Audit / commit discipline (Anthony requires)

- **Fail closed always.** Unsupported constructs must raise / return FAILED,
  never emit plausible-but-wrong output. Never ship red; never fake an audit.
- After `parser.hx` / `kovc.hx` / `lexer.hx` / codegen changes: **broad
  regression before commit** (`-k "(self_host or k2_parity or k2_corpus) and not
  self_host_loop"`), and the **self-host fixpoint must stay green**.
- Never force-push to `main`. Never skip git hooks. Stage explicit paths (no
  broad `git add .`). Claude subscription only; never read
  `C:/Projects/Neptune/api.env`. `reg` is a reserved Helix keyword.

## Telegram

Concise updates after meaningful progress / blockers:
```
python C:\Projects\Kovostov\runtime\lib\kovostov_telegram.py send --chat 8212106071 --msg "Helix update: <status>. Next: <next>."
```
Or the full status panel: `python scripts\helix_status.py --note "<plain text>" --commit <sha>`.

## Important project docs

- `docs\MLIR_NOT_NEEDED_DECISION.md` — MLIR slated for deletion (direct-emit wins).
- `docs\K_BOOTSTRAP_HARD_CONSTRAINT.md` — the fully-in-Helix constraint + stop criterion.
- `docs\HELIX_K_BOOTSTRAP_MASTER_PLAN.md` — K-bootstrap plan.
- `docs\V3_PLAN.md` — prior v3.0 chunk history (pre-pivot context).

## One-sentence takeover

You are running the autonomous self-hosting parity campaign: the bootstrap
self-hosts (K2==K3) and the core language is at 100% parity (277-case corpus);
close the generics gap, then build the K3 trusted seed and port the remaining
subsystems (autodiff / typecheck / GPU / test-infra), proving each via the
parity corpus + self-host fixpoint, toward Python-deletion-ready + 5 clean
multi-agent audits.
