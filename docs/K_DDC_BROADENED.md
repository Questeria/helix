# K_DDC_BROADENED — second-witness DDC broadening (T1) — IN PROGRESS (P0/P1 banked)

Sibling to `K_DDC_RESULT.md`. Tracks T1 of the Helix Completion charter
(`docs/HELIX_COMPLETION.md` §2): reviving the deleted Python `helixc` as a **fenced,
uncommitted** second witness and asserting it agrees with the from-raw `kovc` over a
broadened corpus exercising the dark value-codegen arms. **This file is the running
record; the final "proves / does not prove" write-up lands at P4.**

> **Fence invariant (load-bearing, unchanged):** the shipped tree keeps **exactly one**
> committed `.py` — `verification/oracle/oracle_train.py`. The Python `helixc` witness is
> restored into the **gitignored** `verification/py_witness/` and is **never** committed.
> Verified this chunk: `git ls-files "*.py"` == 1; `git check-ignore verification/py_witness/`
> confirms the dir + every `.py` ignored; `git status` does not list it; nothing `.py` staged.

## P0 — witness restored + runs (DONE, GATE 0 PASS)

- Source: tag `v0-pre-k4-full-with-python`, subtree
  `HELIX_STAGE30_COMPILER_SNAPSHOT/helixc/` (92 `.py`; pure-stdlib `backend/x86_64.py`,
  **no numpy / no third-party deps**), restored via `git archive | tar -x` into
  `verification/py_witness/helixc/` (PEP-420 namespace package, no root `__init__.py`).
- Invocation (WSL, Python 3.12.3): `PYTHONPATH=verification/py_witness
  PYTHONDONTWRITEBYTECODE=1 python3 -m helixc.check -o out.bin src.hx`.
- `fn main()->i32{42}` → 4649-byte ELF → **exit 42**.

## From-raw kovc (K1'/K2) used for the cross-check

Rebuilt this chunk from the existing `seed.bin` + current `k1src.hx`
(`scripts/selfhost_fixpoint_rawbinary.sh` route): seed→K1 = **600783 B**, K1→K2 =
**606680 B**, sha256 `03a456fe…`, **K2==K3 fixpoint OK** — exactly reproduces the
DC3 result (commit `72faee0`). This is the proven self-host compiler, the DDC's
from-raw route.

## P1 — coverage bound (the KEY honest finding)

The snapshot is **Stage 30**, **before** most of v1.1's surface (generics/traits/
closures/turbofish/field-store/break-continue). Probing the v1.0+v1.1 corpus through
the witness:

**Witness-REACHABLE value-codegen arms** (witness compiled it AND the binary ran to the
predicted exit): i32 + i8/i16/u8/u16/u32 + **i64 incl. >2³²**; f64; `+ - * / %`;
`& | ^`; `<< >>`; unary `- ~ !`; six comparisons; `let`/`let mut`/assign/seq/`if`/
`while`/call/recursion; **array index read + index-store `arr[i]=e`** + arena
intrinsics; **struct literal + field read**; tuple literal + single-level field;
**enum + match + match-or + match-range + payload-extract**; generic-enum match
`Opt::Some(x)`; **match guards** (`n if cond`, an H4 v1.1 feature, surprisingly reached);
`&str` literal; cast `as`. ≈ **38–42 of the ~53** value arms.

**Witness-EXCLUDED by source-drift** (logged as drift exclusions; **NOT** counted toward
≥40, never silently folded into an "accepted subset"):
- struct **field-store** `p.x = v` (AST_FIELD_STORE 79) — Stage-30 typecheck rejects;
- `break` / `continue` — Stage-30 backend "not yet supported";
- `bf16` / `f16` scalar float literals — no F16C/AVX-512 codegen path;
- the **generics / traits / closures surface** — inherent & trait `impl` blocks
  (parse: `expected COLON` at `self`), closures `|x|` (parse), turbofish `::<T>`
  (parse/typecheck), bracket generics `[T]` monomorphization (parse/typecheck),
  trait-default resolution (typecheck).
- **GPU/PTX arms** (autodiff/tile/kernel, tags 43/77/78) — carved out per charter
  (ELF-byte-DDC cannot reach PTX-emitting arms; covered by `gate_kovc.sh` PTX-regression
  + capstone finite-diff), listed, not counted as a miss.

**This confirms the charter's named teaching-to-the-test risk**: the newest v1.1
constructs are exactly the arms the frozen Stage-30 frontend rejects. Per-arm witness-
parse status is reported above and machine-recorded in the (gitignored) findings log.

## Initial cross-check (witness vs from-raw K2, direct-compile form)

27/27 witness-reachable corpus programs **behaviorally agree** (same exit), **0
byte-identical**, **0 disagreements**. The byte-DIFF is **expected and correct**: the
witness (Stage-30 standalone codegen; `exit42` = 4649 B) and K2 (v1.1 self-hosted `kovc`
with `emit_start_bigstack`; `exit42` = 4184 B) are **two different compilers** — same ELF
entry `0x401000`, 465-byte layout delta — **not** one `kovc` built two ways. This is the
**strong-independence behavioral form**. The **byte-identical** strong form (as the
original `ddc_battery` 21/21) requires the P3 design below.

## P1 GATE-1: witness building the current `k1src.hx` → K1py (for the byte form)

- Default attempt: Python `RecursionError` on the 1.5 MB deeply-nested source. **Fixed**
  by `sys.setrecursionlimit(2_000_000)` + a 1 GiB thread stack → witness parses the full
  source in ~4 s.
- Residual after the fix: **exactly 7 typecheck divergences, all one kind** —
  `if/else branches differ: () vs i32` (`k1src.hx` lines 10433/10434/10453/10454/10455/
  11359/11371): Stage-30 typecheck rejects `if`-as-statement with divergent branch types;
  v1.1 `kovc` permits it in statement position.
- The drift constructs (`impl`/closure/turbofish/generic/field-store) appear in
  `kovc.hx`/`parser.hx`/`lexer.hx` **only in comments** — the executable bootstrap source
  is within the witness grammar. So the **sole** blocker to a byte-identical K1py is those
  7 `if`-statement sites: a small, named, fixable drift, **not a wall**.

## Realistic path to DDC_BROAD_PASS (given the bound)

- **R1 — byte form on the reachable arms.** Ship the fenced witness recursion/bigstack fix
  + accommodate the 7 `if/else`-statement sites (witness-side typecheck relax **or** a
  fenced source-shim — **never** editing the committed `kovc.hx`). Then the witness mints
  K1py; revive `ddc_battery` as `ddc_battery_broad` over the ≈40 reachable arms and compare
  **K1py-output vs K1'-output byte-identical** per program (the original 21/21 form
  extended upward).
- **R2 — the excluded arms, honestly.** Field-store / break / continue / bf16-f16 / and the
  generics-traits-closures programs are **drift-excluded** and not counted toward ≥40.
  They are covered instead by (a) the from-raw self-host fixpoint `K2==K3==K4` and (b) the
  **gcc-DDC** (DC1–DC3, commit `72faee0`) — a **different-lineage** second witness with zero
  M2-Planet ancestry. Document the bound; a 2nd diverse witness for the v1.1 surface is the
  alternative if byte-DDC of those arms is later required.

**Net:** **≥40/53 witness-reachable is achievable for the value arms via R1**; the v1.1-
surface arms are an honest drift exclusion closed by the gcc-DDC + fixpoint. The ≥40 count
will be computed (not asserted) from `--emit-ast` dynamic-exercise instrumentation at P4.

## Reproduce (from repo root, WSL)

```
# restore the witness (gitignored):
mkdir -p verification/py_witness/helixc
git archive v0-pre-k4-full-with-python:HELIX_STAGE30_COMPILER_SNAPSHOT/helixc \
  | tar -x -C verification/py_witness/helixc/
# GATE 0:
bash verification/py_witness/run_witness.sh \
  verification/py_witness/helixc/examples/exit42.hx /tmp/w.bin   # -> exit 42
# coverage probe + from-raw K2 + cross-check:
bash verification/py_witness/build_k2_fromraw.sh        # K2 = 606680 B, 03a456fe…
bash verification/py_witness/coverage_probe.sh          # 27 OK / drift exclusions
bash verification/py_witness/crosscheck.sh              # 27/27 behavioral agree
```

*Status: P0 done (witness restored + GATE 0). P1 coverage bound measured + initial cross-
check banked. P2–P4 (broadened dark-arm corpus, byte-form K1py via R1, ≥40/53 count, gate
script, `quince` debate) pending. Fence intact throughout (1 committed `.py`).*
