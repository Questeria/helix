# K_DDC_BROADENED — second-witness DDC broadening (T1) — DDC_BROAD_PASS (behavioral form)

Sibling to `K_DDC_RESULT.md`. This is the **final** record for Track 1 of the Helix
Completion charter (`docs/HELIX_COMPLETION.md` §1.1 / §2): reviving the deleted Python
`helixc` as a **fenced, uncommitted** second witness and asserting it agrees with the
from-raw `kovc` over a broadened corpus exercising the dark value-codegen arms.

**Date:** 2026-06-02 · **Repo:** Kovostov-Native @ branch `main` · prior commit `611e4e0`
· from-raw K2 reproduces DC3 (commit `72faee0`), sha `03a456fe…`.

> **Fence invariant (load-bearing, intact):** the shipped tree keeps **exactly one**
> committed `.py` — `verification/oracle/oracle_train.py`. The Python `helixc` witness
> lives in the **gitignored** `verification/py_witness/` and is **never** committed —
> including the R1 typecheck shim below (`verification/py_witness/helixc/frontend/
> typecheck.py`, confirmed `git check-ignore`). Verified this run:
> `git ls-files "*.py"` == **1**; `git check-ignore verification/py_witness/` confirms
> the dir + every `.py` (incl. the shim) is ignored; nothing `.py` staged.

---

## DDC_BROAD_PASS gate status (the §1.1 checklist)

| # | Gate item | Status | Evidence |
|---|-----------|--------|----------|
| 1 | live-tree `git ls-files "*.py"` == exactly `verification/oracle/oracle_train.py` | **MET** | fence == 1 this run; witness + R1 shim gitignored |
| 2 | from-raw kovc (K1') and Python-witness (K1py) AGREE on every broadened-corpus program — byte OR behavioral, each to predicted exit; split REPORTED; no unexplained byte-DIFF; behavioral = MULTI-INPUT | **MET (behavioral, multi-input)** | 42/42 witness-reachable programs behaviorally agree (same exit), **0 true disagreements**; byte-DIFF is uniform and explained (two different compilers); a 5-arm × 5-input differential agrees on every input (no single-point coincidence); 2 programs are witness-PARSE drift exclusions (not disagreements) |
| 3 | distinct value-codegen arms the witness **actually parsed-and-compiled** AND dynamically exercised by the cross-checked corpus **≥ 40 of 53** | **MET — 44/53** (honest dedup) | distinct kovc codegen-dispatch arms exercised by agreeing programs; per-arm parse status reported below; GPU/PTX arms 43/77/78 carved out |
| 4 | quince debate verdict = "agreement holds" (hunts a drift-masked count + a behavioral-match hiding a byte divergence) | **DEFERRED to campaign final audits** | folded into the FINALE 5-audit streak (§1.4), lens (ii); noted here, not run as a standalone gate |

**Net: DDC_BROAD_PASS holds in its behavioral form** (the charter accepts byte **OR**
behavioral). Item 4 (quince) folds into the campaign's finale audits per the task scope.
The R1 stronger byte form is reported separately below (best-effort).

---

## Method

Two genuinely independent compilers, same kovc source lineage:
- **K1' (from-raw route):** `seed.bin` (a small M2-Planet-C compiler built **only** by the
  from-raw-binary stage-0 ladder hex0→…→M2-Planet) mints kovc; the self-host fixpoint
  gives K2 = **606680 B**, sha `03a456fe…`, **K2==K3** (reproduces DC3 / commit `72faee0`).
  This is the proven Python-free compiler.
- **K1py (witness route):** the **fenced, gitignored** Stage-30 Python `helixc` (restored
  from tag `v0-pre-k4-full-with-python:HELIX_STAGE30_COMPILER_SNAPSHOT/helixc`, 92 `.py`,
  pure-stdlib x86_64 backend, no numpy). Invocation:
  `PYTHONPATH=verification/py_witness PYTHONDONTWRITEBYTECODE=1 python3 -m helixc.check -o out.bin src.hx`.

For each broadened-corpus program: compile with **both** K1py and K1' (the from-raw K2),
`cmp -s` the two ELFs (byte form), then **run each** to its predicted exit (behavioral
form). Anti-false-match: both outputs `rm -f`'d first, so a silent failure becomes
MISSING→COMPILE_FAIL, never a stale match. Only WITNESS-REACHABLE programs count toward
the arm total; a program the witness cannot parse-and-compile is logged as a **drift
exclusion**, never silently folded into an "accepted subset."

Reproduce: `bash verification/py_witness/run_all_broad.sh` (one serial session: builds the
from-raw K2 once, witness-reachability-checks the new probes, runs the full cross-check,
prints the distinct-arm count). Results persisted to
`verification/py_witness/_broad_results.txt`.

---

## Result — broadened cross-check (44 programs)

```
total cross-checked     : 44
AGREE byte-identical    : 0      (expected: K1py and K1' are different compilers)
AGREE behavioral-only   : 42     (same predicted exit; byte-DIFF uniform + explained)
DISAGREE (both compiled, differ) : 0     <-- the load-bearing zero
drift exclusions (witness PARSE/typecheck reject) : 2
DISTINCT witness-reachable codegen arms exercised by AGREEING programs : 44 / 53
```

**Zero true disagreements.** Every program the witness could compile produced the **same
exit** as the from-raw kovc. The byte-DIFF on all 42 is **uniform and expected**: the
witness (Stage-30 standalone Python codegen; `exit42` = 4649 B) and K1'/K2 (v1.1
self-hosted kovc with `emit_start_bigstack`; `exit42` = 4184 B) are **two different
compilers**, not one kovc built two ways — same ELF entry `0x401000`, a fixed prologue/
runtime size delta. This is the **strong-independence behavioral form** (the byte form is
R1 below).

### The 44 distinct witness-reachable codegen arms exercised (honest dedup)

Counted as distinct **kovc.hx codegen-dispatch arms** (AST tag identity), after collapsing
every program label to its underlying arm (so e.g. `i64_div`/`i64_mul`/`i64_lit` do not
triple-count the i64 path; `f64_mul` reuses tag 34; the match-pattern variants all reduce
to the one MATCH dispatch tag 62). The deduplicated set:

```
tag  0  INT/i32 literal       tag 28  BAND (&)
tag  1  VAR (load)            tag 29  BOR  (|)
tag  2  ADD (+)               tag 30  BXOR (^)
tag  3  SUB (-)               tag 32  SHL  (<<)
tag  4  MUL (*)               tag 33  SHR  (>>)
tag  5  DIV (/)               tag 34  FLOATLIT_F64
tag  6  CMP_EQ (==)           tag 35  INTLIT_I64 (incl. >2^32)
tag  7  IF                    tag 36  INTLIT_U32
tag  8  LET                   tag 37  INTLIT_U8
tag  9  NEG (-e)              tag 38  INTLIT_U64
tag 10  WHILE                 tag 39  INTLIT_I8
tag 11  ASSIGN                tag 40  INTLIT_I16
tag 12  LET_MUT               tag 41  INTLIT_U16
tag 15  FN_LIST (multi-fn)    tag 50  TUPLE_LIT
tag 16  CALL                  tag 52  TUPLE_FIELD (t.0/t.1)
tag 19  CMP_NE (!=)           tag 53  INDEX (arr[i] read)
tag 20  CMP_LT (<)            tag 55  INDEX_STORE (arr[i]=e / __arena_set)
tag 21  CMP_GT (>)            tag 62  MATCH (+ or/range/guard/payload patterns)
tag 22  CMP_LE (<=)           tag 81  CAST (as; incl. i32<->i64 widen)
tag 23  CMP_GE (>=)           STRUCT  struct-literal + field-load
tag 24  MOD (%)               --------------------------------------------
tag 25  STR_LIT (&str)        43 numeric AST tags + 1 STRUCT arm = 44
tag 26  BNOT (~e)
tag 27  NOT  (!e)
```

Each is **dynamically exercised at runtime**, not merely present in the AST: every probe
computes its exit code through the arm (e.g. NEG/BNOT/NOT/MOD/SHR/CAST operate on
loop-accumulated **runtime** values, never a foldable constant; the str-lit length drives a
runtime loop). Eleven **new** probes were authored for previously-uncovered-but-reachable
arms — `corpus_gen/arm_{neg,bnot,not,mod,shr,str_lit,struct_field,cast_chain,u32_width,
i8_width}.hx` (10 agree on both compilers) — closing NEG, BNOT, NOT, MOD, SHR, STR_LIT,
struct-multifield-read, the i32↔i64 cast chain, and the u32/i8 width arms that the prior
27-program cross-check had not dynamically exercised.

This is **44/53 ≥ 40**, an honest count under strict per-arm deduplication, with the GPU/PTX
carve-out below accounting for the bulk of the residual to 53.

### Multi-input differential (charter §1.1 item-2: behavioral match must be multi-input)

To ensure a behavioral match is **not a single-point coincidence** masking a real
divergence, a representative subset of arms was run over **5 distinct inputs each** (the
arm's operand is a runtime-accumulated value parameterized across the 5 variants), compiling
**both** the witness K1py and the from-raw K2 and requiring identical exits on **every**
input (`verification/py_witness/multiinput_diff.sh`,
`_multiinput_results.txt`):

```
neg  : 5/5 inputs agree     (NEG over {10,30,55,80,158})
mod  : 5/5 inputs agree     (MOD over {3,11,17,23,30})
shr  : 5/5 inputs agree     (SHR over {5,10,20,21,30})
bnot : 5/5 inputs agree     (BNOT over {7,19,42,85,120})
cast : 5/5 inputs agree     (i32->i64->i32 over {3,11,21,50,60})
=> every arm agrees across ALL inputs: the behavioral match is multi-input, not single-point.
```

---

## Per-arm witness-parse status: reached vs DRIFT-EXCLUDED

### Witness-REACHABLE (parsed-and-compiled by the Stage-30 witness, counted): the 44 above.

### DRIFT-EXCLUDED (Stage-30 witness PARSE/typecheck-rejects — NOT counted toward 44)

The snapshot is **Stage 30**, **before** most of v1.1's surface. The witness rejects:

| Surface | kovc tag(s) | Witness failure | Covered instead by |
|---|---|---|---|
| struct **field-STORE** `p.x = v` | 79 | TYPECHECK (invalid assign target) | from-raw fixpoint + gcc-DDC |
| `break` / `continue` | (control) | NotImplemented (Stage-30 backend gap) | from-raw fixpoint + gcc-DDC |
| `bf16` / `f16` scalar float literals | 42 / 80 | NotImplemented (no F16C codegen path) | from-raw fixpoint + gcc-DDC |
| inherent `impl P { fn m(self) }` | (method) | PARSE (`expected COLON` at `self`) | from-raw fixpoint + gcc-DDC |
| trait `impl Trait for P` / trait-default | (trait) | PARSE / TYPECHECK | from-raw fixpoint + gcc-DDC |
| closures `\|x\| ...` | (closure) | PARSE (`expected expression`) | from-raw fixpoint + gcc-DDC |
| turbofish `id::<T>` / `Box::<f32>{..}` | (generic) | PARSE / TYPECHECK | from-raw fixpoint + gcc-DDC |
| generic-fn/struct bracket params `[T]` | (monomorph) | PARSE / TYPECHECK | from-raw fixpoint + gcc-DDC |
| nested tuple field `t.1.0` | 52 (nested) | PARSE (`expected IDENT`) | (single-level t.0/t.1 IS reached) |
| non-generic 3-variant enum ctor in helper `Tri::A(x)` | 62-adjacent | TYPECHECK (`unresolved symbol: Tri::A`) | `result_inline` (Result::Ok/Err) + `e6_bare_match` (generic Opt::Some) DO reach the MATCH/payload arm |

Two of these surfaced as **COMPILE_FAIL** rows in the cross-check (`impl_method.hx`,
`arm_enum_payload3.hx`) — logged here as drift exclusions, **excluded from the 44**, not
counted as disagreements. (`arm_enum_payload3.hx` is committed as a corpus probe because
the **from-raw kovc compiles+runs it correctly to 42** — it marks the witness drift
boundary precisely; it is annotated drift-excluded in `broad_corpus.txt`.)

### CARVED OUT (per charter §1.1 — not a miss): GPU/PTX arms

`autodiff` / `tile` / `kernel` (tags 43 / 77 / 78) emit **PTX**, which ELF-byte-DDC
structurally cannot reach. Carved out by the charter, covered instead by
`gate_kovc.sh` step [3] PTX-regression + `capstone_audit.sh` finite-diff. **Listed, not
counted as a miss.** These account for most of the 53→44 residual together with the
drift-excluded v1.1 surface above.

---

## R1 — the stronger BYTE form (best-effort) — ATTEMPTED, NOT ACHIEVABLE (honest)

R1 attempted the strongest result: have the fenced witness mint `K1py` from the **current**
`k1src.hx`, then byte-compare at the self-host fixpoint. Two fenced fixes (gitignored
witness only; committed `kovc.hx`/seed/ladder **never** touched) were applied:

1. **Recursion + big-stack** — `sys.setrecursionlimit(2_000_000)` + a 1 GiB-stack thread
   (the 1.5 MB nested AST exceeds CPython's default depth). Resolves the prior agent's
   `RecursionError`.
2. **`if`-statement-unit typecheck shim** — at the single `A.If` divergence site in
   `verification/py_witness/helixc/frontend/typecheck.py`, the "branches differ" error is
   suppressed **exactly when one branch is `TyUnit`** (the statement-position `if` that the
   v1.1 kovc permits and the Stage-30 witness rejected at the 7 sites
   `k1src.hx:10433/10434/10453/10454/10455/11359/11371`). Narrow by construction: a real
   `i32`-vs-`f64` (etc.) mismatch still errors.

**Result (`verification/py_witness/_r1_results.txt`):**

```
[1] from-raw K1' = seed.bin(k1src.hx) = 600783 B (sha a435b6ca…); K2_seed = 606680 B (sha 03a456fe…)
[2] witness K1py = witness(k1src.hx):
       parse:    OK  (735 fns, 735 items)      <-- shim works: full current source parses
       typecheck: OK                            <-- the 7 if-unit sites no longer reject
       codegen:   OK  -> K1py = 1019766 B (sha 7394de44…)
    K2_python:  NOT produced  -- running K1py to mint K2 from BIG -> "Illegal instruction (core dumped)"
[3] (i)  K1py (1019766 B) vs K1' (600783 B) -> differ (expected: two different compilers)
    (ii) K2_python vs K2_seed -> UNAVAILABLE (K1py is not a runnable kovc; cannot complete the fixpoint)
```

**Honest conclusion — the byte form is NOT achievable with this witness, and that is
expected.** The fenced shim is a genuine advance — the Stage-30 witness now **parses,
typechecks, and code-generates the full current `k1src.hx`** (the prior blocker is gone).
But the K1py it emits **is not a working kovc**: executing it to mint K2 traps with an
illegal instruction. The Stage-30 witness's *codegen* predates v1.1 and, applied to the
entire 1.5 MB self-hosting compiler, produces a binary whose behavior diverges from a
runnable kovc — so it cannot mint `K2_python`, and the fixpoint byte-identity
(`K2_python == K2_seed`) cannot be computed. (The original DDC's byte-identity in
`K_DDC_RESULT.md` was at this same K2 fixpoint, and it held *because the then-current source
was within the witness's working codegen*; the v1.1 source is not.) The witness remains a
sound **second parser/typechecker** of the current source and a sound **second codegen on
the corpus-sized programs** (the 42 that agree) — which is exactly the behavioral form
below. **Per the charter (byte OR behavioral), the ≥40-arm behavioral form is the gate; R1
is logged as attempted, advanced past the prior blocker, and honestly not completable.** No
unbounded time was sunk: one serial seed build + one witness mint (~3.5 min total).

---

## R2 — the v1.1-surface drift exclusion, honestly (the real residual)

This is the load-bearing honesty section. **The Stage-30 Python witness is frozen before
the v1.1 language surface**, so an entire class of arms is **not reachable by this second
witness at all**:

- struct **field-store** (tag 79), `break`/`continue`, `bf16`/`f16` literals (tags 42/80),
  and the whole **generics / traits / closures / turbofish** surface.

These are **NOT** cross-checked by the Python witness — they are **drift exclusions**, and
they are **not** counted toward the 44/53. That is a **real residual**: an independent
second compiler does **not** re-derive these v1.1-surface arms. Stating it plainly: *the
v1.1-surface arms are un-DDC'd by the Python witness.*

They are covered **instead** by two other, independent mechanisms already in the trust
chain — but it must be said clearly that these are a **different** form of assurance than a
second-witness DDC of those exact arms:

1. **The from-raw self-host fixpoint `K2 == K3 == K4` (byte-identical).** The current
   `kovc.hx` — which *contains and exercises* the v1.1 surface in its own compilation of the
   1.5 MB corpus and bootstrap — mints itself to a byte-identical fixpoint via the from-raw
   seed lineage. This proves the v1.1-surface **codegen is deterministic and self-consistent
   under the from-raw compiler**, but it is **single-lineage** (one compiler agreeing with
   itself), not a second independent witness.
2. **The gcc-vs-M2-Planet seed diverse-double-compile (DC1–DC3, commit `72faee0`,
   `SEED_DDC_CROSSCHECK.md`).** A **different-lineage** second witness (a gcc-built seed vs
   the M2-Planet-built seed) with **zero M2-Planet ancestry on the gcc side** cross-checks
   the seed. This is a genuine second lineage, but it certifies the **seed**, not the full
   v1.1-surface codegen arms program-by-program.

### What this proves / does not prove

- **Proves:** over a broadened corpus exercising **44 of 53** distinct kovc value-codegen
  arms (vs ~15 dynamically cross-checked at the start of T1), a **second, genuinely
  independent compiler** (the fenced Stage-30 Python `helixc`, different language +
  toolchain + codegen, no shared backend) **behaviorally agrees** with the from-raw `kovc`
  on **every** witness-reachable program, each to its predicted exit, with the byte-vs-
  behavioral split reported and **zero true disagreements**. The fence stays intact
  (1 committed `.py`); the witness is never committed.
- **Does NOT prove:** (a) that the **v1.1-surface arms** (generics/traits/closures/
  turbofish/field-store/break/continue/bf16-f16) are agreed by an *independent second
  compiler* — the frozen Stage-30 witness cannot parse them, so they are an **honest drift
  exclusion**, covered only by the *single-lineage* from-raw fixpoint and the *seed-level*
  gcc-DDC, which are weaker for those specific arms than a program-by-program second-witness
  DDC would be. (b) That kovc is **bug-free** — DDC proves equivalence to a second witness on
  the reached arms, not absolute correctness; a bug present **identically in both backends**
  or in the **shared host runtime** (Linux/WSL kernel, the `cmp` harness) is **not** caught
  (the Wheeler trusted-runtime residual, unchanged from `K_DDC_RESULT.md`). (c) The GPU/PTX
  arms — carved out, covered by PTX-regression + capstone finite-diff, not by ELF-DDC.

**Closing the v1.1-surface residual** (if later required) needs a **second diverse witness
that *can* parse the v1.1 surface** — e.g. porting the Helix-native test infra to a
different-lineage compiler, or extending the gcc-DDC to a feature-diverse corpus. Until
then, the v1.1-surface second-witness DDC is a **named, open residual**, not a closed claim.

---

## Reproduce from scratch (WSL, repo root)

```
# 1. restore the witness (gitignored):
mkdir -p verification/py_witness/helixc
git archive v0-pre-k4-full-with-python:HELIX_STAGE30_COMPILER_SNAPSHOT/helixc \
  | tar -x -C verification/py_witness/helixc/
# 2. GATE 0 (witness mints exit42):
bash verification/py_witness/run_witness.sh \
  verification/py_witness/helixc/examples/exit42.hx /tmp/w.bin     # -> exit 42
# 3. broadened DDC (one serial session: from-raw K2 + probe-check + cross-check + arm count):
bash verification/py_witness/run_all_broad.sh                       # -> 44 progs, 42 agree, 44/53 arms
#    results persisted to verification/py_witness/_broad_results.txt
# 4. (best-effort) R1 byte form (fenced recursion+bigstack + if-unit typecheck shim):
bash verification/py_witness/r1_byteform.sh                         # -> K2_python vs K2_seed fixpoint
#    results persisted to verification/py_witness/_r1_results.txt
```

SERIAL discipline: only **one** seed/K1 build runs at a time (a prior agent wedged WSL
`/mnt/c` I/O with concurrent builds). `run_all_broad.sh` and `r1_byteform.sh` each do their
single build in one WSL session and guard against a concurrent build before starting.

---

*Status: P0–P4 DONE. Behavioral-form **DDC_BROAD_PASS** holds: fence == 1 (item 1), 42/42
witness-reachable programs agree with 0 true disagreements (item 2), 44/53 witness-reachable
arms ≥ 40 (item 3). Item 4 (quince) folds into the FINALE audits. R1 byte form reported
above (best-effort). The v1.1-surface drift exclusion is the named honest residual (R2).
Fence intact throughout — 1 committed `.py`, witness + shim gitignored.*

---

# v1.3 V5 — broaden the DDC toward the v1.1 surface (the R2 residual, materially reduced)

**Date:** 2026-06-04 · **Repo:** Kovostov-Native @ branch `main` · prior commit `f7d77fc`
· from-raw K1 = `seed.bin(k1src.hx)` from the committed `kovc.hx`, sha
`64d514940c69214e4e35cdafab30080a1e7332d3b2b6029915cc6b1426a16c10`.

**Charter:** `docs/HELIX_V1_3.md` §1 **V5** — "broaden the DDC toward the v1.1 surface.
… DoD: materially increase independent DDC coverage of the v1.1 surface — via an
extended/second independent witness OR a second-compiler behavioral cross-check over the
v1.1-surface corpus — and HONESTLY document whatever residual remains."

## What V5 does (and what it directly targets)

The R2 section directly above names the open residual at the end of T1: **the Stage-30
Python witness is frozen *before* the v1.1 language surface**, so generics / traits /
closures / turbofish / wide-field / bf16 are **un-DDC'd by a second compiler** — covered
only by the *single-lineage* from-raw fixpoint and the *seed-level* gcc-DDC, which are
weaker for those exact arms than a program-by-program second-witness check. R2 states the
fix explicitly: *"Closing the v1.1-surface residual … needs a second diverse witness that
**can** parse the v1.1 surface."*

**V5 builds exactly that second witness** — and uses it to behaviorally cross-check the
v1.1 surface against the from-raw kovc. This is approach (b) of the DoD menu (a
second-compiler **behavioral** cross-check over the v1.1-surface corpus).

- **The second witness:** `verification/py_witness/v11_interp/interp.py` — a **FENCED,
  GITIGNORED, zero-kovc-lineage** Helix interpreter. It tokenizes, parses, monomorphizes,
  and tree-walk-evaluates the `.hx` **source** and computes each program's exit value
  **purely from the source semantics**. It is a *different kind* of implementation than
  kovc (a tree-walking interpreter, not an ELF-emitting codegen) and shares **no code**
  with kovc.
- **The corpus:** `verification/py_witness/v11_interp/v11_corpus.txt` — **44 rows**
  (`path|expected_exit|surface|new_arms`) over precisely the v1.1 surface the Stage-30
  witness parse-rejects: 18 generics, 8 traits, 10 closures, 1 turbofish-on-enum-ctor,
  4 wide-field (i64/u64/f64), 3 bf16-arith. The documented-**bound** programs
  (`M5_bare_generic_bound`→0, `M7_privacy_bound`→42, `L3_nonexhaustive_bound`→42) are
  **deliberately EXCLUDED**: there kovc *intentionally* diverges from strict/correct
  semantics, so a behavioral "agreement" would be meaningless. They stay covered by the
  gate's bound rows, not by this independent behavioral cross-check.
- **The harness:** `verification/py_witness/v11_interp/xcheck.sh` (FENCED, GITIGNORED).
  Builds the from-raw kovc **once** (`assemble_k1.sh` → `seed.bin k1src.hx /tmp/K1.bin`,
  ~17 min this run), reuses it for all 44 compiles. Per row: compile the `.hx` with the
  from-raw K1 → run the ELF (timeout-wrapped) → `kovc_exit`; run `python3 interp.py
  <path>` → `interp_exit`; assert **`kovc_exit == interp_exit == expected_exit`**.
  Durable results streamed to `_xcheck_results.txt` as it goes.

## Result — the v1.1-surface behavioral cross-check (44 programs)

```
rows cross-checked                 : 44
AGREE  (kovc == interp == expected): 44      <-- every program, all three values match
DISAGREE                           : 0       <-- the load-bearing zero (none to resolve)
UNREACHABLE (interp raised InterpError) : 0   <-- the interpreter evaluated every program
```

**Zero disagreements, zero unreachable.** Every one of the 44 v1.1-surface programs has its
exit value **independently re-derived** by the zero-lineage interpreter, and the from-raw
kovc's actual ELF exit **agrees** — each also matching the program's documented expected
exit. Full per-row table in `verification/py_witness/v11_interp/_xcheck_results.txt`.

### New value-codegen arms independently cross-checked (beyond the Stage-30 witness's 44)

These are the v1.1-surface **kovc.hx value-codegen-dispatch arms** that the Stage-30 Python
witness cannot reach (it parse-rejects the surface) and that the interpreter now
behaviorally cross-checks for the **first time by a second implementation**:

```
MKCLOSURE     kovc.hx tag 82  -- capturing-closure OBJECT construction (v1.3 V3)
CLOSURE_DISP  emit_closure_dispatch -- the closure-VALUE indirect-call dispatch (in AST_CALL)
WIDEFIELD8    AST_TUPLE_FIELD (tag 52) 8-byte REX.W field-read sub-arm (i64/u64/f64; v1.3 V1)
BF16_ARITH    emit_bf16_binop -- bf16 add/mul convert-op-convert + RNE round-back (v1.3 V4)
--------------------------------------------------------------------------------------------
4 NEW arms independently cross-checked.   (F16_ARITH listed in the corpus legend but
                                           NOT counted -- no f16-arith fixture; see residual.)
```

Beyond these 4 new arms, the remaining 40 rows exercise already-counted value arms (INT/VAR/
ADD/SUB/MUL/DIV/CMP/IF/WHILE/CALL/MATCH/CAST/STRUCT/i64-lit/f64-lit/…) **but through the
previously-drift-excluded v1.1 SURFACE** (monomorphized generics, trait-method dispatch,
non-capturing closures, turbofish) — so each is a genuine second-implementation behavioral
check of that surface, even where it does not add a new denominator arm.

## Genuine-independence argument (the load-bearing discipline)

The whole value of a DDC rests on the two sides being independent. interp.py was
spot-audited from disk this campaign and confirmed:

1. **It never reads any kovc artifact.** Its only imports are `sys` and `struct`; its only
   file `open()` reads the `.hx` **source** passed as `argv[1]`. There is **no**
   `subprocess`, `os.system`, socket, ELF/binary read, or any reference to a kovc binary,
   emitted ELF, or kovc stdout anywhere in the file (the only textual occurrences of "kovc"
   are in comments describing this discipline). The harness runs kovc's ELF **separately**
   and compares the two exit values; the interpreter has no knowledge of that side.
2. **Its numeric semantics are re-derived from the format definitions, not copied from
   kovc.hx codegen.** i32/i64/u64 = two's-complement wrap (`wrap_int`); f32/f64 = IEEE-754
   binary32/binary64 via Python `struct` round-trips (so f32 intermediate rounding matches
   the format, independently of how kovc emits `addss`); **bf16/f16 = round-to-nearest-even
   convert-op-convert**, re-derived from first principles (`f32_to_bf16_rne` adds the
   round-half-to-even bias `0x7FFF + lsb` then masks `0xFFFF0000`; `f32_to_f16_rne` uses
   Python's IEEE binary16 `'e'` round-trip). The bf16 RNE algorithm IS the canonical IEEE
   round-back — it is the *correct derivation*, not a copy of kovc's emitter (and it was
   validated by V4's bit-exact gate from the opposite direction).
3. **Different implementation strategy.** kovc is an ELF-emitting native-code compiler; the
   witness is a tree-walking interpreter. They share no lexer, parser, type logic, or
   backend.

This is a **BEHAVIORAL oracle of medium independence** — a second *interpreter* that
re-derives behavior, not a byte-identical second *compiler*.

## Honest residual (V5 — paired with precise scope)

- **The byte-identical form is impossible by construction here.** interp.py emits **no
  machine code at all** (it computes an exit value directly), so there is no second ELF to
  `cmp` against kovc's. V5 is therefore necessarily a **behavioral** cross-check, never the
  byte form. This is *weaker* than a byte-identical second-compiler DDC of these arms — it
  proves the two implementations **agree on observable behavior** (the exit value), not that
  kovc's *instruction stream* is independently reproduced. (The R1 byte form above remains
  not-achievable with the Stage-30 witness, unchanged.)
- **The standard shared-bug DDC residual is unchanged.** A bug present **identically** in
  both interp.py's re-derivation **and** kovc — or in the **shared host runtime** (the WSL
  Linux kernel, the process-exit path, the `timeout` harness) — would not be caught. DDC
  proves *equivalence to a second witness on the reached programs*, not *absolute
  correctness*. (The Wheeler trusted-runtime residual, as in `K_DDC_RESULT.md`.)
- **f16-arithmetic is NOT fixtured.** The corpus gates **bf16** arithmetic bit-exactly
  (3 rows, the `BF16_ARITH` arm) but has **no f16-arith program**, so the `F16_ARITH` arm
  (F16C `vcvtph2ps`/`vcvtps2ph`) is listed for completeness and **explicitly not counted**
  as cross-checked. f16 arith remains covered only by its V4 gate row + the from-raw
  fixpoint — not by this independent witness.
- **No UNREACHABLE rows this run.** All 44 corpus programs were within the interpreter's
  evaluable subset; none degraded to "logged-but-not-cross-checked." (The mechanism exists
  — an InterpError is logged honestly as UNREACHABLE and never counted as agreement — it
  simply did not fire.)
- **Scope of the corpus.** The 44 rows are a *representative* exercise of the v1.1 surface
  (every sub-form of generics/traits/closures + each wide-field width + bf16 add/mul/
  roundtrip), not an exhaustive enumeration of every program kovc accepts. The deliberately
  **excluded** bound programs (M5/M7/L3) are not cross-checked here **by design** (kovc
  intentionally diverges from strict semantics there) and stay covered by the gate's bound
  rows. f64/f16/i64 arithmetic *outside* the wide-field and bf16 fixtures is covered by the
  main gate, not necessarily by this witness.

## What V5 proves / does not prove

- **Proves:** the v1.1 surface (generics / traits / closures / turbofish / wide-field /
  bf16) — the R2 *named open residual* — now has an **independent BEHAVIORAL cross-check**
  by a **second, zero-kovc-lineage** implementation. Over **44/44** v1.1-surface programs,
  the interpreter's source-re-derived exit and the from-raw kovc's ELF exit **agree** (each
  also matching the expected exit), with **0 disagreements** and **0 unreachable**; this
  newly cross-checks **4** value-codegen arms (MKCLOSURE / CLOSURE_DISP / WIDEFIELD8 /
  BF16_ARITH) that the Stage-30 witness's 44 could not reach, plus the rest of the surface
  through previously-drift-excluded forms. The R2 residual is **materially reduced**, which
  is precisely the V5 bar (a real reduction of the gap, not 53/53 forced).
- **Does NOT prove:** (a) a **byte-identical** second-compiler reproduction of these arms —
  impossible by construction (the witness emits no code; V5 is behavioral). (b) **bug-free**
  kovc — a bug shared by both backends or in the shared host runtime is not caught. (c)
  **f16 arithmetic** by this witness — not fixtured. (d) the GPU/PTX arms — out of scope
  (covered by PTX-regression + capstone, as before).

**The R2 residual is now: reduced from "v1.1 surface un-DDC'd by a second compiler" to
"v1.1 surface behaviorally cross-checked by a second zero-lineage interpreter; byte-identical
second-compiler form still not achievable, and f16-arith still un-fixtured."** That is a
genuine, honestly-scoped improvement.

## Fence + toolchain invariants (intact)

V5 changes **nothing** in the shipped toolchain. The interpreter, the corpus, the harness,
and the durable results **all** live under the git-check-ignored `verification/py_witness/`
and are **never** committed. No `kovc.hx` / `parser.hx` / `lexer.hx` / `scripts/gate_kovc.sh`
change — so **no full self-host re-mint is required** (the universal gate's fixpoint is
unperturbed; the last GREEN mint `f7d77fc` stands). Verified this run: `git ls-files "*.py"`
== **1** (`verification/oracle/oracle_train.py`); the whole `verification/py_witness/` tree
is git-check-ignored (interp.py + v11_corpus.txt + xcheck.sh + _xcheck_results.txt all
fenced). The only committed change for V5 is **this documentation section** (plus the
tracker entry); all 44 corpus `.hx` fixtures were already committed under
`stage0/helixc-bootstrap/corpus_gen/`, so the cross-check is reproducible from the tree.

## Reproduce (WSL, repo root)

```
# one serial session: build the from-raw kovc once, cross-check all 44 v1.1-surface rows
bash verification/py_witness/v11_interp/xcheck.sh
#   -> "XCHECK TALLY: rows=44 AGREE=44 DISAGREE=0 UNREACHABLE=0" ; "XCHECK_PASS"
#   per-row results persisted to verification/py_witness/v11_interp/_xcheck_results.txt
# the second witness in isolation (re-derive ONE program's exit from source, no kovc):
python3 verification/py_witness/v11_interp/interp.py \
  stage0/helixc-bootstrap/corpus_gen/V1_i64_wide_field.hx          # -> 50
```

*Status (V5): **DONE** — 44/44 v1.1-surface programs agree `kovc(from-raw) == interp ==
expected`, 0 disagreements, 0 unreachable; 4 new value-codegen arms independently
cross-checked (MKCLOSURE/CLOSURE_DISP/WIDEFIELD8/BF16_ARITH) beyond the Stage-30 witness's
44. The R2 v1.1-surface residual is materially reduced (now a behavioral second-interpreter
cross-check). Honest residual: behavioral-not-byte (impossible by construction here), the
shared-bug DDC residual, f16-arith un-fixtured, no rows unreachable this run. Fence intact —
1 committed `.py`; the interpreter + corpus + harness + results stay gitignored. No
toolchain change → no re-mint. Commit `9ef461d` (V5 docs section + tracker entry).*
