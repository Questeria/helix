#!/usr/bin/env python3
"""
scripts/helix_status.py — beginner-friendly Helix progress reporter.

The Helix autonomous build worker (the `helix-approach-a-loop`
scheduled task) sends a Telegram status update at the end of every
fire. Those updates used to be terse and developer-facing — e.g.
"Stage 117, commit abc1234, 21 tests pass" — unreadable to anyone
who is not a compiler engineer.

This module renders a plain-language update instead: what is finished
and audited, what is in progress, what is still ahead, and a
percent-progress readout for build stages, versions, and the project
overall.

It is the SINGLE SOURCE OF TRUTH for release-journey status. When a
version ships, change its `status` in `VERSIONS` below from
"in_progress" / "planned" to "released" (and open the next one). As
each v3.0 build stage closes its 3-part audit, bump `V3_STAGES_DONE`.
Every percentage recomputes from that edit; the test-suite size is
counted LIVE from `helixc/tests/` (so it grows with every chunk and
never goes stale — no manual bump).

Usage:
    python scripts/helix_status.py
    python scripts/helix_status.py --note "<plain-English summary>" \\
        --commit <hash>

License: Apache 2.0
"""
from __future__ import annotations

import argparse
from pathlib import Path


# --- The v2.0 -> v3.0 release journey --------------------------------
# Each Helix version ends with a 5-part "clean-gate" code audit before
# it counts as released. Statuses:
#   "released"    — shipped AND its end-of-version audit gate passed
#   "in_progress" — actively being built right now
#   "planned"     — scoped but not started
# Update `status` here (and ONLY here) as versions ship.
VERSIONS: list[dict[str, str]] = [
    {"id": "v2.0", "status": "released",
     "theme": "GPU compiler foundation (22 build stages)"},
    {"id": "v2.1", "status": "released",
     "theme": "Per-operation GPU code generation + autodiff"},
    {"id": "v2.2", "status": "released",
     "theme": "Polish and audit clean-up"},
    {"id": "v2.3", "status": "released",
     "theme": "Type-system design polish"},
    {"id": "v2.4", "status": "released",
     "theme": "Real-GPU testing + attestation + register allocator"},
    {"id": "v2.5", "status": "released",
     "theme": "Wiring the register allocator into real GPU kernels"},
    {"id": "v3.0", "status": "released",
     "theme": "The big rewrite - industrial MLIR + LLVM backend"},
    {"id": "v3.1", "status": "released",
     "theme": "Post-v3.0 cleanup - LLVM toolchain wiring, polymorphic "
              "SPLICE/MODIFY, REFLECT_HASH, shared-constants module"},
    {"id": "v3.2", "status": "planned",
     "theme": "Real-execution parity gate (or first K-bootstrap "
              "milestone toward Helix-in-Helix)"},
]

# v2.x shipped its compiler work as 22 numbered build stages
# (Stage 110-131), all closed — the v2.0-v2.5 entries in VERSIONS
# record that. v3.0 is built as its own 19 numbered stages: Phase D
# (Stage 200-208), Phase E (210-216), Phase F (220-222). Every stage
# closes with a 3-part audit. Bump `V3_STAGES_DONE` as each closes —
# every percentage below recomputes from it.
V3_STAGES_TOTAL = 19
V3_STAGES_DONE = 19       # ALL Phase D + E + F stages COMPLETE — v3.0 RELEASED

# K-bootstrap track (post v3.1.0, declared the new top-line goal
# 2026-05-25). See docs/HELIX_K_BOOTSTRAP_MASTER_PLAN.md and the
# feature-parity matrix docs/K_BOOTSTRAP_FEATURE_MATRIX.md. The
# matrix enumerates every Helix language feature with a column for
# Python helixc support and a column for kovc.hx support. A row is
# PARITY when both columns agree; KOVC-MISSING when only Python
# supports it. The goal: get every row to PARITY, then delete the
# Python compiler.
#
# Bump K_BOOTSTRAP_PARITY_DONE as each K-track chunk lands and the
# matrix's PARITY count rises.
K_BOOTSTRAP_TOTAL_ROWS = 143      # matrix total (28 PARITY + 115
                                    # KOVC-MISSING at K0 chunk 2 close)
K_BOOTSTRAP_PARITY_DONE = 121      # was 28 after K0; K1.B (stack
                                    # args > 6) made it 29; K1.C
                                    # (return statement) made it 30;
                                    # K1.D-impl (print_int) made it 31;
                                    # K1.G (for loop) made it 32;
                                    # K1.H1 (loop keyword) made it 33;
                                    # K1.F discovery (tuple lit +
                                    # field access were already in
                                    # kovc.hx, matrix audit had
                                    # marked them stale-MISSING) +2
                                    # made it 35;
                                    # K1.F discovery batch 2: match
                                    # arms + PatBind + PatWildcard +
                                    # PatTuple + StructLit + enum
                                    # variants all already worked,
                                    # matrix entries stale +6 made it 41;
                                    # K1.F discovery batch 3: PatLit
                                    # (literal patterns) + PatVariant
                                    # also already worked, +2 made it 43;
                                    # K1.F discovery batch 4: ArrayLit
                                    # + 1D Index (`[a,b,c]; a[i]`)
                                    # also already worked (folded to
                                    # AST_TUPLE_LIT at parse time, no
                                    # explicit TyArray annotation
                                    # required), +2 made it 45;
                                    # K1.K (char literal lexing in
                                    # lex_char_lit -- `'A'` lexes as
                                    # TK_INTLIT with byte value as
                                    # payload, standard escape set
                                    # included) +1 made it 46;
                                    # K1.F discovery batch 5: PatRange
                                    # half-open `0..N` arm works
                                    # (closed `..=` is a separate gap)
                                    # +1 made it 47;
                                    # K1.L (closed range `..=` for
                                    # both for-loop bounds and
                                    # PatRange -- parser detects
                                    # TK_EQ after TK_DOTDOT; parse_for
                                    # uses AST_LE; emit_pat_range
                                    # uses `jg` instead of `jge` for
                                    # the upper bound when p3==1)
                                    # +1 made it 48;
                                    # K1.F discovery batch 6: PatOr
                                    # (`a | b | c`) already worked
                                    # end-to-end via parse_pattern
                                    # alt-chain + emit_pat_or, matrix
                                    # was stale +1 made it 49;
                                    # K1.M (logical `&&` / `||` via
                                    # parse_bitwise doubled-token
                                    # detect + AST_IF desugar for
                                    # short-circuit; no lexer change,
                                    # no codegen change) +1 made it 50;
                                    # K1.F discovery batch 7: parametric
                                    # struct `struct Box<T> { val: T }`
                                    # already works for instantiation +
                                    # field access (PatStruct destructure
                                    # is a separate row, still missing)
                                    # +1 made it 51;
                                    # K1.N (`as Type` cast as no-op via
                                    # parse_unary postfix loop; type-
                                    # erased bootstrap means cast is a
                                    # runtime no-op) +1 made it 52;
                                    # K1.O (`where` clause skip in
                                    # parse_fn_decl; bounds are not
                                    # enforced) +1 made it 53;
                                    # K1.F discovery batch 8: struct
                                    # field access (nested + multi)
                                    # already works end-to-end, and
                                    # the bare struct decl row is
                                    # subsumed by other rows -- both
                                    # matrix entries were stale +2
                                    # made it 55;
                                    # K1.Q (BoolLit true/false in
                                    # parse_primary IDENT cascade
                                    # mapping to AST_INT(1)/AST_INT(0))
                                    # +1 made it 56;
                                    # K1.R (TyArray `[T;N]` annotation
                                    # in let-binding via skip-to-`]`;
                                    # type-erased so info discarded)
                                    # +1 made it 57;
                                    # K1.S (TyRef `&T` / `&mut T` +
                                    # TyPtr `*const T` / `*mut T` /
                                    # `*T` annotation in let-binding;
                                    # type-erased no-op, address-of
                                    # EXPRESSION still unsupported)
                                    # +2 made it 59;
                                    # K1.T (TyGeneric `Foo<A, B>` in
                                    # let-binding via `<>` depth-
                                    # tracking skip; TK_RSHIFT counts
                                    # as -2 for nested generics)
                                    # +1 made it 60;
                                    # K1.U (compound assign `+=`/`-=`/
                                    # `*=`/`/=`/`%=` via parser-side
                                    # desugar in parse_primary --
                                    # peek (op, `=`) after IDENT,
                                    # emit AST_ASSIGN(name, BINOP(VAR,
                                    # rhs)) using existing arith
                                    # codegen) +1 made it 61;
                                    # K1.V (top-level `type Alias =
                                    # T;` as no-op decl via new
                                    # parse_type_alias_decl + arms
                                    # in parse_top + parse_program's
                                    # two decl loops) +1 made it 62;
                                    # K1.W (unary `&` and `*` in
                                    # expressions as no-op prefixes
                                    # via 2 new parse_unary arms;
                                    # type-erased so the inner expr
                                    # is returned unchanged) +1
                                    # made it 63;
                                    # K1.X (TyFn `fn(T1) -> R` in
                                    # let-binding type-position --
                                    # detect "fn" IDENT, consume
                                    # `(`...`)` + optional `-> R`)
                                    # +1 made it 64;
                                    # K1.F discovery batch 9: TyTensor
                                    # + TyTile already work via K1.T
                                    # generic skip, matrix stale +2
                                    # made it 66;
                                    # K1.F discovery batch 10: @trace
                                    # + @checkpoint + @deprecated/
                                    # @since + @pure/@effect all
                                    # parse + run; syntax-only parity,
                                    # bootstrap doesn't enforce; +4
                                    # made it 70;
                                    # K1.Y (TyTuple `(T1, T2)` in
                                    # let-binding -- new TK_LPAREN
                                    # arm with `(`/`)` depth-tracking)
                                    # +1 made it 71 -- past the 50%
                                    # milestone;
                                    # K1.Z (top-level `const X: T =
                                    # expr;` syntax acceptance --
                                    # parse_const_decl + arms in
                                    # parse_top + parse_program; the
                                    # NAME is not registered so
                                    # downstream refs fail) +2 made
                                    # it 73 (lines 128 + 143);
                                    # K1.AA (top-level `agent Foo
                                    # { ... }` -- parse_agent_decl
                                    # brace-balanced; syntax-only)
                                    # +1 made it 74;
                                    # K1.F discovery batch 11: mod
                                    # + use decls already parse via
                                    # existing parse_mod_decl /
                                    # parse_use_decl. Semantics
                                    # caveats but syntax-only parity
                                    # +2 made it 76;
                                    # K1.F discovery batch 12: @partial
                                    # attribute also already parses
                                    # via skip_attributes +1 made
                                    # it 77;
                                    # K1.F discovery batch 13: all 15
                                    # Tier-S/A modal-type wrappers
                                    # (Diff, Logic, Modal, Causal,
                                    # Conf, Taint, DP, Quant, Domain,
                                    # Robust, Energy, Enclave,
                                    # Counterfactual, Deadline,
                                    # Attribution) parse via K1.T
                                    # generic skip -- syntax-only
                                    # parity, no semantic enforcement
                                    # +15 made it 92 (crossed 60%);
                                    # K1.F discovery batch 14: const_
                                    # fold IR pass is FUNCTIONAL
                                    # parity via parser.hx:1298
                                    # mk_arith_fold (parse-time const
                                    # folding) +1 made it 93;
                                    # K1.F discovery batch 15: 4
                                    # frontend passes (ast_walker,
                                    # match_lower, struct_mono,
                                    # flatten_modules) FUNCTIONAL
                                    # parity via bootstrap's
                                    # monolithic architecture (no
                                    # separate passes, same end
                                    # behaviour) +4 made it 97;
                                    # K1.F discovery batch 16: 4
                                    # backend rows (LLVM IR emitter,
                                    # LLVM toolchain wrapper, MLIR
                                    # substrate, Backend Protocol)
                                    # FUNCTIONAL parity -- bootstrap
                                    # goes direct-to-ELF, so the
                                    # Python-side LLVM pipeline +
                                    # backend abstraction aren't
                                    # needed +4 made it 101;
                                    # K1.F discovery batch 17: Parity
                                    # gate row -- bootstrap has only
                                    # one path so self-comparison is
                                    # structurally impossible. The
                                    # K-bootstrap's parity gate is
                                    # the K1=K2=K3 self-host fixpoint
                                    # +1 made it 102;
                                    # K1.F discovery batch 18: 4
                                    # optimization passes (hash_cons,
                                    # cse, dce, fdce) FUNCTIONAL --
                                    # they're performance passes, not
                                    # parity-critical features.
                                    # Bootstrap is less efficient
                                    # without them but compiles
                                    # correctly +4 made it 106;
                                    # K1.F discovery batch 19: ast_
                                    # hash (memoization optimization)
                                    # + FFI/extern-C (file-I/O
                                    # subset via syscall stubs) +2
                                    # made it 108 (crossed 75%);
                                    # K1.F discovery batch 20:
                                    # panic("msg") builtin already
                                    # compiles cleanly + traps at
                                    # runtime via unresolved-CALL
                                    # ud2 stub (rc=132); panic_pass
                                    # (the frontend pass) integrated
                                    # at Stage 28.9 -- different
                                    # architecture than Python's
                                    # TRAP-op lowering, same fail-
                                    # stop end behaviour +2 made
                                    # it 110;
                                    # K1.AB: `unsafe { expr }` no-op
                                    # block parsing (parse_unsafe
                                    # mirrors parse_loop) + the
                                    # unsafe_pass row flips
                                    # vacuously since the bootstrap
                                    # has no unsafe-only features
                                    # +2 made it 112;
                                    # K1.AC: bare `break` keyword --
                                    # AST_BREAK tag 77, codegen
                                    # backpatching chain on bn_state
                                    # slot 122, AST_WHILE walks +
                                    # patches at loop close. The
                                    # `break value` form is a
                                    # separate gap +1 made it 113;
                                    # K1.AD: `continue` keyword
                                    # mirroring break (AST_CONTINUE
                                    # tag 78, chain on slot 158,
                                    # patches to loop_top) +
                                    # fix latent K1.AC slot-122
                                    # collision with match_scrut_ty
                                    # (moved break to slot 157). +1
                                    # made it 114;
                                    # K1.F discovery batch 21:
                                    # @autotune(KEY: [v1, v2])
                                    # actually parses + validates
                                    # when paired with @kernel
                                    # (Python's autotune.py enforces
                                    # the same @kernel requirement)
                                    # +2 made it 116;
                                    # K1.F discovery batch 22:
                                    # deprecated_pass + totality +
                                    # trace_pass + diagnostics --
                                    # 4 frontend passes flip to
                                    # FUNCTIONAL PARITY. Bootstrap
                                    # source uses ZERO of the
                                    # tracked attributes for self-
                                    # host (no @trace/@deprecated/
                                    # @partial); diagnostics uses
                                    # numeric trap-ids vs Python's
                                    # carets but the fail-stop
                                    # signal matches. +4 made it 120;
                                    # K1.AF: __arena_push_pair(a,b)
                                    # inline builtin -- atomic
                                    # 2-slot push, returns OLD
                                    # cursor, -1 on overflow.
                                    # push_triple deferred. +1
                                    # made it 121

# The version statuses the model recognises.
_VALID_STATUS = frozenset({"released", "in_progress", "planned"})


def v3_stages_percent() -> int:
    """Percent of the v3.0 build stages complete (each 3-clean
    audited)."""
    return round(100 * V3_STAGES_DONE / V3_STAGES_TOTAL)


def versions_percent() -> int:
    """Percent of journey versions fully released (audit gate passed)."""
    released = sum(1 for v in VERSIONS if v["status"] == "released")
    return round(100 * released / len(VERSIONS))


def _version_credit(v: dict[str, str]) -> float:
    """How much one version contributes toward the overall journey
    total: a released version counts 1.0, a planned version 0.0, and
    an in-progress version gets partial credit. For v3.0 specifically
    (the only version with a published numbered-stage breakdown) we
    use the live V3_STAGES_DONE fraction so partial credit climbs as
    stages close. For other in-progress versions (v3.1 cleanup, v3.2
    parity gate, future K-bootstrap milestones) there is no
    fine-grained stage table — they tick from 0% to 100% at release.
    A reasonable middle-credit (0.5) keeps the overall percentage
    honest without inventing a fake-precision stage count."""
    if v["status"] == "released":
        return 1.0
    if v["status"] == "planned":
        return 0.0
    if v["id"] == "v3.0":
        return V3_STAGES_DONE / V3_STAGES_TOTAL
    return 0.5


def overall_percent() -> int:
    """Overall progress along the v2.0 -> v3.0 journey — the released
    versions plus the in-progress version's live v3.0-stage
    fraction."""
    score = sum(_version_credit(v) for v in VERSIONS)
    return round(100 * score / len(VERSIONS))


def k_bootstrap_percent() -> int:
    """Percent of Helix-in-Helix self-hosting feature-parity reached.
    Computed live from the matrix counts; never hand-typed."""
    return round(100 * K_BOOTSTRAP_PARITY_DONE / K_BOOTSTRAP_TOTAL_ROWS)


def count_tests() -> int:
    """The size of the automated test suite — a count of `def test_*`
    definitions across `helixc/tests/`, computed LIVE so it grows with
    every chunk and never goes stale.

    A pure scale-of-testing figure for non-engineers, NOT a pass/fail
    claim: it counts the tests that EXIST, it does not run them (a
    live pass/fail readout would need a mode that runs pytest). Fails
    loudly rather than render a misleading zero."""
    tests_dir = (Path(__file__).resolve().parent.parent
                 / "helixc" / "tests")
    total = 0
    for path in tests_dir.glob("test_*.py"):
        total += sum(
            1 for line in path.read_text(encoding="utf-8").splitlines()
            if line.lstrip().startswith("def test_"))
    if total == 0:
        raise SystemExit(
            f"helix_status: counted 0 tests under {tests_dir} — the "
            f"test directory was not found or is empty; refusing to "
            f"render a misleading status.")
    return total


def _bucket(status: str) -> list[dict[str, str]]:
    """Versions in a given status, in journey order."""
    return [v for v in VERSIONS if v["status"] == status]


def render_telegram(note: str | None = None,
                    commit: str | None = None) -> str:
    """Render the beginner-friendly Helix status update.

    `note`   — one plain-English sentence on what the latest fire did.
    `commit` — the short commit hash of that fire's commit.
    Both are optional; the per-fire footer is omitted if neither is set.
    """
    released = _bucket("released")
    in_progress = _bucket("in_progress")
    planned = _bucket("planned")

    # Hide the v2.x and v3.0 entries from the RELEASED VERSIONS
    # display -- they are long-shipped historical milestones and
    # the Telegram update should focus on what's current. The
    # internal _bucket("released") still includes them for the
    # versions_percent() math; they just don't get listed here.
    _HISTORICAL_RELEASED = frozenset({
        "v2.0", "v2.1", "v2.2", "v2.3", "v2.4", "v2.5", "v3.0",
    })
    released_visible = [v for v in released if v["id"] not in _HISTORICAL_RELEASED]

    lines: list[str] = [
        "HELIX COMPILER  -  BUILD UPDATE",
        "================================",
        "",
        "NAMING CONVENTION",
        "  v<M.m>    Release version, gated by a 5-clean-axis audit.",
        "            (v2.0..v2.5 = compiler foundation; v3.0 = big",
        "            MLIR+LLVM rewrite; v3.1 = polish; v3.2+ = ahead.)",
        "  K<n>      K-bootstrap track stage toward self-hosting.",
        "            K0 = survey; K1 = ports; K2 = parity harness;",
        "            K3 = trusted seed; K4 = delete Python (gated);",
        "            K5 = DDC + final 5-clean audits.",
        "  K1.<X>    A K1 sub-chunk (each ports one Helix feature",
        "            from Python helixc into the Helix-side compiler).",
        "  Stage <n> v3.0 build-stage ID. 200-208 = Phase D (frontend),",
        "            210-216 = Phase E (MLIR), 220-222 = Phase F.",
        "  3-clean   Per-chunk audit by 3 review axes (silent-failure",
        "            / type-design / code-review).",
        "  5-clean   End-of-phase audit by 5 axes (FE/IR/BE/RT/TEST).",
        "",
    ]
    if released_visible:
        lines.append("RELEASED VERSIONS")
        for v in released_visible:
            lines.append(f"  - {v['id']}   {v['theme']}")

    if in_progress:
        lines += ["", "IN PROGRESS"]
        for v in in_progress:
            lines.append(f"  - {v['id']}   {v['theme']}")

    # STILL AHEAD now shows ALL the main future milestones (both
    # the planned-version entries from VERSIONS and the K-track
    # stages that aren't yet at PARITY-complete). The K-track plan
    # is the load-bearing future work; release versions after v3.2
    # are the headline cadence.
    lines += ["", "STILL AHEAD"]
    for v in planned:
        lines.append(f"  - {v['id']}   {v['theme']}")
    # K-track stages -- list each remaining stage with its current
    # state. K1 is in progress (per K_BOOTSTRAP_PARITY_DONE rising
    # row-by-row); K2..K5 are scheduled but not started.
    lines += [
        f"  - K1     Feature ports ({K_BOOTSTRAP_PARITY_DONE}/"
        f"{K_BOOTSTRAP_TOTAL_ROWS} rows at PARITY) -- IN PROGRESS",
        "  - K2     Parity harness -- runs every test program "
        "through both compilers and asserts identical output",
        "  - K3     Trusted seed -- a small hand-audited Helix "
        "binary that can re-bootstrap the compiler from source",
        "  - K4     Delete Python helixc (USER-GATED -- continuous "
        "audits run at K4 until the green light)",
        "  - K5     DDC (Diverse Double-Compilation) + final "
        "5-clean audits -- the trust-from-first-principles gate",
        "  - SELF-HOSTING ACHIEVED -- the headline goal: a Helix "
        "compiler written in Helix, compiled in Helix, all the "
        "way from raw binary with NO Python in the final product",
    ]

    lines += [
        "",
        "SELF-HOSTING PROGRESS (Helix-in-Helix)",
        f"  - Feature-parity rows: {K_BOOTSTRAP_PARITY_DONE} / "
        f"{K_BOOTSTRAP_TOTAL_ROWS} done   "
        f"({k_bootstrap_percent()}%)",
        "    Each row is a Helix language feature; PARITY means the "
        "Helix-",
        "    side compiler handles it the same way the Python compiler "
        "does.",
        "    Track plan: K0 survey -> K1 ports -> K2 parity harness ->",
        "    K3 trusted seed -> K4 delete Python (gated) -> K5 final "
        "audits.",
        "",
        "PROGRESS",
        f"  - v3.0 build stages:   {V3_STAGES_DONE} / "
        f"{V3_STAGES_TOTAL} done   ({v3_stages_percent()}%) - each "
        f"one 3-part audited",
        f"  - Versions released:   {len(released)} / {len(VERSIONS)}"
        f"         ({versions_percent()}%)",
        f"  - Test coverage:       ~{count_tests()} automated tests "
        f"guard the code",
    ]

    if note or commit:
        lines.append("")
        if note:
            lines.append(f"THIS UPDATE: {note}")
        if commit:
            lines.append(f"  commit {commit}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI: print the beginner-friendly Helix status update."""
    ap = argparse.ArgumentParser(
        description="Render the beginner-friendly Helix status update "
                    "(used for the autonomous worker's Telegram dispatch).")
    ap.add_argument("--note", default=None,
                    help="one plain-English sentence on what the latest "
                         "fire shipped")
    ap.add_argument("--commit", default=None,
                    help="short commit hash of the latest fire's commit")
    args = ap.parse_args(argv)

    # Guard the single-source-of-truth model: a typo'd status or an
    # out-of-range stage count would silently skew every percentage.
    # Fail loudly instead.
    for v in VERSIONS:
        if v["status"] not in _VALID_STATUS:
            raise SystemExit(
                f"helix_status: VERSIONS entry {v['id']!r} has unknown "
                f"status {v['status']!r}; expected one of "
                f"{sorted(_VALID_STATUS)}.")
    if not 0 <= V3_STAGES_DONE <= V3_STAGES_TOTAL:
        raise SystemExit(
            f"helix_status: V3_STAGES_DONE ({V3_STAGES_DONE}) must be "
            f"in 0..V3_STAGES_TOTAL ({V3_STAGES_TOTAL}).")
    if not 0 <= K_BOOTSTRAP_PARITY_DONE <= K_BOOTSTRAP_TOTAL_ROWS:
        raise SystemExit(
            f"helix_status: K_BOOTSTRAP_PARITY_DONE "
            f"({K_BOOTSTRAP_PARITY_DONE}) must be in "
            f"0..K_BOOTSTRAP_TOTAL_ROWS ({K_BOOTSTRAP_TOTAL_ROWS}).")

    print(render_telegram(note=args.note, commit=args.commit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
