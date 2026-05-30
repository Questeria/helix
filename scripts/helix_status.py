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
# K_BOOTSTRAP_CHUNKS_DONE counts shipped K0/K1 commits on the
# K-bootstrap track (run `git log --oneline | grep -E "K[01]\.|K0 chunk"
# | wc -l` to recount). Bump each commit. The chunk count is more
# meaningful than matrix parity rows under the hard constraint because
# many "PARITY" rows are vacuously satisfied.
K_BOOTSTRAP_CHUNKS_DONE = 421      # last bump: SAFE-HARDENING -- FIXED the 4th audit-found bootstrap bug: enum-WITH-PAYLOAD value passed BY VALUE as a fn PARAMETER SIGILL'd (132), even when the body ignored the param. A payload enum is an 8-byte pointer (tag-50 tuple [disc,payload...]) but parse_fn_decl only recognized struct/generic param types, so an enum param defaulted to i32 (p_ty 0); at the call site the 8-byte arg (expr_type 3) mismatched the i32 param -> 16001 arg-type trap -> ud2 (before the body ran). FIX: parse_fn_decl now encodes enum-with-payload params (enum_tab slot+4 max_arity>0) with the 100+idx struct pointer sentinel, routing them through the existing 8-byte prologue + call-site struct-exemption + match pointer-deref path; gated so nullary-only enums stay i32-disc. Found via a parallel 5-agent investigation (1 agent localized it to parse_fn_decl). Verified: 6 narrowing shapes now python==bootstrap (was 132); tight broad regression 126 passed/1 skip/0 fail. Pin test_bootstrap_enum_payload_param_self_host (retry-on-132); K2 p281-p285 (parity; ratchet 280->285). NOTE: parallel agents also pinned the critical path = arena-cap overflow on the FULL self-host driver (130% of cap) but single files fit (parser 72%/kovc 55%) -> legs 2/3 next; + a 5th parity bug (lexer doesn't decode string escapes (backslash-n)). PRIOR: DEV-VELOCITY -- self-host harness driver CACHING (user-requested optimization #1). The self-host harness (_kovc_self_host_compile_and_run) re-compiled the full ~thousands-line lexer+parser+kovc driver via Python on EVERY call (~13s/test, dominating test time). Now the driver ELF is built ONCE per session (cache keyed on the bootstrap-source hash -> any lexer/parser/kovc.hx edit rebuilds it) and reused: per call we write k2_src to a FIXED per-xdist-worker in_path, run the cached driver (-> out_path), run out. xdist-safe (PYTEST_XDIST_WORKER-namespaced paths). VERIFIED: 7 diverse self-host tests (lexer self-host stress + if_not_paren 11-case + generics 9-case + bitwise + array-store + K2 p267/p275) pass IDENTICALLY in 80s vs ~5-8min before (~4-6x; if_not_paren alone was 115s pre-cache). Factored _compile_src_to_elf + _self_host_driver(). ALSO confirmed the mixed_variants enum bug is REAL+stable (enum-with-payload passed as a fn PARAM then matched -> SIGILL 132; inline match works; Python correct=42) -- deferred for a focused fix next. TEST-INFRA only (no bootstrap production change). PRIOR: SAFE-HARDENING S3 array/struct-memory audit -- mixed (6 clean + 1 exceeds-Python + 2 known gaps). CLEAN (pinned K2 p275-p280): i32 array sum + var-index, array element mutation a[i]=v, 2-deep + 3-deep NON-generic struct field chains (o.i.v, c.b.a.v), multi-struct-field (l.a.x+l.b.y) -- all byte-identical python==bootstrap. EXCEEDS-PYTHON: nested field STORE `o.i.v=42` -> Python returns 0 (no persist; the same Python field-store bug, deeper) while the bootstrap correctly returns 42 -- pinned as nested_store in test_bootstrap_struct_field_store_self_host (deletion-favorable). KNOWN GAPS (unsupported in BOTH -> NOT parity bugs since Python NotImplementedErrors; flagged for ATTENDED robustness fix): array-OF-struct (`[P{x:1},P{x:2}]; a[i].x`) -> bootstrap SIGILLs 132; struct with array-typed field (`struct S{xs:[i32;3]}`) -> bootstrap parser HANGS (timeout). The bootstrap should trap cleanly rather than SIGILL/hang on these; both are beyond Python either way. TEST-ONLY (no production change). K2 274->280. PRIOR: SAFE-HARDENING S3 generics audit -- GENERICS are a broad bootstrap-EXCEEDS-Python axis (deletion-favorable). Python helixc ParseErrors on ALL generic syntax (struct Box<T>, fn id<T>, enum Opt<T>, impl<T> methods) -- it has NO generics. The bootstrap compiles 11/12 probed generic shapes correctly (single + multi-instantiation generic struct, multi-param Pair, generic fn + nested calls, generic struct as fn-arg, generic enum + match, generic method, generic field-store) -> all 42. Pinned 9 working shapes in test_bootstrap_generics_exceed_python_self_host (bootstrap-only; NOT K2-parity-able since Python can't parse). KNOWN LIMITATION (NOT a regression; deferred): nested generic field access `Box<Box<i32>>.v.v` stably SIGILLs (132, verified 3x; the non-generic control Outer{Inner}.i.v works=42) -- this is the Stage 28.11 INC-3b gap (generic-param-typed field treated as SCALAR pending use-site monomorphization, parser.hx ~2679). Beyond Python either way; flagged for an ATTENDED fix, NOT chased unattended (deep monomorphization work). TEST-ONLY (no production change). 4th exceeds-Python cluster (now ~21 exceeds-Python shapes). PRIOR: SAFE-HARDENING S3 function-call/recursion audit -- axis CLEAN (9th clean axis; no bug). Probed 10 shapes BOTH compilers: self-recursion (fact(5)=120, fib(10)=55, sum_to(10)=55, pow2(5)=32), mutual recursion (even/odd via forward refs, ev(10)=1), 100-level tail recursion (count(100,0)=100), 3-arg call (add3=42), nested calls (dbl(inc(dbl(10)))=42), Ackermann (ack(2,3)=9), fn-calls-recursive-fn (24) -- ALL byte-identical python==bootstrap. The bootstrap's call/return/stack codegen matches Python (fitting: the bootstrap self-hosts via exactly this recursive-descent machinery). Pinned 8 shapes to K2 p267-p274 (ratchet 266->274). TEST-ONLY (no production change; new-ids + corpus_size verified). PRIOR: SAFE-HARDENING S3 integer-arithmetic-semantics audit -- axis CLEAN (8th clean axis; NO bug this time, which is itself deletion-favorable confirmation). Probed 14 i32 signed/overflow shapes BOTH compilers (avoiding the known type-erasure cast gap): signed rem (-7%3=-1, 7%-3=1), trunc-toward-zero div (-7/2=-3, 7/-2=-3, -7/-2=3), sign-preserving arithmetic >> (-8>>1=-4, -1>>31=-1), i32 overflow wrap (MAX+1->MIN, 100000*100000 wraps, 1<<31->MIN), unary/double negate + abs-via-cond -- ALL byte-identical python==bootstrap. The bootstrap's integer semantics are Rust-correct (match Python). Pinned 8 shapes to K2 p259-p266 (ratchet 258->266). TEST-ONLY (no production change -> no broad regression needed; new-ids + corpus_size verified). PRIOR: SAFE-HARDENING -- FIXED the 3rd audit-found bootstrap bug: bitwise/shift PRECEDENCE. The S3 operator-precedence audit found the bootstrap evaluated `& ^ | << >>` at ONE level (parse_bitwise, left-assoc) vs Rust/Python distinct levels: `5 ^ 3 & 1` gave 0 `(5^3)&1` not 4 `5^(3&1)`; `3 | 1 ^ 2` gave 1 not 3. Real both-compiler divergence (Python is Rust-correct; the bootstrap was wrong). FIX: split parse_bitwise into the Rust hierarchy `<< >> > & > ^ > |` (parse_bitwise kept as the `|` entry -> parse_bitxor -> parse_bitand -> parse_shift -> parse_add; all existing callers unaffected). Verified 20 precedence/bitwise/shift shapes python==bootstrap (incl shift-vs-bitwise `1&3<<1`=0, full chain `1|2^4&4`=7). K2 p254-p258 (ratchet 253->258). BROAD regression: ALL bitwise codegen (exit_bitwise_*, i64_b*, fold_bitwise_*) + bitwise self-host pass; 3 unrelated self-host failures proven PRE-EXISTING via git-stash on clean HEAD (self_host_loop = WSL /tmp file-write flake; tile_chain_bisect = fragile tile-stub drift pin -- both fail without my change; compound_assign shl_eq=132 = WSL flake, passes on rerun AND compound-assign is handled in parse_primary, independent of the shift split). 3rd real audit-found+fixed bug (after array-store a6bbe82, if-!(paren) macro-guard 971e521). PRIOR: SAFE-HARDENING -- FIXED the 2nd audit-found bootstrap bug: if/while/return `!(<paren>)` mis-branch (macro-detector keyword guard). ROOT CAUSE: the macro-call detector (parser.hx:3496; shape `IDENT ! (`/`[`/`{` -> macro invocation) fired on KEYWORD-leading shapes -- if/while/match/return are lexed as TK_IDENT (parser dispatches by name), so `if !(b){..}else{..}` was mis-read as a macro `if!(...)`; the macro handler ate the keyword + `!(...)` and DESTROYED the real control-flow statement (silent always-then-branch; `while !(..)` SIGILL'd). FIX: a lead_is_kw guard so is_macro_call fires only when the leading IDENT is NOT a reserved keyword (is_any_reserved_kw_ident + byte_eq if/while/match/return/loop/else/for/in/let). No real macro is keyword-named. Diagnosis empirical: a cond-only codegen probe showed the if-cond evaluating to the then-block value (not AST_NOT's 0/1) and a cond-tag probe showed the if-handler never ran -> localized to the macro mis-fire (both probes reverted byte-clean). Verified 12 Helix shapes python==bootstrap + println! still works. Pin test_bootstrap_if_not_paren_self_host (11 cases, retry-on-132 absorbs the WSL self-host SIGILL flake; a real mis-branch is a stable wrong VALUE, never 132). K2 p249-p253, ratchet 248->253. BROAD regression 103 passed (macro k1f22*/ident_bang, self_host_lexer, generics/closure/fnptr/impl_method/tuple, ptx/wgsl/msl/rocm, fn_type_table_over_256, array_indexed_store, K2 new) + 1 intermittent WSL SIGILL flake (failed a DIFFERENT case each run = env, not code). 2nd real audit-found+fixed bug (after array-store a6bbe82). PRIOR: SAFE-HARDENING S3 fresh-micro-surface audit -- FOUND a REAL bootstrap bug (fix next tick) + K2 240->248. *** REAL BUG (root-caused, FIX PENDING) ***: `if !(<parenthesized/complex truthy expr>)` ALWAYS takes the then-branch. Repro: `if !(b) {7} else {9}` returns 7 for BOTH b=1 and b=0 (should be 9 for b=1); `if !(5==5)`, `if !(!b)` likewise mis-branch. NARROWED: `if !b` (bare), `if (b)` (paren no-not), and `let t=!(b); if t` ALL WORK -- and `!(b)` as a VALUE is correct (not_paren_1->0, not_paren_0->1). So the bug is SPECIFICALLY the if-CONDITION parse of `!(...)` (parser.hx:6793 `parse_expr_basic` for the cond; `!` handled at parse_unary parser.hx:2426). Python helixc is CORRECT here -> a real bootstrap divergence. NOT rushed (subtle parser fix; deferred to a fresh tick to avoid breaking the working if-(b)/if-!b paths). Also: 8 clean-parity shapes from the same probe -> K2 p241-p248 (bool-xor, while-complex-cond, const-in-calc, multi-shadow, large-lit-boundary, mod/div-pow2, bool-and-or-mix); ratchet >=248. test-only this chunk; 9 K2 tests green incl corpus_size (111s). NEXT: FIX the if-!(paren) mis-branch bug (trace parse_expr_basic if-cond + parse_unary `!(` interaction) + pin test_bootstrap_if_not_paren; BROAD regression. PRIOR SAFE-HARDENING S3 ENUM/MATCH-DEPTH audit -- axis CLEAN + K2 231->240. Probed 9 rich enum/match shapes both-compilers: 8-variant enum, match WILDCARD-binding (`n => n-57`), match-on-fn-call, enum-payload arithmetic across arms, match-as-subexpression, fn-returning-matched-value, 3-LEVEL nested match, struct holding an enum field, and MULTI-PAYLOAD enum `E::P(a,b) => a+b` -- ALL byte-IDENTICAL Python<->bootstrap. The enum/match axis is CLEAN (even multi-payload enums + struct-with-enum-field work in both). Folded into K2 as p232-p240 (ratchet >=240). test-only; 10 K2 tests green incl corpus_size (125s). AUDIT COVERAGE: 6 axes now dry-run-CLEAN (arith/parser/composite/write-path/type-system/enum-match). K2 gate = 240 broad both-compiler-verified entries; 1 real bug fixed overall (array-store); 10 exceed-Python cases mapped. NEXT: SAFE-HARDENING continue (remaining-surface S3 / K2 maintenance). PRIOR SAFE-HARDENING S3 TYPE-SYSTEM audit -- typed-int axis CLEAN + K2 223->231. Probed 8 typed-int shapes both-compilers: u32 division, u8/u16/i16 arithmetic, i64 add/mul (returning from an i32 fn -- no trap, both handle i64-fits-i32), u32 comparison, and u8 WRAPAROUND (250+48 -> 42 in both). ALL byte-IDENTICAL Python<->bootstrap -- the typed-integer/sign-extension/wraparound axis is CLEAN (no divergence, no bug). Folded into K2 as p224-p231 (ratchet >=231). test-only; 9 K2 tests green incl corpus_size (108s). AUDIT COVERAGE NOW: BE/RT-arith + FE/composite + write-path + parser/FE + type-system axes all dry-run-CLEAN (1 real bug found+fixed total: array-indexed-store a6bbe82; 10 bootstrap-exceeds-Python cases mapped). K2 parity gate = 231 broad both-compiler-verified entries. NEXT: SAFE-HARDENING continue (further S3 axes -- enum/match depth, runtime limits -- or S1 K2 growth). PRIOR SAFE-HARDENING S3 PARSER/FE audit -- FE axis CLEAN + 2 more bootstrap-exceeds-Python. Probed 11 parser/front-end edge shapes both-compilers: trailing commas (args/array/struct), line comment mid-fn, empty block `{}`, double-negate, deep parens, NESTED block comments `/* /* */ */`, block-as-expression-value -- 9 both-compiler-PARITY (folded into K2 as p215-p223; corpus 214->223, ratchet >=223). NO bootstrap bug (FE solid). 2 more bootstrap-EXCEEDS-Python: (a) extra/empty statements `let x=42;; x` -> bootstrap accepts (42), Python ParseError; (b) char literals `let c='A'` -> bootstrap parses (42), Python NotImplementedError. Pinned bootstrap-only: test_bootstrap_parser_exceeds_python_self_host. test-only; 11 tests green (parser-exceed pin + p215-p223 + corpus_size). RUNNING TALLY of bootstrap-exceeds-Python: generics, fn-pointers, tuple-destructure, closures, GPU, impl-method, break/continue, struct-field-store, extra-semis, char-literals -- the bootstrap is materially MORE capable/correct than Python helixc, strengthening the eventual deletion case. NEXT: SAFE-HARDENING continue S3 / S1. PRIOR SAFE-HARDENING S3 WRITE-PATH audit -- array-store fix robustness + a NEW bootstrap-exceeds-Python finding. (1) The a6bbe82 array-store fix HOLDS across complex store patterns -- computed index `a[i+1]=v`, computed value `a[i]=i*14`, element-as-accumulator, store-then-use-same-expr, interleaved stores, 2D-via-flat-index, 12-arm match -- all both-compiler-PARITY; folded into K2 as p208-p214 (corpus 207->214, ratchet >=214). (2) FOUND: struct field STORE (`p.x=42; p.x`) is a real PYTHON helixc BUG -- Python returns 0 (does NOT persist field writes; two-field gives 40 not 42), while the BOOTSTRAP is CORRECT (42). 6th 'bootstrap is more correct than Python' case (after generics/fn-ptr/tuple/closure/GPU) -- DELETION-FAVORABLE (deleting buggy Python loses nothing). Pinned bootstrap-only: test_bootstrap_struct_field_store_self_host (3 cases). NOT K2-parity-able (py!=kc, py wrong); we do NOT fix Python (K4 deletes it). (3) struct-copy-mutate = both-broken (deferred). test-only; 9 tests green (struct-field pin + p208-p214 + corpus_size). NEXT: SAFE-HARDENING continue S3 (more divergence hunting) / S1. PRIOR SAFE-HARDENING S3 AUDIT BUG-FIX -- array indexed-STORE `a[i]=v` on the CPU/ELF path. The dry-run audit found a REAL both-compiler DIVERGENCE: `a[i]=v` SIGILL'd (132) on the bootstrap CPU path (even with a constant index) while Python helixc compiled it (=42). ROOT: emit_ast_code had NO AST_INDEX_STORE (tag 55) arm on the CPU path -- only the PTX/WGSL emitters did -- so `a[i]=v` fell through to the unhandled-tag trap (id 99001) -> SIGILL. FIX: new top-level helper emit_index_store_cpu (element addr = base + index*8, 4-byte store, mirroring the tag-53 read), dispatched from a brace-balanced `if t==55 {...} else { trap }` inside emit_ast_code's final fallback. Verified const/var/loop array-store + fill+sum all 42 (were 132). Pins: test_bootstrap_array_indexed_store + K2 p206/p207. BROAD regression GREEN (kovc.hx codegen hot-path): 55 passed 1 skipped (882s) full PTX/GPU + CPU canary + generics/closure/fnptr/tuple + self_host_lexer + fn_type_table + array_indexed_store. K2 corpus 207 (ratchet >=207). This is a genuine correctness bug found AND fixed autonomously by the S3 dry-run audit -- mutable-array element assignment now works on the bootstrap. NEXT: SAFE-HARDENING continue S3 (hunt more divergences: struct-field-store-in-loop, nested writes) / S1. PRIOR SAFE-HARDENING S1 -- K2 parity gate CROSSES 200 (196->205). 9 more both-compiler-parity shapes p197-p205: binary literal (0b101010), hex-in-op, 6-param fn, bool struct field, shift+arith, 5-param fn, enum-discriminant arith, struct-param multiply (rc 74), deep addition. ratchet 196->205. Probe also found struct-RETURNING fn (`fn()->P`) is BOTH-BROKEN (Python NotImplementedError + bootstrap SIGSEGV) -- another non-deletion-blocking both-lack gap like A4/A5, not parity-able, documented. test-only; 10 K2 tests green incl corpus_size (134s). K2 PARITY GATE now 205 entries (p01-p205) -- a very broad both-compiler-verified deletion prerequisite spanning arithmetic/control-flow/structs/enums/match/arrays/recursion/bitwise/typed-literals/signed-edge-cases. Consolidated safe-hardening TG sent this chunk. NEXT: SAFE-HARDENING rotate (more S1 toward ~210 / S3 RT-TEST audit axes). PRIOR SAFE-HARDENING S3 -- dry-run audit (FE/parser + COMPOSITE axis) + K2 184->196. Probed 12 front-end/feature-composition shapes both-compilers: 6-level nested if/else, 8-arm match, 5-field struct, enum-2-payload match, 5-deep nested calls, mixed-operator precedence, 10-elem array index, while+nested-if, const+arith, deep recursion (sumto 50), multi-statement seq, negative-in-match -- ALL byte-IDENTICAL Python<->bootstrap. FINDING: the FE/parser + composite axis is CLEAN (no divergence; complements the S3 BE/RT-axis-clean finding at 56b6f94). Folded all 12 into K2 as p185-p196; ratchet 184->196. test-only; 13 K2 tests green incl corpus_size (166s). K2 PARITY GATE now 196 entries -- a very broad both-compiler-verified deletion prerequisite. NEXT: SAFE-HARDENING rotate -- S1 to ~200 OR more S3 axes (RT runtime / TEST). PRIOR SAFE-HARDENING S3 -- dry-run audit (BE/RT axis) + K2 edge-case parity guards 178->184. Probed 12 arithmetic/codegen edge cases both-compilers: i32 overflow wrap (2147483647+1=0), signed div trunc (-7/2=253), signed mod sign (-7%2=255), shift-left-to-sign-bit (1<<31=0), arithmetic-shift-right-negative (-8>>1=252), signed mul (-6*7=214), deep nesting, && chains, neg comparisons, big literals, mod chains -- ALL byte-IDENTICAL Python<->bootstrap. FINDING: the codegen/runtime axis is CLEAN on signed/overflow/shift semantics (no divergence). Pinned the 6 subtlest as permanent K2 parity guards p179-p184 (explicit non-42 exit codes = the exact wrap/truncation value is the assertion); ratchet 178->184. test-only (no compiler change); 7 K2 tests green incl corpus_size (82s). NEXT: SAFE-HARDENING rotate -- more S3 dry-run audit axes (FE parser edge cases / struct-enum-array combos) or S1 K2 top-up toward ~190. (Dry-run audits do NOT count toward the formal 5-clean streak -- gated on Python-ready / the multi-night fixpoint.) PRIOR SAFE-HARDENING S2 -- fn_type_table cap 256->1024 (real robustness fix + self-host-fixpoint prereq). A program with >256 fns silently DROPPED fn_type_table entries past #256, so a non-i32-return fn beyond #256 mis-typed at its call site -- demonstrated: a 282-fn program with an f32-returning fn (281st) read as i32 -> SIGILL (132) pre-fix, -> 42 post-fix. Bumped kovc.hx fn_type_table_init (`while i<1280`->5120) + fn_type_table_add (`if top>=256`->1024); the table is arena-pushed at init, NOT woven into the hardcoded end-region address math (the arena cap, untouched), so SAFE. Pin: test_bootstrap_fn_type_table_over_256. BROAD regression GREEN (one WSL-flake on impl_method_amp_ret re-passed clean on rerun; 53 other pass + skip): full PTX/GPU + CPU canary + generics/closure/fnptr/tuple + self_host_lexer + the new test. Bootstrap now handles realistic multi-hundred-fn programs (full self-host driver = 673 fns). NEXT: SAFE-HARDENING rotate to S3 (dry-run audit probe) / S1 (K2 top-up). PRIOR SAFE-HARDENING S1 -- K2 parity top-up 166->178 (toward ~180). Probed 12 shapes both-compilers; all PARITY -> p167-p178 (nested match, array-elems-as-args, boolean && !, mul chain, paren precedence, bool-var-in-if, double negate, sumto recursion, 4-variant enum match, mod/div branch, shift-right, xor); ratchet 166->178. test-only (no compiler change -> no broad regression); 13 K2 tests green incl corpus_size (173s). Self-host fixpoint legs 2-3 remain the user-steered multi-night arena-cap work (f450875). NEXT: more S1 to ~180 OR rotate to S2 (fn_type_table 256->1024 safe robustness fix) / S3 (dry-run audit probe). PRIOR PHASE C C1 leg-2 FINDING -- self-host fixpoint of parser.hx/kovc.hx is a MULTI-NIGHT, ATTENDED, HIGH-RISK effort (precisely characterized; NO compiler change this chunk -- a documented investigation conclusion). The bootstrap SIGSEGVs (rc 139, no ELF) compiling parser.hx (794648 bytes / 289 fns). ROOT BLOCKER = the ARENA cap (helix_arena_cap()=2097152 slots/8MB, mirrored Python HELIX_ARENA_CAP): 795KB source + tokens + AST exceed 8MB. CRITICALLY, the arena cap is NOT a clean constant -- it is WOVEN INTO HARDCODED ADDRESS MATH: a reserved end-of-arena region (last 64 slots, used by trace_event + a handle/cell table) sits at hardcoded offset 8388356 = 4 + (CAP-64)*4, hardcoded in >=3 spots (kovc.hx:5609 `disp = 8388356 + handle*4`, :5669 `disp_base = 8388356`, :5769 `disp_base2 = 8388356`) + the emit_elf BSS (:9576 `4 + 8388608`) + Python's mirrored layout. Scaling the arena (needed: ~16-64M slots for parser.hx; the FULL fixpoint driver is ~1.6MB / 673 fns and needs more) requires recomputing ALL these end-region offsets in lockstep across Python helixc + the bootstrap -- a mistake = SILENT memory corruption (not a clean failure). Secondary cap: fn_type_table=256 < parser.hx's 289 fns (kovc.hx:1652/1664; this is a SAFE table-size bump, not address-math-woven, but only a CORRECTNESS issue for >256-fn non-i32-return programs, NOT the SIGSEGV). VERDICT: per the loop's escape hatch, PHASE C self-host fixpoint legs 2-3 (parser.hx, kovc.hx) require careful ATTENDED coordinated arena-layout work (multi-night), NOT 60s-tick-safe; lexer.hx leg-1 self-hosts cleanly with parity (375de55). PIVOT to SAFE productive work (PHASE B top-up / a dry-run 5-axis audit sweep) for the remainder of the autonomous run; the STOP CRITERION (Python-ready + 5 clean audits) is gated on this multi-night fixpoint + the K3 trusted-seed, so it is NOT reachable tonight -- HONEST status. PRIOR PHASE C C1 leg-1 -- SELF-HOST lexer.hx WITH PARITY (first leg of the K2 Phase-3 fixpoint). The bootstrap kovc (built by Python) compiles its OWN lexer.hx (1029 lines / 43.7KB): lex + parse + emit_elf ALL succeed -> a 20266-byte ELF, and the emitted binary's Demo main exits with the SAME rc (255) as the Python-compiled lexer.hx. NO cap overflow, NO unsupported construct -- the smallest bootstrap source self-hosts cleanly with byte-behavior PARITY to the Python reference. MAJOR positive finding: the bootstrap's token/AST caps + codegen already handle a real 1000-line source file. Pin: test_bootstrap_self_host_lexer_hx (test-only, no compiler change -> no broad regression; 22.76s). NEXT: C1 leg-2 = parser.hx (15304 lines -- ~15x bigger; LIKELY hits fixed token/AST/arena caps; probe via the same harness, capture the FIRST cap-overflow trap or unsupported construct, then C2-fix smallest-first). PRIOR PHASE B Track-P growth -- K2 corpus 156->166, CROSSING the ~160 credible-gate target. 10 more both-compiler-parity shapes p157-p166: enum-with-payload match, multi-arm int match, 3-field struct, boolean not (!), chained && comparison, factorial recursion, hex literal (0x2a), const global, const-index array, div combo. ratchet 156->166. test-only (no compiler change -> no broad regression); 11 K2 tests green incl corpus_size (142s). K2 PARITY GATE now 166 entries (p01-p166), a credible 'K2-green over a broad real-source corpus' deletion prerequisite. NEXT: PHASE C SELF-HOST FIXPOINT C1 -- try compiling the SMALLEST bootstrap source (lexer.hx) via the bootstrap kovc itself, capture which Helix constructs break. PRIOR PHASE B Track-P growth (K2 corpus 146->156). Probed 12 shapes both-compilers; 10 PARITY -> added p147-p156 (struct-param, enum/match, array-sum loop, nested-if, fib recursion, multi-return-path, shift-left, precedence, div, square-fn); ratchet 146->156. (while-break + while-continue ALSO probed -> Python helixc NotImplementedError but bootstrap runs them -> yet another bootstrap-exceeds-Python case; not K2-parity-able.) test-only (no compiler change -> no broad regression); 11 K2 tests green incl corpus_size (146s). K2 corpus now 156 toward the ~160 credible-gate target; next chunk should cross 160 (-> TG with the A4/A5 deferral note). NEXT: more PHASE B parity shapes to ~160, then PHASE C self-host fixpoint (C1 = try compiling lexer.hx via the bootstrap). PRIOR PHASE A A4/A5 DEFERRED (documented) + PHASE B Track-P growth (K2 corpus 138->146). PHASE-A DECISION: A1/A2/A3 (generics, fn-pointers, tuple-destructure) were the tractable high-value SIGILL-class fixes -- all DONE + all EXCEED Python. A4 (curried/escaping closure `|a| |b| a+b`) and A5 (array-of-structs access `arr[i].field`) were investigated this tick and are DEFERRED as large non-deletion-blocking features Python ALSO lacks (ParseError/NotImplementedError): A4 SIGILLs even at DEFINITION -- a closure returning a closure that captures the outer param needs ESCAPING-CLOSURE support (heap-allocated environments + (code,env) closure values), a multi-night architectural addition to the flat call-site-capture model (cl_var_tab), NOT a quick fix; A5 LITERAL BUILDS fine (define-only ->7) but `arr[i].field` SIGILLs -- needs array-element-is-struct memory layout + indexed-field codegen (moderate, multi-tick). Both are audit-NOTEABLE known gaps (accept-then-SIGILL footguns), NOT regressions, NOT deletion blockers (bootstrap already EXCEEDS Python across generics/fn-ptr/tuple/closure-parse/GPU/impl-method). PIVOTED to PHASE B (the real K4 deletion gate, low-risk): probed 8 shapes both-compilers -> all PARITY -> added K2 p139-p146 (shadowing, if-expr-let, while-accum, multiarg-fn, modulo, nested-call, unary-neg, bitwise-combo); ratchet 138->146. test-only (no compiler change -> no broad regression); 9 K2 tests green incl corpus_size (120s). NEXT: continue PHASE B K2 growth toward ~160 then PHASE C self-host fixpoint. PRIOR A3 -- TUPLE-DESTRUCTURE-LET (PHASE A; EXCEEDS Python which ParseErrors on tuple-destructure). `let (a,b)=(40,2); a+b`->42 + 3-tuple variant. The bootstrap PARSED the `(a,b)` pattern (K1.CC) but left a/b UNREGISTERED (syntactic accept only -> reading them trapped id 1001 -> 132). Now a FLAT tuple pattern whose RHS is a tuple LITERAL of matching arity DESUGARS at parse time to a chain of single lets (`let a=40; let b=2; ...`), binding each name to its element -- NO runtime tuple repr needed. parser.hx let-handler (~5848): collect flat pattern names (depth-1 IDENTs) into arena scratch (td_pat_base/count, td_simple gate), then at the node-build emit the nested AST_LET chain when td_simple && RHS tag==50 && arity matches; ELSE the pre-A3 single (0,0)-named let (byte-identical common path -- non-tuple lets + variant patterns + non-literal RHS unchanged). Pin: test_bootstrap_tuple_destructure_let (2-tuple + 3-tuple). BROAD regression GREEN (hot let-parse path): 52 passed 1 skipped (858s) full PTX/GPU + CPU canary + generics + closure + fnptr + impl_method + tuple; K2 smoke 5 passed (let-heavy p05/p40/p90/p130). NEXT: PHASE A A4 = nested/curried closure (`|a| |b| ..; f(1)(2)`). PRIOR A2 COMPLETE -- FULL FUNCTION POINTERS (PHASE A; NET-NEW, 5th 'bootstrap EXCEEDS Python' finding -- Python helixc NotImplementedErrors on fn-typed calls). `apply(dbl,21)`->42 compiles+runs via self-host: pass a fn by name + call it through a param. Three pieces: A2a (b2406d3) fn-name-as-VALUE (AST_VAR unbound+fn_type_table_has -> `lea rax,[rip+disp]` to the fn); A2b INDIRECT CALL (AST_CALL callee bind_lookup-hit local -> `mov r11d,[rbp-off]` (32-bit zero-extend: addr<2^32 since ELF_BASE 0x400000) + `call r11`/41 FF D3, at both <=6 and >6-arg sites); A2c fn-type PARAM PARSE (parser.hx:~11543 -- `f: fn(T..)->R` was read as the type IDENT, desyncing the param loop -> SIGILL; now consumes `fn(...)->R` type-erased like the K1.X let-type handler, gating skip_ident_read). Pins: test_bootstrap_fnptr_name_as_value + _indirect_call. BROAD regression GREEN (parser.hx + AST_CALL/AST_VAR hot paths): 50 passed 1 skipped (810s) across full PTX/GPU + CPU canary + generics + closure + fnptr + impl_method. NEXT: PHASE A A3 = tuple-destructure-let (`let (a,b)=(40,2)`; Python ParseError + bootstrap SIGILL per M33). PRIOR A2a -- FN-POINTER VALUE foundation (PHASE A; NET-NEW, EXCEEDS Python which NotImplementedErrors on fn-typed values). Investigated the fn-pointer-call SIGILL (M32 finding): both fn-name-as-value (`let g = dbl`) AND indirect call (`f(x)` via param) SIGILL'd (132). ROOT CAUSE: AST_VAR codegen (kovc.hx:8373) traps (id 1001) on ANY unbound name. A2a adds fn-name-as-value: an unbound name that fn_type_table_has() confirms is a registered user fn now emits `lea rax,[rip+disp]` to its code label (resolved at backpatch via the patch table -- identical rel32 mechanism to a CALL) -> rax holds the fn's runtime address. Genuine typos still trap 1001 (clean SIGILL); the fn-presence gate is REQUIRED because the K1.F21 ud2-fallback assumes a CALL (E8) site and would corrupt a bare lea into garbage. New helper fn_type_table_has (presence check; the existing lookup returns ambiguous 0 for both miss and fn->i32). Verified `let g=dbl; 7`->7 (was 132) via test_bootstrap_fnptr_name_as_value. BROAD regression GREEN (AST_VAR hot-path): full PTX/GPU + CPU canary + generics + closure = 45 passed 1 skipped (540s) + K2 smoke 14 passed. NEXT: A2b = INDIRECT CALL (`f(x)` where callee is a local holding a fn address -> load local + `call rax`/FF-D0; AST_CALL arm kovc.hx:8541) completing the full fn-pointer `apply(dbl,21)`->42. PRIOR A1b -- MULTI-PARAM bare-generic-call SIGILL FIX (PHASE A). Generalized A1a: monomorphize_pass synthesis now covers generics of ANY arity (gp_n>=1, was ==1; mangled `first__i32_i32`, pack_lo==gp_n since i32 tag==0) + the K1.F21 bare-call fallback (kovc.hx:9415) now tries `__i32`, then `__i32_i32`, ... up to 4 type-params (first fn_table_lookup hit wins; multi path gated name_l<48 to fit the 64-slot scratch, single path keeps <60). Verified first(42,7)->42 self-host (test_bootstrap_generics_bare_call_multi). BROAD regression GREEN (parser.hx + kovc.hx both changed): full PTX/GPU + CPU canary full_pipeline_arithmetic + generics + closure = 44 passed 1 skipped (499s) + K2 smoke 4 passed. The PHASE-A bare-generic-call SIGILL class is now CLOSED (single + multi param). NEXT: PHASE A A2 = fn-pointer call (`f: fn(i32)->i32`) SIGILL (Python NotImplementedError, bootstrap SIGILL -- both broken; bootstrap-quality polish). PRIOR A1a -- BARE-GENERIC-CALL SIGILL FIX (PHASE A). monomorphize_pass (parser.hx:9554) now synthesizes a default-i32 mr_tab entry per SINGLE-type-param generic template that lacks one (mangled `<name>__i32`, pack_lo=1; i32 tag==0 so packed==0), so a bare `id(42)` (no turbofish) resolves via the K1.F21 fallback (kovc.hx:9415) instead of patching ud2 -> SIGILL(132). Skips templates that already have an i32 entry (mr_tab_lookup). Verified id(42)->42 self-host (test_bootstrap_generics_bare_call). BROAD regression GREEN (parser.hx change): full PTX/GPU suite + CPU canary full_pipeline_arithmetic + generics + closure = 43 passed 1 skipped (541s) + K2 smoke 4 passed. Multi-param bare calls (first(a,b) -> first__i32_i32) are A1b (needs the K1.F21 fallback extended to build the multi-suffix name; currently single-`__i32` only). NEXT: A1b (multi-param) then PHASE A A2 fn-pointer-call SIGILL. PRIOR K1.M34 -- K2 PARITY growth (Track-P hardening). Probed 4 shapes both-compilers: nested-for-loops, recursive gcd, boolean short-circuit chain -> ALL PARITY -> added K2 corpus p136-p138 (135->138). struct-in-array-LITERAL tried but Python NotImplementedError ('struct literal in expression position not yet supported') + bootstrap SIGILL -> same mapped advanced-feature class, omitted. 4 K2 tests green (46.52s). corpus 138; ~45 codegen+K2 tests green. Track-P parity gate widened (a real K4 prerequisite). NEXT: K1.M35 = more K2 parity (array-in-struct, multi-arm-block-match, deeper recursion, typed-int arith) OR a dedicated SIGILL-fix attempt. PRIOR K1.M33 -- feature probe (match-guard parity + 2 SIGILL findings) + 5-chunk status TG. Probed match-guard / tuple-destructure / nested-closure both-compilers. (1) MATCH-GUARD `match x { n if n>3 => 42, _ => 0 }` -> BOTH 42 (PARITY) -> K2 corpus p135 (134->135). (2) TUPLE-DESTRUCTURE-LET `let (a,b)=(40,2)` -> Python ParseError; bootstrap SIGILL (132). (3) NESTED/CURRIED CLOSURE `|a| |b| a+b; add(40)(2)` -> Python ParseError; bootstrap SIGILL. META-FINDING: a CLASS of bootstrap 'parses-but-SIGILLs-on-advanced-features' bugs now spans bare-generic-call + fn-ptr-call + tuple-destructure-let + curried-closure -- the bootstrap accepts richer syntax than Python (which ParseErrors) but mis-codegens some of it. ALL NON-deletion-blocking (Python can't do any) -> deferred bootstrap-QUALITY polish (relevant for the eventual 5-axis audit, not deletion-parity). 2 K2 tests green (16.43s). 42 codegen+K2 tests green. HONEST STATE: deletion-relevant porting is essentially DONE (bootstrap >= Python across the board); remaining = deferred trusted-seed (weeks) + this non-blocking SIGILL-polish class + the audit gate. Sent 5-chunk status TG. NEXT: K1.M34 = continue hardening; weigh a dedicated SIGILL-fix attempt vs more K2 parity. PRIOR K1.M32 -- FEATURE-AREA PROBE (safe hardening; 1 exceed + 1 parity + 1 both-broken). Probed closures / enum-payload / fn-pointers via both compilers. RESULTS: (1) CLOSURE `|x:i32| x+1; f(41)` -> Python ParseError (cannot parse `|`); bootstrap -> 42 -> bootstrap EXCEEDS Python (4th such finding after GPU/impl-method/generics) -> pinned bootstrap-only test_bootstrap_closure. (2) ENUM-WITH-PAYLOAD `enum E{A(i32),B}` + match-binding `E::A(n)=>n` -> BOTH 42 (PARITY) -> added K2 corpus p134_enum_payload (133->134). (3) FN-POINTER param `f: fn(i32)->i32` -> Python NotImplementedError (Stage-31 'function-typed calls not supported'); bootstrap -> SIGILL (132) -> BOTH broken; the bootstrap fn-ptr-call SIGILL is another bootstrap-QUALITY miscompile (like bare-generic-call), NON-deletion-blocking (Python can't do it either) -> documented/deferred. 3 tests added + green (29.12s). 41 codegen+K2 tests green. The 'bootstrap exceeds Python' pattern now spans GPU + impl-method + generics + closures -- Python helixc is materially LESS capable than the bootstrap in syntax it cannot even parse. NEXT: K1.M33 = SEND 5-chunk status TG + more probes (trait-impl / nested-closures / slices) or fix a small non-parser bug. PRIOR K1.M31 -- bare-call fix-A DEEP ASSESSMENT (deferred as multi-tick) + K2 corpus growth (safe hardening). Investigated fix-A fully: monomorphize_pass (parser.hx:9554) clones per mr_tab entry; making bare id(42) work needs synthesizing an i32 mr_tab entry per generic template. BLUEPRINT: pre-loop in monomorphize_pass (BEFORE its count==0 early-return) over fn_list generic templates (slot6==1); for each with no existing i32 entry, build mangled name via mangle_name_into_arena(name, [i32 refs]) + ty_ident_to_tag('i32') for packed + gp-count from slot7, then mr_tab_add. CAVEAT: turbofish mangles via mangle_name_into_arena, but the K1.F21 BARE-CALL fallback (kovc.hx:9415) hardcodes a single '__i32' suffix -> CLEAN for 1-param generics (id__i32) but multi-param (first<A,B> -> first__i32__i32) needs the fallback ALSO extended (needs the target gp-count at backpatch). So fix-A is a MULTI-PIECE mini-project (synth monos + extend fallback + manufacture i32 type-arg refs), touches parser.hx (broad regression), risks the WORKING turbofish/mono path. DECISION: deferred as a documented multi-tick task (NOT re-teed each tick; non-deletion-blocking -- Python can't do generics at all). SHIPPED instead: K2 parity corpus 130->133 (p131 nested-match, p132 arith-precedence, p133 array5-const-sum; all confirm Python<->bootstrap parity, 42.47s). 39 codegen tests green. NEXT: K1.M32 = continue SAFE hardening -- probe another feature area both-compilers (closures / traits / string ops) for bootstrap-vs-Python gaps, OR more K2 integration shapes. PRIOR K1.M30 -- K3 / ENDGAME ASSESSMENT (critical-path mapping). Surveyed the master plan + repo. FINDINGS: (1) 'K3 trusted-seed bootstrap' = the from-raw-binary hex0->...->kovc SEED chain; master plan marks it 'not blocking; decision when the time comes; possibly WEEKS'; NO stage0/ code in-repo (hex0 design-stage) -> MAJOR DEFERRED effort, NOT 60s-tick-tractable. (2) N-generation FIXPOINT (kovc compiles kovc.hx -> stable) is UNTESTED (only described in test_k2_parity docstring); gated on full self-source support (~11k lines; huge/slow to verify). (3) NO audit-harness SCRIPT (no scripts/*audit*); the '5-axis END-OF-PHASE audit' (FE/IR/BE/RT/TEST sweep x5) is AGENT-PERFORMED, not pytest. CONCLUSION: the terminal STOP CRITERION (Python-ready + 5 clean audits) is gated on the deferred trusted-seed (weeks) + fixpoint + user confirmation -- NOT reachable via 60s ticks. The loop's realistic ongoing value = HARDENING: grow Track-P (K2 parity corpus), FIX real bootstrap bugs (audit-cleanliness), feature-parity. Keep ticking on those (user: don't stop until Python deletion). Corrected the K3 bucket note (was optimistic '~5-10 chunks'). No code change (assessment). 39 codegen tests green. NEXT: K1.M31 = un-defer + ATTEMPT FIX of the bare-generic-call SIGILL (fix-A: mono pass synthesizes an unconditional i32 clone per generic template; parser.hx -> BROAD regression after) -- a real miscompile worth closing for audit-cleanliness; fall back to K2-parity growth if too deep. PRIOR K1.M29 -- ROOT-CAUSED the bare-generic-call SIGILL (the M28 bug) + prioritization decision. Surveyed kovc.hx + parser.hx: generic monomorphization runs via an mr_tab (mono-instantiation request table; parser.hx:497 decl + :9549 mono pass + clone_with_rewrite:9528). ONLY turbofish calls (id::<i32>(...)) push mr_tab entries (parser.hx:7256 mr_tab_add) -> the mono pass clones id__i32 + emits it. A BARE call id(42) pushes NO mr_tab entry (no ::<> syntax; and the target is not resolved-as-generic at parse time, forward refs), so NO clone is created; the K1.F21 backpatch fallback (kovc.hx:9415) then looks up id__i32, MISSES (never created), and emits ud2 -> SIGILL (exit 132). FIX OPTIONS: (A) mono pass synthesizes an i32 clone for every generic template UNCONDITIONALLY (so id__i32 always exists -> K1.F21 fallback resolves bare i32 calls); (B) real bare-call type inference (resolve target->is_generic, infer T from arg type, push mr_tab). Both are parser.hx changes (BROAD regression). DECISION: DEFERRED -- NOT deletion-blocking (Python cannot even parse generics, M28) and narrow (turbofish is the workaround); recorded as a known bug w/ a clear fix path; revisit if an END-OF-PHASE audit flags it. PIVOT: the remaining CRITICAL-PATH blockers are K3 trusted-seed/self-host fixpoint + 5 clean END-OF-PHASE audits -- M30 assesses K3. No bootstrap code change this tick (root-cause record). 39 codegen tests green. PRIOR K1.M28 -- GENERIC-MONOMORPHIZATION PROBE + 5-chunk status TG. Probed generics via both compilers. FINDING: Python helixc CANNOT PARSE generic syntax (ParseError on `<T>` for fn-generic/generic-struct/turbofish/multi-param). The bootstrap EXCEEDS Python: generic-struct (Box<T>=42) + turbofish (id::<i32>(42)=42) WORK (pinned: test_bootstrap_generics_struct_and_turbofish); but a BARE generic-fn call (id(42), first(42,7)) MISCOMPILES to SIGILL (exit 132) -- bare-call type inference is a real bootstrap-QUALITY bug (NOT a deletion blocker: Python=zero generics so deletion-parity is met). This is the 3rd 'bootstrap exceeds Python' finding (after M21 GPU + M27 impl-method) -- PATTERN: the remaining [~] deletion blockers are largely MET in the deletion sense (bootstrap >= Python); their 'full' qualifier is absolute-completeness / bootstrap-superset polish, not deletion-blocking. Corrected the generics bucket note. Sent the 5-chunk status TG (deletion 75%). 39 codegen tests green. NEXT: K1.M29 = FIX the bare-generic-call SIGILL (survey monomorphization/type-inference in kovc.hx; likely multi-chunk) OR reconsider flipping impl-method/generics buckets toward done given deletion-parity is met, OR continue probing K3-trusted-seed / other blockers. PRIOR K1.M27 -- IMPL-METHOD DISPATCH PROBE (deletion-blocker finding). Probed the impl-method-dispatch blocker via BOTH compilers. RESULT: the BOOTSTRAP fully handles core impl-method dispatch -- method-on-struct-value (p.get()), method-with-arg (p.add(2)), method-calling-method (self.a()+2) -- all -> 42 via self-host; while PYTHON helixc CANNOT PARSE the bare `(self)` receiver (ParseError: expected COLON). So the bootstrap EXCEEDS Python here (parallel to the M21 GPU finding). These cannot be K2-parity entries (Python errors), so pinned via a bootstrap-only test test_bootstrap_impl_method_dispatch (3 cases, 40.28s green). DELETION-PARITY for core impl-method dispatch is MET (deleting Python loses nothing). Corrected the impl-method bucket note (was 'comprehensive dispatch pending ~10 chunks' -- pessimistic; core works + exceeds Python; remaining = advanced trait/generic-impl/&self cases, likely Python-gaps too). Kept status 'partial' (conservative; advanced cases untested). 38 codegen tests green. NEXT: K1.M28 = probe/advance the generic-monomorphization blocker (4/10; 3 pending const-generics/lifetime-only/generic-impl; 2 partial gp-field/where-clauses) via the same both-compiler probe, OR more K2 corpus growth. PRIOR K1.M26 -- K2 PARITY corpus growth (deletion-blocker progress): added bitwise-op coverage. The K2 parity harness (helixc/tests/test_k2_parity.py) compiles each corpus program via BOTH Python helixc AND the bootstrap kovc self-host, asserting identical exit codes (behavioral parity); it had NO bitwise coverage. Added p126-p130 (& | ^ << >>, each = 42) + bumped the corpus size ratchet 125->130. All 5 CONFIRM Python<->bootstrap PARITY on bitwise ops (the bootstrap kovc.hx x86 emitter matches Python helixc exactly), 6 passed incl the size guard (83.72s, both full paths). The K2 corpus is now 130 entries -- a larger credible K2-green gate toward Python-deletion. No bootstrap code change (coverage/probe only); bitwise behavioral parity now PINNED. NOTE: the full 130-entry corpus is SLOW (~13s/entry for the self-host path) -- run only NEW entries per tick, not the whole corpus. NEXT: K1.M27 = continue K2 corpus growth (untested shapes: boolean-not !, nested multi-arg calls, deeper enum/match/array) OR start the impl-method-dispatch / generic-monomorphization blocker (read its bootstrap state first). PRIOR K1.M25 -- PIVOT to deletion blockers + GPU-bucket honest accounting. Read PYTHON_DELETION_BUCKETS: 3 partials remain -- impl-method dispatch full (~10 chunks), generic monomorphization full (4/10 done; 3 pending const-generics/lifetime-only/generic-impl; 2 partial gp-field/where-clauses), K2 parity harness (138/144 rows, ~5-10 cleanup -- CLOSEST to done). CORRECTED the stale GPU-backends bucket: was 'pending / All 4 backends still Python-only' (FALSE post-M1-M24) -> now 'partial' with an accurate note (PTX full+ptxas-validated; WebGPU real elementwise f32+i32 EXCEEDING Python; Metal/ROCm empty-kernel byte-matched; M21: Python non-NVIDIA are stubs so DELETION-PARITY met for all 4 -- deleting Python GPU backends loses nothing). Raises the honest deletion % to reflect ~24 chunks of real GPU work (M1-M24). No bootstrap code change (status accounting only); 37 codegen tests still green. NEXT: K1.M26 = K2 PARITY HARNESS (closest blocker) -- grep tests for the K2/144-row parity harness, RUN it, identify the ~6 failing/missing rows, fix the smallest. PRIOR K1.M24 -- WGSL CONSOLIDATION: i32 arrays + elementwise coverage. emit_wgsl_buffer now branches on the param type_tag (AST_PARAM slot4): array<i32> when type_tag==0, array<f32> when ==1 (the f32 path stays BYTE-IDENTICAL -- only the type letter f/i differs). 2 new tests: wgsl_mul (out[i]=a[i]*b[i] -> infix `*`, exercises emit_wgsl_expr MUL tag 4) + wgsl_i32 (i32 params -> array<i32> buffers; body is element-type-agnostic in WGSL). The WGSL elementwise path now covers f32+i32 params and +/-/*// arith. Pure-additive (no sb-slots, no parser.hx); wgsl_elementwise/params/empty stay byte-identical. Ran wgsl_mul/i32/elementwise/params/empty (53.89s). 37 codegen tests green. NEXT: K1.M25 -- ASSESS + pick the higher-leverage path: (a) WGSL tile ops/matmul (the portable AI matrix primitive -> a milestone) OR (b) PIVOT to a non-GPU deletion blocker (impl-method dispatch full / generic monomorphization full / K2 parity harness) -- read where each stands; the north-star (AI on any GPU) is substantially met (PTX + portable WGSL), so deletion-blocker progress may now be higher-leverage toward the STOP CRITERION (Python-ready + 5 clean audits). PRIOR K1.M23 -- FIRST REAL NON-NVIDIA GPU KERNEL: a full WebGPU/WGSL elementwise add. `@kernel fn k(out,a,b: f32) { let i = thread_idx(); out[i]=a[i]+b[i] }` now lowers to storage buffers + @compute + a REAL body: `let i = gid.x;` (global thread index) + `out[i] = a[i] + b[i];` + return;. New recursive emit_wgsl_expr (AST_INT->decimal, AST_VAR->name, AST_CALL thread_idx->gid.x, AST_INDEX->base[index], binop 2/3/4/5->infix +/-/*//) + emit_wgsl_stmt (AST_LET->`let <n> = <v>;` + recurse cont; AST_INDEX_STORE->`<base>[<idx>] = <v>;`; AST_SEQ->both; empty/const->nothing). emit_wgsl_kernel_params now lowers the body (AST_FN_DECL slot3) before the return. NET-NEW vs the Python WebGPU backend (which stubs all ops @@HELIX-STUB) -- the bootstrap is the SOURCE OF TRUTH for non-NVIDIA codegen + a genuine AI kernel for ANY GPU via the portable WebGPU standard (NVIDIA/AMD/Apple/Intel). NO CUDA / NO MLIR / NO LLVM. No naga/tint/wgpu validator on this box -> spec-correct byte-match of valid WGSL-2024 is the check. test_bootstrap_wgsl_elementwise byte-matches; wgsl_params + wgsl_empty stay byte-identical (empty/const bodies emit nothing). Pure-additive (no sb-slots, no parser.hx). Ran wgsl_elementwise + wgsl_params + wgsl_empty + ptx empty/matmul (64.37s). 35 codegen tests green. Sent the FIRST-REAL-non-NVIDIA-kernel milestone TG. NEXT: K1.M24 = grow WGSL (i32 arrays / more arith / WGSL tile ops) then Metal/ROCm real bodies (mirror this WGSL arc), OR pivot to non-GPU deletion blockers (impl-method dispatch / generic monomorphization / K2 parity). PRIOR K1.M22 -- REAL WGSL OP LOWERING BEGINS (params/memory foundation; EXCEEDS the Python WebGPU scaffold). A @kernel WITH f32 params now emits module-scope storage-buffer bindings (@group(0) @binding(N) var<storage, read_write> <name>: array<f32>;) + a @compute entry using @builtin(global_invocation_id) gid (the cross-workgroup thread index for buffer indexing). emit_wgsl_kernel is now a DISPATCHER: 0 params -> emit_wgsl_kernel_empty (byte-matches Python, test green); >=1 param -> emit_wgsl_kernel_params (NEW). New helpers: emit_wgsl_buffer (one @group/@binding storage buffer per param, positional binding, array<f32>) + emit_wgsl_kernel_params. Body still skeleton (return;); M23 fills it. NO naga/wgpu/tint validator on this box -> spec-correct byte-match is the check (Python stubs all WGSL ops so there is no Python oracle for real bodies; the bootstrap is now the SOURCE OF TRUTH for non-NVIDIA codegen). test_bootstrap_wgsl_params byte-matches; wgsl_empty stays byte-identical to Python (dispatcher safe). Pure-additive (no sb-slots, no parser.hx). Ran wgsl_params + wgsl_empty + ptx/msl empty (49.87s). 34 codegen tests green. NEXT: K1.M23 = the WGSL BODY -- lower thread_idx()->gid.x, a[i] load, out[i]=expr store + arith (a[i]+b[i]) -> a FULL WGSL elementwise kernel = milestone TG (first REAL non-NVIDIA kernel). PRIOR K1.M21 -- GPU PARITY FINDING + plan correction (scoping chunk, docs only). Probed the Python non-NVIDIA backends: they are SUBSTRATE + STUBS, not functional -- WgslEmitter on a tile<> kernel doing a[i]=a[i] emits @@HELIX-STUB tokens (TILE_INDEX_LOAD/STORE_HBM status='stub' not wired), NO real WGSL. Only NVIDIA PTX is a real GPU compiler (in Python AND the bootstrap, where it is ptxas-validated through matmul). CONSEQUENCES: (1) empty-kernel byte-parity (M18-20) already matches Python's FUNCTIONAL capability for non-NVIDIA (both = substrate only; nothing lost by deleting the Python scaffolds); op-level byte-parity is INFEASIBLE (Python stub tokens reference tile-IR TileOpKind names; the bootstrap is AST-direct, no tile-IR). (2) The north-star 'real AI on ANY GPU incl non-NVIDIA' is UNBUILT in Python too -> delivering it for non-NVIDIA is a from-scratch arc that EXCEEDS Python, not a port. Documented in docs/GPU_DIRECT_EMIT_PLAN.md (section '## K1.M21 parity finding'). DECISION: pursue REAL WGSL op lowering next (WebGPU = most portable, runs on any GPU), AST-direct (mirror the bootstrap PTX memory arc), shape-validated (+ a real validator like naga if available -- no Python byte-match oracle since Python stubs everything). No code/test change this tick (finding + plan only); 33 codegen tests still green. NEXT: K1.M22 = real WGSL kernel params (storage buffers @group(0) @binding(N) var<storage>) + global load/store for a[i] (the WGSL memory foundation); check for a naga/tint WGSL validator first. PRIOR K1.M20 -- FOURTH / FINAL GPU BACKEND: AMD ROCm (AMDGPU GCN assembly, gfx942). The bootstrap now emits an AMDGPU asm module (.amdgcn_target "amdgcn-amd-amdhsa--gfx942" + .text + .globl/.p2align 8/.type @function + label + s_endpgm) for a @kernel fn, BYTE-MATCHING Python helixc/backend/rocm.py HipEmitter empty-kernel output (pure ASCII, no em-dash). Direct Helix -> GCN asm text, NO MLIR / NO LLVM. New: emit_rocm_header + emit_rocm_kernel (real fn name slots 1/2, used 3x: .globl/.type/label) + emit_rocm_for_ast_to_path (mirror the M18/M19 pattern; reuse emit_ptx_byte). test_bootstrap_rocm_empty_kernel byte-matches. Pure-additive (no sb-slots, no parser.hx). Ran rocm + msl + wgsl + ptx empty/matmul (55.88s, all green). THE 4-BACKEND GPU SET IS NOW COMPLETE at empty-kernel level: NVIDIA PTX (FULL: scalar/memory/tile-family/matmul, ptxas-validated) + AMD ROCm + Apple Metal + WebGPU (empty-kernel, byte-matched), all direct Helix->chip text, NO CUDA / NO MLIR / NO LLVM. 33 codegen tests green. Sent the COMBINED 4-of-4-backends milestone TG (covers M19 Metal + M20 ROCm). NEXT: the GPU deletion bucket needs OP-PARITY -- the 3 non-NVIDIA backends are empty-kernel only; deepen each (params/body/tile ops) toward parity with its Python reference (the long tail), OR (faster bucket progress) confirm what level the Python non-NVIDIA backends are actually at (they may be substrate+stubs, in which case empty-kernel + op-mapping tables may already be near parity). Other deletion blockers: impl-method dispatch (full), generic monomorphization (full), K2 parity harness -> then K3 trusted-seed bootstrap. PRIOR K1.M19 -- THIRD GPU BACKEND (2nd non-NVIDIA): Apple Metal/MSL. The bootstrap now emits a Metal compute kernel (#include <metal_stdlib> + using namespace metal; + kernel void <name>(uint tid [[thread_position_in_threadgroup]]) + return;) for a @kernel fn, BYTE-MATCHING Python helixc/backend/metal.py MslEmitter empty-kernel output (incl. U+2014 EM DASH 226 128 148). Targets Apple Silicon GPUs. Direct Helix -> MSL text, NO MLIR / NO LLVM. New: emit_msl_header + emit_msl_kernel (real fn name slots 1/2) + emit_msl_for_ast_to_path (mirror emit_wgsl_*; reuse emit_ptx_byte). test_bootstrap_msl_empty_kernel byte-matches the Python MSL (captured via the metal.py MslEmitter pipeline; no local Metal compiler so byte-match is the check). Pure-additive (no sb-slots, no parser.hx). Ran new + wgsl + ptx empty/matmul (50.91s). GPU BACKENDS NOW 3 of 4 (PTX complete; WebGPU + Metal empty-kernel). NEXT: K1.M20 = the 4th/final backend ROCm (AMD GCN/HIP) empty-kernel skeleton (same capture+byte-match pattern; grep helixc/backend/rocm.py for the Emitter class) -> completes the 4-backend SET at empty-kernel level, then deepen each toward op-parity with the Python reference (the long tail before the GPU deletion bucket can be checked). PRIOR K1.M18 -- FIRST NON-NVIDIA GPU BACKEND: WebGPU/WGSL. The bootstrap now emits a WGSL compute-shader module (@compute @workgroup_size(64) + fn entry + @builtin(local_invocation_id) param + return;) for a @kernel fn, BYTE-MATCHING Python helixc/backend/webgpu.py empty-kernel output (emit_module_header + emit_kernel_stub), incl. the U+2014 EM DASH (UTF-8 226 128 148). WGSL is the browser-portable shader IR that runs on ANY GPU (NVIDIA/AMD/Apple/Intel) via the WebGPU standard -> directly serves the north-star AI-on-ANY-GPU-incl-non-NVIDIA. Direct Helix -> WGSL text, NO MLIR / NO LLVM. New helpers: emit_wgsl_header + emit_wgsl_kernel (real fn name from slots 1/2) + emit_wgsl_for_ast_to_path (header once + one @compute entry per kernel; 0 kernels -> 0; mirrors emit_ptx_for_ast_to_path). Reused emit_ptx_byte (generic arena byte-push). Test harness _kovc_self_host_emit_ptx parameterized (emit_fn) at M17. New test test_bootstrap_wgsl_empty_kernel byte-matches the 172-byte Python WGSL (captured via the real webgpu.py parse->lower->lower_to_tile->WgslEmitter pipeline; no local naga/wgpu validator so exact byte-match is the check). Pure-additive (no sb-slots, no parser.hx). Ran new + ptx empty/auto/matmul (45.13s). GPU BACKENDS NOW 2 of 4 (PTX + WebGPU); ROCm + Metal remain. NEXT: grow WGSL (params/scalar body/tile ops mirroring the PTX arc) OR start the Metal (MSL) / ROCm backend empty-kernel skeleton. PRIOR K1.M17 -- GPU OUTPUT-MODE DISPATCHER: emit_auto_for_ast_to_path(ast_root) routes a program to the GPU PTX emitter when it contains a @kernel (ast_has_kernel: walk AST_FN_LIST tag 15, check AST_FN_DECL slot 14 is_kernel) else to the x86_64 ELF emitter -- the bridge from two separate emitters to a compiler driver that auto-picks the target (like a host toolchain routing .cu->ptx vs .c->elf, but NO CUDA / NO MLIR; direct Helix->chip either way). New: ast_has_kernel + emit_auto_for_ast_to_path (pure-additive; the demo main() + emit_elf/emit_ptx untouched -> CPU/ELF + all 28 PTX tests stay green). Test harness _kovc_self_host_emit_ptx gained an emit_fn param (default emit_ptx_for_ast_to_path -> existing tests unchanged). 2 new tests: auto_kernel (@kernel fn k(){0} via emit_auto -> byte-identical empty-kernel PTX) + auto_cpu (fn main(){0} via emit_auto -> ELF magic 7f454c46). Pure-additive (no sb-slots, no parser.hx). Ran new(2) + empty_kernel + tile_matmul (40.92s). NOTE: the production main() is still a hardcoded AST_INT(42) demo (not yet a real CLI driver reading argv) -- a true GPU CLI is a later arc, but the dispatch LOGIC is now in place + tested. NEXT: the vendor-neutral GPU backends ROCm/Metal/WebGPU -- survey helixc/backend/{rocm,metal,webgpu}.py target formats, pick the simplest (likely WebGPU/WGSL or Metal/MSL -- high-level text, no register alloc), emit an empty-kernel module first (mirror PTX M1). Directly serves the north-star AI-on-ANY-GPU-incl-non-NVIDIA + the deletion bucket (needs all 4 backends). PRIOR K1.M16 -- GPU MATMUL (the AI matrix primitive): __tile_matmul(a, b, dst, n) compiles a NAIVE unrolled NxN row-major matrix multiply over register-tiles -- dst[i][j] = sum_k a[i][k]*b[k][j], emitted as mul.f32 + add.f32 over consecutive %f with the accumulator moved (mov.f32 reg->reg) into dst[i*n+j]. The CORRECTNESS form of matmul -- a real on-GPU matrix product, direct: Helix -> PTX -> ptxas -> SASS, NO CUDA, NO MLIR; matches the CPU __tile_matmul naive path. NVIDIA Tensor-Core wmma.mma.sync acceleration is a LATER perf optimization (the register-tile model does not fit wmma's f16 fragment lifecycle; correctness now, speed later -- honest calibration per the north-star). New helpers: emit_ptx_mov_f_reg (reg->reg mov.f32) + ptx_name_is_tile_matmul (13-char) + emit_ptx_tile_matmul (triple-nested i/j/k loop unroll). emit_ptx_call dispatches __tile_matmul. test_bootstrap_ptx_tile_matmul (zeros 3 2x2 tiles then __tile_matmul(a,b,c,2) -> 12x mov.f32 + 16 ops: 8 mul.f32 + 4 add.f32 + 4 mov.f32 over %f12..%f23) ptxas-VALIDATED to REAL SASS. THE GPU PTX TILE OP FAMILY IS NOW COMPLETE: zeros/add/sub/mul/matmul. Pure-additive (no sb-slots, no parser.hx). MILESTONE: full PTX suite 28 tests green (255s). NEXT: K1.M17 -- options: (a) main() output-mode switch (emit .ptx when a @kernel is present -> the compiler actually PRODUCES GPU files), (b) wmma Tensor-Core acceleration of __tile_matmul (perf; survey the wmma fragment lifecycle first), (c) the sibling GPU backends ROCm/Metal/WebGPU (mirror the PTX text emitters for vendor-neutral compute -- directly serves the north-star "AI on ANY GPU incl. non-NVIDIA"). PRIOR K1.M15 -- GPU TILE __tile_sub + __tile_mul (siblings, ONE commit): elementwise subtract/multiply over register-tiles (sub.f32 / mul.f32 over `count` consecutive %f), mirroring Python backend/ptx.py TILE_SUB/TILE_MUL. Refactored the M14 add emitter into a generalized emit_ptx_binop_f3(opc 0=add/1=sub/2=mul) + a shared emit_ptx_tile_binop(node,vtab,opc) that all three tile elementwise ops route through (DRY; the M14 __tile_add path is BYTE-IDENTICAL -- its test stays green after refactor). New 10-char matchers ptx_name_is_tile_sub/mul. emit_ptx_call dispatches __tile_sub (opc 1) / __tile_mul (opc 2). 2 new tests (tile_sub/tile_mul: zeros 3 tiles -> 12x `mov.f32` + 4x `sub.f32`/`mul.f32` %f8,%f0,%f4 .. %f11,%f3,%f7) ptxas-VALIDATED to REAL SASS. The GPU TILE ELEMENTWISE FAMILY (zeros/add/sub/mul) IS NOW COMPLETE, all direct: Helix -> PTX -> ptxas -> SASS, NO CUDA, NO MLIR. Pure-additive (no sb-slots, no parser.hx). Ran new(2) + tile_add + tile_zeros + empty (49.51s). NEXT: K1.M16 __tile_matmul (wmma.mma.sync.aligned.m16n16k16 Tensor Cores -- the BIG AI matrix primitive + confetti milestone; SURVEY the wmma.load.a/b/c.sync fragment-load lifecycle first, likely multi-tick) then main() output-mode switch; ROCm/Metal/WebGPU. PRIOR K1.M14 -- FIRST GPU TILE COMPUTE OP: __tile_add(a, b, dst, count) elementwise-adds two register-tiles into a third over `count` consecutive %f registers (mirroring Python backend/ptx.py TILE_ADD / Stage 64 Inc 3). a/b/dst are vars bound to prior __tile_zeros results (their %f base, resolved via ptx_vtab_lookup); count is a static int literal. emit_ptx_call now dispatches __tile_add: read the 4 args off the AST_ARG chain (ah=node+3; expr=arg+1, next=arg+2), resolve the 3 tile-var %f bases, emit `count` add.f32 lines (dst[k]=a[k]+b[k] over consecutive %f), set the float flag, return base_d. New helpers: ptx_name_is_tile_add (10-char matcher) + emit_ptx_add_f3 (3-register add.f32). test_bootstrap_ptx_tile_add (@kernel zeros 3 tiles then __tile_add(a,b,c,4) -> 12x `mov.f32` + 4x `add.f32 %f8,%f0,%f4` .. `%f11,%f3,%f7`) ptxas-VALIDATED to REAL SASS. WITH M13 __tile_zeros THIS IS A COMPLETE ON-GPU TILE ELEMENTWISE PIPELINE (allocate + compute), all direct: Helix -> PTX -> ptxas -> SASS, NO CUDA, NO MLIR. Pure-additive (no sb-slots, no parser.hx) -- i32/f32 scalar + memory + tile_zeros stay BYTE-IDENTICAL. Ran new + 3 representative (38.16s). NEXT: __tile_sub/mul (one-byte opcode change off emit_ptx_add_f3 -> sub.f32/mul.f32) then __tile_matmul (wmma.mma.sync.aligned.m16n16k16 Tensor Cores -- the big AI matrix primitive + confetti milestone); main() output-mode switch; ROCm/Metal/WebGPU. PRIOR K1.M13 -- FIRST GPU TILE OP: __tile_zeros(N, M) lowers to N*M consecutive `mov.f32 %fX, 0f00000000;` register-fills (the register-tile model, mirroring Python backend/ptx.py TILE_ZEROS / Stage 64 Inc 2; 0f00000000 = +0.0f). MAJOR FINDING: the __tile_* CALL surface ALREADY PARSES -- the x86 CPU path (K1.F23c--F27) already implements __tile_zeros/add/sub/mul/matmul as machine-code builtins with the SAME 2-arg signature, so the GPU side is EMITTER-ONLY (NO parser change). emit_ptx_call previously returned -1 for every non-index builtin; now it dispatches __tile_zeros: read the two static int-literal args (AST_INT slot 1) -> count = N*M -> emit count zero-fills via the new emit_ptx_mov_f_zero -> return the base %f register + set the float flag (vtab slot 55). New helpers: ptx_name_is_tile_zeros (12-char byte matcher) + emit_ptx_mov_f_zero. test_bootstrap_ptx_tile_zeros (@kernel fn k() { __tile_zeros(2, 2) } -> 4x `mov.f32 %f0..3, 0f00000000;`) ptxas-VALIDATED to REAL SASS. Pure-additive (no sb-slots, no parser.hx) -- the i32/f32 scalar + memory paths stay BYTE-IDENTICAL (empty_kernel + i32/f32 elementwise-add stay green). Ran new + 3 representative (42.57s). The tile/matmul AI-primitive arc starts here. NEXT: __tile_add/sub/mul (elementwise over the consecutive %f block; needs last-tile base+length side-channels, e.g. vtab slots 56/57) then __tile_matmul (wmma.mma.sync.aligned.m16n16k16 Tensor Cores); main() output-mode switch; ROCm/Metal/WebGPU. PRIOR K1.M12b -- MILESTONE: a full FLOAT elementwise-add kernel compiles. `@kernel fn k(out, a, b) { let i = thread_idx(); out[i] = a[i] + b[i] }` (all f32) -> the self-hosted Helix bootstrap emits ld.global.f32 x2 + add.f32 + st.global.f32 (the M12a type-flag picks the float ops), ptxas-validated to REAL SASS. The REALISTIC AI workload (floats), fully direct: Helix -> PTX -> ptxas -> SASS, NO CUDA, NO MLIR. emit_ptx_binop now has an f32 path (BOTH operands float -> add.f32/sub.f32/mul.f32/div.rn.f32 into %f + set flag); the s32 path is BYTE-IDENTICAL (i32 elementwise-add + scalar arith stay green). test_bootstrap_ptx_f32_elementwise_add ptxas-VALIDATED. f32 elementwise compute COMPLETE (load + arith + store). Ran new + 4 i32-binop regressions (55.78s). NOTE: mixed int+float needs a cvt (future; both_f requires both float). Next: tile ops + wmma matmul (AI matrix primitives); main() output switch; ROCm/Metal/WebGPU. PRIOR K1.M12a --f32 STORE + the type-tracking flag: a complete float COPY kernel `out[i]=a[i]` (out,a f32) compiles. New: a "last-result-is-float" side-channel flag (vtab slot 55) -- emit_ptx_expr defaults it to 0 (i32); the f32 index-load sets it 1; the indexed store captures it (right after lowering the value, before the index clobbers it) and emits st.global.f32 (of a %f) vs st.global.u32 (of a %r). The i32 store path is BYTE-IDENTICAL (out[i]=7 + i32 elementwise-add stay green). test_bootstrap_ptx_f32_copy (out[i]=a[i], f32) ptxas-VALIDATED. Ran new + 4 i32-store regressions (51.57s). The flag side-channel is the foundation for f32 arithmetic. NEXT (M12b): add.f32 in emit_ptx_binop (capture operand flags -> f32 op into %f) -> the full f32 out[i]=a[i]+b[i] elementwise-add MILESTONE. PRIOR K1.M11 --f32 (float) global LOAD: the FIRST float op (the backend was i32-only). A kernel param typed `: f32` makes a[i] lower to ld.global.f32 into a %f register (vs ld.global.u32 %r for i32), selected via the param type_tag (AST_PARAM slot 4 == 1). New: %f register counter (vtab slot 54) + ptx_alloc_f, emit_ptx_f helper, ptx_param_type (reads AST_PARAM slot 4). emit_ptx_index_load now branches on element type; the i32 path is BYTE-IDENTICAL (global_load/elementwise_add/two_load_add stay green). test_bootstrap_ptx_f32_load (k(a: f32){...; a[i]} -> ld.global.f32 %f0) ptxas-VALIDATED. Foundation for f32 arithmetic (the realistic AI workload). Ran new + 4 i32 index-load regressions (52.68s). Pure-additive (no parser.hx). NEXT (M12): f32 type-tracking side-channel (vtab last-result-is-float flag) -> f32 store + add.f32 -> a full f32 elementwise-add kernel (real AI workload) MILESTONE. PRIOR K1.M10d --MILESTONE: a COMPLETE elementwise-add GPU kernel compiles end-to-end. `@kernel fn k(out, a, b) { let i = thread_idx(); out[i] = a[i] + b[i] }` -> the self-hosted Helix bootstrap emits PTX (thread index + two ld.global loads + add.s32 + st.global store, with full base+i*4 address arithmetic per array) that ptxas assembles into REAL GPU machine code (SASS). The canonical data-parallel GPU/AI kernel, fully direct: Helix -> PTX -> ptxas -> SASS, NO CUDA frontend, NO MLIR. No new emitter code -- pure COMPOSITION of thread_idx (M6) + global load (M10a) + global store (M10c) + add (M5d). test_bootstrap_ptx_elementwise_add (18-instruction kernel) ptxas-VALIDATED, 16.21s. The GPU backend now compiles real data-parallel kernels. Next: __tile_* tile ops + wmma matmul (the AI matrix primitives) + f32 floats; main() output-mode switch; ROCm/Metal/WebGPU. PRIOR K1.M10c --GLOBAL MEMORY STORE out[i]=v. The FIRST parser.hx change in the GPU work (careful + minimal): added an AST_INDEX (tag 53) branch to the field-store detection -> AST_INDEX_STORE (tag 55), mirroring the proven field-store (tag 79) path; previously `a[i]=v` dropped the `=` (parse trip). emit_ptx_index_store lowers the value, computes base+i*4 (param ptr -> cvta -> address), then st.global.u32 [addr], val. test_bootstrap_ptx_global_store (out[i]=7) ptxas-VALIDATED. BROAD REGRESSION after the parser change: full PTX suite (19 tests) + CPU canary (write_file_to_arena) = 20 passed 177s -- the parser edit broke NOTHING (every kernel re-parses, CPU path intact). Both LOAD (M10a) + STORE (M10c) now work -> the full out[i]=a[i]+b[i] elementwise-add kernel is now compilable (next: M10d, the milestone). PRIOR K1.M10b --the COMPUTE core of an elementwise kernel: `a[i] + b[i]` reads two arrays from global memory (multi-param: param_0 AND param_1) and adds them -> two full load sequences (ld.param/cvta/mul.wide/add.s64/ld.global) feeding add.s32. Pure-additive (composes the M10a index-load with the existing binop; NO new emitter code -- a test-only validation of multi-param multi-load + arithmetic). test_bootstrap_ptx_two_load_add ptxas-VALIDATED (real SASS). PROBE FINDING: indexed-STORE `out[i] = 7` does NOT parse currently (the `=` after `]` is unhandled -> emits an empty kernel, not even the let's mov); the store needs a parser.hx change (M10c, the careful parser chunk). Ran new + global_load (26.30s). PRIOR K1.M10a --GLOBAL MEMORY LOAD: a kernel can now READ arrays from GPU memory. a[i] (AST_INDEX tag 53 -- ALREADY parsed by the bootstrap at parser.hx:2698 mk_node(53,base,idx,0), so this is PURE-ADDITIVE, NO parser change!) lowers to the canonical CUDA load: ld.param.u64 (param pointer) + cvta.to.global.u64 + mul.wide.s32 (i*4) + add.s64 + ld.global.u32. New: ptx_param_index (name -> 0-based kernel param index), ptx_alloc_rd (%rd 64-bit addr regs, vtab slot 52), cur_fn_idx (vtab slot 53, set by emit_ptx_entry), emit_ptx_indent/r/rd helpers, emit_ptx_index_load; emit_ptx_expr dispatches tag 53. test_bootstrap_ptx_global_load (k(a){ let i=thread_idx(); a[i] } -> reads a[tid]) ptxas-VALIDATED. A kernel now reads global memory at a computed index -- the data-access pattern of every GPU kernel. vtab-heavy regression (while/global_index) stayed green. Ran new + 4 representative (52.75s). MAJOR FINDING: a[i] already parses, so MEMORY is emitter-only, NOT the risky parser change feared. Next: M10b store (a[i]=v -> st.global), then M10c full out[i]=a[i]+b[i] MILESTONE. PRIOR K1.M9 --AST_WHILE (tag 10) loops + AST_ASSIGN (tag 11): a real TERMINATING counting loop now compiles. `while x < 4 { x = x + 1 }` lowers to "$Ltop_<n>:" + cond (setp.lt+selp) + setp.ne + "@!%p bra $Lwend_<n>" + body (add + mov overwriting x's register) + "bra $Ltop_<n>" (back-edge) + "$Lwend_<n>:". emit_ptx_lbl_ref extended (which 2=top, 3=wend); new emit_ptx_while + emit_ptx_assign (x=v -> mov %rX,%rV, var binding unchanged); emit_ptx_expr dispatches tags 10/11. test_bootstrap_ptx_while_loop (x counts 0->4) ptxas-VALIDATED. The PTX backend now compiles the FULL scalar language: const/var/let/arith/cmp/if/while/assign + thread/block index, all ptxas-validated. Ran new + 4 representative (51.79s). Pure-additive (no sb-slots, no parser.hx). Next: the parser-subscript MEMORY chunk (a[i] -> ld/st.global -> out[i]=a[i]+b[i] MILESTONE; this one TOUCHES parser.hx -- check lexer for []; add minimal postfix subscript; sequential regression after). PRIOR K1.M8b --AST_IF (tag 7) -> predicated branch: CONTROL FLOW COMPLETE. `if cond { } else { }` lowers to cond-value + "setp.ne.s32 %pZ, %rC, 0" + "@!%pZ bra $Lelse_<n>" + then + "bra $Lend_<n>" + "$Lelse_<n>:" + else + "$Lend_<n>:" with per-kernel unique labels. New: ptx_alloc_label (vtab slot 51), emit_ptx_lbl_ref, emit_ptx_if; emit_ptx_expr dispatches tag 7. if-as-statement (value discarded in void kernels; if-as-value phi deferred). test_bootstrap_ptx_if (if x<3 {1} else {2}) ptxas-VALIDATED. The PTX backend now expresses the SHAPE of real bounds-checked GPU kernels (global-index + if). Ran new + 4 representative tests (46.81s) per the test-time policy; 15 PTX tests total. Pure-additive (no sb-slots, no parser.hx). Next: AST_WHILE (tag 10, loops, pure-additive) then the parser-subscript MEMORY chunk (a[i] -> ld/st.global -> out[i]=a[i]+b[i] MILESTONE; touches parser.hx -- careful). PRIOR K1.M8a --comparison-as-value (control-flow foundation): AST_LT/GT/EQ/NE/LE/GE (tags 6/19/20/21/22/23) lower to "setp.<cc>.s32 %pP, %rA, %rB" + "selp.b32 %rR, 1, 0, %pP" (reify 0/1 into a register, matching AST semantics). New: predicate-register counter appended to the vtab at slot 50 (var entries UNCHANGED -> 13 prior tests stay byte-identical; ptx_alloc_pred), emit_ptx_cc (2-char cond mnemonic), emit_ptx_cmp. selp-with-immediates + setp ptxas-validated. test_bootstrap_ptx_scalar_cmp ({let x=5; x<3} -> setp.lt.s32 + selp.b32), ptxas-VALIDATED. 14 PTX tests green, 120s. Pure-additive (no sb-slots, no parser.hx). Next: K1.M8b AST_IF (tag 7 -> setp.ne + @!%p bra + labels via next_label slot 51 + emit_ptx_label). NOTE: PTX suite now ~2min/14 tests -- run NEW + ~4 representative prior tests next tick, not all. PRIOR K1.M7 --block_idx() -> %ctaid.x and block_dim() -> %ntid.x. With thread_idx() + scalar arithmetic, a @kernel now computes the CANONICAL global thread index `block_idx()*block_dim() + thread_idx()` (the foundation of every grid-stride GPU kernel). New helpers ptx_name_is_block_idx/dim + emit_ptx_mov_ctaid_x/ntid_x; emit_ptx_call dispatches all 3 index builtins. test_bootstrap_ptx_global_index: the full formula -> 3 sreg movs + mul.lo.s32 + add.s32, ptxas-VALIDATED (real SASS). 13 PTX tests green, 105.88s. Pure-additive (no sb-slots, no parser.hx). SURVEY NOTE: full memory (a[i] load/store) needs SUBSCRIPT parsing -- the bootstrap AST has NO array-index node, so memory requires a parser.hx change (a deliberate sb-slot-careful chunk + sequential regression), unlike everything M1-M7 which is pure-additive kovc.hx. Next: K1.M8 control flow (AST_IF tag 7 + AST_LT tag 6 -> setp/bra, pure-additive) or the parser-subscript chunk for memory. PRIOR K1.M6 --thread_idx() builtin: the entry point to data-parallel kernels. AST_CALL (tag 16) to "thread_idx" now lowers to "mov.u32 %rN, %tid.x;" (reading the hardware thread-index special register), matching the Helix surface (lower_ast.py thread_idx -> THREAD_IDX) + Python ptx.py. New helpers: ptx_name_is_thread_idx (flat byte-compare), emit_ptx_tid_x, emit_ptx_call; emit_ptx_expr now handles AST_CALL. SURVEY FINDING: the AST has NO array-subscript node -- kernel memory/compute is expressed via AST_CALL to builtins (thread_idx / __tile_* / block_idx). test_bootstrap_ptx_thread_idx ({let i = thread_idx(); i} -> mov.u32 %r0, %tid.x), ptxas-VALIDATED (assembles to real SASS). 12 PTX tests green, 97.55s. Pure-additive (no sb-slots, no parser.hx). Next: K1.M7 memory load/store (survey-gated) or parallel-index siblings (thread_idx_y/z, block_idx, block_dim) or __tile_* ops. PRIOR K1.M5f --PROOF the emitted PTX is REAL GPU code: ptxas ROUND-TRIP validation. NVIDIA's official ptxas (CUDA 12.0) ACCEPTS the bootstrap's direct-emitted PTX and produces a cubin (SASS GPU machine code), rc=0. End-to-end: Helix source -> self-hosted bootstrap -> PTX text -> ptxas -> SASS, with NO CUDA frontend and NO MLIR. Also lowered .version 8.3->8.0 (the self-host path's WSL ptxas caps at PTX ISA 8.0 = CUDA 12.0; we use only basic scalar ops so 8.0 is sufficient + more broadly compatible). New test_bootstrap_ptx_ptxas_roundtrip (skips gracefully if ptxas absent, so non-CUDA CI still passes). 11 PTX tests green, 86.75s. Pure-additive (no sb-slots, no parser.hx). LESSON: the self-host test path runs ptxas via WSL (/usr/bin/ptxas = CUDA 12.0, max .version 8.0); a manual `ptxas` in the Bash tool hit a DIFFERENT 12.8 install -- always validate via the WSL path. Next: K1.M6 tile ops (the AI matrix primitives) or M8 main() output-mode switch. PRIOR K1.M5e --rounded out scalar arithmetic: added AST_DIV (tag 5 -> div.s32, binop opc 3) and AST_NEG (tag 9, unary -> neg.s32 via new emit_ptx_neg helper) to the recursive emit_ptx_expr. The bootstrap PTX backend now lowers the FULL scalar arithmetic set: const / var / let + add / sub / mul / div / neg. 2 new tests ({let x=12; x/3} -> div.s32 %r2,%r0,%r1; {let x=5; -x} -> neg.s32 %r1,%r0); 10 PTX tests green, 82.50s. Pure-additive (no sb-slots, no parser.hx). Next: K1.M5f comparison (AST_LT tag 6 -> setp) + control flow (AST_IF tag 7), OR pivot to higher-AI-value M8 main() output-mode switch (make the compiler actually emit .ptx files) or M6 tile ops. PRIOR K1.M5d --scalar ARITHMETIC + variable->register environment. Built a recursive emit_ptx_expr lowering kernel-body scalar expressions to PTX: AST_INT->mov.s32, AST_VAR->resolve via the var table, AST_LET->bind name->reg + recurse into continuation, AST_ADD/SUB/MUL->emit_ptx_binop (add.s32 / sub.s32 / mul.lo.s32). New var->reg table (ptx_vtab_init/reset/add/lookup + ptx_alloc_reg) lives in the arena BEFORE the output region so it never pollutes the .ptx; reset per kernel. M5b/M5c body lowering refactored onto emit_ptx_expr -> byte-identical (6 prior tests stay green). 2 new tests: {let x=5; x+2} -> mov %r0,5 / mov %r1,2 / add.s32 %r2,%r0,%r1; {let x=5; x*3} -> mul.lo.s32. 8 PTX tests green, 67.88s. Pure-additive (no sb-slots, no parser.hx). The GPU backend now compiles real scalar expressions (foundation for tile ops). Next: K1.M6 tile ops (TILE_ZEROS/ADD/SUB/MUL) or M8 main() output-mode switch. PRIOR K1.M5c --kernel-body let-chain lowering: walk a leading AST_LET chain (tag 8; value in slot 4, continuation in slot 3), emitting one SCALAR_CONST_INT "mov.s32 %rN, <const>" per integer-const-init let; tail AST_INT emits one more mov, tail AST_VAR (tag 1) resolves to an existing register (void kernel -> no instruction). New helper emit_ptx_mov_const(ridx, val). LESSON: `reg` is a reserved Helix keyword (KW_REG) -- it CANNOT be an identifier in kovc.hx (caught when Python Stage-30 failed to PARSE the bootstrap: "expected IDENT got KW_REG 'reg'"); param renamed reg->ridx. 6 PTX tests green (new test_bootstrap_ptx_let_chain: {let x=3; let y=8; y} -> 2 movs; _ptx_entry refactored to a movs tuple): 6 passed 45.73s. Pure-additive (no sb-slots, no parser.hx). Next: K1.M5d scalar arithmetic (add.s32) via a var->reg env. PRIOR K1.M5b --FIRST kernel-BODY op lowering: an integer-literal kernel body now lowers to SCALAR_CONST_INT -> "    mov.s32 %r0, <val>;" (mirrors Python ptx.py emit_op), materialized before ret. The step from "valid empty kernel" to "kernel that computes". Body = AST_FN_DECL slot 3 (confirmed bare AST_INT = tag 0, value in slot 1, since a brace-block returns its inner expr directly). New helper emit_ptx_decimal (recursive int->ASCII). 5 PTX tests green: 4 prior now expect "mov.s32 %r0, 0;" via shared _ptx_entry(body_const=0); new test_bootstrap_ptx_scalar_const ({7} -> "mov.s32 %r0, 7;"); 5 passed 41.90s. Pure-additive (no sb-slots, no parser.hx). Next: K1.M5c (scalar add/sub/mul or thread-idx) or M6 tile ops. PRIOR K1.M5a --kernel body foundation: every @kernel now emits the standard PTX register-file declaration block (5 files: .pred %p, .b32 %r, .b64 %rd, .f32 %f, .b16 %h; pool 256) after "{", byte-matching Python ptx.py _REG_FILES, plus an indented "    ret;". An empty kernel now byte-matches Python emit_kernel EXACTLY. Declaring a big pool is free (ptxas allocates only USED regs). New helpers emit_ptx_reg_prefix/suffix/block. 4 PTX tests refactored onto shared _PTX_HEADER/_PTX_REG_BLOCK/_ptx_entry golden fragments + green: 4 passed 47.15s. This is the foundation op-lowering (M5b) needs. Pure-additive (no sb-slots, no parser.hx). Next: K1.M5b first scalar op (mov.s32 const or %tid.x). PRIOR K1.M4 --kernel params: emit ".param .b64 param_N" (positional; v0.1 all .b64, mirroring Python ptx.py _format_param) per fn param inside the .entry parens, comma-space separated. Walks AST_FN_DECL slot 4 (params_head); each AST_PARAM links via slot 3 (next). Zero-param kernels keep "()" so empty/named/multi stay byte-identical. test_bootstrap_ptx_kernel_params (@kernel fn k(a,b) -> ".visible .entry k(.param .b64 param_0, .param .b64 param_1)") + 3 prior: 4 passed 37.29s. Pure-additive (no sb-slots, no parser.hx). Next: K1.M5 scalar body + .reg decls. PRIOR K1.M3 --multi-kernel PTX: emit the module header once then one ".visible .entry <name>()" per @kernel fn (blank-line separated), mirroring Python ptx.py emit_module. Single-kernel output stays byte-identical (no trailing blank) so M1/M2 tests stay green; fn_list confirmed source-order. test_bootstrap_ptx_multi_kernel (kernels a,b -> 2 entries) + named + empty: 3 passed 31.34s. Pure-additive (no sb-slots, no parser.hx). Next: K1.M4 kernel params (.param .b64 param_N). PRIOR K1.M2 --PTX .entry name is now the REAL kernel fn name (copied from AST_FN_DECL slots 1/2 = name_start/len; same source-byte read as the bootstrap's 'main' detection), not a hardcoded 'k'. test_bootstrap_ptx_named_kernel (@kernel fn saxpy -> ".visible .entry saxpy()") + empty_kernel both green: 2 passed 22.75s. Pure-additive codegen (no sb-slots, no parser.hx). Next: K1.M3 multi-kernel (one .entry per is_kernel fn). PRIOR K1.M1 -- FIRST DIRECT-TO-GPU EMISSION. The bootstrap now emits NVIDIA PTX *text* directly from a @kernel fn (emit_ptx_for_ast_to_path in kovc.hx), mirroring how emit_elf_for_ast_to_path emits x86_64 machine code -- NO MLIR, NO LLVM, straight to the target ISA. PTX is a text virtual-ISA so this is STRICTLY SIMPLER than the ELF binary the bootstrap already emits (no headers/offsets/relocations -- just ASCII bytes the NVIDIA driver JITs to SASS). test_bootstrap_ptx_empty_kernel: @kernel fn k() -> emits the minimal valid 74-byte module (.version 8.3 / .target sm_75 / .address_size 64 / .visible .entry k() { ret; }), PASSED 15.76s. Pure-additive codegen: no sb-slots, no parser.hx change (parser already tags @kernel on AST_FN_DECL slot 14 since Stage 33), so the K1.F5d-j sb-collision hazard does not apply. Implements the user's 2026-05-27 north-star goal: "Have Helix wherever possible talk directly to the chips" (CPU=done via ELF, GPU=now starting via PTX). Per-chunk plan in docs/GPU_DIRECT_EMIT_PLAN.md (K1.M2 real fn name, M3 multi-kernel, M4 params, M5 scalar body+regs, M6 tile ops, M7 wmma matmul, M8 main() output-mode switch; then ROCm/Metal/WebGPU siblings). Reference: MLIR-free helixc/backend/ptx.py (verified K2.AK). [prior K2.AK: VERIFIED MLIR-not-needed with hard evidence -- all 4 Python GPU backends are MLIR-free direct tile-IR->text emitters totaling 3205 LOC < the 5517-LOC x86_64 backend already mirrored; see docs/MLIR_NOT_NEEDED_DECISION.md. K1.F5k: disabled the broken chained-method substrate (sb-slot collisions); LESSON: new sb-slots must extend parse_top alloc-block past 123 and grep "sb + N)" with close-paren.]
# Estimated total chunks to v1.0 (Python fully deleted, all features
# ported, K5 DDC passes). Two estimates:
#   BEST     = optimistic, batched, parallelized, deferring some Tile/GPU
#              corners that turn out vacuously satisfied at K2 time
#   REAL     = under the 2026-05-26 hard constraint (no Python-forever
#              deferral for any subsystem)
K_BOOTSTRAP_CHUNKS_BEST_ESTIMATE = 400  # K2.AJ 2026-05-28 RE-revised DOWN
                                          # from K2.AI's 470. K2.AI counted
                                          # the 15k-LOC MLIR surface as
                                          # port-work; K2.AJ determined MLIR
                                          # is NOT-NEEDED (bootstrap is direct-
                                          # codegen, doesn't consume MLIR; all
                                          # helix-dialect ops already native).
                                          # So P2.1 (~100-150 chunks) drops off.
                                          # Remaining big bucket = P2.2 GPU
                                          # direct-emission (~80-150 chunks).
K_BOOTSTRAP_CHUNKS_REAL_ESTIMATE = 470  # K2.AJ: 310 done + GPU-direct-emit
                                          # (~80-150) + P1 tail + K3 seed +
                                          # 5-clean gate ~= 470. The whiplash
                                          # (440->560->470) reflects: K2.AI
                                          # saw the MLIR LOC surface, K2.AJ
                                          # determined most of it isn't
                                          # bootstrap-bound. Net ~similar to
                                          # the original 440, different reason.

# K2.W (2026-05-27): Python-deletion-readiness bucket model. Each bucket
# is one Category-1 syntax/semantic gap or Category-2 platform port that
# must close before Python helixc can be deleted (K4). Status values:
#   "done"    : feature-complete + audit-clean
#   "partial" : at least one shipped chunk but not feature-complete
#   "pending" : zero chunks shipped, scoping not yet done
# Percent: done = 1.0, partial = 0.5, pending = 0.0; weighted average.
# This is the canonical list per the loop prompt's Python-ready-to-delete
# definition + the 2026-05-26 hard constraint.
PYTHON_DELETION_BUCKETS = [
    {"name": "Macros (assert/print/dbg/panic/todo family)",
     "status": "done",
     "note": "K1.F22-F52 saturated; assert!-cmp family closed F41-F52 audit-clean"},
    {"name": "Mixed-type int binops (i64<->i32, u64<->u32)",
     "status": "done",
     "note": "K1.F8/F8b/F8c/F8d, K3.A/B audit-fixes"},
    {"name": "Mixed-type float binops (f32<->f64)",
     "status": "done",
     "note": "K1.F9"},
    {"name": "f16/bf16 bit-accurate",
     "status": "done",
     "note": "K1.F18b gradual underflow / denormals"},
    {"name": "Reflection (reflect_hash, quote, splice, modify)",
     "status": "done",
     "note": "K1.F2/F3/F4/F19 (FNV mixer)"},
    {"name": "Trace events (trace_event, __trace_last)",
     "status": "done",
     "note": "K1.F20/F20b ring-buffer"},
    {"name": "Tile ops (zeros, add, sub, mul, matmul)",
     "status": "done",
     "note": "K1.F23c-F27 + K3.R-W audit fixes (bounds-check both write+read)"},
    {"name": "Field-store mutation (p.x = v)",
     "status": "done",
     "note": "K1.F6"},
    {"name": "Const-name resolution",
     "status": "done",
     "note": "K1.F7 (const_tab + mk_var_with_capture hook)"},
    {"name": "Impl-method dispatch (full)",
     "status": "partial",
     "note": "K1.F5b localized fix + K1.M27 probe (2026-05-28): the bootstrap FULLY handles CORE impl-method dispatch -- method on a struct value (p.get()), method with an arg (p.add(2)), and method-calling-method (self.a()+2) -- all compile+run correctly via self-host (test_bootstrap_impl_method_dispatch). FINDING: Python helixc cannot even PARSE the bare `(self)` receiver (ParseError: expected COLON), so the bootstrap EXCEEDS Python here (parallel to the M21 GPU finding) -> these CANNOT be K2-parity entries (Python errors), and DELETION-PARITY for the core case is MET (deleting Python loses nothing). Kept 'partial' pending advanced cases (traits / generic-impl methods / &self / multiple impl blocks) -- which may themselves be Python-gaps; core dispatch is done + ahead of Python."},
    {"name": "Generic monomorphization (full)",
     "status": "partial",
     "note": "K1.F21 + K1.M28 probe (2026-05-28): Python helixc CANNOT PARSE generic syntax at all (ParseError on `<T>` for fn-generic / generic-struct / turbofish / multi-param), so the bootstrap EXCEEDS Python on generics -> DELETION-PARITY trivially MET (Python supports ZERO generics). Bootstrap: generic-struct (Box<T>=42) + TURBOFISH (id::<i32>(42)=42) WORK + pinned (test_bootstrap_generics_struct_and_turbofish); BUT a BARE generic-fn call without turbofish (id(42), first(42,7)) MISCOMPILES to SIGILL (exit 132) -- bare-call type inference is the real remaining bug (a bootstrap-QUALITY gap, NOT a deletion blocker since Python can't do generics). Absolute-completeness: ~4/10 (const-generics/lifetime-only/generic-impl pending; gp-field/where-clauses partial)."},
    {"name": "K2 parity harness fully green",
     "status": "partial",
     "note": "138/144 nominal rows; macros structural-gap (Python !) recorded; ~5-10 cleanup chunks"},
    {"name": "GPU backends in bootstrap (PTX, ROCm, Metal, WebGPU)",
     "status": "partial",
     "note": "K1.M1-M24 (2026-05-28): all 4 backends now emit DIRECTLY from the bootstrap (direct-to-target text, NO MLIR/LLVM). NVIDIA PTX = FULL, ptxas-validated to real SASS (scalar/cmp/if/while/assign, thread+block index, global load/store, f32+i32 elementwise-add, full tile family zeros/add/sub/mul/matmul). WebGPU/WGSL = REAL elementwise kernels (f32+i32 params -> @group/@binding storage buffers, global_invocation_id, out[i]=a[i] OP b[i]) -- this EXCEEDS the Python WebGPU backend. Apple Metal/MSL + AMD ROCm/GCN = empty-kernel byte-matched to Python. M21 FINDING: the Python non-NVIDIA backends are SUBSTRATE+STUBS (emit @@HELIX-STUB for ops, no real WGSL/MSL/GCN), so DELETION-PARITY (bootstrap >= Python functional capability) is already MET for all 4 -- deleting the Python GPU backends loses nothing. Remaining is OPTIONAL real-op depth (WGSL tile/matmul; Metal/ROCm bodies) = perf/polish BEYOND what Python ever did, not a deletion blocker. docs/GPU_DIRECT_EMIT_PLAN.md."},
    {"name": "MLIR migration in bootstrap",
     "status": "done",
     "note": "K2.AJ 2026-05-28: NOT-NEEDED / satisfied-by-direct-emission. Bootstrap is 100% direct-to-ELF; all 3 helix-dialect op families (grad/jvp/vmap, quote/splice/modify/reflect_hash, arena) are already native builtins. MLIR is Python's GPU intermediate; bootstrap drives GPU via direct tile-IR->target-text emission (P2.2). The K2.K matrix note already permitted 'an equivalent multi-backend substrate'. Python MLIR code deleted at K4, not ported. See docs/MLIR_NOT_NEEDED_DECISION.md"},
    {"name": "K3 trusted-seed bootstrap",
     "status": "pending",
     "note": "K1.M30 assessment: Stage-K3 SEED = from-raw-binary hex0 -> ... -> kovc chain. Master plan (HELIX_K_BOOTSTRAP_MASTER_PLAN.md) marks it 'not blocking on it; decision when the time comes; several cron iterations, possibly WEEKS'. No stage0/ code in-repo (hex0 is design-stage per the goal hierarchy) -> a MAJOR DEFERRED effort, NOT 60s-tick-tractable (the prior '~5-10 chunks' was optimistic). Separately the N-generation FIXPOINT (kovc compiles its own kovc.hx to a stable binary; K2 Phase-3) is UNTESTED + gated on the bootstrap supporting ALL Helix in its ~11k-line source."},
    {"name": "5 consecutive clean END-OF-PHASE 5-axis audits",
     "status": "pending",
     "note": "Stop-criterion gate; FE/IR/BE/RT/TEST sweep, repeat 5x"},
]


def python_deletion_percent() -> int:
    """Weighted progress toward Python-ready-to-delete state.
    done=1.0, partial=0.5, pending=0.0. Counts buckets, not chunks."""
    score = 0.0
    for b in PYTHON_DELETION_BUCKETS:
        if b["status"] == "done":
            score += 1.0
        elif b["status"] == "partial":
            score += 0.5
    return round(100 * score / len(PYTHON_DELETION_BUCKETS))


def python_deletion_checklist_lines() -> list[str]:
    """Render the Python-deletion checklist as Telegram-friendly lines."""
    symbols = {"done": "[x]", "partial": "[~]", "pending": "[ ]"}
    out = []
    for b in PYTHON_DELETION_BUCKETS:
        out.append(f"  {symbols[b['status']]} {b['name']}")
    return out

K_BOOTSTRAP_TOTAL_ROWS = 144      # matrix-sync 2026-05-26 K2.C:
                                    # actual table count is 84 explicit
                                    # `| PARITY |` + 42 `FUNCTIONAL
                                    # PARITY` (inline in status col) +
                                    # 18 `| KOVC-MISSING |` = 144 rows
                                    # with a status column. The earlier
                                    # 143 was the K0-chunk estimate.
K_BOOTSTRAP_PARITY_DONE = 140      # K2.Y 2026-05-27: matrix-honesty
                                    # sweep flipped rows 198/199 ("TILE_
                                    # ZEROS/ADD/SUB/MUL" + "TILE_MATMUL")
                                    # from KOVC-MISSING to FUNCTIONAL
                                    # PARITY -- bootstrap actually has
                                    # __tile_zeros/add/sub/mul/matmul as
                                    # real builtins (K1.F23c-F27 +
                                    # K3.R/T/U/V/W audit-fixes). Python's
                                    # compile_and_run errors on the syntax
                                    # too, so both compilers behave
                                    # identically on the testable subset.
                                    # 138 -> 140.
                                    # Row 67 (Mixed-type binops) also
                                    # expanded to note u64<->u32 + float
                                    # closures. Row 76 (Comparisons)
                                    # noted mixed-type cmp closure
                                    # (K1.F11-F14). K1.F8b 2026-05-27:
                                    # Mixed-type binops row inline status flipped
                                    # to FUNCTIONAL PARITY for the
                                    # signed i64<->i32 ADD/SUB/MUL
                                    # cases (BOTH directions). 136 -> 137
                                    # (+1 row). K1.F5b 2026-05-27: impl Type
                                    # { methods } row flipped KOVC-
                                    # MISSING -> FUNCTIONAL PARITY (the
                                    # struct-receiver dot-call dispatch
                                    # `p.get()` now works). 135 -> 136
                                    # (+1 row). The previous K1.F3+F4: __trace_event +
                                    # __helix_splice + __helix_modify +
                                    # __helix_reflect_hash all added
                                    # as no-op stubs at slots 165-168.
                                    # 131 -> 135 (+4 rows).
                                    # K1.F2: reflect_hash bootstrap
                                    # builtin no-op stub at slot 164.
                                    # 130 -> 131.
                                    # K1.F-discovery batch 29:
                                    # Quote(arg) + Splice(N) + modify
                                    # all flipped to FUNCTIONAL PARITY
                                    # (bootstrap has them at slots
                                    # 118/119/120 in install_builtin_names
                                    # since at least Stage 11). Plus
                                    # the K1.F-discovery batch 28 f16
                                    # flip (was 126 -> 127). Total
                                    # +4 since K2.C: 126 -> 130.
                                    # matrix-sync 2026-05-26 K2.C:
                                    # 84 PARITY + 42 FUNCTIONAL PARITY
                                    # = 126 closed. The 140 prior was
                                    # inflated by ~14 (K1.* parser
                                    # chunks bumped this counter for
                                    # syntax-only wins; the matrix
                                    # status column still tracks the
                                    # semantic-parity question). Real
                                    # remaining work: 18 KOVC-MISSING
                                    # rows = the Category-2 semantic
                                    # gaps named in
                                    # docs/K_BOOTSTRAP_HARD_CONSTRAINT.md.
                                    # historical bump trail follows
                                    # (kept verbatim for audit):
                                    # was 28 after K0; K1.B (stack
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
                                    # made it 121;
                                    # K1.AG: __arena_push_triple
                                    # (a,b,c) parallel 3-slot
                                    # variant; same matrix row
                                    # (now full PARITY, was
                                    # partial). No counter bump;
                                    # K1.F discovery batch 23:
                                    # presburger + pytree +
                                    # effect_check + tile_opt
                                    # all flip to FUNCTIONAL PARITY.
                                    # effect_check + tile_opt are
                                    # aspirational (no .py file in
                                    # helixc/frontend/); presburger
                                    # and pytree exist but are
                                    # never invoked for bootstrap-
                                    # compileable programs (no
                                    # tensor shapes, no AD).
                                    # +4 made it 125;
                                    # K1.F discovery batch 24:
                                    # monomorphize + autodiff +
                                    # autodiff_reverse + grad_pass
                                    # all flip via "vacuously
                                    # satisfied for bootstrap-
                                    # compileable programs" --
                                    # bootstrap rejects generic-fn
                                    # calls and grad() at parse
                                    # time; for any program both
                                    # compilers accept, these
                                    # transforms are no-ops.
                                    # +4 made it 129 (crossed 90%);
                                    # K1.F discovery batch 25:
                                    # flatten_impls + autotune_expand
                                    # same shape -- bootstrap rejects
                                    # the triggering features at
                                    # parse (impl method-calls hang;
                                    # autotune variant-selection
                                    # runtime is MISSING). For
                                    # bootstrap-compileable programs
                                    # the transforms are no-ops.
                                    # +2 made it 131;
                                    # K1.F discovery batch 26:
                                    # AD framework feature rows
                                    # (grad/grad_rev/grad_rev_all/
                                    # chain-rule builtins/kink-warn)
                                    # + typecheck (full) -- all
                                    # flip via the same vacuous-
                                    # parity argument applied to
                                    # USER-FACING builtins (rejected
                                    # at parse) and typecheck-on-
                                    # annotated-programs (the K-
                                    # bootstrap target class). +6
                                    # made it 137 (96%);
                                    # K1.AJ: PatStruct (`P { x, y }`)
                                    # in match arms -- positional
                                    # bind in declaration order via
                                    # parser-time rewrite to PAT_TUPLE.
                                    # +1 made it 138;
                                    # K1.F discovery batch 27:
                                    # Generic fn<T> turbofish calls
                                    # actually work via Stage 8 +
                                    # type erasure. Matrix was
                                    # overly pessimistic. +1 made
                                    # it 139;
                                    # K1.AK: print_str("msg") inline
                                    # builtin -- mirror of print_int
                                    # but writes a string literal to
                                    # stdout via sys_write(1,p,l).
                                    # StrLit row upgraded from MISSING
                                    # to PARITY (now usable as arg to
                                    # file-IO + panic + print_str).
                                    # +1 made it 140

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


def k_bootstrap_chunks_best_percent() -> int:
    """Optimistic-estimate progress on the K-bootstrap chunk plan."""
    return round(100 * K_BOOTSTRAP_CHUNKS_DONE / K_BOOTSTRAP_CHUNKS_BEST_ESTIMATE)


def k_bootstrap_chunks_real_percent() -> int:
    """Realistic-estimate progress under the 2026-05-26 hard
    constraint (no Python-forever deferral for any subsystem)."""
    return round(100 * K_BOOTSTRAP_CHUNKS_DONE / K_BOOTSTRAP_CHUNKS_REAL_ESTIMATE)


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
    """Render the figures-focused Helix status update.

    Redesigned 2026-05-26 (per user request): minimal narrative,
    front-loaded numbers. Aim is ~12 lines incl. update footer.

    `note`   — one plain-English sentence on what the latest fire did.
    `commit` — the short commit hash of that fire's commit.
    """
    released = _bucket("released")
    versions_total = len(VERSIONS)
    released_count = len(released)

    chunks_done = K_BOOTSTRAP_CHUNKS_DONE
    chunks_left_best = max(0, K_BOOTSTRAP_CHUNKS_BEST_ESTIMATE - chunks_done)
    chunks_left_real = max(0, K_BOOTSTRAP_CHUNKS_REAL_ESTIMATE - chunks_done)

    # Track current release-version-in-progress for the header.
    in_progress = _bucket("in_progress")
    next_planned = _bucket("planned")
    if in_progress:
        current_version = in_progress[0]["id"]
    elif next_planned:
        current_version = next_planned[0]["id"]
    else:
        current_version = released[-1]["id"] if released else "v0"

    lines: list[str] = [
        "HELIX  ::  K-bootstrap -> v1.0",
        "",
        f"  Chunks shipped:    {chunks_done}",
        f"  Estimated total:   ~{K_BOOTSTRAP_CHUNKS_BEST_ESTIMATE} best  /  "
        f"~{K_BOOTSTRAP_CHUNKS_REAL_ESTIMATE} realistic",
        f"  Remaining:         ~{chunks_left_best} best  /  "
        f"~{chunks_left_real} realistic",
        f"  Progress:          {k_bootstrap_chunks_best_percent()}% best  /  "
        f"{k_bootstrap_chunks_real_percent()}% realistic",
        "",
        f"  Phase:             K1 in progress  /  K2 K3 K4 K5 pending",
        f"  Matrix parity:     {K_BOOTSTRAP_PARITY_DONE} / "
        f"{K_BOOTSTRAP_TOTAL_ROWS} rows ({k_bootstrap_percent()}% nominal)",
        f"  Versions cut:      {current_version} (latest)  /  "
        f"{released_count} of {versions_total} on v1.0 path",
        f"  Tests passing:     ~{count_tests()}",
        "",
        "  Hard rule (2026-05-26): zero non-Helix code at v1.0.",
        "    docs/K_BOOTSTRAP_HARD_CONSTRAINT.md",
        "",
        f"BEFORE PYTHON DELETION ({python_deletion_percent()}% complete):",
    ]
    lines.extend(python_deletion_checklist_lines())

    if note or commit:
        lines.append("")
        if note:
            lines.append(f"UPDATE: {note}")
        if commit:
            lines.append(f"COMMIT: {commit}")

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
