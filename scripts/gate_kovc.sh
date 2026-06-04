#!/usr/bin/env bash
# GATE for a kovc.hx change (Helix v1.0). Run as a FILE under WSL. Verifies, from the
# EDITED kovc.hx, the full discipline before any commit:
#   1. SELF-HOST FIXPOINT: seed -> K1 -> K2 -> K3 -> K4, assert K2==K3==K4 byte-identical
#      (the trust spine still self-hosts; sha may differ from the old mint, that is fine).
#   2. GPU PTX REGRESSION: emit a kernel's PTX with the OLD driver, re-mint the driver from
#      the edited kovc.hx, emit again, byte-diff -- an x86-only fix must NOT change PTX.
#   3. FEATURE CORPUS: compile+run the 17-program corpus via the new K2; report the matrix.
# Overall GATE PASS = fixpoint identical AND PTX identical AND corpus has NO regressions.
set -u
BS=/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap
EX=/mnt/c/Projects/Kovostov-Native/helixc/examples
CD=/tmp/corpus
# T3/G3 (2026-06-02): kovc emits .version 8.3 (TF32 mma). This gate is ptxas-FREE
# (it cmp's emitted PTX vs committed .ref.ptx + greps provenance; the self-host
# fixpoint is pure x86), so the ptxas version does NOT affect it -- but any future
# ptxas leg MUST use the 12.8 ptxas (12.0 rejects 8.3). Pinned here for the record.
PTXAS="${PTXAS:-/usr/local/cuda/bin/ptxas}"
cd "$BS" || { echo "FATAL: no bootstrap dir"; exit 9; }
mkdir -p "$CD"
GATE_OK=1

echo "=== [0] regenerate sources from the edited kovc.hx ==="
bash assemble_k1.sh >/dev/null 2>&1 && echo "  assembled" || { echo "FATAL assemble"; exit 8; }

echo "=== [1] GPU PTX reference (committed sm_86 baseline; fallback: OLD driver) ==="
Kern=$EX/vector_add_kernel.hx
# T2/M0 (2026-06-02): the GPU PTX regression guard now anchors to a COMMITTED
# reference PTX (vector_add_kernel.ref.ptx), not the gitignored on-disk driver.
# M0 INTENTIONALLY changed the emitted PTX (.target sm_75 -> sm_86 for the sm_86
# reference box, kovc.hx:11839-11843); the new committed reference IS the sm_86
# baseline, so the guard becomes "re-minted PTX matches the committed reference."
# Reason recorded: helixc/examples/vector_add_kernel.ref.ptx (sha pinned by git;
# .gitattributes eol=lf). If a later T2 milestone intentionally changes PTX again,
# re-mint + re-commit this reference with its reason (charter 1.0 step 2).
REF=$EX/vector_add_kernel.ref.ptx
if [ -s "$REF" ]; then
  cp "$REF" /tmp/ref.ptx; echo "  ref.ptx <- committed $REF ($(stat -c%s /tmp/ref.ptx) bytes, sm_86 baseline)"
elif [ -f "$Kern" ] && [ -x ./_kovc_ptx_driver.bin ]; then
  cp "$Kern" /tmp/kernel_in.hx; rm -f /tmp/out.ptx
  timeout 30 ./_kovc_ptx_driver.bin >/dev/null 2>&1 || true
  if [ -s /tmp/out.ptx ]; then cp /tmp/out.ptx /tmp/ref.ptx; echo "  ref.ptx $(stat -c%s /tmp/ref.ptx) bytes (old on-disk driver)"; else echo "  WARN: old driver emitted no PTX"; fi
else echo "  WARN: missing committed ref, kernel, or driver -- skipping GPU ref"; fi

echo "=== [2] SELF-HOST FIXPOINT from edited kovc.hx ==="
t0=$SECONDS
timeout 400 ./seed.bin k1src.hx /tmp/K1.bin; echo "  seed->K1 rc=$? ($((SECONDS-t0))s)"
chmod +x /tmp/K1.bin; cp k1input.hx /tmp/k1_in.hx
timeout 60 /tmp/K1.bin; [ -s /tmp/k1_out.bin ] || { echo "FATAL: K2 build failed (kovc.hx may not self-compile)"; exit 7; }
cp /tmp/k1_out.bin /tmp/K2.bin; chmod +x /tmp/K2.bin
cp k1input.hx /tmp/k2_in.hx; timeout 60 /tmp/K2.bin; cp /tmp/k2_out.bin /tmp/K3.bin; chmod +x /tmp/K3.bin
timeout 60 /tmp/K3.bin; cp /tmp/k2_out.bin /tmp/K4.bin
S2=$(sha256sum /tmp/K2.bin|awk '{print $1}'); S3=$(sha256sum /tmp/K3.bin|awk '{print $1}'); S4=$(sha256sum /tmp/K4.bin|awk '{print $1}')
echo "  K2=$S2"; echo "  K3=$S3"; echo "  K4=$S4"
if [ "$S2" = "$S3" ] && [ "$S3" = "$S4" ]; then echo "  FIXPOINT OK (K2==K3==K4 byte-identical)"; else echo "  FIXPOINT FAIL"; GATE_OK=0; fi

echo "=== [3] re-mint PTX driver + GPU PTX regression ==="
t0=$SECONDS
timeout 400 ./seed.bin k1ptxdrv.hx /tmp/newdrv.bin; echo "  seed->newdrv rc=$? ($((SECONDS-t0))s)"
if [ -s /tmp/newdrv.bin ] && [ -s /tmp/ref.ptx ]; then
  chmod +x /tmp/newdrv.bin; cp "$Kern" /tmp/kernel_in.hx; rm -f /tmp/out.ptx
  timeout 30 /tmp/newdrv.bin >/dev/null 2>&1 || true
  if cmp -s /tmp/out.ptx /tmp/ref.ptx; then echo "  GPU PTX REGRESSION OK (PTX byte-identical pre/post fix)";
  else echo "  GPU PTX CHANGED -- inspect (x86-only fix should NOT alter PTX)"; GATE_OK=0; fi
else echo "  WARN: could not run PTX regression (no newdrv or ref)"; fi

# T2/M1 (2026-06-02): tiled-GEMM PTX regression. The SMEM tiled kernel is a
# committed reference (helixc/examples/tiled_matmul_kernel.ref.ptx); re-emit it from
# the freshly-minted driver and assert byte-identical to the committed reference +
# that the OUTPUT carries the expected instruction classes (provenance: grep the
# OUTPUT, never source). Anchors the new emitter the same way vector_add anchors the
# old path.
# T2/G2 (2026-06-02): the emitter INTENTIONALLY changed the tiled PTX (cp.async
# double-buffer); the reference was re-minted + re-committed with that reason (charter
# 1.0 step 2). Provenance now also requires the cp.async.cg.shared.global +
# commit_group + wait_group double-buffer signature in the OUTPUT.
TREF=$EX/tiled_matmul_kernel.ref.ptx
TKern=$EX/tiled_matmul_kernel.hx
if [ -s /tmp/newdrv.bin ] && [ -s "$TREF" ] && [ -f "$TKern" ]; then
  cp "$TKern" /tmp/kernel_in.hx; rm -f /tmp/out.ptx
  timeout 30 /tmp/newdrv.bin >/dev/null 2>&1 || true
  if [ -s /tmp/out.ptx ]; then
    cp "$TREF" /tmp/tref.ptx
    if cmp -s /tmp/out.ptx /tmp/tref.ptx; then echo "  TILED PTX REGRESSION OK (matches committed tiled_matmul_kernel.ref.ptx)";
    else echo "  TILED PTX CHANGED -- re-mint+re-commit the tiled reference with a reason"; GATE_OK=0; fi
    if grep -q '\.shared' /tmp/out.ptx && grep -q 'bar\.sync 0' /tmp/out.ptx \
       && grep -q 'cp\.async\.cg\.shared\.global' /tmp/out.ptx \
       && grep -q 'cp\.async\.commit_group' /tmp/out.ptx \
       && grep -q 'cp\.async\.wait_group' /tmp/out.ptx; then echo "  TILED PROVENANCE OK (.shared + bar.sync + cp.async double-buffer in the OUTPUT)";
    else echo "  TILED PROVENANCE FAIL (missing .shared/bar.sync/cp.async in emitted PTX)"; GATE_OK=0; fi
  else echo "  WARN: tiled kernel emitted no PTX"; GATE_OK=0; fi
else echo "  WARN: could not run tiled PTX regression (no newdrv/ref/kernel)"; fi

echo "=== [4] FEATURE CORPUS via new K2 ==="
gen() { cat > "$CD/$1"; }
gen i64_basic.hx <<'EOF'
fn main() -> i32 { let x: i64 = 42_i64; x as i32 }
EOF
gen i64_mul_beyond.hx <<'EOF'
fn main() -> i32 { let a: i64 = 2000000000_i64; let b: i64 = 3_i64; let c: i64 = a * b; let g: i64 = 1000000000_i64; (c / g) as i32 }
EOF
gen i64_div_beyond.hx <<'EOF'
fn main() -> i32 { let a: i64 = 2000000000_i64; let b: i64 = 2_i64; let big: i64 = a * b; let g: i64 = 80000000_i64; (big / g) as i32 }
EOF
gen i64_cmp.hx <<'EOF'
fn main() -> i32 { let a: i64 = 5_000_000_000_i64; let b: i64 = 4_000_000_000_i64; if a > b { 1 } else { 0 } }
EOF
gen i64_neg.hx <<'EOF'
fn main() -> i32 { let a: i64 = 100_i64; let b: i64 = -a + 105_i64; b as i32 }
EOF
gen u64_shr.hx <<'EOF'
fn shr_u64(x: u64) -> u64 { x >> 63_u64 }
fn main() -> i32 { let x: u64 = 1_u64 << 63_u64; shr_u64(x) as i32 }
EOF
gen u8_wrap.hx <<'EOF'
fn main() -> i32 { let x: u8 = 0_u8 - 1_u8; let y: i32 = x as i32; if y == 255 { 42 } else { 7 } }
EOF
gen u16_wrap.hx <<'EOF'
fn main() -> i32 { let x: u16 = 0_u16 - 1_u16; let y: i32 = x as i32; if y == 65535 { 42 } else { 7 } }
EOF
gen i16_ovf.hx <<'EOF'
fn main() -> i32 { let x: i16 = 32767_i16 + 1_i16; let y: i32 = x as i32; if y < 0 { 42 } else { 7 } }
EOF
gen result_inline.hx <<'EOF'
enum Result { Ok(i32), Err(i32) }
fn main() -> i32 { let r = Result::Ok(42); match r { Result::Ok(x) => x, Result::Err(e) => e } }
EOF
gen assoc_sub.hx <<'EOF'
fn main() -> i32 { 10 - 3 - 2 }
EOF
gen assoc_div.hx <<'EOF'
fn main() -> i32 { 100 / 5 / 2 }
EOF
gen cmp_ne.hx <<'EOF'
fn main() -> i32 { if 5 != 3 { 1 } else { 0 } }
EOF
gen cmp_ge.hx <<'EOF'
fn main() -> i32 { if 5 >= 5 { 1 } else { 0 } }
EOF
gen cmp_le.hx <<'EOF'
fn main() -> i32 { if 3 <= 5 { 1 } else { 0 } }
EOF
gen bit_andor.hx <<'EOF'
fn main() -> i32 { (12 & 10) | 1 }
EOF
gen bit_xor.hx <<'EOF'
fn main() -> i32 { 255 ^ 15 }
EOF
gen bit_shl.hx <<'EOF'
fn main() -> i32 { 1 << 4 }
EOF
gen arr_idx.hx <<'EOF'
fn main() -> i32 { let a = [10, 20, 30]; a[1] }
EOF
gen while_sum.hx <<'EOF'
fn main() -> i32 { let mut s = 0; let mut i = 0; while i < 5 { s = s + i; i = i + 1; } s }
EOF
gen while_break.hx <<'EOF'
fn main() -> i32 { let mut i = 0; while i < 100 { i = i + 1; if i >= 7 { break; } } i }
EOF
gen f64_add.hx <<'EOF'
fn main() -> i32 { let a: f64 = 1.5_f64; let b: f64 = 2.5_f64; (a + b) as i32 }
EOF
gen f64_mul.hx <<'EOF'
fn main() -> i32 { let a: f64 = 3.0_f64; let b: f64 = 4.0_f64; (a * b) as i32 }
EOF
gen tuple2.hx <<'EOF'
fn main() -> i32 { let t = (3, 4); t.0 + t.1 }
EOF
gen impl_method.hx <<'EOF'
struct P { x: i32 }
impl P { fn get(self) -> i32 { self.x } }
fn main() -> i32 { let p = P { x: 42 }; p.get() }
EOF
gen match_or.hx <<'EOF'
fn main() -> i32 { let x = 2; match x { 1 | 2 | 3 => 10, _ => 0 } }
EOF
gen match_range.hx <<'EOF'
fn main() -> i32 { let x = 5; match x { 1..10 => 1, _ => 0 } }
EOF
gen vec_arena.hx <<'EOF'
fn vec_new() -> i32 { __arena_push(0) }
fn vec_push(v: i32, x: i32) -> i32 { let n = __arena_get(v); __arena_set(v, n + 1); __arena_push(x) }
fn vec_len(v: i32) -> i32 { __arena_get(v) }
fn vec_get(v: i32, i: i32) -> i32 { __arena_get(v + 1 + i) }
fn main() -> i32 {
    let v = vec_new();
    vec_push(v, 10);
    vec_push(v, 20);
    vec_push(v, 12);
    vec_get(v, 0) + vec_get(v, 1) + vec_get(v, 2) + vec_len(v)
}
EOF
# T3 >6-arg SysV stack-pass corpus (2026-06-02): no prior Helix fn had >6 params;
# kovc dropped params 7+ (callee bound only rdi..r9, args 7+ trapped ud2 -> SIGILL
# rc 132). These exercise the new caller stack-pass + callee [rbp+16+8*(i-6)] binding.
# f8: 1..8 -> 36 ; f9: 1..9 -> 45 ; f11: 1..11 -> 66 (all < 256 so the exit-code
# byte holds the full sum). Each arg distinct so a dropped/clobbered arg shows.
gen f8_args.hx <<'EOF'
fn f8(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32, h: i32) -> i32 { a+b+c+d+e+f+g+h }
fn main() -> i32 { f8(1,2,3,4,5,6,7,8) }
EOF
gen f9_args.hx <<'EOF'
fn f9(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32, h: i32, i: i32) -> i32 { a+b+c+d+e+f+g+h+i }
fn main() -> i32 { f9(1,2,3,4,5,6,7,8,9) }
EOF
gen f11_args.hx <<'EOF'
fn f11(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32, h: i32, i: i32, j: i32, k: i32) -> i32 { a+b+c+d+e+f+g+h+i+j+k }
fn main() -> i32 { f11(1,2,3,4,5,6,7,8,9,10,11) }
EOF
pass=0; fail=0
chk() { local f="$1" exp="$2" b; b=$(basename "$1")
  [ -f "$f" ] || { echo "  MISSING $b"; fail=$((fail+1)); return; }
  cp "$f" /tmp/k2_in.hx; rm -f /tmp/k2_out.bin; timeout 30 /tmp/K2.bin >/dev/null 2>&1
  [ -s /tmp/k2_out.bin ] || { echo "  COMPILE-FAIL $b"; fail=$((fail+1)); return; }
  chmod +x /tmp/k2_out.bin; timeout 10 /tmp/k2_out.bin; local rc=$?
  if [ "$rc" = "$exp" ]; then echo "  PASS $b ($rc)"; pass=$((pass+1)); else echo "  FAIL $b ($rc!=$exp)"; fail=$((fail+1)); fi
}
chk "$EX/exit42.hx" 42; chk "$EX/matmul_2x2.hx" 69; chk "$EX/hbs_sample_enum_struct.hx" 129
chk "$EX/hbs_sample_option.hx" 42; chk "$EX/hbs_sample_recursion.hx" 120
chk "$EX/dogfood_18_pat_struct_showcase.hx" 42; chk "$CD/result_inline.hx" 42; chk "$EX/gradient_descent.hx" 42
chk "$CD/i64_basic.hx" 42; chk "$CD/i64_mul_beyond.hx" 6; chk "$CD/i64_div_beyond.hx" 50; chk "$CD/i64_cmp.hx" 1; chk "$CD/i64_neg.hx" 5
chk "$CD/u64_shr.hx" 1; chk "$CD/u8_wrap.hx" 42; chk "$CD/u16_wrap.hx" 42; chk "$CD/i16_ovf.hx" 42
chk "$CD/assoc_sub.hx" 5; chk "$CD/assoc_div.hx" 10; chk "$CD/cmp_ne.hx" 1; chk "$CD/cmp_ge.hx" 1; chk "$CD/cmp_le.hx" 1
chk "$CD/bit_andor.hx" 9; chk "$CD/bit_xor.hx" 240; chk "$CD/bit_shl.hx" 16; chk "$CD/arr_idx.hx" 20; chk "$CD/while_sum.hx" 10; chk "$CD/while_break.hx" 7
chk "$CD/f64_add.hx" 4; chk "$CD/f64_mul.hx" 12; chk "$CD/tuple2.hx" 7; chk "$CD/impl_method.hx" 42; chk "$CD/match_or.hx" 10; chk "$CD/match_range.hx" 1; chk "$CD/vec_arena.hx" 45
# H2 generics corpus (2026-06-01): the charter generics items run on the self-hosted compiler.
GENC=$BS/corpus_gen
chk "$GENC/gen_impl_t_single_f32.hx" 5; chk "$GENC/gen_impl_angle_i32.hx" 5; chk "$GENC/gen_impl_ret_f32.hx" 5; chk "$GENC/gen_concrete_on_mono.hx" 7
chk "$GENC/gen_pair_multi.hx" 12; chk "$GENC/gen_vec_i32.hx" 42; chk "$GENC/gen_vec_f32.hx" 5; chk "$GENC/e6_bare_match.hx" 42
# H3 traits + closures corpus (2026-06-01): trait-method dispatch + closure capture codegen.
chk "$GENC/t2_trait_impl.hx" 42; chk "$GENC/t3_closure_call.hx" 42; chk "$GENC/t4_closure_capture.hx" 42; chk "$GENC/t8_closure_two_caps.hx" 42
chk "$GENC/t7_trait_poly.hx" 42; chk "$GENC/t7b_trait_2types.hx" 42; chk "$GENC/t7c_difffields.hx" 42
# H4 pattern guards corpus (2026-06-01): match arm `if cond` guard is now evaluated.
chk "$GENC/g1_guard_true.hx" 1; chk "$GENC/g2_guard_false.hx" 0; chk "$GENC/g3_guard_chain.hx" 2
# H5 i64-literal widening (2026-06-02): codegen decodes the full 64-bit literal from its text ref
# (mirror f64 tag-34 path) via i32-multi-word 16-bit limbs -- no longer truncates at 2^31. Full range
# incl >= 2^32: L2 = 5_000_000_000_i64 (> 2^32) / 1e8 = 50.
chk "$GENC/L1_i64_big.hx" 30; chk "$GENC/L2_i64_bigger.hx" 50; chk "$GENC/L3_i64_just_over.hx" 22
# T3 >6-arg SysV stack-pass (2026-06-02): params beyond the 6th go on the stack.
chk "$CD/f8_args.hx" 36; chk "$CD/f9_args.hx" 45; chk "$CD/f11_args.hx" 66
# T3 L-1 index-STORE hardening (2026-06-03, charter §1.6 LOW): `arr[i] = e` promoted
# [impl]->[proven]. Runtime-computed index+value through the mutable-array store path
# (emit_index_store_cpu, kovc.hx:6896, AST_INDEX_STORE tag 55), incl a same-slot
# overwrite; arr_idx above only READS. -> 42.
chk "$GENC/L1_index_store.hx" 42
# T3 L-7 dark-arm sweep (2026-06-03, charter §1.6 LOW): [impl]->[proven] for the
# FROZEN-denominator arms whose codegen ran dynamically but had no gated row. Each probe
# loop-accumulates a RUNTIME value so the arm cannot be constant-folded, then exits 42.
# Frozen L-7 items covered: `~e`/`!e` unary (AST_NEG 9 / AST_BNOT 26 / AST_NOT 27) +
# `i8`/`u32` width arms (AST_INTLIT_I8 39 / AST_INTLIT_U32 36 -- u8/u16/i16 were already
# gated, i8/u32 were not). No kovc.hx change (codegen pre-existed as [impl]).
chk "$GENC/arm_neg.hx" 42; chk "$GENC/arm_bnot.hx" 42; chk "$GENC/arm_not.hx" 42
chk "$GENC/arm_i8_width.hx" 42; chk "$GENC/arm_u32_width.hx" 42
# T3 L-7 REMAINING frozen-arm sweep (2026-06-03, charter §1.6 LOW -- completes L-7):
# the LAST frozen-denominator arms, all probed [impl]->[proven] via the K2 9cc8f20b
# fixpoint (NO kovc.hx/lexer/parser change -> the sha stays byte-identical). Each
# loop/runtime-derives 42 so nothing constant-folds; the bf16/f16 ARITH row is a
# doc-as-bound neg test (compiles, then SIGILLs 132 -- bf16/f16 are storage-only,
# no x86 baseline hardware arithmetic; fails closed, never silent-wrong).
#   arm_block_comment : nested `/* /* */ */` block comments (lexer skip_block_comment).
#   arm_radix_lits    : hex 0x2A / 0xFF_FF, binary 0b10_1010, octal 0o52 -- WITH `_` seps.
#   arm_char_lit      : char literals as int values ('*'=42, '\n' escape, '0').
#   arm_continue      : `continue` skipping iterations in a while loop.
#   arm_early_return  : early `return` from inside an if / out of a loop (before the tail).
#   arm_tuple_struct  : tuple struct `struct Pair(i32,i32)` + positional `.0`/`.1` access.
#   arm_bf16_f16_decl : bf16/f16 LITERAL declaration (truncated bit pattern emitted) -> 42.
#   arm_bf16_arith_bound : bf16/f16 ARITHMETIC bound -- compiles + traps (SIGILL 132).
chk "$GENC/arm_block_comment.hx" 42; chk "$GENC/arm_radix_lits.hx" 42
chk "$GENC/arm_char_lit.hx" 42; chk "$GENC/arm_continue.hx" 42
chk "$GENC/arm_early_return.hx" 42; chk "$GENC/arm_tuple_struct.hx" 42
chk "$GENC/arm_bf16_f16_decl.hx" 42; chk "$GENC/arm_bf16_arith_bound.hx" 132
# v1.3 V1 (P0, 2026-06-03) -- the silent-bug FIX (charter §1 V1). The v1.2 M-3
# bound (an i64/u64 wide struct-field READ silently truncated to low-32; an f64
# wide field failed closed SIGILL 132) is now CLOSED. Root cause: the field READ
# (AST_TUPLE_FIELD) only emitted an 8-byte REX.W load when p3==1, which the parser
# set ONLY for nested-struct (pointer) fields -- a f64/i64/u64 SCALAR field encoded
# struct_idx == -1 (indistinguishable from i32) so it took the 4-byte path. The fix:
# parse_struct_decl (wide_scalar_field_enc) encodes an 8-byte scalar field as
# 0-(100+tag); the read site decodes 100+tag into AST_TUPLE_FIELD.p3, codegen emits
# a REX.W 8-byte load (p3 >= 100), and expr_type returns p3-100 (the real tag) so
# f64 fields type f64 (SSE arith) and i64/u64 type as 8-byte ints. Field WRITE
# (AST_FIELD_STORE) also REX.W-stores wide fields. The M3_wide_field_bound negative
# test is RETIRED (its bound is closed). This is the i64-first commit: V1_i64 reads
# a runtime-derived field holding 5_000_000_000 (> 2^32); 5e9/1e8 == 50 EXACT (the
# pre-fix low-32 truncation gave 7). parser.hx + kovc.hx changed -> the fixpoint sha
# MOVES (the self-host source uses wide fields) but K2==K3==K4 stay byte-identical.
chk "$GENC/V1_i64_wide_field.hx" 50
# T3 §1.6 PARSER-DESUGAR promotions (2026-06-03, charter §1.6 MED/LOW): three desugars
# that were already implemented in parser.hx but had no gated corpus row. The self-host
# source uses plain `while` / `x = x + ...` / nested `if`, NEVER the new syntax, so these
# promotions keep the fixpoint byte-identical (b7e741c0) -- exercised only here through K2.
#   M-1 `for` loops (parse_for, parser.hx:16017): exclusive/inclusive/var-bound ranges -> 42.
#   M-2 compound assign `op=` (K1.U/K1.AN, parser.hx:7786): all 10 operators +=..>>= -> 42.
#   L-4 `&&`/`||` short-circuit (K1.M-fix, parser.hx:2267): desugars to AST_IF (branch ->
#       untaken arm skipped). PROVES the RHS side effect is NOT evaluated when the LHS
#       decides, via arena-slot side-effect channels (skipped slots stay 0, run slots set 1) -> 42.
chk "$GENC/M1_for_loop.hx" 42; chk "$GENC/M2_compound_assign.hx" 42; chk "$GENC/L4_short_circuit.hx" 42
# T3 §1.6 DOCUMENT-AS-BOUND negative/bound-proving rows (2026-06-03): each LOCKS a real,
# reproduced v1.2 limitation -- the compiler ACCEPTS (or mis-types) code that Rust rejects.
#   M-5 bare non-i32 scalar generic: `id(3.0_f32)` defaults T->i32 -> 0 (NOT 3); the
#       supported idiom is explicit turbofish id::<f32> (->3) / add2::<f32> (->5). Exit 0.
#   M-7 module privacy: a private (non-pub) `secret::hidden()` is ACCEPTED and RUNS (no
#       privacy enforcement) -> 42; Rust rejects (error[E0603] private). Exit 42 proves it.
#   L-3 match-exhaustiveness: a payload-enum match omitting the Err arm is ACCEPTED and runs
#       the covered Ok arm -> 42; Rust rejects (non-exhaustive patterns). Exit 42 proves it.
chk "$GENC/M5_bare_generic_bound.hx" 0; chk "$GENC/M7_privacy_bound.hx" 42; chk "$GENC/L3_nonexhaustive_bound.hx" 42
# T3 H-1 PACKAGED GENERIC COLLECTIONS (2026-06-03, charter §1.6 HIGH): generic
# Vec<T> (new/push/get/set/len/pop with GROWTH on push) + an i32->i32
# open-addressing HashMap (insert/get/contains, collision resolved by linear
# probing). Library = stdlib/collections.hx; these two corpus programs inline
# it (no external-module loader) and exercise every op end-to-end:
#   H1_vec: cap 2 -> push 1..8 (forces TWO relocations 2->4->8) -> assert
#     len==8 & cap==8 -> sum-back via get (=36) -> set(0,7) -> pop()==8 ->
#     live-sum 34 + popped 8 = 42. Proves growth + copy + set + pop + len.
#   H1_hashmap: cap 8; insert keys 3,11,19 (ALL hash to bucket 3 -- a forced
#     COLLISION resolved by linear probing into 3,4,5) + key 6 + overwrite 6 ->
#     count==4 -> get each collided key back (10,20,5,7) -> miss on absent 99 ->
#     contains present/absent/collided-but-absent(27) -> value sum 42.
# Library-level, NO kovc.hx change -> fixpoint byte-identical (verified via the
# fast inner loop: K2 sha == the H-3 mint bdff0049...).
chk "$GENC/H1_vec.hx" 42; chk "$GENC/H1_hashmap.hx" 42
# T3 H-2 RICH STRING (2026-06-03, charter §1.6 HIGH): an arena-backed String
# (str_new/str_push_byte/str_len/str_byte_at/str_concat/str_eq with GROWTH on
# push). Library = stdlib/string.hx; this corpus program inlines it (no
# external-module loader) and exercises the full round-trip end-to-end:
#   H2_string: build "Hel"+"lix" byte-by-byte -> str_concat -> "Hellix" (the
#     concat result starts cap 4, receives 6 bytes -> forces a grow 4->8) ->
#     index byte[0]='H'(72) & byte[5]='x'(120) back out -> str_eq EQUAL (vs an
#     independently-built "Hellix"), UNEQUAL-same-length ("Hel" vs "lix"),
#     UNEQUAL-diff-length (short-circuit), and a one-byte-diff ("Helliy") ->
#     exit = str_len("Hellix") * 7 = 42 (runtime-derived). A string LITERAL as
#     a value lowers to mov eax,0, so every byte/eq/concat op runs at RUNTIME
#     over arena bytes (nothing folds). Library-level, NO kovc.hx change ->
#     fixpoint byte-identical (verified via the fast inner loop: K2 sha == the
#     H-3 mint bdff0049...).
chk "$GENC/H2_string.hx" 42
# T3 H-4 TRAIT DEFAULT METHODS (2026-06-03, charter §1.6 HIGH -- the last HIGH
# item): a trait may declare a method WITH a default body; a type that impls the
# trait but does NOT override that method dispatches to the default, while a type
# that DOES override it uses its own. Implemented in parser.hx: parse_trait_decl
# now STORES each default-bodied method's `fn`-token (tdef_tab) instead of
# brace-skipping it; parse_impl_block SYNTHESIZES a concrete `<Target>__<method>`
# by re-parsing that token range as an impl method of the impl's target type
# (so `self.field`/`self.method()` resolve against the concrete type) -- UNLESS
# the impl provides an explicit override (override wins). The self-host source
# uses traits (signature-only methods + explicit impls), so the fixpoint sha
# MOVES (parser.hx changed) but K2==K3==K4 stay byte-identical; the existing
# trait corpus (t2/t7/t7b/t7c) must not regress.
#   t1_trait_default     DEFAULT-USED: trait Greet { fn hello(self)->i32 {42} } +
#     `impl Greet for P {}` (empty) -> p.hello() dispatches to the default -> 42.
#   t5_trait_default_mix DEFAULT + OVERRIDE: A uses the default hello()=10 ; B
#     OVERRIDES hello()=32 -> a.hello()+b.hello() = 42 (proves override beats the
#     default -- B's 32, not the default 10).
chk "$GENC/t1_trait_default.hx" 42; chk "$GENC/t5_trait_default_mix.hx" 42
# T3 §1.6 AGGREGATE-RETURN-BY-VALUE fix (2026-06-03): returning a struct OR
# enum BY VALUE from a fn previously mis-lowered (the callee returned a
# pointer into its own about-to-be-reclaimed frame, AND the caller stored
# that pointer with a 32-bit truncating mov) -> SIGSEGV(139) for structs,
# SIGILL(132) for the enum-then-match case (arm_enum_payload3). The fix
# (kovc.hx) copies the aggregate run into the CALLER's frame the instant the
# call returns + 64-bit-stores the pointer, and (parser.hx) tags enum
# returns pointer-rep (100+8+enum_idx) so the let-store/match drive the
# pointer path. Closes the L-7 arm_enum_payload3 v-next item + the H-2
# struct-return gap. SELF-HOST SOURCE RETURNS ONLY SCALARS -> this new path
# is never exercised during self-compilation -> fixpoint byte-identical.
# Each program is runtime-derived (loop-accumulated inputs) so nothing
# constant-folds; struct cases cover the 1-field / 16-byte / >16-byte / 40-
# byte size classes + read EVERY field; enum cases match disc + extract
# payload through the copied [disc,payload] run.
chk "$GENC/sret_1field.hx" 42; chk "$GENC/sret_2field.hx" 42
chk "$GENC/sret_3field.hx" 42; chk "$GENC/sret_5field.hx" 42
# arm_enum_payload3: PROMOTED v-next -> gated. 3-variant payload enum
# returned by value from pick() then matched (runtime variant C) -> 42.
chk "$GENC/arm_enum_payload3.hx" 42
# eret_option: 2-variant Option-shape enum returned by value + matched
# (runtime Some(42)) -> 42.
chk "$GENC/eret_option.hx" 42
# T3 §1.6 M-4 TURBOFISH-ON-ENUM-CONSTRUCTOR (2026-06-03, charter §1.6 MED):
# `Opt::<i32>::Some(payload)` / `Opt::<i32>::None` (turbofish on an enum
# constructor) now construct correctly. Pre-fix the form HUNG the compiler
# (mis-routed to the generic-fn turbofish branch, which looped scanning for
# `(args)` after `>` and found `::` -> rc 124 compile timeout; the bare
# `Opt::Some(42)` form already worked). The fix (helixc/bootstrap/parser.hx,
# parse_primary) detects `EnumName::<T>::Variant` when the leading IDENT is a
# registered enum + masks the generic-fn turbofish flag + skips the `::<T>`
# type-arg segment so construction routes to the SAME AST_TUPLE_LIT enum-
# construct path as the bare form (type-erased -- the monomorph carries no
# runtime tag). parser.hx changed, so the fixpoint sha MOVES off 6dbddad8;
# K2==K3==K4 stay byte-identical (the self-host source never uses turbofish
# enum ctors). M4_turbofish_enum: payload Some(40+k) extracted via match (42)
# + unit None turbofish selecting the None arm; runtime-derived so nothing
# folds. gen_option_i32: the charter probe -- turbofish Some matched to 42.
chk "$GENC/M4_turbofish_enum.hx" 42; chk "$GENC/gen_option_i32.hx" 42
# T3 §1.6 M-6 CLOSURE-AS-ARGUMENT (2026-06-03, charter §1.6 MED): passing a
# (non-capturing) closure AS AN ARGUMENT to a higher-order fn that INVOKES it
# now works. Pre-fix the closure literal in arg position lowered to AST_INT(0)
# (the fn pointer was lost) -> the callee did `call *0` -> SIGSEGV(139). The
# fix (helixc/bootstrap/parser.hx, parse_closure_lit) returns
# AST_VAR(__closure_<id>) for a no-capture closure: codegen A2a emits
# `lea rax,[rip+__closure_<id>]` (a real fn pointer), which flows as the arg,
# and the callee invokes it indirectly via A2b (`call r11` through the bound
# param). Capturing closures keep AST_INT(0) (the by-name call path injects
# their env positionally; passing a capturing closure by value is a documented
# v1.2 bound). parser.hx changed -> fixpoint sha MOVES off 0f846aea; K2==K3==K4
# stay byte-identical (the self-host source has no closures-as-args).
#   M6_closure_arg: apply1/twice/apply2/choose -- a closure passed as an arg &
#     invoked (incl the SAME closure twice, two invocations, and TWO distinct
#     closure args); runtime-derived -> 42.
#   t6_closure_arg: the charter probe apply(|y| y+1, 41) (pre-fix SIGSEGV) -> 42.
#   M6_capture_regression: a CAPTURING closure called by name still works
#     (the fix must not disturb the capture path) -> 42.
chk "$GENC/M6_closure_arg.hx" 42; chk "$GENC/t6_closure_arg.hx" 42
chk "$GENC/M6_capture_regression.hx" 42
echo "  CORPUS: $pass passed, $fail failed (expect 96 pass: 35 v1.0 + 8 H2 generics + 7 H3 traits/closures + 3 H4 pattern-guards + 3 H5 i64-literals [3e9->30, 5e9->50 (> 2^32), 2.2e9->22 -- full i64 range, no truncation] + 3 T3 >6-arg [f8->36, f9->45, f11->66] + 1 T3 L-1 index-store [L1_index_store->42] + 5 T3 L-7 dark-arms [neg/bnot/not/i8/u32 ->42] + 3 T3 desugars [M-1 for / M-2 op= / L-4 &&|| ->42] + 3 T3 doc-as-bound [M-5 bare-generic ->0, M-7 privacy ->42, L-3 non-exhaustive ->42] + 2 T3 H-1 collections [H1_vec growth->42, H1_hashmap collision->42] + 1 T3 H-2 rich String [H2_string concat+eq+byte_at->42] + 6 T3 §1.6 aggregate-return-by-value [sret 1/2/3/5-field->42, arm_enum_payload3->42, eret_option->42] + 2 T3 H-4 trait-defaults [t1 default-used->42, t5 default/override-mix->42] + 2 T3 M-4 turbofish-enum-ctor [M4_turbofish_enum payload+unit->42, gen_option_i32 turbofish-match->42] + 3 T3 M-6 closure-as-arg [M6_closure_arg multi-form->42, t6_closure_arg charter-probe->42, M6_capture_regression capturing-by-name->42] + 8 T3 L-7 REMAINING frozen-arm sweep [block-comment/radix+_/char-lit/continue/early-return/tuple-struct/bf16-f16-decl ->42, bf16-arith-bound ->132] + 1 v1.3 V1 i64 wide struct field [V1_i64_wide_field 5e9/1e8->50 EXACT, the silent-bug fix; M-3 bound RETIRED])"

echo "=== [4b] CHECK_ERR negative corpus (H-3 file:line:col diagnostics) ==="
# H-3 (charter §1.6): a malformed program must produce a COMPILE-TIME non-zero
# exit AND a `path:line:col: message` diagnostic with the CORRECT line+col of the
# offending token (not a bare byte offset / runtime trap). The fresh K2 compiles
# each err fixture (reading /tmp/k2_in.hx), so the reported path is /tmp/k2_in.hx.
# chk_err asserts: (1) K2's OWN exit is non-zero, (2) it did NOT write an output
# ELF, (3) its stdout/stderr is EXACTLY `/tmp/k2_in.hx:<line>:<col>: parse error:
# unexpected token` with the line:col we computed by hand from each fixture's bytes.
# A clean program emits NO diagnostic (its AST has no AST_ERR node), so the
# self-host fixpoint + the 71-corpus above are unperturbed.
epass=0; efail=0
chk_err() { # <fixture> <expected_line> <expected_col>
  local f="$1" el="$2" ec="$3" b; b=$(basename "$1")
  [ -f "$f" ] || { echo "  EMISSING $b"; efail=$((efail+1)); return; }
  cp "$f" /tmp/k2_in.hx; rm -f /tmp/k2_out.bin
  local out rc want; out=$(timeout 20 /tmp/K2.bin 2>&1); rc=$?
  want="/tmp/k2_in.hx:${el}:${ec}: parse error: unexpected token"
  if [ "$rc" = "0" ]; then echo "  EFAIL $b (compiler exited 0 on a parse error)"; efail=$((efail+1)); return; fi
  if [ -s /tmp/k2_out.bin ]; then echo "  EFAIL $b (wrote an output ELF despite the error)"; efail=$((efail+1)); return; fi
  if [ "$out" = "$want" ]; then echo "  EPASS $b -> '$out' (exit $rc)"; epass=$((epass+1));
  else echo "  EFAIL $b: got '$out' want '$want' (exit $rc)"; efail=$((efail+1)); fi
}
# Hand-computed line:col of the offending '@' token in each fixture (1-based):
#   err_at_l1.hx:        `fn main() -> i32 { @ }`  -> '@' at byte 19 -> 1:20
#   err_let_rhs.hx:      `fn main() -> i32 { let x = @; x }` -> 1:28
#   err_multiline_l3.hx: '@' is the let-RHS on line 3, col 13 -> 3:13
#   err_after_op_l2.hx:  '@' after `1 + ` on line 2, col 9   -> 2:9
chk_err "$GENC/err_at_l1.hx" 1 20
chk_err "$GENC/err_let_rhs.hx" 1 28
chk_err "$GENC/err_multiline_l3.hx" 3 13
chk_err "$GENC/err_after_op_l2.hx" 2 9
# T3 §1.6 L-2 (2026-06-03) DOCUMENT-AS-BOUND + negative test: a u64 LITERAL whose
# decimal magnitude exceeds 2^32-1 is lex-capped (i32 accumulator) and FAILS CLOSED
# -- lexer tags it token 40 (no parser arm, lexer.hx:580-617 + check_u64_10digit_
# overflow), the parser unexpected-token catch-all makes an AST_ERR, and the H-3
# diagnostic prints `<path>:line:col: parse error: unexpected token` + non-zero exit
# + NO ELF. `5_000_000_000_u64` is the offending token at line 25 col 20 of the
# fixture. (i64 >= 2^32 literals work via the limb path -- i64_cmp/L2_i64_bigger;
# u64 >= 2^32 is reachable by COMPUTATION -- u64_shr; only the u64 *literal* is
# capped. Lex-accumulator widening = v-next.) Proves the bound never silent-wrongs.
chk_err "$GENC/L2_u64_over_2p32.hx" 25 20
echo "  CHECK_ERR: $epass passed, $efail failed (expect 5: file:line:col correct + non-zero exit + no ELF)"

echo "=== GATE VERDICT ==="
# H-3 (2026-06-03): the check_err negative corpus must be all-green (correct
# path:line:col + non-zero compile exit). Any miss fails the gate.
if [ "$efail" -ne 0 ] || [ "$epass" -lt 5 ]; then echo "  CHECK_ERR REGRESSION (epass=$epass efail=$efail; want 5/0)"; GATE_OK=0; fi
# regression guard: the u64_shr must now PASS, and we must not drop below the full corpus count.
# T3 (2026-06-02): bumped 56 -> 59 for the 3 new >6-arg SysV stack-pass cases.
# T3 (2026-06-03): bumped 59 -> 60 for the L-1 index-store program.
# T3 (2026-06-03): bumped 60 -> 65 for the 5 L-7 dark-arm rows (neg/bnot/not/i8/u32).
# T3 (2026-06-03): bumped 65 -> 71 for 3 desugar promotions (M-1 for / M-2 op= / L-4 &&||)
#                  + 3 doc-as-bound bound-provers (M-5 bare-generic / M-7 privacy / L-3 non-exhaustive).
# T3 (2026-06-03): bumped 71 -> 73 for H-1 packaged collections (H1_vec growth + H1_hashmap collision).
# T3 (2026-06-03): bumped 73 -> 74 for H-2 rich String (H2_string: concat + eq + byte_at round-trip).
# T3 (2026-06-03): bumped 74 -> 80 for the §1.6 aggregate-return-by-value fix
#   (sret 1/2/3/5-field structs + arm_enum_payload3 [PROMOTED v-next->gated] + eret_option).
# T3 (2026-06-03): bumped 80 -> 82 for H-4 trait DEFAULT methods (t1 default-used
#   + t5 default/override mix -- the last HIGH §1.6 item).
# T3 (2026-06-03): bumped 82 -> 84 for M-4 turbofish-on-enum-constructor
#   (M4_turbofish_enum payload+unit -> 42, gen_option_i32 turbofish-match -> 42).
# T3 (2026-06-03): bumped 84 -> 87 for M-6 closure-as-argument (M6_closure_arg
#   multi-form -> 42, t6_closure_arg charter probe -> 42, M6_capture_regression
#   capturing-by-name -> 42).
# T3 (2026-06-03): bumped 87 -> 95 for the L-7 REMAINING frozen-arm sweep (8 rows:
#   block-comment / radix+_ / char-lit / continue / early-return / tuple-struct /
#   bf16-f16-decl ->42 + bf16-arith-bound ->132). No kovc.hx change (fixpoint stays
#   byte-identical 9cc8f20b) -- pure promotions of already-implemented arms.
# T3 (2026-06-03): bumped 95 -> 96 for M-3 8-byte-struct-field DOCUMENT-AS-BOUND
#   (M3_wide_field_bound ->132, f64 wide-field read fails closed). No source change.
# v1.3 V1 (2026-06-03): the M-3 bound is now CLOSED (the silent-bug fix). Count stays
#   96: -1 (M3_wide_field_bound bound test RETIRED) +1 (V1_i64_wide_field ->50, the
#   i64-first commit). The u64/f64/multi V1 tests land in the follow-up commit (98).
if [ "$pass" -lt 96 ]; then echo "  CORPUS REGRESSION (pass=$pass < 96)"; GATE_OK=0; fi
if [ "$GATE_OK" = "1" ]; then echo "GATE_PASS"; else echo "GATE_FAIL"; fi
# H-3 (2026-06-03): exit reflects the verdict so the detached runner's
# exit-code check (detached_gate.sh) reports RED on ANY gate failure
# (corpus/check_err/fixpoint/PTX) -- previously the trailing echo masked
# GATE_FAIL as exit 0.
if [ "$GATE_OK" = "1" ]; then exit 0; else exit 1; fi
