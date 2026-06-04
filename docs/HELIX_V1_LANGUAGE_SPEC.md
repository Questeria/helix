# Helix Language Reference — v1.0 frozen surface + v1.3 type/trust deltas

**Status: v1.0 surface FROZEN (2026-06-01); v1.3 type-completeness deltas folded in
(2026-06-04, §9).** This documents the Helix language **as actually implemented by the
self-hosted compiler `kovc`** (helixc/bootstrap/{lexer,parser,kovc}.hx), the only compiler
after K4 (the Python reference was deleted). The v1.0 scope decisions (§7) are **RESOLVED**
(`Ok`/`Err`/`Result` = user-defined). The v1.0 language *surface* is committed (**no
breaking changes after v1.0**). What v1.3 changed is **not surface but depth**: features the
v1.0/v1.1/v1.2 spec marked as *erased / fail-closed-bounded* are now **first-class +
gated** — the type-correctness items V1–V4 of `docs/HELIX_V1_3.md`. Those promotions are
recorded inline below (search "v1.3") and summarized in **§9**. They make this spec *more
honest* (fewer caveats), they do not break v1.0 programs. (`docs/lang/spec.md` is a separate
**v0.1 design-vision draft** — superseded for implementation-status purposes by this file;
it describes design-target syntax against the deleted Python frontend and is NOT the
authoritative as-built reference.)

**Honesty legend** — every feature is marked:
- **[proven]** — exercised + passing in the 35-program feature corpus (`scripts/feature_corpus.sh`), compiled+run on the self-hosted compiler.
- **[impl]** — implemented in `kovc` codegen, not in the sample corpus (works, but not yet corpus-proven).
- **[erased]** — *parsed* but type-erased / not enforced / no real codegen (accepts the syntax, does NOT give the semantics).
- **[unsupported]** — no syntax / not implemented.

Target: **x86-64 Linux** (static, syscall-only ELF) for CPU; **NVIDIA PTX** for `@kernel` GPU functions.

---

## 1. Lexical structure

- **Comments**: `//` line [proven]; `/* … */` nested block [impl].
- **Integer literals**: decimal [proven]; `0x`/`0b`/`0o` hex/binary/octal [impl]; `_` digit separators (`1_000_000`) [impl].
  - **Width/sign suffixes**: `_i8 _i16 _i32 _i64 _u8 _u16 _u32 _u64` ([proven] for i32/i64/u8/u16/u64; [impl] for i8/i16/u32). Default (no suffix) = `i32`.
  - **Wide literals ≥ 2³² — first-class** (was the v1.0 "lexer i32 accumulator" limitation): an **i64** decimal literal of full magnitude decodes to its exact 64-bit value (the lexer carries the literal's source-text ref; codegen re-decodes it full-width via an i32 16-bit-limb path — the f64-literal mechanism mirrored). Corpus: `L2_i64_bigger` (`5_000_000_000_i64`, > 2³², `/1e8 == 50`). **[proven — v1.0/H5]**
  - **u64 literals ≥ 2³² — first-class (v1.3 V2)**: a **u64** literal up to `2⁶⁴-1` parses and computes full-range **unsigned** (the same limb decode, no sign extension; `kovc.hx` AST_INTLIT_U64 tag 38). The former lexer over-range cap (which fail-closed on `> 2³²-1`, the v1.2 L-2 bound) is **retired**. Corpus: `V2_u64_lit_over_2p32` (`5_000_000_000_u64 / 1e8 == 50`), `V2_u64_lit_near_max` (`2⁶⁴-1 > 2⁶³-1` unsigned → 42), `V2_u64_lit_div_max` (`(2⁶⁴-1)/(2⁶³-1) == 2` unsigned). **[proven — v1.3 V2]**
- **Float literals**: `D.D` form; suffixes `_f32` (default) [proven], `_f64` [impl], `_bf16` / `_f16` [impl]. The `_bf16` literal fold is **round-to-nearest-even** (consistent with the `as bf16` cast and bf16 arithmetic — v1.3 V4); the `_f16` literal narrows via an IEEE-754 half conversion.
- **String literals**: `"…"` with escapes `\n \t \r \0 \' \" \\` [impl]; `b"…"`/`r"…"`/`c"…"` prefixes parsed, semantics erased [erased].
- **Char literals**: `'X'` and `'\n'`-style escapes → an integer (byte value) [impl].
- **Identifiers**: `[A-Za-z_][A-Za-z0-9_]*`. `_` alone = match wildcard.
- **Operators**: `+ - * / %` [proven]; `< > <= >= == !=` [proven] (all corpus-verified, LEFT-associative); `& | ^ << >>` bitwise/shift [proven] (`>>` arithmetic for signed, **logical for u64**), `~` [impl]; `!` logical-not [impl]; `=` assignment [proven]; `as` cast [proven]; `.` field access [impl]; `..` range/rest [impl]; `=>` match arm [proven]; `@` attribute [impl]. `&&`/`||` are **not** tokens — use nested `if` [erased].
- **No compound assignment** (`+= -= *= …`) [unsupported].

---

## 2. Types

| Type | Status | Notes |
|------|--------|-------|
| `i8 i16 i32 i64` | i32/i64 [proven], i8/i16 [impl] | signed; full-width arith for i64 |
| `u8 u16 u32 u64` | u8/u16/u64 [proven], u32 [impl] | unsigned; wrap/cast/logical-shift proven |
| `usize` | [erased] | alias parsed, no distinct width tag |
| `f32 f64` | **[proven]** (both) | IEEE-754, SSE codegen (`f64_add`→4, `f64_mul`→12) |
| `bf16` | **[proven]** (v1.3 V4) | add/mul **compute** (convert-to-f32, op, round-to-nearest-even back); bf16→f32 is the identity; needs only SSE2. Bit-exact-gated (`V4_bf16_add/mul/roundtrip`). |
| `f16` | **[proven]** (v1.3 V4 + f16 GAP FIX) | add/mul **compute** via the **F16C** ISA extension (`vcvtph2ps`/`vcvtps2ph` imm8=0 RNE; Ivy Bridge/Jaguar 2012+ — the documented hardware floor), same convert-op-convert shape as bf16. The `f16` type ident + the f16 literal map to type tag **5** (distinct from bf16 tag 4), so `is_f16_expr` fires and `emit_f16_binop` (the F16C path) is reached. **Bit-exact-gated** (`V4_f16_add` 100+28→128 exact; `V4_f16_mul` 7*293 → f32 2051 → **RNE** f16 2052, distinct from a truncating narrow's 2048 — a SHARP check proving the F16C path is genuinely used, not coincidentally right; `vcvtph2ps`/`vcvtps2ph` bytes verified present in the emitted binary). A 16-bit float mixed with a non-16-bit-float operand **traps** (2001/4001; no implicit widening). |
| `bool` | [impl] | represented as i32 0/1; `if` needs an explicit comparison, no implicit int→bool |
| `struct N { … }` | [proven] | named + (positional/tuple [impl]) fields; positional layout. **Wide (i64/u64/f64) scalar fields read + write full 64-bit** (v1.3 V1 — see §2.1). |
| `enum N { V, V(T), … }` | [proven] | tag-only + payload variants; struct-variants [erased] |
| tuples `(a,b)` | **[proven]** | literal + `.0/.1` access (`tuple2`→7) + tuple patterns |
| arrays `[a,b]`, `a[i]` | [proven] | literal + index (`arr_idx`→20); index-store [impl] |
| `tile<ELEM,N,SPACE>` | [impl] | GPU `@kernel` param type only |
| references `&T`/`&mut T`, raw pointers `*T` | [unsupported] | `&` is bitwise-AND only |
| **generics `<T>`/`<T,E>`** | **[erased]** | parsed + depth-balanced-erased; **NO monomorphization** — generic code over differing types is unsafe (see §7). |
| **closures `\|x\| …`** | **[proven]** (v1.3 V3) | non-capturing → raw fn-pointer; **capturing → a real closure object** (arena `[code_ptr, caps…]`) passable by value/as an argument. **Capture-by-value at creation**; **i32-only captures** (a non-i32 capture would truncate in a 4-byte arena cell → fail-closed trap 76003). See §2.2. |

### 2.1 Wide struct fields (i64/u64/f64) — full 64-bit (v1.3 V1)

A struct field of type `i64`, `u64`, or `f64` is read **and** written at its **full 64-bit
width**, and an `f64` field is `f64`-typed so field arithmetic routes through the SSE path.
This **closes the one silent-wrong residual** the v1.2 spec carried (the v1.2 **M-3** bound:
an i64/u64 wide-field READ silently truncated to the low 32 bits, and an f64 wide-field read
fail-closed with SIGILL). The fix is decl-time: `parse_struct_decl` encodes an 8-byte scalar
field (`wide_scalar_field_enc`: f64→tag-2, i64→tag-3, u64→tag-9), the read site
(`AST_TUPLE_FIELD`) emits a REX.W 8-byte load, and `expr_type` recovers the real element tag.

- **Evidence:** `V1_i64_wide_field` (a field holding `5_000_000_000` > 2³², `/1e8 == 50`
  exact — the pre-fix truncation gave 7), `V1_u64_wide_field` (u64 field, unsigned divide →
  50), `V1_f64_wide_field` (f64 field read + `* 2.0` equals an independent f64 local
  reference → 42; pre-fix SIGILL), `V1_multi_wide_field` (a struct mixing i64@slot0 /
  f64@slot1 / i32@slot2 — each field read at its correct offset **and** width → 42).
- **Honest scope:** this is a field-WIDTH fix, gated by the four corpus programs above; it
  does not change the (already-proven) i64/u64/f64 *scalar-local* arithmetic.

### 2.2 Capturing closures as values (v1.3 V3)

A capturing closure (`let c = |y| x + y;`) compiles to a real **closure object** in the
runtime arena — cells `[code_ptr, cap0, cap1, …]` — and its runtime VALUE is the object's
env-index OR-ed with a tag bit (`0x40000000`). The tagged index is a small positive i32 that
survives a by-value i32 parameter (the arena is a low `.data` address < 2³⁰), so a capturing
closure can be **passed by value / as an argument and invoked**. The indirect-call dispatch
(`emit_closure_dispatch`) tag-tests the value: **bit-30 clear** = a non-capturing raw code
pointer → env-less `call` (the v1.2 M-6 path, unchanged); **bit-30 set** = a capturing object
→ untag, load the code pointer from `arena[env]`, pass the env in `rdi`, shift the user args
up one register, call. This **ships the v1.2 M-6 capturing bound**.

- **Capture semantics — CAPTURE-BY-VALUE AT CREATION** (not Rust-style by-reference): each
  captured local's value is snapshotted into the object when the `|…|` literal is evaluated;
  mutating the original afterward does **not** change what the closure sees.
- **Residual (precise):** captures are **i32-only** — a non-i32 capture would be truncated in
  a 4-byte arena cell, so it is **fail-closed (trap 76003), not silent**.
- **Evidence:** `V3_capture_arg` (`x=40; c=|y| x+y; apply(c,2) → 42` — a capturing closure
  passed by value + invoked; pre-fix a SIGSEGV), `V3_multi_capture` (3 captures → 42),
  `V3_modify_after` (capture-by-value-at-creation: mutate the captured local after creation →
  closure still sees the old value → 42, not 1001).

---

## 3. Items

- **Function**: `fn name(p: T, …) -> Ret { body }` [proven]. Body is a single (block) expression. Default return type `i32`. Recursion [proven]. No visibility modifiers (all public). Generic params `<T>` parsed-erased.
- **Struct decl**: `struct Name { f: T, … }` [proven]; tuple struct `struct P(i32,i32)` [impl]; unit struct `struct M;` [unsupported].
- **Enum decl**: `enum Name { V1, V2(T1,T2), … }` [proven] (payload variants proven).
- **Impl block**: `impl Type { fn method(self, …) { … } }` — methods + associated fns **[proven]** (`impl_method`: `p.get()` with `self.x` → 42).
- **`@kernel` function**: GPU kernel, params may be `tile<…>` / `f32` arrays / `i32` scalars; emitted as **PTX** (used by the capstone's 15 kernels) [impl→proven via the capstone, see #3 of the DoD].
- **Attributes**: `@pure` [proven], `@kernel` [impl], `@autotune(…)` [impl], `@deprecated("…")` / `@since("…")` [impl]; Rust `#[…]`/`#![…]` skipped at lex [impl].
- **Module / const / static / trait**: parsed-erased or unsupported (no real semantics) [erased/unsupported].

---

## 4. Expressions & statements

- **`let` / `let mut`**: `let x = e;` / `let mut x = e;` [proven]; `let x: T = e;` [impl]; destructuring `let (x,y)=…` / `let P{x,y}=…` [impl].
- **Assignment**: `x = e;` [proven]; `obj.field = e;` [impl]; `arr[i] = e;` [impl].
- **`if`/`else`** (an **expression** yielding the taken arm's value): `if c { a } else { b }` [proven]. No `else if` keyword — nest in the `else` arm [proven].
- **`while`**: `while c { body }` [proven] (`while_sum`→10); `break` [proven] (`while_break`→7); `continue` [impl]. **No `for`** loop [unsupported] (use `while` + a counter).
- **`match`** (expression) [proven]: arms `pat => body`, comma-separated. Patterns:
  - bind `x` [proven], wildcard `_` [proven], literal `42` [proven], range `a..b` **[proven]** (`match_range`→1),
  - tuple `(a,b)` **[proven]**, **struct `P { x, y }` / `P { x: 0, y }` (literal field) / `O { i: I { v }, t }` (nested) / `P { .. }` (rest)** [proven — fixed 2026-06-01],
  - enum variant `E::V(x)` [proven], or-pattern `A | B` **[proven]** (`match_or`→10).
  - **Guards `pat if cond =>` are parsed but NOT enforced** — every matching arm body runs regardless of the guard [erased]. No exhaustiveness check [unsupported].
- **Blocks / sequencing**: `{ s1; s2; tail }`; `;` separates statements; the trailing expression is the block's value [proven].
- **Cast**: `e as T` — int↔int (width-correct), int↔float, float↔float [proven].
- **Calls**: `f(a, b, …)` [proven]; method `x.m(…)` **[proven]** (`impl_method`); field `s.f` / `t.0` **[proven]**; index `a[i]` [proven].
- **`return e;`** early-exit [impl]. **Unary**: `-e` [proven], `~e` [impl], `!e` [impl].
- **Arithmetic correctness + associativity**: operators are **LEFT-associative** — corpus-proven (2026-06-01): `10 - 3 - 2` → **5** (not 9), `100 / 5 / 2` → **10** (not 40). The full operator set is now corpus-proven: comparisons `!= >= <= == > <`, bitwise `& | ^`, shift `<<`/`>>`, plus arrays, `while`+`break`. Also verified end-to-end by the **capstone** (transformer forward+backward+Adam matching numpy to 0.0009%).

---

## 5. Builtins & intrinsics

- **Arena** (the runtime heap; one i32 slot per element): `__arena_len()` `__arena_get(i)` `__arena_set(i,v)` `__arena_push(v)` [proven]; `__arena_push_pair/triple` [impl].
- **File I/O**: `read_file_to_arena(path)` → byte count (one byte per slot) [proven via the self-host driver]; `write_file_to_arena(path, start, count)` [proven].
- **Process** (Helix-native test-runner primitives): `run_process(path)` → child exit (fork+execve+wait4) [impl, used by `selfhost_bytecmp.hx`]; `set_exec(path)` → chmod 0755 [impl].
- **Print/panic**: `print_str` / `print_str_ln` / `eprint_str(_ln)` [impl]; `panic(msg)` → trap [impl].
- **f32/f64 math** (SSE): `__fadd/__fsub/__fmul/__fdiv/__fneg/__fsqrt/__fabs/__fmin/__fmax`, `__i32_to_f32`/`__f32_to_i32`, bit reinterprets; f64 equivalents + `__f64_pack`/`__bits_{lo,hi}_f64` [impl; the f32 set is capstone-exercised].
- **bf16/f16 arithmetic** (v1.3 V4): `+` and `*` on a `bf16`/`f16` operand pair **compute** via convert-op-convert — operands widen to f32, the op runs in f32 (`addss`/`mulss`), the f32 result rounds back to the 16-bit float with **round-to-nearest-even**. bf16 uses SSE2 (RNE done in integer arithmetic on the f32 bits); f16 uses the **F16C** extension (`vcvtph2ps`/`vcvtps2ph` imm8=0). A 16-bit float mixed with a non-16-bit-float operand still **traps** (no implicit widening). **Both bf16 and f16 are now bit-exact-gated**: bf16 by `V4_bf16_add/mul/roundtrip`; f16 by `V4_f16_add` (100+28→128 exact) and `V4_f16_mul` (7*293 → f32 2051 → RNE f16 2052, a sharp round-to-nearest-even discriminator distinct from truncation's 2048). The f16 GAP FIX (2026-06-04) wired the `f16` ident + f16 literal to type tag 5 so `emit_f16_binop` is reached — previously that F16C path was unreachable dead code and f16 same-type arithmetic silently miscomputed (the gap Finale Audit 2 caught).
- **GPU intrinsics** (in `@kernel`, emitted to PTX): `__gpu_exp`, `__gpu_rsqrt`, `__gpu_i2f`, `__gpu_exp`, threadIdx/blockIdx accessors — the capstone's 15 kernels prove the PTX path on real hardware (DoD #3) [proven for the capstone op set].
- **Autodiff**: `grad(f, idx)` — forward-mode derivative of a named fn [impl; gradient_descent.hx is corpus-proven]. (The capstone uses hand-written verified backward kernels, not the `grad` keyword on GPU — see DoD #4.)
- **Misc**: `__hash_i32`, `__strlen` (compile-time), tile builtins `__tile_{zeros,add,sub,mul,matmul}` [impl]; reflection stubs return 0 [impl].

---

## 6. Codegen targets

- **CPU**: a **static, syscall-only x86-64 Linux ELF** — single `PT_LOAD`, `.text` at `0x401000`, no dynamic linker, System-V AMD64 ABI (6 int args in registers), a big-stack `_start` (mmaps 512 MiB then switches `rsp`, so deep self-compiles need no `ulimit`). Syscalls used: exit/read/write/mmap/mprotect/fork/execve/wait4/chmod. No register allocator or inliner beyond the ABI.
- **GPU**: textual **PTX** for `@kernel` functions (one+ `.entry` per module; the C launcher loads the module and `cuLaunchKernel`s each). Scalar ops, `threadIdx.x`/`blockIdx.x`, the math intrinsics above.

---

## 7. Known limitations & open scope (HONEST)

**Limitations (documented).** v1.3 *retired* several of the v1.0/v1.2 limitations (they now
ship — see §9); the list below is the **post-v1.3** honest residual.

*Retired by v1.3 (no longer limitations — see §9):* the silent i64/u64 wide-field truncation
(V1, the only silent bug — now closed), the i64-literal-≥2³¹ truncation and the u64-literal
over-range cap (V2 — wide literals are first-class), the no-capturing-closures bound (V3),
and the bf16/f16-storage-only bound (V4 — arithmetic computes).

*Still bounded (honest residuals that remain):*
- **Generics are erased** — `<T>` parsed but no general monomorphization of differing element types; generic code over *differing* element types is unsafe. **This is the most significant remaining gap.** [erased]
- **Pattern guards erased** — `if`-guards in match arms are not enforced. [erased]
- **f16 arithmetic — now bit-exact-gated** (was an ungated residual; the f16 GAP FIX of 2026-06-04 closed it). f16 same-type add/mul compute via F16C (`vcvtph2ps`/`vcvtps2ph` RNE) AND are gated by `V4_f16_add`/`V4_f16_mul` (the mul row is a sharp RNE-vs-truncation discriminator). The pre-fix state was worse than "ungated": the F16C path was unreachable dead code, so f16 same-type arithmetic *silently miscomputed* — that silent-wrong path is gone. (Behavioral second-witness cross-check across a zero-kovc-lineage interpreter is still f16-unfixtured — see §9 V5.) [proven, gated]
- **Closure captures are i32-only** — a wider capture fail-closes (trap 76003), it is not silently truncated. [bounded, fail-closed]
- **No** `for`-loop / compound-assignment / `&&`/`||` *as core surface* — these are **parser desugars** (to `while` / `op` + reassign / nested `if`), exercised by the gate's `M1_for_loop`/`M2_compound_assign`/`L4_short_circuit` rows, not first-class control forms. [desugar]
- **No** traits as a checked abstraction, references/pointers, async. [unsupported/erased]
- Lifetimes/`where`-clauses parsed but ignored. [erased]

*Unenforced **by design** (documented bounds — kovc deliberately does not check these; each is locked by a `*_bound` corpus row that proves kovc accepts code a strict checker would reject):*
- **Borrows / `&mut` non-aliasing** — no borrow checker; aliasing/mutation rules are unenforced. [by-design]
- **`const` / `static`** — parsed-erased, no real const/static semantics or const-folding. [by-design]
- **Module privacy** — a private (non-`pub`) item is accepted and runs; no privacy enforcement (`M7_privacy_bound` → 42). [by-design]
- **Match exhaustiveness** — a non-exhaustive `match` is accepted and runs the covered arms (`L3_nonexhaustive_bound` → 42). [by-design]
- **Bare non-i32 scalar generic** — `id(3.0_f32)` defaults `T→i32` (`M5_bare_generic_bound` → 0); the supported idiom is explicit turbofish. [by-design]

**v1.0 SCOPE DECISIONS — RESOLVED 2026-06-01** (see `HELIX_V1_DEFINITION_OF_DONE.md`, "v1.0 SCOPE DECISIONS"):
1. **generics / traits / closures** — were **post-v1.0** at v1.0 freeze. **Closures now SHIP** (v1.3 V3 — capturing closures as values, §2.2); generics remain **erased** and traits remain **unchecked** (the documented residuals in §7). They extend this spec without changing what v1.0 defines.
2. **`Ok`/`Err`/`Result`** — **user-defined `enum`** (not builtins); proven by `result_inline.hx` (→42). The more Helix-native answer; needs no compiler magic.
3. **CUDA C launcher** — **documented trusted-tool boundary** (compute-free C shim, same category as `ld`/`ptxas`; NOT FFI). **numpy oracle** — **fenced external verification reference** (`verification/oracle/`), kept because an independent oracle is required for trustworthy verification.

---

## 8. Proven corpus

The v1.0 acceptance corpus was the 35 programs of `scripts/feature_corpus.sh`:
baseline-literal (42) · scalar-arith (69) · struct+enum+match (129) · payload-enum+match (42) ·
enum+recursion (120) · nested-PatStruct-destructure (42) · user-defined-`enum Result`+match (42) ·
grad+float (42) · i64 cast/cmp/neg · i64 mul-beyond-i32 (6) · i64 div-beyond-i32 (50) ·
u64 logical-shift (1) · u8/u16 wrap-cast (42) · i16 overflow (42) · left-assoc sub/div · comparisons (ne/ge/le) · bitwise (and-or/xor/shl) · array literal+index · while + break · **f64** add/mul · **tuples** · **impl-method** (self) · **match or/range patterns** · **collections** (Vec-on-arena POC →45). **35/35 on the self-hosted K2 (2026-06-01).**

Since v1.0 the gated corpus is **`scripts/gate_kovc.sh`** (run via the universal gate, every
program compiled + run through the fresh self-hosted **K2**). It has grown to **107 passing
programs** (the v1.0 35 + the v1.1/v1.2/v1.3 feature, desugar, document-as-bound, and
type-completeness additions), each row asserting an exact exit code. The **v1.3** additions
specifically (see §9): `V1_{i64,u64,f64,multi}_wide_field`, `V2_u64_lit_{over_2p32,near_max,
div_max}`, `V3_{capture_arg,multi_capture,modify_after}`, `V4_bf16_{add,mul,roundtrip}`, `V4_f16_{add,mul}` (the f16 GAP FIX rows) — and
the corresponding fail-closed negative rows (`M3_wide_field_bound`, `L2_u64_over_2p32`,
`arm_bf16_arith_bound`) are **retired**, because a shipped feature must not still assert
fail-closed.

---

## 9. v1.3 record — "types first-class & residuals stated" (2026-06-04)

v1.3 promotes four type-correctness items from *erased/fail-closed-bounded* to *first-class +
gated*, and deepens trust on two axes. Each is gated by the named corpus test(s) above and
the codegen cited; nothing here is asserted from memory.

| Item | What shipped | Precise residual kept |
|------|--------------|-----------------------|
| **V1** | i64/u64/f64 **wide struct fields read + write full 64-bit** — **the one silent-truncation bug (v1.2 M-3) is CLOSED** (§2.1). | none new; it is a field-width fix (4 gated rows). |
| **V2** | **u64 literals up to 2⁶⁴-1** parse + compute full-range unsigned (§1); the v1.2 L-2 over-range cap retired. | none new for u64 literals. |
| **V3** | **capturing closures as values/args** (closure object; **capture-by-value-at-creation**) (§2.2); v1.2 M-6 shipped. | **i32-only captures** — wider captures **fail-closed** (trap 76003), not silent. |
| **V4** (+ f16 GAP FIX) | **bf16/f16 add/mul compute** convert-op-convert, **round-to-nearest-even** (§5, §2 table). f16 uses F16C (`vcvtph2ps`/`vcvtps2ph`); the f16 GAP FIX (2026-06-04) mapped the `f16` ident + literal to type tag 5 so the F16C path is reached. | **Both bf16 AND f16 are now bit-exact-gated** — bf16 by `V4_bf16_*`, f16 by `V4_f16_add`/`V4_f16_mul` (sharp RNE-vs-trunc discriminator). The pre-fix gap (f16 same-type arith silently miscomputed via an unreachable F16C path — caught by Finale Audit 2) is closed. Remaining: the K_DDC behavioral second-witness is still f16-un-fixtured (V5). |
| **V5** | the v1.1 surface (generics/traits/closures/turbofish/wide-field/bf16) now has an **independent BEHAVIORAL cross-check** — a second, zero-kovc-lineage tree-walking interpreter agrees with the from-raw kovc on **44/44** v1.1-surface programs (`docs/K_DDC_BROADENED.md`). | **BEHAVIORAL, not byte-identical** (the interpreter emits no machine code, so no second ELF to `cmp`); the shared-host-runtime / shared-bug DDC residual is unchanged; **f16-arith is un-fixtured** in this witness too. |
| **V6** | the **trusted-C surface is inventoried + minimized** — 6 dead duplicate `M2libc/bootstrappable.{c,h}` pruned; **24 committed C/H files** classified (`docs/TRUSTED_C_INVENTORY.md`); **`seed.c` = the single irreducible root**. | the **CUDA host launcher** (`cuda_launch.c` / `train_transformer.c`) is the documented **GPU C-FFI boundary**; below PTX it relies on NVIDIA's **closed `ptxas` + driver** — `TRUST_CHAIN_CLOSED.md` **residual #7 STANDS** (porting the launcher moves, does not close, it). |

**The v1.0 surface is unchanged by v1.3** — these are depth/honesty promotions and trust
deepening, not new syntax. The design bounds in §7 (borrows/`&mut` non-aliasing, `const`/
`static`, module privacy, match exhaustiveness, bare non-i32 generics) remain
**unenforced-by-design** and are each locked by a `*_bound` corpus row. The universal gate
(self-host fixpoint K2==K3==K4 byte-identical + GPU-PTX regression + the 107-program corpus)
stays green; the Python fence stays at exactly **1** committed `.py`.

---

*Authored 2026-06-01 from a read-only enumeration of `kovc`/`lexer`/`parser` + the proven
corpus; **v1.0 surface FROZEN 2026-06-01**. **v1.3 deltas folded in 2026-06-04** (§9 + the
inline "v1.3" marks) — every "now-first-class" claim verified against the live `kovc.hx` /
`parser.hx` / `lexer.hx` codegen and the `scripts/gate_kovc.sh` corpus, each residual stated
with its precise scope.*
