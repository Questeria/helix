# Helix v1.0 — Language Reference (DRAFT)

**Status: FROZEN — v1.0 (2026-06-01).** This documents the Helix language **as actually
implemented by the self-hosted compiler `kovc`** (helixc/bootstrap/{lexer,parser,kovc}.hx),
the only compiler after K4 (the Python reference was deleted). The v1.0 scope decisions
(§7) are **RESOLVED** (generics/traits/closures = post-v1.0; `Ok`/`Err`/`Result` =
user-defined — see `HELIX_V1_DEFINITION_OF_DONE.md`, "v1.0 SCOPE DECISIONS"), and the v1.0
language surface is committed: **no breaking changes after v1.0**. Post-v1.0 additions (the
deferred features) extend this spec; they do not change what v1.0 defines.

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
  - **⚠ Limitation**: a decimal source literal **≥ 2³¹ truncates** (the lexer accumulates in i32). Runtime i64 values beyond i32 work via computation on sub-2³¹ literals (corpus-proven). Use `2_000_000_000_i64 * 3_i64` rather than `6_000_000_000_i64`. [limitation, deferred]
- **Float literals**: `D.D` form; suffixes `_f32` (default) [proven], `_f64` [impl], `_bf16` / `_f16` [impl].
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
| `bf16 f16` | [impl] | truncated/half precision |
| `bool` | [impl] | represented as i32 0/1; `if` needs an explicit comparison, no implicit int→bool |
| `struct N { … }` | [proven] | named + (positional/tuple [impl]) fields; positional layout |
| `enum N { V, V(T), … }` | [proven] | tag-only + payload variants; struct-variants [erased] |
| tuples `(a,b)` | **[proven]** | literal + `.0/.1` access (`tuple2`→7) + tuple patterns |
| arrays `[a,b]`, `a[i]` | [proven] | literal + index (`arr_idx`→20); index-store [impl] |
| `tile<ELEM,N,SPACE>` | [impl] | GPU `@kernel` param type only |
| references `&T`/`&mut T`, raw pointers `*T` | [unsupported] | `&` is bitwise-AND only |
| **generics `<T>`/`<T,E>`** | **[erased]** | parsed + depth-balanced-erased; **NO monomorphization** — generic code over differing types is unsafe (see §7). |

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
- **GPU intrinsics** (in `@kernel`, emitted to PTX): `__gpu_exp`, `__gpu_rsqrt`, `__gpu_i2f`, `__gpu_exp`, threadIdx/blockIdx accessors — the capstone's 15 kernels prove the PTX path on real hardware (DoD #3) [proven for the capstone op set].
- **Autodiff**: `grad(f, idx)` — forward-mode derivative of a named fn [impl; gradient_descent.hx is corpus-proven]. (The capstone uses hand-written verified backward kernels, not the `grad` keyword on GPU — see DoD #4.)
- **Misc**: `__hash_i32`, `__strlen` (compile-time), tile builtins `__tile_{zeros,add,sub,mul,matmul}` [impl]; reflection stubs return 0 [impl].

---

## 6. Codegen targets

- **CPU**: a **static, syscall-only x86-64 Linux ELF** — single `PT_LOAD`, `.text` at `0x401000`, no dynamic linker, System-V AMD64 ABI (6 int args in registers), a big-stack `_start` (mmaps 512 MiB then switches `rsp`, so deep self-compiles need no `ulimit`). Syscalls used: exit/read/write/mmap/mprotect/fork/execve/wait4/chmod. No register allocator or inliner beyond the ABI.
- **GPU**: textual **PTX** for `@kernel` functions (one+ `.entry` per module; the C launcher loads the module and `cuLaunchKernel`s each). Scalar ops, `threadIdx.x`/`blockIdx.x`, the math intrinsics above.

---

## 7. Known limitations & open scope (HONEST)

**Limitations (documented, mostly deferred):**
- **Generics are erased** — `<T>` parsed but no monomorphization; generic code over *differing* element types is unsafe. The corpus avoids generics. **This is the most significant gap.** [erased]
- **Pattern guards erased** — `if`-guards in match arms are not enforced. [erased]
- **i64 source literals ≥ 2³¹ truncate** (lexer i32 accumulator). [limitation]
- **No** `for` loops, compound-assignment, traits, closures, references/pointers, module visibility, async, exhaustiveness checks, const-folding. [unsupported/erased]
- Lifetimes/`where`-clauses parsed but ignored. [erased]

**v1.0 SCOPE DECISIONS — RESOLVED 2026-06-01** (see `HELIX_V1_DEFINITION_OF_DONE.md`, "v1.0 SCOPE DECISIONS"):
1. **generics / traits / closures** — **post-v1.0** (deferred). Erased/unsupported above; not in the v1.0 feature set. They extend this spec later without changing what v1.0 defines.
2. **`Ok`/`Err`/`Result`** — **user-defined `enum`** (not builtins); proven by `result_inline.hx` (→42). The more Helix-native answer; needs no compiler magic.
3. **CUDA C launcher** — **documented trusted-tool boundary** (compute-free C shim, same category as `ld`/`ptxas`; NOT FFI). **numpy oracle** — **fenced external verification reference** (`verification/oracle/`), kept because an independent oracle is required for trustworthy verification.

---

## 8. Proven corpus (the 35 programs, `scripts/feature_corpus.sh`)

baseline-literal (42) · scalar-arith (69) · struct+enum+match (129) · payload-enum+match (42) ·
enum+recursion (120) · nested-PatStruct-destructure (42) · user-defined-`enum Result`+match (42) ·
grad+float (42) · i64 cast/cmp/neg · i64 mul-beyond-i32 (6) · i64 div-beyond-i32 (50) ·
u64 logical-shift (1) · u8/u16 wrap-cast (42) · i16 overflow (42) · left-assoc sub/div · comparisons (ne/ge/le) · bitwise (and-or/xor/shl) · array literal+index · while + break · **f64** add/mul · **tuples** · **impl-method** (self) · **match or/range patterns** · **collections** (Vec-on-arena POC →45). **35/35 pass on the self-hosted K2 (2026-06-01).**

---

*Authored 2026-06-01 from a read-only enumeration of `kovc`/`lexer`/`parser` + the proven corpus. **FROZEN v1.0 (2026-06-01)** — §7 scope decisions resolved; v1.0 language surface committed.*
