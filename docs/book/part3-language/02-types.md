# Types: widths, structs, and enums

*What this chapter covers:* the Helix type system **as the from-raw compiler `kovc` actually
implements it** — the signed and unsigned integer widths (including 64-bit literals beyond
2³¹/2³²), the float widths (`f32`, `f64`, and the 16-bit `bf16`/`f16`), `bool`, how `struct` and
`enum` are declared and used, where type annotations go, and how integer literals are typed and
widened. Every width and feature below is tied to the language spec **and** to a real, gate-proven
`.hx` file, with the gate-asserted exit code cited. Where the spec describes a type that is only
partial, parsed-but-erased, or aspirational, this chapter says so plainly rather than implying it
works.

The authoritative source for what the language *is* (as opposed to the v0.1 design vision) is
[`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) — "the language as
actually implemented by the self-hosted compiler `kovc`." The older
[`docs/lang/spec.md`](../../../docs/lang/spec.md) is a **historical v0.1 design draft** that
describes intended/aspirational syntax against the now-deleted Python frontend; it is *not* the
as-built reference, and this chapter follows the v1 spec wherever they differ.

> **For AI agents:** when a width, suffix, or type behavior matters, key off
> [`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §1–§2 and the gate
> corpus in [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh), not
> [`docs/lang/spec.md`](../../../docs/lang/spec.md). The v0.1 draft lists types (`fp8`, `mxfp4`,
> `ternary`, `usize` as a distinct width) that the as-built `kovc` does **not** implement as
> claimed. If the two disagree, the v1 spec and the gate win.

---

## How "proven" is established in this chapter

Helix's whole purpose is auditable trust, so a chapter about its types must hold itself to the same
bar. The standing compile-proof is [the gate](../../../scripts/gate_kovc.sh): on top of verifying
the self-host fixpoint, it compiles **and runs** a 109-program feature corpus through the freshly
self-hosted compiler (`K2`), asserting an exact process exit code for each program. A program that
appears in that corpus with an asserted exit code is *demonstrably* compiled and run by the from-raw
toolchain — that is what lets this chapter label it a **Verified example**.

Two mechanical facts you need to read the examples:

- **The exit-status convention.** A Helix `fn main() -> i32` returns an `i32`, and that value
  becomes the process exit status. The OS truncates the exit status to **8 bits** (`value & 0xFF`),
  so corpus programs are written to return a value in `0..255` — and several width tests below use a
  sentinel like `if … { 42 } else { 0 }` precisely so a wide internal result is checked *before* it
  would be wrapped by the exit byte.
- **Where the corpus lives.** Some width programs are committed as standalone fixtures under
  `stage0/helixc-bootstrap/corpus_gen/`; others are generated inline by the gate via a heredoc and
  then compiled+run the same way. Both are part of the standing 109/0 proof. This chapter cites the
  exact `chk` line in [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) that asserts each
  program's exit code.

> **For AI agents:** the gate asserts exit codes with `chk "<file>" <expected>` (see the `chk()`
> helper, [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`). The OS exit byte is
> `rc & 0xFF`; do not expect a returned value ≥ 256 to survive as-is. When a corpus program returns
> a sentinel (`42`/`0`) instead of the raw computed value, that is *deliberate* — the full-width
> result is compared internally, and only the sentinel crosses the exit boundary.

---

## Integer widths

Helix has the conventional signed and unsigned integer widths. The implementation status, taken
verbatim from the type table in
[`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §2, is:

| Type | Status | Notes |
|------|--------|-------|
| `i8 i16 i32 i64` | `i32`/`i64` proven, `i8`/`i16` impl | signed; full-width arithmetic for `i64` |
| `u8 u16 u32 u64` | `u8`/`u16`/`u64` proven, `u32` impl | unsigned; wrap/cast/logical-shift proven |
| `usize` | erased | the alias is parsed, but there is **no distinct width tag** |

`i32` is the default integer type: an integer literal with no width suffix is an `i32`
([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §1, "Default (no
suffix) = `i32`"). Arithmetic is two's-complement and **wraps** on overflow; Helix does not trap on
integer overflow (the v0.1 draft's "Integer arithmetic semantics" note describes this wraparound
model, and it is what the as-built corpus exercises below).

> **Residual:** `usize` is accepted by the parser as a type name but has no distinct width — the
> spec marks it `[erased]`. Do not rely on `usize` having pointer width or any size-checked
> behavior; use an explicit `i64`/`u64` if you need 64-bit semantics.

### Width and sign suffixes on literals

A numeric literal can carry an explicit width/sign suffix. The lexer recognizes the full set
`_i8 _i16 _i32 _i64 _u8 _u16 _u32 _u64`
([`helixc/bootstrap/lexer.hx`](../../../helixc/bootstrap/lexer.hx), `lex_int` — see the suffix
cascade that sets `is_i64_suffix`, `is_u64_suffix`, `is_u8_suffix`, etc., and the closing block that
emits a distinct token tag per suffix). Underscores are accepted as digit separators
(`1_000_000`), and the lexer is careful to *stop* the digit loop at the `_` of a suffix so that
`42_i64` is tagged as an `i64` literal, not a plain `i32` followed by the separator logic — a real
bug that was fixed (`K1.E1d-fix`, documented inline in `lex_int`). The spec records the suffixes as
`[proven]` for `i32`/`i64`/`u8`/`u16`/`u64` and `[impl]` for `i8`/`i16`/`u32`
([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §1).

The literal base may also be decimal, hex (`0x`), binary (`0b`), or octal (`0o`), each with
underscore separators — the lexer's `lex_int` handles all four
([`helixc/bootstrap/lexer.hx`](../../../helixc/bootstrap/lexer.hx)). The spec marks the non-decimal
bases `[impl]`.

### `i32` — the default width

The smallest corpus program is the canonical first program; it returns a bare integer literal,
which is an `i32`:

**Verified example** — [`helixc/examples/exit42.hx`](../../../helixc/examples/exit42.hx) (the gate
asserts exit `42` — [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh):
`chk "$EX/exit42.hx" 42`):

```helix
// First end-to-end Helix program: compiles, links, runs, exits with status 42.
//
// Demonstrates: function decl, return, integer literal.
// Verification: $? == 42 after running the produced ELF.

fn main() -> i32 {
    42
}
```

`i32` arithmetic, comparisons, bitwise operators, and shifts are all corpus-proven; that surface is
covered in the operators chapter rather than re-listed here.

### `i64` — full 64-bit arithmetic and the H5 wide-literal hardening

`i64` is a proven width with **full 64-bit arithmetic**
([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §2). The basic case —
declaring an `i64`, doing arithmetic, and casting back — is gate-checked by an inline corpus program
([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh), step `[4]`, `chk "$CD/i64_basic.hx" 42`):

**Verified example** — `i64_basic.hx`, generated and gate-checked to exit `42` inline in
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh):

```helix
fn main() -> i32 { let x: i64 = 42_i64; x as i32 }
```

The subtle, trust-critical part of `i64` is **literals whose magnitude exceeds 2³¹ (and 2³²)**.
This was a real hardening item: **H5** in
[`docs/HELIX_V1_1_HARDENING.md`](../../../docs/HELIX_V1_1_HARDENING.md), "i64 source literals ≥
2³¹." The problem was that the bootstrap toolchain's `seed` has no native i64 (it is i32-only), so
an i64 literal could not simply be accumulated in an i32 during lexing without truncating. The fix,
as recorded in the H5 row, mirrors the f64-literal mechanism: the literal's **source text** is
carried through to codegen, which decodes the decimal digits into the full 64-bit value using
i32 multi-word 16-bit limbs (all-positive arithmetic, so the i32-only `seed` can self-compile it).
After the fix, `5_000_000_000_i64` (which is > 2³²) no longer truncates.

This is gate-locked by three standalone fixtures. Each divides a large `i64` by `100_000_000` (1e8)
and checks the exact quotient — a low-32 truncation would give a visibly wrong answer.

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/L2_i64_bigger.hx`](../../../stage0/helixc-bootstrap/corpus_gen/L2_i64_bigger.hx)
(gate-checked to exit `50`; [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh):
`chk "$GENC/L2_i64_bigger.hx" 50`):

```helix
fn main() -> i32 {
    let x: i64 = 5_000_000_000_i64;
    (x / 100000000) as i32
}
```

`5_000_000_000 / 100_000_000 == 50`. The literal `5_000_000_000_i64` is larger than 2³² =
4 294 967 296, so this single line is the proof that the full 64-bit literal materialized: had the
literal been truncated to its low 32 bits, the program would have computed `705032704 / 100000000 ==
7`, not `50`. The two companion fixtures cover values just below and across the boundary —
[`L1_i64_big.hx`](../../../stage0/helixc-bootstrap/corpus_gen/L1_i64_big.hx) uses
`3_000_000_000_i64` and exits `30`, and
[`L3_i64_just_over.hx`](../../../stage0/helixc-bootstrap/corpus_gen/L3_i64_just_over.hx) uses
`2_200_000_000_i64` (just over 2³¹) and exits `22`
([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh): `chk "$GENC/L1_i64_big.hx" 30` and
`chk "$GENC/L3_i64_just_over.hx" 22`).

The gate also exercises `i64` multiplication and division *beyond* the i32 range, comparison, and
negation, via the inline `i64_mul_beyond.hx` (→ `6`), `i64_div_beyond.hx` (→ `50`), `i64_cmp.hx`
(→ `1`), and `i64_neg.hx` (→ `5`) programs
([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh), step `[4]`). For example, `i64_cmp.hx`
compares two literals each larger than 2³²:

**Verified example** — `i64_cmp.hx`, generated and gate-checked to exit `1` inline in
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) (`chk "$CD/i64_cmp.hx" 1`):

```helix
fn main() -> i32 { let a: i64 = 5_000_000_000_i64; let b: i64 = 4_000_000_000_i64; if a > b { 1 } else { 0 } }
```

> **For AI agents:** the wide-`i64`-literal path is **codegen-decoded from the literal's text**, not
> lexed into an integer accumulator. That is why a literal like `5_000_000_000_i64` is correct even
> though the bootstrap `seed` has no i64 type. Do not assume there is an i64 lexer accumulator to
> reason about; the carrier is the source-text reference (the same tag-34 f64-literal mechanism,
> mirrored). See [`docs/HELIX_V1_1_HARDENING.md`](../../../docs/HELIX_V1_1_HARDENING.md) H5.

### `u64` — unsigned, with full-range literals (v1.3 V2)

`u64` is a proven width: unsigned wrap, cast, and **logical** (not arithmetic) right-shift are all
gate-checked. The `>>` operator is logical for `u64`
([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §1). The shift case is
covered by an inline program ([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh):
`chk "$CD/u64_shr.hx" 1`):

**Verified example** — `u64_shr.hx`, generated and gate-checked to exit `1` inline in
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh):

```helix
fn shr_u64(x: u64) -> u64 { x >> 63_u64 }
fn main() -> i32 { let x: u64 = 1_u64 << 63_u64; shr_u64(x) as i32 }
```

`1_u64 << 63` sets only the top bit; a *logical* right shift by 63 brings it down to `1` (an
arithmetic shift would smear the sign and give a different value).

Like `i64`, `u64` got a wide-literal hardening — this one in v1.3, item **V2** of
[`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §9. Earlier, a `u64`
literal above 2³²−1 was **fail-closed** (it produced a compile error, the v1.2 "L-2" bound). That
cap is now **retired**: a `u64` literal up to 2⁶⁴−1 parses and computes full-range *unsigned*,
decoded full-width via the same limb path as `i64` but with no sign extension. The over-range
helpers were removed from the lexer — see the explicit note in
[`helixc/bootstrap/lexer.hx`](../../../helixc/bootstrap/lexer.hx) ("v1.3 V2: the former u64
lex-overflow helpers … are REMOVED").

Two fixtures lock the unsigned range:

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/V2_u64_lit_over_2p32.hx`](../../../stage0/helixc-bootstrap/corpus_gen/V2_u64_lit_over_2p32.hx)
(gate-checked to exit `50`; [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh):
`chk "$GENC/V2_u64_lit_over_2p32.hx" 50`):

```helix
fn main() -> i32 {
    let big: u64 = 5_000_000_000_u64;   // u64 literal > 2^32 (was L-2-capped)
    let g: u64 = 100000000_u64;         // 1e8
    (big / g) as i32                    // 5e9 / 1e8 = 50 EXACT
}
```

The second fixture pushes to the very top of the range and uses an unsigned comparison as a sign-bug
detector:

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/V2_u64_lit_near_max.hx`](../../../stage0/helixc-bootstrap/corpus_gen/V2_u64_lit_near_max.hx)
(gate-checked to exit `42`; [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh):
`chk "$GENC/V2_u64_lit_near_max.hx" 42`):

```helix
fn main() -> i32 {
    let big: u64 = 18446744073709551615_u64;   // 2^64-1 (full unsigned range)
    let imax: u64 = 9223372036854775807_u64;    // 2^63-1 = i64::MAX
    if big > imax { 42 } else { 0 }             // unsigned -> 42 ; signed-bug -> 0
}
```

`18446744073709551615` is 2⁶⁴−1 — the same bit pattern that, read as a *signed* `i64`, would be −1.
The unsigned comparison `big > imax` is therefore the discriminator: the correct unsigned answer is
`true` (→ `42`), while a sign or truncation defect would flip it to `0`. A third fixture,
[`V2_u64_lit_div_max.hx`](../../../stage0/helixc-bootstrap/corpus_gen/V2_u64_lit_div_max.hx),
divides `(2⁶⁴−1)/(2⁶³−1)` and exits `2` on the unsigned path
([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh): `chk "$GENC/V2_u64_lit_div_max.hx" 2`).

### `u8`, `u16`, `i16` — narrow widths and wrap

The narrow widths are exercised by inline corpus programs that deliberately overflow to confirm the
width is respected. `u8` wrap (`0 - 1` as a `u8` is `255`) and `u16` wrap (`65535`) are checked, as
is `i16` overflow (`32767 + 1` wraps negative):

**Verified example** — `u8_wrap.hx`, generated and gate-checked to exit `42` inline in
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) (`chk "$CD/u8_wrap.hx" 42`):

```helix
fn main() -> i32 { let x: u8 = 0_u8 - 1_u8; let y: i32 = x as i32; if y == 255 { 42 } else { 7 } }
```

The companions are `u16_wrap.hx` (checks `0_u16 - 1_u16 == 65535`, → `42`) and `i16_ovf.hx` (checks
`32767_i16 + 1_i16` is negative, → `42`)
([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh): `chk "$CD/u16_wrap.hx" 42` and
`chk "$CD/i16_ovf.hx" 42`). The spec marks `i8` and `u32` as `[impl]` (codegen exists, exercised by
the L-7 `arm_i8_width.hx` / `arm_u32_width.hx` rows in the gate, each → `42`) rather than having a
dedicated overflow demonstration; `i8`/`i16` are listed `[impl]`/`[proven]` accordingly in
[`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §2.

> **For AI agents:** for narrow unsigned/signed widths the corpus encodes the expected wrap value
> (`255`, `65535`, sign-flip) and returns a `42`/`7` sentinel. The exit byte you observe is the
> sentinel, not the wrapped value — read the `if` condition in the program to learn what was
> actually asserted.

---

## Float widths

Helix implements four floating-point widths. From
[`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §2:

| Type | Status | Notes |
|------|--------|-------|
| `f32 f64` | proven (both) | IEEE-754, SSE codegen |
| `bf16` | proven (v1.3 V4) | add/mul **compute** via convert-to-f32, op, round-to-nearest-even; needs only SSE2 |
| `f16` | proven (v1.3 V4 + f16 GAP FIX) | add/mul **compute** via the **F16C** ISA extension (`vcvtph2ps`/`vcvtps2ph`) |

Float literals are written `D.D` and may carry a suffix: `_f32` (the default for a float literal),
`_f64`, `_bf16`, `_f16` ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md)
§1). Because converting a decimal literal to its IEEE-754 bit pattern needs more than i32
arithmetic, the lexer carries the **literal text** through to the parser/codegen rather than
converting it in place — see the float-literal lookahead in
[`helixc/bootstrap/lexer.hx`](../../../helixc/bootstrap/lexer.hx) (`lex_int`'s `is_float` branch:
"the parser/codegen must convert the text to IEEE 754 bits at parse time").

### `f32` and `f64`

Both `f32` and `f64` are proven, with SSE codegen for the arithmetic. The gate checks `f64` add and
multiply with two inline programs that cast the float result back to `i32` for the exit code
([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh): `chk "$CD/f64_add.hx" 4` and
`chk "$CD/f64_mul.hx" 12`):

**Verified example** — `f64_add.hx`, generated and gate-checked to exit `4` inline in
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh):

```helix
fn main() -> i32 { let a: f64 = 1.5_f64; let b: f64 = 2.5_f64; (a + b) as i32 }
```

`1.5 + 2.5 == 4.0`, cast to `4`. The `f64_mul.hx` companion computes `3.0_f64 * 4.0_f64 == 12.0`
(→ `12`). The `f32` path is the one the GPU capstone exercises end-to-end (the spec notes "the f32
set is capstone-exercised"); the standard library's IEEE-754 helpers are written entirely in the
integer subset and produce f32 bit patterns by hand — see
[`helixc/stdlib/ieee754.hx`](../../../helixc/stdlib/ieee754.hx), whose `f32_bits_pos` builds the
sign/exponent/mantissa of an f32 from a decimal `(integer_part, frac_value, frac_digits)` triple
using only i32 arithmetic and `while` loops.

> **Note:** the `ieee754.hx` helpers are a good window into how careful Helix's integer subset is
> about overflow: `f32_bits_pow10` and `f32_bits_pow2` return an explicit `INT32_MIN` sentinel for
> out-of-range inputs instead of silently wrapping, and callers propagate that sentinel rather than
> emit a corrupted bit pattern ([`helixc/stdlib/ieee754.hx`](../../../helixc/stdlib/ieee754.hx),
> the "Cycle 3 R1 fix batch 20" notes). That fail-loud discipline is the same one the trust chain is
> built on.

### `bf16` — brain-float 16 (v1.3 V4)

`bf16` was, before v1.3, **storage-only**: the type and its literal were accepted, but arithmetic on
a `bf16` operand *trapped* (a deliberate fail-closed). Item **V4** of
[`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §9 promoted `bf16` add
and multiply to *compute*: the operands widen to f32, the op runs in f32, and the f32 result is
rounded back to bf16 with **round-to-nearest-even (RNE)**. `bf16` needs only SSE2 (the RNE round-back
is done in integer arithmetic on the f32 bits); `bf16 → f32` is the identity, since a bf16 value is
stored as the valid top 16 bits of an f32.

The arithmetic is **bit-exact-gated** — the corpus does not merely check "no crash," it checks the
exact rounded value. The operands are chosen so the RNE result differs from what a truncating
round-back would give:

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/V4_bf16_add.hx`](../../../stage0/helixc-bootstrap/corpus_gen/V4_bf16_add.hx)
(gate-checked to exit `42`; [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh):
`chk "$GENC/V4_bf16_add.hx" 42`):

```helix
fn main() -> i32 {
    let a: bf16 = 256.0_bf16;
    let b: bf16 = 3.0_bf16;
    let c: bf16 = a + b;                     // f32 sum 259.0 -> RNE bf16 -> 260.0
    if (c as i32) == 260 { 42 } else { 0 }   // 42 iff RNE 260 (trunc path -> 258 -> 0)
}
```

`256` and `3` are both exact in bf16, but their f32 sum `259.0` is *not* representable in bf16 (at
exponent 8 the bf16 step is 2). RNE rounds `259` to the even neighbour `260`; a truncating path would
land on `258`. The `== 260` check therefore proves the round-back is genuinely round-to-nearest-even.
The `V4_bf16_mul.hx` companion checks `17.0 * 19.0` (f32 product `323` → RNE `324`, → `42`), and a
round-trip fixture proves the conversion direction:

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/V4_bf16_roundtrip.hx`](../../../stage0/helixc-bootstrap/corpus_gen/V4_bf16_roundtrip.hx)
(gate-checked to exit `42`; [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh):
`chk "$GENC/V4_bf16_roundtrip.hx" 42`):

```helix
fn main() -> i32 {
    let x: bf16 = 1.1_bf16;            // f32 1.1 -> RNE bf16 -> 1.1015625
    let back: f32 = x as f32;          // bf16 -> f32 (identity) = 1.1015625
    let ref32: f32 = 1.1015625_f32;    // the known bf16-rounded reference
    if back == ref32 { 42 } else { 0 } // 42 iff the round-trip is bit-exact RNE
}
```

The bf16 *literal* fold is RNE too (consistent with the `as bf16` cast and bf16 arithmetic), so
`1.1_bf16` rounds to `1.1015625`, not the truncating `1.09375`.

### `f16` — IEEE-754 half (v1.3 V4 + the f16 GAP FIX)

`f16` is the IEEE-754 half-precision type. Same-type `f16` add and multiply compute via the **F16C**
instruction-set extension (`vcvtph2ps` to widen, `vcvtps2ph` with `imm8=0` to narrow with RNE) —
the documented hardware floor is Ivy Bridge / Jaguar (2012+)
([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §2, §5). The `f16` type
ident and the `f16` literal map to a **distinct type tag (5)** from bf16 (tag 4); this matters,
because of an honest bug that v1.3's "f16 GAP FIX" closed.

> **Residual (honest history):** before the f16 GAP FIX (2026-06-04), the `f16` ident and literal
> never reached type tag 5, so the F16C path (`emit_f16_binop`) was *unreachable dead code* and
> same-type `f16` arithmetic *silently miscomputed* (it mis-routed through the bf16/integer path and
> returned ~0, with no trap). This was caught by an audit. The fix wired the ident + literal to tag
> 5 so the F16C path is actually reached, and added two **sharp** gated rows that distinguish the
> correct F16C result from both the old wrong value and a truncating narrow. See
> [`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §7 / §9 (V4). The
> lesson is the book's: a feature is not "done" until a gated test would *fail* if it regressed.

The two sharp rows:

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/V4_f16_add.hx`](../../../stage0/helixc-bootstrap/corpus_gen/V4_f16_add.hx)
(gate-checked to exit `42`; [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh):
`chk "$GENC/V4_f16_add.hx" 42`):

```helix
fn main() -> i32 {
    let a: f16 = 100.0_f16;
    let b: f16 = 28.0_f16;
    let c: f16 = a + b;                      // F16C: widen->addss->narrow RNE = 128.0
    if (c as i32) == 128 { 42 } else { 0 }   // 42 iff the F16C add gives 128 (old silent-wrong path -> ~0 -> 0)
}
```

`100` and `28` are exact in f16 (which has 10 mantissa bits, so integers up to 2048 are exact), and
their sum `128` is exact — so a correct path yields `128`, while the old silent-wrong path yielded
~0. The multiply fixture is sharper still:

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/V4_f16_mul.hx`](../../../stage0/helixc-bootstrap/corpus_gen/V4_f16_mul.hx)
(gate-checked to exit `42`; [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh):
`chk "$GENC/V4_f16_mul.hx" 42`):

```helix
fn main() -> i32 {
    let a: f16 = 7.0_f16;
    let b: f16 = 293.0_f16;
    let c: f16 = a * b;                       // f32 product 2051 -> RNE f16 -> 2052 (trunc -> 2048; silent -> ~0)
    if (c as i32) == 2052 { 42 } else { 0 }   // 42 iff F16C + RNE gives 2052
}
```

The exact f32 product `7 * 293 = 2051` is not representable in f16 (in `[2048, 4096)` the f16 step is
4, so the neighbours are 2048 and 2052). `2051` sits 3/4-ULP above `2048`, so it is nearer to `2052`
— RNE rounds *up* to `2052`. This single value distinguishes three outcomes: the correct F16C+RNE
result (`2052`), a truncating narrow (`2048`), and the old dead-code bug (~0). The gate's own comment
records that `vcvtph2ps`/`vcvtps2ph` were verified *present* in the emitted binary.

> **Residual:** a 16-bit float (bf16 or f16) mixed with a non-16-bit-float operand still **traps**
> (no implicit widening across the 16-bit boundary)
> ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §2). Convert
> explicitly with `as` if you need to combine a `bf16`/`f16` with an `f32`.

> **For AI agents:** `bf16` runs on SSE2; `f16` arithmetic *requires F16C* (the 2012+ hardware
> floor). If you target a CPU without F16C, `f16` same-type arithmetic is out of scope — there is no
> software fallback. Do not assume `f16` and `bf16` share an instruction path; they map to distinct
> type tags (5 vs 4) and distinct codegen.

### Float types the v0.1 draft lists but `kovc` does **not** implement

The historical [`docs/lang/spec.md`](../../../docs/lang/spec.md) lists `fp8` (E4M3/E5M2), `mxfp4`,
`nvfp4`, and a packed `ternary` type in its primitive-type table. These are **design-vision
entries**, not implemented types — they do not appear in the as-built type table of
[`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §2, and there is no gated
corpus program for them. Treat them as not-implemented; the implemented float set is exactly `f32`,
`f64`, `bf16`, `f16`.

---

## `bool`

`bool` is `[impl]` and is **represented as an `i32` (0 or 1)**
([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §2). The key semantic
consequence: there is **no implicit int → bool coercion**. An `if` condition needs an *explicit
comparison*; you cannot write `if x { … }` for an integer `x` and have nonzero mean true.

This is why Helix code is written with explicit comparisons everywhere — for example, the lexer's
byte-classification helpers return `1`/`0` and the loop conditions test `== 1` rather than relying on
truthiness:

**Fragment** — excerpt from [`helixc/bootstrap/lexer.hx`](../../../helixc/bootstrap/lexer.hx)
(`is_whitespace`; illustrates the i32-as-bool, explicit-comparison style — not a complete program):

```helix
@pure
fn is_whitespace(b: i32) -> i32 {
    if b == 32 { 1 }
    else { if b == 9 { 1 }
    else { if b == 10 { 1 }
    else { if b == 13 { 1 }
    else { 0 }}}}
}
```

> **For AI agents:** there is no implicit truthiness in Helix. Always write an explicit comparison in
> a condition (`if x == 1 { … }`, `while keep == 1 { … }`), and there is no `else if` keyword — nest
> a fresh `if` in the `else` arm, exactly as the lexer does above
> ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §4).

---

## Structs

A struct is declared `struct Name { field: T, … }` and is `[proven]`
([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §2–§3). Fields have a
positional layout; you construct with `Name { field: value, … }` and read with `value.field`.
Tuple/positional structs (`struct P(i32, i32)`) are `[impl]`; a unit struct (`struct M;`) is
`[unsupported]`.

A complete, gate-proven struct (combined here with an `enum` — see the next section) is:

**Verified example** —
[`helixc/examples/hbs_sample_enum_struct.hx`](../../../helixc/examples/hbs_sample_enum_struct.hx)
(gate-checked to exit `129`; [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh):
`chk "$EX/hbs_sample_enum_struct.hx" 129`). The relevant declarations and use:

```helix
enum Kind { Circle, Square, Rectangle }

struct Shape {
    kind: Kind,
    a: i32,
    b: i32,
}
```

…and in `main`, fields are read in place (`circle.a`, `rect.b`, etc.) to compute the answer. Note
that a struct field can itself be an `enum` type (`kind: Kind`), and the program pattern-matches on
`s.kind` inside the `area_squared`/`perimeter` functions. (This example also carries an `@total`
attribute on those functions; attributes are covered below — `@total` is parsed and does not change
runtime behavior here.)

### Wide struct fields read and write full 64-bit (v1.3 V1)

There is one struct-specific width subtlety worth its own section, because it closed the *only*
silent-wrong bug the v1.2 spec carried. A struct field of type `i64`, `u64`, or `f64` is now read
**and** written at its full **64-bit** width, and an `f64` field is f64-typed so field arithmetic
routes through the SSE path. This is item **V1** of
[`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §2.1.

Before the fix (the v1.2 "M-3" bound), an `i64`/`u64` wide-field *read* silently truncated to the low
32 bits, and an `f64` wide-field read fail-closed with SIGILL. The fix is decl-time: the struct
declaration encodes an 8-byte scalar field, the read site emits a REX.W 8-byte load, and the
type system recovers the real element type
([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §2.1).

**Verified example** —
[`stage0/helixc-bootstrap/corpus_gen/V1_i64_wide_field.hx`](../../../stage0/helixc-bootstrap/corpus_gen/V1_i64_wide_field.hx)
(gate-checked to exit `50`; [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh):
`chk "$GENC/V1_i64_wide_field.hx" 50`). The declaration and the load-bearing lines:

```helix
struct Big { v: i64 }
fn main() -> i32 {
    let mut k: i64 = 0_i64;
    let mut i = 0;
    while i < 5 { k = k + 1000000000_i64; i = i + 1; }   // k = 5_000_000_000 (> 2^32) at runtime
    let b = Big { v: k };
    let g: i64 = 100000000_i64;                          // 1e8
    (b.v / g) as i32                                     // 5e9 / 1e8 = 50 (truncated would give 7)
}
```

The field `b.v` holds `5_000_000_000` (built at runtime so nothing constant-folds), and reading it
back must give `50` after dividing by 1e8; the pre-fix truncation gave `7`. The companion fixtures
extend the proof to `u64` (`V1_u64_wide_field.hx` → `50`), `f64` (`V1_f64_wide_field.hx` → `42`,
where the field read must equal an independent f64 local reference), and a **mixed** struct
(`V1_multi_wide_field.hx` → `42`) that places an `i64` at slot 0, an `f64` at slot 1, and an `i32` at
slot 2 — each read at its correct *offset and width*
([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh): `chk "$GENC/V1_u64_wide_field.hx" 50`,
`chk "$GENC/V1_f64_wide_field.hx" 42`, `chk "$GENC/V1_multi_wide_field.hx" 42`):

**Verified example** —
[`stage0/helixc-bootstrap/corpus_gen/V1_multi_wide_field.hx`](../../../stage0/helixc-bootstrap/corpus_gen/V1_multi_wide_field.hx)
(gate-checked to exit `42`). The declaration:

```helix
struct Mix { big: i64, d: f64, small: i32 }
```

> **Residual:** V1 is precisely a field-*width* fix, gated by the four `V1_*` programs above. It does
> not change the (already-proven) `i64`/`u64`/`f64` *scalar-local* arithmetic — only that wide
> *struct fields* now read/write full width
> ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §2.1).

> **For AI agents:** a struct holding an `i64`/`u64`/`f64` field is safe to read at full width *as of
> v1.3 V1*. If you are reasoning about older history or another checkout, note this was once a silent
> low-32 truncation (i64/u64) or a SIGILL (f64). Confirm against
> [`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §2.1 and the `V1_*`
> corpus rows.

---

## Enums

An enum is declared `enum Name { V1, V2(T1, T2), … }` and is `[proven]`, including **payload-bearing
variants** ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §2–§3). You
construct a variant with `Name::Variant` (or `Name::Variant(payload, …)`) and consume it with
`match`. Tag-only and payload variants both work; **struct-variants** (a variant whose payload is a
named-field record, like `V { x: i32 }`) are `[erased]`.

A complete enum program with both a tag-only sentinel variant and payload extraction:

**Verified example** —
[`helixc/examples/hbs_sample_option.hx`](../../../helixc/examples/hbs_sample_option.hx)
(gate-checked to exit `42`; [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh):
`chk "$EX/hbs_sample_option.hx" 42`):

```helix
enum Maybe { None, Some(i32) }
enum Pair { Empty, Cons(i32, i32) }

fn main() -> i32 {
    // Compute Some(40) + Some(2) by extracting payloads and summing.
    let m1 = Maybe::Some(40);
    let m2 = Maybe::Some(2);

    let v1 = match m1 {
        Maybe::Some(x) => x,
        Maybe::None => 0,
    };
    let v2 = match m2 {
        Maybe::Some(x) => x,
        Maybe::None => 0,
    };
    let total1 = v1 + v2;     // 42

    // Pair::Cons unpacking: a + b
    let p = Pair::Cons(15, 25);
    let total2 = match p {
        Pair::Cons(a, b) => a + b,    // 40
        Pair::Empty => 0,
    };

    // Mix: total1 (42) is an i32; total2 is 40. Final answer = 42 (we
    // pick the option-extraction result as the demo).
    let r = total1;
    r
}
```

`Maybe::Some(i32)` is a single-payload variant, `Pair::Cons(i32, i32)` a two-payload variant, and
`Maybe::None`/`Pair::Empty` are tag-only sentinels — all four constructed and then matched, with
payloads extracted in the match arms.

### `Result` / `Ok` / `Err` are user-defined enums, not builtins

A scope decision worth stating because the v0.1 draft and other ecosystems suggest otherwise:
`Ok`/`Err`/`Result` are **user-defined `enum`s in Helix, not compiler builtins**
([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §7, "v1.0 SCOPE
DECISIONS"). The proof is an inline corpus program that *declares* `Result` itself:

**Verified example** — `result_inline.hx`, generated and gate-checked to exit `42` inline in
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) (`chk "$CD/result_inline.hx" 42`):

```helix
enum Result { Ok(i32), Err(i32) }
fn main() -> i32 { let r = Result::Ok(42); match r { Result::Ok(x) => x, Result::Err(e) => e } }
```

> **For AI agents:** do not treat `Result`, `Ok`, or `Err` as built-in to the as-built `kovc` — they
> are ordinary user `enum`s. (An older Python-frontend example, `dogfood_16_result_basic.hx`, uses
> `Result<T, E>`/`Ok`/`Err` as identity-lowered scaffolding; that reflects the historical frontend,
> *not* the shipped from-raw compiler. The grounded, gate-proven idiom is the user-defined `enum`
> above.)

> **Residual:** `match` does **not** enforce exhaustiveness, and pattern **guards** (`pat if cond
> =>`) are parsed but historically were not enforced. Exhaustiveness is unenforced-by-design (locked
> by the `L3_nonexhaustive_bound` corpus row, which proves a non-exhaustive `match` is accepted →
> `42`); guard *enforcement* did land as hardening item H4
> ([`docs/HELIX_V1_1_HARDENING.md`](../../../docs/HELIX_V1_1_HARDENING.md)), but pattern matching and
> its residuals are the subject of the pattern-matching chapter, not this one. See
> [`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §4 / §7.

---

## Type annotations

A type annotation is the `: T` that follows a binding or parameter. Three places take one:

- **`let` bindings:** `let x: T = e;` is `[impl]`; the plain `let x = e;` (type inferred from the
  initializer) is `[proven]`
  ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §4). The width fixtures
  above are full of annotated lets — `let x: i64 = 42_i64;`, `let big: u64 = 5_000_000_000_u64;`,
  `let a: bf16 = 256.0_bf16;` — and they are how you *pin* a width: the annotation plus a matching
  suffix is the idiom the corpus uses throughout.
- **Function parameters and return type:** `fn name(p: T, …) -> Ret { body }`
  ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §3). The default return
  type is `i32` if `-> Ret` is omitted. Every example in this chapter shows this form.
- **Struct fields:** `field: T` inside a `struct` declaration (every struct above).

A `let` annotation and a literal suffix are *both* ways to assert a width; in the corpus they are
typically used together for clarity, e.g. `let x: i64 = 3_000_000_000_i64;`
([`L1_i64_big.hx`](../../../stage0/helixc-bootstrap/corpus_gen/L1_i64_big.hx)).

---

## Integer-literal typing and widening

How does a bare integer literal get its type? Two rules, both grounded:

1. **No suffix ⇒ `i32`.** A decimal literal with no width suffix is an `i32`
   ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §1). This is why
   `exit42.hx`'s `42` is an `i32`, and why `fn main() -> i32 { 100 / 5 / 2 }` is i32 arithmetic.

2. **A suffix selects the width**, and the lexer emits a *distinct token tag* per suffix so that
   codegen can choose the right instruction width — `_i64` becomes a 10-byte `movabs rax, imm64`
   rather than a 5-byte `mov eax, imm32`. This is documented at length inside `lex_int` in
   [`helixc/bootstrap/lexer.hx`](../../../helixc/bootstrap/lexer.hx) (the `K1.E1d-fix` comment
   explains exactly why mis-lexing `42_i64` as a plain `i32` led to a width-mismatch SIGILL — the
   one-byte lookahead at the `_` separator is what keeps the suffix visible).

The hardenings discussed above (H5 for `i64`, V2 for `u64`) are specifically about *widening a
literal whose magnitude exceeds what i32 can hold*. The honest summary is:

- A suffixed `i64` literal of full magnitude (including > 2³² like `5_000_000_000_i64`) decodes to
  its exact 64-bit value — the literal text is carried to codegen and re-decoded full-width via an
  i32 16-bit-limb path ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md)
  §1; [`docs/HELIX_V1_1_HARDENING.md`](../../../docs/HELIX_V1_1_HARDENING.md) H5). Proven by
  `L2_i64_bigger.hx` (→ `50`).
- A suffixed `u64` literal up to 2⁶⁴−1 likewise decodes full-range *unsigned*
  ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §1, V2). Proven by
  `V2_u64_lit_near_max.hx` (→ `42`).

> **Residual (honest scope of "widening"):** what is proven is the **literal-decode** path — a
> suffixed wide literal materializes its exact value. This is *not* the same as general numeric
> type-inference or implicit cross-width promotion. There is **no implicit conversion between numeric
> types** (the v0.1 draft lists "implicit conversions between numeric types" as *outside* the general
> surface, and the as-built spec requires an explicit `as` cast: `e as T`, which does int↔int
> width-correct, int↔float, and float↔float —
> [`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §4). And recall the
> 16-bit-float boundary *traps* on a mixed operand rather than widening. When you need a different
> width, write the suffix or an `as` cast — do not expect the compiler to silently widen.

> **For AI agents:** to pin an integer's width, prefer the explicit suffix on the literal *and/or* a
> `: T` annotation on the binding; for cross-width conversion use `e as T`. The lexer emits a
> width-specific token tag for each suffix
> ([`helixc/bootstrap/lexer.hx`](../../../helixc/bootstrap/lexer.hx), `lex_int`), and a wrong/missing
> suffix can change the emitted instruction width. Do not rely on implicit numeric promotion — it
> does not exist (the 16-bit-float case actively traps).

---

## A note on attributes you will see on typed code

Two attributes appear on the typed examples above and deserve a one-line clarification so you don't
over-read them:

- `@pure` is `[proven]` and is required on functions consumed by the autodiff (`grad`) rewrites
  ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §3, §5). You see it on
  the `ieee754.hx` helpers and the lexer's classifier functions.
- `@total` (seen on `hbs_sample_enum_struct.hx`) is among the attributes the as-built `kovc` parses;
  the spec's attribute list is `@pure` `[proven]`, `@kernel` `[impl]`, `@autotune(…)` `[impl]`,
  `@deprecated`/`@since` `[impl]`, with Rust `#[…]` attributes skipped at lex
  ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §3). These are
  declarations on *functions*, not types, and they do not change the runtime behavior of the
  examples here.

---

## What this chapter established

- Helix's integer widths are `i8/i16/i32/i64` (signed) and `u8/u16/u32/u64` (unsigned), with `i32`
  the default; `i64`/`u64` have full 64-bit semantics, and `usize` is an erased alias with no
  distinct width.
- Wide integer **literals** are proven across the boundary that broke earlier toolchains:
  `5_000_000_000_i64` (> 2³², H5) and `u64` literals up to 2⁶⁴−1 (V2), each decoded full-width from
  the literal's source text and gate-locked (`L2_i64_bigger.hx` → 50, `V2_u64_lit_near_max.hx` → 42).
- The float widths are `f32`, `f64`, `bf16`, and `f16`; bf16/f16 arithmetic *computes* with
  round-to-nearest-even (bf16 on SSE2, f16 on F16C) and is bit-exact-gated — and a 16-bit float mixed
  with a wider operand **traps** rather than widening. The v0.1 draft's `fp8`/`mxfp4`/`nvfp4`/
  `ternary` are design-vision entries, not implemented.
- `bool` is an i32 0/1 with **no implicit truthiness** — conditions need explicit comparisons.
- `struct` (named fields, positional layout, full-width wide scalar fields as of V1) and `enum`
  (tag-only + payload variants; `Result`/`Ok`/`Err` are user-defined) are both proven, each with a
  complete gate-checked example.
- Type annotations live on `let`, function params/return, and struct fields; widening is a
  *literal-decode* property, not implicit numeric promotion — cross-width conversion is the explicit
  `as` cast.

---

**Next:** [Functions, control flow & pattern matching](03-functions-control-flow.md) — how the typed
values above flow through `fn`, `if`/`while`/`match`, and the pattern forms (including the guard
enforcement noted as a residual here). For the trust/performance posture behind any GPU or
self-host claim referenced above, see
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) and Part VIII.
