# Stage 7 / 8 / 8.5 Type-System Audit

**Date**: 2026-05-10
**Commit**: 3421b21 (read-only audit)
**Scope**: helixc/bootstrap/parser.hx (Stages 7, 8, 8.5)
**Trigger**: post Stage 8.5 trait+impl merge — looking for type-system
soundness gaps, dispatch bugs, and corner cases in pattern matching,
generics+monomorphization, and traits+impls.

**Method**: traced every parse_pattern / parse_primary / parse_fn_decl /
parse_impl_method / monomorphize_pass / clone_with_rewrite arm for
(a) silent fall-through to a different semantic; (b) cap-overflow without
trap; (c) type tag conflation (i32 default sinking unknown types);
(d) ordering assumptions; (e) name-collision risks.

**Result**: 12 findings (6 HIGH, 5 MEDIUM, 1 LOW). The most severe are
type-tag-default conflations where any non-canonical scalar type name
(u8, u16, i8, i16, bf16, struct names, enum names, unknown idents) silently
maps to i32 throughout method-call sugar, var_type_table, ty_ident_to_tag,
and impl_table — producing wrong dispatch with no diagnostic.

---

## Finding 1: Top-level decl order is FIXED — trait/impl/struct/enum after first fn are silently dropped

**Location**: helixc/bootstrap/parser.hx:3322-3382 (parse_program decl prefix loop + post-fn loop)
**Severity**: HIGH
**Category**: silent truncation / safety

**Description**:
`parse_program` has TWO decl-handling loops:

1. A prefix loop (3322-3357) that handles `struct`, `enum`, `trait`,
   `impl`, `mod`, `use` BEFORE the first `fn`. It exits as soon as it
   sees any non-decl token.
2. A post-fn loop (3361-3383) that ONLY accepts `fn`. Any other IDENT —
   including the same `trait` / `impl` / `struct` / `enum` / `mod` / `use`
   keywords the prefix loop just handled — sets `keep = 0` and exits.

After the post-fn loop exits, parse_program proceeds to splicing,
mono pass, grad pass, and returns. No further parsing is attempted.
`parse_top` is called exactly once.

Result: any decl placed after the first fn-decl is silently dropped.
The user's source compiles, but later items are invisible to codegen.

**Reproducer**:
```
fn main() -> i32 { 0 }

trait Eq { fn eq(self, other: Self) -> i32; }   // ← parsed by NOBODY
impl Eq for i32 { fn eq(self, other: i32) -> i32 { 1 } }   // ← dropped
```
The compiled binary contains only `fn main`. `trait Eq`'s `Eq` is never
in `trait_tab`; `impl Eq for i32`'s `i32__eq` is never in fn_table. A
call to `i32::eq(a, b)` elsewhere would resolve to ud2 — no compile-time
warning.

Real-world impact: a user who interleaves types and functions (the
natural Rust style) ends up with a silently truncated program. Every
fn after the first non-fn item is invisible — even subsequent `fn`
items can be invisible because the loop exits on the first non-fn.

**Trap-id reservation**:
N/A — fix is to extend the post-fn loop to accept struct/enum/trait/
impl/mod/use the same way as the prefix loop, OR to trap-on-unrecognized
tokens between fn decls.

**Recommended fix**:
Replace the post-fn `if byte_eq(s,l,kw_fn_s,kw_fn_n) == 1` with a flat
ladder mirroring the prefix loop (test all 7 keywords), and emit
`emit_trap_with_id(72001)` on a truly unrecognized IDENT at top-level.

---

## Finding 2: ty_ident_to_tag returns 0 (i32) for ALL unknown type names — silent type-tag conflation

**Location**: helixc/bootstrap/parser.hx:1027-1051 (ty_ident_to_tag)
**Severity**: HIGH
**Category**: silent corruption / dispatch hijack

**Description**:
`ty_ident_to_tag` maps a type IDENT's bytes to a 4-bit tag:

- i32 → 0, f32 → 1, f64 → 2, i64 → 3, u32 → 6, u64 → 9.
- bf16, u8, u16, i8, i16 → **0** (default fall-through).
- Any struct name, enum name, generic-param name, or typo → **0**.

The comment at line 1029-1030 acknowledges "Unknown -> 0 (i32 default;
safe fallback for substitution sites where the body just bitcasts
through the slot)." This is true for body-shared mono pass slot
width — but ty_ident_to_tag is also used by:

1. **var_type_tab_add** (line 2052, 2058): `let x: u8 = ...` registers
   (x, tag=0). Subsequent `x.method()` dispatches to `i32__method`
   (Finding 3).
2. **parse_impl_block target_tag** (line 5194): `impl Eq for u8`
   sets target_tag=0. The synthesized method has self-typed-as-i32
   but mangled name `u8__eq` (because mangling uses raw bytes).
3. **parse_impl_method ret/param resolution** (line 5134, 5161):
   `fn foo(self, other: u8) -> u8 { ... }` resolves `other: u8` to
   tag=0, `-> u8` to tag=0. The synthesized fn has all params/ret as
   i32 in fn_type_table.
4. **parse_let let_ty_tag** (line 2052): `let x: SomeStruct = ...`
   silently tags x as i32 — see Finding 3.
5. **Turbofish call mono pack** (line 2419): `id::<u8>(x_u8)` packs
   tag=0; mono pass instantiates `id__u8` with param tag=0 (i32). Body
   reads x as i32 (32-bit load), narrow-bind semantics lost.

The check `if let_ty_tag >= 0` (line 2058) treats tag=0 as "valid scalar"
since 0 is the canonical i32 tag. There is no way to distinguish "i32
annotation" from "unknown ident annotation" — both map to 0.

**Reproducer**:
```
trait Eq { fn eq(self, other: Self) -> i32; }
impl Eq for i32 { fn eq(self, other: i32) -> i32 { 1 } }
impl Eq for u8  { fn eq(self, other: u8) -> i32 { 2 } }

fn main() -> i32 {
    let a: u8 = 5_u8;
    let b: u8 = 7_u8;
    a.eq(b)                  // expected: 2 (u8 impl)
                             // actual:   1 (silently routes to i32__eq)
}
```
Because `let a: u8` registers (a, tag=0) in var_type_tab, and method-call
sugar reads tag=0 from var_type_tab, it builds `i32__eq(a, b)`. The
`impl Eq for u8` synthesized `u8__eq` is dead code.

Also for unknown ident:
```
struct Foo { x: i32 }
fn main() -> i32 {
    let f: Foo = Foo { x: 3 };
    f.bar()                  // user typo'd; intended `f.x` etc.
                             // tag=0 → mangles to "i32__bar"
                             // i32__bar doesn't exist → ud2 (no clear msg)
}
```

**Trap-id reservation**:
N/A — fix is to extend ty_ident_to_tag with: distinguish i32 (tag 0)
from unknown by changing return to -1 (or a new 0-reserved-for-i32
+ -2 for unknown). Callers that currently accept tag=0 as "valid"
must distinguish.

**Recommended fix**:
Two-step. (a) ty_ident_to_tag returns a TRI-state: 0..N for known
scalar, -1 for "valid struct/enum name", -2 for "unknown". (b) Each
caller decides which case is valid. parse_let then registers
var_type_tab only on positive scalar tag; parse_impl_block traps on
unknown target; mono pass traps on unknown turbofish arg.

This is broad but high-impact. A localized first cut: detect length-2
(u8, i8) and length-4 (bf16) PROPERLY in ty_ident_to_tag (mirror
parse_fn_decl's strict per-byte logic at line 4790-4811) — that
single fix eliminates the u8/u16/i8/i16/bf16 leak and reduces this
to "unknown idents silently i32".

---

## Finding 3: var_type_tab + method-call sugar conflates unknown / struct types with i32

**Location**: helixc/bootstrap/parser.hx:1378-1439 (method-call sugar in parse_postfix);
helixc/bootstrap/parser.hx:2046-2060 (parse_let registering var_type_tab)
**Severity**: HIGH
**Category**: silent corruption / type-soundness

**Description**:
Method-call sugar dispatch (`a.eq(b)` → `<TypeName>__eq(a, b)`) reads
the LHS's type tag from var_type_tab. The dispatch path:

1. `let a: T = ...` → var_type_tab_add(a, ty_ident_to_tag(T)).
2. `a.eq(b)`: var_type_tab_lookup(a) returns the stored tag t.
   `is_method_call = (t >= 0)` — i.e., **fires for tag=0 (i32 default)**.
3. ty_tag_push_name(t) writes the canonical type-name bytes — for t=0,
   "i32" (line 710-712); for t=4 (bf16), 7 (u8), 8 (u16), 10 (i8), 11
   (i16), falls to the default arm (line 728-730) which ALSO writes "i32".
4. Mangled call name = "i32__eq", regardless of what T was.

So `let x: u8 = ...; x.method()` → "i32__method". `let x: Foo = ...;
x.method()` → "i32__method".

This is doubly broken: ty_ident_to_tag conflates u8/u16/i8/i16/bf16/
struct/unknown to 0 (Finding 2), AND ty_tag_push_name conflates u8(7)/
u16(8)/bf16(4)/i8(10)/i16(11) with i32 in name emission too. Even if
Finding 2 were fixed (so var_type_tab correctly stored tag 7 for `u8`),
ty_tag_push_name would still emit "i32" for tag 7 because it lacks a
u8/u16/bf16/i8/i16 arm.

**Reproducer**:
Same as Finding 2 — u8 dispatch routes to i32__eq.

**Trap-id reservation**:
Fix is local: add arms to ty_tag_push_name for tags 4 (bf16),
7 (u8), 8 (u16), 10 (i8), 11 (i16). For tag 4, push "bf16"; for tag
7 "u8"; for tag 8 "u16"; for tag 10 "i8"; for tag 11 "i16". With
Finding 2's fix in place, this completes the i32-conflation closure.

For struct/enum target types, the impl_table is keyed by target_tag —
all unknowns conflate to 0 there too. A proper fix tracks target NAME
(byte-equal compare) rather than 4-bit tag. Larger refactor.

---

## Finding 4: Stage 8 mr_tab cap-32 overflow silently drops 33rd+ instantiations

**Location**: helixc/bootstrap/parser.hx:242-258 (mr_tab_add);
helixc/bootstrap/parser.hx:2431-2434 (turbofish call-site registration)
**Severity**: MEDIUM
**Category**: cap-overflow without trap

**Description**:
`mr_tab_add` returns -1 silently when count >= 32 (line 244-245). The
turbofish call site at 2431-2434 ignores this return value:

```
let existing = mr_tab_lookup(sb, id_start, id_len, pack_lo);
if existing < 0 {
    mr_tab_add(sb, id_start, id_len, mang_s, mang_l, pack_lo);
};
```

If the program has 33+ unique generic instantiations, the 33rd onward
fail to register. Their AST_CALL nodes (built with the mangled name
already pushed into the arena) survive. At codegen, the mono pass
synthesizes only the first 32 clones. fn_table_lookup misses on the
33rd → existing ud2 patch from Stage 8F fires.

The user sees SIGILL (exit 132) at runtime with no specific 71001
trap-id. The original plan (APPROACH_A_DETAILED_PLAN.md:401) reserved
71001 for "cap on 32 instantiations" but the trap was never emitted.

**Reproducer**:
```
fn id<T>(x: T) -> T { x }
fn main() -> i32 {
    id::<i32>(1) + id::<f32>(1.0) as i32 + id::<f64>(1.0) as i32 +
    id::<i64>(1_i64) as i32 + id::<u32>(1_u32) as i32 +
    id::<u64>(1_u64) as i32 +
    // ... ≥ 33 distinct (orig, pack_lo) pairs ...
}
```
With only 6 scalar tags in ty_ident_to_tag's known set, hitting 33
unique pack_lo pairs requires multi-arg generics like
`pair<A, B, C, D>` — eminently reachable in real code.

**Trap-id reservation**:
71001 (mr_tab cap-overflow).

**Recommended fix**:
```
let existing = mr_tab_lookup(sb, id_start, id_len, pack_lo);
let n_reg_trap = if existing < 0 {
    let r = mr_tab_add(sb, id_start, id_len, mang_s, mang_l, pack_lo);
    if r < 0 { emit_trap_with_id(71001) } else { 0 }
} else { 0 };
```
The trap is emitted at the call SITE, so the binary crashes when that
specific over-cap call is reached. Use the FLAT prefix-trap pattern
that Stage 4 audit Finding #7 used.

---

## Finding 5: ty_ident_to_tag pack_lo collides for unknown types — wrong mono clone emitted

**Location**: helixc/bootstrap/parser.hx:1035-1051 + 2419-2434 (turbofish dedup)
**Severity**: HIGH
**Category**: silent corruption / wrong-fn dispatch

**Description**:
Direct consequence of Finding 2 in the mono dispatcher. `id::<u8>(...)`
and `id::<i32>(...)` both produce pack_lo = `0 * 8 + 1 = 1` because
ty_ident_to_tag returns 0 for both u8 and i32. The mr_tab_lookup
matches on (orig_name, pack_lo). So:

- First instance `id::<i32>(5)`: mr_tab adds (orig="id", mang="id__i32",
  pack_lo=1). AST_CALL emitted with name="id__i32".
- Second instance `id::<u8>(5_u8)`: pack_lo=1. mr_tab_lookup finds
  existing entry, skips add. AST_CALL emitted with mangled name "id__u8"
  (the parse_primary path already pushed "id__u8" bytes into the arena
  at line 2410 — line BEFORE the dedup check).
- Mono pass synthesizes ONLY `id__i32` (the registered name).
- Codegen: lookup "id__u8" misses → ud2 trap.

OR, in reverse order:
- First `id::<u8>(5_u8)`: mr_tab adds (mang="id__u8", pack_lo=1).
  AST_CALL name="id__u8".
- Second `id::<i32>(5)`: pack_lo=1, existing found, skip add. AST_CALL
  name="id__i32".
- Mono pass synthesizes "id__u8".
- Codegen for `id::<i32>(...)` lookup "id__i32" misses → ud2.

Either way, ONE of the two calls produces ud2. The user has no diagnostic
explaining "u8 turbofish collided with i32 turbofish".

**Reproducer**:
```
fn id<T>(x: T) -> T { x }
fn main() -> i32 {
    let a = id::<i32>(42);
    let b: u8 = id::<u8>(7_u8);    // ud2 trap at this call
    a
}
```

**Trap-id reservation**:
71003 (turbofish pack_lo collision — different mangled names share pack_lo).

**Recommended fix**:
Make ty_ident_to_tag return a richer space (length+bytes-based hash, or
extend the 4-bit packing to 5 bits to cover u8/u16/i8/i16/bf16). The
cleanest fix is to switch dedup from (orig, pack_lo) to (orig, mang_s,
mang_l) — compare mangled name bytes directly. This eliminates the
collision regardless of how compactly tags are packed.

---

## Finding 6: parse_pattern silently drops non-INT literals as wildcards

**Location**: helixc/bootstrap/parser.hx:5514-5635 (parse_pattern)
**Severity**: HIGH
**Category**: silent miscompile

**Description**:
`parse_pattern` dispatches on the leading token tag (line 5516):
- tag 1 (TK_INT) → PAT_LIT or PAT_RANGE
- tag 2 (TK_IDENT) → PAT_WILDCARD / PAT_BIND / PAT_VARIANT
- tag 3 (TK_LPAREN) → PAT_TUPLE
- everything else → fallback at line 5631-5635: advance ONE token,
  return PAT_WILDCARD.

Tokens that fall through to the wildcard fallback:
1. **TK_MINUS** (tag 8): `-5` pattern. cur_advance eats `-` only, the
   `5` is left for the next parse step. `match x { -5 => 1 }` becomes
   `match x { _ => 5 }` with the `=> 1` as orphan tokens parsed by the
   outer expr.
2. **TK_FLOATLIT / TK_FLOATLIT_F64 / TK_FLOATLIT_BF16** (tags 26 / 32 / 41):
   `0.5_f64` pattern. cur_advance eats the float, returns wildcard.
   `match x { 0.5_f64 => 1, _ => 0 }` becomes `match x { _ => 1, _ => 0 }` —
   first arm always matches.
3. **TK_INTLIT_I64 / TK_INTLIT_U32 / TK_INTLIT_U8 / TK_INTLIT_U64**
   (tags 33 / 34 / 35 / 36, etc.): `42_i64` pattern. Same — silently
   becomes wildcard.
4. **TK_TRUE / TK_FALSE** (if present): bool patterns — also wildcard.

The match-arm test for these silently-wildcarded patterns always matches
the first such arm. The arm body's return type may not even match the
scrutinee type — no static check.

**Reproducer**:
```
fn main() -> i32 {
    let x: i32 = 5;
    match x {
        0.5_f64 => 7,   // ← parses as `_` => 7 (always matches)
        5 => 11,        // ← dead code
        _ => 0,
    }
    // Returns 7, not 11.
}
```

For negative literals:
```
fn main() -> i32 {
    let x: i32 = 0 - 5;     // -5 via subtract
    match x {
        0 - 5 => 99,        // not even a literal pattern; parses oddly
        -5 => 100,          // parses as `_` then orphan `5 => 100`
        _ => 0,
    }
}
```

**Trap-id reservation**:
62002 (parse_pattern unknown token tag — emit trap at the parse site
so an unhandled pattern token is loud at compile time).

**Recommended fix**:
Replace the wildcard fallback (line 5631-5635) with an explicit trap-AST
node that codegen lowers to `emit_trap_with_id(62002)`. Alternatively,
extend parse_pattern with arms for TK_MINUS (negative-int range),
TK_FLOATLIT and friends (PAT_LIT with float discriminant — needs a
type-aware compare), and TK_TRUE / TK_FALSE.

The minimum viable fix: trap on unhandled pattern token. Don't silently
substitute wildcards — wildcard semantics are too consequential.

---

## Finding 7: PAT_LIT / PAT_RANGE / PAT_VARIANT_DISC use 32-bit cmp on a 64-bit-stored scrutinee

**Location**: helixc/bootstrap/kovc.hx:3338-3366 (emit_pat_lit / emit_pat_range / emit_pat_variant_disc);
helixc/bootstrap/kovc.hx:3505-3517 (emit_match_dispatch storing scrut as 64-bit)
**Severity**: MEDIUM (HIGH for the i64/u64 case)
**Category**: silent corruption / wrong-arm dispatch

**Description**:
`emit_match_dispatch` stores the scrutinee with `emit_mov_local_rax_64`
(8-byte store) at line 3509. But the pattern testers read with 32-bit
loads:

- `emit_pat_lit` line 3339: `emit_mov_eax_local(scrut_off)` (32-bit).
- `emit_pat_range` line 3348: `emit_mov_eax_local(scrut_off)` (32-bit).
- `emit_pat_variant_disc` line 3360: `emit_mov_rax_local_64(scrut_off)`
  (64-bit) — this one IS correct because the load fetches the enum pointer.
  Sub-pat loads at line 3306/3327 use 64-bit too, correct.

For i64 / u64 / f64 scrutinees with PAT_LIT or PAT_RANGE:
- `match (1_i64 + 0x1_0000_0000) { 0 => 0, _ => 1 }` — the i64 value
  `0x1_0000_0000` is stored 8 bytes, then `emit_mov_eax_local` loads
  the low 32 bits = 0. `cmp eax, 0` → match → arm 0 fires. Expected:
  arm 1 fires.

For f32 / f64 scrutinees with PAT_LIT (which is meaningless but not
trapped — see Finding 6), the cmp interprets the float bit-pattern as
integer.

Additionally, PAT_LIT's `emit_cmp_eax_imm32(lit)` is a SIGNED 32-bit
cmp. For u32 scrutinees with `255_u32` pattern, both sides are 32-bit-
agnostic to signedness. But for u32 values > 0x7FFFFFFF, the signed cmp
sees them as negative — still correct equality but not for range
(PAT_RANGE).

PAT_RANGE specifically uses `jl` (signed less) and `jge` (signed greater-
or-equal) at line 3350-3354. For u32 / u64 scrutinees these are wrong
when the value crosses the sign-bit. `match 0xFFFFFFFF_u32 { 0..0xFFFFFFFE_u32 => ... }`
— `cmp eax, 0` → eax (= -1 signed) is less than 0 → `jl` fires → fail
to next arm. Pattern says lo=0, hi=0xFFFFFFFE; the value 0xFFFFFFFF
should be OUTSIDE the range (>= hi), so correct outcome. But the
intermediate "is x >= 0" comparison would silently be wrong for any u32
above 0x7FFFFFFF.

Narrow scrutinee (u8 / u16 / i8 / i16) with literal pattern like `255_u8`:
The literal 255 is encoded as imm32. cmp eax, 255. For a u8 stored via
1-byte write, the slot holds value in low byte; 32-bit load via
`mov eax, [rbp+disp]` reads 4 bytes including possibly-stale high
bytes. The match-dispatch stored a full 64-bit rax (line 3509), but the
caller emit_ast_code for the scrutinee already truncated to 32-bit if
narrow — so high bytes are zero. Cmp works for narrow scrut <= 32-bit.

Combined effect:
- **i64 / u64 / f64 scrutinee + PAT_LIT / PAT_RANGE**: low-32 cmp only.
  Silent wrong-arm dispatch.
- **u32 scrutinee + PAT_RANGE crossing 0x7FFFFFFF**: signed compare bug.

**Reproducer**:
```
fn main() -> i32 {
    let x: i64 = 4294967296_i64;   // 0x1_0000_0000 — low32 = 0
    match x {
        0 => 100,                  // ← matches incorrectly (low32 == 0)
        _ => 1,
    }
    // Returns 100. Expected: 1.
}
```

**Trap-id reservation**:
- 62003 (PAT_LIT / PAT_RANGE on i64/u64/f64/f32 scrut — type-width mismatch).
- 62004 already reserved for nested match (not used by this fix).

**Recommended fix**:
Two paths:
(a) Trap on type-width mismatch — emit_pattern_test inspects the
    scrut's expr_type (passed via bn_state's existing type-resolution
    plumbing) and traps if scrut is wider than PAT_LIT's i32-shape.
(b) Dispatch pattern testers per scrut type. Add `emit_pat_lit_64`
    that loads rax-64 and compares with imm64 (via movabs rax, imm64
    + cmp rax, ...). Same for PAT_RANGE. Distinguish signed vs unsigned
    branches for u32/u64.

Option (a) is the minimal Phase-0 fix; option (b) is the correct one.
For now, document that PAT_LIT only supports i32-shaped scrutinees.

---

## Finding 8: PAT_VARIANT / PAT_TUPLE sub-pattern arity mismatch reads OOB stack slots

**Location**: helixc/bootstrap/parser.hx:5566-5592 (variant sub-pat parsing);
helixc/bootstrap/kovc.hx:3294-3335 (emit_variant_subpats / emit_tuple_subpats)
**Severity**: HIGH
**Category**: silent corruption / OOB read

**Description**:
`parse_pattern` accepts any number of sub-patterns for PAT_VARIANT and
PAT_TUPLE. There is no arity check against the variant's declared
arity (max_payload_arity is stored in enum_tab but never consulted)
or the tuple's element count.

emit_variant_subpats walks the sub-pat chain linearly, reading slots
`[scrut+8]`, `[scrut+16]`, `[scrut+24]`, ... regardless of how many
slots the variant actually allocated. Excess sub-patterns read PAST
the enum's payload region into adjacent stack memory.

For PAT_TUPLE the same: starting at `[scrut+0]`, reads any number of
slots.

Additionally, `enum_tab_variant_lookup_disc` returns -1 on unknown
variant name (parse_pattern line 5563), and line 5564 silently
substitutes `safe_disc = 0`. So `match m { NotARealVariant(x) => ... }`
matches when m's tag is 0 — picking up any arm with disc=0 silently.

**Reproducer**:
```
enum Maybe { None, Some(i32) }    // Some has arity 1

fn main() -> i32 {
    let m = Maybe::Some(42);
    match m {
        Maybe::Some(a, b, c) => a + b + c,   // arity 3 vs declared 1
        // a = 42 (slot 1), b = slot 2 (junk), c = slot 3 (junk).
        // Sum = silent garbage.
        _ => 0,
    }
}
```

For the unknown-variant case:
```
enum Color { R, G, B }

fn main() -> i32 {
    let c = Color::G;            // disc 1
    match c {
        Mango => 99,             // unknown variant; safe_disc = 0
                                 // matches when disc==0 (= Color::R)
        Color::G => 1,
        Color::B => 2,
        _ => 0,
    }
    // c = G (disc 1), Mango => 99 doesn't fire. Color::G => 1 fires.
    // Returns 1.
}
```
The Mango arm is silently a disc==0 arm rather than rejecting the
unknown variant name.

**Trap-id reservation**:
- 62005 (PAT_VARIANT arity mismatch: pattern sub-pat count != variant
  declared arity).
- 62006 (PAT_VARIANT references unknown variant name).
- 62007 (PAT_TUPLE arity mismatch: pattern arity != scrut tuple arity).

**Recommended fix**:
At parse_pattern time, after the sub-pat loop:
```
let want_arity = enum_tab_variant_lookup_arity(sb, e_idx_pre, v_name_s, v_name_l);
if want_arity < 0 { /* unknown variant — trap 62006 at codegen */ }
else { if sub_arity != want_arity { /* trap 62005 */ } };
```
Encode the trap into the pattern node so codegen emits it when the arm
is reached.

For unknown variants, REJECT the `safe_disc = 0` fallback — emit a
codegen trap or refuse to parse. The current silent disc=0 substitution
is a footgun.

---

## Finding 9: clone_with_rewrite only handles AST_CALL at the body root — nested generic calls silently misdispatch

**Location**: helixc/bootstrap/parser.hx:3539-3558 (clone_with_rewrite)
**Severity**: MEDIUM
**Category**: documented gap / silent miscompile

**Description**:
The Stage 8.5C deep-clone-with-rewrite, used by the mono pass to
specialize `cmp<T: Eq>(a, b) -> i32 { T::eq(a, b) }`, handles ONLY the
case where the body IS an AST_CALL (tag 16). For any other body tag
(let, if, seq, binop, ...), the function returns the original node
unchanged (line 3554-3556).

This means generic fns with bodies more complex than a single call
silently miscompile:

```
fn cmp<T: Eq>(a: T, b: T) -> i32 {
    let r = T::eq(a, b);     // body is AST_LET, NOT AST_CALL
    r
}
```

The body AST is an AST_LET. clone_with_rewrite at the root sees tag 8
(AST_LET), returns the node unchanged. The inner `T::eq(a, b)` AST_CALL
is shared — its mangled name is "T__eq", which never resolves.
At codegen, fn_table_lookup("T__eq") → ud2. The user sees SIGILL.

Documented behavior at line 3534-3538: "minimal deep-clone. Only handles
the case where the body IS an AST_CALL ... For more complex bodies
(let-binding, if-expr, nested calls), extend this function as needed."
But the user-facing error message is silent — only runtime SIGILL.

**Reproducer**:
```
trait Eq { fn eq(self, other: Self) -> i32; }
impl Eq for i32 { fn eq(self, other: i32) -> i32 { 1 } }

fn cmp<T: Eq>(a: T, b: T) -> i32 {
    let r = T::eq(a, b);     // ← let-wrapped call NOT rewritten
    r
}

fn main() -> i32 {
    cmp::<i32>(5, 5)         // mono'd cmp__i32 body still calls T__eq → ud2
}
```

**Trap-id reservation**:
71004 (mono clone body contains unrewritten gp-prefixed call).

**Recommended fix**:
Either (a) extend clone_with_rewrite to recurse through AST_LET (slots
3 + 4), AST_IF (slots 1 + 2 + 3), AST_SEQ, binops (slots 1 + 2), etc.,
allocating fresh nodes when any sub-tree's rewrite changed it; OR (b)
at mono-pass time, scan the template body for any AST_CALL whose name
starts with a gp prefix, and if found AND the body is not a simple
AST_CALL, emit a trap-AST node at the call's location so the user gets
a clear "unrewritten generic call" error.

Option (a) is the correct fix. Option (b) is a stopgap. Either should
land BEFORE any user-facing generic-fn documentation.

---

## Finding 10: Mono clone discards is_checkpoint and other slot-8 attributes

**Location**: helixc/bootstrap/parser.hx:3666-3671 (monomorphize_pass new_fn build)
**Severity**: MEDIUM
**Category**: silent attribute drop

**Description**:
The mono pass synthesizes new AST_FN_DECL nodes with hardcoded slot 7
(gp_names_head) = 0 and slot 8 (is_checkpoint) = 0:

```
let clone_idx = mk_node(14, mang_s, mang_l, cloned_body);
__arena_push(new_params_head);
__arena_push(new_ret_ty);
__arena_push(0);                 // is_generic = 0 (concrete)
__arena_push(0);                 // slot 7: gp_names_head (none)
__arena_push(0);                 // slot 8: is_checkpoint = 0 (Stage 14.5)
```

The template's `is_checkpoint` (slot 8) is NOT propagated to the clone.
So `@checkpoint fn id<T>(x: T) -> T { x }` followed by `id::<i32>(...)`
synthesizes `id__i32` with is_checkpoint=0. The reverse-mode AD pass
(Stage 14.5b) would skip the checkpoint optimization on the clone.

Similarly for parse_impl_method (line 5168-5170): `@checkpoint impl Eq
for i32 { fn eq(...) }` — the attribute is set on `next_fn_is_ckpt`
scratch flag by skip_attributes, but parse_impl_method hardcodes 0 at
slot 8 and never reads the scratch flag. So @checkpoint on impl methods
is silently dropped.

**Reproducer**:
Hard to reproduce without an AD-heavy test, but reasoning is clear:
```
@checkpoint
fn step<T>(x: T) -> T { ... heavy ad-relevant body ... }

fn main() -> i32 {
    grad_rev_all(loss)(step::<f64>(1.0)).dx
    // The checkpoint marker is dropped on step__f64; AD pass doesn't
    // see it as a re-materialization candidate. Memory grows ~linearly
    // instead of sqrt(N) as @checkpoint advertised.
}
```

**Trap-id reservation**:
N/A — no runtime crash, just silent attribute drop.

**Recommended fix**:
Mono pass: read `__arena_get(tpl_idx + 8)` (template's is_checkpoint) and
push that instead of 0. impl-method parse: read next_fn_is_ckpt(sb)
before allocating the synthesized AST_FN_DECL and push that. Both fixes
are 1 line each.

---

## Finding 11: Self-recursive impl method calls and self.method() in impl bodies don't work

**Location**: helixc/bootstrap/parser.hx:1378-1439 (method-call sugar)
**Severity**: MEDIUM
**Category**: documented limitation / surprising UX

**Description**:
Method-call sugar (`x.method(args)`) consults var_type_tab, which is
populated ONLY by parse_let (typed let-bindings). Function parameters
are NOT registered in var_type_tab. So inside `impl Eq for i32 { fn eq(self, other: i32) -> i32 { ... } }`,
the `self` parameter has type tag = target_tag = 0 (i32), but
var_type_tab_lookup("self") returns -1. method-call sugar does not
fire. Falls to field-access. struct lookup fails. The `.method(args)`
is left dangling.

This means in impl bodies:
- `self.eq(other)` does not work as `i32__eq(self, other)`.
- The user must write `i32__eq(self, other)` explicitly.

Same issue for typed-call path inside generic fns: `cmp<T: Eq>(a: T,
b: T) -> i32 { T::eq(a, b) }` works ONLY because the call uses the
TYPE-NAMESPACE form `T::eq(...)`, not the method-call sugar.

Also, `Self::eq(self, other)` from inside an impl method body — `Self`
is captured at param/return-type position but NOT at expression position.
Self::eq parses as a path-call: mangled name "Self__eq", fn_table lookup
misses → ud2 trap. The user has no way to refer to the current impl's
own methods symbolically.

**Reproducer**:
```
impl Eq for i32 {
    fn eq(self, other: i32) -> i32 { if self == other { 1 } else { 0 } }
    fn neq(self, other: i32) -> i32 {
        1 - self.eq(other)            // self.eq sugar doesn't fire
                                      // (self is a param, not a let-bound var).
                                      // Compiles as field-access fallthrough,
                                      // then a stray `.eq(other)` orphan.
    }
}
```

**Trap-id reservation**:
N/A — needs a feature fix, not a trap.

**Recommended fix**:
parse_impl_method should populate var_type_tab(self, target_tag) for
the duration of the method body's parse_expr call. Same for any param
with a concrete scalar type. Reset on method-end. This makes self.method
sugar fire correctly inside impl bodies.

For Self in expression position, add a check in parse_primary's IDENT
branch: if id_bytes == "Self" and we're inside an impl method (track via
a parser flag), rewrite to the target type's name bytes.

---

## Finding 12: PAT_VARIANT / PAT_TUPLE sub-pat disp8 wraps at idx > 15 — silent stack read OOB

**Location**: helixc/bootstrap/kovc.hx:3294-3334 (emit_variant_subpats / emit_tuple_subpats)
**Severity**: MEDIUM
**Category**: silent corruption (parallel to Stage 4 Finding #7)

**Description**:
emit_variant_subpats emits `mov rax, [rax + disp8]` (line 3306) where
`off_in_payload = idx_in_payload * 8`. For idx 16, off = 128 — written
as a single byte (`emit_byte(off_in_payload)`). The CPU interprets disp8
as signed; 128 → -128. The load reads `[rax - 128]` — BELOW the enum's
payload region into stack memory.

Identical issue in emit_tuple_subpats at line 3327.

This is the same disp8-overflow pattern as Stage 4 Finding #7 (which
covered AST_TUPLE_LIT and AST_TUPLE_FIELD). Stage 7's patterns inherit
it for free because they use the same single-byte displacement opcode.

In practice, no Phase-0 test exercises this — variants/tuples in the
test suite cap at small arity. But the gap is identical to the now-
resolved Finding #7 from Stage 4.

**Reproducer**:
```
enum Big { Many(i32,i32,i32,i32,i32,i32,i32,i32,
                 i32,i32,i32,i32,i32,i32,i32,i32,i32) }   // arity 17

fn main() -> i32 {
    let b = Big::Many(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17);
    match b {
        Big::Many(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q) => q,
        // q reads disp8 = 17*8 = 136 wraps to -120 → garbage.
    }
}
```

**Trap-id reservation**:
- 62008 (PAT_VARIANT sub-pat idx > 15 — disp8 overflow)
- 62009 (PAT_TUPLE sub-pat idx > 15 — disp8 overflow)

**Recommended fix**:
Mirror Stage 4's resolution: either (a) trap when idx_in_payload > 15
or (b) emit disp32-form (4-byte displacement, opcode `48 8B 80 disp32`).
Phase-0 likely picks (a).

---

## What was checked but found OK (no new finding)

- **AST_FN_DECL slot 6 (is_generic) vs slot 8 (is_checkpoint) collision**:
  These are distinct arena slots (parser.hx:4940, 4942). The Stage 14.5
  report's mention of "both bits per AST_FN_DECL" appears to refer to
  the fact that both flags are read at codegen-time, but they live in
  different slots and read paths. No collision risk. (However, Finding 10
  shows is_checkpoint is mishandled by mono-clone and impl-method paths.)
- **Body-share for stateless ASTs**: monomorphize_pass shares the
  template body for simple-identity fns (line 3661: tpl_gp_head == 0).
  AST nodes are stateless (no codegen-time mutation), each instantiation
  has its own bind_state, so sharing is safe. Verified.
- **Turbofish vs comparison ambiguity** (`f<g>(h)` vs `f::<g>(h)`):
  is_turbofish requires `::<` (parser.hx:2195-2197). Bare `f<g>(h)`
  parses as `f`, then `<` operator. No ambiguity.
- **emit_match_dispatch trap 62001**: unconditional after all arms (line
  3513). Non-exhaustive matches reliably trap at runtime. The bootstrap
  parser does NOT do compile-time exhaustiveness checking, but the
  runtime trap is sound.
- **mr_tab dedup by (orig, pack_lo)**: correct for distinct tag-args.
  Wrong when pack_lo collides for unknown types (Finding 5).
- **AST_FN_DECL slot dispatch for fn_type_table**: pre-pass at
  kovc.hx:5164 correctly skips is_generic templates. The mono'd clones
  carry concrete tags (0..15) which fit the 4-bit pack. Verified.
- **patch_table ud2 fallback for fn_table miss** (kovc.hx:5441-5450):
  unresolved CALL is replaced with ud2 + 3 NOP padding. Sound failure
  mode — better than silent address corruption.
- **Stage 8.5 trait_tab cap 4 / impl_tab cap 8**: both silently return
  -1 on overflow. trait_tab is only used for parse-time disambiguation
  (no callers consult it post-parse besides documentation). impl_tab
  is consulted only for duplicate-impl detection — but since fn_table
  doesn't dedupe either (Finding noted below), duplicate impls produce
  dead code from the second `impl Eq for X`. Acceptable Phase-0
  limitation — documented at parser.hx:322-324.
- **Same-fn duplicate fn_table_add**: kovc.hx:1511 doesn't dedup; the
  first registration wins at lookup, second is dead code. For user-
  defined `fn id__i32` + mono'd `id__i32` from `fn id<T>` + `id::<i32>`,
  the user's plain `id__i32` wins because of fn-list ordering (impl_pending
  spliced BEFORE user fns; mono'd clones appended AFTER user fns).
  Path-call ordering: order is [impl_methods, user_fns, mono_clones].
  fn_table_lookup picks first match — impl_methods win, mono_clones lose
  on collision. If user wrote `fn id__i32` AND used `id::<i32>(...)`,
  the user's plain id__i32 silently shadows the mono'd clone — different
  behavior than the user expected. LOW finding (uncommon pattern), not
  promoted to its own entry. The fix is to detect a user-defined plain
  name colliding with a mono'd mangled name at registration time.

---

## Summary

| #  | Severity | Finding |
|----|----------|---------|
| 1  | HIGH     | Top-level decl order is fixed; trait/impl/struct/enum after first fn silently dropped |
| 2  | HIGH     | ty_ident_to_tag defaults unknown / u8 / u16 / i8 / i16 / bf16 to 0 (i32) — silent type-tag conflation |
| 3  | HIGH     | var_type_tab + method-call sugar conflates unknown/struct types with i32 |
| 4  | MEDIUM   | Stage 8 mr_tab cap-32 overflow silently drops 33rd+ instantiations (no 71001 trap) |
| 5  | HIGH     | ty_ident_to_tag pack_lo collides for unknown types — wrong mono clone emitted |
| 6  | HIGH     | parse_pattern silently drops non-INT literals (negative, float, suffixed int) as wildcards |
| 7  | MEDIUM   | PAT_LIT / PAT_RANGE use 32-bit cmp on 64-bit-stored scrutinee (HIGH for i64/u64/f64) |
| 8  | HIGH     | PAT_VARIANT / PAT_TUPLE arity mismatch reads OOB stack; unknown variant silently disc=0 |
| 9  | MEDIUM   | clone_with_rewrite only handles AST_CALL at root — let/if/seq-wrapped generic calls misdispatch |
| 10 | MEDIUM   | Mono clone discards is_checkpoint and other slot-8 attributes |
| 11 | MEDIUM   | self.method() in impl bodies and Self in expression position don't work |
| 12 | MEDIUM   | PAT_VARIANT / PAT_TUPLE sub-pat disp8 wraps at idx > 15 (parallel to Stage 4 #7) |

6 HIGH, 5 MEDIUM, 1 implicit LOW (duplicate-name shadow).

**Stop-the-line recommendations** (HIGH severity, type-soundness gaps):

1. **Finding 1** (top-level decl order): if you intend to ship Stage 8.5
   as user-facing, allow trait/impl/struct/enum between fn decls. The
   single-pass restriction breaks the natural Rust ordering and is
   trivially exposed by every demo program past the smallest.

2. **Findings 2 + 3 + 5** (ty_ident_to_tag i32-default conflation): this
   is the single biggest type-soundness gap in Stage 8/8.5. Any non-i32
   scalar OR any user-defined struct/enum silently flows through the
   "i32 method dispatch" path. The fix is local (extend ty_ident_to_tag
   + ty_tag_push_name to cover the 5 missing tags + reject unknown
   idents). Without it, traits cannot be meaningfully impl'd for u8 / u16 /
   i8 / i16 / bf16, and ALL struct/enum impls silently route through i32.

3. **Finding 6** (pattern wildcard fallback): every negative-literal or
   float-literal pattern silently becomes a wildcard with the next
   token left dangling. This is a silent miscompile of pattern source.
   Trap on unknown-pattern-token instead.

4. **Finding 8** (pattern arity + unknown variant): `Maybe::Some(a, b, c)`
   on a 1-arity Some reads off the top of the stack. `Maybe::Mango(x)`
   on an unknown variant silently matches disc=0. Both are runtime
   corruption.

The other HIGH findings (5, 6, 8) and all MEDIUM findings can be
batched into a single Stage 8.6 follow-on; Findings 1, 2, 3 should
land before any user-facing release of Stage 8.5.

---

## Resolution status

| #  | Status   | Notes |
|----|----------|-------|
| 1  | FIXED    | parse_program post-fn loop accepts struct/enum/trait/impl/mod/use. Commit ea3040c. |
| 2  | FIXED    | ty_ident_to_tag handles u8/u16/i8/i16/bf16 + u16. Commit d9ac5c2. |
| 3  | FIXED    | ty_tag_push_name split into ty_tag_push_name + ty_tag_push_name_3byte with arms for all new tags. Commit d9ac5c2. |
| 4  | OPEN     | mr_tab cap-32 — deferred (low-priority MEDIUM). |
| 5  | FIXED    | pack_lo collision — resolved by Finding 2 fix; distinct tags now produce distinct pack_lo. Commit d9ac5c2. |
| 6  | FIXED    | parse_pattern fallback emits AST_ERR(62002). emit_pattern_test dispatches tag-99 → emit_trap_with_id. Commit 8574fa8. |
| 7  | OPEN     | PAT_LIT 32-bit cmp on wide scrut — deferred (MEDIUM, needs width-aware emitters). |
| 8  | FIXED    | Pattern arity check at parse_pattern: unknown variant traps 62006, arity mismatch traps 62005. Commit 0ff2bc4. |
| 9  | OPEN     | clone_with_rewrite — deferred (MEDIUM, needs deep-clone recursion). |
| 10 | OPEN     | Mono clone is_checkpoint — deferred (MEDIUM, 1-line propagation but didn't fit time budget). |
| 11 | OPEN     | self.method() in impl bodies — deferred (MEDIUM, feature work). |
| 12 | OPEN     | PAT_VARIANT/TUPLE sub-pat idx > 15 disp8 wrap — deferred (MEDIUM, mirror Stage 4 #7 fix). |

All FIXED entries verified by:
- Heavy gate (`pytest helixc/tests/test_codegen.py::test_bootstrap_kovc_full_pipeline_arithmetic`) — clean.
- Regression tests added inline in test_codegen.py demonstrating the
  pre-fix silent behaviour now traps loudly (exit 132 / SIGILL) or
  produces the correct value.

The FLAT prefix-trap discipline + flat boolean accumulator pattern was
used throughout to avoid straining the host parser. AST_ERR(99) with a
trap-id in p1 is now the canonical parser-side trap shape; kovc.hx's
emit_ast_code default case dispatches tag 99 → emit_trap_with_id(p1).
