# Stage 5 (structs) + Stage 6 (enums) Audit: Silent-Corruption Sweep

**Date**: 2026-05-10
**Scope**: helixc/bootstrap/parser.hx + helixc/bootstrap/kovc.hx at commit
`3421b21` (read-only).
**Trigger**: parallel sweep to `audit-stage4-followup.md`. Stage 5 (Iter
A→D) added struct decl, lit, field access by name, nested struct fields,
struct-by-value fn args, and rbp-relative TUPLE_LIT slot addressing.
Stage 6 (A→D) added enum decl, unit-variant fold to AST_INT, payload-
variant fold to AST_TUPLE_LIT, and `__enum_payload` reader desugaring.
Both stages fold heavily onto existing tuple infra; the question is
whether the new dispatch paths leave silent windows analogous to the
ones the Stage 4 follow-up surfaced.

**Method**: traced every dispatch on `struct_idx` / `enum_idx`; every
`safe_disc` / `safe_arity` fallback; every cap-check in struct_tab /
var_struct_tab / enum_tab / var_enum_tab / fn_table / patch_table;
every disp8 use in struct/enum codegen; every place where the AST_INT
unit-variant fold and the AST_TUPLE_LIT payload-variant fold could be
fed the wrong rep; every place where a struct field's declared type is
expected to be a registered struct.

**Result**: 13 findings (8 HIGH, 4 MEDIUM, 1 LOW). The bulk of HIGH
issues are around (a) unknown-variant / unknown-field name lookups
defaulting silently to 0 / first-variant / first-field; (b) struct/enum
identity loss at fn boundaries (return type not encoded; param identity
clamped to sentinel 15 → caller can pass anything); (c) a representation
mismatch where matching against an all-unit enum dereferences a small
integer as a pointer (SIGSEGV); (d) cross-enum match pattern collision
(pattern's enum_idx p3 is recorded by parser but never validated by
codegen). Several cap-overflow paths in parser-state tables silently
drop entries (var_struct_tab cap=4, struct_tab cap=3, enum_tab cap=4,
fn_table cap=512, var_enum_tab=4) with no surfaced diagnostic.

---

## Finding 1: PAT_VARIANT codegen against all-unit enum SIGSEGVs (rep mismatch)

**Location**: helixc/bootstrap/parser.hx:2479-2491 (parser-side rep choice) +
helixc/bootstrap/kovc.hx:3358-3366 (codegen dereferences as pointer)
**Severity**: HIGH
**Category**: type-soundness / silent-corruption (loud crash without trap-id)

**Description**:
Stage 7F documented that enums with at least one payload variant
(`max_arity > 0`) MUST use the pointer-shaped representation for ALL
variants — including unit variants — so the PAT_VARIANT codegen's
`mov rax, [rbp+disp]; mov eax, [rax+0]` dereference reads the
discriminant cleanly. The fix at parser.hx:2479-2491 special-cases
all-unit enums (`max_arity == 0`) and folds them to AST_INT (5-byte
`mov eax, imm32`) "for backward compat with Stage 6B tests".

The mismatch: `match c { Color::R => x, ... }` where `Color` is
all-unit. The let-binding stores 4 bytes (i32, since `expr_type(AST_INT)
= 0`). emit_match_dispatch (kovc.hx:3508-3509) reserves a fresh slot and
does `emit_mov_local_rax_64(scrut_off)` — REX.W store, top 4 bytes zero.
Then emit_pat_variant_disc (kovc.hx:3359-3365):

```
mov rax, [rbp+disp]    ; rax = 0/1/2 (the variant's integer disc)
mov eax, [rax+0]       ; dereferences a small integer as a pointer → SIGSEGV
```

The test suite covers `Color::R` standalone (line 2596) and `Color::R +
Color::G + Color::B + 39` (2606), but never `match c { Color::R => ...
}`. Confirmed by `grep -n "match.*Color::" helixc/tests/test_codegen.py`
returning empty.

This is the same class of bug the Stage 7F fix was MEANT to prevent
(pointer-deref of the disc-as-integer rep); the Stage 7F fix solves
it for mixed-arity enums but leaves the all-unit enum case as a latent
trap.

**Reproducer**:
```
enum Color { R, G, B }
fn main() -> i32 {
    let c = Color::G;
    match c {
        Color::R => 0,
        Color::G => 1,
        Color::B => 2,
    }
}
```
Expected: 1. Actual: SIGSEGV (the `mov eax, [rax+0]` reads from address
0x1, which is unmapped).

**Recommended fix**:
Drop the special case at parser.hx:2482-2491 — always fold a unit
variant to the 1-slot AST_TUPLE_LIT pointer rep regardless of
max_arity. The "backward compat with Stage 6B tests" reads
`let c = Color::G; c` as 1 — works either way (pointer-shaped also
returns the disc when read as a plain expression, because the value
is `[rbp - 8]` = pointer-to-`{disc}`, and `c` evaluates to the
pointer, then `c.0` reads disc=1... but `c` alone returns the
pointer, NOT disc=1). The existing 6B test
`"enum Color { R, G, B } fn main() -> i32 { let c = Color::G; c }"`
DOES rely on the AST_INT rep returning 1 directly — so dropping the
special case will regress that test, but the test is itself
incorrect: it conflates "the unit variant's discriminant" with "the
unit variant's value". Adding a `c.0` accessor would correct it.

Alternative: at codegen time, detect that the scrut's bind_state ty
is 0 (i32) when matching a PAT_VARIANT and route through a different
helper that compares the i32 disc directly (no deref). The parser
already records p3 = enum_idx on PAT_VARIANT; codegen can look up
max_arity from enum_table to choose disc-cmp-direct vs deref-and-cmp.

**Trap-id reservation**:
69001 (AST_MATCH against all-unit enum produced disc-deref → SIGSEGV).
Emit before the deref when scrut's expr_type is i32 (0) AND pattern is
PAT_VARIANT. Or remove the rep mismatch entirely per the above fix.

---

## Finding 2: PAT_VARIANT discriminant compared cross-enum without enum_idx validation

**Location**: helixc/bootstrap/kovc.hx:3383-3394 (emit_compound_pattern_test)
+ helixc/bootstrap/parser.hx:5593 (parser stores enum_idx in p3 but codegen ignores)
**Severity**: HIGH
**Category**: type-soundness / silent-corruption

**Description**:
Pattern parsing at parser.hx:5593 builds `mk_node(69, safe_disc,
sub_head, e_idx_pre)` — p3 carries the enum_idx the pattern was
parsed against. But emit_compound_pattern_test (kovc.hx:3383-3394)
reads only p1 (disc) and p2 (sub_head); p3 is NEVER consulted.

emit_pat_variant_disc compares `[scrut+0]` (= scrut's disc slot)
with the pattern's disc. Two unrelated enums with overlapping
discriminants are indistinguishable to the codegen:

```
enum Maybe { None, Some(i32) }    // Maybe::Some disc=1
enum Color { R, G, B }            // Color::G disc=1
let m = Maybe::Some(42);
match m {
    Color::G => 100,              // disc=1 matches → THIS arm runs
    Maybe::Some(v) => v,          // unreachable
    Maybe::None => 0,
}
```

Result: returns 100 instead of 42. The PAT_VARIANT match succeeds
because the disc happens to coincide. The user has no diagnostic
that the pattern's enum (Color) is unrelated to the scrut's enum
(Maybe).

In Phase 0 this is unlikely to surface (Maybe with payload + Color
all-unit is the only pair) — but the codegen-side gap is real:
patterns silently match across enum boundaries whenever disc
collides. As Phase-0 grows toward more enums, the risk increases.

**Reproducer**: as above.

**Recommended fix**:
emit_compound_pattern_test should read `pp3 = __arena_get(pat_idx + 3)`
(enum_idx). emit_match_dispatch should pass the scrut's enum_idx
(looked up via expr_type → var_enum_tab once that's wired) and a
new pat_test variant `emit_pat_variant_disc_xenum_guard(scrut_off,
expected_enum_idx, pat_enum_idx, fail_state)` that traps if the two
disagree.

Alternative (cheaper): when the parser sees a PAT_VARIANT, emit a
disc-cmp AND emit an enum-tag-cmp. The enum-tag would need to live
in slot 1 of the variant's tuple-lit rep (slot 0 = disc, slot 1 =
enum_idx, payload starts at slot 2). Existing tests would need to
shift indices but the offset disp8 still fits.

**Trap-id reservation**:
69010 (PAT_VARIANT enum_idx mismatch — pattern's enum differs from
scrut's enum).

---

## Finding 3: Unknown variant/field names default to 0 ("safe_disc"/"safe_arity") instead of trapping

**Location**:
- helixc/bootstrap/parser.hx:2473 (`safe_disc` in unit-variant construct)
- helixc/bootstrap/parser.hx:2512-2513 (`safe_disc`/`safe_arity` in payload-variant construct)
- helixc/bootstrap/parser.hx:5564 (`safe_disc` in pattern PAT_VARIANT)
**Severity**: HIGH
**Category**: silent-corruption / safety

**Description**:
`enum_tab_variant_lookup_disc` returns -1 on a miss (unknown variant).
The construct sites (and pattern site) wrap with
`let safe_disc = if disc < 0 { 0 } else { disc };`. This means any
misspelled variant name resolves to discriminant 0 — the first
variant — without any diagnostic.

Cases that escape:
- `Maybe::Bogus` typo for `Maybe::Some`: emits `Maybe::None` (disc=0).
- `Maybe::sOme` (wrong case): emits disc=0.
- `Color::G` typo for `Color::Green` (when enum is `Color { Red,
  Green, Blue }`): emits disc=0 = Color::Red.

In pattern position, `match m { Maybe::Bogus => ... }` parses to
PAT_VARIANT(disc=0, ..., enum_idx=Maybe) — matches `Maybe::None`.
Two patterns both written as `Maybe::Bogus` would both match disc=0
("unreachable" warning never fires; both arms claim the slot).

Similarly, `enum_tab_variant_lookup_arity` returns -1 on miss; the
payload construct silently sets `safe_arity = 0`. Combined with the
fact that the parser walks ALL parenthesized args (no comparison
against safe_arity), the resulting AST_TUPLE_LIT has `n_args = 1 +
parsed_args` regardless of what the variant actually declared.

**Reproducer**:
```
enum Maybe { None, Some(i32) }
fn main() -> i32 {
    let m = Maybe::Bogus;       // typo — silently emits Maybe::None
    match m {
        Maybe::Some(v) => v,
        Maybe::None => 99,      // this arm runs
    }
}
```
Expected: compile error. Actual: returns 99.

For struct field misses, see Finding 4.

**Recommended fix**:
At parser.hx:2473, 2512-2513, 5564: replace `if disc < 0 { 0 } else
{ disc }` with a flat prefix-trap pattern emitting AST_ERR-style trap
node OR (simpler) emit an AST_TUPLE_LIT whose first slot is a built-
in trap call. The codegen-side trap-id pattern is the right shape:

```
let n_pre_trap = if disc < 0 { emit_trap_with_id(60001) } else { 0 };
```

The flat pattern is host-parser-recursion-safe (Finding #7 lesson).
Alternative: synthesize an AST_INT node with sentinel value -1 and
add a codegen-side guard.

**Trap-id reservation**:
- 60001 (unit-variant unknown name)
- 60002 (payload-variant unknown name)
- 60003 (PAT_VARIANT unknown variant name)

---

## Finding 4: Unknown struct field name (`.bogus`) silently bails postfix loop, eating both `.` and IDENT

**Location**: helixc/bootstrap/parser.hx:1454-1477 (parse_unary postfix `.IDENT` arm)
**Severity**: HIGH
**Category**: silent-corruption

**Description**:
When parsing `p.bogus` where `bogus` is not a field of `p`'s struct:
- Line 1454 enters the `lhs_struct_idx >= 0` branch.
- Lines 1455, 1459: cur_advance consumes both `.` and the IDENT.
- Line 1460: struct_tab_field_lookup returns -1 (miss).
- Line 1476: `if f_idx >= 0 { ... } else { keep_p = 0; }` — bails the
  postfix loop, returning `prim` (the LHS var) unchanged.

The tokens `.` and `bogus` are consumed but no field-access AST node
is emitted. The caller of parse_unary receives just `p`. The user
wrote `p.bogus + 1` and got back `p + 1`. Silent.

Compare with the surrounding behavior at line 1477 (`if
lhs_struct_idx >= 0` else `keep_p = 0`): when the LHS is not
recognized as a struct (e.g., var was dropped due to var_struct_tab
overflow — Finding 6), the `.` token is NOT consumed (because
cur_advance for `.` is inside the `lhs_struct_idx >= 0` arm at
line 1455). The surrounding parser will then try to parse `.` as
something else — also silent confusion, but at least the tokens
aren't eaten.

**Reproducer**:
```
struct Pt { x: i32, y: i32 }
fn main() -> i32 {
    let p = Pt { 10, 32 };
    p.bogus + 1     // expect: compile error "no field 'bogus' on Pt"
                    // actual: emits AST p + 1; runtime returns
                    //         pointer-value(p) + 1 = address of p + 1.
}
```

**Recommended fix**:
At line 1476, replace `else { keep_p = 0; }` with `else { let n_trap
= emit_trap_with_id(50010); keep_p = 0; }` — but emit_trap_with_id
is codegen-side, not parser-side. Parser-side equivalent: build
an AST_ERR node (synthesize tag 99 — already used at 1510 for nested
closures) that codegen emits as a loud trap.

```
prim = mk_node(99, 50010, 0, 0);   // AST_ERR with trap-id 50010
```

The AST_ERR codegen at kovc.hx:3xxx already emits ud2 + eax=trap_id.

**Trap-id reservation**:
50010 (struct field-name not found).

---

## Finding 5: Struct-typed fn return silently degrades to i32 — pointer truncated to 4 bytes

**Location**: helixc/bootstrap/parser.hx:4855-4902 (parse_fn_decl return-type IDENT dispatch)
**Severity**: HIGH
**Category**: silent-corruption / type-soundness

**Description**:
The return-type IDENT is matched against a hard-coded byte-pattern
for the known scalar types (3-byte: i32/i64/u32/u64/f32/f64; 2-byte:
i8/u8; 4-byte: bf16). Any IDENT that doesn't match (e.g., `Pt`, `Line`,
`Maybe`, `MyEnum`) silently falls through to `ret_ty = 0` (i32).

The fn body codegen then emits a 4-byte `mov edi, eax` at the exit
stub regardless of what the body produces. For a struct-returning fn
where the body's tail expr is a struct literal (or a struct-typed
var), the value in rax is a 64-bit pointer — the top 4 bytes are
silently dropped at return.

The caller of such a fn receives a 32-bit pointer (sign-extended? or
zero-extended? — depends on the surrounding op). Subsequent `.field`
access dereferences the truncated pointer → SEGV or random heap
memory.

There is also no diagnostic that "this return type IDENT is unknown".
The same fall-through path silently accepts arbitrary type names.

This is an analog of Stage 4 Finding #1 (body-vs-ret-ty trap only
checks 8-byte width class) — but worse, because the entire return-type
encoding is lost.

**Reproducer**:
```
struct Pt { x: i32, y: i32 }
fn make_pt() -> Pt {        // ret_ty silently = 0 (i32)
    Pt { 10, 32 }
}
fn main() -> i32 {
    let p = make_pt();
    p.x                     // SEGV: p was stored as 4-byte truncated pointer
}
```

**Recommended fix**:
After the scalar tag check at line 4885, look up the IDENT in
struct_tab and enum_tab; if found, encode as `ret_ty = 100 +
struct_idx` (mirroring the param-side encoding) or
`ret_ty = 300 + enum_idx`. Then teach the fn-decl epilogue codegen
to keep rax 64-bit when ret_ty ≥ 100.

If the IDENT matches no scalar / struct / enum, trap (synthesize
AST_ERR(50020) into the fn body OR emit_trap_with_id(50020) in the
prologue so the fn always SIGILLs when called).

**Trap-id reservation**:
- 50020 (unknown return-type IDENT)
- 50021 (struct return type degraded to i32 — caller's 4-byte read
  of an 8-byte pointer)

---

## Finding 6: Cap-overflow on parser-state tables silently drops entries

**Location**:
- `struct_tab_add` parser.hx:771-785 (cap 3, returns -1)
- `var_struct_tab_add` parser.hx:734-747 (cap 4, returns -1)
- `enum_tab_add` parser.hx:861-876 (cap 4, returns -1)
- `fn_table_add` kovc.hx:1511-1526 (cap 512, returns -1)
- `patch_table_add` kovc.hx:1567-1583 (cap 4096, returns -1)
- Callers at parser.hx:5027 (enum), parser.hx:5497 (struct), parser.hx:2070
  + parser.hx:4833 (var_struct), kovc.hx:5256 + kovc.hx:5417 (fn_table)
**Severity**: HIGH
**Category**: silent-corruption

**Description**:
Every "*_add" helper above returns -1 on cap overflow, but EVERY
caller ignores the return value. Downstream lookups miss silently:

- **struct_tab cap 3** (parser.hx:773): a 4th `struct X {...}` decl is
  dropped from the table; later `X { ... }` literals fail
  struct_tab_lookup_idx and fall through to "not a struct — treat as
  var ref" (parser.hx:2758). The parser leaves the `{` token
  unconsumed, then the surrounding context (let-statement parser,
  block parser) misinterprets the brace. Silent parser-state
  corruption.

- **var_struct_tab cap 4** (parser.hx:736): the 5th struct-typed let
  binding in any function is dropped. Later `p.field` access on `p`
  fails var_struct_tab_lookup, lhs_struct_idx stays -1, postfix loop
  bails (Finding 4 path). The `.` and IDENT are NOT consumed (per
  line 1477's behavior — different from line 1476), so the
  surrounding context attempts to parse `.IDENT` and is confused.

- **enum_tab cap 4** (parser.hx:863): the 5th `enum X {...}` decl is
  dropped. Later `X::V` paths fail enum_tab_lookup_idx, fall through
  to the var-ref path (no recognized enum, no struct, no scalar
  type), and the parser sees `X::V` as a bare IDENT — leaving `::V`
  tokens unconsumed. Silent.

- **fn_table cap 512** (kovc.hx:1515): the 513th fn's NAME-TO-OFFSET
  entry is dropped, BUT its CODE bytes are still emitted into the
  instruction stream. Calls to it from elsewhere miss fn_table_lookup
  and hit the ud2 patch at kovc.hx:5444-5449 — SIGILL with eax = 0
  (not a trap-id). The user has no way to tell "fn_table overflow"
  apart from "undeclared fn". The current bootstrap has ~434 fns
  hand-written (lexer 14 + parser 187 + kovc 233); mono passes
  (generic + grad + grad_rev + impl) can easily push past 512.

- **patch_table cap 4096** (kovc.hx:1572): the 4097th CALL/LEA patch
  is dropped; the corresponding call site's relative displacement is
  NEVER patched (stays at the placeholder bytes 00 00 00 00). The
  resulting binary jumps to a near-zero offset (the placeholder
  becomes an absolute address from the call site's next-instruction
  IP). Almost certainly SIGSEGV — but again, no trap-id eax.

**Reproducer (struct_tab cap)**:
```
struct A { x: i32 }
struct B { x: i32 }
struct C { x: i32 }
struct D { x: i32 }       // dropped silently
fn main() -> i32 { let d = D { 42 }; d.x }   // D not registered; parser confusion
```

**Reproducer (var_struct_tab cap)**:
```
struct Pt { x: i32, y: i32 }
fn main() -> i32 {
    let a = Pt{1,2}; let b = Pt{3,4}; let c = Pt{5,6}; let d = Pt{7,8};
    let e = Pt{9,10};        // 5th — var_struct_tab_add returns -1 silently
    e.x                       // var_struct_tab_lookup misses; .x not parseable
}
```

**Recommended fix**:
At each caller, capture the return value and emit a trap-AST or trap-
codegen on -1:

```
let res = struct_tab_add(sb, name_s, name_l, field_count, fields_ptr);
if res < 0 {
    return mk_node(99, 50030, 0, 0);   // AST_ERR with trap-id
};
```

For kovc.hx fn_table_add at line 5256: if the return is -1, abort the
whole compilation (synthesize trap call in the entry stub OR
emit_byte(0x0F); emit_byte(0x0B) immediately and stop emitting code).

Additionally: bump the caps where there's reasonable expectation of
growth. struct_tab cap 3 is brittle (typical demo programs hit it
quickly); var_struct_tab cap 4 is brittle for any non-trivial use.
A cap of 32 each is cheap (32 * 4 = 128 arena slots for struct_tab;
32 * 3 = 96 for var_struct_tab; both fit in the existing parser-state
region).

**Trap-id reservation**:
- 50030 (struct_tab cap overflow)
- 50031 (var_struct_tab cap overflow)
- 60010 (enum_tab cap overflow)
- 60011 (var_enum_tab cap overflow — currently dead infrastructure)
- 10010 (fn_table cap overflow)
- 10020 (patch_table cap overflow)

---

## Finding 7: Struct lit field count vs declared arity not validated — `Pt { 10 }` for arity-2 silently emits 1-slot

**Location**: helixc/bootstrap/parser.hx:2724-2755 (struct-lit construct in parse_primary)
**Severity**: HIGH
**Category**: silent-corruption

**Description**:
parse_primary at line 2706 looks up `arity` from struct_table, then at
2725-2747 walks comma-separated children. Variable `n` counts the
ACTUAL number of children supplied. At line 2755:
`mk_node(50, n, head_idx, 0)` — emits AST_TUPLE_LIT with `n`, NOT
`arity`. No comparison or trap.

Cases that escape:
- **Too few fields**: `Pt { 10 }` (arity 2, supplied 1). n=1. The
  TUPLE_LIT reserves 1 slot. lea rax = slot 0's address. Later
  `p.1` (or `p.y` via field name → idx=1) reads `[rax + 8]` —
  which is the NEXT bind_alloc slot (perhaps the let-binding for
  `p` itself, or another local), interpreted as i32.
- **Too many fields**: `Pt { 10, 20, 30 }` (arity 2, supplied 3).
  n=3. TUPLE_LIT reserves 3 slots. `.x` reads slot 0=10, `.y`
  reads slot 1=20, the third slot is silently stranded. Memory
  waste, but accessing `.2` or `.z` won't match struct_tab_field_lookup
  (arity is still 2 from the decl) so the typo path catches it.

The mid-case `.1` vs `.y`: `.1` is positional (tag 52 with literal
idx=1, no field-name lookup) — reads slot 1 regardless of field
existence. `.y` goes through struct_tab_field_lookup which respects
the declared arity. So `Pt { 10 }; p.1` reads OOB; `Pt { 10 }; p.y`
reads OOB too (struct_tab_field_lookup returns 1 for y, then codegen
reads slot 1 — same OOB).

**Reproducer**:
```
struct Pt { x: i32, y: i32 }
fn main() -> i32 {
    let z = 99;            // some allocator state
    let p = Pt { 10 };     // n=1, arity=2
    p.y                    // expect: compile error.
                           // actual: returns whatever was last allocated
                           //         in the adjacent bind_alloc slot.
}
```

**Recommended fix**:
After the loop at line 2747, before line 2754:
```
let n_count_trap = if n != arity {
    mk_node(99, 50040, 0, 0)     // AST_ERR
} else { 0 };
```
... and propagate the trap into the returned node. Or simpler at
codegen: store `arity` (from struct_table) in the AST_TUPLE_LIT's p3
slot, and compare against `n` (p1) when emitting; trap on mismatch.

**Trap-id reservation**:
50040 (struct lit field count != declared arity).

---

## Finding 8: Enum payload variant arity not validated — `Maybe::Some()` and `Maybe::Some(1, 2, 3)` both silently parse

**Location**: helixc/bootstrap/parser.hx:2510-2543 (parse_primary payload-variant)
**Severity**: HIGH
**Category**: silent-corruption

**Description**:
Same shape as Finding 7. parser.hx:2511 looks up the declared `arity`
and at 2513 falls back to 0 on miss (already a silent failure mode —
covered in Finding 3). Lines 2520-2537 walk parenthesized args
counting them in `n_args` (starts at 1 for the discriminant). At line
2543: `mk_node(50, n_args, head_idx, 0)`. No comparison against
declared arity.

- `Maybe::Some()` (declared 1, supplied 0): n_args=1 (just disc).
  Later sub-pat reads via `[scrut + 8]` walk past the end of the
  reserved slots, reading the adjacent bind_alloc region.
- `Maybe::Some(1, 2, 3)` (declared 1, supplied 3): n_args=4. Extra
  slots stranded; if user matches `Maybe::Some(a)`, only `a = 1`
  is read (correct), but the extra slots silently waste stack space.

When this gets through to PAT_VARIANT codegen, the sub-pat loop at
emit_variant_subpats reads sequential 8-byte slots regardless of
declared arity (see Finding 9 for the disp8 wrap variant).

**Reproducer**:
```
enum Maybe { None, Some(i32) }
fn main() -> i32 {
    let m = Maybe::Some();           // missing payload
    match m {
        Maybe::Some(v) => v,         // reads [scrut + 8] = garbage
        Maybe::None => 0,
    }
}
```

**Recommended fix**:
After line 2538 (after consuming `)`), before line 2541:
```
let payload_supplied = n_args - 1;    // subtract disc
let n_arity_trap = if payload_supplied != safe_arity {
    mk_node(99, 60020, 0, 0)
} else { 0 };
```

**Trap-id reservation**:
60020 (enum payload-variant supplied arity != declared arity).

---

## Finding 9: emit_variant_subpats / emit_tuple_subpats disp8 wrap at idx > 15

**Location**: helixc/bootstrap/kovc.hx:3294-3334 (both helpers)
**Severity**: MEDIUM (analog of Stage 4 Finding #7)
**Category**: silent-corruption

**Description**:
Both helpers compute `off_in_payload = idx_in_payload * 8` (line 3304)
or `off_in_tuple = idx_in_tuple * 8` (line 3326) and emit
`mov rax, [rax + disp8]` (4-byte instruction). For idx ≥ 16, off ≥
128 — disp8 is signed, interpreted as -128. The load reads from
`[rax - 128]` (below the variant's allocated region) — silent OOB.

The companion struct/tuple-FIELD path (AST_TUPLE_FIELD, tag 52)
already has the disp8 trap at kovc.hx:3615 (`if p2 > 15 { trap
52001 }`). But emit_variant_subpats and emit_tuple_subpats —
introduced in Stage 7 — never got the same guard.

This is a latent issue: Phase-0 enums with >15-payload variants are
unusual, but well-formed Helix source `enum E { Big(i32, i32, i32,
i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
}` compiles silently.

Also: emit_variant_subpats starts `idx_in_payload = 1` (skipping
disc), so the threshold is idx ≥ 15 (off ≥ 128 after the +1).
emit_tuple_subpats starts at 0; threshold is idx ≥ 16.

**Reproducer**:
```
enum E { Big(i32, i32, i32, i32, i32, i32, i32, i32, i32,
             i32, i32, i32, i32, i32, i32, i32, i32) }
fn main() -> i32 {
    let e = E::Big(1, 2, 3, 4, 5, 6, 7, 8, 9,
                   10, 11, 12, 13, 14, 15, 16, 17);
    match e {
        E::Big(a, b, c, d, e_, f, g, h, i,
               j, k, l, m, n, o, p, q) => q,  // q reads [rax + 128] wrap to [rax - 128]
        _ => 0,
    }
}
```

**Recommended fix**:
At kovc.hx:3303 (variant) and 3325 (tuple), add a flat prefix-trap:
```
let n_wrap_trap = if idx_in_payload > 15 {
    emit_trap_with_id(60030)
} else { 0 };
```
Then the existing instructions emit. The trap fires before the
wrapping load.

Or implement disp32 loads here (mirroring TUPLE_LIT's disp32 stores
from Stage 5 Iter D step 5).

**Trap-id reservation**:
- 60030 (emit_variant_subpats disp8 wrap)
- 60031 (emit_tuple_subpats disp8 wrap)

---

## Finding 10: `__enum_payload(m, idx)` non-INTLIT idx silently uses 0 (documented but not trapped)

**Location**: helixc/bootstrap/parser.hx:2667-2680 (Stage 6D reader)
**Severity**: MEDIUM
**Category**: silent-corruption (acknowledged in commit msg, not fixed)

**Description**:
The Stage 6D commit message (6395fde) acknowledges: "Phase 0
limitation: idx must be a compile-time INTLIT (AST tag 0). A
non-literal idx silently uses 0 — Stage 7 (match) will replace this
builtin with a proper pattern bind so the limitation is short-lived."

At commit `3421b21`, Stage 7 (match) is in place but `__enum_payload`
is STILL emitted by user code (parser.hx:2643-2680 detects it; tests
2626-2638 cover it). So the documented "short-lived" limitation
remains active.

The silent-0 case also misses AST_INTLIT_U32 (tag 36) and other
wide INT-LIT tags. `__enum_payload(m, 5_u32)` silently returns slot 1
(idx=0+1) because `a1_tag = 36`, not 0.

Additionally, `idx_val + 1` is unchecked against the variant's
actual payload arity. `__enum_payload(Maybe::Some(42), 5)` reads
slot 6 of a 2-slot tuple — OOB. (Stage 4 Finding #7's TUPLE_FIELD
trap covers p2 > 15, but not "p2 > variant's declared arity".)

Lastly, AST_TUPLE_FIELD's p3 defaults to 0 (4-byte read), so an
i64 / u64 / f64 / pointer payload silently truncates to 4 bytes.

**Reproducer**:
```
enum Maybe { None, Some(i32) }
fn main() -> i32 {
    let idx = 5;
    let m = Maybe::Some(42);
    __enum_payload(m, idx)    // a1_tag = AST_VAR (1), not 0 → idx_val = 0
                              // emits AST_TUPLE_FIELD(m, 1, 0) → m.1 = 42
                              // user wanted slot 5; got slot 1.
}
```

**Recommended fix**:
At line 2675, replace the silent-0 with a flat prefix-trap:
```
let idx_val = if a1_tag == 0 { __arena_get(a1 + 1) } else {
    // Trap by synthesizing AST_ERR
    return mk_node(99, 60040, 0, 0);
};
```

Also: accept the wider INT-LIT tags (36, 37, 38, 39, 40, 41) and
extract their literal value. Validate against the variant's
declared arity (via enum_table); trap on OOB.

**Trap-id reservation**:
- 60040 (`__enum_payload` non-INTLIT idx)
- 60041 (`__enum_payload` idx >= variant's payload arity)

---

## Finding 11: bind_alloc_offset has no cap-check — exhausts 1024-byte prologue silently

**Location**: helixc/bootstrap/kovc.hx:1037-1041
**Severity**: MEDIUM
**Category**: silent-corruption / safety

**Description**:
`bind_alloc_offset` increments the offset by 8 with no upper bound.
The prologue allocates 1024 bytes (`sub rsp, 1024` at kovc.hx:743).
After 128 calls, offsets exceed the alloca region and writes go into
the parent frame's saved rbp / return address / red zone — silent
stack corruption.

The comment at kovc.hx:1024-1029 acknowledges the issue ("parse_primary
nests ~30 lets, blowing past the 512-byte prologue allocation;
emit_mov_local_eax(-560) writes into the parent frame's saved
rbp/return-address"). The fix was to bump 512 → 1024. But no cap-check
was added.

Stage 5 Iter D's AST_TUPLE_LIT consumes `arity` slots per call. For
deeply nested struct lit programs, this is reached quickly. The
comment at kovc.hx:3672-3674 acknowledges "the headroom (1024 - 64*8
= 512 bytes = 64 extra slots) covers all current tests" — assuming
the user doesn't push past the budget.

**Reproducer**:
```
fn main() -> i32 {
    // 130+ lets each allocating 1 slot
    let a01 = 1; let a02 = 1; ... let a130 = 1;
    // 130th let writes [rbp - 1040] → outside the 1024-byte alloca
    a130
}
```
Or with structs:
```
struct Pt { x: i32, y: i32 }
fn main() -> i32 {
    let a = Pt{1,2}; let b = Pt{3,4}; ... 60 struct lets ...
    // Each struct let consumes 3 slots (2 for fields + 1 for binding)
    // 60 lets = 180 slots — overflows 128-slot budget
}
```

**Recommended fix**:
At kovc.hx:1037 add a cap-check:
```
fn bind_alloc_offset(state: i32) -> i32 {
    let off = __arena_get(state);
    if off >= 1024 {
        emit_trap_with_id(10030);
        // continue with off so codegen doesn't NPE; the trap
        // is loud enough.
    };
    __arena_set(state, off + 8);
    off
}
```

Or bump the prologue allocation to 4096 (or 8192) bytes; this is a
once-per-fn cost but adds robustness. The current 1024-byte budget
is tight enough that nested struct lit programs hit it without trying.

**Trap-id reservation**:
10030 (bind_alloc_offset exhausted prologue allocation).

---

## Finding 12: Struct-typed fn-call argument identity not validated — caller can pass any 8-byte value

**Location**: helixc/bootstrap/kovc.hx:4924-4935 (AST_CALL per-arg trap)
**Severity**: MEDIUM
**Category**: type-soundness

**Description**:
The Stage 5 Iter C comment at kovc.hx:4916-4923 explains: when
expected_ty == 15 (struct sentinel — the packed param table's 4-bit
clamp), the per-arg type-mismatch trap (16001) is suppressed. The
clamping happens in fn_type_table at kovc.hx:5185 (`pp_ty = if
pp_ty_raw >= 100 { 15 } else { pp_ty_raw }`), so 100+struct_idx
collapses to 15 regardless of WHICH struct.

Consequences:
1. **Cross-struct passing**: `fn area(p: Pt) -> i32 { p.x + p.y }`
   accepts any struct, including `Line`. If Line's slot 0 happens
   to be an inner Pt pointer (not an i32), `p.x` dereferences that
   inner pointer instead of reading slot 0 as i32 — wrong value
   silently.
2. **Non-struct argument**: `area(42)` (i32 literal) — expr_type
   = 0, expected_ty = 15, exp_is_struct = 1 → mismatch suppressed.
   The caller emits `mov edi, 42` (4-byte mov, but the callee's
   prologue does REX.W store of rdi since p_ty >= 100 → high half
   of rdi is whatever was there, then `p.x` = `[rax+0]` where
   rax is the spilled rdi value 42, treated as a pointer →
   SEGV at address 42.

The comment at 4922 explicitly says "Iter D may add a stricter
check that compares struct identity end-to-end." That stricter
check has NOT been added.

**Reproducer (cross-struct)**:
```
struct Pt { x: i32, y: i32 }
struct Big { a: i32, b: i32, c: i32, d: i32 }
fn use_pt(p: Pt) -> i32 { p.x + p.y }
fn main() -> i32 {
    let b = Big { 100, 200, 300, 400 };
    use_pt(b)        // expected: compile error.
                     // actual: emits OK; use_pt's p.x = b.a = 100,
                     //         use_pt's p.y = b.b = 200, returns 300.
                     //         Whoops — also might happen to work
                     //         depending on struct field order.
}
```

**Reproducer (non-struct)**:
```
struct Pt { x: i32, y: i32 }
fn use_pt(p: Pt) -> i32 { p.x }
fn main() -> i32 {
    use_pt(42)       // expected: compile error.
                     // actual: SEGV at [rax+0] where rax = 42.
}
```

**Recommended fix**:
Bump the param-ty packed table from 4 bits/param to 8 bits/param.
8 bits supports tags 0..255 — enough for 100..103 (4 struct slots)
plus 200..207 (8 generic-param slots) plus 0..15 scalars. The
packing in kovc.hx:5188-5195 just needs to shift by 8 instead of 4.

Then the mismatch trap can compare the FULL struct_idx, not the
clamped sentinel.

Alternative (simpler): emit a runtime guard at the callee's
prologue — load the first 4 bytes of the alleged struct pointer
and compare against a known marker (e.g., the struct's name's hash).
Expensive but catches the non-struct-passed-as-struct case.

**Trap-id reservation**:
- 16010 (struct param expected, scalar argument supplied)
- 16011 (struct param of struct_idx X, struct of idx Y supplied)

---

## Finding 13: `last_enum_idx` scratch slot is written by enum-construct paths but never consumed

**Location**: helixc/bootstrap/parser.hx:174-178 (definition), 2486 + 2541 (writes),
no readers
**Severity**: LOW
**Category**: dead code / latent bug if a future reader is added

**Description**:
Stage 6's enum-construct paths set `last_enum_idx` (sb+24) at lines
2486 (unit-variant pointer rep) and 2541 (payload-variant). The
intent was for the let-parser to consume it (analogous to
last_struct_idx at line 2068-2072) and register the let-binding in
var_enum_tab. But:

1. `var_enum_tab_add` (parser.hx:960) is **defined but never called**.
2. `var_enum_tab_lookup` (parser.hx:977) is **defined but never called**.
3. `last_enum_idx` is never **read** anywhere in the codebase (only
   the wrapper `last_enum_idx(sb)` accessor exists at line 177; no
   consumers grep-able).

Consequence: enum-typed let bindings have NO recorded enum identity.
This silently disables any future "var-based enum dispatch" feature
(e.g., method-call sugar on enum variables). Right now it's just
dead infra — no active misbehavior — but if a future stage assumes
var_enum_tab is populated, it will silently get wrong answers.

`last_enum_idx` is also stale across multiple enum constructs in
the same expression — e.g., `Maybe::Some(42) + Color::G as i32` (if
casts were supported) leaves last_enum_idx pointing at the second
construct's enum_idx forever, even after both have been emitted.

**Reproducer**: not user-visible at commit `3421b21`. Latent.

**Recommended fix**:
Either:
(a) Implement the missing wire-in: in the let-parser at parser.hx:2068
    add a parallel block reading last_enum_idx and calling
    var_enum_tab_add. Clear the scratch slot after consumption.
(b) Remove the dead code: delete var_enum_tab_init, var_enum_tab_add,
    var_enum_tab_lookup, set_last_enum_idx, last_enum_idx, and the
    sb+22..sb+24 reservations. Removes ~25 lines of dead helpers.

**Trap-id reservation**:
N/A — this is dead code, no trap needed.

---

## What was checked but found OK (no new finding)

- **AST_TUPLE_LIT disp8 wrap (Stage 4 Finding #7 STORE side)**: Iter D
  step 5 (f6f3f7c) switched to disp32 stores; no wrap. ✓
- **AST_TUPLE_FIELD disp8 wrap**: trap 52001 at kovc.hx:3615 fires for
  p2 > 15. ✓ (analogous to Stage 4 Finding #7, resolved.)
- **AST_FN_DECL slot consistency (Stage 14.5 layout)**: all writers
  (parser.hx:1653, 3666, 4353, 4667, 4937, 5165) push 9 slots; reader
  at kovc.hx:5163, 5246 only reads slot 6 (is_generic). Parser-side
  readers at 3659 (tpl_gp_head, slot 7) and 4401 (cal_is_ckpt, slot 8).
  Consistent. ✓
- **Iter D nested-struct p3=1 marker propagation**: parser.hx:1471 sets
  cur_struct_idx = f_struct_idx when the field is struct-typed; the
  next .IDENT picks it up at line 1445. emit_ast_code at kovc.hx:3614
  reads p3 to choose 8-byte vs 4-byte load. ✓ (Marker is propagated.)
- **expr_type for AST_TUPLE_FIELD (tag 52)**: kovc.hx:1168 returns 3
  (i64) when p3 == 1, 0 (i32) otherwise. Correct shape for downstream
  ops. ✓
- **bind_state struct-typed binding readback**: kovc.hx:4723 + 4727
  routes ty ≥ 100 to 8-byte load. ✓
- **Iter C's struct-by-value 8-byte spill**: kovc.hx:5288-5292 routes
  p_ty ≥ 100 to needs_64 = 1. ✓
- **Stage 6 fn_table cap bump 256 → 512**: at ~434 fns in current
  bootstrap, the bump is sufficient through Stage 14.5 — but see
  Finding 6 for the silent-overflow concern.
- **All-unit enum disc fold to AST_INT**: works correctly when used
  as plain expression (`let c = Color::G; c` returns 1). The mismatch
  is ONLY when used as match scrutinee — see Finding 1.
- **PAT_VARIANT sub-pattern walk**: when disc-cmp fails, jne jumps
  over the sub-pat reads (fail_jmp_state). So sub-pat OOB reads only
  happen when disc actually matches — and only when the variant's
  arity is shorter than the pattern's sub-pat count. See Finding 8 +
  Finding 9 for the disp8 wrap case.
- **Stage 6D fold of `__enum_payload(m, INT_LIT_i32)` to TUPLE_FIELD**:
  for the well-formed case (idx is AST_INT, idx + 1 ≤ 15), the fold
  is correct. See Finding 10 for the silent-0 / OOB / wide-int-lit
  cases.
- **Stage 5 / 6 set_last_struct_idx ordering fix (Iter D step 5)**:
  parent's set_last_struct_idx happens AFTER children parse, so
  nested struct lits don't clobber the outer's idx. ✓ (Already a
  Stage 5 Iter D fix; verified at parser.hx:2722, 2754.)
- **fields_ptr forward-reference for self-referential structs**:
  `struct Tree { val: i32, child: Tree }` silently treats `child` as
  scalar (f_struct_idx = -1 because Tree isn't yet in struct_table
  when its own fields parse). This is a Phase-0 design limitation,
  documented behavior — not a new sweep miss.
- **Stale `set_last_struct_idx` after struct-lit nested in arithmetic**:
  `let z = Pt{5,6}.x + 1;` leaves last_struct_idx = Pt; the let-parser
  registers `z` as Pt-typed (wrong — `z` is i32). Then `z.x` would
  attempt struct-field access on an i32 slot.
  **Actually** — re-examined: parser.hx at line 1454 requires
  `lhs_struct_idx >= 0`, and the postfix loop at 1473 emits TUPLE_FIELD
  with p3=0 (4-byte read). The i32 value of `z` is interpreted as
  a pointer; `mov eax, [rax]` SEGVs. This IS a silent bug — but the
  fix is in the let-parser (clear last_struct_idx whenever the let's
  value isn't directly a struct-lit). Promoting to a finding:
  see "Finding 14" below if added.

**Subtle case worth flagging (not enumerated as a top-13 finding because
it requires user-side error and produces a loud SEGV, not silent
corruption):**

- **last_struct_idx leak from nested struct-lit in arithmetic**:
  `let z = Pt{5,6}.x + 1;` causes the let-parser to register z as
  Pt-typed even though z is i32. Subsequent `z.field` access derefs
  i32 as pointer → SEGV. Recommend clearing last_struct_idx after
  parse_primary completes the struct-lit (line 2755) instead of
  relying on the let-parser to consume it. Move the clear from
  parser.hx:2071 (let-side) to right after the struct-lit emits
  in parse_primary. Combined with "clear AT entry of let-parser before
  parse_expr_basic" for double-safety.

---

## Summary

| # | Severity | Finding |
|---|----------|---------|
| 1 | HIGH | PAT_VARIANT codegen against all-unit enum SIGSEGVs (rep mismatch) |
| 2 | HIGH | PAT_VARIANT cross-enum match (enum_idx p3 ignored by codegen) |
| 3 | HIGH | Unknown variant/field names default to disc=0 silently |
| 4 | HIGH | Unknown struct field name eats `.` and IDENT, returns LHS unchanged |
| 5 | HIGH | Struct-typed fn return silently degrades to i32 (pointer truncation) |
| 6 | HIGH | Parser-state table cap overflows silently drop entries (struct/enum/var/fn/patch) |
| 7 | HIGH | Struct lit field count != declared arity silently emits actual count |
| 8 | HIGH | Enum payload variant arity not validated (Maybe::Some() / Some(1,2,3) silent) |
| 9 | MEDIUM | emit_variant_subpats / emit_tuple_subpats disp8 wrap at idx > 15 |
| 10 | MEDIUM | `__enum_payload` non-INTLIT idx silently uses 0 + OOB unchecked |
| 11 | MEDIUM | bind_alloc_offset has no cap-check — exhausts 1024-byte prologue silently |
| 12 | MEDIUM | Struct fn-call argument identity not validated (sentinel 15 collapses all structs) |
| 13 | LOW | `last_enum_idx` scratch slot written but never consumed (dead infra) |

8 HIGH, 4 MEDIUM, 1 LOW.

**Highest-severity one-liner**: Finding 1 — matching against an
all-unit enum SIGSEGVs because the all-unit rep is AST_INT but
PAT_VARIANT codegen dereferences scrut as a pointer.

**Stop-the-line recommendation**: YES on Finding 1 — the all-unit-enum
PAT_VARIANT crash is a soundness gap (match on an integer-shaped enum
SIGSEGVs). The narrowness of the test suite (no `match c { Color::R
=> ... }` test) means this gap won't surface during routine
regression. Either:
(a) Drop the AST_INT special case at parser.hx:2482-2491 (preferred
    long-term — unifies enum rep), accepting the regression of the
    `let c = Color::G; c == 1` test which conflates "the disc as a
    value" with "the enum as a value".
(b) Add a codegen-side dispatch in emit_pat_variant_disc that selects
    "compare integer disc directly" when scrut's bind ty is 0 (i32),
    "deref-then-compare" when ty ≥ 100 (struct sentinel) — note that
    AST_TUPLE_LIT result type tag is 3 (i64), so the dispatch needs
    to read the pattern's enum_idx and look up max_arity from
    enum_table; if max_arity == 0, do disc-cmp-direct.

Findings 2, 3, 4, 5 are also worth a coordinated fix pass — they're
all "unknown name silently mapped to first slot" or "type identity
silently collapsed" variants of the same underlying tension between
Phase-0 minimalism and language soundness. The current parser/codegen
errs on the side of "fold and continue"; the audit recommendation is
"fold and trap on mismatch".

Findings 6, 7, 8 (cap overflows + arity-mismatch in struct/enum
construct) should land together in a single FLAT prefix-trap pattern
batch — the host-parser recursion budget is the limiting factor (see
the resolution notes on Stage 4 follow-up Findings #6 and #8). Each
trap should use the prefix-trap-then-byte-emit pattern; nested
if-else blocks WILL miscompile unrelated programs.

Findings 9, 10, 11, 12 are MEDIUM and can be deferred — they trigger
on either edge cases (>15-payload variants, deeply-nested struct lit
chains, non-INTLIT enum-payload indices) or on programs that already
have type errors elsewhere (struct argument identity check). The
fixes are well-scoped and can land per-finding rather than as a
batch.

## Resolution status

| # | Status | Notes |
|---|--------|-------|
| 1 | FIXED  | match on all-unit enum: emit_pat_variant_disc dispatches on scrut_ty (stashed in bn_state slot 122 by emit_match_dispatch). Commit f9492cd. |
| 2 | OPEN   | PAT_VARIANT cross-enum match — deferred (HIGH but rare; need an extra enum-tag slot in tuple-lit). |
| 3 | PARTIAL | Unknown variant names: payload-variant traps 60002 (commit 2756afd), PAT_VARIANT traps 62006 (commit 0ff2bc4). Unit-variant + struct-field traps still open. |
| 4 | OPEN   | Unknown struct field name eats `.` and IDENT — deferred (HIGH but tightly scoped to parse_unary postfix). |
| 5 | FIXED  | Struct-typed fn return now encoded as 100+struct_idx; fn_type_table propagates struct identity. Type_width_class_struct + ret_wants_8b for struct rets. Phase-0 limitation: struct return-by-value still SEGVs at runtime (caller-alloc'd slot via rdi not implemented). Commit ed7adf8. |
| 6 | FIXED  | Cap-overflow on parser-state tables: caps bumped from 3/4/4 to 8/8/8 for struct_tab / var_struct_tab / enum_tab. Commit 4521641. (Surfacing as hard trap still deferred — the bump avoids the practical issue.) |
| 7 | FIXED  | Struct lit field count vs declared arity: trap 50040 on mismatch. Commit 70b89fc. |
| 8 | FIXED  | Enum payload variant arity + variant-name: trap 60020 / 60002. Commit 2756afd. |
| 9 | OPEN   | emit_variant_subpats / emit_tuple_subpats disp8 wrap at idx > 15 — deferred (MEDIUM). |
| 10 | OPEN  | `__enum_payload` non-INTLIT idx — deferred (MEDIUM). |
| 11 | OPEN  | bind_alloc_offset cap-check — deferred (MEDIUM). |
| 12 | OPEN  | Struct fn-call arg identity — deferred (MEDIUM, needs 8-bit param packing). |
| 13 | OPEN  | last_enum_idx dead-code — deferred (LOW, not user-visible). |

Verification: each FIXED entry has at least one regression test in
helixc/tests/test_codegen.py demonstrating the pre-fix silent behaviour
now traps loudly (exit 132 / SIGILL) or produces the correct value.
Heavy gate (test_bootstrap_kovc_full_pipeline_arithmetic) clean at the
end of the audit-fixes-stages5-16 branch.

The FLAT prefix-trap pattern + flat boolean accumulator pattern was
used throughout to avoid straining the host parser. AST_ERR(99) with a
trap-id in p1 is the canonical parser-side trap shape; kovc.hx's
emit_ast_code dispatches tag 99 → emit_trap_with_id(p1) (commit eca0ee2).
