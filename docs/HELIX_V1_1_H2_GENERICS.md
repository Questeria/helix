# Helix v1.1 H2 — Generics real codegen (monomorphization): design + scope

Synthesized from a 2-agent parallel investigation (2026-06-01) of parser.hx (parse side) +
kovc.hx (codegen side). H2 = make `<T>` produce REAL per-type codegen (no longer erased).

## Current state — NOT pure erasure (~60% scaffolded)
A partial monomorphization pipeline already exists and works for the easy case:
- **Parse**: turbofish `id::<f32>(..)` / `id::[f32](..)` and `fn id[T](..)` → a mono-request in
  `mr_tab`, a mangled name `id__f32` (`mangle_name_into_arena`, parser.hx:1599), and a packed
  4-bit type-tag vector (`ty_ident_to_tag`, parser.hx:1540: i32=0,f32=1,f64=2,i64=3,bf16=4,
  u32=6,u8=7,u16=8,u64=9,i8=10,i16=11). Sites: parser.hx:7543-7650 (fn turbofish),
  8019-8210 (struct turbofish/clone), 12063-12118 (generic fn parse), 14642-15094 (generic struct).
- **Mono pass**: `monomorphize_pass` (parser.hx:9910) finds each template (FN_DECL slot6==1) and
  appends a concrete clone per request (param/ret type tags substituted), plus an auto i32 default.
- **Codegen**: kovc.hx SKIPS templates (slot6==1, kovc.hx:9681/9771), emits only concrete clones,
  registers them in `fn_table` (kovc.hx:1854), resolves calls via backpatch (kovc.hx:10012-10091)
  with a `__i32` end-append fallback (kovc.hx:10021-10076, K1.F21). Op-selection is FULLY
  type-driven: AST_ADD emits addss (f32) / add (i32) / addsd (f64) / REX.W (i64) by operand type
  (kovc.hx:7407-7549); params are bound with their type tag (kovc.hx:9840-9862).
- **=> simple scalar generic FUNCTIONS via turbofish ALREADY monomorphize correctly**
  (`id::<f32>(x)` with param arithmetic emits addss; `id::<i32>` emits add).

## The 3 gaps (where erasure bites) = the H2 work
- **d.1 Shallow body clone** — `clone_with_rewrite` (parser.hx:9884-9903) rewrites only AST_CALL
  *names*; it SHARES every other body node. So a `T`-typed local (`let y: T = ..`), a T literal,
  or nested generic calls beyond one-level `T::method` are frozen to one shape. FIX: deep-clone
  the body with per-instantiation type substitution. **Fixpoint-safe-by-construction** (only
  reached for generic templates, which the self-host source never has).
- **d.2 Generic struct fields scalar-erased** — the struct_tab field entry is 3 slots
  (name_s, name_l, struct_idx); a `T=f32` field stores struct_idx = -1 with NO scalar dtype
  (parser.hx:8126-8138, 1156/1229). So `Box<f32>` is physically identical to `Box<i32>`
  (i32-shaped 4-byte field). FIX: carry the scalar type tag through struct fields end-to-end
  (widen the entry / sentinel-encode) — struct_tab_add/lookup + kovc field load/store
  (kovc.hx:7118-7140) + expr_type field-read (kovc.hx:1395-1419, hard-coded i32/i64). This is the
  fix for the headline (Vec<T>/Option<T>). **NOT fixpoint-safe** (touches non-generic struct
  codegen the self-host source uses) -> requires the FULL fixpoint gate.
- **d.3 Bare-call default-i32** — `monomorphize_pass`'s default synthesis + the `__i32` backpatch
  fallback both assume i32, so bare `id(3.14_f32)` (no turbofish) resolves to `id__i32` -> integer
  ops on f32 bits -> silent wrong value. FIX: argument-type inference, OR scope to
  "explicit turbofish required for non-i32 scalar generics" (a legitimate v1.1 scope decision).

## Concrete failing case (today, silent wrong result)
```
struct Box[T] { v: T }
fn main() -> i32 {
    let a = Box::<f32>{ v: 2.0 };
    let b = Box::<f32>{ v: 3.0 };
    __f32_to_i32(a.v + b.v)        // a.v/b.v read as i32 -> integer add on f32 bits -> garbage, not 5.0
}
```
No existing corpus/example/dogfood defines a generic fn or struct (generics are wholly untested) —
consistent with v1.0 calling generics the biggest gap.

## v1.1 H2 scope + sequence (low-risk first)
The charter H2 corpus = Vec<T>, Option<T>, generic fns over differing types, a generic struct/impl.
- **H2a (first, low-risk)**: generic FUNCTIONS over differing scalar types (turbofish) — deepen
  `clone_with_rewrite` (d.1). Corpus: a generic fn over i32 AND f32 AND i64, correct per-type.
  Fixpoint-safe-by-construction -> probe-first + skip the full fixpoint until the broad corpus.
- **H2b (the headline, higher-risk)**: generic STRUCTS / containers (Vec<T>, Option<T>) over
  differing types — carry scalar tags through struct fields (d.2). FULL fixpoint gate. Corpus:
  Box<f32> field math correct; a generic container over 2 element types.
- **H2c**: bare-call non-i32 — implement inference OR document "turbofish required for non-i32
  scalar generics" (scope decision; explicit turbofish is a reasonable v1.1 bound).

## Fixpoint-safety map (dev-opt #17)
- CONFIRMED: lexer.hx/parser.hx/kovc.hx contain 0 turbofish + 0 generic fn/struct decls. So d.1
  and d.3 (generics-only paths) CANNOT perturb K2==K3 -> probe-first, defer the ~7min fixpoint
  to the broad corpus.
- EXCEPTION: d.2 widens struct_tab + struct field load/store (paths the i32 self-host source USES)
  -> d.2 MUST pass the full fixpoint gate. Direct the gating attention there.

## Honest scope: MEDIUM, not a rewrite
The backend needs ~zero new op-emission (type-driven codegen already handles i32/f32/f64/i64/struct).
The work is parser-side: a deep-clone-with-type-substitution (d.1) + the struct-field scalar-tag
carry (d.2, the risky one) + widening two i32-hardcoded assumptions (d.3). Watch the struct_tab cap
(currently 8; mono clones multiply entries; must stay < 200 per parser.hx:364-370).

*Authored 2026-06-01 from the parallel investigation (agents a4e206bb + ab739e6b). Guides the H2 implementation.*
