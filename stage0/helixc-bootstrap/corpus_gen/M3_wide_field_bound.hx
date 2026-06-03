// M-3 (charter §1.6 MED) — DOCUMENT-AS-BOUND + negative test.
//
// 8-byte SCALAR struct fields (f64 / i64 / u64) are NOT fully supported in
// v1.2. A struct field's slot IS written with the full 64-bit value at
// construction (AST_TUPLE_LIT stores `mov [rbp+disp], rax` REX.W, 8 bytes,
// kovc.hx:7338), but the field READ (AST_TUPLE_FIELD, kovc.hx:7237) only
// emits an 8-byte (REX.W) load when p3==1, and the parser sets p3==1 ONLY for
// nested-struct-typed fields (a child pointer) — NOT for f64/i64/u64 scalar
// fields, whose decl-time type encodes as struct_idx == -1 (same as i32),
// indistinguishable from a 4-byte scalar (parser.hx:2786-2813 + the field-type
// encoding at parser.hx:15495 maps any non-struct, non-generic type name to
// struct_tab_lookup_idx == -1). So `let x = a.field` reads only the LOW 32
// bits of an 8-byte field.
//
// OBSERVED v1.2 behavior (honest):
//   * i64/u64 field read  -> SILENT low-32-bit truncation (e.g. a field holding
//     5_000_000_000_i64 reads back as 705_032_704; 705_032_704/100_000_000 = 7,
//     not 50). This is the one silent-wrong residual, recorded as v-next.
//   * f64 field read  -> the 4-byte-read result is i32-typed, so using it in f64
//     arithmetic (`a.v + 0.0_f64`) hits the mixed-type guard and TRAPS LOUD
//     (ud2 / SIGILL, exit 132) — it FAILS CLOSED, never emits wrong float math.
//
// This fixture exercises the f64 path and is gated at exit 132 (the LOUD trap),
// proving the f64 8-byte-field bound fails closed. The full fix — decl-time
// detection of 8-byte scalar field types + a p3==1 read for them + f64 type-
// tracking of the field-read result + generic use-site monomorphization
// (struct Box[T] with T=f64/i64, the unimplemented Stage-28.11 INC-3b) — is a
// v-next item, NOT a finale-blocker (charter §1.3 permits a MED item to be a
// documented v1.x bound with a negative test). The non-generic 4-byte path is
// unaffected and gated (gen_box_f32 -> 5; sret_*field struct returns -> 42).
//
// (DO NOT "fix" this fixture into a passing exit: the feature it tests is the
//  documented bound; the loud f64 trap IS the recorded fail-closed behavior.)
struct B { v: f64 }
fn main() -> i32 {
    let a = B { v: 100000.25_f64 };
    let x: f64 = a.v + 0.0_f64;   // <- 4-byte read of an 8-byte f64 field is
                                  //    i32-typed -> mixed-type guard -> ud2/SIGILL 132
    x as i32
}
