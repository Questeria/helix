// H-2 corpus (charter §1.6 HIGH): rich String -- str_new / str_push_byte /
// str_len / str_byte_at / str_concat / str_eq, INCLUDING growth-on-push and
// a concat that crosses the initial capacity. Inlined from stdlib/string.hx
// (the gate compiles each corpus program standalone; no external-module
// loader yet -- same inlining convention as H1_vec.hx / H1_hashmap.hx).
//
// A String value is a `String { h: <handle> }` newtype wrapping an arena
// handle -- EXACTLY H1_vec's `Vec::<i32>{ h: vec_with_cap(2) }` pattern. The
// constructors (str_new / str_concat) return the HANDLE (i32), not a String
// value, because returning a struct BY VALUE from a function mis-lowers in
// this from-raw compiler (a v-next codegen gap; passing a struct by value as
// a PARAMETER is fine). So each String is built as `String { h: str_new() }`
// / `String { h: str_concat(a,b) }`, and the ops take String by value.
//
// A String LITERAL used as a runtime value lowers to `mov eax, 0` in kovc (no
// runtime byte pointer), so the rich String is built BYTE-BY-BYTE and every
// op below runs at RUNTIME over arena-stored bytes (nothing folds):
//   "Hel" (72,101,108) ++ "lix" (108,105,120) = "Hellix" via str_concat,
//   whose result starts cap 4 and receives 6 bytes -> forces a grow 4->8;
//   index byte[0]=72('H') and byte[5]=120('x') via str_byte_at; str_eq proves
//   EQUAL (concat vs an independently-built "Hellix"), UNEQUAL-same-length
//   ("Hel" vs "lix"), UNEQUAL-diff-length ("Hel" vs "Hellix", short-circuit),
//   and a one-byte diff ("Helliy"). Exit = str_len("Hellix") * 7 = 42.

struct String { h: i32 }

fn str_with_cap(cap0: i32) -> i32 {
    let h = __arena_push(0);
    __arena_push(0);
    __arena_push(0);
    let data = __arena_len();
    let mut i = 0;
    while i < cap0 { __arena_push(0); i = i + 1; };
    __arena_set(h + 1, cap0);
    __arena_set(h + 2, data);
    h
}
fn str_grow(h: i32) -> i32 {
    let len = __arena_get(h);
    let cap = __arena_get(h + 1);
    let old = __arena_get(h + 2);
    let ncap = cap * 2;
    let ndata = __arena_len();
    let mut i = 0;
    while i < ncap { __arena_push(0); i = i + 1; };
    let mut j = 0;
    while j < len { __arena_set(ndata + j, __arena_get(old + j)); j = j + 1; };
    __arena_set(h + 1, ncap);
    __arena_set(h + 2, ndata);
    0
}
fn str_push_raw(h: i32, b: i32) -> i32 {
    let len = __arena_get(h);
    let cap = __arena_get(h + 1);
    if len >= cap { str_grow(h); };
    let data = __arena_get(h + 2);
    __arena_set(data + len, b & 255);
    __arena_set(h, len + 1);
    0
}
fn str_new() -> i32 { str_with_cap(4) }
fn str_push_byte(s: String, b: i32) -> i32 { str_push_raw(s.h, b) }
fn str_len(s: String) -> i32 { __arena_get(s.h) }
fn str_byte_at(s: String, i: i32) -> i32 { __arena_get(__arena_get(s.h + 2) + i) }
fn str_concat(a: String, b: String) -> i32 {
    let rh = str_new();
    let la = str_len(a);
    let mut i = 0;
    while i < la { str_push_raw(rh, str_byte_at(a, i)); i = i + 1; };
    let lb = str_len(b);
    let mut j = 0;
    while j < lb { str_push_raw(rh, str_byte_at(b, j)); j = j + 1; };
    rh
}
fn str_eq(a: String, b: String) -> i32 {
    let la = str_len(a);
    let lb = str_len(b);
    if la != lb {
        0
    } else {
        let mut i = 0;
        let mut same = 1;
        while i < la {
            if str_byte_at(a, i) != str_byte_at(b, i) { same = 0; };
            i = i + 1;
        };
        same
    }
}
impl String {
    fn len(self) -> i32 { __arena_get(self.h) }
    fn push_byte(self, b: i32) -> i32 { str_push_raw(self.h, b) }
    fn byte_at(self, i: i32) -> i32 { __arena_get(__arena_get(self.h + 2) + i) }
}

fn main() -> i32 {
    // "Hel": build a 3-byte String via the public push-byte op.
    let s1 = String { h: str_new() };
    str_push_byte(s1, 72);                 // 'H'
    str_push_byte(s1, 101);                // 'e'
    str_push_byte(s1, 108);                // 'l'
    // "lix"
    let s2 = String { h: str_new() };
    str_push_byte(s2, 108);                // 'l'
    str_push_byte(s2, 105);                // 'i'
    str_push_byte(s2, 120);                // 'x'
    if str_len(s1) != 3 { return 80; };
    if str_len(s2) != 3 { return 81; };

    // index a byte back out (round-trip the push).
    if str_byte_at(s1, 0) != 72  { return 82; };   // 'H'
    if str_byte_at(s1, 2) != 108 { return 83; };   // 'l'

    // concat -> "Hellix" (6 bytes). The result starts cap 4 and receives 6
    // bytes -> the grow+copy path (4->8) runs inside str_concat.
    let cat = String { h: str_concat(s1, s2) };
    if str_len(cat) != 6 { return 84; };
    if str_byte_at(cat, 0) != 72  { return 85; };  // 'H' (first of s1)
    if str_byte_at(cat, 3) != 108 { return 86; };  // 'l' (first of s2)
    if str_byte_at(cat, 5) != 120 { return 87; };  // 'x' (last of s2)

    // build "Hellix" independently, byte-by-byte (also crosses cap 4 -> 8),
    // and prove str_eq sees them as EQUAL.
    let refs = String { h: str_new() };
    str_push_byte(refs, 72);
    str_push_byte(refs, 101);
    str_push_byte(refs, 108);
    str_push_byte(refs, 108);
    str_push_byte(refs, 105);
    str_push_byte(refs, 120);
    if str_len(refs) != 6 { return 88; };          // growth happened (4->8)
    if str_eq(cat, refs) != 1 { return 89; };      // EQUAL: same len + bytes

    // UNEQUAL, same length, different bytes ("Hel" vs "lix").
    if str_eq(s1, s2) != 0 { return 70; };
    // UNEQUAL, different length ("Hel" vs "Hellix") -- the short-circuit arm.
    if str_eq(s1, cat) != 0 { return 71; };
    // a one-byte difference must also be caught ("Helliy" differs only at idx 5).
    let near = String { h: str_new() };
    str_push_byte(near, 72);
    str_push_byte(near, 101);
    str_push_byte(near, 108);
    str_push_byte(near, 108);
    str_push_byte(near, 105);
    str_push_byte(near, 121);                       // 'y' not 'x'
    if str_eq(cat, near) != 0 { return 72; };       // differs at the last byte

    // also exercise the impl-method form (mirrors H-1's impl<T> Vec<T>, whose
    // receiver is likewise a struct-LITERAL binding).
    let m = String { h: str_new() };
    m.push_byte(65);                                // 'A'
    m.push_byte(66);                                // 'B'
    if m.len() != 2 { return 73; };
    if m.byte_at(1) != 66 { return 74; };

    // exit = len("Hellix") * 7 = 6 * 7 = 42 (runtime-derived).
    str_len(cat) * 7
}
