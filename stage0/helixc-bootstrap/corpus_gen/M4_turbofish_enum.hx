// T3 §1.6 M-4 (2026-06-03): turbofish on an ENUM CONSTRUCTOR.
// `Opt::<i32>::Some(payload)` (payload variant) and `Opt::<i32>::None`
// (unit variant) construct correctly and the payload round-trips through
// a match. Pre-fix the turbofish-on-enum-ctor form HUNG the compiler
// (mis-routed to the generic-fn turbofish branch -> rc 124 timeout).
//
// Runtime-derived inputs (loop-accumulated) so nothing constant-folds:
//   k = 2 after the loop -> p = Some(40 + k) = Some(42).
// Exit = (payload of the turbofish Some) when the match selects Some, plus
//        a unit-variant turbofish None that must select the None arm (adds 0).
// 42 is reachable ONLY if BOTH turbofish constructors parse + the match
// extracts the payload (each wrong path yields a distinct non-42 value).
enum Opt[T] { Some(T), None }
fn main() -> i32 {
    let mut k = 0;
    let mut i = 0;
    while i < 2 { k = k + 1; i = i + 1; }      // k = 2
    // payload-variant turbofish constructor
    let p = Opt::<i32>::Some(40 + k);          // Some(42)
    let a = match p {
        Opt::Some(x) => x,                     // 42
        Opt::None => 70
    };
    // unit-variant turbofish constructor
    let u = Opt::<i32>::None;
    let b = match u {
        Opt::Some(y) => y + 80,                // (wrong path -> >=80)
        Opt::None => 0                         // 0
    };
    a + b                                      // 42 + 0 = 42
}
