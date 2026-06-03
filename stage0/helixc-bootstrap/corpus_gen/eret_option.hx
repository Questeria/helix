// T3 §1.6 enum-return-by-value (2026-06-03): a 2-variant payload enum
// returned BY VALUE from a fn, then matched in the caller. Complements
// arm_enum_payload3 (3-variant) with the canonical Option/Result shape and
// a runtime-selected variant. Pre-fix: pick() returned a dangling pointer
// the caller stored 32-bit + the match read it i32-direct -> SIGILL (132).
// The fix tags the enum return pointer-rep (100+8+enum_idx), 64-bit-stores
// it, copies the [disc,payload] run into the caller frame, and drives the
// match pointer path. Runtime k selects Some so the payload arm runs.
enum Opt { None, Some(i32) }
fn pick(k: i32) -> Opt {
    if k == 0 { Opt::None } else { Opt::Some(40 + k) }
}
fn main() -> i32 {
    let mut k = 0;
    let mut i = 0;
    while i < 2 { k = k + 1; i = i + 1; }   // k = 2 -> Some(42)
    let o = pick(k);
    match o {
        Opt::None => 0,
        Opt::Some(v) => v                    // 40 + 2 = 42
    }
}
