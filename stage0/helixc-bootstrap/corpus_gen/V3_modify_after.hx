// v1.3 V3 (2026-06-04): capture semantics check -- CAPTURE-BY-VALUE AT
// CLOSURE-CREATION. The closure snapshots the captured local's value when the
// |...| literal is evaluated; MODIFYING the original after creation does NOT
// change what the closure sees. (This documents the chosen by-value semantics;
// a by-REFERENCE capture would instead see 999.)
//
// x = 40 at the closure literal; c = |y| x + y captures 40 BY VALUE;
// then x is reassigned to 999; apply(c, 2) must STILL be 40 + 2 = 42
// (NOT 999 + 2 = 1001). All values are loop-derived so nothing constant-folds.
fn apply(f: i32, v: i32) -> i32 { f(v) }
fn main() -> i32 {
    let mut x = 0;
    let mut i = 0;
    while i < 8 { x = x + 5; i = i + 1; }   // x = 40
    let c = |y| x + y;                        // captures x = 40 BY VALUE (now)
    // Mutate the ORIGINAL local AFTER the closure was created.
    let mut j = 0;
    while j < 100 { x = x + 9; j = j + 1; }   // x = 40 + 900 = 940 (then +59 below)
    x = x + 59;                                // x = 999 (irrelevant to the closure)
    apply(c, 2)                                // by-value capture: 40 + 2 = 42
}
