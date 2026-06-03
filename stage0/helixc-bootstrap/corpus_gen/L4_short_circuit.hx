// L-4 (charter HELIX_COMPLETION.md §1.6 LOW): `&&` / `||` short-circuit.
// Promotes the K1.M-fix logical-short-circuit desugar (parser.hx:2267) from
// [impl] to [proven], INCLUDING A PROOF THAT THE RHS IS NOT EVALUATED when
// the LHS already decides the result.
//
// The desugar lowers boolean ops to AST_IF (which branches -- only the taken
// arm's code runs; kovc.hx:8933 emits `test rax,rax` + a conditional jump
// over the untaken arm):
//   a && b  ->  if a { b } else { 0 }      (RHS b is the THEN arm)
//   a || b  ->  if a { 1 } else { b }      (RHS b is the ELSE arm)
// So when `a` is false, `a && b` takes the else arm and b is skipped; when
// `a` is true, `a || b` takes the then arm and b is skipped. This program
// PROVES the skip by giving the RHS an observable side effect (writing a 1
// into an arena slot) and asserting the slot is UNCHANGED after a
// short-circuited expression -- if short-circuit were broken (RHS eagerly
// evaluated), the slot would be 1 and the exit code would move off 42.
//
// The self-host source uses nested `if`, never `&&`/`||`, so this promotion
// leaves the fixpoint byte-identical; the desugar is exercised only here.

// mark(slot, ret): record that the RHS RAN by storing 1 at arena[slot],
// then return `ret` (so the caller can also use it as a boolean operand).
// Side effect through an arena handle mirrors the gated vec_arena pattern.
fn mark(slot: i32, ret: i32) -> i32 {
    __arena_set(slot, 1);
    ret
}

fn main() -> i32 {
    // Four observable side-effect channels, all initialized to 0.
    let s0 = __arena_len(); __arena_push(0);   // && with FALSE lhs  -> RHS must be SKIPPED
    let s1 = __arena_len(); __arena_push(0);   // || with TRUE  lhs  -> RHS must be SKIPPED
    let s2 = __arena_len(); __arena_push(0);   // && with TRUE  lhs  -> RHS must RUN
    let s3 = __arena_len(); __arena_push(0);   // || with FALSE lhs  -> RHS must RUN

    // Build runtime-false / runtime-true so nothing folds at compile time.
    let mut acc = 0;
    let mut i = 0;
    while i < 5 { acc = acc + 1; i = i + 1; }   // acc = 5
    let lhs_false = acc > 100;                  // runtime false (0)
    let lhs_true = acc < 100;                   // runtime true  (1)

    // (1) FALSE && mark(s0): else arm (0) taken -> mark NOT called -> s0 stays 0.
    let r0 = lhs_false && mark(s0, 1);
    // (2) TRUE  || mark(s1): then arm (1) taken -> mark NOT called -> s1 stays 0.
    let r1 = lhs_true || mark(s1, 0);
    // (3) TRUE  && mark(s2): then arm (the RHS) taken -> mark CALLED -> s2 becomes 1.
    let r2 = lhs_true && mark(s2, 1);
    // (4) FALSE || mark(s3): else arm (the RHS) taken -> mark CALLED -> s3 becomes 1.
    let r3 = lhs_false || mark(s3, 1);

    // Read the side-effect channels back.
    let skipped0 = __arena_get(s0);   // must be 0 (RHS skipped)
    let skipped1 = __arena_get(s1);   // must be 0 (RHS skipped)
    let ran2 = __arena_get(s2);       // must be 1 (RHS ran)
    let ran3 = __arena_get(s3);       // must be 1 (RHS ran)

    // Short-circuit proof: the two skip-channels are 0, the two run-channels
    // are 1. Build a guard that is 1 ONLY when all four hold.
    let skip_ok = (skipped0 == 0) && (skipped1 == 0);   // 1 iff both skipped
    let run_ok = (ran2 == 1) && (ran3 == 1);            // 1 iff both ran
    let proof = skip_ok && run_ok;                       // 1 iff short-circuit correct

    // Result values must also be right: r0=0 (false&&_), r1=1 (true||_),
    // r2=1 (true&&1), r3=1 (false||1).
    let values_ok = (r0 == 0) && (r1 == 1) && (r2 == 1) && (r3 == 1);

    // Exit 42 iff BOTH the side-effect proof AND the boolean values are correct.
    if proof && values_ok { 42 } else { 7 }
}
