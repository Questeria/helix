// L-3 (charter HELIX_COMPLETION.md §1.6 LOW) -- match-exhaustiveness, DOCUMENT-AS-BOUND
// + negative test. BOUND: there is NO match-exhaustiveness checking (spec §4).
// A `match` that omits a variant is NOT a compile error (Rust rejects it:
// "non-exhaustive patterns: `Err(_)` not covered"). The bootstrap accepts the
// incomplete match and runs the covered arm normally. Non-exhaustive match as a
// compile error is nice-to-have, not vision-blocking (§1.6 L-3); a proper
// exhaustiveness checker is v-next.
//
// NEGATIVE / bound-proving row: a payload-enum match that OMITS the `Err` arm,
// with the scrutinee on the covered `Ok` arm. Rust REJECTS this at compile
// time; the bootstrap ACCEPTS it and runs Ok(42) -> 42. Exit 42 proves the
// compiler does not reject non-exhaustive matches (the documented bound). If
// exhaustiveness checking is later added (reject), this row fails and flags the
// bound shift (intended).
enum Res { Ok(i32), Err(i32) }
fn main() -> i32 {
    let r = Res::Ok(42);
    match r {
        Res::Ok(x) => x        // Err arm MISSING -> Rust rejects; bootstrap accepts -> 42
    }
}
