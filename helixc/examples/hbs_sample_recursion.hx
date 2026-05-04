// hbs_sample_recursion.hx
//
// HBS dogfood: recursive functions taking enum payloads.
//
// STATUS: KNOWN LIMITATION — currently returns 1 not 120. The
// recursive call through an enum payload doesn't propagate state
// correctly. Plain i32 recursion works fine; this case stresses the
// match → enum-let → pass-by-value chain. Filed for follow-up.
//
// Computes factorial(n) using a Maybe wrapper to model "in progress"
// vs "done" state. Also exercises payload pattern extraction in a
// recursive call site.

enum State { Done, Continue(i32) }

@total
fn step(s: State, acc: i32) -> i32 {
    // If state is Done, return the accumulator.
    // If Continue(n) and n > 1, recurse with acc * n and Continue(n-1).
    // If Continue(1) (or 0), return acc * n (we're done).
    //
    // NOTE: enum constructors must currently be let-bound before
    // passing to a function (they're not yet supported as inline
    // expression args). Workaround: stage the next state in a let.
    match s {
        State::Done => acc,
        State::Continue(n) => {
            if n <= 1 {
                acc
            } else {
                let next = State::Continue(n - 1);
                step(next, acc * n)
            }
        }
    }
}

@total
fn factorial(n: i32) -> i32 {
    let init = State::Continue(n);
    step(init, 1)
}

fn main() -> i32 {
    // 5! = 120, but exit code is 8-bit → 120 fits.
    // Use 5 to keep the result < 256.
    factorial(5)
}
