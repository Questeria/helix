// hbs_sample_recursion.hx
//
// HBS dogfood: recursive functions taking enum payloads. Demonstrates
// that the match → enum-let → pass-by-value chain works across
// self-call boundaries — exactly the shape an AST-walking compiler
// pass needs. Computes 5! = 120 via Maybe-style state machine.
//
// LIMITATION: inline enum constructors as fn args (e.g.
// step(State::Continue(n-1), ...)) don't yet work — must let-bind
// first. Workaround used below.
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
