// hbs_sample_option.hx
//
// HBS dogfood: payload pattern extraction with `Maybe<i32>`-style enum.
//
// Demonstrates:
//   - Payload-bearing enum constructors (Maybe::Some(42))
//   - Match with payload extraction (Maybe::Some(x) => x)
//   - Tag-only variants (Maybe::None) as a sentinel
//   - Multi-arg payloads (Pair::Cons(a, b))
//
// Limitation: enums passed to fn params lose payload access (Tier F #22).
// Workaround: do construction + match in main.

enum Maybe { None, Some(i32) }
enum Pair { Empty, Cons(i32, i32) }

fn main() -> i32 {
    // Compute Some(40) + Some(2) by extracting payloads and summing.
    let m1 = Maybe::Some(40);
    let m2 = Maybe::Some(2);

    let v1 = match m1 {
        Maybe::Some(x) => x,
        Maybe::None => 0,
    };
    let v2 = match m2 {
        Maybe::Some(x) => x,
        Maybe::None => 0,
    };
    let total1 = v1 + v2;     // 42

    // Pair::Cons unpacking: a + b
    let p = Pair::Cons(15, 25);
    let total2 = match p {
        Pair::Cons(a, b) => a + b,    // 40
        Pair::Empty => 0,
    };

    // Mix: total1 (42) is an i32; total2 is 40. Final answer = 42 (we
    // pick the option-extraction result as the demo).
    let r = total1;
    r
}
