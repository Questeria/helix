// hbs_pattern_struct_return.hx
//
// HBS pattern: simulating struct-return-by-value via arena output
// params. Until the compiler supports true struct returns (Tier F
// follow-up), self-host code uses this idiom:
//   - Caller allocates N consecutive arena slots (one per field).
//   - Callee writes its result into those slots.
//   - Caller reads the slots.
// This is identical to how the recursive-enum machinery works.
//
// Example: simulate `fn build_pair() -> Pair { Pair { a: 1, b: 2 } }`
// where Pair has two i32 fields.

// Reserve N consecutive arena slots, return the start index.
@total
fn arena_reserve(n: i32) -> i32 {
    let start = __arena_len();
    let mut i: i32 = 0;
    while i < n {
        __arena_push(0);
        i = i + 1;
    }
    start
}

// "Constructor" — writes into a pre-allocated arena slot range.
@total
fn build_pair_into(out: i32, a: i32, b: i32) -> i32 {
    __arena_set(out, a);
    __arena_set(out + 1, b);
    out
}

// "Accessor" — reads field by name (encoded as offset).
@total
fn pair_a(idx: i32) -> i32 { __arena_get(idx) }

@total
fn pair_b(idx: i32) -> i32 { __arena_get(idx + 1) }

@total
fn pair_sum(idx: i32) -> i32 { pair_a(idx) + pair_b(idx) }

fn main() -> i32 {
    // Allocate 2 slots, "return" a Pair{a:10, b:32} by writing into them.
    let p1 = build_pair_into(arena_reserve(2), 10, 32);
    let p2 = build_pair_into(arena_reserve(2), 0, 0 - 100);

    // Read fields.
    let s1 = pair_sum(p1);                  // 42
    let s2 = pair_sum(p2);                  // -100

    // 42 + (-100) + 100 = 42
    s1 + s2 + 100
}
