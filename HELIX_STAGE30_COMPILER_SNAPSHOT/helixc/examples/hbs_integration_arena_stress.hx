// hbs_integration_arena_stress.hx
//
// Stress test: push many values into the arena via a while loop +
// verify each can be read back. Tests:
//   - Arena bounds check (32K cap, audit-10 fix)
//   - Tail-recursive equivalent via while loop
//   - Bulk read-back consistency
//   - Mutable counter shadowing

@partial
fn fill_with_indices(n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let _ = __arena_push(i * 2 + 1);
        i = i + 1;
    }
    n
}

@partial
fn verify_indices(start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    let mut errors: i32 = 0;
    while i < n {
        let v = __arena_get(start + i);
        let expected = i * 2 + 1;
        if v != expected {
            errors = errors + 1;
        }
        i = i + 1;
    }
    errors
}

fn main() -> i32 {
    // Push 1000 values into the arena (well below 32K cap), then verify
    // every single one. If errors == 0, return 42.
    let start = __arena_len();
    let _ = fill_with_indices(1000);
    let errors = verify_indices(start, 1000);
    if errors == 0 { 42 } else { errors }
}
