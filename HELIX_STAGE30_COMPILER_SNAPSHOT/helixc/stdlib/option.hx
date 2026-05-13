// helixc/stdlib/option.hx — Option<T> for absent-value handling.
//
// Phase 1.9: a thin sum type with Some(T) | None. Use this to represent
// "value or nothing" without resorting to sentinel ints. Pattern-match
// to extract:
//
//   let v: Option = Option::Some(42);
//   match v {
//       Option::Some(x) => x,
//       Option::None => 0,
//   }
//
// LIMITATION: in Phase 1.9 Option is i32-specialised because generics over
// enum variants require type-tagged-payload codegen (Phase 2 item). For
// now the most common case (Option<i32>) is enough for AGI work.
//
// License: Apache 2.0

@pure
enum Option {
    Some(i32),
    None,
}

@pure
fn option_unwrap_or(o: Option, default_v: i32) -> i32 {
    match o {
        Option::Some(x) => x,
        Option::None => default_v,
    }
}

@pure
fn option_is_some(o: Option) -> i32 {
    match o {
        Option::Some(_) => 1,
        Option::None => 0,
    }
}

@pure
fn option_is_none(o: Option) -> i32 {
    match o {
        Option::Some(_) => 0,
        Option::None => 1,
    }
}

@pure
fn option_or_zero(o: Option) -> i32 {
    match o {
        Option::Some(x) => x,
        Option::None => 0,
    }
}

@pure
fn option_or_neg(o: Option) -> i32 {
    match o {
        Option::Some(x) => x,
        Option::None => 0 - 1,
    }
}

@pure
fn option_eq_some(o: Option, target: i32) -> i32 {
    match o {
        Option::Some(x) => if x == target { 1 } else { 0 },
        Option::None => 0,
    }
}

// Pairwise max with None as additive identity:
//   Some(x), Some(y) -> max(x, y)
//   Some(x), None    -> x
//   None, Some(y)    -> y
//   None, None       -> 0   (caller should pre-filter if 0 is a valid datum)
@pure
fn option_max(a: Option, b: Option) -> i32 {
    match a {
        Option::Some(av) => match b {
            Option::Some(bv) => if av > bv { av } else { bv },
            Option::None => av,
        },
        Option::None => match b {
            Option::Some(bv) => bv,
            Option::None => 0,
        },
    }
}

// Pairwise min with None as additive identity (matches option_max):
//   Some(x), Some(y) -> min(x, y)
//   Some(x), None    -> x        (None is identity, not +inf)
//   None, Some(y)    -> y
//   None, None       -> 0        (caller should pre-filter if 0 is a valid datum)
@pure
fn option_min(a: Option, b: Option) -> i32 {
    match a {
        Option::Some(av) => match b {
            Option::Some(bv) => if av < bv { av } else { bv },
            Option::None => av,
        },
        Option::None => match b {
            Option::Some(bv) => bv,
            Option::None => 0,
        },
    }
}

// Sum two options with None as additive 0:
//   Some(x), Some(y) -> x + y
//   Some(x), None    -> x
//   None, Some(y)    -> y
//   None, None       -> 0
@pure
fn option_sum(a: Option, b: Option) -> i32 {
    match a {
        Option::Some(av) => match b {
            Option::Some(bv) => av + bv,
            Option::None => av,
        },
        Option::None => match b {
            Option::Some(bv) => bv,
            Option::None => 0,
        },
    }
}

// Structural equality of two Options:
//   Some(x), Some(y) -> 1 iff x == y
//   None, None       -> 1
//   otherwise        -> 0
@pure
fn option_eq(a: Option, b: Option) -> i32 {
    match a {
        Option::Some(av) => match b {
            Option::Some(bv) => if av == bv { 1 } else { 0 },
            Option::None => 0,
        },
        Option::None => match b {
            Option::Some(_) => 0,
            Option::None => 1,
        },
    }
}

// Companion to option_or_zero: returns 1 (multiplicative identity) on None,
// useful for fold-product chains where None should not absorb the running product.
@pure
fn option_or_one(o: Option) -> i32 {
    match o {
        Option::Some(x) => x,
        Option::None => 1,
    }
}
