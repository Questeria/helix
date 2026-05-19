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
//
// Cycle 3 R1 fix batch 20 (RT MEDIUM-7): Some(av) + Some(bv) was raw i32
// addition; INT32_MAX + 1 silently wrapped to INT32_MIN. Post-fix: i64
// intermediate + INT32 saturation, matching the iterators.hx vec_sum_pure
// template (restart 53 A2). Fold-style accumulator chains over Options
// now saturate cleanly instead of wrapping.
@pure
fn option_sum(a: Option, b: Option) -> i32 {
    match a {
        Option::Some(av) => match b {
            Option::Some(bv) => {
                let total: i64 = (av as i64) + (bv as i64);
                let hi: i64 = 2147483647_i64;
                let lo: i64 = (0_i64 - 2147483647_i64) - 1_i64;
                if total > hi { 2147483647 }
                else if total < lo { (0_i32 - 2147483647_i32) - 1_i32 }
                else { total as i32 }
            },
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
