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
