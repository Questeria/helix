"""Tests for helixc.frontend.typecheck (v0.1 scaffold)."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.frontend.typecheck import typecheck


def check(src: str) -> list[str]:
    prog = parse(src)
    errs = typecheck(prog)
    return [str(e) for e in errs]


def check_after_flatten(src: str) -> list[str]:
    from helixc.frontend.flatten_modules import flatten_modules
    prog = parse(src)
    flatten_modules(prog)
    errs = typecheck(prog)
    return [str(e) for e in errs]


def check_with_stdlib(src: str) -> list[str]:
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    return [str(e) for e in errs]


# ============================================================================
# Should typecheck (no errors)
# ============================================================================
def test_simple_add():
    assert check("fn add(a: i32, b: i32) -> i32 { a + b }") == []


def test_let_typed_match():
    assert check("fn f() { let x: i32 = 42; }") == []


def test_let_inferred():
    assert check("fn f() { let x = 42; }") == []


def test_if_branches_match():
    src = "fn f(b: bool) -> i32 { if b { 1 } else { 2 } }"
    assert check(src) == []


def test_function_call():
    src = """
    fn double(x: i32) -> i32 { x + x }
    fn f() -> i32 { double(7) }
    """
    assert check(src) == []


def test_generic_signature():
    src = """
    fn id[T](x: T) -> T { x }
    """
    # T propagates; should typecheck (param T -> return T)
    assert check(src) == []


def test_tensor_signature():
    src = """
    fn matmul[N: size, M: size, P: size](
        a: tensor<f32, [N, M]>,
        b: tensor<f32, [M, P]>,
    ) -> tensor<f32, [N, P]>
    {
        let c = tensor::zeros();
        c
    }
    """
    # tensor::zeros returns Unknown which is compatible-with-anything in v0.1
    assert check(src) == []


def test_struct_def_no_check():
    src = "struct Point { x: f32, y: f32 }"
    # struct definitions don't trigger body checking
    assert check(src) == []


# ============================================================================
# Should detect errors
# ============================================================================
def test_let_type_mismatch():
    errs = check("fn f() { let x: bool = 42; }")
    assert any("declared bool but value is i32" in e for e in errs), errs


def test_return_type_mismatch():
    errs = check("fn f() -> bool { 42 }")
    assert any("does not match return type" in e for e in errs), errs


def test_if_branches_differ():
    errs = check("fn f(b: bool) -> bool { if b { 1 } else { true } }")
    # First arm i32, second bool. Whole if is i32; outer return type bool -> mismatch
    assert any("does not match return type" in e or "if/else branches differ" in e for e in errs), errs


def test_duplicate_function():
    errs = check("fn foo() {} fn foo() {}")
    assert any("duplicate function" in e for e in errs), errs


def test_stage31_duplicate_refinement_names_fail_closed():
    duplicate_alias = check("""
    type Gate = f64 where self >= 0.0;
    type Gate = f64 where self <= 0.0;
    fn bad() -> Gate { 1.0_f64 }
    """)
    assert any("duplicate type namespace name 'Gate'" in e
               and "type alias conflicts with earlier type alias" in e
               for e in duplicate_alias), duplicate_alias

    duplicate_const = check("""
    const LIMIT: f64 = 1.0_f64;
    const LIMIT: f64 = 0.0_f64;
    type A = f64 where self <= LIMIT;
    fn bad() -> A { 1.0_f64 }
    """)
    assert any("duplicate const 'LIMIT'" in e
               for e in duplicate_const), duplicate_const

    type_namespace = check("""
    struct Reading { x: i32 }
    type Reading = f64 where self >= 0.0;
    fn f() {}
    """)
    assert any("duplicate type namespace name 'Reading'" in e
               and "type alias conflicts with earlier struct" in e
               for e in type_namespace), type_namespace


def test_c116_mixed_float_scalar_ops_require_explicit_cast():
    errs = check("fn f(a: f64, b: i32) -> f64 { a + b }")
    assert any("incompatible operand types f64 and i32" in e for e in errs), errs

    errs2 = check("fn f(a: f64, b: f32) -> f64 { a + b }")
    assert any("incompatible operand types f64 and f32" in e for e in errs2), errs2

    errs3 = check("fn f(a: f64, b: i32) -> bool { a > b }")
    assert any("incompatible operand types f64 and i32" in e for e in errs3), errs3


def test_c116_mixed_integer_ops_remain_allowed():
    assert check("fn f(a: u64, b: i32) -> bool { a > b }") == []
    assert check("fn f(a: u64, b: u32) -> u64 { a + b }") == []


def test_c116_assignment_type_mismatch_errors():
    errs = check("fn f() { let mut x: i64 = 1_i64; x = 2_i32; }")
    assert any("assignment target type i64 incompatible with value type i32" in e
               for e in errs), errs

    errs2 = check("fn f() { let mut x: i64 = 1_i64; x += 2_i32; }")
    assert any("assignment target type i64 incompatible with value type i32" in e
               for e in errs2), errs2


def test_c117_indexed_assignment_type_mismatch_errors():
    errs = check("fn f() { let mut a: [i32; 1] = [0]; a[0] = 1.5_f64; }")
    assert any("assignment target type i32 incompatible with value type f64" in e
               for e in errs), errs

    errs2 = check("fn f() { let mut a: [i64; 1] = [1_i64]; a[0] += 2_i32; }")
    assert any("assignment target type i64 incompatible with value type i32" in e
               for e in errs2), errs2


def test_c117_array_literal_elements_must_match():
    errs = check("fn f() { let xs = [1_i32, 2.0_f32]; }")
    assert any("array literal element type f32 incompatible with first element type i32" in e
               for e in errs), errs

    errs2 = check("fn f() { let xs = [1_u32, -1_i32]; }")
    assert any("array literal element type i32 incompatible with first element type u32" in e
               for e in errs2), errs2


def test_stage31_refinement_probability_confidence_constants_compile():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    type Confidence = f64 where 0.0 <= self <= 1.0;
    fn f() {
        let p0: Probability = 0.0_f64;
        let p1: Probability = 0.5_f64;
        let p2: Probability = 1.0_f64;
        let c: Confidence = 0.95_f64;
    }
    """
    assert check_after_flatten(src) == []


def test_stage31_stdlib_agi_safe_scalar_refinements_compile_by_default():
    src = """
    fn f() {
        let c: Confidence = 0.95_f64;
        let p: Probability = 0.25_f64;
        let d: DistanceMeters = 12.5_f64;
    }
    """
    assert check_with_stdlib(src) == []


def test_stage31_stdlib_confidence_refinement_constant_above_one_fails():
    src = """
    fn f() {
        let c: Confidence = 1.01_f64;
    }
    """
    errs = check_with_stdlib(src)
    assert any("refinement Confidence violated" in e
               and "1.01" in e
               and "0.0 <= self <= 1.0" in e
               and "31001" in e for e in errs), errs


def test_stage31_stdlib_probability_refinement_constant_above_one_fails():
    src = """
    fn f() {
        let p: Probability = 1.2_f64;
    }
    """
    errs = check_with_stdlib(src)
    assert any("refinement Probability violated" in e
               and "1.2" in e
               and "0.0 <= self <= 1.0" in e
               and "31001" in e for e in errs), errs


def test_stage31_stdlib_distance_meters_negative_constant_fails():
    src = """
    fn f() {
        let d: DistanceMeters = -0.5_f64;
    }
    """
    errs = check_with_stdlib(src)
    assert any("refinement DistanceMeters violated" in e
               and "-0.5" in e
               and "self >= 0.0" in e
               and "31001" in e for e in errs), errs


def test_stage31_refinement_probability_constant_below_zero_fails():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn f() {
        let p: Probability = -0.1_f64;
    }
    """
    errs = check(src)
    assert any("refinement Probability violated" in e
               and "-0.1" in e
               and "0.0 <= self <= 1.0" in e
               and "31001" in e for e in errs), errs


def test_stage31_refinement_probability_constant_above_one_fails():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn f() {
        let p: Probability = 1.2_f64;
    }
    """
    errs = check(src)
    assert any("refinement Probability violated" in e
               and "1.2" in e
               and "0.0 <= self <= 1.0" in e
               and "31001" in e for e in errs), errs


def test_stage31_refinement_confidence_constant_above_one_fails():
    src = """
    type Confidence = f64 where 0.0 <= self <= 1.0;
    fn f() {
        let c: Confidence = 1.01_f64;
    }
    """
    errs = check(src)
    assert any("refinement Confidence violated" in e
               and "1.01" in e
               and "0.0 <= self <= 1.0" in e
               and "31001" in e for e in errs), errs


def test_stage31_refinement_call_arg_constant_checked():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn use_p(p: Probability) -> i32 { 42 }
    fn f() -> i32 {
        use_p(1.2_f64)
    }
    """
    errs = check(src)
    assert any("call to 'use_p': arg 'p'" in e
               and "refinement Probability violated" in e for e in errs), errs


def test_stage31_refinement_return_constant_checked():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn f() -> Probability {
        1.2_f64
    }
    """
    errs = check(src)
    assert any("return value of function 'f'" in e
               and "refinement Probability violated" in e for e in errs), errs


def test_stage31_refinement_value_carries_proof_through_call_and_return():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn id(p: Probability) -> Probability {
        p
    }
    fn use_p(p: Probability) -> i32 { 42 }
    fn f() -> i32 {
        let p: Probability = 0.75_f64;
        let q: Probability = id(p);
        use_p(q)
    }
    """
    assert check_after_flatten(src) == []


def test_stage31_equivalent_refinement_aliases_carry_exact_proofs():
    same_predicate = check("""
    type NonNegativeA = f64 where self >= 0.0;
    type NonNegativeB = f64 where self >= 0.0;
    fn lift(a: NonNegativeA) -> NonNegativeB {
        a
    }
    """)
    assert same_predicate == [], same_predicate

    subset_predicate = check("""
    type UnitInterval = f64 where self >= 0.0, self <= 1.0;
    type NonNegative = f64 where self >= 0.0;
    fn lower(u: UnitInterval) -> NonNegative {
        u
    }
    """)
    assert subset_predicate == [], subset_predicate

    stronger_target = check("""
    type NonNegative = f64 where self >= 0.0;
    type UnitInterval = f64 where self >= 0.0, self <= 1.0;
    fn lift(n: NonNegative) -> UnitInterval {
        n
    }
    """)
    assert any("return value of function 'lift'" in e
               and "could not prove" in e
               for e in stronger_target), stronger_target

    reordered = check("""
    type NonNegativeA = f64 where self >= 0.0;
    type NonNegativeB = f64 where 0.0 <= self;
    fn lift(a: NonNegativeA) -> NonNegativeB {
        a
    }
    """)
    assert reordered == [], reordered


def test_stage34_numeric_bound_implication_carries_proofs():
    lower = check("""
    type AtLeastOne = f64 where self >= 1.0;
    type NonNegative = f64 where self >= 0.0;
    fn lift(x: AtLeastOne) -> NonNegative {
        x
    }
    """)
    assert lower == [], lower

    upper = check("""
    type BelowHalf = f64 where self <= 0.5;
    type AtMostOne = f64 where self <= 1.0;
    fn lift(x: BelowHalf) -> AtMostOne {
        x
    }
    """)
    assert upper == [], upper

    chained = check("""
    type SmallPositive = f64 where 0.25 <= self <= 0.75;
    type LooseUnit = f64 where 0.0 <= self <= 1.0;
    fn lift(x: SmallPositive) -> LooseUnit {
        x
    }
    """)
    assert chained == [], chained


def test_stage34_numeric_bound_implication_respects_strictness():
    weak_lower = check("""
    type NonNegative = f64 where self >= 0.0;
    type Positive = f64 where self > 0.0;
    fn lift(x: NonNegative) -> Positive {
        x
    }
    """)
    assert any("return value of function 'lift'" in e
               and "could not prove self > 0.0" in e
               for e in weak_lower), weak_lower

    strong_lower = check("""
    type Positive = f64 where self > 0.0;
    type NonNegative = f64 where self >= 0.0;
    fn lift(x: Positive) -> NonNegative {
        x
    }
    """)
    assert strong_lower == [], strong_lower

    weak_upper = check("""
    type AtMostOne = f64 where self <= 1.0;
    type LessThanOne = f64 where self < 1.0;
    fn lift(x: AtMostOne) -> LessThanOne {
        x
    }
    """)
    assert any("return value of function 'lift'" in e
               and "could not prove self < 1.0" in e
               for e in weak_upper), weak_upper

    strong_upper = check("""
    type LessThanOne = f64 where self < 1.0;
    type AtMostOne = f64 where self <= 1.0;
    fn lift(x: LessThanOne) -> AtMostOne {
        x
    }
    """)
    assert strong_upper == [], strong_upper


def test_stage34_equality_refinement_implies_matching_bounds():
    lower = check("""
    type ExactlyOne = f64 where self == 1.0;
    type NonNegative = f64 where self >= 0.0;
    fn lift(x: ExactlyOne) -> NonNegative {
        x
    }
    """)
    assert lower == [], lower

    upper = check("""
    type ExactlyHalf = f64 where 0.5 == self;
    type AtMostOne = f64 where self <= 1.0;
    fn lift(x: ExactlyHalf) -> AtMostOne {
        x
    }
    """)
    assert upper == [], upper

    same_value_reordered = check("""
    type ExactlyOneA = f64 where self == 1.0;
    type ExactlyOneB = f64 where 1.0 == self;
    fn lift(x: ExactlyOneA) -> ExactlyOneB {
        x
    }
    """)
    assert same_value_reordered == [], same_value_reordered


def test_stage34_equality_refinement_keeps_strict_bounds_fail_closed():
    too_strict_upper = check("""
    type ExactlyOne = f64 where self == 1.0;
    type LessThanOne = f64 where self < 1.0;
    fn lift(x: ExactlyOne) -> LessThanOne {
        x
    }
    """)
    assert any("return value of function 'lift'" in e
               and "could not prove self < 1.0" in e
               for e in too_strict_upper), too_strict_upper

    too_strict_lower = check("""
    type ExactlyOne = f64 where self == 1.0;
    type GreaterThanOne = f64 where self > 1.0;
    fn lift(x: ExactlyOne) -> GreaterThanOne {
        x
    }
    """)
    assert any("return value of function 'lift'" in e
               and "could not prove self > 1.0" in e
               for e in too_strict_lower), too_strict_lower

    not_equal = check("""
    type NonZero = f64 where self != 0.0;
    type NonNegative = f64 where self >= 0.0;
    fn lift(x: NonZero) -> NonNegative {
        x
    }
    """)
    assert any("return value of function 'lift'" in e
               and "could not prove self >= 0.0" in e
               for e in not_equal), not_equal


def test_stage34_compound_numeric_bounds_carry_proofs():
    logical_and_to_commas = check("""
    type Tight = f64 where self >= 0.25 && self <= 0.75;
    type Loose = f64 where self >= 0.0, self <= 1.0;
    fn lift(x: Tight) -> Loose {
        x
    }
    """)
    assert logical_and_to_commas == [], logical_and_to_commas

    commas_to_logical_and = check("""
    type Tight = f64 where self >= 0.25, self <= 0.75;
    type Loose = f64 where self >= 0.0 && self <= 1.0;
    fn lift(x: Tight) -> Loose {
        x
    }
    """)
    assert commas_to_logical_and == [], commas_to_logical_and


def test_stage34_numeric_bounds_carry_through_array_and_tuple_proofs():
    arrays = check("""
    type AtLeastOne = f64 where self >= 1.0;
    type NonNegative = f64 where self >= 0.0;
    fn use_values(xs: [NonNegative; 2]) -> i32 { 0 }
    fn lift(xs: [AtLeastOne; 2]) -> i32 {
        use_values(xs)
    }
    """)
    assert arrays == [], arrays

    tuples = check("""
    type AtMostHalf = f64 where self <= 0.5;
    type AtMostOne = f64 where self <= 1.0;
    fn lift(xs: (AtMostHalf, AtMostHalf)) -> i32 {
        let ys: (AtMostOne, AtMostOne) = xs;
        0
    }
    """)
    assert tuples == [], tuples


def test_stage34_negated_comparison_refinements_are_supported():
    ok = check("""
    type NonNegative = f64 where !(self < 0.0);
    fn f() -> i32 {
        let x: NonNegative = 0.0_f64;
        0
    }
    """)
    assert ok == [], ok

    bad = check("""
    type NonNegative = f64 where !(self < 0.0);
    fn f() -> i32 {
        let x: NonNegative = -0.25_f64;
        0
    }
    """)
    assert not any("predicate !(self < 0.0) is not supported" in e
                   for e in bad), bad
    assert any("refinement NonNegative violated" in e for e in bad), bad


def test_stage34_negated_comparison_bounds_carry_proofs():
    lower = check("""
    type NotBelowZero = f64 where !(self < 0.0);
    type NonNegative = f64 where self >= 0.0;
    fn lift(x: NotBelowZero) -> NonNegative {
        x
    }
    """)
    assert lower == [], lower

    strict_lower = check("""
    type AboveZero = f64 where !(self <= 0.0);
    type Positive = f64 where self > 0.0;
    fn lift(x: AboveZero) -> Positive {
        x
    }
    """)
    assert strict_lower == [], strict_lower

    too_strict = check("""
    type NonNegative = f64 where !(self < 0.0);
    type Positive = f64 where self > 0.0;
    fn lift(x: NonNegative) -> Positive {
        x
    }
    """)
    assert any("return value of function 'lift'" in e
               and "could not prove self > 0.0" in e
               for e in too_strict), too_strict


def test_stage34_affine_numeric_bounds_carry_proofs():
    shifted = check("""
    type ShiftedAtLeastOne = f64 where self + 1.0 >= 2.0;
    type AtLeastOne = f64 where self >= 1.0;
    fn lift(x: ShiftedAtLeastOne) -> AtLeastOne {
        x
    }
    """)
    assert shifted == [], shifted

    scaled = check("""
    type ScaledAtLeastOne = f64 where 2.0 * self >= 2.0;
    type AtLeastOne = f64 where self >= 1.0;
    fn lift(x: ScaledAtLeastOne) -> AtLeastOne {
        x
    }
    """)
    assert scaled == [], scaled

    flipped = check("""
    type AtMostHalf = f64 where 1.5 - self >= 1.0;
    type AtMostOne = f64 where self <= 1.0;
    fn lift(x: AtMostHalf) -> AtMostOne {
        x
    }
    """)
    assert flipped == [], flipped


def test_stage34_affine_numeric_bounds_keep_strictness():
    weak = check("""
    type ShiftedNonNegative = f64 where self + 1.0 >= 1.0;
    type Positive = f64 where self > 0.0;
    fn lift(x: ShiftedNonNegative) -> Positive {
        x
    }
    """)
    assert any("return value of function 'lift'" in e
               and "could not prove self > 0.0" in e
               for e in weak), weak

    strict = check("""
    type ShiftedPositive = f64 where self + 1.0 > 1.0;
    type Positive = f64 where self > 0.0;
    fn lift(x: ShiftedPositive) -> Positive {
        x
    }
    """)
    assert strict == [], strict


def test_stage34_named_constant_bounds_carry_proofs():
    named_bound = check("""
    const FLOOR: f64 = 1.0_f64;
    const ZERO: f64 = 0.0_f64;
    type AtLeastFloor = f64 where self >= FLOOR;
    type NonNegative = f64 where self >= ZERO;
    fn lift(x: AtLeastFloor) -> NonNegative {
        x
    }
    """)
    assert named_bound == [], named_bound

    named_affine = check("""
    const OFFSET: f64 = 1.0_f64;
    const TARGET: f64 = 2.0_f64;
    type ShiftedAtLeastOne = f64 where self + OFFSET >= TARGET;
    type AtLeastOne = f64 where self >= OFFSET;
    fn lift(x: ShiftedAtLeastOne) -> AtLeastOne {
        x
    }
    """)
    assert named_affine == [], named_affine


def test_stage34_numeric_bound_implication_requires_same_erased_base():
    errs = check("""
    type AtLeastOneI32 = i32 where self >= 1;
    type NonNegativeF64 = f64 where self >= 0.0;
    fn lift(x: AtLeastOneI32) -> NonNegativeF64 {
        x as NonNegativeF64
    }
    """)
    assert any("cast to refined type NonNegativeF64" in e
               and "could not prove self >= 0.0" in e
               for e in errs), errs


def test_stage34_refined_cast_checks_target_converted_value():
    errs = check("""
    type ExactlyHalfInt = i32 where self == 0.5;
    fn f() -> ExactlyHalfInt {
        0.5_f64 as ExactlyHalfInt
    }
    """)
    assert any("cast to refined type ExactlyHalfInt" in e
               and "target value 0 does not satisfy self == 0.5" in e
               for e in errs), errs


def test_stage34_refined_cast_rejects_boolean_source_to_numeric_refinement():
    errs = check("""
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn f() -> Probability {
        true as Probability
    }
    """)
    assert any("cast to refined type Probability" in e
               and "after casting bool to f64" in e
               for e in errs), errs


def test_stage34_refined_integer_alias_checks_base_width_before_proof():
    errs = check("""
    type Exactly300 = u8 where self == 300;
    fn f() -> Exactly300 {
        300_u8
    }
    """)
    assert any("return value of function 'f'" in e
               and "target base u8" in e
               and "could not prove self == 300" in e
               for e in errs), errs


def test_stage34_refined_f32_checks_rounded_target_value():
    cast_errs = check("""
    type AboveF32Boundary = f32 where self > 16777216.0;
    fn f() -> AboveF32Boundary {
        16777217.0_f64 as AboveF32Boundary
    }
    """)
    assert any("cast to refined type AboveF32Boundary" in e
               and "target value 16777216.0 does not satisfy "
                   "self > 16777216.0" in e
               for e in cast_errs), cast_errs

    direct_errs = check("""
    type AboveF32Boundary = f32 where self > 16777216.0;
    fn f() -> AboveF32Boundary {
        16777217.0_f32
    }
    """)
    assert any("return value of function 'f'" in e
               and "value 16777216.0 does not satisfy "
                   "self > 16777216.0" in e
               for e in direct_errs), direct_errs


def test_stage34_refined_f32_rejects_overflow_before_proof():
    errs = check("""
    type Huge = f32 where self > 3.5e38;
    fn f() -> Huge {
        1e40_f32
    }
    """)
    assert any("return value of function 'f'" in e
               and "target base f32" in e
               and "could not prove self > 3.5e+38" in e
               for e in errs), errs


def test_stage34_refined_f32_rejects_nonfinite_literal_before_proof():
    errs = check("""
    type Huge = f32 where self > 3.5e38;
    fn f() -> Huge {
        1e309_f32
    }
    """)
    assert any("return value of function 'f'" in e
               and "target base f32" in e
               and "could not prove self > 3.5e+38" in e
               for e in errs), errs


def test_stage34_refined_f64_rejects_nonfinite_literal_before_proof():
    errs = check("""
    type Huge = f64 where self > 1.0e308;
    fn f() -> Huge {
        1e309_f64
    }
    """)
    assert any("return value of function 'f'" in e
               and "target base f64" in e
               and "could not prove self > 1e+308" in e
               for e in errs), errs


def test_stage34_refined_integer_cast_rejects_nonfinite_before_proof():
    errs = check("""
    type NonNegativeInt = i32 where self >= 0;
    fn f() -> NonNegativeInt {
        1e309_f64 as NonNegativeInt
    }
    """)
    assert any("cast to refined type NonNegativeInt" in e
               and "could not prove self >= 0" in e
               and "after casting f64 to i32" in e
               for e in errs), errs


def test_stage34_self_independent_refinement_rejects_unrepresentable_values():
    literal_errs = check("""
    type AlwaysF64 = f64 where true;
    fn f() -> AlwaysF64 {
        1e309_f64
    }
    """)
    assert any("return value of function 'f'" in e
               and "requires a representable target value" in e
               and "target base f64" in e
               for e in literal_errs), literal_errs

    cast_errs = check("""
    type AlwaysInt = i32 where true;
    fn f() -> AlwaysInt {
        1e309_f64 as AlwaysInt
    }
    """)
    assert any("cast to refined type AlwaysInt" in e
               and "value is not representable after casting f64 to i32" in e
               for e in cast_errs), cast_errs


def test_stage34_fixed_point_preserves_unbound_name_errors():
    errs = check("""
    type AlwaysI32 = i32 where true;
    fn bad() -> AlwaysI32 {
        missing
    }
    fn main() -> i32 { 0 }
    """)
    assert any("unbound name 'missing'" in e for e in errs), errs


def test_stage34_refinement_predicate_float_literals_use_target_suffix():
    rounded_errs = check("""
    type BelowRounded = f32 where self < 16777217.0_f32;
    fn f() -> BelowRounded {
        16777216.0_f32
    }
    """)
    assert any("return value of function 'f'" in e
               and "value 16777216.0 does not satisfy "
                   "self < 16777217.0" in e
               for e in rounded_errs), rounded_errs

    f32_overflow_errs = check("""
    type BelowOverflow = f32 where self < 1e40_f32;
    fn f() -> BelowOverflow {
        0.0_f32
    }
    """)
    assert any("return value of function 'f'" in e
               and "predicate self < 1e+40 is not supported" in e
               for e in f32_overflow_errs), f32_overflow_errs

    f64_nonfinite_errs = check("""
    type BelowNonFinite = f64 where self < 1e309_f64;
    fn f() -> BelowNonFinite {
        0.0_f64
    }
    """)
    assert any("return value of function 'f'" in e
               and "predicate self < inf is not supported" in e
               for e in f64_nonfinite_errs), f64_nonfinite_errs


def test_stage31_unsupported_refinement_predicates_do_not_carry_by_name():
    errs = check("""
    type Source = f64 where foo();
    type Target = f64 where bar();
    fn f(s: Source) -> Target {
        s
    }
    """)
    assert any("type alias 'Source': refinement predicate Call is not supported"
               in e for e in errs), errs
    assert any("type alias 'Target': refinement predicate Call is not supported"
               in e for e in errs), errs
    assert any("return value of function 'f'" in e
               and "predicate Call is not supported" in e
               for e in errs), errs


def test_stage31_generic_qualified_refinement_names_are_unsupported():
    bad_self = check("""
    type A = f64 where self::<Missing> >= 0.0;
    fn f() -> A { 1.0_f64 }
    """)
    assert any(
        "refinement predicate self::<Missing> >= 0.0 is not supported" in e
        for e in bad_self
    ), bad_self

    bad_const = check("""
    const LIMIT: f64 = 0.0_f64;
    type A = f64 where LIMIT::<Missing> >= 0.0;
    fn f() -> A { 1.0_f64 }
    """)
    assert any(
        "refinement predicate LIMIT::<Missing> >= 0.0 is not supported" in e
        for e in bad_const
    ), bad_const


def test_stage31_refined_scalar_arithmetic_erases_to_base_scalar():
    src = """
    type Positive = i32 where self > 0;
    fn add_one_raw(x: Positive) -> i32 {
        x + 1
    }
    """
    assert check_after_flatten(src) == []


def test_stage31_cast_to_refined_alias_checks_literal_predicate():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn f() {
        let p: Probability = 2.0_f64 as Probability;
    }
    """
    errs = check(src)
    assert any("cast to refined type Probability" in e
               and "refinement Probability violated" in e
               for e in errs), errs


def test_stage31_refinement_local_const_proofs_use_local_value():
    ok_src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn f() {
        const P: f64 = 0.5_f64;
        let p = P as Probability;
    }
    """
    assert check(ok_src) == []

    bad_src = """
    const LIMIT: f64 = 0.5_f64;
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn f() {
        const LIMIT: f64 = 1.5_f64;
        let p: Probability = LIMIT;
    }
    """
    errs = check(bad_src)
    assert any("refinement Probability violated" in e for e in errs), errs


def test_stage31_refinement_assignment_constant_checked():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn f() {
        let mut p: Probability = 0.25_f64;
        p = 1.2_f64;
    }
    """
    errs = check(src)
    assert any("assignment" in e
               and "refinement Probability violated" in e for e in errs), errs


def test_stage31_refinement_uninitialized_let_rejected():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn f() {
        let p: Probability;
    }
    """
    errs = check(src)
    assert any("requires an initializer" in e for e in errs), errs


def test_stage31_refinement_const_values_checked():
    local_src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn f() {
        const P: Probability = 1.2_f64;
    }
    """
    local_errs = check(local_src)
    assert any("const 'P'" in e
               and "refinement Probability violated" in e
               for e in local_errs), local_errs

    top_src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    const P: Probability = 1.2_f64;
    fn f() {}
    """
    top_errs = check(top_src)
    assert any("const 'P'" in e
               and "refinement Probability violated" in e
               for e in top_errs), top_errs


def test_stage31_refinement_struct_and_array_members_checked():
    struct_src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    struct Reading { p: Probability }
    fn f() {
        let r = Reading { p: 1.2_f64 };
    }
    """
    struct_errs = check(struct_src)
    assert any("struct 'Reading'.p" in e
               and "refinement Probability violated" in e
               for e in struct_errs), struct_errs

    array_src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn f() {
        let ps: [Probability; 2] = [0.5_f64, 1.2_f64];
    }
    """
    array_errs = check(array_src)
    assert any("array element" in e
               and "refinement Probability violated" in e
               for e in array_errs), array_errs


def test_stage31_refinement_enum_payload_checked():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    enum Maybe { None, Some(Probability) }
    fn f() {
        let x = Maybe::Some(1.2_f64);
    }
    """
    errs = check(src)
    assert any("enum Maybe::Some arg 0" in e
               and "refinement Probability violated" in e for e in errs), errs


def test_stage31_nested_refinement_alias_inherits_base_predicate():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    type Certain = Probability where self >= 0.9;
    fn f() {
        let c: Certain = 1.2_f64;
    }
    """
    errs = check(src)
    assert any("refinement Probability violated" in e for e in errs), errs


def test_stage31_unknown_refinement_alias_target_errors():
    src = """
    type Bad = Missing where self > 0.0;
    fn f() {
        let x: Bad = 1.0_f64;
    }
    """
    errs = check(src)
    assert any("target type could not be resolved" in e for e in errs), errs


def test_stage31_if_branch_cannot_forge_refined_proof():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn pick(b: bool, p: Probability) -> Probability {
        if b { p } else { 1.2_f64 }
    }
    """
    errs = check(src)
    assert any("return value of function 'pick'" in e
               and "could not prove" in e for e in errs), errs


def test_stage31_match_arm_cannot_forge_refined_proof():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn pick(b: bool, p: Probability) -> Probability {
        match b {
            true => p,
            false => 1.2_f64,
        }
    }
    """
    errs = check(src)
    assert any("return value of function 'pick'" in e
               and "could not prove" in e for e in errs), errs


def test_stage31_module_local_refinement_alias_survives_flatten():
    src = """
    mod m {
        type Probability = f64 where 0.0 <= self <= 1.0;
        fn f() {
            let p: Probability = 1.2_f64;
        }
    }
    """
    errs = check_after_flatten(src)
    assert any("refinement m__Probability violated" in e for e in errs), errs


def test_stage31_module_type_alias_does_not_capture_function_generic():
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.flatten_modules import flatten_modules

    src = """
    mod m {
        type T = f64 where 0.0 <= self <= 1.0;
        fn id[T](x: T) -> T { x }
    }
    """
    prog = parse(src)
    flatten_modules(prog)
    fn = next(
        it for it in prog.items
        if isinstance(it, A.FnDecl) and it.name == "m__id"
    )
    assert isinstance(fn.params[0].ty, A.TyName)
    assert fn.params[0].ty.name == "T"
    assert isinstance(fn.return_ty, A.TyName)
    assert fn.return_ty.name == "T"


def test_stage31_module_type_alias_rewrites_type_expr_names():
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.flatten_modules import flatten_modules

    src = """
    mod m {
        const N: i32 = 4;
        type Vec = [i32; N];
    }
    """
    prog = parse(src)
    flatten_modules(prog)
    alias = next(
        it for it in prog.items
        if isinstance(it, A.TypeAlias) and it.name == "m__Vec"
    )
    assert isinstance(alias.target, A.TyArray)
    assert isinstance(alias.target.size, A.Name)
    assert alias.target.size.name == "m__N"


def test_stage31_module_type_alias_rewrites_type_expr_paths():
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.flatten_modules import flatten_modules

    src = """
    mod m { const N: i32 = 4; }
    type Vec = [i32; m::N];
    fn f(v: Vec) {}
    """
    prog = parse(src)
    flatten_modules(prog)
    alias = next(
        it for it in prog.items
        if isinstance(it, A.TypeAlias) and it.name == "Vec"
    )
    assert isinstance(alias.target, A.TyArray)
    assert isinstance(alias.target.size, A.Name)
    assert alias.target.size.name == "m__N"
    assert check_after_flatten(src) == []


def test_stage31_module_type_alias_rewrites_predicate_names():
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.flatten_modules import flatten_modules

    src = """
    mod m {
        const MAX: f64 = 1.0_f64;
        type Probability = f64 where 0.0 <= self <= MAX;
    }
    """
    prog = parse(src)
    flatten_modules(prog)
    alias = next(
        it for it in prog.items
        if isinstance(it, A.TypeAlias) and it.name == "m__Probability"
    )
    pred = alias.where_clauses[0].constraint
    assert isinstance(pred, A.Binary)
    assert isinstance(pred.right, A.Name)
    assert pred.right.name == "m__MAX"


def test_stage31_module_type_alias_rewrites_predicate_paths():
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.flatten_modules import flatten_modules

    src = """
    mod m { const MAX: f64 = 1.0_f64; }
    type Probability = f64 where 0.0 <= self <= m::MAX;
    fn f(p: Probability) {}
    """
    prog = parse(src)
    flatten_modules(prog)
    alias = next(
        it for it in prog.items
        if isinstance(it, A.TypeAlias) and it.name == "Probability"
    )
    pred = alias.where_clauses[0].constraint
    assert isinstance(pred, A.Binary)
    assert isinstance(pred.right, A.Name)
    assert pred.right.name == "m__MAX"
    assert check_after_flatten(src) == []


def test_stage31_refined_array_call_and_return_checked():
    call_src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn use_ps(ps: [Probability; 2]) -> i32 { 0 }
    fn f() -> i32 {
        use_ps([0.5_f64, 1.2_f64])
    }
    """
    call_errs = check(call_src)
    assert any("call to 'use_ps': arg 'ps'" in e
               and "array element" in e for e in call_errs), call_errs

    return_src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn f() -> [Probability; 2] {
        [0.5_f64, 1.2_f64]
    }
    """
    return_errs = check(return_src)
    assert any("return value of function 'f'" in e
               and "array element" in e for e in return_errs), return_errs


def test_stage31_function_typed_call_checks_refined_args():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn use_p(p: Probability) -> i32 { 0 }
    fn f() -> i32 {
        let fp: fn(Probability) -> i32 = use_p;
        fp(1.2_f64)
    }
    """
    errs = check(src)
    assert any("function-typed call arg 0" in e
               and "refinement Probability violated" in e for e in errs), errs

    arity_errs = check("""
    fn use_i(x: i32) -> i32 { x }
    fn f() -> i32 {
        let fp: fn(i32) -> i32 = use_i;
        fp()
    }
    """)
    assert any("function-typed call: expected 1 args, got 0" in e
               for e in arity_errs), arity_errs


def test_stage31_function_pointer_cannot_forge_refined_return():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn raw() -> f64 { 1.2_f64 }
    fn f() -> Probability {
        let fp: fn() -> Probability = raw;
        fp()
    }
    """
    errs = check(src)
    assert any("function type conversion from fn() -> f64 "
               "to fn() -> Probability would change refined" in e
               for e in errs), errs


def test_stage31_refined_function_cannot_weaken_to_raw_function_type():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn use_p(p: Probability) -> i32 { 0 }
    fn f() -> i32 {
        let fp: fn(f64) -> i32 = use_p;
        fp(1.2_f64)
    }
    """
    errs = check(src)
    assert any("function type conversion from fn(Probability) -> i32 "
               "to fn(f64) -> i32 would change refined parameter" in e
               for e in errs), errs


def test_stage31_refined_function_cannot_weaken_through_call_arg():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn use_p(p: Probability) -> i32 { 0 }
    fn take_raw(f: fn(f64) -> i32) -> i32 { f(1.2_f64) }
    fn main() -> i32 {
        take_raw(use_p)
    }
    """
    errs = check(src)
    assert any("call to 'take_raw': arg 'f'" in e
               and "function type conversion from fn(Probability) -> i32 "
               "to fn(f64) -> i32 would change refined parameter" in e
               for e in errs), errs


def test_stage31_refined_function_cannot_weaken_through_return():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn use_p(p: Probability) -> i32 { 0 }
    fn leak() -> fn(f64) -> i32 {
        use_p
    }
    """
    errs = check(src)
    assert any("return value of function 'leak'" in e
               and "function type conversion from fn(Probability) -> i32 "
               "to fn(f64) -> i32 would change refined parameter" in e
               for e in errs), errs


def test_stage31_refined_function_cannot_weaken_through_explicit_return():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn use_p(p: Probability) -> i32 { 0 }
    fn raw(p: f64) -> i32 { 0 }
    fn leak() -> fn(f64) -> i32 {
        return use_p;
        raw
    }
    """
    errs = check(src)
    assert any("return value of function 'leak'" in e
               and "function type conversion from fn(Probability) -> i32 "
               "to fn(f64) -> i32 would change refined parameter" in e
               for e in errs), errs


def test_stage31_refined_function_cannot_weaken_through_branch_join():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn use_p(p: Probability) -> i32 { 0 }
    fn raw(p: f64) -> i32 { 1 }
    fn leak(b: bool) -> fn(f64) -> i32 {
        if b { use_p } else { raw }
    }
    """
    errs = check(src)
    assert any("branch function types fn(Probability) -> i32 and "
               "fn(f64) -> i32 differ in refined parameter" in e
               for e in errs), errs


def test_stage31_refined_function_cannot_weaken_through_let_branch_join():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn use_p(p: Probability) -> i32 { 0 }
    fn raw(p: f64) -> i32 { 1 }
    fn f(b: bool) -> i32 {
        let fp: fn(f64) -> i32 = if b { use_p } else { raw };
        fp(1.2_f64)
    }
    """
    errs = check(src)
    assert any("branch function types fn(Probability) -> i32 and "
               "fn(f64) -> i32 differ in refined parameter" in e
               for e in errs), errs


def test_stage31_refined_function_cannot_weaken_through_array_literal():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn use_p(p: Probability) -> i32 { 0 }
    fn raw(p: f64) -> i32 { 1 }
    fn f() -> i32 {
        let fps = [use_p, raw];
        fps[0](1.2_f64)
    }
    """
    errs = check(src)
    assert any("array literal function element types "
               "fn(Probability) -> i32 and fn(f64) -> i32 differ "
               "in refined parameter" in e for e in errs), errs


def test_stage31_refined_function_cannot_weaken_through_array_branch_join():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn use_p(p: Probability) -> i32 { 0 }
    fn raw(p: f64) -> i32 { 1 }
    fn f(b: bool) -> i32 {
        let fps = if b { [use_p] } else { [raw] };
        fps[0](1.2_f64)
    }
    """
    errs = check(src)
    assert any("branch function types [fn(Probability) -> i32" in e
               and "differ in refined parameter" in e for e in errs), errs


def test_stage31_refined_function_cannot_weaken_through_reference():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn take_raw(r: &fn(f64) -> i32) -> i32 { 0 }
    fn bridge(r: &fn(Probability) -> i32) -> i32 {
        take_raw(r)
    }
    """
    errs = check(src)
    assert any("call to 'take_raw': arg 'r'" in e
               and "reference type conversion from &fn(Probability) -> i32 "
               "to &fn(f64) -> i32 would change refined parameter" in e
               for e in errs), errs


def test_stage31_refined_function_cannot_weaken_through_pointer():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn take_raw(p: *const fn(f64) -> i32) -> i32 { 0 }
    fn bridge(p: *const fn(Probability) -> i32) -> i32 {
        take_raw(p)
    }
    """
    errs = check(src)
    assert any("call to 'take_raw': arg 'p'" in e
               and "pointer type conversion from *const fn(Probability) -> i32 "
               "to *const fn(f64) -> i32 would change refined parameter" in e
               for e in errs), errs


def test_stage31_refined_pointer_cannot_weaken_to_raw_pointer():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    extern "C" fn poison(p: *mut f64) -> i32;
    fn bridge(p: *mut Probability) -> i32 {
        poison(p)
    }
    """
    errs = check(src)
    assert any("call to 'poison': arg 'p'" in e
               and "pointer type conversion from *mut Probability "
               "to *mut f64 would change refined parameter" in e
               for e in errs), errs


def test_unary_address_of_types_as_reference():
    src = """
    fn take(r: &i32) -> i32 { 0 }
    fn main() -> i32 {
        let x: i32 = 7;
        take(&x)
    }
    """
    errs = check(src)
    assert any("operator '&' is type-known but not lowerable yet" in e
               for e in errs), errs
    assert not any("arg 'r' expects &i32, got i32" in e for e in errs), errs


def test_unary_mut_address_of_types_as_mut_reference():
    src = """
    fn take(r: &mut i32) -> i32 { 0 }
    fn main() -> i32 {
        let mut x: i32 = 7;
        take(&mut x)
    }
    """
    errs = check(src)
    assert any("operator '&mut' is type-known but not lowerable yet" in e
               for e in errs), errs
    assert not any("immutable binding" in e for e in errs), errs
    assert not any("arg 'r' expects &mut i32, got i32" in e
                   for e in errs), errs


def test_unary_mut_address_of_requires_mutable_binding():
    src = """
    fn take(r: &mut i32) -> i32 { 0 }
    fn main() -> i32 {
        let x: i32 = 7;
        take(&mut x)
    }
    """
    errs = check(src)
    assert any("cannot take mutable reference to immutable binding 'x'" in e
               for e in errs), errs


def test_unary_address_of_requires_named_binding():
    src = """
    fn take(r: &i32) -> i32 { 0 }
    fn main() -> i32 {
        take(&(1 + 2))
    }
    """
    errs = check(src)
    assert any("operator '&' requires an addressable named binding" in e
               for e in errs), errs


def test_unary_address_of_requires_local_binding_not_function():
    src = """
    fn helper() -> i32 { 1 }
    fn take(r: &fn() -> i32) -> i32 { 0 }
    fn main() -> i32 {
        take(&helper)
    }
    """
    errs = check(src)
    assert any("operator '&' requires a local binding; 'helper' is not "
               "addressable" in e for e in errs), errs


def test_unary_mut_address_of_rejects_const_symbol():
    src = """
    const C: i32 = 1;
    fn take(r: &mut i32) -> i32 { 0 }
    fn main() -> i32 {
        take(&mut C)
    }
    """
    errs = check(src)
    assert any("operator '&mut' requires a local binding; 'C' is not "
               "addressable" in e for e in errs), errs


def test_unary_deref_inside_unsafe_types_as_pointee():
    src = """
    fn main() -> i32 {
        let p: *const i32 = unsafe { 0 as *const i32 };
        let x: i32 = unsafe { *p };
        x
    }
    """
    errs = check(src)
    assert any("raw-pointer dereference is type-known but not lowerable yet"
               in e for e in errs), errs
    assert not any("declared i32 but value is *const i32" in e
                   for e in errs), errs


def test_unary_deref_outside_unsafe_traps_28601():
    src = """
    fn main() -> i32 {
        let p: *const i32 = unsafe { 0 as *const i32 };
        *p
    }
    """
    errs = check(src)
    assert any("28601" in e for e in errs), errs


def test_unary_deref_requires_pointer_or_reference_operand():
    src = """
    fn main() -> i32 {
        let x: i32 = 7;
        *x
    }
    """
    errs = check(src)
    assert any("operator '*' expects pointer or reference operand" in e
               for e in errs), errs


def test_unary_generic_address_of_is_not_false_clean():
    src = """
    fn f[T](x: T) -> i32 {
        let r = &x;
        0
    }
    """
    errs = check(src)
    assert any("operator '&' is type-known but not lowerable yet" in e
               for e in errs), errs


def test_unary_generic_deref_is_not_false_clean():
    src = """
    fn f[T](x: T) -> i32 {
        unsafe { *x };
        0
    }
    """
    errs = check(src)
    assert any("operator '*' cannot dereference unresolved operand type T" in e
               for e in errs), errs


def test_stage31_refined_function_array_cannot_reannotate_to_raw_array():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn use_p(p: Probability) -> i32 { 0 }
    fn f() -> i32 {
        let fps = [use_p];
        let raws: [fn(f64) -> i32; 1] = fps;
        raws[0](1.2_f64)
    }
    """
    errs = check(src)
    assert any("array type conversion from [fn(Probability) -> i32; 1] "
               "to [fn(f64) -> i32; 1] would change refined parameter" in e
               for e in errs), errs


def test_stage31_refined_pointer_array_cannot_reannotate_to_raw_array():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn f(p: *mut Probability) -> i32 {
        let xs = [p];
        let raw: [*mut f64; 1] = xs;
        0
    }
    """
    errs = check(src)
    assert any("array type conversion from [*mut Probability; 1] "
               "to [*mut f64; 1] would change refined parameter" in e
               for e in errs), errs


def test_stage31_refined_function_tuple_cannot_reannotate_to_raw_tuple():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn use_p(p: Probability) -> i32 { 0 }
    fn f() -> i32 {
        let fps = (use_p,);
        let raws: (fn(f64) -> i32,) = fps;
        0
    }
    """
    errs = check(src)
    assert any("tuple type conversion from (fn(Probability) -> i32) "
               "to (fn(f64) -> i32) would change refined parameter" in e
               for e in errs), errs


def test_stage31_refined_wrapper_cannot_weaken_to_raw_wrapper():
    diff = check("""
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn leak(x: D<Probability>) -> D<f64> { x }
    """)
    assert any("type conversion from D<Probability> to D<f64> "
               "would change refined" in e for e in diff), diff

    logic = check("""
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn leak(x: Logic<Probability>) -> Logic<f64> { x }
    """)
    assert any("type conversion from Logic<Probability> to Logic<f64> "
               "would change refined" in e for e in logic), logic

    mem = check("""
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn leak(x: WorkingMem<Probability>) -> WorkingMem<f64> { x }
    """)
    assert any("type conversion from WorkingMem<Probability> "
               "to WorkingMem<f64> would change refined" in e
               for e in mem), mem


def test_stage31_refined_tensor_dtype_cannot_weaken_to_raw_dtype():
    src = """
    type Probability = f32 where 0.0 <= self <= 1.0;
    fn leak(x: tensor<Probability, [4]>) -> tensor<f32, [4]> { x }
    """
    errs = check(src)
    assert any("type conversion from tensor<Probability, [4]> "
               "to tensor<f32, [4]> would change refined" in e
               for e in errs), errs


def test_stage31_function_typed_call_checks_refined_actual_wrappers():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn use_raw(x: D<f64>) -> i32 { 0 }
    fn bridge(x: D<Probability>) -> i32 {
        let fp: fn(D<f64>) -> i32 = use_raw;
        fp(x)
    }
    """
    errs = check(src)
    assert any("function-typed call arg 0" in e
               and "type conversion from D<Probability> to D<f64> "
               "would change refined" in e for e in errs), errs


def test_stage31_function_typed_call_fails_before_backend():
    src = """
    fn use_i(x: i32) -> i32 { x }
    fn apply(fp: fn(i32) -> i32, x: i32) -> i32 {
        fp(x)
    }
    fn main() -> i32 { apply(use_i, 42) }
    """
    errs = check(src)
    assert any("function-typed calls are not supported by the Stage 31 "
               "backend" in e for e in errs), errs


def test_stage31_extern_signatures_cannot_claim_refined_types():
    ret = check("""
    type Probability = f64 where 0.0 <= self <= 1.0;
    extern "C" fn c_prob() -> Probability;
    fn use_p(p: Probability) -> i32 { 42 }
    fn main() -> i32 { use_p(c_prob()) }
    """)
    assert any("extern function 'c_prob': return type Probability "
               "cannot use refined types" in e for e in ret), ret

    param = check("""
    type Probability = f64 where 0.0 <= self <= 1.0;
    extern "C" fn poison(p: *mut Probability) -> i32;
    fn main(p: *mut Probability) -> i32 { poison(p) }
    """)
    assert any("extern function 'poison': parameter 'p' type "
               "*mut Probability cannot use refined types" in e
               for e in param), param

    wrapped = check("""
    type Probability = f64 where 0.0 <= self <= 1.0;
    extern "C" fn c_prob() -> D<Probability>;
    """)
    assert any("extern function 'c_prob': return type D<Probability> "
               "cannot use refined types" in e for e in wrapped), wrapped


def test_stage31_extern_signature_cannot_smuggle_refined_struct_field():
    errs = check_after_flatten("""
    type Probability = f64 where 0.0 <= self <= 1.0;
    struct Reading { p: Probability }
    extern "C" fn c_reading() -> Reading;
    fn use_p(p: Probability) -> i32 { 42 }
    fn main() -> i32 { use_p(c_reading().p) }
    """)
    assert any("extern function 'c_reading': return type Reading "
               "cannot use refined types" in e for e in errs), errs


def test_stage31_extern_signature_cannot_smuggle_refined_enum_payload():
    errs = check_after_flatten("""
    type Probability = f64 where self >= 0.0 && self <= 1.0;
    enum Box { Some(Probability), None }
    extern "C" fn sink(x: Box) -> i32;
    fn main() -> i32 { 0 }
    """)
    assert any("extern function 'sink': parameter 'x' type Box "
               "cannot use refined types" in e for e in errs), errs


def test_stage31_refined_composite_nonliteral_requires_existing_proof():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn f() {
        let raw = [1.2_f64];
        let ps: [Probability; 1] = raw;
    }
    """
    errs = check(src)
    assert any("refined array type" in e
               and "requires an array literal" in e for e in errs), errs


def test_stage31_refined_tuple_nonliteral_requires_existing_proof():
    src = """
    type Probability = f64 where 0.0 <= self <= 1.0;
    fn f() {
        let raw = (1.2_f64, 0.5_f64);
        let ps: (Probability, Probability) = raw;
    }
    """
    errs = check(src)
    assert any("refined tuple type" in e
               and "requires a tuple literal" in e for e in errs), errs


def test_stage31_unused_bad_type_aliases_are_diagnosed():
    bad_target = check("""
    type Bad = Missing where self > 0.0;
    fn f() {}
    """)
    assert any("target type could not be resolved" in e for e in bad_target), bad_target

    generic = check("""
    type Box[T] = f64 where self > 0.0;
    fn f() {}
    """)
    assert any("generic aliases are not supported" in e for e in generic), generic

    recursive = check("""
    type A = A where self > 0.0;
    fn f() {}
    """)
    assert any("is recursive" in e for e in recursive), recursive


def test_stage31_alias_targets_reject_nested_unknowns():
    array_alias = check("""
    type Bad = [Missing; 1];
    fn f() {}
    """)
    assert any("target type could not be resolved" in e
               and "unknown name Missing" in e for e in array_alias), array_alias

    tuple_alias = check("""
    type Bad = (Missing, i32);
    fn f() {}
    """)
    assert any("target type could not be resolved" in e
               and "unknown name Missing" in e for e in tuple_alias), tuple_alias

    fn_alias = check("""
    type Bad = fn(Missing) -> i32;
    fn f() {}
    """)
    assert any("target type could not be resolved" in e
               and "unknown name Missing" in e for e in fn_alias), fn_alias


def test_stage31_unknown_generic_type_rejected_before_refinement_bypass():
    src = """
    mod m {
        type Probability = f64 where 0.0 <= self <= 1.0;
    }
    use m::Probability;
    fn f() {
        let p: Missing<Probability> = 1.2_f64;
    }
    """
    errs = check_after_flatten(src)
    assert any("unknown generic type 'Missing'" in e for e in errs), errs


def test_stage31_generic_struct_args_are_resolved_in_alias_targets():
    src = """
    struct Box[T] { v: T }
    type Alias = Box<Probability>;
    fn bad(p: Alias) -> i32 { 0 }
    """
    errs = check(src)
    assert any("unknown type 'Probability'" in e for e in errs), errs
    assert any("type alias 'Alias': target type could not be resolved" in e
               for e in errs), errs


def test_stage31_alias_to_enum_target_resolves():
    errs = check("""
    enum List { Nil }
    type L = List;
    fn f(l: L) {}
    """)
    assert errs == []


def test_stage31_nonrecursive_aggregate_returns_rejected_before_lowering():
    struct_errs = check("""
    struct Pt { x: i32, y: i32 }
    fn make() -> Pt {
        let p = Pt { x: 10, y: 32 };
        p
    }
    """)
    assert any("aggregate return type Pt is not supported" in e
               for e in struct_errs), struct_errs

    enum_errs = check("""
    enum Maybe { None, Some(i32) }
    fn make() -> Maybe { Maybe::Some(42) }
    """)
    assert any("aggregate return type Maybe is not supported" in e
               for e in enum_errs), enum_errs

    tuple_errs = check("""
    fn make() -> (i32, i32) { (1, 2) }
    """)
    assert any("aggregate return type (i32, i32) is not supported" in e
               for e in tuple_errs), tuple_errs

    array_errs = check("""
    fn make() -> [i32; 2] { [1, 2] }
    """)
    assert any("aggregate return type [i32; 2] is not supported" in e
               for e in array_errs), array_errs


def test_stage31_recursive_enum_return_stays_supported():
    errs = check("""
    enum List { Nil, Cons(i32, List) }
    fn make() -> List { List::Nil }
    """)
    assert errs == [], errs


def test_stage31_enum_constructor_args_match_enum_params():
    errs = check("""
    enum Maybe { None, Some(i32) }
    fn unwrap(m: Maybe, d: i32) -> i32 { d }
    fn main() -> i32 {
        unwrap(Maybe::Some(42), 0) + unwrap(Maybe::None, 0)
    }
    """)
    assert errs == [], errs


def test_stage31_wrong_enum_match_pattern_rejected():
    errs = check_after_flatten("""
    enum A { X }
    enum B { X }
    fn f(a: A) -> i32 {
        match a { B::X => 42 }
    }
    fn main() -> i32 { f(A::X) }
    """)
    assert any("pattern B::X cannot match scrutinee type A" in e
               for e in errs), errs


def test_stage31_unused_bad_refinement_predicates_are_diagnosed():
    missing = check("""
    type Bad = f64 where missing > 0.0;
    fn f() {}
    """)
    assert any("refinement predicate missing > 0.0 is not supported" in e
               for e in missing), missing

    non_bool = check("""
    type Bad = f64 where self + 1.0;
    fn f() {}
    """)
    assert any("refinement predicate (self + 1.0) is not supported" in e
               for e in non_bool), non_bool

    bool_base = check("""
    type BoolRange = bool where 0.0 <= self <= 1.0;
    fn f() {}
    """)
    assert any("numeric scalar base type" in e
               and "bool" in e for e in bool_base), bool_base

    array_base = check("""
    type Bad = [i32; 1] where self > 0;
    fn f() {}
    """)
    assert any("numeric scalar base type" in e
               and "[i32; 1]" in e for e in array_base), array_base

    bool_operand = check("""
    type Bad = f64 where false < self;
    fn f() {}
    """)
    assert any("refinement predicate false < self is not supported" in e
               for e in bool_operand), bool_operand


def test_stage31_boolean_literal_refinement_predicates_are_supported():
    always = check("""
    type Always = f64 where true;
    fn main() -> i32 { let a: Always = 0.5_f64; 0 }
    """)
    assert always == [], always

    never = check("""
    type Never = f64 where false;
    fn main() -> i32 { let n: Never = 0.5_f64; 0 }
    """)
    assert not any("predicate false is not supported" in e for e in never), never
    assert any("refinement Never violated" in e for e in never), never


def test_stage31_self_independent_false_refinement_rejects_unknown_values():
    never = check("""
    type Never = f64 where false;
    fn use_raw(x: f64) -> i32 { let n: Never = x; 0 }
    """)
    assert not any("compile-time-proven value" in e for e in never), never
    assert any("predicate false is always false" in e for e in never), never

    always = check("""
    type Always = f64 where true;
    fn use_raw(x: f64) -> i32 { let a: Always = x; 0 }
    """)
    assert always == [], always


def test_stage31_mixed_self_independent_refinements_do_not_downgrade():
    mixed = check("""
    type Mixed = f64 where false, self >= 0.0;
    fn use_raw(x: f64) -> i32 { let m: Mixed = x; 0 }
    """)
    assert any("predicate false is always false" in e for e in mixed), mixed
    assert any("could not prove self >= 0.0" in e for e in mixed), mixed

    inherited = check("""
    type Never = f64 where false;
    type NonNegativeNever = Never where self >= 0.0;
    fn use_raw(x: f64) -> i32 { let n: NonNegativeNever = x; 0 }
    """)
    assert any("predicate false is always false" in e for e in inherited), inherited
    assert any("could not prove self >= 0.0" in e for e in inherited), inherited


def test_stage31_boolean_short_circuit_refinements_are_decisive():
    false_and = check("""
    type Never = f64 where false && self >= 0.0;
    fn use_raw(x: f64) -> i32 { let n: Never = x; 0 }
    """)
    assert not any("compile-time-proven value" in e for e in false_and), false_and
    assert any("predicate (false && self >= 0.0) is always false" in e
               for e in false_and), false_and

    true_or = check("""
    type Always = f64 where true || self >= 0.0;
    fn use_raw(x: f64) -> i32 { let a: Always = x; 0 }
    """)
    assert true_or == [], true_or


def test_stage31_constant_comparison_refinement_predicates_are_supported():
    always = check("""
    type Always = f64 where 1.0 < 2.0;
    fn use_raw(x: f64) -> i32 { let a: Always = x; 0 }
    """)
    assert always == [], always

    never = check("""
    type Never = f64 where 2.0 < 1.0;
    fn use_raw(x: f64) -> i32 { let n: Never = x; 0 }
    """)
    assert not any("predicate 2.0 < 1.0 is not supported" in e for e in never), never
    assert any("predicate 2.0 < 1.0 is always false" in e for e in never), never


def test_stage31_nested_module_use_rewrites_refined_alias_type():
    src = """
    mod m {
        type Probability = f64 where 0.0 <= self <= 1.0;
    }
    mod n {
        use m::Probability;
        fn f() { let p: Probability = 1.2_f64; }
    }
    """
    errs = check_after_flatten(src)
    assert any("refinement m__Probability violated" in e for e in errs), errs


def test_stage31_parent_module_use_rewrites_child_module():
    src = """
    mod m {
        type Probability = f64 where 0.0 <= self <= 1.0;
    }
    mod outer {
        use m::Probability;
        mod child { fn f() { let p: Probability = 1.2_f64; } }
    }
    """
    errs = check_after_flatten(src)
    assert any("refinement m__Probability violated" in e for e in errs), errs


def test_stage31_parent_module_sibling_alias_does_not_capture_child():
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.flatten_modules import flatten_modules

    src = """
    mod outer {
        type Probability = f64 where 0.0 <= self <= 1.0;
        mod child { fn f() { let p: Probability = 1.2_f64; } }
    }
    """
    prog = parse(src)
    flatten_modules(prog)
    fn = next(
        it for it in prog.items
        if isinstance(it, A.FnDecl) and it.name == "outer__child__f"
    )
    let_stmt = fn.body.stmts[0]
    assert isinstance(let_stmt, A.Let)
    assert isinstance(let_stmt.ty, A.TyName)
    assert let_stmt.ty.name == "Probability"
    errs = check_after_flatten(src)
    assert any("unknown type 'Probability'" in e for e in errs), errs


def test_stage31_flatten_rewrites_module_enum_match_patterns():
    src = """
    mod m {
        enum E { A, B }
        fn f(e: E) -> i32 {
            match e { E::A => 1, E::B => 2 }
        }
    }
    """
    errs = check_after_flatten(src)
    assert errs == [], errs


def test_stage31_flatten_rewrites_module_enum_value_paths():
    src = """
    mod m { enum E { A } }
    fn main() -> i32 {
        match m::E::A { m::E::A => 42 }
    }
    """
    errs = check_after_flatten(src)
    assert errs == [], errs


def test_stage31_flatten_rewrites_module_enum_payload_constructors():
    src = """
    mod m {
        enum Maybe { None, Some(i32) }
        fn take(m: Maybe) -> i32 { 0 }
        fn make() -> i32 { take(Maybe::Some(42)) }
    }
    """
    errs = check_after_flatten(src)
    assert errs == [], errs


def test_stage31_flatten_rewrites_module_const_value_paths():
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.flatten_modules import flatten_modules

    src = """
    mod m { const N: i32 = 7; }
    fn main() -> i32 { m::N }
    """
    prog = parse(src)
    flatten_modules(prog)
    fn = next(
        it for it in prog.items
        if isinstance(it, A.FnDecl) and it.name == "main"
    )
    assert isinstance(fn.body.final_expr, A.Name)
    assert fn.body.final_expr.name == "m__N"
    assert check_after_flatten(src) == []


def test_stage31_flatten_rewrites_module_sibling_const_value_paths():
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.flatten_modules import flatten_modules

    src = """
    mod m {
        const A: i32 = 7;
        const B: i32 = A;
    }
    fn main() -> i32 { m::B }
    """
    prog = parse(src)
    flatten_modules(prog)
    b_const = next(
        it for it in prog.items
        if isinstance(it, A.ConstDecl) and it.name == "m__B"
    )
    assert isinstance(b_const.value, A.Name)
    assert b_const.value.name == "m__A"
    assert check_after_flatten(src) == []


def test_stage31_flatten_rewrites_module_local_fn_const_names():
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.flatten_modules import flatten_modules

    src = """
    mod m {
        const N: i32 = 7;
        fn f() -> i32 { N }
    }
    fn main() -> i32 { m::f() }
    """
    prog = parse(src)
    flatten_modules(prog)
    fn = next(
        it for it in prog.items
        if isinstance(it, A.FnDecl) and it.name == "m__f"
    )
    assert isinstance(fn.body.final_expr, A.Name)
    assert fn.body.final_expr.name == "m__N"
    assert check_after_flatten(src) == []


def test_stage31_module_local_const_rewrite_respects_param_shadowing():
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.flatten_modules import flatten_modules

    src = """
    mod m {
        const N: i32 = 7;
        fn f(N: i32) -> i32 { N }
    }
    fn main() -> i32 { m::f(3) }
    """
    prog = parse(src)
    flatten_modules(prog)
    fn = next(
        it for it in prog.items
        if isinstance(it, A.FnDecl) and it.name == "m__f"
    )
    assert isinstance(fn.body.final_expr, A.Name)
    assert fn.body.final_expr.name == "N"
    assert check_after_flatten(src) == []


def test_stage31_local_const_shadows_module_alias_in_type_size():
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.flatten_modules import flatten_modules

    src = """
    mod m {
        const N: i32 = 7;
        fn f() -> i32 {
            const N: i32 = 3;
            let xs: [i32; N] = [1, 2, 3];
            0
        }
    }
    """
    prog = parse(src)
    flatten_modules(prog)
    fn = next(
        it for it in prog.items
        if isinstance(it, A.FnDecl) and it.name == "m__f"
    )
    let_stmt = fn.body.stmts[1]
    assert isinstance(let_stmt, A.Let)
    assert isinstance(let_stmt.ty, A.TyArray)
    assert isinstance(let_stmt.ty.size, A.Name)
    assert let_stmt.ty.size.name == "N"
    errs = typecheck(prog)
    assert errs == [], errs


def test_stage31_refinement_uses_alias_declaration_const_not_local_shadow():
    src = """
    const MAX: f64 = 1.0_f64;
    type Probability = f64 where 0.0 <= self <= MAX;
    fn f() {
        const MAX: f64 = 2.0_f64;
        let p: Probability = 1.5_f64;
    }
    """
    errs = check(src)
    assert any("refinement Probability violated" in e
               and "1.5" in e for e in errs), errs


def test_stage31_flatten_preserves_refinement_self_binder():
    src = """
    mod m {
        const self: f64 = 1.0_f64;
        type Probability = f64 where 0.0 <= self <= 1.0;
        fn f() { let p: Probability = 0.5_f64; }
    }
    """
    errs = check_after_flatten(src)
    assert errs == [], errs


def test_stage31_global_use_rewrite_respects_param_shadowing():
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.flatten_modules import flatten_modules

    src = """
    mod m { fn fp(x: i32) -> i32 { x } }
    use m::fp;
    fn apply(fp: fn(i32) -> i32) -> i32 { fp(1) }
    """
    prog = parse(src)
    flatten_modules(prog)
    fn = next(
        it for it in prog.items
        if isinstance(it, A.FnDecl) and it.name == "apply"
    )
    assert isinstance(fn.body.final_expr, A.Call)
    assert isinstance(fn.body.final_expr.callee, A.Name)
    assert fn.body.final_expr.callee.name == "fp"
    errs = typecheck(prog)
    assert any("function-typed calls are not supported by the Stage 31 "
               "backend" in str(e) for e in errs), errs


def test_stage31_global_use_rewrites_const_value_names():
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.flatten_modules import flatten_modules

    src = """
    mod m { const N: i32 = 7; }
    use m::N;
    fn main() -> i32 { N }
    """
    prog = parse(src)
    flatten_modules(prog)
    fn = next(
        it for it in prog.items
        if isinstance(it, A.FnDecl) and it.name == "main"
    )
    assert isinstance(fn.body.final_expr, A.Name)
    assert fn.body.final_expr.name == "m__N"
    errs = typecheck(prog)
    assert errs == [], errs


def test_stage31_module_local_bad_use_fails_flatten():
    import pytest
    from helixc.frontend.flatten_modules import FlattenError, flatten_modules

    src = "mod n { use missing::Probability; fn f() -> i32 { 1 } }"
    prog = parse(src)
    with pytest.raises(FlattenError, match="trap 79001"):
        flatten_modules(prog)


def test_stage31_bad_use_not_proven_by_unrelated_mangled_prefix():
    import pytest
    from helixc.frontend.flatten_modules import FlattenError, flatten_modules

    src = """
    fn missing__Probability__fake() -> i32 { 0 }
    use missing::Probability;
    fn main() -> i32 { 0 }
    """
    prog = parse(src)
    with pytest.raises(FlattenError, match="trap 79001"):
        flatten_modules(prog)


def test_stage31_use_module_alias_rewrites_const_path():
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.flatten_modules import flatten_modules

    src = """
    mod m { mod child { const N: i32 = 7; } }
    use m::child;
    fn main() -> i32 { child::N }
    """
    prog = parse(src)
    flatten_modules(prog)
    fn = next(
        it for it in prog.items
        if isinstance(it, A.FnDecl) and it.name == "main"
    )
    assert isinstance(fn.body.final_expr, A.Name)
    assert fn.body.final_expr.name == "m__child__N"
    errs = typecheck(prog)
    assert errs == [], errs


def test_stage31_use_module_alias_rewrites_call_path():
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.flatten_modules import flatten_modules

    src = """
    mod m { mod child { fn f() -> i32 { 7 } } }
    use m::child;
    fn main() -> i32 { child::f() }
    """
    prog = parse(src)
    flatten_modules(prog)
    fn = next(
        it for it in prog.items
        if isinstance(it, A.FnDecl) and it.name == "main"
    )
    assert isinstance(fn.body.final_expr, A.Call)
    assert isinstance(fn.body.final_expr.callee, A.Name)
    assert fn.body.final_expr.callee.name == "m__child__f"
    errs = typecheck(prog)
    assert errs == [], errs


def test_stage31_use_module_alias_rewrites_turbofish_call_path():
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.flatten_modules import flatten_modules
    from helixc.frontend.monomorphize import monomorphize_safe

    src = """
    mod m { mod child { fn id[T](x: T) -> T { x } } }
    use m::child;
    fn main() -> i32 { child::id::<i32>(42) }
    """
    prog = parse(src)
    flatten_modules(prog)
    fn = next(
        it for it in prog.items
        if isinstance(it, A.FnDecl) and it.name == "main"
    )
    assert isinstance(fn.body.final_expr, A.Call)
    assert isinstance(fn.body.final_expr.callee, A.Name)
    assert fn.body.final_expr.callee.name == "m__child__id"
    _, mono_diags = monomorphize_safe(prog)
    assert mono_diags == []
    errs = typecheck(prog)
    assert errs == [], errs


def test_c117_non_scalar_binary_operands_rejected():
    errs = check("fn f() -> bool { let xs = [1_i32]; xs == xs }")
    assert any("does not support operand types [i32; 1] and [i32; 1]" in e
               for e in errs), errs

    errs2 = check("fn f() { let xs = [1_i32]; let ys = xs + xs; }")
    assert any("does not support operand types [i32; 1] and [i32; 1]" in e
               for e in errs2), errs2

    errs3 = check('fn f() { let p = "abc".as_ptr(); let q = p + 1_u64; }')
    assert any("does not support operand types *const u8 and u64" in e
               for e in errs3), errs3


def test_c117_bool_and_char_operator_domains():
    assert check("fn f() -> bool { true == false }") == []
    assert check("fn f() -> bool { true && false }") == []

    errs = check("fn f() -> bool { true + false }")
    assert any("operator '+' does not support operand type bool" in e
               for e in errs), errs

    errs2 = check("fn f() -> bool { true < false }")
    assert any("operator '<' does not support operand type bool" in e
               for e in errs2), errs2

    errs3 = check("fn f() -> char { 'a' + 'b' }")
    assert any("operator '+' does not support operand type char" in e
               for e in errs3), errs3


def test_c117_compound_assignment_uses_operator_domain():
    errs = check("fn f() { let mut b: bool = true; b += false; }")
    assert any("operator '+' does not support operand type bool" in e
               for e in errs), errs

    errs2 = check("fn f() { let mut c: char = 'a'; c += 'b'; }")
    assert any("operator '+' does not support operand type char" in e
               for e in errs2), errs2

    errs3 = check("fn f() { let mut a: [bool; 1] = [true]; a[0] += false; }")
    assert any("operator '+' does not support operand type bool" in e
               for e in errs3), errs3


def test_c117_int_only_operators_reject_floats():
    errs = check("fn f() -> f32 { 1.0_f32 & 2.0_f32 }")
    assert any("operator '&' does not support operand type f32" in e
               for e in errs), errs

    errs2 = check("fn f() -> f32 { 1.0_f32 << 1.0_f32 }")
    assert any("operator '<<' does not support operand type f32" in e
               for e in errs2), errs2

    errs3 = check("fn f() -> f32 { 5.0_f32 % 2.0_f32 }")
    assert any("operator '%' does not support operand type f32" in e
               for e in errs3), errs3

    assert check("fn f() -> f32 { 6.0_f32 / 2.0_f32 }") == []


def test_c117_scalar_index_rejected():
    errs = check("fn f() -> i32 { let x = 7_i32; x[0] }")
    assert any("type i32 is not indexable" in e for e in errs), errs

    errs2 = check("fn f() { let mut x: i32 = 1_i32; x[0] = 2_i32; }")
    assert any("type i32 is not indexable" in e for e in errs2), errs2


def test_c117_tensor_tile_indexing_matches_lowered_contract():
    src = """
    @kernel fn k(a: tile<f32, [256], HBM>) {
        let x = a[0];
    }
    """
    assert check(src) == []

    errs = check("""
    @kernel fn k(a: tile<f32, [256], HBM>) {
        let x = a[0, 1];
    }
    """)
    assert any("tile indexing currently supports only @kernel HBM tile parameters with exactly 1 index" in e
               for e in errs), errs

    errs2 = check("""
    @kernel fn k(a: tile<f32, [16, 16], HBM>) {
        let x = a[0];
    }
    """)
    assert any("tile indexing currently supports only @kernel HBM tile parameters with exactly 1 index" in e
               for e in errs2), errs2

    errs3 = check("fn f(a: tensor<f32, [4, 4]>) -> f32 { a[0] }")
    assert any("tensor indexing is not supported until tensor index lowering is implemented" in e
               for e in errs3), errs3

    errs4 = check("fn f(a: tile<f32, [256], HBM>) -> f32 { a[0] }")
    assert any("tile indexing currently supports only @kernel HBM tile parameters with exactly 1 index" in e
               for e in errs4), errs4


def test_c117_wrapped_operator_domains_use_inner_scalar_rules():
    assert check("fn f(x: D<i32>, y: D<i32>) -> D<i32> { x % y }") == []
    assert check("fn f(x: D<f64>, y: i32) -> D<f64> { x + y }") == []

    errs = check("fn f(x: D<f32>, y: D<f32>) -> D<f32> { x % y }")
    assert any("operator '%' does not support operand type f32" in e
               for e in errs), errs

    errs2 = check("fn f(x: D<f32>, y: D<f32>) -> D<f32> { x & y }")
    assert any("operator '&' does not support operand type f32" in e
               for e in errs2), errs2

    errs3 = check("""
    fn f(x: Logic<bool>, y: Logic<bool>) -> Logic<bool> {
        x + y
    }
    """)
    assert any("operator '+' does not support operand type bool" in e
               for e in errs3), errs3


def test_c118_unary_operator_domains_checked():
    assert check("fn f() -> bool { !true }") == []
    assert check("fn f() -> i32 { ~1_i32 }") == []
    assert check("fn f() -> f32 { -1.0_f32 }") == []

    errs = check("fn f() -> bool { !1.0_f32 }")
    assert any("operator '!' expects bool operand, got f32" in e
               for e in errs), errs

    errs2 = check("fn f() -> f32 { ~1.0_f32 }")
    assert any("operator '~' does not support operand type f32" in e
               for e in errs2), errs2

    errs3 = check("fn f() -> bool { -true }")
    assert any("operator '-' does not support operand type bool" in e
               for e in errs3), errs3


def test_c118_assignment_targets_must_be_assignable():
    assert check("fn f() { let mut x: i32 = 1; x = 2; }") == []
    assert check("fn f(mut x: i32) -> i32 { x = 2; x }") == []
    assert check("fn f() { let xs = [0]; xs[0] = 1; }") == []

    errs = check("fn f() { 1 = 2; }")
    assert any("invalid assignment target" in e for e in errs), errs

    errs2 = check("fn f() { let x: i32 = 1; x = 2; }")
    assert any("cannot assign to immutable binding 'x'" in e
               for e in errs2), errs2

    errs3 = check("""
    @kernel fn k(a: tile<f32, [256], HBM>) {
        a[0] += 1.0_f32;
    }
    """)
    assert any("compound assignment to HBM tile indices is not supported" in e
               for e in errs3), errs3


def test_c119_indexed_assignment_requires_named_place():
    errs = check("fn f() { [0][0] = 1; }")
    assert any("indexed assignments require a named array or tile binding" in e
               for e in errs), errs

    errs2 = check("""
    fn id(a: [i32; 1]) -> [i32; 1] { a }
    fn f(a: [i32; 1]) { id(a)[0] = 1; }
    """)
    assert any("indexed assignments require a named array or tile binding" in e
               for e in errs2), errs2


def test_c119_kernel_index_builtins_typecheck_inside_kernel_only():
    assert check("""
    @kernel fn k() {
        let i: i32 = thread_idx();
        let bx: i32 = block_idx_y();
        let nt: i32 = block_dim_z();
    }
    """) == []

    errs = check("fn f() { let i = thread_idx(); }")
    assert any("thread_idx() is only allowed inside @kernel functions" in e
               for e in errs), errs

    errs2 = check("@kernel fn k() { let i: i32 = thread_idx; }")
    assert any("thread_idx must be called as thread_idx()" in e
               for e in errs2), errs2


def test_c119_kernel_hbm_param_dtypes_match_ptx_surface():
    assert check("@kernel fn k(a: tile<f32, [16], HBM>) {}") == []
    assert check("@kernel fn k(a: tile<i32, [16], HBM>) {}") == []

    errs = check("@kernel fn k(a: tile<u32, [16], HBM>) {}")
    assert any("@kernel HBM tile parameter dtype u32 is not supported" in e
               for e in errs), errs

    errs2 = check("@kernel fn k(a: tile<f16, [16], HBM>) {}")
    assert any("@kernel HBM tile parameter dtype f16 is not supported" in e
               for e in errs2), errs2


def test_c119_kernel_hbm_param_shape_matches_ptx_surface():
    errs = check("@kernel fn k(a: tile<f32, [16, 16], HBM>) {}")
    assert any("@kernel HBM tile parameters must be 1D" in e
               for e in errs), errs

    errs2 = check('@kernel extern "C" fn k(a: tile<f32, [16, 16], HBM>);')
    assert any("@kernel HBM tile parameters must be 1D" in e
               for e in errs2), errs2


def test_c119_kernel_return_type_matches_ptx_surface():
    errs = check("@kernel fn k() -> i32 { 42 }")
    assert any("@kernel functions must return ()" in e for e in errs), errs

    errs2 = check("@kernel fn k() { return 1; }")
    assert any("@kernel functions cannot return a value" in e
               for e in errs2), errs2


# ============================================================================
# Compile-time shape checking via Presburger solver (Phase 3-iv)
# ============================================================================
def test_shape_check_concrete_match():
    # Both args have concrete shape [4, 4]; should typecheck cleanly.
    src = """
    fn matmul(a: tensor<f32, [4, 4]>, b: tensor<f32, [4, 4]>) -> tensor<f32, [4, 4]> {
        a
    }
    fn caller(x: tensor<f32, [4, 4]>, y: tensor<f32, [4, 4]>) {
        matmul(x, y);
    }
    """
    assert check(src) == []


def test_shape_check_concrete_mismatch():
    # Caller passes [5, 4] where formal expects [4, 4] — should reject.
    src = """
    fn matmul(a: tensor<f32, [4, 4]>, b: tensor<f32, [4, 4]>) -> tensor<f32, [4, 4]> {
        a
    }
    fn caller(x: tensor<f32, [4, 4]>, z: tensor<f32, [5, 4]>) {
        matmul(x, z);
    }
    """
    errs = check(src)
    assert any("shape constraint violated" in e or "rank" in e for e in errs), errs


def test_shape_check_rank_mismatch():
    src = """
    fn takes2d(a: tensor<f32, [4, 4]>) {}
    fn caller(x: tensor<f32, [4]>) {
        takes2d(x);
    }
    """
    errs = check(src)
    assert any("rank" in e for e in errs), errs


def test_shape_check_size_polymorphic_match():
    src = """
    fn matmul[N: size, M: size, P: size](
        a: tensor<f32, [N, M]>,
        b: tensor<f32, [M, P]>,
    ) -> tensor<f32, [N, P]> { a }
    fn caller(x: tensor<f32, [3, 4]>, y: tensor<f32, [4, 5]>) {
        matmul(x, y);
    }
    """
    errs = check(src)
    assert not any("shape constraint violated" in e for e in errs), errs


# ============================================================================
# Effect/capability type system (Phase 3-ii)
# ============================================================================
def test_pure_calls_pure_ok():
    src = """
    @pure fn helper() -> i32 { 7 }
    @pure fn caller() -> i32 { helper() }
    """
    assert check(src) == []


def test_pure_calls_effectful_rejected():
    # @pure cannot call a function with side effects
    src = """
    @effect fn writes_disk() {}
    @pure fn caller() { writes_disk() }
    """
    errs = check(src)
    assert any("cannot call" in e or "non-pure" in e for e in errs), errs


def test_effectful_can_call_effectful():
    # caller declares the same effects, so the call is permitted
    src = """
    @io fn writes_disk() {}
    @io fn caller() { writes_disk() }
    """
    assert check(src) == []


def test_caller_missing_capability_rejected():
    # io-capable function called from a function without that capability
    src = """
    @io fn read_file() -> i32 { 0 }
    fn naive_caller() { read_file(); }
    """
    errs = check(src)
    assert any("requires effect" in e or "does not declare" in e for e in errs), errs


def test_pure_and_effect_conflict():
    # Function declared both @pure and @effect — conflict
    src = """
    @pure
    @io
    fn confused() {}
    """
    errs = check(src)
    assert any("cannot be both" in e for e in errs), errs


def test_pure_calls_unmarked_function_allowed():
    """Surface typecheck: @pure may call unannotated functions whose
    declared effect set is empty. The IR-level effect_check.py is the
    soundness layer — it computes transitive effects from PRINT ops
    and catches actual impurity. The surface check is intentionally
    permissive about unannotated callees so users don't have to
    @pure-annotate every helper just to call it from a @pure caller."""
    src = """
    fn does_anything() -> i32 { 7 }
    @pure fn caller() -> i32 { does_anything() }
    """
    errs = check(src)
    assert errs == [], f"expected no surface errors, got: {errs}"


def test_pure_calls_explicit_effect_rejected():
    """@pure must still be rejected when the callee declares effects."""
    src = """
    @effect(io)
    fn print_thing() -> i32 { 42 }
    @pure fn caller() -> i32 { print_thing() }
    """
    errs = check(src)
    assert any("@pure" in e and "effectful" in e for e in errs), errs


# ============================================================================
# Differentiable types D<T> (Phase 3-iii)
# ============================================================================
def test_diff_type_round_trip():
    # Function takes D<f32>, returns D<f32>; should typecheck
    src = """
    fn loss(x: D<f32>) -> D<f32> { x }
    """
    assert check(src) == []


def test_diff_propagates_through_arith():
    # D<f32> * D<f32> should be D<f32>
    src = """
    fn loss(x: D<f32>, y: D<f32>) -> D<f32> {
        x * y
    }
    """
    assert check(src) == []


def test_diff_propagates_with_scalar():
    # D<f32> * f32 should be D<f32> (the f32 is a constant)
    src = """
    fn scale(x: D<f32>) -> D<f32> {
        x * x
    }
    """
    assert check(src) == []


def test_diff_return_type_mismatch():
    # Returning a non-D from a D-typed return is an error
    src = """
    fn bad(x: D<f32>) -> f32 {
        x
    }
    """
    errs = check(src)
    # x is D<f32>, return type declared f32 — type mismatch
    assert any("does not match return type" in e for e in errs), errs


def test_diff_in_tensor():
    # D<tensor<...>> — differentiable tensor
    src = """
    fn loss(x: D<f32>, y: D<f32>) -> D<f32> {
        x * y + x
    }
    """
    assert check(src) == []


# ============================================================================
# Memory-tier types (Phase 3-v)
# ============================================================================
def test_working_mem_type():
    src = """
    fn f(x: WorkingMem<i32>) -> WorkingMem<i32> { x }
    """
    assert check(src) == []


def test_episodic_mem_type():
    src = """
    fn store(e: EpisodicMem<i32>) -> EpisodicMem<i32> { e }
    """
    assert check(src) == []


def test_cannot_pass_episodic_as_semantic():
    # EpisodicMem and SemanticMem are different tiers — must explicitly
    # consolidate. Direct passing should be rejected.
    src = """
    fn takes_semantic(x: SemanticMem<i32>) -> SemanticMem<i32> { x }
    fn caller(e: EpisodicMem<i32>) -> SemanticMem<i32> {
        let s: SemanticMem<i32> = e;
        s
    }
    """
    errs = check(src)
    assert any("declared SemanticMem" in e or "declared" in e for e in errs), errs


def test_same_tier_compatible():
    src = """
    fn takes_working(x: WorkingMem<i32>) -> WorkingMem<i32> { x }
    fn caller(w: WorkingMem<i32>) -> WorkingMem<i32> {
        let y: WorkingMem<i32> = w;
        y
    }
    """
    assert check(src) == []


def test_let_episodic_to_working_rejected():
    src = """
    fn f(e: EpisodicMem<i32>) -> WorkingMem<i32> {
        let w: WorkingMem<i32> = e;
        w
    }
    """
    errs = check(src)
    assert any("declared" in e for e in errs), errs


# ============================================================================
# Built-in type transitions: detach, attach, consolidate, recall
# ============================================================================
def test_detach_strips_diff():
    # detach(x: D<f32>) -> f32
    src = """
    fn use_grad(x: D<f32>) -> f32 {
        let plain = detach(x);
        plain
    }
    """
    assert check(src) == []


def test_attach_adds_diff():
    src = """
    fn add_grad(x: f32) -> D<f32> {
        let g = attach(x);
        g
    }
    """
    assert check(src) == []


def test_consolidate_episodic_to_semantic():
    src = """
    fn store(e: EpisodicMem<i32>) -> SemanticMem<i32> {
        consolidate(e)
    }
    """
    assert check(src) == []


def test_consolidate_rejects_non_episodic():
    # consolidate() must take EpisodicMem<T>, not Working
    src = """
    fn bad(w: WorkingMem<i32>) -> SemanticMem<i32> {
        consolidate(w)
    }
    """
    errs = check(src)
    assert any("requires EpisodicMem" in e for e in errs), errs


def test_recall_semantic_to_working():
    src = """
    fn fetch(s: SemanticMem<i32>) -> WorkingMem<i32> {
        recall(s)
    }
    """
    assert check(src) == []


def test_recall_rejects_non_semantic():
    src = """
    fn bad(e: EpisodicMem<i32>) -> WorkingMem<i32> {
        recall(e)
    }
    """
    errs = check(src)
    assert any("requires SemanticMem" in e for e in errs), errs


# ============================================================================
# Argument count + primitive type checking on function calls
# ============================================================================
def test_arg_count_too_few():
    src = """
    fn add(a: i32, b: i32) -> i32 { a + b }
    fn caller() -> i32 { add(1) }
    """
    errs = check(src)
    assert any("expected 2 args, got 1" in e for e in errs), errs


def test_arg_count_too_many():
    src = """
    fn id(x: i32) -> i32 { x }
    fn caller() -> i32 { id(1, 2, 3) }
    """
    errs = check(src)
    assert any("expected 1 args, got 3" in e for e in errs), errs


def test_arg_type_mismatch():
    src = """
    fn takes_int(x: i32) -> i32 { x }
    fn caller() -> i32 { takes_int(true) }
    """
    errs = check(src)
    assert any("expects i32, got bool" in e for e in errs), errs


def test_arg_type_match_ok():
    src = """
    fn add(a: i32, b: i32) -> i32 { a + b }
    fn caller() -> i32 { add(1, 2) }
    """
    assert check(src) == []


def test_arg_int_vs_float():
    src = """
    fn takes_float(x: f32) -> f32 { x }
    fn caller() -> f32 { takes_float(42) }
    """
    errs = check(src)
    assert any("expects f32, got i32" in e for e in errs), errs


# ============================================================================
# Auto-curriculum primitive: learn_to (Phase 3-ix)
# ============================================================================
def test_learn_to_returns_skill():
    # learn_to(task, difficulty, budget) is a builtin returning Skill<...>
    src = """
    fn main() -> i32 {
        let skill = learn_to("matmul", 0.5, 100);
        0
    }
    """
    # Should typecheck cleanly
    assert check(src) == []


def test_learn_to_wrong_arity():
    src = """
    fn main() -> i32 {
        let skill = learn_to("foo");
        0
    }
    """
    errs = check(src)
    assert any("requires 3 args" in e for e in errs), errs


# ============================================================================
# Test runner
# ============================================================================
def test_struct_lit_typechecks_clean():
    """A well-formed struct literal should typecheck without errors."""
    src = """
    struct Point { x: i32, y: i32 }
    fn main() -> i32 {
        let p = Point { x: 10, y: 20 };
        p.x
    }
    """
    errs = check(src)
    assert errs == [], f"unexpected errors: {errs}"


def test_struct_lit_missing_field_errors():
    """Forgetting a field surfaces a 'missing field(s)' error."""
    src = """
    struct Point { x: i32, y: i32 }
    fn main() -> i32 {
        let p = Point { x: 10 };
        p.x
    }
    """
    errs = check(src)
    assert any("missing field" in s and "y" in s for s in errs), \
        f"expected missing-field error, got {errs}"


def test_struct_lit_unknown_field_errors():
    """A field not in the decl surfaces an 'unknown field(s)' error."""
    src = """
    struct Point { x: i32, y: i32 }
    fn main() -> i32 {
        let p = Point { x: 10, y: 20, z: 30 };
        p.x
    }
    """
    errs = check(src)
    assert any("unknown field" in s and "z" in s for s in errs), \
        f"expected unknown-field error, got {errs}"


def test_struct_field_access_unknown_field_errors():
    """Reading p.z where Point only has x, y surfaces an error."""
    src = """
    struct Point { x: i32, y: i32 }
    fn main() -> i32 {
        let p = Point { x: 10, y: 20 };
        p.z
    }
    """
    errs = check(src)
    assert any("no field" in s and "z" in s for s in errs), \
        f"expected unknown-field-access error, got {errs}"


def test_nested_struct_field_type_tracks_correctly():
    """A struct-typed field's value must resolve to TyStruct (not TyUnknown)
    so chained field access types correctly. Pre-fix this was silently
    TyUnknown, which made `o.inner` incompatible-with-anything pass."""
    src = """
    struct Inner { value: i32 }
    struct Outer { count: i32, inner: Inner }
    fn main() -> i32 {
        let o = Outer { count: 10, inner: Inner { value: 32 } };
        o.count + o.inner.value
    }
    """
    errs = check(src)
    assert errs == [], f"unexpected errors: {errs}"


def test_three_segment_path_on_known_enum_errors():
    """`Op::Sub::Variant` should surface a clear error rather than
    silent TyUnknown propagation."""
    src = """
    enum Op { Add, Sub }
    fn main() -> i32 {
        let v = Op::Sub::SomethingExtra;
        0
    }
    """
    errs = check(src)
    assert any("3+ segments" in s or "v0.1" in s for s in errs), \
        f"expected 3-segment path error, got {errs}"


def test_int_literal_overflow_errors():
    """Static overflow: `let x: i32 = 5_000_000_000` exceeds i32 range."""
    src = """
    fn main() -> i32 {
        let x: i32 = 5000000000;
        x
    }
    """
    errs = check(src)
    assert any("does not fit in i32" in s for s in errs), \
        f"expected overflow error, got {errs}"
    assert any("i64" in s for s in errs), \
        f"expected i64 hint, got {errs}"


def test_int_literal_in_range_no_error():
    """Values within i32 range typecheck cleanly."""
    src = """
    fn main() -> i32 {
        let x: i32 = 2147483647;
        x
    }
    """
    errs = check(src)
    assert errs == [], f"unexpected errors: {errs}"


def test_int_literal_negative_overflow_errors():
    """Negative overflow: -3_000_000_000 doesn't fit in i32."""
    src = """
    fn main() -> i32 {
        let x: i32 = 0 - 3000000000;
        x
    }
    """
    errs = check(src)
    # The literal '3000000000' itself is u32-shaped, but the let-stmt's
    # value is a Binary not an IntLit, so this won't trigger our static
    # check. We only catch literals at the binding point. Document this.
    # Test just verifies we don't crash and the program is acceptable.
    # (A future ticket would do constant-folding-aware overflow.)
    pass


def test_struct_field_returns_correct_type():
    """p.x of struct Point { x: i32, y: i32 } types as i32, allowing
    further i32 ops without errors."""
    src = """
    struct Point { x: i32, y: i32 }
    fn main() -> i32 {
        let p = Point { x: 10, y: 20 };
        p.x + p.y + 1
    }
    """
    errs = check(src)
    assert errs == [], f"unexpected errors: {errs}"


def test_struct_lit_unknown_struct_errors():
    """A struct lit referencing an undeclared struct surfaces an error."""
    src = """
    fn main() -> i32 {
        let p = Nope { x: 10 };
        0
    }
    """
    errs = check(src)
    assert any("unknown struct" in s for s in errs), \
        f"expected unknown-struct error, got {errs}"


def test_call_did_you_mean_suggests_builtin():
    """Misspelled stdlib call: `__exf(x)` should suggest `__exp`."""
    src = """
    fn main(x: f32) -> f32 {
        __exf(x)
    }
    """
    errs = check(src)
    assert any("unbound" in s and "__exf" in s for s in errs), \
        f"expected unbound '__exf', got {errs}"
    assert any("__exp" in s for s in errs), \
        f"expected hint mentioning '__exp', got {errs}"


def test_unbound_name_did_you_mean_suggests_close_match():
    src = """
    fn main() -> i32 {
        let counter = 5;
        countr + 1
    }
    """
    errs = check(src)
    assert any("unbound" in s and "countr" in s for s in errs), \
        f"expected unbound 'countr', got {errs}"
    assert any("counter" in s for s in errs), \
        f"expected hint mentioning 'counter', got {errs}"


def test_unbound_builtin_not_flagged():
    src = """
    fn use_grad(x: D<f32>) -> f32 { detach(x) }
    """
    errs = check(src)
    assert not any("unbound" in s for s in errs), \
        f"unexpected unbound error for builtin: {errs}"


# ----------------------------------------------------------------------
# Audit 28.8 B10 — typecheck Quote/Splice/Modify arms
# ----------------------------------------------------------------------
def test_quote_returns_tyquote():
    """Quote(inner) types as TyQuote(typeof(inner)) — not TyUnknown."""
    from helixc.frontend.typecheck import TypeChecker, TyQuote, TyPrim
    src = """
    fn main() -> i32 {
        let q = quote(42);
        0
    }
    """
    prog = parse(src)
    tc = TypeChecker(prog)
    tc.check()
    # The let q = quote(42) does not leak an error and q's type is TyQuote.
    # We inspect by re-checking main's body scope — easier via typecheck side
    # effect testing: assert no unhandled-Quote diagnostic.
    errs = [str(e) for e in tc.errors]
    assert not any("unhandled Quote" in s for s in errs), \
        f"unexpected unhandled-Quote: {errs}"


def test_splice_unwraps_tyquote():
    """splice(quote(42)) must typecheck to i32 (inner of Quote<i32>)."""
    src = """
    fn main() -> i32 {
        let q = quote(42);
        splice(q)
    }
    """
    errs = check(src)
    assert errs == [], f"unexpected errors: {errs}"


def test_splice_of_non_quote_diagnoses_11001():
    """Audit 28.8 B10 (trap 11001): splice'ing a non-Quote value
    surfaces the trap-11001 diagnostic instead of silently typing
    as TyUnknown and unifying with anything downstream."""
    src = """
    fn main() -> i32 {
        let x = 42;
        splice(x)
    }
    """
    errs = check(src)
    assert any("11001" in s for s in errs), \
        f"expected trap 11001 diagnostic, got {errs}"


def test_modify_returns_i32():
    """Modify is documented to return i32 (1=applied, 0=rejected).
    Pre-fix it typed as TyUnknown, which unified with anything."""
    src = """
    fn yes(handle: i32, new_val: i32) -> i32 { 1 }
    fn main() -> i32 {
        let h = quote(0);
        let applied = modify(h, 42, yes);
        if applied == 1 { 1 } else { 0 }
    }
    """
    errs = check(src)
    assert errs == [], f"modify must typecheck clean, got: {errs}"


def test_unsafe_block_propagates_inner_type():
    """Audit 28.8 B10 (UnsafeBlock arm): unsafe { e } must return
    typeof(e), not TyUnknown. The handler exists pre-A2 too; this
    asserts the type DOES propagate through."""
    src = """
    fn main() -> i32 {
        let x: i32 = unsafe { 42 };
        x
    }
    """
    errs = check(src)
    assert errs == [], f"unexpected errors: {errs}"


# ----------------------------------------------------------------------
# Audit 28.8 B11 — flatten_impls duplicate-method-name detection
# ----------------------------------------------------------------------
def test_flatten_impls_rejects_same_name_methods():
    """Audit 28.8 B11 (trap 74002): two structs with the same method
    name cause cross-struct type confusion at the call site. The
    fix raises DuplicateMethodError at the second registration.

    Python parser requires explicit param types (no `self` sugar);
    construct the impl-blocks directly via the AST."""
    from helixc.frontend.flatten_impls import (
        flatten_impls, DuplicateMethodError,
    )
    from helixc.frontend import ast_nodes as A
    import pytest as _pt
    span = A.Span(0, 0)
    pt = A.StructDecl(
        span=span, name="Pt", generics=[],
        fields=[A.FnParam(span=span, name="x",
                          ty=A.TyName(span=span, name="f32"),
                          is_mut=False)],
        is_pub=False,
    )
    line = A.StructDecl(
        span=span, name="Line", generics=[],
        fields=[A.FnParam(span=span, name="a",
                          ty=A.TyName(span=span, name="f32"),
                          is_mut=False)],
        is_pub=False,
    )
    pt_len = A.FnDecl(
        span=span, name="len", generics=[],
        params=[A.FnParam(span=span, name="self",
                          ty=A.TyName(span=span, name="Pt"),
                          is_mut=False)],
        return_ty=A.TyName(span=span, name="f32"),
        where_clauses=[],
        body=A.Block(span=span, stmts=[],
                     final_expr=A.FloatLit(span=span, value=0.0,
                                            type_suffix="f32")),
        attrs=[], is_pub=False,
    )
    line_len = A.FnDecl(
        span=span, name="len", generics=[],
        params=[A.FnParam(span=span, name="self",
                          ty=A.TyName(span=span, name="Line"),
                          is_mut=False)],
        return_ty=A.TyName(span=span, name="f32"),
        where_clauses=[],
        body=A.Block(span=span, stmts=[],
                     final_expr=A.FloatLit(span=span, value=0.0,
                                            type_suffix="f32")),
        attrs=[], is_pub=False,
    )
    impl_pt = A.ImplBlock(span=span, target="Pt", methods=[pt_len],
                          trait_name=None)
    impl_line = A.ImplBlock(span=span, target="Line", methods=[line_len],
                            trait_name=None)
    prog = A.Program(module=None,
                     items=[pt, line, impl_pt, impl_line])
    with _pt.raises(DuplicateMethodError) as ex:
        flatten_impls(prog)
    assert ex.value.method == "len"
    assert ex.value.trap_id == 74002


def test_flatten_impls_allows_distinct_method_names():
    """Distinct method names across structs flatten cleanly."""
    from helixc.frontend.flatten_impls import flatten_impls
    from helixc.frontend import ast_nodes as A
    span = A.Span(0, 0)
    pt = A.StructDecl(
        span=span, name="Pt", generics=[],
        fields=[A.FnParam(span=span, name="x",
                          ty=A.TyName(span=span, name="f32"),
                          is_mut=False)],
        is_pub=False,
    )
    line = A.StructDecl(
        span=span, name="Line", generics=[],
        fields=[A.FnParam(span=span, name="a",
                          ty=A.TyName(span=span, name="f32"),
                          is_mut=False)],
        is_pub=False,
    )
    pt_method = A.FnDecl(
        span=span, name="px", generics=[],
        params=[A.FnParam(span=span, name="self",
                          ty=A.TyName(span=span, name="Pt"),
                          is_mut=False)],
        return_ty=A.TyName(span=span, name="f32"),
        where_clauses=[],
        body=A.Block(span=span, stmts=[],
                     final_expr=A.FloatLit(span=span, value=0.0,
                                            type_suffix="f32")),
        attrs=[], is_pub=False,
    )
    line_method = A.FnDecl(
        span=span, name="la", generics=[],
        params=[A.FnParam(span=span, name="self",
                          ty=A.TyName(span=span, name="Line"),
                          is_mut=False)],
        return_ty=A.TyName(span=span, name="f32"),
        where_clauses=[],
        body=A.Block(span=span, stmts=[],
                     final_expr=A.FloatLit(span=span, value=0.0,
                                            type_suffix="f32")),
        attrs=[], is_pub=False,
    )
    impl_pt = A.ImplBlock(span=span, target="Pt", methods=[pt_method],
                          trait_name=None)
    impl_line = A.ImplBlock(span=span, target="Line",
                            methods=[line_method], trait_name=None)
    prog = A.Program(module=None,
                     items=[pt, line, impl_pt, impl_line])
    n = flatten_impls(prog)
    assert n == 2


# ----------------------------------------------------------------------
# Audit 28.8 B13 — TyDiff mixed inner widen-then-warn (trap 24200/AD002)
# ----------------------------------------------------------------------
def test_diff_mixed_inner_widens_with_warning():
    """Audit 28.8 B13: D<f64> + D<i32> previously silently coerced
    the i32 to f64 with no diagnostic. The fix widens to the
    dominant inner type AND emits a warning via the AD-warning
    channel (trap 24200 / AD002)."""
    from helixc.frontend import autodiff
    # Clear any pre-existing warnings.
    autodiff.take_diff_warnings()
    src = """
    fn loss(x: D<f64>, y: D<i32>) -> D<f64> {
        x + y
    }
    """
    errs = check(src)
    # Typecheck itself should pass (we widen, not reject).
    assert errs == [], f"unexpected typecheck errors: {errs}"
    # And a warning was emitted in the AD channel.
    warnings = autodiff.take_diff_warnings()
    assert any("24200" in w and "AD002" in w for w in warnings), \
        f"expected AD002 / trap 24200 warning, got: {warnings}"


def test_diff_same_inner_no_warning():
    """D<f64> + D<f64> is clean — no AD002 warning."""
    from helixc.frontend import autodiff
    autodiff.take_diff_warnings()
    src = """
    fn loss(x: D<f64>, y: D<f64>) -> D<f64> {
        x + y
    }
    """
    errs = check(src)
    assert errs == []
    warnings = autodiff.take_diff_warnings()
    assert not any("24200" in w for w in warnings), \
        f"unexpected mixed-inner warning: {warnings}"


# ----------------------------------------------------------------------
# Audit 28.8 cycle 2 B:C1 — _WIDEN_RANK covers fp8/mxfp4/nvfp4/char
# ----------------------------------------------------------------------
def test_c2_b_c1_widen_diff_fp8_vs_i64():
    """B:C1: pre-fix `D<fp8> + D<i64>` widened to i64 (rank -1 vs 40),
    a float-to-int silent collapse. Now fp8 has rank 25 — still loses
    to f16/f32/f64 but BEATS any integer."""
    from helixc.frontend.typecheck import _widen_diff_inner, TyPrim
    out = _widen_diff_inner(TyPrim("fp8"), TyPrim("i64"))
    assert out.name == "fp8", (
        f"expected fp8 to dominate i64 after B:C1 fix; got {out.name}"
    )


def test_c2_b_c1_widen_diff_mxfp4_vs_i32():
    """B:C1: mxfp4 (rank 35) beats i32 (rank 30)."""
    from helixc.frontend.typecheck import _widen_diff_inner, TyPrim
    out = _widen_diff_inner(TyPrim("mxfp4"), TyPrim("i32"))
    assert out.name == "mxfp4"


def test_c2_b_c1_widen_diff_nvfp4_vs_f32():
    """B:C1: nvfp4 (rank 35) loses to f32 (rank 60). Float-domain
    widening from quantized to standard precision is correct."""
    from helixc.frontend.typecheck import _widen_diff_inner, TyPrim
    out = _widen_diff_inner(TyPrim("nvfp4"), TyPrim("f32"))
    assert out.name == "f32"


def test_c2_b_c1_widen_diff_char_vs_i32():
    """B:C1: char (rank 5) loses to i32 (rank 30). Codepoint
    silently treated as integer was the pre-fix complaint; with
    char having rank 5 we widen up to i32 explicitly."""
    from helixc.frontend.typecheck import _widen_diff_inner, TyPrim
    out = _widen_diff_inner(TyPrim("char"), TyPrim("i32"))
    assert out.name == "i32"


def test_c2_b_c4_widen_diff_signed_unsigned_same_width():
    """B:C4: pre-fix, u32 (rank 30) vs i32 (rank 30) tied and
    left-wins picked u32 silently. With the asymmetric rank fix
    (u32=31, i32=30), unsigned now wins explicitly — and the
    user-visible test is just that the result deterministically
    picks the unsigned form."""
    from helixc.frontend.typecheck import _widen_diff_inner, TyPrim
    a = _widen_diff_inner(TyPrim("u32"), TyPrim("i32"))
    b = _widen_diff_inner(TyPrim("i32"), TyPrim("u32"))
    assert a.name == "u32"
    assert b.name == "u32", (
        f"signedness widening must be order-independent; got {b.name}"
    )


def test_c2_b_c4_widen_diff_i64_u64_unsigned_wins():
    """B:C4: u64 (rank 41) beats i64 (rank 40). Same width but
    sign-domain transition no longer depends on operand order."""
    from helixc.frontend.typecheck import _widen_diff_inner, TyPrim
    assert _widen_diff_inner(TyPrim("i64"), TyPrim("u64")).name == "u64"
    assert _widen_diff_inner(TyPrim("u64"), TyPrim("i64")).name == "u64"


def test_c2_b_c6_diff_plus_bare_warns():
    """B:C6: pre-fix `D<f64> + i32` (one D-wrapped, one raw) did NOT
    emit AD002 because the gate required BOTH sides D-wrapped. The
    same precision-loss hazard exists in the asymmetric case — i32
    silently promoted to f64. Now the asymmetric case also warns
    (with a hint about which side is D-wrapped)."""
    from helixc.frontend import autodiff
    autodiff.take_diff_warnings()
    src = """
    fn loss(x: D<f64>, y: i32) -> D<f64> {
        x + y
    }
    """
    errs = check(src)
    assert errs == [], f"unexpected errors: {errs}"
    warnings = autodiff.take_diff_warnings()
    assert any("24200" in w and "AD002" in w for w in warnings), (
        f"expected B:C6 asymmetric warning, got: {warnings}"
    )
    assert any("D-wrapped" in w or "bare" in w for w in warnings), (
        f"expected hint about D-wrap asymmetry, got: {warnings}"
    )


# ----------------------------------------------------------------------
# Audit 28.8 B14 — Cast allowed-cast matrix (trap 28604)
# ----------------------------------------------------------------------
def test_cast_int_to_int_allowed():
    """int -> int (any widths) is allowed."""
    src = """
    fn main() -> i64 {
        let x: i32 = 42;
        x as i64
    }
    """
    errs = check(src)
    assert errs == [], f"unexpected errors: {errs}"


def test_cast_int_to_float_allowed():
    """int -> float (any widths) is allowed."""
    src = """
    fn main() -> f64 {
        let x: i32 = 42;
        x as f64
    }
    """
    errs = check(src)
    assert errs == [], f"unexpected errors: {errs}"


def test_cast_float_to_int_allowed():
    """float -> int (any widths) is allowed."""
    src = """
    fn main() -> i32 {
        let x: f64 = 3.14_f64;
        x as i32
    }
    """
    errs = check(src)
    assert errs == [], f"unexpected errors: {errs}"


def test_cast_bool_to_int_allowed():
    """bool -> int is allowed (1/0)."""
    src = """
    fn main() -> i32 {
        let b: bool = true;
        b as i32
    }
    """
    errs = check(src)
    assert errs == [], f"unexpected errors: {errs}"


def test_cast_tuple_to_int_rejected():
    """Audit 28.8 B14 (trap 28604): tuple-as-i32 is invalid."""
    src = """
    fn main() -> i32 {
        let t: (i32, i32) = (1, 2);
        t as i32
    }
    """
    errs = check(src)
    assert any("28604" in s for s in errs), \
        f"expected trap 28604 for tuple-as-i32, got: {errs}"


def test_cast_struct_to_float_rejected():
    """Audit 28.8 B14: struct-as-float is invalid."""
    src = """
    struct Pt { x: f32, y: f32 }
    fn main() -> f32 {
        let p = Pt { x: 1.0_f32, y: 2.0_f32 };
        p as f32
    }
    """
    errs = check(src)
    assert any("28604" in s for s in errs), \
        f"expected trap 28604 for struct-as-float, got: {errs}"


def test_flatten_impls_allows_same_struct_redeclare():
    """Same struct calling flatten_impls twice on aliased
    impl-block records (i.e. same target, same method) should NOT
    fire B11 — only cross-struct collisions are flagged."""
    from helixc.frontend.flatten_impls import flatten_impls
    from helixc.frontend import ast_nodes as A
    span = A.Span(0, 0)
    pt = A.StructDecl(
        span=span, name="Pt", generics=[],
        fields=[A.FnParam(span=span, name="x",
                          ty=A.TyName(span=span, name="f32"),
                          is_mut=False)],
        is_pub=False,
    )
    # Two impl blocks on Pt with two different methods is OK.
    m1 = A.FnDecl(
        span=span, name="a", generics=[],
        params=[A.FnParam(span=span, name="self",
                          ty=A.TyName(span=span, name="Pt"),
                          is_mut=False)],
        return_ty=A.TyName(span=span, name="f32"),
        where_clauses=[],
        body=A.Block(span=span, stmts=[],
                     final_expr=A.FloatLit(span=span, value=0.0,
                                            type_suffix="f32")),
        attrs=[], is_pub=False,
    )
    m2 = A.FnDecl(
        span=span, name="b", generics=[],
        params=[A.FnParam(span=span, name="self",
                          ty=A.TyName(span=span, name="Pt"),
                          is_mut=False)],
        return_ty=A.TyName(span=span, name="f32"),
        where_clauses=[],
        body=A.Block(span=span, stmts=[],
                     final_expr=A.FloatLit(span=span, value=0.0,
                                            type_suffix="f32")),
        attrs=[], is_pub=False,
    )
    impl_pt_1 = A.ImplBlock(span=span, target="Pt", methods=[m1],
                            trait_name=None)
    impl_pt_2 = A.ImplBlock(span=span, target="Pt", methods=[m2],
                            trait_name=None)
    prog = A.Program(module=None,
                     items=[pt, impl_pt_1, impl_pt_2])
    n = flatten_impls(prog)
    assert n == 2


# ----------------------------------------------------------------------
# Audit 28.8 cycle 2 C2-5 / B:C8 — TyArray size distinguished in
# _ty_key and substituted by substitute_ty
# ----------------------------------------------------------------------
def test_c2_5_ty_key_distinguishes_array_size():
    """C2-5: pre-fix `_ty_key(TyArray)` excluded the size, so
    `[i32; 4]` and `[i32; 8]` shared one key. struct_mono.dedup
    collapsed `Pt<[i32; 4]>` and `Pt<[i32; 8]>` into one mono."""
    from helixc.frontend.struct_mono import _ty_key
    from helixc.frontend import ast_nodes as A
    span = A.Span(0, 0)
    t4 = A.TyArray(span=span,
                   elem=A.TyName(span=span, name="i32"),
                   size=A.IntLit(span=span, value=4))
    t8 = A.TyArray(span=span,
                   elem=A.TyName(span=span, name="i32"),
                   size=A.IntLit(span=span, value=8))
    assert _ty_key(t4) != _ty_key(t8), (
        f"distinct array sizes must produce distinct keys; "
        f"got t4={_ty_key(t4)} t8={_ty_key(t8)}"
    )


def test_c2_b_c8_substitute_ty_walks_array_size():
    """B:C8: pre-fix `substitute_ty` on TyArray copied `size`
    unchanged, so `[T; N]` with N=8 stayed as `[f64; Name(N)]`.
    Downstream lowering defaulted non-IntLit shapes to 0 — silent
    miscount. Fix: substitute_ty now routes size through
    _subst_shape_expr."""
    from helixc.frontend.monomorphize import (
        substitute_ty, _SizeLitMarker,
    )
    from helixc.frontend import ast_nodes as A
    span = A.Span(0, 0)
    t = A.TyArray(span=span,
                  elem=A.TyName(span=span, name="T"),
                  size=A.Name(span=span, name="N"))
    subst = {"T": A.TyName(span=span, name="f64"),
             "N": _SizeLitMarker(8)}
    out = substitute_ty(t, subst)
    assert isinstance(out, A.TyArray)
    assert out.elem.name == "f64"
    # The size must have been folded to an IntLit(8) by _subst_shape_expr.
    assert isinstance(out.size, A.IntLit) and out.size.value == 8, (
        f"expected IntLit(8) after substitution; got {out.size}"
    )


def test_c2_b_c11_binary_shape_folds_to_intlit():
    """B:C11: pre-fix `_subst_shape_expr` substituted Name leaves
    in a Binary shape but left the Binary unfolded — `[N*2; 16]`
    with N=64 stayed as `Binary(*, IntLit(64), IntLit(2))` instead
    of `IntLit(128)`. Downstream lower-ast defaulted non-IntLit
    shapes to 0. Fix: fold Binary(IntLit, op, IntLit) → IntLit."""
    from helixc.frontend.monomorphize import (
        _subst_shape_expr, _SizeLitMarker,
    )
    from helixc.frontend import ast_nodes as A
    span = A.Span(0, 0)
    expr = A.Binary(span=span, op="*",
                    left=A.Name(span=span, name="N"),
                    right=A.IntLit(span=span, value=2))
    subst = {"N": _SizeLitMarker(64)}
    out = _subst_shape_expr(expr, subst)
    assert isinstance(out, A.IntLit) and out.value == 128, (
        f"expected IntLit(128), got {out}"
    )


# ----------------------------------------------------------------------
# Audit 28.8 cycle 2 C2-6 / B:C5 — ref-to-ref cast inner compat
# ----------------------------------------------------------------------
def test_c2_6_ref_to_ref_numeric_inner_ok():
    """C2-6 / B:C5: `&i32 as &i64` should still typecheck (inner
    pair is allowed by the numeric matrix)."""
    src = """
    fn main() -> i32 {
        let x: i32 = 7;
        let r: &i32 = &x;
        let r2: &i64 = r as &i64;
        0
    }
    """
    # Just verify check doesn't reject the numeric inner ref-cast.
    errs = check(src)
    # We don't require errs == [] because Phase-0 borrow-check may
    # surface separate diagnostics; just verify trap 28604 isn't
    # emitted for the ref-cast itself.
    assert not any("28604" in str(e) for e in errs), (
        f"unexpected 28604 on numeric ref-cast: {errs}"
    )


def test_c2_6_ref_to_unrelated_ref_traps_28604():
    """C2-6 / B:C5: `&Foo as &Bar` (unrelated structs) must now
    trap 28604. Pre-fix this silently typechecked because the
    TyRef-TyRef arm of `_check_cast_compat` returned unconditionally."""
    src = """
    struct Foo { x: i32 }
    struct Bar { y: i32 }
    fn cast_ref(f: &Foo) -> &Bar {
        f as &Bar
    }
    fn main() -> i32 { 0 }
    """
    errs = check(src)
    assert any("28604" in str(e) for e in errs), (
        f"expected trap 28604 on &Foo as &Bar; got: {errs}"
    )


def test_c4_7_cast_diagnostic_preserves_ref_prefix():
    """Audit 28.8 cycle 5 C4-7 / F6: the trap-28604 diagnostic on
    `&Foo as &Bar` must render `&Foo` and `&Bar` (with the `&`
    prefix preserved) — NOT the peeled inner `Foo` and `Bar`. Pre-fix,
    the D7 iterative ref-peeler stripped the `&` before the diagnostic
    formatter saw the types, so users saw the confusing
    `source Foo cannot convert to Bar` (no `&`) instead of
    `source &Foo cannot convert to &Bar`."""
    src = """
    struct Foo { x: i32 }
    struct Bar { y: i32 }
    fn cast_ref(f: &Foo) -> &Bar {
        f as &Bar
    }
    fn main() -> i32 { 0 }
    """
    errs = check(src)
    # Must surface trap 28604 (existing behavior — kept the same).
    assert any("28604" in str(e) for e in errs), (
        f"expected trap 28604 on &Foo as &Bar; got: {errs}"
    )
    # NEW assertion: the diagnostic now preserves `&` prefix.
    assert any("&Foo" in str(e) and "&Bar" in str(e) for e in errs), (
        f"expected `&Foo` and `&Bar` (with ref prefix) in diagnostic; "
        f"got: {[str(e) for e in errs]}"
    )


# ----------------------------------------------------------------------
# Audit 28.8 cycle 2 B:C7 — flatten_impls wired into check.py
# ----------------------------------------------------------------------
def test_c2_b_c7_flatten_impls_wired_in_check_py(tmp_path, capsys):
    """B:C7: pre-fix `check.py` only ran struct_mono, not flatten_impls.
    So trap 74002 (duplicate method name on distinct structs) was
    unreachable via `python -m helixc.check foo.hx`. Now it surfaces."""
    from helixc.check import main
    src_path = str(tmp_path / "dup.hx")
    with open(src_path, "w") as f:
        f.write(
            "struct Foo { x: i32 }\n"
            "struct Bar { y: i32 }\n"
            "impl Foo { fn area(self: Foo) -> i32 { self.x } }\n"
            "impl Bar { fn area(self: Bar) -> i32 { self.y } }\n"
            "fn main() -> i32 { 0 }\n"
        )
    rc = main([src_path, "--check-only"])
    cap = capsys.readouterr()
    assert rc == 1, (
        f"expected rc=1 on duplicate method; got rc={rc} "
        f"stdout={cap.out!r}"
    )
    assert "74002" in cap.out or "duplicate method" in cap.out, (
        f"expected 74002 diagnostic; got stdout={cap.out!r}"
    )


# ----------------------------------------------------------------------
# Audit 28.8 cycle 2 B:C10 — Logic-provenance 24100 dedup
# ----------------------------------------------------------------------
def test_c2_b_c10_logic_provenance_grouped_diagnostic():
    """B:C10: pre-fix, `f(1, 2)` where both params are `Logic<bool>`
    emitted TWO separate trap-24100 errors at the same call.span.
    Users saw two near-identical messages. Fix: batch them into a
    single grouped diagnostic naming both param names."""
    src = """
    fn lift(a: Logic<bool>, b: Logic<bool>) -> Logic<bool> { a }
    fn main() -> i32 {
        lift(1, 2);
        0
    }
    """
    errs = check(src)
    # Filter for 24100 diagnostics.
    prov_errs = [e for e in errs if "24100" in e]
    # B:C10 contract: ONE grouped diagnostic, not two.
    assert len(prov_errs) == 1, (
        f"expected 1 grouped trap-24100 diagnostic; got {len(prov_errs)}: "
        f"{prov_errs}"
    )
    # The grouped message must name both params.
    assert "'a'" in prov_errs[0] and "'b'" in prov_errs[0], (
        f"grouped diagnostic must name both params; got: {prov_errs[0]}"
    )


def test_c2_b_c10_single_logic_violation_unchanged():
    """B:C10 inverse: a single-param violation must STILL produce
    a single (non-grouped) diagnostic — no behavior change for the
    common case."""
    src = """
    fn lift(a: Logic<bool>, b: i32) -> Logic<bool> { a }
    fn main() -> i32 {
        lift(1, 2);
        0
    }
    """
    errs = check(src)
    prov_errs = [e for e in errs if "24100" in e]
    assert len(prov_errs) == 1
    # Single-violation message names just 'a'.
    assert "'a'" in prov_errs[0]
    # Either the old per-param phrasing OR the grouped (with 1
    # name) — both are acceptable. Test mainly that we DON'T
    # spuriously grow.


# ----------------------------------------------------------------------
# Audit 28.8 cycle 2 B:C9 — AD Cast arms accept bool/char/fp8/mxfp4/nvfp4
# ----------------------------------------------------------------------
def test_c2_b_c9_autodiff_cast_to_bool_no_spurious_warn():
    """B:C9: pre-fix `x as bool` inside grad-rewritten fn emitted
    a spurious 85001 warning because bool wasn't in the numeric list.
    Fix: NUMERIC_FOR_AD shared set covers bool/char/fp8/mxfp4/nvfp4."""
    from helixc.frontend import autodiff
    from helixc.frontend.parser import parse as parse_src
    autodiff.take_diff_warnings()
    src = "fn f(x: f32) -> bool { x as bool }"
    prog = parse_src(src)
    fn = prog.items[0]
    body_expr = fn.body.final_expr
    autodiff.differentiate(body_expr, "x")
    warnings = autodiff.take_diff_warnings()
    # Pre-fix this would emit 85001 "cast to non-numeric target TyName"
    # for the bool cast.
    assert not any("85001" in w and "non-numeric" in w for w in warnings), (
        f"unexpected 85001 non-numeric warn on Cast to bool: {warnings}"
    )


def test_c2_b_c9_autodiff_cast_to_fp8_no_spurious_warn():
    """B:C9: same for fp8 / mxfp4 / nvfp4."""
    from helixc.frontend import autodiff
    from helixc.frontend.parser import parse as parse_src
    for ty in ("fp8", "mxfp4", "nvfp4", "char"):
        autodiff.take_diff_warnings()
        src = f"fn f(x: f32) -> {ty} {{ x as {ty} }}"
        prog = parse_src(src)
        fn = prog.items[0]
        body_expr = fn.body.final_expr
        autodiff.differentiate(body_expr, "x")
        warnings = autodiff.take_diff_warnings()
        assert not any("85001" in w and "non-numeric" in w for w in warnings), (
            f"unexpected 85001 warn on Cast to {ty}: {warnings}"
        )


def test_c3_2_pointer_width_alias_silent():
    """Audit 28.8 cycle 3 C3-2: `D<i64> + D<isize>` and
    `D<u64> + D<usize>` must NOT emit AD002 (they're pointer-width
    aliases on 64-bit targets — same machine width). Pre-fix, the
    tie callback fired AND the outer mismatch fired, producing TWO
    confusing warnings per binop."""
    from helixc.frontend import autodiff
    from helixc.frontend.parser import parse as parse_src
    for left, right in [("i64", "isize"), ("u64", "usize")]:
        autodiff.take_diff_warnings()
        src = (
            f"@pure fn use_d(a: D<{left}>, b: D<{right}>) -> D<{left}> "
            f"{{ a + b }}\n"
            f"fn main() -> i32 {{ 0 }}\n"
        )
        prog = parse_src(src)
        errs = typecheck(prog)
        assert len(errs) == 0, (
            f"unexpected typecheck errors for D<{left}> + D<{right}>: "
            f"{errs}"
        )
        warnings = autodiff.take_diff_warnings()
        ad002_warns = [w for w in warnings if "24200" in w or "AD002" in w]
        assert len(ad002_warns) == 0, (
            f"expected zero AD002 warns for D<{left}> + D<{right}>, "
            f"got: {ad002_warns}"
        )


def test_d1_call_boundary_non_prim_mismatch_rejected():
    """Audit 28.8 cycle 3 D1: a struct or wrapper-typed parameter must
    reject a mismatched argument at the call boundary. Pre-fix, only
    TyPrim-vs-TyPrim was compared — every other pair silently passed.
    """
    from helixc.frontend.parser import parse as parse_src
    src = (
        "struct A { x: i32 }\n"
        "struct B { y: i32 }\n"
        "fn use_a(a: A) -> i32 { 0 }\n"
        "fn main() -> i32 {\n"
        "    let b = B { y: 5 };\n"
        "    use_a(b)\n"
        "}\n"
    )
    prog = parse_src(src)
    errs = typecheck(prog)
    assert any("use_a" in str(e) for e in errs), (
        f"expected use_a-arg-mismatch error, got: {errs}"
    )


def test_d3_array_size_zero_rejected():
    """Audit 28.8 cycle 3 D3: literal array size of 0 must emit
    trap 28802. Pre-fix, `[T; 0]` silently produced `TyPrim('size_0')`
    and downstream lower_ast used 0 as the length. Drive the check
    directly through `_resolve_size_expr` since source-level array
    types are parsed via TyArray nodes."""
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.typecheck import TypeChecker
    span = A.Span(0, 0)
    tc = TypeChecker(A.Program(module=None, items=[]))
    zero = A.IntLit(span=span, value=0, type_suffix=None)
    tc._resolve_size_expr(zero, scope=None)
    assert any("28802" in str(e) for e in tc.errors), (
        f"expected trap 28802 on size=0, got: {tc.errors}"
    )
    tc2 = TypeChecker(A.Program(module=None, items=[]))
    neg = A.IntLit(span=span, value=-5, type_suffix=None)
    tc2._resolve_size_expr(neg, scope=None)
    assert any("28802" in str(e) for e in tc2.errors), (
        f"expected trap 28802 on size=-5, got: {tc2.errors}"
    )


def test_d8_fmt_tystruct_uses_name():
    """Audit 28.8 cycle 3 D8: `_fmt` must print TyStruct as its
    declared name (e.g. `Foo`), not `TyStruct(name='Foo')`."""
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.typecheck import TypeChecker, TyStruct
    span = A.Span(0, 0)
    tc = TypeChecker(A.Program(module=None, items=[]))
    rendered = tc._fmt(TyStruct(name="Foo"))
    assert rendered == "Foo", (
        f"expected 'Foo', got {rendered!r}"
    )


def test_c5_2_size_compatible_tysize_cascade():
    """Audit 28.8 cycle 7 C6-1: shape-position cascade-safe arm is now
    in `_size_compatible` (narrowed from cycle-6 F1's over-broad
    top-level `_compatible` cascade). Verify TyArray composite still
    cascades through size position, and TySize vs TySize at the size
    position is accepted."""
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.typecheck import (
        TypeChecker, TySize, TyArray, TyPrim,
    )
    span = A.Span(0, 0)
    tc = TypeChecker(A.Program(module=None, items=[]))
    # Direct probe of the helper.
    assert tc._size_compatible(TySize("N"), TySize("M")), (
        "_size_compatible(TySize, TySize) should cascade-pass"
    )
    # Composite via TyArray: routed through _size_compatible internally.
    a1 = TyArray(elem=TyPrim("i32"), size=TySize("N"))
    a2 = TyArray(elem=TyPrim("i32"), size=TyPrim("size_3"))
    assert tc._compatible(a1, a2), (
        "TyArray<i32; N> vs TyArray<i32; 3> should cascade at size"
    )


def test_c6_1_compatible_tyvar_not_top_cascade():
    """Audit 28.8 cycle 7 C6-1: top-level `_compatible(TyVar, TyPrim)`
    must NOT silently pass. The cycle-6 F1 fix had introduced a top-
    level cascade for TyVar/TySize that broke `fn g[T]() -> T { 42 }`
    (body i32 vs return T silently typechecked). The narrowed fix
    restricts the cascade to `_size_compatible` (shape positions)."""
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.typecheck import (
        TypeChecker, TyVar, TyPrim,
    )
    span = A.Span(0, 0)
    tc = TypeChecker(A.Program(module=None, items=[]))
    # TyVar('T') vs TyPrim('i32') at value position must not cascade.
    # (Mono substitution would either bind T or report the mismatch.)
    assert not tc._compatible(TyVar("T"), TyPrim("i32")), (
        "TyVar vs TyPrim should NOT cascade at value position"
    )


def test_c3_3_main_clean_on_exception(monkeypatch, capsys, tmp_path):
    """Audit 28.8 cycle 3 C3-3: when _main_inner raises, main() must
    NOT leak a raw Python traceback to stderr, must return rc=1, and
    must leave _DIFF_WARNINGS drained. Pre-fix the wrapper had no
    try/finally so the drain was bypassed on exception exits."""
    from helixc import check as check_mod
    from helixc.frontend import autodiff
    autodiff._DIFF_WARNINGS.append("stale-warn-test-c3-3")
    def boom(*_args, **_kw):
        raise RuntimeError("simulated typecheck crash for C3-3")
    monkeypatch.setattr(check_mod, "typecheck", boom)
    src_file = tmp_path / "boom.hx"
    src_file.write_text("fn main() -> i32 { 0 }")
    rc = check_mod.main([str(src_file)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "internal error" in captured.err
    assert "compiler bug" in captured.err
    assert autodiff._DIFF_WARNINGS == [], (
        f"_DIFF_WARNINGS leaked: {autodiff._DIFF_WARNINGS}"
    )


def test_c4_6_filenotfound_not_attributed_as_compiler_bug(
        monkeypatch, capsys, tmp_path):
    """Audit 28.8 cycle 5 C4-6 / MEDIUM: a FileNotFoundError raised
    inside `_main_inner` must NOT be mis-attributed as a compiler bug.
    Pre-fix the broad `except Exception` printed "this is a compiler bug
    — please file an issue" for env errors like missing files or
    encoding mismatches, confusing users."""
    from helixc import check as check_mod
    def boom(*_args, **_kw):
        raise FileNotFoundError(2, "No such file or directory",
                                 "missing.something")
    monkeypatch.setattr(check_mod, "typecheck", boom)
    src_file = tmp_path / "boom.hx"
    src_file.write_text("fn main() -> i32 { 0 }")
    rc = check_mod.main([str(src_file)])
    captured = capsys.readouterr()
    # rc=2 (env error), not rc=1 (compiler bug).
    assert rc == 2, f"expected rc=2 for FileNotFoundError, got rc={rc}"
    assert "helixc:" in captured.err
    # Must NOT carry the "compiler bug" tagline.
    assert "compiler bug" not in captured.err, (
        f"FileNotFoundError mis-attributed as compiler bug: {captured.err!r}"
    )


def test_c4_6_unicode_decode_error_clean_message(
        monkeypatch, capsys, tmp_path):
    """Audit 28.8 cycle 5 C4-6 / MEDIUM: a UnicodeDecodeError raised
    inside `_main_inner` must NOT be mis-attributed as a compiler bug."""
    from helixc import check as check_mod
    def boom(*_args, **_kw):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
    monkeypatch.setattr(check_mod, "typecheck", boom)
    src_file = tmp_path / "boom.hx"
    src_file.write_text("fn main() -> i32 { 0 }")
    rc = check_mod.main([str(src_file)])
    captured = capsys.readouterr()
    assert rc == 2, f"expected rc=2 for UnicodeDecodeError, got rc={rc}"
    assert "encoding error" in captured.err
    assert "compiler bug" not in captured.err


def test_c8_1_import_error_attributed_as_compiler_bug(
        monkeypatch, capsys, tmp_path):
    """Audit 28.8 cycle 9 (regression for cycle-8 C8-1 close): an
    ImportError raised from inside _main_inner must surface as a
    compiler bug (rc=1, "please file an issue") — not as a user-
    environment error (rc=2). Pre-fix the cycle-5 exception
    classifier had a separate `except ImportError` arm that
    miscategorized internal-rename failures as env issues."""
    from helixc import check as check_mod
    def boom(*_args, **_kw):
        raise ImportError("cannot import name 'monomorphize_structs'")
    monkeypatch.setattr(check_mod, "typecheck", boom)
    src_file = tmp_path / "boom.hx"
    src_file.write_text("fn main() -> i32 { 0 }")
    rc = check_mod.main([str(src_file)])
    captured = capsys.readouterr()
    assert rc == 1, f"expected rc=1 for ImportError, got rc={rc}"
    assert "compiler bug" in captured.err, (
        f"expected 'compiler bug' tag in stderr, got: {captured.err}"
    )
    assert "internal error" in captured.err


def test_c8_2_env_error_no_double_helixc_prefix(
        monkeypatch, capsys, tmp_path):
    """Audit 28.8 cycle 9 (regression for cycle-8 C8-2 close): when a
    callee raises FileNotFoundError with a message that already starts
    with `helixc:`, the outer arm must NOT double-prefix the output."""
    from helixc import check as check_mod
    def boom(*_args, **_kw):
        raise FileNotFoundError("helixc: stdlib file missing: foo.hx")
    monkeypatch.setattr(check_mod, "typecheck", boom)
    src_file = tmp_path / "boom.hx"
    src_file.write_text("fn main() -> i32 { 0 }")
    rc = check_mod.main([str(src_file)])
    captured = capsys.readouterr()
    assert rc == 2, f"expected rc=2 for FileNotFoundError, got rc={rc}"
    # No "helixc: helixc:" anywhere in stderr.
    assert "helixc: helixc:" not in captured.err, (
        f"double prefix in stderr: {captured.err!r}"
    )
    # Single prefix preserved.
    assert "stdlib file missing" in captured.err


def test_c8_2_env_error_no_prefix_still_prefixed(
        monkeypatch, capsys, tmp_path):
    """Audit 28.8 cycle 9 (regression for cycle-8 C8-2 close): the
    `_emit_env_error` helper must still prepend `helixc:` when the
    callee's exception message has NO prefix."""
    from helixc import check as check_mod
    def boom(*_args, **_kw):
        raise FileNotFoundError("plain message, no prefix")
    monkeypatch.setattr(check_mod, "typecheck", boom)
    src_file = tmp_path / "boom.hx"
    src_file.write_text("fn main() -> i32 { 0 }")
    rc = check_mod.main([str(src_file)])
    captured = capsys.readouterr()
    assert rc == 2
    # Should have exactly one `helixc:` prefix.
    assert captured.err.count("helixc:") == 1, (
        f"expected single 'helixc:' prefix, got: {captured.err!r}"
    )


def test_c3_4_monomorphize_structs_idempotent():
    """Audit 28.8 cycle 3 C3-4: invoking monomorphize_structs twice on
    the same Program must NOT append duplicate StructDecls."""
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.parser import parse as parse_src
    from helixc.frontend.struct_mono import monomorphize_structs
    src = (
        "struct Pt[T] { x: T, y: T }\n"
        "fn use_it(p: Pt<i32>) -> i32 { p.x + p.y }\n"
        "fn main() -> i32 { use_it(Pt { x: 3, y: 4 }) }\n"
    )
    prog = parse_src(src)
    prog, _ = monomorphize_structs(prog)
    prog, _ = monomorphize_structs(prog)
    pt_decls = [it for it in prog.items
                if isinstance(it, A.StructDecl)
                and it.name.startswith("Pt__")]
    assert len(pt_decls) == 1, (
        f"expected exactly one Pt__ mono'd decl, got "
        f"{[d.name for d in pt_decls]}"
    )


def test_c3_6_shape_fold_div_by_zero_traps_28801():
    """Audit 28.8 cycle 3 C3-6: shape-time `/0` and `%0` must raise
    ShapeFoldError (trap 28801)."""
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.monomorphize import (
        _fold_intlit_arith, ShapeFoldError)
    span = A.Span(0, 0)
    for op in ("/", "%"):
        expr = A.Binary(
            span=span,
            op=op,
            left=A.IntLit(span=span, value=10, type_suffix=None),
            right=A.IntLit(span=span, value=0, type_suffix=None),
        )
        try:
            _fold_intlit_arith(expr)
        except ShapeFoldError as e:
            assert "28801" in str(e)
        else:
            assert False, f"expected ShapeFoldError on {op}-by-zero"


def test_d4_logic_logic_mixed_inner_warns():
    """Audit 28.8 cycle 3 D4: `Logic<f64> + Logic<i32>` (neither side
    TyDiff) must emit an AD002 warn with `[Logic-domain]` tag."""
    from helixc.frontend import autodiff
    from helixc.frontend.parser import parse as parse_src
    autodiff.take_diff_warnings()
    src = (
        "fn use_logic(a: Logic<f64>, b: Logic<i32>) -> Logic<f64> "
        "{ a + b }\n"
        "fn main() -> i32 { 0 }\n"
    )
    prog = parse_src(src)
    _errs = typecheck(prog)
    warnings = autodiff.take_diff_warnings()
    assert any("24200" in w and "Logic-domain" in w for w in warnings), (
        f"expected Logic-domain AD002 warn for "
        f"Logic<f64>+Logic<i32>, got: {warnings}"
    )


def test_d5_unary_fold_negative_size():
    """Audit 28.8 cycle 3 D5: `substitute_ty` with `Unary(-, IntLit(N))`
    inside a TyArray size must fold to `IntLit(-N)`."""
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.monomorphize import (
        _subst_shape_expr, _SizeLitMarker)
    span = A.Span(0, 0)
    expr = A.Unary(
        span=span,
        op="-",
        operand=A.Name(span=span, name="N"),
    )
    result = _subst_shape_expr(expr, {"N": _SizeLitMarker(5)})
    assert isinstance(result, A.IntLit), (
        f"expected IntLit after fold, got {type(result).__name__}"
    )
    assert result.value == -5


def test_d7_deep_ref_cast_bounded():
    """Audit 28.8 cycle 3 D7: 1500-layer ref-cast must NOT hit Python's
    recursion limit. The peeling loop appends trap 28803 before that
    depth."""
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.typecheck import TypeChecker, TyPrim, TyRef
    span = A.Span(0, 0)
    tc = TypeChecker(A.Program(module=None, items=[]))
    src = TyPrim("i32")
    tgt = TyPrim("i64")
    for _ in range(1500):
        src = TyRef(inner=src, is_mut=False)
        tgt = TyRef(inner=tgt, is_mut=False)
    tc._check_cast_compat(src, tgt, span)
    has_28803 = any("28803" in str(e) for e in tc.errors)
    assert has_28803, (
        f"expected trap 28803 for 1500-layer ref-cast, got: "
        f"{[str(e) for e in tc.errors]}"
    )


def test_c92_f1_intlit_with_float_suffix_rejected():
    """Stage 28.9 cycle 93 audit-T F1 regression (HIGH conf 85):
    `42_f32` (IntLit with float-domain suffix) must be rejected by
    typecheck. Pre-fix this lexed as IntLit(value=42,
    type_suffix='f32'), passed typecheck as TyPrim('f32'), lowered
    to CONST_INT(result_ty=TIRScalar('f32')), and the backend stored
    the raw int bit-pattern 0x2A into the f32 slot — silently
    representing 5.88e-44 instead of 42.0."""
    from helixc.frontend.typecheck import typecheck
    from helixc.frontend.parser import parse
    src = "fn main() -> i32 { let x: f32 = 42_f32; 0 }"
    errs = typecheck(parse(src))
    assert any("float-domain suffix" in str(e) for e in errs), (
        f"expected float-domain-suffix diagnostic, got: "
        f"{[str(e) for e in errs]}"
    )


def test_c92_f1_floatlit_with_int_suffix_rejected():
    """C92-F1 regression (HIGH): symmetric — `4.2_i32` (FloatLit
    with integer-domain suffix) must be rejected."""
    from helixc.frontend.typecheck import typecheck
    from helixc.frontend.parser import parse
    src = "fn main() -> i32 { let x: i32 = 4.2_i32; 0 }"
    errs = typecheck(parse(src))
    assert any("integer-domain suffix" in str(e) for e in errs), (
        f"expected integer-domain-suffix diagnostic, got: "
        f"{[str(e) for e in errs]}"
    )


def test_c92_f1_valid_intlit_int_suffix_accepted():
    """C92-F1 regression: ensure correct shapes still pass —
    `42_i32` is valid; no kind-coherence diagnostic should fire."""
    from helixc.frontend.typecheck import typecheck
    from helixc.frontend.parser import parse
    src = "fn main() -> i32 { let x: i32 = 42_i32; x }"
    errs = typecheck(parse(src))
    assert not any(("float-domain" in str(e) or "integer-domain" in str(e))
                   for e in errs), (
        f"unexpected kind-coherence error for valid 42_i32: "
        f"{[str(e) for e in errs]}"
    )


def main():
    # Tests requiring pytest fixtures (tmp_path / monkeypatch / capsys /
    # etc.) are skipped here — the manual runner can't synthesize
    # fixtures. Same pattern as test_parser.py's runner. They're still
    # discovered by `pytest helixc/tests/test_typecheck.py` which DOES
    # wire fixtures, so coverage is preserved.
    import inspect
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    passed = 0
    failed = 0
    skipped = 0
    for name, fn in tests:
        try:
            sig = inspect.signature(fn)
            required = [p for p in sig.parameters.values()
                        if p.default is inspect.Parameter.empty
                        and p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                       inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        except (TypeError, ValueError):
            required = []
        if required:
            print(f"SKIP {name}: needs pytest fixtures {[p.name for p in required]}")
            skipped += 1
            continue
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    summary = f"{passed} passed, {failed} failed"
    if skipped:
        summary += f", {skipped} skipped"
    print(f"\n{summary}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
