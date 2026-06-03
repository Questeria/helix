// M-5 (charter HELIX_COMPLETION.md §1.6 MED) -- DOCUMENT-AS-BOUND + negative test.
// BOUND: a BARE (no-turbofish) call to a free generic fn at a NON-i32 scalar
// type does NOT infer the type argument -- the monomorphizer defaults the
// generic param to i32, so an f32 value passed through a generic identity is
// reinterpreted as i32 and the f32 bit pattern is lost. The supported idiom
// for non-i32 scalar generics is the EXPLICIT TURBOFISH form `f::<f32>(...)`
// (verified working: id::<f32>(3.0_f32) -> 3, add2::<f32>(2.0,3.0) -> 5), or
// impl-block monomorphization (Box::<f32>). See spec §3 / §1.6 M-5.
//
// This is a NEGATIVE / bound-proving row: it asserts the bare form does NOT
// produce the f32 value 3 -- it produces 0 (the i32-default reinterpretation).
// If a future change adds bare-call type inference, `bare` below becomes 3 and
// this row FAILS, signalling the documented bound has changed (intended).
//
// `id(3.0_f32)` is the bare form (no `::<f32>`). Exit 0 = bound holds.
fn id[T](x: T) -> T { x }
fn main() -> i32 {
    let r: f32 = id(3.0_f32);   // BARE generic at f32 -> T defaults to i32 -> 0
    r as i32                     // documented bound: yields 0, NOT 3
}
