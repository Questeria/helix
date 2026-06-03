fn main() -> i32 {
    /* a block comment /* with a NESTED block comment inside it
       spanning multiple lines */ and text after the inner close */
    let a: i32 = 40; /* trailing block comment */ let b: i32 = 2;
    /* /* /* triple-nested */ */ */
    a + b   // -> 42 ; the comments must be skipped, leaving `a + b`
}
