fn main() -> i32 {
    // hex / binary / octal integer literals, each WITH `_` digit separators.
    let h: i32 = 0x2A;          // 42
    let h2: i32 = 0xFF_FF;      // 65535
    let b: i32 = 0b10_1010;     // 42
    let o: i32 = 0o52;          // 42
    // h + (h2 - 65535) + (b - 42) + (o - 42) = 42 + 0 + 0 + 0 = 42
    h + (h2 - 0xFF_FF) + (b - 0b10_1010) + (o - 0o52)
}
