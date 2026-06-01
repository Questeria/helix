fn main() -> i32 { let x: u16 = 0_u16 - 1_u16; let y: i32 = x as i32; if y == 65535 { 42 } else { 7 } }
