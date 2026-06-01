fn main() -> i32 { let x: i16 = 32767_i16 + 1_i16; let y: i32 = x as i32; if y < 0 { 42 } else { 7 } }
