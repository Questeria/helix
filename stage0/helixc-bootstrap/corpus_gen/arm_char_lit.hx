fn main() -> i32 {
    // char literals are int (byte) values in Helix. '*' = 42.
    let star: i32 = '*';        // 42
    let nl: i32 = '\n';         // 10  (escape)
    let zero: i32 = '0';        // 48
    // star + (nl - 10) + (zero - 48) = 42
    star + (nl - '\n') + (zero - '0')
}
