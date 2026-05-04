// hello_world.hx — first Helix program that produces visible output.
// Demonstrates: print_str builtin (sys_write to stdout), write_file
// builtin (sys_open + sys_write + sys_close).

fn main() -> i32 {
    print_str("Hello from Helix!\n");
    print_str("This program emits a real Linux ELF, prints to stdout via\n");
    print_str("the write(1, ...) syscall, and writes a file via\n");
    print_str("open + write + close.\n");

    let r = write_file("/tmp/helix_hello.txt", "wrote from helix\n");
    if r == 0 {
        print_str("file write succeeded\n");
        42
    } else {
        print_str("file write FAILED\n");
        1
    }
}
