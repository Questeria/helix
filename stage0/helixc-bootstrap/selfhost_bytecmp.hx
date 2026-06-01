// selfhost_bytecmp.hx -- Helix-native byte-identity checker for the self-host
// fixpoint (Helix v1.0 DoD #1 + the de-language direction of #6).
//
// The load-bearing assertion of the self-host fixpoint -- "are two compiler
// generations byte-identical?" -- done by a HELIX program compiled by the
// raw-binary seed (seed.bin), NOT by bash `cmp`. It reads /tmp/cmp_a and
// /tmp/cmp_b into the arena (read_file_to_arena puts one byte per arena slot)
// and returns 0 iff both are non-empty and byte-identical, else 1.
//
// Only seed-supported builtins are used (read_file_to_arena, __arena_len,
// __arena_get) -- run_process/set_exec are kovc.hx-only, so the Python-free
// orchestration (running the generations) stays in the runner for now; this
// program is the equality CHECK. Compile: ./seed.bin selfhost_bytecmp.hx out.bin
// Exit 0 = identical (fixpoint holds), 1 = differ/empty.

fn bytes_equal(ba: i32, na: i32, bb: i32, nb: i32) -> i32 {
    let mut diff = if na == 0 { 1 } else { 0 };
    diff = if na != nb { 1 } else { diff };
    let lim = if nb < na { nb } else { na };
    let mut i = 0;
    while i < lim {
        diff = if __arena_get(ba + i) != __arena_get(bb + i) { 1 } else { diff };
        i = i + 1;
    }
    diff
}

fn main() -> i32 {
    let ba = __arena_len();
    let na = read_file_to_arena("/tmp/cmp_a");
    let bb = __arena_len();
    let nb = read_file_to_arena("/tmp/cmp_b");
    bytes_equal(ba, na, bb, nb)
}
