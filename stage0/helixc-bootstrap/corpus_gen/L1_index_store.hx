// L-1 (charter HELIX_COMPLETION.md §1.6 LOW): index-STORE hardening, `arr[i] = e`.
// Promotes the `emit_index_store_cpu` arm (kovc.hx:6896, AST_INDEX_STORE tag 55)
// from [impl] to [proven] with a corpus row. The existing arr_idx.hx only READS
// (a[1] -> 20); this program WRITES through a mutable array at RUNTIME-computed
// indices/values, then reads the slots back, so the store path provably runs.
//
// a starts [0,0,0,0]. Loop-driven (so neither index nor value can be const-folded
// to a static slot/immediate): on iteration i, write a[i] = (i+1)*7 + 1.
//   a[0]=8, a[1]=15, a[2]=22, a[3]=29.
// Then mutate one slot again to prove a second store to the same index overwrites:
//   a[1] = a[0] - 3  -> 5.
// Sum the slots read back: 8 + 5 + 22 + 29 = 64; subtract 22 -> 42.
fn main() -> i32 {
    let mut a = [0, 0, 0, 0];
    let mut i = 0;
    while i < 4 {
        a[i] = (i + 1) * 7 + 1;   // index-store: runtime index, runtime value
        i = i + 1;
    }
    a[1] = a[0] - 3;              // overwrite an existing slot (8 - 3 = 5)
    a[0] + a[1] + a[2] + a[3] - 22   // 8 + 5 + 22 + 29 - 22 = 42
}
